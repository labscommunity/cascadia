//! Qwen3.6-35B-A3B staged engine: runs the IR-surgery shard chain
//! (tools/qwen36_surgery/export_qwen36_moe.py) in-process on one box
//! (single-box, --total 1) or as an N-node stage pipeline (--rank R
//! --total N). Greedy-only, batch=1; the decode loop is a port of
//! `tools/qwen36_surgery/proto_m3_decode.py`, which measured 64/64
//! greedy token parity vs the whole model.
//!
//! Stage dirs are stateful OV IRs (DeltaNet conv/ssm + attention KV as
//! ReadValue/Assign). Linear state cannot be trimmed: every task starts
//! with `reset_state()` on every stage (position-0 re-entry is the only
//! recovery — docs/architectures/qwen36-moe-support.md §4.1);
//! cancellation drops the task and the next admission's reset restores
//! the invariant.
//!
//! Pipeline mode (docs/architectures/qwen36-moe-support.md): rank 0 holds embeddings
//! + stage 0 + tokenizer and drives decode; middle ranks relay (run
//! their stage, pass the span downstream, return the token back); the
//! last rank holds the logits head and answers each FORWARD with the
//! argmax token. Control frames (HELLO, RESET) chain through middles —
//! one ACK at rank 0 means the whole chain agreed. Frames are lockstep
//! on one transport session per hop: 12-byte BE header
//! [kind][epoch][pos], then a kind-specific body. The stateful `seq>1`
//! reset heuristic is NOT used — chunked prefill would re-trigger it
//! mid-task; downstream ranks reset only on RESET frames (position-0 =
//! new task).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};
use cascadia_transport::{
    ActivationClient, ActivationServer, DType as WireDType, Tensor as WireTensor, TransportError,
};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;
use tokenizers::Tokenizer;
use tracing::{info, warn};

const HIDDEN: usize = 2048;
/// Prefill span per chain pass; bounds the transient [1, T, vocab]
/// logits buffer (~254 MB f32 at 256 with the 248320 vocab) and the
/// per-chunk wire frame (256 * 2048 * 4 B = 2 MiB, day-0 probe sized).
const PREFILL_CHUNK: usize = 256;
const MROPE_ROWS: usize = 4;

// Pipeline wire protocol: header [kind u32][epoch u32][pos u32] (BE), then
// HELLO/HELLO_NAK: [len u32][json], FORWARD: one WireTensor f32
// [1,n,HIDDEN], TOKEN: [i32 BE]. ACKs are header-only. An earlier design
// draft framed the position as an i64 prefix tensor; the shipped header
// carries the same information in the day-0 probe's framing
// (kind+epoch+pos), which the probe validated end-to-end on the live
// relay.
const FRAME_HELLO: u32 = 1;
const FRAME_HELLO_ACK: u32 = 2;
const FRAME_HELLO_NAK: u32 = 3;
const FRAME_RESET: u32 = 4;
const FRAME_RESET_ACK: u32 = 5;
const FRAME_FORWARD: u32 = 6;
const FRAME_TOKEN: u32 = 7;
/// Issue-34 §8: head broadcasts CAPTURE after a turn so every rank snapshots its KV under one
/// content epoch (carried in the body, not the u32 task-epoch header). Append-only frame codes.
#[cfg(feature = "kv_coord")]
const FRAME_CAPTURE: u32 = 8;
#[cfg(feature = "kv_coord")]
const FRAME_CAPTURE_ACK: u32 = 9;
/// Handshake schema version (spec §3.4).
const PROTO_VERSION: u32 = 1;

fn frame_header(kind: u32, epoch: u32, pos: u32) -> [u8; 12] {
    let mut h = [0u8; 12];
    h[0..4].copy_from_slice(&kind.to_be_bytes());
    h[4..8].copy_from_slice(&epoch.to_be_bytes());
    h[8..12].copy_from_slice(&pos.to_be_bytes());
    h
}

fn parse_header(b: &[u8]) -> (u32, u32, u32) {
    let f = |i: usize| u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    (f(0), f(4), f(8))
}

#[derive(serde::Deserialize)]
struct Manifest {
    arch: String,
    /// Exporter sliced the last stage's logits to the final position
    /// ([1,1,vocab]); the engine then skips its own row slicing. Absent
    /// in pre-slice shard trees (default false).
    #[serde(default)]
    last_logits_only: bool,
    stages: Vec<StageInfo>,
}

#[derive(serde::Deserialize)]
struct StageInfo {
    stage: usize,
    layer_start: u32,
    layer_end: u32,
}

pub struct Qwen36Builder {
    pub shards_dir: String,
    pub device: String,
    pub max_tokens_default: u32,
    rank: u32,
    total: u32,
    listen_host: String,
    listen_port: Option<u16>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    runtime_handle: Option<tokio::runtime::Handle>,
    manifest_json: Option<String>,
    emb: Option<Runtime>,
    stages: Option<Vec<Runtime>>,
    tokenizer: Option<Tokenizer>,
    eos: Option<u32>,
    last_logits_only: bool,
}

impl Qwen36Builder {
    pub fn new(shards_dir: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            shards_dir: shards_dir.into(),
            device: device.into(),
            max_tokens_default: 256,
            rank: 0,
            total: 1,
            listen_host: "0.0.0.0".to_string(),
            listen_port: None,
            upstream: None,
            downstream: None,
            runtime_handle: None,
            manifest_json: None,
            emb: None,
            stages: None,
            tokenizer: None,
            eos: None,
            last_logits_only: false,
        }
    }

    pub fn with_rank(mut self, rank: u32, total: u32) -> Self {
        self.rank = rank;
        self.total = total;
        self
    }
}

fn read_eos(dir: &Path) -> Option<u32> {
    let gc = std::fs::read_to_string(dir.join("generation_config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&gc).ok()?;
    match &v["eos_token_id"] {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as u32),
        serde_json::Value::Array(a) => a.first()?.as_u64().map(|x| x as u32),
        _ => None,
    }
}

#[async_trait]
impl Builder for Qwen36Builder {
    fn configure_listen(&mut self, host: &str, port: u16) {
        self.listen_host = host.to_string();
        self.listen_port = Some(port);
    }

    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        if self.total <= 1 {
            if peers.upstream.is_some() || peers.downstream.is_some() {
                return Err(EngineError::PeerRejected(
                    "qwen36-moe --total 1 runs all stages in-process; \
                     do not configure peers"
                        .into(),
                ));
            }
            return Ok(());
        }
        self.runtime_handle = Some(tokio::runtime::Handle::current());
        // Bind the upstream listener before dialing downstream so peers
        // can connect to us first (ov-runtime order).
        if peers.upstream.is_some() {
            let port = self
                .listen_port
                .ok_or_else(|| EngineError::PeerRejected("configure_listen() required".into()))?;
            let mut server = ActivationServer::new(self.listen_host.clone(), port);
            server
                .start()
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            self.upstream = Some(Arc::new(tokio::sync::Mutex::new(server)));
        }
        if let Some(downstream) = peers.downstream {
            let mut client = ActivationClient::new(downstream.host, downstream.port);
            client
                .connect_with_timeout(Duration::from_secs(60))
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            self.downstream = Some(Arc::new(tokio::sync::Mutex::new(client)));
        }
        if let Some(srv) = &self.upstream {
            srv.lock()
                .await
                .accept()
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    async fn load(&mut self, shard: ShardSpec) -> EngineResult<LoadStream> {
        let pipeline = self.total > 1;
        if !pipeline && !(shard.is_first_stage && shard.is_last_stage) {
            return Err(EngineError::ShardRejected(
                "qwen36-moe --total 1 requires a single in-process stage chain".into(),
            ));
        }
        let dir = PathBuf::from(&self.shards_dir);
        let manifest_path = dir.join("manifest.json");
        let manifest_raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| EngineError::ModelNotFound(format!("{}: {e}", manifest_path.display())))?;
        let manifest: Manifest = serde_json::from_str(&manifest_raw)
            .map_err(|e| EngineError::InvalidConfig(format!("manifest.json: {e}")))?;
        if manifest.arch != "qwen3_5_moe" {
            return Err(EngineError::InvalidConfig(format!(
                "manifest arch {:?} is not qwen3_5_moe",
                manifest.arch
            )));
        }
        if pipeline && manifest.stages.len() != self.total as usize {
            return Err(EngineError::ShardRejected(format!(
                "--total ({}) does not match manifest stage count ({})",
                self.total,
                manifest.stages.len()
            )));
        }

        let mut progress = vec![LoadProgress::message(format!(
            "qwen36-moe: {} stages from {} (rank {}/{})",
            manifest.stages.len(),
            dir.display(),
            self.rank,
            self.total
        ))];
        // NOTE: the router runs f16. Raising precision to stabilize near-tie
        // expert selection is NOT possible on the Intel Arc GPU — the fused MoE
        // gemm (`MOE3GemmFusedCompressed`) has an f16-only kernel, so both
        // INFERENCE_PRECISION_HINT=f32 and EXECUTION_MODE_HINT=ACCURACY fail at
        // compile ("No layout format available ... data_type: f32"). The
        // multi-stage long-generation drift is therefore mitigated by config
        // (single-box / 2-stage, aligned GPU drivers), not a precision hint.
        let plugin = PluginConfig::new();

        // Embeddings + tokenizer + eos live with the decode driver only.
        if self.rank == 0 {
            let emb_xml = dir.join("openvino_text_embeddings_model.xml");
            progress.push(LoadProgress::message(
                "compiling text-embeddings IR".to_string(),
            ));
            self.emb = Some(
                Runtime::compile(emb_xml.to_str().unwrap_or_default(), &self.device, &plugin)
                    .map_err(map_ov)?,
            );
            self.tokenizer = Some(
                Tokenizer::from_file(dir.join("tokenizer.json"))
                    .map_err(|e| EngineError::InvalidConfig(format!("tokenizer.json: {e}")))?,
            );
            self.eos = read_eos(&dir);
        }

        let mut stages = Vec::new();
        for s in &manifest.stages {
            if pipeline && s.stage != self.rank as usize {
                continue;
            }
            let xml = dir.join(format!("stage{}", s.stage)).join("stage.xml");
            progress.push(LoadProgress::message(format!(
                "compiling stage{} (layers {}..{}) on {}",
                s.stage, s.layer_start, s.layer_end, self.device
            )));
            stages.push(
                Runtime::compile(xml.to_str().unwrap_or_default(), &self.device, &plugin)
                    .map_err(map_ov)?,
            );
        }

        self.manifest_json = Some(manifest_raw);
        self.last_logits_only = manifest.last_logits_only;
        self.stages = Some(stages);
        progress.push(LoadProgress::ready());
        Ok(Box::pin(stream::iter(progress)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        if self.rank == 0 && (self.emb.is_none() || self.tokenizer.is_none()) {
            return Err(EngineError::NotLoaded);
        }
        Ok(Box::new(Qwen36Engine {
            emb: self.emb,
            stages: self.stages.ok_or(EngineError::NotLoaded)?,
            tokenizer: self.tokenizer,
            eos: self.eos,
            max_tokens_default: self.max_tokens_default,
            last_logits_only: self.last_logits_only,
            rank: self.rank,
            total: self.total,
            upstream: self.upstream,
            downstream: self.downstream,
            runtime_handle: self.runtime_handle,
            manifest_json: self.manifest_json.unwrap_or_default(),
            epoch: 0,
            peer_epoch: 0,
            handshake_done: false,
            poisoned: None,
            pending: Vec::new(),
            active: None,
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
        }))
    }
}

fn map_ov(err: cascadia_ov_genai_shim::Error) -> EngineError {
    match err {
        cascadia_ov_genai_shim::Error::Stub => {
            EngineError::Backend("qwen36-moe requires the `openvino` feature (stub build)".into())
        }
        cascadia_ov_genai_shim::Error::Utf8(s) => EngineError::InvalidConfig(s),
        cascadia_ov_genai_shim::Error::Native(s) => EngineError::Backend(s),
    }
}

fn map_wire(err: TransportError) -> EngineError {
    EngineError::Backend(format!("qwen36 pipeline wire: {err}"))
}

pub struct Qwen36Engine {
    /// Rank 0 only in pipeline mode.
    emb: Option<Runtime>,
    stages: Vec<Runtime>,
    /// Rank 0 only in pipeline mode.
    tokenizer: Option<Tokenizer>,
    eos: Option<u32>,
    max_tokens_default: u32,
    last_logits_only: bool,
    rank: u32,
    total: u32,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    runtime_handle: Option<tokio::runtime::Handle>,
    /// Raw manifest.json for the startup handshake (full-text compare —
    /// small file, stronger than a hash and needs no new dependency).
    manifest_json: String,
    /// Task epoch. Rank 0 bumps per admission; frames carry
    /// it; the downstream drops frames from older epochs.
    epoch: u32,
    /// Downstream side: epoch of the last RESET accepted.
    peer_epoch: u32,
    handshake_done: bool,
    /// Set when the startup handshake found a config mismatch (handshake
    /// rule: refuse to serve). Admissions fail loud with this reason.
    poisoned: Option<String>,
    pending: Vec<GenerationTask>,
    active: Option<ActiveTask>,
    /// Issue-34 Option C: opaque multi-stage KV blob cache for the coordination plane.
    #[cfg(feature = "kv_coord")]
    kv: crate::kv_coordination::OvKvCache,
}

/// In-flight task state. `step()` advances one token per call so the
/// runner can interleave cancel() between steps (a monolithic step()
/// holds the engine mutex for the whole generation, making cancel
/// unreachable) and so streaming emits at the engine's real cadence.
struct ActiveTask {
    task_id: TaskId,
    prompt_ids: Vec<u32>,
    prefill_idx: usize,
    step: usize,
    /// Single-box: last position's logits row. Unused in pipeline mode.
    logits: Vec<f32>,
    /// Pipeline rank 0: next token returned by the downstream argmax.
    next_token: Option<u32>,
    gen_ids: Vec<u32>,
    /// Byte length of the decoded prefix already emitted as chunks.
    emitted: usize,
    max_tokens: usize,
    started: Instant,
    /// Pipeline rank 0: per-frame FORWARD→TOKEN round-trip times for
    /// decode frames (n=1), for the pipeline gate-4 wire histogram.
    wire_ms: Vec<f64>,
}

fn le_bytes_i64(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_bytes_i32(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn le_bytes_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn f32_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_async<F: std::future::Future>(h: &tokio::runtime::Handle, fut: F) -> F::Output {
    crate::dist_spec::run_async_pub(h, fut)
}

/// Bound a driver-side reply exchange. The transport exempts a frame's FIRST
/// byte from the recv timeout so an idle relay can park on the next request
/// without killing its socket; a driver that just sent FORWARD/RESET/HELLO is
/// owed a reply now, so a black-holed downstream must not pin this thread
/// forever. Times out at the same configured activation timeout.
async fn reply_bounded<T, F>(fut: F) -> Result<T, TransportError>
where
    F: std::future::Future<Output = Result<T, TransportError>>,
{
    match tokio::time::timeout(cascadia_transport::recv_timeout(), fut).await {
        Ok(r) => r,
        Err(_) => Err(TransportError::SocketClosed),
    }
}

/// One inbound frame on the upstream session (downstream rank's view).
enum InFrame {
    Hello(Vec<u8>),
    Reset(u32),
    Forward {
        epoch: u32,
        pos: u32,
        hidden: Vec<f32>,
        n: usize,
    },
    /// Issue-34 §8 CAPTURE: snapshot local KV under the head's content `kv_epoch` (body-carried).
    #[cfg(feature = "kv_coord")]
    Capture {
        kv_epoch: u64,
        tokens: Vec<i32>,
    },
}

impl Qwen36Engine {
    /// Embed a token span: [1, n] ids -> flattened [1, n, HIDDEN] f32.
    fn embed_seq(&mut self, toks: &[u32]) -> EngineResult<Vec<f32>> {
        let emb = self
            .emb
            .as_mut()
            .ok_or_else(|| EngineError::Backend("no embeddings on this rank".into()))?;
        let name = emb
            .input_names()
            .map_err(map_ov)?
            .into_iter()
            .next()
            .ok_or_else(|| EngineError::Backend("embeddings IR has no inputs".into()))?;
        let ids: Vec<i64> = toks.iter().map(|&t| t as i64).collect();
        emb.set_input(&name, DType::I64, &[1, toks.len()], &le_bytes_i64(&ids))
            .map_err(map_ov)?;
        emb.infer().map_err(map_ov)?;
        let (dtype, _shape, bytes) = emb.output(0).map_err(map_ov)?;
        if !matches!(dtype, DType::F32) {
            return Err(EngineError::Backend(format!(
                "embeddings output dtype {dtype:?}, expected f32"
            )));
        }
        Ok(f32_from_le(&bytes))
    }

    /// One pass through the local stage chain covering absolute positions
    /// [t0, t1) (T = t1-t0; the stage IRs are dynamic in T). `hidden` is
    /// real embeddings on the global first stage, the upstream stage's
    /// output otherwise. Returns the last stage's full first output,
    /// flattened [1, T, width].
    fn chain_pass(&mut self, embeds: &[f32], t0: usize, t1: usize) -> EngineResult<Vec<f32>> {
        let n = t1 - t0;
        let mask = vec![1i64; t1];
        let pos: Vec<i64> = (0..MROPE_ROWS)
            .flat_map(|_| (t0 as i64)..(t1 as i64))
            .collect();
        let zeros_embeds = vec![0f32; n * HIDDEN];
        let first_global = self.rank == 0;
        let mut hidden: Vec<f32> = embeds.to_vec();
        for (j, st) in self.stages.iter_mut().enumerate() {
            let names = st.input_names().map_err(map_ov)?;
            for name in &names {
                match name.as_str() {
                    "stage_hidden" => st
                        .set_input(name, DType::F32, &[1, n, HIDDEN], &le_bytes_f32(&hidden))
                        .map_err(map_ov)?,
                    s if s.contains("embed") => {
                        // global first stage: real embeds; later stages:
                        // dummy (upstream ShapeOf chains read shapes only)
                        let data = if j == 0 && first_global {
                            &hidden
                        } else {
                            &zeros_embeds
                        };
                        st.set_input(name, DType::F32, &[1, n, HIDDEN], &le_bytes_f32(data))
                            .map_err(map_ov)?;
                    }
                    s if s.contains("attention_mask") => st
                        .set_input(name, DType::I64, &[1, t1], &le_bytes_i64(&mask))
                        .map_err(map_ov)?,
                    s if s.contains("position") => st
                        .set_input(name, DType::I64, &[MROPE_ROWS, 1, n], &le_bytes_i64(&pos))
                        .map_err(map_ov)?,
                    s if s.contains("beam") => st
                        .set_input(name, DType::I32, &[1], &le_bytes_i32(&[0]))
                        .map_err(map_ov)?,
                    other => {
                        return Err(EngineError::Backend(format!(
                            "stage{j}: unexpected input {other:?}"
                        )))
                    }
                }
            }
            st.infer().map_err(map_ov)?;
            let (dtype, _shape, bytes) = st.output(0).map_err(map_ov)?;
            if !matches!(dtype, DType::F32) {
                return Err(EngineError::Backend(format!(
                    "stage{j} output dtype {dtype:?}, expected f32"
                )));
            }
            hidden = f32_from_le(&bytes);
        }
        Ok(hidden)
    }

    /// Run a token span starting at absolute position `t0`; returns the
    /// LAST position's logits row. Single-box only (needs the full chain).
    fn run_span(&mut self, toks: &[u32], t0: usize) -> EngineResult<Vec<f32>> {
        let n = toks.len();
        let e = self.embed_seq(toks)?;
        let out = self.chain_pass(&e, t0, t0 + n)?;
        if self.last_logits_only {
            // Exporter already sliced the last stage to the final
            // position; the output IS the last row.
            return Ok(out);
        }
        let row = out.len() / n;
        Ok(out[(n - 1) * row..].to_vec())
    }

    fn reset_all(&mut self) {
        for st in self.stages.iter_mut() {
            if let Err(e) = st.reset_state() {
                warn!(error = %e, "qwen36: stage reset_state failed");
            }
        }
    }

    /// Finish the active task: reset state, log, emit the final marker.
    fn finalize(&mut self) -> Vec<(TaskId, Chunk)> {
        let Some(t) = self.active.take() else {
            return Vec::new();
        };
        // Issue-34: capture this rank's local KV under (prompt + generated) BEFORE reset_all wipes
        // it. Keyed by the full sequence (session-resume). Gated + best-effort (stub ⇒ no-op).
        #[cfg(feature = "kv_coord")]
        {
            let tokens: Vec<i32> = t
                .prompt_ids
                .iter()
                .chain(t.gen_ids.iter())
                .map(|&u| u as i32)
                .collect();
            // Pipeline head: broadcast CAPTURE so every downstream rank snapshots its slice under the
            // same content epoch (workers have no tokens). Best-effort. rank0's own slice is stashed
            // token-keyed by kv_capture_local below (the NEGOTIATE/offers path).
            if self.total > 1 && self.rank == 0 && self.downstream.is_some() && !tokens.is_empty() {
                let kv_epoch = crate::kv_coordination::synth_epoch(&tokens);
                if let Err(e) = self.forward_capture_downstream(kv_epoch, &tokens) {
                    warn!(error = %e, "qwen36: CAPTURE broadcast failed (best-effort)");
                }
            }
            self.kv_capture_local(tokens);
        }
        self.reset_all();
        let elapsed = t.started.elapsed().as_secs_f64();
        let tok_s = if elapsed > 0.0 {
            t.gen_ids.len() as f64 / elapsed
        } else {
            0.0
        };
        info!(
            task = %t.task_id,
            prompt_tokens = t.prompt_ids.len(),
            tokens = t.gen_ids.len(),
            elapsed_s = elapsed,
            tok_s,
            "qwen36 task done"
        );
        if !t.wire_ms.is_empty() {
            // Pipeline gate 4: decode wire histogram (p95 > 40 ms blocks; see
            // docs/architectures/qwen36-moe-support.md "Pipeline mode").
            let mut w = t.wire_ms.clone();
            w.sort_by(f64::total_cmp);
            let pct = |p: f64| w[((w.len() - 1) as f64 * p) as usize];
            info!(
                task = %t.task_id,
                frames = w.len(),
                p50_ms = pct(0.50),
                p95_ms = pct(0.95),
                max_ms = w[w.len() - 1],
                "qwen36 pipeline decode wire histogram"
            );
        }
        vec![(
            t.task_id.clone(),
            Chunk::final_marker(t.task_id, "").with_prompt_tokens(t.prompt_ids.len() as u32),
        )]
    }

    /// Terminate the active task as FAILED rather than completed: clear
    /// engine state (as `finalize` does) but emit an error chunk so the
    /// API returns a 5xx instead of a 200 with whatever partial text was
    /// streamed. For mid-generation backend/wire errors that previously
    /// fell through to `finalize` and read as a clean (empty/partial)
    /// success.
    fn finalize_error(&mut self, reason: String) -> Vec<(TaskId, Chunk)> {
        let Some(t) = self.active.take() else {
            return Vec::new();
        };
        self.reset_all();
        vec![(t.task_id.clone(), Chunk::error(t.task_id, reason))]
    }

    // -------- pipeline mode --------

    fn handle(&self) -> EngineResult<tokio::runtime::Handle> {
        self.runtime_handle
            .clone()
            .ok_or_else(|| EngineError::Backend("pipeline mode without runtime handle".into()))
    }

    /// Handshake payload: manifest full text, stage layout,
    /// wire dtype, protocol version. The shim exposes no OV version
    /// string; the manifest compare covers export-level skew.
    fn hello_payload(&self) -> Vec<u8> {
        serde_json::json!({
            "proto": PROTO_VERSION,
            "total": self.total,
            "wire": "f32",
            "manifest": self.manifest_json,
        })
        .to_string()
        .into_bytes()
    }

    /// Rank 0: HELLO → HELLO_ACK/NAK before the first admit.
    /// Middles chain the payload on, so one ACK means the whole chain
    /// validated against this rank's manifest.
    fn handshake_a(&mut self) -> EngineResult<()> {
        let payload = self.hello_payload();
        let nak = self.forward_hello_downstream(&payload)?;
        if let Some(reason) = nak {
            self.poisoned = Some(reason.clone());
            return Err(EngineError::Backend(format!(
                "qwen36 pipeline handshake refused: {reason}"
            )));
        }
        self.handshake_done = true;
        info!("qwen36 pipeline handshake ok");
        Ok(())
    }

    /// Rank 0: RESET → RESET_ACK for the current epoch.
    /// Fail-loud, no retry — the API caller retries the task.
    fn reset_exchange(&mut self) -> EngineResult<()> {
        let epoch = self.epoch;
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("rank 0 has no downstream".into()))?;
        let h = self.handle()?;
        run_async(
            &h,
            reply_bounded(async move {
                let mut g = downstream.lock().await;
                g.send_raw(&frame_header(FRAME_RESET, epoch, 0)).await?;
                let hb = g.recv_raw(12).await?;
                let (kind, e, _) = parse_header(&hb);
                if kind != FRAME_RESET_ACK || e != epoch {
                    return Err(TransportError::SocketClosed);
                }
                Ok(())
            }),
        )
        .map_err(|e| EngineError::Backend(format!("qwen36 pipeline RESET not acked: {e}")))
    }

    /// One lockstep FORWARD([1,n,HIDDEN] f32 at pos t0) → TOKEN exchange
    /// with the downstream peer. Returns the chain-end argmax token and
    /// the accumulated downstream infer time (µs) the TOKEN carried.
    fn forward_downstream(
        &mut self,
        epoch: u32,
        hidden: &[f32],
        n: usize,
        t0: usize,
    ) -> EngineResult<(i32, u32)> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let h = self.handle()?;
        let tensor = WireTensor::new(
            WireDType::F32,
            [1, n as u32, HIDDEN as u32],
            le_bytes_f32(hidden),
        );
        run_async(
            &h,
            reply_bounded(async move {
                let mut g = downstream.lock().await;
                g.send_raw(&frame_header(FRAME_FORWARD, epoch, t0 as u32))
                    .await?;
                g.send(&tensor).await?;
                let hb = g.recv_raw(12).await?;
                let (kind, e, _) = parse_header(&hb);
                if kind != FRAME_TOKEN || e != epoch {
                    return Err(TransportError::SocketClosed);
                }
                let tb = g.recv_raw(8).await?;
                Ok((
                    i32::from_be_bytes([tb[0], tb[1], tb[2], tb[3]]),
                    u32::from_be_bytes([tb[4], tb[5], tb[6], tb[7]]),
                ))
            }),
        )
        .map_err(map_wire)
    }

    /// Rank 0 wrapper: token + the round-trip's wire share in ms (RTT
    /// minus the chain's accumulated infer time) for the pipeline gate's
    /// wire histogram.
    fn send_forward_recv_token(
        &mut self,
        hidden: Vec<f32>,
        n: usize,
        t0: usize,
    ) -> EngineResult<(u32, f64)> {
        let epoch = self.epoch;
        let started = Instant::now();
        let (token, infer_us) = self.forward_downstream(epoch, &hidden, n, t0)?;
        let wire_ms =
            (started.elapsed().as_secs_f64() * 1000.0 - infer_us as f64 / 1000.0).max(0.0);
        Ok((token as u32, wire_ms))
    }

    /// Middle/last shared: forward the rank-0 HELLO payload downstream
    /// and return the reply (None = ACK, Some(reason) = NAK).
    fn forward_hello_downstream(&mut self, payload: &[u8]) -> EngineResult<Option<String>> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let h = self.handle()?;
        let payload = payload.to_vec();
        run_async(
            &h,
            reply_bounded(async move {
                let mut g = downstream.lock().await;
                g.send_raw(&frame_header(FRAME_HELLO, 0, 0)).await?;
                g.send_raw(&(payload.len() as u32).to_be_bytes()).await?;
                g.send_raw(&payload).await?;
                let hb = g.recv_raw(12).await?;
                let (kind, _, _) = parse_header(&hb);
                match kind {
                    FRAME_HELLO_ACK => Ok(None),
                    FRAME_HELLO_NAK => {
                        let lb = g.recv_raw(4).await?;
                        let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
                        let rb = g.recv_raw(n).await?;
                        Ok(Some(String::from_utf8_lossy(&rb).into_owned()))
                    }
                    other => Ok(Some(format!("unexpected handshake reply kind {other}"))),
                }
            }),
        )
        .map_err(map_wire)
    }

    /// Middle: forward RESET downstream, await its ack.
    fn forward_reset_downstream(&mut self, epoch: u32) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let h = self.handle()?;
        run_async(
            &h,
            reply_bounded(async move {
                let mut g = downstream.lock().await;
                g.send_raw(&frame_header(FRAME_RESET, epoch, 0)).await?;
                let hb = g.recv_raw(12).await?;
                let (kind, e, _) = parse_header(&hb);
                if kind != FRAME_RESET_ACK || e != epoch {
                    return Err(TransportError::SocketClosed);
                }
                Ok(())
            }),
        )
        .map_err(|e| EngineError::Backend(format!("qwen36 pipeline RESET not acked: {e}")))
    }

    /// Issue-34 §8: send `CAPTURE(kv_epoch, tokens)` to the downstream peer and await its ACK. Used
    /// by the head (rank 0, after a turn) and chained by each middle rank. Frame-header epoch is the
    /// current task epoch (stale-frame machinery); the KV content epoch rides the body.
    #[cfg(feature = "kv_coord")]
    fn forward_capture_downstream(&mut self, kv_epoch: u64, tokens: &[i32]) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let h = self.handle()?;
        let task_epoch = self.epoch;
        let body = crate::kv_coordination::capture_body_bytes(kv_epoch, tokens);
        run_async(
            &h,
            reply_bounded(async move {
                let mut g = downstream.lock().await;
                g.send_raw(&frame_header(FRAME_CAPTURE, task_epoch, 0))
                    .await?;
                g.send_raw(&(body.len() as u32).to_be_bytes()).await?;
                g.send_raw(&body).await?;
                let hb = g.recv_raw(12).await?;
                let (kind, _, _) = parse_header(&hb);
                if kind != FRAME_CAPTURE_ACK {
                    return Err(TransportError::SocketClosed);
                }
                Ok(())
            }),
        )
        .map_err(|e| EngineError::Backend(format!("qwen36 CAPTURE not acked: {e}")))
    }

    /// Rank 0 driver step: same task lifecycle as the single-box step,
    /// with the downstream stage + argmax behind the wire.
    fn step_pipe_first(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.active.is_none() {
            if self.pending.is_empty() {
                return Vec::new();
            }
            let task = self.pending.remove(0);
            if let Some(reason) = &self.poisoned {
                warn!(task = %task.task_id, reason = %reason, "qwen36: refusing task (handshake mismatch)");
                let reason = format!("qwen36 pipeline poisoned by handshake mismatch: {reason}");
                return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
            }
            if !self.handshake_done {
                if let Err(e) = self.handshake_a() {
                    warn!(task = %task.task_id, error = %e, "qwen36: handshake failed");
                    return vec![(
                        task.task_id.clone(),
                        Chunk::error(task.task_id, e.to_string()),
                    )];
                }
            }
            // Admission: reset local state, bump the epoch,
            // RESET/RESET_ACK the downstream, then admit at position 0.
            self.reset_all();
            self.epoch = self.epoch.wrapping_add(1);
            if let Err(e) = self.reset_exchange() {
                warn!(task = %task.task_id, error = %e, "qwen36: admission reset failed");
                return vec![(
                    task.task_id.clone(),
                    Chunk::error(task.task_id, e.to_string()),
                )];
            }
            let tokenizer = self.tokenizer.as_ref().expect("rank 0 has tokenizer");
            let mut prompt_ids: Vec<u32> = match tokenizer.encode(task.prompt.as_str(), true) {
                Ok(e) => e.get_ids().to_vec(),
                Err(e) => {
                    warn!(task = %task.task_id, error = %e, "tokenize failed");
                    let reason = format!("tokenize failed: {e}");
                    return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
                }
            };
            if !task.enable_thinking && !task.prompt.trim_end().ends_with("</think>") {
                // Legacy-rendered prompts only: a chat template injects
                // the empty think block itself (API passes
                // enable_thinking into the render).
                if let Ok(e) = tokenizer.encode("\n<think>\n\n</think>\n\n", false) {
                    prompt_ids.extend(e.get_ids());
                }
            }
            // An empty prompt leaves next_token None through prefill and would
            // panic the decode branch (`expect`); reject at admission.
            if prompt_ids.is_empty() {
                warn!(task = %task.task_id, "qwen36: empty prompt after tokenize; rejecting");
                let e = Chunk::error(
                    task.task_id.clone(),
                    "empty prompt after tokenize".to_string(),
                );
                return vec![(task.task_id, e)];
            }
            let max_tokens = if task.max_tokens > 0 {
                task.max_tokens
            } else {
                self.max_tokens_default
            } as usize;
            self.active = Some(ActiveTask {
                task_id: task.task_id,
                prompt_ids,
                prefill_idx: 0,
                step: 0,
                logits: Vec::new(),
                next_token: None,
                gen_ids: Vec::new(),
                emitted: 0,
                max_tokens,
                started: Instant::now(),
                wire_ms: Vec::new(),
            });
        }
        let t = self.active.as_ref().unwrap();
        let task_id = t.task_id.clone();

        if t.prefill_idx < t.prompt_ids.len() {
            // Full prefill in one step() call, chunked: per chunk the
            // local stage runs, the hidden span crosses the wire, and
            // the downstream answers with its argmax (intermediate
            // chunks' tokens are discarded; the last one is the first
            // decode token). Lockstep keeps the session quiescent
            // between steps, so cancel never races a frame in flight.
            while self.active.as_ref().unwrap().prefill_idx
                < self.active.as_ref().unwrap().prompt_ids.len()
            {
                let t = self.active.as_ref().unwrap();
                let end = (t.prefill_idx + PREFILL_CHUNK).min(t.prompt_ids.len());
                let toks: Vec<u32> = t.prompt_ids[t.prefill_idx..end].to_vec();
                let t0 = t.step;
                let n = toks.len();
                let res = self
                    .embed_seq(&toks)
                    .and_then(|e| self.chain_pass(&e, t0, t0 + n))
                    .and_then(|h| self.send_forward_recv_token(h, n, t0));
                match res {
                    Ok((tok, _wire_ms)) => {
                        let t = self.active.as_mut().unwrap();
                        t.next_token = Some(tok);
                        t.prefill_idx = end;
                        t.step += n;
                    }
                    Err(e) => {
                        warn!(task = %task_id, error = %e, "pipeline prefill failed");
                        return self.finalize_error(format!("pipeline prefill failed: {e}"));
                    }
                }
            }
            Vec::new()
        } else {
            if t.gen_ids.len() >= t.max_tokens {
                return self.finalize();
            }
            let next = t.next_token.expect("pipeline decode without pending token");
            if Some(next) == self.eos {
                return self.finalize();
            }
            let step = t.step;
            let res = self
                .embed_seq(&[next])
                .and_then(|e| self.chain_pass(&e, step, step + 1))
                .and_then(|h| self.send_forward_recv_token(h, 1, step));
            match res {
                Ok((tok, wire_ms)) => {
                    let t = self.active.as_mut().unwrap();
                    t.next_token = Some(tok);
                    t.wire_ms.push(wire_ms);
                    t.gen_ids.push(next);
                    t.step += 1;
                    let full = self
                        .tokenizer
                        .as_ref()
                        .expect("rank 0 has tokenizer")
                        .decode(self.active.as_ref().unwrap().gen_ids.as_slice(), true)
                        .unwrap_or_default();
                    let t = self.active.as_mut().unwrap();
                    let delta = if full.ends_with('\u{FFFD}') {
                        String::new()
                    } else {
                        let d = full.get(t.emitted..).unwrap_or("").to_string();
                        t.emitted = full.len();
                        d
                    };
                    vec![(task_id.clone(), Chunk::token(task_id, next as i64, delta))]
                }
                Err(e) => {
                    warn!(task = %task_id, error = %e, "pipeline decode failed");
                    self.finalize_error(format!("pipeline decode failed: {e}"))
                }
            }
        }
    }

    /// Middle/last ranks: serve one inbound frame per step (relay
    /// loop). Middles run their stage and pass the span downstream;
    /// the chain-end token relays back through them. Blocks
    /// in recv up to the activation timeout; idle timeouts surface as
    /// step errors (warned and ignored by the dispatch, like ov-runtime).
    fn step_pipe_relay(&mut self) -> EngineResult<()> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("last rank has no upstream".into()))?;
        let h = self.handle()?;
        let frame = run_async(&h, async move {
            let mut g = upstream.lock().await;
            let hb = g.recv_raw(12).await?;
            let (kind, epoch, pos) = parse_header(&hb);
            match kind {
                FRAME_HELLO => {
                    let lb = g.recv_raw(4).await?;
                    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
                    Ok(InFrame::Hello(g.recv_raw(n).await?))
                }
                FRAME_RESET => Ok(InFrame::Reset(epoch)),
                #[cfg(feature = "kv_coord")]
                FRAME_CAPTURE => {
                    let lb = g.recv_raw(4).await?;
                    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
                    let body = g.recv_raw(n).await?;
                    let (kv_epoch, tokens) = crate::kv_coordination::parse_capture_body(&body)
                        .ok_or(TransportError::SocketClosed)?;
                    Ok(InFrame::Capture { kv_epoch, tokens })
                }
                FRAME_FORWARD => {
                    let (t, _) = g.recv().await?;
                    if !matches!(t.dtype, WireDType::F32) {
                        return Err(TransportError::SocketClosed);
                    }
                    let n = t.shape[1] as usize;
                    Ok(InFrame::Forward {
                        epoch,
                        pos,
                        hidden: f32_from_le(&t.data),
                        n,
                    })
                }
                _ => Err(TransportError::SocketClosed),
            }
        })
        .map_err(map_wire)?;

        let is_last = self.rank == self.total - 1;
        match frame {
            InFrame::Hello(payload) => {
                // Validate locally, then (middles) chain the ORIGINAL
                // rank-0 payload downstream so every rank checks against
                // the origin; reply upstream with the combined verdict.
                let mut reason = self.validate_hello(&payload);
                if reason.is_none() && !is_last {
                    reason = match self.forward_hello_downstream(&payload) {
                        Ok(r) => r.map(|r| format!("downstream: {r}")),
                        Err(e) => Some(format!("downstream handshake forward failed: {e}")),
                    };
                }
                let h = self.handle()?;
                let upstream = self.upstream.clone().unwrap();
                match reason {
                    None => {
                        run_async(&h, async move {
                            let mut g = upstream.lock().await;
                            g.send_raw(&frame_header(FRAME_HELLO_ACK, 0, 0)).await
                        })
                        .map_err(map_wire)?;
                        info!("qwen36 pipeline handshake ok");
                    }
                    Some(reason) => {
                        warn!(reason = %reason, "qwen36 pipeline handshake mismatch; refusing to serve");
                        self.poisoned = Some(reason.clone());
                        let bytes = reason.into_bytes();
                        run_async(&h, async move {
                            let mut g = upstream.lock().await;
                            g.send_raw(&frame_header(FRAME_HELLO_NAK, 0, 0)).await?;
                            g.send_raw(&(bytes.len() as u32).to_be_bytes()).await?;
                            g.send_raw(&bytes).await
                        })
                        .map_err(map_wire)?;
                    }
                }
                Ok(())
            }
            InFrame::Reset(epoch) => {
                // Chain the reset before acking upstream: the ack means
                // "everything downstream of you is at position 0". A
                // failed downstream reset = no ack = task fails loud at
                // rank 0 (reset protocol).
                self.reset_all();
                self.peer_epoch = epoch;
                if !is_last {
                    self.forward_reset_downstream(epoch)?;
                }
                let h = self.handle()?;
                let upstream = self.upstream.clone().unwrap();
                run_async(&h, async move {
                    let mut g = upstream.lock().await;
                    g.send_raw(&frame_header(FRAME_RESET_ACK, epoch, 0)).await
                })
                .map_err(map_wire)
            }
            #[cfg(feature = "kv_coord")]
            InFrame::Capture { kv_epoch, tokens } => {
                // Snapshot this rank's local KV under the head's content epoch (state is still live —
                // RESET comes at the next admission), chain CAPTURE downstream, then ack upstream.
                // Best-effort: a blob/chain miss degrades to no warm-pull, never breaks generation.
                if let Some(blob) = self.blob_local_stages() {
                    self.kv.capture_under_epoch(kv_epoch, tokens.clone(), blob);
                }
                if !is_last {
                    if let Err(e) = self.forward_capture_downstream(kv_epoch, &tokens) {
                        warn!(error = %e, "qwen36: CAPTURE chain downstream failed (best-effort)");
                    }
                }
                let h = self.handle()?;
                let upstream = self.upstream.clone().unwrap();
                run_async(&h, async move {
                    let mut g = upstream.lock().await;
                    g.send_raw(&frame_header(FRAME_CAPTURE_ACK, 0, 0)).await
                })
                .map_err(map_wire)
            }
            InFrame::Forward {
                epoch,
                pos,
                hidden,
                n,
            } => {
                if self.poisoned.is_some() {
                    return Err(EngineError::Backend(
                        "qwen36 pipeline poisoned by handshake mismatch".into(),
                    ));
                }
                if epoch != self.peer_epoch {
                    // Stale epoch: drop silently; the driver's
                    // recv times out and fails its task loud.
                    warn!(
                        epoch,
                        current = self.peer_epoch,
                        "qwen36: dropping stale frame"
                    );
                    return Ok(());
                }
                if hidden.len() != n * HIDDEN {
                    return Err(EngineError::Backend(format!(
                        "forward frame size {} != n({n}) * HIDDEN",
                        hidden.len()
                    )));
                }
                let infer_started = Instant::now();
                let t0 = pos as usize;
                let out = self.chain_pass(&hidden, t0, t0 + n)?;
                // Own infer only — the downstream wait is wire + their
                // infer; they report their own share, so rank 0's
                // RTT-minus-infer stays the chain's true wire share.
                let own_us = infer_started.elapsed().as_micros().min(u32::MAX as u128) as u32;
                let (next, downstream_us) = if is_last {
                    let logits = if self.last_logits_only {
                        out
                    } else {
                        let row = out.len() / n;
                        out[(n - 1) * row..].to_vec()
                    };
                    let next = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as i32)
                        .unwrap_or(0);
                    (next, 0u32)
                } else {
                    // Middle: pass the hidden span on; the chain end's
                    // token comes back through us.
                    self.forward_downstream(epoch, &out, n, t0)?
                };
                let infer_us = own_us.saturating_add(downstream_us);
                let h = self.handle()?;
                let upstream = self.upstream.clone().unwrap();
                run_async(&h, async move {
                    let mut g = upstream.lock().await;
                    g.send_raw(&frame_header(FRAME_TOKEN, epoch, pos)).await?;
                    g.send_raw(&next.to_be_bytes()).await?;
                    g.send_raw(&infer_us.to_be_bytes()).await
                })
                .map_err(map_wire)
            }
        }
    }

    /// Downstream side of the handshake: compare the peer's payload
    /// against ours field by field.
    fn validate_hello(&self, payload: &[u8]) -> Option<String> {
        let theirs: serde_json::Value = match serde_json::from_slice(payload) {
            Ok(v) => v,
            Err(e) => return Some(format!("unparseable HELLO payload: {e}")),
        };
        if theirs["proto"] != serde_json::json!(PROTO_VERSION) {
            return Some(format!(
                "protocol version mismatch: theirs {} ours {PROTO_VERSION}",
                theirs["proto"]
            ));
        }
        if theirs["total"] != serde_json::json!(self.total) {
            return Some(format!(
                "stage count mismatch: theirs {} ours {}",
                theirs["total"], self.total
            ));
        }
        if theirs["wire"] != serde_json::json!("f32") {
            return Some(format!("wire dtype mismatch: theirs {}", theirs["wire"]));
        }
        if theirs["manifest"].as_str() != Some(self.manifest_json.as_str()) {
            return Some("manifest mismatch between ranks".into());
        }
        None
    }

    /// Single-box step path.
    fn step_local(&mut self) -> Vec<(TaskId, Chunk)> {
        // Admit the next task: position-0 entry per task (linear state
        // cannot be trimmed — docs/architectures/qwen36-moe-support.md §4.1).
        if self.active.is_none() {
            if self.pending.is_empty() {
                return Vec::new();
            }
            let task = self.pending.remove(0);
            // reset moved below — Issue-34 warm-resume may restore a cached prefix blob instead.
            let tokenizer = self.tokenizer.as_ref().expect("single-box has tokenizer");
            let mut prompt_ids: Vec<u32> = match tokenizer.encode(task.prompt.as_str(), true) {
                Ok(e) => e.get_ids().to_vec(),
                Err(e) => {
                    warn!(task = %task.task_id, error = %e, "tokenize failed");
                    let reason = format!("tokenize failed: {e}");
                    return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
                }
            };
            if !task.enable_thinking && !task.prompt.trim_end().ends_with("</think>") {
                // Hybrid-reasoning off: prefill the empty think block the
                // official chat template injects for enable_thinking=false,
                // so decode starts at the answer instead of reasoning.
                // Legacy-rendered prompts only — a chat template injects
                // it itself (API passes enable_thinking into the render).
                if let Ok(e) = tokenizer.encode("\n<think>\n\n</think>\n\n", false) {
                    prompt_ids.extend(e.get_ids());
                }
            }
            // An empty prompt leaves logits empty and would fabricate token 0
            // (garbage decode); reject at admission to match the pipeline path.
            if prompt_ids.is_empty() {
                warn!(task = %task.task_id, "qwen36: empty prompt after tokenize; rejecting");
                let e = Chunk::error(
                    task.task_id.clone(),
                    "empty prompt after tokenize".to_string(),
                );
                return vec![(task.task_id, e)];
            }
            let max_tokens = if task.max_tokens > 0 {
                task.max_tokens
            } else {
                self.max_tokens_default
            } as usize;
            // Issue-34 warm-resume: restore a cached strict-prefix blob and prefill only the suffix;
            // else cold reset. Gated + best-effort (stub ⇒ no blob ⇒ cold). 0 on the default path.
            let warm_prefix: usize = {
                #[cfg(feature = "kv_coord")]
                {
                    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&u| u as i32).collect();
                    match self.kv.take_warm(&prompt_i32) {
                        Some((blob, len)) if self.restore_local_stages(&blob) => {
                            info!(
                                warm_prefix = len,
                                "qwen36 single-box warm-resumed from KV blob"
                            );
                            len
                        }
                        _ => {
                            self.reset_all();
                            0
                        }
                    }
                }
                #[cfg(not(feature = "kv_coord"))]
                {
                    self.reset_all();
                    0
                }
            };
            self.active = Some(ActiveTask {
                task_id: task.task_id,
                prompt_ids,
                prefill_idx: warm_prefix,
                step: warm_prefix,
                logits: Vec::new(),
                next_token: None,
                gen_ids: Vec::new(),
                emitted: 0,
                max_tokens,
                started: Instant::now(),
                wire_ms: Vec::new(),
            });
        }
        let t = self.active.as_ref().unwrap();
        let task_id = t.task_id.clone();

        // Full prefill in one step() call (the runner closes streams
        // after 3 empty steps, so it can't be spread across calls),
        // batched in PREFILL_CHUNK spans — 4.2x TTFT vs T=1 stepping
        // (probe_batched_prefill.py); the chunk bounds the transient
        // [1, T, vocab] logits buffer. Then one decode token per call.
        if t.prefill_idx < t.prompt_ids.len() {
            while self.active.as_ref().unwrap().prefill_idx
                < self.active.as_ref().unwrap().prompt_ids.len()
            {
                let t = self.active.as_ref().unwrap();
                let end = (t.prefill_idx + PREFILL_CHUNK).min(t.prompt_ids.len());
                let toks: Vec<u32> = t.prompt_ids[t.prefill_idx..end].to_vec();
                let t0 = t.step;
                match self.run_span(&toks, t0) {
                    Ok(l) => {
                        let t = self.active.as_mut().unwrap();
                        t.logits = l;
                        t.prefill_idx = end;
                        t.step += toks.len();
                    }
                    Err(e) => {
                        warn!(task = %task_id, error = %e, "prefill failed");
                        return self.finalize_error(format!("prefill failed: {e}"));
                    }
                }
            }
            Vec::new()
        } else {
            if t.gen_ids.len() >= t.max_tokens {
                return self.finalize();
            }
            let next = t
                .logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            if Some(next) == self.eos {
                return self.finalize();
            }
            let step = t.step;
            match self.run_span(&[next], step) {
                Ok(l) => {
                    let t = self.active.as_mut().unwrap();
                    t.gen_ids.push(next);
                    t.logits = l;
                    t.step += 1;
                    // Delta detokenization: decode the full sequence and
                    // emit the unseen suffix. A trailing U+FFFD means a
                    // multi-token UTF-8 char is still incomplete — hold
                    // the suffix back until the next token completes it.
                    let full = self
                        .tokenizer
                        .as_ref()
                        .expect("single-box has tokenizer")
                        .decode(self.active.as_ref().unwrap().gen_ids.as_slice(), true)
                        .unwrap_or_default();
                    let t = self.active.as_mut().unwrap();
                    let delta = if full.ends_with('\u{FFFD}') {
                        String::new()
                    } else {
                        let d = full.get(t.emitted..).unwrap_or("").to_string();
                        t.emitted = full.len();
                        d
                    };
                    vec![(task_id.clone(), Chunk::token(task_id, next as i64, delta))]
                }
                Err(e) => {
                    warn!(task = %task_id, error = %e, "decode failed");
                    self.finalize_error(format!("decode failed: {e}"))
                }
            }
        }
    }
}

impl Engine for Qwen36Engine {
    fn warmup(&mut self) {
        if self.total > 1 && self.rank != 0 {
            // Last rank warms via its first real frame; the relay loop
            // owns the upstream session from here on.
            info!("qwen36-moe rank {}: skipping warmup (relay)", self.rank);
            return;
        }
        if self.total > 1 {
            // Local stage pass only (no logits on rank 0), then the
            // startup handshake — fail-loud at boot, not first request.
            self.reset_all();
            let r = self
                .embed_seq(&[1000])
                .and_then(|e| self.chain_pass(&e, 0, 1));
            match r {
                Ok(_) => info!("qwen36-moe warmup ok (stage0 local)"),
                Err(e) => warn!(error = %e, "qwen36-moe warmup failed"),
            }
            self.reset_all();
            if let Err(e) = self.handshake_a() {
                warn!(error = %e, "qwen36-moe startup handshake failed");
            }
            return;
        }
        self.reset_all();
        match self.run_span(&[1000], 0) {
            Ok(_) => info!("qwen36-moe warmup ok"),
            Err(e) => warn!(error = %e, "qwen36-moe warmup failed"),
        }
        self.reset_all();
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|t| t.task_id == task.task_id) {
            return Ok(());
        }
        if self.pending.len() >= crate::dist_spec::MAX_PENDING_TASKS {
            warn!(
                queued = self.pending.len(),
                cap = crate::dist_spec::MAX_PENDING_TASKS,
                "qwen36: pending queue at cap; rejecting task"
            );
            return Err(EngineError::QueueFull {
                queued: self.pending.len(),
                cap: crate::dist_spec::MAX_PENDING_TASKS,
            });
        }
        self.pending.push(task);
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        // EngineResult migration. step_local/step_pipe_first already absorb their
        // own task-level errors into the emitted Vec (final error chunks), so
        // they Ok-wrap directly.
        if self.total <= 1 {
            return Ok(self.step_local());
        }
        if self.rank == 0 {
            return Ok(self.step_pipe_first());
        }
        // Relay stage: PROPAGATE the error (like runtime/gemma4/sparse-moe). A
        // dead upstream is connection-fatal, and `run_relay_loop` only restarts
        // the stage when step() returns that Err (failure semantics: peer loss = operator
        // restart). Swallowing it into Ok(Vec::new()) would leave a silent
        // zombie that never rebuilds. The runner throttles non-fatal errors via
        // RELAY_ERR_BACKOFF, so no in-engine backoff is needed.
        let result = self.step_pipe_relay().map(|_| Vec::new());
        if let Err(e) = &result {
            warn!(error = %e, "qwen36 pipeline step failed");
        }
        result
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Immediate clear (PR #56 idiom): cancel and step never overlap
        // (both &mut self behind the runner's mutex), and the next
        // admission resets state, so dropping active directly is safe
        // and frees the engine slot without waiting for another poll.
        // Pipeline mode: the session is quiescent between steps
        // (lockstep frames), and the next admission's RESET + epoch
        // bump clears the downstream.
        self.pending.retain(|t| &t.task_id != task_id);
        if self.active.as_ref().is_some_and(|t| &t.task_id == task_id) {
            info!(task = %task_id, "qwen36: cancelled; dropping active task");
            self.active = None;
        }
    }

    #[cfg(feature = "kv_coord")]
    fn kv_coordination(&mut self) -> Option<&mut dyn cascadia_engine::KvCoordination> {
        // Only ranks that hold KV (a loaded stage) participate; emb-only rank-0 with no stage can't.
        if self.stages.is_empty() {
            return None;
        }
        Some(self)
    }
}

#[cfg(feature = "kv_coord")]
impl Qwen36Engine {
    /// Snapshot every local stage's OV KV state into one framed opaque blob (emb is stateless).
    /// `None` if any stage can't snapshot (e.g. stub build) — capture degrades to cold reprefill.
    fn blob_local_stages(&mut self) -> Option<Vec<u8>> {
        let mut blobs = Vec::with_capacity(self.stages.len());
        for st in self.stages.iter_mut() {
            match st.get_state_blob() {
                Ok(b) => blobs.push(b),
                Err(e) => {
                    tracing::debug!(error = %e, "qwen36: get_state_blob skipped (no KV capture)");
                    return None;
                }
            }
        }
        (!blobs.is_empty()).then(|| crate::kv_coordination::frame_blobs(&blobs))
    }

    /// Restore each local stage from a framed blob (inverse of [`Self::blob_local_stages`]).
    fn restore_local_stages(&mut self, blob: &[u8]) -> bool {
        let Some(parts) = crate::kv_coordination::unframe_blobs(blob) else {
            return false;
        };
        if parts.len() != self.stages.len() {
            return false;
        }
        for (st, part) in self.stages.iter_mut().zip(parts.iter()) {
            if let Err(e) = st.set_state_blob(part) {
                warn!(error = %e, "qwen36: set_state_blob failed; cold reprefill");
                return false;
            }
        }
        true
    }

    /// Stable model+rank fingerprint: a rank only matches the identical rank on a peer chain (its
    /// blob is its stage-span's KV, not the whole model's).
    fn kv_fingerprint(&self) -> u64 {
        let mut buf = self.manifest_json.clone().into_bytes();
        buf.extend_from_slice(&self.rank.to_le_bytes());
        buf.extend_from_slice(&self.total.to_le_bytes());
        crate::kv_coordination::fnv1a64(&buf)
    }

    /// Capture this rank's local KV under `tokens` (head/single-box token-keyed path). Called at the
    /// top of `finalize`, before `reset_all` wipes the state. Best-effort.
    fn kv_capture_local(&mut self, tokens: Vec<i32>) {
        if tokens.is_empty() {
            return;
        }
        if let Some(blob) = self.blob_local_stages() {
            self.kv.capture(tokens, blob);
        }
    }
}

#[cfg(feature = "kv_coord")]
impl cascadia_engine::KvCoordination for Qwen36Engine {
    fn model_fingerprint(&self) -> u64 {
        self.kv_fingerprint()
    }
    fn layout_version(&self) -> u16 {
        cascadia_kv_wire::OPAQUE_KV_LAYOUT
    }
    fn engine_rev(&self) -> u64 {
        crate::kv_coordination::KV_ENGINE_REV
    }
    fn tokenize(&self, text: &str) -> Option<Vec<i32>> {
        // add_special_tokens=true mirrors the prefill encode (step_pipe_first / step_local).
        let enc = self.tokenizer.as_ref()?.encode(text, true).ok()?;
        Some(enc.get_ids().iter().map(|&u| u as i32).collect())
    }
    fn lookup(&mut self, _partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        self.kv.lookup(token_ids)
    }
    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(cascadia_kv_wire::Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let fp = self.kv_fingerprint();
        let (prefix, blob) = self.kv.serve(expected_epoch, expected_len)?;
        Some(crate::kv_coordination::blob_to_wire(
            &prefix,
            &blob,
            partner,
            fp,
            expected_epoch,
        ))
    }
    fn insert(
        &mut self,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        let (tokens, blob) = crate::kv_coordination::wire_to_blob(manifest, payloads).ok_or(())?;
        self.kv.capture(tokens, blob);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = frame_header(FRAME_FORWARD, 7, 4096);
        assert_eq!(parse_header(&h), (FRAME_FORWARD, 7, 4096));
    }

    fn bare_engine(total: u32, manifest: &str) -> Qwen36Engine {
        Qwen36Engine {
            emb: None,
            stages: Vec::new(),
            tokenizer: None,
            eos: None,
            max_tokens_default: 256,
            last_logits_only: false,
            rank: 1,
            total,
            upstream: None,
            downstream: None,
            runtime_handle: None,
            manifest_json: manifest.to_string(),
            epoch: 0,
            peer_epoch: 0,
            handshake_done: false,
            poisoned: None,
            pending: Vec::new(),
            active: None,
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
        }
    }

    #[test]
    fn hello_validates_matching_payload() {
        let e = bare_engine(2, r#"{"arch":"qwen3_5_moe"}"#);
        let payload = e.hello_payload();
        assert_eq!(e.validate_hello(&payload), None);
    }

    #[test]
    fn hello_rejects_manifest_skew() {
        let a = bare_engine(2, r#"{"arch":"qwen3_5_moe","stages":1}"#);
        let b = bare_engine(2, r#"{"arch":"qwen3_5_moe","stages":2}"#);
        let reason = b.validate_hello(&a.hello_payload());
        assert!(reason.is_some_and(|r| r.contains("manifest")));
    }

    #[test]
    fn hello_rejects_total_mismatch() {
        let a = bare_engine(2, "{}");
        let b = bare_engine(3, "{}");
        let reason = b.validate_hello(&a.hello_payload());
        assert!(reason.is_some_and(|r| r.contains("stage count")));
    }

    #[test]
    fn poisoned_head_emits_error_chunk_not_empty_success() {
        // C1: a NAK'd handshake poisons the head; the next admitted task
        // must fail LOUD (carry an error) instead of returning an empty
        // final_marker that reads as a successful empty completion.
        let mut e = bare_engine(2, "{}");
        e.rank = 0;
        e.poisoned = Some("manifest mismatch between ranks".into());
        e.pending.push(GenerationTask::new("t0", "hello"));

        let out = e.step_pipe_first();
        assert_eq!(out.len(), 1);
        let (_, chunk) = &out[0];
        assert!(chunk.is_final, "error chunk must still terminate the task");
        assert!(
            chunk.error.is_some(),
            "poisoned head must emit an error chunk, got a silent empty success"
        );
        assert!(chunk.error.as_deref().unwrap().contains("manifest"));
    }

    /// Minimal real tokenizer: whitespace pre-tokenizer over a one-word
    /// vocab. Enough to reach the post-tokenize admission checks without a
    /// model — `""` encodes to zero ids, which is the case under test.
    fn tiny_tokenizer() -> Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"hi":0},"unk_token":"[UNK]"}}"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("build tiny tokenizer")
    }

    /// A prompt that tokenizes to nothing is REJECTED at admission — the
    /// decode branch would otherwise fabricate token 0 and emit garbage.
    /// The rejection must reach the caller as an error, not as an empty
    /// `final_marker`, which the API renders as a 200 with no content and
    /// the runner books as a successful zero-token generation (so
    /// `cascadia_tasks_failed_total` could never see it).
    ///
    /// `enable_thinking` is set so the empty-think-block injection is
    /// skipped — otherwise those ids would make the prompt non-empty and
    /// the branch under test unreachable.
    #[test]
    fn empty_prompt_is_rejected_loud_not_as_empty_success() {
        let mut e = bare_engine(1, "{}");
        e.rank = 0;
        e.tokenizer = Some(tiny_tokenizer());
        let mut task = GenerationTask::new("t-empty", "");
        task.enable_thinking = true;
        e.pending.push(task);

        let out = e.step_local();
        assert_eq!(out.len(), 1);
        let (_, chunk) = &out[0];
        assert!(chunk.is_final, "rejection must terminate the task");
        assert!(
            chunk.error.is_some(),
            "empty prompt must be rejected LOUD; got a silent empty success: {chunk:?}"
        );
        assert!(chunk.error.as_deref().unwrap().contains("empty prompt"));
    }

    #[test]
    fn finalize_error_emits_error_chunk_and_clears_active() {
        // A mid-generation backend/wire failure must terminate the task as
        // FAILED (error chunk + state cleared), not fall through to a clean
        // `finalize` that reads as an empty/partial success.
        let mut e = bare_engine(2, "{}");
        e.active = Some(ActiveTask {
            task_id: "t0".into(),
            prompt_ids: vec![1, 2, 3],
            prefill_idx: 3,
            step: 4,
            logits: Vec::new(),
            next_token: Some(5),
            gen_ids: vec![5],
            emitted: 0,
            max_tokens: 16,
            started: Instant::now(),
            wire_ms: Vec::new(),
        });

        let out = e.finalize_error("pipeline decode failed: wire closed".into());
        assert_eq!(out.len(), 1);
        let (_, chunk) = &out[0];
        assert!(chunk.is_final);
        assert_eq!(
            chunk.error.as_deref(),
            Some("pipeline decode failed: wire closed")
        );
        assert!(e.active.is_none(), "failed task must clear active state");
    }

    #[test]
    fn submit_caps_pending_queue() {
        use crate::dist_spec::MAX_PENDING_TASKS;
        let mut e = bare_engine(1, "{}");
        for i in 0..MAX_PENDING_TASKS {
            assert!(e.submit(GenerationTask::new(format!("t{i}"), "hi")).is_ok());
        }
        // Re-submitting an existing id is a no-op Ok (dedup), not a new slot.
        assert!(e.submit(GenerationTask::new("t0", "hi")).is_ok());
        // A new id past the cap is rejected rather than growing unbounded.
        let over = e.submit(GenerationTask::new("overflow", "hi"));
        assert!(matches!(
            over,
            Err(EngineError::QueueFull { queued, cap })
                if queued == MAX_PENDING_TASKS && cap == MAX_PENDING_TASKS
        ));
    }
}
