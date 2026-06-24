//! Tensor-parallel (Megatron-style) OpenVINO engine.
//!
//! Each TP rank holds a 1/N slice of every layer's weight matrices (exported by
//! `tools/tp_export.py` as segmented per-rank IR) and computes its partial of
//! attention and MLP. Between segments the ranks ALL-REDUCE (sum) the partials
//! over the existing transport, and the residual adds are done here in f32.
//!
//! Per-rank segment chain (from `rank_{tp_rank}/`):
//!   embed             input_ids -> hidden
//!   attn_{i}          hidden,mask,pos (+KV) -> partial_attn   (stateful: this rank's KV heads)
//!   mlp_{i}           hidden -> partial_mlp
//!   head              hidden -> logits                        (rank 0 only)
//!
//! Per token, per layer i (both ranks in lockstep):
//!   pa = attn_i(hidden);  hidden += all_reduce(pa)
//!   pm = mlp_i(hidden);   hidden += all_reduce(pm)
//!
//! Rank 0 drives generation (samples greedily, owns the tokenizer/API) and
//! broadcasts each step's input_ids to rank 1; rank 1 runs its half in a relay
//! loop driven by those broadcasts. Topology is a 2-node bidirectional peer
//! link: each rank has an ActivationServer (peer connects in) + ActivationClient
//! (connect to peer). all_reduce = concurrent send(own partial)/recv(peer's),
//! summed in f32. Both ranks round their own partial to f16 before summing so
//! the reduced result is bit-identical on both (no drift over many all-reduces).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{DType as ShimDType, PluginConfig, Runtime as OvRuntime};
use cascadia_transport::{
    ActivationClient, ActivationServer, DType as WireDType, Tensor as WireTensor, MAX_RANK,
};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::{info, warn};

// control bytes (rank0 -> rank1, prefix of each forward)
const CTRL_DECODE: u8 = 0;
const CTRL_PREFILL: u8 = 1; // reset KV state, then forward
const CTRL_STOP: u8 = 2; // generation finished; rank1 resets + waits

// -------- config --------
#[derive(Debug, Deserialize)]
struct TpPipelineConfig {
    model_id: String,
    tp_size: u32,
    num_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
}

#[derive(Debug, Deserialize)]
struct SegMeta {
    name: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct RankConfig {
    rank: u32,
    tp_size: u32,
    segments: Vec<SegMeta>,
}

// -------- helpers --------
fn i64_to_bytes(v: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}
fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&f16::from_f32(*x).to_bits().to_le_bytes());
    }
    out
}
fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn bytes_to_f32(dtype: ShimDType, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        ShimDType::F16 => f16_bytes_to_f32(bytes),
        _ => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}
fn argmax_last_row(logits: &[f32], vocab: usize) -> i32 {
    let row = &logits[logits.len() - vocab..];
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, v) in row.iter().enumerate() {
        if v.is_finite() && *v > bv {
            bv = *v;
            bi = i;
        }
    }
    bi as i32
}
fn map_ov(e: cascadia_ov_genai_shim::Error) -> EngineError {
    EngineError::Backend(format!("ov: {e:?}"))
}

/// A compiled segment + its canonical-name -> primary-port-name map.
struct Seg {
    rt: OvRuntime,
    names: std::collections::HashMap<String, String>, // canonical -> primary
}
impl Seg {
    fn compile(xml: &Path, device: &str, plugin: &PluginConfig) -> EngineResult<Self> {
        let rt = OvRuntime::compile(xml.to_str().unwrap_or_default(), device, plugin).map_err(map_ov)?;
        let mut names = std::collections::HashMap::new();
        let n = rt.input_count();
        for i in 0..n {
            let aliases = rt.input_aliases(i).map_err(map_ov)?;
            let primary = rt.input_name(i).map_err(map_ov)?;
            for canon in ["input_ids", "hidden_states", "attention_mask", "position_ids", "beam_idx"] {
                if aliases.iter().any(|a| a == canon) {
                    names.insert(canon.to_string(), primary.clone());
                }
            }
        }
        Ok(Seg { rt, names })
    }
    fn set(&mut self, canon: &str, dt: ShimDType, shape: &[usize], data: &[u8]) -> EngineResult<()> {
        let name = self.names.get(canon).cloned().unwrap_or_else(|| canon.to_string());
        self.rt.set_input(&name, dt, shape, data).map_err(map_ov)
    }
    /// infer and return output 0 as f32 + its shape.
    fn run(&mut self) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        self.rt.infer().map_err(map_ov)?;
        let (dt, shape, bytes) = self.rt.output(0).map_err(map_ov)?;
        Ok((bytes_to_f32(dt, &bytes), shape))
    }
}

struct ActiveTask {
    task: GenerationTask,
    prompt_ids: Vec<i64>,
    generated: Vec<i32>,
    last_text: String,
    prefilled: bool,
    last_token: i32,
    started: Instant,
}

pub struct TpRuntimeEngine {
    rank: u32,
    tp: u32,
    num_layers: usize,
    hidden: usize,
    embed: Seg,
    attn: Vec<Seg>,
    mlp: Vec<Seg>,
    head: Option<Seg>,
    kv_heads: usize,
    head_dim: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    eos: Vec<u32>,
    client: Arc<tokio::sync::Mutex<ActivationClient>>,
    server: Arc<tokio::sync::Mutex<ActivationServer>>,
    handle: tokio::runtime::Handle,
    position: i64,
    pending: Vec<GenerationTask>,
    active: Option<ActiveTask>,
    // instrumentation (per task, decode-phase): GPU segment-infer time vs all-reduce time
    t_gpu_us: u128,
    t_ar_us: u128,
}

impl TpRuntimeEngine {
    fn block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        crate::dist_spec::run_async_pub(&self.handle, f)
    }

    fn reset_kv(&mut self) -> EngineResult<()> {
        for s in &mut self.attn {
            s.rt.reset_state().map_err(map_ov)?;
        }
        Ok(())
    }

    /// Exchange `partial` (f16) with the peer and return the f32 sum. Both ranks
    /// round their own partial to f16 first so the reduced result is identical.
    fn all_reduce(&self, partial: &[f32], shape3: [u32; MAX_RANK]) -> EngineResult<Vec<f32>> {
        let own_f16 = f32_to_f16_bytes(partial);
        let wire = WireTensor::new(WireDType::F16, shape3, own_f16.clone());
        let client = self.client.clone();
        let server = self.server.clone();
        let peer = self
            .block_on(async move {
                let mut c = client.lock().await;
                let mut s = server.lock().await;
                let (sr, rr) = tokio::join!(c.send(&wire), s.recv());
                sr?;
                rr.map(|(t, _)| t)
            })
            .map_err(|e: cascadia_transport::TransportError| EngineError::Backend(format!("all_reduce: {e}")))?;
        let own = f16_bytes_to_f32(&own_f16);
        let peer_f = f16_bytes_to_f32(&peer.data);
        if peer_f.len() != own.len() {
            return Err(EngineError::Backend(format!(
                "all_reduce size mismatch own={} peer={}",
                own.len(),
                peer_f.len()
            )));
        }
        Ok(own.iter().zip(peer_f).map(|(a, b)| a + b).collect())
    }

    /// Run the full per-rank forward for `input_ids` at `position`, returning the
    /// logits (rank 0 only) + shape. Both ranks run their segments in lockstep,
    /// all-reducing partials and adding residuals in f32.
    fn forward(&mut self, input_ids: &[i64], position: i64) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        let seq = input_ids.len();
        let total = position as usize + seq;
        let attn_mask = i64_to_bytes(&vec![1i64; total]);
        let pos_ids = i64_to_bytes(&(position..position + seq as i64).collect::<Vec<_>>());
        let shape3: [u32; MAX_RANK] = [1, seq as u32, self.hidden as u32];

        // embed
        self.embed.set("input_ids", ShimDType::I64, &[1, seq], &i64_to_bytes(input_ids))?;
        let (mut hidden, _) = self.embed.run()?;

        for i in 0..self.num_layers {
            // ----- attention segment -----
            let hf16 = f32_to_bytes(&hidden);
            {
                let s = &mut self.attn[i];
                s.set("hidden_states", ShimDType::F32, &[1, seq, self.hidden], &hf16)?;
                s.set("attention_mask", ShimDType::I64, &[1, total], &attn_mask)?;
                s.set("position_ids", ShimDType::I64, &[1, seq], &pos_ids)?;
                if s.names.contains_key("beam_idx") {
                    s.set("beam_idx", ShimDType::I32, &[1], &0i32.to_le_bytes())?;
                }
            }
            let t = Instant::now();
            let (pa, _) = self.attn[i].run()?;
            self.t_gpu_us += t.elapsed().as_micros();
            let t = Instant::now();
            let ra = self.all_reduce(&pa, shape3)?;
            self.t_ar_us += t.elapsed().as_micros();
            for (h, r) in hidden.iter_mut().zip(ra) {
                *h += r;
            }
            // ----- mlp segment -----
            let hf16 = f32_to_bytes(&hidden);
            self.mlp[i].set("hidden_states", ShimDType::F32, &[1, seq, self.hidden], &hf16)?;
            let t = Instant::now();
            let (pm, _) = self.mlp[i].run()?;
            self.t_gpu_us += t.elapsed().as_micros();
            let t = Instant::now();
            let rm = self.all_reduce(&pm, shape3)?;
            self.t_ar_us += t.elapsed().as_micros();
            for (h, r) in hidden.iter_mut().zip(rm) {
                *h += r;
            }
        }

        if let Some(head) = &mut self.head {
            let hf16 = f32_to_bytes(&hidden);
            head.set("hidden_states", ShimDType::F32, &[1, seq, self.hidden], &hf16)?;
            let t = Instant::now();
            let r = head.run();
            self.t_gpu_us += t.elapsed().as_micros();
            r
        } else {
            Ok((Vec::new(), vec![1, seq, 0]))
        }
    }

    // ---- rank-0 driver: send control + input_ids to rank 1 ----
    fn broadcast(&self, ctrl: u8, input_ids: &[i64]) -> EngineResult<()> {
        let client = self.client.clone();
        let frame = if ctrl == CTRL_STOP {
            None
        } else {
            Some(WireTensor::new(
                WireDType::I64,
                [1, 1, input_ids.len() as u32],
                i64_to_bytes(input_ids),
            ))
        };
        self.block_on(async move {
            let mut c = client.lock().await;
            c.send_raw(&[ctrl]).await?;
            if let Some(f) = frame {
                c.send(&f).await?;
            }
            Ok::<_, cascadia_transport::TransportError>(())
        })
        .map_err(|e| EngineError::Backend(format!("broadcast: {e}")))
    }

    // ---- rank-1 worker: receive control + input_ids from rank 0 ----
    fn recv_broadcast(&self) -> EngineResult<Option<(u8, Vec<i64>)>> {
        let server = self.server.clone();
        let (ctrl, ids) = self
            .block_on(async move {
                let mut s = server.lock().await;
                let cb = s.recv_raw(1).await?;
                let ctrl = cb.first().copied().unwrap_or(CTRL_STOP);
                if ctrl == CTRL_STOP {
                    return Ok::<_, cascadia_transport::TransportError>((ctrl, Vec::new()));
                }
                let (t, _) = s.recv().await?;
                let ids: Vec<i64> = t
                    .data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Ok((ctrl, ids))
            })
            .map_err(|e| EngineError::Backend(format!("recv_broadcast: {e}")))?;
        if ctrl == CTRL_STOP {
            Ok(None)
        } else {
            Ok(Some((ctrl, ids)))
        }
    }

    fn vocab_argmax(&self, logits: &[f32], shape: &[usize]) -> EngineResult<i32> {
        let vocab = *shape.last().unwrap_or(&0);
        if vocab == 0 || vocab > logits.len() {
            return Err(EngineError::Backend(format!(
                "bad logits shape {shape:?} len {}",
                logits.len()
            )));
        }
        Ok(argmax_last_row(logits, vocab))
    }

    fn emit(&mut self, token: i32) -> Vec<(TaskId, Chunk)> {
        let tok = match &self.tokenizer {
            Some(t) => t.clone(),
            None => return Vec::new(),
        };
        let a = match self.active.as_mut() {
            Some(a) => a,
            None => return Vec::new(),
        };
        a.generated.push(token);
        a.last_token = token;
        let ids_u32: Vec<u32> = a.generated.iter().map(|&t| t as u32).collect();
        let full = tok.decode(&ids_u32, true).unwrap_or_default();
        let delta = full.strip_prefix(&a.last_text).unwrap_or(&full).to_string();
        a.last_text = full.clone();
        let tid = a.task.task_id.clone();
        let max = a.task.max_tokens.max(1) as usize;
        let done = a.generated.len() >= max || self.eos.contains(&(token as u32));
        let chunk = if done {
            Chunk::final_marker(tid.clone(), delta)
        } else {
            Chunk::token(&tid, (a.generated.len() - 1) as i64, delta)
        };
        if done {
            let elapsed = a.started.elapsed();
            let n = a.generated.len();
            info!(task = %tid, tokens = n, elapsed_s = elapsed.as_secs_f64(),
                  tok_s = n as f64 / elapsed.as_secs_f64(),
                  gpu_ms = self.t_gpu_us as f64 / 1000.0,
                  allreduce_ms = self.t_ar_us as f64 / 1000.0,
                  "tp-runtime task done");
            // tell rank 1 the request is over
            let _ = self.broadcast(CTRL_STOP, &[]);
            self.active = None;
        }
        vec![(tid, chunk)]
    }
}

impl Engine for TpRuntimeEngine {
    fn warmup(&mut self) {
        // every rank JITs its segments on a 1-token forward
        let _ = self.reset_kv();
        let _ = self.forward(&[1i64], 0);
        let _ = self.reset_kv();
        self.position = 0;
        if self.rank == 0 {
            let _ = self.broadcast(CTRL_STOP, &[]);
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|t| t.task_id == task.task_id) {
            return Ok(());
        }
        self.pending.push(task);
        Ok(())
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.rank == 0 {
            self.step_rank0()
        } else {
            if let Err(e) = self.step_rank1() {
                warn!("tp rank1 step: {e}");
            }
            Vec::new()
        }
    }
}

impl TpRuntimeEngine {
    fn step_rank0(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.active.is_none() {
            if self.pending.is_empty() {
                return Vec::new();
            }
            let task = self.pending.remove(0);
            let enc = match &self.tokenizer {
                Some(t) => t.encode(task.prompt.clone(), true),
                None => return Vec::new(),
            };
            let prompt_ids: Vec<i64> = match enc {
                Ok(e) => e.get_ids().iter().map(|&x| x as i64).collect(),
                Err(e) => {
                    warn!("encode: {e}");
                    return Vec::new();
                }
            };
            if let Err(e) = self.reset_kv() {
                warn!("reset: {e}");
                return Vec::new();
            }
            self.position = 0;
            self.t_gpu_us = 0;
            self.t_ar_us = 0;
            self.active = Some(ActiveTask {
                task,
                prompt_ids,
                generated: Vec::new(),
                last_text: String::new(),
                prefilled: false,
                last_token: 0,
                started: Instant::now(),
            });
        }
        let (ctrl, input_ids) = {
            let a = self.active.as_ref().unwrap();
            if !a.prefilled {
                (CTRL_PREFILL, a.prompt_ids.clone())
            } else {
                (CTRL_DECODE, vec![a.last_token as i64])
            }
        };
        if let Err(e) = self.broadcast(ctrl, &input_ids) {
            warn!("broadcast: {e}");
            return Vec::new();
        }
        let pos = self.position;
        let res = self.forward(&input_ids, pos);
        self.position += input_ids.len() as i64;
        if let Some(a) = self.active.as_mut() {
            a.prefilled = true;
        }
        match res {
            Ok((logits, shape)) => match self.vocab_argmax(&logits, &shape) {
                Ok(tok) => self.emit(tok),
                Err(e) => {
                    warn!("argmax: {e}");
                    Vec::new()
                }
            },
            Err(e) => {
                warn!("forward: {e}");
                Vec::new()
            }
        }
    }

    fn step_rank1(&mut self) -> EngineResult<()> {
        let bc = self.recv_broadcast()?;
        let (ctrl, input_ids) = match bc {
            None => {
                // end of request
                self.reset_kv()?;
                self.position = 0;
                return Ok(());
            }
            Some(x) => x,
        };
        if ctrl == CTRL_PREFILL {
            self.reset_kv()?;
            self.position = 0;
        }
        let pos = self.position;
        self.forward(&input_ids, pos)?;
        self.position += input_ids.len() as i64;
        Ok(())
    }
}

// ============================ Builder ============================

#[derive(Default)]
pub struct TpRuntimeBuilder {
    pipeline_dir: PathBuf,
    rank: u32,
    total: u32,
    device: String,
    cache_dir: Option<String>,
    listen_host: String,
    listen_port: Option<u16>,
    client: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    server: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    num_layers: usize,
    hidden: usize,
    kv_heads: usize,
    head_dim: usize,
    embed: Option<Seg>,
    attn: Vec<Option<Seg>>,
    mlp: Vec<Option<Seg>>,
    head: Option<Seg>,
    tokenizer: Option<Arc<Tokenizer>>,
    eos: Vec<u32>,
}

impl TpRuntimeBuilder {
    pub fn new(pipeline_dir: &str, rank: u32, total: u32, device: &str) -> Self {
        Self {
            pipeline_dir: PathBuf::from(pipeline_dir),
            rank,
            total,
            device: device.to_string(),
            listen_host: "0.0.0.0".to_string(),
            ..Default::default()
        }
    }
    pub fn with_cache_dir(mut self, d: &str) -> Self {
        self.cache_dir = Some(d.to_string());
        self
    }
    fn plugin(&self) -> PluginConfig {
        let mut p = PluginConfig::new();
        if let Some(d) = &self.cache_dir {
            p = p.with("CACHE_DIR", d);
        }
        p
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(p: &Path) -> EngineResult<T> {
    let bytes = std::fs::read(p).map_err(|e| EngineError::InvalidConfig(format!("{}: {e}", p.display())))?;
    serde_json::from_slice(&bytes).map_err(|e| EngineError::InvalidConfig(format!("{}: {e}", p.display())))
}

#[async_trait]
impl Builder for TpRuntimeBuilder {
    fn configure_listen(&mut self, host: &str, port: u16) {
        self.listen_host = host.to_string();
        self.listen_port = Some(port);
    }

    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        // 2-node bidirectional peer link: bind our server, connect our client to
        // the peer, then accept the peer's client. Both ranks do this symmetrically.
        let port = self
            .listen_port
            .ok_or_else(|| EngineError::PeerRejected("configure_listen() required for tp-runtime".into()))?;
        let mut server = ActivationServer::new(self.listen_host.clone(), port);
        server.start().await.map_err(|e| EngineError::Backend(e.to_string()))?;
        let server = Arc::new(tokio::sync::Mutex::new(server));
        let peer = peers
            .downstream
            .ok_or_else(|| EngineError::PeerRejected("--next (peer) required for tp-runtime".into()))?;
        let mut client = ActivationClient::new(peer.host, peer.port);
        client
            .connect_with_timeout(std::time::Duration::from_secs(60))
            .await
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        server
            .lock()
            .await
            .accept()
            .await
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        self.server = Some(server);
        self.client = Some(Arc::new(tokio::sync::Mutex::new(client)));
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        let pc: TpPipelineConfig = read_json(&self.pipeline_dir.join("pipeline_config.json"))?;
        if pc.tp_size != self.total {
            return Err(EngineError::ShardRejected(format!(
                "shard tp_size {} != --total {}",
                pc.tp_size, self.total
            )));
        }
        self.num_layers = pc.num_layers as usize;
        self.hidden = pc.hidden_size as usize;
        let rdir = self.pipeline_dir.join(format!("rank_{}", self.rank));
        let rc: RankConfig = read_json(&rdir.join("rank_config.json"))?;
        self.attn = (0..self.num_layers).map(|_| None).collect();
        self.mlp = (0..self.num_layers).map(|_| None).collect();
        let plugin = self.plugin();
        for s in &rc.segments {
            let xml = rdir.join(&s.name).join("openvino_model.xml");
            let seg = Seg::compile(&xml, &self.device, &plugin)?;
            match s.kind.as_str() {
                "embed" => self.embed = Some(seg),
                "head" => self.head = Some(seg),
                "attn" => {
                    let i: usize = s.name.trim_start_matches("attn_").parse().unwrap_or(0);
                    self.attn[i] = Some(seg);
                }
                "mlp" => {
                    let i: usize = s.name.trim_start_matches("mlp_").parse().unwrap_or(0);
                    self.mlp[i] = Some(seg);
                }
                _ => {}
            }
        }
        self.kv_heads = (pc.num_layers as usize).max(1); // placeholder; not used at runtime
        let _ = &mut self.kv_heads;
        // tokenizer + EOS only needed on rank 0 (it samples + emits)
        if self.rank == 0 {
            let tdir = self.pipeline_dir.join("tokenizer");
            if let Ok(t) = Tokenizer::from_file(tdir.join("tokenizer.json")) {
                self.tokenizer = Some(Arc::new(t));
            }
            self.eos = lookup_eos_tp(&tdir);
            if self.eos.is_empty() {
                self.eos = lookup_eos_tp(&self.pipeline_dir);
            }
        }
        let _ = pc.model_id;
        let _ = pc.vocab_size;
        Ok(Box::pin(stream::iter(vec![
            LoadProgress::message("tp-runtime: segments compiled"),
            LoadProgress::ready(),
        ])))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let s = *self;
        let embed = s.embed.ok_or(EngineError::NotLoaded)?;
        let attn: Vec<Seg> = s
            .attn
            .into_iter()
            .map(|x| x.ok_or(EngineError::NotLoaded))
            .collect::<Result<_, _>>()?;
        let mlp: Vec<Seg> = s
            .mlp
            .into_iter()
            .map(|x| x.ok_or(EngineError::NotLoaded))
            .collect::<Result<_, _>>()?;
        let client = s.client.ok_or_else(|| EngineError::PeerRejected("not connected".into()))?;
        let server = s.server.ok_or_else(|| EngineError::PeerRejected("not connected".into()))?;
        Ok(Box::new(TpRuntimeEngine {
            rank: s.rank,
            tp: s.total,
            num_layers: s.num_layers,
            hidden: s.hidden,
            embed,
            attn,
            mlp,
            head: s.head,
            kv_heads: s.kv_heads,
            head_dim: s.head_dim,
            tokenizer: s.tokenizer,
            eos: s.eos,
            client,
            server,
            handle: tokio::runtime::Handle::current(),
            position: 0,
            pending: Vec::new(),
            active: None,
            t_gpu_us: 0,
            t_ar_us: 0,
        }))
    }
}

#[derive(Debug, Deserialize)]
struct GenCfgTp {
    eos_token_id: Option<serde_json::Value>,
}
fn lookup_eos_tp(dir: &Path) -> Vec<u32> {
    for f in ["generation_config.json", "config.json"] {
        if let Ok(b) = std::fs::read(dir.join(f)) {
            if let Ok(g) = serde_json::from_slice::<GenCfgTp>(&b) {
                match g.eos_token_id {
                    Some(serde_json::Value::Number(n)) => {
                        if let Some(i) = n.as_u64() {
                            return vec![i as u32];
                        }
                    }
                    Some(serde_json::Value::Array(a)) => {
                        return a.into_iter().filter_map(|v| v.as_u64().map(|i| i as u32)).collect();
                    }
                    _ => {}
                }
            }
        }
    }
    Vec::new()
}
