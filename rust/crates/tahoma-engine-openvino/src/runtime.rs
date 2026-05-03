//! Multi-stage OpenVINO Runtime engine.
//!
//! Rust port of `tahoma/worker/engines/openvino/ov_runtime.py`. Loads
//! pre-exported per-stage stateful OV IRs (rainier v3+ format), runs them
//! across the existing TCP transport, with stateful KV cache internal to
//! the IR and `reset_state()` between independent generation tasks.
//!
//! Pipeline-dir layout (matches rainier's exporter):
//! ```text
//! <pipeline-dir>/
//!     pipeline_config.json
//!     tokenizer/                 # HF tokenizer.json + special tokens
//!     stage_0/openvino_model.{xml,bin}, stage_config.json
//!     stage_N/...
//! ```
//!
//! Wire format between stages: hidden_states f16. Each stage tracks its
//! own absolute-position counter so it can compute (cos, sin) locally
//! without sending position metadata. Counter resets when an activation
//! with seq_len > 1 arrives (signals new prefill on relay/last stages).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use serde::Deserialize;
use tahoma_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use tahoma_ov_genai_shim::{
    DType as ShimDType, Error as OvError, PluginConfig, Runtime as OvRuntime,
};
use tahoma_transport::{
    ActivationClient, ActivationServer, DType as WireDType, Tensor as WireTensor, MAX_RANK,
};
use tahoma_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::rotary::{load_model_config, Rotary};

// -------- pipeline / stage config --------

#[derive(Debug, Deserialize)]
struct PipelineConfig {
    model_id: String,
    num_stages: u32,
    #[serde(default)]
    num_layers: u32,
    #[serde(default)]
    hidden_size: u32,
    #[serde(default)]
    export_version: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct StageConfig {
    #[serde(default)]
    layer_start: u32,
    #[serde(default)]
    layer_end: u32,
    #[serde(default)]
    has_embed: bool,
    #[serde(default)]
    has_head: bool,
    #[serde(default)]
    export_version: Option<String>,
}

fn read_pipeline_config(p: &Path) -> Result<PipelineConfig, EngineError> {
    let bytes = std::fs::read(p.join("pipeline_config.json"))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| EngineError::InvalidConfig(format!("pipeline_config.json: {e}")))
}

fn read_stage_config(p: &Path) -> Result<StageConfig, EngineError> {
    let bytes = std::fs::read(p.join("stage_config.json"))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| EngineError::InvalidConfig(format!("stage_config.json: {e}")))
}

// -------- generation_config.json (eos_token_id lookup) --------

#[derive(Debug, Deserialize, Default)]
struct GenerationCfg {
    #[serde(default)]
    eos_token_id: Option<EosId>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosId {
    One(u32),
    Many(Vec<u32>),
}

fn lookup_eos(model_dir: &Path) -> Option<u32> {
    for fname in ["generation_config.json", "config.json"] {
        let p = model_dir.join(fname);
        if let Ok(bytes) = std::fs::read(&p) {
            if let Ok(g) = serde_json::from_slice::<GenerationCfg>(&bytes) {
                return match g.eos_token_id {
                    Some(EosId::One(id)) => Some(id),
                    Some(EosId::Many(ids)) => ids.first().copied(),
                    None => None,
                };
            }
        }
    }
    None
}

// -------- helpers: bytes <-> typed slices --------

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

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
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        let h = f16::from_f32(*x);
        out.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    out
}

fn argmax_last_row(logits: &[f32], vocab: usize) -> i32 {
    let row = &logits[logits.len() - vocab..];
    // NaN-aware (see crate::dist_spec::argmax for rationale).
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut saw_finite = false;
    for (i, v) in row.iter().enumerate() {
        if v.is_finite() {
            saw_finite = true;
            if *v > best_v {
                best_v = *v;
                best_i = i;
            }
        }
    }
    if !saw_finite {
        warn!(
            "argmax_last_row: all logits non-finite; returning token 0 — \
             likely indicates a numerically broken forward pass"
        );
    }
    best_i as i32
}

fn map_ov_err(err: OvError) -> EngineError {
    match err {
        OvError::Stub => EngineError::Backend(
            "openvino shim built without --features openvino".into(),
        ),
        OvError::Utf8(s) => EngineError::InvalidConfig(s),
        OvError::Native(s) => EngineError::Backend(s),
    }
}

// -------- Engine --------

struct ActiveTask {
    task: GenerationTask,
    prompt_ids: Vec<i64>,
    generated: Vec<i32>,
    last_text: String,
    prefilled: bool,
    last_token: i32,
}

pub struct OvRuntimeEngine {
    spec: ShardSpec,
    runtime: OvRuntime,
    rotary: Rotary,
    hidden_size: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    eos_token_id: Option<u32>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    runtime_handle: tokio::runtime::Handle,
    position: i64,
    input_names: Vec<String>,
    pending: Vec<GenerationTask>,
    active: Option<ActiveTask>,
}

impl OvRuntimeEngine {
    fn build_feed_first(&mut self, input_ids: &[i64], position: i64) -> EngineResult<()> {
        let seq_len = input_ids.len();
        let (cos, sin) = self.rotary.compute(position, seq_len);
        // v3 inputs are positional: (input_ids|hs, cos, sin).
        let names = &self.input_names;
        if names.len() < 3 {
            return Err(EngineError::Backend(format!(
                "shard expected >=3 inputs, got {}: {:?}",
                names.len(),
                names
            )));
        }
        // input_ids (i64, [1, seq_len])
        let bytes = i64_to_bytes(input_ids);
        self.runtime
            .set_input(&names[0], ShimDType::I64, &[1, seq_len], &bytes)
            .map_err(map_ov_err)?;
        // cos (f16, [1, seq_len, head_dim]) — v3 shards exported with
        // default_dtype=fp16 so the cos/sin graph inputs are f16. The
        // OV GPU plugin won't auto-cast f32 inputs to an f16 port.
        let cos_bytes = f32_to_f16_bytes(&cos);
        self.runtime
            .set_input(
                &names[1],
                ShimDType::F16,
                &[1, seq_len, self.rotary.head_dim()],
                &cos_bytes,
            )
            .map_err(map_ov_err)?;
        // sin (f16, [1, seq_len, head_dim])
        let sin_bytes = f32_to_f16_bytes(&sin);
        self.runtime
            .set_input(
                &names[2],
                ShimDType::F16,
                &[1, seq_len, self.rotary.head_dim()],
                &sin_bytes,
            )
            .map_err(map_ov_err)?;
        Ok(())
    }

    fn build_feed_relay(&mut self, hidden: &[f32], shape: [usize; 3], position: i64) -> EngineResult<()> {
        let seq_len = shape[1];
        let (cos, sin) = self.rotary.compute(position, seq_len);
        let names = &self.input_names;
        if names.len() < 3 {
            return Err(EngineError::Backend(format!(
                "shard expected >=3 inputs, got {}: {:?}",
                names.len(),
                names
            )));
        }
        // hidden_states (f16) — same reason as cos/sin above
        let hs_bytes = f32_to_f16_bytes(hidden);
        self.runtime
            .set_input(&names[0], ShimDType::F16, &shape, &hs_bytes)
            .map_err(map_ov_err)?;
        let cos_bytes = f32_to_f16_bytes(&cos);
        self.runtime
            .set_input(
                &names[1],
                ShimDType::F16,
                &[1, seq_len, self.rotary.head_dim()],
                &cos_bytes,
            )
            .map_err(map_ov_err)?;
        let sin_bytes = f32_to_f16_bytes(&sin);
        self.runtime
            .set_input(
                &names[2],
                ShimDType::F16,
                &[1, seq_len, self.rotary.head_dim()],
                &sin_bytes,
            )
            .map_err(map_ov_err)?;
        Ok(())
    }

    fn run_first(&mut self, input_ids: &[i64], position: i64) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        self.build_feed_first(input_ids, position)?;
        self.runtime.infer().map_err(map_ov_err)?;
        let (dtype, shape, bytes) = self.runtime.output(0).map_err(map_ov_err)?;
        let floats = match dtype {
            ShimDType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
            ShimDType::F16 => f16_bytes_to_f32(&bytes),
            other => {
                return Err(EngineError::Backend(format!(
                    "unexpected output dtype {other:?}"
                )))
            }
        };
        Ok((floats, shape))
    }

    fn run_relay(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        self.build_feed_relay(hidden, shape, position)?;
        self.runtime.infer().map_err(map_ov_err)?;
        let (dtype, out_shape, bytes) = self.runtime.output(0).map_err(map_ov_err)?;
        let floats = match dtype {
            ShimDType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
            ShimDType::F16 => f16_bytes_to_f32(&bytes),
            other => {
                return Err(EngineError::Backend(format!(
                    "unexpected output dtype {other:?}"
                )))
            }
        };
        Ok((floats, out_shape))
    }

    fn block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        // The runner's ChunkStream::poll_next is itself an async fn —
        // calling block_on inside an async context panics with "Cannot
        // start a runtime from within a runtime". Use the same
        // dispatch the dist_spec engine uses (block_in_place when on
        // the async worker thread, naked block_on when on a
        // spawn_blocking thread or off-runtime).
        crate::dist_spec::run_async_pub(&self.runtime_handle, f)
    }

    fn send_hidden_downstream(&mut self, hidden: &[f32], shape: [usize; 3]) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let bytes = f32_to_f16_bytes(hidden);
        let mut wire_shape = [1u32; MAX_RANK];
        for (i, d) in shape.iter().enumerate().take(MAX_RANK) {
            wire_shape[i] = *d as u32;
        }
        let tensor = WireTensor::new(WireDType::F16, wire_shape, bytes);
        self.block_on(async move {
            let mut guard = downstream.lock().await;
            guard.send(&tensor).await
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    fn recv_token_from_downstream(&mut self) -> EngineResult<i32> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let (tensor, _) = self
            .block_on(async move {
                let mut guard = downstream.lock().await;
                guard.recv().await
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        if tensor.data.len() < 4 {
            return Err(EngineError::Backend(format!(
                "downstream sent {}-byte token tensor; need at least 4",
                tensor.data.len()
            )));
        }
        let token = i32::from_le_bytes([
            tensor.data[0], tensor.data[1], tensor.data[2], tensor.data[3],
        ]);
        Ok(token)
    }

    fn recv_hidden_from_upstream(&mut self) -> EngineResult<(Vec<f32>, [usize; 3])> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        let (tensor, _) = self
            .block_on(async move {
                let mut guard = upstream.lock().await;
                guard.recv().await
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let shape = [
            tensor.shape[0] as usize,
            tensor.shape[1] as usize,
            tensor.shape[2] as usize,
        ];
        let floats = match tensor.dtype {
            WireDType::F32 => tensor
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            WireDType::F16 => f16_bytes_to_f32(&tensor.data),
            other => {
                return Err(EngineError::Backend(format!(
                    "unexpected upstream dtype {other:?}"
                )))
            }
        };
        Ok((floats, shape))
    }

    fn send_token_to_upstream(&mut self, token: i32) -> EngineResult<()> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        let bytes = token.to_le_bytes().to_vec();
        let tensor = WireTensor::new(WireDType::I32, [1, 1, 1], bytes);
        self.block_on(async move {
            let mut guard = upstream.lock().await;
            guard.send(&tensor).await
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    fn step_first(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.active.is_none() && !self.pending.is_empty() {
            let task = self.pending.remove(0);
            let tok = self
                .tokenizer
                .clone()
                .ok_or_else(|| EngineError::Backend("first stage requires tokenizer".into()))?;
            let enc = tok
                .encode(task.prompt.clone(), false)
                .map_err(|e| EngineError::Backend(format!("tokenizer encode: {e}")))?;
            let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            self.runtime.reset_state().map_err(map_ov_err)?;
            self.position = 0;
            info!(
                task = %task.task_id,
                prompt_tokens = prompt_ids.len(),
                "task active (ov-runtime)"
            );
            self.active = Some(ActiveTask {
                task,
                prompt_ids,
                generated: Vec::new(),
                last_text: String::new(),
                prefilled: false,
                last_token: 0,
            });
        }

        let active = match self.active.as_mut() {
            Some(a) => a,
            None => return Ok(Vec::new()),
        };

        let input_ids: Vec<i64> = if !active.prefilled {
            active.prefilled = true;
            active.prompt_ids.clone()
        } else {
            vec![active.last_token as i64]
        };

        let position = self.position;
        let (out, shape) = self.run_first(&input_ids, position)?;
        self.position += input_ids.len() as i64;

        // Resolve next_token: if 1-stage IR also produced logits; else send hs
        // downstream and recv next_token back.
        let next_token = if self.spec.is_first_stage && self.spec.is_last_stage {
            // Single-stage: out is logits [1, seq_len, vocab]
            let vocab = shape[shape.len() - 1];
            argmax_last_row(&out, vocab)
        } else {
            let s3 = if shape.len() == 3 {
                [shape[0], shape[1], shape[2]]
            } else {
                [1, shape[0], shape[1]]
            };
            self.send_hidden_downstream(&out, s3)?;
            self.recv_token_from_downstream()?
        };

        // Decode delta + check stop.
        let active = self.active.as_mut().unwrap();
        active.last_token = next_token;
        active.generated.push(next_token);

        let tok = self.tokenizer.as_ref().unwrap();
        let all_ids: Vec<u32> = active.generated.iter().map(|&t| t as u32).collect();
        let full_text = tok
            .decode(&all_ids, true)
            .map_err(|e| EngineError::Backend(format!("tokenizer decode: {e}")))?;
        // Use strip_prefix instead of byte-slice indexing — `last_text`
        // is not always a clean byte-prefix of `full_text` (BPE can
        // emit a partial UTF-8 sequence on token N and complete the
        // glyph on token N+1, in which case the prefix bytes change).
        // Slicing past a UTF-8 boundary panics.
        let delta = full_text
            .strip_prefix(active.last_text.as_str())
            .unwrap_or(&full_text)
            .to_string();
        active.last_text = full_text;

        let max_tokens = active.task.max_tokens.max(1) as usize;
        let is_eos = self
            .eos_token_id
            .map(|eos| next_token as u32 == eos)
            .unwrap_or(false);
        let is_final = active.generated.len() >= max_tokens || is_eos;

        let task_id = active.task.task_id.clone();
        let chunk = if is_final {
            Chunk {
                task_id: task_id.clone(),
                token_id: next_token as i64,
                text: delta,
                is_final: true,
                logprobs: None,
            }
        } else {
            Chunk::token(task_id.clone(), next_token as i64, delta)
        };

        if is_final {
            info!(
                task = %task_id,
                tokens = active.generated.len(),
                "ov-runtime task done"
            );
            self.active = None;
        }

        Ok(vec![(task_id, chunk)])
    }

    fn step_last(&mut self) -> EngineResult<()> {
        let (hidden, shape) = self.recv_hidden_from_upstream()?;
        if shape[1] > 1 {
            self.runtime.reset_state().map_err(map_ov_err)?;
            self.position = 0;
        }
        let (out, out_shape) = self.run_relay(&hidden, shape, self.position)?;
        self.position += shape[1] as i64;
        let vocab = out_shape[out_shape.len() - 1];
        let next = argmax_last_row(&out, vocab);
        self.send_token_to_upstream(next)?;
        Ok(())
    }

    fn step_middle(&mut self) -> EngineResult<()> {
        let (hidden, shape) = self.recv_hidden_from_upstream()?;
        if shape[1] > 1 {
            self.runtime.reset_state().map_err(map_ov_err)?;
            self.position = 0;
        }
        let (out, _) = self.run_relay(&hidden, shape, self.position)?;
        self.position += shape[1] as i64;
        let s3 = [shape[0], shape[1], shape[2]];
        self.send_hidden_downstream(&out, s3)?;
        let token = self.recv_token_from_downstream()?;
        self.send_token_to_upstream(token)?;
        Ok(())
    }
}

impl Engine for OvRuntimeEngine {
    fn warmup(&mut self) {
        if !(self.spec.is_first_stage) {
            info!("ov-runtime warmup skipped on non-first stage");
            return;
        }
        let tok = match self.tokenizer.clone() {
            Some(t) => t,
            None => {
                warn!("ov-runtime warmup skipped: no tokenizer");
                return;
            }
        };
        let enc = match tok.encode("Hi", false) {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "ov-runtime warmup tokenize failed");
                return;
            }
        };
        let ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
        match self.run_first(&ids, 0) {
            Ok(_) => {
                let _ = self.runtime.reset_state();
                self.position = 0;
                info!("ov-runtime warmup ok");
            }
            Err(e) => warn!(error = %e, "ov-runtime warmup failed"),
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if !self.spec.is_first_stage {
            warn!("ov-runtime submit() ignored on non-first stage");
            return Err(EngineError::Backend(
                "non-first stage does not accept tasks directly".into(),
            ));
        }
        if self.pending.iter().any(|t| t.task_id == task.task_id)
            || self
                .active
                .as_ref()
                .is_some_and(|a| a.task.task_id == task.task_id)
        {
            return Ok(());
        }
        if self.pending.len() >= crate::dist_spec::MAX_PENDING_TASKS {
            warn!(
                queued = self.pending.len(),
                cap = crate::dist_spec::MAX_PENDING_TASKS,
                "ov-runtime: pending queue at cap; rejecting task"
            );
            return Err(EngineError::QueueFull {
                queued: self.pending.len(),
                cap: crate::dist_spec::MAX_PENDING_TASKS,
            });
        }
        self.pending.push(task);
        Ok(())
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        let result: EngineResult<Vec<(TaskId, Chunk)>> = if self.spec.is_first_stage {
            self.step_first()
        } else if self.spec.is_last_stage {
            self.step_last().map(|_| Vec::new())
        } else {
            self.step_middle().map(|_| Vec::new())
        };
        match result {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "ov-runtime step failed");
                Vec::new()
            }
        }
    }
}

// -------- Builder --------

#[derive(Default)]
pub struct OvRuntimeBuilder {
    pub pipeline_dir: PathBuf,
    pub rank: u32,
    pub total: u32,
    pub device: String,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    runtime: Option<OvRuntime>,
    spec: Option<ShardSpec>,
    rotary: Option<Rotary>,
    hidden_size: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    eos_token_id: Option<u32>,
    input_names: Vec<String>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    listen_host: String,
    listen_port: Option<u16>,
}

impl OvRuntimeBuilder {
    pub fn new(pipeline_dir: impl Into<PathBuf>, rank: u32, total: u32, device: impl Into<String>) -> Self {
        Self {
            pipeline_dir: pipeline_dir.into(),
            rank,
            total,
            device: device.into(),
            listen_host: "0.0.0.0".into(),
            ..Self::default()
        }
    }

    pub fn with_cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }
    pub fn with_kv_cache_precision(mut self, prec: impl Into<String>) -> Self {
        self.kv_cache_precision = Some(prec.into());
        self
    }
    pub fn with_dyn_quant_group(mut self, group: impl Into<String>) -> Self {
        self.dyn_quant_group = Some(group.into());
        self
    }

    fn plugin(&self) -> PluginConfig {
        let mut p = PluginConfig::new();
        if let Some(d) = &self.cache_dir {
            p = p.with("CACHE_DIR", d);
        }
        if let Some(p2) = &self.kv_cache_precision {
            p = p.with("KV_CACHE_PRECISION", p2);
        }
        if let Some(g) = &self.dyn_quant_group {
            p = p.with("DYNAMIC_QUANTIZATION_GROUP_SIZE", g);
        }
        p
    }
}

#[async_trait]
impl Builder for OvRuntimeBuilder {
    fn configure_listen(&mut self, host: &str, port: u16) {
        self.listen_host = host.to_string();
        self.listen_port = Some(port);
    }

    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        // First, bind upstream listener (so downstream can connect to us
        // before we connect downstream). Mirrors the Python order.
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
            let srv = srv.clone();
            srv.lock()
                .await
                .accept()
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        let mut events = Vec::new();
        events.push(LoadProgress::message(format!(
            "reading {}",
            self.pipeline_dir.display()
        )));

        let pipeline_cfg = read_pipeline_config(&self.pipeline_dir)?;
        if pipeline_cfg.num_stages != self.total {
            return Err(EngineError::ShardRejected(format!(
                "--total ({}) does not match pipeline_config num_stages ({})",
                self.total, pipeline_cfg.num_stages
            )));
        }
        let stage_dir = self.pipeline_dir.join(format!("stage_{}", self.rank));
        let stage_cfg = read_stage_config(&stage_dir)?;

        let is_first = stage_cfg.has_embed;
        let is_last = stage_cfg.has_head;
        let spec = ShardSpec {
            model_id: pipeline_cfg.model_id.clone(),
            layer_start: stage_cfg.layer_start,
            layer_end: stage_cfg.layer_end,
            total_layers: pipeline_cfg.num_layers,
            device: self.device.clone(),
            is_first_stage: is_first,
            is_last_stage: is_last,
            tp_size: 1,
            tp_rank: 0,
        };
        self.hidden_size = pipeline_cfg.hidden_size as usize;
        self.spec = Some(spec);

        events.push(LoadProgress::message(format!(
            "compiling stage {} on {}",
            self.rank, self.device
        )));
        let plugin = self.plugin();
        let xml_path = stage_dir.join("openvino_model.xml");
        let runtime = OvRuntime::compile(
            xml_path.to_str().unwrap_or_default(),
            &self.device,
            &plugin,
        )
        .map_err(map_ov_err)?;
        self.input_names = runtime.input_names().map_err(map_ov_err)?;
        self.runtime = Some(runtime);

        events.push(LoadProgress::message(
            "loading rotary + tokenizer".to_string(),
        ));

        // Rotary from the model's HF config.json. Look in the pipeline
        // tokenizer dir first (rainier exports include config.json there);
        // fall back to the HF cache via env, else error.
        let tokenizer_dir = self.pipeline_dir.join("tokenizer");
        let cfg = match load_model_config(&tokenizer_dir) {
            Ok(c) => c,
            Err(e1) => {
                let alt = self.pipeline_dir.clone();
                load_model_config(&alt).map_err(|e2| {
                    EngineError::InvalidConfig(format!(
                        "config.json not in {tokenizer_dir:?} ({e1}) or {alt:?} ({e2})"
                    ))
                })?
            }
        };
        let rotary = Rotary::from_config(&cfg)
            .map_err(|e| EngineError::InvalidConfig(format!("rotary: {e}")))?;
        self.rotary = Some(rotary);

        if is_first {
            let tok_path = tokenizer_dir.join("tokenizer.json");
            if tok_path.exists() {
                let tok = Tokenizer::from_file(&tok_path)
                    .map_err(|e| EngineError::Backend(format!("tokenizer load: {e}")))?;
                self.tokenizer = Some(Arc::new(tok));
                self.eos_token_id = lookup_eos(&tokenizer_dir).or_else(|| lookup_eos(&self.pipeline_dir));
                events.push(LoadProgress::message(format!(
                    "tokenizer loaded; eos_token_id={:?}",
                    self.eos_token_id
                )));
            } else {
                events.push(LoadProgress::message(format!(
                    "warning: no tokenizer.json at {tok_path:?}; first-stage tokenization will fail"
                )));
            }
        }

        events.push(LoadProgress::ready());
        Ok(Box::pin(stream::iter(events)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let runtime = self.runtime.ok_or(EngineError::NotLoaded)?;
        let spec = self.spec.ok_or(EngineError::NotLoaded)?;
        let rotary = self.rotary.ok_or(EngineError::NotLoaded)?;

        Ok(Box::new(OvRuntimeEngine {
            spec,
            runtime,
            rotary,
            hidden_size: self.hidden_size,
            tokenizer: self.tokenizer,
            eos_token_id: self.eos_token_id,
            upstream: self.upstream,
            downstream: self.downstream,
            runtime_handle: tokio::runtime::Handle::current(),
            position: 0,
            input_names: self.input_names,
            pending: Vec::new(),
            active: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tahoma_types::PeerLayout;

    #[tokio::test]
    async fn rejects_missing_pipeline_config() {
        let mut b = OvRuntimeBuilder::new("/non/existent", 0, 1, "CPU");
        let res = b.load(ShardSpec::single_stage("m", "CPU")).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn build_before_load_errors() {
        let b = Box::new(OvRuntimeBuilder::new("/x", 0, 1, "CPU"));
        assert!(matches!(b.build(), Err(EngineError::NotLoaded)));
    }

    #[tokio::test]
    async fn connect_no_peers_is_noop_for_single_stage() {
        let mut b = OvRuntimeBuilder::new("/x", 0, 1, "CPU");
        b.connect(PeerLayout::single_stage()).await.unwrap();
    }
}
