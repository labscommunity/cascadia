//! Multi-stage OpenVINO Runtime engine.
//!
//! Rust port of `cascadia/worker/engines/openvino/ov_runtime.py`. Loads
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
//! Wire format between stages: hidden_states f16. Stateful shards have each
//! stage track its own absolute-position counter (computing cos/sin locally,
//! no position metadata on the wire); the counter resets when an activation
//! with seq_len > 1 arrives (a prefill signal for relay/last stages).
//!
//! Stateless static-shape (NPU) shards (`stage_config.stateful == false`)
//! instead drive a host-side bounded KV ring per stage (see `StaticKv`).
//! Because static shards are seq=1, the seq>1 prefill signal is unavailable,
//! so the first stage carries the absolute `position` as an 8-byte prefix on
//! each activation; downstream stages reset their ring at position 0 and
//! derive the visible-past count from it, keeping every stage's ring in
//! lockstep. This path works single- or multi-stage (pipeline-parallel NPU).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{
    DType as ShimDType, Error as OvError, PluginConfig, Runtime as OvRuntime,
};
use cascadia_transport::{
    ActivationClient, ActivationServer, DType as WireDType, Tensor as WireTensor, MAX_RANK,
};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::rotary::{load_model_config, Rotary};
use crate::warn_limit::{StepWarn, StepWarnLimiter};

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

fn default_true() -> bool {
    true
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
    /// NPU (`--target npu`) export: KV is stateless (explicit past_kv-in /
    /// present-out) and all shapes are static. Old/standard shards omit the
    /// field and are stateful (default true). When false, the engine drives
    /// the static-KV path (host-side bounded ring) instead of OV state.
    #[serde(default = "default_true")]
    stateful: bool,
    #[serde(default)]
    static_seq: Option<u32>,
    #[serde(default)]
    static_context: Option<u32>,
    #[serde(default)]
    num_kv_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
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

/// Return ALL EOS token ids configured for the model. Many recent
/// instruct models (Llama 3.x, Qwen3, Gemma 2) list multiple EOS:
/// `[<|end_of_text|>, <|eom_id|>, <|eot_id|>]` for Llama 3.3, for
/// example. The chat-end token is usually the LAST entry, not the
/// first — keeping only `ids.first()` made the model run past
/// `<|eot_id|>` and start hallucinating fake "assistant\n\n…" turns.
/// Caller stops generation if the next token matches any of these.
fn lookup_eos(model_dir: &Path) -> Vec<u32> {
    for fname in ["generation_config.json", "config.json"] {
        let p = model_dir.join(fname);
        if let Ok(bytes) = std::fs::read(&p) {
            if let Ok(g) = serde_json::from_slice::<GenerationCfg>(&bytes) {
                return match g.eos_token_id {
                    Some(EosId::One(id)) => vec![id],
                    Some(EosId::Many(ids)) => ids,
                    None => Vec::new(),
                };
            }
        }
    }
    Vec::new()
}

// -------- helpers: bytes <-> typed slices --------

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
        OvError::Stub => {
            EngineError::Backend("openvino shim built without --features openvino".into())
        }
        OvError::Utf8(s) => EngineError::InvalidConfig(s),
        OvError::Native(s) => EngineError::Backend(s),
    }
}

/// Decode a float output port's raw bytes to f32, by its reported dtype.
/// Shared by every output-reading path (run_first / run_relay / static).
fn bytes_to_f32(dtype: ShimDType, bytes: &[u8]) -> EngineResult<Vec<f32>> {
    match dtype {
        ShimDType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        ShimDType::F16 => Ok(f16_bytes_to_f32(bytes)),
        other => Err(EngineError::Backend(format!(
            "unexpected float output dtype {other:?}"
        ))),
    }
}

/// Pick the next token from a logits output, guarding the shape so a
/// rank-0 / zero-width / short logits buffer returns a clear error instead
/// of panicking with a slice/underflow.
fn argmax_logits(logits: &[f32], shape: &[usize]) -> EngineResult<i32> {
    let vocab = match shape.last() {
        Some(&v) if v > 0 => v,
        _ => {
            return Err(EngineError::Backend(format!(
                "logits output has empty/zero last dim: shape={shape:?}"
            )))
        }
    };
    if vocab > logits.len() {
        return Err(EngineError::Backend(format!(
            "logits len {} < vocab {vocab} (shape={shape:?})",
            logits.len()
        )));
    }
    Ok(argmax_last_row(logits, vocab))
}

/// Normalize an output shape to a 3D `[batch, seq, hidden]` for the wire.
fn to_shape3(shape: &[usize]) -> [usize; 3] {
    match shape.len() {
        3 => [shape[0], shape[1], shape[2]],
        2 => [1, shape[0], shape[1]],
        _ => [1, 1, shape.last().copied().unwrap_or(0)],
    }
}

/// Encode the absolute position as its own framed wire tensor (I64 `[1,1,1]`).
/// The static (NPU) path sends this immediately before each hidden activation
/// so relay stages reset/align their KV ring; the transport requires
/// `payload_len == shape*dtype`, so position cannot be packed into the hidden
/// tensor (and MAX_RANK=3 leaves no spare shape slot). Paired with
/// `decode_wire_position` — keep the two in sync.
fn encode_wire_position(position: i64) -> WireTensor {
    WireTensor::new(WireDType::I64, [1, 1, 1], position.to_le_bytes().to_vec())
}

/// Decode + strictly validate a wire position frame. Must be I64 with exactly
/// 8 payload bytes; anything else (a desynced stream, or a stateful peer that
/// sent a hidden tensor where a position was expected) is a hard error rather
/// than a silently zero-padded wrong position.
fn decode_wire_position(t: &WireTensor) -> EngineResult<i64> {
    if t.dtype != WireDType::I64 || t.data.len() != 8 {
        return Err(EngineError::Backend(format!(
            "expected an I64 8-byte position frame, got dtype={:?} len={} — likely a \
             stateful/static pipeline mismatch or a desynced activation stream",
            t.dtype,
            t.data.len()
        )));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&t.data);
    Ok(i64::from_le_bytes(b))
}

// -------- static-KV (NPU) state --------

/// Per-layer port wiring for a stateless static-shape (NPU) shard.
struct StaticKvLayer {
    key_in: String,
    val_in: String,
    key_out: usize,
    val_out: usize,
}

/// Host-side bounded KV ring for the stateless static-shape (NPU) path.
/// The exported IR takes explicit `past_key_values.*` (length `past_len`) +
/// `attention_mask` (length `context = past_len + 1`) + `position_ids`, and
/// returns its primary output (logits for the head stage, hidden_states
/// otherwise) at output index 0, plus `present.*` (length `context`). We hold
/// the KV for the most recent `valid` real tokens left-aligned in
/// `past_len`-slot buffers, feed them each step, and absorb the new token's KV
/// from `present[past_len]`. One ring per stage; in a multi-stage pipeline
/// every stage runs its own ring over its own layers, kept in lockstep by the
/// absolute `position` the first stage carries with each activation.
struct StaticKv {
    past_len: usize,
    context: usize,
    kv_heads: usize,
    head_dim: usize,
    elem_bytes: usize,
    kv_dtype: ShimDType,
    /// Resolved primary input port names (cached at build so the per-token
    /// decode loop does no HashMap lookups / string allocs). `ids_in` is set
    /// for the embed stage, `hidden_in` for relay/head stages.
    ids_in: Option<String>,
    hidden_in: Option<String>,
    attn_in: String,
    pos_in: String,
    layers: Vec<StaticKvLayer>,
    key_buf: Vec<Vec<u8>>, // [layer] = kv_heads * past_len * head_dim * elem_bytes
    val_buf: Vec<Vec<u8>>,
    valid: usize, // number of real past tokens currently in the ring
    /// Reusable attention_mask byte buffer (i64 LE, length context*8), rewritten
    /// in place each token to avoid a per-token allocation.
    mask_bytes: Vec<u8>,
}

impl StaticKv {
    fn reset(&mut self) {
        for b in self.key_buf.iter_mut().chain(self.val_buf.iter_mut()) {
            b.iter_mut().for_each(|x| *x = 0);
        }
        self.valid = 0;
    }

    /// Expected byte length of one layer's `present.*` output:
    /// `kv_heads * context * head_dim * elem_bytes`. Used to validate the IR's
    /// output shape/dtype before `absorb_layer` slices it.
    fn present_layer_bytes(&self) -> usize {
        self.kv_heads * self.context * self.head_dim * self.elem_bytes
    }

    /// Set how many real past tokens are visible for a token at absolute
    /// `position`, resetting the ring at the start of a new sequence
    /// (`position == 0`). The window is bounded by `past_len`.
    fn begin_token(&mut self, position: usize) {
        if position == 0 {
            self.reset();
        }
        self.valid = position.min(self.past_len);
    }

    /// Rewrite `mask_bytes` (i64 LE) for the current `valid`: 1 for the `valid`
    /// real past slots (left-aligned) + the current token (slot `past_len`), 0
    /// for the padding past slots in between. Reuses the buffer's capacity.
    fn write_mask_bytes(&mut self) {
        self.mask_bytes.clear();
        self.mask_bytes.resize(self.context * 8, 0);
        for i in 0..self.context {
            let v: i64 = if i < self.valid || i == self.past_len {
                1
            } else {
                0
            };
            self.mask_bytes[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Copy the new token's K/V (slot `past_len` of `present`) into the
    /// layer's ring buffer, appending at `valid` or sliding the window when
    /// full. Selects `key_buf[li]` or `val_buf[li]` internally so callers can
    /// hold `&mut self` without aliasing the buffer field.
    fn absorb_layer(&mut self, li: usize, is_value: bool, present: &[u8]) {
        // Read scalar fields into locals before borrowing the buffer field,
        // so the mutable buffer borrow stays disjoint from these reads.
        let slot = self.head_dim * self.elem_bytes;
        let present_row = self.context * slot; // per-head stride in present
        let buf_row = self.past_len * slot; // per-head stride in the ring
        let full = self.valid >= self.past_len;
        let kv_heads = self.kv_heads;
        let past_len = self.past_len;
        let valid = self.valid;
        let buf: &mut [u8] = if is_value {
            &mut self.val_buf[li]
        } else {
            &mut self.key_buf[li]
        };
        for h in 0..kv_heads {
            let src = h * present_row + past_len * slot;
            let new = &present[src..src + slot];
            let base = h * buf_row;
            if full {
                buf.copy_within(base + slot..base + buf_row, base); // drop oldest
                let dst = base + (past_len - 1) * slot;
                buf[dst..dst + slot].copy_from_slice(new);
            } else {
                let dst = base + valid * slot;
                buf[dst..dst + slot].copy_from_slice(new);
            }
        }
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
    /// Wall-clock when the task became active. Used to compute the
    /// final tok/s the engine prints in its `task done` log line.
    started: std::time::Instant,
    /// Cumulative time inside `run_first` (stage_0 compute + read).
    t_alpha_compute: std::time::Duration,
    /// Cumulative time inside `send_hidden_downstream` +
    /// `recv_token_from_downstream` — i.e. wire send + charlie wait + recv.
    t_wire: std::time::Duration,
}

pub struct OvRuntimeEngine {
    spec: ShardSpec,
    runtime: OvRuntime,
    rotary: Rotary,
    hidden_size: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    /// All EOS token ids configured for the model. Generation stops on
    /// the first token that matches ANY of these. See `lookup_eos`.
    eos_token_ids: Vec<u32>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    runtime_handle: tokio::runtime::Handle,
    position: i64,
    input_names: Vec<String>,
    /// Map of canonical name (e.g. "input_ids", "attention_mask") to the
    /// IR's primary port name. Resolved at engine build time via the
    /// alias lookup (the IR's primary name is sometimes an internal
    /// node id, not the canonical name). Empty for v3 IRs.
    canonical_inputs: std::collections::HashMap<String, String>,
    pending: Vec<GenerationTask>,
    active: Option<ActiveTask>,
    /// Set for stateless static-shape (NPU) shards; drives the host-side
    /// bounded-KV decode path instead of OV internal state.
    static_kv: Option<StaticKv>,
    step_warn: StepWarnLimiter,
}

impl OvRuntimeEngine {
    /// True if the loaded IR uses v5+ canonical inputs (attention_mask +
    /// position_ids) instead of legacy v3 (cos + sin). Determined from
    /// the alias-resolved canonical_inputs map populated at build time.
    fn is_v5_layout(&self) -> bool {
        self.canonical_inputs.contains_key("attention_mask")
            && self.canonical_inputs.contains_key("position_ids")
    }

    /// Resolve a canonical input name to the IR's primary port name.
    fn input_named(&self, canonical: &str) -> Option<&str> {
        self.canonical_inputs.get(canonical).map(|s| s.as_str())
    }

    fn build_feed_first(&mut self, input_ids: &[i64], position: i64) -> EngineResult<()> {
        if self.is_v5_layout() {
            return self.build_feed_first_v5(input_ids, position);
        }
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

    fn build_feed_first_v5(&mut self, input_ids: &[i64], position: i64) -> EngineResult<()> {
        let seq_len = input_ids.len();
        let total = position as usize + seq_len;
        let attn = vec![1i64; total];
        let pos: Vec<i64> = (position..position + seq_len as i64).collect();

        let in_ids = self
            .input_named("input_ids")
            .ok_or_else(|| EngineError::Backend("v5 IR missing input_ids".into()))?
            .to_string();
        let in_attn = self
            .input_named("attention_mask")
            .ok_or_else(|| EngineError::Backend("v5 IR missing attention_mask".into()))?
            .to_string();
        let in_pos = self
            .input_named("position_ids")
            .ok_or_else(|| EngineError::Backend("v5 IR missing position_ids".into()))?
            .to_string();
        let in_beam = self.input_named("beam_idx").map(|s| s.to_string());

        self.runtime
            .set_input(
                &in_ids,
                ShimDType::I64,
                &[1, seq_len],
                &i64_to_bytes(input_ids),
            )
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_attn, ShimDType::I64, &[1, total], &i64_to_bytes(&attn))
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_pos, ShimDType::I64, &[1, seq_len], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        if let Some(beam) = in_beam {
            self.runtime
                .set_input(&beam, ShimDType::I32, &[1], &0i32.to_le_bytes())
                .map_err(map_ov_err)?;
        }
        Ok(())
    }

    fn build_feed_relay(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<()> {
        if self.is_v5_layout() {
            return self.build_feed_relay_v5(hidden, shape, position);
        }
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

    fn build_feed_relay_v5(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<()> {
        let seq_len = shape[1];
        let total = position as usize + seq_len;
        let attn = vec![1i64; total];
        let pos: Vec<i64> = (position..position + seq_len as i64).collect();

        let in_hs = self
            .input_named("hidden_states")
            .ok_or_else(|| EngineError::Backend("v5 IR missing hidden_states".into()))?
            .to_string();
        let in_attn = self
            .input_named("attention_mask")
            .ok_or_else(|| EngineError::Backend("v5 IR missing attention_mask".into()))?
            .to_string();
        let in_pos = self
            .input_named("position_ids")
            .ok_or_else(|| EngineError::Backend("v5 IR missing position_ids".into()))?
            .to_string();
        let in_beam = self.input_named("beam_idx").map(|s| s.to_string());

        let hs_bytes = f32_to_f16_bytes(hidden);
        self.runtime
            .set_input(&in_hs, ShimDType::F16, &shape, &hs_bytes)
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_attn, ShimDType::I64, &[1, total], &i64_to_bytes(&attn))
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_pos, ShimDType::I64, &[1, seq_len], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        if let Some(beam) = in_beam {
            self.runtime
                .set_input(&beam, ShimDType::I32, &[1], &0i32.to_le_bytes())
                .map_err(map_ov_err)?;
        }
        Ok(())
    }

    fn run_first(
        &mut self,
        input_ids: &[i64],
        position: i64,
    ) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        // Stateless static (NPU): feed one token id + the host-side KV ring.
        // The caller loops one token at a time (static_seq == 1).
        if self.static_kv.is_some() {
            if input_ids.len() != 1 {
                return Err(EngineError::Backend(format!(
                    "static shard processes one token per step, got {}",
                    input_ids.len()
                )));
            }
            let (dtype, shape, bytes) = self.static_infer(
                false,
                ShimDType::I64,
                &[1, 1],
                &i64_to_bytes(input_ids),
                position,
            )?;
            return Ok((bytes_to_f32(dtype, &bytes)?, shape));
        }
        self.build_feed_first(input_ids, position)?;
        self.runtime.infer().map_err(map_ov_err)?;
        let (dtype, shape, bytes) = self.runtime.output(0).map_err(map_ov_err)?;
        Ok((bytes_to_f32(dtype, &bytes)?, shape))
    }

    fn run_relay(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        // Stateless static (NPU): feed the upstream hidden state + the
        // host-side KV ring for this stage's layers.
        if self.static_kv.is_some() {
            let hs_bytes = f32_to_f16_bytes(hidden);
            let (dtype, out_shape, bytes) =
                self.static_infer(true, ShimDType::F16, &shape, &hs_bytes, position)?;
            return Ok((bytes_to_f32(dtype, &bytes)?, out_shape));
        }
        self.build_feed_relay(hidden, shape, position)?;
        self.runtime.infer().map_err(map_ov_err)?;
        let (dtype, out_shape, bytes) = self.runtime.output(0).map_err(map_ov_err)?;
        Ok((bytes_to_f32(dtype, &bytes)?, out_shape))
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

    fn send_hidden_downstream(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let mut wire_shape = [1u32; MAX_RANK];
        for (i, d) in shape.iter().enumerate().take(MAX_RANK) {
            wire_shape[i] = *d as u32;
        }
        let hid = WireTensor::new(WireDType::F16, wire_shape, f32_to_f16_bytes(hidden));
        // Static (NPU) shards need the absolute position downstream so each
        // stage can reset its ring at position 0 and align the visible-past
        // count. The wire shape has only MAX_RANK=3 dims (all used by
        // [1,1,hidden]) and the transport requires payload_len == shape*dtype,
        // so we can't pack it into the hidden tensor — send it as its own
        // framed I64 tensor first. recv_hidden_from_upstream mirrors the order.
        let pos = if self.static_kv.is_some() {
            Some(encode_wire_position(position))
        } else {
            None
        };
        self.block_on(async move {
            let mut guard = downstream.lock().await;
            if let Some(p) = pos {
                guard.send(&p).await?;
            }
            guard.send(&hid).await
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
            tensor.data[0],
            tensor.data[1],
            tensor.data[2],
            tensor.data[3],
        ]);
        Ok(token)
    }

    fn recv_hidden_from_upstream(&mut self) -> EngineResult<(Vec<f32>, [usize; 3], Option<i64>)> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        // Static (NPU) shards send a leading I64 position tensor before the
        // hidden activation (see send_hidden_downstream). Each frame's payload
        // must match its shape*dtype, so we recv two separate tensors here.
        let want_pos = self.static_kv.is_some();
        let (pos_tensor, tensor) = self
            .block_on(async move {
                let mut guard = upstream.lock().await;
                let pos_tensor = if want_pos {
                    Some(guard.recv().await?.0)
                } else {
                    None
                };
                let (t, _) = guard.recv().await?;
                Ok::<_, cascadia_transport::TransportError>((pos_tensor, t))
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        // Decode + strictly validate the position frame outside the transport
        // closure (so a bad frame yields a clear EngineError, not a desync).
        let position = match pos_tensor {
            Some(p) => Some(decode_wire_position(&p)?),
            None => None,
        };
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
        Ok((floats, shape, position))
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
            if self.static_kv.is_none() {
                self.runtime.reset_state().map_err(map_ov_err)?;
            }
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
                started: std::time::Instant::now(),
                t_alpha_compute: std::time::Duration::ZERO,
                t_wire: std::time::Duration::ZERO,
            });
        }

        // Run the actual step in an inner method so we can catch any
        // error and clear engine state. Without this, a single
        // transport / OV-runtime failure leaves `self.active = Some(...)`
        // and `self.position` stale; the next submit() pushes to pending
        // but step_first's "if active.is_none()" gate is false, so the
        // new task is never picked up — runner.poll_next sees empty
        // steps and the API returns `data: [DONE]` with zero chunks.
        let mut res = self.step_first_body();
        if let Err(e) = res {
            // DEBUG, not WARN: the outer step() emits the rate-limited
            // WARN for the same error (StepWarnLimiter); a second
            // unconditional WARN here would bypass the limiter and
            // double-log every first-stage failure.
            tracing::debug!(
                error = %e,
                "step_first failed; clearing active + reset_state so next \
                 task starts fresh (downstream socket may still be dead)"
            );
            // Attribute to the failed task (if one was active) before we
            // null it, so the runner routes the failure to that task's
            // stream instead of ending whichever stream observes the Err.
            let failed = self.active.as_ref().map(|a| a.task.task_id.clone());
            self.active = None;
            let _ = self.runtime.reset_state();
            if let Some(sk) = self.static_kv.as_mut() {
                sk.reset();
            }
            self.position = 0;
            res = Err(match failed {
                Some(id) => e.for_task(id),
                None => e,
            });
        }
        res
    }

    fn step_first_body(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.active.is_none() {
            return Ok(Vec::new());
        }
        let (prefill, tokens) = {
            let a = self.active.as_mut().unwrap();
            if !a.prefilled {
                a.prefilled = true;
                (true, a.prompt_ids.clone())
            } else {
                (false, vec![a.last_token as i64])
            }
        };
        let single_stage = self.spec.is_first_stage && self.spec.is_last_stage;

        // A generation request must carry at least one prompt token; an empty
        // prompt would otherwise emit a fabricated token with no inference (and
        // on the static path skip the position-0 ring reset).
        if prefill && tokens.is_empty() {
            return Err(EngineError::Backend(
                "empty prompt: no tokens to prefill".into(),
            ));
        }

        let next_token = if self.static_kv.is_some() {
            // Static (NPU): one token per inference (static_seq == 1). Prefill
            // round-trips the whole pipeline once per prompt token so every
            // stage's KV ring advances in lockstep; the token produced from
            // the final prompt token is the first generated token.
            if prefill {
                self.position = 0;
                if let Some(sk) = self.static_kv.as_ref() {
                    if tokens.len() > sk.past_len {
                        warn!(
                            prompt_tokens = tokens.len(),
                            past_len = sk.past_len,
                            "prompt exceeds the static KV window (static_context-1); earliest \
                             tokens will be evicted — attention degrades to a sliding window"
                        );
                    }
                }
            }
            let mut nt = 0i32;
            for &t in &tokens {
                let position = self.position;
                let ts = std::time::Instant::now();
                let (out, shape) = self.run_first(&[t], position)?;
                let alpha = ts.elapsed();
                self.position += 1;
                let (token, wire) =
                    self.resolve_next_token(&out, &shape, single_stage, position)?;
                nt = token;
                if let Some(a) = self.active.as_mut() {
                    a.t_alpha_compute += alpha;
                    a.t_wire += wire;
                }
            }
            nt
        } else {
            // Stateful: the whole prompt (prefill) or one decode token in a
            // single multi-token inference; the IR keeps KV internally.
            let position = self.position;
            let ts = std::time::Instant::now();
            let (out, shape) = self.run_first(&tokens, position)?;
            let alpha = ts.elapsed();
            self.position += tokens.len() as i64;
            let (nt, wire) = self.resolve_next_token(&out, &shape, single_stage, position)?;
            if let Some(a) = self.active.as_mut() {
                a.t_alpha_compute += alpha;
                a.t_wire += wire;
            }
            nt
        };

        self.emit_token(next_token)
    }

    /// From a first-stage output, produce the next token: argmax locally when
    /// this is the only stage, otherwise forward the hidden state downstream
    /// and await the token from the pipeline tail. Returns the token + the
    /// wire round-trip time (zero for single-stage).
    fn resolve_next_token(
        &mut self,
        out: &[f32],
        shape: &[usize],
        single_stage: bool,
        position: i64,
    ) -> EngineResult<(i32, std::time::Duration)> {
        if single_stage {
            Ok((argmax_logits(out, shape)?, std::time::Duration::ZERO))
        } else {
            let s3 = to_shape3(shape);
            let ts = std::time::Instant::now();
            self.send_hidden_downstream(out, s3, position)?;
            let token = self.recv_token_from_downstream()?;
            Ok((token, ts.elapsed()))
        }
    }

    /// Decode the delta text for `next_token`, append it to the active task,
    /// check stop conditions, and build the streamed chunk. Shared by the
    /// static and stateful first-stage paths.
    fn emit_token(&mut self, next_token: i32) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| EngineError::Backend("emit_token with no active task".into()))?;
        active.last_token = next_token;
        active.generated.push(next_token);

        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Backend("first stage requires tokenizer".into()))?;
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
        let is_eos = self.eos_token_ids.contains(&(next_token as u32));
        let is_final = active.generated.len() >= max_tokens || is_eos;

        let task_id = active.task.task_id.clone();
        let chunk = if is_final {
            Chunk {
                task_id: task_id.clone(),
                token_id: next_token as i64,
                text: delta,
                is_final: true,
                logprobs: None,
                n_tokens: None,
                prompt_tokens: None,
                error: None,
                // ov-runtime doesn't yet distinguish length vs stop here; the
                // API falls back to "stop" (unchanged behavior). #14 follow-up.
                finish_reason: None,
            }
        } else {
            Chunk::token(task_id.clone(), next_token as i64, delta)
        };

        if is_final {
            let elapsed = active.started.elapsed();
            let tok_s = active.generated.len() as f64 / elapsed.as_secs_f64();
            let alpha_ms = active.t_alpha_compute.as_millis() as u64;
            let wire_ms = active.t_wire.as_millis() as u64;
            let total_ms = elapsed.as_millis() as u64;
            let other_ms = total_ms.saturating_sub(alpha_ms).saturating_sub(wire_ms);
            info!(
                task = %task_id,
                tokens = active.generated.len(),
                elapsed_s = elapsed.as_secs_f64(),
                tok_s,
                alpha_ms,
                wire_ms,
                other_ms,
                "ov-runtime task done"
            );
            self.active = None;
        }

        Ok(vec![(task_id, chunk)])
    }

    /// One static-KV (NPU) inference for the token at absolute `position`.
    /// Feeds `primary_in` ("input_ids" on the first stage, "hidden_states" on
    /// a relay stage) + attention_mask + position_ids + this stage's KV ring,
    /// runs, absorbs the new token's K/V from `present`, and returns output 0
    /// (logits on the head stage, hidden_states otherwise). The ring resets at
    /// `position == 0`, so this is correct for any stage in a pipeline.
    fn static_infer(
        &mut self,
        use_hidden: bool,
        in_dtype: ShimDType,
        in_shape: &[usize],
        in_bytes: &[u8],
        position: i64,
    ) -> EngineResult<(ShimDType, Vec<usize>, Vec<u8>)> {
        // Align the ring to this absolute position (resets at position 0) and
        // refresh the reusable mask buffer.
        {
            let sk = self.static_kv.as_mut().unwrap();
            sk.begin_token(position as usize);
            sk.write_mask_bytes();
        }
        let pos_bytes = position.to_le_bytes();

        // Feed primary input + mask + position + the per-layer past-KV ring.
        // self.runtime (mut) and self.static_kv (shared) are disjoint fields,
        // so we borrow cached names + ring buffers directly — no per-token
        // string or buffer allocation.
        {
            let sk = self.static_kv.as_ref().unwrap();
            let in_main = if use_hidden {
                sk.hidden_in.as_deref()
            } else {
                sk.ids_in.as_deref()
            }
            .ok_or_else(|| EngineError::Backend("static IR missing primary input".into()))?;
            self.runtime
                .set_input(in_main, in_dtype, in_shape, in_bytes)
                .map_err(map_ov_err)?;
            self.runtime
                .set_input(
                    &sk.attn_in,
                    ShimDType::I64,
                    &[1, sk.context],
                    &sk.mask_bytes,
                )
                .map_err(map_ov_err)?;
            self.runtime
                .set_input(&sk.pos_in, ShimDType::I64, &[1, 1], &pos_bytes)
                .map_err(map_ov_err)?;
            let shape = [1, sk.kv_heads, sk.past_len, sk.head_dim];
            for (li, layer) in sk.layers.iter().enumerate() {
                self.runtime
                    .set_input(&layer.key_in, sk.kv_dtype, &shape, &sk.key_buf[li])
                    .map_err(map_ov_err)?;
                self.runtime
                    .set_input(&layer.val_in, sk.kv_dtype, &shape, &sk.val_buf[li])
                    .map_err(map_ov_err)?;
            }
        }
        self.runtime.infer().map_err(map_ov_err)?;

        // Primary output (index 0): logits on the head stage, hidden_states
        // otherwise — the static export emits it before the present.* outputs.
        let (odt, oshape, obytes) = self.runtime.output(0).map_err(map_ov_err)?;

        // Absorb the new token's K/V. Validate each present.* output's byte
        // length AND shape against [1, kv_heads, context, head_dim] so a
        // dtype/factorization mismatch is a clear error, not silent corruption.
        let (n, expect, kvh, ctx, hd) = {
            let sk = self.static_kv.as_ref().unwrap();
            (
                sk.layers.len(),
                sk.present_layer_bytes(),
                sk.kv_heads,
                sk.context,
                sk.head_dim,
            )
        };
        let want_shape = [1usize, kvh, ctx, hd];
        for li in 0..n {
            let (ko, vo) = {
                let l = &self.static_kv.as_ref().unwrap().layers[li];
                (l.key_out, l.val_out)
            };
            let (_, kshape, kpres) = self.runtime.output(ko).map_err(map_ov_err)?;
            let (_, vshape, vpres) = self.runtime.output(vo).map_err(map_ov_err)?;
            if kpres.len() != expect
                || vpres.len() != expect
                || kshape != want_shape
                || vshape != want_shape
            {
                return Err(EngineError::Backend(format!(
                    "static present.{li} mismatch: key shape={kshape:?} len={} val shape={vshape:?} \
                     len={}; expected shape {want_shape:?} ({expect} bytes f16). Check \
                     num_kv_heads/head_dim/KV dtype in stage_config.",
                    kpres.len(),
                    vpres.len(),
                )));
            }
            let sk = self.static_kv.as_mut().unwrap();
            sk.absorb_layer(li, false, &kpres);
            sk.absorb_layer(li, true, &vpres);
        }
        Ok((odt, oshape, obytes))
    }

    fn step_last(&mut self) -> EngineResult<()> {
        let (hidden, shape, pos_opt) = self.recv_hidden_from_upstream()?;
        let (out, out_shape) = match pos_opt {
            // Static (NPU): the carried absolute position drives the ring
            // (reset at 0); seq is always 1.
            Some(pos) => self.run_relay(&hidden, shape, pos)?,
            None => {
                if shape[1] > 1 {
                    self.runtime.reset_state().map_err(map_ov_err)?;
                    self.position = 0;
                }
                let r = self.run_relay(&hidden, shape, self.position)?;
                self.position += shape[1] as i64;
                r
            }
        };
        let next = argmax_logits(&out, &out_shape)?;
        self.send_token_to_upstream(next)?;
        Ok(())
    }

    fn step_middle(&mut self) -> EngineResult<()> {
        let (hidden, shape, pos_opt) = self.recv_hidden_from_upstream()?;
        let (out, out_shape, fwd_pos) = match pos_opt {
            Some(pos) => {
                let (o, s) = self.run_relay(&hidden, shape, pos)?;
                (o, s, pos)
            }
            None => {
                if shape[1] > 1 {
                    self.runtime.reset_state().map_err(map_ov_err)?;
                    self.position = 0;
                }
                let (o, s) = self.run_relay(&hidden, shape, self.position)?;
                self.position += shape[1] as i64;
                (o, s, 0)
            }
        };
        let s3 = to_shape3(&out_shape);
        self.send_hidden_downstream(&out, s3, fwd_pos)?;
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
        // run_first processes one token per call on the static path, so warm
        // with a single token (enough to JIT the OV graph) regardless of path.
        let warm = &ids[..ids.len().min(1)];
        match self.run_first(warm, 0) {
            Ok(_) => {
                if let Some(sk) = self.static_kv.as_mut() {
                    sk.reset();
                } else {
                    let _ = self.runtime.reset_state();
                }
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

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let result: EngineResult<Vec<(TaskId, Chunk)>> = if self.spec.is_first_stage {
            self.step_first()
        } else if self.spec.is_last_stage {
            self.step_last().map(|_| Vec::new())
        } else {
            self.step_middle().map(|_| Vec::new())
        };
        // Keep main's rate-limited WARN (a persistently-failing step() must
        // not flood logs), but surface the Err to the caller — step_first
        // already cleared active + reset_state, so callers can now tell
        // "engine failed" from "still prefilling" (both were empty vecs).
        match &result {
            Ok(v) => {
                // First-stage idle steps return Ok(empty) even mid-failure
                // (a failed step clears `active`; the next poll no-ops), so
                // only a step that did real work closes a failing streak.
                // Relay-stage Ok is always a completed relay round.
                if !v.is_empty() || !self.spec.is_first_stage {
                    if let Some(suppressed) = self.step_warn.on_success() {
                        info!(suppressed, "ov-runtime step recovered");
                    }
                }
            }
            Err(e) => {
                match self.step_warn.on_failure(std::time::Instant::now()) {
                    Some(StepWarn::First) => warn!(error = %e, "ov-runtime step failed"),
                    Some(StepWarn::StillFailing { suppressed }) => {
                        warn!(error = %e, suppressed, "ov-runtime step still failing")
                    }
                    None => {}
                }
            }
        }
        result
    }

    fn cancel(&mut self, task_id: &TaskId) {
        self.pending.retain(|t| t.task_id != *task_id);
        // Abandoning the active task mirrors the step-failure recovery:
        // clear it and reset generation state so the next pending task
        // activates immediately instead of waiting for this one to
        // drain max_tokens worth of inference on the device.
        if self
            .active
            .as_ref()
            .is_some_and(|a| a.task.task_id == *task_id)
        {
            info!(task = %task_id, "ov-runtime cancel: abandoning active task");
            self.active = None;
            let _ = self.runtime.reset_state();
            if let Some(sk) = self.static_kv.as_mut() {
                sk.reset();
            }
            self.position = 0;
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
    eos_token_ids: Vec<u32>,
    input_names: Vec<String>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    listen_host: String,
    listen_port: Option<u16>,
    /// (context, kv_heads, head_dim) for a stateless static-shape (NPU) shard;
    /// None for the stateful path. static_seq is validated == 1 at load and not
    /// stored (the ring math assumes one token per step).
    static_params: Option<(u32, u32, u32)>,
}

impl OvRuntimeBuilder {
    pub fn new(
        pipeline_dir: impl Into<PathBuf>,
        rank: u32,
        total: u32,
        device: impl Into<String>,
    ) -> Self {
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

        self.static_params = if !stage_cfg.stateful {
            let ctx = stage_cfg.static_context.unwrap_or(0);
            let s = stage_cfg.static_seq.unwrap_or(1);
            let kvh = stage_cfg.num_kv_heads.unwrap_or(0);
            let hd = stage_cfg.head_dim.unwrap_or(0);
            // static_seq must be 1 (the runtime decodes one token per step) and
            // static_context must exceed it so past_len = context - 1 >= 1.
            if kvh == 0 || hd == 0 || s != 1 || ctx <= s {
                return Err(EngineError::InvalidConfig(format!(
                    "stateless (NPU) shard needs static_seq=1, static_context>static_seq, \
                     and num_kv_heads/head_dim in stage_config; got seq={s} ctx={ctx} \
                     kvh={kvh} hd={hd}"
                )));
            }
            events.push(LoadProgress::message(format!(
                "stateless static-KV shard: context={ctx} kv_heads={kvh} head_dim={hd}"
            )));
            // static_seq is validated == 1 above and never stored (the ring math
            // assumes one token per step); carry only (ctx, kv_heads, head_dim).
            Some((ctx, kvh, hd))
        } else {
            None
        };

        // A pipeline must be homogeneous: the static path carries a wire
        // position frame the stateful path does not, so a mixed pipeline would
        // desync. Fail fast when sibling stage configs are present locally
        // (single-host multi-process); for distributed deploys each node has
        // only its own stage dir, so the strict wire-frame validation in
        // recv_hidden_from_upstream is the backstop at the first activation.
        for r in 0..pipeline_cfg.num_stages {
            if r == self.rank {
                continue;
            }
            if let Ok(cfg) = read_stage_config(&self.pipeline_dir.join(format!("stage_{r}"))) {
                if cfg.stateful != stage_cfg.stateful {
                    return Err(EngineError::ShardRejected(format!(
                        "pipeline is not homogeneous: stage_{} stateful={} but stage_{r} \
                         stateful={} — all stages must share one KV mode (re-export the whole \
                         pipeline with a single --target)",
                        self.rank, stage_cfg.stateful, cfg.stateful
                    )));
                }
            }
        }

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
        let runtime =
            OvRuntime::compile(xml_path.to_str().unwrap_or_default(), &self.device, &plugin)
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
                let eos_in_tok = lookup_eos(&tokenizer_dir);
                self.eos_token_ids = if eos_in_tok.is_empty() {
                    lookup_eos(&self.pipeline_dir)
                } else {
                    eos_in_tok
                };
                events.push(LoadProgress::message(format!(
                    "tokenizer loaded; eos_token_ids={:?}",
                    self.eos_token_ids
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

        // Resolve canonical port names via the alias list. v5 IRs export
        // input ports under conventional names ("input_ids", "attention_mask",
        // ...) but the IR's "primary" name is sometimes an internal node id;
        // the alias list is what carries the canonical name.
        let mut canonical_inputs = std::collections::HashMap::new();
        let n_inputs = runtime.input_count();
        for canonical in [
            "input_ids",
            "hidden_states",
            "attention_mask",
            "position_ids",
            "beam_idx",
        ] {
            for idx in 0..n_inputs {
                let aliases = runtime.input_aliases(idx).map_err(map_ov_err)?;
                let primary = runtime.input_name(idx).map_err(map_ov_err)?;
                if aliases
                    .iter()
                    .any(|a| a == canonical || a.contains(canonical))
                {
                    canonical_inputs.insert(canonical.to_string(), primary);
                    break;
                }
            }
        }

        // Stateless static-shape (NPU) shards: resolve the explicit
        // past_key_values.* inputs + present.* outputs and allocate the
        // host-side KV ring. The primary output (logits on the head stage,
        // hidden_states otherwise) is output index 0 — no alias guess needed.
        let static_kv = if let Some((ctx, kvh, hd)) = self.static_params {
            let n_out = runtime.output_count();
            let mut layers: Vec<StaticKvLayer> = Vec::new();
            loop {
                let l = layers.len();
                let kin_s = format!("past_key_values.{l}.key");
                let vin_s = format!("past_key_values.{l}.value");
                let kout_s = format!("present.{l}.key");
                let vout_s = format!("present.{l}.value");
                let (mut key_in, mut val_in) = (None, None);
                for idx in 0..n_inputs {
                    let al = runtime.input_aliases(idx).map_err(map_ov_err)?;
                    if al.iter().any(|a| a.contains(&kin_s)) {
                        key_in = Some(runtime.input_name(idx).map_err(map_ov_err)?);
                    } else if al.iter().any(|a| a.contains(&vin_s)) {
                        val_in = Some(runtime.input_name(idx).map_err(map_ov_err)?);
                    }
                }
                let (key_in, val_in) = match (key_in, val_in) {
                    (Some(k), Some(v)) => (k, v),
                    _ => break,
                };
                let (mut key_out, mut val_out) = (None, None);
                for idx in 0..n_out {
                    let al = runtime.output_aliases(idx).map_err(map_ov_err)?;
                    if al.iter().any(|a| a.contains(&kout_s)) {
                        key_out = Some(idx);
                    } else if al.iter().any(|a| a.contains(&vout_s)) {
                        val_out = Some(idx);
                    }
                }
                let key_out = key_out.ok_or_else(|| {
                    EngineError::Backend(format!("static shard missing {kout_s}"))
                })?;
                let val_out = val_out.ok_or_else(|| {
                    EngineError::Backend(format!("static shard missing {vout_s}"))
                })?;
                layers.push(StaticKvLayer {
                    key_in,
                    val_in,
                    key_out,
                    val_out,
                });
            }
            if layers.is_empty() {
                return Err(EngineError::Backend(
                    "static shard: no past_key_values.* inputs found".into(),
                ));
            }
            // Cross-check: the contiguous-from-0 discovery above stops at the
            // first missing index. Compare against the total number of
            // past_key_values.* input ports so a gap / renamed / folded port is
            // a hard error instead of silently building fewer layers.
            let mut kv_input_ports = 0usize;
            for idx in 0..n_inputs {
                let al = runtime.input_aliases(idx).map_err(map_ov_err)?;
                if al.iter().any(|a| a.contains("past_key_values.")) {
                    kv_input_ports += 1;
                }
            }
            if kv_input_ports != layers.len() * 2 {
                return Err(EngineError::Backend(format!(
                    "static shard: resolved {} contiguous KV layers ({} ports) but the IR has \
                     {kv_input_ports} past_key_values.* input ports — a layer port is missing or \
                     misnamed (gap in the past_key_values.N sequence)",
                    layers.len(),
                    layers.len() * 2,
                )));
            }
            let (kvh, hd) = (kvh as usize, hd as usize);
            let past_len = (ctx - 1) as usize;
            // KV activations are f16 in the static export; derive elem_bytes
            // from the dtype so the two can never drift.
            let kv_dtype = ShimDType::F16;
            let elem_bytes = match kv_dtype {
                ShimDType::F16 | ShimDType::Bf16 => 2,
                ShimDType::F32 | ShimDType::I32 => 4,
                ShimDType::I64 => 8,
                ShimDType::I8 => 1,
            };
            let layer_bytes = kvh * past_len * hd * elem_bytes;
            let n = layers.len();
            // Cache the resolved primary/aux input port names so the per-token
            // decode loop does no HashMap lookups or string allocs. A static
            // shard always has attention_mask + position_ids; embed stages have
            // input_ids, relay/head stages have hidden_states.
            let attn_in = canonical_inputs
                .get("attention_mask")
                .cloned()
                .ok_or_else(|| {
                    EngineError::Backend("static IR missing attention_mask input".into())
                })?;
            let pos_in = canonical_inputs
                .get("position_ids")
                .cloned()
                .ok_or_else(|| {
                    EngineError::Backend("static IR missing position_ids input".into())
                })?;
            let ids_in = canonical_inputs.get("input_ids").cloned();
            let hidden_in = canonical_inputs.get("hidden_states").cloned();
            if ids_in.is_none() && hidden_in.is_none() {
                return Err(EngineError::Backend(
                    "static IR has neither input_ids nor hidden_states input".into(),
                ));
            }
            Some(StaticKv {
                past_len,
                context: ctx as usize,
                kv_heads: kvh,
                head_dim: hd,
                elem_bytes,
                kv_dtype,
                ids_in,
                hidden_in,
                attn_in,
                pos_in,
                layers,
                key_buf: vec![vec![0u8; layer_bytes]; n],
                val_buf: vec![vec![0u8; layer_bytes]; n],
                valid: 0,
                mask_bytes: vec![0u8; ctx as usize * 8],
            })
        } else {
            None
        };

        Ok(Box::new(OvRuntimeEngine {
            spec,
            runtime,
            rotary,
            hidden_size: self.hidden_size,
            tokenizer: self.tokenizer,
            eos_token_ids: self.eos_token_ids.clone(),
            upstream: self.upstream,
            downstream: self.downstream,
            runtime_handle: tokio::runtime::Handle::current(),
            position: 0,
            input_names: self.input_names,
            canonical_inputs,
            pending: Vec::new(),
            active: None,
            static_kv,
            step_warn: StepWarnLimiter::default(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::PeerLayout;

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
