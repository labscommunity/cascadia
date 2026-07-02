//! Gemma 4 multi-stage OpenVINO engine (`--engine gemma4`).
//!
//! Serves per-stage `gemma4_cached_v1` OV IR shards (exported by
//! `tools/export_gemma4.py`) across the TCP activation transport, single-stage
//! or pipeline-parallel. Own KV is OV internal state, reset between tasks.
//!
//! Pipeline-dir layout:
//! ```text
//! <pipeline-dir>/
//!     pipeline_config.json
//!     tokenizer/                 # HF tokenizer.json + special tokens
//!     stage_0/openvino_model.{xml,bin}, stage_config.json
//!     stage_N/...
//! ```
//!
//! Wire format (downstream, per step): an absolute-position frame (relay stages
//! use it as their `position_ids` base and reset own KV when it is 0 — correct
//! for any prompt length), then the hidden activation (f32, matching the IR's
//! f32 ports; PLI rides inside it), then a self-describing cross-KV header
//! (count + per-frame tags) and that many cross-stage shared-KV frames.
//!
//! Gemma 4 E2B/E4B reuse a few source layers' KV across later layers; when a
//! shared layer's source sits in an earlier stage, that KV crosses the wire as
//! `cross_kv.*` outputs → `external_kv.*` inputs, matched by source-identity tag
//! (`src_layer*2 + is_value`) so pairing is robust to port ordering and stage
//! count. Dense shards (no KV-sharing) have no such ports and take a zero-cost
//! hidden-only relay. Non-adjacent (multi-hop) KV sharing is rejected at load.

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
use tracing::{debug, info, warn};

use crate::warn_limit::{StepWarn, StepWarnLimiter};

// -------- pipeline / stage config --------

#[derive(Debug, Deserialize)]
struct PipelineConfig {
    model_id: String,
    num_stages: u32,
    #[serde(default)]
    num_layers: u32,
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
    /// All gemma4 shards are stateful (own KV is OV internal state). The field
    /// is kept so the engine can reject a non-stateful (stateless/NPU) shard
    /// with a clear message — the gemma4 engine has no static-KV path.
    #[serde(default = "default_true")]
    stateful: bool,
    /// Local indices (within this stage) of layers whose own KV a LATER stage
    /// reuses — i.e. the source layers behind this stage's `cross_kv.*` outputs,
    /// in `cross_kv.0,1,…` order. Global source-layer id = `layer_start + this`.
    /// Used to tag each cross_kv frame on the wire by source id so the consumer
    /// pairs explicitly (not positionally). Empty unless this stage produces
    /// cross-stage shared KV.
    #[serde(default)]
    cross_stage_sources_local: Vec<u32>,
    /// The cross-stage shared-KV sources this stage CONSUMES, in
    /// `external_kv.0,1,…` order. Each carries the GLOBAL source-layer id used
    /// to match an incoming wire frame to the right `external_kv.*` input.
    #[serde(default)]
    external_shared_sources: Vec<ExternalSrc>,
}

/// One entry of `external_shared_sources` in stage_config.json. Only the global
/// source-layer id is needed at runtime (to match cross-stage KV by source).
#[derive(Debug, Deserialize, Default, Clone)]
struct ExternalSrc {
    #[serde(default)]
    src_global_layer: u32,
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

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
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
/// Shared by both output-reading paths (run_first / run_relay).
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

/// Pack a Gemma 4 `cross_kv.*` output (rank-4 `[1,kvh,ctx,hd]`) into a rank-3
/// wire frame `[kvh,ctx,hd]`. The batch dim is always 1 (single sequence) and
/// the transport caps tensor rank at MAX_RANK=3, so we drop it; the receiver
/// restores it. Payload is unchanged (dropping a unit dim leaves the element
/// count identical), so the transport's strict `payload == shape*dtype` check
/// still holds. Errors if the output dtype is not F32 (the gemma4 IR emits f32
/// KV; mislabeling it would corrupt the consumer) or the shape is not rank-4
/// batch-1.
fn pack_cross_kv_frame(
    dt: ShimDType,
    shape4: &[usize],
    bytes: Vec<u8>,
) -> EngineResult<WireTensor> {
    if dt != ShimDType::F32 {
        return Err(EngineError::Backend(format!(
            "cross_kv output dtype is {dt:?}, expected F32 (the gemma4 IR emits f32 KV); \
             the wire frame would mislabel it"
        )));
    }
    if shape4.len() != 4 || shape4[0] != 1 {
        return Err(EngineError::Backend(format!(
            "cross_kv output shape {shape4:?} is not [1,kvh,ctx,hd]"
        )));
    }
    let ws = [shape4[1] as u32, shape4[2] as u32, shape4[3] as u32];
    Ok(WireTensor::new(WireDType::F32, ws, bytes))
}

/// Encode the cross-KV header: an I64 frame whose first element is the number
/// of cross_kv frames that follow, then that many per-frame TAGS (in send
/// order). Each tag identifies the source layer + key/value role of the frame
/// (`src_global_layer * 2 + is_value`), so the consumer reads exactly the right
/// number of frames (desync-proof) and matches each to its `external_kv.*`
/// input by tag (order- and stage-count-independent). Always sent — a header of
/// `[0]` means no cross-stage KV (single-stage / dense shards).
fn encode_cross_kv_header(tags: &[i64]) -> WireTensor {
    let mut vals = Vec::with_capacity(1 + tags.len());
    vals.push(tags.len() as i64);
    vals.extend_from_slice(tags);
    let n = vals.len() as u32;
    WireTensor::new(WireDType::I64, [1, 1, n], i64_to_bytes(&vals))
}

/// Decode + validate the cross-KV header (see [`encode_cross_kv_header`]).
/// Returns the per-frame tags for the frames that follow.
fn decode_cross_kv_header(t: &WireTensor) -> EngineResult<Vec<i64>> {
    if t.dtype != WireDType::I64 || t.data.is_empty() || !t.data.len().is_multiple_of(8) {
        return Err(EngineError::Backend(format!(
            "expected an I64 cross-KV header frame, got dtype={:?} len={} — likely a \
             desynced activation stream",
            t.dtype,
            t.data.len()
        )));
    }
    let vals: Vec<i64> = t
        .data
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let count = vals[0];
    if count < 0 || count as usize != vals.len() - 1 {
        return Err(EngineError::Backend(format!(
            "cross-KV header declares {count} frames but carries {} tags",
            vals.len() - 1
        )));
    }
    Ok(vals[1..].to_vec())
}

/// Parse a `cross_kv.{i}.key|value` / `external_kv.{i}.key|value` port name into
/// its `(index, is_value)`. Returns None for any other port.
fn parse_kv_port(name: &str, prefix: &str) -> Option<(usize, bool)> {
    let rest = name.strip_prefix(prefix)?;
    let mut it = rest.split('.');
    let idx: usize = it.next()?.parse().ok()?;
    let is_value = match it.next()? {
        "value" => true,
        "key" => false,
        _ => return None,
    };
    Some((idx, is_value))
}

/// A `cross_kv.*` output this stage produces: the OV output port index + the
/// wire tag (`src_global_layer * 2 + is_value`) identifying the source layer
/// and key/value role.
#[derive(Clone)]
struct CrossKvOut {
    out_idx: usize,
    tag: i64,
}

/// An `external_kv.*` input this stage consumes: the IR port name + the wire tag
/// (`src_global_layer * 2 + is_value`) it must be fed from.
#[derive(Clone)]
struct ExternalKvIn {
    name: String,
    tag: i64,
}

/// Encode the absolute start-position as its own framed wire tensor (I64
/// `[1,1,1]`), sent before each hidden activation. Relay stages use it directly
/// as their `position_ids` base and reset their stateful KV when it is 0 (start
/// of a new sequence) — so a 1-token prompt is handled correctly, unlike a
/// `seq_len > 1` heuristic. The transport requires `payload_len == shape*dtype`
/// and MAX_RANK=3 leaves no spare slot in the hidden tensor, so position rides
/// as its own frame. Paired with `decode_wire_position` — keep the two in sync.
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
             desynced activation stream",
            t.dtype,
            t.data.len()
        )));
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&t.data);
    Ok(i64::from_le_bytes(b))
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
    /// `recv_token_from_downstream` — i.e. wire send + downstream wait + recv.
    t_wire: std::time::Duration,
}

pub struct Gemma4Engine {
    spec: ShardSpec,
    runtime: OvRuntime,
    tokenizer: Option<Arc<Tokenizer>>,
    /// All EOS token ids configured for the model. Generation stops on
    /// the first token that matches ANY of these. See `lookup_eos`.
    eos_token_ids: Vec<u32>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    runtime_handle: tokio::runtime::Handle,
    position: i64,
    /// Map of canonical name (e.g. "input_ids", "attention_mask") to the
    /// IR's primary port name. Resolved at engine build time via the
    /// alias lookup (the IR's primary name is sometimes an internal
    /// node id, not the canonical name). Empty for v3 IRs.
    canonical_inputs: std::collections::HashMap<String, String>,
    pending: Vec<GenerationTask>,
    active: Option<ActiveTask>,
    /// Cross-stage shared KV this stage PRODUCES (Gemma 4 E2B/E4B KV-sharing):
    /// each `cross_kv.*` output port + its wire tag (source layer + key/value).
    /// Read after each infer and sent downstream tagged so the consumer pairs
    /// by source identity, not port position. Empty for single-stage / dense
    /// shards (e.g. 31B → zero-cost pure hidden relay).
    cross_kv_out: Vec<CrossKvOut>,
    /// Cross-stage shared KV this stage CONSUMES: each `external_kv.*` input
    /// port + the wire tag it must be fed from. Matched against the tagged
    /// frames received from upstream each step.
    external_kv_in: Vec<ExternalKvIn>,
    /// Cross-stage KV received from upstream this step, keyed by wire tag, ready
    /// to feed into the `external_kv.*` inputs. Rebuilt each step.
    pending_external_kv: std::collections::HashMap<i64, (Vec<usize>, Vec<u8>)>,
    step_warn: StepWarnLimiter,
}

impl Gemma4Engine {
    /// Resolve a canonical input name to the IR's primary port name.
    fn input_named(&self, canonical: &str) -> Option<&str> {
        self.canonical_inputs.get(canonical).map(|s| s.as_str())
    }

    fn build_feed_first(&mut self, input_ids: &[i64], position: i64) -> EngineResult<()> {
        // Gemma4: input_ids + position_ids only. The causal mask is baked into
        // the IR (no attention_mask input) and RoPE is computed internally from
        // position_ids (no cos/sin). Own KV is OV-stateful; cross-stage shared
        // KV (external_kv.*) is fed by feed_external_kv().
        let seq_len = input_ids.len();
        let pos: Vec<i64> = (position..position + seq_len as i64).collect();
        let in_ids = self
            .input_named("input_ids")
            .ok_or_else(|| EngineError::Backend("gemma4 IR missing input_ids".into()))?
            .to_string();
        let in_pos = self
            .input_named("position_ids")
            .ok_or_else(|| EngineError::Backend("gemma4 IR missing position_ids".into()))?
            .to_string();
        self.runtime
            .set_input(
                &in_ids,
                ShimDType::I64,
                &[1, seq_len],
                &i64_to_bytes(input_ids),
            )
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_pos, ShimDType::I64, &[1, seq_len], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        self.feed_external_kv()?;
        Ok(())
    }

    fn build_feed_relay(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<()> {
        // hidden_states (f32) + position_ids. `shape[2]` is the full width
        // (hidden_size + the per-layer-embedding tail when pli_dim>0); the IR
        // input accepts that width, so PLI rides inside hidden_states with no
        // special handling here.
        let seq_len = shape[1];
        let pos: Vec<i64> = (position..position + seq_len as i64).collect();
        let in_hs = self
            .input_named("hidden_states")
            .ok_or_else(|| EngineError::Backend("gemma4 IR missing hidden_states".into()))?
            .to_string();
        let in_pos = self
            .input_named("position_ids")
            .ok_or_else(|| EngineError::Backend("gemma4 IR missing position_ids".into()))?
            .to_string();
        // Gemma 4 keeps activations in f32 — the IR's hidden_states input is
        // f32 (unlike v5, whose hidden is f16). Feed raw f32; the upstream
        // stage sends it as f32 over the wire (see send_hidden_downstream).
        let hs_bytes = f32_to_bytes(hidden);
        self.runtime
            .set_input(&in_hs, ShimDType::F32, &shape, &hs_bytes)
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_pos, ShimDType::I64, &[1, seq_len], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        self.feed_external_kv()?;
        Ok(())
    }

    /// Feed cross-stage shared KV (`external_kv.{i}.{key,value}` inputs) from
    /// the frames received from upstream this step, matched by wire TAG (source
    /// layer + key/value role) rather than by position — so pairing is correct
    /// regardless of port ordering, and a missing source is a clear error, not
    /// silent wrong-KV. No-op for stages with no external sources (single-stage
    /// / dense shards).
    fn feed_external_kv(&mut self) -> EngineResult<()> {
        if self.external_kv_in.is_empty() {
            return Ok(());
        }
        // Move pending + clone the (small) input list to release the &self
        // borrows before the &mut self.runtime calls.
        let pending = std::mem::take(&mut self.pending_external_kv);
        let inputs = self.external_kv_in.clone();
        for ext in &inputs {
            let (shape, bytes) = pending.get(&ext.tag).ok_or_else(|| {
                EngineError::Backend(format!(
                    "external_kv input {} needs cross-stage KV (tag {}) but no matching frame \
                     arrived from upstream — the immediate upstream stage does not produce it. \
                     Non-adjacent KV sharing across more than one pipeline hop is not supported; \
                     export with KV source/consumer on adjacent stages (or fewer stages).",
                    ext.name, ext.tag
                ))
            })?;
            self.runtime
                .set_input(&ext.name, ShimDType::F32, shape, bytes)
                .map_err(map_ov_err)?;
        }
        Ok(())
    }

    fn run_first(
        &mut self,
        input_ids: &[i64],
        position: i64,
    ) -> EngineResult<(Vec<f32>, Vec<usize>)> {
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
        // Gemma 4 activations are f32 (the IR's hidden_states input/output are
        // f32). Send raw f32 so the downstream stage feeds it without a lossy
        // f16 round-trip — matches the f32 Core reference.
        let hid = WireTensor::new(WireDType::F32, wire_shape, f32_to_bytes(hidden));
        // The absolute start-position rides as its own frame (sent first): relay
        // stages use it directly as their position_ids base and reset their KV
        // when it is 0. MAX_RANK=3 + the strict payload==shape*dtype check leave
        // no room to pack it into the hidden tensor.
        let pos = encode_wire_position(position);
        // Gemma 4 E2B/E4B: this stage's source layers produced cross-stage
        // shared KV (`cross_kv.*` outputs) downstream stages consume as their
        // `external_kv.*` inputs. Read each from the just-completed inference,
        // pack the rank-4 [1,kvh,ctx,hd] tensor into a rank-3 wire frame
        // [kvh,ctx,hd] (batch is always 1; transport caps rank at MAX_RANK=3),
        // and tag it by source so the consumer pairs explicitly. A self-
        // describing header (count + tags) precedes the frames so the consumer
        // reads exactly the right number — desync-proof. Empty header [0] for
        // single-stage / dense shards.
        let outs = self.cross_kv_out.clone();
        let mut tags = Vec::with_capacity(outs.len());
        let mut cross_frames = Vec::with_capacity(outs.len());
        for o in &outs {
            let (dt, oshape, bytes) = self.runtime.output(o.out_idx).map_err(map_ov_err)?;
            cross_frames.push(pack_cross_kv_frame(dt, &oshape, bytes)?);
            tags.push(o.tag);
        }
        let header = encode_cross_kv_header(&tags);
        self.block_on(async move {
            let mut guard = downstream.lock().await;
            guard.send(&pos).await?;
            guard.send(&hid).await?;
            guard.send(&header).await?;
            for f in &cross_frames {
                guard.send(f).await?;
            }
            Ok::<(), cascadia_transport::TransportError>(())
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

    fn recv_hidden_from_upstream(&mut self) -> EngineResult<(Vec<f32>, [usize; 3], i64)> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        // Wire order per step: position frame, hidden frame, cross-KV header,
        // then `count` cross-KV frames (count from the header). Read the first
        // three, decode the header to learn `count`, then read that many.
        let (pos_t, hid_t, header_t) = self
            .block_on(async {
                let mut guard = upstream.lock().await;
                let pos = guard.recv().await?.0;
                let hid = guard.recv().await?.0;
                let hdr = guard.recv().await?.0;
                Ok::<_, cascadia_transport::TransportError>((pos, hid, hdr))
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        // Decode + strictly validate the position + header outside the transport
        // closure (so a bad frame is a clear EngineError, not a silent desync).
        let position = decode_wire_position(&pos_t)?;
        let tags = decode_cross_kv_header(&header_t)?;
        let n = tags.len();
        // Strict frame-count guard (restores the old count-based check) AND the
        // distributed-deploy runtime backstop for the load-time adjacency guard
        // (which is skipped when the sibling stage_config isn't on local disk):
        // the immediate upstream must send EXACTLY this stage's external_kv
        // sources — no more (over-production / non-adjacent >1-hop sharing) and
        // no fewer. A mismatch is a clear error, not a silently-ignored frame.
        if n != self.external_kv_in.len() {
            return Err(EngineError::Backend(format!(
                "upstream sent {n} cross-KV frame(s) but this stage consumes {} external_kv \
                 input(s) — a cross-stage KV source/consumer mismatch (non-adjacent KV sharing \
                 across more than one pipeline hop, or a mis-exported pipeline). The immediate \
                 upstream must produce exactly this stage's external_kv sources.",
                self.external_kv_in.len()
            )));
        }
        let up2 = self.upstream.clone().unwrap();
        let ext_tensors = self
            .block_on(async move {
                let mut guard = up2.lock().await;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(guard.recv().await?.0);
                }
                Ok::<_, cascadia_transport::TransportError>(v)
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        // Index each cross-KV frame by its tag (validating dtype), so
        // feed_external_kv can match it to the right `external_kv.*` input.
        self.pending_external_kv.clear();
        for (tag, wt) in tags.iter().zip(ext_tensors.iter()) {
            if wt.dtype != WireDType::F32 {
                return Err(EngineError::Backend(format!(
                    "cross-KV frame (tag {tag}) has dtype {:?}, expected F32",
                    wt.dtype
                )));
            }
            let s = &wt.shape;
            self.pending_external_kv.insert(
                *tag,
                (
                    vec![1usize, s[0] as usize, s[1] as usize, s[2] as usize],
                    wt.data.clone(),
                ),
            );
        }
        let shape = [
            hid_t.shape[0] as usize,
            hid_t.shape[1] as usize,
            hid_t.shape[2] as usize,
        ];
        let floats = match hid_t.dtype {
            WireDType::F32 => hid_t
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            WireDType::F16 => f16_bytes_to_f32(&hid_t.data),
            other => {
                return Err(EngineError::Backend(format!(
                    "unexpected upstream hidden dtype {other:?}"
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
            self.runtime.reset_state().map_err(map_ov_err)?;
            self.position = 0;
            info!(
                task = %task.task_id,
                prompt_tokens = prompt_ids.len(),
                "gemma4 task active"
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
            // DEBUG, not WARN: the outer step() emits the rate-limited WARN
            // for the same error (StepWarnLimiter); a second unconditional
            // WARN here would bypass the limiter and double-log every
            // first-stage failure.
            debug!(
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
        // prompt would otherwise emit a fabricated token with no inference.
        if prefill && tokens.is_empty() {
            return Err(EngineError::Backend(
                "empty prompt: no tokens to prefill".into(),
            ));
        }

        // The whole prompt (prefill) or one decode token in a single
        // multi-token inference; the IR keeps own KV internally. The absolute
        // start-position is sent downstream so relay stages align position_ids
        // and reset on 0.
        let position = self.position;
        let ts = std::time::Instant::now();
        let (out, shape) = self.run_first(&tokens, position)?;
        let alpha = ts.elapsed();
        self.position += tokens.len() as i64;
        let (next_token, wire) = self.resolve_next_token(&out, &shape, single_stage, position)?;
        if let Some(a) = self.active.as_mut() {
            a.t_alpha_compute += alpha;
            a.t_wire += wire;
        }

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
    /// check stop conditions, and build the streamed chunk (first stage only).
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
                "gemma4 task done"
            );
            self.active = None;
        }

        Ok(vec![(task_id, chunk)])
    }

    fn step_last(&mut self) -> EngineResult<()> {
        // The upstream stage carries the absolute start-position; use it
        // directly as the position_ids base and reset own KV when it is 0 (a
        // new sequence). Correct for any prompt length, including 1 token.
        let (hidden, shape, position) = self.recv_hidden_from_upstream()?;
        if position == 0 {
            self.runtime.reset_state().map_err(map_ov_err)?;
        }
        let (out, out_shape) = self.run_relay(&hidden, shape, position)?;
        let next = argmax_logits(&out, &out_shape)?;
        self.send_token_to_upstream(next)?;
        Ok(())
    }

    fn step_middle(&mut self) -> EngineResult<()> {
        let (hidden, shape, position) = self.recv_hidden_from_upstream()?;
        if position == 0 {
            self.runtime.reset_state().map_err(map_ov_err)?;
        }
        let (out, out_shape) = self.run_relay(&hidden, shape, position)?;
        let s3 = to_shape3(&out_shape);
        // Forward the SAME absolute position downstream so every stage aligns.
        self.send_hidden_downstream(&out, s3, position)?;
        let token = self.recv_token_from_downstream()?;
        self.send_token_to_upstream(token)?;
        Ok(())
    }
}

impl Engine for Gemma4Engine {
    fn warmup(&mut self) {
        if !(self.spec.is_first_stage) {
            info!("gemma4 warmup skipped on non-first stage");
            return;
        }
        let tok = match self.tokenizer.clone() {
            Some(t) => t,
            None => {
                warn!("gemma4 warmup skipped: no tokenizer");
                return;
            }
        };
        let enc = match tok.encode("Hi", false) {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "gemma4 warmup tokenize failed");
                return;
            }
        };
        let ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
        // A single token is enough to JIT the OV graph.
        let warm = &ids[..ids.len().min(1)];
        match self.run_first(warm, 0) {
            Ok(_) => {
                let _ = self.runtime.reset_state();
                self.position = 0;
                info!("gemma4 warmup ok");
            }
            Err(e) => warn!(error = %e, "gemma4 warmup failed"),
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if !self.spec.is_first_stage {
            warn!("gemma4 submit() ignored on non-first stage");
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
                "gemma4: pending queue at cap; rejecting task"
            );
            return Err(EngineError::QueueFull {
                queued: self.pending.len(),
                cap: crate::dist_spec::MAX_PENDING_TASKS,
            });
        }
        self.pending.push(task);
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Step-wise engine: drop the queued task and clear the active one so
        // step() stops decoding it. cancel/step never overlap (runner mutex).
        self.pending.retain(|t| &t.task_id != task_id);
        if self
            .active
            .as_ref()
            .is_some_and(|a| &a.task.task_id == task_id)
        {
            self.active = None;
        }
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
                // (a failed step_first clears `active`; the next poll
                // no-ops), so only a step that did real work closes a
                // streak. Relay-stage Ok is always a completed relay round.
                if !v.is_empty() || !self.spec.is_first_stage {
                    if let Some(suppressed) = self.step_warn.on_success() {
                        info!(suppressed, "gemma4 step recovered");
                    }
                }
            }
            Err(e) => match self.step_warn.on_failure(std::time::Instant::now()) {
                Some(StepWarn::First) => warn!(error = %e, "gemma4 step failed"),
                Some(StepWarn::StillFailing { suppressed }) => {
                    warn!(error = %e, suppressed, "gemma4 step still failing")
                }
                None => {}
            },
        }
        result
    }
}

// -------- Builder --------

#[derive(Default)]
pub struct Gemma4Builder {
    pub pipeline_dir: PathBuf,
    pub rank: u32,
    pub total: u32,
    pub device: String,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    runtime: Option<OvRuntime>,
    spec: Option<ShardSpec>,
    tokenizer: Option<Arc<Tokenizer>>,
    eos_token_ids: Vec<u32>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    listen_host: String,
    listen_port: Option<u16>,
    /// Global source-layer id for each `cross_kv.{i}` this stage produces, in
    /// port-index order (= layer_start + cross_stage_sources_local[i]). Used in
    /// build() to tag each cross_kv frame by source. Populated in load().
    cross_src_ids: Vec<i64>,
    /// Global source-layer id for each `external_kv.{i}` this stage consumes, in
    /// port-index order (= external_shared_sources[i].src_global_layer).
    external_src_ids: Vec<i64>,
}

impl Gemma4Builder {
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
impl Builder for Gemma4Builder {
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

        // The gemma4 engine only runs stateful shards (own KV is OV internal
        // state); there is no stateless/static-KV (NPU) path. Reject clearly.
        if !stage_cfg.stateful {
            return Err(EngineError::ShardRejected(format!(
                "stage_{} is a stateless (static-KV/NPU) shard; the gemma4 engine requires \
                 stateful shards. Re-export without the static/NPU target.",
                self.rank
            )));
        }

        // Cross-stage shared-KV source ids (Gemma 4 E2B/E4B KV-sharing).
        // Producer: cross_kv.{i} ← layer_start + cross_stage_sources_local[i].
        // Consumer: external_kv.{i} ← external_shared_sources[i].src_global_layer.
        // build() tags each wire frame by these so pairing is by source identity,
        // never by port position.
        self.cross_src_ids = stage_cfg
            .cross_stage_sources_local
            .iter()
            .map(|s| (stage_cfg.layer_start + s) as i64)
            .collect();
        self.external_src_ids = stage_cfg
            .external_shared_sources
            .iter()
            .map(|e| e.src_global_layer as i64)
            .collect();

        // Load-time adjacency guard: every cross-stage source this stage
        // consumes must be PRODUCED by the immediately-upstream stage (the only
        // peer that sends us cross-KV frames). Non-adjacent sharing (a source >1
        // hop upstream) is not supported — pass-through forwarding is a separate
        // follow-up. Validate against the sibling config when present locally
        // (single-host multi-process); for distributed deploys the per-frame tag
        // match in feed_external_kv is the immediate, deterministic backstop.
        if self.rank > 0 && !self.external_src_ids.is_empty() {
            let up_dir = self.pipeline_dir.join(format!("stage_{}", self.rank - 1));
            if let Ok(up) = read_stage_config(&up_dir) {
                let produced: std::collections::HashSet<i64> = up
                    .cross_stage_sources_local
                    .iter()
                    .map(|s| (up.layer_start + s) as i64)
                    .collect();
                if let Some(missing) = self.external_src_ids.iter().find(|s| !produced.contains(s))
                {
                    return Err(EngineError::ShardRejected(format!(
                        "stage_{} consumes cross-stage KV from source layer {missing}, but the \
                         immediate upstream stage_{} does not produce it (it produces {:?}). \
                         Non-adjacent KV sharing across more than one pipeline hop is not \
                         supported — re-export with KV source/consumer on adjacent stages, or \
                         use fewer stages.",
                        self.rank,
                        self.rank - 1,
                        produced
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
        self.runtime = Some(runtime);

        events.push(LoadProgress::message("loading tokenizer".to_string()));

        // Gemma 4 bakes RoPE into the IR (computed internally from
        // position_ids), so unlike the v3 path there is no host-side rotary to
        // load here — only the tokenizer dir, used for the tokenizer + EOS ids.
        let tokenizer_dir = self.pipeline_dir.join("tokenizer");

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

        // Cross-stage shared-KV ports (Gemma 4 E2B/E4B KV-sharing).
        // `cross_kv.{i}.{key|value}` are OUTPUTS this stage produces; the global
        // source-layer id is cross_src_ids[i] (set in load()). `external_kv.{i}`
        // are INPUTS it consumes, source id external_src_ids[i]. Each gets a
        // wire TAG = src*2 + is_value so the consumer matches frames to inputs
        // by source identity, not port position (robust to ordering and >9
        // ports). Empty for single-stage / dense shards (no such ports).
        let output_names = runtime.output_names().map_err(map_ov_err)?;
        let mut cross_kv_out = Vec::new();
        for (oidx, name) in output_names.iter().enumerate() {
            if let Some((ci, is_value)) = parse_kv_port(name, "cross_kv.") {
                let src = *self.cross_src_ids.get(ci).ok_or_else(|| {
                    EngineError::Backend(format!(
                        "{name}: no source-id metadata (stage_config \
                         cross_stage_sources_local has {} entries)",
                        self.cross_src_ids.len()
                    ))
                })?;
                cross_kv_out.push(CrossKvOut {
                    out_idx: oidx,
                    tag: src * 2 + is_value as i64,
                });
            }
        }
        let input_names = runtime.input_names().map_err(map_ov_err)?;
        let mut external_kv_in = Vec::new();
        for name in &input_names {
            if let Some((ei, is_value)) = parse_kv_port(name, "external_kv.") {
                let src = *self.external_src_ids.get(ei).ok_or_else(|| {
                    EngineError::Backend(format!(
                        "{name}: no source-id metadata (stage_config \
                         external_shared_sources has {} entries)",
                        self.external_src_ids.len()
                    ))
                })?;
                external_kv_in.push(ExternalKvIn {
                    name: name.clone(),
                    tag: src * 2 + is_value as i64,
                });
            }
        }
        if !cross_kv_out.is_empty() || !external_kv_in.is_empty() {
            info!(
                cross_kv_out = cross_kv_out.len(),
                external_kv_in = external_kv_in.len(),
                "gemma4 cross-stage shared KV wired"
            );
        }

        Ok(Box::new(Gemma4Engine {
            spec,
            runtime,
            tokenizer: self.tokenizer,
            eos_token_ids: self.eos_token_ids.clone(),
            upstream: self.upstream,
            downstream: self.downstream,
            runtime_handle: tokio::runtime::Handle::current(),
            position: 0,
            canonical_inputs,
            pending: Vec::new(),
            active: None,
            cross_kv_out,
            external_kv_in,
            pending_external_kv: std::collections::HashMap::new(),
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
        let mut b = Gemma4Builder::new("/non/existent", 0, 1, "CPU");
        let res = b.load(ShardSpec::single_stage("m", "CPU")).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn build_before_load_errors() {
        let b = Box::new(Gemma4Builder::new("/x", 0, 1, "CPU"));
        assert!(matches!(b.build(), Err(EngineError::NotLoaded)));
    }

    #[tokio::test]
    async fn connect_no_peers_is_noop_for_single_stage() {
        let mut b = Gemma4Builder::new("/x", 0, 1, "CPU");
        b.connect(PeerLayout::single_stage()).await.unwrap();
    }

    /// The load-time adjacency guard must reject a 3-stage pipeline where a
    /// stage consumes cross-stage KV from a source the immediate upstream does
    /// NOT produce (non-adjacent / multi-hop sharing) — with a clear error,
    /// before compiling the IR. Hermetic: only stage_config.json files are
    /// needed (the guard runs ahead of OV compile), so no IR / transport.
    #[tokio::test]
    async fn rejects_non_adjacent_cross_stage_kv() {
        let dir = std::env::temp_dir().join(format!("cascadia_g4_guard_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("stage_1")).unwrap();
        std::fs::create_dir_all(dir.join("stage_2")).unwrap();
        std::fs::write(
            dir.join("pipeline_config.json"),
            r#"{"model_id":"m","num_stages":3,"num_layers":30}"#,
        )
        .unwrap();
        // stage_1 (the immediate upstream of stage_2) produces NO cross-stage KV.
        std::fs::write(
            dir.join("stage_1/stage_config.json"),
            r#"{"layer_start":10,"layer_end":20,"stateful":true,"cross_stage_sources_local":[]}"#,
        )
        .unwrap();
        // stage_2 consumes source layer 5 — which lives in stage_0, two hops up.
        std::fs::write(
            dir.join("stage_2/stage_config.json"),
            r#"{"layer_start":20,"layer_end":30,"has_head":true,"stateful":true,
                "external_shared_sources":[{"src_global_layer":5}]}"#,
        )
        .unwrap();
        let mut b = Gemma4Builder::new(&dir, 2, 3, "CPU");
        let res = b.load(ShardSpec::single_stage("m", "CPU")).await;
        std::fs::remove_dir_all(&dir).ok();
        match res {
            Err(EngineError::ShardRejected(msg)) => {
                assert!(
                    msg.contains("non-adjacent") || msg.contains("does not produce"),
                    "expected a non-adjacency rejection, got: {msg}"
                );
            }
            Err(e) => panic!("expected ShardRejected, got a different error: {e:?}"),
            Ok(_) => {
                panic!("expected ShardRejected for non-adjacent KV sharing, but load() succeeded")
            }
        }
    }
}
