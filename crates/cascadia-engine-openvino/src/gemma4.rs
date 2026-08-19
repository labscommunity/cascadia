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
    #[serde(default)]
    num_kv_shared_layers: u32,
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
    /// Option B: leading entries of `generated` that are the resume seed. The
    /// kv_coord capture key must SKIP them — `prompt_ids` already carries the
    /// resume ids, so counting the seed again keys the blob at a depth that
    /// never matches the fed sequence (same defect dist_spec fixed).
    resume_seed_len: usize,
    /// Issue-34: leading prompt tokens already warm in the OV state (RESTOREd). Prefill feeds only
    /// `prompt_ids[warm_prefix..]`. 0 ⇒ cold (default path unchanged).
    warm_prefix: usize,
    /// Wall-clock when the task became active. Used to compute the
    /// final tok/s the engine prints in its `task done` log line.
    started: std::time::Instant,
    /// Cumulative time inside `run_first` (stage_0 compute + read).
    t_alpha_compute: std::time::Duration,
    /// Cumulative time inside `send_hidden_downstream` +
    /// `recv_token_from_downstream` — i.e. wire send + downstream wait + recv.
    t_wire: std::time::Duration,
}

/// Effective prefill span. Default `usize::MAX` = the whole span in ONE pass (unchanged, fastest).
///
/// `CASCADIA_GEMMA4_FORCE_T1_PREFILL=1` ⇒ 1: fold EVERY token through the same T=1 path. A warm-resumed
/// SUFFIX prefill (e.g. 22 tokens at position 71) and a cold FULL prefill (93 tokens at 0) otherwise hit
/// different GEMM batch shapes; over int4 weights the rounding delta flips a token, so cross-chain
/// warm != cold byte-identical. Under T=1 both traverse the identical per-token kernel ⇒ bit-identical.
/// Opt-in only — production keeps the single-pass prefill (warm-resume stays greedy-equivalent there,
/// not bit-identical). Mirrors `qwen36::prefill_chunk`.
fn prefill_chunk() -> usize {
    if std::env::var("CASCADIA_GEMMA4_FORCE_T1_PREFILL")
        .ok()
        .as_deref()
        == Some("1")
    {
        1
    } else {
        usize::MAX
    }
}

pub struct Gemma4Engine {
    spec: ShardSpec,
    /// Count of layers whose own KV lives entirely in stage_0 and is SHARED by later stages
    /// (Gemma 4 E2B/E4B KV-sharing). Own-KV layers are `[0, total_layers - num_kv_shared_layers)`.
    /// Zero for dense models (every stage bears its own KV).
    num_kv_shared_layers: u32,
    /// Total pipeline stages (= `Gemma4Builder::total`). Used to map own-KV layers → KV-bearing ranks.
    num_stages: u32,
    runtime: OvRuntime,
    tokenizer: Option<Arc<Tokenizer>>,
    /// All EOS token ids configured for the model. Generation stops on
    /// the first token that matches ANY of these. See `lookup_eos`.
    eos_token_ids: Vec<u32>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    runtime_handle: tokio::runtime::Handle,
    position: i64,
    /// Issue-34: a warm restore (`set_state_blob`) leaves residue this model's cheap
    /// `reset_state` cannot scrub (see shim.cpp). Track it so the next reset rebuilds the
    /// request instead — mirrors qwen36. Without it, restore-over-live-state corrupts the
    /// warm continuation AND leaks (reasoning-channel `***`) into the following cold turn.
    state_restored: bool,
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
    /// Issue-34 Option C: opaque KV blob cache for the coordination plane.
    #[cfg(feature = "kv_coord")]
    kv: crate::kv_coordination::OvKvCache,
    /// Issue-34 Option C: lock-free holder mirror of `kv` — the capture sites write both, and
    /// `kv_holder()` hands this out so a busy engine answers pulls without contending the engine lock.
    #[cfg(feature = "kv_coord")]
    kv_share: crate::kv_coordination::SharedKvCache,
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
            // Issue-34 diag: on a PREFILL send (multi-token), log this shared-KV frame's context
            // depth. Warm-resume must ship cat(restored_prefix, suffix) — i.e. ctx == full prompt
            // len. If warm ctx == suffix len only, the cross_kv side-output is NOT reading the
            // set_state-restored variable (stale/unfused copy) ⇒ that is the warm≠cold root cause.
            // `kv_coordination` (and so `fnv1a64`) only exists under the `kv_coord` feature, so this
            // diagnostic must be gated with it — ungated it broke `cargo check` with default features.
            #[cfg(feature = "kv_coord")]
            if shape[1] > 1 && oshape.len() == 4 {
                info!(
                    tag = o.tag,
                    ctx = oshape[2],
                    tokens = shape[1],
                    position,
                    fnv = crate::kv_coordination::fnv1a64(&bytes),
                    "gemma4_cross_kv_prefill_ctx"
                );
            }
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

    fn recv_token_from_downstream(&mut self, prefill: bool) -> EngineResult<i32> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let (tensor, _) = self
            .block_on(async move {
                let mut guard = downstream.lock().await;
                // MID-TASK reply — deadlined; see `recv_tensor_reply` (Item 5).
                // A prefill reply waits on every remaining stage's
                // whole-prompt compute — widened budget (see
                // `recv_tensor_reply_prefill`), CEILINGED here: the transport
                // budget multiplies the operator's frame-transfer knob
                // (recv_timeout × 10 — the rig's 600 s made it 6000 s), and the
                // waiting head holds its admission slot the whole time, so one
                // dead downstream turned the node into a refuse-everything wall
                // for 100 minutes (2026-08-17 incident). ov-runtime ceilings
                // the same coupling at TOKEN_RECV_DEADLINE_CEILING. Measured
                // legit worst here: 51 s (4.2k tokens, 2-stage); the ceiling
                // leaves ~6x headroom for deeper pipelines / longer prompts.
                if prefill {
                    recv_prefill_reply_ceilinged(&mut guard).await
                } else {
                    guard.recv_reply().await
                }
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
        // Wire order per step: position frame, hidden frame, cross-KV header, then `count` cross-KV
        // frames. The position read transparently absorbs any I8 control frame (CAPTURE/RESTORE)
        // that arrives between turns (stateful workers only).
        let pos_t = self.recv_position_or_control()?;
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        let (hid_t, header_t) = self
            .block_on(async move {
                let mut guard = upstream.lock().await;
                // pos was already read via recv_position_or_control (the IDLE
                // wait that also absorbs a CAPTURE/RESTORE control frame between
                // turns); the hid + cross-KV header that FOLLOW are mid-step
                // replies — deadline them so a half-sent step can't wedge the
                // stage (Item 5).
                let hid = guard.recv_reply().await?.0;
                let hdr = guard.recv_reply().await?.0;
                Ok::<_, cascadia_transport::TransportError>((hid, hdr))
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
                    v.push(guard.recv_reply().await?.0);
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

    /// Read one frame from upstream, transparently handling any I8 control frame (CAPTURE/RESTORE/
    /// ABORT) and looping until a real (non-control) frame — the position tensor — arrives.
    fn recv_position_or_control(&mut self) -> EngineResult<WireTensor> {
        // Re-iterates only on the kv_coord control-frame path (I8 → continue); without kv_coord
        // that branch is compiled out and the body runs exactly once.
        #[cfg_attr(not(feature = "kv_coord"), allow(clippy::never_loop))]
        loop {
            let upstream = self
                .upstream
                .clone()
                .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
            let t = self
                .block_on(async move {
                    let mut guard = upstream.lock().await;
                    Ok::<_, cascadia_transport::TransportError>(guard.recv().await?.0)
                })
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            #[cfg(feature = "kv_coord")]
            if t.dtype == WireDType::I8 {
                self.handle_inbound_control(&t)?;
                continue;
            }
            return Ok(t);
        }
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
            let mut prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            // Option B forced-prefix resume: append the already-emitted assistant
            // tokens after the rendered prompt (concat, not replace) so the cold
            // prefill below carries them as context. No-op when not resuming
            // (`resume_ids()` normalizes a wire-legal `Some([])` too).
            let resume = task.resume_ids().map(<[i32]>::to_vec);
            if let Some(r) = resume.as_deref() {
                // Peer-supplied indices reach native OV prefill — bound them first.
                let vocab = tok.get_vocab_size(true) as u32;
                if let Err(e) = cascadia_types::validate_resume_ids(r, Some(vocab)) {
                    let id = task.task_id.clone();
                    return Ok(vec![(
                        id.clone(),
                        Chunk::error(id, format!("invalid resume prefix: {e}")),
                    )]);
                }
                // Exhausted budget: finish Length with ZERO new tokens — the old
                // flow pushed the first token before its length check and
                // overshot prefix+new past max_tokens.
                if r.len() >= task.max_tokens.max(1) as usize {
                    let mut c = Chunk::final_marker(task.task_id.clone(), "");
                    c.n_tokens = Some(0);
                    c.finish_reason = Some(cascadia_types::FinishReason::Length);
                    return Ok(vec![(task.task_id.clone(), c)]);
                }
            }
            cascadia_types::append_resume_ids(&mut prompt_ids, resume.as_deref());
            // Issue-34 warm-resume (gated, single-stage for now — multi-stage needs the RESTORE
            // broadcast). Restore a cached strict-prefix blob and prefill only the suffix; else cold.
            #[cfg_attr(not(feature = "kv_coord"), allow(unused_mut))]
            let mut warm_prefix = 0usize;
            #[cfg(feature = "kv_coord")]
            {
                let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
                if let Some((blob, len, plane_pulled)) =
                    self.kv.take_warm(&task.tenant, &prompt_i32)
                {
                    // Restore must land on a CLEAN request: this model's `reset_state` leaves residue
                    // (shim.cpp), and set_state over the prior throwaway turn's live state corrupts the
                    // warm continuation + leaks into the next cold turn. Rebuild first, mark restored so
                    // the following cold reset upgrades to a rebuild too.
                    let set_ok = self.restore_blob_clean(&blob);
                    // Issue-34 diag (mirror qwen36 70687b9): does set_state round-trip at the DECLARED
                    // level on THIS device? gemma4's head IR uniquely has cross_kv side-consumers on the
                    // KV concat. Mismatch ⇒ plugin set_state infidelity (mode A); exact ⇒ the warm≠cold
                    // delta is the cross_kv side-output reading a stale buffer (mode B, exporter fix).
                    if set_ok
                        && std::env::var("CASCADIA_GEMMA4_POSTPREFILL_FP")
                            .ok()
                            .as_deref()
                            == Some("1")
                    {
                        match self.runtime.get_state_blob() {
                            Ok(rt) => {
                                let a = crate::kv_coordination::fnv1a64(&blob);
                                let b = crate::kv_coordination::fnv1a64(&rt);
                                if a != b {
                                    warn!(set_fnv = a, rt_fnv = b, set_len = blob.len(), rt_len = rt.len(),
                                        "gemma4_state_roundtrip_mismatch (set_state lossy at declared level)");
                                } else {
                                    info!(
                                        fnv = a,
                                        len = blob.len(),
                                        "gemma4_state_roundtrip_exact (declared state faithful)"
                                    );
                                }
                            }
                            Err(e) => {
                                warn!(error = %e, "gemma4: get_state_blob round-trip diag failed")
                            }
                        }
                    }
                    if set_ok {
                        // Multi-stage: RESTORE the whole downstream chain (all-or-nothing); any rank
                        // short ⇒ ABORT everyone + cold (never a partial/corrupt warm).
                        let multi = self.downstream.is_some() && !self.spec.is_last_stage;
                        let chain_ok = !multi || {
                            let epoch = crate::kv_coordination::synth_epoch(&prompt_i32[..len]);
                            match self.send_restore_downstream(epoch) {
                                Ok(true) => true,
                                _ => {
                                    let _ = self.send_abort_downstream();
                                    false
                                }
                            }
                        };
                        if chain_ok {
                            // Real KV depth, not the token count (off-by-one — see kv_seq_from_blob).
                            warm_prefix = crate::kv_coordination::kv_seq_from_blob(&blob)
                                .map(|s| s.min(len))
                                .unwrap_or(len);
                            info!(
                                warm_prefix,
                                matched = len,
                                plane_pulled,
                                "gemma4 warm-resumed from KV blob"
                            );
                        } else {
                            let _ = self.runtime.reset_state();
                            warn!("gemma4: pipeline restore incomplete; cold reprefill");
                        }
                    }
                }
            }
            if warm_prefix == 0 {
                // A prior restore leaves residue cheap reset_state can't scrub — rebuild the request so
                // this cold turn (incl. a fresh session after a warm-migrated turn on the same runtime)
                // starts truly clean, not in the donor's reasoning-channel trajectory.
                if self.state_restored {
                    self.runtime.recreate_request().map_err(map_ov_err)?;
                    self.state_restored = false;
                } else {
                    self.runtime.reset_state().map_err(map_ov_err)?;
                }
            }
            self.position = warm_prefix as i64;
            info!(
                task = %task.task_id,
                prompt_tokens = prompt_ids.len(),
                warm_prefix,
                "gemma4 task active"
            );
            // Option B: pre-seed `generated` + `last_text` with the resumed
            // tokens so the budget check bounds prefix+new (not just new) and
            // the first NEW tail token decodes WITH the prefix as context.
            // Empty seed on a normal turn ⇒ identical to today.
            let resume_seed: Vec<i32> = cascadia_types::resume_generated_seed(resume.as_deref())
                .into_iter()
                .map(|t| t as i32)
                .collect();
            let resume_seed_len = resume_seed.len();
            let resume_last_text = if resume_seed.is_empty() {
                String::new()
            } else {
                let seed_u32: Vec<u32> = resume_seed.iter().map(|&t| t as u32).collect();
                tok.decode(&seed_u32, true)
                    .map_err(|e| EngineError::Backend(format!("tokenizer decode: {e}")))?
            };
            self.active = Some(ActiveTask {
                task,
                prompt_ids,
                generated: resume_seed,
                last_text: resume_last_text,
                prefilled: false,
                last_token: 0,
                warm_prefix,
                resume_seed_len,
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
                // warm_prefix is 0 on the cold/default path ⇒ full prompt (unchanged).
                (true, a.prompt_ids[a.warm_prefix..].to_vec())
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
        // Issue-34: fold the prefill in `prefill_chunk()`-sized spans. Default = whole span (one pass,
        // unchanged/fastest). CASCADIA_GEMMA4_FORCE_T1_PREFILL=1 ⇒ T=1, so a warm-resumed SUFFIX prefill
        // and a cold FULL prefill traverse the identical per-token kernel. Without it the two use
        // different GEMM batch shapes (e.g. 22 vs 93 tokens) over int4 weights, and the rounding delta
        // flips a token ⇒ cross-chain warm != cold. Mirrors qwen36's FORCE_T1_PREFILL.
        let chunk = prefill_chunk();
        let mut alpha = std::time::Duration::ZERO;
        let mut wire = std::time::Duration::ZERO;
        let mut next_token: i32;
        let position = self.position;
        if prefill && tokens.len() > 1 && chunk < tokens.len() {
            let mut i = 0usize;
            loop {
                let end = (i + chunk).min(tokens.len());
                let span = &tokens[i..end];
                let pos = self.position;
                let ts = std::time::Instant::now();
                let (out, shape) = self.run_first(span, pos)?;
                alpha += ts.elapsed();
                self.position += span.len() as i64;
                // Every span goes downstream so each stage folds the same way; the tokens from all but
                // the final span are discarded (the last one is the first decode token).
                // Always use the PREFILL (widened) deadline here even for T=1 spans: this is a prefill,
                // and the first span after a warm RESTORE follows a heavy set_state on the downstream
                // stage — the strict decode budget wedges there and the turn returns empty.
                let (tok, w) = self.resolve_next_token(&out, &shape, single_stage, pos, true)?;
                wire += w;
                next_token = tok;
                i = end;
                if i >= tokens.len() {
                    break;
                }
            }
        } else {
            let ts = std::time::Instant::now();
            let (out, shape) = self.run_first(&tokens, position)?;
            alpha = ts.elapsed();
            self.position += tokens.len() as i64;
            // 1-token prompt prefill costs the same downstream as a decode step —
            // keep the strict deadline (mirrors step_middle's shape[1] > 1) so
            // wedge eviction stays fast.
            let (tok, w) = self.resolve_next_token(
                &out,
                &shape,
                single_stage,
                position,
                prefill && tokens.len() > 1,
            )?;
            wire = w;
            next_token = tok;
        }
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
        prefill: bool,
    ) -> EngineResult<(i32, std::time::Duration)> {
        if single_stage {
            Ok((argmax_logits(out, shape)?, std::time::Duration::ZERO))
        } else {
            let s3 = to_shape3(shape);
            let ts = std::time::Instant::now();
            self.send_hidden_downstream(out, s3, position)?;
            let token = self.recv_token_from_downstream(prefill)?;
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
        //
        // On divergence RE-ANCHOR (empty delta), never `unwrap_or(&full_text)`:
        // that re-emits everything decoded so far, which under Option B resume is
        // the whole forced prefix duplicated into the client stream. Same contract
        // as the shim's `advance_emitted` — bounded loss at the seam, never
        // duplication.
        // A trailing U+FFFD means the newest token ends mid-glyph (byte-fallback
        // BPE): the decode is TRANSIENT, not diverged. Hold — emit nothing and
        // keep `last_text`, so the next token completes the glyph and the strip
        // emits it whole. Updating `last_text` here (the old behavior) baked the
        // replacement char into the anchor, so the completed glyph then failed
        // strip_prefix and was dropped as a fake divergence.
        let delta = if full_text.ends_with('\u{FFFD}') {
            String::new()
        } else {
            match full_text.strip_prefix(active.last_text.as_str()) {
                Some(d) => {
                    let d = d.to_string();
                    active.last_text = full_text;
                    d
                }
                None => {
                    warn!(
                        task = %active.task.task_id,
                        "decode diverged from the emitted prefix; re-anchoring (delta dropped)"
                    );
                    active.last_text = full_text;
                    String::new()
                }
            }
        };

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
                token_ids: Vec::new(),
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
            // Issue-34: capture this stage's KV under (prompt + generated) for warm-pull. Best-effort
            // + gated. (Multi-stage CAPTURE broadcast added with the control protocol.)
            #[cfg(feature = "kv_coord")]
            {
                // Skip the resume seed: `prompt_ids` already carries those ids.
                let full: Vec<i32> = active
                    .prompt_ids
                    .iter()
                    .map(|&t| t as i32)
                    .chain(
                        active
                            .generated
                            .iter()
                            .skip(active.resume_seed_len)
                            .copied(),
                    )
                    .collect();
                // H.1b R2: this turn's namespace, read off THIS task's own state — never off a
                // plane-asserted value, which describes a pulled entry, not this turn.
                let tenant = active.task.tenant.clone();
                match self.runtime.get_state_blob() {
                    Ok(blob) => {
                        // Multi-stage head: broadcast CAPTURE so every downstream rank snapshots its
                        // slice under this turn's content epoch. Best-effort.
                        if self.downstream.is_some() && !self.spec.is_last_stage {
                            let epoch = crate::kv_coordination::synth_epoch(&full);
                            if let Err(e) = self.send_capture_downstream(epoch, &full, &tenant) {
                                warn!(error = %e, "gemma4: CAPTURE broadcast failed (best-effort)");
                            }
                        }
                        // Mirror into the lock-free holder cache so a busy node can serve this turn's
                        // KV to a moved peer without the engine lock.
                        self.kv_share
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .capture(&tenant, full.clone(), blob.clone());
                        self.kv.capture(&tenant, full, blob);
                    }
                    Err(e) => tracing::debug!(error = %e, "gemma4 get_state_blob skipped"),
                }
            }
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
        // Multi-token hidden = prefill: the token reply waits on every
        // remaining stage's whole-prompt compute — widened budget.
        let prefill_reply = shape[1] > 1;
        if position == 0 {
            self.runtime.reset_state().map_err(map_ov_err)?;
        }
        let (out, out_shape) = self.run_relay(&hidden, shape, position)?;
        let s3 = to_shape3(&out_shape);
        // Forward the SAME absolute position downstream so every stage aligns.
        self.send_hidden_downstream(&out, s3, position)?;
        let token = self.recv_token_from_downstream(prefill_reply)?;
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

    #[cfg(feature = "kv_coord")]
    fn kv_coordination(&mut self) -> Option<&mut dyn cascadia_engine::KvCoordination> {
        Some(self)
    }

    #[cfg(feature = "kv_coord")]
    fn kv_holder(&self) -> Option<std::sync::Arc<dyn cascadia_engine::KvSnapshotHolder>> {
        Some(std::sync::Arc::new(crate::kv_coordination::OvKvHolder {
            cache: std::sync::Arc::clone(&self.kv_share),
            model_fp: self.kv_model_fingerprint(),
        }))
    }
}

// -------- Issue-34 §8 multi-stage CAPTURE/RESTORE over gemma4's frameless transport --------
// Same I8-control-tensor scheme as ov-runtime (real frames are F16/I64/F32, never I8 ⇒ collision
// -free). Stateful shards only; `kv_coord`-gated. gemma4's recv reads pos+hidden+cross-KV, so the
// demux peeks the FIRST frame's dtype before the multi-frame read.
#[cfg(feature = "kv_coord")]
const G_OPCODE_CAPTURE: u8 = 1;
#[cfg(feature = "kv_coord")]
const G_OPCODE_CAPTURE_ACK: u8 = 2;
#[cfg(feature = "kv_coord")]
const G_OPCODE_RESTORE: u8 = 3;
#[cfg(feature = "kv_coord")]
const G_OPCODE_RESTORE_ACK: u8 = 4;
#[cfg(feature = "kv_coord")]
const G_OPCODE_ABORT: u8 = 5;
#[cfg(feature = "kv_coord")]
const G_OPCODE_ABORT_ACK: u8 = 6;
/// H.1b (R2): CAPTURE whose body also carries the turn's TENANT (`capture_body_bytes_v2`). Separate
/// opcode, not a wider v1 body — the v1 codec enforces an exact length and hard-errors mid-chain on
/// a mismatch. Emitted only for a non-empty tenant, so a chain that names none stays on v1.
/// Ceiling on the widened prefill-reply wait (transport budget = recv_timeout × 10, which is
/// operator-frame-knob-coupled, not compute-coupled). Measured legit worst: 51 s. See the call site.
const G_PREFILL_REPLY_CEILING: std::time::Duration = std::time::Duration::from_secs(300);

/// The prefill reply wait, ceilinged. One definition for the call site and its test — a test that
/// re-implements the pattern inline pins nothing (deleting the call-site ceiling would leave it
/// green; this file has shipped that mistake before).
async fn recv_prefill_reply_ceilinged(
    g: &mut cascadia_transport::ActivationClient,
) -> Result<
    (
        cascadia_transport::Tensor,
        cascadia_transport::TransferStats,
    ),
    cascadia_transport::TransportError,
> {
    match tokio::time::timeout(G_PREFILL_REPLY_CEILING, g.recv_reply_prefill()).await {
        Ok(r) => r,
        Err(_) => Err(cascadia_transport::TransportError::SocketClosed),
    }
}
#[cfg(feature = "kv_coord")]
/// Bound on a downstream CAPTURE/RESTORE/ABORT ack; mirrors ov-runtime's RESTORE_ACK_TIMEOUT.
/// Every driver-side control exchange is bounded (qwen36 does the same via `reply_bounded`):
/// a dead peer must error the exchange, not wedge the engine step forever.
const G_CONTROL_ACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Bounded control-ack recv shared by the CAPTURE/RESTORE/ABORT exchanges. A
/// peer that dies without acking must surface as an error here — the caller
/// holds the downstream lock, so an unbounded wait wedges the engine step (and
/// with it the whole rank) forever.
#[cfg(feature = "kv_coord")]
async fn recv_control_ack_bounded(
    g: &mut cascadia_transport::ActivationClient,
) -> Result<WireTensor, cascadia_transport::TransportError> {
    let (ack, _) = tokio::time::timeout(G_CONTROL_ACK_TIMEOUT, g.recv())
        .await
        .map_err(|_| cascadia_transport::TransportError::SocketClosed)??;
    Ok(ack)
}
const G_OPCODE_CAPTURE_V2: u8 = 7;

#[cfg(feature = "kv_coord")]
impl Gemma4Engine {
    /// A non-empty `tenant` upgrades the frame to `G_OPCODE_CAPTURE_V2` so the downstream rank —
    /// which never sees the `GenerationTask` — can tag its own capture with the same namespace.
    fn send_capture_downstream(
        &mut self,
        epoch: u64,
        tokens: &[i32],
        tenant: &str,
    ) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let mut data = if tenant.is_empty() {
            vec![G_OPCODE_CAPTURE]
        } else {
            vec![G_OPCODE_CAPTURE_V2]
        };
        if tenant.is_empty() {
            data.extend_from_slice(&crate::kv_coordination::capture_body_bytes(epoch, tokens));
        } else {
            data.extend_from_slice(&crate::kv_coordination::capture_body_bytes_v2(
                epoch, tokens, tenant,
            ));
        }
        let t = WireTensor::new(WireDType::I8, [1, 1, data.len() as u32], data);
        let ack = self
            .block_on(async move {
                let mut g = downstream.lock().await;
                g.send(&t).await?;
                // Bounded, for the same reason ov-runtime bounds its CAPTURE ack: a peer on an
                // older build that meets G_OPCODE_CAPTURE_V2 errors WITHOUT acking, and an
                // unbounded wait here wedges this rank while it holds the downstream lock.
                let ack = recv_control_ack_bounded(&mut g).await?;
                Ok::<_, cascadia_transport::TransportError>(ack)
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        if ack.dtype == WireDType::I8 && ack.data.first() == Some(&G_OPCODE_CAPTURE_ACK) {
            Ok(())
        } else {
            Err(EngineError::Backend("gemma4: bad CAPTURE ack".into()))
        }
    }

    fn send_restore_downstream(&mut self, epoch: u64) -> EngineResult<bool> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let mut data = vec![G_OPCODE_RESTORE];
        data.extend_from_slice(&epoch.to_le_bytes());
        let t = WireTensor::new(WireDType::I8, [1, 1, data.len() as u32], data);
        let ack = self
            .block_on(async move {
                let mut g = downstream.lock().await;
                g.send(&t).await?;
                // Bounded like CAPTURE above: an unbounded wait on a dead peer
                // wedges this rank while it holds the downstream lock.
                let ack = recv_control_ack_bounded(&mut g).await?;
                Ok::<_, cascadia_transport::TransportError>(ack)
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        if ack.dtype == WireDType::I8 && ack.data.first() == Some(&G_OPCODE_RESTORE_ACK) {
            Ok(ack.data.get(1) == Some(&1))
        } else {
            Err(EngineError::Backend("gemma4: bad RESTORE ack".into()))
        }
    }

    fn send_abort_downstream(&mut self) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let t = WireTensor::new(WireDType::I8, [1, 1, 1], vec![G_OPCODE_ABORT]);
        self.block_on(async move {
            let mut g = downstream.lock().await;
            g.send(&t).await?;
            // Bounded like CAPTURE above: an unbounded wait on a dead peer
            // wedges this rank while it holds the downstream lock.
            let _ = recv_control_ack_bounded(&mut g).await?;
            Ok::<_, cascadia_transport::TransportError>(())
        })
        .map_err(|e| EngineError::Backend(e.to_string()))
    }

    fn send_control_ack_upstream(&mut self, ack_opcode: u8, payload: &[u8]) -> EngineResult<()> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        let mut data = vec![ack_opcode];
        data.extend_from_slice(payload);
        let t = WireTensor::new(WireDType::I8, [1, 1, data.len() as u32], data);
        self.block_on(async move {
            let mut g = upstream.lock().await;
            g.send(&t).await
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Handle an inbound I8 control tensor on a worker (called transparently inside the recv loop).
    fn handle_inbound_control(&mut self, t: &WireTensor) -> EngineResult<()> {
        match t.data.first().copied() {
            // V2 additionally carries the turn's tenant, which tags the stash and rides the chain on.
            Some(op @ (G_OPCODE_CAPTURE | G_OPCODE_CAPTURE_V2)) => {
                let (epoch, tokens, tenant) = if op == G_OPCODE_CAPTURE_V2 {
                    crate::kv_coordination::parse_capture_body_v2(&t.data[1..])
                        .ok_or_else(|| EngineError::Backend("gemma4: bad CAPTURE_V2 body".into()))?
                } else {
                    let (e, tk) = crate::kv_coordination::parse_capture_body(&t.data[1..])
                        .ok_or_else(|| EngineError::Backend("gemma4: bad CAPTURE body".into()))?;
                    (e, tk, crate::kv_coordination::LOCAL_NS.to_string())
                };
                if let Ok(blob) = self.runtime.get_state_blob() {
                    // Mirror into the lock-free holder cache (worker rank serves rank-N GET from here).
                    self.kv_share
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .capture_under_epoch_ns(&tenant, epoch, tokens.clone(), blob.clone());
                    self.kv
                        .capture_under_epoch_ns(&tenant, epoch, tokens.clone(), blob);
                }
                if !self.spec.is_last_stage {
                    if let Err(e) = self.send_capture_downstream(epoch, &tokens, &tenant) {
                        warn!(error = %e, "gemma4: CAPTURE chain downstream failed (best-effort)");
                    }
                }
                self.send_control_ack_upstream(G_OPCODE_CAPTURE_ACK, &[])
            }
            Some(G_OPCODE_RESTORE) => {
                let epoch = t
                    .data
                    .get(1..9)
                    .and_then(|b| b.try_into().ok())
                    .map(u64::from_le_bytes)
                    .ok_or_else(|| EngineError::Backend("gemma4: bad RESTORE body".into()))?;
                // No warm flag needed: gemma4 workers reset on the carried frame position, so the
                // head's warm prefill (position = warm_len > 0) skips reset, and a cold turn
                // (position 0) resets everyone — restored or not.
                let local_ok = if !self.has_own_kv() {
                    // KV-sharing tail: this stage holds no own KV (all shared upstream), so there is
                    // nothing to restore — a trivial success (never abort the chain on this rank).
                    true
                } else {
                    match self.kv.take_capture(epoch) {
                        Some((_, blob)) => self.restore_blob_clean(&blob),
                        None => false,
                    }
                };
                let down_ok = if self.spec.is_last_stage {
                    true
                } else {
                    self.send_restore_downstream(epoch).unwrap_or(false)
                };
                self.send_control_ack_upstream(
                    G_OPCODE_RESTORE_ACK,
                    &[u8::from(local_ok && down_ok)],
                )
            }
            Some(G_OPCODE_ABORT) => {
                let _ = self.runtime.reset_state();
                self.position = 0;
                if !self.spec.is_last_stage {
                    let _ = self.send_abort_downstream();
                }
                self.send_control_ack_upstream(G_OPCODE_ABORT_ACK, &[])
            }
            other => Err(EngineError::Backend(format!(
                "gemma4: unknown control opcode {other:?}"
            ))),
        }
    }
}

#[cfg(feature = "kv_coord")]
impl Gemma4Engine {
    /// Stable model+stage fingerprint — a stage only matches the identical stage on a peer chain.
    fn kv_model_fingerprint(&self) -> u64 {
        let s = &self.spec;
        let mut buf = s.model_id.clone().into_bytes();
        buf.extend_from_slice(&s.layer_start.to_le_bytes());
        buf.extend_from_slice(&s.layer_end.to_le_bytes());
        buf.extend_from_slice(&s.total_layers.to_le_bytes());
        crate::kv_coordination::fnv1a64(&buf)
    }
    /// Whether this stage bears any OWN KV: true iff its layer range starts before the own-KV
    /// layers `[0, total_layers - num_kv_shared_layers)`. Dense stages are always true (shared = 0);
    /// gemma4's KV-sharing tail (all own-KV upstream in stage_0) is false.
    /// `set_state_blob` onto a CLEAN request. gemma4's `reset_state` leaves residue (shim.cpp), so a
    /// bare set over the prior turn's live state corrupts the warm continuation AND leaks into the next
    /// cold turn. The head path established this as load-bearing, but the worker RESTORE handler and the
    /// plane `apply_warm_resume` were doing a bare set — so any non-head rank that bears own KV restored
    /// over dirty state. Returns false (⇒ that rank votes cold) if the rebuild fails, instead of
    /// proceeding into exactly the corruption the rebuild exists to prevent.
    #[cfg(feature = "kv_coord")]
    fn restore_blob_clean(&mut self, blob: &[u8]) -> bool {
        if let Err(e) = self.runtime.recreate_request() {
            warn!(error = %e, "gemma4: recreate_request before restore failed; cold reprefill");
            self.state_restored = true;
            return false;
        }
        self.state_restored = true;
        self.runtime.set_state_blob(blob).is_ok()
    }

    fn has_own_kv(&self) -> bool {
        self.spec.layer_start
            < self
                .spec
                .total_layers
                .saturating_sub(self.num_kv_shared_layers)
    }
}

#[cfg(feature = "kv_coord")]
impl cascadia_engine::KvCoordination for Gemma4Engine {
    fn model_fingerprint(&self) -> u64 {
        self.kv_model_fingerprint()
    }
    fn layout_version(&self) -> u16 {
        cascadia_kv_wire::OPAQUE_KV_LAYOUT
    }
    fn engine_rev(&self) -> u64 {
        crate::kv_coordination::KV_ENGINE_REV
    }
    fn tokenize(&self, text: &str) -> Option<Vec<i32>> {
        let enc = self.tokenizer.as_ref()?.encode(text, false).ok()?;
        Some(enc.get_ids().iter().map(|&u| u as i32).collect())
    }
    fn lookup(&mut self, partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        self.kv.lookup(partner, token_ids)
    }
    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(cascadia_kv_wire::Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let fp = self.kv_model_fingerprint();
        let (prefix, blob) = self.kv.serve(partner, expected_epoch, expected_len)?;
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
        partner: &str,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        let (tokens, blob) = crate::kv_coordination::wire_to_blob(manifest, payloads).ok_or(())?;
        // H.1b hard gate (§12.10.0a): key on the ASSERTED partner, never `manifest.partner`, which
        // the serving holder stamps and nothing validates.
        self.kv.insert_both(partner, tokens, blob);
        Ok(())
    }

    fn apply_warm_resume(&mut self, epoch: u64) -> bool {
        // Plane path (§0(B), multi-rank downstream): the pull staged this rank's slice under `epoch`;
        // set_state it now. Mirrors the RESTORE handler's local apply. step_first's take_warm then
        // re-warms + sets position (idempotent double-set; gemma4 keeps no warm flag). Not on the
        // total=1 path (the head warms its own rank-0 slice via take_warm).
        match self.kv.take_capture(epoch) {
            Some((_, blob)) => self.restore_blob_clean(&blob),
            None => false,
        }
    }

    fn abort_warm_resume(&mut self, epoch: u64) {
        // Drop a STAGED slice first so a later commit cannot resurrect it.
        let _ = self.kv.take_capture(epoch);
        // Verdict rejected after this rank applied — rebuild the request to drop the restored state.
        // gemma4 keeps no warm flag, so the rebuild alone returns it to cold; `state_restored` makes
        // the following cold reset upgrade to a rebuild too (this model's reset_state leaves residue).
        if let Err(e) = self.runtime.recreate_request() {
            warn!(error = %e, "gemma4: recreate_request on warm-resume abort failed");
        }
        self.state_restored = true;
    }

    fn kv_bearing_ranks(&self, total_ranks: usize) -> usize {
        // A stage bears own KV iff its layer range starts before the own-KV layers [0, own).
        // Layers split ceil(total/num_stages) per stage (matches export_gemma4.py).
        let own = self
            .spec
            .total_layers
            .saturating_sub(self.num_kv_shared_layers);
        if own == 0 || self.num_stages == 0 {
            return total_ranks;
        }
        let per_stage = self.spec.total_layers.div_ceil(self.num_stages);
        if per_stage == 0 {
            return total_ranks;
        }
        let n = (0..self.num_stages)
            .filter(|k| (k * per_stage) < own)
            .count();
        n.max(1).min(total_ranks)
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
    /// Extra `(key, value)` OV plugin properties plumbed verbatim from the CLI.
    pub ov_properties: Vec<(String, String)>,
    runtime: Option<OvRuntime>,
    spec: Option<ShardSpec>,
    /// KV-sharing layer count from pipeline_config.json, stashed in load() for build(). See
    /// [`Gemma4Engine::num_kv_shared_layers`].
    num_kv_shared_layers: u32,
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
    /// Append extra `(key, value)` OV plugin properties (CLI perf flags).
    pub fn with_ov_properties(mut self, props: Vec<(String, String)>) -> Self {
        self.ov_properties.extend(props);
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
        for (key, val) in &self.ov_properties {
            p = p.with(key, val);
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
        self.num_kv_shared_layers = pipeline_cfg.num_kv_shared_layers;

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
            num_kv_shared_layers: self.num_kv_shared_layers,
            num_stages: self.total,
            runtime,
            tokenizer: self.tokenizer,
            eos_token_ids: self.eos_token_ids.clone(),
            upstream: self.upstream,
            downstream: self.downstream,
            runtime_handle: tokio::runtime::Handle::current(),
            position: 0,
            state_restored: false,
            canonical_inputs,
            pending: Vec::new(),
            active: None,
            cross_kv_out,
            external_kv_in,
            pending_external_kv: std::collections::HashMap::new(),
            step_warn: StepWarnLimiter::default(),
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
            #[cfg(feature = "kv_coord")]
            kv_share: std::sync::Arc::new(std::sync::Mutex::new(
                crate::kv_coordination::OvKvCache::default(),
            )),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::PeerLayout;

    /// Rig incident (2026-08-17): RESTORE/ABORT ack recvs were UNBOUNDED while
    /// CAPTURE's was bounded, so a peer that died without acking wedged the
    /// engine step forever (holding the downstream lock). Every control-ack
    /// recv now routes through `recv_control_ack_bounded`, which must ERROR on
    /// a silent peer once the budget lapses, never wedge.
    #[cfg(feature = "kv_coord")]
    #[tokio::test]
    async fn control_ack_recv_errors_instead_of_wedging_on_a_silent_peer() {
        // A peer that accepts and then never writes — the dead-peer shape.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let accept = tokio::spawn(async move { listener.accept().await.unwrap().0 });
        let mut client = cascadia_transport::ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        let _held_open = accept.await.unwrap();

        // Pause AFTER the real connect so auto-advance only drives the recv
        // bound (and the test's own backstop), not the connect timeout.
        tokio::time::pause();
        let res = tokio::time::timeout(
            G_CONTROL_ACK_TIMEOUT * 4,
            super::recv_control_ack_bounded(&mut client),
        )
        .await;
        match res {
            Ok(Err(_)) => {} // bounded: the dead peer surfaces as an error
            Ok(Ok(_)) => panic!("a silent peer cannot produce an ack"),
            Err(_) => panic!("control-ack recv wedged past its bound on a silent peer"),
        }
    }

    /// Same incident, prefill flavor: the transport's widened prefill budget is
    /// recv_timeout × 10 — the rig's 600 s knob made it 6000 s, and the waiting
    /// head held its admission slot the whole time (a one-slot node refused
    /// everything for 100 minutes). The ceiling must error a silent downstream
    /// at G_PREFILL_REPLY_CEILING regardless of how large the operator knob is.
    #[tokio::test]
    async fn prefill_reply_wait_is_ceilinged_on_a_silent_downstream() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let accept = tokio::spawn(async move { listener.accept().await.unwrap().0 });
        let mut client = cascadia_transport::ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        let _held_open = accept.await.unwrap();

        tokio::time::pause();
        // Under a paused clock BOTH the ceiling and the transport's own 10× budget fire instantly
        // in real time, so "errors eventually" cannot distinguish them (a first draft of this test
        // passed with the ceiling deleted). Discriminate on ELAPSED VIRTUAL TIME: with the ceiling,
        // the error lands at exactly G_PREFILL_REPLY_CEILING; without it, at the transport budget.
        let budget = cascadia_transport::recv_timeout()
            .saturating_mul(cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR);
        assert!(
            budget > G_PREFILL_REPLY_CEILING,
            "precondition: the transport budget ({budget:?}) must exceed the ceiling in this test \
             env or the assertion below cannot discriminate — raise the ceiling gap or the env knob"
        );
        let t0 = tokio::time::Instant::now();
        let res = super::recv_prefill_reply_ceilinged(&mut client).await;
        let elapsed = t0.elapsed();
        assert!(
            res.is_err(),
            "a silent downstream cannot produce a prefill reply"
        );
        assert!(
            elapsed <= G_PREFILL_REPLY_CEILING + std::time::Duration::from_secs(1),
            "prefill reply errored only after {elapsed:?} — the ceiling was not applied \
             (transport budget is {budget:?})"
        );
    }

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
