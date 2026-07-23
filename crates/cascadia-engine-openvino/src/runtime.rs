//! Multi-stage OpenVINO Runtime engine.
//!
//! Rust port of `cascadia/worker/engines/openvino/ov_runtime.py`. Loads
//! pre-exported per-stage stateful OV IRs (v3+ shard format), runs them
//! across the existing TCP transport, with stateful KV cache internal to
//! the IR and `reset_state()` between independent generation tasks.
//!
//! Pipeline-dir layout (as produced by `cascadia shard`):
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
use tracing::{debug, info, warn};

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
    /// Chunked-prefill IR variant (openvino_prefill_model.xml) query-window
    /// width, exported via `--static-prefill-seq`. Absent/None on decode-only
    /// static exports and on all stateful exports.
    #[serde(default)]
    static_prefill_seq: Option<u32>,
    /// Total context of the prefill variant; its past-KV length
    /// (`static_prefill_context - static_prefill_seq`) must equal the decode
    /// variant's (`static_context - 1`) so both share one host KV ring.
    #[serde(default)]
    static_prefill_context: Option<u32>,
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

/// Pick the next token from a logits output's LAST row — the seq-agnostic
/// case of `argmax_logits_row`, kept as the common entry point. Shares its
/// shape guards so degenerate-logits handling cannot diverge.
fn argmax_logits(logits: &[f32], shape: &[usize]) -> EngineResult<i32> {
    let vocab = match shape.last() {
        Some(&v) if v > 0 => v,
        _ => {
            return Err(EngineError::Backend(format!(
                "logits output has empty/zero last dim: shape={shape:?}"
            )))
        }
    };
    let rows = logits.len() / vocab;
    if rows == 0 {
        return Err(EngineError::Backend(format!(
            "logits len {} < vocab {vocab} (shape={shape:?})",
            logits.len()
        )));
    }
    // A length that isn't a whole number of rows means the producer handed
    // us a truncated/corrupted buffer — fail loud instead of confidently
    // argmaxing a start-aligned window and silently dropping the tail.
    if logits.len() % vocab != 0 {
        return Err(EngineError::Backend(format!(
            "logits len {} is not a multiple of vocab {vocab} (shape={shape:?})",
            logits.len()
        )));
    }
    argmax_logits_row(logits, shape, rows - 1)
}

/// Argmax over row `row` of a `[.., seq, vocab]` logits output. The chunked
/// prefill path needs the LAST REAL row (`real - 1`), not the last row —
/// rows `real..seq` of a padded chunk are garbage pad-query outputs.
fn argmax_logits_row(logits: &[f32], shape: &[usize], row: usize) -> EngineResult<i32> {
    let vocab = match shape.last() {
        Some(&v) if v > 0 => v,
        _ => {
            return Err(EngineError::Backend(format!(
                "logits output has empty/zero last dim: shape={shape:?}"
            )))
        }
    };
    let end = (row + 1)
        .checked_mul(vocab)
        .filter(|&e| e <= logits.len())
        .ok_or_else(|| {
            EngineError::Backend(format!(
                "logits row {row} out of range: len {} vocab {vocab} (shape={shape:?})",
                logits.len()
            ))
        })?;
    Ok(argmax_last_row(&logits[..end], vocab))
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
/// 8 payload bytes and non-negative; anything else (a desynced stream, a
/// stateful peer that sent a hidden tensor where a position was expected, or
/// a corrupted frame) is a hard error rather than a silently wrong position.
/// The sign check matters: downstream ring math casts to usize and the chunk
/// path adds per-row offsets — a negative value would wrap instead of erroring.
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
    let position = i64::from_le_bytes(b);
    if position < 0 {
        return Err(EngineError::Backend(format!(
            "negative wire position {position} — corrupted or desynced activation stream"
        )));
    }
    Ok(position)
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
    /// Exactly `fill_static_mask` with `real == 1` — the decode mask is the
    /// one-token special case of the chunk mask (unit-tested equivalence).
    fn write_mask_bytes(&mut self) {
        let (ctx, past_len, valid) = (self.context, self.past_len, self.valid);
        fill_static_mask(&mut self.mask_bytes, ctx, past_len, valid, 1);
    }

    /// Copy the new token's K/V (slot `past_len` of `present`) into the
    /// layer's ring buffer, appending at `valid` or sliding the window when
    /// full. Selects `key_buf[li]` or `val_buf[li]` internally so callers can
    /// hold `&mut self` without aliasing the buffer field.
    fn absorb_layer(&mut self, li: usize, is_value: bool, present: &[u8]) {
        let (present_ctx, past_len, valid) = (self.context, self.past_len, self.valid);
        self.absorb_one(li, is_value, present, present_ctx, past_len, valid);
    }

    /// Core single-token absorb, parameterized so the seq=1 decode variant
    /// (`present` rows are `context` slots, new token at slot `past_len`) and
    /// the chunked-prefill variant (rows are `present_ctx` slots, token t of
    /// the chunk at slot `past_len + t`) share one slide/append
    /// implementation. `valid_now` is the number of real past tokens visible
    /// to the absorbed token (drives append-at vs slide-drop-oldest).
    fn absorb_one(
        &mut self,
        li: usize,
        is_value: bool,
        present: &[u8],
        present_ctx: usize,
        src_slot: usize,
        valid_now: usize,
    ) {
        // Read scalar fields into locals before borrowing the buffer field,
        // so the mutable buffer borrow stays disjoint from these reads.
        let slot = self.head_dim * self.elem_bytes;
        let present_row = present_ctx * slot; // per-head stride in present
        let buf_row = self.past_len * slot; // per-head stride in the ring
        let full = valid_now >= self.past_len;
        let kv_heads = self.kv_heads;
        let past_len = self.past_len;
        let buf: &mut [u8] = if is_value {
            &mut self.val_buf[li]
        } else {
            &mut self.key_buf[li]
        };
        for h in 0..kv_heads {
            let src = h * present_row + src_slot * slot;
            let new = &present[src..src + slot];
            let base = h * buf_row;
            if full {
                buf.copy_within(base + slot..base + buf_row, base); // drop oldest
                let dst = base + (past_len - 1) * slot;
                buf[dst..dst + slot].copy_from_slice(new);
            } else {
                let dst = base + valid_now * slot;
                buf[dst..dst + slot].copy_from_slice(new);
            }
        }
    }

    /// Absorb `real` new tokens' K/V from a chunked-prefill `present` output
    /// (per-head rows of `present_ctx` slots; chunk token t sits at slot
    /// `past_len + t`). SLOT PLACEMENT is byte-for-byte what `real`
    /// sequential seq=1 `absorb_layer` calls at positions
    /// `first_position..+real` would do (unit-tested with identical `present`
    /// bytes), including the slide-drop-oldest once the window fills. Note
    /// this says nothing about the KV *values*: those come from the wide
    /// forward, whose attention only matches per-token stepping while every
    /// row's position stays <= past_len (see `chunk_take`).
    /// Does not mutate `valid` — like `absorb_layer`, callers realign via
    /// `begin_token` before the next inference.
    fn absorb_layer_multi(
        &mut self,
        li: usize,
        is_value: bool,
        present: &[u8],
        present_ctx: usize,
        first_position: usize,
        real: usize,
    ) {
        for t in 0..real {
            let valid_t = (first_position + t).min(self.past_len);
            self.absorb_one(
                li,
                is_value,
                present,
                present_ctx,
                self.past_len + t,
                valid_t,
            );
        }
    }

    /// Write a chunked-prefill attention mask into `buf` (i64 LE,
    /// `prefill_ctx` slots) for the current `valid`. See `fill_static_mask`.
    fn write_prefill_mask(&self, buf: &mut Vec<u8>, prefill_ctx: usize, real: usize) {
        fill_static_mask(buf, prefill_ctx, self.past_len, self.valid, real);
    }
}

/// The one static-path attention-mask writer (i64 LE, `ctx` slots): 1 for the
/// `valid` left-aligned real past slots, 1 for the first `real` slots of the
/// query window (slots `past_len..past_len+real`), 0 elsewhere (past padding
/// + chunk-tail padding). Pad queries' outputs are garbage by construction;
/// they are never absorbed or forwarded, and the causal triangle inside the
/// IR keeps real queries from attending to them. The seq=1 decode mask is the
/// `real == 1, ctx == past_len + 1` case; sharing one writer means a mask
/// semantics change cannot silently diverge the two paths.
///
/// NOTE this exposes ALL `valid` past slots to every query row, so a chunk is
/// only equivalent to per-token seq=1 stepping while every row's absolute
/// position stays <= past_len (past eviction happens between seq=1 steps but
/// cannot happen inside a chunk). Callers enforce that cap (`chunk_take`).
fn fill_static_mask(buf: &mut Vec<u8>, ctx: usize, past_len: usize, valid: usize, real: usize) {
    buf.clear();
    buf.resize(ctx * 8, 0);
    for i in 0..ctx {
        let in_past = i < valid;
        let in_window = i >= past_len && i < past_len + real;
        let v: i64 = if in_past || in_window { 1 } else { 0 };
        buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
}

/// Width of the next prefill chunk starting at absolute `position`: bounded
/// by the model's chunk window `c`, the `remaining` prompt tokens, and — the
/// parity cap — the number of rows that keep every query's absolute position
/// <= `past_len`. Beyond that boundary a single chunk-wide mask would let
/// later rows attend to tokens the seq=1 sliding window evicts between steps
/// (see `fill_static_mask`), so callers step the overflow tail one token at a
/// time. Returns 0 when no chunk row fits (position already past the window).
fn chunk_take(c: usize, remaining: usize, position: usize, past_len: usize) -> usize {
    let window = (past_len + 1).saturating_sub(position);
    c.min(remaining).min(window)
}

/// Primary input of one prefill chunk: prompt ids on the embed stage, hidden
/// rows (`hid` floats per token, row-major) on relay/head stages. Conversion
/// into the IR's dtype happens directly inside the reusable `primary_buf`.
#[derive(Clone, Copy)]
enum ChunkInput<'a> {
    Ids(&'a [i64]),
    Hidden(&'a [f32], usize),
}

/// Byte width of a float output element, for prefix-slicing converted rows.
fn float_elem_bytes(dtype: ShimDType) -> EngineResult<usize> {
    match dtype {
        ShimDType::F16 => Ok(2),
        ShimDType::F32 => Ok(4),
        other => Err(EngineError::Backend(format!(
            "unexpected float output dtype {other:?}"
        ))),
    }
}

/// Chunked-prefill sibling of a stateless static-shape shard: the same stage
/// graph reshaped to a fixed seq=`seq` query window (exported via
/// `--static-prefill-seq`), compiled separately — possibly on a different
/// device (`--prefill-device`, e.g. NPU while `--device CPU` decodes). Both
/// variants take identical `[1, kv_heads, past_len, head_dim]` past tensors
/// (the exporter pins `prefill context = past_len + seq`), so this runtime
/// reads and feeds the SAME host `StaticKv` ring as the decode model: the
/// prefill→decode handoff is zero-cost by construction, and the two devices
/// never hold KV — only weights — so DRAM keeps a single KV copy.
struct StaticPrefill {
    /// The compiled prefill model. `None` while PARKED (`--park-prefill`):
    /// dropping the CompiledModel releases its resident weight copy — the
    /// structural memory cost of the two-model split — between prefills;
    /// `ensure_prefill_loaded` re-creates it from `PrefillReload` on demand.
    /// Wiring, window dims, and buffers persist across park/reload.
    runtime: Option<OvRuntime>,
    /// Fixed query-window width C of the prefill IR.
    seq: usize,
    /// Fixed total context of the prefill IR (= ring past_len + seq).
    context: usize,
    ids_in: Option<String>,
    hidden_in: Option<String>,
    attn_in: String,
    pos_in: String,
    layers: Vec<StaticKvLayer>,
    /// Reusable buffers (mask: i64 LE context*8; primary: padded chunk input;
    /// pos: i64 LE seq*8) so the chunk loop does no per-chunk allocation.
    mask_bytes: Vec<u8>,
    primary_buf: Vec<u8>,
    pos_buf: Vec<u8>,
}

/// Everything needed to re-create a parked prefill compilation: the variant's
/// IR path, its device, and the plugin properties the original compile used
/// (CACHE_DIR etc., so a reload hits the OV blob/UMD cache instead of a cold
/// compile). `blob` short-circuits the reload to an AOT blob import.
struct PrefillReload {
    xml: String,
    device: String,
    plugin_entries: Vec<(String, String)>,
    blob: Option<String>,
}

/// Path of a precompiled AOT blob to IMPORT instead of compiling, when the
/// target device is the NPU and a `.blob` sibling of the IR exists (same
/// basename: `openvino_model.xml` → `openvino_model.blob`). Blobs come from
/// `ov::CompiledModel::export_model` — typically cross-compiled on a big-RAM
/// host with `NPU_PLATFORM` set — and importing skips the NPU compiler's
/// ~6–9×-INT4-bytes host-RAM transient entirely (measured ~7 s / no visible
/// RAM movement for an 8B variant, vs a >23 GB on-box compile). NPU only:
/// blobs are device- and driver-specific, and CPU/GPU compiles are cheap.
fn npu_aot_blob(xml_path: &std::path::Path, device: &str) -> Option<std::path::PathBuf> {
    if !device.trim().to_ascii_uppercase().starts_with("NPU") {
        return None;
    }
    let blob = xml_path.with_extension("blob");
    if !blob.exists() {
        return None;
    }
    // Refuse a blob older than its IR: silently serving a stale compile's
    // weights is the worst failure mode. The fallthrough recompiles (slow,
    // and on small-RAM boxes possibly OOM) — but loudly and correctly.
    match (
        blob.metadata().and_then(|m| m.modified()),
        xml_path.metadata().and_then(|m| m.modified()),
    ) {
        (Ok(bm), Ok(xm)) if bm < xm => {
            tracing::error!(
                blob = %blob.display(),
                "AOT blob is OLDER than its IR — ignoring it (re-export or delete \
                 the stale blob); falling back to on-box compile"
            );
            return None;
        }
        (Ok(_), Ok(_)) => {}
        // Could not read both mtimes (unsupported FS, a TOCTOU delete after the
        // exists() check, permissions): the freshness guard cannot run. Import
        // the blob rather than force a needless recompile, but do NOT do it
        // silently — an unverified stale blob imports and infers wrong tokens
        // at a clean 200 OK, exactly the failure mode the guard exists to catch.
        _ => tracing::warn!(
            blob = %blob.display(),
            "AOT blob freshness could not be verified (IR/blob mtime unavailable); \
             importing it unchecked — validate the blob with a real inference at deploy"
        ),
    }
    Some(blob)
}

/// Plugin config for a blob import: the compile-time properties minus
/// CACHE_DIR (an import never writes the compile cache, and core-level cache
/// properties on `import_model` risk an unsupported-property error).
fn import_plugin(plugin: &PluginConfig) -> PluginConfig {
    let mut p = plugin.clone();
    p.entries.retain(|(k, _)| k != "CACHE_DIR");
    p
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
    /// Wall-clock spent in the prefill phase (whole-prompt consumption up to
    /// and including the first generated token). TTFT proxy for the task-done
    /// log; also splits decode tok/s out of the blended rate.
    t_prefill: std::time::Duration,
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
    /// Chunked-prefill variant (static path only): a second compiled model —
    /// possibly on a different device — that consumes the prompt `seq` tokens
    /// per inference, sharing `static_kv`'s ring with the decode model.
    prefill: Option<StaticPrefill>,
    /// `--park-prefill`: drop the prefill CompiledModel after each task's
    /// prefill phase (freeing its resident weight copy) and re-create it on
    /// the next prefill. Trades a per-task reload for ~1× steady-state
    /// weight residency between requests.
    park_prefill: bool,
    /// Reload source for a parked prefill model.
    prefill_reload: Option<PrefillReload>,
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
        // host-side KV ring for this stage's layers. A multi-token frame is a
        // prefill chunk from a chunk-capable upstream stage.
        if self.static_kv.is_some() {
            if shape[1] > 1 {
                return self.run_relay_chunk(hidden, shape, position);
            }
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

    /// Relay/head handling of a multi-token prefill chunk `[1, r, hidden]` on
    /// the static path. With a chunked-prefill variant compiled, wide
    /// inferences cover the chunk (weights stream once per window) — an
    /// oversized incoming frame (an upstream stage exported with a wider
    /// `--static-prefill-seq`) is consumed in sub-chunks of this stage's own
    /// window rather than erroring. Without a variant, or when any row would
    /// sit past the KV window (`chunk_take` parity cap — only an uncapped /
    /// older sender produces such a frame), fall back to r sequential seq=1
    /// inferences: same math and ring state, just unamortized. A middle stage
    /// returns exactly the r real rows (`[1, r, X]`; pad-query garbage never
    /// leaves the stage); the head stage returns only the LAST real row
    /// (`[1, 1, X]`) since its caller argmaxes that row alone — skipping the
    /// conversion of r-1 unused vocab-wide rows. Either way the ring has
    /// absorbed r tokens and the 1-frame-in / 1-reply-out flow holds.
    fn run_relay_chunk(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        let r = shape[1];
        let h = shape[2];
        if h == 0 || hidden.len() != shape[0] * r * h {
            return Err(EngineError::Backend(format!(
                "chunk hidden len {} does not match shape {shape:?}",
                hidden.len()
            )));
        }
        let is_head = self.spec.is_last_stage;
        let past_len = self.static_kv.as_ref().map(|sk| sk.past_len).unwrap_or(0);
        let in_window = (position as usize).saturating_add(r) <= past_len + 1;
        let pf_seq = self.prefill.as_ref().map(|p| p.seq);

        if let (Some(pf_seq), true) = (pf_seq, in_window) {
            self.ensure_prefill_loaded()?;
            let mut rows: Vec<f32> = Vec::new();
            let mut x_out = 0usize;
            let mut off = 0usize;
            while off < r {
                let take = pf_seq.min(r - off);
                let final_sub = off + take == r;
                // Head stage: only the final sub-chunk's last row is ever
                // read — skip the whole-[1,C,vocab] output copy otherwise.
                let want = !is_head || final_sub;
                let (dt, oshape, bytes) = self.static_infer_chunk(
                    ChunkInput::Hidden(&hidden[off * h..(off + take) * h], h),
                    take,
                    position + off as i64,
                    want,
                )?;
                if want {
                    let x = *oshape.last().unwrap_or(&0);
                    let sz = float_elem_bytes(dt)?;
                    if x == 0 || bytes.len() < take * x * sz {
                        return Err(EngineError::Backend(format!(
                            "prefill chunk output too small: len {} shape {oshape:?} take {take}",
                            bytes.len()
                        )));
                    }
                    if is_head {
                        rows = bytes_to_f32(dt, &bytes[(take - 1) * x * sz..take * x * sz])?;
                    } else {
                        // Convert only the real rows — pads never leave.
                        rows.extend(bytes_to_f32(dt, &bytes[..take * x * sz])?);
                    }
                    x_out = x;
                }
                off += take;
            }
            let out_shape = if is_head {
                vec![1, 1, x_out]
            } else {
                vec![1, r, x_out]
            };
            return Ok((rows, out_shape));
        }

        // Fallback: no prefill variant on this stage, or an over-window frame
        // — consume the chunk one token at a time through the seq=1 decode
        // model (exact sliding-window semantics).
        let mut rows: Vec<f32> = Vec::new();
        let mut x = 0usize;
        for t in 0..r {
            let row = f32_to_f16_bytes(&hidden[t * h..(t + 1) * h]);
            let (dt, os, by) =
                self.static_infer(true, ShimDType::F16, &[1, 1, h], &row, position + t as i64)?;
            x = *os.last().unwrap_or(&0);
            if x == 0 || by.len() < x * float_elem_bytes(dt)? {
                return Err(EngineError::Backend(format!(
                    "static output too small: len {} shape {os:?}",
                    by.len()
                )));
            }
            let o = bytes_to_f32(dt, &by)?;
            if is_head {
                // Only the final token's row is read by step_last's argmax.
                if t == r - 1 {
                    rows = o[o.len() - x..].to_vec();
                }
            } else {
                if rows.is_empty() {
                    rows.reserve_exact(r * x);
                }
                rows.extend_from_slice(&o[o.len() - x..]);
            }
        }
        let out_shape = if is_head {
            vec![1, 1, x]
        } else {
            vec![1, r, x]
        };
        Ok((rows, out_shape))
    }

    /// Chunked-prefill first-stage inference: `chunk` prompt ids in one wide
    /// forward on the prefill runtime. With `want_output`, returns ONLY the
    /// real rows (`[1, take, X]` — pads are dropped before conversion); a
    /// single-stage engine passes `want_output = false` for non-final chunks,
    /// whose logits nothing reads.
    fn run_first_chunk(
        &mut self,
        chunk: &[i64],
        position: i64,
        want_output: bool,
    ) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        let take = chunk.len();
        debug!(take, position, "first-stage chunk: infer start");
        let (dtype, shape, bytes) =
            self.static_infer_chunk(ChunkInput::Ids(chunk), take, position, want_output)?;
        debug!(take, position, "first-stage chunk: infer+absorb done");
        if !want_output {
            return Ok((Vec::new(), Vec::new()));
        }
        let x = *shape.last().unwrap_or(&0);
        let sz = float_elem_bytes(dtype)?;
        if x == 0 || bytes.len() < take * x * sz {
            return Err(EngineError::Backend(format!(
                "prefill chunk output too small: len {} shape {shape:?} take {take}",
                bytes.len()
            )));
        }
        Ok((
            bytes_to_f32(dtype, &bytes[..take * x * sz])?,
            vec![1, take, x],
        ))
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
        debug!(?shape, position, "downstream send: start");
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
        debug!(position, "downstream send: done");
        Ok(())
    }

    fn recv_token_from_downstream(&mut self, prefill: bool) -> EngineResult<i32> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        // MID-TASK reply: we just sent a hidden state and the pipeline owes
        // us the sampled token back. Use the deadlined `recv_reply`, NOT the
        // idle-tolerant `recv` — a frame lost between stages (pipeline-leg
        // reset) would otherwise block this step loop forever with the task
        // slot held: the live-rig Item-5 wedge ("task active: 1, task done:
        // 0" all day). On timeout the error propagates to `step_first`'s
        // catch, which clears the active task and resets state — the slot
        // is freed and the next submit starts fresh. A prefill reply waits
        // on every remaining stage's whole-prompt compute, so it gets the
        // widened budget (see `recv_tensor_reply_prefill`) — a long prompt
        // on a slow stage must not read as a wedge.
        let (tensor, _) = self
            .block_on(async move {
                let mut guard = downstream.lock().await;
                if prefill {
                    guard.recv_reply_prefill().await
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

    fn recv_hidden_from_upstream(&mut self) -> EngineResult<(Vec<f32>, [usize; 3], Option<i64>)> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        // Static (NPU) shards send a leading I64 position tensor before the
        // hidden activation (see send_hidden_downstream). Each frame's payload
        // must match its shape*dtype, so we recv two separate tensors here.
        let want_pos = self.static_kv.is_some();
        debug!(want_pos, "upstream recv: waiting");
        let (pos_tensor, tensor) = self
            .block_on(async move {
                let mut guard = upstream.lock().await;
                // First frame of a task hop is an IDLE wait (bounded only by
                // the transport's much larger frame-idle ceiling — "no next
                // request yet" is fine). On the static path the
                // hidden tensor that must FOLLOW the position frame is a
                // mid-pair reply: once the pos frame arrived, the peer owes
                // the hidden promptly — deadline it so a half-sent pair
                // can't wedge the stage (see `recv_tensor_reply`).
                let (pos_tensor, t) = if want_pos {
                    let pos = guard.recv().await?.0;
                    tracing::debug!("upstream recv: position frame arrived");
                    let (t, _) = guard.recv_reply().await?;
                    (Some(pos), t)
                } else {
                    let (t, _) = guard.recv().await?;
                    (None, t)
                };
                Ok::<_, cascadia_transport::TransportError>((pos_tensor, t))
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        debug!("upstream recv: frames arrived");
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
                t_prefill: std::time::Duration::ZERO,
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
            // Failure recovery is also a going-idle transition: release the
            // prefill weight copy (no-op unless --park-prefill).
            self.park_prefill_model();
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
            let nt = if prefill && self.prefill.is_some() {
                // Chunked prefill: up to C prompt tokens per inference
                // instead of one — stage weights stream from DRAM once per
                // chunk (a ~C× cut in prefill weight traffic), and the wide
                // compute-bound matmuls run on the prefill device
                // (`--prefill-device`, e.g. the NPU) while `--device` keeps
                // the bandwidth-lean seq=1 decode loop. Chunks are CAPPED at
                // the KV window (`chunk_take`): any prompt tail whose rows
                // would sit past `past_len` steps one token at a time through
                // the decode model, so the cap itself introduces no divergence
                // from the seq=1 path in any regime, including over-window
                // prompts (the KV ring state matches; greedy tokens can still
                // fork at an argmax near-tie — see `assert_parity`). The token
                // from the final real row is the first generated token, exactly
                // as in the seq=1 path.
                let c = self.prefill.as_ref().map(|p| p.seq).unwrap_or(1);
                let past_len = self.static_kv.as_ref().map(|sk| sk.past_len).unwrap_or(0);
                let mut nt = 0i32;
                let mut idx = 0usize;
                while idx < tokens.len() {
                    let chunk_start = self.position;
                    let take = chunk_take(c, tokens.len() - idx, chunk_start as usize, past_len);
                    let ts = std::time::Instant::now();
                    let (token, wire) = if take >= 2 {
                        // Lazy so a prompt that never chunks (1-token, or
                        // fully over-window) skips a parked model's reload.
                        self.ensure_prefill_loaded()?;
                        let is_final_chunk = idx + take == tokens.len();
                        // Single-stage: nothing reads a non-final chunk's
                        // logits — skip the [1,C,vocab] copy + convert.
                        let want_output = !single_stage || is_final_chunk;
                        let (out, shape) = self.run_first_chunk(
                            &tokens[idx..idx + take],
                            chunk_start,
                            want_output,
                        )?;
                        let alpha = ts.elapsed();
                        self.position += take as i64;
                        idx += take;
                        if let Some(a) = self.active.as_mut() {
                            a.t_alpha_compute += alpha;
                        }
                        if single_stage && !is_final_chunk {
                            (nt, std::time::Duration::ZERO)
                        } else {
                            self.resolve_next_token_chunk(
                                &out,
                                &shape,
                                take,
                                single_stage,
                                chunk_start,
                            )?
                        }
                    } else {
                        // Window cap (or a 1-token remainder): one seq=1 step
                        // through the decode model — the exact sliding-window
                        // semantics for over-window rows.
                        let (out, shape) = self.run_first(&[tokens[idx]], chunk_start)?;
                        let alpha = ts.elapsed();
                        self.position += 1;
                        idx += 1;
                        if let Some(a) = self.active.as_mut() {
                            a.t_alpha_compute += alpha;
                        }
                        self.resolve_next_token(&out, &shape, single_stage, chunk_start, false)?
                    };
                    nt = token;
                    if let Some(a) = self.active.as_mut() {
                        a.t_wire += wire;
                    }
                }
                nt
            } else {
                let mut nt = 0i32;
                for &t in &tokens {
                    let position = self.position;
                    let ts = std::time::Instant::now();
                    let (out, shape) = self.run_first(&[t], position)?;
                    let alpha = ts.elapsed();
                    self.position += 1;
                    // Static prefill round-trips per prompt token — each reply
                    // covers one token's relay compute, so it is never a
                    // prefill-budget wait.
                    let (token, wire) =
                        self.resolve_next_token(&out, &shape, single_stage, position, false)?;
                    nt = token;
                    if let Some(a) = self.active.as_mut() {
                        a.t_alpha_compute += alpha;
                        a.t_wire += wire;
                    }
                }
                nt
            };
            if prefill {
                if let Some(a) = self.active.as_mut() {
                    a.t_prefill = a.started.elapsed();
                }
                // Prefill phase over: with --park-prefill, release the
                // prefill model's resident weights until the next task.
                self.park_prefill_model();
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
            // 1-token prompt prefill costs the same downstream as a decode
            // step — keep the strict deadline (mirrors step_middle's
            // shape[1] > 1) so wedge eviction stays fast.
            let (nt, wire) = self.resolve_next_token(
                &out,
                &shape,
                single_stage,
                position,
                prefill && tokens.len() > 1,
            )?;
            if let Some(a) = self.active.as_mut() {
                a.t_alpha_compute += alpha;
                a.t_wire += wire;
                if prefill {
                    a.t_prefill = a.started.elapsed();
                }
            }
            nt
        };

        self.emit_token(next_token)
    }

    /// Chunk analog of `resolve_next_token`. `out` carries exactly the real
    /// rows `[1, take, X]` (run_first_chunk drops pads before conversion):
    /// argmax the last row locally when single-stage, otherwise forward the
    /// rows downstream — preceded by the chunk-start position frame — and
    /// await the token. A chunk reply waits on whole-chunk compute across
    /// every remaining stage, so it gets the widened prefill budget when
    /// take > 1.
    fn resolve_next_token_chunk(
        &mut self,
        out: &[f32],
        shape: &[usize],
        take: usize,
        single_stage: bool,
        chunk_start: i64,
    ) -> EngineResult<(i32, std::time::Duration)> {
        if single_stage {
            return Ok((
                argmax_logits_row(out, shape, take - 1)?,
                std::time::Duration::ZERO,
            ));
        }
        let x = *shape.last().unwrap_or(&0);
        if x == 0 || out.len() != take * x {
            return Err(EngineError::Backend(format!(
                "chunk output size mismatch: len {} shape {shape:?} take {take}",
                out.len()
            )));
        }
        let ts = std::time::Instant::now();
        self.send_hidden_downstream(out, [1, take, x], chunk_start)?;
        let token = self.recv_token_from_downstream(take > 1)?;
        Ok((token, ts.elapsed()))
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
            // prefill_ms ~ TTFT (prompt consumption through the first
            // generated token); decode_tok_s excludes it so the two phases —
            // which may run on different devices — are visible separately.
            let prefill_ms = active.t_prefill.as_millis() as u64;
            let decode_s = elapsed.saturating_sub(active.t_prefill).as_secs_f64();
            let decode_tok_s = if active.generated.len() > 1 && decode_s > 0.0 {
                (active.generated.len() - 1) as f64 / decode_s
            } else {
                tok_s
            };
            info!(
                task = %task_id,
                tokens = active.generated.len(),
                elapsed_s = elapsed.as_secs_f64(),
                tok_s,
                prefill_ms,
                decode_tok_s,
                alpha_ms,
                wire_ms,
                other_ms,
                "ov-runtime task done"
            );
            if std::env::var("CASCADIA_PERF_DUMP").is_ok_and(|v| v == "1") {
                dump_decode_profile(&self.runtime);
            }
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

    /// Re-create a parked prefill compilation (no-op while loaded or without
    /// a variant). Reload cost is logged; with a CACHE_DIR-backed compile the
    /// reload comes from the OV blob/UMD cache rather than a cold compile.
    fn ensure_prefill_loaded(&mut self) -> EngineResult<()> {
        let parked = self.prefill.as_ref().is_some_and(|p| p.runtime.is_none());
        if !parked {
            return Ok(());
        }
        let src = self.prefill_reload.as_ref().ok_or_else(|| {
            EngineError::Backend("parked prefill model without a reload source".into())
        })?;
        let ts = std::time::Instant::now();
        let plugin = PluginConfig {
            entries: src.plugin_entries.clone(),
        };
        let prt = if let Some(blob) = &src.blob {
            OvRuntime::import_blob(blob, &src.device, &import_plugin(&plugin))
                .map_err(map_ov_err)
                .map_err(|e| {
                    EngineError::Backend(format!("prefill blob re-import on {}: {e}", src.device))
                })?
        } else {
            OvRuntime::compile(&src.xml, &src.device, &plugin)
                .map_err(map_ov_err)
                .map_err(|e| {
                    EngineError::Backend(format!("prefill reload on {}: {e}", src.device))
                })?
        };
        if let Some(pf) = self.prefill.as_mut() {
            pf.runtime = Some(prt);
        }
        info!(
            reload_ms = ts.elapsed().as_millis() as u64,
            device = %src.device,
            "prefill model reloaded (was parked)"
        );
        Ok(())
    }

    /// `--park-prefill`: drop the prefill CompiledModel, releasing its
    /// resident weight copy until the next prefill. Wiring and buffers stay.
    /// Cheap no-op when disabled, already parked, or no variant exists.
    fn park_prefill_model(&mut self) {
        if !self.park_prefill {
            return;
        }
        if let Some(pf) = self.prefill.as_mut() {
            if pf.runtime.take().is_some() {
                info!("prefill model parked (resident weights freed until next prefill)");
            }
        }
    }

    /// Run one chunked-prefill inference for `real` (<= C) tokens starting at
    /// absolute `position`, absorbing their K/V into the shared ring. Returns
    /// the primary output `[1, C, X]` bytes when `want_output`, else empty —
    /// a head stage's non-final chunks skip the whole-`[1,C,vocab]` shim copy
    /// since nothing reads it. The absorb (present.* reads) always happens.
    fn static_infer_chunk(
        &mut self,
        input: ChunkInput<'_>,
        real: usize,
        position: i64,
        want_output: bool,
    ) -> EngineResult<(ShimDType, Vec<usize>, Vec<u8>)> {
        // Split-borrow the ring and the prefill model: disjoint fields of
        // self, so ring buffers feed the prefill runtime with no staging copy
        // (beyond the shim's own per-input memcpy, common to every path).
        let (sk, pf) = match (self.static_kv.as_mut(), self.prefill.as_mut()) {
            (Some(sk), Some(pf)) => (sk, pf),
            _ => {
                return Err(EngineError::Backend(
                    "static_infer_chunk requires the static ring + a prefill variant".into(),
                ))
            }
        };
        if real == 0 || real > pf.seq {
            return Err(EngineError::Backend(format!(
                "chunk of {real} real tokens does not fit the prefill window (seq={})",
                pf.seq
            )));
        }
        if position < 0 {
            return Err(EngineError::Backend(format!(
                "negative chunk position {position}"
            )));
        }
        // Parked check FIRST: begin_token/write_prefill_mask mutate ring
        // state — advancing the cursor for a token whose K/V is never
        // absorbed would desync the ring if this error ever fires.
        let prt = pf.runtime.as_mut().ok_or_else(|| {
            EngineError::Backend(
                "prefill model is parked — callers must ensure_prefill_loaded first".into(),
            )
        })?;
        sk.begin_token(position as usize);
        sk.write_prefill_mask(&mut pf.mask_bytes, pf.context, real);

        // Primary input, converted/padded straight into the reusable
        // `primary_buf` (zero pads: masked out of attention, outputs unused,
        // KV never absorbed) — no intermediate per-chunk allocation.
        pf.primary_buf.clear();
        let in_main = match input {
            ChunkInput::Ids(ids) => {
                if ids.len() != real {
                    return Err(EngineError::Backend(format!(
                        "chunk ids len {} != real {real}",
                        ids.len()
                    )));
                }
                for t in ids {
                    pf.primary_buf.extend_from_slice(&t.to_le_bytes());
                }
                pf.primary_buf.resize(pf.seq * 8, 0);
                pf.ids_in.as_deref()
            }
            ChunkInput::Hidden(rows, hid) => {
                if hid == 0 || rows.len() != real * hid {
                    return Err(EngineError::Backend(format!(
                        "chunk hidden len {} != {real} rows of {hid}",
                        rows.len()
                    )));
                }
                for v in rows {
                    let h = half::f16::from_f32(*v);
                    pf.primary_buf.extend_from_slice(&h.to_bits().to_le_bytes());
                }
                pf.primary_buf.resize(pf.seq * hid * 2, 0);
                pf.hidden_in.as_deref()
            }
        }
        .ok_or_else(|| EngineError::Backend("prefill IR missing primary input".into()))?;

        // position_ids [1, C] = position..position+C; the pad tail continues
        // past the real tokens (masked + never absorbed, value irrelevant).
        pf.pos_buf.clear();
        for p in position..position + pf.seq as i64 {
            pf.pos_buf.extend_from_slice(&p.to_le_bytes());
        }

        match input {
            ChunkInput::Ids(_) => prt
                .set_input(in_main, ShimDType::I64, &[1, pf.seq], &pf.primary_buf)
                .map_err(map_ov_err)?,
            ChunkInput::Hidden(_, hid) => prt
                .set_input(in_main, ShimDType::F16, &[1, pf.seq, hid], &pf.primary_buf)
                .map_err(map_ov_err)?,
        }
        prt.set_input(
            &pf.attn_in,
            ShimDType::I64,
            &[1, pf.context],
            &pf.mask_bytes,
        )
        .map_err(map_ov_err)?;
        prt.set_input(&pf.pos_in, ShimDType::I64, &[1, pf.seq], &pf.pos_buf)
            .map_err(map_ov_err)?;
        let kv_shape = [1, sk.kv_heads, sk.past_len, sk.head_dim];
        for (li, layer) in pf.layers.iter().enumerate() {
            prt.set_input(&layer.key_in, sk.kv_dtype, &kv_shape, &sk.key_buf[li])
                .map_err(map_ov_err)?;
            prt.set_input(&layer.val_in, sk.kv_dtype, &kv_shape, &sk.val_buf[li])
                .map_err(map_ov_err)?;
        }
        prt.infer().map_err(map_ov_err)?;

        let primary = if want_output {
            prt.output(0).map_err(map_ov_err)?
        } else {
            (ShimDType::F16, Vec::new(), Vec::new())
        };

        // Absorb the `real` new tokens' K/V from each present.* (rows of
        // pf.context slots; chunk token t at slot past_len + t). Validate
        // length AND shape first, as the seq=1 path does.
        let expect = sk.kv_heads * pf.context * sk.head_dim * sk.elem_bytes;
        let want_shape = [1usize, sk.kv_heads, pf.context, sk.head_dim];
        for li in 0..pf.layers.len() {
            let (ko, vo) = (pf.layers[li].key_out, pf.layers[li].val_out);
            let (_, kshape, kpres) = prt.output(ko).map_err(map_ov_err)?;
            let (_, vshape, vpres) = prt.output(vo).map_err(map_ov_err)?;
            if kpres.len() != expect
                || vpres.len() != expect
                || kshape != want_shape
                || vshape != want_shape
            {
                return Err(EngineError::Backend(format!(
                    "prefill present.{li} mismatch: key shape={kshape:?} len={} val shape={vshape:?} \
                     len={}; expected shape {want_shape:?} ({expect} bytes f16). Check \
                     static_prefill_seq/static_prefill_context in stage_config.",
                    kpres.len(),
                    vpres.len(),
                )));
            }
            sk.absorb_layer_multi(li, false, &kpres, pf.context, position as usize, real);
            sk.absorb_layer_multi(li, true, &vpres, pf.context, position as usize, real);
        }
        Ok(primary)
    }

    fn step_last(&mut self) -> EngineResult<()> {
        let (hidden, shape, pos_opt) = self.recv_hidden_from_upstream()?;
        // A seq=1 static frame means this task's prefill chunks are done —
        // with --park-prefill, release the prefill model's weights now.
        if pos_opt.is_some() && shape[1] == 1 {
            self.park_prefill_model();
        }
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
        // Multi-token hidden = prefill work downstream: a stateful
        // whole-prompt frame, or (static path) a prefill CHUNK — either way
        // the token reply waits on multi-token compute across every remaining
        // stage, so it gets the widened prefill budget. Seq=1 frames (decode
        // steps, tokenwise prefill) keep the strict decode deadline.
        let prefill_reply = shape[1] > 1;
        // A seq=1 static frame means this task's prefill chunks are done —
        // with --park-prefill, release the prefill model's weights now.
        if pos_opt.is_some() && shape[1] == 1 {
            self.park_prefill_model();
        }
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
        let token = self.recv_token_from_downstream(prefill_reply)?;
        self.send_token_to_upstream(token)?;
        Ok(())
    }
}

impl Engine for OvRuntimeEngine {
    fn warmup(&mut self) {
        if !(self.spec.is_first_stage) {
            // Static relay/head stages: warm both compiled models with a
            // zeroed hidden row so one-time device init (graph JIT/upload —
            // seconds on a cold NPU) doesn't land on the first request's
            // prefill. The garbage KV is discarded by the position-0 ring
            // reset; nothing crosses the wire (run_relay* is stage-local).
            if self.static_kv.is_some() && self.hidden_size > 0 {
                let zero = vec![0f32; self.hidden_size];
                let hs = self.hidden_size;
                if let Err(e) = self.run_relay(&zero, [1, 1, hs], 0) {
                    warn!(error = %e, "ov-runtime relay warmup (decode model) failed");
                }
                if self.prefill.is_some() {
                    if let Err(e) = self.run_relay_chunk(&zero, [1, 1, hs], 0) {
                        warn!(error = %e, "ov-runtime relay warmup (prefill model) failed");
                    }
                }
                if let Some(sk) = self.static_kv.as_mut() {
                    sk.reset();
                }
                self.position = 0;
                info!("ov-runtime warmup ok (static relay)");
            } else {
                info!("ov-runtime warmup skipped on non-first stage");
            }
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
                // Warm the chunked-prefill model too (a 1-token padded
                // chunk): its one-time first-inference cost otherwise
                // inflates the first real request's TTFT — the very metric
                // the phase split exists to improve.
                if self.prefill.is_some() && !warm.is_empty() {
                    // Ensure covers a future re-warm with --park-prefill:
                    // today warmup precedes any park, but that ordering is
                    // an invariant nothing else enforces.
                    if let Err(e) = self
                        .ensure_prefill_loaded()
                        .and_then(|_| self.run_first_chunk(warm, 0, true).map(|_| ()))
                    {
                        warn!(error = %e, "ov-runtime warmup (prefill model) failed");
                    }
                }
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
            Err(e) => match self.step_warn.on_failure(std::time::Instant::now()) {
                Some(StepWarn::First) => warn!(error = %e, "ov-runtime step failed"),
                Some(StepWarn::StillFailing { suppressed }) => {
                    warn!(error = %e, suppressed, "ov-runtime step still failing")
                }
                None => {}
            },
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
            // Cancel is a "task over, going idle" transition — exactly what
            // --park-prefill exists for. Without this, a cancelled task
            // leaves the second weight copy resident until the NEXT task's
            // prefill completes.
            self.park_prefill_model();
        }
    }
}

/// CASCADIA_PERF_DUMP=1 (spike diagnostics): after a task finishes, print an
/// aggregated per-node profile of the decode model's LAST inference — one
/// representative decode token. Needs the model compiled with PERF_COUNT=YES
/// (pass it via the plugin properties). Times inflate under profiling; use
/// for ATTRIBUTION between node kinds, not absolute tok/s math.
fn dump_decode_profile(runtime: &OvRuntime) {
    let raw = match runtime.profiling() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("perf-dump unavailable: {e} (compile with PERF_COUNT=YES)");
            return;
        }
    };
    let mut by_type: std::collections::HashMap<String, (u64, u64, u32)> =
        std::collections::HashMap::new();
    let mut nodes: Vec<(String, String, u64)> = Vec::new();
    let mut total_us = 0u64;
    for line in raw.lines() {
        let mut it = line.split('\t');
        let (name, ntype, etype, real) = (
            it.next().unwrap_or(""),
            it.next().unwrap_or(""),
            it.next().unwrap_or(""),
            it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0),
        );
        let cpu = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        total_us += real;
        let e = by_type.entry(ntype.to_string()).or_default();
        e.0 += real;
        e.1 += cpu;
        e.2 += 1;
        if real > 0 {
            nodes.push((name.to_string(), format!("{ntype}/{etype}"), real));
        }
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by_key(|(_, (real, _, _))| std::cmp::Reverse(*real));
    eprintln!("perf-dump: decode-infer total {total_us} us by node type:");
    for (ntype, (real, cpu, count)) in rows.iter().take(14) {
        eprintln!("  {ntype:<28} real {real:>8} us  cpu {cpu:>8} us  x{count}");
    }
    nodes.sort_by_key(|(_, _, real)| std::cmp::Reverse(*real));
    eprintln!("perf-dump: top nodes:");
    for (name, kind, real) in nodes.iter().take(10) {
        let short: String = name
            .chars()
            .rev()
            .take(48)
            .collect::<Vec<_>>()
            .iter()
            .rev()
            .collect();
        eprintln!("  {real:>8} us  {kind:<28} ...{short}");
    }
}

/// Resolve canonical input-port names ("input_ids", "attention_mask", ...)
/// of a compiled runtime via its alias lists. v5 IRs export ports under
/// conventional names but the IR's primary name is sometimes an internal node
/// id; the alias list carries the canonical name. Shared by the decode and
/// chunked-prefill compilations (same stage graph → same port names) and by
/// dist_spec's v5 loader — one resolver, so a port rename lands everywhere.
pub(crate) fn resolve_canonical_inputs(
    runtime: &OvRuntime,
) -> EngineResult<std::collections::HashMap<String, String>> {
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
    Ok(canonical_inputs)
}

/// Resolve a stateless static compilation's explicit `past_key_values.*`
/// input ports + `present.*` output indices, discovered contiguously from
/// layer 0, with a cross-check so a gap / renamed / folded port is a hard
/// error instead of silently building fewer layers. `what` labels errors
/// ("static shard" / "chunked-prefill variant").
fn resolve_static_layers(runtime: &OvRuntime, what: &str) -> EngineResult<Vec<StaticKvLayer>> {
    let n_inputs = runtime.input_count();
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
        let key_out =
            key_out.ok_or_else(|| EngineError::Backend(format!("{what} missing {kout_s}")))?;
        let val_out =
            val_out.ok_or_else(|| EngineError::Backend(format!("{what} missing {vout_s}")))?;
        layers.push(StaticKvLayer {
            key_in,
            val_in,
            key_out,
            val_out,
        });
    }
    if layers.is_empty() {
        return Err(EngineError::Backend(format!(
            "{what}: no past_key_values.* inputs found"
        )));
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
            "{what}: resolved {} contiguous KV layers ({} ports) but the IR has \
             {kv_input_ports} past_key_values.* input ports — a layer port is missing or \
             misnamed (gap in the past_key_values.N sequence)",
            layers.len(),
            layers.len() * 2,
        )));
    }
    Ok(layers)
}

// -------- Builder --------

#[derive(Default)]
pub struct OvRuntimeBuilder {
    pub pipeline_dir: PathBuf,
    pub rank: u32,
    pub total: u32,
    pub device: String,
    /// Device for the chunked-prefill IR variant (static shards only): the
    /// phase split — e.g. `NPU` here with `--device CPU` runs the wide
    /// compute-bound prefill on the NPU and the bandwidth-bound seq=1 decode
    /// on the CPU, sharing one host KV ring. Default (None) compiles the
    /// prefill variant (when the export has one) on `device`, which still
    /// buys the ~C× prefill weight-stream amortization on a single device.
    pub prefill_device: Option<String>,
    /// Ignore the export's chunked-prefill variant and prefill one token per
    /// step (the pre-variant behavior); conflicts with `prefill_device`.
    pub disable_chunked_prefill: bool,
    /// `--park-prefill`: drop the prefill CompiledModel after each task's
    /// prefill (freeing its resident weight copy — the structural cost of
    /// the two-model split) and re-create it on the next prefill from the
    /// blob cache. For memory-tight stages; costs a per-task reload.
    pub park_prefill: bool,
    /// `--gemv-offload` (SPIKE): compile the decode model with the
    /// CascadiaInt4Gemv extension pass — sym-INT4 weight matmuls execute
    /// straight from the mmapped .bin through a custom op instead of a
    /// plugin-repacked resident copy (~1x steady-state decode weights).
    /// Stateless static exports on a CPU device only.
    pub gemv_offload: bool,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    /// Extra `(key, value)` OV plugin properties plumbed verbatim from the CLI.
    pub ov_properties: Vec<(String, String)>,
    runtime: Option<OvRuntime>,
    prefill_runtime: Option<OvRuntime>,
    prefill_reload: Option<PrefillReload>,
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
    /// (seq, context) of the chunked-prefill IR variant, when the stage
    /// exports one (stage_config.static_prefill_*) and it isn't disabled.
    static_prefill_params: Option<(u32, u32)>,
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

    /// Compile the chunked-prefill variant on a different device than
    /// `device` — the NPU+CPU (or GPU+CPU) phase split.
    pub fn with_prefill_device(mut self, device: impl Into<String>) -> Self {
        self.prefill_device = Some(device.into());
        self
    }
    /// Park the prefill model between prefills (see `park_prefill`).
    pub fn with_prefill_parking(mut self, park: bool) -> Self {
        self.park_prefill = park;
        self
    }
    /// Decode-side INT4 GEMV offload (see `gemv_offload`).
    pub fn with_gemv_offload(mut self, offload: bool) -> Self {
        self.gemv_offload = offload;
        self
    }
    /// Skip the export's chunked-prefill variant (prefill one token per step).
    pub fn with_chunked_prefill_disabled(mut self, disabled: bool) -> Self {
        self.disable_chunked_prefill = disabled;
        self
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

        // Chunked-prefill variant: validate coherence at load, not first
        // inference. The variant is usable only when its past-KV length
        // equals the decode variant's (context - 1) — that identity is what
        // lets both compiled models share one host KV ring.
        let prefill_xml = stage_dir.join("openvino_prefill_model.xml");
        self.static_prefill_params = match (self.static_params, stage_cfg.static_prefill_seq) {
            (Some((ctx, _, _)), Some(pseq)) if !self.disable_chunked_prefill => {
                let pctx = stage_cfg.static_prefill_context.unwrap_or(0);
                if pseq < 2 || pctx != ctx - 1 + pseq {
                    return Err(EngineError::InvalidConfig(format!(
                        "static_prefill_seq={pseq} / static_prefill_context={pctx} inconsistent: \
                         need seq >= 2 and context = (static_context - 1) + seq = {} so the \
                         prefill and decode variants share one past-KV shape — re-export with \
                         --static-prefill-seq",
                        ctx - 1 + pseq
                    )));
                }
                if !prefill_xml.exists() {
                    return Err(EngineError::InvalidConfig(format!(
                        "stage_config advertises a chunked-prefill variant \
                         (static_prefill_seq={pseq}) but {} is missing — re-export the stage",
                        prefill_xml.display()
                    )));
                }
                events.push(LoadProgress::message(format!(
                    "chunked-prefill variant: seq={pseq} context={pctx}"
                )));
                Some((pseq, pctx))
            }
            _ => None,
        };
        // --gemv-offload (spike): the offloaded matmuls run on the CPU
        // plugin's evaluate() fallback, and the rewrite targets the stateless
        // static IR's sym-INT4 pattern — gate both up front.
        if self.gemv_offload {
            if stage_cfg.stateful {
                return Err(EngineError::ShardRejected(
                    "--gemv-offload requires a stateless static export \
                     (tools/export_shards.py --target npu)"
                        .into(),
                ));
            }
            if !self.device.trim().to_ascii_uppercase().starts_with("CPU") {
                return Err(EngineError::ShardRejected(format!(
                    "--gemv-offload executes offloaded matmuls on the CPU evaluate() \
                     fallback — --device must be CPU (got {})",
                    self.device
                )));
            }
        }

        if (self.prefill_device.is_some() || self.park_prefill)
            && self.static_prefill_params.is_none()
        {
            let flag = if self.prefill_device.is_some() {
                "--prefill-device"
            } else {
                "--park-prefill"
            };
            return Err(EngineError::ShardRejected(if stage_cfg.stateful {
                format!(
                    "{flag} requires a stateless static export: re-export with \
                     tools/export_shards.py --target npu --static-prefill-seq N (the stateful \
                     path keeps KV inside OV state, which two devices cannot share)"
                )
            } else if self.disable_chunked_prefill {
                format!("{flag} conflicts with --no-chunked-prefill")
            } else {
                format!(
                    "{flag} requires a chunked-prefill IR variant: re-export this \
                     pipeline with --static-prefill-seq N"
                )
            }));
        }

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
                // The chunk path also couples stages GEOMETRICALLY: each
                // stage derives its KV window (past_len) and the in-window /
                // tokenwise-fallback decision from its OWN static_context. A
                // pipeline stitched from different exports would silently
                // apply different sliding windows per stage — corrupted
                // output, no error. Fail fast instead. (Differing
                // static_prefill_seq alone is fine — relays sub-chunk.)
                if cfg.static_context != stage_cfg.static_context {
                    return Err(EngineError::ShardRejected(format!(
                        "pipeline is not homogeneous: stage_{} static_context={:?} but \
                         stage_{r} static_context={:?} — stages from different exports \
                         apply different KV windows (re-export the whole pipeline)",
                        self.rank, stage_cfg.static_context, cfg.static_context
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
        let runtime = if self.gemv_offload {
            // The custom op's member tensors don't survive blob
            // serialization — strip CACHE_DIR for THIS compile only (the
            // prefill compile below still caches; CPU compiles are fast).
            let mut p = self.plugin();
            let had_cache = p.entries.iter().any(|(k, _)| k == "CACHE_DIR");
            if had_cache {
                p.entries.retain(|(k, _)| k != "CACHE_DIR");
                warn!(
                    "--gemv-offload: decode compile skips CACHE_DIR (custom-op weights \
                     don't survive blob serialization); the prefill compile still caches"
                );
            }
            let (rt, offloaded) = OvRuntime::compile_gemv_offload(
                xml_path.to_str().unwrap_or_default(),
                &self.device,
                &p,
            )
            .map_err(map_ov_err)?;
            events.push(LoadProgress::message(format!(
                "gemv-offload: {offloaded} sym-INT4 matmuls execute from the .bin mmap \
                 (no plugin-resident weight copy)"
            )));
            if offloaded == 0 {
                warn!(
                    "--gemv-offload matched 0 matmuls — the IR does not carry the \
                     expected NNCF sym-INT4 pattern; decode runs fully stock"
                );
            }
            rt
        } else if let Some(blob) = npu_aot_blob(&xml_path, &self.device) {
            events.push(LoadProgress::message(format!(
                "importing precompiled NPU blob (AOT, no on-box compile): {}",
                blob.display()
            )));
            OvRuntime::import_blob(
                blob.to_str().unwrap_or_default(),
                &self.device,
                &import_plugin(&plugin),
            )
            .map_err(map_ov_err)?
        } else {
            OvRuntime::compile(xml_path.to_str().unwrap_or_default(), &self.device, &plugin)
                .map_err(map_ov_err)?
        };
        self.input_names = runtime.input_names().map_err(map_ov_err)?;
        self.runtime = Some(runtime);

        // Second compilation for the chunked-prefill variant — on
        // --prefill-device when given (the NPU+CPU phase split), else on
        // --device (single-device chunked prefill). Note this holds a second
        // copy of the stage weights resident (a compiled model owns its own
        // weight allocation per device) — the price of the split; KV is NOT
        // duplicated (the host ring is shared).
        if let Some((pseq, pctx)) = self.static_prefill_params {
            let pdev = self
                .prefill_device
                .clone()
                .unwrap_or_else(|| self.device.clone());
            events.push(LoadProgress::message(format!(
                "compiling chunked-prefill variant (seq={pseq}, context={pctx}) on {pdev}"
            )));
            // Same-device chunked prefill shares the full decode plugin
            // config. When the phase split targets a DIFFERENT device, only
            // CACHE_DIR carries over: every other entry (KV_CACHE_PRECISION,
            // DYNAMIC_QUANTIZATION_GROUP_SIZE, --ov-property keys) is decode-
            // device tuning — a foreign plugin either rejects the key at
            // compile_model (failing the whole load) or silently mis-tunes
            // the prefill graph.
            let pplugin = if pdev == self.device {
                plugin.clone()
            } else {
                PluginConfig {
                    entries: plugin
                        .entries
                        .iter()
                        .filter(|(k, _)| k == "CACHE_DIR")
                        .cloned()
                        .collect(),
                }
            };
            let pxml = prefill_xml.to_str().unwrap_or_default().to_string();
            let pblob = npu_aot_blob(&prefill_xml, &pdev);
            let prt = if let Some(blob) = &pblob {
                events.push(LoadProgress::message(format!(
                    "importing precompiled NPU prefill blob (AOT): {}",
                    blob.display()
                )));
                OvRuntime::import_blob(
                    blob.to_str().unwrap_or_default(),
                    &pdev,
                    &import_plugin(&pplugin),
                )
                .map_err(map_ov_err)
                .map_err(|e| EngineError::Backend(format!("prefill blob import on {pdev}: {e}")))?
            } else {
                OvRuntime::compile(&pxml, &pdev, &pplugin)
                    .map_err(map_ov_err)
                    .map_err(|e| {
                        EngineError::Backend(format!("chunked-prefill variant on {pdev}: {e}"))
                    })?
            };
            self.prefill_runtime = Some(prt);
            if self.park_prefill && self.cache_dir.is_none() {
                warn!(
                    "--park-prefill without --ov-cache-dir: every reload is a full \
                     cold compile (measured ~minutes on NPU) instead of a blob-cache \
                     import — set --ov-cache-dir"
                );
            }
            // Everything a parked model needs to come back (--park-prefill);
            // the PREFILL device's plugin entries so a reload both hits the
            // compile cache and never re-introduces decode-device tuning.
            self.prefill_reload = Some(PrefillReload {
                xml: pxml,
                device: pdev,
                plugin_entries: pplugin.entries.clone(),
                blob: pblob.map(|b| b.to_string_lossy().into_owned()),
            });
        }

        events.push(LoadProgress::message(
            "loading rotary + tokenizer".to_string(),
        ));

        // Rotary from the model's HF config.json. Look in the pipeline
        // tokenizer dir first (the sharder writes config.json there);
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
        let canonical_inputs = resolve_canonical_inputs(&runtime)?;

        // Stateless static-shape (NPU) shards: resolve the explicit
        // past_key_values.* inputs + present.* outputs and allocate the
        // host-side KV ring. The primary output (logits on the head stage,
        // hidden_states otherwise) is output index 0 — no alias guess needed.
        let static_kv = if let Some((ctx, kvh, hd)) = self.static_params {
            let layers = resolve_static_layers(&runtime, "static shard")?;
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

        // Chunked-prefill variant: a second compiled model (possibly on a
        // different device) with its own port wiring, sharing the ring above.
        // Wiring is re-resolved against the prefill compilation rather than
        // assumed identical, so a divergent variant fails loud at build —
        // including the primary-input KIND (an embed-stage prefill IR built
        // around hidden_states, or vice versa, is not a sibling of one
        // export and would otherwise only fail at the first chunk).
        let prefill = match self.prefill_runtime {
            None => None,
            Some(prt) => {
                let (pseq, pctx) = self.static_prefill_params.ok_or_else(|| {
                    EngineError::Backend(
                        "inconsistent builder state: prefill runtime without prefill params".into(),
                    )
                })?;
                let sk = static_kv.as_ref().ok_or_else(|| {
                    EngineError::Backend(
                        "inconsistent builder state: prefill runtime without static ring".into(),
                    )
                })?;
                let pcanon = resolve_canonical_inputs(&prt)?;
                let players = resolve_static_layers(&prt, "chunked-prefill variant")?;
                if players.len() != sk.layers.len() {
                    return Err(EngineError::Backend(format!(
                        "chunked-prefill variant has {} KV layers but the decode variant has \
                         {} — the two IRs are not siblings of one export",
                        players.len(),
                        sk.layers.len()
                    )));
                }
                let attn_in = pcanon.get("attention_mask").cloned().ok_or_else(|| {
                    EngineError::Backend("prefill IR missing attention_mask input".into())
                })?;
                let pos_in = pcanon.get("position_ids").cloned().ok_or_else(|| {
                    EngineError::Backend("prefill IR missing position_ids input".into())
                })?;
                let ids_in = pcanon.get("input_ids").cloned();
                let hidden_in = pcanon.get("hidden_states").cloned();
                if ids_in.is_none() && hidden_in.is_none() {
                    return Err(EngineError::Backend(
                        "prefill IR has neither input_ids nor hidden_states input".into(),
                    ));
                }
                if ids_in.is_some() != sk.ids_in.is_some()
                    || hidden_in.is_some() != sk.hidden_in.is_some()
                {
                    return Err(EngineError::Backend(format!(
                        "prefill/decode variants disagree on the primary input kind \
                         (decode: {}, prefill: {}) — the two IRs are not siblings of one export",
                        if sk.ids_in.is_some() {
                            "input_ids"
                        } else {
                            "hidden_states"
                        },
                        if ids_in.is_some() {
                            "input_ids"
                        } else {
                            "hidden_states"
                        },
                    )));
                }
                Some(StaticPrefill {
                    runtime: Some(prt),
                    seq: pseq as usize,
                    context: pctx as usize,
                    ids_in,
                    hidden_in,
                    attn_in,
                    pos_in,
                    layers: players,
                    mask_bytes: vec![0u8; pctx as usize * 8],
                    primary_buf: Vec::new(),
                    pos_buf: Vec::with_capacity(pseq as usize * 8),
                })
            }
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
            prefill,
            park_prefill: self.park_prefill,
            prefill_reload: self.prefill_reload,
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

    // ---- chunked-prefill ring math ----

    fn test_ring(past_len: usize, kv_heads: usize, head_dim: usize, layers: usize) -> StaticKv {
        let elem = 2usize;
        let layer_bytes = kv_heads * past_len * head_dim * elem;
        StaticKv {
            past_len,
            context: past_len + 1,
            kv_heads,
            head_dim,
            elem_bytes: elem,
            kv_dtype: ShimDType::F16,
            ids_in: Some("input_ids".into()),
            hidden_in: None,
            attn_in: "attention_mask".into(),
            pos_in: "position_ids".into(),
            layers: (0..layers)
                .map(|_| StaticKvLayer {
                    key_in: "k".into(),
                    val_in: "v".into(),
                    key_out: 1,
                    val_out: 2,
                })
                .collect(),
            key_buf: vec![vec![0u8; layer_bytes]; layers],
            val_buf: vec![vec![0u8; layer_bytes]; layers],
            valid: 0,
            mask_bytes: vec![0u8; (past_len + 1) * 8],
        }
    }

    /// Deterministic per-byte KV pattern for token `tok` (distinct per head
    /// byte offset and per k/v so a wrong-slot or crossed read shows up).
    fn tok_byte(tok: usize, h: usize, b: usize, is_value: bool) -> u8 {
        ((tok * 31 + h * 7 + b * 3 + if is_value { 131 } else { 0 }) % 251) as u8
    }

    /// A seq=1 `present` buffer: the `c=1, real=1` case of `present_chunk`,
    /// aliased so there is exactly one definition of the poisoned layout.
    fn present_seq1(ring: &StaticKv, tok: usize, is_value: bool) -> Vec<u8> {
        present_chunk(ring, 1, tok, 1, is_value)
    }

    /// A chunk `present` buffer (rows of `past_len+c` slots): chunk tokens
    /// `start..start+real` at slots `past_len..past_len+real`, poison
    /// elsewhere (including the pad-token slots `real..c`).
    fn present_chunk(
        ring: &StaticKv,
        c: usize,
        start: usize,
        real: usize,
        is_value: bool,
    ) -> Vec<u8> {
        let slot = ring.head_dim * ring.elem_bytes;
        let ctx = ring.past_len + c;
        let mut p = vec![0xEEu8; ring.kv_heads * ctx * slot];
        for h in 0..ring.kv_heads {
            for t in 0..real {
                let base = h * ctx * slot + (ring.past_len + t) * slot;
                for b in 0..slot {
                    p[base + b] = tok_byte(start + t, h, b, is_value);
                }
            }
        }
        p
    }

    /// The load-bearing equivalence: absorbing a prompt through
    /// `absorb_layer_multi` in chunks of C must leave the ring byte-identical
    /// to absorbing it one token at a time through `absorb_layer`, including
    /// after the window fills and slides. This is what makes the chunked
    /// (possibly other-device) prefill hand exactly the same KV state to the
    /// seq=1 decode loop as the legacy path.
    #[test]
    fn chunked_absorb_matches_sequential() {
        // past_len=4 forces sliding early; prompt=11 with C=3 exercises a
        // full chunk before the slide, chunks straddling the slide boundary,
        // and a partial tail chunk (11 = 3+3+3+2).
        for (past_len, c, n) in [(4usize, 3usize, 11usize), (8, 4, 6), (5, 2, 5), (6, 6, 13)] {
            let layers = 2;
            let mut seq = test_ring(past_len, 2, 3, layers);
            for t in 0..n {
                seq.begin_token(t);
                for li in 0..layers {
                    let k = present_seq1(&seq, t, false);
                    let v = present_seq1(&seq, t, true);
                    seq.absorb_layer(li, false, &k);
                    seq.absorb_layer(li, true, &v);
                }
            }

            let mut chk = test_ring(past_len, 2, 3, layers);
            let mut pos = 0usize;
            while pos < n {
                let take = c.min(n - pos);
                chk.begin_token(pos);
                for li in 0..layers {
                    let k = present_chunk(&chk, c, pos, take, false);
                    let v = present_chunk(&chk, c, pos, take, true);
                    chk.absorb_layer_multi(li, false, &k, past_len + c, pos, take);
                    chk.absorb_layer_multi(li, true, &v, past_len + c, pos, take);
                }
                pos += take;
            }

            assert_eq!(
                seq.key_buf, chk.key_buf,
                "key ring diverged (past_len={past_len} c={c} n={n})"
            );
            assert_eq!(
                seq.val_buf, chk.val_buf,
                "value ring diverged (past_len={past_len} c={c} n={n})"
            );
        }
    }

    #[test]
    fn prefill_mask_layout() {
        let mut ring = test_ring(6, 1, 2, 1);
        let c = 4;
        let pctx = ring.past_len + c;
        let mut buf = Vec::new();

        let read = |buf: &[u8]| -> Vec<i64> {
            buf.chunks_exact(8)
                .map(|ch| i64::from_le_bytes(ch.try_into().unwrap()))
                .collect()
        };

        // Empty ring, 2 real tokens of a 4-wide chunk: only chunk slots 0..2.
        ring.begin_token(0);
        ring.write_prefill_mask(&mut buf, pctx, 2);
        assert_eq!(read(&buf), [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]);

        // 3 past tokens visible, full chunk: past 0..3 + all 4 chunk slots.
        ring.begin_token(3);
        ring.write_prefill_mask(&mut buf, pctx, 4);
        assert_eq!(read(&buf), [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]);

        // Overflowed position clamps to past_len (window full).
        ring.begin_token(9);
        ring.write_prefill_mask(&mut buf, pctx, 1);
        assert_eq!(read(&buf), [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]);
    }

    /// The decode mask must be exactly the chunk mask at real=1 with
    /// ctx=past_len+1 — the property that lets both paths share
    /// `fill_static_mask` without silent divergence.
    #[test]
    fn decode_mask_is_chunk_mask_real_one() {
        let mut ring = test_ring(5, 1, 2, 1);
        for pos in [0usize, 2, 5, 9] {
            ring.begin_token(pos);
            ring.write_mask_bytes();
            let decode = ring.mask_bytes.clone();
            let mut chunk = Vec::new();
            ring.write_prefill_mask(&mut chunk, ring.past_len + 1, 1);
            assert_eq!(decode, chunk, "mask writers diverged at position {pos}");
        }
    }

    /// The parity cap: every chunk row's absolute position must stay
    /// <= past_len, because a single chunk-wide mask cannot express the
    /// per-token eviction the seq=1 sliding window performs. Rows past the
    /// boundary step tokenwise.
    #[test]
    fn chunk_take_caps_at_the_kv_window() {
        let past_len = 8;
        // Plenty of window: bounded by C and remaining.
        assert_eq!(chunk_take(4, 100, 0, past_len), 4);
        assert_eq!(chunk_take(4, 3, 0, past_len), 3);
        // Approaching the boundary: rows 6,7,8 are the last in-window ones.
        assert_eq!(chunk_take(4, 100, 6, past_len), 3);
        // Exactly at the boundary row: one row still fits (position ==
        // past_len sees full history in both paths; eviction happens in
        // absorb, after the forward).
        assert_eq!(chunk_take(4, 100, 8, past_len), 1);
        // Past the boundary: no chunk rows — caller steps tokenwise.
        assert_eq!(chunk_take(4, 100, 9, past_len), 0);
        assert_eq!(chunk_take(4, 100, 1000, past_len), 0);
    }

    /// AOT blob import triggers ONLY for the NPU device and only when a
    /// `.blob` sibling of the IR exists — CPU/GPU always compile (blobs are
    /// device-specific), and an absent blob falls back to the compiler.
    #[test]
    fn npu_aot_blob_selects_only_npu_with_blob_present() {
        let dir = std::env::temp_dir().join(format!("aot-blob-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let xml = dir.join("openvino_model.xml");
        std::fs::write(&xml, "x").unwrap();

        // No blob on disk: never selected.
        assert_eq!(npu_aot_blob(&xml, "NPU"), None);

        let blob = dir.join("openvino_model.blob");
        std::fs::write(&blob, "b").unwrap();
        // Blob present: NPU (any casing/config suffix) imports it.
        assert_eq!(npu_aot_blob(&xml, "NPU"), Some(blob.clone()));
        assert_eq!(npu_aot_blob(&xml, " npu "), Some(blob.clone()));
        // Non-NPU devices always compile.
        assert_eq!(npu_aot_blob(&xml, "CPU"), None);
        assert_eq!(npu_aot_blob(&xml, "GPU"), None);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A blob older than its IR is refused (fall back to a fresh compile):
    /// serving a stale compile's weights is the "worst failure mode" the guard
    /// exists to prevent. The happy-path test above never makes the blob older,
    /// so this pins the `bm < xm` branch specifically.
    #[test]
    fn npu_aot_blob_rejects_blob_older_than_ir() {
        use std::time::{Duration, SystemTime};
        let dir = std::env::temp_dir().join(format!("aot-blob-stale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let xml = dir.join("openvino_model.xml");
        let blob = dir.join("openvino_model.blob");
        std::fs::write(&xml, "x").unwrap();
        std::fs::write(&blob, "b").unwrap();
        // Backdate the blob 60 s behind the IR it claims to be a compile of.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&blob)
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(60))
            .unwrap();
        assert_eq!(npu_aot_blob(&xml, "NPU"), None);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn argmax_row_selects_requested_row() {
        // 3 rows of vocab 4; best of row 1 is index 2; row 2 (pad garbage)
        // would pick index 3 — the chunk path must not read it.
        let logits = [
            0.0, 9.0, 0.0, 0.0, // row 0
            0.0, 0.0, 7.0, 0.0, // row 1
            0.0, 0.0, 0.0, 8.0, // row 2
        ];
        let shape = [1usize, 3, 4];
        assert_eq!(argmax_logits_row(&logits, &shape, 0).unwrap(), 1);
        assert_eq!(argmax_logits_row(&logits, &shape, 1).unwrap(), 2);
        assert_eq!(argmax_logits_row(&logits, &shape, 2).unwrap(), 3);
        assert!(argmax_logits_row(&logits, &shape, 3).is_err());
    }

    // ---- load-time validation of the prefill variant ----

    fn write_stage_dir(tag: &str, stage_cfg: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cascadia-prefill-test-{}-{tag}",
            std::process::id()
        ));
        let stage = dir.join("stage_0");
        std::fs::create_dir_all(&stage).unwrap();
        std::fs::write(
            dir.join("pipeline_config.json"),
            r#"{"model_id":"m","num_stages":1,"num_layers":2,"hidden_size":8}"#,
        )
        .unwrap();
        std::fs::write(stage.join("stage_config.json"), stage_cfg).unwrap();
        dir
    }

    #[tokio::test]
    async fn prefill_device_on_stateful_shard_rejected() {
        let dir = write_stage_dir(
            "stateful",
            r#"{"layer_start":0,"layer_end":2,"has_embed":true,"has_head":true,"stateful":true}"#,
        );
        let mut b = OvRuntimeBuilder::new(&dir, 0, 1, "CPU").with_prefill_device("NPU");
        let err = b
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .err()
            .expect("stateful + --prefill-device must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("--prefill-device"), "got: {msg}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn inconsistent_prefill_context_rejected() {
        // static_context=8 → decode past_len=7; seq=4 needs context 11, not 12.
        let dir = write_stage_dir(
            "badctx",
            r#"{"layer_start":0,"layer_end":2,"has_embed":true,"has_head":true,
                "stateful":false,"static_seq":1,"static_context":8,
                "static_prefill_seq":4,"static_prefill_context":12,
                "num_kv_heads":2,"head_dim":4}"#,
        );
        let mut b = OvRuntimeBuilder::new(&dir, 0, 1, "CPU");
        let err = b
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .err()
            .expect("mismatched prefill context must be rejected");
        assert!(err.to_string().contains("static_prefill"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn advertised_prefill_variant_missing_file_rejected() {
        let dir = write_stage_dir(
            "nofile",
            r#"{"layer_start":0,"layer_end":2,"has_embed":true,"has_head":true,
                "stateful":false,"static_seq":1,"static_context":8,
                "static_prefill_seq":4,"static_prefill_context":11,
                "num_kv_heads":2,"head_dim":4}"#,
        );
        let mut b = OvRuntimeBuilder::new(&dir, 0, 1, "CPU");
        let err = b
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .err()
            .expect("advertised variant without the IR file must be rejected");
        assert!(err.to_string().contains("missing"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn gemv_offload_on_stateful_shard_rejected() {
        let dir = write_stage_dir(
            "gemv-stateful",
            r#"{"layer_start":0,"layer_end":2,"has_embed":true,"has_head":true,"stateful":true}"#,
        );
        let mut b = OvRuntimeBuilder::new(&dir, 0, 1, "CPU").with_gemv_offload(true);
        let err = b
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .err()
            .expect("stateful + --gemv-offload must be rejected");
        assert!(err.to_string().contains("--gemv-offload"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn gemv_offload_on_non_cpu_device_rejected() {
        let dir = write_stage_dir(
            "gemv-npu",
            r#"{"layer_start":0,"layer_end":2,"has_embed":true,"has_head":true,
                "stateful":false,"static_seq":1,"static_context":8,
                "num_kv_heads":2,"head_dim":4}"#,
        );
        let mut b = OvRuntimeBuilder::new(&dir, 0, 1, "NPU").with_gemv_offload(true);
        let err = b
            .load(ShardSpec::single_stage("m", "NPU"))
            .await
            .err()
            .expect("non-CPU device + --gemv-offload must be rejected");
        assert!(err.to_string().contains("CPU"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn park_prefill_on_stateful_shard_rejected() {
        let dir = write_stage_dir(
            "park-stateful",
            r#"{"layer_start":0,"layer_end":2,"has_embed":true,"has_head":true,"stateful":true}"#,
        );
        let mut b = OvRuntimeBuilder::new(&dir, 0, 1, "CPU").with_prefill_parking(true);
        let err = b
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .err()
            .expect("stateful + --park-prefill must be rejected");
        assert!(err.to_string().contains("--park-prefill"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn disable_chunked_prefill_skips_variant() {
        // Same config as above (variant advertised, file missing) but with
        // chunked prefill disabled: validation must not fire; the load then
        // proceeds to the decode-model compile, which errs (stub / no IR) —
        // proving the prefill variant was skipped, not rejected.
        let dir = write_stage_dir(
            "disabled",
            r#"{"layer_start":0,"layer_end":2,"has_embed":true,"has_head":true,
                "stateful":false,"static_seq":1,"static_context":8,
                "static_prefill_seq":4,"static_prefill_context":11,
                "num_kv_heads":2,"head_dim":4}"#,
        );
        let mut b = OvRuntimeBuilder::new(&dir, 0, 1, "CPU").with_chunked_prefill_disabled(true);
        let err = b
            .load(ShardSpec::single_stage("m", "CPU"))
            .await
            .err()
            .expect("load still fails later (no real IR/OV in stub tests)");
        let msg = err.to_string();
        assert!(
            !msg.contains("static_prefill") && !msg.contains("missing —"),
            "prefill validation should be skipped when disabled; got: {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
