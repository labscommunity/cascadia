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
//! Wire format between stages (non-packed path): each activation is TWO frames,
//! `[lead] [hidden_states f16]`. The lead is an I64 tensor carrying the per-hop
//! sequence number, which the downstream neighbour echoes back on its token so
//! a late orphan token can be discarded rather than read by the next request
//! (see `encode_wire_lead`). The token reply is I32 `[1,1,2]` = `[token, seq]`.
//!
//! Stateful shards send lead `[1,1,1]` = `[seq]`: each stage tracks its own
//! absolute-position counter (computing cos/sin locally, no position on the
//! wire), and that counter resets when an activation with seq_len > 1 arrives
//! (a prefill signal for relay/last stages).
//!
//! Stateless static-shape (NPU) shards (`stage_config.stateful == false`)
//! instead drive a host-side bounded KV ring per stage (see `StaticKv`).
//! Because static shards are seq=1, the seq>1 prefill signal is unavailable,
//! so they send lead `[1,1,2]` = `[seq, position]` carrying the absolute
//! position; downstream stages reset their ring at position 0 and derive the
//! visible-past count from it, keeping every stage's ring in lockstep. This
//! path works single- or multi-stage (pipeline-parallel NPU).
//!
//! The lead frame's shape is what distinguishes the two paths on the wire, and
//! a stage rejects a lead that does not match its own staticness.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{
    advance_emitted, DType as ShimDType, Error as OvError, PluginConfig, Runtime as OvRuntime,
};
use cascadia_transport::{
    ActivationClient, ActivationServer, DType as WireDType, Tensor as WireTensor, MAX_RANK,
};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::{debug, error, info, warn};

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
    /// Packed multi-slot IR variant (openvino_packed_model.xml): how many
    /// independent requests share one inference via the sequence dimension.
    #[serde(default)]
    packed_slots: Option<u32>,
    #[serde(default)]
    packed_seq: Option<u32>,
    #[serde(default)]
    packed_context: Option<u32>,
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

pub(crate) fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
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

pub(crate) fn map_ov_err(err: OvError) -> EngineError {
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

/// Refuse a packed variant whose query window is narrower than its slot count.
///
/// A decode step lays down one row per ready slot, but the plan is only
/// `packed_seq` rows wide — so with `packed_seq < slots` the slots past the
/// window never get a row, and sampling then asks for a logits row the output
/// does not contain. `packed_seq > slots` is fine and useful: the extra rows
/// widen a prefill chunk. Only the narrow case is broken.
fn packed_geometry_error(slots: u32, packed_seq: u32) -> Option<String> {
    (packed_seq < slots).then(|| {
        format!(
            "packed variant has packed_seq={packed_seq} but packed_slots={slots}: the query \
             window cannot be narrower than the slot count, or slots beyond it can never \
             decode. Rebuild with `python tools/packed_variant.py <stage_dir> --slots {slots}` \
             (packed_seq defaults to the slot count)"
        )
    })
}

/// Refuse a prompt that cannot fit one packed slot's KV region.
///
/// A slot's region is a bounded window: once it fills, the oldest entries slide
/// out. Serving an over-region prompt would answer from a silently truncated
/// prompt — a request that looks successful and is wrong — so admission rejects
/// it instead. Both remedies are the operator's, so name both.
fn packed_prompt_too_long(prompt_tokens: usize, region: usize) -> Option<String> {
    (prompt_tokens > region).then(|| {
        format!(
            "prompt is {prompt_tokens} tokens but this worker's per-slot KV region holds \
             {region}; raise --static-context or lower --packed-slots (per-slot context is \
             (static_context - 1 - packed_prefix) / packed_slots)"
        )
    })
}

/// Encode a packed step's per-row assignment as one framed I64 `[1, 3, S]`
/// tensor: row 0 slot ids (`-1` = idle row), row 1 absolute positions, row 2
/// the shared-prefix reuse length. The packed analogue of
/// `encode_wire_position` — downstream stages need per-ROW routing, not one
/// scalar, to rebuild the mask and scatter each row into the right slot's ring.
///
/// The reuse length must travel: only the driver stage holds the prompt ids to
/// match against the prefix cache, but EVERY stage has to open the same shared
/// columns or its attention would disagree with stage 0's.
fn encode_wire_plan(rows: &[Option<(usize, i64, usize)>]) -> WireTensor {
    let s = rows.len();
    let mut data = Vec::with_capacity(s * 3 * 8);
    for r in rows {
        let slot: i64 = r.map(|(sl, _, _)| sl as i64).unwrap_or(-1);
        data.extend_from_slice(&slot.to_le_bytes());
    }
    for r in rows {
        let pos: i64 = r.map(|(_, p, _)| p).unwrap_or(0);
        data.extend_from_slice(&pos.to_le_bytes());
    }
    for r in rows {
        let sh: i64 = r.map(|(_, _, sh)| sh as i64).unwrap_or(0);
        data.extend_from_slice(&sh.to_le_bytes());
    }
    WireTensor::new(WireDType::I64, [1, 3, s as u32], data)
}

/// Decode + validate a packed plan frame. Rejects a wrong dtype/rank, a
/// length/shape disagreement, out-of-range slot ids, and negative positions —
/// each of which would otherwise index or wrap somewhere downstream.
fn decode_wire_plan(
    t: &WireTensor,
    slots: usize,
) -> EngineResult<Vec<Option<(usize, i64, usize)>>> {
    let s = t.shape[2] as usize;
    if t.dtype != WireDType::I64 || t.shape[0] != 1 || t.shape[1] != 3 || s == 0 {
        return Err(EngineError::Backend(format!(
            "expected an I64 [1,3,S] packed plan frame, got dtype={:?} shape={:?} — likely a \
             packed/non-packed pipeline mismatch or a desynced activation stream",
            t.dtype, t.shape
        )));
    }
    if t.data.len() != s * 3 * 8 {
        return Err(EngineError::Backend(format!(
            "packed plan frame payload {} bytes does not match shape {:?}",
            t.data.len(),
            t.shape
        )));
    }
    let rd = |i: usize| -> i64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&t.data[i * 8..i * 8 + 8]);
        i64::from_le_bytes(b)
    };
    let mut out = Vec::with_capacity(s);
    for r in 0..s {
        let slot = rd(r);
        let pos = rd(s + r);
        let shared = rd(2 * s + r);
        if slot < 0 {
            out.push(None);
            continue;
        }
        if shared < 0 || shared > pos {
            return Err(EngineError::Backend(format!(
                "packed plan row {r} has shared_use={shared} inconsistent with position {pos}"
            )));
        }
        if slot as usize >= slots {
            return Err(EngineError::Backend(format!(
                "packed plan row {r} names slot {slot} but this stage has {slots} slots"
            )));
        }
        if pos < 0 {
            return Err(EngineError::Backend(format!(
                "packed plan row {r} has negative position {pos}"
            )));
        }
        out.push(Some((slot as usize, pos, shared as usize)));
    }
    Ok(out)
}

/// True when a packed plan carries a prefill chunk: some slot owns more than
/// one row (decode frames have at most one row per slot). Picks the token
/// reply's deadline budget — a prefill reply waits on multi-token compute
/// across every remaining stage. A single-row prefill chunk is
/// indistinguishable from a decode row here and gets the strict decode
/// deadline, which a one-row tail inference fits comfortably.
fn plan_has_prefill_rows(rows: &[Option<(usize, i64, usize)>]) -> bool {
    let mut seen = std::collections::HashSet::new();
    rows.iter()
        .flatten()
        .any(|(slot, _, _)| !seen.insert(*slot))
}

/// Encode the head stage's per-row sampled tokens as I64 `[1, 1, S]`. The
/// non-packed path returns one token; packed returns one per active row, in
/// row order, with 0 for idle rows (never read).
fn encode_wire_tokens(tokens: &[i32]) -> WireTensor {
    let mut data = Vec::with_capacity(tokens.len() * 8);
    for &t in tokens {
        data.extend_from_slice(&(t as i64).to_le_bytes());
    }
    WireTensor::new(WireDType::I64, [1, 1, tokens.len() as u32], data)
}

fn decode_wire_tokens(t: &WireTensor) -> EngineResult<Vec<i32>> {
    if t.dtype != WireDType::I64 || t.data.len() % 8 != 0 || t.data.is_empty() {
        return Err(EngineError::Backend(format!(
            "expected an I64 packed token frame, got dtype={:?} len={}",
            t.dtype,
            t.data.len()
        )));
    }
    Ok(t.data
        .chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            i64::from_le_bytes(b) as i32
        })
        .collect())
}

/// The NACK a stage sends instead of a token reply: an EMPTY token frame,
/// `[1, 1, 0]` I64 with no payload.
///
/// A stage that has consumed a (plan, hidden) pair OWES its upstream a reply,
/// so a failed step must still answer — and this is the answer that means
/// "the batch is lost, the link is not". Emptiness is what carries that: the
/// frame is wire-valid (shape and payload agree, so the stream stays aligned)
/// while a real reply always carries one token per row and can never be
/// empty. Paired with [`is_packed_nack`] deliberately — the sender and the
/// recogniser have to agree on one shape, so they are defined together and
/// tested together.
fn packed_nack_frame() -> WireTensor {
    encode_wire_tokens(&[])
}

/// Is this reply the downstream's NACK rather than a token frame?
///
/// The test is "carries no tokens" — a zero element count, exactly what
/// [`packed_nack_frame`] sends and exactly what a real reply (one token per
/// row, rows >= 1) can never be.
///
/// Deliberately NOT an equality check against the whole frame: a zero-element
/// frame holds no tokens whatever its dtype byte claims, so reading it as a
/// lost batch is right either way, and a stricter test would instead hand it
/// to [`decode_wire_tokens`] to be reported as wire corruption.
///
/// Call this BEFORE `decode_wire_tokens`, which rejects an empty payload as
/// malformed and would otherwise turn every NACK into a bogus frame error.
fn is_packed_nack(reply: &WireTensor) -> bool {
    reply.elements() == Some(0)
}

/// Type the error that retires an in-flight packed batch.
///
/// A lost batch is not a lost link, and [`EngineError::BatchAborted`] says so
/// structurally instead of leaving the answer to whichever substrings the
/// message happens to carry — abort messages quote their cause, and that
/// cause is routinely a transport failure on some *other* rank's socket.
///
/// A cause that is itself connection-fatal (this stage's own poisoned or dead
/// socket) is passed through untouched: that link really is gone, and
/// anything downstream of here that classifies the error — the relay loop
/// today, a dead-wire latch later — must still see it as fatal.
fn packed_abort_error(e: EngineError) -> EngineError {
    match e {
        // Already a lost batch (a downstream NACK relayed up to this stage):
        // reuse it rather than wrapping the same words twice.
        already @ EngineError::BatchAborted(_) => already,
        fatal if fatal.is_connection_fatal() => fatal,
        other => EngineError::BatchAborted(format!("the packed step failed: {other}")),
    }
}

/// The client-facing text every task retired by one aborted packed batch
/// receives. A [`EngineError::BatchAborted`] already reads "batch aborted: …"
/// in its Display, and a cause left connection-fatal keeps its own wording, so
/// name the abort here rather than prefixing it twice.
fn packed_abort_message(aborted: &EngineError) -> String {
    match aborted {
        EngineError::BatchAborted(_) => aborted.to_string(),
        fatal => format!("packed batch aborted: {fatal}"),
    }
}

/// One-way latch: this stage's packed downstream link is dead for the rest of
/// the process.
///
/// A reply-deadline miss POISONS the downstream socket (the transport drops
/// it, so every later use answers `NotConnected`), and nothing reconnects:
/// [`ActivationClient`] dials once, at startup. On a relay rank that is
/// survivable — the step error is connection-fatal, the relay loop exits, and
/// the supervisor rebuilds the stage. Rank 0 has no relay loop: it is driven
/// by stream polls, so it would otherwise admit every new request, run the
/// local prefill, fail the exchange on a socket that can never come back, and
/// abort the batch — 100% failures, forever, behind one uniform per-batch WARN
/// while the rebuilt relay ranks sit in `accept()`.
///
/// The latch turns that into fail-fast-and-loud: the first fatal cause is
/// recorded once (with an `error!` naming the restart as the remedy) and every
/// later request is refused immediately, before any local inference is burned.
/// It is deliberately NOT a recovery mechanism — see [`Self::fail_fast_error`].
#[derive(Default)]
struct WireDeadLatch {
    /// Display text of the FIRST connection-fatal cause observed. `Some` means
    /// latched; the value is kept for attribution, never re-classified.
    cause: Option<String>,
}

impl WireDeadLatch {
    /// Feed the typed cause that just retired a packed batch.
    ///
    /// Classification is structural — [`EngineError::is_connection_fatal`] on
    /// the typed value, exactly as `packed_abort_error` left it. The message is
    /// only ever stored, never re-parsed: a batch abort whose text happens to
    /// quote some other rank's transport failure must not arm this latch, and
    /// [`EngineError::BatchAborted`] answers `false` before any substring is
    /// examined.
    ///
    /// Returns `true` on the latching transition ONLY — the caller logs there,
    /// so the operator-facing error is emitted once per process rather than
    /// once per aborted batch. Later fatal causes are redundant (the first one
    /// killed the link) and leave the stored cause untouched.
    fn observe(&mut self, aborted: &EngineError) -> bool {
        if self.cause.is_some() || !aborted.is_connection_fatal() {
            return false;
        }
        self.cause = Some(aborted.to_string());
        true
    }

    /// The first fatal cause, once latched.
    fn cause(&self) -> Option<&str> {
        self.cause.as_deref()
    }

    /// The error every request gets while latched, or `None` when the wire is
    /// still believed good.
    ///
    /// Two properties are load-bearing:
    ///
    /// * It is `Backend` carrying the literal "not connected", so
    ///   [`EngineError::is_connection_fatal`] answers `true` — fatality stays
    ///   observable to any supervisor plumbing added later. It must never be
    ///   [`EngineError::BatchAborted`], which is structurally non-fatal. The
    ///   phrase is written here rather than inherited from `cause`, because the
    ///   stored text need not match the classifier at all (a latch armed by the
    ///   structural [`EngineError::NotConnected`] displays as "not YET
    ///   connected", which no substring rule catches).
    /// * It quotes the original cause, so the operator and the SSE client both
    ///   learn what actually killed the link, not just that it is gone.
    fn fail_fast_error(&self) -> Option<EngineError> {
        self.cause.as_deref().map(|cause| {
            EngineError::Backend(format!(
                "packed downstream link is not connected: this stage latched a dead wire and \
                 fails every request until the process is restarted (first cause: {cause})"
            ))
        })
    }
}

/// The relay failed to answer its upstream. When the step it was answering
/// for ALSO failed, the send error on its own hides why the batch died — and
/// those two failures are correlated, not independent: a dead upstream is
/// precisely the case where the NACK cannot be delivered either. So carry
/// both in one error.
///
/// The send error stays outermost: it is the transport failure, and its text
/// is what [`EngineError::is_connection_fatal`] inspects to decide whether the
/// relay loop exits for a supervisor rebuild. Embedding the body error's text
/// after it keeps the root cause visible without hiding the link failure.
fn nack_send_error(send_err: EngineError, body_err: Option<&EngineError>) -> EngineError {
    match body_err {
        Some(body) => EngineError::Backend(format!("{send_err} (NACK sent because: {body})")),
        None => send_err,
    }
}

// -------- per-hop sequence echo (token desync guard) --------

/// Encode the LEAD activation frame: the per-hop sequence number, plus the
/// absolute position when the sending stage is static (NPU).
///
/// Each stage stamps a monotonic seq on the hidden it sends downstream; the
/// downstream neighbor echoes it back on the token so a LATE orphaned token
/// (from a slow/recovering peer) can be detected and discarded instead of
/// silently read by the next request. Paired with `decode_wire_lead`.
///
/// The SHAPE is the discriminator: `[1,1,1]` = `[seq]` on the stateful path,
/// `[1,1,2]` = `[seq, position]` on the static path. Carrying the position in
/// this frame rather than a separate one is what makes the wire unambiguous —
/// a standalone seq frame and a standalone position frame were both I64
/// `[1,1,1]` with 8 bytes, i.e. byte-identical, so a static peer predating the
/// seq wire had its position silently bound as a sequence number and the
/// failure surfaced one frame later as a bogus complaint about the hidden
/// frame. It also drops one frame per token per hop, which matters because
/// `set_nodelay` is on: that was a third small TCP segment on every decode step.
fn encode_wire_lead(seq: u32, position: Option<i64>) -> WireTensor {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&(seq as i64).to_le_bytes());
    if let Some(p) = position {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    let lanes = if position.is_some() { 2 } else { 1 };
    WireTensor::new(WireDType::I64, [1, 1, lanes], bytes)
}

/// Decode + strictly validate the lead frame. Symmetric with
/// `encode_wire_lead`; the seq lane's i64 carries a u32, so the cast
/// round-trips (wrap-around values included).
///
/// `want_pos` is the RECEIVING stage's own staticness. The frame must carry
/// exactly the lanes this stage expects, so a stateful/static pipeline
/// mismatch, a desynced stream, or a peer predating the seq wire is a hard
/// error here instead of a silent mis-bind. The sign check on the position
/// lane matters for the same reason it did standalone: ring math casts to
/// usize and a negative value would wrap rather than error.
fn decode_wire_lead(t: &WireTensor, want_pos: bool) -> EngineResult<(u32, Option<i64>)> {
    let lanes: u32 = if want_pos { 2 } else { 1 };
    let want_len = lanes as usize * 8;
    if t.dtype != WireDType::I64 || t.shape != [1, 1, lanes] || t.data.len() != want_len {
        return Err(EngineError::Backend(format!(
            "expected an I64 [1,1,{lanes}] {want_len}-byte lead frame ([seq{}]), got \
             dtype={:?} shape={:?} len={} — likely a stateful/static pipeline mismatch, a \
             desynced activation stream, or a peer that predates the seq-tagged wire",
            if want_pos { ", position" } else { "" },
            t.dtype,
            t.shape,
            t.data.len()
        )));
    }
    let seq = i64::from_le_bytes(t.data[0..8].try_into().unwrap()) as u32;
    let position = if want_pos {
        let p = i64::from_le_bytes(t.data[8..16].try_into().unwrap());
        if p < 0 {
            return Err(EngineError::Backend(format!(
                "negative wire position {p} — corrupted or desynced activation stream"
            )));
        }
        Some(p)
    } else {
        None
    };
    Ok((seq, position))
}

/// Encode a token + the echoed per-hop seq as I32 `[1,1,2]` = `[token, seq]`
/// (8 bytes): the token is element 0, the seq element 1 — i.e. the seq occupies
/// bytes 4..8, the HIGH half if the payload is read as one little-endian i64.
/// `decode_token_with_seq` reverses it.
fn encode_token_with_seq(token: i32, seq: u32) -> WireTensor {
    let mut bytes = Vec::with_capacity(8);
    bytes.extend_from_slice(&token.to_le_bytes());
    bytes.extend_from_slice(&(seq as i32).to_le_bytes());
    WireTensor::new(WireDType::I32, [1, 1, 2], bytes)
}

/// Decode an I32 `[1,1,2]` token frame into `(token, echo_seq)`. The seq cast
/// round-trips the bit pattern (i32 -> u32), so wrap-around values survive.
///
/// Validation is STRICT — dtype, shape and length, matching `decode_wire_lead`
/// and `decode_wire_tokens` rather than checking length alone. A length-only
/// check accepted any 8-byte frame, and the packed path's reply is exactly
/// that: `encode_wire_tokens` for one row is I64 `[1,1,1]`, 8 bytes. It decoded
/// as `token` = the low word and `echo_seq` = 0, and since the first stamped
/// seq was also 0 the FIRST exchange of a non-packed head wired to a packed
/// neighbour matched and emitted a token from a structurally mismatched
/// pipeline; every later step then discarded every reply and stalled the full
/// deadline. A legacy 4-byte I32 `[1,1,1]` token from a pre-seq peer is caught
/// here too, with the remedy named.
fn decode_token_with_seq(t: &WireTensor) -> EngineResult<(i32, u32)> {
    if t.dtype != WireDType::I32 || t.shape != [1, 1, 2] || t.data.len() != 8 {
        return Err(EngineError::Backend(format!(
            "expected an I32 [1,1,2] 8-byte token frame ([token, seq]), got dtype={:?} \
             shape={:?} len={} — a 4-byte I32 [1,1,1] frame means the downstream runs a build \
             predating the seq-tagged token wire (upgrade both stages); an I64 frame means a \
             packed stage is wired to a non-packed one",
            t.dtype,
            t.shape,
            t.data.len()
        )));
    }
    let token = i32::from_le_bytes([t.data[0], t.data[1], t.data[2], t.data[3]]);
    let echo_seq = i32::from_le_bytes([t.data[4], t.data[5], t.data[6], t.data[7]]) as u32;
    Ok((token, echo_seq))
}

/// The token value a relay stage sends when its step FAILED after it had
/// already consumed the upstream's activation group.
///
/// A stage that consumed a group OWES its upstream a reply, so a failed step
/// must still answer — this is the answer that means "the batch is lost, the
/// link is not", the same contract the packed path states at
/// `step_relay_packed`. Real tokens are vocab indices produced by
/// `argmax_last_row`, which starts at `0usize` and only ever returns an index,
/// so a negative value can never be a legitimate token.
///
/// It rides the ORDINARY seq-tagged token frame rather than a distinctly-shaped
/// one, and that is load-bearing in two directions. The upstream's own token
/// wait is bounded by the same budget this stage's was, so by the time a
/// timeout-driven NACK is sent the upstream has usually already given up: the
/// NACK lands in the socket as a stale frame and is read by the NEXT request. A
/// shape-distinguished NACK carries no seq, so nothing could reject it and it
/// would abort the next, healthy generation. Carrying the seq means the
/// existing stale-discard loop throws it away for free. Second, a
/// distinctly-shaped NACK would collide with the two foreign 8-byte frames
/// `decode_token_with_seq` exists to reject — a legacy `[1,1,1]` token and a
/// packed I64 reply.
const NACK_TOKEN: i32 = -1;

/// Send the downstream activation frames in wire order: `[lead] [hidden]`,
/// where `lead` is `[seq]` or `[seq, position]` (see `encode_wire_lead`).
///
/// Takes the VALUES, not pre-encoded frames, and encodes here: the lead and the
/// hidden are both `WireTensor`, so a pre-encoded signature let the two be
/// transposed at the call site and still compile — on a wire whose frames were
/// already hard to tell apart. Shared by `send_hidden_downstream` and its
/// loopback tests so the order is defined in exactly one place.
async fn send_hidden_frames(
    downstream: &Arc<tokio::sync::Mutex<ActivationClient>>,
    seq: u32,
    position: Option<i64>,
    hid: WireTensor,
) -> Result<(), cascadia_transport::TransportError> {
    let lead = encode_wire_lead(seq, position);
    let mut guard = downstream.lock().await;
    guard.send(&lead).await?;
    guard.send(&hid).await?;
    Ok(())
}

/// Receive the upstream activation frames in wire order: `[lead] [hidden]` —
/// the inverse of `send_hidden_frames`. Returns the raw frames so decoding
/// happens outside the transport closure, where a bad frame yields a clear
/// `EngineError` instead of a desync.
///
/// The LEAD frame is the IDLE "next request" wait (bounded only by the
/// transport frame-idle ceiling — "no next request yet" is fine). The hidden
/// frame is a mid-group reply the peer owes promptly once the group has
/// started, so it is deadlined (`recv_reply`): a half-sent pair must not wedge
/// the stage for the whole idle ceiling (#75 mid-pair protection, carried over
/// to the seq-prefixed wire). Note this deadlines the hidden on the STATEFUL
/// path too, which the pre-seq wire left lenient.
///
/// A stateful (non-static) stage may instead be handed a bare I8 KV control
/// frame (CAPTURE/RESTORE) between turns. That frame stands ALONE — no hidden
/// follows it — so it returns as `(frame, None)` and the caller handles it and
/// waits again. Peeking here rather than in the caller is what keeps this the
/// literal body of `recv_hidden_from_upstream`, and so keeps the loopback tests
/// below covering it: a peek inlined at the call site would be untested.
async fn recv_hidden_frames(
    upstream: &Arc<tokio::sync::Mutex<ActivationServer>>,
    want_pos: bool,
) -> Result<(WireTensor, Option<WireTensor>), cascadia_transport::TransportError> {
    let mut guard = upstream.lock().await;
    let lead = guard.recv().await?.0;
    // A static shard never receives control frames, and its lead is I64 either
    // way — so the check costs nothing there and cannot shadow an activation.
    #[cfg(feature = "kv_coord")]
    if !want_pos && lead.dtype == WireDType::I8 {
        return Ok((lead, None));
    }
    let _ = want_pos;
    let hid = guard.recv_reply().await?.0;
    Ok((lead, Some(hid)))
}

/// Hard ceiling on the active token-response wait, independent of the body
/// `recv_timeout()`. `recv_timeout` is operator-tunable for slow stages, and
/// letting it govern the token wait would re-couple the engine-lock hold to it —
/// a high `recv_timeout` would re-grow the self-heal latency the bounded recv
/// exists to cap (internal tracker, issue #40). A token *response* of an ACTIVE
/// generation has a tight real deadline regardless, so cap it here.
const TOKEN_RECV_DEADLINE_CEILING: std::time::Duration = std::time::Duration::from_secs(120);
// Issue-34 warm-resume: bound the downstream RESTORE/ABORT ack. A lost or raced ack must degrade
// to cold reprefill (the caller aborts + reprefills on Err), never hang the serve at the client
// deadline. Warm-resume is an optimization, not a correctness gate — generous enough for a valid
// tail restore, well under any client timeout.
const RESTORE_ACK_TIMEOUT: Duration = Duration::from_secs(15);

/// Overall budget for one token wait.
///
/// A DECODE reply is one token of compute per remaining stage, so it gets the
/// base `recv_timeout` capped at [`TOKEN_RECV_DEADLINE_CEILING`]. A PREFILL
/// reply waits on whole-prompt (or whole-chunk) compute across every remaining
/// stage, so it scales with tokens-per-frame x pipeline depth — the same
/// reasoning, and the same factor, as the transport's
/// [`cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR`]. Both the budget and
/// the ceiling widen, so the cap keeps its meaning ("an operator-raised
/// recv_timeout cannot grow the engine-lock hold without bound") on a path
/// where the wait legitimately scales.
///
/// The ceiling is not absolute for prefill on purpose: a tuned-up `recv_timeout`
/// exists precisely because some stages are slow, and prefill is the case that
/// legitimately needs it. At the default 60s this yields 60s / 600s; at the
/// rig's 120s, 120s / 1200s — the same prefill budget `recv_reply_prefill` gave
/// before, and strictly tighter than it at pathological settings.
///
/// Pure, for testing.
fn token_recv_deadline(recv_timeout: std::time::Duration, prefill: bool) -> std::time::Duration {
    let factor = if prefill {
        cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR
    } else {
        1
    };
    // saturating_mul: an absurdly large configured base must clamp, not panic
    // the engine thread (Duration's Mul panics on overflow).
    recv_timeout
        .saturating_mul(factor)
        .min(TOKEN_RECV_DEADLINE_CEILING.saturating_mul(factor))
}

/// Consecutive token-wait timeouts after which a RELAY rank gives up on its
/// downstream and exits for a supervisor rebuild (see
/// `escalate_if_downstream_is_gone`). Each timeout already burns a full token
/// budget, so a small count is minutes of grace, not seconds.
const RELAY_TOKEN_TIMEOUTS_BEFORE_EXIT: u32 = 3;

/// Fold one token-wait outcome into the consecutive-timeout streak.
///
/// ANY answer resets it — a token, a downstream NACK, even a malformed frame.
/// All three prove bytes are still crossing the link, so the link is not the
/// suspect; only silence is. `saturating_add` because the streak is only ever
/// compared against a small threshold and must not wrap on a stage that has
/// been shouting into the void for a very long time.
///
/// Pure, for testing: the streak's reset semantics decide whether a healthy
/// relay rank gets torn down, and that is not reachable from a unit test
/// through `OvRuntimeEngine`, which needs a compiled IR to exist at all.
fn next_timeout_streak(current: u32, timed_out: bool) -> u32 {
    if timed_out {
        current.saturating_add(1)
    } else {
        0
    }
}

/// Whether a relay rank should stop retrying and exit for a supervisor rebuild.
///
/// Both conditions are required. The streak alone is not enough: a step that
/// SUCCEEDED after an earlier timeout must not trip the exit, and the streak is
/// only cleared by `next_timeout_streak` on the recv path — a step can fail for
/// reasons that never touch the token wait at all.
///
/// Pure, for testing. See [`RELAY_TOKEN_TIMEOUTS_BEFORE_EXIT`].
fn should_escalate(streak: u32, step_failed: bool) -> bool {
    step_failed && streak >= RELAY_TOKEN_TIMEOUTS_BEFORE_EXIT
}

/// Why a bounded token wait ended without a token.
///
/// The distinction exists so relay escalation can key on "the downstream never
/// answered" WITHOUT re-reading error text — the fragility
/// `EngineError::BatchAborted` was introduced to end. A typed two-way split is
/// cheaper than a substring rule and cannot be broken by rewording a message.
#[derive(Debug)]
enum TokenWaitFailure {
    /// The budget elapsed with no answer: a frame-start timeout, or the overall
    /// deadline running out between discards. The only failure that says
    /// anything about the LINK, and so the only one that counts toward
    /// escalating a relay rank.
    TimedOut(EngineError),
    /// Anything else — a downstream NACK, a malformed frame, a dead socket.
    /// Bytes arrived or the verdict is already decided elsewhere, so the link
    /// is not the suspect and the escalation counter resets.
    Other(EngineError),
}

impl TokenWaitFailure {
    fn into_error(self) -> EngineError {
        match self {
            TokenWaitFailure::TimedOut(e) | TokenWaitFailure::Other(e) => e,
        }
    }
}

/// The terminal error for a token wait that ran out of budget.
///
/// Both exits — the overall deadline elapsing between discards, and a
/// frame-start timeout inside one recv — come through here, so the diagnosis
/// does not depend on which one happened to fire. Naming the discard count and
/// the last echoed seq is the point: without them a persistent seq mismatch
/// reads as a slow or wedged stage, and the operator investigates the network
/// instead of a mismatched build. The wording deliberately avoids the
/// substrings `EngineError::is_connection_fatal` treats as fatal — a token wait
/// timing out must stay retryable so the head keeps its un-redialable socket.
fn token_wait_timeout_error(
    budget: std::time::Duration,
    awaiting_seq: u32,
    discarded: u64,
    last_echo: Option<u32>,
    cause: &str,
) -> EngineError {
    let detail = match (discarded, last_echo) {
        (0, _) => "no token frame arrived".to_string(),
        (n, Some(got)) => format!(
            "{n} stale token frame(s) arrived and were discarded, last echoing seq {got} — a \
             non-zero count with no match means the downstream is answering a different request, \
             or runs a build without the seq-echo wire (restart both stages on the same build)"
        ),
        (n, None) => format!("{n} frame(s) discarded"),
    };
    EngineError::Backend(format!(
        "timed out after {budget:?} waiting for the downstream token (expected seq \
         {awaiting_seq}): {detail} [{cause}]"
    ))
}

/// Read a seq-tagged token from `downstream`, discarding any STALE orphan
/// (echoed seq != `awaiting_seq`) and continuing to read. The whole wait is
/// bounded by ONE overall deadline = `min(recv_timeout(), TOKEN_RECV_DEADLINE_CEILING)`:
/// each `recv_token` gets the REMAINING budget, and the deadline elapsing (or a
/// bounded frame-start timeout) returns the Err so the engine lock releases for a
/// retry (#40 self-heal). Returns the token whose echoed seq matches `awaiting_seq`.
async fn recv_token_seq_checked(
    downstream: &Arc<tokio::sync::Mutex<ActivationClient>>,
    awaiting_seq: u32,
    prefill: bool,
) -> Result<i32, TokenWaitFailure> {
    let budget = token_recv_deadline(cascadia_transport::recv_timeout(), prefill);
    let deadline_at = std::time::Instant::now() + budget;
    // Discards are summarised into the terminal error rather than logged per
    // frame. A peer echoing the wrong seq every time — a version skew, or a
    // downstream answering a different request — can stream thousands of frames
    // inside one budget, and a per-frame WARN floods exactly the log an operator
    // has to read the diagnosis out of.
    let mut discarded = 0u64;
    let mut last_echo: Option<u32> = None;
    loop {
        let remaining = deadline_at.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err(TokenWaitFailure::TimedOut(token_wait_timeout_error(
                budget,
                awaiting_seq,
                discarded,
                last_echo,
                "the overall deadline elapsed",
            )));
        }
        let recv = {
            let mut guard = downstream.lock().await;
            guard.recv_token(remaining).await
        };
        let (tensor, _) = recv.map_err(|e| {
            // Classify on the TYPED transport error, before flattening: a
            // frame-start timeout means the downstream never answered, which is
            // the only thing relay escalation may count.
            match e {
                cascadia_transport::TransportError::FrameStartTimeout(_) => {
                    // Report through the same formatter as the overall-deadline
                    // exit. Whether the budget runs out inside one recv or
                    // between two discards is a race, and the operator needs the
                    // discard context either way — returning the bare transport
                    // message here would drop it in the common case.
                    TokenWaitFailure::TimedOut(token_wait_timeout_error(
                        budget,
                        awaiting_seq,
                        discarded,
                        last_echo,
                        &e.to_string(),
                    ))
                }
                other => TokenWaitFailure::Other(EngineError::Backend(other.to_string())),
            }
        })?;
        let (token, echo_seq) = decode_token_with_seq(&tensor).map_err(TokenWaitFailure::Other)?;
        if echo_seq != awaiting_seq {
            // First discard only: one line says the guard fired and names the
            // seqs, which is what the rig validation looks for. Any further
            // discards in the same wait are counted, not logged, and reported
            // together in the terminal error.
            if discarded == 0 {
                warn!(
                    event = "stale_token_discarded",
                    expected = awaiting_seq,
                    got = echo_seq,
                    "discarding a stale orphan token from downstream (chain re-formed mid-wait); \
                     further discards in this wait are counted, not logged"
                );
            }
            discarded += 1;
            last_echo = Some(echo_seq);
            continue;
        }
        if discarded > 0 {
            debug!(
                discarded,
                expected = awaiting_seq,
                "downstream token arrived after discarding stale orphans"
            );
        }
        // A NACK for THIS generation: the downstream consumed our activation
        // group and then failed. The batch is lost, the link is not — so this
        // is `BatchAborted`, which `is_connection_fatal` answers false to
        // structurally, exactly as the packed path does for its empty-frame
        // NACK. A relay rank must back off and keep driving on one of these,
        // not exit for a supervisor rebuild.
        //
        // A NACK for a generation we already abandoned is a stale orphan and
        // was discarded above, by seq, before reaching here.
        if token < 0 {
            return Err(TokenWaitFailure::Other(EngineError::BatchAborted(
                "downstream stage failed its step and NACKed this generation; the pipeline \
                 link stays aligned"
                    .into(),
            )));
        }
        return Ok(token);
    }
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
    /// Leading ring columns the slide never evicts (see
    /// [`crate::packed::KV_SINK`]). Capped at half the window so pinning can
    /// never crowd out the sliding portion.
    sink: usize,
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
        let sink = self.sink;
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
                // Drop the oldest EVICTABLE entry: the leading `sink` columns
                // are pinned, so the shift starts past them. Without that the
                // slide drops token 0 and attention collapses (see KV_SINK).
                buf.copy_within(base + (sink + 1) * slot..base + buf_row, base + sink * slot);
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
    /// Bytes already handed to the client. Not the decoded text: the delta is
    /// computed by `advance_emitted`, which holds back an unresolved
    /// replacement-char run and re-anchors if the decode diverges.
    emitted: Vec<u8>,
    prefilled: bool,
    last_token: i32,
    /// Issue-34: number of leading prompt tokens already warm in the OV state (restored from a
    /// pulled/cached KV blob). The prefill feeds only `prompt_ids[warm_prefix..]`. 0 ⇒ cold (full
    /// prefill), so the default path is unchanged.
    warm_prefix: usize,
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
    /// `--packed-slots`: multi-slot packed decode state (own compiled variant,
    /// per-slot KV regions, slot table). Mutually exclusive with `static_kv`
    /// single-task decode; when set, `step_first` dispatches to the packed path.
    packed: Option<crate::packed_exec::PackedState>,
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
    /// Packed path only: armed the first time a packed batch dies of this
    /// stage's own dead downstream socket. Once armed, `submit` and
    /// `step_first_packed` refuse work immediately instead of serving
    /// guaranteed failures for the life of the process. See [`WireDeadLatch`].
    wire_dead: WireDeadLatch,
    /// Per-hop sequence echo (token desync guard). Token frames carry no task
    /// id, so a LATE orphaned token from a slow/recovering downstream would be
    /// read by the next request → silent off-by-one token desync. Each stage
    /// stamps a monotonic seq on the hidden it sends downstream and the
    /// neighbor echoes it on the token; a mismatched echo is discarded.
    ///
    /// `awaiting_token_seq`: seq stamped on the LAST hidden sent downstream —
    /// PRE-incremented per send, so it doubles as the next seq to stamp. The
    /// token echo must equal it or the token is a stale orphan. Monotonic for
    /// the engine's lifetime (never reset on cancel/reset_state) so a re-formed
    /// generation cannot collide with an orphan of the old one.
    ///
    /// Pre-incrementing also means the first stamped seq is 1, never 0, so a
    /// zero echo — the value a foreign or zero-filled frame decodes to — cannot
    /// match the virgin state of a link that has not sent anything yet.
    awaiting_token_seq: u32,
    /// Seq read from the upstream hidden, echoed back on the token this stage
    /// sends upstream. `None` until a hidden has actually been received:
    /// "you may only echo a seq you were given" is then enforced by the type
    /// rather than by a zero that is indistinguishable from a real seq 0.
    inbound_seq: Option<u32>,
    /// Relay ranks only: consecutive token-wait TIMEOUTS on the downstream
    /// link. Reset by any answer at all (a token, a NACK, even a malformed
    /// frame). Drives `escalate_if_downstream_is_gone`.
    consecutive_token_timeouts: u32,
    /// Issue-34 Option C: opaque KV blob cache + NEGOTIATE/GET offers for the coordination plane.
    #[cfg(feature = "kv_coord")]
    kv: crate::kv_coordination::OvKvCache,
    /// Issue-34 Option C: lock-free holder mirror of `kv` — the capture sites write both, and
    /// `kv_holder()` hands this out so a busy engine answers pulls without contending the engine lock.
    #[cfg(feature = "kv_coord")]
    kv_share: crate::kv_coordination::SharedKvCache,
    /// Issue-34 plane warm-resume: mailbox the plane's commit parks a pulled slice in. Drained from the
    /// `OPCODE_RESTORE` arm — its lock is independent of the engine lock, which is what lets the commit
    /// path deposit while this engine is mid-`step()`.
    #[cfg(feature = "kv_coord")]
    kv_handoff: std::sync::Arc<crate::kv_coordination::KvHandoffMailbox>,
    /// Issue-34 consume: set by a `RESTORE` control frame; suppresses the next prefill's implicit
    /// `reset_state` (this worker's KV is already warm). Cleared by that prefill or by `ABORT`.
    #[cfg(feature = "kv_coord")]
    kv_warm_pending: bool,
    /// Issue-34 plane-restore MODE, read once from `CASCADIA_KV_PLANE_RESTORE` at build. Since the
    /// plane arm moved in-band (`drain_kv_handoff` under `OPCODE_RESTORE`) the chain verdict is binding
    /// in both modes, so this only labels the mode in the warm-resume logs the cert greps.
    #[cfg(feature = "kv_coord")]
    plane_restore: bool,
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
        // Stamp the monotonic per-link seq this stage expects the downstream
        // neighbor to echo back on the token (so a late orphan is detectable).
        self.awaiting_token_seq = self.awaiting_token_seq.wrapping_add(1);
        let seq = self.awaiting_token_seq;
        let mut wire_shape = [1u32; MAX_RANK];
        for (i, d) in shape.iter().enumerate().take(MAX_RANK) {
            wire_shape[i] = *d as u32;
        }
        let hid = WireTensor::new(WireDType::F16, wire_shape, f32_to_f16_bytes(hidden));
        // Static (NPU) shards need the absolute position downstream so each
        // stage can reset its ring at position 0 and align the visible-past
        // count. The wire shape has only MAX_RANK=3 dims (all used by
        // [1,1,hidden]) and the transport requires payload_len == shape*dtype,
        // so it cannot ride in the hidden tensor — it travels in the lead frame
        // alongside the seq. Wire order: [lead] [hidden], where lead is
        // [seq] (stateful) or [seq, position] (static);
        // recv_hidden_from_upstream mirrors it.
        let pos = self.static_kv.is_some().then_some(position);
        self.block_on(send_hidden_frames(&downstream, seq, pos, hid))
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        debug!(position, "downstream send: done");
        Ok(())
    }

    /// `prefill` widens the wait: the reply is owed only after every remaining
    /// stage has run multi-token inference, so the budget scales with the
    /// frame's token count and the pipeline depth (see `token_recv_deadline`).
    fn recv_token_from_downstream(&mut self, prefill: bool) -> EngineResult<i32> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        // #40. This wait was ALREADY deadlined before the bounded recv landed —
        // it used `recv_reply`/`recv_reply_prefill`, whose frame-start runs
        // under `recv_timeout`. What it was not is SURVIVABLE: a reply timeout
        // surfaced as "recv_exact timed out", which both
        // `recv_error_is_connection_fatal` and `EngineError::is_connection_fatal`
        // read as fatal, so the transport dropped the socket — and
        // `ActivationClient` dials once, at startup, with nothing to re-dial it.
        // A head that timed out on an orphaned token wait (its chain re-formed
        // under it after a rank restart) therefore lost its downstream for the
        // life of the process and answered every later request `NotConnected`.
        //
        // So the fix here is the CLASSIFICATION, not the bound: elapsing at the
        // frame start consumes zero bytes (cancel-safe read), leaves the socket
        // aligned, and returns a retryable error, so the caller releases the
        // engine lock and the next request serves on the re-formed chain.
        //
        // Keeping the socket is what admits a late orphan token, which the
        // per-hop seq echo then discards — without it that token would be read
        // as the NEXT request's and silently desync the stream.
        //
        // Unlike the idle-between-requests wait in `recv_hidden_from_upstream`,
        // this one has a real deadline: an active generation owes a token.
        let awaiting = self.awaiting_token_seq;
        let outcome = self.block_on(recv_token_seq_checked(&downstream, awaiting, prefill));
        // Only SILENCE moves the escalation streak. A token, a downstream NACK
        // and a malformed frame all reset it — see `next_timeout_streak`.
        let timed_out = matches!(outcome, Err(TokenWaitFailure::TimedOut(_)));
        self.consecutive_token_timeouts =
            next_timeout_streak(self.consecutive_token_timeouts, timed_out);
        outcome.map_err(TokenWaitFailure::into_error)
    }

    fn recv_hidden_from_upstream(&mut self) -> EngineResult<(Vec<f32>, [usize; 3], Option<i64>)> {
        // Wire order is [lead] [hidden] (see send_hidden_frames), where lead is
        // [seq] or [seq, position]. The LEAD frame's wait is lenient
        // (idle-between-requests); the hidden that must follow it is deadlined,
        // and the bounded token deadline lives on the active wait downstream.
        //
        // `recv_hidden_frames` yields no hidden for a bare I8 KV control frame:
        // handle it and wait again, since the upstream still owes an activation.
        let want_pos = self.static_kv.is_some();
        // Re-iterates only on the kv_coord control-frame path; without kv_coord that
        // branch is compiled out and the body runs exactly once.
        #[cfg_attr(not(feature = "kv_coord"), allow(clippy::never_loop))]
        loop {
            let upstream = self
                .upstream
                .clone()
                .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
            debug!(want_pos, "upstream recv: waiting");
            let (lead, hidden) = self
                .block_on(recv_hidden_frames(&upstream, want_pos))
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            info!(dtype = ?lead.dtype, len = lead.data.len(), want_pos, "ov_tail_upstream_frame_recv");
            let tensor = match hidden {
                Some(t) => t,
                #[cfg(feature = "kv_coord")]
                None => {
                    self.handle_inbound_control(&lead)?;
                    continue;
                }
                // Unreachable without kv_coord: the peek that yields `None` is compiled out.
                #[cfg(not(feature = "kv_coord"))]
                None => {
                    return Err(EngineError::Backend(
                        "upstream sent a bare KV control frame, but kv_coord is not built".into(),
                    ))
                }
            };
            debug!("upstream recv: frames arrived");
            // Record the seq this stage must echo back on the token it sends
            // upstream; decode/validate outside the transport closure so a bad
            // frame yields a clear EngineError, not a desync.
            let (inbound_seq, position) = decode_wire_lead(&lead, want_pos)?;
            self.inbound_seq = Some(inbound_seq);
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
            return Ok((floats, shape, position));
        }
    }

    fn send_token_to_upstream(&mut self, token: i32) -> EngineResult<()> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        // Echo the seq the upstream stamped on the hidden it sent us, so it can
        // detect a stale orphan if this token arrives after its wait moved on.
        // There is no seq to echo before a hidden has been received, and every
        // call site is preceded by one in the same step — so this is a bug
        // guard, not a runtime condition.
        let inbound = self.inbound_seq.ok_or_else(|| {
            EngineError::Backend(
                "no upstream seq recorded: a token was sent before any hidden was received".into(),
            )
        })?;
        let tensor = encode_token_with_seq(token, inbound);
        self.block_on(async move {
            let mut guard = upstream.lock().await;
            guard.send(&tensor).await
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    fn step_first(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.packed.is_some() {
            return self.step_first_packed();
        }
        if self.active.is_none() && !self.pending.is_empty() {
            let task = self.pending.remove(0);
            info!(prompt_len = task.prompt.len(), "ov_prefill_begin");
            let tok = self
                .tokenizer
                .clone()
                .ok_or_else(|| EngineError::Backend("first stage requires tokenizer".into()))?;
            let enc = tok
                .encode(task.prompt.clone(), false)
                .map_err(|e| EngineError::Backend(format!("tokenizer encode: {e}")))?;
            let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            // Issue-34 warm-resume: if a pulled/cached KV blob covers a strict prefix of this
            // prompt, restore it and prefill only the suffix. Gated + best-effort — off-rig
            // set_state_blob returns Stub, so this stays cold. Only the stateful (non-static) path.
            #[cfg_attr(not(feature = "kv_coord"), allow(unused_mut))]
            let mut warm_prefix = 0usize;
            #[cfg(feature = "kv_coord")]
            if self.static_kv.is_none() {
                let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
                if let Some((blob, len, plane_pulled)) =
                    self.kv.take_warm(&task.tenant, &prompt_i32)
                {
                    match self.runtime.set_state_blob(&blob) {
                        Ok(()) => {
                            // Multi-stage: RESTORE the whole downstream chain too (all-or-nothing).
                            // Any rank short ⇒ ABORT everyone + cold (never a partial/corrupt warm).
                            // Dropping the frame skipped the SAME-CHAIN restore too (where no plane pull
                            // ever armed the downstream ranks), leaving head-warm/tail-cold with no
                            // verdict and no fallback; it also starved the downstream `peer_epoch`.
                            let multi = self.downstream.is_some() && !self.spec.is_last_stage;
                            // Effective mode for THIS turn, not the process — what a cert should read.
                            let plane_turn = self.plane_restore && plane_pulled;
                            // Raw bit alongside: the AND is identically false in chain mode, which
                            // hides whether this blob was pulled cross-chain or captured locally.
                            info!(
                                plane_restore = plane_turn,
                                plane_pulled, "ov_step_first_warm_mode"
                            );
                            let chain_ok = !multi || {
                                let epoch = crate::kv_coordination::synth_epoch(&prompt_i32[..len]);
                                // Binding in BOTH modes: a plane rank now arms in-band, inside its own
                                // `OPCODE_RESTORE` handler, so a `false` here means it really is cold.
                                // The old `chain_verdict` override existed for the out-of-band arm and
                                // would now mask exactly that.
                                let ok = matches!(self.send_restore_downstream(epoch), Ok(true));
                                if !ok {
                                    let _ = self.send_abort_downstream();
                                }
                                ok
                            };
                            if chain_ok {
                                // Real KV depth, not the token count (off-by-one — see kv_seq_from_blob).
                                warm_prefix = crate::kv_coordination::kv_seq_from_blob(&blob)
                                    .map(|s| s.min(len))
                                    .unwrap_or(len);
                                // Probe A on the HEAD. Every probe so far looked only at the tail; in
                                // plane mode the head self-pulls (possibly from a different store than
                                // chain mode negotiates against), so if the head's digest differs
                                // between modes the defect is one rank up from where we have been
                                // looking.
                                info!(
                                    warm_prefix,
                                    matched = len,
                                    blob_digest = crate::kv_coordination::byte_digest(&blob),
                                    blob_len = blob.len(),
                                    plane = plane_turn,
                                    plane_pulled,
                                    "ov-runtime warm-resumed from KV blob"
                                );
                            } else {
                                let _ = self.runtime.reset_state();
                                warn!("ov-runtime: pipeline restore incomplete; cold reprefill");
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "set_state_blob failed; cold reprefill");
                            let _ = self.runtime.reset_state();
                        }
                    }
                } else {
                    tracing::info!(target: "cascadia::kv", event = "kv_warm_take_miss",
                        partner_hash = crate::kv_coordination::fnv1a64(task.tenant.as_bytes()),
                        prefix_len = prompt_i32.len());
                }
            }
            if warm_prefix == 0 && self.static_kv.is_none() {
                self.runtime.reset_state().map_err(map_ov_err)?;
            }
            self.position = warm_prefix as i64;
            info!(
                task = %task.task_id,
                prompt_tokens = prompt_ids.len(),
                warm_prefix,
                "task active (ov-runtime)"
            );
            self.active = Some(ActiveTask {
                task,
                prompt_ids,
                generated: Vec::new(),
                emitted: Vec::new(),
                prefilled: false,
                last_token: 0,
                warm_prefix,
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

    /// Upper bound on inferences one packed `step()` may run before returning.
    /// Sized for the worst case: a full-region prompt consumed `packed_seq`
    /// tokens at a time, plus slack. Purely a runaway guard.
    const PACKED_MAX_INFERS_PER_STEP: usize = 4096;

    /// Packed multi-slot step: admit what fits, run ONE inference (a prefill
    /// chunk for one slot, or a decode row for every ready slot), then sample
    /// and emit per slot. Unlike the single-task path this returns chunks for
    /// several tasks at once — which is exactly what the runner's per-task
    /// chunk buffers were built to demultiplex.
    fn step_first_packed(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        // ---- dead-wire latch ----
        // The downstream socket is poisoned for the life of this process, so
        // every exchange below can only fail — after tokenizing, admitting and
        // running a shared local inference first. Short-circuit ahead of all of
        // it. Whatever the engine still holds is retired through the SAME abort
        // machinery a live failure uses, so per-task attribution and slot
        // release happen exactly once; queued-but-unadmitted tasks are folded
        // into the same answer instead of sitting in the queue until the runner
        // gives up on their streams.
        if let Some(fatal) = self.wire_dead.fail_fast_error() {
            if self.pending.is_empty() && self.packed.as_ref().unwrap().occupied() == 0 {
                return Ok(Vec::new());
            }
            let msg = packed_abort_message(&fatal);
            let out = self
                .pending
                .drain(..)
                .map(|t| (t.task_id.clone(), Chunk::error(t.task_id, msg.clone())))
                .collect();
            return self.abort_packed_batch(fatal, out);
        }

        let mut out = Vec::new();
        // ---- admission ----
        while !self.pending.is_empty() {
            let Some(slot) = self.packed.as_ref().unwrap().free_slot() else {
                break;
            };
            let task = self.pending.remove(0);
            let tok = self
                .tokenizer
                .clone()
                .ok_or_else(|| EngineError::Backend("first stage requires tokenizer".into()))?;
            // Admission failures are attributed to the failed task so the
            // runner routes them to its stream — a bare Err here would kill
            // whichever concurrent stream happened to be polling.
            let enc = tok.encode(task.prompt.clone(), false).map_err(|e| {
                EngineError::Backend(format!("tokenizer encode: {e}"))
                    .for_task(task.task_id.clone())
            })?;
            let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            if prompt_ids.is_empty() {
                return Err(
                    EngineError::Backend("empty prompt: no tokens to prefill".into())
                        .for_task(task.task_id),
                );
            }
            info!(
                task = %task.task_id,
                slot,
                prompt_tokens = prompt_ids.len(),
                "task admitted to packed slot"
            );
            self.packed.as_mut().unwrap().admit(slot, task, prompt_ids);
        }

        // Keep inferring until this call produces at least one chunk. A prompt
        // wider than `packed_seq` takes several prefill inferences that emit
        // nothing, and the runner closes a stream that makes no progress for
        // three consecutive steps — so prefill must complete inside one step(),
        // exactly as the single-task static path already does.
        let single_stage = self.spec.is_first_stage && self.spec.is_last_stage;
        for _ in 0..Self::PACKED_MAX_INFERS_PER_STEP {
            // Any failure past admission loses the whole in-flight packed
            // batch (a shared inference, or a shared downstream exchange) —
            // route it through `abort_packed_batch` so every affected stream
            // gets its own attributed error and every slot is retired. A
            // bare `?` would kill whichever stream happened to be polling
            // and leave the other slots wedged-active forever.
            let stepped = match self.packed.as_mut().unwrap().step() {
                Ok(s) => s,
                Err(e) => return self.abort_packed_batch(e, out),
            };
            let Some((odt, oshape, obytes, kind)) = stepped else {
                break;
            };
            let emitted = if single_stage {
                self.emit_packed_rows(odt, &oshape, &obytes, kind, &mut out)
            } else {
                // Pipeline: this stage produced hidden rows. Ship the plan (so
                // every downstream stage can rebuild the same mask and route
                // rows to the same slots) plus the hidden block, then take back
                // one sampled token per row from the tail.
                bytes_to_f32(odt, &obytes).and_then(|hidden| {
                    let plan_rows = self.packed.as_ref().unwrap().last_plan_rows();
                    let tokens = self.exchange_packed_downstream(&hidden, &oshape, &plan_rows)?;
                    self.emit_packed_tokens(&tokens, kind, &mut out)
                })
            };
            if let Err(e) = emitted {
                return self.abort_packed_batch(e, out);
            }
            if !out.is_empty() {
                break;
            }
        }
        Ok(out)
    }

    /// A packed step failed in a way that loses the in-flight batch: retire
    /// every active slot (freeing its KV region) and hand each task its own
    /// final error chunk. The single-task recovery in `step_first` (the
    /// wrapper around `step_first_body`, which the packed path returns before
    /// ever reaching) cannot express a multi-task failure. Slot state
    /// downstream is stale but harmless — the next admission resets its slot
    /// in-band (a row whose position equals its reuse length starts that slot
    /// fresh).
    fn abort_packed_batch(
        &mut self,
        e: EngineError,
        mut out: Vec<(TaskId, Chunk)>,
    ) -> EngineResult<Vec<(TaskId, Chunk)>> {
        warn!(error = %e, "packed step failed; aborting the in-flight packed batch");
        let aborted = packed_abort_error(e);
        // The cause is typed, and `packed_abort_error` has already decided
        // whether it is this stage's own dead socket or merely a lost batch —
        // hand the VALUE to the latch, which asks `is_connection_fatal()`
        // rather than re-reading these words. `observe` answers true exactly
        // once, so this is a transition log, not per-batch noise.
        if self.wire_dead.observe(&aborted) {
            error!(
                cause = %aborted,
                "packed downstream link is dead and cannot be reconnected in-process; this \
                 stage now fails every request immediately — restart the process to rebuild \
                 the pipeline connection"
            );
        }
        let msg = packed_abort_message(&aborted);
        let packed = self.packed.as_mut().unwrap();
        for slot in 0..packed.slots.len() {
            if let Some(ps) = packed.retire(slot) {
                out.push((
                    ps.task.task_id.clone(),
                    Chunk::error(ps.task.task_id, msg.clone()),
                ));
            }
        }
        Ok(out)
    }

    /// Send `[1, S, hidden]` + the packed plan downstream and wait for the tail
    /// stage's per-row tokens.
    fn exchange_packed_downstream(
        &mut self,
        hidden: &[f32],
        shape: &[usize],
        plan_rows: &[Option<(usize, i64, usize)>],
    ) -> EngineResult<Vec<i32>> {
        let down = self.downstream.clone().ok_or_else(|| {
            EngineError::Backend("packed pipeline stage has no downstream".into())
        })?;
        let s3 = to_shape3(shape);
        let plan_frame = encode_wire_plan(plan_rows);
        // f16 on the wire, matching the non-packed path: the receiving stage
        // converts to f16 before feeding the IR regardless, so f32 doubled the
        // bytes per stage hop and bought nothing.
        let hidden_frame = WireTensor::new(
            WireDType::F16,
            [s3[0] as u32, s3[1] as u32, s3[2] as u32],
            f32_to_f16_bytes(hidden),
        );
        // A prefill reply waits on multi-token compute across every remaining
        // stage — widen its deadline like the non-packed path does.
        let prefill = plan_has_prefill_rows(plan_rows);
        self.block_on(async {
            let mut guard = down.lock().await;
            guard
                .send(&plan_frame)
                .await
                .map_err(|e| EngineError::Backend(format!("packed plan send: {e}")))?;
            guard
                .send(&hidden_frame)
                .await
                .map_err(|e| EngineError::Backend(format!("packed hidden send: {e}")))?;
            // Having just sent both frames this stage is OWED a token frame.
            // `recv_reply*`, NOT plain `recv` under an external
            // `tokio::time::timeout` (#122): the deadline lives inside the
            // transport, so a miss surfaces as an error instead of a dropped
            // future (which silently discarded any partially-read bytes and
            // left the stream misaligned), and a failed reply POISONS the
            // connection — the socket is dropped so a late token frame can
            // never be read into the next exchange as fresh data. The
            // poisoned-socket errors ("recv_exact timed out" / "not
            // connected") are connection-fatal, so a middle rank's relay
            // loop exits for a supervisor rebuild rather than grinding on a
            // desynced stream. Those keep their `Backend` type on purpose:
            // the link really is dead, and that classification must survive.
            let (reply, _) = if prefill {
                guard.recv_reply_prefill().await
            } else {
                guard.recv_reply().await
            }
            .map_err(|e| EngineError::Backend(format!("packed token recv: {e}")))?;
            // An EMPTY token frame is the downstream's NACK (see
            // `step_relay_packed`): its step failed AFTER it consumed our
            // pair, so the batch is lost but the link is still
            // frame-aligned. Abort the batch; do not poison the connection.
            //
            // `BatchAborted`, not `Backend`: this error travels up a middle
            // rank's relay loop, which must back off and keep driving rather
            // than exit for a supervisor rebuild. The variant says "healthy
            // link" structurally instead of relying on this message never
            // happening to contain a word the fatal-substring classifier
            // looks for.
            if is_packed_nack(&reply) {
                return Err(EngineError::BatchAborted(
                    "downstream stage failed its packed step and NACKed this batch \
                     (empty token frame); the pipeline link stays aligned"
                        .into(),
                ));
            }
            decode_wire_tokens(&reply)
        })
    }

    /// Emit chunks from a pipeline tail's per-row tokens.
    fn emit_packed_tokens(
        &mut self,
        tokens: &[i32],
        kind: crate::packed_exec::PackedStepKind,
        out: &mut Vec<(TaskId, Chunk)>,
    ) -> EngineResult<()> {
        let sampled: Vec<(usize, usize)> = match kind {
            crate::packed_exec::PackedStepKind::Prefill {
                slot,
                last_row,
                finished_prompt,
            } => {
                if finished_prompt {
                    vec![(last_row, slot)]
                } else {
                    Vec::new()
                }
            }
            crate::packed_exec::PackedStepKind::Decode { rows } => rows,
        };
        for (row, slot) in sampled {
            let token = tokens.get(row).copied().ok_or_else(|| {
                EngineError::Backend(format!(
                    "packed tail returned {} tokens, need row {row}",
                    tokens.len()
                ))
            })?;
            if let Some(chunk) = self.emit_packed_token(slot, token)? {
                out.push(chunk);
            }
        }
        Ok(())
    }

    /// Relay/head stage: consume one packed (plan, hidden) pair. The head
    /// samples every active row and replies with the token vector; a middle
    /// stage forwards both frames on and passes the reply back up.
    fn step_relay_packed(&mut self) -> EngineResult<()> {
        let up = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("packed relay stage has no upstream".into()))?;
        let slots = self.packed.as_ref().unwrap().kv.slots;
        let (plan_res, hidden, hshape) = self.block_on(async {
            let mut guard = up.lock().await;
            // The plan frame is this stage's idle wait for the next unit of
            // work — idle-tolerant `recv`. Once it arrives the upstream owes
            // the hidden frame promptly, so the SECOND frame of the pair is
            // a deadlined `recv_reply` (same rule as the non-packed
            // `recv_hidden_from_upstream`): a half-sent pair must fail fast
            // and poison the socket, not park this stage for the whole
            // frame-idle ceiling. The plan is decoded only AFTER the hidden
            // frame is consumed — bailing between the two frames would
            // leave the hidden frame in the socket and permanently desync
            // every later frame (#122).
            let (pf, _) = guard
                .recv()
                .await
                .map_err(|e| EngineError::Backend(format!("packed plan recv: {e}")))?;
            let (hf, _) = guard
                .recv_reply()
                .await
                .map_err(|e| EngineError::Backend(format!("packed hidden recv: {e}")))?;
            let plan_res = decode_wire_plan(&pf, slots);
            let hidden: Vec<f32> = match hf.dtype {
                WireDType::F16 => f16_bytes_to_f32(&hf.data),
                _ => hf
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            };
            Ok::<_, EngineError>((plan_res, hidden, hf.shape))
        })?;

        // The pair is consumed. From here on, any failure MUST still answer
        // the upstream — otherwise it blocks awaiting a token frame that
        // never comes while this loop swallows the error and moves on: the
        // silent-deadlock half of issue #122. The NACK is an EMPTY token
        // frame ([1,1,0]): wire-aligned, unambiguous (a real reply always
        // has one token per row), and it tells the upstream "batch lost,
        // link healthy".
        let step = self.relay_packed_body(plan_res, hidden, hshape);
        let frame = match &step {
            Ok(tokens) => encode_wire_tokens(tokens),
            Err(e) => {
                // Log the root cause HERE, before the NACK goes out: a dead
                // upstream is exactly when the send fails too, and then the
                // send error is the only thing that would ever reach the
                // operator. This rank is where the batch actually died, so
                // this is where the reason has to be recorded.
                warn!(error = %e, "packed relay step failed; NACKing the upstream");
                packed_nack_frame()
            }
        };
        let sent = self.block_on(async {
            let mut guard = up.lock().await;
            guard
                .send(&frame)
                .await
                .map_err(|e| EngineError::Backend(format!("packed token send: {e}")))
        });
        if let Err(send_err) = sent {
            return Err(nack_send_error(send_err, step.as_ref().err()));
        }
        step.map(|_| ())
    }

    /// Everything `step_relay_packed` does between consuming the (plan,
    /// hidden) pair and answering the upstream. Split out so the caller can
    /// turn any failure into an on-wire NACK instead of a swallowed error.
    fn relay_packed_body(
        &mut self,
        plan_res: EngineResult<Vec<Option<(usize, i64, usize)>>>,
        hidden: Vec<f32>,
        hshape: [u32; MAX_RANK],
    ) -> EngineResult<Vec<i32>> {
        let plan_rows = plan_res?;
        let hidden_size = hshape[2] as usize;
        let plan = crate::packed::PackedPlan {
            rows: plan_rows
                .iter()
                .map(|r| r.map(|(slot, _, _)| crate::packed::PackedRow { slot, order: 0 }))
                .collect(),
        };
        // Same-slot rows in one frame are a prefill chunk: restore their causal
        // order so this stage masks them exactly as the sender did.
        let mut plan = plan;
        let mut seen: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for row in plan.rows.iter_mut().flatten() {
            let n = seen.entry(row.slot).or_insert(0);
            row.order = *n;
            *n += 1;
        }
        let positions: Vec<i64> = plan_rows
            .iter()
            .map(|r| r.map(|(_, p, _)| p).unwrap_or(0))
            .collect();
        let reuse: Vec<usize> = plan_rows
            .iter()
            .map(|r| r.map(|(_, _, sh)| sh).unwrap_or(0))
            .collect();
        let (odt, oshape, obytes) = self.packed.as_mut().unwrap().run_plan(
            &plan,
            crate::packed_exec::PackedPrimary::Hidden(&hidden, hidden_size),
            &positions,
            &reuse,
        )?;

        if self.spec.is_last_stage {
            let logits = bytes_to_f32(odt, &obytes)?;
            let mut tokens = Vec::with_capacity(plan.rows.len());
            for (r, row) in plan.rows.iter().enumerate() {
                tokens.push(if row.is_some() {
                    argmax_logits_row(&logits, &oshape, r)?
                } else {
                    0
                });
            }
            Ok(tokens)
        } else {
            let hidden_out = bytes_to_f32(odt, &obytes)?;
            // A downstream failure (including its NACK) propagates as Err —
            // the caller NACKs our own upstream in turn, so the abort
            // reaches rank 0 no matter how deep the pipeline is.
            self.exchange_packed_downstream(&hidden_out, &oshape, &plan_rows)
        }
    }

    /// Sample the rows one packed inference produced and append their chunks.
    fn emit_packed_rows(
        &mut self,
        odt: ShimDType,
        oshape: &[usize],
        obytes: &[u8],
        kind: crate::packed_exec::PackedStepKind,
        out: &mut Vec<(TaskId, Chunk)>,
    ) -> EngineResult<()> {
        let sampled: Vec<(usize, usize)> = match kind {
            crate::packed_exec::PackedStepKind::Prefill {
                slot,
                last_row,
                finished_prompt,
            } => {
                // Only the chunk that consumed the final prompt token yields a
                // real token; earlier chunks just fill the slot's KV.
                if finished_prompt {
                    vec![(last_row, slot)]
                } else {
                    Vec::new()
                }
            }
            crate::packed_exec::PackedStepKind::Decode { rows } => rows,
        };
        if sampled.is_empty() {
            return Ok(());
        }
        let logits = bytes_to_f32(odt, obytes)?;
        for (row, slot) in sampled {
            let token = argmax_logits_row(&logits, oshape, row)?;
            if let Some(chunk) = self.emit_packed_token(slot, token)? {
                out.push(chunk);
            }
        }
        Ok(())
    }

    /// Append `token` to `slot`'s sequence, decode its text delta, and build the
    /// chunk. Returns None if the slot vanished (cancelled mid-step). Retires
    /// the slot — freeing its KV region for the next admission — on the final
    /// chunk, which is what makes this continuous rather than batch-synchronous.
    fn emit_packed_token(
        &mut self,
        slot: usize,
        token: i32,
    ) -> EngineResult<Option<(TaskId, Chunk)>> {
        let tok = self
            .tokenizer
            .clone()
            .ok_or_else(|| EngineError::Backend("first stage requires tokenizer".into()))?;
        let eos = self.eos_token_ids.clone();
        let packed = self.packed.as_mut().unwrap();
        let Some(t) = packed.slots[slot].as_mut() else {
            return Ok(None);
        };
        t.last_token = token;
        t.generated.push(token);
        // Stop conditions first: they decide `running`, which tells
        // `advance_emitted` whether it may hold back an unresolved
        // replacement-char run or must flush it.
        let max_tokens = t.task.max_tokens.max(1) as usize;
        let is_eos = eos.contains(&(token as u32));
        let hit_cap = t.generated.len() >= max_tokens;
        let is_final = hit_cap || is_eos;
        let task_id = t.task.task_id.clone();

        let all_ids: Vec<u32> = t.generated.iter().map(|&x| x as u32).collect();
        let full_text = tok
            .decode(&all_ids, true)
            .map_err(|e| EngineError::Backend(format!("tokenizer decode: {e}")))?;
        let (delta, resynced) = advance_emitted(&mut t.emitted, full_text.as_bytes(), !is_final);
        if resynced {
            warn!(task = %task_id, slot, "detokenizer decode diverged; re-anchored");
        }

        if !is_final {
            // Explicit count for the same reason: a token whose text lands in
            // the next chunk (BPE splitting a glyph) would otherwise count 0.
            return Ok(Some((
                task_id.clone(),
                Chunk::token(task_id, token as i64, delta).with_n_tokens(1),
            )));
        }
        let n_tokens = t.generated.len() as u32;
        let prompt_tokens = t.prompt_ids.len() as u32;
        let elapsed = t.started.elapsed();
        // n_tokens is THIS chunk's increment, not the running total. The API
        // sums per-chunk counts (falling back to 1 per non-empty chunk), so a
        // cumulative value here double-counts every interim token — measured as
        // completion_tokens = 2N-1. This chunk carries exactly one token, and
        // it is set explicitly so an empty final delta (EOS decoding to "")
        // still counts.
        let chunk = Chunk::final_marker(task_id.clone(), delta)
            .with_n_tokens(1)
            .with_prompt_tokens(prompt_tokens)
            .with_finish_reason(if is_eos {
                cascadia_types::FinishReason::Stop
            } else {
                cascadia_types::FinishReason::Length
            });
        packed.retire(slot);
        info!(
            task = %task_id,
            slot,
            tokens = n_tokens,
            elapsed_s = elapsed.as_secs_f64(),
            tok_s = n_tokens as f64 / elapsed.as_secs_f64().max(1e-9),
            in_flight = packed.occupied(),
            "packed task done"
        );
        Ok(Some((task_id, chunk)))
    }

    fn step_first_body(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.active.is_none() {
            return Ok(Vec::new());
        }
        let (prefill, tokens) = {
            let a = self.active.as_mut().unwrap();
            if !a.prefilled {
                a.prefilled = true;
                // warm_prefix is 0 on the cold/default path ⇒ full prompt (unchanged behaviour).
                (true, a.prompt_ids[a.warm_prefix..].to_vec())
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
            // Stateful: KV is internal, all dims dynamic. Cold feeds the whole prompt at position 0;
            // warm-resume feeds the suffix at position = real KV depth, where mask_len = kv_depth +
            // input_len holds, so one batched forward works. (The per-token feed it replaces was a
            // workaround for the now-fixed off-by-one and stalled the suffix past the gateway deadline.)
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

        // Stop conditions first: they decide `running`, which tells
        // `advance_emitted` whether it may hold back an unresolved
        // replacement-char run or must flush it.
        let max_tokens = active.task.max_tokens.max(1) as usize;
        let is_eos = self.eos_token_ids.contains(&(next_token as u32));
        let is_final = active.generated.len() >= max_tokens || is_eos;
        let task_id = active.task.task_id.clone();

        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Backend("first stage requires tokenizer".into()))?;
        let all_ids: Vec<u32> = active.generated.iter().map(|&t| t as u32).collect();
        let full_text = tok
            .decode(&all_ids, true)
            .map_err(|e| EngineError::Backend(format!("tokenizer decode: {e}")))?;
        let (delta, resynced) =
            advance_emitted(&mut active.emitted, full_text.as_bytes(), !is_final);
        if resynced {
            warn!(task = %task_id, "detokenizer decode diverged; re-anchored");
        }
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
            // Issue-34: capture the full post-turn KV under (prompt + generated) for warm-pull.
            // Best-effort + gated — off-rig get_state_blob returns Stub, so nothing is cached.
            #[cfg(feature = "kv_coord")]
            {
                let mut full: Vec<i32> = active.prompt_ids.iter().map(|&t| t as i32).collect();
                full.extend_from_slice(&active.generated);
                // H.1b R2: the namespace this turn belongs to, read off THIS task's own state —
                // never off a plane-asserted value, which describes a pulled entry, not this turn.
                let tenant = active.task.tenant.clone();
                match self.runtime.get_state_blob() {
                    Ok(blob) => {
                        // Multi-stage head: broadcast CAPTURE so every downstream rank snapshots its
                        // slice under this turn's content epoch. Best-effort. Single-stage (no
                        // downstream) skips straight to the local stash.
                        if self.kv_stateful()
                            && self.downstream.is_some()
                            && !self.spec.is_last_stage
                        {
                            let epoch = crate::kv_coordination::synth_epoch(&full);
                            if let Err(e) = self.send_capture_downstream(epoch, &full, &tenant) {
                                warn!(error = %e, "ov-runtime: CAPTURE broadcast failed (best-effort)");
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
                    Err(e) => tracing::debug!(error = %e, "get_state_blob skipped (no KV capture)"),
                }
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

    /// Answer the upstream for an activation group this stage has ALREADY
    /// consumed: the sampled token, or [`NACK_TOKEN`] if the step failed.
    ///
    /// Once the group is off the wire the upstream is owed a reply. Returning
    /// the step's error without answering leaves it waiting out its whole token
    /// budget for a frame that will never come — a wasted deadline per failure,
    /// and on a relay rank the error is then swallowed by the loop and retried,
    /// so the upstream pays it again. The packed path states the same rule at
    /// `step_relay_packed`; this is the non-packed half of it.
    ///
    /// The body's error is what propagates, so the caller still sees why the
    /// step died. A send failure subsumes it via `nack_send_error`, which keeps
    /// the transport error outermost (it is what decides whether the relay loop
    /// exits) while carrying the root cause in its text — a dead upstream is
    /// exactly when the NACK cannot be delivered either.
    fn answer_upstream(&mut self, step: EngineResult<i32>) -> EngineResult<()> {
        let token = match &step {
            Ok(t) => *t,
            Err(e) => {
                warn!(error = %e, "relay step failed; NACKing the upstream");
                NACK_TOKEN
            }
        };
        match self.send_token_to_upstream(token) {
            Ok(()) => step.map(|_| ()),
            Err(send_err) => Err(nack_send_error(send_err, step.as_ref().err())),
        }
    }

    /// Issue-34: consume the one-shot warm-resume flag. `true` ⇒ this prefill continues a RESTOREd
    /// state, so the worker must NOT reset. Always `false` in the default build (no-op).
    fn kv_consume_warm_pending(&mut self) -> bool {
        #[cfg(feature = "kv_coord")]
        {
            std::mem::take(&mut self.kv_warm_pending)
        }
        #[cfg(not(feature = "kv_coord"))]
        {
            false
        }
    }

    fn step_last(&mut self) -> EngineResult<()> {
        if self.packed.is_some() {
            return self.step_relay_packed();
        }
        // Before this point nothing is owed: a failed recv either got no group
        // at all, or lost frame alignment, in which case the socket is gone and
        // there is nothing to answer on. After it, `inbound_seq` is set and
        // every failure must answer.
        let (hidden, shape, pos_opt) = self.recv_hidden_from_upstream()?;
        let step = self.relay_last_body(hidden, shape, pos_opt);
        self.answer_upstream(step)
    }

    /// Everything `step_last` does between consuming the activation group and
    /// answering the upstream. Split out so the caller can turn any failure
    /// into an on-wire NACK instead of a silent one.
    fn relay_last_body(
        &mut self,
        hidden: Vec<f32>,
        shape: [usize; 3],
        pos_opt: Option<i64>,
    ) -> EngineResult<i32> {
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
                // Reset on a fresh prefill UNLESS a RESTORE warm-resumed this rank (keep that state +
                // its position). The flag is consumed every prefill so it never leaks across turns.
                let warm = self.kv_consume_warm_pending();
                if shape[1] > 1 && !warm {
                    self.runtime.reset_state().map_err(map_ov_err)?;
                    self.position = 0;
                }
                let r = self.run_relay(&hidden, shape, self.position)?;
                self.position += shape[1] as i64;
                r
            }
        };
        argmax_logits(&out, &out_shape)
    }

    fn step_middle(&mut self) -> EngineResult<()> {
        if self.packed.is_some() {
            return self.step_relay_packed();
        }
        // See `step_last`: nothing is owed until the group is consumed.
        let (hidden, shape, pos_opt) = self.recv_hidden_from_upstream()?;
        let step = self.relay_middle_body(hidden, shape, pos_opt);
        let answered = self.answer_upstream(step);
        self.escalate_if_downstream_is_gone(answered)
    }

    /// A middle rank whose downstream has stopped answering must eventually
    /// exit so the supervisor rebuilds the stage.
    ///
    /// Bounding the token wait made its failure NON-fatal, which is right for
    /// the head — it has no relay loop, it is driven by stream polls, and it
    /// dialed its downstream once and cannot re-dial, so tearing it down on a
    /// transient miss strands it permanently. A middle rank is the opposite
    /// case: it has a supervisor, and before the bounded recv its timeout
    /// classified fatal and produced exactly that rebuild. Without this, a
    /// permanently wedged (as opposed to closed) downstream leaves the relay
    /// loop backing off and retrying forever, with no self-heal and no
    /// operator-visible terminal state.
    ///
    /// Only CONSECUTIVE token-wait timeouts count, and only timeouts: a NACK or
    /// a malformed frame proves bytes are still flowing, and
    /// `recv_token_from_downstream` resets the counter on both. Each timeout
    /// already costs a full token budget, so the threshold is a small count
    /// rather than a wall-clock window — no extra clock plumbing for the same
    /// answer.
    ///
    /// The escalation error is `Io(TimedOut)`, which `is_connection_fatal`
    /// answers structurally. It must not be a `Backend` string chosen to
    /// contain a fatal substring: that is the fragility the typed
    /// `BatchAborted` variant was introduced to end.
    fn escalate_if_downstream_is_gone(&mut self, res: EngineResult<()>) -> EngineResult<()> {
        if should_escalate(self.consecutive_token_timeouts, res.is_err()) {
            error!(
                timeouts = self.consecutive_token_timeouts,
                "downstream has not answered a token in {} consecutive attempts; this stage \
                 cannot re-dial it in-process, so it is exiting for the supervisor to rebuild \
                 the pipeline connection",
                self.consecutive_token_timeouts
            );
            return Err(EngineError::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "downstream stage stopped answering ({} consecutive token-wait timeouts); \
                     rebuilding this stage",
                    self.consecutive_token_timeouts
                ),
            )));
        }
        res
    }

    /// Everything `step_middle` does between consuming the activation group and
    /// answering the upstream — including the downstream hop, so a downstream
    /// failure (its NACK included) propagates as an Err and this stage NACKs its
    /// own upstream in turn. That is what carries an abort all the way to the
    /// head however deep the pipeline is.
    fn relay_middle_body(
        &mut self,
        hidden: Vec<f32>,
        shape: [usize; 3],
        pos_opt: Option<i64>,
    ) -> EngineResult<i32> {
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
                let warm = self.kv_consume_warm_pending();
                if shape[1] > 1 && !warm {
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
        self.recv_token_from_downstream(prefill_reply)
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
        // Dead-wire latch: refuse before the request costs anything. This is
        // synchronous with the HTTP request, so the non-streaming path answers
        // 5xx (and the API's readiness tracker sees a failure) rather than
        // committing a 200 and delivering the same error inside a stream, and a
        // streaming client gets an attributed error immediately instead of
        // after a full local prefill. Connection-fatal by construction — see
        // `WireDeadLatch::fail_fast_error`.
        // DEBUG, not WARN: the latching `error!` is the operator signal, and it
        // is emitted once on purpose — a per-request WARN would flood exactly
        // the log the operator has to read it out of, at the request rate. Each
        // refusal is already visible to the caller as an attributed 5xx.
        if let Some(fatal) = self.wire_dead.fail_fast_error() {
            debug!(task = %task.task_id, cause = self.wire_dead.cause().unwrap_or_default(),
                   "refusing task: packed downstream link is dead");
            return Err(fatal);
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
        // Packed slots divide the KV window, so a prompt can be too long for a
        // slot while the model itself could hold it. Reject here rather than at
        // admission: submit() is synchronous with the HTTP request, so the
        // caller gets a 413 before any stream is opened — a streaming client
        // rejected mid-stream would already have had its 200 committed and
        // would receive a non-standard error frame instead.
        if let Some(packed) = self.packed.as_ref() {
            if let Some(tok) = self.tokenizer.as_ref() {
                let n = tok
                    .encode(task.prompt.clone(), false)
                    .map_err(|e| EngineError::Backend(format!("tokenizer encode: {e}")))?
                    .get_ids()
                    .len();
                if let Some(msg) = packed_prompt_too_long(n, packed.kv.region) {
                    warn!(task = %task.task_id, prompt_tokens = n,
                          region = packed.kv.region, "refusing over-region prompt");
                    return Err(EngineError::PromptTooLong(msg));
                }
            }
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
        // Packed path: drop just this request's slot. Its KV region is cleared
        // and returned to the free pool immediately, so a disconnecting client
        // frees capacity for the next admission without disturbing the other
        // in-flight slots sharing the inference.
        if let Some(packed) = self.packed.as_mut() {
            if let Some(slot) = packed
                .slots
                .iter()
                .position(|s| s.as_ref().is_some_and(|t| t.task.task_id == *task_id))
            {
                packed.retire(slot);
                info!(task = %task_id, slot, in_flight = packed.occupied(),
                      "packed cancel: slot released");
            }
            return;
        }
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

    #[cfg(feature = "kv_coord")]
    fn kv_handoff(&self) -> Option<std::sync::Arc<dyn cascadia_engine::KvWarmHandoff>> {
        Some(std::sync::Arc::clone(&self.kv_handoff)
            as std::sync::Arc<dyn cascadia_engine::KvWarmHandoff>)
    }
}

#[cfg(feature = "kv_coord")]
impl OvRuntimeEngine {
    pub(crate) fn shard_spec(&self) -> &ShardSpec {
        &self.spec
    }
    pub(crate) fn tokenizer_ref(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_deref()
    }
    pub(crate) fn kv_cache_mut(&mut self) -> &mut crate::kv_coordination::OvKvCache {
        &mut self.kv
    }
    /// Undo a plane arm: the local-state half of the chain `OPCODE_ABORT` rollback. The mailbox
    /// retraction that handler also does is `clear(epoch)` on this path. Idempotent.
    #[cfg(feature = "kv_coord")]
    pub(crate) fn abort_warm_resume_local(&mut self) {
        let _ = self.runtime.reset_state();
        self.kv_warm_pending = false;
        self.position = 0;
    }
    /// Drain the plane hand-off mailbox and apply the parked slice, if any. `true` ⇒ this rank is now
    /// armed warm, which is what makes the `RESTORE` verdict truthful in plane mode.
    ///
    /// Called from the `OPCODE_RESTORE` arm, not from the node's relay loop: the relay only iterates
    /// once `step()` has returned, by which point this rank has already cold-prefilled the whole turn
    /// and zeroed `position`, so a `set_state` there snaps the state backwards mid-turn and the output
    /// diverges. RESTORE lands before the turn's forward, so `kv_consume_warm_pending()` sees the arm
    /// and the prefill skips its implicit `reset_state`.
    ///
    /// It is also the only call site on the SAME stream as the commit that parks the slice. Driving it
    /// off the activation stream instead left the two unordered: at a short warm prefix the drain
    /// routinely ran first and the slice sat parked forever — rank cold under a warm head.
    #[cfg(feature = "kv_coord")]
    pub(crate) fn drain_kv_handoff(&mut self, expected_epoch: u64) -> bool {
        use crate::kv_coordination::HandoffReject;
        let Some(slot) = self.kv_handoff.take(expected_epoch) else {
            return false;
        };
        let fp = self.kv_model_fingerprint();
        let blob = match crate::kv_coordination::handoff_decision(&slot, fp, self.position) {
            Ok(blob) => blob,
            Err(HandoffReject::Validate) => {
                warn!(target: "cascadia::kv", event = "kv_handoff_validate_failed",
                    epoch = slot.epoch, rev = crate::kv_coordination::KV_ENGINE_REV, fp);
                return false;
            }
            Err(HandoffReject::Decode) => {
                warn!(target: "cascadia::kv", event = "kv_handoff_decode_failed", epoch = slot.epoch);
                return false;
            }
            Err(HandoffReject::TooLate(depth)) => {
                warn!(target: "cascadia::kv", event = "kv_handoff_too_late",
                    epoch = slot.epoch, position = self.position, depth);
                return false;
            }
        };
        if self.apply_warm_resume_blob(&blob) {
            info!(target: "cascadia::kv", event = "kv_handoff_applied_inline",
                epoch = slot.epoch, position = self.position,
                blob_digest = crate::kv_coordination::byte_digest(&blob));
            true
        } else {
            // set_state failed ⇒ this rank stays cold on a turn the commit path armed as warm, and
            // nothing on this side can undo that. The arm exists to make the failure greppable.
            warn!(target: "cascadia::kv", event = "kv_handoff_apply_failed",
                epoch = slot.epoch, position = self.position);
            false
        }
    }

    /// Plane warm-resume (§0(B)): set_state a pulled rank blob directly + arm warm, off the inference
    /// chain. Mirrors the carried-blob RESTORE path; the holder loop drives it via `apply_warm_resume`.
    #[cfg(feature = "kv_coord")]
    pub(crate) fn apply_warm_resume_blob(&mut self, blob: &[u8]) -> bool {
        // Captured BEFORE set_state so the ledger can show whether the engine had already advanced past
        // the resume depth when this landed — the timing candidate's signature.
        let pos_before = self.position;
        // Gate A attributes ~21.7 s of the warm turn to this call, but that figure is a DELTA between
        // two whole runs on different builds. Time it directly: our Rust-side marshalling measures
        // 147 ms on a 114.6 MB payload (`apply_path_cost_split`), so whatever lands here is
        // `ov::VariableState::set_state` and nothing else.
        let t_set_state = std::time::Instant::now();
        let set_state = self.runtime.set_state_blob(blob);
        let set_state_ms = t_set_state.elapsed().as_millis() as u64;
        match set_state {
            Ok(()) => {
                self.position = crate::kv_coordination::kv_seq_from_blob(blob).unwrap_or(0) as i64;
                // Probe A+B (PLANE apply site). `position` settled the depth question (head 97 == tail
                // 97, mismatch refuted). What remains is whether the BYTES differ from the chain path's
                // and WHEN this lands relative to the turn — hence the digest plus the pre-apply
                // position. Compare `blob_digest` here against `ov_tail_restore_carried`'s for the same
                // prompt: equal ⇒ blob-content refuted and the timing ledger decides; unequal ⇒ the two
                // modes are applying different state and the keying/store path is the defect.
                info!(
                    position = self.position,
                    position_before = pos_before,
                    blob_digest = crate::kv_coordination::byte_digest(blob),
                    blob_len = blob.len(),
                    set_state_ms,
                    mode = "plane",
                    "ov-runtime: apply_warm_resume set position"
                );
                crate::kv_coordination::log_blob_tensors("apply_plane", 0, blob);
                self.kv_warm_pending = true;
                true
            }
            Err(e) => {
                warn!(error = %e, "ov-runtime: apply_warm_resume set_state failed; cold");
                false
            }
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

// -------- Issue-34 §8 multi-stage CAPTURE over ov-runtime's frameless transport --------
//
// ov-runtime's wire has no frame-kind header — it's a bare positional WireTensor exchange (F16
// hidden, I64 position, I32 token). A control frame is an **I8** tensor (a dtype no real frame
// uses ⇒ collision-free): `[opcode | capture_body_bytes]`. Stateful shards only (static/NPU shards
// drive a host-side KV ring, not OV state, so they don't participate). All `kv_coord`-gated.
#[cfg(feature = "kv_coord")]
const OPCODE_CAPTURE: u8 = 1;
#[cfg(feature = "kv_coord")]
const OPCODE_CAPTURE_ACK: u8 = 2;
/// Consume: `RESTORE(epoch)` set_states a rank's pulled slice; the ACK's data[1] is an all-or-nothing
/// verdict. On a fail verdict the head `ABORT`s — every rank resets cold (a partial restore corrupts
/// output). `ABORT` also clears a stray `kv_warm_pending` so a cold turn re-prefills clean.
#[cfg(feature = "kv_coord")]
const OPCODE_RESTORE: u8 = 3;
#[cfg(feature = "kv_coord")]
const OPCODE_RESTORE_ACK: u8 = 4;
#[cfg(feature = "kv_coord")]
const OPCODE_ABORT: u8 = 5;
#[cfg(feature = "kv_coord")]
const OPCODE_ABORT_ACK: u8 = 6;
/// H.1b (R2): CAPTURE whose body also carries the TENANT the turn belongs to
/// (`capture_body_bytes_v2`). A separate opcode, not a wider `OPCODE_CAPTURE` body: the v1 codec
/// enforces an exact length and hard-errors `"bad CAPTURE body"` mid-chain on a mismatch, so
/// widening it in place would break any chain whose ranks run different builds. Emitted only for a
/// non-empty tenant, so a chain that never names one stays byte-for-byte on the v1 frame.
#[cfg(feature = "kv_coord")]
const OPCODE_CAPTURE_V2: u8 = 7;

#[cfg(feature = "kv_coord")]
impl OvRuntimeEngine {
    /// True once this rank holds OV state worth coordinating (stateful, post-load).
    fn kv_stateful(&self) -> bool {
        self.static_kv.is_none()
    }

    /// Head/middle → downstream: send `CAPTURE(epoch, tokens)` as an I8 control tensor, await the ACK.
    /// A non-empty `tenant` upgrades the frame to `OPCODE_CAPTURE_V2` so the downstream rank — which
    /// never sees the `GenerationTask` — can tag its own capture with the same namespace.
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
            vec![OPCODE_CAPTURE]
        } else {
            vec![OPCODE_CAPTURE_V2]
        };
        if tenant.is_empty() {
            data.extend_from_slice(&crate::kv_coordination::capture_body_bytes(epoch, tokens));
        } else {
            data.extend_from_slice(&crate::kv_coordination::capture_body_bytes_v2(
                epoch, tokens, tenant,
            ));
        }
        let t = WireTensor::new(WireDType::I8, [1, 1, data.len() as u32], data);
        let ack = self.block_on(async move {
            let mut g = downstream.lock().await;
            g.send(&t)
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            // Bounded like the RESTORE ack: a peer that errors on the frame (e.g. an old build
            // meeting CAPTURE_V2) never ACKs, and an unbounded wait wedges the head while it holds
            // the downstream lock. Timing out degrades exactly as a failed capture already does —
            // the caller warns and the turn simply isn't cached.
            match tokio::time::timeout(RESTORE_ACK_TIMEOUT, g.recv()).await {
                Ok(Ok((ack, _))) => Ok(ack),
                Ok(Err(e)) => Err(EngineError::Backend(e.to_string())),
                Err(_) => Err(EngineError::Backend(
                    "ov-runtime: CAPTURE ack timed out; turn not cached".into(),
                )),
            }
        })?;
        if ack.dtype == WireDType::I8 && ack.data.first() == Some(&OPCODE_CAPTURE_ACK) {
            Ok(())
        } else {
            Err(EngineError::Backend("ov-runtime: bad CAPTURE ack".into()))
        }
    }

    /// Worker → upstream: ACK a control frame (`payload` carries e.g. the RESTORE verdict).
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
        info!(ack_opcode, "ov_tail_ctrl_ack_sent");
        Ok(())
    }

    /// Head/middle → downstream: `RESTORE(epoch)`; returns the chain's all-or-nothing verdict
    /// (ACK data[1] == 1 ⇒ every downstream rank restored). `Ok(false)` ⇒ caller must `ABORT` + cold.
    fn send_restore_downstream(&mut self, epoch: u64) -> EngineResult<bool> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let mut data = vec![OPCODE_RESTORE];
        data.extend_from_slice(&epoch.to_le_bytes());
        // Issue-34 multi-stage cross-chain: ship the downstream rank's pulled blob inline so it can
        // `set_state` (it has no local capture for a foreign chain's epoch). Absent ⇒ bare RESTORE
        // (same-chain path, where the rank restores from its own CAPTURE stash).
        // Prefer the epoch-keyed slot; on a miss fall back to the single stashed blob (2-stage move
        // stashes exactly one downstream slice per turn — see take_downstream_single). The miss is a
        // stash/restore epoch-key drift: the head keys the stash by the pulled rank's manifest tokens
        // while restore keys by its own warm prefix; log both so the residual drift is diagnosable
        // (3+-stage needs per-(epoch,rank) keying — follow-up).
        // This engine is the DRIVER, which is rank 0 by construction (rank>0 runs the worker engine),
        // so the RESTORE it sends always addresses rank 1.
        let down_rank: u16 = 1;
        let carried = self.kv.take_downstream(epoch, down_rank).or_else(|| {
            let n = self.kv.downstream_len();
            let single = self.kv.take_downstream_single(down_rank);
            warn!(
                epoch,
                stashed = n,
                recovered = single.is_some(),
                "ov_restore_carry_epoch_miss; single-slot fallback"
            );
            single
        });
        if let Some(blob) = carried {
            info!(epoch, blob_len = blob.len(), "ov_restore_carry_downstream");
            data.extend_from_slice(&blob);
        }
        let t = WireTensor::new(WireDType::I8, [1, 1, data.len() as u32], data);
        let ack = self.block_on(async move {
            let mut g = downstream.lock().await;
            g.send(&t)
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            match tokio::time::timeout(RESTORE_ACK_TIMEOUT, g.recv()).await {
                Ok(Ok((ack, _))) => Ok(ack),
                Ok(Err(e)) => Err(EngineError::Backend(e.to_string())),
                Err(_) => Err(EngineError::Backend(
                    "ov-runtime: RESTORE ack timed out; cold reprefill".into(),
                )),
            }
        })?;
        if ack.dtype == WireDType::I8 && ack.data.first() == Some(&OPCODE_RESTORE_ACK) {
            Ok(ack.data.get(1) == Some(&1))
        } else {
            Err(EngineError::Backend("ov-runtime: bad RESTORE ack".into()))
        }
    }

    /// Head/middle → downstream: `ABORT` (reset cold + clear warm); await ACK. Best-effort.
    fn send_abort_downstream(&mut self) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let t = WireTensor::new(WireDType::I8, [1, 1, 1], vec![OPCODE_ABORT]);
        self.block_on(async move {
            let mut g = downstream.lock().await;
            g.send(&t).await?;
            // Best-effort ack, bounded: a silent downstream must not wedge the abort path.
            let _ = tokio::time::timeout(RESTORE_ACK_TIMEOUT, g.recv()).await;
            Ok::<_, cascadia_transport::TransportError>(())
        })
        .map_err(|e| EngineError::Backend(e.to_string()))
    }

    /// Handle an inbound I8 control tensor on a worker (called transparently inside the recv loop).
    fn handle_inbound_control(&mut self, t: &WireTensor) -> EngineResult<()> {
        info!(opcode = ?t.data.first().copied(), is_last = self.spec.is_last_stage, "ov_tail_ctrl_recv");
        match t.data.first().copied() {
            // CAPTURE: snapshot this rank's KV under the head's epoch, chain downstream, ack up.
            // V2 additionally carries the turn's tenant, which tags the stash and rides the chain on.
            Some(op @ (OPCODE_CAPTURE | OPCODE_CAPTURE_V2)) => {
                let (epoch, tokens, tenant) = if op == OPCODE_CAPTURE_V2 {
                    crate::kv_coordination::parse_capture_body_v2(&t.data[1..]).ok_or_else(
                        || EngineError::Backend("ov-runtime: bad CAPTURE_V2 body".into()),
                    )?
                } else {
                    let (e, tk) = crate::kv_coordination::parse_capture_body(&t.data[1..])
                        .ok_or_else(|| {
                            EngineError::Backend("ov-runtime: bad CAPTURE body".into())
                        })?;
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
                        warn!(error = %e, "ov-runtime: CAPTURE chain downstream failed (best-effort)");
                    }
                }
                self.send_control_ack_upstream(OPCODE_CAPTURE_ACK, &[])
            }
            // RESTORE: set_state this rank's pulled slice + arm warm_pending; chain downstream; ack
            // the all-or-nothing verdict. A miss anywhere ⇒ verdict 0 ⇒ head ABORTs the chain.
            Some(OPCODE_RESTORE) => {
                let epoch = t
                    .data
                    .get(1..9)
                    .and_then(|b| b.try_into().ok())
                    .map(u64::from_le_bytes)
                    .ok_or_else(|| EngineError::Backend("ov-runtime: bad RESTORE body".into()))?;
                // Issue-34 multi-stage cross-chain: the head ships THIS rank's pulled blob inline
                // (bytes past the 8-byte epoch). Use it directly; else restore from a local CAPTURE
                // stash (the same-chain path, where the rank captured its own slice).
                let carried = t.data.get(9..).filter(|b| !b.is_empty());
                // Drain FIRST. In plane mode the head parks this rank's slice in the mailbox AND
                // still carries a blob inline, so a `||` placed after the carried/capture branch
                // short-circuits the drain away: the rank warms from the carried blob while the
                // plane slice is never read, `kv_handoff_applied_inline` never fires, and the
                // verdict is true so nothing aborts — a hollow warm. The parked slice is the
                // authoritative cross-chain data, so it wins; chain mode parks nothing, so this is
                // a false no-op there and the carried/capture path is unchanged.
                let local_ok = if self.drain_kv_handoff(epoch) {
                    true
                } else if let Some(blob) = carried {
                    let pos_before = self.position;
                    match self.runtime.set_state_blob(blob) {
                        Ok(()) => {
                            self.position =
                                crate::kv_coordination::kv_seq_from_blob(blob).unwrap_or(0) as i64;
                            self.kv_warm_pending = true;
                            // Probe A REFERENCE point: this is the certified byte-identical path, so
                            // its digest is the known-good value the plane apply must match.
                            info!(
                                epoch,
                                blob_len = blob.len(),
                                blob_digest = crate::kv_coordination::byte_digest(blob),
                                position = self.position,
                                position_before = pos_before,
                                mode = "chain",
                                "ov_tail_restore_carried"
                            );
                            crate::kv_coordination::log_blob_tensors("restore_chain", epoch, blob);
                            true
                        }
                        Err(e) => {
                            warn!(error = %e, "ov-runtime: set_state(carried) failed; rank cold");
                            false
                        }
                    }
                } else {
                    match self.kv.take_capture(epoch) {
                        Some((tokens, blob)) => match self.runtime.set_state_blob(&blob) {
                            Ok(()) => {
                                // Real KV depth, not the token count (off-by-one, see kv_seq_from_blob).
                                self.position = crate::kv_coordination::kv_seq_from_blob(&blob)
                                    .map(|s| s.min(tokens.len()))
                                    .unwrap_or(tokens.len())
                                    as i64;
                                self.kv_warm_pending = true;
                                true
                            }
                            Err(e) => {
                                warn!(error = %e, "ov-runtime: set_state failed; rank cold");
                                false
                            }
                        },
                        None => false,
                    }
                };
                let down_ok = if self.spec.is_last_stage {
                    true
                } else {
                    self.send_restore_downstream(epoch).unwrap_or(false)
                };
                self.send_control_ack_upstream(OPCODE_RESTORE_ACK, &[u8::from(local_ok && down_ok)])
            }
            // ABORT: reset cold + clear warm_pending; chain downstream; ack.
            Some(OPCODE_ABORT) => {
                let _ = self.runtime.reset_state();
                self.kv_warm_pending = false;
                self.position = 0;
                // Zeroing `position` disarms the drain's depth guard, so a still-parked slice would
                // apply on a later turn — warm rank under a cold head. This frame carries no epoch,
                // hence the epoch-blind discard.
                if self.kv_handoff.discard_any() {
                    info!(target: "cascadia::kv", event = "kv_handoff_discarded_on_abort");
                }
                if !self.spec.is_last_stage {
                    let _ = self.send_abort_downstream();
                }
                self.send_control_ack_upstream(OPCODE_ABORT_ACK, &[])
            }
            other => Err(EngineError::Backend(format!(
                "ov-runtime: unknown control opcode {other:?}"
            ))),
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
    /// `--packed-slots N`: serve N concurrent requests per inference through
    /// the packed multi-slot variant (continuous batching on a device that
    /// rejects batch > 1). 0 = off.
    pub packed_slots: u32,
    /// `--packed-prefix N`: reserve N KV slots as a read-only SHARED prefix
    /// that every packed slot may attend to — prefix caching without paging.
    /// Taken out of the same window, so it costs per-slot context. 0 = off.
    pub packed_prefix: u32,
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
    /// Resolved packed geometry: (slots, packed_seq, packed_context).
    packed_params: Option<(u32, u32, u32)>,
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
        // Packed multi-slot variant (`--packed-slots`). Like the prefill
        // variant it must share the decode variant's past-KV shape, so its
        // context is past_len + packed_seq.
        self.packed_params = match (self.static_params, self.packed_slots) {
            (_, 0) => None,
            (None, _) => {
                return Err(EngineError::InvalidConfig(
                    "--packed-slots requires a stateless static (--target npu) export; the \
                     stateful path keeps KV inside OV state, which cannot be partitioned \
                     per request"
                        .into(),
                ));
            }
            (Some((ctx, _, _)), want) => {
                let slots = stage_cfg.packed_slots.unwrap_or(0);
                let pseq = stage_cfg.packed_seq.unwrap_or(slots);
                let pctx = stage_cfg.packed_context.unwrap_or(0);
                let packed_xml = stage_dir.join("openvino_packed_model.xml");
                if slots == 0 || !packed_xml.exists() {
                    return Err(EngineError::InvalidConfig(format!(
                        "--packed-slots {want} needs a packed variant beside the decode IR; {} \
                         is missing. Build it with `python tools/packed_variant.py <stage_dir> \
                         --slots {want}` or re-export with --packed-slots {want}",
                        packed_xml.display()
                    )));
                }
                if want != slots {
                    return Err(EngineError::InvalidConfig(format!(
                        "--packed-slots {want} disagrees with the exported variant \
                         (packed_slots={slots}); the slot count is baked into the IR shape"
                    )));
                }
                if let Some(msg) = packed_geometry_error(slots, pseq) {
                    return Err(EngineError::InvalidConfig(msg));
                }
                let past_len = ctx - 1;
                if pctx != past_len + pseq {
                    return Err(EngineError::InvalidConfig(format!(
                        "packed_context={pctx} inconsistent: need past_len + packed_seq = {} so \
                         the packed and decode variants share one past-KV shape",
                        past_len + pseq
                    )));
                }
                if past_len / slots == 0 {
                    return Err(EngineError::InvalidConfig(format!(
                        "packed_slots={slots} exceeds the KV window ({past_len})"
                    )));
                }
                events.push(LoadProgress::message(format!(
                    "packed multi-slot variant: slots={slots} seq={pseq} context={pctx} \
                     per-request context={}",
                    past_len / slots
                )));
                Some((slots, pseq, pctx))
            }
        };

        // The packed variant does its own chunked prefill (a plan whose rows all
        // belong to one slot IS a causal chunk), so the separate prefill model
        // would never be used — skip compiling it and keep its weights off the
        // device. On NPU that is a whole compile (~100 s) and a second resident
        // weight copy saved.
        if self.packed_params.is_some() && self.static_prefill_params.take().is_some() {
            events.push(LoadProgress::message(String::from(
                "packed slots: skipping the chunked-prefill variant (packed inference covers \
                 prefill natively)",
            )));
        }

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
        // Resolved before any field of `self` is moved out below.
        let packed_plugin = self.plugin();
        let packed_xml = self
            .pipeline_dir
            .join(format!("stage_{}", self.rank))
            .join("openvino_packed_model.xml");
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
                sink: crate::packed::KV_SINK.min(past_len / 2),
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

        // Packed multi-slot state: its own compiled variant + per-slot KV.
        let packed = match self.packed_params {
            None => None,
            Some((slots, pseq, _pctx)) => {
                let (ctx, kvh, hd) = self.static_params.expect("checked in load");
                let past_len = (ctx - 1) as usize;
                let prt = OvRuntime::compile(
                    packed_xml.to_string_lossy().as_ref(),
                    &self.device,
                    &packed_plugin,
                )
                .map_err(map_ov_err)?;
                let players = resolve_static_layers(&prt, "packed variant")?
                    .into_iter()
                    .map(|l| crate::packed_exec::PackedLayer {
                        key_in: l.key_in,
                        val_in: l.val_in,
                        key_out: l.key_out,
                        val_out: l.val_out,
                    })
                    .collect::<Vec<_>>();
                let pcanon = resolve_canonical_inputs(&prt)?;
                let ids_in = pcanon.get("input_ids").cloned().unwrap_or_default();
                let hidden_in = pcanon.get("hidden_states").cloned().unwrap_or_default();
                if ids_in.is_empty() && hidden_in.is_empty() {
                    return Err(EngineError::Backend(
                        "packed variant has neither input_ids nor hidden_states".into(),
                    ));
                }
                let pos_in = pcanon.get("position_ids").cloned().ok_or_else(|| {
                    EngineError::Backend("packed variant missing position_ids".into())
                })?;
                // The packed variant replaces the 2D attention_mask with a 4D
                // per-row mask parameter; find it by name and take its dtype
                // from the IR so f16/f32 exports both work.
                let mut mask_in = None;
                let mut mask_dtype = ShimDType::F16;
                for idx in 0..prt.input_count() {
                    let aliases = prt.input_aliases(idx).map_err(map_ov_err)?;
                    if aliases.iter().any(|a| a.contains("attn_mask_4d")) {
                        mask_in = Some(prt.input_name(idx).map_err(map_ov_err)?);
                        mask_dtype = prt.input_dtype(idx).map_err(map_ov_err)?;
                    }
                }
                let mask_in = mask_in.ok_or_else(|| {
                    EngineError::Backend(
                        "packed variant missing the attn_mask_4d input — rebuild it with \
                         tools/packed_variant.py"
                            .into(),
                    )
                })?;
                let kv = crate::packed::PackedKv::new(
                    slots as usize,
                    past_len,
                    pseq as usize,
                    players.len(),
                    kvh as usize,
                    hd as usize,
                    ShimDType::F16,
                    self.packed_prefix as usize,
                );
                info!(
                    slots,
                    packed_seq = pseq,
                    region = kv.region,
                    shared_prefix = kv.prefix_capacity,
                    "packed multi-slot decode active"
                );
                Some(crate::packed_exec::PackedState::new(
                    prt, kv, players, ids_in, hidden_in, mask_in, pos_in, mask_dtype,
                ))
            }
        };

        Ok(Box::new(OvRuntimeEngine {
            packed,
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
            wire_dead: WireDeadLatch::default(),
            awaiting_token_seq: 0,
            inbound_seq: None,
            consecutive_token_timeouts: 0,
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
            #[cfg(feature = "kv_coord")]
            kv_share: std::sync::Arc::new(std::sync::Mutex::new(
                crate::kv_coordination::OvKvCache::default(),
            )),
            #[cfg(feature = "kv_coord")]
            kv_handoff: std::sync::Arc::new(crate::kv_coordination::KvHandoffMailbox::new()),
            #[cfg(feature = "kv_coord")]
            kv_warm_pending: false,
            #[cfg(feature = "kv_coord")]
            plane_restore: std::env::var("CASCADIA_KV_PLANE_RESTORE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
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

    /// Byte-level detokenizers emit one U+FFFD per byte they cannot yet decode,
    /// so a multi-byte glyph arriving across reads appears as a run that later
    /// RESOLVES — the decode is not prefix-stable, and can even get shorter.
    /// The old delta (`full.strip_prefix(last).unwrap_or(full)`) re-emitted the
    /// WHOLE text at that point, so the client saw the prefix twice.
    #[test]
    fn delta_does_not_duplicate_when_a_replacement_char_resolves() {
        // Successive full decodes as tokens arrive: "caf", an undecodable byte,
        // then the completed glyph, then more text.
        let decodes = ["caf", "caf\u{FFFD}", "café", "café au lait"];

        let mut emitted: Vec<u8> = Vec::new();
        let mut client = String::new();
        for (i, full) in decodes.iter().enumerate() {
            let running = i + 1 < decodes.len();
            let (delta, _) = advance_emitted(&mut emitted, full.as_bytes(), running);
            client.push_str(&delta);
        }
        assert_eq!(
            client, "café au lait",
            "the client stream must reconstruct the final decode exactly"
        );
        assert!(
            !client.contains('\u{FFFD}'),
            "an unresolved replacement char must never be handed out: {client:?}"
        );

        // The regression this replaces, computed the old way for contrast.
        let mut old = String::new();
        let mut last = String::new();
        for full in decodes {
            old.push_str(full.strip_prefix(last.as_str()).unwrap_or(full));
            last = full.to_string();
        }
        assert!(
            old.matches("caf").count() > 1,
            "old logic duplicated the prefix, which is the bug: {old:?}"
        );
    }

    /// A packed variant narrower than its own slot count can never decode every
    /// slot: `PackedPlan::decode` only lays down `packed_seq` rows, so slots
    /// beyond that get no row, and the step then samples a logits row the
    /// output does not have. That surfaces as an untyped step error which the
    /// runner charges to whichever stream happened to poll. Refuse the geometry
    /// at load instead.
    #[test]
    fn packed_variant_narrower_than_its_slot_count_is_refused() {
        assert!(
            packed_geometry_error(8, 8).is_none(),
            "seq == slots is the norm"
        );
        assert!(
            packed_geometry_error(8, 16).is_none(),
            "a wider query window than slots is legitimate (fatter prefill chunks)"
        );
        let msg = packed_geometry_error(8, 4).expect("seq < slots must be refused");
        assert!(msg.contains('8') && msg.contains('4'), "{msg}");
        assert!(msg.contains("packed_seq"), "{msg}");
    }

    /// A prompt longer than its slot's KV region cannot be served honestly: the
    /// packed ring would evict its head and answer from a truncated prompt, so
    /// admission must refuse it. The boundary is strict — a prompt that exactly
    /// fills the region still fits.
    #[test]
    fn over_region_prompt_is_refused_at_admission() {
        assert!(
            packed_prompt_too_long(255, 255).is_none(),
            "a prompt that exactly fills the region is servable"
        );
        assert!(packed_prompt_too_long(1, 255).is_none());
        let msg = packed_prompt_too_long(256, 255).expect("over-region must be refused");
        assert!(msg.contains("256") && msg.contains("255"), "{msg}");
        assert!(
            msg.contains("--static-context") && msg.contains("--packed-slots"),
            "the message must name both remedies: {msg}"
        );
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
            sink: crate::packed::KV_SINK.min(past_len / 2),
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
    fn packed_plan_frame_round_trips() {
        let rows = vec![
            Some((0usize, 7i64, 0usize)),
            None,
            Some((3, 12, 12)),
            Some((1, 129, 40)),
        ];
        let frame = encode_wire_plan(&rows);
        assert_eq!(frame.shape, [1, 3, 4]);
        let back = decode_wire_plan(&frame, 4).expect("round trip");
        assert_eq!(back, rows);
    }

    #[test]
    fn packed_plan_frame_rejects_bad_frames() {
        // wrong rank/shape (a non-packed peer's lead frame)
        let lead = encode_wire_lead(5, None);
        assert!(decode_wire_plan(&lead, 4).is_err());
        // slot id beyond this stage's slot count
        let frame = encode_wire_plan(&[Some((9usize, 0i64, 0usize))]);
        let err = decode_wire_plan(&frame, 4).unwrap_err().to_string();
        assert!(err.contains("slot 9"), "{err}");
        // negative position would wrap in the ring math
        let mut bad = encode_wire_plan(&[Some((0usize, 1i64, 0usize))]);
        bad.data[8..16].copy_from_slice(&(-3i64).to_le_bytes());
        assert!(decode_wire_plan(&bad, 4).is_err());
    }

    #[test]
    fn packed_token_frame_round_trips() {
        let toks = vec![5i32, 0, 128009, 42];
        let back = decode_wire_tokens(&encode_wire_tokens(&toks)).expect("round trip");
        assert_eq!(back, toks);
        // a hidden (F32) frame where tokens were expected is a hard error
        let wrong = WireTensor::new(WireDType::F32, [1, 1, 2], vec![0u8; 8]);
        assert!(decode_wire_tokens(&wrong).is_err());
    }

    /// The relay's NACK is an EMPTY token frame — it must be wire-valid
    /// (shape/payload consistent so the transport delivers it), detectable
    /// BEFORE `decode_wire_tokens` (which rejects empties as malformed), and
    /// impossible to confuse with a real reply (a real reply always carries
    /// one token per row, S >= 1).
    ///
    /// Pinned to the NAMED pair, [`packed_nack_frame`] + [`is_packed_nack`],
    /// rather than to an open-coded `elements() == 0`: the sender and the
    /// recogniser only work if they agree, and an anonymous predicate at the
    /// consumer could drift from the producer without anything failing here.
    #[test]
    fn packed_nack_frame_is_empty_and_detectable() {
        let nack = packed_nack_frame();
        // The frame the relay puts on the wire is exactly "no tokens".
        assert_eq!(nack, encode_wire_tokens(&[]));
        assert_eq!(nack.shape, [1, 1, 0]);
        assert!(nack.data.is_empty());
        assert_eq!(nack.elements(), Some(0));
        // Same dtype as a real token reply: only the emptiness distinguishes
        // it, so a peer reading the header cannot mistake it for a hidden
        // state or a plan frame.
        assert_eq!(nack.dtype, WireDType::I64);
        // Payload length and shape agree, or the transport would reject the
        // frame outright and the NACK would never arrive.
        assert_eq!(
            nack.data.len() as u64,
            nack.elements().unwrap() * nack.dtype.bytes_per_element() as u64
        );
        assert!(is_packed_nack(&nack));
        assert!(decode_wire_tokens(&nack).is_err()); // NACK check must come first

        // No real reply is ever mistaken for one, at any row count — and
        // each of them still decodes to exactly its tokens.
        for rows in 1..=8usize {
            let toks: Vec<i32> = (0..rows as i32).collect();
            let real = encode_wire_tokens(&toks);
            assert!(!is_packed_nack(&real), "{rows}-row reply read as a NACK");
            assert_ne!(real.elements(), Some(0));
            assert_eq!(decode_wire_tokens(&real).expect("real reply"), toks);
        }
    }

    /// Aborting a packed batch types its cause, so classification never rides
    /// on the message text. A downstream NACK or a plain step failure lost the
    /// batch, not the link, and must not push a relay rank into a rebuild — but
    /// this stage's own dead socket has to stay connection-fatal.
    #[test]
    fn packed_abort_error_types_the_cause() {
        // A downstream NACK arrives already typed — no double wrapping.
        let nack = EngineError::BatchAborted("downstream NACKed this batch".into());
        let out = packed_abort_error(nack);
        assert_eq!(
            out.to_string(),
            "batch aborted: downstream NACKed this batch"
        );
        assert!(!out.is_connection_fatal());

        // A local step failure becomes an abort, keeping its cause readable.
        let out = packed_abort_error(EngineError::Backend("bad logits shape [1, 0]".into()));
        assert!(matches!(out, EngineError::BatchAborted(_)), "{out:?}");
        assert!(out.to_string().contains("bad logits shape [1, 0]"), "{out}");
        assert!(!out.is_connection_fatal());

        // This stage's own poisoned socket keeps its type AND its fatality.
        let dead = EngineError::Backend("packed token recv: recv_exact timed out after 60s".into());
        let out = packed_abort_error(dead);
        assert!(matches!(out, EngineError::Backend(_)), "{out:?}");
        assert!(out.is_connection_fatal(), "{out}");
    }

    /// A dead upstream fails the step AND the NACK that reports it, so the two
    /// arrive together. Returning only the send error would drop the reason
    /// the batch died — the operator would see "packed token send: …" and
    /// never learn it was a plan decode error, a failed inference, or a
    /// downstream NACK. Both texts must survive, and the send error must stay
    /// classifiable so the relay loop still exits for a supervisor rebuild.
    #[test]
    fn nack_send_failure_carries_the_step_error_too() {
        let send = EngineError::Backend("packed token send: broken pipe".into());
        let body = EngineError::Backend("packed plan decode: slot 9 beyond 4 slots".into());
        let combined = nack_send_error(send, Some(&body));
        let msg = combined.to_string();
        assert!(msg.contains("packed token send: broken pipe"), "{msg}");
        assert!(msg.contains("slot 9 beyond 4 slots"), "{msg}");
        // The transport failure still classifies: a dead link must not be
        // downgraded to a retryable error just because it now carries context.
        assert!(combined.is_connection_fatal(), "{msg}");

        // A successful step that merely failed to send back keeps the send
        // error untouched — there is no root cause to attach.
        let send = EngineError::Backend("packed token send: broken pipe".into());
        let alone = nack_send_error(send, None);
        assert_eq!(
            alone.to_string(),
            "backend error: packed token send: broken pipe"
        );
    }

    /// The dead-wire latch arms on this stage's own dead socket and on
    /// nothing else. A lost batch (a downstream NACK, a bad plan, a failed
    /// inference) leaves a working link and MUST leave the engine serving —
    /// arming there would take a healthy rank 0 permanently offline on one
    /// unlucky batch.
    #[test]
    fn wire_dead_latch_arms_only_on_a_connection_fatal_cause() {
        let mut latch = WireDeadLatch::default();
        assert!(latch.cause().is_none());
        assert!(latch.fail_fast_error().is_none());

        // A downstream NACK: structurally non-fatal, whatever it says.
        assert!(!latch.observe(&EngineError::BatchAborted(
            "downstream stage failed its packed step and NACKed this batch".into()
        )));
        // Even one quoting words the substring classifier hunts for — the
        // dead socket in that text belongs to some OTHER rank.
        assert!(!latch.observe(&EngineError::BatchAborted(
            "the packed step failed: backend error: packed token recv: broken pipe".into()
        )));
        // A plain local step failure.
        assert!(!latch.observe(&EngineError::Backend(
            "packed plan decode: slot 9 beyond 4 slots".into()
        )));
        assert!(latch.cause().is_none(), "{:?}", latch.cause());
        assert!(latch.fail_fast_error().is_none());

        // This stage's own poisoned socket, as `packed_abort_error` passes it
        // through: fatal, and it arms the latch.
        let dead = packed_abort_error(EngineError::Backend(
            "packed token recv: recv_exact timed out after 60s".into(),
        ));
        assert!(latch.observe(&dead));
        let cause = latch.cause().expect("latched");
        assert!(cause.contains("recv_exact timed out after 60s"), "{cause}");
    }

    /// Once latched, every request is refused with an error that is (a)
    /// connection-fatal, so the failure stays visible to the API's readiness
    /// tracker and to any supervisor plumbing added later, (b) never
    /// `BatchAborted`, which is fatality-false by construction, and (c)
    /// carrying the original cause so the operator learns what killed the wire.
    #[test]
    fn latched_wire_fails_fast_with_a_connection_fatal_error() {
        let mut latch = WireDeadLatch::default();
        latch.observe(&EngineError::Backend(
            "packed token recv: recv_exact timed out after 60s".into(),
        ));
        let err = latch.fail_fast_error().expect("latched");
        assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
        assert!(err.is_connection_fatal(), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("recv_exact timed out after 60s"), "{msg}");
        assert!(msg.contains("restart"), "{msg}");

        // Fatality must not depend on the stored cause's wording. `NotConnected`
        // is fatal STRUCTURALLY and displays as "not YET connected", which no
        // substring rule matches — a fail-fast error that merely echoed the
        // cause would silently become non-fatal here.
        let mut latch = WireDeadLatch::default();
        assert!(EngineError::NotConnected.is_connection_fatal());
        assert!(!EngineError::NotConnected.to_string().contains(
            // the classifier's substring, absent from this Display
            "not connected"
        ));
        latch.observe(&EngineError::NotConnected);
        let err = latch.fail_fast_error().expect("latched");
        assert!(err.is_connection_fatal(), "{err}");
        // And the chunk text a retired task receives names the abort once.
        let chunk_msg = packed_abort_message(&err);
        assert!(
            chunk_msg.starts_with("packed batch aborted: "),
            "{chunk_msg}"
        );
        assert!(
            !chunk_msg.contains("batch aborted: batch aborted"),
            "{chunk_msg}"
        );
    }

    /// The operator-facing `error!` fires on the LATCHING TRANSITION only —
    /// `observe` returns true exactly once, so a stage that keeps aborting
    /// batches does not re-log the same dead link per batch. The first cause
    /// is also the one kept: it is the failure that actually killed the wire,
    /// every later one is a consequence.
    #[test]
    fn wire_dead_latch_transition_reports_once_and_keeps_the_first_cause() {
        let mut latch = WireDeadLatch::default();
        assert!(latch.observe(&EngineError::Backend(
            "packed token recv: recv_exact timed out after 60s".into()
        )));
        // Every later fatal cause: no transition, no overwrite.
        for _ in 0..3 {
            assert!(!latch.observe(&EngineError::NotConnected));
            assert!(!latch.observe(&EngineError::Backend(
                "packed plan send: not connected".into()
            )));
        }
        let cause = latch.cause().expect("latched");
        assert!(cause.contains("recv_exact timed out after 60s"), "{cause}");
        assert!(!cause.contains("packed plan send"), "{cause}");
        // A benign cause after latching cannot disarm it either.
        assert!(!latch.observe(&EngineError::BatchAborted("lost batch".into())));
        assert!(latch.fail_fast_error().is_some());
    }

    /// Prefill chunks (several rows for one slot) get the widened reply
    /// budget; decode frames (at most one row per slot) keep the strict
    /// deadline. Idle rows never count.
    #[test]
    fn plan_prefill_detection_by_duplicate_slot() {
        // decode: three distinct slots + an idle row
        assert!(!plan_has_prefill_rows(&[
            Some((0, 5, 0)),
            Some((1, 9, 0)),
            None,
            Some((3, 2, 0)),
        ]));
        // prefill chunk: slot 2 owns several rows
        assert!(plan_has_prefill_rows(&[
            Some((2, 0, 0)),
            Some((2, 1, 0)),
            Some((2, 2, 0)),
            None,
        ]));
        // single-row prefill is indistinguishable from decode — strict
        // deadline by design
        assert!(!plan_has_prefill_rows(&[Some((0, 0, 0))]));
        assert!(!plan_has_prefill_rows(&[]));
    }

    /// Which token sits in each ring slot, recovered by matching the
    /// deterministic per-token byte pattern back.
    fn ring_tokens(ring: &StaticKv, n: usize) -> Vec<Option<usize>> {
        let slot = ring.head_dim * ring.elem_bytes;
        (0..ring.past_len)
            .map(|i| {
                (0..n).find(|&t| {
                    (0..slot).all(|b| ring.key_buf[0][i * slot + b] == tok_byte(t, 0, b, false))
                })
            })
            .collect()
    }

    /// The single-task ring drops its OLDEST entry once full, which evicts
    /// token 0 — the attention sink. Same defect the packed ring had, at the
    /// full window instead of a per-slot region, so it needs a conversation
    /// past `past_len` rather than past `region` to show. Measured on CPU: a
    /// run crossing 1023 total tokens degenerated into "ttettettett...".
    #[test]
    fn single_task_slide_preserves_the_attention_sink() {
        let mut ring = test_ring(10, 1, 1, 1); // sink = min(4, 10/2) = 4
        assert_eq!(ring.sink, 4);
        for t in 0..16 {
            ring.begin_token(t);
            let k = present_seq1(&ring, t, false);
            ring.absorb_layer(0, false, &k);
        }
        assert_eq!(
            ring_tokens(&ring, 16),
            vec![0, 1, 2, 3, 10, 11, 12, 13, 14, 15]
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>(),
            "tokens 0-3 are the sinks and must survive; only the tail slides"
        );
    }

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

    /// A blob import must drop CACHE_DIR (an import never writes the compile
    /// cache, and a core-level cache property on `import_model` risks an
    /// unsupported-property error that breaks AOT blob import outright) while
    /// leaving every other plugin property intact.
    #[test]
    fn import_plugin_strips_cache_dir_keeps_rest() {
        let p = PluginConfig::new()
            .with("CACHE_DIR", "/tmp/x")
            .with("PERFORMANCE_HINT", "LATENCY")
            .with("NPU_USE_NPUW", "YES");
        let out = import_plugin(&p);
        assert!(!out.entries.iter().any(|(k, _)| k == "CACHE_DIR"));
        assert!(out
            .entries
            .iter()
            .any(|(k, v)| k == "PERFORMANCE_HINT" && v == "LATENCY"));
        assert!(out
            .entries
            .iter()
            .any(|(k, v)| k == "NPU_USE_NPUW" && v == "YES"));
        assert_eq!(out.entries.len(), 2);
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

    // -------- per-hop sequence echo (token desync guard) --------
    //
    // `OvRuntimeEngine` can't be constructed without a compiled OpenVINO IR
    // (stub mode errors), so these exercise the extracted wire helpers over real
    // ActivationServer/ActivationClient loopback pairs, mirroring the transport
    // crate's roundtrip tests.
    //
    // send_hidden_frames / recv_hidden_frames / recv_token_seq_checked ARE the
    // bodies of send_hidden_downstream / recv_hidden_from_upstream /
    // recv_token_from_downstream, so those three are covered here.
    // send_token_to_upstream is NOT: its body inlines the encode, the lock and
    // the send, so what these tests cover of it is the encoder alone.
    //
    // Still untested for want of a constructible engine: the field wiring
    // itself — the stamp/echo assignments and the escalation counter — and the
    // engine-lock release that motivates the bounded wait. Those need either a
    // seam that does not exist yet or hardware.

    /// Stand up a connected (client, server) loopback pair. `client` is the
    /// engine's `downstream` (an ActivationClient); `server` plays the
    /// downstream/upstream peer.
    async fn loopback() -> (ActivationClient, ActivationServer) {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        let server = h.await.unwrap();
        (client, server)
    }

    #[tokio::test]
    async fn recv_token_discards_stale_then_returns_correct() {
        let (client, mut server) = loopback().await;
        let awaiting = 5u32;
        // A stale orphan (wrong echoed seq) precedes the correct token.
        server.send(&encode_token_with_seq(99, 4)).await.unwrap();
        server
            .send(&encode_token_with_seq(42, awaiting))
            .await
            .unwrap();
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let tok = recv_token_seq_checked(&downstream, awaiting, false)
            .await
            .unwrap();
        assert_eq!(tok, 42, "stale token must be skipped, correct one returned");
    }

    /// A matching echo returns that token. (Named for what it asserts — the
    /// value — not for timing, which nothing here measures.)
    #[tokio::test]
    async fn recv_token_seq_match_returns_the_token() {
        let (client, mut server) = loopback().await;
        let awaiting = 7u32;
        server
            .send(&encode_token_with_seq(123, awaiting))
            .await
            .unwrap();
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let tok = recv_token_seq_checked(&downstream, awaiting, false)
            .await
            .unwrap();
        assert_eq!(tok, 123);
    }

    #[test]
    fn token_frame_carries_token_and_seq() {
        // send_token_to_upstream emits I32[1,1,2] = [token, inbound_seq].
        let t = encode_token_with_seq(42, 9);
        assert_eq!(t.dtype, WireDType::I32);
        assert_eq!(t.shape, [1, 1, 2]);
        assert_eq!(t.data.len(), 8);
        assert_eq!(&t.data[0..4], &42i32.to_le_bytes());
        assert_eq!(&t.data[4..8], &9i32.to_le_bytes());
        assert_eq!(decode_token_with_seq(&t).unwrap(), (42, 9));
    }

    /// The token frame must be validated by dtype AND shape, not length alone.
    /// Both rejections below are 8-byte frames that a length-only check waved
    /// through — the packed one silently produced a token on the first exchange
    /// of a mismatched pipeline, because `echo_seq` decoded as 0 and the first
    /// stamped seq was 0 too.
    /// A NACK for the generation we are waiting on aborts it — and does so as
    /// `BatchAborted`, which is structurally non-fatal, so a relay rank backs
    /// off and keeps driving instead of exiting for a supervisor rebuild.
    #[tokio::test]
    async fn matching_seq_nack_aborts_the_generation_without_killing_the_link() {
        let (client, mut server) = loopback().await;
        let awaiting = 5u32;
        server
            .send(&encode_token_with_seq(NACK_TOKEN, awaiting))
            .await
            .unwrap();
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let err = recv_token_seq_checked(&downstream, awaiting, false)
            .await
            .unwrap_err();
        // `Other`, not `TimedOut`: a NACK proves the link delivered bytes, so
        // it must not count toward relay escalation.
        assert!(matches!(err, TokenWaitFailure::Other(_)), "{err:?}");
        let err = err.into_error();
        assert!(matches!(err, EngineError::BatchAborted(_)), "{err:?}");
        assert!(
            !err.is_connection_fatal(),
            "a NACKed generation must not tear the stage down: {err}"
        );
    }

    /// The reason the NACK rides the ordinary seq-tagged frame instead of a
    /// distinctly-shaped one.
    ///
    /// Upstream and downstream run the same token budget, so a timeout-driven
    /// NACK is typically sent AFTER the upstream already gave up: it sits in
    /// the socket and is read by the NEXT request. Carrying the seq means the
    /// existing stale-discard loop throws it away. A shapewise NACK would carry
    /// no seq, so nothing could reject it and it would abort the next, healthy
    /// generation instead.
    #[tokio::test]
    async fn a_stale_nack_cannot_poison_the_next_generation() {
        let (client, mut server) = loopback().await;
        // The abandoned generation's NACK, still in the socket...
        server
            .send(&encode_token_with_seq(NACK_TOKEN, 4))
            .await
            .unwrap();
        // ...then the current generation's real token.
        server.send(&encode_token_with_seq(42, 5)).await.unwrap();
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let tok = recv_token_seq_checked(&downstream, 5, false).await.unwrap();
        assert_eq!(tok, 42, "a stale NACK must be discarded, not honoured");
    }

    /// A wait that only ever sees WRONG-seq frames must report why it gave up.
    /// Without the discard count and the last echoed seq, a version skew or a
    /// downstream answering a different request is indistinguishable from a
    /// slow stage, and the operator goes looking at the network.
    #[tokio::test]
    async fn a_wait_that_only_discards_reports_the_discards() {
        let (client, mut server) = loopback().await;
        cascadia_transport::set_activation_timeout_secs(1);
        // Three frames, none of them echoing the seq we are waiting on.
        for s in [11u32, 12, 13] {
            server.send(&encode_token_with_seq(5, s)).await.unwrap();
        }
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let err = recv_token_seq_checked(&downstream, 99, false)
            .await
            .unwrap_err();
        cascadia_transport::set_activation_timeout_secs(0);

        assert!(matches!(err, TokenWaitFailure::TimedOut(_)), "{err:?}");
        let msg = err.into_error().to_string();
        assert!(msg.contains("expected seq 99"), "{msg}");
        assert!(msg.contains("3 stale token frame(s)"), "{msg}");
        assert!(msg.contains("last echoing seq 13"), "{msg}");
        assert!(msg.contains("same build"), "no remedy named: {msg}");
    }

    /// A wait that saw nothing at all says so, rather than implying frames were
    /// discarded — the two point at completely different causes.
    #[tokio::test]
    async fn a_silent_wait_says_no_frame_arrived() {
        let (client, _server) = loopback().await;
        cascadia_transport::set_activation_timeout_secs(1);
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let err = recv_token_seq_checked(&downstream, 7, false)
            .await
            .unwrap_err();
        cascadia_transport::set_activation_timeout_secs(0);
        let msg = err.into_error().to_string();
        assert!(msg.contains("no token frame arrived"), "{msg}");
        // The transport's own wording is carried as the cause, not replaced.
        assert!(msg.contains("frame-start wait timed out"), "{msg}");
    }

    /// #40: a bounded frame-start timeout must classify NON-fatal at the ENGINE
    /// layer too, where the verdict is a substring rule over the flattened
    /// message.
    ///
    /// This assertion lives here, in the crate that depends on BOTH
    /// `cascadia-transport` and `cascadia-engine`, and builds the string from
    /// the real `TransportError` Display. The equivalent test in
    /// `cascadia-engine` hardcoded its own copy of the text — and that crate
    /// does not depend on the transport at all, so rewording the `#[error]`
    /// attribute left the test green while production flipped to fatal and
    /// started dropping a socket the engine cannot re-dial. A pin that cannot
    /// observe what it pins is not a pin.
    #[test]
    fn transport_frame_start_timeout_is_non_fatal_at_the_engine_layer() {
        let display = cascadia_transport::TransportError::FrameStartTimeout(
            std::time::Duration::from_secs(120),
        )
        .to_string();
        assert!(
            !EngineError::Backend(display.clone()).is_connection_fatal(),
            "a retryable frame-start timeout must not read as fatal: {display}"
        );
        // The sibling ceiling stays fatal — the classifier is narrowed, not
        // blunted, and these two differ only in which recv asked.
        let ceiling = cascadia_transport::TransportError::FrameIdleCeiling(
            std::time::Duration::from_secs(900),
        )
        .to_string();
        assert!(
            EngineError::Backend(ceiling.clone()).is_connection_fatal(),
            "the idle ceiling must stay fatal: {ceiling}"
        );
    }

    /// Whichever exit fires, the wait must stay RETRYABLE. The head cannot
    /// re-dial its downstream, so a token timeout classifying fatal would drop
    /// the socket permanently — the #40 brick this bounded recv exists to avoid.
    #[test]
    fn a_token_wait_timeout_is_never_connection_fatal() {
        for (discarded, last) in [(0u64, None), (7, Some(3u32))] {
            let e = token_wait_timeout_error(
                std::time::Duration::from_secs(120),
                42,
                discarded,
                last,
                &cascadia_transport::TransportError::FrameStartTimeout(
                    std::time::Duration::from_secs(120),
                )
                .to_string(),
            );
            assert!(!e.is_connection_fatal(), "{e}");
        }
    }

    /// The token budget: decode gets the base `recv_timeout`, prefill gets the
    /// widened one, and BOTH are capped so an operator-raised `recv_timeout`
    /// cannot grow the engine-lock hold without bound (#40).
    ///
    /// This was untestable while the `min()` lived inline in an async fn that
    /// needed a live socket — flipping it to `max()` passed the whole suite.
    #[test]
    fn token_recv_deadline_widens_for_prefill_and_clamps_both() {
        use std::time::Duration;
        let f = cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR;
        // Below the ceiling: the configured value governs, x factor for prefill.
        assert_eq!(
            token_recv_deadline(Duration::from_secs(60), false),
            Duration::from_secs(60)
        );
        assert_eq!(
            token_recv_deadline(Duration::from_secs(60), true),
            Duration::from_secs(60) * f
        );
        // At the rig's config: decode sits exactly on the ceiling, and prefill
        // gets the same 1200s `recv_reply_prefill` used to give it.
        assert_eq!(
            token_recv_deadline(Duration::from_secs(120), false),
            TOKEN_RECV_DEADLINE_CEILING
        );
        assert_eq!(
            token_recv_deadline(Duration::from_secs(120), true),
            TOKEN_RECV_DEADLINE_CEILING * f
        );
        // Above it, the cap binds on both paths — this is the assertion a
        // `max()` typo fails.
        assert_eq!(
            token_recv_deadline(Duration::from_secs(600), false),
            TOKEN_RECV_DEADLINE_CEILING
        );
        assert_eq!(
            token_recv_deadline(Duration::from_secs(600), true),
            TOKEN_RECV_DEADLINE_CEILING * f
        );
        // An absurd configured base clamps rather than panicking on overflow.
        assert_eq!(
            token_recv_deadline(Duration::MAX, true),
            TOKEN_RECV_DEADLINE_CEILING * f
        );
    }

    /// A frame-start timeout is the ONLY token-wait failure that says anything
    /// about the link, so it is the only one that may count toward escalating a
    /// relay rank. Pinned on the TYPED split rather than on message text —
    /// keying escalation off a substring is the fragility `BatchAborted` exists
    /// to end.
    #[tokio::test]
    async fn only_a_timeout_counts_toward_relay_escalation() {
        // Silent peer → the bounded frame-start elapses → TimedOut.
        let (client, _server) = loopback().await;
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        cascadia_transport::set_activation_timeout_secs(1);
        let err = recv_token_seq_checked(&downstream, 0, false)
            .await
            .unwrap_err();
        cascadia_transport::set_activation_timeout_secs(0);
        assert!(matches!(err, TokenWaitFailure::TimedOut(_)), "{err:?}");

        // A malformed frame is an answer, not a silence: bytes arrived, so the
        // link is not the suspect and the counter must reset.
        let (client, mut server) = loopback().await;
        server.send(&encode_wire_tokens(&[7])).await.unwrap();
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let err = recv_token_seq_checked(&downstream, 0, false)
            .await
            .unwrap_err();
        assert!(matches!(err, TokenWaitFailure::Other(_)), "{err:?}");
    }

    /// The streak counts CONSECUTIVE silences and is cleared by any answer.
    ///
    /// The reset is the load-bearing half: without it, three timeouts spread
    /// across an otherwise healthy day would eventually tear down a working
    /// relay rank. Note a rebuilt middle also costs the head a restart (it
    /// cannot re-dial), so a false positive takes down two ranks, not one.
    #[test]
    fn timeout_streak_counts_silence_and_resets_on_any_answer() {
        assert_eq!(next_timeout_streak(0, true), 1);
        assert_eq!(next_timeout_streak(1, true), 2);
        assert_eq!(next_timeout_streak(2, true), 3);
        // A token, a NACK, or a malformed frame all land here.
        assert_eq!(next_timeout_streak(2, false), 0);
        assert_eq!(next_timeout_streak(0, false), 0);
        // A stage shouting into the void for a very long time must not wrap
        // back under the threshold.
        assert_eq!(next_timeout_streak(u32::MAX, true), u32::MAX);
    }

    /// Escalation needs BOTH a failed step and a full streak.
    #[test]
    fn relay_escalates_only_on_a_full_streak_of_failures() {
        let n = RELAY_TOKEN_TIMEOUTS_BEFORE_EXIT;
        // Short of the threshold: keep serving.
        for s in 0..n {
            assert!(!should_escalate(s, true), "escalated early at streak {s}");
        }
        assert!(should_escalate(n, true));
        assert!(should_escalate(n + 1, true));
        // A step that SUCCEEDED must never trip the exit, whatever the streak —
        // the streak is only cleared on the recv path, and a success here means
        // the pipeline just delivered.
        for s in [0, n, n + 5, u32::MAX] {
            assert!(!should_escalate(s, false), "escalated on success at {s}");
        }
    }

    /// The sequence that distinguishes "dead link" from "unlucky day": an answer
    /// in the middle of a run of timeouts must prevent the exit entirely.
    #[test]
    fn an_answer_between_timeouts_prevents_escalation() {
        let n = RELAY_TOKEN_TIMEOUTS_BEFORE_EXIT;

        // Uninterrupted silence -> escalates exactly at the threshold.
        let mut streak = 0;
        let mut fired_at = None;
        for attempt in 1..=(n + 2) {
            streak = next_timeout_streak(streak, true);
            if should_escalate(streak, true) && fired_at.is_none() {
                fired_at = Some(attempt);
            }
        }
        assert_eq!(fired_at, Some(n), "should exit on the Nth consecutive miss");

        // Same number of timeouts, one answer partway through -> never exits.
        let mut streak = 0;
        for timed_out in [true, true, false, true, true, true] {
            streak = next_timeout_streak(streak, timed_out);
            if timed_out && should_escalate(streak, true) {
                // Only legitimate once the post-reset run reaches the threshold.
                assert!(streak >= n, "escalated on a broken streak: {streak}");
            }
        }
        // Two timeouts, an answer, then three: the trailing run is what counts.
        assert_eq!(streak, 3);
    }

    /// The escalation error must be classifiable by TYPE. `Io(TimedOut)` is
    /// what makes the relay loop exit for a supervisor rebuild; a `Backend`
    /// string that merely happens to contain a fatal substring would be one
    /// rewording away from silently ceasing to escalate.
    #[test]
    fn relay_escalation_error_is_structurally_connection_fatal() {
        let e = EngineError::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "downstream stage stopped answering (3 consecutive token-wait timeouts)",
        ));
        assert!(e.is_connection_fatal());
        // And a NACK, which shares the "downstream failed" wording space, does
        // NOT escalate however it is phrased.
        assert!(!EngineError::BatchAborted(
            "downstream stage failed its step and NACKed this generation".into()
        )
        .is_connection_fatal());
    }

    /// The sentinel must be unreachable as a real token: sampling returns a
    /// vocab INDEX from `argmax_last_row`, which starts at `0usize`.
    #[test]
    fn nack_sentinel_can_never_be_a_sampled_token() {
        assert!(NACK_TOKEN < 0);
        let logits = [0.1f32, 0.9, 0.3];
        assert!(argmax_last_row(&logits, 3) >= 0);
        // And it survives the wire as itself.
        assert_eq!(
            decode_token_with_seq(&encode_token_with_seq(NACK_TOKEN, 7)).unwrap(),
            (NACK_TOKEN, 7)
        );
    }

    #[test]
    fn token_frame_rejects_foreign_8_byte_frames() {
        // A packed stage's single-row reply: I64 [1,1,1], 8 bytes.
        let packed_reply = encode_wire_tokens(&[7]);
        assert_eq!(packed_reply.data.len(), 8, "the collision needs 8 bytes");
        let err = decode_token_with_seq(&packed_reply)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("packed stage is wired to a non-packed"),
            "{err}"
        );

        // A pre-seq peer's legacy token: I32 [1,1,1], 4 bytes.
        let legacy = WireTensor::new(WireDType::I32, [1, 1, 1], 42i32.to_le_bytes().to_vec());
        let err = decode_token_with_seq(&legacy).unwrap_err().to_string();
        assert!(err.contains("predating the seq-tagged token wire"), "{err}");

        // A non-packed lead frame that desynced into the token slot.
        assert!(decode_token_with_seq(&encode_wire_lead(3, None)).is_err());

        // The real thing still decodes.
        assert_eq!(
            decode_token_with_seq(&encode_token_with_seq(42, 9)).unwrap(),
            (42, 9)
        );
    }

    #[test]
    fn lead_frame_roundtrips_on_both_paths() {
        // Stateful: one lane, seq only.
        let f = encode_wire_lead(12345, None);
        assert_eq!(f.dtype, WireDType::I64);
        assert_eq!(f.shape, [1, 1, 1]);
        assert_eq!(decode_wire_lead(&f, false).unwrap(), (12345, None));
        // Static: two lanes, seq THEN position (lane order, not just presence).
        let f = encode_wire_lead(7, Some(11));
        assert_eq!(f.shape, [1, 1, 2]);
        assert_eq!(&f.data[0..8], &7i64.to_le_bytes());
        assert_eq!(&f.data[8..16], &11i64.to_le_bytes());
        assert_eq!(decode_wire_lead(&f, true).unwrap(), (7, Some(11)));
    }

    /// The lead frame's SHAPE is what separates the stateful and static wires.
    /// A stage must reject a lead that doesn't match its own staticness — this
    /// is the check that turns a pre-seq peer, or a mismatched pipeline, into a
    /// hard error instead of a silent mis-bind. Before the position moved into
    /// this frame, a standalone seq frame and a standalone position frame were
    /// byte-identical (I64 `[1,1,1]`, 8 bytes) and nothing could tell them apart.
    #[test]
    fn lead_frame_shape_mismatch_is_rejected_both_ways() {
        // A static peer's lead read by a stateful stage.
        let err = decode_wire_lead(&encode_wire_lead(1, Some(2)), false).unwrap_err();
        assert!(err.to_string().contains("[1,1,1]"), "{err}");
        // A stateful peer's lead — and equally a PRE-SEQ static peer's bare
        // position frame, which has exactly this shape — read by a static stage.
        let err = decode_wire_lead(&encode_wire_lead(1, None), true).unwrap_err();
        assert!(err.to_string().contains("[1,1,2]"), "{err}");
        assert!(err.to_string().contains("predates"), "{err}");
        // Wrong dtype (an F16 hidden where the lead belongs).
        let hid = WireTensor::new(WireDType::F16, [1, 1, 1], vec![0, 0]);
        assert!(decode_wire_lead(&hid, false).is_err());
        // A negative position would wrap the ring math.
        let mut bad = encode_wire_lead(1, Some(0));
        bad.data[8..16].copy_from_slice(&(-3i64).to_le_bytes());
        let err = decode_wire_lead(&bad, true).unwrap_err();
        assert!(err.to_string().contains("negative wire position"), "{err}");
    }

    #[tokio::test]
    async fn hidden_to_token_roundtrip_preserves_seq() {
        // HEAD sends [lead][hidden] downstream; TAIL recvs them, echoes the
        // seq on the token; HEAD reads it back and matches on the same seq.
        let (client, server) = loopback().await;
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let upstream = Arc::new(tokio::sync::Mutex::new(server));

        let stamped = 7u32;
        let hid = WireTensor::new(WireDType::F16, [1, 1, 2], vec![1, 2, 3, 4]);
        send_hidden_frames(&downstream, stamped, None, hid.clone())
            .await
            .unwrap();

        let (lead, hid_t) = recv_hidden_frames(&upstream, false).await.unwrap();
        let hid_t = hid_t.expect("an activation lead is always followed by its hidden");
        let (inbound, pos) = decode_wire_lead(&lead, false).unwrap();
        assert!(pos.is_none());
        assert_eq!(inbound, stamped, "seq must survive the hidden hop");
        assert_eq!(hid_t.dtype, WireDType::F16);
        assert_eq!(hid_t.data, hid.data);

        // TAIL echoes the inbound seq on its token.
        upstream
            .lock()
            .await
            .send(&encode_token_with_seq(55, inbound))
            .await
            .unwrap();
        let tok = recv_token_seq_checked(&downstream, stamped, false)
            .await
            .unwrap();
        assert_eq!(tok, 55);
    }

    /// A bare I8 KV control frame arrives BETWEEN turns and stands alone. It must
    /// come back with no hidden (nothing to wait for — waiting would wedge the
    /// stage for the frame-idle ceiling), and the activation sent after it must
    /// still read back intact: the peek must not consume a frame it did not own.
    #[cfg(feature = "kv_coord")]
    #[tokio::test]
    async fn bare_control_frame_yields_no_hidden_and_does_not_desync_the_next_activation() {
        let (client, server) = loopback().await;
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let upstream = Arc::new(tokio::sync::Mutex::new(server));

        let ctrl = WireTensor::new(WireDType::I8, [1, 1, 1], vec![OPCODE_CAPTURE]);
        downstream.lock().await.send(&ctrl).await.unwrap();

        let (frame, hidden) = recv_hidden_frames(&upstream, false).await.unwrap();
        assert_eq!(frame.dtype, WireDType::I8, "control frame comes back as-is");
        assert!(
            hidden.is_none(),
            "a control frame promises no hidden — waiting for one would wedge the stage"
        );

        // The very next activation must be unaffected.
        let hid = WireTensor::new(WireDType::F16, [1, 1, 2], vec![4, 3, 2, 1]);
        send_hidden_frames(&downstream, 9, None, hid.clone())
            .await
            .unwrap();
        let (lead, hid_t) = recv_hidden_frames(&upstream, false).await.unwrap();
        let hid_t = hid_t.expect("an activation lead is always followed by its hidden");
        assert_eq!(decode_wire_lead(&lead, false).unwrap(), (9, None));
        assert_eq!(
            hid_t.data, hid.data,
            "stream stayed aligned across the control frame"
        );
    }

    #[tokio::test]
    async fn static_shard_wire_is_lead_then_hidden() {
        // Static path: lead carries [seq, position], hidden follows. Distinct
        // values catch a lane swap or a frame transposition.
        let (client, server) = loopback().await;
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let upstream = Arc::new(tokio::sync::Mutex::new(server));

        let hid = WireTensor::new(WireDType::F16, [1, 1, 2], vec![9, 8, 7, 6]);
        send_hidden_frames(&downstream, 3, Some(11), hid.clone())
            .await
            .unwrap();

        let (lead, hid_t) = recv_hidden_frames(&upstream, true).await.unwrap();
        let hid_t = hid_t.expect("an activation lead is always followed by its hidden");
        assert_eq!(
            decode_wire_lead(&lead, true).unwrap(),
            (3, Some(11)),
            "lead carries seq then position"
        );
        assert_eq!(hid_t.data, hid.data, "hidden is the second frame");
    }

    /// The seq is carried as an i64 lane on the lead wire and an i32 lane on
    /// the token wire, so the extreme value has to survive both casts and still
    /// compare equal in the stale check. (This covers the ENCODING, not the
    /// `wrapping_add` at the stamp site — that lives on the engine struct and
    /// is not reachable without a compiled IR.)
    #[tokio::test]
    async fn seq_at_u32_max_survives_both_wires() {
        // A seq at u32::MAX must survive the i32 round-trip on both wires and
        // still match in the stale check (u32::MAX as i32 == -1, back to MAX).
        assert_eq!(
            decode_wire_lead(&encode_wire_lead(u32::MAX, None), false).unwrap(),
            (u32::MAX, None)
        );
        let (client, mut server) = loopback().await;
        server
            .send(&encode_token_with_seq(8, u32::MAX))
            .await
            .unwrap();
        let downstream = Arc::new(tokio::sync::Mutex::new(client));
        let tok = recv_token_seq_checked(&downstream, u32::MAX, false)
            .await
            .unwrap();
        assert_eq!(tok, 8);
    }
}
