//! Gemma 4 multi-stage OpenVINO engine (`--engine gemma4`).
//!
//! Serves per-stage `gemma4_cached_v1.x` OV IR shards (exported by
//! `tools/export_gemma4.py`) across the TCP activation transport, single-stage
//! or pipeline-parallel. Own KV is OV internal state, reset between tasks.
//! `gemma4_cached_v1` trees (pre `sliding_window`) still load but attend past
//! the sliding window on long prompts; `load()` warns about them.
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
//! The tail answers with the picked token. A structured-output task's
//! token-picking steps carry the grammar's token mask inside the hidden frame
//! and get an acknowledged reply (see "structured output on the stage wire"
//! below); every other step's frames are unchanged.
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

use crate::structured::{self, start_grammar};
use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
#[cfg(test)]
use cascadia_grammar::GrammarFactory;
use cascadia_grammar::{CompiledGrammar, GrammarMask};
use cascadia_grammar::{
    MaskRecvCache, MaskSendCache, MASK_MISS_MESSAGE, MASK_MISS_TOKEN, MASK_PROBE,
    MASK_PROBE_OK_TOKEN, MASK_PROBE_REPROBE_AFTER, MASK_PROBE_TIMEOUT, REPEAT,
};
use cascadia_ov_genai_shim::{
    DType as ShimDType, Error as OvError, PluginConfig, Runtime as OvRuntime,
};
use cascadia_transport::{
    ActivationClient, ActivationServer, DType as WireDType, Tensor as WireTensor, MAX_RANK,
};
use cascadia_types::{
    Chunk, FinishReason, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId,
};
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
    /// `gemma4_cached_v1` (no window band) or `gemma4_cached_v1.1` (sliding
    /// layers masked to `sliding_window`). Informational: echoed by the
    /// load-time warning, which keys off `layer_types`/`sliding_window`, not
    /// this string.
    #[serde(default)]
    export_version: Option<String>,
    /// Per-layer attention type of this stage's layers (`sliding_attention` /
    /// `full_attention`), as exported.
    #[serde(default)]
    layer_types: Vec<String>,
    /// Window baked into the stage's `sliding_attention` layers. Absent on
    /// pre-v1.1 exports, which let those layers attend to the whole prefix.
    #[serde(default)]
    sliding_window: Option<u32>,
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

/// True for a stage exported before `gemma4_cached_v1.1`: its sliding layers
/// attend to the whole prefix (correct only while the prompt is shorter than
/// the window, 512–1024 tokens on shipped Gemma 4 checkpoints). Nothing else
/// in this engine reads `export_version` (genai.rs only prefix-matches it to
/// name the engine), so this is the one place a stale tree is called out.
/// `load()` reports it for the stage being loaded only — the adjacency check's
/// read of `stage_{N-1}` would otherwise warn about the same tree twice.
fn sliding_layers_unwindowed(cfg: &StageConfig) -> bool {
    cfg.layer_types.iter().any(|t| t == "sliding_attention") && cfg.sliding_window.is_none()
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
    let vocab = logits_vocab(logits, shape)?;
    Ok(argmax_last_row(logits, vocab))
}

/// The row width of a logits output, checked against the buffer.
fn logits_vocab(logits: &[f32], shape: &[usize]) -> EngineResult<usize> {
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
    Ok(vocab)
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
    /// The grammar matcher of a structured-output task, advanced over every
    /// token in `generated`. `None` for every other task (unmasked path).
    grammar: Option<GrammarMask>,
}

/// One prefill span's end. Prompt segment (`i < seed_split`): the engine's
/// normal folding, `prefill_chunk()`-sized. Seed segment (`i >= seed_split`):
/// T=1, mirroring the donor's token-by-token decode fold — see the resumed-
/// prefill comment in `step_first_body`. `seed_split == tokens.len()` (no
/// resume seed) reduces to the pre-Option-B behavior exactly.
fn prefill_span_end(i: usize, chunk: usize, seed_split: usize) -> usize {
    if i < seed_split {
        i.saturating_add(chunk).min(seed_split)
    } else {
        i + 1
    }
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

/// The task's last chunk. The API reads `prompt_tokens` and `finish_reason` off
/// the chunk with `is_final` set, so leaving them None reported
/// `usage.prompt_tokens: 0` and a hardcoded `"stop"` on every gemma4 response.
///
/// `n_tokens` is THIS chunk's increment, not the running total: the API sums
/// per-chunk counts (falling back to 1 per non-empty chunk), so a cumulative
/// value double-counts every interim token. It is set explicitly so an empty
/// final delta (EOS decoding to "") still counts its token. Mirrors
/// `runtime.rs` / `qwen36.rs`.
///
/// `stop` is a natural stop (EOS, or a structured-output task whose JSON is
/// complete); otherwise the task hit `max_tokens`.
fn final_chunk(task_id: TaskId, token: i64, delta: String, prompt_len: usize, stop: bool) -> Chunk {
    Chunk {
        task_id,
        token_id: token,
        text: delta,
        is_final: true,
        logprobs: None,
        n_tokens: Some(1),
        prompt_tokens: Some(prompt_len as u32),
        completion_tokens: None,
        error: None,
        invalid_request: false,
        token_ids: Vec::new(),
        finish_reason: Some(if stop {
            FinishReason::Stop
        } else {
            FinishReason::Length
        }),
    }
}

/// How a task ends after `generated` tokens: `Stop` on EOS or a grammar finish
/// (also when that lands on the last allowed token), `Length` on the budget,
/// `None` while it continues. Without a grammar `grammar_finished` is false
/// and this is the plain EOS / `max_tokens` check.
fn step_end(
    generated: usize,
    max_tokens: usize,
    is_eos: bool,
    grammar_finished: bool,
) -> Option<FinishReason> {
    if is_eos || grammar_finished {
        Some(FinishReason::Stop)
    } else if generated >= max_tokens {
        Some(FinishReason::Length)
    } else {
        None
    }
}

// -------- structured output (grammar-masked picks) --------

/// Submit-time grammar build: `None` for a task without a grammar; an
/// `InvalidConfig` (400) when the model can't do structured output or the
/// schema doesn't compile. Any topology: on a multi-stage pipeline the mask
/// rides the stage wire.
#[cfg(test)]
fn compile_task_grammar(
    factory: &Result<GrammarFactory, String>,
    task: &GenerationTask,
) -> EngineResult<Option<CompiledGrammar>> {
    structured::compile_task_grammar(factory, task)
}

/// Picks the token from the last row of a logits output. `mask` is passed
/// only when that row picks the next generated token of a structured-output
/// task; then the pick is the best token the grammar allows (the lowest
/// allowed one when every allowed logit is `-inf`/NaN; an error when it allows
/// none). `None` is the plain argmax. The engine calls
/// [`pick_tagged`], which also says how a failure is logged.
#[cfg(test)]
fn pick_next_token(
    logits: &[f32],
    shape: &[usize],
    mask: Option<&mut GrammarMask>,
) -> EngineResult<i32> {
    pick_tagged(logits, shape, mask).map_err(|(e, _)| e)
}

/// [`pick_next_token`] with how a failure is logged: only a mask that allows
/// nothing in the row is a schema dead end; a malformed logits output is the
/// stage's own failure.
fn pick_tagged(
    logits: &[f32],
    shape: &[usize],
    mask: Option<&mut GrammarMask>,
) -> Result<i32, (EngineError, Option<GrammarFailure>)> {
    let Some(mask) = mask else {
        return argmax_logits(logits, shape).map_err(|e| (e, None));
    };
    let vocab = logits_vocab(logits, shape).map_err(|e| (e, None))?;
    mask.pick_greedy(&logits[logits.len() - vocab..])
        .map(|t| t as i32)
        .map_err(|e| (e, Some(GrammarFailure::DeadEnd)))
}

/// Runs the prefill spans of a `len`-token prefill in order (see
/// [`prefill_span_end`]) and returns the last span's token, the first
/// generated one. `span` gets the grammar mask only for that last span: the
/// earlier spans' picks are discarded.
fn fold_prefill<F>(
    len: usize,
    chunk: usize,
    seed_split: usize,
    mut mask: Option<&mut GrammarMask>,
    mut span: F,
) -> EngineResult<i32>
where
    F: FnMut(std::ops::Range<usize>, Option<&mut GrammarMask>) -> EngineResult<i32>,
{
    let mut i = 0usize;
    loop {
        let end = prefill_span_end(i, chunk, seed_split);
        let last = end >= len;
        let m = step_mask(mask.as_deref_mut(), true, end, len);
        let token = span(i..end, m)?;
        if last {
            return Ok(token);
        }
        i = end;
    }
}

/// The matcher for one span of a step, or `None`: a decode step's row is
/// always the next generated token; a prefill span only when it ends the
/// prompt (the earlier spans' picks are discarded).
fn step_mask(
    grammar: Option<&mut GrammarMask>,
    prefill: bool,
    span_end: usize,
    len: usize,
) -> Option<&mut GrammarMask> {
    if !prefill || span_end >= len {
        grammar
    } else {
        None
    }
}

/// Advances the active task's matcher past `token`; `Ok(true)` when the JSON
/// is complete. A token the grammar rejects ends the task: `active` is
/// cleared and the error names it.
fn advance_grammar(active: &mut Option<ActiveTask>, token: i32) -> EngineResult<bool> {
    let Some(a) = active.as_mut() else {
        return Ok(false);
    };
    let Some(mask) = a.grammar.as_mut() else {
        return Ok(false);
    };
    match mask.accept(token as u32) {
        Ok(()) => Ok(mask.finished()),
        Err(e) => {
            let id = a.task.task_id.clone();
            *active = None;
            Err(e.for_task(id))
        }
    }
}

/// A structured-output task failure: it ends that task, says nothing about
/// the stage's health, and so stays out of the step-failure streak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrammarFailure {
    /// The grammar allows no token of the row.
    DeadEnd,
    /// The picked token is not one the grammar allows.
    Rejected,
    /// A later stage couldn't apply the mask to this step.
    Nack,
    /// The chain didn't acknowledge a masked step.
    Upgrade,
    /// Refused at once: the chain didn't acknowledge an earlier masked step.
    Latched,
}

impl GrammarFailure {
    fn describe(self) -> &'static str {
        match self {
            GrammarFailure::DeadEnd => "the schema allows no token at this point",
            GrammarFailure::Rejected => "the picked token is not one the schema allows",
            GrammarFailure::Nack => "a later stage could not apply the mask",
            GrammarFailure::Upgrade => "a later stage did not acknowledge the mask",
            GrammarFailure::Latched => "structured output is off on this pipeline",
        }
    }
}

/// Logs a failed step: a structured-output failure with its own message
/// through its own limiter (clients choose the schemas, so they can make
/// these as often as they like), anything else through the rate-limited
/// step-failure WARN.
fn report_step_error(
    limiter: &mut StepWarnLimiter,
    grammar_limiter: &mut StepWarnLimiter,
    grammar: Option<GrammarFailure>,
    e: &EngineError,
    now: std::time::Instant,
) {
    if let Some(what) = grammar {
        let what = what.describe();
        match grammar_limiter.on_failure(now) {
            Some(StepWarn::First) => {
                warn!(error = %e, "gemma4: structured-output task failed: {what}")
            }
            Some(StepWarn::StillFailing { suppressed }) => {
                warn!(error = %e, suppressed, "gemma4: structured-output task failed: {what}")
            }
            None => {}
        }
        return;
    }
    match limiter.on_failure(now) {
        Some(StepWarn::First) => warn!(error = %e, "gemma4 step failed"),
        Some(StepWarn::StillFailing { suppressed }) => {
            warn!(error = %e, suppressed, "gemma4 step still failing")
        }
        None => {}
    }
}

// -------- structured output on the stage wire --------
//
// On a masked step (the step whose last row picks the next token of a
// structured-output task, multi-stage only) the head sends the grammar's
// token mask to the tail inside the hidden frame, and the tail answers with
// an acknowledged two-lane token reply. Every other step's frames are the
// plain ones above, byte for byte.
//
// Masked hidden frame: I32 `[1, n+R, H]`. Rows `0..n` are the f32 bits of
// the `[1, n, H]` hidden; the `R = ceil((W+1)/H)` trailing rows hold the `W`
// mask words (bit `b` of word `i` allows token `32*i + b`), zero padding,
// and `W` itself in the very last element (where a receiver finds it before
// it knows `n`). An older stage rejects the I32 dtype only after it has read
// every frame of the step (the cross-KV header count is unchanged), so the
// link stays aligned; it sends no reply and the head's bounded wait expires.
//
// Replies: plain steps `[1,1,1]` `[token]` (unchanged); masked steps
// `[1,1,2]` `[token, MASK_ACK]`, `[-1, MASK_ACK]` when the stage can't
// serve the masked step (an acknowledged NACK: the chain masks, this step
// failed), or `[-1, MASK_NOT_ACKED]` from a middle stage whose downstream
// did not acknowledge (the chain can't mask). gemma4 replies carry no
// sequence number, so a plain wait discards any two-lane reply as a late
// answer to an earlier masked step, and a masked step drains such replies
// before it sends.
//
// Repeated masks: a masked step whose mask equals the last one sent on the
// link carries the marker instead, I32 `[1, n, H+1]`: each hidden row's f32
// bits followed by `REPEAT` (no mask words, one extra word per row). Its last
// element is `REPEAT` where a full masked frame has `W`, which tells the two
// apart before any width or row check. The
// receiver uses the last mask it decoded (`MaskSendCache` / `MaskRecvCache`).
// A receiver with nothing to repeat runs nothing and answers
// `[MASK_MISS_TOKEN, MASK_ACK]`; middles pass that up unchanged and the head
// fails the task without latching.
//
// Mask-support probe: before a structured-output task's first masked step on
// a pipeline with no evidence that every later stage masks, the head sends a
// step at position 0 whose hidden is I32 `[1, 1, 2]` = `[0, MASK_PROBE]` (one
// zero row; `MASK_PROBE` where a masked frame has `W`), then the decode
// step's cross-KV header and one F32 `[1,1,1]` zero frame per tag. An older
// stage reads every frame (the header's count is the one it expects) and
// rejects the I32 hidden, so it stays aligned and silent; the head stops
// waiting after `MASK_PROBE_TIMEOUT` and latches. A newer middle forwards its
// own probe and passes the answer up; the tail answers
// `[MASK_PROBE_OK_TOKEN, MASK_ACK]` at once. No stage runs a layer or touches
// its KV, position, cross-KV inputs or mask caches for it.

/// Second lane of a masked step's reply from a stage that acknowledged it.
const MASK_ACK: i32 = 0x4d41_534b;
/// Second lane of a middle stage's reply when its downstream didn't acknowledge the mask.
const MASK_NOT_ACKED: i32 = 0x4e4f_4d4b;
/// First lane of a masked step's reply that carries no token.
const MASK_NACK_TOKEN: i32 = -1;
/// How long a head that saw an unacknowledged masked step refuses structured
/// output before it probes the chain again.
const MASK_LATCH_REPROBE: Duration = Duration::from_secs(600);
/// What a structured-output task on an unacknowledging chain fails with.
const UPGRADE_EVERY_STAGE: &str = cascadia_grammar::STAGE_UPGRADE_MESSAGE;

/// Encodes a step's hidden frame. `mask = None` is the plain F32 frame (the
/// encoding every step used before masks); `Some(words)` the masked I32
/// frame described above.
fn encode_hidden_frame(
    hidden: &[f32],
    shape: [usize; 3],
    mask: Option<&[u32]>,
) -> EngineResult<WireTensor> {
    let Some(words) = mask else {
        let mut wire_shape = [1u32; MAX_RANK];
        for (i, d) in shape.iter().enumerate().take(MAX_RANK) {
            wire_shape[i] = *d as u32;
        }
        return Ok(WireTensor::new(
            WireDType::F32,
            wire_shape,
            f32_to_bytes(hidden),
        ));
    };
    let [batch, n, h] = shape;
    if batch != 1 || n == 0 || h == 0 || hidden.len() != n * h || words.is_empty() {
        return Err(EngineError::Backend(format!(
            "can't send a structured-output mask with a hidden of shape {shape:?} \
             ({} values) and {} mask words",
            hidden.len(),
            words.len()
        )));
    }
    let w = words.len();
    let r = (w + 1).div_ceil(h);
    let bytes = (n + r) * h * 4;
    if bytes > cascadia_transport::MAX_TENSOR_BYTES || u32::try_from(n + r).is_err() {
        return Err(EngineError::Backend(format!(
            "a structured-output step of {n} positions is too large for one frame \
             ({bytes} bytes)"
        )));
    }
    let mut data = Vec::with_capacity(bytes);
    data.extend_from_slice(&f32_to_bytes(hidden));
    let mut tail = vec![0u32; r * h];
    tail[..w].copy_from_slice(words);
    tail[r * h - 1] = w as u32;
    for x in &tail {
        data.extend_from_slice(&x.to_le_bytes());
    }
    Ok(WireTensor::new(
        WireDType::I32,
        [1, (n + r) as u32, h as u32],
        data,
    ))
}

/// Turns a full masked hidden frame (see [`encode_hidden_frame`]) into the
/// repeat marker in place: I32 `[1, n, H+1]`, each hidden row followed by
/// [`REPEAT`]. The mask rows are dropped.
fn into_repeat_frame(hid: &mut WireTensor) {
    let h = hid.shape[2] as usize;
    let rows = hid.shape[1] as usize;
    let len = hid.data.len();
    let w = u32::from_le_bytes(hid.data[len - 4..].try_into().unwrap()) as usize;
    let n = rows - (w + 1).div_ceil(h);
    let mut data = Vec::with_capacity(n * (h + 1) * 4);
    for row in hid.data[..n * h * 4].chunks_exact(h * 4) {
        data.extend_from_slice(row);
        data.extend_from_slice(&REPEAT.to_le_bytes());
    }
    hid.data = data;
    hid.shape[1] = n as u32;
    hid.shape[2] = (h + 1) as u32;
}

/// A received hidden frame.
#[derive(Debug, PartialEq)]
enum HiddenIn {
    Plain {
        floats: Vec<f32>,
        shape: [usize; 3],
    },
    Masked {
        floats: Vec<f32>,
        shape: [usize; 3],
        words: Vec<u32>,
    },
    /// A masked frame this stage can't read; it answers with an acknowledged NACK.
    BadMask(String),
    /// The repeat marker, as decoded: the link's last mask applies (see
    /// [`resolve_hidden`]).
    Repeat {
        floats: Vec<f32>,
        shape: [usize; 3],
    },
    /// A repeat marker with no mask to repeat on this link: the stage runs
    /// nothing and answers [`StageReply::MaskMiss`].
    MaskMiss,
    /// A mask-support probe: the stage runs nothing and changes no state; the
    /// tail answers [`StageReply::ProbeOk`], a middle relays it.
    Probe,
}

/// Resolves a decoded hidden frame against the link's receive cache, once
/// every frame of the step is read: a marker becomes the last mask decoded
/// (or [`HiddenIn::MaskMiss`]), a full mask is committed, and a bad masked
/// frame clears the cache.
fn resolve_hidden(h: HiddenIn, cache: &mut MaskRecvCache) -> HiddenIn {
    match h {
        HiddenIn::Repeat { floats, shape } => {
            match cache.resolve(cascadia_grammar::WireMask::Repeat) {
                Ok(words) => HiddenIn::Masked {
                    floats,
                    shape,
                    words,
                },
                Err(_) => HiddenIn::MaskMiss,
            }
        }
        HiddenIn::Masked {
            floats,
            shape,
            words,
        } => {
            cache.commit(&words);
            HiddenIn::Masked {
                floats,
                shape,
                words,
            }
        }
        HiddenIn::BadMask(why) => {
            cache.clear();
            HiddenIn::BadMask(why)
        }
        other => other,
    }
}

/// Decodes a hidden frame (F32/F16 plain, I32 masked).
fn decode_hidden_frame(t: &WireTensor) -> EngineResult<HiddenIn> {
    let shape = [
        t.shape[0] as usize,
        t.shape[1] as usize,
        t.shape[2] as usize,
    ];
    match t.dtype {
        WireDType::F32 => Ok(HiddenIn::Plain {
            floats: t
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            shape,
        }),
        WireDType::F16 => Ok(HiddenIn::Plain {
            floats: f16_bytes_to_f32(&t.data),
            shape,
        }),
        WireDType::I32 => Ok(decode_masked_hidden(t, shape)),
        other => Err(EngineError::Backend(format!(
            "unexpected upstream hidden dtype {other:?}"
        ))),
    }
}

/// The I32 masked hidden frame (see the layout above).
fn decode_masked_hidden(t: &WireTensor, shape: [usize; 3]) -> HiddenIn {
    let [batch, rows, h] = shape;
    let elems: Vec<u32> = t
        .data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if batch != 1 || h == 0 || elems.len() != rows * h {
        return HiddenIn::BadMask(format!("masked hidden frame of shape {shape:?}"));
    }
    // The probe, then the marker, are checked first: their last element is
    // no word count. A probe is one row of zeros ending in MASK_PROBE.
    if elems.last() == Some(&MASK_PROBE) {
        if rows != 1 || h < 2 || elems[..h - 1].iter().any(|&x| x != 0) {
            return HiddenIn::BadMask(format!("mask-support probe frame of shape {shape:?}"));
        }
        return HiddenIn::Probe;
    }
    // `[1, n, H+1]`, every row ending in REPEAT.
    if elems.last() == Some(&REPEAT) {
        let width = h;
        if rows == 0 || width < 2 || elems.chunks_exact(width).any(|r| r[width - 1] != REPEAT) {
            return HiddenIn::BadMask(format!("repeat-mask hidden frame of shape {shape:?}"));
        }
        return HiddenIn::Repeat {
            floats: elems
                .chunks_exact(width)
                .flat_map(|r| r[..width - 1].iter().map(|&b| f32::from_bits(b)))
                .collect(),
            shape: [1, rows, width - 1],
        };
    }
    let w = elems.last().copied().unwrap_or(0) as usize;
    let r = (w + 1).div_ceil(h);
    if w == 0 || r >= rows {
        return HiddenIn::BadMask(format!(
            "masked hidden frame of shape {shape:?} declares {w} mask words"
        ));
    }
    let n = rows - r;
    let tail = &elems[n * h..];
    if tail[w..tail.len() - 1].iter().any(|&x| x != 0) {
        return HiddenIn::BadMask("masked hidden frame has a non-zero padding".into());
    }
    HiddenIn::Masked {
        floats: elems[..n * h].iter().map(|&b| f32::from_bits(b)).collect(),
        shape: [1, n, h],
        words: tail[..w].to_vec(),
    }
}

/// The head's mask for the next token over `width` tokens (the tokenizer
/// vocab). Fails when the grammar allows nothing.
fn mask_words(mask: &mut GrammarMask, width: usize) -> EngineResult<Vec<u32>> {
    if width == 0 {
        return Err(EngineError::Backend(
            "grammar: the tokenizer vocabulary is empty".into(),
        ));
    }
    crate::structured::mask_words_over(mask, width)
}

/// The tail's pick on a masked step: the best token of the last row the mask
/// words allow (tokens past the words never), the lowest allowed one when
/// every allowed logit is `-inf`/NaN. Words wider than the row allow nothing
/// past it. `Err` when they allow nothing in the row.
fn tail_masked_pick(logits: &[f32], shape: &[usize], words: &[u32]) -> Result<i32, String> {
    let vocab = logits_vocab(logits, shape).map_err(|e| e.to_string())?;
    // A mask wider than the row: tokens past the row are not allowed.
    cascadia_grammar::argmax_allowed(&logits[logits.len() - vocab..], words)
        .map(|t| t as i32)
        .ok_or_else(|| "the mask allows no token of this stage's logits row".into())
}

/// A token reply from downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageReply {
    /// `[1,1,1]` (any non-two-lane reply): a plain step's token.
    Plain(i32),
    /// `[token, MASK_ACK]`.
    Token(i32),
    /// `[-1, MASK_ACK]`.
    Nack,
    /// `[-1, MASK_NOT_ACKED]`.
    NotAcked,
    /// `[MASK_MISS_TOKEN, MASK_ACK]`: a stage got a repeat marker with no
    /// mask to repeat.
    MaskMiss,
    /// `[MASK_PROBE_OK_TOKEN, MASK_ACK]`: every stage below can mask (the
    /// answer to a mask-support probe).
    ProbeOk,
}

/// The plain token reply, `[1,1,1]` I32.
fn encode_token_reply(token: i32) -> WireTensor {
    WireTensor::new(WireDType::I32, [1, 1, 1], token.to_le_bytes().to_vec())
}

/// A masked step's two-lane reply (`Plain` encodes as the plain reply).
fn encode_masked_reply(reply: StageReply) -> WireTensor {
    let (token, lane) = match reply {
        StageReply::Plain(t) => return encode_token_reply(t),
        StageReply::Token(t) => (t, MASK_ACK),
        StageReply::Nack => (MASK_NACK_TOKEN, MASK_ACK),
        StageReply::NotAcked => (MASK_NACK_TOKEN, MASK_NOT_ACKED),
        StageReply::MaskMiss => (MASK_MISS_TOKEN, MASK_ACK),
        StageReply::ProbeOk => (MASK_PROBE_OK_TOKEN, MASK_ACK),
    };
    let mut data = token.to_le_bytes().to_vec();
    data.extend_from_slice(&lane.to_le_bytes());
    WireTensor::new(WireDType::I32, [1, 1, 2], data)
}

/// Decodes a token reply. Anything that is not a two-lane I32 `[1,1,2]` is
/// read the way plain replies always were (first four bytes).
fn decode_stage_reply(t: &WireTensor) -> EngineResult<StageReply> {
    let lane = |i: usize| i32::from_le_bytes(t.data[i * 4..i * 4 + 4].try_into().unwrap());
    if t.dtype == WireDType::I32 && t.shape == [1, 1, 2] && t.data.len() == 8 {
        return match (lane(0), lane(1)) {
            (MASK_NACK_TOKEN, MASK_ACK) => Ok(StageReply::Nack),
            (MASK_NACK_TOKEN, MASK_NOT_ACKED) => Ok(StageReply::NotAcked),
            (MASK_MISS_TOKEN, MASK_ACK) => Ok(StageReply::MaskMiss),
            (MASK_PROBE_OK_TOKEN, MASK_ACK) => Ok(StageReply::ProbeOk),
            (token, MASK_ACK) if token >= 0 => Ok(StageReply::Token(token)),
            (a, b) => Err(EngineError::Backend(format!(
                "downstream sent an unrecognised two-lane token reply [{a}, {b}]"
            ))),
        };
    }
    if t.data.len() < 4 {
        return Err(EngineError::Backend(format!(
            "downstream sent {}-byte token tensor; need at least 4",
            t.data.len()
        )));
    }
    Ok(StageReply::Plain(lane(0)))
}

/// Sends one step: position, hidden, cross-KV header, cross-KV frames.
async fn send_step(
    g: &mut ActivationClient,
    pos: &WireTensor,
    hid: &WireTensor,
    header: &WireTensor,
    cross: &[WireTensor],
) -> Result<(), cascadia_transport::TransportError> {
    g.send(pos).await?;
    g.send(hid).await?;
    g.send(header).await?;
    for f in cross {
        g.send(f).await?;
    }
    Ok(())
}

/// Sends one step with the link's send cache: a masked step whose mask
/// equals the last one sent on the link goes as the repeat marker (its
/// hidden frame rewritten in place). The cache is checked and updated under
/// the caller's lock on the link, around the write; a write error clears it.
async fn send_step_cached(
    g: &mut ActivationClient,
    frames: &mut StepFrames,
    cache: &mut MaskSendCache,
) -> Result<(), cascadia_transport::TransportError> {
    // The links of this engine never re-dial, so one connection generation.
    const GENERATION: u64 = 0;
    if let Some(words) = &frames.mask {
        if cache.repeats(GENERATION, words) {
            into_repeat_frame(&mut frames.hid);
        }
    }
    let sent = send_step(g, &frames.pos, &frames.hid, &frames.header, &frames.cross).await;
    match (&sent, &frames.mask) {
        (Err(_), _) => cache.clear(),
        (Ok(()), Some(words)) => cache.record(GENERATION, words),
        (Ok(()), None) => {}
    }
    sent
}

/// [`wait_plain_reply`] on a link with a send cache: a failed wait clears
/// it, so the next masked step goes full.
async fn plain_reply_settled(
    g: &mut ActivationClient,
    prefill: bool,
    cache: &mut MaskSendCache,
) -> EngineResult<(i32, bool)> {
    wait_plain_reply(g, prefill)
        .await
        .inspect_err(|_| cache.clear())
}

/// Reads the rest of a step after its position frame: the hidden frame, the
/// cross-KV header and its frames (`consumes` = this stage's external-KV
/// input count; a mismatch fails before the frames are read, as always).
async fn recv_step_frames(
    g: &mut ActivationServer,
    consumes: usize,
) -> EngineResult<(WireTensor, Vec<i64>, Vec<WireTensor>)> {
    let be = |e: cascadia_transport::TransportError| EngineError::Backend(e.to_string());
    // Mid-step replies: deadlined so a half-sent step can't wedge the stage.
    let hid = g.recv_reply().await.map_err(be)?.0;
    let header = g.recv_reply().await.map_err(be)?.0;
    let tags = decode_cross_kv_header(&header)?;
    let n = tags.len();
    // Strict frame-count guard AND the distributed-deploy runtime backstop
    // for the load-time adjacency guard (skipped when the sibling
    // stage_config isn't on local disk): the immediate upstream must send
    // EXACTLY this stage's external_kv sources.
    if n != consumes {
        return Err(EngineError::Backend(format!(
            "upstream sent {n} cross-KV frame(s) but this stage consumes {consumes} external_kv \
             input(s) — a cross-stage KV source/consumer mismatch (non-adjacent KV sharing \
             across more than one pipeline hop, or a mis-exported pipeline). The immediate \
             upstream must produce exactly this stage's external_kv sources."
        )));
    }
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        frames.push(g.recv_reply().await.map_err(be)?.0);
    }
    Ok((hid, tags, frames))
}

/// A plain step's token wait (the socket-poisoning reply waits of every
/// plain step). A two-lane reply is a late answer to an earlier masked step:
/// discarded; the bool says whether one of them was acknowledged.
async fn wait_plain_reply(g: &mut ActivationClient, prefill: bool) -> EngineResult<(i32, bool)> {
    let mut late_ack = false;
    loop {
        // MID-TASK reply — deadlined; see `recv_tensor_reply` (Item 5).
        // A prefill reply waits on every remaining stage's whole-prompt
        // compute — widened budget (see `recv_tensor_reply_prefill`),
        // CEILINGED here: the transport budget multiplies the operator's
        // frame-transfer knob (recv_timeout × 10 — the rig's 600 s made it
        // 6000 s), and the waiting head holds its admission slot the whole
        // time, so one dead downstream turned the node into a
        // refuse-everything wall for 100 minutes (2026-08-17 incident).
        // ov-runtime ceilings the same coupling at
        // TOKEN_RECV_DEADLINE_CEILING. Measured legit worst here: 51 s (4.2k
        // tokens, 2-stage); the ceiling leaves ~6x headroom for deeper
        // pipelines / longer prompts.
        let res = if prefill {
            recv_prefill_reply_ceilinged(g).await
        } else {
            g.recv_reply().await
        };
        let (tensor, _) = res.map_err(|e| EngineError::Backend(e.to_string()))?;
        match decode_stage_reply(&tensor)? {
            StageReply::Plain(token) => return Ok((token, late_ack)),
            reply => {
                debug!(
                    ?reply,
                    "discarding a late reply to an earlier structured-output step"
                );
                // A late mask miss proves nothing about the chain.
                late_ack |= matches!(
                    reply,
                    StageReply::Token(_) | StageReply::Nack | StageReply::ProbeOk
                );
            }
        }
    }
}

/// Reads and discards replies that already arrived (late answers to an
/// earlier masked step) before a masked step is sent; true when one of them
/// was acknowledged. It doesn't wait for one: one yield lets the I/O driver
/// record bytes already on the socket, then a reply is read only while the
/// link shows a frame starting (a peek, which consumes nothing). A
/// zero-time read instead would cost a timer tick (about a millisecond) on
/// every masked step, and miss bytes the driver hasn't recorded yet.
async fn drain_late_replies(g: &mut ActivationClient) -> EngineResult<bool> {
    let mut late_ack = false;
    tokio::task::yield_now().await;
    loop {
        match futures::FutureExt::now_or_never(g.wait_readable()) {
            None => return Ok(late_ack),
            Some(Err(e)) => return Err(EngineError::Backend(e.to_string())),
            Some(Ok(())) => {}
        }
        // A frame has started: its bytes are there or on their way.
        match g.recv_token(LATE_REPLY_FRAME_START).await {
            Err(e) => return Err(EngineError::Backend(e.to_string())),
            Ok((tensor, _)) => {
                let reply = decode_stage_reply(&tensor);
                debug!(
                    ?reply,
                    "discarding a late reply to an earlier structured-output step"
                );
                // A late mask miss proves nothing about the chain.
                late_ack |= matches!(
                    reply,
                    Ok(StageReply::Token(_) | StageReply::Nack | StageReply::ProbeOk)
                );
            }
        }
    }
}

/// Frame-start bound of a late reply whose first byte is already readable.
const LATE_REPLY_FRAME_START: Duration = Duration::from_secs(1);

/// How a masked step's wait ended.
#[derive(Debug)]
enum MaskedWait {
    Token(i32),
    /// An acknowledged NACK: the chain masks, but this step failed.
    Nack,
    /// No acknowledgement (no reply in time, a plain reply, or a middle's
    /// not-acknowledged reply): the chain can't mask.
    Unacked(String),
    /// A stage below got a repeat marker with no mask to repeat.
    MaskMiss,
    /// The answer to a mask-support probe: every stage below masks.
    ProbeOk,
    /// The link failed.
    Failed(EngineError),
}

/// Extra wait per stage between this one and the last middle, on a masked
/// decode step: covers that middle's compute and its reply hop.
const MASK_HOP_MARGIN: Duration = Duration::from_secs(5);
/// The same on a masked prefill step (a middle's whole-prompt slice; the
/// measured worst for a whole 2-stage prefill is 51 s).
const MASK_HOP_MARGIN_PREFILL: Duration = Duration::from_secs(60);

/// The bound of a masked step's wait at a stage with `stages_below` stages
/// after it (the head of a 2-stage pipeline: 1). The stage just before the
/// tail waits the plain step's budget; each stage before it waits one margin
/// more than the next, so a middle in front of an old tail times out and its
/// not-acknowledged reply reaches every stage above while it still waits
/// (leaving no late reply on the link).
fn masked_wait_budget(prefill: bool, stages_below: u32) -> Duration {
    let base = cascadia_transport::recv_timeout();
    let hops = stages_below.saturating_sub(1);
    if prefill {
        base.saturating_mul(cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR)
            .min(G_PREFILL_REPLY_CEILING)
            .saturating_add(MASK_HOP_MARGIN_PREFILL.saturating_mul(hops))
    } else {
        base.saturating_add(MASK_HOP_MARGIN.saturating_mul(hops))
    }
}

/// A masked step's wait. Bounded and never closes the socket: a stage that
/// doesn't answer leaves the link usable for the next plain step.
async fn wait_masked_reply(g: &mut ActivationClient, budget: Duration) -> MaskedWait {
    let deadline = std::time::Instant::now() + budget;
    let reply = loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let tensor = match g.recv_token(remaining).await {
            Ok((t, _)) => t,
            Err(cascadia_transport::TransportError::FrameStartTimeout(_)) => {
                return MaskedWait::Unacked(format!(
                    "no reply within {budget:?} to a structured-output step"
                ))
            }
            Err(e) => return MaskedWait::Failed(EngineError::Backend(e.to_string())),
        };
        match decode_stage_reply(&tensor) {
            // A probe answer never answers a step: a late one, skipped.
            Ok(StageReply::ProbeOk) => {
                debug!("discarding a late answer to a mask-support probe");
            }
            reply => break reply,
        }
    };
    match reply {
        Ok(StageReply::ProbeOk) => MaskedWait::ProbeOk,
        Ok(StageReply::Token(t)) => MaskedWait::Token(t),
        Ok(StageReply::Nack) => MaskedWait::Nack,
        Ok(StageReply::MaskMiss) => MaskedWait::MaskMiss,
        Ok(StageReply::NotAcked) => MaskedWait::Unacked(
            "a later stage relayed a structured-output step to a stage that did not \
             acknowledge it"
                .into(),
        ),
        Ok(StageReply::Plain(_)) => MaskedWait::Unacked(
            "the reply to a structured-output step did not acknowledge the mask".into(),
        ),
        Err(e) => MaskedWait::Failed(e),
    }
}

/// What a middle stage sends upstream after its own masked wait, and the
/// error that still fails its step.
fn middle_reply(w: MaskedWait) -> (StageReply, Option<EngineError>) {
    match w {
        MaskedWait::Token(t) => (StageReply::Token(t), None),
        MaskedWait::Nack => (StageReply::Nack, None),
        // Passed up unchanged: the head fails the task without latching.
        MaskedWait::MaskMiss => (StageReply::MaskMiss, None),
        // Never the answer to a masked step (see `wait_masked_reply`).
        MaskedWait::ProbeOk => (StageReply::Nack, None),
        MaskedWait::Unacked(why) => {
            warn!(%why, "gemma4: downstream did not acknowledge a structured-output step");
            (StageReply::NotAcked, None)
        }
        // This stage's link failed, not a stage's version: an acknowledged
        // NACK fails the head's task without disabling structured output.
        MaskedWait::Failed(e) => (StageReply::Nack, Some(e)),
    }
}

/// One step's frames, in send order (see [`send_step`]).
struct StepFrames {
    pos: WireTensor,
    hid: WireTensor,
    header: WireTensor,
    cross: Vec<WireTensor>,
    /// The mask `hid` carries, on a masked step.
    mask: Option<Vec<u32>>,
}

/// A masked step on the downstream link, as the head and every middle run
/// it: discard late replies, send the step (a repeated mask as the marker,
/// see [`send_step_cached`]), wait for its reply. The bool says a discarded
/// late reply was acknowledged. Any outcome but a token clears `cache`.
async fn masked_round(
    g: &mut ActivationClient,
    frames: &mut StepFrames,
    budget: Duration,
    cache: &mut MaskSendCache,
) -> (MaskedWait, bool) {
    let late_ack = match drain_late_replies(g).await {
        Ok(late_ack) => late_ack,
        Err(e) => {
            cache.clear();
            return (MaskedWait::Failed(e), false);
        }
    };
    if let Err(e) = send_step_cached(g, frames, cache).await {
        return (
            MaskedWait::Failed(EngineError::Backend(e.to_string())),
            late_ack,
        );
    }
    let wait = wait_masked_reply(g, budget).await;
    if !matches!(wait, MaskedWait::Token(_)) {
        cache.clear();
    }
    (wait, late_ack)
}

/// A middle stage's masked step after it has read the step from upstream:
/// `prepared` is its relayed step (or why it couldn't relay it). Returns the
/// reply it owes upstream (always one, so the head never waits out its
/// deadline on a failed middle), the error that still fails its step, and
/// whether the round discarded an acknowledged late reply from below (see
/// [`OwedLateAck::after_masked_reply`]).
async fn middle_masked_step(
    down: &mut ActivationClient,
    prepared: EngineResult<StepFrames>,
    budget: Duration,
    cache: &mut MaskSendCache,
) -> (StageReply, Option<EngineError>, bool) {
    match prepared {
        Err(e) => (StageReply::Nack, Some(e), false),
        Ok(mut frames) => {
            let (wait, late_ack) = masked_round(down, &mut frames, budget, cache).await;
            let (reply, err) = middle_reply(wait);
            (reply, err, late_ack)
        }
    }
}

/// The tail's masked step after its relay: the reply it owes upstream and
/// the error that still fails its step.
fn tail_masked_reply(
    relayed: EngineResult<(Vec<f32>, Vec<usize>)>,
    words: &[u32],
) -> (StageReply, Option<EngineError>) {
    let (out, out_shape) = match relayed {
        Ok(r) => r,
        Err(e) => return (StageReply::Nack, Some(e)),
    };
    match tail_masked_pick(&out, &out_shape, words) {
        Ok(token) => (StageReply::Token(token), None),
        Err(why) => {
            warn!(%why, "gemma4: can't apply a structured-output mask; answering NACK");
            (StageReply::Nack, None)
        }
    }
}

/// Middles: what this stage owes upstream after answering a masked step
/// without an acknowledgement. Once a stage below acknowledges anything (a
/// slow but capable tail), the head's latch can clear: this stage passes an
/// acknowledged NACK up before its next plain reply or control ack, which
/// the head skips as a late reply and reads as that evidence. Never before a
/// masked reply: those carry no sequence number, so the head would take it
/// for the current step's answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum OwedLateAck {
    #[default]
    Nothing,
    /// Answered a masked step unacknowledged; nothing below has acked since.
    Owed,
    /// And a stage below has acknowledged since: pass it up.
    Proven,
}

impl OwedLateAck {
    /// This stage answered a masked step upstream with `reply`.
    fn on_masked_reply_sent(&mut self, reply: StageReply) {
        *self = if matches!(reply, StageReply::NotAcked) {
            Self::Owed
        } else {
            // An acknowledged reply already told the head.
            Self::Nothing
        };
    }

    /// A stage below acknowledged a reply this stage discarded as late.
    fn on_ack_from_below(&mut self) {
        if *self == Self::Owed {
            *self = Self::Proven;
        }
    }

    /// This stage answered a masked step upstream with `reply`, after its
    /// masked round discarded a late reply from below (`late_ack`: that reply
    /// was acknowledged). The step's own answer comes first, then the
    /// evidence, so an unacknowledged answer still passes it up.
    fn after_masked_reply(&mut self, reply: StageReply, late_ack: bool) {
        self.on_masked_reply_sent(reply);
        if late_ack {
            self.on_ack_from_below();
        }
    }
}

/// The frames that answer upstream with `reply` (a plain token reply or a
/// control ack): an acknowledged NACK first when a late acknowledgement is
/// owed and proven (see [`OwedLateAck`]).
fn upstream_plain_frames(owed: &mut OwedLateAck, reply: WireTensor) -> Vec<WireTensor> {
    let mut frames = Vec::with_capacity(2);
    if *owed == OwedLateAck::Proven {
        *owed = OwedLateAck::Nothing;
        frames.push(encode_masked_reply(StageReply::Nack));
    }
    frames.push(reply);
    frames
}

/// "This pipeline can't mask": set by an unacknowledged masked step, it makes
/// structured-output tasks fail at once (no frames sent) until a late
/// acknowledged reply clears it or [`MASK_LATCH_REPROBE`] has passed. Lives on
/// the engine, so it outlives any re-dial of the link.
#[derive(Debug, Default)]
struct MaskLatch {
    since: Option<std::time::Instant>,
    /// How long the latch set at `since` refuses.
    reprobe_after: Duration,
    /// A later stage has acknowledged a masked step or a mask-support probe
    /// since the last latch: no probe is needed.
    evidence: bool,
}

impl MaskLatch {
    /// A masked step went unacknowledged.
    fn trip(&mut self, now: std::time::Instant) {
        self.trip_for(now, MASK_LATCH_REPROBE);
    }

    /// A mask-support probe went unanswered: a shorter latch, since a probe
    /// costs a request only a short wait.
    fn trip_probe(&mut self, now: std::time::Instant) {
        self.trip_for(now, MASK_PROBE_REPROBE_AFTER);
    }

    fn trip_for(&mut self, now: std::time::Instant, reprobe_after: Duration) {
        if self.since.is_none() {
            warn!(
                reprobe_after = ?reprobe_after,
                "gemma4: structured output disabled on this pipeline: {UPGRADE_EVERY_STAGE}"
            );
        }
        self.since = Some(now);
        self.reprobe_after = reprobe_after;
        self.evidence = false;
    }

    /// A later stage acknowledged a structured-output step or probe.
    fn on_evidence(&mut self) {
        self.evidence = true;
    }

    /// A late acknowledged reply: the chain masks after all.
    fn clear(&mut self) {
        self.evidence = true;
        if self.since.take().is_some() {
            info!("gemma4: a later stage acknowledged a structured-output step; structured output re-enabled");
        }
    }

    /// `Err` while latched; an expired latch is dropped (the next masked
    /// step probes the chain again).
    fn refusal(&mut self, now: std::time::Instant) -> EngineResult<()> {
        match self.since {
            Some(t) if now.saturating_duration_since(t) < self.reprobe_after => {
                Err(EngineError::Backend(UPGRADE_EVERY_STAGE.into()))
            }
            Some(_) => {
                self.since = None;
                Ok(())
            }
            None => Ok(()),
        }
    }
}

/// The structured-output gate of a task on this pipeline: `Err` (the
/// upgrade message) while the latch is set, for a structured-output task on
/// a multi-stage pipeline. Checked at activation, before any warm-resume
/// exchange or frame, and again on every step of such a task.
fn latched_refusal(
    latch: &mut MaskLatch,
    multi_stage: bool,
    has_grammar: bool,
    now: std::time::Instant,
) -> EngineResult<()> {
    if multi_stage && has_grammar {
        latch.refusal(now)
    } else {
        Ok(())
    }
}

/// The head's token from a masked step's wait, or the error that fails the
/// task with how to log it.
fn head_masked_token(
    w: MaskedWait,
    latch: &mut MaskLatch,
    now: std::time::Instant,
) -> Result<i32, (EngineError, Option<GrammarFailure>)> {
    match w {
        MaskedWait::Token(t) => {
            latch.on_evidence();
            Ok(t)
        }
        MaskedWait::Nack => Err((
            EngineError::Backend(
                "a later pipeline stage could not apply the structured-output mask to this step"
                    .into(),
            ),
            Some(GrammarFailure::Nack),
        ))
        .inspect_err(|_| latch.on_evidence()),
        // A stage below lost its mask state: fail the task (the client may
        // retry), never latch.
        MaskedWait::MaskMiss => Err((EngineError::Backend(MASK_MISS_MESSAGE.into()), None)),
        MaskedWait::Unacked(why) => {
            latch.trip(now);
            Err((
                EngineError::Backend(format!("{why}; {UPGRADE_EVERY_STAGE}")),
                Some(GrammarFailure::Upgrade),
            ))
        }
        // Never the answer to a masked step (see `wait_masked_reply`).
        MaskedWait::ProbeOk => Err((
            EngineError::Backend("a mask-support probe answer to a structured-output step".into()),
            None,
        )),
        MaskedWait::Failed(e) => Err((e, None)),
    }
}

/// The frames of a mask-support probe (see the stage-wire notes above
/// `MASK_ACK`): position 0, the one-row probe hidden, and for each of `tags`
/// (this stage's cross-KV outputs, so the next stage reads the count it
/// expects) an F32 `[1,1,1]` zero frame.
fn probe_frames(tags: &[i64]) -> StepFrames {
    let mut hid = 0u32.to_le_bytes().to_vec();
    hid.extend_from_slice(&MASK_PROBE.to_le_bytes());
    let zero = WireTensor::new(WireDType::F32, [1, 1, 1], vec![0; 4]);
    StepFrames {
        pos: encode_wire_position(0),
        hid: WireTensor::new(WireDType::I32, [1, 1, 2], hid),
        header: encode_cross_kv_header(tags),
        cross: vec![zero; tags.len()],
        mask: None,
    }
}

/// The wait for a probe's answer at a stage with `stages_below` stages after
/// it: `probe_timeout` just before the tail, a fifth of it more per stage
/// above (the [`masked_wait_budget`] pattern), so a middle above a silent
/// stage answers inside its upstream's wait.
fn probe_wait_budget(probe_timeout: Duration, stages_below: u32) -> Duration {
    let hops = stages_below.saturating_sub(1);
    probe_timeout.saturating_add((probe_timeout / 5).saturating_mul(hops))
}

/// A probe's wait: bounded, never closes the socket. Its answer is
/// [`StageReply::ProbeOk`]; a middle's not-acknowledged answer means a stage
/// below can't mask; an acknowledged NACK means a stage that masks failed it
/// (a lost link below a middle). Every other reply is a late one to an earlier step
/// (gemma4 replies carry no sequence number): skipped, the wait goes on. The
/// bool says a skipped reply was acknowledged (a late mask miss proves
/// nothing).
async fn wait_probe_reply(g: &mut ActivationClient, budget: Duration) -> (MaskedWait, bool) {
    let deadline = std::time::Instant::now() + budget;
    let mut late_ack = false;
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let tensor = match g.recv_token(remaining).await {
            Ok((t, _)) => t,
            Err(cascadia_transport::TransportError::FrameStartTimeout(_)) => {
                return (
                    MaskedWait::Unacked(format!(
                        "no answer within {budget:?} to a mask-support probe"
                    )),
                    late_ack,
                )
            }
            Err(e) => {
                return (
                    MaskedWait::Failed(EngineError::Backend(e.to_string())),
                    late_ack,
                )
            }
        };
        match decode_stage_reply(&tensor) {
            Ok(StageReply::ProbeOk) => return (MaskedWait::ProbeOk, late_ack),
            // A stage that masks failed the probe (a middle lost its link
            // below): a failed step, not an older stage. The late replies
            // before the probe were drained, so this is its answer.
            Ok(StageReply::Nack) => return (MaskedWait::Nack, late_ack),
            Ok(StageReply::NotAcked) => {
                return (
                    MaskedWait::Unacked(
                        "a later stage relayed a mask-support probe to a stage that did not \
                         acknowledge it"
                            .into(),
                    ),
                    late_ack,
                )
            }
            Ok(reply) => {
                debug!(
                    ?reply,
                    "discarding a late reply in a mask-support probe's wait"
                );
                late_ack |= matches!(reply, StageReply::Token(_));
            }
            Err(e) => return (MaskedWait::Failed(e), late_ack),
        }
    }
}

/// A probe on the downstream link, as the head and every middle run it:
/// discard late replies, send the probe (the send cache untouched), wait for
/// its answer. The bool says a discarded late reply was acknowledged.
async fn probe_round(
    g: &mut ActivationClient,
    frames: StepFrames,
    budget: Duration,
) -> (MaskedWait, bool) {
    let late_ack = match drain_late_replies(g).await {
        Ok(late_ack) => late_ack,
        Err(e) => return (MaskedWait::Failed(e), false),
    };
    if let Err(e) = send_step(g, &frames.pos, &frames.hid, &frames.header, &frames.cross).await {
        return (
            MaskedWait::Failed(EngineError::Backend(e.to_string())),
            late_ack,
        );
    }
    let (wait, more) = wait_probe_reply(g, budget).await;
    (wait, late_ack || more)
}

/// A middle's probe after it read one from upstream: forward `frames` (its
/// own probe) and wait. Returns the answer it owes upstream —
/// [`StageReply::ProbeOk`] only when the stage below answered so, else
/// [`StageReply::NotAcked`] — and whether the round discarded an
/// acknowledged late reply from below.
async fn middle_probe_step(
    down: &mut ActivationClient,
    frames: StepFrames,
    budget: Duration,
) -> (StageReply, bool) {
    let (wait, late_ack) = probe_round(down, frames, budget).await;
    let reply = match wait {
        MaskedWait::ProbeOk => StageReply::ProbeOk,
        // This stage's link below failed, or a stage below that masks failed
        // the probe: an acknowledged NACK, so the head fails the task without
        // taking it for an older stage.
        MaskedWait::Failed(e) => {
            warn!(error = %e, "gemma4: the link below failed a mask-support probe");
            StageReply::Nack
        }
        MaskedWait::Nack => StageReply::Nack,
        other => {
            warn!(
                ?other,
                "gemma4: the stage below did not acknowledge a mask-support probe"
            );
            StageReply::NotAcked
        }
    };
    (reply, late_ack)
}

/// The head's verdict on a probe's answer: `Ok` records the evidence that
/// every later stage masks; silence or a not-acknowledged answer latches
/// (for [`MASK_PROBE_REPROBE_AFTER`]) and fails the task with the upgrade
/// message; a link failure fails the task without latching.
fn head_probe_verdict(
    w: MaskedWait,
    latch: &mut MaskLatch,
    now: std::time::Instant,
) -> Result<(), (EngineError, Option<GrammarFailure>)> {
    match w {
        MaskedWait::ProbeOk => {
            latch.on_evidence();
            Ok(())
        }
        MaskedWait::Unacked(why) => {
            latch.trip_probe(now);
            Err((
                EngineError::Backend(format!("{why}; {UPGRADE_EVERY_STAGE}")),
                Some(GrammarFailure::Upgrade),
            ))
        }
        MaskedWait::Failed(e) => Err((e, None)),
        // A stage that masks failed the probe: the chain masks; the task
        // fails as a lost step, nothing is latched.
        MaskedWait::Nack => {
            latch.on_evidence();
            Err((
                EngineError::Backend(
                    "a later pipeline stage failed the mask-support probe; retry the request"
                        .into(),
                ),
                Some(GrammarFailure::Nack),
            ))
        }
        other => Err((
            EngineError::Backend(format!(
                "unexpected answer to a mask-support probe: {other:?}"
            )),
            None,
        )),
    }
}

/// Whether a structured-output task admitted now must probe the pipeline
/// first: more than one stage, and no evidence yet that every later stage
/// masks (never shown, or latched since).
fn needs_mask_probe(latch: &MaskLatch, multi_stage: bool, has_grammar: bool) -> bool {
    multi_stage && has_grammar && !latch.evidence
}

/// A queued task with its grammar compiled at submit.
struct PendingTask {
    task: GenerationTask,
    grammar: Option<CompiledGrammar>,
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
    /// Builds structured-output grammars for this tokenizer; `Err` holds why
    /// it couldn't be built (structured-output tasks are then refused).
    grammar_factory: structured::LazyGrammarFactory,
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
    pending: Vec<PendingTask>,
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
    /// Rate limit of the structured-output task-failure WARN.
    grammar_warn: StepWarnLimiter,
    /// Why the current step failed when the failure is a structured-output
    /// task's own (set where it happens, taken by `step`); `None` otherwise.
    grammar_failure: Option<GrammarFailure>,
    /// Head: set when a later stage didn't acknowledge a masked step.
    mask_latch: MaskLatch,
    /// The last token mask sent downstream, so a repeat goes as the short
    /// marker. Cleared on every failed step, an ABORT and a latch.
    mask_send: MaskSendCache,
    /// Relay stages: the last token mask decoded from the upstream, which a
    /// repeat marker stands for. The upstream link never re-dials.
    mask_recv: MaskRecvCache,
    /// Middles: see [`OwedLateAck`].
    owed_late_ack: OwedLateAck,
    /// Pipeline stages after this one (0 on the tail); sizes masked waits.
    stages_below: u32,
    /// How long the stage just before the tail waits for the answer to a
    /// mask-support probe; stages above it wait a little more
    /// ([`probe_wait_budget`]).
    mask_probe_timeout: Duration,
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

    /// Sends one step downstream. `mask` (the next token's mask words) only
    /// on a masked step: the hidden frame then carries it (see
    /// [`encode_hidden_frame`]); every other frame is unchanged.
    fn send_hidden_downstream(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
        mask: Option<Vec<u32>>,
    ) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let mut f = self.step_frames(hidden, shape, position, mask)?;
        let cache = &mut self.mask_send;
        crate::dist_spec::run_async_pub(&self.runtime_handle, async move {
            let mut guard = downstream.lock().await;
            send_step_cached(&mut guard, &mut f, cache).await
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    /// The frames of one step from this stage's last inference. `mask` moves
    /// into the frames: the send compares and records it there.
    fn step_frames(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
        mask: Option<Vec<u32>>,
    ) -> EngineResult<StepFrames> {
        // Gemma 4 activations are f32 (the IR's hidden_states input/output are
        // f32). Send raw f32 so the downstream stage feeds it without a lossy
        // f16 round-trip — matches the f32 Core reference.
        let hid = encode_hidden_frame(hidden, shape, mask.as_deref())?;
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
        Ok(StepFrames {
            pos,
            hid,
            header: encode_cross_kv_header(&tags),
            cross: cross_frames,
            mask,
        })
    }

    /// A plain step's token (see [`wait_plain_reply`]). A late acknowledged
    /// reply it skips proves the chain masks: it clears the latch.
    fn recv_token_from_downstream(&mut self, prefill: bool) -> EngineResult<i32> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
        let cache = &mut self.mask_send;
        let (token, late_ack) =
            crate::dist_spec::run_async_pub(&self.runtime_handle, async move {
                let mut guard = downstream.lock().await;
                plain_reply_settled(&mut guard, prefill, cache).await
            })?;
        if late_ack {
            self.mask_latch.clear();
            self.owed_late_ack.on_ack_from_below();
        }
        Ok(token)
    }

    /// The upstream's next step: its hidden frame, resolved against this
    /// link's receive cache once every frame of the step is read (a repeat
    /// marker becomes the last mask decoded, or [`HiddenIn::MaskMiss`]), and
    /// its position. A read or decode error clears the cache.
    fn recv_hidden_from_upstream(&mut self) -> EngineResult<(HiddenIn, i64)> {
        match self.recv_step_from_upstream() {
            Ok((hidden_in, position)) => {
                Ok((resolve_hidden(hidden_in, &mut self.mask_recv), position))
            }
            Err(e) => {
                self.mask_recv.clear();
                Err(e)
            }
        }
    }

    fn recv_step_from_upstream(&mut self) -> EngineResult<(HiddenIn, i64)> {
        // Wire order per step: position frame, hidden frame, cross-KV header, then `count` cross-KV
        // frames. The position read transparently absorbs any I8 control frame (CAPTURE/RESTORE)
        // that arrives between turns (stateful workers only).
        let pos_t = self.recv_position_or_control()?;
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        let consumes = self.external_kv_in.len();
        // pos was already read via recv_position_or_control (the IDLE wait that
        // also absorbs a CAPTURE/RESTORE control frame between turns); the
        // frames that FOLLOW are mid-step replies (Item 5).
        let (hid_t, tags, ext_tensors) = self.block_on(async move {
            let mut guard = upstream.lock().await;
            recv_step_frames(&mut guard, consumes).await
        })?;
        // Decode + strictly validate the position outside the transport
        // closure (so a bad frame is a clear EngineError, not a silent desync).
        let position = decode_wire_position(&pos_t)?;
        // Every frame of the step is read before the hidden is decoded, so a
        // frame this stage rejects leaves the link aligned. A mask-support
        // probe leaves this stage's cross-KV inputs as they were.
        let hidden_in = decode_hidden_frame(&hid_t)?;
        if hidden_in == HiddenIn::Probe {
            return Ok((hidden_in, position));
        }
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
        Ok((hidden_in, position))
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
        let frames = upstream_plain_frames(&mut self.owed_late_ack, encode_token_reply(token));
        self.block_on(async move {
            let mut guard = upstream.lock().await;
            for f in &frames {
                guard.send(f).await?;
            }
            Ok::<_, cascadia_transport::TransportError>(())
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    /// A masked step's two-lane reply upstream. `late_ack`: this step's
    /// masked round discarded an acknowledged late reply from below.
    fn send_masked_reply_upstream(
        &mut self,
        reply: StageReply,
        late_ack: bool,
    ) -> EngineResult<()> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        let tensor = encode_masked_reply(reply);
        self.owed_late_ack.after_masked_reply(reply, late_ack);
        self.block_on(async move {
            let mut guard = upstream.lock().await;
            guard.send(&tensor).await
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        Ok(())
    }

    fn step_first(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.active.is_none() && !self.pending.is_empty() {
            let PendingTask { task, grammar } = self.pending.remove(0);
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
                let vocab = crate::resume_vocab_bound(&tok);
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
            // Structured output: the matcher starts after the resume seed (if
            // any), before any model work, so a seed the grammar rejects costs
            // nothing and fails this task only.
            let grammar = match grammar.as_ref().map(|g| {
                start_grammar(g, &cascadia_types::resume_generated_seed(resume.as_deref()))
            }) {
                None => None,
                Some(Ok(mask)) => Some(mask),
                Some(Err(e)) => {
                    let id = task.task_id.clone();
                    return Ok(vec![(id.clone(), Chunk::error(id, e.to_string()))]);
                }
            };
            // A structured-output task on a pipeline that didn't acknowledge an
            // earlier masked step fails here, before any warm-resume exchange.
            // A slow but new tail's late answer, already in the socket, clears
            // the latch first.
            let multi_stage = !(self.spec.is_first_stage && self.spec.is_last_stage);
            if multi_stage && grammar.is_some() && self.mask_latch.since.is_some() {
                self.drain_late_downstream_replies();
            }
            if let Err(e) = latched_refusal(
                &mut self.mask_latch,
                multi_stage,
                grammar.is_some(),
                std::time::Instant::now(),
            ) {
                self.grammar_failure = Some(GrammarFailure::Latched);
                return Err(e.for_task(task.task_id.clone()));
            }
            // No evidence yet that every later stage masks: probe now, before
            // any warm-resume exchange or prefill, so an older stage costs this
            // task a short probe wait, not a token deadline.
            if needs_mask_probe(&self.mask_latch, multi_stage, grammar.is_some()) {
                if let Err((e, kind)) = self.probe_downstream() {
                    self.grammar_failure = kind;
                    return Err(e.for_task(task.task_id.clone()));
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
                match tok.decode(&seed_u32, true) {
                    Ok(t) => t,
                    Err(e) => {
                        // Attributed like the validation branch above: the task is
                        // already popped from pending, so a bare Err from step()
                        // fails whichever queued stream happens to poll while THIS
                        // task's stream hangs with nothing further.
                        let id = task.task_id.clone();
                        return Ok(vec![(
                            id.clone(),
                            Chunk::error(id, format!("resume seed decode failed: {e}")),
                        )]);
                    }
                }
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
                grammar,
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
        let (prefill, tokens, seed_len) = {
            let a = self.active.as_mut().unwrap();
            if !a.prefilled {
                a.prefilled = true;
                // warm_prefix is 0 on the cold/default path ⇒ full prompt (unchanged).
                (
                    true,
                    a.prompt_ids[a.warm_prefix..].to_vec(),
                    a.resume_seed_len,
                )
            } else {
                (false, vec![a.last_token as i64], 0)
            }
        };
        let single_stage = self.spec.is_first_stage && self.spec.is_last_stage;

        // A structured-output task on a pipeline that didn't acknowledge an
        // earlier masked step fails at once, before any frame is sent.
        let has_grammar = self.active.as_ref().is_some_and(|a| a.grammar.is_some());
        if let Err(e) = latched_refusal(
            &mut self.mask_latch,
            !single_stage,
            has_grammar,
            std::time::Instant::now(),
        ) {
            self.grammar_failure = Some(GrammarFailure::Latched);
            return Err(e);
        }

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
        // Option B: a resumed prefill must reproduce the DONOR's compute
        // pattern — the rendered prompt in the engine's normal (batched)
        // prefill shape, then the resume SEED folded one token at a time,
        // exactly as the donor generated it token-by-token. Folding the seed
        // into the prompt batch changes the int4 GEMM batch shape, and the
        // rounding delta at the seam flips the first continued token under
        // greedy (rig 2026-08-27: the alternate gemma4 continued 3 tokens
        // then bailed to its EOS mode — REGENERATED verdict). Same numerics
        // class the FORCE_T1_PREFILL knob exists for, but resume needs the
        // SPLIT shape (all-T=1 would instead diverge the PROMPT segment from
        // the donor's batched prefill), and needs it unconditionally.
        let seed_split = if prefill {
            tokens.len().saturating_sub(seed_len)
        } else {
            tokens.len()
        };
        let mut alpha = std::time::Duration::ZERO;
        let mut wire = std::time::Duration::ZERO;
        let next_token: i32;
        let position = self.position;
        // A structured-output task's matcher, held out of `active` for this
        // step's pick (emit_token puts it back). `None` for every other task.
        let mut grammar = self.active.as_mut().and_then(|a| a.grammar.take());
        if prefill && tokens.len() > 1 && (chunk < tokens.len() || seed_split < tokens.len()) {
            next_token = fold_prefill(
                tokens.len(),
                chunk,
                seed_split,
                grammar.as_mut(),
                |range, mask| {
                    let span = &tokens[range];
                    let pos = self.position;
                    let ts = std::time::Instant::now();
                    let (out, shape) = self.run_first(span, pos)?;
                    alpha += ts.elapsed();
                    self.position += span.len() as i64;
                    // Every span goes downstream so each stage folds the same way; the tokens from all
                    // but the final span are discarded (the last one is the first decode token).
                    // Always use the PREFILL (widened) deadline here even for T=1 spans: this is a
                    // prefill, and the first span after a warm RESTORE follows a heavy set_state on the
                    // downstream stage — the strict decode budget wedges there and the turn returns empty.
                    let (tok, w) =
                        self.resolve_next_token(&out, &shape, single_stage, pos, true, mask)?;
                    wire += w;
                    Ok(tok)
                },
            )?;
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
                step_mask(grammar.as_mut(), prefill, tokens.len(), tokens.len()),
            )?;
            wire = w;
            next_token = tok;
        }
        if let Some(a) = self.active.as_mut() {
            a.t_alpha_compute += alpha;
            a.t_wire += wire;
            a.grammar = grammar;
        }

        self.emit_token(next_token)
    }

    /// From a first-stage output, produce the next token: argmax locally when
    /// this is the only stage, otherwise forward the hidden state downstream
    /// and await the token from the pipeline tail. Returns the token + the
    /// wire round-trip time (zero for single-stage).
    ///
    /// `mask` is the structured-output matcher, passed only when this output's
    /// token is the next generated one (see [`pick_next_token`]). On a
    /// multi-stage pipeline its mask words ride the step to the tail, which
    /// picks the token and acknowledges the mask.
    fn resolve_next_token(
        &mut self,
        out: &[f32],
        shape: &[usize],
        single_stage: bool,
        position: i64,
        prefill: bool,
        mask: Option<&mut GrammarMask>,
    ) -> EngineResult<(i32, std::time::Duration)> {
        if single_stage {
            match pick_tagged(out, shape, mask) {
                Ok(token) => Ok((token, std::time::Duration::ZERO)),
                Err((e, kind)) => {
                    self.grammar_failure = kind;
                    Err(e)
                }
            }
        } else if let Some(mask) = mask {
            let s3 = to_shape3(shape);
            let ts = std::time::Instant::now();
            let width = self
                .grammar_factory
                .mask_width()
                .map_err(EngineError::Backend)?;
            // A zero width is a broken factory, not a schema dead end.
            let words = mask_words(mask, width).inspect_err(|_| {
                self.grammar_failure = (width > 0).then_some(GrammarFailure::DeadEnd)
            })?;
            let mut frames = self.step_frames(out, s3, position, Some(words))?;
            let downstream = self
                .downstream
                .clone()
                .ok_or_else(|| EngineError::Backend("no downstream".into()))?;
            let budget = masked_wait_budget(prefill, self.stages_below);
            let cache = &mut self.mask_send;
            let (wait, late_ack) =
                crate::dist_spec::run_async_pub(&self.runtime_handle, async move {
                    let mut guard = downstream.lock().await;
                    masked_round(&mut guard, &mut frames, budget, cache).await
                });
            if late_ack {
                self.mask_latch.clear();
            }
            match head_masked_token(wait, &mut self.mask_latch, std::time::Instant::now()) {
                Ok(token) => Ok((token, ts.elapsed())),
                Err((e, kind)) => {
                    self.grammar_failure = kind;
                    Err(e)
                }
            }
        } else {
            let s3 = to_shape3(shape);
            let ts = std::time::Instant::now();
            self.send_hidden_downstream(out, s3, position, None)?;
            let token = self.recv_token_from_downstream(prefill)?;
            Ok((token, ts.elapsed()))
        }
    }

    /// Decode the delta text for `next_token`, append it to the active task,
    /// check stop conditions, and build the streamed chunk (first stage only).
    fn emit_token(&mut self, next_token: i32) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.active.is_none() {
            return Err(EngineError::Backend(
                "emit_token with no active task".into(),
            ));
        }
        // Structured output: advance the matcher past the picked token (a
        // rejection ends the task); a complete JSON ends it with `stop`.
        let grammar_finished = advance_grammar(&mut self.active, next_token)
            .inspect_err(|_| self.grammar_failure = Some(GrammarFailure::Rejected))?;
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
        let end = step_end(active.generated.len(), max_tokens, is_eos, grammar_finished);
        let is_final = end.is_some();

        let task_id = active.task.task_id.clone();
        let chunk = if is_final {
            final_chunk(
                task_id.clone(),
                next_token as i64,
                delta,
                active.prompt_ids.len(),
                end == Some(FinishReason::Stop),
            )
        } else {
            // Explicit count: a token whose text lands in the next chunk (BPE
            // splitting a glyph) would otherwise count 0 via the non-empty fallback.
            Chunk::token(task_id.clone(), next_token as i64, delta)
                .with_n_tokens(1)
                .with_token_ids(vec![next_token as i64])
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

    /// Relay stages: reset own KV at a new sequence (position 0) and run
    /// this stage's slice of the model.
    fn relay(
        &mut self,
        hidden: &[f32],
        shape: [usize; 3],
        position: i64,
    ) -> EngineResult<(Vec<f32>, Vec<usize>)> {
        if position == 0 {
            self.runtime.reset_state().map_err(map_ov_err)?;
        }
        self.run_relay(hidden, shape, position)
    }

    fn step_last(&mut self) -> EngineResult<()> {
        // The upstream stage carries the absolute start-position; use it
        // directly as the position_ids base and reset own KV when it is 0 (a
        // new sequence). Correct for any prompt length, including 1 token.
        let (hidden_in, position) = self.recv_hidden_from_upstream()?;
        match hidden_in {
            HiddenIn::Plain { floats, shape } => {
                let (out, out_shape) = self.relay(&floats, shape, position)?;
                let next = argmax_logits(&out, &out_shape)?;
                self.send_token_to_upstream(next)
            }
            HiddenIn::Masked {
                floats,
                shape,
                words,
            } => {
                let relayed = self.relay(&floats, shape, position);
                let (reply, err) = tail_masked_reply(relayed, &words);
                // A failed relay still answers (an acknowledged NACK), so the
                // head fails this task instead of waiting out its deadline.
                let sent = self.send_masked_reply_upstream(reply, false);
                match err {
                    Some(e) => Err(e),
                    None => sent,
                }
            }
            HiddenIn::BadMask(why) => {
                warn!(%why, "gemma4: can't read a structured-output step; answering NACK");
                self.send_masked_reply_upstream(StageReply::Nack, false)
            }
            HiddenIn::MaskMiss => self.answer_mask_miss(),
            // Nothing runs: the answer goes up at once.
            HiddenIn::Probe => self.send_masked_reply_upstream(StageReply::ProbeOk, false),
            // `recv_hidden_from_upstream` resolves every marker.
            HiddenIn::Repeat { .. } => self.send_masked_reply_upstream(StageReply::Nack, false),
        }
    }

    /// A middle's mask-support probe: forward its own probe (nothing runs)
    /// and pass the answer up (see [`middle_probe_step`]).
    fn relay_probe(&mut self) -> EngineResult<()> {
        let tags: Vec<i64> = self.cross_kv_out.iter().map(|o| o.tag).collect();
        let budget = probe_wait_budget(self.mask_probe_timeout, self.stages_below);
        let (reply, late_ack) = match self.downstream.clone() {
            Some(downstream) => crate::dist_spec::run_async_pub(&self.runtime_handle, async move {
                let mut guard = downstream.lock().await;
                middle_probe_step(&mut guard, probe_frames(&tags), budget).await
            }),
            None => (StageReply::Nack, false),
        };
        self.send_masked_reply_upstream(reply, late_ack)
    }

    /// Head: probe the pipeline for mask support (see [`probe_round`]) and
    /// judge the answer ([`head_probe_verdict`]).
    fn probe_downstream(&mut self) -> Result<(), (EngineError, Option<GrammarFailure>)> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| (EngineError::Backend("no downstream".into()), None))?;
        let tags: Vec<i64> = self.cross_kv_out.iter().map(|o| o.tag).collect();
        let budget = probe_wait_budget(self.mask_probe_timeout, self.stages_below);
        let t0 = std::time::Instant::now();
        let (wait, late_ack) = crate::dist_spec::run_async_pub(&self.runtime_handle, async move {
            let mut guard = downstream.lock().await;
            probe_round(&mut guard, probe_frames(&tags), budget).await
        });
        if late_ack {
            self.mask_latch.clear();
        }
        let verdict = head_probe_verdict(wait, &mut self.mask_latch, std::time::Instant::now());
        info!(
            ok = verdict.is_ok(),
            took_ms = t0.elapsed().as_millis() as u64,
            "gemma4: mask-support probe answered"
        );
        verdict
    }

    /// Head: skip, without waiting, replies already in the downstream socket;
    /// an acknowledged one (a slow but new tail's late answer) clears the
    /// latch.
    fn drain_late_downstream_replies(&mut self) {
        let Some(downstream) = self.downstream.clone() else {
            return;
        };
        let drained = crate::dist_spec::run_async_pub(&self.runtime_handle, async move {
            let mut guard = downstream.lock().await;
            drain_late_replies(&mut guard).await
        });
        match drained {
            Ok(true) => self.mask_latch.clear(),
            Ok(false) => {}
            Err(e) => warn!(error = %e, "gemma4: reading late replies from downstream failed"),
        }
    }

    /// Answers a repeat marker this stage has no mask for: the mask-miss
    /// reply, with no layer run and nothing forwarded.
    fn answer_mask_miss(&mut self) -> EngineResult<()> {
        warn!("gemma4: a repeat token mask arrived with no mask to repeat; answering a mask miss");
        self.send_masked_reply_upstream(StageReply::MaskMiss, false)
    }

    fn step_middle(&mut self) -> EngineResult<()> {
        let (hidden_in, position) = self.recv_hidden_from_upstream()?;
        let (hidden, shape, words) = match hidden_in {
            HiddenIn::Plain { floats, shape } => (floats, shape, None),
            HiddenIn::Masked {
                floats,
                shape,
                words,
            } => (floats, shape, Some(words)),
            HiddenIn::BadMask(why) => {
                warn!(%why, "gemma4: can't read a structured-output step; answering NACK");
                return self.send_masked_reply_upstream(StageReply::Nack, false);
            }
            HiddenIn::MaskMiss => return self.answer_mask_miss(),
            HiddenIn::Probe => return self.relay_probe(),
            // `recv_hidden_from_upstream` resolves every marker.
            HiddenIn::Repeat { .. } => {
                return self.send_masked_reply_upstream(StageReply::Nack, false)
            }
        };
        // Multi-token hidden = prefill: the token reply waits on every
        // remaining stage's whole-prompt compute — widened budget.
        let prefill_reply = shape[1] > 1;
        let Some(words) = words else {
            let (out, out_shape) = self.relay(&hidden, shape, position)?;
            let s3 = to_shape3(&out_shape);
            // Forward the SAME absolute position downstream so every stage aligns.
            self.send_hidden_downstream(&out, s3, position, None)?;
            let token = self.recv_token_from_downstream(prefill_reply)?;
            return self.send_token_to_upstream(token);
        };
        // Masked step: relay the mask with the hidden, and pass the
        // acknowledgement up only when the downstream acknowledged. A failure
        // here still answers upstream (an acknowledged NACK), so the head
        // fails this task instead of waiting out its deadline.
        let prepared = self
            .relay(&hidden, shape, position)
            .and_then(|(out, out_shape)| {
                self.step_frames(&out, to_shape3(&out_shape), position, Some(words))
            });
        let budget = masked_wait_budget(prefill_reply, self.stages_below);
        let cache = &mut self.mask_send;
        let (reply, err, late_ack) = match self.downstream.clone() {
            Some(downstream) => crate::dist_spec::run_async_pub(&self.runtime_handle, async move {
                let mut guard = downstream.lock().await;
                middle_masked_step(&mut guard, prepared, budget, cache).await
            }),
            None => (
                StageReply::Nack,
                Some(EngineError::Backend("no downstream".into())),
                false,
            ),
        };
        let sent = self.send_masked_reply_upstream(reply, late_ack);
        match err {
            Some(e) => Err(e),
            None => sent,
        }
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
        if self.pending.iter().any(|p| p.task.task_id == task.task_id)
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
        let grammar = self
            .grammar_factory
            .compile(self.tokenizer.as_deref(), &task)?;
        structured::log_grammar_warnings("gemma4", &task.task_id, grammar.as_ref());
        self.pending.push(PendingTask { task, grammar });
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Step-wise engine: drop the queued task and clear the active one so
        // step() stops decoding it. cancel/step never overlap (runner mutex).
        self.pending.retain(|p| &p.task.task_id != task_id);
        if self
            .active
            .as_ref()
            .is_some_and(|a| &a.task.task_id == task_id)
        {
            self.active = None;
        }
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        self.grammar_failure = None;
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
            Err(e) => report_step_error(
                &mut self.step_warn,
                &mut self.grammar_warn,
                self.grammar_failure.take(),
                e,
                std::time::Instant::now(),
            ),
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
///
/// Two-lane I32 `[1,1,2]` replies are late answers to an earlier
/// structured-output step (they carry no sequence number): skipped, so they
/// are never taken for the ack, which would shift every later reply by one.
/// The bool says one of them was acknowledged. One bound covers the whole read.
#[cfg(feature = "kv_coord")]
async fn recv_control_ack_bounded(
    g: &mut cascadia_transport::ActivationClient,
) -> Result<(WireTensor, bool), cascadia_transport::TransportError> {
    tokio::time::timeout(G_CONTROL_ACK_TIMEOUT, async {
        let mut late_ack = false;
        loop {
            let (t, _) = g.recv().await?;
            if t.dtype == WireDType::I32 && t.shape == [1, 1, 2] {
                let reply = decode_stage_reply(&t);
                debug!(
                    ?reply,
                    "discarding a late reply to an earlier structured-output step"
                );
                // A late mask miss proves nothing about the chain.
                late_ack |= matches!(
                    reply,
                    Ok(StageReply::Token(_) | StageReply::Nack | StageReply::ProbeOk)
                );
                continue;
            }
            return Ok((t, late_ack));
        }
    })
    .await
    .map_err(|_| cascadia_transport::TransportError::SocketClosed)?
}

/// One CAPTURE/RESTORE/ABORT exchange with downstream: send `frame`, read its
/// ack (see [`recv_control_ack_bounded`]); the bool says a late acknowledged
/// structured-output reply was discarded on the way.
#[cfg(feature = "kv_coord")]
async fn control_exchange(
    g: &mut cascadia_transport::ActivationClient,
    frame: &WireTensor,
) -> Result<(WireTensor, bool), cascadia_transport::TransportError> {
    g.send(frame).await?;
    recv_control_ack_bounded(g).await
}

/// An I8 control frame: opcode then payload.
#[cfg(feature = "kv_coord")]
fn control_frame(opcode: u8, payload: &[u8]) -> WireTensor {
    let mut data = vec![opcode];
    data.extend_from_slice(payload);
    WireTensor::new(WireDType::I8, [1, 1, data.len() as u32], data)
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
        let t = if tenant.is_empty() {
            control_frame(
                G_OPCODE_CAPTURE,
                &crate::kv_coordination::capture_body_bytes(epoch, tokens),
            )
        } else {
            control_frame(
                G_OPCODE_CAPTURE_V2,
                &crate::kv_coordination::capture_body_bytes_v2(epoch, tokens, tenant),
            )
        };
        // Bounded, for the same reason ov-runtime bounds its CAPTURE ack: a peer on an
        // older build that meets G_OPCODE_CAPTURE_V2 errors WITHOUT acking, and an
        // unbounded wait here wedges this rank while it holds the downstream lock.
        let ack = self.control_exchange_downstream(downstream, t)?;
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
        let t = control_frame(G_OPCODE_RESTORE, &epoch.to_le_bytes());
        // Bounded like CAPTURE above: an unbounded wait on a dead peer
        // wedges this rank while it holds the downstream lock.
        let ack = self.control_exchange_downstream(downstream, t)?;
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
        // Bounded like CAPTURE above: an unbounded wait on a dead peer
        // wedges this rank while it holds the downstream lock.
        // The chain resets: the next masked step goes full.
        self.mask_send.clear();
        self.control_exchange_downstream(downstream, control_frame(G_OPCODE_ABORT, &[]))
            .map(|_| ())
    }

    /// [`control_exchange`] on the downstream link; a late acknowledged
    /// structured-output reply it skipped clears the latch.
    fn control_exchange_downstream(
        &mut self,
        downstream: Arc<tokio::sync::Mutex<ActivationClient>>,
        frame: WireTensor,
    ) -> EngineResult<WireTensor> {
        let (ack, late_ack) = self
            .block_on(async move {
                let mut g = downstream.lock().await;
                control_exchange(&mut g, &frame).await
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        if late_ack {
            self.mask_latch.clear();
            self.owed_late_ack.on_ack_from_below();
        }
        Ok(ack)
    }

    fn send_control_ack_upstream(&mut self, ack_opcode: u8, payload: &[u8]) -> EngineResult<()> {
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream".into()))?;
        let frames =
            upstream_plain_frames(&mut self.owed_late_ack, control_frame(ack_opcode, payload));
        self.block_on(async move {
            let mut g = upstream.lock().await;
            for f in &frames {
                g.send(f).await?;
            }
            Ok::<_, cascadia_transport::TransportError>(())
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

        // A pre-v1.1 tree still loads (the gap only shows past the window), but
        // it has to reach the operator: on a multi-node deployment nobody reads
        // each node's stderr, so the same warning also rides the load stream.
        if sliding_layers_unwindowed(&stage_cfg) {
            warn!(
                stage_dir = %stage_dir.display(),
                export_version = stage_cfg.export_version.as_deref().unwrap_or("?"),
                "gemma4: sliding_attention layers carry no sliding_window (pre-v1.1 export): \
                 prompts longer than the model's window will degrade; re-export with \
                 `cascadia shard` to get gemma4_cached_v1.1"
            );
            events.push(LoadProgress::message(format!(
                "warning: {} has sliding_attention layers with no sliding_window (pre-v1.1 \
                 export); prompts longer than the model's window will degrade — re-export with \
                 `cascadia shard`",
                stage_dir.display()
            )));
        }

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

        // Structured output is built on the first json task, on the stage
        // that receives tasks.
        let grammar_factory = if spec.is_first_stage {
            structured::LazyGrammarFactory::new("gemma4", self.eos_token_ids.clone())
        } else {
            structured::LazyGrammarFactory::unavailable("this stage has no tokenizer")
        };

        Ok(Box::new(Gemma4Engine {
            spec,
            num_kv_shared_layers: self.num_kv_shared_layers,
            num_stages: self.total,
            runtime,
            tokenizer: self.tokenizer,
            eos_token_ids: self.eos_token_ids.clone(),
            grammar_factory,
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
            grammar_warn: StepWarnLimiter::default(),
            grammar_failure: None,
            mask_latch: MaskLatch::default(),
            mask_send: MaskSendCache::default(),
            mask_recv: MaskRecvCache::default(),
            owed_late_ack: OwedLateAck::default(),
            mask_probe_timeout: MASK_PROBE_TIMEOUT,
            stages_below: self.total.saturating_sub(self.rank + 1),
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
        let _knob = crate::timeout_knob_lock();
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

    /// A pre-v1.1 tree: sliding layers, no `sliding_window`, no
    /// `export_version`. Both optional keys must be absent-tolerant (a stale
    /// tree has to LOAD, warned, not fail to parse) and the predicate must
    /// catch it — this is the shape the warning exists for.
    #[test]
    fn pre_v1_1_stage_config_reads_as_unwindowed() {
        let cfg: StageConfig = serde_json::from_str(
            r#"{"layer_start":0,"layer_end":6,"has_embed":true,"stateful":true,
                "layer_types":["sliding_attention","full_attention"]}"#,
        )
        .unwrap();
        assert_eq!(cfg.export_version, None);
        assert_eq!(cfg.sliding_window, None);
        assert!(sliding_layers_unwindowed(&cfg));
    }

    /// The shape `gemma4_cached_v1.1` writes for a stage with sliding layers:
    /// window present, so nothing to warn about.
    #[test]
    fn v1_1_stage_config_carries_the_window() {
        let cfg: StageConfig = serde_json::from_str(
            r#"{"layer_start":0,"layer_end":6,"has_embed":true,"stateful":true,
                "layer_types":["sliding_attention","full_attention"],
                "sliding_window":1024,"export_version":"gemma4_cached_v1.1"}"#,
        )
        .unwrap();
        assert_eq!(cfg.sliding_window, Some(1024));
        assert_eq!(cfg.export_version.as_deref(), Some("gemma4_cached_v1.1"));
        assert!(!sliding_layers_unwindowed(&cfg));
    }

    /// An architecture with no window: the exporter writes an explicit
    /// `"sliding_window": null`. That must deserialise (not error) and must not
    /// be read as a stale tree — there are no sliding layers to bound.
    #[test]
    fn full_attention_only_stage_with_null_window_is_not_stale() {
        let cfg: StageConfig = serde_json::from_str(
            r#"{"layer_start":0,"layer_end":6,"stateful":true,
                "layer_types":["full_attention","full_attention"],
                "sliding_window":null,"export_version":"gemma4_cached_v1.1"}"#,
        )
        .unwrap();
        assert_eq!(cfg.sliding_window, None);
        assert!(!sliding_layers_unwindowed(&cfg));
    }

    /// With no `layer_types` key at all there is no evidence of a sliding
    /// layer, so the predicate stays quiet instead of warning on every config
    /// written before the key existed.
    #[test]
    fn stage_config_without_layer_types_is_not_stale() {
        let cfg: StageConfig =
            serde_json::from_str(r#"{"layer_start":0,"layer_end":6,"stateful":true}"#).unwrap();
        assert!(cfg.layer_types.is_empty());
        assert!(!sliding_layers_unwindowed(&cfg));
    }

    #[test]
    fn prefill_spans_without_seed_match_the_old_folding() {
        // chunk = usize::MAX (default single-pass): one span over everything.
        assert_eq!(prefill_span_end(0, usize::MAX, 10), 10);
        // explicit chunking, no seed: plain min(i+chunk, len).
        assert_eq!(prefill_span_end(0, 4, 10), 4);
        assert_eq!(prefill_span_end(4, 4, 10), 8);
        assert_eq!(prefill_span_end(8, 4, 10), 10);
    }

    #[test]
    fn prefill_spans_fold_the_resume_seed_at_t1() {
        // 10 tokens, last 3 are the seed: prompt in one MAX-chunk span, then
        // three T=1 spans — the donor's exact compute pattern.
        let (chunk, split, len) = (usize::MAX, 7, 10);
        let mut i = 0;
        let mut spans = Vec::new();
        while i < len {
            let end = prefill_span_end(i, chunk, split);
            spans.push((i, end));
            i = end;
        }
        assert_eq!(spans, vec![(0, 7), (7, 8), (8, 9), (9, 10)]);
    }

    #[test]
    fn prefill_spans_warm_covered_seed_is_all_t1() {
        // warm_prefix ate into the seed: seed_split saturates to 0 ⇒ every
        // remaining token folds at T=1.
        assert_eq!(prefill_span_end(0, usize::MAX, 0), 1);
        assert_eq!(prefill_span_end(1, usize::MAX, 0), 2);
    }

    /// The EOS path carries the usage the API reads off the final chunk. Every
    /// gemma4 response reported `usage.prompt_tokens: 0` because these three
    /// fields were left None.
    #[test]
    fn final_chunk_on_eos_carries_usage_and_stop() {
        let c = final_chunk("t1".to_string(), 106, "!".to_string(), 93, true);
        assert!(c.is_final);
        assert_eq!(c.finish_reason, Some(FinishReason::Stop));
        assert_eq!(c.prompt_tokens, Some(93));
        assert_eq!(c.n_tokens, Some(1));
        // The wire shape of the chunk is otherwise unchanged.
        assert_eq!(c.token_id, 106);
        assert_eq!(c.text, "!");
    }

    /// A max_tokens stop must not read as a natural stop — the API maps
    /// `finish_reason` straight through, and client retry logic keys on
    /// `"length"`.
    #[test]
    fn final_chunk_on_max_tokens_reports_length() {
        let c = final_chunk("t1".to_string(), 42, "x".to_string(), 7, false);
        assert_eq!(c.finish_reason, Some(FinishReason::Length));
        assert_eq!(c.prompt_tokens, Some(7));
    }

    /// EOS usually detokenizes to "", and the API's fallback counts a non-empty
    /// chunk as 1 token — so the last token would go uncounted unless
    /// `n_tokens` is explicit.
    #[test]
    fn final_chunk_with_an_empty_delta_still_counts_its_token() {
        let c = final_chunk("t1".to_string(), 106, String::new(), 93, true);
        assert_eq!(c.n_tokens, Some(1));
        assert_eq!(c.token_count(), 1);
    }

    // ---- structured output (grammar-masked picks) ----

    /// A control byte JSON never allows: the raw argmax the mask must reject.
    const BAD: u8 = 0x01;
    const OK_ENUM: &str = r#"{"type":"string","enum":["ok"]}"#;

    fn json_spec(body: &str) -> cascadia_types::GrammarSpec {
        cascadia_types::GrammarSpec {
            kind: cascadia_types::GrammarKind::JsonSchema,
            body: body.to_string(),
        }
    }

    fn ok_grammar() -> CompiledGrammar {
        GrammarFactory::single_byte_for_tests()
            .compile(&json_spec(OK_ENUM))
            .unwrap()
    }

    fn seed(s: &str) -> Vec<i64> {
        s.bytes().map(i64::from).collect()
    }

    /// A one-row logits output peaking on `BAD`, with `t` as the runner-up.
    fn bad_then(width: usize, t: u32) -> (Vec<f32>, Vec<usize>) {
        let mut r = vec![0.0f32; width];
        r[BAD as usize] = 10.0;
        r[t as usize] = 5.0;
        (r, vec![1, 1, width])
    }

    #[test]
    fn plain_pick_is_the_unmasked_argmax() {
        let (row, shape) = bad_then(300, b'"' as u32);
        assert_eq!(pick_next_token(&row, &shape, None).unwrap(), BAD as i32);
        assert_eq!(
            pick_next_token(&row, &shape, None).unwrap(),
            argmax_logits(&row, &shape).unwrap()
        );
    }

    #[test]
    fn masked_pick_takes_the_best_allowed_token_of_the_last_row() {
        let mut mask = start_grammar(&ok_grammar(), &[]).unwrap();
        // Two rows: the earlier one is a prompt position and is never read.
        let mut logits = vec![0.0f32; 600];
        logits[b'"' as usize] = 20.0;
        logits[300 + BAD as usize] = 10.0;
        logits[300 + b'"' as usize] = 5.0;
        let shape = [1, 2, 300];
        assert_eq!(
            pick_next_token(&logits, &shape, Some(&mut mask)).unwrap(),
            b'"' as i32
        );
    }

    #[test]
    fn only_the_last_prefill_span_is_masked() {
        let mut mask = start_grammar(&ok_grammar(), &[]).unwrap();
        let mut masked = Vec::new();
        // 7 tokens in spans of 3: [0,3) [3,6) [6,7).
        let tok = fold_prefill(7, 3, 7, Some(&mut mask), |span, m| {
            masked.push((span.clone(), m.is_some()));
            let (row, shape) = bad_then(300, b'"' as u32);
            pick_next_token(&row, &shape, m)
        })
        .unwrap();
        assert_eq!(
            masked,
            vec![(0..3, false), (3..6, false), (6..7, true)],
            "earlier spans' picks are discarded and must not be masked"
        );
        assert_eq!(tok, b'"' as i32);
    }

    #[test]
    fn prefill_spans_fold_the_seed_and_mask_only_the_last() {
        let mut masked = Vec::new();
        // 4 prompt tokens in one span, then a 2-token seed at T=1.
        let mut mask = start_grammar(&ok_grammar(), &[]).unwrap();
        fold_prefill(6, usize::MAX, 4, Some(&mut mask), |span, m| {
            masked.push((span, m.is_some()));
            Ok(0)
        })
        .unwrap();
        assert_eq!(masked, vec![(0..4, false), (4..5, false), (5..6, true)]);
    }

    #[test]
    fn prefill_without_a_grammar_passes_no_mask() {
        let mut seen = Vec::new();
        let tok = fold_prefill(5, 2, 5, None, |span, m| {
            seen.push(m.is_some());
            Ok(span.end as i32)
        })
        .unwrap();
        assert_eq!(seen, vec![false, false, false]);
        assert_eq!(tok, 5);
    }

    #[test]
    fn a_dead_end_is_an_error_not_an_unmasked_pick() {
        let mut mask = start_grammar(&ok_grammar(), &[]).unwrap();
        // A row narrower than any allowed token: nothing in it is allowed.
        let row = vec![10.0f32; 4];
        let err = pick_next_token(&row, &[1, 1, 4], Some(&mut mask)).unwrap_err();
        assert!(matches!(err, EngineError::Backend(_)), "{err:?}");
    }

    #[test]
    fn a_grammar_finish_on_the_last_allowed_token_is_stop() {
        // `"ok` replayed; the closing quote both finishes the JSON and reaches max_tokens.
        let mut mask = start_grammar(&ok_grammar(), &seed("\"ok")).unwrap();
        mask.accept(b'"' as u32).unwrap();
        assert!(mask.finished());
        assert_eq!(
            step_end(4, 4, false, mask.finished()),
            Some(FinishReason::Stop)
        );
        let c = final_chunk("t".into(), b'"' as i64, "\"".into(), 3, true);
        assert_eq!(c.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn step_end_without_a_grammar_is_unchanged() {
        assert_eq!(step_end(3, 4, false, false), None);
        assert_eq!(step_end(4, 4, false, false), Some(FinishReason::Length));
        assert_eq!(step_end(2, 4, true, false), Some(FinishReason::Stop));
        assert_eq!(step_end(4, 4, true, false), Some(FinishReason::Stop));
    }

    #[test]
    fn a_grammar_task_on_the_budget_without_a_finish_is_length() {
        assert_eq!(step_end(4, 4, false, false), Some(FinishReason::Length));
        assert_eq!(step_end(2, 4, false, true), Some(FinishReason::Stop));
    }

    #[test]
    fn a_rejected_token_is_an_error() {
        let mut mask = start_grammar(&ok_grammar(), &[]).unwrap();
        assert!(mask.accept(BAD as u32).is_err());
    }

    fn active_task(grammar: Option<GrammarMask>) -> ActiveTask {
        ActiveTask {
            task: GenerationTask::new("t1", "p"),
            prompt_ids: vec![1, 2],
            generated: Vec::new(),
            last_text: String::new(),
            prefilled: true,
            last_token: 0,
            resume_seed_len: 0,
            warm_prefix: 0,
            started: std::time::Instant::now(),
            t_alpha_compute: std::time::Duration::ZERO,
            t_wire: std::time::Duration::ZERO,
            grammar,
        }
    }

    #[test]
    fn a_decode_step_row_is_masked_and_only_the_last_prefill_span() {
        let mut mask = start_grammar(&ok_grammar(), &[]).unwrap();
        // Decode: one token, always the next generated one.
        assert!(step_mask(Some(&mut mask), false, 1, 1).is_some());
        // Prefill: only the span that ends the prompt.
        assert!(step_mask(Some(&mut mask), true, 3, 7).is_none());
        assert!(step_mask(Some(&mut mask), true, 7, 7).is_some());
        // No grammar: never.
        assert!(step_mask(None, false, 1, 1).is_none());
        assert!(step_mask(None, true, 7, 7).is_none());
    }

    #[test]
    fn a_rejected_token_clears_the_active_task_and_names_it() {
        let mut active = Some(active_task(Some(
            start_grammar(&ok_grammar(), &[]).unwrap(),
        )));
        let err = advance_grammar(&mut active, BAD as i32).unwrap_err();
        assert!(active.is_none(), "a rejected token must end the task");
        assert!(
            matches!(&err, EngineError::Task { task_id, .. } if task_id == "t1"),
            "{err:?}"
        );
        // An allowed token keeps it; a task without a grammar is untouched.
        let mut active = Some(active_task(Some(
            start_grammar(&ok_grammar(), &[]).unwrap(),
        )));
        assert!(!advance_grammar(&mut active, b'"' as i32).unwrap());
        assert!(active.is_some());
        let mut plain = Some(active_task(None));
        assert!(!advance_grammar(&mut plain, BAD as i32).unwrap());
        assert!(plain.is_some());
    }

    #[test]
    fn a_structured_output_failure_does_not_start_a_step_failure_streak() {
        let mut limiter = StepWarnLimiter::default();
        let mut grammar = StepWarnLimiter::default();
        let now = std::time::Instant::now();
        let e = EngineError::Backend("grammar: no token of the logits row is allowed".into());
        report_step_error(
            &mut limiter,
            &mut grammar,
            Some(GrammarFailure::DeadEnd),
            &e,
            now,
        );
        report_step_error(
            &mut limiter,
            &mut grammar,
            Some(GrammarFailure::Rejected),
            &e,
            now,
        );
        // The next real failure is still the first of a streak (logged in full).
        assert!(matches!(limiter.on_failure(now), Some(StepWarn::First)));
        // A real failure does count.
        let mut limiter = StepWarnLimiter::default();
        let mut grammar = StepWarnLimiter::default();
        report_step_error(&mut limiter, &mut grammar, None, &e, now);
        assert!(limiter.on_failure(now).is_none());
        assert!(matches!(grammar.on_failure(now), Some(StepWarn::First)));
    }

    #[test]
    fn structured_output_failure_warnings_are_rate_limited() {
        let mut limiter = StepWarnLimiter::default();
        let mut grammar = StepWarnLimiter::default();
        let now = std::time::Instant::now();
        let e = EngineError::Backend("x".into());
        report_step_error(
            &mut limiter,
            &mut grammar,
            Some(GrammarFailure::DeadEnd),
            &e,
            now,
        );
        // Within the interval a further one is suppressed, not logged.
        assert!(grammar.on_failure(now).is_none());
    }

    #[test]
    fn only_a_mask_that_allows_nothing_is_a_dead_end() {
        let mut mask = start_grammar(&ok_grammar(), &[]).unwrap();
        // A malformed logits output (zero last dim): the stage's own failure.
        let (_, kind) = pick_tagged(&[], &[1, 1, 0], Some(&mut mask)).unwrap_err();
        assert_eq!(kind, None);
        let (_, kind) = pick_tagged(&[1.0; 4], &[1, 1, 8], Some(&mut mask)).unwrap_err();
        assert_eq!(kind, None);
        // Nothing in the row allowed: a dead end.
        let (_, kind) = pick_tagged(&[10.0; 4], &[1, 1, 4], Some(&mut mask)).unwrap_err();
        assert_eq!(kind, Some(GrammarFailure::DeadEnd));
        // Without a mask, nothing is tagged.
        let (_, kind) = pick_tagged(&[], &[1, 1, 0], None).unwrap_err();
        assert_eq!(kind, None);
    }

    #[test]
    fn single_stage_compiles_at_submit_and_reports_bad_schemas_and_missing_factories() {
        let factory = Ok(GrammarFactory::single_byte_for_tests());
        let good = GenerationTask::new("t", "p").with_grammar(json_spec(OK_ENUM));
        assert!(compile_task_grammar(&factory, &good).unwrap().is_some());
        let bad = GenerationTask::new("t", "p").with_grammar(json_spec(r#"{"type":42}"#));
        assert!(matches!(
            compile_task_grammar(&factory, &bad),
            Err(EngineError::InvalidConfig(_))
        ));
        let none: Result<GrammarFactory, String> = Err("no tokenizer".into());
        match compile_task_grammar(&none, &good) {
            Err(EngineError::InvalidConfig(m)) => assert_eq!(
                m,
                "structured output is not available for this model: no tokenizer"
            ),
            other => panic!("{:?}", other.map(|g| g.is_some())),
        }
    }

    // -------- structured output on the stage wire --------

    use cascadia_transport::TransportError;

    fn ok_words() -> Vec<u32> {
        let mut m = start_grammar(&ok_grammar(), &[]).unwrap();
        mask_words(&mut m, GrammarFactory::single_byte_for_tests().mask_width()).unwrap()
    }

    /// A one-row logits output peaking on `BAD`, `"` the runner-up: the masked
    /// pick of the `"ok"` enum's first token is `"`.
    fn tail_row() -> (Vec<f32>, Vec<usize>) {
        bad_then(300, b'"' as u32)
    }

    fn sample_hidden() -> (Vec<f32>, [usize; 3]) {
        ((0..8).map(|i| i as f32 * 0.25 - 1.0).collect(), [1, 2, 4])
    }

    fn sample_cross() -> (Vec<i64>, Vec<WireTensor>) {
        let f = |v: f32| WireTensor::new(WireDType::F32, [1, 1, 2], f32_to_bytes(&[v, v]));
        (vec![10, 11], vec![f(1.0), f(2.0)])
    }

    fn step_bytes(frames: &[&WireTensor]) -> Vec<u8> {
        let mut out = Vec::new();
        // A Vec sink needs no reactor, so this also runs inside async tests.
        for f in frames {
            futures::executor::block_on(cascadia_transport::send_tensor(&mut out, f)).unwrap();
        }
        out
    }

    async fn link() -> (ActivationServer, ActivationClient) {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let mut client = ActivationClient::new("127.0.0.1", server.port());
        let (a, c) = tokio::join!(server.accept(), client.connect());
        a.unwrap();
        c.unwrap();
        (server, client)
    }

    async fn send_sample_step(
        g: &mut ActivationClient,
        position: i64,
        mask: Option<&[u32]>,
    ) -> (Vec<f32>, [usize; 3]) {
        let (hidden, shape) = sample_hidden();
        let (tags, cross) = sample_cross();
        let hid = encode_hidden_frame(&hidden, shape, mask).unwrap();
        send_step(
            g,
            &encode_wire_position(position),
            &hid,
            &encode_cross_kv_header(&tags),
            &cross,
        )
        .await
        .unwrap();
        (hidden, shape)
    }

    /// The relay stage's step read before masks existed, kept verbatim as the
    /// stand-in for an older stage: what it does with a masked step is what a
    /// deployed older build does.
    async fn old_stage_read(
        g: &mut ActivationServer,
        consumes: usize,
    ) -> EngineResult<(Vec<f32>, [usize; 3], i64)> {
        let be = |e: TransportError| EngineError::Backend(e.to_string());
        let pos_t = g.recv().await.map_err(be)?.0;
        let hid_t = g.recv_reply().await.map_err(be)?.0;
        let header_t = g.recv_reply().await.map_err(be)?.0;
        let position = decode_wire_position(&pos_t)?;
        let tags = decode_cross_kv_header(&header_t)?;
        let n = tags.len();
        if n != consumes {
            return Err(EngineError::Backend("cross-KV count mismatch".into()));
        }
        let mut ext = Vec::with_capacity(n);
        for _ in 0..n {
            ext.push(g.recv_reply().await.map_err(be)?.0);
        }
        for (tag, wt) in tags.iter().zip(ext.iter()) {
            if wt.dtype != WireDType::F32 {
                return Err(EngineError::Backend(format!(
                    "cross-KV frame (tag {tag}) has dtype {:?}, expected F32",
                    wt.dtype
                )));
            }
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

    /// One relay-stage step read the way the engine reads it.
    async fn new_stage_read(g: &mut ActivationServer, consumes: usize) -> (HiddenIn, i64) {
        let pos_t = g.recv().await.unwrap().0;
        let (hid, _tags, _frames) = recv_step_frames(g, consumes).await.unwrap();
        (
            decode_hidden_frame(&hid).unwrap(),
            decode_wire_position(&pos_t).unwrap(),
        )
    }

    /// The tail's masked step (`step_last`, the model replaced by `tail_row`).
    async fn new_tail_step(up: &mut ActivationServer) {
        let (hidden_in, _) = new_stage_read(up, 2).await;
        let reply = match hidden_in {
            HiddenIn::Masked { words, .. } => {
                let (row, shape) = tail_row();
                let (reply, err) = tail_masked_reply(Ok((row, shape)), &words);
                assert!(err.is_none());
                reply
            }
            HiddenIn::BadMask(_) => StageReply::Nack,
            other => panic!("expected a masked step, got {other:?}"),
        };
        up.send(&encode_masked_reply(reply)).await.unwrap();
    }

    /// The middle's masked step (`step_middle`, the model an identity).
    async fn new_middle_step(
        up: &mut ActivationServer,
        down: &mut ActivationClient,
        budget: Duration,
    ) {
        let pos_t = up.recv().await.unwrap().0;
        let (hid, tags, frames) = recv_step_frames(up, 2).await.unwrap();
        let HiddenIn::Masked {
            floats,
            shape,
            words,
        } = decode_hidden_frame(&hid).unwrap()
        else {
            panic!("expected a masked step")
        };
        let prepared = encode_hidden_frame(&floats, shape, Some(&words)).map(|hid| StepFrames {
            pos: pos_t,
            hid,
            header: encode_cross_kv_header(&tags),
            cross: frames,
            mask: Some(words),
        });
        let (reply, err, _) =
            middle_masked_step(down, prepared, budget, &mut MaskSendCache::default()).await;
        up.send(&encode_masked_reply(reply)).await.unwrap();
        assert!(err.is_none());
    }

    #[test]
    fn plain_step_frames_are_byte_identical_to_the_base_encoding() {
        // Bytes of the frames before masks existed: 20-byte big-endian header
        // (payload len, dtype, shape) then the little-endian payload.
        let hid = encode_hidden_frame(&[1.0, -2.0], [1, 1, 2], None).unwrap();
        assert_eq!(
            step_bytes(&[&hid]),
            [
                0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, //
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0,
            ]
        );
        assert_eq!(
            step_bytes(&[&encode_token_reply(7)]),
            [0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 7, 0, 0, 0]
        );
        assert_eq!(
            step_bytes(&[&encode_wire_position(5)]),
            [0, 0, 0, 8, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0]
        );
        // The plain hidden frame is exactly the old construction.
        let (hidden, shape) = sample_hidden();
        assert_eq!(
            encode_hidden_frame(&hidden, shape, None).unwrap(),
            WireTensor::new(WireDType::F32, [1, 2, 4], f32_to_bytes(&hidden))
        );
        // A plain reply decodes as the first four bytes, as before.
        assert_eq!(
            decode_stage_reply(&encode_token_reply(7)).unwrap(),
            StageReply::Plain(7)
        );
        assert!(
            decode_stage_reply(&WireTensor::new(WireDType::I8, [1, 1, 2], vec![1, 2])).is_err()
        );
    }

    #[tokio::test]
    async fn a_plain_step_on_the_wire_is_the_base_byte_stream() {
        let (mut server, mut client) = link().await;
        let (hidden, shape) = sample_hidden();
        let (tags, cross) = sample_cross();
        let pos = encode_wire_position(3);
        let hid = WireTensor::new(WireDType::F32, [1, 2, 4], f32_to_bytes(&hidden));
        let header = encode_cross_kv_header(&tags);
        let expected = step_bytes(&[&pos, &hid, &header, &cross[0], &cross[1]]);
        send_sample_step(&mut client, 3, None).await;
        let got = server.recv_raw(expected.len()).await.unwrap();
        assert_eq!(got, expected);
        let _ = shape;
    }

    #[test]
    fn a_masked_hidden_frame_round_trips_for_any_hidden_width() {
        let words = ok_words();
        assert_eq!(words.len(), 262usize.div_ceil(32));
        for h in [1usize, 2, 3, 4, 9, 10, 300] {
            let n = 3;
            let hidden: Vec<f32> = (0..n * h).map(|i| i as f32 * 0.5 - 7.25).collect();
            let t = encode_hidden_frame(&hidden, [1, n, h], Some(&words)).unwrap();
            let r = (words.len() + 1).div_ceil(h);
            assert_eq!(t.dtype, WireDType::I32);
            assert_eq!(t.shape, [1, (n + r) as u32, h as u32], "h={h}");
            assert_eq!(
                decode_hidden_frame(&t).unwrap(),
                HiddenIn::Masked {
                    floats: hidden.clone(),
                    shape: [1, n, h],
                    words: words.clone()
                },
                "h={h}"
            );
        }
    }

    #[test]
    fn the_head_mask_is_the_grammar_mask_bit_for_bit() {
        let width = GrammarFactory::single_byte_for_tests().mask_width();
        let words = ok_words();
        let mut m = start_grammar(&ok_grammar(), &[]).unwrap();
        for tok in 0..width {
            let bit = words[tok / 32] >> (tok % 32) & 1 == 1;
            assert_eq!(bit, m.allowed(tok as u32), "token {tok}");
        }
        // A grammar that allows nothing is a dead end at the head, before any frame.
        let mut m = start_grammar(&ok_grammar(), &[]).unwrap();
        let _ = m.accept(BAD as u32);
        assert!(mask_words(&mut m, width).is_err());
    }

    /// The head's encoder reads the allowed set from the matcher, not from
    /// `apply` on a zero row; the words are unchanged.
    #[test]
    fn the_head_mask_equals_the_apply_readback() {
        let width = GrammarFactory::single_byte_for_tests().mask_width();
        for done in ["", "\"", "\"o", "\"ok\""] {
            let mut m = start_grammar(&ok_grammar(), &seed(done)).unwrap();
            let want = crate::structured::words_via_apply(&mut m, width).unwrap();
            assert_eq!(mask_words(&mut m, width).unwrap(), want, "after {done:?}");
        }
    }

    /// An all `-inf` (or NaN) row whose last index is the disallowed EOS:
    /// the head's and the tail's picks are the lowest allowed token.
    #[test]
    fn an_all_neg_inf_row_never_picks_a_disallowed_eos() {
        let width = GrammarFactory::single_byte_for_tests().mask_width();
        let eos = width as u32 - 1;
        let shape = [1, 1, width];
        let mut m = start_grammar(&ok_grammar(), &[]).unwrap();
        assert!(!m.allowed(eos));
        let words = mask_words(&mut m, width).unwrap();
        for fill in [f32::NEG_INFINITY, f32::NAN] {
            let row = vec![fill; width];
            assert_eq!(
                pick_next_token(&row, &shape, Some(&mut m)).unwrap(),
                b'"' as i32
            );
            assert_eq!(tail_masked_pick(&row, &shape, &words).unwrap(), b'"' as i32);
        }
    }

    #[test]
    fn the_tail_masks_its_row_and_nacks_what_it_cannot_serve() {
        let words = ok_words();
        let (row, shape) = tail_row();
        assert_eq!(argmax_logits(&row, &shape).unwrap(), BAD as i32);
        assert_eq!(tail_masked_pick(&row, &shape, &words).unwrap(), b'"' as i32);
        // Words wider than the tail's logits row: tokens past the row are
        // simply not allowed, as on the other engines' tails.
        let narrow = vec![1.0f32; 64];
        assert_eq!(
            tail_masked_pick(&narrow, &[1, 1, 64], &words).unwrap(),
            b'"' as i32
        );
        let err = tail_masked_pick(&narrow[..16], &[1, 1, 16], &words).unwrap_err();
        assert!(err.contains("allows no token"), "{err}");
        // Nothing allowed in the row.
        assert!(tail_masked_pick(&row, &shape, &[0; 9]).is_err());
        // A padded row (wider than the mask): the excess is masked.
        let mut padded = vec![0.0f32; 400];
        padded[350] = 99.0;
        padded[b'"' as usize] = 1.0;
        assert_eq!(
            tail_masked_pick(&padded, &[1, 1, 400], &words).unwrap(),
            b'"' as i32
        );
    }

    #[test]
    fn a_malformed_masked_frame_is_nacked_not_read_as_hidden() {
        let words = ok_words();
        let (hidden, shape) = sample_hidden();
        let good = encode_hidden_frame(&hidden, shape, Some(&words)).unwrap();
        let mut zero_w = good.clone();
        let len = zero_w.data.len();
        zero_w.data[len - 4..].copy_from_slice(&0i32.to_le_bytes());
        let mut huge_w = good.clone();
        huge_w.data[len - 4..].copy_from_slice(&10_000i32.to_le_bytes());
        let mut dirty_pad = good.clone();
        dirty_pad.data[len - 8..len - 4].copy_from_slice(&1i32.to_le_bytes());
        for t in [zero_w, huge_w, dirty_pad] {
            assert!(
                matches!(decode_hidden_frame(&t).unwrap(), HiddenIn::BadMask(_)),
                "{:?}",
                t.shape
            );
        }
    }

    #[test]
    fn masked_replies_round_trip() {
        for r in [
            StageReply::Token(42),
            StageReply::Nack,
            StageReply::NotAcked,
        ] {
            let t = encode_masked_reply(r);
            assert_eq!((t.dtype, t.shape), (WireDType::I32, [1, 1, 2]));
            assert_eq!(decode_stage_reply(&t).unwrap(), r);
        }
        let ack = encode_masked_reply(StageReply::Token(42));
        assert_eq!(&ack.data[4..], &MASK_ACK.to_le_bytes());
        // An unknown second lane is an error, never a token.
        let odd = WireTensor::new(WireDType::I32, [1, 1, 2], i64_to_bytes(&[7]));
        assert!(decode_stage_reply(&odd).is_err());
    }

    #[test]
    fn an_unacknowledged_masked_step_fails_with_the_upgrade_error_and_latches() {
        let t0 = std::time::Instant::now();
        let sec = Duration::from_secs(1);
        let mut latch = MaskLatch::default();
        assert!(latch.refusal(t0).is_ok());
        let (e, kind) =
            head_masked_token(MaskedWait::Unacked("no reply".into()), &mut latch, t0).unwrap_err();
        assert!(
            matches!(&e, EngineError::Backend(m) if m.contains(UPGRADE_EVERY_STAGE)),
            "{e}"
        );
        assert_eq!(kind, Some(GrammarFailure::Upgrade));
        // Later structured-output tasks fail at once with the upgrade message.
        match latch.refusal(t0 + sec) {
            Err(EngineError::Backend(m)) => assert_eq!(m, UPGRADE_EVERY_STAGE),
            other => panic!("{other:?}"),
        }
        assert!(latch.refusal(t0 + MASK_LATCH_REPROBE - sec).is_err());
        // Re-probed after the latch period.
        assert!(latch.refusal(t0 + MASK_LATCH_REPROBE).is_ok());
        assert!(latch.refusal(t0 + MASK_LATCH_REPROBE + sec).is_ok());
        // A late acknowledged reply clears it.
        latch.trip(t0);
        latch.clear();
        assert!(latch.refusal(t0 + sec).is_ok());
        // An acknowledged NACK is a failed step, not an old stage: no latch.
        let mut latch = MaskLatch::default();
        let (_, kind) = head_masked_token(MaskedWait::Nack, &mut latch, t0).unwrap_err();
        assert_eq!(kind, Some(GrammarFailure::Nack));
        assert!(latch.refusal(t0 + sec).is_ok());
        // A link failure is the step's own error, logged as one.
        let (_, kind) = head_masked_token(
            MaskedWait::Failed(EngineError::Backend("reset".into())),
            &mut latch,
            t0,
        )
        .unwrap_err();
        assert_eq!(kind, None);
        assert!(latch.refusal(t0 + sec).is_ok());
        assert_eq!(
            head_masked_token(MaskedWait::Token(9), &mut latch, t0).unwrap(),
            9
        );
    }

    #[tokio::test]
    async fn a_masked_wait_needs_the_acknowledgement() {
        let (mut server, mut client) = link().await;
        let budget = Duration::from_secs(5);
        server.send(&encode_token_reply(BAD as i32)).await.unwrap();
        assert!(matches!(
            wait_masked_reply(&mut client, budget).await,
            MaskedWait::Unacked(_)
        ));
        server
            .send(&encode_masked_reply(StageReply::NotAcked))
            .await
            .unwrap();
        assert!(matches!(
            wait_masked_reply(&mut client, budget).await,
            MaskedWait::Unacked(_)
        ));
        server
            .send(&encode_masked_reply(StageReply::Nack))
            .await
            .unwrap();
        assert!(matches!(
            wait_masked_reply(&mut client, budget).await,
            MaskedWait::Nack
        ));
        server
            .send(&encode_masked_reply(StageReply::Token(34)))
            .await
            .unwrap();
        assert!(matches!(
            wait_masked_reply(&mut client, budget).await,
            MaskedWait::Token(34)
        ));
    }

    #[tokio::test]
    async fn a_masked_step_round_trips_through_a_new_tail() {
        let (mut server, mut client) = link().await;
        let words = ok_words();
        let tail = tokio::spawn(async move {
            new_tail_step(&mut server).await;
            server
        });
        send_sample_step(&mut client, 0, Some(&words)).await;
        let w = wait_masked_reply(&mut client, Duration::from_secs(5)).await;
        assert!(
            matches!(w, MaskedWait::Token(t) if t == b'"' as i32),
            "{w:?}"
        );
        // The tail decoded the hidden rows intact.
        let mut server = tail.await.unwrap();
        send_sample_step(&mut client, 9, Some(&words)).await;
        let (hidden_in, pos) = new_stage_read(&mut server, 2).await;
        let (hidden, shape) = sample_hidden();
        assert_eq!(pos, 9);
        assert_eq!(
            hidden_in,
            HiddenIn::Masked {
                floats: hidden,
                shape,
                words
            }
        );
    }

    #[tokio::test]
    async fn an_old_stage_reads_every_frame_of_a_masked_step_then_fails() {
        let (mut server, mut client) = link().await;
        let words = ok_words();
        send_sample_step(&mut client, 4, Some(&words)).await;
        send_sample_step(&mut client, 6, None).await;
        let err = old_stage_read(&mut server, 2).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("unexpected upstream hidden dtype I32"),
            "{err}"
        );
        // Still aligned: the next plain step decodes.
        let (floats, shape, pos) = old_stage_read(&mut server, 2).await.unwrap();
        let (hidden, hshape) = sample_hidden();
        assert_eq!((floats, shape, pos), (hidden, hshape, 6));
    }

    #[tokio::test]
    async fn a_masked_step_times_out_on_an_old_tail_and_the_link_survives() {
        let (mut server, mut client) = link().await;
        let words = ok_words();
        send_sample_step(&mut client, 0, Some(&words)).await;
        // The old tail consumes the step and answers nothing.
        assert!(old_stage_read(&mut server, 2).await.is_err());
        let w = wait_masked_reply(&mut client, Duration::from_millis(200)).await;
        let mut latch = MaskLatch::default();
        let now = std::time::Instant::now();
        assert!(matches!(&w, MaskedWait::Unacked(_)), "{w:?}");
        let (e, _) = head_masked_token(w, &mut latch, now).unwrap_err();
        assert!(e.to_string().contains(UPGRADE_EVERY_STAGE));
        assert!(latch.refusal(now).is_err());
        // The socket survived the timeout: a plain request works after it.
        send_sample_step(&mut client, 0, None).await;
        let (_, _, pos) = old_stage_read(&mut server, 2).await.unwrap();
        assert_eq!(pos, 0);
        server.send(&encode_token_reply(17)).await.unwrap();
        assert_eq!(
            wait_plain_reply(&mut client, false).await.unwrap(),
            (17, false)
        );
    }

    /// Drains of an empty link find nothing and don't wait: the fastest of
    /// 20 is far under a timer tick (a 1 ms wait measured about 2 ms each).
    /// The fastest, so a busy test machine's preemption doesn't count.
    async fn assert_empty_drains_dont_wait(client: &mut ActivationClient) {
        let mut fastest = Duration::MAX;
        for _ in 0..20 {
            let t = std::time::Instant::now();
            assert!(!drain_late_replies(client).await.unwrap());
            fastest = fastest.min(t.elapsed());
        }
        assert!(fastest < Duration::from_micros(500), "{fastest:?}");
    }

    /// The drain before a masked step doesn't wait: a reply that landed
    /// while the stage computed is drained, and an empty link returns at once.
    /// The thread sleeps instead of awaiting, so the I/O driver doesn't run
    /// between send and drain (on the current-thread flavor only the drain's
    /// own yield runs it).
    async fn drain_sees_a_reply_sent_just_before() {
        let (mut server, mut client) = link().await;
        server
            .send(&encode_masked_reply(StageReply::Token(34)))
            .await
            .unwrap();
        std::thread::sleep(Duration::from_millis(2));
        assert!(drain_late_replies(&mut client).await.unwrap());
        assert_empty_drains_dont_wait(&mut client).await;
        server.send(&encode_token_reply(19)).await.unwrap();
        assert_eq!(
            wait_plain_reply(&mut client, false).await.unwrap(),
            (19, false)
        );
    }

    #[tokio::test]
    async fn the_drain_is_immediate_and_sees_a_reply_already_sent() {
        drain_sees_a_reply_sent_just_before().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_drain_is_immediate_and_sees_a_reply_already_sent_multi_thread() {
        drain_sees_a_reply_sent_just_before().await;
    }

    /// The engine's own context: `block_on` from a blocking thread, where the
    /// yield can't run the driver and the runtime's workers record the
    /// bytes. A late reply lands while the stage still computes (here 20 ms
    /// before the drain, enough for the driver to record it on a busy
    /// machine); the drain sees it and an empty link costs no wait.
    #[test]
    fn the_drain_sees_a_reply_already_sent_from_a_blocking_thread() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (mut server, mut client) = rt.block_on(link());
        let handle = rt.handle().clone();
        std::thread::spawn(move || {
            for _ in 0..50 {
                handle.block_on(async {
                    server
                        .send(&encode_masked_reply(StageReply::Nack))
                        .await
                        .unwrap();
                });
                std::thread::sleep(Duration::from_millis(20));
                handle.block_on(async {
                    assert!(drain_late_replies(&mut client).await.unwrap());
                    assert_empty_drains_dont_wait(&mut client).await;
                });
            }
        })
        .join()
        .unwrap();
    }

    /// A middle that answered a masked step unacknowledged, and then sees an
    /// acknowledged reply from below (a slow but capable tail), passes an
    /// acknowledged NACK up before its next plain reply, so the head's latch
    /// clears. Never before a masked reply: those carry no sequence number,
    /// so the head would read it as the current step's answer.
    #[tokio::test]
    async fn a_middle_passes_a_late_acknowledgement_up_before_its_next_plain_reply() {
        let mut owed = OwedLateAck::default();
        // Nothing owed: today's frames.
        assert_eq!(
            upstream_plain_frames(&mut owed, encode_token_reply(5)).len(),
            1
        );
        owed.on_masked_reply_sent(StageReply::NotAcked);
        // Owed but nothing below has acknowledged yet: nothing extra.
        assert_eq!(
            upstream_plain_frames(&mut owed, encode_token_reply(5)).len(),
            1
        );
        owed.on_ack_from_below();
        let frames = upstream_plain_frames(&mut owed, encode_token_reply(6));
        assert_eq!(
            frames,
            vec![encode_masked_reply(StageReply::Nack), encode_token_reply(6)]
        );
        // Paid once.
        assert_eq!(
            upstream_plain_frames(&mut owed, encode_token_reply(7)).len(),
            1
        );
        // An acknowledged masked reply settles what was owed.
        owed.on_masked_reply_sent(StageReply::NotAcked);
        owed.on_masked_reply_sent(StageReply::Token(3));
        owed.on_ack_from_below();
        assert_eq!(
            upstream_plain_frames(&mut owed, encode_token_reply(8)).len(),
            1
        );

        // The head reads the pair: the token, and the latch-clearing ack.
        let (mut server, mut client) = link().await;
        for f in &frames {
            server.send(f).await.unwrap();
        }
        assert_eq!(
            wait_plain_reply(&mut client, false).await.unwrap(),
            (6, true)
        );
    }

    #[tokio::test]
    async fn a_late_acknowledged_reply_is_discarded_as_stale() {
        let (mut server, mut client) = link().await;
        // A slow new tail: the masked wait expires, then its reply lands.
        let w = wait_masked_reply(&mut client, Duration::from_millis(100)).await;
        assert!(matches!(w, MaskedWait::Unacked(_)));
        server
            .send(&encode_masked_reply(StageReply::Token(34)))
            .await
            .unwrap();
        server.send(&encode_token_reply(17)).await.unwrap();
        // The next plain wait skips it and reports the acknowledgement.
        assert_eq!(
            wait_plain_reply(&mut client, false).await.unwrap(),
            (17, true)
        );
        // A middle's not-acknowledged reply is skipped without clearing anything.
        server
            .send(&encode_masked_reply(StageReply::NotAcked))
            .await
            .unwrap();
        server.send(&encode_token_reply(18)).await.unwrap();
        assert_eq!(
            wait_plain_reply(&mut client, true).await.unwrap(),
            (18, false)
        );
        // Before a masked step, replies that already arrived are drained.
        server
            .send(&encode_masked_reply(StageReply::Nack))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(drain_late_replies(&mut client).await.unwrap());
        assert!(!drain_late_replies(&mut client).await.unwrap());
        server.send(&encode_token_reply(19)).await.unwrap();
        assert_eq!(
            wait_plain_reply(&mut client, false).await.unwrap(),
            (19, false)
        );
    }

    #[tokio::test]
    async fn a_three_stage_chain_relays_the_mask_and_the_acknowledgement() {
        let (mut mid_up, mut head) = link().await;
        let (mut tail_up, mut mid_down) = link().await;
        let words = ok_words();
        let tail = tokio::spawn(async move { new_tail_step(&mut tail_up).await });
        let mid = tokio::spawn(async move {
            new_middle_step(&mut mid_up, &mut mid_down, Duration::from_secs(5)).await;
        });
        send_sample_step(&mut head, 0, Some(&words)).await;
        let w = wait_masked_reply(&mut head, Duration::from_secs(10)).await;
        assert!(
            matches!(w, MaskedWait::Token(t) if t == b'"' as i32),
            "{w:?}"
        );
        tail.await.unwrap();
        mid.await.unwrap();
    }

    #[test]
    fn each_stage_waits_longer_than_the_stage_after_it() {
        let _knob = crate::timeout_knob_lock();
        for prefill in [false, true] {
            // A 2-stage head waits exactly the plain step's budget.
            let plain = if prefill {
                cascadia_transport::recv_timeout()
                    .saturating_mul(cascadia_transport::PREFILL_REPLY_TIMEOUT_FACTOR)
                    .min(G_PREFILL_REPLY_CEILING)
            } else {
                cascadia_transport::recv_timeout()
            };
            assert_eq!(masked_wait_budget(prefill, 1), plain);
            for k in 1..6 {
                assert!(
                    masked_wait_budget(prefill, k + 1)
                        >= masked_wait_budget(prefill, k) + Duration::from_secs(5),
                    "prefill={prefill} k={k}"
                );
            }
        }
    }

    /// With the engine's own budgets (base 1 s), a middle in front of an old
    /// tail times out first and its not-acknowledged reply reaches the head
    /// inside the head's wait, so no late reply is left on the link.
    #[test]
    fn a_middle_whose_downstream_is_old_answers_not_acknowledged_in_time() {
        let _knob = crate::timeout_knob_lock();
        /// Puts the process-wide knob back even when an assertion fails.
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                cascadia_transport::set_activation_timeout_secs(0);
            }
        }
        cascadia_transport::set_activation_timeout_secs(1);
        let _reset = Reset;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let (mut mid_up, mut head) = link().await;
            let (mut tail_up, mut mid_down) = link().await;
            let words = ok_words();
            let tail = tokio::spawn(async move {
                assert!(old_stage_read(&mut tail_up, 2).await.is_err());
                tail_up
            });
            let mid = tokio::spawn(async move {
                new_middle_step(&mut mid_up, &mut mid_down, masked_wait_budget(false, 1)).await;
                (mid_up, mid_down)
            });
            send_sample_step(&mut head, 0, Some(&words)).await;
            let w = wait_masked_reply(&mut head, masked_wait_budget(false, 2)).await;
            assert!(
                matches!(&w, MaskedWait::Unacked(m) if m.contains("relayed")),
                "the head must get the middle's answer, not time out: {w:?}"
            );
            let mut latch = MaskLatch::default();
            let (e, _) = head_masked_token(w, &mut latch, std::time::Instant::now()).unwrap_err();
            assert!(e.to_string().contains(UPGRADE_EVERY_STAGE));
            let _links = mid.await.unwrap();
            let _tail = tail.await.unwrap();
            assert!(!drain_late_replies(&mut head).await.unwrap());
        });
    }

    #[test]
    fn a_middle_relays_the_acknowledgement_only_when_its_downstream_acked() {
        assert_eq!(middle_reply(MaskedWait::Token(5)).0, StageReply::Token(5));
        assert_eq!(middle_reply(MaskedWait::Nack).0, StageReply::Nack);
        assert_eq!(
            middle_reply(MaskedWait::Unacked("x".into())).0,
            StageReply::NotAcked
        );
        let (r, e) = middle_reply(MaskedWait::Failed(EngineError::Backend("x".into())));
        assert_eq!(r, StageReply::Nack);
        assert!(e.is_some());
    }

    #[test]
    fn multi_stage_pipelines_compile_grammar_tasks_at_submit() {
        let factory = Ok(GrammarFactory::single_byte_for_tests());
        let task = GenerationTask::new("t", "p").with_grammar(json_spec(OK_ENUM));
        assert!(compile_task_grammar(&factory, &task).unwrap().is_some());
        let plain = GenerationTask::new("t", "p");
        assert!(compile_task_grammar(&factory, &plain).unwrap().is_none());
    }

    #[test]
    fn a_latched_pipeline_refuses_structured_output_tasks_before_any_exchange() {
        let now = std::time::Instant::now();
        let mut latch = MaskLatch::default();
        assert!(latched_refusal(&mut latch, true, true, now).is_ok());
        latch.trip(now);
        match latched_refusal(&mut latch, true, true, now) {
            Err(EngineError::Backend(m)) => assert_eq!(m, UPGRADE_EVERY_STAGE),
            other => panic!("{other:?}"),
        }
        // Plain tasks and single-stage pipelines are never refused.
        assert!(latched_refusal(&mut latch, true, false, now).is_ok());
        assert!(latched_refusal(&mut latch, false, true, now).is_ok());
    }

    /// A late two-lane reply still queued when a control exchange starts must
    /// not be read as its ack: that would shift every later reply by one, and
    /// every plain step would return the previous step's token.
    #[cfg(feature = "kv_coord")]
    #[tokio::test]
    async fn a_late_masked_reply_is_not_read_as_a_restore_ack() {
        let (mut server, mut client) = link().await;
        server
            .send(&encode_masked_reply(StageReply::NotAcked))
            .await
            .unwrap();
        server
            .send(&encode_masked_reply(StageReply::Token(34)))
            .await
            .unwrap();
        let peer = tokio::spawn(async move {
            let (t, _) = server.recv().await.unwrap();
            assert_eq!(t, control_frame(G_OPCODE_RESTORE, &7u64.to_le_bytes()));
            server
                .send(&control_frame(G_OPCODE_RESTORE_ACK, &[1]))
                .await
                .unwrap();
            let _ = server.recv().await.unwrap();
            server.send(&encode_token_reply(17)).await.unwrap();
        });
        let restore = control_frame(G_OPCODE_RESTORE, &7u64.to_le_bytes());
        let (ack, late_ack) = control_exchange(&mut client, &restore).await.unwrap();
        assert_eq!(ack, control_frame(G_OPCODE_RESTORE_ACK, &[1]));
        assert!(late_ack, "the acknowledged late reply clears the latch");
        client.send(&encode_wire_position(0)).await.unwrap();
        assert_eq!(
            wait_plain_reply(&mut client, false).await.unwrap(),
            (17, false)
        );
        peer.await.unwrap();
    }

    /// Same through a middle: its RESTORE relay to the tail skips the late
    /// reply, so its ack upstream and the next plain token line up.
    #[cfg(feature = "kv_coord")]
    #[tokio::test]
    async fn a_middle_restore_relay_skips_a_late_masked_reply() {
        let (mut mid_up, mut head) = link().await;
        let (mut tail_up, mut mid_down) = link().await;
        tail_up
            .send(&encode_masked_reply(StageReply::NotAcked))
            .await
            .unwrap();
        let tail = tokio::spawn(async move {
            let (t, _) = tail_up.recv().await.unwrap();
            assert_eq!(t.data[0], G_OPCODE_RESTORE);
            tail_up
                .send(&control_frame(G_OPCODE_RESTORE_ACK, &[1]))
                .await
                .unwrap();
            let _ = tail_up.recv().await.unwrap();
            tail_up.send(&encode_token_reply(21)).await.unwrap();
        });
        let mid = tokio::spawn(async move {
            // handle_inbound_control's RESTORE: relay down, ack up.
            let (t, _) = mid_up.recv().await.unwrap();
            let (ack, _) = control_exchange(&mut mid_down, &t).await.unwrap();
            let ok = ack.data.first() == Some(&G_OPCODE_RESTORE_ACK) && ack.data.get(1) == Some(&1);
            mid_up
                .send(&control_frame(G_OPCODE_RESTORE_ACK, &[u8::from(ok)]))
                .await
                .unwrap();
            // The next plain step.
            let t = mid_up.recv().await.unwrap().0;
            mid_down.send(&t).await.unwrap();
            let (token, _) = wait_plain_reply(&mut mid_down, false).await.unwrap();
            mid_up.send(&encode_token_reply(token)).await.unwrap();
        });
        let restore = control_frame(G_OPCODE_RESTORE, &9u64.to_le_bytes());
        let (ack, _) = control_exchange(&mut head, &restore).await.unwrap();
        assert_eq!(ack, control_frame(G_OPCODE_RESTORE_ACK, &[1]));
        head.send(&encode_wire_position(0)).await.unwrap();
        assert_eq!(
            wait_plain_reply(&mut head, false).await.unwrap(),
            (21, false)
        );
        mid.await.unwrap();
        tail.await.unwrap();
    }

    #[test]
    fn a_tail_whose_relay_fails_still_answers_with_an_acknowledged_nack() {
        let words = ok_words();
        let (reply, err) =
            tail_masked_reply(Err(EngineError::Backend("infer failed".into())), &words);
        assert_eq!(reply, StageReply::Nack);
        assert!(err.is_some(), "the tail's step still fails");
        // A mask that allows nothing in its row: NACK, but the step itself is fine.
        let (reply, err) = tail_masked_reply(Ok((vec![1.0; 16], vec![1, 1, 16])), &words);
        assert_eq!(reply, StageReply::Nack);
        assert!(err.is_none());
        let (row, shape) = tail_row();
        assert_eq!(
            tail_masked_reply(Ok((row, shape)), &words).0,
            StageReply::Token(b'"' as i32)
        );
    }

    #[tokio::test]
    async fn a_middle_whose_relay_fails_answers_nack_and_sends_nothing_down() {
        let (mut tail_up, mut down) = link().await;
        let (reply, err, _) = middle_masked_step(
            &mut down,
            Err(EngineError::Backend("infer failed".into())),
            Duration::from_secs(5),
            &mut MaskSendCache::default(),
        )
        .await;
        assert_eq!(reply, StageReply::Nack);
        assert!(err.is_some());
        // Nothing reached the tail.
        let nothing = tokio::time::timeout(Duration::from_millis(100), tail_up.recv()).await;
        assert!(nothing.is_err(), "{nothing:?}");
    }

    #[tokio::test]
    async fn a_middle_whose_downstream_link_fails_answers_nack() {
        let (tail_up, mut down) = link().await;
        drop(tail_up);
        let words = ok_words();
        let (hidden, shape) = sample_hidden();
        let (tags, cross) = sample_cross();
        let frames = StepFrames {
            pos: encode_wire_position(0),
            hid: encode_hidden_frame(&hidden, shape, Some(&words)).unwrap(),
            header: encode_cross_kv_header(&tags),
            cross,
            mask: Some(words.clone()),
        };
        let (reply, err, _) = middle_masked_step(
            &mut down,
            Ok(frames),
            Duration::from_secs(5),
            &mut MaskSendCache::default(),
        )
        .await;
        assert_eq!(reply, StageReply::Nack);
        assert!(err.is_some());
    }

    /// A middle's masked round drains a late acknowledged reply from below,
    /// then its own step goes unanswered. The drained acknowledgement is not
    /// dropped: the middle passes it up before its next plain reply.
    #[tokio::test]
    async fn a_middle_keeps_the_late_acknowledgement_its_masked_round_drained() {
        let (mut server, mut down) = link().await;
        server
            .send(&encode_masked_reply(StageReply::Token(5)))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let words = ok_words();
        let (hidden, shape) = sample_hidden();
        let (tags, cross) = sample_cross();
        let frames = StepFrames {
            pos: encode_wire_position(0),
            hid: encode_hidden_frame(&hidden, shape, Some(&words)).unwrap(),
            header: encode_cross_kv_header(&tags),
            cross,
            mask: Some(words.clone()),
        };
        // The tail reads nothing and never answers: the wait times out.
        let (reply, err, late_ack) = middle_masked_step(
            &mut down,
            Ok(frames),
            Duration::from_millis(100),
            &mut MaskSendCache::default(),
        )
        .await;
        assert_eq!(reply, StageReply::NotAcked);
        assert!(err.is_none());
        assert!(late_ack, "the drained reply was acknowledged");

        let mut owed = OwedLateAck::default();
        owed.after_masked_reply(reply, late_ack);
        assert_eq!(owed, OwedLateAck::Proven);
        let frames = upstream_plain_frames(&mut owed, encode_token_reply(7));
        assert_eq!(frames.len(), 2, "the acknowledged NACK goes up first");
        assert_eq!(decode_stage_reply(&frames[0]).unwrap(), StageReply::Nack);

        // Without a drained acknowledgement the step is only owed.
        let mut owed = OwedLateAck::default();
        owed.after_masked_reply(StageReply::NotAcked, false);
        assert_eq!(owed, OwedLateAck::Owed);
        // An acknowledged reply already told the head.
        let mut owed = OwedLateAck::Owed;
        owed.after_masked_reply(StageReply::Token(3), true);
        assert_eq!(owed, OwedLateAck::Nothing);
    }

    #[tokio::test]
    async fn a_masked_round_drains_then_sends_then_waits() {
        let (mut server, mut client) = link().await;
        // A late acknowledged reply already queued.
        server
            .send(&encode_masked_reply(StageReply::Token(5)))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let words = ok_words();
        let tail = tokio::spawn(async move {
            new_tail_step(&mut server).await;
        });
        let (hidden, shape) = sample_hidden();
        let (tags, cross) = sample_cross();
        let frames = StepFrames {
            pos: encode_wire_position(0),
            hid: encode_hidden_frame(&hidden, shape, Some(&words)).unwrap(),
            header: encode_cross_kv_header(&tags),
            cross,
            mask: Some(words.clone()),
        };
        let mut frames = frames;
        let (w, late_ack) = masked_round(
            &mut client,
            &mut frames,
            Duration::from_secs(5),
            &mut MaskSendCache::default(),
        )
        .await;
        assert!(late_ack);
        assert!(
            matches!(w, MaskedWait::Token(t) if t == b'"' as i32),
            "{w:?}"
        );
        tail.await.unwrap();
    }

    // ---- repeated masks on the stage wire ----

    /// A different mask of the same width as [`ok_words`].
    fn other_words() -> Vec<u32> {
        vec![u32::MAX; ok_words().len()]
    }

    /// The sample step's frames, carrying `mask`.
    fn sample_frames(mask: Option<&[u32]>) -> StepFrames {
        let (hidden, shape) = sample_hidden();
        let (tags, cross) = sample_cross();
        StepFrames {
            pos: encode_wire_position(0),
            hid: encode_hidden_frame(&hidden, shape, mask).unwrap(),
            header: encode_cross_kv_header(&tags),
            cross,
            mask: mask.map(<[u32]>::to_vec),
        }
    }

    /// The marker's exact hidden frame for `hidden` of `[1, n, H]`: I32
    /// `[1, n, H+1]`, each row's H f32 bits followed by `REPEAT`.
    fn repeat_frame_of(hidden: &[f32], [_, n, h]: [usize; 3]) -> WireTensor {
        let mut data = Vec::new();
        for row in hidden.chunks(h) {
            data.extend_from_slice(&f32_to_bytes(row));
            data.extend_from_slice(&REPEAT.to_le_bytes());
        }
        WireTensor::new(WireDType::I32, [1, n as u32, h as u32 + 1], data)
    }

    fn sample_repeat_frame() -> WireTensor {
        let (hidden, shape) = sample_hidden();
        repeat_frame_of(&hidden, shape)
    }

    /// The marker costs the plain frame plus one word per row, whatever the
    /// vocab: n=1, H=4096 with a 128k-token mask (4096 words, which a full
    /// masked frame needs a second row for).
    #[test]
    fn a_marker_costs_one_word_per_row_over_the_plain_frame() {
        for (n, h) in [(1usize, 4096usize), (3, 4096), (2, 4)] {
            let hidden: Vec<f32> = (0..n * h).map(|i| i as f32 * 0.5).collect();
            let words = vec![0x5555_5555u32; 131_072 / 32];
            let plain = encode_hidden_frame(&hidden, [1, n, h], None).unwrap();
            let mut hid = encode_hidden_frame(&hidden, [1, n, h], Some(&words)).unwrap();
            into_repeat_frame(&mut hid);
            assert_eq!(hid, repeat_frame_of(&hidden, [1, n, h]));
            assert_eq!(hid.data.len(), plain.data.len() + 4 * n, "n={n} H={h}");
            let mut recv = MaskRecvCache::default();
            recv.commit(&words);
            match resolve_hidden(decode_hidden_frame(&hid).unwrap(), &mut recv) {
                HiddenIn::Masked {
                    floats,
                    shape,
                    words: got,
                } => {
                    assert_eq!((floats, shape), (hidden.clone(), [1, n, h]));
                    assert_eq!(got, words);
                }
                other => panic!("{other:?}"),
            }
        }
    }

    /// A relay stage's step read the way the engine reads it, with the link's
    /// receive cache.
    async fn cached_stage_read(
        g: &mut ActivationServer,
        recv: &mut MaskRecvCache,
    ) -> (WireTensor, HiddenIn) {
        let _pos = g.recv().await.unwrap().0;
        let (hid, _tags, _frames) = recv_step_frames(g, 2).await.unwrap();
        let decoded = resolve_hidden(decode_hidden_frame(&hid).unwrap(), recv);
        (hid, decoded)
    }

    /// The tail's step as `step_last` runs it on the cached wire: a miss runs
    /// nothing and answers the mask-miss reply.
    async fn cached_tail_step(up: &mut ActivationServer, recv: &mut MaskRecvCache) -> HiddenIn {
        let (_, hidden_in) = cached_stage_read(up, recv).await;
        let reply = match &hidden_in {
            HiddenIn::Masked { words, .. } => {
                let (row, shape) = tail_row();
                tail_masked_reply(Ok((row, shape)), words).0
            }
            HiddenIn::MaskMiss => StageReply::MaskMiss,
            other => panic!("expected a masked step, got {other:?}"),
        };
        up.send(&encode_masked_reply(reply)).await.unwrap();
        hidden_in
    }

    /// A repeated mask goes as the marker, a frame of the hidden plus one
    /// row whatever the vocab; the tail resolves it and picks the same token.
    #[tokio::test]
    async fn a_repeated_mask_goes_as_a_short_marker_and_the_tail_picks_the_same_token() {
        let (mut server, mut client) = link().await;
        let words = ok_words();
        let mut send = MaskSendCache::default();
        let mut recv = MaskRecvCache::default();
        let (hidden, shape) = sample_hidden();
        for step in 0..3 {
            let mut frames = sample_frames(Some(&words));
            let tail = async {
                let (hid, decoded) = cached_stage_read(&mut server, &mut recv).await;
                let (row, rshape) = tail_row();
                let HiddenIn::Masked { words: got, .. } = &decoded else {
                    panic!("{decoded:?}")
                };
                let reply = tail_masked_reply(Ok((row, rshape)), got).0;
                server.send(&encode_masked_reply(reply)).await.unwrap();
                (hid, decoded)
            };
            let ((w, _), (hid, decoded)) = tokio::join!(
                masked_round(&mut client, &mut frames, Duration::from_secs(5), &mut send),
                tail
            );
            assert!(
                matches!(w, MaskedWait::Token(t) if t == b'"' as i32),
                "{w:?}"
            );
            if step == 0 {
                assert_eq!(
                    hid,
                    encode_hidden_frame(&hidden, shape, Some(&words)).unwrap()
                );
            } else {
                assert_eq!(hid, sample_repeat_frame(), "step {step}");
                assert_eq!(
                    hid.data.len(),
                    shape[1] * (shape[2] + 1) * 4,
                    "one marker word per row, no mask words"
                );
            }
            assert_eq!(
                decoded,
                HiddenIn::Masked {
                    floats: hidden.clone(),
                    shape,
                    words: words.clone()
                }
            );
        }
    }

    /// The first masked step and every changed mask go full; a plain step in
    /// between is the base byte stream and doesn't reset the link.
    #[tokio::test]
    async fn the_first_mask_and_every_changed_mask_go_full() {
        let (mut server, mut client) = link().await;
        let (a, b) = (ok_words(), other_words());
        let mut send = MaskSendCache::default();
        let mut recv = MaskRecvCache::default();
        let (hidden, shape) = sample_hidden();
        let steps: [(Option<&[u32]>, bool); 6] = [
            (Some(&a), false),
            (Some(&a), true),
            (None, false),
            (Some(&a), true),
            (Some(&b), false),
            (Some(&a), false),
        ];
        for (mask, repeat) in steps {
            let mut frames = sample_frames(mask);
            send_step_cached(&mut client, &mut frames, &mut send)
                .await
                .unwrap();
            let (hid, decoded) = cached_stage_read(&mut server, &mut recv).await;
            match mask {
                None => assert_eq!(hid, encode_hidden_frame(&hidden, shape, None).unwrap()),
                Some(_) if repeat => assert_eq!(hid, sample_repeat_frame()),
                Some(words) => {
                    assert_eq!(
                        hid,
                        encode_hidden_frame(&hidden, shape, Some(words)).unwrap()
                    )
                }
            }
            match (mask, decoded) {
                (None, HiddenIn::Plain { .. }) => {}
                (Some(want), HiddenIn::Masked { words, .. }) => assert_eq!(words, want),
                (m, d) => panic!("{m:?} {d:?}"),
            }
        }
    }

    /// A marker on a fresh receiver is a miss: the tail runs nothing and
    /// answers `[-2, MASK_ACK]`. The head fails the task with the retry
    /// message, clears its send cache, and doesn't latch; the next masked
    /// step goes full.
    #[tokio::test]
    async fn a_repeat_on_a_fresh_receiver_fails_the_task_without_latching() {
        let (mut server, mut client) = link().await;
        let words = ok_words();
        let mut send = MaskSendCache::default();
        send.record(0, &words);
        let mut recv = MaskRecvCache::default();
        let mut frames = sample_frames(Some(&words));
        let ((w, late), got) = tokio::join!(
            masked_round(&mut client, &mut frames, Duration::from_secs(5), &mut send),
            cached_tail_step(&mut server, &mut recv)
        );
        assert_eq!(got, HiddenIn::MaskMiss);
        assert!(!late);
        assert!(matches!(w, MaskedWait::MaskMiss), "{w:?}");
        assert!(send.is_empty(), "the head forgets the mask");
        let mut latch = MaskLatch::default();
        let now = std::time::Instant::now();
        let (err, kind) = head_masked_token(w, &mut latch, now).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("backend error: {MASK_MISS_MESSAGE}")
        );
        assert_eq!(kind, None);
        assert!(latch.refusal(now).is_ok(), "no latch");

        // The miss reply's exact bytes.
        let miss = encode_masked_reply(StageReply::MaskMiss);
        assert_eq!(miss.shape, [1, 1, 2]);
        let mut want = MASK_MISS_TOKEN.to_le_bytes().to_vec();
        want.extend_from_slice(&MASK_ACK.to_le_bytes());
        assert_eq!(miss.data, want);
        assert_eq!(decode_stage_reply(&miss).unwrap(), StageReply::MaskMiss);

        // The next step goes full and succeeds.
        let mut frames = sample_frames(Some(&words));
        let ((w, _), got) = tokio::join!(
            masked_round(&mut client, &mut frames, Duration::from_secs(5), &mut send),
            cached_tail_step(&mut server, &mut recv)
        );
        assert!(matches!(got, HiddenIn::Masked { .. }), "{got:?}");
        assert!(
            matches!(w, MaskedWait::Token(t) if t == b'"' as i32),
            "{w:?}"
        );
    }

    /// The middle's step as `step_middle` runs it on the cached wire (the
    /// model an identity): a miss from upstream answers the mask-miss reply
    /// and sends nothing down; else it re-encodes with its own send cache.
    /// `fail_own_step` fails its own work before it forwards.
    async fn cached_middle_step(
        up: &mut ActivationServer,
        down: &mut ActivationClient,
        recv: &mut MaskRecvCache,
        send: &mut MaskSendCache,
        fail_own_step: bool,
    ) -> StageReply {
        let pos_t = up.recv().await.unwrap().0;
        let (hid, tags, frames) = recv_step_frames(up, 2).await.unwrap();
        let reply = match resolve_hidden(decode_hidden_frame(&hid).unwrap(), recv) {
            HiddenIn::MaskMiss => StageReply::MaskMiss,
            HiddenIn::Masked {
                floats,
                shape,
                words,
            } => {
                let prepared = if fail_own_step {
                    Err(EngineError::Backend("own".into()))
                } else {
                    encode_hidden_frame(&floats, shape, Some(&words)).map(|hid| StepFrames {
                        pos: pos_t,
                        hid,
                        header: encode_cross_kv_header(&tags),
                        cross: frames,
                        mask: Some(words),
                    })
                };
                middle_masked_step(down, prepared, Duration::from_secs(5), send)
                    .await
                    .0
            }
            other => panic!("expected a masked step, got {other:?}"),
        };
        up.send(&encode_masked_reply(reply)).await.unwrap();
        reply
    }

    /// Head → middle → tail, miss at the tail: the middle passes the
    /// mask-miss reply up unchanged; the head fails the task without
    /// latching, and the middle forgets its downstream mask.
    #[tokio::test]
    async fn a_miss_at_the_tail_reaches_the_head_through_a_middle() {
        let (mut mid_up, mut head) = link().await;
        let (mut tail_up, mut mid_down) = link().await;
        let words = ok_words();
        let mut head_send = MaskSendCache::default();
        let mut mid_recv = MaskRecvCache::default();
        let mut mid_send = MaskSendCache::default();
        // The middle believes the tail holds `words`; the tail is fresh.
        mid_send.record(0, &words);
        let mut tail_recv = MaskRecvCache::default();
        let mut frames = sample_frames(Some(&words));
        let ((w, _), mid_reply, tail_got) = tokio::join!(
            masked_round(
                &mut head,
                &mut frames,
                Duration::from_secs(10),
                &mut head_send
            ),
            cached_middle_step(
                &mut mid_up,
                &mut mid_down,
                &mut mid_recv,
                &mut mid_send,
                false
            ),
            cached_tail_step(&mut tail_up, &mut tail_recv)
        );
        assert_eq!(tail_got, HiddenIn::MaskMiss);
        assert_eq!(
            mid_reply,
            StageReply::MaskMiss,
            "relayed, not turned into a NACK"
        );
        assert!(mid_send.is_empty());
        let mut latch = MaskLatch::default();
        let now = std::time::Instant::now();
        let (err, _) = head_masked_token(w, &mut latch, now).unwrap_err();
        assert!(err.to_string().contains(MASK_MISS_MESSAGE), "{err}");
        assert!(latch.refusal(now).is_ok(), "no latch");
        assert!(head_send.is_empty());

        // The next step goes full on both hops and succeeds.
        let mut frames = sample_frames(Some(&words));
        let ((w, _), mid_reply, tail_got) = tokio::join!(
            masked_round(
                &mut head,
                &mut frames,
                Duration::from_secs(10),
                &mut head_send
            ),
            cached_middle_step(
                &mut mid_up,
                &mut mid_down,
                &mut mid_recv,
                &mut mid_send,
                false
            ),
            cached_tail_step(&mut tail_up, &mut tail_recv)
        );
        assert!(matches!(tail_got, HiddenIn::Masked { .. }), "{tail_got:?}");
        assert_eq!(mid_reply, StageReply::Token(b'"' as i32));
        assert!(
            matches!(w, MaskedWait::Token(t) if t == b'"' as i32),
            "{w:?}"
        );
    }

    /// A middle resolves the upstream's marker and re-encodes it with its own
    /// cache. When it fails a step it decoded without forwarding it, the
    /// upstream's next marker goes downstream full.
    #[tokio::test]
    async fn a_marker_from_upstream_goes_down_full_after_the_middle_dropped_a_step() {
        let (mut mid_up, mut head) = link().await;
        let (mut tail_up, mut mid_down) = link().await;
        let (a, b) = (ok_words(), other_words());
        let mut head_send = MaskSendCache::default();
        let mut mid_recv = MaskRecvCache::default();
        let mut mid_send = MaskSendCache::default();
        let mut tail_recv = MaskRecvCache::default();
        let (hidden, shape) = sample_hidden();
        let mut tail_hids = Vec::new();
        // (mask the head sends, the middle fails its own step)
        let steps: [(&[u32], bool); 4] = [(&a, false), (&a, false), (&b, true), (&b, false)];
        for (words, fail) in steps {
            let mut frames = sample_frames(Some(words));
            // The head keeps its cache across the middle's failure: the
            // middle decoded every frame it was sent.
            send_step_cached(&mut head, &mut frames, &mut head_send)
                .await
                .unwrap();
            let tail = async {
                if fail {
                    return;
                }
                let (hid, decoded) = cached_stage_read(&mut tail_up, &mut tail_recv).await;
                tail_hids.push(hid);
                let HiddenIn::Masked { words: got, .. } = decoded else {
                    panic!("{decoded:?}")
                };
                assert_eq!(got, words);
                tail_up
                    .send(&encode_masked_reply(StageReply::Token(1)))
                    .await
                    .unwrap();
            };
            let (reply, ()) = tokio::join!(
                cached_middle_step(
                    &mut mid_up,
                    &mut mid_down,
                    &mut mid_recv,
                    &mut mid_send,
                    fail
                ),
                tail
            );
            let w = wait_masked_reply(&mut head, Duration::from_secs(5)).await;
            if fail {
                assert_eq!(reply, StageReply::Nack);
                assert!(matches!(w, MaskedWait::Nack), "{w:?}");
            } else {
                assert!(matches!(w, MaskedWait::Token(1)), "{w:?}");
            }
        }
        assert_eq!(tail_hids.len(), 3);
        assert_eq!(
            tail_hids[0],
            encode_hidden_frame(&hidden, shape, Some(&a)).unwrap()
        );
        assert_eq!(
            tail_hids[1],
            sample_repeat_frame(),
            "the middle re-encodes as a marker"
        );
        assert_eq!(
            tail_hids[2],
            encode_hidden_frame(&hidden, shape, Some(&b)).unwrap(),
            "the tail never saw b: it goes full"
        );
    }

    /// Every failed step on a link clears its send cache: an acknowledged
    /// NACK, an unacknowledged reply, silence, a failed plain wait, a write
    /// error. A good reply keeps it.
    #[tokio::test]
    async fn the_sender_forgets_the_mask_on_every_failed_step() {
        let words = ok_words();
        let replies = [
            (Some(StageReply::Nack), false),
            (Some(StageReply::NotAcked), false),
            (Some(StageReply::Plain(3)), false),
            (None, false),
            (Some(StageReply::Token(3)), true),
        ];
        for (reply, kept) in replies {
            let (mut server, mut client) = link().await;
            let mut send = MaskSendCache::default();
            send.record(0, &words);
            let mut frames = sample_frames(Some(&words));
            let tail = async {
                let _ = cached_stage_read(&mut server, &mut MaskRecvCache::default()).await;
                if let Some(r) = reply {
                    server.send(&encode_masked_reply(r)).await.unwrap();
                }
            };
            let ((w, _), ()) = tokio::join!(
                masked_round(
                    &mut client,
                    &mut frames,
                    // Short only where no reply comes.
                    if reply.is_some() {
                        Duration::from_secs(5)
                    } else {
                        Duration::from_millis(100)
                    },
                    &mut send
                ),
                tail
            );
            assert_eq!(!send.is_empty(), kept, "{reply:?} {w:?}");
        }

        // A failed plain wait (the downstream went away).
        let (server, mut client) = link().await;
        drop(server);
        let mut send = MaskSendCache::default();
        send.record(0, &words);
        assert!(plain_reply_settled(&mut client, false, &mut send)
            .await
            .is_err());
        assert!(send.is_empty());

        // A write error.
        let (server, mut client) = link().await;
        drop(server);
        client.close().await;
        let mut send = MaskSendCache::default();
        send.record(0, &words);
        let mut frames = sample_frames(Some(&words));
        assert!(send_step_cached(&mut client, &mut frames, &mut send)
            .await
            .is_err());
        assert!(send.is_empty());
    }

    /// A stage that predates the marker reads every frame of a marker step,
    /// then fails on the I32 hidden exactly as for a full mask, and stays
    /// aligned.
    #[tokio::test]
    async fn an_old_stage_reads_every_frame_of_a_marker_step_then_fails() {
        let (mut server, mut client) = link().await;
        let words = ok_words();
        let mut send = MaskSendCache::default();
        send.record(0, &words);
        let mut frames = sample_frames(Some(&words));
        frames.pos = encode_wire_position(4);
        send_step_cached(&mut client, &mut frames, &mut send)
            .await
            .unwrap();
        assert_eq!(frames.hid, sample_repeat_frame());
        send_sample_step(&mut client, 6, None).await;
        let err = old_stage_read(&mut server, 2).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("unexpected upstream hidden dtype I32"),
            "{err}"
        );
        let (floats, shape, pos) = old_stage_read(&mut server, 2).await.unwrap();
        let (hidden, hshape) = sample_hidden();
        assert_eq!((floats, shape, pos), (hidden, hshape, 6));
    }

    /// A malformed marker (non-zero padding, or more than one trailing row)
    /// is a bad mask, and a bad masked frame clears the receive cache: a
    /// marker after it misses.
    #[test]
    fn a_bad_masked_frame_clears_the_receive_cache() {
        let words = ok_words();
        let mut recv = MaskRecvCache::default();
        recv.commit(&words);
        // A row whose last element isn't the marker.
        let mut bad = sample_repeat_frame();
        bad.data[16..20].copy_from_slice(&7u32.to_le_bytes());
        let decoded = decode_hidden_frame(&bad).unwrap();
        assert!(matches!(decoded, HiddenIn::BadMask(_)), "{decoded:?}");
        assert!(matches!(
            resolve_hidden(decoded, &mut recv),
            HiddenIn::BadMask(_)
        ));
        let repeat = decode_hidden_frame(&sample_repeat_frame()).unwrap();
        assert_eq!(resolve_hidden(repeat, &mut recv), HiddenIn::MaskMiss);
    }

    /// A late mask-miss reply is discarded like any late masked reply (by the
    /// plain wait and by the drain before a masked step), but unlike the
    /// others it is no evidence that the chain masks.
    #[tokio::test]
    async fn a_late_mask_miss_reply_is_discarded_as_stale() {
        let (mut server, mut client) = link().await;
        server
            .send(&encode_masked_reply(StageReply::MaskMiss))
            .await
            .unwrap();
        server.send(&encode_token_reply(7)).await.unwrap();
        // Discarded, and no evidence that the chain masks.
        assert_eq!(
            wait_plain_reply(&mut client, false).await.unwrap(),
            (7, false)
        );
        server
            .send(&encode_masked_reply(StageReply::MaskMiss))
            .await
            .unwrap();
        // The reply is on the socket before the drain looks.
        tokio::time::timeout(Duration::from_secs(5), client.wait_readable())
            .await
            .expect("the late reply arrives")
            .unwrap();
        assert!(!drain_late_replies(&mut client).await.unwrap());
        // Drained: the link is empty again.
        let empty = tokio::time::timeout(Duration::from_millis(200), client.wait_readable()).await;
        assert!(empty.is_err());
    }

    // ---- the mask-support probe ----

    /// A short probe wait for tests; the real one is `MASK_PROBE_TIMEOUT`.
    const TEST_PROBE: Duration = Duration::from_millis(60);

    /// The tail's step on a probe or a masked step, as `step_last` answers
    /// them (the model replaced by `tail_row`). Returns what it read.
    async fn new_tail_answer(up: &mut ActivationServer, cache: &mut MaskRecvCache) -> HiddenIn {
        let (hidden_in, _) = new_stage_read(up, 2).await;
        let hidden_in = resolve_hidden(hidden_in, cache);
        let reply = match &hidden_in {
            HiddenIn::Probe => StageReply::ProbeOk,
            HiddenIn::Masked { words, .. } => {
                let (row, shape) = tail_row();
                tail_masked_reply(Ok((row, shape)), words).0
            }
            other => panic!("expected a probe or a masked step, got {other:?}"),
        };
        up.send(&encode_masked_reply(reply)).await.unwrap();
        hidden_in
    }

    /// The middle's step on a probe (`step_middle`): forward its own probe
    /// and pass the answer up.
    async fn new_middle_probe(
        up: &mut ActivationServer,
        down: &mut ActivationClient,
        budget: Duration,
    ) {
        let (hidden_in, _) = new_stage_read(up, 2).await;
        assert_eq!(hidden_in, HiddenIn::Probe);
        let (reply, _) = middle_probe_step(down, probe_frames(&[10, 11]), budget).await;
        up.send(&encode_masked_reply(reply)).await.unwrap();
    }

    /// The probe's exact frames: position 0, an I32 `[1,1,2]` hidden of one
    /// zero element and `MASK_PROBE`, the decode step's cross-KV header and
    /// one F32 `[1,1,1]` zero frame per tag. Its answer is `[-3, MASK_ACK]`.
    #[test]
    fn the_probe_frames_and_their_answer_have_exact_bytes() {
        let f = probe_frames(&[10, 11]);
        assert_eq!(f.pos, encode_wire_position(0));
        let mut hid = 0u32.to_le_bytes().to_vec();
        hid.extend_from_slice(&MASK_PROBE.to_le_bytes());
        assert_eq!(f.hid, WireTensor::new(WireDType::I32, [1, 1, 2], hid));
        assert_eq!(f.header, encode_cross_kv_header(&[10, 11]));
        let zero = WireTensor::new(WireDType::F32, [1, 1, 1], vec![0; 4]);
        assert_eq!(f.cross, vec![zero.clone(), zero]);
        assert!(f.mask.is_none(), "a probe carries no mask");
        assert!(probe_frames(&[]).cross.is_empty());
        let mut ok = (-3i32).to_le_bytes().to_vec();
        ok.extend_from_slice(&MASK_ACK.to_le_bytes());
        assert_eq!(
            encode_masked_reply(StageReply::ProbeOk),
            WireTensor::new(WireDType::I32, [1, 1, 2], ok)
        );
        assert_eq!(
            decode_stage_reply(&encode_masked_reply(StageReply::ProbeOk)).unwrap(),
            StageReply::ProbeOk
        );
    }

    /// New decoders read a probe before a repeat marker or a mask; anything
    /// but one zero row ending in `MASK_PROBE` is a bad masked frame.
    #[test]
    fn a_probe_hidden_decodes_as_a_probe_and_nothing_else_does() {
        let f = probe_frames(&[]);
        assert_eq!(decode_hidden_frame(&f.hid).unwrap(), HiddenIn::Probe);
        let i32s = |v: &[u32], shape: [u32; 3]| {
            WireTensor::new(
                WireDType::I32,
                shape,
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        };
        // Wider rows are a probe too.
        assert_eq!(
            decode_hidden_frame(&i32s(&[0, 0, 0, MASK_PROBE], [1, 1, 4])).unwrap(),
            HiddenIn::Probe
        );
        for bad in [
            i32s(&[7, MASK_PROBE], [1, 1, 2]),
            i32s(&[0, MASK_PROBE, 0, MASK_PROBE], [1, 2, 2]),
            i32s(&[MASK_PROBE], [1, 1, 1]),
        ] {
            assert!(
                matches!(decode_hidden_frame(&bad).unwrap(), HiddenIn::BadMask(_)),
                "{bad:?}"
            );
        }
    }

    /// A chain that masks answers the probe at once; the head records the
    /// evidence, stops probing, and its masked step runs.
    #[tokio::test]
    async fn a_new_tail_acknowledges_the_probe_at_once_then_the_masked_step_runs() {
        let (mut up, mut head) = link().await;
        let mut latch = MaskLatch::default();
        assert!(needs_mask_probe(&latch, true, true));
        let mut recv = MaskRecvCache::default();
        let t0 = std::time::Instant::now();
        let ((wait, _), read) = tokio::join!(
            probe_round(&mut head, probe_frames(&[10, 11]), MASK_PROBE_TIMEOUT),
            new_tail_answer(&mut up, &mut recv)
        );
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        assert_eq!(read, HiddenIn::Probe);
        head_probe_verdict(wait, &mut latch, std::time::Instant::now()).unwrap();
        assert!(
            !needs_mask_probe(&latch, true, true),
            "evidence: no more probes"
        );

        let words = ok_words();
        let (hidden, shape) = sample_hidden();
        let (tags, cross) = sample_cross();
        let mut frames = StepFrames {
            pos: encode_wire_position(0),
            hid: encode_hidden_frame(&hidden, shape, Some(&words)).unwrap(),
            header: encode_cross_kv_header(&tags),
            cross,
            mask: Some(words),
        };
        let mut send = MaskSendCache::default();
        let ((wait, _), _) = tokio::join!(
            masked_round(&mut head, &mut frames, MASK_PROBE_TIMEOUT, &mut send),
            new_tail_answer(&mut up, &mut recv)
        );
        assert!(
            matches!(wait, MaskedWait::Token(t) if t == b'"' as i32),
            "{wait:?}"
        );
    }

    /// An older tail (the relay read before masks) reads every probe frame,
    /// the cross-KV header's count included, rejects the hidden and answers
    /// nothing. The head gives up at its short probe wait, latches with the
    /// upgrade message, and the link stays aligned for the next plain step.
    #[tokio::test]
    async fn an_old_tail_leaves_the_probe_unanswered_and_the_head_latches_quickly() {
        for i in 0..25 {
            let (mut up, mut head) = link().await;
            let mut latch = MaskLatch::default();
            let t0 = std::time::Instant::now();
            let ((wait, _), old) = tokio::join!(
                probe_round(&mut head, probe_frames(&[10, 11]), TEST_PROBE),
                old_stage_read(&mut up, 2)
            );
            let took = t0.elapsed();
            let why = old.unwrap_err().to_string();
            assert!(
                why.contains("hidden dtype"),
                "{i}: rejected at the hidden: {why}"
            );
            let (e, kind) =
                head_probe_verdict(wait, &mut latch, std::time::Instant::now()).unwrap_err();
            assert!(e.to_string().contains(UPGRADE_EVERY_STAGE), "{e}");
            assert_eq!(kind, Some(GrammarFailure::Upgrade));
            assert!(took >= TEST_PROBE, "{i}: {took:?}");
            assert!(took < Duration::from_secs(5), "{i}: {took:?}");
            assert!(latch.refusal(std::time::Instant::now()).is_err(), "latched");
            // The next plain step: read whole by the old stage, answered.
            send_sample_step(&mut head, 0, None).await;
            let (_, shape, pos) = old_stage_read(&mut up, 2).await.unwrap();
            assert_eq!((shape, pos), (sample_hidden().1, 0), "{i}: aligned");
            up.send(&encode_token_reply(42)).await.unwrap();
            assert_eq!(
                wait_plain_reply(&mut head, false).await.unwrap(),
                (42, false)
            );
        }
    }

    /// A probe changes no state at a new stage: the receive cache keeps its
    /// last mask, and the head's send cache isn't touched.
    #[tokio::test]
    async fn a_probe_touches_no_mask_cache() {
        let words = ok_words();
        let (mut up, mut head) = link().await;
        let mut recv = MaskRecvCache::default();
        recv.commit(&words);
        let ((wait, _), _) = tokio::join!(
            probe_round(&mut head, probe_frames(&[10, 11]), MASK_PROBE_TIMEOUT),
            new_tail_answer(&mut up, &mut recv)
        );
        assert!(matches!(wait, MaskedWait::ProbeOk), "{wait:?}");
        assert_eq!(recv.last(), Some(words.as_slice()));
        assert_eq!(resolve_hidden(HiddenIn::Probe, &mut recv), HiddenIn::Probe);
        assert_eq!(recv.last(), Some(words.as_slice()));
    }

    /// Head → middle → tail: the middle forwards its own probe and passes
    /// the tail's answer up. A middle's probe wait is shorter than the stage
    /// above it.
    #[tokio::test]
    async fn a_middle_relays_the_probe_and_its_answer() {
        for k in 1..5 {
            assert!(
                probe_wait_budget(MASK_PROBE_TIMEOUT, k)
                    < probe_wait_budget(MASK_PROBE_TIMEOUT, k + 1)
            );
        }
        assert_eq!(probe_wait_budget(MASK_PROBE_TIMEOUT, 1), MASK_PROBE_TIMEOUT);
        let (mut mid_up, mut head) = link().await;
        let (mut tail_up, mut mid_down) = link().await;
        let mut recv = MaskRecvCache::default();
        let mut latch = MaskLatch::default();
        let ((wait, _), _, read) = tokio::join!(
            probe_round(
                &mut head,
                probe_frames(&[10, 11]),
                probe_wait_budget(MASK_PROBE_TIMEOUT, 2)
            ),
            new_middle_probe(
                &mut mid_up,
                &mut mid_down,
                probe_wait_budget(MASK_PROBE_TIMEOUT, 1)
            ),
            new_tail_answer(&mut tail_up, &mut recv)
        );
        assert_eq!(read, HiddenIn::Probe);
        head_probe_verdict(wait, &mut latch, std::time::Instant::now()).unwrap();
        assert!(!needs_mask_probe(&latch, true, true));
    }

    /// Head → middle → old tail: the middle's shorter wait runs out first and
    /// its not-acknowledged answer reaches the head inside the head's wait.
    #[tokio::test]
    async fn a_middle_above_an_old_tail_answers_the_probe_inside_the_heads_wait() {
        for i in 0..10 {
            let (mut mid_up, mut head) = link().await;
            let (mut tail_up, mut mid_down) = link().await;
            let mut latch = MaskLatch::default();
            // A 2 s base gives the head a 400 ms margin over the middle, so a
            // loaded test runner can't let the head's own timer win the race.
            let base = std::time::Duration::from_secs(2);
            let head_budget = probe_wait_budget(base, 2);
            let t0 = std::time::Instant::now();
            let ((wait, _), _, old) = tokio::join!(
                probe_round(&mut head, probe_frames(&[10, 11]), head_budget),
                new_middle_probe(&mut mid_up, &mut mid_down, probe_wait_budget(base, 1)),
                old_stage_read(&mut tail_up, 2)
            );
            assert!(old.is_err());
            assert!(t0.elapsed() < head_budget, "{i}: {:?}", t0.elapsed());
            assert!(
                matches!(&wait, MaskedWait::Unacked(m) if m.contains("relayed")),
                "{i}: the middle's answer, not the head's own timeout: {wait:?}"
            );
            assert!(head_probe_verdict(wait, &mut latch, std::time::Instant::now()).is_err());
            assert!(latch.refusal(std::time::Instant::now()).is_err());
        }
    }

    /// A slow but new tail whose answer arrives after the head gave up: the
    /// next structured-output request drains it before the refusal, and the
    /// acknowledged late answer clears the latch.
    #[tokio::test]
    async fn a_late_probe_answer_clears_the_latch_on_the_next_request() {
        for i in 0..20 {
            let (mut up, mut head) = link().await;
            let mut latch = MaskLatch::default();
            let ((wait, _), _) = tokio::join!(
                probe_round(&mut head, probe_frames(&[10, 11]), TEST_PROBE),
                new_stage_read(&mut up, 2)
            );
            assert!(head_probe_verdict(wait, &mut latch, std::time::Instant::now()).is_err());
            assert!(
                !drain_late_replies(&mut head).await.unwrap(),
                "{i}: nothing yet"
            );
            up.send(&encode_masked_reply(StageReply::ProbeOk))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert!(
                drain_late_replies(&mut head).await.unwrap(),
                "{i}: the late answer"
            );
            latch.clear();
            assert!(latch.refusal(std::time::Instant::now()).is_ok());
            assert!(!needs_mask_probe(&latch, true, true), "{i}: evidence");
        }
    }

    /// In a probe's wait every reply but its own answer (or an acknowledged
    /// NACK) is a late one: skipped (acknowledged ones counted as evidence)
    /// while the wait goes on. A plain wait and the drain skip a late probe
    /// answer the same way.
    #[tokio::test]
    async fn the_probe_wait_skips_late_replies() {
        let (mut up, mut head) = link().await;
        for r in [
            encode_token_reply(4),
            encode_masked_reply(StageReply::Token(5)),
            encode_masked_reply(StageReply::MaskMiss),
            encode_masked_reply(StageReply::ProbeOk),
        ] {
            up.send(&r).await.unwrap();
        }
        let (wait, late_ack) = wait_probe_reply(&mut head, TEST_PROBE).await;
        assert!(matches!(wait, MaskedWait::ProbeOk), "{wait:?}");
        assert!(late_ack);
        // An acknowledged NACK answers the probe: a stage that masks failed it.
        up.send(&encode_masked_reply(StageReply::Nack))
            .await
            .unwrap();
        let (wait, _) = wait_probe_reply(&mut head, TEST_PROBE).await;
        assert!(matches!(wait, MaskedWait::Nack), "{wait:?}");
        // Only a mask miss before it: no evidence, and the wait runs out.
        up.send(&encode_masked_reply(StageReply::MaskMiss))
            .await
            .unwrap();
        let (wait, late_ack) = wait_probe_reply(&mut head, TEST_PROBE).await;
        assert!(matches!(wait, MaskedWait::Unacked(_)), "{wait:?}");
        assert!(!late_ack);

        up.send(&encode_masked_reply(StageReply::ProbeOk))
            .await
            .unwrap();
        up.send(&encode_token_reply(42)).await.unwrap();
        assert_eq!(
            wait_plain_reply(&mut head, false).await.unwrap(),
            (42, true)
        );
        up.send(&encode_masked_reply(StageReply::ProbeOk))
            .await
            .unwrap();
        up.send(&encode_masked_reply(StageReply::Token(7)))
            .await
            .unwrap();
        let w = wait_masked_reply(&mut head, TEST_PROBE).await;
        assert!(matches!(w, MaskedWait::Token(7)), "{w:?}");
    }

    /// A latch set by an unanswered probe refuses for a minute; one set by a
    /// masked step keeps ten. Either way the next request after it probes.
    /// Only a structured-output task on a multi-stage pipeline without
    /// evidence probes.
    #[test]
    fn a_probe_latch_re_probes_after_a_minute() {
        let t0 = std::time::Instant::now();
        let sec = Duration::from_secs(1);
        let mut latch = MaskLatch::default();
        latch.trip_probe(t0);
        assert!(latch.refusal(t0 + MASK_PROBE_REPROBE_AFTER - sec).is_err());
        assert!(latch.refusal(t0 + MASK_PROBE_REPROBE_AFTER).is_ok());
        assert!(needs_mask_probe(&latch, true, true));

        let mut latch = MaskLatch::default();
        latch.on_evidence();
        assert!(!needs_mask_probe(&latch, true, true));
        latch.trip(t0);
        assert!(latch.refusal(t0 + MASK_PROBE_REPROBE_AFTER).is_err());
        assert!(latch.refusal(t0 + MASK_LATCH_REPROBE).is_ok());
        assert!(needs_mask_probe(&latch, true, true));

        assert!(
            !needs_mask_probe(&MaskLatch::default(), true, false),
            "plain task"
        );
        assert!(
            !needs_mask_probe(&MaskLatch::default(), false, true),
            "one stage"
        );
        // A masked step's acknowledged reply is evidence too.
        let mut latch = MaskLatch::default();
        head_masked_token(MaskedWait::Token(3), &mut latch, t0).unwrap();
        assert!(!needs_mask_probe(&latch, true, true));
        let mut latch = MaskLatch::default();
        let _ = head_masked_token(MaskedWait::Nack, &mut latch, t0);
        assert!(!needs_mask_probe(&latch, true, true));
    }

    /// A middle whose own downstream is gone can't forward the probe: it
    /// answers the acknowledged NACK (a stage that masks failed the step),
    /// and the head fails the task at once without latching or naming an
    /// upgrade.
    #[tokio::test]
    async fn a_middle_that_loses_its_downstream_fails_the_probe_without_a_latch() {
        for i in 0..20 {
            let (mut mid_up, mut head) = link().await;
            let (mut tail_up, mut mid_down) = link().await;
            tail_up.close().await;
            drop(tail_up);
            let mut latch = MaskLatch::default();
            let t0 = std::time::Instant::now();
            let ((wait, _), _) = tokio::join!(
                probe_round(&mut head, probe_frames(&[10, 11]), MASK_PROBE_TIMEOUT),
                new_middle_probe(
                    &mut mid_up,
                    &mut mid_down,
                    probe_wait_budget(MASK_PROBE_TIMEOUT, 1)
                )
            );
            assert!(
                t0.elapsed() < MASK_PROBE_TIMEOUT / 4,
                "{i}: {:?}",
                t0.elapsed()
            );
            let (e, kind) =
                head_probe_verdict(wait, &mut latch, std::time::Instant::now()).unwrap_err();
            assert!(!e.to_string().contains(UPGRADE_EVERY_STAGE), "{i}: {e}");
            assert_eq!(kind, Some(GrammarFailure::Nack), "{i}");
            assert!(
                latch.refusal(std::time::Instant::now()).is_ok(),
                "{i}: no latch"
            );
        }
    }
}
