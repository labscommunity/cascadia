//! Qwen3.5-family staged engine (`--engine qwen35`, alias `qwen36-moe`):
//! runs the IR-surgery shard chain (tools/qwen36_surgery/export_qwen36_moe.py)
//! in-process on one box (single-box, --total 1) or as an N-node stage
//! pipeline (--rank R --total N). Serves both `qwen3_5_moe` (Qwen3.5/
//! 3.6-35B-A3B, hidden 2048, 40 layers) and dense `qwen3_5` (Qwen3.8-27B,
//! hidden 5120, 64 layers): the stage IRs are opaque to the engine, which
//! only needs the hidden width — read from `manifest.json` — to size the
//! activation frames. Greedy-only, batch=1; the decode loop is a port of
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
use cascadia_ov_genai_shim::{advance_emitted, DType, PluginConfig, RequestState, Runtime};
use cascadia_transport::{
    ActivationClient, ActivationServer, DType as WireDType, Tensor as WireTensor, TransportError,
    TransportResult,
};
use cascadia_types::{
    Chunk, FinishReason, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId,
};
use futures::stream;
use tokenizers::Tokenizer;
use tracing::{error, info, warn};

use crate::pass_timing::{ms, PassTiming};

/// Hidden width of shard trees exported before `manifest.hidden_size`
/// existed (the Qwen3.6-35B-A3B surgery, hidden 2048). Newer manifests
/// carry the width explicitly; the 27B dense export is 5120.
const LEGACY_HIDDEN: usize = 2048;
/// Prefill span per chain pass; bounds the transient [1, T, vocab]
/// logits buffer (~254 MB f32 at 256 with the 248320 vocab) and the
/// per-chunk wire frame (256 * hidden * 4 B: 2 MiB at hidden 2048,
/// 5 MiB at 5120 — both far under the transport's 256 MiB tensor cap).
const PREFILL_CHUNK: usize = 256;

/// How long a stage request a failed infer left busy gets to return to idle (OpenVINO `cancel()`,
/// then `wait_for`) before the engine abandons it (Task 46) and recompiles the stage (Task 54).
const STAGE_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Stages the engine may recompile after an abandon before it fails instead (Task 54 fix round 2).
/// Each recompile leaks the old stage's whole `Runtime` (compiled model and weights: about 1.5 GB
/// for stage6 and 2.5 GB for stage7 of the 9B int8 tree, on a GPU that shares system RAM), so a
/// fault that recurs cannot exhaust memory: the abandon past this marks the engine failed.
const MAX_RECOMPILES: usize = 2;
/// [`MAX_RECOMPILES`] for an engine whose recompile would leak more than [`LARGE_LEAK_BYTES`]
/// (Task 58): two such leaks are not safe on a box that shares system RAM with the GPU. The first
/// fault recovers (and the request is retried, Task 57); the second fails the engine, and the
/// server exits for a supervisor restart (Task 56).
const LARGE_LEAK_MAX_RECOMPILES: usize = 1;
/// The weight bytes of a local stage (the copy one recompile leaks) above which the engine
/// recompiles at most [`LARGE_LEAK_MAX_RECOMPILES`] times (Task 58). Measured stage weight files:
/// the 8-stage 9B int8 tree's largest are 1.49 GB (stage6) and 2.47 GB (stage7), which keep
/// [`MAX_RECOMPILES`]; a 1-stage 9B tree's one stage is the whole model (about 8.4 GiB), and a
/// 27B 2-stage pipeline rank's stage is 12-15 GB, which both get the cap of 1 (two leaks of
/// either can exhaust a 32 GB box).
const LARGE_LEAK_BYTES: u64 = 4 << 30;
/// Successful recompiles at which the engine is flagged degraded (Task 54 fix round 3): each one
/// leaves one leaked stage copy, and two leaked copies cost decode about 16%. A recompile that
/// fails also leaks a copy, but it fails the engine, so no later recompile can warn. Under
/// [`LARGE_LEAK_MAX_RECOMPILES`] only one recompile can succeed, so this never warns there.
const DEGRADED_AT_RECOMPILES: usize = 2;
/// Bound on one stage recompile, run on a helper thread (Task 54 fix round 2). A cold compile
/// without the model cache can take tens of seconds; a cache-warm one measured 1.2-1.7 s on p01.
const RECOMPILE_TIMEOUT: Duration = Duration::from_secs(180);
/// Stage `Runtime`s this process has leaked on purpose ([`leak_stage`]: never dropped, a destructor
/// could wait forever on the stuck request). A copy left with a timed-out or dead helper thread is
/// leaked by `Forget` and not counted here. Logged as `process_leaked`.
static LEAKED_RUNTIMES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// One `wait_for` slice inside [`STAGE_IDLE_TIMEOUT`]: the state is re-read after each.
const STAGE_IDLE_POLL: Duration = Duration::from_millis(100);

/// Effective prefill span. Default = PREFILL_CHUNK (T>1 batched, ~4.2x TTFT win).
///
/// `CASCADIA_QWEN36_FORCE_T1_PREFILL=1` ⇒ 1: fold EVERY token through the same T=1 path that
/// generation (decode) already uses. The DeltaNet/SSM recurrent scan is non-associative in f16, so a
/// T>1 chunked prefill and the T=1 decode fold diverge sub-ulp; the f16 MoE router then amplifies that
/// to a token flip. Under T=1 everywhere, turn-1 prefill, turn-1 decode, and a cold reprefill all
/// traverse the identical kernel ⇒ bit-identical states ⇒ byte-identical greedy (cross-chain warm==cold
/// cert passes). Opt-in only — production keeps chunked prefill; warm-resume there stays
/// greedy-equivalent, not bit-identical. Read per chunk (a handful of times per prefill; negligible).
/// `CASCADIA_QWEN36_FP_AT=<pos>` ⇒ fingerprint the declared state whenever a fold reaches `pos`,
/// on BOTH the warm path (right after restore) and the cold one (mid-prefill). Warm resumes at
/// `pos` and cold passes through it, so the pair isolates where they part: equal at `pos` but
/// unequal later ⇒ the divergence comes from state OUTSIDE the declared blob (set_state_blob
/// round-trips exactly, so a declared-level restore alone cannot explain it); unequal already at
/// `pos` ⇒ the captured turn-1 state itself differs from a cold fold of the same tokens.
fn fp_at() -> Option<usize> {
    std::env::var("CASCADIA_QWEN36_FP_AT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

/// `CASCADIA_QWEN36_SELFCHK=<pos>` ⇒ on a COLD prefill, when the fold reaches `pos`, capture the
/// state, recreate the request, restore the capture, and carry on. Same tokens, same code path, same
/// positions as a plain cold run — the ONLY difference is that the state made a round trip through
/// capture→recreate→restore. If the continuation still diverges from an untouched cold run, the
/// restore machinery itself is unfaithful (and this is a minimal OV repro, with zero warm-path
/// confounders). If it matches, restore is sound and the warm path's *inputs* are the problem.
fn selfchk_at() -> Option<usize> {
    std::env::var("CASCADIA_QWEN36_SELFCHK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
}

/// Pre-restore clear strategy. **Default: none** — `set_state_blob` writes straight over the live
/// request. This is the bar-#1 fix.
///
/// It is the CLEAR that breaks the restore, not the write. Measured with `SELFCHK`, where the bytes
/// written are the ones just captured from this same request so the write is semantically a no-op and
/// the clear is the only variable: `recreate_request` and `reset_state` both diverge from an untouched
/// cold run — identically, same first-drift step and same flipped token — while skipping the clear is
/// byte-identical. Declared state and first token match at the instant decode begins; state drifts one
/// fold later, globally (attention 10/10, DeltaNet 24/30 tensors), and the token only flips 22 steps
/// after that. So the perturbation is invisible to `get_state_blob` and is introduced by tearing the
/// state down and re-establishing it.
///
/// Removing the clear took qwen36 from chain 5/6 + plane 7/8 to **chain 6/6 + plane 8/8**, bar #1
/// included, plus 3/3 single-chain.
///
/// The clear existed because restoring over live folded state once made the model regurgitate the
/// prompt. That does not reproduce. `set_state_blob` overwrites every one of the 40 VariableStates
/// (5 attention key + 5 value + 15 DeltaNet conv + 15 ssm), so no residue of the previous sequence
/// survives for a clear to scrub — including on the cross-chain move, where the live state belongs to
/// a DIFFERENT sequence and which certified 6/6 (chain) and 8/8 (plane).
///
/// `CASCADIA_QWEN36_RESTORE_CLEAR=recreate|reset` restores the old behaviour for A/B work.
fn restore_clear_mode() -> &'static str {
    // Tests set the knob per thread: the variable is process-global and read on every restore, so
    // setting it would change every other restore test running in parallel.
    #[cfg(test)]
    if let Some(v) = tests::restore_clear_knob() {
        return restore_clear_mode_of(Some(v));
    }
    restore_clear_mode_of(
        std::env::var("CASCADIA_QWEN36_RESTORE_CLEAR")
            .ok()
            .as_deref(),
    )
}

/// [`restore_clear_mode`] for a value of `CASCADIA_QWEN36_RESTORE_CLEAR`.
fn restore_clear_mode_of(knob: Option<&str>) -> &'static str {
    match knob {
        Some("recreate") => "recreate_request",
        Some("reset") => "reset_state",
        _ => "none",
    }
}

/// `CASCADIA_QWEN36_DECODE_FP=1` ⇒ fingerprint the local stage state after EVERY decode step.
///
/// Why this exists: with `SELFCHK`, the post-prefill state fingerprint and the first emitted token are
/// BYTE-IDENTICAL to an untouched cold run, yet the text diverges ~15 tokens later. So the perturbation
/// is invisible to `get_state_blob` at the point decode begins and only emerges while decoding.
/// `qwen36_postprefill_state` fires once per turn, which cannot locate that. Comparing this per-step
/// series between a plain cold run and a SELFCHK run gives the first step at which state (or the
/// emitted token) diverges — the difference between a state drift and a decode-only flip.
///
/// Diagnostic only: it calls `get_state_blob` every step, which copies the whole local state.
fn decode_fp() -> bool {
    std::env::var("CASCADIA_QWEN36_DECODE_FP").ok().as_deref() == Some("1")
}

/// Gate for the per-turn `qwen36_postprefill_state` fingerprint (and the set_state round-trip
/// fingerprint): both call `get_state_blob`, a full KV-state copy. Off by default so a `kv_coord`
/// build pays nothing on the no-move hot path — matches every sibling diagnostic here.
fn postprefill_fp() -> bool {
    std::env::var("CASCADIA_QWEN36_POSTPREFILL_FP")
        .ok()
        .as_deref()
        == Some("1")
}

fn prefill_chunk() -> usize {
    if std::env::var("CASCADIA_QWEN36_FORCE_T1_PREFILL")
        .ok()
        .as_deref()
        == Some("1")
    {
        1
    } else {
        PREFILL_CHUNK
    }
}
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
/// Issue-34 consume: head broadcasts RESTORE at admission so every rank `set_state`s its
/// pulled+inserted slice. The ACK's `pos` field carries an all-or-nothing verdict (1 = the whole
/// downstream chain restored; 0 = some rank couldn't ⇒ head falls back to a cold RESET).
#[cfg(feature = "kv_coord")]
const FRAME_RESTORE: u32 = 10;
#[cfg(feature = "kv_coord")]
const FRAME_RESTORE_ACK: u32 = 11;
/// Ceiling on a carried RESTORE blob. `recv_raw`'s own 64 KiB cap guards control-byte reads and
/// is far below one rank's whole-state KV (tens of MB), so the blob is read in capped chunks —
/// this bounds what a peer can make us buffer, the job that cap was doing.
#[cfg(feature = "kv_coord")]
const MAX_CARRY_BLOB_BYTES: usize = 256 * 1024 * 1024;
/// H.1b (R2): CAPTURE whose body also carries the turn's TENANT (`capture_body_bytes_v2`). A new
/// frame code, not a wider `FRAME_CAPTURE` body: the v1 codec enforces an exact length and an
/// unknown/short frame here desyncs the stream (`TransportError::SocketClosed`) rather than
/// degrading. Only emitted for a non-empty tenant AND a chain that advertised support at HELLO.
#[cfg(feature = "kv_coord")]
const FRAME_CAPTURE_V2: u32 = 12;
/// Handshake schema version (spec §3.4).
const PROTO_VERSION: u32 = 1;
/// H.1b (R2) HELLO capability level for `FRAME_CAPTURE_V2`. Advertised in the payload (downstream
/// ranks NAK a peer that names a different level) and echoed in `HELLO_ACK`'s `pos` field so the
/// head learns the chain-wide FLOOR — an old build sends `pos = 0` there, which reads as "no v2"
/// and keeps the whole chain on v1 instead of desyncing it on an unknown frame kind mid-turn.
#[cfg(feature = "kv_coord")]
const CAPTURE_V2_CAP: u32 = 1;
#[cfg(not(feature = "kv_coord"))]
const CAPTURE_V2_CAP: u32 = 0;

/// Decision readout: rank 0 → chain, body = decision::encode_readout. Lockstep-acked.
const FRAME_DECIDE: u32 = 13;
const FRAME_DECIDE_ACK: u32 = 14;
/// Bounds on skipping late replies of earlier tasks (see `recv_reply_header`). One timed-out
/// exchange owes at most one reply, so 16 covers 16 timeouts in a row; the byte bound is two
/// worst-case DECISION bodies. The time bound is the exchange's own `recv_timeout`.
const MAX_STALE_REPLY_FRAMES: usize = 16;
const MAX_STALE_REPLY_BYTES: usize = 2 * crate::decision::MAX_DECISION_BODY_BYTES;

/// After the prefill FORWARDs: run the head on the last rank.
const FRAME_DECIDE_RUN: u32 = 15;
/// Last rank → rank 0 (middles relay verbatim), body = decision::encode_result.
const FRAME_DECISION: u32 = 16;

/// HELLO_ACK `pos` is a capability bit set. Bit 0 keeps the pre-existing capture_v2 meaning
/// (old ranks send exactly 0 or 1 there), bit 1 = this rank speaks the decision frames.
const CAP_CAPTURE_V2: u32 = 1;
const CAP_DECISION: u32 = 2;

/// The scalar stage input of a switchable-LoRA tree (an export with the removed
/// `--switchable-lora`). Such trees are refused at load ([`SWITCHABLE_REFUSAL`]).
const LORA_SCALE_INPUT: &str = "lora_scale";
/// The load error of a switchable-LoRA tree: nothing feeds its `lora_scale` input any more.
const SWITCHABLE_REFUSAL: &str =
    "switchable trees are no longer supported; re-export without --switchable-lora";

/// This rank's capability bits. CAP_DECISION only when the rank can take part in a decision pass
/// (`Qwen36Engine::serves_decisions`), so a chain with a rank that would refuse at DECIDE_RUN,
/// after a full chain prefill, is refused by rank 0 before any frame goes out.
fn own_caps(serves_decisions: bool) -> u32 {
    (CAPTURE_V2_CAP & CAP_CAPTURE_V2) | if serves_decisions { CAP_DECISION } else { 0 }
}

fn combine_caps(own: u32, down: u32) -> u32 {
    own & down
}

fn chain_decision(caps: u32) -> bool {
    caps & CAP_DECISION != 0
}

/// The one method chunked body reads need, on both ends of a pipeline link.
trait RawRecv {
    fn recv_raw(
        &mut self,
        n: usize,
    ) -> impl std::future::Future<Output = TransportResult<Vec<u8>>> + Send;
}

impl RawRecv for ActivationServer {
    fn recv_raw(
        &mut self,
        n: usize,
    ) -> impl std::future::Future<Output = TransportResult<Vec<u8>>> + Send {
        ActivationServer::recv_raw(self, n)
    }
}

impl RawRecv for ActivationClient {
    fn recv_raw(
        &mut self,
        n: usize,
    ) -> impl std::future::Future<Output = TransportResult<Vec<u8>>> + Send {
        ActivationClient::recv_raw(self, n)
    }
}

/// Read exactly `n` body bytes in ≤`MAX_RAW_BYTES` chunks, accepting at most `ceiling`.
/// `recv_raw` refuses a single over-cap read BEFORE consuming a byte, so an unchunked body read
/// (or an over-ceiling length rejected without draining) leaves the payload in the stream and
/// desyncs every later frame. A CAPTURE body passes 64 KiB at ≈16k tokens — reachable in any
/// long session, since `tokens = prompt ++ generated`. Over-ceiling lengths are drained
/// (discarded) to stay framed, then rejected.
async fn recv_body_chunked<G: RawRecv>(
    g: &mut G,
    n: usize,
    ceiling: usize,
    what: &str,
) -> Result<Vec<u8>, TransportError> {
    let accept = n <= ceiling;
    let mut buf = Vec::new();
    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(cascadia_transport::MAX_RAW_BYTES);
        let chunk = g.recv_raw(take).await?;
        if chunk.len() != take {
            return Err(TransportError::SocketClosed);
        }
        if accept {
            buf.extend_from_slice(&chunk);
        }
        remaining -= take;
    }
    if !accept {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "{what} length {n} exceeds {ceiling}; drained to stay framed"
        ))));
    }
    Ok(buf)
}

/// Read one `[u32 BE byte length][JSON]` decision body (FRAME_DECIDE / FRAME_DECISION), bounded
/// by [`crate::decision::MAX_DECISION_BODY_BYTES`]. Returned still length-prefixed: the decoders
/// and the verbatim relays take it whole.
async fn recv_decision_body<G: RawRecv>(g: &mut G, what: &str) -> Result<Vec<u8>, TransportError> {
    let lb = g.recv_raw(4).await?;
    if lb.len() != 4 {
        return Err(TransportError::SocketClosed);
    }
    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
    let mut body = lb;
    body.extend(recv_body_chunked(g, n, crate::decision::MAX_DECISION_BODY_BYTES, what).await?);
    Ok(body)
}

/// The one final chunk of a successful decision pass (single box and pipeline rank 0).
fn decision_final_chunk(id: TaskId, prompt_tokens: u32, logits: Vec<f32>) -> (TaskId, Chunk) {
    (
        id.clone(),
        Chunk::final_marker(id, "")
            .with_prompt_tokens(prompt_tokens)
            .with_finish_reason(FinishReason::Stop)
            .with_decision_logits(logits),
    )
}

/// A single-box head failure's reason. A coded refusal (`decision_unsupported: …`, …) passes
/// through unprefixed, since the API maps it by prefix (501, 413, …); anything else is a head
/// fault (a 503).
fn head_failure_reason(e: String) -> String {
    if e.starts_with("decision_") {
        e
    } else {
        format!("decision head failed: {e}")
    }
}

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

#[derive(Debug, serde::Deserialize)]
struct Manifest {
    /// HF model_type of the source IR. Any `qwen3_5`-PREFIXED type is
    /// accepted — `qwen3_5_moe` (Qwen3.5/3.6) and `qwen3_5` (dense
    /// Qwen3.8) are the ones that ship, and a later `qwen3_5*` cut of
    /// the same surgery drives the same way. Anything else is not a
    /// surgery tree this engine can drive.
    arch: String,
    /// Activation width the stage IRs exchange ([1, T, hidden_size]).
    /// Required except on `qwen3_5_moe`, the one arch whose trees were
    /// cut before the exporter wrote the key (always 2048 there) — see
    /// [`Manifest::hidden`].
    hidden_size: Option<usize>,
    /// Exporter sliced the last stage's logits to the final position
    /// ([1,1,vocab]); the engine then skips its own row slicing. Absent
    /// in pre-slice shard trees (default false).
    #[serde(default)]
    last_logits_only: bool,
    /// Last stage exposes output 1 = final-norm hidden [1, n, hidden] for decision heads
    /// (exporter `--final-hidden`). Absent in generation-only trees (default false).
    #[serde(default)]
    final_hidden_output: bool,
    stages: Vec<StageInfo>,
}

impl Manifest {
    fn parse(raw: &str) -> Result<Self, String> {
        let m: Manifest = serde_json::from_str(raw).map_err(|e| format!("manifest.json: {e}"))?;
        if !m.arch.starts_with("qwen3_5") {
            return Err(format!(
                "manifest arch {:?} is not a Qwen3.5-family surgery tree (qwen3_5 / qwen3_5_moe)",
                m.arch
            ));
        }
        // A missing key defaults to 2048 only for the arch that shipped without it. Anywhere else
        // (a dense tree with the key misspelled, say) the default is a wrong width that survives
        // load and dies at the first inference as an opaque OV shape error.
        if m.hidden_size.is_none() && m.arch != "qwen3_5_moe" {
            return Err(format!(
                "manifest.json: hidden_size is required for arch {:?}",
                m.arch
            ));
        }
        if m.hidden() == 0 {
            return Err("manifest hidden_size must be > 0".into());
        }
        // The chain is built in manifest order, so an entry that declares a different index would
        // silently chain the stages in the wrong order.
        for (i, s) in m.stages.iter().enumerate() {
            if s.stage != i {
                return Err(format!(
                    "manifest.json: stages are chained in manifest order, but entry {i} declares stage {}",
                    s.stage
                ));
            }
        }
        Ok(m)
    }

    /// Activation width. `parse` has rejected an absent key on every arch but `qwen3_5_moe`, whose
    /// pre-key trees are all 2048.
    fn hidden(&self) -> usize {
        self.hidden_size.unwrap_or(LEGACY_HIDDEN)
    }

    /// The tree was exported with a switchable LoRA: some stage lists a `lora_scale` input. Read
    /// from the manifest, so every rank of a chain refuses it, including one whose own stage has
    /// no such input.
    fn lora_switchable(&self) -> bool {
        self.stages.iter().any(StageInfo::lora_input)
    }
}

#[derive(Debug, serde::Deserialize)]
struct StageInfo {
    stage: usize,
    layer_start: u32,
    layer_end: u32,
    /// The stage IR's parameter names, as the exporter wrote them (absent in hand-written test
    /// manifests). A switchable-LoRA tree (refused) lists `lora_scale` on each stage with a branch.
    #[serde(default)]
    inputs: Vec<String>,
}

impl StageInfo {
    fn lora_input(&self) -> bool {
        self.inputs.iter().any(|n| n == LORA_SCALE_INPUT)
    }
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
    /// How `load` compiled the embeddings and each local stage, kept so a faulted stage can be
    /// recompiled the same way (Task 54 fix round 2).
    emb_recipe: Option<StageRecipe>,
    stage_recipes: Vec<StageRecipe>,
    tokenizer: Option<Tokenizer>,
    eos: Option<u32>,
    last_logits_only: bool,
    /// Activation width from the manifest (set by `load`).
    hidden: usize,
    /// Issue-34: KV-cache STORAGE precision. Distinct from the compute-precision hints noted at the
    /// PluginConfig site (those are f16-only on the fused MoE gemm and fail to compile).
    ///
    /// MEASURED INERT for this export, and it did NOT explain warm!=cold (that was the pre-restore
    /// state clear; see `restore_clear_mode`). Setting f32 left cold output BYTE-IDENTICAL to the
    /// default over a 48-token greedy turn and triggered NO recompile — neither is possible if a 35B
    /// model's KV storage precision actually changed. This export's KV are graph-level ReadValue/Assign
    /// variables (`cache_params.past.*`, see tools/qwen36_surgery/export_qwen36_moe.py), which the
    /// plugin property does not reach. Kept as an inert opt-in rather than deleted: the OV CPU
    /// StatefulSDPAFusion pass *can* absorb a stateful KV pattern into a plugin-managed cache, so this
    /// is an empirical fact about this OV version and export, not a structural guarantee.
    ///
    /// Do NOT re-derive "inert" from unchanged tensor byte-width — get_state reports the DECLARED type
    /// either way, so that observation is consistent with both readings and proves nothing.
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    /// User OV plugin properties from the CLI `--ov-*` flags, applied after the cache/KV/dyn-quant
    /// keys and forwarded verbatim. On Arc GPU, INFERENCE_PRECISION_HINT=f32 and
    /// EXECUTION_MODE_HINT=ACCURACY fail at compile (f16-only fused MoE gemm; see the NOTE in `load`).
    pub ov_properties: Vec<(String, String)>,
    /// OV compiled-blob cache. Without it this 35B MoE recompiles from scratch on EVERY spawn (and any
    /// plugin-property change forces a full uncached rebuild that can exceed the rig's serve window).
    /// runtime/gemma4/dist_spec all set this; qwen36 did not.
    pub cache_dir: Option<String>,
    /// Byte budget of the always-on prefix cache (`crate::prefix_cache`); 0 disables it.
    pub prefix_cache_bytes: usize,
    /// `<|im_start|>` id from the tokenizer (rank 0), the chat-boundary marker for snapshots.
    im_start_id: Option<u32>,
    /// `max_position_embeddings` from the tree's config.json (`None` = unknown, no guard).
    max_ctx: Option<usize>,
    /// Manifest `final_hidden_output` (set by `load`).
    final_hidden_output: bool,
    /// Decision head assets (`<model>/decision/`), loaded only on the head-owning process.
    decision_assets: Option<crate::decision::HeadAssets>,
    /// Vocabulary bound from config.json (every rank; tolerant, see `read_vocab`).
    vocab: Option<usize>,
    /// Lower bound on a prefix-cache snapshot's bytes from config.json (decision trees only; see
    /// `prefix_cache::config_state_floor`).
    state_floor: Option<(usize, usize)>,
}

impl Qwen36Builder {
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
    pub fn with_ov_properties(mut self, props: Vec<(String, String)>) -> Self {
        self.ov_properties.extend(props);
        self
    }
    /// Prefix-cache byte budget (0 = off). Default [`crate::prefix_cache::DEFAULT_PREFIX_CACHE_BYTES`].
    pub fn with_prefix_cache_bytes(mut self, bytes: usize) -> Self {
        self.prefix_cache_bytes = bytes;
        self
    }
    fn build_plugin_config(&self) -> PluginConfig {
        let mut plugin = PluginConfig::new();
        if let Some(d) = &self.cache_dir {
            plugin = plugin.with("CACHE_DIR", d);
        }
        if let Some(p2) = &self.kv_cache_precision {
            plugin = plugin.with("KV_CACHE_PRECISION", p2);
        }
        if let Some(g) = &self.dyn_quant_group {
            plugin = plugin.with("DYNAMIC_QUANTIZATION_GROUP_SIZE", g);
        }
        for (k, v) in &self.ov_properties {
            plugin = plugin.with(k, v);
        }
        plugin
    }
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
            emb_recipe: None,
            stage_recipes: Vec::new(),
            tokenizer: None,
            eos: None,
            last_logits_only: false,
            hidden: LEGACY_HIDDEN,
            kv_cache_precision: None,
            dyn_quant_group: None,
            ov_properties: Vec::new(),
            cache_dir: None,
            prefix_cache_bytes: crate::prefix_cache::DEFAULT_PREFIX_CACHE_BYTES,
            im_start_id: None,
            max_ctx: None,
            final_hidden_output: false,
            decision_assets: None,
            vocab: None,
            state_floor: None,
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
                    "qwen35 --total 1 runs all stages in-process; \
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
                "qwen35 --total 1 requires a single in-process stage chain".into(),
            ));
        }
        let dir = PathBuf::from(&self.shards_dir);
        let manifest_path = dir.join("manifest.json");
        let manifest_raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| EngineError::ModelNotFound(format!("{}: {e}", manifest_path.display())))?;
        let manifest = Manifest::parse(&manifest_raw).map_err(EngineError::InvalidConfig)?;
        // Before any compile: nothing feeds a switchable tree's `lora_scale` input any more, and
        // an unfed input would fail or misbehave at the first infer.
        if manifest.lora_switchable() {
            return Err(EngineError::InvalidConfig(SWITCHABLE_REFUSAL.into()));
        }
        if pipeline && manifest.stages.len() != self.total as usize {
            return Err(EngineError::ShardRejected(format!(
                "--total ({}) does not match manifest stage count ({})",
                self.total,
                manifest.stages.len()
            )));
        }

        let mut progress = vec![LoadProgress::message(format!(
            "qwen35 ({}, hidden {}): {} stages from {} (rank {}/{})",
            manifest.arch,
            manifest.hidden(),
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
        // KV-cache STORAGE precision is a DIFFERENT property from the compute hints above: it does not
        // touch the MoE gemm kernel, so it compiles where INFERENCE_PRECISION_HINT/EXECUTION_MODE_HINT
        // do not. Measured inert on this export and NOT the cause of warm!=cold — see the
        // `kv_cache_precision` field doc. Left wired so the knob behaves as declared if a future export
        // does route KV through a plugin-managed cache.
        let plugin = self.build_plugin_config();

        // Embeddings + tokenizer + eos live with the decode driver only.
        if self.rank == 0 {
            let emb_xml = dir.join("openvino_text_embeddings_model.xml");
            progress.push(LoadProgress::message(
                "compiling text-embeddings IR".to_string(),
            ));
            let recipe = StageRecipe::new(&emb_xml, &self.device, &plugin);
            self.emb = Some(compile_runtime(&recipe).map_err(map_ov)?);
            self.emb_recipe = Some(recipe);
            self.tokenizer = Some(
                Tokenizer::from_file(dir.join("tokenizer.json"))
                    .map_err(|e| EngineError::InvalidConfig(format!("tokenizer.json: {e}")))?,
            );
            self.im_start_id = self
                .tokenizer
                .as_ref()
                .and_then(|t| t.token_to_id("<|im_start|>"));
            if self.im_start_id.is_none() {
                warn!(
                    "qwen35: tokenizer has no <|im_start|>; the prefix cache has no chat boundary \
                     to snapshot at and stays empty (every turn cold)"
                );
            }
            self.max_ctx = read_max_ctx(&dir);
            self.eos = read_eos(&dir);
        }

        // Every rank: pipeline ranks other than 0 have no tokenizer, yet validate readouts too.
        // A missing config.json only matters to a decision tree; chat-only pipeline ranks
        // (manifest.json + stage<i>/) must not warn about it.
        let decision_tree = dir.join("decision.json").exists() || dir.join("decision").exists();
        self.vocab = read_vocab(&dir, decision_tree);
        // Only decision passes read the size estimate: a chat-only tree keeps its cache as is.
        self.state_floor = if decision_tree {
            read_state_floor(&dir)
        } else {
            None
        };
        let head_owner = owns_head(self.rank, self.total);
        self.decision_assets = if head_owner && dir.join("decision").join("head.json").exists() {
            progress.push(LoadProgress::message(
                "loading decision head assets".to_string(),
            ));
            let assets = crate::decision::HeadAssets::load(&dir, manifest.hidden())
                .map_err(EngineError::Backend)?;
            info!(
                head = assets.kind(),
                hidden = manifest.hidden(),
                "qwen35: loaded the decision head"
            );
            if assets.reads_hidden() && !manifest.final_hidden_output {
                warn!(
                    head = assets.kind(),
                    "qwen35: the tree was exported without --final-hidden; every decision pass \
                     will be refused (501)"
                );
            }
            Some(assets)
        } else {
            if head_owner && dir.join("decision.json").exists() {
                warn!(
                    "qwen35: decision.json is present but decision/head.json is not; every \
                     decision pass will be refused (501)"
                );
            }
            None
        };

        let mut stages = Vec::new();
        let mut recipes = Vec::new();
        for s in &manifest.stages {
            if pipeline && s.stage != self.rank as usize {
                continue;
            }
            let xml = dir.join(format!("stage{}", s.stage)).join("stage.xml");
            progress.push(LoadProgress::message(format!(
                "compiling stage{} (layers {}..{}) on {}",
                s.stage, s.layer_start, s.layer_end, self.device
            )));
            let recipe = StageRecipe::new(&xml, &self.device, &plugin);
            let rt = compile_runtime(&recipe).map_err(map_ov)?;
            // A manifest without stage inputs (hand-written, or older) cannot say: the IR can.
            if rt
                .input_names()
                .map_err(map_ov)?
                .iter()
                .any(|n| n == LORA_SCALE_INPUT)
            {
                return Err(EngineError::InvalidConfig(SWITCHABLE_REFUSAL.into()));
            }
            stages.push(rt);
            recipes.push(recipe);
        }
        self.stage_recipes = recipes;

        self.manifest_json = Some(manifest_raw);
        self.last_logits_only = manifest.last_logits_only;
        self.final_hidden_output = manifest.final_hidden_output;
        self.hidden = manifest.hidden();
        self.stages = Some(stages);
        progress.push(LoadProgress::ready());
        Ok(Box::pin(stream::iter(progress)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        if self.rank == 0 && (self.emb.is_none() || self.tokenizer.is_none()) {
            return Err(EngineError::NotLoaded);
        }
        // Single-process only: in pipeline mode the downstream ranks' state is not ours to snapshot
        // (the kv_coord plane covers that with CAPTURE/RESTORE frames).
        let prefix_cache_bytes = if self.total == 1 {
            self.prefix_cache_bytes
        } else {
            warn!(
                total = self.total,
                "qwen35: the prefix cache is single-process only; disabled for this pipeline \
                 (the kv_coord plane's CAPTURE/RESTORE covers multi-rank)"
            );
            0
        };
        info!(
            budget_gib = prefix_cache_bytes >> 30,
            im_start = ?self.im_start_id,
            max_ctx = ?self.max_ctx,
            "qwen35 prefix cache"
        );
        let stages: Vec<OvStage> = self
            .stages
            .ok_or(EngineError::NotLoaded)?
            .into_iter()
            .map(OvStage::Ov)
            .collect();
        let largest_leak_bytes = largest_leak_bytes(self.emb_recipe.as_ref(), &self.stage_recipes);
        let cap = recompile_cap(largest_leak_bytes);
        info!(
            total = self.total,
            local_stages = stages.len(),
            largest_leak_bytes,
            cap,
            "qwen36: stage recompile cap"
        );
        Ok(Box::new(Qwen36Engine {
            emb: self.emb.map(OvStage::Ov),
            stages,
            tokenizer: self.tokenizer,
            eos: self.eos,
            max_tokens_default: self.max_tokens_default,
            last_logits_only: self.last_logits_only,
            hidden: self.hidden,
            rank: self.rank,
            total: self.total,
            upstream: self.upstream,
            downstream: self.downstream,
            runtime_handle: self.runtime_handle,
            manifest_json: self.manifest_json.unwrap_or_default(),
            epoch: 0,
            peer_epoch: 0,
            handshake_done: false,
            chain_capture_v2: 0,
            chain_caps: 0,
            poisoned: None,
            down_link: DownLink::Clean,
            pending: Vec::new(),
            active: None,
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
            #[cfg(feature = "kv_coord")]
            plane_restore: std::env::var("CASCADIA_KV_PLANE_RESTORE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            state_restored: false,
            #[cfg(feature = "kv_coord")]
            kv_share: std::sync::Arc::new(std::sync::Mutex::new(
                crate::kv_coordination::OvKvCache::default(),
            )),
            #[cfg(feature = "kv_coord")]
            kv_handoff: std::sync::Arc::new(crate::kv_coordination::KvHandoffMailbox::new()),
            prefix_cache: new_prefix_cache(prefix_cache_bytes, self.state_floor),
            im_start_id: self.im_start_id,
            primed: false,
            stages_dirty: false,
            scrub_pending: false,
            decision_scrub_deferred: false,
            infer_failed: false,
            retry: None,
            #[cfg(test)]
            no_retry: false,
            engine_failed: None,
            failure: Default::default(),
            stage_idle_timeout: STAGE_IDLE_TIMEOUT,
            rebuild: Rebuild::new(self.emb_recipe, self.stage_recipes, cap),
            fresh_requests: Vec::new(),
            max_ctx: self.max_ctx,
            refresh_costs: Default::default(),
            last_restore_bytes: 0,
            final_hidden_output: self.final_hidden_output,
            decision_assets: self.decision_assets,
            vocab: self.vocab,
            want_final_hidden: false,
            last_final_hidden: None,
            pending_decision: None,
            last_logits: Vec::new(),
            timing: PassTiming::default(),
            last_timing: None,
            pass_started: None,
        }))
    }
}

/// A compiled IR the engine runs: OpenVINO when serving, a scripted stage in unit tests (a stub
/// build cannot compile one).
enum OvStage {
    Ov(Runtime),
    #[cfg(test)]
    Mock(tests::MockStage),
    /// The stage was leaked by fault recovery: handed to a bounded native call that never
    /// returned or could not start ([`bounded_call`]), faulted on a recompiled copy's first infer,
    /// replaced by a recompile that did not produce a fresh copy, or left over when the recovery
    /// failed the engine. Every call here fails. The engine is failed.
    Gone(&'static str),
}

macro_rules! ov_stage_call {
    ($s:expr, $m:ident($($a:expr),*)) => {
        match $s {
            OvStage::Ov(r) => r.$m($($a),*),
            #[cfg(test)]
            OvStage::Mock(m) => m.$m($($a),*),
            OvStage::Gone(what) => Err(cascadia_ov_genai_shim::Error::Native(format!(
                "this stage was lost to a native {what} that never returned"
            ))),
        }
    };
}

impl OvStage {
    fn input_names(&self) -> cascadia_ov_genai_shim::Result<Vec<String>> {
        ov_stage_call!(self, input_names())
    }
    fn set_input(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[usize],
        data: &[u8],
    ) -> cascadia_ov_genai_shim::Result<()> {
        ov_stage_call!(self, set_input(name, dtype, shape, data))
    }
    fn infer(&mut self) -> cascadia_ov_genai_shim::Result<()> {
        ov_stage_call!(self, infer())
    }
    fn output(&self, idx: usize) -> cascadia_ov_genai_shim::Result<(DType, Vec<usize>, Vec<u8>)> {
        ov_stage_call!(self, output(idx))
    }
    fn reset_state(&mut self) -> cascadia_ov_genai_shim::Result<()> {
        ov_stage_call!(self, reset_state())
    }
    fn recreate_request(&mut self) -> cascadia_ov_genai_shim::Result<()> {
        ov_stage_call!(self, recreate_request())
    }
    fn request_state(&self) -> cascadia_ov_genai_shim::Result<RequestState> {
        ov_stage_call!(self, request_state())
    }
    fn cancel(&mut self) -> cascadia_ov_genai_shim::Result<()> {
        ov_stage_call!(self, cancel())
    }
    fn wait_for(&mut self, timeout: Duration) -> cascadia_ov_genai_shim::Result<bool> {
        ov_stage_call!(self, wait_for(timeout))
    }
    fn abandon_request(&mut self) -> cascadia_ov_genai_shim::Result<()> {
        ov_stage_call!(self, abandon_request())
    }
    fn get_state_blob(&mut self) -> cascadia_ov_genai_shim::Result<Vec<u8>> {
        ov_stage_call!(self, get_state_blob())
    }
    fn set_state_blob(&mut self, blob: &[u8]) -> cascadia_ov_genai_shim::Result<()> {
        ov_stage_call!(self, set_state_blob(blob))
    }
}

/// Bring one stage request back to idle after a failed infer, without ever blocking on it (Task 46).
///
/// OpenVINO's request destructor waits, unbounded, on any run still outstanding, and its
/// `query_state`/`set_tensor` refuse a busy request. On the GPU a request stayed busy after
/// `unknown C++ exception in infer`, and the next recreate released it and never returned. So:
/// read the state (non-blocking); if it is not idle, `cancel()` and `wait_for` in slices until it
/// is, up to `timeout`; if it never gets there, abandon the request (the shim leaks it rather than
/// run its destructor) and return why. One `error` line per action: stage, action, outcome, ms.
fn make_idle(label: &str, st: &mut OvStage, timeout: Duration) -> Result<(), Stuck> {
    let t0 = Instant::now();
    let ms = |t0: Instant| t0.elapsed().as_millis() as u64;
    let state = match st.request_state() {
        Ok(RequestState::Idle) => {
            error!(
                stage = label,
                action = "probe",
                outcome = "idle",
                elapsed_ms = ms(t0),
                "qwen36: stage recovery"
            );
            return Ok(());
        }
        Ok(RequestState::None) => {
            error!(
                stage = label,
                action = "probe",
                outcome = "no request",
                elapsed_ms = ms(t0),
                "qwen36: stage recovery"
            );
            return Err(Stuck::NoRequest(format!("{label} has no infer request")));
        }
        Ok(s) => s,
        Err(e) => {
            error!(stage = label, action = "probe", outcome = "error", error = %e, elapsed_ms = ms(t0), "qwen36: stage recovery");
            return Err(Stuck::Failed(format!(
                "{label}'s request state is unreadable: {e}"
            )));
        }
    };
    match st.cancel() {
        Ok(()) => {
            error!(stage = label, action = "cancel", outcome = "ok", was = ?state, elapsed_ms = ms(t0), "qwen36: stage recovery")
        }
        Err(e) => {
            error!(stage = label, action = "cancel", outcome = "error", was = ?state, error = %e, elapsed_ms = ms(t0), "qwen36: stage recovery")
        }
    }
    let mut last = state;
    let mut polls = 0u32;
    loop {
        let left = timeout.saturating_sub(t0.elapsed());
        if left.is_zero() {
            break;
        }
        let slice = left.min(STAGE_IDLE_POLL);
        let w0 = Instant::now();
        polls += 1;
        // A failed wait (the request is gone) is read back as the state below.
        let _ = st.wait_for(slice);
        match st.request_state() {
            Ok(RequestState::Idle) => {
                error!(
                    stage = label,
                    action = "wait",
                    outcome = "idle",
                    polls,
                    elapsed_ms = ms(t0),
                    "qwen36: stage recovery"
                );
                return Ok(());
            }
            Ok(RequestState::None) | Err(_) => {
                last = RequestState::None;
                break;
            }
            Ok(s) => last = s,
        }
        // `wait_for` returns at once on a request with nothing outstanding: never spin.
        if let Some(rest) = slice.checked_sub(w0.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    error!(stage = label, action = "wait", outcome = ?last, polls, elapsed_ms = ms(t0), "qwen36: stage recovery");
    let why = format!(
        "{label}'s infer request was still {last:?} {} ms after a failed infer (cancel + wait)",
        ms(t0)
    );
    match st.abandon_request() {
        Ok(()) => {
            error!(
                stage = label,
                action = "abandon",
                outcome = "ok",
                elapsed_ms = ms(t0),
                "qwen36: stage recovery"
            );
            Err(Stuck::Abandoned(format!("{why}; abandoned")))
        }
        Err(e) => {
            error!(stage = label, action = "abandon", outcome = "error", error = %e, elapsed_ms = ms(t0), "qwen36: stage recovery");
            Err(Stuck::Failed(format!("{why}; abandon failed: {e}")))
        }
    }
}

/// What [`run_bounded`] (and [`bounded_call`]) got back.
enum Bounded<T> {
    /// The call returned in time.
    Done(T),
    /// The call did not return within the bound. Whatever it holds stays with its thread, leaked.
    TimedOut,
    /// The helper thread ended without an answer (it panicked): whatever it held is leaked.
    Died,
    /// No helper thread could be started.
    NoThread(std::io::Error),
}

/// A value a bounded call may still hand back after the engine stopped waiting: it is leaked,
/// never dropped. Dropping a stage would run native destructors (the request's and the compiled
/// model's) on a faulted device, from a stray thread.
struct Forget<T>(Option<T>);

impl<T> Drop for Forget<T> {
    fn drop(&mut self) {
        if let Some(v) = self.0.take() {
            std::mem::forget(v);
        }
    }
}

/// Run `f` on a helper thread named `qwen36-<what>`, and wait at most `timeout` for it (Task 54).
/// Its result is wrapped in [`Forget`] before it is sent, so a late result (or one left in the
/// channel) is leaked, not dropped.
fn run_bounded<T: Send + 'static>(
    what: &'static str,
    timeout: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> Bounded<T> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<Forget<T>>(1);
    let body = move || {
        let v = Forget(Some(f()));
        // Late (the engine stopped waiting): the send fails and `Forget` leaks the value.
        let _ = tx.send(v);
    };
    // Tests stand in for an OS that cannot start a thread: `spawn` then drops `body`, so do that.
    #[cfg(test)]
    if tests::spawn_refused(what) {
        drop(body);
        return Bounded::NoThread(std::io::Error::other("thread spawn refused (test)"));
    }
    let spawned = std::thread::Builder::new()
        .name(format!("qwen36-{what}"))
        .spawn(body);
    // A failed spawn drops `body` and with it `f`: whatever `f` owns must already be leak-safe
    // (`bounded_call` wraps its stage in `Forget` before handing it over).
    if let Err(e) = spawned {
        return Bounded::NoThread(e);
    }
    match rx.recv_timeout(timeout) {
        Ok(mut v) => match v.0.take() {
            Some(v) => Bounded::Done(v),
            None => Bounded::Died,
        },
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Bounded::TimedOut,
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Bounded::Died,
    }
}

/// Run one native call on `st` on a helper thread, bounded by `timeout` (Task 54, fix round 1).
///
/// After a recompile the engine's first infer runs on a stage nothing has proven yet; if the
/// device queue is wedged it could block forever while holding the engine. The stage moves to the
/// thread for the call (OpenVINO's own synchronous call, unchanged) and comes back when it
/// returns. Otherwise the slot holds [`OvStage::Gone`] and the stage is leaked, never dropped: with
/// its thread after a timeout or a panic, or at once when no thread could be started.
fn bounded_call<T: Send + 'static>(
    st: &mut OvStage,
    what: &'static str,
    timeout: Duration,
    f: impl FnOnce(&mut OvStage) -> cascadia_ov_genai_shim::Result<T> + Send + 'static,
) -> Bounded<cascadia_ov_genai_shim::Result<T>> {
    // Wrapped before it moves into the closure: a panic in `f`, or a thread that never starts
    // (the closure is then dropped unrun), leaks the stage instead of dropping it.
    let stage = Forget(Some(std::mem::replace(st, OvStage::Gone(what))));
    let r = run_bounded(what, timeout, move || {
        let mut stage = stage;
        let r = match stage.0.as_mut() {
            Some(s) => f(s),
            None => unreachable!(),
        };
        (stage, r)
    });
    match r {
        Bounded::Done((mut stage, r)) => {
            if let Some(s) = stage.0.take() {
                *st = s;
            }
            Bounded::Done(r)
        }
        Bounded::TimedOut => Bounded::TimedOut,
        Bounded::Died => Bounded::Died,
        Bounded::NoThread(e) => Bounded::NoThread(e),
    }
}

/// How `load` compiled one IR, so a faulted stage can be compiled again the same way (Task 54 fix
/// round 2): the same xml path, device and plugin config. qwen36 has one compile variant
/// ([`Runtime::compile`]), so there is nothing else to record.
#[derive(Clone, Debug)]
struct StageRecipe {
    xml: String,
    device: String,
    plugin: PluginConfig,
    /// Size of the IR's weight file (the `.bin` next to the xml; 0 when unreadable): what a leaked
    /// copy of this stage holds, logged as `leaked_bytes`.
    weight_bytes: u64,
}

impl StageRecipe {
    fn new(xml: &Path, device: &str, plugin: &PluginConfig) -> Self {
        Self {
            xml: xml.to_str().unwrap_or_default().to_string(),
            device: device.to_string(),
            plugin: plugin.clone(),
            weight_bytes: std::fs::metadata(xml.with_extension("bin"))
                .map(|m| m.len())
                .unwrap_or(0),
        }
    }
}

/// The most one recompile of this engine can leak: the largest weight file among its local slots
/// (the embeddings and its stages; 0 for a file whose size could not be read).
fn largest_leak_bytes(emb: Option<&StageRecipe>, stages: &[StageRecipe]) -> u64 {
    emb.into_iter()
        .chain(stages)
        .map(|r| r.weight_bytes)
        .max()
        .unwrap_or(0)
}

/// The recompile cap for an engine whose largest recompile leaks `largest_leak_bytes`:
/// [`LARGE_LEAK_MAX_RECOMPILES`] above [`LARGE_LEAK_BYTES`] (a 1-stage tree, a big pipeline
/// stage), [`MAX_RECOMPILES`] otherwise.
fn recompile_cap(largest_leak_bytes: u64) -> usize {
    if largest_leak_bytes > LARGE_LEAK_BYTES {
        LARGE_LEAK_MAX_RECOMPILES
    } else {
        MAX_RECOMPILES
    }
}

/// The compile `load` runs, and the one a recompile repeats.
fn compile_runtime(r: &StageRecipe) -> cascadia_ov_genai_shim::Result<Runtime> {
    Runtime::compile(&r.xml, &r.device, &r.plugin)
}

/// Builds a fresh stage from its recipe. [`compile_runtime`] when serving; tests inject a mock.
type StageFactory =
    Arc<dyn Fn(&StageRecipe) -> cascadia_ov_genai_shim::Result<OvStage> + Send + Sync>;

/// What the engine needs to recompile a faulted stage (Task 54 fix round 2).
struct Rebuild {
    emb: Option<StageRecipe>,
    stages: Vec<StageRecipe>,
    factory: StageFactory,
    /// [`RECOMPILE_TIMEOUT`]; shorter in tests.
    timeout: Duration,
    /// [`recompile_cap`]; changed in tests.
    cap: usize,
    /// Recompiles this engine has started.
    recompiled: usize,
    /// Recompiles that succeeded (the fresh copy is installed). The degraded warning keys on
    /// this ([`DEGRADED_AT_RECOMPILES`]).
    recompiles_ok: usize,
    /// Stage copies this engine's fault recovery leaked ([`leak_stage`] or a bounded call's
    /// `Forget`), whatever happened next: the faulted copy every recompile replaces, whether or
    /// not the recompile succeeds; a faulted copy left in its slot when the engine fails (cap
    /// reached, no recipe, or a request that cannot be abandoned); and a recompiled copy whose
    /// first infer failed or did not return.
    leaked_copies: usize,
    /// Sum of the leaked copies' `weight_bytes` (0 for a slot with no recipe).
    leaked_bytes: u64,
    /// Helper threads leaked by a timed-out recompile (not counted in `leaked_copies`).
    leaked_threads: usize,
    /// The degraded-engine warning has been logged (once per engine; tests read it).
    degraded_warned: bool,
}

impl Rebuild {
    fn new(emb: Option<StageRecipe>, stages: Vec<StageRecipe>, cap: usize) -> Self {
        Self {
            emb,
            stages,
            factory: Arc::new(|r: &StageRecipe| compile_runtime(r).map(OvStage::Ov)),
            timeout: RECOMPILE_TIMEOUT,
            cap,
            recompiled: 0,
            recompiles_ok: 0,
            leaked_copies: 0,
            leaked_bytes: 0,
            leaked_threads: 0,
            degraded_warned: false,
        }
    }

    /// Fault recovery leaked one copy of `slot`'s stage: add it to the engine's totals and to the
    /// process's `cascadia_engine_leaked_stage_bytes` gauge. Every leaked copy is counted, whether
    /// or not a recompile follows or succeeds.
    fn note_leaked(&mut self, slot: Slot) {
        let copy_bytes = self.recipe(slot).map_or(0, |r| r.weight_bytes);
        self.leaked_copies += 1;
        self.leaked_bytes += copy_bytes;
        cascadia_metrics::ENGINE_LEAKED_STAGE_BYTES
            .add(i64::try_from(copy_bytes).unwrap_or(i64::MAX));
    }

    /// A recompile succeeded (its old copy was already counted by [`Self::note_leaked`]): count
    /// it, and warn once at the [`DEGRADED_AT_RECOMPILES`]th. Measured on an Arc 140V: one leaked
    /// copy is harmless, two cost decode about 16%. Any failed step of a recovery fails the engine,
    /// so when this warns every leaked copy came from a recompile that succeeded.
    fn note_recompiled(&mut self) {
        self.recompiles_ok += 1;
        cascadia_metrics::ENGINE_STAGE_RECOMPILES_TOTAL.inc();
        if self.recompiles_ok >= DEGRADED_AT_RECOMPILES && !self.degraded_warned {
            self.degraded_warned = true;
            let (n, m, cap) = (self.leaked_copies, self.leaked_bytes, self.cap);
            let msg = degraded_message(n, m, self.leaked_threads);
            warn!(
                leaked_copies = n,
                leaked_bytes = m,
                cap,
                leaked_threads = self.leaked_threads,
                "{msg}"
            );
        }
    }

    fn recipe(&self, slot: Slot) -> Option<&StageRecipe> {
        match slot {
            Slot::Emb => self.emb.as_ref(),
            Slot::Stage(j) => self.stages.get(j),
        }
    }
}

/// The degraded-engine warning text. `threads` is the number of recompile helper threads a
/// timed-out recompile leaked (not stage copies), mentioned only when non-zero. Today it is always
/// zero when this is logged: a timed-out recompile fails the engine, so no later recompile
/// succeeds to warn.
fn degraded_message(copies: usize, bytes: u64, threads: usize) -> String {
    let mut m = format!(
        "qwen36: engine degraded: {copies} stage copies leaked by fault recovery ({bytes} bytes); \
         decode is slower until the server restarts (measured +16% with two on an Arc 140V); \
         restart recommended"
    );
    if threads > 0 {
        m.push_str(&format!(
            "; {threads} recompile helper thread(s) also leaked after a timeout"
        ));
    }
    m
}

/// Leak a stage's `Runtime` on purpose (never drop it: its destructor may wait forever on the
/// stuck request). Returns this process's count of leaked runtimes. The caller counts the copy
/// ([`Rebuild::note_leaked`]).
fn leak_stage(st: OvStage) -> usize {
    drop(Forget(Some(st)));
    LEAKED_RUNTIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// Why [`make_idle`] could not bring a request back to idle.
enum Stuck {
    /// Abandoned (leaked) after the timeout: the stage may be recompiled.
    Abandoned(String),
    /// There was no request to recover (the slot was already empty).
    NoRequest(String),
    /// The request is still in place and not idle, or its state is unknown: nothing may replace
    /// or release it, so the engine fails.
    Failed(String),
}

/// A slot the engine may recompile (Task 54): the embeddings, or a local stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Emb,
    Stage(usize),
}

impl Slot {
    fn label(self) -> String {
        match self {
            Slot::Emb => "emb".to_string(),
            Slot::Stage(j) => format!("stage{j}"),
        }
    }
}

/// A recompiled stage's first infer, bounded ([`bounded_call`], [`STAGE_IDLE_TIMEOUT`]), Task 54.
/// Ok on success; otherwise the infer's error and the reason the engine fails: the fresh copy
/// failed its first infer (the device itself is then suspect, so the engine fails rather than
/// recompile again), did not return within `timeout`, or its helper thread died or could not be
/// started. On every failure the fresh copy is leaked, never dropped (its request may be stuck):
/// the slot holds [`OvStage::Gone`], and the caller counts the copy ([`Rebuild::note_leaked`]).
fn first_infer_bounded(
    slot: Slot,
    st: &mut OvStage,
    timeout: Duration,
) -> Result<(), (cascadia_ov_genai_shim::Error, String)> {
    let label = slot.label();
    let t0 = Instant::now();
    let ms = |t0: Instant| t0.elapsed().as_millis() as u64;
    let why = match bounded_call(st, "first_infer", timeout, |s| s.infer()) {
        Bounded::Done(Ok(())) => {
            error!(
                stage = %label,
                action = "first_infer",
                outcome = "ok",
                elapsed_ms = ms(t0),
                "qwen36: stage recovery"
            );
            return Ok(());
        }
        Bounded::Done(Err(e)) => {
            error!(stage = %label, action = "first_infer", outcome = "failed", error = %e, elapsed_ms = ms(t0), "qwen36: stage recovery");
            // The copy came back from its thread, but its request just faulted: leak it.
            leak_stage(std::mem::replace(st, OvStage::Gone("first_infer")));
            let why = format!("{label} was recompiled and its first infer failed: {e}");
            error!(reason = %why, "qwen36: engine failed; every later task is refused until a restart");
            return Err((e, why));
        }
        Bounded::TimedOut => {
            error!(stage = %label, action = "first_infer", outcome = "timeout", elapsed_ms = ms(t0), "qwen36: stage recovery");
            format!(
                "{label} was recompiled and its first infer did not return within {} ms; the \
                 stage is leaked",
                timeout.as_millis()
            )
        }
        Bounded::Died => {
            error!(stage = %label, action = "first_infer", outcome = "helper thread died", elapsed_ms = ms(t0), "qwen36: stage recovery");
            format!(
                "{label} was recompiled and the helper thread running its first infer died; the \
                 stage is leaked"
            )
        }
        Bounded::NoThread(e) => {
            error!(stage = %label, action = "first_infer", outcome = "failed", error = %e, elapsed_ms = ms(t0), "qwen36: stage recovery");
            format!("{label} was recompiled and its first infer could not be run bounded: {e}")
        }
    };
    error!(reason = %why, "qwen36: engine failed; every later task is refused until a restart");
    Err((cascadia_ov_genai_shim::Error::Native(why.clone()), why))
}

fn map_ov(err: cascadia_ov_genai_shim::Error) -> EngineError {
    match err {
        cascadia_ov_genai_shim::Error::Stub => {
            EngineError::Backend("qwen35 requires the `openvino` feature (stub build)".into())
        }
        cascadia_ov_genai_shim::Error::Utf8(s) => EngineError::InvalidConfig(s),
        cascadia_ov_genai_shim::Error::Native(s) => EngineError::Backend(s),
    }
}

/// `max_position_embeddings` from the tree's config.json (`text_config` when nested). `None` when
/// the tree does not state it — the prompt-window guard in `submit` is then off, so say which of
/// the three ways that happened rather than letting an unreadable config disable it silently.
fn read_max_ctx(dir: &Path) -> Option<usize> {
    let raw = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(raw) => raw,
        Err(e) => {
            warn!(error = %e, "qwen35: no readable config.json; the prompt-window guard is OFF");
            return None;
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "qwen35: config.json is not valid JSON; the prompt-window guard is OFF");
            return None;
        }
    };
    let cfg = if v.get("text_config").is_some() {
        &v["text_config"]
    } else {
        &v
    };
    let max = cfg["max_position_embeddings"]
        .as_u64()
        .filter(|&m| m > 0)
        .map(|m| m as usize);
    if max.is_none() {
        warn!(
            "qwen35: config.json states no max_position_embeddings; the prompt-window guard is OFF"
        );
    }
    max
}

/// `text_config.vocab_size` (or top-level `vocab_size`) from the tree's config.json; `None` when
/// absent (with a warning when `warn_missing`: a decision tree). Only decision admission needs it.
fn read_vocab(dir: &Path, warn_missing: bool) -> Option<usize> {
    let raw = match std::fs::read_to_string(dir.join("config.json")) {
        Ok(r) => r,
        Err(e) => {
            if warn_missing {
                warn!(error = %e, "qwen35: no config.json; decision passes will be refused");
            }
            return None;
        }
    };
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let v = cfg["text_config"]["vocab_size"]
        .as_u64()
        .or_else(|| cfg["vocab_size"].as_u64());
    if v.is_none() && warn_missing {
        warn!("qwen35: config.json states no vocab_size; decision passes will be refused");
    }
    v.map(|v| v as usize)
}

/// `prefix_cache::config_state_floor` of the tree's config.json, with a warning when a decision
/// tree states too little to bound a snapshot (the first over-budget request then pays one pass
/// before its 413).
fn read_state_floor(dir: &Path) -> Option<(usize, usize)> {
    let cfg: serde_json::Value = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())?;
    let floor = crate::prefix_cache::config_state_floor(&cfg);
    if floor.is_none() {
        warn!(
            "qwen35: config.json does not state the attention dims; the prefix cache cannot bound \
             a decision snapshot's size before measuring one"
        );
    }
    floor
}

/// The engine's prefix cache, seeded with the load-time snapshot size floor when one is known, so
/// a decision snapshot that can never fit is refused before the first request prefills (P3-CE-1).
fn new_prefix_cache(
    budget_bytes: usize,
    floor: Option<(usize, usize)>,
) -> crate::prefix_cache::PrefixCache {
    let mut cache = crate::prefix_cache::PrefixCache::new(budget_bytes);
    if let Some((fixed, per_token)) = floor.filter(|_| cache.enabled()) {
        cache.seed_floor(fixed, per_token);
    }
    cache
}

/// This pass's snapshot plan. Only a `needs_snapshot` decision pass (the first pass of a request
/// admitted on the shared-prefix discount) bypasses the chat refresh rule: its later passes must
/// find that exact key (P2-CE-1). Every other pass, decision or chat, plans as a chat turn does,
/// so an unflagged decision pass does not pay a multi-GB copy for a few tokens (P3-REG-1). A
/// decision pass, flagged or not, also never copies a snapshot the cache can never hold
/// ([`crate::prefix_cache::PrefixCache::can_hold`]: the key's length and the size estimate),
/// which `insert` would only drop after the copy (P4-CE-2). Chat keeps its own rule unchanged.
fn plan_pass_snapshots(
    decision: Option<&cascadia_types::DecisionReadout>,
    prompt: &[u32],
    boundaries: &[usize],
    warm_prefix: usize,
    cache: &crate::prefix_cache::PrefixCache,
    costs: &crate::prefix_cache::RefreshCosts,
    state_bytes: usize,
) -> Vec<usize> {
    if decision.is_some_and(|d| d.needs_snapshot) {
        crate::prefix_cache::plan_decision_snapshots(prompt, boundaries, warm_prefix, cache)
    } else {
        let mut plan = crate::prefix_cache::plan_snapshots(
            prompt,
            boundaries,
            warm_prefix,
            cache,
            costs,
            state_bytes,
        );
        if decision.is_some() {
            plan.retain(|&b| cache.can_hold(b));
        }
        plan
    }
}

/// Decision passes prefill the codec's exact ids: no tokenization, no think block.
fn decision_admission(task: &GenerationTask) -> Option<Vec<u32>> {
    task.decision.as_ref().map(|d| d.prompt_ids.clone())
}

/// A decision pass snapshots only where its request's passes share a prefix (Kev: the state);
/// chat-style `<|im_start|>` boundaries would only evict chat entries. A prefix under
/// [`crate::prefix_cache::MIN_PREFIX_TOKENS`] is never a boundary: `insert` would drop it after
/// the copy (P4-CE-2), as [`crate::prefix_cache::chat_boundaries`] already filters for chat.
fn decision_snapshot_boundaries(d: &cascadia_types::DecisionReadout, cache_on: bool) -> Vec<usize> {
    match d.cache_prefix {
        Some(c) if cache_on && c as usize >= crate::prefix_cache::MIN_PREFIX_TOKENS => {
            vec![c as usize]
        }
        _ => Vec::new(),
    }
}

/// Rank `rank` of `total` runs the last stage (single box: always).
fn owns_head(rank: u32, total: u32) -> bool {
    total <= 1 || rank + 1 == total
}

/// The readout's head reads final-hidden rows (Pointer, Joint), not only the last logits.
fn head_reads_hidden(d: &cascadia_types::DecisionReadout) -> bool {
    d.head.hidden_need() != cascadia_types::HiddenNeed::None
}

/// Prompt bound a decision readout is validated against: the model window TIGHTENS the decision
/// cap, never widens it (Qwen3.5's 262144 window must not lift the 131072 decision bound).
fn decision_prompt_cap(max_ctx: Option<usize>) -> usize {
    max_ctx.map_or(cascadia_types::MAX_DECISION_PROMPT, |m| {
        m.min(cascadia_types::MAX_DECISION_PROMPT)
    })
}

/// A decision pass prefills exactly the codec's validated ids; an Option-B resume prefix would be
/// appended after them, unvalidated. Shared by single-box and pipeline rank-0 admission.
fn decision_resume_conflict(task: &GenerationTask) -> bool {
    task.decision.is_some() && task.resume_ids().is_some()
}

/// Last rank: the readout a FRAME_DECIDE announced, under its epoch, with the collector that
/// gathers the hidden rows its head reads over the prefill FORWARDs. `Err` = the readout was
/// refused; it is still acked (lockstep) and the error is reported at FRAME_DECIDE_RUN.
type PendingDecision = Option<(
    u32,
    Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String>,
)>;

/// The last rank's DECIDE_RUN lookup: always takes the entry (a stale one must not leak into
/// the next task), and turns a missing, stale or invalid readout into the error rank 0 reports.
fn take_pending(
    pending: &mut Option<(
        u32,
        Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String>,
    )>,
    epoch: u32,
) -> Result<(cascadia_types::DecisionReadout, crate::decision::Collector), String> {
    match pending.take() {
        Some((e, r)) if e == epoch => r,
        Some((e, _)) => Err(format!(
            "decision pending for epoch {e}, run asked for {epoch}"
        )),
        None => Err("no decision pending for this epoch".into()),
    }
}

/// The prefix-cache probe for a decision pass: restore no further than the first row/logit the
/// head reads (`warm_limit`; Joint: 0 = cold). `PrefixCache::longest_prefix` matches only STRICT
/// prefixes of the slice it is given, so the probe is one token longer than the limit: any hit is
/// at most `warm_limit` tokens.
fn decision_warm_probe<'a>(d: &cascadia_types::DecisionReadout, ids: &'a [u32]) -> &'a [u32] {
    match d.warm_limit() {
        0 => &ids[..0],
        lim => &ids[..(lim + 1).min(ids.len())],
    }
}

/// Prefix-cache refusals of a decision pass, checked after the restore and before any prefill
/// (SEC-3, P2-CE-1/2, P2-SEC-1/2). `need` is the request's shared prefix (`cache_prefix`).
/// - `decision_uncacheable` (the API's 413): the cache can never hold a snapshot at `need`
///   (disabled, under `MIN_PREFIX_TOKENS`, or measured/estimated over the budget), on the first
///   pass (`needs_snapshot`) of a request whose later passes must resume there, or on such a
///   later pass. Refusing the FIRST pass spends no prefill on a request that cannot finish.
/// - `decision_cold` (the API's 503, retry): a `require_warm` pass whose snapshot fits but is
///   gone (evicted by other requests' passes). Prefilling the prefix again would exceed the
///   budget the API admitted the request under.
fn decision_cache_refusal(
    d: &cascadia_types::DecisionReadout,
    warm_prefix: usize,
    cache: &crate::prefix_cache::PrefixCache,
) -> Option<String> {
    let need = d.cache_prefix? as usize;
    if !(d.needs_snapshot || d.require_warm) || warm_prefix >= need {
        return None;
    }
    if !cache.can_hold(need) {
        return Some(format!(
            "decision_uncacheable: a snapshot of the shared prefix ({need} tokens) cannot fit the \
             prefix cache"
        ));
    }
    d.require_warm.then(|| {
        format!(
            "decision_cold: the shared prefix snapshot ({need} tokens) is not cached; restored \
             {warm_prefix}"
        )
    })
}

/// The empty think block the official chat template injects for `enable_thinking=false`.
const THINK_BLOCK: &str = "\n<think>\n\n</think>\n\n";

/// The ids admission prefills for `task`, ahead of any Option-B resume prefix: the rendered
/// prompt plus the think block when hybrid reasoning is off (prefilling it makes decode start at
/// the answer instead of reasoning; legacy-rendered prompts only — a chat template injects it
/// itself, and the API passes `enable_thinking` into the render).
///
/// One implementation so `submit`'s window guard and `step_local` cannot size different sequences:
/// counting the bare prompt let a `max_ctx - 1` prompt through and then prefilled past
/// `max_position_embeddings`.
fn admitted_prompt_ids(tok: &Tokenizer, task: &GenerationTask) -> Result<Vec<u32>, String> {
    if let Some(ids) = decision_admission(task) {
        return Ok(ids);
    }
    let mut ids = tok
        .encode(task.prompt.as_str(), true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    if !task.enable_thinking && !task.prompt.trim_end().ends_with("</think>") {
        if let Ok(e) = tok.encode(THINK_BLOCK, false) {
            ids.extend(e.get_ids());
        }
    }
    Ok(ids)
}

fn map_wire(err: TransportError) -> EngineError {
    EngineError::Backend(format!("qwen36 pipeline wire: {err}"))
}

pub struct Qwen36Engine {
    /// Rank 0 only in pipeline mode.
    emb: Option<OvStage>,
    stages: Vec<OvStage>,
    /// Rank 0 only in pipeline mode.
    tokenizer: Option<Tokenizer>,
    eos: Option<u32>,
    max_tokens_default: u32,
    last_logits_only: bool,
    /// Activation width ([1, T, hidden]) the stage IRs exchange; sizes
    /// the dummy embeds, the FORWARD wire frame, and the relay-side
    /// frame check. From `manifest.hidden_size`.
    hidden: usize,
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
    /// H.1b R2: chain-wide `FRAME_CAPTURE_V2` capability floor, negotiated at HELLO (0 ⇒ some rank
    /// is a pre-R2 build). Gates emission so a tenant-bearing turn on a mixed chain falls back to
    /// the v1 frame — the workers then capture un-namespaced, which is cold-but-correct, instead of
    /// desyncing the stream on an unknown frame kind.
    chain_capture_v2: u32,
    /// Chain-wide HELLO_ACK capability bits (`CAP_*`), ANDed along the chain at HELLO. 0 before the
    /// handshake; `CAP_DECISION` clear ⇒ some rank predates the decision frames.
    chain_caps: u32,
    /// Set when the startup handshake found a config mismatch (handshake
    /// rule: refuse to serve). Admissions fail loud with this reason.
    poisoned: Option<String>,
    /// Framing state of the downstream link across failed reply exchanges (see [`DownLink`]).
    down_link: DownLink,
    pending: Vec<GenerationTask>,
    active: Option<ActiveTask>,
    /// Issue-34 Option C: opaque multi-stage KV blob cache for the coordination plane.
    #[cfg(feature = "kv_coord")]
    kv: crate::kv_coordination::OvKvCache,
    /// Issue-34 Option C: lock-free holder mirror of `kv` — the capture sites write both, and
    /// `kv_holder()` hands this out so a busy engine answers pulls without contending the engine lock.
    #[cfg(feature = "kv_coord")]
    kv_share: crate::kv_coordination::SharedKvCache,
    /// Issue-34 plane warm-resume: mailbox the plane's commit parks a pulled slice in. Drained from
    /// the `InFrame::Restore` handler — its lock is independent of the engine lock, which is what lets
    /// the commit path deposit while this engine is mid-`step()`.
    #[cfg(feature = "kv_coord")]
    kv_handoff: std::sync::Arc<crate::kv_coordination::KvHandoffMailbox>,
    /// Plane-restore MODE, parity with ov-runtime (`runtime.rs` `plane_restore`): downstream ranks
    /// warm-resume over the KV plane rather than from this rank's forwarded RESTORE. Read once from
    /// `CASCADIA_KV_PLANE_RESTORE` at build. **Observability only** — it no longer softens the chain
    /// verdict, because a plane rank now arms in-band inside its own `OPCODE_RESTORE` handler.
    #[cfg(feature = "kv_coord")]
    plane_restore: bool,
    /// A `set_state_blob` has been applied to the stages and not yet cleared. `reset_state` alone
    /// does not scrub that residue on this model (its DeltaNet recurrent states are fixed-shape, so
    /// there is no seq dim to collapse to zero), which left every post-migration turn serving
    /// garbage until the process restarted. Makes the next `reset_all` rebuild the InferRequest.
    state_restored: bool,
    /// Always-on snapshot LRU (`crate::prefix_cache`): system-block-end + chat-boundary state
    /// blobs (`prefix_cache::chat_boundaries`), restored at admission for a prompt that extends
    /// a cached prefix. Budget 0 ⇒ off.
    prefix_cache: crate::prefix_cache::PrefixCache,
    im_start_id: Option<u32>,
    /// The stage requests have executed at least once since their last reset/recreate. A
    /// `set_state_blob` onto a request that has NOT is silently discarded by the plugin on the
    /// next inference (measured on the GPU plugin: KV depth = tokens folded since, logits off by
    /// ~19), while the same restore onto a request that has run holds bit-exactly. Restores
    /// therefore prime un-run stages with one dummy fold first.
    primed: bool,
    /// The last `reset_all` failed on some stage: its state is unknown, so the prefix cache
    /// must not snapshot from it (a poisoned entry would serve every later hit). Cleared by
    /// the next successful reset.
    stages_dirty: bool,
    /// A task failed and the stage requests still hold its state: `finalize_error` leaves the
    /// scrub (`reset_all`) to the next admission, so the failed task's error chunk is returned
    /// first. On the GPU after a failed infer the scrub did block (Task 46: it released a request
    /// still marked busy); it now makes the requests idle first, bounded, or fails the engine.
    scrub_pending: bool,
    /// Task 55 (H-lite), single box only: a decision pass finished and its state is still in the
    /// stage requests, which stay primed. `finish_decision` sets it instead of running `reset_all`.
    /// The next admission either restores a prefix-cache entry over the live requests (a decision
    /// warm hit: `set_state_blob` overwrites every VariableState or fails, so no recreate and no
    /// priming fold are needed) or runs the deferred `reset_all` first
    /// ([`Self::scrub_deferred_decision`]): a recreate after a warm pass (`state_restored` is
    /// set), else the cheap `reset_state` of a cold pass. No prefill starts on the
    /// finished pass's state. Cleared by every `reset_all` and by a successful decision warm
    /// restore. Never set on a pipeline (`total > 1`).
    decision_scrub_deferred: bool,
    /// An infer on a stage (or the embeddings) returned an error since the last recovery: its
    /// request may still be marked busy, and OpenVINO blocks forever releasing a busy request.
    /// The next `reset_all` makes every request idle first ([`make_idle`]), Task 46.
    infer_failed: bool,
    /// Task 57, single box: the task [`Self::fail_local`] just re-queued at the front of `pending`
    /// after a native infer failure. Taken by the next admission, which is the retry and runs in
    /// the same `step`.
    retry: Option<Retry>,
    /// Tests of the failed-task path (Task 45/46/54): a native infer failure fails the task, as
    /// before Task 57, so the NEXT task runs the recovery.
    #[cfg(test)]
    no_retry: bool,
    /// Set when the engine cannot keep serving after a failed infer (Task 46, narrowed by Task
    /// 54): a request that could not be made idle could not be recovered either (the stage's
    /// recompile failed or timed out, or [`Rebuild::cap`] recompiles were already used), or a
    /// recompiled stage failed its first infer. Every later task fails at once with this reason (a
    /// `backend error`, a 5xx at the API), and nothing touches the stage requests again. Only a
    /// restart clears it.
    engine_failed: Option<String>,
    /// Task 56: rank 0 / single box marks this with the `engine_failed` reason
    /// ([`Self::fail_engine`]) so the server exits for a supervisor restart. Handed out by
    /// `failure()`, read without the engine lock. A relay rank exits through
    /// [`Self::relay_engine_failed`] instead and never marks it.
    failure: cascadia_engine::EngineFailure,
    /// [`STAGE_IDLE_TIMEOUT`]; shorter in tests.
    stage_idle_timeout: Duration,
    /// How to recompile a faulted stage, and how many times this engine has (Task 54 fix round 2).
    rebuild: Rebuild,
    /// Recompiled stages that have not yet completed an infer (Task 54). A failed first infer on
    /// one of them fails the engine.
    fresh_requests: Vec<Slot>,
    /// `max_position_embeddings` of the model (from the tree's config.json); prompts at or past it
    /// are refused at submit rather than prefilled for hours on the single batch=1 slot. `None`
    /// when the tree does not state it (guard off; `read_max_ctx` warns which case it was).
    max_ctx: Option<usize>,
    /// Measured prefill and snapshot-copy rates for the warm-turn refresh rule
    /// (`prefix_cache::RefreshCosts::refresh_pays`).
    refresh_costs: crate::prefix_cache::RefreshCosts,
    /// Bytes of the parts restored by the latest prefix-cache hit: the size estimate for the
    /// state a refresh snapshot of this conversation would copy.
    last_restore_bytes: usize,
    /// Manifest `final_hidden_output`: the last stage's output 1 is the final-norm hidden.
    final_hidden_output: bool,
    /// Decision head assets; `Some` only on the head-owning process of a tree with `decision/`.
    decision_assets: Option<crate::decision::HeadAssets>,
    /// Vocabulary bound decision readouts are validated against (`read_vocab`).
    vocab: Option<usize>,
    /// The active pass needs final hidden rows: `chain_pass_inner` copies output 1 only while
    /// set, so chat traffic on a decision tree never pays the [1, n, hidden] copy. Assigned on
    /// every single-box and rank-0 admission; on the pipeline last rank it is set at
    /// FRAME_DECIDE and cleared by RESET, RESTORE and DECIDE_RUN.
    want_final_hidden: bool,
    /// Final hidden of the latest `chain_pass` (when `want_final_hidden`), taken by `run_span`
    /// (single box) or by the pipeline last rank's FORWARD arm.
    last_final_hidden: Option<Vec<f32>>,
    /// Pipeline last rank: the readout FRAME_DECIDE announced (see [`PendingDecision`]).
    pending_decision: PendingDecision,
    /// Pipeline last rank: the latest FORWARD's last-position logits row, kept only while a
    /// decision is pending (a Verbalizer head reads it at FRAME_DECIDE_RUN).
    last_logits: Vec<f32>,
    /// Single box: the timings of the pass being served (`crate::pass_timing`), reset at each
    /// admission. Chat turns fill it too but never log it.
    timing: PassTiming,
    /// The timings of the last finished decision pass (read by tests).
    last_timing: Option<PassTiming>,
    /// Single box: when the pass being served was popped from `pending` (`PassTiming::total`).
    pass_started: Option<Instant>,
}

/// In-flight task state. `step()` advances one token per call so the
/// runner can interleave cancel() between steps (a monolithic step()
/// holds the engine mutex for the whole generation, making cancel
/// unreachable) and so streaming emits at the engine's real cadence.
struct ActiveTask {
    task_id: TaskId,
    /// H.1b R2: the KV namespace this turn belongs to, seeded from `GenerationTask.tenant` at
    /// admission. The capture path reads it from HERE, never from a plane-asserted partner — that
    /// value describes a pulled entry, not this turn.
    #[cfg_attr(not(feature = "kv_coord"), allow(dead_code))] // only the capture path reads it
    tenant: String,
    prompt_ids: Vec<u32>,
    prefill_idx: usize,
    step: usize,
    /// Tokens restored from a cached prefix at admission (0 = cold prefill).
    warm_prefix: usize,
    /// Single-box: prefill span boundaries (ascending) at which the chain state is snapshotted
    /// into the prefix cache — the end of the system block and the position before the prompt's
    /// last `<|im_start|>` (`prefix_cache::chat_boundaries`). Empty = no snapshot.
    snapshot_at: Vec<usize>,
    /// Single-box: last position's logits row. Unused in pipeline mode.
    logits: Vec<f32>,
    /// Pipeline rank 0: next token returned by the downstream argmax.
    next_token: Option<u32>,
    gen_ids: Vec<u32>,
    /// Bytes already handed to the client (the seed's decoded bytes on a
    /// resumed turn). Deltas come from the shim's `advance_emitted`, which
    /// holds back an unresolved U+FFFD run and RE-ANCHORS on a diverged
    /// re-decode — the old byte-offset arithmetic silently duplicated or
    /// dropped seam text when a diverged decode landed on a char boundary.
    emitted: Vec<u8>,
    /// Option B: leading entries of `gen_ids` that are the resume seed. The
    /// kv_coord capture key must SKIP them — `prompt_ids` already carries the
    /// resume ids (same defect dist_spec fixed).
    resume_seed_len: usize,
    max_tokens: usize,
    /// Set after admission (after any prefix-cache restore), so `started.elapsed()` at the end of
    /// prefill is prefill wall time plus the snapshot copies taken inside it.
    started: Instant,
    /// Seconds spent in prefix-cache snapshot copies during this turn's prefill, excluded from
    /// the prefill-rate measurement.
    snapshot_secs: f64,
    /// Pipeline rank 0: per-frame FORWARD→TOKEN round-trip times for
    /// decode frames (n=1), for the pipeline gate-4 wire histogram.
    wire_ms: Vec<f64>,
    /// Decision readout: `Some` ⇒ prefill only, then the head runs instead of decode.
    decision: Option<cascadia_types::DecisionReadout>,
    /// Hidden rows the decision head reads, gathered per prefill span (head-owning process).
    collector: Option<crate::decision::Collector>,
    /// Task 57, single box: the task as admitted, kept so a native infer failure can re-queue it
    /// once ([`Qwen36Engine::fail_local`]). `None` on pipeline rank 0, which never retries.
    retry_source: Option<GenerationTask>,
    /// Task 57: this admission is that retry, so its failure fails the task.
    retried: bool,
    /// Task 57: the tokens a retried chat's failed attempt generated, which the retry prefills as
    /// part of `prompt_ids`. Not prompt tokens to the client: the final chunk's `prompt_tokens`
    /// leaves them out, so usage matches an uninterrupted run.
    retry_resumed: usize,
}

/// Task 57: what a retried task's admission needs from its failed attempt.
struct Retry {
    task_id: TaskId,
    /// The bytes the failed attempt had already handed to the client; `None` for a decision pass,
    /// which emits nothing before its final chunk.
    emitted: Option<Vec<u8>>,
    /// The tokens the failed attempt generated, now part of the retry's prompt (`retry_resumed`).
    generated: usize,
}

impl ActiveTask {
    /// Where the task is: `prefill`, `head` (a decision pass past its prefill) or `decode`.
    fn phase(&self) -> &'static str {
        if self.prefill_idx < self.prompt_ids.len() {
            "prefill"
        } else if self.decision.is_some() {
            "head"
        } else {
            "decode"
        }
    }
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

/// Framing state of a rank's downstream link across failed reply exchanges.
///
/// Every reply frame (TOKEN, RESET_ACK, DECIDE_ACK, DECISION, RESTORE_ACK) carries the task
/// epoch in its header, and the downstream answers requests in order. So a reply that times out
/// while rank 0 is still waiting for its first byte leaves the stream frame-aligned: the late
/// reply arrives later, tagged with an epoch the next task does not use, and is skipped by
/// [`recv_reply_header`]. A timeout anywhere else (mid-request, mid-reply, or on a reply with no
/// task tag) leaves the stream at an unknown offset, and no tag can be read from there.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DownLink {
    /// Frame-aligned, nothing owed.
    Clean,
    /// Frame-aligned, but late replies of earlier exchanges may still arrive. The next exchange
    /// skips frames until its own `(kind, epoch)`; reading it proves nothing older is left.
    Owed,
    /// The startup HELLO timed out before its reply's first byte (a slow peer load). Its
    /// HELLO_ACK/NAK carries no task tag, so the retried handshake reads that reply instead of
    /// sending a second HELLO, whose reply would be left behind for the first task.
    HelloOwed,
    /// Alignment lost: every later exchange fails fast with this message until restart.
    Desynced(String),
}

/// What a reply that times out before its first byte leaves behind (see `run_exchange`).
#[derive(Clone, Copy)]
enum Late {
    /// Tagged with the task epoch: the next exchange skips it.
    Skip,
    /// The HELLO reply: the retried handshake waits for it.
    Await,
    /// Untagged (CAPTURE_ACK carries epoch 0): it cannot be told from a later reply. Only the
    /// kv_coord CAPTURE exchange uses it.
    #[cfg_attr(not(feature = "kv_coord"), allow(dead_code))]
    Desync,
}

/// How far one reply exchange got, read when its bound elapses. Shared by reference between the
/// exchange future and [`Qwen36Engine::run_exchange`].
struct ExchangePhase(std::sync::atomic::AtomicU8);

impl ExchangePhase {
    /// The request is not fully written.
    const SENDING: u8 = 0;
    /// The request is written and no reply byte has been read: still frame-aligned.
    const AWAITING: u8 = 1;
    /// A reply frame (ours or a stale one) has started.
    const READING: u8 = 2;

    fn new() -> Self {
        Self(std::sync::atomic::AtomicU8::new(Self::SENDING))
    }
    fn set(&self, p: u8) {
        self.0.store(p, std::sync::atomic::Ordering::Relaxed);
    }
    fn get(&self) -> u8 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Read the header of the reply owed for `(kind, epoch)`, marking `phase` so a timeout can tell
/// a frame-aligned wait from a cut frame. The first byte is read on its own for that reason:
/// `recv_raw`'s first read is cancel-safe, its remainder is not.
///
/// On a clean link (`drain` false) the next frame is returned as is and the caller checks it as
/// before. On an [`DownLink::Owed`] link, frames of earlier exchanges are skipped (body
/// included) until `(kind, epoch)` arrives, bounded by [`MAX_STALE_REPLY_FRAMES`] and
/// [`MAX_STALE_REPLY_BYTES`]; the caller's `recv_timeout` bounds the time.
async fn recv_reply_header<G: RawRecv>(
    g: &mut G,
    phase: &ExchangePhase,
    drain: bool,
    kind: u32,
    epoch: u32,
) -> Result<(u32, u32, u32), TransportError> {
    let mut frames = 0usize;
    let mut bytes = 0usize;
    loop {
        phase.set(ExchangePhase::AWAITING);
        let mut hb = g.recv_raw(1).await?;
        phase.set(ExchangePhase::READING);
        hb.extend(g.recv_raw(11).await?);
        let (k, e, p) = parse_header(&hb);
        if !drain || (k == kind && e == epoch) {
            return Ok((k, e, p));
        }
        frames += 1;
        if frames > MAX_STALE_REPLY_FRAMES {
            return Err(TransportError::Io(std::io::Error::other(format!(
                "more than {MAX_STALE_REPLY_FRAMES} stale replies before kind {kind} epoch {epoch}"
            ))));
        }
        bytes += 12 + skip_stale_body(g, k).await?;
        if bytes > MAX_STALE_REPLY_BYTES {
            return Err(TransportError::Io(std::io::Error::other(format!(
                "stale replies exceed {MAX_STALE_REPLY_BYTES} bytes"
            ))));
        }
        warn!(
            stale_kind = k,
            stale_epoch = e,
            kind,
            epoch,
            "qwen36: skipped a late reply of an earlier task"
        );
    }
}

/// Consume the body of a stale reply frame; returns its size. A kind with no known body layout
/// fails, which poisons the link.
async fn skip_stale_body<G: RawRecv>(g: &mut G, kind: u32) -> Result<usize, TransportError> {
    match kind {
        // Only replies a tagged exchange can leave behind: an untagged one (HELLO, CAPTURE)
        // that times out poisons the link instead of leaving it owed.
        FRAME_RESET_ACK | FRAME_DECIDE_ACK => Ok(0),
        #[cfg(feature = "kv_coord")]
        FRAME_RESTORE_ACK => Ok(0),
        FRAME_TOKEN => Ok(g.recv_raw(8).await?.len()),
        FRAME_DECISION => Ok(recv_decision_body(g, "stale decision body").await?.len()),
        other => Err(TransportError::Io(std::io::Error::other(format!(
            "unknown reply kind {other} while skipping late replies"
        )))),
    }
}

/// One inbound frame on the upstream session (downstream rank's view).
enum InFrame {
    Hello(Vec<u8>),
    Reset(u32),
    /// FRAME_DECIDE: the task's readout, still length-prefixed (`decision::encode_readout`).
    Decide {
        epoch: u32,
        body: Vec<u8>,
    },
    /// FRAME_DECIDE_RUN: prefill is done, run the head (last rank) or relay (middle).
    DecideRun(u32),
    Forward {
        epoch: u32,
        pos: u32,
        hidden: Vec<f32>,
        n: usize,
    },
    /// Issue-34 §8 CAPTURE: snapshot local KV under the head's content `kv_epoch` (body-carried).
    /// `partner` is `Some` only for `FRAME_CAPTURE_V2` (H.1b R2); `None` keeps the v1 frame's
    /// un-namespaced stash.
    #[cfg(feature = "kv_coord")]
    Capture {
        kv_epoch: u64,
        tokens: Vec<i32>,
        partner: Option<String>,
    },
    /// Issue-34 consume RESTORE: `set_state` the pulled slice stashed under `kv_epoch`. `task_epoch`
    /// (frame header) advances `peer_epoch` exactly like RESET. `blob` is the head's inline carry for
    /// the CROSS-chain case (this rank has no local capture for a foreign chain's epoch); empty on
    /// the same-chain path, where the rank restores from its own CAPTURE stash.
    #[cfg(feature = "kv_coord")]
    Restore {
        task_epoch: u32,
        kv_epoch: u64,
        blob: Vec<u8>,
    },
}

impl Qwen36Engine {
    /// Embed a token span: [1, n] ids -> flattened [1, n, hidden] f32.
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
        if self.fresh_requests.contains(&Slot::Emb) {
            // A recompiled stage's first infer: bounded, and any failure fails the engine (Task 54).
            self.fresh_requests.retain(|s| *s != Slot::Emb);
            if let Err((e, why)) = first_infer_bounded(Slot::Emb, emb, self.stage_idle_timeout) {
                self.rebuild.note_leaked(Slot::Emb);
                self.infer_failed = true;
                self.fail_engine(why);
                return Err(map_ov(e));
            }
        } else if let Err(e) = emb.infer() {
            self.infer_failed = true;
            return Err(map_ov(e));
        }
        let (dtype, shape, bytes) = emb.output(0).map_err(map_ov)?;
        if !matches!(dtype, DType::F32) {
            return Err(EngineError::Backend(format!(
                "embeddings output dtype {dtype:?}, expected f32"
            )));
        }
        // The IR's own width against the manifest's: a wrong `hidden_size` otherwise reaches the
        // stages as a shape the chain declares differently, and OV reports it as an opaque
        // reshape failure deep in stage0. Free here, and it fires at warmup.
        if shape.last() != Some(&self.hidden) {
            return Err(EngineError::Backend(format!(
                "embeddings output shape {shape:?} does not end in the manifest's hidden_size {}",
                self.hidden
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
        let out = self.chain_pass_inner(embeds, t0, t1)?;
        // Only a pass that ran every local stage counts as priming (see `primed`).
        self.primed = true;
        Ok(out)
    }

    fn chain_pass_inner(&mut self, embeds: &[f32], t0: usize, t1: usize) -> EngineResult<Vec<f32>> {
        let n = t1 - t0;
        let mask = vec![1i64; t1];
        let pos: Vec<i64> = (0..MROPE_ROWS)
            .flat_map(|_| (t0 as i64)..(t1 as i64))
            .collect();
        let width = self.hidden;
        let zeros_embeds = vec![0f32; n * width];
        let first_global = self.rank == 0;
        let mut hidden: Vec<f32> = embeds.to_vec();
        let n_stages = self.stages.len();
        let want_fh = self.want_final_hidden && self.final_hidden_output && self.owns_head();
        let mut final_hidden = None;
        for (j, st) in self.stages.iter_mut().enumerate() {
            let t_in = Instant::now();
            let names = st.input_names().map_err(map_ov)?;
            for name in &names {
                match name.as_str() {
                    "stage_hidden" => st
                        .set_input(name, DType::F32, &[1, n, width], &le_bytes_f32(&hidden))
                        .map_err(map_ov)?,
                    s if s.contains("embed") => {
                        // global first stage: real embeds; later stages:
                        // dummy (upstream ShapeOf chains read shapes only)
                        let data = if j == 0 && first_global {
                            &hidden
                        } else {
                            &zeros_embeds
                        };
                        st.set_input(name, DType::F32, &[1, n, width], &le_bytes_f32(data))
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
            self.timing.host_copy += t_in.elapsed();
            let t_inf = Instant::now();
            let fresh =
                !self.fresh_requests.is_empty() && self.fresh_requests.contains(&Slot::Stage(j));
            let r = if fresh {
                // A recompiled stage's first infer: bounded, and any failure fails the engine (Task 54).
                self.fresh_requests.retain(|s| *s != Slot::Stage(j));
                first_infer_bounded(Slot::Stage(j), st, self.stage_idle_timeout)
                    .map_err(|(e, why)| (e, Some(why)))
            } else {
                st.infer().map_err(|e| (e, None))
            };
            if let Some(s) = self.timing.stage_infer.get_mut(j) {
                *s += t_inf.elapsed();
            }
            if let Err((e, failed)) = r {
                if let Some(why) = failed {
                    self.rebuild.note_leaked(Slot::Stage(j));
                    self.fail_engine(why);
                }
                self.infer_failed = true;
                return Err(map_ov(e));
            }
            let t_out = Instant::now();
            let (dtype, _shape, bytes) = st.output(0).map_err(map_ov)?;
            if !matches!(dtype, DType::F32) {
                return Err(EngineError::Backend(format!(
                    "stage{j} output dtype {dtype:?}, expected f32"
                )));
            }
            if want_fh && j + 1 == n_stages {
                let (_, _, fh) = st.output(1).map_err(map_ov)?;
                final_hidden = Some(f32_from_le(&fh));
            }
            hidden = f32_from_le(&bytes);
            self.timing.host_copy += t_out.elapsed();
        }
        self.last_final_hidden = final_hidden;
        Ok(hidden)
    }

    /// Run a token span starting at absolute position `t0`; returns the
    /// LAST position's logits row. Single-box only (needs the full chain).
    fn run_span(&mut self, toks: &[u32], t0: usize) -> EngineResult<Vec<f32>> {
        let n = toks.len();
        let t_embed = Instant::now();
        let e = self.embed_seq(toks)?;
        self.timing.embed += t_embed.elapsed();
        self.timing.add_span(n);
        let out = self.chain_pass(&e, t0, t0 + n)?;
        if let Some(fh) = self.last_final_hidden.take() {
            if let Some(c) = self.active.as_mut().and_then(|t| t.collector.as_mut()) {
                c.absorb(t0, n, &fh)
                    .map_err(|e| EngineError::Backend(format!("decision_invalid: {e}")))?;
            }
        }
        if self.last_logits_only {
            // Exporter already sliced the last stage to the final
            // position; the output IS the last row.
            return Ok(out);
        }
        let row = out.len() / n;
        Ok(out[(n - 1) * row..].to_vec())
    }

    /// Make every request idle after a failed infer ([`make_idle`]), embeddings first. A request
    /// that never gets there is abandoned and its stage recompiled from the same IR ([`Rebuild`],
    /// Task 54 fix round 2), up to [`Rebuild::cap`] recompiles per engine. False when the engine
    /// is failed instead (`engine_failed`): a request that could not be abandoned, an abandon past
    /// the cap, or a recompile that failed, timed out or lost its helper thread. A stage that did
    /// not come back to idle is never dropped, not even with the engine: it is leaked, and counted
    /// ([`Rebuild::note_leaked`]), whether it is recompiled or the engine fails.
    fn recover_requests(&mut self) -> bool {
        let timeout = self.stage_idle_timeout;
        let emb = self.emb.iter_mut().map(|s| (Slot::Emb, s));
        let stages = self
            .stages
            .iter_mut()
            .enumerate()
            .map(|(i, s)| (Slot::Stage(i), s));
        'stages: for (slot, st) in emb.chain(stages) {
            let label = slot.label();
            let failed = 'slot: {
                let why = match make_idle(&label, st, timeout) {
                    Ok(()) => continue 'stages,
                    Err(Stuck::Failed(why)) => break 'slot why,
                    Err(Stuck::Abandoned(why) | Stuck::NoRequest(why)) => why,
                };
                let cap = self.rebuild.cap;
                let recompiled = self.rebuild.recompiled + 1;
                if recompiled > cap {
                    error!(
                        stage = %label,
                        action = "recompile",
                        outcome = "refused: cap",
                        elapsed_ms = 0u64,
                        recompiled = recompiled - 1,
                        cap,
                        leaked_bytes = 0u64,
                        process_leaked =
                            LEAKED_RUNTIMES.load(std::sync::atomic::Ordering::Relaxed),
                        "qwen36: stage recovery"
                    );
                    break 'slot format!(
                        "{why}; this engine has already recompiled {} stages, the cap of {cap}, \
                         so {label} is not recompiled",
                        recompiled - 1
                    );
                }
                let Some(recipe) = self.rebuild.recipe(slot).cloned() else {
                    error!(stage = %label, action = "recompile", outcome = "failed", error = "no recipe", "qwen36: stage recovery");
                    break 'slot format!("{why}; there is no recipe to recompile {label} from");
                };
                self.rebuild.recompiled = recompiled;
                // The old stage is never dropped, whatever the recompile does: its request may be
                // stuck, and a destructor would wait on it forever. It is counted now, so a
                // recompile that fails still shows in the totals and the gauge.
                let process_leaked = leak_stage(std::mem::replace(st, OvStage::Gone("recompile")));
                self.rebuild.note_leaked(slot);
                let leaked_bytes = recipe.weight_bytes;
                let bound = self.rebuild.timeout;
                let factory = Arc::clone(&self.rebuild.factory);
                let t0 = Instant::now();
                let r = run_bounded("recompile", bound, move || factory(&recipe));
                let elapsed_ms = t0.elapsed().as_millis() as u64;
                let (outcome, failure) = match r {
                    Bounded::Done(Ok(fresh)) => {
                        *st = fresh;
                        ("ok", None)
                    }
                    Bounded::Done(Err(e)) => (
                        "failed",
                        Some(format!("{why}; recompiling {label} failed: {e}")),
                    ),
                    Bounded::NoThread(e) => (
                        "failed",
                        Some(format!(
                            "{why}; recompiling {label} failed: no thread to bound it: {e}"
                        )),
                    ),
                    Bounded::Died => (
                        "failed",
                        Some(format!(
                            "{why}; recompiling {label} failed: its helper thread died"
                        )),
                    ),
                    Bounded::TimedOut => (
                        "timeout",
                        Some(format!(
                            "{why}; recompiling {label} timed out after {} ms",
                            bound.as_millis()
                        )),
                    ),
                };
                error!(
                    stage = %label,
                    action = "recompile",
                    outcome,
                    elapsed_ms,
                    recompiled,
                    cap,
                    leaked_bytes,
                    process_leaked,
                    leaked_requests = cascadia_ov_genai_shim::leaked_request_count(),
                    "qwen36: stage recovery"
                );
                if outcome == "timeout" {
                    self.rebuild.leaked_threads += 1;
                }
                match failure {
                    None => {
                        self.rebuild.note_recompiled();
                        self.fresh_requests.push(slot);
                        continue 'stages;
                    }
                    Some(why) => why,
                }
            };
            // The engine fails with this stage still in its slot (cap reached, no recipe, or a
            // request that could not be abandoned or read): leak it, so it is not dropped with the
            // engine. A recompile that failed already leaked it (the slot holds `Gone`).
            if !matches!(st, OvStage::Gone(_)) {
                let process_leaked = leak_stage(std::mem::replace(st, OvStage::Gone("recovery")));
                self.rebuild.note_leaked(slot);
                error!(
                    stage = %label,
                    action = "leak",
                    outcome = "ok",
                    elapsed_ms = 0u64,
                    leaked_bytes = self.rebuild.recipe(slot).map_or(0, |r| r.weight_bytes),
                    process_leaked,
                    "qwen36: stage recovery"
                );
            }
            error!(reason = %failed, "qwen36: engine failed; every later task is refused until a restart");
            self.fail_engine(failed);
            return false;
        }
        self.infer_failed = false;
        true
    }

    /// Fail the engine (`engine_failed`) with `why`. Rank 0 / single box also marks `failure`, so
    /// the server exits and the supervisor restarts it (Task 56); a relay rank already exits
    /// through [`Self::relay_engine_failed`].
    fn fail_engine(&mut self, why: String) {
        if self.rank == 0 {
            self.failure.set(why.clone());
        }
        self.engine_failed = Some(why);
    }

    /// The refusal every task gets once the engine has failed (`engine_failed`): at once, as a
    /// `backend error`.
    fn engine_failed_refusal(&self, task_id: &TaskId) -> Option<Vec<(TaskId, Chunk)>> {
        let why = self.engine_failed.as_ref()?;
        warn!(task = %task_id, reason = %why, "qwen36: refusing task (engine failed)");
        let reason =
            EngineError::Backend(format!("qwen36 engine failed: {why}; restart the server"))
                .to_string();
        Some(vec![(
            task_id.clone(),
            Chunk::error(task_id.clone(), reason),
        )])
    }

    /// A relay rank's engine failed (`engine_failed`): the connection-fatal error that makes
    /// `run_relay_loop` exit for a supervisor rebuild, which also closes the link to the head.
    fn relay_engine_failed(&self) -> Option<EngineError> {
        let why = self.engine_failed.as_ref()?;
        Some(EngineError::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionAborted,
            format!(
                "qwen36 engine failed: {why}; this pipeline stage exits so the supervisor \
                 rebuilds it"
            ),
        )))
    }

    fn reset_all(&mut self) {
        self.primed = false;
        // This is the scrub a failed task left, too (`scrub_failed_task`), and the one a finished
        // single-box decision pass deferred (`scrub_deferred_decision`).
        self.scrub_pending = false;
        self.decision_scrub_deferred = false;
        // Task 46: an abandoned request is never touched again (a call on it fails anyway, but
        // nothing here may release it), and its state is unknown.
        if self.engine_failed.is_some() {
            self.stages_dirty = true;
            return;
        }
        // After a failed infer a request may still be busy, and releasing (recreate) or resetting
        // it then blocks or fails: make every request idle first, or fail the engine.
        let recovering = self.infer_failed;
        if recovering && !self.recover_requests() {
            self.stages_dirty = true;
            return;
        }
        // After a restore, `reset_state` leaves residue on this model, so rebuild the request
        // instead. With the prefix cache always on that is every WARM turn — `state_restored` is
        // set by each hit — and the cheap reset is what a cold turn (no hit, nothing restored)
        // gets. A dirty engine escalates the same way: the cheap reset is exactly what failed.
        let recreate = self.state_restored || self.stages_dirty;
        let mut all_ok = true;
        let reset_started = Instant::now();
        for (i, st) in self.stages.iter_mut().enumerate() {
            // A stage this recovery just recompiled is fresh (default state, never run): nothing
            // to scrub. Its next device call is the bounded first infer (Task 54); only host-side
            // input_names/set_input run on it before that.
            if self.fresh_requests.contains(&Slot::Stage(i)) {
                error!(
                    stage = format!("stage{i}"),
                    action = if recreate { "recreate" } else { "reset" },
                    outcome = "skipped: recompiled",
                    elapsed_ms = 0u64,
                    "qwen36: stage recovery"
                );
                continue;
            }
            let t0 = Instant::now();
            let r = if recreate {
                st.recreate_request()
            } else {
                st.reset_state()
            };
            if recovering {
                error!(
                    stage = format!("stage{i}"),
                    action = if recreate { "recreate" } else { "reset" },
                    outcome = if r.is_ok() { "ok" } else { "error" },
                    elapsed_ms = t0.elapsed().as_millis() as u64,
                    "qwen36: stage recovery"
                );
            }
            if let Err(e) = r {
                all_ok = false;
                // A failed scrub leaves the donor's state live — every later turn serves garbage.
                // Loud, and the flag stays set so the next reset retries instead of silently
                // downgrading to the cheap path that cannot clear it.
                error!(error = %e, recreate, "qwen36: stage reset failed; state may be dirty");
            }
        }
        self.timing.reset = reset_started.elapsed();
        self.timing.recreate = recreate;
        if all_ok {
            self.state_restored = false;
        }
        // A failed scrub leaves whatever was in the requests live; no snapshot of it may enter the
        // prefix cache until a reset succeeds (a warm restore overwrites it and is still allowed).
        self.stages_dirty = !all_ok;
    }

    /// Finish the active task: reset state, log, emit the final marker.
    ///
    /// `reason` is why decode stopped and rides on the final marker: without
    /// it the API layer falls back to its `"stop"` default, so a turn cut off
    /// at `max_tokens` (or at the context window, which admission folds into
    /// `max_tokens`) was indistinguishable from a natural EOS.
    fn finalize(&mut self, reason: FinishReason) -> Vec<(TaskId, Chunk)> {
        let Some(t) = self.active.take() else {
            return Vec::new();
        };
        // Issue-34: capture this rank's local KV under (prompt + generated) BEFORE reset_all wipes
        // it. Keyed by the full sequence (session-resume). Gated + best-effort (stub ⇒ no-op).
        #[cfg(feature = "kv_coord")]
        {
            // Skip the resume seed: `prompt_ids` already carries those ids, so
            // counting them again keys the blob at a depth that never matches
            // the fed sequence (same defect dist_spec fixed).
            let tokens: Vec<i32> = t
                .prompt_ids
                .iter()
                .chain(t.gen_ids.iter().skip(t.resume_seed_len))
                .map(|&u| u as i32)
                .collect();
            // Pipeline head: broadcast CAPTURE so every downstream rank snapshots its slice under the
            // same content epoch (workers have no tokens). Best-effort. rank0's own slice is stashed
            // token-keyed by kv_capture_local below (the NEGOTIATE/offers path).
            if self.total > 1 && self.rank == 0 && self.downstream.is_some() && !tokens.is_empty() {
                let kv_epoch = crate::kv_coordination::synth_epoch(&tokens);
                if let Err(e) = self.forward_capture_downstream(kv_epoch, &tokens, &t.tenant) {
                    warn!(error = %e, "qwen36: CAPTURE broadcast failed (best-effort)");
                }
            }
            self.kv_capture_local(&t.tenant, tokens);
        }
        // Measured BEFORE the reset: the scrub is teardown, not generation, and
        // folding it in was deflating the reported decode rate.
        let elapsed = t.started.elapsed().as_secs_f64();
        let tok_s = if elapsed > 0.0 {
            t.gen_ids.len() as f64 / elapsed
        } else {
            0.0
        };
        // Which branch `reset_all` will take, sampled before the call clears
        // `state_restored`. With the prefix cache always on, every warm turn
        // pays `recreate_request()` per stage — `reset_s` is what that costs.
        let recreate = self.state_restored || self.stages_dirty;
        let reset_t0 = Instant::now();
        self.reset_all();
        let reset_s = reset_t0.elapsed().as_secs_f64();
        info!(
            task = %t.task_id,
            prompt_tokens = t.prompt_ids.len() - t.retry_resumed,
            warm_prefix = t.warm_prefix,
            tokens = t.gen_ids.len(),
            elapsed_s = elapsed,
            tok_s,
            reset_s,
            recreate,
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
            Chunk::final_marker(t.task_id, "")
                .with_prompt_tokens((t.prompt_ids.len() - t.retry_resumed) as u32)
                .with_finish_reason(reason),
        )]
    }

    /// Drop every piece of per-task decision state (pipeline last rank and single box alike). Called
    /// wherever a new task starts on a downstream rank (RESET, RESTORE) and after DECIDE_RUN, so a
    /// decision that rank 0 aborted after DECIDE was acked cannot leave a later chat turn paying the
    /// `[1, n, hidden]` final-hidden copy or the collector's absorb.
    fn clear_decision_state(&mut self) {
        self.want_final_hidden = false;
        self.last_final_hidden = None;
        self.pending_decision = None;
        self.last_logits = Vec::new();
    }

    /// This process runs the last stage, where the final hidden states and the decision head live.
    fn owns_head(&self) -> bool {
        owns_head(self.rank, self.total)
    }

    /// The rank can take part in a decision pass. Rank 0 and the head-owning (last) rank validate
    /// readouts against their vocab, and the head owner also needs head assets; a middle rank
    /// only relays FRAME_DECIDE, so it needs neither (P2-REG-1). Advertised in the HELLO
    /// capability bits.
    fn serves_decisions(&self) -> bool {
        let reads_readouts = self.rank == 0 || self.owns_head();
        !reads_readouts
            || (self.vocab.is_some() && (!self.owns_head() || self.decision_assets.is_some()))
    }

    /// Rank 0's up-front refusal of a decision pass: a malformed readout, or a chain in which some
    /// rank (an older build, or one without config.json vocab_size or decision/ assets) cannot
    /// serve it. Refused before any frame of the task goes out.
    fn pipe_decision_refusal(&self, d: &cascadia_types::DecisionReadout) -> Option<String> {
        if d.require_warm || d.needs_snapshot {
            // Pipeline ranks never resume a decision prefix (the API sets these only with reuse).
            return Some(
                "decision_invalid: require_warm/needs_snapshot need the single-box prefix cache"
                    .into(),
            );
        }
        self.admit_decision_check(d).err().or_else(|| {
            (!chain_decision(self.chain_caps)).then(|| {
                "decision_unsupported: a pipeline rank cannot serve decision passes (an older \
                 build, or no config.json vocab_size or decision/ head assets on that rank)"
                    .to_string()
            })
        })
    }

    /// End a decision pass: run the head on the collected state, emit ONE final chunk, reset.
    fn finish_decision(&mut self) -> Vec<(TaskId, Chunk)> {
        self.want_final_hidden = false;
        self.last_final_hidden = None;
        let Some(t) = self.active.as_ref() else {
            return Vec::new();
        };
        let d = t.decision.as_ref().expect("decision task");
        let empty = crate::decision::Collector::new(cascadia_types::HiddenNeed::None, self.hidden);
        let col = t.collector.as_ref().unwrap_or(&empty);
        let d_kind = match &d.head {
            cascadia_types::ReadoutHead::Verbalizer { .. } => "verbalizer",
            cascadia_types::ReadoutHead::Pointer { .. } => "pointer",
            cascadia_types::ReadoutHead::Joint { .. } => "joint",
        };
        let t_head = Instant::now();
        let result = match self.decision_assets.as_mut() {
            Some(a) => crate::decision::run_head(&d.head, a, &d.prompt_ids, &t.logits, col),
            None => Err("decision_unsupported: no head assets".into()),
        };
        self.timing.head = t_head.elapsed();
        match result {
            Ok(logits) => {
                let n = t.prompt_ids.len() as u32;
                let id = t.task_id.clone();
                self.active = None;
                if self.total <= 1 {
                    // Task 55 (H-lite): keep the primed requests for the next pass. A warm hit
                    // restores over them; every other admission runs this scrub first.
                    self.decision_scrub_deferred = true;
                } else {
                    self.reset_all();
                }
                if let Some(started) = self.pass_started.take() {
                    self.timing.total = started.elapsed();
                }
                let t = &self.timing;
                tracing::debug!(
                    target: "cascadia::perf",
                    task = %id, head = d_kind, prompt = t.prompt, warm_prefix = t.warm_prefix,
                    prefilled = t.prefilled(), spans = t.spans, span_max = t.span_max,
                    admit_ms = ms(t.admit), admit_reset_ms = ms(t.admit_reset),
                    restore_ms = ms(t.restore), prime_ms = ms(t.prime),
                    set_state_ms = ms(t.set_state), snapshot_ms = ms(t.snapshot),
                    embed_ms = ms(t.embed), infer_ms = ms(t.infer()), stage_ms = %t.stage_ms(),
                    host_copy_ms = ms(t.host_copy), head_ms = ms(t.head), reset_ms = ms(t.reset),
                    recreate = t.recreate, total_ms = ms(t.total),
                    "qwen36 decision pass done"
                );
                self.last_timing = Some(self.timing.clone());
                vec![decision_final_chunk(id, n, logits)]
            }
            Err(e) => self.fail_local(head_failure_reason(e)),
        }
    }

    /// Everything a decision pass needs before it touches a stage: a well-formed readout
    /// (ids < vocab, spans/positions inside the prompt: the readout may come from a peer),
    /// the hidden output its head reads (manifest-wide, so checked on every rank), and, on the
    /// head-owning process, the head assets.
    fn admit_decision_check(&self, d: &cascadia_types::DecisionReadout) -> Result<(), String> {
        let vocab = self
            .vocab
            .ok_or("decision_unsupported: tree config.json states no vocab_size")?;
        d.validate(vocab, decision_prompt_cap(self.max_ctx))
            .map_err(|e| format!("decision_invalid: {e}"))?;
        if self.owns_head() && self.decision_assets.is_none() {
            return Err("decision_unsupported: model has no decision/ head assets".into());
        }
        // Every rank evaluates this (HELLO guarantees one manifest across the chain), so rank 0
        // refuses before any frame goes out instead of the last rank refusing after a full chain
        // prefill. The last rank's own check stays as defence in depth.
        if head_reads_hidden(d) && !self.final_hidden_output {
            return Err("decision_unsupported: tree was exported without --final-hidden".into());
        }
        // Joint (Clef) inputs must fit the shim's per-input limit, or set_input fails after a
        // full prefill. `hidden` comes from the shared manifest, so rank 0 refuses up front too.
        if let cascadia_types::ReadoutHead::Joint { questions } = &d.head {
            let options: usize = questions.iter().map(|q| q.options.len()).sum();
            cascadia_types::joint_inputs_fit(
                d.prompt_ids.len(),
                questions.len() + options,
                options,
                self.hidden,
            )
            .map_err(|e| format!("decision_too_long: {e}"))?;
        }
        Ok(())
    }

    /// Terminate the active task as FAILED rather than completed: clear
    /// engine state (as `finalize` does) but emit an error chunk so the
    /// API returns a 5xx instead of a 200 with whatever partial text was
    /// streamed. For mid-generation backend/wire errors that previously
    /// fell through to `finalize` and read as a clean (empty/partial)
    /// success.
    /// Fail the active task: one warn line (task, phase, prompt length, the prefix it resumed
    /// from the cache), and its error chunk, now, carrying that prefix (`warm_prefix`, 0 = cold)
    /// so the API's own line can report it. The stage requests are scrubbed when the next task
    /// is admitted ([`Self::scrub_failed_task`]), not here: the chunk only reaches its stream
    /// once `step` returns, so a scrub that blocks after a failed native call (a GPU infer that
    /// threw) would keep the error from a request that is still waiting for it. That scrub makes
    /// every request idle first and is bounded by `stage_idle_timeout` (Task 46, `reset_all`).
    fn finalize_error(&mut self, reason: String) -> Vec<(TaskId, Chunk)> {
        self.want_final_hidden = false;
        self.last_final_hidden = None;
        let Some(t) = self.active.take() else {
            return Vec::new();
        };
        warn!(
            task = %t.task_id,
            phase = t.phase(),
            prompt = t.prompt_ids.len(),
            warm_prefix = t.warm_prefix,
            generated = t.gen_ids.len(),
            reason = %reason,
            "qwen36 task failed"
        );
        self.scrub_pending = true;
        let chunk = Chunk::error(t.task_id.clone(), reason).with_warm_prefix(t.warm_prefix);
        vec![(t.task_id, chunk)]
    }

    /// Task 57: fail the active single-box task, or retry it once. A native infer failure
    /// (`infer_failed` set, the engine not failed) of a task that has not been retried yet
    /// re-queues it at the front of `pending` and admits it again in this same `step`: the
    /// admission runs the failed task's scrub first (`scrub_failed_task`: the Task 46/54 recovery,
    /// bounded) and then the prefill, exactly as a new admission would. A chat resumes over the
    /// tokens it already generated (Option B, so `max_tokens` still bounds prefix + new) and
    /// carries the bytes already emitted; a decision pass re-runs unchanged. Greedy decode is
    /// deterministic and a recovered stage answers as before, so the client sees only latency, and
    /// no chunk of the failed attempt. Re-admitting here, not at the next step, keeps the retried
    /// task's stream inside the runner's empty-step limit (`MAX_CONSECUTIVE_EMPTY_STEPS`, 3): a
    /// decision pass's head failure is its second empty step, and the retried prefill rides it.
    /// Anything else, and a failure of the retry, fails the task (`finalize_error`); the
    /// retry's reason says so. Pipeline rank 0 never comes here (`finalize_error` directly).
    fn fail_local(&mut self, reason: String) -> Vec<(TaskId, Chunk)> {
        let Some(t) = self.active.as_ref() else {
            return Vec::new();
        };
        if t.retried {
            return self.finalize_error(format!("{reason} (on the retry after an infer failure)"));
        }
        let native = self.infer_failed && self.engine_failed.is_none();
        #[cfg(test)]
        let native = native && !self.no_retry;
        if self.total > 1 || !native || t.retry_source.is_none() {
            return self.finalize_error(reason);
        }
        let mut t = self.active.take().expect("checked above");
        self.want_final_hidden = false;
        self.last_final_hidden = None;
        let mut task = t.retry_source.take().expect("checked above");
        // A decision pass emits nothing before its final chunk and refuses a resume prefix
        // (`decision_resume_conflict`), so it re-runs as it came. A chat resumes over every token
        // it generated, any resume seed it came with included (`gen_ids` starts with it).
        let emitted = if t.decision.is_none() {
            task.resume_token_ids =
                (!t.gen_ids.is_empty()).then(|| t.gen_ids.iter().map(|&id| id as i32).collect());
            Some(std::mem::take(&mut t.emitted))
        } else {
            None
        };
        let generated = t.gen_ids.len() - t.resume_seed_len;
        warn!(
            task = %t.task_id,
            phase = t.phase(),
            prompt = t.prompt_ids.len(),
            warm_prefix = t.warm_prefix,
            generated = t.gen_ids.len(),
            attempt = 2,
            reason = %reason,
            "qwen36: retrying task after infer failure"
        );
        self.scrub_pending = true;
        self.retry = Some(Retry {
            task_id: task.task_id.clone(),
            emitted,
            generated,
        });
        self.pending.insert(0, task);
        self.step_local()
    }

    /// The scrub a failed task left (see [`Self::finalize_error`]), before anything reads or
    /// restores over the stage requests: every admission runs it first. Any `reset_all` in
    /// between does the same job and clears it.
    fn scrub_failed_task(&mut self) {
        if self.scrub_pending {
            self.reset_all();
        }
    }

    /// The scrub a finished single-box decision pass deferred (Task 55, see
    /// `decision_scrub_deferred`). Every admission and restore runs it before it prefills or
    /// writes state, except a decision pass's prefix-cache hit in the default restore mode,
    /// which overwrites every state of the live, primed requests instead (`local_warm_or_cold`).
    /// This `reset_all` recreates after a warm pass (`state_restored` is set), else it is the
    /// cheap `reset_state` of a cold pass (no restored state, so no residue).
    fn scrub_deferred_decision(&mut self) {
        if self.decision_scrub_deferred {
            self.reset_all();
        }
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
        #[cfg_attr(not(feature = "kv_coord"), allow(unused_mut))]
        let mut v = serde_json::json!({
            "proto": PROTO_VERSION,
            "total": self.total,
            "wire": "f32",
            "manifest": self.manifest_json,
        });
        // Advertised only by a build that can actually speak v2. A build without `kv_coord` has no
        // CAPTURE path at all, so it must read to its peers as a LEGACY rank (key absent ⇒ chain
        // floor 0 ⇒ v1) rather than as an explicit disagreement that would NAK the handshake.
        #[cfg(feature = "kv_coord")]
        {
            v["capture_v2"] = serde_json::json!(CAPTURE_V2_CAP);
        }
        v.to_string().into_bytes()
    }

    /// Rank 0: HELLO → HELLO_ACK/NAK before the first admit.
    /// Middles chain the payload on, so one ACK means the whole chain
    /// validated against this rank's manifest.
    fn handshake_a(&mut self) -> EngineResult<()> {
        let payload = self.hello_payload();
        let (nak, down_cap) = self.forward_hello_downstream(&payload)?;
        if let Some(reason) = nak {
            self.poisoned = Some(reason.clone());
            return Err(EngineError::Backend(format!(
                "qwen36 pipeline handshake refused: {reason}"
            )));
        }
        let caps = combine_caps(own_caps(self.serves_decisions()), down_cap);
        self.chain_caps = caps;
        self.chain_capture_v2 = (caps & CAP_CAPTURE_V2).min(CAPTURE_V2_CAP);
        self.handshake_done = true;
        info!(
            caps,
            capture_v2 = self.chain_capture_v2,
            "qwen36 pipeline handshake ok"
        );
        Ok(())
    }

    /// Fail fast on a desynced downstream link; otherwise whether the next exchange must first
    /// skip late replies of earlier ones.
    fn link_ready(&self) -> EngineResult<bool> {
        match &self.down_link {
            DownLink::Clean => Ok(false),
            DownLink::Owed => Ok(true),
            // Only the handshake retry may run on a link whose HELLO reply is still owed: a
            // tagged exchange would read that untagged reply as its own. Rank 0 always retries
            // the handshake before a task, so this is refused, not reached.
            DownLink::HelloOwed => Err(EngineError::Backend(
                "qwen36 pipeline handshake reply still owed; retry the handshake first".into(),
            )),
            DownLink::Desynced(why) => Err(EngineError::Backend(why.clone())),
        }
    }

    /// Mark the downstream link desynced (logged once) so no later exchange reads it.
    fn desync(&mut self, cause: String) {
        if matches!(self.down_link, DownLink::Desynced(_)) {
            return;
        }
        let msg = format!("qwen36 pipeline link desynced after {cause}; restart the pipeline");
        error!(rank = self.rank, "{msg}");
        self.down_link = DownLink::Desynced(msg);
    }

    /// Run one downstream request → reply exchange and settle the link's framing state from how
    /// it ended. The transport exempts a frame's FIRST byte from its recv timeout so an idle
    /// relay can park on the next request; a driver that just sent a request is owed a reply
    /// now, so the whole exchange is bounded by the same configured activation timeout.
    ///
    /// `late` says what a reply that times out before its first byte leaves behind; any other
    /// failure (a timeout mid-request or mid-frame, a wire error, an unexpected frame) desyncs.
    fn run_exchange<T>(
        &mut self,
        h: &tokio::runtime::Handle,
        what: &str,
        late: Late,
        phase: &ExchangePhase,
        fut: impl std::future::Future<Output = Result<T, TransportError>>,
    ) -> Result<T, TransportError> {
        // The timer is built inside the future: `timeout` needs the runtime it runs on.
        let bounded = async { tokio::time::timeout(cascadia_transport::recv_timeout(), fut).await };
        match run_async(h, bounded) {
            Ok(Ok(v)) => {
                // Our own reply came back, so every late one before it has been read past.
                self.down_link = DownLink::Clean;
                Ok(v)
            }
            Ok(Err(e)) => {
                self.desync(format!("a failed {what} reply ({e})"));
                Err(e)
            }
            Err(_) => {
                let at = phase.get();
                match (late, at) {
                    (Late::Skip, ExchangePhase::AWAITING) => {
                        if self.down_link == DownLink::Clean {
                            warn!(
                                what,
                                "qwen36: pipeline reply timed out; its late reply will be skipped"
                            );
                            self.down_link = DownLink::Owed;
                        }
                    }
                    (Late::Await, ExchangePhase::AWAITING) => {
                        warn!("qwen36: handshake reply timed out; the retry waits for it");
                        self.down_link = DownLink::HelloOwed;
                    }
                    _ => {
                        let at = match at {
                            ExchangePhase::SENDING => "mid-request",
                            ExchangePhase::AWAITING => "with no task tag to skip it by",
                            _ => "mid-frame",
                        };
                        self.desync(format!("a {what} reply timeout {at}"));
                    }
                }
                Err(TransportError::SocketClosed)
            }
        }
    }

    /// Rank 0: RESET → RESET_ACK for the current epoch.
    /// Fail-loud, no retry — the API caller retries the task.
    fn reset_exchange(&mut self) -> EngineResult<()> {
        let epoch = self.epoch;
        self.forward_reset_downstream(epoch)
    }

    /// One lockstep FORWARD([1,n,hidden] f32 at pos t0) → TOKEN exchange
    /// with the downstream peer. Returns the chain-end argmax token and
    /// the accumulated downstream infer time (µs) the TOKEN carried.
    fn forward_downstream(
        &mut self,
        epoch: u32,
        hidden: &[f32],
        n: usize,
        t0: usize,
    ) -> EngineResult<(i32, u32)> {
        let drain = self.link_ready()?;
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let h = self.handle()?;
        let tensor = WireTensor::new(
            WireDType::F32,
            [1, n as u32, self.hidden as u32],
            le_bytes_f32(hidden),
        );
        let phase = ExchangePhase::new();
        let ph = &phase;
        self.run_exchange(&h, "TOKEN", Late::Skip, ph, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&frame_header(FRAME_FORWARD, epoch, t0 as u32))
                .await?;
            g.send(&tensor).await?;
            let (kind, e, _) = recv_reply_header(&mut *g, ph, drain, FRAME_TOKEN, epoch).await?;
            if kind != FRAME_TOKEN || e != epoch {
                return Err(TransportError::SocketClosed);
            }
            let tb = g.recv_raw(8).await?;
            Ok((
                i32::from_be_bytes([tb[0], tb[1], tb[2], tb[3]]),
                u32::from_be_bytes([tb[4], tb[5], tb[6], tb[7]]),
            ))
        })
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

    /// Middle/last shared: forward the rank-0 HELLO payload downstream and return
    /// `(reply, downstream_capture_v2_floor)` — reply None = ACK, Some(reason) = NAK. The floor is
    /// the ACK header's `pos`; a pre-R2 peer sends 0 there, which correctly reads as "no v2".
    fn forward_hello_downstream(&mut self, payload: &[u8]) -> EngineResult<(Option<String>, u32)> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        if let DownLink::Desynced(why) = &self.down_link {
            return Err(EngineError::Backend(why.clone()));
        }
        // A HELLO that timed out is still answered (the peer was loading): read that reply.
        let resend = self.down_link != DownLink::HelloOwed;
        let h = self.handle()?;
        let payload = payload.to_vec();
        let phase = ExchangePhase::new();
        let ph = &phase;
        self.run_exchange(&h, "HELLO", Late::Await, ph, async move {
            let mut g = downstream.lock().await;
            if resend {
                g.send_raw(&frame_header(FRAME_HELLO, 0, 0)).await?;
                g.send_raw(&(payload.len() as u32).to_be_bytes()).await?;
                g.send_raw(&payload).await?;
            }
            let (kind, _, cap) = recv_reply_header(&mut *g, ph, false, FRAME_HELLO_ACK, 0).await?;
            match kind {
                FRAME_HELLO_ACK => Ok((None, cap)),
                FRAME_HELLO_NAK => {
                    let lb = g.recv_raw(4).await?;
                    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
                    let rb = g.recv_raw(n).await?;
                    Ok((Some(String::from_utf8_lossy(&rb).into_owned()), 0))
                }
                other => Ok((Some(format!("unexpected handshake reply kind {other}")), 0)),
            }
        })
        .map_err(map_wire)
    }

    /// Middle: forward RESET downstream, await its ack.
    fn forward_reset_downstream(&mut self, epoch: u32) -> EngineResult<()> {
        let drain = self.link_ready()?;
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let h = self.handle()?;
        let phase = ExchangePhase::new();
        let ph = &phase;
        self.run_exchange(&h, "RESET_ACK", Late::Skip, ph, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&frame_header(FRAME_RESET, epoch, 0)).await?;
            let (kind, e, _) =
                recv_reply_header(&mut *g, ph, drain, FRAME_RESET_ACK, epoch).await?;
            if kind != FRAME_RESET_ACK || e != epoch {
                return Err(TransportError::SocketClosed);
            }
            Ok(())
        })
        .map_err(|e| EngineError::Backend(format!("qwen36 pipeline RESET not acked: {e}")))
    }

    /// Rank 0 / middle: FRAME_DECIDE (header + length-prefixed readout body) → FRAME_DECIDE_ACK
    /// for `epoch`. Lockstep like RESET: the ack means every rank below has the readout.
    fn forward_decide_downstream(&mut self, epoch: u32, body: &[u8]) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let drain = self.link_ready()?;
        let h = self.handle()?;
        let body = body.to_vec();
        let phase = ExchangePhase::new();
        let ph = &phase;
        self.run_exchange(&h, "DECIDE_ACK", Late::Skip, ph, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&frame_header(FRAME_DECIDE, epoch, 0)).await?;
            g.send_raw(&body).await?;
            let (kind, e, _) =
                recv_reply_header(&mut *g, ph, drain, FRAME_DECIDE_ACK, epoch).await?;
            if kind != FRAME_DECIDE_ACK || e != epoch {
                return Err(TransportError::SocketClosed);
            }
            Ok(())
        })
        .map_err(|e| EngineError::Backend(format!("qwen36 pipeline DECIDE not acked: {e}")))
    }

    /// Rank 0: announce the active task's readout to the chain for the current epoch.
    fn decide_exchange(&mut self, d: &cascadia_types::DecisionReadout) -> EngineResult<()> {
        let epoch = self.epoch;
        let body = crate::decision::encode_readout(d)
            .map_err(|e| EngineError::Backend(format!("qwen36 pipeline DECIDE body: {e}")))?;
        self.forward_decide_downstream(epoch, &body)
    }

    /// Rank 0 / middle: FRAME_DECIDE_RUN → FRAME_DECISION for `epoch`; returns the result body
    /// still length-prefixed (a middle relays it verbatim, rank 0 decodes it).
    fn forward_decide_run_downstream(&mut self, epoch: u32) -> EngineResult<Vec<u8>> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let drain = self.link_ready()?;
        let h = self.handle()?;
        let phase = ExchangePhase::new();
        let ph = &phase;
        self.run_exchange(&h, "DECISION", Late::Skip, ph, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&frame_header(FRAME_DECIDE_RUN, epoch, 0))
                .await?;
            let (kind, e, _) = recv_reply_header(&mut *g, ph, drain, FRAME_DECISION, epoch).await?;
            if kind != FRAME_DECISION || e != epoch {
                return Err(TransportError::SocketClosed);
            }
            recv_decision_body(&mut *g, "decision body").await
        })
        .map_err(map_wire)
    }

    /// Rank 0: run the head at the chain end and collect its result. Outer `Err` = wire failure;
    /// inner `Err` = the last rank refused the readout or its head failed.
    fn decide_run_exchange(&mut self) -> EngineResult<Result<Vec<f32>, String>> {
        let epoch = self.epoch;
        let body = self.forward_decide_run_downstream(epoch)?;
        crate::decision::decode_result(&body)
            .map_err(|e| EngineError::Backend(format!("qwen36 pipeline DECISION body: {e}")))
    }

    /// Rank 0: end a pipeline decision pass with the chain end's result (one final chunk).
    fn finish_pipe_decision(&mut self) -> Vec<(TaskId, Chunk)> {
        self.want_final_hidden = false;
        self.last_final_hidden = None;
        match self.decide_run_exchange() {
            Ok(Ok(logits)) => {
                let Some(t) = self.active.take() else {
                    return Vec::new();
                };
                let n = t.prompt_ids.len() as u32;
                self.reset_all();
                vec![decision_final_chunk(t.task_id, n, logits)]
            }
            Ok(Err(why)) => self.finalize_error(why),
            Err(e) => {
                warn!(error = %e, "pipeline decision failed");
                self.finalize_error(format!("pipeline decision failed: {e}"))
            }
        }
    }

    /// Middle/last: write one reply upstream: a header (`pos` 0), then `body` verbatim if any.
    fn reply_upstream(&mut self, kind: u32, epoch: u32, body: Vec<u8>) -> EngineResult<()> {
        let h = self.handle()?;
        let upstream = self
            .upstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no upstream peer".into()))?;
        run_async(&h, async move {
            let mut g = upstream.lock().await;
            g.send_raw(&frame_header(kind, epoch, 0)).await?;
            if !body.is_empty() {
                g.send_raw(&body).await?;
            }
            Ok::<(), TransportError>(())
        })
        .map_err(map_wire)
    }

    /// Issue-34 §8: send `CAPTURE(kv_epoch, tokens)` to the downstream peer and await its ACK. Used
    /// by the head (rank 0, after a turn) and chained by each middle rank. Frame-header epoch is the
    /// current task epoch (stale-frame machinery); the KV content epoch rides the body.
    /// A non-empty `tenant` upgrades the frame to `FRAME_CAPTURE_V2` so the downstream rank — which
    /// never sees the `GenerationTask` — can tag its own capture with the same namespace. Requires a
    /// chain that advertised v2 at HELLO; otherwise the v1 frame goes out unchanged.
    #[cfg(feature = "kv_coord")]
    fn forward_capture_downstream(
        &mut self,
        kv_epoch: u64,
        tokens: &[i32],
        tenant: &str,
    ) -> EngineResult<()> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        let h = self.handle()?;
        let task_epoch = self.epoch;
        let v2 = !tenant.is_empty() && self.chain_capture_v2 >= CAPTURE_V2_CAP;
        let (kind, body) = if v2 {
            (
                FRAME_CAPTURE_V2,
                crate::kv_coordination::capture_body_bytes_v2(kv_epoch, tokens, tenant),
            )
        } else {
            (
                FRAME_CAPTURE,
                crate::kv_coordination::capture_body_bytes(kv_epoch, tokens),
            )
        };
        let drain = self.link_ready()?;
        let phase = ExchangePhase::new();
        let ph = &phase;
        // CAPTURE_ACK carries epoch 0, not the task's: a late one cannot be skipped, so a timeout
        // desyncs. Skipping stale frames before it still works, as theirs are tagged.
        self.run_exchange(&h, "CAPTURE_ACK", Late::Desync, ph, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&frame_header(kind, task_epoch, 0)).await?;
            g.send_raw(&(body.len() as u32).to_be_bytes()).await?;
            g.send_raw(&body).await?;
            let (kind, _, _) = recv_reply_header(&mut *g, ph, drain, FRAME_CAPTURE_ACK, 0).await?;
            if kind != FRAME_CAPTURE_ACK {
                return Err(TransportError::SocketClosed);
            }
            Ok(())
        })
        .map_err(|e| EngineError::Backend(format!("qwen36 CAPTURE not acked: {e}")))
    }

    /// Issue-34 consume: send `RESTORE(kv_epoch)` downstream and return the chain's all-or-nothing
    /// verdict (ACK `pos` == 1 ⇒ every downstream rank restored). Used by the head (admission) and
    /// chained by each middle. `Ok(false)` ⇒ the head must fall back to a cold RESET.
    #[cfg(feature = "kv_coord")]
    fn forward_restore_downstream(&mut self, task_epoch: u32, kv_epoch: u64) -> EngineResult<bool> {
        let downstream = self
            .downstream
            .clone()
            .ok_or_else(|| EngineError::Backend("no downstream peer".into()))?;
        // Cross-chain: the head pulled every rank's KV but can't apply a downstream rank's slice
        // locally, so ship it inline — that rank has no CAPTURE stash for a foreign chain's epoch and
        // would otherwise vote cold, collapsing the all-or-nothing verdict. Empty on the same-chain
        // path (the rank restores from its own stash).
        // Take THIS frame's recipient (self.rank + 1), not "whatever was stashed for this epoch":
        // every rank of one pull shares the content epoch, so an epoch-only take returned the
        // last-stashed rank's tensors — wrong state for any chain deeper than 2 stages.
        // A desynced link fails before the stashed blob is taken.
        let drain = self.link_ready()?;
        let down_rank = (self.rank + 1) as u16;
        let blob = self
            .kv
            .take_downstream(kv_epoch, down_rank)
            .or_else(|| self.kv.take_downstream_single(down_rank))
            .unwrap_or_default();
        if !blob.is_empty() {
            info!(
                kv_epoch,
                blob_len = blob.len(),
                "qwen36_restore_carry_downstream"
            );
        }
        let h = self.handle()?;
        let phase = ExchangePhase::new();
        let ph = &phase;
        self.run_exchange(&h, "RESTORE_ACK", Late::Skip, ph, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&frame_header(FRAME_RESTORE, task_epoch, 0))
                .await?;
            g.send_raw(&kv_epoch.to_le_bytes()).await?;
            g.send_raw(&(blob.len() as u32).to_be_bytes()).await?;
            if !blob.is_empty() {
                g.send_raw(&blob).await?;
            }
            let (kind, _, verdict) =
                recv_reply_header(&mut *g, ph, drain, FRAME_RESTORE_ACK, task_epoch).await?;
            if kind != FRAME_RESTORE_ACK {
                return Err(TransportError::SocketClosed);
            }
            Ok(verdict == 1)
        })
        .map_err(|e| EngineError::Backend(format!("qwen36 RESTORE not acked: {e}")))
    }

    /// Rank 0 driver step: same task lifecycle as the single-box step,
    /// with the downstream stage + argmax behind the wire.
    fn step_pipe_first(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.active.is_none() {
            if self.pending.is_empty() {
                return Vec::new();
            }
            self.scrub_failed_task();
            let task = self.pending.remove(0);
            if let Some(refusal) = self.engine_failed_refusal(&task.task_id) {
                return refusal;
            }
            if let Some(reason) = &self.poisoned {
                warn!(task = %task.task_id, reason = %reason, "qwen36: refusing task (handshake mismatch)");
                let reason = format!("qwen36 pipeline poisoned by handshake mismatch: {reason}");
                return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
            }
            if let DownLink::Desynced(reason) = &self.down_link {
                warn!(task = %task.task_id, "qwen36: refusing task (pipeline link desynced)");
                return vec![(
                    task.task_id.clone(),
                    Chunk::error(task.task_id, reason.clone()),
                )];
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
            // Admission (spec §3.2): bump the epoch (frames carry it), tokenize, then either
            // warm-resume (chain-wide RESTORE of the pulled KV) or cold-reset to position 0.
            self.epoch = self.epoch.wrapping_add(1);
            // Rank 0 never owns the head, so its decision state stays clear on every admission.
            self.want_final_hidden = false;
            self.last_final_hidden = None;
            // Decision task: validate the readout (rank 0 holds no head, so this is the
            // well-formedness check) and require every rank to speak the decision frames,
            // before any frame of this task goes out.
            if let Some(d) = &task.decision {
                if let Some(reason) = self.pipe_decision_refusal(d) {
                    let id = task.task_id.clone();
                    return vec![(id.clone(), Chunk::error(id, reason))];
                }
            }
            if decision_resume_conflict(&task) {
                let id = task.task_id.clone();
                let reason = "decision_invalid: resume_token_ids not allowed on a decision task";
                return vec![(id.clone(), Chunk::error(id, reason))];
            }
            let tokenizer = self.tokenizer.as_ref().expect("rank 0 has tokenizer");
            // Same encode + think-block append as before, shared with single-box admission; a
            // decision task prefills exactly the codec's ids instead.
            let mut prompt_ids: Vec<u32> = match admitted_prompt_ids(tokenizer, &task) {
                Ok(ids) => ids,
                Err(e) => {
                    warn!(task = %task.task_id, error = %e, "tokenize failed");
                    let reason = format!("tokenize failed: {e}");
                    return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
                }
            };
            // Option B forced-prefix resume: append the already-emitted assistant
            // tokens after the rendered prompt (concat, not replace) so the cold
            // prefill below carries them as context. No-op when not resuming.
            // `resume_ids()` normalizes `Some([])`; validation bounds peer-supplied
            // indices before they reach native prefill (and makes the u32 casts safe).
            let resume = task.resume_ids().map(<[i32]>::to_vec);
            if let Some(r) = resume.as_deref() {
                let vocab = crate::resume_vocab_bound(tokenizer);
                if let Err(e) = cascadia_types::validate_resume_ids(r, Some(vocab)) {
                    let id = task.task_id.clone();
                    let c = Chunk::error(id.clone(), format!("invalid resume prefix: {e}"));
                    return vec![(id, c)];
                }
            }
            {
                let mut prompt_ids_i64: Vec<i64> = prompt_ids.iter().map(|&u| u as i64).collect();
                cascadia_types::append_resume_ids(&mut prompt_ids_i64, resume.as_deref());
                prompt_ids = prompt_ids_i64.into_iter().map(|t| t as u32).collect();
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
            // Option B: pre-seed `gen_ids` + `emitted` with the resumed tokens so the
            // budget check bounds prefix+new (not just new) and the first NEW tail
            // token's delta decode starts after the prefix text. Empty on a normal turn.
            let resume_gen_ids: Vec<u32> = cascadia_types::resume_generated_seed(resume.as_deref())
                .into_iter()
                .map(|t| t as u32)
                .collect();
            let resume_seed_len = resume_gen_ids.len();
            let resume_emitted: Vec<u8> = if resume_gen_ids.is_empty() {
                Vec::new()
            } else {
                match tokenizer.decode(&resume_gen_ids, true) {
                    Ok(s) => s.into_bytes(),
                    Err(e) => {
                        warn!(task = %task.task_id, error = %e, "resume seed decode failed");
                        let reason = format!("resume seed decode failed: {e}");
                        return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
                    }
                }
            };
            let max_tokens = if task.max_tokens > 0 {
                task.max_tokens
            } else {
                self.max_tokens_default
            } as usize;
            // Cold admission: RESET local + downstream to position 0. Early-returns the task's error
            // chunk on a downstream RESET failure. Used by every non-warm path.
            macro_rules! cold_admit {
                () => {{
                    self.reset_all();
                    if let Err(e) = self.reset_exchange() {
                        warn!(task = %task.task_id, error = %e, "qwen36: admission reset failed");
                        return vec![(
                            task.task_id.clone(),
                            Chunk::error(task.task_id, e.to_string()),
                        )];
                    }
                }};
            }
            // Issue-34 warm-resume: a cached strict-prefix blob ⇒ RESTORE this rank + the whole
            // downstream chain (all-or-nothing); any rank short ⇒ cold. 0 ⇒ full prefill (default).
            // A decision pass never warm-resumes in pipeline mode: the ranks keep no local prefix
            // cache, so it re-prefills (only the single box reuses state across questions).
            let warm_prefix: usize = if task.decision.is_some() {
                cold_admit!();
                0
            } else {
                #[cfg(feature = "kv_coord")]
                {
                    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&u| u as i32).collect();
                    match self.kv.take_warm(&task.tenant, &prompt_i32) {
                        Some((blob, len, plane_pulled)) => {
                            let kv_epoch = crate::kv_coordination::synth_epoch(&prompt_i32[..len]);
                            let local_ok = self.restore_local_stages(&blob);
                            // ALWAYS send the RESTORE, including in plane mode. It is the only thing on
                            // the warm path that advances the downstream ranks' `peer_epoch` (their
                            // InFrame::Restore sets it) and `self.epoch` was already bumped at
                            // admission — short-circuiting it left every subsequent FORWARD dropped as
                            // "stale frame". It is also the real restore on the same-chain path, where
                            // no plane pull ever armed the downstream ranks.
                            let plane_turn = self.plane_restore && plane_pulled;
                            let chain_ok = local_ok && {
                                // Binding in BOTH modes, matching ov-runtime: a plane rank arms
                                // in-band inside its own OPCODE_RESTORE handler, so a `false` here
                                // means it really is cold. The old `chain_verdict` override existed
                                // for the out-of-band arm and would now mask exactly that — turning
                                // a retracted or dropped downstream slice into a warm head over a
                                // cold rank, which is wrong output rather than a cold reprefill.
                                self.forward_restore_downstream(self.epoch, kv_epoch)
                                    .unwrap_or(false)
                            };
                            if chain_ok {
                                // Real KV depth, not the token count (off-by-one — see kv_seq_from_blob).
                                // kv_seq_from_framed_blob now ignores the fixed-shape DeltaNet conv/ssm
                                // states (they poisoned the depth max with a constant 128), so it returns
                                // the TRUE attention fold depth. Resume there (`.min(len)`, matching the
                                // GREEN ov-runtime path). Resuming at len-1 re-fed the last folded token,
                                // which the position-free DeltaNet layers double-applied ⇒ cross-chain
                                // DIVERGE; the attention-only depth is correct for both attention and SSM.
                                let warm = crate::kv_coordination::kv_seq_from_framed_blob(&blob)
                                    .map(|s| s.min(len))
                                    .unwrap_or(len);
                                info!(task = %task.task_id, warm_prefix = warm, matched = len, plane_pulled, plane_turn, "qwen36 pipeline warm-resumed");
                                warm
                            } else {
                                warn!(task = %task.task_id, "qwen36: pipeline restore incomplete; cold reset");
                                cold_admit!();
                                0
                            }
                        }
                        None => {
                            cold_admit!();
                            0
                        }
                    }
                }
                #[cfg(not(feature = "kv_coord"))]
                {
                    cold_admit!();
                    0
                }
            };
            // Announce the readout after the RESET, before the prefill FORWARDs.
            if let Some(d) = &task.decision {
                if let Err(e) = self.decide_exchange(d) {
                    warn!(task = %task.task_id, error = %e, "qwen36: DECIDE exchange failed");
                    self.reset_all();
                    return vec![(
                        task.task_id.clone(),
                        Chunk::error(task.task_id, e.to_string()),
                    )];
                }
            }
            self.active = Some(ActiveTask {
                task_id: task.task_id,
                tenant: task.tenant,
                prompt_ids,
                prefill_idx: warm_prefix,
                step: warm_prefix,
                warm_prefix,
                snapshot_at: Vec::new(),
                logits: Vec::new(),
                next_token: None,
                gen_ids: resume_gen_ids,
                emitted: resume_emitted,
                resume_seed_len,
                max_tokens,
                started: Instant::now(),
                snapshot_secs: 0.0,
                wire_ms: Vec::new(),
                decision: task.decision.clone(),
                collector: None,
                retry_source: None,
                retried: false,
                retry_resumed: 0,
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
                let end = (t.prefill_idx + prefill_chunk()).min(t.prompt_ids.len());
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
                        let reached = t.step;
                        // Cold self-checkpoint control (see selfchk_at). `state_restored` is only set
                        // by a real warm restore, so this never fires on the warm path.
                        #[cfg(feature = "kv_coord")]
                        if selfchk_at() == Some(reached) && !self.state_restored {
                            match self.blob_local_stages() {
                                Some(blob) => {
                                    let ok = self.restore_local_stages(&blob);
                                    info!(task = %task_id, pos = reached, ok,
                                        "qwen36_selfchk (cold capture->recreate->restore)");
                                }
                                None => warn!(task = %task_id, pos = reached,
                                    "qwen36_selfchk: capture failed"),
                            }
                        }
                        #[cfg(feature = "kv_coord")]
                        if fp_at() == Some(reached) {
                            let fps: Vec<u64> = self
                                .stages
                                .iter_mut()
                                .map(|st| {
                                    st.get_state_blob()
                                        .map(|b| crate::kv_coordination::fnv1a64(&b))
                                        .unwrap_or(0)
                                })
                                .collect();
                            info!(task = %task_id, pos = reached, ?fps, "qwen36_fp_at (cold fold)");
                        }
                    }
                    Err(e) => {
                        warn!(task = %task_id, error = %e, "pipeline prefill failed");
                        return self.finalize_error(format!("pipeline prefill failed: {e}"));
                    }
                }
            }
            // Issue-34 bar-#1 diag: fingerprint the post-prefill state. A warm-resumed turn and a
            // cold one run the SAME turn-2 prompt, so they reach this point at the same `pos` — if
            // `fps` differs between them, restore+suffix-prefill does not reproduce a cold fold and
            // the divergence is upstream of decode. If `fps` matches yet the text still differs, the
            // fold is faithful and the flip is in decode. The two cases need different fixes.
            #[cfg(feature = "kv_coord")]
            if postprefill_fp() {
                let (pos, first) = {
                    let t = self.active.as_ref().unwrap();
                    (t.step, t.next_token)
                };
                let fps: Vec<u64> = self
                    .stages
                    .iter_mut()
                    .map(|st| {
                        st.get_state_blob()
                            .map(|b| crate::kv_coordination::fnv1a64(&b))
                            .unwrap_or(0)
                    })
                    .collect();
                info!(task = %task_id, pos, first_token = ?first, ?fps, "qwen36_postprefill_state");
            }
            // Decision pass: collect the chain end's head result instead of decoding.
            if self.active.as_ref().is_some_and(|t| t.decision.is_some()) {
                return self.finish_pipe_decision();
            }
            Vec::new()
        } else {
            if t.decision.is_some() {
                return self.finish_pipe_decision();
            }
            if t.gen_ids.len() >= t.max_tokens {
                return self.finalize(FinishReason::Length);
            }
            let next = t.next_token.expect("pipeline decode without pending token");
            if Some(next) == self.eos {
                return self.finalize(FinishReason::Stop);
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
                    let step_now = t.step;
                    let gen_n = t.gen_ids.len();
                    // Multi-stage path only: this is the topology bar #1 fails on. See decode_fp.
                    #[cfg(feature = "kv_coord")]
                    if decode_fp() {
                        let mut fps: Vec<u64> = Vec::with_capacity(self.stages.len());
                        for st in self.stages.iter_mut() {
                            match st.get_state_blob() {
                                Ok(b) => {
                                    fps.push(crate::kv_coordination::fnv1a64(&b));
                                    // Per-stage fps says THAT the state drifted at the first decode
                                    // fold; it cannot say WHICH tensors. Attention-only would point at
                                    // positional/layout handling, conv/ssm-only at the DeltaNet
                                    // recurrent state. Capped at the first few steps: 40 tensors x 48
                                    // steps is noise, and the drift is already known to start at fold 1.
                                    // Self-gated on CASCADIA_KV_TENSOR_DUMP.
                                    if gen_n <= 3 {
                                        crate::kv_coordination::log_blob_tensors(
                                            "qwen36_decode",
                                            step_now as u64,
                                            &b,
                                        );
                                    }
                                }
                                Err(_) => fps.push(0),
                            }
                        }
                        info!(task = %task_id, pos = step_now, tok, ?fps, "qwen36_decode_fp");
                    }
                    let full = self
                        .tokenizer
                        .as_ref()
                        .expect("rank 0 has tokenizer")
                        .decode(self.active.as_ref().unwrap().gen_ids.as_slice(), true)
                        .unwrap_or_default();
                    let t = self.active.as_mut().unwrap();
                    // Shim `advance_emitted`: FFFD-run holdback + starts_with
                    // re-anchor, one implementation shared with ov-runtime so
                    // the two cannot drift.
                    let (delta, reanchored) =
                        advance_emitted(&mut t.emitted, full.as_bytes(), true);
                    if reanchored {
                        warn!(task = %task_id, "decode diverged from the emitted prefix; re-anchored (delta dropped)");
                    }
                    vec![(
                        task_id.clone(),
                        Chunk::token(task_id, next as i64, delta).with_token_ids(vec![next as i64]),
                    )]
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
                FRAME_DECIDE => Ok(InFrame::Decide {
                    epoch,
                    body: recv_decision_body(&mut *g, "decide body").await?,
                }),
                FRAME_DECIDE_RUN => Ok(InFrame::DecideRun(epoch)),
                #[cfg(feature = "kv_coord")]
                FRAME_CAPTURE => {
                    let lb = g.recv_raw(4).await?;
                    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
                    let body =
                        recv_body_chunked(&mut *g, n, MAX_CARRY_BLOB_BYTES, "capture body").await?;
                    let (kv_epoch, tokens) = crate::kv_coordination::parse_capture_body(&body)
                        .ok_or(TransportError::SocketClosed)?;
                    Ok(InFrame::Capture {
                        kv_epoch,
                        tokens,
                        partner: None,
                    })
                }
                #[cfg(feature = "kv_coord")]
                FRAME_CAPTURE_V2 => {
                    let lb = g.recv_raw(4).await?;
                    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
                    let body =
                        recv_body_chunked(&mut *g, n, MAX_CARRY_BLOB_BYTES, "capture v2 body")
                            .await?;
                    let (kv_epoch, tokens, partner) =
                        crate::kv_coordination::parse_capture_body_v2(&body)
                            .ok_or(TransportError::SocketClosed)?;
                    Ok(InFrame::Capture {
                        kv_epoch,
                        tokens,
                        partner: Some(partner),
                    })
                }
                #[cfg(feature = "kv_coord")]
                FRAME_RESTORE => {
                    let eb = g.recv_raw(8).await?;
                    let kv_epoch = u64::from_le_bytes(
                        eb.as_slice()
                            .try_into()
                            .map_err(|_| TransportError::SocketClosed)?,
                    );
                    // Always length-prefixed (0 ⇒ no carry) so the frame stays self-describing.
                    let lb = g.recv_raw(4).await?;
                    let blob_len = u32::from_be_bytes(
                        lb.as_slice()
                            .try_into()
                            .map_err(|_| TransportError::SocketClosed)?,
                    ) as usize;
                    // Chunked + drain-on-reject: an over-ceiling blob rejected WITHOUT consuming
                    // it stays in the stream and desyncs every later frame.
                    let blob = if blob_len == 0 {
                        Vec::new()
                    } else {
                        recv_body_chunked(
                            &mut *g,
                            blob_len,
                            MAX_CARRY_BLOB_BYTES,
                            "restore carried blob",
                        )
                        .await?
                    };
                    Ok(InFrame::Restore {
                        task_epoch: epoch,
                        kv_epoch,
                        blob,
                    })
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
                // Chain capability bits: this rank's own, ANDed with everything below it. The last
                // rank has nothing below, so it reports its own. An old downstream rank replies its
                // capture floor (0/1), which clears CAP_DECISION and keeps bit 0's meaning.
                let mut cap = own_caps(self.serves_decisions());
                if reason.is_none() && !is_last {
                    match self.forward_hello_downstream(&payload) {
                        Ok((r, down_cap)) => {
                            reason = r.map(|r| format!("downstream: {r}"));
                            cap = combine_caps(cap, down_cap);
                        }
                        Err(e) => {
                            reason = Some(format!("downstream handshake forward failed: {e}"))
                        }
                    }
                }
                self.chain_caps = cap;
                self.chain_capture_v2 = (cap & CAP_CAPTURE_V2).min(CAPTURE_V2_CAP);
                let h = self.handle()?;
                let upstream = self.upstream.clone().unwrap();
                match reason {
                    None => {
                        run_async(&h, async move {
                            let mut g = upstream.lock().await;
                            g.send_raw(&frame_header(FRAME_HELLO_ACK, 0, cap)).await
                        })
                        .map_err(map_wire)?;
                        info!(
                            caps = cap,
                            capture_v2 = self.chain_capture_v2,
                            "qwen36 pipeline handshake ok"
                        );
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
                // Task 46: a stage abandoned by this reset's recovery. No RESET_ACK: the stage
                // exits (connection-fatal) and its closed socket fails the head's exchange now.
                if let Some(e) = self.relay_engine_failed() {
                    return Err(e);
                }
                self.peer_epoch = epoch;
                // A new task: no decision state may survive from the previous one.
                self.clear_decision_state();
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
            InFrame::Capture {
                kv_epoch,
                tokens,
                partner,
            } => {
                // Snapshot this rank's local KV under the head's content epoch (state is still live —
                // RESET comes at the next admission), chain CAPTURE downstream, then ack upstream.
                // Best-effort: a blob/chain miss degrades to no warm-pull, never breaks generation.
                let ns = partner.unwrap_or_else(|| crate::kv_coordination::LOCAL_NS.to_string());
                if let Some(blob) = self.blob_local_stages() {
                    // Mirror into the lock-free holder cache (worker rank serves rank-N GET from here).
                    self.kv_share
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .capture_under_epoch_ns(&ns, kv_epoch, tokens.clone(), blob.clone());
                    self.kv
                        .capture_under_epoch_ns(&ns, kv_epoch, tokens.clone(), blob);
                }
                if !is_last {
                    if let Err(e) = self.forward_capture_downstream(kv_epoch, &tokens, &ns) {
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
            #[cfg(feature = "kv_coord")]
            InFrame::Restore {
                task_epoch,
                kv_epoch,
                blob,
            } => {
                // Consume warm-resume: set_state this rank's pulled slice, chain RESTORE downstream,
                // ack with the all-or-nothing verdict (local && downstream restored). A miss anywhere
                // ⇒ verdict 0 ⇒ the head re-RESETs the chain cold (never a partial/corrupt restore).
                self.peer_epoch = task_epoch;
                // RESTORE starts a new (warm) task exactly like RESET: drop any decision state an
                // aborted decision left, or every chat FORWARD after it pays the final-hidden copy.
                self.clear_decision_state();
                // Carried blob wins: on a CROSS-chain move this rank has no capture under the source
                // chain's epoch, so the local stash is empty and only the head's inline copy exists.
                // Drain FIRST — same reason as ov-runtime's RESTORE arm: in plane mode the head
                // parks this rank's slice AND still carries a blob inline, so a `||` after the
                // carried branch short-circuits the drain away and the rank warms from carried data
                // while the plane slice goes unread. Chain mode parks nothing, so this is a false
                // no-op there and the carried/capture path below is unchanged.
                let local_ok = if self.drain_kv_handoff(kv_epoch) {
                    true
                } else if !blob.is_empty() {
                    let ok = self.restore_local_stages(&blob);
                    // Cert marker: proves the CARRIED (cross-chain) branch ran, not the same-chain
                    // capture fallback — the two are indistinguishable in the verdict alone. Emitted
                    // ONLY on success, matching `ov_tail_restore_carried`, so the cert's gate counts
                    // successes for both engines rather than attempts for one of them.
                    if ok {
                        info!(
                            kv_epoch,
                            blob_len = blob.len(),
                            "qwen36_tail_restore_carried"
                        );
                    } else {
                        warn!(
                            kv_epoch,
                            blob_len = blob.len(),
                            "qwen36_tail_restore_carried_failed"
                        );
                    }
                    ok
                } else {
                    match self.kv.take_capture(kv_epoch) {
                        Some((_, blob)) => self.restore_local_stages(&blob),
                        None => false,
                    }
                };
                let down_ok = if is_last {
                    true
                } else {
                    self.forward_restore_downstream(task_epoch, kv_epoch)
                        .unwrap_or(false)
                };
                let verdict = u32::from(local_ok && down_ok);
                let h = self.handle()?;
                let upstream = self.upstream.clone().unwrap();
                run_async(&h, async move {
                    let mut g = upstream.lock().await;
                    g.send_raw(&frame_header(FRAME_RESTORE_ACK, task_epoch, verdict))
                        .await
                })
                .map_err(map_wire)
            }
            InFrame::Decide { epoch, body } => {
                if self.poisoned.is_some() {
                    return Err(EngineError::Backend(
                        "qwen36 pipeline poisoned by handshake mismatch".into(),
                    ));
                }
                if epoch != self.peer_epoch {
                    // Stale epoch: drop like FORWARD; the driver's recv times out loud.
                    warn!(
                        epoch,
                        current = self.peer_epoch,
                        "qwen36: dropping stale DECIDE frame"
                    );
                    return Ok(());
                }
                if is_last {
                    // Refused readouts are still acked (lockstep); the error surfaces at RUN.
                    let r = crate::decision::decode_readout(&body)
                        .and_then(|d| self.admit_decision_check(&d).map(|_| d));
                    self.want_final_hidden = r.as_ref().is_ok_and(head_reads_hidden);
                    self.last_final_hidden = None;
                    self.last_logits = Vec::new();
                    let hidden = self.hidden;
                    self.pending_decision = Some((
                        epoch,
                        r.map(|d| {
                            let col = crate::decision::Collector::new(d.head.hidden_need(), hidden);
                            (d, col)
                        }),
                    ));
                } else {
                    self.forward_decide_downstream(epoch, &body)?;
                }
                self.reply_upstream(FRAME_DECIDE_ACK, epoch, Vec::new())
            }
            InFrame::DecideRun(epoch) => {
                if !is_last {
                    // Relay: the chain end's DECISION comes back through us verbatim.
                    let body = self.forward_decide_run_downstream(epoch)?;
                    return self.reply_upstream(FRAME_DECISION, epoch, body);
                }
                let out: Result<Vec<f32>, String> = take_pending(&mut self.pending_decision, epoch)
                    .and_then(|(d, col)| match self.decision_assets.as_mut() {
                        Some(assets) => crate::decision::run_head(
                            &d.head,
                            assets,
                            &d.prompt_ids,
                            &self.last_logits,
                            &col,
                        ),
                        None => Err("decision_unsupported: no head assets".into()),
                    });
                if let Err(why) = &out {
                    warn!(epoch, error = %why, "qwen36: pipeline decision head refused/failed");
                }
                self.clear_decision_state();
                self.reset_all();
                self.reply_upstream(FRAME_DECISION, epoch, crate::decision::encode_result(&out))
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
                if hidden.len() != n * self.hidden {
                    return Err(EngineError::Backend(format!(
                        "forward frame size {} != n({n}) * hidden({})",
                        hidden.len(),
                        self.hidden
                    )));
                }
                let infer_started = Instant::now();
                let t0 = pos as usize;
                let out = self.chain_pass(&hidden, t0, t0 + n)?;
                // rank>0 counterpart of the head's `qwen36_fp_at (cold fold)`. The tail folds via this
                // frame, not the prefill loop, so without this its COLD state at `pos` is unmeasurable
                // and only its warm-RESTORED state (logged by restore_local_stages) is visible. Single-box
                // proved local restore faithful, so any remaining bar-#1 divergence must show up as a
                // tail warm-vs-cold mismatch here.
                #[cfg(feature = "kv_coord")]
                if fp_at() == Some(t0 + n) {
                    let fps: Vec<u64> = self
                        .stages
                        .iter_mut()
                        .map(|st| {
                            st.get_state_blob()
                                .map(|b| crate::kv_coordination::fnv1a64(&b))
                                .unwrap_or(0)
                        })
                        .collect();
                    info!(pos = t0 + n, ?fps, "qwen36_fp_at (tail fold)");
                }
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
                    // Decision pass: gather this span's hidden rows and keep the logits row
                    // the head reads. Gated on a pending decision, so chat pays nothing.
                    if self.pending_decision.is_some() {
                        let fh = self.last_final_hidden.take();
                        if let Some((e, Ok((_, col)))) = self.pending_decision.as_mut() {
                            if let Some(fh) = fh {
                                if let Err(why) = col.absorb(t0, n, &fh) {
                                    let e = *e;
                                    self.pending_decision =
                                        Some((e, Err(format!("decision_invalid: {why}"))));
                                }
                            }
                        }
                        self.last_logits = logits;
                    }
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
        // H.1b R2. ABSENT ⇒ a pre-R2 (or non-`kv_coord`) build: accepted, and its `HELLO_ACK` carries
        // no capability, so the head reads the chain floor as 0 and stays on v1. PRESENT and
        // different ⇒ two v2-aware builds that disagree about the CAPTURE body shape; refuse now
        // rather than desync the frame stream on an unknown kind mid-turn.
        #[cfg(feature = "kv_coord")]
        if !theirs["capture_v2"].is_null()
            && theirs["capture_v2"] != serde_json::json!(CAPTURE_V2_CAP)
        {
            return Some(format!(
                "CAPTURE v2 capability mismatch: theirs {} ours {CAPTURE_V2_CAP}",
                theirs["capture_v2"]
            ));
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
            self.scrub_failed_task();
            let task = self.pending.remove(0);
            // Task 57: the retry `fail_local` re-queued in this same step, if this is it. Taken
            // by every admission, so a stale one can never reach a later task.
            let retry = self.retry.take().filter(|r| r.task_id == task.task_id);
            // Task 55: only a decision pass may restore over the previous pass's live requests
            // (`local_warm_or_cold`); chat, completion and the kv_coord restore start scrubbed.
            if task.decision.is_none() {
                self.scrub_deferred_decision();
            }
            let admit_started = Instant::now();
            self.pass_started = Some(admit_started);
            self.timing = PassTiming::new(self.stages.len(), 0);
            if let Some(refusal) = self.engine_failed_refusal(&task.task_id) {
                return refusal;
            }
            // reset moved below — Issue-34 warm-resume may restore a cached prefix blob instead.
            let tokenizer = self.tokenizer.as_ref().expect("single-box has tokenizer");
            let mut prompt_ids: Vec<u32> = match admitted_prompt_ids(tokenizer, &task) {
                Ok(ids) => ids,
                Err(e) => {
                    warn!(task = %task.task_id, error = %e, "tokenize failed");
                    let reason = format!("tokenize failed: {e}");
                    return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
                }
            };
            // Every admission (chat too) clears it here and sets it only once the task is admitted
            // (just before `active` is filled), so neither a stale flag nor a refused admission
            // can make a later pass copy `final_hidden`.
            self.want_final_hidden = false;
            if let Some(d) = &task.decision {
                if let Err(reason) = self.admit_decision_check(d) {
                    let id = task.task_id.clone();
                    return vec![(id.clone(), Chunk::error(id, reason))];
                }
            }
            if decision_resume_conflict(&task) {
                let id = task.task_id.clone();
                let reason = "decision_invalid: resume_token_ids not allowed on a decision task";
                return vec![(id.clone(), Chunk::error(id, reason))];
            }
            // Option B forced-prefix resume: append the already-emitted assistant
            // tokens after the rendered prompt (concat, not replace) so the cold
            // prefill below carries them as context. No-op when not resuming.
            // `resume_ids()` normalizes `Some([])`; validation bounds peer-supplied
            // indices before they reach native prefill (and makes the u32 casts safe).
            let resume = task.resume_ids().map(<[i32]>::to_vec);
            if let Some(r) = resume.as_deref() {
                let vocab = crate::resume_vocab_bound(tokenizer);
                if let Err(e) = cascadia_types::validate_resume_ids(r, Some(vocab)) {
                    let id = task.task_id.clone();
                    let c = Chunk::error(id.clone(), format!("invalid resume prefix: {e}"));
                    return vec![(id, c)];
                }
            }
            {
                let mut prompt_ids_i64: Vec<i64> = prompt_ids.iter().map(|&u| u as i64).collect();
                cascadia_types::append_resume_ids(&mut prompt_ids_i64, resume.as_deref());
                prompt_ids = prompt_ids_i64.into_iter().map(|t| t as u32).collect();
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
            // Option B: pre-seed `gen_ids` + `emitted` with the resumed tokens so the
            // budget check bounds prefix+new (not just new) and the first NEW tail
            // token's delta decode starts after the prefix text. Empty on a normal turn.
            let resume_gen_ids: Vec<u32> = cascadia_types::resume_generated_seed(resume.as_deref())
                .into_iter()
                .map(|t| t as u32)
                .collect();
            let resume_seed_len = resume_gen_ids.len();
            // Task 57: a retried chat carries the bytes its failed attempt already handed to the
            // client. Re-decoding the seed could differ from them: the U+FFFD holdback keeps an
            // incomplete UTF-8 char out of `emitted`, and that char would then be lost.
            let retried = retry.is_some();
            let retry_resumed = retry.as_ref().map_or(0, |r| r.generated);
            let carried = retry.and_then(|r| r.emitted);
            let resume_emitted: Vec<u8> = if let Some(bytes) = carried {
                bytes
            } else if resume_gen_ids.is_empty() {
                Vec::new()
            } else {
                match tokenizer.decode(&resume_gen_ids, true) {
                    Ok(s) => s.into_bytes(),
                    Err(e) => {
                        warn!(task = %task.task_id, error = %e, "resume seed decode failed");
                        let reason = format!("resume seed decode failed: {e}");
                        return vec![(task.task_id.clone(), Chunk::error(task.task_id, reason))];
                    }
                }
            };
            let max_tokens = if task.max_tokens > 0 {
                task.max_tokens
            } else {
                self.max_tokens_default
            } as usize;
            // The window bounds prompt + completion, not just the prompt: `submit` refuses prompts
            // at or past `max_ctx`, and this caps the budget so the existing max_tokens stop ends
            // the turn AT the window instead of decoding past `max_position_embeddings`. `gen_ids`
            // is pre-seeded with the resume tokens, which `prompt_ids` already carries.
            let max_tokens = match self.max_ctx {
                Some(max) => max_tokens.min(resume_seed_len + max.saturating_sub(prompt_ids.len())),
                None => max_tokens,
            };
            // Issue-34 warm-resume: restore a cached strict-prefix blob and prefill only the suffix;
            // else cold reset. Gated + best-effort (stub ⇒ no blob ⇒ cold). 0 on the default path.
            let warm_prefix: usize = if let Some(d) = &task.decision {
                self.local_warm_or_cold(decision_warm_probe(d, &prompt_ids))
            } else {
                #[cfg(feature = "kv_coord")]
                {
                    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&u| u as i32).collect();
                    match self.kv.take_warm(&task.tenant, &prompt_i32) {
                        Some((blob, len, plane_pulled)) if self.restore_local_stages(&blob) => {
                            // Real KV depth, not the token count (off-by-one — see kv_seq_from_blob).
                            // See the sibling site: kv_seq_from_framed_blob now skips conv/ssm and returns
                            // the true attention depth, so resume at `.min(len)` (matching ov-runtime).
                            let warm = crate::kv_coordination::kv_seq_from_framed_blob(&blob)
                                .map(|s| s.min(len))
                                .unwrap_or(len);
                            info!(
                                warm_prefix = warm,
                                matched = len,
                                plane_pulled,
                                "qwen36 single-box warm-resumed from KV blob"
                            );
                            warm
                        }
                        Some(_) => {
                            // A plane blob that failed to restore may have dirtied some stages:
                            // scrub before the local cache gets to try.
                            self.reset_all();
                            self.local_warm_or_cold(&prompt_ids)
                        }
                        None => self.local_warm_or_cold(&prompt_ids),
                    }
                }
                #[cfg(not(feature = "kv_coord"))]
                {
                    self.local_warm_or_cold(&prompt_ids)
                }
            };
            // Every arm above resets (or restores over) the stage requests. If the reset FAILED the
            // previous turn's DeltaNet state is still live, and a prefill on top of it decodes
            // confident garbage that the API would return as a 200. Refuse the turn instead — the
            // snapshot path already refuses to cache from this state; serving from it is worse.
            // A request abandoned during this admission (a failed priming fold): the engine's own
            // refusal, not a reset retry.
            if let Some(refusal) = self.engine_failed_refusal(&task.task_id) {
                return refusal;
            }
            if self.stages_dirty {
                error!(task = %task.task_id, "qwen35: stage reset failed; refusing to serve");
                // The warm arm restores over the LIVE requests without a reset of its own, so
                // nothing else would retry the rebuild and every cache-hitting turn would refuse:
                // try it here (dirty ⇒ `recreate_request`) so the next turn can serve.
                self.reset_all();
                let id = task.task_id.clone();
                let c = Chunk::error(
                    id.clone(),
                    "stage state reset failed; refusing to serve on unknown state",
                );
                return vec![(id, c)];
            }
            if let Some(reason) = task
                .decision
                .as_ref()
                .and_then(|d| decision_cache_refusal(d, warm_prefix, &self.prefix_cache))
            {
                warn!(task = %task.task_id, %reason, "qwen35: decision pass refused");
                let id = task.task_id.clone();
                return vec![(id.clone(), Chunk::error(id, reason))];
            }
            // Snapshot points for THIS turn (end of the system block, and the position before the
            // prompt's last `<|im_start|>`, which the next turn re-sends verbatim): only positions
            // this turn actually prefills (past `warm_prefix`), never from stages whose reset
            // failed, and not for keys the cache already holds (a shared system prompt). A cold
            // turn snapshots every boundary. A warm turn may too — a primed restore reads back the
            // full state (OV-level probe: depth and logits identical to the cold fold) — but each
            // copy is paid inside this turn's TTFT, so it is a *refresh* taken only once the tail
            // it would save has grown to cost as much as the copy (measured rates, see
            // `RefreshCosts::refresh_pays`); a conversation therefore never goes cold, and short
            // follow-ups pay restore + tail only. A boundary at or past a length the budget has
            // already refused is dropped by `plan_snapshots` before the copy, not by `insert`
            // after it (`PrefixCache::worth_copying`).
            let state_bytes = self.last_restore_bytes;
            let boundaries = match &task.decision {
                Some(d) => decision_snapshot_boundaries(
                    d,
                    self.prefix_cache.enabled() && !self.stages_dirty,
                ),
                None => self
                    .im_start_id
                    .filter(|_| self.prefix_cache.enabled() && !self.stages_dirty)
                    .map(|im| crate::prefix_cache::chat_boundaries(&prompt_ids, im))
                    .unwrap_or_default(),
            };
            // A needs_snapshot decision pass always snapshots its request's shared prefix when the
            // cache can hold it (its later passes resume there, P2-CE-1); every other pass keeps
            // the chat refresh rule (P3-REG-1).
            let snapshot_at = plan_pass_snapshots(
                task.decision.as_ref(),
                &prompt_ids,
                &boundaries,
                warm_prefix,
                &self.prefix_cache,
                &self.refresh_costs,
                state_bytes,
            );
            if self.prefix_cache.enabled() {
                info!(
                    warm_prefix,
                    ?snapshot_at,
                    prompt = prompt_ids.len(),
                    prefill_tok_s = ?self.refresh_costs.prefill_tok_s,
                    "qwen35 prefix-cache snapshot plan"
                );
            }
            self.want_final_hidden = task.decision.as_ref().is_some_and(head_reads_hidden);
            // `reset` is still zero unless this admission ran its own `reset_all` (a cold pass).
            self.timing.admit_reset = self.timing.reset;
            self.timing.prompt = prompt_ids.len();
            self.timing.warm_prefix = warm_prefix;
            self.timing.admit = admit_started.elapsed();
            self.active = Some(ActiveTask {
                task_id: task.task_id.clone(),
                tenant: task.tenant.clone(),
                prompt_ids,
                prefill_idx: warm_prefix,
                step: warm_prefix,
                warm_prefix,
                snapshot_at,
                logits: Vec::new(),
                next_token: None,
                gen_ids: resume_gen_ids,
                emitted: resume_emitted,
                resume_seed_len,
                max_tokens,
                started: Instant::now(),
                snapshot_secs: 0.0,
                wire_ms: Vec::new(),
                collector: task
                    .decision
                    .as_ref()
                    .map(|d| crate::decision::Collector::new(d.head.hidden_need(), self.hidden)),
                decision: task.decision.clone(),
                retry_source: Some(task),
                retried,
                retry_resumed,
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
                let end = crate::prefix_cache::next_prefill_end(
                    t.prefill_idx,
                    t.prompt_ids.len(),
                    prefill_chunk(),
                    &t.snapshot_at,
                );
                let toks: Vec<u32> = t.prompt_ids[t.prefill_idx..end].to_vec();
                let t0 = t.step;
                match self.run_span(&toks, t0) {
                    Ok(l) => {
                        let t = self.active.as_mut().unwrap();
                        t.logits = l;
                        t.prefill_idx = end;
                        t.step += toks.len();
                        let snap_key =
                            (t.snapshot_at.contains(&end)).then(|| t.prompt_ids[..end].to_vec());
                        // Same diagnostics as the multi-stage loop, mirrored onto the SINGLE-STAGE
                        // path. On a 1-stage (single-box) topology every stage is local, so these
                        // fingerprints cover the WHOLE model state — the 2-stage head-only view left
                        // the tail unmeasured, and head/tail disagreed there.
                        let reached = t.step;
                        if let Some(key) = snap_key {
                            self.prefix_cache_snapshot(&task_id, key);
                        }
                        #[cfg(feature = "kv_coord")]
                        if selfchk_at() == Some(reached) && !self.state_restored {
                            match self.blob_local_stages() {
                                Some(blob) => {
                                    let ok = self.restore_local_stages(&blob);
                                    info!(task = %task_id, pos = reached, ok,
                                        "qwen36_selfchk (cold capture->recreate->restore)");
                                }
                                None => warn!(task = %task_id, pos = reached,
                                    "qwen36_selfchk: capture failed"),
                            }
                        }
                        #[cfg(feature = "kv_coord")]
                        if fp_at() == Some(reached) {
                            let fps: Vec<u64> = self
                                .stages
                                .iter_mut()
                                .map(|st| {
                                    st.get_state_blob()
                                        .map(|b| crate::kv_coordination::fnv1a64(&b))
                                        .unwrap_or(0)
                                })
                                .collect();
                            info!(task = %task_id, pos = reached, ?fps, "qwen36_fp_at (cold fold)");
                        }
                    }
                    Err(e) => {
                        warn!(task = %task_id, error = %e, "prefill failed");
                        return self.fail_local(format!("prefill failed: {e}"));
                    }
                }
            }
            // Prefill rate for the refresh rule: this turn's prefilled tokens over the wall time
            // since admission minus the snapshot copies inside it. Short tails are dominated by
            // per-span overhead and would under-read the rate, so only a tail of a full chunk or
            // more updates it.
            {
                let t = self.active.as_ref().unwrap();
                let prefilled = t.prompt_ids.len() - t.warm_prefix;
                let secs = t.started.elapsed().as_secs_f64() - t.snapshot_secs;
                if prefilled >= prefill_chunk() && secs > 0.0 {
                    self.refresh_costs.prefill_tok_s = Some(prefilled as f64 / secs);
                }
            }
            #[cfg(feature = "kv_coord")]
            {
                let (pos, first) = {
                    let t = self.active.as_ref().unwrap();
                    let n = t
                        .logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32);
                    (t.step, n)
                };
                let fps: Vec<u64> = self
                    .stages
                    .iter_mut()
                    .map(|st| {
                        st.get_state_blob()
                            .map(|b| crate::kv_coordination::fnv1a64(&b))
                            .unwrap_or(0)
                    })
                    .collect();
                info!(task = %task_id, pos, first_token = ?first, ?fps, "qwen36_postprefill_state");
            }
            Vec::new()
        } else {
            if t.decision.is_some() {
                return self.finish_decision();
            }
            if t.gen_ids.len() >= t.max_tokens {
                return self.finalize(FinishReason::Length);
            }
            let next = t
                .logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            if Some(next) == self.eos {
                return self.finalize(FinishReason::Stop);
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
                    // Shim `advance_emitted`: FFFD-run holdback + starts_with
                    // re-anchor, one implementation shared with ov-runtime so
                    // the two cannot drift.
                    let (delta, reanchored) =
                        advance_emitted(&mut t.emitted, full.as_bytes(), true);
                    if reanchored {
                        warn!(task = %task_id, "decode diverged from the emitted prefix; re-anchored (delta dropped)");
                    }
                    vec![(
                        task_id.clone(),
                        Chunk::token(task_id, next as i64, delta).with_token_ids(vec![next as i64]),
                    )]
                }
                Err(e) => {
                    warn!(task = %task_id, error = %e, "decode failed");
                    self.fail_local(format!("decode failed: {e}"))
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
            info!("qwen35 rank {}: skipping warmup (relay)", self.rank);
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
                Ok(_) => info!("qwen35 warmup ok (stage0 local)"),
                Err(e) => warn!(error = %e, "qwen35 warmup failed"),
            }
            self.reset_all();
            if let Err(e) = self.handshake_a() {
                warn!(error = %e, "qwen35 startup handshake failed");
            }
            return;
        }
        self.reset_all();
        match self.run_span(&[1000], 0) {
            Ok(_) => {
                info!("qwen35 warmup ok");
                self.observe_warmup_state();
            }
            Err(e) => warn!(error = %e, "qwen35 warmup failed"),
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
        // The API bounds request BYTES; this bounds tokens against the model's window so a
        // multi-MB prompt is a 413, not an hours-long prefill holding the batch=1 slot.
        if let Some(max) = self.max_ctx {
            if let Some(tok) = &self.tokenizer {
                // The SEQUENCE admission prefills, not the bare prompt: it also carries the
                // injected think block and the resume prefix, either of which can push a prompt
                // that passed this guard past the window.
                if let Ok(ids) = admitted_prompt_ids(tok, &task) {
                    let n = ids.len() + task.resume_ids().map_or(0, <[i32]>::len);
                    if n >= max {
                        return Err(EngineError::PromptTooLong(format!(
                            "prompt is {n} tokens; the model's window is {max} (max_position_embeddings)"
                        )));
                    }
                }
            }
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
        //
        // Task 46: a relay whose engine failed (a stage request abandoned busy) must not keep
        // the link open, acking RESETs and dropping FORWARDs: the head would wait out its reply
        // bound on every task. It returns a connection-fatal error instead, before and after
        // any frame, so `run_relay_loop` exits for a supervisor rebuild and the head sees the
        // socket close at once. No new frame: any peer build handles a closed socket.
        if let Some(e) = self.relay_engine_failed() {
            return Err(e);
        }
        let result = self.step_pipe_relay().map(|_| Vec::new()).and_then(|out| {
            match self.relay_engine_failed() {
                Some(e) => Err(e),
                None => Ok(out),
            }
        });
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
            self.want_final_hidden = false;
            self.last_final_hidden = None;
            // A prefix-cache hit restores over the live requests without its own reset, so the
            // only certified restore sequence is reset → prime → set_state: reset here rather
            // than leaving the cancelled task's state in place.
            self.reset_all();
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

    #[cfg(feature = "kv_coord")]
    fn kv_holder(&self) -> Option<std::sync::Arc<dyn cascadia_engine::KvSnapshotHolder>> {
        if self.stages.is_empty() {
            return None;
        }
        Some(std::sync::Arc::new(crate::kv_coordination::OvKvHolder {
            cache: std::sync::Arc::clone(&self.kv_share),
            model_fp: self.kv_fingerprint(),
        }))
    }

    /// Not gated on `stages` like the two above: a stage-less rank is emb-only rank 0, which is the
    /// head — it warms through `take_warm`, never through a RESTORE frame, so it is never parked into.
    #[cfg(feature = "kv_coord")]
    fn kv_handoff(&self) -> Option<std::sync::Arc<dyn cascadia_engine::KvWarmHandoff>> {
        Some(std::sync::Arc::clone(&self.kv_handoff)
            as std::sync::Arc<dyn cascadia_engine::KvWarmHandoff>)
    }

    fn failure(&self) -> cascadia_engine::EngineFailure {
        self.failure.clone()
    }
}

#[cfg(feature = "kv_coord")]
impl Qwen36Engine {
    /// Drain the plane hand-off mailbox and apply the parked slice, if any. See
    /// [`crate::kv_coordination::drain_handoff`]; called from the `InFrame::Restore` handler for the
    /// same two reasons ov-runtime calls it from `OPCODE_RESTORE`: it lands before the turn's forward,
    /// and it is the only site on the same stream as the commit that parks the slice.
    ///
    /// Position 0 because this engine has no KV cursor to protect — see the note in
    /// `apply_warm_resume`, where the same absence is why the plane depth is logged, not guarded.
    fn drain_kv_handoff(&mut self, expected_epoch: u64) -> bool {
        let mailbox = std::sync::Arc::clone(&self.kv_handoff);
        let fp = self.kv_fingerprint();
        crate::kv_coordination::drain_handoff(&mailbox, fp, 0, expected_epoch, |blob| {
            self.restore_local_stages(blob)
        })
    }
}

impl Qwen36Engine {
    /// Admission on the single-process path when the plane had no warm blob for this turn: it is
    /// not compiled in, it held nothing, or the blob it held FAILED to restore (that arm scrubs
    /// the half-applied state first, then falls through here). Restores the longest cached prefix
    /// of `prompt_ids` over the live requests and returns the resume depth, else cold-resets and
    /// returns 0. The live requests are either already scrubbed or, since Task 55, the previous
    /// decision pass's primed ones (its finish deferred the scrub). A caller that is not a
    /// decision warm hit must run the deferred scrub (`scrub_deferred_decision`) first.
    fn local_warm_or_cold(&mut self, prompt_ids: &[u32]) -> usize {
        if self.prefix_cache.enabled() {
            if let Some((parts, len)) = self.prefix_cache.longest_prefix(prompt_ids) {
                let started = Instant::now();
                // Task 55 (H-lite): a finished decision pass left its primed requests live. The
                // default restore writes over them with no recreate and no priming fold: the
                // shim's `set_state_blob` requires the blob's state count to equal the request's
                // and every state to apply, else it fails and the cold path below recreates. A
                // pre-restore clear (the A/B knob) would `reset_state` a restored request, which
                // leaves residue: scrub first then, as before Task 55.
                if self.decision_scrub_deferred && restore_clear_mode() != "none" {
                    self.reset_all();
                }
                if self.restore_local_parts(&parts) {
                    // Every state now holds the restored prefix: nothing of the previous pass
                    // is left to scrub.
                    self.decision_scrub_deferred = false;
                    self.timing.restore = started.elapsed();
                    self.last_restore_bytes = parts.iter().map(Vec::len).sum();
                    // Attention depth from the blob, not the token count (see kv_seq_from_blob).
                    let warm = crate::prefix_cache::kv_seq_from_parts(&parts)
                        .map(|s| s.min(len))
                        .unwrap_or(len);
                    let (hits, misses) = self.prefix_cache.stats();
                    info!(
                        warm_prefix = warm,
                        matched = len,
                        prompt = prompt_ids.len(),
                        restore_ms = started.elapsed().as_millis() as u64,
                        hits,
                        misses,
                        "qwen35 prefix-cache hit"
                    );
                    return warm;
                }
                warn!(
                    matched = len,
                    "qwen35 prefix-cache restore failed; cold prefill"
                );
            }
        }
        self.reset_all();
        0
    }

    /// Snapshot the chain state under `key` into the prefix cache (best-effort). The copy costs
    /// ~2 s per GB of state, so `prefix_cache::plan_snapshots` no longer routes a boundary here
    /// once one at or below that length has been refused for exceeding the budget: the first
    /// refusal is what measures the threshold, and `PrefixCache::insert` reports it.
    fn prefix_cache_snapshot(&mut self, task_id: &TaskId, key: Vec<u32>) {
        let started = Instant::now();
        let pos = key.len();
        if self.stages_dirty {
            warn!(task = %task_id, pos, "qwen35 prefix-cache: stages dirty after a failed reset; not snapshotting");
            return;
        }
        match self.parts_local_stages() {
            Ok(parts) => {
                let bytes: usize = parts.iter().map(Vec::len).sum();
                debug_assert_eq!(parts.len(), self.stages.len());
                let stored = self.prefix_cache.insert(key, parts);
                let secs = started.elapsed().as_secs_f64();
                self.timing.snapshot += Duration::from_secs_f64(secs);
                if bytes > 0 && secs > 0.0 {
                    self.refresh_costs.snapshot_s_per_byte = Some(secs / bytes as f64);
                }
                if let Some(t) = self.active.as_mut() {
                    t.snapshot_secs += secs;
                }
                info!(
                    task = %task_id,
                    pos,
                    bytes,
                    stored,
                    snapshot_ms = (secs * 1000.0) as u64,
                    live_mib = self.prefix_cache.live_bytes() >> 20,
                    entries = self.prefix_cache.len(),
                    "qwen35 prefix-cache snapshot"
                );
            }
            Err(e) => {
                warn!(task = %task_id, pos, error = %e, "qwen35 prefix-cache snapshot failed")
            }
        }
    }

    /// Snapshot every local stage's OV KV state into one framed opaque blob (emb is stateless).
    /// `None` if any stage can't snapshot (e.g. stub build) — capture degrades to cold reprefill,
    /// which is best-effort on the plane path, so the reason stays at `debug!` here.
    fn blob_local_stages(&mut self) -> Option<Vec<u8>> {
        match self.parts_local_stages() {
            Ok(parts) => Some(crate::prefix_cache::frame_blobs(&parts)),
            Err(e) => {
                tracing::debug!(error = %e, "qwen36: get_state_blob skipped (no KV capture)");
                None
            }
        }
    }

    /// Decision trees with the prefix cache on: copy the warmup's one-token state (the fixed
    /// recurrent state plus one token of KV, one copy of tens of MB) and read the snapshot size
    /// model from it, without caching it: each KV state's bytes per token and the fixed rest
    /// ([`crate::prefix_cache::PrefixCache::observe_state`]), so the size estimate is exact from
    /// the first request (P3-CE-1, P4-CE-1). Chat-only trees skip the copy.
    fn observe_warmup_state(&mut self) {
        if !self.prefix_cache.enabled() || self.decision_assets.is_none() || self.stages_dirty {
            return;
        }
        match self.parts_local_stages() {
            Ok(parts) => {
                let bytes: usize = parts.iter().map(Vec::len).sum();
                match self.prefix_cache.observe_state(&parts) {
                    Some((fixed, per_token)) => info!(
                        bytes,
                        fixed,
                        per_token,
                        "qwen35 prefix-cache: measured the snapshot size model from the warmup state"
                    ),
                    None => warn!(
                        bytes,
                        "qwen35 prefix-cache: the warmup state has no readable KV layout; only a \
                         lower bound on a decision snapshot's size is known until one is taken"
                    ),
                }
            }
            Err(e) => warn!(error = %e, "qwen35 prefix-cache: could not measure the state size"),
        }
    }

    /// One `get_state_blob` per local stage, unframed (the prefix cache stores these as-is; the
    /// kv_coord plane frames them with [`crate::prefix_cache::frame_blobs`]). The failure reason
    /// is returned rather than logged: it is the only signal the always-on prefix cache has when
    /// a snapshot does not happen, and that caller reports it at `warn!`.
    fn parts_local_stages(&mut self) -> Result<Vec<Vec<u8>>, String> {
        let mut blobs = Vec::with_capacity(self.stages.len());
        for st in self.stages.iter_mut() {
            match st.get_state_blob() {
                Ok(b) => blobs.push(b),
                Err(e) => return Err(e.to_string()),
            }
        }
        if blobs.is_empty() {
            return Err("no local stages".into());
        }
        Ok(blobs)
    }

    /// Restore each local stage from a framed blob (inverse of [`Self::blob_local_stages`]).
    fn restore_local_stages(&mut self, blob: &[u8]) -> bool {
        let Some(parts) = crate::prefix_cache::unframe_blobs(blob) else {
            return false;
        };
        // The kv_coord restores (take_warm, the plane apply and hand-off) never write over a
        // finished decision pass's live requests (Task 55): its admission scrubbed already, and
        // this is the backstop for a path that did not.
        self.scrub_deferred_decision();
        self.restore_local_parts(&parts)
    }

    /// Restore one per-stage state blob per local stage. Order matters on the GPU plugin: any
    /// pre-restore clear (the `CASCADIA_QWEN36_RESTORE_CLEAR` A/B knob) leaves a request un-run,
    /// and a `set_state_blob` onto an un-run request is silently dropped on the next inference —
    /// so the clear pass comes first, then ONE priming fold, then the writes (see `primed`).
    fn restore_local_parts(&mut self, parts: &[Vec<u8>]) -> bool {
        if parts.len() != self.stages.len() {
            return false;
        }
        // Default: write over the LIVE requests, no pre-clear. See restore_clear_mode: clearing is
        // what broke bar #1, and `set_state_blob` overwrites every VariableState, so there is no
        // residue for a clear to scrub. Logged so an A/B can tell a no-op arm from an env var that
        // never reached the node.
        let mode = restore_clear_mode();
        info!(mode, "qwen36_restore_clear");
        if mode == "recreate_request" || mode == "reset_state" {
            for st in self.stages.iter_mut() {
                let clear = if mode == "recreate_request" {
                    st.recreate_request()
                } else {
                    st.reset_state()
                };
                if let Err(e) = clear {
                    warn!(error = %e, mode, "qwen36: pre-restore state clear failed; cold reprefill");
                    self.state_restored = true;
                    return false;
                }
            }
            self.primed = false;
        }
        // Prime un-run requests: one dummy single-token fold at position 0 so the plugin has
        // materialised its state buffers — otherwise the writes below are dropped on the next
        // inference. The fold's garbage state is overwritten by the restore.
        if !self.primed {
            let zeros = vec![0f32; self.hidden];
            let t_prime = Instant::now();
            let primed = self.chain_pass(&zeros, 0, 1);
            self.timing.prime += t_prime.elapsed();
            if let Err(e) = primed {
                warn!(error = %e, "qwen36: priming fold before restore failed; cold reprefill");
                self.state_restored = true;
                return false;
            }
        }
        for (st, part) in self.stages.iter_mut().zip(parts.iter()) {
            // The per-tensor discriminator was written for THIS bug and had no qwen36 caller, so a
            // full tensor-dump rig run produced zero lines. Logs (name, rank, seq, nbytes, digest) per
            // state so an A-capture vs B-restore diff can say WHICH tensor differs — attention-only
            // points at layout mapping, conv/ssm-only at recurrent state handled as sequence-addressable.
            #[cfg(feature = "kv_coord")]
            crate::kv_coordination::log_blob_tensors("qwen36_restore", 0, part);
            let t_set = Instant::now();
            let set = st.set_state_blob(part);
            self.timing.set_state += t_set.elapsed();
            if let Err(e) = set {
                warn!(error = %e, "qwen36: set_state_blob failed; cold reprefill");
                // A partial apply still dirtied earlier stages — make the next reset scrub properly.
                self.state_restored = true;
                return false;
            }
            // Issue-34 diag (gated: extra get_state_blob copy per restored stage): does the OV state
            // round-trip at the DECLARED level? get_state_blob right after set — if it differs from
            // `part`, set_state is lossy (fixable serialization). If it matches yet warm still flips vs
            // cold, the delta is OV-internal (blocked layout / higher precision) = a floor at this OV version.
            #[cfg(feature = "kv_coord")]
            if postprefill_fp() {
                match st.get_state_blob() {
                    Ok(rt) => {
                        let set_fnv = crate::kv_coordination::fnv1a64(part);
                        let rt_fnv = crate::kv_coordination::fnv1a64(&rt);
                        if set_fnv != rt_fnv {
                            warn!(
                                set_fnv,
                                rt_fnv,
                                set_len = part.len(),
                                rt_len = rt.len(),
                                "qwen36_state_roundtrip_mismatch (set_state lossy at declared level)"
                            );
                        } else {
                            info!(
                                fnv = set_fnv,
                                len = part.len(),
                                "qwen36_state_roundtrip_exact (declared state faithful)"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "qwen36: get_state_blob for round-trip diag failed")
                    }
                }
            }
        }
        // Warm side of CASCADIA_QWEN36_FP_AT: the state as restored, before any suffix fold. Compare
        // against the cold path's "qwen36_fp_at (cold fold)" at the same pos.
        #[cfg(feature = "kv_coord")]
        if fp_at().is_some() {
            let fps: Vec<u64> = self
                .stages
                .iter_mut()
                .map(|st| {
                    st.get_state_blob()
                        .map(|b| crate::kv_coordination::fnv1a64(&b))
                        .unwrap_or(0)
                })
                .collect();
            info!(?fps, "qwen36_fp_at (warm restored)");
        }
        self.state_restored = true;
        true
    }
}

#[cfg(feature = "kv_coord")]
impl Qwen36Engine {
    /// MODEL-level fingerprint (the manifest alone — identical on every rank of a tree), NOT per-rank.
    /// A cross-chain pull asserts ONE fingerprint — the entry head's — for EVERY rank's GET, so all
    /// ranks must share it; folding in `rank`/`total` made rank>0 GETs reject a legitimate move.
    /// Per-rank slice selection is by the dial INDEX (rank N → that rank's holder), and a sharding
    /// mismatch degrades safely (the opaque blob's `set_state` size-rejects ⇒ cold), so the stage span
    /// is not needed as a guard. Mirrors `kv_coordination::kv_model_fingerprint`.
    fn kv_fingerprint(&self) -> u64 {
        crate::kv_coordination::fnv1a64(self.manifest_json.as_bytes())
    }

    /// Capture this rank's local KV under `tokens` (head/single-box token-keyed path). Called at the
    /// top of `finalize`, before `reset_all` wipes the state. Best-effort.
    fn kv_capture_local(&mut self, tenant: &str, tokens: Vec<i32>) {
        if tokens.is_empty() {
            return;
        }
        if let Some(blob) = self.blob_local_stages() {
            // Mirror into the lock-free holder cache so a busy node serves this turn's KV unblocked.
            self.kv_share
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .capture(tenant, tokens.clone(), blob.clone());
            self.kv.capture(tenant, tokens, blob);
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
    fn lookup(&mut self, partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        self.kv.lookup(partner, token_ids)
    }
    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(cascadia_kv_wire::Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let fp = self.kv_fingerprint();
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
        // Mirror into the lock-free holder cache, exactly as `kv_capture_local` does for a locally
        // captured turn. Without this a warm-PULLED node can use the KV itself but cannot serve it
        // onward — `OvKvHolder` reads the share, not `self.kv` — so a chained move A→B→C goes cold
        // at C even though B is holding the bytes C needs. Pulled KV is a valid holding; D10 has
        // replicas serve. Tenant-tagged with the ASSERTED partner, same as the line below.
        self.kv_share
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert_both(partner, tokens.clone(), blob.clone());
        // H.1b hard gate (§12.10.0a): key on the ASSERTED partner, never `manifest.partner`, which
        // the serving holder stamps and nothing validates.
        self.kv.insert_both(partner, tokens, blob);
        Ok(())
    }

    fn stash_downstream_rank(
        &mut self,
        rank: u16,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        // Without this the trait default returns Err ⇒ the head DROPS every downstream rank's pulled
        // blob, that rank finds no capture for the foreign epoch, votes cold, and the all-or-nothing
        // verdict collapses the whole chain to "restore incomplete; cold reset". Keyed by the content
        // epoch so it matches the RESTORE the head sends. 2-stage today: only the head pulls all
        // ranks, so a middle rank has nothing to carry to ITS downstream (3+ stages stay cold).
        let (_tokens, blob) = crate::kv_coordination::wire_to_blob(manifest, payloads).ok_or(())?;
        let epoch = crate::kv_coordination::synth_epoch(&manifest.token_ids);
        self.kv.stash_downstream(epoch, rank, blob);
        Ok(())
    }

    fn apply_warm_resume(&mut self, epoch: u64) -> bool {
        // Plane path (§0(B), multi-rank downstream): the pull staged this rank's slice under `epoch`;
        // restore it now. Mirrors the InFrame::Restore handler's local apply. Not on the total=1 path
        // (the head warms its own rank-0 slice via take_warm in step_first).
        match self.kv.take_capture(epoch) {
            Some((_, blob)) => {
                // The head computes `warm` from RANK 0's slice alone and ships it to every rank as the
                // FORWARD `pos`; this rank never computes its own restored depth, so a stage whose
                // attention shape[2] differs folds the suffix at the wrong absolute offset (wrong
                // RoPE/mask) with nothing to catch it. Within one healthy chain all ranks fold the same
                // tokens, so depths agree by construction; the plane can break that, because its epoch
                // is content-keyed (`synth_epoch(prefix)`) and two ranks may legitimately pull from
                // DIFFERENT donor chains that captured at different lengths.
                //
                // A guard is cheap and does NOT need new wire fields: the head's value already arrives
                // as the FORWARD header `pos` (see frame_header / the FORWARD parse), so stashing this
                // depth and comparing on the next FORWARD would do it. Logged rather than guarded only
                // because it is not on bar #1's path. Note a mis-phased conv/ssm state is invisible to
                // any such check — those carry no depth at all.
                //
                // (ov-runtime's `kv_handoff_too_late` is NOT this guard: it compares its own advancing
                // `position` against blob depth on a single-rank engine. qwen36 has no `self.position`.)
                let depth = crate::kv_coordination::kv_seq_from_framed_blob(&blob).unwrap_or(0);
                info!(epoch, rank_depth = depth, blob_len = blob.len(),
                    "qwen36: plane apply — THIS rank's restored depth (compare vs head warm_prefix)");
                self.restore_local_stages(&blob)
            }
            None => false,
        }
    }

    fn abort_warm_resume(&mut self, epoch: u64) {
        // Drop a STAGED slice (trigger ran, no commit) so a later commit can't resurrect it, and scrub
        // an APPLIED one back to cold. `reset_all` is the same scrub the cold-admit path uses;
        // `state_restored` upgrades the following reset to a rebuild (reset_state alone leaves residue).
        let _ = self.kv.take_capture(epoch);
        // Retract pulled DOWNSTREAM stashes too — they key by a different epoch than the abort's,
        // and an orphan from an aborted move would ride the single-slot fallback on a later,
        // unrelated RESTORE (another session's KV served warm with an all-green verdict).
        self.kv.clear_downstream();
        self.state_restored = true;
        self.reset_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// The `what` of the bounded call whose helper thread "cannot be started" on this test's
        /// thread ([`run_bounded`]).
        static REFUSE_SPAWN: std::cell::Cell<Option<&'static str>> =
            const { std::cell::Cell::new(None) };
        /// This test thread's `CASCADIA_QWEN36_RESTORE_CLEAR` ([`restore_clear_mode`]).
        static RESTORE_CLEAR: std::cell::Cell<Option<&'static str>> =
            const { std::cell::Cell::new(None) };
    }

    pub(super) fn spawn_refused(what: &str) -> bool {
        REFUSE_SPAWN.with(|c| c.get() == Some(what))
    }

    pub(super) fn restore_clear_knob() -> Option<&'static str> {
        RESTORE_CLEAR.with(|c| c.get())
    }

    #[test]
    fn capability_bits_degrade_to_old_ranks() {
        let own = own_caps(true);
        assert!(
            !chain_decision(own_caps(false)),
            "a rank that cannot serve decisions clears the bit"
        );
        assert_eq!(own_caps(false) & CAP_CAPTURE_V2, own & CAP_CAPTURE_V2);
        assert!(
            chain_decision(own),
            "a new build advertises decision frames"
        );
        // an old last rank replies its capture floor (0 or 1): bit 1 is clear either way
        assert!(!chain_decision(combine_caps(own, 0)));
        assert!(!chain_decision(combine_caps(own, 1)));
        // all-new chain keeps both bits it has
        assert_eq!(combine_caps(own, own), own);
        // capture bit keeps today's meaning
        assert_eq!(combine_caps(own, 1) & CAP_CAPTURE_V2, own & CAP_CAPTURE_V2);
    }

    #[test]
    fn decision_frame_codes_are_distinct() {
        // CAPTURE/RESTORE codes (8..=12) exist only under the kv_coord feature (off by default).
        let decision = [
            FRAME_DECIDE,
            FRAME_DECIDE_ACK,
            FRAME_DECIDE_RUN,
            FRAME_DECISION,
        ];
        let mut all = vec![
            FRAME_HELLO,
            FRAME_HELLO_ACK,
            FRAME_HELLO_NAK,
            FRAME_RESET,
            FRAME_RESET_ACK,
            FRAME_FORWARD,
            FRAME_TOKEN,
        ];
        all.extend(decision);
        #[cfg(feature = "kv_coord")]
        all.extend([
            FRAME_CAPTURE,
            FRAME_CAPTURE_ACK,
            FRAME_RESTORE,
            FRAME_RESTORE_ACK,
            FRAME_CAPTURE_V2,
        ]);
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n);
        assert!(
            decision.iter().all(|c| !(8..=12).contains(c)),
            "decision codes must not reuse kv_coord codes"
        );
    }

    #[test]
    fn ov_properties_reach_plugin_config() {
        // The --ov-* CLI flags must land in the PluginConfig handed to compile, alongside the
        // cache/KV wiring, and repeated with_ov_properties calls must accumulate, not replace.
        let b = Qwen36Builder::new("/x", "GPU")
            .with_cache_dir("/tmp/ovc")
            .with_kv_cache_precision("u8")
            .with_ov_properties(vec![("INFERENCE_NUM_THREADS".into(), "8".into())])
            .with_ov_properties(vec![("PERFORMANCE_HINT".into(), "LATENCY".into())]);
        let cfg = b.build_plugin_config();
        let has = |k: &str, v: &str| cfg.entries.iter().any(|(ek, ev)| ek == k && ev == v);
        assert!(has("CACHE_DIR", "/tmp/ovc"));
        assert!(has("KV_CACHE_PRECISION", "u8"));
        assert!(has("INFERENCE_NUM_THREADS", "8"));
        assert!(has("PERFORMANCE_HINT", "LATENCY"));
    }

    #[test]
    fn header_roundtrip() {
        let h = frame_header(FRAME_FORWARD, 7, 4096);
        assert_eq!(parse_header(&h), (FRAME_FORWARD, 7, 4096));
    }

    /// A Qwen3.6-era tree (no hidden_size) keeps the 2048 width; a dense
    /// Qwen3.8 tree carries 5120 and the `qwen3_5` arch.
    #[test]
    fn manifest_hidden_defaults_to_legacy_and_reads_dense() {
        let legacy = Manifest::parse(r#"{"arch":"qwen3_5_moe","stages":[]}"#).unwrap();
        assert_eq!(legacy.hidden(), 2048);
        assert!(!legacy.last_logits_only);
        let dense = Manifest::parse(
            r#"{"arch":"qwen3_5","hidden_size":5120,"num_layers":64,"last_logits_only":true,
                "stages":[{"stage":0,"layer_start":0,"layer_end":31}]}"#,
        )
        .unwrap();
        assert_eq!(dense.hidden(), 5120);
        assert!(dense.last_logits_only);
        assert_eq!(dense.stages.len(), 1);
    }

    /// 2048 is the width of the trees that shipped BEFORE the key existed, all of them
    /// `qwen3_5_moe`. Defaulting a dense tree to it produces a manifest that loads and then dies
    /// at the first inference on an opaque OV shape error (the IR emits [1,n,5120]).
    #[test]
    fn manifest_requires_hidden_size_off_the_legacy_arch() {
        let err = Manifest::parse(r#"{"arch":"qwen3_5","stages":[]}"#).unwrap_err();
        assert!(err.contains("hidden_size"), "{err}");
        assert!(
            err.contains("qwen3_5"),
            "the error must name the arch: {err}"
        );
    }

    /// The chain is built in manifest order, so a mis-ordered entry would chain the stages wrong
    /// (and, in pipeline mode, hand a rank the other rank's layers) without a word.
    #[test]
    fn manifest_rejects_stages_out_of_order() {
        let err = Manifest::parse(
            r#"{"arch":"qwen3_5","hidden_size":5120,"stages":[
                {"stage":1,"layer_start":32,"layer_end":63},
                {"stage":0,"layer_start":0,"layer_end":31}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("stage"), "{err}");
    }

    #[test]
    fn manifest_rejects_foreign_arch_and_zero_width() {
        let err = Manifest::parse(r#"{"arch":"llama","stages":[]}"#).unwrap_err();
        assert!(err.contains("not a Qwen3.5-family"), "{err}");
        let err = Manifest::parse(r#"{"arch":"qwen3_5","hidden_size":0,"stages":[]}"#).unwrap_err();
        assert!(err.contains("hidden_size"), "{err}");
    }

    #[test]
    fn read_max_ctx_prefers_nested_text_config_and_is_none_when_unstated() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_max_ctx(dir.path()), None, "no config.json");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"qwen3_5","text_config":{"max_position_embeddings":262144},"max_position_embeddings":4096}"#,
        )
        .unwrap();
        assert_eq!(read_max_ctx(dir.path()), Some(262144));
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"qwen3_5_moe","max_position_embeddings":40960}"#,
        )
        .unwrap();
        assert_eq!(read_max_ctx(dir.path()), Some(40960));
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        assert_eq!(read_max_ctx(dir.path()), None, "key absent");
        std::fs::write(dir.path().join("config.json"), "{ not json").unwrap();
        assert_eq!(read_max_ctx(dir.path()), None, "unparseable config.json");
    }

    fn bare_engine(total: u32, manifest: &str) -> Qwen36Engine {
        Qwen36Engine {
            emb: None,
            stages: Vec::new(),
            tokenizer: None,
            eos: None,
            max_tokens_default: 256,
            last_logits_only: false,
            hidden: LEGACY_HIDDEN,
            rank: 1,
            total,
            upstream: None,
            downstream: None,
            runtime_handle: None,
            manifest_json: manifest.to_string(),
            epoch: 0,
            peer_epoch: 0,
            handshake_done: false,
            chain_capture_v2: 0,
            chain_caps: 0,
            poisoned: None,
            down_link: DownLink::Clean,
            pending: Vec::new(),
            active: None,
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
            #[cfg(feature = "kv_coord")]
            plane_restore: std::env::var("CASCADIA_KV_PLANE_RESTORE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            state_restored: false,
            #[cfg(feature = "kv_coord")]
            kv_share: std::sync::Arc::new(std::sync::Mutex::new(
                crate::kv_coordination::OvKvCache::default(),
            )),
            #[cfg(feature = "kv_coord")]
            kv_handoff: std::sync::Arc::new(crate::kv_coordination::KvHandoffMailbox::new()),
            prefix_cache: crate::prefix_cache::PrefixCache::new(0),
            im_start_id: None,
            primed: false,
            stages_dirty: false,
            scrub_pending: false,
            decision_scrub_deferred: false,
            infer_failed: false,
            retry: None,
            no_retry: false,
            engine_failed: None,
            failure: Default::default(),
            stage_idle_timeout: STAGE_IDLE_TIMEOUT,
            rebuild: Rebuild::new(None, Vec::new(), MAX_RECOMPILES),
            fresh_requests: Vec::new(),
            max_ctx: None,
            refresh_costs: Default::default(),
            last_restore_bytes: 0,
            final_hidden_output: false,
            decision_assets: None,
            vocab: None,
            want_final_hidden: false,
            last_final_hidden: None,
            pending_decision: None,
            last_logits: Vec::new(),
            timing: PassTiming::default(),
            last_timing: None,
            pass_started: None,
        }
    }

    /// A warm-PULLED node must be able to serve that KV onward.
    ///
    /// `kv_capture_local` mirrors a locally captured turn into the lock-free holder share; `insert`
    /// (the pull path) used to write only `self.kv`. Since `OvKvHolder` serves from the SHARE, a node
    /// that had just pulled 100+ MB could not answer for it — a chained move A→B→C went cold at C
    /// while B was holding exactly the bytes C needed.
    #[cfg(feature = "kv_coord")]
    #[test]
    fn a_warm_pulled_blob_is_servable_onward_from_the_holder_share() {
        use cascadia_engine::KvCoordination;

        let mut e = bare_engine(2, r#"{"arch":"qwen3_5_moe"}"#);
        let tokens = vec![11, 22, 33];
        let blob = vec![0xAB, 0xCD];
        let (manifest, payloads) = crate::kv_coordination::blob_to_wire(
            &tokens,
            &blob,
            "acme",
            e.kv_fingerprint(),
            crate::kv_coordination::synth_epoch(&tokens),
        );

        e.insert("acme", &manifest, &payloads)
            .expect("pull insert must succeed");

        // The share is what the wire path serves from.
        let epoch = crate::kv_coordination::synth_epoch(&tokens);
        let served = e
            .kv_share
            .lock()
            .unwrap()
            .serve("acme", epoch, tokens.len() as u32);
        assert!(
            served.is_some(),
            "a pulled blob must land in the holder share, or this node cannot serve it onward              and a chained move goes cold one hop early"
        );
        // And it stays tenant-confined there.
        assert!(
            e.kv_share
                .lock()
                .unwrap()
                .serve("evil", epoch, tokens.len() as u32)
                .is_none(),
            "sharing must not widen the tenant boundary"
        );
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

    /// H.1b R2. A pre-R2 peer omits `capture_v2` entirely; it must still handshake, and its
    /// `HELLO_ACK` (`pos = 0`) leaves the head's chain floor at 0 so no v2 frame is ever emitted at
    /// it. Two capability-aware builds that disagree refuse instead of desyncing mid-turn.
    #[cfg(feature = "kv_coord")]
    #[test]
    fn hello_accepts_a_legacy_peer_and_refuses_a_capability_disagreement() {
        let e = bare_engine(2, "{}");
        let mut legacy: serde_json::Value =
            serde_json::from_slice(&e.hello_payload()).expect("payload is json");
        legacy.as_object_mut().unwrap().remove("capture_v2");
        assert_eq!(e.validate_hello(legacy.to_string().as_bytes()), None);

        let mut skewed: serde_json::Value =
            serde_json::from_slice(&e.hello_payload()).expect("payload is json");
        skewed["capture_v2"] = serde_json::json!(CAPTURE_V2_CAP + 1);
        let reason = e.validate_hello(skewed.to_string().as_bytes());
        assert!(reason.is_some_and(|r| r.contains("CAPTURE v2 capability")));
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
    /// The extra pieces are the ones `THINK_BLOCK` splits into, so the
    /// window guard's think-block arm is reachable here too.
    fn tiny_tokenizer() -> Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel",
                "vocab":{"hi":0,"<":1,">":2,"</":3,"think":4},"unk_token":"[UNK]"}}"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("build tiny tokenizer")
    }

    /// One `hi` per token, so a prompt's length is its word count.
    fn windowed_task(id: &str, prompt: &str) -> GenerationTask {
        let mut t = GenerationTask::new(id, prompt);
        // Isolate the prompt's own length from the think-block injection.
        t.enable_thinking = true;
        t
    }

    /// "At or past" the window is refused — at-limit is the case the PR promises and the one a
    /// `>` would leak. An unknown window admits everything (no config.json to read it from).
    #[test]
    fn submit_refuses_prompts_at_or_past_the_window() {
        let mut e = bare_engine(1, "{}");
        e.tokenizer = Some(tiny_tokenizer());
        e.max_ctx = Some(4);

        assert!(e.submit(windowed_task("under", "hi hi hi")).is_ok());
        assert!(matches!(
            e.submit(windowed_task("at", "hi hi hi hi")),
            Err(EngineError::PromptTooLong(_))
        ));
        assert!(matches!(
            e.submit(windowed_task("past", "hi hi hi hi hi")),
            Err(EngineError::PromptTooLong(_))
        ));

        e.max_ctx = None;
        assert!(
            e.submit(windowed_task("no-window", "hi hi hi hi hi hi"))
                .is_ok(),
            "an unstated max_position_embeddings must not refuse anything"
        );
    }

    /// The guard must size the sequence admission PREFILLS, not the bare prompt: admission appends
    /// the injected think block and the Option-B resume prefix after the point this used to check,
    /// so a prompt just under the window prefilled past `max_position_embeddings`.
    #[test]
    fn submit_counts_the_think_block_and_the_resume_prefix() {
        let mut e = bare_engine(1, "{}");
        e.tokenizer = Some(tiny_tokenizer());
        e.max_ctx = Some(5);

        // Same 3-token prompt: fits with hybrid reasoning on, refused with it off because
        // admission then prefills the empty think block too.
        assert!(e.submit(windowed_task("thinking", "hi hi hi")).is_ok());
        assert!(
            matches!(
                e.submit(GenerationTask::new("no-thinking", "hi hi hi")),
                Err(EngineError::PromptTooLong(_))
            ),
            "the injected think block is prefilled, so it counts against the window"
        );

        let mut resuming = windowed_task("resuming", "hi hi");
        resuming.resume_token_ids = Some(vec![0, 0, 0]);
        assert!(
            matches!(e.submit(resuming), Err(EngineError::PromptTooLong(_))),
            "the resume prefix is prefilled with the prompt, so it counts too"
        );
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
            tenant: String::new(),
            prompt_ids: vec![1, 2, 3],
            prefill_idx: 3,
            step: 4,
            warm_prefix: 0,
            snapshot_at: Vec::new(),
            logits: Vec::new(),
            next_token: Some(5),
            gen_ids: vec![5],
            emitted: Vec::new(),
            resume_seed_len: 0,
            max_tokens: 16,
            started: Instant::now(),
            snapshot_secs: 0.0,
            wire_ms: Vec::new(),
            decision: None,
            collector: None,
            retry_source: None,
            retried: false,
            retry_resumed: 0,
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

    /// An `ActiveTask` parked at the decode stop, ready for `finalize`.
    fn finalizable_task(max_tokens: usize, gen: usize) -> ActiveTask {
        ActiveTask {
            task_id: "t0".into(),
            tenant: String::new(),
            prompt_ids: vec![1, 2, 3],
            prefill_idx: 3,
            step: 3 + gen,
            warm_prefix: 0,
            snapshot_at: Vec::new(),
            logits: Vec::new(),
            next_token: Some(5),
            gen_ids: vec![5; gen],
            emitted: Vec::new(),
            resume_seed_len: 0,
            max_tokens,
            started: Instant::now(),
            snapshot_secs: 0.0,
            wire_ms: Vec::new(),
            decision: None,
            collector: None,
            retry_source: None,
            retried: false,
            retry_resumed: 0,
        }
    }

    /// A turn cut off at `max_tokens` must say so on the final marker. Without
    /// a `finish_reason` the API layer defaults to `"stop"`, so a truncated
    /// completion reads to the client as a natural end and the retry/continue
    /// logic that keys on `"length"` never fires. Admission also caps
    /// `max_tokens` at the context window, so hitting the window lands here too.
    #[test]
    fn finalize_reports_length_on_the_final_marker() {
        let mut e = bare_engine(1, "{}");
        e.active = Some(finalizable_task(4, 4));

        let out = e.finalize(FinishReason::Length);
        assert_eq!(out.len(), 1);
        let (_, chunk) = &out[0];
        assert!(chunk.is_final);
        assert_eq!(
            chunk.finish_reason,
            Some(FinishReason::Length),
            "a max_tokens stop must not read as a natural stop"
        );
        assert_eq!(
            chunk.prompt_tokens,
            Some(3),
            "the finish reason must not displace the prompt-token count"
        );
        assert!(e.active.is_none());
    }

    /// The EOS path keeps reporting `Stop` — the reason is carried, not guessed.
    #[test]
    fn finalize_reports_stop_on_the_final_marker() {
        let mut e = bare_engine(1, "{}");
        e.active = Some(finalizable_task(16, 1));

        let out = e.finalize(FinishReason::Stop);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1.finish_reason, Some(FinishReason::Stop));
    }

    /// End to end through the local decode loop: a task already at
    /// `max_tokens` stops on the next `step_local` and that marker carries
    /// `Length`. Prefill is complete and the stop check precedes any stage
    /// call, so this runs on a stage-less engine.
    #[test]
    fn local_decode_max_tokens_stop_reports_length() {
        let mut e = bare_engine(1, "{}");
        e.active = Some(finalizable_task(4, 4));

        let out = e.step_local();
        assert_eq!(out.len(), 1);
        let (_, chunk) = &out[0];
        assert!(chunk.is_final);
        assert_eq!(chunk.finish_reason, Some(FinishReason::Length));
    }

    /// `cancel()` must reset the stage requests, not just drop the task. A prefix-cache hit
    /// restores over the LIVE requests with no reset of its own (the certified sequence is
    /// reset → prime → set_state), so a cancelled turn's mid-prefill state would be the base the
    /// next restore writes onto. `reset_all` clears `primed` unconditionally, which is the part
    /// observable without a compiled stage.
    #[test]
    fn cancel_resets_the_stage_requests() {
        let mut e = bare_engine(1, "{}");
        e.primed = true;
        e.active = Some(ActiveTask {
            task_id: "t0".into(),
            tenant: String::new(),
            prompt_ids: vec![1, 2, 3],
            prefill_idx: 2,
            step: 2,
            warm_prefix: 0,
            snapshot_at: Vec::new(),
            logits: Vec::new(),
            next_token: None,
            gen_ids: Vec::new(),
            emitted: Vec::new(),
            resume_seed_len: 0,
            max_tokens: 16,
            started: Instant::now(),
            snapshot_secs: 0.0,
            wire_ms: Vec::new(),
            decision: None,
            collector: None,
            retry_source: None,
            retried: false,
            retry_resumed: 0,
        });

        let id: TaskId = "t0".into();
        e.cancel(&id);

        assert!(e.active.is_none(), "cancel must free the engine slot");
        assert!(
            !e.primed,
            "cancel must reset the stage requests, or the next restore lands on the cancelled \
             turn's mid-prefill state"
        );
    }

    /// Without a mailbox this rank refuses the plane trigger, the head falls back to its local KV,
    /// and the whole plane no-ops on qwen36 (rig: dist-spec PLANE 6/10, `plane_pulled=false`).
    #[cfg(feature = "kv_coord")]
    #[test]
    fn kv_handoff_is_advertised() {
        let e = bare_engine(2, "{}");
        assert!(cascadia_engine::Engine::kv_handoff(&e).is_some());
    }

    /// The handle the plane parks into must be the one the RESTORE drain reads, or the slice stays
    /// parked forever and the rank is cold under a warm head. Only the take is observable off-rig —
    /// the apply needs a compiled stage — so this asserts the retraction can no longer find it.
    #[cfg(feature = "kv_coord")]
    #[test]
    fn kv_handoff_drain_consumes_what_the_plane_parked() {
        let mut e = bare_engine(2, "{}");
        let mb = cascadia_engine::Engine::kv_handoff(&e).unwrap();
        let (manifest, payloads) = crate::kv_coordination::blob_to_wire(
            &[1, 2, 3],
            &[0xAB],
            "acme",
            e.kv_fingerprint(),
            0xE7,
        );
        mb.put(0xE7, manifest, payloads);
        assert!(
            !e.drain_kv_handoff(0xE7),
            "no stage loaded ⇒ the apply cannot arm"
        );
        assert!(!mb.clear(0xE7), "drain must have taken the parked slice");
        // `clear` returning false no longer proves the drain TOOK it: since the drain became
        // epoch-bound, a foreign-epoch take also empties the slot, so this assertion passed with a
        // mismatched epoch too. `epoch_mismatches` is what discriminates, and it must be 0 — the
        // drain above asked for the epoch the slice was parked under.
        assert_eq!(
            e.kv_handoff.epoch_mismatches(),
            0,
            "the slot must have been TAKEN by a matching drain, not dropped as foreign"
        );
    }

    // NOT UNIT-TESTABLE, stated rather than faked: the seven `drain_kv_handoff` CALL SITES forward
    // the RESTORE's epoch, and every guard in `KvHandoffMailbox::take` is inert if a call site hands
    // it the wrong value — pass `epoch ^ 1` at all seven and the whole suite stays green while every
    // plane move colds. The arms live in `step_pipe_relay`/`OPCODE_RESTORE`, which need a live
    // pipeline transport and a compiled stage, so no test in this crate reaches them. A first attempt
    // here called `drain_kv_handoff` directly and was worthless: mutating the real call site left it
    // green. The cover is the rig cert's `kv_handoff_epoch_mismatch` bar (OV engines only) — sparse-moe
    // has no cert cell, so sites 4-7 are uncovered. Do not replace this note with a test that drives
    // the drain directly.

    // NOT UNIT-TESTABLE, stated rather than faked: the admission refusal on `stages_dirty` (and the
    // `recreate_request` escalation `reset_all` takes when it is set) needs a stage whose reset can
    // FAIL. `bare_engine` has zero stages, so `reset_all` succeeds trivially and clears the flag on
    // the way into admission — a test that pre-sets `stages_dirty` asserts nothing about the guard.
    // Certified on hardware only: cancel mid-turn so `recreate_request` fails on a stage, then submit
    // again and expect the error chunk instead of a 200 carrying the previous turn's state.

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

    fn decision_task(
        head: cascadia_types::ReadoutHead,
        cache_prefix: Option<u32>,
    ) -> GenerationTask {
        GenerationTask {
            task_id: "d".into(),
            prompt: "ignored".into(),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            max_tokens: 1,
            temperature: 0.0,
            logprobs: 0,
            sampling: Default::default(),
            enable_thinking: false,
            trust_remote_code: false,
            tenant: String::new(),
            resume_token_ids: None,
            decision: Some(cascadia_types::DecisionReadout {
                prompt_ids: vec![7, 8, 9, 10],
                head,
                cache_prefix,
                require_warm: false,
                needs_snapshot: false,
            }),
        }
    }

    #[test]
    fn decision_admission_uses_exact_ids_and_no_think_block() {
        let mut t = decision_task(
            cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] },
            None,
        );
        assert_eq!(decision_admission(&t), Some(vec![7, 8, 9, 10]));
        t.decision = None;
        assert_eq!(decision_admission(&t), None);
    }

    #[test]
    fn decision_snapshots_only_at_the_shared_prefix() {
        let t = decision_task(
            cascadia_types::ReadoutHead::Pointer {
                decide_pos: 3,
                option_pos: vec![2],
            },
            Some(2),
        );
        let mut d = t.decision.clone().unwrap();
        assert!(
            decision_snapshot_boundaries(&d, true).is_empty(),
            "under MIN_PREFIX_TOKENS: never cached, never copied (P4-CE-2)"
        );
        let min = crate::prefix_cache::MIN_PREFIX_TOKENS;
        d.cache_prefix = Some(min as u32);
        assert_eq!(decision_snapshot_boundaries(&d, true), vec![min]);
        assert!(
            decision_snapshot_boundaries(&d, false).is_empty(),
            "prefix cache disabled"
        );
        let j = decision_task(
            cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] },
            None,
        );
        assert!(decision_snapshot_boundaries(j.decision.as_ref().unwrap(), true).is_empty());
    }

    #[test]
    fn manifest_reads_final_hidden_output() {
        let m = Manifest::parse(
            r#"{"arch":"qwen3_5","hidden_size":4096,"final_hidden_output":true,"stages":[]}"#,
        )
        .unwrap();
        assert!(m.final_hidden_output);
        let m = Manifest::parse(r#"{"arch":"qwen3_5","hidden_size":4096,"stages":[]}"#).unwrap();
        assert!(!m.final_hidden_output);
    }

    #[test]
    fn decision_with_resume_ids_conflicts() {
        let mut t = decision_task(
            cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] },
            None,
        );
        assert!(!decision_resume_conflict(&t));
        t.resume_token_ids = Some(vec![5]);
        assert!(decision_resume_conflict(&t));
        t.decision = None;
        assert!(!decision_resume_conflict(&t), "chat resume is fine");
    }

    fn readout_n(n: u32, head: cascadia_types::ReadoutHead) -> cascadia_types::DecisionReadout {
        cascadia_types::DecisionReadout {
            prompt_ids: (100..100 + n).collect(),
            head,
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        }
    }

    #[test]
    fn decision_warm_probe_bounds_the_restore() {
        use cascadia_types::{JointOption, JointQuestion, ReadoutHead};
        let v = readout_n(10, ReadoutHead::Verbalizer { token_ids: vec![1] });
        assert_eq!(
            decision_warm_probe(&v, &v.prompt_ids).len(),
            10,
            "strict match => <= n-1 restored"
        );
        let p = readout_n(
            10,
            ReadoutHead::Pointer {
                decide_pos: 9,
                option_pos: vec![6, 4],
            },
        );
        assert_eq!(decision_warm_probe(&p, &p.prompt_ids), &p.prompt_ids[..5]);
        let j = readout_n(
            10,
            ReadoutHead::Joint {
                questions: vec![JointQuestion {
                    qtype: 0,
                    span: [0, 1],
                    options: vec![JointOption { span: [1, 2] }],
                }],
            },
        );
        assert!(decision_warm_probe(&j, &j.prompt_ids).is_empty());
    }

    #[test]
    fn decision_warm_probe_hits_the_cache_prefix_and_never_past_it() {
        use cascadia_types::ReadoutHead;
        let mut d = readout_n(
            40,
            ReadoutHead::Pointer {
                decide_pos: 30,
                option_pos: vec![20, 25],
            },
        );
        d.cache_prefix = Some(20);
        let mut pc = crate::prefix_cache::PrefixCache::new(1 << 20);
        for len in [20usize, 21, 25] {
            assert!(pc.insert(d.prompt_ids[..len].to_vec(), vec![vec![0u8; 8]]));
        }
        let (_, len) = pc
            .longest_prefix(decision_warm_probe(&d, &d.prompt_ids))
            .expect("hit");
        assert_eq!(
            len, 20,
            "restores exactly the state prefix, never a row the head reads"
        );
    }

    #[test]
    fn decision_prompt_cap_is_the_min_of_window_and_cap() {
        use cascadia_types::MAX_DECISION_PROMPT;
        assert_eq!(decision_prompt_cap(None), MAX_DECISION_PROMPT);
        assert_eq!(
            decision_prompt_cap(Some(262_144)),
            MAX_DECISION_PROMPT,
            "window never widens"
        );
        assert_eq!(decision_prompt_cap(Some(40_960)), 40_960, "window tightens");
    }

    #[test]
    fn rank0_refuses_a_hidden_readout_on_a_tree_without_final_hidden_before_prefill() {
        use cascadia_types::{DecisionReadout, ReadoutHead};
        // Rank 0 of 2 does not own the head, but HELLO guarantees every rank has this manifest:
        // a Pointer/Joint readout on a tree exported without --final-hidden must be refused here,
        // not after a full chain prefill at the last rank.
        let mut e = bare_engine(2, "{}");
        e.rank = 0;
        e.vocab = Some(16);
        let pointer = DecisionReadout {
            prompt_ids: vec![1, 2, 3],
            head: ReadoutHead::Pointer {
                decide_pos: 2,
                option_pos: vec![0, 1],
            },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        let err = e.admit_decision_check(&pointer).unwrap_err();
        assert!(
            err.starts_with("decision_unsupported") && err.contains("--final-hidden"),
            "{err}"
        );
        // A verbalizer readout reads only logits; rank 0 holds no head assets and needs none.
        let verbalizer = DecisionReadout {
            prompt_ids: vec![1, 2, 3],
            head: ReadoutHead::Verbalizer { token_ids: vec![1] },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        assert!(e.admit_decision_check(&verbalizer).is_ok());
        e.final_hidden_output = true;
        assert!(e.admit_decision_check(&pointer).is_ok());
        // The last rank keeps its own check (defence in depth), plus the assets check.
        e.rank = 1;
        e.final_hidden_output = false;
        assert!(e
            .admit_decision_check(&pointer)
            .unwrap_err()
            .contains("decision_unsupported"));
    }

    #[test]
    fn rank0_refuses_a_chain_that_cannot_serve_decisions_up_front() {
        // QUA-3 / Task 15 ruling: an old rank (or one without vocab or head assets) clears
        // CAP_DECISION, and rank 0 refuses with 501 before any frame goes out.
        use cascadia_types::{DecisionReadout, ReadoutHead};
        let mut e = bare_engine(2, "{}");
        e.rank = 0;
        e.vocab = Some(16);
        let verbalizer = DecisionReadout {
            prompt_ids: vec![1, 2, 3],
            head: ReadoutHead::Verbalizer { token_ids: vec![1] },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        // An old last rank replies its capture floor (0 or 1).
        for old in [0, 1] {
            e.chain_caps = combine_caps(own_caps(e.serves_decisions()), old);
            let why = e.pipe_decision_refusal(&verbalizer).unwrap();
            assert!(why.starts_with("decision_unsupported"), "{why}");
        }
        e.chain_caps = combine_caps(own_caps(true), own_caps(true));
        assert_eq!(e.pipe_decision_refusal(&verbalizer), None);
        // A malformed readout is refused first, as decision_invalid.
        let bad = DecisionReadout {
            prompt_ids: vec![99],
            ..verbalizer.clone()
        };
        assert!(e
            .pipe_decision_refusal(&bad)
            .unwrap()
            .starts_with("decision_invalid"));
    }

    fn kev_pass(n: usize, cache_prefix: u32) -> cascadia_types::DecisionReadout {
        cascadia_types::DecisionReadout {
            prompt_ids: (0..n as u32).collect(),
            head: cascadia_types::ReadoutHead::Pointer {
                decide_pos: n as u32 - 1,
                option_pos: vec![n as u32 - 3, n as u32 - 2],
            },
            cache_prefix: Some(cache_prefix),
            require_warm: false,
            needs_snapshot: false,
        }
    }

    #[test]
    fn a_require_warm_pass_that_missed_its_snapshot_is_refused() {
        // SEC-3: a later Kev pass admitted on the shared-prefix discount must not re-prefill the
        // state when the snapshot is gone. Evicted (the cache could hold it): the retryable 503.
        use crate::prefix_cache::PrefixCache;
        let cache = PrefixCache::new(1 << 20);
        let mut d = kev_pass(30, 20);
        d.require_warm = true;
        let why = decision_cache_refusal(&d, 0, &cache).unwrap();
        assert!(why.starts_with("decision_cold"), "{why}");
        assert!(
            decision_cache_refusal(&d, 19, &cache).is_some(),
            "a shorter restore re-prefills too"
        );
        assert_eq!(decision_cache_refusal(&d, 20, &cache), None);
        assert_eq!(decision_cache_refusal(&d, 21, &cache), None);
        d.require_warm = false;
        assert_eq!(
            decision_cache_refusal(&d, 0, &cache),
            None,
            "cold is fine when not required"
        );
        // Pipelines never resume a decision prefix, nor snapshot one.
        d.require_warm = true;
        let mut e = bare_engine(2, "{}");
        e.rank = 0;
        e.vocab = Some(64);
        e.final_hidden_output = true;
        e.chain_caps = own_caps(true);
        assert!(e
            .pipe_decision_refusal(&d)
            .unwrap()
            .starts_with("decision_invalid"));
        d.require_warm = false;
        d.needs_snapshot = true;
        assert!(e
            .pipe_decision_refusal(&d)
            .unwrap()
            .starts_with("decision_invalid"));
    }

    #[test]
    fn a_first_pass_whose_snapshot_can_never_fit_is_refused_before_prefill() {
        // P2-CE-2 / P2-SEC-1/2: the first pass of a request whose later passes must resume is
        // refused at admission (no prefill) when the cache can never hold its shared prefix, and
        // the API answers 413, not a retryable 503.
        use crate::prefix_cache::{PrefixCache, MIN_PREFIX_TOKENS};
        let uncacheable = |d: &cascadia_types::DecisionReadout, warm, c: &PrefixCache| {
            decision_cache_refusal(d, warm, c)
                .is_some_and(|w| w.starts_with("decision_uncacheable"))
        };
        let mut first = kev_pass(40, 20);
        first.needs_snapshot = true;
        // Cache disabled.
        assert!(uncacheable(&first, 0, &PrefixCache::new(0)));
        // A tiny state: under the shortest key the cache stores.
        let mut tiny = kev_pass(40, (MIN_PREFIX_TOKENS - 1) as u32);
        tiny.needs_snapshot = true;
        assert!(uncacheable(&tiny, 0, &PrefixCache::new(1 << 20)));
        // Measured over the budget: a snapshot that long was already refused.
        let mut small = PrefixCache::new(100);
        assert!(!small.insert(vec![1; 20], vec![vec![0; 101]]));
        assert!(uncacheable(&first, 0, &small));
        // Estimated over the budget from two measured snapshots (fixed 0, 4 bytes per token).
        let mut est = PrefixCache::new(70);
        assert!(est.insert(vec![2; 16], vec![vec![0; 64]]));
        assert!(est.insert(vec![3; 17], vec![vec![0; 68]]));
        assert!(uncacheable(&first, 0, &est), "20 tokens ~ 80 bytes > 70");
        // A cache that can hold it: the first pass runs (cold) and takes the snapshot.
        let ok = PrefixCache::new(1 << 20);
        assert_eq!(decision_cache_refusal(&first, 0, &ok), None);
        assert_eq!(
            crate::prefix_cache::plan_decision_snapshots(&first.prompt_ids, &[20], 0, &ok),
            vec![20]
        );
        // A later pass whose snapshot turned out not to fit is a 413 too, not "retry".
        let mut later = kev_pass(40, 20);
        later.require_warm = true;
        assert!(uncacheable(&later, 0, &small));
        // Passes that do not depend on the cache are never refused.
        assert_eq!(
            decision_cache_refusal(&kev_pass(40, 20), 0, &PrefixCache::new(0)),
            None
        );
    }

    #[test]
    fn a_fresh_engine_refuses_an_uncacheable_first_pass_before_any_prefill() {
        // P3-CE-1: nothing measured yet on a fresh server. The size floor seeded at load (here
        // a 25 MB fixed state against a 1 MiB budget) refuses the first pass with
        // decision_uncacheable at warm 0, before its prefill and copy.
        let cache = new_prefix_cache(1 << 20, Some((25 << 20, 4096)));
        let mut first = kev_pass(9100, 9001);
        first.needs_snapshot = true;
        let why = decision_cache_refusal(&first, 0, &cache).unwrap();
        assert!(why.starts_with("decision_uncacheable"), "{why}");
        // A budget that holds the floor still lets the pass run and snapshot.
        let roomy = new_prefix_cache(1 << 30, Some((25 << 20, 4096)));
        assert_eq!(decision_cache_refusal(&first, 0, &roomy), None);
        // Chat-only or unknown dims: no floor, the cache behaves as before.
        assert!(new_prefix_cache(1 << 20, None).can_hold(9001));
        assert!(!new_prefix_cache(0, Some((1, 1))).enabled());
    }

    #[test]
    fn only_a_needs_snapshot_pass_bypasses_the_chat_refresh_rule() {
        // P3-REG-1: a decision pass that restored a shorter entry (its state extends a cached one)
        // snapshots its own prefix regardless of cost only when its later passes depend on it.
        // Other decision passes keep the chat refresh rule: no multi-GB copy for a few tokens.
        use crate::prefix_cache::{PrefixCache, RefreshCosts};
        let mut cache = PrefixCache::new(1 << 30);
        let mut d = kev_pass(60, 40);
        assert!(cache.insert(d.prompt_ids[..30].to_vec(), vec![vec![0; 8]]));
        let costs = RefreshCosts {
            prefill_tok_s: Some(455.0),
            snapshot_s_per_byte: Some(2.6 / 1.2e9),
        };
        let bytes = 1_200_000_000;
        let plan = |d: Option<&cascadia_types::DecisionReadout>| {
            plan_pass_snapshots(
                d,
                &d.map_or(vec![0; 60], |d| d.prompt_ids.clone()),
                &[40],
                30,
                &cache,
                &costs,
                bytes,
            )
        };
        assert!(
            plan(Some(&d)).is_empty(),
            "unflagged: the refresh rule says no"
        );
        d.require_warm = true;
        assert!(
            plan(Some(&d)).is_empty(),
            "require_warm resumes; it never needs the copy"
        );
        d.require_warm = false;
        d.needs_snapshot = true;
        assert_eq!(plan(Some(&d)), vec![40]);
        // A cold unflagged pass snapshots as a cold chat turn does.
        let cold = kev_pass(60, 40);
        assert_eq!(
            plan_pass_snapshots(
                Some(&cold),
                &cold.prompt_ids,
                &[40],
                0,
                &cache,
                &costs,
                bytes
            ),
            vec![40]
        );
    }

    #[test]
    fn an_unflagged_decision_pass_never_copies_a_snapshot_the_cache_cannot_hold() {
        // P4-CE-2: an unflagged decision pass plans through the chat rule, but a snapshot the
        // cache can never keep (a key under MIN_PREFIX_TOKENS, or one the size estimate already
        // puts over the budget) is not copied: insert would only drop it.
        use crate::prefix_cache::{PrefixCache, RefreshCosts, MIN_PREFIX_TOKENS};
        let costs = RefreshCosts::default();
        let roomy = PrefixCache::new(1 << 30);
        let tiny = kev_pass(40, 5);
        let boundaries = decision_snapshot_boundaries(&tiny, true);
        assert!(boundaries.is_empty(), "a 5-token state is never a boundary");
        assert!(
            plan_pass_snapshots(Some(&tiny), &tiny.prompt_ids, &[5], 0, &roomy, &costs, 0)
                .is_empty(),
            "a key under MIN_PREFIX_TOKENS plans nothing"
        );
        let at_min = kev_pass(40, MIN_PREFIX_TOKENS as u32);
        assert_eq!(
            decision_snapshot_boundaries(&at_min, true),
            vec![MIN_PREFIX_TOKENS]
        );
        // The load-time estimate already puts this state over a 1 MiB budget.
        let small = new_prefix_cache(1 << 20, Some((25 << 20, 4096)));
        let fits = kev_pass(60, 40);
        assert!(
            plan_pass_snapshots(Some(&fits), &fits.prompt_ids, &[40], 0, &small, &costs, 0)
                .is_empty(),
            "an estimate over the budget plans nothing"
        );
        assert_eq!(
            plan_pass_snapshots(Some(&fits), &fits.prompt_ids, &[40], 0, &roomy, &costs, 0),
            vec![40]
        );
        // Chat is unchanged: it plans from its own boundaries without the size estimate.
        assert_eq!(
            plan_pass_snapshots(None, &fits.prompt_ids, &[40], 0, &small, &costs, 0),
            vec![40]
        );
    }

    #[test]
    fn a_state_that_extends_a_cached_state_resumes_its_own_snapshot() {
        // P2-CE-1 / P2-SEC-1: request A cached its state (key 30). Request B's state extends it
        // (cache_prefix 40). B's first pass restores A's entry, plans B's own boundary (no chat
        // refresh rule), and B's require_warm passes then resume at 40: no refusal (a 200).
        use crate::prefix_cache::{plan_decision_snapshots, PrefixCache};
        let mut cache = PrefixCache::new(1 << 20);
        let b_first = {
            let mut d = kev_pass(60, 40);
            d.needs_snapshot = true;
            d
        };
        assert!(cache.insert(b_first.prompt_ids[..30].to_vec(), vec![vec![0; 8]]));
        let warm = cache
            .longest_prefix(decision_warm_probe(&b_first, &b_first.prompt_ids))
            .unwrap()
            .1;
        assert_eq!(warm, 30, "B's first pass restores A's shorter entry");
        assert_eq!(decision_cache_refusal(&b_first, warm, &cache), None);
        let plan = plan_decision_snapshots(&b_first.prompt_ids, &[40], warm, &cache);
        assert_eq!(plan, vec![40]);
        assert!(cache.insert(b_first.prompt_ids[..40].to_vec(), vec![vec![0; 8]]));
        let mut b_later = kev_pass(60, 40);
        b_later.require_warm = true;
        let warm = cache
            .longest_prefix(decision_warm_probe(&b_later, &b_later.prompt_ids))
            .unwrap()
            .1;
        assert_eq!(warm, 40);
        assert_eq!(decision_cache_refusal(&b_later, warm, &cache), None);
    }

    #[test]
    fn a_rank_without_vocab_or_head_assets_does_not_advertise_decisions() {
        // REG-2: a chat-only last rank (no config.json, no decision/) clears CAP_DECISION, so rank
        // 0 refuses up front instead of the last rank refusing after a full chain prefill.
        let mut e = bare_engine(2, "{}");
        e.rank = 0;
        assert!(!e.serves_decisions(), "no vocab");
        e.vocab = Some(16);
        assert!(e.serves_decisions(), "rank 0 needs no head assets");
        e.rank = 1;
        assert!(
            !e.serves_decisions(),
            "the head-owning rank needs decision/ assets"
        );
        e.decision_assets = Some(crate::decision::HeadAssets::Verbalizer);
        assert!(e.serves_decisions());
        // P2-REG-1: a middle rank only relays FRAME_DECIDE; it never reads a vocab or a head, so
        // a manifest + stage-only middle rank still serves decisions.
        let mut mid = bare_engine(3, "{}");
        mid.rank = 1;
        assert!(mid.vocab.is_none() && mid.decision_assets.is_none());
        assert!(mid.serves_decisions(), "middle rank relays only");
        mid.rank = 0;
        assert!(
            !mid.serves_decisions(),
            "rank 0 validates against its vocab"
        );
        mid.rank = 2;
        assert!(!mid.serves_decisions(), "the last rank runs the head");
    }

    #[test]
    fn read_vocab_is_quiet_on_chat_only_ranks() {
        let d = std::env::temp_dir().join(format!("vocab-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(read_vocab(&d, false), None);
        std::fs::write(
            d.join("config.json"),
            r#"{"text_config": {"vocab_size": 7}}"#,
        )
        .unwrap();
        assert_eq!(read_vocab(&d, false), Some(7));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn joint_readout_over_the_shim_input_limit_is_refused_before_prefill() {
        use cascadia_types::{DecisionReadout, JointOption, JointQuestion, ReadoutHead};
        let joint = |n: usize| DecisionReadout {
            prompt_ids: vec![1; n],
            head: ReadoutHead::Joint {
                questions: vec![JointQuestion {
                    qtype: 0,
                    span: [0, 1],
                    options: vec![JointOption { span: [1, 2] }, JointOption { span: [2, 3] }],
                }],
            },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        let mut e = bare_engine(2, "{}");
        e.rank = 0;
        e.vocab = Some(16);
        e.final_hidden_output = true;
        e.hidden = 5120; // Clef-27B: [16384, 5120] f32 is exactly the shim's 320 MiB
        assert!(e.admit_decision_check(&joint(16_384)).is_ok());
        e.hidden = 4096; // Clef-flash: the full 16384 fits
        assert!(e.admit_decision_check(&joint(16_384)).is_ok());
        e.hidden = 8192; // a wider tree: 320 MiB / (4 * 8192) = 10240 rows
        assert!(e.admit_decision_check(&joint(10_240)).is_ok());
        let err = e.admit_decision_check(&joint(10_241)).unwrap_err();
        assert!(err.starts_with("decision_too_long"), "{err}");
    }

    #[test]
    fn owns_head_is_single_box_or_last_rank() {
        assert!(owns_head(0, 1));
        assert!(owns_head(0, 0));
        assert!(!owns_head(0, 2));
        assert!(owns_head(1, 2));
    }

    #[test]
    fn clear_decision_state_drops_every_decision_field() {
        use cascadia_types::{DecisionReadout, HiddenNeed, ReadoutHead};
        let mut e = bare_engine(2, "{}");
        let r = DecisionReadout {
            prompt_ids: vec![1, 2],
            head: ReadoutHead::Verbalizer { token_ids: vec![1] },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        e.want_final_hidden = true;
        e.last_final_hidden = Some(vec![0.5; 8]);
        e.pending_decision = Some((
            3,
            Ok((r, crate::decision::Collector::new(HiddenNeed::None, 4))),
        ));
        e.last_logits = vec![1.0; 8];
        e.clear_decision_state();
        assert!(!e.want_final_hidden);
        assert!(e.last_final_hidden.is_none());
        assert!(e.pending_decision.is_none());
        assert!(e.last_logits.is_empty());
    }

    #[test]
    fn collector_debug_prints_shape_not_rows() {
        use cascadia_types::HiddenNeed;
        let mut c = crate::decision::Collector::new(HiddenNeed::All, 2);
        c.absorb(0, 2, &[0.123_456, 0.654_321, 0.111_111, 0.999_999])
            .unwrap();
        let s = format!("{c:?}");
        assert!(s.contains("hidden: 2"), "{s}");
        assert!(!s.contains("0.123"), "row data leaked: {s}");
    }

    #[test]
    fn take_pending_matches_epoch_and_surfaces_errors() {
        use cascadia_types::{DecisionReadout, HiddenNeed, ReadoutHead};
        let r = DecisionReadout {
            prompt_ids: vec![1, 2],
            head: ReadoutHead::Verbalizer { token_ids: vec![1] },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        let mut p = Some((
            7,
            Ok((
                r.clone(),
                crate::decision::Collector::new(HiddenNeed::None, 4),
            )),
        ));
        assert!(take_pending(&mut p, 8).is_err(), "stale epoch");
        assert!(
            p.is_none(),
            "taken even on mismatch so it cannot leak into the next task"
        );
        let mut p = Some((
            7,
            Ok((r, crate::decision::Collector::new(HiddenNeed::None, 4))),
        ));
        assert!(take_pending(&mut p, 7).is_ok());
        let mut p: Option<(
            u32,
            Result<(DecisionReadout, crate::decision::Collector), String>,
        )> = Some((7, Err("decision_invalid: x".into())));
        assert_eq!(take_pending(&mut p, 7).unwrap_err(), "decision_invalid: x");
        assert!(take_pending(&mut None, 7).is_err(), "nothing pending");
    }

    // ---- B1: a late pipeline reply never desyncs the next task ----

    /// A rank-0 engine wired to a scripted fake rank 1 over loopback TCP. The fake runs on
    /// `rt`; the engine's exchanges block on the same runtime through `run_async`, as in serving.
    fn pipe_rank0(rt: &tokio::runtime::Runtime) -> (Qwen36Engine, ActivationServer) {
        let (client, server) = rt.block_on(async {
            let mut server = ActivationServer::new("127.0.0.1", 0);
            server.start().await.unwrap();
            let mut client = ActivationClient::new("127.0.0.1", server.port());
            let (c, a) = tokio::join!(client.connect(), server.accept());
            c.unwrap();
            a.unwrap();
            (client, server)
        });
        let mut e = bare_engine(2, "{}");
        e.rank = 0;
        e.handshake_done = true;
        e.downstream = Some(Arc::new(tokio::sync::Mutex::new(client)));
        e.runtime_handle = Some(rt.handle().clone());
        (e, server)
    }

    fn pipe_rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    /// Fake rank 1: read one request header, check its kind and epoch, and skip its body.
    async fn fake_expect(s: &mut ActivationServer, kind: u32, epoch: u32) {
        let (k, e, _) = parse_header(&s.recv_raw(12).await.unwrap());
        assert_eq!(
            (k, e),
            (kind, epoch),
            "fake rank 1 got an unexpected request"
        );
        match k {
            FRAME_FORWARD => {
                s.recv().await.unwrap();
            }
            FRAME_DECIDE => {
                recv_decision_body(s, "decide body").await.unwrap();
            }
            _ => {}
        }
    }

    async fn fake_token(s: &mut ActivationServer, epoch: u32, tok: i32) {
        s.send_raw(&frame_header(FRAME_TOKEN, epoch, 0))
            .await
            .unwrap();
        s.send_raw(&tok.to_be_bytes()).await.unwrap();
        s.send_raw(&0u32.to_be_bytes()).await.unwrap();
    }

    async fn fake_ack(s: &mut ActivationServer, kind: u32, epoch: u32) {
        s.send_raw(&frame_header(kind, epoch, 0)).await.unwrap();
    }

    fn one_row(e: &Qwen36Engine) -> Vec<f32> {
        vec![0.0; e.hidden]
    }

    /// The audit's B1 scenario, for TOKEN and both decision replies: the last rank answers after
    /// rank 0's reply bound, so rank 0 fails that task and moves on while the late frame is still
    /// on its way. The next task must read past it to its own reply instead of failing, and the
    /// task after that must not inherit anything either.
    #[test]
    fn a_late_reply_never_desyncs_the_next_task() {
        let _knob = crate::timeout_knob_lock();
        cascadia_transport::set_activation_timeout_secs(1);
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        let late = Duration::from_millis(1600);
        let fake = rt.spawn(async move {
            let s = &mut server;
            // Task 1 (chat): the TOKEN comes back late.
            fake_expect(s, FRAME_FORWARD, 1).await;
            tokio::time::sleep(late).await;
            fake_token(s, 1, 7).await;
            // Task 2 (decision): RESET and DECIDE on time, the DECISION late.
            fake_expect(s, FRAME_RESET, 2).await;
            fake_ack(s, FRAME_RESET_ACK, 2).await;
            fake_expect(s, FRAME_DECIDE, 2).await;
            fake_ack(s, FRAME_DECIDE_ACK, 2).await;
            fake_expect(s, FRAME_DECIDE_RUN, 2).await;
            tokio::time::sleep(late).await;
            fake_ack(s, FRAME_DECISION, 2).await;
            s.send_raw(&crate::decision::encode_result(&Ok(vec![0.25, 0.75])))
                .await
                .unwrap();
            // Task 3 (decision): the DECIDE_ACK comes back late.
            fake_expect(s, FRAME_RESET, 3).await;
            fake_ack(s, FRAME_RESET_ACK, 3).await;
            fake_expect(s, FRAME_DECIDE, 3).await;
            tokio::time::sleep(late).await;
            fake_ack(s, FRAME_DECIDE_ACK, 3).await;
            // Task 4 (chat): everything on time.
            fake_expect(s, FRAME_RESET, 4).await;
            fake_ack(s, FRAME_RESET_ACK, 4).await;
            fake_expect(s, FRAME_FORWARD, 4).await;
            fake_token(s, 4, 42).await;
        });
        let body = crate::decision::encode_result(&Ok(vec![1.0]));

        e.epoch = 1;
        assert!(
            e.send_forward_recv_token(one_row(&e), 1, 0).is_err(),
            "TOKEN past the bound"
        );

        e.epoch = 2;
        e.reset_exchange()
            .expect("the late TOKEN of task 1 must not fail task 2's RESET");
        e.forward_decide_downstream(2, &body)
            .expect("DECIDE acked on time");
        assert!(e.decide_run_exchange().is_err(), "DECISION past the bound");

        e.epoch = 3;
        e.reset_exchange()
            .expect("the late DECISION of task 2 must not fail task 3's RESET");
        assert!(
            e.forward_decide_downstream(3, &body).is_err(),
            "DECIDE_ACK past the bound"
        );

        e.epoch = 4;
        e.reset_exchange()
            .expect("the late DECIDE_ACK of task 3 must not fail task 4's RESET");
        let (tok, _) = e
            .send_forward_recv_token(one_row(&e), 1, 0)
            .expect("task 4 reads its own TOKEN");
        assert_eq!(tok, 42, "task 4 got a stale token");
        cascadia_transport::set_activation_timeout_secs(0);
        rt.block_on(fake).unwrap();
    }

    /// A reply that times out MID-FRAME leaves the stream at an unknown offset: no tag can be
    /// read from there. The link is poisoned and every later task fails fast with the restart
    /// error, without writing to the link, until the pipeline restarts.
    #[test]
    fn a_reply_cut_mid_frame_poisons_the_link_until_restart() {
        let _knob = crate::timeout_knob_lock();
        cascadia_transport::set_activation_timeout_secs(1);
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        let fake = rt.spawn(async move {
            fake_expect(&mut server, FRAME_FORWARD, 1).await;
            // Five bytes of a TOKEN header, then silence past the bound.
            let h = frame_header(FRAME_TOKEN, 1, 0);
            server.send_raw(&h[..5]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(1600)).await;
            server.send_raw(&h[5..]).await.unwrap();
            // Nothing more may arrive: rank 0 must not write to a desynced link.
            let quiet = tokio::time::timeout(Duration::from_millis(500), server.recv_raw(1)).await;
            // Silence or EOF (the transport may drop a socket cut mid-frame), never a byte.
            assert!(
                !matches!(quiet, Ok(Ok(_))),
                "rank 0 wrote to a desynced link"
            );
        });

        e.epoch = 1;
        assert!(e.send_forward_recv_token(one_row(&e), 1, 0).is_err());
        e.epoch = 2;
        let err = e
            .reset_exchange()
            .expect_err("a desynced link must fail fast")
            .to_string();
        assert!(err.contains("desynced") && err.contains("restart"), "{err}");

        // The next admission is refused before any frame goes out.
        e.pending.push(GenerationTask::new("t9", "hello"));
        let out = e.step_pipe_first();
        let msg = out[0].1.error.clone().expect("an error chunk");
        assert!(msg.contains("desynced") && msg.contains("restart"), "{msg}");
        cascadia_transport::set_activation_timeout_secs(0);
        rt.block_on(fake).unwrap();
    }

    /// A slow peer load (the 27B runbook's case): the startup HELLO times out, and the next
    /// admission retries the handshake. The retry reads the first HELLO's reply rather than
    /// sending a second HELLO, whose ACK would otherwise sit in front of the first RESET_ACK.
    #[test]
    fn a_late_handshake_reply_is_awaited_not_resent() {
        let _knob = crate::timeout_knob_lock();
        cascadia_transport::set_activation_timeout_secs(1);
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        e.handshake_done = false;
        let fake = rt.spawn(async move {
            let s = &mut server;
            let (k, _, _) = parse_header(&s.recv_raw(12).await.unwrap());
            assert_eq!(k, FRAME_HELLO);
            let lb = s.recv_raw(4).await.unwrap();
            s.recv_raw(u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize)
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(1600)).await;
            s.send_raw(&frame_header(FRAME_HELLO_ACK, 0, 0))
                .await
                .unwrap();
            // The next request must be the first task's RESET, not a second HELLO.
            fake_expect(s, FRAME_RESET, 1).await;
            fake_ack(s, FRAME_RESET_ACK, 1).await;
        });
        assert!(e.handshake_a().is_err(), "HELLO past the bound");
        e.handshake_a().expect("the retry reads the late HELLO_ACK");
        assert!(e.handshake_done);
        e.epoch = 1;
        e.reset_exchange()
            .expect("no second HELLO_ACK in front of RESET_ACK");
        cascadia_transport::set_activation_timeout_secs(0);
        rt.block_on(fake).unwrap();
    }

    /// CAPTURE_ACK carries epoch 0, not the task's, so a late one cannot be told from a later
    /// CAPTURE's: its timeout poisons the link instead of leaving it owed.
    #[cfg(feature = "kv_coord")]
    #[test]
    fn a_late_capture_ack_poisons_the_link() {
        let _knob = crate::timeout_knob_lock();
        cascadia_transport::set_activation_timeout_secs(1);
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        let fake = rt.spawn(async move {
            let s = &mut server;
            let (k, _, _) = parse_header(&s.recv_raw(12).await.unwrap());
            assert_eq!(k, FRAME_CAPTURE);
            let lb = s.recv_raw(4).await.unwrap();
            let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
            recv_body_chunked(s, n, MAX_CARRY_BLOB_BYTES, "capture body")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(1600)).await;
            s.send_raw(&frame_header(FRAME_CAPTURE_ACK, 0, 0))
                .await
                .unwrap();
        });
        e.epoch = 1;
        assert!(e.forward_capture_downstream(7, &[1, 2, 3], "").is_err());
        e.epoch = 2;
        let err = e.reset_exchange().expect_err("poisoned").to_string();
        assert!(err.contains("desynced") && err.contains("restart"), "{err}");
        cascadia_transport::set_activation_timeout_secs(0);
        rt.block_on(fake).unwrap();
    }

    /// Draining is bounded: a peer that floods stale frames poisons the link rather than
    /// holding rank 0 in the drain.
    #[test]
    fn a_stale_reply_flood_poisons_the_link() {
        let _knob = crate::timeout_knob_lock();
        cascadia_transport::set_activation_timeout_secs(1);
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        let fake = rt.spawn(async move {
            fake_expect(&mut server, FRAME_FORWARD, 1).await;
            tokio::time::sleep(Duration::from_millis(1600)).await;
            for _ in 0..=MAX_STALE_REPLY_FRAMES {
                fake_token(&mut server, 1, 7).await;
            }
            fake_expect(&mut server, FRAME_RESET, 2).await;
            fake_ack(&mut server, FRAME_RESET_ACK, 2).await;
        });
        e.epoch = 1;
        assert!(e.send_forward_recv_token(one_row(&e), 1, 0).is_err());
        e.epoch = 2;
        let err = e
            .reset_exchange()
            .expect_err("flood must not be drained forever");
        let err = err.to_string();
        assert!(err.contains("RESET"), "{err}");
        let err = e
            .reset_exchange()
            .expect_err("poisoned after the flood")
            .to_string();
        assert!(err.contains("desynced") && err.contains("restart"), "{err}");
        cascadia_transport::set_activation_timeout_secs(0);
        rt.block_on(fake).unwrap();
    }

    #[test]
    fn a_coded_head_refusal_keeps_its_code() {
        // H23: the API maps `decision_*` reasons by prefix; a "decision head failed: " prefix
        // turned a 501 (`decision_unsupported`) into a 503.
        assert_eq!(
            head_failure_reason("decision_unsupported: no head assets".into()),
            "decision_unsupported: no head assets"
        );
        assert_eq!(
            head_failure_reason("joint infer: boom".into()),
            "decision head failed: joint infer: boom"
        );
        let mut e = bare_engine(1, "{}");
        e.active = Some(ActiveTask {
            decision: Some(cascadia_types::DecisionReadout {
                prompt_ids: vec![1, 2, 3],
                head: cascadia_types::ReadoutHead::Verbalizer { token_ids: vec![1] },
                cache_prefix: None,
                require_warm: false,
                needs_snapshot: false,
            }),
            ..finalizable_task(4, 0)
        });
        let out = e.finish_decision();
        assert_eq!(
            out[0].1.error.as_deref(),
            Some("decision_unsupported: no head assets")
        );
        assert!(e.active.is_none());
    }

    /// A RawRecv over a byte buffer that records every read size; a read past the end is short.
    struct BufRecv {
        data: std::collections::VecDeque<u8>,
        reads: Vec<usize>,
    }

    impl BufRecv {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data: data.into(),
                reads: Vec::new(),
            }
        }
    }

    impl RawRecv for BufRecv {
        fn recv_raw(
            &mut self,
            n: usize,
        ) -> impl std::future::Future<Output = TransportResult<Vec<u8>>> + Send {
            self.reads.push(n);
            let take = n.min(self.data.len());
            std::future::ready(Ok(self.data.drain(..take).collect()))
        }
    }

    #[test]
    fn chunked_body_reads_drain_an_over_ceiling_body_to_stay_framed() {
        // H25: an accepted body is read in <= MAX_RAW_BYTES chunks; an over-ceiling one is read
        // to its end and discarded, so the next frame starts where it should.
        let rt = pipe_rt();
        let max = cascadia_transport::MAX_RAW_BYTES;
        let n = 2 * max + 5;
        let mut body = vec![7u8; n];
        body.push(0xAB);
        let mut g = BufRecv::new(body.clone());
        let got = rt
            .block_on(recv_body_chunked(&mut g, n, n, "body"))
            .unwrap();
        assert_eq!(got, body[..n]);
        assert_eq!(g.reads, vec![max, max, 5]);
        let mut g = BufRecv::new(body);
        let err = rt
            .block_on(recv_body_chunked(&mut g, n, n - 1, "body"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("drained to stay framed"), "{err}");
        assert_eq!(g.reads, vec![max, max, 5]);
        assert_eq!(g.data, [0xAB], "the next frame is untouched");
        // A short read (the peer closed mid-body) is a closed socket, not a short body.
        let mut g = BufRecv::new(vec![1, 2, 3]);
        assert!(matches!(
            rt.block_on(recv_body_chunked(&mut g, 4, 4, "body")),
            Err(TransportError::SocketClosed)
        ));
    }

    #[test]
    fn decision_bodies_keep_their_length_prefix() {
        // H25: recv_decision_body returns the body still framed (relays forward it verbatim).
        let rt = pipe_rt();
        let framed = crate::decision::encode_result(&Ok(vec![0.5]));
        let mut wire = framed.clone();
        wire.push(0xAB);
        let mut g = BufRecv::new(wire);
        let got = rt
            .block_on(recv_decision_body(&mut g, "decision body"))
            .unwrap();
        assert_eq!(got, framed);
        assert_eq!(g.data, [0xAB]);
        // An over-ceiling length is drained, then refused.
        let n = crate::decision::MAX_DECISION_BODY_BYTES + 1;
        let mut wire = (n as u32).to_be_bytes().to_vec();
        wire.resize(4 + n, 0);
        wire.push(0xAB);
        let mut g = BufRecv::new(wire);
        assert!(rt
            .block_on(recv_decision_body(&mut g, "decision body"))
            .is_err());
        assert_eq!(g.data, [0xAB]);
        let mut g = BufRecv::new(vec![0, 0]);
        assert!(matches!(
            rt.block_on(recv_decision_body(&mut g, "decision body")),
            Err(TransportError::SocketClosed)
        ));
    }

    /// Task 27 M2: on a link whose HELLO reply is still owed, only the handshake retry may run;
    /// a tagged exchange is refused before it writes anything.
    #[test]
    fn a_hello_owed_link_admits_only_the_handshake_retry() {
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        e.down_link = DownLink::HelloOwed;
        e.epoch = 1;
        let err = e.reset_exchange().expect_err("owed HELLO").to_string();
        assert!(err.contains("handshake reply still owed"), "{err}");
        let fake = rt.spawn(async move {
            let s = &mut server;
            // The late HELLO_ACK; the retry sends no second HELLO, so the next request is RESET.
            s.send_raw(&frame_header(FRAME_HELLO_ACK, 0, 0))
                .await
                .unwrap();
            fake_expect(s, FRAME_RESET, 1).await;
            fake_ack(s, FRAME_RESET_ACK, 1).await;
        });
        assert_eq!(e.forward_hello_downstream(b"m").unwrap(), (None, 0));
        assert_eq!(e.down_link, DownLink::Clean);
        e.reset_exchange().expect("a clean link after the retry");
        rt.block_on(fake).unwrap();
    }

    // ---- Mock stages (unit tests without an `openvino` build) ----

    /// A scripted stage for unit tests (no `openvino` build). Output 0 is `[1, n, width]`, every
    /// row zero but `peak` (a logits argmax).
    pub(super) struct MockStage {
        inputs: Vec<String>,
        width: usize,
        peak: Option<usize>,
        n: usize,
        state: Vec<u8>,
        /// The infer (counted from 0) that fails, as the GPU plugin's did on hardware.
        fail_infer_at: Option<usize>,
        /// Task 57: a second infer that fails the same way (the retry's).
        fail_infer_again: Option<usize>,
        infers: usize,
        /// Every infer and reset, in order; and the recovery's cancel, wait and abandon (Task 46).
        events: Arc<std::sync::Mutex<Vec<&'static str>>>,
        /// Task 46: after the failing infer the request stays busy for this many `wait_for`
        /// calls (`u32::MAX`: forever), as the GPU plugin's did. Busy refuses what OpenVINO
        /// refuses.
        busy_after_fail: u32,
        busy: u32,
        cancelled: bool,
        abandoned: bool,
        /// Task 54 fix round 1: the infer (counted from 0) that never returns.
        hang_infer_at: Option<usize>,
        /// Task 54 fix round 2: counts drops of this stage (a recompile must never drop the old
        /// one: its native destructor could wait forever on the stuck request).
        drops: Arc<std::sync::atomic::AtomicUsize>,
        /// Task 55: a model of the request's state, so a stale or dropped state changes the
        /// answer. `folded` is the tokens the state holds; the peak logit reads it.
        folded: usize,
        /// The request has run an infer since it was created or reset. A `set_state_blob` onto
        /// a request that has not is dropped, as the GPU plugin drops it (see `primed`).
        ran: bool,
        /// A `set_state_blob` landed: `reset_state` then leaves the state in place, as it leaves
        /// residue on the GPU (see `state_restored`). Only `recreate_request` clears it.
        residue: bool,
        /// Task 55: `set_state_blob` refuses, as the shim does on a state-count mismatch.
        refuse_set: bool,
        /// Task 56: `abandon_request` fails, so the request can be neither idled nor abandoned.
        refuse_abandon: bool,
    }

    impl Drop for MockStage {
        fn drop(&mut self) {
            self.drops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl MockStage {
        fn new(inputs: &[&str], width: usize, peak: Option<usize>) -> Self {
            Self {
                inputs: inputs.iter().map(|s| s.to_string()).collect(),
                width,
                peak,
                n: 0,
                state: vec![3u8; 64],
                fail_infer_at: None,
                fail_infer_again: None,
                infers: 0,
                events: Default::default(),
                busy_after_fail: 0,
                busy: 0,
                cancelled: false,
                abandoned: false,
                hang_infer_at: None,
                drops: Default::default(),
                folded: 0,
                ran: false,
                residue: false,
                refuse_set: false,
                refuse_abandon: false,
            }
        }
        pub(super) fn input_names(&self) -> cascadia_ov_genai_shim::Result<Vec<String>> {
            Ok(self.inputs.clone())
        }
        /// What OpenVINO answers a call on a request that is not idle (or the shim, on none).
        fn refuse(&self) -> cascadia_ov_genai_shim::Result<()> {
            let why = if self.abandoned {
                "null runtime handle"
            } else if self.busy > 0 && self.cancelled {
                "Infer Request was canceled"
            } else if self.busy > 0 {
                "Infer Request is busy"
            } else {
                return Ok(());
            };
            Err(cascadia_ov_genai_shim::Error::Native(why.into()))
        }
        pub(super) fn request_state(&self) -> cascadia_ov_genai_shim::Result<RequestState> {
            Ok(if self.abandoned {
                RequestState::None
            } else if self.busy == 0 {
                RequestState::Idle
            } else if self.cancelled {
                RequestState::Cancelled
            } else {
                RequestState::Busy
            })
        }
        pub(super) fn cancel(&mut self) -> cascadia_ov_genai_shim::Result<()> {
            self.events.lock().unwrap().push("cancel");
            if self.busy > 0 {
                self.cancelled = true;
            }
            Ok(())
        }
        /// Like `ov::InferRequest::wait_for`: blocks the slice while the run is outstanding. Each
        /// wait is one step toward idle (`busy` counts them down).
        pub(super) fn wait_for(
            &mut self,
            timeout: Duration,
        ) -> cascadia_ov_genai_shim::Result<bool> {
            self.events.lock().unwrap().push("wait");
            self.refuse_if_abandoned()?;
            if self.busy == u32::MAX {
                std::thread::sleep(timeout);
                return Ok(false);
            }
            self.busy = self.busy.saturating_sub(1);
            if self.busy == 0 {
                self.cancelled = false;
            }
            Ok(self.busy == 0)
        }
        fn refuse_if_abandoned(&self) -> cascadia_ov_genai_shim::Result<()> {
            if self.abandoned {
                return Err(cascadia_ov_genai_shim::Error::Native(
                    "null runtime handle".into(),
                ));
            }
            Ok(())
        }
        pub(super) fn abandon_request(&mut self) -> cascadia_ov_genai_shim::Result<()> {
            self.events.lock().unwrap().push("abandon");
            if self.refuse_abandon {
                return Err(cascadia_ov_genai_shim::Error::Native(
                    "abandon refused (test)".into(),
                ));
            }
            self.abandoned = true;
            Ok(())
        }
        pub(super) fn set_input(
            &mut self,
            name: &str,
            _dtype: DType,
            shape: &[usize],
            _data: &[u8],
        ) -> cascadia_ov_genai_shim::Result<()> {
            self.refuse()?;
            if name == "input_ids" || name == "stage_hidden" || name.contains("embed") {
                self.n = shape[1];
            }
            Ok(())
        }
        pub(super) fn infer(&mut self) -> cascadia_ov_genai_shim::Result<()> {
            self.refuse()?;
            self.events.lock().unwrap().push("infer");
            self.infers += 1;
            if self.hang_infer_at == Some(self.infers - 1) {
                hang_forever();
            }
            let at = Some(self.infers - 1);
            if self.fail_infer_at == at || self.fail_infer_again == at {
                self.busy = self.busy_after_fail;
                return Err(cascadia_ov_genai_shim::Error::Native(
                    "unknown C++ exception in infer".into(),
                ));
            }
            self.folded += self.n;
            self.ran = true;
            Ok(())
        }
        pub(super) fn output(
            &self,
            _idx: usize,
        ) -> cascadia_ov_genai_shim::Result<(DType, Vec<usize>, Vec<u8>)> {
            let mut row = vec![0f32; self.width];
            if let Some(p) = self.peak {
                // Still the argmax; its value says how many tokens the state holds (Task 55).
                row[p] = 1.0 + 0.01 * self.folded as f32;
            }
            let data: Vec<f32> = (0..self.n).flat_map(|_| row.clone()).collect();
            Ok((DType::F32, vec![1, self.n, self.width], le_bytes_f32(&data)))
        }
        pub(super) fn reset_state(&mut self) -> cascadia_ov_genai_shim::Result<()> {
            self.refuse()?;
            self.events.lock().unwrap().push("reset");
            if !self.residue {
                self.folded = 0;
            }
            self.ran = false;
            Ok(())
        }
        /// The shim refuses to release a request that is not idle (it would block forever).
        pub(super) fn recreate_request(&mut self) -> cascadia_ov_genai_shim::Result<()> {
            self.refuse()?;
            self.events.lock().unwrap().push("reset");
            self.folded = 0;
            self.ran = false;
            self.residue = false;
            Ok(())
        }
        /// The state bytes, with `folded` in the last 8 (a blob `kv_seq_from_blob` cannot parse,
        /// so the restore depth is the matched length).
        pub(super) fn get_state_blob(&mut self) -> cascadia_ov_genai_shim::Result<Vec<u8>> {
            let mut b = self.state.clone();
            let at = b.len() - 8;
            b[at..].copy_from_slice(&(self.folded as u64).to_le_bytes());
            Ok(b)
        }
        pub(super) fn set_state_blob(&mut self, blob: &[u8]) -> cascadia_ov_genai_shim::Result<()> {
            self.events.lock().unwrap().push("set_state");
            if self.refuse_set {
                return Err(cascadia_ov_genai_shim::Error::Native(
                    "set_state_blob: blob has 7 states, model has 8".into(),
                ));
            }
            self.residue = true;
            if self.ran {
                self.state = blob.to_vec();
                let at = blob.len() - 8;
                self.folded = u64::from_le_bytes(blob[at..].try_into().unwrap()) as usize;
            }
            Ok(())
        }
    }

    /// A native call that never returns (its thread is leaked; the test process exits anyway).
    fn hang_forever() -> ! {
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }

    const MOCK_HIDDEN: usize = 4;
    const MOCK_VOCAB: usize = 8;

    /// The last stage's inputs.
    const LAST_STAGE_INPUTS: [&str; 5] = [
        "stage_hidden",
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "beam_idx",
    ];

    /// The first stage's inputs.
    const FIRST_STAGE_INPUTS: [&str; 4] = [
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "beam_idx",
    ];

    /// A single box over two mock IRs.
    fn mock_box() -> Qwen36Engine {
        let mut e = bare_engine(1, "{}");
        e.rank = 0;
        e.hidden = MOCK_HIDDEN;
        e.tokenizer = Some(tiny_tokenizer());
        e.vocab = Some(MOCK_VOCAB);
        e.decision_assets = Some(crate::decision::HeadAssets::Verbalizer);
        let emb = MockStage::new(&["input_ids"], MOCK_HIDDEN, None);
        let s0 = MockStage::new(&FIRST_STAGE_INPUTS, MOCK_HIDDEN, None);
        let s1 = MockStage::new(&LAST_STAGE_INPUTS, MOCK_VOCAB, Some(1));
        e.emb = Some(OvStage::Mock(emb));
        e.stages = vec![OvStage::Mock(s0), OvStage::Mock(s1)];
        arm_recompile(&mut e);
        e
    }

    /// Task 54 fix round 2: what the mock recompile does next.
    #[derive(Default)]
    struct RecompileCtl {
        /// The compile fails.
        fail: bool,
        /// The compile never returns.
        hang: bool,
        /// The recompiled stage's first infer fails.
        fail_first_infer: bool,
        /// The recompiled stage's first infer never returns.
        hang_first_infer: bool,
        /// Compiles run so far, by recipe (xml).
        built: Vec<String>,
        /// The drop counter of each fresh copy built, in order.
        fresh_drops: Vec<Arc<std::sync::atomic::AtomicUsize>>,
    }

    /// Task 54 fix round 2: give every mock slot of `e` a recipe, and a factory that "compiles" a
    /// fresh mock of the same shape that logs into the same event list (with a `recompile` event)
    /// and has its own drop counter. Returns the control for the next compiles.
    fn arm_recompile(e: &mut Qwen36Engine) -> Arc<std::sync::Mutex<RecompileCtl>> {
        arm_recompile_sized(e, 1000)
    }

    /// [`arm_recompile`] with every slot's weight file `weight_bytes` long (Task 58: the cap
    /// follows the largest).
    fn arm_recompile_sized(
        e: &mut Qwen36Engine,
        weight_bytes: u64,
    ) -> Arc<std::sync::Mutex<RecompileCtl>> {
        type Template = (
            Vec<String>,
            usize,
            Option<usize>,
            Arc<std::sync::Mutex<Vec<&'static str>>>,
        );
        let mut templates: Vec<(String, Template)> = Vec::new();
        let recipe = |xml: &str| StageRecipe {
            xml: xml.to_string(),
            device: "MOCK".to_string(),
            plugin: PluginConfig::new(),
            weight_bytes,
        };
        let mut template = |xml: &str, st: &OvStage| {
            if let OvStage::Mock(m) = st {
                templates.push((
                    xml.to_string(),
                    (m.inputs.clone(), m.width, m.peak, Arc::clone(&m.events)),
                ));
            }
            recipe(xml)
        };
        let emb = e.emb.as_ref().map(|st| template("emb", st));
        let stages: Vec<StageRecipe> = e
            .stages
            .iter()
            .enumerate()
            .map(|(j, st)| template(&format!("stage{j}"), st))
            .collect();
        let ctl = Arc::new(std::sync::Mutex::new(RecompileCtl::default()));
        let c = Arc::clone(&ctl);
        let cap = recompile_cap(largest_leak_bytes(emb.as_ref(), &stages));
        e.rebuild = Rebuild::new(emb, stages, cap);
        e.rebuild.factory = Arc::new(move |r: &StageRecipe| {
            let (_, (inputs, width, peak, events)) = templates
                .iter()
                .find(|(xml, _)| *xml == r.xml)
                .expect("a template for the recipe");
            events.lock().unwrap().push("recompile");
            let (fail, hang, fail_first, hang_first) = {
                let mut c = c.lock().unwrap();
                c.built.push(r.xml.clone());
                (c.fail, c.hang, c.fail_first_infer, c.hang_first_infer)
            };
            if hang {
                hang_forever();
            }
            if fail {
                return Err(cascadia_ov_genai_shim::Error::Native(
                    "compile_model failed".into(),
                ));
            }
            let inputs: Vec<&str> = inputs.iter().map(String::as_str).collect();
            let mut m = MockStage::new(&inputs, *width, *peak);
            m.events = Arc::clone(events);
            if fail_first {
                m.fail_infer_at = Some(0);
            }
            if hang_first {
                m.hang_infer_at = Some(0);
            }
            c.lock().unwrap().fresh_drops.push(Arc::clone(&m.drops));
            Ok(OvStage::Mock(m))
        });
        ctl
    }

    /// Step until the task's final chunk.
    fn run_to_final(e: &mut Qwen36Engine) -> Chunk {
        for _ in 0..50 {
            for (_, c) in e.step().expect("step") {
                if c.is_final {
                    return c;
                }
            }
        }
        panic!("no final chunk");
    }

    fn chat(id: &str) -> GenerationTask {
        let mut t = windowed_task(id, "hi hi hi");
        t.max_tokens = 2;
        t
    }

    fn verbalizer_pass(id: &str) -> GenerationTask {
        let mut t = decision_task(
            cascadia_types::ReadoutHead::Verbalizer {
                token_ids: vec![1, 2],
            },
            None,
        );
        t.task_id = id.into();
        t.decision.as_mut().unwrap().prompt_ids = vec![0, 1, 2, 3];
        t
    }

    /// Infers run on one mock stage so far.
    fn infers(e: &Qwen36Engine, stage: usize) -> usize {
        match &e.stages[stage] {
            OvStage::Mock(m) => m.infers,
            _ => unreachable!(),
        }
    }

    // ---- Task 53: switchable-LoRA trees are refused ----

    #[test]
    fn the_manifest_says_which_trees_were_exported_switchable() {
        let sw = Manifest::parse(
            r#"{"arch":"qwen3_5","hidden_size":4096,"stages":[
                {"stage":0,"layer_start":0,"layer_end":3,"inputs":["inputs_embeds","beam_idx"]},
                {"stage":1,"layer_start":4,"layer_end":7,
                 "inputs":["stage_hidden","inputs_embeds","lora_scale"]}]}"#,
        )
        .unwrap();
        assert!(sw.lora_switchable());
        assert!(!sw.stages[0].lora_input() && sw.stages[1].lora_input());
        for baked in [
            r#"{"arch":"qwen3_5","hidden_size":4096,"stages":[
                {"stage":0,"layer_start":0,"layer_end":7,"inputs":["inputs_embeds"]}]}"#,
            // a manifest without stage inputs (hand-written): the IR check at compile decides
            r#"{"arch":"qwen3_5","hidden_size":4096,"stages":[
                {"stage":0,"layer_start":0,"layer_end":7}]}"#,
        ] {
            assert!(!Manifest::parse(baked).unwrap().lora_switchable());
        }
    }

    #[test]
    fn a_switchable_tree_is_refused_at_load() {
        // Trees exported with the removed `--switchable-lora` (e.g. trees/jev9b-int8-sw) have a
        // `lora_scale` stage input that nothing feeds any more: refused before any compile, on
        // the single box and on every pipeline rank (the manifest lists every stage's inputs).
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("manifest.json"),
            r#"{"arch":"qwen3_5","hidden_size":4096,"stages":[
                {"stage":0,"layer_start":0,"layer_end":3,"inputs":["inputs_embeds","beam_idx"]},
                {"stage":1,"layer_start":4,"layer_end":7,
                 "inputs":["stage_hidden","inputs_embeds","lora_scale"]}]}"#,
        )
        .unwrap();
        std::fs::write(
            d.path().join("decision.json"),
            r#"{"family":"jev","jev":{"lora_scale":2.0}}"#,
        )
        .unwrap();
        let spec = ShardSpec {
            model_id: "jev".into(),
            layer_start: 0,
            layer_end: 7,
            total_layers: 8,
            device: "CPU".into(),
            is_first_stage: true,
            is_last_stage: true,
            tp_size: 1,
            tp_rank: 0,
        };
        let mut b = Qwen36Builder::new(d.path().to_str().unwrap(), "CPU");
        let rt = pipe_rt();
        match rt.block_on(b.load(spec)) {
            Err(EngineError::InvalidConfig(why)) => assert_eq!(
                why,
                "switchable trees are no longer supported; re-export without --switchable-lora"
            ),
            Err(other) => panic!("wrong refusal: {other}"),
            Ok(_) => panic!("a switchable tree loaded"),
        }
    }

    // ---- Task 53: the pipeline wire is the pre-Task-41 layout ----

    #[test]
    fn own_caps_are_the_pre_task_41_bits() {
        // Bits 0 (capture_v2) and 1 (decision frames) only. A Task 42 build also set bit 2; the
        // AND in combine_caps drops it, so a chain with one reads as this layout.
        assert_eq!(own_caps(true) & !(CAP_CAPTURE_V2 | CAP_DECISION), 0);
        assert_eq!(own_caps(false) & !CAP_CAPTURE_V2, 0);
        assert!(chain_decision(own_caps(true)));
        let task42_peer = CAP_CAPTURE_V2 | CAP_DECISION | 4;
        assert_eq!(combine_caps(own_caps(true), task42_peer), own_caps(true));
        for old in [0, 1, CAP_CAPTURE_V2 | CAP_DECISION] {
            assert_eq!(combine_caps(own_caps(true), old), own_caps(true) & old);
        }
    }

    #[test]
    fn hello_carries_no_lora_scale_key_and_accepts_a_peer_that_does() {
        let e = bare_engine(2, "{}");
        let ours: serde_json::Value = serde_json::from_slice(&e.hello_payload()).unwrap();
        assert!(ours.get("lora_scale").is_none(), "{ours}");
        // A Task 42 build sends `"lora_scale": 1`; on a baked tree that is all that differs.
        let mut task42 = ours.clone();
        task42["lora_scale"] = serde_json::json!(1);
        assert_eq!(e.validate_hello(task42.to_string().as_bytes()), None);
        assert_eq!(e.validate_hello(&e.hello_payload()), None);
    }

    /// Fake rank 1: answer one HELLO with `cap`.
    async fn fake_hello(s: &mut ActivationServer, cap: u32) {
        let (k, _, _) = parse_header(&s.recv_raw(12).await.unwrap());
        assert_eq!(k, FRAME_HELLO);
        let lb = s.recv_raw(4).await.unwrap();
        s.recv_raw(u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize)
            .await
            .unwrap();
        s.send_raw(&frame_header(FRAME_HELLO_ACK, 0, cap))
            .await
            .unwrap();
    }

    #[test]
    fn rank0_accepts_old_and_task_42_downstream_ranks_at_hello() {
        let rt = pipe_rt();
        for cap in [
            CAP_CAPTURE_V2 | CAP_DECISION,
            CAP_CAPTURE_V2 | CAP_DECISION | 4,
        ] {
            let (mut e, mut server) = pipe_rank0(&rt);
            e.handshake_done = false;
            let fake = rt.spawn(async move { fake_hello(&mut server, cap).await });
            e.handshake_a().expect("accepted");
            assert!(e.handshake_done);
            assert_eq!(e.chain_caps & 4, 0, "no bit beyond the pre-41 ones: {cap}");
            rt.block_on(fake).unwrap();
        }
    }

    #[test]
    fn rank0_sends_reset_with_pos_zero() {
        // RESET's `pos` is 0, as before Task 41 (Task 42 put the LoRA scale's bits there).
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        e.hidden = MOCK_HIDDEN;
        e.tokenizer = Some(tiny_tokenizer());
        let emb = MockStage::new(&["input_ids"], MOCK_HIDDEN, None);
        let s0 = MockStage::new(&FIRST_STAGE_INPUTS, MOCK_HIDDEN, None);
        e.emb = Some(OvStage::Mock(emb));
        e.stages = vec![OvStage::Mock(s0)];
        let fake = rt.spawn(async move {
            let s = &mut server;
            let mut seen = Vec::new();
            for epoch in [1, 2] {
                let (k, ep, pos) = parse_header(&s.recv_raw(12).await.unwrap());
                assert_eq!((k, ep), (FRAME_RESET, epoch));
                seen.push(pos);
                fake_ack(s, FRAME_RESET_ACK, epoch).await;
                fake_expect(s, FRAME_FORWARD, epoch).await;
                fake_token(s, epoch, 1).await;
            }
            seen
        });
        for id in ["t1", "t2"] {
            e.pending.push(chat(id));
            assert!(e.step_pipe_first().is_empty(), "prefill done, no error");
            e.cancel(&id.to_string());
        }
        assert_eq!(rt.block_on(fake).unwrap(), vec![0, 0]);
        assert_eq!(infers(&e, 0), 2);
    }

    #[test]
    fn a_failed_decode_returns_its_error_before_the_scrub() {
        // Task 45: on the GPU, a decode infer threw and the request never heard of it; the warn
        // was logged, and the scrub that ran before the error chunk was returned did not return.
        // The error chunk now comes first; the next admission scrubs before it touches a stage.
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        let events = match &mut e.stages[1] {
            OvStage::Mock(m) => {
                m.fail_infer_at = Some(1); // the prefill is infer 0, the first decode infer 1
                Arc::clone(&m.events)
            }
            _ => unreachable!(),
        };
        let since = |n: usize| events.lock().unwrap()[n..].to_vec();
        e.submit(chat("c1")).unwrap();
        let out = e.step().unwrap();
        assert!(out.is_empty(), "admission and prefill: {out:?}");
        let before = events.lock().unwrap().len();
        let out = e.step().unwrap();
        assert_eq!(out.len(), 1);
        let (id, c) = &out[0];
        assert_eq!(id, "c1");
        assert!(c.is_final, "{c:?}");
        assert_eq!(
            c.error.as_deref(),
            Some("decode failed: backend error: unknown C++ exception in infer"),
            "the error Task 44 saw"
        );
        assert_eq!(c.warm_prefix, Some(0), "a cold prefill: a cache miss");
        assert!(e.active.is_none(), "the failed task is gone");
        assert_eq!(
            since(before),
            ["infer"],
            "no reset before the error is returned"
        );

        // The next task (chat too) is served on scrubbed requests.
        e.submit(chat("c2")).unwrap();
        let c = run_to_final(&mut e);
        assert!(c.error.is_none(), "{c:?}");
        let after = since(before + 1);
        assert_eq!(after.first(), Some(&"reset"), "scrubbed first: {after:?}");
        // Scrubbed once: a second task's admission has nothing left to scrub.
        assert!(!e.scrub_pending);
    }

    /// Task 46: stage 1 of a mock box fails infer `at` and then reads busy for `busy` state reads.
    fn busy_after_failed_infer(
        e: &mut Qwen36Engine,
        at: usize,
        busy: u32,
    ) -> Arc<std::sync::Mutex<Vec<&'static str>>> {
        let OvStage::Mock(m) = &mut e.stages[1] else {
            unreachable!()
        };
        m.fail_infer_at = Some(at);
        m.busy_after_fail = busy;
        Arc::clone(&m.events)
    }

    #[test]
    fn a_request_a_failed_infer_left_busy_is_made_idle_before_the_scrub() {
        // Task 46, pawan-03: after `unknown C++ exception in infer` the GPU request stayed busy;
        // the scrub's `reset_state` was refused ("Infer Request is busy") and the recreate that
        // followed released the busy request and never returned. The scrub now cancels and waits
        // the request back to idle (bounded) before it resets anything. A decision pass fails
        // here; the chat task after it gets the recovery.
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        let events = busy_after_failed_infer(&mut e, 0, 3); // the pass's prefill infer
        e.submit(verbalizer_pass("d1")).unwrap();
        let c = run_to_final(&mut e);
        let why = c.error.as_deref().expect("the pass fails");
        assert!(why.contains("unknown C++ exception in infer"), "{why}");
        assert!(
            e.infer_failed,
            "the failed infer is remembered for the scrub"
        );
        let before = events.lock().unwrap().len();

        let t0 = Instant::now();
        e.submit(chat("c2")).unwrap();
        let c = run_to_final(&mut e);
        assert!(c.error.is_none(), "served after the recovery: {c:?}");
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        let after = events.lock().unwrap()[before..].to_vec();
        let reset = after.iter().position(|&a| a == "reset").expect("scrubbed");
        assert_eq!(
            after[..reset],
            ["cancel", "wait", "wait", "wait"],
            "cancel, then wait until idle, and only then reset: {after:?}"
        );
        assert!(!after.contains(&"abandon"), "{after:?}");
        assert!(!e.infer_failed && e.engine_failed.is_none() && !e.stages_dirty);
    }

    /// Task 54: arm stage 1 of a mock box so its next infer fails and leaves the request
    /// `Cancelled` for good once cancelled (`busy` forever), as the GPU request did on pawan-01.
    fn stuck_on_next_infer(e: &mut Qwen36Engine) -> Arc<std::sync::Mutex<Vec<&'static str>>> {
        let at = infers(e, 1);
        busy_after_failed_infer(e, at, u32::MAX)
    }

    fn mock1(e: &mut Qwen36Engine) -> &mut MockStage {
        let OvStage::Mock(m) = &mut e.stages[1] else {
            unreachable!()
        };
        m
    }

    fn drops_of(m: &MockStage) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&m.drops)
    }

    fn dropped(d: &Arc<std::sync::atomic::AtomicUsize>) -> usize {
        d.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[test]
    fn a_stage_whose_request_never_goes_idle_is_recompiled_and_chat_and_decisions_keep_serving() {
        // Task 54 fix round 2, pawan-01: after `unknown C++ exception in infer` stage6 stayed
        // `Cancelled` for the whole wait and was abandoned. A fresh request from the same compiled
        // model then failed its first infer (the fault corrupts per-CompiledModel state), so the
        // stage is now recompiled from its IR, the old copy leaked (never dropped), and the next
        // task is served. Both routes: a failed chat and a failed decision pass, each followed by
        // the other kind and by its own.
        for (fail, next) in [
            (chat("c1"), chat("c2")),
            (verbalizer_pass("d1"), verbalizer_pass("d2")),
            (chat("c3"), verbalizer_pass("d3")),
            (verbalizer_pass("d4"), chat("c4")),
        ] {
            let what = format!("{} then {}", fail.task_id, next.task_id);
            let mut e = mock_box();
            e.no_retry = true; // Task 57 off: the task fails, the next one recovers
            e.stage_idle_timeout = Duration::from_millis(200);
            let events = stuck_on_next_infer(&mut e);
            let old_drops = drops_of(mock1(&mut e));
            e.submit(fail).unwrap();
            let c = run_to_final(&mut e);
            let why = c.error.as_deref().expect("the first task fails");
            assert!(
                why.contains("unknown C++ exception in infer"),
                "{what}: {why}"
            );
            let before = events.lock().unwrap().len();

            let t0 = Instant::now();
            e.submit(next).unwrap();
            let c = run_to_final(&mut e);
            let took = t0.elapsed();
            assert!(
                c.error.is_none(),
                "{what}: served on the recompiled stage: {c:?}"
            );
            assert!(
                took >= Duration::from_millis(200) && took < Duration::from_secs(3),
                "{what}: one bounded wait: {took:?}"
            );
            let after = events.lock().unwrap()[before..].to_vec();
            let abandon = after
                .iter()
                .position(|&a| a == "abandon")
                .expect("abandoned");
            assert_eq!(after[0], "cancel", "{what}: {after:?}");
            assert!(
                after[1..abandon].iter().all(|&a| a == "wait"),
                "{what}: {after:?}"
            );
            // Review N1: no reset or recreate on the fresh copy; its first call is the infer.
            assert_eq!(
                &after[abandon..abandon + 3],
                ["abandon", "recompile", "infer"],
                "{what}: recompiled, then served: {after:?}"
            );
            assert!(e.engine_failed.is_none() && !e.stages_dirty, "{what}");
            assert_eq!(
                e.failure().reason(),
                None,
                "{what}: a recovery is not a failure"
            );
            assert_eq!(e.rebuild.recompiled, 1, "{what}");
            assert!(
                e.fresh_requests.is_empty(),
                "{what}: the recompiled stage has proven itself"
            );
            assert!(
                !Arc::ptr_eq(&old_drops, &mock1(&mut e).drops),
                "{what}: a new stage is installed"
            );
            assert_eq!(dropped(&old_drops), 0, "{what}: the old stage is leaked");

            // And the task after that, with no recovery at all.
            let n = events.lock().unwrap().len();
            e.submit(chat("again")).unwrap();
            assert!(run_to_final(&mut e).error.is_none(), "{what}");
            let tail = events.lock().unwrap()[n..].to_vec();
            assert!(
                !tail.iter().any(|&a| a == "cancel" || a == "recompile"),
                "{what}: {tail:?}"
            );
            drop(e);
            assert_eq!(
                dropped(&old_drops),
                0,
                "{what}: never dropped, even with the engine"
            );
        }
    }

    #[test]
    fn the_degraded_warning_fires_once_at_the_second_recompile_and_the_metrics_count() {
        // Task 54 fix round 3: one leaked copy is harmless, two slow decode ~16%. The warn comes
        // at the second successful recompile, never at the first, never again. The metrics are
        // process-global and other recompile tests run in parallel, so they are asserted as
        // lower-bound deltas; the engine's own totals are exact.
        use cascadia_metrics::{ENGINE_LEAKED_STAGE_BYTES, ENGINE_STAGE_RECOMPILES_TOTAL};
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        e.stage_idle_timeout = Duration::from_millis(100);
        arm_recompile(&mut e);
        let (c0, b0) = (
            ENGINE_STAGE_RECOMPILES_TOTAL.get(),
            ENGINE_LEAKED_STAGE_BYTES.get(),
        );
        for round in 1..=2usize {
            stuck_on_next_infer(&mut e);
            e.submit(chat("fails")).unwrap();
            assert!(run_to_final(&mut e).error.is_some(), "round {round}");
            e.submit(verbalizer_pass("served")).unwrap();
            assert!(run_to_final(&mut e).error.is_none(), "round {round}");
            assert_eq!(e.rebuild.recompiled, round);
            assert_eq!(e.rebuild.recompiles_ok, round);
            assert_eq!(e.rebuild.leaked_copies, round);
            assert_eq!(e.rebuild.leaked_bytes, 1000 * round as u64);
            assert!(ENGINE_STAGE_RECOMPILES_TOTAL.get() - c0 >= round as u64);
            assert!(ENGINE_LEAKED_STAGE_BYTES.get() - b0 >= 1000 * round as i64);
            assert_eq!(e.rebuild.degraded_warned, round == 2, "round {round}");
        }
        // A later task does not warn again (the flag gates the one `warn!`).
        e.submit(chat("again")).unwrap();
        assert!(run_to_final(&mut e).error.is_none());
        assert!(e.rebuild.degraded_warned);
        assert_eq!(e.rebuild.recompiles_ok, 2);
    }

    #[test]
    fn the_degraded_message_is_the_specified_text_and_names_a_leaked_thread() {
        let base = "qwen36: engine degraded: 2 stage copies leaked by fault recovery (2000 \
                    bytes); decode is slower until the server restarts (measured +16% with two \
                    on an Arc 140V); restart recommended";
        assert_eq!(degraded_message(2, 2000, 0), base);
        assert_eq!(
            degraded_message(2, 2000, 1),
            format!("{base}; 1 recompile helper thread(s) also leaked after a timeout")
        );
    }

    #[test]
    fn the_third_abandon_fails_the_engine_past_the_cap_of_two_recompiles() {
        // Task 54 fix round 2: each recompile leaks a whole stage (about 1.5-2.5 GB on the 9B
        // tree), so an engine recompiles at most twice; the third abandon fails it.
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        assert_eq!((e.rebuild.cap, MAX_RECOMPILES), (2, 2));
        e.stage_idle_timeout = Duration::from_millis(100);
        let ctl = arm_recompile(&mut e);
        let watched = e.failure();
        for round in 1..=2 {
            stuck_on_next_infer(&mut e);
            e.submit(chat("fails")).unwrap();
            assert!(run_to_final(&mut e).error.is_some(), "round {round}");
            e.submit(verbalizer_pass("served")).unwrap();
            let c = run_to_final(&mut e);
            assert!(c.error.is_none(), "round {round}: {c:?}");
            assert_eq!(e.rebuild.recompiled, round);
            assert_eq!(watched.reason(), None, "round {round}: recovered");
        }
        let events = stuck_on_next_infer(&mut e);
        let old_drops = drops_of(mock1(&mut e));
        e.submit(verbalizer_pass("fails")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        let before = events.lock().unwrap().len();
        e.submit(chat("refused")).unwrap();
        let c = run_to_final(&mut e);
        let why = c.error.as_deref().expect("refused");
        assert!(
            why.starts_with(
                "backend error: qwen36 engine failed: stage1's infer request was still"
            ) && why.contains("this engine has already recompiled 2 stages, the cap of 2")
                && why.ends_with("restart the server"),
            "{why}"
        );
        let after = events.lock().unwrap()[before..].to_vec();
        assert_eq!(after.last(), Some(&"abandon"), "not recompiled: {after:?}");
        assert_eq!(ctl.lock().unwrap().built, ["stage1", "stage1"]);
        assert_eq!(e.rebuild.recompiled, 2);
        assert!(e.engine_failed.is_some());
        // Task 56: the handle grabbed before the failure carries the same reason.
        assert_eq!(watched.reason(), e.engine_failed);

        // Every later task, either route: refused at once, nothing touches a stage.
        let n = events.lock().unwrap().len();
        for task in [verbalizer_pass("d"), chat("c")] {
            let t0 = Instant::now();
            e.submit(task).unwrap();
            let c = run_to_final(&mut e);
            assert!(t0.elapsed() < Duration::from_millis(100));
            assert_eq!(c.error.as_deref(), Some(why));
        }
        assert_eq!(
            events.lock().unwrap().len(),
            n,
            "no stage call after the failure"
        );
        assert_eq!(dropped(&old_drops), 0);
        // Sweep item 2: the cap-refused copy is leaked and counted, not left to drop with the
        // engine. Two recompiled-away copies plus this one.
        assert!(matches!(e.stages[1], OvStage::Gone("recovery")));
        assert_eq!((e.rebuild.leaked_copies, e.rebuild.leaked_bytes), (3, 3000));
        assert_eq!(e.rebuild.recompiles_ok, 2);
        drop(e);
        assert_eq!(
            dropped(&old_drops),
            0,
            "the cap-refused copy is never dropped, even with the engine"
        );
    }

    /// Task 58: measured stage weight files (bytes).
    const STAGE6_9B: u64 = 1_490_000_000;
    const STAGE7_9B: u64 = 2_470_000_000;
    const WHOLE_9B: u64 = 8_966_000_000; // 7.398 + 0.948 GiB
    const STAGE_27B: u64 = 13_000_000_000;

    fn recipe_of(weight_bytes: u64) -> StageRecipe {
        StageRecipe {
            xml: String::new(),
            device: "MOCK".to_string(),
            plugin: PluginConfig::new(),
            weight_bytes,
        }
    }

    #[test]
    fn small_stages_keep_the_cap_of_two() {
        // Task 58: the 8-stage 9B tree's stages leak at most 2.47 GB each.
        let stages: Vec<_> = [STAGE6_9B, STAGE7_9B, STAGE6_9B]
            .into_iter()
            .map(recipe_of)
            .collect();
        let leak = largest_leak_bytes(Some(&recipe_of(1_018_000_000)), &stages);
        assert_eq!(leak, STAGE7_9B);
        assert_eq!(recompile_cap(leak), MAX_RECOMPILES);
        assert_eq!(
            recompile_cap(LARGE_LEAK_BYTES),
            2,
            "the threshold itself keeps 2"
        );
        assert_eq!(recompile_cap(0), 2, "unknown sizes keep 2");
        assert_eq!(mock_box().rebuild.cap, 2);
    }

    #[test]
    fn one_big_local_stage_gets_the_cap_of_one() {
        // Task 58: a 1-stage 9B tree's one stage is the whole model.
        assert_eq!(LARGE_LEAK_MAX_RECOMPILES, 1);
        assert_eq!(largest_leak_bytes(None, &[recipe_of(WHOLE_9B)]), WHOLE_9B);
        assert_eq!(recompile_cap(WHOLE_9B), 1);
        assert_eq!(mock_one_stage_box().rebuild.cap, 1);
    }

    #[test]
    fn a_pipeline_rank_with_a_big_stage_gets_the_cap_of_one() {
        // Task 58: a 27B 2-stage pipeline rank leaks 12-15 GB per recompile; two of those on a
        // 32 GB box that shares RAM with the GPU are not safe either.
        assert_eq!(recompile_cap(STAGE_27B), 1);
        for (bytes, cap) in [(STAGE_27B, 1), (STAGE7_9B, 2)] {
            let mut relay = bare_engine(2, "{}");
            relay.stages = vec![OvStage::Mock(MockStage::new(
                &LAST_STAGE_INPUTS,
                MOCK_VOCAB,
                Some(1),
            ))];
            arm_recompile_sized(&mut relay, bytes);
            assert_eq!(relay.rebuild.cap, cap, "{bytes}");
        }
    }

    /// Task 58: a single box over one mock IR (a 1-stage tree), recompile armed.
    fn mock_one_stage_box() -> Qwen36Engine {
        let mut e = bare_engine(1, "{}");
        e.rank = 0;
        e.hidden = MOCK_HIDDEN;
        e.tokenizer = Some(tiny_tokenizer());
        e.vocab = Some(MOCK_VOCAB);
        e.decision_assets = Some(crate::decision::HeadAssets::Verbalizer);
        let emb = MockStage::new(&["input_ids"], MOCK_HIDDEN, None);
        let s0 = MockStage::new(&FIRST_STAGE_INPUTS, MOCK_VOCAB, Some(1));
        e.emb = Some(OvStage::Mock(emb));
        e.stages = vec![OvStage::Mock(s0)];
        arm_recompile_sized(&mut e, WHOLE_9B);
        e
    }

    /// Stage 0's next infer fails and its request never goes idle (abandoned, then recompiled).
    fn stage0_stuck_on_next_infer(
        e: &mut Qwen36Engine,
    ) -> Arc<std::sync::Mutex<Vec<&'static str>>> {
        let OvStage::Mock(m) = &mut e.stages[0] else {
            unreachable!()
        };
        m.fail_infer_at = Some(m.infers);
        m.busy_after_fail = u32::MAX;
        Arc::clone(&m.events)
    }

    #[test]
    fn a_box_with_one_big_stage_recovers_once_and_its_second_abandon_fails_the_engine() {
        // Task 58: the first fault recovers (one recompile) and Task 57 retries the request, which
        // is served; the second abandon is past the cap of one, so the engine fails (Task 56: the
        // server exits for a supervisor restart) instead of leaking a second whole-model copy.
        let mut e = mock_one_stage_box();
        e.stage_idle_timeout = Duration::from_millis(100);
        let ctl = arm_recompile_sized(&mut e, WHOLE_9B);
        assert_eq!(e.rebuild.cap, 1);
        let watched = e.failure();

        let mut clean = mock_one_stage_box();
        clean.submit(chat("c1")).unwrap();
        let (want, _) = drive(&mut clean);
        assert!(want.iter().all(|c| c.error.is_none()), "{want:?}");

        stage0_stuck_on_next_infer(&mut e);
        e.submit(chat("c1")).unwrap();
        let (got, _) = drive(&mut e);
        assert!(
            got.iter().all(|c| c.error.is_none()),
            "retried and served: {got:?}"
        );
        assert_eq!(text(&got), text(&want));
        assert_eq!(e.rebuild.recompiled, 1);
        assert_eq!(e.rebuild.recompiles_ok, 1);
        assert!(
            !e.rebuild.degraded_warned,
            "one recompile is below DEGRADED_AT_RECOMPILES"
        );
        assert_eq!(watched.reason(), None, "recovered");

        let events = stage0_stuck_on_next_infer(&mut e);
        e.submit(chat("c2")).unwrap();
        let (got, _) = drive(&mut e);
        let why = got.last().and_then(|c| c.error.as_deref()).expect("fails");
        assert!(
            why.contains("this engine has already recompiled 1 stages, the cap of 1")
                && why.ends_with("restart the server"),
            "{why}"
        );
        assert_eq!(ctl.lock().unwrap().built, ["stage0"], "no second recompile");
        assert_eq!(events.lock().unwrap().last(), Some(&"abandon"));
        assert_eq!(e.rebuild.recompiled, 1);
        assert!(e.engine_failed.is_some());
        assert_eq!(watched.reason(), e.engine_failed);
        assert!(matches!(e.stages[0], OvStage::Gone("recovery")));
        assert_eq!(
            (e.rebuild.leaked_copies, e.rebuild.leaked_bytes),
            (2, 2 * WHOLE_9B)
        );
    }

    #[test]
    fn a_request_that_cannot_be_abandoned_fails_the_engine_and_reports_it() {
        // Task 56: a stuck request whose abandon fails is not recompiled; the engine fails, and
        // rank 0 reports it on the handle the server watches, with the same reason.
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        e.stage_idle_timeout = Duration::from_millis(100);
        let watched = e.failure();
        stuck_on_next_infer(&mut e);
        mock1(&mut e).refuse_abandon = true;
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        assert_eq!(
            watched.reason(),
            None,
            "a failed task alone is not a failed engine"
        );
        e.submit(chat("c2")).unwrap();
        let c = run_to_final(&mut e);
        let why = c.error.as_deref().expect("refused");
        assert!(
            why.contains("abandon failed: ") && why.contains("abandon refused (test)"),
            "{why}"
        );
        let reason = e.engine_failed.clone().expect("the engine is failed");
        assert!(reason.contains("abandon failed"), "{reason}");
        assert_eq!(watched.reason(), Some(reason));
        assert_eq!(e.rebuild.recompiled, 0);
    }

    #[test]
    fn an_abandoned_stage_with_no_recipe_is_leaked_and_never_dropped() {
        // Sweep item 2: no recipe to recompile from fails the engine; the abandoned copy is
        // leaked, not left in its slot to drop with the engine.
        let mut e = mock_box();
        e.stage_idle_timeout = Duration::from_millis(100);
        e.rebuild.stages.clear();
        stuck_on_next_infer(&mut e);
        let old_drops = drops_of(mock1(&mut e));
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        e.submit(chat("c2")).unwrap();
        let c = run_to_final(&mut e);
        let why = c.error.as_deref().expect("refused");
        assert!(
            why.contains("there is no recipe to recompile stage1 from"),
            "{why}"
        );
        assert!(e.engine_failed.is_some());
        assert_eq!(e.rebuild.recompiled, 0);
        assert!(matches!(e.stages[1], OvStage::Gone("recovery")));
        // Counted, with 0 bytes: there is no recipe to size it from.
        assert_eq!((e.rebuild.leaked_copies, e.rebuild.leaked_bytes), (1, 0));
        drop(e);
        assert_eq!(dropped(&old_drops), 0);
    }

    #[test]
    fn a_failed_recompile_fails_the_engine() {
        // Task 54 fix round 2: if the IR cannot be compiled again, there is nothing to serve on:
        // the engine fails and says which stage and that the compile failed.
        let mut e = mock_box();
        e.stage_idle_timeout = Duration::from_millis(100);
        let ctl = arm_recompile(&mut e);
        ctl.lock().unwrap().fail = true;
        let events = stuck_on_next_infer(&mut e);
        let old_drops = drops_of(mock1(&mut e));
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        e.submit(verbalizer_pass("d2")).unwrap();
        let c = run_to_final(&mut e);
        let why = c.error.as_deref().expect("refused");
        assert!(
            why.starts_with("backend error: qwen36 engine failed: stage1's infer request")
                && why.contains("recompiling stage1 failed: ")
                && why.contains("compile_model failed")
                && why.ends_with("restart the server"),
            "{why}"
        );
        let ev = events.lock().unwrap().clone();
        assert_eq!(ev.last(), Some(&"recompile"), "{ev:?}");
        assert!(matches!(e.stages[1], OvStage::Gone("recompile")));
        assert_eq!(
            dropped(&old_drops),
            0,
            "the old stage is leaked, not dropped"
        );
        // Sweep item 4: the failed recompile's leaked copy is counted; it is not a recompile.
        assert_eq!((e.rebuild.leaked_copies, e.rebuild.leaked_bytes), (1, 1000));
        assert_eq!(e.rebuild.recompiles_ok, 0);
        assert!(!e.rebuild.degraded_warned);
        assert!(e.engine_failed.is_some());
        assert_eq!(e.failure().reason(), e.engine_failed);
        let t0 = Instant::now();
        e.submit(chat("c3")).unwrap();
        assert_eq!(run_to_final(&mut e).error.as_deref(), Some(why));
        assert!(t0.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn a_recompile_that_never_returns_fails_the_engine_within_the_bound() {
        // Task 54 fix round 2: the recompile runs on a helper thread bounded by
        // `RECOMPILE_TIMEOUT` (short here); a compile that never returns fails the engine after
        // the bound instead of hanging, and later tasks are refused at once.
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        e.stage_idle_timeout = Duration::from_millis(150);
        let ctl = arm_recompile(&mut e);
        e.rebuild.timeout = Duration::from_millis(150);
        ctl.lock().unwrap().hang = true;
        stuck_on_next_infer(&mut e);
        let old_drops = drops_of(mock1(&mut e));
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        let t0 = Instant::now();
        e.submit(verbalizer_pass("d2")).unwrap();
        let c = run_to_final(&mut e);
        let took = t0.elapsed();
        let why = c.error.as_deref().expect("refused");
        assert!(
            why.contains("recompiling stage1 timed out after 150 ms")
                && why.ends_with("restart the server"),
            "{why}"
        );
        // The idle wait, then the compile's bound: two bounds, not forever.
        assert!(
            took >= Duration::from_millis(300) && took < Duration::from_secs(3),
            "{took:?}"
        );
        assert!(matches!(e.stages[1], OvStage::Gone("recompile")));
        assert_eq!(dropped(&old_drops), 0);
        assert!(e.engine_failed.is_some());
        assert_eq!(e.failure().reason(), e.engine_failed);
        let t0 = Instant::now();
        e.submit(chat("c3")).unwrap();
        assert_eq!(run_to_final(&mut e).error.as_deref(), Some(why));
        assert!(t0.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn a_recompiled_stage_that_fails_its_first_infer_fails_the_engine() {
        // Task 54: a fresh copy whose first infer fails points at the device, not the stage: the
        // engine fails rather than recompile again.
        let mut e = mock_box();
        e.stage_idle_timeout = Duration::from_millis(100);
        let ctl = arm_recompile(&mut e);
        ctl.lock().unwrap().fail_first_infer = true;
        stuck_on_next_infer(&mut e);
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        e.submit(chat("c2")).unwrap();
        let c = run_to_final(&mut e);
        let why = c.error.as_deref().expect("fails");
        assert!(why.contains("unknown C++ exception in infer"), "{why}");
        let reason = e.engine_failed.clone().expect("the engine is failed");
        assert!(
            reason.starts_with("stage1 was recompiled and its first infer failed"),
            "{reason}"
        );
        assert_eq!(e.failure().reason(), Some(reason));
        e.submit(chat("c3")).unwrap();
        let c = run_to_final(&mut e);
        assert!(
            c.error
                .as_deref()
                .is_some_and(|w| w.contains("was recompiled and its first infer failed")),
            "{c:?}"
        );
        assert_eq!(e.rebuild.recompiled, 1, "never recompiled a second time");
        assert_eq!(ctl.lock().unwrap().built.len(), 1);
        // The fresh copy faulted on its first infer: leaked and counted, like the old one.
        assert!(matches!(e.stages[1], OvStage::Gone("first_infer")));
        assert_eq!((e.rebuild.leaked_copies, e.rebuild.leaked_bytes), (2, 2000));
        let fresh = Arc::clone(&ctl.lock().unwrap().fresh_drops[0]);
        drop(e);
        assert_eq!(
            dropped(&fresh),
            0,
            "the faulted fresh copy is never dropped"
        );
    }

    #[test]
    fn a_recompiled_stage_whose_first_infer_gets_no_thread_is_leaked_not_dropped() {
        // Sweep item 3: if the helper thread for the fresh copy's first infer cannot be started,
        // `spawn` drops the closure that owns the copy. The copy is wrapped in `Forget` first, so
        // it is leaked, and the engine fails as before.
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        e.stage_idle_timeout = Duration::from_millis(100);
        let ctl = arm_recompile(&mut e);
        stuck_on_next_infer(&mut e);
        let old_drops = drops_of(mock1(&mut e));
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        REFUSE_SPAWN.with(|c| c.set(Some("first_infer")));
        e.submit(chat("c2")).unwrap();
        let c = run_to_final(&mut e);
        REFUSE_SPAWN.with(|c| c.set(None));
        assert!(c.error.is_some(), "{c:?}");
        let reason = e.engine_failed.clone().expect("the engine is failed");
        assert!(
            reason.contains("stage1 was recompiled and its first infer could not be run bounded")
                && reason.contains("thread spawn refused (test)"),
            "{reason}"
        );
        assert!(matches!(e.stages[1], OvStage::Gone("first_infer")));
        assert_eq!(e.rebuild.recompiled, 1);
        assert_eq!((e.rebuild.leaked_copies, e.rebuild.leaked_bytes), (2, 2000));
        let fresh = Arc::clone(&ctl.lock().unwrap().fresh_drops[0]);
        assert_eq!(dropped(&fresh), 0, "the never-run fresh copy is leaked");
        drop(e);
        assert_eq!(dropped(&fresh), 0);
        assert_eq!(dropped(&old_drops), 0);
    }

    #[test]
    fn a_recompiled_stage_whose_first_infer_never_returns_fails_the_engine_within_the_bound() {
        // Task 54 fix round 1, kept: the fresh copy's first infer is bounded (a wedged device
        // queue). Chat and decisions both: the task fails within the bound, later ones at once.
        for next in [chat("c2"), verbalizer_pass("d2")] {
            let id = next.task_id.clone();
            let mut e = mock_box();
            e.no_retry = true; // Task 57 off: the task fails, the next one recovers
            e.stage_idle_timeout = Duration::from_millis(150);
            let ctl = arm_recompile(&mut e);
            ctl.lock().unwrap().hang_first_infer = true;
            stuck_on_next_infer(&mut e);
            e.submit(chat("c1")).unwrap();
            assert!(run_to_final(&mut e).error.is_some());
            let t0 = Instant::now();
            e.submit(next).unwrap();
            let c = run_to_final(&mut e);
            let took = t0.elapsed();
            let why = c.error.as_deref().expect("fails");
            assert!(
                why.contains("its first infer did not return"),
                "{id}: {why}"
            );
            assert!(
                took >= Duration::from_millis(300) && took < Duration::from_secs(3),
                "{id}: {took:?}"
            );
            let reason = e.engine_failed.clone().expect("failed");
            assert!(
                reason.starts_with(
                    "stage1 was recompiled and its first infer did not return within 150 ms"
                ),
                "{id}: {reason}"
            );
            assert!(matches!(e.stages[1], OvStage::Gone("first_infer")), "{id}");
            let t0 = Instant::now();
            e.submit(chat("c3")).unwrap();
            let c = run_to_final(&mut e);
            assert!(
                c.error.as_deref().is_some_and(|w| w.contains(&reason)),
                "{id}"
            );
            assert!(t0.elapsed() < Duration::from_millis(100), "{id}");
        }
    }

    #[test]
    fn a_recompiled_emb_whose_first_infer_never_returns_fails_the_engine_within_the_bound() {
        // Task 54 re-review: the embeddings take the same path as a stage. A stuck emb request is
        // abandoned and the emb recompiled; its first infer is bounded too.
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        e.stage_idle_timeout = Duration::from_millis(150);
        let ctl = arm_recompile(&mut e);
        ctl.lock().unwrap().hang_first_infer = true;
        let (events, old_drops) = {
            let Some(OvStage::Mock(m)) = e.emb.as_mut() else {
                unreachable!()
            };
            m.fail_infer_at = Some(m.infers);
            m.busy_after_fail = u32::MAX;
            (Arc::clone(&m.events), drops_of(m))
        };
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        let t0 = Instant::now();
        e.submit(verbalizer_pass("d2")).unwrap();
        let c = run_to_final(&mut e);
        let took = t0.elapsed();
        assert!(c.error.is_some());
        assert!(
            took >= Duration::from_millis(300) && took < Duration::from_secs(3),
            "{took:?}"
        );
        let reason = e.engine_failed.clone().expect("failed");
        assert!(
            reason
                .starts_with("emb was recompiled and its first infer did not return within 150 ms"),
            "{reason}"
        );
        assert!(matches!(e.emb, Some(OvStage::Gone("first_infer"))));
        assert_eq!(e.failure().reason().as_deref(), Some(reason.as_str()));
        assert_eq!(ctl.lock().unwrap().built, ["emb"]);
        let ev = events.lock().unwrap().clone();
        let i = ev
            .iter()
            .position(|&a| a == "recompile")
            .expect("recompiled");
        assert_eq!(ev[i - 1], "abandon", "{ev:?}");
        assert_eq!(dropped(&old_drops), 0);
        let t0 = Instant::now();
        e.submit(chat("c3")).unwrap();
        let c = run_to_final(&mut e);
        assert!(c.error.as_deref().is_some_and(|w| w.contains(&reason)));
        assert!(t0.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn a_priming_fold_that_leaves_a_request_stuck_is_recompiled_within_its_own_admission() {
        // Task 46 failed the admitting task here (the fold that primes a warm restore left its
        // request busy). Task 54: the cold path's reset recompiles the stage, and the same task
        // is served from a cold prefill. Since Task 55 a warm pass right after a decision pass
        // restores over the primed requests with no fold, so a chat turn runs in between: its
        // finish resets the requests, and the next restore must prime them.
        let mut e = mock_box();
        e.prefix_cache = crate::prefix_cache::PrefixCache::new(1 << 20);
        e.stage_idle_timeout = Duration::from_millis(200);
        let min = crate::prefix_cache::MIN_PREFIX_TOKENS;
        let pass = |id: &str, first: bool| {
            let mut t = verbalizer_pass(id);
            let d = t.decision.as_mut().unwrap();
            d.prompt_ids = (0..min as u32 + 8).map(|i| i % MOCK_VOCAB as u32).collect();
            d.cache_prefix = Some(min as u32);
            d.needs_snapshot = first;
            t
        };
        e.submit(pass("first", true)).unwrap();
        assert!(run_to_final(&mut e).error.is_none());
        e.submit(chat("between")).unwrap();
        assert!(run_to_final(&mut e).error.is_none());
        assert!(!e.primed, "the chat's finish reset the requests");
        let events = stuck_on_next_infer(&mut e); // the priming fold of the next restore
        let t0 = Instant::now();
        e.submit(pass("warm", false)).unwrap();
        let c = run_to_final(&mut e);
        assert!(c.error.is_none(), "{c:?}");
        assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
        assert!(events.lock().unwrap().contains(&"recompile"));
        assert!(e.engine_failed.is_none() && e.fresh_requests.is_empty());
        assert_eq!(e.rebuild.recompiled, 1);
    }

    #[test]
    fn a_failed_task_reports_the_prefix_it_resumed_from_the_cache() {
        // Task 45: the BUG-1 shape, a warm restore and then a throw in the suffix's prefill. The
        // error chunk says how much was resumed, so the API's failure line can say "hit".
        let mut e = mock_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        e.prefix_cache = crate::prefix_cache::PrefixCache::new(1 << 20);
        let min = crate::prefix_cache::MIN_PREFIX_TOKENS;
        let pass = |id: &str, first: bool| {
            let mut t = verbalizer_pass(id);
            let d = t.decision.as_mut().unwrap();
            d.prompt_ids = (0..min as u32 + 8).map(|i| i % MOCK_VOCAB as u32).collect();
            d.cache_prefix = Some(min as u32);
            d.needs_snapshot = first;
            d.require_warm = !first;
            t
        };
        e.submit(pass("first", true)).unwrap();
        assert!(run_to_final(&mut e).error.is_none());
        let OvStage::Mock(m) = &mut e.stages[1] else {
            unreachable!()
        };
        // Task 55: the restore writes over the requests the first pass left primed, with no
        // priming fold, so the suffix is the next infer.
        m.fail_infer_at = Some(m.infers);
        e.submit(pass("warm", false)).unwrap();
        let c = run_to_final(&mut e);
        let why = c.error.as_deref().expect("failed");
        assert!(why.starts_with("prefill failed"), "{why}");
        assert_eq!(
            c.warm_prefix,
            Some(min as u32),
            "a cache hit at the shared prefix"
        );
    }

    #[test]
    fn a_decision_pass_records_its_timing() {
        let mut e = mock_box();
        e.submit(verbalizer_pass("t")).unwrap();
        assert!(run_to_final(&mut e).error.is_none());
        let t = e
            .last_timing
            .clone()
            .expect("a finished decision pass records its timing");
        assert_eq!(
            (t.prompt, t.warm_prefix, t.prefilled(), t.spans, t.span_max),
            (4, 0, 4, 1, 4)
        );
        assert_eq!(t.stage_infer.len(), 2, "one entry per local stage");
        assert!(!t.recreate, "a cold pass takes the cheap reset");
    }

    #[test]
    fn a_chat_turn_records_no_decision_timing() {
        let mut e = mock_box();
        e.submit(chat("c")).unwrap();
        assert!(run_to_final(&mut e).error.is_none());
        assert!(e.last_timing.is_none());
    }

    // ---- Task 55: H-lite, decision requests stay primed between restored passes ----

    /// A pass of one Kev-like request: every question shares the prefix `0..min` (the cached
    /// state); the suffix differs per question `q`. The first pass snapshots the prefix, the
    /// later ones require it.
    fn shared_prefix_pass(id: &str, q: u32, first: bool) -> GenerationTask {
        let min = crate::prefix_cache::MIN_PREFIX_TOKENS as u32;
        let mut t = verbalizer_pass(id);
        let d = t.decision.as_mut().unwrap();
        d.prompt_ids = (0..min)
            .map(|i| i % MOCK_VOCAB as u32)
            .chain((0..8).map(|i| (i + q) % MOCK_VOCAB as u32))
            .collect();
        d.cache_prefix = Some(min);
        d.needs_snapshot = first;
        d.require_warm = !first;
        t
    }

    /// A box with the prefix cache on.
    fn cached_box() -> Qwen36Engine {
        let mut e = mock_box();
        e.prefix_cache = crate::prefix_cache::PrefixCache::new(1 << 20);
        e
    }

    /// The answer a fresh engine gives `task` from a cold prefill (no cache, nothing restored).
    fn fresh_cold_answer(task: &GenerationTask) -> Vec<f32> {
        let mut t = task.clone();
        let d = t.decision.as_mut().unwrap();
        d.cache_prefix = None;
        d.needs_snapshot = false;
        d.require_warm = false;
        let mut e = mock_box();
        e.submit(t).unwrap();
        let c = run_to_final(&mut e);
        assert_eq!(e.last_timing.as_ref().unwrap().warm_prefix, 0, "cold");
        c.decision_logits.expect("an answer")
    }

    fn answer(e: &mut Qwen36Engine, task: GenerationTask) -> Vec<f32> {
        e.submit(task).unwrap();
        let c = run_to_final(e);
        assert!(c.error.is_none(), "{c:?}");
        c.decision_logits.expect("an answer")
    }

    fn events_of(e: &Qwen36Engine, stage: usize) -> Arc<std::sync::Mutex<Vec<&'static str>>> {
        let OvStage::Mock(m) = &e.stages[stage] else {
            unreachable!()
        };
        Arc::clone(&m.events)
    }

    #[test]
    fn warm_passes_after_a_warm_pass_restore_over_the_live_requests() {
        // H-lite: hits 2..n write the restored state over the requests the previous pass left
        // primed: no recreate, no priming fold, no reset. The answers equal a fresh engine's cold
        // ones (the mock's logit reads the state depth, so a stale or dropped state would show).
        let min = crate::prefix_cache::MIN_PREFIX_TOKENS;
        let mut e = cached_box();
        let first = shared_prefix_pass("q0", 0, true);
        assert_eq!(answer(&mut e, first.clone()), fresh_cold_answer(&first));
        assert!(
            e.decision_scrub_deferred && e.primed,
            "the finish kept the requests"
        );
        for q in 1..5 {
            let ev = [events_of(&e, 0), events_of(&e, 1)];
            let n = ev
                .iter()
                .map(|v| v.lock().unwrap().len())
                .collect::<Vec<_>>();
            let task = shared_prefix_pass(&format!("q{q}"), q, false);
            assert_eq!(
                answer(&mut e, task.clone()),
                fresh_cold_answer(&task),
                "q{q}"
            );
            let t = e.last_timing.clone().unwrap();
            assert_eq!(t.warm_prefix, min, "q{q}: a hit");
            assert_eq!(t.prime, Duration::ZERO, "q{q}: no priming fold");
            assert!(!t.recreate, "q{q}: no recreate");
            assert_eq!(t.reset, Duration::ZERO, "q{q}: no reset at all");
            assert_eq!(t.spans, 1, "q{q}: the suffix only");
            for (s, v) in ev.iter().enumerate() {
                let since = v.lock().unwrap()[n[s]..].to_vec();
                assert_eq!(since, ["set_state", "infer"], "q{q} stage{s}: {since:?}");
            }
            assert!(e.decision_scrub_deferred && e.primed && e.state_restored);
        }
    }

    #[test]
    fn a_cold_decision_after_a_warm_one_matches_a_fresh_engines_cold_pass() {
        let mut e = cached_box();
        answer(&mut e, shared_prefix_pass("q0", 0, true));
        answer(&mut e, shared_prefix_pass("q1", 1, false));
        let ev = events_of(&e, 1);
        let n = ev.lock().unwrap().len();
        let cold = verbalizer_pass("cold"); // shares nothing with the cached prefix
        assert_eq!(answer(&mut e, cold.clone()), fresh_cold_answer(&cold));
        let t = e.last_timing.clone().unwrap();
        assert_eq!(t.warm_prefix, 0);
        assert!(
            t.recreate,
            "the deferred scrub recreates a restored request"
        );
        assert_eq!(t.reset, t.admit_reset, "the admission's own reset");
        let since = ev.lock().unwrap()[n..].to_vec();
        assert_eq!(since, ["reset", "infer"], "scrubbed before the prefill");
        assert!(e.decision_scrub_deferred, "and this pass defers its own");
    }

    #[test]
    fn a_chat_after_a_decision_pass_is_served_on_scrubbed_requests() {
        let min = crate::prefix_cache::MIN_PREFIX_TOKENS;
        let tokens = |e: &mut Qwen36Engine, id: &str| {
            // Each `hi` is token 0.
            let mut t = windowed_task(id, &"hi ".repeat(min + 4));
            t.max_tokens = 2;
            e.submit(t).unwrap();
            let mut ids = Vec::new();
            for _ in 0..50 {
                for (_, c) in e.step().expect("step") {
                    assert!(c.error.is_none(), "{c:?}");
                    ids.extend(c.token_ids.iter().copied());
                    if c.is_final {
                        return ids;
                    }
                }
            }
            panic!("no final chunk");
        };
        let fresh = tokens(&mut mock_box(), "fresh");
        let mut e = cached_box();
        answer(&mut e, shared_prefix_pass("q0", 0, true));
        answer(&mut e, shared_prefix_pass("q1", 1, false));
        // A cached prefix of the chat prompt, so the chat turn is a prefix-cache hit: only a
        // decision pass's own hit may write over the live requests.
        let parts = e.parts_local_stages().unwrap();
        assert!(e.prefix_cache.insert(vec![0; min], parts));
        let ev = [events_of(&e, 0), events_of(&e, 1)];
        let n = ev
            .iter()
            .map(|v| v.lock().unwrap().len())
            .collect::<Vec<_>>();
        assert_eq!(tokens(&mut e, "c"), fresh);
        for (s, v) in ev.iter().enumerate() {
            let since = v.lock().unwrap()[n[s]..].to_vec();
            assert_eq!(
                since[..3],
                ["reset", "infer", "set_state"],
                "stage{s}: scrubbed (recreate), primed, then restored: {since:?}"
            );
        }
        assert!(!e.decision_scrub_deferred);
    }

    #[test]
    fn a_failed_decision_pass_still_scrubs() {
        let mut e = cached_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        answer(&mut e, shared_prefix_pass("q0", 0, true));
        // The warm pass restores over the live requests, then its suffix infer throws.
        let at = infers(&e, 1);
        mock1(&mut e).fail_infer_at = Some(at);
        e.submit(shared_prefix_pass("q1", 1, false)).unwrap();
        let c = run_to_final(&mut e);
        assert!(
            c.error.as_deref().unwrap().starts_with("prefill failed"),
            "{c:?}"
        );
        assert!(e.scrub_pending, "the failure scrub is pending");
        let ev = events_of(&e, 1);
        let n = ev.lock().unwrap().len();
        let task = shared_prefix_pass("q2", 2, false);
        assert_eq!(answer(&mut e, task.clone()), fresh_cold_answer(&task));
        let since = ev.lock().unwrap()[n..].to_vec();
        assert_eq!(
            since,
            ["reset", "infer", "set_state", "infer"],
            "scrub (recreate), prime, restore, suffix"
        );
        let t = e.last_timing.clone().unwrap();
        assert_eq!(t.warm_prefix, crate::prefix_cache::MIN_PREFIX_TOKENS);
        assert!(!e.scrub_pending);
    }

    #[test]
    fn with_the_restore_clear_knob_a_hit_after_a_decision_pass_scrubs_clears_and_primes() {
        // Sweep item 6 (the A/B branch in `local_warm_or_cold`): with
        // `CASCADIA_QWEN36_RESTORE_CLEAR=recreate|reset` a decision pass's hit runs the deferred
        // scrub first (a recreate: the previous pass restored), then the knob's clear, then one
        // priming fold, then the set. `reset_state` therefore never runs on a restored request.
        for knob in ["recreate", "reset"] {
            RESTORE_CLEAR.with(|c| c.set(Some(knob)));
            let mut e = cached_box();
            answer(&mut e, shared_prefix_pass("q0", 0, true));
            answer(&mut e, shared_prefix_pass("q1", 1, false));
            assert!(e.decision_scrub_deferred && e.state_restored, "{knob}");
            let ev = [events_of(&e, 0), events_of(&e, 1)];
            let n = ev
                .iter()
                .map(|v| v.lock().unwrap().len())
                .collect::<Vec<_>>();
            let task = shared_prefix_pass("q2", 2, false);
            assert_eq!(
                answer(&mut e, task.clone()),
                fresh_cold_answer(&task),
                "{knob}"
            );
            RESTORE_CLEAR.with(|c| c.set(None));
            let t = e.last_timing.clone().unwrap();
            assert_eq!(
                t.warm_prefix,
                crate::prefix_cache::MIN_PREFIX_TOKENS,
                "{knob}: a hit"
            );
            assert!(t.recreate, "{knob}: the deferred scrub recreated");
            for (s, v) in ev.iter().enumerate() {
                let since = v.lock().unwrap()[n[s]..].to_vec();
                assert_eq!(
                    since,
                    ["reset", "reset", "infer", "set_state", "infer"],
                    "{knob} stage{s}: scrub, clear, prime, set, suffix"
                );
            }
            assert!(
                e.decision_scrub_deferred,
                "{knob}: and this pass defers its own"
            );
        }
        assert_eq!(restore_clear_mode_of(Some("recreate")), "recreate_request");
        assert_eq!(restore_clear_mode_of(Some("reset")), "reset_state");
        assert_eq!(restore_clear_mode_of(None), "none");
        assert_eq!(restore_clear_mode_of(Some("bogus")), "none");
    }

    #[test]
    fn a_restore_onto_a_just_recompiled_stage_primes_it_first() {
        // Task 54 x Task 55: a warm pass gets stage 1 stuck; the next warm pass's admission
        // recompiles it. The fresh stage is un-run, and a set_state onto an un-run request is
        // dropped (the mock drops it too, which would change the answer), so the restore must
        // prime it with one fold first, bounded as its first infer.
        let mut e = cached_box();
        e.no_retry = true; // Task 57 off: the task fails, the next one recovers
        e.stage_idle_timeout = Duration::from_millis(200);
        answer(&mut e, shared_prefix_pass("q0", 0, true));
        let events = stuck_on_next_infer(&mut e); // q1's suffix (no priming fold since Task 55)
        e.submit(shared_prefix_pass("q1", 1, false)).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        let n = events.lock().unwrap().len();
        let task = shared_prefix_pass("q2", 2, false);
        assert_eq!(answer(&mut e, task.clone()), fresh_cold_answer(&task));
        let since = events.lock().unwrap()[n..].to_vec();
        let abandon = since
            .iter()
            .position(|&a| a == "abandon")
            .expect("abandoned");
        assert_eq!(
            &since[abandon..],
            ["abandon", "recompile", "infer", "set_state", "infer"],
            "recompiled, primed, restored, suffix: {since:?}"
        );
        assert_eq!(e.rebuild.recompiled, 1);
        assert!(e.fresh_requests.is_empty() && e.engine_failed.is_none());
        let t = e.last_timing.clone().unwrap();
        assert_eq!(t.warm_prefix, crate::prefix_cache::MIN_PREFIX_TOKENS);
    }

    #[test]
    fn a_warm_restore_the_shim_refuses_falls_back_to_a_recreate_and_a_cold_prefill() {
        // The restore over live requests is safe only because it writes every state: the shim
        // refuses a blob whose state count differs from the request's (or any state it cannot
        // apply). Then the pass recreates the requests and prefills cold.
        let mut e = cached_box();
        answer(&mut e, shared_prefix_pass("q0", 0, true));
        mock1(&mut e).refuse_set = true;
        let mut task = shared_prefix_pass("q1", 1, false);
        task.decision.as_mut().unwrap().require_warm = false;
        assert_eq!(answer(&mut e, task.clone()), fresh_cold_answer(&task));
        let t = e.last_timing.clone().unwrap();
        assert_eq!(t.warm_prefix, 0, "cold");
        assert!(t.recreate, "the half-written requests are recreated");
    }

    #[test]
    fn a_pipeline_decision_finish_still_resets_and_only_a_single_box_defers() {
        for total in [1u32, 2] {
            let mut e = mock_box();
            e.total = total;
            e.primed = true;
            e.state_restored = true;
            e.active = Some(ActiveTask {
                decision: verbalizer_pass("d").decision,
                logits: vec![0.0; MOCK_VOCAB],
                ..finalizable_task(4, 0)
            });
            let ev = events_of(&e, 0);
            let out = e.finish_decision();
            assert!(out[0].1.error.is_none(), "total {total}: {:?}", out[0].1);
            let resets = ev.lock().unwrap().iter().filter(|&&a| a == "reset").count();
            if total == 1 {
                assert!(e.decision_scrub_deferred && e.primed && e.state_restored);
                assert_eq!(resets, 0);
            } else {
                assert!(!e.decision_scrub_deferred && !e.primed && !e.state_restored);
                assert_eq!(resets, 1, "the finish recreates, as before Task 55");
                assert!(e.timing.recreate);
            }
        }
    }

    #[test]
    fn a_relay_whose_engine_failed_fails_the_heads_task_well_under_the_reply_bound() {
        // Task 46 review I1: a relay rank whose stage request stays busy used to ack RESET and
        // then drop every FORWARD, so each head task waited out `recv_timeout` (60 s). The relay
        // now returns a connection-fatal error instead of the ack; `run_relay_loop` exits and the
        // closed socket fails the head's exchange at once. Rank 1 is driven here as
        // `run_relay_loop` drives it: on a connection-fatal error the stage is dropped. Task 54:
        // a relay recompiles the stage like a single box does, so this rank has used up its
        // recompiles (cap 0) and its engine fails.
        let _knob = crate::timeout_knob_lock();
        cascadia_transport::set_activation_timeout_secs(30);
        let rt = pipe_rt();
        let (server, client) = rt.block_on(async {
            let mut server = ActivationServer::new("127.0.0.1", 0);
            server.start().await.unwrap();
            let mut client = ActivationClient::new("127.0.0.1", server.port());
            let (c, a) = tokio::join!(client.connect(), server.accept());
            c.unwrap();
            a.unwrap();
            (server, client)
        });

        // Rank 1: its last infer failed and left the request busy for good.
        let mut relay = bare_engine(2, "{}");
        relay.rank = 1;
        relay.hidden = MOCK_HIDDEN;
        relay.upstream = Some(Arc::new(tokio::sync::Mutex::new(server)));
        relay.runtime_handle = Some(rt.handle().clone());
        relay.stage_idle_timeout = Duration::from_millis(200);
        let mut s1 = MockStage::new(&LAST_STAGE_INPUTS, MOCK_VOCAB, Some(1));
        s1.busy = u32::MAX;
        relay.stages = vec![OvStage::Mock(s1)];
        relay.infer_failed = true;
        relay.rebuild.cap = 0;
        let relay_loop = std::thread::spawn(move || loop {
            match relay.step() {
                Err(e) if e.is_connection_fatal() => {
                    let failed = relay.engine_failed.is_some();
                    // Task 56: a relay exits through this error, never through the handle.
                    assert_eq!(relay.failure().reason(), None);
                    drop(relay); // the process exit closes the socket
                    return (e.to_string(), failed);
                }
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
                Ok(_) => {}
            }
        });

        // Rank 0.
        let mut head = bare_engine(2, "{}");
        head.rank = 0;
        head.handshake_done = true;
        head.downstream = Some(Arc::new(tokio::sync::Mutex::new(client)));
        head.runtime_handle = Some(rt.handle().clone());
        head.hidden = MOCK_HIDDEN;
        head.tokenizer = Some(tiny_tokenizer());
        let emb = MockStage::new(&["input_ids"], MOCK_HIDDEN, None);
        let s0 = MockStage::new(&FIRST_STAGE_INPUTS, MOCK_HIDDEN, None);
        head.emb = Some(OvStage::Mock(emb));
        head.stages = vec![OvStage::Mock(s0)];

        let t0 = Instant::now();
        head.pending.push(chat("t1"));
        let out = head.step_pipe_first();
        let took = t0.elapsed();
        let msg = out
            .first()
            .and_then(|(_, c)| c.error.clone())
            .expect("the head's task fails");
        assert!(
            took < Duration::from_secs(5),
            "failed in {took:?}, not after the 30 s reply bound: {msg}"
        );
        assert!(
            msg.contains("RESET not acked"),
            "no ack, a closed link: {msg}"
        );
        let (why, failed) = relay_loop.join().unwrap();
        assert!(failed, "rank 1's engine is failed");
        assert!(
            why.contains("qwen36 engine failed: stage0") && why.contains("supervisor rebuilds"),
            "{why}"
        );

        // The task after it fails at once too: the link is gone.
        let t0 = Instant::now();
        head.pending.push(chat("t2"));
        let out = head.step_pipe_first();
        assert!(out.first().and_then(|(_, c)| c.error.as_ref()).is_some());
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        cascadia_transport::set_activation_timeout_secs(0);
    }

    #[test]
    fn a_relay_recompiles_a_stuck_stage_and_the_heads_task_is_served() {
        // Task 54 in pipeline mode: the relay's RESET runs the same recovery as a single box, so a
        // request left stuck on rank 1 is abandoned and its stage recompiled inside the RESET, the
        // ack follows, and the head's task is served. No frame changes: the head only sees a
        // slower ack.
        let _knob = crate::timeout_knob_lock();
        cascadia_transport::set_activation_timeout_secs(30);
        let rt = pipe_rt();
        let (server, client) = rt.block_on(async {
            let mut server = ActivationServer::new("127.0.0.1", 0);
            server.start().await.unwrap();
            let mut client = ActivationClient::new("127.0.0.1", server.port());
            let (c, a) = tokio::join!(client.connect(), server.accept());
            c.unwrap();
            a.unwrap();
            (server, client)
        });

        let mut relay = bare_engine(2, "{}");
        relay.rank = 1;
        relay.hidden = MOCK_HIDDEN;
        relay.upstream = Some(Arc::new(tokio::sync::Mutex::new(server)));
        relay.runtime_handle = Some(rt.handle().clone());
        relay.stage_idle_timeout = Duration::from_millis(200);
        let mut s1 = MockStage::new(&LAST_STAGE_INPUTS, MOCK_VOCAB, Some(1));
        s1.busy = u32::MAX;
        let events = Arc::clone(&s1.events);
        let old_drops = drops_of(&s1);
        relay.stages = vec![OvStage::Mock(s1)];
        let ctl = arm_recompile(&mut relay);
        relay.infer_failed = true;
        let relay_loop = std::thread::spawn(move || loop {
            match relay.step() {
                Err(e) if e.is_connection_fatal() => {
                    return (relay.engine_failed.clone(), relay.rebuild.recompiled);
                }
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
                Ok(_) => {}
            }
        });

        let mut head = bare_engine(2, "{}");
        head.rank = 0;
        head.handshake_done = true;
        head.downstream = Some(Arc::new(tokio::sync::Mutex::new(client)));
        head.runtime_handle = Some(rt.handle().clone());
        head.hidden = MOCK_HIDDEN;
        head.tokenizer = Some(tiny_tokenizer());
        let emb = MockStage::new(&["input_ids"], MOCK_HIDDEN, None);
        let s0 = MockStage::new(&FIRST_STAGE_INPUTS, MOCK_HIDDEN, None);
        head.emb = Some(OvStage::Mock(emb));
        head.stages = vec![OvStage::Mock(s0)];

        let t0 = Instant::now();
        head.pending.push(chat("t1"));
        let mut fin = None;
        for _ in 0..50 {
            if let Some((_, c)) = head.step_pipe_first().into_iter().find(|(_, c)| c.is_final) {
                fin = Some(c);
                break;
            }
        }
        let c = fin.expect("the head's task finishes");
        assert!(
            c.error.is_none(),
            "served after the relay's recompile: {c:?}"
        );
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        drop(head); // closes the link; the relay loop exits
        let (failed, recompiled) = relay_loop.join().unwrap();
        assert!(failed.is_none(), "{failed:?}");
        assert_eq!(recompiled, 1);
        assert_eq!(ctl.lock().unwrap().built, ["stage0"]);
        let ev = events.lock().unwrap().clone();
        let i = ev
            .iter()
            .position(|&a| a == "recompile")
            .expect("recompiled");
        assert_eq!(ev[i - 1], "abandon", "{ev:?}");
        assert_eq!(ev.get(i + 1), Some(&"infer"), "{ev:?}");
        assert_eq!(dropped(&old_drops), 0, "the relay's old stage is leaked");
        cascadia_transport::set_activation_timeout_secs(0);
    }

    #[test]
    fn a_downstream_rank_serves_resets_as_before_task_41() {
        // A pre-41 rank 0 sends RESET with `pos` 0; the relay acks it and serves the task.
        let rt = pipe_rt();
        let (server, mut client) = rt.block_on(async {
            let mut server = ActivationServer::new("127.0.0.1", 0);
            server.start().await.unwrap();
            let mut client = ActivationClient::new("127.0.0.1", server.port());
            let (c, a) = tokio::join!(client.connect(), server.accept());
            c.unwrap();
            a.unwrap();
            (server, client)
        });
        let mut e = bare_engine(2, "{}");
        e.rank = 1;
        e.hidden = MOCK_HIDDEN;
        e.upstream = Some(Arc::new(tokio::sync::Mutex::new(server)));
        e.runtime_handle = Some(rt.handle().clone());
        let s1 = MockStage::new(&LAST_STAGE_INPUTS, MOCK_VOCAB, Some(1));
        e.stages = vec![OvStage::Mock(s1)];
        let fake = rt.spawn(async move {
            let c = &mut client;
            for epoch in [1u32, 2] {
                c.send_raw(&frame_header(FRAME_RESET, epoch, 0))
                    .await
                    .unwrap();
                let (k, _, _) = parse_header(&c.recv_raw(12).await.unwrap());
                assert_eq!(k, FRAME_RESET_ACK);
                c.send_raw(&frame_header(FRAME_FORWARD, epoch, 0))
                    .await
                    .unwrap();
                let t = WireTensor::new(
                    WireDType::F32,
                    [1, 1, MOCK_HIDDEN as u32],
                    le_bytes_f32(&[0.0; MOCK_HIDDEN]),
                );
                c.send(&t).await.unwrap();
                let (k, _, _) = parse_header(&c.recv_raw(12).await.unwrap());
                assert_eq!(k, FRAME_TOKEN);
                let tb = c.recv_raw(8).await.unwrap();
                assert_eq!(i32::from_be_bytes([tb[0], tb[1], tb[2], tb[3]]), 1);
            }
        });
        for _ in 0..4 {
            e.step_pipe_relay().expect("relay step");
        }
        rt.block_on(fake).unwrap();
        assert_eq!(infers(&e, 0), 2);
    }

    // ---- Task 57: retry a request once after a native infer failure recovers ----

    /// cascadia-runner's `MAX_CONSECUTIVE_EMPTY_STEPS`: a stream closes at this many empty steps
    /// in a row.
    const RUNNER_EMPTY_LIMIT: usize = 3;

    /// Step as the runner does until the task's final chunk: every chunk, and the longest run of
    /// consecutive empty steps.
    fn drive(e: &mut Qwen36Engine) -> (Vec<Chunk>, usize) {
        let (mut chunks, mut run, mut longest) = (Vec::new(), 0, 0);
        for _ in 0..50 {
            let out = e.step().expect("step");
            if out.is_empty() {
                run += 1;
                longest = longest.max(run);
                continue;
            }
            run = 0;
            for (_, c) in out {
                let last = c.is_final;
                chunks.push(c);
                if last {
                    return (chunks, longest);
                }
            }
        }
        panic!("no final chunk");
    }

    fn text(chunks: &[Chunk]) -> String {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    fn long_chat(id: &str) -> GenerationTask {
        let mut t = chat(id);
        t.max_tokens = 4;
        t
    }

    #[test]
    fn a_decode_failure_mid_chat_is_retried_and_the_stream_matches_an_uninterrupted_run() {
        let mut clean = mock_box();
        clean.submit(long_chat("c1")).unwrap();
        let (want, _) = drive(&mut clean);
        assert_eq!(want.len(), 5, "4 tokens and the final marker: {want:?}");
        // Stage 1's infers: the prefill 0, decodes 1 and 2; decode 2 fails, one token in. The
        // request idles after 3 waits (kept and reset), or never: abandoned after the full idle
        // bound and recompiled. Either way inside the failing step, so the stream sees one empty
        // step (the retry's prefill) between two tokens.
        for stuck in [false, true] {
            let mut e = mock_box();
            if stuck {
                e.stage_idle_timeout = Duration::from_millis(200);
            }
            let events = busy_after_failed_infer(&mut e, 2, if stuck { u32::MAX } else { 3 });
            e.submit(long_chat("c1")).unwrap();
            let t0 = Instant::now();
            let (got, longest) = drive(&mut e);
            assert!(
                got.iter().all(|c| c.error.is_none()),
                "stuck={stuck}: {got:?}"
            );
            assert_eq!(text(&got), text(&want), "stuck={stuck}");
            assert_eq!(got.len(), want.len(), "max_tokens bounds prefix + new");
            let fin = got.last().unwrap();
            assert_eq!(fin.finish_reason, want.last().unwrap().finish_reason);
            assert_eq!(fin.finish_reason, Some(FinishReason::Length));
            let usage = want.last().unwrap().prompt_tokens;
            assert_eq!(
                fin.prompt_tokens, usage,
                "the resumed tokens are not prompt tokens"
            );
            assert_eq!(longest, 1, "stuck={stuck}: one empty step at a time");
            let ev = events.lock().unwrap().clone();
            assert!(ev.contains(&"cancel"), "the recovery ran: {ev:?}");
            assert_eq!(ev.contains(&"recompile"), stuck, "{ev:?}");
            if stuck {
                assert!(t0.elapsed() >= Duration::from_millis(200));
            }
            assert!(e.retry.is_none() && e.pending.is_empty() && !e.scrub_pending);
            assert!(!e.infer_failed && e.engine_failed.is_none());
        }
    }

    #[test]
    fn a_prefill_failure_is_retried_and_served() {
        let mut clean = mock_box();
        clean.submit(chat("c1")).unwrap();
        let (want, _) = drive(&mut clean);
        let mut e = mock_box();
        busy_after_failed_infer(&mut e, 0, 3);
        e.submit(chat("c1")).unwrap();
        let (got, longest) = drive(&mut e);
        assert!(got.iter().all(|c| c.error.is_none()), "{got:?}");
        assert_eq!(text(&got), text(&want));
        assert_eq!(got.len(), want.len());
        assert_eq!(
            longest, 1,
            "the failed prefill and the retried one are one step"
        );
        assert_eq!(
            infers(&e, 1),
            1 + 1 + 2,
            "failed prefill, prefill, 2 decodes"
        );
    }

    #[test]
    fn a_failed_decision_pass_is_retried_unchanged_and_answers_as_an_uninterrupted_pass() {
        let mut clean = mock_box();
        clean.submit(verbalizer_pass("d1")).unwrap();
        let (want, _) = drive(&mut clean);
        let want = want[0].decision_logits.clone().expect("the pass's logits");

        // Prefill: the pass's stage-1 infer 0 fails.
        let mut e = mock_box();
        busy_after_failed_infer(&mut e, 0, 3);
        e.submit(verbalizer_pass("d1")).unwrap();
        let (got, longest) = drive(&mut e);
        assert_eq!(
            got.len(),
            1,
            "one final chunk, no chunk for the failed attempt: {got:?}"
        );
        assert!(got[0].error.is_none(), "{got:?}");
        assert_eq!(got[0].decision_logits.as_ref(), Some(&want));
        assert_eq!(longest, 1);

        // Head: no head reads a native infer today (the joint head's does not set
        // `infer_failed`), so this simulates one: the flag is set and this pass's head cannot
        // read its answer. The re-queued task is the original, so the retry answers.
        let mut e = mock_box();
        e.submit(verbalizer_pass("d2")).unwrap();
        assert!(e.step().unwrap().is_empty(), "admission and prefill");
        e.infer_failed = true;
        let t = e.active.as_mut().unwrap();
        let Some(cascadia_types::ReadoutHead::Verbalizer { token_ids }) =
            t.decision.as_mut().map(|d| &mut d.head)
        else {
            unreachable!()
        };
        token_ids[0] = 99;
        assert!(
            e.step().unwrap().is_empty(),
            "the head failed; the retry's prefill rides the same step"
        );
        let (got, longest) = drive(&mut e);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(got[0].error.is_none(), "{got:?}");
        assert_eq!(got[0].decision_logits.as_ref(), Some(&want));
        assert_eq!(
            longest, 0,
            "prefill, head failure + retried prefill, then the answer"
        );
        assert!(e.retry.is_none() && e.pending.is_empty());
    }

    #[test]
    fn a_failure_of_the_retry_fails_the_task_and_says_it_was_the_retry() {
        let mut e = mock_box();
        // Stage 1: prefill 0, decode 1, decode 2 fails; the retry's prefill 3, its decode 4 fails.
        busy_after_failed_infer(&mut e, 2, 3);
        mock1(&mut e).fail_infer_again = Some(4);
        e.submit(long_chat("c1")).unwrap();
        let (got, longest) = drive(&mut e);
        assert_eq!(got.len(), 2, "one token, then the error: {got:?}");
        assert!(got[0].error.is_none());
        assert_eq!(
            got[1].error.as_deref(),
            Some(
                "decode failed: backend error: unknown C++ exception in infer (on the retry after \
                 an infer failure)"
            )
        );
        assert!(longest < RUNNER_EMPTY_LIMIT);
        assert!(
            e.pending.is_empty() && e.retry.is_none(),
            "no third attempt"
        );
        assert!(
            e.scrub_pending,
            "the next admission recovers, as before Task 57"
        );
        e.submit(chat("c2")).unwrap();
        let c = run_to_final(&mut e);
        assert!(c.error.is_none(), "{c:?}");
    }

    #[test]
    fn pipeline_rank0_does_not_retry_a_failed_infer() {
        let rt = pipe_rt();
        let (mut e, mut server) = pipe_rank0(&rt);
        e.hidden = MOCK_HIDDEN;
        e.tokenizer = Some(tiny_tokenizer());
        let emb = MockStage::new(&["input_ids"], MOCK_HIDDEN, None);
        let mut s0 = MockStage::new(&FIRST_STAGE_INPUTS, MOCK_HIDDEN, None);
        s0.fail_infer_at = Some(0);
        let events = Arc::clone(&s0.events);
        e.emb = Some(OvStage::Mock(emb));
        e.stages = vec![OvStage::Mock(s0)];
        let fake = rt.spawn(async move {
            let s = &mut server;
            let (k, ep, _) = parse_header(&s.recv_raw(12).await.unwrap());
            assert_eq!((k, ep), (FRAME_RESET, 1));
            fake_ack(s, FRAME_RESET_ACK, 1).await;
        });
        e.pending.push(chat("p1"));
        let out = e.step().unwrap();
        rt.block_on(fake).unwrap();
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(
            out[0].1.error.as_deref(),
            Some("pipeline prefill failed: backend error: unknown C++ exception in infer")
        );
        assert!(e.infer_failed, "a native failure, and still no retry");
        assert!(e.active.is_none() && e.pending.is_empty() && e.retry.is_none());
        assert!(e.scrub_pending, "the next admission recovers, as before");
        let ev = events.lock().unwrap().clone();
        assert_eq!(ev, ["reset", "infer"], "one attempt: {ev:?}");
    }

    #[test]
    fn a_retry_whose_recovery_fails_the_engine_gets_the_refusal_and_nothing_loops() {
        let mut e = mock_box();
        e.stage_idle_timeout = Duration::from_millis(100);
        let ctl = arm_recompile(&mut e);
        ctl.lock().unwrap().fail = true;
        busy_after_failed_infer(&mut e, 1, u32::MAX); // the first decode, stuck for good
        e.submit(chat("c1")).unwrap();
        assert!(e.step().unwrap().is_empty(), "admission and prefill");
        let out = e.step().unwrap();
        assert_eq!(out.len(), 1, "{out:?}");
        let why = out[0].1.error.as_deref().expect("refused");
        assert!(
            why.starts_with("backend error: qwen36 engine failed: stage1's infer request")
                && why.ends_with("restart the server"),
            "the normal engine-failed refusal: {why}"
        );
        assert!(e.engine_failed.is_some());
        assert_eq!(e.failure().reason(), e.engine_failed);
        assert!(e.active.is_none() && e.pending.is_empty() && e.retry.is_none());
        assert!(e.step().unwrap().is_empty(), "nothing left to run");
    }

    #[test]
    fn a_failure_that_fails_the_engine_is_not_retried() {
        // A recompiled stage whose first infer fails fails the engine at the failure itself
        // (Task 54): no retry, the task's own error.
        let mut e = mock_box();
        e.stage_idle_timeout = Duration::from_millis(100);
        let ctl = arm_recompile(&mut e);
        ctl.lock().unwrap().fail_first_infer = true;
        let events = stuck_on_next_infer(&mut e);
        e.no_retry = true;
        e.submit(chat("c1")).unwrap();
        assert!(run_to_final(&mut e).error.is_some());
        e.no_retry = false;
        e.submit(chat("c2")).unwrap();
        let (got, _) = drive(&mut e);
        assert_eq!(got.len(), 1, "{got:?}");
        let why = got[0].error.as_deref().expect("fails");
        assert!(
            why.starts_with("prefill failed: ") && !why.contains("retry"),
            "{why}"
        );
        assert!(e.engine_failed.is_some());
        assert!(e.pending.is_empty() && e.retry.is_none());
        let recompiles = events
            .lock()
            .unwrap()
            .iter()
            .filter(|&&a| a == "recompile")
            .count();
        assert_eq!(recompiles, 1, "no second recovery");
    }

    /// Byte-fallback tokens for "€" (E2 82 AC) after `hi`: a partial sequence decodes to one
    /// U+FFFD per byte, as a byte-level vocab does mid-character.
    fn byte_tokenizer() -> Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":{"type":"ByteFallback"},
            "model":{"type":"WordLevel",
                "vocab":{"hi":0,"<0xE2>":1,"<0x82>":2,"<0xAC>":3},"unk_token":"[UNK]"}}"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("build byte tokenizer")
    }

    #[test]
    fn a_retried_chat_carries_the_bytes_already_emitted_across_a_held_back_partial_char() {
        // The chat generates E2, 82 (both held back: an incomplete char), then AC, whose decode
        // infer fails. The retry resumes over [E2, 82] with the bytes emitted so far (none). A
        // re-decoded seed would be two U+FFFD (6 bytes), and "€" would then be dropped as a
        // diverged decode.
        let run = |fail: bool| {
            let mut e = mock_box();
            e.tokenizer = Some(byte_tokenizer());
            if fail {
                busy_after_failed_infer(&mut e, 3, 3); // prefill 0, decodes 1 (E2), 2 (82), 3 (AC)
            }
            let mut task = chat("u1");
            task.max_tokens = 3;
            e.submit(task).unwrap();
            let mut chunks = Vec::new();
            // Stage 1's peak is the argmax its next infer outputs: the prefill's picks E2, the
            // first decode's 82, every later one AC.
            for i in 0..20usize {
                mock1(&mut e).peak = Some((i + 1).min(3));
                for (_, c) in e.step().expect("step") {
                    let last = c.is_final;
                    chunks.push(c);
                    if last {
                        return chunks;
                    }
                }
            }
            panic!("no final chunk");
        };
        let want = run(false);
        assert_eq!(text(&want), "€");
        let got = run(true);
        assert!(got.iter().all(|c| c.error.is_none()), "{got:?}");
        assert_eq!(text(&got), "€", "{got:?}");
        assert_eq!(got.len(), want.len());
        assert_eq!(got.last().unwrap().prompt_tokens, Some(3));
    }
}
