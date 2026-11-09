//! Wire protocol for pipeline-parallel sparse-MoE inference.
//!
//! Frames flow over the existing cascadia-transport TCP sockets. Each
//! frame begins with a 4-byte big-endian `FrameKind` code; some frames
//! carry length-prefixed tensors after the code via the standard
//! `ActivationServer::send` / `recv` path.
//!
//! Directions:
//! - `Forward` and `Reset` flow downstream (rank N → rank N+1).
//! - `Token` flows upstream (last rank → rank 0).
//!
//! Each rank holds at most one upstream socket (server-side, accepting
//! incoming connection from rank-1) and one downstream socket
//! (client-side, connected outbound to rank+1). The first rank has no
//! upstream; the last rank has no downstream.
//!
//! `Forward` is the workhorse: one per decoded token. Body:
//!   - 4 B big-endian u32 past_seq_len
//!   - 28 B SamplingConfig payload (see [`SamplingWire`]):
//!       - 4 B f32 BE temperature
//!       - 4 B f32 BE top_p
//!       - 4 B f32 BE repetition_penalty
//!       - 8 B u64 BE repetition_window
//!       - 8 B u64 BE seed (0 sentinel = no seed / use entropy)
//!   - F32 hidden tensor [1, 1, hidden_size]
//!
//! `Reset` clears KV state on downstream for a new generation. No body.
//!
//! `Token` returns one sampled token i64 to upstream after the last
//! rank runs head + sample. Body:
//!   - 8 B big-endian i64 token_id
//!
//! `ForwardBatch` carries K hidden states in one frame, used by the
//! pipeline-parallel speculative-decode path. Body:
//!   - 4 B big-endian u32 past_seq_len (KV slot for the FIRST hidden
//!     in the batch; subsequent hiddens occupy past_seq_len+1, +2, ...)
//!   - 4 B big-endian u32 batch_count (K — the number of hidden rows
//!     in the tensor)
//!   - 28 B SamplingConfig payload (same as `Forward` — the sampler
//!     applies the same config to all K positions)
//!   - F32 hidden tensor [1, K, hidden_size]
//!
//! `TokenBatch` returns K sampled token ids upstream in one frame —
//! the response shape that pairs with `ForwardBatch`. Body:
//!   - 4 B big-endian u32 batch_count (K)
//!   - K × 8 B big-endian i64 token_ids
//!
//! Backward compatibility: an older worker that doesn't recognize the
//! new codes returns an error from `parse_kind`. The rank-0 driver
//! defaults to the existing single-token Forward path; the
//! pipeline-parallel spec-decode caller must explicitly opt in.

use std::sync::Arc;

use cascadia_transport::{
    ActivationClient, ActivationServer, DType, Tensor, TransportError, TransportResult,
    MAX_RAW_BYTES,
};
use tokio::sync::Mutex;

use crate::sampling::SamplingConfig;

/// 4-byte big-endian frame kinds. Chosen to be disjoint from
/// `dist_spec`'s `FrameKind` so a stray cross-engine connection would
/// fail fast on an unrecognized code.
///
/// Versioning convention: the low byte of `Forward` is bumped whenever
/// the Forward body layout changes in a wire-incompatible way. Old
/// peers reject the new code at `parse_kind` rather than mis-parsing
/// the body. `Reset` / `Token` have stable bodies and don't need
/// versioning.
///
/// History:
/// - 0x01 — past_seq_len + hidden tensor only (pre-PR #10)
/// - 0x02 — adds 28-byte SamplingConfig between past_seq_len and tensor (PR #10)
/// - 0x03 — adds ForwardBatch + TokenBatch (PR #11, spec-decode forward
///   batching). Forward / Reset / Token codes unchanged so a pre-0x03
///   worker is still wire-compatible for the single-token path; only
///   peers that explicitly send a ForwardBatch trip the unknown-kind
///   check on an old worker.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    // Forward / ForwardBatch carry the SamplingConfig block, which grew from
    // 28 to 40 bytes when top_k + frequency/presence penalties were added
    // (#14). Their codes were bumped (0x02→0x04, 0x03→0x05) so a peer running
    // the old 28-byte layout trips the unknown-kind check in `parse_kind`
    // instead of mis-reading the larger frame. Reset/Token carry no sampling
    // and keep their codes.
    //
    // The single-token Forward family (Forward / ForwardNoSample / ForwardPrefill)
    // then grew the 1-byte `push_history` flag (issue #34), so those three codes
    // were bumped again (0x04→0x0C, 0x06→0x0D, 0x07→0x0E): the transport is a raw
    // byte stream with no message boundary, so an old peer reading the new body
    // would leave the extra byte in the stream and shift every later field by one.
    // The batch forwards did not gain the byte and keep their codes.
    Forward = 0x53_4D_45_0C, // "SME\x0C" — was 0x04 (no push_history byte)
    Reset = 0x53_4D_45_10,   // "SME\x10"
    Token = 0x53_4D_45_20,   // "SME\x20"
    ForwardBatch = 0x53_4D_45_05, // "SME\x05" — was 0x03; batched K-step verify
    TokenBatch = 0x53_4D_45_21, // "SME\x21" — batched K-step response
    /// Prefill-intermediate Forward (0x06): identical body to `Forward`,
    /// but the last rank must run its layers WITHOUT head/sample/record and
    /// reply with a dummy `Token(-1)`. Rank 0 discards intermediate prefill
    /// tokens anyway — but before this kind existed the last rank still
    /// *sampled* one token per prompt token and pushed each into its
    /// repetition/frequency/presence-penalty history, so the first real
    /// token was sampled against a history full of phantom prompt
    /// continuations (e.g. " Paris" is sampled after the prefix "The
    /// capital of France is" and recorded — then the default rep-penalty
    /// 1.05 demotes the real " Paris"). Single-stage `generate()` never
    /// records prompt-loop samples, so only distributed runs corrupted.
    /// Also skips the pointless vocab-width head GEMV per prompt token.
    ForwardNoSample = 0x53_4D_45_0D, // "SME\x0D" — was 0x06 (no push_history byte)
    /// Streamed prefill Forward (0x07): identical body to `ForwardNoSample`
    /// (the receiver advances KV but skips head/sample/record), except it is
    /// **one-way** — no `Token(-1)` ack. Rank 0 fires one per prompt token
    /// (except the last) WITHOUT blocking, and mid ranks relay downstream
    /// without waiting for a reply, so the prompt tokens pipeline through the
    /// ranks (per-rank compute overlaps) instead of one blocking 6-hop
    /// round-trip each. The final prompt token still goes as a sampling
    /// `Forward`, whose returned token is the first generated token.
    ForwardPrefill = 0x53_4D_45_0E, // "SME\x0E" — was 0x07 (no push_history byte)
    /// Batched prefill Forward (0x08): identical body to `ForwardBatch`, but the
    /// last rank runs its layers over ALL rows (batch-union) and samples ONLY
    /// the final row — the first generated token — pushing just that one into
    /// its penalty history (the earlier prompt rows must NOT be recorded, same
    /// reason as `ForwardNoSample`). Replies with a single `Token`, not a
    /// `TokenBatch`. Lets rank 0 push the whole prompt in one frame while the
    /// MoE dedups overlapping experts across the prompt's positions.
    ForwardBatchPrefill = 0x53_4D_45_08, // "SME\x08"
    /// Distributed KV-prefix cache (glm5). One-way, relayed downstream; body is
    /// an 8 B big-endian u64 prefix key. `RestorePrefix`: restore the cached KV
    /// slice for the key before the suffix prefill. `CachePrefix`: snapshot the
    /// current KV slice under the key after prefill. Runners without prefix
    /// caching (dsv4) treat both as no-ops.
    RestorePrefix = 0x53_4D_45_09, // "SME\x09"
    CachePrefix = 0x53_4D_45_0A, // "SME\x0A"
    /// Intermediate batched prefill (0x0B): identical body to `ForwardBatchPrefill`,
    /// but the last rank advances KV over ALL rows and samples NOTHING — it replies
    /// with a dummy `Token(-1)` ack and records no penalty history. Lets rank 0
    /// prefill a prompt longer than `MAX_BATCH_COUNT` as a sequence of ≤256-row
    /// windows: every window but the last is `NoSample`, and only the final
    /// `ForwardBatchPrefill` samples the first generated token.
    ForwardBatchPrefillNoSample = 0x53_4D_45_0B, // "SME\x0B"
    // Issue-34 Task 1.3 (multi-stage KV capture, §8). Appended codes only — never reorder existing
    // ones; an older peer that lacks these rejects them in `from_code` (loud, not silent corruption).
    Capture = 0x53_4D_45_30, // "SME\x30" — head→down: snapshot each stage's KV under epoch E
    CaptureAck = 0x53_4D_45_31, // "SME\x31" — up: "captured @ E" (propagates head-ward like Token)
    // Issue-34 consume path (§8). Appended codes only — never reorder existing ones. RESTORE(E) flows
    // downstream like Capture; each rank restores its slice captured under E from its own stash.
    // RestoreAck(E, verdict) flows upstream like CaptureAck, carrying the all-or-nothing verdict
    // (1 ⇒ that rank AND its whole downstream restored). An older peer lacking these rejects them in
    // `from_code` (loud, not silent corruption).
    Restore = 0x53_4D_45_40, // "SME\x40" — head→down: restore each stage's KV captured under epoch E
    RestoreAck = 0x53_4D_45_41, // "SME\x41" — up: verdict byte (1 = restored, 0 = missed ⇒ head cold)
    // Issue-34 cross-chain: like Restore but CARRIES the pulled slice blob inline (epoch + len + blob).
    // The moved-to head sends this to a moved-to tail that has NO local capture for a FOREIGN chain's
    // epoch; the tail applies the carried blob directly. Appended code — never reorder existing ones.
    RestoreCarry = 0x53_4D_45_42, // "SME\x42" — head→down: RESTORE carrying the pulled slice blob
    // H.1a close: CAPTURE carrying the head's turn tenant, so a worker rank — which never sees the
    // GenerationTask — can tag its stashed slice and `export` can confine it to that tenant. Sent
    // only when the tenant is non-empty; an empty tenant keeps sending the legacy `Capture` frame
    // byte-identical to today. Appended code — never reorder existing ones.
    CaptureV2 = 0x53_4D_4532, // "SME\x32" — head→down: CAPTURE(epoch, tenant, tokens)
    // Inkling expert-parallel dispatch (star topology: one driver, W expert workers; see
    // `crate::inkling::ep`). Appended codes — never reorder existing ones. The driver sends one
    // `ExpertDispatch` per involved worker per MoE layer and awaits one `ExpertResult` from each;
    // a worker is stateless and replies `ExpertResult{status 1}` to any other kind.
    ExpertDispatch = 0x53_4D_45_50, // "SME\x50" — driver → worker: layer, rows, k, hidden, ids
    /// Version 1 weighted fused-shard request; result uses ExpertResult with k=1.
    FusedExpertDispatch = 0x53_4D_45_52,
    ExpertResult = 0x53_4D_45_51, // "SME\x51" — worker → driver: status + outputs | message
    // Multi-stream pipeline decode (continuous batching across ranks; see
    // `PipelineEngine::step_streams`). Appended codes — never reorder existing
    // ones. Every rank keeps one KV slot per stream; rank 0 picks the slot ids.
    /// down: `batch_id u32 | slot u32 | rows u32 | 40 B SamplingConfig | tensor [1, rows, H]` —
    /// open `slot` on every rank and prefill its prompt rows; the last rank
    /// samples the final row with a fresh per-slot sampler and replies
    /// `StreamTokens{batch_id, [(slot, token)]}`.
    StreamOpen = 0x53_4D_45_60, // "SME\x60"
    /// down: `batch_id u32 | rows u32 | (slot u32, pos u32) × rows | tensor [1, rows, H]` — one
    /// decode token per listed stream; the last rank samples each row with that
    /// slot's sampler and replies `StreamTokens`.
    StreamDecode = 0x53_4D_45_61, // "SME\x61"
    /// down, one-way: `slot u32` — release the slot on every rank.
    StreamClose = 0x53_4D_45_62, // "SME\x62"
    /// up: `batch_id u32 | rows u32 | (slot u32, token i64) × rows`.
    StreamTokens = 0x53_4D_45_63, // "SME\x63"
    /// down: `batch_id u32 | slot u32 | rows u32 | flags u32 | SamplingConfig | tensor [1, rows, H]`
    /// — one window of a prompt longer than a `StreamOpen` may carry
    /// ([`MAX_STREAM_ROWS`]). [`STREAM_FEED_OPEN`] on the first window opens
    /// the slot, the rows append at the slot's position, and only the window
    /// with [`STREAM_FEED_FINAL`] is sampled and answered with `StreamTokens`.
    /// Windows travel the pipeline back to back, so rank 1 works on the first
    /// while rank 0 computes the second.
    StreamFeed = 0x534D_4564, // "SME\x64"
    /// up: `token i64 | tag u32 | status u32` — the reply to a sampling
    /// frame that carried a structured-output mask (see [`MaskCols`]). Only
    /// stages that apply masks send it; `tag` echoes the masked row's tag and
    /// `status` is [`MASK_ACK`] (token `-1` = this step could not be served)
    /// or [`MASK_UNCONFIRMED`] (a stage further down did not acknowledge).
    TokenAck = 0x53_4D_45_22, // "SME\x22"
    /// up: `batch_id u32 | status u32 | rows u32 | (slot u32, token i64) × rows`
    /// — the reply to a stream frame that carried masks.
    StreamTokensAck = 0x53_4D_45_65, // "SME\x65"
}

impl FrameKind {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            x if x == FrameKind::Forward as u32 => Some(FrameKind::Forward),
            x if x == FrameKind::Reset as u32 => Some(FrameKind::Reset),
            x if x == FrameKind::Token as u32 => Some(FrameKind::Token),
            x if x == FrameKind::ForwardBatch as u32 => Some(FrameKind::ForwardBatch),
            x if x == FrameKind::TokenBatch as u32 => Some(FrameKind::TokenBatch),
            x if x == FrameKind::ForwardNoSample as u32 => Some(FrameKind::ForwardNoSample),
            x if x == FrameKind::ForwardPrefill as u32 => Some(FrameKind::ForwardPrefill),
            x if x == FrameKind::ForwardBatchPrefill as u32 => Some(FrameKind::ForwardBatchPrefill),
            x if x == FrameKind::ForwardBatchPrefillNoSample as u32 => {
                Some(FrameKind::ForwardBatchPrefillNoSample)
            }
            x if x == FrameKind::RestorePrefix as u32 => Some(FrameKind::RestorePrefix),
            x if x == FrameKind::CachePrefix as u32 => Some(FrameKind::CachePrefix),
            x if x == FrameKind::Capture as u32 => Some(FrameKind::Capture),
            x if x == FrameKind::CaptureAck as u32 => Some(FrameKind::CaptureAck),
            x if x == FrameKind::Restore as u32 => Some(FrameKind::Restore),
            x if x == FrameKind::RestoreAck as u32 => Some(FrameKind::RestoreAck),
            x if x == FrameKind::RestoreCarry as u32 => Some(FrameKind::RestoreCarry),
            x if x == FrameKind::CaptureV2 as u32 => Some(FrameKind::CaptureV2),
            x if x == FrameKind::ExpertDispatch as u32 => Some(FrameKind::ExpertDispatch),
            x if x == FrameKind::FusedExpertDispatch as u32 => Some(FrameKind::FusedExpertDispatch),
            x if x == FrameKind::ExpertResult as u32 => Some(FrameKind::ExpertResult),
            x if x == FrameKind::StreamOpen as u32 => Some(FrameKind::StreamOpen),
            x if x == FrameKind::StreamDecode as u32 => Some(FrameKind::StreamDecode),
            x if x == FrameKind::StreamClose as u32 => Some(FrameKind::StreamClose),
            x if x == FrameKind::StreamTokens as u32 => Some(FrameKind::StreamTokens),
            x if x == FrameKind::StreamFeed as u32 => Some(FrameKind::StreamFeed),
            x if x == FrameKind::TokenAck as u32 => Some(FrameKind::TokenAck),
            x if x == FrameKind::StreamTokensAck as u32 => Some(FrameKind::StreamTokensAck),
            _ => None,
        }
    }
}

/// Read one frame-kind code from the wire. Returns `None` on a clean
/// socket close (peer closed during recv on no in-flight frame), which
/// callers should treat as end-of-task rather than an error.
pub async fn recv_kind_server(srv: &Mutex<ActivationServer>) -> TransportResult<Option<FrameKind>> {
    let raw = {
        let mut guard = srv.lock().await;
        guard.recv_raw(4).await
    };
    match raw {
        Ok(bytes) => parse_kind(&bytes).map(Some),
        Err(TransportError::SocketClosed) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Read the next reply frame's kind from downstream, first reading and
/// dropping any late structured-output reply (`TokenAck`,
/// `StreamTokensAck`): a masked step that timed out keeps its socket, and
/// its reply may land in front of any later exchange (token waits, KV
/// capture/restore acks, batched replies). The waits that own such replies
/// use [`recv_kind_client_raw`].
pub async fn recv_kind_client(cli: &Mutex<ActivationClient>) -> TransportResult<Option<FrameKind>> {
    loop {
        match recv_kind_client_raw(cli).await? {
            Some(FrameKind::TokenAck) => {
                recv_token_ack_body_client(cli).await?;
            }
            Some(FrameKind::StreamTokensAck) => {
                recv_stream_tokens_ack_body_client(cli).await?;
            }
            other => return Ok(other),
        }
    }
}

/// [`recv_kind_client`] without dropping late acknowledged replies.
pub async fn recv_kind_client_raw(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<Option<FrameKind>> {
    let raw = {
        let mut guard = cli.lock().await;
        guard.recv_raw(4).await
    };
    match raw {
        Ok(bytes) => parse_kind(&bytes).map(Some),
        Err(TransportError::SocketClosed) => Ok(None),
        Err(e) => Err(e),
    }
}

fn parse_kind(bytes: &[u8]) -> TransportResult<FrameKind> {
    // `recv_raw(4)` always returns exactly 4 bytes on success.
    debug_assert_eq!(bytes.len(), 4);
    let code = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    FrameKind::from_code(code).ok_or_else(|| {
        TransportError::Io(std::io::Error::other(format!(
            "unknown sparse-MoE frame kind 0x{code:08x}; peer may be on an older Forward layout"
        )))
    })
}

/// Encode hidden state f32 values + shape into a wire tensor.
fn hidden_to_tensor(hidden_f32: &[f32], shape3: [u32; 3]) -> Tensor {
    let mut bytes = Vec::with_capacity(hidden_f32.len() * 4);
    for v in hidden_f32 {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::new(DType::F32, shape3, bytes)
}

fn tensor_to_hidden(t: &Tensor) -> TransportResult<(Vec<f32>, [u32; 3])> {
    if t.dtype != DType::F32 {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "expected F32 hidden tensor, got {:?}",
            t.dtype
        ))));
    }
    let n = t.data.len() / 4;
    let mut out = Vec::with_capacity(n);
    for c in t.data.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((out, t.shape))
}

/// On-wire encoding of `SamplingConfig`. Fixed 40 bytes BE; the `seed`
/// field uses 0 as the sentinel for "no seed" (the engine treats 0 as
/// invalid anyway — `init_rng` clamps to `max(seed, 1)`). Bytes 28..40 were
/// added in #14 (top_k + frequency/presence penalties); the Forward frame
/// kind was bumped alongside so a 28-byte peer fails fast.
pub const SAMPLING_WIRE_BYTES: usize = 40;

pub fn encode_sampling(cfg: &SamplingConfig, out: &mut [u8; SAMPLING_WIRE_BYTES]) {
    out[0..4].copy_from_slice(&cfg.temperature.to_be_bytes());
    out[4..8].copy_from_slice(&cfg.top_p.to_be_bytes());
    out[8..12].copy_from_slice(&cfg.repetition_penalty.to_be_bytes());
    out[12..20].copy_from_slice(&(cfg.repetition_window as u64).to_be_bytes());
    let seed = cfg.seed.unwrap_or(0);
    out[20..28].copy_from_slice(&seed.to_be_bytes());
    out[28..32].copy_from_slice(&cfg.top_k.to_be_bytes());
    out[32..36].copy_from_slice(&cfg.frequency_penalty.to_be_bytes());
    out[36..40].copy_from_slice(&cfg.presence_penalty.to_be_bytes());
}

pub fn decode_sampling(bytes: &[u8; SAMPLING_WIRE_BYTES]) -> SamplingConfig {
    // Defensive: a malformed or out-of-version peer could send NaN /
    // negative / out-of-range values that would silently poison the
    // sampler (NaN temperature → 1/NaN → all logits NaN → argmax 0
    // every step). Clamp each f32 field to its valid domain and drop
    // NaN to the default.
    let temperature = sanitize_f32(
        f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0.0,
        0.0,
    );
    let top_p = sanitize_f32(
        f32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        1.0,
        0.0,
    )
    .min(1.0);
    // repetition_penalty is a divisor in the sampler (`logit / α`), so a wire
    // value of 0 from a corrupt/out-of-version peer would map a positive logit
    // to +inf. Clamp anything <= 0 (and NaN) back to 1.0 (no-op) — the min is
    // MIN_POSITIVE, not 0.0, so exactly-zero falls back too.
    let repetition_penalty = sanitize_f32(
        f32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        1.0,
        f32::MIN_POSITIVE,
    );
    let rep_window = u64::from_be_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]) as usize;
    let seed_raw = u64::from_be_bytes([
        bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
    ]);
    let top_k = u32::from_be_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
    // Penalties may legitimately be negative (OpenAI range -2.0..=2.0), so
    // only guard NaN here rather than clamping to a min like the others.
    let frequency_penalty = sanitize_f32(
        f32::from_be_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]),
        0.0,
        f32::NEG_INFINITY,
    );
    let presence_penalty = sanitize_f32(
        f32::from_be_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]),
        0.0,
        f32::NEG_INFINITY,
    );
    SamplingConfig {
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        repetition_window: rep_window,
        seed: if seed_raw == 0 { None } else { Some(seed_raw) },
    }
}

/// Replace NaN / -inf / values below `min` with `fallback`. Used by
/// `decode_sampling` to keep wire input from poisoning the sampler.
fn sanitize_f32(x: f32, fallback: f32, min: f32) -> f32 {
    if x.is_nan() || x < min {
        fallback
    } else {
        x
    }
}

/// Send a Forward-shaped frame downstream: kind + past_seq_len (u32 BE) +
/// SamplingConfig block + 1 B `push_history` flag + hidden tensor. Shared by
/// [`send_forward`], [`send_forward_nosample`] and [`send_forward_prefill`] —
/// the kinds carry identical bodies.
///
/// `push_history` tells the last rank whether this forward's sampled token is a
/// *kept* generated token (true ⇒ include it in the rep-penalty history) vs a
/// discarded prefill sample (false). It lets the multi-stage rep-penalty window
/// cover generated tokens only — matching the single-stage path — and is what
/// makes a warm-resumed run byte-identical to a cold one (the skipped prefill
/// forwards would otherwise desync the history). The layout is extended
/// unconditionally (no version byte); mixed builds are covered by the frame-kind
/// bump (0x04→0x0C etc.) — an old peer rejects the new code at `parse_kind`
/// instead of leaving the extra byte in the raw stream.
async fn send_forward_kind(
    cli: &Mutex<ActivationClient>,
    kind: FrameKind,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
    push_history: bool,
) -> TransportResult<()> {
    let mut header = [0u8; 8 + SAMPLING_WIRE_BYTES + 1];
    header[0..4].copy_from_slice(&(kind as u32).to_be_bytes());
    header[4..8].copy_from_slice(&past_seq_len.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[8..8 + SAMPLING_WIRE_BYTES].copy_from_slice(&sbytes);
    header[8 + SAMPLING_WIRE_BYTES] = u8::from(push_history);
    let tensor = hidden_to_tensor(hidden_f32, hidden_shape);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Send a sampling Forward frame downstream: kind + past_seq_len (u32 BE) +
/// 28 B SamplingConfig + 1 B `push_history` + hidden tensor.
pub async fn send_forward(
    cli: &Mutex<ActivationClient>,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
    push_history: bool,
) -> TransportResult<()> {
    send_forward_kind(
        cli,
        FrameKind::Forward,
        past_seq_len,
        sampling,
        hidden_f32,
        hidden_shape,
        push_history,
    )
    .await
}

/// Send a prefill-intermediate Forward ([`FrameKind::ForwardNoSample`]):
/// identical body to [`send_forward`], but the last rank runs its layers
/// without head/sample/record and replies with a dummy `Token(-1)`. Rank 0
/// sends this for every prompt token except the last, so intermediate
/// prefill samples never pollute the last rank's penalty history.
pub async fn send_forward_nosample(
    cli: &Mutex<ActivationClient>,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    send_forward_kind(
        cli,
        FrameKind::ForwardNoSample,
        past_seq_len,
        sampling,
        hidden_f32,
        hidden_shape,
        // the sample is discarded, so it never enters the rep-penalty history
        false,
    )
    .await
}

/// Send a STREAMED prefill Forward ([`FrameKind::ForwardPrefill`]): identical
/// body to [`send_forward_nosample`], but one-way — the receiver advances KV
/// and does NOT reply. Rank 0 fires these back-to-back for the prompt tokens
/// (except the last) so they pipeline through the ranks.
pub async fn send_forward_prefill(
    cli: &Mutex<ActivationClient>,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    send_forward_kind(
        cli,
        FrameKind::ForwardPrefill,
        past_seq_len,
        sampling,
        hidden_f32,
        hidden_shape,
        // the sample is discarded, so it never enters the rep-penalty history
        false,
    )
    .await
}

/// Receive a Forward frame's body (the kind code has already been
/// consumed by `recv_kind_*`). Returns
/// `(past_seq_len, sampling, push_history, hidden_f32, shape)`.
pub async fn recv_forward_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, SamplingConfig, bool, Vec<f32>, [u32; 3])> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(4 + SAMPLING_WIRE_BYTES + 1).await?;
    if raw.len() != 4 + SAMPLING_WIRE_BYTES + 1 {
        return Err(TransportError::SocketClosed);
    }
    let past_seq_len = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[4..4 + SAMPLING_WIRE_BYTES]);
    let sampling = decode_sampling(&sbytes);
    let push_history = raw[4 + SAMPLING_WIRE_BYTES] != 0;
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    let (h, shape) = tensor_to_hidden(&tensor)?;
    Ok((past_seq_len, sampling, push_history, h, shape))
}

/// Send a Reset frame downstream — clears KV state for a new task.
pub async fn send_reset(cli: &Mutex<ActivationClient>) -> TransportResult<()> {
    let mut header = [0u8; 4];
    header[0..4].copy_from_slice(&(FrameKind::Reset as u32).to_be_bytes());
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    Ok(())
}

async fn send_key_frame(
    cli: &Mutex<ActivationClient>,
    kind: FrameKind,
    key: u64,
) -> TransportResult<()> {
    let mut bytes = [0u8; 12];
    bytes[0..4].copy_from_slice(&(kind as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&key.to_be_bytes());
    let mut guard = cli.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Send a RestorePrefix frame downstream (one-way): the receiver restores its
/// cached KV slice for `key` before the suffix prefill, then relays.
pub async fn send_restore_prefix(cli: &Mutex<ActivationClient>, key: u64) -> TransportResult<()> {
    send_key_frame(cli, FrameKind::RestorePrefix, key).await
}

/// Send a CachePrefix frame downstream (one-way): the receiver snapshots its
/// current KV slice under `key` after prefill, then relays.
pub async fn send_cache_prefix(cli: &Mutex<ActivationClient>, key: u64) -> TransportResult<()> {
    send_key_frame(cli, FrameKind::CachePrefix, key).await
}

/// Receive an 8 B big-endian u64 key body from upstream (kind already consumed).
pub async fn recv_key_body_server(srv: &Mutex<ActivationServer>) -> TransportResult<u64> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(8).await?;
    drop(guard);
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    Ok(u64::from_be_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

/// Send a Token frame upstream (last rank → rank 0 path).
/// Body: 8 B big-endian i64 token id.
pub async fn send_token_upstream(
    srv: &Mutex<ActivationServer>,
    token_id: i64,
) -> TransportResult<()> {
    let mut bytes = [0u8; 12];
    bytes[0..4].copy_from_slice(&(FrameKind::Token as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&token_id.to_be_bytes());
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a Token frame's body (kind already consumed). Returns the
/// sampled token id.
pub async fn recv_token_body_client(cli: &Mutex<ActivationClient>) -> TransportResult<i64> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(8).await?;
    drop(guard);
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    Ok(i64::from_be_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

/// Await the single `Token` reply owed to a `Forward` we just sent `down`,
/// bounded by `deadline`.
///
/// Design rule: a reply to in-flight work uses a strict deadline, never the
/// idle ceiling. The underlying [`recv_kind_client`] is idle-tolerant — it
/// waits up to the ~900 s frame-idle ceiling for the next frame to *start* so
/// legitimately idle relays are not killed. That tolerance is exactly wrong
/// here: the frame is already owed. A downstream that dies mid-request (killed
/// peer, black-holed socket with no FIN/RST) would otherwise pin this recv on
/// the idle ceiling — the task never finalizes, and every upstream rank plus
/// the driving API request wedges behind it instead of surfacing a fast error.
/// On timeout this drops the connection and returns `Err`; later sends fail
/// fast with `NotConnected` until a supervisor restarts the chain. The caller
/// picks `deadline` for the frame it just sent; for dsv4's token-at-a-time
/// forwarding that is a single per-token budget
/// ([`cascadia_transport::recv_timeout`]) whether the token is prefill or
/// decode, since each reply carries the same single-token downstream compute.
///
/// Dropping the socket is not optional. Timing out does not cancel the work
/// downstream: the peer is usually alive and still computing, and its `Token`
/// lands on the socket after we stopped waiting. The body is eight raw bytes
/// with no sequence number, so a reused connection hands that reply to the
/// NEXT request — every later token off by one frame, and coherent enough that
/// nothing looks wrong. Where the stale reply is an intermediate window's `-1`
/// ack it is worse than wrong output: rank 0 embeds it as `u32::MAX` and
/// panics out of bounds.
///
/// This is the same hazard [`cascadia_transport::ActivationServer`] poisons its
/// own connection for on a failed reply recv; the deadline here made this the
/// one reply-wait that kept its socket.
///
/// A structured-output step's late acknowledged reply (`TokenAck` /
/// `StreamTokensAck`, see [`recv_token_reply_masked`]) is read and dropped
/// first: no unmasked step is ever answered with one.
pub async fn recv_token_reply(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
) -> Result<i64, String> {
    recv_token_reply_with(down, deadline, None).await
}

/// [`recv_token_reply`] that clears rank 0's `latch` when the late reply it
/// drops was acknowledged (the stage behind is slow, not old).
pub async fn recv_token_reply_with(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
    latch: Option<&MaskLatch>,
) -> Result<i64, String> {
    match tokio::time::timeout(deadline, async {
        loop {
            match recv_kind_client_raw(down).await {
                Ok(Some(FrameKind::Token)) => {
                    return recv_token_body_client(down)
                        .await
                        .map_err(|e| format!("recv_token: {e}"))
                }
                Ok(Some(FrameKind::TokenAck)) => {
                    let (_, _, status) = recv_token_ack_body_client(down)
                        .await
                        .map_err(|e| format!("recv_token_ack: {e}"))?;
                    note_stale_ack(latch, status);
                }
                Ok(Some(FrameKind::StreamTokensAck)) => {
                    let (_, status, _) = recv_stream_tokens_ack_body_client(down)
                        .await
                        .map_err(|e| format!("recv_stream_tokens_ack: {e}"))?;
                    note_stale_ack(latch, status);
                }
                Ok(Some(other)) => return Err(format!("expected Token reply, got {other:?}")),
                Ok(None) => return Err("downstream closed before Token".into()),
                Err(e) => return Err(format!("recv_kind: {e}")),
            }
        }
    })
    .await
    {
        Ok(res) => res,
        Err(_) => {
            // The inner future — and any guard it held mid-read — is dropped by
            // the time we get here, so re-locking cannot deadlock.
            down.lock().await.close().await;
            Err(format!(
                "reply timeout after {deadline:?}: downstream silent (dead peer?); \
                 connection dropped to avoid reading this reply as the next request's"
            ))
        }
    }
}

// ───────────────────────── Issue-34 Task 1.3: multi-stage KV capture (§8) ─────────────────────────
//
// CAPTURE(epoch, tokens) flows downstream like Forward; each stage snapshots its KV slice under the
// head-assigned `epoch` (so workers never derive it locally, §8) and keys it by `tokens` (so its
// served Manifest carries the prefix the consumer re-validates). CaptureAck(epoch) flows upstream like
// Token; a mid-stage acks up only after its downstream acked, so the head's single ack = "all captured".

/// Hard cap on a Capture frame's token count — bounds the recv-side allocation from the 4-byte count
/// field against a corrupt/adversarial peer (mirrors `MAX_BATCH_COUNT`). 1 Mi ids = 4 MiB worst case.
pub const MAX_CAPTURE_TOKENS: u32 = 1 << 20;

/// Cap on a CaptureV2 frame's tenant byte length — bounds the recv-side allocation the same way
/// `MAX_CAPTURE_TOKENS` does, and no legitimate tenant id approaches it.
pub const MAX_CAPTURE_TENANT_BYTES: u16 = 256;

/// Bound on every CaptureAck wait. An older peer that meets `CaptureV2` errors WITHOUT acking
/// (`from_code` → None), and the only ceiling otherwise is the transport's frame-idle default —
/// reached BEFORE the turn's chunks are returned, so an unbounded wait withholds a finished turn.
/// Capture is best-effort: on timeout the sender warns, skips the capture, and delivers the turn.
pub const CAPTURE_ACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Pure frame encoder (testable): kind(4) + epoch(8 BE) + count(4 BE) + count×i32 (BE).
fn capture_frame_bytes(epoch: u64, tokens: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(16 + tokens.len() * 4);
    b.extend_from_slice(&(FrameKind::Capture as u32).to_be_bytes());
    b.extend_from_slice(&epoch.to_be_bytes());
    b.extend_from_slice(&(tokens.len() as u32).to_be_bytes());
    for &t in tokens {
        b.extend_from_slice(&t.to_be_bytes());
    }
    b
}

/// Pure V2 frame encoder (testable): kind(4) + epoch(8 BE) + tenant_len(2 BE) + tenant UTF-8 +
/// count(4 BE) + count×i32 (BE). Only ever built with a non-empty tenant (empty stays v1).
fn capture_frame_bytes_v2(epoch: u64, tokens: &[i32], tenant: &str) -> Vec<u8> {
    let t = tenant.as_bytes();
    debug_assert!(!t.is_empty() && t.len() <= MAX_CAPTURE_TENANT_BYTES as usize);
    let mut b = Vec::with_capacity(18 + t.len() + tokens.len() * 4);
    b.extend_from_slice(&(FrameKind::CaptureV2 as u32).to_be_bytes());
    b.extend_from_slice(&epoch.to_be_bytes());
    b.extend_from_slice(&(t.len() as u16).to_be_bytes());
    b.extend_from_slice(t);
    b.extend_from_slice(&(tokens.len() as u32).to_be_bytes());
    for &tok in tokens {
        b.extend_from_slice(&tok.to_be_bytes());
    }
    b
}

/// Send a Capture frame downstream (head/mid → next stage). A non-empty `tenant` upgrades the
/// frame to `CaptureV2` so the downstream rank — which never sees the `GenerationTask` — can tag
/// its stashed slice; an empty tenant sends the legacy `Capture` frame byte-identical to today.
pub async fn send_capture(
    cli: &Mutex<ActivationClient>,
    epoch: u64,
    tokens: &[i32],
    tenant: &str,
) -> TransportResult<()> {
    // An oversized tenant must FAIL the capture, never silently fall back to the untagged v1
    // frame: untagged ("") stashes are deliberately servable to ANY partner, so the fallback
    // would convert a length overrun into a cross-tenant read with every guard passing. The
    // caller treats the error as "capture skipped" (best-effort; the turn is still delivered).
    if tenant.len() > MAX_CAPTURE_TENANT_BYTES as usize {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "capture tenant of {} bytes exceeds MAX_CAPTURE_TENANT_BYTES {MAX_CAPTURE_TENANT_BYTES}; \
             refusing the untagged-v1 downgrade",
            tenant.len()
        ))));
    }
    let bytes = if tenant.is_empty() {
        capture_frame_bytes(epoch, tokens)
    } else {
        capture_frame_bytes_v2(epoch, tokens, tenant)
    };
    let mut guard = cli.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a Capture body from upstream (kind already consumed). Returns `(epoch, tokens)`.
pub async fn recv_capture_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u64, Vec<i32>)> {
    let mut guard = srv.lock().await;
    let head = guard.recv_raw(12).await?;
    if head.len() != 12 {
        return Err(TransportError::SocketClosed);
    }
    let epoch = u64::from_be_bytes(head[0..8].try_into().unwrap());
    let count = u32::from_be_bytes(head[8..12].try_into().unwrap());
    if count > MAX_CAPTURE_TOKENS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "capture token count {count} exceeds MAX_CAPTURE_TOKENS {MAX_CAPTURE_TOKENS}"
        ))));
    }
    let n = count as usize;
    // Chunked: n*4 can exceed the transport's single-read cap (see recv_exact_chunked).
    let body = recv_exact_chunked(&mut guard, n * 4).await?;
    drop(guard);
    let mut tokens = Vec::with_capacity(n);
    for c in body.chunks_exact(4) {
        tokens.push(i32::from_be_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((epoch, tokens))
}

/// Receive a CaptureV2 body from upstream (kind already consumed). Returns `(epoch, tokens, tenant)`.
pub async fn recv_capture_v2_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u64, Vec<i32>, String)> {
    let mut guard = srv.lock().await;
    let head = guard.recv_raw(10).await?;
    if head.len() != 10 {
        return Err(TransportError::SocketClosed);
    }
    let epoch = u64::from_be_bytes(head[0..8].try_into().unwrap());
    let t_len = u16::from_be_bytes(head[8..10].try_into().unwrap());
    if t_len == 0 || t_len > MAX_CAPTURE_TENANT_BYTES {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "capture tenant length {t_len} outside (0, {MAX_CAPTURE_TENANT_BYTES}]"
        ))));
    }
    let t_raw = guard.recv_raw(t_len as usize).await?;
    if t_raw.len() != t_len as usize {
        return Err(TransportError::SocketClosed);
    }
    let tenant = String::from_utf8(t_raw)
        .map_err(|_| TransportError::Io(std::io::Error::other("capture tenant not UTF-8")))?;
    let cnt_raw = guard.recv_raw(4).await?;
    if cnt_raw.len() != 4 {
        return Err(TransportError::SocketClosed);
    }
    let count = u32::from_be_bytes(cnt_raw[0..4].try_into().unwrap());
    if count > MAX_CAPTURE_TOKENS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "capture token count {count} exceeds MAX_CAPTURE_TOKENS {MAX_CAPTURE_TOKENS}"
        ))));
    }
    let n = count as usize;
    // Chunked: n*4 can exceed the transport's single-read cap (see recv_exact_chunked).
    let body = recv_exact_chunked(&mut guard, n * 4).await?;
    drop(guard);
    let mut tokens = Vec::with_capacity(n);
    for c in body.chunks_exact(4) {
        tokens.push(i32::from_be_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((epoch, tokens, tenant))
}

/// Read exactly `total` bytes as ≤[`MAX_RAW_BYTES`] chunks. `recv_raw` refuses any single read
/// over the transport cap BEFORE consuming a byte, so one over-cap read of an in-band frame body
/// leaves the body in the stream and desyncs every later frame (token payload parsed as frame
/// kinds). RESTORE's carried blob already reads chunked; CAPTURE's token body was missed — its
/// ceiling was ≈16,381 tokens, and `tokens = prompt ++ generated` grows past that in any long
/// session. `total` must already be validated against its own cap by the caller.
async fn recv_exact_chunked(
    guard: &mut ActivationServer,
    total: usize,
) -> TransportResult<Vec<u8>> {
    let mut out = Vec::with_capacity(total);
    let mut remaining = total;
    while remaining > 0 {
        let take = remaining.min(MAX_RAW_BYTES);
        let chunk = guard.recv_raw(take).await?;
        if chunk.len() != take {
            return Err(TransportError::SocketClosed);
        }
        out.extend_from_slice(&chunk);
        remaining -= take;
    }
    Ok(out)
}

/// Send a CaptureAck frame upstream (stage → head-ward). Body: 8 B BE epoch.
pub async fn send_capture_ack_upstream(
    srv: &Mutex<ActivationServer>,
    epoch: u64,
) -> TransportResult<()> {
    let mut bytes = [0u8; 12];
    bytes[0..4].copy_from_slice(&(FrameKind::CaptureAck as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&epoch.to_be_bytes());
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a CaptureAck body from downstream (kind already consumed). Returns the acked epoch.
pub async fn recv_capture_ack_body_client(cli: &Mutex<ActivationClient>) -> TransportResult<u64> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(8).await?;
    drop(guard);
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    Ok(u64::from_be_bytes(raw[0..8].try_into().unwrap()))
}

// ───────────────────────── Issue-34 consume path: multi-stage KV RESTORE (§8) ─────────────────────────
//
// RESTORE(epoch) flows downstream like Capture but carries NO token body — the epoch alone keys each
// rank's existing CAPTURE stash. RestoreAck(epoch, verdict) flows upstream like CaptureAck; a mid-stage
// folds its downstream verdict into its own (local && down) so the head's single verdict==1 means every
// stage restored. Any miss anywhere ⇒ verdict 0 ⇒ the head discards and cold-runs (all-or-nothing).

/// Pure frame encoder (testable): kind(4) + epoch(8 BE). No token body (unlike Capture).
fn restore_frame_bytes(epoch: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(12);
    b.extend_from_slice(&(FrameKind::Restore as u32).to_be_bytes());
    b.extend_from_slice(&epoch.to_be_bytes());
    b
}

/// Pure ack encoder (testable): kind(4) + epoch(8 BE) + verdict(1).
fn restore_ack_bytes(epoch: u64, verdict: u8) -> [u8; 13] {
    let mut bytes = [0u8; 13];
    bytes[0..4].copy_from_slice(&(FrameKind::RestoreAck as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&epoch.to_be_bytes());
    bytes[12] = verdict;
    bytes
}

/// Send a Restore frame downstream (head/mid → next stage).
pub async fn send_restore(cli: &Mutex<ActivationClient>, epoch: u64) -> TransportResult<()> {
    let bytes = restore_frame_bytes(epoch);
    let mut guard = cli.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a Restore body from upstream (kind already consumed). Returns the epoch to restore under.
pub async fn recv_restore_body_server(srv: &Mutex<ActivationServer>) -> TransportResult<u64> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(8).await?;
    drop(guard);
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    Ok(u64::from_be_bytes(raw[0..8].try_into().unwrap()))
}

/// Hard cap on a RestoreCarry blob — bounds the recv-side allocation from the 4-byte length field
/// against a corrupt/adversarial peer (mirrors `MAX_CAPTURE_TOKENS`). The opaque f32 KV slice is
/// ~tens of MiB for a small model; 512 MiB is a generous upper bound.
pub const MAX_RESTORE_BLOB_BYTES: u32 = 512 << 20;

/// Pure RestoreCarry encoder (testable): kind(4) + epoch(8 BE) + blob_len(4 BE) + blob.
fn restore_carry_bytes(epoch: u64, blob: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(16 + blob.len());
    b.extend_from_slice(&(FrameKind::RestoreCarry as u32).to_be_bytes());
    b.extend_from_slice(&epoch.to_be_bytes());
    b.extend_from_slice(&(blob.len() as u32).to_be_bytes());
    b.extend_from_slice(blob);
    b
}

/// Send a RestoreCarry frame downstream (head → next stage), carrying the pulled slice blob inline.
pub async fn send_restore_carry(
    cli: &Mutex<ActivationClient>,
    epoch: u64,
    blob: &[u8],
) -> TransportResult<()> {
    let bytes = restore_carry_bytes(epoch, blob);
    let mut guard = cli.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a RestoreCarry body from upstream (kind already consumed). Returns `(epoch, blob)`.
///
/// The KV blob routinely exceeds `recv_raw`'s per-read cap ([`MAX_RAW_BYTES`], 64 KiB) — a real
/// model's slice is many MiB — so read it in `MAX_RAW_BYTES` chunks. ALWAYS consume exactly `len`
/// bytes off the wire (accept OR drain), so a rejected/oversized frame never leaves the framed pipeline
/// stream desynced. A blob over [`MAX_RESTORE_BLOB_BYTES`] is drained in-chunks (bounded allocation)
/// and rejected. The sender writes the blob contiguously (`send_raw` is uncapped), so the wire format
/// is unchanged — only the read is chunked.
pub async fn recv_restore_carry_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u64, Vec<u8>)> {
    let mut guard = srv.lock().await;
    let head = guard.recv_raw(12).await?;
    if head.len() != 12 {
        return Err(TransportError::SocketClosed);
    }
    let epoch = u64::from_be_bytes(head[0..8].try_into().unwrap());
    let len = u32::from_be_bytes(head[8..12].try_into().unwrap()) as usize;
    // Over the accept ceiling ⇒ drain (discard) rather than assemble, but still consume every byte.
    let accept = len <= MAX_RESTORE_BLOB_BYTES as usize;
    let mut blob = Vec::new();
    let mut remaining = len;
    while remaining > 0 {
        let take = remaining.min(MAX_RAW_BYTES);
        let chunk = guard.recv_raw(take).await?;
        if chunk.len() != take {
            return Err(TransportError::SocketClosed);
        }
        if accept {
            blob.extend_from_slice(&chunk);
        }
        remaining -= take;
    }
    drop(guard);
    if !accept {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "restore-carry blob {len} exceeds MAX_RESTORE_BLOB_BYTES {MAX_RESTORE_BLOB_BYTES}"
        ))));
    }
    Ok((epoch, blob))
}

/// Send a RestoreAck frame upstream (stage → head-ward). Body: kind(4) + epoch(8 BE) + verdict(1).
pub async fn send_restore_ack_upstream(
    srv: &Mutex<ActivationServer>,
    epoch: u64,
    verdict: u8,
) -> TransportResult<()> {
    let bytes = restore_ack_bytes(epoch, verdict);
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a RestoreAck body from downstream (kind already consumed). Returns `(epoch, verdict)`.
pub async fn recv_restore_ack_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<(u64, u8)> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(9).await?;
    drop(guard);
    if raw.len() != 9 {
        return Err(TransportError::SocketClosed);
    }
    let epoch = u64::from_be_bytes(raw[0..8].try_into().unwrap());
    Ok((epoch, raw[8]))
}

/// Bound on every RestoreAck wait. Same design rule as [`recv_token_reply`]: a reply to in-flight
/// work uses a strict deadline, never the ~900 s frame-idle ceiling — RESTORE runs at admission
/// (`step_first`), so an unbounded wait stalls every warm-hit request behind a silent downstream,
/// and an older peer that errors on the frame never acks at all. Matches ov-runtime's
/// `RESTORE_ACK_TIMEOUT`.
pub const RESTORE_ACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Await the RestoreAck verdict for a RESTORE/RESTORE_CARRY just sent on `down`, bounded by
/// [`RESTORE_ACK_TIMEOUT`]. On timeout the connection is dropped — same hazard
/// [`recv_token_reply`] documents: the late ack (13 raw bytes, no sequence number) would otherwise
/// be read as the next exchange's reply on a reused connection.
pub async fn recv_restore_verdict(down: &Mutex<ActivationClient>) -> Result<bool, String> {
    match tokio::time::timeout(RESTORE_ACK_TIMEOUT, async {
        match recv_kind_client(down).await {
            Ok(Some(FrameKind::RestoreAck)) => recv_restore_ack_body_client(down)
                .await
                .map(|(_, v)| v == 1)
                .map_err(|e| format!("recv_restore_ack: {e}")),
            Ok(Some(other)) => Err(format!("expected RestoreAck, got {other:?}")),
            Ok(None) => Err("downstream closed during restore-ack".into()),
            Err(e) => Err(format!("recv_kind (restore-ack): {e}")),
        }
    })
    .await
    {
        Ok(res) => res,
        Err(_) => {
            // The inner future — and any guard it held mid-read — is dropped by
            // the time we get here, so re-locking cannot deadlock.
            down.lock().await.close().await;
            Err(format!(
                "restore ack timeout after {RESTORE_ACK_TIMEOUT:?}: downstream silent (dead peer?); \
                 connection dropped to avoid reading the late ack as the next exchange's reply"
            ))
        }
    }
}

#[cfg(test)]
mod capture_frame_tests {
    use super::*;

    #[test]
    fn capture_frame_bytes_layout_roundtrips() {
        let toks = vec![1, -2, 300, i32::MIN, i32::MAX, 0];
        let epoch = 0xDEAD_BEEF_0000_0001u64;
        let b = capture_frame_bytes(epoch, &toks);
        assert_eq!(b[0..4], (FrameKind::Capture as u32).to_be_bytes());
        assert_eq!(u64::from_be_bytes(b[4..12].try_into().unwrap()), epoch);
        assert_eq!(
            u32::from_be_bytes(b[12..16].try_into().unwrap()),
            toks.len() as u32
        );
        let got: Vec<i32> = b[16..]
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, toks);
        assert_eq!(b.len(), 16 + toks.len() * 4);
    }

    #[test]
    fn capture_v2_frame_bytes_layout_roundtrips() {
        let toks = vec![7, -8, 0, i32::MAX];
        let epoch = 0xCAFE_F00D_0000_0002u64;
        let tenant = "tenant-a";
        let b = capture_frame_bytes_v2(epoch, &toks, tenant);
        assert_eq!(b[0..4], (FrameKind::CaptureV2 as u32).to_be_bytes());
        assert_eq!(u64::from_be_bytes(b[4..12].try_into().unwrap()), epoch);
        let t_len = u16::from_be_bytes(b[12..14].try_into().unwrap()) as usize;
        assert_eq!(t_len, tenant.len());
        assert_eq!(&b[14..14 + t_len], tenant.as_bytes());
        let c0 = 14 + t_len;
        assert_eq!(
            u32::from_be_bytes(b[c0..c0 + 4].try_into().unwrap()),
            toks.len() as u32
        );
        let got: Vec<i32> = b[c0 + 4..]
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, toks);
        assert_eq!(b.len(), 18 + t_len + toks.len() * 4);
    }

    #[test]
    fn capture_v2_code_is_appended_and_distinct() {
        // The legacy v1 frame must stay byte-identical (an empty tenant keeps sending it), and the
        // v2 code must collide with nothing existing.
        assert_eq!(FrameKind::CaptureV2 as u32, 0x53_4D_4532);
        for k in [
            FrameKind::Forward,
            FrameKind::Reset,
            FrameKind::Token,
            FrameKind::ForwardBatch,
            FrameKind::TokenBatch,
            FrameKind::ForwardNoSample,
            FrameKind::ForwardPrefill,
            FrameKind::Capture,
            FrameKind::CaptureAck,
            FrameKind::Restore,
            FrameKind::RestoreAck,
            FrameKind::RestoreCarry,
        ] {
            assert_ne!(FrameKind::CaptureV2 as u32, k as u32);
        }
        assert_eq!(
            FrameKind::from_code(FrameKind::CaptureV2 as u32),
            Some(FrameKind::CaptureV2)
        );
    }

    #[test]
    fn restore_carry_bytes_layout_roundtrips() {
        let blob = vec![0u8, 1, 2, 255, 128, 7, 42];
        let epoch = 0x0102_0304_0506_0708u64;
        let b = restore_carry_bytes(epoch, &blob);
        assert_eq!(b[0..4], (FrameKind::RestoreCarry as u32).to_be_bytes());
        assert_eq!(u64::from_be_bytes(b[4..12].try_into().unwrap()), epoch);
        assert_eq!(
            u32::from_be_bytes(b[12..16].try_into().unwrap()),
            blob.len() as u32
        );
        assert_eq!(&b[16..], &blob[..]);
        assert_eq!(b.len(), 16 + blob.len());
        // Empty blob: header only, len field 0.
        let e = restore_carry_bytes(epoch, &[]);
        assert_eq!(u32::from_be_bytes(e[12..16].try_into().unwrap()), 0);
        assert_eq!(e.len(), 16);
        // Appended code, distinct from Restore/RestoreAck.
        assert_ne!(FrameKind::RestoreCarry as u32, FrameKind::Restore as u32);
        assert_eq!(
            FrameKind::from_code(FrameKind::RestoreCarry as u32),
            Some(FrameKind::RestoreCarry)
        );
    }

    #[test]
    fn capture_frame_bytes_empty_tokens() {
        let b = capture_frame_bytes(7, &[]);
        assert_eq!(b.len(), 16);
        assert_eq!(u32::from_be_bytes(b[12..16].try_into().unwrap()), 0);
    }

    #[test]
    fn restore_frame_bytes_layout() {
        let epoch = 0xDEAD_BEEF_0000_0001u64;
        let b = restore_frame_bytes(epoch);
        assert_eq!(b.len(), 12);
        assert_eq!(b[0..4], (FrameKind::Restore as u32).to_be_bytes());
        assert_eq!(u64::from_be_bytes(b[4..12].try_into().unwrap()), epoch);
    }

    #[test]
    fn restore_ack_bytes_verdict_roundtrips() {
        let epoch = 0x0102_0304_0506_0708u64;
        for verdict in [0u8, 1u8] {
            let b = restore_ack_bytes(epoch, verdict);
            assert_eq!(b.len(), 13);
            assert_eq!(b[0..4], (FrameKind::RestoreAck as u32).to_be_bytes());
            // Decode exactly as `recv_restore_ack_body_client` does (after the 4-byte kind is consumed).
            assert_eq!(u64::from_be_bytes(b[4..12].try_into().unwrap()), epoch);
            assert_eq!(b[12], verdict);
        }
    }

    #[test]
    fn restore_and_ack_codes_are_distinct_and_recognized() {
        // Appended codes must round-trip through `from_code` and not collide with existing kinds.
        assert_eq!(
            FrameKind::from_code(FrameKind::Restore as u32),
            Some(FrameKind::Restore)
        );
        assert_eq!(
            FrameKind::from_code(FrameKind::RestoreAck as u32),
            Some(FrameKind::RestoreAck)
        );
        assert_ne!(FrameKind::Restore as u32, FrameKind::Capture as u32);
        assert_ne!(FrameKind::RestoreAck as u32, FrameKind::CaptureAck as u32);
    }
}

/// Hard cap on the number of hidden positions one ForwardBatch /
/// TokenBatch frame can carry. Caps the worst-case allocation on the
/// receiving side so an adversarial / corrupt peer cannot ask us to
/// allocate megabytes of token-buffer or hidden-state buffer from a
/// 4-byte count field. 256 is well above any sensible spec-decode K
/// (the reference implementation uses K=8; even K=64 is extreme).
pub const MAX_BATCH_COUNT: u32 = 256;

/// Send a ForwardBatch frame downstream: kind + past_seq_len_start
/// (u32 BE) + batch_count (u32 BE) + 28 B SamplingConfig + hidden tensor.
///
/// `hidden_f32` must be `[1, batch_count, hidden_size]` row-major. The
/// receiver runs `batch_count` sequential single-token forwards
/// internally, each occupying KV slot `past_seq_len_start + i`.
pub async fn send_forward_batch(
    cli: &Mutex<ActivationClient>,
    past_seq_len_start: u32,
    batch_count: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    send_forward_batch_kind(
        FrameKind::ForwardBatch,
        cli,
        past_seq_len_start,
        batch_count,
        sampling,
        hidden_f32,
        hidden_shape,
    )
    .await
}

/// Like [`send_forward_batch`], but tags the frame [`FrameKind::ForwardBatchPrefill`]
/// so the last rank samples only the final row (prompt prefill, not spec-verify).
pub async fn send_forward_batch_prefill(
    cli: &Mutex<ActivationClient>,
    past_seq_len_start: u32,
    batch_count: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    send_forward_batch_kind(
        FrameKind::ForwardBatchPrefill,
        cli,
        past_seq_len_start,
        batch_count,
        sampling,
        hidden_f32,
        hidden_shape,
    )
    .await
}

/// Like [`send_forward_batch_prefill`], but tags the frame
/// [`FrameKind::ForwardBatchPrefillNoSample`] so the last rank advances KV over
/// the rows without sampling or recording — an intermediate window of a prompt
/// longer than [`MAX_BATCH_COUNT`]. Replies with a dummy `Token(-1)` ack.
pub async fn send_forward_batch_prefill_nosample(
    cli: &Mutex<ActivationClient>,
    past_seq_len_start: u32,
    batch_count: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    send_forward_batch_kind(
        FrameKind::ForwardBatchPrefillNoSample,
        cli,
        past_seq_len_start,
        batch_count,
        sampling,
        hidden_f32,
        hidden_shape,
    )
    .await
}

async fn send_forward_batch_kind(
    kind: FrameKind,
    cli: &Mutex<ActivationClient>,
    past_seq_len_start: u32,
    batch_count: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    if batch_count == 0 || batch_count > MAX_BATCH_COUNT {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_forward_batch: batch_count {batch_count} out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    if hidden_shape[1] != batch_count {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_forward_batch: shape[1]={} does not match batch_count={batch_count}",
            hidden_shape[1]
        ))));
    }
    let expected =
        (hidden_shape[0] as usize) * (hidden_shape[1] as usize) * (hidden_shape[2] as usize);
    if hidden_f32.len() != expected {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_forward_batch: hidden.len={} != prod(shape)={expected}",
            hidden_f32.len()
        ))));
    }
    let mut header = [0u8; 12 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(kind as u32).to_be_bytes());
    header[4..8].copy_from_slice(&past_seq_len_start.to_be_bytes());
    header[8..12].copy_from_slice(&batch_count.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[12..12 + SAMPLING_WIRE_BYTES].copy_from_slice(&sbytes);
    let tensor = hidden_to_tensor(hidden_f32, hidden_shape);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Receive a ForwardBatch frame's body (kind already consumed). Returns
/// `(past_seq_len_start, batch_count, sampling, hidden_f32, shape)`.
///
/// The receiver is expected to run `batch_count` sequential forwards
/// internally, each pulling the corresponding `[hidden_size]` row out of
/// the returned `hidden_f32` buffer.
pub async fn recv_forward_batch_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, u32, SamplingConfig, Vec<f32>, [u32; 3])> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(8 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 8 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let past_seq_len_start = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let batch_count = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);
    if batch_count == 0 || batch_count > MAX_BATCH_COUNT {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_forward_batch_body: batch_count {batch_count} out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[8..8 + SAMPLING_WIRE_BYTES]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != batch_count {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_forward_batch_body: tensor shape[1]={} != batch_count={batch_count}",
            tensor.shape[1]
        ))));
    }
    let (h, shape) = tensor_to_hidden(&tensor)?;
    Ok((past_seq_len_start, batch_count, sampling, h, shape))
}

/// Send a TokenBatch frame upstream: K sampled token ids in one frame.
pub async fn send_token_batch_upstream(
    srv: &Mutex<ActivationServer>,
    tokens: &[i64],
) -> TransportResult<()> {
    let n = tokens.len();
    if n == 0 || n > MAX_BATCH_COUNT as usize {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_token_batch_upstream: tokens.len={n} out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    let mut bytes = Vec::with_capacity(8 + n * 8);
    bytes.extend_from_slice(&(FrameKind::TokenBatch as u32).to_be_bytes());
    bytes.extend_from_slice(&(n as u32).to_be_bytes());
    for &t in tokens {
        bytes.extend_from_slice(&t.to_be_bytes());
    }
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a TokenBatch frame's body (kind already consumed). Returns
/// the K sampled token ids.
pub async fn recv_token_batch_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<Vec<i64>> {
    let mut guard = cli.lock().await;
    let header = guard.recv_raw(4).await?;
    if header.len() != 4 {
        return Err(TransportError::SocketClosed);
    }
    let n = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if n == 0 || n > MAX_BATCH_COUNT {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_token_batch_body: batch_count {n} out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    let raw = guard.recv_raw((n as usize) * 8).await?;
    drop(guard);
    if raw.len() != (n as usize) * 8 {
        return Err(TransportError::SocketClosed);
    }
    let mut tokens = Vec::with_capacity(n as usize);
    for i in 0..(n as usize) {
        let off = i * 8;
        tokens.push(i64::from_be_bytes([
            raw[off],
            raw[off + 1],
            raw[off + 2],
            raw[off + 3],
            raw[off + 4],
            raw[off + 5],
            raw[off + 6],
            raw[off + 7],
        ]));
    }
    Ok(tokens)
}

// ───────────────────────── multi-stream pipeline frames ─────────────────────────

/// Rows one stream frame may carry (streams per decode step, or prompt rows
/// of one prefill).
pub const MAX_STREAM_ROWS: u32 = MAX_BATCH_COUNT;

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// down: open `slot` with `rows` prompt positions (`hidden` = `[rows, h]`).
pub async fn send_stream_open(
    cli: &Mutex<ActivationClient>,
    batch_id: u32,
    slot: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    rows: u32,
    h: u32,
) -> TransportResult<()> {
    let mut header = [0u8; 16 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(FrameKind::StreamOpen as u32).to_be_bytes());
    header[4..8].copy_from_slice(&batch_id.to_be_bytes());
    header[8..12].copy_from_slice(&slot.to_be_bytes());
    header[12..16].copy_from_slice(&rows.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[16..].copy_from_slice(&sbytes);
    let tensor = hidden_to_tensor(hidden_f32, [1, rows, h]);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Body of a `StreamOpen` (kind consumed): `(batch_id, slot, rows, sampling, hidden)`.
pub async fn recv_stream_open_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, u32, u32, SamplingConfig, Vec<f32>)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(12 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 12 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, slot, rows) = (be_u32(&raw[0..4]), be_u32(&raw[4..8]), be_u32(&raw[8..12]));
    if rows == 0 || rows > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream open: rows {rows} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[12..]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != rows {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream open: tensor shape[1]={} != rows {rows}",
            tensor.shape[1]
        ))));
    }
    let (h, _) = tensor_to_hidden(&tensor)?;
    Ok((batch_id, slot, rows, sampling, h))
}

/// `StreamFeed` flag: this is the prompt's first window, open the slot.
pub const STREAM_FEED_OPEN: u32 = 1;
/// `StreamFeed` flag: this is the prompt's last window, sample its final row
/// and reply `StreamTokens`.
pub const STREAM_FEED_FINAL: u32 = 2;

/// down: `rows` more prompt positions of `slot` (see [`FrameKind::StreamFeed`]).
#[allow(clippy::too_many_arguments)]
pub async fn send_stream_feed(
    cli: &Mutex<ActivationClient>,
    batch_id: u32,
    slot: u32,
    flags: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    rows: u32,
    h: u32,
) -> TransportResult<()> {
    let mut header = [0u8; 20 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(FrameKind::StreamFeed as u32).to_be_bytes());
    header[4..8].copy_from_slice(&batch_id.to_be_bytes());
    header[8..12].copy_from_slice(&slot.to_be_bytes());
    header[12..16].copy_from_slice(&rows.to_be_bytes());
    header[16..20].copy_from_slice(&flags.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[20..].copy_from_slice(&sbytes);
    let tensor = hidden_to_tensor(hidden_f32, [1, rows, h]);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Body of a `StreamFeed` (kind consumed): `(batch_id, slot, rows, flags, sampling, hidden)`.
pub async fn recv_stream_feed_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, u32, u32, u32, SamplingConfig, Vec<f32>)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(16 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 16 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, slot, rows, flags) = (
        be_u32(&raw[0..4]),
        be_u32(&raw[4..8]),
        be_u32(&raw[8..12]),
        be_u32(&raw[12..16]),
    );
    if rows == 0 || rows > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream feed: rows {rows} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[16..]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != rows {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream feed: tensor shape[1]={} != rows {rows}",
            tensor.shape[1]
        ))));
    }
    let (h, _) = tensor_to_hidden(&tensor)?;
    Ok((batch_id, slot, rows, flags, sampling, h))
}

/// down: one decode token for each `(slot, pos)` (`hidden` = `[rows.len(), h]`).
pub async fn send_stream_decode(
    cli: &Mutex<ActivationClient>,
    batch_id: u32,
    rows: &[(u32, u32)],
    hidden_f32: &[f32],
    h: u32,
) -> TransportResult<()> {
    let n = rows.len() as u32;
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream decode: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let mut header = Vec::with_capacity(12 + rows.len() * 8);
    header.extend_from_slice(&(FrameKind::StreamDecode as u32).to_be_bytes());
    header.extend_from_slice(&batch_id.to_be_bytes());
    header.extend_from_slice(&n.to_be_bytes());
    for &(slot, pos) in rows {
        header.extend_from_slice(&slot.to_be_bytes());
        header.extend_from_slice(&pos.to_be_bytes());
    }
    let tensor = hidden_to_tensor(hidden_f32, [1, n, h]);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Body of a `StreamDecode` (kind consumed): `(batch_id, [(slot, pos)], hidden)`.
pub async fn recv_stream_decode_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, Vec<(u32, u32)>, Vec<f32>)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(8).await?;
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, n) = (be_u32(&raw[0..4]), be_u32(&raw[4..8]));
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream decode: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let raw = guard.recv_raw(n as usize * 8).await?;
    if raw.len() != n as usize * 8 {
        return Err(TransportError::SocketClosed);
    }
    let rows: Vec<(u32, u32)> = raw
        .chunks_exact(8)
        .map(|c| (be_u32(&c[0..4]), be_u32(&c[4..8])))
        .collect();
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != n {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream decode: tensor shape[1]={} != rows {n}",
            tensor.shape[1]
        ))));
    }
    let (h, _) = tensor_to_hidden(&tensor)?;
    Ok((batch_id, rows, h))
}

/// down, one-way: release `slot` on every downstream rank.
pub async fn send_stream_close(cli: &Mutex<ActivationClient>, slot: u32) -> TransportResult<()> {
    let mut bytes = [0u8; 8];
    bytes[0..4].copy_from_slice(&(FrameKind::StreamClose as u32).to_be_bytes());
    bytes[4..8].copy_from_slice(&slot.to_be_bytes());
    let mut guard = cli.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Body of a `StreamClose` (kind consumed): the slot.
pub async fn recv_stream_close_body_server(srv: &Mutex<ActivationServer>) -> TransportResult<u32> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(4).await?;
    drop(guard);
    if raw.len() != 4 {
        return Err(TransportError::SocketClosed);
    }
    Ok(be_u32(&raw))
}

/// up: the sampled token of each `(slot, token)` for `batch_id`.
pub async fn send_stream_tokens_upstream(
    srv: &Mutex<ActivationServer>,
    batch_id: u32,
    tokens: &[(u32, i64)],
) -> TransportResult<()> {
    let n = tokens.len() as u32;
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream tokens: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let mut bytes = Vec::with_capacity(12 + tokens.len() * 12);
    bytes.extend_from_slice(&(FrameKind::StreamTokens as u32).to_be_bytes());
    bytes.extend_from_slice(&batch_id.to_be_bytes());
    bytes.extend_from_slice(&n.to_be_bytes());
    for &(slot, tok) in tokens {
        bytes.extend_from_slice(&slot.to_be_bytes());
        bytes.extend_from_slice(&tok.to_be_bytes());
    }
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Body of a `StreamTokens` (kind consumed): `(batch_id, [(slot, token)])`.
pub async fn recv_stream_tokens_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<(u32, Vec<(u32, i64)>)> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(8).await?;
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, n) = (be_u32(&raw[0..4]), be_u32(&raw[4..8]));
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream tokens: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let raw = guard.recv_raw(n as usize * 12).await?;
    drop(guard);
    if raw.len() != n as usize * 12 {
        return Err(TransportError::SocketClosed);
    }
    let toks = raw
        .chunks_exact(12)
        .map(|c| {
            (
                be_u32(&c[0..4]),
                i64::from_be_bytes([c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]]),
            )
        })
        .collect();
    Ok((batch_id, toks))
}

/// Await the `StreamTokens` reply owed to a stream frame just sent `down`,
/// bounded by `deadline` (same drop-on-timeout rule as [`recv_token_reply`]).
pub async fn recv_stream_tokens_reply(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
) -> Result<(u32, Vec<(u32, i64)>), String> {
    match tokio::time::timeout(deadline, async {
        match recv_kind_client(down).await {
            Ok(Some(FrameKind::StreamTokens)) => recv_stream_tokens_body_client(down)
                .await
                .map_err(|e| format!("recv_stream_tokens: {e}")),
            Ok(Some(other)) => Err(format!("expected StreamTokens reply, got {other:?}")),
            Ok(None) => Err("downstream closed before StreamTokens".into()),
            Err(e) => Err(format!("recv_kind: {e}")),
        }
    })
    .await
    {
        Ok(res) => res,
        Err(_) => {
            down.lock().await.close().await;
            Err(format!(
                "stream reply timeout after {deadline:?}: downstream silent (dead peer?); \
                 connection dropped"
            ))
        }
    }
}

/// Forward a Reset frame downstream — used by mid ranks after consuming
/// the kind code from upstream. The body is empty so this is just a
/// `send_reset` to the next peer; the `upstream` argument is held by the
/// caller and the kind is already consumed before this point.
pub async fn forward_reset(downstream: &Mutex<ActivationClient>) -> TransportResult<()> {
    send_reset(downstream).await
}

/// Bundle of the per-rank transport state. The Builder constructs this
/// during `connect()` and hands it to the Engine.
#[derive(Default)]
pub struct StageTransport {
    pub upstream: Option<Arc<Mutex<ActivationServer>>>,
    pub downstream: Option<Arc<Mutex<ActivationClient>>>,
}

impl StageTransport {
    pub fn is_first(&self) -> bool {
        self.upstream.is_none()
    }

    pub fn is_last(&self) -> bool {
        self.downstream.is_none()
    }
}

// ───────────────────── Inkling expert-parallel dispatch (`crate::inkling::ep`) ─────────────────────
//
// Star topology: the driver (rank 0 of a `total = 1` pipeline) routes every MoE layer locally and
// sends each involved expert worker the layer's hidden rows plus the expert ids that worker must
// serve; the worker replies with the RAW expert outputs (no weights applied — the driver applies
// the gate weights and sums in gate order, so the result is bit-identical to the single-process
// `MoeLayer`). Bodies, after the 4-byte kind:
//
//   ExpertDispatch:  u32 layer | u32 rows | u32 k        (big-endian, like every header field)
//                    Tensor F32 [rows, hidden, 1]         the MoE input rows
//                    Tensor I32 [rows, k, 1]              expert ids; EXPERT_PAD (-1) = padding slot
//   ExpertResult:    u8 status                            0 = ok, 1 = error
//                    ok:  Tensor F32 [rows, k, hidden]    E_id(h_row) per slot; zeros in pad slots
//                    err: u32 len | UTF-8 message         (len <= MAX_EXPERT_ERR_BYTES)
//
// Tensor payloads are little-endian element bytes (the transport's convention — see
// `hidden_to_tensor`). `rows` is capped at MAX_BATCH_COUNT per frame; the driver chunks longer
// prefills. A worker owns no sequence state, so no Reset / prefix frame is ever sent to it.

/// Padding expert id in an `ExpertDispatch` ids tensor: the worker skips the slot and returns
/// zeros for it; the driver never reads a pad slot back.
pub const EXPERT_PAD: i32 = -1;

/// Cap on an `ExpertResult` error message (bounds the recv-side allocation from the 4-byte length
/// field; must fit one `recv_raw`, i.e. <= [`MAX_RAW_BYTES`]).
pub const MAX_EXPERT_ERR_BYTES: usize = 4096;

/// A decoded `ExpertDispatch` body. Shapes are returned as carried on the wire; the worker
/// validates them against `rows` / `k` / its hidden size (a mismatch is a status-1 reply, not a
/// transport error — the frame has been consumed whole, so the stream stays aligned).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertDispatchBody {
    pub layer: u32,
    pub rows: u32,
    pub k: u32,
    /// `[rows · hidden]` row-major (`hidden` = `hidden_shape[1]`).
    pub hidden: Vec<f32>,
    pub hidden_shape: [u32; 3],
    /// `[rows · k]` row-major; [`EXPERT_PAD`] marks an unused slot.
    pub ids: Vec<i32>,
    pub ids_shape: [u32; 3],
}

impl ExpertDispatchBody {
    /// The hidden width this frame carries (`hidden_shape[1]`).
    pub fn hidden_size(&self) -> usize {
        self.hidden_shape[1] as usize
    }
}

fn ids_to_tensor(ids: &[i32], shape3: [u32; 3]) -> Tensor {
    let mut bytes = Vec::with_capacity(ids.len() * 4);
    for v in ids {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::new(DType::I32, shape3, bytes)
}

fn tensor_to_ids(t: &Tensor) -> TransportResult<(Vec<i32>, [u32; 3])> {
    if t.dtype != DType::I32 {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "expected I32 expert-id tensor, got {:?}",
            t.dtype
        ))));
    }
    let out = t
        .data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((out, t.shape))
}

/// Send an `ExpertDispatch` frame to one expert worker: kind + layer + rows + k (u32 BE each) +
/// the hidden rows `[rows, hidden, 1]` (F32) + the expert ids `[rows, k, 1]` (I32, [`EXPERT_PAD`]
/// in unused slots). `hidden` is `[rows · hidden_size]`, `ids` is `[rows · k]`, both row-major.
pub async fn send_expert_dispatch(
    cli: &Mutex<ActivationClient>,
    layer: u32,
    rows: u32,
    k: u32,
    hidden_size: u32,
    hidden: &[f32],
    ids: &[i32],
) -> TransportResult<()> {
    send_expert_dispatch_impl(cli, layer, rows, k, hidden_size, hidden, ids, None).await
}

/// Versioned fused request: the normal dispatch body plus original F32 gate
/// weights [rows,k,1]. The worker replies with one weighted partial per row.
#[allow(clippy::too_many_arguments)]
pub async fn send_fused_expert_dispatch(
    cli: &Mutex<ActivationClient>,
    layer: u32,
    rows: u32,
    k: u32,
    hidden_size: u32,
    hidden: &[f32],
    ids: &[i32],
    weights: &[f32],
) -> TransportResult<()> {
    send_expert_dispatch_impl(cli, layer, rows, k, hidden_size, hidden, ids, Some(weights)).await
}

#[allow(clippy::too_many_arguments)]
async fn send_expert_dispatch_impl(
    cli: &Mutex<ActivationClient>,
    layer: u32,
    rows: u32,
    k: u32,
    hidden_size: u32,
    hidden: &[f32],
    ids: &[i32],
    weights: Option<&[f32]>,
) -> TransportResult<()> {
    if weights.is_some_and(|w| w.len() != ids.len() || w.iter().any(|x| !x.is_finite())) {
        return Err(TransportError::Io(std::io::Error::other(
            "invalid fused routing weights",
        )));
    }
    if rows == 0 || rows > MAX_BATCH_COUNT {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_expert_dispatch: rows {rows} out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    if k == 0 {
        return Err(TransportError::Io(std::io::Error::other(
            "send_expert_dispatch: k must be >= 1 (an uninvolved worker gets no frame)",
        )));
    }
    if hidden.len() != rows as usize * hidden_size as usize {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_expert_dispatch: hidden.len={} != rows·hidden={}",
            hidden.len(),
            rows as usize * hidden_size as usize
        ))));
    }
    if ids.len() != rows as usize * k as usize {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_expert_dispatch: ids.len={} != rows·k={}",
            ids.len(),
            rows as usize * k as usize
        ))));
    }
    let mut header = [0u8; 16];
    let kind = if weights.is_some() {
        FrameKind::FusedExpertDispatch
    } else {
        FrameKind::ExpertDispatch
    };
    header[0..4].copy_from_slice(&(kind as u32).to_be_bytes());
    header[4..8].copy_from_slice(&layer.to_be_bytes());
    header[8..12].copy_from_slice(&rows.to_be_bytes());
    header[12..16].copy_from_slice(&k.to_be_bytes());
    let ht = hidden_to_tensor(hidden, [rows, hidden_size, 1]);
    let it = ids_to_tensor(ids, [rows, k, 1]);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&ht).await?;
    guard.send(&it).await?;
    if let Some(weights) = weights {
        guard.send(&hidden_to_tensor(weights, [rows, k, 1])).await?;
    }
    Ok(())
}

/// Receive an `ExpertDispatch` body on the worker (kind already consumed). Reads the whole frame
/// (header + both tensors) before returning so a semantic mismatch can be answered with a status-1
/// reply on an aligned stream; only dtype / transport failures are errors here. `rows` above
/// [`MAX_BATCH_COUNT`] is rejected AFTER the tensors are consumed (their own byte caps bound the
/// allocation).
pub async fn recv_expert_dispatch_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<ExpertDispatchBody> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(12).await?;
    if raw.len() != 12 {
        return Err(TransportError::SocketClosed);
    }
    let layer = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let rows = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let k = u32::from_be_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let (ht, _) = guard.recv().await?;
    let (it, _) = guard.recv().await?;
    drop(guard);
    if rows == 0 || rows > MAX_BATCH_COUNT {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_expert_dispatch_body: rows {rows} out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    let (hidden, hidden_shape) = tensor_to_hidden(&ht)?;
    let (ids, ids_shape) = tensor_to_ids(&it)?;
    Ok(ExpertDispatchBody {
        layer,
        rows,
        k,
        hidden,
        hidden_shape,
        ids,
        ids_shape,
    })
}

/// Receive the entire weighted request before semantic validation.
pub async fn recv_fused_expert_dispatch_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(ExpertDispatchBody, Vec<f32>, [u32; 3])> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(12).await?;
    if raw.len() != 12 {
        return Err(TransportError::SocketClosed);
    }
    let (ht, _) = guard.recv().await?;
    let (it, _) = guard.recv().await?;
    let (wt, _) = guard.recv().await?;
    drop(guard);
    // Drain all three tensors even if the header, IDs, or weights are
    // semantically invalid. Reusing the two-tensor decoder here would leave
    // weights unread when that decoder rejects an oversized row count.
    let layer = u32::from_be_bytes(raw[0..4].try_into().unwrap());
    let rows = u32::from_be_bytes(raw[4..8].try_into().unwrap());
    let k = u32::from_be_bytes(raw[8..12].try_into().unwrap());
    if rows == 0 || rows > MAX_BATCH_COUNT {
        return Err(TransportError::Io(std::io::Error::other(
            "fused dispatch row count out of range",
        )));
    }
    let (hidden, hidden_shape) = tensor_to_hidden(&ht)?;
    let (ids, ids_shape) = tensor_to_ids(&it)?;
    let (weights, shape) = tensor_to_hidden(&wt)?;
    Ok((
        ExpertDispatchBody {
            layer,
            rows,
            k,
            hidden,
            hidden_shape,
            ids,
            ids_shape,
        },
        weights,
        shape,
    ))
}

/// Reply `ExpertResult{status 0}` from the worker: kind + status byte + the outputs
/// `[rows, k, hidden]` (F32; `out` is `[rows · k · hidden]` row-major, zeros in pad slots).
pub async fn send_expert_result_ok(
    srv: &Mutex<ActivationServer>,
    rows: u32,
    k: u32,
    hidden_size: u32,
    out: &[f32],
) -> TransportResult<()> {
    let want = rows as usize * k as usize * hidden_size as usize;
    if out.len() != want {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_expert_result_ok: out.len={} != rows·k·hidden={want}",
            out.len()
        ))));
    }
    let mut header = [0u8; 5];
    header[0..4].copy_from_slice(&(FrameKind::ExpertResult as u32).to_be_bytes());
    header[4] = 0;
    let t = hidden_to_tensor(out, [rows, k, hidden_size]);
    let mut guard = srv.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&t).await?;
    Ok(())
}

/// Losslessly encode an expert reply as FP16 * 2^exponent (status 2), falling
/// back to the original F32 frame for any finite value f16 cannot represent
/// exactly (checked bit for bit, signed zero included, BEFORE anything is
/// sent). A nonfinite value, wrong length, or exponent > 8 is a hard error,
/// not a fallback. Older clients reject status 2; enable only after upgrading
/// all peers. Returns true when the compact frame was sent.
pub async fn send_expert_result_lossless(
    srv: &Mutex<ActivationServer>,
    rows: u32,
    k: u32,
    hidden_size: u32,
    out: &[f32],
    exponent: u8,
) -> TransportResult<bool> {
    let want = (rows as usize)
        .checked_mul(k as usize)
        .and_then(|n| n.checked_mul(hidden_size as usize));
    if want != Some(out.len()) || exponent > 8 || out.iter().any(|v| !v.is_finite()) {
        return Err(TransportError::Io(std::io::Error::other(
            "invalid lossless expert reply",
        )));
    }
    let scale = 2.0f32.powi(exponent as i32);
    let mut bytes = Vec::with_capacity(out.len() * 2);
    for &v in out {
        let half = half::f16::from_f32(v / scale);
        if (half.to_f32() * scale).to_bits() != v.to_bits() {
            send_expert_result_ok(srv, rows, k, hidden_size, out).await?;
            return Ok(false);
        }
        bytes.extend_from_slice(&half.to_le_bytes());
    }
    let mut header = [0u8; 6];
    header[..4].copy_from_slice(&(FrameKind::ExpertResult as u32).to_be_bytes());
    header[4] = 2;
    header[5] = exponent;
    let tensor = Tensor::new(DType::F16, [rows, k, hidden_size], bytes);
    let mut guard = srv.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(true)
}

/// Reply `ExpertResult{status 1}` from the worker: kind + status byte + u32 BE length + UTF-8
/// message (truncated to [`MAX_EXPERT_ERR_BYTES`] on a char boundary).
pub async fn send_expert_result_err(
    srv: &Mutex<ActivationServer>,
    msg: &str,
) -> TransportResult<()> {
    let mut end = msg.len().min(MAX_EXPERT_ERR_BYTES);
    while end > 0 && !msg.is_char_boundary(end) {
        end -= 1;
    }
    let m = &msg.as_bytes()[..end];
    let mut bytes = Vec::with_capacity(9 + m.len());
    bytes.extend_from_slice(&(FrameKind::ExpertResult as u32).to_be_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&(m.len() as u32).to_be_bytes());
    bytes.extend_from_slice(m);
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive an `ExpertResult` body on the driver (kind already consumed). `Ok(Ok((out, shape)))`
/// for status 0 — `out` is `[rows · k · hidden]`, `shape` is `[rows, k, hidden]` as carried;
/// `Ok(Err(message))` for status 1; `Err` only for transport / framing failures.
pub async fn recv_expert_result_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<Result<(Vec<f32>, [u32; 3]), String>> {
    let mut guard = cli.lock().await;
    let status = guard.recv_raw(1).await?;
    if status.len() != 1 {
        return Err(TransportError::SocketClosed);
    }
    match status[0] {
        0 => {
            let (t, _) = guard.recv().await?;
            drop(guard);
            let (out, shape) = tensor_to_hidden(&t)?;
            Ok(Ok((out, shape)))
        }
        2 => {
            let exponent = guard.recv_raw(1).await?;
            if exponent.len() != 1 {
                return Err(TransportError::SocketClosed);
            }
            // Consume the tensor before validating the new status payload, so
            // a malformed scale/dtype cannot leave the next frame misaligned.
            let (tensor, _) = guard.recv().await?;
            drop(guard);
            let size = tensor
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d as usize));
            if exponent[0] > 8
                || tensor.dtype != DType::F16
                || size.and_then(|n| n.checked_mul(2)) != Some(tensor.data.len())
            {
                return Err(TransportError::Io(std::io::Error::other(
                    "invalid scaled FP16 expert tensor",
                )));
            }
            let scale = 2.0f32.powi(exponent[0] as i32);
            let out: Vec<f32> = tensor
                .data
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32() * scale)
                .collect();
            if out.iter().any(|v| !v.is_finite()) {
                return Err(TransportError::Io(std::io::Error::other(
                    "nonfinite scaled FP16 expert tensor",
                )));
            }
            Ok(Ok((out, tensor.shape)))
        }
        1 => {
            let raw = guard.recv_raw(4).await?;
            if raw.len() != 4 {
                return Err(TransportError::SocketClosed);
            }
            let len = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            if len > MAX_EXPERT_ERR_BYTES {
                return Err(TransportError::Io(std::io::Error::other(format!(
                    "expert result error message of {len} bytes exceeds MAX_EXPERT_ERR_BYTES {MAX_EXPERT_ERR_BYTES}"
                ))));
            }
            let msg = if len == 0 {
                Vec::new()
            } else {
                let m = guard.recv_raw(len).await?;
                if m.len() != len {
                    return Err(TransportError::SocketClosed);
                }
                m
            };
            drop(guard);
            Ok(Err(String::from_utf8_lossy(&msg).into_owned()))
        }
        s => Err(TransportError::Io(std::io::Error::other(format!(
            "expert result: unknown status byte {s}"
        )))),
    }
}

#[cfg(test)]
mod expert_frame_tests {
    use super::*;

    #[test]
    fn expert_codes_are_appended_and_distinct() {
        assert_eq!(FrameKind::ExpertDispatch as u32, 0x53_4D_45_50);
        assert_eq!(FrameKind::ExpertResult as u32, 0x53_4D_45_51);
        for k in [
            FrameKind::Forward,
            FrameKind::Reset,
            FrameKind::Token,
            FrameKind::ForwardBatch,
            FrameKind::TokenBatch,
            FrameKind::ForwardNoSample,
            FrameKind::ForwardPrefill,
            FrameKind::ForwardBatchPrefill,
            FrameKind::ForwardBatchPrefillNoSample,
            FrameKind::RestorePrefix,
            FrameKind::CachePrefix,
            FrameKind::Capture,
            FrameKind::CaptureAck,
            FrameKind::Restore,
            FrameKind::RestoreAck,
            FrameKind::RestoreCarry,
            FrameKind::CaptureV2,
        ] {
            assert_ne!(FrameKind::ExpertDispatch as u32, k as u32);
            assert_ne!(FrameKind::ExpertResult as u32, k as u32);
        }
        assert_eq!(
            FrameKind::from_code(FrameKind::ExpertDispatch as u32),
            Some(FrameKind::ExpertDispatch)
        );
        assert_eq!(
            FrameKind::from_code(FrameKind::ExpertResult as u32),
            Some(FrameKind::ExpertResult)
        );
    }

    #[test]
    fn ids_tensor_roundtrips_including_pad() {
        let ids = [3i32, EXPERT_PAD, 0, i32::MAX, -7];
        let t = ids_to_tensor(&ids, [1, 5, 1]);
        assert_eq!(t.dtype, DType::I32);
        assert_eq!(t.data.len(), 20);
        let (back, shape) = tensor_to_ids(&t).unwrap();
        assert_eq!(back, ids);
        assert_eq!(shape, [1, 5, 1]);
        // An F32 tensor where ids are expected is a framing error, not a silent reinterpretation.
        let f = hidden_to_tensor(&[1.0, 2.0], [1, 2, 1]);
        assert!(tensor_to_ids(&f).is_err());
    }
}

// ───────────────────── structured-output masks on the stage wire ─────────────────────
//
// A json request's sampling steps (the last prompt row and every decode row) carry the
// grammar mask rank 0 computed, so the last stage, which samples but has no grammar, can
// apply it. Frame kinds and headers stay as they are; only the trailing tensor changes:
// I32 `[1, rows, H + C]` instead of F32 `[1, rows, H]`. Each row is the f32 bits of its
// hidden followed by C mask columns `[tag u32, mask words…]` (all little-endian, like
// every tensor payload). Tag 0 marks a row without a mask; bit `i` of the words allows
// token `i`, and tokens past the last word are not allowed. Every masked row of a frame
// has the same column count.
//
// A stage on an older build reads the whole tensor and then fails on its dtype in
// `tensor_to_hidden`, so the byte stream stays aligned. Such a stage never answers the
// step; a masked step is only answered by `TokenAck` / `StreamTokensAck`, kinds only
// stages on this build send. Requests without a grammar send exactly the old bytes: the
// `*_step` encoders hand an unmasked frame to the original encoder.

/// Acknowledgement status: every stage down the chain applied the mask.
pub const MASK_ACK: u32 = 0x4D41_534B; // "MASK"
/// Acknowledgement status from a middle stage: its downstream did not
/// acknowledge the mask (timed out, closed, or replied without an ack).
pub const MASK_UNCONFIRMED: u32 = 0x4E4F_4D4B; // "NOMK"
/// How long a link that could not mask stays refused before the next json
/// request probes it again.
pub const MASK_REPROBE: std::time::Duration = std::time::Duration::from_secs(600);

/// The mask columns of a masked frame: one row per tensor row,
/// `[tag, mask words…]` each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskCols {
    rows: usize,
    width: usize,
    data: Vec<u32>,
}

impl MaskCols {
    /// `rows` unmasked rows with room for `words` mask words each.
    pub fn new(rows: usize, words: usize) -> Self {
        let width = 1 + words;
        Self {
            rows,
            width,
            data: vec![0; rows * width],
        }
    }

    /// Mask `row` with `words` under the non-zero `tag`.
    pub fn set(&mut self, row: usize, tag: u32, words: &[u32]) {
        assert_ne!(tag, 0, "tag 0 marks an unmasked row");
        assert_eq!(words.len() + 1, self.width, "mask word count");
        let r = &mut self.data[row * self.width..(row + 1) * self.width];
        r[0] = tag;
        r[1..].copy_from_slice(words);
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns per row (tag + mask words).
    pub fn width(&self) -> usize {
        self.width
    }

    /// The row's tag; 0 for an unmasked row (or no columns at all).
    pub fn tag(&self, row: usize) -> u32 {
        if self.width == 0 || row >= self.rows {
            return 0;
        }
        self.data[row * self.width]
    }

    /// The row's mask words: `None` for an unmasked row, and also for a
    /// masked row that carries no word (a frame this stage can't serve).
    pub fn words(&self, row: usize) -> Option<&[u32]> {
        if self.tag(row) == 0 || self.width < 2 {
            return None;
        }
        Some(&self.data[row * self.width + 1..(row + 1) * self.width])
    }

    /// Whether any row is masked.
    pub fn any(&self) -> bool {
        (0..self.rows).any(|r| self.tag(r) != 0)
    }
}

/// `hidden` (`shape` = `[a, b, h]`) plus `cols` as one I32 tensor
/// `[a, b, h + cols.width()]`. Refused when it would exceed the transport's
/// tensor cap.
fn masked_tensor(hidden: &[f32], shape: [u32; 3], cols: &MaskCols) -> TransportResult<Tensor> {
    let rows = shape[0] as usize * shape[1] as usize;
    let h = shape[2] as usize;
    if hidden.len() != rows * h || cols.rows() != rows {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "masked frame: {} floats and {} mask rows for shape {shape:?}",
            hidden.len(),
            cols.rows()
        ))));
    }
    let total = h + cols.width();
    let bytes = rows
        .checked_mul(total)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= cascadia_transport::MAX_TENSOR_BYTES)
        .ok_or_else(|| {
            TransportError::Io(std::io::Error::other(format!(
                "masked frame of {rows} rows x {total} columns exceeds the tensor cap"
            )))
        })?;
    let mut data = Vec::with_capacity(bytes);
    for r in 0..rows {
        for v in &hidden[r * h..(r + 1) * h] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for w in &cols.data[r * cols.width()..(r + 1) * cols.width()] {
            data.extend_from_slice(&w.to_le_bytes());
        }
    }
    Ok(Tensor::new(
        DType::I32,
        [shape[0], shape[1], total as u32],
        data,
    ))
}

/// A step tensor's hidden rows, the shape a plain frame would carry, and
/// the mask columns of a masked frame.
pub type SplitStep = (Vec<f32>, [u32; 3], Option<MaskCols>);

/// The trailing tensor of a sampling-family frame as this build reads it.
#[derive(Debug)]
pub enum StepTensor {
    /// F32 hidden rows, as every older build sends.
    Plain(Vec<f32>, [u32; 3]),
    /// Hidden rows plus mask columns; [`StepTensor::split`] separates them.
    Masked(Tensor),
}

fn step_tensor(t: Tensor) -> TransportResult<StepTensor> {
    match t.dtype {
        DType::F32 => {
            let (h, shape) = tensor_to_hidden(&t)?;
            Ok(StepTensor::Plain(h, shape))
        }
        DType::I32 => Ok(StepTensor::Masked(t)),
        other => Err(TransportError::Io(std::io::Error::other(format!(
            "expected F32 hidden tensor, got {other:?}"
        )))),
    }
}

impl StepTensor {
    pub fn is_masked(&self) -> bool {
        matches!(self, StepTensor::Masked(_))
    }

    /// Rows of hidden width `h`, the shape a plain frame would have carried,
    /// and the mask columns of a masked frame. `Err` when a masked frame's
    /// rows are narrower than `h`.
    pub fn split(self, h: usize) -> Result<SplitStep, String> {
        let t = match self {
            StepTensor::Plain(hidden, shape) => return Ok((hidden, shape, None)),
            StepTensor::Masked(t) => t,
        };
        let [a, b, total] = t.shape;
        let total = total as usize;
        if total < h {
            return Err(format!(
                "masked frame rows are {total} wide, narrower than the hidden width {h}"
            ));
        }
        let rows = a as usize * b as usize;
        let width = total - h;
        let word = |i: usize| {
            let c = &t.data[i * 4..i * 4 + 4];
            u32::from_le_bytes([c[0], c[1], c[2], c[3]])
        };
        let mut hidden = Vec::with_capacity(rows * h);
        let mut data = Vec::with_capacity(rows * width);
        for r in 0..rows {
            let base = r * total;
            hidden.extend((base..base + h).map(|i| f32::from_bits(word(i))));
            data.extend((base + h..base + total).map(word));
        }
        Ok((
            hidden,
            [a, b, h as u32],
            Some(MaskCols { rows, width, data }),
        ))
    }
}

fn forward_header(
    kind: FrameKind,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    push_history: bool,
) -> [u8; 8 + SAMPLING_WIRE_BYTES + 1] {
    let mut header = [0u8; 8 + SAMPLING_WIRE_BYTES + 1];
    header[0..4].copy_from_slice(&(kind as u32).to_be_bytes());
    header[4..8].copy_from_slice(&past_seq_len.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[8..8 + SAMPLING_WIRE_BYTES].copy_from_slice(&sbytes);
    header[8 + SAMPLING_WIRE_BYTES] = u8::from(push_history);
    header
}

async fn send_header_and_tensor(
    cli: &Mutex<ActivationClient>,
    header: &[u8],
    tensor: &Tensor,
) -> TransportResult<()> {
    let mut guard = cli.lock().await;
    guard.send_raw(header).await?;
    guard.send(tensor).await?;
    Ok(())
}

/// A single-row Forward-family frame (`Forward`, `ForwardNoSample`,
/// `ForwardPrefill`); `cols: None` sends exactly what [`send_forward`] and
/// its siblings send.
#[allow(clippy::too_many_arguments)]
pub async fn send_forward_step(
    cli: &Mutex<ActivationClient>,
    kind: FrameKind,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
    push_history: bool,
    cols: Option<&MaskCols>,
) -> TransportResult<()> {
    let Some(cols) = cols else {
        return send_forward_kind(
            cli,
            kind,
            past_seq_len,
            sampling,
            hidden_f32,
            hidden_shape,
            push_history,
        )
        .await;
    };
    let tensor = masked_tensor(hidden_f32, hidden_shape, cols)?;
    let header = forward_header(kind, past_seq_len, sampling, push_history);
    send_header_and_tensor(cli, &header, &tensor).await
}

/// Body of a Forward-family frame (kind consumed), masked or not:
/// `(past_seq_len, sampling, push_history, tensor)`. Reads the whole frame
/// before returning.
pub async fn recv_forward_step_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, SamplingConfig, bool, StepTensor)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(4 + SAMPLING_WIRE_BYTES + 1).await?;
    if raw.len() != 4 + SAMPLING_WIRE_BYTES + 1 {
        return Err(TransportError::SocketClosed);
    }
    let past_seq_len = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[4..4 + SAMPLING_WIRE_BYTES]);
    let sampling = decode_sampling(&sbytes);
    let push_history = raw[4 + SAMPLING_WIRE_BYTES] != 0;
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    Ok((past_seq_len, sampling, push_history, step_tensor(tensor)?))
}

/// A batched forward (`ForwardBatch`, `ForwardBatchPrefill`,
/// `ForwardBatchPrefillNoSample`); `cols: None` sends exactly what
/// [`send_forward_batch_prefill`] and its siblings send.
#[allow(clippy::too_many_arguments)]
pub async fn send_forward_batch_step(
    kind: FrameKind,
    cli: &Mutex<ActivationClient>,
    past_seq_len_start: u32,
    batch_count: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
    cols: Option<&MaskCols>,
) -> TransportResult<()> {
    let Some(cols) = cols else {
        return send_forward_batch_kind(
            kind,
            cli,
            past_seq_len_start,
            batch_count,
            sampling,
            hidden_f32,
            hidden_shape,
        )
        .await;
    };
    if batch_count == 0 || batch_count > MAX_BATCH_COUNT || hidden_shape[1] != batch_count {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_forward_batch: batch_count {batch_count} (shape {hidden_shape:?}) out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    let tensor = masked_tensor(hidden_f32, hidden_shape, cols)?;
    let mut header = [0u8; 12 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(kind as u32).to_be_bytes());
    header[4..8].copy_from_slice(&past_seq_len_start.to_be_bytes());
    header[8..12].copy_from_slice(&batch_count.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[12..12 + SAMPLING_WIRE_BYTES].copy_from_slice(&sbytes);
    send_header_and_tensor(cli, &header, &tensor).await
}

/// Body of a batched forward (kind consumed), masked or not:
/// `(past_seq_len_start, batch_count, sampling, tensor)`.
pub async fn recv_forward_batch_step_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, u32, SamplingConfig, StepTensor)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(8 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 8 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let past_seq_len_start = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let batch_count = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);
    if batch_count == 0 || batch_count > MAX_BATCH_COUNT {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_forward_batch_body: batch_count {batch_count} out of range 1..={MAX_BATCH_COUNT}"
        ))));
    }
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[8..8 + SAMPLING_WIRE_BYTES]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != batch_count {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_forward_batch_body: tensor shape[1]={} != batch_count={batch_count}",
            tensor.shape[1]
        ))));
    }
    Ok((
        past_seq_len_start,
        batch_count,
        sampling,
        step_tensor(tensor)?,
    ))
}

fn stream_rows_tensor(
    hidden_f32: &[f32],
    rows: u32,
    h: u32,
    cols: Option<&MaskCols>,
) -> TransportResult<Tensor> {
    match cols {
        None => Ok(hidden_to_tensor(hidden_f32, [1, rows, h])),
        Some(c) => masked_tensor(hidden_f32, [1, rows, h], c),
    }
}

/// [`send_stream_open`] with optional mask columns (`None`: the same bytes).
#[allow(clippy::too_many_arguments)]
pub async fn send_stream_open_step(
    cli: &Mutex<ActivationClient>,
    batch_id: u32,
    slot: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    rows: u32,
    h: u32,
    cols: Option<&MaskCols>,
) -> TransportResult<()> {
    if cols.is_none() {
        return send_stream_open(cli, batch_id, slot, sampling, hidden_f32, rows, h).await;
    }
    let tensor = stream_rows_tensor(hidden_f32, rows, h, cols)?;
    let mut header = [0u8; 16 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(FrameKind::StreamOpen as u32).to_be_bytes());
    header[4..8].copy_from_slice(&batch_id.to_be_bytes());
    header[8..12].copy_from_slice(&slot.to_be_bytes());
    header[12..16].copy_from_slice(&rows.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[16..].copy_from_slice(&sbytes);
    send_header_and_tensor(cli, &header, &tensor).await
}

/// Body of a `StreamOpen` (kind consumed), masked or not:
/// `(batch_id, slot, rows, sampling, tensor)`.
pub async fn recv_stream_open_step_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, u32, u32, SamplingConfig, StepTensor)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(12 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 12 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, slot, rows) = (be_u32(&raw[0..4]), be_u32(&raw[4..8]), be_u32(&raw[8..12]));
    if rows == 0 || rows > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream open: rows {rows} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[12..]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != rows {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream open: tensor shape[1]={} != rows {rows}",
            tensor.shape[1]
        ))));
    }
    Ok((batch_id, slot, rows, sampling, step_tensor(tensor)?))
}

/// [`send_stream_feed`] with optional mask columns (`None`: the same bytes).
#[allow(clippy::too_many_arguments)]
pub async fn send_stream_feed_step(
    cli: &Mutex<ActivationClient>,
    batch_id: u32,
    slot: u32,
    flags: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    rows: u32,
    h: u32,
    cols: Option<&MaskCols>,
) -> TransportResult<()> {
    if cols.is_none() {
        return send_stream_feed(cli, batch_id, slot, flags, sampling, hidden_f32, rows, h).await;
    }
    let tensor = stream_rows_tensor(hidden_f32, rows, h, cols)?;
    let mut header = [0u8; 20 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(FrameKind::StreamFeed as u32).to_be_bytes());
    header[4..8].copy_from_slice(&batch_id.to_be_bytes());
    header[8..12].copy_from_slice(&slot.to_be_bytes());
    header[12..16].copy_from_slice(&rows.to_be_bytes());
    header[16..20].copy_from_slice(&flags.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[20..].copy_from_slice(&sbytes);
    send_header_and_tensor(cli, &header, &tensor).await
}

/// Body of a `StreamFeed` (kind consumed), masked or not:
/// `(batch_id, slot, rows, flags, sampling, tensor)`.
pub async fn recv_stream_feed_step_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, u32, u32, u32, SamplingConfig, StepTensor)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(16 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 16 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, slot, rows, flags) = (
        be_u32(&raw[0..4]),
        be_u32(&raw[4..8]),
        be_u32(&raw[8..12]),
        be_u32(&raw[12..16]),
    );
    if rows == 0 || rows > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream feed: rows {rows} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[16..]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != rows {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream feed: tensor shape[1]={} != rows {rows}",
            tensor.shape[1]
        ))));
    }
    Ok((batch_id, slot, rows, flags, sampling, step_tensor(tensor)?))
}

/// [`send_stream_decode`] with optional mask columns (`None`: the same bytes).
pub async fn send_stream_decode_step(
    cli: &Mutex<ActivationClient>,
    batch_id: u32,
    rows: &[(u32, u32)],
    hidden_f32: &[f32],
    h: u32,
    cols: Option<&MaskCols>,
) -> TransportResult<()> {
    if cols.is_none() {
        return send_stream_decode(cli, batch_id, rows, hidden_f32, h).await;
    }
    let n = rows.len() as u32;
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream decode: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let tensor = stream_rows_tensor(hidden_f32, n, h, cols)?;
    let mut header = Vec::with_capacity(12 + rows.len() * 8);
    header.extend_from_slice(&(FrameKind::StreamDecode as u32).to_be_bytes());
    header.extend_from_slice(&batch_id.to_be_bytes());
    header.extend_from_slice(&n.to_be_bytes());
    for &(slot, pos) in rows {
        header.extend_from_slice(&slot.to_be_bytes());
        header.extend_from_slice(&pos.to_be_bytes());
    }
    send_header_and_tensor(cli, &header, &tensor).await
}

/// Body of a `StreamDecode` (kind consumed), masked or not:
/// `(batch_id, [(slot, pos)], tensor)`.
pub async fn recv_stream_decode_step_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, Vec<(u32, u32)>, StepTensor)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(8).await?;
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, n) = (be_u32(&raw[0..4]), be_u32(&raw[4..8]));
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream decode: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let raw = guard.recv_raw(n as usize * 8).await?;
    if raw.len() != n as usize * 8 {
        return Err(TransportError::SocketClosed);
    }
    let rows: Vec<(u32, u32)> = raw
        .chunks_exact(8)
        .map(|c| (be_u32(&c[0..4]), be_u32(&c[4..8])))
        .collect();
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    if tensor.shape[1] != n {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream decode: tensor shape[1]={} != rows {n}",
            tensor.shape[1]
        ))));
    }
    Ok((batch_id, rows, step_tensor(tensor)?))
}

fn token_ack_bytes(token_id: i64, tag: u32, status: u32) -> [u8; 20] {
    let mut bytes = [0u8; 20];
    bytes[0..4].copy_from_slice(&(FrameKind::TokenAck as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&token_id.to_be_bytes());
    bytes[12..16].copy_from_slice(&tag.to_be_bytes());
    bytes[16..20].copy_from_slice(&status.to_be_bytes());
    bytes
}

/// Reply to a masked step: `TokenAck(token, tag, status)`.
pub async fn send_token_ack_upstream(
    srv: &Mutex<ActivationServer>,
    token_id: i64,
    tag: u32,
    status: u32,
) -> TransportResult<()> {
    let bytes = token_ack_bytes(token_id, tag, status);
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Body of a `TokenAck` (kind consumed): `(token, tag, status)`.
pub async fn recv_token_ack_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<(i64, u32, u32)> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(16).await?;
    drop(guard);
    if raw.len() != 16 {
        return Err(TransportError::SocketClosed);
    }
    Ok((
        i64::from_be_bytes(raw[0..8].try_into().unwrap()),
        be_u32(&raw[8..12]),
        be_u32(&raw[12..16]),
    ))
}

/// Reply to a masked stream frame: `StreamTokensAck(batch_id, status, rows)`.
pub async fn send_stream_tokens_ack_upstream(
    srv: &Mutex<ActivationServer>,
    batch_id: u32,
    status: u32,
    tokens: &[(u32, i64)],
) -> TransportResult<()> {
    let n = tokens.len() as u32;
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream tokens: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let mut bytes = Vec::with_capacity(16 + tokens.len() * 12);
    bytes.extend_from_slice(&(FrameKind::StreamTokensAck as u32).to_be_bytes());
    bytes.extend_from_slice(&batch_id.to_be_bytes());
    bytes.extend_from_slice(&status.to_be_bytes());
    bytes.extend_from_slice(&n.to_be_bytes());
    for &(slot, tok) in tokens {
        bytes.extend_from_slice(&slot.to_be_bytes());
        bytes.extend_from_slice(&tok.to_be_bytes());
    }
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Body of a `StreamTokensAck` (kind consumed): `(batch_id, status, rows)`.
pub async fn recv_stream_tokens_ack_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<(u32, u32, Vec<(u32, i64)>)> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(12).await?;
    if raw.len() != 12 {
        return Err(TransportError::SocketClosed);
    }
    let (batch_id, status, n) = (be_u32(&raw[0..4]), be_u32(&raw[4..8]), be_u32(&raw[8..12]));
    if n == 0 || n > MAX_STREAM_ROWS {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "stream tokens: rows {n} out of range 1..={MAX_STREAM_ROWS}"
        ))));
    }
    let raw = guard.recv_raw(n as usize * 12).await?;
    drop(guard);
    if raw.len() != n as usize * 12 {
        return Err(TransportError::SocketClosed);
    }
    let toks = raw
        .chunks_exact(12)
        .map(|c| {
            (
                be_u32(&c[0..4]),
                i64::from_be_bytes(c[4..12].try_into().unwrap()),
            )
        })
        .collect();
    Ok((batch_id, status, toks))
}

/// Rank 0's per-engine memory that its pipeline could not apply a mask: set
/// after a masked step that timed out or came back without an
/// acknowledgement, so later json requests fail at once instead of stalling
/// the pipeline each time. It outlives a re-dial, is cleared when a late
/// acknowledged reply turns up (the stage was slow, not old), and stops
/// refusing after [`MASK_REPROBE`] so the next json request probes again.
///
/// On a middle stage the same record notes a late acknowledged reply it
/// dropped ([`MaskLatch::take_owed`]), so it can tell the stages above with
/// its next reply (see [`send_owed_late_ack`]).
#[derive(Debug, Default)]
pub struct MaskLatch {
    at: std::sync::Mutex<Option<std::time::Instant>>,
    owed: std::sync::atomic::AtomicBool,
}

impl MaskLatch {
    fn slot(&self) -> std::sync::MutexGuard<'_, Option<std::time::Instant>> {
        self.at.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether a late acknowledged reply was dropped since the last call.
    pub fn take_owed(&self) -> bool {
        self.owed.swap(false, std::sync::atomic::Ordering::SeqCst)
    }

    pub fn set(&self) {
        self.set_at(std::time::Instant::now());
    }

    pub fn set_at(&self, at: std::time::Instant) {
        *self.slot() = Some(at);
    }

    pub fn clear(&self) {
        *self.slot() = None;
    }

    pub fn is_set(&self) -> bool {
        self.slot().is_some()
    }

    /// Whether a json request must be refused now.
    pub fn refuses(&self) -> bool {
        self.refuses_at(std::time::Instant::now())
    }

    pub fn refuses_at(&self, now: std::time::Instant) -> bool {
        self.slot()
            .is_some_and(|t| now.saturating_duration_since(t) < MASK_REPROBE)
    }
}

/// A late acknowledged reply read and dropped: the stage behind it does mask.
fn note_stale_ack(latch: Option<&MaskLatch>, status: u32) {
    if status == MASK_ACK {
        if let Some(l) = latch {
            l.clear();
            l.owed.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Tag of the marker a middle stage sends up for a late acknowledged reply
/// it dropped. Rank 0 never tags a step 0, so every wait above reads the
/// marker as a stale acknowledged reply: it is dropped, and it clears rank
/// 0's latch.
pub const LATE_ACK_TAG: u32 = 0;

/// A middle stage: before its next reply upstream, pass on a late
/// acknowledged reply it dropped (`TokenAck(-1, LATE_ACK_TAG, MASK_ACK)`).
pub async fn send_owed_late_ack(
    up: &Mutex<ActivationServer>,
    late: &MaskLatch,
) -> TransportResult<()> {
    if late.take_owed() {
        send_token_ack_upstream(up, -1, LATE_ACK_TAG, MASK_ACK).await?;
    }
    Ok(())
}

/// Whether `cols` is the mask-support probe: one masked row with no mask
/// words. A stage on this build answers it without running its layers or
/// touching any state (middles pass it on); see [`probe_mask_support`].
pub fn is_mask_probe(cols: &MaskCols) -> bool {
    cols.rows() == 1 && cols.width() == 1 && cols.tag(0) != 0
}

/// Rank 0: ask every stage whether it applies masks, with a masked
/// single-row `Forward` that carries no mask words. Every stage on this
/// build acknowledges it without touching any state; a stage on an older
/// build fails to decode it, survives (a failed `Forward` is not fatal on
/// any older worker), and never answers. `Err` names why there was no
/// acknowledgement; the socket is kept as on any masked step.
pub async fn probe_mask_support(
    down: &Mutex<ActivationClient>,
    h: usize,
    deadline: std::time::Duration,
    tag: u32,
    latch: &MaskLatch,
) -> Result<(), String> {
    let mut cols = MaskCols::new(1, 0);
    cols.set(0, tag, &[]);
    send_forward_step(
        down,
        FrameKind::Forward,
        0,
        &SamplingConfig::default(),
        &vec![0.0; h],
        [1, 1, h as u32],
        false,
        Some(&cols),
    )
    .await
    .map_err(|e| format!("send mask probe: {e}"))?;
    match wait_token_ack(down, deadline, tag, Some(latch)).await {
        AckWait::Acked(_, MASK_ACK) => Ok(()),
        AckWait::Acked(_, MASK_UNCONFIRMED) => {
            Err("a later stage did not acknowledge the mask".into())
        }
        AckWait::Acked(_, s) => Err(format!("unknown acknowledgement status 0x{s:08x}")),
        AckWait::NoAck(reason) => Err(reason),
    }
}

/// How a masked step's reply wait ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaskedReply {
    /// Every later stage applied the mask; the token it sampled.
    Token(i64),
    /// The stages apply masks, but the last one could not serve this step
    /// (its mask columns did not fit, or nothing in the row was allowed).
    Refused,
    /// No acknowledgement: a timeout, a closed link, a reply without an ack,
    /// or a middle stage whose downstream did not acknowledge. The likely
    /// cause is a stage on an older build.
    Unconfirmed(String),
}

enum AckWait {
    Acked(i64, u32),
    NoAck(String),
}

/// Wait for the `TokenAck` tagged `tag`, dropping stale acknowledged replies
/// on the way. The socket is kept on timeout (a stage on an older build
/// stays aligned and simply never answers; a slow new stage's late reply is
/// recognised and dropped by the next wait), and dropped only when the
/// stream is out of step (an unexpected frame).
async fn wait_token_ack(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
    tag: u32,
    latch: Option<&MaskLatch>,
) -> AckWait {
    let waited = tokio::time::timeout(deadline, async {
        loop {
            match recv_kind_client_raw(down).await {
                Ok(Some(FrameKind::TokenAck)) => {
                    let (token, got, status) = recv_token_ack_body_client(down)
                        .await
                        .map_err(|e| (format!("recv_token_ack: {e}"), true))?;
                    if got != tag {
                        note_stale_ack(latch, status);
                        continue;
                    }
                    return Ok((token, status));
                }
                Ok(Some(FrameKind::StreamTokensAck)) => {
                    let (_, status, _) = recv_stream_tokens_ack_body_client(down)
                        .await
                        .map_err(|e| (format!("recv_stream_tokens_ack: {e}"), true))?;
                    note_stale_ack(latch, status);
                }
                Ok(Some(FrameKind::Token)) => {
                    recv_token_body_client(down)
                        .await
                        .map_err(|e| (format!("recv_token: {e}"), true))?;
                    return Err(("a reply without acknowledgement".to_string(), false));
                }
                Ok(Some(other)) => return Err((format!("unexpected {other:?} reply"), true)),
                Ok(None) => return Err(("downstream closed before the reply".to_string(), false)),
                Err(e) => return Err((format!("recv_kind: {e}"), true)),
            }
        }
    })
    .await;
    match waited {
        Ok(Ok((token, status))) => AckWait::Acked(token, status),
        Ok(Err((reason, out_of_step))) => {
            if out_of_step {
                down.lock().await.close().await;
            }
            AckWait::NoAck(reason)
        }
        Err(_) => AckWait::NoAck(format!("no reply within {deadline:?}")),
    }
}

/// Rank 0: await the reply to a masked step tagged `tag`, bounded by
/// `deadline`, keeping the socket on timeout (see [`wait_token_ack`]).
pub async fn recv_token_reply_masked(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
    tag: u32,
    latch: &MaskLatch,
) -> MaskedReply {
    match wait_token_ack(down, deadline, tag, Some(latch)).await {
        AckWait::Acked(token, MASK_ACK) if token >= 0 => MaskedReply::Token(token),
        AckWait::Acked(_, MASK_ACK) => MaskedReply::Refused,
        AckWait::Acked(_, MASK_UNCONFIRMED) => {
            MaskedReply::Unconfirmed("a later stage did not acknowledge the mask".into())
        }
        AckWait::Acked(_, s) => {
            MaskedReply::Unconfirmed(format!("unknown acknowledgement status 0x{s:08x}"))
        }
        AckWait::NoAck(reason) => MaskedReply::Unconfirmed(reason),
    }
}

/// A middle stage on a masked step: pass the downstream's acknowledged reply
/// up as it came, or `TokenAck(-1, tag, MASK_UNCONFIRMED)` when the
/// downstream did not acknowledge within `deadline` (socket kept, as on
/// rank 0). The answer upstream is always a `TokenAck`, so a late one is
/// recognised as stale there too.
pub async fn relay_masked_token_reply(
    up: &Mutex<ActivationServer>,
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
    tag: u32,
    late: &MaskLatch,
) -> Result<(), String> {
    let (token, status) = match wait_token_ack(down, deadline, tag, Some(late)).await {
        AckWait::Acked(token, status) => (token, status),
        AckWait::NoAck(reason) => {
            tracing::warn!(%reason, "masked step not acknowledged downstream");
            (-1, MASK_UNCONFIRMED)
        }
    };
    send_owed_late_ack(up, late)
        .await
        .map_err(|e| format!("relay late ack: {e}"))?;
    send_token_ack_upstream(up, token, tag, status)
        .await
        .map_err(|e| format!("relay token ack: {e}"))
}

/// The masked-step budget of the stage at `rank` of `total`, given rank 0's
/// budget `head` for that step: `head · (total − rank) / total`. Each stage
/// waits less than the one upstream of it (by `head / total`), so a middle
/// stage whose downstream never answers sends its unconfirmed reply before
/// any stage above it gives up.
pub fn mid_mask_budget(head: std::time::Duration, rank: u32, total: u32) -> std::time::Duration {
    let total = total.max(1);
    head * total.saturating_sub(rank) / total
}

/// A middle stage's unbounded wait for the `Token` owed to an unmasked frame
/// it relayed, dropping any late acknowledged reply first.
///
/// A dropped acknowledged reply is noted in `late` (see
/// [`send_owed_late_ack`]).
pub async fn recv_token_relay(
    down: &Mutex<ActivationClient>,
    late: Option<&MaskLatch>,
) -> Result<i64, String> {
    loop {
        match recv_kind_client_raw(down).await {
            Ok(Some(FrameKind::Token)) => {
                return recv_token_body_client(down)
                    .await
                    .map_err(|e| format!("recv_token: {e}"))
            }
            Ok(Some(FrameKind::TokenAck)) => {
                let (_, _, status) = recv_token_ack_body_client(down)
                    .await
                    .map_err(|e| format!("recv_token_ack: {e}"))?;
                note_stale_ack(late, status);
            }
            Ok(Some(other)) => return Err(format!("mid rank expected Token, got {other:?}")),
            Ok(None) => return Err("downstream closed before Token".into()),
            Err(e) => return Err(format!("recv_kind: {e}")),
        }
    }
}

/// One reply to a stream frame: plain (`ack: None`) or acknowledged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamReply {
    pub batch_id: u32,
    pub tokens: Vec<(u32, i64)>,
    pub ack: Option<u32>,
}

/// Await the reply to the stream frame `expect`, bounded by `deadline`
/// (dropping the connection on timeout, as [`recv_stream_tokens_reply`]
/// does: the frames of other groups share the reply order). Acknowledged
/// replies for other frames are stale and dropped; a plain reply is returned
/// whatever its batch id, for the caller's mismatch check.
pub async fn recv_stream_reply(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
    expect: u32,
    latch: Option<&MaskLatch>,
) -> Result<StreamReply, String> {
    match tokio::time::timeout(deadline, async {
        loop {
            match recv_kind_client_raw(down).await {
                Ok(Some(FrameKind::StreamTokens)) => {
                    let (batch_id, tokens) = recv_stream_tokens_body_client(down)
                        .await
                        .map_err(|e| format!("recv_stream_tokens: {e}"))?;
                    return Ok(StreamReply {
                        batch_id,
                        tokens,
                        ack: None,
                    });
                }
                Ok(Some(FrameKind::StreamTokensAck)) => {
                    let (batch_id, status, tokens) = recv_stream_tokens_ack_body_client(down)
                        .await
                        .map_err(|e| format!("recv_stream_tokens_ack: {e}"))?;
                    if batch_id != expect {
                        note_stale_ack(latch, status);
                        continue;
                    }
                    return Ok(StreamReply {
                        batch_id,
                        tokens,
                        ack: Some(status),
                    });
                }
                Ok(Some(FrameKind::TokenAck)) => {
                    let (_, _, status) = recv_token_ack_body_client(down)
                        .await
                        .map_err(|e| format!("recv_token_ack: {e}"))?;
                    note_stale_ack(latch, status);
                }
                Ok(Some(other)) => {
                    return Err(format!("expected StreamTokens reply, got {other:?}"))
                }
                Ok(None) => return Err("downstream closed before StreamTokens".into()),
                Err(e) => return Err(format!("recv_kind: {e}")),
            }
        }
    })
    .await
    {
        Ok(res) => res,
        Err(_) => {
            down.lock().await.close().await;
            Err(format!(
                "stream reply timeout after {deadline:?}: downstream silent (dead peer?); \
                 connection dropped"
            ))
        }
    }
}

#[cfg(test)]
mod mask_wire_tests {
    use super::*;
    use std::time::Duration;

    async fn pair() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let server = Arc::new(Mutex::new(server));
        let s = server.clone();
        let accept = tokio::spawn(async move { s.lock().await.accept().await.unwrap() });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client
            .connect_with_timeout(Duration::from_secs(5))
            .await
            .unwrap();
        accept.await.unwrap();
        (server, Arc::new(Mutex::new(client)))
    }

    /// Every byte the client sent since the last call (reads until 200 ms
    /// pass without one).
    async fn raw_bytes(srv: &Mutex<ActivationServer>) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let read = tokio::time::timeout(Duration::from_millis(200), async {
                srv.lock().await.recv_raw(1).await
            })
            .await;
            match read {
                Ok(Ok(b)) => out.extend(b),
                _ => return out,
            }
        }
    }

    fn cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 5,
            seed: Some(11),
            ..SamplingConfig::default()
        }
    }

    fn rows_hidden(rows: usize, h: usize) -> Vec<f32> {
        (0..rows * h).map(|i| i as f32 * 0.25 - 1.0).collect()
    }

    fn one_mask(rows: usize, tag: u32) -> MaskCols {
        let mut c = MaskCols::new(rows, 2);
        c.set(rows - 1, tag, &[0b1010, 1 << 31]);
        c
    }

    #[test]
    fn ack_kinds_are_new_codes() {
        for (k, code) in [
            (FrameKind::TokenAck, 0x53_4D_45_22u32),
            (FrameKind::StreamTokensAck, 0x53_4D_45_65),
        ] {
            assert_eq!(k as u32, code);
            assert_eq!(FrameKind::from_code(code), Some(k));
        }
        assert_eq!(token_ack_bytes(-1, 7, MASK_ACK).len(), 20);
    }

    #[tokio::test]
    async fn unmasked_forward_bytes_are_the_original_encoders() {
        let (srv, cli) = pair().await;
        let h = rows_hidden(1, 3);
        send_forward(&cli, 9, &cfg(), &h, [1, 1, 3], true)
            .await
            .unwrap();
        let old = raw_bytes(&srv).await;
        send_forward_step(
            &cli,
            FrameKind::Forward,
            9,
            &cfg(),
            &h,
            [1, 1, 3],
            true,
            None,
        )
        .await
        .unwrap();
        assert_eq!(raw_bytes(&srv).await, old);
        // Pinned layout: kind, past_seq_len, sampling, push byte, then the
        // F32 tensor frame (len, dtype 0, shape, little-endian payload).
        let mut want = Vec::new();
        want.extend((FrameKind::Forward as u32).to_be_bytes());
        want.extend(9u32.to_be_bytes());
        let mut sb = [0u8; SAMPLING_WIRE_BYTES];
        encode_sampling(&cfg(), &mut sb);
        want.extend(sb);
        want.push(1);
        want.extend(12u32.to_be_bytes());
        want.extend(0u32.to_be_bytes());
        for d in [1u32, 1, 3] {
            want.extend(d.to_be_bytes());
        }
        for v in &h {
            want.extend(v.to_le_bytes());
        }
        assert_eq!(old, want);
    }

    #[tokio::test]
    async fn unmasked_batch_and_stream_bytes_are_the_original_encoders() {
        let (srv, cli) = pair().await;
        let h = rows_hidden(3, 2);
        let c = cfg();
        send_forward_batch_prefill(&cli, 4, 3, &c, &h, [1, 3, 2])
            .await
            .unwrap();
        let old = raw_bytes(&srv).await;
        send_forward_batch_step(
            FrameKind::ForwardBatchPrefill,
            &cli,
            4,
            3,
            &c,
            &h,
            [1, 3, 2],
            None,
        )
        .await
        .unwrap();
        assert_eq!(raw_bytes(&srv).await, old, "ForwardBatchPrefill");

        send_stream_open(&cli, 5, 1, &c, &h, 3, 2).await.unwrap();
        let old = raw_bytes(&srv).await;
        send_stream_open_step(&cli, 5, 1, &c, &h, 3, 2, None)
            .await
            .unwrap();
        assert_eq!(raw_bytes(&srv).await, old, "StreamOpen");

        send_stream_feed(&cli, 6, 1, STREAM_FEED_FINAL, &c, &h, 3, 2)
            .await
            .unwrap();
        let old = raw_bytes(&srv).await;
        send_stream_feed_step(&cli, 6, 1, STREAM_FEED_FINAL, &c, &h, 3, 2, None)
            .await
            .unwrap();
        assert_eq!(raw_bytes(&srv).await, old, "StreamFeed");

        let rows = [(0u32, 7u32), (2, 9), (1, 4)];
        send_stream_decode(&cli, 8, &rows, &h, 2).await.unwrap();
        let old = raw_bytes(&srv).await;
        send_stream_decode_step(&cli, 8, &rows, &h, 2, None)
            .await
            .unwrap();
        assert_eq!(raw_bytes(&srv).await, old, "StreamDecode");
    }

    #[tokio::test]
    async fn masked_forward_round_trips_hidden_and_mask() {
        let (srv, cli) = pair().await;
        let h = rows_hidden(1, 4);
        let cols = one_mask(1, 42);
        send_forward_step(
            &cli,
            FrameKind::Forward,
            3,
            &cfg(),
            &h,
            [1, 1, 4],
            true,
            Some(&cols),
        )
        .await
        .unwrap();
        assert_eq!(
            recv_kind_server(&srv).await.unwrap(),
            Some(FrameKind::Forward)
        );
        let (psl, sampling, push, step) = recv_forward_step_body_server(&srv).await.unwrap();
        assert_eq!((psl, push, sampling.top_k), (3, true, 5));
        assert!(step.is_masked());
        let (hidden, shape, got) = step.split(4).unwrap();
        assert_eq!(hidden, h);
        assert_eq!(shape, [1, 1, 4]);
        let got = got.unwrap();
        assert_eq!(got, cols);
        assert_eq!(got.tag(0), 42);
        assert_eq!(got.words(0), Some(&[0b1010, 1 << 31][..]));

        // The last stage applies it: the favoured token 0 is not allowed,
        // so the pick is the best allowed one (3 over 1 and 63).
        let mut logits = vec![0.0f32; 64];
        logits[0] = 9.0;
        logits[3] = 5.0;
        logits[1] = 4.0;
        logits[63] = 1.0;
        let mut rng = 1;
        let greedy = SamplingConfig::default();
        let words = got.words(0).unwrap();
        let t = crate::sampling::sample_with_words(&logits, &[], &greedy, &mut rng, words);
        assert_eq!(t.unwrap(), 3);
        // A row the mask leaves empty is an error, not an unmasked pick.
        assert!(
            crate::sampling::sample_with_words(&logits[..1], &[], &greedy, &mut rng, words)
                .is_err()
        );
    }

    #[tokio::test]
    async fn masked_batch_and_stream_frames_round_trip() {
        let (srv, cli) = pair().await;
        let h = rows_hidden(3, 2);
        let cols = one_mask(3, 5);
        send_forward_batch_step(
            FrameKind::ForwardBatchPrefill,
            &cli,
            0,
            3,
            &cfg(),
            &h,
            [1, 3, 2],
            Some(&cols),
        )
        .await
        .unwrap();
        recv_kind_server(&srv).await.unwrap();
        let (_, n, _, step) = recv_forward_batch_step_body_server(&srv).await.unwrap();
        assert_eq!(n, 3);
        let (hidden, shape, got) = step.split(2).unwrap();
        assert_eq!(
            (hidden, shape, got),
            (h.clone(), [1, 3, 2], Some(cols.clone()))
        );
        let got = cols.clone();
        assert_eq!((got.tag(0), got.tag(1), got.tag(2)), (0, 0, 5));
        assert_eq!(got.words(0), None);

        let mut dec = MaskCols::new(2, 2);
        dec.set(0, 9, &[1, 2]);
        let rows = [(1u32, 4u32), (0, 6)];
        send_stream_decode_step(&cli, 3, &rows, &h[..4], 2, Some(&dec))
            .await
            .unwrap();
        recv_kind_server(&srv).await.unwrap();
        let (bid, got_rows, step) = recv_stream_decode_step_body_server(&srv).await.unwrap();
        assert_eq!((bid, got_rows), (3, rows.to_vec()));
        let (hidden, _, got) = step.split(2).unwrap();
        assert_eq!(hidden, h[..4].to_vec());
        let got = got.unwrap();
        assert_eq!((got.tag(0), got.tag(1)), (9, 0));
    }

    #[test]
    fn a_masked_frame_narrower_than_the_hidden_is_refused_by_split() {
        let t = masked_tensor(&[1.0, 2.0], [1, 1, 2], &MaskCols::new(1, 0)).unwrap();
        assert!(StepTensor::Masked(t).split(4).is_err());
        let cols = MaskCols::new(1, 0);
        assert_eq!(cols.words(0), None);
    }

    /// An older stage's body decoders (the original functions, unchanged)
    /// read the whole masked frame, fail on its dtype, and leave the stream
    /// aligned: the next plain frame decodes.
    #[tokio::test]
    async fn old_decoders_consume_a_masked_frame_and_stay_aligned() {
        let (srv, cli) = pair().await;
        let c = cfg();
        let h1 = rows_hidden(1, 4);
        let h3 = rows_hidden(3, 4);
        let next_plain_forward = |s: Arc<Mutex<ActivationServer>>| async move {
            assert_eq!(
                recv_kind_server(&s).await.unwrap(),
                Some(FrameKind::Forward)
            );
            let (psl, _, _, h, shape) = recv_forward_body_server(&s).await.unwrap();
            assert_eq!((psl, shape), (77, [1, 1, 4]));
            h
        };

        // Forward.
        send_forward_step(
            &cli,
            FrameKind::Forward,
            1,
            &c,
            &h1,
            [1, 1, 4],
            true,
            Some(&one_mask(1, 1)),
        )
        .await
        .unwrap();
        send_forward(&cli, 77, &c, &h1, [1, 1, 4], true)
            .await
            .unwrap();
        recv_kind_server(&srv).await.unwrap();
        let e = recv_forward_body_server(&srv).await.unwrap_err();
        assert!(e.to_string().contains("expected F32 hidden tensor"), "{e}");
        assert_eq!(next_plain_forward(srv.clone()).await, h1);

        // ForwardBatchPrefill.
        send_forward_batch_step(
            FrameKind::ForwardBatchPrefill,
            &cli,
            0,
            3,
            &c,
            &h3,
            [1, 3, 4],
            Some(&one_mask(3, 2)),
        )
        .await
        .unwrap();
        send_forward(&cli, 77, &c, &h1, [1, 1, 4], true)
            .await
            .unwrap();
        recv_kind_server(&srv).await.unwrap();
        assert!(recv_forward_batch_body_server(&srv).await.is_err());
        assert_eq!(next_plain_forward(srv.clone()).await, h1);

        // StreamOpen.
        send_stream_open_step(&cli, 1, 0, &c, &h3, 3, 4, Some(&one_mask(3, 3)))
            .await
            .unwrap();
        send_forward(&cli, 77, &c, &h1, [1, 1, 4], true)
            .await
            .unwrap();
        recv_kind_server(&srv).await.unwrap();
        assert!(recv_stream_open_body_server(&srv).await.is_err());
        assert_eq!(next_plain_forward(srv.clone()).await, h1);

        // StreamFeed.
        send_stream_feed_step(
            &cli,
            2,
            0,
            STREAM_FEED_FINAL,
            &c,
            &h3,
            3,
            4,
            Some(&one_mask(3, 4)),
        )
        .await
        .unwrap();
        send_forward(&cli, 77, &c, &h1, [1, 1, 4], true)
            .await
            .unwrap();
        recv_kind_server(&srv).await.unwrap();
        assert!(recv_stream_feed_body_server(&srv).await.is_err());
        assert_eq!(next_plain_forward(srv.clone()).await, h1);

        // StreamDecode.
        send_stream_decode_step(&cli, 3, &[(0, 5)], &h1, 4, Some(&one_mask(1, 5)))
            .await
            .unwrap();
        send_forward(&cli, 77, &c, &h1, [1, 1, 4], true)
            .await
            .unwrap();
        recv_kind_server(&srv).await.unwrap();
        assert!(recv_stream_decode_body_server(&srv).await.is_err());
        assert_eq!(next_plain_forward(srv.clone()).await, h1);
    }

    const SHORT: Duration = Duration::from_millis(300);

    #[tokio::test]
    async fn masked_wait_takes_only_an_acknowledged_reply() {
        let (srv, cli) = pair().await;
        let latch = MaskLatch::default();
        // Acknowledged token.
        send_token_ack_upstream(&srv, 5, 3, MASK_ACK).await.unwrap();
        assert_eq!(
            recv_token_reply_masked(&cli, SHORT, 3, &latch).await,
            MaskedReply::Token(5)
        );
        // Acknowledged NACK: the stages mask, this step failed.
        send_token_ack_upstream(&srv, -1, 4, MASK_ACK)
            .await
            .unwrap();
        assert_eq!(
            recv_token_reply_masked(&cli, SHORT, 4, &latch).await,
            MaskedReply::Refused
        );
        // A plain reply (no acknowledgement) is not accepted.
        send_token_upstream(&srv, 6).await.unwrap();
        assert!(matches!(
            recv_token_reply_masked(&cli, SHORT, 5, &latch).await,
            MaskedReply::Unconfirmed(r) if r.contains("without acknowledgement")
        ));
        // A plain NACK neither.
        send_token_upstream(&srv, -1).await.unwrap();
        assert!(matches!(
            recv_token_reply_masked(&cli, SHORT, 6, &latch).await,
            MaskedReply::Unconfirmed(_)
        ));
        // A middle stage whose downstream did not acknowledge.
        send_token_ack_upstream(&srv, -1, 7, MASK_UNCONFIRMED)
            .await
            .unwrap();
        assert!(matches!(
            recv_token_reply_masked(&cli, SHORT, 7, &latch).await,
            MaskedReply::Unconfirmed(_)
        ));
        // A stale acknowledged reply is dropped, and clears the latch.
        latch.set();
        send_token_ack_upstream(&srv, 1, 99, MASK_ACK)
            .await
            .unwrap();
        send_token_ack_upstream(&srv, 2, 8, MASK_ACK).await.unwrap();
        assert_eq!(
            recv_token_reply_masked(&cli, SHORT, 8, &latch).await,
            MaskedReply::Token(2)
        );
        assert!(!latch.is_set());
    }

    #[tokio::test]
    async fn masked_timeout_keeps_the_socket_and_the_late_ack_is_dropped() {
        let (srv, cli) = pair().await;
        let latch = MaskLatch::default();
        let r = recv_token_reply_masked(&cli, SHORT, 11, &latch).await;
        assert!(
            matches!(r, MaskedReply::Unconfirmed(ref m) if m.starts_with("no reply within")),
            "{r:?}"
        );
        latch.set();
        // The slow stage's reply lands late, then the next (plain) request's.
        send_token_ack_upstream(&srv, 3, 11, MASK_ACK)
            .await
            .unwrap();
        send_forward(&cli, 0, &cfg(), &[0.0; 4], [1, 1, 4], true)
            .await
            .expect("socket still open after the masked timeout");
        send_token_upstream(&srv, 21).await.unwrap();
        assert_eq!(
            recv_token_reply_with(&cli, SHORT, Some(&latch)).await,
            Ok(21)
        );
        assert!(
            !latch.is_set(),
            "an acknowledged late reply clears the latch"
        );
        // The server side still reads the plain request in order.
        assert_eq!(
            recv_kind_server(&srv).await.unwrap(),
            Some(FrameKind::Forward)
        );
    }

    #[tokio::test]
    async fn unmasked_waits_drop_late_acks() {
        let (srv, cli) = pair().await;
        send_stream_tokens_ack_upstream(&srv, 4, MASK_ACK, &[(0, 1)])
            .await
            .unwrap();
        send_token_ack_upstream(&srv, 3, 1, MASK_UNCONFIRMED)
            .await
            .unwrap();
        send_token_upstream(&srv, 8).await.unwrap();
        assert_eq!(recv_token_reply(&cli, SHORT).await, Ok(8));
        send_token_ack_upstream(&srv, 3, 1, MASK_ACK).await.unwrap();
        send_token_upstream(&srv, 9).await.unwrap();
        assert_eq!(recv_token_relay(&cli, None).await, Ok(9));
        // Stream waits: a stale ack is dropped, the expected one returned.
        send_stream_tokens_ack_upstream(&srv, 1, MASK_ACK, &[(0, 1)])
            .await
            .unwrap();
        send_stream_tokens_ack_upstream(&srv, 2, MASK_ACK, &[(0, 5), (1, 6)])
            .await
            .unwrap();
        send_stream_tokens_upstream(&srv, 3, &[(1, 7)])
            .await
            .unwrap();
        let r = recv_stream_reply(&cli, SHORT, 2, None).await.unwrap();
        assert_eq!(
            r,
            StreamReply {
                batch_id: 2,
                tokens: vec![(0, 5), (1, 6)],
                ack: Some(MASK_ACK)
            }
        );
        let r = recv_stream_reply(&cli, SHORT, 3, None).await.unwrap();
        assert_eq!((r.batch_id, r.ack), (3, None));
    }

    #[tokio::test]
    async fn a_middle_stage_passes_up_an_ack_only_when_downstream_acked() {
        // head --cli_up/srv_up-- mid --cli_dn/srv_dn-- tail
        let (srv_up, cli_up) = pair().await;
        let (srv_dn, cli_dn) = pair().await;
        // Downstream acknowledged: relayed as it came.
        send_token_ack_upstream(&srv_dn, 12, 5, MASK_ACK)
            .await
            .unwrap();
        relay_masked_token_reply(&srv_up, &cli_dn, SHORT, 5, &MaskLatch::default())
            .await
            .unwrap();
        recv_kind_client_raw(&cli_up).await.unwrap();
        assert_eq!(
            recv_token_ack_body_client(&cli_up).await.unwrap(),
            (12, 5, MASK_ACK)
        );
        // Downstream silent (an older tail): unconfirmed, socket kept.
        relay_masked_token_reply(&srv_up, &cli_dn, SHORT, 6, &MaskLatch::default())
            .await
            .unwrap();
        recv_kind_client_raw(&cli_up).await.unwrap();
        assert_eq!(
            recv_token_ack_body_client(&cli_up).await.unwrap(),
            (-1, 6, MASK_UNCONFIRMED)
        );
        // Downstream replied plain: unconfirmed.
        send_token_upstream(&srv_dn, 4).await.unwrap();
        relay_masked_token_reply(&srv_up, &cli_dn, SHORT, 7, &MaskLatch::default())
            .await
            .unwrap();
        recv_kind_client_raw(&cli_up).await.unwrap();
        assert_eq!(
            recv_token_ack_body_client(&cli_up).await.unwrap(),
            (-1, 7, MASK_UNCONFIRMED)
        );
    }

    /// A masked step's reply that lands late must not shift any later
    /// read: a control exchange in between (KV restore and capture acks)
    /// still reads its own reply, and the next plain step its own token.
    #[tokio::test]
    async fn a_late_ack_before_a_control_exchange_does_not_shift_later_reads() {
        let (srv, cli) = pair().await;
        send_token_ack_upstream(&srv, 3, 1, MASK_ACK).await.unwrap();
        send_restore_ack_upstream(&srv, 9, 1).await.unwrap();
        assert_eq!(recv_restore_verdict(&cli).await, Ok(true));
        send_stream_tokens_ack_upstream(&srv, 2, MASK_ACK, &[(0, 4)])
            .await
            .unwrap();
        send_token_ack_upstream(&srv, 5, 2, MASK_UNCONFIRMED)
            .await
            .unwrap();
        send_capture_ack_upstream(&srv, 77).await.unwrap();
        assert_eq!(
            recv_kind_client(&cli).await.unwrap(),
            Some(FrameKind::CaptureAck)
        );
        assert_eq!(recv_capture_ack_body_client(&cli).await.unwrap(), 77);
        send_token_ack_upstream(&srv, 6, 3, MASK_ACK).await.unwrap();
        send_token_batch_upstream(&srv, &[10, 11]).await.unwrap();
        assert_eq!(
            recv_kind_client(&cli).await.unwrap(),
            Some(FrameKind::TokenBatch)
        );
        assert_eq!(
            recv_token_batch_body_client(&cli).await.unwrap(),
            vec![10, 11]
        );
        send_token_upstream(&srv, 42).await.unwrap();
        assert_eq!(recv_token_reply(&cli, SHORT).await, Ok(42));
    }

    /// A middle that drops a slow tail's late acknowledged reply passes it on
    /// with its next reply, so rank 0's latch clears.
    #[tokio::test]
    async fn a_middle_passes_a_dropped_late_ack_up_with_its_next_reply() {
        let (srv_up, cli_up) = pair().await;
        let (srv_dn, cli_dn) = pair().await;
        let late = MaskLatch::default();
        send_token_ack_upstream(&srv_dn, 3, 9, MASK_ACK)
            .await
            .unwrap();
        send_token_upstream(&srv_dn, 5).await.unwrap();
        assert_eq!(recv_token_relay(&cli_dn, Some(&late)).await, Ok(5));
        send_owed_late_ack(&srv_up, &late).await.unwrap();
        send_token_upstream(&srv_up, 5).await.unwrap();
        let head = MaskLatch::default();
        head.set();
        assert_eq!(
            recv_token_reply_with(&cli_up, SHORT, Some(&head)).await,
            Ok(5)
        );
        assert!(!head.is_set(), "the late ack cleared rank 0's latch");
        // Owed once only.
        assert!(!late.take_owed());
        // The masked relay carries it too.
        send_token_ack_upstream(&srv_dn, 1, 2, MASK_ACK)
            .await
            .unwrap();
        send_token_ack_upstream(&srv_dn, 7, 4, MASK_ACK)
            .await
            .unwrap();
        relay_masked_token_reply(&srv_up, &cli_dn, SHORT, 4, &late)
            .await
            .unwrap();
        head.set();
        assert_eq!(
            recv_token_reply_masked(&cli_up, SHORT, 4, &head).await,
            MaskedReply::Token(7)
        );
        assert!(!head.is_set());
    }

    #[tokio::test]
    async fn the_probe_is_acknowledged_without_a_mask_word() {
        let (srv, cli) = pair().await;
        let latch = MaskLatch::default();
        let tail = tokio::spawn(async move {
            recv_kind_server(&srv).await.unwrap();
            let (_, _, _, step) = recv_forward_step_body_server(&srv).await.unwrap();
            let (_, _, cols) = step.split(4).unwrap();
            let cols = cols.unwrap();
            assert!(is_mask_probe(&cols));
            send_token_ack_upstream(&srv, -1, cols.tag(0), MASK_ACK)
                .await
                .unwrap();
            srv
        });
        assert_eq!(probe_mask_support(&cli, 4, SHORT, 3, &latch).await, Ok(()));
        let srv = tail.await.unwrap();
        // Unanswered (an older stage): an error, socket kept.
        assert!(probe_mask_support(&cli, 4, SHORT, 4, &latch).await.is_err());
        send_token_upstream(&srv, 8).await.unwrap();
        assert_eq!(recv_token_reply(&cli, SHORT).await, Ok(8));
        let mut real = MaskCols::new(1, 1);
        real.set(0, 1, &[1]);
        assert!(!is_mask_probe(&real));
    }

    #[test]
    fn each_stage_waits_less_than_the_one_upstream() {
        for head in [Duration::from_secs(60), Duration::from_secs(600)] {
            for total in 2..=12u32 {
                let mut upstream = head;
                for rank in 1..total {
                    let b = mid_mask_budget(head, rank, total);
                    assert!(b + head / total <= upstream, "{head:?} {rank}/{total}");
                    assert!(b > Duration::ZERO);
                    upstream = b;
                }
            }
        }
    }

    #[test]
    fn latch_refuses_until_the_reprobe_interval_passes() {
        let latch = MaskLatch::default();
        let t0 = std::time::Instant::now();
        assert!(!latch.refuses_at(t0));
        latch.set_at(t0);
        assert!(latch.refuses_at(t0 + Duration::from_secs(599)));
        assert!(!latch.refuses_at(t0 + MASK_REPROBE));
        latch.clear();
        assert!(!latch.refuses_at(t0));
    }

    #[test]
    fn largest_masked_frames_fit_the_tensor_cap() {
        // 256 rows (the frame row cap) of a 7168-wide hidden with a
        // 262144-token vocab mask: (7168 + 1 + 8192) * 4 * 256 bytes.
        let bytes = MAX_BATCH_COUNT as usize * (7168 + 1 + 262_144 / 32) * 4;
        assert!(bytes <= cascadia_transport::MAX_TENSOR_BYTES, "{bytes}");
    }
}
