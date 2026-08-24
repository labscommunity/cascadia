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
    Forward = 0x53_4D_45_0C,      // "SME\x0C" — was 0x04 (no push_history byte)
    Reset = 0x53_4D_45_10,        // "SME\x10"
    Token = 0x53_4D_45_20,        // "SME\x20"
    ForwardBatch = 0x53_4D_45_05, // "SME\x05" — was 0x03; batched K-step verify
    TokenBatch = 0x53_4D_45_21,   // "SME\x21" — batched K-step response
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
    CachePrefix = 0x53_4D_45_0A,  // "SME\x0A"
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

pub async fn recv_kind_client(cli: &Mutex<ActivationClient>) -> TransportResult<Option<FrameKind>> {
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
pub async fn recv_token_reply(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
) -> Result<i64, String> {
    match tokio::time::timeout(deadline, async {
        match recv_kind_client(down).await {
            Ok(Some(FrameKind::Token)) => recv_token_body_client(down)
                .await
                .map_err(|e| format!("recv_token: {e}")),
            Ok(Some(other)) => Err(format!("expected Token reply, got {other:?}")),
            Ok(None) => Err("downstream closed before Token".into()),
            Err(e) => Err(format!("recv_kind: {e}")),
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
    let body = if n > 0 {
        guard.recv_raw(n * 4).await?
    } else {
        Vec::new()
    };
    drop(guard);
    if body.len() != n * 4 {
        return Err(TransportError::SocketClosed);
    }
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
    let body = if n > 0 {
        guard.recv_raw(n * 4).await?
    } else {
        Vec::new()
    };
    drop(guard);
    if body.len() != n * 4 {
        return Err(TransportError::SocketClosed);
    }
    let mut tokens = Vec::with_capacity(n);
    for c in body.chunks_exact(4) {
        tokens.push(i32::from_be_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((epoch, tokens, tenant))
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
