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
    Forward = 0x53_4D_45_04,      // "SME\x04" — was 0x02 (28-byte sampling)
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
    ForwardNoSample = 0x53_4D_45_06, // "SME\x06"
    /// Streamed prefill Forward (0x07): identical body to `ForwardNoSample`
    /// (the receiver advances KV but skips head/sample/record), except it is
    /// **one-way** — no `Token(-1)` ack. Rank 0 fires one per prompt token
    /// (except the last) WITHOUT blocking, and mid ranks relay downstream
    /// without waiting for a reply, so the prompt tokens pipeline through the
    /// ranks (per-rank compute overlaps) instead of one blocking 6-hop
    /// round-trip each. The final prompt token still goes as a sampling
    /// `Forward`, whose returned token is the first generated token.
    ForwardPrefill = 0x53_4D_45_07, // "SME\x07"
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
/// SamplingConfig block + hidden tensor. Shared by [`send_forward`] and
/// [`send_forward_nosample`] — the two kinds carry identical bodies.
async fn send_forward_kind(
    cli: &Mutex<ActivationClient>,
    kind: FrameKind,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    let mut header = [0u8; 8 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(kind as u32).to_be_bytes());
    header[4..8].copy_from_slice(&past_seq_len.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[8..8 + SAMPLING_WIRE_BYTES].copy_from_slice(&sbytes);
    let tensor = hidden_to_tensor(hidden_f32, hidden_shape);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Send a Forward frame downstream: kind + past_seq_len (u32 BE) + 28 B
/// SamplingConfig + hidden tensor.
pub async fn send_forward(
    cli: &Mutex<ActivationClient>,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    send_forward_kind(
        cli,
        FrameKind::Forward,
        past_seq_len,
        sampling,
        hidden_f32,
        hidden_shape,
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
    )
    .await
}

/// Receive a Forward frame's body (the kind code has already been
/// consumed by `recv_kind_*`). Returns
/// `(past_seq_len, sampling, hidden_f32, shape)`.
pub async fn recv_forward_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, SamplingConfig, Vec<f32>, [u32; 3])> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(4 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 4 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let past_seq_len = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[4..4 + SAMPLING_WIRE_BYTES]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    let (h, shape) = tensor_to_hidden(&tensor)?;
    Ok((past_seq_len, sampling, h, shape))
}

/// Send a Reset frame downstream — clears KV state for a new task.
pub async fn send_reset(cli: &Mutex<ActivationClient>) -> TransportResult<()> {
    let mut header = [0u8; 4];
    header[0..4].copy_from_slice(&(FrameKind::Reset as u32).to_be_bytes());
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    Ok(())
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
/// On timeout this returns `Err`, so the caller tears the stage down and
/// reconnects. The caller picks `deadline` for the frame it just sent; for
/// dsv4's token-at-a-time forwarding that is a single per-token budget
/// ([`cascadia_transport::recv_timeout`]) whether the token is prefill or
/// decode, since each reply carries the same single-token downstream compute.
pub async fn recv_token_reply(
    down: &Mutex<ActivationClient>,
    deadline: std::time::Duration,
) -> Result<i64, String> {
    tokio::time::timeout(deadline, async {
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
    .map_err(|_| format!("reply timeout after {deadline:?}: downstream silent (dead peer?)"))?
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
    header[0..4].copy_from_slice(&(FrameKind::ForwardBatch as u32).to_be_bytes());
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
