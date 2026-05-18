//! Wire protocol for pipeline-parallel sparse-MoE inference.
//!
//! Frames flow over the existing tahoma-transport TCP sockets. Each
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
//! `HeadPartial` carries one rank's contribution to a tensor-parallel
//! head: a contiguous slice of the vocab dim of the logits. Flowing
//! direction is **upstream** (any earlier rank that holds a head slice
//! → the sampling rank), but the wire body is symmetric so a future
//! ring-style partition could re-use it. Body:
//!   - 4 B big-endian u32 vocab_start
//!   - 4 B big-endian u32 vocab_end  (exclusive)
//!   - F32 partial logits tensor [1, 1, vocab_end - vocab_start]
//!
//! Head TP is OFF by default. When enabled (engine config
//! `enable_head_tp`), rank-0 holds the lower vocab slice + computes its
//! partial in parallel with the sampling rank's slice. See
//! `crates/tahoma-int4-gemm/src/head.rs` for the math contract.

use std::sync::Arc;

use tahoma_transport::{
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
/// - HeadPartial introduced for head TP (PR #11+, default-off; existing
///   peers reject it as unknown if they don't recognize the code, which
///   is the desired failure mode)
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Forward = 0x53_4D_45_02,     // "SME\x02"
    Reset = 0x53_4D_45_10,       // "SME\x10"
    Token = 0x53_4D_45_20,       // "SME\x20"
    HeadPartial = 0x53_4D_45_30, // "SME\x30" — per-rank partial logits for head TP
}

impl FrameKind {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            x if x == FrameKind::Forward as u32 => Some(FrameKind::Forward),
            x if x == FrameKind::Reset as u32 => Some(FrameKind::Reset),
            x if x == FrameKind::Token as u32 => Some(FrameKind::Token),
            x if x == FrameKind::HeadPartial as u32 => Some(FrameKind::HeadPartial),
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

/// On-wire encoding of `SamplingConfig`. Fixed 28 bytes BE; the `seed`
/// field uses 0 as the sentinel for "no seed" (the engine treats 0 as
/// invalid anyway — `init_rng` clamps to `max(seed, 1)`).
pub const SAMPLING_WIRE_BYTES: usize = 28;

pub fn encode_sampling(cfg: &SamplingConfig, out: &mut [u8; SAMPLING_WIRE_BYTES]) {
    out[0..4].copy_from_slice(&cfg.temperature.to_be_bytes());
    out[4..8].copy_from_slice(&cfg.top_p.to_be_bytes());
    out[8..12].copy_from_slice(&cfg.repetition_penalty.to_be_bytes());
    out[12..20].copy_from_slice(&(cfg.repetition_window as u64).to_be_bytes());
    let seed = cfg.seed.unwrap_or(0);
    out[20..28].copy_from_slice(&seed.to_be_bytes());
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
    let repetition_penalty = sanitize_f32(
        f32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        1.0,
        0.0,
    );
    let rep_window = u64::from_be_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]) as usize;
    let seed_raw = u64::from_be_bytes([
        bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
    ]);
    SamplingConfig {
        temperature,
        top_p,
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

/// Send a Forward frame downstream: kind + past_seq_len (u32 BE) + 28 B
/// SamplingConfig + hidden tensor.
pub async fn send_forward(
    cli: &Mutex<ActivationClient>,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    let mut header = [0u8; 8 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
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

/// Forward a Reset frame downstream — used by mid ranks after consuming
/// the kind code from upstream. The body is empty so this is just a
/// `send_reset` to the next peer; the `upstream` argument is held by the
/// caller and the kind is already consumed before this point.
pub async fn forward_reset(downstream: &Mutex<ActivationClient>) -> TransportResult<()> {
    send_reset(downstream).await
}

/// Send a HeadPartial frame upstream (rank holding a vocab slice →
/// sampling rank). Body:
/// - kind (4 B BE u32)
/// - vocab_start (4 B BE u32)
/// - vocab_end (4 B BE u32, exclusive)
/// - F32 logits tensor [1, 1, vocab_end - vocab_start]
///
/// The shape is `[1, 1, slice_len]` to match the other tensor frames'
/// `[batch, seq, dim]` convention and let the existing tahoma-transport
/// `Tensor` carry it without a new wire shape variant.
pub async fn send_head_partial(
    srv: &Mutex<ActivationServer>,
    vocab_start: u32,
    vocab_end: u32,
    partial_f32: &[f32],
) -> TransportResult<()> {
    if (vocab_end - vocab_start) as usize != partial_f32.len() {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_head_partial: vocab range {}..{} = {} != partial len {}",
            vocab_start,
            vocab_end,
            vocab_end - vocab_start,
            partial_f32.len()
        ))));
    }
    let mut header = [0u8; 12];
    header[0..4].copy_from_slice(&(FrameKind::HeadPartial as u32).to_be_bytes());
    header[4..8].copy_from_slice(&vocab_start.to_be_bytes());
    header[8..12].copy_from_slice(&vocab_end.to_be_bytes());
    let tensor = hidden_to_tensor(partial_f32, [1, 1, (vocab_end - vocab_start)]);
    let mut guard = srv.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Receive a HeadPartial frame's body on the client side (sampling rank
/// reads from the rank that owns a vocab slice upstream of it on the
/// pipeline). Returns `(vocab_start, vocab_end, partial_logits)`.
///
/// The kind code has already been consumed by `recv_kind_*`.
pub async fn recv_head_partial_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<(u32, u32, Vec<f32>)> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(8).await?;
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    let vocab_start = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let vocab_end = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    let (logits, shape) = tensor_to_hidden(&tensor)?;
    let expected = (vocab_end - vocab_start) as usize;
    if logits.len() != expected {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_head_partial: shape {:?} (len {}) != expected slice {}",
            shape,
            logits.len(),
            expected
        ))));
    }
    Ok((vocab_start, vocab_end, logits))
}

/// Server-side variant of `recv_head_partial_body_client` — used when
/// the rank receiving a HeadPartial accepted the upstream connection
/// (rather than dialing it). For the v1 wiring rank-0 dials the
/// downstream rank-1, so the sampling rank (rank-1) reads via its
/// server socket from rank-0. Symmetric with the client variant.
pub async fn recv_head_partial_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, u32, Vec<f32>)> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(8).await?;
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    let vocab_start = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let vocab_end = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    let (logits, shape) = tensor_to_hidden(&tensor)?;
    let expected = (vocab_end - vocab_start) as usize;
    if logits.len() != expected {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "recv_head_partial: shape {:?} (len {}) != expected slice {}",
            shape,
            logits.len(),
            expected
        ))));
    }
    Ok((vocab_start, vocab_end, logits))
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
