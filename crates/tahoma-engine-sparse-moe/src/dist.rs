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
//!   - F32 hidden tensor [1, 1, hidden_size]
//!
//! `Reset` clears KV state on downstream for a new generation. No body.
//!
//! `Token` returns one sampled token i64 to upstream after the last
//! rank runs head + sample. Body:
//!   - 8 B big-endian i64 token_id

use std::sync::Arc;

use tahoma_transport::{
    ActivationClient, ActivationServer, DType, Tensor, TransportError, TransportResult,
};
use tokio::sync::Mutex;

/// 4-byte big-endian frame kinds. Chosen to be disjoint from
/// `dist_spec`'s `FrameKind` so a stray cross-engine connection would
/// fail fast on an unrecognized code.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Forward = 0x53_4D_45_01, // "SME\x01"
    Reset = 0x53_4D_45_02,   // "SME\x02"
    Token = 0x53_4D_45_03,   // "SME\x03"
}

impl FrameKind {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            x if x == FrameKind::Forward as u32 => Some(FrameKind::Forward),
            x if x == FrameKind::Reset as u32 => Some(FrameKind::Reset),
            x if x == FrameKind::Token as u32 => Some(FrameKind::Token),
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
    if bytes.len() != 4 {
        return Err(TransportError::SocketClosed);
    }
    let code = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    FrameKind::from_code(code).ok_or_else(|| {
        TransportError::Io(std::io::Error::other(format!(
            "unknown sparse-MoE frame kind 0x{code:08x}"
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

/// Send a Forward frame downstream: kind + past_seq_len (u32 BE) + hidden tensor.
pub async fn send_forward(
    cli: &Mutex<ActivationClient>,
    past_seq_len: u32,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    let mut header = [0u8; 8];
    header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
    header[4..8].copy_from_slice(&past_seq_len.to_be_bytes());
    let tensor = hidden_to_tensor(hidden_f32, hidden_shape);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Receive a Forward frame's body (the kind code has already been
/// consumed by `recv_kind_*`). Returns `(past_seq_len, hidden_f32, shape)`.
pub async fn recv_forward_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, Vec<f32>, [u32; 3])> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(4).await?;
    if raw.len() != 4 {
        return Err(TransportError::SocketClosed);
    }
    let past_seq_len = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    let (h, shape) = tensor_to_hidden(&tensor)?;
    Ok((past_seq_len, h, shape))
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

/// Middle ranks relay Token upstream when they receive it from
/// downstream. Reads + re-sends in one call.
pub async fn relay_token_body(
    downstream: &Mutex<ActivationClient>,
    upstream: &Mutex<ActivationServer>,
) -> TransportResult<i64> {
    let token = recv_token_body_client(downstream).await?;
    send_token_upstream(upstream, token).await?;
    Ok(token)
}

/// Forward a Reset frame upstream → downstream (mid ranks).
pub async fn forward_reset(
    upstream: &Mutex<ActivationServer>,
    downstream: &Mutex<ActivationClient>,
) -> TransportResult<()> {
    // Reset has no body; we've already consumed the kind on upstream.
    // The downstream send is identical to send_reset.
    let _ = upstream; // present for symmetry / future control flow
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
