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
///
/// `KvMigration` is a NEW frame (skeleton; see
/// `docs/architecture/kv-migration.md`). It ships a slab of per-layer
/// KV state from one rank to another. Body layout is documented on
/// `send_kv_migration` / `recv_kv_migration_body_*`. The 0x30 code is
/// chosen above Token (0x20) but below any future range we might
/// allocate for migration-related controls (KvMigrationAck etc).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Forward = 0x53_4D_45_02,     // "SME\x02"
    Reset = 0x53_4D_45_10,       // "SME\x10"
    Token = 0x53_4D_45_20,       // "SME\x20"
    KvMigration = 0x53_4D_45_30, // "SME\x30"
}

impl FrameKind {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            x if x == FrameKind::Forward as u32 => Some(FrameKind::Forward),
            x if x == FrameKind::Reset as u32 => Some(FrameKind::Reset),
            x if x == FrameKind::Token as u32 => Some(FrameKind::Token),
            x if x == FrameKind::KvMigration as u32 => Some(FrameKind::KvMigration),
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

// ============================================================
// KvMigration frame (skeleton — see docs/architecture/kv-migration.md)
//
// Body layout (after the 4 B FrameKind code):
//
//   20 B header:
//     [4 B BE u32 lid]
//     [4 B BE u32 past_seq_len]
//     [4 B BE u32 num_heads]
//     [4 B BE u32 qk_head_dim]
//     [4 B BE u32 v_head_dim]
//
//   I8 tensor with shape [1, 1, total_bytes]:
//     [num_heads * past_seq_len * qk_head_dim * 4 bytes: K (f32 LE)]
//     [num_heads * past_seq_len * v_head_dim  * 4 bytes: V (f32 LE)]
//
// One frame ships ONE layer's KV state. Multi-layer migration is sent
// as N consecutive KvMigration frames. This keeps every individual
// frame well under `MAX_TENSOR_BYTES` (256 MiB) for any reasonable
// context length: K2.6 per-layer at past_seq_len=2048 ≈ 164 MiB.
//
// Longer contexts (past_seq_len > ~2k) would still exceed the cap on
// a single layer. A future revision can add a slot-chunked variant
// (sub-frame `[slot_start, slot_end)` on the header) — out of scope
// for v1, blocker logged in the design doc.
// ============================================================

/// Fixed-size per-layer header preceding the KV-tensor body. Big-endian.
pub const KV_MIGRATION_HEADER_BYTES: usize = 20;

/// Encode the per-layer header inline.
fn write_kv_header(
    out: &mut [u8; KV_MIGRATION_HEADER_BYTES],
    lid: u32,
    past_seq_len: u32,
    num_heads: u32,
    qk_head_dim: u32,
    v_head_dim: u32,
) {
    out[0..4].copy_from_slice(&lid.to_be_bytes());
    out[4..8].copy_from_slice(&past_seq_len.to_be_bytes());
    out[8..12].copy_from_slice(&num_heads.to_be_bytes());
    out[12..16].copy_from_slice(&qk_head_dim.to_be_bytes());
    out[16..20].copy_from_slice(&v_head_dim.to_be_bytes());
}

fn read_kv_header(bytes: &[u8; KV_MIGRATION_HEADER_BYTES]) -> (u32, u32, u32, u32, u32) {
    (
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
    )
}

/// Decoded per-layer KV-migration payload as it arrives off the wire.
/// `body_bytes` is the raw `K then V` payload (f32 LE); the receiver
/// is expected to feed it straight into `Runner::install_kv_slab` after
/// re-prepending the matching per-layer header. (See
/// `kv_slab_layer_to_bytes` for the helper that does this assembly.)
#[derive(Clone, Debug)]
pub struct KvMigrationLayer {
    pub lid: u32,
    pub past_seq_len: u32,
    pub num_heads: u32,
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    pub body_bytes: Vec<u8>,
}

impl KvMigrationLayer {
    /// Re-serialize this single-layer payload into the `Vec<u8>` layout
    /// expected by `Runner::install_kv_slab`. Cheap — one Vec
    /// concatenation, no tensor decoding.
    pub fn into_install_slab(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(KV_MIGRATION_HEADER_BYTES + self.body_bytes.len());
        let mut hdr = [0u8; KV_MIGRATION_HEADER_BYTES];
        write_kv_header(
            &mut hdr,
            self.lid,
            self.past_seq_len,
            self.num_heads,
            self.qk_head_dim,
            self.v_head_dim,
        );
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&self.body_bytes);
        out
    }
}

/// Send one `KvMigration` frame downstream. Caller must have already
/// quiesced both ranks (no in-flight Forward). The body is a single
/// layer's KV state — call this once per layer being migrated.
pub async fn send_kv_migration(
    cli: &Mutex<ActivationClient>,
    lid: u32,
    past_seq_len: u32,
    num_heads: u32,
    qk_head_dim: u32,
    v_head_dim: u32,
    kv_body: &[u8],
) -> TransportResult<()> {
    let mut header = [0u8; 4 + KV_MIGRATION_HEADER_BYTES];
    header[0..4].copy_from_slice(&(FrameKind::KvMigration as u32).to_be_bytes());
    let mut hdr = [0u8; KV_MIGRATION_HEADER_BYTES];
    write_kv_header(
        &mut hdr,
        lid,
        past_seq_len,
        num_heads,
        qk_head_dim,
        v_head_dim,
    );
    header[4..].copy_from_slice(&hdr);
    // Bound check: with very large past_seq_len, the per-layer body
    // can exceed MAX_TENSOR_BYTES. Fail loudly here rather than let
    // `recv_tensor` reject the peer later.
    let body_total = (num_heads as u64)
        .saturating_mul(past_seq_len as u64)
        .saturating_mul((qk_head_dim + v_head_dim) as u64)
        .saturating_mul(4);
    if body_total > tahoma_transport::MAX_TENSOR_BYTES as u64 {
        return Err(TransportError::PayloadTooLarge(body_total));
    }
    if kv_body.len() as u64 != body_total {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "send_kv_migration L{lid}: body length {} != expected {}",
            kv_body.len(),
            body_total
        ))));
    }
    // Wrap the body as an I8 tensor so it rides the standard tensor
    // wire (length-prefixed, capped at MAX_TENSOR_BYTES). I8 is the
    // smallest element type, so `shape * 1 == byte len` round-trips
    // cleanly through `recv_tensor`'s shape-vs-bytes invariant.
    let body_len_u32 = kv_body.len() as u32;
    let tensor = Tensor::new(DType::I8, [1, 1, body_len_u32], kv_body.to_vec());
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Receive one `KvMigration` frame's body on the server side (kind
/// already consumed by `recv_kind_server`).
pub async fn recv_kv_migration_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<KvMigrationLayer> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(KV_MIGRATION_HEADER_BYTES).await?;
    if raw.len() != KV_MIGRATION_HEADER_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let mut hdr = [0u8; KV_MIGRATION_HEADER_BYTES];
    hdr.copy_from_slice(&raw);
    let (lid, ps, nh, qkd, vd) = read_kv_header(&hdr);
    let (tensor, _stats) = guard.recv().await?;
    drop(guard);
    validate_kv_body(&tensor, ps, nh, qkd, vd)?;
    Ok(KvMigrationLayer {
        lid,
        past_seq_len: ps,
        num_heads: nh,
        qk_head_dim: qkd,
        v_head_dim: vd,
        body_bytes: tensor.data,
    })
}

/// Receive one `KvMigration` frame on the client side (used when a
/// migration is initiated upstream-bound — symmetric to
/// `recv_kv_migration_body_server`).
pub async fn recv_kv_migration_body_client(
    cli: &Mutex<ActivationClient>,
) -> TransportResult<KvMigrationLayer> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(KV_MIGRATION_HEADER_BYTES).await?;
    if raw.len() != KV_MIGRATION_HEADER_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let mut hdr = [0u8; KV_MIGRATION_HEADER_BYTES];
    hdr.copy_from_slice(&raw);
    let (lid, ps, nh, qkd, vd) = read_kv_header(&hdr);
    let (tensor, _stats) = guard.recv().await?;
    drop(guard);
    validate_kv_body(&tensor, ps, nh, qkd, vd)?;
    Ok(KvMigrationLayer {
        lid,
        past_seq_len: ps,
        num_heads: nh,
        qk_head_dim: qkd,
        v_head_dim: vd,
        body_bytes: tensor.data,
    })
}

fn validate_kv_body(
    tensor: &Tensor,
    past_seq_len: u32,
    num_heads: u32,
    qk_head_dim: u32,
    v_head_dim: u32,
) -> TransportResult<()> {
    if tensor.dtype != DType::I8 {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "kv migration: expected I8 carrier tensor, got {:?}",
            tensor.dtype
        ))));
    }
    let expected = (num_heads as u64)
        .saturating_mul(past_seq_len as u64)
        .saturating_mul((qk_head_dim + v_head_dim) as u64)
        .saturating_mul(4);
    if tensor.data.len() as u64 != expected {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "kv migration: body bytes {} != expected {}",
            tensor.data.len(),
            expected
        ))));
    }
    Ok(())
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
