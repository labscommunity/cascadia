//! TCP-based activation tensor relay between pipeline stages.
//!
//! Wire format (big-endian, identical to `tahoma/worker/transport.py`
//! when compression is `None`):
//!
//! ```text
//! [4B payload_len] [4B dtype_code] [4B dim0] [4B dim1] [4B dim2]
//! [payload_len bytes raw row-major data]
//! ```
//!
//! dtype codes: `0=f32`, `1=f16`, `2=i8`, `3=i32`, `4=i64`.
//!
//! Tensors up to 3D supported. Lower-rank tensors are wire-padded with
//! leading-1 dimensions; receiver returns the wire-encoded shape.
//!
//! ## Compression
//!
//! Opt-in per-tensor compression overloads the high byte of `dtype_code`
//! (the 4-byte BE field at offset 4) as a compression flag. The on-disk
//! layout becomes:
//!
//! ```text
//! [0..4]   payload_len   u32 BE   - compressed payload length
//! [4]      compression   u8       - 0=None, 1=Zstd, 2=Lz4
//! [5..7]   reserved      [u8;3]   - zero
//! [7]      dtype         u8       - 0..=4 (low byte of original dtype_code)
//! [8..12]  dim0          u32 BE
//! [12..16] dim1          u32 BE
//! [16..20] dim2          u32 BE
//! [20..]   compressed body bytes
//! ```
//!
//! When `compression == None` the high byte stays zero and the four
//! bytes parse identically to the old `dtype_code` u32 (values 0..=4),
//! so a pre-compression sender and receiver are byte-compatible with
//! the new code path. When `compression != None`, both sides must be
//! running the new code AND must agree on the compression scheme — old
//! receivers will misread the compression flag as a bogus dtype code
//! and fail downstream (acceptable: compression is opt-in via CLI).
//!
//! For compressed payloads, the receiver knows the expected
//! uncompressed length from `shape × dtype.bytes_per_element()` and
//! refuses to over-allocate. There is no separate uncompressed-length
//! field on the wire.
//!
//! This is intentionally simple — raw TCP, point-to-point. It is the
//! data plane between adjacent pipeline stages.

use std::io;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

pub const HEADER_SIZE: usize = 20;
pub const MAX_RANK: usize = 3;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum tensor payload accepted on the wire. Caps the worst-case
/// allocation when reading a length-prefixed tensor from an untrusted
/// peer. 256 MiB is far above any legitimate hidden-state shard for
/// the model sizes we run (Llama 3.1 8B INT4 ⇒ logits f16
/// `4 * 128k * 2 = 1 MiB`; 70B-class hidden states even at very long
/// sequence lengths fit well under 256 MiB).
pub const MAX_TENSOR_BYTES: usize = 256 * 1024 * 1024;

/// Maximum bytes accepted by [`ActivationServer::recv_raw`] /
/// [`ActivationClient::recv_raw`] in a single call. The dist-spec
/// frame protocol uses recv_raw for control bytes (4 or 8 bytes); a
/// generous cap here defends against a peer claiming to send a
/// gigabyte of "control bytes".
pub const MAX_RAW_BYTES: usize = 64 * 1024;

/// zstd compression level used by [`Compression::Zstd`]. Level 1 is
/// the fastest zstd setting; on f32 hidden states it still hits
/// ~50-65% ratio while staying inside ~1 GB/s encode on a single
/// core. Higher levels (3, 6) gain only a few percent of ratio at
/// 2-4x the encode cost — not worth the per-token latency for our
/// 7 KiB hidden states.
pub const ZSTD_LEVEL: i32 = 1;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("socket closed during recv")]
    SocketClosed,

    #[error("tensor rank > {MAX_RANK} not supported (got {0} dims)")]
    RankTooHigh(usize),

    #[error("payload size {0} exceeds MAX_TENSOR_BYTES ({})", MAX_TENSOR_BYTES)]
    PayloadTooLarge(u64),

    #[error("recv_raw size {0} exceeds MAX_RAW_BYTES ({})", MAX_RAW_BYTES)]
    RawSizeTooLarge(usize),

    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("connect timed out after {0:?}")]
    ConnectTimeout(Duration),

    #[error("server not started; call start() before accept()")]
    NotStarted,

    #[error("not connected; call connect()/accept() first")]
    NotConnected,

    #[error("unknown compression code {0}")]
    UnknownCompression(u8),

    #[error("compression {scheme:?} failed: {detail}")]
    CompressionFailed { scheme: Compression, detail: String },

    #[error("decompression {scheme:?} failed: {detail}")]
    DecompressionFailed { scheme: Compression, detail: String },
}

pub type TransportResult<T> = Result<T, TransportError>;

/// Wire compression scheme. Configured at the [`ActivationServer`] /
/// [`ActivationClient`] level via [`ActivationServer::with_compression`]
/// / [`ActivationClient::with_compression`]; both peers must use the
/// same scheme (this crate does not negotiate — the CLI flag does).
///
/// - `None` — raw bytes; byte-compatible with the pre-compression wire.
/// - `Zstd` — `zstd` level [`ZSTD_LEVEL`], typically 50-65% on f32 hidden states.
/// - `Lz4` — `lz4_flex` block compression, typically 25-40% on f32 hidden
///   states but ~3x faster than zstd at level 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Compression {
    #[default]
    None = 0,
    Zstd = 1,
    Lz4 = 2,
}

impl Compression {
    pub fn from_code(code: u8) -> TransportResult<Self> {
        match code {
            0 => Ok(Self::None),
            1 => Ok(Self::Zstd),
            2 => Ok(Self::Lz4),
            other => Err(TransportError::UnknownCompression(other)),
        }
    }

    /// Parse from a lowercase CLI string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "off" | "no" => Some(Self::None),
            "zstd" => Some(Self::Zstd),
            "lz4" => Some(Self::Lz4),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd => "zstd",
            Self::Lz4 => "lz4",
        }
    }

    /// Compress `src` according to `self`. Returns `src.to_vec()` for
    /// `None` (callers should special-case None to skip the copy when
    /// they have ownership).
    pub fn compress(self, src: &[u8]) -> TransportResult<Vec<u8>> {
        match self {
            Self::None => Ok(src.to_vec()),
            Self::Zstd => zstd::bulk::compress(src, ZSTD_LEVEL).map_err(|e| {
                TransportError::CompressionFailed {
                    scheme: self,
                    detail: e.to_string(),
                }
            }),
            Self::Lz4 => Ok(lz4_flex::compress(src)),
        }
    }

    /// Decompress `src` into a buffer of exactly `expected_len` bytes.
    /// The caller supplies `expected_len` from the tensor's shape ×
    /// dtype, which both bounds the allocation (so a hostile peer can't
    /// claim 1 GiB unpacks from 100 B) and lets us validate output
    /// length.
    pub fn decompress(self, src: &[u8], expected_len: usize) -> TransportResult<Vec<u8>> {
        match self {
            Self::None => Ok(src.to_vec()),
            Self::Zstd => {
                let out = zstd::bulk::decompress(src, expected_len).map_err(|e| {
                    TransportError::DecompressionFailed {
                        scheme: self,
                        detail: e.to_string(),
                    }
                })?;
                if out.len() != expected_len {
                    return Err(TransportError::DecompressionFailed {
                        scheme: self,
                        detail: format!(
                            "zstd output length {} != expected {}",
                            out.len(),
                            expected_len
                        ),
                    });
                }
                Ok(out)
            }
            Self::Lz4 => {
                let out = lz4_flex::decompress(src, expected_len).map_err(|e| {
                    TransportError::DecompressionFailed {
                        scheme: self,
                        detail: e.to_string(),
                    }
                })?;
                if out.len() != expected_len {
                    return Err(TransportError::DecompressionFailed {
                        scheme: self,
                        detail: format!(
                            "lz4 output length {} != expected {}",
                            out.len(),
                            expected_len
                        ),
                    });
                }
                Ok(out)
            }
        }
    }
}

/// dtype codes — wire-compatible with the Python `DTYPE_MAP`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    I32 = 3,
    I64 = 4,
}

impl DType {
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Self::F16,
            2 => Self::I8,
            3 => Self::I32,
            4 => Self::I64,
            _ => Self::F32,
        }
    }
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::I64 => 8,
        }
    }
}

/// A 3-D tensor (lower ranks are padded with leading-1 dims on the wire).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: [u32; MAX_RANK],
    /// Row-major raw bytes; length = product(shape) * dtype.bytes_per_element().
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new(dtype: DType, shape: [u32; MAX_RANK], data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }

    pub fn from_2d(dtype: DType, rows: u32, cols: u32, data: Vec<u8>) -> Self {
        Self::new(dtype, [1, rows, cols], data)
    }

    /// Element count using checked multiplication. Returns `None` if
    /// the shape product would overflow `u64` (defense against
    /// adversarially-large shape headers).
    pub fn elements(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(*d as u64))
    }

    /// Expected raw payload length in bytes — `product(shape) * bpe`.
    /// Used as the decompression target length when compression != None.
    pub fn expected_raw_bytes(&self) -> Option<u64> {
        self.elements()
            .and_then(|e| e.checked_mul(self.dtype.bytes_per_element() as u64))
    }
}

/// Timing + bytes for a single send or recv.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransferStats {
    pub elapsed_ms: f64,
    /// Bytes that hit the wire (header + body). When compression is
    /// active this is the compressed body length plus the 20-byte header.
    pub bytes: usize,
    /// Compression scheme actually used on the wire for this transfer.
    pub compression: Compression,
    /// Body bytes before compression (None when compression is None or
    /// when this is the recv-side stats and the caller didn't pre-know
    /// the raw length). Equals `tensor.data.len()`.
    pub raw_body_bytes: Option<usize>,
}

/// Pack the (compression, dtype) pair into a 4-byte BE field that
/// occupies bytes 4..8 of the wire header.
///
/// Layout (network order):
///   [compression u8][0u8][0u8][dtype u8]
///
/// With `compression=None` the field equals `dtype as u32` — byte-
/// identical to the pre-compression wire format.
#[inline]
fn pack_dtype_field(compression: Compression, dtype: DType) -> u32 {
    ((compression as u32) << 24) | (dtype as u32 & 0xFF)
}

/// Inverse of `pack_dtype_field`. Returns `(compression, dtype)`.
/// Ignores the two middle reserved bytes; they're zero in our writer
/// and we don't enforce that on read so a future version can squeeze
/// flags in there without breaking us.
#[inline]
fn unpack_dtype_field(field: u32) -> TransportResult<(Compression, DType)> {
    let compression_code = ((field >> 24) & 0xFF) as u8;
    let dtype_code = field & 0xFF;
    let compression = Compression::from_code(compression_code)?;
    let dtype = DType::from_code(dtype_code);
    Ok((compression, dtype))
}

/// Send a tensor over a connected stream with the configured
/// compression scheme. The pre-compression call sites pass
/// `Compression::None`, producing a byte-identical wire frame to the
/// old code path.
pub async fn send_tensor_with(
    sock: &mut TcpStream,
    tensor: &Tensor,
    compression: Compression,
) -> TransportResult<TransferStats> {
    let start = Instant::now();
    let raw_body_bytes = tensor.data.len();

    let body: Vec<u8> = match compression {
        Compression::None => tensor.data.clone(),
        _ => compression.compress(&tensor.data)?,
    };
    // Defense-in-depth: a pathological input could produce a
    // compressed payload larger than our wire cap. Cap before sending
    // so the receiver's own cap rejection has a matching cap on the
    // sender side (avoids "I sent it, you rejected it" surprise).
    if body.len() > MAX_TENSOR_BYTES {
        return Err(TransportError::PayloadTooLarge(body.len() as u64));
    }

    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&(body.len() as u32).to_be_bytes());
    let dtype_field = pack_dtype_field(compression, tensor.dtype);
    header[4..8].copy_from_slice(&dtype_field.to_be_bytes());
    header[8..12].copy_from_slice(&tensor.shape[0].to_be_bytes());
    header[12..16].copy_from_slice(&tensor.shape[1].to_be_bytes());
    header[16..20].copy_from_slice(&tensor.shape[2].to_be_bytes());

    sock.write_all(&header).await?;
    sock.write_all(&body).await?;
    sock.flush().await?;

    Ok(TransferStats {
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        bytes: HEADER_SIZE + body.len(),
        compression,
        raw_body_bytes: Some(raw_body_bytes),
    })
}

/// Backward-compat alias for the no-compression send path. Existing
/// call sites work unchanged.
pub async fn send_tensor(sock: &mut TcpStream, tensor: &Tensor) -> TransportResult<TransferStats> {
    send_tensor_with(sock, tensor, Compression::None).await
}

/// Receive a tensor from a connected stream.
///
/// Reads the 20-byte header, parses the high byte of the dtype field
/// as a compression flag, decompresses if needed, and returns the
/// uncompressed `Tensor`.
///
/// Bounds the per-call allocation at [`MAX_TENSOR_BYTES`] to defend
/// against a malicious or malformed peer claiming a multi-GB payload.
/// Also cross-checks the per-element count against the declared
/// payload length so a peer can't claim `shape=[u32::MAX, ...]` to
/// trigger overflow downstream.
///
/// When the header indicates a compression scheme, the sanity check
/// is relaxed: we accept any `payload_len <= MAX_TENSOR_BYTES` and
/// then verify the *decompressed* length equals
/// `shape × bytes_per_element`. This catches a malformed-shape peer
/// even with compression on.
pub async fn recv_tensor(sock: &mut TcpStream) -> TransportResult<(Tensor, TransferStats)> {
    let start = Instant::now();
    let mut header = [0u8; HEADER_SIZE];
    recv_exact(sock, &mut header).await?;

    let payload_len = u32::from_be_bytes(header[0..4].try_into().unwrap());
    let dtype_field = u32::from_be_bytes(header[4..8].try_into().unwrap());
    let d0 = u32::from_be_bytes(header[8..12].try_into().unwrap());
    let d1 = u32::from_be_bytes(header[12..16].try_into().unwrap());
    let d2 = u32::from_be_bytes(header[16..20].try_into().unwrap());

    if (payload_len as usize) > MAX_TENSOR_BYTES {
        return Err(TransportError::PayloadTooLarge(payload_len as u64));
    }

    let (compression, dtype) = unpack_dtype_field(dtype_field)?;
    let elems = (d0 as u64)
        .checked_mul(d1 as u64)
        .and_then(|x| x.checked_mul(d2 as u64));
    let Some(elements) = elems else {
        return Err(TransportError::PayloadTooLarge(payload_len as u64));
    };
    let expected_raw = elements
        .checked_mul(dtype.bytes_per_element() as u64)
        .ok_or(TransportError::PayloadTooLarge(payload_len as u64))?;
    if expected_raw > MAX_TENSOR_BYTES as u64 {
        return Err(TransportError::PayloadTooLarge(expected_raw));
    }

    // Uncompressed: payload_len must match the shape-derived size,
    // exactly as the pre-compression code did. Compressed: payload_len
    // is the wire-bytes count; the decompression step validates the
    // expanded size.
    if matches!(compression, Compression::None) && expected_raw != payload_len as u64 {
        return Err(TransportError::PayloadTooLarge(payload_len as u64));
    }

    let mut data = vec![0u8; payload_len as usize];
    recv_exact(sock, &mut data).await?;

    let raw_body_bytes;
    let tensor_data = match compression {
        Compression::None => {
            raw_body_bytes = data.len();
            data
        }
        _ => {
            let out = compression.decompress(&data, expected_raw as usize)?;
            raw_body_bytes = out.len();
            out
        }
    };

    let tensor = Tensor::new(dtype, [d0, d1, d2], tensor_data);
    let stats = TransferStats {
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        bytes: HEADER_SIZE + payload_len as usize,
        compression,
        raw_body_bytes: Some(raw_body_bytes),
    };
    Ok((tensor, stats))
}

async fn recv_exact(sock: &mut TcpStream, buf: &mut [u8]) -> TransportResult<()> {
    // DEFAULT_TIMEOUT bounds total wall-clock time we'll wait for `buf`
    // to fill. A peer that opens a connection and stops sending — or
    // sends one byte per second — must not be able to pin a worker
    // thread forever. 60 s is generous for a single tensor frame on a
    // multi-MB Llama hidden state over Thunderbolt, and small enough
    // that a wedged peer is detected within one or two heartbeats.
    let read_fut = async {
        let mut read = 0;
        while read < buf.len() {
            let n = sock.read(&mut buf[read..]).await?;
            if n == 0 {
                return Err(TransportError::SocketClosed);
            }
            read += n;
        }
        Ok(())
    };
    match tokio::time::timeout(DEFAULT_TIMEOUT, read_fut).await {
        Ok(res) => res,
        Err(_) => Err(TransportError::Io(io::Error::new(
            io::ErrorKind::TimedOut,
            format!("recv_exact timed out after {DEFAULT_TIMEOUT:?}"),
        ))),
    }
}

/// TCP server that receives activations from upstream.
pub struct ActivationServer {
    bind_host: String,
    bind_port: u16,
    listener: Option<TcpListener>,
    client: Option<TcpStream>,
    accepted_addr: Option<SocketAddr>,
    actual_port: u16,
    compression: Compression,
}

impl ActivationServer {
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            bind_host: host.into(),
            bind_port: port,
            listener: None,
            client: None,
            accepted_addr: None,
            actual_port: port,
            compression: Compression::None,
        }
    }

    /// Set the compression scheme used by [`Self::send`]. [`Self::recv`]
    /// reads the per-frame flag from the wire and decompresses
    /// accordingly — it does not consult this field.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    pub fn set_compression(&mut self, compression: Compression) {
        self.compression = compression;
    }

    pub fn compression(&self) -> Compression {
        self.compression
    }

    pub async fn start(&mut self) -> TransportResult<()> {
        let listener = TcpListener::bind((self.bind_host.as_str(), self.bind_port)).await?;
        self.actual_port = listener.local_addr()?.port();
        self.listener = Some(listener);
        info!(
            host = %self.bind_host,
            port = self.actual_port,
            compression = self.compression.as_str(),
            "ActivationServer listening"
        );
        Ok(())
    }

    pub fn port(&self) -> u16 {
        self.actual_port
    }

    pub async fn accept(&mut self) -> TransportResult<()> {
        let listener = self.listener.as_ref().ok_or(TransportError::NotStarted)?;
        let (sock, addr) = listener.accept().await?;
        sock.set_nodelay(true).ok();
        info!(peer = %addr, "ActivationServer accepted connection");
        self.client = Some(sock);
        self.accepted_addr = Some(addr);
        Ok(())
    }

    pub async fn recv(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        recv_tensor(sock).await
    }

    pub async fn send(&mut self, tensor: &Tensor) -> TransportResult<TransferStats> {
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        send_tensor_with(sock, tensor, self.compression).await
    }

    /// Send raw bytes over the established connection. Used by the
    /// dist-spec engines to prefix tensor frames with control bytes
    /// (kind + logical_pos_start).
    pub async fn send_raw(&mut self, bytes: &[u8]) -> TransportResult<()> {
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        sock.write_all(bytes).await?;
        sock.flush().await?;
        Ok(())
    }

    /// Receive exactly `n` raw bytes from the established connection.
    /// Capped at [`MAX_RAW_BYTES`] to bound allocation when an
    /// untrusted caller picks `n`.
    pub async fn recv_raw(&mut self, n: usize) -> TransportResult<Vec<u8>> {
        if n > MAX_RAW_BYTES {
            return Err(TransportError::RawSizeTooLarge(n));
        }
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        let mut buf = vec![0u8; n];
        recv_exact(sock, &mut buf).await?;
        Ok(buf)
    }

    pub async fn close(&mut self) {
        if let Some(mut sock) = self.client.take() {
            let _ = sock.shutdown().await;
        }
        self.listener = None;
        self.accepted_addr = None;
    }
}

/// TCP client that sends activations to downstream.
pub struct ActivationClient {
    host: String,
    port: u16,
    sock: Option<TcpStream>,
    compression: Compression,
}

impl ActivationClient {
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            sock: None,
            compression: Compression::None,
        }
    }

    /// Set the compression scheme used by [`Self::send`]. [`Self::recv`]
    /// reads the per-frame flag from the wire and decompresses
    /// accordingly — it does not consult this field.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    pub fn set_compression(&mut self, compression: Compression) {
        self.compression = compression;
    }

    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// Connect with retries until `timeout` elapses (mirrors the Python
    /// implementation's wait-for-peer behaviour during pipeline startup).
    pub async fn connect_with_timeout(&mut self, timeout: Duration) -> TransportResult<()> {
        let deadline = Instant::now() + timeout;
        let mut last_err: Option<io::Error> = None;
        while Instant::now() < deadline {
            match TcpStream::connect((self.host.as_str(), self.port)).await {
                Ok(sock) => {
                    sock.set_nodelay(true).ok();
                    info!(
                        host = %self.host,
                        port = self.port,
                        compression = self.compression.as_str(),
                        "ActivationClient connected"
                    );
                    self.sock = Some(sock);
                    return Ok(());
                }
                Err(err) => {
                    last_err = Some(err);
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
        warn!(host = %self.host, port = self.port, "ActivationClient connect timed out");
        if let Some(err) = last_err {
            return Err(err.into());
        }
        Err(TransportError::ConnectTimeout(timeout))
    }

    pub async fn connect(&mut self) -> TransportResult<()> {
        self.connect_with_timeout(DEFAULT_CONNECT_TIMEOUT).await
    }

    pub async fn send(&mut self, tensor: &Tensor) -> TransportResult<TransferStats> {
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        send_tensor_with(sock, tensor, self.compression).await
    }

    pub async fn recv(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        recv_tensor(sock).await
    }

    pub async fn send_raw(&mut self, bytes: &[u8]) -> TransportResult<()> {
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        sock.write_all(bytes).await?;
        sock.flush().await?;
        Ok(())
    }

    pub async fn recv_raw(&mut self, n: usize) -> TransportResult<Vec<u8>> {
        if n > MAX_RAW_BYTES {
            return Err(TransportError::RawSizeTooLarge(n));
        }
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        let mut buf = vec![0u8; n];
        recv_exact(sock, &mut buf).await?;
        Ok(buf)
    }

    pub async fn close(&mut self) {
        if let Some(mut sock) = self.sock.take() {
            let _ = sock.shutdown().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn roundtrip_f32_2d() {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();

        let server_handle = tokio::spawn(async move {
            server.accept().await.unwrap();
            let (got, _) = server.recv().await.unwrap();
            got
        });

        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        let payload = vec![0u8, 0, 128, 63, 0, 0, 0, 64]; // f32: 1.0, 2.0
        let tensor = Tensor::from_2d(DType::F32, 1, 2, payload.clone());
        client.send(&tensor).await.unwrap();
        let got = server_handle.await.unwrap();
        assert_eq!(got.dtype, DType::F32);
        assert_eq!(got.shape, [1, 1, 2]);
        assert_eq!(got.data, payload);
    }

    #[tokio::test]
    async fn roundtrip_i64_3d() {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            let (got, _) = server.recv().await.unwrap();
            got
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        let payload: Vec<u8> = (0..2 * 3 * 4 * 8).map(|i| (i % 256) as u8).collect();
        let tensor = Tensor::new(DType::I64, [2, 3, 4], payload.clone());
        client.send(&tensor).await.unwrap();
        let got = h.await.unwrap();
        assert_eq!(got.shape, [2, 3, 4]);
        assert_eq!(got.data, payload);
    }

    #[tokio::test]
    async fn connect_timeout_to_unbound_port() {
        let mut client = ActivationClient::new("127.0.0.1", 1);
        let res = client
            .connect_with_timeout(Duration::from_millis(200))
            .await;
        assert!(res.is_err());
    }

    #[test]
    fn dtype_from_code() {
        assert_eq!(DType::from_code(0), DType::F32);
        assert_eq!(DType::from_code(1), DType::F16);
        assert_eq!(DType::from_code(2), DType::I8);
        assert_eq!(DType::from_code(3), DType::I32);
        assert_eq!(DType::from_code(4), DType::I64);
        assert_eq!(DType::from_code(99), DType::F32);
    }

    #[test]
    fn bytes_per_element() {
        assert_eq!(DType::F32.bytes_per_element(), 4);
        assert_eq!(DType::F16.bytes_per_element(), 2);
        assert_eq!(DType::I8.bytes_per_element(), 1);
        assert_eq!(DType::I32.bytes_per_element(), 4);
        assert_eq!(DType::I64.bytes_per_element(), 8);
    }

    /// Synthetic 7168-dim f32 hidden state at K2.6 scale — the
    /// distribution that drives our compression-ratio choice. Values
    /// drawn from a smooth pseudo-distribution in [-2, 2] to mimic
    /// real hidden-state stats (mean ≈ 0, mostly small magnitudes,
    /// long-tail outliers).
    fn synthetic_hidden_state() -> Vec<u8> {
        let n = 7168;
        let mut out = Vec::with_capacity(n * 4);
        let mut x = 0.1f32;
        for i in 0..n {
            // Pseudo-random in [-2, 2] with most mass near zero.
            x = (x * 1.31 + (i as f32) * 0.0007).sin();
            let v = x * (1.5 + 0.5 * ((i as f32) * 0.001).cos());
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn compression_parse_str() {
        assert_eq!(Compression::parse("none"), Some(Compression::None));
        assert_eq!(Compression::parse("NONE"), Some(Compression::None));
        assert_eq!(Compression::parse("off"), Some(Compression::None));
        assert_eq!(Compression::parse("zstd"), Some(Compression::Zstd));
        assert_eq!(Compression::parse("Lz4"), Some(Compression::Lz4));
        assert_eq!(Compression::parse("snappy"), None);
    }

    #[test]
    fn compression_roundtrip_zstd_bytes_identical() {
        let raw = synthetic_hidden_state();
        let comp = Compression::Zstd.compress(&raw).unwrap();
        let back = Compression::Zstd.decompress(&comp, raw.len()).unwrap();
        assert_eq!(back, raw);
        // Sanity: zstd actually compressed it on this distribution.
        assert!(
            comp.len() < raw.len(),
            "zstd output {} >= input {}",
            comp.len(),
            raw.len()
        );
    }

    #[test]
    fn compression_roundtrip_lz4_bytes_identical() {
        let raw = synthetic_hidden_state();
        let comp = Compression::Lz4.compress(&raw).unwrap();
        let back = Compression::Lz4.decompress(&comp, raw.len()).unwrap();
        assert_eq!(back, raw);
    }

    #[test]
    fn pack_unpack_dtype_field_none_back_compat() {
        // With compression=None the packed field must equal `dtype as u32` —
        // proves byte-identical wire compatibility with pre-PR-068 senders.
        for dtype in [DType::F32, DType::F16, DType::I8, DType::I32, DType::I64] {
            let field = pack_dtype_field(Compression::None, dtype);
            assert_eq!(field, dtype as u32);
            let (cp, dt) = unpack_dtype_field(field).unwrap();
            assert_eq!(cp, Compression::None);
            assert_eq!(dt, dtype);
        }
    }

    #[test]
    fn pack_unpack_dtype_field_with_compression() {
        for cp in [Compression::Zstd, Compression::Lz4] {
            for dtype in [DType::F32, DType::F16, DType::I64] {
                let field = pack_dtype_field(cp, dtype);
                let (got_cp, got_dt) = unpack_dtype_field(field).unwrap();
                assert_eq!(got_cp, cp);
                assert_eq!(got_dt, dtype);
            }
        }
    }

    #[test]
    fn unpack_dtype_field_rejects_unknown_compression() {
        // Forge a field with compression byte = 0x77 (unknown).
        let field = (0x77u32 << 24) | (DType::F32 as u32);
        let err = unpack_dtype_field(field).unwrap_err();
        assert!(matches!(err, TransportError::UnknownCompression(0x77)));
    }

    async fn wire_roundtrip(compression: Compression) -> Tensor {
        let mut server = ActivationServer::new("127.0.0.1", 0).with_compression(compression);
        server.start().await.unwrap();
        let port = server.port();
        let server_handle = tokio::spawn(async move {
            server.accept().await.unwrap();
            let (got, stats) = server.recv().await.unwrap();
            assert_eq!(stats.compression, compression);
            got
        });
        let mut client = ActivationClient::new("127.0.0.1", port).with_compression(compression);
        client.connect().await.unwrap();
        let payload = super::tests::synthetic_hidden_state();
        let tensor = Tensor::from_2d(DType::F32, 1, (payload.len() / 4) as u32, payload.clone());
        let send_stats = client.send(&tensor).await.unwrap();
        assert_eq!(send_stats.compression, compression);
        // We don't assert wire < raw here for non-None schemes — lz4
        // sometimes expands near-random f32 by ~0.5% (block headers >
        // savings on entropy-rich data). The real compression-ratio
        // claim is empirical (see bench_wire_compression.rs) and lives
        // outside the unit test.
        server_handle.await.unwrap()
    }

    #[tokio::test]
    async fn wire_roundtrip_none_bytes_identical() {
        let got = wire_roundtrip(Compression::None).await;
        assert_eq!(got.data, super::tests::synthetic_hidden_state());
    }

    #[tokio::test]
    async fn wire_roundtrip_zstd_bytes_identical() {
        let got = wire_roundtrip(Compression::Zstd).await;
        assert_eq!(got.data, super::tests::synthetic_hidden_state());
    }

    #[tokio::test]
    async fn wire_roundtrip_lz4_bytes_identical() {
        let got = wire_roundtrip(Compression::Lz4).await;
        assert_eq!(got.data, super::tests::synthetic_hidden_state());
    }

    /// Cross-version compatibility: a pre-compression sender (one that
    /// wrote bytes 4..8 as plain `dtype_code` u32) must still be
    /// understood by the new receiver. Simulate by writing the legacy
    /// wire format directly into a pair of TCP sockets.
    #[tokio::test]
    async fn legacy_sender_to_new_receiver() {
        use tokio::io::AsyncWriteExt;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let server_handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            // Pretend the server side has compression=Zstd configured,
            // but the inbound frame has compression byte = 0 (legacy
            // sender). The recv path consults the wire flag, not the
            // server config, so it must decode raw.
            let (tensor, stats) = recv_tensor(&mut sock).await.unwrap();
            assert_eq!(stats.compression, Compression::None);
            tensor
        });

        let mut client = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        // Build a legacy frame BY HAND: 20-byte header with dtype_code
        // as plain u32 (= 0 for F32), then 8 bytes of f32 payload.
        let payload: [u8; 8] = [0u8, 0, 128, 63, 0, 0, 0, 64]; // 1.0, 2.0
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&(payload.len() as u32).to_be_bytes());
        header[4..8].copy_from_slice(&0u32.to_be_bytes()); // legacy dtype_code = F32
        header[8..12].copy_from_slice(&1u32.to_be_bytes());
        header[12..16].copy_from_slice(&1u32.to_be_bytes());
        header[16..20].copy_from_slice(&2u32.to_be_bytes());
        client.write_all(&header).await.unwrap();
        client.write_all(&payload).await.unwrap();
        client.flush().await.unwrap();

        let got = server_handle.await.unwrap();
        assert_eq!(got.dtype, DType::F32);
        assert_eq!(got.shape, [1, 1, 2]);
        assert_eq!(got.data, payload);
    }

    /// Compressed payload that decompresses to the wrong size is rejected.
    #[test]
    fn decompress_size_mismatch_rejected() {
        let raw = vec![0u8; 64];
        let comp = Compression::Zstd.compress(&raw).unwrap();
        // Ask for 32 bytes back — zstd will reject (output overflow)
        // before returning a short buffer.
        let err = Compression::Zstd.decompress(&comp, 32).unwrap_err();
        assert!(matches!(err, TransportError::DecompressionFailed { .. }));
    }
}
