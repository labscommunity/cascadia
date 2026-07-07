//! TCP-based activation tensor relay between pipeline stages.
//!
//! Wire format (big-endian, identical to `cascadia/worker/transport.py`):
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

    /// Frame-start idle ceiling fired: the peer is connected but silent
    /// (black-holed). Connection-fatal — the server/client wrappers drop
    /// the socket so a frame the peer sends later can never be read into
    /// a different request; subsequent calls fail fast with
    /// [`Self::NotConnected`].
    #[error("frame-start idle ceiling hit after {0:?}; connection dropped (black-holed peer?)")]
    FrameIdleCeiling(Duration),
}

pub type TransportResult<T> = Result<T, TransportError>;

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
}

/// Timing + bytes for a single send or recv.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransferStats {
    pub elapsed_ms: f64,
    pub bytes: usize,
}

/// Send a tensor over a connected stream.
pub async fn send_tensor(sock: &mut TcpStream, tensor: &Tensor) -> TransportResult<TransferStats> {
    let start = Instant::now();
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&(tensor.data.len() as u32).to_be_bytes());
    header[4..8].copy_from_slice(&(tensor.dtype as u32).to_be_bytes());
    header[8..12].copy_from_slice(&tensor.shape[0].to_be_bytes());
    header[12..16].copy_from_slice(&tensor.shape[1].to_be_bytes());
    header[16..20].copy_from_slice(&tensor.shape[2].to_be_bytes());

    sock.write_all(&header).await?;
    sock.write_all(&tensor.data).await?;
    sock.flush().await?;

    Ok(TransferStats {
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        bytes: HEADER_SIZE + tensor.data.len(),
    })
}

/// Receive a tensor from a connected stream.
///
/// Bounds the per-call allocation at [`MAX_TENSOR_BYTES`] to defend
/// against a malicious or malformed peer claiming a multi-GB payload.
/// Also cross-checks the per-element count against the declared
/// payload length so a peer can't claim `shape=[u32::MAX, ...]` to
/// trigger overflow downstream.
pub async fn recv_tensor(sock: &mut TcpStream) -> TransportResult<(Tensor, TransferStats)> {
    recv_tensor_inner(sock, None).await
}

/// Like [`recv_tensor`] but for a MID-TASK reply: the peer owes us a prompt
/// response to a frame we just sent (e.g. the pipeline tail returning the
/// sampled token for a hidden state), so the header's FIRST byte is read
/// under the strict recv timeout instead of the idle-tolerant wait (bounded
/// only by the much larger `frame_idle_ceiling`). Call-site rule: a stage
/// waiting for the NEXT task idles on `recv` (idle-ceiling bound only —
/// "no work yet" is not a failure); a stage waiting for a reply to
/// in-flight work uses `recv_reply` — otherwise a single frame lost
/// mid-task (e.g. a pipeline-leg reset between stages) stalls the engine's
/// step loop for the whole idle ceiling with the task slot held
/// (overload-backlog Item 5: forwarded-to head wedges, task never
/// finalizes).
pub async fn recv_tensor_reply(sock: &mut TcpStream) -> TransportResult<(Tensor, TransferStats)> {
    recv_tensor_inner(sock, Some(recv_timeout())).await
}

/// A PREFILL reply is owed only after every remaining downstream stage has
/// run whole-prompt inference sequentially — the wait scales with prompt
/// length × pipeline depth, not with a single frame transfer (which is what
/// the base recv timeout was sized for). Budget: this factor × the base
/// recv timeout, sized to comfortably cover whole-prompt compute across the
/// deepest pipelines we run; decode replies (sub-second when healthy) keep
/// the strict [`recv_tensor_reply`] deadline so wedge eviction stays fast
/// where it matters.
pub const PREFILL_REPLY_TIMEOUT_FACTOR: u32 = 10;

/// [`recv_tensor_reply`] with the widened prefill budget. Use for the token
/// reply to a multi-token (prefill) hidden state; everything else uses
/// `recv_tensor_reply`.
pub async fn recv_tensor_reply_prefill(
    sock: &mut TcpStream,
) -> TransportResult<(Tensor, TransferStats)> {
    // saturating_mul: an absurdly large configured base must clamp, not
    // panic the engine thread (Duration's Mul panics on overflow).
    recv_tensor_inner(
        sock,
        Some(recv_timeout().saturating_mul(PREFILL_REPLY_TIMEOUT_FACTOR)),
    )
    .await
}

async fn recv_tensor_inner(
    sock: &mut TcpStream,
    deadline_first_byte: Option<Duration>,
) -> TransportResult<(Tensor, TransferStats)> {
    let start = Instant::now();
    let mut header = [0u8; HEADER_SIZE];
    match deadline_first_byte {
        Some(to) => recv_exact_within(sock, &mut header, to).await?,
        None => recv_exact_frame_start(sock, &mut header).await?,
    }

    let payload_len = u32::from_be_bytes(header[0..4].try_into().unwrap());
    let dtype_code = u32::from_be_bytes(header[4..8].try_into().unwrap());
    let d0 = u32::from_be_bytes(header[8..12].try_into().unwrap());
    let d1 = u32::from_be_bytes(header[12..16].try_into().unwrap());
    let d2 = u32::from_be_bytes(header[16..20].try_into().unwrap());

    if (payload_len as usize) > MAX_TENSOR_BYTES {
        return Err(TransportError::PayloadTooLarge(payload_len as u64));
    }

    // Sanity check shape × dtype.bytes_per_element against payload_len
    // to catch malformed/forged headers before we allocate. Use
    // checked_mul to avoid silent u64 wrap on adversarial shapes.
    let dtype = DType::from_code(dtype_code);
    let elems = (d0 as u64)
        .checked_mul(d1 as u64)
        .and_then(|x| x.checked_mul(d2 as u64));
    if let Some(e) = elems {
        let expected = e.checked_mul(dtype.bytes_per_element() as u64);
        if expected != Some(payload_len as u64) {
            return Err(TransportError::PayloadTooLarge(payload_len as u64));
        }
    } else {
        return Err(TransportError::PayloadTooLarge(payload_len as u64));
    }

    let mut data = vec![0u8; payload_len as usize];
    recv_exact(sock, &mut data).await?;

    let tensor = Tensor::new(dtype, [d0, d1, d2], data);
    let stats = TransferStats {
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        bytes: HEADER_SIZE + payload_len as usize,
    };
    Ok((tensor, stats))
}

/// Config override (seconds) for the activation recv timeout; 0 = unset.
static ACTIVATION_TIMEOUT_SECS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Set the activation recv timeout from node config; takes precedence over the env var.
/// Process-global, last writer wins — one value per process, not per shard.
pub fn set_activation_timeout_secs(secs: u64) {
    ACTIVATION_TIMEOUT_SECS.store(secs, std::sync::atomic::Ordering::Relaxed);
}

/// Precedence: config override > env > default. Pure, for testing.
fn resolve_recv_timeout(explicit_secs: u64, env_secs: Option<u64>) -> Duration {
    if explicit_secs > 0 {
        Duration::from_secs(explicit_secs)
    } else {
        env_secs.map(Duration::from_secs).unwrap_or(DEFAULT_TIMEOUT)
    }
}

/// Per-hop activation recv timeout: config > `CASCADIA_ACTIVATION_TIMEOUT_SECS` > 60s.
/// Public so a driver awaiting an owed reply can bound the frame-start wait
/// that [`recv_raw`] intentionally exempts for idle relays.
pub fn recv_timeout() -> Duration {
    use std::sync::OnceLock;
    static ENV: OnceLock<Option<u64>> = OnceLock::new(); // env read once
    let env = *ENV.get_or_init(|| {
        std::env::var("CASCADIA_ACTIVATION_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
    });
    resolve_recv_timeout(
        ACTIVATION_TIMEOUT_SECS.load(std::sync::atomic::Ordering::Relaxed),
        env,
    )
}

/// Default frame-start idle ceiling. Generous enough for legitimate long
/// gaps between frames (cold compiles between stages, idle chains between
/// requests) which the strict per-frame recv timeout must not kill; small
/// enough that a black-holed peer — connected but silent, no FIN/RST —
/// eventually surfaces as an error instead of pinning the stage forever.
pub const DEFAULT_FRAME_IDLE_CEILING: Duration = Duration::from_secs(900);

/// Config override for the frame-start idle ceiling, stored as secs+1 so
/// 0 can mean "unset" while a configured 0 ("no ceiling") stays expressible.
static FRAME_IDLE_CEILING_SECS_PLUS_ONE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Set the frame-start idle ceiling from node config (seconds); takes
/// precedence over the env var. `0` disables the ceiling entirely (the
/// historical unbounded idle wait).
pub fn set_frame_idle_ceiling_secs(secs: u64) {
    FRAME_IDLE_CEILING_SECS_PLUS_ONE
        .store(secs.saturating_add(1), std::sync::atomic::Ordering::Relaxed);
}

/// Precedence: config override > env > default; 0 at any level = no ceiling.
/// Pure, for testing.
fn resolve_frame_idle_ceiling(stored_plus_one: u64, env_secs: Option<u64>) -> Option<Duration> {
    let secs = if stored_plus_one > 0 {
        stored_plus_one - 1
    } else {
        match env_secs {
            Some(s) => s,
            None => return Some(DEFAULT_FRAME_IDLE_CEILING),
        }
    };
    (secs > 0).then(|| Duration::from_secs(secs))
}

/// An enabled ceiling below the strict per-frame recv timeout would make
/// frame-START stricter than mid-frame (design inversion) and silently
/// re-create the premature idle-kill the ceiling exists to avoid — floor
/// the effective value at the recv timeout. `None` (no ceiling) passes
/// through. Pure, for testing.
fn clamp_frame_idle_ceiling(
    configured: Option<Duration>,
    recv_timeout: Duration,
) -> Option<Duration> {
    configured.map(|c| c.max(recv_timeout))
}

/// Whether a recv error leaves the socket unusable for subsequent reads, so
/// the owner must drop it. Cases, all leaving the link dead or frame
/// alignment lost:
///
/// * frame-START idle ceiling ([`TransportError::FrameIdleCeiling`]) — the
///   peer is connected but silent; a frame it sends later must never land in
///   the next request (token frames carry no task id).
/// * MID-frame deadline — a frame began but stalled past the strict
///   [`recv_timeout`]; [`recv_exact`] surfaces this as `Io(TimedOut)`,
///   leaving a half-consumed frame on the wire that the next recv would read
///   as a corrupt header.
/// * peer crash — a process dying hard sends TCP RST (and a send/half-close
///   races as BrokenPipe/ConnectionAborted/UnexpectedEof). These surface as
///   `Io(ConnectionReset | BrokenPipe | ConnectionAborted | UnexpectedEof)`;
///   the socket is dead, so drop it now and let the next call fail fast with
///   [`TransportError::NotConnected`] (the dominant dead-peer case).
///
/// A clean EOF is NOT fatal here: it surfaces as
/// [`TransportError::SocketClosed`], needs no drop, and the next call fails
/// cleanly on its own. Pure, for testing.
fn recv_error_is_connection_fatal(err: &TransportError) -> bool {
    match err {
        TransportError::FrameIdleCeiling(_) => true,
        TransportError::Io(e) => matches!(
            e.kind(),
            io::ErrorKind::TimedOut
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::BrokenPipe
                | io::ErrorKind::ConnectionAborted
                | io::ErrorKind::UnexpectedEof
        ),
        _ => false,
    }
}

/// Frame-start idle ceiling: config > `CASCADIA_FRAME_IDLE_CEILING_SECS` > 900s,
/// floored at [`recv_timeout`] when enabled (warns once when the floor engages).
fn frame_idle_ceiling() -> Option<Duration> {
    use std::sync::OnceLock;
    static ENV: OnceLock<Option<u64>> = OnceLock::new(); // env read once
    let env = *ENV.get_or_init(|| {
        std::env::var("CASCADIA_FRAME_IDLE_CEILING_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
    });
    let configured = resolve_frame_idle_ceiling(
        FRAME_IDLE_CEILING_SECS_PLUS_ONE.load(std::sync::atomic::Ordering::Relaxed),
        env,
    );
    let effective = clamp_frame_idle_ceiling(configured, recv_timeout());
    if effective != configured {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            warn!(
                configured = ?configured,
                effective = ?effective,
                "frame idle ceiling below the activation recv timeout; clamping up"
            );
        }
    }
    effective
}

/// Wait for the first byte of a NEW frame exempt from the per-frame recv
/// timeout (but still bounded by the much larger [`frame_idle_ceiling`], not
/// unbounded), then read the remainder under the strict recv timeout. A
/// pipeline stage is idle between requests by design — "no next frame yet"
/// is not a failure, and treating it as one made every idle chain kill its
/// own sockets on a timer. A dead peer still fails fast here via EOF/reset;
/// a silent black-hole peer (connected, no FIN/RST) is bounded only by the
/// much larger [`frame_idle_ceiling`], surfaced as the dedicated
/// [`TransportError::FrameIdleCeiling`] — distinguishable from the
/// per-frame deadline's `Io(TimedOut)` so the wrappers can treat it as
/// connection-fatal.
async fn recv_exact_frame_start(sock: &mut TcpStream, buf: &mut [u8]) -> TransportResult<()> {
    let first_read = sock.read(buf);
    let n = match frame_idle_ceiling() {
        Some(ceiling) => match tokio::time::timeout(ceiling, first_read).await {
            Ok(res) => res?,
            Err(_) => return Err(TransportError::FrameIdleCeiling(ceiling)),
        },
        None => first_read.await?,
    };
    if n == 0 {
        return Err(TransportError::SocketClosed);
    }
    if n < buf.len() {
        recv_exact(sock, &mut buf[n..]).await?;
    }
    Ok(())
}

async fn recv_exact(sock: &mut TcpStream, buf: &mut [u8]) -> TransportResult<()> {
    // DEFAULT_TIMEOUT bounds total wall-clock time we'll wait for `buf`
    // to fill. A peer that opens a connection and stops sending — or
    // sends one byte per second — must not be able to pin a worker
    // thread forever. 60 s is generous for a single tensor frame on a
    // multi-MB Llama hidden state over Thunderbolt, and small enough
    // that a wedged peer is detected within one or two heartbeats.
    // Waiting for a frame to BEGIN is exempt — see recv_exact_frame_start.
    //
    // NOTE: this is a single wall-clock bound over the WHOLE remaining
    // buffer, not an idle/no-progress timeout. Assumption: inter-stage
    // frames are small (hidden states, KB–MB), so 60 s wall-clock ≈ a
    // genuine stall, not a slow-but-progressing transfer. A mid-frame
    // timeout is now connection-fatal (the socket is dropped), so a
    // slow-but-progressing LARGE transfer on a degraded link would be
    // killed — acceptable at the current frame sizes. If large mid-frame
    // transfers ever land (e.g. KV-cache blobs), switch to an idle/
    // no-progress timeout (reset the deadline on each read that returns
    // bytes) so progress, not total size, is what's bounded.
    recv_exact_within(sock, buf, recv_timeout()).await
}

async fn recv_exact_within(
    sock: &mut TcpStream,
    buf: &mut [u8],
    to: Duration,
) -> TransportResult<()> {
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
    match tokio::time::timeout(to, read_fut).await {
        Ok(res) => res,
        Err(_) => Err(TransportError::Io(io::Error::new(
            io::ErrorKind::TimedOut,
            format!("recv_exact timed out after {to:?}"),
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
        }
    }

    pub async fn start(&mut self) -> TransportResult<()> {
        let listener = TcpListener::bind((self.bind_host.as_str(), self.bind_port)).await?;
        self.actual_port = listener.local_addr()?.port();
        self.listener = Some(listener);
        info!(
            host = %self.bind_host,
            port = self.actual_port,
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
        let res = recv_tensor(sock).await;
        self.drop_connection_if_recv_fatal(res.as_ref().err());
        res
    }

    /// A recv timeout is connection-fatal: drop the socket here, at the
    /// transport layer, so a half-consumed or abandoned frame can never be
    /// read into a different request (token frames carry no task id). Covers
    /// both the frame-start idle ceiling and a mid-frame stall — see
    /// [`recv_error_is_connection_fatal`]. Subsequent calls fail fast with
    /// [`TransportError::NotConnected`].
    fn drop_connection_if_recv_fatal(&mut self, err: Option<&TransportError>) {
        if err.is_some_and(recv_error_is_connection_fatal) {
            self.client = None; // drop closes the fd
            self.accepted_addr = None;
        }
    }

    /// Mid-task reply recv — strict deadline on the first byte. See
    /// [`recv_tensor_reply`] for the call-site rule. A failed reply
    /// poisons the connection (see `poison`: the socket is dropped
    /// and later calls fail fast with `NotConnected`; recover with a
    /// fresh connection).
    pub async fn recv_reply(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        let res = recv_tensor_reply(sock).await;
        if res.is_err() {
            self.poison().await;
        }
        res
    }

    /// Prefill-budget reply recv — see [`recv_tensor_reply_prefill`].
    /// A failed reply poisons the connection (see `poison`: the socket
    /// is dropped and later calls fail fast with `NotConnected`;
    /// recover with a fresh connection).
    pub async fn recv_reply_prefill(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        let res = recv_tensor_reply_prefill(sock).await;
        if res.is_err() {
            self.poison().await;
        }
        res
    }

    /// A failed reply leaves the stream in an unknown framing state: a
    /// late-but-healthy reply may still arrive and would be consumed by the
    /// NEXT task as fresh data (silent cross-task token corruption), and a
    /// partially-read header misaligns every later frame. Drop + shutdown
    /// the connection so later calls fail fast with `NotConnected` instead
    /// of corrupting — recovery is a fresh connection, not reuse.
    async fn poison(&mut self) {
        if let Some(mut s) = self.client.take() {
            let _ = s.shutdown().await;
        }
        self.accepted_addr = None;
    }

    pub async fn send(&mut self, tensor: &Tensor) -> TransportResult<TransferStats> {
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        send_tensor(sock, tensor).await
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
        let res = recv_exact_frame_start(sock, &mut buf).await;
        self.drop_connection_if_recv_fatal(res.as_ref().err());
        res?;
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
}

impl ActivationClient {
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            sock: None,
        }
    }

    /// Connect with retries until `timeout` elapses (mirrors the Python
    /// implementation's wait-for-peer behaviour during pipeline startup).
    pub async fn connect_with_timeout(&mut self, timeout: Duration) -> TransportResult<()> {
        let start = Instant::now();
        let deadline = start + timeout;
        // Tell the operator up-front what we're waiting on. Without this
        // the worker looks hung for up to `timeout` with no output — the
        // single most-reported "is it broken?" moment in multi-node
        // bring-up. We log the target and the budget so the wait is
        // legible, then a progress line every few seconds.
        info!(
            host = %self.host,
            port = self.port,
            timeout_s = timeout.as_secs(),
            "waiting for downstream peer to accept (start the downstream worker first)"
        );
        let mut last_err: Option<io::Error> = None;
        let mut next_progress = start + Duration::from_secs(5);
        while Instant::now() < deadline {
            match TcpStream::connect((self.host.as_str(), self.port)).await {
                Ok(sock) => {
                    sock.set_nodelay(true).ok();
                    info!(host = %self.host, port = self.port, "ActivationClient connected");
                    self.sock = Some(sock);
                    return Ok(());
                }
                Err(err) => {
                    last_err = Some(err);
                    let now = Instant::now();
                    if now >= next_progress {
                        warn!(
                            host = %self.host,
                            port = self.port,
                            waited_s = now.duration_since(start).as_secs(),
                            timeout_s = timeout.as_secs(),
                            "still waiting for downstream peer (not accepting yet)"
                        );
                        next_progress = now + Duration::from_secs(5);
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
        // Actionable timeout: name the address and the usual causes so an
        // operator doesn't have to reverse-engineer a bare io::Error.
        warn!(
            host = %self.host,
            port = self.port,
            timeout_s = timeout.as_secs(),
            last_error = ?last_err.as_ref().map(|e| e.to_string()),
            "could not connect to downstream peer within timeout — check that the \
             downstream worker is running, that its --listen port matches this \
             --next, and that no firewall blocks the port"
        );
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
        send_tensor(sock, tensor).await
    }

    pub async fn recv(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        let res = recv_tensor(sock).await;
        self.drop_connection_if_recv_fatal(res.as_ref().err());
        res
    }

    /// See [`ActivationServer::drop_connection_if_recv_fatal`]: a recv
    /// timeout (frame-start ceiling or mid-frame stall) is connection-fatal
    /// at the transport layer.
    fn drop_connection_if_recv_fatal(&mut self, err: Option<&TransportError>) {
        if err.is_some_and(recv_error_is_connection_fatal) {
            self.sock = None; // drop closes the fd
        }
    }

    /// Mid-task reply recv — strict deadline on the first byte. See
    /// [`recv_tensor_reply`] for the call-site rule. A failed reply
    /// poisons the connection (see `poison`: the socket is dropped
    /// and later calls fail fast with `NotConnected`; recover with a
    /// fresh connection).
    pub async fn recv_reply(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        let res = recv_tensor_reply(sock).await;
        if res.is_err() {
            self.poison().await;
        }
        res
    }

    /// Prefill-budget reply recv — see [`recv_tensor_reply_prefill`].
    /// A failed reply poisons the connection (see `poison`: the socket
    /// is dropped and later calls fail fast with `NotConnected`;
    /// recover with a fresh connection).
    pub async fn recv_reply_prefill(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        let res = recv_tensor_reply_prefill(sock).await;
        if res.is_err() {
            self.poison().await;
        }
        res
    }

    /// See [`ActivationServer::poison`]: a failed reply means unknown
    /// framing state — fail fast on reuse instead of corrupting.
    async fn poison(&mut self) {
        if let Some(mut s) = self.sock.take() {
            let _ = s.shutdown().await;
        }
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
        let res = recv_exact_frame_start(sock, &mut buf).await;
        self.drop_connection_if_recv_fatal(res.as_ref().err());
        res?;
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

    #[test]
    fn recv_timeout_precedence_config_over_env_over_default() {
        // explicit config override (node.toml) wins over env + default
        assert_eq!(
            resolve_recv_timeout(120, Some(30)),
            Duration::from_secs(120)
        );
        // no config -> env wins over default
        assert_eq!(resolve_recv_timeout(0, Some(90)), Duration::from_secs(90));
        // neither -> 60s default
        assert_eq!(resolve_recv_timeout(0, None), DEFAULT_TIMEOUT);
    }

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

    /// Serializes tests that mutate the global activation timeout.
    static TIMEOUT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A pipeline tail is idle between requests by design; waiting for the
    /// NEXT frame longer than the recv timeout must not kill the socket.
    #[tokio::test]
    async fn idle_gap_longer_than_timeout_does_not_kill_recv() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(1);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server.recv().await
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        tokio::time::sleep(Duration::from_millis(2500)).await; // idle > timeout
        let tensor = Tensor::from_2d(DType::F32, 1, 2, vec![0, 0, 128, 63, 0, 0, 0, 64]);
        client.send(&tensor).await.unwrap();
        let got = h.await.unwrap();
        set_activation_timeout_secs(0);
        assert!(
            got.is_ok(),
            "idle wait for the next frame must not time out: {:?}",
            got.err()
        );
    }

    /// A MID-TASK reply (`recv_reply`) is the opposite contract: the peer
    /// owes us a prompt response, so a silent peer must fail fast instead
    /// of blocking the engine's step loop forever with the task slot held
    /// (overload-backlog Item 5).
    #[tokio::test]
    async fn reply_wait_longer_than_timeout_fails_fast() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(1);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            let started = std::time::Instant::now();
            let res = server.recv_reply().await;
            (res, started.elapsed())
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        // Send nothing: the "reply" never comes.
        let (got, waited) = h.await.unwrap();
        set_activation_timeout_secs(0);
        assert!(
            got.is_err(),
            "a missing mid-task reply must time out, got {got:?}"
        );
        assert!(
            waited < Duration::from_secs(5),
            "must fail within ~the recv timeout, waited {waited:?}"
        );
    }

    /// Happy path: a reply arriving within the deadline is received normally.
    #[tokio::test]
    async fn reply_within_timeout_succeeds() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(2);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server.recv_reply().await
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        let tensor = Tensor::from_2d(DType::F32, 1, 2, vec![0, 0, 128, 63, 0, 0, 0, 64]);
        client.send(&tensor).await.unwrap();
        let got = h.await.unwrap();
        set_activation_timeout_secs(0);
        assert!(
            got.is_ok(),
            "in-deadline reply must succeed: {:?}",
            got.err()
        );
    }

    /// A timed-out reply leaves the stream in an unknown framing state — a
    /// late-but-healthy reply could otherwise be consumed by the NEXT task
    /// as fresh data (silent cross-task corruption). The connection must be
    /// poisoned: later recvs fail fast with NotConnected.
    #[tokio::test]
    async fn reply_timeout_poisons_connection() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(1);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            let first = server.recv_reply().await;
            let second = server.recv().await;
            (first, second)
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        tokio::time::sleep(Duration::from_millis(2500)).await; // miss the 1s deadline
        let tensor = Tensor::from_2d(DType::F32, 1, 2, vec![0, 0, 128, 63, 0, 0, 0, 64]);
        let _ = client.send(&tensor).await; // late reply — server may have hung up
        let (first, second) = h.await.unwrap();
        set_activation_timeout_secs(0);
        assert!(
            first.is_err(),
            "the late reply must time out, got {first:?}"
        );
        assert!(
            matches!(second, Err(TransportError::NotConnected)),
            "a poisoned connection must fail fast, not serve the stale frame: {second:?}"
        );
    }

    /// A prefill reply legitimately includes the remaining stages'
    /// whole-prompt compute — it gets the widened budget, not the
    /// single-frame deadline.
    #[tokio::test]
    async fn prefill_reply_outlives_base_timeout() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(1);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server.recv_reply_prefill().await
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        // > 1s base deadline, < the 10x prefill budget.
        tokio::time::sleep(Duration::from_millis(2500)).await;
        let tensor = Tensor::from_2d(DType::F32, 1, 2, vec![0, 0, 128, 63, 0, 0, 0, 64]);
        client.send(&tensor).await.unwrap();
        let got = h.await.unwrap();
        set_activation_timeout_secs(0);
        assert!(
            got.is_ok(),
            "a slow-but-in-budget prefill reply must succeed: {:?}",
            got.err()
        );
    }

    /// Once a frame has started, a stalled peer must still hit the timeout
    /// (slow-loris / wedged-peer bound is preserved).
    #[tokio::test]
    async fn mid_frame_stall_still_times_out() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(1);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server.recv().await
        });
        let mut sock = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        sock.write_all(&[0u8; 4]).await.unwrap(); // partial header, then stall
        sock.flush().await.unwrap();
        let got = h.await.unwrap();
        set_activation_timeout_secs(0);
        assert!(got.is_err(), "partial frame then stall must still time out");
    }

    #[test]
    fn frame_idle_ceiling_precedence_config_over_env_over_default() {
        // unset everywhere -> generous default
        assert_eq!(
            resolve_frame_idle_ceiling(0, None),
            Some(DEFAULT_FRAME_IDLE_CEILING)
        );
        // no config -> env wins; env 0 = no ceiling
        assert_eq!(
            resolve_frame_idle_ceiling(0, Some(30)),
            Some(Duration::from_secs(30))
        );
        assert_eq!(resolve_frame_idle_ceiling(0, Some(0)), None);
        // config (stored as secs+1) wins over env; configured 0 = no ceiling
        assert_eq!(
            resolve_frame_idle_ceiling(6, Some(30)),
            Some(Duration::from_secs(5))
        );
        assert_eq!(resolve_frame_idle_ceiling(1, Some(30)), None);
    }

    #[test]
    fn ceiling_clamped_to_recv_timeout_when_enabled() {
        // ceiling below the recv timeout -> floored at the recv timeout
        assert_eq!(
            clamp_frame_idle_ceiling(Some(Duration::from_secs(5)), Duration::from_secs(60)),
            Some(Duration::from_secs(60))
        );
        // ceiling above the recv timeout -> unchanged
        assert_eq!(
            clamp_frame_idle_ceiling(Some(Duration::from_secs(900)), Duration::from_secs(60)),
            Some(Duration::from_secs(900))
        );
        // 0 = unbounded stays unbounded; the clamp never re-enables it
        assert_eq!(
            clamp_frame_idle_ceiling(None, Duration::from_secs(60)),
            None
        );
        // raised activation timeout drags the default ceiling up with it
        // (frame-start must never be stricter than mid-frame)
        assert_eq!(
            clamp_frame_idle_ceiling(Some(DEFAULT_FRAME_IDLE_CEILING), Duration::from_secs(1200)),
            Some(Duration::from_secs(1200))
        );
    }

    /// A black-holed peer (connected, silent, no FIN/RST) must hit the
    /// frame-start idle ceiling instead of blocking recv forever.
    #[tokio::test]
    async fn frame_start_idle_ceiling_fires_on_silent_peer() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_frame_idle_ceiling_secs(1);
        set_activation_timeout_secs(1); // keep the 1s ceiling under the clamp
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server.recv().await
        });
        // Connect, then go silent — never send a byte.
        let _sock = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let got = h.await.unwrap();
        set_frame_idle_ceiling_secs(0);
        set_activation_timeout_secs(0);
        match got {
            Err(TransportError::FrameIdleCeiling(_)) => {}
            other => panic!("expected FrameIdleCeiling, got {other:?}"),
        }
    }

    /// A ceiling fire must kill the connection: the next recv on the same
    /// server fails fast with NotConnected, and a frame the black-holed
    /// peer sends late must never be readable as a valid frame (it would
    /// otherwise leak into the NEXT request — token frames carry no task
    /// id).
    #[tokio::test]
    async fn ceiling_fire_is_connection_fatal() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_frame_idle_ceiling_secs(1);
        set_activation_timeout_secs(1);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        // Connect, then go silent — never send a byte.
        let mut peer = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        server.accept().await.unwrap();
        let first = server.recv().await;
        assert!(
            matches!(first, Err(TransportError::FrameIdleCeiling(_))),
            "expected FrameIdleCeiling, got {first:?}"
        );
        // The abandoned frame arrives LATE, after the ceiling fired.
        let tensor = Tensor::from_2d(DType::F32, 1, 2, vec![0, 0, 128, 63, 0, 0, 0, 64]);
        let _ = send_tensor(&mut peer, &tensor).await; // peer may already see RST
                                                       // Subsequent use must fail fast on a dead connection — the late
                                                       // frame must NOT come back as a valid frame.
        let start = Instant::now();
        let second = server.recv().await;
        set_frame_idle_ceiling_secs(0);
        set_activation_timeout_secs(0);
        assert!(
            matches!(second, Err(TransportError::NotConnected)),
            "recv after ceiling fire must fail NotConnected, got {second:?}"
        );
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "recv after ceiling fire must fail fast"
        );
    }

    #[test]
    fn recv_error_fatal_classification() {
        // Frame-start ceiling: fatal (black-holed peer).
        assert!(recv_error_is_connection_fatal(
            &TransportError::FrameIdleCeiling(Duration::from_secs(1))
        ));
        // Mid-frame deadline surfaces as Io(TimedOut): fatal (half frame
        // left on the wire would desync the next read).
        assert!(recv_error_is_connection_fatal(&TransportError::Io(
            io::Error::new(io::ErrorKind::TimedOut, "recv_exact timed out")
        )));
        // Clean EOF: NOT fatal — the next call fails cleanly on its own.
        assert!(!recv_error_is_connection_fatal(
            &TransportError::SocketClosed
        ));
        // Peer crash: RST / broken pipe / aborted / unexpected EOF all leave
        // the socket dead — fatal so the owner drops it (the dominant
        // dead-peer case).
        for kind in [
            io::ErrorKind::ConnectionReset,
            io::ErrorKind::BrokenPipe,
            io::ErrorKind::ConnectionAborted,
            io::ErrorKind::UnexpectedEof,
        ] {
            assert!(
                recv_error_is_connection_fatal(&TransportError::Io(io::Error::new(kind, "peer"))),
                "expected fatal: {kind:?}"
            );
        }
        // An unrelated Io kind (e.g. would-block) is NOT fatal here.
        assert!(!recv_error_is_connection_fatal(&TransportError::Io(
            io::Error::new(io::ErrorKind::WouldBlock, "retryable")
        )));
    }

    /// A mid-frame stall (header arrives, payload never does, past the recv
    /// timeout) must be connection-fatal: the half-consumed frame leaves the
    /// wire desynced, so the next recv must fail fast with NotConnected, and
    /// a late completion of that frame must never be read back as valid.
    #[tokio::test]
    async fn mid_frame_stall_is_connection_fatal() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(1);
        // Keep the frame-start ceiling well above the recv timeout so the
        // first read (the header) is NOT what fires — we want the mid-frame
        // deadline to be the trigger.
        set_frame_idle_ceiling_secs(60);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let mut peer = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        server.accept().await.unwrap();

        // Send a complete, valid header for an 8-byte f32 [1,1,2] payload,
        // then stall — never send the payload. recv_exact_frame_start reads
        // the header, recv_tensor then blocks in recv_exact for the body.
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&8u32.to_be_bytes()); // payload_len
        header[4..8].copy_from_slice(&(DType::F32 as u32).to_be_bytes());
        header[8..12].copy_from_slice(&1u32.to_be_bytes());
        header[12..16].copy_from_slice(&1u32.to_be_bytes());
        header[16..20].copy_from_slice(&2u32.to_be_bytes());
        peer.write_all(&header).await.unwrap();
        peer.flush().await.unwrap();

        let first = server.recv().await;
        assert!(
            matches!(&first, Err(TransportError::Io(e)) if e.kind() == io::ErrorKind::TimedOut),
            "expected mid-frame Io(TimedOut), got {first:?}"
        );

        // The peer finally sends the rest of the abandoned frame, late.
        peer.write_all(&[0, 0, 128, 63, 0, 0, 0, 64]).await.ok();
        peer.flush().await.ok();

        // The socket must already be dead: the next recv fails fast with
        // NotConnected, NOT a corrupt frame read from the late payload.
        let start = Instant::now();
        let second = server.recv().await;
        set_activation_timeout_secs(0);
        set_frame_idle_ceiling_secs(0);
        assert!(
            matches!(second, Err(TransportError::NotConnected)),
            "recv after mid-frame stall must fail NotConnected, got {second:?}"
        );
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "recv after mid-frame stall must fail fast (socket already dropped)"
        );
    }

    /// A peer process dying hard sends TCP RST, surfaced on the recv side as
    /// `Io(ConnectionReset)` (the dominant dead-peer case). The owner must
    /// treat that as connection-fatal and drop the socket, so the next recv
    /// fails fast with NotConnected instead of retrying a dead link. Drives
    /// the owner's drop path with the synthesized error a real RST produces
    /// (forcing a true RST portably needs the unstable `set_linger` or an
    /// extra socket crate). The kind→fatal mapping itself is proven by
    /// `recv_error_fatal_classification`.
    #[tokio::test]
    async fn peer_rst_drops_socket_fast_fail() {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let _peer = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        server.accept().await.unwrap();
        assert!(server.client.is_some(), "precondition: connection accepted");

        // The recv read returned ConnectionReset (peer RST). The owner drops
        // the socket on this.
        server.drop_connection_if_recv_fatal(Some(&TransportError::Io(io::Error::new(
            io::ErrorKind::ConnectionReset,
            "connection reset by peer",
        ))));
        assert!(
            server.client.is_none(),
            "peer RST must drop the socket so the next call fails fast"
        );

        let start = Instant::now();
        let next = server.recv().await;
        assert!(
            matches!(next, Err(TransportError::NotConnected)),
            "recv after peer RST must fail NotConnected, got {next:?}"
        );
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "recv after peer RST must fail fast"
        );
    }

    /// The ceiling must not erode the idle-tolerance guarantee: an idle
    /// gap longer than the strict recv timeout but under the (generous)
    /// ceiling still delivers the next frame intact.
    #[tokio::test]
    async fn generous_idle_ceiling_does_not_fire_on_idle_gap() {
        let _g = TIMEOUT_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_activation_timeout_secs(1);
        set_frame_idle_ceiling_secs(60);
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            server.recv().await
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        tokio::time::sleep(Duration::from_millis(2500)).await; // idle > timeout
        let tensor = Tensor::from_2d(DType::F32, 1, 2, vec![0, 0, 128, 63, 0, 0, 0, 64]);
        client.send(&tensor).await.unwrap();
        let got = h.await.unwrap();
        set_activation_timeout_secs(0);
        set_frame_idle_ceiling_secs(0);
        assert!(
            got.is_ok(),
            "idle gap under the ceiling must not time out: {:?}",
            got.err()
        );
    }
}
