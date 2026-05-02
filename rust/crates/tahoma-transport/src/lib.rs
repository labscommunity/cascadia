//! TCP-based activation tensor relay between pipeline stages.
//!
//! Wire format (big-endian, identical to `tahoma/worker/transport.py`):
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

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("socket closed during recv")]
    SocketClosed,

    #[error("tensor rank > {MAX_RANK} not supported (got {0} dims)")]
    RankTooHigh(usize),

    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("connect timed out after {0:?}")]
    ConnectTimeout(Duration),

    #[error("server not started; call start() before accept()")]
    NotStarted,

    #[error("not connected; call connect()/accept() first")]
    NotConnected,
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

    pub fn elements(&self) -> u64 {
        self.shape.iter().map(|d| *d as u64).product()
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
pub async fn recv_tensor(sock: &mut TcpStream) -> TransportResult<(Tensor, TransferStats)> {
    let start = Instant::now();
    let mut header = [0u8; HEADER_SIZE];
    recv_exact(sock, &mut header).await?;

    let payload_len = u32::from_be_bytes(header[0..4].try_into().unwrap());
    let dtype_code = u32::from_be_bytes(header[4..8].try_into().unwrap());
    let d0 = u32::from_be_bytes(header[8..12].try_into().unwrap());
    let d1 = u32::from_be_bytes(header[12..16].try_into().unwrap());
    let d2 = u32::from_be_bytes(header[16..20].try_into().unwrap());

    let mut data = vec![0u8; payload_len as usize];
    recv_exact(sock, &mut data).await?;

    let tensor = Tensor::new(DType::from_code(dtype_code), [d0, d1, d2], data);
    let stats = TransferStats {
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        bytes: HEADER_SIZE + payload_len as usize,
    };
    Ok((tensor, stats))
}

async fn recv_exact(sock: &mut TcpStream, buf: &mut [u8]) -> TransportResult<()> {
    let mut read = 0;
    while read < buf.len() {
        let n = sock.read(&mut buf[read..]).await?;
        if n == 0 {
            return Err(TransportError::SocketClosed);
        }
        read += n;
    }
    Ok(())
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
        recv_tensor(sock).await
    }

    pub async fn send(&mut self, tensor: &Tensor) -> TransportResult<TransferStats> {
        let sock = self.client.as_mut().ok_or(TransportError::NotConnected)?;
        send_tensor(sock, tensor).await
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
        let deadline = Instant::now() + timeout;
        let mut last_err: Option<io::Error> = None;
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
        send_tensor(sock, tensor).await
    }

    pub async fn recv(&mut self) -> TransportResult<(Tensor, TransferStats)> {
        let sock = self.sock.as_mut().ok_or(TransportError::NotConnected)?;
        recv_tensor(sock).await
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
}
