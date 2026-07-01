//! QUIC backend for the activation relay (feature `quic`).
//!
//! The pipeline framing (see the crate docs) is transport-agnostic: it is a
//! plain ordered byte stream. QUIC gives us exactly that via a **single
//! long-lived bidirectional stream** per connection, so the identical
//! [`send_tensor`](crate::send_tensor) / [`recv_tensor`](crate::recv_tensor)
//! framing rides a `quinn` [`SendStream`]/[`RecvStream`] unchanged.
//!
//! Design choices, and why:
//!
//! * **One bidi stream, not many.** The relay is a strictly-ordered
//!   request/response byte pipe between two adjacent stages. QUIC's stream
//!   multiplexing buys nothing here, so we mirror the TCP model: one stream,
//!   framed by hand. The client `open_bi()`s it; the server `accept_bi()`s.
//! * **1-byte connect preamble.** A QUIC stream is invisible to the peer
//!   until the opener writes to it, and `accept_bi()` blocks until then.
//!   The client always sends first in this protocol, but stage *readiness*
//!   must fire at connection time (like a TCP accept), not at first-request
//!   time — otherwise a middle stage's `connect()` would block until real
//!   traffic flows. The client writes one [`PREAMBLE`] byte immediately
//!   after `open_bi()`; the server reads+discards it in `accept()`, giving
//!   TCP-parity accept semantics.
//! * **Keep-alive < idle-timeout.** Pipeline stages are idle between
//!   requests by design. Without keep-alive, QUIC's idle timeout would tear
//!   an idle connection down. [`KEEP_ALIVE_INTERVAL`] PINGs reset the idle
//!   timer on both ends, so a live-but-idle link stays up indefinitely,
//!   while a genuinely dead peer is still detected within
//!   [`MAX_IDLE_TIMEOUT`] (like a TCP RST/EOF, only faster than a stalled
//!   TCP read).
//! * **Self-signed + skip-verify.** Matches the locked zero-config P2P
//!   design: every node mints an ephemeral self-signed cert; the client
//!   accepts any cert. This encrypts the wire but does NOT authenticate the
//!   peer (MITM-able) — acceptable for a v1 opt-in on trusted/tailscale
//!   fabrics; peer authentication (pinned RPK) is a documented follow-up.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{ClientConfig, Connection, Endpoint, RecvStream, SendStream, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime};

use crate::TransportError;

/// ALPN protocol id. Set on both ends; a mismatched peer is rejected at
/// the TLS layer.
const ALPN: &[u8] = b"cascadia-quic/1";

/// Written by the client right after `open_bi()` and consumed by the
/// server in `accept()` — see the module docs.
const PREAMBLE: u8 = 0x1c;

/// PING cadence that keeps an idle relay alive. Must stay comfortably
/// below [`MAX_IDLE_TIMEOUT`].
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(10);

/// Idle-timeout after which a silent peer's connection is torn down. With
/// keep-alive at 10 s, a live link never reaches this; a dead peer does.
const MAX_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-attempt bound on the QUIC handshake. UDP has no connection-refused,
/// so a not-yet-listening downstream would otherwise hang until the idle
/// timeout; this lets the caller's retry loop re-poll quickly instead.
const HANDSHAKE_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(2);

/// The server name presented in the TLS SNI. Arbitrary (the client skips
/// verification) but must parse as a DNS name.
const SERVER_NAME: &str = "cascadia";

fn quic_err(context: &str, e: impl std::fmt::Display) -> TransportError {
    TransportError::Quic(format!("{context}: {e}"))
}

/// Shared transport params (keep-alive + idle timeout + flow control).
fn transport_config() -> Arc<quinn::TransportConfig> {
    let mut tc = quinn::TransportConfig::default();
    tc.keep_alive_interval(Some(KEEP_ALIVE_INTERVAL));
    // try_from only fails for absurd (multi-year) durations; ours is 30 s.
    if let Ok(idle) = quinn::IdleTimeout::try_from(MAX_IDLE_TIMEOUT) {
        tc.max_idle_timeout(Some(idle));
    }
    // Generous flow-control windows. With quinn's defaults a large activation
    // frame stalls on MAX_STREAM_DATA / MAX_DATA credit round-trips — measured
    // ~2.5 RTT for a 64 KiB frame at 20-50 ms RTT (netem). The relay is one
    // long-lived connection per stage carrying frames up to MAX_TENSOR_BYTES,
    // so a fixed multi-MiB window (cheap memory) lets a frame stream in ~1 RTT.
    let stream_win: u32 = 16 * 1024 * 1024; // 16 MiB per stream
    let conn_win: u32 = 32 * 1024 * 1024; // 32 MiB connection-wide
    tc.stream_receive_window(stream_win.into());
    tc.receive_window(conn_win.into());
    tc.send_window(conn_win as u64);
    Arc::new(tc)
}

/// Explicit ring provider — avoids depending on a process-default
/// `CryptoProvider` being installed (rustls 0.23 requires one otherwise).
///
/// `CASCADIA_QUIC_CIPHER` optionally pins the TLS 1.3 AEAD to `aesgcm` or
/// `chacha`. Diagnostic knob: lets a benchmark measure crypto's share of the
/// per-byte cost. Unset = rustls' normal preference (AES-GCM where AES-NI is
/// present, which is every CPU we target).
fn ring_provider() -> Arc<rustls::crypto::CryptoProvider> {
    use rustls::crypto::ring::cipher_suite::{
        TLS13_AES_128_GCM_SHA256, TLS13_CHACHA20_POLY1305_SHA256,
    };
    let mut provider = rustls::crypto::ring::default_provider();
    // AES-128-GCM must stay present regardless — QUIC derives its *initial*
    // packet keys from it. The negotiated 1-RTT suite is the list head.
    match std::env::var("CASCADIA_QUIC_CIPHER").as_deref() {
        Ok("aesgcm") | Ok("aes") => {
            provider.cipher_suites = vec![TLS13_AES_128_GCM_SHA256];
        }
        Ok("chacha") | Ok("chacha20") => {
            provider.cipher_suites = vec![TLS13_CHACHA20_POLY1305_SHA256, TLS13_AES_128_GCM_SHA256];
        }
        _ => {}
    }
    Arc::new(provider)
}

/// Build a QUIC server config from a fresh ephemeral self-signed cert.
fn server_config() -> Result<ServerConfig, TransportError> {
    let cert = rcgen::generate_simple_self_signed(vec![SERVER_NAME.to_string()])
        .map_err(|e| quic_err("self-signed cert", e))?;
    let cert_der: CertificateDer<'static> = cert.cert.der().clone();
    let key = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

    let mut crypto = rustls::ServerConfig::builder_with_provider(ring_provider())
        .with_safe_default_protocol_versions()
        .map_err(|e| quic_err("server tls versions", e))?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key.into())
        .map_err(|e| quic_err("server single cert", e))?;
    crypto.alpn_protocols = vec![ALPN.to_vec()];

    let qsc = QuicServerConfig::try_from(crypto).map_err(|e| quic_err("quic server crypto", e))?;
    let mut sc = ServerConfig::with_crypto(Arc::new(qsc));
    sc.transport_config(transport_config());
    Ok(sc)
}

/// Build a QUIC client config that accepts any server cert (skip-verify).
fn client_config() -> Result<ClientConfig, TransportError> {
    let mut crypto = rustls::ClientConfig::builder_with_provider(ring_provider())
        .with_safe_default_protocol_versions()
        .map_err(|e| quic_err("client tls versions", e))?
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    crypto.alpn_protocols = vec![ALPN.to_vec()];

    let qcc = QuicClientConfig::try_from(crypto).map_err(|e| quic_err("quic client crypto", e))?;
    let mut cc = ClientConfig::new(Arc::new(qcc));
    cc.transport_config(transport_config());
    Ok(cc)
}

/// Resolve `host:port` to a single socket address (first result).
pub(crate) async fn resolve(host: &str, port: u16) -> Result<SocketAddr, TransportError> {
    tokio::net::lookup_host((host, port))
        .await
        .map_err(TransportError::Io)?
        .next()
        .ok_or_else(|| TransportError::Quic(format!("no address for {host}:{port}")))
}

/// An unspecified bind address in the same family as `target`, so a v6
/// downstream gets a v6 client socket and a v4 downstream a v4 one.
fn unspecified_like(target: SocketAddr) -> SocketAddr {
    match target {
        SocketAddr::V4(_) => "0.0.0.0:0".parse().unwrap(),
        SocketAddr::V6(_) => "[::]:0".parse().unwrap(),
    }
}

/// A live QUIC connection carrying the one relay stream. Holds the
/// `Connection` so its background driver stays alive for the stream's life.
pub(crate) struct QuicConn {
    pub send: SendStream,
    pub recv: RecvStream,
    _conn: Connection,
}

impl QuicConn {
    /// Signal end-of-stream to the peer. Best-effort; the connection then
    /// winds down when both ends drop or the idle timer fires.
    ///
    /// Note: unlike a TCP close (which flushes OS buffers via FIN), dropping
    /// a QUIC `Connection` emits an immediate `CONNECTION_CLOSE` that can
    /// truncate still-in-flight stream data. That is safe here because the
    /// relay connection lives for the whole pipeline session and is only
    /// closed after the final frame has been consumed by the reader — the
    /// same lifetime the TCP path relies on.
    pub(crate) fn shutdown(&mut self) {
        let _ = self.send.finish();
    }
}

/// Bound QUIC server endpoint (the listener half).
pub(crate) struct QuicListener {
    endpoint: Endpoint,
}

impl QuicListener {
    pub(crate) async fn bind(host: &str, port: u16) -> Result<Self, TransportError> {
        let addr = resolve(host, port).await?;
        let endpoint =
            Endpoint::server(server_config()?, addr).map_err(|e| quic_err("quic bind", e))?;
        Ok(Self { endpoint })
    }

    pub(crate) fn local_port(&self) -> u16 {
        self.endpoint
            .local_addr()
            .map(|a| a.port())
            .unwrap_or_default()
    }

    /// Accept one connection and its single bidi stream, then consume the
    /// client's connect preamble so the first real frame reads clean.
    pub(crate) async fn accept(&self) -> Result<(QuicConn, SocketAddr), TransportError> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| TransportError::Quic("endpoint closed".into()))?;
        let conn = incoming.await.map_err(|e| quic_err("quic accept", e))?;
        let addr = conn.remote_address();
        let (send, mut recv) = conn
            .accept_bi()
            .await
            .map_err(|e| quic_err("quic accept_bi", e))?;
        // Consume the 1-byte preamble the client sent to materialise the
        // stream. quinn's inherent `RecvStream::read_exact` errors if the
        // peer finished/reset the stream before sending it.
        let mut pre = [0u8; 1];
        recv.read_exact(&mut pre)
            .await
            .map_err(|e| quic_err("quic preamble", e))?;
        Ok((
            QuicConn {
                send,
                recv,
                _conn: conn,
            },
            addr,
        ))
    }
}

/// Bound QUIC client endpoint. Kept alive by the caller for the life of
/// the connection so the connection driver keeps running.
pub(crate) struct QuicClientEndpoint {
    endpoint: Endpoint,
}

impl QuicClientEndpoint {
    /// Bind an ephemeral client socket in the same address family as the
    /// downstream target and install the skip-verify client config.
    pub(crate) async fn bind_for(target: SocketAddr) -> Result<Self, TransportError> {
        let mut endpoint = Endpoint::client(unspecified_like(target))
            .map_err(|e| quic_err("quic client bind", e))?;
        endpoint.set_default_client_config(client_config()?);
        Ok(Self { endpoint })
    }

    /// One connect attempt: QUIC handshake, open the relay stream, send the
    /// preamble. Bounded by [`HANDSHAKE_ATTEMPT_TIMEOUT`] so the caller's
    /// retry loop re-polls quickly against a not-yet-listening peer.
    pub(crate) async fn connect_once(&self, addr: SocketAddr) -> Result<QuicConn, TransportError> {
        let connecting = self
            .endpoint
            .connect(addr, SERVER_NAME)
            .map_err(|e| quic_err("quic connect", e))?;
        let conn = match tokio::time::timeout(HANDSHAKE_ATTEMPT_TIMEOUT, connecting).await {
            Ok(res) => res.map_err(|e| quic_err("quic handshake", e))?,
            Err(_) => return Err(TransportError::Quic("handshake timed out".into())),
        };
        let (mut send, recv) = conn
            .open_bi()
            .await
            .map_err(|e| quic_err("quic open_bi", e))?;
        use tokio::io::AsyncWriteExt;
        send.write_all(&[PREAMBLE])
            .await
            .map_err(|e| quic_err("quic preamble", e))?;
        send.flush().await.map_err(|e| quic_err("quic flush", e))?;
        Ok(QuicConn {
            send,
            recv,
            _conn: conn,
        })
    }
}

/// Certificate verifier that accepts any server cert. Encrypts the wire
/// without authenticating the peer — see the module docs' security note.
/// Structure taken from quinn's `insecure_connection.rs` example.
#[derive(Debug)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self(ring_provider()))
    }
}

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}
