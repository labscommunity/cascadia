//! A physicalai remote-inference server whose calls go to an openpi
//! WebSocket server: Cascadia's own, or a gateway in front of one.
//!
//! One bridge serves one physicalai client on loopback. It answers the
//! handshake with a `server_id` minted for its own lifetime (an upstream
//! reconnect is not a restart to the client) and with the identity the
//! upstream's metadata states. Each predict becomes one openpi call; a call
//! still running when its `budget_ms` ends is answered `expired`, and its
//! late reply is read and dropped before the next call is sent.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cascadia_types::{Observation, PolicySpec};
use futures::{SinkExt, StreamExt};
use parking_lot::Mutex;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{header::AUTHORIZATION, HeaderValue};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{Connector, MaybeTlsStream, WebSocketStream};

use super::zenoh_server::{
    Advert, Backend, BackendError, Computed, ZenohError, ZenohFrontEnd, ZenohListen,
};
use super::{check_reply, decode_metadata, decode_reply, encode_request, PackageIdentity};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// How long the first connection may take.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// The upstream openpi server.
#[derive(Clone)]
pub struct Upstream {
    /// `ws://` or `wss://` URL of the policy socket.
    pub url: String,
    /// Sent as `Authorization: Bearer <token>`; never logged.
    pub token: Option<String>,
    /// PEM file of the CA certificates a `wss://` upstream is checked
    /// against, in place of the public web roots.
    pub ca_file: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for Upstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upstream")
            .field("url", &display_url(&self.url))
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("ca_file", &self.ca_file)
            .finish()
    }
}

/// `(scheme, authority, path)`; the authority keeps any userinfo, the path
/// drops the query and fragment.
fn split_url(url: &str) -> (&str, &str, &str) {
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(end);
    let path = &tail[..tail.find(['?', '#']).unwrap_or(tail.len())];
    (scheme, authority, path)
}

/// The URL as it may be logged: `scheme://host:port/path`, without userinfo,
/// query or fragment, any of which can carry a credential.
pub fn display_url(url: &str) -> String {
    let (scheme, authority, path) = split_url(url);
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    format!("{scheme}://{host}{path}")
}

fn host_is_loopback(url: &str) -> bool {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or_default(),
        None => host.split(':').next().unwrap_or_default(),
    };
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

impl Upstream {
    /// Refuse a URL that is not a WebSocket, and a token over plain `ws://`
    /// to anything but loopback.
    pub fn check(&self) -> Result<(), ZenohError> {
        let tls = self.url.starts_with("wss://");
        if !tls && !self.url.starts_with("ws://") {
            return Err(ZenohError(
                "--upstream must be a ws:// or wss:// URL".into(),
            ));
        }
        if split_url(&self.url).1.contains('@') {
            return Err(ZenohError(
                "--upstream must not carry user:password@; put the credential in --token-file"
                    .into(),
            ));
        }
        if self.token.is_some() && !tls && !host_is_loopback(&self.url) {
            return Err(ZenohError(
                "a token is only sent over wss://, or ws:// to loopback".into(),
            ));
        }
        if self.ca_file.is_some() && !tls {
            return Err(ZenohError(
                "--ca-file applies to a wss:// upstream only".into(),
            ));
        }
        Ok(())
    }

    /// Open the socket and read the metadata it sends first. `tls` replaces
    /// the public web roots (`--ca-file`).
    async fn connect(
        &self,
        tls: Option<&Connector>,
    ) -> Result<(Ws, PolicySpec, PackageIdentity), String> {
        let mut req = self
            .url
            .as_str()
            .into_client_request()
            .map_err(|e| format!("upstream URL: {}", redact(&e.to_string(), self)))?;
        if let Some(t) = &self.token {
            let mut v = HeaderValue::from_str(&format!("Bearer {t}"))
                .map_err(|_| "the token is not a valid header value".to_string())?;
            v.set_sensitive(true);
            req.headers_mut().insert(AUTHORIZATION, v);
        }
        // The error may echo the request; keep only its kind.
        let connected = match tls {
            Some(c) => {
                tokio_tungstenite::connect_async_tls_with_config(req, None, false, Some(c.clone()))
                    .await
            }
            None => tokio_tungstenite::connect_async(req).await,
        };
        let (mut ws, _) = connected.map_err(|e| {
            format!(
                "connect {}: {}",
                display_url(&self.url),
                redact(&e.to_string(), self)
            )
        })?;
        loop {
            match ws.next().await {
                Some(Ok(Message::Binary(b))) => {
                    let (spec, id) = decode_metadata(&b).map_err(|e| e.0)?;
                    return Ok((ws, spec, id));
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                Some(Ok(Message::Text(t))) => return Err(format!("upstream refused: {t}")),
                Some(Ok(other)) => return Err(format!("upstream sent {other:?} before metadata")),
                Some(Err(e)) => return Err(format!("upstream: {e}")),
                None => return Err("upstream closed before sending metadata".into()),
            }
        }
    }
}

/// `s` without the token or the URL's query and fragment, for an error that
/// may echo the request.
fn redact(s: &str, up: &Upstream) -> String {
    let mut s = s.to_string();
    if let Some(t) = up.token.as_deref().filter(|t| !t.is_empty()) {
        s = s.replace(t, "<redacted>");
    }
    let (scheme, authority, path) = split_url(&up.url);
    let kept = scheme.len() + "://".len() + authority.len() + path.len();
    if let Some(tail) = up.url.get(kept..).filter(|t| !t.is_empty()) {
        s = s.replace(tail, "");
    }
    s
}

/// The CA certificates a `--ca-file` holds, as the only roots a `wss://`
/// upstream's certificate may chain to.
struct TrustedCa {
    connector: Connector,
    roots: usize,
}

/// Read `path` as PEM and trust its `CERTIFICATE` blocks; other blocks are
/// skipped. Errors name the file and never quote it.
fn load_ca_file(path: &std::path::Path) -> Result<TrustedCa, ZenohError> {
    use rustls::pki_types::{pem::PemObject, CertificateDer};
    let bad = |why: String| ZenohError(format!("--ca-file {}: {why}", path.display()));
    let certs = CertificateDer::pem_file_iter(path)
        .and_then(|it| it.collect::<Result<Vec<_>, _>>())
        .map_err(|e| match e {
            rustls::pki_types::pem::Error::Io(e) => bad(e.to_string()),
            // Other variants quote the offending line.
            _ => bad("is not a valid PEM file".into()),
        })?;
    if certs.is_empty() {
        return Err(bad(
            "no PEM certificate (-----BEGIN CERTIFICATE-----) in it".into(),
        ));
    }
    let mut roots = rustls::RootCertStore::empty();
    for (i, cert) in certs.into_iter().enumerate() {
        roots.add(cert).map_err(|e| {
            bad(format!(
                "certificate {} is not a valid certificate: {e}",
                i + 1
            ))
        })?;
    }
    let n = roots.len();
    let config = rustls::ClientConfig::builder_with_provider(std::sync::Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| bad(e.to_string()))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    Ok(TrustedCa {
        connector: Connector::Rustls(std::sync::Arc::new(config)),
        roots: n,
    })
}

/// Identity fields the client pins; a reconnect that changes one is refused.
fn same_policy(a: &PolicySpec, b: &PolicySpec) -> bool {
    a == b
}

struct Conn {
    ws: Option<Ws>,
    /// Calls sent whose replies have not been read: they arrived late.
    stale: u32,
}

/// Counters a test or an operator can read.
#[derive(Debug, Default)]
pub struct BridgeCounts {
    /// Late upstream replies read and dropped.
    pub stale_dropped: AtomicU64,
    /// Upstream connections opened after the first.
    pub reconnects: AtomicU64,
}

pub struct BridgeBackend {
    upstream: Upstream,
    tls: Option<Connector>,
    spec: PolicySpec,
    rt: tokio::runtime::Handle,
    conn: Mutex<Conn>,
    counts: std::sync::Arc<BridgeCounts>,
    warned_latency: AtomicBool,
}

enum Step {
    Done(Result<Computed, BackendError>),
    /// The connection failed; reconnect and send again if time allows.
    Retry(String),
}

impl BridgeBackend {
    /// Once per bridge: the upstream is slower than the client allows.
    fn warn_latency(&self, budget: Duration, infer_ms: Option<f64>) {
        if !self.warned_latency.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                budget_ms = budget.as_millis() as u64,
                upstream_infer_ms = ?infer_ms,
                "the upstream takes longer per call than the client's budget; raise \
                 request_timeout_s or use a faster upstream"
            );
        }
    }

    async fn call(&self, c: &mut Conn, obs: &Observation, deadline: Instant) -> Step {
        let left = |now: Instant| deadline.saturating_duration_since(now);
        if c.ws.is_none() {
            let res = tokio::time::timeout(
                left(Instant::now()),
                self.upstream.connect(self.tls.as_ref()),
            )
            .await;
            match res {
                Err(_) => return Step::Done(Err(BackendError::Expired)),
                Ok(Err(e)) => return Step::Done(Err(BackendError::Model(e))),
                Ok(Ok((ws, spec, _))) => {
                    if !same_policy(&spec, &self.spec) {
                        return Step::Done(Err(BackendError::Model(
                            "the upstream now serves a different policy than the handshake stated"
                                .into(),
                        )));
                    }
                    self.counts.reconnects.fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        upstream = %display_url(&self.upstream.url),
                        "reconnected to the upstream"
                    );
                    c.ws = Some(ws);
                    c.stale = 0;
                }
            }
        }
        let ws = c.ws.as_mut().expect("connected above");
        while c.stale > 0 {
            match tokio::time::timeout(left(Instant::now()), ws.next()).await {
                Err(_) => return Step::Done(Err(BackendError::Expired)),
                Ok(Some(Ok(m @ (Message::Binary(_) | Message::Text(_))))) => {
                    c.stale -= 1;
                    self.counts.stale_dropped.fetch_add(1, Ordering::Relaxed);
                    let infer_ms = match &m {
                        Message::Binary(b) => decode_reply(b).ok().map(|r| r.infer_ms),
                        _ => None,
                    };
                    tracing::info!(upstream_infer_ms = ?infer_ms, "dropped a late upstream reply");
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(e))) => return Step::Retry(e.to_string()),
                Ok(None) => return Step::Retry("upstream closed".into()),
            }
        }
        let request = match encode_request(obs, &self.spec) {
            Ok(r) => r,
            Err(e) => return Step::Done(Err(BackendError::Invalid(e.0))),
        };
        let started = Instant::now();
        if let Err(e) = ws.send(Message::Binary(request)).await {
            return Step::Retry(e.to_string());
        }
        let budget = left(started);
        loop {
            match tokio::time::timeout(left(Instant::now()), ws.next()).await {
                Err(_) => {
                    c.stale += 1;
                    self.warn_latency(budget, None);
                    return Step::Done(Err(BackendError::Expired));
                }
                Ok(Some(Ok(Message::Binary(b)))) => {
                    let res =
                        decode_reply(&b).and_then(|chunk| check_reply(&chunk).map(|()| chunk));
                    return Step::Done(match res {
                        Ok(chunk) => {
                            if chunk.infer_ms > budget.as_secs_f64() * 1000.0 {
                                self.warn_latency(budget, Some(chunk.infer_ms));
                            }
                            Ok(Computed { chunk, started })
                        }
                        Err(e) => Err(BackendError::Model(format!("upstream reply: {}", e.0))),
                    });
                }
                Ok(Some(Ok(Message::Text(t)))) => {
                    return Step::Done(Err(BackendError::Model(format!("upstream: {t}"))))
                }
                Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
                    return Step::Retry("upstream closed".into())
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(e))) => return Step::Retry(e.to_string()),
            }
        }
    }
}

impl Backend for BridgeBackend {
    fn infer(&self, obs: &Observation, deadline: Instant) -> Result<Computed, BackendError> {
        let mut c = self.conn.lock();
        for attempt in 0..2 {
            match self.rt.block_on(self.call(&mut c, obs, deadline)) {
                Step::Done(res) => return res,
                Step::Retry(why) => {
                    tracing::info!(reason = %why, "upstream connection lost");
                    c.ws = None;
                    c.stale = 0;
                    if attempt == 1 || Instant::now() >= deadline {
                        return Err(BackendError::Model(format!("upstream: {why}")));
                    }
                }
            }
        }
        unreachable!("the loop returns on its second attempt")
    }

    /// physicalai's reset starts a new episode: start it on a fresh connection.
    fn reset(&self) {
        let mut c = self.conn.lock();
        if let Some(mut ws) = c.ws.take() {
            let _ = self.rt.block_on(async {
                tokio::time::timeout(Duration::from_secs(1), ws.close(None)).await
            });
        }
        c.stale = 0;
    }
}

/// A running bridge.
pub struct Bridge {
    pub front: ZenohFrontEnd,
    pub counts: std::sync::Arc<BridgeCounts>,
}

/// Connect to `upstream`, then serve physicalai under `listen` (loopback
/// only). An empty `listen.name` takes the upstream's policy name.
pub async fn start_bridge(
    upstream: Upstream,
    mut listen: ZenohListen,
) -> Result<Bridge, ZenohError> {
    upstream.check()?;
    if listen.allow_all_interfaces {
        return Err(ZenohError("the bridge listens on loopback only".into()));
    }
    let tls = match &upstream.ca_file {
        Some(path) => {
            let ca = load_ca_file(path)?;
            tracing::info!(ca_file = %path.display(), roots = ca.roots, "trusting only --ca-file");
            Some(ca.connector)
        }
        None => None,
    };
    let (ws, spec, identity) =
        tokio::time::timeout(CONNECT_TIMEOUT, upstream.connect(tls.as_ref()))
            .await
            .map_err(|_| {
                ZenohError(format!(
                    "no metadata from {} within 10 s",
                    display_url(&upstream.url)
                ))
            })?
            .map_err(ZenohError)?;
    if listen.name.is_empty() {
        listen.name = identity.policy_name.clone();
    }
    tracing::info!(
        upstream = %display_url(&upstream.url),
        policy = %identity.policy_name,
        manifest_sha256 = %spec.manifest_sha256,
        package_sha256 = %spec.package_sha256,
        precision = %spec.precision,
        "upstream policy"
    );
    let counts = std::sync::Arc::new(BridgeCounts::default());
    let backend = BridgeBackend {
        upstream,
        tls,
        spec: spec.clone(),
        rt: tokio::runtime::Handle::current(),
        conn: Mutex::new(Conn {
            ws: Some(ws),
            stale: 0,
        }),
        counts: counts.clone(),
        warned_latency: AtomicBool::new(false),
    };
    let advert = Advert {
        device: spec.device.clone(),
        spec,
        identity,
        backend: "openvino".into(),
    };
    let front = ZenohFrontEnd::start(&listen, advert, Box::new(backend)).await?;
    Ok(Bridge { front, counts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn up(url: &str, token: bool) -> Upstream {
        Upstream {
            url: url.into(),
            token: token.then(|| "s3cret".to_string()),
            ca_file: None,
        }
    }

    /// A throwaway self-signed CA made for this test (its key was discarded).
    const TEST_CA: &str = "-----BEGIN CERTIFICATE-----
MIIBfDCCASKgAwIBAgIUCphMVlh5do4Qx3v5Gvjw3FNcBakwCgYIKoZIzj0EAwIw
GzEZMBcGA1UEAwwQY2FzY2FkaWEgdGVzdCBDQTAgFw0yNjExMTYxNDI1MjVaGA8y
MTI2MTAyMzE0MjUyNVowGzEZMBcGA1UEAwwQY2FzY2FkaWEgdGVzdCBDQTBZMBMG
ByqGSM49AgEGCCqGSM49AwEHA0IABJlpWpKsOp2K4smHmA9TnmVByK6xJkfOcDbT
MCVcbVX4SRdtc6IvokYP/Egfcq450Cw4NmlMuNo3mQlzNQPGZZ+jQjBAMB0GA1Ud
DgQWBBSJxMu853vt4Qj7vAD+SQeT8Cu/pTAPBgNVHRMBAf8EBTADAQH/MA4GA1Ud
DwEB/wQEAwICBDAKBggqhkjOPQQDAgNIADBFAiBXfUj2fbs151EW0gAXbKOQOW8k
mkjA+/cDNceaJbMkLwIhAPIzdcgHgqRqhxVhV+fgiFEG9PzyIw+VzdcidPmTYk9q
-----END CERTIFICATE-----
";

    fn ca_file(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn a_ca_file_with_a_certificate_is_trusted() {
        let t = tempfile::tempdir().unwrap();
        let p = ca_file(t.path(), "ca.crt", TEST_CA);
        assert_eq!(load_ca_file(&p).unwrap().roots, 1);
        // Two CAs, and a non-certificate block between them, are fine.
        let two =
            format!("{TEST_CA}-----BEGIN X509 CRL-----\nAAAA\n-----END X509 CRL-----\n{TEST_CA}");
        let p = ca_file(t.path(), "two.crt", &two);
        assert_eq!(load_ca_file(&p).unwrap().roots, 2);
    }

    fn refused(path: &std::path::Path) -> String {
        match load_ca_file(path) {
            Ok(_) => panic!("{} was accepted", path.display()),
            Err(e) => e.0,
        }
    }

    #[test]
    fn a_bad_ca_file_is_refused_without_echoing_it() {
        let t = tempfile::tempdir().unwrap();
        let body = "not a certificate SECRETISH-LINE\n";
        let garbage = ca_file(t.path(), "garbage.crt", body);
        let e = refused(&garbage);
        assert!(
            e.contains("garbage.crt") && e.contains("no PEM certificate"),
            "{e}"
        );
        assert!(!e.contains("SECRETISH"), "{e}");

        let bad_der = "-----BEGIN CERTIFICATE-----\nAAAAAAAA\n-----END CERTIFICATE-----\n";
        let e = refused(&ca_file(t.path(), "bad.crt", bad_der));
        assert!(
            e.contains("bad.crt") && e.contains("not a valid certificate"),
            "{e}"
        );
        assert!(!e.contains("AAAAAAAA"), "{e}");

        let cut = &TEST_CA[..TEST_CA.len() - 30];
        let e = refused(&ca_file(t.path(), "cut.crt", cut));
        assert!(e.contains("cut.crt"), "{e}");
        assert!(!e.contains("MIIB"), "{e}");

        let e = refused(&t.path().join("missing.crt"));
        assert!(e.contains("missing.crt"), "{e}");
    }

    #[test]
    fn a_ca_file_needs_wss() {
        let mut u = up("ws://127.0.0.1:8000/v1/policy/ws", false);
        u.ca_file = Some("ca.crt".into());
        assert!(u.check().unwrap_err().0.contains("wss://"));
        u.url = "wss://gw.example/v1/policy/ws".into();
        u.check().unwrap();
    }

    #[test]
    fn tokens_travel_over_tls_or_loopback_only() {
        up("ws://127.0.0.1:8000/v1/policy/ws", true)
            .check()
            .unwrap();
        up("ws://[::1]:8000/v1/policy/ws", true).check().unwrap();
        up("ws://localhost/v1/policy/ws", true).check().unwrap();
        up("wss://gw.example/v1/policy/ws", true).check().unwrap();
        up("ws://gw.example/v1/policy/ws", false).check().unwrap();
        assert!(up("ws://gw.example/v1/policy/ws", true).check().is_err());
        assert!(up("ws://127.0.0.1.evil/x", true).check().is_err());
        assert!(up("http://127.0.0.1/x", false).check().is_err());
    }

    #[test]
    fn logged_urls_drop_userinfo_query_and_fragment() {
        assert_eq!(
            display_url("wss://user:pass@gw.example:8443/v1/policy/ws?token=abc#frag"),
            "wss://gw.example:8443/v1/policy/ws"
        );
        assert_eq!(
            display_url("ws://127.0.0.1:8000?api_key=k"),
            "ws://127.0.0.1:8000"
        );
        assert_eq!(
            display_url("ws://[::1]:8000/v1/policy/ws"),
            "ws://[::1]:8000/v1/policy/ws"
        );
        let u = up("wss://gw.example/v1/policy/ws?api_key=k", false);
        assert!(!format!("{u:?}").contains("api_key"));
    }

    #[test]
    fn userinfo_in_the_upstream_url_is_refused() {
        for url in [
            "wss://user:pass@gw.example/v1/policy/ws",
            "ws://tok@127.0.0.1:8000/x",
        ] {
            let e = up(url, false).check().unwrap_err().0;
            assert!(e.contains("--token-file"), "{url}: {e}");
        }
        // An '@' after the authority is not userinfo.
        up("ws://127.0.0.1:8000/v1/policy/ws?x=a@b", false)
            .check()
            .unwrap();
    }

    #[test]
    fn the_token_is_never_printed() {
        let u = up("wss://gw.example/v1/policy/ws", true);
        assert!(!format!("{u:?}").contains("s3cret"));
        assert_eq!(redact("bad s3cret here", &u), "bad <redacted> here");
        let q = up("ws://127.0.0.1:1/x?api_key=k#f", false);
        assert_eq!(
            redact("Unable to connect to ws://127.0.0.1:1/x?api_key=k#f", &q),
            "Unable to connect to ws://127.0.0.1:1/x"
        );
    }
}
