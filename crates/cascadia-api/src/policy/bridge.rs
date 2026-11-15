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
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

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
}

impl std::fmt::Debug for Upstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upstream")
            .field("url", &self.url)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
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
        if self.token.is_some() && !tls && !host_is_loopback(&self.url) {
            return Err(ZenohError(
                "a token is only sent over wss://, or ws:// to loopback".into(),
            ));
        }
        Ok(())
    }

    /// Open the socket and read the metadata it sends first.
    async fn connect(&self) -> Result<(Ws, PolicySpec, PackageIdentity), String> {
        let mut req = self
            .url
            .as_str()
            .into_client_request()
            .map_err(|e| format!("upstream URL: {e}"))?;
        if let Some(t) = &self.token {
            let mut v = HeaderValue::from_str(&format!("Bearer {t}"))
                .map_err(|_| "the token is not a valid header value".to_string())?;
            v.set_sensitive(true);
            req.headers_mut().insert(AUTHORIZATION, v);
        }
        // The error may echo the request; keep only its kind.
        let (mut ws, _) = tokio_tungstenite::connect_async(req).await.map_err(|e| {
            format!(
                "connect {}: {}",
                self.url,
                redact(&e.to_string(), &self.token)
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

fn redact(s: &str, token: &Option<String>) -> String {
    match token {
        Some(t) if !t.is_empty() => s.replace(t.as_str(), "<redacted>"),
        _ => s.to_string(),
    }
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
            let res = tokio::time::timeout(left(Instant::now()), self.upstream.connect()).await;
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
                    tracing::info!(upstream = %self.upstream.url, "reconnected to the upstream");
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
        let started = Instant::now();
        if let Err(e) = ws
            .send(Message::Binary(encode_request(obs, &self.spec)))
            .await
        {
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
    let (ws, spec, identity) = tokio::time::timeout(CONNECT_TIMEOUT, upstream.connect())
        .await
        .map_err(|_| ZenohError(format!("no metadata from {} within 10 s", upstream.url)))?
        .map_err(ZenohError)?;
    if listen.name.is_empty() {
        listen.name = identity.policy_name.clone();
    }
    tracing::info!(
        upstream = %upstream.url,
        policy = %identity.policy_name,
        manifest_sha256 = %spec.manifest_sha256,
        precision = %spec.precision,
        "upstream policy"
    );
    let counts = std::sync::Arc::new(BridgeCounts::default());
    let backend = BridgeBackend {
        upstream,
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
        }
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
    fn the_token_is_never_printed() {
        let u = up("wss://gw.example/v1/policy/ws", true);
        assert!(!format!("{u:?}").contains("s3cret"));
        assert_eq!(redact("bad s3cret here", &u.token), "bad <redacted> here");
    }
}
