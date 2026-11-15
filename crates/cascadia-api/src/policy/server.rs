//! The policy HTTP surface: `GET /health`, `GET /v1/policy/metadata` and
//! `WS /v1/policy/ws`, on their own state (the chat `AppState` is not involved),
//! plus the hyper serve loop that lets the WebSocket upgrade.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use cascadia_engine::{payload_cap, EngineError, EngineResult, PolicyEngine};
use cascadia_types::{ActionChunk, PolicySpec};
use futures::{SinkExt, StreamExt};
use parking_lot::Mutex;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};

use super::{check_reply, decode_request, encode_metadata, encode_reply, metadata_json};

/// How often the server pings; a connection whose last ping got no pong by
/// the next one is closed, so a dropped robot frees its slot.
pub const PING_EVERY: Duration = Duration::from_secs(20);

/// The HTTP request headers must arrive within this.
pub const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);

pub const DEFAULT_MAX_CONNECTIONS: usize = 4;

#[derive(Clone, Debug)]
pub struct PolicyServerConfig {
    /// `Origin` values allowed to open the socket. A request without an
    /// `Origin` header (every non-browser client) is always allowed.
    pub allow_origins: Vec<String>,
    pub max_connections: usize,
    pub ping_every: Duration,
}

impl Default for PolicyServerConfig {
    fn default() -> Self {
        Self {
            allow_origins: Vec::new(),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            ping_every: PING_EVERY,
        }
    }
}

type SharedEngine = Arc<Mutex<Box<dyn PolicyEngine>>>;

#[derive(Clone)]
struct PolicyState {
    engine: SharedEngine,
    spec: Arc<PolicySpec>,
    metadata: Arc<Vec<u8>>,
    metadata_json: Arc<String>,
    max_payload: usize,
    slots: Arc<Semaphore>,
    cfg: Arc<PolicyServerConfig>,
}

/// The policy router for a loaded (and warmed) engine.
///
/// Refuses a spec whose WebSocket cap exceeds [`cascadia_engine::PAYLOAD_MAX_BYTES`].
pub fn policy_router(
    engine: Box<dyn PolicyEngine>,
    cfg: PolicyServerConfig,
) -> EngineResult<Router> {
    if cfg.max_connections == 0 {
        return Err(EngineError::InvalidConfig(
            "the policy server needs at least one connection slot".into(),
        ));
    }
    let spec = engine.spec().clone();
    let max_payload = payload_cap(&spec)? as usize;
    let state = PolicyState {
        metadata: Arc::new(encode_metadata(&spec)),
        metadata_json: Arc::new(metadata_json(&spec)),
        spec: Arc::new(spec),
        engine: Arc::new(Mutex::new(engine)),
        max_payload,
        slots: Arc::new(Semaphore::new(cfg.max_connections)),
        cfg: Arc::new(cfg),
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/v1/policy/metadata", get(metadata))
        .route("/v1/policy/ws", get(ws))
        .with_state(state))
}

async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({ "status": "ok" }))
}

async fn metadata(State(st): State<PolicyState>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json")],
        st.metadata_json.as_str().to_owned(),
    )
}

async fn ws(
    State(st): State<PolicyState>,
    headers: HeaderMap,
    upgrade: Result<WebSocketUpgrade, axum::extract::ws::rejection::WebSocketUpgradeRejection>,
) -> Response {
    if let Some(origin) = headers.get(header::ORIGIN) {
        let ok = origin
            .to_str()
            .is_ok_and(|o| st.cfg.allow_origins.iter().any(|a| a == o));
        if !ok {
            return (
                StatusCode::FORBIDDEN,
                "this Origin may not open the policy socket (see --policy-allow-origin)",
            )
                .into_response();
        }
    }
    let upgrade = match upgrade {
        Ok(u) => u,
        Err(rejection) => return rejection.into_response(),
    };
    let Ok(permit) = st.slots.clone().try_acquire_owned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "the policy server is full: {} connections are open",
                st.cfg.max_connections
            ),
        )
            .into_response();
    };
    upgrade
        .max_message_size(st.max_payload)
        .max_frame_size(st.max_payload)
        .on_upgrade(move |socket| serve_socket(socket, st, permit))
}

fn client_message(e: EngineError) -> String {
    match e {
        EngineError::InvalidConfig(m) => m,
        other => other.to_string(),
    }
}

/// One connection. The socket is read throughout, including while a request
/// waits for the model lock and runs, so pings are answered and pongs seen.
async fn serve_socket(socket: WebSocket, st: PolicyState, slot: OwnedSemaphorePermit) {
    // Shared with the in-flight call, so a client that disconnects mid-call
    // keeps its slot until the engine is done with its work.
    let slot = Arc::new(slot);
    let (mut tx, mut rx) = socket.split();
    if tx
        .send(Message::Binary(st.metadata.as_ref().clone()))
        .await
        .is_err()
    {
        return;
    }
    let (done_tx, mut done_rx) = mpsc::channel::<Result<ActionChunk, String>>(1);
    let mut busy = false;
    let every = st.cfg.ping_every;
    let mut ping = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut pong_due = false;
    loop {
        let out = tokio::select! {
            msg = rx.next() => match msg {
                Some(Ok(Message::Binary(_))) if busy => Some(Message::Text(
                    "one request at a time: wait for the reply before sending the next".into(),
                )),
                Some(Ok(Message::Binary(bytes))) => match decode_request(&bytes, &st.spec) {
                    Err(e) => Some(Message::Text(e.0)),
                    Ok(obs) => {
                        busy = true;
                        let engine = st.engine.clone();
                        let done = done_tx.clone();
                        let slot = slot.clone();
                        tokio::task::spawn_blocking(move || {
                            let _slot = slot;
                            if done.is_closed() {
                                return;
                            }
                            let mut engine = engine.lock();
                            // The connection may have gone while this waited for the lock.
                            if done.is_closed() {
                                return;
                            }
                            let res = engine.infer(&obs).map_err(client_message);
                            drop(engine);
                            let _ = done.blocking_send(res);
                        });
                        None
                    }
                },
                Some(Ok(Message::Text(_))) => Some(Message::Text(
                    "send requests as msgpack binary messages".into(),
                )),
                Some(Ok(Message::Pong(_))) => {
                    pong_due = false;
                    None
                }
                // Pings are answered by the socket itself while it is read.
                Some(Ok(Message::Ping(_))) => None,
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
            },
            Some(res) = done_rx.recv() => {
                busy = false;
                Some(match res.and_then(|c| check_reply(&c).map(|()| c).map_err(|e| e.0)) {
                    Ok(chunk) => Message::Binary(encode_reply(&chunk)),
                    Err(m) => Message::Text(m),
                })
            }
            _ = ping.tick() => {
                if pong_due {
                    tracing::info!("policy socket missed a pong; closing it");
                    break;
                }
                pong_due = true;
                Some(Message::Ping(Vec::new()))
            }
        };
        if let Some(m) = out {
            if tx.send(m).await.is_err() {
                break;
            }
        }
    }
    let _ = tokio::time::timeout(Duration::from_secs(1), tx.close()).await;
}

/// Serve `app` over HTTP/1.1 with upgrades enabled, which the WebSocket
/// needs (a loop without `.with_upgrades()` answers 101 and never upgrades),
/// and with a deadline on the request headers.
pub async fn serve_policy(
    listener: tokio::net::TcpListener,
    app: Router,
    header_read_timeout: Duration,
) -> std::io::Result<()> {
    use hyper::server::conn::http1;
    use hyper_util::rt::{TokioIo, TokioTimer};
    use std::convert::Infallible;
    use tower::Service;

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "accept failed; retrying after 50ms");
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        let app = app.clone();
        let svc = hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
            let mut router = app.clone();
            let fut = router.call(req.map(axum::body::Body::new));
            async move { Ok::<_, Infallible>(fut.await.unwrap_or_else(|e: Infallible| match e {})) }
        });
        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(header_read_timeout)
                .serve_connection(TokioIo::new(stream), svc)
                .with_upgrades()
                .await
            {
                tracing::debug!(error = %e, "policy connection ended");
            }
        });
    }
}
