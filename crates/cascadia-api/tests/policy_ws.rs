//! The policy WebSocket over a real TCP socket through `serve_policy`
//! (router-level `oneshot` tests cannot see an upgrade), against
//! `MockPolicyEngine`.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cascadia_api::policy::server::{policy_router, serve_policy, PolicyServerConfig};
use cascadia_api::policy::{decode_request, encode_metadata, encode_reply, max_payload_bytes};
use cascadia_engine::{EngineResult, MockPolicyEngine, PolicyEngine};
use cascadia_types::{ActionChunk, NoiseMode, Observation, PolicySpec, ResizeConvention};
use futures::{SinkExt, StreamExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

const REQUEST: &[u8] = include_bytes!("fixtures/policy/request.msgpack");

fn spec() -> PolicySpec {
    PolicySpec {
        model: "mock".into(),
        device: "CPU".into(),
        cameras: vec!["image".into(), "wrist".into()],
        height: 4,
        width: 4,
        state_len: 3,
        chunk: [2, 3],
        noise: NoiseMode::Input,
        noise_shape: Some(vec![2, 3]),
        resize: ResizeConvention::TopLeft,
        prompt_max_bytes: 64,
        attribution: None,
    }
}

/// The mock, made to take `delay` per call.
struct Slow(MockPolicyEngine, Duration);

impl PolicyEngine for Slow {
    fn spec(&self) -> &PolicySpec {
        self.0.spec()
    }
    fn warmup(&mut self) -> EngineResult<Vec<u32>> {
        self.0.warmup()
    }
    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        std::thread::sleep(self.1);
        self.0.infer(obs)
    }
}

/// Counts calls that reach the engine, takes `delay` each, and panics on the
/// prompt "panic".
struct Probe {
    inner: MockPolicyEngine,
    delay: Duration,
    calls: Arc<AtomicUsize>,
}

impl PolicyEngine for Probe {
    fn spec(&self) -> &PolicySpec {
        self.inner.spec()
    }
    fn warmup(&mut self) -> EngineResult<Vec<u32>> {
        self.inner.warmup()
    }
    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(
            obs.prompt.trim() != "panic",
            "probe engine panics on request"
        );
        std::thread::sleep(self.delay);
        self.inner.infer(obs)
    }
}

fn probe(delay: Duration) -> (Box<Probe>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = Probe {
        inner: MockPolicyEngine::new(spec()),
        delay,
        calls: calls.clone(),
    };
    (Box::new(engine), calls)
}

async fn start_engine(engine: Box<dyn PolicyEngine>, cfg: PolicyServerConfig) -> SocketAddr {
    let app = policy_router(engine, cfg).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve_policy(listener, app, Duration::from_secs(10)));
    addr
}

/// `REQUEST` with its prompt replaced.
fn request_with_prompt(prompt: &str) -> Vec<u8> {
    let old = b"pick up the cube";
    assert_eq!(prompt.len(), old.len());
    let at = REQUEST.windows(old.len()).position(|w| w == old).unwrap();
    let mut r = REQUEST.to_vec();
    r[at..at + old.len()].copy_from_slice(prompt.as_bytes());
    r
}

async fn start(delay: Duration, cfg: PolicyServerConfig, header_timeout: Duration) -> SocketAddr {
    let engine = Box::new(Slow(MockPolicyEngine::new(spec()), delay));
    let app = policy_router(engine, cfg).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve_policy(listener, app, header_timeout));
    addr
}

async fn start_default() -> SocketAddr {
    start(
        Duration::ZERO,
        PolicyServerConfig::default(),
        Duration::from_secs(10),
    )
    .await
}

fn url(addr: SocketAddr) -> String {
    format!("ws://{addr}/v1/policy/ws")
}

async fn connect(addr: SocketAddr) -> Ws {
    let (mut ws, _) = connect_async(url(addr)).await.unwrap();
    match next(&mut ws).await {
        Message::Binary(b) => assert_eq!(b, encode_metadata(&spec())),
        other => panic!("expected metadata first, got {other:?}"),
    }
    ws
}

async fn next(ws: &mut Ws) -> Message {
    tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("no message within 5 s")
        .expect("socket ended")
        .expect("socket error")
}

/// The next data frame, skipping server pings (answered by the read itself).
async fn next_data(ws: &mut Ws) -> Message {
    loop {
        match next(ws).await {
            Message::Ping(_) | Message::Pong(_) => continue,
            m => return m,
        }
    }
}

fn expected_reply() -> Vec<u8> {
    let obs = decode_request(REQUEST, &spec()).unwrap();
    encode_reply(&MockPolicyEngine::new(spec()).infer(&obs).unwrap())
}

/// True once the server has closed `ws` (a close frame, an error or EOF).
async fn closed_within(ws: &mut Ws, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), ws.next()).await {
            Ok(None) | Ok(Some(Err(_))) | Ok(Some(Ok(Message::Close(_)))) => return true,
            _ => {}
        }
    }
    false
}

#[tokio::test]
async fn metadata_first_then_binary_actions() {
    let addr = start_default().await;
    let mut ws = connect(addr).await;
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    assert_eq!(next_data(&mut ws).await, Message::Binary(expected_reply()));
    // The connection serves request after request.
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    assert_eq!(next_data(&mut ws).await, Message::Binary(expected_reply()));
}

#[tokio::test]
async fn a_bad_request_gets_a_text_error_and_the_socket_stays_open() {
    let addr = start_default().await;
    let mut ws = connect(addr).await;
    let mut bad = REQUEST.to_vec();
    bad.pop();
    ws.send(Message::Binary(bad)).await.unwrap();
    match next_data(&mut ws).await {
        Message::Text(t) => assert!(t.contains("truncated"), "{t}"),
        other => panic!("expected a text error, got {other:?}"),
    }
    ws.send(Message::Text("hello".into())).await.unwrap();
    match next_data(&mut ws).await {
        Message::Text(t) => assert!(t.contains("msgpack binary"), "{t}"),
        other => panic!("expected a text error, got {other:?}"),
    }
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    assert_eq!(next_data(&mut ws).await, Message::Binary(expected_reply()));
}

#[tokio::test]
async fn an_oversize_message_closes_the_connection() {
    let addr = start_default().await;
    let mut ws = connect(addr).await;
    let cap = max_payload_bytes(&spec()) as usize;
    ws.send(Message::Binary(vec![0; cap + 1])).await.unwrap();
    assert!(closed_within(&mut ws, Duration::from_secs(5)).await);
}

#[tokio::test]
async fn a_foreign_origin_is_refused() {
    let cfg = PolicyServerConfig {
        allow_origins: vec!["http://robot.local".into()],
        ..Default::default()
    };
    let addr = start(Duration::ZERO, cfg, Duration::from_secs(10)).await;
    let mut req = url(addr).into_client_request().unwrap();
    req.headers_mut()
        .insert("Origin", "http://evil.example".parse().unwrap());
    match connect_async(req).await {
        Err(WsError::Http(resp)) => assert_eq!(resp.status(), 403),
        other => panic!("expected 403, got {other:?}"),
    }
    let mut req = url(addr).into_client_request().unwrap();
    req.headers_mut()
        .insert("Origin", "http://robot.local".parse().unwrap());
    connect_async(req).await.unwrap();
}

#[tokio::test]
async fn at_capacity_the_upgrade_gets_503() {
    let cfg = PolicyServerConfig {
        max_connections: 1,
        ..Default::default()
    };
    let addr = start(Duration::ZERO, cfg, Duration::from_secs(10)).await;
    let mut first = connect(addr).await;
    match connect_async(url(addr)).await {
        Err(WsError::Http(resp)) => {
            assert_eq!(resp.status(), 503);
            let body = String::from_utf8(resp.body().clone().unwrap_or_default()).unwrap();
            assert!(body.contains("full"), "{body}");
        }
        other => panic!("expected 503, got {other:?}"),
    }
    first.close(None).await.unwrap();
    drop(first);
    // The slot frees once the server sees the close.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match connect_async(url(addr)).await {
            Ok(_) => break,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            Err(e) => panic!("slot never freed: {e}"),
        }
    }
}

#[tokio::test]
async fn a_slow_call_does_not_block_pings_or_reads() {
    let cfg = PolicyServerConfig {
        ping_every: Duration::from_millis(200),
        ..Default::default()
    };
    let addr = start(Duration::from_millis(1500), cfg, Duration::from_secs(10)).await;
    let mut ws = connect(addr).await;
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    // A client ping during the call is answered before the reply.
    tokio::time::sleep(Duration::from_millis(100)).await;
    ws.send(Message::Ping(b"x".to_vec())).await.unwrap();
    let (mut server_pings, mut pong) = (0, false);
    let reply = loop {
        match next(&mut ws).await {
            Message::Ping(_) => server_pings += 1,
            Message::Pong(p) => {
                assert_eq!(p, b"x");
                pong = true;
            }
            m => break m,
        }
    };
    assert!(pong, "the client's ping went unanswered during the call");
    assert!(server_pings >= 3, "{server_pings} server pings in 1.5 s");
    assert_eq!(reply, Message::Binary(expected_reply()));
    // A second request while one is in flight is refused, not queued.
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    match next_data(&mut ws).await {
        Message::Text(t) => assert!(t.contains("one request at a time"), "{t}"),
        other => panic!("expected a text error, got {other:?}"),
    }
    assert_eq!(next_data(&mut ws).await, Message::Binary(expected_reply()));
}

#[tokio::test]
async fn a_client_that_disconnects_mid_call_keeps_its_slot_until_the_call_ends() {
    let cfg = PolicyServerConfig {
        max_connections: 1,
        ..Default::default()
    };
    let addr = start(Duration::from_millis(1500), cfg, Duration::from_secs(10)).await;
    let mut ws = connect(addr).await;
    let sent = Instant::now();
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    ws.close(None).await.unwrap();
    drop(ws);
    // While the engine still runs the abandoned call, the slot stays taken.
    tokio::time::sleep(Duration::from_millis(300)).await;
    match connect_async(url(addr)).await {
        Err(WsError::Http(resp)) => assert_eq!(resp.status(), 503),
        Ok(_) => panic!("slot freed while the engine was still busy"),
        Err(e) => panic!("unexpected error {e}"),
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match connect_async(url(addr)).await {
            Ok(_) => break,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            Err(e) => panic!("slot never freed: {e}"),
        }
    }
    assert!(sent.elapsed() >= Duration::from_millis(1500));
}

#[tokio::test]
async fn an_engine_panic_gets_an_error_frame_and_the_socket_keeps_serving() {
    let (engine, calls) = probe(Duration::ZERO);
    let addr = start_engine(engine, PolicyServerConfig::default()).await;
    let mut ws = connect(addr).await;
    // Padded to the fixture prompt's length; the probe trims it.
    ws.send(Message::Binary(request_with_prompt("panic           ")))
        .await
        .unwrap();
    match next_data(&mut ws).await {
        Message::Text(t) => assert!(t.contains("internal error"), "{t}"),
        other => panic!("expected a text error, got {other:?}"),
    }
    ws.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    assert_eq!(next_data(&mut ws).await, Message::Binary(expected_reply()));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_queued_call_whose_client_left_never_runs() {
    let cfg = PolicyServerConfig {
        max_connections: 2,
        ..Default::default()
    };
    let call = Duration::from_millis(800);
    let (engine, calls) = probe(call);
    let addr = start_engine(engine, cfg).await;
    let mut a = connect(addr).await;
    a.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    // B queues behind A's call for the engine lock, then leaves.
    let mut b = connect(addr).await;
    b.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    b.close(None).await.unwrap();
    drop(b);
    assert_eq!(next_data(&mut a).await, Message::Binary(expected_reply()));
    // A's next call starts at once instead of waiting out B's.
    let sent = Instant::now();
    a.send(Message::Binary(REQUEST.to_vec())).await.unwrap();
    assert_eq!(next_data(&mut a).await, Message::Binary(expected_reply()));
    assert!(sent.elapsed() < call * 3 / 2, "{:?}", sent.elapsed());
    assert_eq!(calls.load(Ordering::SeqCst), 2, "B's abandoned call ran");
}

#[tokio::test]
async fn a_client_that_stops_answering_pings_is_closed_and_frees_its_slot() {
    let cfg = PolicyServerConfig {
        max_connections: 1,
        ping_every: Duration::from_millis(100),
        ..Default::default()
    };
    let addr = start(Duration::ZERO, cfg, Duration::from_secs(10)).await;
    // Never read again, so the client never answers the server's pings.
    let _silent = connect(addr).await;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match connect_async(url(addr)).await {
            Ok(_) => break,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            Err(e) => panic!("silent client kept its slot: {e}"),
        }
    }
}

async fn http_get(addr: SocketAddr, path: &str) -> String {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes())
        .await
        .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).await.unwrap();
    out
}

#[tokio::test]
async fn metadata_and_health_over_http() {
    let addr = start_default().await;
    let meta = http_get(addr, "/v1/policy/metadata").await;
    assert!(meta.starts_with("HTTP/1.1 200"), "{meta}");
    let body = meta.split("\r\n\r\n").nth(1).unwrap();
    let got: PolicySpec = serde_json::from_str(body).unwrap();
    assert_eq!(got, spec());
    let health = http_get(addr, "/health").await;
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
}

#[tokio::test]
async fn slow_request_headers_are_cut_off() {
    let addr = start(
        Duration::ZERO,
        PolicyServerConfig::default(),
        Duration::from_millis(200),
    )
    .await;
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(b"GET /health HTTP/1.1\r\n").await.unwrap();
    let mut buf = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(3), s.read_to_end(&mut buf)).await;
    assert!(read.is_ok(), "the server kept a half-sent request open");
}
