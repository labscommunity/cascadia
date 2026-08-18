//! Regression guard for the pipeline-wedge bug: a downstream rank that dies
//! mid-request (killed peer, or a black-holed socket that never sends FIN/RST)
//! must surface as a *fast* error, never an unbounded hang.
//!
//! Before the fix, the rank-0 head and every mid relay awaited the owed Token
//! reply with the idle-tolerant `recv_kind_client`, which waits up to the
//! ~900 s frame-idle ceiling for the next frame to start. Killing a mid rank
//! therefore left the whole pipeline pinned on that ceiling: the API request
//! wedged (curl hung with no HTTP status) instead of returning a fast 5xx, and
//! /health could not even observe the failure. `recv_token_reply` bounds that
//! wait on a strict deadline; these tests pin that contract over REAL loopback
//! transport (no mocks) so a future refactor that drops the bound fails here.
use cascadia_engine_sparse_moe::dist::{recv_token_reply, send_token_upstream};
use cascadia_transport::{ActivationClient, ActivationServer};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

async fn pair() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
    let mut s = ActivationServer::new("127.0.0.1", 0);
    s.start().await.unwrap();
    let port = s.port();
    let s = Arc::new(Mutex::new(s));
    let sc = s.clone();
    let t = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let mut c = ActivationClient::new("127.0.0.1", port);
    c.connect_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    let c = Arc::new(Mutex::new(c));
    t.await.unwrap();
    (s, c)
}

/// Connected-but-silent downstream: the reply must time out near the deadline,
/// nowhere near the 900 s idle ceiling an unbounded recv would wait on.
#[tokio::test]
async fn silent_downstream_reply_times_out_fast() {
    let (_s, c) = pair().await; // server accepts, then never sends a Token
    let deadline = Duration::from_millis(500);
    let start = Instant::now();
    let res = recv_token_reply(&c, deadline).await;
    let elapsed = start.elapsed();
    let err = res.expect_err("silent peer must yield Err, not a token");
    assert!(
        err.contains("reply timeout"),
        "expected a timeout error, got: {err}"
    );
    // Generous ceiling: the point is it returns in ~deadline, not ~900 s.
    assert!(
        elapsed < Duration::from_secs(10),
        "reply must return near the deadline, not hang on the idle ceiling: {elapsed:?}"
    );
}

/// A downstream that drops the connection (FIN) must also fail fast — the
/// closed socket surfaces immediately, well inside the deadline.
#[tokio::test]
async fn closed_downstream_reply_errors_fast() {
    let (s, c) = pair().await;
    drop(s); // tear the downstream down before it ever replies
    let start = Instant::now();
    let res = recv_token_reply(&c, Duration::from_secs(30)).await;
    assert!(res.is_err(), "closed peer must yield Err, got {res:?}");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "a closed socket must surface at once, not wait the full deadline"
    );
}

/// A timeout must drop the connection, so a late Token cannot be handed to the
/// NEXT request as its reply.
///
/// Timing out does not cancel the work downstream. The peer is usually alive
/// and still computing; its Token lands on the socket after we stopped waiting.
/// The body is eight raw bytes with no sequence number, so a reused connection
/// serves that stale reply to the following request — every later token off by
/// one frame, coherent enough that nothing looks broken.
#[tokio::test]
async fn late_token_after_timeout_is_not_served_to_the_next_request() {
    let (s, c) = pair().await;

    // Request 1: downstream stays silent past the deadline.
    let err = recv_token_reply(&c, Duration::from_millis(300))
        .await
        .expect_err("silent peer must yield Err");
    assert!(err.contains("reply timeout"), "unexpected error: {err}");

    // Downstream finishes late and sends the token it owed request 1.
    let sb = s.clone();
    let _ = tokio::spawn(async move { send_token_upstream(&sb, 4242).await }).await;

    // Request 2 must not be given it.
    let start = Instant::now();
    let res = recv_token_reply(&c, Duration::from_secs(30)).await;
    assert!(
        res.is_err(),
        "request 1's late token was served to request 2: {res:?}"
    );
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "a dropped connection must fail fast, not wait out another deadline"
    );
}

/// Happy path: a live downstream that replies with a Token returns its value.
#[tokio::test]
async fn downstream_token_reply_returns_value() {
    let (s, c) = pair().await;
    let sb = s.clone();
    let sender = tokio::spawn(async move { send_token_upstream(&sb, 4242).await.unwrap() });
    let got = recv_token_reply(&c, Duration::from_secs(5))
        .await
        .expect("live downstream reply should succeed");
    assert_eq!(got, 4242, "must return the token the downstream sampled");
    sender.await.unwrap();
}
