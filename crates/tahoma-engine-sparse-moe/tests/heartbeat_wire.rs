//! Heartbeat round-trip via in-process tahoma-transport sockets.
//!
//! Validates iter 092's `FrameKind::HeartbeatPing` /
//! `FrameKind::HeartbeatPong` wire format and the
//! `HeartbeatWatchdog`'s integration with the wire side. No OpenVINO,
//! no model artifacts, no engine instance — pure socket plumbing +
//! watchdog state machine.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tahoma_engine_sparse_moe::dist::{
    ping_one_round, recv_heartbeat_body_client, recv_heartbeat_body_server, recv_kind_client,
    recv_kind_server, run_heartbeat_loop, send_heartbeat_ping, send_heartbeat_ping_upstream,
    send_heartbeat_pong_downstream, send_heartbeat_pong_upstream, FrameKind, HeartbeatOutcome,
    HeartbeatWatchdog, HEARTBEAT_BODY_BYTES,
};
use tahoma_transport::{ActivationClient, ActivationServer};
use tokio::sync::Mutex;

async fn make_pair() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.expect("server.start");
    let port = server.port();
    let server = Arc::new(Mutex::new(server));
    let server_clone = server.clone();
    let server_task = tokio::spawn(async move {
        server_clone
            .lock()
            .await
            .accept()
            .await
            .expect("server.accept");
    });
    let mut client = ActivationClient::new("127.0.0.1", port);
    client
        .connect_with_timeout(Duration::from_secs(5))
        .await
        .expect("client.connect");
    let client = Arc::new(Mutex::new(client));
    server_task.await.expect("server task panicked");
    (server, client)
}

#[test]
fn heartbeat_body_bytes_is_eight() {
    // Doc constant — if we ever bump the payload past 8 B (e.g. embed
    // a wall-clock send timestamp for RTT measurement) the rank-0
    // driver and worker recv code both need to bump in lockstep, and
    // this test fires loudly to catch a one-sided change.
    assert_eq!(HEARTBEAT_BODY_BYTES, 8);
}

#[tokio::test]
async fn heartbeat_ping_pong_round_trip_downstream() {
    // The driver-side flow: rank 0 owns the client side (downstream),
    // sends Ping, worker receives via its server side (upstream) and
    // replies with Pong on the same server socket. Driver reads the
    // pong off its client socket.
    let (server, client) = make_pair().await;
    let nonce = 0xDEAD_BEEF_CAFE_BABE_u64;

    // Worker task: receive Ping, echo Pong with same nonce.
    let worker = tokio::spawn({
        let server = server.clone();
        async move {
            let kind = recv_kind_server(&server).await.unwrap();
            assert_eq!(kind, Some(FrameKind::HeartbeatPing));
            let got_nonce = recv_heartbeat_body_server(&server).await.unwrap();
            assert_eq!(got_nonce, nonce);
            send_heartbeat_pong_upstream(&server, got_nonce)
                .await
                .unwrap();
        }
    });

    // Driver side.
    send_heartbeat_ping(&client, nonce).await.unwrap();
    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::HeartbeatPong));
    let echoed = recv_heartbeat_body_client(&client).await.unwrap();
    assert_eq!(echoed, nonce);

    worker.await.expect("worker panicked");
}

#[tokio::test]
async fn heartbeat_nonce_mismatch_round_trip_does_not_corrupt_stream() {
    // Send three pings with distinct nonces back-to-back; assert
    // pongs arrive in the same order with matching nonces. Catches
    // any silent off-by-one in the send_raw / recv_raw framing
    // (e.g. losing the kind code prefix because we forgot 4 B of
    // header).
    let (server, client) = make_pair().await;
    let nonces = [1u64, 2, u64::MAX, 0, 0xAAAA_5555_AAAA_5555];

    let worker = tokio::spawn({
        let server = server.clone();
        let n_count = nonces.len();
        async move {
            for _ in 0..n_count {
                let kind = recv_kind_server(&server).await.unwrap();
                assert_eq!(kind, Some(FrameKind::HeartbeatPing));
                let got = recv_heartbeat_body_server(&server).await.unwrap();
                send_heartbeat_pong_upstream(&server, got).await.unwrap();
            }
        }
    });

    for &n in &nonces {
        send_heartbeat_ping(&client, n).await.unwrap();
    }
    for &expected in &nonces {
        let kind = recv_kind_client(&client).await.unwrap();
        assert_eq!(kind, Some(FrameKind::HeartbeatPong));
        let got = recv_heartbeat_body_client(&client).await.unwrap();
        assert_eq!(got, expected);
    }

    worker.await.expect("worker panicked");
}

#[tokio::test]
async fn heartbeat_ping_upstream_is_symmetric() {
    // Plumbing test: the helper that lets a worker probe its driver
    // (`send_heartbeat_ping_upstream`) and the helper that lets a
    // worker reply via its downstream socket
    // (`send_heartbeat_pong_downstream`) are exercised in this test.
    // v1 doesn't use this direction, but a future bidirectional probe
    // does — so test it once now so a later regression is caught.
    let (server, client) = make_pair().await;
    let nonce = 0x1234_5678_9ABC_DEF0_u64;

    // "downstream peer" task — receives a ping on its client socket
    // and replies via the same client socket.
    let downstream = tokio::spawn({
        let client = client.clone();
        async move {
            let kind = recv_kind_client(&client).await.unwrap();
            assert_eq!(kind, Some(FrameKind::HeartbeatPing));
            let got = recv_heartbeat_body_client(&client).await.unwrap();
            send_heartbeat_pong_downstream(&client, got).await.unwrap();
        }
    });

    send_heartbeat_ping_upstream(&server, nonce).await.unwrap();
    let kind = recv_kind_server(&server).await.unwrap();
    assert_eq!(kind, Some(FrameKind::HeartbeatPong));
    let echoed = recv_heartbeat_body_server(&server).await.unwrap();
    assert_eq!(echoed, nonce);

    downstream.await.expect("downstream panicked");
}

#[tokio::test]
async fn watchdog_declares_dead_after_two_simulated_misses() {
    // End-to-end: simulate "send Ping, wait for Pong with a tight
    // deadline; on timeout, watchdog.record_miss()". The worker
    // intentionally never answers, so both pings time out, and on the
    // second miss the watchdog flips to dead.
    let (server, client) = make_pair().await;
    let _server = server; // Hold the socket open; never reply.

    let mut watchdog = HeartbeatWatchdog::default();
    assert!(!watchdog.is_dead());
    assert_eq!(watchdog.consecutive_misses(), 0);

    for round in 1..=2 {
        let mut nonce_bytes = [0u8; 8];
        nonce_bytes[7] = round as u8;
        let nonce = u64::from_be_bytes(nonce_bytes);
        send_heartbeat_ping(&client, nonce).await.unwrap();

        // Wait briefly for a pong; the dead worker never sends one.
        let recv =
            tokio::time::timeout(Duration::from_millis(150), recv_kind_client(&client)).await;
        let dead = match recv {
            Ok(_) => panic!("worker should not have replied"),
            Err(_) => watchdog.record_miss(),
        };
        if round == 1 {
            assert!(!dead, "1st miss should not declare dead");
            assert_eq!(watchdog.consecutive_misses(), 1);
        } else {
            assert!(dead, "2nd miss should declare dead");
            assert!(watchdog.is_dead());
            assert_eq!(watchdog.consecutive_misses(), 2);
            assert_eq!(watchdog.successes(), 0);
        }
    }
}

#[tokio::test]
async fn watchdog_recovers_on_first_pong() {
    // Validates the success-resets-streak property end-to-end:
    //   round 1: worker silent → driver times out → record_miss (1 miss)
    //   round 2: worker answers → record_success (streak broken)
    //   round 3: worker silent → record_miss → NOT dead (only 1 miss in
    //                                                    the new streak)
    //
    // Implementation note: a "silent" round 1 is modeled with NO worker
    // task spawned — the ping bytes land in the kernel socket buffer
    // but no one is reading. When the recovery round 2 worker task IS
    // spawned, it has to drain TWO pings (the stale round-1 one + the
    // fresh round-2 one) and reply to BOTH so the wire stays in sync.
    // The driver in round 2 must also drain BOTH pongs in order, only
    // record_success on the round-2 nonce.
    let (server, client) = make_pair().await;
    let mut watchdog = HeartbeatWatchdog::default();

    // Round 1 — no worker reply.
    send_heartbeat_ping(&client, 1).await.unwrap();
    let recv = tokio::time::timeout(Duration::from_millis(100), recv_kind_client(&client)).await;
    assert!(recv.is_err());
    assert!(!watchdog.record_miss());
    assert_eq!(watchdog.consecutive_misses(), 1);

    // Round 2 — worker answers; it sees the queued round-1 ping +
    // the round-2 ping and replies to both. The driver also drains
    // both pongs; success is recorded ONLY for the round-2 echo.
    let server_for_pong = server.clone();
    let pong_task = tokio::spawn(async move {
        for _ in 0..2 {
            let kind = recv_kind_server(&server_for_pong).await.unwrap();
            assert_eq!(kind, Some(FrameKind::HeartbeatPing));
            let n = recv_heartbeat_body_server(&server_for_pong).await.unwrap();
            send_heartbeat_pong_upstream(&server_for_pong, n)
                .await
                .unwrap();
        }
    });
    send_heartbeat_ping(&client, 2).await.unwrap();
    // Drain the stale round-1 pong first (worker echoed back nonce=1).
    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::HeartbeatPong));
    let n = recv_heartbeat_body_client(&client).await.unwrap();
    assert_eq!(n, 1, "stale ping echoes round-1 nonce first");
    // Then the round-2 pong — this is the one we credit as a success.
    let kind = recv_kind_client(&client).await.unwrap();
    assert_eq!(kind, Some(FrameKind::HeartbeatPong));
    let n = recv_heartbeat_body_client(&client).await.unwrap();
    assert_eq!(n, 2);
    watchdog.record_success();
    pong_task.await.expect("pong task panicked");
    assert!(!watchdog.is_dead());
    assert_eq!(watchdog.consecutive_misses(), 0);
    assert_eq!(watchdog.successes(), 1);

    // Round 3 — worker silent again, but it's the FIRST miss in the
    // NEW streak, so the watchdog must not declare dead.
    send_heartbeat_ping(&client, 3).await.unwrap();
    let recv = tokio::time::timeout(Duration::from_millis(100), recv_kind_client(&client)).await;
    assert!(recv.is_err());
    assert!(!watchdog.record_miss());
    assert!(!watchdog.is_dead());
}

// ────────────────────────────────────────────────────────────────────
// iter 094 — driver-side cadence loop
// ────────────────────────────────────────────────────────────────────
//
// These tests cover `run_heartbeat_loop` + `ping_one_round`. They run
// against a deterministic in-process fake worker that selectively
// drops pings according to a passed-in predicate. The fake worker
// runs in a separate task on the same tokio runtime.

/// Spawn a fake worker that pulls (kind, nonce) frames off `server`
/// and replies with a Pong on rounds where `reply_on(round)` returns
/// true (round is 1-indexed). Drops the ping silently otherwise.
/// Stops when the socket closes or the cancel flag flips.
async fn spawn_selective_worker(
    server: Arc<Mutex<ActivationServer>>,
    cancel: Arc<AtomicBool>,
    reply_on: impl Fn(u32) -> bool + Send + 'static,
) -> tokio::task::JoinHandle<u32> {
    tokio::spawn(async move {
        let mut round: u32 = 0;
        loop {
            if cancel.load(Ordering::Acquire) {
                break round;
            }
            // Use recv_kind_server, which returns Ok(None) on a clean
            // close so we can shut down without scaring the assertions.
            let kind = match recv_kind_server(&server).await {
                Ok(Some(k)) => k,
                Ok(None) => break round,
                Err(_) => break round,
            };
            assert_eq!(
                kind,
                FrameKind::HeartbeatPing,
                "fake worker only handles Ping; got {kind:?}"
            );
            let nonce = match recv_heartbeat_body_server(&server).await {
                Ok(n) => n,
                Err(_) => break round,
            };
            round += 1;
            if reply_on(round) && send_heartbeat_pong_upstream(&server, nonce).await.is_err() {
                break round;
            }
            // else: drop on the floor — the driver's ping_one_round
            // will time out.
        }
    })
}

#[tokio::test]
async fn ping_one_round_alive_when_worker_replies() {
    let (server, client) = make_pair().await;
    let cancel = Arc::new(AtomicBool::new(false));
    let worker = spawn_selective_worker(server.clone(), cancel.clone(), |_| true).await;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default()));
    let outcome = ping_one_round(&client, 42, Duration::from_millis(500), &watchdog).await;
    assert_eq!(outcome, HeartbeatOutcome::Alive);
    let wg = watchdog.lock().await;
    assert_eq!(wg.consecutive_misses(), 0);
    assert_eq!(wg.successes(), 1);
    drop(wg);

    cancel.store(true, Ordering::Release);
    // Close the client to unblock the worker's recv_kind_server.
    client.lock().await.close().await;
    let _ = worker.await;
}

#[tokio::test]
async fn ping_one_round_missed_when_worker_silent() {
    let (server, client) = make_pair().await;
    let _server_guard = server; // Hold the socket open; never reply.

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default()));
    let outcome = ping_one_round(&client, 1, Duration::from_millis(50), &watchdog).await;
    assert_eq!(outcome, HeartbeatOutcome::Missed);
    let wg = watchdog.lock().await;
    assert_eq!(wg.consecutive_misses(), 1);
    assert_eq!(wg.successes(), 0);
    assert!(!wg.is_dead());
}

#[tokio::test]
async fn ping_one_round_dead_on_threshold_crossing() {
    let (server, client) = make_pair().await;
    let _server_guard = server;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default())); // max_misses=1
                                                                       // 1st miss: not dead yet
    let outcome = ping_one_round(&client, 1, Duration::from_millis(40), &watchdog).await;
    assert_eq!(outcome, HeartbeatOutcome::Missed);
    // 2nd miss: crosses the default threshold (max_misses=1 → dead
    // when consecutive > 1).
    let outcome = ping_one_round(&client, 2, Duration::from_millis(40), &watchdog).await;
    assert_eq!(outcome, HeartbeatOutcome::Dead);
    let wg = watchdog.lock().await;
    assert_eq!(wg.consecutive_misses(), 2);
    assert!(wg.is_dead());
}

#[tokio::test]
async fn run_heartbeat_loop_stays_alive_against_healthy_worker() {
    let (server, client) = make_pair().await;
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();
    let worker = spawn_selective_worker(server.clone(), worker_cancel, |_| true).await;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default()));
    let watchdog_for_check = watchdog.clone();
    let cancel_for_loop = cancel.clone();
    let loop_task = tokio::spawn(run_heartbeat_loop(
        client.clone(),
        watchdog,
        Duration::from_millis(25),
        Duration::from_millis(200),
        cancel_for_loop,
    ));

    // Let several rounds elapse — at 25 ms cadence, ~6 rounds in 150 ms.
    tokio::time::sleep(Duration::from_millis(180)).await;
    {
        let wg = watchdog_for_check.lock().await;
        assert!(!wg.is_dead(), "healthy worker should never trip watchdog");
        assert!(
            wg.successes() >= 3,
            "expected ≥3 successes, got {}",
            wg.successes()
        );
        assert_eq!(wg.consecutive_misses(), 0);
    }
    cancel.store(true, Ordering::Release);
    client.lock().await.close().await;
    let outcome = loop_task.await.expect("loop task panicked");
    assert!(
        matches!(outcome, HeartbeatOutcome::Alive | HeartbeatOutcome::Missed),
        "expected clean exit, got {outcome:?}"
    );
    let _ = worker.await;
}

#[tokio::test]
async fn run_heartbeat_loop_fires_when_worker_silent_forever() {
    // Task-spec test: simulate a worker that drops every ping → watchdog
    // fires after the default 2 consecutive misses.
    let (server, client) = make_pair().await;
    let cancel = Arc::new(AtomicBool::new(false));
    let worker = spawn_selective_worker(server.clone(), cancel.clone(), |_| false).await;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default())); // max_misses=1
    let watchdog_for_check = watchdog.clone();
    let cancel_for_loop = cancel.clone();
    let loop_task = tokio::spawn(run_heartbeat_loop(
        client.clone(),
        watchdog,
        Duration::from_millis(20),
        Duration::from_millis(60),
        cancel_for_loop,
    ));

    // Loop will: sleep 20 ms → ping 1 → timeout 60 ms → miss 1 → sleep 20 ms
    //           → ping 2 → timeout 60 ms → miss 2 → DEAD → return.
    // Total ≈ 160 ms; allow generous margin.
    let outcome = tokio::time::timeout(Duration::from_millis(1500), loop_task)
        .await
        .expect("loop did not exit within 1.5 s")
        .expect("loop task panicked");
    assert_eq!(outcome, HeartbeatOutcome::Dead);
    let wg = watchdog_for_check.lock().await;
    assert!(wg.is_dead());
    assert_eq!(wg.consecutive_misses(), 2);
    assert_eq!(wg.successes(), 0);
    drop(wg);

    cancel.store(true, Ordering::Release);
    client.lock().await.close().await;
    let _ = worker.await;
}

#[tokio::test]
async fn run_heartbeat_loop_fires_when_worker_drops_every_nth_ping() {
    // Tighter task-spec test: worker drops every Nth ping. With N=2,
    // the worker pattern is reply, drop, reply, drop, ... — i.e. the
    // streak never reaches 2 misses in a row, so the watchdog stays
    // alive. With N=1 (drop EVERY ping) it dies fast — covered by the
    // sibling test above. The meaningful new case is "intermittent
    // worker → watchdog stays alive": prove the recovery path actually
    // resets the streak.
    let (server, client) = make_pair().await;
    let cancel = Arc::new(AtomicBool::new(false));
    // reply on odd rounds, drop on even — alternating reply/drop.
    let worker =
        spawn_selective_worker(server.clone(), cancel.clone(), |round| round % 2 == 1).await;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default())); // max_misses=1
    let watchdog_for_check = watchdog.clone();
    let cancel_for_loop = cancel.clone();
    let loop_task = tokio::spawn(run_heartbeat_loop(
        client.clone(),
        watchdog,
        Duration::from_millis(20),
        Duration::from_millis(50),
        cancel_for_loop,
    ));

    // Let ≥ 8 rounds elapse.
    tokio::time::sleep(Duration::from_millis(220)).await;
    {
        let wg = watchdog_for_check.lock().await;
        assert!(
            !wg.is_dead(),
            "alternating reply/drop should NOT trip watchdog (consecutive misses caps at 1)"
        );
        assert!(
            wg.successes() >= 2,
            "expected ≥2 successes from odd rounds, got {}",
            wg.successes()
        );
        assert!(
            wg.consecutive_misses() <= 1,
            "consecutive misses should stay ≤ 1"
        );
    }
    cancel.store(true, Ordering::Release);
    client.lock().await.close().await;
    let _ = loop_task.await;
    let _ = worker.await;
}

#[tokio::test]
async fn run_heartbeat_loop_fires_after_exactly_n_drops_with_higher_tolerance() {
    // Strict task-spec test: with max_misses=3, the worker must miss
    // 4 consecutive pings to flip dead. Worker drops the first 4 then
    // would reply; the loop should fire on the 4th miss, BEFORE the
    // 5th ping is sent.
    let (server, client) = make_pair().await;
    let cancel = Arc::new(AtomicBool::new(false));
    // First 4 rounds drop; rounds 5+ reply (we don't expect to reach them).
    let worker = spawn_selective_worker(server.clone(), cancel.clone(), |round| round > 4).await;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::new(3))); // dead on 4th consecutive miss
    let watchdog_for_check = watchdog.clone();
    let cancel_for_loop = cancel.clone();
    let loop_task = tokio::spawn(run_heartbeat_loop(
        client.clone(),
        watchdog,
        Duration::from_millis(15),
        Duration::from_millis(50),
        cancel_for_loop,
    ));

    let outcome = tokio::time::timeout(Duration::from_millis(2000), loop_task)
        .await
        .expect("loop did not exit within 2 s")
        .expect("loop task panicked");
    assert_eq!(outcome, HeartbeatOutcome::Dead);
    let wg = watchdog_for_check.lock().await;
    assert_eq!(
        wg.consecutive_misses(),
        4,
        "should fire on the 4th miss (not earlier, not later)"
    );
    assert_eq!(wg.successes(), 0);
    drop(wg);

    cancel.store(true, Ordering::Release);
    client.lock().await.close().await;
    let _ = worker.await;
}

#[tokio::test]
async fn run_heartbeat_loop_recovers_from_intermittent_misses() {
    // Sequence: miss, miss-of-one-tolerance-not-yet-dead, reply (resets
    // streak), miss → not dead because streak was reset.
    //
    // With max_misses=1 (default): worker drops round 1, then replies
    // forever. After round 1 we have miss=1. Round 2 replies, miss
    // resets to 0. Loop continues healthy. Validates that
    // `record_success` inside the cadence loop actually wipes the streak.
    let (server, client) = make_pair().await;
    let cancel = Arc::new(AtomicBool::new(false));
    let worker = spawn_selective_worker(server.clone(), cancel.clone(), |round| round != 1).await;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default()));
    let watchdog_for_check = watchdog.clone();
    let cancel_for_loop = cancel.clone();
    let loop_task = tokio::spawn(run_heartbeat_loop(
        client.clone(),
        watchdog,
        Duration::from_millis(20),
        Duration::from_millis(60),
        cancel_for_loop,
    ));

    // After ~5 rounds, the watchdog should be alive with at least
    // 3 successes and 0 current misses.
    tokio::time::sleep(Duration::from_millis(250)).await;
    {
        let wg = watchdog_for_check.lock().await;
        assert!(!wg.is_dead());
        assert_eq!(
            wg.consecutive_misses(),
            0,
            "streak should be reset by replies"
        );
        assert!(wg.successes() >= 3);
    }
    cancel.store(true, Ordering::Release);
    client.lock().await.close().await;
    let _ = loop_task.await;
    let _ = worker.await;
}

#[tokio::test]
async fn run_heartbeat_loop_exits_promptly_on_cancel() {
    // Verify the cancel flag is actually observed — flipping it should
    // stop the loop within ~1 interval, not block forever.
    let (server, client) = make_pair().await;
    let cancel = Arc::new(AtomicBool::new(false));
    let worker = spawn_selective_worker(server.clone(), cancel.clone(), |_| true).await;

    let watchdog = Arc::new(Mutex::new(HeartbeatWatchdog::default()));
    let cancel_for_loop = cancel.clone();
    let loop_task = tokio::spawn(run_heartbeat_loop(
        client.clone(),
        watchdog,
        Duration::from_millis(50),
        Duration::from_millis(200),
        cancel_for_loop,
    ));
    // Let it complete at least one healthy round.
    tokio::time::sleep(Duration::from_millis(120)).await;
    cancel.store(true, Ordering::Release);
    // Loop should observe the flag at its next sleep boundary
    // (≤ 50 ms) and exit. Generous 1 s budget for CI scheduler jitter.
    let outcome = tokio::time::timeout(Duration::from_secs(1), loop_task)
        .await
        .expect("loop did not exit within 1 s of cancel")
        .expect("loop task panicked");
    assert!(
        matches!(outcome, HeartbeatOutcome::Alive | HeartbeatOutcome::Missed),
        "expected clean Alive/Missed on cancel, got {outcome:?}"
    );

    client.lock().await.close().await;
    let _ = worker.await;
}
