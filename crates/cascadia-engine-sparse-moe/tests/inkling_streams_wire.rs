//! Multi-stream decode across a real 3-rank loopback pipeline: the tokens
//! each task receives must equal what the single-stage engine produces for
//! the same tasks, with streams admitted while others decode and slots
//! reused after a stream finishes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cascadia_engine::Engine;
use cascadia_engine_sparse_moe::dist::{
    recv_kind_client, recv_stream_tokens_body_client, send_forward, send_reset, send_stream_decode,
    send_stream_feed, send_stream_open, FrameKind, StageTransport, STREAM_FEED_OPEN,
};
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::sampling::SamplingConfig;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_transport::{ActivationClient, ActivationServer};
use cascadia_types::{FinishReason, GenerationTask};
use tokio::sync::Mutex;

fn fixture() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    if dir.join("tokenizer.json").exists() {
        Some(dir)
    } else {
        eprintln!("inkling_export fixture (with tokenizer.json) missing; skipping");
        None
    }
}

const PROMPTS: [&str; 5] = [
    "a5 a33 a81 a53 a92 a85 a83 a43 a74 a10 a86 a84",
    "a84 a86 a10 a74 a43 a83 a85 a92 a53 a81 a33 a5",
    "a12 a60 a99 a3 a47 a101 a66 a9",
    "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14",
    "a7 a7 a7 a100 a2",
];
const MAX_TOKENS: [u32; 5] = [8, 8, 5, 8, 6];

fn tasks() -> Vec<GenerationTask> {
    PROMPTS
        .iter()
        .zip(MAX_TOKENS)
        .enumerate()
        .map(|(i, (p, n))| {
            let mut t = GenerationTask::new(format!("t{i}"), *p);
            t.max_tokens = n;
            t.temperature = 0.0;
            t
        })
        .collect()
}

/// Drive `engine` on the calling (non-async) thread until every task in
/// `ids` has produced its final marker; returns each task's token ids.
fn collect(engine: &mut dyn Engine, ids: &[String]) -> Vec<Vec<i64>> {
    let mut toks: Vec<Vec<i64>> = vec![Vec::new(); ids.len()];
    let mut done = vec![false; ids.len()];
    let mut steps = 0;
    while done.iter().any(|d| !d) {
        steps += 1;
        assert!(steps < 500, "engine did not finish the tasks");
        let chunks = engine.step().expect("step");
        for (id, c) in chunks {
            let i = ids.iter().position(|x| x == &id).expect("known task");
            assert!(c.error.is_none(), "task {id} errored: {:?}", c.error);
            if c.is_final {
                done[i] = true;
            } else {
                toks[i].push(c.token_id);
            }
        }
    }
    toks
}

async fn link() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.unwrap();
    let port = server.port();
    let server = Arc::new(Mutex::new(server));
    let sc = server.clone();
    let accept = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let mut client = ActivationClient::new("127.0.0.1", port);
    client
        .connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    accept.await.unwrap();
    (server, Arc::new(Mutex::new(client)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_rank_multistream_matches_single_stage() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ids: Vec<String> = (0..PROMPTS.len()).map(|i| format!("t{i}")).collect();

    // Reference: the single-stage one-task path.
    let expected = {
        let runner =
            InklingRunner::load_staged(&dir, 64, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
        let mut e = PipelineEngine::new(
            runner,
            Some(tok.clone()),
            StageTransport::default(),
            handle.clone(),
            0,
            1,
            None,
        );
        let ids2 = ids.clone();
        tokio::task::spawn_blocking(move || {
            for t in tasks() {
                e.submit(t).unwrap();
            }
            collect(&mut e, &ids2)
        })
        .await
        .unwrap()
    };
    assert!(expected.iter().all(|t| !t.is_empty()));

    // Pipeline: 3 ranks over loopback, layers [0,2) [2,3) [3,4), 3 stream slots each
    // (5 tasks -> slots are reused).
    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let r0 = InklingRunner::load_staged(&dir, 64, 0, 3, 0, 2, Some("eager".into()), None).unwrap();
    let r1 = InklingRunner::load_staged(&dir, 64, 1, 3, 2, 3, Some("eager".into()), None).unwrap();
    let r2 = InklingRunner::load_staged(&dir, 64, 2, 3, 3, 4, Some("eager".into()), None).unwrap();
    let mut e0 = PipelineEngine::new(
        r0,
        Some(tok),
        StageTransport {
            upstream: None,
            downstream: Some(c01.clone()),
        },
        handle.clone(),
        0,
        3,
        None,
    );
    let mut e1 = PipelineEngine::new(
        r1,
        None,
        StageTransport {
            upstream: Some(s01),
            downstream: Some(c12.clone()),
        },
        handle.clone(),
        1,
        3,
        None,
    );
    let mut e2 = PipelineEngine::new(
        r2,
        None,
        StageTransport {
            upstream: Some(s12),
            downstream: None,
        },
        handle.clone(),
        2,
        3,
        None,
    );
    assert_eq!(e0.enable_streams(3), 3);
    assert_eq!(e1.enable_streams(3), 3);
    assert_eq!(e2.enable_streams(3), 3);

    let stop = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = [
        (Box::new(e1) as Box<dyn Engine>, stop.clone()),
        (Box::new(e2) as Box<dyn Engine>, stop.clone()),
    ]
    .into_iter()
    .map(|(mut e, stop)| {
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if e.step().is_err() {
                    break;
                }
            }
        })
    })
    .collect();

    let ids2 = ids.clone();
    let got = tokio::task::spawn_blocking(move || {
        for t in tasks() {
            e0.submit(t).unwrap();
        }
        let got = collect(&mut e0, &ids2);
        drop(e0);
        got
    })
    .await
    .unwrap();
    stop.store(true, Ordering::Relaxed);
    // closing rank 0's client ends the workers' recv loops
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    for w in workers {
        let _ = w.join();
    }

    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            g, e,
            "task {i}: 3-rank multi-stream tokens differ from the single-stage path"
        );
        assert_eq!(g.len() as u32, MAX_TOKENS[i], "task {i}: token count");
    }
}

/// A worker step loop as the relay loop runs it, until `stop`. Returns
/// whether any step failed: a worker re-links in place on a lost link, so
/// none may (a failed step would end the relay loop, and nothing rebuilds an
/// embedded stage).
fn spawn_worker(mut e: Box<dyn Engine>, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<bool> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            if e.step().is_err() {
                return true;
            }
        }
        false
    })
}

/// Join a worker from `spawn_worker` and assert it neither panicked nor failed a step.
async fn join_worker(w: std::thread::JoinHandle<bool>, rank: &str) {
    let r = tokio::task::spawn_blocking(move || w.join()).await.unwrap();
    assert!(r.is_ok(), "{rank} panicked");
    assert!(!r.unwrap(), "{rank} failed a step instead of re-linking");
}

/// A last rank whose upstream dies hard (TCP reset, as when the previous
/// rank's process is killed) keeps its listener and serves the upstream's
/// next dial in the same process. `streams` = 0 runs the one-task path, more
/// runs the stream-mode readiness wait.
async fn last_rank_reaccepts_after_upstream_reset(streams: usize) {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.unwrap();
    let port = server.port();
    let server = Arc::new(Mutex::new(server));
    let sc = server.clone();
    let accept = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let upstream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    accept.await.unwrap();

    let r2 = InklingRunner::load_staged(&dir, 64, 2, 3, 3, 4, Some("eager".into()), None).unwrap();
    let hs = r2.hidden_size();
    let mut e2 = PipelineEngine::new(
        r2,
        None,
        StageTransport {
            upstream: Some(server),
            downstream: None,
        },
        handle.clone(),
        2,
        3,
        None,
    );
    if streams > 0 {
        assert_eq!(e2.enable_streams(streams), streams);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let worker = spawn_worker(Box::new(e2), stop.clone());
    // Let the worker settle into its blocking receive, then kill the peer
    // hard: linger 0 turns the close into a reset. (tokio deprecates the
    // setter because a non-zero linger blocks the thread on drop; zero does
    // not.)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    #[allow(deprecated)]
    upstream
        .set_linger(Some(std::time::Duration::ZERO))
        .unwrap();
    drop(upstream);

    // The restarted upstream dials the same port and is served.
    let mut c = ActivationClient::new("127.0.0.1", port);
    c.connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    let c = Mutex::new(c);
    let cfg = SamplingConfig::default();
    let reply = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        if streams > 0 {
            send_stream_open(&c, 1, 0, &cfg, &vec![0.1; 2 * hs], 2, hs as u32)
                .await
                .unwrap();
            recv_kind_client(&c).await
        } else {
            send_reset(&c).await.unwrap();
            send_forward(&c, 0, &cfg, &vec![0.1; hs], [1, 1, hs as u32], true)
                .await
                .unwrap();
            recv_kind_client(&c).await
        }
    })
    .await
    .expect("the last rank did not serve its re-dialed upstream");
    let want = if streams > 0 {
        FrameKind::StreamTokens
    } else {
        FrameKind::Token
    };
    assert!(
        matches!(reply, Ok(Some(k)) if k == want),
        "expected {want:?} on the new link, got {reply:?}"
    );
    stop.store(true, Ordering::Relaxed);
    c.lock().await.close().await;
    join_worker(worker, "the last rank").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn last_rank_reaccepts_after_upstream_reset_one_task() {
    last_rank_reaccepts_after_upstream_reset(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn last_rank_reaccepts_after_upstream_reset_streams() {
    last_rank_reaccepts_after_upstream_reset(3).await;
}

/// A lone stream-mode last rank (rank 2 of 3) that the test feeds frames by
/// hand. Returns the test's end of the link, the hidden width, the stop flag
/// and the worker thread (see `spawn_worker`).
async fn lone_last_rank(
    dir: &std::path::Path,
    handle: &tokio::runtime::Handle,
) -> (
    Arc<Mutex<ActivationClient>>,
    usize,
    Arc<AtomicBool>,
    std::thread::JoinHandle<bool>,
) {
    let (s, c) = link().await;
    let r2 = InklingRunner::load_staged(dir, 64, 2, 3, 3, 4, Some("eager".into()), None).unwrap();
    let hs = r2.hidden_size();
    let mut e2 = PipelineEngine::new(
        r2,
        None,
        StageTransport {
            upstream: Some(s),
            downstream: None,
        },
        handle.clone(),
        2,
        3,
        None,
    );
    assert_eq!(e2.enable_streams(3), 3);
    let stop = Arc::new(AtomicBool::new(false));
    let worker = spawn_worker(Box::new(e2), stop.clone());
    (c, hs, stop, worker)
}

/// The last rank drops a link that sent it a frame it rejected: the test's
/// end reads EOF (or an error) instead of a reply, within 5 s.
async fn assert_link_closed(c: &Mutex<ActivationClient>, what: &str) {
    let reply = tokio::time::timeout(std::time::Duration::from_secs(5), recv_kind_client(c))
        .await
        .unwrap_or_else(|_| panic!("{what}: the last rank neither replied nor closed the link"));
    assert!(
        !matches!(reply, Ok(Some(_))),
        "{what}: expected the link to close, got {reply:?}"
    );
}

/// A `StreamDecode` frame that names a free slot, or one slot two times, is
/// rejected: the worker drops the link and keeps stepping, without a panic.
/// Before the fix both reached the runner's asserts and panicked the worker
/// process.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bad_stream_decode_frame_is_rejected_without_panic() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();

    // A free slot at position 0.
    let (c, hs, stop, worker) = lone_last_rank(&dir, &handle).await;
    send_stream_decode(&c, 1, &[(0, 0)], &vec![0.1; hs], hs as u32)
        .await
        .unwrap();
    assert_link_closed(&c, "a decode of a free slot").await;
    stop.store(true, Ordering::Relaxed);
    join_worker(worker, "a decode of a free slot: the last rank").await;

    // An open slot listed two times.
    let (c, hs, stop, worker) = lone_last_rank(&dir, &handle).await;
    let cfg = SamplingConfig::default();
    send_stream_open(&c, 1, 0, &cfg, &vec![0.1; 2 * hs], 2, hs as u32)
        .await
        .unwrap();
    assert!(matches!(
        recv_kind_client(&c).await,
        Ok(Some(FrameKind::StreamTokens))
    ));
    recv_stream_tokens_body_client(&c).await.unwrap();
    send_stream_decode(&c, 2, &[(0, 2), (0, 2)], &vec![0.1; 2 * hs], hs as u32)
        .await
        .unwrap();
    assert_link_closed(&c, "a decode that lists one slot two times").await;
    stop.store(true, Ordering::Relaxed);
    join_worker(worker, "a duplicate-slot decode: the last rank").await;
}

/// A stream frame that the last rank cannot serve, here a decode of a slot
/// whose prompt it never sampled, must close the link at once. Before the
/// fix the worker logged it and kept going without a reply, so rank 0 waited
/// its whole reply deadline and then reported a dead peer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_stream_frame_closes_the_link_at_once() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let (c, hs, stop, worker) = lone_last_rank(&dir, &handle).await;
    // The first window of a long prompt opens the slot; only the final
    // window would create its sampler.
    let cfg = SamplingConfig::default();
    send_stream_feed(
        &c,
        1,
        0,
        STREAM_FEED_OPEN,
        &cfg,
        &vec![0.1; 2 * hs],
        2,
        hs as u32,
    )
    .await
    .unwrap();
    send_stream_decode(&c, 2, &[(0, 2)], &vec![0.1; hs], hs as u32)
        .await
        .unwrap();
    assert_link_closed(&c, "a decode of an unsampled slot").await;
    stop.store(true, Ordering::Relaxed);
    join_worker(worker, "the last rank").await;
}

/// A 3-rank loopback pipeline with `caps[r]` stream slots on rank `r`.
/// Returns rank 0, the stop flag, rank 0's two links and the worker threads
/// for ranks 1 and 2 (see `spawn_worker`).
#[allow(clippy::type_complexity)]
async fn pipeline(
    dir: &std::path::Path,
    handle: &tokio::runtime::Handle,
    caps: [usize; 3],
) -> (
    PipelineEngine<InklingRunner>,
    Arc<AtomicBool>,
    [Arc<Mutex<ActivationClient>>; 2],
    Vec<std::thread::JoinHandle<bool>>,
) {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let load = |rank, lo, hi| {
        InklingRunner::load_staged(dir, 64, rank, 3, lo, hi, Some("eager".into()), None).unwrap()
    };
    let transport = |up, down| StageTransport {
        upstream: up,
        downstream: down,
    };
    let mut e0 = PipelineEngine::new(
        load(0, 0, 2),
        Some(tok),
        transport(None, Some(c01.clone())),
        handle.clone(),
        0,
        3,
        None,
    );
    let mut e1 = PipelineEngine::new(
        load(1, 2, 3),
        None,
        transport(Some(s01), Some(c12.clone())),
        handle.clone(),
        1,
        3,
        None,
    );
    let mut e2 = PipelineEngine::new(
        load(2, 3, 4),
        None,
        transport(Some(s12), None),
        handle.clone(),
        2,
        3,
        None,
    );
    for (e, n) in [&mut e0, &mut e1, &mut e2].into_iter().zip(caps) {
        assert_eq!(e.enable_streams(n), n);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let workers = [Box::new(e1) as Box<dyn Engine>, Box::new(e2)]
        .into_iter()
        .map(|e| spawn_worker(e, stop.clone()))
        .collect();
    (e0, stop, [c01, c12], workers)
}

/// Rank 1 configured with fewer stream slots than rank 0 (`CASCADIA_STREAMS`
/// not set on every rank). Rank 1 must drop the link at the first slot it
/// cannot hold (and re-link, not fail its step), and rank 0's task must get an
/// error chunk soon, not hang until its reply deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rank_with_fewer_slots_fails_fast() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let (e0, stop, [c01, c12], workers) = pipeline(&dir, &handle, [3, 1, 3]).await;
    let started = std::time::Instant::now();
    let (e0, got) = tokio::task::spawn_blocking(move || {
        let mut e0 = e0;
        let got = run_round(&mut e0, "mismatch", 3);
        (e0, got)
    })
    .await
    .unwrap();
    assert!(
        got.iter().any(|r| r.is_err()),
        "a stream on a slot rank 1 cannot hold must fail, got {got:?}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "and fail fast, not at the reply deadline"
    );

    stop.store(true, Ordering::Relaxed);
    drop(e0);
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    for (w, rank) in workers.into_iter().zip(["rank 1", "rank 2"]) {
        join_worker(w, rank).await;
    }
}

/// One task that asks for more tokens than the 64-position budget holds
/// must end with `FinishReason::Length`, on a single stage and on the
/// pipeline, with the same tokens.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn over_budget_stream_finishes_with_length() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    fn run(e: &mut dyn Engine) -> (Vec<i64>, Option<FinishReason>) {
        let mut t = GenerationTask::new("long".to_string(), PROMPTS[0]);
        t.max_tokens = 100;
        t.temperature = 0.0;
        e.submit(t).unwrap();
        let mut toks = Vec::new();
        for _ in 0..500 {
            for (_, c) in e.step().expect("step") {
                assert!(c.error.is_none(), "task errored: {:?}", c.error);
                if c.is_final {
                    return (toks, c.finish_reason);
                }
                toks.push(c.token_id);
            }
        }
        panic!("the task did not finish");
    }

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let runner =
        InklingRunner::load_staged(&dir, 64, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
    let mut single = PipelineEngine::new(
        runner,
        Some(tok),
        StageTransport::default(),
        handle.clone(),
        0,
        1,
        None,
    );
    assert_eq!(single.enable_streams(3), 3);
    let (single_toks, single_reason) = tokio::task::spawn_blocking(move || run(&mut single))
        .await
        .unwrap();
    assert_eq!(single_reason, Some(FinishReason::Length), "single stage");
    assert!(!single_toks.is_empty() && single_toks.len() < 100);

    let (e0, stop, [c01, c12], workers) = pipeline(&dir, &handle, [3, 3, 3]).await;
    let (piped_toks, piped_reason) = tokio::task::spawn_blocking(move || {
        let mut e0 = e0;
        run(&mut e0)
    })
    .await
    .unwrap();
    stop.store(true, Ordering::Relaxed);
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    for (w, rank) in workers.into_iter().zip(["rank 1", "rank 2"]) {
        join_worker(w, rank).await;
    }
    assert_eq!(piped_reason, Some(FinishReason::Length), "pipeline");
    assert_eq!(
        piped_toks, single_toks,
        "pipeline tokens differ from single stage"
    );
}

/// Like `collect`, but a task may end in an error chunk (returned as Err) and
/// a step may report a dead link once (ignored: the driver keeps stepping).
fn collect_lenient(engine: &mut dyn Engine, ids: &[String]) -> Vec<Result<Vec<i64>, String>> {
    let mut res: Vec<Result<Vec<i64>, String>> = ids.iter().map(|_| Ok(Vec::new())).collect();
    let mut done = vec![false; ids.len()];
    let mut steps = 0;
    while done.iter().any(|d| !d) {
        steps += 1;
        assert!(steps < 500, "engine did not finish the tasks");
        let Ok(chunks) = engine.step() else { continue };
        for (id, c) in chunks {
            let i = ids.iter().position(|x| x == &id).expect("known task");
            if let Some(err) = c.error.clone() {
                res[i] = Err(err.to_string());
                done[i] = true;
            } else if c.is_final {
                done[i] = true;
            } else if let Ok(t) = &mut res[i] {
                t.push(c.token_id);
            }
        }
    }
    res
}

/// Submit the first `n` prompts under fresh task ids and collect them.
fn run_round(e0: &mut dyn Engine, tag: &str, n: usize) -> Vec<Result<Vec<i64>, String>> {
    let ids: Vec<String> = (0..n).map(|i| format!("{tag}-{i}")).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut t = GenerationTask::new(id.clone(), PROMPTS[i]);
        t.max_tokens = MAX_TOKENS[i];
        t.temperature = 0.0;
        e0.submit(t).unwrap();
    }
    collect_lenient(e0, &ids)
}

/// Ranks 1 and 2 of the loopback pipeline plus a TCP forwarder standing in
/// for the wire between rank 0 and rank 1. Cutting the forwarder kills the
/// link the way a dying neighbour or network does: rank 0's own socket stays
/// open and reads EOF (closing rank 0's client from the test would not
/// exercise that). Ranks 1 and 2 keep running across a cut, as they do in the
/// embedded node, where the process does not restart: they re-link in place
/// once rank 0 dials in again.
struct Tail {
    /// Forwarder (rank 0 dials it) and rank 1's listener.
    ports: (u16, u16),
    /// Completes when the forwarder has carried rank 0's next dial to rank 1.
    dialed: Option<tokio::sync::oneshot::Receiver<()>>,
    fwd: Option<tokio::task::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    workers: Vec<std::thread::JoinHandle<bool>>,
}

/// Accept every dial on `listener` and splice it to rank 1's port. Aborting
/// the task drops the listener and every spliced connection.
fn forward(
    listener: tokio::net::TcpListener,
    to: u16,
    dialed: tokio::sync::oneshot::Sender<()>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut dialed = Some(dialed);
        let mut links = tokio::task::JoinSet::new();
        loop {
            let Ok((mut a, _)) = listener.accept().await else {
                return;
            };
            let Ok(mut b) = tokio::net::TcpStream::connect(("127.0.0.1", to)).await else {
                continue;
            };
            a.set_nodelay(true).ok();
            b.set_nodelay(true).ok();
            if let Some(d) = dialed.take() {
                let _ = d.send(());
            }
            links.spawn(async move {
                let _ = tokio::io::copy_bidirectional(&mut a, &mut b).await;
            });
        }
    })
}

impl Tail {
    async fn start(dir: &std::path::Path, handle: &tokio::runtime::Handle) -> Tail {
        let (s12, c12) = link().await;

        // Rank 1 takes rank 0's first dial the way its builder does; later
        // dials it re-accepts itself after a lost link.
        let mut s01 = ActivationServer::new("127.0.0.1", 0);
        s01.start().await.unwrap();
        let p1 = s01.port();
        let s01 = Arc::new(Mutex::new(s01));
        let sc = s01.clone();
        tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });

        let load = |rank, lo, hi| {
            InklingRunner::load_staged(dir, 64, rank, 3, lo, hi, Some("eager".into()), None)
                .unwrap()
        };
        let mut e1 = PipelineEngine::new(
            load(1, 2, 3),
            None,
            StageTransport {
                upstream: Some(s01),
                downstream: Some(c12),
            },
            handle.clone(),
            1,
            3,
            None,
        );
        let mut e2 = PipelineEngine::new(
            load(2, 3, 4),
            None,
            StageTransport {
                upstream: Some(s12),
                downstream: None,
            },
            handle.clone(),
            2,
            3,
            None,
        );
        assert_eq!(e1.enable_streams(3), 3);
        assert_eq!(e2.enable_streams(3), 3);
        let stop = Arc::new(AtomicBool::new(false));
        let workers = [Box::new(e1) as Box<dyn Engine>, Box::new(e2)]
            .into_iter()
            .map(|e| spawn_worker(e, stop.clone()))
            .collect();
        let mut tail = Tail {
            ports: (0, p1),
            dialed: None,
            fwd: None,
            stop,
            workers,
        };
        tail.restore(0).await;
        tail
    }

    /// The wire to rank 0 comes back on the forwarder port (`0` picks one).
    async fn restore(&mut self, port: u16) {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .unwrap();
        self.ports.0 = listener.local_addr().unwrap().port();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.dialed = Some(rx);
        self.fwd = Some(forward(listener, self.ports.1, tx));
    }

    /// The wire to rank 0 is cut and stays down (dials are refused) until
    /// `restore`. Rank 1 reads EOF and re-links; so does rank 2.
    async fn cut(&mut self) {
        if let Some(f) = self.fwd.take() {
            f.abort();
            // Wait for the listener and the spliced links to be dropped.
            let _ = f.await;
        }
    }

    /// End the test: cut, stop the workers, and assert neither failed a step.
    async fn stop(mut self) {
        self.cut().await;
        self.stop.store(true, Ordering::Relaxed);
        for (w, rank) in self.workers.into_iter().zip(["rank 1", "rank 2"]) {
            join_worker(w, rank).await;
        }
    }
}

/// Rank 0 keeps its process (and its API) across a restart of the ranks
/// behind it. Before the fix it kept the dead socket and answered every
/// request with its error until someone restarted rank 0 by hand (seen on the
/// four-box bed after a re-install, and after an idle-ceiling teardown).
///
/// 1. neighbours restart while rank 0 idles: the next requests are served on
///    a fresh connection and none fails (the idle link is probed before
///    admitting);
/// 2. neighbours down: a request fails fast with an error instead of hanging;
/// 3. neighbours back: requests are served again, same tokens throughout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rank0_redials_after_downstream_restart() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let n = PROMPTS.len();

    let mut tail = Tail::start(&dir, &handle).await;
    let mut c01 = ActivationClient::new("127.0.0.1", tail.ports.0);
    c01.connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    let r0 = InklingRunner::load_staged(&dir, 64, 0, 3, 0, 2, Some("eager".into()), None).unwrap();
    let mut e0 = PipelineEngine::new(
        r0,
        Some(tok),
        StageTransport {
            upstream: None,
            downstream: Some(Arc::new(Mutex::new(c01))),
        },
        handle.clone(),
        0,
        3,
        None,
    );
    assert_eq!(e0.enable_streams(3), 3);

    // Rank 0 steps on a blocking thread; it travels in and out of each round.
    async fn round(
        e0: PipelineEngine<InklingRunner>,
        tag: &'static str,
        n: usize,
    ) -> (PipelineEngine<InklingRunner>, Vec<Result<Vec<i64>, String>>) {
        tokio::task::spawn_blocking(move || {
            let mut e0 = e0;
            let got = run_round(&mut e0, tag, n);
            (e0, got)
        })
        .await
        .unwrap()
    }

    let (e0, healthy) = round(e0, "healthy", n).await;
    let expected: Vec<Vec<i64>> = healthy
        .into_iter()
        .map(|r| r.expect("healthy round"))
        .collect();
    assert!(expected.iter().all(|t| !t.is_empty()));

    // 1. The link drops while rank 0 idles, and no request arrives. Rank 0
    // must notice the dead link and dial rank 1 again by itself: until it
    // does, rank 1 waits for that dial, and on a real fleet every rank behind
    // it waits on the one before (seen with 11 boxes: ranks 1-5 all "waiting
    // for the previous rank to dial in" behind a rank 0 that was "serving").
    let port = tail.ports.0;
    tail.cut().await;
    tail.restore(port).await;
    let dialed = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tail.dialed.take().unwrap(),
    )
    .await;
    assert!(
        dialed.is_ok(),
        "rank 0 did not re-dial rank 1 while idle (no request was made)"
    );
    let (e0, got) = round(e0, "restarted", n).await;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g.as_ref(), Ok(e), "task {i} after the link came back");
    }

    // 2. The neighbours are unreachable.
    tail.cut().await;
    let started = std::time::Instant::now();
    let (e0, got) = round(e0, "down", 1).await;
    assert!(
        got[0].is_err(),
        "a request with the pipeline down must fail, got {:?}",
        got[0]
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "and fail fast"
    );

    // 3. They come back; rank 0 re-dials once its retry interval has passed.
    tail.restore(port).await;
    tokio::time::sleep(std::time::Duration::from_millis(3200)).await;
    let (e0, got) = round(e0, "back", n).await;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g.as_ref(), Ok(e), "task {i} after the pipeline came back");
    }

    drop(e0);
    tail.stop().await;
}

/// The link to rank 1 dies while streams are in flight: each in-flight task
/// ends with exactly one error chunk and gets no chunk after it, and once the
/// neighbours are back, rank 0 re-dials and serves the next round.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn link_loss_mid_round_fails_each_stream_once() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let n = PROMPTS.len();

    let mut tail = Tail::start(&dir, &handle).await;
    let mut c01 = ActivationClient::new("127.0.0.1", tail.ports.0);
    c01.connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    let r0 = InklingRunner::load_staged(&dir, 64, 0, 3, 0, 2, Some("eager".into()), None).unwrap();
    let mut e0 = PipelineEngine::new(
        r0,
        Some(tok),
        StageTransport {
            upstream: None,
            downstream: Some(Arc::new(Mutex::new(c01))),
        },
        handle.clone(),
        0,
        3,
        None,
    );
    assert_eq!(e0.enable_streams(3), 3);

    let (e0, healthy) = tokio::task::spawn_blocking(move || {
        let got = run_round(&mut e0, "healthy", n);
        (e0, got)
    })
    .await
    .unwrap();
    let expected: Vec<Vec<i64>> = healthy
        .into_iter()
        .map(|r| r.expect("healthy round"))
        .collect();

    // 3 tasks, one per slot and group. After the first step that emits a
    // token, every group has a decode frame in flight; the test then cuts
    // the link and lets rank 0 step on.
    let (in_flight_tx, in_flight_rx) = std::sync::mpsc::channel::<()>();
    let (cut_tx, cut_rx) = std::sync::mpsc::channel::<()>();
    let cut = tokio::task::spawn_blocking(move || {
        let mut e0 = e0;
        let ids: Vec<String> = (0..3).map(|i| format!("cut-{i}")).collect();
        for (i, id) in ids.iter().enumerate() {
            let mut t = GenerationTask::new(id.clone(), PROMPTS[i]);
            t.max_tokens = MAX_TOKENS[i];
            t.temperature = 0.0;
            e0.submit(t).unwrap();
        }
        let mut errors = vec![0usize; ids.len()];
        let mut ended = vec![false; ids.len()];
        let mut signalled = false;
        let mut steps = 0;
        while ended.iter().any(|d| !d) {
            steps += 1;
            assert!(steps < 500, "engine did not end the tasks");
            let Ok(chunks) = e0.step() else { continue };
            let mut token = false;
            for (id, c) in chunks {
                let i = ids.iter().position(|x| x == &id).expect("known task");
                assert!(!ended[i], "task {id} got a chunk after its end: {c:?}");
                if c.error.is_some() {
                    errors[i] += 1;
                    ended[i] = true;
                } else {
                    assert!(!c.is_final, "task {id} ended without the cut: {c:?}");
                    token = true;
                }
            }
            if token && !signalled {
                signalled = true;
                in_flight_tx.send(()).unwrap();
                cut_rx.recv().unwrap();
            }
        }
        // A few more steps: nothing more may arrive for the ended tasks.
        for _ in 0..5 {
            if let Ok(chunks) = e0.step() {
                assert!(
                    chunks.is_empty(),
                    "chunks after the tasks ended: {chunks:?}"
                );
            }
        }
        (e0, errors)
    });
    tokio::task::spawn_blocking(move || in_flight_rx.recv().unwrap())
        .await
        .unwrap();
    let port = tail.ports.0;
    tail.cut().await;
    cut_tx.send(()).unwrap();
    let (e0, errors) = cut.await.unwrap();
    assert_eq!(
        errors,
        vec![1; 3],
        "each in-flight task must end with exactly one error chunk"
    );

    // The link comes back; rank 0 re-dials once its retry interval has passed.
    tail.restore(port).await;
    tokio::time::sleep(std::time::Duration::from_millis(3200)).await;
    let (e0, got) = tokio::task::spawn_blocking(move || {
        let mut e0 = e0;
        let got = run_round(&mut e0, "back", n);
        (e0, got)
    })
    .await
    .unwrap();
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g.as_ref(), Ok(e), "task {i} after the link came back");
    }

    drop(e0);
    tail.stop().await;
}

/// Submit t0-t2, cancel t1 after its second token while t0 and t2 decode,
/// then submit t3, which can only take t1's freed slot. Returns the tokens
/// of t0-t3 (t1's up to the cancel) and the chunks t1 got after the cancel.
fn run_with_cancel(e: &mut dyn Engine) -> (Vec<Vec<i64>>, usize) {
    let tasks = tasks();
    for t in &tasks[..3] {
        e.submit(t.clone()).unwrap();
    }
    let ids: Vec<String> = (0..4).map(|i| format!("t{i}")).collect();
    let mut toks: Vec<Vec<i64>> = vec![Vec::new(); 4];
    let mut done = [false, true, false, false];
    let mut cancelled = false;
    let mut after = 0;
    let mut steps = 0;
    while done.iter().any(|d| !d) {
        steps += 1;
        assert!(steps < 500, "engine did not finish the tasks");
        for (id, c) in e.step().expect("step") {
            let i = ids.iter().position(|x| x == &id).expect("known task");
            assert!(c.error.is_none(), "task {id} errored: {:?}", c.error);
            if i == 1 && cancelled {
                after += 1;
            } else if c.is_final {
                done[i] = true;
            } else {
                toks[i].push(c.token_id);
            }
        }
        if !cancelled && toks[1].len() >= 2 {
            e.cancel(&ids[1]);
            cancelled = true;
            e.submit(tasks[3].clone()).unwrap();
        }
    }
    (toks, after)
}

/// A stream cancelled during decode, with its frame in flight, retires
/// silently on rank 0 and on the ranks behind; the other streams and the
/// task that takes its slot still get the single-stage tokens.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_rank_cancel_during_decode_frees_the_slot() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ids: Vec<String> = (0..PROMPTS.len()).map(|i| format!("t{i}")).collect();
    let expected = {
        let runner =
            InklingRunner::load_staged(&dir, 64, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
        let mut e = PipelineEngine::new(
            runner,
            Some(tok.clone()),
            StageTransport::default(),
            handle.clone(),
            0,
            1,
            None,
        );
        tokio::task::spawn_blocking(move || {
            for t in tasks() {
                e.submit(t).unwrap();
            }
            collect(&mut e, &ids)
        })
        .await
        .unwrap()
    };

    let tail = Tail::start(&dir, &handle).await;
    let mut c01 = ActivationClient::new("127.0.0.1", tail.ports.0);
    c01.connect_with_timeout(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    let r0 = InklingRunner::load_staged(&dir, 64, 0, 3, 0, 2, Some("eager".into()), None).unwrap();
    let mut e0 = PipelineEngine::new(
        r0,
        Some(tok),
        StageTransport {
            upstream: None,
            downstream: Some(Arc::new(Mutex::new(c01))),
        },
        handle.clone(),
        0,
        3,
        None,
    );
    assert_eq!(e0.enable_streams(3), 3);
    let (got, after) = tokio::task::spawn_blocking(move || {
        let r = run_with_cancel(&mut e0);
        drop(e0);
        r
    })
    .await
    .unwrap();
    tail.stop().await;
    assert_eq!(after, 0, "the cancelled task got chunks after the cancel");
    assert_eq!(got[1], expected[1][..2], "t1 before the cancel");
    for i in [0, 2, 3] {
        assert_eq!(got[i], expected[i], "task {i} next to a cancelled stream");
    }
}
