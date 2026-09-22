//! Multi-stream decode across a real 3-rank loopback pipeline: the tokens
//! each task receives must equal what the single-stage engine produces for
//! the same tasks, with streams admitted while others decode and slots
//! reused after a stream finishes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cascadia_engine::Engine;
use cascadia_engine_sparse_moe::dist::StageTransport;
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_transport::{ActivationClient, ActivationServer};
use cascadia_types::GenerationTask;
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
        // Five requests at once with three slots: the first three must have
        // gone down as one prefill frame (they share expert reads), and the
        // tokens below must not notice.
        assert!(
            e0.batched_admissions() >= 3,
            "a burst was admitted one frame per prompt ({} batched)",
            e0.batched_admissions()
        );
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

/// A last rank whose upstream dies hard (TCP reset, as when the previous
/// rank's process is killed) must exit its step loop with an error so the
/// supervisor rebuilds it, instead of spinning on `NotConnected` forever
/// while the restarted upstream can never reconnect (the listener accepted
/// exactly once).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn last_rank_exits_after_upstream_reset() {
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
    let exited = Arc::new(AtomicBool::new(false));
    let flag = exited.clone();
    let worker = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while std::time::Instant::now() < deadline {
            if e2.step().is_err() {
                flag.store(true, Ordering::Relaxed);
                return;
            }
        }
    });
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
    worker.join().unwrap();
    assert!(
        exited.load(Ordering::Relaxed),
        "last rank kept stepping after its upstream reset; the supervisor can never rebuild it"
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
/// link the way a dying neighbour does: rank 0's own socket stays open and
/// reads EOF (closing rank 0's client from the test would not exercise that).
struct Tail {
    ports: (u16, u16, u16),
    /// Completes when rank 1 has accepted rank 0's dial.
    accepted: Option<tokio::task::JoinHandle<()>>,
    fwd: tokio::task::JoinHandle<()>,
    c12: Arc<Mutex<ActivationClient>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Tail {
    /// `ports` = (forwarder, rank 1, rank 2); zeros pick free ports.
    async fn start(
        dir: &std::path::Path,
        handle: &tokio::runtime::Handle,
        ports: (u16, u16, u16),
    ) -> Tail {
        let mut s12 = ActivationServer::new("127.0.0.1", ports.2);
        s12.start().await.unwrap();
        let p2 = s12.port();
        let s12 = Arc::new(Mutex::new(s12));
        let sc = s12.clone();
        let accept = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
        let mut c12 = ActivationClient::new("127.0.0.1", p2);
        c12.connect_with_timeout(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        accept.await.unwrap();
        let c12 = Arc::new(Mutex::new(c12));

        // Rank 1 accepts whenever rank 0 dials in through the forwarder.
        let mut s01 = ActivationServer::new("127.0.0.1", ports.1);
        s01.start().await.unwrap();
        let p1 = s01.port();
        let s01 = Arc::new(Mutex::new(s01));
        let sc = s01.clone();
        let accepted = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", ports.0))
            .await
            .unwrap();
        let pf = listener.local_addr().unwrap().port();
        let fwd = tokio::spawn(async move {
            let (mut a, _) = listener.accept().await.unwrap();
            let mut b = tokio::net::TcpStream::connect(("127.0.0.1", p1))
                .await
                .unwrap();
            a.set_nodelay(true).ok();
            b.set_nodelay(true).ok();
            let _ = tokio::io::copy_bidirectional(&mut a, &mut b).await;
        });

        let load = |rank, lo, hi| {
            InklingRunner::load_staged(dir, 64, rank, 3, lo, hi, Some("eager".into()), None)
                .unwrap()
        };
        let mut e1 = PipelineEngine::new(
            load(1, 2, 3),
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
        let workers = [Box::new(e1) as Box<dyn Engine>, Box::new(e2)]
            .into_iter()
            .map(|mut e| {
                std::thread::spawn(move || {
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
                    while std::time::Instant::now() < deadline {
                        if e.step().is_err() {
                            return;
                        }
                    }
                    panic!("worker rank never exited");
                })
            })
            .collect();
        Tail {
            ports: (pf, p1, p2),
            accepted: Some(accepted),
            fwd,
            c12,
            workers,
        }
    }

    /// The neighbours die: the wire to rank 0 is cut, rank 1 exits on its
    /// upstream EOF, rank 2 on rank 1's. Returns the ports for a restart.
    async fn kill(self) -> (u16, u16, u16) {
        self.fwd.abort();
        let mut workers = self.workers.into_iter();
        let w1 = workers.next().unwrap();
        tokio::task::spawn_blocking(move || w1.join().unwrap())
            .await
            .unwrap();
        self.c12.lock().await.close().await;
        let w2 = workers.next().unwrap();
        tokio::task::spawn_blocking(move || w2.join().unwrap())
            .await
            .unwrap();
        self.ports
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

    let tail = Tail::start(&dir, &handle, (0, 0, 0)).await;
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

    // 1. The neighbours restart while rank 0 idles, and no request arrives.
    // Rank 0 must notice the dead link and dial the new rank 1 by itself:
    // until it does, rank 1 waits for that dial, unloaded, and on a real fleet
    // every rank that restarts after it waits on the one before (seen with 11
    // boxes: ranks 1-5 all "waiting for the previous rank to dial in" behind a
    // rank 0 that was "serving").
    let ports = tail.kill().await;
    let mut tail = Tail::start(&dir, &handle, ports).await;
    let dialed = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tail.accepted.take().unwrap(),
    )
    .await;
    assert!(
        dialed.is_ok(),
        "rank 0 did not re-dial the restarted rank 1 while idle (no request was made)"
    );
    let (e0, got) = round(e0, "restarted", n).await;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g.as_ref(), Ok(e), "task {i} after the neighbours restarted");
    }

    // 2. The neighbours are down.
    let ports = tail.kill().await;
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
    let tail = Tail::start(&dir, &handle, ports).await;
    tokio::time::sleep(std::time::Duration::from_millis(3200)).await;
    let (e0, got) = round(e0, "back", n).await;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g.as_ref(), Ok(e), "task {i} after the pipeline came back");
    }

    drop(e0);
    tail.kill().await;
}
