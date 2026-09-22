//! Admission refuses requests until an idle handshake traverses every worker.
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
async fn requests_are_refused_until_every_rank_is_processing_frames() {
    let dir = fixture().expect("readiness test requires the committed Inkling fixture");
    std::env::set_var("CASCADIA_STREAMS_READY_GATE", "1");
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
    let last_may_run = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = [
        (Box::new(e1) as Box<dyn Engine>, stop.clone()),
        (Box::new(e2) as Box<dyn Engine>, stop.clone()),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (mut e, stop))| {
        let last_may_run = last_may_run.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if i == 1 && !last_may_run.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    continue;
                }
                if e.step().is_err() {
                    break;
                }
            }
        })
    })
    .collect();

    let ids2 = ids.clone();
    let got = tokio::task::spawn_blocking(move || {
        // Both TCP connections exist; the final worker has not started its
        // receive loop. A TCP-only readiness test would incorrectly admit.
        for _ in 0..20 {
            assert!(matches!(
                e0.submit(tasks().remove(0)),
                Err(cascadia_engine::EngineError::NotConnected)
            ));
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        last_may_run.store(true, Ordering::SeqCst);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match e0.submit(tasks().remove(0)) {
                Ok(()) => break,
                Err(cascadia_engine::EngineError::NotConnected) => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "readiness did not recover without an engine step"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => panic!("unexpected admission error: {e}"),
            }
        }
        for t in tasks().into_iter().skip(1) {
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
