//! Direct reply link (`CASCADIA_STREAMS_RETURN_PORT`): the last rank answers
//! rank 0 on its own connection instead of through every rank in between.
//! Rank 0 reads replies only from that link once it is configured, so the test
//! finishing at all means the replies came that way; the tokens must be the
//! single-stage engine's. One binary, one test (process-wide environment).

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
async fn replies_come_back_on_the_direct_link() {
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

    // A free port for the return link.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    std::env::set_var("CASCADIA_STREAMS_RETURN_PORT", port.to_string());
    std::env::set_var("CASCADIA_STREAMS_RETURN_HOST", "127.0.0.1");

    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let (s23, c23) = link().await;
    let load = |rank: u32, lo: u32, hi: u32| {
        InklingRunner::load_staged(&dir, 64, rank, 4, lo, hi, Some("eager".into()), None).unwrap()
    };
    let mut e0 = PipelineEngine::new(
        load(0, 0, 1),
        Some(tok),
        StageTransport {
            upstream: None,
            downstream: Some(c01.clone()),
        },
        handle.clone(),
        0,
        4,
        None,
    );
    let mut workers_e: Vec<Box<dyn Engine>> = Vec::new();
    for (rank, up, down) in [
        (1u32, s01, Some(c12.clone())),
        (2, s12, Some(c23.clone())),
        (3, s23, None),
    ] {
        let mut e = PipelineEngine::new(
            load(rank, rank, rank + 1),
            None,
            StageTransport {
                upstream: Some(up),
                downstream: down,
            },
            handle.clone(),
            rank,
            4,
            None,
        );
        // enable_streams blocks on the runtime (the last rank binds its listener)
        let e = tokio::task::spawn_blocking(move || {
            assert_eq!(e.enable_streams(3), 3);
            e
        })
        .await
        .unwrap();
        workers_e.push(Box::new(e));
    }
    let mut e0 = tokio::task::spawn_blocking(move || {
        assert_eq!(e0.enable_streams(3), 3);
        e0
    })
    .await
    .unwrap();
    std::env::remove_var("CASCADIA_STREAMS_RETURN_PORT");
    std::env::remove_var("CASCADIA_STREAMS_RETURN_HOST");

    let stop = Arc::new(AtomicBool::new(false));
    let died = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = workers_e
        .into_iter()
        .map(|mut e| {
            let (stop, died) = (stop.clone(), died.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if e.step().is_err() {
                        died.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            })
        })
        .collect();

    let ids2 = ids.clone();
    let (got, direct) = tokio::task::spawn_blocking(move || {
        for t in tasks() {
            e0.submit(t).unwrap();
        }
        let got = collect(&mut e0, &ids2);
        let direct = e0.return_link_active();
        drop(e0);
        (got, direct)
    })
    .await
    .unwrap();
    assert!(!died.load(Ordering::Relaxed), "a worker rank dropped out");
    assert!(direct, "rank 0 never connected the direct reply link");
    stop.store(true, Ordering::Relaxed);
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    c23.lock().await.close().await;
    for w in workers {
        let _ = w.join();
    }
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g, e, "task {i}: tokens differ with the direct reply link");
    }
}
