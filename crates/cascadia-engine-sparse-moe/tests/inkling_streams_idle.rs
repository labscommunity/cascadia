//! A multi-stream pipeline that idles between requests for longer than the
//! transport's frame idle ceiling must keep its links. The workers wait for
//! the next frame with a readiness peek that has no ceiling; a plain frame
//! receive would fail after the ceiling and tear the chain down (each link
//! is accepted once, so nothing could re-dial it). One binary, one test: the
//! ceiling and the recv timeout are process-wide settings.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

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

const PROMPTS: [&str; 3] = [
    "a5 a33 a81 a53 a92 a85 a83 a43 a74 a10 a86 a84",
    "a12 a60 a99 a3 a47 a101 a66 a9",
    "a7 a7 a7 a100 a2",
];

/// Submit the prompts under fresh task ids and collect their tokens; an
/// error chunk fails the test.
fn round(engine: &mut dyn Engine, tag: &str) -> Vec<Vec<i64>> {
    let ids: Vec<String> = (0..PROMPTS.len()).map(|i| format!("{tag}-{i}")).collect();
    for (id, p) in ids.iter().zip(PROMPTS) {
        let mut t = GenerationTask::new(id.clone(), p);
        t.max_tokens = 6;
        t.temperature = 0.0;
        engine.submit(t).unwrap();
    }
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
        .connect_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    accept.await.unwrap();
    (server, Arc::new(Mutex::new(client)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stream_pipeline_survives_idle_past_the_frame_ceiling() {
    let Some(dir) = fixture() else { return };
    // The ceiling is floored at the recv timeout, so lower that too. Rank 0
    // keeps its prefill reply budget 30 s under the ceiling, so a ceiling
    // under 31 s would leave it no budget at all.
    cascadia_transport::set_activation_timeout_secs(1);
    cascadia_transport::set_frame_idle_ceiling_secs(31);
    assert_eq!(
        cascadia_transport::frame_idle_ceiling(),
        Some(Duration::from_secs(31))
    );
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");

    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let load = |rank, lo, hi| {
        InklingRunner::load_staged(&dir, 64, rank, 3, lo, hi, Some("eager".into()), None).unwrap()
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
    for e in [&mut e0, &mut e1, &mut e2] {
        assert_eq!(e.enable_streams(3), 3);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = [Box::new(e1) as Box<dyn Engine>, Box::new(e2)]
        .into_iter()
        .map(|mut e| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if e.step().is_err() {
                        break;
                    }
                }
            })
        })
        .collect();

    let (mut e0, first) = tokio::task::spawn_blocking(move || {
        let got = round(&mut e0, "first");
        (e0, got)
    })
    .await
    .unwrap();
    assert!(first.iter().all(|t| !t.is_empty()));

    // Idle past the ceiling with no frame on any link.
    tokio::time::sleep(Duration::from_secs(34)).await;
    assert!(
        workers.iter().all(|w| !w.is_finished()),
        "a worker rank exited while the pipeline idled"
    );

    let second = tokio::task::spawn_blocking(move || {
        let got = round(&mut e0, "second");
        drop(e0);
        got
    })
    .await
    .unwrap();
    assert_eq!(second, first, "tokens after the idle gap differ");

    stop.store(true, Ordering::Relaxed);
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    for w in workers {
        let _ = w.join();
    }
}
