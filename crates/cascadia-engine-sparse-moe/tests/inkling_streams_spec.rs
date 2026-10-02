//! Pipelined speculation for a lone stream (`CASCADIA_STREAMS_SPEC=1`): guessed
//! tokens ride behind the real one so several ranks work on one stream at
//! once; a wrong guess is rolled back on every rank. Whatever the guesses do,
//! the tokens must be exactly those of plain decoding. One binary, one test:
//! the switch is read from the process environment.

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

/// Repetitive prompts make the n-gram drafter guess (right where the model
/// repeats itself, wrong where it does not); the last two arrive together, so
/// the second one interrupts the first one's speculation.
const PROMPTS: [&str; 5] = [
    "a1 a2 a3 a1 a2 a3 a1 a2 a3 a1 a2",
    "a7 a7 a7 a7 a7 a7 a7 a7",
    "a84 a86 a10 a74 a84 a86 a10 a74 a84 a86",
    "a5 a33 a81 a5 a33 a81 a5 a33",
    "a12 a60 a12 a60 a12 a60 a12",
];
const MAX_TOKENS: [u32; 5] = [24, 24, 20, 24, 16];

fn task(i: usize) -> GenerationTask {
    let mut t = GenerationTask::new(format!("t{i}"), PROMPTS[i]);
    t.max_tokens = MAX_TOKENS[i];
    t.temperature = 0.0;
    t
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
async fn speculation_never_changes_the_tokens() {
    let Some(dir) = fixture() else { return };
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ids: Vec<String> = (0..PROMPTS.len()).map(|i| format!("t{i}")).collect();

    // Reference: the single-stage engine, one task after another.
    let expected = {
        let runner =
            InklingRunner::load_staged(&dir, 96, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
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
            let mut all = Vec::new();
            for i in 0..PROMPTS.len() {
                e.submit(task(i)).unwrap();
                all.extend(collect(&mut e, &ids2[i..=i]));
            }
            all
        })
        .await
        .unwrap()
    };

    std::env::set_var("CASCADIA_STREAMS_SPEC", "1");
    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let (s23, c23) = link().await;
    let load = |rank: u32, lo: u32, hi: u32| {
        InklingRunner::load_staged(&dir, 96, rank, 4, lo, hi, Some("eager".into()), None).unwrap()
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
        assert_eq!(e.enable_streams(3), 3);
        workers_e.push(Box::new(e));
    }
    assert_eq!(e0.enable_streams(3), 3);
    std::env::remove_var("CASCADIA_STREAMS_SPEC");

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
    let (got, stats) = tokio::task::spawn_blocking(move || {
        let mut all = Vec::new();
        // Lone streams, one after another: each speculates from its first token.
        for i in 0..3 {
            e0.submit(task(i)).unwrap();
            all.extend(collect(&mut e0, &ids2[i..=i]));
        }
        // A second request arrives while the first is speculating: the guesses
        // in flight are read out, then both streams decode side by side.
        e0.submit(task(3)).unwrap();
        for _ in 0..4 {
            let chunks = e0.step().expect("step");
            assert!(chunks.iter().all(|(_, c)| c.error.is_none()));
            all_tokens(&chunks, &ids2, 3);
        }
        e0.submit(task(4)).unwrap();
        let mut both = collect(&mut e0, &ids2[3..=4]);
        // `collect` saw task 3 only from here on: put its first tokens back.
        let mut head = EARLY.lock().unwrap().clone();
        head.append(&mut both[0]);
        both[0] = head;
        all.extend(both);
        let stats = e0.speculation_stats();
        drop(e0);
        (all, stats)
    })
    .await
    .unwrap();
    assert!(
        !died.load(Ordering::Relaxed),
        "a worker rank dropped out during speculation"
    );
    stop.store(true, Ordering::Relaxed);
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    c23.lock().await.close().await;
    for w in workers {
        let _ = w.join();
    }
    eprintln!(
        "guesses sent {}, right {}, wrong {}",
        stats.0, stats.1, stats.2
    );
    assert!(
        stats.0 > 0,
        "the drafter never guessed: the test exercised nothing"
    );
    assert!(
        stats.2 > 0,
        "no wrong guess: the rollback path was not exercised"
    );
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(g, e, "task {i}: speculation changed the tokens");
        assert_eq!(g.len() as u32, MAX_TOKENS[i], "task {i}: token count");
    }
}

static EARLY: std::sync::Mutex<Vec<i64>> = std::sync::Mutex::new(Vec::new());

/// Remember task `i`'s tokens from steps driven outside `collect`.
fn all_tokens(
    chunks: &[(cascadia_types::TaskId, cascadia_types::Chunk)],
    ids: &[String],
    i: usize,
) {
    for (id, c) in chunks {
        if id == &ids[i] && !c.is_final {
            EARLY.lock().unwrap().push(c.token_id);
        }
    }
}
