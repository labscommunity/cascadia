//! Prompts longer than one prefill frame. The multi-stream pipeline used to
//! send a whole prompt in one `StreamOpen`; the receiver rejects more than
//! `MAX_STREAM_ROWS` (256) rows, so on the 11-box fleet every prompt over 256
//! tokens made rank 1 drop the link and the whole chain rebuild (about two
//! minutes of outage per request). Long prompts now travel as `StreamFeed`
//! windows. Here the window is 4 rows, so ordinary fixture prompts split into
//! 2-4 windows, and the tokens must equal the single-stage engine's (which
//! never windows). One binary, one test: the window size is read once per
//! process from the environment.

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

/// 14, 12 (a whole number of windows), 9 and 2 tokens (one plain
/// `StreamOpen`), and a second long one that reuses a freed slot.
const PROMPTS: [&str; 5] = [
    "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14",
    "a84 a86 a10 a74 a43 a83 a85 a92 a53 a81 a33 a5",
    "a12 a60 a99 a3 a47 a101 a66 a9 a31",
    "a7 a100",
    "a5 a33 a81 a53 a92 a85 a83 a43 a74 a10 a86 a84 a2",
];
const MAX_TOKENS: [u32; 6] = [6, 8, 5, 7, 6, 5];
/// The fixture prompts plus one of ~400 tokens: at a 4-row window that is
/// ~100 windows, far more than the three steps the runner's watchdog allows
/// without a chunk (it needs the progress chunks the feed emits).
fn prompts() -> Vec<String> {
    let mut v: Vec<String> = PROMPTS.iter().map(|p| p.to_string()).collect();
    // a pseudo-random sequence of the fixture's `a<n>` words (runs of
    // consecutive words merge into few tokens; random ones do not)
    let mut x: u32 = 12345;
    let words: Vec<String> = (0..700)
        .map(|_| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            format!("a{}", (x >> 16) % 100 + 1)
        })
        .collect();
    v.push(words.join(" "));
    v
}
const MAX_SEQ: usize = 512;

fn tasks() -> Vec<GenerationTask> {
    prompts()
        .into_iter()
        .zip(MAX_TOKENS)
        .enumerate()
        .map(|(i, (p, n))| {
            let mut t = GenerationTask::new(format!("t{i}"), p);
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
        // the runner closes every stream after 3 consecutive steps that
        // return nothing: a long prompt must keep the steps non-empty
        if chunks.is_empty() {
            let n = EMPTY_RUN.fetch_add(1, Ordering::Relaxed) + 1;
            assert!(
                n < 3,
                "3 consecutive empty steps: the runner's watchdog would close the stream"
            );
        } else {
            EMPTY_RUN.store(0, Ordering::Relaxed);
        }
        for (id, c) in chunks {
            let i = ids.iter().position(|x| x == &id).expect("known task");
            assert!(c.error.is_none(), "task {id} errored: {:?}", c.error);
            if c.is_progress() {
                PROGRESS.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if c.is_final {
                done[i] = true;
            } else {
                toks[i].push(c.token_id);
            }
        }
    }
    toks
}

static EMPTY_RUN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PROGRESS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
async fn windowed_prompts_match_single_stage() {
    let Some(dir) = fixture() else { return };
    std::env::set_var("CASCADIA_STREAMS_PREFILL_WINDOW", "4");
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ids: Vec<String> = (0..prompts().len()).map(|i| format!("t{i}")).collect();
    let lens: Vec<usize> = prompts()
        .iter()
        .map(|p| tok.encode(p.as_str(), true).unwrap().get_ids().len())
        .collect();
    assert!(
        lens[5] > 300,
        "the long prompt must be hundreds of tokens: {}",
        lens[5]
    );
    assert!(
        lens.iter().filter(|&&n| n > 4).count() >= 4 && lens.iter().any(|&n| n <= 4),
        "the fixture prompts must cover windowed and plain admission: {lens:?}"
    );

    // Reference: the single-stage engine, which never windows a prompt.
    let expected = {
        let runner =
            InklingRunner::load_staged(&dir, MAX_SEQ, 0, 1, 0, 0, Some("eager".into()), None)
                .unwrap();
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

    // 3 ranks over loopback, 3 slots for 5 tasks (slots are reused), plus one
    // long task that is cancelled while its windows are still going down.
    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let r0 =
        InklingRunner::load_staged(&dir, MAX_SEQ, 0, 3, 0, 2, Some("eager".into()), None).unwrap();
    let r1 =
        InklingRunner::load_staged(&dir, MAX_SEQ, 1, 3, 2, 3, Some("eager".into()), None).unwrap();
    let r2 =
        InklingRunner::load_staged(&dir, MAX_SEQ, 2, 3, 3, 4, Some("eager".into()), None).unwrap();
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
    let died = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = [Box::new(e1) as Box<dyn Engine>, Box::new(e2)]
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
    let got = tokio::task::spawn_blocking(move || {
        // A long prompt cancelled after its first window: its slot must come
        // back on every rank, or the five tasks below run out of slots.
        let mut doomed = GenerationTask::new("doomed", PROMPTS[0]);
        doomed.max_tokens = 4;
        doomed.temperature = 0.0;
        e0.submit(doomed).unwrap();
        let first = e0.step().expect("step");
        assert!(
            first.iter().all(|(_, c)| c.is_progress()),
            "no token before the last window (progress chunks only)"
        );
        e0.cancel(&"doomed".to_string().into());
        for t in tasks() {
            e0.submit(t).unwrap();
        }
        let got = collect(&mut e0, &ids2);
        drop(e0);
        got
    })
    .await
    .unwrap();
    assert!(
        !died.load(Ordering::Relaxed),
        "a worker rank dropped out while the prompts were fed"
    );
    stop.store(true, Ordering::Relaxed);
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    for w in workers {
        let _ = w.join();
    }
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            g, e,
            "task {i} ({} prompt tokens): windowed prefill tokens differ from the single-stage path",
            lens[i]
        );
        if i < 5 {
            assert_eq!(g.len() as u32, MAX_TOKENS[i], "task {i}: token count");
        } else {
            // the word-salad prompt ends in EOS early on both paths; what matters
            // is that it was answered at all, identically
            assert!(!g.is_empty(), "task {i}: no tokens");
        }
    }
    {
        let n = PROGRESS.load(Ordering::Relaxed);
        assert!(
            n >= 50,
            "the long prompt should have produced one progress chunk per window (got {n})"
        );
    }
}
