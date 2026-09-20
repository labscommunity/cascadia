//! The speculation path with a drafter MODEL (`CASCADIA_STREAMS_SPEC_LM`, see
//! `lm_draft.rs`): the guesses come as text from an HTTP server, are cut into
//! the target's tokens, and ride behind the real frame. The server here knows
//! what the model will say and lies about every fourth word, so right guesses,
//! wrong guesses and restarts of the drafter all happen. Whatever it says, the
//! tokens must be exactly those of plain decoding. One binary, one test: the
//! switches are read from the process environment.

use std::io::{BufRead, BufReader, Read, Write};
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
    let mut empty = 0;
    while done.iter().any(|d| !d) {
        steps += 1;
        assert!(steps < 200_000, "engine did not finish the tasks");
        let chunks = engine.step().expect("step");
        // The serving loop (cascadia-runner) reads three empty steps in a row
        // as a wedged engine and fails the request: a waiting round must not
        // surface as an empty step.
        empty = if chunks.is_empty() { empty + 1 } else { 0 };
        assert!(
            empty < 3,
            "three empty steps in a row: the runner would fail the request"
        );
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

/// A stand-in for llama-server. `known` = (prompt text, the text the model will write).
fn fake_drafter(known: Vec<(String, String)>) -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = l.local_addr().expect("addr").to_string();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut c) = c else { continue };
            let mut r = BufReader::new(c.try_clone().expect("clone"));
            let mut len = 0usize;
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; len];
            if r.read_exact(&mut body).is_err() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
            let prompt = v["prompt"].as_str().unwrap_or("").to_string();
            // "plain" template: the user text, a blank line, the output so far.
            let (user, sofar) = prompt.split_once("\n\n").unwrap_or((prompt.as_str(), ""));
            let cont: String = known
                .iter()
                .find(|(p, _)| p.trim() == user.trim())
                .and_then(|(_, full)| full.strip_prefix(sofar))
                .map(|rest| {
                    rest.split_inclusive(' ')
                        .enumerate()
                        .map(|(i, w)| {
                            if i % 4 == 3 {
                                "a99 ".to_string()
                            } else {
                                w.to_string()
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
            // A real drafter needs tens of milliseconds to start over after a
            // wrong guess: rank 0 must find the pipeline with room and no guess.
            std::thread::sleep(std::time::Duration::from_millis(40));
            let words: Vec<&str> = cont.split_inclusive(' ').take(12).collect();
            let n = words.len().max(1);
            for i in 0..n {
                let w = words.get(i).copied().unwrap_or("");
                let ev = format!(
                    "data: {}\n\n",
                    serde_json::json!({"content": w, "stop": i + 1 == n})
                );
                if c.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes())
                    .is_err()
                {
                    break;
                }
            }
            let _ = c.write_all(b"0\r\n\r\n");
        }
    });
    addr
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drafter_model_never_changes_the_tokens() {
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
    let known: Vec<(String, String)> = (0..PROMPTS.len())
        .map(|i| {
            let p: Vec<u32> = tok.encode(PROMPTS[i], false).unwrap().get_ids().to_vec();
            let out: Vec<u32> = expected[i].iter().map(|&t| t as u32).collect();
            (
                tok.decode(&p, true).unwrap(),
                tok.decode(&out, true).unwrap(),
            )
        })
        .collect();
    eprintln!("the model will write: {:?}", known[0].1);
    let addr = fake_drafter(known);

    std::env::set_var("CASCADIA_STREAMS_SPEC", "1");
    std::env::set_var("CASCADIA_STREAMS_SPEC_LM", format!("http://{addr}"));
    std::env::set_var("CASCADIA_STREAMS_SPEC_LM_TEMPLATE", "plain");
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
    std::env::remove_var("CASCADIA_STREAMS_SPEC_LM");
    std::env::remove_var("CASCADIA_STREAMS_SPEC_LM_TEMPLATE");

    let stop = Arc::new(AtomicBool::new(false));
    let died = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = workers_e
        .into_iter()
        .map(|mut e| {
            let (stop, died) = (stop.clone(), died.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    // Real ranks take tens of milliseconds per frame: replies
                    // must be slow enough here for rank 0 to find none waiting.
                    std::thread::sleep(std::time::Duration::from_millis(8));
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
        for i in 0..PROMPTS.len() {
            e0.submit(task(i)).unwrap();
            all.extend(collect(&mut e0, &ids2[i..=i]));
        }
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
        stats.1 > 0,
        "the drafter model was never right: its guesses did not reach the pipeline"
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
