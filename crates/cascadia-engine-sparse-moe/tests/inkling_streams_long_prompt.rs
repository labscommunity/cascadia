//! Prompts longer than one prefill frame. The multi-stream pipeline used to
//! send a whole prompt in one `StreamOpen`; the receiver rejects more than
//! `MAX_STREAM_ROWS` (256) rows, so on the 11-box fleet every prompt over 256
//! tokens made rank 1 drop the link and the whole chain rebuild (about two
//! minutes of outage per request). Long prompts now travel as `StreamFeed`
//! windows. Here the window is 4 rows, so ordinary fixture prompts split into
//! 2-4 windows, and the tokens must equal the single-stage engine's (which
//! never windows). The window size is read once per process from the
//! environment, so every test in this binary sets the same value.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_engine_sparse_moe::dist::StageTransport;
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_runner::Runner;
use cascadia_transport::{ActivationClient, ActivationServer};
use cascadia_types::{GenerationTask, PeerLayout, ShardSpec};
use futures::StreamExt;
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
const MAX_TOKENS: [u32; 5] = [6, 8, 5, 7, 6];

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

/// The tokens of the single-stage engine (which never windows a prompt).
async fn reference(
    dir: &std::path::Path,
    tok: &tokenizers::Tokenizer,
    tasks: Vec<GenerationTask>,
) -> Vec<Vec<i64>> {
    let runner =
        InklingRunner::load_staged(dir, 64, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
    let mut e = PipelineEngine::new(
        runner,
        Some(tok.clone()),
        StageTransport::default(),
        tokio::runtime::Handle::current(),
        0,
        1,
        None,
    );
    let ids: Vec<String> = tasks.iter().map(|t| t.task_id.to_string()).collect();
    tokio::task::spawn_blocking(move || {
        for t in tasks {
            e.submit(t).unwrap();
        }
        collect(&mut e, &ids)
    })
    .await
    .unwrap()
}

/// Ranks 1 and 2 of a 3-rank loopback pipeline, stepped on their own threads.
struct Workers {
    stop: Arc<AtomicBool>,
    died: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
    links: [Arc<Mutex<ActivationClient>>; 2],
}

impl Workers {
    /// Stop the workers; `true` if one of them dropped out before.
    async fn shutdown(self) -> bool {
        let died = self.died.load(Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        for c in &self.links {
            c.lock().await.close().await;
        }
        for w in self.threads {
            let _ = w.join();
        }
        died
    }
}

/// A 3-rank loopback pipeline with 3 stream slots: rank 0, and the workers.
async fn pipeline(
    dir: &std::path::Path,
    tok: tokenizers::Tokenizer,
) -> (PipelineEngine<InklingRunner>, Workers) {
    let handle = tokio::runtime::Handle::current();
    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let r0 = InklingRunner::load_staged(dir, 64, 0, 3, 0, 2, Some("eager".into()), None).unwrap();
    let r1 = InklingRunner::load_staged(dir, 64, 1, 3, 2, 3, Some("eager".into()), None).unwrap();
    let r2 = InklingRunner::load_staged(dir, 64, 2, 3, 3, 4, Some("eager".into()), None).unwrap();
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
    let threads = [Box::new(e1) as Box<dyn Engine>, Box::new(e2)]
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
    let workers = Workers {
        stop,
        died,
        threads,
        links: [c01, c12],
    };
    (e0, workers)
}

/// One step of rank 0 while only the task "warm" may emit: push its tokens
/// to `toks`; `true` once it is done.
fn step_warm(e0: &mut PipelineEngine<InklingRunner>, toks: &mut Vec<i64>) -> bool {
    let mut done = false;
    for (id, c) in e0.step().expect("step") {
        assert_eq!(id, "warm", "only the short task emits");
        assert!(c.error.is_none(), "warm errored: {:?}", c.error);
        if c.is_final {
            done = true;
        } else {
            toks.push(c.token_id);
        }
    }
    done
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn windowed_prompts_match_single_stage() {
    let Some(dir) = fixture() else { return };
    std::env::set_var("CASCADIA_STREAMS_PREFILL_WINDOW", "4");
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ids: Vec<String> = (0..PROMPTS.len()).map(|i| format!("t{i}")).collect();
    let lens: Vec<usize> = PROMPTS
        .iter()
        .map(|p| tok.encode(*p, true).unwrap().get_ids().len())
        .collect();
    assert!(
        lens.iter().filter(|&&n| n > 4).count() >= 4 && lens.iter().any(|&n| n <= 4),
        "the fixture prompts must cover windowed and plain admission: {lens:?}"
    );

    let expected = reference(&dir, &tok, tasks()).await;
    assert!(expected.iter().all(|t| !t.is_empty()));

    // 3 ranks over loopback, 3 slots for 5 tasks (slots are reused), plus one
    // long task that is cancelled while its windows are still going down.
    let (mut e0, workers) = pipeline(&dir, tok).await;
    let ids2 = ids.clone();
    let warm_ref = expected[3][..4].to_vec();
    let got = tokio::task::spawn_blocking(move || {
        // A 40-token prompt cancelled while its windows go down. A short task
        // decodes next to it, so a step sends only some of its 10 windows.
        let doomed_id = "doomed".to_string();
        let prompt: Vec<String> = (1..=40).map(|i| format!("a{i}")).collect();
        let mut doomed = GenerationTask::new(doomed_id.clone(), prompt.join(" "));
        doomed.max_tokens = 4;
        doomed.temperature = 0.0;
        let mut warm = tasks().remove(3);
        warm.task_id = "warm".into();
        warm.max_tokens = 4;
        e0.submit(warm).unwrap();
        e0.submit(doomed).unwrap();
        let mut warm_toks = Vec::new();
        step_warm(&mut e0, &mut warm_toks);
        assert!(e0.has_stream(&doomed_id), "the long prompt is admitted");
        e0.cancel(&doomed_id);
        // Rank 0 admits only while it holds fewer streams than slots, so a
        // stale entry would only slow the tasks below: check it directly.
        // The ranks behind must free the slot too, or the tasks below that
        // reuse it fail.
        let mut warm_done = step_warm(&mut e0, &mut warm_toks);
        assert!(
            !e0.has_stream(&doomed_id),
            "the cancelled prompt still holds a stream on rank 0"
        );
        let mut steps = 0;
        while !warm_done {
            steps += 1;
            assert!(steps < 100, "the short task did not finish");
            warm_done = step_warm(&mut e0, &mut warm_toks);
        }
        assert_eq!(warm_toks, warm_ref, "the short task's tokens");
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
        !workers.shutdown().await,
        "a worker rank dropped out while the prompts were fed"
    );
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            g, e,
            "task {i} ({} prompt tokens): windowed prefill tokens differ from the single-stage path",
            lens[i]
        );
        assert_eq!(g.len() as u32, MAX_TOKENS[i], "task {i}: token count");
    }
}

struct PrebuiltBuilder {
    engine: Option<Box<dyn Engine>>,
}

#[async_trait]
impl Builder for PrebuiltBuilder {
    async fn connect(&mut self, _peers: PeerLayout) -> EngineResult<()> {
        Ok(())
    }
    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        Ok(Box::pin(futures::stream::iter(Vec::new())))
    }
    fn build(mut self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        self.engine.take().ok_or(EngineError::NotLoaded)
    }
}

/// A prompt of more than three rounds of windows, alone on an idle pipeline,
/// through the runner's chunk stream. The runner fails a task after three
/// steps with no chunk; a step used to send only one window per group.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lone_long_prompt_finishes_through_the_runner() {
    let Some(dir) = fixture() else { return };
    std::env::set_var("CASCADIA_STREAMS_PREFILL_WINDOW", "4");
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let prompt: Vec<String> = (1..=44).map(|i| format!("a{i}")).collect();
    let prompt = prompt.join(" ");
    let task = || {
        let mut t = GenerationTask::new("long", prompt.as_str());
        t.max_tokens = 6;
        t.temperature = 0.0;
        t
    };
    // 3 groups, 4-row windows: more than 3 rounds of windows, inside max_seq.
    let n = tok.encode(prompt.as_str(), true).unwrap().get_ids().len();
    assert!(n > 3 * 3 * 4 && n + 6 <= 64, "prompt tokens: {n}");

    let expected = reference(&dir, &tok, vec![task()]).await.remove(0);
    assert_eq!(expected.len(), 6);

    let (e0, workers) = pipeline(&dir, tok).await;
    let runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
        engine: Some(Box::new(e0)),
    })));
    let spec = ShardSpec {
        model_id: "inkling".into(),
        layer_start: 0,
        layer_end: 2,
        total_layers: 4,
        device: "CPU".into(),
        is_first_stage: true,
        is_last_stage: false,
        tp_size: 1,
        tp_rank: 0,
    };
    runner.start(PeerLayout::default(), spec).await.unwrap();
    let mut stream = runner.generate_async(task()).await.unwrap();
    let mut got = Vec::new();
    while let Some(c) = stream.next().await {
        assert!(c.error.is_none(), "the long prompt failed: {:?}", c.error);
        if !c.is_final {
            got.push(c.token_id);
        }
    }
    drop(stream);
    runner.close();
    assert!(
        !workers.shutdown().await,
        "a worker rank dropped out while the prompt was fed"
    );
    assert_eq!(got, expected, "tokens differ from the single-stage path");
}
