//! The single-stage multi-stream scheduler (`CASCADIA_STREAMS=N` on one box,
//! `PipelineEngine` with `total == 1` and streams enabled): the engine-level
//! path that admits, batches, samples per stream and retires. The wire test
//! uses the one-task path as its reference and the runner tests exercise the
//! layers directly, so this is the only test of the scheduler itself.
//!
//! 1. Greedy: tasks decoded together (admitted while others decode, slots
//!    reused after a stream finishes) produce exactly the one-task path's
//!    tokens.
//! 2. Sampled: a seeded temperature > 0 task produces the same tokens whether
//!    it decodes alone through the one-task path or in a batch of streams —
//!    each stream carries its own seeded rng — and two batched runs agree.

use std::path::PathBuf;

use cascadia_engine::Engine;
use cascadia_engine_sparse_moe::dist::StageTransport;
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_types::GenerationTask;

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

fn greedy_tasks() -> Vec<GenerationTask> {
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

fn sampled_task(id: &str, prompt: &str, seed: u64) -> GenerationTask {
    let mut t = GenerationTask::new(id.to_string(), prompt);
    t.max_tokens = 10;
    t.temperature = 0.9;
    t.sampling.top_p = 1.0;
    t.sampling.top_k = 0;
    t.sampling.seed = Some(seed);
    t
}

/// Drive `engine` on the calling thread until every task in `ids` has
/// produced its final marker; returns each task's token ids.
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

fn single_stage(dir: &PathBuf, handle: tokio::runtime::Handle) -> PipelineEngine<InklingRunner> {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let runner =
        InklingRunner::load_staged(dir, 64, 0, 1, 0, 0, Some("eager".into()), None).unwrap();
    PipelineEngine::new(
        runner,
        Some(tok),
        StageTransport::default(),
        handle,
        0,
        1,
        None,
    )
}

/// Run `tasks` through a fresh single-stage engine; `streams` = Some(n)
/// enables the multi-stream scheduler with n slots, None = the one-task path.
async fn run(dir: &PathBuf, tasks: Vec<GenerationTask>, streams: Option<usize>) -> Vec<Vec<i64>> {
    let handle = tokio::runtime::Handle::current();
    let mut e = single_stage(dir, handle);
    if let Some(n) = streams {
        assert_eq!(e.enable_streams(n), n, "stream slots");
    }
    let ids: Vec<String> = tasks.iter().map(|t| t.task_id.clone()).collect();
    tokio::task::spawn_blocking(move || {
        for t in tasks {
            e.submit(t).unwrap();
        }
        collect(&mut e, &ids)
    })
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_stage_streams_match_one_task_path() {
    let Some(dir) = fixture() else { return };
    let expected = run(&dir, greedy_tasks(), None).await;
    assert!(expected.iter().all(|t| !t.is_empty()));
    // 3 slots for 5 tasks: two are admitted only once a slot frees up, so
    // admission-while-decoding and slot reuse are both exercised.
    let got = run(&dir, greedy_tasks(), Some(3)).await;
    for i in 0..PROMPTS.len() {
        assert_eq!(
            got[i], expected[i],
            "task {i}: multi-stream single-stage tokens differ from the one-task path"
        );
        assert_eq!(got[i].len() as u32, MAX_TOKENS[i], "task {i}: token count");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_stage_streams_keep_per_stream_seeded_sampling() {
    let Some(dir) = fixture() else { return };
    let seeded = || sampled_task("s", PROMPTS[0], 7);
    // Alone, through the one-task path.
    let alone = run(&dir, vec![seeded()], None).await;
    assert_eq!(alone[0].len(), 10);
    // In a batch: the seeded stream next to greedy neighbours (admitted
    // before and after it), twice.
    let batch = || {
        let mut v = greedy_tasks();
        v.insert(2, seeded());
        v
    };
    let a = run(&dir, batch(), Some(3)).await;
    let b = run(&dir, batch(), Some(3)).await;
    assert_eq!(a[2], b[2], "seeded stream: two batched runs disagree");
    assert_eq!(
        a[2], alone[0],
        "seeded stream: batched tokens differ from the same task sampled alone"
    );
    // The greedy neighbours are untouched by the sampled stream.
    let greedy = run(&dir, greedy_tasks(), None).await;
    for (i, j) in [(0usize, 0usize), (1, 1), (3, 2), (4, 3), (5, 4)] {
        assert_eq!(a[i], greedy[j], "greedy task {j} next to a sampled stream");
    }
}
