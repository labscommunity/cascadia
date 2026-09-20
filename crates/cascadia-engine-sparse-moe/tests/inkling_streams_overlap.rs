//! The multi-stream pipeline must overlap ranks. A wrapper runner charges
//! `base + per_row · rows` per decode micro-batch (a resident rank: fixed
//! weight reads plus per-row work), so the fixture model's µs compute cannot
//! hide the wire behaviour. With one group every rank waits for the whole
//! pipeline each step; with G groups the ranks work on different groups'
//! smaller frames at once, and smaller frames are cheaper per row.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cascadia_engine::Engine;
use cascadia_engine_sparse_moe::dist::StageTransport;
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_transport::{ActivationClient, ActivationServer};
use cascadia_types::GenerationTask;
use tokio::sync::Mutex;

/// Delegates everything to the real runner, sleeping `cost` per decode
/// micro-batch (what a resident rank's expert reads would cost).
struct SlowRunner {
    inner: InklingRunner,
    /// Per-frame base cost and per-row cost.
    cost: (Duration, Duration),
    /// (decode calls, rows, slept) — printed per rank at the end.
    stats: Arc<std::sync::Mutex<(u64, u64, Duration)>>,
    /// The most rows any one decode micro-batch carried.
    max_rows: Arc<AtomicUsize>,
    /// Ranks inside a decode micro-batch right now, and the most seen at
    /// once, shared by the pipeline's ranks: overlap observed directly, not
    /// inferred from wall time.
    busy: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl StagedRunner for SlowRunner {
    fn arch_name(&self) -> &'static str {
        "inkling-slow"
    }
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
    fn max_seq(&self) -> usize {
        self.inner.max_seq()
    }
    fn eos_token_ids(&self) -> &[u32] {
        self.inner.eos_token_ids()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn embed_token(&self, token: u32) -> Vec<f32> {
        self.inner.embed_token(token)
    }
    fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, token: Option<u32>) -> Vec<f32> {
        self.inner.forward_layers(hidden, pos, token)
    }
    fn forward_layers_batch(&mut self, hidden: Vec<f32>, base: usize, rows: usize) -> Vec<f32> {
        self.inner.forward_layers_batch(hidden, base, rows)
    }
    fn supports_batched_prefill(&self) -> bool {
        true
    }
    fn head_logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.inner.head_logits(hidden)
    }
    fn head_logits_rows(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        self.inner.head_logits_rows(hidden, rows)
    }
    fn stream_capacity(&self) -> usize {
        self.inner.stream_capacity()
    }
    fn configure_streams(&mut self, n: usize) -> bool {
        self.inner.configure_streams(n)
    }
    fn open_stream(&mut self) -> Option<usize> {
        self.inner.open_stream()
    }
    fn open_stream_at(&mut self, slot: usize) -> bool {
        self.inner.open_stream_at(slot)
    }
    fn close_stream(&mut self, slot: usize) {
        self.inner.close_stream(slot)
    }
    fn stream_pos(&self, slot: usize) -> usize {
        self.inner.stream_pos(slot)
    }
    fn prefill_stream(&mut self, slot: usize, hidden: Vec<f32>, rows: usize) -> Vec<f32> {
        self.inner.prefill_stream(slot, hidden, rows)
    }
    fn decode_streams(&mut self, hidden: Vec<f32>, slots: &[usize]) -> Vec<f32> {
        let t0 = Instant::now();
        let now = self.busy.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.max_rows.fetch_max(slots.len(), Ordering::SeqCst);
        std::thread::sleep(self.cost.0 + self.cost.1 * slots.len() as u32);
        self.busy.fetch_sub(1, Ordering::SeqCst);
        let mut g = self.stats.lock().unwrap();
        g.0 += 1;
        g.1 += slots.len() as u64;
        g.2 += t0.elapsed();
        drop(g);
        self.inner.decode_streams(hidden, slots)
    }
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

/// Run `n_streams` tasks of `tokens` tokens through a 4-rank pipeline with
/// `groups` frames in flight; returns the wall time of the decode phase, the
/// tokens, the most ranks that were decoding at the same moment and the most
/// rows one decode micro-batch carried. `trickle`: the requests arrive one
/// per round instead of all before the first (what a server sees: a submit
/// waits for the engine lock, which a round holds).
async fn run(
    dir: &PathBuf,
    groups: usize,
    n_streams: usize,
    tokens: u32,
    cost: (Duration, Duration),
    trickle: bool,
) -> (Duration, Vec<Vec<i64>>, usize, usize) {
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    // 4 ranks: layers [0,1) [1,2) [2,3) [3,4)
    let (s01, c01) = link().await;
    let (s12, c12) = link().await;
    let (s23, c23) = link().await;
    let stats: Vec<Arc<std::sync::Mutex<(u64, u64, Duration)>>> = (0..4)
        .map(|_| Arc::new(std::sync::Mutex::new((0, 0, Duration::ZERO))))
        .collect();
    let busy = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let max_rows = Arc::new(AtomicUsize::new(0));
    let mk = |rank: u32, lo: u32, hi: u32| SlowRunner {
        inner: InklingRunner::load_staged(dir, 64, rank, 4, lo, hi, Some("eager".into()), None)
            .unwrap(),
        cost,
        stats: stats[rank as usize].clone(),
        busy: busy.clone(),
        peak: peak.clone(),
        max_rows: max_rows.clone(),
    };
    std::env::set_var("CASCADIA_STREAMS_INFLIGHT", groups.to_string());
    let mut e0 = PipelineEngine::new(
        mk(0, 0, 1),
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
    let mut e1 = PipelineEngine::new(
        mk(1, 1, 2),
        None,
        StageTransport {
            upstream: Some(s01),
            downstream: Some(c12.clone()),
        },
        handle.clone(),
        1,
        4,
        None,
    );
    let mut e2 = PipelineEngine::new(
        mk(2, 2, 3),
        None,
        StageTransport {
            upstream: Some(s12),
            downstream: Some(c23.clone()),
        },
        handle.clone(),
        2,
        4,
        None,
    );
    let mut e3 = PipelineEngine::new(
        mk(3, 3, 4),
        None,
        StageTransport {
            upstream: Some(s23),
            downstream: None,
        },
        handle.clone(),
        3,
        4,
        None,
    );
    for e in [&mut e0, &mut e1, &mut e2, &mut e3] {
        assert_eq!(e.enable_streams(n_streams), n_streams);
    }
    std::env::remove_var("CASCADIA_STREAMS_INFLIGHT");
    let stop = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = [Box::new(e1) as Box<dyn Engine>, Box::new(e2), Box::new(e3)]
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
    let ids: Vec<String> = (0..n_streams).map(|i| format!("s{i}")).collect();
    let ids2 = ids.clone();
    let (wall, toks) = tokio::task::spawn_blocking(move || {
        let task = |i: usize| {
            let mut t = GenerationTask::new(
                ids2[i].clone(),
                format!("a{} a{} a{} a{}", 5 + i, 33, 81 + i, 53),
            );
            t.max_tokens = tokens;
            t.temperature = 0.0;
            t
        };
        let mut submitted = 0;
        if !trickle {
            for i in 0..n_streams {
                e0.submit(task(i)).unwrap();
            }
            submitted = n_streams;
        }
        // admit everything first (prefills are round trips), then time decode
        let mut toks: Vec<Vec<i64>> = vec![Vec::new(); n_streams];
        let mut done = vec![false; n_streams];
        let mut started: Option<Instant> = None;
        let mut steps = 0;
        while done.iter().any(|d| !d) {
            steps += 1;
            assert!(steps < 2000, "did not finish");
            if submitted < n_streams {
                e0.submit(task(submitted)).unwrap();
                submitted += 1;
            }
            let chunks = e0.step().expect("step");
            for (id, c) in chunks {
                let i = ids2.iter().position(|x| x == &id).unwrap();
                assert!(c.error.is_none(), "{:?}", c.error);
                if c.is_final {
                    done[i] = true;
                } else {
                    toks[i].push(c.token_id);
                    started.get_or_insert_with(Instant::now);
                }
            }
        }
        (started.map(|s| s.elapsed()).unwrap_or_default(), toks)
    })
    .await
    .unwrap();
    stop.store(true, Ordering::Relaxed);
    c01.lock().await.close().await;
    c12.lock().await.close().await;
    c23.lock().await.close().await;
    for w in workers {
        let _ = w.join();
    }
    for (r, st) in stats.iter().enumerate() {
        let g = st.lock().unwrap();
        eprintln!(
            "   rank {r}: {} decode calls, {} rows, slept {:?} ({:.1} ms/call)",
            g.0,
            g.1,
            g.2,
            g.2.as_secs_f64() * 1e3 / g.0.max(1) as f64
        );
    }
    (
        wall,
        toks,
        peak.load(Ordering::SeqCst),
        max_rows.load(Ordering::SeqCst),
    )
}

/// Diagnostic sweep, run on demand (`--ignored`). Not part of the suite: `run`
/// selects the in-flight depth through a process-wide env var, so a sweep
/// running next to the asserted test below could hand it the wrong depth.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn probe_timings() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    if !dir.join("tokenizer.json").exists() {
        return;
    }
    for (g, streams, base_ms, row_ms) in [
        (1usize, 6usize, 10u64, 5u64),
        (2, 6, 10, 5),
        (4, 6, 10, 5),
        (1, 6, 0, 0),
        (4, 6, 0, 0),
    ] {
        let (w, _, peak, _) = run(
            &dir,
            g,
            streams,
            12,
            (
                Duration::from_millis(base_ms),
                Duration::from_millis(row_ms),
            ),
            false,
        )
        .await;
        eprintln!(
            "probe groups={g} streams={streams} cost={base_ms}+{row_ms}/row ms: decode wall {w:?}, peak busy ranks {peak}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn groups_in_flight_overlap_the_ranks() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    if !dir.join("tokenizer.json").exists() {
        eprintln!("fixture missing; skipping");
        return;
    }
    let (tokens, streams) = (12u32, 6usize);
    let cost = (Duration::from_millis(10), Duration::from_millis(5));
    // 1 group: one 6-row frame per step, 40 ms per rank, 4 ranks in series.
    let (w1, t1, peak1, _) = run(&dir, 1, streams, tokens, cost, false).await;
    // 4 groups (one per rank): 1-2 row frames, ~15-20 ms per rank, all ranks busy.
    let (w4, t4, peak4, _) = run(&dir, 4, streams, tokens, cost, false).await;
    assert_eq!(t1, t4, "tokens must not depend on the in-flight depth");
    // Wall time is reported, not asserted: about 1.6-1.9x on an idle machine,
    // but sleep precision and a loaded CI runner move it. The overlap itself
    // is observed directly.
    eprintln!(
        "decode wall: 1 group {w1:?}, 4 groups {w4:?} ({:.2}x); peak busy ranks {peak1} vs {peak4}",
        w1.as_secs_f64() / w4.as_secs_f64()
    );
    assert_eq!(peak1, 1, "one group in flight: the ranks must take turns");
    assert!(
        peak4 >= 2,
        "no overlap: with 4 groups in flight at most {peak4} rank decoded at a time"
    );
    // Requests that arrive one per round (a server's submits wait for the
    // engine lock a round holds) must still spread over the groups. Every
    // round starts at group 0, so admitting into "the group whose turn it is"
    // put all four streams into group 0: one 4-row frame in flight, three
    // ranks idle. (Here in the same test because `run` picks the depth
    // through a process-wide env var.)
    let (_, tt, _, max_rows) = run(&dir, 4, 4, tokens, cost, true).await;
    assert_eq!(
        max_rows, 1,
        "4 streams over 4 groups must be one row per frame, saw a {max_rows}-row frame"
    );
    assert_eq!(tt[..], t4[..4], "tokens must not depend on the grouping");
}
