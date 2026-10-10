//! A local failure on rank 0 of a multi-stream pipeline (a panic inside its
//! own `decode_streams`) aborts the streams but must leave the link to rank 1
//! usable: the replies that the downstream still owes for the frames of the
//! other groups are read and dropped, so the next request does not read a
//! stale reply and rank 0 does not re-dial a healthy rank 1 (whose listener
//! accepts once, so a re-dial tears the whole chain down).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cascadia_engine::Engine;
use cascadia_engine_sparse_moe::dist::StageTransport;
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_transport::{ActivationClient, ActivationServer};
use cascadia_types::GenerationTask;
use tokio::sync::Mutex;

/// Delegates everything to the real runner. `decode_streams` panics on the
/// call that brings `fuse` from 1 to 0; a `fuse` of 0 never panics.
struct PanicRunner {
    inner: InklingRunner,
    fuse: Arc<AtomicUsize>,
}

impl StagedRunner for PanicRunner {
    fn arch_name(&self) -> &'static str {
        "inkling-panic"
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
    fn stream_pos(&self, slot: usize) -> Option<usize> {
        self.inner.stream_pos(slot)
    }
    fn prefill_stream(&mut self, slot: usize, hidden: Vec<f32>, rows: usize) -> Vec<f32> {
        self.inner.prefill_stream(slot, hidden, rows)
    }
    fn decode_streams(&mut self, hidden: Vec<f32>, slots: &[usize]) -> Vec<f32> {
        let left = self.fuse.load(Ordering::SeqCst);
        if left > 0 {
            self.fuse.store(left - 1, Ordering::SeqCst);
            assert!(left != 1, "injected decode failure");
        }
        self.inner.decode_streams(hidden, slots)
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

/// Submit the first `n` prompts under fresh task ids and step until each
/// ends: its tokens, or the error of its error chunk. A chunk after the end
/// of a task fails the test.
fn run_round(e0: &mut dyn Engine, tag: &str, n: usize) -> Vec<Result<Vec<i64>, String>> {
    let ids: Vec<String> = (0..n).map(|i| format!("{tag}-{i}")).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut t = GenerationTask::new(id.clone(), PROMPTS[i]);
        t.max_tokens = MAX_TOKENS[i];
        t.temperature = 0.0;
        e0.submit(t).unwrap();
    }
    let mut res: Vec<Result<Vec<i64>, String>> = ids.iter().map(|_| Ok(Vec::new())).collect();
    let mut done = vec![false; n];
    let mut steps = 0;
    while done.iter().any(|d| !d) {
        steps += 1;
        assert!(steps < 500, "engine did not finish the tasks");
        let Ok(chunks) = e0.step() else { continue };
        for (id, c) in chunks {
            let i = ids.iter().position(|x| x == &id).expect("known task");
            assert!(!done[i], "task {id} got a chunk after its end: {c:?}");
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rank0_decode_panic_keeps_the_link() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    if !dir.join("tokenizer.json").exists() {
        eprintln!("inkling_export fixture (with tokenizer.json) missing; skipping");
        return;
    }
    let handle = tokio::runtime::Handle::current();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let load = |rank, lo, hi| {
        InklingRunner::load_staged(&dir, 64, rank, 3, lo, hi, Some("eager".into()), None).unwrap()
    };

    // Rank 1 -> rank 2.
    let mut s12 = ActivationServer::new("127.0.0.1", 0);
    s12.start().await.unwrap();
    let p2 = s12.port();
    let s12 = Arc::new(Mutex::new(s12));
    let sc = s12.clone();
    let accept = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let mut c12 = ActivationClient::new("127.0.0.1", p2);
    c12.connect_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    accept.await.unwrap();
    let c12 = Arc::new(Mutex::new(c12));

    // Rank 0 -> rank 1 through a forwarder that counts rank 0's dials. Rank 1
    // accepts once; a second dial is counted and dropped.
    let mut s01 = ActivationServer::new("127.0.0.1", 0);
    s01.start().await.unwrap();
    let p1 = s01.port();
    let s01 = Arc::new(Mutex::new(s01));
    let sc = s01.clone();
    let accepted = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let pf = listener.local_addr().unwrap().port();
    let dials = Arc::new(AtomicUsize::new(0));
    let dc = dials.clone();
    let fwd = tokio::spawn(async move {
        loop {
            let (mut a, _) = listener.accept().await.unwrap();
            if dc.fetch_add(1, Ordering::SeqCst) > 0 {
                continue;
            }
            tokio::spawn(async move {
                let mut b = tokio::net::TcpStream::connect(("127.0.0.1", p1))
                    .await
                    .unwrap();
                a.set_nodelay(true).ok();
                b.set_nodelay(true).ok();
                let _ = tokio::io::copy_bidirectional(&mut a, &mut b).await;
            });
        }
    });
    let mut c01 = ActivationClient::new("127.0.0.1", pf);
    c01.connect_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    accepted.await.unwrap();

    let fuse = Arc::new(AtomicUsize::new(0));
    // Groups default to the rank count: 3 frames in flight.
    let mut e0 = PipelineEngine::new(
        PanicRunner {
            inner: load(0, 0, 2),
            fuse: fuse.clone(),
        },
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
    assert_eq!(e0.enable_streams(3), 3);
    assert_eq!(e1.enable_streams(3), 3);
    assert_eq!(e2.enable_streams(3), 3);
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

    let n = PROMPTS.len();
    let f = fuse.clone();
    let (e0, rounds) = tokio::task::spawn_blocking(move || {
        let mut e0 = e0;
        let healthy = run_round(&mut e0, "healthy", n);
        // 3 streams, one per group: decode calls 1-3 are each group's first
        // decode, so the 4th call fails with the other groups' frames on
        // the wire.
        f.store(4, Ordering::SeqCst);
        let failed = run_round(&mut e0, "failed", 3);
        // Give the idle link keeper time to look at the link.
        std::thread::sleep(Duration::from_secs(2));
        let after = run_round(&mut e0, "after", n);
        (e0, (healthy, failed, after))
    })
    .await
    .unwrap();
    let (healthy, failed, after) = rounds;
    let expected: Vec<Vec<i64>> = healthy
        .into_iter()
        .map(|r| r.expect("healthy round"))
        .collect();
    assert_eq!(fuse.load(Ordering::SeqCst), 0, "the panic was not injected");
    for (i, r) in failed.iter().enumerate() {
        assert!(
            r.is_err(),
            "task {i} of the failed round did not error: {r:?}"
        );
    }
    for (i, (g, e)) in after.iter().zip(&expected).enumerate() {
        assert_eq!(g.as_ref(), Ok(e), "task {i} after the local failure");
    }
    assert_eq!(
        dials.load(Ordering::SeqCst),
        1,
        "rank 0 re-dialed a healthy rank 1 after a local failure"
    );
    assert!(
        workers.iter().all(|w| !w.is_finished()),
        "a worker rank exited after a local failure on rank 0"
    );

    stop.store(true, Ordering::Relaxed);
    drop(e0);
    fwd.abort();
    c12.lock().await.close().await;
    for w in workers {
        let _ = w.join();
    }
}
