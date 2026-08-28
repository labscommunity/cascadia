//! Task 2 (issue-34 Option B): `SparseMoEEngine`'s multi-stage rank-0 path
//! converted from a whole-turn burst driver to a per-token streaming state
//! machine (`begin_generation_sparse` / `decode_step_sparse` /
//! `finalize_sparse`). These tests drive that state machine through
//! `Engine::step` over a REAL two-rank loopback transport — the same wire
//! path production uses — so a regression in the splice between prefill and
//! decode, or in the resume-seed streaming shape, shows up here instead of
//! only in the rig cert.
//!
//! Gated on `K26_TINY_HEAD_DIR`: a K2.6 tiny tree exported WITH
//! `tools/export_kimi_k26.py --head`, built `--features openvino`. The worker
//! rank here is a LAST stage, which hard-requires `head/openvino_model.xml`
//! and a real OV runtime to compile it — deliberately a DIFFERENT env var
//! from `K26_TINY_DIR`, whose tree `tests/k26_native_tiny.rs` asserts is
//! head-LESS (one tree cannot satisfy both files). The fixture cannot be
//! committed (~1.4 GiB, K2.6 dims pinned at compile time). CI skips this
//! file; the hardware cert is the enforcement gate for what's below.
//!
//! `Engine::step` is SYNC and `block_on`s internally (`engine.rs`'s
//! `SparseMoEEngine::block_on`) — every test here drives `head.step()` from a
//! plain `#[test]` thread. A `#[tokio::test]` would deadlock on that
//! `block_on` (the calling thread would already be inside the single-threaded
//! test runtime `step()` tries to re-enter).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use cascadia_engine::{Builder, Engine};
use cascadia_engine_sparse_moe::{Manifest, SparseMoEBuilder, SparseMoEBuilderConfig};
use cascadia_types::{FinishReason, GenerationTask, PeerEndpoint, PeerLayout, ShardSpec};

fn tiny_dir() -> Option<PathBuf> {
    if cfg!(not(feature = "openvino")) {
        eprintln!(
            "sparse_streaming: skipping (build with --features openvino — \
             the last-stage worker compiles the OV head IR)"
        );
        return None;
    }
    let dir = match std::env::var("K26_TINY_HEAD_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("K26_TINY_HEAD_DIR not set; skipping");
            return None;
        }
    };
    if !dir.join("head").join("openvino_model.xml").exists() {
        eprintln!(
            "K26_TINY_HEAD_DIR tree has no head/openvino_model.xml \
             (re-export with export_kimi_k26.py --head); skipping"
        );
        return None;
    }
    Some(dir)
}

/// `layer_end` covering every MoE layer in the manifest (ids are 1-based) —
/// same helper as `tests/k26_native_tiny.rs`, duplicated rather than shared
/// because integration test binaries can't import each other.
fn moe_end(m: &Manifest) -> u32 {
    m.moe_layer_ids().last().copied().unwrap_or(0) + 1
}

/// BOTH ranks get the full MoE layer range: these tests target the WIRE path
/// (streaming shape, resume seams, chain death), not memory economy, and the
/// tiny fixture makes a duplicate stack cheap. Contrast ovmoe_streaming.rs,
/// whose 0/0 layer_end gets a real even split for free from `load_staged`.
fn shard(m: &Manifest, is_first: bool, is_last: bool) -> ShardSpec {
    ShardSpec {
        model_id: "k26_tiny".into(),
        layer_start: 1,
        layer_end: moe_end(m),
        total_layers: m.num_layers,
        device: "CPU".into(),
        is_first_stage: is_first,
        is_last_stage: is_last,
        tp_size: 1,
        tp_rank: 0,
    }
}

/// The model's own tokenizer — needed so the seed-text comparison in
/// `resume_seed_streams_continuation_only` decodes with the exact vocab the
/// engine under test uses (the fixed ids in these tests, e.g. `[3, 4]`, are
/// only guaranteed in-vocab for THIS tokenizer, not any arbitrary one).
fn test_tokenizer() -> tokenizers::Tokenizer {
    let dir = tiny_dir().expect("caller must gate on tiny_dir() first");
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("model tokenizer")
}

/// OS-assigned free TCP port. Small bind-then-drop TOCTOU race (another
/// process could grab it before our own bind below), acceptable for a
/// rig-only, non-CI test.
fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = l.local_addr().expect("local_addr").port();
    drop(l);
    port
}

/// Build a live 2-rank chain: rank 0 (head, the API-facing driver under
/// test) and rank 1 (worker/last-stage, drives itself off upstream frames)
/// wired over a real loopback socket — the transport scaffold in
/// `tests/dsv4_wire_e2e.rs:27-60`, generalized to full `SparseMoEEngine`s
/// built the way `tests/k26_native_tiny.rs:124-149` builds one.
///
/// Each rank gets its OWN tokio runtime (mirrors `packed_wire_stress.rs`'s
/// DRIVER/RELAY split — one process per rank in production). The worker's
/// runtime + step loop live on a dedicated thread for the lifetime of the
/// test process; the head's runtime is intentionally leaked (`mem::forget`)
/// because `head`'s captured `Handle` must outlive every `head.step()` call
/// the *caller* makes, and this function has no join point to hand that
/// lifetime back through the 2-tuple the tests destructure.
fn two_rank_harness() -> (Box<dyn Engine>, JoinHandle<()>) {
    let (head, worker, _kill) = two_rank_harness_killable();
    (head, worker)
}

/// `two_rank_harness` plus a kill switch: setting the flag makes the worker
/// loop return (dropping the engine + its runtime, closing the sockets), so a
/// test can simulate mid-decode chain death and join the worker thread.
fn two_rank_harness_killable() -> (Box<dyn Engine>, JoinHandle<()>, Arc<AtomicBool>) {
    let (head, worker, kill, _mute) = two_rank_harness_faultable();
    (head, worker, kill)
}

/// `two_rank_harness_killable` plus a MUTE switch: setting it makes the worker
/// loop spin WITHOUT stepping — its sockets stay open but it never replies.
/// This is the rig's real mid-decode death shape: the enterprise bridge parks
/// a dead pipeline leg instead of closing the engine-side loopback, so the
/// head sees silence, not a transport error (rig 2026-08-27 resume stall).
fn two_rank_harness_faultable() -> (
    Box<dyn Engine>,
    JoinHandle<()>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
) {
    let dir = tiny_dir().expect("caller must gate on tiny_dir() first");
    let manifest = Manifest::load(&dir).expect("manifest");
    let port = free_port();

    // The worker's startup errors must reach the test: its JoinHandle is
    // never joined, so a bare `expect` panic there is swallowed and the
    // head then blocks in connect for CASCADIA_CONNECT_TIMEOUT_SECS
    // (default 300s) with the real cause invisible.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let kill = Arc::new(AtomicBool::new(false));
    let kill_worker = Arc::clone(&kill);
    let mute = Arc::new(AtomicBool::new(false));
    let mute_worker = Arc::clone(&mute);
    let worker_dir = dir.clone();
    let worker_shard = shard(&manifest, false, true);
    let worker_thread = std::thread::spawn(move || {
        let worker_rt = tokio::runtime::Runtime::new().expect("worker runtime");
        let built: Result<Box<dyn Engine>, String> = worker_rt.block_on(async move {
            let mut wb = SparseMoEBuilder::new(
                SparseMoEBuilderConfig::new(worker_dir.to_str().expect("utf8 path"), "CPU")
                    .with_rank(1, 2),
            );
            wb.configure_listen("127.0.0.1", port);
            wb.connect(PeerLayout::last_of(PeerEndpoint::new("127.0.0.1", port)))
                .await
                .map_err(|e| format!("worker connect: {e:?}"))?;
            let _progress = wb
                .load(worker_shard)
                .await
                .map_err(|e| format!("worker load: {e:?}"))?;
            Box::new(wb)
                .build()
                .map_err(|e| format!("worker build: {e:?}"))
        });
        let mut worker = match built {
            Ok(w) => {
                let _ = ready_tx.send(Ok(()));
                w
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                return;
            }
        };
        // Drive the worker for the rest of the test process. When the head
        // side eventually drops (process exit), `step_worker` degrades to
        // its WORKER_BACKOFF idle loop rather than erroring hard, so this
        // never needs an explicit stop signal.
        loop {
            if kill_worker.load(Ordering::Relaxed) {
                // Simulated chain death: return, dropping the engine and its
                // runtime — sockets close and the head's next forward fails.
                return;
            }
            if mute_worker.load(Ordering::Relaxed) {
                // Mute: alive, sockets open, never replies.
                std::thread::sleep(std::time::Duration::from_millis(20));
                continue;
            }
            let _ = worker.step();
        }
    });

    // The worker's accept only unblocks once the head dials in, so the head
    // MUST be built concurrently — readiness is checked after, not before.
    let head_shard = shard(&manifest, true, false);
    let head_rt = tokio::runtime::Runtime::new().expect("head runtime");
    let head_res: Result<Box<dyn Engine>, String> = head_rt.block_on(async move {
        let mut hb = SparseMoEBuilder::new(
            SparseMoEBuilderConfig::new(dir.to_str().expect("utf8 path"), "CPU").with_rank(0, 2),
        );
        hb.connect(PeerLayout::first_of(PeerEndpoint::new("127.0.0.1", port)))
            .await
            .map_err(|e| format!("head connect: {e:?}"))?;
        let _progress = hb
            .load(head_shard)
            .await
            .map_err(|e| format!("head load: {e:?}"))?;
        Box::new(hb)
            .build()
            .map_err(|e| format!("head build: {e:?}"))
    });
    std::mem::forget(head_rt);
    let head = match head_res {
        Ok(h) => h,
        Err(e) => {
            // Surface the worker's failure as the likely root cause.
            let worker_err = match ready_rx.try_recv() {
                Ok(Err(w)) => format!(" (worker: {w})"),
                _ => String::new(),
            };
            panic!("head startup failed: {e}{worker_err}");
        }
    };
    match ready_rx.recv_timeout(std::time::Duration::from_secs(300)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("worker startup failed: {e}"),
        Err(_) => panic!("worker not ready within 300s"),
    }

    (head, worker_thread, kill, mute)
}

// Streaming shape: interior token chunks + one empty final marker.
// EOS EXCLUSION: the eos token, when sampled, is NOT in the output ids.
#[test]
fn multi_stage_streams_per_token_with_empty_final() {
    let Some(_dir) = tiny_dir() else {
        return; // gate already printed why
    };
    let (mut head, _worker) = two_rank_harness();
    head.submit(GenerationTask::new("t1", "hello").with_max_tokens(4))
        .unwrap();
    let mut chunks = Vec::new();
    for _ in 0..64 {
        chunks.extend(head.step().unwrap());
        if chunks.iter().any(|(_, c)| c.is_final) {
            break;
        }
    }
    let finals: Vec<_> = chunks.iter().filter(|(_, c)| c.is_final).collect();
    assert_eq!(finals.len(), 1);
    assert!(finals[0].1.text.is_empty(), "final marker carries no text");
    assert_eq!(finals[0].1.n_tokens, Some(0));
    let toks: Vec<_> = chunks.iter().filter(|(_, c)| !c.is_final).collect();
    assert!(!toks.is_empty(), "interior token chunks must exist");
    for (_, c) in &toks {
        assert_eq!(c.n_tokens, Some(1));
        assert_eq!(
            c.token_ids,
            vec![c.token_id],
            "token_ids stamped (poison bypass)"
        );
    }
}

#[test]
fn resume_seed_streams_continuation_only() {
    let Some(_dir) = tiny_dir() else {
        return; // gate already printed why
    };
    let (mut head, _worker) = two_rank_harness();
    let mut task = GenerationTask::new("t2", "hello").with_max_tokens(6);
    task.resume_token_ids = Some(vec![3, 4]); // in-vocab for the test model
    head.submit(task).unwrap();
    let mut chunks = Vec::new();
    for _ in 0..64 {
        chunks.extend(head.step().unwrap());
        if chunks.iter().any(|(_, c)| c.is_final) {
            break;
        }
    }
    // resume_max_new: 6 - 2 = at most 4 NEW tokens; the seed ids are never re-emitted.
    let toks: Vec<_> = chunks.iter().filter(|(_, c)| !c.is_final).collect();
    assert!(
        !toks.is_empty(),
        "resume must stream at least one continuation token — an empty stream \
         passes every assertion below vacuously"
    );
    assert!(toks.len() <= 4);
    let fin = chunks
        .iter()
        .find(|(_, c)| c.is_final)
        .expect("a final chunk");
    if fin.1.finish_reason == Some(FinishReason::Length) {
        assert_eq!(
            toks.len(),
            4,
            "Length final => exactly max_tokens - seed_len interior chunks"
        );
    }
    // The seed text must NOT re-stream: concatenated deltas start strictly
    // after the decoded prefix (emitted-cursor seeding).
    let streamed: String = toks.iter().map(|(_, c)| c.text.as_str()).collect();
    let seed_text = test_tokenizer().decode(&[3u32, 4], true).unwrap();
    assert!(
        !seed_text.is_empty(),
        "fixture ids must decode to text or the re-emission assertion below is vacuous"
    );
    assert!(
        !streamed.starts_with(&seed_text),
        "streamed deltas re-emitted the forced prefix: {streamed:?}"
    );
}

#[test]
fn zero_budget_resume_finals_immediately_length() {
    let Some(_dir) = tiny_dir() else {
        return; // gate already printed why
    };
    let (mut head, _worker) = two_rank_harness();
    let mut task = GenerationTask::new("t3", "hello").with_max_tokens(2);
    task.resume_token_ids = Some(vec![3, 4]); // budget fully consumed by seed
    head.submit(task).unwrap();
    let chunks = head.step().unwrap();
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].1.is_final);
    assert_eq!(chunks[0].1.n_tokens, Some(0));
    assert_eq!(chunks[0].1.finish_reason, Some(FinishReason::Length));
}

#[test]
fn cancel_mid_decode_clears_active() {
    let Some(_dir) = tiny_dir() else {
        return; // gate already printed why
    };
    let (mut head, _worker) = two_rank_harness();
    head.submit(GenerationTask::new("t4", "hello").with_max_tokens(50))
        .unwrap();
    let first = head.step().unwrap(); // begin + first token
    assert!(!first.is_empty());
    assert!(
        !first[0].1.is_final,
        "first step must be an interior token — a final here means no \
         generation was active and the cancel below tests nothing"
    );
    head.cancel(&"t4".to_string());
    let after = head.step().unwrap();
    assert!(
        after.is_empty(),
        "cancelled task must emit nothing and free the slot"
    );
    // Recovery: the worker was abandoned mid-sequence; the next task's begin
    // issues a fresh chain reset, so a full turn must still complete cleanly
    // (protocol desync here is the actual failure cancel risks).
    head.submit(GenerationTask::new("t5", "hello again").with_max_tokens(3))
        .unwrap();
    let mut chunks = Vec::new();
    for _ in 0..64 {
        chunks.extend(head.step().unwrap());
        if chunks.iter().any(|(_, c)| c.is_final) {
            break;
        }
    }
    assert!(
        chunks.iter().any(|(_, c)| c.is_final && c.error.is_none()),
        "post-cancel task must complete cleanly: {chunks:?}"
    );
}

/// The headline Option B invariant: a mid-decode chain death surfaces an
/// ERROR chunk and NEVER a success-shaped final — a final marker would trip
/// the scheduler's saw_final latch and permanently block the forced-prefix
/// rescue. (Deliberate divergence from PipelineEngine's partial-final; see
/// the decode step's Err arm.) Reverting that arm to the old partial-final
/// behavior passes every other test in this suite.
#[test]
fn mid_decode_chain_death_is_an_error_chunk_never_a_final() {
    let Some(_dir) = tiny_dir() else {
        return; // gate already printed why
    };
    let (mut head, worker, kill) = two_rank_harness_killable();
    head.submit(GenerationTask::new("tkill", "hello").with_max_tokens(400))
        .unwrap();
    // Stream a few tokens first so the death is genuinely MID-decode.
    let mut chunks = Vec::new();
    for _ in 0..3 {
        chunks.extend(head.step().unwrap());
    }
    assert!(
        chunks.iter().all(|(_, c)| !c.is_final),
        "budget 400 must not finish within 3 steps: {chunks:?}"
    );
    kill.store(true, Ordering::Relaxed);
    worker.join().expect("worker thread exits on kill");
    // Keep stepping until the failure surfaces (the in-flight recv trips its
    // bounded transport timeout first).
    let mut saw_error = false;
    'outer: for _ in 0..64 {
        let out = head.step().unwrap();
        for (_, c) in &out {
            assert!(
                c.error.is_some() || !c.is_final,
                "chain death must never produce a success-shaped final: {c:?}"
            );
            if c.error.is_some() {
                saw_error = true;
                break 'outer;
            }
        }
    }
    assert!(saw_error, "chain death must surface an error chunk");
}

/// The 2026-08-27 rig stall: a downstream that goes SILENT (socket open, no
/// replies — the bridge parks dead legs, it does not close them) must surface
/// an error chunk within the engine's bounded reply deadline, well under the
/// gateway's 180 s inter-token budget — not hang until the gateway kills the
/// stream with nothing for the resume splicer to rescue.
#[test]
fn mid_decode_silent_worker_errors_within_deadline() {
    let Some(_dir) = tiny_dir() else {
        return; // gate already printed why
    };
    let (mut head, _worker, _kill, mute) = two_rank_harness_faultable();
    head.submit(GenerationTask::new("tmute", "hello").with_max_tokens(400))
        .unwrap();
    let mut chunks = Vec::new();
    for _ in 0..3 {
        chunks.extend(head.step().unwrap());
    }
    assert!(chunks.iter().all(|(_, c)| !c.is_final));
    mute.store(true, Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let mut saw_error = false;
    'outer: while t0.elapsed() < std::time::Duration::from_secs(170) {
        let out = head.step().unwrap();
        for (_, c) in &out {
            assert!(
                c.error.is_some() || !c.is_final,
                "silent worker must never produce a success-shaped final: {c:?}"
            );
            if c.error.is_some() {
                saw_error = true;
                break 'outer;
            }
        }
    }
    assert!(
        saw_error,
        "no error chunk within 170s — the bounded reply deadline did not fire \
         (gateway inter-token budget is 180s; the engine must beat it)"
    );
}
