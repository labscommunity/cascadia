//! Task 3 (issue-34 Option B): `OvMoeEngine`'s multi-stage rank-0 path
//! converted from a whole-turn burst driver to a per-token streaming state
//! machine (`begin_generation_ovmoe` / `decode_step_ovmoe` / `finalize_ovmoe`),
//! plus streamed Option B resume and the single-stage sentinel decline.
//! Mirrors `tests/sparse_streaming.rs` (Task 2) but drives the OV-IR
//! (MiniMax-M2) backend over a REAL two-rank loopback transport.
//!
//! Gated on `M2_MODEL_DIR` (see `tests/minimax_m2_eval.rs`): the fixture
//! cannot be committed. CI skips this file; the rig cert is the enforcement
//! gate for what's below.
//!
//! `Engine::step` is SYNC and `block_on`s internally — every test here drives
//! `head.step()` from a plain `#[test]` thread. A `#[tokio::test]` would
//! deadlock (the calling thread would already be inside the single-threaded
//! test runtime `step()` tries to re-enter).

use std::path::PathBuf;
use std::thread::JoinHandle;

use cascadia_engine::{Builder, Engine};
use cascadia_engine_sparse_moe::{Manifest, SparseMoEBuilder, SparseMoEBuilderConfig};
use cascadia_types::{FinishReason, GenerationTask, PeerEndpoint, PeerLayout, ShardSpec};

fn model_dir() -> Option<PathBuf> {
    std::env::var("M2_MODEL_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn shard(is_first: bool, is_last: bool) -> ShardSpec {
    ShardSpec {
        model_id: "m2".into(),
        // 0/0: `load_staged` treats an unset layer_end as "even split" for
        // OV-IR shells (engine.rs `is_ov` branch) — no need to compute it here.
        layer_start: 0,
        layer_end: 0,
        total_layers: 0,
        device: "CPU".into(),
        is_first_stage: is_first,
        is_last_stage: is_last,
        tp_size: 1,
        tp_rank: 0,
    }
}

/// OS-assigned free TCP port. Small bind-then-drop TOCTOU race, acceptable
/// for a rig-only, non-CI test.
fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = l.local_addr().expect("local_addr").port();
    drop(l);
    port
}

/// Build a live 2-rank OV-IR chain: rank 0 (head, the API-facing driver
/// under test) and rank 1 (worker/last-stage). Same scaffold as
/// `tests/sparse_streaming.rs::two_rank_harness`, generalized to the OV-IR
/// (MiniMax-M2) backend built the way `tests/minimax_m2_eval.rs`'s
/// `minimax_m2_two_rank_pipeline_matches_hf_reference` builds one, but
/// through the real `Engine`/wire path instead of driving `OvMoeRunner`
/// directly.
fn two_rank_harness() -> (Box<dyn Engine>, JoinHandle<()>) {
    let dir = model_dir().expect("caller must gate on model_dir() first");
    let port = free_port();

    let worker_dir = dir.clone();
    let worker_thread = std::thread::spawn(move || {
        let worker_rt = tokio::runtime::Runtime::new().expect("worker runtime");
        let mut worker: Box<dyn Engine> = worker_rt.block_on(async move {
            let mut wb = SparseMoEBuilder::new(
                SparseMoEBuilderConfig::new(worker_dir.to_str().expect("utf8 path"), "CPU")
                    .with_rank(1, 2),
            );
            wb.configure_listen("127.0.0.1", port);
            wb.connect(PeerLayout::last_of(PeerEndpoint::new("127.0.0.1", port)))
                .await
                .expect("worker connect");
            let _progress = wb.load(shard(false, true)).await.expect("worker load");
            Box::new(wb).build().expect("worker build")
        });
        // Drive the worker for the rest of the test process; no explicit
        // stop signal needed (matches sparse_streaming.rs).
        loop {
            let _ = worker.step();
        }
    });

    let head_shard_dir = dir.clone();
    let head_rt = tokio::runtime::Runtime::new().expect("head runtime");
    let head: Box<dyn Engine> = head_rt.block_on(async move {
        let mut hb = SparseMoEBuilder::new(
            SparseMoEBuilderConfig::new(head_shard_dir.to_str().expect("utf8 path"), "CPU")
                .with_rank(0, 2),
        );
        hb.connect(PeerLayout::first_of(PeerEndpoint::new("127.0.0.1", port)))
            .await
            .expect("head connect");
        let _progress = hb.load(shard(true, false)).await.expect("head load");
        Box::new(hb).build().expect("head build")
    });
    std::mem::forget(head_rt);

    (head, worker_thread)
}

/// Single-stage (`total == 1`) OV-IR engine, used only to exercise the
/// sentinel decline path — MiniMax-M2 single-stage has no forced-prefix
/// resume implementation.
fn single_stage_ovmoe_harness() -> Box<dyn Engine> {
    let dir = model_dir().expect("caller must gate on model_dir() first");
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let engine: Box<dyn Engine> = rt.block_on(async move {
        let mut b = SparseMoEBuilder::new(SparseMoEBuilderConfig::new(
            dir.to_str().expect("utf8 path"),
            "CPU",
        ));
        let _progress = b.load(shard(true, true)).await.expect("load");
        Box::new(b).build().expect("build")
    });
    std::mem::forget(rt);
    engine
}

// Streaming shape: interior token chunks + one empty final marker.
// OvMoe INCLUDES the EOS token in the output (push-then-test), unlike
// SparseMoE — no separate exclusion assertion needed here, just the shape.
#[test]
fn multi_stage_streams_per_token_with_empty_final() {
    let Some(_dir) = model_dir() else {
        eprintln!("M2_MODEL_DIR not set; skipping");
        return;
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

/// OvMoe INCLUDES the EOS token in the output — push-then-test, preserved
/// from the old monolithic decode loop (`generated.push(next_u)` BEFORE the
/// `eos.contains(&next_u)` check). Drive with a generous budget so a model
/// that naturally reaches EOS within it does so; if it does, EOS must be the
/// LAST interior token chunk (nothing streams after it) and count toward
/// n_tokens. If the model never reaches EOS within budget, the shape check
/// alone (exactly one empty final) still holds.
#[test]
fn eos_token_is_included_in_output() {
    let Some(dir) = model_dir() else {
        eprintln!("M2_MODEL_DIR not set; skipping");
        return;
    };
    let manifest = Manifest::load(&dir).expect("manifest");
    let eos = manifest.eos_token_ids.clone();
    let (mut head, _worker) = two_rank_harness();
    head.submit(GenerationTask::new("teos", "hello").with_max_tokens(64))
        .unwrap();
    let mut chunks = Vec::new();
    for _ in 0..256 {
        chunks.extend(head.step().unwrap());
        if chunks.iter().any(|(_, c)| c.is_final) {
            break;
        }
    }
    let finals: Vec<_> = chunks.iter().filter(|(_, c)| c.is_final).collect();
    assert_eq!(finals.len(), 1);
    let toks: Vec<_> = chunks.iter().filter(|(_, c)| !c.is_final).collect();
    if let Some(pos) = toks
        .iter()
        .position(|(_, c)| eos.contains(&(c.token_id as u32)))
    {
        assert_eq!(
            pos,
            toks.len() - 1,
            "EOS must be the LAST interior chunk — OvMoe includes it, so nothing streams after"
        );
        assert_eq!(finals[0].1.finish_reason, Some(FinishReason::Stop));
    }
}

#[test]
fn resume_seed_streams_continuation_only() {
    let Some(dir) = model_dir() else {
        eprintln!("M2_MODEL_DIR not set; skipping");
        return;
    };
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("model tokenizer");
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
    assert!(toks.len() <= 4);
    let streamed: String = toks.iter().map(|(_, c)| c.text.as_str()).collect();
    let seed_text = tokenizer.decode(&[3u32, 4], true).unwrap();
    assert!(
        !streamed.starts_with(&seed_text) || seed_text.is_empty(),
        "streamed deltas re-emitted the forced prefix: {streamed:?}"
    );
}

#[test]
fn zero_budget_resume_finals_immediately_length() {
    let Some(_dir) = model_dir() else {
        eprintln!("M2_MODEL_DIR not set; skipping");
        return;
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
    let Some(_dir) = model_dir() else {
        eprintln!("M2_MODEL_DIR not set; skipping");
        return;
    };
    let (mut head, _worker) = two_rank_harness();
    head.submit(GenerationTask::new("t4", "hello").with_max_tokens(50))
        .unwrap();
    let first = head.step().unwrap(); // begin + first token
    assert!(!first.is_empty());
    head.cancel(&"t4".to_string());
    let after = head.step().unwrap();
    assert!(
        after.is_empty(),
        "cancelled task must emit nothing and free the slot"
    );
}

/// Single-stage MiniMax-M2 has no forced-prefix resume implementation
/// (unlike single-stage SparseMoE — see `crates/cascadia-types/src/task.rs`'s
/// `append_resume_ids`). A resumed task must be declined with the
/// `resume_unsupported:` sentinel as the FIRST and ONLY chunk, so the
/// scheduler's B6 retry excludes this peer and re-routes instead of silently
/// regenerating from scratch.
#[test]
fn single_stage_declines_seeded_task_with_sentinel_first_chunk() {
    let Some(_dir) = model_dir() else {
        eprintln!("M2_MODEL_DIR not set; skipping");
        return;
    };
    let mut eng = single_stage_ovmoe_harness();
    let mut task = GenerationTask::new("t", "hi").with_max_tokens(4);
    task.resume_token_ids = Some(vec![3]);
    eng.submit(task).unwrap();
    let chunks = eng.step().unwrap();
    assert_eq!(chunks.len(), 1, "decline must be the FIRST and only chunk");
    let c = &chunks[0].1;
    let reason = c
        .error
        .as_deref()
        .expect("decline is an error chunk (Chunk.error field)");
    assert!(
        reason.starts_with("resume_unsupported:"),
        "sentinel must PREFIX the reason (enterprise starts_with carve-out)"
    );
}
