//! Single-stage in-flight cancel for the DeepSeek-V4 staged runner on the
//! committed tiny export.
//!
//! The single-stage path makes the whole answer in one blocking
//! `generate_reason_timed_polled` call; the engine polls a cancel flag before
//! every prefill token and every decode token and ends the run the moment it is
//! set. These tests drive that polled call directly — deterministically, with a
//! `should_stop` that trips a real `InflightCancel` handle at a known token —
//! and drive the single-stage `PipelineEngine` through `Engine::step` for the
//! handle it hands out.
//!
//! This export ships no tokenizer, so the engine's text path (encode/decode,
//! and the in-engine stop-string check that needs decoded text) is exercised on
//! the Inkling export, which does ship one. Here the runner is driven on raw
//! token ids — the same ids the reference records. DeepSeek-V4 prefills one
//! token at a time (`supports_batched_prefill` is false), so a cancel during
//! prefill stops before the next prefill token.

use std::path::PathBuf;
use std::sync::Arc;

use cascadia_engine::{Engine, InflightCancel};
use cascadia_engine_sparse_moe::dist::StageTransport;
use cascadia_engine_sparse_moe::dsv4::stage::Dsv4Runner;
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_types::{FinishReason, GenerationTask};

const MAX_SEQ: usize = 64;
/// Requested token cap for the reference run — large enough to cancel well
/// inside the answer, within the fixture's context budget.
const N: usize = 40;

fn fixture() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export");
    if dir.join("manifest.json").exists() {
        Some(dir)
    } else {
        eprintln!("dsv4_export fixture missing; skipping");
        None
    }
}

/// The prompt the export's reference records, as runner token ids.
fn reference_prompt(dir: &PathBuf) -> Vec<u32> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    v["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn runner(dir: &PathBuf) -> Dsv4Runner {
    Dsv4Runner::load_staged(dir, MAX_SEQ, 0, 1, 0, 0).expect("single-stage Dsv4Runner::load_staged")
}

/// A cancel tripped after `k` decoded tokens ends the run holding exactly the
/// reference's first `k` tokens, and leaves the runner clean for the next run.
#[test]
fn dsv4_single_stage_cancel_mid_decode_returns_the_reference_prefix() {
    let Some(dir) = fixture() else {
        return;
    };
    let prompt = reference_prompt(&dir);
    let cfg = SamplingConfig::default(); // greedy (temperature 0)
    let mut r = runner(&dir);

    // T0: the full greedy reference.
    let (baseline, _cap, _t) = r.generate_reason_timed_polled(&prompt, N, &cfg, &mut |_| false);
    assert!(
        baseline.len() >= 6,
        "baseline too short to cancel mid-way: {baseline:?}"
    );
    let k = (baseline.len() / 3).max(1);

    // Trip the in-flight handle once k tokens have been decoded. The runner
    // polls `should_stop` before each decode forward, so the run ends holding
    // exactly the first k tokens.
    let h = InflightCancel::new();
    let id = "t".to_string();
    assert!(!h.begin(&id));
    let mut decode_polls = 0usize;
    let (got, _cap, _t) = r.generate_reason_timed_polled(&prompt, N, &cfg, &mut |ids| {
        if !ids.is_empty() {
            decode_polls += 1;
        }
        if ids.len() >= k {
            h.cancel(&id);
        }
        h.cancelled()
    });
    h.end();
    assert_eq!(
        got,
        baseline[..k],
        "a cancel at k returns the reference prefix"
    );
    assert!(
        got.len() < baseline.len(),
        "ended early: {} < {}",
        got.len(),
        baseline.len()
    );
    assert_eq!(
        decode_polls, k,
        "polled once per decoded token up to the cancel"
    );

    // No state leaked from the cut run: the next identical run equals T0.
    let (again, _cap, _t) = r.generate_reason_timed_polled(&prompt, N, &cfg, &mut |_| false);
    assert_eq!(again, baseline, "the next run is byte-identical to T0");

    eprintln!(
        "dsv4: N={N} baseline_tokens={} cancel_at={k}",
        baseline.len()
    );
}

/// A cancel seen during prefill ends the run before a single token is decoded.
#[test]
fn dsv4_single_stage_cancel_during_prefill_generates_nothing() {
    let Some(dir) = fixture() else {
        return;
    };
    let prompt = reference_prompt(&dir);
    assert!(prompt.len() > 2, "the prompt must outlast the cut point");
    let cfg = SamplingConfig::default();
    let mut r = runner(&dir);
    let (baseline, _cap, _t) = r.generate_reason_timed_polled(&prompt, N, &cfg, &mut |_| false);

    // DeepSeek-V4 prefills one token at a time, polling before each. Cancel at
    // the second prefill poll: prefill never finishes, so nothing is decoded.
    let h = InflightCancel::new();
    let id = "t".to_string();
    assert!(!h.begin(&id));
    let mut prefill_polls = 0usize;
    let (got, _cap, _t) = r.generate_reason_timed_polled(&prompt, N, &cfg, &mut |ids| {
        if ids.is_empty() {
            prefill_polls += 1;
            if prefill_polls >= 2 {
                h.cancel(&id);
            }
        }
        h.cancelled()
    });
    h.end();
    assert!(
        got.is_empty(),
        "a cancel during prefill generates nothing: {got:?}"
    );
    assert_eq!(prefill_polls, 2, "stopped at the second prefill poll");

    // The single-stage path keeps no prefix cache, so a partial prefill writes
    // nothing to protect; the next identical run still equals T0.
    let (again, _cap, _t) = r.generate_reason_timed_polled(&prompt, N, &cfg, &mut |_| false);
    assert_eq!(again, baseline, "the next run is byte-identical to T0");
}

/// The single-stage engine hands out one stable in-flight handle; a cancel set
/// before the task begins ends it as an empty cancelled final, before it ever
/// tokenizes or forwards (so this runs without a tokenizer).
#[test]
fn dsv4_single_stage_engine_hands_out_the_handle_and_cancels_before_begin() {
    let Some(dir) = fixture() else {
        return;
    };
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let r = runner(&dir);
    let mut e = PipelineEngine::new(
        r,
        None::<tokenizers::Tokenizer>,
        StageTransport::default(),
        rt.handle().clone(),
        0,
        1,
        None,
    );
    let h = e
        .inflight_cancel()
        .expect("single-stage engine hands out a handle");
    assert!(
        Arc::ptr_eq(&h, &e.inflight_cancel().unwrap()),
        "the same handle each time"
    );

    h.cancel(&"c1".to_string());
    e.submit(GenerationTask::new("c1", "unused").with_max_tokens(8))
        .unwrap();
    let out = e.step().unwrap();
    assert_eq!(out.len(), 1, "only the cancelled final: {out:?}");
    let c = &out[0].1;
    assert!(
        c.is_final && c.error.is_none() && c.text.is_empty(),
        "{c:?}"
    );
    assert_eq!(c.finish_reason, Some(FinishReason::Cancelled));
    assert!(!h.cancelled(), "end() freed the handle");
}
