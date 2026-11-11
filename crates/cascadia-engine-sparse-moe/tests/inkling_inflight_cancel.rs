//! Single-stage in-flight cancel and stop-string for the Inkling staged runner
//! on the committed tiny export.
//!
//! The single-stage path makes the whole answer in one blocking runner call;
//! the engine polls a cancel flag before every decode token and around the one
//! prefill batch, and — with the engine stop check on — ends the run at the
//! first `stop` string found in a short decode of the newest tokens. The runner
//! tests drive the polled call directly (deterministic: a `should_stop` that
//! trips a real `InflightCancel` handle at a known token). This export is the
//! one that ships a tokenizer, so the engine tests drive the single-stage
//! `PipelineEngine` through `Engine::step` on a text prompt: the handle it hands
//! out, the empty cancelled final, and the stop-string path on the engine's own
//! decoded text.
//!
//! Inkling prefills the whole prompt as one batch (`supports_batched_prefill`
//! is true), polling only before and after it, so a cancel during prefill waits
//! for that batch and then ends without decoding.

use std::path::PathBuf;

use cascadia_engine::{Engine, InflightCancel};
use cascadia_engine_sparse_moe::dist::StageTransport;
use cascadia_engine_sparse_moe::engine::PipelineEngine;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_types::{Chunk, FinishReason, GenerationTask};

const MAX_SEQ: usize = 64;
/// Requested token cap for the reference run — large enough to cancel/stop well
/// inside the answer, within the fixture's context budget.
const N: u32 = 32;
/// The prompt the export's reference records, as runner token ids.
const PROMPT_IDS: [u32; 12] = [5, 33, 81, 53, 92, 85, 83, 43, 74, 10, 86, 84];
/// The same prompt as text (each `aNN` token encodes to id `NN`), for the
/// tokenizer-backed engine tests.
const PROMPT_TEXT: &str = "a5 a33 a81 a53 a92 a85 a83 a43 a74 a10 a86 a84";

fn fixture() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export");
    if dir.join("tokenizer.json").exists() {
        Some(dir)
    } else {
        eprintln!("inkling_export fixture (with tokenizer.json) missing; skipping");
        None
    }
}

fn runner(dir: &PathBuf) -> InklingRunner {
    InklingRunner::load_staged(dir, MAX_SEQ, 0, 1, 0, 0, Some("eager".into()), None)
        .expect("single-stage InklingRunner::load_staged")
}

fn engine(dir: &PathBuf, handle: tokio::runtime::Handle, stop_check: bool) -> Box<dyn Engine> {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let e = PipelineEngine::new(
        runner(dir),
        Some(tok),
        StageTransport::default(),
        handle,
        0,
        1,
        None,
    )
    .with_engine_stop_check(stop_check);
    Box::new(e)
}

/// A greedy (temperature 0, fixed seed) task.
fn task(id: &str, prompt: &str, max_tokens: u32) -> GenerationTask {
    let mut t = GenerationTask::new(id, prompt).with_max_tokens(max_tokens);
    t.temperature = 0.0;
    t.sampling.seed = Some(0);
    t
}

/// Run one task through the single-stage engine to its final marker; returns
/// the generated token ids, the generated text, and the final chunk.
fn drive(e: &mut dyn Engine, t: GenerationTask) -> (Vec<i64>, String, Chunk) {
    let id = t.task_id.clone();
    e.submit(t).unwrap();
    let mut ids = Vec::new();
    let mut text = String::new();
    let mut fin = None;
    // The single-stage path emits every token chunk plus the final marker in one
    // step(); loop a few times defensively in case that ever splits.
    for _ in 0..8 {
        if fin.is_some() {
            break;
        }
        for (tid, c) in e.step().unwrap() {
            assert_eq!(tid, id, "a chunk for another task");
            assert!(c.error.is_none(), "task errored: {:?}", c.error);
            if c.is_final {
                fin = Some(c);
            } else {
                ids.push(c.token_id);
                text.push_str(&c.text);
            }
        }
    }
    (ids, text, fin.expect("a final chunk"))
}

/// Four chars from the second half of `text` (no whitespace-only, no U+FFFD),
/// usable as a `stop` taken from the answer's own decoded text.
fn middle_stop(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 16 {
        return None;
    }
    let at = chars.len() / 2;
    let s: String = chars[at..at + 4].iter().collect();
    (!s.trim().is_empty() && !s.contains('\u{FFFD}')).then_some(s)
}

/// A cancel tripped after `k` decoded tokens ends the run holding exactly the
/// reference's first `k` tokens, and leaves the runner clean for the next run.
#[test]
fn inkling_single_stage_cancel_mid_decode_returns_the_reference_prefix() {
    let Some(dir) = fixture() else {
        return;
    };
    let cfg = SamplingConfig::default(); // greedy (temperature 0)
    let mut r = runner(&dir);

    let (baseline, _cap, _t) =
        r.generate_reason_timed_polled(&PROMPT_IDS, N as usize, &cfg, &mut |_| false);
    assert!(
        baseline.len() >= 6,
        "baseline too short to cancel mid-way: {baseline:?}"
    );
    let k = (baseline.len() / 3).max(1);

    let h = InflightCancel::new();
    let id = "t".to_string();
    assert!(!h.begin(&id));
    let mut decode_polls = 0usize;
    let (got, _cap, _t) =
        r.generate_reason_timed_polled(&PROMPT_IDS, N as usize, &cfg, &mut |ids| {
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

    let (again, _cap, _t) =
        r.generate_reason_timed_polled(&PROMPT_IDS, N as usize, &cfg, &mut |_| false);
    assert_eq!(again, baseline, "the next run is byte-identical to T0");

    eprintln!(
        "inkling runner: N={N} baseline_tokens={} cancel_at={k}",
        baseline.len()
    );
}

/// A cancel seen after the one prefill batch ends the run before a token is
/// decoded.
#[test]
fn inkling_single_stage_cancel_during_prefill_generates_nothing() {
    let Some(dir) = fixture() else {
        return;
    };
    let cfg = SamplingConfig::default();
    let mut r = runner(&dir);
    let (baseline, _cap, _t) =
        r.generate_reason_timed_polled(&PROMPT_IDS, N as usize, &cfg, &mut |_| false);

    let h = InflightCancel::new();
    let id = "t".to_string();
    assert!(!h.begin(&id));
    let mut prefill_polls = 0usize;
    let (got, _cap, _t) =
        r.generate_reason_timed_polled(&PROMPT_IDS, N as usize, &cfg, &mut |ids| {
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
    assert_eq!(
        prefill_polls, 2,
        "the batch ran, then the after-batch poll ended it"
    );

    let (again, _cap, _t) =
        r.generate_reason_timed_polled(&PROMPT_IDS, N as usize, &cfg, &mut |_| false);
    assert_eq!(again, baseline, "the next run is byte-identical to T0");
}

/// The single-stage engine hands out one stable handle; a cancel set before the
/// task begins ends it as an empty cancelled final, and the next task runs
/// normally to completion.
#[test]
fn inkling_single_stage_engine_cancel_before_begin_then_runs_normally() {
    let Some(dir) = fixture() else {
        return;
    };
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut e = engine(&dir, rt.handle().clone(), true);
    let h = e
        .inflight_cancel()
        .expect("single-stage engine hands out a handle");

    h.cancel(&"c1".to_string());
    e.submit(task("c1", PROMPT_TEXT, 8)).unwrap();
    let out = e.step().unwrap();
    assert_eq!(out.len(), 1, "only the cancelled final: {out:?}");
    let c = &out[0].1;
    assert!(
        c.is_final && c.error.is_none() && c.text.is_empty(),
        "{c:?}"
    );
    assert_eq!(c.finish_reason, Some(FinishReason::Cancelled));
    assert!(!h.cancelled(), "end() freed the handle");

    // The next task runs to completion, uncancelled.
    let (ids, _text, fin) = drive(&mut *e, task("c2", PROMPT_TEXT, 4));
    assert_eq!(ids.len(), 4, "the next task decoded its full budget");
    assert_ne!(fin.finish_reason, Some(FinishReason::Cancelled));
}

/// A stop taken from the middle of the engine's own baseline text ends the run
/// early with `stop`; the kept text holds the stop and the kept ids are T0's
/// prefix. With the check off, the same request runs to the end. The same
/// engine then reproduces T0 byte-for-byte.
#[test]
fn inkling_single_stage_engine_stops_at_a_stop_string() {
    let Some(dir) = fixture() else {
        return;
    };
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut e = engine(&dir, rt.handle().clone(), true);

    // T0.
    let (ids0, text0, _fin0) = drive(&mut *e, task("s0", PROMPT_TEXT, N));
    assert!(ids0.len() >= 8, "baseline too short: {ids0:?}");
    let Some(stop) = middle_stop(&text0) else {
        eprintln!("baseline text too short for a stop test: {text0:?}; skipping");
        return;
    };

    // Same task + a stop from the baseline's own text.
    let mut t = task("s1", PROMPT_TEXT, N);
    t.sampling.stop = vec![stop.clone()];
    let (ids1, text1, fin1) = drive(&mut *e, t);
    assert_eq!(fin1.finish_reason, Some(FinishReason::Stop), "{fin1:?}");
    assert!(text1.contains(&stop), "{text1:?} lacks {stop:?}");
    assert!(
        ids1.len() < ids0.len(),
        "ended before T0's length: {} < {}",
        ids1.len(),
        ids0.len()
    );
    assert_eq!(ids1[..], ids0[..ids1.len()], "the kept ids are T0's prefix");

    // The same engine runs T0 again byte-identically (no leaked state).
    let (again, _text, _fin) = drive(&mut *e, task("s2", PROMPT_TEXT, N));
    assert_eq!(again, ids0, "the next run equals T0");

    eprintln!(
        "inkling engine: N={N} baseline_tokens={} stop_at={} stop={stop:?}",
        ids0.len(),
        ids1.len()
    );

    // The check off: the same stop does not end the run early.
    drop(e);
    let mut off = engine(&dir, rt.handle().clone(), false);
    let mut t = task("s3", PROMPT_TEXT, N);
    t.sampling.stop = vec![stop];
    let (ids_off, _text, fin_off) = drive(&mut *off, t);
    assert_eq!(ids_off, ids0, "stop check off runs to the full length");
    assert_ne!(fin_off.finish_reason, Some(FinishReason::Stop));
}
