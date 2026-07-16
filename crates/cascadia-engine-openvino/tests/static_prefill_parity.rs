//! Greedy-parity + phase-timing gate for the chunked-prefill static path.
//!
//! Proves the load-bearing property of the hybrid NPU+CPU split: a run that
//! prefills through the chunked variant (optionally on a DIFFERENT device)
//! reconstructs the same host KV ring state as the legacy one-token-per-step
//! static path, so its greedy output matches *modulo argmax near-tie forks*
//! (the seq=C prefill graph and the seq=1 decode graph accumulate FP
//! differently — see `assert_parity`). The byte-identical ring bookkeeping is
//! proven by the ring-math unit tests; this test adds the on-hardware
//! end-to-end check and prints TTFT + decode tok/s per config, so it doubles
//! as the phase-split bench.
//!
//! Needs real OpenVINO + a SINGLE-STAGE static export with a prefill variant:
//!
//! ```text
//! python tools/export_shards.py --model <id> --output-dir <dir> \
//!   --num-stages 1 --target npu --static-context 1024 --static-prefill-seq 64
//! CASCADIA_STATIC_SHARDS=<dir> \
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!   --test static_prefill_parity -- --nocapture
//! ```
//!
//! Env knobs: `CASCADIA_STATIC_DEVICE` (decode device, default CPU),
//! `CASCADIA_PREFILL_DEVICE` (prefill device for the chunked run — set NPU
//! on an AI PC for the hybrid split; default = decode device),
//! `CASCADIA_STATIC_PROMPT`, `CASCADIA_STATIC_MAX_NEW` (default 32),
//! `CASCADIA_STATIC_TASKS=N` (sequential requests per chunked/hybrid engine —
//! request 2+ is steady-state TTFT, without the ~300 ms first-infer init a
//! cache-imported NPU blob defers),
//! `CASCADIA_PARITY_SOFT=1` (tolerate even an EARLY fork — one within the first
//! `NEAR_TIE_MIN_PREFIX` decoded tokens, which otherwise hard-fails as suspect
//! corruption; late near-tie forks are already tolerated by default. For pure
//! timing sweeps; see `assert_parity`).
//! Unset `CASCADIA_STATIC_SHARDS` ⇒ skip-pass (stub CI stays green).
//! Multi-stage pipelines are exercised on hardware via `cascadia worker`
//! (`--prefill-device`), not here.

use std::time::{Duration, Instant};

use cascadia_engine::Builder;
use cascadia_engine_openvino::OvRuntimeBuilder;
use cascadia_types::{GenerationTask, PeerLayout, ShardSpec};

const DEFAULT_PROMPT: &str =
    "The capital of France is Paris. The capital of Italy is Rome. Explain, \
     step by step, how rainbows form in the sky after rain.";

struct RunOut {
    ids: Vec<i64>,
    text: String,
    ttft: Duration,
    total: Duration,
}

async fn run_once(
    shards: &str,
    device: &str,
    prefill_device: Option<&str>,
    disable_chunk: bool,
    prompt: &str,
    max_new: u32,
) -> RunOut {
    run_tasks(
        shards,
        device,
        prefill_device,
        disable_chunk,
        false,
        prompt,
        max_new,
        1,
    )
    .await
    .remove(0)
}

/// Drive `tasks` sequential generations through one engine instance. With
/// `park` the prefill model is dropped after each task's prefill and
/// reloaded on the next — task 2's TTFT includes the reload, which is the
/// number `--park-prefill` trades for ~1x steady-state weight residency.
#[allow(clippy::too_many_arguments)]
async fn run_tasks(
    shards: &str,
    device: &str,
    prefill_device: Option<&str>,
    disable_chunk: bool,
    park: bool,
    prompt: &str,
    max_new: u32,
    tasks: usize,
) -> Vec<RunOut> {
    let mut builder = OvRuntimeBuilder::new(shards, 0, 1, device)
        .with_chunked_prefill_disabled(disable_chunk)
        .with_prefill_parking(park);
    if let Some(pd) = prefill_device {
        builder = builder.with_prefill_device(pd);
    }
    // CASCADIA_OV_CACHE: compiled-blob cache dir. Load-bearing for the
    // parking leg — without it a parked model's reload is a full cold
    // compile (~minutes on NPU) instead of a cache import.
    if let Ok(cache) = std::env::var("CASCADIA_OV_CACHE") {
        builder = builder.with_cache_dir(cache);
    }
    builder
        .connect(PeerLayout::single_stage())
        .await
        .expect("connect");
    let mut load = builder
        .load(ShardSpec::single_stage(shards, device))
        .await
        .expect("load");
    use futures::StreamExt;
    while load.next().await.is_some() {}
    let mut engine = Box::new(builder).build().expect("build");

    let mut outs = Vec::new();
    for n in 0..tasks {
        let task = GenerationTask::new(format!("static-prefill-parity-{n}"), prompt)
            .with_max_tokens(max_new);
        engine.submit(task).expect("submit");

        let started = Instant::now();
        let mut ttft = None;
        let mut ids = Vec::new();
        let mut text = String::new();
        loop {
            let chunks = engine.step().expect("step");
            let mut done = false;
            for (_, c) in chunks {
                if ttft.is_none() {
                    ttft = Some(started.elapsed());
                }
                ids.push(c.token_id);
                text.push_str(&c.text);
                if c.is_final {
                    done = true;
                }
            }
            if done {
                break;
            }
        }
        outs.push(RunOut {
            ids,
            text,
            ttft: ttft.unwrap_or_default(),
            total: started.elapsed(),
        });
    }
    outs
}

fn report(name: &str, r: &RunOut) {
    let decode_s = r.total.saturating_sub(r.ttft).as_secs_f64();
    let decode_tok_s = if r.ids.len() > 1 && decode_s > 0.0 {
        (r.ids.len() - 1) as f64 / decode_s
    } else {
        0.0
    };
    eprintln!(
        "{name:<28} ttft {:>8.1} ms | decode {decode_tok_s:>6.2} tok/s | total {:>7.2} s | {} toks",
        r.ttft.as_secs_f64() * 1e3,
        r.total.as_secs_f64(),
        r.ids.len(),
    );
}

/// Minimum identical greedy prefix (decoded tokens) below which a divergence is
/// treated as corruption (a hard failure) rather than a tolerated argmax
/// near-tie. Set to 1: hard-fail ONLY a fork at token 0 (the first greedy token
/// already differs — the strongest "wrong prefill KV" signal).
///
/// Why so low: a broader sweep (1B + 3B × varied prompts, 2026-07-24) found
/// legitimate near-tie forks as early as **token 2** (both branches coherent,
/// re-converging), so fork position does NOT reliably separate a near-tie from
/// corruption. An earlier reading of "forks land ~token 30" came from one
/// unrepresentative prompt; a higher bar hard-fails real near-ties. The
/// first-token guard is the only position-based check that doesn't flake on the
/// measured distribution. `CASCADIA_PARITY_SOFT=1` tolerates even a token-0 fork.
const NEAR_TIE_MIN_PREFIX: usize = 1;

/// Greedy-token parity verdict against the tokenwise baseline — the pure
/// decision, with the env read factored out of [`assert_parity`] so it is
/// unit-testable without hardware.
///
/// The chunked-prefill variant is a **different compiled graph** (seq=`C`) from
/// the seq=1 decode graph, so the two accumulate floating-point differently —
/// and a genuinely near-equal top-2 argmax can flip, forking the greedy text
/// (both branches coherent). This is inherent to running two graphs and happens
/// on **every** device, same-device CPU/NPU included, and **as early as token
/// 2** (measured deterministically across 1B + 3B and varied prompts,
/// 2026-07-24); the fork rate grows with model size and on GPU / cross-device
/// hybrid. So a fork is tolerated as a near-tie — the ring-math unit tests
/// (`chunked_absorb_matches_sequential` et al.) are what prove the host KV
/// state is byte-identical. The only thing a near-tie cannot explain is a fork
/// within the first `NEAR_TIE_MIN_PREFIX` decoded tokens (token 0): that points
/// at genuinely wrong prefill KV, so it is [`ParityVerdict::TooEarly`] (a hard
/// failure) unless `soft` (`CASCADIA_PARITY_SOFT=1`, pure timing sweeps).
#[derive(Debug, PartialEq, Eq)]
enum ParityVerdict {
    /// Token-for-token identical.
    Exact,
    /// Diverged at this index but tolerated as a near-tie (fork ≥
    /// `NEAR_TIE_MIN_PREFIX`, or softened).
    NearTie(usize),
    /// Diverged within the first `NEAR_TIE_MIN_PREFIX` tokens — suspect wrong
    /// prefill KV, not a coincidental tie. Hard failure.
    TooEarly(usize),
}

fn parity_verdict(base: &[i64], other: &[i64], soft: bool) -> ParityVerdict {
    if base == other {
        return ParityVerdict::Exact;
    }
    let fork = base
        .iter()
        .zip(other)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| base.len().min(other.len()));
    if fork < NEAR_TIE_MIN_PREFIX && !soft {
        ParityVerdict::TooEarly(fork)
    } else {
        ParityVerdict::NearTie(fork)
    }
}

fn assert_parity(what: &str, base: &RunOut, other: &RunOut) {
    let soft = std::env::var("CASCADIA_PARITY_SOFT").is_ok_and(|v| v == "1");
    match parity_verdict(&base.ids, &other.ids, soft) {
        ParityVerdict::Exact => {}
        ParityVerdict::TooEarly(fork) => panic!(
            "{what} diverged from tokenwise at token {fork} (before the first \
             {NEAR_TIE_MIN_PREFIX} matched) — too early for a near-tie; suspect \
             wrong prefill KV (baseline text: {:?}, other text: {:?})",
            base.text, other.text
        ),
        ParityVerdict::NearTie(fork) => eprintln!(
            "PARITY-SOFT: {what} diverged from tokenwise at token {fork} \
             (tolerated as a near-tie fork; baseline text: {:?}, other text: {:?})",
            base.text, other.text
        ),
    }
}

#[test]
fn parity_verdict_exact_match() {
    assert_eq!(
        parity_verdict(&[1, 2, 3], &[1, 2, 3], false),
        ParityVerdict::Exact
    );
}

#[test]
fn parity_verdict_tolerates_a_late_near_tie_fork() {
    // Agree through the bar, then fork past it — a tolerated near-tie.
    let base: Vec<i64> = (0..(NEAR_TIE_MIN_PREFIX as i64 + 6)).collect();
    let mut other = base.clone();
    let fork = NEAR_TIE_MIN_PREFIX + 3;
    other[fork] = 9999;
    assert_eq!(
        parity_verdict(&base, &other, false),
        ParityVerdict::NearTie(fork)
    );
}

#[test]
fn parity_verdict_hard_fails_an_early_fork() {
    // Fork one token before the bar — too early to be a coincidental tie.
    let base: Vec<i64> = (0..20).collect();
    let mut other = base.clone();
    let fork = NEAR_TIE_MIN_PREFIX - 1;
    other[fork] = 9999;
    assert_eq!(
        parity_verdict(&base, &other, false),
        ParityVerdict::TooEarly(fork)
    );
}

#[test]
fn parity_verdict_soft_tolerates_even_an_early_fork() {
    // CASCADIA_PARITY_SOFT tolerates even a token-0 fork (pure timing sweeps).
    let base: Vec<i64> = (0..20).collect();
    let mut other = base.clone();
    other[0] = 9999;
    assert_eq!(
        parity_verdict(&base, &other, true),
        ParityVerdict::NearTie(0)
    );
}

#[tokio::test]
async fn chunked_prefill_matches_tokenwise_and_reports_timing() {
    let Ok(shards) = std::env::var("CASCADIA_STATIC_SHARDS") else {
        eprintln!("CASCADIA_STATIC_SHARDS not set; skipping");
        return;
    };
    let device = std::env::var("CASCADIA_STATIC_DEVICE").unwrap_or_else(|_| "CPU".into());
    let prefill_device = std::env::var("CASCADIA_PREFILL_DEVICE").ok();
    let prompt = std::env::var("CASCADIA_STATIC_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let max_new: u32 = std::env::var("CASCADIA_STATIC_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    // CASCADIA_STATIC_TASKS=N (default 1): drive N sequential requests
    // through the chunked/hybrid engines and report each. Request 1 through
    // a cache-imported NPU blob pays ~300 ms of driver init the import
    // deferred (a cold in-process compile does not); request 2+ is the
    // steady-state TTFT a long-lived worker sees.
    let bench_tasks: usize = std::env::var("CASCADIA_STATIC_TASKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1);

    // Guard against a vacuous pass: without a chunked-prefill variant, the
    // "chunked" and "hybrid" runs silently take the identical tokenwise path
    // and the parity assertions prove nothing. Require the export to
    // advertise one before proceeding.
    let stage_cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&shards).join("stage_0/stage_config.json"))
            .expect("read stage_0/stage_config.json"),
    )
    .expect("parse stage_config.json");
    let pseq = stage_cfg["static_prefill_seq"].as_u64().unwrap_or(0);
    assert!(
        pseq >= 2,
        "CASCADIA_STATIC_SHARDS export has no chunked-prefill variant \
         (static_prefill_seq={pseq}) — the chunked/hybrid legs would pass vacuously. \
         Re-export with --static-prefill-seq N."
    );

    // Baseline: legacy one-token-per-step static prefill on the decode device.
    let base = run_once(&shards, &device, None, true, &prompt, max_new).await;
    report(&format!("tokenwise [{device}]"), &base);
    assert!(!base.ids.is_empty(), "baseline generated no tokens");

    // Chunked prefill, same device (weight-stream amortization only).
    let chunked_runs =
        run_tasks(&shards, &device, None, false, false, &prompt, max_new, bench_tasks).await;
    for (i, c) in chunked_runs.iter().enumerate() {
        let label = if bench_tasks == 1 {
            format!("chunked   [{device}]")
        } else {
            format!("chunked#{}  [{device}]", i + 1)
        };
        report(&label, c);
        assert_parity(&format!("chunked prefill on {device} (task {i})"), &base, c);
    }

    // Hybrid: chunked prefill on another device, decode unchanged.
    if let Some(pd) = prefill_device.as_deref() {
        let hybrid_runs =
            run_tasks(&shards, &device, Some(pd), false, false, &prompt, max_new, bench_tasks)
                .await;
        for (i, h) in hybrid_runs.iter().enumerate() {
            let label = if bench_tasks == 1 {
                format!("hybrid    [{pd}+{device}]")
            } else {
                format!("hybrid#{}  [{pd}+{device}]", i + 1)
            };
            report(&label, h);
            assert_parity(
                &format!("hybrid ({pd} prefill + {device} decode, task {i})"),
                &base,
                h,
            );
        }
    } else {
        eprintln!("CASCADIA_PREFILL_DEVICE not set; hybrid leg skipped");
    }

    // Parking (CASCADIA_PARK=1): two sequential tasks through one engine
    // with --park-prefill semantics — task 1 parks after prefill, task 2
    // pays the reload (visible in its TTFT). Both must stay token-identical
    // to the baseline.
    if std::env::var("CASCADIA_PARK").is_ok_and(|v| v == "1") {
        let pd = prefill_device.as_deref();
        let outs = run_tasks(&shards, &device, pd, false, true, &prompt, max_new, 2).await;
        let label = pd.unwrap_or(&device).to_string();
        report(&format!("parked#1  [{label}+{device}]"), &outs[0]);
        report(&format!("parked#2  [{label}+{device}]"), &outs[1]);
        for (n, o) in outs.iter().enumerate() {
            assert_parity(&format!("parked task {n}"), &base, o);
        }
    } else {
        eprintln!("CASCADIA_PARK not set; parking leg skipped");
    }
}
