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
    let mut builder =
        OvRuntimeBuilder::new(shards, 0, 1, device).with_chunked_prefill_disabled(disable_chunk);
    if let Some(pd) = prefill_device {
        builder = builder.with_prefill_device(pd);
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

    let task = GenerationTask::new("static-prefill-parity", prompt).with_max_tokens(max_new);
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
    RunOut {
        ids,
        text,
        ttft: ttft.unwrap_or_default(),
        total: started.elapsed(),
    }
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

/// Minimum identical greedy prefix (decoded tokens) required to treat a
/// divergence as a legitimate argmax near-tie rather than corruption. A single
/// near-tie can only flip *after* the prefill has produced this many correct
/// tokens; a fork earlier than this is too soon to be a coincidental tie and
/// points at wrong prefill KV. Checking only the first token is too weak — a
/// subtly-corrupted KV can emit a couple of plausible tokens before drifting.
/// Measured near-tie forks (2026-07-23, 1B) landed at token ~29–30, well clear
/// of this bar. Runs shorter than this treat any fork as suspect (intended).
const NEAR_TIE_MIN_PREFIX: usize = 10;

/// Greedy-token parity against the tokenwise baseline. The chunked-prefill
/// variant is a **different compiled graph** (seq=`C`) from the seq=1 decode
/// graph, so the two accumulate floating-point differently — and a genuinely
/// near-equal top-2 argmax can flip, forking the greedy text (both branches
/// coherent). This is inherent to running two graphs and happens on **every**
/// device, same-device CPU/NPU included: a 1B same-device CPU run forks
/// ~token 30, deterministically (measured 2026-07-23), and the fork rate
/// grows with model size and on GPU / cross-device hybrid. So a fork is
/// tolerated as a near-tie with a loud report — the ring-math unit tests
/// (`chunked_absorb_matches_sequential` et al.) are what prove the host KV
/// state is byte-identical. What a single near-tie CANNOT explain is a fork
/// within the first `NEAR_TIE_MIN_PREFIX` decoded tokens: that points at
/// genuinely wrong prefill KV, so it stays a hard failure.
/// `CASCADIA_PARITY_SOFT=1` tolerates even an early fork (pure timing sweeps).
/// CI is unaffected (no `CASCADIA_STATIC_SHARDS` there).
fn assert_parity(what: &str, base: &RunOut, other: &RunOut) {
    if base.ids == other.ids {
        return;
    }
    let fork = base
        .ids
        .iter()
        .zip(&other.ids)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| base.ids.len().min(other.ids.len()));
    let soft = std::env::var("CASCADIA_PARITY_SOFT").is_ok_and(|v| v == "1");
    // A fork within the first NEAR_TIE_MIN_PREFIX tokens is too early to be a
    // coincidental argmax near-tie — the prefill likely handed decode wrong KV.
    // Keep it fatal unless explicitly softened.
    if fork < NEAR_TIE_MIN_PREFIX && !soft {
        panic!(
            "{what} diverged from tokenwise at token {fork} (before the first \
             {NEAR_TIE_MIN_PREFIX} matched) — too early for a near-tie; suspect \
             wrong prefill KV (baseline text: {:?}, other text: {:?})",
            base.text, other.text
        );
    }
    eprintln!(
        "PARITY-SOFT: {what} diverged from tokenwise at token {fork} \
         (tolerated as a near-tie fork; baseline text: {:?}, other text: {:?})",
        base.text, other.text
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
    let chunked = run_once(&shards, &device, None, false, &prompt, max_new).await;
    report(&format!("chunked   [{device}]"), &chunked);
    assert_parity(&format!("chunked prefill on {device}"), &base, &chunked);

    // Hybrid: chunked prefill on another device, decode unchanged.
    if let Some(pd) = prefill_device.as_deref() {
        let hybrid = run_once(&shards, &device, Some(pd), false, &prompt, max_new).await;
        report(&format!("hybrid    [{pd}+{device}]"), &hybrid);
        assert_parity(
            &format!("hybrid ({pd} prefill + {device} decode)"),
            &base,
            &hybrid,
        );
    } else {
        eprintln!("CASCADIA_PREFILL_DEVICE not set; hybrid leg skipped");
    }
}
