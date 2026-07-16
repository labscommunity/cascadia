//! Greedy-parity + phase-timing gate for the chunked-prefill static path.
//!
//! Proves the load-bearing property of the hybrid NPU+CPU split: a run that
//! prefills through the chunked variant (optionally on a DIFFERENT device)
//! must produce token-for-token the same greedy output as the legacy
//! one-token-per-step static path — i.e. the shared host KV ring hands the
//! decode loop identical state no matter which device (or window width)
//! filled it. Also prints TTFT + decode tok/s per config, so on hardware it
//! doubles as the phase-split bench.
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
//! `CASCADIA_PARITY_SOFT=1` (report a token divergence instead of failing —
//! for GPU / cross-device perf sweeps where greedy near-ties legitimately
//! fork; see `assert_parity`).
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

/// Greedy-token parity against the tokenwise baseline. Same-device CPU and
/// NPU runs are byte-stable across the seq=1/seq=64 graph variants and stay
/// HARD-asserted. On GPU the two variants compile to different kernel /
/// accumulation choices, and a cross-device hybrid prefill hands decode
/// KV computed with another device's rounding — either can flip a genuine
/// argmax near-tie and fork the greedy text (both branches coherent; observed
/// fork rate grows with model size: 0 at 1B same-device CPU/NPU, ~half of
/// GPU/hybrid runs by 3B). `CASCADIA_PARITY_SOFT=1` downgrades the mismatch
/// to a loud report with the fork index so perf sweeps on those configs still
/// print every leg. Never set it for CPU/NPU validation runs; CI is
/// unaffected (no `CASCADIA_STATIC_SHARDS` there).
fn assert_parity(what: &str, base: &RunOut, other: &RunOut) {
    if base.ids == other.ids {
        return;
    }
    if std::env::var("CASCADIA_PARITY_SOFT").is_ok_and(|v| v == "1") {
        let fork = base
            .ids
            .iter()
            .zip(&other.ids)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| base.ids.len().min(other.ids.len()));
        eprintln!(
            "PARITY-SOFT: {what} diverged from tokenwise at token {fork} \
             (tolerated as a near-tie fork; baseline text: {:?}, other text: {:?})",
            base.text, other.text
        );
        return;
    }
    panic!(
        "{what} diverged from tokenwise \
         (baseline text: {:?}, other text: {:?})",
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
