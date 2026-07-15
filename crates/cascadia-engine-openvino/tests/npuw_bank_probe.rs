//! EXPERIMENT probe: can two user-level NPU compilations share one weight
//! allocation through the (undocumented) NPUW weights bank?
//!
//! Background: NPUW shares a single weight bank across ITS internal
//! prefill/generate/head submodels via a process-global bank keyed by
//! `NPUW_WEIGHTS_BANK`. If two independent `compile_model` calls of sibling
//! IRs (our decode + chunked-prefill variants, byte-identical weights) can
//! join one bank, an all-NPU phase split would hold ~1x weights instead of
//! 2x. This probe measures exactly that, nothing more — it is NOT a
//! supported configuration (see experiments/npuw-bank-probe/NOTES.md).
//!
//! Env-gated (skip=pass): CASCADIA_NPUW_PROBE=1 + CASCADIA_STATIC_SHARDS
//! (single-stage static export WITH a prefill variant). CASCADIA_NPUW=1 adds
//! the NPUW properties (run once without for the baseline). The process
//! prints PROBE-HOLD and sleeps CASCADIA_PROBE_HOLD_S (default 12) after
//! generating, so an external wrapper can sample its working set:
//!
//! ```text
//! CASCADIA_NPUW_PROBE=1 [CASCADIA_NPUW=1] CASCADIA_STATIC_SHARDS=<dir> \
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!   --test npuw_bank_probe --release -- --nocapture
//! ```

use cascadia_engine::Builder;
use cascadia_engine_openvino::OvRuntimeBuilder;
use cascadia_types::{GenerationTask, PeerLayout, ShardSpec};

#[tokio::test]
async fn npuw_bank_probe() {
    if std::env::var("CASCADIA_NPUW_PROBE").ok().as_deref() != Some("1") {
        eprintln!("CASCADIA_NPUW_PROBE not set; skipping");
        return;
    }
    let Ok(shards) = std::env::var("CASCADIA_STATIC_SHARDS") else {
        eprintln!("CASCADIA_STATIC_SHARDS not set; skipping");
        return;
    };
    let with_npuw = std::env::var("CASCADIA_NPUW").ok().as_deref() == Some("1");
    let hold_s: u64 = std::env::var("CASCADIA_PROBE_HOLD_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    // All-NPU: decode AND prefill variants both compile on the NPU — the
    // configuration whose 2x weight residency the bank would deduplicate.
    let mut builder = OvRuntimeBuilder::new(&shards, 0, 1, "NPU").with_prefill_device("NPU");
    if with_npuw {
        builder = builder.with_ov_properties(vec![
            ("NPU_USE_NPUW".into(), "YES".into()),
            ("NPUW_WEIGHTS_BANK".into(), "cascadia-bank-probe".into()),
        ]);
        eprintln!("PROBE-MODE npuw-bank");
    } else {
        eprintln!("PROBE-MODE baseline");
    }

    builder
        .connect(PeerLayout::single_stage())
        .await
        .expect("connect");
    let load = builder.load(ShardSpec::single_stage(&shards, "NPU")).await;
    let mut load = match load {
        Ok(l) => l,
        Err(e) => {
            // A compile failure IS a probe result (NPUW may reject already-
            // sharded stateless graphs) — report and end without failing the
            // suite, so the wrapper can record the outcome.
            eprintln!("PROBE-RESULT load-failed: {e}");
            return;
        }
    };
    use futures::StreamExt;
    while load.next().await.is_some() {}
    let mut engine = match Box::new(builder).build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("PROBE-RESULT build-failed: {e}");
            return;
        }
    };

    // Sanity generation so the probe also reports whether NPUW-compiled
    // variants still produce sane output through the ring.
    let task = GenerationTask::new("npuw-probe", "The capital of France is").with_max_tokens(8);
    engine.submit(task).expect("submit");
    let mut text = String::new();
    loop {
        let chunks = engine.step().expect("step");
        let mut done = false;
        for (_, c) in chunks {
            text.push_str(&c.text);
            if c.is_final {
                done = true;
            }
        }
        if done {
            break;
        }
    }
    eprintln!("PROBE-GEN {text:?}");
    eprintln!("PROBE-HOLD {hold_s}s (sample this process's working set now)");
    std::thread::sleep(std::time::Duration::from_secs(hold_s));
    eprintln!("PROBE-RESULT ok");
}
