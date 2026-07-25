//! EXPERIMENT probe: resident-memory delta of the CascadiaInt4Gemv decode
//! offload. Loads ONLY the decode model (chunked prefill disabled) on CPU,
//! generates a few tokens, then holds so an external wrapper can sample the
//! process — the metric is PRIVATE bytes: the stock path's repacked weight
//! copy is private commit, the offloaded path's weights are shareable
//! mmapped file pages of the IR .bin.
//!
//! Env-gated (skip=pass): CASCADIA_GEMV_PROBE=1 + CASCADIA_STATIC_SHARDS;
//! CASCADIA_GEMV=1 enables the offload (run once without for baseline);
//! CASCADIA_PROBE_HOLD_S (default 12).

use cascadia_engine::Builder;
use cascadia_engine_openvino::OvRuntimeBuilder;
use cascadia_types::{GenerationTask, PeerLayout, ShardSpec};

#[tokio::test]
async fn gemv_residency_probe() {
    if std::env::var("CASCADIA_GEMV_PROBE").ok().as_deref() != Some("1") {
        eprintln!("CASCADIA_GEMV_PROBE not set; skipping");
        return;
    }
    let Ok(shards) = std::env::var("CASCADIA_STATIC_SHARDS") else {
        eprintln!("CASCADIA_STATIC_SHARDS not set; skipping");
        return;
    };
    let offload = std::env::var("CASCADIA_GEMV").ok().as_deref() == Some("1");
    let hold_s: u64 = std::env::var("CASCADIA_PROBE_HOLD_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    eprintln!(
        "PROBE-MODE {}",
        if offload { "gemv-offload" } else { "stock" }
    );

    // Both modes disable the chunked-prefill variant so exactly ONE compiled
    // model (decode) is resident — the delta between modes is the decode
    // weight copy alone.
    let mut builder = OvRuntimeBuilder::new(&shards, 0, 1, "CPU")
        .with_chunked_prefill_disabled(true)
        .with_gemv_offload(offload);
    builder
        .connect(PeerLayout::single_stage())
        .await
        .expect("connect");
    let mut load = builder
        .load(ShardSpec::single_stage(&shards, "CPU"))
        .await
        .expect("load");
    use futures::StreamExt;
    while load.next().await.is_some() {}
    let mut engine = Box::new(builder).build().expect("build");

    let task = GenerationTask::new("gemv-probe", "The capital of France is").with_max_tokens(8);
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
    eprintln!("PROBE-HOLD {hold_s}s (sample private bytes + WS now)");
    std::thread::sleep(std::time::Duration::from_secs(hold_s));
    eprintln!("PROBE-RESULT ok");
}
