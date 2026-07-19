//! Cache-warming probe (hardware-gated; skip-pass without env).
//!
//! Compiles the given IR(s) one at a time in this single process, populating
//! the OV blob cache (`CASCADIA_OV_CACHE`). Ops tool for multi-model /
//! multi-stage NPU deployments on small-RAM boxes: the NPU compiler's host
//! transient (~5.5× a model's INT4 bytes) is paid strictly SEQUENTIALLY here,
//! so a later pipeline bring-up that starts all stages concurrently only
//! ever imports cached blobs (~blob-size peak, no transient overlap).
//!
//! ```text
//! CASCADIA_WARM_XML="a.xml;b.xml" CASCADIA_WARM_DEVICE=NPU \
//! CASCADIA_OV_CACHE=C:\cache cargo test -p cascadia-engine-openvino \
//!   --features openvino --test compile_warm_probe -- --nocapture
//! ```

use std::time::Instant;

use cascadia_ov_genai_shim::{PluginConfig, Runtime};

#[test]
fn warms_blob_cache_sequentially() {
    let Ok(xmls) = std::env::var("CASCADIA_WARM_XML") else {
        eprintln!("CASCADIA_WARM_XML not set; skipping");
        return;
    };
    let device = std::env::var("CASCADIA_WARM_DEVICE").unwrap_or_else(|_| "NPU".into());
    let mut plugin = PluginConfig::default();
    if let Ok(cache) = std::env::var("CASCADIA_OV_CACHE") {
        plugin.entries.push(("CACHE_DIR".into(), cache));
    }

    // CASCADIA_WARM_INFER=1: after each compile, run ONE inference with the
    // request's default (zeroed) input tensors and time it — isolates
    // graph-specific first-inference pathologies (observed: a 70B
    // embed-carrying stage hangs its first infer on iGPU/NPU while smaller
    // stages run fine) without any pipeline machinery around them.
    let do_infer = std::env::var("CASCADIA_WARM_INFER").is_ok_and(|v| v == "1");

    for xml in xmls.split(';').filter(|s| !s.trim().is_empty()) {
        let t0 = Instant::now();
        match Runtime::compile(xml, &device, &plugin) {
            Ok(mut rt) => {
                eprintln!(
                    "WARM OK [{device}] {xml}: {:.0}s",
                    t0.elapsed().as_secs_f64()
                );
                if do_infer {
                    let ti = Instant::now();
                    match rt.infer() {
                        Ok(()) => eprintln!(
                            "INFER OK [{device}] {xml}: {:.1}s",
                            ti.elapsed().as_secs_f64()
                        ),
                        Err(e) => eprintln!("INFER FAILED [{device}] {xml}: {e}"),
                    }
                }
            }
            Err(e) => panic!("WARM FAILED [{device}] {xml}: {e}"),
        }
        // rt drops here — the transient and the compiled model are gone
        // before the next compile starts.
    }
}
