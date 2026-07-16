//! AOT blob-import probe (hardware-gated; skip-pass without env).
//!
//! Validates the escape hatch for the NPU compile-memory spike: a blob
//! produced by `ov::CompiledModel::export_model` on a big-RAM host (offline
//! cross-compile via `NPU_PLATFORM`) is imported here WITHOUT running the
//! compiler — the ~5.5×-INT4-bytes host transient never happens on this box.
//! Success = the driver accepted the blob (metadata/platform handshake) and
//! graph init + device weight load completed (the shim creates the infer
//! request eagerly).
//!
//! ```text
//! CASCADIA_BLOB_IMPORT=C:\path\to\model.blob [CASCADIA_BLOB_DEVICE=NPU] \
//!   cargo test -p cascadia-engine-openvino --features openvino \
//!   --test blob_import_probe -- --nocapture
//! ```

use std::time::Instant;

use cascadia_ov_genai_shim::{PluginConfig, Runtime};

#[test]
fn imports_aot_blob_without_compiling() {
    let Ok(blob) = std::env::var("CASCADIA_BLOB_IMPORT") else {
        eprintln!("CASCADIA_BLOB_IMPORT not set; skipping");
        return;
    };
    let device = std::env::var("CASCADIA_BLOB_DEVICE").unwrap_or_else(|_| "NPU".into());
    let plugin = PluginConfig::default();

    let t0 = Instant::now();
    match Runtime::import_blob(&blob, &device, &plugin) {
        Ok(rt) => {
            let secs = t0.elapsed().as_secs_f64();
            let inputs = rt.input_names().map(|n| n.len()).unwrap_or(0);
            eprintln!(
                "BLOB-IMPORT OK [{device}] {blob}: {secs:.1}s, {inputs} inputs \
                 (graph initialized; no compiler ran on this host)"
            );
        }
        Err(e) => panic!("BLOB-IMPORT FAILED [{device}] {blob}: {e}"),
    }
}
