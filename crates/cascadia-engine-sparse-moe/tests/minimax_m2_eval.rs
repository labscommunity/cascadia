//! End-to-end correctness gate for the MiniMax-M2 OV-IR backend.
//!
//! Gated on `M2_MODEL_DIR` pointing at an export produced by
//! `tools/export_minimax_m2.py --tiny --no-quant` (which also writes a
//! `reference.json` containing the canonical HF greedy output). The test
//! runs the exact OV graphs through [`OvMoeRunner`] and asserts the greedy
//! token stream matches the reference — i.e. the Rust runtime reproduces
//! the PyTorch model. Skips (passes) when the env var is unset, so the
//! suite stays green on machines without OpenVINO or the fixture.
//!
//! Run on the miner:
//!   M2_MODEL_DIR=/media/tatef/extssd/m2/tiny_fp32 \
//!     cargo test -p cascadia-engine-sparse-moe --test minimax_m2_eval -- --nocapture

use std::path::PathBuf;

use cascadia_engine_sparse_moe::OvMoeRunner;
use cascadia_ov_genai_shim::PluginConfig;

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
fn minimax_m2_tiny_matches_hf_reference() {
    let Some(model_dir) = env_dir("M2_MODEL_DIR") else {
        eprintln!("M2_MODEL_DIR not set / missing; skipping MiniMax-M2 e2e test");
        return;
    };

    let ref_path = model_dir.join("reference.json");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_path).expect("read reference.json"))
            .expect("parse reference.json");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let greedy: Vec<u32> = reference["greedy_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let first_next = reference["first_next_token"].as_u64().unwrap() as u32;

    let device = std::env::var("CASCADIA_DEVICE").unwrap_or_else(|_| "CPU".to_string());
    let mut runner =
        OvMoeRunner::load(model_dir, &device, PluginConfig::new(), None).expect("load OvMoeRunner");

    let n_new = greedy.len() - prompt.len();
    let generated = runner
        .generate_argmax(&prompt, n_new)
        .expect("generate_argmax");

    let mut full = prompt.clone();
    full.extend_from_slice(&generated);
    eprintln!("reference greedy: {greedy:?}");
    eprintln!("ours     greedy : {full:?}");

    assert_eq!(
        generated.first().copied(),
        Some(first_next),
        "first generated token must match the HF reference"
    );
    assert_eq!(
        full, greedy,
        "full greedy token stream must match HF reference"
    );
}
