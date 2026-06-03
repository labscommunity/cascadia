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

/// Real-model generation smoke test. Gated on `M2_GEN_DIR` pointing at a
/// full MiniMax-M2 export (with tokenizer.json). There's no exact HF
/// reference at 230B (won't fit in RAM), so this just confirms the Rust
/// runtime loads the real INT4 graphs, generates tokens through the
/// single-stage OV-IR pipeline, and decodes to non-empty text — the
/// "single-stage run on the miner" deliverable. Prompt overridable via
/// `M2_PROMPT`; token budget via `M2_MAX_NEW` (default 30).
#[test]
fn minimax_m2_full_generate_smoke() {
    use tokenizers::Tokenizer;

    let Some(model_dir) = env_dir("M2_GEN_DIR") else {
        eprintln!("M2_GEN_DIR not set / missing; skipping MiniMax-M2 full-model smoke test");
        return;
    };
    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path).expect("load tokenizer.json");
    let prompt = std::env::var("M2_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
    let max_new: usize = std::env::var("M2_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    eprintln!("prompt={prompt:?} prompt_ids={ids:?}");

    let device = std::env::var("CASCADIA_DEVICE").unwrap_or_else(|_| "CPU".to_string());
    let cap = std::env::var("CASCADIA_MAX_EXPERTS_CACHED")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .and_then(std::num::NonZeroUsize::new);
    let mut runner =
        OvMoeRunner::load(model_dir, &device, PluginConfig::new(), cap).expect("load OvMoeRunner");

    let started = std::time::Instant::now();
    let generated = runner
        .generate_argmax(&ids, max_new)
        .expect("generate_argmax");
    let secs = started.elapsed().as_secs_f64();
    let text = tokenizer.decode(&generated, true).unwrap_or_default();
    eprintln!(
        "generated {} tokens in {:.1}s ({:.2} tok/s)",
        generated.len(),
        secs,
        generated.len() as f64 / secs.max(1e-9)
    );
    eprintln!("completion: {text:?}");

    assert!(!generated.is_empty(), "model produced no tokens");
    assert!(!text.trim().is_empty(), "decoded completion is empty");
}
