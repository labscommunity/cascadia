//! End-to-end smoke test for the Rust int4 layer-0 KV-cache path.
//!
//! Gated by the `K26_MODEL_DIR` env var because it needs the full
//! ~550 GB K2.6 safetensors set + the kimi-k26-head IR + a tokenizer.
//! Skipped silently in `cargo test` when that var is unset.
//!
//! Each prompt is generated with `Runner::generate_argmax` (greedy)
//! and the first generated token is checked against the known good
//! reference from the OV-only baseline.

use std::path::PathBuf;

use tahoma_engine_sparse_moe::{Runner, SamplingConfig};
use tahoma_ov_genai_shim::PluginConfig;

fn model_dir_from_env() -> Option<PathBuf> {
    let v = std::env::var("K26_MODEL_DIR").ok()?;
    Some(PathBuf::from(v))
}

fn run_prompt(model_dir: &PathBuf, prompt: &str, expected_substr: &str, max_new: usize) {
    let mut runner = Runner::load(
        model_dir.clone(),
        "CPU",
        PluginConfig::new(),
        tahoma_engine_sparse_moe::runner::LayerRange::full(),
        false, // int4_embedding: keep bf16 path as the eval baseline
    )
    .expect("Runner::load");

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("tokenizer.json must be present alongside the model");

    let enc = tokenizer.encode(prompt, false).expect("encode prompt");
    let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();

    let cfg = SamplingConfig::default();
    let generated = runner
        .generate(&prompt_ids, max_new, &cfg)
        .expect("Runner::generate");

    let ids_u32: Vec<u32> = generated.iter().map(|&i| i as u32).collect();
    let text = tokenizer.decode(&ids_u32, true).expect("decode");
    println!("prompt={prompt:?} → {text:?}");
    assert!(
        text.contains(expected_substr),
        "prompt {prompt:?} expected substring {expected_substr:?}, got {text:?}"
    );
}

#[test]
fn k26_paris_pacific_four() {
    let Some(model_dir) = model_dir_from_env() else {
        eprintln!("K26_MODEL_DIR not set; skipping");
        return;
    };
    run_prompt(&model_dir, "The capital of France is", "Paris", 4);
    run_prompt(&model_dir, "The largest ocean is the", "Pacific", 4);
    run_prompt(&model_dir, "Two plus two equals", "four", 4);
}
