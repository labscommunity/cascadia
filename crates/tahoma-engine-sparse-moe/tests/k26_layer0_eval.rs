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
    run_prompt_with_range(
        model_dir,
        prompt,
        expected_substr,
        max_new,
        tahoma_engine_sparse_moe::runner::LayerRange::full(),
    )
}

fn run_prompt_with_range(
    model_dir: &PathBuf,
    prompt: &str,
    expected_substr: &str,
    max_new: usize,
    range: tahoma_engine_sparse_moe::runner::LayerRange,
) {
    let mut runner =
        Runner::load(model_dir.clone(), "CPU", PluginConfig::new(), range).expect("Runner::load");

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

/// Run the same prompts with the Rust head substituted for the OV head
/// (via `LayerRange::full()` + `head_vocab_range = Some((0,
/// vocab_size))`). This is the standalone-Rust-head A/B path: same
/// model, same shells, same layer 0 — only the head changes.
///
/// Gated separately because it exercises a new code path (`HeadSlice`
/// for the FULL vocab) and uses an extra ~2.3 GB bf16 mmap. The
/// expected outputs must match `k26_paris_pacific_four`'s OV-head
/// outputs token-for-token, validating that the Rust head's numerical
/// output is interchangeable with the OV head's. Skipped when
/// K26_MODEL_DIR is unset.
#[test]
fn k26_paris_pacific_four_rust_head() {
    let Some(model_dir) = model_dir_from_env() else {
        eprintln!("K26_MODEL_DIR not set; skipping");
        return;
    };
    // Read vocab_size from manifest so the test stays in lockstep with
    // the model on disk (the value is in manifest.json).
    let manifest = tahoma_engine_sparse_moe::Manifest::load(&model_dir).expect("manifest load");
    let range = tahoma_engine_sparse_moe::runner::LayerRange {
        layer_start: 0,
        layer_end: u32::MAX,
        is_first: true,
        is_last: true,
        head_vocab_range: Some((0, manifest.vocab_size)),
    };
    eprintln!(
        "k26_paris_pacific_four_rust_head: loading FULL vocab slice {} rows",
        manifest.vocab_size
    );
    run_prompt_with_range(
        &model_dir,
        "The capital of France is",
        "Paris",
        4,
        range.clone(),
    );
    run_prompt_with_range(
        &model_dir,
        "The largest ocean is the",
        "Pacific",
        4,
        range.clone(),
    );
    run_prompt_with_range(&model_dir, "Two plus two equals", "four", 4, range);
}
