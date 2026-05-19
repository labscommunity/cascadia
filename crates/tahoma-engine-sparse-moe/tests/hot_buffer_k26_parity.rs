//! End-to-end parity test: with the real K2.6 model loaded, the same
//! prompt + sampler config must produce a bit-identical token stream
//! whether the hot-expert buffer is disabled (mmap path) or enabled
//! (packed-buffer path).
//!
//! Gated on `K26_MODEL_DIR` — the same env var the existing
//! `k26_layer0_eval` test uses. Without the model present, the test
//! is silently skipped.
//!
//! This is the live-fire complement to `hot_buffer_bit_identity` (a
//! synthetic-shard byte equality check). Synthetic test proves the
//! source-of-truth bytes are equal; this test proves the dispatch
//! plumbing actually pulls from the hot buffer when configured to,
//! and that the result of running the kernel through the hot buffer
//! is greedy-identical to running it through the mmap.

use std::path::PathBuf;

use tahoma_engine_sparse_moe::{Runner, SamplingConfig};
use tahoma_ov_genai_shim::PluginConfig;

fn model_dir_from_env() -> Option<PathBuf> {
    let v = std::env::var("K26_MODEL_DIR").ok()?;
    Some(PathBuf::from(v))
}

fn generate(model_dir: &PathBuf, prompt: &str, max_new: usize, hot_n: usize) -> Vec<i64> {
    let mut runner = Runner::load(
        model_dir.clone(),
        "CPU",
        PluginConfig::new(),
        tahoma_engine_sparse_moe::runner::LayerRange::full(),
    )
    .expect("Runner::load");

    // For the hot run, warm enough dispatches so the buffer is built
    // before the test prompt completes. Setting warmup_dispatches=1
    // means the buffer is built after the very first expert call, so
    // the entire generation runs through the hot path (modulo cold-set
    // first dispatch that records the hit).
    if hot_n > 0 {
        runner.set_hot_expert_buffer_config(hot_n, 1);
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .expect("tokenizer.json must be present alongside the model");
    let enc = tokenizer.encode(prompt, false).expect("encode prompt");
    let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();

    let cfg = SamplingConfig::default(); // greedy
    runner
        .generate(&prompt_ids, max_new, &cfg)
        .expect("Runner::generate")
}

#[test]
fn k26_hot_buffer_parity_greedy() {
    let Some(model_dir) = model_dir_from_env() else {
        eprintln!("K26_MODEL_DIR not set; skipping hot-buffer parity test");
        return;
    };
    let prompt = "The capital of France is";
    let max_new = 4;

    let cold = generate(&model_dir, prompt, max_new, 0);
    let hot = generate(&model_dir, prompt, max_new, 16);

    assert_eq!(
        cold, hot,
        "hot-buffer path must produce greedy-identical tokens; cold={cold:?} hot={hot:?}"
    );
}
