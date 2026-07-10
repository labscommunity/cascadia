//! Seeded stochastic (temperature > 0) parity: single-stage `Dsv4Runner::
//! generate` must produce the SAME tokens as the pipeline for the same seed.
//!
//! The pipeline advances the RNG exactly once per generated token — intermediate
//! prompt tokens ride as ForwardNoSample and never draw; only the last prompt
//! token seeds+samples. `generate` must match that discipline (sample once after
//! prefill, not once per prompt token), otherwise seeded temp>0 output diverges
//! between a 1-node and a multi-node deployment. Greedy argmax hides this (it
//! never touches the RNG), so the argmax wire tests can't catch it — hence this
//! temp>0 test.

use cascadia_engine_sparse_moe::dsv4::stage::Dsv4Runner;
use cascadia_engine_sparse_moe::sampling::{init_rng, sample};
use cascadia_engine_sparse_moe::SamplingConfig;
use std::path::PathBuf;

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export")
}

fn prompt_ids() -> Vec<u32> {
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(export_dir().join("reference.json")).unwrap(),
    )
    .unwrap();
    v["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

const MAX_SEQ: usize = 64;

/// A seeded, non-greedy config (full-softmax multinomial draw).
fn seeded_cfg() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.8,
        seed: Some(42),
        ..SamplingConfig::default()
    }
}

/// Drive a 2-stage split (layers [0,2) then [2,4)) token-by-token with the
/// engine's exact sampling discipline: intermediate prompt tokens forward with
/// no head/sample (ForwardNoSample), the last prompt token and every decode
/// token seed-once + sample-with-history on the last stage. Mirrors
/// `Dsv4Engine::step_first` + `handle_forward`.
fn pipeline_generate(
    r0: &mut Dsv4Runner,
    r1: &mut Dsv4Runner,
    prompt: &[u32],
    max_new: usize,
    cfg: &SamplingConfig,
    eos: &[u32],
) -> Vec<u32> {
    r0.reset();
    r1.reset();
    let mut rng = init_rng(cfg.seed);
    let n = prompt.len();
    // Prefill: forward every token; keep only the last stage's last logits.
    let mut last_logits: Vec<f32> = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        let h0 = r0.forward_layers(r0.embed_token(t), pos, Some(t));
        let h1 = r1.forward_layers(h0, pos, None);
        if pos + 1 == n {
            last_logits = r1.head_logits(&h1);
        }
    }
    let mut history: Vec<i64> = Vec::new();
    let mut next = sample(&last_logits, &history, cfg, &mut rng);
    let mut out: Vec<u32> = Vec::new();
    let mut pos = n;
    loop {
        let tok = next as u32;
        out.push(tok);
        history.push(next);
        if out.len() >= max_new || eos.contains(&tok) || pos >= MAX_SEQ {
            break;
        }
        let h0 = r0.forward_layers(r0.embed_token(tok), pos, Some(tok));
        let h1 = r1.forward_layers(h0, pos, None);
        next = sample(&r1.head_logits(&h1), &history, cfg, &mut rng);
        pos += 1;
    }
    out
}

#[test]
fn seeded_temp_single_stage_matches_pipeline() {
    let dir = export_dir();
    let prompt = prompt_ids();
    let cfg = seeded_cfg();
    let max_new = 8;

    // Single stage: all 4 layers in one runner.
    let mut single = Dsv4Runner::load_staged(&dir, MAX_SEQ, 0, 1, 0, 0).unwrap();
    let out_single = single.generate(&prompt, max_new, &cfg);

    // Pipeline: layers [0,2) (first) -> [2,4) (last).
    let mut r0 = Dsv4Runner::load_staged(&dir, MAX_SEQ, 0, 2, 0, 2).unwrap();
    let mut r1 = Dsv4Runner::load_staged(&dir, MAX_SEQ, 1, 2, 2, 4).unwrap();
    let eos = single.eos_token_ids().to_vec();
    let out_pipe = pipeline_generate(&mut r0, &mut r1, &prompt, max_new, &cfg, &eos);

    assert_eq!(
        out_single, out_pipe,
        "seeded temp>0 diverges single-stage vs pipeline (RNG draw count must match)"
    );
}

#[test]
fn seeded_temp_is_reproducible() {
    let dir = export_dir();
    let prompt = prompt_ids();
    let cfg = seeded_cfg();
    let mut r = Dsv4Runner::load_staged(&dir, MAX_SEQ, 0, 1, 0, 0).unwrap();
    let a = r.generate(&prompt, 8, &cfg);
    let b = r.generate(&prompt, 8, &cfg);
    assert_eq!(a, b, "same seed must give the same tokens");
}
