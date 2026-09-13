//! Complete autoregressive Inkling decode, separate from the synthetic layer probe.
//!
//! --export DIR --cases cases.json [--tokens 64] [--samples 3] [--out result.json]
//! cases.json: [{"name":"case", "prompt_ids":[...], "greedy_ids":[...]}].
//! Omit greedy_ids only when recording an initial baseline (not correctness-verified).
//! The large 975B architecture is required unless --allow-fixture is explicit.
//! Fixture runs emit fixture_decode_tokens_per_s and can never satisfy the target.
//! Generation stops at EOS. The first generated token belongs to prefill; decode
//! throughput counts only subsequent tokens, with complete layers/head/argmax.

use std::path::PathBuf;
use std::time::Instant;

use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::inkling::loader::{load_model_with, read_manifest};
use cascadia_engine_sparse_moe::inkling::model::{argmax, Model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case {
    name: String,
    prompt_ids: Vec<u32>,
    #[serde(default)]
    greedy_ids: Option<Vec<u32>>,
}

#[derive(Serialize)]
struct Sample {
    case: String,
    repetition: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    decode_steps: usize,
    generated_ids: Vec<u32>,
}

fn hash_logits(hash: &mut u64, logits: &[f32]) {
    for value in logits {
        assert!(value.is_finite(), "non-finite logits");
        for byte in value.to_bits().to_le_bytes() {
            *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
}

fn generate(model: &mut Model, case: &Case, tokens: usize, eos: &[u32], hash: &mut u64) -> Sample {
    model.reset();
    let start = Instant::now();
    let logits = model.prefill(&case.prompt_ids);
    let mut next = argmax(&logits) as u32;
    let prefill_seconds = start.elapsed().as_secs_f64();
    hash_logits(hash, &logits);
    let mut generated_ids = vec![next];
    let start = Instant::now();
    while generated_ids.len() < tokens && !eos.contains(&next) {
        let logits = model.forward_token(next);
        hash_logits(hash, &logits);
        next = argmax(&logits) as u32;
        generated_ids.push(next);
    }
    let decode_seconds = start.elapsed().as_secs_f64();
    if let Some(expected) = &case.greedy_ids {
        assert_eq!(&generated_ids, expected, "greedy mismatch: {}", case.name);
    }
    Sample {
        case: case.name.clone(),
        repetition: 0,
        prefill_seconds,
        decode_seconds,
        decode_steps: generated_ids.len() - 1,
        generated_ids,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut export = None;
    let mut cases_path = None;
    let mut out = None;
    let mut tokens = 64usize;
    let mut repetitions = 3usize;
    let mut allow_fixture = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--allow-fixture" {
            allow_fixture = true;
            continue;
        }
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--export" => export = Some(PathBuf::from(value)),
            "--cases" => cases_path = Some(PathBuf::from(value)),
            "--tokens" => tokens = value.parse()?,
            "--samples" => repetitions = value.parse()?,
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    assert!(
        tokens >= 2 && repetitions >= 1,
        "need >=2 tokens and >=1 samples"
    );
    let export = export.ok_or("--export is required")?;
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(cases_path.ok_or("--cases is required")?)?)?;
    assert!(!cases.is_empty(), "need at least one prompt");
    let manifest = read_manifest(&export)?;
    let large = manifest.num_layers == 66
        && manifest.hidden_size == 6144
        && manifest.vocab_size == 201024
        && manifest.num_attention_heads == 64
        && manifest.num_kv_heads == 8
        && manifest.head_dim == 128
        && manifest.dense_layers == [0, 1]
        && manifest.dense_intermediate == 24576
        && manifest.moe_intermediate == 3072
        && manifest.num_experts == 256
        && manifest.top_k == 6
        && manifest.n_shared_experts == 2;
    assert!(
        large || allow_fixture,
        "expected the complete large Inkling architecture"
    );
    let full_model = large && !allow_fixture;
    let correctness_verified = cases.iter().all(|c| c.greedy_ids.is_some());
    for c in &cases {
        assert!(!c.prompt_ids.is_empty(), "empty prompt: {}", c.name);
        assert!(c
            .prompt_ids
            .iter()
            .all(|&t| (t as usize) < manifest.vocab_size));
    }
    let max_seq = cases
        .iter()
        .map(|c| c.prompt_ids.len())
        .max()
        .unwrap()
        .checked_add(tokens)
        .ok_or("sequence length overflow")?;
    let load = Instant::now();
    let mut model = load_model_with(&export, max_seq, ExpertsMode::Mmap)?;
    assert_eq!(model.layers().len(), manifest.num_layers);
    println!("load_seconds={}", load.elapsed().as_secs_f64());
    println!("full_model={}", u8::from(full_model));
    let mut samples = Vec::new();
    let mut reference_hash = None;
    for rep in 0..repetitions {
        let mut hash = 0xcbf29ce484222325;
        for case in &cases {
            let mut sample = generate(&mut model, case, tokens, &manifest.eos_token_ids, &mut hash);
            sample.repetition = rep;
            assert!(
                sample.decode_steps > 0,
                "{} ended during prefill; choose a longer prompt",
                case.name
            );
            println!("sample_json={}", serde_json::to_string(&sample)?);
            samples.push(sample);
        }
        if let Some(expected) = reference_hash {
            assert_eq!(hash, expected, "non-repeatable logits");
        } else {
            reference_hash = Some(hash);
        }
    }
    // Conservative primary metric: slowest complete case/repetition, including
    // the first decode and all validation/hash overhead. No warm-cache exclusions.
    let rate = samples
        .iter()
        .map(|s| s.decode_steps as f64 / s.decode_seconds)
        .fold(f64::INFINITY, f64::min);
    let steps = samples.iter().map(|s| s.decode_steps).min().unwrap();
    let hash = format!("{:016x}", reference_hash.unwrap());
    let scope = if full_model {
        "full_large_model_decode"
    } else {
        "fixture_model_decode"
    };
    println!("scope={scope}");
    println!("output_hash={hash}");
    println!("correctness_verified={}", u8::from(correctness_verified));
    println!("decode_steps_min={steps}");
    println!("repetitions={repetitions}");
    let metric = if full_model {
        "decode_tokens_per_s"
    } else {
        "fixture_decode_tokens_per_s"
    };
    println!("{metric}={rate}");
    if let Some(out) = out {
        std::fs::write(
            out,
            serde_json::to_vec_pretty(&serde_json::json!({
                "scope": scope, "export": export, "manifest": {"layers":manifest.num_layers,
                    "hidden":manifest.hidden_size, "experts":manifest.num_experts},
                "output_hash":hash, "correctness_verified":correctness_verified,
                "slowest_case_decode_tokens_per_s":rate, "samples":samples
            }))?,
        )?;
    }
    Ok(())
}
