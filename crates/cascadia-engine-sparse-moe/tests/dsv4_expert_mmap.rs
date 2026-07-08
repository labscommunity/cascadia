//! Mmap'd int4 experts must be BIT-IDENTICAL to the eager f32 path — same
//! nibble decode, same accumulation order — so the exact-greedy guarantees
//! carry over to the production (mmap) configuration unchanged.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::loader::{load_model, load_stage_mode, ExpertsMode};

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export")
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Every expert of every layer: eager forward == mmap forward, bitwise.
#[test]
fn mmap_expert_forward_bitwise_matches_eager() {
    let dir = export_dir();
    let eager = load_model(&dir, 64).expect("eager load");
    let mmap = load_stage_mode(
        &dir,
        64,
        0,
        eager.layers.len(),
        true,
        true,
        ExpertsMode::Mmap,
    )
    .expect("mmap load");
    let dim = eager.cfg.dim;
    // deterministic bf16-representable probe (matches runtime activations,
    // which are bf16-rounded before the FFN)
    let x: Vec<f32> = (0..dim)
        .map(|i| {
            let v = ((i as f32) * 0.37).sin() * 0.5;
            half::bf16::from_f32(v).to_f32()
        })
        .collect();
    let limit = eager.cfg.swiglu_limit;
    for (li, (le, lm)) in eager.layers.iter().zip(mmap.layers.iter()).enumerate() {
        for (ei, (a, b)) in le
            .experts
            .iter()
            .zip(lm.experts.iter())
            .chain(std::iter::once((&le.shared, &lm.shared)))
            .enumerate()
        {
            let ya = a.forward(&x, dim, limit, Some(0.7));
            let yb = b.forward(&x, dim, limit, Some(0.7));
            assert_eq!(ya, yb, "layer {li} expert {ei}: mmap != eager");
        }
    }
}

/// End-to-end greedy through mmap'd experts must match reference.json —
/// identical bar to the eager engine test.
#[test]
fn mmap_greedy_matches_reference() {
    let dir = export_dir();
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let ids = |k: &str| {
        reference[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    };
    let (prompt, want) = (ids("prompt_ids"), ids("generated"));

    let mut model = {
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
                .unwrap();
        let n = m["num_layers"].as_u64().unwrap() as usize;
        load_stage_mode(&dir, 64, 0, n, true, true, ExpertsMode::Mmap).expect("mmap load")
    };
    let logits = model.prefill(&prompt);
    let mut got = vec![argmax(&logits)];
    for step in 1..want.len() {
        let logits = model.decode(got[step - 1], prompt.len() + step - 1);
        got.push(argmax(&logits));
    }
    assert_eq!(got, want, "mmap-experts greedy diverges from reference");
}
