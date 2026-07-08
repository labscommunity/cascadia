//! Pipeline staging: split the tiny export across 2 and 4 stages, chain the
//! stages' hidden-copies activations (exactly what will ride the sparse-moe
//! `Forward` wire with hidden_size = hc*dim), and require the greedy tokens
//! to match reference.json — same bar as the single-stage engine test.
//!
//! Token ids stay on stage 0 (hash-gate layers always live there); downstream
//! stages receive activations only.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::loader::load_stage;
use cascadia_engine_sparse_moe::dsv4::model::DsV4Model;

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export")
}

fn reference() -> (Vec<usize>, Vec<usize>) {
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(export_dir().join("reference.json")).unwrap(),
    )
    .unwrap();
    let ids = |k: &str| {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    };
    (ids("prompt_ids"), ids("generated"))
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Greedy-decode through a chain of stages; stage 0 embeds (and owns the
/// ids), the last stage produces logits.
fn pipeline_greedy(stages: &mut [DsV4Model], prompt: &[usize], n_gen: usize) -> Vec<usize> {
    let n = stages.len();
    // prefill
    let mut acts = stages[0].embed_ids(prompt);
    for (r, stage) in stages.iter_mut().enumerate() {
        let ids = if r == 0 { Some(prompt) } else { None };
        acts = stage.forward_layers_prefill(acts, ids);
    }
    let mut toks = vec![argmax(&stages[n - 1].logits(&acts[prompt.len() - 1]))];
    // decode
    for step in 1..n_gen {
        let pos = prompt.len() + step - 1;
        let id = toks[step - 1];
        let mut act = stages[0].embed_ids(&[id]).pop().unwrap();
        for (r, stage) in stages.iter_mut().enumerate() {
            let sid = if r == 0 { Some(id) } else { None };
            act = stage.forward_layers_decode(act, pos, sid);
        }
        toks.push(argmax(&stages[n - 1].logits(&act)));
    }
    toks
}

#[test]
fn two_stage_pipeline_matches_reference() {
    let dir = export_dir();
    let (prompt, want) = reference();
    let mut stages = vec![
        load_stage(&dir, 64, 0, 2, true, false).unwrap(),
        load_stage(&dir, 64, 2, 4, false, true).unwrap(),
    ];
    let got = pipeline_greedy(&mut stages, &prompt, want.len());
    assert_eq!(got, want, "2-stage greedy diverges");
}

#[test]
fn four_stage_pipeline_matches_reference() {
    let dir = export_dir();
    let (prompt, want) = reference();
    let mut stages: Vec<DsV4Model> = (0..4)
        .map(|r| load_stage(&dir, 64, r, r + 1, r == 0, r == 3).unwrap())
        .collect();
    let got = pipeline_greedy(&mut stages, &prompt, want.len());
    assert_eq!(got, want, "4-stage greedy diverges");
}

#[test]
fn hash_layers_must_stay_on_stage_zero() {
    // stage starting at layer 0 that is NOT first must be rejected (ids
    // don't cross the wire).
    let err = load_stage(&export_dir(), 64, 0, 2, false, false);
    assert!(err.is_err(), "hash layers outside stage 0 must be rejected");
}
