//! Loader round-trip: `inkling::loader::load_model` reads a
//! `tools/export_inkling.py --tiny` layout (bf16/f32 shell safetensors + int4
//! expert bins) and must generate the exact greedy tokens transformers' native
//! `InklingForCausalLM` produces from the SAME (int4-dequantized) weights —
//! `reference.json` next to the export. Token-exact through int4 dequant
//! validates the exporter write, the loader read and the int4 numeric
//! contract together. The staged runner (rank 0 of 1) must match the
//! single-process model on the same prompt.
//!
//! Regenerate the export:
//!   python tools/export_inkling.py --tiny \
//!       crates/cascadia-engine-sparse-moe/tests/fixtures/inkling_export

use std::path::PathBuf;

use cascadia_engine_sparse_moe::inkling::loader::load_model;
use cascadia_engine_sparse_moe::inkling::model::argmax;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export")
}

struct Reference {
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    first_argmax: Option<u32>,
}

fn reference() -> Option<Reference> {
    let p = export_dir().join("reference.json");
    if !p.exists() {
        eprintln!("inkling_export/reference.json missing; skipping (run export_inkling.py --tiny)");
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .unwrap_or_else(|| panic!("reference.json: {k}"))
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    Some(Reference {
        prompt: ids("prompt_ids"),
        greedy: ids("greedy_ids"),
        first_argmax: v["first_logits_argmax"].as_u64().map(|x| x as u32),
    })
}

#[test]
fn loader_greedy_matches_hf_reference() {
    let Some(r) = reference() else { return };
    let mut model = load_model(&export_dir(), 64).expect("load inkling export");
    if let Some(want) = r.first_argmax {
        let logits = model.prefill(&r.prompt);
        assert_eq!(argmax(&logits) as u32, want, "first-token argmax");
        model.reset();
    }
    let got = model.greedy(&r.prompt, r.greedy.len());
    assert_eq!(got, r.greedy, "loader greedy mismatch vs HF reference");
}

#[test]
fn staged_runner_single_rank_matches_model() {
    let Some(r) = reference() else { return };
    let mut runner =
        InklingRunner::load_staged(&export_dir(), 64, 0, 1, 0, 0, Some("eager".into()))
            .expect("load rank 0 of 1");
    runner.reset();
    // Token-by-token drive (the pipeline's decode path).
    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in &r.prompt {
        let h = runner.embed_token(t);
        let h = runner.forward_layers(h, pos, None);
        next = argmax(&runner.head_logits(&h)) as u32;
        pos += 1;
    }
    let mut got = vec![next];
    for _ in 1..r.greedy.len() {
        let h = runner.embed_token(next);
        let h = runner.forward_layers(h, pos, None);
        next = argmax(&runner.head_logits(&h)) as u32;
        pos += 1;
        got.push(next);
    }
    assert_eq!(got, r.greedy, "staged token-by-token greedy mismatch");

    // Batched prefill (the pipeline's prefill path) must land on the same first token.
    runner.reset();
    let hs = runner.hidden_size();
    let mut rows = Vec::with_capacity(r.prompt.len() * hs);
    for &t in &r.prompt {
        rows.extend(runner.embed_token(t));
    }
    let out = runner.forward_layers_batch(rows, 0, r.prompt.len());
    let last = &out[(r.prompt.len() - 1) * hs..];
    assert_eq!(
        argmax(&runner.head_logits(last)) as u32,
        r.greedy[0],
        "batched prefill first token"
    );
}

#[test]
fn layer_split_covers_every_layer_exactly_once() {
    use cascadia_engine_sparse_moe::inkling::loader::read_manifest;
    use cascadia_engine_sparse_moe::inkling::stage::layer_split;
    let Ok(m) = read_manifest(&export_dir()) else {
        eprintln!("inkling_export/manifest.json missing; skipping");
        return;
    };
    for total in 1..=m.num_layers as u32 {
        let mut next = 0usize;
        for rank in 0..total {
            let (lo, hi) = layer_split(&m, rank, total).unwrap();
            assert_eq!(lo, next, "rank {rank}/{total} contiguity");
            assert!(hi > lo, "rank {rank}/{total} non-empty");
            next = hi;
        }
        assert_eq!(next, m.num_layers, "total {total} covers all layers");
    }
    // One rank more than layers: the LAST rank would own zero layers.
    let n = m.num_layers as u32;
    assert!(layer_split(&m, n, n + 1).is_err());
}
