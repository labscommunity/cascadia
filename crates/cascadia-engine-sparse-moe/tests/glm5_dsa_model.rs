//! End-to-end DSA wiring through the loader: a model exported WITH indexer
//! weights and a small `index_topk` must (a) load and (b) produce a different
//! greedy stream than its dense twin (same weights, `index_topk` >> seq → every
//! key selected). This proves the loader builds and attaches the indexer and
//! that it is actually live during generation. The pruned-attention *math* is
//! covered tightly by `glm5_dsa_attn`; here we only assert the wire activates.
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::loader::load_model;

fn dir(tag: &str) -> Option<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(tag);
    d.join("manifest.json").exists().then_some(d)
}

#[test]
fn dsa_model_prunes_and_differs_from_dense() {
    let sparse = dir("glm5_export_dsa")
        .expect("glm5_export_dsa fixture missing (run tools/glm5_ref/gen_fixtures.py)");
    let dense = dir("glm5_export_dsa_dense")
        .expect("glm5_export_dsa_dense fixture missing (run tools/glm5_ref/gen_fixtures.py)");
    let prompt = [1u32, 2, 3, 4];
    let (n_gen, max_seq) = (4usize, 32usize);

    let sparse_tok = load_model(&sparse, max_seq)
        .expect("load DSA model")
        .generate(&prompt, n_gen);
    let dense_tok = load_model(&dense, max_seq)
        .expect("load dense twin")
        .generate(&prompt, n_gen);

    // index_topk=2 prunes keys from position 2 on; the dense twin (topk 10000)
    // attends to every key. Same weights → the only difference is the pruning.
    assert_ne!(
        sparse_tok, dense_tok,
        "DSA(index_topk=2) produced the same stream as dense — indexer not active"
    );

    // Deterministic across loads.
    let again = load_model(&sparse, max_seq)
        .expect("reload")
        .generate(&prompt, n_gen);
    assert_eq!(sparse_tok, again, "DSA generation is not deterministic");
}
