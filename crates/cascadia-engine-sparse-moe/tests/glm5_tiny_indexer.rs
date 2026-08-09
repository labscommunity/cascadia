//! Proves the exporter's *default* `--tiny` indexer config (index_n_heads=2,
//! index_head_dim=16, index_topk=8, full/shared/full over 3 layers — see
//! `tools/export_glm5.py::TINY_INDEXER_KW`) actually engages during
//! generation, not just that it loads. Same sparse-vs-dense-diff pattern as
//! `glm5_dsa_model`, pinned to the CLI's actual default rather than an ad-hoc
//! config, so a `python tools/export_glm5.py --tiny <dir>` fixture is provably
//! exercising the sparse boundary (a later iGPU-vs-Rust parity harness depends
//! on this). The stronger, direct observable — `Indexer::select()`'s returned
//! length pinned at `topk` once past budget — is unit-tested in
//! `glm::indexer::tests::select_prunes_once_past_topk` (no fixture needed);
//! this test is the end-to-end companion proving the wiring through the loader.
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
fn tiny_default_indexer_prunes_and_differs_from_dense() {
    let (Some(sparse), Some(dense)) = (
        dir("glm5_export_tiny_indexer"),
        dir("glm5_export_tiny_indexer_dense"),
    ) else {
        eprintln!("SKIP: tiny-indexer export fixtures absent (run tools/glm5_ref/gen_fixtures.py)");
        return;
    };
    let prompt = [1u32, 2, 3, 4];
    // index_topk=8; prompt(4) + 10 generated tokens carries the cached length
    // to 14 — well past the boundary, not just brushing it.
    let (n_gen, max_seq) = (10usize, 32usize);

    let sparse_tok = load_model(&sparse, max_seq)
        .expect("load tiny default-indexer model")
        .generate(&prompt, n_gen);
    let dense_tok = load_model(&dense, max_seq)
        .expect("load dense twin (index_topk=10000)")
        .generate(&prompt, n_gen);

    // Same weights, only index_topk differs. Past position 8 the sparse export
    // prunes keys the dense twin still attends to; identical output there would
    // mean the loader isn't actually attaching/consulting the indexer.
    assert_ne!(
        sparse_tok, dense_tok,
        "--tiny's default indexer (index_topk=8) produced the same stream as \
         its dense twin past the boundary — the default indexer is not engaging"
    );

    // Deterministic across loads (no hidden nondeterminism in the sparse path).
    let again = load_model(&sparse, max_seq)
        .expect("reload")
        .generate(&prompt, n_gen);
    assert_eq!(
        sparse_tok, again,
        "tiny default-indexer generation is not deterministic"
    );
}
