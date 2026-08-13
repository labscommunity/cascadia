//! End-to-end IndexShare through the loader. The fixture has `indexer_types =
//! [full, shared, full, shared]`: layers 1 and 3 own NO indexer and must reuse
//! the previous full layer's top-k (carried forward). Two things are asserted:
//!
//!  1. The model LOADS — the loader attaches indexers only on the full layers
//!     and marks the shared ones (it does not try to read the missing indexer
//!     tensors of layers 1,3).
//!  2. With a small `index_topk` the sparse variant must diverge from its dense
//!     twin (same weights, `index_topk` >> seq). Since the shared layers own no
//!     indexer, the ONLY way they can prune is via the carried selection — so a
//!     divergence proves carry-forward reaches the shared layers.
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
fn indexshare_shared_layers_reuse_carried_topk() {
    let sparse = dir("glm5_export_indexshare")
        .expect("glm5_export_indexshare fixture missing (run tools/glm5_ref/gen_fixtures.py)");
    let dense = dir("glm5_export_indexshare_dense").expect(
        "glm5_export_indexshare_dense fixture missing (run tools/glm5_ref/gen_fixtures.py)",
    );
    let prompt = [1u32, 2, 3, 4];
    let (n_gen, max_seq) = (4usize, 32usize);

    // (1) loads despite layers 1,3 having no indexer tensors (they're "shared").
    let sparse_tok = load_model(&sparse, max_seq)
        .expect("load IndexShare model (shared layers must not require indexer tensors)")
        .generate(&prompt, n_gen);
    let dense_tok = load_model(&dense, max_seq)
        .expect("load dense twin")
        .generate(&prompt, n_gen);

    // (2) shared layers prune only via the carried top-k -> sparse must diverge.
    assert_ne!(
        sparse_tok, dense_tok,
        "IndexShare(topk=2) matched dense — carry-forward did not reach the shared layers"
    );

    // deterministic across loads.
    let again = load_model(&sparse, max_seq)
        .expect("reload")
        .generate(&prompt, n_gen);
    assert_eq!(
        sparse_tok, again,
        "IndexShare generation is not deterministic"
    );
}
