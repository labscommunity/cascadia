//! Loader round-trip: the Rust `glm::loader::load_model` reads a
//! `tools/export_glm5.py --tiny` layout (bf16 shell safetensors + int4 expert
//! bins) and must generate the exact greedy tokens the CPU reference produces
//! from the SAME files (`tools/glm5_ref/load_export.py`). Token-exact through
//! int4 dequant validates the exporter write, the loader read, and the int4
//! numeric contract together.
//!
//! Regenerate the export:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::loader::load_model;

#[test]
fn loader_greedy_matches_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    if !dir.join("manifest.json").exists() {
        eprintln!(
            "SKIP: {} absent (run tools/glm5_ref/gen_fixtures.py)",
            dir.display()
        );
        return;
    }
    let mut model = load_model(&dir, 32).expect("load glm5 export");
    let got = model.generate(&[1, 2, 3, 4], 4);
    // meta printed by gen_fixtures: [loader round-trip] ... greedy [4, 10, 3, 15]
    assert_eq!(got, vec![4u32, 10, 3, 15], "loader greedy mismatch");
}
