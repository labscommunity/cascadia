//! Real-exporter round-trip: `export_glm5.py export_real` converts a synthetic
//! GLM-5.2-FP8 checkpoint (real HF tensor names, fp8 e4m3 + block-128
//! weight_scale_inv) to the engine layout; the Rust loader must load it and
//! reproduce the greedy tokens the export reader produces. Validates the FP8
//! dequant + HF->shell name mapping + int4 repack end-to-end, no 744B
//! checkpoint required.
//!
//! Regenerate: python tools/glm5_ref/gen_fixtures.py --out .../fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::loader::load_model;

#[test]
fn real_exporter_roundtrip_matches_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_fp8_export");
    if !dir.join("manifest.json").exists() {
        eprintln!("SKIP: glm5_fp8_export absent (run tools/glm5_ref/gen_fixtures.py)");
        return;
    }
    let mut model = load_model(&dir, 32).expect("load fp8-exported model");
    let got = model.generate(&[1, 2, 3, 4], 4);
    // meta printed by gen_fixtures: [fp8 round-trip] ... greedy [9, 15, 0, 0]
    assert_eq!(
        got,
        vec![9u32, 15, 0, 0],
        "fp8-exported loader greedy mismatch"
    );
}
