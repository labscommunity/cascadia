//! GLM-5.2 staged runner (single-stage): the `GlmRunner` driven through the
//! `StagedRunner` surface must (a) generate the same greedy tokens as the
//! loader/reference, and (b) compose identically to the full `GlmModel`
//! (embed_token -> forward_layers -> head_logits == forward_token).
//!
//! Regenerate the export:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::loader::load_model;
use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

fn export_dir() -> Option<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    d.join("manifest.json").exists().then_some(d)
}

#[test]
fn runner_greedy_matches_reference() {
    let Some(dir) = export_dir() else {
        eprintln!("SKIP: glm5_export absent (run tools/glm5_ref/gen_fixtures.py)");
        return;
    };
    let mut runner = GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0).expect("load glm5 stage");
    let got = runner.generate_argmax(&[1, 2, 3, 4], 4);
    assert_eq!(got, vec![4u32, 10, 3, 15], "runner greedy mismatch");
}

#[test]
fn staged_composition_matches_full_model() {
    let Some(dir) = export_dir() else {
        eprintln!("SKIP: glm5_export absent (run tools/glm5_ref/gen_fixtures.py)");
        return;
    };
    let mut model = load_model(&dir, 32).expect("load full");
    let mut runner = GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0).expect("load stage");
    model.reset();
    runner.reset();
    // Drive both over the same tokens; the split staged path must equal the
    // full forward_token logits at every position (same weights, same ops).
    for (pos, &tok) in [1u32, 2, 3, 4].iter().enumerate() {
        let full = model.forward_token(tok);
        let h = runner.embed_token(tok);
        let h = runner.forward_layers(h, pos, None);
        let split = runner.head_logits(&h);
        assert_eq!(full.len(), split.len());
        let worst = full
            .iter()
            .zip(&split)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "pos {pos}: staged vs full logits diverge by {worst}");
    }
}
