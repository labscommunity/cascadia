//! Eager vs mmap expert parity: loading the SAME export with int4 experts held
//! eager (dequantized f32) vs mmap'd (dequant rows on the fly) must produce the
//! same tokens and near-identical logits. The two kernels reorder the f32
//! summation, so they agree within bf16 tolerance, not bitwise (as dsv4's
//! MmapExpert does vs its eager path).
//!
//! Requires the export fixture (run tools/glm5_ref/gen_fixtures.py).

use std::path::PathBuf;

use cascadia_engine_sparse_moe::glm::stage::GlmRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &x)| {
                if x > bv {
                    (i, x)
                } else {
                    (bi, bv)
                }
            },
        )
        .0
}

#[test]
fn eager_and_mmap_experts_agree() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export");
    if !dir.join("manifest.json").exists() {
        eprintln!("SKIP: glm5_export absent (run tools/glm5_ref/gen_fixtures.py)");
        return;
    }

    // Force each mode via the env override, load, then clear (single test in
    // this binary -> no env race).
    std::env::set_var("CASCADIA_GLM5_EXPERTS", "mmap");
    let mut mm =
        GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0, Default::default()).expect("mmap load");
    std::env::set_var("CASCADIA_GLM5_EXPERTS", "eager");
    let mut eg =
        GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0, Default::default()).expect("eager load");
    std::env::remove_var("CASCADIA_GLM5_EXPERTS");

    mm.reset();
    eg.reset();
    for (pos, &tok) in [1u32, 2, 3, 4].iter().enumerate() {
        let lm = {
            let h = mm.embed_token(tok);
            let h = mm.forward_layers(h, pos, None);
            mm.head_logits(&h)
        };
        let le = {
            let h = eg.embed_token(tok);
            let h = eg.forward_layers(h, pos, None);
            eg.head_logits(&h)
        };
        let worst = lm
            .iter()
            .zip(&le)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 5e-2,
            "pos {pos}: eager vs mmap logits diverge by {worst}"
        );
        assert_eq!(
            argmax(&lm),
            argmax(&le),
            "pos {pos}: argmax differs (eager vs mmap)"
        );
    }

    // End-to-end greedy must be token-identical.
    std::env::set_var("CASCADIA_GLM5_EXPERTS", "mmap");
    let got_mmap = GlmRunner::load_staged(&dir, 32, 0, 1, 0, 0, Default::default())
        .unwrap()
        .generate_argmax(&[1, 2, 3, 4], 4);
    std::env::remove_var("CASCADIA_GLM5_EXPERTS");
    assert_eq!(
        got_mmap,
        vec![4u32, 10, 3, 15],
        "mmap greedy diverges from reference"
    );
}
