//! Golden tests for GLM-5.2 rope against the CPU reference
//! (`tools/glm5_ref/gen_fixtures.py`). Confirms the dsv4 pair-interleaved
//! rotation reused by `glm::rope` reproduces GLM's config exactly (dim 64,
//! theta 8e6, no YaRN, no inverse).
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::rope::{apply_rope_row, glm_freqs, ROPE_DIM};

macro_rules! fixtures {
    () => {{
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm5/fixtures.safetensors");
        StFile::open(&p).expect("open fixtures")
    }};
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        // A NaN never trips the tolerance test below (`NaN > x` is false), so an
        // all-NaN kernel output would skip every element and pass. Reject
        // non-finite values on both sides explicitly.
        assert!(
            g.is_finite() && w.is_finite(),
            "{name}: non-finite value at [{i}]: got {g} want {w}"
        );
        let d = (g - w).abs();
        if d > atol + rtol * w.abs() && d > worst.1 {
            worst = (i, d, g, w);
        }
    }
    assert!(
        worst.1 == 0.0,
        "{name}: worst diff {} at [{}]: got {} want {} (atol {atol} rtol {rtol})",
        worst.1,
        worst.0,
        worst.2,
        worst.3
    );
}

#[test]
fn glm_freqs_match_reference() {
    let fx = fixtures!();
    // fixtures store torch.view_as_real(freqs_cis): [seq, dim/2, 2] flattened,
    // interleaved (cos, sin) — the same layout as Freqs::data.
    let (shape, want) = fx.f32("rope.freqs").unwrap();
    let seq = shape[0];
    let f = glm_freqs(seq);
    assert_close("rope.freqs", &f.data, &want, 1e-4, 1e-4);
}

#[test]
fn glm_rope_apply_matches_reference() {
    let fx = fixtures!();
    let (shape, x0) = fx.f32("rope.apply.in").unwrap(); // [s, h, dim]
    let (_, want) = fx.f32("rope.apply.out").unwrap();
    let (s, h, d) = (shape[0], shape[1], shape[2]);
    assert_eq!(d, ROPE_DIM, "fixture rope dim != GLM ROPE_DIM");
    let freqs = glm_freqs(s);
    let mut x = x0.clone();
    for si in 0..s {
        for hi in 0..h {
            let off = (si * h + hi) * d;
            apply_rope_row(&mut x[off..off + d], &freqs, si, d, false);
        }
    }
    assert_close("rope.apply", &x, &want, 1e-3, 0.01);
}
