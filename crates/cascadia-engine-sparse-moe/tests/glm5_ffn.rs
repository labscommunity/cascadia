//! Golden test for the GLM SwiGLU FFN against the CPU reference
//! (`tools/glm5_ref::swiglu_ref`).
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::ffn::swiglu;

macro_rules! fixtures {
    () => {{
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm5/fixtures.safetensors");
        if !p.exists() {
            eprintln!(
                "SKIP: {} absent (run tools/glm5_ref/gen_fixtures.py)",
                p.display()
            );
            return;
        }
        StFile::open(&p).expect("open fixtures")
    }};
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
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

fn bits(f: &[f32]) -> Vec<u16> {
    f.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
fn swiglu_matches_reference() {
    let fx = fixtures!();
    let (hidden, inter) = (32usize, 20usize);
    let (xshape, x) = fx.f32("ffn.x").unwrap(); // [rows, hidden]
    let (_, want) = fx.f32("ffn.out").unwrap();
    let wg = bits(&fx.f32("ffn.wg").unwrap().1);
    let wu = bits(&fx.f32("ffn.wu").unwrap().1);
    let wd = bits(&fx.f32("ffn.wd").unwrap().1);
    let rows = xshape[0];

    let mut got = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let y = swiglu(
            &x[r * hidden..(r + 1) * hidden],
            &wg,
            &wu,
            &wd,
            hidden,
            inter,
        );
        got[r * hidden..(r + 1) * hidden].copy_from_slice(&y);
    }
    assert_close("ffn.out", &got, &want, 1e-3, 1e-2);
}
