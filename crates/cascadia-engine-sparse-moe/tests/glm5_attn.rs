//! Golden test for GLM-5.2 MLA attention against the naive CPU reference
//! (`tools/glm5_ref`). The Rust shell runs the ABSORBED-latent decode path;
//! the reference is the obviously-correct materialized form. They agree to
//! f32-ULP + bf16-rounding, which validates the absorb algebra.
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::attn::{AttentionLayer, AttnWeights};

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

/// bf16-valued f32 -> bf16 bits (exact truncation; the low 16 mantissa bits of
/// a bf16-valued f32 are zero, so no rounding is needed).
fn bits(f: &[f32]) -> Vec<u16> {
    f.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
fn mla_attention_matches_reference() {
    let fx = fixtures!();
    let (xshape, x) = fx.f32("attn.x").unwrap(); // [S, hidden]

    // dims match tools/glm5_ref/gen_fixtures.py::acfg
    let (hidden, h, nope, rope, vh, kvl, ql) = (32usize, 3, 6, 4, 6, 8, 16);
    let theta = 8.0e6f32;
    let seq = xshape[0];
    assert_eq!(xshape[1], hidden);

    let w = AttnWeights {
        wq_a: bits(&fx.f32("attn.wq_a").unwrap().1),
        q_a_ln: fx.f32("attn.q_a_ln").unwrap().1,
        wq_b: bits(&fx.f32("attn.wq_b").unwrap().1),
        wkv_a: bits(&fx.f32("attn.wkv_a").unwrap().1),
        kv_a_ln: fx.f32("attn.kv_a_ln").unwrap().1,
        wkv_b: bits(&fx.f32("attn.wkv_b").unwrap().1),
        wo: bits(&fx.f32("attn.wo").unwrap().1),
    };
    let freqs = precompute_freqs(rope, seq, 0, theta, 1.0, 32.0, 1.0);
    let mut layer = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, seq, w, freqs);

    let (_, want) = fx.f32("attn.out").unwrap();
    let mut got = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let o = layer.forward_token(&x[s * hidden..(s + 1) * hidden], &mut None);
        got[s * hidden..(s + 1) * hidden].copy_from_slice(&o);
    }
    assert_close("attn.out", &got, &want, 1e-2, 1e-2);
}
