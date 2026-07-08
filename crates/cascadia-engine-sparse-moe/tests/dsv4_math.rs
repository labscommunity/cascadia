//! Golden tests for the dsv4 numeric primitives against fixtures generated
//! by the CPU reference (`tools/deepseek_v4_ref/gen_fixtures.py`).
//!
//! Regenerate fixtures:
//!   python tools/deepseek_v4_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/dsv4

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::attn::sparse_attn_pos;
use cascadia_engine_sparse_moe::dsv4::hc::hc_split_sinkhorn;
use cascadia_engine_sparse_moe::dsv4::math::{act_quant_sim, fp4_act_quant_sim, hadamard, rmsnorm};
use cascadia_engine_sparse_moe::dsv4::rope::{apply_rope_row, precompute_freqs};
use cascadia_engine_sparse_moe::dsv4::st::StFile;

fn fixtures() -> StFile {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4/fixtures.safetensors");
    StFile::open(&p).expect("open fixtures (run gen_fixtures.py)")
}

/// Mixed abs/rel closeness: |a-b| <= atol + rtol*|b|. Reports worst offender.
fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        let bound = atol + rtol * w.abs();
        if d > bound && d > worst.1 {
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
fn hadamard_matches_reference() {
    let fx = fixtures();
    let (shape, mut x) = fx.f32("hadamard.in").unwrap();
    let (_, want) = fx.f32("hadamard.out").unwrap();
    let d = *shape.last().unwrap();
    hadamard(&mut x, d, (d as f32).powf(-0.5));
    assert_close("hadamard", &x, &want, 1e-6, 1e-3);
}

#[test]
fn act_quant_sim_matches_reference_exactly() {
    let fx = fixtures();
    let (_, mut x) = fx.f32("act_quant.in").unwrap();
    let (_, want) = fx.f32("act_quant.out").unwrap();
    act_quant_sim(&mut x, 64);
    assert_close("act_quant", &x, &want, 0.0, 0.0);
}

#[test]
fn fp4_act_quant_sim_matches_reference_exactly() {
    let fx = fixtures();
    let (_, mut x) = fx.f32("fp4_act_quant.in").unwrap();
    let (_, want) = fx.f32("fp4_act_quant.out").unwrap();
    fp4_act_quant_sim(&mut x, 32);
    assert_close("fp4_act_quant", &x, &want, 0.0, 0.0);
}

#[test]
fn sinkhorn_matches_reference() {
    let fx = fixtures();
    let (mshape, mixes) = fx.f32("sinkhorn.mixes").unwrap();
    let (_, scale) = fx.f32("sinkhorn.scale").unwrap();
    let (_, base) = fx.f32("sinkhorn.base").unwrap();
    let (_, pre_w) = fx.f32("sinkhorn.pre").unwrap();
    let (_, post_w) = fx.f32("sinkhorn.post").unwrap();
    let (_, comb_w) = fx.f32("sinkhorn.comb").unwrap();
    let hc = 4usize;
    let mix_hc = (2 + hc) * hc;
    assert_eq!(*mshape.last().unwrap(), mix_hc);
    let tokens = mixes.len() / mix_hc;
    for t in 0..tokens {
        let c = hc_split_sinkhorn(
            &mixes[t * mix_hc..(t + 1) * mix_hc],
            &scale,
            &base,
            hc,
            20,
            1e-6,
        );
        assert_close(
            &format!("sinkhorn.pre[{t}]"),
            &c.pre,
            &pre_w[t * hc..(t + 1) * hc],
            2e-5,
            1e-5,
        );
        assert_close(
            &format!("sinkhorn.post[{t}]"),
            &c.post,
            &post_w[t * hc..(t + 1) * hc],
            2e-5,
            1e-5,
        );
        assert_close(
            &format!("sinkhorn.comb[{t}]"),
            &c.comb,
            &comb_w[t * hc * hc..(t + 1) * hc * hc],
            2e-5,
            1e-5,
        );
    }
}

#[test]
fn rmsnorm_matches_reference() {
    let fx = fixtures();
    let (_, mut x) = fx.f32("rmsnorm.in").unwrap();
    let (_, w) = fx.f32("rmsnorm.w").unwrap();
    let (_, want) = fx.f32("rmsnorm.out").unwrap();
    rmsnorm(&mut x, &w, 1e-6);
    assert_close("rmsnorm", &x, &want, 1e-6, 0.01);
}

#[test]
fn yarn_freqs_match_reference() {
    let fx = fixtures();
    // fixtures store torch.view_as_real(freqs_cis): [seq, half, 2]
    let (shape, want) = fx.f32("rope.freqs_yarn").unwrap();
    let (seq, half) = (shape[0], shape[1]);
    let f = precompute_freqs(half * 2, seq, 32, 160000.0, 16.0, 32.0, 1.0);
    assert_close("freqs_yarn", &f.data, &want, 1e-4, 1e-4);

    let (shape, want) = fx.f32("rope.freqs_plain").unwrap();
    let f = precompute_freqs(shape[1] * 2, shape[0], 0, 10000.0, 16.0, 32.0, 1.0);
    assert_close("freqs_plain", &f.data, &want, 1e-4, 1e-4);
}

#[test]
fn rope_apply_matches_reference() {
    let fx = fixtures();
    let plain = precompute_freqs(16, 64, 0, 10000.0, 16.0, 32.0, 1.0);
    let yarn = precompute_freqs(16, 64, 32, 160000.0, 16.0, 32.0, 1.0);

    // 4-d path [1, 6, 4, 16]: rows are heads, position = s index, rd = 16
    let (shape, x0) = fx.f32("rope.apply4.in").unwrap();
    let (s, h, d) = (shape[1], shape[2], shape[3]);
    for (name, inverse) in [("rope.apply4.out", false), ("rope.apply4.inv_out", true)] {
        let (_, want) = fx.f32(name).unwrap();
        let mut x = x0.clone();
        for si in 0..s {
            for hi in 0..h {
                let off = (si * h + hi) * d;
                apply_rope_row(&mut x[off..off + d], &plain, si, d, inverse);
            }
        }
        assert_close(name, &x, &want, 1e-3, 0.01);
    }

    // 3-d path [1, 6, 16] with the YaRN table (compressor kv convention)
    let (shape, x0) = fx.f32("rope.apply3.in").unwrap();
    let (s, d) = (shape[1], shape[2]);
    let (_, want) = fx.f32("rope.apply3.out").unwrap();
    let mut x = x0.clone();
    for si in 0..s {
        let off = si * d;
        apply_rope_row(&mut x[off..off + d], &yarn, si, d, false);
    }
    assert_close("rope.apply3", &x, &want, 1e-3, 0.01);
}

#[test]
fn sparse_attn_matches_reference() {
    let fx = fixtures();
    let (qs, q) = fx.f32("sparse_attn.q").unwrap(); // [1, 3, 4, 16]
    let (_, kv) = fx.f32("sparse_attn.kv").unwrap(); // [1, 10, 16]
    let (_, sink) = fx.f32("sparse_attn.sink").unwrap(); // [4]
    let (is, idxs) = fx.i32("sparse_attn.idxs").unwrap(); // [1, 3, 6]
    let (_, want) = fx.f32("sparse_attn.out").unwrap();
    let (s, h, d) = (qs[1], qs[2], qs[3]);
    let topk = is[2];
    let scale = (d as f32).powf(-0.5);
    let mut out = vec![0.0f32; s * h * d];
    for si in 0..s {
        sparse_attn_pos(
            &q[si * h * d..(si + 1) * h * d],
            &kv,
            &sink,
            &idxs[si * topk..(si + 1) * topk],
            h,
            d,
            scale,
            &mut out[si * h * d..(si + 1) * h * d],
        );
    }
    assert_close("sparse_attn", &out, &want, 1e-3, 0.01);
}
