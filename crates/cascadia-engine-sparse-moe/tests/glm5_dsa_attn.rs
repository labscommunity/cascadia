//! Golden test for DSA-gated MLA attention: the lightning indexer selects the
//! top-`index_topk` causal keys and the absorbed decode attends only to those.
//! Validated against `tools/glm5_ref::attention_ref_dsa` (naive attention over
//! the same indexer selection). The fixture uses index_topk=3 with a length-6
//! sequence, so positions 3..5 actually prune keys — a wrong selection or a
//! wrong restricted-softmax would diverge far beyond the ULP tolerance.
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::attn::{AttentionLayer, AttnWeights};
use cascadia_engine_sparse_moe::glm::indexer::{Indexer, IndexerWeights};

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

fn bits(f: &[f32]) -> Vec<u16> {
    f.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
fn dsa_gated_attention_matches_reference() {
    let fx = fixtures!();
    // dims match tools/glm5_ref/gen_fixtures.py DSA block (acfg + dnh/dhd/DTOPK).
    let (hidden, h, nope, rope, vh, kvl, ql) = (32usize, 3, 6, 4, 6, 8, 16);
    let (nh, hd) = (2usize, 8); // index_n_heads, index_head_dim
    let (theta, ln_eps) = (8.0e6f32, 1e-6f32);
    let index_topk = 3usize;

    let (xshape, x) = fx.f32("dsa.x").unwrap(); // [S, hidden]
    let seq = xshape[0];
    assert_eq!(xshape[1], hidden);

    let aw = AttnWeights {
        wq_a: bits(&fx.f32("dsa.wq_a").unwrap().1),
        q_a_ln: fx.f32("dsa.q_a_ln").unwrap().1,
        wq_b: bits(&fx.f32("dsa.wq_b").unwrap().1),
        wkv_a: bits(&fx.f32("dsa.wkv_a").unwrap().1),
        kv_a_ln: fx.f32("dsa.kv_a_ln").unwrap().1,
        wkv_b: bits(&fx.f32("dsa.wkv_b").unwrap().1),
        wo: bits(&fx.f32("dsa.wo").unwrap().1),
    };
    let freqs = precompute_freqs(rope, seq, 0, theta, 1.0, 32.0, 1.0);
    let mut layer = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, seq, aw, freqs);

    let iw = IndexerWeights {
        ix_wq: bits(&fx.f32("dsa.ix_wq").unwrap().1),
        ix_wk: bits(&fx.f32("dsa.ix_wk").unwrap().1),
        ix_wp: bits(&fx.f32("dsa.ix_wp").unwrap().1),
        k_norm_w: fx.f32("dsa.k_norm_w").unwrap().1,
        k_norm_b: fx.f32("dsa.k_norm_b").unwrap().1,
    };
    // The indexer ropes the qk_rope prefix of each head at the same positions,
    // so it shares the attention rope config (dim=qk_rope, same theta).
    let ifreqs = precompute_freqs(rope, seq, 0, theta, 1.0, 32.0, 1.0);
    let indexer = Indexer::new(hidden, ql, nh, hd, rope, seq, ln_eps, iw, ifreqs);
    layer.attach_indexer(indexer, index_topk);

    let (_, want) = fx.f32("dsa.out").unwrap();
    let mut got = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let o = layer.forward_token(&x[s * hidden..(s + 1) * hidden], &mut None);
        got[s * hidden..(s + 1) * hidden].copy_from_slice(&o);
    }
    assert_close("dsa.out", &got, &want, 2e-2, 1e-2);
}
