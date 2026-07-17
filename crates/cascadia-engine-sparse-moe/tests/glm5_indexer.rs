//! Golden test for the GLM-5.2 DSA lightning indexer against the CPU reference
//! (`tools/glm5_ref::indexer_ref`): per-key scores + top-k selection.
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::indexer::{Indexer, IndexerWeights};

macro_rules! fixtures {
    () => {{
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm5/fixtures.safetensors");
        if !p.exists() {
            eprintln!("SKIP: {} absent (run tools/glm5_ref/gen_fixtures.py)", p.display());
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
fn dsa_indexer_matches_reference() {
    let fx = fixtures!();
    // dims match tools/glm5_ref/gen_fixtures.py::icfg
    let (hidden, q_lora, nh, hd, rope_dim) = (32usize, 16, 2, 8, 4);
    let (eps, theta) = (1e-6f32, 8.0e6f32);
    let (query_pos, topk) = (5usize, 3usize);

    let (xshape, x) = fx.f32("idx.x").unwrap(); // [S, hidden]
    let (_, qr) = fx.f32("idx.qr").unwrap(); // [S, q_lora]
    let (_, isc_want) = fx.f32("idx.isc").unwrap(); // [query_pos+1]
    let (_, sel_want_i) = fx.i32("idx.sel").unwrap(); // [keep]
    let seq = xshape[0];

    let w = IndexerWeights {
        ix_wq: bits(&fx.f32("idx.ix_wq").unwrap().1),
        ix_wk: bits(&fx.f32("idx.ix_wk").unwrap().1),
        ix_wp: bits(&fx.f32("idx.ix_wp").unwrap().1),
        k_norm_w: fx.f32("idx.k_norm_w").unwrap().1,
        k_norm_b: fx.f32("idx.k_norm_b").unwrap().1,
    };
    let freqs = precompute_freqs(rope_dim, seq, 0, theta, 1.0, 32.0, 1.0);
    let mut idx = Indexer::new(hidden, q_lora, nh, hd, rope_dim, seq, eps, w, freqs);

    for s in 0..seq {
        idx.append_key(&x[s * hidden..(s + 1) * hidden]);
    }

    let qq = &qr[query_pos * q_lora..(query_pos + 1) * q_lora];
    let xq = &x[query_pos * hidden..(query_pos + 1) * hidden];

    let isc = idx.scores(qq, xq, query_pos);
    assert_close("idx.isc", &isc, &isc_want, 1e-3, 1e-2);

    let sel = idx.select(qq, xq, query_pos, topk);
    let sel_want: Vec<u32> = sel_want_i.iter().map(|&v| v as u32).collect();
    assert_eq!(sel, sel_want, "idx.sel");
}
