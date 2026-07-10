//! Golden tests for the dsv4 Compressor (attention + indexer variants) and
//! the DSA Indexer, against stateful prefill+decode sequences captured from
//! the CPU reference (`gen_fixtures.py` section 8b).

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::compress::{Compressor, CompressorWeights};
use cascadia_engine_sparse_moe::dsv4::indexer::{Indexer, IndexerWeights};
use cascadia_engine_sparse_moe::dsv4::rope::{precompute_freqs, Freqs};
use cascadia_engine_sparse_moe::dsv4::st::StFile;

const DIM: usize = 64; // tiny model hidden
const RD: usize = 16; // rope_head_dim
const MAX_SEQ: usize = 64;
const PREFILL: usize = 18;

/// Open the gen_fixtures.py output, or SKIP the test (return early) when it is
/// absent. The *.safetensors fixtures are gitignored, so a fresh CI checkout
/// (stub mode) does not have them; they exist after running gen_fixtures.py.
macro_rules! fixtures {
    () => {{
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/dsv4/fixtures.safetensors");
        if !p.exists() {
            eprintln!("SKIP: {} absent (run gen_fixtures.py)", p.display());
            return;
        }
        StFile::open(&p).expect("open fixtures")
    }};
}

/// The fixture components all use the compress-layer table:
/// theta = compress_rope_theta (160000), YaRN off (original_seq_len = 0).
fn table() -> Freqs {
    precompute_freqs(RD, MAX_SEQ, 0, 160000.0, 16.0, 32.0, 1.0)
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(
            d <= atol + rtol * w.abs(),
            "{name}: diff {d} at [{i}]: got {g} want {w}"
        );
    }
}

fn build_compressor(fx: &StFile, tag: &str, d: usize, ratio: usize, rotate: bool) -> Compressor {
    let w = CompressorWeights {
        wkv: fx.f32(&format!("{tag}.wkv_w")).unwrap().1,
        wgate: fx.f32(&format!("{tag}.wgate_w")).unwrap().1,
        ape: fx.f32(&format!("{tag}.ape")).unwrap().1,
        norm_w: fx.f32(&format!("{tag}.norm_w")).unwrap().1,
    };
    Compressor::new(DIM, d, RD, ratio, rotate, 1e-6, w)
}

/// Run the fixture's prefill + decode sequence through the Rust compressor,
/// maintaining a cache like the reference does, and compare fired rows +
/// the final cache.
fn run_compressor_case(
    tag: &str,
    d: usize,
    ratio: usize,
    rotate: bool,
    n_dec: usize,
    fires_at: &[usize],
) {
    let fx = fixtures!();
    let f = table();
    let mut comp = build_compressor(&fx, tag, d, ratio, rotate);
    let rows_total = MAX_SEQ / ratio;
    let mut cache = vec![0.0f32; rows_total * d];

    let (_, x) = fx.f32(&format!("{tag}.x_prefill")).unwrap();
    let rows = comp.prefill(&x, PREFILL, &f);
    if PREFILL / ratio > 0 {
        let (_, want) = fx.f32(&format!("{tag}.prefill_ret")).unwrap();
        assert_eq!(rows.len(), PREFILL / ratio, "{tag}: prefill row count");
        let got: Vec<f32> = rows.iter().flatten().copied().collect();
        assert_close(&format!("{tag}.prefill_ret"), &got, &want, 1e-3, 0.02);
    }
    for (blk, row) in rows.iter().enumerate() {
        cache[blk * d..(blk + 1) * d].copy_from_slice(row);
    }

    let (_, xd) = fx.f32(&format!("{tag}.x_dec")).unwrap();
    for i in 0..n_dec {
        let pos = PREFILL + i;
        let fired = comp.decode(&xd[i * DIM..(i + 1) * DIM], pos, &f);
        if fires_at.contains(&pos) {
            let row = fired.unwrap_or_else(|| panic!("{tag}: expected fire at pos {pos}"));
            let (_, want) = fx.f32(&format!("{tag}.dec_ret_{pos}")).unwrap();
            assert_close(&format!("{tag}.dec_ret_{pos}"), &row, &want, 1e-3, 0.02);
            let slot = pos / ratio;
            cache[slot * d..(slot + 1) * d].copy_from_slice(&row);
        } else {
            assert!(fired.is_none(), "{tag}: unexpected fire at pos {pos}");
        }
    }

    let (_, want_cache) = fx.f32(&format!("{tag}.cache_final")).unwrap();
    assert_close(
        &format!("{tag}.cache_final"),
        &cache,
        &want_cache,
        1e-3,
        0.02,
    );
}

#[test]
fn compressor_ratio4_overlap_matches_reference() {
    run_compressor_case("comp4", 80, 4, false, 8, &[19, 23]);
}

#[test]
fn compressor_ratio16_matches_reference() {
    run_compressor_case("comp16", 80, 16, false, 14, &[31]);
}

#[test]
fn compressor_indexer_variant_matches_reference() {
    run_compressor_case("compidx", 32, 4, true, 8, &[19, 23]);
}

#[test]
fn indexer_matches_reference() {
    let fx = fixtures!();
    let f = table();
    let (h, d, topk, ratio, q_lora) = (2usize, 32usize, 4usize, 4usize, 32usize);
    let comp = Compressor::new(
        DIM,
        d,
        RD,
        ratio,
        true,
        1e-6,
        CompressorWeights {
            wkv: fx.f32("idx.comp_wkv_w").unwrap().1,
            wgate: fx.f32("idx.comp_wgate_w").unwrap().1,
            ape: fx.f32("idx.comp_ape").unwrap().1,
            norm_w: fx.f32("idx.comp_norm_w").unwrap().1,
        },
    );
    let w = IndexerWeights {
        wq_b: fx.f32("idx.wq_b_w").unwrap().1,
        weights_proj: fx.f32("idx.weights_proj_w").unwrap().1,
    };
    let mut idx = Indexer::new(DIM, q_lora, h, d, RD, topk, ratio, MAX_SEQ, w, comp);

    let sort = |mut v: Vec<i32>| {
        v.sort_unstable();
        v
    };

    let (_, x) = fx.f32("idx.x_prefill").unwrap();
    let (_, qr) = fx.f32("idx.qr_prefill").unwrap();
    let got = idx.prefill(&x, &qr, PREFILL, &f, 0);
    let (tshape, want) = fx.i32("idx.topk_prefill").unwrap();
    let k = tshape[2];
    for t in 0..PREFILL {
        let w_t = sort(want[t * k..(t + 1) * k].to_vec());
        let g_t = sort(got[t].clone());
        assert_eq!(g_t, w_t, "idx.topk_prefill[{t}]");
    }

    let (_, xd) = fx.f32("idx.x_dec").unwrap();
    let (_, qd) = fx.f32("idx.qr_dec").unwrap();
    let (dshape, want_dec) = fx.i32("idx.topk_dec").unwrap();
    let kd = dshape[1];
    for i in 0..8 {
        let pos = PREFILL + i;
        let got = sort(idx.decode(
            &xd[i * DIM..(i + 1) * DIM],
            &qd[i * q_lora..(i + 1) * q_lora],
            pos,
            &f,
            0,
        ));
        let want = sort(want_dec[i * kd..(i + 1) * kd].to_vec());
        assert_eq!(got, want, "idx.topk_dec[{i}] (pos {pos})");
    }

    let (_, want_cache) = fx.f32("idx.cache_final").unwrap();
    assert_close("idx.cache_final", &idx.kv_cache, &want_cache, 1e-3, 0.02);
}
