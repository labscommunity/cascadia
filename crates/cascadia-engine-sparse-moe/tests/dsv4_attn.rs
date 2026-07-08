//! Golden tests for the assembled dsv4 attention layer against the
//! hook-captured per-layer inputs/outputs of the tiny reference model
//! (all three regimes: window-only, ratio-4 + indexer, ratio-16), across
//! prefill + two decode steps.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::attn_layer::{AttentionLayer, AttnWeights};
use cascadia_engine_sparse_moe::dsv4::compress::{Compressor, CompressorWeights};
use cascadia_engine_sparse_moe::dsv4::indexer::{Indexer, IndexerWeights};
use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;

// tiny model geometry (TINY_KW in export_deepseek_v4.py)
const DIM: usize = 64;
const H: usize = 4;
const HD: usize = 80;
const RD: usize = 16;
const Q_LORA: usize = 32;
const O_LORA: usize = 32;
const GROUPS: usize = 2;
const WIN: usize = 8;
const MAX_SEQ: usize = 64;
const PREFILL: usize = 18;
const IDX_H: usize = 2;
const IDX_D: usize = 32;
const IDX_TOPK: usize = 4;
const RATIOS: [usize; 4] = [0, 4, 16, 4];

fn fixtures() -> StFile {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4/fixtures.safetensors");
    StFile::open(&p).expect("open fixtures (run gen_fixtures.py)")
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

fn build_layer(fx: &StFile, li: usize) -> AttentionLayer {
    let g = |suf: &str| fx.f32(&format!("int.w.layers.{li}.attn.{suf}")).unwrap().1;
    let ratio = RATIOS[li];
    let w = AttnWeights {
        wq_a: g("wq_a.weight"),
        q_norm_w: g("q_norm.weight"),
        wq_b: g("wq_b.weight"),
        wkv: g("wkv.weight"),
        kv_norm_w: g("kv_norm.weight"),
        wo_a: g("wo_a.weight"),
        wo_b: g("wo_b.weight"),
        sink: g("attn_sink"),
    };
    let comp = (ratio > 0).then(|| {
        Compressor::new(
            DIM,
            HD,
            RD,
            ratio,
            false,
            1e-6,
            CompressorWeights {
                wkv: g("compressor.wkv.weight"),
                wgate: g("compressor.wgate.weight"),
                ape: g("compressor.ape"),
                norm_w: g("compressor.norm.weight"),
            },
        )
    });
    let idx = (ratio == 4).then(|| {
        let ic = Compressor::new(
            DIM,
            IDX_D,
            RD,
            ratio,
            true,
            1e-6,
            CompressorWeights {
                wkv: g("indexer.compressor.wkv.weight"),
                wgate: g("indexer.compressor.wgate.weight"),
                ape: g("indexer.compressor.ape"),
                norm_w: g("indexer.compressor.norm.weight"),
            },
        );
        Indexer::new(
            DIM,
            Q_LORA,
            IDX_H,
            IDX_D,
            RD,
            IDX_TOPK,
            ratio,
            MAX_SEQ,
            IndexerWeights {
                wq_b: g("indexer.wq_b.weight"),
                weights_proj: g("indexer.weights_proj.weight"),
            },
            ic,
        )
    });
    // compress layers: compress_rope_theta; window-only: rope_theta.
    let theta = if ratio > 0 { 160000.0 } else { 10000.0 };
    let freqs = precompute_freqs(RD, MAX_SEQ, 0, theta, 16.0, 32.0, 1.0);
    AttentionLayer::new(
        DIM, H, HD, RD, Q_LORA, O_LORA, GROUPS, WIN, ratio, MAX_SEQ, 1e-6, w, comp, idx, freqs,
    )
}

fn run_layer(li: usize, atol: f32, rtol: f32) {
    let fx = fixtures();
    let mut layer = build_layer(&fx, li);

    let (_, x) = fx.f32(&format!("int.L{li}.attn.prefill.in")).unwrap();
    let (_, want) = fx.f32(&format!("int.L{li}.attn.prefill.out")).unwrap();
    let got = layer.prefill(&x, PREFILL);
    assert_close(&format!("L{li}.attn.prefill"), &got, &want, atol, rtol);

    for (step, pos) in [(0usize, PREFILL), (1, PREFILL + 1)] {
        let (_, x) = fx.f32(&format!("int.L{li}.attn.decode{step}.in")).unwrap();
        let (_, want) = fx.f32(&format!("int.L{li}.attn.decode{step}.out")).unwrap();
        let got = layer.decode(&x, pos);
        assert_close(&format!("L{li}.attn.decode{step}"), &got, &want, atol, rtol);
    }
}

#[test]
fn attn_layer0_window_hash_matches_reference() {
    run_layer(0, 2e-3, 0.03);
}

#[test]
fn attn_layer1_ratio4_indexer_matches_reference() {
    run_layer(1, 2e-3, 0.03);
}

#[test]
fn attn_layer2_ratio16_matches_reference() {
    run_layer(2, 2e-3, 0.03);
}

#[test]
fn attn_layer3_ratio4_indexer_matches_reference() {
    run_layer(3, 2e-3, 0.03);
}
