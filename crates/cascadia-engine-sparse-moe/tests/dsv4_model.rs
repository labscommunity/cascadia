//! Phase-2 exit test: the fully assembled Rust dsv4 model runs the tiny
//! reference's prefill + 2 decode steps and must reproduce the greedy
//! tokens, the logits (atol 0.05, rtol 5%), and every per-layer hidden
//! state. Hidden tolerances (5e-2, 10%) are the observed f32-accumulation-
//! order noise envelope (drift compounds ~2x per layer through HC mixing);
//! the discriminating assertions are the logits and the EXACT greedy tokens.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::attn_layer::{AttentionLayer, AttnWeights};
use cascadia_engine_sparse_moe::dsv4::compress::{Compressor, CompressorWeights};
use cascadia_engine_sparse_moe::dsv4::indexer::{Indexer, IndexerWeights};
use cascadia_engine_sparse_moe::dsv4::model::{
    AnyExpert, DsV4Model, Expert, GateWeights, HcParams, Head, Layer, ModelCfg,
};
use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;

const DIM: usize = 64;
const H: usize = 4;
const HD: usize = 80;
const RD: usize = 16;
const Q_LORA: usize = 32;
const O_LORA: usize = 32;
const GROUPS: usize = 2;
const WIN: usize = 8;
const MAX_SEQ: usize = 64;
const IDX_H: usize = 2;
const IDX_D: usize = 32;
const IDX_TOPK: usize = 4;
const RATIOS: [usize; 4] = [0, 4, 16, 4];
const N_EXPERTS: usize = 8;
const PROMPT: [usize; 18] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
];

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

fn build_model(fx: &StFile) -> DsV4Model {
    let g = |name: &str| fx.f32(&format!("int.w.{name}")).unwrap().1;
    let mut layers = Vec::new();
    for li in 0..4 {
        let ga = |suf: &str| g(&format!("layers.{li}.attn.{suf}"));
        // projection weights are stored bf16 (see AttnWeights)
        let gab = |suf: &str| {
            ga(suf)
                .iter()
                .map(|v| half::bf16::from_f32(*v).to_bits())
                .collect::<Vec<u16>>()
        };
        let ratio = RATIOS[li];
        let attn_w = AttnWeights {
            wq_a: gab("wq_a.weight"),
            q_norm_w: ga("q_norm.weight"),
            wq_b: gab("wq_b.weight"),
            wkv: gab("wkv.weight"),
            kv_norm_w: ga("kv_norm.weight"),
            wo_a: gab("wo_a.weight"),
            wo_b: gab("wo_b.weight"),
            sink: ga("attn_sink"),
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
                    wkv: ga("compressor.wkv.weight"),
                    wgate: ga("compressor.wgate.weight"),
                    ape: ga("compressor.ape"),
                    norm_w: ga("compressor.norm.weight"),
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
                    wkv: ga("indexer.compressor.wkv.weight"),
                    wgate: ga("indexer.compressor.wgate.weight"),
                    ape: ga("indexer.compressor.ape"),
                    norm_w: ga("indexer.compressor.norm.weight"),
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
                    wq_b: ga("indexer.wq_b.weight"),
                    weights_proj: ga("indexer.weights_proj.weight"),
                },
                ic,
            )
        });
        let theta = if ratio > 0 { 160000.0 } else { 10000.0 };
        let freqs = precompute_freqs(RD, MAX_SEQ, 0, theta, 16.0, 32.0, 1.0);
        let attn = AttentionLayer::new(
            DIM, H, HD, RD, Q_LORA, O_LORA, GROUPS, WIN, ratio, MAX_SEQ, 1e-6, attn_w, comp, idx,
            freqs,
        );

        let expert = |base: &str| {
            let w2 = g(&format!("{base}.w2.weight"));
            let inter = w2.len() / DIM;
            Expert {
                w1: g(&format!("{base}.w1.weight")),
                w3: g(&format!("{base}.w3.weight")),
                w2,
                inter,
            }
        };
        let experts: Vec<AnyExpert> = (0..N_EXPERTS)
            .map(|e| AnyExpert::Eager(expert(&format!("layers.{li}.ffn.experts.{e}"))))
            .collect();
        let shared = AnyExpert::Eager(expert(&format!("layers.{li}.ffn.shared_experts")));
        let is_hash = li < 1; // tiny: n_hash_layers = 1
        let gate = GateWeights {
            weight: g(&format!("layers.{li}.ffn.gate.weight")),
            bias: (!is_hash).then(|| g(&format!("layers.{li}.ffn.gate.bias"))),
            tid2eid: is_hash.then(|| {
                fx.i32(&format!("int.w.layers.{li}.ffn.gate.tid2eid"))
                    .unwrap()
                    .1
            }),
        };
        layers.push(Layer {
            attn,
            attn_norm_w: g(&format!("layers.{li}.attn_norm.weight")),
            ffn_norm_w: g(&format!("layers.{li}.ffn_norm.weight")),
            gate,
            experts,
            shared,
            hc_attn: HcParams {
                fn_w: g(&format!("layers.{li}.hc_attn_fn")),
                base: g(&format!("layers.{li}.hc_attn_base")),
                scale: g(&format!("layers.{li}.hc_attn_scale")),
            },
            hc_ffn: HcParams {
                fn_w: g(&format!("layers.{li}.hc_ffn_fn")),
                base: g(&format!("layers.{li}.hc_ffn_base")),
                scale: g(&format!("layers.{li}.hc_ffn_scale")),
            },
        });
    }
    DsV4Model {
        cfg: ModelCfg {
            dim: DIM,
            vocab: 128,
            hc: 4,
            sinkhorn_iters: 20,
            hc_eps: 1e-6,
            norm_eps: 1e-6,
            topk: 2,
            route_scale: 1.5,
            swiglu_limit: 10.0,
            n_hash_layers: 1,
        },
        embed: Some(g("embed.weight")),
        layers,
        head: Some(Head {
            norm_w: g("norm.weight"),
            weight: g("head.weight"),
            hc_fn: g("hc_head_fn"),
            hc_base: g("hc_head_base"),
            hc_scale: g("hc_head_scale"),
        }),
        last_hiddens: Vec::new(),
        last_attn: Vec::new(),
        last_moe_in: Vec::new(),
        first_layer: 0,
        ov_experts: None,
    }
}

#[test]
fn full_model_matches_reference_greedy() {
    let fx = fixtures();
    let mut model = build_model(&fx);

    // prefill
    let logits = model.prefill(&PROMPT);
    for li in 0..4 {
        let (_, want) = fx.f32(&format!("int.prefill.h_after_L{li}")).unwrap();
        assert_close(
            &format!("prefill.h_after_L{li}"),
            &model.last_hiddens[li],
            &want,
            5e-2,
            0.10,
        );
    }
    let (_, want_logits) = fx.f32("int.prefill.logits").unwrap();
    assert_close("prefill.logits", &logits, &want_logits, 0.05, 0.05);
    let tok0 = argmax(&logits);

    // decode steps
    let (_, want_d0) = fx.f32("int.decode0.logits").unwrap();
    let logits = model.decode(tok0, PROMPT.len());
    for li in 0..4 {
        let (_, want) = fx.f32(&format!("int.decode0.h_after_L{li}")).unwrap();
        assert_close(
            &format!("decode0.h_after_L{li}"),
            &model.last_hiddens[li],
            &want,
            5e-2,
            0.10,
        );
    }
    assert_close("decode0.logits", &logits, &want_d0, 0.05, 0.05);
    let tok1 = argmax(&logits);

    let (_, want_d1) = fx.f32("int.decode1.logits").unwrap();
    let logits = model.decode(tok1, PROMPT.len() + 1);
    assert_close("decode1.logits", &logits, &want_d1, 0.05, 0.05);
    let tok2 = argmax(&logits);

    // the ultimate bar: greedy tokens byte-match the reference
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/dsv4/fixtures_meta.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let want: Vec<usize> = meta["greedy_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(vec![tok0, tok1, tok2], want, "greedy tokens diverge");
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}
