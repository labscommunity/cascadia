//! Golden test for a full GLM-5.2 transformer layer (norms + attention +
//! residual + MoE + residual) against the CPU reference
//! (`tools/glm5_ref::layer_ref`).
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::math::rmsnorm;
use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::attn::{AttentionLayer, AttnWeights};
use cascadia_engine_sparse_moe::glm::model::{GlmLayer, LayerMlp};
use cascadia_engine_sparse_moe::glm::moe::{AnyExpert, ExpertW, MoeLayer, MoeWeights};

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
fn layer_attn_stage_matches_reference() {
    let fx = fixtures!();
    let (hidden, h, nope, rope, vh, kvl, ql) = (32usize, 3, 6, 4, 6, 8, 16);
    let (eps, theta) = (1e-5f32, 8.0e6f32);
    let (xshape, x) = fx.f32("layer.x").unwrap();
    let seq = xshape[0];
    let aw = AttnWeights {
        wq_a: bits(&fx.f32("layer.attn.wq_a").unwrap().1),
        q_a_ln: fx.f32("layer.attn.q_a_ln").unwrap().1,
        wq_b: bits(&fx.f32("layer.attn.wq_b").unwrap().1),
        wkv_a: bits(&fx.f32("layer.attn.wkv_a").unwrap().1),
        kv_a_ln: fx.f32("layer.attn.kv_a_ln").unwrap().1,
        wkv_b: bits(&fx.f32("layer.attn.wkv_b").unwrap().1),
        wo: bits(&fx.f32("layer.attn.wo").unwrap().1),
    };
    let freqs = precompute_freqs(rope, seq, 0, theta, 1.0, 32.0, 1.0);
    let mut attn = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, seq, aw, freqs);
    let in_ln = fx.f32("layer.in_ln").unwrap().1;
    let (_, want) = fx.f32("layer.h_mid").unwrap();
    let mut got = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let xs = &x[s * hidden..(s + 1) * hidden];
        let mut nrm = xs.to_vec();
        rmsnorm(&mut nrm, &in_ln, eps);
        let a = attn.forward_token(&nrm);
        for j in 0..hidden {
            got[s * hidden + j] = xs[j] + a[j];
        }
    }
    assert_close("layer.h_mid", &got, &want, 1e-2, 1e-2);
}

#[test]
fn layer_moe_stage_on_ref_h_matches_reference() {
    // Feed the REFERENCE post-attention h into the Rust MoE stage. Isolates MoE
    // correctness from attention's bf16-ULP propagation: if this matches tightly
    // the MoE wiring is correct and the end-to-end layer diff is pure ULP.
    let fx = fixtures!();
    let (hidden, n_experts, top_k, moe_inter, scale) = (32usize, 4usize, 2usize, 10usize, 2.5f32);
    let eps = 1e-5f32;
    let (hshape, hmid) = fx.f32("layer.h_mid").unwrap();
    let (_, out_want) = fx.f32("layer.out").unwrap();
    let seq = hshape[0];

    let load_expert = |p: &str| ExpertW {
        wg: bits(&fx.f32(&format!("{p}.wg")).unwrap().1),
        wu: bits(&fx.f32(&format!("{p}.wu")).unwrap().1),
        wd: bits(&fx.f32(&format!("{p}.wd")).unwrap().1),
    };
    let experts: Vec<AnyExpert> = (0..n_experts)
        .map(|e| load_expert(&format!("layer.moe.e{e}")).into())
        .collect();
    let mw = MoeWeights {
        router_w: fx.f32("layer.moe.router_w").unwrap().1,
        router_bias: fx.f32("layer.moe.router_bias").unwrap().1,
        experts,
        shared: load_expert("layer.moe.sh").into(),
    };
    let moe = MoeLayer::new(hidden, n_experts, top_k, moe_inter, moe_inter, scale, mw);
    let post_ln = fx.f32("layer.post_ln").unwrap().1;

    let mut got = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let hs = &hmid[s * hidden..(s + 1) * hidden];
        let mut nrm2 = hs.to_vec();
        rmsnorm(&mut nrm2, &post_ln, eps);
        let f = moe.forward_token(&nrm2);
        for j in 0..hidden {
            got[s * hidden + j] = hs[j] + f[j];
        }
    }
    assert_close("layer.moe_on_ref_h", &got, &out_want, 2e-3, 1e-2);
}

#[test]
fn transformer_layer_matches_reference() {
    let fx = fixtures!();
    // dims match tools/glm5_ref/gen_fixtures.py::lcfg
    let (hidden, h, nope, rope, vh, kvl, ql) = (32usize, 3, 6, 4, 6, 8, 16);
    let (n_experts, top_k, moe_inter, scale) = (4usize, 2usize, 10usize, 2.5f32);
    let (eps, theta) = (1e-5f32, 8.0e6f32);
    let (xshape, x) = fx.f32("layer.x").unwrap();
    let seq = xshape[0];

    let aw = AttnWeights {
        wq_a: bits(&fx.f32("layer.attn.wq_a").unwrap().1),
        q_a_ln: fx.f32("layer.attn.q_a_ln").unwrap().1,
        wq_b: bits(&fx.f32("layer.attn.wq_b").unwrap().1),
        wkv_a: bits(&fx.f32("layer.attn.wkv_a").unwrap().1),
        kv_a_ln: fx.f32("layer.attn.kv_a_ln").unwrap().1,
        wkv_b: bits(&fx.f32("layer.attn.wkv_b").unwrap().1),
        wo: bits(&fx.f32("layer.attn.wo").unwrap().1),
    };
    let freqs = precompute_freqs(rope, seq, 0, theta, 1.0, 32.0, 1.0);
    let attn = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, seq, aw, freqs);

    let load_expert = |p: &str| ExpertW {
        wg: bits(&fx.f32(&format!("{p}.wg")).unwrap().1),
        wu: bits(&fx.f32(&format!("{p}.wu")).unwrap().1),
        wd: bits(&fx.f32(&format!("{p}.wd")).unwrap().1),
    };
    let experts: Vec<AnyExpert> = (0..n_experts)
        .map(|e| load_expert(&format!("layer.moe.e{e}")).into())
        .collect();
    let mw = MoeWeights {
        router_w: fx.f32("layer.moe.router_w").unwrap().1,
        router_bias: fx.f32("layer.moe.router_bias").unwrap().1,
        experts,
        shared: load_expert("layer.moe.sh").into(),
    };
    let moe = MoeLayer::new(hidden, n_experts, top_k, moe_inter, moe_inter, scale, mw);

    let mut layer = GlmLayer::new(
        hidden,
        eps,
        fx.f32("layer.in_ln").unwrap().1,
        fx.f32("layer.post_ln").unwrap().1,
        attn,
        LayerMlp::Moe(moe),
    );

    let (_, want) = fx.f32("layer.out").unwrap();
    let mut got = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let o = layer.forward_token(&x[s * hidden..(s + 1) * hidden]);
        got[s * hidden..(s + 1) * hidden].copy_from_slice(&o);
    }
    // Propagation-bounded tolerance: attention's f32 accumulation-order makes h
    // differ from the reference by ~1 bf16 ULP (~0.008 at O(1)); rmsnorm -> MoE
    // amplifies that via the 2.5x routed_scaling_factor over top-k + shared, so
    // the end-to-end element bound is ~3e-2. The tight guards are the two stage
    // tests above (attn h within 1e-2, MoE-on-ref-h within 2e-3); model-level
    // parity (later) uses greedy-token-exact, which is robust to this ULP.
    assert_close("layer.out", &got, &want, 4e-2, 1e-2);
}
