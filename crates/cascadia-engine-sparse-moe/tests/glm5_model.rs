//! Full-model greedy parity: the Rust `GlmModel` must generate the exact same
//! token sequence as the CPU reference (`tools/glm5_ref::model_ref`). Token
//! match (argmax) is robust to the bf16-ULP that limits element-wise parity.
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::attn::{AttentionLayer, AttnWeights};
use cascadia_engine_sparse_moe::glm::model::{GlmLayer, GlmModel, LayerMlp};
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

fn bits(f: &[f32]) -> Vec<u16> {
    f.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
}

/// Build the fixture model (dims match tools/glm5_ref/gen_fixtures.py::mcfg).
fn build_model(fx: &StFile, max_seq: usize) -> GlmModel {
    let (vocab, hidden) = (16usize, 32usize);
    let (h, nope, rope, vh, kvl, ql) = (3usize, 6, 4, 6, 8, 16);
    let (n_layers, first_dense, dense_inter) = (3usize, 1usize, 12usize);
    let (n_experts, top_k, moe_inter, scale) = (4usize, 2usize, 10usize, 2.5f32);
    let (eps, theta) = (1e-5f32, 8.0e6f32);

    let load_expert = |p: &str| ExpertW {
        wg: bits(&fx.f32(&format!("{p}.wg")).unwrap().1),
        wu: bits(&fx.f32(&format!("{p}.wu")).unwrap().1),
        wd: bits(&fx.f32(&format!("{p}.wd")).unwrap().1),
    };

    let mut layers = Vec::new();
    for li in 0..n_layers {
        let lp = format!("model.L{li}");
        let aw = AttnWeights {
            wq_a: bits(&fx.f32(&format!("{lp}.attn.wq_a")).unwrap().1),
            q_a_ln: fx.f32(&format!("{lp}.attn.q_a_ln")).unwrap().1,
            wq_b: bits(&fx.f32(&format!("{lp}.attn.wq_b")).unwrap().1),
            wkv_a: bits(&fx.f32(&format!("{lp}.attn.wkv_a")).unwrap().1),
            kv_a_ln: fx.f32(&format!("{lp}.attn.kv_a_ln")).unwrap().1,
            wkv_b: bits(&fx.f32(&format!("{lp}.attn.wkv_b")).unwrap().1),
            wo: bits(&fx.f32(&format!("{lp}.attn.wo")).unwrap().1),
        };
        let freqs = precompute_freqs(rope, max_seq, 0, theta, 1.0, 32.0, 1.0);
        let attn = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, max_seq, aw, freqs);

        let mlp = if li < first_dense {
            LayerMlp::Dense { w: load_expert(&format!("{lp}.dense")).into(), inter: dense_inter }
        } else {
            let experts: Vec<AnyExpert> = (0..n_experts)
                .map(|e| load_expert(&format!("{lp}.moe.e{e}")).into())
                .collect();
            let mw = MoeWeights {
                router_w: fx.f32(&format!("{lp}.moe.router_w")).unwrap().1,
                router_bias: fx.f32(&format!("{lp}.moe.router_bias")).unwrap().1,
                experts,
                shared: load_expert(&format!("{lp}.moe.sh")).into(),
            };
            LayerMlp::Moe(MoeLayer::new(hidden, n_experts, top_k, moe_inter, moe_inter, scale, mw))
        };
        layers.push(GlmLayer::new(
            hidden,
            eps,
            fx.f32(&format!("{lp}.in_ln")).unwrap().1,
            fx.f32(&format!("{lp}.post_ln")).unwrap().1,
            attn,
            mlp,
        ));
    }

    GlmModel::new(
        hidden,
        vocab,
        eps,
        fx.f32("model.embed").unwrap().1,
        layers,
        fx.f32("model.final_norm").unwrap().1,
        fx.f32("model.lm_head").unwrap().1,
    )
}

#[test]
fn model_greedy_matches_reference() {
    let fx = fixtures!();
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let n_gen = 4usize;
    let want: Vec<u32> = vec![0, 12, 3, 2]; // meta.model.greedy
    let mut model = build_model(&fx, prompt.len() + n_gen);
    let got = model.generate(&prompt, n_gen);
    assert_eq!(got, want, "greedy token mismatch");
}

/// Batched prefill must be BIT-IDENTICAL to looping forward_token over the
/// prompt: attention is causal (unchanged per position) and the batch-union
/// MoE is bit-exact, so the last-position logits must match exactly.
#[test]
fn model_prefill_bit_exact_vs_per_token() {
    let fx = fixtures!();
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let mut model = build_model(&fx, prompt.len() + 4);

    model.reset();
    let mut per_token = Vec::new();
    for &t in &prompt {
        per_token = model.forward_token(t);
    }

    model.reset();
    let prefilled = model.prefill(&prompt);

    assert_eq!(prefilled, per_token, "prefill last-position logits diverge from per-token");
}
