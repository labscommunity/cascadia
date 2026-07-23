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

/// KV-prefix cache parity: snapshotting the first `k` tokens' KV, restoring it
/// into a reset model, and prefilling only the suffix must be BIT-IDENTICAL to a
/// full prefill of the whole prompt — both the last-position logits and a
/// following decode step. This is the correctness core of prefix caching (skip
/// re-prefilling a shared prompt prefix); the per-rank/cross-rank plumbing is
/// built on top of it.
#[test]
fn prefix_snapshot_restore_bit_exact_vs_full() {
    let fx = fixtures!();
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let k = 2usize; // cached prefix length
    let mut model = build_model(&fx, prompt.len() + 2);

    // Full: prefill the whole prompt, then one fixed decode step at position N.
    model.reset();
    let full = model.prefill(&prompt);
    let full_step = model.forward_token(9);

    // Cached: prefill the prefix, snapshot; reset; restore; prefill the suffix;
    // then the same decode step. The restored KV makes the suffix's attention
    // see the same [0, k) keys, so every downstream logit must match exactly.
    model.reset();
    let _ = model.prefill(&prompt[..k]);
    let snap = model.snapshot_prefix();
    model.reset();
    model.restore_prefix(&snap);
    let reuse = model.prefill(&prompt[k..]);
    let reuse_step = model.forward_token(9);

    assert_eq!(reuse, full, "prefix-cache prefill logits diverge from full prefill");
    assert_eq!(reuse_step, full_step, "decode after prefix restore diverges from full");
}

/// End-to-end KvPrefixCache: warming the cache with a base prompt, then
/// generating for an extension of it, must (a) reuse the whole base prefix and
/// (b) produce tokens bit-identical to an uncached full generation.
#[test]
fn prefix_cache_reuse_matches_uncached() {
    use cascadia_engine_sparse_moe::glm::kv_cache::KvPrefixCache;
    let fx = fixtures!();
    let base: Vec<u32> = vec![1, 2, 3];
    let ext: Vec<u32> = vec![1, 2, 3, 4, 5]; // base is a prefix of ext
    let mut cache = KvPrefixCache::new(4);

    let mut m = build_model(&fx, 16);
    let (_o0, r0) = m.generate_with_prefix_cache(&mut cache, &base, 2);
    assert_eq!(r0, 0, "first call must be a cold miss");

    let (o_cached, r1) = m.generate_with_prefix_cache(&mut cache, &ext, 3);
    assert_eq!(r1, base.len(), "ext should reuse the whole cached base prefix");

    let mut m2 = build_model(&fx, 16);
    let o_ref = m2.generate(&ext, 3);
    assert_eq!(o_cached, o_ref, "prefix-cached generation diverges from uncached");
}
