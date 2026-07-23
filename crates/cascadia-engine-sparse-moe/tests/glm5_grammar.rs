//! Grammar-constrained decoding with forced-run batching. A template grammar
//! forces a run, then a free position, then another run. `generate_grammar` must
//! emit the exact same tokens as a naive per-token constrained decode (forced
//! tokens are mandated; free tokens are masked-argmax on identical logits), but
//! spend FEWER model forwards — the forced runs are advanced in one batch each.
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::attn::{AttentionLayer, AttnWeights};
use cascadia_engine_sparse_moe::glm::grammar::{masked_argmax, Grammar};
use cascadia_engine_sparse_moe::glm::model::{GlmLayer, GlmModel, LayerMlp};
use cascadia_engine_sparse_moe::glm::moe::{AnyExpert, ExpertW, MoeLayer, MoeWeights};

macro_rules! fixtures {
    () => {{
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/glm5/fixtures.safetensors");
        if !p.exists() {
            eprintln!(
                "SKIP: {} absent (run tools/glm5_ref/gen_fixtures.py)",
                p.display()
            );
            return;
        }
        StFile::open(&p).expect("open fixtures")
    }};
}

fn bits(f: &[f32]) -> Vec<u16> {
    f.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
}

/// Same fixture model as glm5_model (dims from gen_fixtures.py::mcfg).
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
            LayerMlp::Dense {
                w: load_expert(&format!("{lp}.dense")).into(),
                inter: dense_inter,
            }
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
            LayerMlp::Moe(MoeLayer::new(
                hidden, n_experts, top_k, moe_inter, moe_inter, scale, mw,
            ))
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

/// Template over the tiny vocab (16): forced [3,5,9], one FREE token (∈{1,8,14}),
/// forced [2,2], then accept. Positions are keyed by how many tokens are out.
struct Template;
const RUN1: [u32; 3] = [3, 5, 9];
const RUN2: [u32; 2] = [2, 2];
const FREE_SET: [u32; 3] = [1, 8, 14];

impl Grammar for Template {
    fn forced_run(&self, emitted: &[u32]) -> Vec<u32> {
        let n = emitted.len();
        if n < 3 {
            RUN1[n..].to_vec() // remaining of the first forced run
        } else if n == 3 {
            Vec::new() // the free position
        } else if n < 6 {
            RUN2[n - 4..].to_vec() // remaining of the second forced run
        } else {
            Vec::new()
        }
    }
    fn allows(&self, emitted: &[u32], token: u32) -> bool {
        if emitted.len() == 3 {
            FREE_SET.contains(&token)
        } else {
            true
        }
    }
    fn can_end(&self, emitted: &[u32]) -> bool {
        emitted.len() >= 6
    }
}

/// Naive constrained decode: one model forward per emitted token (forced or free).
fn naive_constrained(
    model: &mut GlmModel,
    prompt: &[u32],
    g: &Template,
    max_new: usize,
) -> Vec<u32> {
    model.reset();
    let mut logits = Vec::new();
    for &t in prompt {
        logits = model.forward_token(t);
    }
    let mut out = Vec::new();
    while out.len() < max_new {
        let forced = g.forced_run(&out);
        if !forced.is_empty() {
            let tok = forced[0];
            out.push(tok);
            logits = model.forward_token(tok);
        } else if g.can_end(&out) {
            break;
        } else {
            let tok = masked_argmax(&logits, g, &out);
            out.push(tok);
            logits = model.forward_token(tok);
        }
    }
    out
}

#[test]
fn grammar_forced_run_batching_matches_naive() {
    let fx = fixtures!();
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let max_new = 10usize;
    let mut model = build_model(&fx, prompt.len() + max_new);

    let g = Template;
    let out = model.generate_grammar(&prompt, &g, max_new);
    let naive = naive_constrained(&mut model, &prompt, &g, max_new);

    // Same tokens as per-token constrained decode, and the template structure.
    assert_eq!(
        out.tokens, naive,
        "grammar generation diverged from naive decode"
    );
    assert_eq!(out.tokens.len(), 6, "template should emit 6 tokens");
    assert_eq!(&out.tokens[0..3], &RUN1, "first forced run");
    assert!(
        FREE_SET.contains(&out.tokens[3]),
        "free token must be in the allowed set"
    );
    assert_eq!(&out.tokens[4..6], &RUN2, "second forced run");

    // The win: forced runs batched → fewer forwards than tokens. Here:
    // 1 prefill + [3,5,9] + free + [2,2] = 4 forwards for 6 tokens.
    assert!(
        out.forwards < out.tokens.len(),
        "expected forced-run batching to save forwards: {} forwards for {} tokens",
        out.forwards,
        out.tokens.len()
    );
    assert_eq!(
        out.forwards, 4,
        "1 prefill + 2 forced runs + 1 free position"
    );
}
