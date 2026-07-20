//! MTP speculative-decode correctness: `GlmModel::generate_spec` must produce a
//! token-for-token IDENTICAL sequence to plain greedy `generate`, for any draft
//! window `g`. Speculative decode only changes *how many* forward passes run —
//! never the output — because every committed token is either the model's own
//! greedy argmax or an accepted draft that equals it. The first mismatch's KV is
//! rewound (`truncate`), so the accepted-prefix cache is exactly what sequential
//! greedy would have built.
//!
//! Two tests:
//!   1. `spec_matches_greedy` — the fixture model (varied weights): the MTP block
//!      differs from the model layers, so most drafts are rejected. Exercises the
//!      reject + KV-rewind path across g ∈ {0,1,2,3,4}.
//!   2. `full_acceptance_vocab1` — a hand-built vocab-1 model: every prediction is
//!      token 0, so every draft is accepted. Exercises the accept + bonus-token
//!      path (n_acc == g, no rewind) deterministically.
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
use cascadia_engine_sparse_moe::glm::mtp::MtpHead;

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

/// Build the fixture MTP head. `block_max_seq` sizes the MTP block's KV (it
/// resets and consumes ≤ g positions per draft round).
fn build_mtp(fx: &StFile, block_max_seq: usize) -> MtpHead {
    let (vocab, hidden) = (16usize, 32usize);
    let (h, nope, rope, vh, kvl, ql) = (3usize, 6, 4, 6, 8, 16);
    let (n_experts, top_k, moe_inter, scale) = (4usize, 2usize, 10usize, 2.5f32);
    let (eps, theta) = (1e-5f32, 8.0e6f32);

    let aw = AttnWeights {
        wq_a: bits(&fx.f32("mtp.block.attn.wq_a").unwrap().1),
        q_a_ln: fx.f32("mtp.block.attn.q_a_ln").unwrap().1,
        wq_b: bits(&fx.f32("mtp.block.attn.wq_b").unwrap().1),
        wkv_a: bits(&fx.f32("mtp.block.attn.wkv_a").unwrap().1),
        kv_a_ln: fx.f32("mtp.block.attn.kv_a_ln").unwrap().1,
        wkv_b: bits(&fx.f32("mtp.block.attn.wkv_b").unwrap().1),
        wo: bits(&fx.f32("mtp.block.attn.wo").unwrap().1),
    };
    let freqs = precompute_freqs(rope, block_max_seq, 0, theta, 1.0, 32.0, 1.0);
    let attn = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, block_max_seq, aw, freqs);

    let load_expert = |p: &str| ExpertW {
        wg: bits(&fx.f32(&format!("{p}.wg")).unwrap().1),
        wu: bits(&fx.f32(&format!("{p}.wu")).unwrap().1),
        wd: bits(&fx.f32(&format!("{p}.wd")).unwrap().1),
    };
    let experts: Vec<AnyExpert> = (0..n_experts)
        .map(|e| load_expert(&format!("mtp.block.moe.e{e}")).into())
        .collect();
    let mw = MoeWeights {
        router_w: fx.f32("mtp.block.moe.router_w").unwrap().1,
        router_bias: fx.f32("mtp.block.moe.router_bias").unwrap().1,
        experts,
        shared: load_expert("mtp.block.moe.sh").into(),
    };
    let moe = MoeLayer::new(hidden, n_experts, top_k, moe_inter, moe_inter, scale, mw);
    let block = GlmLayer::new(
        hidden,
        eps,
        fx.f32("mtp.block.in_ln").unwrap().1,
        fx.f32("mtp.block.post_ln").unwrap().1,
        attn,
        LayerMlp::Moe(moe),
    );

    MtpHead::new(
        hidden,
        vocab,
        eps,
        fx.f32("mtp.enorm").unwrap().1,
        fx.f32("mtp.hnorm").unwrap().1,
        fx.f32("mtp.mtp_norm").unwrap().1,
        bits(&fx.f32("mtp.eh_proj").unwrap().1),
        block,
    )
}

#[test]
fn spec_matches_greedy() {
    let fx = fixtures!();
    let prompt: Vec<u32> = vec![1, 2, 3, 4];
    let n_gen = 8usize;
    const G_MAX: usize = 4;
    // Transient KV peak during a verify round is (prompt+committed) + g positions.
    let max_seq = prompt.len() + n_gen + G_MAX + 2;

    // Ground truth: plain greedy.
    let want = build_model(&fx, max_seq).generate(&prompt, n_gen);
    assert_eq!(want.len(), n_gen);

    for g in [0usize, 1, 2, 3, 4] {
        let mut model = build_model(&fx, max_seq);
        model.set_mtp(build_mtp(&fx, G_MAX.max(1)));
        let out = model.generate_spec(&prompt, n_gen, g);
        assert_eq!(out.tokens, want, "spec (g={g}) diverged from greedy");
        assert_eq!(out.tokens.len(), n_gen);
        // A verify round consumes ≥1 committed token, so forwards ≤ n_gen + 1
        // (prefill), and equals that only when nothing is accepted.
        assert!(
            out.forwards <= n_gen + 1,
            "g={g}: {} forwards",
            out.forwards
        );
        assert!(out.accepted <= out.drafted, "g={g}: accepted > drafted");
        if g == 0 {
            assert_eq!(out.drafted, 0, "g=0 must not draft");
        }
        eprintln!(
            "g={g}: forwards={} drafted={} accepted={} ({:.0}% accept)",
            out.forwards,
            out.drafted,
            out.accepted,
            if out.drafted > 0 {
                100.0 * out.accepted as f32 / out.drafted as f32
            } else {
                0.0
            }
        );
    }
}

fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}
fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

/// A degenerate vocab-1 model: `argmax` of a length-1 logit vector is always
/// token 0, so both the MTP drafts and the model's verify predictions are always
/// 0 — every draft is accepted (n_acc == g each round). All linear weights are
/// zero (rmsnorm's eps keeps it NaN-free); only the norm weights are 1.
fn build_vocab1(hidden: usize, max_seq: usize, block_max_seq: usize) -> GlmModel {
    let (h, nope, rope, vh, kvl, ql) = (1usize, 2, 2, 2, 2, 2);
    let qk_head = nope + rope;
    let kv_out = nope + vh;
    let inter = 4usize;
    let (eps, theta) = (1e-5f32, 8.0e6f32);
    let vocab = 1usize;

    let mk_attn = |ms: usize| {
        let aw = AttnWeights {
            wq_a: bits(&zeros(ql * hidden)),
            q_a_ln: ones(ql),
            wq_b: bits(&zeros(h * qk_head * ql)),
            wkv_a: bits(&zeros((kvl + rope) * hidden)),
            kv_a_ln: ones(kvl),
            wkv_b: bits(&zeros(h * kv_out * kvl)),
            wo: bits(&zeros(hidden * h * vh)),
        };
        let freqs = precompute_freqs(rope, ms, 0, theta, 1.0, 32.0, 1.0);
        AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, ms, aw, freqs)
    };
    let mk_dense = || LayerMlp::Dense {
        w: ExpertW {
            wg: bits(&zeros(inter * hidden)),
            wu: bits(&zeros(inter * hidden)),
            wd: bits(&zeros(hidden * inter)),
        }
        .into(),
        inter,
    };

    let layer = GlmLayer::new(
        hidden,
        eps,
        ones(hidden),
        ones(hidden),
        mk_attn(max_seq),
        mk_dense(),
    );
    let mut model = GlmModel::new(
        hidden,
        vocab,
        eps,
        zeros(vocab * hidden), // embed
        vec![layer],
        ones(hidden),          // final_norm
        zeros(vocab * hidden), // lm_head
    );

    let block = GlmLayer::new(
        hidden,
        eps,
        ones(hidden),
        ones(hidden),
        mk_attn(block_max_seq),
        mk_dense(),
    );
    let mtp = MtpHead::new(
        hidden,
        vocab,
        eps,
        ones(hidden),                      // enorm
        ones(hidden),                      // hnorm
        ones(hidden),                      // mtp_norm
        bits(&zeros(hidden * 2 * hidden)), // eh_proj
        block,
    );
    model.set_mtp(mtp);
    model
}

#[test]
fn full_acceptance_vocab1() {
    let (hidden, n_gen, g) = (4usize, 6usize, 3usize);
    let max_seq = 1 + n_gen + g + 2;
    let mut model = build_vocab1(hidden, max_seq, g);

    let out = model.generate_spec(&[0], n_gen, g);
    assert_eq!(out.tokens, vec![0u32; n_gen], "vocab-1 must emit token 0");
    // Every draft accepted ⇒ each verify round commits g+1 tokens, so far fewer
    // forwards than tokens, and accepted == drafted.
    assert_eq!(out.accepted, out.drafted, "vocab-1 must accept every draft");
    assert!(out.drafted > 0, "should have drafted");
    assert!(
        out.forwards < n_gen,
        "full acceptance ⇒ forwards ({}) < n_gen ({n_gen})",
        out.forwards
    );
    // Plain greedy agrees.
    let plain = build_vocab1(hidden, max_seq, g).generate(&[0], n_gen);
    assert_eq!(plain, vec![0u32; n_gen]);
    eprintln!(
        "vocab1: forwards={} drafted={} accepted={}",
        out.forwards, out.drafted, out.accepted
    );
}
