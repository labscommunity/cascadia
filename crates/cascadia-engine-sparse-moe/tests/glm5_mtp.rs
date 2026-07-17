//! Golden test for the GLM-5.2 MTP head draft chain: the Rust `MtpHead` must
//! propose the exact same draft tokens as the CPU reference
//! (`tools/glm5_ref::mtp_draft_ref`). Token-exact (argmax).
//!
//! Regenerate fixtures:
//!   python tools/glm5_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::rope::precompute_freqs;
use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::glm::attn::{AttentionLayer, AttnWeights};
use cascadia_engine_sparse_moe::glm::model::{GlmLayer, LayerMlp};
use cascadia_engine_sparse_moe::glm::moe::{ExpertW, MoeLayer, MoeWeights};
use cascadia_engine_sparse_moe::glm::mtp::MtpHead;

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

#[test]
fn mtp_draft_matches_reference() {
    let fx = fixtures!();
    let (vocab, hidden) = (16usize, 32usize);
    let (h, nope, rope, vh, kvl, ql) = (3usize, 6, 4, 6, 8, 16);
    let (n_experts, top_k, moe_inter, scale) = (4usize, 2usize, 10usize, 2.5f32);
    let (eps, theta) = (1e-5f32, 8.0e6f32);
    let (next_tok, g_steps) = (5u32, 3usize);
    let want: Vec<u32> = vec![11, 6, 0]; // meta.mtp.drafts

    // MTP transformer block (attn + MoE).
    let aw = AttnWeights {
        wq_a: bits(&fx.f32("mtp.block.attn.wq_a").unwrap().1),
        q_a_ln: fx.f32("mtp.block.attn.q_a_ln").unwrap().1,
        wq_b: bits(&fx.f32("mtp.block.attn.wq_b").unwrap().1),
        wkv_a: bits(&fx.f32("mtp.block.attn.wkv_a").unwrap().1),
        kv_a_ln: fx.f32("mtp.block.attn.kv_a_ln").unwrap().1,
        wkv_b: bits(&fx.f32("mtp.block.attn.wkv_b").unwrap().1),
        wo: bits(&fx.f32("mtp.block.attn.wo").unwrap().1),
    };
    let freqs = precompute_freqs(rope, g_steps, 0, theta, 1.0, 32.0, 1.0);
    let attn = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, g_steps, eps, aw, freqs);

    let load_expert = |p: &str| ExpertW {
        wg: bits(&fx.f32(&format!("{p}.wg")).unwrap().1),
        wu: bits(&fx.f32(&format!("{p}.wu")).unwrap().1),
        wd: bits(&fx.f32(&format!("{p}.wd")).unwrap().1),
    };
    let experts: Vec<ExpertW> = (0..n_experts)
        .map(|e| load_expert(&format!("mtp.block.moe.e{e}")))
        .collect();
    let mw = MoeWeights {
        router_w: fx.f32("mtp.block.moe.router_w").unwrap().1,
        router_bias: fx.f32("mtp.block.moe.router_bias").unwrap().1,
        experts,
        shared: load_expert("mtp.block.moe.sh"),
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

    let mut mtp = MtpHead::new(
        hidden,
        vocab,
        eps,
        fx.f32("mtp.enorm").unwrap().1,
        fx.f32("mtp.hnorm").unwrap().1,
        fx.f32("mtp.mtp_norm").unwrap().1,
        bits(&fx.f32("mtp.eh_proj").unwrap().1),
        block,
    );

    let hlast = fx.f32("mtp.hlast").unwrap().1;
    let embed = fx.f32("mtp.embed").unwrap().1;
    let final_norm = fx.f32("mtp.final_norm").unwrap().1;
    let lm_head = fx.f32("mtp.lm_head").unwrap().1;

    let got = mtp.draft(&hlast, next_tok, g_steps, &embed, &final_norm, &lm_head);
    assert_eq!(got, want, "MTP draft token mismatch");
}
