//! GLM-5.2 MTP (multi-token-prediction) head — the DeepSeek-V3 draft chain
//! that makes native speculative decoding possible.
//!
//! Given the true pre-norm hidden `hlast` at the last position and the
//! just-emitted token, it proposes `G` greedy draft tokens:
//!   x  = rmsnorm(embed[tok], enorm)
//!   h  = rmsnorm(g==0 ? final_norm(hlast) : h_prev, hnorm)
//!   hx = eh_proj · [x ; h]                    // [2H] -> [H], the block input
//!   hx = MTP_block(hx)                        // a full transformer layer, own KV
//!   draft = argmax(lm_head · rmsnorm(hx, mtp_norm)); tok = draft; h = hx
//!
//! In deployment the MTP head weights are int8 (int4 collapses acceptance to
//! 0-4%); the shell keeps the same math. `embed`, `final_norm`, and `lm_head`
//! are shared with the model and passed in. Whether MTP speculative decode is a
//! net throughput win on the target hardware is gated by a separate microbench
//! (see docs/architectures/glm5.md); this is the correctness core.

use super::model::GlmLayer;
use crate::dsv4::math::{linear_bf16_w, linear_f32, rmsnorm};

pub struct MtpHead {
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    enorm: Vec<f32>,   // [hidden]
    hnorm: Vec<f32>,   // [hidden]
    mtp_norm: Vec<f32>, // shared_head.norm [hidden]
    eh_proj: Vec<u16>, // [hidden, 2*hidden] (bf16 bits)
    block: GlmLayer,   // the MTP transformer block (attn + MoE)
}

impl MtpHead {
    pub fn new(
        hidden: usize,
        vocab: usize,
        eps: f32,
        enorm: Vec<f32>,
        hnorm: Vec<f32>,
        mtp_norm: Vec<f32>,
        eh_proj: Vec<u16>,
        block: GlmLayer,
    ) -> Self {
        assert_eq!(enorm.len(), hidden);
        assert_eq!(hnorm.len(), hidden);
        assert_eq!(mtp_norm.len(), hidden);
        assert_eq!(eh_proj.len(), hidden * 2 * hidden);
        Self { hidden, vocab, eps, enorm, hnorm, mtp_norm, eh_proj, block }
    }

    /// Propose `g_steps` greedy draft tokens from the true pre-norm hidden
    /// `hlast` and the just-emitted `next_tok`. `embed` `[vocab, hidden]`,
    /// `final_norm` `[hidden]`, `lm_head` `[vocab, hidden]` are the model's
    /// (f32-stored). Resets the block's KV before drafting.
    pub fn draft(
        &mut self,
        hlast: &[f32],
        next_tok: u32,
        g_steps: usize,
        embed: &[f32],
        final_norm: &[f32],
        lm_head: &[f32],
    ) -> Vec<u32> {
        assert_eq!(hlast.len(), self.hidden);
        self.block.reset();
        let hidden = self.hidden;
        let mut h = hlast.to_vec();
        let mut tok = next_tok;
        let mut drafts = Vec::with_capacity(g_steps);
        for g in 0..g_steps {
            let t = tok as usize;
            let mut x = embed[t * hidden..(t + 1) * hidden].to_vec();
            rmsnorm(&mut x, &self.enorm, self.eps);
            if g == 0 {
                rmsnorm(&mut h, final_norm, self.eps);
            }
            rmsnorm(&mut h, &self.hnorm, self.eps);
            let mut cat = vec![0.0f32; 2 * hidden];
            cat[..hidden].copy_from_slice(&x);
            cat[hidden..].copy_from_slice(&h);
            let mut hx = vec![0.0f32; hidden];
            linear_bf16_w(&cat, &self.eh_proj, hidden, 2 * hidden, &mut hx);
            let hx_post = self.block.forward_token(&hx);
            let mut row = hx_post.clone();
            rmsnorm(&mut row, &self.mtp_norm, self.eps);
            let mut logit = vec![0.0f32; self.vocab];
            linear_f32(&row, lm_head, self.vocab, hidden, &mut logit);
            let d = argmax(&logit) as u32;
            drafts.push(d);
            tok = d;
            h = hx_post;
        }
        drafts
    }
}

/// Index of the first maximum (ties -> lowest index, matching `torch.argmax`).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}
