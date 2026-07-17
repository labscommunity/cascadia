//! GLM-5.2 transformer layer + (later) full model assembly.
//!
//! Layer forward is pre-norm:
//!   h   = x + attention(rmsnorm(x, input_layernorm))
//!   out = h + mlp(rmsnorm(h, post_attention_layernorm))
//! where `mlp` is the MoE block on sparse layers or a dense SwiGLU on the first
//! `first_k_dense_replace` (3) layers. Both norms are RMSNorm, eps
//! `rms_norm_eps` (1e-5).

use super::attn::AttentionLayer;
use super::ffn::swiglu;
use super::moe::{ExpertW, MoeLayer};
use crate::dsv4::math::rmsnorm;

/// The per-layer feed-forward: routed MoE, or a dense SwiGLU (first-k layers).
pub enum LayerMlp {
    Moe(MoeLayer),
    Dense { w: ExpertW, inter: usize },
}

pub struct GlmLayer {
    pub hidden: usize,
    pub eps: f32,
    in_ln: Vec<f32>,   // input_layernorm weight [hidden]
    post_ln: Vec<f32>, // post_attention_layernorm weight [hidden]
    attn: AttentionLayer,
    mlp: LayerMlp,
}

impl GlmLayer {
    pub fn new(
        hidden: usize,
        eps: f32,
        in_ln: Vec<f32>,
        post_ln: Vec<f32>,
        attn: AttentionLayer,
        mlp: LayerMlp,
    ) -> Self {
        assert_eq!(in_ln.len(), hidden);
        assert_eq!(post_ln.len(), hidden);
        Self { hidden, eps, in_ln, post_ln, attn, mlp }
    }

    /// Clear the attention KV cache (new sequence).
    pub fn reset(&mut self) {
        self.attn.reset();
    }

    /// Process one token at the next cached position. `x` is the residual-stream
    /// hidden `[hidden]`; returns the updated hidden after this layer.
    pub fn forward_token(&mut self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        // h = x + attn(rmsnorm(x, in_ln))
        let mut nrm = x.to_vec();
        rmsnorm(&mut nrm, &self.in_ln, self.eps);
        let a = self.attn.forward_token(&nrm);
        let mut h: Vec<f32> = x.iter().zip(&a).map(|(&xi, &ai)| xi + ai).collect();
        // out = h + mlp(rmsnorm(h, post_ln))
        let mut nrm2 = h.clone();
        rmsnorm(&mut nrm2, &self.post_ln, self.eps);
        let f = match &self.mlp {
            LayerMlp::Moe(m) => m.forward_token(&nrm2),
            LayerMlp::Dense { w, inter } => {
                swiglu(&nrm2, &w.wg, &w.wu, &w.wd, self.hidden, *inter)
            }
        };
        for (hi, &fi) in h.iter_mut().zip(&f) {
            *hi += fi;
        }
        h
    }
}
