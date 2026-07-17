//! GLM-5.2 MoE block: `out = Σ_{i∈topk} w_i·expert_i(x) + shared(x)`.
//!
//! Router logits are computed in f32 (no bf16 rounding — a plain
//! f32 `matmul`), scored by the sigmoid + `noaux_tc` gate ([`crate::glm::gate`]),
//! whose weights already carry `routed_scaling_factor`. Each routed expert and
//! the always-on shared expert is a SwiGLU FFN ([`crate::glm::ffn`]).
//! Accumulation order: routed experts in gate order, then the shared
//! expert.
//!
//! The first `first_k_dense_replace` (3) layers are dense (no routing) — that
//! path just calls `ffn::swiglu` directly and does not use this module.

use super::ffn::swiglu;
use super::gate::moe_gate;
use crate::dsv4::math::linear_f32;

/// One expert's SwiGLU weights (bf16 bits).
pub struct ExpertW {
    pub wg: Vec<u16>, // [inter, hidden]
    pub wu: Vec<u16>, // [inter, hidden]
    pub wd: Vec<u16>, // [hidden, inter]
}

pub struct MoeWeights {
    /// Router projection `[n_experts, hidden]`, kept f32 (logits are not
    /// bf16-rounded).
    pub router_w: Vec<f32>,
    /// `e_score_correction_bias` `[n_experts]`.
    pub router_bias: Vec<f32>,
    /// `n_experts` routed experts, each `moe_inter`-wide.
    pub experts: Vec<ExpertW>,
    /// The shared expert (`moe_inter · n_shared`-wide).
    pub shared: ExpertW,
}

pub struct MoeLayer {
    pub hidden: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub scale: f32, // routed_scaling_factor
    w: MoeWeights,
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        n_experts: usize,
        top_k: usize,
        moe_inter: usize,
        shared_inter: usize,
        scale: f32,
        w: MoeWeights,
    ) -> Self {
        assert_eq!(w.router_w.len(), n_experts * hidden);
        assert_eq!(w.router_bias.len(), n_experts);
        assert_eq!(w.experts.len(), n_experts);
        Self {
            hidden,
            n_experts,
            top_k,
            moe_inter,
            shared_inter,
            scale,
            w,
        }
    }

    /// MoE for one token `x` (`[hidden]`). Returns `[hidden]`.
    pub fn forward_token(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        // router logits (f32) -> sigmoid + noaux_tc gate.
        let mut logits = vec![0.0f32; self.n_experts];
        linear_f32(x, &self.w.router_w, self.n_experts, self.hidden, &mut logits);
        let gate = moe_gate(&logits, &self.w.router_bias, self.top_k, self.scale, true);

        // routed experts in gate order, then the shared expert.
        let mut out = vec![0.0f32; self.hidden];
        for (&e, &wj) in gate.idx.iter().zip(&gate.weight) {
            let ex = &self.w.experts[e as usize];
            let y = swiglu(x, &ex.wg, &ex.wu, &ex.wd, self.hidden, self.moe_inter);
            for (o, &yi) in out.iter_mut().zip(&y) {
                *o += wj * yi;
            }
        }
        let sh = &self.w.shared;
        let s = swiglu(x, &sh.wg, &sh.wu, &sh.wd, self.hidden, self.shared_inter);
        for (o, &si) in out.iter_mut().zip(&s) {
            *o += si;
        }
        out
    }
}
