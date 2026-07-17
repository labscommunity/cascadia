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

use super::ffn::{swiglu, swiglu_f32w, swiglu_mmap};
use super::gate::moe_gate;
use crate::dsv4::expert_mmap::MmapExpert;
use crate::dsv4::math::linear_f32;

/// One expert's SwiGLU weights (bf16 bits) — the synthetic-golden / shell path.
pub struct ExpertW {
    pub wg: Vec<u16>, // [inter, hidden]
    pub wu: Vec<u16>, // [inter, hidden]
    pub wd: Vec<u16>, // [hidden, inter]
}

/// How an expert's weights are held, per its numeric contract:
/// - `Bf16`: bf16-bit weights (goldens; the shell's native dtype).
/// - `EagerF32`: int4-dequantized f32 weights (the on-disk `int4_bin` path).
///   int4 values are not exactly bf16-representable, so they stay f32 and run
///   through [`swiglu_f32w`] (same op order / bf16 activation boundaries as
///   `swiglu`, only the weight dtype differs).
pub enum AnyExpert {
    Bf16(ExpertW),
    EagerF32 { wg: Vec<f32>, wu: Vec<f32>, wd: Vec<f32> },
    /// mmap'd int4 bin, rows dequantized on the fly — the only mode that fits
    /// the real model (eager f32 experts would be hundreds of GB per rank).
    Mmap(MmapExpert),
}

impl AnyExpert {
    /// One expert's SwiGLU FFN for token `x`. `inter` is this expert's
    /// intermediate width (routed = `moe_inter`, shared = `moe_inter·n_shared`).
    pub fn forward(&self, x: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
        match self {
            AnyExpert::Bf16(e) => swiglu(x, &e.wg, &e.wu, &e.wd, hidden, inter),
            AnyExpert::EagerF32 { wg, wu, wd } => swiglu_f32w(x, wg, wu, wd, hidden, inter),
            AnyExpert::Mmap(m) => swiglu_mmap(m, x),
        }
    }
}

impl From<ExpertW> for AnyExpert {
    fn from(e: ExpertW) -> Self {
        AnyExpert::Bf16(e)
    }
}

pub struct MoeWeights {
    /// Router projection `[n_experts, hidden]`, kept f32 (logits are not
    /// bf16-rounded).
    pub router_w: Vec<f32>,
    /// `e_score_correction_bias` `[n_experts]`.
    pub router_bias: Vec<f32>,
    /// `n_experts` routed experts, each `moe_inter`-wide.
    pub experts: Vec<AnyExpert>,
    /// The shared expert (`moe_inter · n_shared`-wide).
    pub shared: AnyExpert,
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
            let y = self.w.experts[e as usize].forward(x, self.hidden, self.moe_inter);
            for (o, &yi) in out.iter_mut().zip(&y) {
                *o += wj * yi;
            }
        }
        let s = self.w.shared.forward(x, self.hidden, self.shared_inter);
        for (o, &si) in out.iter_mut().zip(&s) {
            *o += si;
        }
        out
    }
}
