//! LatentMoE block — K3's routed experts run in a 3584-dim latent, not in the
//! 7168 residual stream.
//!
//! ```text
//! idx, w = gate(x)                    // gate reads HIDDEN, not the latent
//! x_lat  = routed_expert_down_proj(x) // 7168 -> 3584
//! y      = sum_k w_k * expert_k(x_lat)
//! y      = routed_expert_norm(y)      // RMSNorm on the COMBINED output
//! y      = routed_expert_up_proj(y)   // 3584 -> 7168
//! out    = y + shared_experts(x)      // shared run on HIDDEN, inter = moe_inter * n_shared
//! ```
//!
//! The router is [`crate::glm::gate::moe_gate`] unchanged — K3's `KimiMoEGate`
//! has identical semantics (sigmoid scoring, `noaux_tc` selection bias,
//! norm-topk, then `* routed_scaling_factor`, which is 1.0 for K3).

use crate::dsv4::math::{linear_bf16_w, rmsnorm, to_bf16};
use crate::glm::gate::moe_gate;
use crate::k3::expert_fp4;
use crate::k3::situ::situ;

/// Shape contract for one LatentMoE layer.
#[derive(Clone, Copy, Debug)]
pub struct MoeDims {
    pub hidden: usize,
    pub latent: usize,
    pub inter: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub scale: f32,
    pub renormalize: bool,
    pub situ_beta: f32,
    pub situ_linear_beta: Option<f32>,
    pub eps: f32,
    pub use_norm: bool,
}

/// Per-layer non-expert weights. Experts live in their own fp4 blobs.
pub struct MoeWeights {
    pub gate: Vec<f32>,
    pub e_score_correction_bias: Vec<f32>,
    pub down_proj: Vec<u16>,
    pub up_proj: Vec<u16>,
    pub norm: Vec<f32>,
    pub shared_w1: Vec<u16>,
    pub shared_w3: Vec<u16>,
    pub shared_w2: Vec<u16>,
}

/// Source of one expert's packed fp4 bytes (`w1`, `w3`, `w2` back to back).
pub trait ExpertSource {
    fn expert_bytes(&self, expert: usize) -> &[u8];
}

/// A flat in-memory expert set — `n_experts * expert_bytes(latent, inter)`.
pub struct FlatExperts {
    pub data: Vec<u8>,
    pub stride: usize,
}

impl ExpertSource for FlatExperts {
    fn expert_bytes(&self, expert: usize) -> &[u8] {
        &self.data[expert * self.stride..(expert + 1) * self.stride]
    }
}

/// SiTU FFN over fp4-packed weights: `w2(SiTU(w1(x), w3(x)))`.
fn fp4_expert_forward(bytes: &[u8], x: &[f32], d: MoeDims, out: &mut [f32]) {
    let sec_gate = expert_fp4::section_bytes(d.inter, d.latent);
    let sec_down = expert_fp4::section_bytes(d.latent, d.inter);
    debug_assert_eq!(bytes.len(), 2 * sec_gate + sec_down);

    let mut g = vec![0.0f32; d.inter];
    let mut u = vec![0.0f32; d.inter];
    expert_fp4::gemv(&bytes[..sec_gate], d.inter, d.latent, x, &mut g);
    expert_fp4::gemv(&bytes[sec_gate..2 * sec_gate], d.inter, d.latent, x, &mut u);

    let mut h = vec![0.0f32; d.inter];
    situ(&g, &u, &mut h, d.situ_beta, d.situ_linear_beta);
    for v in h.iter_mut() {
        *v = to_bf16(*v);
    }
    expert_fp4::gemv(&bytes[2 * sec_gate..], d.latent, d.inter, &h, out);
}

/// One token through the LatentMoE block. `x`, `out`: `[hidden]`.
pub fn moe_forward<E: ExpertSource>(
    x: &[f32],
    w: &MoeWeights,
    d: MoeDims,
    experts: &E,
    out: &mut [f32],
) {
    // router reads the HIDDEN stream
    let mut logits = vec![0.0f32; d.n_experts];
    for (lg, row) in logits.iter_mut().zip(w.gate.chunks_exact(d.hidden)) {
        *lg = row.iter().zip(x).map(|(&a, &b)| a * b).sum();
    }
    let sel = moe_gate(
        &logits,
        &w.e_score_correction_bias,
        d.top_k,
        d.scale,
        d.renormalize,
    );

    // down-project once, then accumulate the selected experts in latent space
    let mut x_lat = vec![0.0f32; d.latent];
    linear_bf16_w(x, &w.down_proj, d.latent, d.hidden, &mut x_lat);

    let mut acc = vec![0.0f32; d.latent];
    let mut eo = vec![0.0f32; d.latent];
    for (i, &e) in sel.idx.iter().enumerate() {
        fp4_expert_forward(experts.expert_bytes(e as usize), &x_lat, d, &mut eo);
        let wt = sel.weight[i];
        for (a, &v) in acc.iter_mut().zip(eo.iter()) {
            *a += wt * v;
        }
    }
    for v in acc.iter_mut() {
        *v = to_bf16(*v);
    }
    if d.use_norm {
        rmsnorm(&mut acc, &w.norm, d.eps);
    }
    linear_bf16_w(&acc, &w.up_proj, d.hidden, d.latent, out);

    // shared experts run on the HIDDEN stream, width moe_inter * n_shared
    if d.n_shared > 0 {
        let si = d.inter * d.n_shared;
        let mut g = vec![0.0f32; si];
        let mut u = vec![0.0f32; si];
        linear_bf16_w(x, &w.shared_w1, si, d.hidden, &mut g);
        linear_bf16_w(x, &w.shared_w3, si, d.hidden, &mut u);
        let mut h = vec![0.0f32; si];
        situ(&g, &u, &mut h, d.situ_beta, d.situ_linear_beta);
        for v in h.iter_mut() {
            *v = to_bf16(*v);
        }
        let mut sh = vec![0.0f32; d.hidden];
        linear_bf16_w(&h, &w.shared_w2, d.hidden, si, &mut sh);
        for (o, &v) in out.iter_mut().zip(sh.iter()) {
            *o = to_bf16(*o + v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> MoeDims {
        MoeDims {
            hidden: 8,
            latent: 32,
            inter: 32,
            n_experts: 4,
            top_k: 2,
            n_shared: 1,
            scale: 1.0,
            renormalize: true,
            situ_beta: 4.0,
            situ_linear_beta: Some(25.0),
            eps: 1e-5,
            use_norm: true,
        }
    }

    fn bf(n: usize, k: f32) -> Vec<u16> {
        (0..n)
            .map(|i| half::bf16::from_f32(((i as f32) * k).sin() * 0.2).to_bits())
            .collect()
    }

    fn weights(d: MoeDims) -> MoeWeights {
        MoeWeights {
            gate: (0..d.n_experts * d.hidden)
                .map(|i| ((i as f32) * 0.37).sin())
                .collect(),
            e_score_correction_bias: vec![0.0; d.n_experts],
            down_proj: bf(d.latent * d.hidden, 0.11),
            up_proj: bf(d.hidden * d.latent, 0.13),
            norm: vec![1.0; d.latent],
            shared_w1: bf(d.inter * d.n_shared * d.hidden, 0.17),
            shared_w3: bf(d.inter * d.n_shared * d.hidden, 0.19),
            shared_w2: bf(d.hidden * d.inter * d.n_shared, 0.23),
        }
    }

    fn experts(d: MoeDims) -> FlatExperts {
        let stride = expert_fp4::expert_bytes(d.latent, d.inter);
        let mut data = vec![0u8; stride * d.n_experts];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 31 % 251) as u8;
        }
        // keep every E8M0 scale near 2^0 so the test values stay in range
        for e in 0..d.n_experts {
            let base = e * stride;
            let mut off = base;
            for (o, inn) in [
                (d.inter, d.latent),
                (d.inter, d.latent),
                (d.latent, d.inter),
            ] {
                let nib = o * inn / 2;
                for b in data[off + nib..off + expert_fp4::section_bytes(o, inn)].iter_mut() {
                    *b = 127;
                }
                off += expert_fp4::section_bytes(o, inn);
            }
        }
        FlatExperts { data, stride }
    }

    #[test]
    fn output_is_finite_and_shaped() {
        let d = dims();
        let (w, ex) = (weights(d), experts(d));
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.29).cos()).collect();
        let mut out = vec![0.0f32; d.hidden];
        moe_forward(&x, &w, d, &ex, &mut out);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite: {out:?}");
    }

    #[test]
    fn dropping_shared_experts_changes_the_output() {
        // Guards against the shared branch being silently skipped.
        let d = dims();
        let (w, ex) = (weights(d), experts(d));
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.29).cos()).collect();
        let mut with = vec![0.0f32; d.hidden];
        moe_forward(&x, &w, d, &ex, &mut with);
        let mut without = vec![0.0f32; d.hidden];
        let d0 = MoeDims { n_shared: 0, ..d };
        moe_forward(&x, &w, d0, &ex, &mut without);
        assert_ne!(with, without, "shared expert contribution went missing");
    }

    #[test]
    fn top_k_selection_bias_steers_the_router() {
        // A large bias on one expert must force it into the selection.
        let d = dims();
        let (mut w, ex) = (weights(d), experts(d));
        w.e_score_correction_bias = vec![0.0, 0.0, 0.0, 100.0];
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.29).cos()).collect();
        let mut logits = vec![0.0f32; d.n_experts];
        for e in 0..d.n_experts {
            let row = &w.gate[e * d.hidden..(e + 1) * d.hidden];
            logits[e] = row.iter().zip(&x).map(|(&a, &b)| a * b).sum();
        }
        let sel = moe_gate(&logits, &w.e_score_correction_bias, d.top_k, d.scale, true);
        assert!(
            sel.idx.contains(&3),
            "biased expert not selected: {:?}",
            sel.idx
        );
        let mut out = vec![0.0f32; d.hidden];
        moe_forward(&x, &w, d, &ex, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
