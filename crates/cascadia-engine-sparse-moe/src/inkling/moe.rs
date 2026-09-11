//! Inkling MoE block (`InklingMoE`) and dense MLP (`InklingMLP`).
//!
//! MoE: `out = Σ_{i ∈ sel} w_i · E_i(x) + Σ_s γ_s · S_s(x)` — router logits
//! `[n_routed + n_shared]` in f32 (`mlp.gate.weight`, shared rows last),
//! scored by [`inkling_gate`] (whose weights / gammas already carry
//! `route_scale · gate.global_scale`), routed experts in gate order, then the
//! `n_shared` (2) shared experts each with its own gamma. Every expert is a
//! SwiGLU FFN held as a glm [`AnyExpert`] (bf16 goldens / mmap'd int4 bins).
//!
//! HF applies a routed weight AFTER the expert's down-proj and a shared gamma
//! BEFORE it (on the `silu·up` product); by linearity both equal `γ · S(x)`,
//! and this shell applies both after (one code path, bf16 write-back is the
//! only difference — well inside the fixture tolerance).
//!
//! Dense (`dense_layers`): `down(silu(gate·x) · up·x) · mlp.global_scale`.

use super::ffn::AnyExpert;
use super::gate::{inkling_gate, GateOut};
use crate::dsv4::math::linear_f32;

/// Router + expert weights of one MoE layer.
pub struct MoeWeights {
    /// `mlp.gate.weight` `[n_routed + n_shared, hidden]`, f32 (logits are not
    /// bf16-rounded).
    pub router_w: Vec<f32>,
    /// `mlp.gate.bias` (`e_score_correction_bias`) `[n_routed]`.
    pub router_bias: Vec<f32>,
    /// `mlp.gate.global_scale` — multiplies routed weights AND shared gammas.
    pub global_scale: f32,
    /// The routed experts, each `inter` wide.
    pub experts: Vec<AnyExpert>,
    /// The shared experts (2), each `inter` wide, each with its own gamma.
    pub shared: Vec<AnyExpert>,
}

pub struct MoeLayer {
    pub hidden: usize,
    pub n_routed: usize,
    pub n_shared: usize,
    pub top_k: usize,
    /// Expert intermediate width (`moe_intermediate`), routed and shared alike.
    pub inter: usize,
    pub route_scale: f32,
    w: MoeWeights,
}

impl MoeLayer {
    /// Rows per batch-union block (bounds the per-block expert-output scratch
    /// `ROW_BLOCK · top_k · hidden` f32; correctness is independent of it).
    const ROW_BLOCK: usize = 128;

    pub fn new(hidden: usize, inter: usize, top_k: usize, route_scale: f32, w: MoeWeights) -> Self {
        let (n_routed, n_shared) = (w.experts.len(), w.shared.len());
        assert!(n_routed > 0, "moe: no routed experts");
        assert!(
            top_k >= 1 && top_k <= n_routed,
            "moe: top_k {top_k} vs {n_routed} experts"
        );
        assert_eq!(
            w.router_w.len(),
            (n_routed + n_shared) * hidden,
            "moe: router_w must be [n_routed + n_shared, hidden]"
        );
        assert_eq!(
            w.router_bias.len(),
            n_routed,
            "moe: router_bias len != n_routed"
        );
        Self {
            hidden,
            n_routed,
            n_shared,
            top_k,
            inter,
            route_scale,
            w,
        }
    }

    /// The routed experts (for pinning / prefetch enumeration).
    pub fn experts(&self) -> &[AnyExpert] {
        &self.w.experts
    }

    /// The shared experts (always active — top pin candidates).
    pub fn shared(&self) -> &[AnyExpert] {
        &self.w.shared
    }

    pub fn global_scale(&self) -> f32 {
        self.w.global_scale
    }

    /// Route one token: router GEMV (f32) + [`inkling_gate`]. No expert
    /// compute — also the prediction hook for prefetch.
    pub fn route(&self, x: &[f32]) -> GateOut {
        assert_eq!(x.len(), self.hidden, "moe route: x len");
        let n_total = self.n_routed + self.n_shared;
        let mut logits = vec![0.0f32; n_total];
        linear_f32(x, &self.w.router_w, n_total, self.hidden, &mut logits);
        inkling_gate(
            &logits,
            &self.w.router_bias,
            self.top_k,
            self.n_shared,
            self.route_scale,
            self.w.global_scale,
        )
    }

    /// MoE for one token `x` (`[hidden]`, the mlp-normed hidden). Returns `[hidden]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let gate = self.route(x);
        let mut out = vec![0.0f32; self.hidden];
        for (&e, &wj) in gate.idx.iter().zip(&gate.w) {
            let y = self.w.experts[e].forward(x, self.hidden, self.inter);
            for (o, &yi) in out.iter_mut().zip(&y) {
                *o += wj * yi;
            }
        }
        self.add_shared(x, &gate.gammas, &mut out);
        out
    }

    /// `out += Σ_s γ_s · S_s(x)` — the shared experts, in order.
    fn add_shared(&self, x: &[f32], gammas: &[f32], out: &mut [f32]) {
        for (s, &g) in self.w.shared.iter().zip(gammas) {
            let y = s.forward(x, self.hidden, self.inter);
            for (o, &yi) in out.iter_mut().zip(&y) {
                *o += g * yi;
            }
        }
    }

    /// Batch-union MoE for `rows` tokens (`xs` is `[rows, hidden]`), returning
    /// `[rows, hidden]`. Bit-identical to [`Self::forward`] per row (same
    /// router, same gate-order accumulation, then shared), but each unique
    /// routed expert is visited once per block and computes all its rows back
    /// to back — so an mmap'd expert's int4 pages are faulted in once. The
    /// prefill / batched-verify path.
    pub fn forward_batch(&self, xs: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(xs.len(), rows * self.hidden, "moe forward_batch: xs len");
        let mut out = vec![0.0f32; rows * self.hidden];
        let mut lo = 0;
        while lo < rows {
            let hi = (lo + Self::ROW_BLOCK).min(rows);
            self.forward_block(xs, lo, hi, &mut out);
            lo = hi;
        }
        out
    }

    fn forward_block(&self, xs: &[f32], lo: usize, hi: usize, out: &mut [f32]) {
        let (hidden, k) = (self.hidden, self.top_k);
        let nblk = hi - lo;

        // 1. Route every row; remember each (row, slot)'s expert + weight and the
        //    per-expert occurrence list.
        let mut slot_w = vec![0.0f32; nblk * k];
        let mut gammas = vec![0.0f32; nblk * self.n_shared];
        let mut occ: Vec<Vec<usize>> = vec![Vec::new(); self.n_routed];
        for br in 0..nblk {
            let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
            let gate = self.route(x);
            for (slot, (&e, &wj)) in gate.idx.iter().zip(&gate.w).enumerate() {
                let s = br * k + slot;
                slot_w[s] = wj;
                occ[e].push(s);
            }
            gammas[br * self.n_shared..(br + 1) * self.n_shared].copy_from_slice(&gate.gammas);
        }

        // 2. One visit per unique routed expert, its rows hot.
        let mut ey = vec![0.0f32; nblk * k * hidden];
        for (e, slots) in occ.iter().enumerate() {
            for &s in slots {
                let br = s / k;
                let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
                let y = self.w.experts[e].forward(x, hidden, self.inter);
                ey[s * hidden..(s + 1) * hidden].copy_from_slice(&y);
            }
        }

        // 3. Per row: routed in gate order, then shared — forward()'s op order.
        for br in 0..nblk {
            let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
            let o = &mut out[(lo + br) * hidden..(lo + br + 1) * hidden];
            for slot in 0..k {
                let s = br * k + slot;
                let wj = slot_w[s];
                for (oo, &yi) in o.iter_mut().zip(&ey[s * hidden..(s + 1) * hidden]) {
                    *oo += wj * yi;
                }
            }
            self.add_shared(x, &gammas[br * self.n_shared..(br + 1) * self.n_shared], o);
        }
    }
}

/// The dense first-`dense_mlp_idx` layers' MLP: one SwiGLU FFN scaled by
/// `mlp.global_scale`.
pub struct DenseMlp {
    pub w: AnyExpert,
    /// `dense_intermediate`.
    pub inter: usize,
    pub global_scale: f32,
}

impl DenseMlp {
    pub fn new(w: AnyExpert, inter: usize, global_scale: f32) -> Self {
        Self {
            w,
            inter,
            global_scale,
        }
    }

    /// `down(silu(gate·x) · up·x) · global_scale` for one token (`[hidden]`).
    pub fn forward(&self, x: &[f32], hidden: usize) -> Vec<f32> {
        let mut y = self.w.forward(x, hidden, self.inter);
        for v in y.iter_mut() {
            *v *= self.global_scale;
        }
        y
    }
}
