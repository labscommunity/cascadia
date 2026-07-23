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

use std::sync::{Arc, Mutex};

use super::ffn::{swiglu, swiglu_f32w, swiglu_mmap};
use super::gate::moe_gate;
use super::prof;
use super::residency::UsageStats;
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
    EagerF32 {
        wg: Vec<f32>,
        wu: Vec<f32>,
        wd: Vec<f32>,
    },
    /// mmap'd int4 bin, rows dequantized on the fly — the only mode that fits
    /// the real model (eager f32 experts would be hundreds of GB per rank).
    Mmap(MmapExpert),
}

/// Light-R1 read path: explicit concurrent whole-expert reads instead of
/// mmap-fault-on-touch. Opt-in via `CASCADIA_GLM5_R1READ` (measured ~1.4× on
/// NVMe; a no-op / possibly worse on slow disks). Read once.
fn r1_read() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("CASCADIA_GLM5_R1READ").is_some())
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

impl AnyExpert {
    /// The mmap'd int4 expert, if this is the `Mmap` variant (for pinning).
    pub fn as_mmap(&self) -> Option<&MmapExpert> {
        match self {
            AnyExpert::Mmap(m) => Some(m),
            _ => None,
        }
    }

    /// `madvise(WILLNEED)` hint (mmap experts only; no-op otherwise).
    #[inline]
    pub fn prefetch(&self) {
        if let AnyExpert::Mmap(m) = self {
            m.prefetch();
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
    /// Optional routing recorder (learned-pin). None on the golden path.
    usage: Option<Arc<Mutex<UsageStats>>>,
    /// This layer's GLOBAL index (for the usage histogram key).
    layer_idx: u32,
}

impl MoeLayer {
    /// Rows per batch-union block. Bounds the intermediate expert-output storage
    /// (`ROW_BLOCK · top_k · hidden` f32) while keeping enough rows per block for
    /// heavy expert overlap. Correctness is independent of this value.
    const ROW_BLOCK: usize = 128;

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
            usage: None,
            layer_idx: 0,
        }
    }

    /// Attach the learned-pin routing recorder + this layer's global index.
    pub fn attach_usage(&mut self, layer_idx: u32, usage: Arc<Mutex<UsageStats>>) {
        self.layer_idx = layer_idx;
        self.usage = Some(usage);
    }

    /// The routed experts (for pinning enumeration).
    pub fn experts(&self) -> &[AnyExpert] {
        &self.w.experts
    }

    /// The shared expert (always active — a top pin candidate).
    pub fn shared(&self) -> &AnyExpert {
        &self.w.shared
    }

    /// Prefetch the experts this layer is most likely to fire next — the shared
    /// expert (always active) + the `n` hottest routed experts from the
    /// learned-pin histogram (if attached). Best-effort `madvise(WILLNEED)`; no
    /// effect on output. Issued for the NEXT layer while the current one
    /// computes, so the expert reads overlap compute instead of stalling.
    pub fn prefetch_hot(&self, n: usize) {
        self.w.shared.prefetch();
        if let Some(u) = &self.usage {
            if let Ok(g) = u.lock() {
                for e in g.hottest_for(self.layer_idx, n) {
                    if let Some(exp) = self.w.experts.get(e as usize) {
                        exp.prefetch();
                    }
                }
            }
        }
    }

    /// MoE for one token `x` (`[hidden]`). Returns `[hidden]`.
    pub fn forward_token(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        let t_router = std::time::Instant::now();
        // router logits (f32) -> sigmoid + noaux_tc gate.
        let mut logits = vec![0.0f32; self.n_experts];
        linear_f32(
            x,
            &self.w.router_w,
            self.n_experts,
            self.hidden,
            &mut logits,
        );
        let gate = moe_gate(&logits, &self.w.router_bias, self.top_k, self.scale, true);
        // record the routing for learned-pin (no-op when no recorder attached).
        if let Some(u) = &self.usage {
            if let Ok(mut g) = u.lock() {
                for &e in &gate.idx {
                    g.record(self.layer_idx, e);
                }
            }
        }
        prof::add(prof::ROUTER, t_router);

        // routed experts in gate order, then the shared expert.
        let t_exp = std::time::Instant::now();
        let mut out = vec![0.0f32; self.hidden];
        if r1_read() {
            // Light-R1: read the routed experts' whole bins up-front and
            // CONCURRENTLY (off the compute path, at full sequential bandwidth),
            // then compute from the buffers — instead of faulting mmap pages in
            // mid-GEMV one expert at a time. Bit-identical to the mmap path.
            use rayon::prelude::*;
            let bufs: Vec<Option<Vec<u8>>> = gate
                .idx
                .par_iter()
                .map(|&e| {
                    self.w.experts[e as usize]
                        .as_mmap()
                        .and_then(|m| m.read_bytes().ok())
                })
                .collect();
            for (slot, (&e, &wj)) in gate.idx.iter().zip(&gate.weight).enumerate() {
                let ex = &self.w.experts[e as usize];
                let y = match (bufs[slot].as_ref(), ex.as_mmap()) {
                    (Some(b), Some(m)) => m.swiglu_from(b, x),
                    _ => ex.forward(x, self.hidden, self.moe_inter),
                };
                for (o, &yi) in out.iter_mut().zip(&y) {
                    *o += wj * yi;
                }
            }
        } else {
            for (&e, &wj) in gate.idx.iter().zip(&gate.weight) {
                let y = self.w.experts[e as usize].forward(x, self.hidden, self.moe_inter);
                for (o, &yi) in out.iter_mut().zip(&y) {
                    *o += wj * yi;
                }
            }
        }
        let s = self.w.shared.forward(x, self.hidden, self.shared_inter);
        for (o, &si) in out.iter_mut().zip(&s) {
            *o += si;
        }
        prof::add(prof::EXPERTS, t_exp);
        out
    }

    /// Batch-union MoE for `rows` tokens (`xs` is `[rows, hidden]`), returning
    /// `[rows, hidden]`. **Bit-identical** to calling [`Self::forward_token`] per
    /// row — same router, same per-row gate-order accumulation — but each unique
    /// routed expert is visited once and computes all of its assigned rows back
    /// to back, so its int4 weights (an mmap page fault or a dequant) are paid
    /// for once and stay hot. This is the prefill / batched-verify path; at high
    /// node counts it cuts the NVMe traffic that dominates, since a prompt's
    /// tokens route to heavily overlapping experts. Rows are processed in blocks
    /// so the intermediate storage stays bounded (dedup is within a block).
    pub fn forward_batch(&self, xs: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(xs.len(), rows * self.hidden);
        let mut out = vec![0.0f32; rows * self.hidden];
        let mut lo = 0;
        while lo < rows {
            let hi = (lo + Self::ROW_BLOCK).min(rows);
            self.forward_block(xs, lo, hi, &mut out);
            lo = hi;
        }
        out
    }

    /// Batch-union over rows `[lo, hi)`, writing into `out[lo..hi]`.
    fn forward_block(&self, xs: &[f32], lo: usize, hi: usize, out: &mut [f32]) {
        let (hidden, k) = (self.hidden, self.top_k);
        let nblk = hi - lo;

        // 1. Route every row in the block; record the (expert -> occurrences)
        //    map and the per-slot expert/weight, exactly as forward_token would.
        let mut slot_w = vec![0.0f32; nblk * k]; // gate weight per (blockrow, slot)
        let mut occ: Vec<Vec<u32>> = vec![Vec::new(); self.n_experts]; // expert -> [blockrow*k+slot]
        let mut logits = vec![0.0f32; self.n_experts];
        for br in 0..nblk {
            let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
            // `linear_f32` overwrites every element, so no pre-zeroing needed.
            linear_f32(x, &self.w.router_w, self.n_experts, hidden, &mut logits);
            let gate = moe_gate(&logits, &self.w.router_bias, k, self.scale, true);
            if let Some(u) = &self.usage {
                if let Ok(mut g) = u.lock() {
                    for &e in &gate.idx {
                        g.record(self.layer_idx, e);
                    }
                }
            }
            for (slot, (&e, &wj)) in gate.idx.iter().zip(&gate.weight).enumerate() {
                let s = br * k + slot;
                slot_w[s] = wj;
                occ[e as usize].push(s as u32);
            }
        }

        // 2. Dedup compute: one visit per unique routed expert, its rows hot.
        let mut ey = vec![0.0f32; nblk * k * hidden]; // expert output per (blockrow, slot)
        for (e, slots) in occ.iter().enumerate() {
            if slots.is_empty() {
                continue;
            }
            let exp = &self.w.experts[e];
            for &s in slots {
                let br = (s as usize) / k;
                let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
                let y = exp.forward(x, hidden, self.moe_inter);
                ey[s as usize * hidden..(s as usize + 1) * hidden].copy_from_slice(&y);
            }
        }

        // 3. Per row: accumulate experts in gate order, then shared — the exact
        //    op order of forward_token, so the result is bit-for-bit identical.
        for br in 0..nblk {
            let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
            let o = &mut out[(lo + br) * hidden..(lo + br + 1) * hidden];
            for slot in 0..k {
                let s = br * k + slot;
                let wj = slot_w[s];
                let y = &ey[s * hidden..(s + 1) * hidden];
                for (oo, &yi) in o.iter_mut().zip(y) {
                    *oo += wj * yi;
                }
            }
            let sh = self.w.shared.forward(x, hidden, self.shared_inter);
            for (oo, &si) in o.iter_mut().zip(&sh) {
                *oo += si;
            }
        }
    }
}
