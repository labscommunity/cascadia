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
//!
//! Expert I/O (the mmap'd real model): after routing, every expert the token
//! touches — the `top_k` routed plus both shared — is prefetched
//! (`madvise(WILLNEED)` / `PrefetchVirtualMemory`) and then read whole,
//! concurrently (rayon over the selection, one sequential `read` per bin), and
//! the GEMVs run from those buffers — glm's light-R1 path
//! ([`MmapExpert::read_bytes`](crate::dsv4::expert_mmap::MmapExpert::read_bytes)
//! → `swiglu_from`). The bytes are the mmap's bytes and the kernel is the
//! same, so the output is bit-identical to faulting the pages in mid-GEMV one
//! expert at a time; only the disk sees the difference (8 experts in flight
//! instead of one). `CASCADIA_INKLING_SEQ_READS=1` restores the serial
//! fault-on-touch behaviour (the family's escape hatch; glm's
//! `CASCADIA_GLM5_R1READ` is the same switch with the opposite default). The
//! batch-union prefill prefetches every expert with rows before its expert
//! pass and computes from the mmap (each expert's pages are touched once per
//! block anyway).
//!
//! Expert-parallel (remote) experts: a layer built with
//! [`ExpertSet::None`](super::loader::ExpertSet::None) holds the router only
//! and, once [`MoeLayer::attach_remote`] hands it an [`EpClient`], routes
//! locally and dispatches every expert evaluation (routed AND shared — the
//! shared experts are dispatched like routed ones with their gammas as
//! weights) to the expert workers. The workers return raw `E(h)`; the driver
//! applies the weights and sums in the same gate order as the local path, so
//! the output is bit-identical to a single-process layer on the same weights
//! (`tests/inkling_ep.rs` asserts exact equality).

use std::sync::Arc;

use super::env_flag;
use super::ep::EpClient;
use super::ffn::AnyExpert;
use super::gate::{inkling_gate, GateOut};
use crate::dsv4::math::linear_f32;

/// `CASCADIA_INKLING_SEQ_READS`: serial expert reads (fault each mmap'd
/// expert's pages in during its own GEMV) instead of the concurrent
/// whole-bin reads. Read once. Shared with the expert worker
/// ([`super::ep::ExpertBank`]), which mirrors the decode read path.
pub(crate) fn seq_reads() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| env_flag("CASCADIA_INKLING_SEQ_READS"))
}

/// Router + expert weights of one MoE layer.
pub struct MoeWeights {
    /// `mlp.gate.weight` `[n_routed + n_shared, hidden]`, f32 (logits are not
    /// bf16-rounded).
    pub router_w: Vec<f32>,
    /// `mlp.gate.bias` (`e_score_correction_bias`) `[n_routed]`.
    pub router_bias: Vec<f32>,
    /// `mlp.gate.global_scale` — multiplies routed weights AND shared gammas.
    pub global_scale: f32,
    /// The routed experts, each `inter` wide — `n_routed` of them, or EMPTY
    /// for a router-only layer (expert-parallel driver: the experts live on
    /// the workers; see [`MoeLayer::attach_remote`]).
    pub experts: Vec<AnyExpert>,
    /// The shared experts (2), each `inter` wide, each with its own gamma —
    /// `n_shared` of them, or empty alongside an empty `experts`.
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
    /// Expert-parallel dispatch: `(absolute layer index, client)`. When set,
    /// every expert evaluation goes to the workers ([`Self::forward_remote`]).
    remote: Option<(u32, Arc<EpClient>)>,
}

impl MoeLayer {
    /// Rows per batch-union block (bounds the per-block expert-output scratch
    /// `ROW_BLOCK · top_k · hidden` f32; correctness is independent of it).
    const ROW_BLOCK: usize = 128;

    /// `n_routed` / `n_shared` come from the router (`router_bias` is
    /// `[n_routed]`, `router_w` is `[n_routed + n_shared, hidden]`), so the
    /// expert tables may be either complete (`n_routed` routed + `n_shared`
    /// shared) or both empty — a router-only layer that must have a remote
    /// attached ([`Self::attach_remote`]) before it can run.
    pub fn new(hidden: usize, inter: usize, top_k: usize, route_scale: f32, w: MoeWeights) -> Self {
        assert!(hidden > 0, "moe: hidden must be > 0");
        let n_routed = w.router_bias.len();
        assert!(n_routed > 0, "moe: no routed experts (empty router_bias)");
        assert!(
            w.router_w.len() >= n_routed * hidden && w.router_w.len().is_multiple_of(hidden),
            "moe: router_w len {} is not [n_routed + n_shared, hidden] for hidden {hidden}, n_routed {n_routed}",
            w.router_w.len()
        );
        let n_shared = w.router_w.len() / hidden - n_routed;
        assert!(
            top_k >= 1 && top_k <= n_routed,
            "moe: top_k {top_k} vs {n_routed} experts"
        );
        let local = !w.experts.is_empty() || !w.shared.is_empty();
        if local {
            assert_eq!(
                w.experts.len(),
                n_routed,
                "moe: {} routed experts loaded for a {n_routed}-expert router",
                w.experts.len()
            );
            assert_eq!(
                w.shared.len(),
                n_shared,
                "moe: {} shared experts loaded for a router with {n_shared}",
                w.shared.len()
            );
        }
        Self {
            hidden,
            n_routed,
            n_shared,
            top_k,
            inter,
            route_scale,
            w,
            remote: None,
        }
    }

    /// Whether this layer holds its experts locally (routed + shared). False
    /// for a router-only layer built with `ExpertSet::None`.
    pub fn has_local_experts(&self) -> bool {
        !self.w.experts.is_empty()
    }

    /// Dispatch every expert evaluation of this layer to the expert workers
    /// behind `client`, as absolute layer `layer` (the worker's bank is
    /// indexed by absolute layer number). Overrides any local experts.
    /// Panics if the client's dims disagree with the layer — a driver
    /// misconfiguration ([`super::stage::InklingRunner::load_staged`] checks
    /// first and returns an error).
    pub fn attach_remote(&mut self, layer: u32, client: Arc<EpClient>) {
        assert_eq!(
            client.hidden(),
            self.hidden,
            "moe layer {layer}: expert client hidden {} != layer hidden {}",
            client.hidden(),
            self.hidden
        );
        assert_eq!(
            client.n_routed(),
            self.n_routed,
            "moe layer {layer}: expert client n_routed {} != layer n_routed {}",
            client.n_routed(),
            self.n_routed
        );
        assert_eq!(
            client.n_shared(),
            self.n_shared,
            "moe layer {layer}: expert client n_shared {} != layer n_shared {}",
            client.n_shared(),
            self.n_shared
        );
        self.remote = Some((layer, client));
    }

    /// The attached expert-parallel client and this layer's absolute index.
    pub fn remote(&self) -> Option<(u32, &Arc<EpClient>)> {
        self.remote.as_ref().map(|(l, c)| (*l, c))
    }

    /// Route `rows` tokens (`xs` is `[rows, hidden]`) and evaluate their
    /// experts on the workers. Per row the `(expert id, weight)` list is the
    /// routed selection in gate order followed by the shared experts
    /// (`n_routed + s`, gamma_s); [`EpClient::dispatch`] accumulates
    /// `Σ w · E(h)` in exactly that order, so the result is bit-identical to
    /// [`Self::forward`] / [`Self::forward_batch`] on local experts. A
    /// worker / transport failure is a hard error of the forward (there is
    /// no error channel through the layer stack): it panics with the
    /// worker's message, which names the worker index and the layer.
    fn forward_remote(&self, xs: &[f32], rows: usize) -> Vec<f32> {
        let (layer, client) = self
            .remote
            .as_ref()
            .expect("forward_remote without a remote client");
        assert_eq!(xs.len(), rows * self.hidden, "moe forward_remote: xs len");
        let per_row: Vec<Vec<(usize, f32)>> = xs
            .chunks_exact(self.hidden)
            .map(|x| {
                let gate = self.route(x);
                gate.idx
                    .iter()
                    .zip(&gate.w)
                    .map(|(&e, &w)| (e, w))
                    .chain(
                        gate.gammas
                            .iter()
                            .enumerate()
                            .map(|(s, &g)| (self.n_routed + s, g)),
                    )
                    .collect()
            })
            .collect();
        match client.dispatch(*layer, xs, &per_row) {
            Ok(out) => out,
            Err(e) => panic!("inkling expert-parallel dispatch failed (layer {layer}): {e}"),
        }
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

    /// MoE for one token `x` (`[hidden]`, the mlp-normed hidden). Returns
    /// `[hidden]`. Routed experts accumulate in gate order, then the shared
    /// experts — the order [`Self::forward_batch`] reproduces per row.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        if self.remote.is_some() {
            return self.forward_remote(x, 1);
        }
        assert!(
            self.has_local_experts(),
            "inkling MoE layer has no local experts and no expert-parallel client attached \
             (a driver built with ExpertSet::None must attach_remote before running)"
        );
        let gate = self.route(x);
        // Every expert this token touches, in accumulation order, with its weight.
        let sel: Vec<&AnyExpert> = gate
            .idx
            .iter()
            .map(|&e| &self.w.experts[e])
            .chain(self.w.shared.iter())
            .collect();
        let weights = gate.w.iter().chain(gate.gammas.iter());
        // Kick the OS read-ahead for all of them before any compute.
        for e in &sel {
            e.prefetch();
        }
        // Overlapped reads: the mmap'd experts' whole bins, concurrently, into
        // owned buffers the GEMVs then run from (bit-identical to the mmap).
        let bufs: Vec<Option<Vec<u8>>> =
            if !seq_reads() && sel.iter().any(|e| e.as_mmap().is_some()) {
                use rayon::prelude::*;
                sel.par_iter()
                    .map(|e| e.as_mmap().and_then(|m| m.read_bytes().ok()))
                    .collect()
            } else {
                vec![None; sel.len()]
            };
        let mut out = vec![0.0f32; self.hidden];
        for ((e, buf), &wj) in sel.iter().zip(&bufs).zip(weights) {
            let y = match (buf, e.as_mmap()) {
                (Some(b), Some(m)) => m.swiglu_from(b, x),
                _ => e.forward(x, self.hidden, self.inter),
            };
            for (o, &yi) in out.iter_mut().zip(&y) {
                *o += wj * yi;
            }
        }
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
        if self.remote.is_some() {
            return self.forward_remote(xs, rows);
        }
        assert!(
            self.has_local_experts(),
            "inkling MoE layer has no local experts and no expert-parallel client attached \
             (a driver built with ExpertSet::None must attach_remote before running)"
        );
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

        // 2. Read-ahead for every expert this block touches (routed with
        //    rows, plus the shared pair), so the expert pass overlaps its I/O.
        for (e, slots) in occ.iter().enumerate() {
            if !slots.is_empty() {
                self.w.experts[e].prefetch();
            }
        }
        for s in &self.w.shared {
            s.prefetch();
        }

        // 3. One visit per unique routed expert, its rows hot.
        let mut ey = vec![0.0f32; nblk * k * hidden];
        for (e, slots) in occ.iter().enumerate() {
            for &s in slots {
                let br = s / k;
                let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
                let y = self.w.experts[e].forward(x, hidden, self.inter);
                ey[s * hidden..(s + 1) * hidden].copy_from_slice(&y);
            }
        }

        // 4. Per row: routed in gate order, then shared — forward()'s op order.
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
