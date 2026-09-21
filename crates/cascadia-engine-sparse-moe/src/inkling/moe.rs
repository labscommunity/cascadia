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
//! (`madvise(WILLNEED)` / `PrefetchVirtualMemory`). An mmap'd expert whose
//! pages are NOT mostly resident (a 64-page `mincore` sample, see
//! [`MmapExpert::mostly_resident`](crate::dsv4::expert_mmap::MmapExpert::mostly_resident))
//! is then streamed whole into an owned buffer, concurrently with the others
//! (rayon over the selection, one sequential `read` per bin), and its GEMV
//! runs from that buffer — glm's light-R1 path
//! ([`MmapExpert::read_bytes`](crate::dsv4::expert_mmap::MmapExpert::read_bytes)
//! → `swiglu_from`). An expert already resident is computed straight off the
//! mapping (the whole-bin copy would only cost when the pages are already in
//! RAM). Either way the bytes are the mmap's bytes and the kernel is the
//! same, so the output is bit-identical to faulting the pages in mid-GEMV one
//! expert at a time; only the disk sees the difference (paged-out experts in
//! flight together instead of one). `CASCADIA_INKLING_SEQ_READS=1` forces the
//! straight-off-mapping (fault-on-touch) path for every expert, resident or
//! not (the family's escape hatch; glm's `CASCADIA_GLM5_R1READ` is the same
//! switch with the opposite default). The batch-union prefill prefetches every
//! expert with rows before its expert pass and computes from the mmap (each
//! expert's pages are touched once per block anyway).
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

/// `CASCADIA_INKLING_SEQ_READS`: always compute straight off an mmap'd
/// expert (fault its pages in during its own GEMV) instead of the bulk
/// whole-bin read a paged-out expert gets by default. Read once. Shared
/// with the expert worker ([`super::ep::ExpertBank`]), which mirrors the
/// decode read path.
pub(crate) fn seq_reads() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| env_flag("CASCADIA_INKLING_SEQ_READS"))
}

/// Retain a bounded pool of bulk-read destination buffers across layers/tokens.
/// Default off; direct mapped execution takes precedence over this option.
fn reuse_read_buffers() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| env_flag("CASCADIA_INKLING_REUSE_READ_BUFFERS"))
}

/// Opt-in overlap of each selected expert's read and compute. Requires the
/// reusable-buffer path and parallel experts; defaults remain unchanged.
fn pipeline_reads() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| env_flag("CASCADIA_INKLING_PIPELINE_READS"))
}

/// Optional bounded bulk reads for each prefill expert's complete row group.
/// Uses the same kernels and requires the reusable bulk-read configuration.
fn prefill_reads() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| env_flag("CASCADIA_INKLING_PREFILL_READS"))
        && reuse_read_buffers()
        && !seq_reads()
}

static PIPELINED_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Number of decode layer calls that actually used overlapped reads/compute.
pub fn pipeline_read_layer_count() -> u64 {
    PIPELINED_LAYERS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Skip the serial hint phase before bulk decode reads. Prefill and direct
/// mapped execution retain their hints. This measured alternative is opt-in.
fn skip_bulk_prefetch() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| env_flag("CASCADIA_INKLING_SKIP_BULK_PREFETCH"))
}

/// `CASCADIA_INKLING_SERIAL_EXPERTS`: run a token's selected experts one
/// after another (each GEMV row-parallel on its own) instead of
/// concurrently. Same values either way; only the schedule differs. Read
/// once.
pub(crate) fn par_experts() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| !env_flag("CASCADIA_INKLING_SERIAL_EXPERTS"))
}

/// `CASCADIA_INKLING_ROW_GEMM` (default ON; `0`/`false`/`no`/`off` restores
/// the per-row kernels): in a batch-union block, an int4 expert with two or
/// more rows — and both shared experts, which every row uses — runs ONE
/// multi-input pass ([`MmapExpert::swiglu_rows_from`](crate::dsv4::expert_mmap::MmapExpert::swiglu_rows_from))
/// instead of a GEMV per row, so its packed bytes cross the memory bus once
/// per block. Same bits per row either way; only the schedule differs. Read
/// once.
fn row_gemm() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| match std::env::var("CASCADIA_INKLING_ROW_GEMM") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("no")
                || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    })
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

/// Optional diagnostic callback. Observers receive the completed routing result;
/// they must not change the floating-point environment or perform blocking I/O.
pub type RouteObserver = Arc<dyn Fn(&GateOut) + Send + Sync>;

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
    route_observer: Option<RouteObserver>,
    expert_cache: super::expert_cache::ExpertCache,
    /// Optional OpenVINO expert backend (`(layer index, backend)`), attached
    /// by the loader when `CASCADIA_INKLING_OV_EXPERTS` is set: every
    /// selected expert then runs its compiled IR (iGPU / NPU / CPU), falling
    /// back per call to the Rust kernel. See [`super::ov_expert`].
    ov: Option<(u32, Arc<super::ov_expert::OvExperts>)>,
    /// Optional fused-MoE backend (`(layer index, backend)`): the whole layer
    /// as one compiled OpenVINO model, routing from [`Self::route`]. Takes
    /// precedence over `ov` for the routed + shared experts. See
    /// [`super::ov_moe`].
    ov_moe: Option<(u32, Arc<super::ov_moe::OvMoe>)>,
}

impl MoeLayer {
    /// Read every routed expert into the resident cache now (as far as its
    /// capacity goes) instead of on first use. A pipeline rank's cache is sized
    /// to hold its whole slice, but it used to fill only as requests touched
    /// experts: after every restart the first request took 34 s to its first
    /// token and decode ran at half speed for minutes. Returns
    /// `(experts loaded, bytes)`.
    pub fn prewarm_expert_cache(&self) -> (usize, usize) {
        use rayon::prelude::*;
        if self.expert_cache.stats().capacity_bytes == 0 {
            return (0, 0);
        }
        let ids: Vec<usize> = (0..self.w.experts.len()).collect();
        let mut loaded = (0usize, 0usize);
        // Cohorts bound how many 32 MB buffers are in flight at once.
        for cohort in ids.chunks(16) {
            let done: Vec<usize> = cohort
                .par_iter()
                .map(|&e| {
                    let Some(m) = self.w.experts[e].as_mmap() else {
                        return 0;
                    };
                    let mut lease = super::read_buffers::ReadBuffers::acquire(1);
                    if lease.buffers[0]
                        .read_prefill(m.bin_path(), m.bin_len())
                        .is_err()
                    {
                        return 0;
                    }
                    let size = lease.buffers[0].as_slice().len();
                    if self.expert_cache.preload(e, &mut lease.buffers[0]) {
                        size
                    } else {
                        0
                    }
                })
                .collect();
            loaded.0 += done.iter().filter(|&&b| b > 0).count();
            loaded.1 += done.iter().sum::<usize>();
        }
        loaded
    }

    pub fn expert_cache_stats(&self) -> super::ExpertCacheStats {
        self.expert_cache.stats()
    }

    pub(super) fn prediction_reads_enabled(&self) -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| env_flag("CASCADIA_INKLING_PREDICT_READS"))
            && self.remote.is_none()
            && self.has_local_experts()
            && self.expert_cache.stats().capacity_bytes > 0
    }

    pub(super) fn start_predicted_read(
        &self,
        prediction: &GateOut,
    ) -> Option<super::predicted_read::PendingReadGroup> {
        if !self.prediction_reads_enabled() {
            return None;
        }
        let selected = if super::predicted_read::second_reads_requested() {
            self.expert_cache.predicted_uncached(
                &prediction.idx,
                super::predicted_read::second_prediction_rank_ceiling(),
                super::predicted_read::third_reads_requested(),
            )
        } else {
            [
                self.expert_cache.first_uncached(&prediction.idx),
                None,
                None,
            ]
        };
        let first = selected[0].and_then(|expert| {
            let mapped = self.w.experts[expert].as_mmap()?;
            super::predicted_read::start(expert, mapped.bin_path(), mapped.bin_len())
        });
        let second = selected[1].and_then(|expert| {
            let mapped = self.w.experts[expert].as_mmap()?;
            super::predicted_read::start_second(expert, mapped.bin_path(), mapped.bin_len())
        });
        let third = selected[2].and_then(|expert| {
            let mapped = self.w.experts[expert].as_mmap()?;
            super::predicted_read::start_third(expert, mapped.bin_path(), mapped.bin_len())
        });
        super::predicted_read::PendingReadGroup::new(first, second, third)
    }

    pub(crate) fn reset_expert_cache_history(&self) {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if *ENABLED.get_or_init(|| env_flag("CASCADIA_INKLING_CACHE_RESET_HISTORY")) {
            self.expert_cache.reset_history();
        }
    }

    /// Owned packed shared-expert bytes, excluding routed experts and scratch.
    pub fn owned_shared_bytes(&self) -> usize {
        self.w.shared.iter().map(AnyExpert::owned_int4_bytes).sum()
    }

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
        let cache_bytes =
            if local && pipeline_reads() && reuse_read_buffers() && !seq_reads() && par_experts() {
                super::expert_cache::ExpertCache::configured_bytes()
            } else {
                0
            };
        Self {
            hidden,
            n_routed,
            n_shared,
            top_k,
            inter,
            route_scale,
            w,
            remote: None,
            route_observer: None,
            expert_cache: super::expert_cache::ExpertCache::new(n_routed, cache_bytes),
            ov: None,
            ov_moe: None,
        }
    }

    /// Whether this layer holds its experts locally (routed + shared). False
    /// for a router-only layer built with `ExpertSet::None`.
    pub fn has_local_experts(&self) -> bool {
        !self.w.experts.is_empty()
    }

    /// Install or remove an opt-in observer for routing diagnostics. Defaults off.
    /// Direct calls to `route` are observed as well as prefill/decode dispatch.
    pub fn set_route_observer(&mut self, observer: Option<RouteObserver>) {
        self.route_observer = observer;
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
    /// Route this layer's experts through an OpenVINO backend (see
    /// [`super::ov_expert`]); `layer` is the global layer index the IRs are
    /// filed under.
    pub fn attach_ov(&mut self, layer: u32, ov: Arc<super::ov_expert::OvExperts>) {
        self.ov = Some((layer, ov));
    }

    pub fn ov(&self) -> Option<(u32, &Arc<super::ov_expert::OvExperts>)> {
        self.ov.as_ref().map(|(l, o)| (*l, o))
    }

    /// Route this layer through a fused-MoE backend (see [`super::ov_moe`]);
    /// `layer` is the global layer index its IR is filed under.
    pub fn attach_ov_moe(&mut self, layer: u32, ov: Arc<super::ov_moe::OvMoe>) {
        self.ov_moe = Some((layer, ov));
    }

    pub fn ov_moe(&self) -> Option<(u32, &Arc<super::ov_moe::OvMoe>)> {
        self.ov_moe.as_ref().map(|(l, o)| (*l, o))
    }

    /// Compile this layer's fused IR ahead of time; `None` without a backend.
    /// A fused IR whose `k_total` disagrees with this layer's `top_k + n_shared`
    /// can never serve it (`forward_ov_moe` would decline every token), so it is
    /// reported FAILED here rather than warmed "ok" and never actually used.
    pub fn warm_ov_moe(&self) -> Option<bool> {
        let (lid, ov) = self.ov_moe.as_ref()?;
        let k = self.top_k + self.w.shared.len();
        if ov.k_total() != k {
            ov.mark_k_mismatch(*lid, k);
            return Some(false);
        }
        if !ov.warm(*lid) {
            return Some(false);
        }
        Some(self.check_ov_moe_high_ids())
    }

    /// The device path against the Rust kernels on the HIGHEST expert ids of
    /// this layer (the last routed experts and the shared ones). The GPU
    /// plugin's decode kernels index an expert's weights with a 32-bit product
    /// that overflows from id 228 at this model's sizes: low ids compute
    /// correctly and high ones read out of bounds, so a warm-up on ids 0..k
    /// proves nothing. A layer that fails is taken off the device (loudly)
    /// rather than left to write garbage into the residual stream.
    fn check_ov_moe_high_ids(&self) -> bool {
        let Some((lid, ov)) = self.ov_moe.as_ref() else {
            return true;
        };
        if self.ov.is_some() || !self.has_local_experts() || self.n_routed < self.top_k {
            return true; // nothing on the host to compare with
        }
        let k = self.top_k + self.w.shared.len();
        let x: Vec<f32> = (0..self.hidden)
            .map(|i| ((i as f32 * 0.37).sin() + (i as f32 * 0.011).cos()) * 0.05)
            .collect();
        let mut ids: Vec<usize> = (self.n_routed - self.top_k..self.n_routed).collect();
        ids.extend((0..self.w.shared.len()).map(|s| self.n_routed + s));
        let w = vec![1.0f32 / k as f32; k];
        let ids32: Vec<i32> = ids.iter().map(|&e| e as i32).collect();
        let Some(dev) = ov.forward(*lid, &x, 1, &ids32, &w) else {
            return false;
        };
        let mut host = vec![0.0f32; self.hidden];
        for (&e, &we) in ids.iter().zip(&w) {
            for (h, y) in host.iter_mut().zip(self.expert_ov_or_rust(e, &x)) {
                *h += we * y;
            }
        }
        let dot: f64 = dev
            .iter()
            .zip(&host)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let nd: f64 = dev
            .iter()
            .map(|a| *a as f64 * *a as f64)
            .sum::<f64>()
            .sqrt();
        let nh: f64 = host
            .iter()
            .map(|a| *a as f64 * *a as f64)
            .sum::<f64>()
            .sqrt();
        let cosine = if nd > 0.0 && nh > 0.0 {
            dot / (nd * nh)
        } else {
            0.0
        };
        // A layer taken from the regrouped folder carries re-quantised weights:
        // measured 0.992 against the group-32 kernels (12 % of the block's
        // output); an out-of-bounds read gives noise, far below either bar.
        let bar = if ov.uses_decode_kernels(*lid) { 0.98 } else { 0.995 };
        let ok = cosine.is_finite() && cosine > bar;
        tracing::info!(
            target: "cascadia::inkling",
            event = "ov_moe_high_id_check",
            layer = *lid,
            cosine,
            ok,
        );
        // Also as integers on a "stage profile" line: the fleet's beacon relays
        // those from every rank, the log itself stays on the box.
        println!(
            "MC{lid} probe stage profile layer={lid} cos_ppm={} ok={}",
            (cosine.clamp(0.0, 1.0) * 1e6) as u64,
            u8::from(ok)
        );
        // Reported everywhere; ENFORCED (the layer leaves the device) only with
        // CASCADIA_INKLING_OV_MOE_CHECK=enforce, until the check has a record.
        let enforce =
            std::env::var("CASCADIA_INKLING_OV_MOE_CHECK").is_ok_and(|v| v.trim() == "enforce");
        if !ok && !enforce {
            return true;
        }
        if !ok {
            ov.fail_layer(
                *lid,
                &format!("device output for expert ids {ids:?} disagrees with the host kernels (cosine {cosine:.4})"),
            );
        }
        ok
    }

    /// `rows` rows through the fused backend: route each row here, dispatch
    /// the ids + weights (shared experts as `n_routed + s` with their gammas)
    /// in one call. `None` when the backend declines (the caller falls back).
    fn forward_ov_moe(&self, xs: &[f32], rows: usize) -> Option<Vec<f32>> {
        let (lid, ov) = self.ov_moe.as_ref()?;
        let k = self.top_k + self.w.shared.len();
        if ov.k_total() != k {
            // A K mismatch is a permanent per-layer defect of the IR: count the
            // bypass and report it once, so it is never a silent fall-through.
            ov.note_k_mismatch(*lid, k);
            return None;
        }
        let mut ids = Vec::with_capacity(rows * k);
        let mut wts = Vec::with_capacity(rows * k);
        for r in 0..rows {
            let gate = self.route(&xs[r * self.hidden..(r + 1) * self.hidden]);
            ids.extend(gate.idx.iter().map(|&e| e as i32));
            ids.extend((0..self.w.shared.len()).map(|s| (self.n_routed + s) as i32));
            wts.extend_from_slice(&gate.w);
            wts.extend_from_slice(&gate.gammas);
        }
        ov.forward(*lid, xs, rows, &ids, &wts)
    }

    /// Compile every expert of this layer on the attached OV backend (a
    /// benchmark's warm-up); returns `(compiled, failed keys)`, or `None`
    /// when no backend is attached.
    pub fn warm_ov(&self) -> Option<(usize, Vec<(u32, u32)>)> {
        if self.ov_moe.is_some() {
            return None; // the fused backend serves this layer; its per-expert IRs are only a fallback
        }
        let (lid, ov) = self.ov.as_ref()?;
        let n = self.w.experts.len() + self.w.shared.len();
        let keys: Vec<(u32, u32)> = (0..n as u32).map(|e| (*lid, e)).collect();
        let bad = ov.warm(&keys, self.n_routed as u32);
        Some((n - bad.len(), bad))
    }

    /// One expert (`id` in the Rust id space: routed `0..n_routed`, shared
    /// `n_routed + s`) on `x`: the OV backend when attached and it answers,
    /// else the Rust kernel straight off the expert's storage.
    fn expert_ov_or_rust(&self, id: usize, x: &[f32]) -> Vec<f32> {
        if let Some((lid, ov)) = &self.ov {
            if let Some(y) = ov.expert(*lid, id as u32, self.n_routed as u32, x) {
                return y;
            }
        }
        let e = if id < self.n_routed {
            &self.w.experts[id]
        } else {
            &self.w.shared[id - self.n_routed]
        };
        e.forward(x, self.hidden, self.inter)
    }

    /// Decode through the OV backend: the token's routed + shared experts as
    /// compiled IRs (concurrently under rayon when `par_experts`), summed in
    /// gate order like [`Self::forward`]. Bypasses the mmap read machinery —
    /// the device holds the weights.
    fn forward_ov(&self, x: &[f32]) -> Vec<f32> {
        let gate = self.route(x);
        let ids: Vec<usize> = gate
            .idx
            .iter()
            .copied()
            .chain((0..self.w.shared.len()).map(|s| self.n_routed + s))
            .collect();
        let ys: Vec<Vec<f32>> = if par_experts() {
            use rayon::prelude::*;
            ids.par_iter()
                .map(|&id| self.expert_ov_or_rust(id, x))
                .collect()
        } else {
            ids.iter()
                .map(|&id| self.expert_ov_or_rust(id, x))
                .collect()
        };
        let weights = gate.w.iter().chain(gate.gammas.iter());
        let mut out = vec![0.0f32; self.hidden];
        for (y, &wj) in ys.iter().zip(weights) {
            for (o, &yi) in out.iter_mut().zip(y) {
                *o += wj * yi;
            }
        }
        out
    }

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
        let gate = self.route_unobserved(x);
        if let Some(observer) = &self.route_observer {
            observer(&gate);
        }
        gate
    }

    /// Evaluate the router without reporting an actual expert selection or
    /// touching cache history. Used only by opt-in prediction diagnostics.
    pub(crate) fn route_unobserved(&self, x: &[f32]) -> GateOut {
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
        self.forward_with_prediction(x, None)
    }

    pub(super) fn forward_with_prediction(
        &self,
        x: &[f32],
        prediction: Option<super::predicted_read::PendingReadGroup>,
    ) -> Vec<f32> {
        if self.remote.is_some() {
            return self.forward_remote(x, 1);
        }
        assert!(
            self.has_local_experts(),
            "inkling MoE layer has no local experts and no expert-parallel client attached \
             (a driver built with ExpertSet::None must attach_remote before running)"
        );
        if self.ov_moe.is_some() {
            if let Some(y) = self.forward_ov_moe(x, 1) {
                return y;
            }
        }
        if self.ov.is_some() {
            return self.forward_ov(x);
        }
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
        if seq_reads() || !skip_bulk_prefetch() {
            for e in &sel {
                e.prefetch();
            }
        }
        // Overlapped reads: an mmap'd expert that is paged out is streamed
        // whole, concurrently with the others, into an owned buffer its GEMV
        // then runs from (bit-identical to the mmap). On the non-cache path an
        // already-resident expert is computed straight off the mapping — the
        // copy would only cost. The explicit-cache branch below deliberately
        // reads resident experts too, to admit valid bytes (documented there).
        let bulk_read = !seq_reads() && sel.iter().any(|e| e.as_mmap().is_some());
        let mut reused = (bulk_read && reuse_read_buffers())
            .then(|| super::read_buffers::ReadBuffers::acquire(sel.len()));
        let ys: Vec<Vec<f32>> = if pipeline_reads() && par_experts() && reused.is_some() {
            use rayon::prelude::*;
            PIPELINED_LAYERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let hits = self.expert_cache.lookup(&gate.idx);
            // A buffer stays exclusively borrowed through both the read and
            // its kernel. Ready experts can compute while other reads finish;
            // indexed collection still preserves gate accumulation order.
            let completed: Vec<(Vec<f32>, bool)> = sel
                .par_iter()
                .enumerate()
                .zip(reused.as_mut().unwrap().buffers.par_iter_mut())
                .map(|((index, expert), bytes)| {
                    if let Some(mapped) = expert.as_mmap() {
                        if let Some(Some(hit)) = hits.as_ref().and_then(|h| h.get(index)) {
                            return (mapped.swiglu_from(hit.as_slice(), x), false);
                        }
                        if let (Some(pending), Some(&expert)) = (&prediction, gate.idx.get(index)) {
                            if pending.take_for(expert, bytes) {
                                return (mapped.swiglu_from(bytes.as_slice(), x), true);
                            }
                        }
                        // With the explicit cache enabled, misses get a complete
                        // read even if OS sampling reports a resident mapping.
                        // This gives admission valid bytes and a measurable I/O
                        // cost; shared owned experts never enter the cache.
                        if hits.is_some() || !mapped.mostly_resident() {
                            match bytes.read(mapped.bin_path(), mapped.bin_len()) {
                                Ok(()) => {
                                    return (mapped.swiglu_from(bytes.as_slice(), x), true)
                                }
                                Err(err) => tracing::warn!(
                                    slot = index,
                                    bin = %mapped.bin_path().display(),
                                    "inkling decode expert read failed; using mmap compute fallback: {err}"
                                ),
                            }
                        }
                    }
                    (expert.forward(x, self.hidden, self.inter), false)
                })
                .collect();
            let caching = hits.is_some();
            drop(hits);
            if caching {
                // Retain in gate order after all compute, independent of the
                // parallel I/O schedule. Failed reads and hits are never admitted
                // from a scratch buffer left over from a different expert.
                for (index, &expert) in gate.idx.iter().enumerate() {
                    if completed[index].1 {
                        self.expert_cache
                            .retain(expert, &mut reused.as_mut().unwrap().buffers[index]);
                    }
                }
            }
            completed.into_iter().map(|(y, _)| y).collect()
        } else {
            let ready: Vec<bool> = if let Some(reused) = &mut reused {
                use rayon::prelude::*;
                sel.par_iter()
                    .enumerate()
                    .zip(reused.buffers.par_iter_mut())
                    .map(|((slot, e), bytes)| {
                        let Some(m) = e.as_mmap().filter(|m| !m.mostly_resident()) else {
                            return false;
                        };
                        match bytes.read(m.bin_path(), m.bin_len()) {
                            Ok(()) => true,
                            Err(err) => {
                                tracing::warn!(
                                    slot,
                                    bin = %m.bin_path().display(),
                                    "inkling decode expert read failed; using mmap compute fallback: {err}"
                                );
                                false
                            }
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let bufs: Vec<Option<Vec<u8>>> = if bulk_read && reused.is_none() {
                use rayon::prelude::*;
                sel.par_iter()
                    .map(|e| {
                        e.as_mmap()
                            .filter(|m| !m.mostly_resident())
                            .and_then(|m| m.read_bytes().ok())
                    })
                    .collect()
            } else {
                vec![None; sel.len()]
            };
            // The selected experts' FFNs run concurrently (each GEMV is itself
            // row-parallel; rayon's work stealing nests them). Every y_j is the
            // same value the serial loop produced, and the accumulation below
            // keeps gate order, so the result is bit-identical.
            let ffn = |(index, e): (usize, &&AnyExpert)| {
                let buf = match &reused {
                    Some(reused) if ready[index] => Some(reused.buffers[index].as_slice()),
                    Some(_) => None,
                    None => bufs[index].as_deref(),
                };
                match (buf, e.as_mmap()) {
                    (Some(b), Some(m)) => m.swiglu_from(b, x),
                    _ => e.forward(x, self.hidden, self.inter),
                }
            };
            if par_experts() {
                use rayon::prelude::*;
                sel.par_iter().enumerate().map(ffn).collect()
            } else {
                sel.iter().enumerate().map(ffn).collect()
            }
        };
        let mut out = vec![0.0f32; self.hidden];
        for (y, &wj) in ys.iter().zip(weights) {
            for (o, &yi) in out.iter_mut().zip(y) {
                *o += wj * yi;
            }
        }
        out
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
        if self.ov_moe.is_some() {
            if let Some(y) = self.forward_ov_moe(xs, rows) {
                return y;
            }
        }
        if self.ov.is_some() {
            // Per-expert IRs take one row at a time; rows run concurrently.
            // Same per-row result as `forward_ov`, so decode and prefill agree.
            let row = |br: usize| self.forward_ov(&xs[br * self.hidden..(br + 1) * self.hidden]);
            let ys: Vec<Vec<f32>> = if par_experts() {
                use rayon::prelude::*;
                (0..rows).into_par_iter().map(row).collect()
            } else {
                (0..rows).map(row).collect()
            };
            return ys.concat();
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
        self.forward_block_with_reads(xs, lo, hi, out, prefill_reads());
    }

    fn forward_block_with_reads(
        &self,
        xs: &[f32],
        lo: usize,
        hi: usize,
        out: &mut [f32],
        streamed: bool,
    ) {
        self.forward_block_impl(xs, lo, hi, out, streamed, row_gemm());
    }

    /// `gemm`: run an int4 expert's rows (and the shared experts' block) as one
    /// multi-input pass — see [`row_gemm`]. Bit-identical to `gemm = false`.
    fn forward_block_impl(
        &self,
        xs: &[f32],
        lo: usize,
        hi: usize,
        out: &mut [f32],
        streamed: bool,
        gemm: bool,
    ) {
        let (hidden, k) = (self.hidden, self.top_k);
        let nblk = hi - lo;
        let row = |br: usize| &xs[(lo + br) * hidden..(lo + br + 1) * hidden];

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
            if !streamed && !slots.is_empty() {
                self.w.experts[e].prefetch();
            }
        }
        for s in &self.w.shared {
            s.prefetch();
        }

        // 2b. The explicit expert cache (the streamed/owned-buffer path only):
        //     one lookup for the block's unique experts — a hit computes its
        //     rows straight from the retained bytes, a miss is read once and
        //     admitted after compute. This is what lets multi-stream decode
        //     (which batches rows exactly like prefill) run from the cache
        //     instead of re-reading every expert per step.
        let cache_on = streamed && self.expert_cache.stats().capacity_bytes > 0;
        let unique: Vec<usize> = occ
            .iter()
            .enumerate()
            .filter(|(_, slots)| !slots.is_empty())
            .map(|(e, _)| e)
            .collect();
        let hit_map: std::collections::HashMap<usize, Arc<super::read_buffers::ReadBuffer>> =
            if cache_on {
                match self.expert_cache.lookup(&unique) {
                    Some(hits) => unique
                        .iter()
                        .zip(hits)
                        .filter_map(|(&e, h)| h.map(|b| (e, b)))
                        .collect(),
                    None => Default::default(),
                }
            } else {
                Default::default()
            };

        // 3. One visit per unique routed expert, its rows hot. The experts
        //    run concurrently; each computes its rows back to back, so an
        //    mmap'd expert's int4 pages are still faulted in once.
        let mut ey = vec![0.0f32; nblk * k * hidden];
        let visit = |(e, slots): (usize, &Vec<usize>)| {
            let mapped = self.w.experts[e].as_mmap();
            if let (Some(m), Some(hit)) = (mapped, hit_map.get(&e)) {
                if gemm && slots.len() >= 2 {
                    let rows: Vec<&[f32]> = slots.iter().map(|&s| row(s / k)).collect();
                    return (e, m.swiglu_rows_from(hit.as_slice(), &rows));
                }
                let mut ys = Vec::with_capacity(slots.len() * hidden);
                for &s in slots {
                    let br = s / k;
                    let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
                    ys.extend_from_slice(&m.swiglu_from(hit.as_slice(), x));
                }
                return (e, ys);
            }
            let mut lease = (streamed && mapped.is_some())
                .then(|| super::read_buffers::ReadBuffers::acquire(1));
            let ready = match (mapped, lease.as_mut()) {
                (Some(mapped), Some(lease)) => {
                    match lease.buffers[0].read_prefill(mapped.bin_path(), mapped.bin_len()) {
                        Ok(()) => true,
                        Err(err) => {
                            tracing::warn!(
                                expert = e,
                                bin = %mapped.bin_path().display(),
                                "inkling prefill expert read failed; using mmap compute fallback: {err}"
                            );
                            false
                        }
                    }
                }
                _ => false,
            };
            let ys = if ready && gemm && slots.len() >= 2 {
                let rows: Vec<&[f32]> = slots.iter().map(|&s| row(s / k)).collect();
                mapped
                    .unwrap()
                    .swiglu_rows_from(lease.as_ref().unwrap().buffers[0].as_slice(), &rows)
            } else {
                let mut ys = Vec::with_capacity(slots.len() * hidden);
                for &s in slots {
                    let br = s / k;
                    let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
                    let y = if ready {
                        mapped
                            .unwrap()
                            .swiglu_from(lease.as_ref().unwrap().buffers[0].as_slice(), x)
                    } else {
                        self.w.experts[e].forward(x, hidden, self.inter)
                    };
                    ys.extend_from_slice(&y);
                }
                ys
            };
            if ready && cache_on {
                // Admit the freshly read bytes (a miss) after its rows computed;
                // the lease gets any evicted allocation back and drops it.
                self.expert_cache
                    .retain(e, &mut lease.as_mut().unwrap().buffers[0]);
            }
            (e, ys)
        };
        let visits: Vec<(usize, Vec<f32>)> = if streamed {
            // Nested Rayon GEMVs can suspend an outer expert task while its
            // buffer remains live. Fixed cohorts bound that retention to eight
            // experts, regardless of work-stealing order or prompt routing.
            let active: Vec<_> = occ
                .iter()
                .enumerate()
                .filter(|(_, slots)| !slots.is_empty())
                .collect();
            let mut visits = Vec::with_capacity(active.len());
            for cohort in active.chunks(8) {
                let completed: Vec<_> = if par_experts() {
                    use rayon::prelude::*;
                    cohort.par_iter().copied().map(visit).collect()
                } else {
                    cohort.iter().copied().map(visit).collect()
                };
                visits.extend(completed);
            }
            visits
        } else if par_experts() {
            use rayon::prelude::*;
            occ.par_iter()
                .enumerate()
                .filter(|(_, slots)| !slots.is_empty())
                .map(visit)
                .collect()
        } else {
            occ.iter()
                .enumerate()
                .filter(|(_, slots)| !slots.is_empty())
                .map(visit)
                .collect()
        };
        for (e, ys) in visits {
            for (i, &s) in occ[e].iter().enumerate() {
                ey[s * hidden..(s + 1) * hidden].copy_from_slice(&ys[i * hidden..(i + 1) * hidden]);
            }
        }
        // The shared experts (S_s(x_row), in s order). Every row uses both, so
        // with `gemm` each runs ONCE over the whole block (`shared_blk[s]` is
        // `[nblk, hidden]`); otherwise per row (`shared_y[br][s]`).
        let shared_blk: Vec<Vec<f32>> = if gemm && nblk >= 2 {
            let rows: Vec<&[f32]> = (0..nblk).map(row).collect();
            let one = |s: &AnyExpert| {
                super::ffn::forward_rows(s, &rows, hidden, self.inter, par_experts())
            };
            if par_experts() {
                use rayon::prelude::*;
                self.w.shared.par_iter().map(one).collect()
            } else {
                self.w.shared.iter().map(one).collect()
            }
        } else {
            Vec::new()
        };
        let shared_row = |br: usize| -> Vec<Vec<f32>> {
            let x = &xs[(lo + br) * hidden..(lo + br + 1) * hidden];
            self.w
                .shared
                .iter()
                .map(|s| s.forward(x, hidden, self.inter))
                .collect()
        };
        let shared_y: Vec<Vec<Vec<f32>>> = if !shared_blk.is_empty() || self.w.shared.is_empty() {
            Vec::new()
        } else if par_experts() {
            use rayon::prelude::*;
            (0..nblk).into_par_iter().map(shared_row).collect()
        } else {
            (0..nblk).map(shared_row).collect()
        };

        // 4. Per row: routed in gate order, then shared — forward()'s op order.
        for br in 0..nblk {
            let o = &mut out[(lo + br) * hidden..(lo + br + 1) * hidden];
            for slot in 0..k {
                let s = br * k + slot;
                let wj = slot_w[s];
                for (oo, &yi) in o.iter_mut().zip(&ey[s * hidden..(s + 1) * hidden]) {
                    *oo += wj * yi;
                }
            }
            let g = &gammas[br * self.n_shared..(br + 1) * self.n_shared];
            for (si, &gs) in g.iter().enumerate().take(self.w.shared.len()) {
                let y: &[f32] = match shared_blk.get(si) {
                    Some(blk) => &blk[br * hidden..(br + 1) * hidden],
                    None => &shared_y[br][si],
                };
                for (oo, &yi) in o.iter_mut().zip(y) {
                    *oo += gs * yi;
                }
            }
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
    /// Optional OpenVINO backend for this MLP (`(layer index, backend)`).
    ov: Option<(u32, Arc<super::ov_expert::OvExperts>)>,
    /// Optional all-rows-in-one-call device backend ([`super::ov_dense`]).
    ov_dense: Option<(u32, Arc<super::ov_dense::OvDense>)>,
    /// The same MLP as an all-slices-active fused-experts layer
    /// (`(layer, backend, slices)`, see [`Self::attach_ov_dense_moe`]).
    ov_dense_moe: Option<(u32, Arc<super::ov_moe::OvMoe>, usize)>,
}

impl DenseMlp {
    pub fn new(w: AnyExpert, inter: usize, global_scale: f32) -> Self {
        Self {
            w,
            inter,
            global_scale,
            ov: None,
            ov_dense: None,
            ov_dense_moe: None,
        }
    }

    /// Run this MLP through the GPU plugin's fused-experts op: the inner
    /// neurons cut into `slices` "experts" as wide as a routed one, every row
    /// selecting all of them with weight 1 (`tools/inkling_moe_layer_ov.py
    /// --dense-as-moe`, `<model>/dense_moe_ov`). `down(silu(gate x) * up x)`
    /// is a sum over inner neurons, so the slices add up to the same vector and
    /// no weight is requantised. Why: three compressed MatMuls 24,576 wide read
    /// their 4-bit weights at ~26 GB/s on the Arc B390 (9.6 ms a layer per
    /// frame), the fused op reads the same bytes at > 80 GB/s, and the two
    /// dense layers made rank 0 the slowest stage of the pipeline.
    pub fn attach_ov_dense_moe(
        &mut self,
        layer: u32,
        ov: Arc<super::ov_moe::OvMoe>,
        slices: usize,
    ) {
        self.ov_dense_moe = Some((layer, ov, slices));
    }

    /// `rows` rows through the fused-experts form; `None` = not attached or
    /// the device call failed (the next backend runs). Without `global_scale`.
    pub fn dense_moe_rows(&self, xs: &[f32], rows: usize) -> Option<Vec<f32>> {
        let (lid, ov, slices) = self.ov_dense_moe.as_ref()?;
        let ids: Vec<i32> = (0..rows).flat_map(|_| 0..*slices as i32).collect();
        let w = vec![1.0f32; rows * slices];
        ov.forward(*lid, xs, rows, &ids, &w)
    }

    /// The three-MatMul device form, for the load-time comparison.
    pub fn dense_ov_rows(&self, xs: &[f32], rows: usize) -> Option<Vec<f32>> {
        let (lid, ov) = self.ov_dense.as_ref()?;
        ov.forward(*lid, xs, rows)
    }

    /// Once at load: the fused-experts form against the form it replaces (the
    /// three-MatMul device call, else the host kernel) on two synthetic rows,
    /// and what a call of each costs (median of 17, microseconds). Goes out on
    /// a "stage profile" line; a form that disagrees (cosine <= 0.9999) is
    /// detached and the previous path keeps running.
    pub fn check_dense_moe(&mut self, hidden: usize) -> bool {
        let Some((lid, _, _)) = self.ov_dense_moe.as_ref() else {
            return true;
        };
        let lid = *lid;
        let rows = 2usize;
        let x: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i as f32 * 0.37).sin() + (i as f32 * 0.011).cos()) * 0.05)
            .collect();
        let reference = |me: &Self| -> Vec<f32> {
            me.dense_ov_rows(&x, rows).unwrap_or_else(|| {
                x.chunks_exact(hidden)
                    .flat_map(|r| me.w.forward(r, hidden, me.inter))
                    .collect()
            })
        };
        let median = |f: &dyn Fn() -> bool| -> u128 {
            let mut us: Vec<u128> = (0..20)
                .filter_map(|i| {
                    let t0 = std::time::Instant::now();
                    let ok = f();
                    (i >= 3 && ok).then(|| t0.elapsed().as_micros())
                })
                .collect();
            us.sort_unstable();
            us.get(us.len() / 2).copied().unwrap_or(0)
        };
        let got = self.dense_moe_rows(&x, rows);
        let want = reference(self);
        let (cosine, rel) = match got.as_ref() {
            Some(g) if g.len() == want.len() => {
                let dot: f64 = g.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
                let ng: f64 = g.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let nw: f64 = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let diff: f64 = g
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if ng > 0.0 && nw > 0.0 {
                    (dot / (ng * nw), diff / nw)
                } else {
                    (0.0, 1.0)
                }
            }
            _ => (0.0, 1.0),
        };
        let x1 = &x[..hidden];
        let moe1 = median(&|| self.dense_moe_rows(x1, 1).is_some());
        let moe2 = median(&|| self.dense_moe_rows(&x, 2).is_some());
        let ref1 = median(&|| self.dense_ov_rows(x1, 1).is_some());
        let ref2 = median(&|| self.dense_ov_rows(&x, 2).is_some());
        let ok = cosine.is_finite() && cosine > 0.9999;
        tracing::info!(target: "cascadia::inkling", event = "dense_moe_check", layer = lid, cosine, rel, moe1_us = moe1 as u64, ref1_us = ref1 as u64, ok);
        let line = format!(
            "DM{lid} probe stage profile layer={lid} cos_ppm={} rel_ppm={} moe1_us={moe1} moe2_us={moe2} ref1_us={ref1} ref2_us={ref2} ok={}",
            (cosine.clamp(0.0, 1.0) * 1e6) as u64,
            (rel.clamp(0.0, 1.0) * 1e6) as u64,
            u8::from(ok)
        );
        for _ in 0..3 {
            println!("{line}");
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
        if !ok {
            self.ov_dense_moe = None;
        }
        ok
    }

    /// Run this MLP's rows on the device backend (see [`super::ov_dense`]).
    pub fn attach_ov_dense(&mut self, layer: u32, ov: Arc<super::ov_dense::OvDense>) {
        self.ov_dense = Some((layer, ov));
    }

    pub fn ov_dense(&self) -> Option<(u32, &Arc<super::ov_dense::OvDense>)> {
        self.ov_dense.as_ref().map(|(l, o)| (*l, o))
    }

    pub fn attach_ov(&mut self, layer: u32, ov: Arc<super::ov_expert::OvExperts>) {
        self.ov = Some((layer, ov));
    }

    pub fn ov(&self) -> Option<(u32, &Arc<super::ov_expert::OvExperts>)> {
        self.ov.as_ref().map(|(l, o)| (*l, o))
    }

    /// Compile this MLP on the attached OV backend; `(compiled, failed)`.
    pub fn warm_ov(&self) -> Option<(usize, Vec<(u32, u32)>)> {
        let (lid, ov) = self.ov.as_ref()?;
        let bad = ov.warm(&[(*lid, super::ov_expert::DENSE)], 0);
        Some((1 - bad.len(), bad))
    }

    /// [`Self::forward`] for `rows` rows (`xs` = `[rows, hidden]`), bit-identical
    /// per row. The int4 weights (226 MB a layer) cross the memory bus once for
    /// the block instead of once per row: on the pipeline's rank 0 the two
    /// dense layers cost more per row than its four MoE layers and made it the
    /// slowest stage of the fleet.
    pub fn forward_rows(&self, xs: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
        if let Some(mut y) = self.dense_moe_rows(xs, rows) {
            for v in y.iter_mut() {
                *v *= self.global_scale;
            }
            return y;
        }
        if let Some((lid, ov)) = self.ov_dense.as_ref() {
            if let Some(mut y) = ov.forward(*lid, xs, rows) {
                for v in y.iter_mut() {
                    *v *= self.global_scale;
                }
                return y;
            }
        }
        if rows < 2 || self.ov.is_some() || !row_gemm() {
            let mut out = Vec::with_capacity(rows * hidden);
            for row in xs.chunks_exact(hidden) {
                out.extend(self.forward(row, hidden));
            }
            return out;
        }
        let views: Vec<&[f32]> = xs.chunks_exact(hidden).collect();
        let mut y = super::ffn::forward_rows(&self.w, &views, hidden, self.inter, true);
        for v in y.iter_mut() {
            *v *= self.global_scale;
        }
        y
    }

    /// `down(silu(gate·x) · up·x) · global_scale` for one token (`[hidden]`).
    pub fn forward(&self, x: &[f32], hidden: usize) -> Vec<f32> {
        if self.ov_dense.is_some() || self.ov_dense_moe.is_some() {
            return self.forward_rows(x, 1, hidden);
        }
        let mut y = match &self.ov {
            Some((lid, ov)) => ov
                .dense(*lid, x)
                .unwrap_or_else(|| self.w.forward(x, hidden, self.inter)),
            None => self.w.forward(x, hidden, self.inter),
        };
        for v in y.iter_mut() {
            *v *= self.global_scale;
        }
        y
    }
}

#[cfg(test)]
mod prefill_read_tests {
    use super::*;

    #[test]
    fn streamed_prefill_preserves_real_int4_bits_across_multiple_cohorts() {
        use crate::dsv4::expert_mmap::MmapExpert;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling_export/experts/layer_01");
        let mut router_w = vec![0.0; 16 * 64];
        for expert in 0..16 {
            router_w[expert * 64 + expert] = 2.0;
        }
        let weights = MoeWeights {
            router_w,
            router_bias: vec![0.0; 16],
            global_scale: 1.0,
            experts: (0..16)
                .map(|expert| {
                    AnyExpert::Mmap(
                        MmapExpert::open(
                            &directory.join(format!("expert_{:03}.bin", expert % 8)),
                            64,
                            32,
                        )
                        .unwrap(),
                    )
                })
                .collect(),
            shared: vec![],
        };
        let layer = MoeLayer::new(64, 32, 1, 1.0, weights);
        let mut xs = vec![0.0; 17 * 64];
        for row in 0..17 {
            xs[row * 64 + row % 16] = 4.0;
            assert_eq!(
                layer.route(&xs[row * 64..(row + 1) * 64]).idx,
                vec![row % 16]
            );
        }
        let mut expected = vec![0.0; xs.len()];
        let mut actual = vec![0.0; xs.len()];
        layer.forward_block_with_reads(&xs, 0, 17, &mut expected, false);
        layer.forward_block_with_reads(&xs, 0, 17, &mut actual, true);
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    /// `CASCADIA_INKLING_ROW_GEMM` on vs off, on the real int4 fixture bins: 8
    /// routed experts top-3 + both shared experts (one mapped, one an owned
    /// copy — `CASCADIA_INKLING_OWN_SHARED`), 29 rows so every expert carries
    /// several. The block through the multi-input kernel must equal the
    /// per-row kernels bit for bit, mapped and streamed, and so must a row
    /// computed alone (a stream decoded alone vs in a batch).
    #[test]
    fn row_gemm_block_is_bit_identical_to_the_per_row_kernels() {
        use crate::dsv4::expert_mmap::MmapExpert;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling_export/experts/layer_01");
        let open = |name: String| MmapExpert::open(&directory.join(name), 64, 32).unwrap();
        let (hidden, n_routed, n_shared, rows) = (64usize, 8usize, 2usize, 29usize);
        let mut seed = 0x9E37_79B9u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        };
        let weights = MoeWeights {
            router_w: (0..(n_routed + n_shared) * hidden)
                .map(|_| next())
                .collect(),
            router_bias: vec![0.0; n_routed],
            global_scale: 8.0,
            experts: (0..n_routed)
                .map(|e| AnyExpert::Mmap(open(format!("expert_{e:03}.bin"))))
                .collect(),
            shared: vec![
                AnyExpert::Mmap(open("expert_shared0.bin".into())),
                AnyExpert::Mmap(open("expert_shared1.bin".into()))
                    .into_owned_int4()
                    .unwrap(),
            ],
        };
        let layer = MoeLayer::new(hidden, 32, 3, 1.0, weights);
        let xs: Vec<f32> = (0..rows * hidden).map(|_| 4.0 * next()).collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut per_row = vec![0.0; xs.len()];
        layer.forward_block_impl(&xs, 0, rows, &mut per_row, false, false);
        assert!(per_row.iter().any(|&v| v != 0.0));
        for streamed in [false, true] {
            let mut gemm = vec![0.0; xs.len()];
            layer.forward_block_impl(&xs, 0, rows, &mut gemm, streamed, true);
            assert_eq!(bits(&gemm), bits(&per_row), "block, streamed={streamed}");
            let mut alone = vec![0.0; xs.len()];
            for r in 0..rows {
                layer.forward_block_impl(&xs, r, r + 1, &mut alone, streamed, true);
            }
            assert_eq!(bits(&alone), bits(&per_row), "alone, streamed={streamed}");
        }
    }
}
