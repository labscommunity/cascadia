//! GLM-5.2 stage runner — the [`StagedRunner`] the pipeline engine drives.
//! One contiguous layer slice per rank; `GlmRunner` itself is the staged
//! container (embed only on rank 0, head only on the last rank), so `GlmModel`
//! stays the untouched single-process form its goldens validate.
//!
//! Position: GLM's `AttentionLayer` appends KV at its own internal counter and
//! ignores the wire `pos` — the two stay in sync only via reset +
//! exactly-one-advance-per-forward. `forward_layers` asserts `pos == self.pos`
//! so a dropped/replayed frame is a loud worker death, never silent garbage.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::attn::AttnKv;
use super::kv_cache::SliceKvCache;
use super::loader::{load_stage, read_manifest, GlmManifest};
use super::model::GlmLayer;
use super::moe::AnyExpert;
use super::residency::{self, UsageStats};
use crate::dsv4::loader::{ExpertsMode, LoadError};
use crate::dsv4::math::{linear_bf16_w, linear_f32, rmsnorm};
use crate::dsv4::stage::even_layer_split;
use crate::staged::StagedRunner;

/// Default context budget when the caller passes none (the checkpoint allows up
/// to 1M; the KV caches scale with this).
pub const GLM5_DEFAULT_MAX_SEQ: usize = 4096;

/// How many of a layer's hottest routed experts to prefetch ahead (plus its
/// always-active shared expert). ~4x top_k covers the common routes.
const PREFETCH_EXPERTS: usize = 32;

/// The layer range `[lo, hi)` that rank `rank` of `total` owns.
///
/// Single source of truth for the split: [`GlmRunner::load_staged`] derives its
/// slice from this, and out-of-process consumers call it to fill the
/// `ShardDescriptor` layer range the scheduler's contiguity rule reads.
/// A descriptor that disagrees with what the engine actually loaded breaks chain
/// formation silently, so there must be exactly one implementation.
///
/// Ends are EXCLUSIVE. The guarantees below are asserted here rather than left to
/// callers, because none of them surface as an inference error:
///
/// - rank 0 starts at layer 0 and the last rank ends at `num_layers`
/// - ranges are contiguous and non-overlapping
/// - no rank owns zero layers
pub fn layer_split(m: &GlmManifest, rank: u32, total: u32) -> Result<(usize, usize), String> {
    let n = m.num_layers;
    let total = total.max(1);
    if rank >= total {
        return Err(format!("rank {rank} out of range for total {total}"));
    }
    let (lo, hi) = if !m.indexer_types.is_empty() {
        index_aligned_split(n, &m.indexer_types, rank, total)
    } else {
        even_layer_split(n, rank, total)
    };
    if lo >= hi {
        return Err(format!(
            "rank {rank} of {total} owns zero layers [{lo}, {hi}) — total exceeds the \
             model's {n} layers; reduce --total"
        ));
    }
    // `index_aligned_split` anchors rank 0 at `fulls[0]`, which is 0 only because
    // glm5's layer 0 happens to be "full". An export whose layer 0 is "shared"
    // would orphan `[0, fulls[0])` with nothing to catch it — the scheduler's
    // rule 3 checks adjacency between stages, never that the chain spans the
    // whole model.
    if rank == 0 && lo != 0 {
        return Err(format!(
            "split orphans layer 0: rank 0 starts at {lo}; the first indexer layer \
             must be \"full\""
        ));
    }
    if rank + 1 == total && hi != n {
        return Err(format!(
            "split orphans tail layers [{hi}, {n}): last rank must end at num_layers"
        ));
    }
    Ok((lo, hi))
}

/// Split `n` layers across `total` ranks so each rank's first layer is a `"full"`
/// IndexShare layer (owns an indexer / computes its own top-k). Boundaries are
/// the `"full"` positions nearest the even split, kept strictly increasing so
/// every rank is non-empty. Falls back to [`even_layer_split`] when there are
/// fewer full layers than ranks. Deterministic (same result on every rank).
fn index_aligned_split(n: usize, itypes: &[String], rank: u32, total: u32) -> (usize, usize) {
    let total = (total.max(1)) as usize;
    let is_full = |i: usize| itypes.get(i).map(|t| t == "full").unwrap_or(true);
    let fulls: Vec<usize> = (0..n).filter(|&i| is_full(i)).collect();
    if total <= 1 || fulls.len() < total {
        return even_layer_split(n, rank.min(total as u32 - 1), total as u32);
    }
    let mut starts = Vec::with_capacity(total);
    starts.push(fulls[0]); // 0 for GLM (layer 0 is "full")
    let mut used = 0usize; // index into `fulls` of the last chosen start
    for k in 1..total {
        let target = k * n / total;
        let max_j = fulls.len() - (total - k); // leave room for remaining starts
        let (mut best, mut bestd) = (used + 1, usize::MAX);
        for j in (used + 1)..=max_j {
            let d = fulls[j].abs_diff(target);
            if d < bestd {
                bestd = d;
                best = j;
            }
        }
        starts.push(fulls[best]);
        used = best;
    }
    let r = rank.min(total as u32 - 1) as usize;
    let lo = starts[r];
    let hi = if r + 1 < total { starts[r + 1] } else { n };
    (lo, hi)
}

/// A `vocab × hidden` edge table (embedding or lm_head) held either exact f32 or,
/// when `CASCADIA_GLM5_BF16_HEAD` is set, bf16 bits — halving its RAM (~3 GB →
/// ~1.5 GB per edge rank) so autopin can mlock more experts. bf16 rounds the
/// looked-up embedding (negligible) and the final logits (a precision change;
/// the model's attention shells are already bf16, but validate greedy parity
/// before defaulting this on). Only the edge ranks hold one.
enum WideTable {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl WideTable {
    /// Downcast a freshly-loaded f32 table to bf16 bits when `bf16` is set. The
    /// f32 buffer is dropped here, so call this BEFORE mlocking experts to make
    /// the freed RAM actually available to the pin budget.
    fn from_f32(v: Vec<f32>, bf16: bool) -> Self {
        if bf16 {
            WideTable::Bf16(
                v.iter()
                    .map(|&x| half::bf16::from_f32(x).to_bits())
                    .collect(),
            )
        } else {
            WideTable::F32(v)
        }
    }

    /// Row `t` (length `hidden`) as f32 — the embedding lookup.
    fn row(&self, t: usize, hidden: usize) -> Vec<f32> {
        let r = t * hidden..(t + 1) * hidden;
        match self {
            WideTable::F32(v) => v[r].to_vec(),
            WideTable::Bf16(v) => v[r]
                .iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect(),
        }
    }

    /// `out[vocab] = self[vocab, hidden] · x[hidden]` — the lm_head projection.
    /// The bf16 path uses the same kernel as the model's bf16 shells (f32 FMA
    /// accumulation, bf16-rounded output).
    fn matvec(&self, x: &[f32], vocab: usize, hidden: usize, out: &mut [f32]) {
        match self {
            WideTable::F32(w) => linear_f32(x, w, vocab, hidden, out),
            WideTable::Bf16(w) => linear_bf16_w(x, w, vocab, hidden, out),
        }
    }
}

/// Precision choices for the staged runner's f32-vs-bf16 tables. `Default` is
/// exact f32 — what the correctness tests and the single-process reference
/// compare against. Production selects precision via [`StageOpts::from_env`], so
/// the env is read in ONE place (not scattered through the loader).
// No `Copy`: `experts_mode` is an owned `String`. Callers that previously relied
// on implicit copies clone explicitly — this struct is built once per load.
#[derive(Clone, Default)]
pub struct StageOpts {
    /// embed + lm_head as bf16 (frees ~3 GB/edge-rank → more resident experts;
    /// ~1.2× decode, output within bf16 tolerance of f32).
    pub bf16_head: bool,
    /// MLA latent / k_pe KV cache as bf16 (frees KV RAM ∝ `max_seq`).
    pub bf16_kv: bool,
    /// Expert storage mode (`"eager"` | `"mmap"`). `None` → `CASCADIA_GLM5_EXPERTS`,
    /// else the size heuristic below.
    pub experts_mode: Option<String>,
    /// Async lookahead expert prefetch. `None` → `CASCADIA_GLM5_LOOKAHEAD`.
    pub lookahead: Option<bool>,
    /// Per-rank KV-prefix-cache depth. `None` → `CASCADIA_GLM5_PREFIX_CACHE`.
    pub prefix_cache_depth: Option<u32>,
    /// OV expert cache count backstop. `None` → `CASCADIA_GLM5_OV_CACHE`.
    pub ov_cache_entries: Option<u32>,
    /// OV expert cache byte budget (MiB), the primary bound. `None` →
    /// `CASCADIA_GLM5_OV_CACHE_MB`.
    pub ov_cache_mb: Option<u64>,
}

impl StageOpts {
    /// Production defaults from env: bf16 head **on** (opt out with
    /// `CASCADIA_GLM5_F32_HEAD`), bf16 KV **off** (opt in with
    /// `CASCADIA_GLM5_BF16_KV`). Call this only from the serving entry point;
    /// tests/examples pass an explicit `StageOpts` (default = exact f32).
    ///
    /// The three `Option` fields stay `None` here: their env fallback happens at
    /// their read sites in [`GlmRunner::load_staged`], where the existing parse
    /// logic already lives.
    pub fn from_env() -> Self {
        Self {
            bf16_head: !crate::glm::env_flag("CASCADIA_GLM5_F32_HEAD"),
            bf16_kv: crate::glm::env_flag("CASCADIA_GLM5_BF16_KV"),
            experts_mode: None,
            lookahead: None,
            prefix_cache_depth: None,
            ov_cache_entries: None,
            ov_cache_mb: None,
        }
    }

    /// Resolve from explicit config, falling back to env for any field left
    /// `None`.
    ///
    /// `f32_head` is modelled as the OPT-OUT to match `CASCADIA_GLM5_F32_HEAD`:
    /// bf16 head is the production default, so a `bf16_head` config field
    /// defaulting to `false` would silently cost ~1.2× decode on every
    /// deployment that didn't set it.
    ///
    /// Exists because in-process hosts cannot set the environment for a single
    /// engine (`set_var` is `unsafe` under edition 2024, and process-global).
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        f32_head: Option<bool>,
        bf16_kv: Option<bool>,
        experts_mode: Option<String>,
        lookahead: Option<bool>,
        prefix_cache_depth: Option<u32>,
        ov_cache_entries: Option<u32>,
        ov_cache_mb: Option<u64>,
    ) -> Self {
        let env = Self::from_env();
        Self {
            bf16_head: f32_head.map(|f| !f).unwrap_or(env.bf16_head),
            bf16_kv: bf16_kv.unwrap_or(env.bf16_kv),
            experts_mode,
            lookahead,
            prefix_cache_depth,
            ov_cache_entries,
            ov_cache_mb,
        }
    }
}

/// Prefill layer-streaming mode, from `CASCADIA_GLM5_PREFILL_STREAM`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum PrefillStream {
    /// Warm only routed experts whose pages probe non-resident (production).
    Gated,
    /// Warm every routed expert of the next layer (`=all`) — deterministic
    /// coverage for tests and an A/B lever for the residency gate.
    All,
}

/// Parse `CASCADIA_GLM5_PREFILL_STREAM`. Off-values follow
/// [`crate::glm::env_flag`]; `all` disables the residency gate; anything else
/// enables gated streaming.
fn parse_prefill_stream(v: Option<&str>) -> Option<PrefillStream> {
    let v = v.map(str::trim).unwrap_or("");
    if v.is_empty()
        || v == "0"
        || v.eq_ignore_ascii_case("false")
        || v.eq_ignore_ascii_case("no")
        || v.eq_ignore_ascii_case("off")
    {
        None
    } else if v.eq_ignore_ascii_case("all") {
        Some(PrefillStream::All)
    } else {
        Some(PrefillStream::Gated)
    }
}

pub struct GlmRunner {
    embed: Option<WideTable>,            // [vocab, hidden] on rank 0
    layers: Vec<GlmLayer>,               // this rank's slice
    head: Option<(Vec<f32>, WideTable)>, // (final_norm, lm_head) on the last rank
    hidden: usize,
    vocab: usize,
    eps: f32,
    max_seq: usize,
    eos: Vec<u32>,
    pos: usize,
    pub rank: u32,
    pub total: u32,
    /// Shared learned-pin routing histogram (attached to every owned MoE layer).
    usage: Arc<Mutex<UsageStats>>,
    /// Where [`Self::save_usage`] persists the histogram (`<dir>/.coli_usage`).
    usage_path: PathBuf,
    /// Router-lookahead prefetch enabled (CASCADIA_GLM5_PREFETCH) — histogram
    /// (learned-pin) prediction of the next layer's experts.
    prefetch: bool,
    /// Async lookahead prefetch (CASCADIA_GLM5_LOOKAHEAD): a background thread
    /// does REAL cross-layer reads of the predicted, non-resident experts
    /// (warming the OS page cache) so demand faults hit the cache. Predicts via
    /// the next layer's own router on an attention-free proxy. `Some` only on an
    /// mmap-expert rank when a consumer flag is set (decode lookahead and/or
    /// prefill streaming — the worker is shared). Takes precedence over
    /// `prefetch`.
    lookahead: Option<super::lookahead::Lookahead>,
    /// Decode-side router-proxy lookahead enabled (CASCADIA_GLM5_LOOKAHEAD).
    /// Kept separate from `lookahead`'s presence because the prefill streamer
    /// below shares the same worker without opting into decode prediction.
    lookahead_decode: bool,
    /// Prefill layer streaming (CASCADIA_GLM5_PREFILL_STREAM): while the
    /// batch-union prefill computes layer i, the lookahead worker sequentially
    /// warms layer i+1's not-yet-resident routed bins, so prefill's expert
    /// reads overlap compute instead of serializing between layers —
    /// FreeToken's full-layer double buffering (arXiv:2608.16157) on the OS
    /// page cache. No routing prediction: prefill touches most of the next
    /// layer's experts, so the whole (cold) set is warmed. Best-effort and
    /// output-identical. `All` skips the residency gate (tests / bench lever).
    prefill_stream: Option<PrefillStream>,
    /// Per-rank KV-prefix cache of this rank's layer slice, keyed by a prefix key
    /// rank 0 assigns. Disabled (cap 0) unless `CASCADIA_GLM5_PREFIX_CACHE` is set.
    prefix_cache: SliceKvCache,
}

impl GlmRunner {
    /// Load rank `rank` of `total`. `layer_start/layer_end` from the ShardSpec
    /// override the even split when nonzero.
    pub fn load_staged(
        dir: &Path,
        max_seq: usize,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
        opts: StageOpts,
    ) -> Result<Self, LoadError> {
        let m = read_manifest(dir)?;
        let n = m.num_layers;
        let total = total.max(1);
        let rank = rank.min(total - 1);
        let (lo, hi) = if layer_end > 0 {
            (layer_start as usize, layer_end as usize)
        } else {
            // Derived split — including the IndexShare boundary snapping — lives
            // in `layer_split` so out-of-process consumers publish the same range
            // this rank actually loads.
            layer_split(&m, rank, total).map_err(LoadError::Manifest)?
        };
        // A rank must own at least one layer; `total > num_layers` would leave a
        // dead rank on the wire, paying every round-trip's latency for nothing.
        //
        // Redundant for the derived arm (`layer_split` checks it too) but NOT for
        // the explicit `layer_start`/`layer_end` arm above, which never reaches
        // `layer_split` — an inverted ShardSpec range would otherwise sail past.
        if lo >= hi {
            return Err(LoadError::Manifest(format!(
                "rank {rank} of {total} owns zero layers [{lo}, {hi}) — total exceeds the \
                 model's {n} layers; reduce --total"
            )));
        }
        // IndexShare invariant: a rank's first owned layer must be `"full"` (owns
        // an indexer / computes its own top-k). A `"shared"` first layer needs the
        // top-k carry from a `"full"` layer on the PREVIOUS rank, which the
        // pipeline does not forward — attention would silently fall back to DENSE
        // (correct only while context ≤ index_topk; DIVERGES for long context). The
        // index-aligned split guarantees this, but an explicit ShardSpec
        // (`layer_start`/`layer_end`) may not, so reject a bad boundary loudly
        // rather than serve silently-wrong long-context output.
        if lo != 0
            && m.indexer_types
                .get(lo)
                .map(|t| t != "full")
                .unwrap_or(false)
        {
            return Err(LoadError::Manifest(format!(
                "rank {rank} starts on non-'full' IndexShare layer {lo} ('{}'): a rank \
                 boundary must land on a 'full' layer or that layer runs dense attention. \
                 Use index-aligned shard boundaries.",
                m.indexer_types[lo]
            )));
        }
        let first = rank == 0;
        let last = rank == total - 1;
        // Real-model expert sets can't be held dequantized; tiny/dev ones are
        // faster eager. CASCADIA_GLM5_EXPERTS=eager|mmap overrides.
        // Config first, then env, then the size heuristic — same precedence for
        // every knob below.
        let experts_mode = opts
            .experts_mode
            .clone()
            .or_else(|| std::env::var("CASCADIA_GLM5_EXPERTS").ok());
        let mode = match experts_mode.as_deref() {
            Some("eager") => ExpertsMode::Eager,
            Some("mmap") => ExpertsMode::Mmap,
            _ if m.num_experts > 32 => ExpertsMode::Mmap,
            _ => ExpertsMode::Eager,
        };
        // Precision comes from the caller (StageOpts), never env here — so the
        // pin-budget reservation and the actual KV/head allocations derive from
        // the SAME value and can't disagree.
        let (bf16_head, bf16_kv) = (opts.bf16_head, opts.bf16_kv);
        let mut s = load_stage(dir, max_seq, lo, hi, first, last, mode, bf16_kv)?;

        // Half-precision edge tables: store embed / lm_head as bf16 to free
        // ~3 GB/edge-rank back into the pin budget. Converted HERE — before
        // mlocking experts below — so the f32 buffers are dropped and the RAM is
        // actually available.
        let embed_tab = s.embed.take().map(|v| WideTable::from_f32(v, bf16_head));
        let head_tab = s
            .head
            .take()
            .map(|(norm, lm)| (norm, WideTable::from_f32(lm, bf16_head)));

        // --- learned-pin residency (cap_for_ram budget + AUTOPIN) -----------
        // Attach the routing histogram to every owned MoE layer (records which
        // experts fire), and mlock the hottest known experts in this slice up to
        // the RAM budget. No-op on the eager path (as_mmap -> None).
        let usage = Arc::new(Mutex::new(UsageStats::new()));
        let usage_path = dir.join(".coli_usage");
        let _ = usage.lock().unwrap().load(&usage_path);

        // Optional OpenVINO expert backend (iGPU / NPU / CPU), shared across this
        // rank's MoE layers. `Some` only when `CASCADIA_GLM5_OV_EXPERTS=1` and the
        // model ships an `experts_ov/` dir; otherwise every layer keeps the Rust
        // int4 path. Built once and Arc-shared so all layers share one LRU of
        // compiled IRs. Worth enabling only once a rank's experts are RAM-resident
        // (shard wide enough) — on a disk-streaming rank it just moves the same
        // NVMe wait onto the accelerator.
        let ov = super::ov_expert::OvExperts::from_env_with(
            dir,
            m.hidden_size,
            opts.ov_cache_entries,
            opts.ov_cache_mb,
        )
        .map(Arc::new);

        for (i, layer) in s.layers.iter_mut().enumerate() {
            if let Some(ml) = layer.moe_mut() {
                ml.attach_usage((lo + i) as u32, Arc::clone(&usage));
                if let Some(ov) = &ov {
                    ml.attach_ov(Arc::clone(ov));
                }
            }
        }

        // `CASCADIA_GLM5_NOPIN=1` disables ALL residency pinning: the hard
        // minimum-working-set raise below AND both pin passes.
        //
        // Distinct from `CASCADIA_GLM5_NOPIN_ACTIVE`, which gates only the
        // always-active pins — `reserve_lockable` and the hot-expert loop run
        // regardless of it. That makes NOPIN_ACTIVE unusable as a pin A/B: a run
        // with it set still raises the working-set minimum and still pins
        // hundreds of hot experts, so a failure proves nothing about pinning.
        //
        // Exists for the OV-on-GPU compile investigation: every expert fails to
        // compile on the iGPU inside the worker but compiles standalone, and the
        // leading hypothesis is that a hard-enforced working-set minimum (up to
        // 80% of available RAM) plus hundreds of VirtualLock'd bins starves the
        // GPU driver's residency request during weight upload. This knob is what
        // makes that testable; it is not itself the fix.
        let nopin_env = crate::glm::env_flag("CASCADIA_GLM5_NOPIN");
        // Pinning and accelerator offload are mutually exclusive on shared-RAM
        // devices: pins + held compiled experts crowd the pool GPU builds
        // allocate from (measured: combined = 100/100 CL_INVALID_EVENT, either
        // half alone = 0 failures).
        let ov_accel = ov.as_ref().is_some_and(|o| o.on_accelerator());
        if ov_accel && !nopin_env {
            tracing::info!(
                "skipping expert pinning: OV offload on an accelerator shares its RAM pool"
            );
        }
        let nopin = nopin_env || ov_accel;
        if mode == ExpertsMode::Mmap && !nopin {
            // Windows caps VirtualLock'd pages at the process minimum working set,
            // which defaults to a few MB — so raise it to cover everything we pin
            // (always-active + hot experts) before pinning, or every pin fails and
            // nothing stays resident. Sized from the pin budget and a sampled
            // expert size; capped at ~80% of available RAM. No-op off Windows.
            let budget = Self::pin_budget(&m, lo, hi, first, last, max_seq, bf16_head, bf16_kv);
            let per_expert = s
                .layers
                .iter()
                .find_map(|l| l.moe())
                .and_then(|ml| ml.experts().first())
                .and_then(AnyExpert::as_mmap)
                .map(|mm| mm.bin_len())
                .unwrap_or(19 * 1024 * 1024);
            let owned = (lo..hi).count();
            let want = (budget + owned * 4 + 16).saturating_mul(per_expert);
            let cap = (residency::mem_available() as usize / 5) * 4;
            crate::dsv4::expert_mmap::reserve_lockable(want.min(cap));

            // Always-active residency: the shared expert (fires every token) and
            // the dense-layer MLPs (first_k_dense_replace) are read on EVERY
            // forward, so under the routed-expert churn their pages get evicted
            // and re-faulted each token. mlock them unconditionally (~1.8 GB for
            // the full model) so they stay resident — guaranteed hits, ~1/9 of
            // active expert bytes. Disable for an A/B with CASCADIA_GLM5_NOPIN_ACTIVE=1.
            if !crate::glm::env_flag("CASCADIA_GLM5_NOPIN_ACTIVE") {
                let mut act = 0usize;
                for layer in s.layers.iter() {
                    let e = match (layer.moe(), layer.dense_expert()) {
                        (Some(ml), _) => ml.shared().as_mmap(),
                        (None, Some(d)) => d.as_mmap(),
                        _ => None,
                    };
                    if let Some(mm) = e {
                        if mm.pin().is_ok() {
                            act += 1;
                        }
                    }
                }
                if act > 0 {
                    eprintln!(
                        "[glm5] rank {rank}: mlock'd {act} always-active experts (shared + dense)"
                    );
                }
            }

            let (total, hot) = {
                let u = usage.lock().unwrap();
                (
                    u.total,
                    u.hottest_in_range(lo, hi, residency::autopin_count(u.total, budget)),
                )
            };
            let n_pin = hot.len();
            let mut pinned = 0usize;
            for (gl, e) in hot {
                let gl = gl as usize; // hottest_in_range already restricts to [lo, hi)
                if let Some(mm) = s.layers[gl - lo]
                    .moe()
                    .and_then(|ml| ml.experts().get(e as usize))
                    .and_then(AnyExpert::as_mmap)
                {
                    if mm.pin().is_ok() {
                        pinned += 1;
                    }
                }
            }
            if n_pin > 0 {
                eprintln!(
                    "[glm5] rank {rank}: mlock'd {pinned}/{n_pin} hot experts (budget {budget}, history {total})"
                );
            }
        }

        // Working-set governor: on the mmap path the working set grows far past
        // the pinned floor and Windows trims it lazily, so a 32 GB node settles
        // at a few hundred MB of MemAvailable while holding gigabytes of clean,
        // instantly-reclaimable pages — and the scheduler (correctly) marks it
        // unservable. When available RAM sinks below the threshold, trim the
        // excess to standby (pins and the hard floor survive; standby pages
        // soft-fault back without disk I/O). Threshold default 2560 MB — above
        // the scheduler's 1 GiB serve floor with margin; CASCADIA_GLM5_TRIM_MB
        // overrides, 0 disables. Healthy nodes never cross it.
        if mode == ExpertsMode::Mmap {
            let threshold_mb = std::env::var("CASCADIA_GLM5_TRIM_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(2560);
            if threshold_mb > 0 {
                Self::start_ws_governor(threshold_mb);
            }
        }

        // Async lookahead: build the per-layer expert-bin table (local index -> routed
        // bin paths) and spawn the background lookahead worker. Only on the mmap path
        // (streamed experts); eager/bf16 have nothing to warm.
        let lookahead_on = opts
            .lookahead
            .unwrap_or_else(|| crate::glm::env_flag("CASCADIA_GLM5_LOOKAHEAD"));
        // Prefill layer streaming shares the same worker; mmap ranks only
        // (eager/bf16 experts have nothing to warm).
        let prefill_stream = if mode == ExpertsMode::Mmap {
            parse_prefill_stream(
                std::env::var("CASCADIA_GLM5_PREFILL_STREAM")
                    .ok()
                    .as_deref(),
            )
        } else {
            None
        };
        let lookahead = if mode == ExpertsMode::Mmap && (lookahead_on || prefill_stream.is_some()) {
            let table: super::lookahead::LookaheadTable = s
                .layers
                .iter()
                .map(|l| l.moe().map(|ml| ml.expert_bins()))
                .collect();
            eprintln!(
                "[glm5] rank {rank}: lookahead prefetch thread started (decode={lookahead_on} prefill_stream={})",
                prefill_stream.is_some()
            );
            Some(super::lookahead::Lookahead::new(table))
        } else {
            None
        };

        Ok(Self {
            embed: embed_tab,
            layers: s.layers,
            head: head_tab,
            hidden: s.hidden,
            vocab: s.vocab,
            eps: s.eps,
            max_seq,
            eos: s.eos,
            pos: 0,
            rank,
            total,
            usage,
            usage_path,
            prefetch: crate::glm::env_flag("CASCADIA_GLM5_PREFETCH"),
            lookahead,
            lookahead_decode: lookahead_on,
            prefill_stream,
            prefix_cache: SliceKvCache::new(
                opts.prefix_cache_depth
                    .map(|d| d as usize)
                    .or_else(|| {
                        std::env::var("CASCADIA_GLM5_PREFIX_CACHE")
                            .ok()
                            .and_then(|s| s.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0),
            ),
        })
    }

    /// Spawn the process-wide working-set governor (once; engine rebuilds
    /// reuse it). Every `CASCADIA_GLM5_TRIM_SECS` (default 20) it samples
    /// `MemAvailable`; below `threshold_mb` it calls
    /// [`trim_working_set`](crate::dsv4::expert_mmap::trim_working_set) and
    /// logs the recovery. Windows-only in effect — the trim is a no-op
    /// elsewhere, so no thread is spawned.
    fn start_ws_governor(threshold_mb: u64) {
        #[cfg(windows)]
        {
            static STARTED: std::sync::Once = std::sync::Once::new();
            STARTED.call_once(move || {
                let period = std::env::var("CASCADIA_GLM5_TRIM_SECS")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .filter(|&s| s > 0)
                    .unwrap_or(20);
                std::thread::Builder::new()
                    .name("glm5-ws-governor".into())
                    .spawn(move || loop {
                        std::thread::sleep(std::time::Duration::from_secs(period));
                        let before_mb = super::residency::mem_available() / (1024 * 1024);
                        if before_mb >= threshold_mb {
                            continue;
                        }
                        let ok = crate::dsv4::expert_mmap::trim_working_set();
                        let after_mb = super::residency::mem_available() / (1024 * 1024);
                        tracing::info!(
                            target: "cascadia::glm5",
                            event = "ws_trimmed",
                            ok,
                            before_mb,
                            after_mb,
                            threshold_mb,
                        );
                    })
                    .ok();
            });
        }
        #[cfg(not(windows))]
        {
            let _ = threshold_mb;
        }
    }

    /// RAM budget (in whole experts) this rank may `mlock`, after reserving its
    /// bf16 dense shells, KV latent caches, the batch-union working set, and the
    /// page-cache reserve.
    fn pin_budget(
        m: &super::loader::GlmManifest,
        lo: usize,
        hi: usize,
        first: bool,
        last: bool,
        max_seq: usize,
        bf16_head: bool,
        bf16_kv: bool,
    ) -> usize {
        let owned = hi - lo;
        let heads = m.num_attention_heads;
        let qk_head = m.qk_nope_head_dim + m.qk_rope_head_dim;
        // Per-layer attention shell params (bf16): wq_a, wq_b, wkv_a, wkv_b, wo.
        let attn = m.q_lora_rank * m.hidden_size
            + heads * qk_head * m.q_lora_rank
            + (m.kv_lora_rank + m.qk_rope_head_dim) * m.hidden_size
            + heads * (m.qk_nope_head_dim + m.v_head_dim) * m.kv_lora_rank
            + m.hidden_size * heads * m.v_head_dim;
        let mut resident = (owned * attn) as u64 * 2;
        // Router projections (f32) on owned MoE layers.
        let moe_owned = (lo..hi).filter(|li| !m.dense_layers.contains(li)).count();
        resident += (moe_owned * m.num_experts * m.hidden_size) as u64 * 4;
        // Embedding / lm_head tables live on the edge ranks. f32 (4 B/elem) by
        // default; bf16 (2 B) when CASCADIA_GLM5_BF16_HEAD has freed that RAM, so
        // the pin budget grows by exactly what the downcast returned.
        let head_elem = if bf16_head { 2 } else { 4 };
        if first {
            resident += (m.vocab_size * m.hidden_size) as u64 * head_elem;
        }
        if last {
            resident += (m.vocab_size * m.hidden_size) as u64 * head_elem;
        }
        // Absorbed-MLA latent KV cache: (kv_lora + qk_rope) f32 per token per layer,
        // plus the DSA indexer key cache (index_head_dim f32 per token per OWNED
        // "full" layer). Both scale with max_seq — significant at agentic context.
        let full_owned = (lo..hi)
            .filter(|li| {
                m.indexer_types
                    .get(*li)
                    .map(|t| t == "full")
                    .unwrap_or(true)
            })
            .count();
        // Latent/k_pe cache: f32 (4 B) by default; bf16 (2 B) when
        // CASCADIA_GLM5_BF16_KV frees it. The DSA indexer keys stay f32 (4 B).
        let kv_elem = if bf16_kv { 2 } else { 4 };
        let kv =
            owned as u64 * max_seq as u64 * (m.kv_lora_rank + m.qk_rope_head_dim) as u64 * kv_elem
                + full_owned as u64 * max_seq as u64 * m.index_head_dim as u64 * 4;
        let eb = residency::int4_expert_bytes(m.hidden_size, m.expert_intermediate);
        residency::pin_expert_count(residency::mem_available(), resident, kv, eb)
    }

    /// Persist the learned-pin routing histogram to `<dir>/.coli_usage` so the
    /// next run mlocks a better initial set ("faster the more you use it").
    /// Each node writes its own file (it only records its own layers); best-effort.
    /// Enqueue local layer `li`'s routed experts for the lookahead worker to
    /// warm — the whole set under `All`, only the not-yet-resident ones under
    /// `Gated` (the probe costs microseconds; re-reading a hot bin costs
    /// milliseconds of NVMe bandwidth). Shared experts are excluded like the
    /// decode lookahead: they are pinned whenever pinning is on.
    fn stream_enqueue_layer(
        lk: &super::lookahead::Lookahead,
        layers: &[GlmLayer],
        li: usize,
        mode: PrefillStream,
    ) {
        if let Some(m) = layers[li].moe() {
            for e in 0..m.n_experts as u32 {
                if mode == PrefillStream::All || m.expert_cold(e) {
                    lk.enqueue(li, e);
                }
            }
        }
    }

    pub fn save_usage(&self) -> std::io::Result<()> {
        self.usage.lock().unwrap().save(&self.usage_path)
    }
}

impl Drop for GlmRunner {
    /// Persist the routing histogram on teardown (save-on-exit), so a
    /// run's routing feeds the next run's pin set. Skipped when nothing was ever
    /// recorded (an unused runner must not clobber existing history with empties).
    fn drop(&mut self) {
        let has_history = self.usage.lock().map(|u| u.total > 0).unwrap_or(false);
        if has_history {
            let _ = self.save_usage();
        }
        // Final OV-stats emission, ignoring the rate limit — otherwise up to one
        // interval's worth of the run's tail never reaches the log. No-op unless
        // CASCADIA_GLM5_OV_STATS is set.
        super::ov_expert::stats::dump_now();
    }
}

impl GlmRunner {
    /// Snapshot this rank's layer-slice KV (`[0, pos)`) for prefix caching.
    fn snapshot_slice(&self) -> Vec<AttnKv> {
        self.layers.iter().map(|l| l.snapshot_kv()).collect()
    }
}

impl StagedRunner for GlmRunner {
    fn arch_name(&self) -> &'static str {
        "glm5"
    }
    fn hidden_size(&self) -> usize {
        self.hidden
    }
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }
    fn supports_batched_prefill(&self) -> bool {
        true // glm layers ignore the token id; batch-union prefill is bit-exact
    }
    fn reset(&mut self) {
        self.pos = 0;
        for l in &mut self.layers {
            l.reset();
        }
    }

    fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache.enabled()
    }

    fn cache_prefix(&mut self, key: u64) {
        if self.prefix_cache.enabled() {
            let snap = self.snapshot_slice();
            self.prefix_cache.insert(key, snap);
        }
    }

    fn restore_prefix(&mut self, key: u64) -> Option<usize> {
        // Clone (not take+reinsert) so a hit does NOT change the cache's LRU
        // order: rank 0's token index and every rank's SliceKvCache must evict in
        // lockstep, which requires pure insertion-order LRU (no refresh on read).
        let snap = self.prefix_cache.get(key)?.clone();
        let len = snap.first().map(AttnKv::len).unwrap_or(0);
        for (l, kv) in self.layers.iter_mut().zip(&snap) {
            l.restore_kv(kv);
        }
        self.pos = len;
        Some(len)
    }
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let e = self
            .embed
            .as_ref()
            .expect("embed_token on a non-first rank");
        e.row(token as usize, self.hidden)
    }
    fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize, _token: Option<u32>) -> Vec<f32> {
        assert_eq!(
            pos, self.pos,
            "glm5 stage position desync (expected {}, got {pos})",
            self.pos
        );
        let mut x = hidden;
        // IndexShare carry threads within this rank's slice. (Cross-rank carry —
        // a "shared" layer at a rank boundary reusing an upstream full layer's
        // selection — is the not-yet-wired distributed case; it falls back to
        // full causal, correct for a single-rank run.)
        let mut carry: Option<Vec<usize>> = None;
        // Router-lookahead prefetch (opt-in, CASCADIA_GLM5_PREFETCH): while layer
        // i computes, madvise(WILLNEED) layer i+1's likely experts (shared + the
        // learned-pin hottest) so the NVMe read overlaps compute. Hint-only — the
        // output is identical whether or not it fires.
        let n = self.layers.len();
        for i in 0..n {
            if let (true, Some(lookahead)) = (self.lookahead_decode, self.lookahead.as_ref()) {
                // Async lookahead: mark the current layer, compute it, then
                // predict layer i+1's experts (its router on an attention-free
                // proxy of i's output) and enqueue the non-resident ones for the
                // background worker to warm. Hint-free, output-identical.
                lookahead.set_cur_layer(i);
                x = self.layers[i].forward_token(&x, &mut carry);
                if i + 1 < n {
                    let next = &self.layers[i + 1];
                    if let Some(m) = next.moe() {
                        let mut proxy = x.clone();
                        rmsnorm(&mut proxy, next.post_ln(), next.eps);
                        for e in m.predict_experts(&proxy) {
                            if m.expert_cold(e) {
                                lookahead.enqueue(i + 1, e);
                            }
                        }
                    }
                }
            } else if self.prefetch && i + 1 < n {
                let (head, tail) = self.layers.split_at_mut(i + 1);
                if let Some(m) = tail[0].moe() {
                    m.prefetch_hot(PREFETCH_EXPERTS);
                }
                x = head[i].forward_token(&x, &mut carry);
            } else {
                x = self.layers[i].forward_token(&x, &mut carry);
            }
        }
        self.pos += 1;
        // Per-token decode profile (no-op unless CASCADIA_GLM5_PROFILE is set);
        // in the pipeline each rank dumps its own layer-slice split.
        super::prof::dump("decode");
        // OV expert cache hit/miss + working-set stats (no-op unless
        // CASCADIA_GLM5_OV_STATS is set), emitted next to the decode profile.
        super::ov_expert::stats::dump();
        x
    }
    fn forward_layers_batch(&mut self, hidden: Vec<f32>, base: usize, rows: usize) -> Vec<f32> {
        assert_eq!(
            base, self.pos,
            "glm5 stage batch position desync (expected {}, got {base})",
            self.pos
        );
        assert_eq!(
            hidden.len(),
            rows * self.hidden,
            "glm5 batch: bad hidden length"
        );
        // Each layer runs per-position attention (KV in order) + batch-union MoE.
        let mut x = hidden;
        let mut carries: Vec<Option<Vec<usize>>> = vec![None; rows]; // per-row IndexShare
        let n = self.layers.len();
        // Prefill layer streaming (FreeToken-style double buffering,
        // arXiv:2608.16157): warm layer i+1's cold routed bins while layer i
        // computes, and layer 0's before the loop so its warm overlaps layer
        // 0's per-row attention. Enqueue-only + best-effort — output is
        // identical whether or not a warm lands in time.
        if let (Some(mode), Some(lk)) = (self.prefill_stream, self.lookahead.as_ref()) {
            lk.set_cur_layer(0);
            Self::stream_enqueue_layer(lk, &self.layers, 0, mode);
        }
        for i in 0..n {
            if let (Some(mode), Some(lk)) = (self.prefill_stream, self.lookahead.as_ref()) {
                lk.set_cur_layer(i);
                if i + 1 < n {
                    Self::stream_enqueue_layer(lk, &self.layers, i + 1, mode);
                }
            }
            x = self.layers[i].forward_prefill(&x, rows, &mut carries);
        }
        self.pos += rows;
        x
    }
    fn head_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (final_norm, lm_head) = self.head.as_ref().expect("head_logits on a non-last rank");
        let mut x = hidden.to_vec();
        rmsnorm(&mut x, final_norm, self.eps);
        let mut logits = vec![0.0f32; self.vocab];
        lm_head.matvec(&x, self.vocab, self.hidden, &mut logits);
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::WideTable;
    use super::{parse_prefill_stream, PrefillStream};

    /// `CASCADIA_GLM5_PREFILL_STREAM` parsing: off-values match `env_flag`'s
    /// off set, `all` skips the residency gate, anything else gates.
    #[test]
    fn prefill_stream_env_parse() {
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("no"),
            Some("OFF"),
        ] {
            assert_eq!(parse_prefill_stream(off), None, "{off:?} must disable");
        }
        for all in ["all", "ALL", " all "] {
            assert_eq!(parse_prefill_stream(Some(all)), Some(PrefillStream::All));
        }
        for on in ["1", "true", "gated", "yes"] {
            assert_eq!(
                parse_prefill_stream(Some(on)),
                Some(PrefillStream::Gated),
                "{on:?} must gate"
            );
        }
    }

    /// The bf16 edge table must round-trip its rows and produce logits within
    /// bf16 tolerance of the f32 path — the opt-in `CASCADIA_GLM5_BF16_HEAD`
    /// swap is a precision change, not a correctness one.
    #[test]
    fn widetable_bf16_tracks_f32() {
        let (vocab, hidden) = (5usize, 16usize);
        let w: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32 * 0.017 - 0.4).sin())
            .collect();
        let f = WideTable::from_f32(w.clone(), false);
        let b = WideTable::from_f32(w, true);

        // Embedding lookup: bf16 rounds each element to ~3 significant bits of
        // mantissa (rel err < 2^-8).
        let (rf, rb) = (f.row(2, hidden), b.row(2, hidden));
        for (a, c) in rf.iter().zip(&rb) {
            assert!((a - c).abs() <= a.abs() / 128.0 + 1e-3, "row {a} vs {c}");
        }

        // lm_head projection: logits accumulate in f32, so the only error is the
        // bf16-rounded weights + output — a few percent, never garbage.
        let x: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.1 - 0.3).collect();
        let mut lf = vec![0.0f32; vocab];
        let mut lb = vec![0.0f32; vocab];
        f.matvec(&x, vocab, hidden, &mut lf);
        b.matvec(&x, vocab, hidden, &mut lb);
        for (a, c) in lf.iter().zip(&lb) {
            assert!((a - c).abs() <= a.abs() * 0.05 + 1e-2, "logit {a} vs {c}");
        }
    }
}
