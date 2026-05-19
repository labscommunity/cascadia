//! Core sparse-MoE inference loop.
//!
//! Holds compiled handles for every shell + layer-0 + head, a manifest,
//! the per-layer KV caches, and an LRU cache of compiled experts. Driven
//! by [`Runner::generate_argmax`], which generates `max_tokens` greedy
//! tokens for a prompt.
//!
//! Not async, not Send. Each generation owns its own KV state; the
//! Engine wrapper above this drives one call at a time.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use half::bf16;
use tahoma_int4_gemm::layer0_int4::{
    embed_token_bf16, layer0_forward_decode_int4_with_capacity, Int4Layer0,
};
use tahoma_int4_gemm::safetensors_source::Shard;
use tahoma_int4_gemm::shell::{NUM_HEADS, N_ROUTED_EXPERTS, QK_HEAD_DIM, TOPK, V_HEAD_DIM};
use tahoma_int4_gemm::shell_int4::{
    shell_forward_decode_int4_multi_with_capacity, shell_forward_decode_int4_predict_n,
    shell_forward_decode_int4_with_capacity, Int4Shell,
};
use tahoma_int4_gemm::{
    expert_forward as int4_expert_forward, ExpertWeights, SafetensorsExpert,
    SafetensorsExpertSource,
};
use tahoma_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};
use thiserror::Error;
use tracing::{debug, info};

use crate::hot_buffer::{ExpertHits, LayerHotBuffer};
use crate::manifest::{Manifest, ManifestError};
use crate::tensors::{bf16_bytes_to_f32, f16_bytes_to_f32, f32_to_bf16_bytes};

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("manifest: {0}")]
    Manifest(#[from] ManifestError),
    #[error("ov: {0}")]
    Ov(#[from] OvError),
    #[error("model file missing: {0}")]
    MissingFile(PathBuf),
    #[error("internal: {0}")]
    Internal(String),
}

/// Initial KV-cache slot capacity per layer. Doubles on overflow so
/// the cumulative alloc/copy traffic is O(N) instead of the old O(N²)
/// from reallocating on every token. 32 fits the short-prompt eval
/// without a grow; long-context decodes pay log2(N/32) grows total.
const INITIAL_KV_CAPACITY: usize = 32;

/// Per-MoE-layer state held across generation steps.
///
/// The shell forward runs through `tahoma_int4_gemm::shell_int4::shell_forward_decode_int4_with_capacity`,
/// which expects flat KV caches in `[NUM_HEADS, capacity, D]` row-major layout where only the
/// first `past_seq_len` slots per head are populated. We track `kv_capacity` separately
/// from `past_seq_len` so a steady-state generation never triggers a realloc.
///
/// **autolab campaign 029 (A8): KV is stored as bf16-as-u16.** Halves
/// the per-layer footprint and the per-token bandwidth touched at
/// attention time (the dominant cost per q1 once expert dispatch is
/// thinned by A3 K-override). The SDPA kernel upconverts to f32 inline.
struct LayerState {
    lid: u32,
    int4_shell: Int4Shell,
    /// Layout: `[NUM_HEADS, kv_capacity, QK_HEAD_DIM]` row-major, bf16-as-u16.
    /// Slots `past_seq_len..kv_capacity` per head are reserved but
    /// unpopulated (their contents don't matter).
    past_k: Vec<u16>,
    /// Layout: `[NUM_HEADS, kv_capacity, V_HEAD_DIM]` row-major, bf16-as-u16.
    past_v: Vec<u16>,
    past_seq_len: usize,
    /// Slots allocated per head. Doubles on overflow. Survives across
    /// generations via `reset_kv` (resetting clears past_seq_len but
    /// keeps the buffers — the next prompt reuses the allocation).
    kv_capacity: usize,
}

/// First-stage layer-0 state: Rust int4 forward + KV cache + a
/// pinned mmap of the bf16 embed_tokens table.
///
/// The original code ran the layer-0 OV IR stateless on the full
/// prefix for every decode step (O(N²) attention across a generation).
/// This struct gives layer 0 its own pre-allocated KV cache so it
/// joins the shells on the O(N) per-token path. The embed_tokens
/// lookup is done in Rust against a mmap'd safetensors shard.
struct Layer0State {
    int4_layer0: Int4Layer0,
    /// Keeps the embed_tokens mmap alive as long as we hold the slice.
    _embed_pin: Arc<Shard>,
    /// Pointer into the mmap — bf16 `[vocab_size, HIDDEN]` flat.
    embed_tokens_bf16: &'static [u8],
    /// bf16-as-u16 KV (autolab 029 / A8).
    past_k: Vec<u16>,
    past_v: Vec<u16>,
    past_seq_len: usize,
    kv_capacity: usize,
}

impl LayerState {
    fn new(lid: u32, int4_shell: Int4Shell) -> Self {
        let cap = INITIAL_KV_CAPACITY;
        Self {
            lid,
            int4_shell,
            past_k: vec![0u16; NUM_HEADS * cap * QK_HEAD_DIM],
            past_v: vec![0u16; NUM_HEADS * cap * V_HEAD_DIM],
            past_seq_len: 0,
            kv_capacity: cap,
        }
    }
}

/// Two-mode expert cache: either compiled OV IR per (layer, expert), or
/// mmap'd flat int4 binaries served by the tahoma-int4-gemm AVX-512
/// kernel. The mode is fixed at construction time based on the
/// manifest's `experts_format` field.
enum ExpertCache {
    OvIr(OvIrExpertCache),
    Int4Bin(Int4BinExpertCache),
    SafetensorsBin(SafetensorsExpertCache),
}

struct OvIrExpertCache {
    model_dir: PathBuf,
    manifest_layer_xml: Box<dyn Fn(&PathBuf, u32, u32) -> PathBuf + Send>,
    device: String,
    plugin: PluginConfig,
    map: HashMap<(u32, u32), Runtime>,
    compile_count: u64,
    compile_secs: f64,
}

struct Int4BinExpertCache {
    model_dir: PathBuf,
    manifest_layer_bin: Box<dyn Fn(&PathBuf, u32, u32) -> PathBuf + Send>,
    /// Mmap'd expert weights — cheap to hold many of these since the OS
    /// pages them in lazily, so we keep all (layer, expert) pairs we've
    /// touched.
    map: HashMap<(u32, u32), ExpertWeights>,
}

/// Variant that reads experts directly from the safetensors shards
/// (`<model_dir>/safetensors/<shard>`) — no on-disk duplication.
struct SafetensorsExpertCache {
    /// Shared with the Runner; one mmap set, multiple consumers.
    source: Arc<SafetensorsExpertSource>,
    /// Cached SafetensorsExpert holders. Each pins its shard mmaps.
    map: HashMap<(u32, u32), SafetensorsExpert>,
}

impl OvIrExpertCache {
    fn get(&mut self, lid: u32, eid: u32) -> Result<&mut Runtime, RunnerError> {
        let key = (lid, eid);
        if !self.map.contains_key(&key) {
            let xml = (self.manifest_layer_xml)(&self.model_dir, lid, eid);
            let xml_s = xml
                .to_str()
                .ok_or_else(|| RunnerError::Internal("non-utf8 expert path".into()))?;
            let t0 = Instant::now();
            let rt = Runtime::compile(xml_s, &self.device, &self.plugin)?;
            self.compile_count += 1;
            self.compile_secs += t0.elapsed().as_secs_f64();
            self.map.insert(key, rt);
        }
        Ok(self.map.get_mut(&key).unwrap())
    }
}

impl Int4BinExpertCache {
    fn get(&mut self, lid: u32, eid: u32) -> Result<&ExpertWeights, RunnerError> {
        let key = (lid, eid);
        if !self.map.contains_key(&key) {
            let path = (self.manifest_layer_bin)(&self.model_dir, lid, eid);
            let w = ExpertWeights::open(&path).map_err(|e| {
                RunnerError::Internal(format!("open expert.bin {}: {}", path.display(), e))
            })?;
            self.map.insert(key, w);
        }
        Ok(self.map.get(&key).unwrap())
    }
}

/// autolab iter 029 (C1): one prefetch request — kindly hint the kernel
/// to start reading the weights for expert `eid` on layer `lid`. The
/// prefetcher thread translates this into `madvise(MADV_WILLNEED)` calls
/// on the six safetensors slices that make up the expert. Sent on a
/// bounded `sync_channel`; if the prefetcher falls behind we drop the
/// request (the read happens on demand later, which is the no-prefetch
/// baseline).
#[derive(Copy, Clone, Debug)]
struct PrefetchReq {
    lid: u32,
    eid: u32,
}

/// Background thread that consumes [`PrefetchReq`] from a channel and
/// issues `madvise(MADV_WILLNEED)` on each expert's six tensor byte
/// ranges. Owned by [`Runner`]; the channel sender is dropped on
/// `Runner` drop, terminating the consumer.
///
/// One thread is plenty — madvise is cheap (~µs per call) and the goal
/// is just to kick off async page-in. The bottleneck is the OS's
/// readahead queue, not our scheduler.
struct Prefetcher {
    /// SyncSender so we have bounded backpressure; if we overrun the
    /// queue we silently drop (the request just becomes a cache miss).
    tx: Option<SyncSender<PrefetchReq>>,
    join: Option<JoinHandle<()>>,
    /// Diagnostic counters (per-Runner lifetime).
    drops: Arc<AtomicU64>,
    submits: Arc<AtomicU64>,
    processed: Arc<AtomicU64>,
}

impl Prefetcher {
    fn spawn(source: Arc<SafetensorsExpertSource>) -> Self {
        // 16K slots = ~11 tokens of N=24 × 60 layers (iter 047 worst
        // case) or ~50 tokens at the iter 033 K=8 baseline. Generous
        // headroom: madvise is sub-µs per call so even when the
        // prefetcher is the slow side, the queue drains fast. Overrun
        // just drops the prefetch (fall-back to cache miss = the
        // no-prefetch baseline), so this only caps how far ahead we
        // can race the OS readahead.
        let (tx, rx) = mpsc::sync_channel::<PrefetchReq>(16 * 1024);
        let drops = Arc::new(AtomicU64::new(0));
        let submits = Arc::new(AtomicU64::new(0));
        let processed = Arc::new(AtomicU64::new(0));
        let source_for_thread = source.clone();
        let processed_thread = processed.clone();
        let join = thread::Builder::new()
            .name("expert-prefetch".into())
            .spawn(move || {
                // Plain blocking recv loop. Terminates when the sender
                // side is dropped (i.e. when the Runner is being torn
                // down).
                while let Ok(req) = rx.recv() {
                    let _hits = source_for_thread.prefetch_expert(req.lid, req.eid);
                    processed_thread.fetch_add(1, AtomicOrdering::Relaxed);
                }
            })
            .expect("spawn expert-prefetch thread");
        Self {
            tx: Some(tx),
            join: Some(join),
            drops,
            submits,
            processed,
        }
    }

    /// Non-blocking enqueue. Drops the request on overflow rather than
    /// stalling the inference path. Bumps the `submits` counter on
    /// success and `drops` on overflow/disconnect.
    fn try_submit(&self, lid: u32, eid: u32) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(PrefetchReq { lid, eid }) {
            Ok(()) => {
                self.submits.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.drops.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    /// Snapshot the (submits, drops, processed) counters. Used by the
    /// instrumentation log emitted every few tokens.
    fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.submits.load(AtomicOrdering::Relaxed),
            self.drops.load(AtomicOrdering::Relaxed),
            self.processed.load(AtomicOrdering::Relaxed),
        )
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Close the sender so the thread's recv loop terminates.
        drop(self.tx.take());
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Per-rank slice of the model the Runner should hold.
///
/// `layer_start..layer_end` is the half-open range of *MoE layer ids*
/// (1-based, matching the manifest convention) this Runner is
/// responsible for. The dense layer 0 is implicit and tracked
/// separately via `is_first` (only rank 0 needs it).
#[derive(Clone, Debug)]
pub struct LayerRange {
    pub layer_start: u32,
    pub layer_end: u32,
    pub is_first: bool,
    pub is_last: bool,
}

impl LayerRange {
    /// Range that loads everything — single-stage default.
    pub fn full() -> Self {
        Self {
            layer_start: 0,
            layer_end: u32::MAX,
            is_first: true,
            is_last: true,
        }
    }
}

/// Per-layer hot-expert buffer state. None means feature is disabled
/// for this layer (either globally disabled via `hot_expert_buffer_n == 0`
/// or we haven't built it yet — see `hot_buffers_built`).
type HotBufferMap = HashMap<u32, LayerHotBuffer>;

/// Main inference runner.
pub struct Runner {
    pub manifest: Manifest,
    _model_dir: PathBuf,
    _device: String,
    _plugin: PluginConfig,
    pub range: LayerRange,
    layer0: Option<Layer0State>,
    head: Option<Runtime>,
    layers: Vec<LayerState>,
    experts: ExpertCache,
    /// Shared safetensors handle used to construct each Int4Shell at
    /// load time and (when experts_format=safetensors_bin) to serve
    /// expert weights at runtime. Held in an Arc so the expert cache
    /// can clone it without duplicating mmaps.
    _safetensors_source: Arc<SafetensorsExpertSource>,
    /// autolab iter 069: per-(layer, expert) dispatch counts in the
    /// flat `ExpertHits` (HashMap-keyed by `(lid, eid)`) form. Used
    /// solely to drive the hot-buffer build trigger. Independent of
    /// the iter 054 `expert_hits` map below so the merge between the
    /// chain is non-destructive (this counter feeds hot_buffer logic;
    /// the chain's expert_hits feeds pin / cache-aware / spec prefetch
    /// / prefill-hint).
    hot_expert_hits: ExpertHits,
    /// Number of hot experts to pack per layer. `0` disables the
    /// feature entirely (the hit map is still tracked but no buffer
    /// is built).
    hot_expert_buffer_n: usize,
    /// Build the hot buffer once `hot_expert_hits.total >= warmup_dispatches`.
    /// Defaulted to 1 full forward pass per layer; the CLI exposes the
    /// raw token threshold via `--hot-expert-warmup-tokens`.
    hot_buffer_warmup_dispatches: u64,
    /// Layer-to-hot-buffer map. Only populated for layers held by this
    /// rank, only after warmup, and only when feature is enabled.
    /// Dispatch checks `hot_buffers.get(&lid)?.slice(eid)` first; misses
    /// fall through to the mmap source.
    hot_buffers: HotBufferMap,
    /// Latched flag: once we've built the hot buffer set, we don't
    /// rebuild it (would be a waste — distribution stabilizes fast).
    /// Reset only by `reset_hot_buffer` for tests.
    hot_buffers_built: bool,
    // ============================================================
    // chain (iter 033 + 047 + 054 + 056 + 057 + 065): tracks per-
    // position per-expert hit counts to drive: predictor (047),
    // pin (054), cache-aware dispatch (056), speculative prefetch
    // (057), and prefill-hint static schedule (065).
    // ============================================================
    /// autolab campaign 004 (A3): if Some(k') with k' < manifest.top_k,
    /// forward_shells dispatches only the first k' of the routed top-K
    /// experts per token. The shell's router still computes full top-K;
    /// we just skip the dispatch of the tail. Sigmoid-router models
    /// (K2.6 / DeepSeek-V3) tolerate this with minimal quality loss.
    top_k_override: Option<u32>,
    /// autolab campaign 007 (A2): if Some(t) with t > 0, skip experts
    /// whose routing weight falls below t. Applied AFTER top_k_override.
    /// Per-token effective K varies; safer than fixed K if router
    /// confidence is uneven.
    routing_threshold: Option<f32>,
    /// autolab iter 029 (C1): cache of the *previous* token's predicted
    /// next-token expert IDs per layer (indexed by position in
    /// `self.layers`, not by `lid`). Empty `Vec<u32>` means "no history
    /// yet" (just after `reset_kv` or first prefill token). At the
    /// start of each `forward_shells` we push these IDs to the
    /// prefetcher so the OS can start pulling pages while this token's
    /// earlier layers run.
    ///
    /// **autolab iter 047:** what we store here depends on
    /// `prefetch_n`. With `prefetch_n == TOPK` (default, == iter 033
    /// behavior) these are just the previous token's actually-fired
    /// IDs. With `prefetch_n > TOPK` these are the top-N by router
    /// score from the previous token's router — a wider net intended
    /// to land more of the next token's actually-fired experts in
    /// pre-paged memory. Tracked length per layer: `prefetch_n`.
    last_routing_ids: Vec<Vec<u32>>,
    /// autolab iter 029 (C1): background prefetcher fed by
    /// `last_routing_ids` at the start of each `forward_shells`. `None`
    /// when prefetching is disabled (env var `TAHOMA_EXPERT_PREFETCH=0`
    /// or experts_format != safetensors_bin).
    prefetcher: Option<Prefetcher>,
    /// autolab iter 047 (C1 better predictor): how many top-by-router-
    /// score experts to record per layer per token for next-token
    /// prefetch prediction. Must be in `[TOPK, N_ROUTED_EXPERTS]`.
    /// Default == [`TOPK`] (back-compat with iter 033 — same-as-last-
    /// token predictor). N > TOPK adds insurance: the `N - TOPK` extras
    /// cover the next token's likely-different expert selection at the
    /// cost of more `madvise(WILLNEED)` calls + more page-cache churn.
    prefetch_n: u32,
    /// autolab iter 047 (C1 better predictor): per-token hit-rate
    /// instrumentation. Counter pairs (correct_predictions, total) sum
    /// over the entire generation; deltas at decode-time tell us how
    /// well `last_routing_ids[i]` covered the experts the next token
    /// actually fired. "Correct" = an expert that fired in this token
    /// appeared in this token's `last_routing_ids[i]` (i.e. was
    /// pre-paged or at least prefetch-requested). Read by the
    /// instrumentation log and tests.
    prefetch_hits: u64,
    prefetch_chances: u64,
    /// autolab iter 054 (expert pinning): per-layer per-expert hit
    /// counts accumulated across every dispatched expert in
    /// `forward_shells`. Indexed by position in `self.layers` (same as
    /// `last_routing_ids`); the inner map is `expert_id → fire_count`.
    /// Used by `pin_top_n_per_layer` to identify the hot-set per layer
    /// for `mlock`. Persists across resets so steady-state pinning
    /// reflects the full workload distribution, not just one prompt.
    expert_hits: Vec<HashMap<u32, u64>>,
    /// autolab iter 054 (expert pinning): when `Some(n)`, after the
    /// first prompt completes (or `n` decoded tokens, whichever comes
    /// first) the runner pins the top-`n` experts *per layer* via
    /// `pin_top_n_per_layer`. Composes with C1 prefetch: pinned
    /// experts are immune to eviction; the prefetcher handles the
    /// long-tail unpinned experts. Default `None` ⇒ feature off.
    pin_top_n: Option<u32>,
    /// autolab iter 054: set to `true` after the first pin pass runs;
    /// gates re-pinning so we don't repeatedly call `mlock` across
    /// every generation. Re-pinning can be forced via
    /// `unpin_all_experts() + force_pin_top_n_per_layer`.
    pin_pass_done: bool,
    /// autolab iter 054: number of decoded tokens to accumulate hit
    /// data before the first pin pass fires. Lower = pins sooner but
    /// on less data (early prompt biased); higher = better hot-set
    /// quality but you eat the disk-IO penalty for longer. Default
    /// = 16 (one full sentence worth of router decisions, ~16 × 60
    /// = 960 dispatch events per layer ≈ heavy-tail emerges).
    pin_after_tokens: u32,
    /// autolab iter 054: count of forward_shells calls since the most
    /// recent `reset_kv`. Drives `pin_after_tokens` gating.
    decoded_tokens_since_reset: u32,
    /// autolab iter 056 (cache-aware dispatch): if `true`, reorder the
    /// per-layer top-K dispatch loop by descending hit count using
    /// `expert_hits[i]` so the hottest experts run first and stay
    /// L3-resident across layers. The output is bit-identical to the
    /// default router-score order because router weights are still
    /// summed in the original index order — only the `dispatch_expert`
    /// calls are reordered. Default `false` for back-compat.
    cache_aware_dispatch: bool,
    /// autolab iter 057 (async kernel scheduling — speculative prefetch):
    /// if `Some(n)` with `n > 0`, before each layer `i`'s expert dispatch
    /// begins, submit `madvise(WILLNEED)` for the top-`n` hit-frequent
    /// experts of layer `i + 1` (looked up via `expert_hits[i + 1]` and
    /// `select_top_n_by_hits`). The OS pages those experts in while
    /// layer `i`'s ~150 ms expert dispatch is the only thing on the
    /// critical path, so by the time layer `i + 1`'s router fires the
    /// page-cache miss bill is already paid.
    ///
    /// **Composes with iter 047 (whole-token predictor).** The iter 047
    /// predictor fires *once per token* before any layer runs, working
    /// from the previous token's routing decision. iter 057 fires *N - 1
    /// times per token* inside the layer loop, working from the current
    /// generation's accumulated `expert_hits`. The two predictors
    /// overlap (both madvise the same expert set when iter 057's
    /// top-n is also in iter 047's last_routing_ids) — duplicates are
    /// cheap (madvise on already-paged-in ranges is sub-µs) and the
    /// overlap shrinks as iter 047's prefetch_n widens. The point of
    /// iter 057 is that it kicks off prefetch *during* expert compute,
    /// not before it: the iter 047 prefetch races against layer 0's
    /// router (~5 ms), iter 057 races against layer i's expert dispatch
    /// (~150 ms). Disk-bound prefetches that need > 5 ms but < 150 ms
    /// are pure latency win for iter 057.
    ///
    /// **Correctness:** the prefetcher's `try_submit` is a non-blocking
    /// kernel hint. Wrong guesses waste OS readahead bandwidth (cache
    /// miss = the no-prefetch baseline) but cannot affect model output.
    /// The dispatch path still pulls weights from `dispatch_expert` and
    /// the actual routing decision is made fresh on real hidden states.
    ///
    /// Default `None` (off) for back-compat with iter 056 baseline. Set
    /// via `set_speculative_prefetch_n`.
    speculative_prefetch_n: Option<u32>,
    /// autolab iter 057: cumulative count of speculative prefetch
    /// submissions across the runner's lifetime. Bumped once per
    /// `(layer_i, expert)` pair that we tried to `try_submit` for the
    /// next-layer hot-set. Logged alongside the per-token iter 029 /
    /// 047 prefetch counters so A/B campaigns can attribute readahead
    /// bandwidth between the two predictors.
    speculative_prefetch_submitted: u64,
    /// autolab iter 065 (prefill-hint static schedule): per-layer
    /// per-expert observation counts accumulated **only during prefill**.
    /// Mirrors `expert_hits` in shape but is fed solely by prefill
    /// dispatches when `prefill_hint_weight > 0`. At the end of each
    /// prompt's prefill pass `exit_prefill_and_merge_hints` is called
    /// to fold these observations into `expert_hits` with the configured
    /// weight, so decode iteration #1 sees a `expert_hits` map already
    /// shaped by the prompt's actual routing — giving iter 054 (pin
    /// top-N), iter 056 (cache-aware dispatch), and iter 057 (speculative
    /// next-layer prefetch) a useful prior from token zero of decode
    /// rather than the empty / cold map they see today.
    ///
    /// Indexed by position in `self.layers` (same as `expert_hits`).
    /// Cleared by `reset_kv` so each prompt starts with a fresh
    /// observation window. Always allocated to `n_layers` length so
    /// indexing is panic-free regardless of whether the hint is enabled.
    prefill_expert_observations: Vec<HashMap<u32, u64>>,
    /// autolab iter 065 (prefill-hint static schedule): merge weight for
    /// folding `prefill_expert_observations` into `expert_hits` at the
    /// end of prefill. `0.0` (default) **disables the entire hint path**:
    /// `forward_shells` falls back to bumping `expert_hits` during prefill
    /// exactly as iter 054 does today, and `prefill_expert_observations`
    /// is never populated or merged.
    ///
    /// With `w > 0.0`, prefill dispatches stop bumping `expert_hits` and
    /// instead bump `prefill_expert_observations`. At the end of prefill
    /// (driven by `exit_prefill_and_merge_hints`) we fold each observation
    /// into `expert_hits[i][eid] += round(w * obs_count)`. So:
    ///
    ///   - `w = 1.0` matches today's iter 054 behavior: prefill firings
    ///     count 1:1 with decode firings.
    ///   - `w = 0.5` cuts the prefill prior in half — useful when prompt
    ///     vocabulary diverges sharply from decode vocabulary.
    ///   - `w = 2.0` over-weights the prior — useful when the prompt is
    ///     known to be highly representative of the eventual decode
    ///     distribution (e.g. continuation tasks).
    ///
    /// Set via `set_prefill_hint_weight` from the engine config.
    prefill_hint_weight: f32,
    /// autolab iter 065: gate used by `forward_shells` to decide whether
    /// expert-hit bumps go to `expert_hits` (decode, or hint disabled)
    /// or to `prefill_expert_observations` (prefill, hint enabled).
    /// Toggled by `enter_prefill` / `exit_prefill_and_merge_hints`. The
    /// gate is layered on top of `prefill_hint_weight > 0.0` so the
    /// flag itself is harmless when the hint is disabled — the dispatch
    /// loop checks the weight first and short-circuits.
    ///
    /// Distributed callers (engine `drive_generation_first` /
    /// `step_worker`) own the same enter/exit calls so prefill on every
    /// rank tracks the same set of observations independently.
    in_prefill: bool,
}

impl Runner {
    /// Compile only the layers + head + layer0 needed for this rank.
    ///
    /// Single-stage callers can pass `LayerRange::full()` to keep the
    /// pre-pipeline-parallel behavior. Distributed callers pass a
    /// range covering just their stage's layers, with `is_first` /
    /// `is_last` set accordingly — non-first stages skip layer 0
    /// compilation; non-last stages skip the head.
    pub fn load(
        model_dir: PathBuf,
        device: &str,
        plugin: PluginConfig,
        range: LayerRange,
    ) -> Result<Self, RunnerError> {
        let manifest = Manifest::load(&model_dir)?;
        info!(
            arch = %manifest.arch,
            num_layers = manifest.num_layers,
            num_experts = manifest.num_experts,
            top_k = manifest.top_k,
            layer_start = range.layer_start,
            layer_end = range.layer_end,
            is_first = range.is_first,
            is_last = range.is_last,
            "loading sparse-MoE model"
        );

        let utf8 = |p: &PathBuf| -> Result<String, RunnerError> {
            p.to_str()
                .map(str::to_owned)
                .ok_or_else(|| RunnerError::Internal(format!("non-UTF-8 path: {}", p.display())))
        };

        // layer 0 is constructed below from the safetensors source
        // (we need it open first). Defer until after.
        let mut layer0_holder: Option<Layer0State> = None;

        let head = if range.is_last {
            let head_xml = manifest.head_xml(&model_dir);
            if !head_xml.exists() {
                return Err(RunnerError::MissingFile(head_xml));
            }
            let rt = Runtime::compile(&utf8(&head_xml)?, device, &plugin)?;
            info!("compiled head (RMSNorm + lm_head)");
            Some(rt)
        } else {
            info!("skipping head (not last stage)");
            None
        };

        // Shells always come from safetensors now (the OV shell IRs are
        // numerically broken for K2.6 — see k26_output_divergence). The
        // safetensors source is the same one experts use when
        // experts_format=safetensors_bin, so we open it once and share.
        let st_dir = model_dir.join("safetensors");
        let st_dir = if st_dir.exists() {
            st_dir
        } else {
            model_dir.clone()
        };
        let safetensors_source = Arc::new(
            SafetensorsExpertSource::open(st_dir)
                .map_err(|e| RunnerError::Internal(format!("safetensors open: {e}")))?,
        );

        if range.is_first {
            // Construct dense layer 0 from safetensors. The old OV IR
            // path was stateless — each decode step ran the full prefix
            // through attention, making prefill O(N²). The Rust path
            // owns a pre-allocated KV cache that mirrors the shells.
            let st_layer0 = safetensors_source
                .layer0()
                .map_err(|e| RunnerError::Internal(format!("safetensors layer0: {e}")))?;
            let int4_layer0 = Int4Layer0::from_safetensors(&st_layer0);
            let (embed_pin, embed_bytes) = safetensors_source
                .embed_tokens()
                .map_err(|e| RunnerError::Internal(format!("safetensors embed_tokens: {e}")))?;
            let cap = INITIAL_KV_CAPACITY;
            layer0_holder = Some(Layer0State {
                int4_layer0,
                _embed_pin: embed_pin,
                embed_tokens_bf16: embed_bytes,
                past_k: vec![0u16; NUM_HEADS * cap * QK_HEAD_DIM],
                past_v: vec![0u16; NUM_HEADS * cap * V_HEAD_DIM],
                past_seq_len: 0,
                kv_capacity: cap,
            });
            info!("constructed Rust int4 layer 0 + embed_tokens mmap");
        } else {
            info!("skipping layer 0 (not first stage)");
        }

        let all_moe_ids = manifest.moe_layer_ids();
        let in_range: Vec<u32> = all_moe_ids
            .iter()
            .copied()
            .filter(|&lid| lid >= range.layer_start && lid < range.layer_end)
            .collect();
        info!(
            "this rank holds {} MoE layers (of {} total)",
            in_range.len(),
            all_moe_ids.len()
        );
        let mut layers = Vec::with_capacity(in_range.len());
        let shell_load_t0 = Instant::now();
        for (i, &lid) in in_range.iter().enumerate() {
            let st_shell = safetensors_source
                .shell(lid)
                .map_err(|e| RunnerError::Internal(format!("safetensors shell L{lid}: {e}")))?;
            let int4_shell = Int4Shell::from_safetensors(&st_shell);
            layers.push(LayerState::new(lid, int4_shell));
            if (i + 1) % 10 == 0 || i + 1 == in_range.len() {
                info!(
                    "loaded int4 shells {}/{} ({:.1}s elapsed)",
                    i + 1,
                    in_range.len(),
                    shell_load_t0.elapsed().as_secs_f64()
                );
            }
        }

        let manifest_clone = manifest.clone();
        let experts = match manifest.experts_format.as_str() {
            "int4_bin" => {
                info!("expert backend: int4_bin (mmap + AVX-512 kernel)");
                ExpertCache::Int4Bin(Int4BinExpertCache {
                    model_dir: model_dir.clone(),
                    manifest_layer_bin: Box::new(move |md, lid, eid| {
                        manifest_clone.expert_bin(md, lid, eid)
                    }),
                    map: HashMap::new(),
                })
            }
            "safetensors_bin" => {
                info!("expert backend: safetensors_bin (shared with shell source)");
                ExpertCache::SafetensorsBin(SafetensorsExpertCache {
                    source: safetensors_source.clone(),
                    map: HashMap::new(),
                })
            }
            other => {
                if other != "ov_ir" {
                    return Err(RunnerError::Internal(format!(
                        "unknown experts_format {:?}; expected 'ov_ir', 'int4_bin', or 'safetensors_bin'",
                        other
                    )));
                }
                info!("expert backend: ov_ir (per-expert OV CPU plugin call)");
                ExpertCache::OvIr(OvIrExpertCache {
                    model_dir: model_dir.clone(),
                    manifest_layer_xml: Box::new(move |md, lid, eid| {
                        manifest_clone.expert_xml(md, lid, eid)
                    }),
                    device: device.to_string(),
                    plugin: plugin.clone(),
                    map: HashMap::new(),
                    compile_count: 0,
                    compile_secs: 0.0,
                })
            }
        };

        // autolab iter 029 (C1): spin up the prefetcher thread when we're
        // serving experts directly from safetensors mmaps. Disabled when
        // TAHOMA_EXPERT_PREFETCH=0 so it's easy to A/B at runtime. Other
        // expert backends (ov_ir, int4_bin) don't benefit from madvise
        // because their weights aren't served from the safetensors mmaps,
        // so the prefetcher would do nothing useful there.
        let prefetcher = match (&experts, std::env::var("TAHOMA_EXPERT_PREFETCH").as_deref()) {
            (ExpertCache::SafetensorsBin(_), Ok("0")) => {
                info!("expert prefetch: disabled via TAHOMA_EXPERT_PREFETCH=0");
                None
            }
            (ExpertCache::SafetensorsBin(_), _) => {
                info!(
                    "expert prefetch: enabled (madvise(WILLNEED) on predicted next-token experts)"
                );
                Some(Prefetcher::spawn(safetensors_source.clone()))
            }
            _ => None,
        };
        let last_routing_ids: Vec<Vec<u32>> = (0..layers.len()).map(|_| Vec::new()).collect();
        let expert_hits: Vec<HashMap<u32, u64>> =
            (0..layers.len()).map(|_| HashMap::new()).collect();
        // autolab iter 065 (prefill-hint static schedule): per-prompt
        // observation buffer; always allocated to n_layers so the
        // dispatch loop can `get_mut(i)` without bounds checks.
        let prefill_expert_observations: Vec<HashMap<u32, u64>> =
            (0..layers.len()).map(|_| HashMap::new()).collect();

        Ok(Self {
            manifest,
            _model_dir: model_dir,
            _device: device.to_string(),
            _plugin: plugin,
            range,
            layer0: layer0_holder,
            head,
            layers,
            experts,
            _safetensors_source: safetensors_source,
            top_k_override: None,
            routing_threshold: None,
            last_routing_ids,
            prefetcher,
            // Default: N == TOPK → same-as-last-token predictor. With
            // no A2 threshold and no A3 K-override, this is
            // behaviorally identical to iter 033 (the prefetched IDs
            // are the same TOPK that fired). With A2 or A3 active,
            // we prefetch the full TOPK by router score rather than
            // only the dispatched k' — strictly more prefetch (~480
            // madvise/tok vs 360 at K=6) but the same predictive
            // accuracy, because madvise is non-blocking sub-µs.
            prefetch_n: TOPK as u32,
            prefetch_hits: 0,
            prefetch_chances: 0,
            expert_hits,
            pin_top_n: None,
            pin_pass_done: false,
            // 16 tokens ≈ 1 sentence; the heavy-tail expert distribution
            // emerges after ~10 tokens × 60 layers × ~K experts =
            // ~5000+ dispatch events, which is enough to make the
            // top-N stable. Configurable via `set_pin_after_tokens`.
            pin_after_tokens: 16,
            decoded_tokens_since_reset: 0,
            // iter 056: opt-in cache-aware dispatch order. Default off
            // for back-compat with iter 047 / 054 — A/B campaigns will
            // toggle this once the branch lands.
            cache_aware_dispatch: false,
            // iter 057: opt-in speculative next-layer prefetch. Default
            // off for back-compat with iter 056 baseline. A/B campaigns
            // toggle this with `--speculative-prefetch <N>` once the
            // branch lands.
            speculative_prefetch_n: None,
            speculative_prefetch_submitted: 0,
            // iter 065: prefill-hint static schedule. Default disabled
            // (weight 0.0) for back-compat with iter 057 — prefill
            // bumps `expert_hits` directly as iter 054 does today.
            // A/B campaigns toggle with `--prefill-hint-weight <W>`.
            prefill_expert_observations,
            prefill_hint_weight: 0.0,
            in_prefill: false,
            // iter 069 (hot-expert buffer): independent per-(layer,
            // expert) hit counter feeding only the hot-buffer build
            // logic. Disabled by default; enabled via
            // `set_hot_expert_buffer_config` from the engine config.
            hot_expert_hits: ExpertHits::new(),
            hot_expert_buffer_n: 0,
            // Default: ~3 full forward passes' worth of dispatches
            // before we trust the distribution. K2.6 fires
            // n_layers × top_k = 60 × 8 = 480 experts per token, so
            // the default 1500-dispatch warmup is ~3 tokens. Override
            // via `set_hot_expert_buffer_config`.
            hot_buffer_warmup_dispatches: 1500,
            hot_buffers: HotBufferMap::new(),
            hot_buffers_built: false,
        })
    }

    // ============================================================
    // iter 069 (hot-expert buffer) accessors.
    // ============================================================

    /// Configure the hot-expert buffer.
    ///
    /// - `n` == 0: disabled (the default). The hit map is still tracked
    ///   so per-layer top-N counts are available for inspection, but
    ///   no buffer is built and dispatch always uses the mmap source.
    /// - `n` > 0: after `warmup_dispatches` total expert dispatches,
    ///   build a hot buffer per layer containing the top-`n` experts.
    ///   Memory cost is `n × per-expert-bytes × n_layers_on_this_rank`
    ///   (~25 MiB per expert per layer for K2.6).
    pub fn set_hot_expert_buffer_config(&mut self, n: usize, warmup_dispatches: u64) {
        self.hot_expert_buffer_n = n;
        self.hot_buffer_warmup_dispatches = warmup_dispatches;
        // Reset latch so a re-config takes effect on the next dispatch
        // after the new warmup threshold is reached.
        self.hot_buffers.clear();
        self.hot_buffers_built = false;
    }

    /// Inspect the iter 069 flat per-(layer, expert) dispatch hit count
    /// total. Used by tests to verify the warmup gate fires; the chain's
    /// `expert_hits_snapshot` returns the per-position map used by
    /// pin / cache-aware dispatch / spec-prefetch / prefill-hint.
    pub fn hot_expert_hits_total(&self) -> u64 {
        self.hot_expert_hits.total
    }

    /// Build the per-layer hot buffers from the current dispatch
    /// histogram. Called automatically inside `dispatch_expert` once
    /// the warmup threshold is crossed; exposed publicly so callers
    /// (e.g. tests) can force a build at a known point.
    pub fn build_hot_buffers_now(&mut self) -> Result<(), RunnerError> {
        if self.hot_expert_buffer_n == 0 {
            return Ok(());
        }
        let n = self.hot_expert_buffer_n;
        // Only build for layers this rank actually holds — building
        // a buffer for a layer we don't dispatch is wasted memory.
        let lids_held: Vec<u32> = self.layers.iter().map(|l| l.lid).collect();
        let mut total_bytes: usize = 0;
        for lid in lids_held {
            // Skip layers with no recorded hits — they likely had no
            // chance to fire yet (e.g. distribution is heavily skewed
            // toward the early shells in a short warmup). Building a
            // hot buffer with the first N numeric eids would just
            // burn memory on a wild guess.
            let top = self.hot_expert_hits.top_n_for_layer(lid, n);
            if top.is_empty() {
                continue;
            }
            let lhb = LayerHotBuffer::build(&self._safetensors_source, lid, &top)
                .map_err(|e| RunnerError::Internal(format!("hot buffer L{lid}: {e}")))?;
            total_bytes += lhb.bytes();
            self.hot_buffers.insert(lid, lhb);
        }
        self.hot_buffers_built = true;
        info!(
            n = n,
            layers_built = self.hot_buffers.len(),
            bytes_mib = (total_bytes as f64) / (1024.0 * 1024.0),
            "hot expert buffers built"
        );
        Ok(())
    }

    /// Set the per-token top-K dispatch override. None = use manifest's top_k.
    /// Values > manifest.top_k are silently clamped down (no point dispatching
    /// experts the router didn't route to).
    pub fn set_top_k_override(&mut self, v: Option<u32>) {
        self.top_k_override = v;
        info!(
            top_k_override = ?v,
            manifest_top_k = self.manifest.top_k,
            "set_top_k_override"
        );
    }

    /// Set the per-token routing-weight threshold (A2). None / 0.0 = disabled.
    /// Experts whose routing weight is < threshold are skipped during
    /// forward_shells dispatch.
    pub fn set_routing_threshold(&mut self, v: Option<f32>) {
        self.routing_threshold = v.filter(|t| *t > 0.0);
        info!(routing_threshold = ?self.routing_threshold, "set_routing_threshold");
    }

    /// autolab iter 047 (C1 better predictor): set how many top-by-
    /// router-score expert IDs to record per layer per token for
    /// next-token prefetch prediction. Silently clamped to
    /// `[TOPK, N_ROUTED_EXPERTS]`. `None` keeps the default (`TOPK`,
    /// same-as-last-token = iter 033 behavior).
    pub fn set_prefetch_n(&mut self, v: Option<u32>) {
        let clamped = v
            .map(|n| n.max(TOPK as u32).min(N_ROUTED_EXPERTS as u32))
            .unwrap_or(TOPK as u32);
        self.prefetch_n = clamped;
        info!(
            prefetch_n = clamped,
            topk = TOPK,
            n_routed = N_ROUTED_EXPERTS,
            "set_prefetch_n"
        );
    }

    /// autolab iter 047 (C1): cumulative prefetch hit-rate counters
    /// since Runner load (or since the last `reset_prefetch_stats()`
    /// call). Returns `(hits, chances)` where `chances` counts the
    /// total number of actually-fired experts (across all layers, all
    /// tokens) that had a chance to be in the predictor (i.e. tokens
    /// after the very first per-prompt token where the predictor was
    /// still empty). `hits` counts how many of those were actually in
    /// the previous token's predicted top-N. Hit-rate = hits / chances.
    pub fn prefetch_stats(&self) -> (u64, u64) {
        (self.prefetch_hits, self.prefetch_chances)
    }

    /// autolab iter 054 (expert pinning): set the per-layer top-N hot
    /// experts to `mlock` after the warmup window. `None` (default) =
    /// pinning off. Composes with C1 prefetch — pinning makes the
    /// top-N immune to page eviction; the prefetcher hides the tail.
    ///
    /// Pins do NOT fire on `set_pin_top_n` — they fire after
    /// `pin_after_tokens` decoded tokens have accumulated hit data
    /// for `forward_shells` to choose a stable hot-set. To pin
    /// immediately on data you control (e.g. after a warmup prompt),
    /// call `pin_top_n_per_layer(n)` directly.
    ///
    /// Pre-flight: log a warn if `RLIMIT_MEMLOCK` is below the
    /// estimated need (`n × num_layers × ~21 MB`). On miner the
    /// memlock rlimit should be `unlimited` for root or a Xeon SKU
    /// preset — when running as non-root use `ulimit -l unlimited`
    /// or `prlimit --pid $PID --memlock=unlimited:unlimited` before
    /// process start.
    pub fn set_pin_top_n(&mut self, v: Option<u32>) {
        self.pin_top_n = v;
        if let Some(n) = v {
            // Per-expert size ≈ 21 MB on K2.6 int4 (six tensor slices).
            // We don't know the actual size until pin time but use this
            // for the rlimit pre-flight warning.
            let estimated_per_expert_bytes: u64 = 21 * 1024 * 1024;
            let num_layers = self.layers.len() as u64;
            let estimated_total_bytes = (n as u64)
                .saturating_mul(num_layers)
                .saturating_mul(estimated_per_expert_bytes);
            match SafetensorsExpertSource::rlimit_memlock_soft() {
                Some(soft) if soft >= estimated_total_bytes => {
                    info!(
                        pin_top_n = n,
                        num_layers,
                        estimated_total_mb = estimated_total_bytes / (1024 * 1024),
                        rlimit_memlock_soft_mb = if soft == u64::MAX {
                            u64::MAX
                        } else {
                            soft / (1024 * 1024)
                        },
                        "set_pin_top_n: rlimit_memlock OK"
                    );
                }
                Some(soft) => {
                    tracing::warn!(
                        pin_top_n = n,
                        num_layers,
                        estimated_total_mb = estimated_total_bytes / (1024 * 1024),
                        rlimit_memlock_soft_mb = soft / (1024 * 1024),
                        "set_pin_top_n: RLIMIT_MEMLOCK too low — pins will fail silently. \
                         Run `ulimit -l unlimited` (or `prlimit --pid <pid> --memlock=unlimited:unlimited`) \
                         and restart."
                    );
                }
                None => {
                    info!(
                        pin_top_n = n,
                        num_layers,
                        estimated_total_mb = estimated_total_bytes / (1024 * 1024),
                        "set_pin_top_n: rlimit check unavailable on this platform"
                    );
                }
            }
        } else {
            info!("set_pin_top_n: disabled");
        }
    }

    /// autolab iter 054: override the number of decoded tokens to wait
    /// for hit data before the first pin pass. Default 16. Setting to
    /// 0 will trigger pinning on the very first `forward_shells` call
    /// — useful for benchmarks where the prompt distribution is known
    /// to be representative.
    pub fn set_pin_after_tokens(&mut self, v: u32) {
        self.pin_after_tokens = v;
        info!(pin_after_tokens = v, "set_pin_after_tokens");
    }

    /// autolab iter 054: pin the top-N experts *per layer* by current
    /// hit count via `mlock` / `VirtualLock`. Idempotent at the per-
    /// expert level — pinning an already-pinned expert is a no-op.
    /// Returns `(experts_pinned_this_call, bytes_pinned_this_call)`.
    /// Bumps `self.pin_pass_done = true`.
    ///
    /// Strategy: for each layer position `i` in `self.layers`, sort
    /// `expert_hits[i]` by descending fire count and take the first N
    /// `(expert_id, _count)` pairs. Layers with fewer than N distinct
    /// experts pin all observed ones (e.g. if a layer only saw 12
    /// experts fire during warmup we still get those 12 locked).
    ///
    /// Threadsafe to call from any context; the pinning syscalls don't
    /// block on the inference path. However, the actual `mlock` calls
    /// **do** block during the page-in (kernel reads every page from
    /// disk before returning), so for a 47 GB top-set this can take
    /// tens of seconds on a cold cache. Call it during a warmup window
    /// the user is happy to wait through.
    pub fn pin_top_n_per_layer(&mut self, n: u32) -> (usize, u64) {
        let mut experts_pinned = 0usize;
        let mut bytes_pinned = 0u64;
        // Borrow split: hold an Arc clone of source and read expert_hits
        // by position — we don't need &mut self after we read pin_top_n.
        let source = self._safetensors_source.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let lid = layer.lid;
            let empty_map = HashMap::new();
            let hits = self.expert_hits.get(i).unwrap_or(&empty_map);
            for eid in select_top_n_by_hits(hits, n) {
                let pinned = source.pin_expert(lid, eid);
                if pinned > 0 {
                    experts_pinned += 1;
                    bytes_pinned += pinned as u64;
                }
            }
        }
        self.pin_pass_done = true;
        info!(
            n,
            experts_pinned,
            bytes_pinned,
            total_pinned_experts = source.pinned_expert_count(),
            total_pinned_bytes = source.pinned_bytes(),
            "pin_top_n_per_layer done"
        );
        (experts_pinned, bytes_pinned)
    }

    /// autolab iter 054: convenience wrapper — unpin everything this
    /// runner has pinned (delegates to the source's `unpin_all_experts`)
    /// and clears `pin_pass_done` so a subsequent pin pass can re-arm.
    pub fn unpin_all_experts(&mut self) -> (usize, u64) {
        let (n, released) = self._safetensors_source.unpin_all_experts();
        self.pin_pass_done = false;
        info!(
            experts_unpinned = n,
            bytes_released = released,
            "unpin_all_experts"
        );
        (n, released)
    }

    /// autolab iter 056 (cache-aware dispatch): enable / disable the
    /// hit-frequency dispatch reorder inside `forward_shells`. When
    /// `true`, per-layer top-K experts are dispatched in descending
    /// `expert_hits[i]` order so the hottest experts run first and
    /// stay L3-resident across the next layer's dispatch. Default
    /// `false` keeps the iter 047 router-score order.
    ///
    /// Bit-identical to router-score order on the output side: router
    /// weights are still summed in the original index order. Only the
    /// `dispatch_expert` *call order* changes — every other side effect
    /// (`prefetch_hits` counters, `expert_hits` bumps, weighted-sum
    /// accumulation order) is byte-identical to the default. Verified
    /// by `cache_aware_dispatch_bit_identical_to_router_score_order`.
    pub fn set_cache_aware_dispatch(&mut self, v: bool) {
        self.cache_aware_dispatch = v;
        info!(cache_aware_dispatch = v, "set_cache_aware_dispatch");
    }

    /// autolab iter 056: read-only accessor for tests + benches.
    pub fn cache_aware_dispatch_enabled(&self) -> bool {
        self.cache_aware_dispatch
    }

    /// autolab iter 057 (async kernel scheduling — speculative prefetch):
    /// set the per-layer top-N hit-frequent experts to speculatively
    /// `madvise(WILLNEED)` for layer `i + 1` immediately before layer
    /// `i`'s expert dispatch begins. `None` (default) = off; `Some(n)`
    /// with `n > 0` enables the scheduler. Silently clamped to
    /// `[1, N_ROUTED_EXPERTS]` to bound runaway prefetch storms.
    ///
    /// Behaviorally a no-op until `expert_hits[i + 1]` has data — i.e.
    /// the first prefill token across the whole layer stack is a
    /// pure no-op (every layer's hit map is empty), so by construction
    /// iter 057 is byte-identical to iter 056 on the very first
    /// per-prompt forward_shells call. Steady-state (after the iter
    /// 054 warmup window worth of tokens) the prefetcher submits up
    /// to `n × (n_layers - 1)` madvise requests per token in addition
    /// to the iter 047 `prefetch_n × n_layers` submitted at the top
    /// of `forward_shells`. Together they pre-page both this token's
    /// next layer (iter 057, tight overlap) and the next token's
    /// entire layer stack (iter 047, loose overlap).
    ///
    /// The prefetcher is a non-blocking kernel hint; misses are wasted
    /// bandwidth but never affect output. Pass `None` to disable.
    pub fn set_speculative_prefetch_n(&mut self, v: Option<u32>) {
        let clamped = v.map(|n| n.min(N_ROUTED_EXPERTS as u32).max(1));
        self.speculative_prefetch_n = clamped;
        info!(
            speculative_prefetch_n = ?clamped,
            n_routed = N_ROUTED_EXPERTS,
            "set_speculative_prefetch_n"
        );
    }

    /// autolab iter 057: read-only accessor for tests + benches.
    pub fn speculative_prefetch_n(&self) -> Option<u32> {
        self.speculative_prefetch_n
    }

    /// autolab iter 057: cumulative count of speculative prefetch
    /// submissions since `Runner::load`. Used by the instrumentation
    /// log emitted from `forward_shells` and by unit tests that drive
    /// the scheduling logic synthetically.
    pub fn speculative_prefetch_submitted(&self) -> u64 {
        self.speculative_prefetch_submitted
    }

    /// autolab iter 065 (prefill-hint static schedule): set the merge
    /// weight applied to per-prompt prefill observations when folding
    /// them into `expert_hits` at the end of prefill. `0.0` (default)
    /// **disables the hint path entirely** — prefill `forward_shells`
    /// calls bump `expert_hits` directly (iter 054 behavior), and the
    /// per-prompt observation buffer stays empty.
    ///
    /// With `w > 0.0`, prefill dispatches stop bumping `expert_hits`
    /// and instead record into `prefill_expert_observations`. At
    /// `exit_prefill_and_merge_hints` the observations are folded back
    /// in as `expert_hits[i][eid] += round(w * obs_count)`. The
    /// downstream consumers — iter 054 (`pin_top_n_per_layer`), iter
    /// 056 (`cache_aware_dispatch_order`), iter 057
    /// (`speculative_prefetch_expert_ids`) — all read from the same
    /// `expert_hits` map, so a non-zero weight seeds them with the
    /// prompt's actually-fired routing distribution before decode
    /// iteration #1.
    ///
    /// Negative weights are silently clamped to 0.0 (no merge). NaN /
    /// infinite weights are silently clamped to 0.0 as well; the merge
    /// uses `round() as u64` which would otherwise saturate or panic.
    pub fn set_prefill_hint_weight(&mut self, w: f32) {
        let clean = if w.is_finite() && w >= 0.0 { w } else { 0.0 };
        self.prefill_hint_weight = clean;
        info!(
            prefill_hint_weight = clean,
            enabled = clean > 0.0,
            "set_prefill_hint_weight"
        );
    }

    /// autolab iter 065: read-only accessor for tests + benches.
    pub fn prefill_hint_weight(&self) -> f32 {
        self.prefill_hint_weight
    }

    /// autolab iter 065: mark the runner as entering the prefill phase
    /// for the next prompt. When the hint is enabled
    /// (`prefill_hint_weight > 0.0`) this re-routes
    /// `forward_shells`'s per-expert hit bumps from `expert_hits` into
    /// `prefill_expert_observations` so the per-prompt prefill firing
    /// distribution can be merged back with the configured weight at
    /// `exit_prefill_and_merge_hints`.
    ///
    /// When the hint is disabled (`prefill_hint_weight == 0.0`) this
    /// is a cheap no-op: the dispatch loop checks the weight first and
    /// keeps bumping `expert_hits` directly, preserving iter 054
    /// behavior bit-for-bit.
    ///
    /// Idempotent — repeated calls without an intervening
    /// `exit_prefill_and_merge_hints` are harmless. Called from
    /// `Runner::generate` (single-stage) and from the engine's
    /// distributed driver (`drive_generation_first`) at the top of
    /// the prompt loop.
    pub fn enter_prefill(&mut self) {
        self.in_prefill = true;
        info!(
            prefill_hint_weight = self.prefill_hint_weight,
            "enter_prefill"
        );
    }

    /// autolab iter 065: mark prefill complete and fold the per-prompt
    /// `prefill_expert_observations` into `expert_hits` using
    /// `prefill_hint_weight`. The merge formula is
    /// `expert_hits[i][eid] += round(w * obs_count)`, with saturating
    /// `u64` addition so over-long prompts can't wrap. Observations are
    /// cleared after the merge so the next prompt's prefill starts with
    /// a fresh window.
    ///
    /// When the hint is disabled (`prefill_hint_weight == 0.0`) this
    /// is a cheap no-op: nothing was recorded in
    /// `prefill_expert_observations`, and `expert_hits` already reflects
    /// the prefill firings (bumped directly during the prefill calls).
    ///
    /// Returns the total number of `(layer, expert)` entries merged so
    /// the engine can log + tests can verify the merge fired.
    pub fn exit_prefill_and_merge_hints(&mut self) -> usize {
        // Always clear the in-prefill gate so decode-phase hit bumps go
        // to `expert_hits` even if the caller forgot to enable the hint.
        self.in_prefill = false;
        if self.prefill_hint_weight <= 0.0 {
            // Hint disabled: prefill_expert_observations is empty by
            // construction (the dispatch loop never wrote to it).
            return 0;
        }
        let w = self.prefill_hint_weight;
        let merged = merge_prefill_observations_into_hits(
            &mut self.expert_hits,
            &self.prefill_expert_observations,
            w,
        );
        for obs in self.prefill_expert_observations.iter_mut() {
            obs.clear();
        }
        info!(
            prefill_hint_weight = w,
            entries_merged = merged,
            "exit_prefill_and_merge_hints"
        );
        merged
    }

    /// autolab iter 065: read-only snapshot of the per-prompt prefill
    /// observation map. Used by tests to verify the dispatch loop wrote
    /// the right entries during prefill before `exit_prefill_and_merge_hints`
    /// flushes them.
    pub fn prefill_expert_observations_snapshot(&self) -> Vec<HashMap<u32, u64>> {
        self.prefill_expert_observations.clone()
    }

    /// autolab iter 054: snapshot of (pinned_experts_count, pinned_bytes).
    /// Used by the instrumentation log + tests.
    pub fn pinned_stats(&self) -> (usize, u64) {
        (
            self._safetensors_source.pinned_expert_count(),
            self._safetensors_source.pinned_bytes(),
        )
    }

    /// autolab iter 054: clone of the per-layer hit map. Used by tests
    /// and external callers that want to inspect the heavy-tail shape
    /// without taking `&mut self`.
    pub fn expert_hits_snapshot(&self) -> Vec<HashMap<u32, u64>> {
        self.expert_hits.clone()
    }

    /// Run one expert. Returns the f32 output vector (length = hidden_size).
    /// `attn_row` is the f32 hidden state for one token (length =
    /// hidden_size). Backend is chosen by the manifest's experts_format.
    fn dispatch_expert(
        &mut self,
        lid: u32,
        eid: u32,
        attn_row: &[f32],
    ) -> Result<Vec<f32>, RunnerError> {
        let hidden = self.manifest.hidden_size as usize;
        // iter 069: record the hit into the flat (lid, eid) HashMap
        // that drives the hot-buffer warmup gate. This is independent
        // from `self.expert_hits` (the iter 054 per-position map used
        // by pin / cache-aware dispatch / spec-prefetch / prefill-hint
        // — bumped inside forward_shells from the dispatch loop).
        self.hot_expert_hits.record(lid, eid);

        // Lazy hot-buffer build: once we've crossed the warmup
        // threshold we copy the per-layer top-N into contiguous owned
        // memory. After that, dispatches to a hot eid go through the
        // packed buffer instead of the mmap.
        if self.hot_expert_buffer_n > 0
            && !self.hot_buffers_built
            && self.hot_expert_hits.total >= self.hot_buffer_warmup_dispatches
        {
            self.build_hot_buffers_now()?;
        }

        // Hot path: only valid for the int4 kernel backends. OV IR
        // path runs through OpenVINO which doesn't share the
        // gate/up/down byte layout, so the hot buffer is a no-op for
        // it. Skip the lookup if we know it'll miss.
        let use_hot_path = matches!(
            self.experts,
            ExpertCache::Int4Bin(_) | ExpertCache::SafetensorsBin(_)
        );
        if use_hot_path && self.hot_buffers_built {
            if let Some(lhb) = self.hot_buffers.get(&lid) {
                if let Some(view) = lhb.slice(eid) {
                    let x_bf16: Vec<bf16> = attn_row.iter().map(|v| bf16::from_f32(*v)).collect();
                    let mut out_bf16 = vec![bf16::ZERO; hidden];
                    int4_expert_forward(
                        &x_bf16,
                        view.gate_packed,
                        view.gate_scale,
                        view.up_packed,
                        view.up_scale,
                        view.down_packed,
                        view.down_scale,
                        &mut out_bf16,
                    );
                    return Ok(out_bf16.iter().map(|b| b.to_f32()).collect());
                }
            }
        }

        match &mut self.experts {
            ExpertCache::OvIr(c) => {
                let attn_bf16 = f32_to_bf16_bytes(attn_row);
                let rt = c.get(lid, eid)?;
                rt.set_input("x", DType::Bf16, &[1, 1, hidden], &attn_bf16)?;
                rt.infer()?;
                let (e_dt, _, e_bytes) = rt.output(0)?;
                Ok(match e_dt {
                    DType::F32 => read_f32(&e_bytes),
                    DType::Bf16 => bf16_bytes_to_f32(&e_bytes),
                    DType::F16 => f16_bytes_to_f32(&e_bytes),
                    _ => {
                        return Err(RunnerError::Internal(format!(
                            "expert dtype {:?} not f32-castable",
                            e_dt
                        )))
                    }
                })
            }
            ExpertCache::Int4Bin(c) => {
                let w = c.get(lid, eid)?;
                let x_bf16: Vec<bf16> = attn_row.iter().map(|v| bf16::from_f32(*v)).collect();
                let mut out_bf16 = vec![bf16::ZERO; hidden];
                int4_expert_forward(
                    &x_bf16,
                    w.gate_packed_bytes(),
                    w.gate_scale_bits(),
                    w.up_packed_bytes(),
                    w.up_scale_bits(),
                    w.down_packed_bytes(),
                    w.down_scale_bits(),
                    &mut out_bf16,
                );
                Ok(out_bf16.iter().map(|b| b.to_f32()).collect())
            }
            ExpertCache::SafetensorsBin(c) => {
                let key = (lid, eid);
                if !c.map.contains_key(&key) {
                    let e = c.source.expert(lid, eid).map_err(|e| {
                        RunnerError::Internal(format!("safetensors expert {lid}/{eid}: {e}"))
                    })?;
                    c.map.insert(key, e);
                }
                let w = c.map.get(&key).unwrap();
                let x_bf16: Vec<bf16> = attn_row.iter().map(|v| bf16::from_f32(*v)).collect();
                let mut out_bf16 = vec![bf16::ZERO; hidden];
                int4_expert_forward(
                    &x_bf16,
                    w.gate_packed,
                    w.gate_scale,
                    w.up_packed,
                    w.up_scale,
                    w.down_packed,
                    w.down_scale,
                    &mut out_bf16,
                );
                Ok(out_bf16.iter().map(|b| b.to_f32()).collect())
            }
        }
    }

    /// Reset all per-layer KV caches. Call between independent generations.
    pub fn reset_kv(&mut self) {
        for l in &mut self.layers {
            // Keep the past_k / past_v allocations — slots
            // 0..past_seq_len are simply abandoned. Resetting them to
            // zero is not required because forward_shells only reads
            // the populated prefix `0..past_seq_len`.
            l.past_seq_len = 0;
        }
        if let Some(l0) = self.layer0.as_mut() {
            l0.past_seq_len = 0;
        }
        // autolab iter 029 (C1): a fresh prompt has zero correlation
        // with the previous prompt's expert routing — wipe the
        // same-as-last-token predictor so we don't waste prefetch
        // bandwidth on irrelevant experts.
        for hist in self.last_routing_ids.iter_mut() {
            hist.clear();
        }
        // iter 047: hit-rate counters are cumulative across this
        // generation only — reset alongside the predictor so per-prompt
        // stats aren't confused with the previous prompt's tail.
        self.prefetch_hits = 0;
        self.prefetch_chances = 0;
        // iter 054 (expert pinning): zero the per-reset token counter
        // so the next prompt's `pin_after_tokens` window starts fresh.
        // The pin state itself (pin_pass_done + the locked pages) is
        // NOT cleared — pinning is a long-lived process-wide property
        // we want to survive prompt boundaries. The per-layer
        // `expert_hits` map is also NOT cleared: hot experts identified
        // on prompt 1 stay hot across prompts (the heavy-tail is a
        // model-level property, not a prompt-level one), and keeping
        // the data means re-pin passes (after `unpin_all_experts`) get
        // increasingly accurate top-Ns rather than starting from zero.
        self.decoded_tokens_since_reset = 0;
        // iter 065 (prefill-hint): per-prompt observations are
        // **single-prompt-scoped** — clear them on every reset so the
        // next prompt's prefill window doesn't double-count the prior
        // prompt's prefill firings. Also clear the in-prefill gate so
        // a fresh prompt enters in a known state regardless of whether
        // the previous prompt called `exit_prefill_and_merge_hints`.
        for obs in self.prefill_expert_observations.iter_mut() {
            obs.clear();
        }
        self.in_prefill = false;
    }

    /// Run one forward pass:
    /// - `full_ids` is the FULL prefix-so-far (1D i64), used by stateless
    ///   layer 0.
    /// - The shells consume just the trailing `tail_len` tokens, with the
    ///   per-layer KV state representing the prior
    ///   `past_seq_len = full_ids.len - tail_len` tokens.
    ///
    /// Returns the FP32 logits for the last position (`vocab_size` elements).
    fn step(&mut self, full_ids: &[i64], tail_len: usize) -> Result<Vec<f32>, RunnerError> {
        if tail_len == 0 || tail_len > full_ids.len() {
            return Err(RunnerError::Internal(format!(
                "invalid tail_len {}, full_ids.len={}",
                tail_len,
                full_ids.len()
            )));
        }
        let past_seq_len = full_ids.len() - tail_len;
        let hidden = self.manifest.hidden_size as usize;

        // 1) Layer 0 (now stateful) → tail of hidden state.
        // The engine drives prefill + decode token-by-token, so tail_len
        // is always 1 in practice — but if a future caller passes a
        // larger tail, we just advance layer 0 one token at a time.
        let mut h_f32 = vec![0.0f32; tail_len * hidden];
        for k in 0..tail_len {
            let id = full_ids[past_seq_len + k];
            let row = self.forward_layer0_step(id)?;
            h_f32[k * hidden..(k + 1) * hidden].copy_from_slice(&row);
        }
        let h_shape = vec![1usize, tail_len, hidden];

        // 2) Run all my shells.
        let h_f32 = self.forward_shells(&h_f32, &h_shape, past_seq_len)?;

        // 3) Head on the LAST token.
        self.forward_head_last(&h_f32, tail_len)
    }

    /// Run one stateful step of layer 0 for a single token, updating
    /// the layer's KV cache and returning the resulting hidden row.
    /// First-stage only.
    ///
    /// The shape contract matches `forward_shells`: returns `[HIDDEN]`,
    /// already passed through attention + dense MLP + residual.
    pub fn forward_layer0_step(&mut self, token_id: i64) -> Result<Vec<f32>, RunnerError> {
        let _t0 = Instant::now();
        let l0 = self.layer0.as_mut().ok_or_else(|| {
            RunnerError::Internal("forward_layer0_step on non-first stage".into())
        })?;

        let x_f32 = embed_token_bf16(l0.embed_tokens_bf16, token_id);

        if l0.past_seq_len + 1 > l0.kv_capacity {
            grow_layer0_kv_capacity(l0)?;
        }
        let capacity = l0.kv_capacity;
        let past_seq_len = l0.past_seq_len;

        let outs = layer0_forward_decode_int4_with_capacity(
            &l0.int4_layer0,
            &x_f32,
            &l0.past_k,
            &l0.past_v,
            past_seq_len,
            capacity,
        );

        write_present_kv(
            &mut l0.past_k,
            &outs.present_k,
            past_seq_len,
            capacity,
            NUM_HEADS,
            QK_HEAD_DIM,
        );
        write_present_kv(
            &mut l0.past_v,
            &outs.present_v,
            past_seq_len,
            capacity,
            NUM_HEADS,
            V_HEAD_DIM,
        );
        l0.past_seq_len = past_seq_len + 1;

        // autolab/k26-perf q1 instrumentation: per-token layer-0 timing.
        info!(
            stage = "layer0",
            duration_us = _t0.elapsed().as_micros() as u64,
            past_seq_len,
            "stage_timing"
        );
        Ok(outs.hidden_out)
    }

    /// Forward one token through the shells this rank owns.
    /// `h_f32` is the input hidden state shaped `[1, 1, hidden]`
    /// (row-major). Returns the same shape after this rank's MoE
    /// layers, with per-layer KV cache updated.
    ///
    /// The int4 shell forward only supports seq=1 (decode mode). The
    /// engine already drives prefill token-by-token, so this is the
    /// only shape that ever shows up here. The mask is implicit: a
    /// single token attends to all past + itself, no -inf positions
    /// needed.
    pub fn forward_shells(
        &mut self,
        h_in: &[f32],
        h_shape: &[usize],
        past_seq_len: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let _t0 = Instant::now();
        let mut shell_attn_total_us: u64 = 0;
        let mut experts_total_us: u64 = 0;
        let mut combine_total_us: u64 = 0;
        let hidden = self.manifest.hidden_size as usize;
        let manifest_top_k = self.manifest.top_k as usize;
        // autolab campaign 004 (A3): if an override is set, only dispatch
        // the first k' of the routed top-K experts per token. The shell's
        // router still returns the full manifest top_k.
        let effective_top_k = self
            .top_k_override
            .map(|v| (v as usize).min(manifest_top_k))
            .unwrap_or(manifest_top_k);
        let top_k = manifest_top_k; // alias for the router contract check below
        if h_shape.len() != 3 || h_shape[0] != 1 || h_shape[1] != 1 || h_shape[2] != hidden {
            return Err(RunnerError::Internal(format!(
                "forward_shells: int4 shells require shape [1, 1, {hidden}], got {h_shape:?}"
            )));
        }
        let mut h_f32 = h_in.to_vec();

        let n_layers = self.layers.len();

        // autolab iter 029 (C1): kick off madvise(WILLNEED) for every
        // predicted next-token expert before we run any layer's attn.
        // Predictor is "same as last token" — i.e. the IDs we stored in
        // `last_routing_ids[i]` after the previous call. This races the
        // OS readahead against this token's compute, so by the time we
        // hit each layer's dispatch_expert the pages are (hopefully)
        // already warm. Skipped for the very first token after
        // `reset_kv` when last_routing_ids[i] is still empty.
        let mut prefetch_submitted: u64 = 0;
        if let Some(pf) = self.prefetcher.as_ref() {
            for (i, hist) in self.last_routing_ids.iter().enumerate() {
                if hist.is_empty() {
                    continue;
                }
                let lid = self.layers[i].lid;
                for &eid in hist.iter() {
                    pf.try_submit(lid, eid);
                    prefetch_submitted += 1;
                }
            }
        }

        for i in 0..n_layers {
            let lid = self.layers[i].lid;
            if self.layers[i].past_seq_len != past_seq_len {
                return Err(RunnerError::Internal(format!(
                    "L{lid}: past_seq_len mismatch (caller {past_seq_len} vs layer {})",
                    self.layers[i].past_seq_len
                )));
            }

            // Ensure the pre-allocated KV buffers have room for the new
            // slot. Geometric grow when we hit capacity — total
            // alloc/copy traffic across a full generation is O(N), not
            // O(N²) like the old `append_kv_inplace`.
            if past_seq_len + 1 > self.layers[i].kv_capacity {
                grow_kv_capacity(&mut self.layers[i])?;
            }
            let capacity = self.layers[i].kv_capacity;

            // Run Rust shell forward — same int4 kernel rainier's eval
            // used via the cdylib, just called directly since we're in
            // the same Cargo workspace. The `_predict_n` variant lets
            // us pass a pre-allocated [H, capacity, D] buffer with
            // only the first `past_seq_len` slots populated, AND emits
            // the top-N expert ids by router score for next-token
            // prefetch prediction (autolab iter 047 C1 better
            // predictor; N == TOPK is back-compat with iter 033).
            let shell_t0 = Instant::now();
            let outs = shell_forward_decode_int4_predict_n(
                &self.layers[i].int4_shell,
                &h_f32,
                &self.layers[i].past_k,
                &self.layers[i].past_v,
                past_seq_len,
                capacity,
                self.prefetch_n as usize,
            );
            shell_attn_total_us += shell_t0.elapsed().as_micros() as u64;

            // Write present_k / present_v into the existing capacity
            // buffer at slot `past_seq_len` for each head. No alloc.
            write_present_kv(
                &mut self.layers[i].past_k,
                &outs.present_k,
                past_seq_len,
                capacity,
                NUM_HEADS,
                QK_HEAD_DIM,
            );
            write_present_kv(
                &mut self.layers[i].past_v,
                &outs.present_v,
                past_seq_len,
                capacity,
                NUM_HEADS,
                V_HEAD_DIM,
            );
            self.layers[i].past_seq_len = past_seq_len + 1;

            // Expert dispatch — top-k weighted sum over the
            // routing_ids/weights the shell already chose for us.
            if outs.routing_ids.len() != top_k || outs.routing_weights.len() != top_k {
                return Err(RunnerError::Internal(format!(
                    "L{lid} routing shape unexpected: ids={} weights={} (top_k={})",
                    outs.routing_ids.len(),
                    outs.routing_weights.len(),
                    top_k
                )));
            }

            // autolab iter 057 (async kernel scheduling — speculative
            // prefetch of layer i+1's hit-frequent experts). We're about
            // to spend ~150 ms inside the expert dispatch below; that
            // window is wasted on the prefetcher side if we don't
            // schedule the *next* layer's likely-fired weights now.
            // Delegates the selection to `speculative_prefetch_targets`
            // so the per-call logic is exercised by the runner tests
            // without needing a loaded K2.6 model.
            //
            // Wrong guesses cost OS readahead bandwidth but cannot
            // affect model output — the dispatch path below still pulls
            // weights via `dispatch_expert` from real routing decisions.
            if let (Some(n_spec), Some(pf)) =
                (self.speculative_prefetch_n, self.prefetcher.as_ref())
            {
                let next_i = i + 1;
                if next_i < n_layers {
                    let next_lid = self.layers[next_i].lid;
                    let eids = speculative_prefetch_expert_ids(&self.expert_hits[next_i], n_spec);
                    for eid in eids {
                        pf.try_submit(next_lid, eid);
                        self.speculative_prefetch_submitted =
                            self.speculative_prefetch_submitted.saturating_add(1);
                    }
                }
            }

            let experts_t0 = Instant::now();
            let mut moe = vec![0.0f32; hidden];
            // autolab campaign 007 (A2): apply routing-weight threshold.
            // We still iterate over `effective_top_k` to honor the A3 cap,
            // but skip experts below the threshold within that range.
            let threshold = self.routing_threshold.unwrap_or(0.0);
            // autolab iter 047 (C1 hit-rate): each actually-fired expert
            // is a "chance" — was it in the previous token's prediction?
            // Skip the very first per-prompt token (predictor is empty,
            // counting those would dilute the rate spuriously).
            let predictor_was_warm = !self.last_routing_ids[i].is_empty();

            // autolab iter 056 (cache-aware dispatch): three-phase
            // restructure that gates the optional dispatch reorder
            // while preserving byte-identical output and all observable
            // side effects.
            //
            // Phase 1 (original-order bookkeeping): walk
            // `k = 0..effective_top_k` in router-score order to
            // (a) skip threshold misses, (b) count prefetch hits/chances,
            // (c) bump expert_hits. This is byte-identical to the
            // pre-056 loop's side-effect ordering — important because
            // autolab campaigns A/B these counters across iterations.
            // Mark "active" indices for the dispatch pass.
            let mut active_ks: Vec<usize> = Vec::with_capacity(effective_top_k);
            for k in 0..effective_top_k {
                let w = outs.routing_weights[k];
                if w < threshold {
                    continue;
                }
                let eid = outs.routing_ids[k] as u32;
                if predictor_was_warm {
                    self.prefetch_chances += 1;
                    if self.last_routing_ids[i].contains(&eid) {
                        self.prefetch_hits += 1;
                    }
                }
                // autolab iter 054: bump per-(layer-position, expert)
                // fire count for the pin-top-N heuristic. Persists
                // across reset_kv so steady-state pin sets reflect the
                // full workload, not a single prompt. Negligible cost:
                // one HashMap entry update per dispatched expert
                // (~K=8 per layer per token).
                //
                // autolab iter 065 (prefill-hint static schedule): when
                // the hint is enabled (`prefill_hint_weight > 0.0`) and
                // we're inside the prefill phase, the bump is diverted
                // to `prefill_expert_observations` so it can be folded
                // back into `expert_hits` with the configured weight
                // at `exit_prefill_and_merge_hints`. When the hint is
                // disabled (weight 0.0) or we're in decode, bumps go
                // straight to `expert_hits` exactly as iter 054 does.
                let route_to_observations = self.in_prefill && self.prefill_hint_weight > 0.0;
                if route_to_observations {
                    *self
                        .prefill_expert_observations
                        .get_mut(i)
                        .expect("layer-indexed prefill observations map")
                        .entry(eid)
                        .or_insert(0) += 1;
                } else {
                    *self
                        .expert_hits
                        .get_mut(i)
                        .expect("layer-indexed hit map")
                        .entry(eid)
                        .or_insert(0) += 1;
                }
                active_ks.push(k);
            }

            // Phase 2 (dispatch in cache-aware or router-score order):
            // compute each active expert and stash its output indexed
            // by the original `k`. With `cache_aware_dispatch == false`
            // this is byte-identical to the pre-056 path: the iteration
            // happens in router-score order and the stash is trivial.
            // With it enabled the iteration happens in descending
            // `expert_hits[i]` order (broken on the original k) so the
            // hottest experts run first and stay L3-warm across the
            // next layer's hot prefix.
            //
            // NOTE: `expert_hits` was just updated in Phase 1 for this
            // token. That means a freshly-fired expert sees its own
            // bump reflected in the reorder — desired, because a
            // single L3-warm expert that just fired is the cheapest
            // to call back-to-back from the next layer's dispatch.
            let dispatch_seq: Vec<usize> = if self.cache_aware_dispatch {
                // Build a slice of just the active k's so the helper
                // sees the same `routing_ids` view the dispatch path
                // sees. We pass the original `routing_ids` slice but
                // restrict the returned permutation to entries that
                // are in `active_ks` (set difference is rare — most
                // tokens hit threshold for every k). Two-pass:
                // permutation over the full top-K, then filter.
                let full = cache_aware_dispatch_order(&outs.routing_ids, &self.expert_hits[i]);
                let active_set: std::collections::HashSet<usize> =
                    active_ks.iter().copied().collect();
                full.into_iter()
                    .filter(|k| active_set.contains(k))
                    .collect()
            } else {
                active_ks.clone()
            };
            let mut expert_outs: Vec<Option<Vec<f32>>> =
                (0..effective_top_k).map(|_| None).collect();
            for &k in dispatch_seq.iter() {
                let eid = outs.routing_ids[k] as u32;
                let y_f32 = self.dispatch_expert(lid, eid, &outs.attn_out_post_norm)?;
                expert_outs[k] = Some(y_f32);
            }

            // Phase 3 (original-order weighted sum): accumulate into
            // `moe` in ascending `k` so the floating-point rounding
            // chain is identical regardless of `cache_aware_dispatch`.
            // Skipped slots (threshold misses) stay `None` and are
            // ignored, byte-identical to the pre-056 `continue` path.
            for (k, slot) in expert_outs.iter().enumerate() {
                if let Some(y_f32) = slot {
                    let w = outs.routing_weights[k];
                    for j in 0..hidden {
                        moe[j] += w * y_f32[j];
                    }
                }
            }
            // autolab iter 047: stash the top-N expert IDs for the next
            // token's prefetch. With prefetch_n == TOPK (default) this
            // is byte-identical to iter 033's behavior: the same TOPK
            // actually-fired IDs go in. With prefetch_n > TOPK we get
            // the top-N by router score — the actually-fired TOPK are
            // guaranteed in there plus `prefetch_n - TOPK` insurance
            // experts. The dispatch path above doesn't see these
            // (still only iterates `effective_top_k` of routing_ids).
            //
            // Tracking it costs ~N*u32 per layer per token (negligible
            // vs the expert GEMM bandwidth) and makes it cheaper to
            // toggle the prefetcher mid-run.
            self.last_routing_ids[i] = outs
                .predicted_top_n_ids
                .iter()
                .map(|&id| id as u32)
                .collect();
            experts_total_us += experts_t0.elapsed().as_micros() as u64;

            // Combine: h_next = residual + shared + moe (single token).
            let combine_t0 = Instant::now();
            for j in 0..hidden {
                h_f32[j] = outs.attn_residual[j] + outs.shared_expert_out[j] + moe[j];
            }
            combine_total_us += combine_t0.elapsed().as_micros() as u64;
        }

        // autolab iter 054 (expert pinning): bump the per-reset token
        // counter, and if it's crossed the pin threshold AND the user
        // requested pinning AND we haven't pinned yet — fire the pin
        // pass now. We do this AFTER the layer loop so the very same
        // token that triggers pinning has its hit data folded in
        // before we choose the top-N. The pin call may take seconds
        // on a cold cache (kernel reads every page to RAM); the
        // caller's per-token tok/s will dip on this single token then
        // recover with the hot-set locked.
        self.decoded_tokens_since_reset = self.decoded_tokens_since_reset.saturating_add(1);
        if let Some(n) = self.pin_top_n {
            if !self.pin_pass_done && self.decoded_tokens_since_reset >= self.pin_after_tokens {
                info!(
                    n,
                    decoded_tokens_since_reset = self.decoded_tokens_since_reset,
                    pin_after_tokens = self.pin_after_tokens,
                    "expert pinning: firing pin_top_n_per_layer (warmup window reached)"
                );
                let (pinned, bytes) = self.pin_top_n_per_layer(n);
                info!(
                    pinned,
                    bytes_mb = bytes / (1024 * 1024),
                    "expert pinning: pin pass complete"
                );
            }
        }

        // autolab/k26-perf q1 instrumentation: per-token shells breakdown.
        // iter 029 (C1): also log prefetch counters so we can see how the
        // submit/drop ratio evolves across a generation. Counters are
        // cumulative-since-Runner-load, so the deltas across consecutive
        // tokens tell us submits-per-token and drops-per-token.
        // iter 047 (better predictor): log hit-rate (predict-only-correct
        // = good prefetch; predict-mostly-wrong = wasted bandwidth) and
        // the active prefetch_n so A/B campaigns can correlate.
        // iter 054 (pinning): log pinned-experts count + bytes for A/B
        // campaigns to correlate hit-rate with pinning coverage.
        let (pf_submits, pf_drops, pf_processed) = self
            .prefetcher
            .as_ref()
            .map(|p| p.snapshot())
            .unwrap_or((0, 0, 0));
        let (pinned_count, pinned_bytes) = self.pinned_stats();
        info!(
            stage = "shells",
            n_layers,
            top_k,
            effective_top_k,
            prefetch_n = self.prefetch_n,
            shell_attn_us = shell_attn_total_us,
            experts_us = experts_total_us,
            combine_us = combine_total_us,
            prefetch_submitted_this_call = prefetch_submitted,
            prefetch_total_submits = pf_submits,
            prefetch_total_drops = pf_drops,
            prefetch_total_processed = pf_processed,
            prefetch_hits = self.prefetch_hits,
            prefetch_chances = self.prefetch_chances,
            pinned_experts = pinned_count,
            pinned_bytes_mb = pinned_bytes / (1024 * 1024),
            // iter 057 (speculative prefetch): cumulative count of
            // madvise requests submitted from inside the layer loop
            // for layer i+1's hit-frequent experts. Per-token delta
            // ≈ speculative_prefetch_n × (n_layers - 1) when enabled
            // and expert_hits has warmed up; 0 when disabled or on
            // the very first per-prompt token before any dispatch.
            speculative_prefetch_n = ?self.speculative_prefetch_n,
            speculative_prefetch_submitted = self.speculative_prefetch_submitted,
            total_us = _t0.elapsed().as_micros() as u64,
            "stage_timing"
        );
        Ok(h_f32)
    }

    /// Multi-token shell forward: same semantics as a `seq`-iteration
    /// loop over [`Self::forward_shells`], but routes the int4 GEMM
    /// projections through the iter 041/042/046 multi-token kernels
    /// (`shell_forward_decode_int4_multi_with_capacity`). At seq=1 this
    /// is a no-op wrapper over the existing seq=1 path; at seq>=2 the
    /// projections amortize the weight load across tokens (1.4-4.75x
    /// per projection per iter 042 microbench; +40% on oproj/shared_down
    /// at seq>=4 per iter 046).
    ///
    /// **Caller contract.**
    /// - `h_in` is `[1, seq, hidden]` flat, row-major: token `t`'s row
    ///   lives at `h_in[t * hidden .. (t + 1) * hidden]`.
    /// - `past_seq_len` is the populated KV length on entry. After this
    ///   call returns, each shell's `past_seq_len` advances by `seq`.
    /// - Returns `[1, seq, hidden]` flat — the post-MoE hidden state for
    ///   each input token. Callers that only need the last token's
    ///   logits slice `out[(seq-1)*hidden .. seq*hidden]`.
    ///
    /// **Why a separate API.** The seq=1 hot path
    /// ([`Self::forward_shells`]) is unchanged — every K2.6 inference
    /// today runs through it. This is the seam that the
    /// chunked-prefill (iter 040) and spec-decode verify (iter 036/039)
    /// driver loops plug into to batch multiple tokens per shell call,
    /// turning N sequential seq=1 GEMVs into one seq=N GEMM and
    /// recovering the iter 042/046 SIMD wins at the engine level.
    ///
    /// **Bit-identity.** The underlying multi-token kernels are
    /// bit-identical per-cell to the scalar seq=1 path (proved by the
    /// `multi_batched_matches_scalar*` tests in
    /// `tahoma-int4-gemm/src/shell_int4.rs`). Combined with the
    /// per-token expert dispatch — which is identical between this
    /// function and `forward_shells` since both apply the same
    /// `routing_ids` × `routing_weights` to the same per-token attn_out
    /// — the engine-level path produces byte-identical outputs to N
    /// sequential `forward_shells` calls.
    pub fn forward_shells_multi(
        &mut self,
        h_in: &[f32],
        h_shape: &[usize],
        past_seq_len: usize,
        seq: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let hidden = self.manifest.hidden_size as usize;
        let top_k = self.manifest.top_k as usize;
        if seq == 0 {
            return Err(RunnerError::Internal(
                "forward_shells_multi: seq must be >= 1".into(),
            ));
        }
        if h_shape.len() != 3 || h_shape[0] != 1 || h_shape[1] != seq || h_shape[2] != hidden {
            return Err(RunnerError::Internal(format!(
                "forward_shells_multi: int4 shells require shape [1, {seq}, {hidden}], got {h_shape:?}"
            )));
        }
        if h_in.len() != seq * hidden {
            return Err(RunnerError::Internal(format!(
                "forward_shells_multi: h_in.len={} != seq*hidden={}*{}={}",
                h_in.len(),
                seq,
                hidden,
                seq * hidden
            )));
        }
        // Fast-path seq=1 by delegating to the existing seq=1 forward.
        // This keeps the seq=1 hot path bit-identical and avoids paying
        // the scalar-loop dispatch overhead inside
        // `_multi_with_capacity` for a single token.
        if seq == 1 {
            return self.forward_shells(h_in, &[1, 1, hidden], past_seq_len);
        }

        let mut h_f32 = h_in.to_vec();
        let n_layers = self.layers.len();
        for i in 0..n_layers {
            let lid = self.layers[i].lid;
            if self.layers[i].past_seq_len != past_seq_len {
                return Err(RunnerError::Internal(format!(
                    "L{lid}: past_seq_len mismatch (caller {past_seq_len} vs layer {})",
                    self.layers[i].past_seq_len
                )));
            }
            // Grow KV until it fits past_seq_len + seq. Geometric grow
            // (doubling) preserves O(N) cumulative alloc traffic; a
            // single seq=8 step from kv_capacity=32 needs at most two
            // doublings.
            while past_seq_len + seq > self.layers[i].kv_capacity {
                grow_kv_capacity(&mut self.layers[i])?;
            }
            let capacity = self.layers[i].kv_capacity;

            // shell_forward_decode_int4_multi_with_capacity writes
            // present_k / present_v in place into slots
            // [past_seq_len, past_seq_len + seq) of each head — no
            // separate write_present_kv calls needed at the engine
            // boundary.
            //
            // Destructure the layer into disjoint &mut borrows so the
            // shell (immut), past_k (mut), past_v (mut) all coexist —
            // otherwise indexing `self.layers[i]` three times trips
            // the borrow checker (E0502).
            let outs = {
                let layer = &mut self.layers[i];
                shell_forward_decode_int4_multi_with_capacity(
                    &layer.int4_shell,
                    &h_f32,
                    &mut layer.past_k,
                    &mut layer.past_v,
                    past_seq_len,
                    capacity,
                    seq,
                )
            };
            self.layers[i].past_seq_len = past_seq_len + seq;

            // Per-token expert dispatch + residual combine. Same logic
            // as forward_shells, just looped over seq tokens. The shell
            // chose the routing_ids/weights per token in the batched
            // forward; we just apply them.
            if outs.routing_ids.len() != seq * top_k || outs.routing_weights.len() != seq * top_k {
                return Err(RunnerError::Internal(format!(
                    "L{lid} multi-token routing shape unexpected: ids={} weights={} (expected seq*top_k = {}*{})",
                    outs.routing_ids.len(),
                    outs.routing_weights.len(),
                    seq,
                    top_k
                )));
            }
            for t in 0..seq {
                let attn_post_t = &outs.attn_out_post_norm[t * hidden..(t + 1) * hidden];
                let mut moe = vec![0.0f32; hidden];
                for k in 0..top_k {
                    let eid = outs.routing_ids[t * top_k + k] as u32;
                    let w = outs.routing_weights[t * top_k + k];
                    let y_f32 = self.dispatch_expert(lid, eid, attn_post_t)?;
                    for j in 0..hidden {
                        moe[j] += w * y_f32[j];
                    }
                }
                let attn_res_t = &outs.attn_residual[t * hidden..(t + 1) * hidden];
                let shared_t = &outs.shared_expert_out[t * hidden..(t + 1) * hidden];
                let h_next_t = &mut h_f32[t * hidden..(t + 1) * hidden];
                for j in 0..hidden {
                    h_next_t[j] = attn_res_t[j] + shared_t[j] + moe[j];
                }
            }
        }

        Ok(h_f32)
    }

    /// Take the last position of `h_f32` (shape `[1, tail_len, hidden]`,
    /// row-major) and run the head IR. Returns the vocab-sized logits
    /// at that position. Last-stage only.
    pub fn forward_head_last(
        &mut self,
        h_f32: &[f32],
        tail_len: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let _t0 = Instant::now();
        let hidden = self.manifest.hidden_size as usize;
        if tail_len == 0 || h_f32.len() < tail_len * hidden {
            return Err(RunnerError::Internal(format!(
                "forward_head_last: tail_len={} h.len={} hidden={}",
                tail_len,
                h_f32.len(),
                hidden
            )));
        }
        let head = self
            .head
            .as_mut()
            .ok_or_else(|| RunnerError::Internal("forward_head on non-last stage".into()))?;
        let last_off = (tail_len - 1) * hidden;
        let last_h_bf16 = f32_to_bf16_bytes(&h_f32[last_off..last_off + hidden]);
        let head_in = head.input_name(0)?;
        head.set_input(&head_in, DType::Bf16, &[1, 1, hidden], &last_h_bf16)?;
        head.infer()?;
        let (head_dt, head_shape, head_bytes) = head.output(0)?;
        if head_shape.last() != Some(&(self.manifest.vocab_size as usize)) {
            return Err(RunnerError::Internal(format!(
                "head output shape {:?} doesn't end with vocab_size {}",
                head_shape, self.manifest.vocab_size
            )));
        }
        let logits = match head_dt {
            DType::F32 => read_f32(&head_bytes),
            DType::F16 => f16_bytes_to_f32(&head_bytes),
            DType::Bf16 => bf16_bytes_to_f32(&head_bytes),
            _ => return Err(RunnerError::Internal(format!("head dtype {:?}", head_dt))),
        };
        // autolab/k26-perf q1 instrumentation: per-token head timing.
        info!(
            stage = "head",
            duration_us = _t0.elapsed().as_micros() as u64,
            tail_len,
            "stage_timing"
        );
        Ok(logits)
    }

    /// Generate tokens with full sampling (temperature / top-p /
    /// repetition penalty / EOS stop). Returns the vector of generated
    /// token IDs **excluding** the prompt and **excluding** the EOS
    /// token that triggered termination.
    pub fn generate(
        &mut self,
        prompt_ids: &[i64],
        max_tokens: usize,
        cfg: &crate::sampling::SamplingConfig,
    ) -> Result<Vec<i64>, RunnerError> {
        self.reset_kv();
        let eos: Vec<i64> = self
            .manifest
            .eos_token_ids
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut rng = crate::sampling::init_rng(cfg.seed);
        let mut generated = Vec::with_capacity(max_tokens);

        // Prefill token-by-token to keep shell input shapes uniform (avoids
        // the OV 2026.1.0 CPU snippets shape-specialization bug we hit on
        // shape changes).
        //
        // iter 065 (prefill-hint static schedule): bracket the prefill
        // loop with enter / exit so per-expert hit bumps during prefill
        // get diverted to `prefill_expert_observations` when the hint
        // is enabled. The `exit_prefill_and_merge_hints` call folds
        // them back into `expert_hits` with the configured weight, so
        // decode iteration #1 below sees a hint-seeded map. When the
        // hint is disabled (default 0.0 weight) both calls are no-ops.
        info!(prompt_len = prompt_ids.len(), "prefill (token-by-token)");
        let mut history: Vec<i64> = Vec::with_capacity(prompt_ids.len() + max_tokens);
        let mut last_logits: Option<Vec<f32>> = None;
        let t_pre = Instant::now();
        self.enter_prefill();
        for (i, &t) in prompt_ids.iter().enumerate() {
            history.push(t);
            let logits = self.step(&history, 1)?;
            last_logits = Some(logits);
            if (i + 1) % 8 == 0 || i + 1 == prompt_ids.len() {
                info!(
                    "prefill {}/{} elapsed={:.1}s",
                    i + 1,
                    prompt_ids.len(),
                    t_pre.elapsed().as_secs_f64()
                );
            }
        }
        let prefill_secs = t_pre.elapsed().as_secs_f64();
        let merged = self.exit_prefill_and_merge_hints();
        info!(
            secs = prefill_secs,
            tok_per_s = prompt_ids.len() as f64 / prefill_secs,
            prefill_hint_entries_merged = merged,
            "prefill done"
        );

        // First generated token from the LAST prefill step's logits.
        if let Some(l) = last_logits {
            let next = crate::sampling::sample(&l, &history, cfg, &mut rng);
            if eos.contains(&next) {
                return Ok(generated);
            }
            history.push(next);
            generated.push(next);
        }

        // Decode.
        for step_i in 1..max_tokens {
            let t_step = Instant::now();
            let logits = self.step(&history, 1)?;
            let next = crate::sampling::sample(&logits, &history, cfg, &mut rng);
            if eos.contains(&next) {
                debug!(step = step_i, token = next, "EOS — stopping");
                break;
            }
            history.push(next);
            generated.push(next);
            debug!(
                step = step_i,
                token = next,
                elapsed_ms = t_step.elapsed().as_secs_f64() * 1000.0,
                cached_experts = match &self.experts {
                    ExpertCache::OvIr(c) => c.map.len(),
                    ExpertCache::Int4Bin(c) => c.map.len(),
                    ExpertCache::SafetensorsBin(c) => c.map.len(),
                },
                "decode step"
            );
        }
        Ok(generated)
    }

    /// Back-compat: equivalent to `generate(..., &SamplingConfig::default())`.
    pub fn generate_argmax(
        &mut self,
        prompt_ids: &[i64],
        max_tokens: usize,
    ) -> Result<Vec<i64>, RunnerError> {
        self.generate(
            prompt_ids,
            max_tokens,
            &crate::sampling::SamplingConfig::default(),
        )
    }
}

fn read_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut a = [0u8; 4];
        a.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
        out.push(f32::from_le_bytes(a));
    }
    out
}

/// autolab iter 054 (expert pinning): pure helper — pick the top-`n`
/// expert IDs from a hit map, breaking ties on ascending expert id
/// for determinism. Returns at most `n` IDs (fewer if the map has
/// fewer distinct entries). Stable across runs so A/B campaigns
/// reproduce identical pin sets when fed identical hit histograms.
///
/// Separated from `Runner::pin_top_n_per_layer` so unit tests can
/// exercise the selection logic without spinning up a full Runner
/// (which would require K2.6 weights on disk).
fn select_top_n_by_hits(hits: &HashMap<u32, u64>, n: u32) -> Vec<u32> {
    if n == 0 || hits.is_empty() {
        return Vec::new();
    }
    let mut by_count: Vec<(u32, u64)> = hits.iter().map(|(&k, &v)| (k, v)).collect();
    // Sort descending by count, ascending by id on ties.
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let take = (n as usize).min(by_count.len());
    by_count
        .into_iter()
        .take(take)
        .map(|(eid, _)| eid)
        .collect()
}

/// autolab iter 056 (cache-aware dispatch): produce a permutation of
/// `0..routing_ids.len()` ordered by descending hit count in `hits`,
/// breaking ties by ascending original index (which is router-score
/// order — TOPK[0] is the highest-scored expert). Returns indices into
/// `routing_ids`, NOT expert IDs — the dispatch loop uses these to
/// look up `routing_ids[k]` and `routing_weights[k]` from the shell
/// output without losing the original alignment.
///
/// Example: `routing_ids = [42, 99, 7, 200, 50, 1, 88, 12]` (router
/// top-8 by score), `hits = {42: 10, 99: 0, 7: 50, 200: 5, 50: 100, 1:
/// 0, 88: 50, 12: 0}` ⇒ returned permutation `[4, 2, 6, 0, 3, 1, 5,
/// 7]` (50 first at 100 hits, then 7 and 88 tied at 50 in router-
/// score order, then 42 at 10, then 200 at 5, then 99/1/12 tied at 0
/// in router-score order).
///
/// **Pure function — no side effects.** The runner's dispatch loop
/// preserves byte-identical output by:
/// (a) doing hit-rate / `expert_hits` bookkeeping in the original
///     `k = 0..effective_top_k` order,
/// (b) calling `dispatch_expert` in the cache-aware order (this loop),
/// (c) summing the weighted expert outputs into `moe` in original
///     order so the float-addition rounding chain is unchanged.
///
/// `hits` missing an expert ID counts as zero hits — typical on the
/// very first prefill token before any hit data has accumulated. In
/// that case the permutation degenerates to the identity (all zeros,
/// tie-broken on ascending k), which matches the current router-score
/// dispatch order exactly.
///
/// autolab iter 057 (async kernel scheduling — speculative prefetch):
/// pure helper — pick the top-`n` hit-frequent expert IDs from layer
/// `i + 1`'s histogram so the runner can `try_submit` them to the C1
/// prefetcher right before layer `i`'s ~150 ms expert dispatch begins.
/// This is a thin wrapper around `select_top_n_by_hits` so the runner
/// site (which has to deal with the prefetcher option, the lid
/// lookup, and the cumulative submitted counter) stays free of
/// selection logic and the same heavy-tail tie-breaking rules apply
/// (descending count, ascending expert id).
///
/// Returns an empty vec when:
///   - `n == 0` (caller disabled speculative prefetch),
///   - `next_layer_hits` is empty (the very first prefill token before
///     any dispatch data has accumulated for that layer — degenerate
///     no-op, ensuring iter 057 is byte-identical to iter 056 on the
///     first per-prompt forward_shells call).
///
/// Stable across runs (HashMap iteration order is sorted away by the
/// inner `select_top_n_by_hits`) so A/B campaigns reproduce identical
/// prefetch streams when fed identical hit histograms — critical for
/// attributing tok/s deltas to the scheduling change rather than
/// noise in the prefetcher's submission pattern.
///
/// Separated from `Runner::forward_shells` so unit tests can drive
/// the scheduling logic without a loaded Runner.
fn speculative_prefetch_expert_ids(next_layer_hits: &HashMap<u32, u64>, n: u32) -> Vec<u32> {
    select_top_n_by_hits(next_layer_hits, n)
}

/// autolab iter 065 (prefill-hint static schedule): pure helper —
/// fold a per-layer prefill-observation map into a per-layer
/// `expert_hits` map using a configurable weight. Mutates `hits`
/// in place; reads `obs` non-destructively (the caller clears it
/// after the merge so observation buffers can be reused across
/// prompts without realloc).
///
/// Per `(layer i, expert eid)`: `hits[i][eid] += round(w * obs[i][eid])`.
/// `saturating_add` on the `u64` slot so a pathologically long prompt
/// at high weight can't wrap silently. Zero / NaN-weighted entries are
/// skipped (no map insert, no `merged` count bump) so the returned
/// count reflects only effective merges.
///
/// Separated from `Runner::exit_prefill_and_merge_hints` so unit
/// tests can exercise the merge math against synthetic histograms
/// without spinning up a loaded Runner (which needs K2.6 IRs on disk).
/// Also keeps the runner-side method short and one-concern.
///
/// Returns the count of `(layer, expert)` slots that received a
/// nonzero contribution. Useful for both the runner's info-log and
/// the unit tests that assert "yes, the merge actually happened".
fn merge_prefill_observations_into_hits(
    hits: &mut [HashMap<u32, u64>],
    obs: &[HashMap<u32, u64>],
    w: f32,
) -> usize {
    // Runner allocates `hits` and `obs` to identical length at load
    // time, so in production this never trips. We still tolerate a
    // mismatch at runtime (skip the over-length obs slots) instead
    // of asserting — callers in defensive code paths (e.g. a future
    // multi-stage worker that hadn't yet allocated its hits map)
    // shouldn't panic the inference loop on a length skew.
    if !(w.is_finite() && w > 0.0) {
        return 0;
    }
    let mut merged = 0usize;
    for (i, obs_i) in obs.iter().enumerate() {
        let target = match hits.get_mut(i) {
            Some(t) => t,
            None => continue,
        };
        for (&eid, &count) in obs_i.iter() {
            // f32 multiply then round-to-nearest. For the weights we
            // accept (0..~10) and any plausible prompt length, the f32
            // mantissa is ample. Saturating cast guards the u64 add.
            let weighted_f = (w * count as f32).round();
            if !weighted_f.is_finite() || weighted_f <= 0.0 {
                continue;
            }
            let weighted = weighted_f as u64;
            if weighted == 0 {
                continue;
            }
            let slot = target.entry(eid).or_insert(0);
            *slot = slot.saturating_add(weighted);
            merged += 1;
        }
    }
    merged
}

/// Separated from `Runner::forward_shells` so unit tests can verify
/// the permutation logic without a loaded model.
fn cache_aware_dispatch_order(routing_ids: &[i64], hits: &HashMap<u32, u64>) -> Vec<usize> {
    let mut order: Vec<(usize, u64)> = routing_ids
        .iter()
        .enumerate()
        .map(|(k, &id)| {
            // Guard cast: router IDs are nonneg expert ids in [0, 384).
            // A pathological negative id would just look up as zero hits.
            let count = if id >= 0 {
                *hits.get(&(id as u32)).unwrap_or(&0)
            } else {
                0
            };
            (k, count)
        })
        .collect();
    // Sort descending by count, ascending by original index (= router
    // score order) on ties. This guarantees that when `hits` is empty
    // (e.g. very first prefill token) the permutation is the identity.
    order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    order.into_iter().map(|(k, _)| k).collect()
}

/// Double the per-head slot capacity of one layer's KV buffers,
/// preserving the populated `past_seq_len` rows for each head.
///
/// Old layout: `[NUM_HEADS, old_cap, HEAD_DIM]`, with head h's data
/// living at offset `h * old_cap * HEAD_DIM`. New layout has the
/// same head-major arrangement but with `new_cap = 2 * old_cap`, so
/// every head's base shifts. We allocate a fresh buffer and copy the
/// populated prefix `[0..past_seq_len]` per head. Returns
/// `RunnerError::Internal` if either allocation fails — long-context
/// generations could plausibly OOM mid-decode.
fn grow_kv_capacity(layer: &mut LayerState) -> Result<(), RunnerError> {
    let old_cap = layer.kv_capacity;
    let new_cap = old_cap * 2;
    let ps = layer.past_seq_len;
    let new_k = grow_kv_buffer(&layer.past_k, ps, old_cap, new_cap, QK_HEAD_DIM)
        .map_err(|e| RunnerError::Internal(format!("L{} grow past_k: {e}", layer.lid)))?;
    let new_v = grow_kv_buffer(&layer.past_v, ps, old_cap, new_cap, V_HEAD_DIM)
        .map_err(|e| RunnerError::Internal(format!("L{} grow past_v: {e}", layer.lid)))?;
    layer.past_k = new_k;
    layer.past_v = new_v;
    layer.kv_capacity = new_cap;
    Ok(())
}

/// Same geometric growth as `grow_kv_capacity` but for the first
/// stage's layer-0 cache. Kept as a sibling rather than generalized
/// over a trait because layer-0 has different surrounding fields
/// (embedding pin, no `lid`, no `int4_shell`) and the rare grow call
/// would not benefit from the abstraction.
fn grow_layer0_kv_capacity(l0: &mut Layer0State) -> Result<(), RunnerError> {
    let old_cap = l0.kv_capacity;
    let new_cap = old_cap * 2;
    let ps = l0.past_seq_len;
    let new_k = grow_kv_buffer(&l0.past_k, ps, old_cap, new_cap, QK_HEAD_DIM)
        .map_err(|e| RunnerError::Internal(format!("L0 grow past_k: {e}")))?;
    let new_v = grow_kv_buffer(&l0.past_v, ps, old_cap, new_cap, V_HEAD_DIM)
        .map_err(|e| RunnerError::Internal(format!("L0 grow past_v: {e}")))?;
    l0.past_k = new_k;
    l0.past_v = new_v;
    l0.kv_capacity = new_cap;
    Ok(())
}

/// Inner helper: allocate a fresh `[NUM_HEADS, new_cap, head_dim]`
/// buffer of bf16-as-u16, copy the populated `past_seq` prefix per head
/// from a `[NUM_HEADS, old_cap, head_dim]` source. The rest is zero.
/// Pure over buffers — no Int4Shell required, which keeps it unit-testable.
///
/// Uses `try_reserve_exact` + `resize` so OOM at long context bubbles
/// up as a recoverable `Err` instead of an abort from the global
/// allocator. autolab 029 (A8): elements are 2 bytes, not 4.
fn grow_kv_buffer(
    src: &[u16],
    past_seq: usize,
    old_cap: usize,
    new_cap: usize,
    head_dim: usize,
) -> Result<Vec<u16>, String> {
    debug_assert!(new_cap >= old_cap);
    debug_assert!(past_seq <= old_cap);
    debug_assert_eq!(src.len(), NUM_HEADS * old_cap * head_dim);
    let total = NUM_HEADS * new_cap * head_dim;
    let mut dst: Vec<u16> = Vec::new();
    dst.try_reserve_exact(total).map_err(|e| {
        format!(
            "alloc {total} u16/bf16 ({:.1} MB) failed: {e}",
            (total * 2) as f64 / 1e6
        )
    })?;
    dst.resize(total, 0u16);
    if past_seq == 0 {
        return Ok(dst);
    }
    for h in 0..NUM_HEADS {
        let s = h * old_cap * head_dim;
        let d = h * new_cap * head_dim;
        dst[d..d + past_seq * head_dim].copy_from_slice(&src[s..s + past_seq * head_dim]);
    }
    Ok(dst)
}

/// Convert one f32 to bf16 bits via round-to-nearest-even. Matches the
/// rounding `half::bf16::from_f32` would do; inlined here so the hot
/// per-token write loop doesn't depend on the `half` crate at this site.
#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN — keep mantissa nonzero so the round-trip stays a NaN
        // rather than collapsing to ±inf when we shift back.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// Write the new step's per-head K (or V) row at slot `past_seq`
/// inside a `[NUM_HEADS, capacity, HEAD_DIM]` bf16-as-u16 buffer.
/// `present` is still f32 (we are the conversion site). No allocation,
/// no shift — the slot exists because the caller pre-allocated /
/// grew `capacity` to be `> past_seq`.
fn write_present_kv(
    buf: &mut [u16],
    present: &[f32],
    past_seq: usize,
    capacity: usize,
    num_heads: usize,
    head_dim: usize,
) {
    debug_assert!(past_seq < capacity);
    debug_assert_eq!(buf.len(), num_heads * capacity * head_dim);
    debug_assert_eq!(present.len(), num_heads * head_dim);
    for h in 0..num_heads {
        let dst_off = h * capacity * head_dim + past_seq * head_dim;
        let src_off = h * head_dim;
        let dst = &mut buf[dst_off..dst_off + head_dim];
        let src = &present[src_off..src_off + head_dim];
        for i in 0..head_dim {
            dst[i] = f32_to_bf16_bits(src[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upconvert one bf16 bit-pattern to f32 the same way the SDPA
    /// kernels do (used in test assertions below).
    fn bf16_bits_to_f32(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    #[test]
    fn write_present_kv_into_empty_slot() {
        // 2 heads, capacity=3, head_dim=2, past_seq=0.
        // Buffer starts zero; expect present rows at slot 0 of each head.
        // (autolab 029 / A8: buf is now bf16-as-u16; we feed f32 in.)
        let mut buf = vec![0u16; 2 * 3 * 2];
        let present = vec![1.0_f32, 2.0, 3.0, 4.0]; // h0=[1,2], h1=[3,4]
        write_present_kv(&mut buf, &present, 0, 3, 2, 2);
        // head 0 base = 0,        slot 0 = [1, 2]
        // head 1 base = capacity*head_dim = 6, slot 0 = [3, 4]
        // Small integers round-trip exactly through bf16.
        assert_eq!(bf16_bits_to_f32(buf[0]), 1.0);
        assert_eq!(bf16_bits_to_f32(buf[1]), 2.0);
        assert_eq!(bf16_bits_to_f32(buf[6]), 3.0);
        assert_eq!(bf16_bits_to_f32(buf[7]), 4.0);
        // Unfilled slots untouched.
        assert_eq!(&buf[2..6], &[0u16; 4]);
        assert_eq!(&buf[8..12], &[0u16; 4]);
    }

    #[test]
    fn write_present_kv_into_middle_slot() {
        // 2 heads, capacity=4, head_dim=2, past_seq=2.
        // Pre-populate slots 0..2 of each head with bf16 values, then write slot 2.
        let mut buf = vec![0u16; 2 * 4 * 2];
        // head 0 base=0
        buf[0..2].copy_from_slice(&[super::f32_to_bf16_bits(10.0), super::f32_to_bf16_bits(11.0)]);
        buf[2..4].copy_from_slice(&[super::f32_to_bf16_bits(12.0), super::f32_to_bf16_bits(13.0)]);
        // head 1 base = capacity*head_dim = 8
        buf[8..10].copy_from_slice(&[super::f32_to_bf16_bits(20.0), super::f32_to_bf16_bits(21.0)]);
        buf[10..12]
            .copy_from_slice(&[super::f32_to_bf16_bits(22.0), super::f32_to_bf16_bits(23.0)]);
        let present = vec![14.0_f32, 15.0, 24.0, 25.0]; // h0=[14,15], h1=[24,25]
        write_present_kv(&mut buf, &present, 2, 4, 2, 2);
        // head 0 slot 2
        assert_eq!(bf16_bits_to_f32(buf[4]), 14.0);
        assert_eq!(bf16_bits_to_f32(buf[5]), 15.0);
        // head 1 slot 2
        assert_eq!(bf16_bits_to_f32(buf[12]), 24.0);
        assert_eq!(bf16_bits_to_f32(buf[13]), 25.0);
        // Existing slots untouched
        assert_eq!(bf16_bits_to_f32(buf[0]), 10.0);
        assert_eq!(bf16_bits_to_f32(buf[8]), 20.0);
    }

    #[test]
    fn grow_kv_buffer_doubles_and_preserves_data() {
        // Stamp a unique value at head h, slot 0, dim 0 of a
        // [NUM_HEADS, 2, QK_HEAD_DIM] u16 buffer, then double to cap=4.
        // Each head's base offset shifts from h*2*D to h*4*D — the
        // stamp should still be at the new base offset.
        let mut src = vec![0u16; NUM_HEADS * 2 * QK_HEAD_DIM];
        for h in 0..NUM_HEADS {
            // store small ints as bf16 bits — h+1
            let v = (h + 1) as f32;
            src[h * 2 * QK_HEAD_DIM] = super::f32_to_bf16_bits(v);
        }
        let dst = grow_kv_buffer(&src, 1, 2, 4, QK_HEAD_DIM).expect("alloc");
        assert_eq!(dst.len(), NUM_HEADS * 4 * QK_HEAD_DIM);
        for h in 0..NUM_HEADS {
            assert_eq!(
                bf16_bits_to_f32(dst[h * 4 * QK_HEAD_DIM]),
                (h + 1) as f32,
                "head {h} stamp lost"
            );
        }
    }

    #[test]
    fn grow_kv_buffer_from_empty_is_zero_filled() {
        let src = vec![0u16; NUM_HEADS * 2 * QK_HEAD_DIM];
        let dst = grow_kv_buffer(&src, 0, 2, 4, QK_HEAD_DIM).expect("alloc");
        assert_eq!(dst.len(), NUM_HEADS * 4 * QK_HEAD_DIM);
        assert!(dst.iter().all(|&x| x == 0));
    }

    #[test]
    fn f32_to_bf16_bits_matches_half_crate() {
        // Cross-check our hand-rolled rounding against `half::bf16::from_f32`
        // for a handful of values: zero, ±1, ±0.5, small powers of 2,
        // a denormal-ish value, and a few transcendentals.
        use half::bf16;
        let cases: &[f32] = &[
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 0.125, 1.0e-30, 3.14159265, -42.5,
        ];
        for &x in cases {
            let ours = super::f32_to_bf16_bits(x);
            let theirs = bf16::from_f32(x).to_bits();
            assert_eq!(
                ours, theirs,
                "mismatch for f32={x:?}: ours=0x{ours:04x} theirs=0x{theirs:04x}"
            );
        }
        // NaN: any pattern with exp=0xFF and nonzero mantissa is valid.
        // Bit-equality not required for NaN.
        let nan_ours = super::f32_to_bf16_bits(f32::NAN);
        let nan_back = f32::from_bits((nan_ours as u32) << 16);
        assert!(nan_back.is_nan(), "ours: 0x{nan_ours:04x} not NaN");
    }

    /// autolab iter 054 (expert pinning): pure-function tests for the
    /// selection helper. We can't exercise `Runner::pin_top_n_per_layer`
    /// directly from a unit test (it needs a loaded Runner ⇒ K2.6 IRs
    /// on disk) but the selection logic is the load-bearing part and
    /// is straightforward to test in isolation.
    #[test]
    fn select_top_n_picks_highest_counts() {
        let mut hits = HashMap::new();
        hits.insert(10u32, 50u64);
        hits.insert(20, 100);
        hits.insert(30, 75);
        hits.insert(40, 25);
        let top = super::select_top_n_by_hits(&hits, 2);
        assert_eq!(top, vec![20, 30], "top-2 should be highest two counts");
    }

    #[test]
    fn select_top_n_breaks_ties_on_ascending_id() {
        let mut hits = HashMap::new();
        // Three experts tied at 50 hits each; we expect ascending id order.
        hits.insert(7u32, 50u64);
        hits.insert(3, 50);
        hits.insert(11, 50);
        // One expert with strictly higher count.
        hits.insert(99, 100);
        let top = super::select_top_n_by_hits(&hits, 3);
        // 99 first (highest count), then 3, 7 (tied, ascending id).
        assert_eq!(top, vec![99, 3, 7]);
    }

    #[test]
    fn select_top_n_returns_all_when_fewer_distinct_experts_than_n() {
        let mut hits = HashMap::new();
        hits.insert(1u32, 5u64);
        hits.insert(2, 10);
        // Asking for top-100 from only 2 entries returns both.
        let top = super::select_top_n_by_hits(&hits, 100);
        assert_eq!(top, vec![2, 1]);
    }

    #[test]
    fn select_top_n_with_zero_n_returns_empty() {
        let mut hits = HashMap::new();
        hits.insert(1u32, 100u64);
        assert!(super::select_top_n_by_hits(&hits, 0).is_empty());
    }

    #[test]
    fn select_top_n_with_empty_hits_returns_empty() {
        let hits: HashMap<u32, u64> = HashMap::new();
        assert!(super::select_top_n_by_hits(&hits, 10).is_empty());
    }

    #[test]
    fn select_top_n_deterministic_across_call_order() {
        // Same logical histogram inserted in different orders should
        // produce identical results (HashMap iteration is otherwise
        // nondeterministic). This is the property A/B campaigns rely on.
        let mut a = HashMap::new();
        a.insert(5u32, 50u64);
        a.insert(1, 50);
        a.insert(9, 50);
        let mut b = HashMap::new();
        b.insert(9u32, 50u64);
        b.insert(5, 50);
        b.insert(1, 50);
        assert_eq!(
            super::select_top_n_by_hits(&a, 3),
            super::select_top_n_by_hits(&b, 3),
            "tie-breaking must be insertion-order-independent"
        );
    }

    /// autolab iter 054: a heavy-tailed distribution like K2.6's real
    /// router output should pick the long-tail-head exactly. Mocks the
    /// canonical "10% of experts cover 80% of fires" shape and verifies
    /// the top-N selection captures the hot-set.
    #[test]
    fn select_top_n_captures_heavy_tail_head() {
        let mut hits = HashMap::new();
        // 38 hot experts each at 100 hits ⇒ 3800 fires.
        for eid in 0..38u32 {
            hits.insert(eid, 100);
        }
        // 346 cold experts each at 3 hits ⇒ 1038 fires. Total ~4838.
        // The 38 hot experts cover 3800/4838 = 78.5%, matching the
        // "~80% with 10%" heuristic the task description cites.
        for eid in 38..384u32 {
            hits.insert(eid, 3);
        }
        let top = super::select_top_n_by_hits(&hits, 38);
        assert_eq!(top.len(), 38);
        // Sorted ascending after the heavy-tail head (tied at 100), so
        // the result is exactly 0..38 in order.
        let expected: Vec<u32> = (0..38).collect();
        assert_eq!(top, expected);
    }

    // ====================================================================
    // autolab iter 056 (cache-aware dispatch) tests
    // ====================================================================
    //
    // The dispatch reorder lives inside `Runner::forward_shells`, which
    // requires a loaded K2.6 model to drive end-to-end. The unit tests
    // below cover:
    //
    //  (1) the pure permutation helper `cache_aware_dispatch_order` —
    //      correctness across the cases the dispatch loop will see
    //      (heavy-tail, all-zero hits, ties, etc.);
    //  (2) the load-bearing bit-identity property — Phase 2 + Phase 3 of
    //      the dispatch loop reproduced as a pure simulator so we can
    //      assert that for ANY permutation, the accumulated `moe[]`
    //      vector is byte-identical to the router-score-order baseline.
    //      This is the property the task says must hold.

    #[test]
    fn cache_aware_dispatch_order_with_empty_hits_is_identity() {
        // First prefill token: expert_hits is empty for this layer.
        // The permutation must degenerate to the identity so behavior
        // is byte-identical to router-score order until hit data warms.
        let routing_ids: Vec<i64> = vec![42, 99, 7, 200, 50, 1, 88, 12];
        let hits: HashMap<u32, u64> = HashMap::new();
        let order = super::cache_aware_dispatch_order(&routing_ids, &hits);
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn cache_aware_dispatch_order_sorts_by_descending_hits() {
        // Routing-score order: 42 (top), 99, 7, 200, 50, 1, 88, 12.
        // Hit counts: 50 hottest at 100, then 7 and 88 tied at 50, then
        // 42 at 10, then 200 at 5, then 99/1/12 at 0. Expected permutation:
        // [4 (50), 2 (7, tied), 6 (88, tied), 0 (42), 3 (200), 1 (99, tied),
        //  5 (1, tied), 7 (12, tied)].
        let routing_ids: Vec<i64> = vec![42, 99, 7, 200, 50, 1, 88, 12];
        let mut hits = HashMap::new();
        hits.insert(42u32, 10u64);
        hits.insert(99, 0);
        hits.insert(7, 50);
        hits.insert(200, 5);
        hits.insert(50, 100);
        hits.insert(1, 0);
        hits.insert(88, 50);
        hits.insert(12, 0);
        let order = super::cache_aware_dispatch_order(&routing_ids, &hits);
        assert_eq!(order, vec![4, 2, 6, 0, 3, 1, 5, 7]);
    }

    #[test]
    fn cache_aware_dispatch_order_missing_ids_treated_as_zero() {
        // Hits map only covers two of the five experts; the other three
        // count as zero hits and tie-break by original index.
        let routing_ids: Vec<i64> = vec![10, 20, 30, 40, 50];
        let mut hits = HashMap::new();
        hits.insert(30u32, 100u64);
        hits.insert(50, 50);
        let order = super::cache_aware_dispatch_order(&routing_ids, &hits);
        // 30 first (100), then 50 (50), then 10/20/40 tied at 0 in
        // original index order.
        assert_eq!(order, vec![2, 4, 0, 1, 3]);
    }

    #[test]
    fn cache_aware_dispatch_order_heavy_tail_puts_hot_first() {
        // Simulate the K2.6 heavy-tail: of the 8 routed experts, 2 are
        // hot (in the warm-set), 6 are cold. Permutation must put the
        // 2 hot ones first regardless of their router-score positions.
        let routing_ids: Vec<i64> = vec![5, 1, 9, 3, 7, 2, 8, 4];
        let mut hits = HashMap::new();
        // Hot set: experts 3 and 8 — at indices 3 and 6 in routing_ids.
        hits.insert(3u32, 10_000);
        hits.insert(8, 10_000);
        // Cold set: everyone else at uniformly low count.
        for &id in &[5u32, 1, 9, 7, 2, 4] {
            hits.insert(id, 1);
        }
        let order = super::cache_aware_dispatch_order(&routing_ids, &hits);
        // First two slots must be the hot indices in router-score-tied
        // ascending order: 3 (idx 3), then 8 (idx 6). The rest are at
        // count 1, tie-broken by ascending original index.
        assert_eq!(&order[..2], &[3, 6]);
        // The cold tail is a permutation of the remaining six indices,
        // ordered by ascending original index (all tied at count 1).
        assert_eq!(&order[2..], &[0, 1, 2, 4, 5, 7]);
    }

    #[test]
    fn cache_aware_dispatch_order_is_a_permutation() {
        // For any input the returned vec must be a valid permutation of
        // 0..len — same length, no duplicates, every index present.
        let routing_ids: Vec<i64> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let mut hits = HashMap::new();
        for (i, &id) in routing_ids.iter().enumerate() {
            // Spread counts so every order pair is exercised.
            hits.insert(id as u32, (i * 17 + 3) as u64);
        }
        let order = super::cache_aware_dispatch_order(&routing_ids, &hits);
        assert_eq!(order.len(), routing_ids.len());
        let mut sorted = order.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..routing_ids.len()).collect();
        assert_eq!(sorted, expected, "must be a permutation of 0..K");
    }

    /// Reproduce Phases 2 + 3 of the `forward_shells` dispatch loop as
    /// a pure simulator. `dispatch_seq` is the order in which we call
    /// the (mocked) expert; the per-expert outputs are looked up by
    /// original index from `expert_outs_by_k`. Phase 3 sums in
    /// ascending original-index order regardless of dispatch order.
    fn simulate_phase23(
        routing_weights: &[f32],
        expert_outs_by_k: &[Vec<f32>],
        dispatch_seq: &[usize],
        hidden: usize,
    ) -> Vec<f32> {
        let effective_top_k = routing_weights.len();
        let mut stash: Vec<Option<Vec<f32>>> = (0..effective_top_k).map(|_| None).collect();
        // Phase 2: dispatch in the given order, stash outputs by k.
        for &k in dispatch_seq {
            stash[k] = Some(expert_outs_by_k[k].clone());
        }
        // Phase 3: accumulate weighted sum in original k order.
        let mut moe = vec![0.0f32; hidden];
        for k in 0..effective_top_k {
            if let Some(y) = &stash[k] {
                let w = routing_weights[k];
                for j in 0..hidden {
                    moe[j] += w * y[j];
                }
            }
        }
        moe
    }

    /// **The load-bearing test for iter 056.** For K=8 on a hidden=128
    /// vector with synthetic-but-non-trivial weights + expert outputs,
    /// the simulated dispatch in router-score order must produce
    /// byte-identical `moe[]` to the simulated dispatch in cache-aware
    /// (and reverse, and shuffled) order. This is the property that
    /// makes cache-aware dispatch safe to enable: it cannot change
    /// model output.
    #[test]
    fn cache_aware_dispatch_bit_identical_to_router_score_order() {
        const K: usize = 8;
        const HIDDEN: usize = 128;

        // Pseudo-random but deterministic weights and outputs so the
        // test reproduces across runs. Mixed-sign, mixed-magnitude
        // values to make floating-point reordering actually matter
        // (catches a regression where Phase 3 also reordered).
        let routing_weights: Vec<f32> = (0..K)
            .map(|k| {
                // K2.6-shaped weights: heavy on the first few, decaying.
                let raw = 1.0 / (k as f32 + 1.0);
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                raw * sign * 0.7
            })
            .collect();
        let expert_outs_by_k: Vec<Vec<f32>> = (0..K)
            .map(|k| {
                (0..HIDDEN)
                    .map(|j| {
                        // Non-trivial values per (k, j): mix of trig +
                        // polynomial so different orderings of the
                        // weighted sum produce different float results
                        // unless Phase 3 actually fixes the order.
                        let kx = k as f32;
                        let jx = j as f32;
                        (kx * 0.31 + jx * 0.017).sin() * 1.7 + (jx * 0.005).cos() - kx * 0.013
                    })
                    .collect()
            })
            .collect();

        // Baseline: router-score order = identity permutation.
        let router_score_order: Vec<usize> = (0..K).collect();
        let baseline = simulate_phase23(
            &routing_weights,
            &expert_outs_by_k,
            &router_score_order,
            HIDDEN,
        );

        // Cache-aware order driven by a synthetic heavy-tail hit map.
        // We don't need real expert IDs for the simulator — the order
        // helper operates on indices, so we wire routing_ids = [10..18)
        // and put descending hits on every other expert.
        let routing_ids: Vec<i64> = (10..(10 + K as i64)).collect();
        let mut hits: HashMap<u32, u64> = HashMap::new();
        hits.insert(10u32, 50); // k=0
        hits.insert(11, 0); // k=1
        hits.insert(12, 100); // k=2 (hottest)
        hits.insert(13, 0); // k=3
        hits.insert(14, 25); // k=4
        hits.insert(15, 75); // k=5
        hits.insert(16, 0); // k=6
        hits.insert(17, 10); // k=7
        let cache_aware = super::cache_aware_dispatch_order(&routing_ids, &hits);
        // Expected: 2 (100), 5 (75), 0 (50), 4 (25), 7 (10), then 1/3/6
        // tied at 0 in ascending original index order.
        assert_eq!(cache_aware, vec![2, 5, 0, 4, 7, 1, 3, 6]);

        let cache_aware_out =
            simulate_phase23(&routing_weights, &expert_outs_by_k, &cache_aware, HIDDEN);
        // Byte-identical: every f32 in the output vector matches by bit.
        for j in 0..HIDDEN {
            assert_eq!(
                baseline[j].to_bits(),
                cache_aware_out[j].to_bits(),
                "j={j} baseline={} cache_aware={} (bit-identity required)",
                baseline[j],
                cache_aware_out[j]
            );
        }

        // Even-more-permuted dispatch orders must also match. Try a
        // reverse + a couple of hand-crafted shuffles to make sure
        // Phase 3 truly fixes the accumulation order independent of
        // Phase 2's dispatch order.
        let mut reversed = router_score_order.clone();
        reversed.reverse();
        let reversed_out = simulate_phase23(&routing_weights, &expert_outs_by_k, &reversed, HIDDEN);
        for j in 0..HIDDEN {
            assert_eq!(
                baseline[j].to_bits(),
                reversed_out[j].to_bits(),
                "j={j} reversed dispatch order broke bit-identity"
            );
        }

        let weird: Vec<usize> = vec![3, 7, 1, 0, 6, 2, 5, 4];
        let weird_out = simulate_phase23(&routing_weights, &expert_outs_by_k, &weird, HIDDEN);
        for j in 0..HIDDEN {
            assert_eq!(
                baseline[j].to_bits(),
                weird_out[j].to_bits(),
                "j={j} hand-shuffled dispatch order broke bit-identity"
            );
        }
    }

    /// Same property but at the "real" K=8 with hidden=7168 (K2.6's
    /// hidden_size). This guards against any HIDDEN-dependent
    /// regression — e.g. an optimization that splits the Phase 3 sum
    /// into tiles and accidentally folds reordering into the tile
    /// boundary.
    #[test]
    fn cache_aware_dispatch_bit_identity_at_k26_hidden() {
        const K: usize = 8;
        const HIDDEN: usize = 7168;
        let routing_weights: Vec<f32> = (0..K).map(|k| 1.0 / ((k as f32 + 1.0).powi(2))).collect();
        let expert_outs_by_k: Vec<Vec<f32>> = (0..K)
            .map(|k| {
                (0..HIDDEN)
                    .map(|j| {
                        let kx = k as f32;
                        let jx = j as f32;
                        ((kx * 7.0 + jx) * 0.0011).sin() - 0.3 * (jx * 0.0007).cos()
                    })
                    .collect()
            })
            .collect();
        let identity: Vec<usize> = (0..K).collect();
        let baseline = simulate_phase23(&routing_weights, &expert_outs_by_k, &identity, HIDDEN);
        // 5! permutations is too many; sample a handful with broad
        // coverage (reverse, even-odd interleave, "hot in the middle").
        let permutations: [Vec<usize>; 4] = [
            vec![7, 6, 5, 4, 3, 2, 1, 0],
            vec![0, 2, 4, 6, 1, 3, 5, 7],
            vec![4, 3, 5, 2, 6, 1, 7, 0],
            vec![5, 0, 7, 1, 6, 3, 4, 2],
        ];
        for perm in permutations.iter() {
            let out = simulate_phase23(&routing_weights, &expert_outs_by_k, perm, HIDDEN);
            for j in 0..HIDDEN {
                assert_eq!(
                    baseline[j].to_bits(),
                    out[j].to_bits(),
                    "j={j}, perm={perm:?}: bit-identity broken"
                );
            }
        }
    }

    /// Threshold-skip (iter 007 A2) carves out specific `k` slots from
    /// dispatch. Those slots must stay zero in the `moe` accumulator
    /// regardless of dispatch order — i.e. dropping an entry from the
    /// stash is byte-equivalent to a `continue` in the pre-056 loop.
    #[test]
    fn cache_aware_dispatch_threshold_skip_byte_identical() {
        const K: usize = 8;
        const HIDDEN: usize = 64;
        let routing_weights: Vec<f32> = vec![0.5, 0.05, 0.3, 0.02, 0.4, 0.1, 0.08, 0.2];
        let expert_outs_by_k: Vec<Vec<f32>> = (0..K)
            .map(|k| {
                (0..HIDDEN)
                    .map(|j| (k as f32) * 0.1 + (j as f32) * 0.001)
                    .collect()
            })
            .collect();
        // Threshold 0.1 skips k=1 (0.05), k=3 (0.02), k=6 (0.08).
        // Active: 0, 2, 4, 5, 7.
        let active: Vec<usize> = vec![0, 2, 4, 5, 7];

        // Baseline (router-score order over active).
        let baseline = simulate_phase23(&routing_weights, &expert_outs_by_k, &active, HIDDEN);

        // Cache-aware: shuffle the active set. Phase 3 still skips
        // inactive slots, so output stays byte-identical.
        let shuffled_active: Vec<usize> = vec![5, 0, 7, 2, 4];
        let shuffled = simulate_phase23(
            &routing_weights,
            &expert_outs_by_k,
            &shuffled_active,
            HIDDEN,
        );
        for j in 0..HIDDEN {
            assert_eq!(
                baseline[j].to_bits(),
                shuffled[j].to_bits(),
                "j={j}: threshold-skipped dispatch must be bit-identical across orderings"
            );
        }
    }

    // ====================================================================
    // autolab iter 057 (async kernel scheduling — speculative prefetch) tests
    // ====================================================================
    //
    // The runner-side scheduling lives inside `forward_shells`, which
    // needs a loaded K2.6 model end-to-end. The unit tests below cover
    // the pure-helper layer:
    //
    //  (1) `speculative_prefetch_expert_ids` — picks the right top-N
    //      from a layer's hit histogram, degenerates to empty for the
    //      first-prefill (no data) case, and is deterministic under
    //      ties (so A/B prefetch streams are reproducible).
    //
    //  (2) Boundary semantics — the last-layer-in-this-rank case is
    //      handled at the call site (`next_i < n_layers`), not in the
    //      helper, so we verify the runner-loop behavior by simulating
    //      the same loop predicate the call site uses.
    //
    //  (3) Composition with iter 054 — the same `expert_hits` histogram
    //      feeds both pin-top-N and iter 057 prefetch. Verify that for
    //      an identical histogram the two helpers produce the same
    //      first-N expert id sequence (i.e. iter 057 prefetch is a
    //      no-op when the iter 054 pinned set is a superset of the
    //      iter 057 target N).

    /// Simple top-N case: clearly hot experts should come out in
    /// descending count order. Tests the basic flow the runner uses.
    #[test]
    fn speculative_prefetch_expert_ids_picks_top_n_hot_experts() {
        let mut next_layer_hits = HashMap::new();
        next_layer_hits.insert(101u32, 500u64);
        next_layer_hits.insert(202, 1000);
        next_layer_hits.insert(303, 750);
        next_layer_hits.insert(404, 250);
        next_layer_hits.insert(505, 100);
        let ids = super::speculative_prefetch_expert_ids(&next_layer_hits, 3);
        // Descending by count: 202 (1000), 303 (750), 101 (500).
        assert_eq!(ids, vec![202, 303, 101]);
    }

    /// First-prefill-token case: layer hit map is empty (no dispatches
    /// have happened yet for that layer), so the speculative scheduler
    /// must return an empty target set. This is the load-bearing
    /// no-op-on-first-token invariant — without it iter 057 would
    /// submit phantom prefetches for expert id 0 on every layer at the
    /// very start of every prompt.
    #[test]
    fn speculative_prefetch_expert_ids_empty_hits_returns_empty() {
        let empty: HashMap<u32, u64> = HashMap::new();
        assert!(super::speculative_prefetch_expert_ids(&empty, 16).is_empty());
        assert!(super::speculative_prefetch_expert_ids(&empty, 1).is_empty());
        assert!(super::speculative_prefetch_expert_ids(&empty, 384).is_empty());
    }

    /// N=0 is the "off" case at the helper level. The runner site also
    /// gates on `self.speculative_prefetch_n.is_some()`, but a caller
    /// that passes Some(0) (e.g. a test) should still get an empty set
    /// rather than every expert.
    #[test]
    fn speculative_prefetch_expert_ids_n_zero_returns_empty() {
        let mut hits = HashMap::new();
        hits.insert(1u32, 100u64);
        hits.insert(2, 200);
        assert!(super::speculative_prefetch_expert_ids(&hits, 0).is_empty());
    }

    /// Asking for more experts than the hit map has should return
    /// every distinct expert (not panic, not duplicate). On a real
    /// run the hit map will eventually contain all 384 experts; this
    /// covers the early-warm case where only ~20 have fired.
    #[test]
    fn speculative_prefetch_expert_ids_n_exceeds_distinct_returns_all() {
        let mut hits = HashMap::new();
        hits.insert(5u32, 10u64);
        hits.insert(15, 20);
        hits.insert(25, 30);
        let ids = super::speculative_prefetch_expert_ids(&hits, 100);
        // Three distinct experts, descending count: 25, 15, 5.
        assert_eq!(ids, vec![25, 15, 5]);
    }

    /// A/B reproducibility: for the same logical histogram inserted in
    /// different orders, the prefetch stream must be identical so the
    /// autolab campaign sees per-token deltas attributable to the
    /// scheduling change, not HashMap iteration order.
    #[test]
    fn speculative_prefetch_expert_ids_deterministic_across_insertion_order() {
        let mut a = HashMap::new();
        a.insert(3u32, 50u64);
        a.insert(1, 50);
        a.insert(9, 50);
        a.insert(7, 100);
        let mut b = HashMap::new();
        b.insert(7u32, 100u64);
        b.insert(9, 50);
        b.insert(3, 50);
        b.insert(1, 50);
        assert_eq!(
            super::speculative_prefetch_expert_ids(&a, 4),
            super::speculative_prefetch_expert_ids(&b, 4),
            "iter 057 prefetch stream must be insertion-order-independent"
        );
    }

    /// The K2.6 heavy-tail shape (10% of experts cover ~80% of fires)
    /// is the realistic distribution iter 057 targets. Verify the
    /// top-N helper picks exactly the hot-set head and pads with the
    /// cold tail when N > hot-set-size.
    #[test]
    fn speculative_prefetch_expert_ids_heavy_tail_picks_hot_head_then_cold_tail() {
        let mut hits = HashMap::new();
        // 38 hot experts at 100 fires each.
        for eid in 0..38u32 {
            hits.insert(eid, 100);
        }
        // 346 cold experts at 3 fires each.
        for eid in 38..384u32 {
            hits.insert(eid, 3);
        }
        // N = 16 — well inside the hot-set. Should return the first
        // 16 hot IDs (tied at 100 fires, ascending id).
        let n16 = super::speculative_prefetch_expert_ids(&hits, 16);
        assert_eq!(n16, (0..16).collect::<Vec<u32>>());
        // N = 50 — overshoots the hot-set by 12. First 38 hot IDs in
        // ascending order, then 12 cold tail IDs (also tied, ascending).
        let n50 = super::speculative_prefetch_expert_ids(&hits, 50);
        assert_eq!(n50.len(), 50);
        assert_eq!(&n50[..38], &(0..38).collect::<Vec<u32>>()[..]);
        assert_eq!(&n50[38..], &(38..50).collect::<Vec<u32>>()[..]);
    }

    /// Composition test: the iter 054 `select_top_n_by_hits` (used by
    /// `pin_top_n_per_layer`) and the iter 057 `speculative_prefetch_
    /// expert_ids` are supposed to share the same selection semantics,
    /// so that prefetching the iter 057 hot-set is a no-op when the
    /// iter 054 pinned set is a superset. Verify they produce the
    /// same expert id sequence on the same histogram for N matching.
    #[test]
    fn speculative_prefetch_matches_iter_054_pin_selection() {
        let mut hits = HashMap::new();
        // Mixed-magnitude histogram to exercise tie-break + ordering.
        for (eid, count) in [
            (10u32, 50u64),
            (20, 100),
            (30, 30),
            (40, 100),
            (50, 75),
            (60, 5),
            (70, 75),
            (80, 1),
        ] {
            hits.insert(eid, count);
        }
        for n in [1u32, 2, 4, 8, 16] {
            let pin = super::select_top_n_by_hits(&hits, n);
            let spec = super::speculative_prefetch_expert_ids(&hits, n);
            assert_eq!(
                pin, spec,
                "iter 054 pin and iter 057 prefetch must select the same top-{n} expert IDs"
            );
        }
    }

    /// Runner-loop boundary: the iter 057 call site only fires when
    /// `next_i < n_layers` so the last layer in this rank's slice
    /// has no i+1 to prefetch. Simulate that gate here so the
    /// invariant is regression-covered without needing a loaded
    /// Runner. (Cross-token coverage for the last-layer boundary is
    /// the responsibility of iter 047's `last_routing_ids` predictor,
    /// which fires at the top of the next `forward_shells` call.)
    #[test]
    fn speculative_prefetch_skips_last_layer_in_rank_slice() {
        // 4-layer rank slice. Layers 0/1/2 each prefetch the next
        // layer's hot-set; layer 3 (last) skips. Each layer's hit map
        // has its own canary expert id so we can verify which layer's
        // map was consulted.
        let mut expert_hits: Vec<HashMap<u32, u64>> = vec![HashMap::new(); 4];
        for (layer_i, &canary_eid) in [42u32, 99, 7, 200].iter().enumerate() {
            // Each layer's "hot expert" has a high count.
            expert_hits[layer_i].insert(canary_eid, 1000);
        }
        let n_layers = expert_hits.len();
        let mut submitted: Vec<(usize, Vec<u32>)> = Vec::new();
        for i in 0..n_layers {
            let next_i = i + 1;
            if next_i < n_layers {
                let eids = super::speculative_prefetch_expert_ids(&expert_hits[next_i], 1);
                submitted.push((i, eids));
            }
        }
        // Layer 0 prefetches layer 1's canary (99).
        // Layer 1 prefetches layer 2's canary (7).
        // Layer 2 prefetches layer 3's canary (200).
        // Layer 3 (last) skips.
        assert_eq!(
            submitted,
            vec![(0, vec![99]), (1, vec![7]), (2, vec![200])],
            "iter 057 must consult layer i+1's hit map, and must skip the last layer"
        );
    }

    // ====================================================================
    // autolab iter 065 (prefill-hint static schedule) tests
    // ====================================================================
    //
    // The runner-side bracket (`enter_prefill` → `forward_shells` x N →
    // `exit_prefill_and_merge_hints`) lives inside `Runner::generate`,
    // which needs a loaded K2.6 model to drive end-to-end. The unit
    // tests below cover the load-bearing pure helper
    // `merge_prefill_observations_into_hits`. The pure helper is what
    // `exit_prefill_and_merge_hints` actually delegates to, so this
    // is a faithful test of the merge math.
    //
    // The integration property the task spec demands — "prefill firing
    // layer-30 expert-42 produces a hint count change in expert_hits"
    // — is reproduced as `prefill_firing_l30_e42_produces_hint_count_change`
    // below: we synthesize an `obs[30][42] = N` observation, call the
    // merge helper at the configured weight, and assert that
    // `hits[30][42]` shifted by `round(weight * N)` while every other
    // slot stays empty.

    /// Baseline: an observation of `{layer 30 → expert 42 fires N
    /// times}` at weight 0.5 merges into `hits` as
    /// `hits[30][42] += round(0.5 * N)`. Other layers / experts stay
    /// untouched. This is the "Test:" from the iter 065 task spec.
    #[test]
    fn prefill_firing_l30_e42_produces_hint_count_change() {
        const N_LAYERS: usize = 60;
        const PREFILL_FIRINGS: u64 = 8;
        const WEIGHT: f32 = 0.5;

        let mut hits: Vec<HashMap<u32, u64>> = (0..N_LAYERS).map(|_| HashMap::new()).collect();
        let mut obs: Vec<HashMap<u32, u64>> = (0..N_LAYERS).map(|_| HashMap::new()).collect();
        obs[30].insert(42u32, PREFILL_FIRINGS);

        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, WEIGHT);

        // Exactly one (layer, expert) entry was merged.
        assert_eq!(merged, 1, "exactly one entry should merge");
        // hits[30][42] should now hold round(0.5 * 8) = 4.
        assert_eq!(
            hits[30].get(&42).copied(),
            Some(4),
            "hits[30][42] should reflect round(weight * prefill_firings)"
        );
        // Every other layer's hit map should still be empty.
        for (i, h) in hits.iter().enumerate() {
            if i == 30 {
                continue;
            }
            assert!(
                h.is_empty(),
                "layer {i} hit map should be untouched by an L30 observation"
            );
        }
        // Layer 30 should hold only expert 42.
        assert_eq!(
            hits[30].len(),
            1,
            "layer 30 should hold only the merged expert"
        );
    }

    /// Weight 0.0 must short-circuit: zero entries merged, hits map
    /// untouched even when observations are present. This is the
    /// back-compat path the CLI's default flag value relies on.
    #[test]
    fn merge_with_zero_weight_is_a_noop() {
        let mut hits = vec![HashMap::new(); 4];
        let mut obs = vec![HashMap::new(); 4];
        obs[0].insert(1u32, 100u64);
        obs[2].insert(7u32, 50u64);
        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, 0.0);
        assert_eq!(merged, 0, "weight=0.0 must merge nothing");
        for h in &hits {
            assert!(h.is_empty(), "hits map must be untouched at weight 0.0");
        }
    }

    /// Weight 1.0 is the "treat prefill as a decode firing" identity
    /// case — the iter 054 status quo, but rerouted through the hint
    /// plumbing. Verify the merged counts match prefill_obs exactly.
    #[test]
    fn merge_with_unit_weight_copies_observations_into_hits() {
        let mut hits = vec![HashMap::new(); 3];
        let mut obs = vec![HashMap::new(); 3];
        obs[0].insert(10u32, 5);
        obs[0].insert(11, 7);
        obs[2].insert(99, 12);
        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, 1.0);
        assert_eq!(merged, 3, "all three observation entries should merge");
        assert_eq!(hits[0].get(&10).copied(), Some(5));
        assert_eq!(hits[0].get(&11).copied(), Some(7));
        assert!(hits[1].is_empty());
        assert_eq!(hits[2].get(&99).copied(), Some(12));
    }

    /// The merge must ADD on top of existing hit counts (e.g. from a
    /// previous prompt's decode that left `expert_hits` warm), not
    /// overwrite them. Otherwise a fresh prefill on prompt 2 would
    /// wipe the workload-level heavy-tail data that iter 054 pinning
    /// relies on.
    #[test]
    fn merge_adds_on_top_of_existing_hits() {
        let mut hits = vec![HashMap::new(); 2];
        hits[0].insert(5u32, 100); // carryover from prior decode
        hits[1].insert(7u32, 200);
        let mut obs = vec![HashMap::new(); 2];
        obs[0].insert(5u32, 4); // same expert fired again in this prompt's prefill
        obs[1].insert(8u32, 6); // a new expert this prompt fired
        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, 0.5);
        assert_eq!(merged, 2, "two distinct observation entries should merge");
        // hits[0][5] = 100 + round(0.5 * 4) = 102
        assert_eq!(hits[0].get(&5).copied(), Some(102));
        // hits[1][7] untouched (no observation for it)
        assert_eq!(hits[1].get(&7).copied(), Some(200));
        // hits[1][8] = round(0.5 * 6) = 3 (new entry)
        assert_eq!(hits[1].get(&8).copied(), Some(3));
    }

    /// The K2.6 heavy-tail prefill shape (e.g. a code prompt that
    /// fires a handful of experts a lot, a long tail a tiny bit each)
    /// should fold into hits proportionally to its real distribution.
    /// Verify the relative ordering survives a 0.5 weight so iter 054
    /// pin / iter 056 cache-aware / iter 057 prefetch all see the
    /// same heavy-tail head after the merge.
    #[test]
    fn merge_preserves_heavy_tail_shape_relative_ordering() {
        let mut hits = vec![HashMap::new(); 1];
        let mut obs = vec![HashMap::new(); 1];
        // Hot head: experts 0..3 fire 200 times each during prefill.
        for eid in 0u32..3 {
            obs[0].insert(eid, 200);
        }
        // Warm middle: experts 10..15 fire 50 times each.
        for eid in 10u32..15 {
            obs[0].insert(eid, 50);
        }
        // Cold tail: experts 100..120 fire 4 times each.
        for eid in 100u32..120 {
            obs[0].insert(eid, 4);
        }
        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, 0.5);
        assert_eq!(merged, 3 + 5 + 20);
        // Verify relative ordering survives. Hot > warm > cold.
        for hot in 0u32..3 {
            assert_eq!(hits[0].get(&hot).copied(), Some(100));
        }
        for warm in 10u32..15 {
            assert_eq!(hits[0].get(&warm).copied(), Some(25));
        }
        for cold in 100u32..120 {
            assert_eq!(hits[0].get(&cold).copied(), Some(2));
        }
        // And the iter 054 / 056 / 057 helpers must rank them correctly.
        let top3 = super::select_top_n_by_hits(&hits[0], 3);
        assert_eq!(top3, vec![0, 1, 2], "top-3 must be the hot head");
        let top8 = super::select_top_n_by_hits(&hits[0], 8);
        assert_eq!(
            top8,
            vec![0, 1, 2, 10, 11, 12, 13, 14],
            "top-8 must drop into the warm middle after exhausting the hot head"
        );
    }

    /// Empty observations (the first call ever, or after `reset_kv`)
    /// must merge zero entries regardless of weight. Guards against
    /// "did we accidentally insert phantom expert 0 entries?".
    #[test]
    fn merge_with_empty_observations_merges_nothing() {
        let mut hits = vec![HashMap::new(); 10];
        let obs = vec![HashMap::new(); 10];
        for w in [0.5_f32, 1.0, 2.0, 10.0] {
            let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, w);
            assert_eq!(merged, 0, "empty obs must merge zero entries at w={w}");
            for h in &hits {
                assert!(h.is_empty(), "hits must stay empty when obs is empty");
            }
        }
    }

    /// Non-finite / negative weights must short-circuit to a no-op,
    /// mirroring `set_prefill_hint_weight`'s clamping. A NaN that
    /// snuck through the API setter (or a future code path that
    /// doesn't call the setter) must not panic the merge.
    #[test]
    fn merge_with_invalid_weight_is_a_noop() {
        let mut hits = vec![HashMap::new(); 2];
        let mut obs = vec![HashMap::new(); 2];
        obs[0].insert(1u32, 100);
        for bad_w in [-1.0_f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, bad_w);
            assert_eq!(merged, 0, "weight={bad_w:?} must merge nothing");
            for h in &hits {
                assert!(
                    h.is_empty(),
                    "hits must stay untouched on invalid weight={bad_w:?}"
                );
            }
        }
    }

    /// Round-to-nearest: `round(0.5 * 1) = 1` (banker's rounding would
    /// give 0; we want 1 so a single prefill firing always leaves a
    /// trace at any positive weight ≥ 0.5). Without this, a w=0.5
    /// hint would silently drop every odd-count observation.
    #[test]
    fn merge_rounds_to_nearest_not_truncate() {
        let mut hits = vec![HashMap::new(); 1];
        let mut obs = vec![HashMap::new(); 1];
        // 1 firing at w=0.5 → 0.5 → round to 1 (round-half-away).
        obs[0].insert(7u32, 1);
        // 3 firings at w=0.5 → 1.5 → round to 2.
        obs[0].insert(8u32, 3);
        // 4 firings at w=0.5 → 2.0 → round to 2.
        obs[0].insert(9u32, 4);
        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, 0.5);
        assert_eq!(merged, 3);
        assert_eq!(hits[0].get(&7).copied(), Some(1), "1*0.5 should round to 1");
        assert_eq!(hits[0].get(&8).copied(), Some(2), "3*0.5 should round to 2");
        assert_eq!(hits[0].get(&9).copied(), Some(2), "4*0.5 should round to 2");
    }

    /// Sub-rounding firings (e.g. 1 firing at w=0.1 = 0.1 ≈ rounds to
    /// 0) must NOT be counted in the returned `merged` total and must
    /// NOT insert a zero entry into `hits`. Otherwise the helper would
    /// silently bloat the hits map with no-op entries that perturb
    /// downstream iter 054 / 056 / 057 selection.
    #[test]
    fn merge_skips_sub_rounding_contributions() {
        let mut hits = vec![HashMap::new(); 1];
        let mut obs = vec![HashMap::new(); 1];
        // 1 firing at w=0.1 → 0.1 → rounds to 0 → skip.
        obs[0].insert(1u32, 1);
        // 2 firings at w=0.1 → 0.2 → rounds to 0 → skip.
        obs[0].insert(2u32, 2);
        // 5 firings at w=0.1 → 0.5 → rounds to 1 (round-half-away).
        obs[0].insert(3u32, 5);
        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, 0.1);
        assert_eq!(
            merged, 1,
            "only the expert with >= round-up contribution should count"
        );
        assert!(!hits[0].contains_key(&1), "expert 1 should NOT be inserted");
        assert!(!hits[0].contains_key(&2), "expert 2 should NOT be inserted");
        assert_eq!(hits[0].get(&3).copied(), Some(1), "expert 3 should be 1");
        assert_eq!(hits[0].len(), 1, "no phantom zero entries");
    }

    /// Length-mismatched inputs should not panic — short observations
    /// vs longer hits is fine (extra hit slots stay untouched); short
    /// hits vs longer observations short-circuits the extra obs slots
    /// because `hits.get_mut(i)` returns None.
    #[test]
    fn merge_tolerates_length_mismatch_without_panic() {
        // hits shorter than obs: extra obs entries are skipped.
        let mut hits = vec![HashMap::new(); 2];
        let mut obs = vec![HashMap::new(); 5];
        obs[0].insert(1u32, 4);
        obs[3].insert(7u32, 4); // out-of-bounds for hits
        let merged = super::merge_prefill_observations_into_hits(&mut hits, &obs, 0.5);
        // Only obs[0] could merge.
        assert_eq!(merged, 1);
        assert_eq!(hits[0].get(&1).copied(), Some(2));
        assert!(hits[1].is_empty());
    }
}
