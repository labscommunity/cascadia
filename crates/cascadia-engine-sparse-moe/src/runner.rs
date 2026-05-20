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

use cascadia_int4_gemm::layer0_int4::{
    embed_token_bf16, layer0_forward_decode_int4_multi_with_capacity,
    layer0_forward_decode_int4_with_capacity, Int4Layer0,
};
use cascadia_int4_gemm::safetensors_source::Shard;
use cascadia_int4_gemm::shell::{
    HIDDEN as SHELL_HIDDEN, NUM_HEADS, QK_HEAD_DIM, TOPK as SHELL_TOPK, V_HEAD_DIM,
};
use cascadia_int4_gemm::shell_int4::{
    shell_forward_decode_int4_multi_with_capacity, shell_forward_decode_int4_with_capacity,
    Int4Shell,
};
use cascadia_int4_gemm::{
    expert_forward as int4_expert_forward, f32_to_bf16_bits, ExpertWeights, SafetensorsExpert,
    SafetensorsExpertSource,
};
use cascadia_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};
use half::bf16;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::kv_prefix_cache::{KvPrefixCache, KvSnapshot, LayerKvSlice, ModelFingerprint};
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
/// The shell forward runs through `cascadia_int4_gemm::shell_int4::shell_forward_decode_int4_with_capacity`,
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
/// mmap'd flat int4 binaries served by the cascadia-int4-gemm AVX-512
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
        // 4096 slots = ~22 tokens of 6 experts × 30 layers at K=6.
        // Plenty of headroom; if we overrun this we're either way ahead
        // of the consumer or producing more prefetches than we should.
        let (tx, rx) = mpsc::sync_channel::<PrefetchReq>(4096);
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
    /// If `Some(k')` with k' < manifest.top_k, forward_shells dispatches
    /// only the first k' of the routed top-K experts per token. See
    /// `docs/A3_TOPK_REDUCTION.md` for the productionization Pareto.
    top_k_override: Option<u32>,
    /// If `Some(t)` with t > 0, skip experts whose routing weight is
    /// below t. Applied AFTER `top_k_override`.
    routing_threshold: Option<f32>,
    /// autolab iter 029 (C1): cache of the *previous* token's routed
    /// expert IDs per layer (indexed by position in `self.layers`, not
    /// by `lid`). Empty `Vec<u32>` means "no history yet" (just after
    /// `reset_kv` or first prefill token). Used as a simple
    /// same-as-last-token predictor: at the start of each
    /// `forward_shells` we push these IDs to the prefetcher so the OS
    /// can start pulling pages while this token's earlier layers run.
    last_routing_ids: Vec<Vec<u32>>,
    /// autolab iter 029 (C1): background prefetcher fed by
    /// `last_routing_ids` at the start of each `forward_shells`. `None`
    /// when prefetching is disabled (env var `CASCADIA_EXPERT_PREFETCH=0`
    /// or experts_format != safetensors_bin).
    prefetcher: Option<Prefetcher>,
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
        // CASCADIA_EXPERT_PREFETCH=0 so it's easy to A/B at runtime. Other
        // expert backends (ov_ir, int4_bin) don't benefit from madvise
        // because their weights aren't served from the safetensors mmaps,
        // so the prefetcher would do nothing useful there.
        let prefetcher = match (
            &experts,
            std::env::var("CASCADIA_EXPERT_PREFETCH").as_deref(),
        ) {
            (ExpertCache::SafetensorsBin(_), Ok("0")) => {
                info!("expert prefetch: disabled via CASCADIA_EXPERT_PREFETCH=0");
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
        })
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

    /// Set the per-token routing-weight threshold. None / 0.0 = disabled.
    pub fn set_routing_threshold(&mut self, v: Option<f32>) {
        self.routing_threshold = v.filter(|t| *t > 0.0);
        info!(routing_threshold = ?self.routing_threshold, "set_routing_threshold");
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
    }

    /// Truncate the per-layer KV caches by `n` slots. Used by speculative
    /// decoding to discard rejected-draft positions without paying the cost
    /// of resetting + replaying the kept prefix.
    ///
    /// **Equivalence to mask-based rewind.** rainier's `MaskedReq.rewind(k)`
    /// (see `docs/SPEC_DECODE_SUMMARY.md`) leaves K/V values physically in
    /// the OV stateful cache and flips an `attention_mask[j] = 0` flag so
    /// future queries skip those positions. We don't have an OV-stateful
    /// cache here — the int4 shell's K/V is a raw f32 buffer that we
    /// already address via `past_seq_len`. So we can do the simpler
    /// in-place truncation: shrink `past_seq_len` by `n`. The next
    /// `forward_shells` call writes its new slot at the now-vacant
    /// position; the rejected K/V values become dead memory until
    /// overwritten or the cache is reset.
    ///
    /// This is symmetric across layer 0 and every shell layer this rank
    /// owns. Saturating on `n > past_seq_len` clamps to 0 (equivalent
    /// to `reset_kv` for that layer).
    pub fn rewind_kv(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        for l in &mut self.layers {
            l.past_seq_len = l.past_seq_len.saturating_sub(n);
        }
        if let Some(l0) = self.layer0.as_mut() {
            l0.past_seq_len = l0.past_seq_len.saturating_sub(n);
        }
    }

    /// Snapshot of the current KV-cache lengths across the rank's
    /// layers (layer 0 + every MoE layer this rank owns). All values
    /// should be equal during normal operation; if they diverge that
    /// indicates a bug in the spec-decode rewind path. Public for tests
    /// and for the spec-decode loop's debug assertions.
    pub fn kv_past_seq_lens(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.layers.len() + 1);
        if let Some(l0) = self.layer0.as_ref() {
            out.push(l0.past_seq_len);
        }
        for l in &self.layers {
            out.push(l.past_seq_len);
        }
        out
    }

    /// Estimated bytes the KV-prefix cache would consume per cached
    /// **token**, summed across every layer the rank owns (layer 0
    /// when `is_first`, every MoE shell). Multiply by the typical
    /// prompt length to budget a [`KvPrefixCache`] capacity.
    ///
    /// Formula: `kv_layers * NUM_HEADS * (qk_head_dim + v_head_dim) * 2`
    /// — bf16 storage matches the runner's live KV layout. Excludes
    /// the IndexMap key + hash overhead, which is negligible next to
    /// the KV tensors at K2.6 dims (~150 MiB / 512-token snapshot).
    pub fn estimated_snapshot_bytes_per_token(&self) -> usize {
        let kv_layers = self.layers.len() + if self.layer0.is_some() { 1 } else { 0 };
        let per_head_dim = self.manifest.qk_head_dim as usize + self.manifest.v_head_dim as usize;
        kv_layers * NUM_HEADS * per_head_dim * std::mem::size_of::<u16>()
    }

    /// Build a [`ModelFingerprint`] from the manifest + the rank's
    /// `LayerRange`. Used as part of the [`KvPrefixCache`] key — see
    /// the cache module docs for the "what affects KV bits" rule.
    pub fn fingerprint(&self) -> ModelFingerprint {
        ModelFingerprint {
            arch: self.manifest.arch.clone(),
            num_layers: self.manifest.num_layers,
            num_experts: self.manifest.num_experts,
            top_k: self.manifest.top_k,
            hidden_size: self.manifest.hidden_size,
            num_kv_heads: self.manifest.num_kv_heads,
            qk_head_dim: self.manifest.qk_head_dim,
            v_head_dim: self.manifest.v_head_dim,
            vocab_size: self.manifest.vocab_size,
            layer_start: self.range.layer_start,
            layer_end: self.range.layer_end,
            is_first: self.range.is_first,
            is_last: self.range.is_last,
        }
    }

    /// Snapshot the rank's current KV state into a [`KvSnapshot`].
    /// Captures only the populated `0..past_seq_len` slots per head —
    /// the live capacity buffers are NOT serialized, so a 32-capacity
    /// runner with 8 slots filled produces an 8-slot snapshot.
    ///
    /// Every layer (layer 0 if owned + every MoE shell) must agree on
    /// `past_seq_len`; mismatches return an `Internal` error since they
    /// indicate a bug in the engine's prefill driver, not a recoverable
    /// state. (Compare `kv_past_seq_lens` on `main` post-spec-decode.)
    pub fn snapshot_kv(&self) -> Result<KvSnapshot, RunnerError> {
        let ps = if let Some(l0) = self.layer0.as_ref() {
            l0.past_seq_len
        } else if let Some(first) = self.layers.first() {
            first.past_seq_len
        } else {
            return Err(RunnerError::Internal(
                "snapshot_kv: rank holds neither layer 0 nor any shell".into(),
            ));
        };
        if let Some(l0) = self.layer0.as_ref() {
            if l0.past_seq_len != ps {
                return Err(RunnerError::Internal(format!(
                    "snapshot_kv: layer-0 past_seq_len {} != expected {}",
                    l0.past_seq_len, ps
                )));
            }
        }
        for l in &self.layers {
            if l.past_seq_len != ps {
                return Err(RunnerError::Internal(format!(
                    "snapshot_kv: layer {} past_seq_len {} != expected {}",
                    l.lid, l.past_seq_len, ps
                )));
            }
        }
        let layer0 = self
            .layer0
            .as_ref()
            .map(|l0| pack_layer_slice(0, &l0.past_k, &l0.past_v, ps, l0.kv_capacity));
        let shells = self
            .layers
            .iter()
            .map(|l| pack_layer_slice(l.lid, &l.past_k, &l.past_v, ps, l.kv_capacity))
            .collect();
        Ok(KvSnapshot {
            past_seq_len: ps,
            num_heads: NUM_HEADS as u32,
            qk_head_dim: QK_HEAD_DIM as u32,
            v_head_dim: V_HEAD_DIM as u32,
            layer0,
            shells,
        })
    }

    /// Restore a previously-captured snapshot into the runner's KV
    /// state. Validates shape against the rank's live layout — wrong
    /// `num_heads` or `head_dim` is a hard error (the fingerprint check
    /// upstream should have caught it, but defence in depth).
    ///
    /// After this returns Ok, every layer's `past_seq_len` is set to
    /// the snapshot's value and the populated prefix is bit-identical
    /// to what `forward_shells` would have written. The next forward
    /// step writes its new slot at offset `past_seq_len`, exactly as
    /// it would after a fresh prefill.
    ///
    /// Grows capacity buffers if needed (a snapshot from a long prompt
    /// restored into a freshly-loaded runner with default 32-cap).
    pub fn restore_kv(&mut self, snap: &KvSnapshot) -> Result<(), RunnerError> {
        if snap.num_heads as usize != NUM_HEADS
            || snap.qk_head_dim as usize != QK_HEAD_DIM
            || snap.v_head_dim as usize != V_HEAD_DIM
        {
            return Err(RunnerError::Internal(format!(
                "restore_kv: snapshot shape ({}H, {}D_qk, {}D_v) != runner ({}H, {}D_qk, {}D_v)",
                snap.num_heads,
                snap.qk_head_dim,
                snap.v_head_dim,
                NUM_HEADS,
                QK_HEAD_DIM,
                V_HEAD_DIM
            )));
        }
        let ps = snap.past_seq_len;
        let layer0_present = self.layer0.is_some();
        let snap_layer0_present = snap.layer0.is_some();
        if layer0_present != snap_layer0_present {
            return Err(RunnerError::Internal(format!(
                "restore_kv: layer-0 presence mismatch (runner={}, snapshot={})",
                layer0_present, snap_layer0_present
            )));
        }
        if snap.shells.len() != self.layers.len() {
            return Err(RunnerError::Internal(format!(
                "restore_kv: shell count mismatch (runner={}, snapshot={})",
                self.layers.len(),
                snap.shells.len()
            )));
        }
        for (l, s) in self.layers.iter().zip(snap.shells.iter()) {
            if l.lid != s.lid {
                return Err(RunnerError::Internal(format!(
                    "restore_kv: layer-id mismatch at position (runner={}, snapshot={})",
                    l.lid, s.lid
                )));
            }
        }

        if let (Some(l0), Some(snap_l0)) = (self.layer0.as_mut(), snap.layer0.as_ref()) {
            while ps > l0.kv_capacity {
                grow_layer0_kv_capacity(l0)?;
            }
            unpack_layer_slice(&mut l0.past_k, &mut l0.past_v, snap_l0, ps, l0.kv_capacity)?;
            l0.past_seq_len = ps;
        }
        for (l, snap_l) in self.layers.iter_mut().zip(snap.shells.iter()) {
            while ps > l.kv_capacity {
                grow_kv_capacity(l)?;
            }
            unpack_layer_slice(&mut l.past_k, &mut l.past_v, snap_l, ps, l.kv_capacity)?;
            l.past_seq_len = ps;
        }
        Ok(())
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
        // Per-token effective top-K. The shell router still returns the full
        // manifest top-K of routing_ids/weights; we skip dispatches based on
        // the override (fixed K') and the threshold (sigmoid weight gate).
        let effective_top_k = self
            .top_k_override
            .map(|v| (v as usize).min(manifest_top_k))
            .unwrap_or(manifest_top_k);
        let threshold = self.routing_threshold.unwrap_or(0.0);
        let top_k = manifest_top_k; // for the router contract check below
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
            // the same Cargo workspace. The `_with_capacity` variant
            // lets us pass a pre-allocated [H, capacity, D] buffer
            // with only the first `past_seq_len` slots populated.
            let shell_t0 = Instant::now();
            let outs = shell_forward_decode_int4_with_capacity(
                &self.layers[i].int4_shell,
                &h_f32,
                &self.layers[i].past_k,
                &self.layers[i].past_v,
                past_seq_len,
                capacity,
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
            let experts_t0 = Instant::now();
            let mut moe = vec![0.0f32; hidden];
            // autolab iter 029 (C1): collect this layer's actually-fired
            // expert IDs so the next forward_shells call can use them as
            // its prefetch predictor. Same-as-last-token heuristic.
            let mut this_token_ids: Vec<u32> = Vec::with_capacity(effective_top_k);
            for k in 0..effective_top_k {
                let w = outs.routing_weights[k];
                if w < threshold {
                    continue;
                }
                let eid = outs.routing_ids[k] as u32;
                this_token_ids.push(eid);
                let y_f32 = self.dispatch_expert(lid, eid, &outs.attn_out_post_norm)?;
                for j in 0..hidden {
                    moe[j] += w * y_f32[j];
                }
            }
            // Stash the IDs for the next token's prefetch. We do this
            // regardless of whether prefetching is currently enabled;
            // tracking it costs ~k*u32 per layer per token (negligible)
            // and makes it cheaper to toggle the prefetcher mid-run.
            self.last_routing_ids[i] = this_token_ids;
            experts_total_us += experts_t0.elapsed().as_micros() as u64;

            // Combine: h_next = residual + shared + moe (single token).
            let combine_t0 = Instant::now();
            for j in 0..hidden {
                h_f32[j] = outs.attn_residual[j] + outs.shared_expert_out[j] + moe[j];
            }
            combine_total_us += combine_t0.elapsed().as_micros() as u64;
        }

        // autolab/k26-perf q1 instrumentation: per-token shells breakdown.
        // iter 029 (C1): also log prefetch counters so we can see how the
        // submit/drop ratio evolves across a generation. Counters are
        // cumulative-since-Runner-load, so the deltas across consecutive
        // tokens tell us submits-per-token and drops-per-token.
        let (pf_submits, pf_drops, pf_processed) = self
            .prefetcher
            .as_ref()
            .map(|p| p.snapshot())
            .unwrap_or((0, 0, 0));
        info!(
            stage = "shells",
            n_layers,
            top_k,
            effective_top_k,
            shell_attn_us = shell_attn_total_us,
            experts_us = experts_total_us,
            combine_us = combine_total_us,
            prefetch_submitted_this_call = prefetch_submitted,
            prefetch_total_submits = pf_submits,
            prefetch_total_drops = pf_drops,
            prefetch_total_processed = pf_processed,
            total_us = _t0.elapsed().as_micros() as u64,
            "stage_timing"
        );
        Ok(h_f32)
    }

    // NOTE: iter 050 also added a `forward_shells_multi` variant with
    // a redundant `h_shape: &[usize]` validation parameter and no
    // per-token threshold / top-K-override support. That variant was
    // superseded by the iter 044 variant defined further down (single
    // source of truth, adds the A2 routing-threshold filter + A3
    // top-K override + full per-stage timing the spec-decode driver
    // depends on). The iter 050 variant has been removed from this
    // spinout to avoid the duplicate-definition error.
    //
    // See the iter 044 `forward_shells_multi` below (the dispatcher
    // that `step_multi` calls).

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

    /// Multi-token layer-0 step. Advances layer-0 KV by `seq` slots and
    /// returns the layer-0 hidden output for all `seq` tokens
    /// concatenated as `[seq, HIDDEN]` flat.
    ///
    /// Calls
    /// [`layer0_forward_decode_int4_multi_with_capacity`]
    /// once instead of `seq` individual `forward_layer0_step` calls. As
    /// of iter 042 layer 0's multi path is still a scalar loop (no tiled
    /// kernel), so the win at this seam is marginal — the speedup is
    /// concentrated in the shell's multi path. We still wire it through
    /// here so future tiled-GEMM work on layer 0 hooks in for free.
    pub fn forward_layer0_multi(
        &mut self,
        token_ids: &[i64],
        seq: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let _t0 = Instant::now();
        let l0 = self.layer0.as_mut().ok_or_else(|| {
            RunnerError::Internal("forward_layer0_multi on non-first stage".into())
        })?;
        assert!(seq >= 1, "forward_layer0_multi: seq must be >= 1");
        assert_eq!(token_ids.len(), seq);

        // Grow KV capacity if the next `seq` slots would overflow. Single
        // grow loop instead of one per token — the geometric grow guarantees
        // O(log) total grows over a generation, but a multi-token call could
        // straddle a grow boundary so we may need >1 grow here.
        while l0.past_seq_len + seq > l0.kv_capacity {
            grow_layer0_kv_capacity(l0)?;
        }
        let capacity = l0.kv_capacity;
        let past_seq_len = l0.past_seq_len;

        // Build `[seq, HIDDEN]` flat embed input.
        let mut xs_f32 = vec![0.0f32; seq * SHELL_HIDDEN];
        for (t, &id) in token_ids.iter().enumerate() {
            let row = embed_token_bf16(l0.embed_tokens_bf16, id);
            xs_f32[t * SHELL_HIDDEN..(t + 1) * SHELL_HIDDEN].copy_from_slice(&row);
        }

        let outs = layer0_forward_decode_int4_multi_with_capacity(
            &l0.int4_layer0,
            &xs_f32,
            &mut l0.past_k,
            &mut l0.past_v,
            past_seq_len,
            capacity,
            seq,
        );
        l0.past_seq_len = past_seq_len + seq;

        info!(
            stage = "layer0_multi",
            duration_us = _t0.elapsed().as_micros() as u64,
            seq,
            past_seq_len,
            "stage_timing"
        );
        Ok(outs.hidden_out)
    }

    /// Multi-token shell forward over this rank's MoE layers. For each
    /// layer this rank owns: one
    /// [`shell_forward_decode_int4_multi_with_capacity`] call (the tiled
    /// AVX-VNNI multi path from iter 042) followed by per-token expert
    /// dispatch + combine.
    ///
    /// Bit-identical to `forward_shells` invoked `seq` times sequentially
    /// — the shell's `_multi` path is itself bit-identical to the scalar
    /// loop (per `multi_seq_3_matches_sequential_seq_1_calls` in
    /// `shell_int4`), and we still apply A2/A3 routing-threshold + top-K
    /// override identically per token.
    ///
    /// Input `h_in` is `[seq, HIDDEN]` flat. Returns `[seq, HIDDEN]` flat.
    pub fn forward_shells_multi(
        &mut self,
        h_in: &[f32],
        past_seq_len: usize,
        seq: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let _t0 = Instant::now();
        let mut shell_attn_total_us: u64 = 0;
        let mut experts_total_us: u64 = 0;
        let mut combine_total_us: u64 = 0;
        let hidden = self.manifest.hidden_size as usize;
        let manifest_top_k = self.manifest.top_k as usize;
        let effective_top_k = self
            .top_k_override
            .map(|v| (v as usize).min(manifest_top_k))
            .unwrap_or(manifest_top_k);
        let top_k = manifest_top_k;
        assert!(seq >= 1, "forward_shells_multi: seq must be >= 1");
        if h_in.len() != seq * hidden {
            return Err(RunnerError::Internal(format!(
                "forward_shells_multi: h_in.len={} != seq*hidden={}",
                h_in.len(),
                seq * hidden
            )));
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

            // Grow KV capacity to fit the next `seq` slots. May need
            // multiple grows when straddling a doubling boundary.
            while past_seq_len + seq > self.layers[i].kv_capacity {
                grow_kv_capacity(&mut self.layers[i])?;
            }
            let capacity = self.layers[i].kv_capacity;

            // One multi-token shell forward — this is the iter 042 win.
            // For seq=1 it dispatches to the scalar path; for seq>=2 the
            // batched/tiled-GEMM path amortizes weight loads across tokens.
            //
            // Borrow-checker dance: `_multi_with_capacity` takes
            // `&mut past_k/past_v` while also taking `&shell`. We split
            // the `LayerState` mutable borrow by destructuring through a
            // temporary `&mut LayerState` so the shell field can be
            // re-borrowed immutably while the K/V fields are borrowed
            // mutably.
            let shell_t0 = Instant::now();
            let layer = &mut self.layers[i];
            let outs = shell_forward_decode_int4_multi_with_capacity(
                &layer.int4_shell,
                &h_f32,
                &mut layer.past_k,
                &mut layer.past_v,
                past_seq_len,
                capacity,
                seq,
            );
            shell_attn_total_us += shell_t0.elapsed().as_micros() as u64;
            self.layers[i].past_seq_len = past_seq_len + seq;

            // Per-token expert dispatch + combine. Same A2 threshold +
            // A3 top-K override logic as `forward_shells`, just iterated
            // across `seq` tokens. Expert dispatch cannot batch across
            // tokens — different tokens route to different experts.
            let threshold = self.routing_threshold.unwrap_or(0.0);
            let experts_t0 = Instant::now();
            for t in 0..seq {
                let ids = &outs.routing_ids[t * SHELL_TOPK..(t + 1) * SHELL_TOPK];
                let ws = &outs.routing_weights[t * SHELL_TOPK..(t + 1) * SHELL_TOPK];
                if ids.len() != top_k || ws.len() != top_k {
                    return Err(RunnerError::Internal(format!(
                        "L{lid} t{t} routing shape unexpected: ids={} weights={} (top_k={})",
                        ids.len(),
                        ws.len(),
                        top_k
                    )));
                }
                let attn_t = &outs.attn_out_post_norm[t * hidden..(t + 1) * hidden];
                let mut moe = vec![0.0f32; hidden];
                for k in 0..effective_top_k {
                    let w = ws[k];
                    if w < threshold {
                        continue;
                    }
                    let eid = ids[k] as u32;
                    let y_f32 = self.dispatch_expert(lid, eid, attn_t)?;
                    for j in 0..hidden {
                        moe[j] += w * y_f32[j];
                    }
                }

                // Combine: h_next = residual + shared + moe (per token).
                let combine_t0 = Instant::now();
                let residual_t = &outs.attn_residual[t * hidden..(t + 1) * hidden];
                let shared_t = &outs.shared_expert_out[t * hidden..(t + 1) * hidden];
                let h_dst = &mut h_f32[t * hidden..(t + 1) * hidden];
                for j in 0..hidden {
                    h_dst[j] = residual_t[j] + shared_t[j] + moe[j];
                }
                combine_total_us += combine_t0.elapsed().as_micros() as u64;
            }
            experts_total_us += experts_t0.elapsed().as_micros() as u64;
        }

        info!(
            stage = "shells_multi",
            n_layers,
            seq,
            top_k,
            effective_top_k,
            shell_attn_us = shell_attn_total_us,
            experts_us = experts_total_us,
            combine_us = combine_total_us,
            total_us = _t0.elapsed().as_micros() as u64,
            "stage_timing"
        );
        Ok(h_f32)
    }

    /// Run the head over every position of a `[seq, HIDDEN]` flat input,
    /// returning `seq` vocab-sized logit rows.
    ///
    /// The head IR runs at `[1, 1, HIDDEN]` shape per the existing
    /// `forward_head_last`; running multi-position means K sequential
    /// invocations. At K=4 this is ~25 ms × 4 = 100 ms per round, a
    /// rounding error vs the shell forward's ~7-9 s — but it is fully
    /// per-token serial, no batching unlock. Future work: a true multi-
    /// position head IR.
    pub fn forward_head_multi(
        &mut self,
        h_f32: &[f32],
        seq: usize,
    ) -> Result<Vec<Vec<f32>>, RunnerError> {
        let hidden = self.manifest.hidden_size as usize;
        if h_f32.len() < seq * hidden {
            return Err(RunnerError::Internal(format!(
                "forward_head_multi: h.len={} < seq*hidden={}",
                h_f32.len(),
                seq * hidden
            )));
        }
        let mut out = Vec::with_capacity(seq);
        for t in 0..seq {
            // Reuse `forward_head_last` by handing it a single-position
            // slice — same code path, no special-casing needed.
            let slice = &h_f32[t * hidden..(t + 1) * hidden];
            let logits = self.forward_head_last(slice, 1)?;
            out.push(logits);
        }
        Ok(out)
    }

    /// Multi-token forward pass equivalent to calling [`Self::step`]
    /// `tail_len` times sequentially.
    ///
    /// - `full_ids`: full prefix-so-far. The last `tail_len` entries are
    ///   the tokens we will advance KV through.
    /// - Returns `tail_len` vocab-sized logit rows, one per advanced
    ///   token. `logits[i]` predicts the token AFTER position
    ///   `past_seq_len + i`, where `past_seq_len = full_ids.len -
    ///   tail_len`.
    ///
    /// **Use case.** Speculative decode's K-token verify pass. The
    /// iter 042 multi-token shell GEMM amortizes weight loads across
    /// K tokens, giving the per-projection speedup the entire stack
    /// was built for. For `tail_len == 1` this is bit-identical to
    /// (and roughly the same cost as) `step`; for `tail_len >= 2` it
    /// is meaningfully faster.
    ///
    /// First/single stage only — no pipeline parallelism here; the
    /// pipeline-parallel multi-token path lives in
    /// [`crate::engine::SparseMoEEngine::step_first`].
    pub fn step_multi(
        &mut self,
        full_ids: &[i64],
        tail_len: usize,
    ) -> Result<Vec<Vec<f32>>, RunnerError> {
        if tail_len == 0 || tail_len > full_ids.len() {
            return Err(RunnerError::Internal(format!(
                "step_multi: invalid tail_len {}, full_ids.len={}",
                tail_len,
                full_ids.len()
            )));
        }
        let past_seq_len = full_ids.len() - tail_len;
        let tail = &full_ids[past_seq_len..];

        // 1) Layer 0 over the tail.
        let h_f32 = self.forward_layer0_multi(tail, tail_len)?;

        // 2) Shells (this is where iter 042's tiled GEMM kicks in).
        let h_f32 = self.forward_shells_multi(&h_f32, past_seq_len, tail_len)?;

        // 3) Head per position.
        self.forward_head_multi(&h_f32, tail_len)
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
        self.generate_with_cache(prompt_ids, max_tokens, cfg, None)
    }

    /// Generate with an optional KV-prefix cache. Behaviour is byte-
    /// identical to `generate` when `cache` is `None` or empty. On a
    /// hit, the cached snapshot is restored into the runner's KV
    /// buffers (skipping the matched prefill tokens) and prefill
    /// proceeds for the remaining suffix tokens only.
    ///
    /// On a miss, the full prefill runs as normal and — if the suffix
    /// after the system-prompt boundary heuristic crosses the
    /// `min_cache_prefix` threshold — a snapshot is inserted into the
    /// cache after the prefill completes. The cache is keyed on the
    /// **full prompt token sequence**; common-prefix matching is
    /// O(entries) at lookup time, not at insert time. See the
    /// [`crate::kv_prefix_cache`] module docs for the cache semantics.
    pub fn generate_with_cache(
        &mut self,
        prompt_ids: &[i64],
        max_tokens: usize,
        cfg: &crate::sampling::SamplingConfig,
        mut cache: Option<&mut KvPrefixCache>,
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

        // Cache lookup. Returns Some(matched_len) if we restored a
        // snapshot; None otherwise. `fp` is only computed when the
        // cache is actually present + enabled so the `generate` →
        // `generate_with_cache(None)` path stays allocation-identical
        // to the pre-cache `generate` (preserves the strict
        // "default off = byte-identical" invariant called out in the
        // PR description and CLI flag docs).
        let mut fp: Option<ModelFingerprint> = None;
        let mut cache_skip: usize = 0;
        let mut cache_hit = false;
        if let Some(c) = cache.as_mut() {
            if c.enabled() {
                let fingerprint = fp.insert(self.fingerprint());
                if let Some(snap) = c.lookup(prompt_ids, fingerprint) {
                    // Restore into the runner's KV buffers. The
                    // restore validates shape against fingerprint;
                    // a failure here means cache + runner disagree on
                    // dimensions (should never happen — fingerprint
                    // covers num_heads/head_dim — but bubble up the
                    // error rather than silently corrupt).
                    cache_skip = snap.past_seq_len;
                    self.restore_kv(&snap)?;
                    cache_hit = true;
                    info!(
                        cached_prefix_len = cache_skip,
                        full_prompt_len = prompt_ids.len(),
                        "kv-prefix-cache HIT — skipping prefix tokens"
                    );
                }
            }
        }

        // Prefill token-by-token to keep shell input shapes uniform (avoids
        // the OV 2026.1.0 CPU snippets shape-specialization bug we hit on
        // shape changes). On a cache hit, `cache_skip` of the prompt
        // is already in KV; we still push those tokens into `history`
        // so the shells' `past_seq_len` accounting agrees with our
        // bookkeeping. We only execute `step()` on the suffix.
        info!(
            prompt_len = prompt_ids.len(),
            cache_skip, "prefill (token-by-token)"
        );
        let mut history: Vec<i64> = Vec::with_capacity(prompt_ids.len() + max_tokens);
        let mut last_logits: Option<Vec<f32>> = None;
        let t_pre = Instant::now();
        for (i, &t) in prompt_ids.iter().enumerate() {
            history.push(t);
            if i < cache_skip {
                // Token's KV bits are already in the cache restore.
                // Skip the forward — but DO NOT skip the last one
                // (the suffix's first token), because we still need
                // a logits row from the LAST prefill step to sample
                // the first generated token from. The `lookup`
                // contract enforces `cache_skip < prompt.len()`, so
                // there is always at least one suffix token below.
                continue;
            }
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
        let suffix_len = prompt_ids.len().saturating_sub(cache_skip);
        info!(
            secs = prefill_secs,
            suffix_len,
            tok_per_s = if prefill_secs > 0.0 {
                suffix_len as f64 / prefill_secs
            } else {
                0.0
            },
            "prefill done"
        );

        // Insert a snapshot on miss. We only cache the FULL prompt
        // (not arbitrary intermediate prefixes) — this keeps the
        // cache small and matches the chat-completion access pattern
        // where the system+user prompt is stable across requests but
        // the user message varies. Insert happens after prefill so
        // the snapshot reflects what `forward_shells` actually wrote.
        //
        // We could be cleverer (cache after each step, or at
        // configurable boundaries) — defer until profiling shows
        // demand. The dominant cost on the rainier baseline is
        // prefill, and a full-prompt snapshot pays it off in one hit.
        if !cache_hit {
            if let Some(c) = cache.as_mut() {
                if c.enabled() && !prompt_ids.is_empty() {
                    // `fp` was computed at lookup time on the
                    // enabled-cache path; reuse it instead of paying
                    // a second fingerprint clone here.
                    let fingerprint = fp.get_or_insert_with(|| self.fingerprint());
                    match self.snapshot_kv() {
                        Ok(snap) => {
                            let bytes = snap.approx_bytes();
                            let evicted = c.insert(prompt_ids.to_vec(), fingerprint, snap);
                            info!(
                                cached_bytes = bytes,
                                cache_len = c.len(),
                                evicted,
                                "kv-prefix-cache MISS — inserted snapshot"
                            );
                        }
                        Err(e) => {
                            // Snapshot failure is non-fatal — the
                            // generation has already completed
                            // prefill correctly; we just won't cache.
                            warn!(error = %e, "kv-prefix-cache snapshot failed; not caching");
                        }
                    }
                }
            }
        }

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

    /// Speculative-decode wrapper around [`Self::generate`]. Uses the
    /// caller-provided [`crate::ngram_draft::Draft`] as a zero-compute
    /// draft model and verifies each round of `K` drafts through the
    /// full target forward path (token-by-token shell calls — there is
    /// **no multi-token shell forward kernel yet**, so each verify pass
    /// pays K full target forwards). The win is therefore concentrated
    /// in the wire/orchestration overhead the engine pays per token, not
    /// in the per-token compute. On the single-stage Rust engine that
    /// overhead is just one round-trip through `step()` + sampler per
    /// token — small relative to K2.6's ~9 s/token, so single-stage
    /// speedup is bounded by *whatever fraction of per-token cost is
    /// NOT in `forward_shells`* (today: layer 0 + head + sampler, ~50ms
    /// of a ~9 s budget, i.e. negligible).
    ///
    /// The real payoff is in pipeline-parallel mode: every accepted
    /// draft saves one full wire round-trip through the pipeline. At
    /// the 22 ms cross-host RT measured on the cascadia-enterprise fleet, K=8 with
    /// 70% acceptance saves `5.6 × 22 ms = 123 ms` per round — roughly
    /// the same proportional cost as rainier reported on llama 3.1 8B.
    ///
    /// **Distributed support is currently OUT OF SCOPE**. The
    /// pipeline-parallel `step_first` / `step_worker` paths in
    /// [`crate::engine::SparseMoEEngine`] do not yet know about
    /// multi-token verify frames — adding that requires extending
    /// [`crate::dist::FrameKind`] with a `ForwardBatch(K)` variant.
    /// This function is single-stage only for now; callers in multi-stage
    /// configs should fall back to [`Self::generate`].
    ///
    /// Implementation notes:
    /// - Greedy acceptance: we accept consecutive matches between
    ///   draft and target greedy argmax, stopping at the first
    ///   mismatch. The "bonus" target token at the first mismatch is
    ///   itself a valid sample we keep (matches rainier's
    ///   `k26_spec_decode.py` line 282).
    /// - KV rewind: after each round, `rewind_kv(rejected)` truncates
    ///   the K/V buffers back to the accepted prefix. The draft's
    ///   history is mirrored via [`crate::ngram_draft::Draft::rewind`].
    /// - **Temperature handling**: we currently only support greedy
    ///   spec-decode (temperature ≤ 0). At non-zero temperature, the
    ///   classical Leviathan et al. "rejection sampling" acceptance
    ///   rule applies; we punt to plain [`Self::generate`] in that
    ///   case rather than ship an incorrect acceptance test. This
    ///   matches rainier's `MaskedReq.feed` path which is also
    ///   greedy-only.
    ///
    /// Returns the generated tokens (excluding the prompt; excluding
    /// any trailing EOS).
    pub fn generate_speculative(
        &mut self,
        prompt_ids: &[i64],
        max_tokens: usize,
        cfg: &crate::sampling::SamplingConfig,
        draft: &mut crate::ngram_draft::Draft,
    ) -> Result<Vec<i64>, RunnerError> {
        // Non-greedy sampling falls through to the standard path —
        // greedy-acceptance spec decode would change the output
        // distribution under temperature > 0.
        if cfg.temperature > 0.0 {
            return self.generate(prompt_ids, max_tokens, cfg);
        }

        self.reset_kv();
        draft.reset();
        let eos: Vec<i64> = self
            .manifest
            .eos_token_ids
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut generated: Vec<i64> = Vec::with_capacity(max_tokens);
        let mut history: Vec<i64> = Vec::with_capacity(prompt_ids.len() + max_tokens);

        // Prefill — same shape as `generate`. We load the prompt into
        // the draft alongside the target's KV cache so the draft has
        // its prompt-suffix k-grams indexed.
        info!(
            prompt_len = prompt_ids.len(),
            "spec: prefill (token-by-token)"
        );
        let mut last_logits: Option<Vec<f32>> = None;
        let t_pre = Instant::now();
        for &t in prompt_ids.iter() {
            history.push(t);
            let logits = self.step(&history, 1)?;
            last_logits = Some(logits);
        }
        draft.warm_with_prompt(prompt_ids);
        info!(secs = t_pre.elapsed().as_secs_f64(), "spec: prefill done");

        // First generated token from the last prefill step's logits.
        let first = match last_logits {
            Some(l) => argmax_i64(&l),
            None => return Ok(generated),
        };
        if eos.contains(&first) {
            return Ok(generated);
        }
        history.push(first);
        generated.push(first);
        draft.append(first);

        // Spec-decode rounds.
        let mut n_rounds: u32 = 0;
        let mut n_drafts_total: u32 = 0;
        let mut n_accepted_total: u32 = 0;
        while generated.len() < max_tokens {
            n_rounds += 1;
            let budget = max_tokens - generated.len();
            // 1. Propose draft tokens via n-gram lookup.
            let mut drafts = draft.propose();
            // Trim the proposal to the remaining budget — if budget is
            // smaller than the proposal length, cap so we don't have
            // to throw away forwards we'd never emit.
            if drafts.len() > budget {
                drafts.truncate(budget);
            }

            // Path 1: empty proposal → fall back to one standard
            // forward step. Same cost as plain `generate`, plus a
            // hash-table miss. No spec round counted.
            if drafts.is_empty() {
                let logits = self.step(&history, 1)?;
                let next = argmax_i64(&logits);
                if eos.contains(&next) {
                    break;
                }
                history.push(next);
                generated.push(next);
                draft.append(next);
                continue;
            }

            // 2. Verify drafts in a single multi-token forward.
            //
            // Conceptually the K verify forwards process the K tokens
            // `[bonus, draft[0], draft[1], ..., draft[K-2]]` and produce
            // K logit rows — the i-th row predicts the token at position
            // `past_seq_len + i + 1`, which we compare against `draft[i]`
            // for the acceptance check.
            //
            // Sequentially this would be K independent `step()` calls;
            // iter 041's `_multi` API + iter 042's tiled AVX-VNNI GEMM
            // collapse them into one `step_multi(seq=K)` that amortizes
            // weight loads across the K tokens. Bit-identical to the
            // K-step path (the shell's `_multi` is bit-identical to its
            // scalar reference; layer 0's `_multi` is a scalar loop with
            // the same writes; head runs per-token) — only wall-clock
            // moves.
            //
            // Pending-token convention: before this section history's
            // last token (the bonus) is in history but its KV slot is
            // empty. We push the first K-1 drafts so that the K-token
            // tail of history is exactly `[bonus, draft[0..K-2]]`. The
            // multi-call fills the bonus's KV slot + the first K-1
            // drafts' slots, and the last draft (draft[K-1]) gets pushed
            // post-call to retain the same pending-token drift downstream.
            n_drafts_total += drafts.len() as u32;
            for &draft_tok in drafts.iter().take(drafts.len() - 1) {
                history.push(draft_tok);
            }
            let logits_rows = self.step_multi(&history, drafts.len())?;
            let mut target_samples: Vec<i64> = Vec::with_capacity(drafts.len() + 1);
            for row in &logits_rows {
                target_samples.push(argmax_i64(row));
            }
            // Push the last draft to restore the post-loop history layout
            // (every draft in history, KV trailing by 1) that the original
            // K-step loop produced.
            history.push(drafts[drafts.len() - 1]);

            // 3. Acceptance: longest matching prefix between drafts
            // and target_samples. See [`crate::spec_decode::count_accepted`].
            let accepted = crate::spec_decode::count_accepted(&drafts, &target_samples);
            n_accepted_total += accepted as u32;

            // 4. Bonus token: target's prediction at the first
            // rejection boundary (in the partial-accept case), OR the
            // result of one extra forward when all K drafts accepted
            // (so the next round has a fresh `prev_correction`).
            let bonus_forward_ran = accepted == drafts.len();
            let bonus: i64 = if !bonus_forward_ran {
                target_samples[accepted]
            } else {
                let logits = self.step(&history, 1)?;
                argmax_i64(&logits)
            };

            // 5. Reconcile state via the pure helper. After this,
            // `history` + KV represent the accepted prefix (NOT
            // including the bonus — we append it explicitly below
            // so the EOS check can roll it back cleanly).
            //
            // `pending_token_in_history=true`: this driver pre-pushes
            // `first_gen` to history before round 1 and appends each
            // round's `bonus` to history at end-of-round — both ride
            // ahead of KV by 1 slot. The K-loop's first verify forward
            // catches up the pending token's KV slot as a side effect,
            // so the helper must rewind one LESS than the clean
            // convention. See `spec_decode::reconcile_after_round` for
            // the convention contract and the pending-token regression
            // tests added in fix/spec-decode-reconcile-off-by-one-043.
            let r = crate::spec_decode::reconcile_after_round(
                drafts.len(),
                accepted,
                bonus_forward_ran,
                true,
            );
            if r.history_pop > 0 {
                history.truncate(history.len() - r.history_pop);
            }
            if r.kv_rewind > 0 {
                self.rewind_kv(r.kv_rewind);
            }

            // 6. Emit accepted drafts + bonus into the public output.
            // Stop on EOS or max_tokens, undoing the bonus from KV
            // when EOS is reached BEFORE the bonus would be emitted
            // (the bonus is NOT in KV at this point — the next round's
            // first forward writes its slot — so no KV undo needed for
            // the bonus itself).
            let mut hit_eos = false;
            let mut bonus_pushed_to_history = false;
            for &t in drafts.iter().take(accepted) {
                if eos.contains(&t) {
                    hit_eos = true;
                    break;
                }
                generated.push(t);
                draft.append(t);
                if generated.len() >= max_tokens {
                    break;
                }
            }
            if !hit_eos && generated.len() < max_tokens {
                if eos.contains(&bonus) {
                    hit_eos = true;
                } else {
                    history.push(bonus);
                    draft.append(bonus);
                    generated.push(bonus);
                    bonus_pushed_to_history = true;
                }
            }

            // Debug invariant: every layer's past_seq_len should trail
            // history.len() by exactly 1 when the bonus rode through
            // (the next round's first verify forward will fill its
            // slot), and by 0 when we cut the round short (EOS hit, or
            // max_tokens saturated before the bonus push). Strip from
            // prod paths.
            //
            // Convention contract: this matches the
            // `pending_token_in_history=true` convention
            // `reconcile_after_round` uses; see its docs for the full
            // mathematical statement.
            let expected_drift = if bonus_pushed_to_history { 1 } else { 0 };
            debug_assert!(
                self.kv_invariant_holds(&history, expected_drift),
                "KV invariant broken (expected drift {expected_drift})"
            );

            if hit_eos {
                break;
            }
        }

        info!(
            tokens = generated.len(),
            n_rounds,
            total_drafts = n_drafts_total,
            total_accepted = n_accepted_total,
            accept_rate = if n_drafts_total > 0 {
                n_accepted_total as f32 / n_drafts_total as f32
            } else {
                0.0
            },
            "spec_decode done"
        );
        Ok(generated)
    }

    /// Debug helper used by `generate_speculative` to assert all layers
    /// agree on past_seq_len, and that it matches the public history
    /// length minus `pending_drift`. Returns true on agreement, false
    /// otherwise. Public so the unit tests in `crate::spec_decode` can
    /// call it.
    ///
    /// `pending_drift = 0` is the "clean" invariant (KV == history.len)
    /// used by `generate`. `pending_drift = 1` is the spec-decode
    /// runner / pipeline-parallel convention where history pre-pushes
    /// `first_gen` (and each round's `bonus`) ahead of KV by one slot;
    /// the next forward will catch the slot up. See
    /// [`crate::spec_decode::reconcile_after_round`] for the contract.
    pub fn kv_invariant_holds(&self, history: &[i64], pending_drift: usize) -> bool {
        if history.len() < pending_drift {
            return false;
        }
        let target = history.len() - pending_drift;
        if let Some(l0) = self.layer0.as_ref() {
            if l0.past_seq_len != target {
                return false;
            }
        }
        for l in &self.layers {
            if l.past_seq_len != target {
                return false;
            }
        }
        true
    }
}

/// Greedy argmax over a logits row. Mirrors the inline impl in
/// `dist_spec.rs::argmax` — kept private to the sparse-moe crate to
/// avoid a cross-engine dep.
fn argmax_i64(xs: &[f32]) -> i64 {
    let mut best = 0i64;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v.is_finite() && v > best_v {
            best_v = v;
            best = i as i64;
        }
    }
    best
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

/// Copy a layer's populated KV prefix (`[NUM_HEADS, past_seq, head_dim]`)
/// out of its capacity buffer (`[NUM_HEADS, capacity, head_dim]`) into
/// a fresh packed `Vec<u16>` suitable for serialization. The capacity
/// buffer's per-head bases shift on grow; packing strips that variance
/// so a snapshot taken at cap=32 restores correctly into cap=64.
fn pack_layer_slice(
    lid: u32,
    past_k: &[u16],
    past_v: &[u16],
    past_seq: usize,
    capacity: usize,
) -> LayerKvSlice {
    debug_assert_eq!(past_k.len(), NUM_HEADS * capacity * QK_HEAD_DIM);
    debug_assert_eq!(past_v.len(), NUM_HEADS * capacity * V_HEAD_DIM);
    debug_assert!(past_seq <= capacity);
    let mut k_out = Vec::with_capacity(NUM_HEADS * past_seq * QK_HEAD_DIM);
    let mut v_out = Vec::with_capacity(NUM_HEADS * past_seq * V_HEAD_DIM);
    for h in 0..NUM_HEADS {
        let k_src_base = h * capacity * QK_HEAD_DIM;
        k_out.extend_from_slice(&past_k[k_src_base..k_src_base + past_seq * QK_HEAD_DIM]);
        let v_src_base = h * capacity * V_HEAD_DIM;
        v_out.extend_from_slice(&past_v[v_src_base..v_src_base + past_seq * V_HEAD_DIM]);
    }
    LayerKvSlice {
        lid,
        past_k: k_out,
        past_v: v_out,
    }
}

/// Inverse of `pack_layer_slice`: copy a packed
/// `[NUM_HEADS, past_seq, head_dim]` snapshot into a runner's
/// `[NUM_HEADS, capacity, head_dim]` buffer at the per-head bases
/// the runner expects. Slots `past_seq..capacity` per head are left
/// untouched (they were never read; their contents don't matter
/// because forward_shells only reads `0..past_seq_len`).
fn unpack_layer_slice(
    past_k: &mut [u16],
    past_v: &mut [u16],
    snap: &LayerKvSlice,
    past_seq: usize,
    capacity: usize,
) -> Result<(), RunnerError> {
    if snap.past_k.len() != NUM_HEADS * past_seq * QK_HEAD_DIM {
        return Err(RunnerError::Internal(format!(
            "unpack: L{} past_k len {} != expected {}",
            snap.lid,
            snap.past_k.len(),
            NUM_HEADS * past_seq * QK_HEAD_DIM
        )));
    }
    if snap.past_v.len() != NUM_HEADS * past_seq * V_HEAD_DIM {
        return Err(RunnerError::Internal(format!(
            "unpack: L{} past_v len {} != expected {}",
            snap.lid,
            snap.past_v.len(),
            NUM_HEADS * past_seq * V_HEAD_DIM
        )));
    }
    debug_assert_eq!(past_k.len(), NUM_HEADS * capacity * QK_HEAD_DIM);
    debug_assert_eq!(past_v.len(), NUM_HEADS * capacity * V_HEAD_DIM);
    for h in 0..NUM_HEADS {
        let k_src_base = h * past_seq * QK_HEAD_DIM;
        let k_dst_base = h * capacity * QK_HEAD_DIM;
        past_k[k_dst_base..k_dst_base + past_seq * QK_HEAD_DIM]
            .copy_from_slice(&snap.past_k[k_src_base..k_src_base + past_seq * QK_HEAD_DIM]);
        let v_src_base = h * past_seq * V_HEAD_DIM;
        let v_dst_base = h * capacity * V_HEAD_DIM;
        past_v[v_dst_base..v_dst_base + past_seq * V_HEAD_DIM]
            .copy_from_slice(&snap.past_v[v_src_base..v_src_base + past_seq * V_HEAD_DIM]);
    }
    Ok(())
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

    // Cross-check of f32_to_bf16_bits against half::bf16::from_f32 now
    // lives in cascadia-int4-gemm::format::tests, the canonical home for
    // the helper imported above. Removed from here to avoid duplicating
    // the contract pin.

    #[test]
    fn argmax_i64_finds_max_index() {
        let xs = vec![0.1, 0.9, 0.3, 0.8];
        assert_eq!(argmax_i64(&xs), 1);
    }

    #[test]
    fn argmax_i64_skips_non_finite() {
        // NaN and Inf are skipped; the largest finite wins. This
        // matches the dist_spec.rs argmax behavior — without the
        // is_finite() guard, a NaN early in the logits row would
        // silently make argmax return 0.
        let xs = vec![f32::NAN, -10.0, 0.5, f32::NEG_INFINITY, 0.3];
        assert_eq!(argmax_i64(&xs), 2);
    }

    #[test]
    fn argmax_i64_all_non_finite_returns_zero() {
        let xs = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        // All non-finite (INFINITY is technically finite()=false in Rust).
        // Wait — INFINITY.is_finite() is false. So all should skip and
        // best stays 0.
        assert_eq!(argmax_i64(&xs), 0);
    }

    /// Stamp a capacity buffer with a unique per-(head, slot, dim)
    /// signature, pack it into a `LayerKvSlice`, then unpack into a
    /// fresh capacity buffer and verify the populated prefix matches
    /// bit-for-bit. This is the byte-identity guarantee the task
    /// brief calls out as load-bearing: "Cache must be byte-identical
    /// to live prefill (KV bits must match)".
    #[test]
    fn pack_unpack_roundtrip_is_bit_identical() {
        let cap = 8;
        let past = 5;
        let mut k = vec![0u16; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut v = vec![0u16; NUM_HEADS * cap * V_HEAD_DIM];
        // Unique signature per cell so any mis-indexing surfaces.
        let stamp = |h: usize, s: usize, d: usize| -> u16 {
            ((h * 100_000 + s * 1_000 + d) & 0xFFFF) as u16
        };
        for h in 0..NUM_HEADS {
            for s in 0..past {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    k[off] = stamp(h, s, d);
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    v[off] = stamp(h, s, d).wrapping_add(0x8000);
                }
            }
        }
        let slice = pack_layer_slice(7, &k, &v, past, cap);
        assert_eq!(slice.lid, 7);
        assert_eq!(slice.past_k.len(), NUM_HEADS * past * QK_HEAD_DIM);
        assert_eq!(slice.past_v.len(), NUM_HEADS * past * V_HEAD_DIM);
        // Unpack into a fresh capacity-buffer (sentinel pattern in untouched slots).
        const SENTINEL: u16 = 0xBEEF;
        let mut k2 = vec![SENTINEL; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut v2 = vec![SENTINEL; NUM_HEADS * cap * V_HEAD_DIM];
        unpack_layer_slice(&mut k2, &mut v2, &slice, past, cap).expect("unpack");
        // Populated prefix must be bit-identical.
        for h in 0..NUM_HEADS {
            for s in 0..past {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    assert_eq!(k2[off], k[off], "k[h={h}, s={s}, d={d}] mismatch");
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    assert_eq!(v2[off], v[off], "v[h={h}, s={s}, d={d}] mismatch");
                }
            }
        }
    }

    /// Unpack from a `past=5` snapshot into a fresh `cap=16` buffer
    /// (twice the original). Per-head bases shift; the populated
    /// prefix should still land at the right offsets, and slots
    /// `past..cap` should be left at their pre-fill value (NaN here)
    /// so any accidental read past `past_seq_len` surfaces as a test
    /// failure rather than silently reading stale zeros.
    #[test]
    fn unpack_into_larger_capacity_preserves_layout() {
        let src_cap = 8;
        let past = 5;
        let mut k = vec![0u16; NUM_HEADS * src_cap * QK_HEAD_DIM];
        let mut v = vec![0u16; NUM_HEADS * src_cap * V_HEAD_DIM];
        let stamp_k = |h: usize, s: usize| -> u16 { ((h * 100 + s) & 0xFFFF) as u16 };
        let stamp_v = |h: usize, s: usize| -> u16 { stamp_k(h, s).wrapping_add(0x8000) };
        for h in 0..NUM_HEADS {
            for s in 0..past {
                k[h * src_cap * QK_HEAD_DIM + s * QK_HEAD_DIM] = stamp_k(h, s);
                v[h * src_cap * V_HEAD_DIM + s * V_HEAD_DIM] = stamp_v(h, s);
            }
        }
        let slice = pack_layer_slice(3, &k, &v, past, src_cap);
        let dst_cap = 16;
        const SENTINEL: u16 = 0xBEEF;
        let mut k2 = vec![SENTINEL; NUM_HEADS * dst_cap * QK_HEAD_DIM];
        let mut v2 = vec![SENTINEL; NUM_HEADS * dst_cap * V_HEAD_DIM];
        unpack_layer_slice(&mut k2, &mut v2, &slice, past, dst_cap).expect("unpack");
        for h in 0..NUM_HEADS {
            for s in 0..past {
                let k_off = h * dst_cap * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let v_off = h * dst_cap * V_HEAD_DIM + s * V_HEAD_DIM;
                assert_eq!(k2[k_off], stamp_k(h, s));
                assert_eq!(v2[v_off], stamp_v(h, s));
            }
            // Slots past..dst_cap should still be SENTINEL — never read.
            for s in past..dst_cap {
                let k_off = h * dst_cap * QK_HEAD_DIM + s * QK_HEAD_DIM;
                assert_eq!(k2[k_off], SENTINEL, "unpack wrote into reserved slot");
            }
        }
    }

    #[test]
    fn unpack_rejects_wrong_length_slice() {
        let cap = 4;
        let past = 2;
        let mut k = vec![0u16; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut v = vec![0u16; NUM_HEADS * cap * V_HEAD_DIM];
        // Snapshot claims past=2 but past_k length only matches past=1.
        let bad = LayerKvSlice {
            lid: 0,
            past_k: vec![0u16; NUM_HEADS * QK_HEAD_DIM], // claims past=1 (deliberately wrong)
            past_v: vec![0u16; NUM_HEADS * past * V_HEAD_DIM],
        };
        let err = unpack_layer_slice(&mut k, &mut v, &bad, past, cap)
            .expect_err("expected length-check error");
        let msg = format!("{err}");
        assert!(msg.contains("past_k"), "error should mention past_k: {msg}");
    }
}
