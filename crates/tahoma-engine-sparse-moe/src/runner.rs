//! Core sparse-MoE inference loop.
//!
//! Holds compiled handles for every shell + layer-0 + head, a manifest,
//! the per-layer KV caches, and an LRU cache of compiled experts. Driven
//! by [`Runner::generate_argmax`], which generates `max_tokens` greedy
//! tokens for a prompt.
//!
//! Not async, not Send. Each generation owns its own KV state; the
//! Engine wrapper above this drives one call at a time.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use half::bf16;
use tahoma_int4_gemm::layer0_int4::{
    embed_token_bf16, layer0_forward_decode_int4_with_capacity, Int4Layer0,
};
use tahoma_int4_gemm::safetensors_source::Shard;
use tahoma_int4_gemm::shell::{NUM_HEADS, QK_HEAD_DIM, V_HEAD_DIM};
use tahoma_int4_gemm::shell_int4::{shell_forward_decode_int4_with_capacity, Int4Shell};
use tahoma_int4_gemm::{
    expert_forward as int4_expert_forward, ExpertWeights, SafetensorsExpert,
    SafetensorsExpertSource,
};
use tahoma_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};
use thiserror::Error;
use tracing::{debug, info};

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
struct LayerState {
    lid: u32,
    int4_shell: Int4Shell,
    /// Layout: `[NUM_HEADS, kv_capacity, QK_HEAD_DIM]` row-major.
    /// Slots `past_seq_len..kv_capacity` per head are reserved but
    /// unpopulated (their contents don't matter).
    past_k: Vec<f32>,
    /// Layout: `[NUM_HEADS, kv_capacity, V_HEAD_DIM]` row-major.
    past_v: Vec<f32>,
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
    past_k: Vec<f32>,
    past_v: Vec<f32>,
    past_seq_len: usize,
    kv_capacity: usize,
}

impl LayerState {
    fn new(lid: u32, int4_shell: Int4Shell) -> Self {
        let cap = INITIAL_KV_CAPACITY;
        Self {
            lid,
            int4_shell,
            past_k: vec![0.0f32; NUM_HEADS * cap * QK_HEAD_DIM],
            past_v: vec![0.0f32; NUM_HEADS * cap * V_HEAD_DIM],
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
    /// autolab iter 088 (cross-layer expert share): per-(layer-position,
    /// expert) fire count. Same shape as iter 054's `expert_hits` —
    /// kept here standalone (the iter 054 branch isn't merged into
    /// main) so this iteration's investigation doesn't depend on it.
    /// One HashMap per layer slot held by this rank. Persists across
    /// `reset_kv` so steady-state stats reflect the full workload.
    expert_hits: Vec<HashMap<u32, u64>>,
    /// autolab iter 088 (cross-layer expert share): IDs actually
    /// dispatched in the *previously processed* MoE layer of the
    /// current token. Cleared at the start of every `forward_shells`
    /// call so the cross-layer sharing signal only counts experts
    /// that fire in immediately consecutive layers of the same token.
    /// Rebuilt at the end of each layer iteration.
    last_layer_routing_ids: HashSet<u32>,
    /// autolab iter 088: cumulative count of (layer-position, expert)
    /// dispatches across all (token, layer N+1) pairs that this rank
    /// has run since startup. Denominator of the "share fraction"
    /// metric. Excludes the first layer of each token (it has no
    /// "previous layer" within the current token).
    cross_layer_total: u64,
    /// autolab iter 088: cumulative count of (layer-position, expert)
    /// dispatches that ALSO appeared in the immediately preceding
    /// layer of the same token. Numerator of the share fraction.
    cross_layer_overlap: u64,
    /// autolab iter 088: per-(prev-layer-position, this-layer-position,
    /// expert) co-occurrence count, optional and gated by
    /// `cross_layer_pair_tracking_enabled`. Bounded in size by the
    /// number of distinct (prev-pos, this-pos, eid) triples that ever
    /// fire — for K2.6 top-K=8 across 60 layers, this is at most
    /// ~60 * 8 * 384 = 184K entries in the absolute worst case but
    /// dispatch sparsity drops the practical max into the low
    /// thousands. Off by default to keep the steady-state work
    /// proportional to top-K, not top-K * cache-miss-rate.
    cross_layer_pair_hits: HashMap<(u32, u32, u32), u64>,
    cross_layer_pair_tracking_enabled: bool,
    /// autolab iter 088: when true, `forward_shells` reorders each
    /// layer N+1's dispatch sequence so experts that ALSO fired in
    /// layer N run first, keeping their weights L3-resident across
    /// the layer boundary. Phase 3 weighted-sum still walks the
    /// original `k = 0..top_k` order so the FP rounding chain is
    /// byte-identical regardless of dispatch sequence.
    cross_layer_dispatch: bool,
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
                past_k: vec![0.0f32; NUM_HEADS * cap * QK_HEAD_DIM],
                past_v: vec![0.0f32; NUM_HEADS * cap * V_HEAD_DIM],
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

        let n_my_layers = layers.len();
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
            // iter 088: per-position hit maps + cross-layer share counters,
            // all zero at startup. Pair tracking + reorder both default
            // OFF; turn them on via `set_cross_layer_pair_tracking` /
            // `set_cross_layer_dispatch` before generation.
            expert_hits: (0..n_my_layers).map(|_| HashMap::new()).collect(),
            last_layer_routing_ids: HashSet::new(),
            cross_layer_total: 0,
            cross_layer_overlap: 0,
            cross_layer_pair_hits: HashMap::new(),
            cross_layer_pair_tracking_enabled: false,
            cross_layer_dispatch: false,
        })
    }

    /// autolab iter 088: enable per-(prev-pos, this-pos, eid)
    /// co-occurrence tracking. Costs one `HashMap<(u32,u32,u32),u64>`
    /// insert per dispatched expert per layer N+1; off by default
    /// because the main per-token metric (`cross_layer_total` /
    /// `cross_layer_overlap`) doesn't need it. Used by the iter 088
    /// research bench to inspect *which* layer pairs share the most.
    pub fn set_cross_layer_pair_tracking(&mut self, enabled: bool) {
        self.cross_layer_pair_tracking_enabled = enabled;
    }

    /// autolab iter 088: enable cross-layer-share-aware dispatch
    /// reordering. When true, `forward_shells` permutes each non-first
    /// layer's dispatch order so experts that ALSO fired in the
    /// immediately preceding layer of the same token run FIRST.
    /// Phase 3 weighted sum still walks the original `k = 0..top_k`
    /// order so the FP rounding chain is unchanged and the output is
    /// byte-identical regardless of this flag. Off by default; turn
    /// on for the iter 088 A/B bench.
    pub fn set_cross_layer_dispatch(&mut self, enabled: bool) {
        self.cross_layer_dispatch = enabled;
    }

    /// autolab iter 088: read accessor — used by the engine layer to
    /// echo the flag back out of `Engine::warmup` / logs without
    /// owning a duplicate copy.
    pub fn cross_layer_dispatch_enabled(&self) -> bool {
        self.cross_layer_dispatch
    }

    /// autolab iter 088: snapshot of the cross-layer share counters.
    /// Returns `(total, overlap)` — the share fraction is
    /// `overlap as f64 / total as f64` (guard against zero on cold
    /// start). Cumulative since `Runner::load`; not reset by
    /// `reset_kv` so the bench can read a single number at the end
    /// of a 100-prompt sweep.
    pub fn cross_layer_share_snapshot(&self) -> (u64, u64) {
        (self.cross_layer_total, self.cross_layer_overlap)
    }

    /// autolab iter 088: snapshot of the per-(prev-pos, this-pos, eid)
    /// co-occurrence map. Empty unless `set_cross_layer_pair_tracking`
    /// was called with `true` before generation. Clone is bounded by
    /// the number of distinct triples — at most ~hundreds of K entries
    /// even for K2.6's 60-layer top-8 dispatch in steady state. Used
    /// by the iter 088 bench to drive a heatmap of which layer pairs
    /// have the densest share.
    pub fn cross_layer_pair_snapshot(&self) -> HashMap<(u32, u32, u32), u64> {
        self.cross_layer_pair_hits.clone()
    }

    /// autolab iter 088: read accessor for the per-layer-position hit
    /// map. Cumulative since `Runner::load`; not reset by `reset_kv`.
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
        let hidden = self.manifest.hidden_size as usize;
        let top_k = self.manifest.top_k as usize;
        if h_shape.len() != 3 || h_shape[0] != 1 || h_shape[1] != 1 || h_shape[2] != hidden {
            return Err(RunnerError::Internal(format!(
                "forward_shells: int4 shells require shape [1, 1, {hidden}], got {h_shape:?}"
            )));
        }
        let mut h_f32 = h_in.to_vec();

        // iter 088 (cross-layer expert share): the "previously dispatched
        // experts" signal only counts experts that fire in *immediately
        // consecutive* layers of the same token. Clear at token boundary
        // (= start of `forward_shells`) so signal is per-token.
        self.last_layer_routing_ids.clear();

        let n_layers = self.layers.len();
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
            let outs = shell_forward_decode_int4_with_capacity(
                &self.layers[i].int4_shell,
                &h_f32,
                &self.layers[i].past_k,
                &self.layers[i].past_v,
                past_seq_len,
                capacity,
            );

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

            // iter 088 (cross-layer expert share): three-phase
            // restructure modeled on iter 056. Gates the optional
            // dispatch reorder while preserving byte-identical output
            // (Phase 3 enforces ascending-k FP accumulation order).
            //
            // Phase 1 (original-order bookkeeping): walk
            // `k = 0..top_k` in router-score order to update
            // (a) per-position hit count, (b) cross-layer share
            // numerator + denominator, (c) optional per-pair
            // co-occurrence map. Byte-identical side-effect ordering
            // to the pre-088 loop's hit-rate counters (none, since
            // those were added in iter 054 which isn't merged into
            // main — iter 088 introduces them standalone).
            //
            // The "previous layer" reference is `last_layer_routing_ids`,
            // built at the END of the *previous* iteration of this
            // for-loop. For i == 0 it's empty (cleared above before the
            // for-loop), so the first layer of every token contributes
            // 0 to the share denominator/numerator and the reorder
            // degenerates to identity (router-score order).
            let is_first_layer_of_token = i == 0;
            for k in 0..top_k {
                let eid = outs.routing_ids[k] as u32;
                *self
                    .expert_hits
                    .get_mut(i)
                    .expect("layer-indexed hit map")
                    .entry(eid)
                    .or_insert(0) += 1;
                if !is_first_layer_of_token {
                    // Cross-layer share metric: this layer's dispatched
                    // expert counts toward the denominator unconditionally
                    // (we touched it). Counts toward the numerator iff
                    // it also fired in the *immediately preceding* layer
                    // of this same token.
                    self.cross_layer_total += 1;
                    if self.last_layer_routing_ids.contains(&eid) {
                        self.cross_layer_overlap += 1;
                    }
                    if self.cross_layer_pair_tracking_enabled {
                        // Layer pair key: (prev-position, this-position,
                        // eid). The position is `i - 1` / `i` (zero-
                        // based index into this rank's layer slice),
                        // NOT the lid — pair stats are meaningful per
                        // *adjacent* layers of this rank's stage, and
                        // the next stage's first layer is a different
                        // story (it sees a fresh `last_layer_routing_ids`
                        // = {} since that state lives in this Runner
                        // and doesn't cross the wire).
                        *self
                            .cross_layer_pair_hits
                            .entry((i as u32 - 1, i as u32, eid))
                            .or_insert(0) += 1;
                    }
                }
            }

            // Phase 2 (dispatch in cross-layer-aware OR router-score
            // order): compute each expert and stash its output indexed
            // by the original `k`. With `cross_layer_dispatch == false`
            // (default) the sequence is the identity = byte-identical
            // to the pre-088 path. With it enabled and we're past the
            // first layer of this token, the sequence puts experts
            // that overlap with the previous layer FIRST (ascending
            // k tie-break), so the kernel re-reads weights still warm
            // in L3 from the just-finished layer.
            let dispatch_seq: Vec<usize> = if self.cross_layer_dispatch && !is_first_layer_of_token
            {
                cross_layer_dispatch_order(&outs.routing_ids, &self.last_layer_routing_ids)
            } else {
                (0..top_k).collect()
            };
            let mut expert_outs: Vec<Option<Vec<f32>>> = (0..top_k).map(|_| None).collect();
            for &k in dispatch_seq.iter() {
                let eid = outs.routing_ids[k] as u32;
                let y_f32 = self.dispatch_expert(lid, eid, &outs.attn_out_post_norm)?;
                expert_outs[k] = Some(y_f32);
            }

            // Phase 3 (original-order weighted sum): accumulate into
            // `moe` in ascending `k` so the FP rounding chain is
            // identical regardless of `dispatch_seq` ordering. THIS
            // is the bit-identity hinge — without it, reordering
            // Phase 2 would silently flip output bits via different
            // f32 accumulation roundings.
            let mut moe = vec![0.0f32; hidden];
            for (k, slot) in expert_outs.iter().enumerate() {
                let y_f32 = slot
                    .as_ref()
                    .expect("Phase 2 dispatched every k (no threshold gating on main)");
                let w = outs.routing_weights[k];
                for j in 0..hidden {
                    moe[j] += w * y_f32[j];
                }
            }

            // Combine: h_next = residual + shared + moe (single token).
            for j in 0..hidden {
                h_f32[j] = outs.attn_residual[j] + outs.shared_expert_out[j] + moe[j];
            }

            // iter 088: stash the IDs we just dispatched so the next
            // iteration of this for-loop can see them. We rebuild
            // (rather than incrementally update) so the set reflects
            // *exactly* what fired this layer — no stale entries from
            // older layers. Cheap: top-K=8 inserts per layer.
            self.last_layer_routing_ids.clear();
            for k in 0..top_k {
                self.last_layer_routing_ids
                    .insert(outs.routing_ids[k] as u32);
            }
        }

        // iter 088: emit a one-line per-token summary of the cross-layer
        // share metric. Cumulative counters so the bench can read the
        // final ratio at the end of a sweep; logged here for the
        // research loop to scrape per-step deltas if needed.
        debug!(
            stage = "cross_layer_share",
            n_layers,
            past_seq_len,
            cross_layer_total = self.cross_layer_total,
            cross_layer_overlap = self.cross_layer_overlap,
            share_frac = if self.cross_layer_total == 0 {
                0.0
            } else {
                self.cross_layer_overlap as f64 / self.cross_layer_total as f64
            },
            cross_layer_dispatch = self.cross_layer_dispatch,
            "iter 088 cross-layer share snapshot"
        );

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
        info!(prompt_len = prompt_ids.len(), "prefill (token-by-token)");
        let mut history: Vec<i64> = Vec::with_capacity(prompt_ids.len() + max_tokens);
        let mut last_logits: Option<Vec<f32>> = None;
        let t_pre = Instant::now();
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
        info!(
            secs = prefill_secs,
            tok_per_s = prompt_ids.len() as f64 / prefill_secs,
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

/// autolab iter 088 (cross-layer expert share): produce a permutation
/// of `0..routing_ids.len()` ordered so that any expert ID also
/// present in `prev_layer_ids` appears FIRST, with ascending
/// original index as the tie-break. Returns indices into
/// `routing_ids`, NOT expert IDs — the dispatch loop uses these to
/// look up `routing_ids[k]` and `routing_weights[k]` from the shell
/// output without losing the original alignment.
///
/// Example: `routing_ids = [42, 99, 7, 200, 50, 1, 88, 12]` (router
/// top-8 by score), `prev_layer_ids = {7, 50, 88}` ⇒ returned
/// permutation `[2, 4, 6, 0, 1, 3, 5, 7]`. The three shared experts
/// (7, 50, 88) land at slots 0-2 in router-score order amongst
/// themselves; the five non-shared experts follow in router-score
/// order.
///
/// **Pure function — no side effects.** The runner's dispatch loop
/// preserves byte-identical output by:
/// (a) doing all hit-rate / cross-layer-share bookkeeping in the
///     original `k = 0..top_k` order (Phase 1),
/// (b) calling `dispatch_expert` in the cross-layer-aware order
///     (Phase 2, this permutation),
/// (c) summing the weighted expert outputs into `moe` in original
///     order (Phase 3) so the float-addition rounding chain is
///     unchanged.
///
/// `prev_layer_ids` empty (= first layer of a token, or the cross-
/// layer-dispatch flag is off) implies the permutation degenerates
/// to the identity — every "is-shared" key is `false` and the tie-
/// break collapses to the original index. Identical to router-score
/// dispatch order.
///
/// Separated from `Runner::forward_shells` so unit tests can verify
/// the permutation logic without a loaded model.
fn cross_layer_dispatch_order(routing_ids: &[i64], prev_layer_ids: &HashSet<u32>) -> Vec<usize> {
    let mut order: Vec<(usize, bool)> = routing_ids
        .iter()
        .enumerate()
        .map(|(k, &id)| {
            // Guard cast: router IDs are non-negative expert ids in
            // [0, num_experts). A pathological negative id would look
            // up as non-shared (false), keeping it at the tail of the
            // permutation.
            let shared = if id >= 0 {
                prev_layer_ids.contains(&(id as u32))
            } else {
                false
            };
            (k, shared)
        })
        .collect();
    // Primary key: `shared` (true first). Secondary: ascending
    // original index = router-score order. Using `bool::cmp` and
    // reversing for descending — `true.cmp(&false) == Greater`, so
    // `b.1.cmp(&a.1)` puts true first.
    order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    order.into_iter().map(|(k, _)| k).collect()
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
/// buffer, copy the populated `past_seq` prefix per head from a
/// `[NUM_HEADS, old_cap, head_dim]` source. The rest is zero. Pure
/// over buffers — no Int4Shell required, which keeps it unit-testable.
///
/// Uses `try_reserve_exact` + `resize` so OOM at long context bubbles
/// up as a recoverable `Err` instead of an abort from the global
/// allocator.
fn grow_kv_buffer(
    src: &[f32],
    past_seq: usize,
    old_cap: usize,
    new_cap: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    debug_assert!(new_cap >= old_cap);
    debug_assert!(past_seq <= old_cap);
    debug_assert_eq!(src.len(), NUM_HEADS * old_cap * head_dim);
    let total = NUM_HEADS * new_cap * head_dim;
    let mut dst: Vec<f32> = Vec::new();
    dst.try_reserve_exact(total).map_err(|e| {
        format!(
            "alloc {total} f32 ({:.1} MB) failed: {e}",
            (total * 4) as f64 / 1e6
        )
    })?;
    dst.resize(total, 0.0f32);
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

/// Write the new step's per-head K (or V) row at slot `past_seq`
/// inside a `[NUM_HEADS, capacity, HEAD_DIM]` buffer. No allocation,
/// no shift — the slot exists because the caller pre-allocated /
/// grew `capacity` to be `> past_seq`.
fn write_present_kv(
    buf: &mut [f32],
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
        buf[dst_off..dst_off + head_dim]
            .copy_from_slice(&present[h * head_dim..(h + 1) * head_dim]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_present_kv_into_empty_slot() {
        // 2 heads, capacity=3, head_dim=2, past_seq=0.
        // Buffer starts zero; expect present rows at slot 0 of each head.
        let mut buf = vec![0.0f32; 2 * 3 * 2];
        let present = vec![1.0, 2.0, 3.0, 4.0]; // h0=[1,2], h1=[3,4]
        write_present_kv(&mut buf, &present, 0, 3, 2, 2);
        // head 0 base = 0,        slot 0 = [1, 2]
        // head 1 base = capacity*head_dim = 6, slot 0 = [3, 4]
        assert_eq!(buf[0..2], [1.0, 2.0]);
        assert_eq!(buf[6..8], [3.0, 4.0]);
        // Unfilled slots untouched.
        assert_eq!(buf[2..6], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(buf[8..12], [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn write_present_kv_into_middle_slot() {
        // 2 heads, capacity=4, head_dim=2, past_seq=2.
        // Pre-populate slots 0..2 of each head, then write slot 2.
        let mut buf = vec![0.0f32; 2 * 4 * 2];
        // head 0 base=0
        buf[0..2].copy_from_slice(&[10.0, 11.0]); // slot 0
        buf[2..4].copy_from_slice(&[12.0, 13.0]); // slot 1
                                                  // head 1 base = capacity*head_dim = 8
        buf[8..10].copy_from_slice(&[20.0, 21.0]); // slot 0
        buf[10..12].copy_from_slice(&[22.0, 23.0]); // slot 1
        let present = vec![14.0, 15.0, 24.0, 25.0]; // h0=[14,15], h1=[24,25]
        write_present_kv(&mut buf, &present, 2, 4, 2, 2);
        // head 0 slot 2
        assert_eq!(buf[4..6], [14.0, 15.0]);
        // head 1 slot 2
        assert_eq!(buf[12..14], [24.0, 25.0]);
        // Existing slots untouched
        assert_eq!(buf[0..2], [10.0, 11.0]);
        assert_eq!(buf[8..10], [20.0, 21.0]);
    }

    #[test]
    fn grow_kv_buffer_doubles_and_preserves_data() {
        // Stamp a unique value at head h, slot 0, dim 0 of a
        // [NUM_HEADS, 2, QK_HEAD_DIM] buffer, then double to cap=4.
        // Each head's base offset shifts from h*2*D to h*4*D — the
        // stamp should still be at the new base offset.
        let mut src = vec![0.0f32; NUM_HEADS * 2 * QK_HEAD_DIM];
        for h in 0..NUM_HEADS {
            src[h * 2 * QK_HEAD_DIM] = (h + 1) as f32;
        }
        let dst = grow_kv_buffer(&src, 1, 2, 4, QK_HEAD_DIM).expect("alloc");
        assert_eq!(dst.len(), NUM_HEADS * 4 * QK_HEAD_DIM);
        for h in 0..NUM_HEADS {
            assert_eq!(
                dst[h * 4 * QK_HEAD_DIM],
                (h + 1) as f32,
                "head {h} stamp lost"
            );
        }
    }

    #[test]
    fn grow_kv_buffer_from_empty_is_zero_filled() {
        let src = vec![0.0f32; NUM_HEADS * 2 * QK_HEAD_DIM];
        let dst = grow_kv_buffer(&src, 0, 2, 4, QK_HEAD_DIM).expect("alloc");
        assert_eq!(dst.len(), NUM_HEADS * 4 * QK_HEAD_DIM);
        assert!(dst.iter().all(|&x| x == 0.0));
    }

    // ----- iter 088: cross-layer expert share helper tests -----
    //
    // These exercise the *pure* permutation logic in
    // `cross_layer_dispatch_order` without spinning up a full Runner
    // (which would require K2.6 weights on disk). The bit-identity
    // properties of the Phase 1/2/3 dispatch loop in `forward_shells`
    // depend on Phase 3 walking ascending k — that's enforced by the
    // shape of the loop and not by these tests, but the permutation
    // helper IS what we tune for L3 locality.

    #[test]
    fn cross_layer_dispatch_order_empty_prev_is_identity() {
        // First layer of every token: prev_layer_ids is empty,
        // permutation degenerates to identity. Important: this is the
        // semantic that makes the cross-layer-dispatch flag safe to
        // turn on for prompts where the first layer hasn't seen any
        // prior dispatch — output stays byte-identical because the
        // dispatch order doesn't change.
        let routing_ids: Vec<i64> = vec![42, 99, 7, 200, 50, 1, 88, 12];
        let prev = HashSet::new();
        let perm = cross_layer_dispatch_order(&routing_ids, &prev);
        assert_eq!(perm, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn cross_layer_dispatch_order_shared_first() {
        // 3 of 8 routing_ids overlap with prev_layer_ids: 7, 50, 88.
        // They land at slots 0..3 in router-score order amongst
        // themselves (k=2: id 7, k=4: id 50, k=6: id 88). The five
        // non-shared experts follow in router-score order.
        let routing_ids: Vec<i64> = vec![42, 99, 7, 200, 50, 1, 88, 12];
        let prev: HashSet<u32> = [7u32, 50, 88].iter().copied().collect();
        let perm = cross_layer_dispatch_order(&routing_ids, &prev);
        assert_eq!(perm, vec![2, 4, 6, 0, 1, 3, 5, 7]);
    }

    #[test]
    fn cross_layer_dispatch_order_all_shared_is_identity() {
        // Pathological: every routing_id was in the previous layer.
        // All keys tie on `true`, tie-break to ascending original k =
        // identity. This is the upper-bound case for the L3 win —
        // every dispatched expert is already warm in L3.
        let routing_ids: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let prev: HashSet<u32> = (1u32..=8).collect();
        let perm = cross_layer_dispatch_order(&routing_ids, &prev);
        assert_eq!(perm, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn cross_layer_dispatch_order_no_shared_is_identity() {
        // No expert in prev_layer_ids appears in routing_ids: every
        // key is `false`, tie-break to ascending k = identity. Same
        // ordering as the all-shared case (= router-score), which is
        // the floor on the L3 win (no shared experts to warm).
        let routing_ids: Vec<i64> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let prev: HashSet<u32> = [1u32, 2, 3].iter().copied().collect();
        let perm = cross_layer_dispatch_order(&routing_ids, &prev);
        assert_eq!(perm, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn cross_layer_dispatch_order_is_a_permutation() {
        // Output must be a valid permutation of 0..K: no duplicates,
        // no gaps. Property check across a few hand-picked shapes.
        for &k in &[1usize, 2, 4, 8] {
            let routing_ids: Vec<i64> = (0..k as i64).collect();
            let prev: HashSet<u32> = (0..k).filter(|i| i % 2 == 0).map(|i| i as u32).collect();
            let perm = cross_layer_dispatch_order(&routing_ids, &prev);
            assert_eq!(perm.len(), k);
            let mut sorted = perm.clone();
            sorted.sort();
            assert_eq!(
                sorted,
                (0..k).collect::<Vec<_>>(),
                "k={k}: not a permutation"
            );
        }
    }

    #[test]
    fn cross_layer_dispatch_order_negative_id_treated_as_not_shared() {
        // Pathological: a routing_id is negative (shouldn't happen in
        // K2.6, but the cast guard is part of the contract). It hashes
        // as "not shared" and lands in the tail by router-score order.
        let routing_ids: Vec<i64> = vec![5, -1, 7];
        let prev: HashSet<u32> = [5u32].iter().copied().collect();
        let perm = cross_layer_dispatch_order(&routing_ids, &prev);
        // Order: k=0 (id 5, shared) first; then k=1 (id -1, not shared)
        // and k=2 (id 7, not shared) in ascending k.
        assert_eq!(perm, vec![0, 1, 2]);
    }

    #[test]
    fn cross_layer_dispatch_order_preserves_router_score_among_shared() {
        // When multiple shared experts exist, they are in router-score
        // order amongst themselves (= ascending original k). This is
        // a deliberate choice: among the L3-warm set, the highest-
        // scored expert is the one whose weights are most likely to
        // be MOST warm (it was the last one called previous layer if
        // dispatch order was reverse-score, or first if forward —
        // either way, ascending-k keeps the secondary order stable).
        let routing_ids: Vec<i64> = vec![100, 99, 98, 97, 96, 95, 94, 93];
        let prev: HashSet<u32> = [93u32, 99, 96].iter().copied().collect();
        let perm = cross_layer_dispatch_order(&routing_ids, &prev);
        // Shared in router-score (ascending k): k=1 (99), k=4 (96), k=7 (93).
        // Non-shared in router-score: k=0, 2, 3, 5, 6.
        assert_eq!(perm, vec![1, 4, 7, 0, 2, 3, 5, 6]);
    }
}
