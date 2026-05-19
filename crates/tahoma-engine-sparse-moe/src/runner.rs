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
use tracing::{debug, info, warn};

use crate::kv_prefix_cache::{KvPrefixCache, KvSnapshot, LayerKvSlice, ModelFingerprint};
use crate::kv_session_cache::KvSessionCache;
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
        })
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
            let mut moe = vec![0.0f32; hidden];
            for k in 0..top_k {
                let eid = outs.routing_ids[k] as u32;
                let w = outs.routing_weights[k];
                let y_f32 = self.dispatch_expert(lid, eid, &outs.attn_out_post_norm)?;
                for j in 0..hidden {
                    moe[j] += w * y_f32[j];
                }
            }

            // Combine: h_next = residual + shared + moe (single token).
            for j in 0..hidden {
                h_f32[j] = outs.attn_residual[j] + outs.shared_expert_out[j] + moe[j];
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
        self.generate_with_caches(prompt_ids, max_tokens, cfg, None, None, None)
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
    ///
    /// Convenience wrapper around [`Self::generate_with_caches`] for
    /// callers that only have a prefix cache (no session-cache).
    pub fn generate_with_cache(
        &mut self,
        prompt_ids: &[i64],
        max_tokens: usize,
        cfg: &crate::sampling::SamplingConfig,
        cache: Option<&mut KvPrefixCache>,
    ) -> Result<Vec<i64>, RunnerError> {
        self.generate_with_caches(prompt_ids, max_tokens, cfg, cache, None, None)
    }

    /// Full generate path with both caches.
    ///
    /// - `prefix_cache` (iter 060) — shared-prefix cache keyed on the
    ///   token sequence. Hit when many requests share a system prompt
    ///   prefix; snapshot taken AFTER PREFILL.
    /// - `session_cache` + `session_id` (iter 072) — multi-turn chat
    ///   warm-restart cache keyed on the caller-supplied session id.
    ///   Hit when the request prompt extends a previously-snapshotted
    ///   session's `system+user+assistant…` history; snapshot taken
    ///   AFTER GENERATION (so it includes the assistant's reply, ready
    ///   for the next turn's user message to append).
    ///
    /// When both caches could fire, we pick the LONGER match — the
    /// session cache will usually win on turn ≥ 2 of a conversation
    /// (it has the whole prior history); the prefix cache wins on
    /// the very first turn when only the system prompt is shared.
    /// Exactly one restore happens, so we never pay restore cost twice.
    ///
    /// **Bit-identity contract**: with both caches passed but `None`
    /// returned by every lookup, this method's behaviour is byte-
    /// identical to the cache-disabled path. With a hit, the restored
    /// KV bits are byte-identical to what a full re-prefill would have
    /// written (the snapshot was packed from those exact KV buffers
    /// in a prior generation; the runner's snapshot/restore round-trip
    /// is unit-tested at `pack_unpack_roundtrip_is_bit_identical`).
    pub fn generate_with_caches(
        &mut self,
        prompt_ids: &[i64],
        max_tokens: usize,
        cfg: &crate::sampling::SamplingConfig,
        mut prefix_cache: Option<&mut KvPrefixCache>,
        mut session_cache: Option<&mut KvSessionCache>,
        session_id: Option<&str>,
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

        let fp = self.fingerprint();

        // --- Cache lookup ----------------------------------------------------
        //
        // Consult both caches independently, pick the longer match. We
        // probe the session cache first because it usually wins on turn
        // ≥ 2 (includes generated tokens from the previous turn), so
        // on a session hit we can skip touching the prefix cache at
        // all. But the prefix cache might still hold a longer match in
        // edge cases (e.g. shared system prompt + user message identical
        // across two distinct sessions), so we honestly compare lengths
        // and pick the bigger one. Restoring is the expensive part —
        // doing it twice would erase the cache's value.
        let mut chosen_skip: usize = 0;
        let mut chosen_source: &'static str = "miss";
        let mut chosen_snap: Option<KvSnapshot> = None;

        if let (Some(sid), Some(sc)) = (session_id, session_cache.as_mut()) {
            if sc.enabled() {
                if let Some(hit) = sc.lookup(sid, prompt_ids, &fp) {
                    chosen_skip = hit.matched_len;
                    chosen_snap = Some(hit.snapshot);
                    chosen_source = "session";
                }
            }
        }
        if let Some(pc) = prefix_cache.as_mut() {
            if pc.enabled() {
                if let Some(snap) = pc.lookup(prompt_ids, &fp) {
                    if snap.past_seq_len > chosen_skip {
                        chosen_skip = snap.past_seq_len;
                        chosen_snap = Some(snap);
                        chosen_source = "prefix";
                    }
                }
            }
        }

        let mut cache_hit = false;
        let mut prefix_hit = false; // tracks whether to skip prefix-cache insert
        if let Some(snap) = chosen_snap {
            // Validates shape against fingerprint defensively; an
            // error here means cache + runner disagree on dimensions,
            // which the digest check upstream should prevent — surface
            // the bug rather than silently corrupt.
            self.restore_kv(&snap)?;
            cache_hit = true;
            prefix_hit = chosen_source == "prefix";
            info!(
                source = chosen_source,
                cached_prefix_len = chosen_skip,
                full_prompt_len = prompt_ids.len(),
                "kv-cache HIT — skipping prefix tokens"
            );
        }
        let cache_skip = chosen_skip;

        // --- Prefill ---------------------------------------------------------
        //
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

        // --- Prefix-cache insert on miss ------------------------------------
        //
        // Insert a snapshot on miss. We only cache the FULL prompt
        // (not arbitrary intermediate prefixes) — this keeps the
        // cache small and matches the chat-completion access pattern
        // where the system+user prompt is stable across requests but
        // the user message varies. Insert happens after prefill so
        // the snapshot reflects what `forward_shells` actually wrote.
        //
        // Skip the insert when the hit came from the prefix cache
        // itself — the entry is already there (just promoted by the
        // lookup). Skip on a session hit too; the prefix cache would
        // get a different entry for the same KV bits but the
        // marginal value is low (session snapshot will be more
        // up-to-date next turn anyway).
        //
        // We could be cleverer (cache after each step, or at
        // configurable boundaries) — defer until profiling shows
        // demand. The dominant cost on the rainier baseline is
        // prefill, and a full-prompt snapshot pays it off in one hit.
        if !cache_hit {
            if let Some(c) = prefix_cache.as_mut() {
                if c.enabled() && !prompt_ids.is_empty() {
                    match self.snapshot_kv() {
                        Ok(snap) => {
                            let bytes = snap.approx_bytes();
                            let evicted = c.insert(prompt_ids.to_vec(), &fp, snap);
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
        // Avoid unused-warning if the only consumer of `prefix_hit` is
        // gated on future tracing. It's load-bearing once we add a
        // metric for "hit by source" but is kept here as the boolean
        // we tested above.
        let _ = prefix_hit;

        // --- Generate --------------------------------------------------------
        // First generated token from the LAST prefill step's logits.
        if let Some(l) = last_logits {
            let next = crate::sampling::sample(&l, &history, cfg, &mut rng);
            if eos.contains(&next) {
                // EOS on first generated token: we still insert a
                // session snapshot so a subsequent turn that begins
                // with this exact (prompt+EOS-or-nothing) state can
                // warm-restart. The snapshot at this point covers
                // `prompt_ids` (no generated tokens were appended to
                // history before EOS was returned).
                self.maybe_insert_session_snapshot(
                    session_id,
                    session_cache.as_deref_mut(),
                    &fp,
                    &history,
                );
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

        // --- Session-cache insert at end of generation ----------------------
        //
        // Snapshot the full `prompt + generated` history. This is the
        // critical move for multi-turn warm-restart: the next turn's
        // prompt will be `prompt + generated + new_user_message`, and
        // looking up the same session id with that new prompt will
        // hit on the prefix `prompt + generated` (matched_len ==
        // history.len()), letting the runner skip ALL of it.
        //
        // We always insert (even on a hit) because `history` has more
        // tokens now than the snapshot we restored from — the new
        // entry supersedes the old. The session cache's insert handles
        // replacement of the same session id in place.
        self.maybe_insert_session_snapshot(session_id, session_cache.as_deref_mut(), &fp, &history);
        Ok(generated)
    }

    /// Snapshot the runner's current KV state and insert it into the
    /// session cache under `session_id`. No-op when no session_id was
    /// supplied or the cache is disabled. Snapshot failures are
    /// logged but non-fatal (the generation has already succeeded).
    fn maybe_insert_session_snapshot(
        &self,
        session_id: Option<&str>,
        session_cache: Option<&mut KvSessionCache>,
        fingerprint: &ModelFingerprint,
        history: &[i64],
    ) {
        let Some(sid) = session_id else {
            return;
        };
        let Some(sc) = session_cache else {
            return;
        };
        if !sc.enabled() || history.is_empty() {
            return;
        }
        match self.snapshot_kv() {
            Ok(snap) => {
                let bytes = snap.approx_bytes();
                let evicted = sc.insert(sid.to_string(), history.to_vec(), fingerprint, snap);
                info!(
                    session_id = sid,
                    cached_bytes = bytes,
                    cache_len = sc.len(),
                    total_bytes = sc.total_bytes(),
                    evicted,
                    "kv-session-cache inserted snapshot"
                );
            }
            Err(e) => {
                // Snapshot failure is non-fatal — the generation has
                // already completed correctly; we just won't be able
                // to warm-restart the next turn.
                warn!(
                    session_id = sid,
                    error = %e,
                    "kv-session-cache snapshot failed; not caching"
                );
            }
        }
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

/// Copy a layer's populated KV prefix (`[NUM_HEADS, past_seq, head_dim]`)
/// out of its capacity buffer (`[NUM_HEADS, capacity, head_dim]`) into
/// a fresh packed `Vec<f32>` suitable for serialization. The capacity
/// buffer's per-head bases shift on grow; packing strips that variance
/// so a snapshot taken at cap=32 restores correctly into cap=64.
fn pack_layer_slice(
    lid: u32,
    past_k: &[f32],
    past_v: &[f32],
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
    past_k: &mut [f32],
    past_v: &mut [f32],
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
        let mut k = vec![0.0f32; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut v = vec![0.0f32; NUM_HEADS * cap * V_HEAD_DIM];
        // Unique signature per cell so any mis-indexing surfaces.
        for h in 0..NUM_HEADS {
            for s in 0..past {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    k[off] = (h * 100_000 + s * 1_000 + d) as f32;
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    v[off] = -((h * 100_000 + s * 1_000 + d) as f32);
                }
            }
        }
        let slice = pack_layer_slice(7, &k, &v, past, cap);
        assert_eq!(slice.lid, 7);
        assert_eq!(slice.past_k.len(), NUM_HEADS * past * QK_HEAD_DIM);
        assert_eq!(slice.past_v.len(), NUM_HEADS * past * V_HEAD_DIM);
        // Unpack into a fresh capacity-buffer.
        let mut k2 = vec![f32::NAN; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut v2 = vec![f32::NAN; NUM_HEADS * cap * V_HEAD_DIM];
        unpack_layer_slice(&mut k2, &mut v2, &slice, past, cap).expect("unpack");
        // Populated prefix must be bit-identical.
        for h in 0..NUM_HEADS {
            for s in 0..past {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    assert_eq!(
                        k2[off].to_bits(),
                        k[off].to_bits(),
                        "k[h={h}, s={s}, d={d}] mismatch"
                    );
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    assert_eq!(
                        v2[off].to_bits(),
                        v[off].to_bits(),
                        "v[h={h}, s={s}, d={d}] mismatch"
                    );
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
        let mut k = vec![0.0f32; NUM_HEADS * src_cap * QK_HEAD_DIM];
        let mut v = vec![0.0f32; NUM_HEADS * src_cap * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..past {
                k[h * src_cap * QK_HEAD_DIM + s * QK_HEAD_DIM] = (h * 100 + s) as f32;
                v[h * src_cap * V_HEAD_DIM + s * V_HEAD_DIM] = -((h * 100 + s) as f32);
            }
        }
        let slice = pack_layer_slice(3, &k, &v, past, src_cap);
        let dst_cap = 16;
        let mut k2 = vec![f32::NAN; NUM_HEADS * dst_cap * QK_HEAD_DIM];
        let mut v2 = vec![f32::NAN; NUM_HEADS * dst_cap * V_HEAD_DIM];
        unpack_layer_slice(&mut k2, &mut v2, &slice, past, dst_cap).expect("unpack");
        for h in 0..NUM_HEADS {
            for s in 0..past {
                let k_off = h * dst_cap * QK_HEAD_DIM + s * QK_HEAD_DIM;
                let v_off = h * dst_cap * V_HEAD_DIM + s * V_HEAD_DIM;
                assert_eq!(k2[k_off], (h * 100 + s) as f32);
                assert_eq!(v2[v_off], -((h * 100 + s) as f32));
            }
            // Slots past..dst_cap should still be NaN — never read.
            for s in past..dst_cap {
                let k_off = h * dst_cap * QK_HEAD_DIM + s * QK_HEAD_DIM;
                assert!(k2[k_off].is_nan(), "unpack wrote into reserved slot");
            }
        }
    }

    #[test]
    fn unpack_rejects_wrong_length_slice() {
        let cap = 4;
        let past = 2;
        let mut k = vec![0.0f32; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut v = vec![0.0f32; NUM_HEADS * cap * V_HEAD_DIM];
        // Snapshot claims past=2 but past_k length only matches past=1.
        let bad = LayerKvSlice {
            lid: 0,
            past_k: vec![0.0f32; NUM_HEADS * QK_HEAD_DIM], // claims past=1 (deliberately wrong)
            past_v: vec![0.0f32; NUM_HEADS * past * V_HEAD_DIM],
        };
        let err = unpack_layer_slice(&mut k, &mut v, &bad, past, cap)
            .expect_err("expected length-check error");
        let msg = format!("{err}");
        assert!(msg.contains("past_k"), "error should mention past_k: {msg}");
    }

    /// Multi-turn session bit-identity. Simulates two scenarios that
    /// MUST produce identical KV bits after the session cache settles:
    ///
    /// 1. Cold-path "no cache": one big prefill that fills slots 0..N.
    /// 2. Warm-path "session restore": prefill 0..K, pack a snapshot,
    ///    unpack it back into a FRESH buffer at the same capacity,
    ///    then prefill K..N. The final populated K and V slots must
    ///    match cold-path byte-for-byte.
    ///
    /// This is the load-bearing invariant the task brief calls out
    /// for the session cache: "cached-session vs non-cached must
    /// produce identical tokens". With a deterministic forward
    /// function (here: a hash-of-(layer,slot,h,d) stamp), identical
    /// KV bits ⇒ identical attention outputs ⇒ identical logits ⇒
    /// identical sampled tokens.
    #[test]
    fn session_round_trip_matches_full_prefill_bit_identical() {
        let cap = 16;
        let prefix_len = 6; // "turn 1 prompt + assistant reply"
        let suffix_len = 4; // "turn 2 user message"
        let total = prefix_len + suffix_len;

        // Cold path: fill slots 0..total directly. The stamp function
        // mimics a deterministic forward: cell(h,s,d) = h*1e6 + s*1e3 + d.
        let mut cold_k = vec![0.0f32; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut cold_v = vec![0.0f32; NUM_HEADS * cap * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..total {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    cold_k[off] = (h * 1_000_000 + s * 1_000 + d) as f32;
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    cold_v[off] = -((h * 1_000_000 + s * 1_000 + d) as f32);
                }
            }
        }

        // Warm path turn 1: fill slots 0..prefix_len, then pack
        // (mimics end-of-turn-1 snapshot insertion).
        let mut warm_k = vec![0.0f32; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut warm_v = vec![0.0f32; NUM_HEADS * cap * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..prefix_len {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    warm_k[off] = (h * 1_000_000 + s * 1_000 + d) as f32;
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    warm_v[off] = -((h * 1_000_000 + s * 1_000 + d) as f32);
                }
            }
        }
        let snapshot = pack_layer_slice(0, &warm_k, &warm_v, prefix_len, cap);

        // Turn 2: start from a FRESH buffer (mimics a stateless
        // engine that just woke up from cache). Restore the snapshot,
        // then continue filling slots prefix_len..total.
        let mut warm2_k = vec![0.0f32; NUM_HEADS * cap * QK_HEAD_DIM];
        let mut warm2_v = vec![0.0f32; NUM_HEADS * cap * V_HEAD_DIM];
        unpack_layer_slice(&mut warm2_k, &mut warm2_v, &snapshot, prefix_len, cap)
            .expect("restore");
        // "Suffix prefill" — write the new tokens' KV cells.
        for h in 0..NUM_HEADS {
            for s in prefix_len..total {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    warm2_k[off] = (h * 1_000_000 + s * 1_000 + d) as f32;
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    warm2_v[off] = -((h * 1_000_000 + s * 1_000 + d) as f32);
                }
            }
        }

        // Bit-identity: every populated slot of warm2 must match cold
        // exactly. We don't compare slots [total..cap] — those were
        // never read by the forward path; their contents are
        // intentionally undefined (the warm path may have ZEROS where
        // cold has ZEROS, but in production both are NaN-ish).
        for h in 0..NUM_HEADS {
            for s in 0..total {
                for d in 0..QK_HEAD_DIM {
                    let off = h * cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    assert_eq!(
                        warm2_k[off].to_bits(),
                        cold_k[off].to_bits(),
                        "K mismatch at h={h} s={s} d={d}: warm={} cold={}",
                        warm2_k[off],
                        cold_k[off]
                    );
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    assert_eq!(
                        warm2_v[off].to_bits(),
                        cold_v[off].to_bits(),
                        "V mismatch at h={h} s={s} d={d}: warm={} cold={}",
                        warm2_v[off],
                        cold_v[off]
                    );
                }
            }
        }
    }

    /// Same multi-turn round-trip as above, but the warm path also
    /// crosses a grow_kv_capacity boundary (snapshot taken at cap=8,
    /// restored into a cap=16 buffer, then suffix prefill). The
    /// per-head bases shift after a grow; this test catches any
    /// indexing mis-offset that would silently corrupt KV bits when
    /// a long-running session crosses the initial capacity threshold.
    #[test]
    fn session_round_trip_survives_capacity_grow() {
        let src_cap = 8;
        let dst_cap = 16;
        let prefix_len = 5;
        let suffix_len = 6;
        let total = prefix_len + suffix_len;

        // Turn 1 buffer at src_cap.
        let mut warm_k = vec![0.0f32; NUM_HEADS * src_cap * QK_HEAD_DIM];
        let mut warm_v = vec![0.0f32; NUM_HEADS * src_cap * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..prefix_len {
                for d in 0..QK_HEAD_DIM {
                    warm_k[h * src_cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d] =
                        (h * 1_000_000 + s * 1_000 + d) as f32;
                }
                for d in 0..V_HEAD_DIM {
                    warm_v[h * src_cap * V_HEAD_DIM + s * V_HEAD_DIM + d] =
                        -((h * 1_000_000 + s * 1_000 + d) as f32);
                }
            }
        }
        let snapshot = pack_layer_slice(0, &warm_k, &warm_v, prefix_len, src_cap);

        // Turn 2 buffer at dst_cap (= 2 × src_cap, simulating one
        // grow_kv_capacity doubling).
        let mut warm2_k = vec![0.0f32; NUM_HEADS * dst_cap * QK_HEAD_DIM];
        let mut warm2_v = vec![0.0f32; NUM_HEADS * dst_cap * V_HEAD_DIM];
        unpack_layer_slice(&mut warm2_k, &mut warm2_v, &snapshot, prefix_len, dst_cap)
            .expect("restore");
        for h in 0..NUM_HEADS {
            for s in prefix_len..total {
                for d in 0..QK_HEAD_DIM {
                    warm2_k[h * dst_cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d] =
                        (h * 1_000_000 + s * 1_000 + d) as f32;
                }
                for d in 0..V_HEAD_DIM {
                    warm2_v[h * dst_cap * V_HEAD_DIM + s * V_HEAD_DIM + d] =
                        -((h * 1_000_000 + s * 1_000 + d) as f32);
                }
            }
        }

        // The cold-path equivalent: a single dst_cap buffer filled
        // 0..total. Both must match cell-by-cell across the populated
        // prefix AND suffix slots.
        let mut cold_k = vec![0.0f32; NUM_HEADS * dst_cap * QK_HEAD_DIM];
        let mut cold_v = vec![0.0f32; NUM_HEADS * dst_cap * V_HEAD_DIM];
        for h in 0..NUM_HEADS {
            for s in 0..total {
                for d in 0..QK_HEAD_DIM {
                    cold_k[h * dst_cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d] =
                        (h * 1_000_000 + s * 1_000 + d) as f32;
                }
                for d in 0..V_HEAD_DIM {
                    cold_v[h * dst_cap * V_HEAD_DIM + s * V_HEAD_DIM + d] =
                        -((h * 1_000_000 + s * 1_000 + d) as f32);
                }
            }
        }
        for h in 0..NUM_HEADS {
            for s in 0..total {
                for d in 0..QK_HEAD_DIM {
                    let off = h * dst_cap * QK_HEAD_DIM + s * QK_HEAD_DIM + d;
                    assert_eq!(warm2_k[off].to_bits(), cold_k[off].to_bits());
                }
                for d in 0..V_HEAD_DIM {
                    let off = h * dst_cap * V_HEAD_DIM + s * V_HEAD_DIM + d;
                    assert_eq!(warm2_v[off].to_bits(), cold_v[off].to_bits());
                }
            }
        }
    }
}
