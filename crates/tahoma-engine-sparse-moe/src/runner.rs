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
use tahoma_int4_gemm::shell_int4::{
    shell_forward_decode_int4_multi_with_capacity, shell_forward_decode_int4_with_capacity,
    Int4Shell,
};
use tahoma_int4_gemm::{
    expert_forward as int4_expert_forward, expert_forward_multi as int4_expert_forward_multi,
    ExpertWeights, SafetensorsExpert, SafetensorsExpertSource,
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

    /// Per-layer expert dispatch with **input batching across tokens that
    /// share an expert**. Replaces the per-token, per-expert loop in
    /// [`Self::forward_shells_multi`] for shapes where seq > 1.
    ///
    /// **The ceiling this breaks.** With K=top_k=8 and seq=4 candidate
    /// tokens (the spec-decode verify width), the original dispatch does
    /// `seq * top_k = 32` `expert_forward` calls per layer, even when the
    /// 4 tokens collectively touch only ~6-12 distinct experts (high
    /// temporal locality in adjacent draft tokens — see iter 044 root
    /// cause). Each call loads ~21 MB of int4 weights from DRAM. Total
    /// weight motion per layer: 32 × 21 = ~670 MB / layer / step.
    ///
    /// The batched dispatch groups assignments by `eid`, calls
    /// `int4_expert_forward_multi` once per unique expert with the union
    /// of input rows, and scatters outputs back. For the same K=8 seq=4
    /// case at ~50% expert reuse (typical for the K2.6 router), weight
    /// motion drops to ~16 unique experts × 21 MB = ~336 MB / layer /
    /// step — a 2× reduction on the dominant cost.
    ///
    /// **Semantics.** The MoE accumulator output `moe[t * hidden + j] =
    /// sum_k w_tk * expert_output(eid_tk)[j]` is mathematically identical
    /// to the per-token loop. Because addition over the k axis is the
    /// same per-token operation, and `expert_forward_multi` is
    /// bit-near-identical to per-token `expert_forward` (kernel-level
    /// test in `tahoma-int4-gemm/src/kernel.rs`), the engine-level output
    /// is bit-near-identical to `forward_shells_multi` at the same
    /// `seq, past_seq_len, h_in` inputs.
    ///
    /// **Backend coverage.** Only the int4 backends
    /// (`int4_bin`, `safetensors_bin`) get the batched kernel call. The
    /// OvIr backend stays per-token (it's the slow legacy path; each call
    /// hits the OV CPU plugin one Linear at a time and there's no
    /// multi-token entry on that side).
    ///
    /// **Inputs.**
    /// - `lid`: MoE layer id (for expert weight lookup).
    /// - `attn_out_post_norm`: `[seq, hidden]` flat per-token expert inputs.
    /// - `routing_ids`: `[seq, top_k]` flat — token t's k-th routed expert id.
    /// - `routing_weights`: `[seq, top_k]` flat — corresponding weights.
    /// - `seq`: number of tokens in this batch.
    /// - `top_k`: experts routed per token.
    /// - `hidden`: hidden dim.
    ///
    /// **Output.** `[seq, hidden]` flat MoE accumulator, ready to be
    /// added to `attn_residual + shared_expert_out` to form `h_next`.
    fn dispatch_experts_batched(
        &mut self,
        lid: u32,
        attn_out_post_norm: &[f32],
        routing_ids: &[i64],
        routing_weights: &[f32],
        seq: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        debug_assert_eq!(attn_out_post_norm.len(), seq * hidden);
        debug_assert_eq!(routing_ids.len(), seq * top_k);
        debug_assert_eq!(routing_weights.len(), seq * top_k);

        // 1) Bucket assignments by eid. See `bucket_assignments` for the
        // semantics — pure function, unit-tested in isolation.
        let buckets = bucket_assignments(routing_ids, seq, top_k);

        // 2) Compute every unique expert's batched output. Storage scales
        // with the number of unique experts (typically 6-16 of 384 for
        // K2.6 spec-decode width 4-8).
        let mut expert_outs: HashMap<u32, Vec<bf16>> = HashMap::with_capacity(buckets.len());
        for (&eid, assigns) in &buckets {
            let n = assigns.len();
            let out = self.dispatch_expert_multi(lid, eid, attn_out_post_norm, assigns, hidden)?;
            debug_assert_eq!(out.len(), n * hidden);
            expert_outs.insert(eid, out);
        }

        // 3) Scatter into the MoE accumulator in [t, k] order. The fp
        // sum order matches the per-token reference loop exactly because
        // we accumulate `moe[t][j] += w_tk * y_tk[j]` in the same
        // (t outer, k inner) order. See `scatter_moe` for the
        // bookkeeping (also unit-tested).
        Ok(scatter_moe(
            routing_ids,
            routing_weights,
            &expert_outs,
            seq,
            top_k,
            hidden,
        ))
    }

    /// Run one expert over a batch of input rows (one per assignment in
    /// `assigns`). Returns `[assigns.len(), hidden]` flat bf16 outputs.
    ///
    /// **Backend dispatch.**
    /// - `Int4Bin` / `SafetensorsBin`: call `int4_expert_forward_multi`
    ///   once with the gathered inputs — this is where the win lives.
    /// - `OvIr`: fall back to N per-token calls of the OV runtime
    ///   (no multi-token entry on that side; legacy path only).
    ///
    /// `assigns[i].0` is the token index in `attn_out_post_norm`.
    /// `assigns[i].1` (the weight) is unused here — weights are applied
    /// at scatter time in the caller.
    fn dispatch_expert_multi(
        &mut self,
        lid: u32,
        eid: u32,
        attn_out_post_norm: &[f32],
        assigns: &[(usize, f32)],
        hidden: usize,
    ) -> Result<Vec<bf16>, RunnerError> {
        let n = assigns.len();
        debug_assert!(n >= 1);

        // Gather input rows for this expert into a contiguous
        // [n, hidden] bf16 buffer. (bf16 conversion is the input format
        // expected by both int4_expert_forward and int4_expert_forward_multi.)
        let mut xs_bf16 = vec![bf16::ZERO; n * hidden];
        for (i, &(t, _w)) in assigns.iter().enumerate() {
            let src = &attn_out_post_norm[t * hidden..(t + 1) * hidden];
            let dst = &mut xs_bf16[i * hidden..(i + 1) * hidden];
            for (j, &v) in src.iter().enumerate() {
                dst[j] = bf16::from_f32(v);
            }
        }
        let mut out_bf16 = vec![bf16::ZERO; n * hidden];

        match &mut self.experts {
            ExpertCache::Int4Bin(c) => {
                let w = c.get(lid, eid)?;
                if n == 1 {
                    // n=1: skip the multi tile's per-row scatter overhead.
                    // The auto dispatcher inside expert_forward_multi
                    // would already fall back here, but the explicit
                    // branch makes the seq=1 cost path obvious in profiles.
                    int4_expert_forward(
                        &xs_bf16,
                        w.gate_packed_bytes(),
                        w.gate_scale_bits(),
                        w.up_packed_bytes(),
                        w.up_scale_bits(),
                        w.down_packed_bytes(),
                        w.down_scale_bits(),
                        &mut out_bf16,
                    );
                } else {
                    int4_expert_forward_multi(
                        &xs_bf16,
                        w.gate_packed_bytes(),
                        w.gate_scale_bits(),
                        w.up_packed_bytes(),
                        w.up_scale_bits(),
                        w.down_packed_bytes(),
                        w.down_scale_bits(),
                        n,
                        &mut out_bf16,
                    );
                }
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
                if n == 1 {
                    int4_expert_forward(
                        &xs_bf16,
                        w.gate_packed,
                        w.gate_scale,
                        w.up_packed,
                        w.up_scale,
                        w.down_packed,
                        w.down_scale,
                        &mut out_bf16,
                    );
                } else {
                    int4_expert_forward_multi(
                        &xs_bf16,
                        w.gate_packed,
                        w.gate_scale,
                        w.up_packed,
                        w.up_scale,
                        w.down_packed,
                        w.down_scale,
                        n,
                        &mut out_bf16,
                    );
                }
            }
            ExpertCache::OvIr(_) => {
                // Legacy backend: no multi-token entry. Fall back to N
                // per-token dispatch_expert calls; we just upconvert
                // bf16 -> f32 each time. This keeps OvIr correct but
                // costs the OV plugin call overhead per token; if anyone
                // actually runs OvIr at seq>1 they should switch to int4.
                for (i, &(t, _w)) in assigns.iter().enumerate() {
                    let attn_row = &attn_out_post_norm[t * hidden..(t + 1) * hidden];
                    let y_f32 = self.dispatch_expert(lid, eid, attn_row)?;
                    let dst = &mut out_bf16[i * hidden..(i + 1) * hidden];
                    for (j, &v) in y_f32.iter().enumerate() {
                        dst[j] = bf16::from_f32(v);
                    }
                }
            }
        }

        Ok(out_bf16)
    }

    /// Iter-051 variant of [`Self::forward_shells_multi`] with
    /// **expert dispatch batched across tokens that share an expert**.
    ///
    /// Identical to `forward_shells_multi` except that the per-token,
    /// per-expert dispatch loop is replaced by
    /// [`Self::dispatch_experts_batched`]. See that method for the
    /// memory-motion / weight-reuse argument.
    ///
    /// **Why an additive seam.** The original `forward_shells_multi` is
    /// load-bearing (chunked-prefill / spec-decode-verify call it
    /// today). This variant ships alongside it so callers can opt in;
    /// switching the default is a separate change once the bench
    /// confirms the speedup.
    ///
    /// **Hot path preserved.** seq=1 delegates to [`Self::forward_shells`]
    /// — every today-shipping K2.6 inference is still bit-identical.
    /// seq>=2 takes the batched path.
    pub fn forward_shells_multi_batched_experts(
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
                "forward_shells_multi_batched_experts: seq must be >= 1".into(),
            ));
        }
        if h_shape.len() != 3 || h_shape[0] != 1 || h_shape[1] != seq || h_shape[2] != hidden {
            return Err(RunnerError::Internal(format!(
                "forward_shells_multi_batched_experts: int4 shells require shape [1, {seq}, {hidden}], got {h_shape:?}"
            )));
        }
        if h_in.len() != seq * hidden {
            return Err(RunnerError::Internal(format!(
                "forward_shells_multi_batched_experts: h_in.len={} != seq*hidden={}*{}={}",
                h_in.len(),
                seq,
                hidden,
                seq * hidden
            )));
        }
        // seq=1 hot-path: same delegation as forward_shells_multi.
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
            while past_seq_len + seq > self.layers[i].kv_capacity {
                grow_kv_capacity(&mut self.layers[i])?;
            }
            let capacity = self.layers[i].kv_capacity;
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

            if outs.routing_ids.len() != seq * top_k || outs.routing_weights.len() != seq * top_k {
                return Err(RunnerError::Internal(format!(
                    "L{lid} multi-token routing shape unexpected: ids={} weights={} (expected seq*top_k = {}*{})",
                    outs.routing_ids.len(),
                    outs.routing_weights.len(),
                    seq,
                    top_k
                )));
            }

            // The iter 051 lift: batched expert dispatch.
            let moe = self.dispatch_experts_batched(
                lid,
                &outs.attn_out_post_norm,
                &outs.routing_ids,
                &outs.routing_weights,
                seq,
                top_k,
                hidden,
            )?;

            // Per-token residual combine, identical to forward_shells_multi.
            for t in 0..seq {
                let attn_res_t = &outs.attn_residual[t * hidden..(t + 1) * hidden];
                let shared_t = &outs.shared_expert_out[t * hidden..(t + 1) * hidden];
                let h_next_t = &mut h_f32[t * hidden..(t + 1) * hidden];
                let moe_t = &moe[t * hidden..(t + 1) * hidden];
                for j in 0..hidden {
                    h_next_t[j] = attn_res_t[j] + shared_t[j] + moe_t[j];
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

// ----- iter 051 expert-batching dispatch helpers -----
//
// Free functions extracted from `Runner::dispatch_experts_batched` so the
// bookkeeping (bucket build, scatter order) can be unit-tested without
// constructing a real Runner + expert backend. The kernel-level
// correctness (expert_forward_multi == loop of expert_forward) is proved
// separately in `tahoma-int4-gemm/src/kernel.rs::tests`.

/// Build `HashMap<eid, Vec<(token_idx, weight)>>` from a `[seq, top_k]`
/// routing matrix. Sweep order is `t` outer, `k` inner, so assignment
/// insertion order inside each bucket is exactly `[t0_k0, t0_k1, ...,
/// t1_k0, ...]`. That ordering is load-bearing — `scatter_moe` relies
/// on it to map bucket positions back to (t, k) coordinates.
fn bucket_assignments(
    routing_ids: &[i64],
    seq: usize,
    top_k: usize,
) -> HashMap<u32, Vec<(usize, f32)>> {
    let mut buckets: HashMap<u32, Vec<(usize, f32)>> = HashMap::with_capacity(seq * top_k);
    for t in 0..seq {
        for k in 0..top_k {
            let eid = routing_ids[t * top_k + k] as u32;
            // weight isn't needed in the bucket itself for the current
            // scatter strategy (we look weights up at scatter time from
            // the original `routing_weights` slice), but we keep the
            // (t, w) tuple so the bucket carries enough info to drive
            // both gather (token_idx) and scatter (in callers that
            // want to keep weights co-located).
            buckets.entry(eid).or_default().push((t, 0.0));
        }
    }
    buckets
}

/// Scatter per-expert outputs back into a `[seq, hidden]` MoE accumulator.
///
/// `expert_outs[eid]` must be `[N_e * hidden]` flat, with row order
/// matching the `(t, k)` sweep order — same as what
/// [`bucket_assignments`] produces.
///
/// We iterate `(t, k)` and look up the row inside `expert_outs[eid]`
/// using a position counter per eid. Because the bucket insertion order
/// is the same `(t, k)` order, the position counter aligns one-to-one
/// with the assignment row.
///
/// **fp determinism.** The per-token reference loop in
/// `forward_shells_multi` accumulates:
///
/// ```text
/// moe[t][j] = sum_{k=0..top_k} w_tk * expert_output(eid_tk)[j]
/// ```
///
/// The sum is over `k` with `t` outer, so the addition order matters
/// only within each token. This scatter walks `(t, k)` in the same
/// order, applying `moe[t][j] += w_tk * y_tk[j]` for each assignment.
/// Result: bit-identical fp accumulation order to the per-token loop,
/// independent of expert_outs' compute order.
fn scatter_moe(
    routing_ids: &[i64],
    routing_weights: &[f32],
    expert_outs: &HashMap<u32, Vec<bf16>>,
    seq: usize,
    top_k: usize,
    hidden: usize,
) -> Vec<f32> {
    debug_assert_eq!(routing_ids.len(), seq * top_k);
    debug_assert_eq!(routing_weights.len(), seq * top_k);
    let mut moe = vec![0.0f32; seq * hidden];
    let mut bucket_pos: HashMap<u32, usize> = HashMap::with_capacity(expert_outs.len());
    for t in 0..seq {
        for k in 0..top_k {
            let eid = routing_ids[t * top_k + k] as u32;
            let w = routing_weights[t * top_k + k];
            let pos = bucket_pos.entry(eid).or_insert(0);
            let row_off = *pos * hidden;
            *pos += 1;
            let expert_out = expert_outs
                .get(&eid)
                .expect("bucket bookkeeping invariant: every routed eid has an output");
            let row = &expert_out[row_off..row_off + hidden];
            let moe_row = &mut moe[t * hidden..(t + 1) * hidden];
            for j in 0..hidden {
                moe_row[j] += w * row[j].to_f32();
            }
        }
    }
    moe
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

    // ----- iter 051 expert-batching dispatch tests -----
    //
    // Two regimes to guard:
    //   (A) bucket_assignments groups the (t, k) slots by eid in the
    //       deterministic (t outer, k inner) insertion order that
    //       scatter_moe relies on.
    //   (B) scatter_moe, given a per-expert output table, produces the
    //       same MoE accumulator that a straight per-token loop would —
    //       INCLUDING when 2+ tokens hit the same expert.

    /// Per-token reference loop. Mirrors the math in `forward_shells_multi`
    /// — `moe[t][j] = sum_k w_tk * expert_lookup(eid_tk)[t? no, the
    /// ROW INDEX matches the position inside the bucket]`. To get a
    /// fair comparison we use the same per-expert output table built by
    /// the bucketing path; the only difference is the loop nesting.
    fn per_token_reference(
        routing_ids: &[i64],
        routing_weights: &[f32],
        expert_outs: &HashMap<u32, Vec<bf16>>,
        seq: usize,
        top_k: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let mut moe = vec![0.0f32; seq * hidden];
        let mut bucket_pos: HashMap<u32, usize> = HashMap::new();
        for t in 0..seq {
            for k in 0..top_k {
                let eid = routing_ids[t * top_k + k] as u32;
                let w = routing_weights[t * top_k + k];
                let pos = bucket_pos.entry(eid).or_insert(0);
                let row_off = *pos * hidden;
                *pos += 1;
                let row = &expert_outs.get(&eid).unwrap()[row_off..row_off + hidden];
                for j in 0..hidden {
                    moe[t * hidden + j] += w * row[j].to_f32();
                }
            }
        }
        moe
    }

    /// Build a tiny per-expert output table where each output row is
    /// deterministically derived from (eid, bucket position).
    fn synthetic_expert_outs(
        buckets: &HashMap<u32, Vec<(usize, f32)>>,
        hidden: usize,
    ) -> HashMap<u32, Vec<bf16>> {
        let mut out: HashMap<u32, Vec<bf16>> = HashMap::new();
        for (&eid, assigns) in buckets {
            let n = assigns.len();
            let mut v = vec![bf16::ZERO; n * hidden];
            for (pos, _) in assigns.iter().enumerate() {
                for j in 0..hidden {
                    // Distinct value per (eid, pos, j) so bucket-position
                    // misalignment shows up as a wrong scatter.
                    let f = (eid as f32) * 0.01 + (pos as f32) * 0.1 + (j as f32) * 0.001;
                    v[pos * hidden + j] = bf16::from_f32(f);
                }
            }
            out.insert(eid, v);
        }
        out
    }

    #[test]
    fn bucket_assignments_groups_by_eid() {
        // seq=3, top_k=2, ids: [[5, 7], [5, 9], [7, 5]]
        // Expected buckets:
        //   5: [(0, _), (1, _), (2, _)]    // appears at (0,0), (1,0), (2,1)
        //   7: [(0, _), (2, _)]            // appears at (0,1), (2,0)
        //   9: [(1, _)]                    // appears at (1,1)
        let routing_ids = vec![5, 7, 5, 9, 7, 5];
        let buckets = bucket_assignments(&routing_ids, 3, 2);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[&5], vec![(0, 0.0), (1, 0.0), (2, 0.0)]);
        assert_eq!(buckets[&7], vec![(0, 0.0), (2, 0.0)]);
        assert_eq!(buckets[&9], vec![(1, 0.0)]);
    }

    #[test]
    fn bucket_assignments_seq_1_top_k_8_each_unique() {
        // seq=1, top_k=8, all distinct ids. Should produce 8 buckets each
        // with one assignment.
        let routing_ids: Vec<i64> = (0..8).collect();
        let buckets = bucket_assignments(&routing_ids, 1, 8);
        assert_eq!(buckets.len(), 8);
        for k in 0..8 {
            assert_eq!(buckets[&(k as u32)], vec![(0, 0.0)]);
        }
    }

    #[test]
    fn bucket_assignments_all_same_eid() {
        // seq=4, top_k=8, EVERY routing slot picks expert 42. Should
        // produce one bucket with 32 assignments in (t, k) order.
        let routing_ids = vec![42i64; 32];
        let buckets = bucket_assignments(&routing_ids, 4, 8);
        assert_eq!(buckets.len(), 1);
        let assigns = &buckets[&42];
        assert_eq!(assigns.len(), 32);
        for t in 0..4 {
            for k in 0..8 {
                // Check (t, k) order: position `t * 8 + k` should be (t, _).
                let (got_t, _) = assigns[t * 8 + k];
                assert_eq!(got_t, t, "wrong token at pos {}*8+{}", t, k);
            }
        }
    }

    #[test]
    fn scatter_moe_matches_per_token_seq_2_top_k_2_no_sharing() {
        // Sanity: when each (t, k) routes to a unique eid, the batched
        // scatter must produce the same MoE accumulator as the per-token
        // loop. (Trivial case — no expert sharing — but exercises the
        // weight + lookup wiring.)
        let routing_ids = vec![10, 20, 30, 40];
        let routing_weights = vec![0.5, 0.5, 0.25, 0.75];
        let hidden = 4;
        let buckets = bucket_assignments(&routing_ids, 2, 2);
        let expert_outs = synthetic_expert_outs(&buckets, hidden);
        let got = scatter_moe(&routing_ids, &routing_weights, &expert_outs, 2, 2, hidden);
        let want = per_token_reference(&routing_ids, &routing_weights, &expert_outs, 2, 2, hidden);
        assert_eq!(got, want);
    }

    #[test]
    fn scatter_moe_matches_per_token_seq_4_top_k_2_with_sharing() {
        // The interesting case: experts shared across tokens.
        // Routing: [[1, 2], [1, 3], [2, 1], [3, 2]]
        // Bucket sizes: 1 -> 3 tokens, 2 -> 3 tokens, 3 -> 2 tokens.
        let routing_ids = vec![1i64, 2, 1, 3, 2, 1, 3, 2];
        let routing_weights = vec![0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.45, 0.55];
        let hidden = 16;
        let buckets = bucket_assignments(&routing_ids, 4, 2);
        let expert_outs = synthetic_expert_outs(&buckets, hidden);
        let got = scatter_moe(&routing_ids, &routing_weights, &expert_outs, 4, 2, hidden);
        let want = per_token_reference(&routing_ids, &routing_weights, &expert_outs, 4, 2, hidden);
        assert_eq!(
            got, want,
            "expert-sharing scatter doesn't match per-token reference"
        );
    }

    #[test]
    fn scatter_moe_matches_per_token_seq_4_top_k_8_random_sharing() {
        // K2.6-flavored shape: seq=4 (spec-decode width), top_k=8, with
        // ~50% expert reuse simulated by drawing eids from {0..16}.
        let seq = 4;
        let top_k = 8;
        let hidden = 32;
        let mut routing_ids = vec![0i64; seq * top_k];
        let mut routing_weights = vec![0.0f32; seq * top_k];
        // Deterministic seed.
        for t in 0..seq {
            for k in 0..top_k {
                let eid = (((t * 13 + k * 7) % 16) + 1) as i64;
                let w = 0.1 + ((t * 17 + k * 3) % 7) as f32 * 0.05;
                routing_ids[t * top_k + k] = eid;
                routing_weights[t * top_k + k] = w;
            }
        }
        let buckets = bucket_assignments(&routing_ids, seq, top_k);
        // Confirm we got reuse (else the test isn't actually exercising
        // the batching path).
        assert!(
            buckets.len() < seq * top_k,
            "expected expert reuse: got {} unique experts of {} slots",
            buckets.len(),
            seq * top_k
        );
        let expert_outs = synthetic_expert_outs(&buckets, hidden);
        let got = scatter_moe(
            &routing_ids,
            &routing_weights,
            &expert_outs,
            seq,
            top_k,
            hidden,
        );
        let want = per_token_reference(
            &routing_ids,
            &routing_weights,
            &expert_outs,
            seq,
            top_k,
            hidden,
        );
        assert_eq!(
            got, want,
            "K2.6-shape (seq=4 top_k=8) scatter doesn't match per-token reference"
        );
    }

    #[test]
    fn scatter_moe_matches_per_token_all_same_expert() {
        // Extreme case: every (t, k) routes to the same expert. Scatter
        // must still produce the right per-token accumulation.
        let seq = 3;
        let top_k = 4;
        let hidden = 8;
        let routing_ids = vec![7i64; seq * top_k];
        let routing_weights: Vec<f32> = (0..(seq * top_k)).map(|i| (i + 1) as f32 * 0.1).collect();
        let buckets = bucket_assignments(&routing_ids, seq, top_k);
        let expert_outs = synthetic_expert_outs(&buckets, hidden);
        let got = scatter_moe(
            &routing_ids,
            &routing_weights,
            &expert_outs,
            seq,
            top_k,
            hidden,
        );
        let want = per_token_reference(
            &routing_ids,
            &routing_weights,
            &expert_outs,
            seq,
            top_k,
            hidden,
        );
        assert_eq!(got, want);
    }
}
