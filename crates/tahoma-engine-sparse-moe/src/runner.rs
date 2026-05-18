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
            top_k_override: None,
            routing_threshold: None,
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

    /// Set the per-token routing-weight threshold (A2). None / 0.0 = disabled.
    /// Experts whose routing weight is < threshold are skipped during
    /// forward_shells dispatch.
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
            // autolab campaign 007 (A2): apply routing-weight threshold.
            // We still iterate over `effective_top_k` to honor the A3 cap,
            // but skip experts below the threshold within that range.
            let threshold = self.routing_threshold.unwrap_or(0.0);
            for k in 0..effective_top_k {
                let w = outs.routing_weights[k];
                if w < threshold {
                    continue;
                }
                let eid = outs.routing_ids[k] as u32;
                let y_f32 = self.dispatch_expert(lid, eid, &outs.attn_out_post_norm)?;
                for j in 0..hidden {
                    moe[j] += w * y_f32[j];
                }
            }
            experts_total_us += experts_t0.elapsed().as_micros() as u64;

            // Combine: h_next = residual + shared + moe (single token).
            let combine_t0 = Instant::now();
            for j in 0..hidden {
                h_f32[j] = outs.attn_residual[j] + outs.shared_expert_out[j] + moe[j];
            }
            combine_total_us += combine_t0.elapsed().as_micros() as u64;
        }

        // autolab/k26-perf q1 instrumentation: per-token shells breakdown.
        info!(
            stage = "shells",
            n_layers,
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
    /// the 22 ms cross-host RT measured on the cascadia fleet, K=8 with
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

            // 2. Verify drafts one-by-one through the target forward.
            // The int4 shell only supports seq=1, so this runs K
            // sequential single-token forwards — NOT a batched K-pass.
            // (Adding a multi-token shell forward is the next perf
            // unlock; see [`crate::spec_decode`] module docs.) Each
            // forward conditions on the previous drafted token in
            // history and produces the target's prediction for the
            // next position. We compare predictions vs drafts below.
            n_drafts_total += drafts.len() as u32;
            let mut target_samples: Vec<i64> = Vec::with_capacity(drafts.len() + 1);
            for &draft_tok in drafts.iter() {
                let logits = self.step(&history, 1)?;
                target_samples.push(argmax_i64(&logits));
                // Push the drafted token into history regardless of
                // whether it will eventually be accepted — the next
                // forward needs it to condition correctly. We trim
                // back any rejected suffix below.
                history.push(draft_tok);
            }

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
}
