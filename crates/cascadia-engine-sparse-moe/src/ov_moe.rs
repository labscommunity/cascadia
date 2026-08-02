//! OV-IR sparse-MoE pipeline for architecture-in-graph models (MiniMax-M2).
//!
//! Unlike the K2.6 path ([`crate::runner::Runner`]), which runs a hardcoded
//! MLA attention kernel in Rust and reads shell weights from safetensors,
//! this backend keeps the runtime architecture-agnostic: every model detail
//! (GQA, full-width QK-norm, partial RoPE, sigmoid routing with
//! `e_score_correction_bias`, SwiGLU experts) lives in the OpenVINO graphs
//! produced by `tools/export_minimax_m2.py`. The runtime only:
//!
//!   1. embeds the token (`layer0` IR: ids -> hidden),
//!   2. runs each layer's shell IR (the 7-tensor contract), threading a
//!      per-layer KV cache in/out,
//!   3. dispatches the top-k experts the shell selected and combines
//!      `attn_residual + shared_expert_out + Σ wₖ·expertₖ(attn_out_post_norm)`,
//!   4. runs the head IR (final norm + lm_head); the caller samples the
//!      next token (greedy / repetition-penalty / temperature).
//!
//! Experts run through one of two backends (manifest `experts_format`):
//! per-expert OpenVINO IR (`ov_ir`) or flat int4 binaries fed to the
//! cascadia-int4-gemm AVX-512 kernel (`int4_bin`, no per-call OV overhead).
//!
//! **Intel iGPU note:** the MoE router subgraph (sigmoid + TopK + Gather)
//! makes the OpenVINO GPU plugin emit a kernel that fails at runtime with
//! `CL_OUT_OF_RESOURCES` (it can wedge the OpenCL context); the attention/KV
//! core runs on the iGPU fine. So when targeting a non-CPU device and the
//! shells have been split by `tools/split_m2_shells.py`, this runner loads
//! the GPU-friendly `shell_core` on the device and the carved `router` on
//! the CPU plugin (the router is trivial compute). See [`OvMoeRunner`].
//!
//! Because the graphs carry their own shapes, this path is fully
//! dimension-agnostic — the same code runs the tiny synthetic M2 used by
//! the correctness test and the full 230B model. Single-stage (whole model)
//! and pipeline-parallel (a contiguous layer slice per rank, via
//! [`OvMoeRunner::load_staged`] + [`OvMoeRunner::embed_token`] /
//! [`OvMoeRunner::forward_layers`] / [`OvMoeRunner::head_logits`]) share
//! this same per-layer math.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use cascadia_int4_gemm::expert_forward_sparse;
use half::bf16;
use lru::LruCache;
use tracing::info;

use cascadia_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};

use crate::kv_prefix_cache::ModelFingerprint;
use crate::manifest::{Manifest, ManifestError};
use crate::ov_kv_cache::{OvLayerKvSlice, OvMoeKvPrefixCache, OvMoeKvSnapshot};
use crate::sampling::{init_rng, sample, SamplingConfig};

/// Group size of the int4_bin quantization (cols per scale group).
const INT4_GROUP: usize = 32;
/// Near-zero FFN-sparsity threshold: routes through the dim-generic sparse
/// kernel (the dense fallback is hardcoded to K2.6 dims) while keeping
/// effectively all activations (cutoff = τ·max_abs ≈ 0).
const INT4_KEEP_ALL_TAU: f32 = 1e-9;

/// Expert weight backend for [`OvMoeRunner`].
enum ExpertCache {
    /// One compiled OpenVINO model per (layer, expert). Simple but pays
    /// the OV CPU-plugin per-call overhead.
    OvIr(LruCache<(u32, u32), Runtime>),
    /// Flat int4-packed expert binaries (compressed-tensors layout) run
    /// through the cascadia-int4-gemm AVX-512 kernel. No per-call OV
    /// overhead. Cache holds the raw file bytes (mmap-free; bounded by the
    /// LRU). See [`crate::ov_moe`] module docs for the on-disk layout.
    Int4Bin(LruCache<(u32, u32), Arc<Vec<u8>>>),
}

#[derive(Debug, thiserror::Error)]
pub enum OvMoeError {
    #[error("manifest: {0}")]
    Manifest(#[from] ManifestError),
    #[error("ov: {0}")]
    Ov(#[from] OvError),
    #[error("missing file: {0}")]
    MissingFile(PathBuf),
    #[error("shell output {0:?} not found in graph (have {1:?})")]
    MissingOutput(String, Vec<String>),
    #[error("internal: {0}")]
    Internal(String),
}

// ---- little-endian byte helpers (no extra deps) -------------------------
fn f32_as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit-patterns for transmute-to-bytes,
    // and the lifetime is tied to the input slice.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_i64(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// Read an output port by one of its tensor aliases (set by the exporter).
fn out_idx(rt: &Runtime, name: &str) -> Result<usize, OvMoeError> {
    for i in 0..rt.output_count() {
        if let Ok(aliases) = rt.output_aliases(i) {
            if aliases.iter().any(|a| a == name) {
                return Ok(i);
            }
        }
    }
    let have = rt.output_names().unwrap_or_default();
    Err(OvMoeError::MissingOutput(name.to_string(), have))
}

/// Output-port indices for the attention/KV part of a shell — present in
/// both the monolithic shell and the GPU-friendly `shell_core` split.
#[derive(Clone, Copy)]
struct CorePorts {
    apn: usize,
    residual: usize,
    shared: usize,
    present_k: usize,
    present_v: usize,
}

impl CorePorts {
    fn resolve(rt: &Runtime) -> Result<Self, OvMoeError> {
        Ok(Self {
            apn: out_idx(rt, "attn_out_post_norm")?,
            residual: out_idx(rt, "attn_residual")?,
            shared: out_idx(rt, "shared_expert_out")?,
            present_k: out_idx(rt, "present_k")?,
            present_v: out_idx(rt, "present_v")?,
        })
    }
}

/// Output-port indices for the MoE router (top-k ids + weights). Resolved
/// from the monolithic shell (CPU path) or the carved `router` graph (the
/// non-CPU split, where the router runs on the CPU plugin).
#[derive(Clone, Copy)]
struct RouterPorts {
    ids: usize,
    weights: usize,
}

impl RouterPorts {
    fn resolve(rt: &Runtime) -> Result<Self, OvMoeError> {
        Ok(Self {
            ids: out_idx(rt, "routing_ids")?,
            weights: out_idx(rt, "routing_weights")?,
        })
    }
}

/// Per-layer running KV cache, stored as flat `[KV, P, D]` row-major f32.
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    seq: usize,
}

impl LayerKv {
    fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            seq: 0,
        }
    }
    /// Append this token's present_k/v (`[KV,1,D]`, order h*D+d) onto the
    /// `[KV,P,D]` cache, producing `[KV,P+1,D]`.
    fn append(&mut self, present_k: &[f32], present_v: &[f32], kv: usize, d: usize) {
        let p = self.seq;
        let mut nk = Vec::with_capacity(kv * (p + 1) * d);
        let mut nv = Vec::with_capacity(kv * (p + 1) * d);
        for h in 0..kv {
            nk.extend_from_slice(&self.k[h * p * d..h * p * d + p * d]);
            nk.extend_from_slice(&present_k[h * d..h * d + d]);
            nv.extend_from_slice(&self.v[h * p * d..h * p * d + p * d]);
            nv.extend_from_slice(&present_v[h * d..h * d + d]);
        }
        self.k = nk;
        self.v = nv;
        self.seq = p + 1;
    }
    fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
        self.seq = 0;
    }
}

/// Per-generation timing. `decode_secs[i]` is the wall time of the i-th
/// decode step (producing the logits for generated token i+1). On the
/// ov_ir expert backend the first few entries are dominated by cold
/// expert compilation; the tail is warm steady-state.
#[derive(Default, Clone, Debug)]
pub struct GenStats {
    pub prefill_secs: f64,
    pub decode_secs: Vec<f64>,
}

impl GenStats {
    /// Warm steady-state tok/s, estimated from the second half of decode
    /// steps (skips cold expert-compilation transients).
    pub fn warm_tok_s(&self) -> f64 {
        if self.decode_secs.len() < 2 {
            return 0.0;
        }
        let tail = &self.decode_secs[self.decode_secs.len() / 2..];
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        if mean > 0.0 {
            1.0 / mean
        } else {
            0.0
        }
    }
}

/// Single-stage OV-IR MoE runner for MiniMax-M2-style models.
pub struct OvMoeRunner {
    manifest: Manifest,
    model_dir: PathBuf,
    device: String,
    plugin: PluginConfig,

    /// Embedding (`layer0`) IR — present only on the first rank
    /// (`is_first`); `None` on downstream ranks, which receive the
    /// already-embedded hidden state over the wire.
    embed: Option<Runtime>,
    /// Final norm + lm_head IR — present only on the last rank
    /// (`is_last`); `None` on upstream ranks, which forward the hidden
    /// state downstream instead of producing logits.
    head: Option<Runtime>,
    /// Per-layer shells. Either the monolithic shell (CPU / no split) or the
    /// GPU-friendly `shell_core` (attention + KV, no router) when running on
    /// a non-CPU device with split graphs present.
    shells: Vec<Runtime>,
    /// Per-layer router graphs compiled on the CPU plugin, used only in the
    /// split path (the MoE router wedges the Intel GPU plugin). `None` in the
    /// monolithic path, where the router outputs come from `shells`.
    routers: Option<Vec<Runtime>>,
    core_ports: CorePorts,
    router_ports: RouterPorts,
    layer_ids: Vec<u32>,

    experts: ExpertCache,

    kv: Vec<LayerKv>,
    // pipeline-parallel role
    is_first: bool,
    is_last: bool,
    // cached dims
    hidden: usize,
    kv_heads: usize,
    head_dim: usize,
    top_k: usize,
    expert_intermediate: usize,
}

impl OvMoeRunner {
    /// Load the whole model on one node (single-stage). Equivalent to
    /// [`Self::load_staged`] with `rank=0, total=1`.
    pub fn load(
        model_dir: PathBuf,
        device: &str,
        plugin: PluginConfig,
        max_cached_experts: Option<NonZeroUsize>,
    ) -> Result<Self, OvMoeError> {
        Self::load_staged(
            model_dir,
            device,
            plugin,
            max_cached_experts,
            0,
            1,
            0,
            0,
            false,
        )
    }

    /// Load this rank's contiguous slice of the model for pipeline-parallel
    /// inference. Rank 0 owns the embedding (`layer0` IR); the last rank
    /// owns the head. The layer slice is `[layer_start, layer_end)` (global,
    /// half-open) when `layer_end > layer_start` — an explicit, possibly
    /// asymmetric split — otherwise an even split across ranks. `total == 1`
    /// loads the whole model (the path [`Self::load`] uses). The hidden
    /// state flows rank→rank as F32 over `cascadia-transport`; each rank
    /// keeps only its own layers' KV.
    #[allow(clippy::too_many_arguments)]
    pub fn load_staged(
        model_dir: PathBuf,
        device: &str,
        plugin: PluginConfig,
        max_cached_experts: Option<NonZeroUsize>,
        rank: u32,
        total: u32,
        layer_start: u32,
        layer_end: u32,
        force_split: bool,
    ) -> Result<Self, OvMoeError> {
        let manifest = Manifest::load(&model_dir)?;
        if !manifest.is_ov_shell() {
            return Err(OvMoeError::Internal(format!(
                "manifest shell_backend={:?} is not 'ov_ir'",
                manifest.shell_backend
            )));
        }
        // The shells run as OV graphs with a per-token-growing past_k/past_v
        // dimension, which trips the OV 2026.1 CPU snippets
        // shape-specialization bug — numerical errors that accumulate over
        // the sequence and silently corrupt output after ~10 tokens (the
        // K2.6 path dodges this by running shells in its Rust kernel). The
        // exporter/Python reference sets SNIPPETS_MODE=DISABLE; the engine
        // must too. This was THE cause of the int4/int8/NF4 "degrades after
        // the first sentence" — not expert quantization.
        //
        // SNIPPETS_MODE is a CPU-plugin option; the GPU plugin rejects it
        // ("Option not found: SNIPPETS_MODE") and the snippets bug is
        // CPU-specific anyway, so only set it when targeting CPU. This lets
        // a pipeline rank run on an Intel iGPU (e.g. a NUC stage) while the
        // CPU ranks keep the workaround.
        let plugin = if device.eq_ignore_ascii_case("CPU") {
            plugin.with("SNIPPETS_MODE", "DISABLE")
        } else {
            plugin
        };
        let utf8 = |p: &PathBuf| -> Result<String, OvMoeError> {
            p.to_str()
                .map(str::to_owned)
                .ok_or_else(|| OvMoeError::Internal(format!("non-UTF-8 path {}", p.display())))
        };
        let compile = |p: PathBuf| -> Result<Runtime, OvMoeError> {
            if !p.exists() {
                return Err(OvMoeError::MissingFile(p));
            }
            Ok(Runtime::compile(&utf8(&p)?, device, &plugin)?)
        };

        // Split the transformer shells into one contiguous slice per
        // rank. Rank 0 also carries the embedding; the last rank also
        // carries the head. The middle ranks are shells-only relays.
        let total = total.max(1);
        let rank = rank.min(total - 1);
        let is_first = rank == 0;
        let is_last = rank == total - 1;
        let all_ids = manifest.ov_layer_ids();
        let n = all_ids.len();
        // Layer slice. With an explicit `[layer_start, layer_end)` (global,
        // half-open; from the CLI --layer-start/--layer-end) we pin this
        // rank's layers — enabling an asymmetric ring (e.g. a high-RAM CPU
        // node holding most layers, small iGPU nodes a few each). The ranges
        // must partition `0..num_layers` contiguously in rank order (the
        // driver streams hidden states down the chain). Otherwise (both 0)
        // an even split by rank/total.
        let layer_ids: Vec<u32> = if layer_end > layer_start {
            all_ids
                .iter()
                .copied()
                .filter(|&l| l >= layer_start && l < layer_end)
                .collect()
        } else {
            let per = n / total as usize;
            let rem = n % total as usize;
            let r = rank as usize;
            let start = r * per + r.min(rem);
            let cnt = per + if r < rem { 1 } else { 0 };
            all_ids[start..start + cnt].to_vec()
        };
        if layer_ids.is_empty() {
            return Err(OvMoeError::Internal(format!(
                "rank {rank}/{total} holds zero of {n} layers (check --layer-start/--layer-end / total)"
            )));
        }

        info!(
            arch = %manifest.arch,
            num_layers = manifest.num_layers,
            num_experts = manifest.num_experts,
            top_k = manifest.top_k,
            rank,
            total,
            layers = format!("{}..{}", layer_ids.first().copied().unwrap_or(0), layer_ids.last().copied().unwrap_or(0)),
            "loading OV-IR sparse-MoE model"
        );

        let embed = if is_first {
            Some(compile(manifest.layer0_xml(&model_dir))?)
        } else {
            None
        };
        let head = if is_last {
            Some(compile(manifest.head_xml(&model_dir))?)
        } else {
            None
        };

        // iGPU/NPU path: the MoE router subgraph wedges the Intel GPU plugin
        // (CL_OUT_OF_RESOURCES — it can even brick the OpenCL context), but
        // the attention/KV core runs there fine. When targeting a non-CPU
        // device and the split graphs exist (`tools/split_m2_shells.py`),
        // load the GPU-friendly `shell_core` on the device and the carved
        // `router` on the CPU plugin. On CPU the monolithic shell is used.
        // Split the shell into shell_core (device) + router (CPU) for non-CPU
        // execution. `force_split` also forces it on CPU — used by the
        // split-path test to validate the carve is numerically identical to
        // the monolithic shell without needing a GPU.
        let split = manifest.shell_core_xml(&model_dir, layer_ids[0]).exists()
            && (force_split || !device.eq_ignore_ascii_case("CPU"));

        let mut shells = Vec::with_capacity(layer_ids.len());
        let routers = if split {
            // The router is tiny + static (no growing past_k), so it needs no
            // SNIPPETS workaround on the CPU plugin.
            let cpu_plugin = PluginConfig::new();
            let mut routers = Vec::with_capacity(layer_ids.len());
            for &lid in &layer_ids {
                let core_xml = manifest.shell_core_xml(&model_dir, lid);
                let router_xml = manifest.router_xml(&model_dir, lid);
                if !router_xml.exists() {
                    return Err(OvMoeError::MissingFile(router_xml));
                }
                shells.push(compile(core_xml)?);
                routers.push(Runtime::compile(&utf8(&router_xml)?, "CPU", &cpu_plugin)?);
            }
            Some(routers)
        } else {
            for &lid in &layer_ids {
                shells.push(compile(manifest.shell_xml(&model_dir, lid))?);
            }
            None
        };
        let core_ports = CorePorts::resolve(&shells[0])?;
        let router_ports = match &routers {
            Some(rs) => RouterPorts::resolve(&rs[0])?,
            None => RouterPorts::resolve(&shells[0])?,
        };
        info!(
            "compiled {} shells on {} (router_split={}, embed={}, head={})",
            shells.len(),
            device,
            split,
            is_first,
            is_last
        );

        let expert_intermediate = manifest.expert_intermediate as usize;
        let experts = match manifest.experts_format.as_str() {
            "int4_bin" => {
                if expert_intermediate == 0 {
                    return Err(OvMoeError::Internal(
                        "manifest experts_format=int4_bin requires expert_intermediate > 0".into(),
                    ));
                }
                info!("expert backend: int4_bin (AVX-512 kernel, no OV per-call overhead)");
                ExpertCache::Int4Bin(match max_cached_experts {
                    Some(n) => LruCache::new(n),
                    None => LruCache::unbounded(),
                })
            }
            _ => {
                info!("expert backend: ov_ir (per-expert OV CPU plugin call)");
                ExpertCache::OvIr(match max_cached_experts {
                    Some(n) => LruCache::new(n),
                    None => LruCache::unbounded(),
                })
            }
        };
        let kv = (0..layer_ids.len()).map(|_| LayerKv::new()).collect();

        Ok(Self {
            hidden: manifest.hidden_size as usize,
            kv_heads: manifest.num_kv_heads as usize,
            head_dim: manifest.qk_head_dim as usize,
            top_k: manifest.top_k as usize,
            expert_intermediate,
            manifest,
            model_dir,
            device: device.to_string(),
            plugin,
            embed,
            head,
            shells,
            routers,
            core_ports,
            router_ports,
            layer_ids,
            experts,
            kv,
            is_first,
            is_last,
        })
    }

    /// True on the first rank (owns the embedding).
    pub fn is_first(&self) -> bool {
        self.is_first
    }

    /// True on the last rank (owns the head / produces logits).
    pub fn is_last(&self) -> bool {
        self.is_last
    }

    /// Hidden-state width (`hidden_size`) — the length of the F32 vector
    /// exchanged between pipeline ranks.
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn reset(&mut self) {
        for k in &mut self.kv {
            k.reset();
        }
    }

    /// [`ModelFingerprint`] for this rank — the KV-prefix cache key. Same
    /// rule as the K2.6 runner: any field that changes the KV bits (arch,
    /// dims, this rank's layer slice) is baked in, so a snapshot from a
    /// different model or rank can never be restored.
    pub fn fingerprint(&self) -> ModelFingerprint {
        let layer_start = self.layer_ids.first().copied().unwrap_or(0);
        let layer_end = self.layer_ids.last().map(|&l| l + 1).unwrap_or(0);
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
            layer_start,
            layer_end,
            is_first: self.is_first,
            is_last: self.is_last,
        }
    }

    /// Snapshot this rank's KV (only its owned layers) into an
    /// [`OvMoeKvSnapshot`]. The `f32` buffers are cloned verbatim — no
    /// precision loss — so a later [`Self::restore_kv`] is bit-identical to
    /// a cold prefill. Every owned layer must agree on `seq` (they advance
    /// in lockstep through `forward_layers`); a mismatch is an engine bug,
    /// not recoverable state.
    /// This rank's current KV depth, or 0 if it owns no layers. Unlike [`Self::snapshot_kv`] this
    /// never errors on a cross-layer disagreement — the plane's depth guard wants a cursor, not a
    /// consistency verdict, and a disagreeing rank is caught by the snapshot path anyway.
    pub fn kv_past_seq_len(&self) -> usize {
        self.kv.first().map(|l| l.seq).unwrap_or(0)
    }

    pub fn snapshot_kv(&self) -> Result<OvMoeKvSnapshot, OvMoeError> {
        let ps = match self.kv.first() {
            Some(l) => l.seq,
            None => {
                return Err(OvMoeError::Internal(
                    "snapshot_kv: rank holds no layers".into(),
                ))
            }
        };
        for (idx, l) in self.kv.iter().enumerate() {
            if l.seq != ps {
                return Err(OvMoeError::Internal(format!(
                    "snapshot_kv: layer {idx} seq {} != expected {ps}",
                    l.seq
                )));
            }
        }
        let layers = self
            .kv
            .iter()
            .map(|l| OvLayerKvSlice {
                k: l.k.clone(),
                v: l.v.clone(),
                seq: l.seq,
            })
            .collect();
        Ok(OvMoeKvSnapshot {
            past_seq_len: ps,
            layers,
        })
    }

    /// Restore a previously-captured snapshot into this rank's KV. The
    /// snapshot must have one slice per owned layer (fingerprint check
    /// upstream guarantees the same rank/model; this is defence in depth).
    /// After Ok, every layer's `seq` equals the snapshot's `past_seq_len`
    /// and the next `forward_layers` writes its slot at that offset —
    /// exactly as after a fresh prefill of the matched prefix.
    pub fn restore_kv(&mut self, snap: &OvMoeKvSnapshot) -> Result<(), OvMoeError> {
        if snap.layers.len() != self.kv.len() {
            return Err(OvMoeError::Internal(format!(
                "restore_kv: snapshot has {} layers, rank owns {}",
                snap.layers.len(),
                self.kv.len()
            )));
        }
        for (dst, src) in self.kv.iter_mut().zip(&snap.layers) {
            dst.k = src.k.clone();
            dst.v = src.v.clone();
            dst.seq = src.seq;
        }
        Ok(())
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.manifest.eos_token_ids
    }

    /// Run expert `(lid, eid)` on `apn` (the post-attn-norm hidden state),
    /// returning the FFN output. Dispatches to the OV-IR or int4_bin
    /// backend per the manifest.
    fn dispatch_expert(&mut self, lid: u32, eid: u32, apn: &[f32]) -> Result<Vec<f32>, OvMoeError> {
        let h = self.hidden;
        if matches!(self.experts, ExpertCache::Int4Bin(_)) {
            let inter = self.expert_intermediate;
            let bytes = self.int4_expert_bytes(lid, eid)?;
            Ok(int4_expert_forward(&bytes, apn, h, inter))
        } else {
            self.ovir_ensure(lid, eid)?;
            let ExpertCache::OvIr(cache) = &mut self.experts else {
                unreachable!()
            };
            let rt = cache.get_mut(&(lid, eid)).unwrap();
            rt.set_input("x", DType::F32, &[1, 1, h], f32_as_bytes(apn))?;
            rt.infer()?;
            Ok(bytes_to_f32(&rt.output(0)?.2))
        }
    }

    /// Ensure the OV-IR expert is compiled and cached.
    fn ovir_ensure(&mut self, lid: u32, eid: u32) -> Result<(), OvMoeError> {
        let key = (lid, eid);
        if let ExpertCache::OvIr(c) = &self.experts {
            if c.contains(&key) {
                return Ok(());
            }
        }
        let xml = self.manifest.expert_xml(&self.model_dir, lid, eid);
        if !xml.exists() {
            return Err(OvMoeError::MissingFile(xml));
        }
        let path = xml
            .to_str()
            .ok_or_else(|| OvMoeError::Internal("non-UTF-8 expert path".into()))?
            .to_owned();
        let rt = Runtime::compile(&path, &self.device, &self.plugin)?;
        if let ExpertCache::OvIr(c) = &mut self.experts {
            c.put(key, rt);
        }
        Ok(())
    }

    /// Return the cached raw bytes of int4_bin expert `(lid, eid)`, reading
    /// (and LRU-caching) the file on a miss.
    fn int4_expert_bytes(&mut self, lid: u32, eid: u32) -> Result<Arc<Vec<u8>>, OvMoeError> {
        let key = (lid, eid);
        if let ExpertCache::Int4Bin(c) = &mut self.experts {
            if let Some(b) = c.get(&key) {
                return Ok(b.clone());
            }
        }
        let path = self.manifest.expert_bin(&self.model_dir, lid, eid);
        if !path.exists() {
            return Err(OvMoeError::MissingFile(path));
        }
        let bytes = Arc::new(
            std::fs::read(&path)
                .map_err(|e| OvMoeError::Internal(format!("read expert bin: {e}")))?,
        );
        if let ExpertCache::Int4Bin(c) = &mut self.experts {
            c.put(key, bytes.clone());
        }
        Ok(bytes)
    }

    /// Embed one token into a hidden state `[H]` (rank-0 / `is_first`
    /// only). The downstream ranks receive this over the wire instead.
    pub fn embed_token(&mut self, token: u32) -> Result<Vec<f32>, OvMoeError> {
        let embed = self.embed.as_mut().ok_or_else(|| {
            OvMoeError::Internal("embed_token on a rank without the embedding (not rank 0)".into())
        })?;
        let ids = [token as i64];
        embed.set_input("input_ids", DType::I64, &[1, 1], i64_as_bytes(&ids))?;
        embed.infer()?;
        let (_, _, hb) = embed.output(0)?;
        Ok(bytes_to_f32(&hb)) // [1,1,H]
    }

    /// Run this rank's shell+expert layers on the incoming hidden state at
    /// absolute position `pos`, updating this rank's per-layer KV cache and
    /// returning the outgoing hidden state `[H]`. Architecture-agnostic:
    /// the same call drives single-stage (all layers) and each pipeline
    /// rank (its slice). `pos` is the wire `past_seq_len`; the local KV
    /// `seq` tracks it in lockstep across ranks.
    pub fn forward_layers(&mut self, hidden: Vec<f32>, pos: usize) -> Result<Vec<f32>, OvMoeError> {
        let h = self.hidden;
        let kv = self.kv_heads;
        let d = self.head_dim;
        let mut hidden = hidden;

        for (idx, &lid) in self.layer_ids.clone().iter().enumerate() {
            let p = self.kv[idx].seq;
            let pos_i = [pos as i64];
            {
                let sh = &mut self.shells[idx];
                sh.set_input("x", DType::F32, &[1, 1, h], f32_as_bytes(&hidden))?;
                sh.set_input(
                    "past_k",
                    DType::F32,
                    &[1, kv, p, d],
                    f32_as_bytes(&self.kv[idx].k),
                )?;
                sh.set_input(
                    "past_v",
                    DType::F32,
                    &[1, kv, p, d],
                    f32_as_bytes(&self.kv[idx].v),
                )?;
                sh.set_input("past_seq_len", DType::I64, &[], i64_as_bytes(&pos_i))?;
                sh.infer()?;
            }
            let cp = self.core_ports;
            let rp = self.router_ports;
            let (apn, residual, shared, present_k, present_v) = {
                let sh = &self.shells[idx];
                (
                    bytes_to_f32(&sh.output(cp.apn)?.2),
                    bytes_to_f32(&sh.output(cp.residual)?.2),
                    bytes_to_f32(&sh.output(cp.shared)?.2),
                    bytes_to_f32(&sh.output(cp.present_k)?.2),
                    bytes_to_f32(&sh.output(cp.present_v)?.2),
                )
            };
            self.kv[idx].append(&present_k, &present_v, kv, d);

            // Routing ids/weights: from the carved CPU router graph when the
            // shells are split for non-CPU execution, else from the shell.
            let (ids_out, weights) = if let Some(routers) = self.routers.as_mut() {
                let rt = &mut routers[idx];
                rt.set_input("apn_in", DType::F32, &[1, 1, h], f32_as_bytes(&apn))?;
                rt.infer()?;
                (
                    bytes_to_i64(&rt.output(rp.ids)?.2),
                    bytes_to_f32(&rt.output(rp.weights)?.2),
                )
            } else {
                let sh = &self.shells[idx];
                (
                    bytes_to_i64(&sh.output(rp.ids)?.2),
                    bytes_to_f32(&sh.output(rp.weights)?.2),
                )
            };

            let mut moe = vec![0.0f32; h];
            for k in 0..self.top_k.min(ids_out.len()) {
                let eid = ids_out[k] as u32;
                let w = weights[k];
                let y = self.dispatch_expert(lid, eid, &apn)?;
                for j in 0..h {
                    moe[j] += w * y[j];
                }
            }
            for j in 0..h {
                hidden[j] = residual[j] + shared[j] + moe[j];
            }
        }
        Ok(hidden)
    }

    /// Run the head (final norm + lm_head) on the hidden state, returning
    /// the raw logits (last rank / `is_last` only). The caller samples.
    pub fn head_logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, OvMoeError> {
        let h = self.hidden;
        let head = self.head.as_mut().ok_or_else(|| {
            OvMoeError::Internal(
                "head_logits on a rank without the head (not the last rank)".into(),
            )
        })?;
        head.set_input("x", DType::F32, &[1, 1, h], f32_as_bytes(hidden))?;
        head.infer()?;
        Ok(bytes_to_f32(&head.output(0)?.2))
    }

    /// One single-stage decode step: returns the raw logits for `token` at
    /// absolute position `pos`, updating every layer's KV cache. Composes
    /// embed → layers → head, so the pipeline path and the single-stage
    /// path run identical math. The caller picks the next token.
    fn step_logits(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, OvMoeError> {
        let hidden = self.embed_token(token)?;
        let hidden = self.forward_layers(hidden, pos)?;
        self.head_logits(&hidden)
    }

    /// Generate with a sampling config (repetition penalty / temperature /
    /// top-p). Returns the generated tokens (excluding the prompt). The
    /// repetition-penalty `history` is the running generated stream. A
    /// default [`SamplingConfig`] (temperature 0, penalty 1.0) reduces to
    /// greedy argmax — see [`Self::generate_argmax`].
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
    ) -> Result<Vec<u32>, OvMoeError> {
        Ok(self.generate_timed(prompt_ids, max_new, cfg)?.0)
    }

    /// Like [`Self::generate`] but also returns per-step timing so callers
    /// can separate cold (first-token, expert-compilation-heavy on the
    /// ov_ir backend) from warm steady-state throughput.
    pub fn generate_timed(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
    ) -> Result<(Vec<u32>, GenStats), OvMoeError> {
        if prompt_ids.is_empty() || max_new == 0 {
            return Ok((Vec::new(), GenStats::default()));
        }
        self.reset();
        let mut rng = init_rng(cfg.seed);
        let eos = self.manifest.eos_token_ids.clone();
        let mut pos = 0usize;
        let prefill_t = std::time::Instant::now();
        let mut logits = Vec::new();
        for &t in prompt_ids {
            logits = self.step_logits(t, pos)?;
            pos += 1;
        }
        let prefill_secs = prefill_t.elapsed().as_secs_f64();
        let mut decode_secs: Vec<f64> = Vec::with_capacity(max_new);
        let mut history: Vec<i64> = Vec::with_capacity(max_new);
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        loop {
            let next = sample(&logits, &history, cfg, &mut rng) as u32;
            out.push(next);
            history.push(next as i64);
            if out.len() >= max_new || eos.contains(&next) {
                break;
            }
            let step_t = std::time::Instant::now();
            logits = self.step_logits(next, pos)?;
            pos += 1;
            decode_secs.push(step_t.elapsed().as_secs_f64());
        }
        Ok((
            out,
            GenStats {
                prefill_secs,
                decode_secs,
            },
        ))
    }

    /// Greedy generation (argmax). Returns the generated tokens.
    pub fn generate_argmax(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>, OvMoeError> {
        self.generate(prompt_ids, max_new, &SamplingConfig::default())
    }

    /// Generate with an optional KV-prefix cache (Issue-34: local warm-resume + cross-chain source).
    /// On a cache HIT, restores the matched prefix's KV and prefills only the suffix; on a MISS,
    /// snapshots the FULL prompt's KV (depth == prompt.len, taken at the prefill boundary before
    /// decode — no off-by-one) and inserts it keyed by the prompt. Returns the generated tokens plus
    /// the snapshot inserted on a miss, so the engine can mirror it into its lock-free holder cache.
    /// `cache = None` is byte-identical to [`Self::generate`]. Prefill uses `step_logits` (no
    /// sampling), so the rep-penalty `history` covers generated tokens only — a warm run matches cold.
    pub fn generate_with_cache(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        cfg: &SamplingConfig,
        mut cache: Option<&mut OvMoeKvPrefixCache>,
    ) -> Result<(Vec<u32>, Option<OvMoeKvSnapshot>), OvMoeError> {
        if prompt_ids.is_empty() || max_new == 0 {
            return Ok((Vec::new(), None));
        }
        self.reset();
        let mut rng = init_rng(cfg.seed);
        let eos = self.manifest.eos_token_ids.clone();
        let prompt64: Vec<i64> = prompt_ids.iter().map(|&t| i64::from(t)).collect();

        // Cache lookup / restore. `fp` computed only when the cache is present + enabled, so the
        // `None` path stays allocation-identical to `generate`.
        let mut fp: Option<ModelFingerprint> = None;
        let mut cache_skip = 0usize;
        let mut cache_hit = false;
        if let Some(c) = cache.as_mut() {
            if c.enabled() {
                let fingerprint = fp.insert(self.fingerprint());
                if let Some((snap, _)) = c.lookup(&prompt64, fingerprint) {
                    cache_skip = snap.past_seq_len;
                    self.restore_kv(&snap)?;
                    cache_hit = true;
                }
            }
        }

        // Prefill the suffix (positions [cache_skip, prompt.len)); the restored prefix's KV is already
        // in place. `pos` advances for skipped tokens too so past_seq_len stays correct.
        let mut pos = 0usize;
        let mut logits: Vec<f32> = Vec::new();
        for (i, &t) in prompt_ids.iter().enumerate() {
            if i < cache_skip {
                pos += 1;
                continue;
            }
            logits = self.step_logits(t, pos)?;
            pos += 1;
        }

        // On a miss, snapshot the full prompt's KV and insert it (keyed by the prompt).
        let mut cached_snap: Option<OvMoeKvSnapshot> = None;
        if !cache_hit {
            if let Some(c) = cache.as_mut() {
                if c.enabled() {
                    let fingerprint = fp.get_or_insert_with(|| self.fingerprint());
                    if let Ok(snap) = self.snapshot_kv() {
                        c.insert(prompt64.clone(), fingerprint, snap.clone());
                        cached_snap = Some(snap);
                    }
                }
            }
        }

        // First generated token from the last prefill step's logits, then decode.
        let mut history: Vec<i64> = Vec::with_capacity(max_new);
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        if logits.is_empty() {
            // No suffix token (cache_skip == prompt.len) — the lookup contract forbids this, but
            // return safely rather than sample from stale logits.
            return Ok((out, cached_snap));
        }
        loop {
            let next = sample(&logits, &history, cfg, &mut rng) as u32;
            out.push(next);
            history.push(next as i64);
            if out.len() >= max_new || eos.contains(&next) {
                break;
            }
            logits = self.step_logits(next, pos)?;
            pos += 1;
        }
        Ok((out, cached_snap))
    }
}

fn i64_as_bytes(v: &[i64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Run one int4_bin expert (SwiGLU FFN) via the cascadia-int4-gemm AVX-512
/// kernel. `bytes` is the flat expert binary laid out as six contiguous
/// regions — gate/up/down each a packed-nibble matrix (`out*in/2` bytes,
/// low nibble = even col) followed by its bf16 per-32-col-group scales
/// (`out*(in/INT4_GROUP)*2` bytes). `apn` is the fp32 input `[hidden]`;
/// returns the fp32 output `[hidden]`. Dims are derived from `hidden` and
/// `inter` (gate/up are `[inter, hidden]`, down is `[hidden, inter]`).
fn int4_expert_forward(bytes: &[u8], apn: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
    let gate_packed_len = inter * (hidden / 2);
    let gate_scale_len = inter * (hidden / INT4_GROUP) * 2;
    let down_packed_len = hidden * (inter / 2);
    let down_scale_len = hidden * (inter / INT4_GROUP) * 2;
    let mut o = 0usize;
    let mut take = |n: usize| {
        let s = &bytes[o..o + n];
        o += n;
        s
    };
    let gate_packed = take(gate_packed_len);
    let gate_scale = take(gate_scale_len);
    let up_packed = take(gate_packed_len);
    let up_scale = take(gate_scale_len);
    let down_packed = take(down_packed_len);
    let down_scale = take(down_scale_len);

    let x_bf16: Vec<bf16> = apn.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut out_bf16 = vec![bf16::from_f32(0.0); hidden];
    expert_forward_sparse(
        &x_bf16,
        gate_packed,
        gate_scale,
        up_packed,
        up_scale,
        down_packed,
        down_scale,
        &mut out_bf16,
        INT4_KEEP_ALL_TAU,
    );
    out_bf16.iter().map(|b| b.to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: usize = INT4_GROUP;

    /// Pack a `[out, in]` fp32 matrix into the int4_bin layout the kernel
    /// reads: per-32-col-group symmetric int4 (scale = max_abs/7), nibbles
    /// packed two-per-byte (low = even col), bf16 LE scales. Mirrors the
    /// Python exporter so the Rust unit test and the kernel agree on the
    /// convention before any full-model export.
    fn pack(w: &[f32], out: usize, inn: usize) -> (Vec<u8>, Vec<u8>) {
        let ng = inn / G;
        let mut packed = vec![0u8; out * inn / 2];
        let mut scale = Vec::with_capacity(out * ng * 2);
        for r in 0..out {
            for g in 0..ng {
                let base = r * inn + g * G;
                let max_abs = (0..G).map(|i| w[base + i].abs()).fold(0.0f32, f32::max);
                let s = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
                // bf16 round-to-nearest-even of the scale.
                let bits = s.to_bits();
                let bf = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
                scale.extend_from_slice(&bf.to_le_bytes());
                for i in 0..G {
                    let q = (w[base + i] / s).round().clamp(-8.0, 7.0) as i32;
                    let nib = (q + 8) as u8;
                    let col = g * G + i;
                    let byte = r * (inn / 2) + col / 2;
                    if col % 2 == 0 {
                        packed[byte] |= nib & 0x0F;
                    } else {
                        packed[byte] |= (nib & 0x0F) << 4;
                    }
                }
            }
        }
        (packed, scale)
    }

    fn fp32_swiglu(
        x: &[f32],
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        h: usize,
        m: usize,
    ) -> Vec<f32> {
        let lin = |w: &[f32], x: &[f32], o: usize, i: usize| -> Vec<f32> {
            (0..o)
                .map(|r| (0..i).map(|c| w[r * i + c] * x[c]).sum::<f32>())
                .collect()
        };
        let g = lin(gate, x, m, h);
        let u = lin(up, x, m, h);
        let inter: Vec<f32> = (0..m)
            .map(|i| (g[i] / (1.0 + (-g[i]).exp())) * u[i])
            .collect();
        lin(down, &inter, h, m)
    }

    #[test]
    fn int4_bin_expert_matches_fp32_within_tolerance() {
        // Small but group-aligned dims.
        let (h, m) = (64usize, 32usize);
        let mk = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32;
                    (x / 1000.0 - 0.5) * 0.2
                })
                .collect()
        };
        let gate = mk(m * h, 1);
        let up = mk(m * h, 2);
        let down = mk(h * m, 3);
        let x = mk(h, 4);

        let (gp, gs) = pack(&gate, m, h);
        let (upk, us) = pack(&up, m, h);
        let (dp, ds) = pack(&down, h, m);
        let mut bytes = Vec::new();
        for s in [&gp, &gs, &upk, &us, &dp, &ds] {
            bytes.extend_from_slice(s);
        }

        let got = int4_expert_forward(&bytes, &x, h, m);
        let want = fp32_swiglu(&x, &gate, &up, &down, h, m);

        let num: f32 = got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum();
        let den: f32 = want.iter().map(|b| b * b).sum::<f32>().max(1e-12);
        let rel = (num / den).sqrt();
        eprintln!("int4_bin vs fp32 relative L2 error = {rel:.4}");
        assert!(rel < 0.1, "int4_bin expert error too high: {rel}");
    }
}
