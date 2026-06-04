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
//! Because the graphs carry their own shapes, this path is fully
//! dimension-agnostic — the same code runs the tiny synthetic M2 used by
//! the correctness test and the full 230B model. Single-stage only.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use cascadia_int4_gemm::expert_forward_sparse;
use half::bf16;
use lru::LruCache;
use tracing::info;

use cascadia_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};

use crate::manifest::{Manifest, ManifestError};
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

/// Resolved output-port indices for a shell graph (same layout every layer).
#[derive(Clone, Copy)]
struct ShellPorts {
    apn: usize,
    residual: usize,
    shared: usize,
    ids: usize,
    weights: usize,
    present_k: usize,
    present_v: usize,
}

impl ShellPorts {
    fn resolve(rt: &Runtime) -> Result<Self, OvMoeError> {
        Ok(Self {
            apn: out_idx(rt, "attn_out_post_norm")?,
            residual: out_idx(rt, "attn_residual")?,
            shared: out_idx(rt, "shared_expert_out")?,
            ids: out_idx(rt, "routing_ids")?,
            weights: out_idx(rt, "routing_weights")?,
            present_k: out_idx(rt, "present_k")?,
            present_v: out_idx(rt, "present_v")?,
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

    embed: Runtime,
    head: Runtime,
    shells: Vec<Runtime>,
    shell_ports: ShellPorts,
    layer_ids: Vec<u32>,

    experts: ExpertCache,

    kv: Vec<LayerKv>,
    // cached dims
    hidden: usize,
    kv_heads: usize,
    head_dim: usize,
    top_k: usize,
    expert_intermediate: usize,
}

impl OvMoeRunner {
    pub fn load(
        model_dir: PathBuf,
        device: &str,
        plugin: PluginConfig,
        max_cached_experts: Option<NonZeroUsize>,
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
        let plugin = plugin.with("SNIPPETS_MODE", "DISABLE");
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

        info!(
            arch = %manifest.arch,
            num_layers = manifest.num_layers,
            num_experts = manifest.num_experts,
            top_k = manifest.top_k,
            "loading OV-IR sparse-MoE model (single-stage)"
        );

        let embed = compile(manifest.layer0_xml(&model_dir))?;
        let head = compile(manifest.head_xml(&model_dir))?;

        let layer_ids = manifest.ov_layer_ids();
        let mut shells = Vec::with_capacity(layer_ids.len());
        for &lid in &layer_ids {
            shells.push(compile(manifest.shell_xml(&model_dir, lid))?);
        }
        let shell_ports = ShellPorts::resolve(&shells[0])?;
        info!("compiled {} shells + embed + head", shells.len());

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
            shell_ports,
            layer_ids,
            experts,
            kv,
        })
    }

    pub fn reset(&mut self) {
        for k in &mut self.kv {
            k.reset();
        }
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

    /// One decode step: returns the raw logits for `token` at absolute
    /// position `pos`, updating every layer's KV cache. The caller picks
    /// the next token (argmax / sampling).
    fn step_logits(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, OvMoeError> {
        let h = self.hidden;
        let kv = self.kv_heads;
        let d = self.head_dim;

        // embed
        let ids = [token as i64];
        self.embed
            .set_input("input_ids", DType::I64, &[1, 1], i64_as_bytes(&ids))?;
        self.embed.infer()?;
        let (_, _, hb) = self.embed.output(0)?;
        let mut hidden = bytes_to_f32(&hb); // [1,1,H]

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
            let ports = self.shell_ports;
            let sh = &self.shells[idx];
            let apn = bytes_to_f32(&sh.output(ports.apn)?.2);
            let residual = bytes_to_f32(&sh.output(ports.residual)?.2);
            let shared = bytes_to_f32(&sh.output(ports.shared)?.2);
            let ids_out = bytes_to_i64(&sh.output(ports.ids)?.2);
            let weights = bytes_to_f32(&sh.output(ports.weights)?.2);
            let present_k = bytes_to_f32(&sh.output(ports.present_k)?.2);
            let present_v = bytes_to_f32(&sh.output(ports.present_v)?.2);
            self.kv[idx].append(&present_k, &present_v, kv, d);

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

        // head
        self.head
            .set_input("x", DType::F32, &[1, 1, h], f32_as_bytes(&hidden))?;
        self.head.infer()?;
        Ok(bytes_to_f32(&self.head.output(0)?.2))
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
