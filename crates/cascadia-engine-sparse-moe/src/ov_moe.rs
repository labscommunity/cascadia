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
//!   4. runs the head IR (final norm + lm_head) and argmaxes.
//!
//! Because the graphs carry their own shapes, this path is fully
//! dimension-agnostic — the same code runs the tiny synthetic M2 used by
//! the correctness test and the full 230B model. Single-stage only.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use lru::LruCache;
use tracing::info;

use cascadia_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};

use crate::manifest::{Manifest, ManifestError};

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

    experts: LruCache<(u32, u32), Runtime>,

    kv: Vec<LayerKv>,
    // cached dims
    hidden: usize,
    kv_heads: usize,
    head_dim: usize,
    top_k: usize,
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

        let experts = match max_cached_experts {
            Some(n) => LruCache::new(n),
            None => LruCache::unbounded(),
        };
        let kv = (0..layer_ids.len()).map(|_| LayerKv::new()).collect();

        Ok(Self {
            hidden: manifest.hidden_size as usize,
            kv_heads: manifest.num_kv_heads as usize,
            head_dim: manifest.qk_head_dim as usize,
            top_k: manifest.top_k as usize,
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

    fn expert(&mut self, lid: u32, eid: u32) -> Result<&mut Runtime, OvMoeError> {
        let key = (lid, eid);
        if !self.experts.contains(&key) {
            let xml = self.manifest.expert_xml(&self.model_dir, lid, eid);
            if !xml.exists() {
                return Err(OvMoeError::MissingFile(xml));
            }
            let path = xml
                .to_str()
                .ok_or_else(|| OvMoeError::Internal("non-UTF-8 expert path".into()))?
                .to_owned();
            let rt = Runtime::compile(&path, &self.device, &self.plugin)?;
            self.experts.put(key, rt);
        }
        Ok(self.experts.get_mut(&key).unwrap())
    }

    /// One decode step: returns the argmax next-token id for `token` at
    /// absolute position `pos`, updating every layer's KV cache.
    fn step(&mut self, token: u32, pos: usize) -> Result<u32, OvMoeError> {
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
                let ex = self.expert(lid, eid)?;
                ex.set_input("x", DType::F32, &[1, 1, h], f32_as_bytes(&apn))?;
                ex.infer()?;
                let y = bytes_to_f32(&ex.output(0)?.2);
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
        let logits = bytes_to_f32(&self.head.output(0)?.2);
        Ok(argmax(&logits) as u32)
    }

    /// Greedy generation. Returns the generated tokens (excluding prompt).
    pub fn generate_argmax(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>, OvMoeError> {
        if prompt_ids.is_empty() || max_new == 0 {
            return Ok(Vec::new());
        }
        self.reset();
        let mut pos = 0usize;
        let mut first = 0u32;
        for &t in prompt_ids {
            first = self.step(t, pos)?;
            pos += 1;
        }
        let mut out = vec![first];
        let eos = self.manifest.eos_token_ids.clone();
        let mut cur = first;
        while out.len() < max_new {
            if eos.contains(&cur) {
                break;
            }
            cur = self.step(cur, pos)?;
            pos += 1;
            out.push(cur);
        }
        Ok(out)
    }
}

fn i64_as_bytes(v: &[i64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}
