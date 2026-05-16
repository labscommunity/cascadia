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
use std::time::Instant;

use half::bf16;
use tahoma_int4_gemm::{
    expert_forward as int4_expert_forward, ExpertWeights, SafetensorsExpert,
    SafetensorsExpertSource,
};
use tahoma_ov_genai_shim::{DType, Error as OvError, PluginConfig, Runtime};
use thiserror::Error;
use tracing::{debug, info};

use crate::manifest::{Manifest, ManifestError};
use crate::tensors::{
    bf16_bytes_to_f32, bytes_to_i64, causal_mask_f32, concat_along_axis, f16_bytes_to_f32,
    f32_to_bf16_bytes, f32_to_bytes, i64_to_bytes,
};

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

/// Per-MoE-layer state held across generation steps.
struct LayerState {
    lid: u32,
    shell: Runtime,
    /// Last shell output names so we can look them up by index. OV inputs
    /// can have ports addressed by 0-based index OR by name string;
    /// `Runtime::output` only takes index, so we cache the index for each
    /// canonical name once.
    output_idx: HashMap<&'static str, usize>,
    /// past_k bytes + shape `[1, num_kv_heads, past_seq, qk_head_dim]` f32
    past_k: Vec<u8>,
    past_k_shape: Vec<usize>,
    past_v: Vec<u8>,
    past_v_shape: Vec<usize>,
}

impl LayerState {
    fn new(
        lid: u32,
        shell: Runtime,
        num_kv_heads: u32,
        qk_head_dim: u32,
        v_head_dim: u32,
    ) -> Result<Self, RunnerError> {
        let mut output_idx = HashMap::new();
        let names = shell.output_names()?;
        let aliases: Vec<Vec<String>> = (0..names.len())
            .map(|i| shell.output_aliases(i).unwrap_or_default())
            .collect();
        for canonical in [
            "attn_out_post_norm",
            "attn_residual",
            "shared_expert_out",
            "routing_ids",
            "routing_weights",
            "present_k",
            "present_v",
        ] {
            let idx = aliases
                .iter()
                .position(|als| als.iter().any(|a| a == canonical))
                .or_else(|| names.iter().position(|n| n == canonical))
                .ok_or_else(|| {
                    RunnerError::Internal(format!(
                        "shell output `{}` not found in IR (got: {:?})",
                        canonical, names
                    ))
                })?;
            output_idx.insert(canonical, idx);
        }
        Ok(Self {
            lid,
            shell,
            output_idx,
            past_k: Vec::new(),
            past_k_shape: vec![1, num_kv_heads as usize, 0, qk_head_dim as usize],
            past_v: Vec::new(),
            past_v_shape: vec![1, num_kv_heads as usize, 0, v_head_dim as usize],
        })
    }

    fn out_idx(&self, name: &str) -> usize {
        self.output_idx[name]
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
    source: SafetensorsExpertSource,
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

/// Main inference runner.
pub struct Runner {
    pub manifest: Manifest,
    _model_dir: PathBuf,
    _device: String,
    _plugin: PluginConfig,
    layer0: Runtime,
    head: Runtime,
    layers: Vec<LayerState>,
    experts: ExpertCache,
}

impl Runner {
    /// Compile all shells + layer0 + head. Experts are loaded lazily.
    pub fn load(
        model_dir: PathBuf,
        device: &str,
        plugin: PluginConfig,
    ) -> Result<Self, RunnerError> {
        let manifest = Manifest::load(&model_dir)?;
        info!(
            arch = %manifest.arch,
            num_layers = manifest.num_layers,
            num_experts = manifest.num_experts,
            top_k = manifest.top_k,
            "loading sparse-MoE model"
        );

        let utf8 = |p: &PathBuf| -> Result<String, RunnerError> {
            p.to_str()
                .map(str::to_owned)
                .ok_or_else(|| RunnerError::Internal(format!("non-UTF-8 path: {}", p.display())))
        };

        let layer0_xml = manifest.layer0_xml(&model_dir);
        if !layer0_xml.exists() {
            return Err(RunnerError::MissingFile(layer0_xml));
        }
        let layer0 = Runtime::compile(&utf8(&layer0_xml)?, device, &plugin)?;
        info!("compiled layer 0 (stateless embed+dense)");

        let head_xml = manifest.head_xml(&model_dir);
        if !head_xml.exists() {
            return Err(RunnerError::MissingFile(head_xml));
        }
        let head = Runtime::compile(&utf8(&head_xml)?, device, &plugin)?;
        info!("compiled head (RMSNorm + lm_head)");

        let moe_layer_ids = manifest.moe_layer_ids();
        let mut layers = Vec::with_capacity(moe_layer_ids.len());
        for (i, &lid) in moe_layer_ids.iter().enumerate() {
            let xml = manifest.shell_xml(&model_dir, lid);
            if !xml.exists() {
                return Err(RunnerError::MissingFile(xml));
            }
            let rt = Runtime::compile(&utf8(&xml)?, device, &plugin)?;
            layers.push(LayerState::new(
                lid,
                rt,
                manifest.num_kv_heads,
                manifest.qk_head_dim,
                manifest.v_head_dim,
            )?);
            if (i + 1) % 10 == 0 || i + 1 == moe_layer_ids.len() {
                info!("compiled shells {}/{}", i + 1, moe_layer_ids.len());
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
                info!("expert backend: safetensors_bin (direct safetensors mmap + AVX-512)");
                let st_dir = model_dir.join("safetensors");
                let st_dir = if st_dir.exists() {
                    st_dir
                } else {
                    model_dir.clone()
                };
                let source = SafetensorsExpertSource::open(st_dir)
                    .map_err(|e| RunnerError::Internal(format!("safetensors open: {e}")))?;
                ExpertCache::SafetensorsBin(SafetensorsExpertCache {
                    source,
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
            layer0,
            head,
            layers,
            experts,
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
            l.past_k.clear();
            l.past_k_shape[2] = 0;
            l.past_v.clear();
            l.past_v_shape[2] = 0;
        }
    }

    /// Run one forward pass:
    /// - `full_ids` is the FULL prefix-so-far (1D i64), used by stateless
    ///   layer 0.
    /// - The shells consume just the trailing `tail_len` tokens, with the
    ///   per-layer KV state representing the prior `past_seq_len = full_ids.len - tail_len` tokens.
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
        let top_k = self.manifest.top_k as usize;

        // 1) Layer 0 (stateless): hidden out for the full prefix.
        let ids_bytes = i64_to_bytes(full_ids);
        let ids_shape = [1usize, full_ids.len()];
        let input_name = self.layer0.input_name(0)?;
        self.layer0
            .set_input(&input_name, DType::I64, &ids_shape, &ids_bytes)?;
        self.layer0.infer()?;
        let (l0_dtype, l0_shape, l0_bytes) = self.layer0.output(0)?;
        // l0_shape: [1, full_ids.len(), 7168]
        if l0_shape.len() != 3
            || l0_shape[0] != 1
            || l0_shape[1] != full_ids.len()
            || l0_shape[2] != hidden
        {
            return Err(RunnerError::Internal(format!(
                "layer0 unexpected output shape {:?}",
                l0_shape
            )));
        }
        let l0_f32: Vec<f32> = match l0_dtype {
            DType::F32 => read_f32(&l0_bytes),
            DType::Bf16 => bf16_bytes_to_f32(&l0_bytes),
            DType::F16 => f16_bytes_to_f32(&l0_bytes),
            _ => {
                return Err(RunnerError::Internal(format!(
                    "layer0 dtype {:?}",
                    l0_dtype
                )))
            }
        };
        // Slice the last `tail_len` positions.
        let row_off = past_seq_len * hidden;
        let mut h_f32 = l0_f32[row_off..].to_vec();
        let mut h_shape = vec![1usize, tail_len, hidden];

        // 2) Each MoE shell + experts.
        let (mask_f32, mask_shape) = causal_mask_f32(tail_len, past_seq_len);
        let mask_bytes = f32_to_bytes(&mask_f32);
        let past_len_bytes = (past_seq_len as i64).to_le_bytes().to_vec();

        // We need to iterate layers in order; for each, compile or
        // reuse expert IRs. The `experts` cache borrow-conflicts with
        // the &mut self.layers borrow, so do them by index.
        let n_layers = self.layers.len();
        for i in 0..n_layers {
            let lid = self.layers[i].lid;

            // 2a) Set shell inputs.
            let h_bf16 = f32_to_bf16_bytes(&h_f32);
            let h_shape_now = h_shape.clone();
            let past_k_bytes = std::mem::take(&mut self.layers[i].past_k);
            let past_k_shape = self.layers[i].past_k_shape.clone();
            let past_v_bytes = std::mem::take(&mut self.layers[i].past_v);
            let past_v_shape = self.layers[i].past_v_shape.clone();

            self.layers[i]
                .shell
                .set_input("x.1", DType::Bf16, &h_shape_now, &h_bf16)?;
            self.layers[i]
                .shell
                .set_input("past_k", DType::F32, &past_k_shape, &past_k_bytes)?;
            self.layers[i]
                .shell
                .set_input("past_v", DType::F32, &past_v_shape, &past_v_bytes)?;
            self.layers[i].shell.set_input(
                "attn_mask_ext",
                DType::F32,
                &mask_shape,
                &mask_bytes,
            )?;
            self.layers[i]
                .shell
                .set_input("past_seq_len", DType::I64, &[], &past_len_bytes)?;
            self.layers[i].shell.infer()?;

            // 2b) Read outputs.
            let read_f32_out =
                |layer: &Runtime, idx: usize| -> Result<(Vec<usize>, Vec<f32>), RunnerError> {
                    let (dt, shape, bytes) = layer.output(idx)?;
                    let v = match dt {
                        DType::F32 => read_f32(&bytes),
                        DType::Bf16 => bf16_bytes_to_f32(&bytes),
                        DType::F16 => f16_bytes_to_f32(&bytes),
                        _ => {
                            return Err(RunnerError::Internal(format!(
                                "shell L{} output dtype {:?} not f32-castable",
                                idx, dt
                            )));
                        }
                    };
                    Ok((shape, v))
                };
            let attn_out_idx = self.layers[i].out_idx("attn_out_post_norm");
            let residual_idx = self.layers[i].out_idx("attn_residual");
            let shared_idx = self.layers[i].out_idx("shared_expert_out");
            let routing_ids_idx = self.layers[i].out_idx("routing_ids");
            let routing_weights_idx = self.layers[i].out_idx("routing_weights");
            let present_k_idx = self.layers[i].out_idx("present_k");
            let present_v_idx = self.layers[i].out_idx("present_v");

            let (_, attn_out_f32) = read_f32_out(&self.layers[i].shell, attn_out_idx)?;
            let (_, residual_f32) = read_f32_out(&self.layers[i].shell, residual_idx)?;
            let (_, shared_f32) = read_f32_out(&self.layers[i].shell, shared_idx)?;
            let (routing_ids_shape, routing_ids_bytes) = self.layers[i]
                .shell
                .output(routing_ids_idx)
                .map(|(_, s, b)| (s, b))?;
            let routing_ids = bytes_to_i64(&routing_ids_bytes);
            let (_, routing_weights_f32) =
                read_f32_out(&self.layers[i].shell, routing_weights_idx)?;
            let (pk_dt, pk_shape, pk_bytes) = self.layers[i].shell.output(present_k_idx)?;
            let (pv_dt, pv_shape, pv_bytes) = self.layers[i].shell.output(present_v_idx)?;
            if pk_dt != DType::F32 || pv_dt != DType::F32 {
                return Err(RunnerError::Internal(format!(
                    "present_k/v dtype not f32 ({:?}, {:?})",
                    pk_dt, pv_dt
                )));
            }

            // 2c) Append the freshly-computed present_k/v onto the running cache.
            let (new_past_k, new_past_k_shape) =
                concat_along_axis(&past_k_bytes, &past_k_shape, &pk_bytes, &pk_shape, 2, 4);
            let (new_past_v, new_past_v_shape) =
                concat_along_axis(&past_v_bytes, &past_v_shape, &pv_bytes, &pv_shape, 2, 4);
            self.layers[i].past_k = new_past_k;
            self.layers[i].past_k_shape = new_past_k_shape;
            self.layers[i].past_v = new_past_v;
            self.layers[i].past_v_shape = new_past_v_shape;

            // 2d) Expert dispatch. attn_out_f32 has shape [seq, 7168].
            // Each expert call takes [1, seq_token, 7168] bf16 in.
            // routing_ids has shape [seq, top_k].
            if routing_ids_shape.len() != 2 || routing_ids_shape[1] != top_k {
                return Err(RunnerError::Internal(format!(
                    "routing_ids shape unexpected {:?}",
                    routing_ids_shape
                )));
            }
            // moe accumulator: same shape as residual = [1, seq, 7168] f32, zero.
            let mut moe = vec![0.0f32; residual_f32.len()];
            for tok_idx in 0..tail_len {
                let attn_row = &attn_out_f32[tok_idx * hidden..(tok_idx + 1) * hidden];
                for k in 0..top_k {
                    let eid = routing_ids[tok_idx * top_k + k] as u32;
                    let w = routing_weights_f32[tok_idx * top_k + k];
                    let y_f32 = self.dispatch_expert(lid, eid, attn_row)?;
                    let dst_off = tok_idx * hidden;
                    for j in 0..hidden {
                        moe[dst_off + j] += w * y_f32[j];
                    }
                }
            }

            // 2e) Combine: h_next = residual + shared + moe
            let mut h_next = vec![0.0f32; residual_f32.len()];
            for j in 0..residual_f32.len() {
                h_next[j] = residual_f32[j] + shared_f32[j] + moe[j];
            }
            h_f32 = h_next;
            h_shape = vec![1, tail_len, hidden];
        }

        // 3) Head on the LAST token.
        let last_off = (tail_len - 1) * hidden;
        let last_h_bf16 = f32_to_bf16_bytes(&h_f32[last_off..last_off + hidden]);
        let head_in = self.head.input_name(0)?;
        self.head
            .set_input(&head_in, DType::Bf16, &[1, 1, hidden], &last_h_bf16)?;
        self.head.infer()?;
        let (head_dt, head_shape, head_bytes) = self.head.output(0)?;
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

fn argmax(xs: &[f32]) -> i64 {
    let mut best = 0i64;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as i64;
        }
    }
    best
}
