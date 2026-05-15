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

/// Lazy-load cache: compiled per-(layer, expert) IR.
struct ExpertCache {
    model_dir: PathBuf,
    manifest_layer_xml: Box<dyn Fn(&PathBuf, u32, u32) -> PathBuf + Send>,
    device: String,
    plugin: PluginConfig,
    map: HashMap<(u32, u32), Runtime>,
    compile_count: u64,
    compile_secs: f64,
}

impl ExpertCache {
    fn new(
        model_dir: PathBuf,
        manifest_layer_xml: impl Fn(&PathBuf, u32, u32) -> PathBuf + Send + 'static,
        device: String,
        plugin: PluginConfig,
    ) -> Self {
        Self {
            model_dir,
            manifest_layer_xml: Box::new(manifest_layer_xml),
            device,
            plugin,
            map: HashMap::new(),
            compile_count: 0,
            compile_secs: 0.0,
        }
    }

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

        let layer0_xml = manifest.layer0_xml(&model_dir);
        if !layer0_xml.exists() {
            return Err(RunnerError::MissingFile(layer0_xml));
        }
        let layer0 = Runtime::compile(
            layer0_xml.to_str().unwrap(),
            device,
            &plugin,
        )?;
        info!("compiled layer 0 (stateless embed+dense)");

        let head_xml = manifest.head_xml(&model_dir);
        if !head_xml.exists() {
            return Err(RunnerError::MissingFile(head_xml));
        }
        let head = Runtime::compile(head_xml.to_str().unwrap(), device, &plugin)?;
        info!("compiled head (RMSNorm + lm_head)");

        let moe_layer_ids = manifest.moe_layer_ids();
        let mut layers = Vec::with_capacity(moe_layer_ids.len());
        for (i, &lid) in moe_layer_ids.iter().enumerate() {
            let xml = manifest.shell_xml(&model_dir, lid);
            if !xml.exists() {
                return Err(RunnerError::MissingFile(xml));
            }
            let rt = Runtime::compile(xml.to_str().unwrap(), device, &plugin)?;
            layers.push(LayerState::new(
                lid,
                rt,
                manifest.num_kv_heads,
                manifest.qk_head_dim,
                manifest.v_head_dim,
            )?);
            if (i + 1) % 10 == 0 || i + 1 == moe_layer_ids.len() {
                info!(
                    "compiled shells {}/{}",
                    i + 1,
                    moe_layer_ids.len()
                );
            }
        }

        let manifest_clone = manifest.clone();
        let experts = ExpertCache::new(
            model_dir.clone(),
            move |md, lid, eid| manifest_clone.expert_xml(md, lid, eid),
            device.to_string(),
            plugin.clone(),
        );

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
    fn step(
        &mut self,
        full_ids: &[i64],
        tail_len: usize,
    ) -> Result<Vec<f32>, RunnerError> {
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
        self.layer0.set_input(&input_name, DType::I64, &ids_shape, &ids_bytes)?;
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
            _ => return Err(RunnerError::Internal(format!("layer0 dtype {:?}", l0_dtype))),
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

            self.layers[i].shell.set_input(
                "x.1",
                DType::Bf16,
                &h_shape_now,
                &h_bf16,
            )?;
            self.layers[i].shell.set_input(
                "past_k",
                DType::F32,
                &past_k_shape,
                &past_k_bytes,
            )?;
            self.layers[i].shell.set_input(
                "past_v",
                DType::F32,
                &past_v_shape,
                &past_v_bytes,
            )?;
            self.layers[i].shell.set_input(
                "attn_mask_ext",
                DType::F32,
                &mask_shape,
                &mask_bytes,
            )?;
            self.layers[i].shell.set_input(
                "past_seq_len",
                DType::I64,
                &[],
                &past_len_bytes,
            )?;
            self.layers[i].shell.infer()?;

            // 2b) Read outputs.
            let read_f32_out = |layer: &Runtime, idx: usize| -> Result<(Vec<usize>, Vec<f32>), RunnerError> {
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

            let (_, attn_out_f32) =
                read_f32_out(&self.layers[i].shell, attn_out_idx)?;
            let (_, residual_f32) =
                read_f32_out(&self.layers[i].shell, residual_idx)?;
            let (_, shared_f32) = read_f32_out(&self.layers[i].shell, shared_idx)?;
            let (routing_ids_shape, routing_ids_bytes) =
                self.layers[i].shell.output(routing_ids_idx).map(|(_, s, b)| (s, b))?;
            let routing_ids = bytes_to_i64(&routing_ids_bytes);
            let (_, routing_weights_f32) =
                read_f32_out(&self.layers[i].shell, routing_weights_idx)?;
            let (pk_dt, pk_shape, pk_bytes) =
                self.layers[i].shell.output(present_k_idx)?;
            let (pv_dt, pv_shape, pv_bytes) =
                self.layers[i].shell.output(present_v_idx)?;
            if pk_dt != DType::F32 || pv_dt != DType::F32 {
                return Err(RunnerError::Internal(format!(
                    "present_k/v dtype not f32 ({:?}, {:?})",
                    pk_dt, pv_dt
                )));
            }

            // 2c) Append the freshly-computed present_k/v onto the running cache.
            let (new_past_k, new_past_k_shape) = concat_along_axis(
                &past_k_bytes,
                &past_k_shape,
                &pk_bytes,
                &pk_shape,
                2,
                4,
            );
            let (new_past_v, new_past_v_shape) = concat_along_axis(
                &past_v_bytes,
                &past_v_shape,
                &pv_bytes,
                &pv_shape,
                2,
                4,
            );
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
                let attn_bf16 = f32_to_bf16_bytes(attn_row);
                for k in 0..top_k {
                    let eid = routing_ids[tok_idx * top_k + k] as u32;
                    let w = routing_weights_f32[tok_idx * top_k + k];
                    let rt = self.experts.get(lid, eid)?;
                    rt.set_input("x", DType::Bf16, &[1, 1, hidden], &attn_bf16)?;
                    rt.infer()?;
                    let (e_dt, _e_shape, e_bytes) = rt.output(0)?;
                    let y_f32 = match e_dt {
                        DType::F32 => read_f32(&e_bytes),
                        DType::Bf16 => bf16_bytes_to_f32(&e_bytes),
                        DType::F16 => f16_bytes_to_f32(&e_bytes),
                        _ => {
                            return Err(RunnerError::Internal(format!(
                                "expert output dtype {:?} not f32-castable",
                                e_dt
                            )));
                        }
                    };
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
        self.head.set_input(&head_in, DType::Bf16, &[1, 1, hidden], &last_h_bf16)?;
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

    /// Greedy argmax generation. Returns the vector of generated token IDs
    /// (excluding the prompt). Stops on EOS or after `max_tokens`.
    pub fn generate_argmax(
        &mut self,
        prompt_ids: &[i64],
        max_tokens: usize,
    ) -> Result<Vec<i64>, RunnerError> {
        self.reset_kv();
        let eos: Vec<i64> = self.manifest.eos_token_ids.iter().map(|&x| x as i64).collect();
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
            let next = argmax(&l);
            history.push(next);
            generated.push(next);
        }

        // Decode.
        for step_i in 1..max_tokens {
            if !generated.is_empty() && eos.contains(generated.last().unwrap()) {
                break;
            }
            let t_step = Instant::now();
            let logits = self.step(&history, 1)?;
            let next = argmax(&logits);
            history.push(next);
            generated.push(next);
            debug!(
                step = step_i,
                token = next,
                elapsed_ms = t_step.elapsed().as_secs_f64() * 1000.0,
                cached_experts = self.experts.map.len(),
                "decode step"
            );
        }
        Ok(generated)
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
