//! Load a `tools/export_kimi_k3.py` export into the structs [`crate::k3::model`]
//! runs.
//!
//! ```text
//! <dir>/manifest.json                      (arch == "kimi_k3")
//! <dir>/embed.safetensors                  rank 0 only
//! <dir>/head.safetensors                   last rank only
//! <dir>/shells/layer_NN.safetensors        per-layer bf16 shell
//! <dir>/experts/layer_NN/expert_EEE.bin    fp4 e2m1 + E8M0 scales
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

use crate::dsv4::st::StFile;
use crate::k3::attn::{MlaDims, MlaKv, MlaWeights};
use crate::k3::expert_fp4;
use crate::k3::model::{
    K3Dims, K3Layer, KdaDims, KdaState, KdaWeights, LayerAttn, LayerFfn, LayerState,
};
use crate::k3::moe::{MmapExperts, MoeDims, MoeWeights};

#[derive(Debug, Error)]
pub enum K3LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest: {0}")]
    Manifest(String),
    #[error("tensor {0}: {1}")]
    Tensor(String, String),
}

/// `<dir>/manifest.json` as the k3 shell needs it.
#[derive(Debug, Clone, Deserialize)]
pub struct K3Manifest {
    pub arch: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub first_k_dense_replace: usize,
    pub attn_res_block_size: usize,
    pub kda_layers: Vec<usize>,
    pub rms_norm_eps: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: Option<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_size: usize,
    pub gate_lower_bound: Option<f32>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub num_shared_experts: usize,
    pub routed_expert_hidden_size: usize,
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub latent_moe_use_norm: bool,
    pub routed_scaling_factor: f32,
    pub moe_renormalize: bool,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
}

impl K3Manifest {
    pub fn load(dir: &Path) -> Result<Self, K3LoadError> {
        let p = dir.join("manifest.json");
        let b = std::fs::read(&p)?;
        let m: Self = serde_json::from_slice(&b)
            .map_err(|e| K3LoadError::Manifest(format!("{}: {e}", p.display())))?;
        if m.arch != "kimi_k3" {
            return Err(K3LoadError::Manifest(format!(
                "arch is {:?}, expected \"kimi_k3\"",
                m.arch
            )));
        }
        Ok(m)
    }

    pub fn is_kda(&self, layer: usize) -> bool {
        self.kda_layers.contains(&layer)
    }

    pub fn attn_heads(&self) -> usize {
        self.num_attention_heads.unwrap_or(self.num_heads)
    }

    pub fn dims(&self) -> K3Dims {
        K3Dims {
            hidden: self.hidden_size,
            num_layers: self.num_hidden_layers,
            block_size: self.attn_res_block_size,
            eps: self.rms_norm_eps,
            situ_beta: self.situ_beta,
            situ_linear_beta: self.situ_linear_beta,
        }
    }

    pub fn kda_dims(&self) -> KdaDims {
        KdaDims {
            hidden: self.hidden_size,
            heads: self.num_heads,
            head_dim: self.head_dim,
            conv_size: self.conv_size,
            gate_lower_bound: self.gate_lower_bound,
            eps: self.rms_norm_eps,
        }
    }

    pub fn mla_dims(&self) -> MlaDims {
        MlaDims {
            hidden: self.hidden_size,
            heads: self.attn_heads(),
            q_lora_rank: self.q_lora_rank,
            kv_lora_rank: self.kv_lora_rank,
            qk_nope: self.qk_nope_head_dim,
            qk_rope: self.qk_rope_head_dim,
            v_head: self.v_head_dim,
            eps: self.rms_norm_eps,
        }
    }

    pub fn moe_dims(&self) -> MoeDims {
        MoeDims {
            hidden: self.hidden_size,
            latent: self.routed_expert_hidden_size,
            inter: self.moe_intermediate_size,
            n_experts: self.num_experts,
            top_k: self.top_k,
            n_shared: self.num_shared_experts,
            scale: self.routed_scaling_factor,
            renormalize: self.moe_renormalize,
            situ_beta: self.situ_beta,
            situ_linear_beta: self.situ_linear_beta,
            eps: self.rms_norm_eps,
            use_norm: self.latent_moe_use_norm,
        }
    }
}

fn f32s(f: &StFile, name: &str) -> Result<Vec<f32>, K3LoadError> {
    f.f32(name)
        .map(|(_, v)| v)
        .map_err(|e| K3LoadError::Tensor(name.into(), e.to_string()))
}

/// Read an f32 tensor and store it bf16 — the shell's weight precision.
fn bf16s(f: &StFile, name: &str) -> Result<Vec<u16>, K3LoadError> {
    Ok(f32s(f, name)?
        .into_iter()
        .map(|v| half::bf16::from_f32(v).to_bits())
        .collect())
}

/// Embedding table (rank 0).
pub fn load_embed(dir: &Path) -> Result<Vec<f32>, K3LoadError> {
    let f = StFile::open(&dir.join("embed.safetensors"))
        .map_err(|e| K3LoadError::Tensor("embed".into(), e.to_string()))?;
    f32s(&f, "embed")
}

/// Final norm, lm_head, and the model-level AttnRes pair (last rank).
pub struct K3Head {
    pub norm: Vec<f32>,
    pub lm_head: Vec<u16>,
    pub out_res_proj: Vec<f32>,
    pub out_res_norm: Vec<f32>,
}

pub fn load_head(dir: &Path) -> Result<K3Head, K3LoadError> {
    let f = StFile::open(&dir.join("head.safetensors"))
        .map_err(|e| K3LoadError::Tensor("head".into(), e.to_string()))?;
    Ok(K3Head {
        norm: f32s(&f, "norm")?,
        lm_head: bf16s(&f, "lm_head")?,
        out_res_proj: f32s(&f, "output_attn_res_proj")?,
        out_res_norm: f32s(&f, "output_attn_res_norm")?,
    })
}

fn shell_path(dir: &Path, layer: usize) -> PathBuf {
    dir.join("shells")
        .join(format!("layer_{layer:02}.safetensors"))
}

/// Map one MoE layer's expert bin. Never reads it into RAM — a real layer is
/// ~15.7 GB and the model is ~1.45 TB.
fn load_experts(dir: &Path, layer: usize, m: &K3Manifest) -> Result<MmapExperts, K3LoadError> {
    let stride = expert_fp4::expert_bytes(m.routed_expert_hidden_size, m.moe_intermediate_size);
    let p = dir.join("experts").join(format!("layer_{layer:02}.bin"));
    MmapExperts::open(&p, stride, m.num_experts)
        .map_err(|e| K3LoadError::Tensor(p.display().to_string(), e.to_string()))
}

/// Load layers `lo..hi` of an export.
pub fn load_layers(
    dir: &Path,
    m: &K3Manifest,
    lo: usize,
    hi: usize,
) -> Result<(Vec<K3Layer<MmapExperts>>, Vec<LayerState>), K3LoadError> {
    let mut layers = Vec::with_capacity(hi - lo);
    let mut states = Vec::with_capacity(hi - lo);

    for idx in lo..hi {
        let f = StFile::open(&shell_path(dir, idx))
            .map_err(|e| K3LoadError::Tensor(format!("shell {idx}"), e.to_string()))?;

        let (attn, state) = if m.is_kda(idx) {
            let w = KdaWeights {
                q_proj: bf16s(&f, "attn.q_proj")?,
                k_proj: bf16s(&f, "attn.k_proj")?,
                v_proj: bf16s(&f, "attn.v_proj")?,
                q_conv1d: f32s(&f, "attn.q_conv1d")?,
                k_conv1d: f32s(&f, "attn.k_conv1d")?,
                v_conv1d: f32s(&f, "attn.v_conv1d")?,
                f_a_proj: bf16s(&f, "attn.f_a_proj")?,
                f_b_proj: bf16s(&f, "attn.f_b_proj")?,
                a_log: f32s(&f, "attn.A_log")?,
                dt_bias: f32s(&f, "attn.dt_bias")?,
                b_proj: bf16s(&f, "attn.b_proj")?,
                g_proj: bf16s(&f, "attn.g_proj")?,
                o_norm: f32s(&f, "attn.o_norm")?,
                o_proj: bf16s(&f, "attn.o_proj")?,
            };
            let d = m.kda_dims();
            (
                LayerAttn::Kda(Box::new(w), d),
                LayerState::Kda(Box::new(KdaState::new(d))),
            )
        } else {
            let w = MlaWeights {
                q_a_proj: bf16s(&f, "attn.q_a_proj")?,
                q_a_layernorm: f32s(&f, "attn.q_a_layernorm")?,
                q_b_proj: bf16s(&f, "attn.q_b_proj")?,
                kv_a_proj_with_mqa: bf16s(&f, "attn.kv_a_proj_with_mqa")?,
                kv_a_layernorm: f32s(&f, "attn.kv_a_layernorm")?,
                kv_b_proj: bf16s(&f, "attn.kv_b_proj")?,
                g_proj: bf16s(&f, "attn.g_proj")?,
                o_proj: bf16s(&f, "attn.o_proj")?,
            };
            (
                LayerAttn::Mla(Box::new(w), m.mla_dims()),
                LayerState::Mla(MlaKv::default()),
            )
        };

        let ffn = if idx >= m.first_k_dense_replace {
            let w = MoeWeights {
                gate: f32s(&f, "moe.gate_weight")?,
                e_score_correction_bias: f32s(&f, "moe.e_score_correction_bias")?,
                down_proj: bf16s(&f, "moe.routed_expert_down_proj")?,
                up_proj: bf16s(&f, "moe.routed_expert_up_proj")?,
                norm: f32s(&f, "moe.routed_expert_norm")?,
                shared_w1: bf16s(&f, "moe.shared_w1")?,
                shared_w3: bf16s(&f, "moe.shared_w3")?,
                shared_w2: bf16s(&f, "moe.shared_w2")?,
            };
            LayerFfn::Moe(Box::new(w), m.moe_dims(), load_experts(dir, idx, m)?)
        } else {
            LayerFfn::Dense {
                w1: bf16s(&f, "w1")?,
                w3: bf16s(&f, "w3")?,
                w2: bf16s(&f, "w2")?,
                inter: m.intermediate_size,
            }
        };

        layers.push(K3Layer {
            idx,
            input_layernorm: f32s(&f, "input_layernorm")?,
            post_attention_layernorm: f32s(&f, "post_attention_layernorm")?,
            attn_res_proj: f32s(&f, "attn_res_proj")?,
            attn_res_norm: f32s(&f, "attn_res_norm")?,
            mlp_res_proj: f32s(&f, "mlp_res_proj")?,
            mlp_res_norm: f32s(&f, "mlp_res_norm")?,
            attn,
            ffn,
        });
        states.push(state);
    }
    Ok((layers, states))
}
