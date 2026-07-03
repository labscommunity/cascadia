//! Sparse-MoE model directory manifest + path resolution.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("missing manifest.json at {0}")]
    Missing(PathBuf),
    #[error("invalid manifest: {0}")]
    Invalid(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// What's in `<model_dir>/manifest.json`. Kept narrow on purpose —
/// the runtime only needs counts + relative paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Architecture name (e.g. "kimi_k2.6"). Informational.
    pub arch: String,
    /// Total number of transformer layers (e.g. 61 = 1 dense + 60 MoE).
    pub num_layers: u32,
    /// Layers that are dense MLP (no MoE). Typically [0]. The rest are MoE.
    pub dense_layers: Vec<u32>,
    /// Number of experts per MoE layer (e.g. 384).
    pub num_experts: u32,
    /// top-k routing (e.g. 8).
    pub top_k: u32,
    /// hidden_size (e.g. 7168).
    pub hidden_size: u32,
    /// num_kv_heads in the attention (e.g. 64). Cache shape uses this.
    pub num_kv_heads: u32,
    /// qk_head_dim (qk_nope + qk_rope), shape last dim of past_k (e.g. 192).
    pub qk_head_dim: u32,
    /// v_head_dim, last dim of past_v (e.g. 128).
    pub v_head_dim: u32,
    /// Vocab size (e.g. 163840).
    pub vocab_size: u32,
    /// EOS token IDs. Generation stops on any of these.
    pub eos_token_ids: Vec<u32>,
    /// Expert dispatch backend. "ov_ir" (default) loads one OV-compiled
    /// model per (layer, expert) and runs inference via the OpenVINO
    /// CPU plugin. "int4_bin" mmap's a single flat binary per expert
    /// matching the compressed-tensors on-disk format and runs the
    /// cascadia-int4-gemm AVX-512 kernel directly. The int4_bin path
    /// skips the ~5 ms OV per-call overhead and is ~6x faster on the
    /// Xeon bench host for tiny experts like K2.6's 2048×7168 MLPs.
    #[serde(default = "default_experts_format")]
    pub experts_format: String,
    /// Shell execution backend. "rust_k26" (default) runs the hardcoded
    /// K2.6 MLA Rust kernel reading shell weights from safetensors.
    /// "ov_ir" runs each layer's attention+routing as a compiled
    /// OpenVINO shell graph (the [`crate::ov_moe`] path) — used for
    /// architectures whose math lives in the traced graph (e.g.
    /// MiniMax-M2: GQA + QK-norm + partial RoPE + sigmoid routing).
    #[serde(default = "default_shell_backend")]
    pub shell_backend: String,
    /// Subset of layer ids actually exported (debug/partial exports).
    /// Empty = all `0..num_layers`. Only consulted by the ov_ir backend.
    #[serde(default)]
    pub exported_layers: Vec<u32>,
    /// Expert FFN intermediate dim (e.g. 1536 for M2). Needed by the
    /// `int4_bin` backend to compute per-expert byte offsets. 0 = unset.
    #[serde(default)]
    pub expert_intermediate: u32,
}

fn default_experts_format() -> String {
    "ov_ir".to_string()
}

fn default_shell_backend() -> String {
    "rust_k26".to_string()
}

impl Manifest {
    pub fn load(model_dir: &Path) -> Result<Self, ManifestError> {
        let p = model_dir.join("manifest.json");
        if !p.exists() {
            return Err(ManifestError::Missing(p));
        }
        let bytes = std::fs::read(&p)?;
        serde_json::from_slice(&bytes).map_err(|e| ManifestError::Invalid(e.to_string()))
    }

    /// Path to the stateless layer-0 IR (embed + dense + RMSNorm).
    pub fn layer0_xml(&self, model_dir: &Path) -> PathBuf {
        model_dir.join("layer0").join("openvino_model.xml")
    }

    /// Path to the final RMSNorm + lm_head IR.
    pub fn head_xml(&self, model_dir: &Path) -> PathBuf {
        model_dir.join("head").join("openvino_model.xml")
    }

    /// Path to the MoE shell-kv for layer `lid`.
    pub fn shell_xml(&self, model_dir: &Path, lid: u32) -> PathBuf {
        model_dir
            .join("shells")
            .join(format!("layer_{:02}", lid))
            .join("openvino_model.xml")
    }

    /// Path to the GPU-friendly shell-core IR for layer `lid` (attention +
    /// KV, no router) — produced by `tools/split_m2_shells.py`. Present only
    /// when the shells have been split for non-CPU (iGPU/NPU) execution.
    pub fn shell_core_xml(&self, model_dir: &Path, lid: u32) -> PathBuf {
        model_dir
            .join("shells")
            .join(format!("layer_{:02}", lid))
            .join("shell_core.xml")
    }

    /// Path to the CPU router IR for layer `lid` (`apn_in` -> routing
    /// ids/weights), the sibling of [`Self::shell_core_xml`].
    pub fn router_xml(&self, model_dir: &Path, lid: u32) -> PathBuf {
        model_dir
            .join("shells")
            .join(format!("layer_{:02}", lid))
            .join("router.xml")
    }

    /// Path to one per-(layer, expert) IR.
    pub fn expert_xml(&self, model_dir: &Path, lid: u32, eid: u32) -> PathBuf {
        model_dir
            .join("experts")
            .join(format!("layer_{:02}", lid))
            .join(format!("expert_{:03}", eid))
            .join("openvino_model.xml")
    }

    /// Path to one per-(layer, expert) flat int4 binary (for the
    /// `experts_format = "int4_bin"` path).
    pub fn expert_bin(&self, model_dir: &Path, lid: u32, eid: u32) -> PathBuf {
        model_dir
            .join("experts")
            .join(format!("layer_{:02}", lid))
            .join(format!("expert_{:03}.bin", eid))
    }

    /// Convenience: the list of MoE layer indices (everything not in dense_layers).
    pub fn moe_layer_ids(&self) -> Vec<u32> {
        (0..self.num_layers)
            .filter(|i| !self.dense_layers.contains(i))
            .collect()
    }

    /// True when shells should run as compiled OV-IR graphs (MiniMax-M2
    /// and other arch-in-graph models) rather than the K2.6 Rust kernel.
    pub fn is_ov_shell(&self) -> bool {
        self.shell_backend == "ov_ir"
    }

    /// Layers the ov_ir backend should compile shells for. Falls back to
    /// every layer when `exported_layers` is empty.
    pub fn ov_layer_ids(&self) -> Vec<u32> {
        if self.exported_layers.is_empty() {
            (0..self.num_layers).collect()
        } else {
            self.exported_layers.clone()
        }
    }
}
