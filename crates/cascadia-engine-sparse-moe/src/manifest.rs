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
    /// Digest of the whole `manifest.json` ([`config_digest`]), set by [`Manifest::load`]. It
    /// goes into the KV fingerprint, so a re-export under one id with any other config (say a
    /// different `partial_rotary_dim`, which no typed field above carries) refuses this export's KV.
    #[serde(skip)]
    pub config_digest: u64,
}

/// FNV-1a over `manifest.json` parsed and re-serialized (sorted keys, no whitespace), so line
/// endings or indentation cannot split one export. `exported_layers` is left out: it is a debug
/// subset that a per-rank tree may narrow, and the fingerprint must match on every rank. 0 when
/// the bytes are not a JSON object.
pub fn config_digest(bytes: &[u8]) -> u64 {
    let Ok(serde_json::Value::Object(mut v)) = serde_json::from_slice(bytes) else {
        return 0;
    };
    v.remove("exported_layers");
    let canon = serde_json::to_vec(&v).unwrap_or_default();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &canon {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
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
        let mut m: Self =
            serde_json::from_slice(&bytes).map_err(|e| ManifestError::Invalid(e.to_string()))?;
        m.config_digest = config_digest(&bytes);
        Ok(m)
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

#[cfg(test)]
mod tests {
    use super::*;

    const M2: &str = r#"{"arch": "minimax_m2", "num_layers": 3, "dense_layers": [],
        "num_experts": 8, "top_k": 2, "hidden_size": 128, "num_kv_heads": 2,
        "qk_head_dim": 32, "v_head_dim": 32, "vocab_size": 256, "eos_token_ids": [200020],
        "partial_rotary_dim": 16, "exported_layers": [0, 1, 2]}"#;

    #[test]
    fn load_digests_the_untyped_fields_too() {
        let dir = tempfile::tempdir().unwrap();
        let load = |text: &str| {
            std::fs::write(dir.path().join("manifest.json"), text).unwrap();
            Manifest::load(dir.path()).unwrap().config_digest
        };
        let fresh = load(M2);
        assert_ne!(fresh, 0);
        assert_ne!(
            fresh,
            load(&M2.replace("\"partial_rotary_dim\": 16", "\"partial_rotary_dim\": 32")),
            "a stale export with another rotary dim must not share the digest"
        );
        assert_eq!(
            fresh,
            load(&M2.replace("[0, 1, 2]", "[1]")),
            "exported_layers may differ per rank"
        );
    }
}
