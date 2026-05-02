//! RoPE (rotary position embedding) — pure-Rust port of the
//! `transformers.LlamaRotaryEmbedding` path used by the Python ov-runtime
//! engine to compute (cos, sin) tables locally on each pipeline stage.
//!
//! Supports:
//! * Default RoPE: `inv_freq[i] = 1 / theta^(2i / head_dim)`
//! * Llama 3 scaling per `rope_scaling = {rope_type: "llama3", factor, ...}`
//!
//! Output is `(cos, sin)` each of shape `[1, seq_len, head_dim]`, dtype f32,
//! matching the Python reference. Both shapes go on the wire to the OV IR.

use serde::Deserialize;

#[derive(Debug, Deserialize, Default, Clone)]
pub struct RopeScalingConfig {
    #[serde(default, alias = "type")]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<u32>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct ModelTextConfig {
    #[serde(default)]
    pub hidden_size: u32,
    #[serde(default)]
    pub num_attention_heads: u32,
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    /// Many configs put head_dim explicitly; some derive it as
    /// `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<u32>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub max_position_embeddings: u32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    /// HF wraps multimodal LMs; if present we read these instead.
    #[serde(default)]
    pub text_config: Option<Box<ModelTextConfig>>,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

/// Pre-computed rotary state. Construct once per model load; call
/// [`compute`] for any (position_id range) needed at inference time.
pub struct Rotary {
    head_dim: usize,
    /// Per-frequency inverse: shape [head_dim/2].
    inv_freq: Vec<f32>,
}

impl Rotary {
    pub fn from_config_json(config_json: &str) -> Result<Self, String> {
        let cfg: ModelTextConfig =
            serde_json::from_str(config_json).map_err(|e| format!("config.json parse: {e}"))?;
        Self::from_config(&cfg)
    }

    pub fn from_config(cfg: &ModelTextConfig) -> Result<Self, String> {
        let cfg = if let Some(text) = &cfg.text_config {
            text.as_ref()
        } else {
            cfg
        };

        let head_dim = cfg
            .head_dim
            .map(|d| d as usize)
            .or_else(|| {
                if cfg.num_attention_heads == 0 || cfg.hidden_size == 0 {
                    None
                } else {
                    Some((cfg.hidden_size / cfg.num_attention_heads) as usize)
                }
            })
            .ok_or_else(|| "could not determine head_dim".to_string())?;

        if head_dim % 2 != 0 {
            return Err(format!("head_dim must be even (got {head_dim})"));
        }

        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let exp = (2.0 * i as f32) / head_dim as f32;
            inv_freq.push(1.0 / cfg.rope_theta.powf(exp));
        }

        if let Some(scale) = &cfg.rope_scaling {
            if scale.rope_type.as_deref() == Some("llama3") {
                Self::apply_llama3_scaling(&mut inv_freq, scale)?;
            }
            // Other scaling types (linear, dynamic, yarn) are no-ops here for now;
            // they don't apply to Llama 3.1 8B which is our primary target.
        }

        Ok(Self { head_dim, inv_freq })
    }

    fn apply_llama3_scaling(
        inv_freq: &mut [f32],
        scale: &RopeScalingConfig,
    ) -> Result<(), String> {
        let factor = scale.factor.ok_or("llama3 scaling needs factor")?;
        let low = scale.low_freq_factor.ok_or("llama3 needs low_freq_factor")?;
        let high = scale.high_freq_factor.ok_or("llama3 needs high_freq_factor")?;
        let old_ctx = scale
            .original_max_position_embeddings
            .ok_or("llama3 needs original_max_position_embeddings")?
            as f32;

        let low_freq_wavelen = old_ctx / low;
        let high_freq_wavelen = old_ctx / high;
        let pi2 = std::f32::consts::PI * 2.0;

        for f in inv_freq.iter_mut() {
            let wavelen = pi2 / *f;
            if wavelen < high_freq_wavelen {
                // High freq: keep as-is.
            } else if wavelen > low_freq_wavelen {
                // Low freq: scale down by factor.
                *f /= factor;
            } else {
                // Smooth interpolation in the middle band.
                let smooth = (old_ctx / wavelen - low) / (high - low);
                *f = (1.0 - smooth) * (*f / factor) + smooth * *f;
            }
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Compute `(cos, sin)` for positions `[start, start + seq_len)`.
    /// Each output is `[1, seq_len, head_dim]` row-major f32, matching the
    /// HF transformers convention used by Python's ov-runtime engine.
    pub fn compute(&self, start: i64, seq_len: usize) -> (Vec<f32>, Vec<f32>) {
        let head_dim = self.head_dim;
        let mut cos = vec![0.0f32; seq_len * head_dim];
        let mut sin = vec![0.0f32; seq_len * head_dim];
        let half = head_dim / 2;
        for s in 0..seq_len {
            let pos = (start + s as i64) as f32;
            let row = s * head_dim;
            for k in 0..half {
                let angle = pos * self.inv_freq[k];
                let c = angle.cos();
                let si = angle.sin();
                cos[row + k] = c;
                cos[row + k + half] = c;
                sin[row + k] = si;
                sin[row + k + half] = si;
            }
        }
        (cos, sin)
    }
}

/// Convenience: read a HuggingFace `config.json` from a directory. Falls
/// back to fields under `text_config` when the top-level config is the
/// multimodal wrapper.
pub fn load_model_config(model_dir: &std::path::Path) -> Result<ModelTextConfig, String> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("read {path:?}: {e}"))?;
    let cfg: ModelTextConfig =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {path:?}: {e}"))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama31_config() -> ModelTextConfig {
        ModelTextConfig {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            head_dim: Some(128),
            rope_theta: 500_000.0,
            max_position_embeddings: 131072,
            rope_scaling: Some(RopeScalingConfig {
                rope_type: Some("llama3".into()),
                factor: Some(8.0),
                low_freq_factor: Some(1.0),
                high_freq_factor: Some(4.0),
                original_max_position_embeddings: Some(8192),
            }),
            text_config: None,
        }
    }

    #[test]
    fn rotary_shape_is_seq_by_head_dim() {
        let cfg = llama31_config();
        let r = Rotary::from_config(&cfg).unwrap();
        let (cos, sin) = r.compute(0, 4);
        assert_eq!(r.head_dim(), 128);
        assert_eq!(cos.len(), 4 * 128);
        assert_eq!(sin.len(), 4 * 128);
    }

    #[test]
    fn rotary_pos_zero_is_one_zero() {
        // At position=0 every angle is 0 → cos=1, sin=0 throughout.
        let cfg = llama31_config();
        let r = Rotary::from_config(&cfg).unwrap();
        let (cos, sin) = r.compute(0, 1);
        for c in &cos {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in &sin {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn llama3_scaling_changes_low_frequencies() {
        // Without scaling vs with scaling should differ at low-frequency bands.
        let mut no_scale = llama31_config();
        no_scale.rope_scaling = None;
        let with_scale = llama31_config();
        let a = Rotary::from_config(&no_scale).unwrap();
        let b = Rotary::from_config(&with_scale).unwrap();
        // The first inv_freq entry is highest-frequency (smallest wavelength)
        // and should be unchanged. The last is lowest-frequency and should
        // be divided by factor.
        let last = a.inv_freq.len() - 1;
        let unscaled_low = a.inv_freq[last];
        let scaled_low = b.inv_freq[last];
        assert!(
            (scaled_low * 8.0 - unscaled_low).abs() < 1e-6,
            "low-band inv_freq should be / 8.0; unscaled={unscaled_low} scaled={scaled_low}"
        );
    }

    #[test]
    fn config_can_load_text_config_wrapper() {
        let json = r#"{
            "model_type": "llama4",
            "text_config": {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "head_dim": 128,
                "rope_theta": 500000.0,
                "max_position_embeddings": 131072
            }
        }"#;
        let cfg: ModelTextConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.text_config.is_some());
        let r = Rotary::from_config(&cfg).unwrap();
        assert_eq!(r.head_dim(), 128);
    }
}
