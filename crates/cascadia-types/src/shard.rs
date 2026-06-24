use serde::{Deserialize, Serialize};

use crate::Error;

/// One rank's slice of a model — what one Engine will run.
///
/// For pure pipeline parallelism (`tp_size == 1`) this is a contiguous range
/// of layers. When `tp_size > 1`, the same layer range is held by `tp_size`
/// peers, each with a 1/N slice of every weight matrix; engines that opt
/// into TP must perform an all-reduce after attention output and MLP output.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardSpec {
    pub model_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    /// OV device hint. Forwarded verbatim to `ov::Core::compile_model`.
    /// Valid forms: `"CPU"`, `"GPU"`, `"GPU.N"`, `"NPU"`, `"NPU.N"`,
    /// `"AUTO"`, `"MULTI:dev1,dev2,..."`, `"HETERO:dev1,dev2,..."`,
    /// `"BATCH:dev"`. See `cascadia doctor` for available devices.
    pub device: String,
    pub is_first_stage: bool,
    pub is_last_stage: bool,
    #[serde(default = "one")]
    pub tp_size: u32,
    #[serde(default)]
    pub tp_rank: u32,
}

fn one() -> u32 {
    1
}

impl ShardSpec {
    pub fn num_layers(&self) -> u32 {
        self.layer_end.saturating_sub(self.layer_start)
    }

    pub fn is_tp(&self) -> bool {
        self.tp_size > 1
    }

    /// Single-stage shard for engines that build their own layer plan
    /// internally (e.g. `ov-genai` reading a packaged IR directory).
    pub fn single_stage(model_id: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            layer_start: 0,
            layer_end: 0,
            total_layers: 0,
            device: device.into(),
            is_first_stage: true,
            is_last_stage: true,
            tp_size: 1,
            tp_rank: 0,
        }
    }
}

/// Cluster-wide pipeline layout: one `ShardSpec` per stage.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardPlan {
    pub model_id: String,
    pub total_layers: u32,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub vocab_size: u32,
    pub stages: Vec<ShardSpec>,
}

impl ShardPlan {
    pub fn total_stages(&self) -> usize {
        self.stages.len()
    }

    /// Even split of `total_layers` across `num_stages`. Trailing layers
    /// (when `total_layers % num_stages != 0`) go to the last stage.
    /// `devices` is a per-stage device hint; defaults to all CPU.
    pub fn uniform(
        model_id: impl Into<String>,
        total_layers: u32,
        hidden_size: u32,
        num_attention_heads: u32,
        vocab_size: u32,
        num_stages: u32,
        devices: Option<Vec<String>>,
    ) -> Result<Self, Error> {
        if num_stages < 1 {
            return Err(Error::InvalidShard(format!(
                "num_stages must be >= 1 (got {num_stages})"
            )));
        }
        if num_stages > total_layers {
            return Err(Error::InvalidShard(format!(
                "num_stages ({num_stages}) exceeds total_layers ({total_layers})"
            )));
        }

        let model_id = model_id.into();
        let devices = devices.unwrap_or_else(|| vec!["CPU".to_string(); num_stages as usize]);
        if devices.len() != num_stages as usize {
            return Err(Error::InvalidShard(format!(
                "devices ({}) must match num_stages ({num_stages})",
                devices.len()
            )));
        }

        let per_stage = total_layers / num_stages;
        let mut cursor = 0u32;
        let mut stages = Vec::with_capacity(num_stages as usize);
        for i in 0..num_stages {
            let count = if i < num_stages - 1 {
                per_stage
            } else {
                total_layers - cursor
            };
            stages.push(ShardSpec {
                model_id: model_id.clone(),
                layer_start: cursor,
                layer_end: cursor + count,
                total_layers,
                device: devices[i as usize].clone(),
                is_first_stage: i == 0,
                is_last_stage: i == num_stages - 1,
                tp_size: 1,
                tp_rank: 0,
            });
            cursor += count;
        }

        Ok(Self {
            model_id,
            total_layers,
            hidden_size,
            num_attention_heads,
            vocab_size,
            stages,
        })
    }
}
