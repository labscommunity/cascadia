//! Mmap-based safetensors reader that exposes K2.6 experts directly
//! from the model's on-disk format.
//!
//! Avoids duplicating the ~553 GB of expert weights — the safetensors
//! shards are already on disk and have the same packed-int4 layout the
//! kernel expects. We just need to find each (layer, expert) tensor's
//! byte range and pass slices into the existing kernel.
//!
//! Format reminder: each safetensors shard is
//!
//! ```text
//!   bytes 0..8         : header_len u64 little-endian
//!   bytes 8..8+header_len  : UTF-8 JSON metadata
//!   bytes 8+header_len.. : raw tensor data
//! ```
//!
//! Metadata format::
//!
//!   {
//!     "<tensor_name>": {
//!       "dtype": "BF16" | "I32" | ...,
//!       "shape": [...],
//!       "data_offsets": [start, end]   // relative to start of data section
//!     },
//!     "__metadata__": {...}
//!   }

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use parking_lot::RwLock;

use crate::format::GemmError;

/// Per-shard mmap + lookup table: tensor name → (data start, length) in
/// the mmap. Public because it appears in the return type of
/// `tensor_bytes` / `layer0` / `embed_tokens` (callers pin the Arc to
/// keep the returned slice valid), but the fields are all private.
pub struct Shard {
    mmap: Mmap,
    data_start: usize,
    tensors: HashMap<String, (usize, usize)>, // (data_offset_in_data_section, length)
}

impl Shard {
    fn open(path: &Path) -> Result<Self, GemmError> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < 8 {
            return Err(GemmError::Truncated {
                expected: 8,
                actual: mmap.len(),
            });
        }
        let mut hdr = [0u8; 8];
        hdr.copy_from_slice(&mmap[..8]);
        let header_len = u64::from_le_bytes(hdr) as usize;
        if 8 + header_len > mmap.len() {
            return Err(GemmError::Truncated {
                expected: 8 + header_len,
                actual: mmap.len(),
            });
        }
        let json_bytes = &mmap[8..8 + header_len];
        let json: serde_json::Value = serde_json::from_slice(json_bytes).map_err(|e| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("safetensors json parse: {e}"),
            ))
        })?;
        let map = json.as_object().ok_or_else(|| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "safetensors json not object",
            ))
        })?;
        let mut tensors = HashMap::with_capacity(map.len());
        for (k, v) in map {
            if k == "__metadata__" {
                continue;
            }
            let offsets = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| {
                    GemmError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("missing data_offsets for {k}"),
                    ))
                })?;
            if offsets.len() != 2 {
                return Err(GemmError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("data_offsets for {k} not length 2"),
                )));
            }
            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;
            if end < start {
                continue;
            }
            tensors.insert(k.clone(), (start, end - start));
        }
        Ok(Self {
            mmap,
            data_start: 8 + header_len,
            tensors,
        })
    }

    fn slice(&self, tensor_name: &str) -> Option<&[u8]> {
        self.tensors.get(tensor_name).map(|&(off, len)| {
            let start = self.data_start + off;
            &self.mmap[start..start + len]
        })
    }
}

/// Source of safetensors-backed expert weights. Caches mmaps lazily.
///
/// Construct once per process with the model directory; clone is cheap
/// (Arc'd).
#[derive(Clone)]
pub struct SafetensorsExpertSource {
    model_dir: PathBuf,
    /// Map from tensor name → shard filename, from
    /// `model.safetensors.index.json`.
    weight_map: Arc<HashMap<String, String>>,
    /// Lazy mmap cache: shard filename → Shard.
    shards: Arc<RwLock<HashMap<String, Arc<Shard>>>>,
}

impl SafetensorsExpertSource {
    /// Open the model dir and parse `model.safetensors.index.json`.
    pub fn open(model_dir: impl Into<PathBuf>) -> Result<Self, GemmError> {
        let model_dir = model_dir.into();
        let idx_path = model_dir.join("model.safetensors.index.json");
        let idx_bytes = std::fs::read(&idx_path)?;
        let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).map_err(|e| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("safetensors index json parse: {e}"),
            ))
        })?;
        let weight_map = idx
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                GemmError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "weight_map missing",
                ))
            })?;
        let mut map: HashMap<String, String> = HashMap::with_capacity(weight_map.len());
        for (k, v) in weight_map {
            if let Some(s) = v.as_str() {
                map.insert(k.clone(), s.to_string());
            }
        }
        Ok(Self {
            model_dir,
            weight_map: Arc::new(map),
            shards: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn shard_for(&self, tensor_name: &str) -> Result<Arc<Shard>, GemmError> {
        let shard_name = self.weight_map.get(tensor_name).ok_or_else(|| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {tensor_name} not in weight_map"),
            ))
        })?;
        if let Some(s) = self.shards.read().get(shard_name) {
            return Ok(Arc::clone(s));
        }
        // Race: another thread may insert it; tolerated.
        let path = self.model_dir.join(shard_name);
        let s = Arc::new(Shard::open(&path)?);
        self.shards
            .write()
            .insert(shard_name.clone(), Arc::clone(&s));
        Ok(s)
    }

    fn slice(&self, tensor_name: &str) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        let s = self.shard_for(tensor_name)?;
        let bytes = s.slice(tensor_name).ok_or_else(|| {
            GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("tensor {tensor_name} not in shard"),
            ))
        })?;
        // Cast lifetime: bytes are tied to Arc<Shard>; we return both
        // so caller pins it.
        let static_bytes: &'static [u8] = unsafe { std::mem::transmute(bytes) };
        Ok((s, static_bytes))
    }

    /// Fetch one expert's six tensor slices (gate/up/down × packed/scale).
    /// Returns a struct holding Arc references to the mmaps so the
    /// slices stay valid as long as the result is alive.
    pub fn expert(&self, layer: u32, expert: u32) -> Result<SafetensorsExpert, GemmError> {
        let base = format!(
            "language_model.model.layers.{}.mlp.experts.{}",
            layer, expert
        );
        let mut shard_pins = Vec::with_capacity(6);
        let mut slices = [&[][..]; 6];
        for (i, (proj, suf)) in [
            ("gate_proj", "weight_packed"),
            ("gate_proj", "weight_scale"),
            ("up_proj", "weight_packed"),
            ("up_proj", "weight_scale"),
            ("down_proj", "weight_packed"),
            ("down_proj", "weight_scale"),
        ]
        .iter()
        .enumerate()
        {
            let name = format!("{}.{}.{}", base, proj, suf);
            let (shard, bytes) = self.slice(&name)?;
            shard_pins.push(shard);
            slices[i] = bytes;
        }
        Ok(SafetensorsExpert {
            _pins: shard_pins,
            gate_packed: slices[0],
            gate_scale: slices[1],
            up_packed: slices[2],
            up_scale: slices[3],
            down_packed: slices[4],
            down_scale: slices[5],
        })
    }
}

/// One expert's weights backed by safetensors mmaps. The Arc refs in
/// `_pins` keep the shard mmaps alive for the byte slices.
pub struct SafetensorsExpert {
    _pins: Vec<Arc<Shard>>,
    pub gate_packed: &'static [u8],
    pub gate_scale: &'static [u8],
    pub up_packed: &'static [u8],
    pub up_scale: &'static [u8],
    pub down_packed: &'static [u8],
    pub down_scale: &'static [u8],
}

unsafe impl Send for SafetensorsExpert {}
unsafe impl Sync for SafetensorsExpert {}

impl SafetensorsExpertSource {
    /// Generic tensor lookup: returns raw bytes for any named tensor in
    /// the model. Useful for fetching shell weights (attention,
    /// layernorm, router, shared expert) which aren't expert-indexed.
    pub fn tensor_bytes(
        &self,
        tensor_name: &str,
    ) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        self.slice(tensor_name)
    }
}

/// One shell's weights for layer L. All bf16 except for one f32 bias
/// (`gate_correction_bias`). The Arc shards in `_pins` keep the
/// safetensors mmaps alive while these slices are in use.
pub struct SafetensorsShell {
    _pins: Vec<Arc<Shard>>,
    pub layer: u32,

    /// `input_layernorm.weight` — bf16 [hidden].
    pub input_norm: &'static [u8],
    /// `self_attn.q_a_proj.weight` — bf16 [q_lora_rank=1536, hidden=7168].
    pub q_a_proj: &'static [u8],
    /// `self_attn.q_a_layernorm.weight` — bf16 [q_lora_rank].
    pub q_a_norm: &'static [u8],
    /// `self_attn.q_b_proj.weight` — bf16 [heads*qk_head_dim, q_lora_rank].
    pub q_b_proj: &'static [u8],
    /// `self_attn.kv_a_proj_with_mqa.weight` — bf16 [kv_lora_rank+qk_rope, hidden].
    pub kv_a_proj: &'static [u8],
    /// `self_attn.kv_a_layernorm.weight` — bf16 [kv_lora_rank=512].
    pub kv_a_norm: &'static [u8],
    /// `self_attn.kv_b_proj.weight` — bf16 [heads*(qk_nope+v_head), kv_lora_rank].
    pub kv_b_proj: &'static [u8],
    /// `self_attn.o_proj.weight` — bf16 [hidden, heads*v_head].
    pub o_proj: &'static [u8],

    /// `post_attention_layernorm.weight` — bf16 [hidden].
    pub post_norm: &'static [u8],

    /// `mlp.gate.weight` — bf16 [n_routed_experts=384, hidden].
    pub router_weight: &'static [u8],
    /// `mlp.gate.e_score_correction_bias` — f32 [n_routed_experts].
    pub router_bias: &'static [u8],

    /// `mlp.shared_experts.gate_proj.weight` — bf16 [intermediate=2048, hidden].
    pub shared_gate: &'static [u8],
    /// `mlp.shared_experts.up_proj.weight` — bf16 [intermediate, hidden].
    pub shared_up: &'static [u8],
    /// `mlp.shared_experts.down_proj.weight` — bf16 [hidden, intermediate].
    pub shared_down: &'static [u8],
}

unsafe impl Send for SafetensorsShell {}
unsafe impl Sync for SafetensorsShell {}

/// Layer 0 of K2.6 is dense (not MoE) — same MLA attention as a shell,
/// but the MLP is a single SwiGLU instead of a router + 384 experts +
/// shared expert.
pub struct SafetensorsLayer0 {
    _pins: Vec<Arc<Shard>>,
    pub layer: u32,

    /// `input_layernorm.weight` — bf16 [hidden].
    pub input_norm: &'static [u8],
    /// `self_attn.q_a_proj.weight` — bf16 [q_lora_rank, hidden].
    pub q_a_proj: &'static [u8],
    pub q_a_norm: &'static [u8],
    pub q_b_proj: &'static [u8],
    pub kv_a_proj: &'static [u8],
    pub kv_a_norm: &'static [u8],
    pub kv_b_proj: &'static [u8],
    pub o_proj: &'static [u8],

    pub post_norm: &'static [u8],

    /// `mlp.gate_proj.weight` — bf16 [intermediate_dense=18432, hidden].
    pub gate_proj: &'static [u8],
    /// `mlp.up_proj.weight` — bf16 [intermediate_dense, hidden].
    pub up_proj: &'static [u8],
    /// `mlp.down_proj.weight` — bf16 [hidden, intermediate_dense].
    pub down_proj: &'static [u8],
}

unsafe impl Send for SafetensorsLayer0 {}
unsafe impl Sync for SafetensorsLayer0 {}

impl SafetensorsExpertSource {
    /// Fetch one layer's shell tensors. Mmaps the relevant safetensors
    /// shards lazily (with internal caching).
    pub fn shell(&self, layer: u32) -> Result<SafetensorsShell, GemmError> {
        let base = format!("language_model.model.layers.{layer}");
        let names: [&str; 14] = [
            "input_layernorm.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
            "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.up_proj.weight",
            "mlp.shared_experts.down_proj.weight",
        ];
        let mut pins = Vec::with_capacity(names.len());
        let mut slices: [&'static [u8]; 14] = [&[]; 14];
        for (i, suf) in names.iter().enumerate() {
            let full = format!("{}.{}", base, suf);
            let (shard, bytes) = self.slice(&full)?;
            pins.push(shard);
            slices[i] = bytes;
        }
        Ok(SafetensorsShell {
            _pins: pins,
            layer,
            input_norm: slices[0],
            q_a_proj: slices[1],
            q_a_norm: slices[2],
            q_b_proj: slices[3],
            kv_a_proj: slices[4],
            kv_a_norm: slices[5],
            kv_b_proj: slices[6],
            o_proj: slices[7],
            post_norm: slices[8],
            router_weight: slices[9],
            router_bias: slices[10],
            shared_gate: slices[11],
            shared_up: slices[12],
            shared_down: slices[13],
        })
    }

    /// Fetch the dense layer-0 tensors (attention + SwiGLU MLP).
    /// Mmaps the relevant safetensors shards lazily.
    pub fn layer0(&self) -> Result<SafetensorsLayer0, GemmError> {
        let layer: u32 = 0;
        let base = format!("language_model.model.layers.{layer}");
        let names: [&str; 12] = [
            "input_layernorm.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        let mut pins = Vec::with_capacity(names.len());
        let mut slices: [&'static [u8]; 12] = [&[]; 12];
        for (i, suf) in names.iter().enumerate() {
            let full = format!("{base}.{suf}");
            let (shard, bytes) = self.slice(&full)?;
            pins.push(shard);
            slices[i] = bytes;
        }
        Ok(SafetensorsLayer0 {
            _pins: pins,
            layer,
            input_norm: slices[0],
            q_a_proj: slices[1],
            q_a_norm: slices[2],
            q_b_proj: slices[3],
            kv_a_proj: slices[4],
            kv_a_norm: slices[5],
            kv_b_proj: slices[6],
            o_proj: slices[7],
            post_norm: slices[8],
            gate_proj: slices[9],
            up_proj: slices[10],
            down_proj: slices[11],
        })
    }

    /// Fetch the model's input embedding table — bf16
    /// `[vocab_size, hidden_size]`, flat row-major. Returns the
    /// pinned shard reference plus a slice into the mmap.
    pub fn embed_tokens(&self) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        self.slice("language_model.model.embed_tokens.weight")
    }

    /// Fetch the final pre-head RMSNorm weight — bf16 `[hidden_size]`.
    /// Used by the Rust head path (norm + lm_head replacement for the
    /// OV head IR).
    pub fn final_norm(&self) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        self.slice("language_model.model.norm.weight")
    }

    /// Fetch a contiguous row-slice of the lm_head weight — bf16
    /// `[vocab_end - vocab_start, hidden_size]`, flat row-major. The
    /// slice is byte-contiguous within the safetensors mmap because the
    /// row index (vocab) is the leading dim.
    ///
    /// This is the key primitive for **head tensor parallelism**: each
    /// rank loads only its slice of the vocab dimension, computes its
    /// partial logits independently, and the last rank concatenates
    /// before sampling.
    ///
    /// Caller is responsible for `vocab_start < vocab_end <= vocab_size`
    /// and `hidden_size` consistency — out-of-range slicing returns
    /// [`GemmError::Truncated`].
    pub fn lm_head_slice(
        &self,
        vocab_start: usize,
        vocab_end: usize,
        hidden_size: usize,
    ) -> Result<(Arc<Shard>, &'static [u8]), GemmError> {
        if vocab_end <= vocab_start {
            return Err(GemmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("lm_head_slice: vocab_end {vocab_end} <= vocab_start {vocab_start}"),
            )));
        }
        let (shard, full_bytes) = self.slice("language_model.lm_head.weight")?;
        let row_bytes = hidden_size * 2; // bf16
        let start = vocab_start
            .checked_mul(row_bytes)
            .ok_or_else(|| GemmError::Io(std::io::Error::other("lm_head_slice: start overflow")))?;
        let end = vocab_end
            .checked_mul(row_bytes)
            .ok_or_else(|| GemmError::Io(std::io::Error::other("lm_head_slice: end overflow")))?;
        if end > full_bytes.len() {
            return Err(GemmError::Truncated {
                expected: end,
                actual: full_bytes.len(),
            });
        }
        Ok((shard, &full_bytes[start..end]))
    }
}
