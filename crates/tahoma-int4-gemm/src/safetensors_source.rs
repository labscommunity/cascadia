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
/// the mmap.
struct Shard {
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
            GemmError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("safetensors json parse: {e}")))
        })?;
        let map = json.as_object().ok_or_else(|| {
            GemmError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "safetensors json not object"))
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
            GemmError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("safetensors index json parse: {e}")))
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
        self.shards.write().insert(shard_name.clone(), Arc::clone(&s));
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
