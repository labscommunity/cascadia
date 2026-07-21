//! Minimal safetensors reader for dsv4 shells and test fixtures.
//!
//! Unlike `cascadia_int4_gemm::safetensors_source::Shard` (mmap'd, expert-
//! oriented, K2.6-shaped accessors), this is a small generic name -> tensor
//! reader: whole-file load, header parse, dtype-aware decode to f32/i32.
//! Shell files are a few hundred MB at most; fixtures are kilobytes.

use std::collections::HashMap;
use std::path::Path;

use half::bf16;

#[derive(Debug, thiserror::Error)]
pub enum StError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad safetensors header: {0}")]
    Header(String),
    #[error("tensor not found: {0}")]
    NotFound(String),
    #[error("dtype {0} unsupported for {1}")]
    Dtype(String, String),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// Whole-file safetensors: header + owned data buffer.
pub struct StFile {
    data: Vec<u8>,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
}

impl StFile {
    pub fn open(path: &Path) -> Result<Self, StError> {
        let data = std::fs::read(path)?;
        if data.len() < 8 {
            return Err(StError::Header("file shorter than header length".into()));
        }
        let mut hdr = [0u8; 8];
        hdr.copy_from_slice(&data[..8]);
        let hlen = u64::from_le_bytes(hdr) as usize;
        if 8 + hlen > data.len() {
            return Err(StError::Header("header length exceeds file".into()));
        }
        let json: serde_json::Value = serde_json::from_slice(&data[8..8 + hlen])
            .map_err(|e| StError::Header(e.to_string()))?;
        let obj = json
            .as_object()
            .ok_or_else(|| StError::Header("header not a json object".into()))?;
        let mut tensors = HashMap::with_capacity(obj.len());
        for (k, v) in obj {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| StError::Header(format!("{k}: missing dtype")))?
                .to_string();
            let shape = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| StError::Header(format!("{k}: missing shape")))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| StError::Header(format!("{k}: missing data_offsets")))?;
            let start = off[0].as_u64().unwrap_or(0) as usize;
            let end = off[1].as_u64().unwrap_or(0) as usize;
            tensors.insert(
                k.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            data,
            data_start: 8 + hlen,
            tensors,
        })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, StError> {
        self.tensors
            .get(name)
            .ok_or_else(|| StError::NotFound(name.into()))
    }

    fn bytes(&self, name: &str) -> Result<(&TensorInfo, &[u8]), StError> {
        let ti = self.info(name)?;
        Ok((
            ti,
            &self.data[self.data_start + ti.start..self.data_start + ti.end],
        ))
    }

    /// Decode to f32 (accepts F32, BF16, F16).
    pub fn f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), StError> {
        let (ti, b) = self.bytes(name)?;
        let out = match ti.dtype.as_str() {
            "F32" => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "BF16" => b
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            "F16" => b
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => return Err(StError::Dtype(other.into(), name.into())),
        };
        Ok((ti.shape.clone(), out))
    }

    /// Decode to i32 (accepts I32, I64 with lossy narrow).
    pub fn i32(&self, name: &str) -> Result<(Vec<usize>, Vec<i32>), StError> {
        let (ti, b) = self.bytes(name)?;
        let out = match ti.dtype.as_str() {
            "I32" => b
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "I64" => b
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as i32
                })
                .collect(),
            other => return Err(StError::Dtype(other.into(), name.into())),
        };
        Ok((ti.shape.clone(), out))
    }
}
