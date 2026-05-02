//! Distributed speculative decoding (v5 shards, mask-based KV-cache rewind).
//!
//! Rust port of `tahoma/worker/engines/openvino/dist_spec.py` +
//! `dist_spec_protocol.py` + `spec_decode.py`. See those files for the
//! complete design rationale; the architecture-level summary:
//!
//! * **Driver (rank 0)** runs the spec-decode loop locally: holds a small
//!   draft model + a `DistributedMaskedReq` that wraps the multi-stage
//!   target across N stages.
//! * **Workers (rank 1..N-1)** run a per-stage portion of the target and
//!   service `FORWARD` / `RESET` frames from upstream. Last stage applies
//!   `lm_head` and returns logits.
//!
//! Wire frames (mirrors `dist_spec_protocol.py` exactly):
//! ```text
//! FORWARD          [4B kind=1 BE][4B logical_pos_start BE]
//!                  + send_tensor(attention_mask)   # i64 [1, total_seq_len]
//!                  + send_tensor(hidden_states)    # f16 [1, new_tokens, hidden_size]
//!
//! RESET            [4B kind=3 BE]
//!
//! LOGITS_RESPONSE  [4B kind=4 BE]
//!                  + send_tensor(logits)           # f16 [1, new_tokens, vocab_size]
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use serde::Deserialize;
use tahoma_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use tahoma_ov_genai_shim::{
    DType as ShimDType, Error as OvError, PluginConfig, Runtime as OvRuntime,
};
use tahoma_transport::{
    recv_tensor, send_tensor, ActivationClient, ActivationServer, DType as WireDType,
    Tensor as WireTensor, MAX_RANK,
};
use tahoma_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use tokenizers::Tokenizer;
use tokio::net::TcpStream;
use tracing::{info, warn};

// -------- frame protocol --------

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Forward = 1,
    Reset = 3,
    LogitsResponse = 4,
}

impl FrameKind {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Forward),
            3 => Some(Self::Reset),
            4 => Some(Self::LogitsResponse),
            _ => None,
        }
    }
}

// -------- helpers --------

fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        let h = f16::from_f32(*x);
        out.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    out
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    use half::f16;
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn i64_to_bytes(v: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn argmax(slice: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in slice.iter().enumerate() {
        if *v > best_v {
            best_v = *v;
            best_i = i;
        }
    }
    best_i
}

fn map_ov_err(err: OvError) -> EngineError {
    match err {
        OvError::Stub => EngineError::Backend(
            "openvino shim built without --features openvino".into(),
        ),
        OvError::Utf8(s) => EngineError::InvalidConfig(s),
        OvError::Native(s) => EngineError::Backend(s),
    }
}

/// Bridge sync code to an async future. Works whether we're called
/// from inside a tokio worker thread (e.g. `ChunkStream::poll_next` on
/// the driver) or from a plain blocking thread (e.g.
/// `Runner::run_relay_loop` via `spawn_blocking` on workers). Without
/// the dispatch, calling `block_on` from a worker thread panics with
/// "Cannot start a runtime from within a runtime".
fn run_async<F: std::future::Future>(handle: &tokio::runtime::Handle, fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(_) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => handle.block_on(fut),
    }
}

// -------- pipeline / stage config --------

#[derive(Debug, Deserialize)]
struct PipelineConfig {
    #[allow(dead_code)]
    model_id: String,
    #[serde(default)]
    export_version: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct StageConfig {
    #[serde(default)]
    has_embed: bool,
    #[serde(default)]
    has_head: bool,
    #[serde(default)]
    export_version: Option<String>,
}

fn read_pipeline_config(p: &std::path::Path) -> Result<PipelineConfig, EngineError> {
    let bytes = std::fs::read(p.join("pipeline_config.json"))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| EngineError::InvalidConfig(format!("pipeline_config.json: {e}")))
}

fn read_stage_config(p: &std::path::Path) -> Result<StageConfig, EngineError> {
    let bytes = std::fs::read(p.join("stage_config.json"))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| EngineError::InvalidConfig(format!("stage_config.json: {e}")))
}

fn v5_inputs(
    runtime: &OvRuntime,
) -> Result<std::collections::HashMap<String, String>, EngineError> {
    use std::collections::HashMap;
    let n_inputs = runtime.input_count();
    let mut map: HashMap<String, String> = HashMap::new();
    // Mirrors the Python _v5_inputs(): for each input port, check ALL
    // its aliases against each canonical name; if a match is found,
    // map the canonical name -> the port's primary (first) name.
    for canonical in [
        "input_ids",
        "hidden_states",
        "attention_mask",
        "position_ids",
        "beam_idx",
    ] {
        for idx in 0..n_inputs {
            let aliases = runtime.input_aliases(idx).map_err(map_ov_err)?;
            let primary = runtime.input_name(idx).map_err(map_ov_err)?;
            if aliases
                .iter()
                .any(|a| a == canonical || a.contains(canonical))
            {
                map.insert(canonical.to_string(), primary);
                break;
            }
        }
    }
    Ok(map)
}

// -------- frame send / recv on the underlying TcpStream --------

async fn send_forward(
    sock: &mut TcpStream,
    logical_pos_start: u32,
    attn_mask_total: usize,
    attn_mask: &[i64],
    hidden_shape: [u32; MAX_RANK],
    hidden_data: Vec<u8>,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut header = [0u8; 8];
    header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
    header[4..8].copy_from_slice(&logical_pos_start.to_be_bytes());
    sock.write_all(&header).await?;
    let attn_tensor = WireTensor::new(
        WireDType::I64,
        [1, 1, attn_mask_total as u32],
        i64_to_bytes(attn_mask),
    );
    send_tensor(sock, &attn_tensor).await.map_err(io_err)?;
    let hidden_tensor = WireTensor::new(WireDType::F16, hidden_shape, hidden_data);
    send_tensor(sock, &hidden_tensor).await.map_err(io_err)?;
    Ok(())
}

async fn send_reset(sock: &mut TcpStream) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let bytes = (FrameKind::Reset as u32).to_be_bytes();
    sock.write_all(&bytes).await?;
    sock.flush().await?;
    Ok(())
}

async fn send_logits(
    sock: &mut TcpStream,
    logits_shape: [u32; MAX_RANK],
    logits_data: Vec<u8>,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let bytes = (FrameKind::LogitsResponse as u32).to_be_bytes();
    sock.write_all(&bytes).await?;
    let tensor = WireTensor::new(WireDType::F16, logits_shape, logits_data);
    send_tensor(sock, &tensor).await.map_err(io_err)?;
    Ok(())
}

async fn recv_kind(sock: &mut TcpStream) -> std::io::Result<FrameKind> {
    use tokio::io::AsyncReadExt;
    let mut buf = [0u8; 4];
    sock.read_exact(&mut buf).await?;
    let v = u32::from_be_bytes(buf);
    FrameKind::from_u32(v).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad kind: {v}"))
    })
}

/// After `recv_kind` returned `Forward`, read the rest of the body.
/// Returns `(logical_pos_start, attention_mask, hidden_states_f32, hidden_shape)`.
async fn recv_forward_body(
    sock: &mut TcpStream,
) -> std::io::Result<(u32, Vec<i64>, Vec<f32>, [usize; MAX_RANK])> {
    use tokio::io::AsyncReadExt;
    let mut pos_buf = [0u8; 4];
    sock.read_exact(&mut pos_buf).await?;
    let logical_pos_start = u32::from_be_bytes(pos_buf);
    let (attn, _) = recv_tensor(sock).await.map_err(io_err)?;
    let attn_mask = bytes_to_i64(&attn.data);
    let (hidden, _) = recv_tensor(sock).await.map_err(io_err)?;
    let hs_f32 = match hidden.dtype {
        WireDType::F32 => hidden
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        WireDType::F16 => f16_bytes_to_f32(&hidden.data),
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unexpected hidden dtype {other:?}"),
            ))
        }
    };
    let shape = [
        hidden.shape[0] as usize,
        hidden.shape[1] as usize,
        hidden.shape[2] as usize,
    ];
    Ok((logical_pos_start, attn_mask, hs_f32, shape))
}

async fn recv_logits_body(sock: &mut TcpStream) -> std::io::Result<(Vec<f32>, [usize; MAX_RANK])> {
    let (t, _) = recv_tensor(sock).await.map_err(io_err)?;
    let f = match t.dtype {
        WireDType::F32 => t
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        WireDType::F16 => f16_bytes_to_f32(&t.data),
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unexpected logits dtype {other:?}"),
            ))
        }
    };
    Ok((
        f,
        [
            t.shape[0] as usize,
            t.shape[1] as usize,
            t.shape[2] as usize,
        ],
    ))
}

fn io_err(e: tahoma_transport::TransportError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}

// -------- MaskedReq (local draft / target wrapper with mask-based rewind) --------

pub struct MaskedReq {
    runtime: OvRuntime,
    has_beam: bool,
    valid_mask: Vec<i64>,
    cache_len: usize,
    logical_pos: usize,
    inputs: std::collections::HashMap<String, String>,
}

impl MaskedReq {
    pub fn new(runtime: OvRuntime) -> Result<Self, EngineError> {
        let n_inputs = runtime.input_count();
        let mut inputs = std::collections::HashMap::new();
        let mut has_beam = false;
        for canonical in ["input_ids", "attention_mask", "position_ids", "beam_idx"] {
            for idx in 0..n_inputs {
                let aliases = runtime.input_aliases(idx).map_err(map_ov_err)?;
                let primary = runtime.input_name(idx).map_err(map_ov_err)?;
                if aliases
                    .iter()
                    .any(|a| a == canonical || a.contains(canonical))
                {
                    if canonical == "beam_idx" {
                        has_beam = true;
                    }
                    inputs.insert(canonical.to_string(), primary);
                    break;
                }
            }
        }
        Ok(Self {
            runtime,
            has_beam,
            valid_mask: vec![1i64; 4096],
            cache_len: 0,
            logical_pos: 0,
            inputs,
        })
    }

    pub fn reset(&mut self) -> Result<(), EngineError> {
        self.runtime.reset_state().map_err(map_ov_err)?;
        for m in self.valid_mask.iter_mut() {
            *m = 1;
        }
        self.cache_len = 0;
        self.logical_pos = 0;
        Ok(())
    }

    pub fn feed(&mut self, input_ids: &[i64]) -> Result<(Vec<f32>, [usize; 3]), EngineError> {
        let n = input_ids.len();
        let total = self.cache_len + n;
        if total > self.valid_mask.len() {
            let new_size = (total * 2).max(self.valid_mask.len() * 2);
            self.valid_mask.resize(new_size, 1);
        }
        let mut attn = vec![0i64; total];
        attn[..self.cache_len].copy_from_slice(&self.valid_mask[..self.cache_len]);
        for v in attn[self.cache_len..].iter_mut() {
            *v = 1;
        }
        let pos: Vec<i64> = (self.logical_pos as i64..(self.logical_pos + n) as i64).collect();

        let in_ids_name = self
            .inputs
            .get("input_ids")
            .ok_or_else(|| EngineError::Backend("missing input_ids name".into()))?
            .clone();
        let attn_name = self
            .inputs
            .get("attention_mask")
            .ok_or_else(|| EngineError::Backend("missing attention_mask name".into()))?
            .clone();
        let pos_name = self
            .inputs
            .get("position_ids")
            .ok_or_else(|| EngineError::Backend("missing position_ids name".into()))?
            .clone();

        self.runtime
            .set_input(&in_ids_name, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&attn_name, ShimDType::I64, &[1, total], &i64_to_bytes(&attn))
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&pos_name, ShimDType::I64, &[1, n], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        if self.has_beam {
            if let Some(beam_name) = self.inputs.get("beam_idx").cloned() {
                let beam = vec![0i32];
                let mut bytes = Vec::with_capacity(4);
                for v in &beam {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                self.runtime
                    .set_input(&beam_name, ShimDType::I32, &[1], &bytes)
                    .map_err(map_ov_err)?;
            }
        }
        self.runtime.infer().map_err(map_ov_err)?;
        let (dtype, shape, bytes) = self.runtime.output(0).map_err(map_ov_err)?;
        let floats = match dtype {
            ShimDType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
            ShimDType::F16 => f16_bytes_to_f32(&bytes),
            other => {
                return Err(EngineError::Backend(format!(
                    "unexpected output dtype {other:?}"
                )))
            }
        };
        self.cache_len += n;
        self.logical_pos += n;
        let mut out_shape = [1, 1, floats.len()];
        if shape.len() == 3 {
            out_shape = [shape[0], shape[1], shape[2]];
        } else if shape.len() == 2 {
            out_shape = [1, shape[0], shape[1]];
        }
        Ok((floats, out_shape))
    }

    pub fn rewind(&mut self, k: usize) {
        if k == 0 {
            return;
        }
        let lo = self.cache_len.saturating_sub(k);
        for i in lo..self.cache_len {
            self.valid_mask[i] = 0;
        }
        self.logical_pos = self.logical_pos.saturating_sub(k);
    }
}

// -------- DistributedMaskedReq (driver-side wrapper for multi-stage target) --------

pub struct DistributedMaskedReq {
    stage0: OvRuntime,
    stage0_inputs: std::collections::HashMap<String, String>,
    downstream: Arc<tokio::sync::Mutex<ActivationClient>>,
    runtime_handle: tokio::runtime::Handle,
    valid_mask: Vec<i64>,
    cache_len: usize,
    logical_pos: usize,
}

impl DistributedMaskedReq {
    pub fn new(
        stage0: OvRuntime,
        downstream: Arc<tokio::sync::Mutex<ActivationClient>>,
        runtime_handle: tokio::runtime::Handle,
    ) -> Result<Self, EngineError> {
        let inputs = v5_inputs(&stage0)?;
        for k in ["input_ids", "attention_mask", "position_ids", "beam_idx"] {
            if !inputs.contains_key(k) {
                return Err(EngineError::Backend(format!(
                    "stage 0 IR is missing v5 input: {k}"
                )));
            }
        }
        Ok(Self {
            stage0,
            stage0_inputs: inputs,
            downstream,
            runtime_handle,
            valid_mask: vec![1i64; 4096],
            cache_len: 0,
            logical_pos: 0,
        })
    }

    pub fn reset(&mut self) -> Result<(), EngineError> {
        self.stage0.reset_state().map_err(map_ov_err)?;
        let downstream = self.downstream.clone();
        run_async(&self.runtime_handle, async move {
                let mut g = downstream.lock().await;
                let raw = (FrameKind::Reset as u32).to_be_bytes();
                g.send_raw(&raw).await
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        for v in self.valid_mask.iter_mut() {
            *v = 1;
        }
        self.cache_len = 0;
        self.logical_pos = 0;
        Ok(())
    }

    pub fn feed(&mut self, input_ids: &[i64]) -> Result<(Vec<f32>, [usize; 3]), EngineError> {
        let n = input_ids.len();
        let total = self.cache_len + n;
        if total > self.valid_mask.len() {
            let new_size = (total * 2).max(self.valid_mask.len() * 2);
            self.valid_mask.resize(new_size, 1);
        }
        let mut attn = vec![0i64; total];
        attn[..self.cache_len].copy_from_slice(&self.valid_mask[..self.cache_len]);
        for v in attn[self.cache_len..].iter_mut() {
            *v = 1;
        }
        let pos: Vec<i64> =
            (self.logical_pos as i64..(self.logical_pos + n) as i64).collect();

        // Run stage 0 locally.
        let in_ids = self.stage0_inputs.get("input_ids").unwrap().clone();
        let attn_name = self.stage0_inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.stage0_inputs.get("position_ids").unwrap().clone();
        let beam_name = self.stage0_inputs.get("beam_idx").unwrap().clone();
        self.stage0
            .set_input(&in_ids, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(&attn_name, ShimDType::I64, &[1, total], &i64_to_bytes(&attn))
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(&pos_name, ShimDType::I64, &[1, n], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        let beam_bytes = 0i32.to_le_bytes().to_vec();
        self.stage0
            .set_input(&beam_name, ShimDType::I32, &[1], &beam_bytes)
            .map_err(map_ov_err)?;
        self.stage0.infer().map_err(map_ov_err)?;
        let (dtype, hidden_shape, hidden_bytes) =
            self.stage0.output(0).map_err(map_ov_err)?;
        let hidden_f32 = match dtype {
            ShimDType::F32 => hidden_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
            ShimDType::F16 => f16_bytes_to_f32(&hidden_bytes),
            other => {
                return Err(EngineError::Backend(format!(
                    "unexpected stage0 dtype {other:?}"
                )))
            }
        };

        let hidden_f16 = f32_to_f16_bytes(&hidden_f32);
        let mut hidden_shape_wire = [1u32; MAX_RANK];
        for (i, d) in hidden_shape.iter().enumerate().take(MAX_RANK) {
            hidden_shape_wire[i] = *d as u32;
        }

        // Send FORWARD to downstream + recv LOGITS_RESPONSE.
        let logical_pos_start = self.logical_pos as u32;
        let downstream = self.downstream.clone();
        let attn_clone = attn.clone();
        let (logits, _logits_shape) = run_async(&self.runtime_handle, async move {
                let mut g = downstream.lock().await;
                // We need raw socket access. The ActivationClient exposes
                // send_raw/recv_raw + send_tensor/recv_tensor; build the
                // FORWARD frame inline (kind+pos via send_raw, then
                // attention_mask + hidden via send_tensor).
                let mut header = [0u8; 8];
                header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
                header[4..8].copy_from_slice(&logical_pos_start.to_be_bytes());
                g.send_raw(&header).await?;
                let attn_tensor = WireTensor::new(
                    WireDType::I64,
                    [1, 1, total as u32],
                    i64_to_bytes(&attn_clone),
                );
                g.send(&attn_tensor).await?;
                let hidden_tensor = WireTensor::new(WireDType::F16, hidden_shape_wire, hidden_f16);
                g.send(&hidden_tensor).await?;

                // Read LOGITS_RESPONSE.
                let kind_bytes = g.recv_raw(4).await?;
                let kind = u32::from_be_bytes([
                    kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3],
                ]);
                if kind != FrameKind::LogitsResponse as u32 {
                    return Err(tahoma_transport::TransportError::SocketClosed);
                }
                let (t, _) = g.recv().await?;
                let logits_f32 = match t.dtype {
                    WireDType::F32 => t
                        .data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<_>>(),
                    WireDType::F16 => f16_bytes_to_f32(&t.data),
                    _ => return Err(tahoma_transport::TransportError::SocketClosed),
                };
                Ok::<_, tahoma_transport::TransportError>((
                    logits_f32,
                    [t.shape[0] as usize, t.shape[1] as usize, t.shape[2] as usize],
                ))
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;

        self.cache_len += n;
        self.logical_pos += n;
        let s3 = if hidden_shape.len() == 3 {
            [hidden_shape[0], hidden_shape[1], hidden_shape[2]]
        } else if hidden_shape.len() == 2 {
            [1, hidden_shape[0], hidden_shape[1]]
        } else {
            [1, 1, hidden_f32.len()]
        };
        let _ = s3;
        Ok((
            logits,
            [
                _logits_shape[0],
                _logits_shape[1],
                _logits_shape[2],
            ],
        ))
    }

    pub fn rewind(&mut self, k: usize) {
        if k == 0 {
            return;
        }
        let lo = self.cache_len.saturating_sub(k);
        for i in lo..self.cache_len {
            self.valid_mask[i] = 0;
        }
        self.logical_pos = self.logical_pos.saturating_sub(k);
    }
}

// -------- spec_decode loop --------

#[derive(Debug, Default, Clone)]
pub struct SpecDecodeStats {
    pub n_steps: u32,
    pub total_drafts: u32,
    pub total_accepted: u32,
}

impl SpecDecodeStats {
    pub fn accept_rate(&self) -> f32 {
        if self.total_drafts == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_drafts as f32
        }
    }
}

/// Run greedy speculative decoding to completion, accumulating tokens.
/// Mirrors `spec_decode_greedy_stream` from the Python port. Returns the
/// list of generated token ids.
pub fn spec_decode_greedy(
    target: &mut DistributedMaskedReq,
    draft: &mut MaskedReq,
    prompt_ids: &[i64],
    max_tokens: usize,
    k: usize,
    stats: &mut SpecDecodeStats,
) -> Result<Vec<i64>, EngineError> {
    let mut out: Vec<i64> = Vec::new();
    target.reset()?;
    draft.reset()?;

    let (t_logits, t_shape) = target.feed(prompt_ids)?;
    draft.feed(prompt_ids)?;

    let vocab = t_shape[2];
    let last_row = &t_logits[t_logits.len() - vocab..];
    let first = argmax(last_row) as i64;
    out.push(first);
    if out.len() >= max_tokens {
        return Ok(out);
    }

    let mut prev_correction = first;
    let (d_logits, d_shape) = draft.feed(&[first])?;
    let dv = d_shape[2];
    let mut d_last_logit: Vec<f32> = d_logits[d_logits.len() - dv..].to_vec();

    while out.len() < max_tokens {
        stats.n_steps += 1;

        let mut drafts: Vec<i64> = vec![argmax(&d_last_logit) as i64];
        for _i in 1..k {
            if out.len() + drafts.len() >= max_tokens {
                break;
            }
            let prev = *drafts.last().unwrap();
            let (l, sh) = draft.feed(&[prev])?;
            let dv2 = sh[2];
            drafts.push(argmax(&l[l.len() - dv2..]) as i64);
        }
        let d_advanced = drafts.len() - 1;
        stats.total_drafts += drafts.len() as u32;

        // Verify [prev_correction, drafts...] in one forward
        let mut verify = Vec::with_capacity(drafts.len() + 1);
        verify.push(prev_correction);
        verify.extend_from_slice(&drafts);
        let (t_logits, t_shape) = target.feed(&verify)?;
        let v = t_shape[2];
        // Greedy per row
        let mut t_greedy = Vec::with_capacity(verify.len());
        for i in 0..verify.len() {
            let row = &t_logits[i * v..(i + 1) * v];
            t_greedy.push(argmax(row) as i64);
        }

        // Acceptance: longest matching prefix
        let mut accepted = 0;
        for i in 0..drafts.len() {
            if t_greedy[i] == drafts[i] {
                accepted += 1;
            } else {
                break;
            }
        }
        stats.total_accepted += accepted as u32;

        let correction = if accepted < drafts.len() {
            t_greedy[accepted]
        } else {
            t_greedy[drafts.len()]
        };

        for &t in drafts[..accepted].iter().chain(std::iter::once(&correction)) {
            if out.len() >= max_tokens {
                break;
            }
            out.push(t);
        }

        target.rewind(drafts.len() - accepted);

        // Draft rewind / catch-up
        if accepted < drafts.len() {
            draft.rewind(d_advanced - accepted);
            let (l, sh) = draft.feed(&[correction])?;
            let dv = sh[2];
            d_last_logit = l[l.len() - dv..].to_vec();
        } else {
            let (_, _) = draft.feed(&[*drafts.last().unwrap()])?;
            let (l, sh) = draft.feed(&[correction])?;
            let dv = sh[2];
            d_last_logit = l[l.len() - dv..].to_vec();
        }
        prev_correction = correction;
    }

    Ok(out)
}

// -------- Driver-side Engine + Builder --------

pub struct OvDistSpecEngine {
    target: DistributedMaskedReq,
    draft: MaskedReq,
    tokenizer: Arc<Tokenizer>,
    eos_token_id: Option<u32>,
    k: usize,
    pending: Vec<GenerationTask>,
    active: Option<(GenerationTask, Vec<i64>, String, SpecDecodeStats)>,
}

impl Engine for OvDistSpecEngine {
    fn warmup(&mut self) {
        // Optional: skip a real warmup; spec decode has too many moving parts.
        info!("ov-dist-spec warmup skipped (driven on first task)");
    }

    fn submit(&mut self, task: GenerationTask) {
        if self.pending.iter().any(|t| t.task_id == task.task_id)
            || self
                .active
                .as_ref()
                .is_some_and(|(t, ..)| t.task_id == task.task_id)
        {
            return;
        }
        self.pending.push(task);
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        if self.active.is_none() && !self.pending.is_empty() {
            let task = self.pending.remove(0);
            let enc = match self.tokenizer.encode(task.prompt.clone(), false) {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = %e, "tokenize failed");
                    return Vec::new();
                }
            };
            let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            info!(task = %task.task_id, prompt_tokens = prompt_ids.len(), k = self.k, "ov-dist-spec task active");
            // Run the full spec decode loop in this single step (Python did the
            // same — per-token streaming wasn't free either since drafts are
            // generated per-round, not per-token).
            let mut stats = SpecDecodeStats::default();
            let max_tokens = task.max_tokens.max(1) as usize;
            let result = spec_decode_greedy(
                &mut self.target,
                &mut self.draft,
                &prompt_ids,
                max_tokens,
                self.k,
                &mut stats,
            );
            let task_id = task.task_id.clone();
            match result {
                Err(e) => {
                    warn!(task = %task.task_id, error = %e, "ov-dist-spec failed");
                    let chunk = Chunk::final_marker(task.task_id, "");
                    return vec![(task_id, chunk)];
                }
                Ok(tokens) => {
                    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                    let text = self
                        .tokenizer
                        .decode(&ids, true)
                        .unwrap_or_default();
                    info!(
                        task = %task.task_id,
                        tokens = tokens.len(),
                        steps = stats.n_steps,
                        accept = stats.accept_rate(),
                        "ov-dist-spec done"
                    );
                    let chunk = Chunk {
                        task_id: task.task_id.clone(),
                        token_id: 0,
                        text,
                        is_final: true,
                        logprobs: None,
                    };
                    return vec![(task_id, chunk)];
                }
            }
        }
        Vec::new()
    }
}

pub struct OvDistSpecBuilder {
    pub pipeline_dir: PathBuf,
    pub draft_model_path: String,
    pub device: String,
    pub k: u32,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    stage0: Option<OvRuntime>,
    draft: Option<OvRuntime>,
    tokenizer: Option<Arc<Tokenizer>>,
    eos_token_id: Option<u32>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
}

impl OvDistSpecBuilder {
    pub fn new(
        pipeline_dir: impl Into<PathBuf>,
        draft_model_path: impl Into<String>,
        device: impl Into<String>,
        k: u32,
    ) -> Self {
        Self {
            pipeline_dir: pipeline_dir.into(),
            draft_model_path: draft_model_path.into(),
            device: device.into(),
            k,
            cache_dir: None,
            kv_cache_precision: None,
            dyn_quant_group: None,
            stage0: None,
            draft: None,
            tokenizer: None,
            eos_token_id: None,
            downstream: None,
        }
    }

    pub fn with_cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }
    pub fn with_kv_cache_precision(mut self, prec: impl Into<String>) -> Self {
        self.kv_cache_precision = Some(prec.into());
        self
    }
    pub fn with_dyn_quant_group(mut self, group: impl Into<String>) -> Self {
        self.dyn_quant_group = Some(group.into());
        self
    }

    fn plugin(&self) -> PluginConfig {
        let mut p = PluginConfig::new();
        if let Some(d) = &self.cache_dir {
            p = p.with("CACHE_DIR", d);
        }
        if let Some(p2) = &self.kv_cache_precision {
            p = p.with("KV_CACHE_PRECISION", p2);
        }
        if let Some(g) = &self.dyn_quant_group {
            p = p.with("DYNAMIC_QUANTIZATION_GROUP_SIZE", g);
        }
        p
    }
}

#[async_trait]
impl Builder for OvDistSpecBuilder {
    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        if peers.upstream.is_some() {
            return Err(EngineError::PeerRejected(
                "driver should not have an upstream".into(),
            ));
        }
        let downstream = peers
            .downstream
            .ok_or_else(|| EngineError::PeerRejected("driver requires --next host:port".into()))?;
        let mut client = ActivationClient::new(downstream.host, downstream.port);
        client
            .connect_with_timeout(Duration::from_secs(60))
            .await
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        self.downstream = Some(Arc::new(tokio::sync::Mutex::new(client)));
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        let mut events = Vec::new();
        let pipeline_cfg = read_pipeline_config(&self.pipeline_dir)?;
        let stage_dir = self.pipeline_dir.join("stage_0");
        let stage_cfg = read_stage_config(&stage_dir)?;
        if !stage_cfg.has_embed {
            return Err(EngineError::ShardRejected(
                "driver expects stage 0 with has_embed=true".into(),
            ));
        }
        if pipeline_cfg
            .export_version
            .as_deref()
            .unwrap_or_default()
            .starts_with("v3")
            || stage_cfg
                .export_version
                .as_deref()
                .unwrap_or_default()
                .starts_with("v3")
        {
            return Err(EngineError::ShardRejected(
                "driver requires v5 shards (canonical inputs)".into(),
            ));
        }
        let plugin = self.plugin();

        events.push(LoadProgress::message("compiling target stage 0".to_string()));
        let stage0_xml = stage_dir.join("openvino_model.xml");
        let stage0 = OvRuntime::compile(stage0_xml.to_str().unwrap_or_default(), &self.device, &plugin)
            .map_err(map_ov_err)?;
        self.stage0 = Some(stage0);

        events.push(LoadProgress::message("loading tokenizer".to_string()));
        let tok_path = self.pipeline_dir.join("tokenizer/tokenizer.json");
        if tok_path.exists() {
            let tok = Tokenizer::from_file(&tok_path)
                .map_err(|e| EngineError::Backend(format!("tokenizer load: {e}")))?;
            self.tokenizer = Some(Arc::new(tok));
            self.eos_token_id =
                lookup_eos(&self.pipeline_dir.join("tokenizer")).or_else(|| lookup_eos(&self.pipeline_dir));
        } else {
            return Err(EngineError::Backend(format!(
                "tokenizer.json not found at {tok_path:?}"
            )));
        }

        events.push(LoadProgress::message(format!(
            "compiling draft {}",
            self.draft_model_path
        )));
        let draft_xml = std::path::Path::new(&self.draft_model_path).join("openvino_model.xml");
        let draft = OvRuntime::compile(
            draft_xml.to_str().unwrap_or_default(),
            &self.device,
            &plugin,
        )
        .map_err(map_ov_err)?;
        self.draft = Some(draft);

        events.push(LoadProgress::ready());
        Ok(Box::pin(stream::iter(events)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let stage0 = self.stage0.ok_or(EngineError::NotLoaded)?;
        let draft = self.draft.ok_or(EngineError::NotLoaded)?;
        let tokenizer = self.tokenizer.ok_or(EngineError::NotLoaded)?;
        let downstream = self.downstream.ok_or(EngineError::NotConnected)?;
        let target = DistributedMaskedReq::new(stage0, downstream, tokio::runtime::Handle::current())?;
        let masked_draft = MaskedReq::new(draft)?;
        Ok(Box::new(OvDistSpecEngine {
            target,
            draft: masked_draft,
            tokenizer,
            eos_token_id: self.eos_token_id,
            k: self.k as usize,
            pending: Vec::new(),
            active: None,
        }))
    }
}

// -------- Worker-side Engine + Builder --------

pub struct OvDistSpecWorkerEngine {
    is_last: bool,
    runtime: OvRuntime,
    inputs: std::collections::HashMap<String, String>,
    upstream: Arc<tokio::sync::Mutex<ActivationServer>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    runtime_handle: tokio::runtime::Handle,
}

impl Engine for OvDistSpecWorkerEngine {
    fn warmup(&mut self) {
        info!("ov-dist-spec worker warmup skipped");
    }

    fn submit(&mut self, _task: GenerationTask) {
        warn!("ov-dist-spec worker cannot accept tasks directly");
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        let result = self.handle_one_frame();
        if let Err(e) = result {
            // Transport-closed errors signal the driver disconnected;
            // don't spam the log. Drop the upstream/downstream so the
            // next step exits the relay loop cleanly via NotConnected.
            let msg = e.to_string();
            if msg.contains("socket closed") || msg.contains("not connected") {
                warn!("ov-dist-spec worker: upstream closed, exiting");
                // Mark engine as drained by clearing connections.
                let _ = self
                    .runtime_handle
                    .clone()
                    .block_on(async {
                        let mut g = self.upstream.lock().await;
                        g.close().await;
                        Ok::<_, tahoma_transport::TransportError>(())
                    });
                if let Some(d) = &self.downstream {
                    let _ = self.runtime_handle.clone().block_on(async {
                        let mut g = d.lock().await;
                        g.close().await;
                        Ok::<_, tahoma_transport::TransportError>(())
                    });
                }
                // Sleep a tick so the relay loop's busy spin doesn't
                // hot-loop until the runner notices.
                std::thread::sleep(std::time::Duration::from_millis(200));
            } else {
                warn!(error = %e, "ov-dist-spec worker step error");
            }
        }
        Vec::new()
    }
}

impl OvDistSpecWorkerEngine {
    fn handle_one_frame(&mut self) -> Result<(), EngineError> {
        let upstream = self.upstream.clone();
        let downstream = self.downstream.clone();

        // 1. Read kind from upstream.
        let kind = run_async(&self.runtime_handle, async {
                let mut g = upstream.lock().await;
                g.recv_raw(4).await.map(|b| {
                    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
                })
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        let kind = FrameKind::from_u32(kind)
            .ok_or_else(|| EngineError::Backend(format!("bad kind {kind}")))?;

        match kind {
            FrameKind::Reset => {
                self.runtime.reset_state().map_err(map_ov_err)?;
                if let Some(d) = downstream {
                    run_async(&self.runtime_handle, async move {
                        let mut dg = d.lock().await;
                        dg.send_raw(&(FrameKind::Reset as u32).to_be_bytes()).await
                    })
                        .map_err(|e| EngineError::Backend(e.to_string()))?;
                }
                Ok(())
            }
            FrameKind::LogitsResponse => Err(EngineError::Backend(
                "worker received LOGITS_RESPONSE".into(),
            )),
            FrameKind::Forward => {
                // 2. Read FORWARD body (logical_pos, attn, hidden) from upstream.
                let upstream2 = upstream.clone();
                let (logical_pos_start, attn, hidden_f32, hidden_shape) = run_async(&self.runtime_handle, async move {
                        let mut g = upstream2.lock().await;
                        let pos_bytes = g.recv_raw(4).await?;
                        let logical_pos_start = u32::from_be_bytes([
                            pos_bytes[0], pos_bytes[1], pos_bytes[2], pos_bytes[3],
                        ]);
                        let (attn_t, _) = g.recv().await?;
                        let attn_i64 = bytes_to_i64(&attn_t.data);
                        let (hidden_t, _) = g.recv().await?;
                        let hs_f32 = match hidden_t.dtype {
                            WireDType::F32 => hidden_t
                                .data
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect::<Vec<_>>(),
                            WireDType::F16 => f16_bytes_to_f32(&hidden_t.data),
                            _ => return Err(tahoma_transport::TransportError::SocketClosed),
                        };
                        Ok::<_, tahoma_transport::TransportError>((
                            logical_pos_start,
                            attn_i64,
                            hs_f32,
                            [
                                hidden_t.shape[0] as usize,
                                hidden_t.shape[1] as usize,
                                hidden_t.shape[2] as usize,
                            ],
                        ))
                    })
                    .map_err(|e| EngineError::Backend(e.to_string()))?;

                // 3. Run inference on this stage.
                let new_tokens = hidden_shape[1];
                let pos: Vec<i64> = (logical_pos_start as i64
                    ..(logical_pos_start as i64 + new_tokens as i64))
                    .collect();
                let in_hs = self
                    .inputs
                    .get("hidden_states")
                    .cloned()
                    .ok_or_else(|| EngineError::Backend("missing hidden_states input".into()))?;
                let in_attn = self
                    .inputs
                    .get("attention_mask")
                    .cloned()
                    .ok_or_else(|| EngineError::Backend("missing attention_mask input".into()))?;
                let in_pos = self
                    .inputs
                    .get("position_ids")
                    .cloned()
                    .ok_or_else(|| EngineError::Backend("missing position_ids input".into()))?;
                let in_beam = self
                    .inputs
                    .get("beam_idx")
                    .cloned()
                    .ok_or_else(|| EngineError::Backend("missing beam_idx input".into()))?;

                // The v5 shard's hidden_states port is f16 (not f32). The
                // Python engine does .astype(np.float32) and pip-OV
                // silently down-casts; the Rust GPU plugin does NOT
                // auto-cast and rejects with "tensor f16 vs port f32".
                // Pass the raw f16 bytes (wire format) directly.
                let hs_bytes_f16 = f32_to_f16_bytes(&hidden_f32);
                self.runtime
                    .set_input(&in_hs, ShimDType::F16, &hidden_shape, &hs_bytes_f16)
                    .map_err(map_ov_err)?;
                self.runtime
                    .set_input(
                        &in_attn,
                        ShimDType::I64,
                        &[1, attn.len()],
                        &i64_to_bytes(&attn),
                    )
                    .map_err(map_ov_err)?;
                self.runtime
                    .set_input(&in_pos, ShimDType::I64, &[1, new_tokens], &i64_to_bytes(&pos))
                    .map_err(map_ov_err)?;
                self.runtime
                    .set_input(&in_beam, ShimDType::I32, &[1], &0i32.to_le_bytes())
                    .map_err(map_ov_err)?;
                self.runtime.infer().map_err(map_ov_err)?;
                let (out_dtype, out_shape, out_bytes) =
                    self.runtime.output(0).map_err(map_ov_err)?;
                let out_f16_bytes = match out_dtype {
                    ShimDType::F16 => out_bytes,
                    ShimDType::F32 => {
                        // Convert to f16 for wire transport
                        let f: Vec<f32> = out_bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        f32_to_f16_bytes(&f)
                    }
                    other => {
                        return Err(EngineError::Backend(format!(
                            "unexpected output dtype {other:?}"
                        )))
                    }
                };
                // Avoid unused-var warning when not used below.
                let _ = hidden_f32;
                let mut out_shape_wire = [1u32; MAX_RANK];
                for (i, d) in out_shape.iter().enumerate().take(MAX_RANK) {
                    out_shape_wire[i] = *d as u32;
                }

                if self.is_last {
                    // Send LOGITS_RESPONSE back to upstream
                    let upstream3 = upstream.clone();
                    run_async(&self.runtime_handle, async move {
                        let mut g = upstream3.lock().await;
                        g.send_raw(&(FrameKind::LogitsResponse as u32).to_be_bytes())
                            .await?;
                        let t = WireTensor::new(WireDType::F16, out_shape_wire, out_f16_bytes);
                        g.send(&t).await
                    })
                    .map_err(|e| EngineError::Backend(e.to_string()))?;
                } else {
                    // Forward downstream, then relay LOGITS_RESPONSE upstream
                    let downstream =
                        downstream.ok_or_else(|| EngineError::Backend("no downstream".into()))?;
                    let upstream3 = upstream.clone();
                    run_async(&self.runtime_handle, async move {
                            // Send FORWARD downstream
                            let attn_t = WireTensor::new(
                                WireDType::I64,
                                [1, 1, attn.len() as u32],
                                i64_to_bytes(&attn),
                            );
                            let mut header = [0u8; 8];
                            header[0..4]
                                .copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
                            header[4..8].copy_from_slice(&logical_pos_start.to_be_bytes());
                            {
                                let mut dg = downstream.lock().await;
                                dg.send_raw(&header).await?;
                                dg.send(&attn_t).await?;
                                let t = WireTensor::new(
                                    WireDType::F16,
                                    out_shape_wire,
                                    out_f16_bytes.clone(),
                                );
                                dg.send(&t).await?;
                                let kind_bytes = dg.recv_raw(4).await?;
                                let kv = u32::from_be_bytes([
                                    kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3],
                                ]);
                                if kv != FrameKind::LogitsResponse as u32 {
                                    return Err(tahoma_transport::TransportError::SocketClosed);
                                }
                                let (logits_t, _) = dg.recv().await?;
                                let mut g = upstream3.lock().await;
                                g.send_raw(&(FrameKind::LogitsResponse as u32).to_be_bytes())
                                    .await?;
                                g.send(&logits_t).await?;
                            }
                            Ok::<_, tahoma_transport::TransportError>(())
                        })
                        .map_err(|e| EngineError::Backend(e.to_string()))?;
                }
                Ok(())
            }
        }
    }
}

pub struct OvDistSpecWorkerBuilder {
    pub pipeline_dir: PathBuf,
    pub rank: u32,
    pub total: u32,
    pub device: String,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    runtime: Option<OvRuntime>,
    inputs: std::collections::HashMap<String, String>,
    upstream: Option<Arc<tokio::sync::Mutex<ActivationServer>>>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    listen_host: String,
    listen_port: Option<u16>,
}

impl OvDistSpecWorkerBuilder {
    pub fn new(
        pipeline_dir: impl Into<PathBuf>,
        rank: u32,
        total: u32,
        device: impl Into<String>,
    ) -> Self {
        Self {
            pipeline_dir: pipeline_dir.into(),
            rank,
            total,
            device: device.into(),
            cache_dir: None,
            kv_cache_precision: None,
            dyn_quant_group: None,
            runtime: None,
            inputs: std::collections::HashMap::new(),
            upstream: None,
            downstream: None,
            listen_host: "0.0.0.0".into(),
            listen_port: None,
        }
    }

    pub fn with_cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }
    pub fn with_kv_cache_precision(mut self, prec: impl Into<String>) -> Self {
        self.kv_cache_precision = Some(prec.into());
        self
    }
    pub fn with_dyn_quant_group(mut self, group: impl Into<String>) -> Self {
        self.dyn_quant_group = Some(group.into());
        self
    }

    fn plugin(&self) -> PluginConfig {
        let mut p = PluginConfig::new();
        if let Some(d) = &self.cache_dir {
            p = p.with("CACHE_DIR", d);
        }
        if let Some(p2) = &self.kv_cache_precision {
            p = p.with("KV_CACHE_PRECISION", p2);
        }
        if let Some(g) = &self.dyn_quant_group {
            p = p.with("DYNAMIC_QUANTIZATION_GROUP_SIZE", g);
        }
        p
    }
}

#[async_trait]
impl Builder for OvDistSpecWorkerBuilder {
    fn configure_listen(&mut self, host: &str, port: u16) {
        self.listen_host = host.to_string();
        self.listen_port = Some(port);
    }

    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()> {
        let _ = peers.upstream.ok_or_else(|| {
            EngineError::PeerRejected("worker requires upstream".into())
        })?;
        let port = self
            .listen_port
            .ok_or_else(|| EngineError::PeerRejected("configure_listen() required".into()))?;
        let mut server = ActivationServer::new(self.listen_host.clone(), port);
        server
            .start()
            .await
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        self.upstream = Some(Arc::new(tokio::sync::Mutex::new(server)));
        if let Some(downstream) = peers.downstream {
            let mut client = ActivationClient::new(downstream.host, downstream.port);
            client
                .connect_with_timeout(Duration::from_secs(60))
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
            self.downstream = Some(Arc::new(tokio::sync::Mutex::new(client)));
        }
        if let Some(srv) = &self.upstream {
            let srv = srv.clone();
            srv.lock()
                .await
                .accept()
                .await
                .map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        let mut events = Vec::new();
        let stage_dir = {
            let p = self.pipeline_dir.join(format!("stage_{}", self.rank));
            if p.is_dir() {
                p
            } else {
                self.pipeline_dir.clone()
            }
        };
        let stage_cfg = read_stage_config(&stage_dir)?;
        let is_last = self.rank == self.total - 1;
        if is_last && !stage_cfg.has_head {
            return Err(EngineError::ShardRejected(format!(
                "last stage (rank {}) expected has_head=true",
                self.rank
            )));
        }
        if stage_cfg
            .export_version
            .as_deref()
            .unwrap_or_default()
            .starts_with("v3")
        {
            return Err(EngineError::ShardRejected(
                "worker requires v5 shards".into(),
            ));
        }
        let plugin = self.plugin();
        events.push(LoadProgress::message(format!(
            "compiling stage {}",
            self.rank
        )));
        let xml_path = stage_dir.join("openvino_model.xml");
        let runtime = OvRuntime::compile(
            xml_path.to_str().unwrap_or_default(),
            &self.device,
            &plugin,
        )
        .map_err(map_ov_err)?;
        self.inputs = v5_inputs(&runtime)?;
        for k in ["hidden_states", "attention_mask", "position_ids", "beam_idx"] {
            if !self.inputs.contains_key(k) {
                return Err(EngineError::Backend(format!(
                    "stage {} IR is missing v5 input: {k}",
                    self.rank
                )));
            }
        }
        self.runtime = Some(runtime);
        events.push(LoadProgress::ready());
        Ok(Box::pin(stream::iter(events)))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        let runtime = self.runtime.ok_or(EngineError::NotLoaded)?;
        let upstream = self.upstream.ok_or(EngineError::NotConnected)?;
        let is_last = self.rank == self.total - 1;
        Ok(Box::new(OvDistSpecWorkerEngine {
            is_last,
            runtime,
            inputs: self.inputs,
            upstream,
            downstream: self.downstream,
            runtime_handle: tokio::runtime::Handle::current(),
        }))
    }
}

// -------- shared eos lookup (re-exported across crate via runtime.rs) --------

#[derive(Debug, Deserialize, Default)]
struct GenerationCfg {
    #[serde(default)]
    eos_token_id: Option<EosId>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosId {
    One(u32),
    Many(Vec<u32>),
}

fn lookup_eos(model_dir: &std::path::Path) -> Option<u32> {
    for fname in ["generation_config.json", "config.json"] {
        let p = model_dir.join(fname);
        if let Ok(bytes) = std::fs::read(&p) {
            if let Ok(g) = serde_json::from_slice::<GenerationCfg>(&bytes) {
                return match g.eos_token_id {
                    Some(EosId::One(id)) => Some(id),
                    Some(EosId::Many(ids)) => ids.first().copied(),
                    None => None,
                };
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tahoma_transport::{ActivationClient, ActivationServer};

    #[test]
    fn frame_kind_roundtrip() {
        for k in [FrameKind::Forward, FrameKind::Reset, FrameKind::LogitsResponse] {
            let bytes = (k as u32).to_be_bytes();
            let v = u32::from_be_bytes(bytes);
            assert_eq!(FrameKind::from_u32(v), Some(k));
        }
    }

    #[test]
    fn frame_kind_unknown_returns_none() {
        assert_eq!(FrameKind::from_u32(99), None);
        assert_eq!(FrameKind::from_u32(0), None);
        assert_eq!(FrameKind::from_u32(2), None); // gap between 1 and 3
    }

    #[test]
    fn argmax_basic() {
        let v = vec![0.1, 0.5, 0.2, 0.05];
        assert_eq!(argmax(&v), 1);
    }

    #[test]
    fn argmax_picks_first_on_ties() {
        let v = vec![0.5, 0.5, 0.5];
        assert_eq!(argmax(&v), 0);
    }

    #[test]
    fn dtype_conversions_roundtrip() {
        let f = vec![1.0f32, -2.5, 3.14, 0.0];
        let bytes16 = f32_to_f16_bytes(&f);
        assert_eq!(bytes16.len(), f.len() * 2);
        let back = f16_bytes_to_f32(&bytes16);
        for (a, b) in f.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "lost precision: {a} vs {b}");
        }
    }

    #[test]
    fn i64_bytes_roundtrip() {
        let xs = vec![1i64, -2, 3, 4_000_000_000];
        let bytes = i64_to_bytes(&xs);
        assert_eq!(bytes.len(), xs.len() * 8);
        assert_eq!(bytes_to_i64(&bytes), xs);
    }

    /// FORWARD frame round-trip via real TCP + ActivationClient/Server,
    /// mirrors `tests/test_dist_spec_protocol.py::test_forward_roundtrip`.
    #[tokio::test]
    async fn forward_roundtrip() {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            let kb = server.recv_raw(4).await.unwrap();
            let kind = FrameKind::from_u32(u32::from_be_bytes([
                kb[0], kb[1], kb[2], kb[3],
            ]))
            .unwrap();
            let pos_b = server.recv_raw(4).await.unwrap();
            let pos = u32::from_be_bytes([pos_b[0], pos_b[1], pos_b[2], pos_b[3]]);
            let (attn, _) = server.recv().await.unwrap();
            let (hidden, _) = server.recv().await.unwrap();
            (kind, pos, attn.data, hidden.data)
        });

        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();

        // Send FORWARD frame inline (mirroring DistributedMaskedReq.feed)
        let mut header = [0u8; 8];
        header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
        header[4..8].copy_from_slice(&42u32.to_be_bytes());
        client.send_raw(&header).await.unwrap();
        let attn = vec![1i64, 1, 1, 0, 1];
        let attn_t = WireTensor::new(WireDType::I64, [1, 1, 5], i64_to_bytes(&attn));
        client.send(&attn_t).await.unwrap();
        let hidden_data = (0..16u8).collect::<Vec<u8>>();
        let hidden_t = WireTensor::new(WireDType::F16, [1, 2, 4], hidden_data.clone());
        client.send(&hidden_t).await.unwrap();

        let (kind, pos, attn_rx, hidden_rx) = h.await.unwrap();
        assert_eq!(kind, FrameKind::Forward);
        assert_eq!(pos, 42);
        assert_eq!(attn_rx, i64_to_bytes(&attn));
        assert_eq!(hidden_rx, hidden_data);
    }

    #[tokio::test]
    async fn reset_kind_only() {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            let kb = server.recv_raw(4).await.unwrap();
            FrameKind::from_u32(u32::from_be_bytes([kb[0], kb[1], kb[2], kb[3]])).unwrap()
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        let raw = (FrameKind::Reset as u32).to_be_bytes();
        client.send_raw(&raw).await.unwrap();
        assert_eq!(h.await.unwrap(), FrameKind::Reset);
    }

    #[tokio::test]
    async fn logits_response_roundtrip() {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let h = tokio::spawn(async move {
            server.accept().await.unwrap();
            let kb = server.recv_raw(4).await.unwrap();
            let kind = FrameKind::from_u32(u32::from_be_bytes([
                kb[0], kb[1], kb[2], kb[3],
            ]))
            .unwrap();
            let (logits, _) = server.recv().await.unwrap();
            (kind, logits.data)
        });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        client
            .send_raw(&(FrameKind::LogitsResponse as u32).to_be_bytes())
            .await
            .unwrap();
        let logits = (0..200u8).collect::<Vec<u8>>();
        let t = WireTensor::new(WireDType::F16, [1, 2, 50], logits.clone());
        client.send(&t).await.unwrap();
        let (kind, rx) = h.await.unwrap();
        assert_eq!(kind, FrameKind::LogitsResponse);
        assert_eq!(rx, logits);
    }
}
