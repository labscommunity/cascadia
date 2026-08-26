//! Distributed speculative decoding (v5 shards, mask-based KV-cache rewind).
//!
//! Rust port of `cascadia/worker/engines/openvino/dist_spec.py` +
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
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_ov_genai_shim::{
    advance_emitted, DType as ShimDType, Error as OvError, PluginConfig, Runtime as OvRuntime,
};
use cascadia_transport::{
    recv_tensor, send_tensor, ActivationClient, ActivationServer, DType as WireDType,
    Tensor as WireTensor, MAX_RANK, MAX_RAW_BYTES,
};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::stream;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tokio::net::TcpStream;
#[cfg(feature = "kv_coord")]
use tracing::error;
use tracing::{info, warn};

// -------- frame protocol --------

/// Ceiling on a carried RESTORE blob; guards against a desynced frame stream misreading the length
/// as garbage and triggering an unbounded read. One rank's whole-state KV is tens of MB.
#[cfg(feature = "kv_coord")]
const MAX_CARRY_BLOB_BYTES: usize = 256 * 1024 * 1024;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Forward = 1,
    Reset = 3,
    LogitsResponse = 4,
    // Issue-34 §8 multi-stage KV coordination (append-only; `kv_coord`-gated). CAPTURE snapshots a
    // rank's KV under the head epoch; RESTORE set_states a pulled slice (ACK body carries the
    // all-or-nothing verdict). The existing RESET is the cold fallback.
    #[cfg(feature = "kv_coord")]
    Capture = 5,
    #[cfg(feature = "kv_coord")]
    CaptureAck = 6,
    #[cfg(feature = "kv_coord")]
    Restore = 7,
    #[cfg(feature = "kv_coord")]
    RestoreAck = 8,
    /// H.1b (R2): CAPTURE whose body also carries the turn's TENANT
    /// (`capture_body_bytes_masked_v2`). A separate kind, not a wider `Capture` body: `from_u32`
    /// hard-errors on an unknown kind and the masked codec enforces an exact length, so widening
    /// in place would break any chain whose ranks run different builds. dist-spec's CAPTURE always
    /// carries a valid_mask, and mask and tenant are byte-indistinguishable suffixes — the kind is
    /// the ONLY thing that tells the two bodies apart.
    #[cfg(feature = "kv_coord")]
    CaptureV2 = 9,
}

impl FrameKind {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Forward),
            3 => Some(Self::Reset),
            4 => Some(Self::LogitsResponse),
            #[cfg(feature = "kv_coord")]
            5 => Some(Self::Capture),
            #[cfg(feature = "kv_coord")]
            6 => Some(Self::CaptureAck),
            #[cfg(feature = "kv_coord")]
            7 => Some(Self::Restore),
            #[cfg(feature = "kv_coord")]
            8 => Some(Self::RestoreAck),
            #[cfg(feature = "kv_coord")]
            9 => Some(Self::CaptureV2),
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
    // NaN propagates: `*v > best_v` is false for any NaN, so an all-NaN
    // logits row would silently return index 0 (often EOS-adjacent or
    // <bos> in Llama tokenizers — easy to miss). Track whether we ever
    // saw a finite value; if not, log and return 0 explicitly so the
    // failure is observable.
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut saw_finite = false;
    for (i, v) in slice.iter().enumerate() {
        if v.is_finite() {
            saw_finite = true;
            if *v > best_v {
                best_v = *v;
                best_i = i;
            }
        }
    }
    if !saw_finite {
        warn!(
            "argmax: all logits non-finite (NaN/Inf); returning token 0 — \
             likely indicates a numerically broken forward pass"
        );
    }
    best_i
}

fn map_ov_err(err: OvError) -> EngineError {
    match err {
        OvError::Stub => {
            EngineError::Backend("openvino shim built without --features openvino".into())
        }
        OvError::Utf8(s) => EngineError::InvalidConfig(s),
        OvError::Native(s) => EngineError::Backend(s),
    }
}

/// Maximum tasks an engine will queue. The HTTP layer applies its own
/// concurrency cap upstream; this is a defense-in-depth bound so a
/// caller that bypasses the HTTP layer (e.g. tests, scripts) can't
/// drive `pending` to OOM with cheap submit() calls. 256 is far
/// above the steady-state queue any single-engine deployment will
/// see.
pub const MAX_PENDING_TASKS: usize = 256;

/// Bridge sync code to an async future.
///
/// Two contexts hit this path:
///
/// * Driver via `ChunkStream::poll_next` — running inside a tokio
///   worker thread polling an async task. Naked `block_on` panics
///   with "Cannot start a runtime from within a runtime"; need
///   `block_in_place` to migrate other tasks off this worker first.
///
/// * Worker via `Runner::run_relay_loop`, dispatched through
///   `tokio::task::spawn_blocking`. Spawn_blocking threads are NOT
///   polling tasks; `Handle::block_on` works directly. Wrapping with
///   `block_in_place` is unnecessary AND expensive on Windows
///   (empirically ~20 ms per call vs ~5–30 µs the docs would suggest;
///   adds ~60 ms per worker frame round-trip with 3 wire I/O calls).
///
/// We dispatch via a thread-local flag managed by [`BlockingContextGuard`].
/// Workers acquire the guard once per `step()` (idempotent within a
/// single thread); the guard is RAII-scoped so the flag never leaks
/// across spawn_blocking thread reuse boundaries.
/// Bridge sync engine code to an async transport future. Thin wrapper
/// around `cascadia_runner::run_async` — the dispatch logic + thread-local
/// guard now live in the runner crate so both engines (ov-dist-spec and
/// sparse-moe) share the same fast path on spawn_blocking threads.
pub(crate) fn run_async_pub<F: std::future::Future>(
    handle: &tokio::runtime::Handle,
    fut: F,
) -> F::Output {
    cascadia_runner::run_async(handle, fut)
}

/// Private alias used by call sites within this file — keeps the
/// existing `run_async(&handle, async move { ... })` ergonomics without
/// every site importing `cascadia_runner::run_async` directly.
fn run_async<F: std::future::Future>(handle: &tokio::runtime::Handle, fut: F) -> F::Output {
    cascadia_runner::run_async(handle, fut)
}

// `BlockingContextGuard` / `BLOCKING_CONTEXT` moved to `cascadia_runner`.
// `Runner::run_relay_loop` enters the guard once per OS thread, so
// engines don't need to acquire it inside `step()` anymore.
pub(crate) use cascadia_runner::BlockingContextGuard;

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
    // One shared resolver across the OV engines (mirrors the Python
    // _v5_inputs()); see runtime::resolve_canonical_inputs.
    crate::runtime::resolve_canonical_inputs(runtime)
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

fn io_err(e: cascadia_transport::TransportError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}

// -------- MaskedReq (local draft / target wrapper with mask-based rewind) --------

/// Bound on every §8 CAPTURE/RESTORE ack wait — a reply to in-flight work must use a strict
/// deadline, never the transport's ~900 s frame-idle ceiling (capture runs inside
/// `finalize_active` holding the downstream lock; restore runs at admission). Matches
/// ov-runtime's `RESTORE_ACK_TIMEOUT`; every sibling engine bounds these waits.
#[cfg(feature = "kv_coord")]
const KV_ACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Cap on a §8 CAPTURE body (recv-side allocation bound). Worst case at `MAX_CAPTURE_TOKENS`
/// (1 Mi): 4 B/token + 8 B/mask-entry + headers ≈ 12 MiB; 16 MiB leaves headroom without
/// letting a corrupt length reserve unbounded memory.
#[cfg(feature = "kv_coord")]
const MAX_CAPTURE_BODY_BYTES: usize = 16 << 20;

pub struct MaskedReq {
    runtime: OvRuntime,
    has_beam: bool,
    valid_mask: Vec<i64>,
    cache_len: usize,
    logical_pos: usize,
    inputs: std::collections::HashMap<String, String>,
}

#[cfg(feature = "kv_coord")]
impl MaskedReq {
    pub(crate) fn kv_blob(&mut self) -> Option<Vec<u8>> {
        self.runtime.get_state_blob().ok()
    }
    pub(crate) fn kv_restore(&mut self, b: &[u8]) -> bool {
        self.runtime.set_state_blob(b).is_ok()
    }
    /// Scrub after a `set_state_blob` this turn is abandoning — failed (the request may be
    /// PARTIALLY restored; entries apply as the blob parses) or applied-then-discarded (chain
    /// NAK). `reset_state` cannot clear post-`set_state` residue (see the shim's
    /// `recreate_request` doc), so rebuild the request, then `reset` the host cursors — the
    /// cold reprefill must not run over half the donor's KV.
    pub(crate) fn kv_scrub(&mut self) {
        if let Err(e) = self.runtime.recreate_request() {
            error!(error = %e, "ov-dist-spec: draft recreate_request scrub failed; KV state may be dirty");
        }
        let _ = self.reset();
    }
    /// After a warm `set_state_blob`, align the host-side cursors to the restored prefix length so
    /// the next `feed` appends the suffix at the right position (mirrors `reset`, but to `len`).
    pub(crate) fn kv_set_pos(&mut self, len: usize) {
        if len + 1 > self.valid_mask.len() {
            self.valid_mask.resize((len + 1) * 2, 1);
        }
        for m in self.valid_mask.iter_mut() {
            *m = 1;
        }
        self.cache_len = len;
        self.logical_pos = len;
    }
    /// Host valid_mask over the live KV ([0..cache_len)): 1=accepted, 0=rejected draft. Feeds
    /// `kv_compact_blob` at capture so the stored blob drops speculative positions.
    pub(crate) fn valid_prefix(&self) -> &[i64] {
        &self.valid_mask[..self.cache_len.min(self.valid_mask.len())]
    }
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

        let _ts_setup = std::time::Instant::now();
        self.runtime
            .set_input(
                &in_ids_name,
                ShimDType::I64,
                &[1, n],
                &i64_to_bytes(input_ids),
            )
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(
                &attn_name,
                ShimDType::I64,
                &[1, total],
                &i64_to_bytes(&attn),
            )
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
        let setup_us = _ts_setup.elapsed().as_micros();
        let _ts_infer = std::time::Instant::now();
        self.runtime.infer().map_err(map_ov_err)?;
        let infer_us = _ts_infer.elapsed().as_micros();
        let _ts_out = std::time::Instant::now();
        let (dtype, shape, bytes) = self.runtime.output(0).map_err(map_ov_err)?;
        let output_us = _ts_out.elapsed().as_micros();
        let _ts_conv = std::time::Instant::now();
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
        let convert_us = _ts_conv.elapsed().as_micros();
        tracing::debug!(
            n,
            setup_us,
            infer_us,
            output_us,
            convert_us,
            vocab = floats.len(),
            "draft.feed timing"
        );
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

/// Handle to a target.feed network round-trip in flight. Created by
/// `feed_send_async`, awaited by `feed_recv_async`.
///
/// **Drop semantics**: dropping a `TargetSendHandle` without calling
/// `feed_recv_async` does NOT cancel the spawned network task — tokio's
/// `JoinHandle::drop` only discards the result, the task itself continues
/// to completion. The bytes are still sent and received over the wire,
/// and the next operation on the same `DistributedMaskedReq` will queue
/// behind the orphan via the `downstream` mutex. To actually cancel,
/// call `JoinHandle::abort` (not currently exposed). In practice
/// `spec_decode_greedy` always awaits the handle, so this is a
/// theoretical concern for unusual callers.
pub struct TargetSendHandle {
    join:
        tokio::task::JoinHandle<Result<(Vec<f32>, [usize; 3]), cascadia_transport::TransportError>>,
}

pub struct DistributedMaskedReq {
    stage0: OvRuntime,
    stage0_inputs: std::collections::HashMap<String, String>,
    downstream: Arc<tokio::sync::Mutex<ActivationClient>>,
    runtime_handle: tokio::runtime::Handle,
    valid_mask: Vec<i64>,
    cache_len: usize,
    logical_pos: usize,
    // Per-task accumulators — `spec_decode_greedy` resets these at task
    // start so the final `spec_decode timing` line attributes target
    // time correctly between rank-0-side compute and wire (downstream+net).
    pub t_alpha_setup: std::time::Duration,
    pub t_alpha_infer: std::time::Duration,
    pub t_alpha_output: std::time::Duration,
    pub t_wire: std::time::Duration,
}

#[cfg(feature = "kv_coord")]
impl DistributedMaskedReq {
    pub(crate) fn stage0_blob(&mut self) -> Option<Vec<u8>> {
        self.stage0.get_state_blob().ok()
    }
    pub(crate) fn stage0_restore(&mut self, b: &[u8]) -> bool {
        self.stage0.set_state_blob(b).is_ok()
    }
    /// Scrub after an abandoned `set_state_blob` (see `MaskedReq::kv_scrub`): rebuild the
    /// stage-0 request — `reset_state` cannot clear post-`set_state` residue — then `reset`,
    /// which also re-broadcasts Reset down the chain.
    pub(crate) fn kv_scrub(&mut self) {
        if let Err(e) = self.stage0.recreate_request() {
            error!(error = %e, "ov-dist-spec: target recreate_request scrub failed; KV state may be dirty");
        }
        let _ = self.reset();
    }
    /// Align host-side cursors to a restored prefix (see `MaskedReq::kv_set_pos`).
    pub(crate) fn kv_set_pos(&mut self, len: usize) {
        if len + 1 > self.valid_mask.len() {
            self.valid_mask.resize((len + 1) * 2, 1);
        }
        for m in self.valid_mask.iter_mut() {
            *m = 1;
        }
        self.cache_len = len;
        self.logical_pos = len;
    }

    /// Driver valid_mask over the target chain's live KV ([0..cache_len)). Every stage shares the
    /// driver's per-FORWARD attn, so this one mask compacts stage0 locally AND every worker's blob.
    pub(crate) fn valid_prefix(&self) -> &[i64] {
        &self.valid_mask[..self.cache_len.min(self.valid_mask.len())]
    }

    /// §8 CAPTURE: broadcast `(epoch, tokens, valid_mask)` down the target chain; await the aggregate
    /// ACK. The mask lets each worker compact its own padded KV blob (workers hold no mask of their own).
    ///
    /// (Timeout const lives at module level: [`KV_ACK_TIMEOUT`].)
    /// A non-empty `tenant` upgrades the frame to `CaptureV2` so each worker — which never sees the
    /// `GenerationTask` — can tag its own capture with the same namespace.
    pub(crate) fn capture_downstream(
        &mut self,
        epoch: u64,
        tokens: &[i32],
        tenant: &str,
    ) -> Result<(), EngineError> {
        let (kind, body) = if tenant.is_empty() {
            (
                FrameKind::Capture,
                crate::kv_coordination::capture_body_bytes_masked(
                    epoch,
                    tokens,
                    self.valid_prefix(),
                ),
            )
        } else {
            (
                FrameKind::CaptureV2,
                crate::kv_coordination::capture_body_bytes_masked_v2(
                    epoch,
                    tokens,
                    self.valid_prefix(),
                    tenant,
                ),
            )
        };
        let downstream = self.downstream.clone();
        run_async(&self.runtime_handle, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&(kind as u32).to_be_bytes()).await?;
            g.send_raw(&(body.len() as u32).to_be_bytes()).await?;
            g.send_raw(&body).await?;
            // Bounded: a reply to in-flight work must not ride the ~900 s frame-idle ceiling —
            // this runs inside finalize_active holding the downstream lock, so an unbounded wait
            // withholds a finished turn behind a silent peer (every sibling engine bounds this).
            // Close on elapse: the late ack has no sequence number and would pair with the next
            // exchange on a reused connection.
            match tokio::time::timeout(KV_ACK_TIMEOUT, g.recv_raw(4)).await {
                Ok(kb) => {
                    let kb = kb?;
                    if u32::from_be_bytes([kb[0], kb[1], kb[2], kb[3]])
                        != FrameKind::CaptureAck as u32
                    {
                        return Err(cascadia_transport::TransportError::SocketClosed);
                    }
                    Ok(())
                }
                Err(_) => {
                    g.close().await;
                    Err(cascadia_transport::TransportError::Io(
                        std::io::Error::other(
                            "capture ack timed out; connection dropped to avoid pairing the late \
                         ack with the next exchange",
                        ),
                    ))
                }
            }
        })
        .map_err(|e| EngineError::Backend(e.to_string()))
    }

    /// §8 RESTORE: broadcast `epoch` (+ the downstream rank's carried blob, if any) down the target
    /// chain; return the all-or-nothing verdict. `carried` is the head-pulled slice for the next rank on
    /// a cross-chain move (empty ⇒ same-chain: the rank restores from its own CAPTURE). Wire: [epoch:8]
    /// [carried_len:8 LE][carried].
    pub(crate) fn restore_downstream(
        &mut self,
        epoch: u64,
        carried: Vec<u8>,
    ) -> Result<bool, EngineError> {
        let downstream = self.downstream.clone();
        run_async(&self.runtime_handle, async move {
            let mut g = downstream.lock().await;
            g.send_raw(&(FrameKind::Restore as u32).to_be_bytes())
                .await?;
            g.send_raw(&epoch.to_le_bytes()).await?;
            g.send_raw(&(carried.len() as u64).to_le_bytes()).await?;
            if !carried.is_empty() {
                g.send_raw(&carried).await?;
            }
            // Bounded + close-on-elapse, same rationale as capture_downstream above — RESTORE
            // runs at admission, so an unbounded wait stalls every warm-hit request.
            match tokio::time::timeout(KV_ACK_TIMEOUT, async {
                let kb = g.recv_raw(4).await?;
                if u32::from_be_bytes([kb[0], kb[1], kb[2], kb[3]]) != FrameKind::RestoreAck as u32
                {
                    return Err(cascadia_transport::TransportError::SocketClosed);
                }
                let vb = g.recv_raw(1).await?;
                Ok(vb.first() == Some(&1))
            })
            .await
            {
                Ok(r) => r,
                Err(_) => {
                    g.close().await;
                    Err(cascadia_transport::TransportError::Io(
                        std::io::Error::other(
                            "restore ack timed out; connection dropped to avoid pairing the late \
                         ack with the next exchange",
                        ),
                    ))
                }
            }
        })
        .map_err(|e| EngineError::Backend(e.to_string()))
    }
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
            t_alpha_setup: std::time::Duration::ZERO,
            t_alpha_infer: std::time::Duration::ZERO,
            t_alpha_output: std::time::Duration::ZERO,
            t_wire: std::time::Duration::ZERO,
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

    /// Async-split: does sync rank-0-side stage_0 work then spawns the network
    /// round-trip as a tokio task. Returns a handle the caller awaits via
    /// `feed_recv_async`. Lets the caller do other rank-0 work (next-round
    /// drafts, post-round draft.feed) DURING the downstream wait window.
    /// State (cache_len, logical_pos) is updated on send so back-to-back
    /// `feed_send_async` calls stay consistent.
    pub fn feed_send_async(&mut self, input_ids: &[i64]) -> Result<TargetSendHandle, EngineError> {
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

        // Run stage 0 locally (synchronous GPU compute).
        let in_ids = self.stage0_inputs.get("input_ids").unwrap().clone();
        let attn_name = self.stage0_inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.stage0_inputs.get("position_ids").unwrap().clone();
        let beam_name = self.stage0_inputs.get("beam_idx").unwrap().clone();
        let _ts = std::time::Instant::now();
        self.stage0
            .set_input(&in_ids, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(
                &attn_name,
                ShimDType::I64,
                &[1, total],
                &i64_to_bytes(&attn),
            )
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(&pos_name, ShimDType::I64, &[1, n], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        let beam_bytes = 0i32.to_le_bytes().to_vec();
        self.stage0
            .set_input(&beam_name, ShimDType::I32, &[1], &beam_bytes)
            .map_err(map_ov_err)?;
        let setup_us = _ts.elapsed().as_micros();
        let _ts = std::time::Instant::now();
        self.stage0.infer().map_err(map_ov_err)?;
        let infer_us = _ts.elapsed().as_micros();
        let _ts = std::time::Instant::now();
        let (dtype, hidden_shape, hidden_bytes) = self.stage0.output(0).map_err(map_ov_err)?;
        let output_us = _ts.elapsed().as_micros();
        // Pass-through f16 if possible (avoid f16→f32→f16 round-trip).
        let hidden_f16: Vec<u8> = match dtype {
            ShimDType::F32 => {
                let f = hidden_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>();
                f32_to_f16_bytes(&f)
            }
            ShimDType::F16 => hidden_bytes,
            other => {
                return Err(EngineError::Backend(format!(
                    "unexpected stage0 dtype {other:?}"
                )))
            }
        };
        let mut hidden_shape_wire = [1u32; MAX_RANK];
        for (i, d) in hidden_shape.iter().enumerate().take(MAX_RANK) {
            hidden_shape_wire[i] = *d as u32;
        }

        let logical_pos_start = self.logical_pos as u32;
        self.cache_len += n;
        self.logical_pos += n;

        self.t_alpha_setup += std::time::Duration::from_micros(setup_us as u64);
        self.t_alpha_infer += std::time::Duration::from_micros(infer_us as u64);
        self.t_alpha_output += std::time::Duration::from_micros(output_us as u64);

        let downstream = self.downstream.clone();
        let attn_clone = attn;
        let total_clone = total;
        let join = self.runtime_handle.spawn(async move {
            let mut g = downstream.lock().await;
            let mut header = [0u8; 8];
            header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
            header[4..8].copy_from_slice(&logical_pos_start.to_be_bytes());
            g.send_raw(&header).await?;
            let attn_tensor = WireTensor::new(
                WireDType::I64,
                [1, 1, total_clone as u32],
                i64_to_bytes(&attn_clone),
            );
            g.send(&attn_tensor).await?;
            let hidden_tensor = WireTensor::new(WireDType::F16, hidden_shape_wire, hidden_f16);
            g.send(&hidden_tensor).await?;
            let kind_bytes = g.recv_raw(4).await?;
            let kind =
                u32::from_be_bytes([kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3]]);
            if kind != FrameKind::LogitsResponse as u32 {
                return Err(cascadia_transport::TransportError::SocketClosed);
            }
            let (t, _) = g.recv().await?;
            let logits_f32 = match t.dtype {
                WireDType::F32 => t
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>(),
                WireDType::F16 => f16_bytes_to_f32(&t.data),
                _ => return Err(cascadia_transport::TransportError::SocketClosed),
            };
            Ok::<_, cascadia_transport::TransportError>((
                logits_f32,
                [
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                ],
            ))
        });
        Ok(TargetSendHandle { join })
    }

    /// Block on the network round-trip started by `feed_send_async`.
    pub fn feed_recv_async(
        &mut self,
        handle: TargetSendHandle,
    ) -> Result<(Vec<f32>, [usize; 3]), EngineError> {
        let _ts = std::time::Instant::now();
        let result = run_async(&self.runtime_handle, async move {
            handle
                .join
                .await
                .map_err(|e| cascadia_transport::TransportError::Io(std::io::Error::other(e)))?
        });
        let wire_us = _ts.elapsed().as_micros();
        self.t_wire += std::time::Duration::from_micros(wire_us as u64);
        result.map_err(|e| EngineError::Backend(e.to_string()))
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

        // Run stage 0 locally.
        let in_ids = self.stage0_inputs.get("input_ids").unwrap().clone();
        let attn_name = self.stage0_inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.stage0_inputs.get("position_ids").unwrap().clone();
        let beam_name = self.stage0_inputs.get("beam_idx").unwrap().clone();
        let _ts = std::time::Instant::now();
        self.stage0
            .set_input(&in_ids, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(
                &attn_name,
                ShimDType::I64,
                &[1, total],
                &i64_to_bytes(&attn),
            )
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(&pos_name, ShimDType::I64, &[1, n], &i64_to_bytes(&pos))
            .map_err(map_ov_err)?;
        let beam_bytes = 0i32.to_le_bytes().to_vec();
        self.stage0
            .set_input(&beam_name, ShimDType::I32, &[1], &beam_bytes)
            .map_err(map_ov_err)?;
        let setup_us = _ts.elapsed().as_micros();
        let _ts = std::time::Instant::now();
        self.stage0.infer().map_err(map_ov_err)?;
        let infer_us = _ts.elapsed().as_micros();
        let _ts = std::time::Instant::now();
        let (dtype, hidden_shape, hidden_bytes) = self.stage0.output(0).map_err(map_ov_err)?;
        let output_us = _ts.elapsed().as_micros();
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
        let _ts = std::time::Instant::now();
        let mut t_send = std::time::Duration::ZERO;
        let mut t_recv = std::time::Duration::ZERO;
        let mut t_lock = std::time::Duration::ZERO;
        let _ts_lock = std::time::Instant::now();
        let (logits, _logits_shape, send_d, recv_d) = run_async(&self.runtime_handle, async move {
            let mut g = downstream.lock().await;
            // We need raw socket access. The ActivationClient exposes
            // send_raw/recv_raw + send_tensor/recv_tensor; build the
            // FORWARD frame inline (kind+pos via send_raw, then
            // attention_mask + hidden via send_tensor).
            let _ts_send = std::time::Instant::now();
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
            let send_dur = _ts_send.elapsed();

            // Read LOGITS_RESPONSE.
            let _ts_kind = std::time::Instant::now();
            let kind_bytes = g.recv_raw(4).await?;
            let recv_kind_dur = _ts_kind.elapsed();
            let kind =
                u32::from_be_bytes([kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3]]);
            if kind != FrameKind::LogitsResponse as u32 {
                return Err(cascadia_transport::TransportError::SocketClosed);
            }
            let _ts_payload = std::time::Instant::now();
            let (t, _) = g.recv().await?;
            let recv_payload_dur = _ts_payload.elapsed();
            let _ts_conv = std::time::Instant::now();
            let logits_f32 = match t.dtype {
                WireDType::F32 => t
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>(),
                WireDType::F16 => f16_bytes_to_f32(&t.data),
                _ => return Err(cascadia_transport::TransportError::SocketClosed),
            };
            let convert_dur = _ts_conv.elapsed();
            tracing::debug!(
                recv_kind_us = recv_kind_dur.as_micros() as u64,
                recv_payload_us = recv_payload_dur.as_micros() as u64,
                convert_us = convert_dur.as_micros() as u64,
                payload_bytes = t.data.len(),
                "driver recv breakdown"
            );
            let recv_dur = _ts_send.elapsed() - send_dur;
            Ok::<_, cascadia_transport::TransportError>((
                logits_f32,
                [
                    t.shape[0] as usize,
                    t.shape[1] as usize,
                    t.shape[2] as usize,
                ],
                send_dur,
                recv_dur,
            ))
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        t_lock = _ts_lock.elapsed();
        let wire_us = _ts.elapsed().as_micros();
        t_send = send_d;
        t_recv = recv_d;
        tracing::debug!(
            wire_us,
            send_us = t_send.as_micros() as u64,
            recv_us = t_recv.as_micros() as u64,
            lock_us = t_lock.as_micros() as u64,
            "wire breakdown"
        );
        tracing::debug!(
            n = n,
            setup_us,
            infer_us,
            output_us,
            wire_us,
            "target.feed timing"
        );

        // Per-task accumulators for the spec_decode summary line.
        self.t_alpha_setup += std::time::Duration::from_micros(setup_us as u64);
        self.t_alpha_infer += std::time::Duration::from_micros(infer_us as u64);
        self.t_alpha_output += std::time::Duration::from_micros(output_us as u64);
        self.t_wire += std::time::Duration::from_micros(wire_us as u64);

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
            [_logits_shape[0], _logits_shape[1], _logits_shape[2]],
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
    let mut t_target = std::time::Duration::ZERO;
    let mut t_draft = std::time::Duration::ZERO;
    let t_total = std::time::Instant::now();
    target.reset()?;
    target.t_alpha_setup = std::time::Duration::ZERO;
    target.t_alpha_infer = std::time::Duration::ZERO;
    target.t_alpha_output = std::time::Duration::ZERO;
    target.t_wire = std::time::Duration::ZERO;
    draft.reset()?;

    let _ts = std::time::Instant::now();
    let (t_logits, t_shape) = target.feed(prompt_ids)?;
    t_target += _ts.elapsed();
    let _ts = std::time::Instant::now();
    draft.feed(prompt_ids)?;
    t_draft += _ts.elapsed();

    let vocab = t_shape[2];
    let last_row = &t_logits[t_logits.len() - vocab..];
    let first = argmax(last_row) as i64;
    out.push(first);
    if out.len() >= max_tokens {
        return Ok(out);
    }

    let mut prev_correction = first;
    let _ts = std::time::Instant::now();
    let (d_logits, d_shape) = draft.feed(&[first])?;
    t_draft += _ts.elapsed();
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
            let _ts = std::time::Instant::now();
            let (l, sh) = draft.feed(&[prev])?;
            t_draft += _ts.elapsed();
            let dv2 = sh[2];
            drafts.push(argmax(&l[l.len() - dv2..]) as i64);
        }
        let d_advanced = drafts.len() - 1;
        stats.total_drafts += drafts.len() as u32;

        // Verify [prev_correction, drafts...] — ASYNC SPLIT.
        // Send target verify (rank-0 stage_0 sync, then spawn network task).
        // While the downstream rank computes stage_1 (~43ms), do the SPECULATIVE first
        // half of post-round draft work: draft.feed(drafts.last()) is needed
        // in the all-accepted case anyway. If speculation is wrong (some
        // drafts rejected), we'll rewind the draft state.
        let mut verify = Vec::with_capacity(drafts.len() + 1);
        verify.push(prev_correction);
        verify.extend_from_slice(&drafts);
        let _ts = std::time::Instant::now();
        let target_handle = target.feed_send_async(&verify)?;

        // Speculative draft.feed during target wait. Saves ~draft.feed time
        // when all K drafts are accepted (joint = p_single^K). For K=1 with
        // p=0.8 this is 80% beneficial; for K=3 with p=0.6 it's ~22%.
        // Cost when wrong: extra rewind (~free, just decrements counter).
        let speculative_draft_done = if !drafts.is_empty() {
            let _ts_d = std::time::Instant::now();
            let last_draft = *drafts.last().unwrap();
            let (l, sh) = draft.feed(&[last_draft])?;
            t_draft += _ts_d.elapsed();
            let dv = sh[2];
            // Save the would-be d_last_logit if all accepted
            Some(l[l.len() - dv..].to_vec())
        } else {
            None
        };

        // Now block on target.
        let (t_logits, t_shape) = target.feed_recv_async(target_handle)?;
        t_target += _ts.elapsed();
        let v = t_shape[2];
        let mut t_greedy = Vec::with_capacity(verify.len());
        for i in 0..verify.len() {
            let row = &t_logits[i * v..(i + 1) * v];
            t_greedy.push(argmax(row) as i64);
        }

        // Acceptance
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

        for &t in drafts[..accepted]
            .iter()
            .chain(std::iter::once(&correction))
        {
            if out.len() >= max_tokens {
                break;
            }
            out.push(t);
        }

        target.rewind(drafts.len() - accepted);

        // Draft rewind / catch-up — RECONCILE WITH SPECULATION
        if accepted == drafts.len() {
            // ALL ACCEPTED — speculative draft.feed(drafts.last()) was correct.
            // Skip the redundant draft.feed and use cached d_last_logit if we have it,
            // OR just compute draft.feed(correction).
            let _ts = std::time::Instant::now();
            let (l, sh) = draft.feed(&[correction])?;
            t_draft += _ts.elapsed();
            let dv = sh[2];
            d_last_logit = l[l.len() - dv..].to_vec();
            let _ = speculative_draft_done; // discard saved logit
        } else {
            // Speculation wrong: drafts.last() wasn't on the accepted path.
            // Rewind 1 (for the speculative draft.feed) + (d_advanced - accepted)
            // for the rejected drafts already in cache.
            let total_rewind = 1 + d_advanced - accepted;
            draft.rewind(total_rewind);
            let _ts = std::time::Instant::now();
            let (l, sh) = draft.feed(&[correction])?;
            t_draft += _ts.elapsed();
            let dv = sh[2];
            d_last_logit = l[l.len() - dv..].to_vec();
        }
        prev_correction = correction;
    }

    let total = t_total.elapsed();
    let other = total.saturating_sub(t_target).saturating_sub(t_draft);
    info!(
        target_ms = t_target.as_millis() as u64,
        draft_ms = t_draft.as_millis() as u64,
        other_ms = other.as_millis() as u64,
        total_ms = total.as_millis() as u64,
        // Per-task target.feed breakdown:
        // alpha_setup (set_input) + alpha_infer (stage_0 GPU compute)
        // + alpha_output (read stage_0 output) + wire (send +
        // downstream stage_1 + recv).
        target_alpha_setup_ms = target.t_alpha_setup.as_millis() as u64,
        target_alpha_infer_ms = target.t_alpha_infer.as_millis() as u64,
        target_alpha_output_ms = target.t_alpha_output.as_millis() as u64,
        target_wire_ms = target.t_wire.as_millis() as u64,
        "spec_decode timing"
    );
    Ok(out)
}

// -------- Driver-side Engine + Builder --------

/// In-flight spec-decode state. The whole spec_decode_greedy loop is
/// rebuilt into a per-`step()` state machine so the engine emits one
/// chunk per spec round (1 + accepted-drafts tokens at a time) instead
/// of buffering all output and emitting one giant final chunk.
struct ActiveSpec {
    task: GenerationTask,
    /// Issue-34: the prompt tokens (capture key = prompt ++ out). `kv_coord` only.
    #[cfg(feature = "kv_coord")]
    prompt_ids: Vec<i64>,
    /// Accumulated accepted tokens (including the first sampled token).
    out: Vec<i64>,
    /// Option B: length of the resume-seed prefix at the front of `out` (0 when
    /// not resuming). Unconditional (not `kv_coord`-gated) so both builds see
    /// the same `ActiveSpec` layout; used to keep the kv-capture key from
    /// double-counting the resume ids (they're already in `prompt_ids`).
    #[cfg_attr(not(feature = "kv_coord"), allow(dead_code))]
    resume_seed_len: usize,
    /// Cumulative byte-length of the detokenized text emitted so far.
    /// Used to compute the streaming delta on each step.
    /// Bytes already handed to the client (see `decode_delta`).
    emitted: Vec<u8>,
    /// Token id that became the "always-accepted" prefix for the next
    /// verify call (i.e. the correction from the previous round, or
    /// the first sampled token on the very first round).
    prev_correction: i64,
    /// Last draft logits (so we can argmax for the next round's first
    /// draft without re-feeding draft).
    d_last_logit: Vec<f32>,
    stats: SpecDecodeStats,
    /// Whether we already emitted the first sampled token's chunk.
    initialized: bool,
}

pub struct OvDistSpecEngine {
    target: DistributedMaskedReq,
    draft: MaskedReq,
    tokenizer: Arc<Tokenizer>,
    /// All EOS token ids configured for the model. Generation truncates
    /// at the first token that matches ANY of these.
    eos_token_ids: Vec<u32>,
    k: usize,
    pending: Vec<GenerationTask>,
    active: Option<ActiveSpec>,
    /// Issue-34: opaque KV blob cache + a stable model id (the target pipeline dir) for the
    /// coordination-plane fingerprint. `kv_coord` only.
    #[cfg(feature = "kv_coord")]
    kv: crate::kv_coordination::OvKvCache,
    /// Lock-free holder mirror of `kv` so a busy driver still answers cross-chain pulls.
    #[cfg(feature = "kv_coord")]
    kv_share: crate::kv_coordination::SharedKvCache,
    #[cfg(feature = "kv_coord")]
    kv_model_id: String,
}

impl Engine for OvDistSpecEngine {
    fn warmup(&mut self) {
        // Run a tiny spec-decode round to pre-pay the cold OV initialization
        // cost (kernel JIT, compiled-blob load, plugin first-touch).
        // Mirrors Python's `OvDistributedSpecEngine.warmup()` which does
        // `list(spec_decode_greedy_stream(target, draft, "Hi" tokens, max=2, k))`.
        // Without this the first user task pays ~700 ms of cold-init cost
        // and Rust shows up as 23% slower than Python in micro-benchmarks
        // even though the steady-state per-token cost is the same.
        let enc = match self.tokenizer.encode("Hi", false) {
            Ok(e) => e,
            Err(err) => {
                warn!(error = %err, "ov-dist-spec warmup tokenize failed");
                return;
            }
        };
        let prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
        let mut stats = SpecDecodeStats::default();
        match spec_decode_greedy(
            &mut self.target,
            &mut self.draft,
            &prompt_ids,
            2,
            self.k,
            &mut stats,
        ) {
            Ok(_) => info!(k = self.k, "ov-dist-spec warmup ok"),
            Err(err) => warn!(error = %err, "ov-dist-spec warmup failed"),
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|t| t.task_id == task.task_id)
            || self
                .active
                .as_ref()
                .is_some_and(|s| s.task.task_id == task.task_id)
        {
            return Ok(());
        }
        if self.pending.len() >= MAX_PENDING_TASKS {
            warn!(
                queued = self.pending.len(),
                cap = MAX_PENDING_TASKS,
                "ov-dist-spec: pending queue at cap; rejecting task"
            );
            return Err(EngineError::QueueFull {
                queued: self.pending.len(),
                cap: MAX_PENDING_TASKS,
            });
        }
        self.pending.push(task);
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        // Step-wise driver (rank 0): drop the queued task and clear the active
        // speculation so step() stops verifying rounds for it. The next
        // admission's reset re-syncs the worker downstream.
        self.pending.retain(|t| &t.task_id != task_id);
        if self
            .active
            .as_ref()
            .is_some_and(|a| &a.task.task_id == task_id)
        {
            self.active = None;
        }
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        // ---- Activate next pending task (one-time prompt feed + first sample).
        if self.active.is_none() {
            if self.pending.is_empty() {
                return Ok(Vec::new());
            }
            let task = self.pending.remove(0);
            let enc = match self.tokenizer.encode(task.prompt.clone(), false) {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = %e, "tokenize failed");
                    let task_id = task.task_id.clone();
                    let c = Chunk::error(task.task_id, format!("tokenize failed: {e}"));
                    return Ok(vec![(task_id, c)]);
                }
            };
            let mut prompt_ids: Vec<i64> = enc.get_ids().iter().map(|&u| u as i64).collect();
            // Option B forced-prefix resume: append the already-emitted assistant
            // tokens after the rendered prompt (concat, not replace) — flows into
            // BOTH the draft and target prefill via start_task's shared feed_ids.
            // No-op when not resuming (`resume_ids()` normalizes `Some([])` too).
            let resume = task.resume_ids().map(<[i32]>::to_vec);
            if let Some(r) = resume.as_deref() {
                // Peer-supplied indices reach native OV prefill — bound them first.
                let vocab = crate::resume_vocab_bound(&self.tokenizer);
                if let Err(e) = cascadia_types::validate_resume_ids(r, Some(vocab)) {
                    let task_id = task.task_id.clone();
                    let c = Chunk::error(task.task_id, format!("invalid resume prefix: {e}"));
                    return Ok(vec![(task_id, c)]);
                }
                // Exhausted budget (prefix >= max_tokens): finish Length with ZERO
                // new tokens BEFORE start_task — start_task unconditionally samples
                // and pushes the first token, which would overshoot the budget.
                if r.len() >= task.max_tokens.max(1) as usize {
                    let mut c = Chunk::final_marker(task.task_id.clone(), "");
                    c.n_tokens = Some(0);
                    c.finish_reason = Some(cascadia_types::FinishReason::Length);
                    return Ok(vec![(task.task_id.clone(), c)]);
                }
            }
            cascadia_types::append_resume_ids(&mut prompt_ids, resume.as_deref());
            info!(
                task = %task.task_id,
                prompt_tokens = prompt_ids.len(),
                k = self.k,
                "ov-dist-spec task active"
            );

            let task_id = task.task_id.clone();
            match self.start_task(task, &prompt_ids) {
                Ok(active) => self.active = Some(active),
                Err(e) => {
                    warn!(task = %task_id, error = %e, "ov-dist-spec start failed");
                    let c = Chunk::error(task_id.clone(), format!("start failed: {e}"));
                    return Ok(vec![(task_id, c)]);
                }
            }
        }

        // ---- Drive one spec round (or emit the first-token chunk).
        let active = self.active.as_mut().expect("just-set above");
        let task_id = active.task.task_id.clone();
        let max_tokens = active.task.max_tokens.max(1) as usize;

        // First step after init: emit a chunk for the first sampled token.
        let chunks = if !active.initialized {
            active.initialized = true;
            // If the first token already hits max_tokens or EOS, finalize now.
            // Decided before decoding: it is what makes this the terminal read,
            // which tells decode_delta whether it may hold anything back.
            // Option B: resume seeds a prefix ahead of the freshly sampled token,
            // so the newly sampled token is `out.last()`, not `out[0]` — checking
            // `out[0]` would test a RESUMED token for EOS and finalize instantly.
            let first_new = *active.out.last().unwrap_or(&0);
            let hit_eos = self.eos_token_ids.contains(&(first_new as u32));
            let finished = active.out.len() >= max_tokens || hit_eos;
            let new_text =
                decode_delta(&self.tokenizer, &active.out, &mut active.emitted, !finished);
            if finished {
                self.finish_task(task_id.clone(), new_text, 1)
            } else {
                vec![(
                    task_id,
                    Chunk {
                        task_id: active.task.task_id.clone(),
                        token_id: first_new,
                        text: new_text,
                        is_final: false,
                        logprobs: None,
                        n_tokens: Some(1),
                        prompt_tokens: None,
                        error: None,
                        token_ids: Vec::new(),
                        finish_reason: None,
                    },
                )]
            }
        } else {
            // Subsequent steps: do one full spec round.
            match self.do_one_round(max_tokens) {
                Err(e) => {
                    warn!(task = %task_id, error = %e, "ov-dist-spec round failed");
                    // Surface, don't `finish_task`: finishing turns a dead-peer transport
                    // error into a clean `[DONE]`, so the client gets a truncated answer
                    // presented as complete and issue-34's splicer — which only triggers on
                    // a surfaced error — never fires. Clearing `active` first (as runtime.rs
                    // does) keeps a stale task from wedging the next `submit()`.
                    // `for_task` is load-bearing: an unattributed Err from step() makes the
                    // runner fail whichever queued stream happened to poll, while THIS
                    // task's stream hangs with nothing further (review finding).
                    self.active = None;
                    return Err(e.for_task(task_id));
                }
                Ok(RoundResult {
                    delta,
                    finished,
                    n_tokens,
                    token_ids,
                }) => {
                    if finished {
                        self.finish_task(task_id, delta, n_tokens)
                    } else {
                        let active_ref = self.active.as_ref().unwrap();
                        let last_id = *active_ref.out.last().unwrap_or(&0);
                        vec![(
                            task_id.clone(),
                            Chunk {
                                task_id,
                                token_id: last_id,
                                text: delta,
                                is_final: false,
                                logprobs: None,
                                n_tokens: Some(n_tokens),
                                prompt_tokens: None,
                                error: None,
                                token_ids,
                                finish_reason: None,
                            },
                        )]
                    }
                }
            }
        };
        Ok(chunks)
    }

    #[cfg(feature = "kv_coord")]
    fn kv_coordination(&mut self) -> Option<&mut dyn cascadia_engine::KvCoordination> {
        Some(self)
    }

    #[cfg(feature = "kv_coord")]
    fn kv_holder(&self) -> Option<std::sync::Arc<dyn cascadia_engine::KvSnapshotHolder>> {
        Some(std::sync::Arc::new(crate::kv_coordination::OvKvHolder {
            cache: std::sync::Arc::clone(&self.kv_share),
            // Same fp the engine advertises via `model_fingerprint()` (model-level; see kv_fingerprint).
            model_fp: self.kv_fingerprint(),
        }))
    }
}

struct RoundResult {
    delta: String,
    finished: bool,
    /// How many model tokens this round added to `out`. Used to fill
    /// Chunk::n_tokens so downstream tok/s metrics stay accurate when
    /// each chunk carries multiple tokens (1..=K+1).
    n_tokens: u32,
    /// The token ids added to `out` this round, in order (accepted drafts
    /// plus the correction). Surfaced via `Chunk::token_ids` so a dist-spec
    /// multi-token chunk can serve as a resume source.
    token_ids: Vec<i64>,
}

impl OvDistSpecEngine {
    /// Initial setup for a new task: reset target+draft, feed the prompt,
    /// sample the first token (always-accepted), and prime the draft for
    /// the first round.
    fn start_task(
        &mut self,
        task: GenerationTask,
        prompt_ids: &[i64],
    ) -> Result<ActiveSpec, EngineError> {
        // Issue-34: warm-resume a cached strict prefix (restores draft+target+chain, all-or-nothing)
        // and feed only the suffix; else cold-reset + feed the full prompt. 0 on the default build.
        #[cfg(feature = "kv_coord")]
        let warm_len = self.kv_try_warm_resume(&task.tenant, prompt_ids);
        #[cfg(not(feature = "kv_coord"))]
        let warm_len = 0usize;
        if warm_len == 0 {
            self.target.reset()?;
            self.draft.reset()?;
        }
        self.target.t_alpha_setup = std::time::Duration::ZERO;
        self.target.t_alpha_infer = std::time::Duration::ZERO;
        self.target.t_alpha_output = std::time::Duration::ZERO;
        self.target.t_wire = std::time::Duration::ZERO;

        let feed_ids = &prompt_ids[warm_len..];
        let (t_logits, t_shape) = self.target.feed(feed_ids)?;
        self.draft.feed(feed_ids)?;
        let vocab = t_shape[2];
        let first = argmax(&t_logits[t_logits.len() - vocab..]) as i64;

        let (d_logits, d_shape) = self.draft.feed(&[first])?;
        let dv = d_shape[2];
        let d_last_logit = d_logits[d_logits.len() - dv..].to_vec();

        // Option B: pre-seed `out` with the resumed tokens so `out.len()` bounds
        // prefix+new against max_tokens (no separate subtraction needed), and seed
        // `emitted` with the seed's DECODED BYTES so `decode_delta` hands the client
        // only `full[emitted.len()..]` — the forced prefix is never re-emitted.
        // Empty on a normal turn ⇒ identical to today.
        let mut out = cascadia_types::resume_generated_seed(task.resume_ids());
        // Captured BEFORE `out.push(first)`: the kv_coord capture key must count
        // only the resumed prefix, not the freshly sampled token.
        let resume_seed_len = out.len();
        let emitted: Vec<u8> = if out.is_empty() {
            Vec::new()
        } else {
            let seed_u32: Vec<u32> = out.iter().map(|&t| t as u32).collect();
            // Propagate, never `unwrap_or_default()`: an empty seed here would make the
            // first delta re-emit the entire forced prefix to the client. Failing the
            // task is the honest outcome — same call as the qwen36 seed-decode fix.
            self.tokenizer
                .decode(&seed_u32, true)
                .map_err(|e| EngineError::Backend(format!("resume seed decode failed: {e}")))?
                .into_bytes()
        };
        out.push(first);

        Ok(ActiveSpec {
            task,
            #[cfg(feature = "kv_coord")]
            prompt_ids: prompt_ids.to_vec(),
            out,
            emitted,
            resume_seed_len,
            prev_correction: first,
            d_last_logit,
            stats: SpecDecodeStats::default(),
            initialized: false,
        })
    }

    /// Run one spec round: generate K drafts → target verifies → emit
    /// accepted tokens (truncated at EOS) → set up state for next round.
    fn do_one_round(&mut self, max_tokens: usize) -> Result<RoundResult, EngineError> {
        let active = self.active.as_mut().expect("active set");
        active.stats.n_steps += 1;

        // Generate K drafts greedily off the draft model.
        let mut drafts: Vec<i64> = vec![argmax(&active.d_last_logit) as i64];
        for _ in 1..self.k {
            if active.out.len() + drafts.len() >= max_tokens {
                break;
            }
            let prev = *drafts.last().unwrap();
            let (l, sh) = self.draft.feed(&[prev])?;
            let dv = sh[2];
            drafts.push(argmax(&l[l.len() - dv..]) as i64);
        }
        let d_advanced = drafts.len() - 1;
        active.stats.total_drafts += drafts.len() as u32;

        // Target verifies [prev_correction, drafts...].
        let mut verify = Vec::with_capacity(drafts.len() + 1);
        verify.push(active.prev_correction);
        verify.extend_from_slice(&drafts);
        let (t_logits, t_shape) = self.target.feed(&verify)?;
        let v = t_shape[2];
        let mut t_greedy = Vec::with_capacity(verify.len());
        for i in 0..verify.len() {
            t_greedy.push(argmax(&t_logits[i * v..(i + 1) * v]) as i64);
        }

        // Greedy acceptance: accept consecutive matches.
        let mut accepted = 0usize;
        for i in 0..drafts.len() {
            if t_greedy[i] == drafts[i] {
                accepted += 1;
            } else {
                break;
            }
        }
        active.stats.total_accepted += accepted as u32;

        let correction = if accepted < drafts.len() {
            t_greedy[accepted]
        } else {
            t_greedy[drafts.len()]
        };

        // Append [drafts[..accepted], correction] to out, but truncate at first EOS.
        let out_len_before = active.out.len();
        let mut hit_eos = false;
        let mut hit_max = false;
        for &t in drafts[..accepted]
            .iter()
            .chain(std::iter::once(&correction))
        {
            if active.out.len() >= max_tokens {
                hit_max = true;
                break;
            }
            if self.eos_token_ids.contains(&(t as u32)) {
                hit_eos = true;
                break;
            }
            active.out.push(t);
        }
        let n_tokens_added = (active.out.len() - out_len_before) as u32;
        // Slice the ids this round actually appended to `out`, in order — the
        // per-round accepted-ids source for `Chunk::token_ids`.
        let token_ids_added: Vec<i64> = active.out[out_len_before..].to_vec();

        // Rewind any rejected drafts from the target's KV cache.
        self.target.rewind(drafts.len() - accepted);

        // Reconcile the draft's KV cache to match what the target accepted,
        // then re-prime with the correction so next round's drafts continue
        // from the right state.
        if !hit_eos && !hit_max {
            self.draft.rewind(d_advanced - accepted);
            let (l, sh) = self.draft.feed(&[correction])?;
            let dv = sh[2];
            active.d_last_logit = l[l.len() - dv..].to_vec();
            active.prev_correction = correction;
        }

        let delta = decode_delta(
            &self.tokenizer,
            &active.out,
            &mut active.emitted,
            !(hit_eos || hit_max),
        );
        Ok(RoundResult {
            delta,
            finished: hit_eos || hit_max,
            n_tokens: n_tokens_added,
            token_ids: token_ids_added,
        })
    }

    /// Drop the active task, log the spec-decode summary, and return a
    /// final-marker chunk carrying the last text delta + token count.
    fn finish_task(
        &mut self,
        task_id: TaskId,
        last_delta: String,
        n_tokens: u32,
    ) -> Vec<(TaskId, Chunk)> {
        let Some(active) = self.active.take() else {
            return vec![(task_id.clone(), Chunk::final_marker(task_id, last_delta))];
        };
        // Issue-34: capture draft+target KV under (prompt ++ accepted) before the next start_task
        // resets. Best-effort + gated.
        #[cfg(feature = "kv_coord")]
        {
            // `prompt_ids` already carries the resume ids (Option B appends them
            // before start_task); `out` is seeded with the same ids so the fed
            // sequence lines up round-to-round. Skip the seed here so the capture
            // key matches the real fed depth instead of double-counting it.
            let tokens: Vec<i32> = active
                .prompt_ids
                .iter()
                .chain(active.out.iter().skip(active.resume_seed_len))
                .map(|&t| t as i32)
                .collect();
            // H.1b R2: this turn's namespace, read off THIS task's own state — never off a
            // plane-asserted value, which describes a pulled entry, not this turn.
            self.kv_capture(&active.task.tenant, tokens);
        }
        info!(
            task = %active.task.task_id,
            tokens = active.out.len(),
            steps = active.stats.n_steps,
            accept = active.stats.accept_rate(),
            "ov-dist-spec done"
        );
        // Surface every id this round appended, in order — including a
        // single-token final: `token_id == 0` with an empty `token_ids` reads
        // as id-less, and token 0 is a legal sample. A ZERO-token final keeps
        // it empty (this round appended nothing; `token_id` there is a prior
        // round's last token, not this chunk's).
        let token_ids = if n_tokens > 0 {
            let start = active.out.len().saturating_sub(n_tokens as usize);
            active.out[start..].to_vec()
        } else {
            Vec::new()
        };
        vec![(
            task_id.clone(),
            Chunk {
                task_id,
                token_id: *active.out.last().unwrap_or(&0),
                text: last_delta,
                is_final: true,
                logprobs: None,
                n_tokens: if n_tokens > 0 { Some(n_tokens) } else { None },
                prompt_tokens: None,
                error: None,
                token_ids,
                finish_reason: None,
            },
        )]
    }
}

/// Detokenize `tokens` and return only the text added since the last call.
/// `emitted` holds the bytes already handed out; `running` is false on the
/// terminal read, which flushes anything held back.
///
/// Delegates to the shim's `advance_emitted` rather than remembering a byte
/// LENGTH and slicing at it. That offset is not safe: a detokenizer can rewrite
/// earlier bytes, so a 3-byte U+FFFD run resolving into a wider glyph leaves
/// the offset inside that glyph, and slicing a `str` off a char boundary
/// panics. Guarding only on "the decode got shorter" missed that case.
fn decode_delta(
    tokenizer: &Tokenizer,
    tokens: &[i64],
    emitted: &mut Vec<u8>,
    running: bool,
) -> String {
    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let full = tokenizer.decode(&ids, true).unwrap_or_default();
    let (delta, diverged) = advance_emitted(emitted, full.as_bytes(), running);
    if diverged {
        warn!("spec-decode detokenizer decode diverged; re-anchored");
    }
    delta
}

pub struct OvDistSpecBuilder {
    pub pipeline_dir: PathBuf,
    pub draft_model_path: String,
    pub device: String,
    /// Device for the draft model. Defaults to `device` if unset.
    /// Setting CPU here lets the draft run concurrent with target stage_0
    /// on the GPU — frees the GPU during draft compute.
    pub draft_device: String,
    pub k: u32,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    /// Extra `(key, value)` OV plugin properties plumbed verbatim from the CLI.
    pub ov_properties: Vec<(String, String)>,
    stage0: Option<OvRuntime>,
    draft: Option<OvRuntime>,
    tokenizer: Option<Arc<Tokenizer>>,
    eos_token_ids: Vec<u32>,
    downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
}

impl OvDistSpecBuilder {
    pub fn new(
        pipeline_dir: impl Into<PathBuf>,
        draft_model_path: impl Into<String>,
        device: impl Into<String>,
        k: u32,
    ) -> Self {
        let device_s: String = device.into();
        Self {
            pipeline_dir: pipeline_dir.into(),
            draft_model_path: draft_model_path.into(),
            device: device_s.clone(),
            draft_device: device_s,
            k,
            cache_dir: None,
            kv_cache_precision: None,
            dyn_quant_group: None,
            ov_properties: Vec::new(),
            stage0: None,
            draft: None,
            tokenizer: None,
            eos_token_ids: Vec::new(),
            downstream: None,
        }
    }

    pub fn with_draft_device(mut self, device: impl Into<String>) -> Self {
        self.draft_device = device.into();
        self
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
    /// Append extra `(key, value)` OV plugin properties (CLI perf flags).
    pub fn with_ov_properties(mut self, props: Vec<(String, String)>) -> Self {
        self.ov_properties.extend(props);
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
        for (key, val) in &self.ov_properties {
            p = p.with(key, val);
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

        events.push(LoadProgress::message(
            "compiling target stage 0".to_string(),
        ));
        let stage0_xml = stage_dir.join("openvino_model.xml");
        let stage0 = OvRuntime::compile(
            stage0_xml.to_str().unwrap_or_default(),
            &self.device,
            &plugin,
        )
        .map_err(map_ov_err)?;
        self.stage0 = Some(stage0);

        events.push(LoadProgress::message("loading tokenizer".to_string()));
        let tok_path = self.pipeline_dir.join("tokenizer/tokenizer.json");
        if tok_path.exists() {
            let tok = Tokenizer::from_file(&tok_path)
                .map_err(|e| EngineError::Backend(format!("tokenizer load: {e}")))?;
            self.tokenizer = Some(Arc::new(tok));
            let eos_in_tok = lookup_eos(&self.pipeline_dir.join("tokenizer"));
            self.eos_token_ids = if eos_in_tok.is_empty() {
                lookup_eos(&self.pipeline_dir)
            } else {
                eos_in_tok
            };
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
            &self.draft_device,
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
        let target =
            DistributedMaskedReq::new(stage0, downstream, tokio::runtime::Handle::current())?;
        let masked_draft = MaskedReq::new(draft)?;
        Ok(Box::new(OvDistSpecEngine {
            #[cfg(feature = "kv_coord")]
            kv_model_id: self.pipeline_dir.to_string_lossy().into_owned(),
            target,
            draft: masked_draft,
            tokenizer,
            eos_token_ids: self.eos_token_ids.clone(),
            k: self.k as usize,
            pending: Vec::new(),
            active: None,
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
            #[cfg(feature = "kv_coord")]
            kv_share: std::sync::Arc::new(std::sync::Mutex::new(
                crate::kv_coordination::OvKvCache::default(),
            )),
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
    /// Issue-34: opaque KV blob cache + a model-level id for the coordination-plane fingerprint.
    /// `kv_coord` only.
    #[cfg(feature = "kv_coord")]
    kv: crate::kv_coordination::OvKvCache,
    /// Lock-free holder mirror of `kv` so a busy worker still answers cross-chain per-rank GETs.
    #[cfg(feature = "kv_coord")]
    kv_share: crate::kv_coordination::SharedKvCache,
    /// Issue-34 plane warm-resume: mailbox the plane's commit parks this rank's pulled slice in.
    /// Drained from the `FrameKind::Restore` handler; its lock is independent of the engine lock, so
    /// the commit path can deposit while this worker is mid-`step()`.
    #[cfg(feature = "kv_coord")]
    kv_handoff: std::sync::Arc<crate::kv_coordination::KvHandoffMailbox>,
    #[cfg(feature = "kv_coord")]
    kv_model_id: String,
}

impl Engine for OvDistSpecWorkerEngine {
    fn warmup(&mut self) {
        info!("ov-dist-spec worker warmup skipped");
    }

    fn submit(&mut self, _task: GenerationTask) -> EngineResult<()> {
        warn!("ov-dist-spec worker cannot accept tasks directly");
        Err(EngineError::Backend(
            "worker stage does not accept tasks directly".into(),
        ))
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        // RAII guard: marks the current thread as blocking-pool for
        // the duration of this step so run_async takes the cheap
        // bare-block_on path. The guard restores the previous flag
        // value on drop, preventing stale state if the spawn_blocking
        // pool ever migrates this thread to non-blocking work.
        let _guard = BlockingContextGuard::enter();
        let result = self.handle_one_frame();
        if let Err(ref e) = result {
            // Transport-closed errors signal the driver disconnected; don't
            // spam the log. Drop the upstream/downstream so subsequent steps
            // fail fast with NotConnected; the relay loop exits ConnectionFatal
            // on this Err (clean supervisor rebuild), and the sleep below only
            // throttles the single round before the loop observes the drop.
            // Fatal-classification is delegated to EngineError::is_connection_fatal
            // (single source of truth) — clean teardown, black-holed peer (idle
            // ceiling), peer crash (RST/broken pipe/aborted), mid-frame stall.
            if e.is_connection_fatal() {
                warn!(error = %e, "ov-dist-spec worker: connection-fatal transport error, dropping links");
                // Mark engine as drained by clearing connections.
                let _ = self.runtime_handle.clone().block_on(async {
                    let mut g = self.upstream.lock().await;
                    g.close().await;
                    Ok::<_, cascadia_transport::TransportError>(())
                });
                if let Some(d) = &self.downstream {
                    let _ = self.runtime_handle.clone().block_on(async {
                        let mut g = d.lock().await;
                        g.close().await;
                        Ok::<_, cascadia_transport::TransportError>(())
                    });
                }
                // Sleep a tick so the relay loop's busy spin doesn't
                // hot-loop until the runner notices.
                std::thread::sleep(std::time::Duration::from_millis(200));
            } else {
                // Non-connection-fatal error (bad frame kind, stray
                // LOGITS_RESPONSE, reset/inference failure). recv_raw keeps
                // succeeding on a connected-but-misbehaving peer, so without
                // a cool-off a bad-frame flood hot-spins the worker loop and
                // floods the log. Same 200 ms cadence as the fatal branch;
                // the sleep also caps the warn rate to ~5/s.
                warn!(error = %e, "ov-dist-spec worker step error");
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
        result.map(|_| Vec::new())
    }

    #[cfg(feature = "kv_coord")]
    fn kv_coordination(&mut self) -> Option<&mut dyn cascadia_engine::KvCoordination> {
        Some(self)
    }

    #[cfg(feature = "kv_coord")]
    fn kv_holder(&self) -> Option<std::sync::Arc<dyn cascadia_engine::KvSnapshotHolder>> {
        Some(std::sync::Arc::new(crate::kv_coordination::OvKvHolder {
            cache: std::sync::Arc::clone(&self.kv_share),
            // Same model-level fp the engine advertises (see kv_fingerprint) so the pull's single
            // asserted fingerprint matches this rank's GET.
            model_fp: self.kv_fingerprint(),
        }))
    }

    #[cfg(feature = "kv_coord")]
    fn kv_handoff(&self) -> Option<std::sync::Arc<dyn cascadia_engine::KvWarmHandoff>> {
        Some(std::sync::Arc::clone(&self.kv_handoff)
            as std::sync::Arc<dyn cascadia_engine::KvWarmHandoff>)
    }
}

impl OvDistSpecWorkerEngine {
    fn handle_one_frame(&mut self) -> Result<(), EngineError> {
        let upstream = self.upstream.clone();
        let downstream = self.downstream.clone();

        // 1. Read kind from upstream.
        let _ts_recv_kind = std::time::Instant::now();
        let kind = run_async(&self.runtime_handle, async {
            let mut g = upstream.lock().await;
            g.recv_raw(4)
                .await
                .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        })
        .map_err(|e| EngineError::Backend(e.to_string()))?;
        let recv_kind_us = _ts_recv_kind.elapsed().as_micros();
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
            // Issue-34 §8: snapshot this rank's KV under the head epoch, chain downstream, ack up.
            // V2 additionally carries the turn's tenant, which tags the stash and rides the chain on.
            #[cfg(feature = "kv_coord")]
            kind @ (FrameKind::Capture | FrameKind::CaptureV2) => {
                let up = self.upstream.clone();
                let body = run_async(&self.runtime_handle, async move {
                    let mut g = up.lock().await;
                    let lb = g.recv_raw(4).await?;
                    let n = u32::from_be_bytes([lb[0], lb[1], lb[2], lb[3]]) as usize;
                    // Chunked + drain-on-reject: recv_raw refuses a single over-cap read BEFORE
                    // consuming a byte, and a rejected-but-unconsumed body stays in the stream
                    // and desyncs every later frame. The body grows with the session
                    // (tokens ++ mask), so it passes 64 KiB at ≈5k tokens. Consume in capped
                    // chunks either way; assemble only an acceptable length.
                    let accept = n <= MAX_CAPTURE_BODY_BYTES;
                    let mut buf = Vec::new();
                    let mut remaining = n;
                    while remaining > 0 {
                        let take = remaining.min(MAX_RAW_BYTES);
                        let chunk = g.recv_raw(take).await?;
                        if chunk.len() != take {
                            return Err(cascadia_transport::TransportError::SocketClosed);
                        }
                        if accept {
                            buf.extend_from_slice(&chunk);
                        }
                        remaining -= take;
                    }
                    if !accept {
                        return Err(cascadia_transport::TransportError::Io(
                            std::io::Error::other(format!(
                                "capture body {n} exceeds MAX_CAPTURE_BODY_BYTES {MAX_CAPTURE_BODY_BYTES}"
                            )),
                        ));
                    }
                    Ok(buf)
                })
                .map_err(|e| EngineError::Backend(e.to_string()))?;
                let (epoch, tokens, valid, tenant) = if kind == FrameKind::CaptureV2 {
                    crate::kv_coordination::parse_capture_body_masked_v2(&body).ok_or_else(
                        || EngineError::Backend("ov-dist-spec: bad CAPTURE_V2 body".into()),
                    )?
                } else {
                    let (e, tk, v) = crate::kv_coordination::parse_capture_body_masked(&body)
                        .ok_or_else(|| {
                            EngineError::Backend("ov-dist-spec: bad CAPTURE body".into())
                        })?;
                    (e, tk, v, crate::kv_coordination::LOCAL_NS.to_string())
                };
                // Compact this rank's KV with the driver's mask (shared across the target chain).
                if let Ok(blob) = self.runtime.get_state_blob() {
                    match crate::kv_coordination::kv_compact_blob(&blob, &valid) {
                        Some(blob) => {
                            // Mirror into the lock-free holder cache so this rank serves its slice unblocked.
                            self.kv_share
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .capture_under_epoch_ns(
                                    &tenant,
                                    epoch,
                                    tokens.clone(),
                                    blob.clone(),
                                );
                            self.kv
                                .capture_under_epoch_ns(&tenant, epoch, tokens.clone(), blob);
                        }
                        None => warn!("ov-dist-spec worker: KV compaction failed; skip capture"),
                    }
                }
                if !self.is_last {
                    let _ = self.kv_chain_capture(epoch, &tokens, &valid, &tenant);
                }
                self.kv_ack_upstream(FrameKind::CaptureAck as u32, &[])
            }
            // RESTORE: set_state this rank's pulled slice; chain; ack the all-or-nothing verdict.
            #[cfg(feature = "kv_coord")]
            FrameKind::Restore => {
                let up = self.upstream.clone();
                // [epoch:8][carried_len:8 LE][carried] — carried is the head's inline cross-chain slice.
                let (eb, carried) = run_async(&self.runtime_handle, async move {
                    let mut g = up.lock().await;
                    let eb = g.recv_raw(8).await?;
                    let lb = g.recv_raw(8).await?;
                    let n =
                        u64::from_le_bytes(lb.as_slice().try_into().unwrap_or([0u8; 8])) as usize;
                    // recv_raw caps each read at MAX_RAW_BYTES; a whole-state KV blob is far larger, so
                    // drain it in capped chunks (matching the send-side write_all) or the stream desyncs.
                    // That includes an OVER-CEILING blob: rejecting without consuming it leaves the
                    // whole payload in the stream — drain, then reject.
                    let accept = n <= MAX_CARRY_BLOB_BYTES;
                    let blob = {
                        let mut buf = Vec::new();
                        let mut remaining = n;
                        while remaining > 0 {
                            let take = remaining.min(MAX_RAW_BYTES);
                            let chunk = g.recv_raw(take).await?;
                            if chunk.len() != take {
                                return Err(cascadia_transport::TransportError::SocketClosed);
                            }
                            if accept {
                                buf.extend_from_slice(&chunk);
                            }
                            remaining -= take;
                        }
                        buf
                    };
                    if !accept {
                        return Err(cascadia_transport::TransportError::Io(
                            std::io::Error::other(format!(
                                "restore carried blob {n} exceeds MAX_CARRY_BLOB_BYTES {MAX_CARRY_BLOB_BYTES}"
                            )),
                        ));
                    }
                    Ok::<_, cascadia_transport::TransportError>((eb, blob))
                })
                .map_err(|e| EngineError::Backend(e.to_string()))?;
                let epoch =
                    u64::from_le_bytes(eb.as_slice().try_into().map_err(|_| {
                        EngineError::Backend("ov-dist-spec: bad RESTORE body".into())
                    })?);
                // A non-empty carried slice IS the cross-chain move — the head pulled and handed this
                // rank its slice, so use it (mirrors the certified ov-runtime tail path). Only when
                // carried is empty (same-chain restore) fall back to this rank's own CAPTURE. Preferring
                // a stale local CAPTURE would shadow the carried path whenever the content epoch already
                // exists locally (e.g. the cert's B-chain warm-up serves the same prompt).
                // Drain FIRST — same reason as ov-runtime's RESTORE arm: in plane mode the head
                // parks this rank's slice AND still carries a blob inline, so a `||` after the
                // carried branch short-circuits the drain away and the rank warms from carried data
                // while the plane slice goes unread. Chain mode parks nothing, so this is a false
                // no-op there and the carried/capture path below is unchanged.
                let local_ok = if self.drain_kv_handoff(epoch) {
                    true
                } else if !carried.is_empty() {
                    match self.runtime.set_state_blob(&carried) {
                        Ok(()) => {
                            info!(
                                epoch,
                                blob_len = carried.len(),
                                "distspec_tail_restore_carried"
                            );
                            true
                        }
                        Err(e) => {
                            warn!(error = %e, epoch, "ov-dist-spec worker: set_state(carried) failed; rank cold");
                            self.kv_worker_scrub();
                            false
                        }
                    }
                } else if let Some((_, blob)) = self.kv.take_capture(epoch) {
                    match self.runtime.set_state_blob(&blob) {
                        Ok(()) => true,
                        Err(e) => {
                            warn!(error = %e, epoch, "ov-dist-spec worker: set_state(capture) failed; rank cold");
                            self.kv_worker_scrub();
                            false
                        }
                    }
                } else {
                    false
                };
                // Plane mode parks this rank's slice in the mailbox instead of carrying it on the
                // RESTORE, so without this both arms above come up empty and the verdict is
                // structurally false — the head then aborts and falls back to its local KV.

                let down_ok = if self.is_last {
                    true
                } else {
                    self.kv_chain_restore(epoch).unwrap_or(false)
                };
                self.kv_ack_upstream(
                    FrameKind::RestoreAck as u32,
                    &[u8::from(local_ok && down_ok)],
                )
            }
            #[cfg(feature = "kv_coord")]
            FrameKind::CaptureAck | FrameKind::RestoreAck => Err(EngineError::Backend(
                "ov-dist-spec worker: unexpected ack frame from upstream".into(),
            )),
            FrameKind::Forward => {
                // 2. Read FORWARD body (logical_pos, attn, hidden) from upstream.
                //    Keep hidden as raw bytes (f16 wire format) — the IR
                //    expects f16 for the hidden_states port and we want
                //    to avoid the f16→f32→f16 round-trip the previous
                //    impl did per spec round.
                let upstream2 = upstream.clone();
                let _ts_recv_body = std::time::Instant::now();
                let (logical_pos_start, attn, hidden_bytes, hidden_dtype, hidden_shape) =
                    run_async(&self.runtime_handle, async move {
                        let mut g = upstream2.lock().await;
                        let pos_bytes = g.recv_raw(4).await?;
                        let logical_pos_start = u32::from_be_bytes([
                            pos_bytes[0],
                            pos_bytes[1],
                            pos_bytes[2],
                            pos_bytes[3],
                        ]);
                        let (attn_t, _) = g.recv().await?;
                        let attn_i64 = bytes_to_i64(&attn_t.data);
                        let (hidden_t, _) = g.recv().await?;
                        Ok::<_, cascadia_transport::TransportError>((
                            logical_pos_start,
                            attn_i64,
                            hidden_t.data,
                            hidden_t.dtype,
                            [
                                hidden_t.shape[0] as usize,
                                hidden_t.shape[1] as usize,
                                hidden_t.shape[2] as usize,
                            ],
                        ))
                    })
                    .map_err(|e| EngineError::Backend(e.to_string()))?;
                let recv_body_us = _ts_recv_body.elapsed().as_micros();

                // 3. Run inference on this stage.
                let new_tokens = hidden_shape[1];
                let pos: Vec<i64> = (logical_pos_start as i64
                    ..(logical_pos_start as i64 + new_tokens as i64))
                    .collect();
                let in_hs =
                    self.inputs.get("hidden_states").cloned().ok_or_else(|| {
                        EngineError::Backend("missing hidden_states input".into())
                    })?;
                let in_attn =
                    self.inputs.get("attention_mask").cloned().ok_or_else(|| {
                        EngineError::Backend("missing attention_mask input".into())
                    })?;
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

                // The v5 shard's hidden_states port is f16. Pass the wire
                // bytes through directly to avoid f16→f32→f16 round-trip.
                let shim_dtype = match hidden_dtype {
                    WireDType::F16 => ShimDType::F16,
                    WireDType::F32 => ShimDType::F32,
                    _ => {
                        return Err(EngineError::Backend(format!(
                            "unexpected hidden dtype on wire {hidden_dtype:?}"
                        )))
                    }
                };
                let _ts_setup = std::time::Instant::now();
                self.runtime
                    .set_input(&in_hs, shim_dtype, &hidden_shape, &hidden_bytes)
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
                    .set_input(
                        &in_pos,
                        ShimDType::I64,
                        &[1, new_tokens],
                        &i64_to_bytes(&pos),
                    )
                    .map_err(map_ov_err)?;
                self.runtime
                    .set_input(&in_beam, ShimDType::I32, &[1], &0i32.to_le_bytes())
                    .map_err(map_ov_err)?;
                let setup_us = _ts_setup.elapsed().as_micros();
                let _ts_infer = std::time::Instant::now();
                self.runtime.infer().map_err(map_ov_err)?;
                let infer_us = _ts_infer.elapsed().as_micros();
                let _ts_out = std::time::Instant::now();
                let (out_dtype, out_shape, out_bytes) =
                    self.runtime.output(0).map_err(map_ov_err)?;
                let output_us = _ts_out.elapsed().as_micros();
                tracing::debug!(
                    n = new_tokens,
                    setup_us,
                    infer_us,
                    output_us,
                    "worker.frame timing"
                );
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
                let mut out_shape_wire = [1u32; MAX_RANK];
                for (i, d) in out_shape.iter().enumerate().take(MAX_RANK) {
                    out_shape_wire[i] = *d as u32;
                }

                if self.is_last {
                    // Send LOGITS_RESPONSE back to upstream
                    let upstream3 = upstream.clone();
                    let _ts_send = std::time::Instant::now();
                    run_async(&self.runtime_handle, async move {
                        let mut g = upstream3.lock().await;
                        g.send_raw(&(FrameKind::LogitsResponse as u32).to_be_bytes())
                            .await?;
                        let t = WireTensor::new(WireDType::F16, out_shape_wire, out_f16_bytes);
                        g.send(&t).await
                    })
                    .map_err(|e| EngineError::Backend(e.to_string()))?;
                    let send_us = _ts_send.elapsed().as_micros();
                    tracing::debug!(
                        recv_kind_us,
                        recv_body_us,
                        send_us,
                        "worker.frame wire timing (last stage)"
                    );
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
                        header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
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
                                kind_bytes[0],
                                kind_bytes[1],
                                kind_bytes[2],
                                kind_bytes[3],
                            ]);
                            if kv != FrameKind::LogitsResponse as u32 {
                                return Err(cascadia_transport::TransportError::SocketClosed);
                            }
                            let (logits_t, _) = dg.recv().await?;
                            let mut g = upstream3.lock().await;
                            g.send_raw(&(FrameKind::LogitsResponse as u32).to_be_bytes())
                                .await?;
                            g.send(&logits_t).await?;
                        }
                        Ok::<_, cascadia_transport::TransportError>(())
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
    /// Extra `(key, value)` OV plugin properties plumbed verbatim from the CLI.
    pub ov_properties: Vec<(String, String)>,
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
            ov_properties: Vec::new(),
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
    /// Append extra `(key, value)` OV plugin properties (CLI perf flags).
    pub fn with_ov_properties(mut self, props: Vec<(String, String)>) -> Self {
        self.ov_properties.extend(props);
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
        for (key, val) in &self.ov_properties {
            p = p.with(key, val);
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
        let _ = peers
            .upstream
            .ok_or_else(|| EngineError::PeerRejected("worker requires upstream".into()))?;
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
        let runtime =
            OvRuntime::compile(xml_path.to_str().unwrap_or_default(), &self.device, &plugin)
                .map_err(map_ov_err)?;
        self.inputs = v5_inputs(&runtime)?;
        for k in [
            "hidden_states",
            "attention_mask",
            "position_ids",
            "beam_idx",
        ] {
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
            #[cfg(feature = "kv_coord")]
            kv: crate::kv_coordination::OvKvCache::default(),
            #[cfg(feature = "kv_coord")]
            kv_share: std::sync::Arc::new(std::sync::Mutex::new(
                crate::kv_coordination::OvKvCache::default(),
            )),
            #[cfg(feature = "kv_coord")]
            kv_handoff: std::sync::Arc::new(crate::kv_coordination::KvHandoffMailbox::new()),
            #[cfg(feature = "kv_coord")]
            kv_model_id: self.pipeline_dir.to_string_lossy().into_owned(),
        }))
    }
}

// -------- Issue-34 §8: KvCoordination for the dist-spec head + worker --------

#[cfg(feature = "kv_coord")]
impl OvDistSpecEngine {
    /// head = rank 0 (holds draft + target stage0); workers are ranks 1.. .
    /// MODEL-level fp (the target pipeline dir — identical on every rank of a chain), NOT per-rank: a
    /// cross-chain pull asserts ONE fingerprint (the entry head's) for every rank's GET, so a per-rank
    /// fp would reject rank>0. Mirrors qwen36 (fcd3e4e). Identical sharding across chains guards the
    /// stage span, replacing the dropped per-rank component.
    fn kv_fingerprint(&self) -> u64 {
        crate::kv_coordination::fnv1a64(self.kv_model_id.as_bytes())
    }
    /// Snapshot draft + target stage0 into one bundle keyed by `tokens`, and broadcast CAPTURE down
    /// the target chain so every worker rank stashes its slice. Best-effort.
    fn kv_capture(&mut self, tenant: &str, tokens: Vec<i32>) {
        let (Some(d), Some(s)) = (self.draft.kv_blob(), self.target.stage0_blob()) else {
            return;
        };
        // Compact away spec-decode's proposed-then-rejected positions so the stored blob is dense +
        // self-describing (draft and target each use their own mask). Skip capture if either fails —
        // a padded blob would crash/attend junk on warm-resume (next resume falls back to cold).
        let (Some(d), Some(s)) = (
            crate::kv_coordination::kv_compact_blob(&d, self.draft.valid_prefix()),
            crate::kv_coordination::kv_compact_blob(&s, self.target.valid_prefix()),
        ) else {
            warn!("ov-dist-spec: KV compaction failed; skip capture");
            return;
        };
        let blob = crate::kv_coordination::frame_blobs(&[d, s]);
        let epoch = crate::kv_coordination::synth_epoch(&tokens);
        if let Err(e) = self.target.capture_downstream(epoch, &tokens, tenant) {
            warn!(error = %e, "ov-dist-spec: CAPTURE broadcast failed (best-effort)");
        }
        // Mirror into the lock-free holder cache so a busy driver serves this turn's KV unblocked.
        self.kv_share
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .capture(tenant, tokens.clone(), blob.clone());
        self.kv.capture(tenant, tokens, blob);
    }

    /// Try to warm-resume: restore draft + target-stage0 + the whole target chain (all-or-nothing)
    /// for a cached strict prefix. Returns the warm prefix length (0 ⇒ cold). On partial restore,
    /// resets everything cold (a partial restore would corrupt spec-decode).
    fn kv_try_warm_resume(&mut self, tenant: &str, prompt_ids: &[i64]) -> usize {
        // Spec-decode leaves the KV padded with proposed-then-rejected tokens (rig: draft 148 / target
        // 168 deep for a 98-token accepted prefix). `kv_capture` now compacts every rank's blob with its
        // host valid_mask (shipped down the CAPTURE chain), so stored blobs are dense + self-describing —
        // restorable like any other engine. After compaction the draft sits exactly one token behind the
        // target (the target holds the last bonus token the draft hasn't fed yet); the resume below
        // catches the draft up so both align before `start_task` feeds one shared suffix.
        let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
        let Some((blob, len, plane_pulled)) = self.kv.take_warm(tenant, &prompt_i32) else {
            return 0;
        };
        let Some(parts) = crate::kv_coordination::unframe_blobs(&blob) else {
            return 0;
        };
        if parts.len() != 2 {
            return 0;
        }
        let epoch = crate::kv_coordination::synth_epoch(&prompt_i32[..len]);
        // Cross-chain: hand the pulled downstream slice to the tail inline (stash_downstream_rank stashed
        // it under `epoch`; single-slot fallback for the 2-stage move). Empty ⇒ same-chain (tail restores
        // from its own CAPTURE).
        let carried = self
            .kv
            // Driver is rank 0 (rank>0 runs the worker engine), so this RESTORE addresses rank 1.
            .take_downstream(epoch, 1)
            .or_else(|| self.kv.take_downstream_single(1))
            .unwrap_or_default();
        let ok = self.draft.kv_restore(&parts[0])
            && self.target.stage0_restore(&parts[1])
            && self
                .target
                .restore_downstream(epoch, carried)
                .unwrap_or(false);
        if ok {
            // Each model's cursor = its real KV depth, not the token count (off-by-one — see
            // kv_seq_from_blob). Steady state: draft is 0..=1 behind the target. Align to the target by
            // feeding the draft the gap token(s) (their KV is exactly what the target already holds), so
            // `start_task`'s single shared suffix feed lands at the right position on both. Draft ahead,
            // or a wider gap, can't be reconciled by one suffix ⇒ cold.
            // Neither position may be CLAMPED or GUESSED. The guard below exists to reject a draft that
            // is ahead of the target, and both `.min(len)` and `unwrap_or(len)` erased exactly that
            // signal: a draft deeper than the matched prefix — or one whose depth failed to parse — came
            // back as `d_pos == len == t_pos`, which reads as a legal zero gap. `kv_set_pos` then pointed
            // the draft past what its tensors actually hold and the next shared-suffix feed died inside
            // OpenVINO's eltwise broadcast ("Argument shapes are inconsistent"), taking the turn (and the
            // engine) with it. Rig: warm_prefix=98/d_pos=97 resumed cleanly; warm_prefix=154/d_pos=154
            // faulted. Unknown depth is now cold, not a fabricated position — cold is always recoverable.
            let (Some(d_pos), Some(t_raw)) = (
                crate::kv_coordination::kv_seq_from_blob(&parts[0]),
                crate::kv_coordination::kv_seq_from_blob(&parts[1]),
            ) else {
                warn!("ov-dist-spec: draft/target KV depth unreadable; cold reprefill");
                self.target.kv_scrub();
                self.draft.kv_scrub();
                return 0;
            };
            // The target defines the resume point, so it alone clamps to the matched prefix.
            let t_pos = t_raw.min(len);
            if d_pos <= t_pos && t_pos - d_pos <= 1 {
                self.target.kv_set_pos(t_pos);
                self.draft.kv_set_pos(d_pos);
                // Catch the draft up to the target (gap is prompt[d_pos..t_pos], the accepted tail).
                if d_pos < t_pos && self.draft.feed(&prompt_ids[d_pos..t_pos]).is_err() {
                    self.target.kv_scrub();
                    self.draft.kv_scrub();
                    return 0;
                }
                info!(
                    warm_prefix = t_pos,
                    d_pos,
                    matched = len,
                    plane_pulled,
                    "ov-dist-spec pipeline warm-resumed"
                );
                t_pos
            } else {
                warn!(
                    d_pos,
                    t_pos, len, "ov-dist-spec: draft/target KV depth out of range; cold reprefill"
                );
                self.target.kv_scrub();
                self.draft.kv_scrub();
                0
            }
        } else {
            // Restore failed (possibly partial) or the chain NAK'd after a full apply — either
            // way donor state may be live; reset_state alone cannot clear it.
            warn!("ov-dist-spec: warm restore abandoned; scrubbing cold");
            self.target.kv_scrub();
            self.draft.kv_scrub();
            0
        }
    }
}

#[cfg(feature = "kv_coord")]
impl cascadia_engine::KvCoordination for OvDistSpecEngine {
    fn model_fingerprint(&self) -> u64 {
        self.kv_fingerprint()
    }
    fn layout_version(&self) -> u16 {
        cascadia_kv_wire::OPAQUE_KV_LAYOUT
    }
    fn engine_rev(&self) -> u64 {
        crate::kv_coordination::KV_ENGINE_REV
    }
    fn tokenize(&self, text: &str) -> Option<Vec<i32>> {
        let enc = self.tokenizer.encode(text, false).ok()?;
        Some(enc.get_ids().iter().map(|&u| u as i32).collect())
    }
    fn lookup(&mut self, partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        self.kv.lookup(partner, token_ids)
    }
    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(cascadia_kv_wire::Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let fp = self.kv_fingerprint();
        let (prefix, blob) = self.kv.serve(partner, expected_epoch, expected_len)?;
        Some(crate::kv_coordination::blob_to_wire(
            &prefix,
            &blob,
            partner,
            fp,
            expected_epoch,
        ))
    }
    fn insert(
        &mut self,
        partner: &str,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        let (tokens, blob) = crate::kv_coordination::wire_to_blob(manifest, payloads).ok_or(())?;
        // H.1b hard gate (§12.10.0a): key on the ASSERTED partner, never `manifest.partner`, which
        // the serving holder stamps and nothing validates.
        self.kv.insert_both(partner, tokens, blob);
        Ok(())
    }
    // Issue-34 cross-chain CHAIN path: a downstream rank's pulled blob can't be used locally; stash it
    // under the content epoch so `restore_downstream` ships it inline in the RESTORE frame to that rank.
    // Without this the trait default returns Err ⇒ the head drops every downstream rank's pulled blob,
    // the pull votes cold (no hit), and the tail never receives a carried slice. Mirrors ov-runtime/qwen36.
    fn stash_downstream_rank(
        &mut self,
        rank: u16,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        let (_tokens, blob) = crate::kv_coordination::wire_to_blob(manifest, payloads).ok_or(())?;
        let epoch = crate::kv_coordination::synth_epoch(&manifest.token_ids);
        self.kv.stash_downstream(epoch, rank, blob);
        Ok(())
    }
}

#[cfg(feature = "kv_coord")]
impl OvDistSpecWorkerEngine {
    /// MODEL-level fp — identical on every rank so a cross-chain pull's single asserted fingerprint
    /// matches rank>0 GETs. See the driver's `kv_fingerprint` for the full rationale (fcd3e4e).
    fn kv_fingerprint(&self) -> u64 {
        crate::kv_coordination::fnv1a64(self.kv_model_id.as_bytes())
    }
    /// Drain the plane hand-off mailbox and apply the parked slice, if any. See
    /// [`crate::kv_coordination::drain_handoff`]; called from `FrameKind::Restore` for the same two
    /// reasons ov-runtime calls it from `OPCODE_RESTORE`: it lands before the turn's forward, and it
    /// is the only site on the same stream as the commit that parks the slice.
    ///
    /// Position 0 because a worker holds no KV cursor of its own — the driver owns the spec-decode
    /// positions — so the depth guard has nothing to compare against.
    fn drain_kv_handoff(&mut self, expected_epoch: u64) -> bool {
        let mailbox = std::sync::Arc::clone(&self.kv_handoff);
        let fp = self.kv_fingerprint();
        let runtime = &mut self.runtime;
        crate::kv_coordination::drain_handoff(&mailbox, fp, 0, expected_epoch, |blob| {
            match runtime.set_state_blob(blob) {
                Ok(()) => true,
                Err(e) => {
                    warn!(error = %e, "ov-dist-spec worker: set_state(handoff) failed; rank cold");
                    // A failed set_state may be PARTIAL; reset_state cannot scrub it.
                    if let Err(e2) = runtime.recreate_request() {
                        error!(error = %e2, "ov-dist-spec worker: recreate_request scrub failed; KV state may be dirty");
                    }
                    false
                }
            }
        })
    }
    /// Scrub the worker request after a failed (possibly partial) `set_state_blob` —
    /// `reset_state` cannot clear post-`set_state` residue, and the head's cold fallback
    /// only sends Reset, which bottoms out in exactly that insufficient scrub.
    fn kv_worker_scrub(&mut self) {
        if let Err(e) = self.runtime.recreate_request() {
            error!(error = %e, "ov-dist-spec worker: recreate_request scrub failed; KV state may be dirty");
        }
    }
    fn kv_ack_upstream(&mut self, ack_kind: u32, payload: &[u8]) -> Result<(), EngineError> {
        let up = self.upstream.clone();
        let payload = payload.to_vec();
        run_async(&self.runtime_handle, async move {
            let mut g = up.lock().await;
            g.send_raw(&ack_kind.to_be_bytes()).await?;
            if !payload.is_empty() {
                g.send_raw(&payload).await?;
            }
            Ok::<_, cascadia_transport::TransportError>(())
        })
        .map_err(|e| EngineError::Backend(e.to_string()))
    }
    /// Relay CAPTURE on. `tenant` is the one the frame carried (empty on a v1 frame) — a worker has
    /// no other source for it, so the relayed frame keeps whatever shape it arrived in.
    fn kv_chain_capture(
        &mut self,
        epoch: u64,
        tokens: &[i32],
        valid: &[i64],
        tenant: &str,
    ) -> Result<(), EngineError> {
        let Some(down) = self.downstream.clone() else {
            return Ok(());
        };
        let (kind, body) = if tenant.is_empty() {
            (
                FrameKind::Capture,
                crate::kv_coordination::capture_body_bytes_masked(epoch, tokens, valid),
            )
        } else {
            (
                FrameKind::CaptureV2,
                crate::kv_coordination::capture_body_bytes_masked_v2(epoch, tokens, valid, tenant),
            )
        };
        run_async(&self.runtime_handle, async move {
            let mut g = down.lock().await;
            g.send_raw(&(kind as u32).to_be_bytes()).await?;
            g.send_raw(&(body.len() as u32).to_be_bytes()).await?;
            g.send_raw(&body).await?;
            // Bounded + close-on-elapse (see KV_ACK_TIMEOUT): a mid-rank wedged on a silent
            // downstream otherwise never acks upstream either, wedging the whole chain.
            match tokio::time::timeout(KV_ACK_TIMEOUT, g.recv_raw(4)).await {
                Ok(kb) => {
                    let kb = kb?;
                    if u32::from_be_bytes([kb[0], kb[1], kb[2], kb[3]])
                        != FrameKind::CaptureAck as u32
                    {
                        return Err(cascadia_transport::TransportError::SocketClosed);
                    }
                    Ok(())
                }
                Err(_) => {
                    g.close().await;
                    Err(cascadia_transport::TransportError::Io(
                        std::io::Error::other(
                            "capture relay ack timed out; connection dropped to avoid pairing the \
                         late ack with the next exchange",
                        ),
                    ))
                }
            }
        })
        .map_err(|e| EngineError::Backend(e.to_string()))
    }
    fn kv_chain_restore(&mut self, epoch: u64) -> Result<bool, EngineError> {
        let Some(down) = self.downstream.clone() else {
            return Ok(true);
        };
        run_async(&self.runtime_handle, async move {
            let mut g = down.lock().await;
            g.send_raw(&(FrameKind::Restore as u32).to_be_bytes())
                .await?;
            g.send_raw(&epoch.to_le_bytes()).await?;
            // MUST match the reader's [epoch:8][carried_len:8 LE][carried]. Omitting carried_len made
            // every rank1→rank2 hop malformed: the receiver blocks reading a length that never arrives
            // while this side blocks on the ack (or, if any later bytes land, reads them AS the length).
            // Zero-length is the correct value here — a mid-chain rank only ever receives its OWN
            // carried slice, never the next rank's, so downstream restores from its own CAPTURE. On a
            // cross-chain move a 3rd+ rank has no capture under the donor epoch, so it verdicts 0 and
            // the chain cold-resets: correct-and-cold rather than a desynced stream.
            g.send_raw(&0u64.to_le_bytes()).await?;
            // Bounded + close-on-elapse (see KV_ACK_TIMEOUT).
            match tokio::time::timeout(KV_ACK_TIMEOUT, async {
                let kb = g.recv_raw(4).await?;
                if u32::from_be_bytes([kb[0], kb[1], kb[2], kb[3]]) != FrameKind::RestoreAck as u32
                {
                    return Err(cascadia_transport::TransportError::SocketClosed);
                }
                let vb = g.recv_raw(1).await?;
                Ok(vb.first() == Some(&1))
            })
            .await
            {
                Ok(r) => r,
                Err(_) => {
                    g.close().await;
                    Err(cascadia_transport::TransportError::Io(
                        std::io::Error::other(
                            "restore relay ack timed out; connection dropped to avoid pairing the \
                         late ack with the next exchange",
                        ),
                    ))
                }
            }
        })
        .map_err(|e| EngineError::Backend(e.to_string()))
    }
}

#[cfg(feature = "kv_coord")]
impl cascadia_engine::KvCoordination for OvDistSpecWorkerEngine {
    fn model_fingerprint(&self) -> u64 {
        self.kv_fingerprint()
    }
    fn layout_version(&self) -> u16 {
        cascadia_kv_wire::OPAQUE_KV_LAYOUT
    }
    fn engine_rev(&self) -> u64 {
        crate::kv_coordination::KV_ENGINE_REV
    }
    fn tokenize(&self, _text: &str) -> Option<Vec<i32>> {
        None // workers have no tokenizer + never NEGOTIATE
    }
    fn lookup(&mut self, _partner: &str, _token_ids: &[i32]) -> Option<(u64, u32)> {
        None
    }
    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(cascadia_kv_wire::Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let fp = self.kv_fingerprint();
        let (prefix, blob) = self.kv.serve(partner, expected_epoch, expected_len)?;
        Some(crate::kv_coordination::blob_to_wire(
            &prefix,
            &blob,
            partner,
            fp,
            expected_epoch,
        ))
    }
    fn insert(
        &mut self,
        partner: &str,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        let (tokens, blob) = crate::kv_coordination::wire_to_blob(manifest, payloads).ok_or(())?;
        // H.1b hard gate (§12.10.0a): key on the ASSERTED partner, never `manifest.partner`, which
        // the serving holder stamps and nothing validates.
        self.kv.insert_both(partner, tokens, blob);
        Ok(())
    }
    // Issue-34 cross-chain CHAIN path: a downstream rank's pulled blob can't be used locally; stash it
    // under the content epoch so `restore_downstream` ships it inline in the RESTORE frame to that rank.
    // Without this the trait default returns Err ⇒ the head drops every downstream rank's pulled blob,
    // the pull votes cold (no hit), and the tail never receives a carried slice. Mirrors ov-runtime/qwen36.
    fn stash_downstream_rank(
        &mut self,
        rank: u16,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        let (_tokens, blob) = crate::kv_coordination::wire_to_blob(manifest, payloads).ok_or(())?;
        let epoch = crate::kv_coordination::synth_epoch(&manifest.token_ids);
        self.kv.stash_downstream(epoch, rank, blob);
        Ok(())
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

/// Return ALL EOS token ids configured for the model. Mirrors the same
/// fix applied to runtime.rs — Llama 3.x lists multiple EOS tokens
/// (`<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`) and the chat-end one
/// is usually the LAST entry. spec_decode_greedy itself doesn't check
/// EOS (no parameter), so the caller (OvDistSpecEngine::step) truncates
/// the returned token vector at the first match.
fn lookup_eos(model_dir: &std::path::Path) -> Vec<u32> {
    for fname in ["generation_config.json", "config.json"] {
        let p = model_dir.join(fname);
        if let Ok(bytes) = std::fs::read(&p) {
            if let Ok(g) = serde_json::from_slice::<GenerationCfg>(&bytes) {
                return match g.eos_token_id {
                    Some(EosId::One(id)) => vec![id],
                    Some(EosId::Many(ids)) => ids,
                    None => Vec::new(),
                };
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_transport::{ActivationClient, ActivationServer};

    /// `decode_delta` used to remember a byte LENGTH and slice `full[len..]`.
    /// Its only guard was `full.len() <= last_text_len`, which catches a decode
    /// that got shorter but not one that grew while the boundary moved: a
    /// 3-byte U+FFFD run resolving into a 4-byte glyph leaves the offset inside
    /// that glyph, and slicing a `str` off a char boundary PANICS — taking the
    /// worker down, not just corrupting one response.
    #[test]
    fn delta_survives_a_replacement_run_resolving_into_a_wider_glyph() {
        let decodes = ["abc", "abc\u{FFFD}", "abc\u{1F600}", "abc\u{1F600}!"];
        // The offset the old code would have sliced at, inside the emoji.
        let stale = "abc\u{FFFD}".len();
        assert!(
            !"abc\u{1F600}".is_char_boundary(stale),
            "this sequence is only interesting because the offset lands mid-glyph"
        );

        let mut emitted: Vec<u8> = Vec::new();
        let mut client = String::new();
        for (i, full) in decodes.iter().enumerate() {
            let (delta, _) = advance_emitted(&mut emitted, full.as_bytes(), i + 1 < decodes.len());
            client.push_str(&delta);
        }
        assert_eq!(client, "abc\u{1F600}!");
        assert!(!client.contains('\u{FFFD}'));
    }

    #[test]
    fn frame_kind_roundtrip() {
        for k in [
            FrameKind::Forward,
            FrameKind::Reset,
            FrameKind::LogitsResponse,
        ] {
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
        let f = vec![1.0f32, -2.5, 3.5, 0.0];
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
            let kind =
                FrameKind::from_u32(u32::from_be_bytes([kb[0], kb[1], kb[2], kb[3]])).unwrap();
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
            let kind =
                FrameKind::from_u32(u32::from_be_bytes([kb[0], kb[1], kb[2], kb[3]])).unwrap();
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
