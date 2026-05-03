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
    /// v6: hidden_states + 4D additive f16 attention_mask + custom position_ids.
    /// Wire layout:
    ///   [4B kind=5 BE][4B logical_pos_start BE]
    ///   + send_tensor(attention_mask)  # f16 [1, n, total]   (interpreted as [1,1,n,total])
    ///   + send_tensor(position_ids)    # i64 [1, 1, n]
    ///   + send_tensor(hidden_states)   # f16 [1, n, hidden]
    ForwardV6 = 5,
}

impl FrameKind {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Forward),
            3 => Some(Self::Reset),
            4 => Some(Self::LogitsResponse),
            5 => Some(Self::ForwardV6),
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

/// f16 numeric "minus infinity" used in additive masks. PyTorch uses
/// `torch.finfo(f16).min` ≈ -65504.0; matches what the v6 export
/// script bakes into the trace example.
const F16_NEG_INF_F32: f32 = -65504.0_f32;

/// Build a chain-spec causal+pad attention mask in additive f16 form.
/// Output shape: [1, 1, query_len, total_seq_len] flattened to [1, query_len, total_seq_len]
/// for wire transport (the leading "1, 1" both collapse to a single "1" — same byte layout).
///
/// `valid_mask`: per-key allow/block flags for cached keys (1 = allowed, 0 = blocked).
/// `cache_len`: number of past keys (these come first in `valid_mask`).
/// `query_len`: number of new query tokens. Their cache positions are
/// `cache_len .. cache_len + query_len` and are always allowed.
/// `query_logical_pos`: absolute logical position of the first query token
/// (used to enforce causal triangle relative to historical positions when
/// queries fan out at the same depth — for chain spec, this is just
/// `cache_len` semantically).
///
/// Returns: f16 bytes of length `query_len * total * 2` (LE).
fn build_chain_mask_f16(valid_mask: &[i64], cache_len: usize, query_len: usize) -> Vec<u8> {
    use half::f16;
    let total = cache_len + query_len;
    let mut out = Vec::with_capacity(query_len * total * 2);
    let zero = f16::from_f32(0.0_f32);
    let blk = f16::from_f32(F16_NEG_INF_F32);
    for q in 0..query_len {
        // Query at relative position `q` sees:
        //  - cached keys [0, cache_len): allowed iff valid_mask[k] == 1
        //  - new query keys [cache_len, cache_len + q]: allowed (causal)
        //  - new query keys (cache_len + q, total): blocked (future)
        for k in 0..total {
            let allowed = if k < cache_len {
                valid_mask[k] != 0
            } else {
                k <= cache_len + q
            };
            let v = if allowed { zero } else { blk };
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    out
}

/// Build a "two parallel chains" attention mask in additive f16 form.
/// Used by parallel-draft tree-spec: at each depth iteration we feed
/// `[L_i, R_i]` together and need each token to see only its own chain's
/// past entries plus prev-round (valid_mask) cache.
///
/// `chain_owner`: per cache slot in `[round_base, cache_len)`, 0 if owned
/// by the LEFT chain (visible to L_i but blocked from R_i), 1 if owned
/// by the RIGHT chain (vice versa).
/// `round_base`: cache_len at the start of THIS spec-decode round (entries
/// before this index are common past, after are this round's chain entries).
/// Returns f16 bytes for shape `[1, 1, 2, cache_len + 2]` (two queries, the
/// pair `[L_i, R_i]`).
fn build_pair_mask_f16(
    valid_mask: &[i64],
    cache_len: usize,
    round_base: usize,
    chain_owner: &[u8],
) -> Vec<u8> {
    use half::f16;
    let total = cache_len + 2;
    let mut out = Vec::with_capacity(2 * total * 2);
    let zero = f16::from_f32(0.0_f32);
    let blk = f16::from_f32(F16_NEG_INF_F32);
    for query_chain in 0..2u8 {
        for k in 0..total {
            let allowed = if k < round_base {
                valid_mask[k] != 0
            } else if k < cache_len {
                let owner = chain_owner[k - round_base];
                valid_mask[k] != 0 && owner == query_chain
            } else {
                // New query slots: each query sees only itself.
                k == cache_len + query_chain as usize
            };
            let v = if allowed { zero } else { blk };
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    out
}

/// Build a tree-spec attention mask in additive f16 form.
///
/// `valid_mask` / `cache_len`: same as chain.
/// `parents`: for each new query token i ∈ [0, query_len), the index
/// of its parent in the same query batch, or -1 if its parent is the
/// last cached token. Token i may attend to:
///   - all valid cached keys (same as chain),
///   - itself (k == cache_len + i),
///   - and all of its ancestors in the query batch.
///
/// This produces a topology-aware mask consistent with the SpecInfer /
/// EAGLE-2 "flat tree" formulation.
fn build_tree_mask_f16(valid_mask: &[i64], cache_len: usize, parents: &[i32]) -> Vec<u8> {
    use half::f16;
    let query_len = parents.len();
    let total = cache_len + query_len;
    let mut out = Vec::with_capacity(query_len * total * 2);
    let zero = f16::from_f32(0.0_f32);
    let blk = f16::from_f32(F16_NEG_INF_F32);

    // Precompute ancestor sets: for each query token i, the set of
    // {i, parent(i), grandparent(i), ...} (within the query batch).
    let mut ancestors: Vec<Vec<usize>> = vec![Vec::new(); query_len];
    for i in 0..query_len {
        let mut cur = i as i32;
        while cur >= 0 {
            ancestors[i].push(cur as usize);
            cur = parents[cur as usize];
        }
    }

    for i in 0..query_len {
        for k in 0..total {
            let allowed = if k < cache_len {
                valid_mask[k] != 0
            } else {
                let q_pos = k - cache_len;
                ancestors[i].contains(&q_pos)
            };
            let v = if allowed { zero } else { blk };
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    out
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
pub(crate) fn run_async_pub<F: std::future::Future>(
    handle: &tokio::runtime::Handle,
    fut: F,
) -> F::Output {
    run_async(handle, fut)
}

fn run_async<F: std::future::Future>(handle: &tokio::runtime::Handle, fut: F) -> F::Output {
    if BLOCKING_CONTEXT.with(|f| f.get()) {
        // We're on a spawn_blocking thread (worker relay loop); naked
        // block_on is safe and ~250x cheaper than block_in_place wrap.
        handle.block_on(fut)
    } else if tokio::runtime::Handle::try_current().is_ok() {
        // We're on an async worker thread (driver's ChunkStream poll).
        tokio::task::block_in_place(|| handle.block_on(fut))
    } else {
        // No tokio context at all — naked block_on.
        handle.block_on(fut)
    }
}

thread_local! {
    static BLOCKING_CONTEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII guard that marks the current thread as a blocking-pool thread
/// for the duration of its scope. Acquired by the worker engine at the
/// top of each `step()`. Scoped so that if the spawn_blocking thread
/// pool ever reuses this thread for non-blocking work later, the flag
/// is cleared correctly. The previous `pub fn enter_blocking_context()`
/// was a one-way door; this guard fixes that.
pub(crate) struct BlockingContextGuard {
    prev: bool,
}

impl BlockingContextGuard {
    pub(crate) fn enter() -> Self {
        let prev = BLOCKING_CONTEXT.with(|f| {
            let old = f.get();
            f.set(true);
            old
        });
        Self { prev }
    }
}

impl Drop for BlockingContextGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        BLOCKING_CONTEXT.with(|f| f.set(prev));
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

/// v6 FORWARD: same as Forward but carries 4D additive f16 mask + custom position_ids.
async fn send_forward_v6(
    sock: &mut TcpStream,
    attn_mask_f16: Vec<u8>,
    query_len: usize,
    total_keys: usize,
    position_ids: &[i64],
    hidden_shape: [u32; MAX_RANK],
    hidden_data: Vec<u8>,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let kind = (FrameKind::ForwardV6 as u32).to_be_bytes();
    sock.write_all(&kind).await?;
    let attn_tensor = WireTensor::new(
        WireDType::F16,
        [1, query_len as u32, total_keys as u32],
        attn_mask_f16,
    );
    send_tensor(sock, &attn_tensor).await.map_err(io_err)?;
    let pos_tensor = WireTensor::new(WireDType::I64, [1, 1, query_len as u32], i64_to_bytes(position_ids));
    send_tensor(sock, &pos_tensor).await.map_err(io_err)?;
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
    /// True when this IR was exported with v6 4D additive f16 mask.
    /// Set externally via `set_v6` based on stage_config.json detection.
    pub is_v6: bool,
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
            is_v6: false,
        })
    }

    pub fn set_v6(&mut self, v: bool) {
        self.is_v6 = v;
    }

    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    pub fn logical_pos(&self) -> usize {
        self.logical_pos
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
        if self.is_v6 {
            let attn_f16 = build_chain_mask_f16(&self.valid_mask, self.cache_len, n);
            self.runtime
                .set_input(&attn_name, ShimDType::F16, &[1, 1, n, total], &attn_f16)
                .map_err(map_ov_err)?;
        } else {
            self.runtime
                .set_input(
                    &attn_name,
                    ShimDType::I64,
                    &[1, total],
                    &i64_to_bytes(&attn),
                )
                .map_err(map_ov_err)?;
        }
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

    /// Invalidate the last `n` cache entries WITHOUT touching `logical_pos`.
    /// Use when cache entries don't map 1:1 to logical positions (e.g.
    /// `feed_pair` writes 2 cache entries but advances logical_pos by 1
    /// because the siblings share their absolute position).
    pub fn invalidate_recent(&mut self, n: usize) {
        let lo = self.cache_len.saturating_sub(n);
        for i in lo..self.cache_len {
            self.valid_mask[i] = 0;
        }
    }

    /// Tree-feed: process a flat tree of `n` tokens in a single batched
    /// forward, with topology determined by `parents`. Requires v6 IR
    /// (4D mask). Returns logits at every tree position so the caller
    /// can walk arbitrary chains.
    ///
    /// `input_ids`, `position_ids`, `parents` must all have the same length.
    /// `parents[i]` is the index of token `i`'s parent in the same flat
    /// sequence, or -1 if its parent is the most recent cached entry.
    ///
    /// Cache management: `cache_len` advances by `n`; `logical_pos` is NOT
    /// updated (caller advances it via `confirm_tree_path` after picking
    /// the winning path).
    pub fn feed_tree(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        parents: &[i32],
    ) -> Result<(Vec<f32>, [usize; 3]), EngineError> {
        if !self.is_v6 {
            return Err(EngineError::Backend(
                "MaskedReq.feed_tree requires v6 IR".into(),
            ));
        }
        let n = input_ids.len();
        if position_ids.len() != n || parents.len() != n {
            return Err(EngineError::Backend(format!(
                "feed_tree length mismatch: ids={}, pos={}, parents={}",
                n, position_ids.len(), parents.len(),
            )));
        }
        let total = self.cache_len + n;
        if total > self.valid_mask.len() {
            let new_size = (total * 2).max(self.valid_mask.len() * 2);
            self.valid_mask.resize(new_size, 1);
        }
        for v in self.valid_mask[self.cache_len..total].iter_mut() {
            *v = 1;
        }
        let attn_f16 = build_tree_mask_f16(&self.valid_mask, self.cache_len, parents);

        let in_ids_name = self.inputs.get("input_ids").unwrap().clone();
        let attn_name = self.inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.inputs.get("position_ids").unwrap().clone();

        self.runtime
            .set_input(&in_ids_name, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&attn_name, ShimDType::F16, &[1, 1, n, total], &attn_f16)
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&pos_name, ShimDType::I64, &[1, n], &i64_to_bytes(position_ids))
            .map_err(map_ov_err)?;
        if self.has_beam {
            if let Some(beam_name) = self.inputs.get("beam_idx").cloned() {
                let bytes = 0i32.to_le_bytes().to_vec();
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
        let out_shape = if shape.len() == 3 {
            [shape[0], shape[1], shape[2]]
        } else if shape.len() == 2 {
            [1, shape[0], shape[1]]
        } else {
            [1, 1, floats.len()]
        };
        Ok((floats, out_shape))
    }

    /// Mirror of `DistributedMaskedReq::confirm_tree_path` for the local
    /// draft. Invalidate non-winning entries; advance logical_pos by the
    /// accepted path length.
    pub fn confirm_tree_path(&mut self, tree_size: usize, accepted_path_offsets: &[usize]) {
        let tree_base = self.cache_len.saturating_sub(tree_size);
        for i in tree_base..self.cache_len {
            self.valid_mask[i] = 0;
        }
        for &off in accepted_path_offsets {
            self.valid_mask[tree_base + off] = 1;
        }
        self.logical_pos += accepted_path_offsets.len();
    }

    /// Feed a `[L_i, R_i]` pair simultaneously: one batched forward call
    /// that returns logits at both positions. Each token sees its own
    /// chain's prior entries (since `round_base`) plus all valid past
    /// cache entries before `round_base`. Requires v6 IR.
    ///
    /// `position_l`, `position_r` are the absolute logical positions of
    /// L_i and R_i (siblings — same value).
    /// `round_base` = cache_len at the start of THIS round (before any
    /// pair feeds for this round). All entries in [round_base, cache_len)
    /// are this-round chain entries with owners from `chain_owner`.
    /// `chain_owner[k]` = 0 if cache slot `round_base + k` is on LEFT, 1 if RIGHT.
    pub fn feed_pair(
        &mut self,
        l_token: i64,
        r_token: i64,
        position_l: i64,
        position_r: i64,
        round_base: usize,
        chain_owner: &[u8],
    ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        if !self.is_v6 {
            return Err(EngineError::Backend("MaskedReq.feed_pair requires v6".into()));
        }
        let total = self.cache_len + 2;
        if total > self.valid_mask.len() {
            let new_size = (total * 2).max(self.valid_mask.len() * 2);
            self.valid_mask.resize(new_size, 1);
        }
        // Mark the two new slots provisionally valid (they get invalidated
        // when the round ends if their chain loses).
        self.valid_mask[self.cache_len] = 1;
        self.valid_mask[self.cache_len + 1] = 1;

        let attn_f16 = build_pair_mask_f16(&self.valid_mask, self.cache_len, round_base, chain_owner);
        let in_ids_name = self.inputs.get("input_ids").unwrap().clone();
        let attn_name = self.inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.inputs.get("position_ids").unwrap().clone();

        let pair_ids = [l_token, r_token];
        let pair_pos = [position_l, position_r];
        self.runtime
            .set_input(&in_ids_name, ShimDType::I64, &[1, 2], &i64_to_bytes(&pair_ids))
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&attn_name, ShimDType::F16, &[1, 1, 2, total], &attn_f16)
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&pos_name, ShimDType::I64, &[1, 2], &i64_to_bytes(&pair_pos))
            .map_err(map_ov_err)?;
        if self.has_beam {
            if let Some(beam_name) = self.inputs.get("beam_idx").cloned() {
                let bytes = 0i32.to_le_bytes().to_vec();
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
        // Output shape should be [1, 2, vocab].
        let vocab = if shape.len() == 3 {
            shape[2]
        } else {
            floats.len() / 2
        };
        let l_logit = floats[..vocab].to_vec();
        let r_logit = floats[vocab..2 * vocab].to_vec();
        self.cache_len += 2;
        Ok((l_logit, r_logit))
    }
}

// -------- DistributedMaskedReq (driver-side wrapper for multi-stage target) --------

/// Handle to a target.feed network round-trip in flight. Created by
/// `feed_send_async`, awaited by `feed_recv_async`. Drop-safe — if dropped
/// without await, the spawned task is cancelled (`tokio::task::JoinHandle`
/// behavior).
pub struct TargetSendHandle {
    join: tokio::task::JoinHandle<
        Result<(Vec<f32>, [usize; 3]), tahoma_transport::TransportError>,
    >,
}

pub struct DistributedMaskedReq {
    stage0: OvRuntime,
    stage0_inputs: std::collections::HashMap<String, String>,
    downstream: Arc<tokio::sync::Mutex<ActivationClient>>,
    runtime_handle: tokio::runtime::Handle,
    valid_mask: Vec<i64>,
    cache_len: usize,
    logical_pos: usize,
    /// v6 mode: stage_0 IR expects a 4D additive f16 attention_mask
    /// `[1, 1, q, total]` instead of the v5 2D i64 pad mask
    /// `[1, total]`. Driver builds the chain mask client-side and sends
    /// it via `FrameKind::ForwardV6` so charlie's stage_1 (also v6) can
    /// receive it identically.
    pub is_v6: bool,
    // Per-task accumulators — `spec_decode_greedy` resets these at task
    // start so the final `spec_decode timing` line attributes target
    // time correctly between alpha-side compute and wire (charlie+net).
    pub t_alpha_setup: std::time::Duration,
    pub t_alpha_infer: std::time::Duration,
    pub t_alpha_output: std::time::Duration,
    pub t_wire: std::time::Duration,
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
            is_v6: false,
            t_alpha_setup: std::time::Duration::ZERO,
            t_alpha_infer: std::time::Duration::ZERO,
            t_alpha_output: std::time::Duration::ZERO,
            t_wire: std::time::Duration::ZERO,
        })
    }

    pub fn set_v6(&mut self, v: bool) {
        self.is_v6 = v;
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

    /// Async-split: does sync alpha-side stage_0 work then spawns the network
    /// round-trip as a tokio task. Returns a handle the caller awaits via
    /// `feed_recv_async`. Lets the caller do other alpha work (next-round
    /// drafts, post-round draft.feed) DURING the charlie wait window.
    /// State (cache_len, logical_pos) is updated on send so back-to-back
    /// `feed_send_async` calls stay consistent.
    pub fn feed_send_async(
        &mut self,
        input_ids: &[i64],
    ) -> Result<TargetSendHandle, EngineError> {
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

        // v6 chain mask (built only if needed).
        let attn_f16: Vec<u8> = if self.is_v6 {
            build_chain_mask_f16(&self.valid_mask, self.cache_len, n)
        } else {
            Vec::new()
        };

        // Run stage 0 locally (sync — alpha GPU compute).
        let in_ids = self.stage0_inputs.get("input_ids").unwrap().clone();
        let attn_name = self.stage0_inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.stage0_inputs.get("position_ids").unwrap().clone();
        let beam_name = self.stage0_inputs.get("beam_idx").unwrap().clone();
        let _ts = std::time::Instant::now();
        self.stage0
            .set_input(&in_ids, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        if self.is_v6 {
            // 4D additive mask [1, 1, n, total] f16. Wire layout = byte-identical to [1, n, total].
            self.stage0
                .set_input(&attn_name, ShimDType::F16, &[1, 1, n, total], &attn_f16)
                .map_err(map_ov_err)?;
        } else {
            self.stage0
                .set_input(&attn_name, ShimDType::I64, &[1, total], &i64_to_bytes(&attn))
                .map_err(map_ov_err)?;
        }
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
        let is_v6 = self.is_v6;
        let pos_clone = pos.clone();
        let attn_f16_clone = attn_f16;
        let n_clone = n;
        let join = self.runtime_handle.spawn(async move {
            let mut g = downstream.lock().await;
            if is_v6 {
                // FrameKind::ForwardV6 — kind only (no logical_pos_start; positions are explicit).
                let kind = (FrameKind::ForwardV6 as u32).to_be_bytes();
                g.send_raw(&kind).await?;
                let attn_tensor = WireTensor::new(
                    WireDType::F16,
                    [1, n_clone as u32, total_clone as u32],
                    attn_f16_clone,
                );
                g.send(&attn_tensor).await?;
                let pos_tensor = WireTensor::new(
                    WireDType::I64,
                    [1, 1, n_clone as u32],
                    i64_to_bytes(&pos_clone),
                );
                g.send(&pos_tensor).await?;
                let hidden_tensor = WireTensor::new(WireDType::F16, hidden_shape_wire, hidden_f16);
                g.send(&hidden_tensor).await?;
            } else {
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
            }
            let kind_bytes = g.recv_raw(4).await?;
            let kind = u32::from_be_bytes([
                kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3],
            ]);
            if kind != FrameKind::LogitsResponse as u32 {
                return Err(tahoma_transport::TransportError::SocketClosed);
            }
            let (t, _) = g.recv().await?;
            let logits_f32 = match t.dtype {
                WireDType::F32 => t.data
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
        });
        Ok(TargetSendHandle { join })
    }

    /// Tree-spec variant of `feed_send_async`. Caller provides:
    /// - `input_ids`: flat sequence of `n` drafted tokens (DFS order over the tree)
    /// - `position_ids`: per-token absolute logical position (siblings share)
    /// - `parents`: per-token parent index in the same flat sequence (-1 = root)
    ///
    /// Builds a tree-topology mask + sets stage_0 inputs, spawns the network
    /// task. `cache_len` and `logical_pos` advance by `n` (we keep all tree
    /// entries in the KV cache; subsequent rounds rewind via valid_mask).
    pub fn feed_tree_send_async(
        &mut self,
        input_ids: &[i64],
        position_ids: &[i64],
        parents: &[i32],
    ) -> Result<TargetSendHandle, EngineError> {
        if !self.is_v6 {
            return Err(EngineError::Backend(
                "feed_tree_send_async requires v6 shards".into(),
            ));
        }
        let n = input_ids.len();
        if position_ids.len() != n || parents.len() != n {
            return Err(EngineError::Backend(format!(
                "tree feed length mismatch: ids={}, pos={}, parents={}",
                n, position_ids.len(), parents.len(),
            )));
        }
        let total = self.cache_len + n;
        if total > self.valid_mask.len() {
            let new_size = (total * 2).max(self.valid_mask.len() * 2);
            self.valid_mask.resize(new_size, 1);
        }
        // Mark the new positions as valid (caller calls `prune_tree_kv` after
        // verify to mark rejected branches as invalid).
        for v in self.valid_mask[self.cache_len..total].iter_mut() {
            *v = 1;
        }
        let attn_f16 = build_tree_mask_f16(&self.valid_mask, self.cache_len, parents);

        let in_ids = self.stage0_inputs.get("input_ids").unwrap().clone();
        let attn_name = self.stage0_inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.stage0_inputs.get("position_ids").unwrap().clone();
        let beam_name = self.stage0_inputs.get("beam_idx").unwrap().clone();

        let _ts = std::time::Instant::now();
        self.stage0
            .set_input(&in_ids, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(&attn_name, ShimDType::F16, &[1, 1, n, total], &attn_f16)
            .map_err(map_ov_err)?;
        self.stage0
            .set_input(&pos_name, ShimDType::I64, &[1, n], &i64_to_bytes(position_ids))
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

        // Update local state. Tree feed grows cache_len by n (all branches
        // get KV entries) — caller invalidates rejected entries afterwards.
        self.cache_len += n;
        // logical_pos advances by the longest accepted path length, set
        // by the caller via `confirm_tree_path`.

        self.t_alpha_setup += std::time::Duration::from_micros(setup_us as u64);
        self.t_alpha_infer += std::time::Duration::from_micros(infer_us as u64);
        self.t_alpha_output += std::time::Duration::from_micros(output_us as u64);

        let downstream = self.downstream.clone();
        let pos_clone = position_ids.to_vec();
        let n_clone = n;
        let total_clone = total;
        let join = self.runtime_handle.spawn(async move {
            let mut g = downstream.lock().await;
            let kind = (FrameKind::ForwardV6 as u32).to_be_bytes();
            g.send_raw(&kind).await?;
            let attn_tensor = WireTensor::new(
                WireDType::F16,
                [1, n_clone as u32, total_clone as u32],
                attn_f16,
            );
            g.send(&attn_tensor).await?;
            let pos_tensor = WireTensor::new(
                WireDType::I64,
                [1, 1, n_clone as u32],
                i64_to_bytes(&pos_clone),
            );
            g.send(&pos_tensor).await?;
            let hidden_tensor = WireTensor::new(WireDType::F16, hidden_shape_wire, hidden_f16);
            g.send(&hidden_tensor).await?;
            let kind_bytes = g.recv_raw(4).await?;
            let kind = u32::from_be_bytes([
                kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3],
            ]);
            if kind != FrameKind::LogitsResponse as u32 {
                return Err(tahoma_transport::TransportError::SocketClosed);
            }
            let (t, _) = g.recv().await?;
            let logits_f32 = match t.dtype {
                WireDType::F32 => t.data
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
        });
        Ok(TargetSendHandle { join })
    }

    /// After a tree round, mark the rejected (non-winning-path) KV entries
    /// as invalid so subsequent forwards mask them out. Walk the accepted
    /// path indices from the verifier; everything else added in the last
    /// `tree_size` cache entries is invalidated.
    ///
    /// `tree_size`: number of tokens just inserted (== `n` from the tree feed).
    /// `accepted_path_offsets`: indices INTO the tree (0..tree_size) that are
    /// the accepted ancestor chain. Logical position is advanced by the length
    /// of this path.
    pub fn confirm_tree_path(&mut self, tree_size: usize, accepted_path_offsets: &[usize]) {
        let tree_base = self.cache_len.saturating_sub(tree_size);
        // Mark everything as invalid first, then re-validate the path.
        for i in tree_base..self.cache_len {
            self.valid_mask[i] = 0;
        }
        for &off in accepted_path_offsets {
            self.valid_mask[tree_base + off] = 1;
        }
        self.logical_pos += accepted_path_offsets.len();
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
                .map_err(|e| tahoma_transport::TransportError::Io(std::io::Error::other(e)))?
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

        let attn_f16: Vec<u8> = if self.is_v6 {
            build_chain_mask_f16(&self.valid_mask, self.cache_len, n)
        } else {
            Vec::new()
        };

        // Run stage 0 locally.
        let in_ids = self.stage0_inputs.get("input_ids").unwrap().clone();
        let attn_name = self.stage0_inputs.get("attention_mask").unwrap().clone();
        let pos_name = self.stage0_inputs.get("position_ids").unwrap().clone();
        let beam_name = self.stage0_inputs.get("beam_idx").unwrap().clone();
        let _ts = std::time::Instant::now();
        self.stage0
            .set_input(&in_ids, ShimDType::I64, &[1, n], &i64_to_bytes(input_ids))
            .map_err(map_ov_err)?;
        if self.is_v6 {
            self.stage0
                .set_input(&attn_name, ShimDType::F16, &[1, 1, n, total], &attn_f16)
                .map_err(map_ov_err)?;
        } else {
            self.stage0
                .set_input(
                    &attn_name,
                    ShimDType::I64,
                    &[1, total],
                    &i64_to_bytes(&attn),
                )
                .map_err(map_ov_err)?;
        }
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
        let attn_f16_clone = attn_f16;
        let pos_clone = pos.clone();
        let n_clone = n;
        let is_v6 = self.is_v6;
        let _ts = std::time::Instant::now();
        let mut t_send = std::time::Duration::ZERO;
        let mut t_recv = std::time::Duration::ZERO;
        let mut t_lock = std::time::Duration::ZERO;
        let _ts_lock = std::time::Instant::now();
        let (logits, _logits_shape, send_d, recv_d) = run_async(&self.runtime_handle, async move {
            let mut g = downstream.lock().await;
            // Build either Forward (v5) or ForwardV6 (v6) frame.
            let _ts_send = std::time::Instant::now();
            if is_v6 {
                let kind = (FrameKind::ForwardV6 as u32).to_be_bytes();
                g.send_raw(&kind).await?;
                let attn_tensor = WireTensor::new(
                    WireDType::F16,
                    [1, n_clone as u32, total as u32],
                    attn_f16_clone,
                );
                g.send(&attn_tensor).await?;
                let pos_tensor = WireTensor::new(
                    WireDType::I64,
                    [1, 1, n_clone as u32],
                    i64_to_bytes(&pos_clone),
                );
                g.send(&pos_tensor).await?;
                let hidden_tensor = WireTensor::new(WireDType::F16, hidden_shape_wire, hidden_f16);
                g.send(&hidden_tensor).await?;
            } else {
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
            }
            let send_dur = _ts_send.elapsed();

            // Read LOGITS_RESPONSE.
            let _ts_kind = std::time::Instant::now();
            let kind_bytes = g.recv_raw(4).await?;
            let recv_kind_dur = _ts_kind.elapsed();
            let kind =
                u32::from_be_bytes([kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3]]);
            if kind != FrameKind::LogitsResponse as u32 {
                return Err(tahoma_transport::TransportError::SocketClosed);
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
                _ => return Err(tahoma_transport::TransportError::SocketClosed),
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
            Ok::<_, tahoma_transport::TransportError>((
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
        // Send target verify (alpha stage_0 sync, then spawn network task).
        // While charlie computes stage_1 (~43ms), do the SPECULATIVE first
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
        // charlie stage_1 + recv).
        target_alpha_setup_ms = target.t_alpha_setup.as_millis() as u64,
        target_alpha_infer_ms = target.t_alpha_infer.as_millis() as u64,
        target_alpha_output_ms = target.t_alpha_output.as_millis() as u64,
        target_wire_ms = target.t_wire.as_millis() as u64,
        "spec_decode timing"
    );
    Ok(out)
}

/// Tree-spec greedy decode: width-2-at-root topology.
///
/// For each round we maintain TWO parallel chains of length K, sharing
/// the previous correction as their root:
///   - Left chain  = draft top-1 chain of length K starting from root
///   - Right chain = draft top-2 of root, then top-1 chain of length K-1 from there
///
/// Tree layout (flat, length 2K, parents indexed into the same flat array):
///   index   0    1    2  ...  K-1   K     K+1   ... 2K-1
///   token  L0   L1   L2  ...  L_{K-1}   R0   R1   ... R_{K-2}
///   parent -1   0    1   ...  K-2   -1    K     ...  2K-3
///   pos    P    P+1  P+2 ...  P+K-1  P    P+1   ...  P+K-2
///
/// Both chains share absolute logical positions [P .. P+K-1] but with
/// the tree mask isolating sibling subtrees.
///
/// `tree_preset == 1` enables this topology. Future presets can vary depth
/// or fan-out without changing the engine API.
pub fn spec_decode_greedy_tree(
    target: &mut DistributedMaskedReq,
    draft: &mut MaskedReq,
    prompt_ids: &[i64],
    max_tokens: usize,
    k: usize,
    tree_preset: u32,
    stats: &mut SpecDecodeStats,
) -> Result<Vec<i64>, EngineError> {
    if !target.is_v6 {
        return Err(EngineError::Backend(
            "spec_decode_greedy_tree requires v6 target shards (4D additive mask)".into(),
        ));
    }
    // Preset 99 = chain-only via tree code path (debug: should match chain spec).
    let skip_right_chain = tree_preset == 99;
    // Preset 2 = parallel-draft tree (feed_pair); requires v6 draft.
    let parallel_draft = tree_preset == 2;
    if parallel_draft && !draft.is_v6 {
        return Err(EngineError::Backend(
            "tree_preset=2 (parallel draft) requires v6 draft model".into(),
        ));
    }
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

    // Prefill prompt on both target and draft (chain mode — uses 4D mask
    // built client-side via the same code path as ordinary chain spec on v6).
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

    // Bootstrap d_last_logit by feeding `first` to the draft.
    let mut prev_correction = first;
    let _ts = std::time::Instant::now();
    let (d_logits, d_shape) = draft.feed(&[first])?;
    t_draft += _ts.elapsed();
    let dv = d_shape[2];
    let mut d_last_logit: Vec<f32> = d_logits[d_logits.len() - dv..].to_vec();

    while out.len() < max_tokens {
        stats.n_steps += 1;

        // -------- Build LEFT and RIGHT chains --------
        let (left, right, left_feeds, right_feeds) = if parallel_draft && !skip_right_chain {
            // Parallel build via feed_pair: K-1 batched 2-token forward calls.
            // L_0 = top-1 of d_last_logit (free), R_0 = top-2 (free).
            // chain_owner records which chain each cache slot belongs to so
            // the pair-mask isolates siblings.
            let mut left = vec![argmax(&d_last_logit) as i64];
            let mut right = vec![top2(&d_last_logit) as i64];
            let p_root_draft = draft.logical_pos() as i64;
            let round_base = draft.cache_len();
            let mut owners: Vec<u8> = Vec::with_capacity(2 * (k - 1));
            for i in 1..k {
                let l_token = *left.last().unwrap();
                let r_token = *right.last().unwrap();
                // Both new entries land at logical position p_root_draft + i.
                let pos = p_root_draft + i as i64;
                let _ts = std::time::Instant::now();
                let (l_logit, r_logit) = draft.feed_pair(
                    l_token,
                    r_token,
                    pos,
                    pos,
                    round_base,
                    &owners,
                )?;
                t_draft += _ts.elapsed();
                // After this feed: 2 new cache entries appended. Owner 0 = L, 1 = R.
                owners.push(0);
                owners.push(1);
                left.push(argmax(&l_logit) as i64);
                right.push(argmax(&r_logit) as i64);
            }
            // Build phase added 2*(k-1) entries; owners describes them.
            // For the catchup logic, treat as: left "feeds" = right "feeds" = k-1
            // (each chain extended k-1 times).
            (left, right, k - 1, k - 1)
        } else {
            // Sequential build: LEFT chain first, then RIGHT.
            let mut left: Vec<i64> = vec![argmax(&d_last_logit) as i64];
            for _i in 1..k {
                let prev = *left.last().unwrap();
                let _ts = std::time::Instant::now();
                let (l, sh) = draft.feed(&[prev])?;
                t_draft += _ts.elapsed();
                let dv2 = sh[2];
                left.push(argmax(&l[l.len() - dv2..]) as i64);
            }
            let left_feeds = k - 1;

            let right: Vec<i64> = if skip_right_chain {
                Vec::new()
            } else {
                draft.rewind(left_feeds);
                let alt_root = top2(&d_last_logit) as i64;
                let mut right: Vec<i64> = vec![alt_root];
                let _ts = std::time::Instant::now();
                let (l, sh) = draft.feed(&[alt_root])?;
                t_draft += _ts.elapsed();
                let dv_r = sh[2];
                let mut r_last: Vec<f32> = l[l.len() - dv_r..].to_vec();
                for _i in 1..k {
                    let next = argmax(&r_last) as i64;
                    right.push(next);
                    let _ts = std::time::Instant::now();
                    let (l2, sh2) = draft.feed(&[next])?;
                    t_draft += _ts.elapsed();
                    let dv_r2 = sh2[2];
                    r_last = l2[l2.len() - dv_r2..].to_vec();
                }
                right
            };
            let right_feeds = if skip_right_chain { 0 } else { k };
            (left, right, left_feeds, right_feeds)
        };
        let drafted = left.len() + right.len();
        stats.total_drafts += drafted as u32;

        // -------- Build tree flat sequence --------
        // Token order: [prev_correction, L0, L1, ..., L_{k-1}, R0, R1, ..., R_{k-1}]
        // We send the K+K = 2k drafted tokens (NOT prev_correction — it's already
        // in the target KV cache from the previous round).
        // Parents (relative to start of drafted chunk):
        //   L0: -1 (root = prev_correction in cache)
        //   Li: i-1 for i in [1, k-1]
        //   R0: -1 (root = prev_correction in cache)
        //   Ri: k + i - 1 for i in [1, k-1]
        let mut tree_ids: Vec<i64> = Vec::with_capacity(2 * k);
        let mut tree_pos: Vec<i64> = Vec::with_capacity(2 * k);
        let mut tree_parents: Vec<i32> = Vec::with_capacity(2 * k);
        // We need to send `prev_correction` as the FIRST tree token because
        // in chain spec, target hasn't seen the most recent accepted token yet
        // (it's only in our `out` buffer). Same convention as chain spec which
        // puts `prev_correction` at index 0 of `verify`.
        tree_ids.push(prev_correction);
        tree_pos.push(target.logical_pos as i64);
        tree_parents.push(-1);

        let p_root = target.logical_pos as i64; // this is where prev_correction lands
        // Left chain
        for (i, &t) in left.iter().enumerate() {
            tree_ids.push(t);
            tree_pos.push(p_root + 1 + i as i64);
            tree_parents.push(i as i32); // parent = previous tree index (0 for L0, then 1, 2, ...)
        }
        // Right chain
        let right_base = 1 + left.len(); // tree index of R0
        for (i, &t) in right.iter().enumerate() {
            tree_ids.push(t);
            tree_pos.push(p_root + 1 + i as i64);
            if i == 0 {
                tree_parents.push(0); // R0's parent = prev_correction (tree index 0)
            } else {
                tree_parents.push((right_base + i - 1) as i32);
            }
        }

        // -------- Verify (async send + speculative draft work during wait) --------
        let _ts = std::time::Instant::now();
        let target_handle =
            target.feed_tree_send_async(&tree_ids, &tree_pos, &tree_parents)?;

        // We can't easily speculate on the tree path here (we don't know which
        // chain wins), so just block on target now. Future enhancement: do a
        // post-round prefetch on whichever chain has higher cumulative draft
        // confidence.
        let (t_logits, t_shape) = target.feed_recv_async(target_handle)?;
        t_target += _ts.elapsed();

        let v = t_shape[2];
        let mut t_greedy = Vec::with_capacity(tree_ids.len());
        for i in 0..tree_ids.len() {
            let row = &t_logits[i * v..(i + 1) * v];
            t_greedy.push(argmax(row) as i64);
        }

        // -------- Walk both chains, pick the longer accepted path --------
        // For the LEFT chain, walk: at tree index i (1..=k), check if
        // t_greedy[i-1] (target's prediction AFTER the parent at index i-1)
        // matches drafts L_{i-1}.
        let mut left_accept = 0usize;
        for i in 0..left.len() {
            // Parent is at tree index i (root=0 for i=0, then i for i>=1).
            let parent_tree_idx = i; // parent at tree index i (which is root for i=0, L_{i-1} for i>0)
            if t_greedy[parent_tree_idx] == left[i] {
                left_accept += 1;
            } else {
                break;
            }
        }
        let mut right_accept = 0usize;
        for i in 0..right.len() {
            let parent_tree_idx = if i == 0 { 0 } else { right_base + i - 1 };
            if t_greedy[parent_tree_idx] == right[i] {
                right_accept += 1;
            } else {
                break;
            }
        }

        // Decide winning path. Tie-break: prefer LEFT (the top-1 path).
        let (winning_chain_left, accepted, accepted_offsets, correction_idx) =
            if left_accept >= right_accept {
                let mut offsets = vec![0usize]; // tree index 0 = prev_correction
                for i in 0..left_accept {
                    offsets.push(1 + i);
                }
                // Correction = target's argmax AFTER the last accepted left token
                let corr_parent_tree_idx = if left_accept < left.len() {
                    left_accept // tree index of the rejected token's parent
                } else {
                    left.len() // tree index of L_{k-1}, target's prediction after it
                };
                (true, left_accept, offsets, corr_parent_tree_idx)
            } else {
                let mut offsets = vec![0usize];
                for i in 0..right_accept {
                    offsets.push(right_base + i);
                }
                let corr_parent_tree_idx = if right_accept < right.len() {
                    if right_accept == 0 {
                        0
                    } else {
                        right_base + right_accept - 1
                    }
                } else {
                    right_base + right.len() - 1
                };
                (false, right_accept, offsets, corr_parent_tree_idx)
            };

        let correction = t_greedy[correction_idx];
        stats.total_accepted += accepted as u32;

        let chosen_drafts: &[i64] = if winning_chain_left {
            &left[..accepted]
        } else {
            &right[..accepted]
        };
        for &t in chosen_drafts.iter().chain(std::iter::once(&correction)) {
            if out.len() >= max_tokens {
                break;
            }
            out.push(t);
        }

        // -------- Confirm tree path on target (mark non-winning entries invalid) --------
        // accepted_offsets are tree-relative indices [0, 1+left_accept] etc.
        // confirm_tree_path expects offsets into the tree we just sent (size 2k+1).
        target.confirm_tree_path(tree_ids.len(), &accepted_offsets);

        // -------- Reconcile draft state --------
        // Reset the draft cache back to round-base state. Use `invalidate_recent`
        // (which doesn't touch logical_pos) for the parallel_draft path because
        // feed_pair adds 2 cache slots per logical-position step.
        if parallel_draft {
            // 2 cache entries per pair iteration × (k-1) iterations.
            draft.invalidate_recent(left_feeds + right_feeds);
        } else if skip_right_chain {
            draft.rewind(left_feeds);
        } else {
            // Sequential: after right build, draft has left_feeds (invalidated
            // by earlier rewind) + right_feeds (live). Only rewind right_feeds.
            draft.rewind(right_feeds);
        }
        let mut catchup: Vec<i64> = Vec::with_capacity(accepted + 1);
        catchup.extend_from_slice(chosen_drafts);
        catchup.push(correction);
        let _ts = std::time::Instant::now();
        let (l, sh) = draft.feed(&catchup)?;
        t_draft += _ts.elapsed();
        let dv = sh[2];
        d_last_logit = l[l.len() - dv..].to_vec();
        if stats.n_steps <= 3 {
            tracing::info!(
                step = stats.n_steps,
                left = ?left,
                right = ?right,
                t_greedy_first = t_greedy.first().copied(),
                left_accept,
                right_accept,
                winning_left = winning_chain_left,
                accepted,
                correction,
                "tree-spec round trace"
            );
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
        target_alpha_setup_ms = target.t_alpha_setup.as_millis() as u64,
        target_alpha_infer_ms = target.t_alpha_infer.as_millis() as u64,
        target_alpha_output_ms = target.t_alpha_output.as_millis() as u64,
        target_wire_ms = target.t_wire.as_millis() as u64,
        "spec_decode_tree timing"
    );

    Ok(out)
}

/// Return the index of the second-largest element. Ties broken in favor
/// of the lower index. Used by tree-spec to pick the alt root.
fn top2(slice: &[f32]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = (0usize, f32::NEG_INFINITY);
    for (i, &v) in slice.iter().enumerate() {
        if v > best.1 {
            second = best;
            best = (i, v);
        } else if v > second.1 {
            second = (i, v);
        }
    }
    second.0
}

// -------- Driver-side Engine + Builder --------

pub struct OvDistSpecEngine {
    target: DistributedMaskedReq,
    draft: MaskedReq,
    tokenizer: Arc<Tokenizer>,
    eos_token_id: Option<u32>,
    k: usize,
    /// 0 = chain spec (existing K-chain). 1+ = tree-spec preset; see `TreePreset`.
    tree_preset: u32,
    pending: Vec<GenerationTask>,
    active: Option<(GenerationTask, Vec<i64>, String, SpecDecodeStats)>,
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
        // Warmup uses chain spec — tree spec adds overhead we don't need
        // for the cold-init pre-pay; the chain warmup also primes the
        // 4D-mask path on v6 stages because chain spec on v6 still uses
        // 4D additive mask (built client-side), just with the causal
        // shape.
        match spec_decode_greedy(
            &mut self.target,
            &mut self.draft,
            &prompt_ids,
            2,
            self.k,
            &mut stats,
        ) {
            Ok(_) => info!(k = self.k, tree = self.tree_preset, "ov-dist-spec warmup ok"),
            Err(err) => warn!(error = %err, "ov-dist-spec warmup failed"),
        }
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        if self.pending.iter().any(|t| t.task_id == task.task_id)
            || self
                .active
                .as_ref()
                .is_some_and(|(t, ..)| t.task_id == task.task_id)
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
            let result = if self.tree_preset > 0 {
                spec_decode_greedy_tree(
                    &mut self.target,
                    &mut self.draft,
                    &prompt_ids,
                    max_tokens,
                    self.k,
                    self.tree_preset,
                    &mut stats,
                )
            } else {
                spec_decode_greedy(
                    &mut self.target,
                    &mut self.draft,
                    &prompt_ids,
                    max_tokens,
                    self.k,
                    &mut stats,
                )
            };
            let task_id = task.task_id.clone();
            match result {
                Err(e) => {
                    warn!(task = %task.task_id, error = %e, "ov-dist-spec failed");
                    let chunk = Chunk::final_marker(task.task_id, "");
                    return vec![(task_id, chunk)];
                }
                Ok(tokens) => {
                    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                    let text = self.tokenizer.decode(&ids, true).unwrap_or_default();
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
    /// Device for the draft model. Defaults to `device` if unset.
    /// Setting CPU here lets the draft run concurrent with target stage_0
    /// on the GPU — frees the GPU during draft compute.
    pub draft_device: String,
    pub k: u32,
    pub cache_dir: Option<String>,
    pub kv_cache_precision: Option<String>,
    pub dyn_quant_group: Option<String>,
    /// Tree-spec topology preset (0 = chain spec, 1 = width-2 at root, 2 = width-2 at root + depth 1).
    /// Requires v6 shards. See `TreePreset`.
    pub tree_preset: u32,
    stage0: Option<OvRuntime>,
    draft: Option<OvRuntime>,
    tokenizer: Option<Arc<Tokenizer>>,
    eos_token_id: Option<u32>,
    is_v6: bool,
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
            tree_preset: 0,
            stage0: None,
            draft: None,
            tokenizer: None,
            eos_token_id: None,
            is_v6: false,
            downstream: None,
        }
    }

    pub fn with_draft_device(mut self, device: impl Into<String>) -> Self {
        self.draft_device = device.into();
        self
    }

    pub fn with_tree_preset(mut self, preset: u32) -> Self {
        self.tree_preset = preset;
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
                "driver requires v5 or newer shards (canonical inputs)".into(),
            ));
        }
        // v6 stage_0 has a 4D additive f16 attention_mask. Detect from the
        // export_version string the export script writes into stage_config.json.
        self.is_v6 = stage_cfg
            .export_version
            .as_deref()
            .unwrap_or_default()
            .starts_with("v6")
            || pipeline_cfg
                .export_version
                .as_deref()
                .unwrap_or_default()
                .starts_with("v6");
        if self.tree_preset > 0 && !self.is_v6 {
            return Err(EngineError::ShardRejected(
                "tree-spec preset requires v6 shards (4D mask)".into(),
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
            self.eos_token_id = lookup_eos(&self.pipeline_dir.join("tokenizer"))
                .or_else(|| lookup_eos(&self.pipeline_dir));
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
        let mut target =
            DistributedMaskedReq::new(stage0, downstream, tokio::runtime::Handle::current())?;
        target.set_v6(self.is_v6);
        // Detect whether the draft was exported with a v6 4D additive mask.
        // We check both `stage_0/stage_config.json` and `stage_config.json`
        // to handle single-stage and 1-stage-as-stage-0 layouts.
        let draft_path = std::path::Path::new(&self.draft_model_path);
        let draft_v6 = {
            let cfg_paths = [
                draft_path.join("stage_config.json"),
                draft_path.join("stage_0").join("stage_config.json"),
            ];
            cfg_paths.iter().any(|p| {
                std::fs::read_to_string(p)
                    .ok()
                    .and_then(|s| serde_json::from_str::<StageConfig>(&s).ok())
                    .and_then(|c| c.export_version)
                    .map(|v| v.starts_with("v6"))
                    .unwrap_or(false)
            })
        };
        let mut masked_draft = MaskedReq::new(draft)?;
        masked_draft.set_v6(draft_v6);
        if draft_v6 {
            info!("draft model detected as v6 (4D additive mask)");
        }
        Ok(Box::new(OvDistSpecEngine {
            target,
            draft: masked_draft,
            tokenizer,
            eos_token_id: self.eos_token_id,
            k: self.k as usize,
            tree_preset: self.tree_preset,
            pending: Vec::new(),
            active: None,
        }))
    }
}

// -------- Worker-side Engine + Builder --------

pub struct OvDistSpecWorkerEngine {
    is_last: bool,
    /// True when this stage's IR was exported with v6 4D additive f16 mask.
    /// Determines whether the worker accepts ForwardV6 frames vs Forward.
    pub is_v6: bool,
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

    fn submit(&mut self, _task: GenerationTask) -> EngineResult<()> {
        warn!("ov-dist-spec worker cannot accept tasks directly");
        Err(EngineError::Backend(
            "worker stage does not accept tasks directly".into(),
        ))
    }

    fn step(&mut self) -> Vec<(TaskId, Chunk)> {
        // RAII guard: marks the current thread as blocking-pool for
        // the duration of this step so run_async takes the cheap
        // bare-block_on path. The guard restores the previous flag
        // value on drop, preventing stale state if the spawn_blocking
        // pool ever migrates this thread to non-blocking work.
        let _guard = BlockingContextGuard::enter();
        let result = self.handle_one_frame();
        if let Err(e) = result {
            // Transport-closed errors signal the driver disconnected;
            // don't spam the log. Drop the upstream/downstream so the
            // next step exits the relay loop cleanly via NotConnected.
            let msg = e.to_string();
            if msg.contains("socket closed") || msg.contains("not connected") {
                warn!("ov-dist-spec worker: upstream closed, exiting");
                // Mark engine as drained by clearing connections.
                let _ = self.runtime_handle.clone().block_on(async {
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
            FrameKind::ForwardV6 => self.handle_forward_v6(upstream, downstream),
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
                        Ok::<_, tahoma_transport::TransportError>((
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

impl OvDistSpecWorkerEngine {
    /// Handle a ForwardV6 frame: 4D additive f16 attention_mask + custom
    /// position_ids + hidden_states. Different from `Forward` (v5) in that
    /// (a) the mask is f16 4D (built by the driver) and (b) position_ids
    /// are explicit on the wire (so tree-spec can pass non-monotonic positions).
    fn handle_forward_v6(
        &mut self,
        upstream: Arc<tokio::sync::Mutex<ActivationServer>>,
        downstream: Option<Arc<tokio::sync::Mutex<ActivationClient>>>,
    ) -> Result<(), EngineError> {
        let upstream2 = upstream.clone();
        let _ts_recv = std::time::Instant::now();
        let (attn_bytes, attn_shape, pos_bytes, pos_n, hidden_bytes, hidden_dtype, hidden_shape) =
            run_async(&self.runtime_handle, async move {
                let mut g = upstream2.lock().await;
                let (attn_t, _) = g.recv().await?;
                let (pos_t, _) = g.recv().await?;
                let (hidden_t, _) = g.recv().await?;
                Ok::<_, tahoma_transport::TransportError>((
                    attn_t.data,
                    [attn_t.shape[0] as usize, attn_t.shape[1] as usize, attn_t.shape[2] as usize],
                    pos_t.data,
                    pos_t.shape[2] as usize,
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
        let recv_us = _ts_recv.elapsed().as_micros();

        // attn shape on wire is [1, n, total], reshape to [1, 1, n, total] for the model.
        let n_query = attn_shape[1];
        let total_keys = attn_shape[2];

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
        // 4D mask: shape [1, 1, n, total] with same byte layout as wire [1, n, total].
        self.runtime
            .set_input(&in_attn, ShimDType::F16, &[1, 1, n_query, total_keys], &attn_bytes)
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_pos, ShimDType::I64, &[1, pos_n], &pos_bytes)
            .map_err(map_ov_err)?;
        self.runtime
            .set_input(&in_beam, ShimDType::I32, &[1], &0i32.to_le_bytes())
            .map_err(map_ov_err)?;
        let setup_us = _ts_setup.elapsed().as_micros();
        let _ts_infer = std::time::Instant::now();
        self.runtime.infer().map_err(map_ov_err)?;
        let infer_us = _ts_infer.elapsed().as_micros();
        let _ts_out = std::time::Instant::now();
        let (out_dtype, out_shape, out_bytes) = self.runtime.output(0).map_err(map_ov_err)?;
        let output_us = _ts_out.elapsed().as_micros();
        tracing::debug!(n = n_query, recv_us, setup_us, infer_us, output_us, "worker.frame_v6 timing");

        let out_f16_bytes = match out_dtype {
            ShimDType::F16 => out_bytes,
            ShimDType::F32 => {
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
            run_async(&self.runtime_handle, async move {
                let mut g = upstream.lock().await;
                g.send_raw(&(FrameKind::LogitsResponse as u32).to_be_bytes()).await?;
                let t = WireTensor::new(WireDType::F16, out_shape_wire, out_f16_bytes);
                g.send(&t).await
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        } else {
            // Relay ForwardV6 downstream, then forward LOGITS_RESPONSE upstream.
            let downstream =
                downstream.ok_or_else(|| EngineError::Backend("no downstream".into()))?;
            let upstream3 = upstream.clone();
            let attn_clone = attn_bytes;
            let pos_clone = pos_bytes;
            run_async(&self.runtime_handle, async move {
                let mut dg = downstream.lock().await;
                let kind = (FrameKind::ForwardV6 as u32).to_be_bytes();
                dg.send_raw(&kind).await?;
                let attn_t = WireTensor::new(
                    WireDType::F16,
                    [1, n_query as u32, total_keys as u32],
                    attn_clone,
                );
                dg.send(&attn_t).await?;
                let pos_t = WireTensor::new(
                    WireDType::I64,
                    [1, 1, pos_n as u32],
                    pos_clone,
                );
                dg.send(&pos_t).await?;
                let hidden_t = WireTensor::new(WireDType::F16, out_shape_wire, out_f16_bytes);
                dg.send(&hidden_t).await?;
                let kind_bytes = dg.recv_raw(4).await?;
                let kv = u32::from_be_bytes([
                    kind_bytes[0],
                    kind_bytes[1],
                    kind_bytes[2],
                    kind_bytes[3],
                ]);
                if kv != FrameKind::LogitsResponse as u32 {
                    return Err(tahoma_transport::TransportError::SocketClosed);
                }
                let (logits_t, _) = dg.recv().await?;
                let mut g = upstream3.lock().await;
                g.send_raw(&(FrameKind::LogitsResponse as u32).to_be_bytes()).await?;
                g.send(&logits_t).await?;
                Ok::<_, tahoma_transport::TransportError>(())
            })
            .map_err(|e| EngineError::Backend(e.to_string()))?;
        }
        Ok(())
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
    is_v6: bool,
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
            is_v6: false,
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
                "worker requires v5 or newer shards".into(),
            ));
        }
        self.is_v6 = stage_cfg
            .export_version
            .as_deref()
            .unwrap_or_default()
            .starts_with("v6");
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
            is_v6: self.is_v6,
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
