//! Wire protocol for pipeline-parallel sparse-MoE inference.
//!
//! Frames flow over the existing tahoma-transport TCP sockets. Each
//! frame begins with a 4-byte big-endian `FrameKind` code; some frames
//! carry length-prefixed tensors after the code via the standard
//! `ActivationServer::send` / `recv` path.
//!
//! Directions:
//! - `Forward` and `Reset` flow downstream (rank N → rank N+1).
//! - `Token` flows upstream (last rank → rank 0).
//!
//! Each rank holds at most one upstream socket (server-side, accepting
//! incoming connection from rank-1) and one downstream socket
//! (client-side, connected outbound to rank+1). The first rank has no
//! upstream; the last rank has no downstream.
//!
//! `Forward` is the workhorse: one per decoded token. Body:
//!   - 4 B big-endian u32 past_seq_len
//!   - 28 B SamplingConfig payload (see [`SamplingWire`]):
//!       - 4 B f32 BE temperature
//!       - 4 B f32 BE top_p
//!       - 4 B f32 BE repetition_penalty
//!       - 8 B u64 BE repetition_window
//!       - 8 B u64 BE seed (0 sentinel = no seed / use entropy)
//!   - F32 hidden tensor [1, 1, hidden_size]
//!
//! `Reset` clears KV state on downstream for a new generation. No body.
//!
//! `Token` returns one sampled token i64 to upstream after the last
//! rank runs head + sample. Body:
//!   - 8 B big-endian i64 token_id

use std::sync::Arc;
use std::time::Duration;

use tahoma_transport::{
    ActivationClient, ActivationServer, DType, Tensor, TransportError, TransportResult,
};
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::sampling::SamplingConfig;

/// 4-byte big-endian frame kinds. Chosen to be disjoint from
/// `dist_spec`'s `FrameKind` so a stray cross-engine connection would
/// fail fast on an unrecognized code.
///
/// Versioning convention: the low byte of `Forward` is bumped whenever
/// the Forward body layout changes in a wire-incompatible way. Old
/// peers reject the new code at `parse_kind` rather than mis-parsing
/// the body. `Reset` / `Token` have stable bodies and don't need
/// versioning.
///
/// History:
/// - 0x01 — past_seq_len + hidden tensor only (pre-PR #10)
/// - 0x02 — adds 28-byte SamplingConfig between past_seq_len and tensor (PR #10)
///
/// Heartbeat (iter 092):
/// - `HeartbeatPing` / `HeartbeatPong` carry an 8-byte BE u64 nonce so a
///   sender can match a pong to the ping that produced it (rejects
///   stale pongs after a transport retry, lets the orchestrator measure
///   per-RTT liveness, gives recovery code a "last good RTT" hook). The
///   nonce is opaque to the worker — it echoes whatever it received.
/// - Codes 0x40 / 0x41 are in the heartbeat namespace; future control-
///   plane frames (HeartbeatLost, OrchestratorRestart) reserve 0x42+.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Forward = 0x53_4D_45_02, // "SME\x02"
    Reset = 0x53_4D_45_10,   // "SME\x10"
    Token = 0x53_4D_45_20,   // "SME\x20"
    /// Liveness ping; body = 8 B BE u64 nonce. Sender → receiver.
    HeartbeatPing = 0x53_4D_45_40, // "SME\x40"
    /// Liveness pong; body = 8 B BE u64 nonce (echoes the ping). Receiver → sender.
    HeartbeatPong = 0x53_4D_45_41, // "SME\x41"
}

impl FrameKind {
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            x if x == FrameKind::Forward as u32 => Some(FrameKind::Forward),
            x if x == FrameKind::Reset as u32 => Some(FrameKind::Reset),
            x if x == FrameKind::Token as u32 => Some(FrameKind::Token),
            x if x == FrameKind::HeartbeatPing as u32 => Some(FrameKind::HeartbeatPing),
            x if x == FrameKind::HeartbeatPong as u32 => Some(FrameKind::HeartbeatPong),
            _ => None,
        }
    }
}

/// Read one frame-kind code from the wire. Returns `None` on a clean
/// socket close (peer closed during recv on no in-flight frame), which
/// callers should treat as end-of-task rather than an error.
pub async fn recv_kind_server(srv: &Mutex<ActivationServer>) -> TransportResult<Option<FrameKind>> {
    let raw = {
        let mut guard = srv.lock().await;
        guard.recv_raw(4).await
    };
    match raw {
        Ok(bytes) => parse_kind(&bytes).map(Some),
        Err(TransportError::SocketClosed) => Ok(None),
        Err(e) => Err(e),
    }
}

pub async fn recv_kind_client(cli: &Mutex<ActivationClient>) -> TransportResult<Option<FrameKind>> {
    let raw = {
        let mut guard = cli.lock().await;
        guard.recv_raw(4).await
    };
    match raw {
        Ok(bytes) => parse_kind(&bytes).map(Some),
        Err(TransportError::SocketClosed) => Ok(None),
        Err(e) => Err(e),
    }
}

fn parse_kind(bytes: &[u8]) -> TransportResult<FrameKind> {
    // `recv_raw(4)` always returns exactly 4 bytes on success.
    debug_assert_eq!(bytes.len(), 4);
    let code = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    FrameKind::from_code(code).ok_or_else(|| {
        TransportError::Io(std::io::Error::other(format!(
            "unknown sparse-MoE frame kind 0x{code:08x}; peer may be on an older Forward layout"
        )))
    })
}

/// Encode hidden state f32 values + shape into a wire tensor.
fn hidden_to_tensor(hidden_f32: &[f32], shape3: [u32; 3]) -> Tensor {
    let mut bytes = Vec::with_capacity(hidden_f32.len() * 4);
    for v in hidden_f32 {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Tensor::new(DType::F32, shape3, bytes)
}

fn tensor_to_hidden(t: &Tensor) -> TransportResult<(Vec<f32>, [u32; 3])> {
    if t.dtype != DType::F32 {
        return Err(TransportError::Io(std::io::Error::other(format!(
            "expected F32 hidden tensor, got {:?}",
            t.dtype
        ))));
    }
    let n = t.data.len() / 4;
    let mut out = Vec::with_capacity(n);
    for c in t.data.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok((out, t.shape))
}

/// On-wire encoding of `SamplingConfig`. Fixed 28 bytes BE; the `seed`
/// field uses 0 as the sentinel for "no seed" (the engine treats 0 as
/// invalid anyway — `init_rng` clamps to `max(seed, 1)`).
pub const SAMPLING_WIRE_BYTES: usize = 28;

pub fn encode_sampling(cfg: &SamplingConfig, out: &mut [u8; SAMPLING_WIRE_BYTES]) {
    out[0..4].copy_from_slice(&cfg.temperature.to_be_bytes());
    out[4..8].copy_from_slice(&cfg.top_p.to_be_bytes());
    out[8..12].copy_from_slice(&cfg.repetition_penalty.to_be_bytes());
    out[12..20].copy_from_slice(&(cfg.repetition_window as u64).to_be_bytes());
    let seed = cfg.seed.unwrap_or(0);
    out[20..28].copy_from_slice(&seed.to_be_bytes());
}

pub fn decode_sampling(bytes: &[u8; SAMPLING_WIRE_BYTES]) -> SamplingConfig {
    // Defensive: a malformed or out-of-version peer could send NaN /
    // negative / out-of-range values that would silently poison the
    // sampler (NaN temperature → 1/NaN → all logits NaN → argmax 0
    // every step). Clamp each f32 field to its valid domain and drop
    // NaN to the default.
    let temperature = sanitize_f32(
        f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        0.0,
        0.0,
    );
    let top_p = sanitize_f32(
        f32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        1.0,
        0.0,
    )
    .min(1.0);
    let repetition_penalty = sanitize_f32(
        f32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        1.0,
        0.0,
    );
    let rep_window = u64::from_be_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]) as usize;
    let seed_raw = u64::from_be_bytes([
        bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
    ]);
    SamplingConfig {
        temperature,
        top_p,
        repetition_penalty,
        repetition_window: rep_window,
        seed: if seed_raw == 0 { None } else { Some(seed_raw) },
    }
}

/// Replace NaN / -inf / values below `min` with `fallback`. Used by
/// `decode_sampling` to keep wire input from poisoning the sampler.
fn sanitize_f32(x: f32, fallback: f32, min: f32) -> f32 {
    if x.is_nan() || x < min {
        fallback
    } else {
        x
    }
}

/// Send a Forward frame downstream: kind + past_seq_len (u32 BE) + 28 B
/// SamplingConfig + hidden tensor.
pub async fn send_forward(
    cli: &Mutex<ActivationClient>,
    past_seq_len: u32,
    sampling: &SamplingConfig,
    hidden_f32: &[f32],
    hidden_shape: [u32; 3],
) -> TransportResult<()> {
    let mut header = [0u8; 8 + SAMPLING_WIRE_BYTES];
    header[0..4].copy_from_slice(&(FrameKind::Forward as u32).to_be_bytes());
    header[4..8].copy_from_slice(&past_seq_len.to_be_bytes());
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    encode_sampling(sampling, &mut sbytes);
    header[8..8 + SAMPLING_WIRE_BYTES].copy_from_slice(&sbytes);
    let tensor = hidden_to_tensor(hidden_f32, hidden_shape);
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    guard.send(&tensor).await?;
    Ok(())
}

/// Receive a Forward frame's body (the kind code has already been
/// consumed by `recv_kind_*`). Returns
/// `(past_seq_len, sampling, hidden_f32, shape)`.
pub async fn recv_forward_body_server(
    srv: &Mutex<ActivationServer>,
) -> TransportResult<(u32, SamplingConfig, Vec<f32>, [u32; 3])> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(4 + SAMPLING_WIRE_BYTES).await?;
    if raw.len() != 4 + SAMPLING_WIRE_BYTES {
        return Err(TransportError::SocketClosed);
    }
    let past_seq_len = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let mut sbytes = [0u8; SAMPLING_WIRE_BYTES];
    sbytes.copy_from_slice(&raw[4..4 + SAMPLING_WIRE_BYTES]);
    let sampling = decode_sampling(&sbytes);
    let (tensor, _) = guard.recv().await?;
    drop(guard);
    let (h, shape) = tensor_to_hidden(&tensor)?;
    Ok((past_seq_len, sampling, h, shape))
}

/// Send a Reset frame downstream — clears KV state for a new task.
pub async fn send_reset(cli: &Mutex<ActivationClient>) -> TransportResult<()> {
    let mut header = [0u8; 4];
    header[0..4].copy_from_slice(&(FrameKind::Reset as u32).to_be_bytes());
    let mut guard = cli.lock().await;
    guard.send_raw(&header).await?;
    Ok(())
}

/// Send a Token frame upstream (last rank → rank 0 path).
/// Body: 8 B big-endian i64 token id.
pub async fn send_token_upstream(
    srv: &Mutex<ActivationServer>,
    token_id: i64,
) -> TransportResult<()> {
    let mut bytes = [0u8; 12];
    bytes[0..4].copy_from_slice(&(FrameKind::Token as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&token_id.to_be_bytes());
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a Token frame's body (kind already consumed). Returns the
/// sampled token id.
pub async fn recv_token_body_client(cli: &Mutex<ActivationClient>) -> TransportResult<i64> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(8).await?;
    drop(guard);
    if raw.len() != 8 {
        return Err(TransportError::SocketClosed);
    }
    Ok(i64::from_be_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

/// Forward a Reset frame downstream — used by mid ranks after consuming
/// the kind code from upstream. The body is empty so this is just a
/// `send_reset` to the next peer; the `upstream` argument is held by the
/// caller and the kind is already consumed before this point.
pub async fn forward_reset(downstream: &Mutex<ActivationClient>) -> TransportResult<()> {
    send_reset(downstream).await
}

/// On-wire size of the heartbeat body (nonce only, no shape header).
pub const HEARTBEAT_BODY_BYTES: usize = 8;

/// Send a HeartbeatPing downstream (rank N → rank N+1) with a caller-
/// provided nonce. The receiver echoes the nonce back in a
/// HeartbeatPong so the sender can correlate pong-to-ping.
///
/// Cheap: 12 bytes (4 B kind + 8 B nonce) over an already-open TCP
/// stream + a single flush. Negligible cost at heartbeat cadences ≥ 100
/// ms. Held under the same mutex as Forward/Reset, so a heartbeat
/// cannot interleave with the body of a Forward frame and corrupt the
/// wire — the trade is that during a long Forward send the heartbeat
/// queues behind it (acceptable: a healthy Forward is the strongest
/// possible liveness signal).
pub async fn send_heartbeat_ping(cli: &Mutex<ActivationClient>, nonce: u64) -> TransportResult<()> {
    let mut bytes = [0u8; 4 + HEARTBEAT_BODY_BYTES];
    bytes[0..4].copy_from_slice(&(FrameKind::HeartbeatPing as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&nonce.to_be_bytes());
    let mut guard = cli.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Send a HeartbeatPing upstream (worker → driver). Used when an
/// intermediate or last-rank worker wants to assert liveness back to
/// rank 0 without waiting to be polled — e.g. an idle warmup window.
/// Not used by v1's driver-pulls-pong design (rank 0 sends Ping,
/// downstream echoes Pong) but plumbed symmetrically so a future
/// bidirectional probe doesn't need a new helper.
pub async fn send_heartbeat_ping_upstream(
    srv: &Mutex<ActivationServer>,
    nonce: u64,
) -> TransportResult<()> {
    let mut bytes = [0u8; 4 + HEARTBEAT_BODY_BYTES];
    bytes[0..4].copy_from_slice(&(FrameKind::HeartbeatPing as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&nonce.to_be_bytes());
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Send a HeartbeatPong upstream (worker → driver) echoing the nonce
/// the matched ping carried. Symmetric to `send_token_upstream`: uses
/// the server-side socket the upstream peer connected on.
pub async fn send_heartbeat_pong_upstream(
    srv: &Mutex<ActivationServer>,
    nonce: u64,
) -> TransportResult<()> {
    let mut bytes = [0u8; 4 + HEARTBEAT_BODY_BYTES];
    bytes[0..4].copy_from_slice(&(FrameKind::HeartbeatPong as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&nonce.to_be_bytes());
    let mut guard = srv.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Send a HeartbeatPong downstream (driver → worker) echoing the nonce.
/// Used by mid-rank workers when they receive a Ping from upstream and
/// want to reply without involving their downstream peer.
pub async fn send_heartbeat_pong_downstream(
    cli: &Mutex<ActivationClient>,
    nonce: u64,
) -> TransportResult<()> {
    let mut bytes = [0u8; 4 + HEARTBEAT_BODY_BYTES];
    bytes[0..4].copy_from_slice(&(FrameKind::HeartbeatPong as u32).to_be_bytes());
    bytes[4..12].copy_from_slice(&nonce.to_be_bytes());
    let mut guard = cli.lock().await;
    guard.send_raw(&bytes).await?;
    Ok(())
}

/// Receive a heartbeat body (8-byte BE nonce). The kind code has
/// already been consumed by `recv_kind_*`. Returns the echoed nonce.
pub async fn recv_heartbeat_body_server(srv: &Mutex<ActivationServer>) -> TransportResult<u64> {
    let mut guard = srv.lock().await;
    let raw = guard.recv_raw(HEARTBEAT_BODY_BYTES).await?;
    drop(guard);
    if raw.len() != HEARTBEAT_BODY_BYTES {
        return Err(TransportError::SocketClosed);
    }
    Ok(u64::from_be_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

/// Receive a heartbeat body from the client-side socket (e.g. a
/// driver waiting on a downstream pong).
pub async fn recv_heartbeat_body_client(cli: &Mutex<ActivationClient>) -> TransportResult<u64> {
    let mut guard = cli.lock().await;
    let raw = guard.recv_raw(HEARTBEAT_BODY_BYTES).await?;
    drop(guard);
    if raw.len() != HEARTBEAT_BODY_BYTES {
        return Err(TransportError::SocketClosed);
    }
    Ok(u64::from_be_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

/// Heartbeat watchdog state — tracks consecutive misses against a fixed
/// threshold and answers `is_dead()` once the worker has missed more
/// pongs than the threshold tolerates.
///
/// Scope: failure detection only. Re-spawning the worker process,
/// re-connecting the transport, and re-routing the layer range belong
/// to the orchestrator track and are out of scope for iter 092
/// (FOLLOW-UP-orchestrator, see docs/architecture/heartbeat-recovery.md).
#[derive(Clone, Debug)]
pub struct HeartbeatWatchdog {
    /// Tolerance: declare dead once `consecutive_misses > max_misses`.
    pub max_misses: u32,
    consecutive_misses: u32,
    /// Bumped on every successful ping/pong round-trip; useful to gate
    /// "we never even got a first heartbeat" from "we lost the worker
    /// after running for an hour" in the orchestrator's restart policy.
    successes: u64,
}

impl HeartbeatWatchdog {
    /// `max_misses` is the consecutive-miss tolerance. Per task spec
    /// "2 misses in a row → mark worker as dead", default is 1 (so the
    /// second miss trips the watchdog).
    pub fn new(max_misses: u32) -> Self {
        Self {
            max_misses,
            consecutive_misses: 0,
            successes: 0,
        }
    }

    /// Record a successful ping → pong round-trip. Resets the miss
    /// counter atomically with the success count bump.
    pub fn record_success(&mut self) {
        self.consecutive_misses = 0;
        self.successes = self.successes.saturating_add(1);
    }

    /// Record a missed pong (either an explicit timeout or a transport
    /// error on the ping itself). Returns `true` once the worker has
    /// crossed the death threshold.
    pub fn record_miss(&mut self) -> bool {
        self.consecutive_misses = self.consecutive_misses.saturating_add(1);
        self.is_dead()
    }

    pub fn is_dead(&self) -> bool {
        self.consecutive_misses > self.max_misses
    }

    pub fn consecutive_misses(&self) -> u32 {
        self.consecutive_misses
    }

    pub fn successes(&self) -> u64 {
        self.successes
    }
}

impl Default for HeartbeatWatchdog {
    fn default() -> Self {
        // Task spec: "2 misses in a row → mark worker as dead".
        // record_miss returns true once consecutive_misses > max_misses
        // (strict GT). So max_misses=1 trips on the SECOND consecutive
        // miss, exactly matching the spec.
        Self::new(1)
    }
}

/// Outcome of one heartbeat round. `Alive` and `Missed` keep the loop
/// running; `Dead` ends it (the watchdog crossed its threshold) and
/// `WireBroken` ends it (socket-level failure — separate signal so the
/// orchestrator can distinguish "worker hung" from "TCP died").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeartbeatOutcome {
    /// Pong with matching nonce arrived inside the timeout.
    Alive,
    /// Timeout fired or an unexpected frame arrived; watchdog has
    /// recorded the miss but has not yet crossed `max_misses`.
    Missed,
    /// Watchdog crossed its `max_misses` threshold on this round's miss.
    /// Caller should stop pinging and escalate to the orchestrator.
    Dead,
    /// Transport-level failure (socket closed, IO error). Distinct from
    /// `Dead` because the worker may still be alive — the link is what
    /// broke. Caller should also stop and escalate.
    WireBroken,
}

/// Run one heartbeat round against `downstream`: send Ping with `nonce`,
/// wait up to `timeout` for the matching Pong. The mutex is held for the
/// entire round so the wire stays in sync — see "Mutex serialization" in
/// `docs/architecture/heartbeat-recovery.md`.
///
/// Returns the new outcome AFTER updating the watchdog: `Alive` on
/// success, `Missed` on timeout if still under threshold, `Dead` on
/// timeout if the watchdog just crossed threshold, `WireBroken` on a
/// socket-level error.
///
/// Stale pongs (Pong frames whose nonce doesn't match this round's
/// nonce) are drained silently and the loop continues waiting for the
/// matching pong until the deadline. This handles the recovery path
/// where a previous round timed out, released the lock, and the worker
/// later sent the late pong that's now sitting in the receive buffer.
pub async fn ping_one_round(
    downstream: &Mutex<ActivationClient>,
    nonce: u64,
    timeout: Duration,
    watchdog: &Mutex<HeartbeatWatchdog>,
) -> HeartbeatOutcome {
    let mut guard = downstream.lock().await;

    // Send the ping. A send-side failure means the link is broken;
    // count it as a miss too so the watchdog escalates if it persists.
    let mut ping = [0u8; 4 + HEARTBEAT_BODY_BYTES];
    ping[0..4].copy_from_slice(&(FrameKind::HeartbeatPing as u32).to_be_bytes());
    ping[4..12].copy_from_slice(&nonce.to_be_bytes());
    if guard.send_raw(&ping).await.is_err() {
        drop(guard);
        let mut wg = watchdog.lock().await;
        let dead = wg.record_miss();
        return if dead {
            HeartbeatOutcome::Dead
        } else {
            HeartbeatOutcome::WireBroken
        };
    }

    let deadline = Instant::now() + timeout;
    // Loop until: matched pong (Alive), deadline elapsed (Missed/Dead),
    // unexpected frame / socket error (WireBroken).
    let outcome = loop {
        let now = Instant::now();
        if now >= deadline {
            break MissOutcome::Timeout;
        }
        let remaining = deadline - now;
        let kind_recv = tokio::time::timeout(remaining, guard.recv_raw(4)).await;
        let kind_bytes = match kind_recv {
            Ok(Ok(b)) => b,
            Ok(Err(_)) => break MissOutcome::WireError,
            Err(_) => break MissOutcome::Timeout,
        };
        if kind_bytes.len() != 4 {
            break MissOutcome::WireError;
        }
        let code = u32::from_be_bytes([kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3]]);
        match FrameKind::from_code(code) {
            Some(FrameKind::HeartbeatPong) => {
                // Body comes next — read it under the same guard.
                let body_recv = tokio::time::timeout(
                    deadline.saturating_duration_since(Instant::now()),
                    guard.recv_raw(HEARTBEAT_BODY_BYTES),
                )
                .await;
                let body = match body_recv {
                    Ok(Ok(b)) if b.len() == HEARTBEAT_BODY_BYTES => b,
                    _ => break MissOutcome::WireError,
                };
                let got_nonce = u64::from_be_bytes([
                    body[0], body[1], body[2], body[3], body[4], body[5], body[6], body[7],
                ]);
                if got_nonce == nonce {
                    break MissOutcome::Matched;
                }
                // Stale pong from an earlier round — drain it and keep
                // waiting for the current round's matching pong.
                continue;
            }
            Some(_) | None => {
                // An unexpected frame on a channel reserved (in this
                // call) for heartbeat traffic indicates the wire is
                // desynced. Surface as WireBroken so the orchestrator
                // can tear down the link rather than spin reading
                // garbage. Note: `ping_one_round` assumes the caller
                // serializes against the Forward path — if a Forward
                // response (Token) arrives mid-heartbeat that's the
                // caller's bug, not the worker's.
                break MissOutcome::WireError;
            }
        }
    };
    drop(guard);

    let mut wg = watchdog.lock().await;
    match outcome {
        MissOutcome::Matched => {
            wg.record_success();
            HeartbeatOutcome::Alive
        }
        MissOutcome::Timeout => {
            let dead = wg.record_miss();
            if dead {
                HeartbeatOutcome::Dead
            } else {
                HeartbeatOutcome::Missed
            }
        }
        MissOutcome::WireError => {
            let dead = wg.record_miss();
            if dead {
                HeartbeatOutcome::Dead
            } else {
                HeartbeatOutcome::WireBroken
            }
        }
    }
}

/// Internal helper enum kept private so the public `HeartbeatOutcome`
/// API stays minimal (callers care about Alive/Missed/Dead/WireBroken;
/// "was it a timeout or a wire error?" is an implementation detail of
/// the bookkeeping path).
enum MissOutcome {
    Matched,
    Timeout,
    WireError,
}

/// Driver-side cadence loop. Sends a ping every `interval`, waits up to
/// `timeout` for the matching pong, updates the watchdog, and returns
/// when the watchdog declares the worker dead OR the cancel flag flips
/// to `true` OR the transport breaks.
///
/// Designed to be `tokio::spawn`'d from rank-0 setup. Cancellation: flip
/// `cancel` to `true` — the loop will exit at its next sleep boundary
/// (≤ `interval` later). Calling `JoinHandle::abort` also works but
/// leaves the watchdog in whatever state the last round landed in.
///
/// **Concurrency contract:** the caller must guarantee no other task
/// reads from or writes to `downstream` while this loop is running OR
/// must serialize against the heartbeat via the same socket mutex (the
/// Forward path does the latter by taking `downstream.lock()` in
/// `forward_one_token_first`). Violating this races for frame ordering
/// — see the "Mutex serialization" section of the design doc.
///
/// Returns the final outcome that ended the loop (`Dead`, `WireBroken`,
/// or, if cancelled cleanly, the most recent round's outcome).
pub async fn run_heartbeat_loop(
    downstream: Arc<Mutex<ActivationClient>>,
    watchdog: Arc<Mutex<HeartbeatWatchdog>>,
    interval: Duration,
    timeout: Duration,
    cancel: Arc<std::sync::atomic::AtomicBool>,
) -> HeartbeatOutcome {
    use std::sync::atomic::Ordering;
    // Monotonic nonce — never reused within the life of this loop so a
    // stale pong from a previous round can be detected (and dropped).
    let mut nonce: u64 = 0;
    let mut last_outcome = HeartbeatOutcome::Alive;
    loop {
        if cancel.load(Ordering::Acquire) {
            return last_outcome;
        }
        tokio::time::sleep(interval).await;
        if cancel.load(Ordering::Acquire) {
            return last_outcome;
        }
        nonce = nonce.wrapping_add(1);
        let outcome = ping_one_round(&downstream, nonce, timeout, &watchdog).await;
        last_outcome = outcome;
        match outcome {
            HeartbeatOutcome::Alive | HeartbeatOutcome::Missed => continue,
            HeartbeatOutcome::Dead | HeartbeatOutcome::WireBroken => return outcome,
        }
    }
}

/// Bundle of the per-rank transport state. The Builder constructs this
/// during `connect()` and hands it to the Engine.
#[derive(Default)]
pub struct StageTransport {
    pub upstream: Option<Arc<Mutex<ActivationServer>>>,
    pub downstream: Option<Arc<Mutex<ActivationClient>>>,
}

impl StageTransport {
    pub fn is_first(&self) -> bool {
        self.upstream.is_none()
    }

    pub fn is_last(&self) -> bool {
        self.downstream.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_kind_round_trip_includes_heartbeat() {
        for k in [
            FrameKind::Forward,
            FrameKind::Reset,
            FrameKind::Token,
            FrameKind::HeartbeatPing,
            FrameKind::HeartbeatPong,
        ] {
            assert_eq!(
                FrameKind::from_code(k as u32),
                Some(k),
                "round-trip failed for {k:?}"
            );
        }
        assert_eq!(FrameKind::from_code(0xDEAD_BEEF), None);
    }

    #[test]
    fn heartbeat_codes_disjoint_from_other_frames() {
        // The recovery design relies on the kind code being unique so a
        // stale heartbeat can never be confused with a Forward / Token
        // body. A regression here would silently misroute the first
        // byte of a tensor as a frame code.
        let codes = [
            FrameKind::Forward as u32,
            FrameKind::Reset as u32,
            FrameKind::Token as u32,
            FrameKind::HeartbeatPing as u32,
            FrameKind::HeartbeatPong as u32,
        ];
        for (i, a) in codes.iter().enumerate() {
            for (j, b) in codes.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "FrameKind codes collide at {i}/{j}");
                }
            }
        }
    }

    #[test]
    fn watchdog_default_is_two_misses() {
        // Spec: "2 misses in a row → mark worker as dead". Default
        // max_misses=1 means record_miss returns true on the SECOND
        // miss, not the first.
        let mut w = HeartbeatWatchdog::default();
        assert!(!w.is_dead());
        assert!(!w.record_miss()); // 1st miss
        assert!(!w.is_dead());
        assert!(w.record_miss()); // 2nd miss — declared dead
        assert!(w.is_dead());
    }

    #[test]
    fn watchdog_success_resets_miss_counter() {
        let mut w = HeartbeatWatchdog::default();
        assert!(!w.record_miss()); // 1st miss
        w.record_success(); // wipes the streak
        assert!(!w.is_dead());
        assert_eq!(w.consecutive_misses(), 0);
        assert_eq!(w.successes(), 1);
        assert!(!w.record_miss()); // back to 1st miss in a NEW streak
        assert!(!w.is_dead());
    }

    #[test]
    fn watchdog_with_higher_tolerance() {
        // A flaky link wants more tolerance — e.g. tolerate 5 misses.
        // Spec semantics extend: dead when consecutive_misses > max_misses.
        let mut w = HeartbeatWatchdog::new(5);
        for i in 0..5 {
            assert!(!w.record_miss(), "miss {i} should not declare dead");
        }
        assert!(w.record_miss(), "6th miss should declare dead");
    }

    #[test]
    fn watchdog_successes_saturate() {
        // Defensive: a watchdog that runs for years shouldn't wrap
        // u64::MAX → 0 and erase the long-running-process signal.
        let mut w = HeartbeatWatchdog::default();
        // Force the saturate edge directly — running 2^64 cycles in a
        // unit test isn't going to happen.
        for _ in 0..10 {
            w.record_success();
        }
        assert_eq!(w.successes(), 10);
    }
}
