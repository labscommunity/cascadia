//! Engine + Builder trait definitions.
//!
//! Mirrors `cascadia/worker/engines/base.py`. Two narrow concerns:
//!
//! * [`Builder`] — configure listening, connect to peers, load the shard,
//!   then construct the [`Engine`].
//! * [`Engine`] — submit tasks; poll [`Engine::step`] for emitted chunks.
//!
//! `Engine::step` is intentionally synchronous — engines run an inference
//! request through to completion in a single call (matching the Python
//! semantics today). The async surface lives in [`Builder`] for I/O during
//! load and connect.

use std::pin::Pin;

use async_trait::async_trait;
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::Stream;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("not yet loaded; call load() before build()")]
    NotLoaded,

    #[error("not yet connected; call connect() before build()")]
    NotConnected,

    #[error("peer layout rejected: {0}")]
    PeerRejected(String),

    #[error("shard rejected: {0}")]
    ShardRejected(String),

    #[error("model not found at {0}")]
    ModelNotFound(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("queue full ({queued} pending, cap {cap})")]
    QueueFull { queued: usize, cap: usize },

    /// The prompt cannot fit this engine's per-request window. Distinct from
    /// `QueueFull`: that one clears when load drops, this one never does, so
    /// the API must answer 413 rather than a retryable 5xx.
    #[error("prompt too long: {0}")]
    PromptTooLong(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The in-flight batch was abandoned, but the engine and its peer links
    /// are healthy. A pipeline stage that fails its packed step NACKs its
    /// upstream (an empty token frame) instead of going silent; the batch is
    /// lost, the wire stays frame-aligned, and the next batch can proceed.
    ///
    /// Its own variant on purpose. This is the one failure that MUST NOT be
    /// [connection-fatal](EngineError::is_connection_fatal): a NACK has to
    /// make the relay loop back off and continue, not exit for a supervisor
    /// rebuild. Left as a [`EngineError::Backend`] string that correctness
    /// hung on a substring classifier never matching the message — one
    /// reworded message mentioning a dropped or unreachable peer and every
    /// NACK would start tearing stages down. The variant makes that
    /// structural: `is_connection_fatal` answers `false` here before it looks
    /// at any text, so the message is free to say whatever an operator (and
    /// the SSE client that receives it) needs to read.
    #[error("batch aborted: {0}")]
    BatchAborted(String),

    /// A `step()` failure attributed to a specific task. Engines wrap their
    /// underlying error in this at the failure site when the active task is
    /// known, so the runner can route the failure to that task's stream
    /// instead of ending whichever stream happens to observe it.
    #[error("task {task_id}: {source}")]
    Task {
        task_id: TaskId,
        #[source]
        source: Box<EngineError>,
    },
}

impl EngineError {
    /// The task this error is attributed to, if any. Returns `None` for
    /// engine-level / task-less failures.
    pub fn task_id(&self) -> Option<&TaskId> {
        match self {
            EngineError::Task { task_id, .. } => Some(task_id),
            _ => None,
        }
    }

    /// Attribute an error to `task_id`, wrapping it in [`EngineError::Task`]
    /// unless it already carries an attribution.
    pub fn for_task(self, task_id: TaskId) -> Self {
        match self {
            EngineError::Task { .. } => self,
            source => EngineError::Task {
                task_id,
                source: Box::new(source),
            },
        }
    }

    /// Whether this error means the engine's peer link is dead and cannot
    /// recover in-process — a worker stage whose upstream socket is dropped
    /// can only get a fresh connection by being rebuilt (re-`accept()` only
    /// happens at startup). The relay loop exits on this so the supervisor
    /// (systemd `Restart=on-failure`) rebuilds the stage, rather than
    /// spin-and-flood at the backoff rate forever.
    ///
    /// Transport errors reach the engine flattened to strings via
    /// [`EngineError::Backend`] (the worker calls `e.to_string()`), so this
    /// matches the same substrings the dist-spec worker uses to classify a
    /// fatal link drop, plus the structural [`EngineError::NotConnected`].
    /// Covered: clean teardown ("socket closed"/"not connected"), a
    /// black-holed peer ("idle ceiling"), a peer crash ("connection
    /// reset"/"broken pipe"/"connection aborted" — TCP RST is the dominant
    /// dead-peer case), and a mid-frame stall ("recv_exact timed out"). A
    /// connected-but-misbehaving peer (bad frame kind, stray response) is
    /// NOT fatal — that link can still deliver a good frame next.
    ///
    /// [`EngineError::BatchAborted`] is answered structurally, ahead of any
    /// string inspection: an aborted batch leaves a healthy link, so the
    /// relay loop must back off and continue. Because the check never reads
    /// its message, no rewording of an abort text can turn a NACK into a
    /// stage teardown.
    pub fn is_connection_fatal(&self) -> bool {
        match self {
            EngineError::NotConnected => true,
            // Structural, and BEFORE any substring matching: the batch is
            // gone, the link is not. Its message is operator/SSE-facing text
            // and must never be able to reclassify the error.
            EngineError::BatchAborted(_) => false,
            EngineError::Task { source, .. } => source.is_connection_fatal(),
            EngineError::Backend(msg) => {
                let msg = msg.to_ascii_lowercase();
                // Clean teardown / black-holed peer.
                msg.contains("socket closed")
                    || msg.contains("not connected")
                    || msg.contains("idle ceiling")
                    // Peer crash: TCP RST / broken pipe / aborted.
                    || msg.contains("connection reset")
                    || msg.contains("broken pipe")
                    || msg.contains("connection aborted")
                    // Mid-frame deadline (recv_exact wall-clock bound).
                    || msg.contains("recv_exact timed out")
            }
            // A structurally-typed io error is fatal for the same kinds the
            // transport layer treats as fatal (see
            // `recv_error_is_connection_fatal`).
            //
            // This arm is LIVE, and deliberately so: the ov-runtime relay
            // escalation raises `Io(TimedOut)` when a middle rank's downstream
            // has stopped answering, precisely so that "this link is gone" is
            // answered by the error's TYPE rather than by a `Backend` string
            // hand-crafted to contain a substring below. Anything that needs to
            // be classifiable should arrive here, not there. (The dist-spec
            // worker still flattens its recv errors to `Backend`.)
            EngineError::Io(e) => matches!(
                e.kind(),
                std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::UnexpectedEof
            ),
            _ => false,
        }
    }
}

pub type EngineResult<T> = Result<T, EngineError>;

/// Stream of load-progress events yielded by [`Builder::load`].
pub type LoadStream = Pin<Box<dyn Stream<Item = LoadProgress> + Send>>;

/// Engine-side: an active inference runtime. Tasks are submitted via
/// [`submit`] and emitted via [`step`].
///
/// Implementations are not required to be `Send` themselves but the
/// runner holds them behind a `Mutex`, so they MUST be `Send`.
/// Issue-34 Option C: the engine's host-side KV export/import contract (the plane's `KvCoordination`
/// boundary). An engine that holds a prefix KV cache returns `Some` from [`Engine::kv_coordination`];
/// engines without one (mock / openvino) keep the default `None`. Wire-typed (`cascadia_kv_wire`) so
/// the enterprise plane needs no engine-internal types. All host-side buffer ops — no device FFI.
#[cfg(feature = "kv_coord")]
pub trait KvCoordination {
    /// This rank's model fingerprint — cache key + cross-rev guard.
    fn model_fingerprint(&self) -> u64;
    /// KV buffer layout version (codec rejects a mismatch).
    fn layout_version(&self) -> u16;
    /// Engine build revision (codec rejects a mismatch).
    fn engine_rev(&self) -> u64;
    /// Tokenize a rendered prompt with the engine's own tokenizer, so the head's NEGOTIATE uses the
    /// exact token sequence the prefill will key the prefix cache on. `None` if no tokenizer.
    fn tokenize(&self, text: &str) -> Option<Vec<i32>>;
    /// NEGOTIATE: longest-common-prefix of `token_ids` against this holder's cache for `partner`.
    /// Returns the stamped `(snapshot_epoch, prefix_token_len)`, or `None` ⇒ NotFound.
    fn lookup(&mut self, partner: &str, token_ids: &[i32]) -> Option<(u64, u32)>;
    /// GET: export the snapshot asserted by `(epoch, len)` → wire `Manifest` + per-layer `(k, v)`
    /// byte payloads. `None` if the holder's `(epoch, len)` ≠ asserted (evicted / drifted).
    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(cascadia_kv_wire::Manifest, Vec<(Vec<u8>, Vec<u8>)>)>;
    /// Consumer INSERT: materialize a pulled, validated snapshot into the cache so the next prefill
    /// auto-hits. `Err(())` ⇒ rejected / OOM (the rank votes fail).
    #[allow(clippy::result_unit_err)]
    fn insert(
        &mut self,
        manifest: &cascadia_kv_wire::Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()>;
    /// Issue-34 multi-stage: stash a pulled DOWNSTREAM rank's snapshot (rank ≠ this head) for inline
    /// delivery in the head's `RESTORE` frame — the head can't use a downstream rank's KV locally.
    /// Default: unsupported (single-stage engines never see a rank > 0 insert) ⇒ that rank votes cold.
    #[allow(clippy::result_unit_err)]
    fn stash_downstream_rank(
        &mut self,
        _rank: u16,
        _manifest: &cascadia_kv_wire::Manifest,
        _payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        Err(())
    }
    /// Plane-based cross-chain warm-resume: applies the rank's pulled KV staged under `epoch` in this
    /// engine's cache (splice-agnostic; §0(B)). Returns true on success (state set + warm armed).
    /// Default: unsupported.
    #[allow(clippy::result_unit_err)]
    fn apply_warm_resume(&mut self, _epoch: u64) -> bool {
        false
    }
    /// Undo an [`Self::apply_warm_resume`] that the chain-wide verdict then rejected: drop the armed
    /// state and fall back to cold. The plane's verdict is meant to be all-or-nothing, but each rank
    /// applies BEFORE it confirms, so a later rank's failure would otherwise leave this rank warm while
    /// the head goes cold — the head then prefills a full cold prompt through a rank whose KV is
    /// pre-seeded, producing wrong tokens. A stale arm also contaminates the NEXT request, so this must
    /// be safe to call at any time, including for an epoch this rank never armed.
    ///
    /// Default: no-op — correct for engines whose `apply_warm_resume` is itself unsupported.
    fn abort_warm_resume(&mut self, _epoch: u64) {}
    /// Number of ranks in an N-stage chain that bear OWN KV (the cross-chain pull must GET + restore
    /// only these). Default: all `total_ranks` (every rank has its own KV — ov-runtime/qwen36/dist-spec).
    /// KV-sharing engines (gemma4: all own-KV in stage_0) override to return fewer.
    fn kv_bearing_ranks(&self, total_ranks: usize) -> usize {
        total_ranks
    }
}

/// Issue-34 Option C: the **holder-serve** half of [`KvCoordination`], decoupled from the engine
/// lock. `lookup`/`export` read the captured-snapshot cache only (no live inference state), so this is
/// `&self` + `Send + Sync` and serves over a shared handle the engine hands out via
/// [`Engine::kv_holder`]. The point: a node is moved-away-from *because* it is busy generating (which
/// holds the engine lock); routing the holder through that lock would starve every pull. Serving from
/// this handle instead lets a busy node still answer NEGOTIATE/GET — it touches only the snapshot
/// cache's own (uncontended-mid-generation) lock.
#[cfg(feature = "kv_coord")]
pub trait KvSnapshotHolder: Send + Sync {
    /// This rank's model fingerprint (static; cache key + cross-rev guard).
    fn model_fingerprint(&self) -> u64;
    /// NEGOTIATE: longest-common-prefix of `token_ids` against the captured cache.
    fn lookup(&self, partner: &str, token_ids: &[i32]) -> Option<(u64, u32)>;
    /// GET: export the snapshot asserted by `(epoch, len)` as wire `Manifest` + per-layer payloads.
    fn export(
        &self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(cascadia_kv_wire::Manifest, Vec<(Vec<u8>, Vec<u8>)>)>;
}

/// Issue-34 plane warm-resume: the **hand-off** half, decoupled from the engine lock in the other
/// direction from [`KvSnapshotHolder`]. The plane's commit path runs while the engine is usually
/// parked inside `step()` holding the engine mutex, so it cannot reach the engine to apply a pulled
/// slice — that is the deadlock this exists to avoid. Instead it parks the slice behind this handle's
/// own lock and the engine drains it from inside its recv loop, before the turn's forward.
#[cfg(feature = "kv_coord")]
pub trait KvWarmHandoff: Send + Sync {
    /// Park a pulled slice for the engine to apply. Never blocks on engine work.
    fn put(
        &self,
        epoch: u64,
        manifest: cascadia_kv_wire::Manifest,
        payloads: Vec<(Vec<u8>, Vec<u8>)>,
    );
}

pub trait Engine: Send {
    /// One short forward to compile kernels and warm device caches.
    fn warmup(&mut self);

    /// Enqueue a task. The engine is free to defer execution to a later
    /// `step()` call. Submitting an already-pending task is a no-op.
    /// Returns [`EngineError::QueueFull`] when the engine's pending
    /// queue is at capacity.
    fn submit(&mut self, task: GenerationTask) -> EngineResult<()>;

    /// Make progress on at most one pending task and return any chunks
    /// emitted. Returns an empty Vec when no work is in flight.
    ///
    /// `Err` means the engine failed to make progress (backend or
    /// transport failure) and is terminal for the in-flight task —
    /// engines recover their own state so the next submitted task
    /// starts fresh, but callers must not retry the failed one.
    ///
    /// Task attribution: `Err` affects at most the active task; queued
    /// tasks survive. Implementors that know the failed task SHOULD wrap
    /// their error with [`EngineError::for_task`] (or return
    /// [`EngineError::Task`]) so the runner can route the failure to that
    /// task's stream. A task-less `Err` (no active task, or a genuinely
    /// engine-level failure) ends whichever stream observes it — correct
    /// for engine death, the documented behavior for worker stages that
    /// own no user task.
    ///
    /// Failure idiom: implementors should return `Err` for engine-level
    /// failures (engine unusable / task aborted by the engine); emit a
    /// final-marker chunk for per-task completion, including task-level
    /// failure where the engine remains healthy.
    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>>;

    /// Cancel a task. Implementors SHOULD guarantee that after `cancel`
    /// returns, `step()` emits no further chunks for `task_id`: a queued
    /// task is dropped from the pending
    /// queue; an active task is abandoned and the engine's generation
    /// state reset so the next task starts fresh instead of waiting for
    /// the abandoned one to drain to completion. Engines that cannot
    /// cancel mid-stream may keep the default no-op — the runner still
    /// suppresses the task's chunks, but the engine slot stays busy
    /// until the task finishes on its own.
    fn cancel(&mut self, _task_id: &TaskId) {}

    /// Issue-34 Option C: the engine's KV export/import surface, if it holds a prefix cache. Default
    /// `None` — engines without KV coordination opt out. `&mut` because lookup/export/insert mutate
    /// the cache (LRU touch, restore). Gated so the wire crate stays out of default trees.
    #[cfg(feature = "kv_coord")]
    fn kv_coordination(&mut self) -> Option<&mut dyn KvCoordination> {
        None
    }

    /// Issue-34 Option C: a lock-free **holder** handle over this engine's captured-snapshot cache, if
    /// any. Default `None`. Grabbed once at engine load (cheap `Arc` clone) and served from a holder
    /// task, so a pull never contends the engine lock the live inference holds. `&self` because it
    /// shares the cache rather than mutating engine state.
    #[cfg(feature = "kv_coord")]
    fn kv_holder(&self) -> Option<std::sync::Arc<dyn KvSnapshotHolder>> {
        None
    }

    /// Issue-34 plane warm-resume: the mailbox a plane commit parks a pulled slice in, if this engine
    /// applies one itself. Default `None`. Grabbed once at engine load, like [`Self::kv_holder`].
    #[cfg(feature = "kv_coord")]
    fn kv_handoff(&self) -> Option<std::sync::Arc<dyn KvWarmHandoff>> {
        None
    }

    /// Tear down the engine. Idempotent.
    fn close(&mut self) {}
}

/// Builder-side: lifecycle of an [`Engine`] from CLI args → configured
/// listener → connected peers → loaded shard → live engine.
#[async_trait]
pub trait Builder: Send {
    /// Optional pre-connect hook for engines that need to bind a listening
    /// socket *before* peers connect to them. Engines without inbound
    /// peers (single-stage / first-stage) can leave this as a no-op.
    fn configure_listen(&mut self, _host: &str, _port: u16) {}

    /// Wire up to the upstream/downstream peers for this rank.
    /// Single-stage engines must reject any non-empty layout.
    async fn connect(&mut self, peers: PeerLayout) -> EngineResult<()>;

    /// Load model weights. Streams progress events.
    async fn load(&mut self, shard: ShardSpec) -> EngineResult<LoadStream>;

    /// Construct the live engine. Must be called *after* `connect` + `load`.
    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>>;

    /// Tear down any partially-initialised resources (sockets, weights).
    fn close(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_fatal_classification() {
        // Structural variant.
        assert!(EngineError::NotConnected.is_connection_fatal());
        // Flattened transport strings the dist-spec worker produces.
        for msg in [
            "socket closed during recv",
            "not connected; call connect()/accept() first",
            "frame-start idle ceiling hit after 900s; connection dropped",
            // Peer crash (TCP RST and its send/half-close variants).
            "io error: connection reset by peer",
            "io error: broken pipe",
            "io error: connection aborted",
            // Mid-frame stall surfaced by recv_exact's wall-clock bound.
            "io error: recv_exact timed out after 60s",
        ] {
            assert!(
                EngineError::Backend(msg.into()).is_connection_fatal(),
                "expected fatal: {msg}"
            );
        }
        // Structurally-typed io errors classify by kind (mirrors the
        // transport recv-fatal set), independent of the Backend-string path.
        use std::io::{Error as IoError, ErrorKind};
        for kind in [
            ErrorKind::TimedOut,
            ErrorKind::ConnectionReset,
            ErrorKind::BrokenPipe,
            ErrorKind::ConnectionAborted,
            ErrorKind::UnexpectedEof,
        ] {
            assert!(
                EngineError::Io(IoError::from(kind)).is_connection_fatal(),
                "expected fatal io kind: {kind:?}"
            );
        }
        assert!(!EngineError::Io(IoError::from(ErrorKind::NotFound)).is_connection_fatal());
        // Recoverable / unrelated failures are NOT fatal.
        assert!(!EngineError::Backend("bad kind 7".into()).is_connection_fatal());
        assert!(
            !EngineError::Backend("worker received LOGITS_RESPONSE".into()).is_connection_fatal()
        );
        assert!(!EngineError::NotLoaded.is_connection_fatal());
        // A task-attributed fatal error unwraps to its source.
        let wrapped = EngineError::Backend("socket closed".into()).for_task(TaskId::from("t1"));
        assert!(wrapped.is_connection_fatal());
    }

    /// An aborted batch is never a dead link. The relay loop must back off
    /// and keep driving on a NACK — exiting would hand the supervisor a
    /// rebuild for a stage whose socket is perfectly fine.
    ///
    /// The point of the variant is that this holds for ANY message: the abort
    /// text is operator- and SSE-facing prose that names the underlying cause,
    /// and that cause is frequently a transport failure on the *other* side of
    /// the pipeline, so it quotes exactly the substrings the classifier hunts
    /// for. As a `Backend` string every one of these would have been
    /// misclassified as fatal.
    #[test]
    fn batch_aborted_is_never_connection_fatal() {
        for msg in [
            "downstream stage failed its packed step and NACKed this batch \
             (empty token frame); the pipeline link stays aligned",
            // Abort messages that quote a peer's transport failure — the
            // exact fragility a substring classifier could not survive.
            "the packed step failed: backend error: packed token recv: recv_exact timed out",
            "the packed step failed: backend error: not connected; call connect() first",
            "the packed step failed: backend error: socket closed during recv",
            "the packed step failed: backend error: io error: broken pipe",
            "the packed step failed: backend error: connection reset by peer",
            "",
        ] {
            let e = EngineError::BatchAborted(msg.into());
            assert!(!e.is_connection_fatal(), "must not be fatal: {msg}");
            // And attribution must not resurrect fatality either.
            assert!(
                !e.for_task(TaskId::from("t1")).is_connection_fatal(),
                "{msg}"
            );
        }
        // The same texts as genuine transport failures ARE still fatal — the
        // variant narrows the classifier, it does not blunt it.
        for msg in [
            "packed token recv: recv_exact timed out",
            "not connected; call connect() first",
            "socket closed during recv",
            "io error: broken pipe",
            "connection reset by peer",
        ] {
            assert!(
                EngineError::Backend(msg.into()).is_connection_fatal(),
                "must stay fatal: {msg}"
            );
        }
    }
}
