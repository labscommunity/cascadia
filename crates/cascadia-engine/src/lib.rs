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

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

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
    pub fn is_connection_fatal(&self) -> bool {
        match self {
            EngineError::NotConnected => true,
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
}
