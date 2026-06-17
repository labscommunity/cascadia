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
use cascadia_transport::ByteStream;
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::Stream;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
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

    #[error("stream injection unsupported by this engine")]
    StreamInjectionUnsupported,
}

pub type EngineResult<T> = Result<T, EngineError>;

#[cfg(test)]
mod error_tests {
    use super::*;
    #[test]
    fn stream_injection_unsupported_displays() {
        let e = EngineError::StreamInjectionUnsupported;
        assert_eq!(e.to_string(), "stream injection unsupported by this engine");
    }

    struct OptOut;
    impl Engine for OptOut {
        fn warmup(&mut self) {}
        fn submit(&mut self, _t: GenerationTask) -> EngineResult<()> {
            Ok(())
        }
        fn step(&mut self) -> Vec<(TaskId, Chunk)> {
            Vec::new()
        }
    }

    #[test]
    fn reattach_default_none_none_is_ok_but_some_is_unsupported() {
        let mut e = OptOut;
        assert!(e.reattach_streams(None, None).is_ok());
        let s: cascadia_transport::ByteStream = Box::new(tokio::io::duplex(8).0);
        assert!(matches!(
            e.reattach_streams(Some(s), None),
            Err(EngineError::StreamInjectionUnsupported)
        ));
    }
}

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
    fn step(&mut self) -> Vec<(TaskId, Chunk)>;

    /// Best-effort cancellation of an in-flight task. Engines that do
    /// not support mid-stream cancellation may treat this as a no-op.
    fn cancel(&mut self, _task_id: &TaskId) {}

    /// Tear down the engine. Idempotent.
    fn close(&mut self) {}

    /// Re-attach a fresh stream pair to a LIVE engine after one died, without
    /// reloading weights. `Some` replaces that side (old dropped); `None`
    /// leaves it. `(None, None)` is an Ok no-op on every engine. Sync + no
    /// `block_on`: it only swaps stored handles. Default: opt out — `Ok` for
    /// `(None, None)`, else `StreamInjectionUnsupported`.
    fn reattach_streams(
        &mut self,
        up: Option<ByteStream>,
        down: Option<ByteStream>,
    ) -> EngineResult<()> {
        if up.is_none() && down.is_none() {
            Ok(())
        } else {
            Err(EngineError::StreamInjectionUnsupported)
        }
    }
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

    /// Initial inject: wire up to peers via embedder-supplied streams instead
    /// of dialing/binding TCP. `None` = no peer on that side. Default: opt out
    /// — `Ok` for `(None, None)`, else `StreamInjectionUnsupported`.
    async fn connect_streams(
        &mut self,
        up: Option<ByteStream>,
        down: Option<ByteStream>,
    ) -> EngineResult<()> {
        if up.is_none() && down.is_none() {
            Ok(())
        } else {
            Err(EngineError::StreamInjectionUnsupported)
        }
    }

    /// Construct the live engine. Must be called *after* `connect` + `load`.
    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>>;

    /// Tear down any partially-initialised resources (sockets, weights).
    fn close(&mut self) {}
}
