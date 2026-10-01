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
//!
//! # Cargo features
//!
//! * `kv_coord` — the KV coordination surface (issue 34). Off by default.
//! * `injected_streams` — issue #76: `Builder::connect_streams`,
//!   `Engine::reattach_streams`, `StreamLinks`, `LinkShape` and the
//!   validators `check_connect_streams` / `check_reattach_streams`. Off by
//!   default so a default consumer's tree stays free of tokio/socket2/
//!   prometheus (it pulls in `cascadia-transport`). Enable it on every crate
//!   that implements or calls these items: engine crates that accept injected
//!   streams, and the embedder (`cascadia-runner` already enables it). The
//!   stream type `ByteStream` is re-exported here; its contract lives on
//!   `cascadia_transport::InjectedStream`.

use std::pin::Pin;

use async_trait::async_trait;
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::Stream;
use thiserror::Error;

#[cfg(feature = "kv_coord")]
pub mod kv_handoff;
#[cfg(feature = "kv_coord")]
pub use kv_handoff::{KvHandoffMailbox, KvHandoffSlot};

#[cfg(feature = "injected_streams")]
pub use cascadia_transport::ByteStream;

/// Issue #76: the activation links handed to [`Builder::connect_streams`] and
/// [`Engine::reattach_streams`]. Pipeline stages use `upstream`/`downstream`;
/// an inkling expert-parallel (EP) worker uses `ep_driver`; an EP driver uses
/// `ep_workers`, one entry per expert worker in `--ep-workers` order. On
/// re-attach a `None` keeps that link, and an empty `ep_workers` keeps all.
///
/// `#[non_exhaustive]`: build one with [`StreamLinks::pipeline`],
/// [`StreamLinks::ep_driver`], [`StreamLinks::ep_workers`] or `default()`;
/// the fields stay public for reading, taking and assigning.
#[cfg(feature = "injected_streams")]
#[derive(Default)]
#[non_exhaustive]
pub struct StreamLinks {
    pub upstream: Option<ByteStream>,
    pub downstream: Option<ByteStream>,
    pub ep_driver: Option<ByteStream>,
    pub ep_workers: Vec<Option<ByteStream>>,
}

#[cfg(feature = "injected_streams")]
impl StreamLinks {
    /// Pipeline links only.
    pub fn pipeline(upstream: Option<ByteStream>, downstream: Option<ByteStream>) -> Self {
        Self {
            upstream,
            downstream,
            ..Self::default()
        }
    }

    /// An EP worker's single link to its driver.
    pub fn ep_driver(stream: ByteStream) -> Self {
        Self {
            ep_driver: Some(stream),
            ..Self::default()
        }
    }

    /// An EP driver's links to its expert workers, one entry per worker in
    /// `--ep-workers` order.
    ///
    /// For `connect_streams` every entry must be `Some` and the length must
    /// equal the worker count. For `reattach_streams` an empty vector keeps
    /// every worker's link, and a `None` entry keeps that worker's link (the
    /// length must then equal the worker count).
    pub fn ep_workers(streams: Vec<Option<ByteStream>>) -> Self {
        Self {
            ep_workers: streams,
            ..Self::default()
        }
    }

    /// Which links this value carries; `ep_workers` counts the `Some` entries.
    pub fn shape(&self) -> LinkShape {
        LinkShape {
            upstream: self.upstream.is_some(),
            downstream: self.downstream.is_some(),
            ep_driver: self.ep_driver.is_some(),
            ep_workers: self.ep_workers.iter().filter(|s| s.is_some()).count(),
        }
    }
}

/// Issue #76: which links a stage has (for validation) or a [`StreamLinks`]
/// carries. Pipeline stages and EP roles never mix (the EP driver runs as a
/// single stage).
///
/// Public for out-of-tree `Engine`/`Builder` implementers, who describe their
/// stage with it and call [`check_connect_streams`] /
/// [`check_reattach_streams`]. `#[non_exhaustive]`: build one with
/// [`LinkShape::pipeline`], [`LinkShape::ep_driver`],
/// [`LinkShape::ep_workers`] or `default()`.
#[cfg(feature = "injected_streams")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LinkShape {
    pub upstream: bool,
    pub downstream: bool,
    pub ep_driver: bool,
    pub ep_workers: usize,
}

#[cfg(feature = "injected_streams")]
impl LinkShape {
    /// A pipeline stage: head `(false, true)`, relay `(true, true)`, tail
    /// `(true, false)`, standalone `(false, false)`.
    pub fn pipeline(upstream: bool, downstream: bool) -> Self {
        Self {
            upstream,
            downstream,
            ..Self::default()
        }
    }

    /// An EP worker: only the link to its driver.
    pub fn ep_driver() -> Self {
        Self {
            ep_driver: true,
            ..Self::default()
        }
    }

    /// An EP driver with `n` expert workers.
    pub fn ep_workers(n: usize) -> Self {
        Self {
            ep_workers: n,
            ..Self::default()
        }
    }
}

/// Issue #76: validate a `connect_streams` call. Every link the stage has must
/// be supplied (every `ep_workers` entry `Some`, count equal) and nothing else.
///
/// Public for out-of-tree `Engine`/`Builder` implementers.
#[cfg(feature = "injected_streams")]
pub fn check_connect_streams(stage: LinkShape, links: &StreamLinks) -> EngineResult<()> {
    let got = links.shape();
    if got != stage || links.ep_workers.len() != stage.ep_workers {
        return Err(EngineError::PeerRejected(format!(
            "connect_streams links {got:?} (ep_workers entries: {}) do not match this stage {stage:?}",
            links.ep_workers.len()
        )));
    }
    Ok(())
}

/// Issue #76: the [`EngineError::PeerRejected`] for a re-attach that keeps a
/// link which can no longer carry frames: closed by this stage, dropped by the
/// transport after a dead-link error, or closed by its peer (the transport's
/// `is_connected()` is false). Keeping it would park the stage again on its
/// first frame, so every in-tree engine checks each kept link before
/// swapping anything and refuses with this error, naming the side.
#[cfg(feature = "injected_streams")]
pub fn kept_link_dead(side: &str) -> EngineError {
    EngineError::PeerRejected(format!(
        "reattach: the kept {side} link is dead (closed by this stage, dropped after a link \
         error, or closed by its peer); replace it too"
    ))
}

/// Issue #76: validate a `reattach_streams` call. At least one link must be
/// replaced, only links the stage has, and `ep_workers` must be empty (keep
/// all) or exactly as long as the stage's worker count (`None` = keep that
/// worker's link).
///
/// **Relay rule:** a stage that has an upstream link must replace it on every
/// re-attach. An idle relay's `step()` holds the engine lock while blocked on
/// its upstream frame-start read, and only closing THAT stream frees it; a
/// re-attach that keeps the upstream would wait behind that read. So any
/// re-attach cascades to the head, which also refreshes head-side session
/// state (prefix index, handshake, RESET). Heads (no upstream), EP workers
/// and EP drivers are unaffected.
///
/// Every failure is [`EngineError::PeerRejected`] (nothing swapped). Public
/// for out-of-tree `Engine`/`Builder` implementers; an engine may layer
/// stricter rules on top.
#[cfg(feature = "injected_streams")]
pub fn check_reattach_streams(stage: LinkShape, links: &StreamLinks) -> EngineResult<()> {
    let got = links.shape();
    if got == LinkShape::default() {
        return Err(EngineError::PeerRejected(
            "reattach_streams needs at least one replacement stream".into(),
        ));
    }
    let lacks = (got.upstream && !stage.upstream)
        || (got.downstream && !stage.downstream)
        || (got.ep_driver && !stage.ep_driver);
    if lacks {
        return Err(EngineError::PeerRejected(format!(
            "reattach_streams links {got:?} include a link this stage {stage:?} does not have"
        )));
    }
    if !links.ep_workers.is_empty() && links.ep_workers.len() != stage.ep_workers {
        return Err(EngineError::PeerRejected(format!(
            "reattach_streams got {} ep_workers entries; this stage has {} expert workers",
            links.ep_workers.len(),
            stage.ep_workers
        )));
    }
    if stage.upstream && !got.upstream {
        return Err(EngineError::PeerRejected(
            "reattach_streams must replace this stage's upstream link: an idle relay blocks \
             on its upstream read while holding the engine lock, and only closing that stream \
             frees it (re-attach the upstream link too, cascading to the head)"
                .into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("not yet loaded; call load() before build()")]
    NotLoaded,

    /// No usable peer link: the builder was never connected (call `connect()`
    /// or `connect_streams()` before `build()`), the link died, or (issue
    /// #76) the runner is fenced or dropped the request on a re-attach.
    ///
    /// Connection-fatal structurally (see [`EngineError::is_connection_fatal`]).
    /// The Display text deliberately avoids the classifier's "not connected"
    /// substring, so flattening this into a `Backend` string never changes how
    /// that string classifies.
    #[error("peer link unavailable: not yet connected, dead, or replaced")]
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
    ///
    /// `partner` is the tenant the PULLER asserted in its own GET — never `manifest.partner`, which
    /// the serving holder stamps and nothing validates (H.1b hard gate, design §12.10.0a). Keying on
    /// the echoed value lets a hostile or misconfigured holder return a blob stamped `tenant-b` for
    /// `tenant-a`'s pull: `tenant-a`'s warm resume silently goes cold, and `tenant-b`'s next
    /// NEGOTIATE against this node answers `Some((epoch, len))` for a prefix it never sent — the
    /// incremental length oracle H.1 exists to close, re-opened by a remote party.
    #[allow(clippy::result_unit_err)]
    fn insert(
        &mut self,
        partner: &str,
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

    /// Retract a parked slice when the head aborts a set this rank had already committed. `true` only
    /// if the slice was still parked; `false` means it is gone — the engine already took it, so this
    /// rank MAY be warm under a cold head — and the caller must report that, not treat the abort as
    /// clean.
    ///
    /// Deliberately no default: an impl that cannot retract has to say so in its own body.
    fn clear(&self, epoch: u64) -> bool;
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

    /// Issue #76: replace one or more links after a link failure and return
    /// the engine to a clean between-requests state. A `None` link (or `None`
    /// / absent `ep_workers` entry) keeps that link's current stream. Called
    /// with the runner's engine lock held, so no `step()` is in flight.
    ///
    /// Contract: return [`EngineError::PeerRejected`] ONLY for validation
    /// failures detected before any stream is swapped (engine untouched). Any
    /// failure after a swap must be a different variant; the runner fences
    /// the engine on those. Validate with [`check_reattach_streams`], which
    /// enforces the relay rule: a stage with an upstream link must replace it
    /// on every re-attach (an idle relay's `step()` is blocked on that link's
    /// read, so only closing it lets this call take the engine lock), and any
    /// re-attach therefore cascades to the head.
    #[cfg(feature = "injected_streams")]
    fn reattach_streams(&mut self, _links: StreamLinks) -> EngineResult<()> {
        Err(EngineError::PeerRejected(
            "this engine does not support re-attach".into(),
        ))
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

    /// Issue #76: wire up with already-connected streams instead of
    /// dialing/listening. Links must match this stage exactly (see
    /// [`check_connect_streams`]). `configure_listen` is ignored on this path.
    /// Streams must satisfy the contract on `cascadia_transport::InjectedStream`.
    #[cfg(feature = "injected_streams")]
    async fn connect_streams(&mut self, _links: StreamLinks) -> EngineResult<()> {
        Err(EngineError::PeerRejected(
            "this engine does not accept injected streams".into(),
        ))
    }

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

    #[cfg(feature = "injected_streams")]
    fn duplex_end() -> ByteStream {
        Box::new(tokio::io::duplex(64).0)
    }

    #[cfg(feature = "injected_streams")]
    #[test]
    fn stream_links_shape_reports_present_links() {
        let mut links = StreamLinks::ep_workers(vec![Some(duplex_end()), None, Some(duplex_end())]);
        links.upstream = Some(duplex_end());
        assert_eq!(
            links.shape(),
            LinkShape {
                upstream: true,
                downstream: false,
                ep_driver: false,
                ep_workers: 2
            }
        );
        assert_eq!(StreamLinks::default().shape(), LinkShape::default());
        assert_eq!(
            StreamLinks::pipeline(None, Some(duplex_end())).shape(),
            LinkShape::pipeline(false, true)
        );
        assert_eq!(
            StreamLinks::ep_driver(duplex_end()).shape(),
            LinkShape::ep_driver()
        );
        assert_eq!(LinkShape::ep_workers(3).ep_workers, 3);
    }

    #[cfg(feature = "injected_streams")]
    #[test]
    fn check_connect_streams_requires_an_exact_match() {
        let middle = LinkShape::pipeline(true, true);
        assert!(check_connect_streams(
            middle,
            &StreamLinks::pipeline(Some(duplex_end()), Some(duplex_end()))
        )
        .is_ok());
        assert!(matches!(
            check_connect_streams(middle, &StreamLinks::pipeline(Some(duplex_end()), None)),
            Err(EngineError::PeerRejected(_))
        ));
        // Standalone stage: nothing supplied.
        assert!(check_connect_streams(LinkShape::default(), &StreamLinks::default()).is_ok());
        assert!(check_connect_streams(
            LinkShape::default(),
            &StreamLinks::pipeline(None, Some(duplex_end()))
        )
        .is_err());
        // A pipeline stage refuses EP links.
        let mut ep_on_pipeline = StreamLinks::pipeline(Some(duplex_end()), Some(duplex_end()));
        ep_on_pipeline.ep_driver = Some(duplex_end());
        assert!(check_connect_streams(middle, &ep_on_pipeline).is_err());
        // EP worker: exactly the driver link.
        let worker = LinkShape::ep_driver();
        assert!(check_connect_streams(worker, &StreamLinks::ep_driver(duplex_end())).is_ok());
        assert!(check_connect_streams(worker, &StreamLinks::default()).is_err());
        // EP driver with 2 workers: both entries, both Some.
        let driver = LinkShape::ep_workers(2);
        let two = StreamLinks::ep_workers(vec![Some(duplex_end()), Some(duplex_end())]);
        assert!(check_connect_streams(driver, &two).is_ok());
        let short = StreamLinks::ep_workers(vec![Some(duplex_end())]);
        assert!(check_connect_streams(driver, &short).is_err());
        let hole = StreamLinks::ep_workers(vec![Some(duplex_end()), None]);
        assert!(check_connect_streams(driver, &hole).is_err());
    }

    #[cfg(feature = "injected_streams")]
    #[test]
    fn check_reattach_streams_rules() {
        let middle = LinkShape::pipeline(true, true);
        // relay: replace upstream (keep downstream) / replace both
        assert!(
            check_reattach_streams(middle, &StreamLinks::pipeline(Some(duplex_end()), None))
                .is_ok()
        );
        assert!(check_reattach_streams(
            middle,
            &StreamLinks::pipeline(Some(duplex_end()), Some(duplex_end()))
        )
        .is_ok());
        // replacing nothing
        assert!(matches!(
            check_reattach_streams(middle, &StreamLinks::default()),
            Err(EngineError::PeerRejected(_))
        ));
        // head (downstream only): may keep its (absent) upstream; an upstream
        // stream is a role error
        let head = LinkShape::pipeline(false, true);
        assert!(
            check_reattach_streams(head, &StreamLinks::pipeline(None, Some(duplex_end()))).is_ok()
        );
        assert!(
            check_reattach_streams(head, &StreamLinks::pipeline(Some(duplex_end()), None)).is_err()
        );
        // tail (upstream only)
        let tail = LinkShape::pipeline(true, false);
        assert!(
            check_reattach_streams(tail, &StreamLinks::pipeline(Some(duplex_end()), None)).is_ok()
        );
        // EP link on a pipeline stage
        assert!(check_reattach_streams(middle, &StreamLinks::ep_driver(duplex_end())).is_err());
        // EP worker: only the driver link
        let worker = LinkShape::ep_driver();
        assert!(check_reattach_streams(worker, &StreamLinks::ep_driver(duplex_end())).is_ok());
        assert!(
            check_reattach_streams(worker, &StreamLinks::pipeline(Some(duplex_end()), None))
                .is_err()
        );
        // EP driver with 3 workers: replace worker 1 only; wrong length; all None
        let driver = LinkShape::ep_workers(3);
        let one = StreamLinks::ep_workers(vec![None, Some(duplex_end()), None]);
        assert!(check_reattach_streams(driver, &one).is_ok());
        let wrong_len = StreamLinks::ep_workers(vec![Some(duplex_end())]);
        assert!(check_reattach_streams(driver, &wrong_len).is_err());
        let none = StreamLinks::ep_workers(vec![None, None, None]);
        assert!(check_reattach_streams(driver, &none).is_err());
    }

    #[cfg(feature = "injected_streams")]
    #[test]
    fn check_reattach_streams_rejects_a_relay_keeping_its_upstream() {
        // Relay rule: a stage with an upstream link must replace it on every
        // re-attach, so a downstream-only re-attach on a relay is refused.
        let middle = LinkShape::pipeline(true, true);
        match check_reattach_streams(middle, &StreamLinks::pipeline(None, Some(duplex_end()))) {
            Err(EngineError::PeerRejected(msg)) => {
                assert!(msg.contains("upstream"), "message explains the rule: {msg}");
            }
            other => panic!("expected PeerRejected, got {other:?}"),
        }
    }

    #[cfg(feature = "injected_streams")]
    #[test]
    fn engine_reattach_streams_defaults_to_rejection() {
        struct E;
        impl Engine for E {
            fn warmup(&mut self) {}
            fn submit(&mut self, _t: GenerationTask) -> EngineResult<()> {
                Ok(())
            }
            fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
                Ok(vec![])
            }
        }
        assert!(matches!(
            E.reattach_streams(StreamLinks::default()),
            Err(EngineError::PeerRejected(_))
        ));
    }

    #[cfg(feature = "injected_streams")]
    #[tokio::test]
    async fn builder_connect_streams_defaults_to_rejection() {
        struct NoStreams;
        #[async_trait]
        impl Builder for NoStreams {
            async fn connect(&mut self, _peers: PeerLayout) -> EngineResult<()> {
                Ok(())
            }
            async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
                Ok(Box::pin(futures::stream::empty()))
            }
            fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
                Err(EngineError::NotLoaded)
            }
        }
        let mut b = NoStreams;
        assert!(matches!(
            b.connect_streams(StreamLinks::default()).await,
            Err(EngineError::PeerRejected(_))
        ));
    }

    /// The two "frame-start" errors are NOT the same error, and conflating them is easy: both
    /// start with the same words and both mention a timeout. Only one is fatal.
    ///
    ///   frame-start WAIT TIMED OUT after <deadline> ... (retryable)  -> NON-fatal, retry on the
    ///       same socket. Zero bytes were consumed, so the frame stays aligned. A slow upstream
    ///       stage must NOT tear the chain down.
    ///   frame-start IDLE CEILING hit after <ceiling> ... dropped     -> FATAL. The connection was
    ///       actually dropped (black-holed peer); the loopback is gone.
    ///
    /// A rig report attributed a `worker_relay_connection_fatal` exit to the retryable one; the
    /// archived logs showed every such exit carried "idle ceiling" or "socket closed" instead.
    /// Pin both directions so a rewording cannot silently flip either.
    #[test]
    fn frame_start_retryable_is_not_confused_with_the_fatal_idle_ceiling() {
        let retryable = "frame-start wait timed out after 59.9999999s with no bytes (retryable)";
        let fatal =
            "frame-start idle ceiling hit after 900s; connection dropped (black-holed peer?)";

        assert!(
            !EngineError::Backend(retryable.into()).is_connection_fatal(),
            "the retryable frame-start wait must not exit the worker relay loop"
        );
        assert!(
            EngineError::Backend(fatal.into()).is_connection_fatal(),
            "the idle-ceiling drop is a real dead connection and must stay fatal"
        );
        // The discriminator is the substring, not the shared "frame-start"/"timed out" prefix.
        assert!(retryable.contains("frame-start") && fatal.contains("frame-start"));
        assert!(!retryable.contains("idle ceiling"));
    }

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
        // #40: the bounded frame-start token wait timing out is NON-fatal — it
        // flattens to this Backend string, which contains "timed out" but NOT
        // "recv_exact timed out", so it must classify recoverable (the caller
        // retries on the same live socket). Pin it so a reworded message can't
        // silently flip it to fatal + drop the once-dialed engine loopback.
        // The literal duration used to be hardcoded here as "120s", which never matched what
        // production emits: the message is formatted from the ACTUAL deadline
        // (`TransportError::FrameStartTimeout(Duration)` → `{0:?}`), and the deadline is
        // `base.min(TOKEN_RECV_DEADLINE_CEILING)` — so a 60s operator timeout logs ~60s while
        // the 120s ceiling only caps it. A reader comparing the log against this fixture would
        // conclude two deadlines disagreed. Assert across a range instead: the classification
        // must not depend on the number at all.
        for secs in [1u64, 60, 120, 900] {
            let msg = format!("frame-start wait timed out after {secs}s with no bytes (retryable)");
            assert!(
                !EngineError::Backend(msg.clone()).is_connection_fatal(),
                "retryable frame-start timeout must NOT be connection-fatal: {msg}"
            );
        }
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
