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

#[cfg(feature = "kv_coord")]
pub mod kv_handoff;
#[cfg(feature = "kv_coord")]
pub use kv_handoff::{KvHandoffMailbox, KvHandoffSlot};

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
    /// the API must answer OpenAI's 400 `context_length_exceeded` rather than
    /// a retryable 5xx.
    #[error("prompt too long: {0}")]
    PromptTooLong(String),

    /// The request carries input this engine cannot consume (today: images
    /// on a text-only engine or tree). Permanent for the request, like
    /// `PromptTooLong`, so the API answers 400 rather than a retryable 5xx.
    #[error("unsupported input: {0}")]
    UnsupportedInput(String),

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

/// The [`GenerationTask::media`] contract for an engine that cannot consume
/// media: fail the task rather than answer from its text alone. Call first in
/// `submit`; `engine` names the engine in the message the caller sees.
pub fn refuse_media(task: &GenerationTask, engine: &str) -> EngineResult<()> {
    if task.media.is_empty() {
        return Ok(());
    }
    Err(EngineError::UnsupportedInput(format!(
        "the {engine} engine does not accept image input ({} image(s) in the request)",
        task.media.len()
    )))
}

/// The [`MediaKind::Video`](cascadia_types::MediaKind::Video) contract for an
/// engine that takes images but not video: fail the task rather than read a
/// frame stack as one image. The API sends such an engine a video's frames as
/// images, so this only fires on a routing bug. Call first in `submit`.
pub fn refuse_video(task: &GenerationTask, engine: &str) -> EngineResult<()> {
    let videos = task
        .media
        .iter()
        .filter(|m| matches!(m.kind, cascadia_types::MediaKind::Video))
        .count();
    if videos == 0 {
        return Ok(());
    }
    Err(EngineError::UnsupportedInput(format!(
        "the {engine} engine does not accept native video input ({videos} video(s) in the \
         request); its videos are served as frames"
    )))
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

    /// The HTTP status a request rejected with this error gets. One table for
    /// every caller that fronts an engine (cascadia-api's server, and any
    /// gateway that submits tasks itself), so they classify alike. Permanent
    /// rejections are 4xx: retrying the identical request (on this or any peer)
    /// cannot succeed. OpenAI answers an over-context prompt with a 400 and its
    /// SDKs retry neither a 400 nor a 413. Exhaustive on purpose: a new variant
    /// must be classified here before it compiles.
    pub fn http_status(&self) -> u16 {
        match self {
            EngineError::QueueFull { .. } | EngineError::NotLoaded | EngineError::NotConnected => {
                503
            }
            EngineError::PromptTooLong(_)
            | EngineError::InvalidConfig(_)
            | EngineError::PeerRejected(_)
            | EngineError::ShardRejected(_)
            | EngineError::UnsupportedInput(_) => 400,
            EngineError::ModelNotFound(_) => 404,
            EngineError::Backend(_)
            | EngineError::Io(_)
            | EngineError::Task { .. }
            | EngineError::BatchAborted(_) => 500,
        }
    }

    /// The request is at fault (a 4xx [`http_status`](Self::http_status)): a
    /// caller must not retry it unchanged or count it against the peer's health.
    pub fn is_client_error(&self) -> bool {
        (400..500).contains(&self.http_status())
    }

    /// The rejection-metric reason, named for the knob that fixes it:
    /// `capacity`, `prompt_over_window` (a sizing change), `engine_unavailable`
    /// (bring the stage or its peer up), `engine_error` (read that node's logs),
    /// `invalid_request` (the caller sent something this worker can't serve).
    pub fn rejection_reason(&self) -> &'static str {
        match self {
            EngineError::QueueFull { .. } => "capacity",
            EngineError::PromptTooLong(_) => "prompt_over_window",
            EngineError::NotLoaded | EngineError::NotConnected => "engine_unavailable",
            EngineError::Backend(_)
            | EngineError::Io(_)
            | EngineError::Task { .. }
            | EngineError::BatchAborted(_) => "engine_error",
            EngineError::InvalidConfig(_)
            | EngineError::PeerRejected(_)
            | EngineError::ShardRejected(_)
            | EngineError::UnsupportedInput(_)
            | EngineError::ModelNotFound(_) => "invalid_request",
        }
    }
}

/// Prefix of a [`Chunk::error`](cascadia_types::Chunk::error) reason that the
/// request caused after admission (a malformed resume prefix, an empty prompt):
/// permanent for the request, so not a reason to retry it or demote the peer.
pub const CLIENT_ERROR_PREFIX: &str = "invalid_request: ";

/// Whether a stream's error reason (a [`Chunk`](cascadia_types::Chunk)'s
/// `error`) is the request's fault. Engines refuse a request they cannot serve
/// at submit with a typed [`EngineError`] (use
/// [`is_client_error`](EngineError::is_client_error) there); a refusal that
/// reaches a stream as text carries one of these stable prefixes: an
/// [`EngineError`]'s own `Display` prefix for a client variant (`prompt too
/// long:`, `unsupported input:`, ...), [`CLIENT_ERROR_PREFIX`], or
/// `resume_unsupported:`, optionally behind a `task <id>: ` attribution and a
/// `backend error: ` wrapper.
pub fn is_client_error_reason(reason: &str) -> bool {
    const PREFIXES: [&str; 8] = [
        CLIENT_ERROR_PREFIX,
        "resume_unsupported:",
        "prompt too long: ",
        "unsupported input: ",
        "invalid configuration: ",
        "peer layout rejected: ",
        "shard rejected: ",
        "model not found at ",
    ];
    let mut r = reason;
    if let Some(rest) = r.strip_prefix("task ") {
        r = rest.split_once(": ").map_or(r, |(_, after)| after);
    }
    // A step-path refusal reaches the stream as a `Backend` error's text.
    if let Some(rest) = r.strip_prefix("backend error: ") {
        r = rest;
    }
    PREFIXES.iter().any(|p| r.starts_with(p))
}

/// Why GenAI cannot serve this VLM export's images correctly, if it cannot.
///
/// GenAI 2026.4.1 has no Pixtral path: `model_type: "llava"` maps a Pixtral
/// checkpoint onto its LLaVA-1.5 preprocessing (`visual_language/llava/
/// classes.cpp`), which resizes the shortest edge and center-crops a fixed
/// square and expands each image without Pixtral's `[IMG_BREAK]` row tokens
/// and `[IMG_END]`, behind a hard-coded `<image>` tag that the Pixtral
/// tokenizer splits into ordinary text tokens, so no image row is merged at
/// all. The model then answers about a different picture (903 prompt tokens
/// where HF counts 832 on a 448x448 image). Such exports serve text only here; the message names the
/// ov-runtime commands that serve their images, and the API returns it with
/// the 400 that refuses them.
pub fn genai_unsupported_vision(model_dir: &std::path::Path) -> Option<String> {
    let cfg: serde_json::Value = std::fs::read(model_dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())?;
    let vision = cfg
        .get("vision_config")
        .and_then(|v| v.get("model_type"))
        .and_then(serde_json::Value::as_str)?;
    // Mistral 3 exports carry a Pixtral vision tower too; their cut needs
    // `--family mistral3`.
    let family = match cfg.get("model_type").and_then(serde_json::Value::as_str) {
        Some("mistral3") => "mistral3",
        _ => "pixtral",
    };
    (vision == "pixtral").then(|| {
        format!(
            "ov-genai serves this Pixtral export as text only: GenAI preprocesses Pixtral as \
             LLaVA-1.5 (center crop, no [IMG_BREAK]/[IMG_END]) and answers about a different \
             image. Serve its images with ov-runtime on a one-stage tree: \
             `python tools/vlm_surgery/cut_vlm_lm.py --model <export dir> --output-dir <tree> \
             --num-stages 1 --family {family}`, then `cascadia worker --total 1 --rank 0 \
             --engine ov-runtime --model <tree> --api <host:port>`"
        )
    })
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

#[cfg(test)]
mod refuse_media_tests {
    use super::*;

    #[test]
    fn text_tasks_pass_and_image_tasks_are_refused_with_the_engine_named() {
        let mut t = GenerationTask::new("t", "hello");
        assert!(refuse_media(&t, "x").is_ok());
        t.media.push(cascadia_types::MediaInput {
            kind: cascadia_types::MediaKind::Image,
            pixels: std::sync::Arc::from(vec![0u8; 3]),
            num_frames: 1,
            width: 1,
            height: 1,
            timestamps_ms: None,
            sha256: [0; 32],
        });
        let e = refuse_media(&t, "gemma4").unwrap_err();
        assert!(matches!(e, EngineError::UnsupportedInput(_)));
        assert!(
            e.to_string()
                .contains("gemma4 engine does not accept image input"),
            "{e}"
        );
        assert!(!e.is_connection_fatal());
    }

    #[test]
    fn image_tasks_pass_and_video_tasks_are_refused() {
        let mut t = GenerationTask::new("t", "hello");
        let item = |kind, num_frames| cascadia_types::MediaInput {
            kind,
            pixels: std::sync::Arc::from(vec![0u8; 3 * num_frames as usize]),
            num_frames,
            width: 1,
            height: 1,
            timestamps_ms: None,
            sha256: [0; 32],
        };
        assert!(refuse_video(&t, "x").is_ok());
        t.media.push(item(cascadia_types::MediaKind::Image, 1));
        assert!(refuse_video(&t, "x").is_ok());
        t.media.push(item(cascadia_types::MediaKind::Video, 4));
        let e = refuse_video(&t, "gemma4").unwrap_err();
        assert!(matches!(e, EngineError::UnsupportedInput(_)));
        assert!(
            e.to_string()
                .contains("gemma4 engine does not accept native video input (1 video(s)"),
            "{e}"
        );
        assert!(!e.is_connection_fatal());
    }
}

#[cfg(test)]
mod client_error_tests {
    use super::*;

    #[test]
    fn statuses_and_client_faults_follow_one_table() {
        let client = [
            EngineError::PromptTooLong("x".into()),
            EngineError::UnsupportedInput("x".into()),
            EngineError::InvalidConfig("x".into()),
            EngineError::PeerRejected("x".into()),
            EngineError::ShardRejected("x".into()),
            EngineError::ModelNotFound("x".into()),
        ];
        for e in &client {
            assert!(e.is_client_error(), "{e}");
            assert!(is_client_error_reason(&e.to_string()), "{e}");
        }
        assert_eq!(EngineError::PromptTooLong("x".into()).http_status(), 400);
        assert_eq!(EngineError::ModelNotFound("x".into()).http_status(), 404);
        let server = [
            EngineError::QueueFull { queued: 1, cap: 1 },
            EngineError::NotLoaded,
            EngineError::NotConnected,
            EngineError::Backend("boom".into()),
            EngineError::BatchAborted("x".into()),
            EngineError::Task {
                task_id: "t".into(),
                source: Box::new(EngineError::Backend("x".into())),
            },
        ];
        for e in &server {
            assert!(!e.is_client_error(), "{e}");
            assert!(!is_client_error_reason(&e.to_string()), "{e}");
        }
        assert_eq!(
            EngineError::QueueFull { queued: 1, cap: 1 }.http_status(),
            503
        );
        assert_eq!(EngineError::Backend("x".into()).http_status(), 500);
        assert_eq!(
            EngineError::PromptTooLong("x".into()).rejection_reason(),
            "prompt_over_window"
        );
    }

    /// The texts engines put on a stream for a request's own fault, as they
    /// reach a caller: bare, attributed to a task, or wrapped as a step's
    /// backend error.
    #[test]
    fn stream_reasons_from_client_faults_are_recognised() {
        let empty = EngineError::Backend(format!(
            "{CLIENT_ERROR_PREFIX}empty prompt: no tokens to prefill"
        ))
        .for_task("t-1".into());
        for r in [
            format!("{CLIENT_ERROR_PREFIX}invalid resume prefix: id 9 out of range"),
            format!("{CLIENT_ERROR_PREFIX}empty prompt after tokenize"),
            empty.to_string(),
            "unsupported input: resume_unsupported: a resumed turn cannot carry images".into(),
            "resume_unsupported: ov-genai has no forced-prefix path".into(),
            "task t-2: prompt too long: prompt is at least 5875 tokens".into(),
        ] {
            assert!(is_client_error_reason(&r), "{r}");
        }
        for r in [
            "tokenize failed: bad utf8",
            "image rows missing at admission",
            "task t-3: backend error: stage 1 hung up",
            "batch aborted: NACK",
        ] {
            assert!(!is_client_error_reason(r), "{r}");
        }
    }
}
