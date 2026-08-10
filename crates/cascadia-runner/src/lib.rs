//! Per-stage Runner.
//!
//! Mirrors `cascadia/worker/runner.py`. Lifecycle:
//!
//! 1. [`Runner::start`] — connect transport, load weights, build engine, warm up.
//! 2a. (first stage)  [`Runner::generate`] — submit task and stream chunks.
//! 2b. (other stages) [`Runner::run_relay_loop`] — drive engine forever.
//! 3. [`Runner::close`].
//!
//! `generate()` is safe to call concurrently. Each call shares the engine
//! through a single mutex; chunks for other tasks emitted during *our*
//! `step()` turns are buffered for their owners, and vice versa.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use cascadia_engine::{Builder, Engine, EngineError};
use cascadia_types::{Chunk, FinishReason, GenerationTask, PeerLayout, ShardSpec, TaskId};
use futures::Stream;
use parking_lot::Mutex;
use tracing::{info, warn};

/// When `generate()` sees this many consecutive empty steps with no new
/// chunks for *any* task, it returns rather than block forever on a
/// misbehaving engine.
const MAX_CONSECUTIVE_EMPTY_STEPS: usize = 3;

/// Cool-off `run_relay_loop` applies after a `step()` returns `Err`, so a
/// persistently-failing engine (dead peer, bad-frame flood, black-holed
/// upstream) is throttled regardless of whether the engine self-throttles.
/// Matches the cadence the sparse-moe / dist_spec workers use for the same
/// situation (`WORKER_BACKOFF`).
const RELAY_ERR_BACKOFF: std::time::Duration = std::time::Duration::from_millis(200);

/// Why [`Runner::run_relay_loop`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelayExit {
    /// The engine slot emptied (clean teardown via [`Runner::close`]).
    SlotEmpty,
    /// A `step()` hit an unrecoverable peer-link failure; the caller should
    /// exit non-zero so the supervisor rebuilds the stage.
    ConnectionFatal,
}

// ---------------------------------------------------------------------------
// Shared block_on dispatch + BlockingContextGuard.
//
// Engines call `run_async(handle, fut)` to bridge sync `Engine::step` code
// to async transport futures. There are two contexts:
//
// * **Driver via `ChunkStream::poll_next`** — running on a tokio worker
//   thread that's actively polling an async task. Naked `Handle::block_on`
//   panics with "Cannot start a runtime from within a runtime"; we need
//   `block_in_place` to migrate other tasks off this worker first.
//
// * **Worker via `Runner::run_relay_loop`**, dispatched through
//   `tokio::task::spawn_blocking`. Spawn_blocking threads are NOT polling
//   tasks; `Handle::block_on` works directly. Wrapping with
//   `block_in_place` is unnecessary AND expensive on Windows
//   (empirically ~20 ms per call vs ~5–30 µs the docs would suggest;
//   adds ~60 ms per worker frame round-trip with 3 wire I/O calls).
//
// `run_relay_loop` enters a `BlockingContextGuard` automatically before
// each `step()` so engines get the fast path on workers without any code
// of their own. The driver path leaves the flag clear, so `run_async`
// falls through to `block_in_place + block_on` and stays safe.
// ---------------------------------------------------------------------------

thread_local! {
    static BLOCKING_CONTEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII guard marking the current thread as a `spawn_blocking` worker.
/// `run_async` consults this flag to pick the fastest safe `block_on`
/// variant. Scoped so that if the spawn_blocking thread pool reuses this
/// OS thread for non-blocking work later, the flag resets to its prior
/// value automatically.
pub struct BlockingContextGuard {
    prev: bool,
}

impl BlockingContextGuard {
    pub fn enter() -> Self {
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

/// Bridge sync engine code to an async transport future. Engines should
/// call this instead of `Handle::block_on` directly.
pub fn run_async<F: std::future::Future>(handle: &tokio::runtime::Handle, fut: F) -> F::Output {
    if BLOCKING_CONTEXT.with(|f| f.get()) {
        handle.block_on(fut)
    } else if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(|| handle.block_on(fut))
    } else {
        handle.block_on(fut)
    }
}

#[derive(Default)]
struct Buffers {
    chunks: HashMap<TaskId, VecDeque<Chunk>>,
    cancelled: std::collections::HashSet<TaskId>,
    /// Streams parked because another caller held the engine (#122). Every
    /// engine-lock release drains + wakes these, so a parked stream re-polls
    /// as soon as the engine may be free. Registration happens ONLY after a
    /// failed `try_lock`, and the registrant re-checks the lock afterwards,
    /// so a release can't slip between the check and the registration.
    step_wakers: Vec<std::task::Waker>,
    /// Cancels that could not take the engine lock without blocking a
    /// worker thread (#122). Applied by the next engine-lock holder before
    /// it steps — the same effective timing a blocking cancel had, since a
    /// running step can't be interrupted anyway.
    deferred_cancels: Vec<TaskId>,
}

/// Drains + wakes [`Buffers::step_wakers`] when it leaves scope — including
/// when a panic unwinds out of that scope.
///
/// The drain after an engine-lock release is straight-line code, so an
/// unwind skips it: a panic inside `step()` (or anything else run under the
/// engine guard) leaves every parked stream `Pending` forever with nothing
/// in the log to say why. `parking_lot` does not poison, so the engine
/// itself is released and perfectly usable — there is simply no one left to
/// poll it. Running the drain from a destructor covers the unwind path too.
///
/// DECLARE THIS BEFORE the engine guard it protects: locals drop in reverse
/// declaration order, so the engine lock is released first and the wake runs
/// with the engine free. Waking while the lock is still held lets a woken
/// stream re-park behind it and be stranded exactly as before.
struct WakeParkedOnDrop<'a> {
    buffers: &'a Mutex<Buffers>,
    /// Armed only once the engine lock is actually held. A contended
    /// `poll_next` that gives up and returns `Pending` has just registered
    /// its OWN waker; draining from there would wake it straight back and
    /// spin the worker for the whole of the holder's step.
    armed: bool,
}

impl<'a> WakeParkedOnDrop<'a> {
    /// Inert until [`arm`](Self::arm) is called.
    fn disarmed(buffers: &'a Mutex<Buffers>) -> Self {
        Self {
            buffers,
            armed: false,
        }
    }

    fn arm(&mut self) {
        self.armed = true;
    }
}

impl Drop for WakeParkedOnDrop<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let wakers = std::mem::take(&mut self.buffers.lock().step_wakers);
        for w in wakers {
            w.wake();
        }
    }
}

pub struct Runner {
    /// Mutex so the Runner is `Sync` even with a `dyn Builder` inside;
    /// taken once during `start()` and dropped.
    builder: Mutex<Option<Box<dyn Builder>>>,
    /// `Mutex<Option<...>>` so `close()` can drop the engine while other
    /// callers hold references to the runner; subsequent calls fail
    /// cleanly with [`EngineError::NotLoaded`].
    engine: Arc<Mutex<Option<Box<dyn Engine>>>>,
    buffers: Arc<Mutex<Buffers>>,
    /// Model id captured from the [`ShardSpec`] at `start()`, used as the
    /// `model` label on generation metrics (#16). `None` until started.
    model: Mutex<Option<Arc<str>>>,
    /// Set by `close()` BEFORE the engine slot is emptied, so an in-flight
    /// generation can tell "the server is shutting down" from "my client
    /// hung up". Without it the outcome depends on an unsynchronised race
    /// between a stream's next poll and its `Drop`, and every restart books
    /// a nondeterministic number of client cancellations.
    closing: Arc<AtomicBool>,
}

impl Runner {
    pub fn new(builder: Box<dyn Builder>) -> Self {
        Self {
            builder: Mutex::new(Some(builder)),
            engine: Arc::new(Mutex::new(None)),
            buffers: Arc::new(Mutex::new(Buffers::default())),
            model: Mutex::new(None),
            closing: Arc::new(AtomicBool::new(false)),
        }
    }

    /// `model` label for generation metrics; "unknown" before `start()`.
    fn model_label(&self) -> Arc<str> {
        self.model
            .lock()
            .clone()
            .unwrap_or_else(|| Arc::from("unknown"))
    }

    /// Connect transports, load weights, build engine, warm up. Pass
    /// `listen_addr = Some((host, port))` for stages that need to bind a
    /// listener for an upstream peer (non-first stages); `None` is fine
    /// for single-stage / first-stage engines.
    pub async fn start_with_listen(
        &self,
        peers: PeerLayout,
        shard: ShardSpec,
        listen_addr: Option<(&str, u16)>,
    ) -> Result<(), EngineError> {
        let mut builder = self.builder.lock().take().ok_or(EngineError::NotLoaded)?;
        if let Some((host, port)) = listen_addr {
            builder.configure_listen(host, port);
        }
        info!(
            upstream = ?peers.upstream,
            downstream = ?peers.downstream,
            "runner connect"
        );
        builder.connect(peers).await?;

        info!("runner load");
        let model = shard.model_id.clone();
        let device = shard.device.clone();
        let load_started = Instant::now();
        let mut load_stream = builder.load(shard).await?;
        // drain the load progress stream
        use futures::StreamExt;
        while let Some(progress) = load_stream.next().await {
            info!(message = %progress.message, "load progress");
        }

        info!("runner build engine");
        let mut engine = builder.build()?;
        cascadia_metrics::ENGINE_LOAD_DURATION_SECONDS
            .with_label_values(&[&model, &device])
            .set(load_started.elapsed().as_secs_f64());
        info!("runner warmup");
        let warmup_started = Instant::now();
        engine.warmup();
        cascadia_metrics::ENGINE_WARMUP_DURATION_SECONDS
            .with_label_values(&[&model, &device])
            .set(warmup_started.elapsed().as_secs_f64());
        info!("runner ready");

        *self.model.lock() = Some(Arc::from(model));
        *self.engine.lock() = Some(engine);
        Ok(())
    }

    /// Backwards-compatible shortcut equivalent to
    /// `start_with_listen(peers, shard, None)`. Engines that need a
    /// listener (workers, middle stages) should call
    /// [`start_with_listen`] with the correct address.
    pub async fn start(&self, peers: PeerLayout, shard: ShardSpec) -> Result<(), EngineError> {
        self.start_with_listen(peers, shard, None).await
    }

    pub fn submit(&self, task: GenerationTask) -> Result<(), EngineError> {
        // NOTE: blocks the calling thread for up to one engine step if a
        // step is in flight. Async callers must use [`Runner::generate`]'s
        // async submit path (or their own `spawn_blocking`) — a worker
        // thread blocked here counts toward the driver starvation that
        // wedges the pipeline (#122).
        let res = {
            let mut guard = self.engine.lock();
            // Route the empty slot through `res` rather than `?`: an early
            // return here released the engine lock and skipped the wake
            // below, and an empty slot means `close()` already ran — no
            // later lock holder is coming to wake them instead.
            match guard.as_mut() {
                Some(engine) => engine.submit(task),
                None => Err(EngineError::NotLoaded),
            }
        };
        self.wake_parked_streams();
        res
    }

    /// Drain + wake every stream that parked on engine-lock contention.
    /// Called after every engine-lock release (#122): a parked stream is
    /// only ever woken from here, so skipping a release site would strand it.
    fn wake_parked_streams(&self) {
        let wakers = std::mem::take(&mut self.buffers.lock().step_wakers);
        for w in wakers {
            w.wake();
        }
    }

    /// Cooperatively cancel a task.
    pub fn cancel(&self, task_id: &TaskId) {
        // Lock order is engine-before-buffers everywhere (ChunkStream::poll_next
        // holds both while distributing a step). Taking buffers first here and
        // then reaching for the engine would invert that and deadlock, so the
        // tombstone is written and released before the engine is touched. A
        // chunk that lands in between is dropped by the cancelled-check on the
        // distribution path, which is the behaviour either way.
        {
            let mut bufs = self.buffers.lock();
            bufs.cancelled.insert(task_id.clone());
            bufs.chunks.remove(task_id);
        }
        // Never block a (possibly async) caller on the engine mutex — a
        // worker thread parked here counts toward the driver starvation
        // that wedges the pipeline (#122). If a step is in flight, defer:
        // the next engine-lock holder applies it before stepping, which is
        // when a blocking cancel would have run anyway.
        match self.engine.try_lock() {
            Some(mut guard) => {
                if let Some(engine) = guard.as_mut() {
                    engine.cancel(task_id);
                }
                drop(guard);
                self.wake_parked_streams();
            }
            None => {
                self.buffers.lock().deferred_cancels.push(task_id.clone());
            }
        }
    }

    /// Submit a task and return a stream of chunks. Stops on the final
    /// chunk, on cancellation, on an engine step error, or after
    /// MAX_CONSECUTIVE_EMPTY_STEPS empty engine polls (engine appears
    /// stuck).
    pub fn generate(&self, task: GenerationTask) -> Result<ChunkStream, EngineError> {
        self.submit(task.clone())?;
        Ok(self.stream_for(task.task_id))
    }

    /// Async [`Runner::generate`]: submits off the runtime via
    /// `spawn_blocking`. Async servers must use this — `submit` blocks on
    /// the engine mutex for up to a full engine step, and enough worker
    /// threads parked there starve the tokio I/O + timer drivers, which is
    /// the #122 pipeline wedge.
    pub async fn generate_async(
        self: &Arc<Self>,
        task: GenerationTask,
    ) -> Result<ChunkStream, EngineError> {
        let this = self.clone();
        let submitted = task.clone();
        tokio::task::spawn_blocking(move || this.submit(submitted))
            .await
            .map_err(|e| EngineError::Backend(format!("submit task join: {e}")))??;
        Ok(self.stream_for(task.task_id))
    }

    fn stream_for(&self, task_id: TaskId) -> ChunkStream {
        ChunkStream {
            task_id,
            engine: self.engine.clone(),
            buffers: self.buffers.clone(),
            consecutive_empty: 0,
            last_errored_task: None,
            consecutive_foreign_err: 0,
            done: false,
            model: self.model_label(),
            submitted_at: Instant::now(),
            last_chunk_at: None,
            metrics_finalized: false,
            closing: self.closing.clone(),
        }
    }

    /// Step the engine forever; exits when the engine slot empties (clean
    /// teardown) or the engine's peer link dies unrecoverably. Used by
    /// non-first pipeline stages.
    ///
    /// A `step()` that returns a *connection-fatal* `Err` ([dead/dropped
    /// socket, idle-ceiling fire](EngineError::is_connection_fatal)) ends
    /// the loop with [`RelayExit::ConnectionFatal`]: the worker's upstream
    /// socket can only be re-accepted by a rebuild, so the caller exits
    /// non-zero and lets the supervisor (systemd `Restart=on-failure`)
    /// rebuild the stage — rather than spin-and-flood at the backoff rate
    /// forever. A *non-fatal* `Err` (bad frame kind, transient inference
    /// failure) is logged and driving continues, throttled by
    /// `RELAY_ERR_BACKOFF` so it can't peg a core; the backoff resets on
    /// any non-empty `Ok` step. Engines MAY additionally self-throttle.
    ///
    /// Enters a `BlockingContextGuard` once per OS thread (since this
    /// loop runs on a single `spawn_blocking` thread) so that engines'
    /// `run_async` calls hit the naked-`block_on` path instead of
    /// `block_in_place` — ~60 ms/frame savings on Windows.
    pub fn run_relay_loop(&self) -> RelayExit {
        let _blocking = BlockingContextGuard::enter();
        loop {
            let mut guard = self.engine.lock();
            let Some(engine) = guard.as_mut() else {
                drop(guard);
                info!("relay loop exited: engine slot empty");
                return RelayExit::SlotEmpty;
            };
            // Engine.step is sync; just drain. A non-fatal Err is logged
            // but the loop keeps driving — relay engines recover their own
            // state and a transient frame error must not take the stage
            // down. A connection-fatal Err is unrecoverable in-process, so
            // bail and let the supervisor rebuild us.
            let failed = match engine.step() {
                Ok(_) => false,
                Err(e) if e.is_connection_fatal() => {
                    drop(guard);
                    warn!(error = %e, "relay step hit a dead peer link; exiting for supervisor rebuild");
                    return RelayExit::ConnectionFatal;
                }
                Err(e) => {
                    warn!(error = %e, "relay step failed");
                    true
                }
            };
            // Don't hold the lock for long under the loop — yield to
            // other generate() callers between rounds.
            drop(guard);
            // Throttle a persistently-failing step() so it can't hot-spin
            // this thread; a successful round (even an idle empty one)
            // just yields, keeping relay latency low.
            if failed {
                std::thread::sleep(RELAY_ERR_BACKOFF);
            } else {
                std::thread::yield_now();
            }
        }
    }

    pub fn close(&self) {
        // BEFORE the slot empties, so no in-flight stream can observe a gone
        // engine without also seeing that this is a shutdown. Ordering is the
        // whole point: set it after, and a stream that polls in between books
        // a client cancellation for a server restart.
        self.closing.store(true, Ordering::SeqCst);
        if let Some(engine) = self.engine.lock().as_mut() {
            engine.close();
        }
        *self.engine.lock() = None;
        // Streams parked on engine-lock contention must re-poll to observe
        // the emptied slot, or a teardown would strand them Pending forever.
        self.wake_parked_streams();
        if let Some(builder) = self.builder.lock().as_mut() {
            builder.close();
        }
    }
}

pub struct ChunkStream {
    task_id: TaskId,
    engine: Arc<Mutex<Option<Box<dyn Engine>>>>,
    buffers: Arc<Mutex<Buffers>>,
    consecutive_empty: usize,
    /// Foreign-task hot-spin guard: the last foreign task-id an engine `step()`
    /// failed, and how many times in a row it has failed *that same* id. A
    /// contract-violating engine that re-emits the same foreign Err forever
    /// (never clearing it) trips the bound; distinct foreign failures mean the
    /// engine IS clearing each and progressing, so the run resets. Separate
    /// from `consecutive_empty` so a healthy task isn't false-closed by a few
    /// distinct foreign failures it observes while waiting its turn.
    last_errored_task: Option<TaskId>,
    consecutive_foreign_err: usize,
    done: bool,
    /// `model` label for the generation metrics below (#16).
    model: Arc<str>,
    /// When the task was submitted — the zero point for TTFT and duration.
    submitted_at: Instant,
    /// When the previous chunk was delivered (inter-token latency).
    last_chunk_at: Option<Instant>,
    /// A terminal outcome (completed / failed / cancelled / teardown) has
    /// been recorded; guards exactly-once accounting between the
    /// final-chunk paths, the cancel path, and `Drop`.
    metrics_finalized: bool,
    /// Shared with the [`Runner`]: set before `close()` empties the engine
    /// slot, so an abandoned generation is attributed to the shutdown
    /// rather than to the client regardless of poll/drop ordering.
    closing: Arc<AtomicBool>,
}

impl ChunkStream {
    /// Record per-chunk generation metrics for a chunk delivered to OUR
    /// consumer (never for chunks routed to other tasks' buffers).
    fn record_chunk_metrics(&mut self, chunk: &Chunk) {
        if self.metrics_finalized {
            return;
        }
        let now = Instant::now();
        if chunk.error.is_some() {
            cascadia_metrics::TASKS_FAILED_TOTAL
                .with_label_values(&[&self.model])
                .inc();
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[&self.model, "error"])
                .observe((now - self.submitted_at).as_secs_f64());
            self.metrics_finalized = true;
            return;
        }
        // Same convention as the API's usage accounting (Chunk::token_count):
        // `n_tokens` is authoritative when set; otherwise one token per
        // non-empty chunk, so the empty final markers contribute 0.
        let n = chunk.token_count();
        // Only token-bearing chunks are timing samples: the separate empty
        // final marker most engines emit would otherwise add one artificial
        // inter-token gap per generation (and a zero-token generation would
        // record a bogus TTFT).
        if n > 0 {
            match self.last_chunk_at {
                None => cascadia_metrics::GENERATION_TTFT_SECONDS
                    .with_label_values(&[&self.model])
                    .observe((now - self.submitted_at).as_secs_f64()),
                Some(prev) => cascadia_metrics::GENERATION_INTER_TOKEN_SECONDS
                    .with_label_values(&[&self.model])
                    .observe((now - prev).as_secs_f64()),
            }
            self.last_chunk_at = Some(now);
            cascadia_metrics::TOKENS_GENERATED_TOTAL
                .with_label_values(&[&self.model])
                .inc_by(n as u64);
        }
        if chunk.is_final {
            if let Some(p) = chunk.prompt_tokens {
                cascadia_metrics::TOKENS_PROMPT_TOTAL
                    .with_label_values(&[&self.model])
                    .inc_by(p as u64);
            }
            // `Cancelled` is surfaced here (metrics-only per FinishReason's
            // contract); the API maps it to "stop" on the wire. An engine
            // that acknowledges a cancel with a Cancelled final marker must
            // land in the SAME counter as the tombstone/Drop cancel paths,
            // or the cancelled duration histogram and cancelled counter
            // would diverge.
            let reason = match chunk.finish_reason {
                Some(FinishReason::Length) => "length",
                Some(FinishReason::Cancelled) => {
                    cascadia_metrics::TASKS_CANCELLED_TOTAL
                        .with_label_values(&[&self.model])
                        .inc();
                    "cancelled"
                }
                Some(FinishReason::Stop) | None => "stop",
            };
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[&self.model, reason])
                .observe((now - self.submitted_at).as_secs_f64());
            self.metrics_finalized = true;
        }
    }

    /// Terminal failure of OUR stream: close the buffers entry, build the
    /// final error chunk, and record it — one place, so every error-return
    /// site in `poll_next` gets terminal accounting by construction.
    fn fail_stream(&mut self, reason: String) -> Poll<Option<Chunk>> {
        self.done = true;
        self.buffers.lock().chunks.remove(&self.task_id);
        let chunk = Chunk::error(self.task_id.clone(), reason);
        self.record_chunk_metrics(&chunk);
        Poll::Ready(Some(chunk))
    }

    /// The engine slot emptied under us: the server is shutting down.
    ///
    /// Ends the stream with an ERROR chunk, not a bare `None`. A bare
    /// end-of-stream is indistinguishable from a completed generation, so
    /// the API built a 200 carrying the partial text with
    /// `finish_reason: "stop"` — telling the client the model finished
    /// normally when the server was actually going away. The error chunk
    /// routes through the existing 503 / SSE-error paths instead.
    ///
    /// Deliberately NOT `fail_stream`: that books `tasks_failed_total`, and
    /// a planned restart must not land on the primary failure SLO.
    fn fail_teardown(&mut self) -> Poll<Option<Chunk>> {
        self.done = true;
        self.buffers.lock().chunks.remove(&self.task_id);
        warn!(
            task = %self.task_id,
            "engine slot emptied mid-generation (server teardown); failing the \
             request rather than returning a truncated success"
        );
        self.record_teardown_metrics();
        Poll::Ready(Some(Chunk::error(
            self.task_id.clone(),
            "server is shutting down".to_string(),
        )))
    }

    /// Record an abandoned generation: explicit cancel, or the consumer
    /// dropped the stream before the final chunk.
    ///
    /// A shutdown wins over both. `close()` sets `closing` before it empties
    /// the engine slot, so this is decided by a flag rather than by whether
    /// the stream happened to be polled once more before being dropped —
    /// which is what previously made every restart book an unpredictable
    /// number of "client cancelled".
    fn record_cancelled_metrics(&mut self) {
        if self.metrics_finalized {
            return;
        }
        if self.closing.load(Ordering::SeqCst) {
            self.record_teardown_metrics();
            return;
        }
        cascadia_metrics::TASKS_CANCELLED_TOTAL
            .with_label_values(&[&self.model])
            .inc();
        cascadia_metrics::GENERATION_DURATION_SECONDS
            .with_label_values(&[&self.model, "cancelled"])
            .observe(self.submitted_at.elapsed().as_secs_f64());
        self.metrics_finalized = true;
    }

    /// Record a generation cut short by server teardown.
    ///
    /// Its own `finish_reason` rather than silence: recording nothing left
    /// `completed + failed + cancelled < started` after every restart with
    /// no metric to explain the difference, so "how many answers did that
    /// restart cut off?" was unanswerable. And its own reason rather than
    /// `cancelled` or `error`: a graceful restart is neither a client
    /// hanging up nor an engine fault, and folding it into either puts
    /// planned maintenance on a panel someone pages on.
    fn record_teardown_metrics(&mut self) {
        if self.metrics_finalized {
            return;
        }
        cascadia_metrics::GENERATION_DURATION_SECONDS
            .with_label_values(&[&self.model, "teardown"])
            .observe(self.submitted_at.elapsed().as_secs_f64());
        self.metrics_finalized = true;
    }
}

impl Stream for ChunkStream {
    type Item = Chunk;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Chunk>> {
        let this = self.get_mut();
        loop {
            if this.done {
                return Poll::Ready(None);
            }

            // 1) Drain anything already buffered for us.
            {
                let mut bufs = this.buffers.lock();
                if bufs.cancelled.contains(&this.task_id) {
                    this.done = true;
                    bufs.chunks.remove(&this.task_id);
                    bufs.cancelled.remove(&this.task_id);
                    drop(bufs);
                    this.record_cancelled_metrics();
                    return Poll::Ready(None);
                }
                if let Some(buf) = bufs.chunks.get_mut(&this.task_id) {
                    if let Some(chunk) = buf.pop_front() {
                        let is_final = chunk.is_final;
                        if is_final {
                            this.done = true;
                            bufs.chunks.remove(&this.task_id);
                        }
                        drop(bufs);
                        this.record_chunk_metrics(&chunk);
                        return Poll::Ready(Some(chunk));
                    }
                }
            }

            // 2) Drive the engine one step. An Err is terminal for the
            //    failed task: the engine has already abandoned it and reset
            //    its own state. If the error names a *different* task than
            //    ours (concurrent serving), route the failure to that task
            //    and keep polling — ending our healthy stream would kill the
            //    wrong request. An error for our task, or a task-less /
            //    engine-level error, ends this stream.
            //    Distribution happens while the engine lock is still held.
            //    Releasing it first and buffering afterwards leaves a window in
            //    which another stream completes a LATER step and buffers its
            //    round before this one lands — reordering a single request's
            //    own text. Step and distribute have to be one critical section.
            //    The engine mutex is a sync lock and `step()` spans real
            //    network round trips, so NEVER block this worker thread on
            //    it: enough streams parked in a hard `lock()` here (plus one
            //    stepping) can pin every tokio worker, and with no worker
            //    left to poll the I/O + timer drivers the step-holder's
            //    `recv()` never wakes — no bytes delivered, no timeout fired,
            //    the whole pipeline permanently wedged (#122). Contended =
            //    someone else is stepping; park with a registered waker and
            //    let their release wake us.
            let step_result = {
                // Wake-on-release, unwind included: `step()` runs arbitrary
                // engine code under this lock, and a panic there must not
                // strand the streams parked behind it (see
                // `WakeParkedOnDrop`). Declared BEFORE the engine guard so
                // reverse-declaration drop order releases the engine first
                // and wakes second, on the normal path and on the unwind.
                let mut wake_on_release = WakeParkedOnDrop::disarmed(&this.buffers);
                let mut guard = match this.engine.try_lock() {
                    Some(g) => g,
                    None => {
                        this.buffers.lock().step_wakers.push(cx.waker().clone());
                        // Re-check after registering: the holder may have
                        // released (and drained wakers) between the failed
                        // try_lock and the registration. A stale registration
                        // from the success arm only costs a spurious wake.
                        match this.engine.try_lock() {
                            Some(g) => g,
                            None => return Poll::Pending,
                        }
                    }
                };
                // The engine is ours: every exit from here — value, early
                // return, or panic — owes the parked streams a wake.
                wake_on_release.arm();
                let result = match guard.as_mut() {
                    // Slot empty: `Runner::close` took the engine, i.e. server
                    // teardown. Handled below rather than here because
                    // `fail_teardown` needs `&mut *this` and this guard still
                    // borrows `this.engine`.
                    None => None,
                    Some(engine) => {
                        // Apply cancels that arrived while the engine was
                        // busy (Runner::cancel / stream Drop never block on
                        // the engine mutex — see deferred_cancels).
                        let deferred = std::mem::take(&mut this.buffers.lock().deferred_cancels);
                        for tid in &deferred {
                            engine.cancel(tid);
                        }
                        Some(match engine.step() {
                            Ok(produced) => {
                                let empty = produced.is_empty();
                                let mut bufs = this.buffers.lock();
                                for (tid, chunk) in produced {
                                    if bufs.cancelled.contains(&tid) {
                                        continue;
                                    }
                                    bufs.chunks.entry(tid).or_default().push_back(chunk);
                                }
                                Ok(empty)
                            }
                            Err(e) => Err(e),
                        })
                    }
                };
                // Engine released here; `wake_on_release` drains + wakes
                // every stream parked on contention right after.
                drop(guard);
                result
            };
            let Some(step_result) = step_result else {
                return this.fail_teardown();
            };
            let produced_empty = match step_result {
                Ok(empty) => empty,
                Err(e) => match e.task_id() {
                    Some(failed) if failed != &this.task_id => {
                        // Misattribution guard: fail the named task's stream,
                        // not ours. Route the failure as a final error chunk
                        // into its buffer — a cancel marker would end its
                        // stream silently and its client would get a
                        // truncated 200. Skip the push if its buffer already
                        // ends in an error chunk (a non-clearing engine
                        // re-emits the same failure every step; one is
                        // enough).
                        let failed = failed.clone();
                        warn!(
                            failed_task = %failed,
                            polled_task = %this.task_id,
                            error = %e,
                            "engine step failed for another task; routing failure to it"
                        );
                        let mut bufs = this.buffers.lock();
                        // Cancelled task = no consumer left; recreating its
                        // buffer entry would leak until close() (mirror the
                        // cancelled-check on the distribution path below).
                        if !bufs.cancelled.contains(&failed) {
                            let buf = bufs.chunks.entry(failed.clone()).or_default();
                            // Dedup: one error chunk max. If the buffer
                            // already ends in a NORMAL final chunk the task
                            // completed first — the drain delivers that and
                            // drops the appended error with the entry, which
                            // is deliberate (a post-completion failure is
                            // contract noise; the warn above records it).
                            if !buf.back().is_some_and(|c| c.error.is_some()) {
                                buf.push_back(Chunk::error(failed.clone(), e.to_string()));
                            }
                        }
                        drop(bufs);
                        // Bound only a *non-clearing* engine: one that re-emits
                        // the SAME foreign-task Err every step (never dropping
                        // it) would busy-spin our `continue` forever, so count
                        // repeats of the same id toward the guard. Distinct
                        // foreign failures mean the engine IS clearing each and
                        // making progress — don't penalize our healthy stream
                        // for merely observing them, so reset the run on a new
                        // id. (Tracked separately from `consecutive_empty` so a
                        // burst of distinct foreign errors can't false-close us.)
                        if this.last_errored_task.as_ref() == Some(&failed) {
                            this.consecutive_foreign_err += 1;
                        } else {
                            this.last_errored_task = Some(failed.clone());
                            this.consecutive_foreign_err = 1;
                        }
                        if this.consecutive_foreign_err >= MAX_CONSECUTIVE_EMPTY_STEPS {
                            // Abnormal close of a healthy stream — fail it
                            // loud (see the own-task arm below).
                            warn!(
                                task = %this.task_id,
                                "engine re-failed the same foreign task for {} consecutive steps; closing stream",
                                MAX_CONSECUTIVE_EMPTY_STEPS
                            );
                            return this.fail_stream(format!(
                                "engine wedged re-failing task {failed} for {} consecutive steps: {e}",
                                MAX_CONSECUTIVE_EMPTY_STEPS
                            ));
                        }
                        continue;
                    }
                    _ => {
                        // Surface the failure as a final error chunk: a bare
                        // end-of-stream builds a 200 with truncated text and
                        // finish_reason "stop" at the API layer. The error
                        // chunk routes through the existing 503 / SSE-error
                        // paths instead.
                        warn!(
                            task = %this.task_id,
                            error = %e,
                            "engine step failed; closing stream"
                        );
                        return this.fail_stream(e.to_string());
                    }
                },
            };

            if produced_empty {
                this.consecutive_empty += 1;
                if this.consecutive_empty >= MAX_CONSECUTIVE_EMPTY_STEPS {
                    // An engine that wedges by stalling (Ok-empty forever)
                    // is a failure the client must see — fail loud like the
                    // step-error arms above.
                    warn!(
                        task = %this.task_id,
                        "engine made no progress for {} consecutive steps; closing stream",
                        MAX_CONSECUTIVE_EMPTY_STEPS
                    );
                    return this.fail_stream(format!(
                        "engine made no progress for {} consecutive steps",
                        MAX_CONSECUTIVE_EMPTY_STEPS
                    ));
                }
                continue;
            }
            // Real progress this round: clear both hot-spin guards.
            this.consecutive_empty = 0;
            this.last_errored_task = None;
            this.consecutive_foreign_err = 0;

            // 3) Everything this round produced — including our own chunks —
            //    is already in its owner's buffer, so loop and let step 1 hand
            //    ours back in FIFO order. Terminal/TTFT accounting rides that
            //    same drain, so metrics stay delivery-timed.
            //
            //    Returning our chunk straight from the step instead would jump
            //    it ahead of anything another stream buffered for us while we
            //    were blocked on the engine lock, delivering one request's own
            //    text out of order. That was unreachable until an engine emitted
            //    chunks for several tasks in a single step — every other engine
            //    produces for at most the one task it is working on. Continuous
            //    batching does, and with it six identical concurrent prompts
            //    came back as six different permutations of the right answer.
        }
    }
}

impl Drop for ChunkStream {
    fn drop(&mut self) {
        // A stream dropped before any terminal outcome was recorded is an
        // abandoned generation (client disconnect, or a cancel the poll
        // loop never observed) — count it exactly once.
        self.record_cancelled_metrics();
        // Tell the engine to abandon this task. Without this an SSE
        // client that disconnects mid-generation leaves the engine
        // grinding through max_tokens worth of chunks that no one
        // will ever drain — the chunk buffer for this task accretes
        // until close() and the engine slot stays busy.
        //
        // Never block on the engine mutex here: Drop runs on whatever
        // (often tokio worker) thread drops the stream, and a hard lock()
        // during another stream's step counts toward the worker starvation
        // that wedges the pipeline (#122). Busy engine = defer; the next
        // engine-lock holder applies it before stepping.
        {
            // Same wake-on-release protocol as `poll_next`, unwind
            // included: `engine.cancel()` is engine code too, and a panic
            // here would otherwise strand every parked stream. Declared
            // BEFORE the guard so the engine lock is released first.
            let mut wake_on_release = WakeParkedOnDrop::disarmed(&self.buffers);
            match self.engine.try_lock() {
                Some(mut guard) => {
                    wake_on_release.arm();
                    if let Some(engine) = guard.as_mut() {
                        engine.cancel(&self.task_id);
                    }
                }
                None => {
                    self.buffers
                        .lock()
                        .deferred_cancels
                        .push(self.task_id.clone());
                }
            }
        }
        let mut bufs = self.buffers.lock();
        bufs.chunks.remove(&self.task_id);
        // Tombstone rather than remove: an engine that defers its final
        // chunk past cancel would re-buffer it for this dead stream via
        // another caller's distribution pass, leaking the map entry until
        // close(). Bounded: task ids are UUIDs (no reuse), so the rare
        // wholesale clear only risks re-buffering one late chunk.
        if bufs.cancelled.len() >= 4096 {
            bufs.cancelled.clear();
        }
        bufs.cancelled.insert(self.task_id.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_engine_mock::MockBuilder;
    use futures::StreamExt;

    #[test]
    fn blocking_context_guard_sets_and_restores_flag() {
        assert!(!BLOCKING_CONTEXT.with(|f| f.get()));
        {
            let _g = BlockingContextGuard::enter();
            assert!(BLOCKING_CONTEXT.with(|f| f.get()));
        }
        // RAII restored the prior value.
        assert!(!BLOCKING_CONTEXT.with(|f| f.get()));
    }

    #[test]
    fn blocking_context_guard_nests_correctly() {
        let _outer = BlockingContextGuard::enter();
        assert!(BLOCKING_CONTEXT.with(|f| f.get()));
        {
            let _inner = BlockingContextGuard::enter();
            assert!(BLOCKING_CONTEXT.with(|f| f.get()));
        }
        // Inner drop should restore outer's value (true), not false.
        assert!(BLOCKING_CONTEXT.with(|f| f.get()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_async_works_inside_async_context() {
        // Without the guard set, we're on a tokio worker thread —
        // run_async must pick the block_in_place path and not panic.
        let h = tokio::runtime::Handle::current();
        let result = tokio::task::spawn_blocking(move || {
            // spawn_blocking thread; mimic what `run_relay_loop` does
            // and enter the guard before calling run_async.
            let _g = BlockingContextGuard::enter();
            run_async(&h, async { 42 })
        })
        .await
        .unwrap();
        assert_eq!(result, 42);
    }

    async fn make_runner() -> Runner {
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        runner
    }

    #[tokio::test]
    async fn generate_yields_then_final() {
        let runner = make_runner().await;
        let task = GenerationTask::new("t1", "the quick brown fox").with_max_tokens(2);
        let mut stream = runner.generate(task).unwrap();
        let mut chunks = Vec::new();
        while let Some(c) = stream.next().await {
            chunks.push(c);
        }
        assert!(chunks.len() >= 1);
        assert!(chunks.last().unwrap().is_final);
    }

    #[tokio::test]
    async fn cancel_terminates_stream() {
        let runner = make_runner().await;
        let task = GenerationTask::new("t1", "alpha bravo charlie delta").with_max_tokens(64);
        let task_id = task.task_id.clone();
        let mut stream = runner.generate(task).unwrap();
        let _first = stream.next().await;
        runner.cancel(&task_id);
        // After cancel, the stream should terminate quickly.
        let mut polls = 0;
        loop {
            match stream.next().await {
                None => break,
                Some(_) => {
                    polls += 1;
                    if polls > 32 {
                        panic!("cancel did not terminate stream");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn close_without_start_does_not_panic() {
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner.close();
    }

    /// An engine whose every step fails (e.g. dead transport).
    struct FailingEngine;

    impl Engine for FailingEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            Err(EngineError::Backend("transport down".into()))
        }
    }

    struct FailingBuilder;

    #[async_trait::async_trait]
    impl Builder for FailingBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(FailingEngine))
        }
    }

    /// A failing engine that counts `step()` calls, so a test can assert
    /// the relay loop throttled instead of hot-spinning.
    struct CountingFailingEngine(Arc<std::sync::atomic::AtomicUsize>);

    impl Engine for CountingFailingEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(EngineError::Backend("transport down".into()))
        }
    }

    #[test]
    fn relay_loop_throttles_persistently_failing_step() {
        use std::sync::atomic::Ordering;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine: Box<dyn Engine> = Box::new(CountingFailingEngine(calls.clone()));
        let runner = Arc::new(Runner {
            builder: Mutex::new(None),
            engine: Arc::new(Mutex::new(Some(engine))),
            buffers: Arc::new(Mutex::new(Buffers::default())),
            model: Mutex::new(None),
            closing: Arc::new(AtomicBool::new(false)),
        });

        // Drive the relay loop on a worker thread, let it run for a window
        // several backoffs wide, then empty the engine slot to stop it.
        let driver = runner.clone();
        let handle = std::thread::spawn(move || driver.run_relay_loop());
        std::thread::sleep(RELAY_ERR_BACKOFF * 5);
        runner.close(); // engine slot -> None: loop breaks next round
        assert_eq!(handle.join().unwrap(), RelayExit::SlotEmpty);

        // With a 200 ms backoff per Err round, ~1 s of wall time admits a
        // handful of iterations, not the millions a hot-spin would do.
        // Bound generously to stay non-flaky on a loaded CI box.
        let n = calls.load(Ordering::SeqCst);
        assert!(
            n <= 20,
            "relay loop hot-spun on persistently-failing step(): {n} calls in ~1s"
        );
        assert!(n >= 1, "relay loop never stepped the engine");
    }

    /// A worker whose every step hits a dead peer link (flattened transport
    /// error). The relay loop must exit ConnectionFatal so the supervisor
    /// rebuilds the stage, instead of spinning at the backoff rate forever.
    struct DeadLinkEngine(Arc<std::sync::atomic::AtomicUsize>);

    impl Engine for DeadLinkEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Same shape the dist-spec worker produces once its socket is
            // dropped: TransportError::NotConnected flattened to a string.
            Err(EngineError::Backend(
                "not connected; call connect()/accept() first".into(),
            ))
        }
    }

    #[test]
    fn relay_loop_exits_on_connection_fatal_step() {
        use std::sync::atomic::Ordering;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine: Box<dyn Engine> = Box::new(DeadLinkEngine(calls.clone()));
        let runner = Arc::new(Runner {
            builder: Mutex::new(None),
            engine: Arc::new(Mutex::new(Some(engine))),
            buffers: Arc::new(Mutex::new(Buffers::default())),
            model: Mutex::new(None),
            closing: Arc::new(AtomicBool::new(false)),
        });

        // No external stop: the loop must terminate on its own. A timed join
        // guards against a regression that lets it spin forever.
        let driver = runner.clone();
        let handle = std::thread::spawn(move || driver.run_relay_loop());
        let start = std::time::Instant::now();
        while !handle.is_finished() {
            if start.elapsed() > std::time::Duration::from_secs(5) {
                panic!("relay loop did not exit on connection-fatal step()");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(handle.join().unwrap(), RelayExit::ConnectionFatal);
        // It bails on the first fatal Err — no spin.
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "relay loop should exit on the first connection-fatal step()"
        );
    }

    /// A worker step whose flattened transport error is a peer crash (TCP
    /// RST) must also exit ConnectionFatal — the dominant dead-peer case,
    /// not just a clean FIN. Mirrors the test above with the RST string.
    struct RstStepEngine(EngineError);

    impl Engine for RstStepEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            Err(EngineError::Backend(self.0.to_string()))
        }
    }

    #[test]
    fn relay_loop_exits_on_peer_rst_and_mid_frame_step() {
        // Both the peer-crash string and the mid-frame stall string must be
        // treated as connection-fatal by the relay loop.
        for msg in ["connection reset by peer", "recv_exact timed out after 60s"] {
            let engine: Box<dyn Engine> = Box::new(RstStepEngine(EngineError::Backend(msg.into())));
            let runner = Arc::new(Runner {
                builder: Mutex::new(None),
                engine: Arc::new(Mutex::new(Some(engine))),
                buffers: Arc::new(Mutex::new(Buffers::default())),
                model: Mutex::new(None),
                closing: Arc::new(AtomicBool::new(false)),
            });
            let driver = runner.clone();
            let handle = std::thread::spawn(move || driver.run_relay_loop());
            let start = std::time::Instant::now();
            while !handle.is_finished() {
                if start.elapsed() > std::time::Duration::from_secs(5) {
                    panic!("relay loop did not exit on connection-fatal step() for {msg:?}");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(
                handle.join().unwrap(),
                RelayExit::ConnectionFatal,
                "expected ConnectionFatal exit for {msg:?}"
            );
        }
    }

    #[tokio::test]
    async fn step_error_surfaces_error_chunk_then_terminates() {
        let runner = Runner::new(Box::new(FailingBuilder));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        let mut stream = runner
            .generate(GenerationTask::new("t1", "hello").with_max_tokens(8))
            .unwrap();
        // The first poll hits the step error. The failure must reach the
        // consumer as a final error chunk — a bare end-of-stream builds a
        // 200 with truncated text and finish_reason "stop" at the API layer,
        // indistinguishable from the model choosing to stop.
        let chunk = stream
            .next()
            .await
            .expect("failed task must surface a final error chunk, not a silent end");
        assert!(chunk.is_final);
        assert!(
            chunk
                .error
                .as_deref()
                .is_some_and(|e| e.contains("transport down")),
            "error chunk must carry the step failure, got: {:?}",
            chunk.error
        );
        // And then the stream terminates — no empty-step spinning.
        assert!(stream.next().await.is_none());
    }

    /// Engine that fails one named task with a task-attributed error on its
    /// first step, then serves every other task one token + a final marker.
    /// Models concurrent serving where task A's transport dies while B is
    /// healthy.
    struct SelectiveFailEngine {
        fail: TaskId,
        tasks: Vec<TaskId>,
        served: std::collections::HashSet<TaskId>,
    }

    impl Engine for SelectiveFailEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> Result<(), EngineError> {
            self.tasks.push(task.task_id);
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            if self.tasks.iter().any(|t| t == &self.fail) {
                // Abandon the doomed task and report it by id.
                self.tasks.retain(|t| t != &self.fail);
                return Err(
                    EngineError::Backend("transport down".into()).for_task(self.fail.clone())
                );
            }
            // Serve the first not-yet-served task a single final chunk.
            let next = self
                .tasks
                .iter()
                .find(|t| !self.served.contains(*t))
                .cloned();
            match next {
                Some(tid) => {
                    self.served.insert(tid.clone());
                    Ok(vec![(tid.clone(), Chunk::final_marker(tid, "ok"))])
                }
                None => Ok(Vec::new()),
            }
        }
    }

    struct SelectiveFailBuilder(TaskId);

    #[async_trait::async_trait]
    impl Builder for SelectiveFailBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(SelectiveFailEngine {
                fail: self.0,
                tasks: Vec::new(),
                served: std::collections::HashSet::new(),
            }))
        }
    }

    #[tokio::test]
    async fn step_err_for_other_task_does_not_kill_healthy_stream() {
        // Two concurrent streams share one engine. Task "a" fails with a
        // task-attributed error; task "b" must survive and finish, and only
        // "a" ends.
        let runner = Runner::new(Box::new(SelectiveFailBuilder("a".to_string())));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        let mut stream_a = runner
            .generate(GenerationTask::new("a", "x").with_max_tokens(4))
            .unwrap();
        let mut stream_b = runner
            .generate(GenerationTask::new("b", "y").with_max_tokens(4))
            .unwrap();

        // Drive B first: it polls the engine, observes the err attributed to
        // "a", routes the failure to "a", and keeps going to serve itself.
        let b_chunk = stream_b.next().await;
        assert!(
            b_chunk.as_ref().map(|c| c.is_final).unwrap_or(false),
            "healthy task b's stream was killed by task a's failure: {b_chunk:?}"
        );

        // A's stream surfaces the routed failure as a final error chunk —
        // ending it silently would give A's client a truncated 200.
        let a_chunk = stream_a
            .next()
            .await
            .expect("failed task a must surface a final error chunk, not a silent end");
        assert!(a_chunk.is_final);
        assert!(
            a_chunk
                .error
                .as_deref()
                .is_some_and(|e| e.contains("transport down")),
            "task a's error chunk must carry the failure, got: {:?}",
            a_chunk.error
        );
        assert!(stream_a.next().await.is_none());
        // B's stream is now complete too.
        assert!(stream_b.next().await.is_none());
    }

    /// Contract-violating engine: every step re-emits the SAME foreign-task
    /// Err and never clears it (real engines drop the failed task). Models a
    /// pathological non-clearing engine that would otherwise busy-spin the
    /// observing stream's `continue`.
    struct ForeignErrSpinEngine {
        fail: TaskId,
        steps: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Engine for ForeignErrSpinEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.steps.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(EngineError::Backend("transport down".into()).for_task(self.fail.clone()))
        }
    }

    struct ForeignErrSpinBuilder {
        fail: TaskId,
        steps: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Builder for ForeignErrSpinBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(ForeignErrSpinEngine {
                fail: self.fail,
                steps: self.steps,
            }))
        }
    }

    #[tokio::test]
    async fn repeated_foreign_task_err_bounds_observing_stream() {
        use std::sync::atomic::Ordering;
        // The engine fails task "other" on every step and never clears it.
        // Our stream ("ours") never gets served, but the cross-task error arm
        // must count toward the no-progress guard so we terminate rather than
        // spin a tokio worker forever.
        let steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runner = Runner::new(Box::new(ForeignErrSpinBuilder {
            fail: "other".to_string(),
            steps: steps.clone(),
        }));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        let mut stream = runner
            .generate(GenerationTask::new("ours", "x").with_max_tokens(64))
            .unwrap();

        // Must terminate, not hang — and the abnormal close must surface as
        // a final error chunk (a bare end-of-stream would build a truncated
        // 200 at the API layer), bounded by the no-progress guard.
        let last = stream
            .next()
            .await
            .expect("observing stream must surface an error chunk when the engine wedges");
        assert!(last.is_final);
        assert!(
            last.error.is_some(),
            "guard close must carry an error, got: {last:?}"
        );
        assert!(stream.next().await.is_none());
        // One step per no-progress round; capped by the guard, not unbounded.
        let n = steps.load(Ordering::SeqCst);
        assert!(
            n <= MAX_CONSECUTIVE_EMPTY_STEPS,
            "stream busy-spun on a non-clearing foreign-task Err: {n} steps"
        );
    }

    /// A foreign-task failure routed to a task whose client already went
    /// away (cancelled / stream dropped) must NOT recreate that task's
    /// buffer entry — nothing will ever drain it, so it would leak until
    /// close(). Mirrors the cancelled-check on the distribution path.
    #[tokio::test]
    async fn foreign_err_for_cancelled_task_does_not_recreate_buffer() {
        let steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runner = Runner::new(Box::new(ForeignErrSpinBuilder {
            fail: "dead".to_string(),
            steps,
        }));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        // "dead" was cancelled before its failure surfaced (the dead
        // transport is often exactly why the client disconnected).
        runner.cancel(&"dead".to_string());
        let mut stream = runner
            .generate(GenerationTask::new("ours", "x").with_max_tokens(4))
            .unwrap();
        // Drive until our stream closes (the engine never serves us).
        while stream.next().await.is_some() {}
        assert!(
            !runner.buffers.lock().chunks.contains_key("dead"),
            "routing a failure to a cancelled task must not recreate its buffer entry"
        );
    }

    /// Engine whose first `step()` panics while holding the engine lock —
    /// an engine bug, but one whose unwind passes straight through
    /// `ChunkStream::poll_next`. It blocks in `step()` until the test has a
    /// second stream parked behind the lock, then panics. Later steps fail
    /// cleanly so a woken stream can terminate instead of panicking in turn.
    struct PanicStepEngine {
        entered: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        panicked: Arc<AtomicBool>,
    }

    impl Engine for PanicStepEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            if self.panicked.load(Ordering::SeqCst) {
                return Err(EngineError::Backend("engine panicked earlier".into()));
            }
            self.entered.store(true, Ordering::SeqCst);
            // Bounded, so a broken test fails rather than hangs.
            let start = Instant::now();
            while !self.release.load(Ordering::SeqCst)
                && start.elapsed() < std::time::Duration::from_secs(10)
            {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.panicked.store(true, Ordering::SeqCst);
            panic!("engine step blew up");
        }
    }

    struct PanicStepBuilder {
        entered: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        panicked: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl Builder for PanicStepBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(PanicStepEngine {
                entered: self.entered,
                release: self.release,
                panicked: self.panicked,
            }))
        }
    }

    /// A panic inside `step()` must still wake the streams parked on the
    /// engine lock.
    ///
    /// `parking_lot` doesn't poison, so the unwind releases the engine and
    /// the next poller could drive it fine — but the drain + wake used to be
    /// straight-line code after the release, which an unwind skips. Every
    /// parked stream then waited `Pending` forever, with no log line and a
    /// perfectly healthy-looking engine.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn step_panic_still_wakes_parked_streams() {
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let panicked = Arc::new(AtomicBool::new(false));
        let runner = Arc::new(Runner::new(Box::new(PanicStepBuilder {
            entered: entered.clone(),
            release: release.clone(),
            panicked: panicked.clone(),
        })));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();

        let stream_a = runner
            .generate(GenerationTask::new("panic-a", "x").with_max_tokens(4))
            .unwrap();
        let mut stream_b = runner
            .generate(GenerationTask::new("panic-b", "y").with_max_tokens(4))
            .unwrap();

        // Poll A on a blocking thread and catch the panic THERE, keeping A's
        // ChunkStream alive across the unwind: dropping it would run
        // `ChunkStream::drop`, whose own release wakes B and would mask the
        // bug under test.
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let a = tokio::task::spawn_blocking(move || {
            let mut sa = stream_a;
            let waker = futures::task::noop_waker();
            let mut cx = Context::from_waker(&waker);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = Pin::new(&mut sa).poll_next(&mut cx);
            }));
            let _ = hold_rx.recv(); // hold A's stream until the test is done
            caught.is_err()
        });

        // Wait until A is inside step(), holding the engine lock.
        let start = Instant::now();
        while !entered.load(Ordering::SeqCst) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "stream A never entered step()"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }

        // B polls, finds the engine contended, and parks with a waker.
        let b = tokio::spawn(async move { stream_b.next().await });
        let start = Instant::now();
        while runner.buffers.lock().step_wakers.is_empty() {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "stream B never parked on the engine lock"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }

        // Blow up the step with B parked behind the lock.
        release.store(true, Ordering::SeqCst);

        let chunk = tokio::time::timeout(std::time::Duration::from_secs(10), b)
            .await
            .expect("parked stream was never woken after a panicking step()")
            .unwrap();
        let chunk = chunk.expect("woken stream must terminate loud, not silently");
        assert!(chunk.is_final);
        assert!(
            chunk.error.is_some(),
            "woken stream must surface the broken engine: {chunk:?}"
        );

        let _ = hold_tx.send(());
        assert!(a.await.unwrap(), "step() was expected to panic");
    }

    /// Engine that never errors and never produces: every step is Ok(empty).
    /// Models a wedged engine that stalls instead of failing.
    struct StallingEngine;

    impl Engine for StallingEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            Ok(Vec::new())
        }
    }

    struct StallingBuilder;

    #[async_trait::async_trait]
    impl Builder for StallingBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(StallingEngine))
        }
    }

    /// The no-progress guard must fail loud: an engine that wedges by
    /// producing nothing (rather than by erroring) is a failure the client
    /// must see, not a truncated 200 with finish_reason "stop".
    #[tokio::test]
    async fn no_progress_close_surfaces_error_chunk() {
        let runner = Runner::new(Box::new(StallingBuilder));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        let mut stream = runner
            .generate(GenerationTask::new("t1", "hello").with_max_tokens(8))
            .unwrap();
        let last = stream
            .next()
            .await
            .expect("no-progress close must surface a final error chunk");
        assert!(last.is_final);
        assert!(
            last.error
                .as_deref()
                .is_some_and(|e| e.contains("no progress")),
            "error chunk must name the stall, got: {:?}",
            last.error
        );
        assert!(stream.next().await.is_none());
    }

    /// Engine that holds a task and makes real progress for several steps
    /// without producing a token — a chunked continuous-batching prefill —
    /// signalling liveness with zero-token chunks, then emits output.
    struct SlowPrefillEngine {
        task: Option<TaskId>,
        prefill_steps: usize,
    }

    impl Engine for SlowPrefillEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> Result<(), EngineError> {
            self.task = Some(task.task_id);
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            let Some(id) = self.task.clone() else {
                return Ok(Vec::new());
            };
            if self.prefill_steps > 0 {
                self.prefill_steps -= 1;
                return Ok(vec![(
                    id.clone(),
                    Chunk::token(id, 0, String::new()).with_n_tokens(0),
                )]);
            }
            self.task = None;
            Ok(vec![(
                id.clone(),
                Chunk::final_marker(id, "done").with_n_tokens(1),
            )])
        }
    }

    struct SlowPrefillBuilder {
        prefill_steps: usize,
    }

    #[async_trait::async_trait]
    impl Builder for SlowPrefillBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(SlowPrefillEngine {
                task: None,
                prefill_steps: self.prefill_steps,
            }))
        }
    }

    /// The no-progress guard must not fire on an engine that IS progressing
    /// but has no token to show yet. A continuous-batching prefill spans
    /// ceil(prompt_tokens / max_num_batched_tokens) scheduler iterations
    /// producing nothing, which is far more than MAX_CONSECUTIVE_EMPTY_STEPS
    /// for an ordinary prompt — so such an engine must signal liveness with a
    /// zero-token chunk rather than returning an empty Vec, and the stream
    /// must survive it. (Empty Vec is reserved for "nothing in flight".)
    #[tokio::test]
    async fn zero_token_liveness_chunks_survive_the_no_progress_guard() {
        let runner = Runner::new(Box::new(SlowPrefillBuilder {
            prefill_steps: MAX_CONSECUTIVE_EMPTY_STEPS * 3,
        }));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        let mut stream = runner
            .generate(GenerationTask::new("t1", "a long prompt").with_max_tokens(8))
            .unwrap();
        let mut last_text = String::new();
        let mut saw_final = false;
        while let Some(c) = stream.next().await {
            assert!(
                c.error.is_none(),
                "a progressing prefill must not trip the stall guard: {:?}",
                c.error
            );
            if c.is_final {
                saw_final = true;
                last_text = c.text.clone();
            }
        }
        assert!(saw_final, "stream ended without a final chunk");
        assert_eq!(last_text, "done");
    }

    /// Engine that emits a chunk for EVERY in-flight task on every step — the
    /// shape continuous batching introduces, and the one no engine had before
    /// it. Chunks are numbered so a consumer can tell order from content.
    struct MultiTaskEngine {
        tasks: Vec<TaskId>,
        seq: usize,
        total: usize,
    }

    impl Engine for MultiTaskEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            if self.seq >= self.total {
                return Ok(Vec::new());
            }
            let n = self.seq;
            self.seq += 1;
            let last = self.seq == self.total;
            // Widen the window in which the other stream can be mid-poll while
            // this round is distributed; the race is otherwise rare enough on a
            // fast machine to make the test toothless.
            std::thread::yield_now();
            Ok(self
                .tasks
                .iter()
                .map(|t| {
                    let text = format!("{n},");
                    let c = if last {
                        Chunk::final_marker(t.clone(), text)
                    } else {
                        Chunk::token(t.clone(), 0, text)
                    };
                    (t.clone(), c)
                })
                .collect())
        }
    }

    struct MultiTaskBuilder {
        tasks: Vec<TaskId>,
        total: usize,
    }

    #[async_trait::async_trait]
    impl Builder for MultiTaskBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(MultiTaskEngine {
                tasks: self.tasks.clone(),
                seq: 0,
                total: self.total,
            }))
        }
    }

    /// A stream must deliver ITS OWN chunks in the order the engine produced
    /// them, even while another stream is driving the same engine.
    ///
    /// The failure this guards: poll_next used to return a chunk taken straight
    /// from its own step() call. A stream that found its buffer empty, then
    /// blocked on the engine lock, could have a chunk buffered for it by the
    /// stream holding that lock — and would then return its newer chunk first.
    /// On hardware this turned six identical concurrent prompts into six
    /// different permutations of the right answer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_streams_receive_their_own_chunks_in_order() {
        const N: usize = 40;
        let ids: Vec<TaskId> = vec!["a".to_string(), "b".to_string()];
        let runner = Arc::new(Runner::new(Box::new(MultiTaskBuilder {
            tasks: ids.clone(),
            total: N,
        })));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();

        let expected: String = (0..N).map(|i| format!("{i},")).collect();
        let mut handles = Vec::new();
        for id in ids {
            let r = runner.clone();
            handles.push(tokio::spawn(async move {
                let mut stream = r
                    .generate(GenerationTask::new(id.clone(), "x").with_max_tokens(64))
                    .unwrap();
                let mut got = String::new();
                while let Some(c) = stream.next().await {
                    assert!(c.error.is_none(), "unexpected error chunk: {:?}", c.error);
                    got.push_str(&c.text);
                }
                (id, got)
            }));
        }
        for h in handles {
            let (id, got) = h.await.unwrap();
            assert_eq!(
                got, expected,
                "stream {id} received its chunks out of order"
            );
        }
    }

    /// #16: each stream's metrics must be attributed to ITS OWN generation
    /// while several streams share one engine.
    ///
    /// Every other metrics test drives a single stream, where "recorded at
    /// buffer time" and "recorded at delivery" are the same instant on the
    /// same object — so none of them can tell the two apart. This one can,
    /// and it is the case continuous batching actually runs.
    ///
    /// If `record_chunk_metrics` were called from the distribution loop
    /// (where the polling stream sees OTHER tasks' chunks) instead of from
    /// the step-1 drain, all four assertions below break at once: the
    /// poller would stamp foreign chunks against its own `submitted_at`, a
    /// foreign `is_final` would set the poller's `metrics_finalized` and
    /// silently swallow its own terminal sample, and tokens would be
    /// counted twice — once by the driver, once by the owner.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_streams_attribute_metrics_to_their_own_generation() {
        const MODEL: &str = "runner-metrics-concurrent-model";
        const N: usize = 8;
        const STREAMS: u64 = 2;
        let ids: Vec<TaskId> = vec!["ca".to_string(), "cb".to_string()];
        let runner = Arc::new(Runner::new(Box::new(MultiTaskBuilder {
            tasks: ids.clone(),
            total: N,
        })));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage(MODEL, "CPU"),
            )
            .await
            .unwrap();

        let mut handles = Vec::new();
        for id in ids {
            let r = runner.clone();
            handles.push(tokio::spawn(async move {
                let mut stream = r
                    .generate(GenerationTask::new(id, "x").with_max_tokens(64))
                    .unwrap();
                while stream.next().await.is_some() {}
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        // Every chunk MultiTaskEngine emits carries text, so all N per task
        // are token-bearing: N tokens, 1 TTFT and N-1 inter-token gaps each.
        assert_eq!(
            cascadia_metrics::TOKENS_GENERATED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            STREAMS * N as u64,
            "tokens must be counted once, by the owning stream"
        );
        assert_eq!(
            cascadia_metrics::GENERATION_TTFT_SECONDS
                .with_label_values(&[MODEL])
                .get_sample_count(),
            STREAMS,
            "one TTFT sample per generation, not per poller"
        );
        assert_eq!(
            cascadia_metrics::GENERATION_INTER_TOKEN_SECONDS
                .with_label_values(&[MODEL])
                .get_sample_count(),
            STREAMS * (N as u64 - 1)
        );
        // The terminal sample is the one a foreign is_final would swallow.
        assert_eq!(
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[MODEL, "stop"])
                .get_sample_count(),
            STREAMS,
            "every generation must book its own terminal outcome"
        );
        for (counter, name) in [
            (&cascadia_metrics::TASKS_CANCELLED_TOTAL, "cancelled"),
            (&cascadia_metrics::TASKS_FAILED_TOTAL, "failed"),
        ] {
            assert_eq!(
                counter.with_label_values(&[MODEL]).get(),
                0,
                "clean concurrent completion must not book {name}"
            );
        }
    }

    /// Engine that fails a sequence of DISTINCT foreign tasks, one per step
    /// (clearing each — a correctly-behaving engine), then serves the polled
    /// task a final chunk. Models concurrent serving where several other
    /// requests die in a row while a healthy task waits its turn.
    struct DistinctForeignFailEngine {
        fails: std::collections::VecDeque<TaskId>,
        serve: TaskId,
    }

    impl Engine for DistinctForeignFailEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            if let Some(f) = self.fails.pop_front() {
                // Each foreign failure is distinct and cleared (popped) — the
                // engine is making progress, not hot-spinning one task.
                return Err(EngineError::Backend("transport down".into()).for_task(f));
            }
            Ok(vec![(
                self.serve.clone(),
                Chunk::final_marker(self.serve.clone(), "ok"),
            )])
        }
    }

    struct DistinctForeignFailBuilder {
        fails: Vec<TaskId>,
        serve: TaskId,
    }

    #[async_trait::async_trait]
    impl Builder for DistinctForeignFailBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(DistinctForeignFailEngine {
                fails: self.fails.into_iter().collect(),
                serve: self.serve,
            }))
        }
    }

    #[tokio::test]
    async fn distinct_foreign_task_errs_do_not_kill_healthy_stream() {
        // Three DISTINCT foreign tasks fail in a row (>= MAX_CONSECUTIVE_EMPTY_
        // STEPS) before our task is served. The old "any foreign err bumps the
        // bound" logic false-closed us here; the same-task refinement must let
        // our healthy stream survive and finish.
        let runner = Runner::new(Box::new(DistinctForeignFailBuilder {
            fails: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            serve: "ours".to_string(),
        }));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        let mut stream = runner
            .generate(GenerationTask::new("ours", "x").with_max_tokens(4))
            .unwrap();
        let chunk = stream.next().await;
        assert!(
            chunk.as_ref().map(|c| c.is_final).unwrap_or(false),
            "healthy stream was false-closed by distinct foreign-task errors: {chunk:?}"
        );
        assert!(stream.next().await.is_none(), "stream should be complete");
    }

    /// #16: metric tests use a DEDICATED model id per test — the registry is
    /// process-global and tests in this binary run concurrently, so exact
    /// assertions are only safe on labels no other test touches.
    async fn make_runner_for_model(model: &str) -> Runner {
        let runner = Runner::new(Box::new(MockBuilder::new()));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage(model, "CPU"),
            )
            .await
            .unwrap();
        runner
    }

    #[tokio::test]
    async fn completed_generation_records_metrics() {
        const MODEL: &str = "runner-metrics-complete-model";
        let runner = make_runner_for_model(MODEL).await;
        // 4-word prompt, max_tokens 2 → two 1-token chunks, then an empty
        // final marker with finish_reason "length".
        let task = GenerationTask::new("t-m1", "alpha bravo charlie delta").with_max_tokens(2);
        let mut stream = runner.generate(task).unwrap();
        while stream.next().await.is_some() {}

        assert_eq!(
            cascadia_metrics::TOKENS_GENERATED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            2
        );
        assert_eq!(
            cascadia_metrics::GENERATION_TTFT_SECONDS
                .with_label_values(&[MODEL])
                .get_sample_count(),
            1
        );
        // Two token chunks → exactly ONE inter-token gap. The separate
        // empty final marker must not add an artificial sample.
        assert_eq!(
            cascadia_metrics::GENERATION_INTER_TOKEN_SECONDS
                .with_label_values(&[MODEL])
                .get_sample_count(),
            1
        );
        assert_eq!(
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[MODEL, "length"])
                .get_sample_count(),
            1
        );
        // Load + warmup gauges were set at start() (values may be ~0 for the
        // mock; presence of the label pair is the contract).
        //
        // Assert that through the EXPOSITION, not `with_label_values().get()`:
        // that call CREATES the child on first access, so it would fabricate
        // the very label pair it claims to check and then compare a fresh
        // gauge's 0.0 against >= 0.0 — an assertion that cannot fail even if
        // start() never touched either gauge.
        let (_, buf) = cascadia_metrics::encode_text();
        let text = String::from_utf8(buf).expect("exposition is utf-8");
        for family in [
            "cascadia_engine_model_load_duration_seconds",
            "cascadia_engine_warmup_duration_seconds",
        ] {
            let needle = format!("{family}{{device=\"CPU\",model=\"{MODEL}\"}}");
            assert!(text.contains(&needle), "missing {needle} in:\n{text}");
        }
    }

    #[tokio::test]
    async fn dropped_stream_counts_cancelled_exactly_once() {
        const MODEL: &str = "runner-metrics-cancel-model";
        let runner = make_runner_for_model(MODEL).await;
        let task = GenerationTask::new("t-m2", "alpha bravo charlie delta").with_max_tokens(64);
        let mut stream = runner.generate(task).unwrap();
        let _ = stream.next().await; // one chunk, then the client goes away
        drop(stream);
        assert_eq!(
            cascadia_metrics::TASKS_CANCELLED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            1
        );
        assert_eq!(
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[MODEL, "cancelled"])
                .get_sample_count(),
            1
        );
    }

    #[tokio::test]
    async fn completed_stream_does_not_count_cancelled() {
        const MODEL: &str = "runner-metrics-clean-model";
        let runner = make_runner_for_model(MODEL).await;
        let task = GenerationTask::new("t-m3", "alpha bravo").with_max_tokens(8);
        let mut stream = runner.generate(task).unwrap();
        while stream.next().await.is_some() {}
        drop(stream);
        assert_eq!(
            cascadia_metrics::TASKS_CANCELLED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            0
        );
    }

    /// Server teardown (Runner::close with a generation in flight) must NOT
    /// be booked as a client cancellation — the metric's contract is
    /// "explicit cancel or client disconnect".
    #[tokio::test]
    async fn teardown_mid_generation_is_not_counted_cancelled() {
        const MODEL: &str = "runner-metrics-teardown-model";
        let runner = make_runner_for_model(MODEL).await;
        let task = GenerationTask::new("t-td", "alpha bravo charlie delta").with_max_tokens(64);
        let mut stream = runner.generate(task).unwrap();
        let _ = stream.next().await;
        runner.close(); // engine slot empties mid-generation

        // The stream must END LOUD. A bare None is indistinguishable from a
        // completed generation, and the API turns it into a 200 carrying the
        // partial text with finish_reason "stop" — telling the client the
        // model finished normally while the server was going away.
        let last = stream
            .next()
            .await
            .expect("teardown must surface a terminal chunk, not a silent end-of-stream");
        assert!(
            last.error.is_some(),
            "teardown chunk must carry an error so the API fails the request: {last:?}"
        );
        assert!(stream.next().await.is_none());
        drop(stream);

        assert_eq!(
            cascadia_metrics::TASKS_CANCELLED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            0,
            "a restart is not a client cancellation"
        );
        // Nor is it an engine fault: booking it failed would put planned
        // maintenance on the primary failure SLO.
        assert_eq!(
            cascadia_metrics::TASKS_FAILED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            0,
            "a restart is not an engine failure"
        );
        // But it IS an outcome. Recording nothing left completed+failed+
        // cancelled < started after every restart, with no metric to explain
        // the gap.
        assert_eq!(
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[MODEL, "teardown"])
                .get_sample_count(),
            1
        );
    }

    /// The same shutdown, with the opposite interleaving: the stream is
    /// DROPPED before it is ever polled again.
    ///
    /// Both orderings are reachable on the real shutdown path — per-connection
    /// tasks are spawned, so they keep running while `close()` is dispatched —
    /// and nothing sequences them. Before `Runner::closing`, which one won
    /// decided whether the generation was booked as a client cancellation or
    /// as nothing at all, so every restart produced a nondeterministic cancel
    /// spike whose size no one could correct for.
    #[tokio::test]
    async fn teardown_books_the_same_outcome_when_the_stream_is_dropped_first() {
        const MODEL: &str = "runner-metrics-teardown-drop-model";
        let runner = make_runner_for_model(MODEL).await;
        let task = GenerationTask::new("t-td2", "alpha bravo charlie delta").with_max_tokens(64);
        let mut stream = runner.generate(task).unwrap();
        let _ = stream.next().await;
        runner.close();
        drop(stream); // never polled again — Drop decides the outcome

        assert_eq!(
            cascadia_metrics::TASKS_CANCELLED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            0,
            "drop-first teardown must not book a client cancellation either"
        );
        assert_eq!(
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[MODEL, "teardown"])
                .get_sample_count(),
            1,
            "both orderings must book the SAME outcome"
        );
    }

    /// Engine that acknowledges every task with a final marker tagged
    /// FinishReason::Cancelled — the contract-blessed way for an engine to
    /// surface a cancel it honored.
    struct CancelAckEngine(Vec<TaskId>);

    impl Engine for CancelAckEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> Result<(), EngineError> {
            self.0.push(task.task_id);
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            Ok(self
                .0
                .pop()
                .map(|tid| {
                    let c = Chunk::final_marker(tid.clone(), "")
                        .with_finish_reason(cascadia_types::FinishReason::Cancelled);
                    (tid, c)
                })
                .into_iter()
                .collect())
        }
    }

    struct CancelAckBuilder;

    #[async_trait::async_trait]
    impl Builder for CancelAckBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            Ok(Box::new(CancelAckEngine(Vec::new())))
        }
    }

    /// An engine-acknowledged cancel (Cancelled final marker) must land in
    /// TASKS_CANCELLED_TOTAL exactly once — same counter as the tombstone /
    /// Drop cancel paths — so the cancelled counter and the cancelled
    /// duration histogram never diverge.
    #[tokio::test]
    async fn cancel_ack_final_chunk_counts_cancelled_exactly_once() {
        const MODEL: &str = "runner-metrics-cancelack-model";
        let runner = Runner::new(Box::new(CancelAckBuilder));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage(MODEL, "CPU"),
            )
            .await
            .unwrap();
        let mut stream = runner
            .generate(GenerationTask::new("t-ca", "x").with_max_tokens(4))
            .unwrap();
        while stream.next().await.is_some() {}
        drop(stream); // Drop must not double-count (metrics_finalized set)
        assert_eq!(
            cascadia_metrics::TASKS_CANCELLED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            1
        );
        assert_eq!(
            cascadia_metrics::GENERATION_DURATION_SECONDS
                .with_label_values(&[MODEL, "cancelled"])
                .get_sample_count(),
            1
        );
    }

    #[tokio::test]
    async fn failed_generation_counts_failed_metric() {
        const MODEL: &str = "runner-metrics-fail-model";
        let runner = Runner::new(Box::new(FailingBuilder));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage(MODEL, "CPU"),
            )
            .await
            .unwrap();
        let mut stream = runner
            .generate(GenerationTask::new("t-m4", "hello").with_max_tokens(4))
            .unwrap();
        while stream.next().await.is_some() {}
        drop(stream);
        assert_eq!(
            cascadia_metrics::TASKS_FAILED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            1
        );
        // The failure is terminal accounting — the drop must not ALSO count
        // it as cancelled.
        assert_eq!(
            cascadia_metrics::TASKS_CANCELLED_TOTAL
                .with_label_values(&[MODEL])
                .get(),
            0
        );
    }
}
