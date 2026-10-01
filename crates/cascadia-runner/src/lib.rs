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

use std::collections::{HashMap, HashSet, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use cascadia_engine::{Builder, Engine, EngineError, StreamLinks};
use cascadia_types::{Chunk, FinishReason, GenerationTask, PeerLayout, ShardSpec, TaskId};
use futures::Stream;
use parking_lot::Mutex;
use tracing::{debug, info, warn};

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

/// Ceiling on [`Buffers::deferred_cancels`]. Matches the `cancelled`
/// tombstone bound: every engine-lock acquisition drains the queue, so
/// reaching this means the engine is not being locked at all and the
/// cancels are moot anyway — but an unbounded vec on that path would grow
/// for the life of the process.
const MAX_DEFERRED_CANCELS: usize = 4096;

/// How long [`Runner::close`] waits for the engine lock before it logs why
/// it is still waiting (it then keeps waiting).
const CLOSE_WARN_AFTER: Duration = Duration::from_secs(5);

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
    /// Streams parked because another caller held the engine (#122).
    /// [`EngineGuard`] drains + wakes these on every engine-lock release,
    /// so a parked stream re-polls as soon as the engine may be free.
    /// Registration happens ONLY after a failed `try_lock`, and the
    /// registrant re-checks the lock afterwards, so a release can't slip
    /// between the check and the registration.
    step_wakers: Vec<std::task::Waker>,
    /// Cancels that could not take the engine lock without blocking a
    /// worker thread (#122). Applied by the next holder to ACQUIRE the
    /// engine lock — `submit`, `poll_next`, another `cancel`, or a relay
    /// round — before it does anything else with the engine. That is the
    /// same effective timing a blocking cancel had, since a running step
    /// can't be interrupted anyway.
    ///
    /// Draining only from `poll_next` was not enough: cancelling the one
    /// in-flight request leaves nobody polling, so its engine-side slot and
    /// KV region stayed occupied until some later request happened along —
    /// hours, on an idle server. A relay-stage runner takes `cancel()`
    /// calls and never polls a stream at all, so the queue there only grew.
    ///
    /// Deduped on push, and capped at [`MAX_DEFERRED_CANCELS`] like the
    /// `cancelled` tombstone set, so a queue that is somehow not being
    /// drained cannot grow without bound.
    deferred_cancels: Vec<TaskId>,
    /// Issue #76: tasks admitted to the engine by [`Runner::generate`] /
    /// [`Runner::generate_async`] whose stream has not been dropped yet,
    /// whether still queued in the engine or already running. A re-attach
    /// that may have swapped links cancels every one of them in the engine
    /// and moves it to `orphaned` (see [`Runner::reattach`]). Removed by
    /// the stream's `Drop` and by [`Runner::cancel`].
    admitted: HashSet<TaskId>,
    /// Issue #76: tasks a re-attach cancelled. The distribution pass drops
    /// any chunk the engine still produces for them (a cancel ack, say), so
    /// the stream ends with the re-attach error instead of new output.
    /// Removed together with `admitted`.
    orphaned: HashSet<TaskId>,
}

/// The engine, plus the state every release of it has to service: the
/// wakers of the streams parked on contention, and the cancels deferred
/// while it was busy (#122).
///
/// One `Arc<EngineSlot>` is shared by the [`Runner`] and every
/// [`ChunkStream`] it hands out. They used to hold loose `Arc` clones of an
/// engine mutex and a buffers mutex, which is why the release protocol
/// ("drop the engine lock, THEN drain + wake") could only be a convention,
/// re-implemented by hand at six sites and silently skippable at a seventh.
/// Going through [`EngineSlot::lock`] / [`EngineSlot::try_lock`] makes it a
/// property of the type instead: there is no way to hold the engine without
/// holding an [`EngineGuard`], and no way to drop that guard without waking.
struct EngineSlot {
    /// `Mutex<Option<...>>` so `close()` can drop the engine while other
    /// callers hold references to the runner; subsequent calls fail
    /// cleanly with [`EngineError::NotLoaded`].
    ///
    /// Deliberately a SYNC `parking_lot` mutex: `run_relay_loop` takes it
    /// from a `spawn_blocking` thread and `submit` from the blocking pool,
    /// neither of which can await. Contended async callers must park on a
    /// failed `try_lock` rather than block a tokio worker here (#122).
    engine: Mutex<Option<Box<dyn Engine>>>,
    buffers: Mutex<Buffers>,
    /// Issue #76: true while the last re-attach failed after swapping
    /// streams, so the engine's state is unspecified until one succeeds.
    /// Lives on the slot (not the runner) because [`ChunkStream`] reaches
    /// the engine through the slot alone. Written ONLY while holding the
    /// engine lock (in [`Runner::reattach`]), and every binding read is
    /// made after acquiring it, so no holder can step or submit between a
    /// failed swap and the fence going up.
    fenced: AtomicBool,
    /// Issue #76: which set of links the engine is on. Bumped (under the
    /// engine lock, in [`Runner::reattach`]) whenever `reattach_streams` ran
    /// and did not return `PeerRejected`, i.e. whenever a swap may have
    /// happened. A [`ChunkStream`] records it at submit and fails once it
    /// moves: the engine reset its session state with the swap, so a request
    /// in flight across it can never finish. Same locking discipline as
    /// `fenced`.
    attach_epoch: AtomicU64,
}

/// An engine lock that cannot be released without paying what the release
/// owes: draining [`Buffers::step_wakers`] and waking every parked stream.
///
/// Two orderings this encodes, both of which were load-bearing bugs when
/// left to hand-written call sites:
///
/// * **Release, THEN wake.** Waking while the engine lock is still held
///   lets a woken stream re-poll, fail its `try_lock`, and re-park —
///   stranded again, and this time with no one left to wake it. Rust drops
///   struct fields AFTER `Drop::drop` returns, so the inner `parking_lot`
///   guard lives in an `Option` that `drop` explicitly takes and releases
///   first; making it a plain field would reintroduce exactly this bug.
/// * **Wake on unwind too.** A panic inside `step()` unwinds past any
///   straight-line drain. `parking_lot` doesn't poison, so the engine comes
///   out of it usable — but every parked stream waits `Pending` forever,
///   with nothing in the log. A destructor covers that path by
///   construction.
///
/// A failed `try_lock` never produces a guard, so the contended
/// `poll_next` path still returns `Pending` without waking the waker it
/// just registered (which would spin the worker for the holder's step).
struct EngineGuard<'a> {
    slot: &'a EngineSlot,
    /// `Some` for the guard's whole life; taken by `drop` to release the
    /// engine lock before the wake. See the ordering note above.
    guard: Option<parking_lot::MutexGuard<'a, Option<Box<dyn Engine>>>>,
}

impl EngineGuard<'_> {
    /// The engine, or `None` once `close()` has emptied the slot.
    fn engine(&mut self) -> Option<&mut Box<dyn Engine>> {
        self.guard.as_mut().and_then(|g| g.as_mut())
    }

    /// Fill the slot (`start`) or empty it (`close`). An emptied slot makes
    /// every later lock holder fail cleanly with [`EngineError::NotLoaded`];
    /// the outgoing engine is dropped under the lock.
    fn set(&mut self, engine: Option<Box<dyn Engine>>) {
        if let Some(g) = self.guard.as_mut() {
            **g = engine;
        }
    }

    /// Hand the engine the cancels that arrived while it was busy, before
    /// this holder does anything else with it (see
    /// [`Buffers::deferred_cancels`]). Draining an emptied slot's queue is
    /// still correct: `close()` has run, so there is nothing left to cancel.
    fn apply_deferred_cancels(&mut self) {
        let deferred = std::mem::take(&mut self.slot.buffers.lock().deferred_cancels);
        if deferred.is_empty() {
            return;
        }
        if let Some(engine) = self.engine() {
            for tid in &deferred {
                engine.cancel(tid);
            }
        }
    }
}

impl Drop for EngineGuard<'_> {
    fn drop(&mut self) {
        // Release the engine FIRST: `Option::take` + drop, because struct
        // fields would otherwise drop after this body has already woken.
        drop(self.guard.take());
        let wakers = std::mem::take(&mut self.slot.buffers.lock().step_wakers);
        for w in wakers {
            w.wake();
        }
    }
}

impl EngineSlot {
    fn new(engine: Option<Box<dyn Engine>>) -> Self {
        Self {
            engine: Mutex::new(engine),
            buffers: Mutex::new(Buffers::default()),
            fenced: AtomicBool::new(false),
            attach_epoch: AtomicU64::new(0),
        }
    }

    /// Binding only when read while holding the engine lock (see
    /// [`EngineSlot::fenced`]).
    fn is_fenced(&self) -> bool {
        self.fenced.load(Ordering::SeqCst)
    }

    /// Binding only when read while holding the engine lock (see
    /// [`EngineSlot::attach_epoch`]).
    fn attach_epoch(&self) -> u64 {
        self.attach_epoch.load(Ordering::SeqCst)
    }

    /// Take the engine lock, blocking the calling thread until it is free.
    /// Sync callers only — see the mutex's note on #122.
    fn lock(&self) -> EngineGuard<'_> {
        EngineGuard {
            slot: self,
            guard: Some(self.engine.lock()),
        }
    }

    /// [`lock`](Self::lock), but call `on_slow` once if the lock is still
    /// held after `after`, then keep waiting.
    fn lock_noting_delay(&self, after: Duration, on_slow: impl FnOnce()) -> EngineGuard<'_> {
        let guard = match self.engine.try_lock_for(after) {
            Some(g) => g,
            None => {
                on_slow();
                self.engine.lock()
            }
        };
        EngineGuard {
            slot: self,
            guard: Some(guard),
        }
    }

    /// Take the engine lock only if it is free right now. `None` means
    /// someone is mid-step and the caller must park (`poll_next`) or defer
    /// (`cancel_or_defer`) instead of blocking a tokio worker (#122).
    fn try_lock(&self) -> Option<EngineGuard<'_>> {
        self.engine.try_lock().map(|g| EngineGuard {
            slot: self,
            guard: Some(g),
        })
    }

    /// [`lock`](Self::lock), then hand the engine everything queued in
    /// [`Buffers::deferred_cancels`]. Every acquisition that can act on
    /// that queue goes through this or its `try_` twin, which is what makes
    /// the queue's contract ("applied by the next engine-lock holder") true
    /// rather than "applied by the next stream poll, if one ever comes".
    fn lock_applying_cancels(&self) -> EngineGuard<'_> {
        let mut guard = self.lock();
        guard.apply_deferred_cancels();
        guard
    }

    /// [`try_lock`](Self::try_lock) + [`lock_applying_cancels`] semantics.
    fn try_lock_applying_cancels(&self) -> Option<EngineGuard<'_>> {
        let mut guard = self.try_lock()?;
        guard.apply_deferred_cancels();
        Some(guard)
    }

    /// Cancel `task_id` on the engine, or queue it for the next lock holder
    /// when a step is in flight.
    ///
    /// Shared by [`Runner::cancel`] and [`ChunkStream::drop`], which have
    /// the same constraint: both run on whatever (often tokio worker)
    /// thread calls them, and a hard `lock()` during another stream's step
    /// counts toward the worker starvation that wedges the pipeline (#122).
    /// Deferring costs nothing — a running step can't be interrupted
    /// anyway, so the next holder applies it at the same effective time a
    /// blocking cancel would have.
    fn cancel_or_defer(&self, task_id: &TaskId) {
        match self.try_lock_applying_cancels() {
            Some(mut guard) => {
                if let Some(engine) = guard.engine() {
                    engine.cancel(task_id);
                }
            }
            None => self.defer_cancel(task_id),
        }
    }

    /// Queue a cancel for the next engine-lock holder.
    ///
    /// Deduped: the queue is a to-do list, and cancelling the same task
    /// twice buys nothing. Capped at [`MAX_DEFERRED_CANCELS`], dropping the
    /// newest with a warning rather than clearing the queue wholesale — the
    /// tombstone set can afford a wholesale clear because re-buffering one
    /// late chunk is the whole cost, whereas dropping queued cancels leaves
    /// that many engine-side slots occupied.
    fn defer_cancel(&self, task_id: &TaskId) {
        let mut bufs = self.buffers.lock();
        if bufs.deferred_cancels.iter().any(|t| t == task_id) {
            return;
        }
        if bufs.deferred_cancels.len() >= MAX_DEFERRED_CANCELS {
            warn!(
                task = %task_id,
                "deferred cancel queue is full ({MAX_DEFERRED_CANCELS}); dropping this cancel — \
                 the engine keeps the task until it finishes on its own"
            );
            return;
        }
        bufs.deferred_cancels.push(task_id.clone());
    }
}

/// Issue #76: stream-mode link bookkeeping. `generation` bumps on every
/// re-attach that reached the engine (success, failure or rejection); the relay loop
/// parks until it changes. `closed` is set by `close()` under this mutex so a
/// parked relay cannot miss it. The failed-re-attach fence itself lives on
/// [`EngineSlot::fenced`].
#[derive(Default)]
struct LinkState {
    generation: u64,
    closed: bool,
}

/// `parking_lot`, not `std::sync`: like the engine slot, it must not poison.
/// Lock order: engine slot → `state`, never the reverse.
#[derive(Default)]
struct Link {
    state: Mutex<LinkState>,
    cv: parking_lot::Condvar,
    /// Test-only hook around the relay loop's unlocked fence pre-check; see
    /// [`Link::hooked_fence_read`].
    #[cfg(test)]
    precheck_hook: Mutex<Option<PrecheckHook>>,
}

#[cfg(test)]
type PrecheckHook = Arc<dyn Fn(Option<bool>) + Send + Sync>;

#[cfg(test)]
impl Link {
    /// Run the relay loop's fence pre-check `read` with the test hook around
    /// it: called with `None` after the generation snapshot and before the
    /// flag read, then with `Some(fenced)` after it. Lets a test pin the
    /// interleaving "relay reads the flag while a re-attach holds it
    /// transiently up".
    fn hooked_fence_read(&self, read: impl FnOnce() -> bool) -> bool {
        let hook = self.precheck_hook.lock().clone();
        if let Some(hook) = &hook {
            hook(None);
        }
        let fenced = read();
        if let Some(hook) = &hook {
            hook(Some(fenced));
        }
        fenced
    }
}

/// Issue #76: the relay loop's "link down" warning fires once per link death,
/// not on every wake. Keyed on the attach epoch the engine was on when the
/// step failed: a rejected re-attach wakes the relay without changing the
/// links, so its next fatal step is the same death and stays quiet.
#[derive(Default)]
struct LinkDownLog {
    warned_at: Option<u64>,
}

impl LinkDownLog {
    /// True the first time a dead link is seen on `epoch`.
    fn first_for(&mut self, epoch: u64) -> bool {
        if self.warned_at == Some(epoch) {
            return false;
        }
        self.warned_at = Some(epoch);
        true
    }
}

pub struct Runner {
    /// Mutex so the Runner is `Sync` even with a `dyn Builder` inside;
    /// taken once during `start()` and dropped.
    builder: Mutex<Option<Box<dyn Builder>>>,
    /// Engine + parked-stream bookkeeping, shared with every
    /// [`ChunkStream`] this runner hands out.
    slot: Arc<EngineSlot>,
    /// Model id captured from the [`ShardSpec`] at `start()`, used as the
    /// `model` label on generation metrics (#16). `None` until started.
    model: Mutex<Option<Arc<str>>>,
    /// Set by `close()` BEFORE the engine slot is emptied, so an in-flight
    /// generation can tell "the server is shutting down" from "my client
    /// hung up". Without it the outcome depends on an unsynchronised race
    /// between a stream's next poll and its `Drop`, and every restart books
    /// a nondeterministic number of client cancellations.
    closing: Arc<AtomicBool>,
    /// Set only by [`Runner::start_with_streams`] (issue #76). In stream mode
    /// a dead link is recoverable via [`Runner::reattach`]; in TCP mode it is
    /// terminal, exactly as before.
    stream_mode: AtomicBool,
    /// Issue #76: see [`Link`]. Only used in stream mode. An `Arc`, like
    /// `slot`, so [`Runner::reattach`] can take `&self` and still hand both
    /// to its blocking half.
    link: Arc<Link>,
}

impl Runner {
    pub fn new(builder: Box<dyn Builder>) -> Self {
        Self {
            builder: Mutex::new(Some(builder)),
            slot: Arc::new(EngineSlot::new(None)),
            model: Mutex::new(None),
            closing: Arc::new(AtomicBool::new(false)),
            stream_mode: AtomicBool::new(false),
            link: Arc::new(Link::default()),
        }
    }

    /// A started runner around `engine`, without a builder.
    #[cfg(test)]
    fn for_test(engine: Box<dyn Engine>, stream_mode: bool) -> Arc<Self> {
        Arc::new(Self {
            builder: Mutex::new(None),
            slot: Arc::new(EngineSlot::new(Some(engine))),
            model: Mutex::new(None),
            closing: Arc::new(AtomicBool::new(false)),
            stream_mode: AtomicBool::new(stream_mode),
            link: Arc::new(Link::default()),
        })
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
        self.load_build_warmup(builder, shard).await
    }

    /// Issue #76: like [`Runner::start_with_listen`], but the caller supplies
    /// already-connected streams (see the contract on
    /// `cascadia_transport::InjectedStream`). Puts the runner in stream
    /// mode, which enables [`Runner::reattach`].
    ///
    /// In stream mode a dead link does not end [`Runner::run_relay_loop`]:
    /// the relay parks until [`Runner::reattach`] or [`Runner::close`]. There
    /// is no link-death event — the embedder watches its own ends of the
    /// streams and re-attaches when one dies.
    ///
    /// **[`Runner::close`] is mandatory in stream mode.** A relay parked on a
    /// dead link only returns when re-attached or closed; dropping the runner
    /// without closing leaves its `spawn_blocking` thread parked forever, and
    /// tokio runtime shutdown waits on it and hangs.
    ///
    /// # Errors
    ///
    /// [`EngineError::NotLoaded`] if the runner was already started (the
    /// builder is consumed by the first start); otherwise whatever the
    /// builder's `connect_streams` / `load` / `build` returns, e.g.
    /// [`EngineError::PeerRejected`] when `links` does not match the stage's
    /// shape.
    pub async fn start_with_streams(
        &self,
        links: StreamLinks,
        shard: ShardSpec,
    ) -> Result<(), EngineError> {
        let mut builder = self.builder.lock().take().ok_or(EngineError::NotLoaded)?;
        info!(links = ?links.shape(), "runner connect (injected streams)");
        builder.connect_streams(links).await?;
        // Before `load_build_warmup` places the engine in the slot: anything
        // that can reach a live engine must already see stream mode (a relay
        // loop started on it reads the flag once, at entry).
        self.stream_mode.store(true, Ordering::SeqCst);
        self.load_build_warmup(builder, shard).await
    }

    async fn load_build_warmup(
        &self,
        mut builder: Box<dyn Builder>,
        shard: ShardSpec,
    ) -> Result<(), EngineError> {
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
        self.slot.lock().set(Some(engine));
        Ok(())
    }

    /// Backwards-compatible shortcut equivalent to
    /// `start_with_listen(peers, shard, None)`. Engines that need a
    /// listener (workers, middle stages) should call
    /// [`start_with_listen`] with the correct address.
    pub async fn start(&self, peers: PeerLayout, shard: ShardSpec) -> Result<(), EngineError> {
        self.start_with_listen(peers, shard, None).await
    }

    /// Enqueue a task with the engine.
    ///
    /// **Blocks the calling thread** for up to a full engine step when one is
    /// in flight. Async contexts must not call this directly — use
    /// [`Runner::generate_async`], which dispatches it via `spawn_blocking`.
    ///
    /// Refuses (as a benign no-op `Ok(())`) any task whose id already carries
    /// a `cancelled` tombstone — see the comment in the lock region below.
    ///
    /// A task submitted this way has no stream, so a re-attach does not
    /// cancel it (only [`Runner::generate`] / [`Runner::generate_async`]
    /// tasks are tracked); cancel it yourself if it must not outlive one.
    pub fn submit(&self, task: GenerationTask) -> Result<(), EngineError> {
        self.submit_at_epoch(task, false).map(|_| ())
    }

    /// [`Runner::submit`], also returning the attach epoch read under the
    /// same engine lock that admitted the task (see
    /// [`EngineSlot::attach_epoch`]); the task's [`ChunkStream`] records it.
    /// With `track`, an admitted task is recorded in [`Buffers::admitted`]
    /// under that lock, so a later re-attach can cancel it.
    fn submit_at_epoch(&self, task: GenerationTask, track: bool) -> Result<u64, EngineError> {
        // NOTE: the lock below is a hard `lock()`, not a `try_lock()` — this
        // is the one engine-lock caller that must not give up, since there is
        // nowhere to park a submit. A tokio worker thread blocked here counts
        // toward the driver starvation that wedges the pipeline (#122), which
        // is why the async entry point (`Runner::generate_async`) keeps this
        // call off the runtime.
        // Releasing the guard wakes the streams parked on contention, on
        // the empty-slot path as much as the submitting one — an empty slot
        // means `close()` already ran, so no later lock holder is coming to
        // wake them instead. Taking the lock also flushes cancels deferred
        // while the engine was busy, freeing their slots before this task
        // asks for one.
        let mut guard = self.slot.lock_applying_cancels();
        let epoch = self.slot.attach_epoch();
        // ==== BEGIN cancellation-safety check — keep INSIDE the lock ====
        // NEVER hand the engine a task that is already tombstoned
        // cancelled. `generate_async` arms a cancel guard BEFORE it
        // dispatches this submit, so a caller future dropped at that
        // await (axum drops handler futures on client disconnect) writes
        // the tombstone while the detached submit closure is still queued
        // or parked on this very mutex. Admitting the task then creates a
        // ghost: no `ChunkStream` was ever constructed for it, so nothing
        // drains its chunks and nothing cancels it — it occupies a packed
        // slot and grinds through max_tokens driven by other streams'
        // polls, buffering under an id nobody reads, until process exit.
        //
        // Task ids are per-request UUIDs at the API layer, so a
        // tombstoned id can only mean "this exact request was already
        // cancelled" — refusing is a no-op, not a lost request. Read
        // while holding the engine mutex so it cannot interleave with a
        // concurrent step's distribution pass; buffers-under-engine is
        // the lock order used everywhere else (see `Runner::cancel`).
        //
        // "Inside the lock" now means inside `guard`'s scope: the refusal
        // is an arm of this expression rather than an early `return`, so
        // the guard is still live and still owes — and on drop pays — the
        // drain + wake every engine-lock release owes its parked streams.
        if self.is_fenced() {
            // Issue #76: last re-attach failed after swapping streams; the
            // engine's state is unspecified until a re-attach succeeds.
            Err(EngineError::NotConnected)
        } else if self.slot.buffers.lock().cancelled.contains(&task.task_id) {
            Ok(epoch)
        } else {
            // ==== END; the branch below is the original submit path ====
            match guard.engine() {
                Some(engine) => {
                    let task_id = task.task_id.clone();
                    engine.submit(task)?;
                    if track {
                        self.slot.buffers.lock().admitted.insert(task_id);
                    }
                    Ok(epoch)
                }
                None => Err(EngineError::NotLoaded),
            }
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
            let mut bufs = self.slot.buffers.lock();
            bufs.cancelled.insert(task_id.clone());
            bufs.chunks.remove(task_id);
            bufs.admitted.remove(task_id);
            bufs.orphaned.remove(task_id);
        }
        self.slot.cancel_or_defer(task_id);
    }

    /// Submit a task and return a stream of chunks. Stops on the final
    /// chunk, on cancellation, on an engine step error, or after
    /// MAX_CONSECUTIVE_EMPTY_STEPS empty engine polls (engine appears
    /// stuck).
    ///
    /// **Blocks the calling thread** for up to a full engine step: the submit
    /// runs inline via [`Runner::submit`], and only the returned stream is
    /// async. Async contexts must use [`Runner::generate_async`] instead.
    ///
    /// **Task ids must be unique** among live requests (here and in
    /// [`Runner::generate_async`]). Chunks are routed by id: after a
    /// re-attach the dropped requests' ids stay marked orphaned until their
    /// streams are dropped, and a new task reusing such an id meanwhile gets
    /// none of its output (its chunks are discarded). A cancelled id is
    /// refused outright (see [`Runner::submit`]).
    pub fn generate(&self, task: GenerationTask) -> Result<ChunkStream, EngineError> {
        let epoch = self.submit_at_epoch(task.clone(), true)?;
        Ok(self.stream_for(task.task_id, epoch))
    }

    /// Async [`Runner::generate`]: submits off the runtime via
    /// `spawn_blocking`. Async servers must use this — `submit` blocks on
    /// the engine mutex for up to a full engine step, and enough worker
    /// threads parked there starve the tokio I/O + timer drivers, which is
    /// the #122 pipeline wedge.
    ///
    /// **Cancellation-safe**: this future may be dropped at any await point
    /// (axum drops handler futures on client disconnect, and disconnects
    /// cluster exactly when the submit wait is longest) without leaking a
    /// task. Dropping the `JoinHandle` detaches rather than cancels, so the
    /// submit closure still runs and may admit the task with no `ChunkStream`
    /// to own it — a ghost that would occupy an engine slot and grind through
    /// `max_tokens` with nobody draining its chunks. Two mechanisms close
    /// that, both required:
    ///
    /// * A cancel guard armed BEFORE the submit is dispatched. Unless the
    ///   stream is successfully constructed (which disarms it), the guard's
    ///   `Drop` calls [`Runner::cancel`] — non-blocking by construction, so
    ///   it is safe on any thread a dropped future lands on.
    /// * [`Runner::submit`] refuses tombstoned ids, for the reverse ordering
    ///   where the guard's cancel runs BEFORE the queued/parked submit does.
    ///
    /// Guarantee: on return of this future, either the caller owns a
    /// [`ChunkStream`] for the task, or the task is cancelled — never
    /// submitted-and-unowned.
    pub async fn generate_async(
        self: &Arc<Self>,
        task: GenerationTask,
    ) -> Result<ChunkStream, EngineError> {
        let task_id = task.task_id.clone();
        // Armed BEFORE the dispatch below: from here until `disarm()`, every
        // exit path — `?`, a panic, or the caller's future being dropped at
        // the await — cancels the task.
        let mut guard = SubmitCancelGuard {
            runner: self.clone(),
            task_id: task_id.clone(),
            armed: true,
        };
        let this = self.clone();
        let joined = tokio::task::spawn_blocking(move || this.submit_at_epoch(task, true)).await;
        let epoch = match joined {
            // The engine rejected the task (QueueFull, NotLoaded, …). Per the
            // `Engine::submit` contract nothing was enqueued, so there is
            // nothing to cancel — disarm rather than write a tombstone for a
            // task that never existed (an overload burst would otherwise grow
            // the tombstone set with ids no stream will ever consume).
            Ok(Err(e)) => {
                guard.disarm();
                return Err(e);
            }
            // Join error: the submit panicked (or was cancelled). Whether it
            // landed is unknowable from here, so leave the guard ARMED and
            // let it cancel — cancelling an id the engine never admitted is a
            // documented no-op for every `Engine` impl.
            Err(e) => return Err(EngineError::Backend(format!("submit task join: {e}"))),
            Ok(Ok(epoch)) => epoch,
        };
        // The task now has an owner: from here its `Drop` handles cancellation.
        let stream = self.stream_for(task_id, epoch);
        guard.disarm();
        Ok(stream)
    }

    fn stream_for(&self, task_id: TaskId, epoch: u64) -> ChunkStream {
        ChunkStream {
            task_id,
            slot: self.slot.clone(),
            epoch,
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

    /// Issue #76: replace this stage's dead link(s) on the live engine. A
    /// `None` link (or `None` / absent `ep_workers` entry) keeps that link.
    /// Stream mode only.
    ///
    /// # Before calling
    ///
    /// * **Close your own end of every stream being replaced first**
    ///   (contract item 3 on `cascadia_transport::InjectedStream`). The step
    ///   that is blocked on a dead link holds the engine lock this call needs;
    ///   only closing that stream makes the step return and frees the lock.
    /// * **Relay rule:** a stage that has an upstream link must replace it on
    ///   every re-attach (an idle relay's step blocks on its upstream read
    ///   while holding the engine lock, and only closing that stream frees
    ///   it). Any re-attach therefore cascades to the head, which refreshes
    ///   the head's session state. `check_reattach_streams` enforces this.
    /// * There is no link-death event: the embedder detects a dead link by
    ///   watching its own ends.
    /// * Issue one re-attach per outage, for all its links at once.
    ///   Concurrent or duplicate re-attaches for the same outage are an
    ///   embedder bug: the second one swaps out the links the first just
    ///   installed. (Retrying after a dropped call is fine; see
    ///   *Cancellation*.)
    ///
    /// # In-flight requests
    ///
    /// Requests on the head when its links are swapped cannot finish (the
    /// engine resets its session state). That covers every request submitted
    /// before the re-attach through [`Runner::generate`] /
    /// [`Runner::generate_async`], whether it was already running or still
    /// queued in the engine: each is cancelled in the engine, and its stream
    /// ends, without emitting anything new, with an error chunk carrying
    /// [`EngineError::NotConnected`] and "link re-attached; in-flight request
    /// dropped" (chunks it buffered before the swap are delivered first).
    /// Resubmit them. A task given to [`Runner::submit`] directly has no
    /// stream and is not tracked.
    ///
    /// # Cancellation
    ///
    /// Dropping this future before its blocking half has taken the engine
    /// lock cancels the re-attach: nothing is swapped, fenced or bumped.
    /// Once the lock is taken, the re-attach completes even if the future is
    /// dropped: its result is lost, but it is still counted and logged.
    /// A caller that gave up (a `timeout` around this call, say) can simply
    /// call `reattach` again with fresh streams, after closing its ends of
    /// the streams it handed to the dropped call: if that call did swap, the
    /// retry replaces those links the same way any re-attach does, and if it
    /// did not, the retry is the only one. Either way the result is one
    /// completed re-attach on the retry's streams.
    ///
    /// # Errors
    ///
    /// * [`EngineError::PeerRejected`]: validation failed and nothing was
    ///   swapped; the old links are intact. Returned for a runner not in
    ///   stream mode (TCP), for a shape the stage does not have, and for a
    ///   relay keeping its upstream (relay rule).
    /// * [`EngineError::NotLoaded`]: the runner was never started, or was
    ///   closed.
    /// * Any other error: streams may have been swapped and the engine's
    ///   state is unspecified, so the runner is fenced — `submit` and open
    ///   streams fail with [`EngineError::NotConnected`] and a relay parks —
    ///   until a re-attach succeeds. Retry with fresh streams for every link
    ///   this call replaced.
    pub async fn reattach(&self, links: StreamLinks) -> Result<(), EngineError> {
        if !self.stream_mode.load(Ordering::SeqCst) {
            return Err(EngineError::PeerRejected(
                "reattach requires a runner started with start_with_streams".into(),
            ));
        }
        let replaced = links.shape();
        // Cancel safety: set when this future is dropped. The blocking half
        // checks it right after taking the slot lock, so a caller that gave
        // up while it was still waiting never swaps anything.
        let abort = Arc::new(AtomicBool::new(false));
        let _abort_on_drop = AbortOnDrop(abort.clone());
        let (slot, link) = (self.slot.clone(), self.link.clone());
        // The "link re-attached" / failure logs are emitted from the blocking
        // half (so a dropped-but-completed call still logs); carry the
        // caller's span AND subscriber over so they keep its context (a
        // blocking-pool thread only sees the global subscriber otherwise, so a
        // caller with a scoped one would lose them).
        let span = tracing::Span::current();
        let dispatch = tracing::dispatcher::get_default(|d| d.clone());
        tokio::task::spawn_blocking(move || {
            let _dispatch = tracing::dispatcher::set_default(&dispatch);
            let _span = span.enter();
            let _blocking = BlockingContextGuard::enter();
            let mut guard = slot.lock_applying_cancels();
            if abort.load(Ordering::SeqCst) {
                // Nobody is waiting for this result.
                return Err(EngineError::NotConnected);
            }
            let Some(engine) = guard.engine() else {
                return Err(EngineError::NotLoaded);
            };
            // Fence and bump the epoch BEFORE the call, still under the
            // engine lock: if `reattach_streams` panics mid-swap, unwinding
            // drops the guard with the fence up and the epoch moved, so no
            // holder can step or submit against a half-swapped engine (the
            // join error surfaces as Backend) and no older stream survives.
            let was_fenced = slot.fenced.swap(true, Ordering::SeqCst);
            let prev_epoch = slot.attach_epoch.fetch_add(1, Ordering::SeqCst);
            let result = engine.reattach_streams(links);
            if matches!(result, Err(EngineError::PeerRejected(_))) {
                // Validation failure: nothing was swapped; fence and epoch
                // go back to their previous values. Every epoch read holds
                // the engine lock, so its transient bump was never seen. The
                // flag was transiently up, though, and a relay's unlocked
                // pre-check may have read it and parked: bump the generation
                // (still under the slot lock, order slot → link) so that
                // relay re-checks. A spurious wake is harmless.
                slot.fenced.store(was_fenced, Ordering::SeqCst);
                slot.attach_epoch.store(prev_epoch, Ordering::SeqCst);
                let mut st = link.state.lock();
                st.generation += 1;
                link.cv.notify_all();
                return result;
            }
            // Every task admitted before this call ran on the old links and
            // the engine's old session, so none of them can finish: cancel
            // each in the engine now, still under its lock, so no later step
            // admits a queued one, and mark them orphaned so their streams
            // fail with the re-attach error instead of emitting anything new.
            let stale: Vec<TaskId> = {
                let mut bufs = slot.buffers.lock();
                let stale: Vec<TaskId> = bufs.admitted.drain().collect();
                bufs.orphaned.extend(stale.iter().cloned());
                stale
            };
            for tid in &stale {
                engine.cancel(tid);
            }
            // Fence (or unfence) while still holding the engine lock: every
            // other engine holder reads the flag after acquiring that lock,
            // so none can step or submit against a half-reset engine.
            slot.fenced.store(result.is_err(), Ordering::SeqCst);
            // Still under the slot lock (order: slot → link).
            let generation = {
                let mut st = link.state.lock();
                st.generation += 1;
                link.cv.notify_all();
                st.generation
            };
            // Metric and logs here, not after the await: once the lock was
            // taken this re-attach completes even if the caller dropped the
            // future, and it must still be counted and logged.
            match &result {
                Ok(()) => {
                    let counts = [
                        ("upstream", u64::from(replaced.upstream)),
                        ("downstream", u64::from(replaced.downstream)),
                        ("ep_driver", u64::from(replaced.ep_driver)),
                        ("ep_worker", replaced.ep_workers as u64),
                    ];
                    for (side, n) in counts {
                        if n > 0 {
                            cascadia_metrics::LINK_REATTACH_TOTAL
                                .with_label_values(&[side])
                                .inc_by(n);
                        }
                    }
                    info!(generation, links = ?replaced, dropped_requests = stale.len(), "link re-attached");
                }
                Err(e) => {
                    warn!(error = %e, "re-attach failed after swapping streams; engine fenced until a successful re-attach");
                }
            }
            result
        })
        .await
        .map_err(|e| EngineError::Backend(format!("reattach join: {e}")))?
    }

    /// Park until the link generation moves past `snap` or the runner closes.
    fn wait_for_link_change(&self, snap: u64) {
        let mut st = self.link.state.lock();
        while st.generation == snap && !st.closed {
            self.link.cv.wait(&mut st);
        }
    }

    /// See [`EngineSlot::is_fenced`].
    fn is_fenced(&self) -> bool {
        self.slot.is_fenced()
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
    /// In stream mode (issue #76) a connection-fatal step does not exit: the
    /// loop parks until [`Runner::reattach`] or [`Runner::close`].
    ///
    /// Enters a `BlockingContextGuard` once per OS thread (since this
    /// loop runs on a single `spawn_blocking` thread) so that engines'
    /// `run_async` calls hit the naked-`block_on` path instead of
    /// `block_in_place` — ~60 ms/frame savings on Windows.
    pub fn run_relay_loop(&self) -> RelayExit {
        let _blocking = BlockingContextGuard::enter();
        let stream_mode = self.stream_mode.load(Ordering::SeqCst);
        let mut link_down = LinkDownLog::default();
        loop {
            // Stream mode: snapshot the link generation BEFORE stepping, so a
            // re-attach that lands between a fatal step and the wait below is
            // never lost. The fenced check here only avoids a needless lock;
            // the binding check is under the slot lock below.
            let snap = if stream_mode {
                let st = self.link.state.lock();
                if st.closed {
                    info!("relay loop exited: runner closed");
                    return RelayExit::SlotEmpty;
                }
                let generation = st.generation;
                drop(st);
                // Generation is read before the flag. Invariant: every path
                // that changes or transiently sets the flag bumps the
                // generation under the link mutex before releasing the slot
                // lock — so whatever flag value this read observes, the
                // generation has moved past `generation` by the time the
                // re-attach is over, and the wait returns.
                // Tests hook this read to pin the interleaving with a re-attach.
                #[cfg(test)]
                let fenced = self.link.hooked_fence_read(|| self.is_fenced());
                #[cfg(not(test))]
                let fenced = self.is_fenced();
                if fenced {
                    self.wait_for_link_change(generation);
                    continue;
                }
                generation
            } else {
                0
            };
            // Through the guard like every other holder: a relay-stage
            // runner serves no streams today, so it has nothing to wake —
            // but a runner that both relays and generates would strand
            // every parked stream behind this loop, and the guard makes
            // that impossible for free. It is also the only thing that ever
            // drains a relay-stage runner's deferred cancels: nobody polls a
            // stream here, so without this round the queue only grows.
            let mut guard = self.slot.lock_applying_cancels();
            if stream_mode && self.is_fenced() {
                drop(guard);
                self.wait_for_link_change(snap);
                continue;
            }
            let Some(engine) = guard.engine() else {
                drop(guard);
                info!("relay loop exited: engine slot empty");
                return RelayExit::SlotEmpty;
            };
            // Engine.step is sync; just drain. A non-fatal Err is logged
            // but the loop keeps driving — relay engines recover their own
            // state and a transient frame error must not take the stage
            // down. A connection-fatal Err is unrecoverable in-process in
            // TCP mode, so bail and let the supervisor rebuild us; in
            // stream mode park until the embedder re-attaches (or closes).
            let failed = match engine.step() {
                Ok(_) => false,
                Err(e) if e.is_connection_fatal() => {
                    // Read under the engine lock, like every epoch read.
                    let epoch = self.slot.attach_epoch();
                    drop(guard);
                    if stream_mode {
                        if link_down.first_for(epoch) {
                            warn!(error = %e, "relay step hit a dead peer link; waiting for re-attach");
                        } else {
                            debug!(error = %e, "relay step still on a dead peer link; waiting for re-attach");
                        }
                        self.wait_for_link_change(snap);
                        continue;
                    }
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

    /// Tear the engine down. Later calls fail with [`EngineError::NotLoaded`];
    /// in-flight streams end with a "server is shutting down" error chunk.
    ///
    /// Mandatory in stream mode (issue #76): it is the only thing besides
    /// [`Runner::reattach`] that wakes a relay parked on a dead link, so
    /// skipping it leaves that `spawn_blocking` thread parked forever and
    /// tokio runtime shutdown hangs waiting for it.
    ///
    /// Blocks until no step holds the engine. A step blocked on a link read
    /// (an idle relay waiting on its upstream) holds it until that link
    /// delivers, dies or hits the frame idle ceiling; with injected streams,
    /// close your ends of them first so `close` returns at once. After
    /// 5 s (`CLOSE_WARN_AFTER`) of waiting it logs one warning naming that cause,
    /// then keeps waiting.
    pub fn close(&self) {
        // BEFORE the slot empties, so no in-flight stream can observe a gone
        // engine without also seeing that this is a shutdown. Ordering is the
        // whole point: set it after, and a stream that polls in between books
        // a client cancellation for a server restart.
        self.closing.store(true, Ordering::SeqCst);
        // Issue #76: wake a stream-mode relay parked on a dead link. Done
        // under the link mutex (its predicate reads `closed`), and before the
        // slot lock, never while holding it in the reverse order.
        {
            let mut st = self.link.state.lock();
            st.closed = true;
            self.link.cv.notify_all();
        }
        {
            // Tear down and empty the slot under ONE guard, where this used
            // to take the lock twice. Releasing in between now means waking,
            // and a stream woken there would find the engine closed but
            // still present, step a torn-down transport, and book a failure
            // instead of the teardown outcome. One acquisition closes that
            // window; the single release still wakes the streams parked on
            // contention, which must re-poll to observe the emptied slot or
            // a teardown would strand them Pending forever.
            //
            // The lock can be held by a step blocked on a link read (an idle
            // relay waits on its upstream for the next frame), which only
            // returns when that link delivers, dies, or hits the frame idle
            // ceiling. Say so once if it takes a while, then keep waiting.
            let mut guard = self.slot.lock_noting_delay(CLOSE_WARN_AFTER, || {
                warn!(
                    waited_secs = CLOSE_WARN_AFTER.as_secs(),
                    "close() is still waiting for the engine lock: a step is blocked on a \
                     peer link (an idle relay waits on its upstream for the next frame). \
                     It returns when that link delivers or dies, or at the frame idle \
                     ceiling (CASCADIA_FRAME_IDLE_CEILING_SECS, default 900 s); with \
                     injected streams, close your ends of them to release it now"
                );
            });
            if let Some(engine) = guard.engine() {
                engine.close();
            }
            guard.set(None);
        }
        if let Some(builder) = self.builder.lock().as_mut() {
            builder.close();
        }
    }
}

/// Cancels a task on drop unless disarmed — the ownership hand-off between
/// "submitted" and "a [`ChunkStream`] exists to own it".
///
/// [`Runner::generate_async`] arms one before dispatching its submit, so the
/// window in which a task is submitted but unowned is covered by an RAII
/// cancel rather than by hoping the caller's future is never dropped. It is
/// disarmed the instant a `ChunkStream` exists, because from then on that
/// stream's own `Drop` is the cancel path.
///
/// `Drop` may run on any thread (a dropped axum handler future lands on
/// whichever worker polled it last), so it must never block: it goes through
/// [`Runner::cancel`], which try_locks the engine and defers when a step is
/// in flight (#122).
struct SubmitCancelGuard {
    runner: Arc<Runner>,
    task_id: TaskId,
    armed: bool,
}

impl SubmitCancelGuard {
    /// The task found an owner (or was never admitted): stand down.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SubmitCancelGuard {
    fn drop(&mut self) {
        if self.armed {
            // Writes the `cancelled` tombstone and cancels the engine-side
            // task (or defers the engine half). The tombstone is what makes a
            // submit that has not run yet refuse the task — see the check in
            // `Runner::submit`.
            self.runner.cancel(&self.task_id);
        }
    }
}

/// Sets its flag on drop: [`Runner::reattach`]'s "my caller is gone" signal
/// to its detached blocking half. Also set on normal completion, when the
/// blocking half has already finished and no longer reads it.
struct AbortOnDrop(Arc<AtomicBool>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

pub struct ChunkStream {
    task_id: TaskId,
    /// The engine slot, shared with the [`Runner`] that made this stream and
    /// with every sibling stream. One `Arc` rather than loose clones of the
    /// engine and buffer handles, so a stream reaches the engine through the
    /// same [`EngineGuard`] every other holder uses and cannot re-implement
    /// the release protocol by hand.
    slot: Arc<EngineSlot>,
    /// Issue #76: the [`EngineSlot::attach_epoch`] the task was admitted
    /// under. A different value at a later poll means the links were swapped
    /// (and the engine's session reset) under this request.
    epoch: u64,
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
        self.slot.buffers.lock().chunks.remove(&self.task_id);
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
        self.slot.buffers.lock().chunks.remove(&self.task_id);
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

/// Why [`ChunkStream::poll_next`]'s step round produced no chunks.
enum StepFail {
    /// The engine's `step()` (or the fence) failed.
    Engine(EngineError),
    /// The links were re-attached since this stream's task was admitted.
    Orphaned,
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
                let mut bufs = this.slot.buffers.lock();
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
                //    Acquiring also applies the cancels deferred while the
                //    engine was busy (Runner::cancel / stream Drop never
                //    block on the engine mutex — see deferred_cancels), so
                //    this step doesn't spend a round on abandoned tasks.
                let mut guard = match this.slot.try_lock_applying_cancels() {
                    Some(g) => g,
                    None => {
                        this.slot
                            .buffers
                            .lock()
                            .step_wakers
                            .push(cx.waker().clone());
                        // Re-check after registering: the holder may have
                        // released (and drained wakers) between the failed
                        // try_lock and the registration. A stale registration
                        // from the success arm only costs a spurious wake.
                        // No guard exists on this path, so returning Pending
                        // does NOT wake — waking the waker just registered
                        // would spin this worker for the holder's whole step.
                        match this.slot.try_lock_applying_cancels() {
                            Some(g) => g,
                            None => return Poll::Pending,
                        }
                    }
                };
                let result = match guard.engine() {
                    // Issue #76: the links were swapped since this task was
                    // admitted, so the engine reset the session it ran in.
                    // Read under the engine lock; fail without stepping.
                    Some(_) if this.slot.attach_epoch() != this.epoch => {
                        Some(Err(StepFail::Orphaned))
                    }
                    // Issue #76: a failed post-swap re-attach fenced the
                    // engine. Read under the engine lock; end this stream
                    // through the own-task error path without stepping.
                    Some(_) if this.slot.is_fenced() => {
                        Some(Err(StepFail::Engine(EngineError::NotConnected)))
                    }
                    // Slot empty: `Runner::close` took the engine, i.e. server
                    // teardown. Handled below rather than here because
                    // `fail_teardown` needs `&mut *this` and this guard still
                    // borrows `this.slot`.
                    None => None,
                    Some(engine) => Some(match engine.step() {
                        Ok(produced) => {
                            let empty = produced.is_empty();
                            let mut bufs = this.slot.buffers.lock();
                            for (tid, chunk) in produced {
                                // Issue #76: an orphaned task's stream ends
                                // with the re-attach error, never new output.
                                if bufs.cancelled.contains(&tid) || bufs.orphaned.contains(&tid) {
                                    continue;
                                }
                                bufs.chunks.entry(tid).or_default().push_back(chunk);
                            }
                            Ok(empty)
                        }
                        Err(e) => Err(StepFail::Engine(e)),
                    }),
                };
                // Releasing the guard is what drains + wakes every stream
                // parked on contention — engine first, wake second, and on
                // an unwind out of `step()` just the same.
                drop(guard);
                result
            };
            let Some(step_result) = step_result else {
                return this.fail_teardown();
            };
            let produced_empty = match step_result {
                Ok(empty) => empty,
                Err(StepFail::Orphaned) => {
                    warn!(
                        task = %this.task_id,
                        "link re-attached under an in-flight request; failing its stream"
                    );
                    return this.fail_stream(format!(
                        "link re-attached; in-flight request dropped ({})",
                        EngineError::NotConnected
                    ));
                }
                Err(StepFail::Engine(e)) => match e.task_id() {
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
                        let mut bufs = this.slot.buffers.lock();
                        // Cancelled task = no consumer left; recreating its
                        // buffer entry would leak until close() (mirror the
                        // cancelled-check on the distribution path below).
                        // Issue #76: an orphaned task's stream ends with the
                        // re-attach error (its own poll checks the epoch
                        // before anything else), never a routed engine error.
                        if !bufs.cancelled.contains(&failed) && !bufs.orphaned.contains(&failed) {
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
        // engine-lock holder applies it before stepping. Same helper
        // `Runner::cancel` uses, and the guard inside it handles the
        // release-then-wake ordering for both.
        self.slot.cancel_or_defer(&self.task_id);
        let mut bufs = self.slot.buffers.lock();
        bufs.chunks.remove(&self.task_id);
        bufs.admitted.remove(&self.task_id);
        bufs.orphaned.remove(&self.task_id);
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
        let runner = Runner::for_test(engine, false);

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
        let runner = Runner::for_test(engine, false);

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
            let runner = Runner::for_test(engine, false);
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
            !runner.slot.buffers.lock().chunks.contains_key("dead"),
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
        while runner.slot.buffers.lock().step_wakers.is_empty() {
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

    /// Engine that blocks inside `step()` until the test releases it and
    /// records every `cancel()` it is handed. Lets a test hold the engine
    /// lock at a chosen moment, then observe exactly which later lock
    /// acquisition delivered a deferred cancel.
    struct GatedCancelEngine {
        serve: TaskId,
        entered: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        cancels: Arc<Mutex<Vec<TaskId>>>,
    }

    impl Engine for GatedCancelEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn cancel(&mut self, task_id: &TaskId) {
            self.cancels.lock().push(task_id.clone());
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.entered.store(true, Ordering::SeqCst);
            // Bounded, so a broken test fails rather than hangs.
            let start = Instant::now();
            while !self.release.load(Ordering::SeqCst)
                && start.elapsed() < std::time::Duration::from_secs(10)
            {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Ok(vec![(
                self.serve.clone(),
                Chunk::token(self.serve.clone(), 0, "x"),
            )])
        }
    }

    struct GatedCancelBuilder {
        serve: TaskId,
        entered: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        cancels: Arc<Mutex<Vec<TaskId>>>,
    }

    #[async_trait::async_trait]
    impl Builder for GatedCancelBuilder {
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
            Ok(Box::new(GatedCancelEngine {
                serve: self.serve,
                entered: self.entered,
                release: self.release,
                cancels: self.cancels,
            }))
        }
    }

    /// A cancel deferred past a busy engine must be applied by the next
    /// engine-lock ACQUISITION, not only by the next stream poll.
    ///
    /// Cancelling the sole in-flight request leaves nobody polling, so a
    /// poll-only drain kept the task's engine-side slot and KV region
    /// occupied until some unrelated request happened along — hours, on an
    /// idle server — while the queue's own contract said "the next
    /// engine-lock holder". A plain `submit` is such a holder.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn deferred_cancel_is_applied_by_the_next_lock_acquisition() {
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let polled = Arc::new(AtomicBool::new(false));
        let cancels = Arc::new(Mutex::new(Vec::new()));
        let runner = Arc::new(Runner::new(Box::new(GatedCancelBuilder {
            serve: "holder".to_string(),
            entered: entered.clone(),
            release: release.clone(),
            cancels: cancels.clone(),
        })));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();

        let holder = runner
            .generate(GenerationTask::new("holder", "x").with_max_tokens(64))
            .unwrap();
        // Exactly ONE manual poll, on a blocking thread: one engine-lock
        // acquisition, and the stream is kept alive afterwards so its Drop
        // (which also takes the lock) can't be what applies the cancel.
        let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
        let polled_by_task = polled.clone();
        let poll = tokio::task::spawn_blocking(move || {
            let mut s = holder;
            let waker = futures::task::noop_waker();
            let mut cx = Context::from_waker(&waker);
            let out = Pin::new(&mut s).poll_next(&mut cx);
            polled_by_task.store(true, Ordering::SeqCst);
            let _ = hold_rx.recv();
            out.is_ready()
        });

        // Wait until the poll is inside step(), holding the engine lock.
        let start = Instant::now();
        while !entered.load(Ordering::SeqCst) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "the holding stream never entered step()"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }

        // Mid-step: this cancel cannot take the lock, so it is deferred.
        // Twice, because the queue is a to-do list, not a log.
        runner.cancel(&"victim".to_string());
        runner.cancel(&"victim".to_string());
        assert_eq!(
            runner.slot.buffers.lock().deferred_cancels,
            vec!["victim".to_string()],
            "a cancel behind a busy engine must be deferred, and deduped"
        );
        assert!(
            cancels.lock().is_empty(),
            "the engine was mid-step; no cancel could have reached it yet"
        );

        // Let the step finish. The poll returns, releasing the engine.
        release.store(true, Ordering::SeqCst);
        let start = Instant::now();
        while !polled.load(Ordering::SeqCst) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(10),
                "the holding stream's poll never returned"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        assert!(
            cancels.lock().is_empty(),
            "nothing has taken the engine lock since the cancel was deferred"
        );

        // A plain submit — no stream polled anywhere — is the next holder.
        runner
            .submit(GenerationTask::new("later", "y").with_max_tokens(1))
            .unwrap();
        assert_eq!(
            &*cancels.lock(),
            &["victim".to_string()],
            "the next engine-lock acquisition must apply the deferred cancel"
        );
        assert!(
            runner.slot.buffers.lock().deferred_cancels.is_empty(),
            "applying the queue must clear it"
        );

        let _ = hold_tx.send(());
        assert!(poll.await.unwrap(), "the gated poll should have produced");
    }

    /// The deferred-cancel queue is a bounded to-do list. A runner whose
    /// engine nobody locks (a relay stage taking cancel() calls) would
    /// otherwise accrete one entry per cancelled request for the life of
    /// the process.
    #[test]
    fn deferred_cancel_queue_dedups_and_is_bounded() {
        let slot = EngineSlot::new(None);
        for _ in 0..8 {
            slot.defer_cancel(&"same".to_string());
        }
        assert_eq!(
            slot.buffers.lock().deferred_cancels,
            vec!["same".to_string()],
            "re-cancelling one task must not queue it twice"
        );
        for i in 0..MAX_DEFERRED_CANCELS + 16 {
            slot.defer_cancel(&format!("t{i}"));
        }
        assert_eq!(
            slot.buffers.lock().deferred_cancels.len(),
            MAX_DEFERRED_CANCELS,
            "the queue must stop growing at its cap"
        );
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

    // -----------------------------------------------------------------
    // H1: `generate_async` cancellation-safety.
    //
    // `spawn_blocking(...).await` is a cancellation point that can pend for
    // a full engine step. Dropping the caller's future there DETACHES the
    // closure rather than cancelling it, so the submit still runs — and
    // before the fix no `ChunkStream` was ever constructed for the task, so
    // nothing cancelled it and nothing drained it.
    // -----------------------------------------------------------------

    /// What the engine was actually asked to do. Enough to tell "never
    /// admitted" from "admitted and ground through tokens nobody read".
    #[derive(Default, Debug)]
    struct EngineLog {
        submitted: Vec<TaskId>,
        cancelled: Vec<TaskId>,
        produced: Vec<TaskId>,
    }

    /// Lets a test pin the engine mutex down for a deterministic window:
    /// `step()` parks inside the lock until the gate opens.
    #[derive(Default)]
    struct Gate {
        open: AtomicBool,
        entered: std::sync::atomic::AtomicUsize,
    }

    impl Gate {
        fn opened() -> Arc<Self> {
            let g = Arc::new(Gate::default());
            g.open.store(true, Ordering::SeqCst);
            g
        }
        fn closed() -> Arc<Self> {
            Arc::new(Gate::default())
        }
        fn release(&self) {
            self.open.store(true, Ordering::SeqCst);
        }
        fn steps_entered(&self) -> usize {
            self.entered.load(Ordering::SeqCst)
        }
    }

    /// Recording engine: one token per active task per `step()`, final marker
    /// at `max_tokens`, and a log of every submit / cancel / emission.
    struct RecordingEngine {
        log: Arc<Mutex<EngineLog>>,
        gate: Arc<Gate>,
        active: Vec<(GenerationTask, u32)>,
    }

    impl Engine for RecordingEngine {
        fn warmup(&mut self) {}

        fn submit(&mut self, task: GenerationTask) -> Result<(), EngineError> {
            self.log.lock().submitted.push(task.task_id.clone());
            self.active.push((task, 0));
            Ok(())
        }

        fn cancel(&mut self, task_id: &TaskId) {
            self.log.lock().cancelled.push(task_id.clone());
            // An id the engine never admitted is a harmless no-op here, as
            // the `Engine::cancel` contract requires and every real impl
            // (ov-runtime, genai, dist_spec, sparse-moe) implements via
            // `retain` / an id-matched conditional.
            self.active.retain(|(t, _)| &t.task_id != task_id);
        }

        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.gate.entered.fetch_add(1, Ordering::SeqCst);
            while !self.gate.open.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            if self.active.is_empty() {
                return Ok(Vec::new());
            }
            let mut out = Vec::new();
            let mut still = Vec::new();
            let mut produced = Vec::new();
            for (task, emitted) in self.active.drain(..) {
                let n = emitted + 1;
                produced.push(task.task_id.clone());
                out.push((
                    task.task_id.clone(),
                    Chunk::token(&task.task_id, n as i64, "x "),
                ));
                if n >= task.max_tokens {
                    out.push((task.task_id.clone(), Chunk::final_marker(&task.task_id, "")));
                } else {
                    still.push((task, n));
                }
            }
            self.active = still;
            self.log.lock().produced.extend(produced);
            Ok(out)
        }
    }

    struct RecordingBuilder {
        log: Arc<Mutex<EngineLog>>,
        gate: Arc<Gate>,
    }

    #[async_trait::async_trait]
    impl Builder for RecordingBuilder {
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
            Ok(Box::new(RecordingEngine {
                log: self.log,
                gate: self.gate,
                active: Vec::new(),
            }))
        }
    }

    async fn recording_runner(gate: Arc<Gate>) -> (Arc<Runner>, Arc<Mutex<EngineLog>>) {
        let log = Arc::new(Mutex::new(EngineLog::default()));
        let runner = Arc::new(Runner::new(Box::new(RecordingBuilder {
            log: log.clone(),
            gate,
        })));
        runner
            .start(
                PeerLayout::single_stage(),
                ShardSpec::single_stage("m", "CPU"),
            )
            .await
            .unwrap();
        (runner, log)
    }

    /// Poll `cond` until it holds, or fail the test. Used instead of a fixed
    /// sleep so the blocking-pool hand-offs below are waited on, not guessed.
    async fn await_until(label: &str, mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {label}");
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    }

    /// (a) The caller's future is dropped while its submit is parked on the
    /// engine mutex behind another stream's in-flight step — the axum
    /// client-disconnect shape, hitting at the moment the wait is longest.
    ///
    /// The detached submit runs anyway. It must not leave a ghost: no engine
    /// admission, no chunks accreting under an id nobody drains, once the
    /// lock churns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dropped_generate_async_leaves_no_ghost_task() {
        let gate = Gate::closed();
        let (runner, log) = recording_runner(gate.clone()).await;

        // A holder stream takes the engine mutex and parks inside `step()`.
        let holder_runner = runner.clone();
        let holder = tokio::spawn(async move {
            let mut s = holder_runner
                .generate_async(GenerationTask::new("holder", "x").with_max_tokens(4))
                .await
                .unwrap();
            let mut n = 0usize;
            while let Some(c) = s.next().await {
                assert!(c.error.is_none(), "holder failed: {:?}", c.error);
                n += 1;
            }
            n
        });
        await_until("the holder to enter step() with the engine locked", || {
            gate.steps_entered() >= 1
        })
        .await;

        // The engine mutex is held for the whole of this window, so the
        // ghost's submit cannot complete and the caller goes away first.
        let dropped = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            runner.generate_async(GenerationTask::new("ghost", "x").with_max_tokens(64)),
        )
        .await;
        assert!(
            dropped.is_err(),
            "the submit must still be in flight when the caller's future is dropped"
        );
        // Release BEFORE asserting: the holder is parked inside `step()` with
        // the engine mutex held, so a panic here would wedge the runtime's
        // teardown instead of reporting the failure. `timeout` has already
        // dropped the inner future (and run the guard) by the time it returns
        // Err, so the window under test is closed either way.
        gate.release();
        assert!(
            runner.slot.buffers.lock().cancelled.contains("ghost"),
            "the drop guard must have tombstoned the abandoned task"
        );

        // Churn the lock: the holder finishes, then two more full generations
        // run. Every one of those polls would step a ghost that got admitted.
        assert!(holder.await.unwrap() > 0);
        for i in 0..2 {
            let mut s = runner
                .generate_async(GenerationTask::new(format!("after-{i}"), "x").with_max_tokens(2))
                .await
                .unwrap();
            while s.next().await.is_some() {}
        }

        let log = log.lock();
        assert!(
            !log.produced.iter().any(|t| t == "ghost"),
            "abandoned task was admitted and generated tokens nobody reads: {log:?}"
        );
        assert!(
            !log.submitted.iter().any(|t| t == "ghost")
                || log.cancelled.iter().any(|t| t == "ghost"),
            "abandoned task was neither refused at submit nor cancelled: {log:?}"
        );
        drop(log);
        assert!(
            !runner.slot.buffers.lock().chunks.contains_key("ghost"),
            "chunks accreted for a task with no owner"
        );
    }

    /// The other half of (a): the submit LANDS, and only then is the caller's
    /// future dropped. Nothing refuses it, so the drop guard is the only
    /// thing that can retire the task — it must cancel it at the engine.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_async_dropped_after_submit_landed_cancels_the_task() {
        let (runner, log) = recording_runner(Gate::opened()).await;

        {
            let fut = runner.generate_async(GenerationTask::new("ghost", "x").with_max_tokens(64));
            futures::pin_mut!(fut);
            // First poll dispatches the submit; the join is still pending.
            assert!(futures::poll!(fut.as_mut()).is_pending());
            await_until("the detached submit to reach the engine", || {
                log.lock().submitted.iter().any(|t| t == "ghost")
            })
            .await;
            // Dropped without ever being polled again — no stream is built.
        }

        assert!(
            log.lock().cancelled.iter().any(|t| t == "ghost"),
            "an admitted-but-unowned task must be cancelled at the engine: {:?}",
            log.lock()
        );
        // Churn the engine: a still-admitted ghost would generate here.
        let mut s = runner
            .generate_async(GenerationTask::new("after", "x").with_max_tokens(2))
            .await
            .unwrap();
        while s.next().await.is_some() {}
        assert!(
            !log.lock().produced.iter().any(|t| t == "ghost"),
            "cancelled ghost still generated: {:?}",
            log.lock()
        );
    }

    /// (b), unit form: a tombstoned id is refused inside `submit` — a benign
    /// no-op `Ok(())`, and the engine never sees the task. A fresh id still
    /// lands, so the check refuses exactly one thing.
    #[tokio::test]
    async fn submit_refuses_an_already_tombstoned_id() {
        let (runner, log) = recording_runner(Gate::opened()).await;
        runner.cancel(&"ghost".to_string());
        runner
            .submit(GenerationTask::new("ghost", "x").with_max_tokens(8))
            .expect("a refused submit is a no-op Ok, not an error");
        runner
            .submit(GenerationTask::new("live", "x").with_max_tokens(8))
            .unwrap();
        let log = log.lock();
        assert!(
            !log.submitted.iter().any(|t| t == "ghost"),
            "a cancelled task must never reach the engine: {log:?}"
        );
        assert!(
            log.submitted.iter().any(|t| t == "live"),
            "the tombstone check must not refuse healthy submits: {log:?}"
        );
    }

    /// (b), the real ordering race: the caller's future is dropped BEFORE the
    /// submit closure has run at all, so the guard's cancel executes FIRST
    /// and the submit would admit the task afterwards. Forced deterministic
    /// by capping the blocking pool at one thread and occupying it, which is
    /// exactly the production shape (every worker busy) that makes the race
    /// reachable.
    #[test]
    fn submit_queued_behind_a_drop_is_refused_when_it_finally_runs() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let (runner, log) = recording_runner(Gate::opened()).await;

            // Occupy the pool's only blocking thread.
            let (release, blocked) = std::sync::mpsc::channel::<()>();
            let started = Arc::new(AtomicBool::new(false));
            let started_in = started.clone();
            let occupier = tokio::task::spawn_blocking(move || {
                started_in.store(true, Ordering::SeqCst);
                let _ = blocked.recv();
            });
            await_until("the blocking pool's only thread to be occupied", || {
                started.load(Ordering::SeqCst)
            })
            .await;

            // Dispatch + drop. The submit closure is queued, never run.
            let dropped = tokio::time::timeout(
                std::time::Duration::from_millis(50),
                runner.generate_async(GenerationTask::new("ghost", "x").with_max_tokens(64)),
            )
            .await;
            assert!(dropped.is_err());
            assert!(
                log.lock().submitted.is_empty(),
                "the submit closure must still be queued behind the occupier"
            );
            assert!(runner.slot.buffers.lock().cancelled.contains("ghost"));

            // Release the pool. The blocking queue is FIFO, so the barrier
            // below can only run once the ghost's submit closure has finished.
            drop(release);
            occupier.await.unwrap();
            tokio::task::spawn_blocking(|| {}).await.unwrap();

            assert!(
                !log.lock().submitted.iter().any(|t| t == "ghost"),
                "a submit that runs after its caller cancelled must be refused: {:?}",
                log.lock()
            );

            // And the runner is still usable for real work afterwards.
            let mut s = runner
                .generate_async(GenerationTask::new("fresh", "x").with_max_tokens(2))
                .await
                .unwrap();
            let mut n = 0usize;
            while let Some(c) = s.next().await {
                assert!(c.error.is_none(), "{:?}", c.error);
                n += 1;
            }
            assert!(n > 0, "a fresh request must still stream");
        });
    }

    /// (c) The normal path is untouched: polled to completion,
    /// `generate_async` streams exactly what it always did.
    #[tokio::test]
    async fn generate_async_polled_to_completion_is_unaffected() {
        let runner = Arc::new(make_runner().await);
        let task = GenerationTask::new("t-ga", "the quick brown fox").with_max_tokens(2);
        let mut stream = runner.generate_async(task).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(c) = stream.next().await {
            chunks.push(c);
        }
        // Two tokens then a final marker — same as the sync `generate()`.
        // A guard that failed to disarm would have cancelled us to zero.
        assert_eq!(chunks.len(), 3, "{chunks:?}");
        assert_eq!(chunks[0].text, "the ");
        assert_eq!(chunks[1].text, "quick ");
        assert!(chunks.last().unwrap().is_final);
        assert!(chunks.iter().all(|c| c.error.is_none()));
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

    // -----------------------------------------------------------------
    // M7(b): `close()` while a stream is parked Pending.
    //
    // A stream that parked on engine contention is woken by exactly one
    // thing: an engine-lock RELEASE (`EngineGuard::drop` drains
    // `step_wakers`). Teardown is the release with nobody coming after it —
    // if `close()` ever empties the slot without paying that drain, every
    // parked stream waits `Pending` forever with no error, no log and no
    // later holder to rescue it, and its client hangs until it gives up.
    // Every wait below is bounded, so that regression FAILS the test
    // instead of hanging the suite.
    // -----------------------------------------------------------------

    /// Stream A pins the engine inside a gated `step()`; stream B parks on
    /// the contention; `close()` starts while B is parked. B must wake and
    /// terminate on the teardown outcome — an error chunk saying the server
    /// is going away, not a bare end-of-stream (which the API layer would
    /// turn into a truncated 200) and not silence.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn close_wakes_a_stream_parked_on_engine_contention() {
        let gate = Gate::closed();
        let (runner, _log) = recording_runner(gate.clone()).await;

        // Both tasks are submitted up front: `generate_async` takes the
        // engine lock only for the submit itself, never for a step, so this
        // runs before anyone is holding it.
        let mut a = runner
            .generate_async(GenerationTask::new("parked-holder", "x").with_max_tokens(2))
            .await
            .unwrap();
        let mut b = runner
            // Far more tokens than this test will ever let it produce: B
            // cannot reach a normal final chunk, so the ONLY terminal
            // outcome available to it is the teardown one.
            .generate_async(GenerationTask::new("parked-waiter", "x").with_max_tokens(1_000_000))
            .await
            .unwrap();

        // A takes the engine lock and parks inside `step()`, holding it.
        let holder = tokio::spawn(async move {
            let mut n = 0usize;
            while a.next().await.is_some() {
                n += 1;
            }
            n
        });
        await_until("the holder to enter step() with the engine locked", || {
            gate.steps_entered() >= 1
        })
        .await;

        // B polls, fails its `try_lock`, registers a waker and parks. From
        // here nothing but a release can move it.
        let waiter = tokio::spawn(async move {
            let mut last = None;
            while let Some(c) = b.next().await {
                last = Some(c);
            }
            last
        });
        await_until("the waiter to park on engine contention", || {
            !runner.slot.buffers.lock().step_wakers.is_empty()
        })
        .await;

        // Teardown begins WHILE B is parked. `close()` blocks on the engine
        // lock until the gate opens, so it needs its own thread; the
        // `closing` flag it sets first (before it reaches for the lock) is
        // the handshake that it is really in flight.
        let closer = {
            let runner = runner.clone();
            tokio::task::spawn_blocking(move || runner.close())
        };
        await_until("close() to start tearing the runner down", || {
            runner.closing.load(Ordering::SeqCst)
        })
        .await;

        // Released LAST, so B spends the whole window parked with a
        // teardown already underway. Two releases follow — A's and
        // `close()`'s — and B must be woken by one of them.
        gate.release();

        let last = tokio::time::timeout(std::time::Duration::from_secs(10), waiter)
            .await
            .expect("the parked stream never woke: teardown stranded it Pending")
            .unwrap()
            .expect("the parked stream ended with no chunk at all (a truncated success)");
        assert_eq!(
            last.error.as_deref(),
            Some("server is shutting down"),
            "a stream parked across teardown must end on the teardown error: {last:?}"
        );
        assert!(last.is_final, "{last:?}");

        closer.await.unwrap();
        // The holder must not be stranded by the teardown either.
        tokio::time::timeout(std::time::Duration::from_secs(10), holder)
            .await
            .expect("the lock holder never finished after teardown")
            .unwrap();
    }

    /// Builder that records which connect path the runner used.
    struct PathBuilder(Arc<parking_lot::Mutex<Vec<&'static str>>>);

    #[async_trait::async_trait]
    impl Builder for PathBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            self.0.lock().push("tcp");
            Ok(())
        }
        async fn connect_streams(
            &mut self,
            links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            assert!(links.upstream.is_some() && links.downstream.is_none());
            self.0.lock().push("streams");
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

    fn test_shard() -> ShardSpec {
        ShardSpec::single_stage("m", "CPU")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn start_with_streams_uses_connect_streams_and_sets_stream_mode() {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let runner = Runner::new(Box::new(PathBuilder(log.clone())));
        let (a, _b) = tokio::io::duplex(64);
        runner
            .start_with_streams(
                cascadia_engine::StreamLinks::pipeline(Some(Box::new(a)), None),
                test_shard(),
            )
            .await
            .unwrap();
        assert_eq!(*log.lock(), vec!["streams"]);
        assert!(runner.stream_mode.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn start_with_listen_leaves_stream_mode_off() {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let runner = Runner::new(Box::new(PathBuilder(log.clone())));
        runner
            .start_with_listen(PeerLayout::single_stage(), test_shard(), None)
            .await
            .unwrap();
        assert_eq!(*log.lock(), vec!["tcp"]);
        assert!(!runner.stream_mode.load(Ordering::SeqCst));
    }

    use std::sync::atomic::AtomicBool as Flag;

    /// Engine whose link is dead while `dead` is set. `reattach_streams`
    /// revives it, or fails with Backend when `fail_reattach` is set.
    struct LinkEngine {
        dead: Arc<Flag>,
        fail_reattach: Arc<Flag>,
        panic_reattach: Arc<Flag>,
        steps: Arc<std::sync::atomic::AtomicUsize>,
        submits: Arc<std::sync::atomic::AtomicUsize>,
        reattaches: Arc<std::sync::atomic::AtomicUsize>,
        /// Signals each step (capacity 1, dropped when full), so a test can
        /// wait for a step, or for its absence, instead of sleeping.
        stepped: std::sync::mpsc::SyncSender<()>,
    }

    impl Engine for LinkEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            self.submits.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.steps.fetch_add(1, Ordering::SeqCst);
            let _ = self.stepped.try_send(());
            if self.dead.load(Ordering::SeqCst) {
                Err(EngineError::Backend("socket closed during recv".into()))
            } else {
                Ok(vec![])
            }
        }
        fn reattach_streams(
            &mut self,
            links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            self.reattaches.fetch_add(1, Ordering::SeqCst);
            cascadia_engine::check_reattach_streams(
                cascadia_engine::LinkShape::pipeline(true, true),
                &links,
            )?;
            if self.panic_reattach.load(Ordering::SeqCst) {
                panic!("reattach_streams panicked mid-swap");
            }
            if self.fail_reattach.load(Ordering::SeqCst) {
                return Err(EngineError::Backend("reset failed".into()));
            }
            self.dead.store(false, Ordering::SeqCst);
            Ok(())
        }
    }

    struct LinkHarness {
        runner: Arc<Runner>,
        dead: Arc<Flag>,
        fail_reattach: Arc<Flag>,
        panic_reattach: Arc<Flag>,
        steps: Arc<std::sync::atomic::AtomicUsize>,
        submits: Arc<std::sync::atomic::AtomicUsize>,
        reattaches: Arc<std::sync::atomic::AtomicUsize>,
        stepped: std::sync::mpsc::Receiver<()>,
    }

    impl LinkHarness {
        /// The engine steps within `STEP_WAIT`.
        fn expect_step(&self, what: &str) {
            self.stepped
                .recv_timeout(STEP_WAIT)
                .unwrap_or_else(|_| panic!("timed out: {what}"));
        }

        /// The engine does not step within `NO_STEP_WINDOW`. Drains a step
        /// signalled earlier first, so only steps from now on count.
        fn expect_no_step(&self, what: &str) {
            let _ = self.stepped.try_recv();
            assert!(self.stepped.recv_timeout(NO_STEP_WINDOW).is_err(), "{what}");
        }
    }

    const STEP_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
    /// How long "no step" is watched for. A regression (a relay that keeps
    /// stepping) signals within microseconds, so this only bounds the cost.
    const NO_STEP_WINDOW: std::time::Duration = std::time::Duration::from_millis(150);

    fn link_runner(stream_mode: bool) -> LinkHarness {
        let dead = Arc::new(Flag::new(false));
        let fail_reattach = Arc::new(Flag::new(false));
        let panic_reattach = Arc::new(Flag::new(false));
        let steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let submits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let reattaches = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (stepped_tx, stepped) = std::sync::mpsc::sync_channel(1);
        let engine: Box<dyn Engine> = Box::new(LinkEngine {
            stepped: stepped_tx,
            dead: dead.clone(),
            fail_reattach: fail_reattach.clone(),
            panic_reattach: panic_reattach.clone(),
            steps: steps.clone(),
            submits: submits.clone(),
            reattaches: reattaches.clone(),
        });
        let runner = Runner::for_test(engine, stream_mode);
        LinkHarness {
            runner,
            dead,
            fail_reattach,
            panic_reattach,
            steps,
            submits,
            reattaches,
            stepped,
        }
    }

    fn duplex_end() -> cascadia_engine::ByteStream {
        Box::new(tokio::io::duplex(64).0)
    }

    fn up_link() -> cascadia_engine::StreamLinks {
        cascadia_engine::StreamLinks::pipeline(Some(duplex_end()), None)
    }

    fn wait_until(what: &str, f: impl Fn() -> bool) {
        let start = std::time::Instant::now();
        while !f() {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(5),
                "timed out: {what}"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn task(id: &str) -> GenerationTask {
        GenerationTask::new(id, "x")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stream_mode_relay_parks_on_dead_link_and_resumes_after_reattach() {
        let h = link_runner(true);
        h.dead.store(true, Ordering::SeqCst);
        let driver = h.runner.clone();
        let loop_handle = std::thread::spawn(move || driver.run_relay_loop());
        h.expect_step("first fatal step");
        h.expect_no_step("a parked relay must not keep stepping");
        assert!(
            !loop_handle.is_finished(),
            "stream-mode relay must park, not exit"
        );
        assert_eq!(h.steps.load(Ordering::SeqCst), 1);

        h.runner.reattach(up_link()).await.unwrap();
        h.expect_step("resumed stepping");

        h.runner.close();
        assert_eq!(loop_handle.join().unwrap(), RelayExit::SlotEmpty);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn close_wakes_a_parked_relay() {
        let h = link_runner(true);
        h.dead.store(true, Ordering::SeqCst);
        let driver = h.runner.clone();
        let loop_handle = std::thread::spawn(move || driver.run_relay_loop());
        wait_until("first fatal step", || h.steps.load(Ordering::SeqCst) >= 1);
        h.runner.close();
        wait_until("relay exit", || loop_handle.is_finished());
        assert_eq!(loop_handle.join().unwrap(), RelayExit::SlotEmpty);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn wait_for_link_change_returns_at_once_for_a_stale_snapshot() {
        // What makes a re-attach landing between the relay's fatal step and
        // its park not lost: the relay waits on the generation snapshot it
        // took BEFORE stepping, and a wait on a stale snapshot is a no-op.
        let h = link_runner(true);
        h.dead.store(true, Ordering::SeqCst);
        let snap = h.runner.link.state.lock().generation;
        h.runner.reattach(up_link()).await.unwrap();
        // The relay's wait with the pre-reattach snapshot must return at once.
        let r = h.runner.clone();
        let waiter = std::thread::spawn(move || r.wait_for_link_change(snap));
        wait_until("wait returns", || waiter.is_finished());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_rejected_in_tcp_mode_and_for_bad_shape() {
        let h = link_runner(false);
        assert!(matches!(
            h.runner.reattach(up_link()).await,
            Err(EngineError::PeerRejected(_))
        ));
        // TCP-mode rejection happens before the slot lock: nothing moves.
        assert_eq!(h.runner.link.state.lock().generation, 0);
        assert!(!h.runner.is_fenced());
        let h = link_runner(true);
        let gen_before = h.runner.link.state.lock().generation;
        assert!(matches!(
            h.runner
                .reattach(cascadia_engine::StreamLinks::default())
                .await,
            Err(EngineError::PeerRejected(_))
        ));
        // A shape rejection leaves the fence where it was; the generation
        // moves (the flag was transiently up, so parked relays re-check).
        assert!(h.runner.link.state.lock().generation > gen_before);
        assert!(!h.runner.is_fenced());
        // Nothing was swapped, so in-flight requests are not orphaned.
        assert_eq!(h.runner.slot.attach_epoch(), 0);
    }

    /// Healthy relay engine whose `reattach_streams` announces itself, blocks
    /// until released, then rejects (as the dist_spec partial-link rule does).
    struct GatedRejectEngine {
        steps: Arc<std::sync::atomic::AtomicUsize>,
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
    }

    impl Engine for GatedRejectEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.steps.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(vec![])
        }
        fn reattach_streams(
            &mut self,
            _links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            Err(EngineError::PeerRejected("partial re-attach".into()))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejected_reattach_wakes_a_relay_that_saw_the_transient_fence() {
        // The relay's unlocked pre-check reads the flag while a re-attach
        // holds it transiently up; the re-attach is then rejected. The relay
        // parked on that read must be woken, not stranded on a live engine.
        let steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let engine: Box<dyn Engine> = Box::new(GatedRejectEngine {
            steps: steps.clone(),
            entered: entered_tx,
            release: release_rx,
        });
        let runner = Runner::for_test(engine, true);

        // Hook: once armed, the next pre-check reports in, waits until the
        // re-attach is inside `reattach_streams` (flag up, slot lock held),
        // then reports the flag value it read.
        let armed = Arc::new(Flag::new(false));
        let in_hook = Arc::new(Flag::new(false));
        let (hit_tx, hit_rx) = std::sync::mpsc::channel::<()>();
        let (read_tx, read_rx) = std::sync::mpsc::channel::<bool>();
        let entered_rx = Mutex::new(entered_rx);
        let hook: PrecheckHook = {
            let (armed, in_hook) = (armed.clone(), in_hook.clone());
            Arc::new(move |fenced| match fenced {
                None => {
                    if armed.swap(false, Ordering::SeqCst) {
                        in_hook.store(true, Ordering::SeqCst);
                        hit_tx.send(()).unwrap();
                        entered_rx.lock().recv().unwrap();
                    }
                }
                Some(f) => {
                    if in_hook.swap(false, Ordering::SeqCst) {
                        read_tx.send(f).unwrap();
                    }
                }
            })
        };
        *runner.link.precheck_hook.lock() = Some(hook);

        let driver = runner.clone();
        let loop_handle = std::thread::spawn(move || driver.run_relay_loop());
        wait_until("relay steps", || steps.load(Ordering::SeqCst) >= 1);

        armed.store(true, Ordering::SeqCst);
        hit_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("relay reaches the pre-check");
        // Relay sits in the pre-check without the slot lock: the re-attach
        // takes it, raises the flag and blocks inside `reattach_streams`.
        let r = runner.clone();
        let reattach = tokio::spawn(async move { r.reattach(up_link()).await });
        let saw = read_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("relay reads the flag");
        assert!(saw, "relay must observe the transient fence");
        // The relay has acted on `true` (it parks); now reject the re-attach.
        release_tx.send(()).unwrap();
        assert!(matches!(
            reattach.await.unwrap(),
            Err(EngineError::PeerRejected(_))
        ));
        assert!(
            !runner.is_fenced(),
            "a rejected re-attach restores the fence"
        );

        let after = steps.load(Ordering::SeqCst);
        wait_until(
            "relay resumes stepping after the rejected re-attach",
            || steps.load(Ordering::SeqCst) > after,
        );

        runner.close();
        assert_eq!(loop_handle.join().unwrap(), RelayExit::SlotEmpty);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_after_close_is_not_loaded() {
        let h = link_runner(true);
        h.runner.close();
        assert!(matches!(
            h.runner.reattach(up_link()).await,
            Err(EngineError::NotLoaded)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_reattach_fences_until_a_successful_one() {
        let h = link_runner(true);
        h.fail_reattach.store(true, Ordering::SeqCst);
        assert!(matches!(
            h.runner.reattach(up_link()).await,
            Err(EngineError::Backend(_))
        ));
        // Fenced: submit refused without reaching the engine.
        let r = h.runner.clone();
        let res = tokio::task::spawn_blocking(move || r.submit(task("t1")))
            .await
            .unwrap();
        assert!(matches!(res, Err(EngineError::NotConnected)), "got {res:?}");
        assert_eq!(h.submits.load(Ordering::SeqCst), 0);
        // Fenced: relay loop does not step.
        let driver = h.runner.clone();
        let loop_handle = std::thread::spawn(move || driver.run_relay_loop());
        h.expect_no_step("fenced relay must not step");
        assert_eq!(h.steps.load(Ordering::SeqCst), 0);
        // A successful reattach clears the fence and wakes the relay.
        h.fail_reattach.store(false, Ordering::SeqCst);
        h.runner.reattach(up_link()).await.unwrap();
        h.expect_step("relay steps after unfence");
        let r = h.runner.clone();
        let res = tokio::task::spawn_blocking(move || r.submit(task("t2")))
            .await
            .unwrap();
        assert!(res.is_ok());
        h.runner.close();
        assert_eq!(loop_handle.join().unwrap(), RelayExit::SlotEmpty);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn panicking_reattach_leaves_the_engine_fenced() {
        // A panic inside `reattach_streams` may have swapped some streams and
        // not others: the engine state is unspecified, so it must stay fenced.
        let h = link_runner(true);
        h.panic_reattach.store(true, Ordering::SeqCst);
        let res = h.runner.reattach(up_link()).await;
        assert!(matches!(res, Err(EngineError::Backend(_))), "got {res:?}");
        assert!(
            h.runner.slot.is_fenced(),
            "a panicked re-attach must leave the fence up"
        );
        assert_eq!(
            h.runner.slot.attach_epoch(),
            1,
            "a panicked re-attach may have swapped: older streams are orphaned"
        );
        let r = h.runner.clone();
        let res = tokio::task::spawn_blocking(move || r.submit(task("t1")))
            .await
            .unwrap();
        assert!(matches!(res, Err(EngineError::NotConnected)), "got {res:?}");
        assert_eq!(h.submits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_reattach_fences_open_generate_streams() {
        // A stream opened before a failed post-swap re-attach must not step
        // the half-reset engine: its next poll ends it with NotConnected.
        let h = link_runner(true);
        let mut stream = h.runner.generate_async(task("open")).await.unwrap();
        h.fail_reattach.store(true, Ordering::SeqCst);
        assert!(matches!(
            h.runner.reattach(up_link()).await,
            Err(EngineError::Backend(_))
        ));
        let steps_before = h.steps.load(Ordering::SeqCst);
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
            .await
            .expect("poll must not hang")
            .expect("fenced stream yields a final error chunk");
        let err = chunk.error.as_deref().expect("error chunk");
        assert!(
            err.contains(&EngineError::NotConnected.to_string()),
            "got {err:?}"
        );
        assert!(chunk.is_final);
        assert_eq!(
            h.steps.load(Ordering::SeqCst),
            steps_before,
            "a fenced stream must not step the engine"
        );
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn tcp_mode_relay_still_exits_connection_fatal() {
        let h = link_runner(false);
        h.dead.store(true, Ordering::SeqCst);
        let driver = h.runner.clone();
        let loop_handle = std::thread::spawn(move || driver.run_relay_loop());
        wait_until("relay exit", || loop_handle.is_finished());
        assert_eq!(loop_handle.join().unwrap(), RelayExit::ConnectionFatal);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_fails_in_flight_streams_with_a_clear_error() {
        // A request in flight on the head when its links are swapped can
        // never finish (the engine reset its session state): the stream must
        // end at once with a NotConnected error naming the re-attach, not
        // limp on until the "no progress" guard trips.
        let h = link_runner(true);
        let mut stream = h.runner.generate_async(task("inflight")).await.unwrap();
        h.runner.reattach(up_link()).await.unwrap();
        let steps_before = h.steps.load(Ordering::SeqCst);
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
            .await
            .expect("poll must not hang")
            .expect("orphaned stream yields a final error chunk");
        let err = chunk.error.as_deref().expect("error chunk");
        assert!(
            err.contains("link re-attached; in-flight request dropped"),
            "got {err:?}"
        );
        assert!(
            err.contains(&EngineError::NotConnected.to_string()),
            "got {err:?}"
        );
        assert!(chunk.is_final);
        assert_eq!(
            h.steps.load(Ordering::SeqCst),
            steps_before,
            "an orphaned stream must not step the engine"
        );
        assert!(stream.next().await.is_none());
        // A request submitted after the re-attach is unaffected.
        let mut fresh = h.runner.generate_async(task("fresh")).await.unwrap();
        let chunk = fresh.next().await.expect("chunk");
        assert!(
            !chunk
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("re-attached"),
            "got {chunk:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_dropped_before_the_slot_lock_does_not_swap() {
        // Dropping the `reattach` future while its blocking half still waits
        // for the slot lock must cancel it: nothing swapped, no fence moved,
        // no generation or epoch bump.
        let h = link_runner(true);
        let baseline = Arc::strong_count(&h.runner.slot);
        let gen_before = h.runner.link.state.lock().generation;
        let guard = h.runner.slot.lock();
        let mut fut = Box::pin(h.runner.reattach(up_link()));
        assert!(futures::poll!(&mut fut).is_pending());
        drop(fut);
        drop(guard);
        // The detached blocking half holds a clone of the slot until it has
        // run: wait for it to finish.
        wait_until("blocking half finished", || {
            Arc::strong_count(&h.runner.slot) == baseline
        });
        assert_eq!(h.runner.slot.attach_epoch(), 0);
        assert_eq!(
            h.reattaches.load(Ordering::SeqCst),
            0,
            "a cancelled re-attach must not reach the engine"
        );
        assert!(!h.runner.is_fenced());
        assert_eq!(h.runner.link.state.lock().generation, gen_before);
    }

    /// Builder whose `build()` records whether the runner was already in
    /// stream mode, i.e. before the engine lands in the slot.
    struct ModeProbeBuilder {
        runner: Arc<parking_lot::Mutex<std::sync::Weak<Runner>>>,
        seen: Arc<parking_lot::Mutex<Option<bool>>>,
    }

    #[async_trait::async_trait]
    impl Builder for ModeProbeBuilder {
        async fn connect(&mut self, _peers: PeerLayout) -> Result<(), EngineError> {
            Ok(())
        }
        async fn connect_streams(
            &mut self,
            _links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            Ok(())
        }
        async fn load(
            &mut self,
            _shard: ShardSpec,
        ) -> Result<cascadia_engine::LoadStream, EngineError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn build(self: Box<Self>) -> Result<Box<dyn Engine>, EngineError> {
            let runner = self.runner.lock().upgrade().expect("runner alive");
            *self.seen.lock() = Some(runner.stream_mode.load(Ordering::SeqCst));
            Ok(Box::new(FailingEngine))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn start_with_streams_sets_stream_mode_before_the_engine_is_live() {
        let weak = Arc::new(parking_lot::Mutex::new(std::sync::Weak::new()));
        let seen = Arc::new(parking_lot::Mutex::new(None));
        let runner = Arc::new(Runner::new(Box::new(ModeProbeBuilder {
            runner: weak.clone(),
            seen: seen.clone(),
        })));
        *weak.lock() = Arc::downgrade(&runner);
        runner
            .start_with_streams(
                cascadia_engine::StreamLinks::pipeline(Some(duplex_end()), None),
                test_shard(),
            )
            .await
            .unwrap();
        assert_eq!(
            *seen.lock(),
            Some(true),
            "stream mode must be on before the engine is placed in the slot"
        );
    }

    #[test]
    fn link_down_warns_once_per_attach_epoch() {
        let mut log = LinkDownLog::default();
        assert!(log.first_for(0), "first death warns");
        assert!(!log.first_for(0), "a re-wake on the same links is quiet");
        assert!(log.first_for(1), "a death after a swap warns again");
        assert!(!log.first_for(1));
    }

    /// Serializes the tests that pin `cascadia_link_reattach_total` deltas
    /// for the non-upstream sides.
    static REATTACH_METRIC_TEST: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Accepts any non-empty set of links (no stage shape), for counting.
    struct AnyLinkEngine;

    impl Engine for AnyLinkEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            Ok(vec![])
        }
        fn reattach_streams(
            &mut self,
            links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            if links.shape() == cascadia_engine::LinkShape::default() {
                return Err(EngineError::PeerRejected("nothing to replace".into()));
            }
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_counts_each_replaced_link_per_side() {
        // `upstream` is left out: other tests in this binary re-attach
        // upstream links concurrently, so its delta is not ours to pin.
        let _metric = REATTACH_METRIC_TEST.lock().await;
        let count = |side: &str| {
            cascadia_metrics::LINK_REATTACH_TOTAL
                .with_label_values(&[side])
                .get()
        };
        let sides = ["downstream", "ep_driver", "ep_worker"];
        let before: Vec<u64> = sides.iter().map(|s| count(s)).collect();
        let runner = Runner::for_test(Box::new(AnyLinkEngine), true);

        // Two of three workers replaced (`None` keeps one), plus downstream.
        let mut links = cascadia_engine::StreamLinks::ep_workers(vec![
            Some(duplex_end()),
            None,
            Some(duplex_end()),
        ]);
        links.downstream = Some(duplex_end());
        runner.reattach(links).await.unwrap();
        runner
            .reattach(cascadia_engine::StreamLinks::ep_driver(duplex_end()))
            .await
            .unwrap();
        // A rejected re-attach replaced nothing and counts nothing.
        assert!(matches!(
            runner
                .reattach(cascadia_engine::StreamLinks::default())
                .await,
            Err(EngineError::PeerRejected(_))
        ));

        let delta: Vec<u64> = sides
            .iter()
            .zip(&before)
            .map(|(s, b)| count(s) - b)
            .collect();
        assert_eq!(delta, vec![1, 1, 2], "per-side deltas for {sides:?}");
    }

    /// One recorded event: its message and the name of the span it fired in.
    type RecordedEvent = (String, Option<String>);

    /// Records each event's message and the name of the span it fired in.
    #[derive(Clone, Default)]
    struct SpanRecorder(Arc<Mutex<Vec<RecordedEvent>>>);

    impl<S> tracing_subscriber::Layer<S> for SpanRecorder
    where
        S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            struct Message(String);
            impl tracing::field::Visit for Message {
                fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
                    if f.name() == "message" {
                        self.0 = format!("{v:?}");
                    }
                }
            }
            let mut msg = Message(String::new());
            event.record(&mut msg);
            let span = ctx.event_span(event).map(|s| s.name().to_string());
            self.0.lock().push((msg.0, span));
        }
    }

    #[test]
    fn reattach_logs_stay_in_the_callers_span_and_subscriber() {
        use tracing::Instrument;
        use tracing_subscriber::layer::SubscriberExt;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        // `upstream`: the side the per-side metric tests leave unpinned.
        let upstream = || cascadia_engine::StreamLinks::pipeline(Some(duplex_end()), None);
        let runner = Runner::for_test(Box::new(AnyLinkEngine), true);
        // Register the re-attach log callsites on this thread first. Other
        // tests in this binary hit them with no subscriber installed; a
        // registration racing ours could otherwise cache "never interested"
        // after the rebuild below and hide the event from the scoped
        // subscriber.
        rt.block_on(runner.reattach(upstream())).unwrap();

        let recorder = SpanRecorder::default();
        let subscriber = tracing_subscriber::registry().with(recorder.clone());
        // A scoped (not global) subscriber: the blocking half of `reattach`
        // runs on a pool thread, which only sees it if it is carried over.
        tracing::subscriber::with_default(subscriber, || {
            tracing::callsite::rebuild_interest_cache();
            rt.block_on(
                runner
                    .reattach(upstream())
                    .instrument(tracing::info_span!("embedder_reattach")),
            )
            .unwrap();
        });
        let events = recorder.0.lock();
        let reattached = events
            .iter()
            .find(|(msg, _)| msg == "link re-attached")
            .unwrap_or_else(|| {
                panic!("the re-attach log reached the caller's subscriber: {events:?}")
            });
        assert_eq!(reattached.1.as_deref(), Some("embedder_reattach"));
    }

    /// FIFO engine: `submit` queues, each `step` finishes the oldest queued
    /// task with one final chunk. Like the real engines, `reattach_streams`
    /// leaves the queue alone; `cancel` removes a task and is logged.
    #[derive(Default)]
    struct QueueEngine {
        queue: VecDeque<TaskId>,
        cancelled: Arc<Mutex<Vec<TaskId>>>,
    }

    impl Engine for QueueEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, task: GenerationTask) -> Result<(), EngineError> {
            self.queue.push_back(task.task_id);
            Ok(())
        }
        fn cancel(&mut self, task_id: &TaskId) {
            self.queue.retain(|t| t != task_id);
            self.cancelled.lock().push(task_id.clone());
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            Ok(self
                .queue
                .pop_front()
                .map(|tid| {
                    let mut chunk = Chunk::token(tid.clone(), 1, "tok");
                    chunk.is_final = true;
                    (tid, chunk)
                })
                .into_iter()
                .collect())
        }
        fn reattach_streams(
            &mut self,
            _links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_drops_queued_tasks_before_they_emit() {
        // A task still queued (submitted, never stepped) when the links are
        // swapped must be cancelled in the engine and its stream must fail
        // with the re-attach error -- not be admitted later by a sibling's
        // step and finish (or fail mid-output) on the new links.
        let cancelled = Arc::new(Mutex::new(Vec::new()));
        let engine = QueueEngine {
            cancelled: cancelled.clone(),
            ..Default::default()
        };
        let runner = Runner::for_test(Box::new(engine), true);
        let mut queued = runner.generate_async(task("queued")).await.unwrap();
        runner.reattach(up_link()).await.unwrap();
        assert_eq!(
            *cancelled.lock(),
            vec![TaskId::from("queued")],
            "the re-attach cancels the stale task in the engine"
        );
        // A request submitted after the re-attach steps the engine first.
        let mut fresh = runner.generate_async(task("fresh")).await.unwrap();
        let chunk = fresh.next().await.expect("fresh chunk");
        assert_eq!(chunk.task_id, TaskId::from("fresh"), "got {chunk:?}");
        assert!(chunk.error.is_none(), "got {chunk:?}");
        // The stale stream's first chunk is the re-attach error.
        let chunk = queued.next().await.expect("queued chunk");
        let err = chunk
            .error
            .as_deref()
            .unwrap_or_else(|| panic!("got {chunk:?}"));
        assert!(
            err.contains("link re-attached; in-flight request dropped"),
            "got {err:?}"
        );
        assert!(queued.next().await.is_none());
    }

    /// First step fails naming the `stale` task; later steps finish `fresh`.
    #[derive(Default)]
    struct ForeignFailEngine {
        steps: usize,
    }

    impl Engine for ForeignFailEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            self.steps += 1;
            if self.steps == 1 {
                return Err(EngineError::Backend("boom".into()).for_task("stale".into()));
            }
            Ok(vec![("fresh".into(), Chunk::final_marker("fresh", "ok"))])
        }
        fn reattach_streams(
            &mut self,
            _links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_orphaned_stream_ends_with_the_reattach_error_not_a_routed_engine_error() {
        // A sibling's step fails naming the orphaned task: that failure must
        // not be routed into the orphaned stream, which ends with the
        // re-attach error regardless.
        let runner = Runner::for_test(Box::new(ForeignFailEngine::default()), true);
        let mut stale = runner.generate_async(task("stale")).await.unwrap();
        runner.reattach(up_link()).await.unwrap();
        let mut fresh = runner.generate_async(task("fresh")).await.unwrap();
        let chunk = fresh.next().await.expect("fresh chunk");
        assert_eq!(chunk.text, "ok", "got {chunk:?}");
        let chunk = stale.next().await.expect("stale chunk");
        let err = chunk
            .error
            .as_deref()
            .unwrap_or_else(|| panic!("got {chunk:?}"));
        assert!(
            err.contains("link re-attached; in-flight request dropped"),
            "got {err:?}"
        );
        assert!(stale.next().await.is_none());
    }

    /// `reattach_streams` signals `entered`, then blocks until `release`.
    struct GatedReattachEngine {
        entered: std::sync::mpsc::SyncSender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl Engine for GatedReattachEngine {
        fn warmup(&mut self) {}
        fn submit(&mut self, _task: GenerationTask) -> Result<(), EngineError> {
            Ok(())
        }
        fn step(&mut self) -> Result<Vec<(TaskId, Chunk)>, EngineError> {
            Ok(vec![])
        }
        fn reattach_streams(
            &mut self,
            _links: cascadia_engine::StreamLinks,
        ) -> Result<(), EngineError> {
            self.entered.send(()).unwrap();
            self.release.lock().recv().unwrap();
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reattach_dropped_after_the_slot_lock_still_counts_and_logs() {
        // Once the blocking half holds the lock the re-attach completes even
        // if the caller drops the future -- and must still be counted (and
        // logged, from the same place).
        let _metric = REATTACH_METRIC_TEST.lock().await;
        let count = || {
            cascadia_metrics::LINK_REATTACH_TOTAL
                .with_label_values(&["ep_driver"])
                .get()
        };
        let before = count();
        let (entered_tx, entered) = std::sync::mpsc::sync_channel(1);
        let (release, release_rx) = std::sync::mpsc::sync_channel(1);
        let runner = Runner::for_test(
            Box::new(GatedReattachEngine {
                entered: entered_tx,
                release: Mutex::new(release_rx),
            }),
            true,
        );
        let baseline = Arc::strong_count(&runner.slot);
        let mut fut =
            Box::pin(runner.reattach(cascadia_engine::StreamLinks::ep_driver(duplex_end())));
        assert!(futures::poll!(&mut fut).is_pending());
        entered
            .recv_timeout(STEP_WAIT)
            .expect("blocking half reached the engine");
        drop(fut);
        release.send(()).unwrap();
        wait_until("blocking half finished", || {
            Arc::strong_count(&runner.slot) == baseline
        });
        assert_eq!(runner.slot.attach_epoch(), 1, "the swap completed");
        assert_eq!(
            count() - before,
            1,
            "a dropped but completed re-attach is counted"
        );
    }

    #[test]
    fn close_lock_notes_a_long_wait_once_and_keeps_waiting() {
        let slot = Arc::new(EngineSlot::new(None));
        let noted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Free lock: no note.
        drop(
            slot.lock_noting_delay(std::time::Duration::from_millis(20), || {
                noted.fetch_add(1, Ordering::SeqCst);
            }),
        );
        assert_eq!(noted.load(Ordering::SeqCst), 0);
        // Held past the delay: noted once, then still acquired on release.
        let held = slot.lock();
        let acquired = Arc::new(Flag::new(false));
        let waiter = {
            let (slot, noted, acquired) = (slot.clone(), noted.clone(), acquired.clone());
            std::thread::spawn(move || {
                let _g = slot.lock_noting_delay(std::time::Duration::from_millis(20), || {
                    noted.fetch_add(1, Ordering::SeqCst);
                });
                acquired.store(true, Ordering::SeqCst);
            })
        };
        wait_until("long wait noted", || noted.load(Ordering::SeqCst) == 1);
        assert!(
            !acquired.load(Ordering::SeqCst),
            "still waiting after the note"
        );
        drop(held);
        waiter.join().unwrap();
        assert!(acquired.load(Ordering::SeqCst));
        assert_eq!(noted.load(Ordering::SeqCst), 1, "noted exactly once");
    }
}
