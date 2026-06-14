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
use std::sync::Arc;
use std::task::{Context, Poll};

use cascadia_engine::{Builder, Engine, EngineError};
use cascadia_types::{Chunk, GenerationTask, PeerLayout, ShardSpec, TaskId};
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
}

impl Runner {
    pub fn new(builder: Box<dyn Builder>) -> Self {
        Self {
            builder: Mutex::new(Some(builder)),
            engine: Arc::new(Mutex::new(None)),
            buffers: Arc::new(Mutex::new(Buffers::default())),
        }
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
        let mut load_stream = builder.load(shard).await?;
        // drain the load progress stream
        use futures::StreamExt;
        while let Some(progress) = load_stream.next().await {
            info!(message = %progress.message, "load progress");
        }

        info!("runner build engine");
        let mut engine = builder.build()?;
        info!("runner warmup");
        engine.warmup();
        info!("runner ready");

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
        let mut guard = self.engine.lock();
        let engine = guard.as_mut().ok_or(EngineError::NotLoaded)?;
        engine.submit(task)
    }

    /// Cooperatively cancel a task.
    pub fn cancel(&self, task_id: &TaskId) {
        let mut bufs = self.buffers.lock();
        bufs.cancelled.insert(task_id.clone());
        bufs.chunks.remove(task_id);
        if let Some(engine) = self.engine.lock().as_mut() {
            engine.cancel(task_id);
        }
    }

    /// Submit a task and return a stream of chunks. Stops on the final
    /// chunk, on cancellation, on an engine step error, or after
    /// MAX_CONSECUTIVE_EMPTY_STEPS empty engine polls (engine appears
    /// stuck).
    pub fn generate(&self, task: GenerationTask) -> Result<ChunkStream, EngineError> {
        self.submit(task.clone())?;
        Ok(ChunkStream {
            task_id: task.task_id,
            engine: self.engine.clone(),
            buffers: self.buffers.clone(),
            consecutive_empty: 0,
            done: false,
        })
    }

    /// Step the engine forever; exits only when the engine slot empties.
    /// Step errors are logged and driving continues (engines own their
    /// recovery). Used by non-first pipeline stages.
    ///
    /// The loop throttles itself: a `step()` that returns `Err` sleeps
    /// `RELAY_ERR_BACKOFF` before the next round so a persistently-failing
    /// engine (dead peer, bad-frame flood, black-holed upstream) can't peg
    /// a core regardless of whether the engine self-throttles. The backoff
    /// resets on any non-empty `Ok` step. Engines MAY additionally
    /// self-throttle (e.g. the workers sleep on a dead peer too); the two
    /// are belt-and-suspenders.
    ///
    /// Enters a `BlockingContextGuard` once per OS thread (since this
    /// loop runs on a single `spawn_blocking` thread) so that engines'
    /// `run_async` calls hit the naked-`block_on` path instead of
    /// `block_in_place` — ~60 ms/frame savings on Windows.
    pub fn run_relay_loop(&self) {
        let _blocking = BlockingContextGuard::enter();
        loop {
            let mut guard = self.engine.lock();
            let Some(engine) = guard.as_mut() else { break };
            // Engine.step is sync; just drain. An Err is logged but the
            // loop keeps driving — relay engines recover their own state
            // (and self-throttle on dead peers), and a transient frame
            // error must not take the whole stage down.
            let failed = match engine.step() {
                Ok(_) => false,
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
        info!("relay loop exited");
    }

    pub fn close(&self) {
        if let Some(engine) = self.engine.lock().as_mut() {
            engine.close();
        }
        *self.engine.lock() = None;
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
    done: bool,
}

impl Stream for ChunkStream {
    type Item = Chunk;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Chunk>> {
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
                    return Poll::Ready(None);
                }
                if let Some(buf) = bufs.chunks.get_mut(&this.task_id) {
                    if let Some(chunk) = buf.pop_front() {
                        let is_final = chunk.is_final;
                        if is_final {
                            this.done = true;
                            bufs.chunks.remove(&this.task_id);
                        }
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
            let step_result = {
                let mut guard = this.engine.lock();
                let Some(engine) = guard.as_mut() else {
                    this.done = true;
                    return Poll::Ready(None);
                };
                engine.step()
            };
            let produced = match step_result {
                Ok(v) => v,
                Err(e) => match e.task_id() {
                    Some(failed) if failed != &this.task_id => {
                        // Misattribution guard: fail the named task's stream,
                        // not ours. Cancelling it makes its own poll_next
                        // observe termination on its next turn.
                        let failed = failed.clone();
                        warn!(
                            failed_task = %failed,
                            polled_task = %this.task_id,
                            error = %e,
                            "engine step failed for another task; routing failure to it"
                        );
                        let mut bufs = this.buffers.lock();
                        bufs.cancelled.insert(failed.clone());
                        bufs.chunks.remove(&failed);
                        continue;
                    }
                    _ => {
                        this.done = true;
                        warn!(
                            task = %this.task_id,
                            error = %e,
                            "engine step failed; closing stream"
                        );
                        return Poll::Ready(None);
                    }
                },
            };

            if produced.is_empty() {
                this.consecutive_empty += 1;
                if this.consecutive_empty >= MAX_CONSECUTIVE_EMPTY_STEPS {
                    this.done = true;
                    warn!(
                        task = %this.task_id,
                        "engine made no progress for {} consecutive steps; closing stream",
                        MAX_CONSECUTIVE_EMPTY_STEPS
                    );
                    return Poll::Ready(None);
                }
                continue;
            }
            this.consecutive_empty = 0;

            // 3) Distribute produced chunks: ours go to caller; others to
            //    their owners' buffers. We may then loop to attempt the
            //    cancelled-check or buffered-drain again.
            let mut bufs = this.buffers.lock();
            let mut ours: Option<Chunk> = None;
            for (tid, chunk) in produced {
                if bufs.cancelled.contains(&tid) {
                    continue;
                }
                if tid == this.task_id && ours.is_none() {
                    ours = Some(chunk);
                } else {
                    bufs.chunks.entry(tid).or_default().push_back(chunk);
                }
            }
            drop(bufs);
            if let Some(c) = ours {
                let is_final = c.is_final;
                if is_final {
                    this.done = true;
                    let mut bufs = this.buffers.lock();
                    bufs.chunks.remove(&this.task_id);
                }
                return Poll::Ready(Some(c));
            }
            // else loop: nothing for us this round.
        }
    }
}

impl Drop for ChunkStream {
    fn drop(&mut self) {
        // Tell the engine to abandon this task. Without this an SSE
        // client that disconnects mid-generation leaves the engine
        // grinding through max_tokens worth of chunks that no one
        // will ever drain — the chunk buffer for this task accretes
        // until close() and the engine slot stays busy.
        if let Some(engine) = self.engine.lock().as_mut() {
            engine.cancel(&self.task_id);
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
        });

        // Drive the relay loop on a worker thread, let it run for a window
        // several backoffs wide, then empty the engine slot to stop it.
        let driver = runner.clone();
        let handle = std::thread::spawn(move || driver.run_relay_loop());
        std::thread::sleep(RELAY_ERR_BACKOFF * 5);
        runner.close(); // engine slot -> None: loop breaks next round
        handle.join().unwrap();

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

    #[tokio::test]
    async fn step_error_terminates_stream() {
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
        // The first poll hits the step error and ends the stream — no
        // empty-step spinning, no chunks.
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

        // A's stream ends (None) — it was the failed task.
        assert!(
            stream_a.next().await.is_none(),
            "failed task a's stream should have ended"
        );
        // B's stream is now complete too.
        assert!(stream_b.next().await.is_none());
    }
}
