//! Per-stage Runner.
//!
//! Mirrors `tahoma/worker/runner.py`. Lifecycle:
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

use futures::Stream;
use parking_lot::Mutex;
use tahoma_engine::{Builder, Engine, EngineError};
use tahoma_types::{Chunk, GenerationTask, PeerLayout, ShardSpec, TaskId};
use tracing::{info, warn};

/// When `generate()` sees this many consecutive empty steps with no new
/// chunks for *any* task, it returns rather than block forever on a
/// misbehaving engine.
const MAX_CONSECUTIVE_EMPTY_STEPS: usize = 3;

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
        let mut builder = self
            .builder
            .lock()
            .take()
            .ok_or(EngineError::NotLoaded)?;
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
        engine.submit(task);
        Ok(())
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
    /// chunk, on cancellation, or after MAX_CONSECUTIVE_EMPTY_STEPS empty
    /// engine polls (engine appears stuck).
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

    /// Step the engine forever; exits when the engine signals io error
    /// (transport closed). Used by non-first pipeline stages.
    pub fn run_relay_loop(&self) {
        loop {
            let mut guard = self.engine.lock();
            let Some(engine) = guard.as_mut() else { break };
            // Engine.step is sync; just drain.
            let _produced = engine.step();
            // Don't hold the lock for long under the loop — yield to
            // other generate() callers between rounds.
            drop(guard);
            std::thread::yield_now();
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

            // 2) Drive the engine one step.
            let produced = {
                let mut guard = this.engine.lock();
                let Some(engine) = guard.as_mut() else {
                    this.done = true;
                    return Poll::Ready(None);
                };
                engine.step()
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
        let mut bufs = self.buffers.lock();
        bufs.chunks.remove(&self.task_id);
        bufs.cancelled.remove(&self.task_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tahoma_engine_mock::MockBuilder;

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
}
