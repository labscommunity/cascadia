//! Issue #76: run a two-stage pipeline over injected byte streams, kill the
//! link between the stages, re-attach fresh streams and keep generating.
//!
//! ```text
//! cargo run -p cascadia-runner --example injected_duplex
//! ```
//!
//! The "network" is a pair of `tokio::io::duplex` pipes joined by a pump task
//! that stands in for the embedder's own transport (an encrypted p2p stream, a
//! QUIC stream, ...). The engines are a toy defined below: the head sends its
//! current value downstream and the tail answers with `value + 1`, so prompt
//! `"40"` with `max_tokens = 3` generates `41 42 43`.
//!
//! The flow is the one a real embedder follows:
//!
//! 1. `Runner::start_with_streams` on both stages (stream mode), then run the
//!    tail's relay loop on a blocking thread.
//! 2. Generate.
//! 3. The link dies. The embedder closes its own ends of the dead streams
//!    first (here: aborting the pump drops both of its ends). The tail's
//!    blocked `step()` returns a connection-fatal error and its relay loop
//!    parks instead of exiting.
//! 4. `Runner::reattach` with fresh streams on both sides of the hop. The tail
//!    must replace its upstream (relay rule); the head replaces its downstream.
//! 5. Generate again on the same live engines.
//! 6. Close the embedder's ends, then `Runner::close` on both runners. Close
//!    is mandatory in stream mode: it is what wakes a parked relay loop so
//!    runtime shutdown does not hang.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use cascadia_engine::{
    check_connect_streams, check_reattach_streams, Builder, ByteStream, Engine, EngineError,
    EngineResult, LinkShape, LoadStream, StreamLinks,
};
use cascadia_runner::{run_async, RelayExit, Runner};
use cascadia_transport::{
    attach_client, attach_server, ActivationClient, ActivationServer, DType, Tensor,
};
use cascadia_types::{Chunk, GenerationTask, LoadProgress, PeerLayout, ShardSpec, TaskId};
use futures::StreamExt;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// Head (`upstream: false, downstream: true`) or tail (`true, false`).
#[derive(Clone, Copy)]
enum Role {
    Head,
    Tail,
}

impl Role {
    fn shape(self) -> LinkShape {
        match self {
            Role::Head => LinkShape::pipeline(false, true),
            Role::Tail => LinkShape::pipeline(true, false),
        }
    }
}

/// The handles the builder hands to the engine. Shared as
/// `Arc<tokio::sync::Mutex<..>>` because re-attach swaps the stream INSIDE
/// the existing handle (`attach_client` / `attach_server`).
enum Links {
    Head(Arc<Mutex<ActivationClient>>),
    Tail(Arc<Mutex<ActivationServer>>),
}

struct ToyBuilder {
    role: Role,
    links: Option<Links>,
}

#[async_trait]
impl Builder for ToyBuilder {
    async fn connect(&mut self, _peers: PeerLayout) -> EngineResult<()> {
        Err(EngineError::PeerRejected(
            "this toy engine only runs over injected streams".into(),
        ))
    }

    async fn connect_streams(&mut self, mut links: StreamLinks) -> EngineResult<()> {
        // Every link the stage has, and nothing else.
        check_connect_streams(self.role.shape(), &links)?;
        self.links = Some(match self.role {
            Role::Head => {
                let s = links.downstream.take().expect("validated");
                Links::Head(Arc::new(Mutex::new(ActivationClient::from_stream(s))))
            }
            Role::Tail => {
                let s = links.upstream.take().expect("validated");
                Links::Tail(Arc::new(Mutex::new(ActivationServer::from_stream(s))))
            }
        });
        Ok(())
    }

    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        Ok(Box::pin(futures::stream::iter([LoadProgress::ready()])))
    }

    fn build(self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        // `build` runs inside `Runner::start_with_streams`, on the runtime.
        let rt = tokio::runtime::Handle::current();
        Ok(match self.links.ok_or(EngineError::NotConnected)? {
            Links::Head(down) => Box::new(HeadEngine {
                rt,
                down,
                queue: VecDeque::new(),
            }),
            Links::Tail(up) => Box::new(TailEngine { rt, up }),
        })
    }
}

struct Job {
    id: TaskId,
    value: i64,
    left: u32,
}

/// Stage 0: owns the requests. One step = one round trip to the tail.
struct HeadEngine {
    rt: tokio::runtime::Handle,
    down: Arc<Mutex<ActivationClient>>,
    queue: VecDeque<Job>,
}

impl Engine for HeadEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        let value = task.prompt.trim().parse().map_err(|_| {
            EngineError::InvalidConfig(format!("prompt {:?} is not an integer", task.prompt))
        })?;
        self.queue.push_back(Job {
            id: task.task_id,
            value,
            left: task.max_tokens,
        });
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let Some(mut job) = self.queue.pop_front() else {
            return Ok(Vec::new());
        };
        let down = self.down.clone();
        let value = job.value;
        let reply = run_async(&self.rt, async move {
            let mut down = down.lock().await;
            down.send(&encode(value)).await?;
            down.recv().await
        });
        let next = match reply.map(|(t, _)| decode(&t)) {
            Ok(Some(next)) => next,
            // The request cannot finish; the engine stays healthy.
            Ok(None) => return Ok(vec![(job.id.clone(), Chunk::error(job.id, "bad reply"))]),
            Err(e) => {
                let msg = format!("downstream link: {e}");
                return Ok(vec![(job.id.clone(), Chunk::error(job.id, msg))]);
            }
        };
        job.value = next;
        job.left = job.left.saturating_sub(1);
        let mut out = vec![(
            job.id.clone(),
            Chunk::token(job.id.clone(), next, format!("{next} ")),
        )];
        if job.left == 0 {
            out.push((job.id.clone(), Chunk::final_marker(job.id, "")));
        } else {
            self.queue.push_front(job);
        }
        Ok(out)
    }

    fn cancel(&mut self, task_id: &TaskId) {
        self.queue.retain(|j| &j.id != task_id);
    }

    fn reattach_streams(&mut self, mut links: StreamLinks) -> EngineResult<()> {
        // PeerRejected only BEFORE anything is swapped.
        check_reattach_streams(Role::Head.shape(), &links)?;
        let fresh = links
            .downstream
            .take()
            .expect("validated: a head has only a downstream");
        run_async(&self.rt, attach_client(&self.down, fresh));
        // Back to a clean between-requests state. The runner fails the
        // streams of requests that were in flight; their callers resubmit.
        self.queue.clear();
        Ok(())
    }
}

/// Stage 1: a relay with no requests of its own. Its `step()` blocks on the
/// upstream read while the runner holds the engine lock.
struct TailEngine {
    rt: tokio::runtime::Handle,
    up: Arc<Mutex<ActivationServer>>,
}

impl Engine for TailEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, _task: GenerationTask) -> EngineResult<()> {
        Err(EngineError::InvalidConfig(
            "the tail stage takes no requests".into(),
        ))
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let up = self.up.clone();
        run_async(&self.rt, async move {
            let mut up = up.lock().await;
            // A toy treats every recv failure as a dead link. NotConnected
            // is connection-fatal: in stream mode the relay loop parks.
            let (frame, _) = up.recv().await.map_err(|_| EngineError::NotConnected)?;
            let value = decode(&frame).ok_or_else(|| EngineError::Backend("bad frame".into()))?;
            up.send(&encode(value + 1))
                .await
                .map_err(|_| EngineError::NotConnected)?;
            Ok(Vec::new())
        })
    }

    fn reattach_streams(&mut self, mut links: StreamLinks) -> EngineResult<()> {
        // The relay rule: a stage with an upstream must replace it.
        check_reattach_streams(Role::Tail.shape(), &links)?;
        let fresh = links
            .upstream
            .take()
            .expect("validated: a tail has only an upstream");
        run_async(&self.rt, attach_server(&self.up, fresh));
        // The tail holds no session state to reset.
        Ok(())
    }
}

fn encode(value: i64) -> Tensor {
    Tensor::from_2d(DType::I64, 1, 1, value.to_le_bytes().to_vec())
}

fn decode(t: &Tensor) -> Option<i64> {
    let bytes: [u8; 8] = t.data.get(..8)?.try_into().ok()?;
    Some(i64::from_le_bytes(bytes))
}

/// One hop of the embedder's "network": the head's and the tail's streams,
/// plus the pump that carries bytes between them. Aborting the pump drops the
/// embedder's ends, which is how this example closes them.
fn hop() -> (ByteStream, ByteStream, JoinHandle<()>) {
    let (head_end, mut head_side) = tokio::io::duplex(64 * 1024);
    let (tail_end, mut tail_side) = tokio::io::duplex(64 * 1024);
    let pump = tokio::spawn(async move {
        let _ = tokio::io::copy_bidirectional(&mut head_side, &mut tail_side).await;
    });
    (Box::new(head_end), Box::new(tail_end), pump)
}

fn shard(first: bool, last: bool) -> ShardSpec {
    ShardSpec {
        is_first_stage: first,
        is_last_stage: last,
        ..ShardSpec::single_stage("toy", "CPU")
    }
}

async fn generate(head: &Arc<Runner>, id: &str, prompt: &str) -> Result<Vec<i64>, String> {
    let mut task = GenerationTask::new(id, prompt);
    task.max_tokens = 3;
    let mut stream = head.generate_async(task).await.map_err(|e| e.to_string())?;
    let mut tokens = Vec::new();
    while let Some(chunk) = stream.next().await {
        if let Some(err) = chunk.error {
            return Err(err);
        }
        if !chunk.is_final {
            tokens.push(chunk.token_id);
        }
    }
    Ok(tokens)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let head = Arc::new(Runner::new(Box::new(ToyBuilder {
        role: Role::Head,
        links: None,
    })));
    let tail = Arc::new(Runner::new(Box::new(ToyBuilder {
        role: Role::Tail,
        links: None,
    })));

    // 1. Start both stages over the first hop.
    let (head_end, tail_end, pump) = hop();
    let (h, t) = tokio::join!(
        head.start_with_streams(
            StreamLinks::pipeline(None, Some(head_end)),
            shard(true, false)
        ),
        tail.start_with_streams(
            StreamLinks::pipeline(Some(tail_end), None),
            shard(false, true)
        ),
    );
    h.expect("start head");
    t.expect("start tail");
    let relay = {
        let tail = tail.clone();
        tokio::task::spawn_blocking(move || tail.run_relay_loop())
    };

    // 2. Generate.
    let first = generate(&head, "req-1", "40").await.expect("first request");
    println!("before the link failure: {first:?}");
    assert_eq!(first, [41, 42, 43]);

    // 3. The link dies. Close our ends of the dead streams first: the
    //    tail's blocked step returns, releasing its engine lock, and its
    //    relay loop parks waiting for a re-attach.
    pump.abort();
    let _ = pump.await;
    println!("link killed; the tail's relay loop parks instead of exiting");

    // A rejected re-attach swaps nothing and leaves the old links in place:
    // the tail has no downstream to replace (and must replace its upstream).
    let (_, spare, _spare_pump) = hop();
    let rejected = tail
        .reattach(StreamLinks::pipeline(None, Some(spare)))
        .await;
    assert!(
        matches!(rejected, Err(EngineError::PeerRejected(_))),
        "{rejected:?}"
    );
    println!("rejected re-attach: {}", rejected.unwrap_err());

    // 4. Re-attach fresh streams on both sides of the hop.
    let (head_end, tail_end, pump) = hop();
    tail.reattach(StreamLinks::pipeline(Some(tail_end), None))
        .await
        .expect("re-attach tail");
    head.reattach(StreamLinks::pipeline(None, Some(head_end)))
        .await
        .expect("re-attach head");
    println!("re-attached both ends of the hop");

    // 5. Same live engines, new link.
    let second = generate(&head, "req-2", "100")
        .await
        .expect("second request");
    println!("after the re-attach: {second:?}");
    assert_eq!(second, [101, 102, 103]);

    // 6. Shut down. Close our ends first: an idle relay's step is blocked
    //    on its upstream read while holding the engine lock, and `close()`
    //    needs that lock. Then `close()`, which is mandatory in stream mode:
    //    it is what wakes the parked relay loop so it can exit.
    pump.abort();
    let _ = pump.await;
    tail.close();
    head.close();
    let exit = relay.await.expect("relay thread");
    assert_eq!(exit, RelayExit::SlotEmpty);
    println!("closed; relay loop exited with {exit:?}");
}
