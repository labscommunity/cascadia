//! Internal tracker, issue #40: a head whose downstream stops answering must
//! release the engine mutex on its own deadline, and must NOT drop the socket.
//!
//! Both halves are the whole point of the bounded token recv, and neither is
//! observable from `cascadia-engine-openvino`'s unit tests — `OvRuntimeEngine`
//! cannot be constructed without a compiled OpenVINO IR, so those tests
//! exercise the wire helpers in isolation and never see the engine lock.
//!
//! This harness rebuilds the production shape without OpenVINO: an engine whose
//! `step()` waits for a downstream token exactly as
//! `recv_token_from_downstream` does — a real loopback `ActivationClient`, a
//! real bounded `recv_token`, dispatched through `run_async`
//! (`block_in_place` + `block_on`) while holding the sync engine mutex — with a
//! peer that accepts the connection and then says nothing.
//!
//! What it pins:
//!
//! * the wait ends on the CALLER's deadline, not the transport's frame-idle
//!   ceiling (900s by default), so the mutex is free for the next request;
//! * a second request really does get served afterwards, which is the
//!   user-visible #40 symptom — on the pre-fix path the head's socket was
//!   dropped as connection-fatal and, because `ActivationClient` dials once
//!   with nothing to re-dial it, every later request failed `NotConnected`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_runner::{run_async, Runner};
use cascadia_transport::{ActivationClient, ActivationServer};
use cascadia_types::{Chunk, GenerationTask, PeerLayout, ShardSpec, TaskId};
use futures::{stream, StreamExt};
use tokio::sync::Mutex as TokioMutex;

/// The bounded wait each `step()` gives the downstream. Small enough to keep
/// the test quick, large enough that the assertions below are not racing
/// scheduler noise.
const TOKEN_DEADLINE: Duration = Duration::from_millis(400);

/// A head stage whose downstream never answers.
///
/// `step()` mirrors `recv_token_from_downstream`: clone the downstream handle,
/// then `run_async` a bounded `recv_token` — the same `block_in_place` +
/// `block_on` dispatch the real engine uses, executed while the runner holds
/// the sync engine mutex.
struct SilentDownstreamEngine {
    downstream: Arc<TokioMutex<ActivationClient>>,
    active: Option<TaskId>,
    handle: tokio::runtime::Handle,
}

impl Engine for SilentDownstreamEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        self.active = Some(task.task_id);
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let Some(task_id) = self.active.clone() else {
            return Ok(Vec::new());
        };
        let downstream = self.downstream.clone();
        let res = run_async(&self.handle, async move {
            let mut guard = downstream.lock().await;
            guard.recv_token(TOKEN_DEADLINE).await
        });
        // The task is retired either way: `step_first`'s catch clears the
        // active task so the next submit starts fresh.
        self.active = None;
        match res {
            // The peer is silent, so this branch is what the test exercises.
            // Flattened to `Backend` exactly as the real engine does — and it
            // must stay NON-fatal so the socket survives.
            Err(e) => Err(EngineError::Backend(e.to_string()).for_task(task_id)),
            Ok(_) => Ok(Vec::new()),
        }
    }
}

struct PrebuiltBuilder {
    engine: Option<Box<dyn Engine>>,
}

#[async_trait]
impl Builder for PrebuiltBuilder {
    async fn connect(&mut self, _peers: PeerLayout) -> EngineResult<()> {
        Ok(())
    }
    async fn load(&mut self, _shard: ShardSpec) -> EngineResult<LoadStream> {
        Ok(Box::pin(stream::iter(Vec::new())))
    }
    fn build(mut self: Box<Self>) -> EngineResult<Box<dyn Engine>> {
        self.engine.take().ok_or(EngineError::NotLoaded)
    }
}

fn spec() -> ShardSpec {
    ShardSpec {
        model_id: "token-wait".into(),
        layer_start: 0,
        layer_end: 1,
        total_layers: 2,
        device: "CPU".into(),
        is_first_stage: true,
        is_last_stage: false,
        tp_size: 1,
        tp_rank: 0,
    }
}

/// Drive one request to completion, returning its terminal error text (these
/// requests are all expected to fail — the downstream never answers).
async fn run_one(runner: Arc<Runner>, id: &str) -> Option<String> {
    let task = GenerationTask::new(id, "prompt").with_max_tokens(4);
    let mut stream = match runner.generate_async(task).await {
        Ok(s) => s,
        Err(e) => return Some(e.to_string()),
    };
    while let Some(chunk) = stream.next().await {
        if let Some(err) = chunk.error {
            return Some(err);
        }
    }
    None
}

#[test]
fn a_silent_downstream_releases_the_engine_lock_on_its_own_deadline() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        // A peer that accepts and then never sends: "connected but silent",
        // the black-holed downstream #40 is about.
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let accepted = tokio::spawn(async move {
            server.accept().await.unwrap();
            // Hold the server alive for the whole test; dropping it would
            // close the socket and turn this into a clean-EOF test instead.
            tokio::time::sleep(Duration::from_secs(30)).await;
        });

        let mut client = ActivationClient::new("127.0.0.1", port);
        client.connect().await.unwrap();
        let downstream = Arc::new(TokioMutex::new(client));

        let engine = SilentDownstreamEngine {
            downstream: downstream.clone(),
            active: None,
            handle: tokio::runtime::Handle::current(),
        };
        let runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
            engine: Some(Box::new(engine)),
        })));
        runner.start(PeerLayout::default(), spec()).await.unwrap();

        // FIRST request: wedges on the silent downstream.
        let t0 = Instant::now();
        let first = run_one(runner.clone(), "req-1").await;
        let first_elapsed = t0.elapsed();

        assert!(first.is_some(), "a silent downstream must fail the request");
        // The load-bearing bound. Pre-fix, the frame-start wait was exempt from
        // the deadline and ran to the frame-idle ceiling (900s by default), so
        // this is the assertion that distinguishes a bounded wait from an
        // unbounded one. Generous multiple of the deadline: what matters is
        // "the caller's deadline", not "the 900s ceiling".
        assert!(
            first_elapsed < Duration::from_secs(10),
            "the token wait ignored its deadline: {first_elapsed:?}"
        );

        // SECOND request: the engine mutex was released, so this is served
        // (and fails the same bounded way) rather than hanging behind a lock
        // held for the idle ceiling.
        let t1 = Instant::now();
        let second = run_one(runner.clone(), "req-2").await;
        let second_elapsed = t1.elapsed();

        assert!(
            second.is_some(),
            "the second request should reach the engine and fail on the same silent peer"
        );
        assert!(
            second_elapsed < Duration::from_secs(10),
            "the engine lock was not released after the first wait: {second_elapsed:?}"
        );

        // The socket must have SURVIVED the first wait. This is the half that
        // actually fixes #40: the engine dialed once and cannot re-dial, so a
        // fatal classification would strand the head permanently and every
        // later request would read "not connected" instead of timing out
        // against a live peer.
        let second_err = second.unwrap();
        assert!(
            second_err.contains("frame-start wait timed out"),
            "the second request should still be talking to a LIVE socket, got: {second_err}"
        );
        assert!(
            !second_err.contains("not connected"),
            "the downstream socket was dropped by the first timeout: {second_err}"
        );

        accepted.abort();
    });
}
