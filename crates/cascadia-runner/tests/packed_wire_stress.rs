//! Issue #122: multi-stage packed decode wedges under sustained concurrent
//! load — rank 1 writes the token-frame reply, rank 0's `recv()` never wakes.
//!
//! This harness reproduces the production concurrency shape without OpenVINO:
//!
//! * A DRIVER runtime (rank 0): requests are spawned as tasks (like axum
//!   handlers), each polling a `ChunkStream` whose `poll_next` locks the
//!   sync engine mutex and runs `step()` — fake inference hard-blocks the
//!   polling thread (like `packed.step()`), then the packed wire exchange
//!   runs through `cascadia_runner::run_async` (`block_in_place` +
//!   `block_on`, exactly like `exchange_packed_downstream`).
//! * A RELAY runtime (rank 1): its own tokio runtime (a separate process in
//!   production), driving `Runner::run_relay_loop` on a `spawn_blocking`
//!   thread (`BlockingContextGuard` → naked `block_on`), recv-pair /
//!   infer / send-reply like `step_relay_packed`.
//!
//! The load mirrors `packed_accuracy.py`: solo requests, then a concurrent
//! batch, repeated. A wedge shows up as the watchdog firing with requests
//! stuck mid-generation.
//!
//! The last scenario drives the OTHER half of that wire contract over the
//! same pipeline: a relay whose step fails still OWES its upstream a reply,
//! so it answers with the empty-token-frame NACK, and the driver has to
//! retire the in-flight batch per task while leaving the link alive.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cascadia_engine::{Builder, Engine, EngineError, EngineResult, LoadStream};
use cascadia_runner::{run_async, RelayExit, Runner};
use cascadia_transport::{ActivationClient, ActivationServer, DType, Tensor};
use cascadia_types::{Chunk, GenerationTask, PeerLayout, ShardSpec, TaskId};
use futures::{stream, StreamExt};
use tokio::sync::Mutex as TokioMutex;

const HIDDEN: usize = 64;
const SLOTS: usize = 4;

/// What a NACKed batch tells its tasks. Same words the runtime's
/// `exchange_packed_downstream` uses, so an abort here is recognisable as
/// THE NACK path and not as a teardown, a no-progress close or a dead link.
const NACK_ABORT_REASON: &str = "downstream stage failed its packed step and NACKed this batch \
     (empty token frame); the pipeline link stays aligned";

fn backend(e: impl std::fmt::Display) -> EngineError {
    EngineError::Backend(e.to_string())
}

/// The relay's NACK frame, byte-for-byte what
/// `cascadia_engine_openvino::runtime::packed_nack_frame()` puts on the wire:
/// an EMPTY I64 `[1, 1, 0]` token frame.
///
/// Rebuilt from the same `cascadia-transport` primitives rather than
/// imported: the encoder is a private fn, and `cascadia-engine-openvino`
/// depends on `cascadia-runner`, so it cannot be pulled in here without
/// dragging the whole OpenVINO stack (tokenizers, the genai shim) into this
/// crate's test build. The shape is pinned by
/// `nack_frame_is_the_empty_token_frame` below and by the engine crate's own
/// `packed_nack_frame_is_empty_and_detectable`; a drift in either direction
/// fails one of them.
fn packed_nack_frame() -> Tensor {
    Tensor::new(DType::I64, [1, 1, 0], Vec::new())
}

/// The driver-side NACK predicate, mirroring `is_packed_nack` in the runtime:
/// a reply that carries NO tokens is the downstream saying "batch lost, link
/// fine". Checked before the reply is read as tokens.
fn is_packed_nack(reply: &Tensor) -> bool {
    reply.elements() == Some(0)
}

/// The NACK is empty, wire-consistent (or the transport would refuse to
/// deliver it) and never confusable with a real per-row token reply.
#[test]
fn nack_frame_is_the_empty_token_frame() {
    let nack = packed_nack_frame();
    assert_eq!(nack.shape, [1, 1, 0]);
    assert_eq!(nack.dtype, DType::I64);
    assert!(nack.data.is_empty());
    assert!(is_packed_nack(&nack));
    for rows in 1..=SLOTS as u32 {
        let real = Tensor::new(DType::I64, [1, 1, rows], vec![0u8; rows as usize * 8]);
        assert!(!is_packed_nack(&real), "{rows}-row reply read as a NACK");
    }
}

/// Rank 0 packed engine: admit → fake inference (hard-blocks the polling
/// thread) → plan+hidden/token exchange, one decode token per active task
/// per step. Wire calls mirror `exchange_packed_downstream` on main.
struct DriverEngine {
    handle: tokio::runtime::Handle,
    down: Arc<TokioMutex<ActivationClient>>,
    pending: Vec<GenerationTask>,
    active: Vec<(GenerationTask, u32)>,
    infer: Duration,
    seq: Arc<AtomicU64>,
}

impl DriverEngine {
    /// Mirror of the runtime's `abort_packed_batch`: a failure that loses the
    /// in-flight packed batch retires EVERY active slot and hands each task
    /// its own final error chunk, so no stream is left wedged-active waiting
    /// for tokens that will never come. The downstream link is untouched —
    /// this engine keeps serving and the next step admits whatever is still
    /// pending.
    fn abort_packed_batch(&mut self, aborted: EngineError) -> Vec<(TaskId, Chunk)> {
        // The whole point of the typed variant: a lost batch must never be
        // read as a dead link (which would exit a relay loop and, on rank 0,
        // arm the dead-wire latch).
        assert!(
            !aborted.is_connection_fatal(),
            "a NACKed batch must not classify as a dead link: {aborted}"
        );
        let msg = aborted.to_string();
        self.active
            .drain(..)
            .map(|(task, _)| {
                (
                    task.task_id.clone(),
                    Chunk::error(task.task_id, msg.clone()),
                )
            })
            .collect()
    }
}

impl Engine for DriverEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        self.pending.push(task);
        Ok(())
    }

    fn cancel(&mut self, task_id: &TaskId) {
        self.pending.retain(|t| &t.task_id != task_id);
        self.active.retain(|(t, _)| &t.task_id != task_id);
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        while self.active.len() < SLOTS && !self.pending.is_empty() {
            let t = self.pending.remove(0);
            self.active.push((t, 0));
        }
        if self.active.is_empty() {
            return Ok(Vec::new());
        }

        // Fake stage-0 packed inference: hard-blocks the polling thread the
        // same way `packed.step()` (OV infer) does — no block_in_place.
        std::thread::sleep(self.infer);

        let rows = self.active.len();
        let plan = Tensor::new(DType::I64, [1, 3, rows as u32], vec![0u8; 3 * rows * 8]);
        let hidden = Tensor::new(
            DType::F16,
            [1, rows as u32, HIDDEN as u32],
            vec![0u8; rows * HIDDEN * 2],
        );
        let down = self.down.clone();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);

        // Mirrors exchange_packed_downstream: one lock over send(plan) +
        // send(hidden) + a deadlined, poisoning recv_reply for the owed
        // token frame, and the NACK check that runs BEFORE the reply is
        // read as tokens.
        let exchanged = run_async(&self.handle, async move {
            let mut guard = down.lock().await;
            guard.send(&plan).await.map_err(backend)?;
            guard.send(&hidden).await.map_err(backend)?;
            let (reply, _) = guard
                .recv_reply()
                .await
                .map_err(|e| EngineError::Backend(format!("token recv (seq={seq}): {e}")))?;
            // An EMPTY token frame is the downstream's NACK: its step failed
            // AFTER it consumed our pair, so the batch is lost while the link
            // stays frame-aligned. Typed `BatchAborted` exactly as the runtime
            // types it, so nothing downstream can classify a lost batch as a
            // dead link.
            if is_packed_nack(&reply) {
                return Err(EngineError::BatchAborted(NACK_ABORT_REASON.into()));
            }
            Ok::<usize, EngineError>(reply.shape[2] as usize)
        });
        let reply_rows = match exchanged {
            Ok(rows) => rows,
            // The runtime's `abort_packed_batch`: retire every active slot
            // with its own final error chunk and keep serving.
            Err(aborted @ EngineError::BatchAborted(_)) => {
                return Ok(self.abort_packed_batch(aborted))
            }
            Err(e) => return Err(e),
        };
        if reply_rows != rows {
            return Err(EngineError::Backend(format!(
                "reply rows {reply_rows} != sent rows {rows}"
            )));
        }

        let mut out = Vec::new();
        let mut still = Vec::new();
        for (task, emitted) in self.active.drain(..) {
            let n = emitted + 1;
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
        Ok(out)
    }
}

/// Rank 1 packed relay: recv plan+hidden under the upstream lock, release,
/// fake tail inference, re-lock, send one token per row. Mirrors
/// `step_relay_packed` on main.
struct RelayEngine {
    handle: tokio::runtime::Handle,
    up: Arc<TokioMutex<ActivationServer>>,
    infer: Duration,
}

impl Engine for RelayEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, _task: GenerationTask) -> EngineResult<()> {
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let up = self.up.clone();
        let rows: usize = run_async(&self.handle, async move {
            let mut guard = up.lock().await;
            let (_pf, _) = guard.recv().await.map_err(backend)?;
            // The owed second frame of the pair: deadlined recv_reply,
            // mirroring the fixed step_relay_packed.
            let (hf, _) = guard.recv_reply().await.map_err(backend)?;
            Ok::<usize, EngineError>(hf.shape[1] as usize)
        })?;

        std::thread::sleep(self.infer); // fake tail inference, lock released

        let frame = Tensor::new(DType::I64, [1, 1, rows as u32], vec![0u8; rows * 8]);
        let up = self.up.clone();
        run_async(&self.handle, async move {
            let mut guard = up.lock().await;
            guard.send(&frame).await.map_err(backend)?;
            Ok::<(), EngineError>(())
        })?;
        Ok(Vec::new())
    }
}

/// Builder that hands out a pre-built engine (sockets already connected).
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

fn spec(first: bool) -> ShardSpec {
    ShardSpec {
        model_id: "stress".into(),
        layer_start: 0,
        layer_end: 1,
        total_layers: 2,
        device: "CPU".into(),
        is_first_stage: first,
        is_last_stage: !first,
        tp_size: 1,
        tp_rank: 0,
    }
}

async fn run_one(runner: Arc<Runner>, id: String, max_tokens: u32) -> Result<usize, String> {
    let task = GenerationTask::new(id, "stress prompt").with_max_tokens(max_tokens);
    // The async submit path, like the API layer: sync `generate()` from a
    // worker task blocks the worker on the engine mutex behind an in-flight
    // step — enough simultaneous submits starve the I/O driver (#122).
    let mut stream = runner
        .generate_async(task)
        .await
        .map_err(|e| e.to_string())?;
    let mut n = 0usize;
    while let Some(chunk) = stream.next().await {
        if let Some(err) = chunk.error {
            return Err(err);
        }
        n += 1;
    }
    Ok(n)
}

struct Params {
    workers: usize,
    batch: usize,
    rounds: usize,
    solo_per_round: usize,
    max_tokens: u32,
    watchdog: Duration,
}

/// Build the two-runtime pipeline, run the load, return Ok(()) or a
/// description of what wedged/failed.
fn run_stress(p: Params) -> Result<(), String> {
    let driver_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(p.workers)
        .enable_all()
        .build()
        .unwrap();
    let relay_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    // Wire the single duplex TCP connection: server socket lives on the
    // relay runtime's driver, client socket on the driver runtime's driver —
    // exactly the two-process registration split production has.
    let (server, port) = relay_rt.block_on(async {
        let mut s = ActivationServer::new("127.0.0.1", 0);
        s.start().await.unwrap();
        let port = s.port();
        (s, port)
    });
    let accept = relay_rt.spawn(async move {
        let mut s = server;
        s.accept().await.unwrap();
        s
    });
    let client = driver_rt.block_on(async {
        let mut c = ActivationClient::new("127.0.0.1", port);
        c.connect().await.unwrap();
        c
    });
    let server = relay_rt.block_on(accept).unwrap();

    let seq = Arc::new(AtomicU64::new(0));
    let driver_engine = DriverEngine {
        handle: driver_rt.handle().clone(),
        down: Arc::new(TokioMutex::new(client)),
        pending: Vec::new(),
        active: Vec::new(),
        infer: Duration::from_micros(300),
        seq: seq.clone(),
    };
    let relay_engine = RelayEngine {
        handle: relay_rt.handle().clone(),
        up: Arc::new(TokioMutex::new(server)),
        infer: Duration::from_micros(500),
    };

    let driver_runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
        engine: Some(Box::new(driver_engine)),
    })));
    let relay_runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
        engine: Some(Box::new(relay_engine)),
    })));
    driver_rt
        .block_on(driver_runner.start(PeerLayout::default(), spec(true)))
        .unwrap();
    relay_rt
        .block_on(relay_runner.start(PeerLayout::default(), spec(false)))
        .unwrap();

    let relay_for_loop = relay_runner.clone();
    let relay_join = relay_rt.spawn_blocking(move || relay_for_loop.run_relay_loop());

    // The load, in its own thread so the test thread can watchdog it.
    let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let load_runner = driver_runner.clone();
    let handle = driver_rt.handle().clone();
    let batch = p.batch;
    let rounds = p.rounds;
    let solo = p.solo_per_round;
    let max_tokens = p.max_tokens;
    std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            for round in 0..rounds {
                for i in 0..solo {
                    let r = load_runner.clone();
                    let id = format!("solo-{round}-{i}");
                    let jh = handle.spawn(run_one(r, id.clone(), max_tokens));
                    let out = handle.block_on(jh).map_err(|e| format!("join {id}: {e}"))?;
                    out.map_err(|e| format!("{id}: {e}"))?;
                }
                let mut joins = Vec::new();
                for i in 0..batch {
                    let r = load_runner.clone();
                    let id = format!("batch-{round}-{i}");
                    joins.push((id.clone(), handle.spawn(run_one(r, id, max_tokens))));
                }
                for (id, jh) in joins {
                    let out = handle.block_on(jh).map_err(|e| format!("join {id}: {e}"))?;
                    out.map_err(|e| format!("{id}: {e}"))?;
                }
            }
            Ok(())
        })();
        let _ = done_tx.send(result);
    });

    let verdict = match done_rx.recv_timeout(p.watchdog) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(format!("request failed: {e}")),
        Err(_) => {
            let done = seq.load(Ordering::Relaxed);
            // Grace period: distinguish a permanent wedge from a slow run.
            match done_rx.recv_timeout(Duration::from_secs(10)) {
                Ok(Ok(())) => Err("SLOW: finished only in the grace period".into()),
                Ok(Err(e)) => Err(format!("request failed late: {e}")),
                Err(_) => {
                    let now = seq.load(Ordering::Relaxed);
                    Err(format!(
                        "WEDGED: load incomplete after {:?}+10s; exchanges {done} -> {now} \
                         ({} in grace period)",
                        p.watchdog,
                        now - done
                    ))
                }
            }
        }
    };

    if verdict.is_ok() {
        // Clean teardown only on success: a wedged engine holds its mutex
        // forever, so close()/Drop would deadlock this thread. On failure we
        // leak the runtimes deliberately and let the panic end the process.
        driver_runner.close();
        relay_runner.close();
        drop(relay_join); // exits via ConnectionFatal/SlotEmpty once sockets drop
        driver_rt.shutdown_background();
        relay_rt.shutdown_background();
    } else {
        std::mem::forget(driver_rt);
        std::mem::forget(relay_rt);
        std::mem::forget(driver_runner);
        std::mem::forget(relay_runner);
        std::mem::forget(relay_join);
    }
    let _ = RelayExit::SlotEmpty; // silence unused import when cfg varies
    verdict
}

/// Production-like shape: 8 workers, batch of 4 — the packed_accuracy load.
#[test]
fn packed_multistage_survives_sustained_concurrency() {
    let p = Params {
        workers: 8,
        batch: 4,
        rounds: 4,
        solo_per_round: 12,
        max_tokens: 16,
        watchdog: Duration::from_secs(120),
    };
    if let Err(e) = run_stress(p) {
        panic!("{e}");
    }
}

/// Small box / higher admission: more concurrent streams than worker
/// threads. Pre-fix this wedged deterministically on the first concurrent
/// batch: every submit and every stream poll hard-blocked a worker on the
/// sync engine mutex behind the in-flight step; with all workers blocked
/// nothing could poll the tokio I/O + timer drivers, so the token frame sat
/// unread in the socket buffer and no recv deadline could fire either —
/// issue #122's exact signature (reply written, recv never woke, no WARN or
/// ERROR, never recovered).
#[test]
fn packed_multistage_survives_more_streams_than_workers() {
    let p = Params {
        workers: 4,
        batch: 6,
        rounds: 3,
        solo_per_round: 8,
        max_tokens: 16,
        watchdog: Duration::from_secs(120),
    };
    if let Err(e) = run_stress(p) {
        panic!("{e}");
    }
}

/// Clients hanging up mid-generation must not wedge the survivors: stream
/// Drop and Runner::cancel no longer block on the engine mutex (deferred
/// cancels), so a disconnect burst during a concurrent batch cannot eat
/// worker threads either.
#[test]
fn packed_multistage_survives_mid_generation_disconnects() {
    let driver_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let relay_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (server, port) = relay_rt.block_on(async {
        let mut s = ActivationServer::new("127.0.0.1", 0);
        s.start().await.unwrap();
        let port = s.port();
        (s, port)
    });
    let accept = relay_rt.spawn(async move {
        let mut s = server;
        s.accept().await.unwrap();
        s
    });
    let client = driver_rt.block_on(async {
        let mut c = ActivationClient::new("127.0.0.1", port);
        c.connect().await.unwrap();
        c
    });
    let server = relay_rt.block_on(accept).unwrap();

    let seq = Arc::new(AtomicU64::new(0));
    let driver_runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
        engine: Some(Box::new(DriverEngine {
            handle: driver_rt.handle().clone(),
            down: Arc::new(TokioMutex::new(client)),
            pending: Vec::new(),
            active: Vec::new(),
            infer: Duration::from_micros(300),
            seq: seq.clone(),
        })),
    })));
    let relay_runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
        engine: Some(Box::new(RelayEngine {
            handle: relay_rt.handle().clone(),
            up: Arc::new(TokioMutex::new(server)),
            infer: Duration::from_micros(500),
        })),
    })));
    driver_rt
        .block_on(driver_runner.start(PeerLayout::default(), spec(true)))
        .unwrap();
    relay_rt
        .block_on(relay_runner.start(PeerLayout::default(), spec(false)))
        .unwrap();
    let relay_for_loop = relay_runner.clone();
    let _relay_join = relay_rt.spawn_blocking(move || relay_for_loop.run_relay_loop());

    let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let handle = driver_rt.handle().clone();
    let load_runner = driver_runner.clone();
    std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            for round in 0..3 {
                let mut joins = Vec::new();
                for i in 0..6 {
                    let r = load_runner.clone();
                    let id = format!("dc-{round}-{i}");
                    // Odd streams are abandoned after the first chunk (client
                    // disconnect → ChunkStream::drop on a worker thread).
                    let abandon = i % 2 == 1;
                    joins.push((
                        id.clone(),
                        handle.spawn(async move {
                            let task = GenerationTask::new(id, "stress prompt").with_max_tokens(24);
                            let mut stream =
                                r.generate_async(task).await.map_err(|e| e.to_string())?;
                            let mut n = 0usize;
                            while let Some(chunk) = stream.next().await {
                                if let Some(err) = chunk.error {
                                    return Err(err);
                                }
                                n += 1;
                                if abandon && n >= 1 {
                                    return Ok(n); // drops the stream mid-generation
                                }
                            }
                            Ok(n)
                        }),
                    ));
                }
                for (id, jh) in joins {
                    let out = handle.block_on(jh).map_err(|e| format!("join {id}: {e}"))?;
                    out.map_err(|e| format!("{id}: {e}"))?;
                }
            }
            Ok(())
        })();
        let _ = done_tx.send(result);
    });

    match done_rx.recv_timeout(Duration::from_secs(120)) {
        Ok(Ok(())) => {
            driver_runner.close();
            relay_runner.close();
            driver_rt.shutdown_background();
            relay_rt.shutdown_background();
        }
        Ok(Err(e)) => {
            std::mem::forget(driver_rt);
            std::mem::forget(relay_rt);
            panic!("request failed: {e}");
        }
        Err(_) => {
            std::mem::forget(driver_rt);
            std::mem::forget(relay_rt);
            panic!(
                "WEDGED with disconnect load (exchanges: {})",
                seq.load(Ordering::Relaxed)
            );
        }
    }
}

// =====================================================================
// Scenario 4 (review finding M7a): relay failure → NACK → driver-side
// batch abort, under concurrency.
//
// The engine-side hardening (a relay that always answers, an empty token
// frame as the NACK, `abort_packed_batch` retiring every slot with its own
// error chunk, `BatchAborted` keeping the link alive) had no end-to-end
// coverage: the real `exchange_packed_downstream` / `abort_packed_batch`
// live on `OvRuntimeEngine`, which cannot be constructed without an
// OpenVINO runtime. What IS reachable is the protocol and the runner
// behaviour around it — the mock engines below speak the real wire format
// over a real loopback socket, so everything from "the relay's step failed"
// to "each affected client gets its own error chunk and the next request
// still works" is exercised for real.
//
// The pipeline setup is deliberately duplicated from `run_stress` rather
// than factored out of it: the three scenarios above are the #122
// regression guard and stay untouched.
// =====================================================================

/// What the relay should do with the exchanges it is about to answer.
/// Flipped by the load thread between phases, read by the relay engine.
#[derive(Default)]
struct NackPlan {
    /// NACK every Nth exchange; 0 = injection off.
    every: AtomicU64,
    /// NACK the very next exchange, once. Used for the deterministic
    /// full-slate abort: arm, submit a whole batch, only then poll.
    once: AtomicBool,
    /// Exchanges answered so far.
    seen: AtomicU64,
    /// NACKs actually put on the wire.
    sent: AtomicU64,
}

impl NackPlan {
    /// Decide — and record — whether this exchange is answered with a NACK.
    /// `Some(n)` is the 1-based index of the NACK, for the error text.
    fn should_nack(&self) -> Option<u64> {
        let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
        let every = self.every.load(Ordering::SeqCst);
        let nack =
            self.once.swap(false, Ordering::SeqCst) || (every != 0 && n.is_multiple_of(every));
        nack.then(|| self.sent.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn arm_every(&self, every: u64) {
        self.every.store(every, Ordering::SeqCst);
    }

    fn arm_once(&self) {
        self.once.store(true, Ordering::SeqCst);
    }

    fn nacks_sent(&self) -> u64 {
        self.sent.load(Ordering::SeqCst)
    }
}

/// Rank 1 packed relay that can fail its tail inference and NACK, exactly
/// the way `step_relay_packed` does: it consumes the (plan, hidden) pair,
/// the body fails, and it STILL answers — with the empty token frame —
/// before reporting the failure to its own relay loop.
///
/// The invariant the whole scenario rests on: one pair consumed, one reply
/// sent, always, so the link stays frame-aligned across any number of
/// failures.
struct NackingRelayEngine {
    handle: tokio::runtime::Handle,
    up: Arc<TokioMutex<ActivationServer>>,
    infer: Duration,
    plan: Arc<NackPlan>,
}

impl Engine for NackingRelayEngine {
    fn warmup(&mut self) {}

    fn submit(&mut self, _task: GenerationTask) -> EngineResult<()> {
        Ok(())
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        let up = self.up.clone();
        let rows: usize = run_async(&self.handle, async move {
            let mut guard = up.lock().await;
            let (_pf, _) = guard.recv().await.map_err(backend)?;
            let (hf, _) = guard.recv_reply().await.map_err(backend)?;
            Ok::<usize, EngineError>(hf.shape[1] as usize)
        })?;

        std::thread::sleep(self.infer); // fake tail inference, lock released

        // The pair is consumed, so from here EVERY path has to answer the
        // upstream — a swallowed failure leaves it awaiting a token frame
        // that never comes (the silent-deadlock half of #122).
        let failure = self.plan.should_nack().map(|n| {
            EngineError::Backend(format!(
                "injected relay tail-inference failure #{n} ({rows} rows)"
            ))
        });
        let frame = match &failure {
            None => Tensor::new(DType::I64, [1, 1, rows as u32], vec![0u8; rows * 8]),
            Some(_) => packed_nack_frame(),
        };
        let up = self.up.clone();
        run_async(&self.handle, async move {
            let mut guard = up.lock().await;
            guard.send(&frame).await.map_err(backend)?;
            Ok::<(), EngineError>(())
        })?;

        match failure {
            // What `step_relay_packed` returns once the NACK is out: the
            // body error. Non-fatal, so `run_relay_loop` backs off and keeps
            // driving instead of exiting for a supervisor rebuild — the
            // injected text must never trip the fatal-substring classifier.
            Some(e) => {
                assert!(
                    !e.is_connection_fatal(),
                    "an injected step failure must stay non-fatal: {e}"
                );
                Err(e)
            }
            None => Ok(Vec::new()),
        }
    }
}

/// How one request ended. An aborted batch is an OUTCOME here, not a test
/// failure — the point of the scenario is that aborts get DELIVERED rather
/// than hung.
enum Outcome {
    Completed(usize),
    Aborted(String),
}

#[derive(Default)]
struct Tally {
    completed: usize,
    aborted: usize,
    chunks: usize,
    aborts: Vec<String>,
}

impl Tally {
    fn record(&mut self, o: Outcome) {
        match o {
            Outcome::Completed(n) => {
                self.completed += 1;
                self.chunks += n;
            }
            Outcome::Aborted(msg) => {
                self.aborted += 1;
                self.aborts.push(msg);
            }
        }
    }

    fn merge(&mut self, other: Tally) {
        self.completed += other.completed;
        self.aborted += other.aborted;
        self.chunks += other.chunks;
        self.aborts.extend(other.aborts);
    }

    fn total(&self) -> usize {
        self.completed + self.aborted
    }

    /// Every abort must be THE NACK abort: a lost batch, naming the
    /// downstream NACK. A teardown, a no-progress close or a dead-link error
    /// would all satisfy "ended with an error chunk" while meaning the
    /// opposite of what this scenario claims.
    fn check_aborts_are_nacks(&self, phase: &str) -> Result<(), String> {
        for msg in &self.aborts {
            if !msg.starts_with("batch aborted: ") || !msg.contains("NACKed this batch") {
                return Err(format!(
                    "{phase}: a request ended on something other than the NACK abort: {msg}"
                ));
            }
        }
        Ok(())
    }
}

/// Drain a stream to its terminal chunk. The error chunk is final, so this
/// returns as soon as one arrives — a hang can only show up as the watchdog
/// firing, never as a silent pass.
async fn drain_outcome(mut stream: cascadia_runner::ChunkStream) -> Outcome {
    let mut n = 0usize;
    while let Some(chunk) = stream.next().await {
        if let Some(err) = chunk.error {
            return Outcome::Aborted(err);
        }
        n += 1;
    }
    Outcome::Completed(n)
}

async fn run_one_outcome(
    runner: Arc<Runner>,
    id: String,
    max_tokens: u32,
) -> Result<Outcome, String> {
    let task = GenerationTask::new(id, "stress prompt").with_max_tokens(max_tokens);
    let stream = runner
        .generate_async(task)
        .await
        .map_err(|e| e.to_string())?;
    Ok(drain_outcome(stream).await)
}

/// One phase of the NACK load, shaped like a `run_stress` round: `solo`
/// sequential requests, then one concurrent batch of `batch`.
fn nack_phase(
    handle: &tokio::runtime::Handle,
    runner: &Arc<Runner>,
    label: &str,
    solo: usize,
    batch: usize,
    max_tokens: u32,
) -> Result<Tally, String> {
    let mut tally = Tally::default();
    for i in 0..solo {
        let id = format!("{label}-solo-{i}");
        let jh = handle.spawn(run_one_outcome(runner.clone(), id.clone(), max_tokens));
        let out = handle.block_on(jh).map_err(|e| format!("join {id}: {e}"))?;
        tally.record(out.map_err(|e| format!("{id}: {e}"))?);
    }
    let mut joins = Vec::new();
    for i in 0..batch {
        let id = format!("{label}-batch-{i}");
        joins.push((
            id.clone(),
            handle.spawn(run_one_outcome(runner.clone(), id, max_tokens)),
        ));
    }
    for (id, jh) in joins {
        let out = handle.block_on(jh).map_err(|e| format!("join {id}: {e}"))?;
        tally.record(out.map_err(|e| format!("{id}: {e}"))?);
    }
    Ok(tally)
}

/// Submit a FULL slate before anything polls, so the driver admits every row
/// in one admission pass and the very next exchange carries all of them.
/// That makes the concurrent abort deterministic: one NACK, one exchange,
/// every in-flight task retired together.
///
/// `generate_async` returns only once its submit has landed, so awaiting all
/// of them puts every task in the engine's queue with no step yet run.
fn nack_full_slate(
    handle: &tokio::runtime::Handle,
    runner: &Arc<Runner>,
    label: &str,
    slate: usize,
    max_tokens: u32,
) -> Result<Tally, String> {
    let mut streams = Vec::new();
    for i in 0..slate {
        let id = format!("{label}-{i}");
        let runner = runner.clone();
        let task = GenerationTask::new(id.clone(), "stress prompt").with_max_tokens(max_tokens);
        let stream = handle
            .block_on(async move { runner.generate_async(task).await })
            .map_err(|e| format!("{id}: submit: {e}"))?;
        streams.push((id, stream));
    }
    let joins: Vec<_> = streams
        .into_iter()
        .map(|(id, s)| (id, handle.spawn(drain_outcome(s))))
        .collect();
    let mut tally = Tally::default();
    for (id, jh) in joins {
        tally.record(handle.block_on(jh).map_err(|e| format!("join {id}: {e}"))?);
    }
    Ok(tally)
}

struct NackParams {
    workers: usize,
    batch: usize,
    rounds: usize,
    solo_per_round: usize,
    max_tokens: u32,
    /// NACK every Nth exchange during the sustained phase. Kept ABOVE
    /// `max_tokens` on purpose: a solo request spans at most `max_tokens`
    /// consecutive exchanges, so no two consecutive solos can both be hit
    /// and the phase is guaranteed to contain completions as well as aborts.
    every: u64,
    /// Fewest NACKs the sustained phase must actually deliver, so "the link
    /// survives N failures" means something.
    min_nacks: u64,
    watchdog: Duration,
}

/// Build the pipeline with a NACK-injecting relay and run the four phases.
fn run_nack_stress(p: NackParams) -> Result<(), String> {
    assert!(
        p.every > p.max_tokens as u64,
        "every ({}) must exceed max_tokens ({}) — see NackParams::every",
        p.every,
        p.max_tokens
    );
    let driver_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(p.workers)
        .enable_all()
        .build()
        .unwrap();
    let relay_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (server, port) = relay_rt.block_on(async {
        let mut s = ActivationServer::new("127.0.0.1", 0);
        s.start().await.unwrap();
        let port = s.port();
        (s, port)
    });
    let accept = relay_rt.spawn(async move {
        let mut s = server;
        s.accept().await.unwrap();
        s
    });
    let client = driver_rt.block_on(async {
        let mut c = ActivationClient::new("127.0.0.1", port);
        c.connect().await.unwrap();
        c
    });
    let server = relay_rt.block_on(accept).unwrap();

    let seq = Arc::new(AtomicU64::new(0));
    let plan = Arc::new(NackPlan::default());
    let driver_runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
        engine: Some(Box::new(DriverEngine {
            handle: driver_rt.handle().clone(),
            down: Arc::new(TokioMutex::new(client)),
            pending: Vec::new(),
            active: Vec::new(),
            infer: Duration::from_micros(300),
            seq: seq.clone(),
        })),
    })));
    let relay_runner = Arc::new(Runner::new(Box::new(PrebuiltBuilder {
        engine: Some(Box::new(NackingRelayEngine {
            handle: relay_rt.handle().clone(),
            up: Arc::new(TokioMutex::new(server)),
            infer: Duration::from_micros(500),
            plan: plan.clone(),
        })),
    })));
    driver_rt
        .block_on(driver_runner.start(PeerLayout::default(), spec(true)))
        .unwrap();
    relay_rt
        .block_on(relay_runner.start(PeerLayout::default(), spec(false)))
        .unwrap();

    let relay_for_loop = relay_runner.clone();
    let relay_join = relay_rt.spawn_blocking(move || relay_for_loop.run_relay_loop());

    let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let load_runner = driver_runner.clone();
    let handle = driver_rt.handle().clone();
    let load_plan = plan.clone();
    let batch = p.batch;
    let rounds = p.rounds;
    let solo = p.solo_per_round;
    let max_tokens = p.max_tokens;
    let every = p.every;
    let min_nacks = p.min_nacks;
    std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            // ---- A. baseline, injection off ----------------------------
            // Establishes that any abort later can only have come from an
            // injected NACK, not from the harness being flaky.
            let warm = nack_phase(&handle, &load_runner, "warm", solo, batch, max_tokens)?;
            if warm.aborted != 0 {
                return Err(format!(
                    "baseline: {} of {} requests aborted with injection OFF: {:?}",
                    warm.aborted,
                    warm.total(),
                    warm.aborts
                ));
            }
            if load_plan.nacks_sent() != 0 {
                return Err("baseline: the relay NACKed with injection OFF".into());
            }

            // ---- B. ONE NACK onto a full slate -------------------------
            // The concurrent case, made deterministic: `SLOTS` requests are
            // submitted before anything polls, so the first exchange carries
            // every row and the single NACK has to retire the whole batch —
            // each task with its own error chunk, none left wedged-active.
            load_plan.arm_once();
            let slate = nack_full_slate(&handle, &load_runner, "slate", SLOTS, max_tokens)?;
            slate.check_aborts_are_nacks("full slate")?;
            if load_plan.nacks_sent() != 1 {
                return Err(format!(
                    "full slate: expected exactly 1 NACK, the relay sent {}",
                    load_plan.nacks_sent()
                ));
            }
            if slate.aborted != SLOTS {
                return Err(format!(
                    "full slate: one NACK retired {} of {SLOTS} in-flight requests (the whole \
                     batch is lost, so every one of them owes its client an error chunk): {:?}",
                    slate.aborted, slate.aborts
                ));
            }

            // ---- C. sustained: NACK every Nth exchange -----------------
            load_plan.arm_every(every);
            let mut hot = Tally::default();
            for round in 0..rounds {
                let label = format!("nack-{round}");
                hot.merge(nack_phase(
                    &handle,
                    &load_runner,
                    &label,
                    solo,
                    batch,
                    max_tokens,
                )?);
            }
            load_plan.arm_every(0);
            hot.check_aborts_are_nacks("under load")?;
            let nacks = load_plan.nacks_sent() - 1; // minus the full-slate one
            if nacks < min_nacks {
                return Err(format!(
                    "under load: only {nacks} NACKs were injected (wanted >= {min_nacks}); the \
                     scenario did not exercise what it claims"
                ));
            }
            // Each NACK retires every row of the exchange it answered, and a
            // retired task cannot be retired twice — so aborts must at least
            // keep pace with NACKs. Fewer means a task was silently dropped
            // instead of being handed its error chunk.
            if (hot.aborted as u64) < nacks {
                return Err(format!(
                    "under load: {nacks} NACKs retired only {} requests — every NACKed batch \
                     owes each of its tasks a final error chunk",
                    hot.aborted
                ));
            }
            if hot.completed == 0 || hot.chunks == 0 {
                return Err(format!(
                    "under load: no request completed while NACKs were being injected \
                     ({} aborted, {} chunks) — progress stopped instead of continuing",
                    hot.aborted, hot.chunks
                ));
            }

            // ---- D. the link survived ----------------------------------
            // Injection off again: after N aborted batches the socket must
            // still be frame-aligned and every request must complete
            // normally. A torn-down or desynced link fails here.
            let after = nack_phase(&handle, &load_runner, "after", solo, batch, max_tokens)?;
            if after.aborted != 0 {
                return Err(format!(
                    "the link did not survive {nacks} NACKs: {} of {} later requests aborted: {:?}",
                    after.aborted,
                    after.total(),
                    after.aborts
                ));
            }
            if after.chunks == 0 {
                return Err("no tokens were produced after injection stopped".into());
            }
            Ok(())
        })();
        let _ = done_tx.send(result);
    });

    let verdict = match done_rx.recv_timeout(p.watchdog) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(_) => {
            let done = seq.load(Ordering::Relaxed);
            match done_rx.recv_timeout(Duration::from_secs(10)) {
                Ok(Ok(())) => Err("SLOW: finished only in the grace period".into()),
                Ok(Err(e)) => Err(format!("failed late: {e}")),
                Err(_) => {
                    let now = seq.load(Ordering::Relaxed);
                    Err(format!(
                        "WEDGED: NACK load incomplete after {:?}+10s; exchanges {done} -> {now} \
                         ({} in grace period), NACKs sent {}",
                        p.watchdog,
                        now - done,
                        plan.nacks_sent()
                    ))
                }
            }
        }
    };

    if verdict.is_ok() {
        driver_runner.close();
        relay_runner.close();
        drop(relay_join);
        driver_rt.shutdown_background();
        relay_rt.shutdown_background();
    } else {
        // Same rule as `run_stress`: a wedged engine holds its mutex, so
        // close()/Drop would deadlock this thread. Leak deliberately and let
        // the panic end the process.
        std::mem::forget(driver_rt);
        std::mem::forget(relay_rt);
        std::mem::forget(driver_runner);
        std::mem::forget(relay_runner);
        std::mem::forget(relay_join);
    }
    verdict
}

/// A relay that fails its step NACKs the driver instead of going silent, and
/// the driver retires the lost batch per task without losing the link:
/// affected requests end on an error chunk (never a hang, never a truncated
/// success), unaffected and subsequent ones complete normally, and the
/// pipeline keeps making progress across a run of failures.
#[test]
fn packed_multistage_nack_aborts_the_batch_without_killing_the_link() {
    let p = NackParams {
        workers: 4,
        batch: 6,
        rounds: 3,
        solo_per_round: 4,
        max_tokens: 4,
        every: 8,
        min_nacks: 3,
        watchdog: Duration::from_secs(120),
    };
    if let Err(e) = run_nack_stress(p) {
        panic!("{e}");
    }
}
