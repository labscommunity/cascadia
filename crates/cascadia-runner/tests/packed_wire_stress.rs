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

use std::sync::atomic::{AtomicU64, Ordering};
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

fn backend(e: impl std::fmt::Display) -> EngineError {
    EngineError::Backend(e.to_string())
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
        // token frame.
        let reply_rows: usize = run_async(&self.handle, async move {
            let mut guard = down.lock().await;
            guard.send(&plan).await.map_err(backend)?;
            guard.send(&hidden).await.map_err(backend)?;
            let (reply, _) = guard
                .recv_reply()
                .await
                .map_err(|e| EngineError::Backend(format!("token recv (seq={seq}): {e}")))?;
            Ok::<usize, EngineError>(reply.shape[2] as usize)
        })?;
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
