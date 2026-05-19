//! Failover orchestrator skeleton — composes iter 091 (KV migration
//! wire frame + Runner `extract_kv_slab` / `install_kv_slab`), iter 092
//! (heartbeat wire + watchdog), and iter 094 (rank-0 cadence loop) into
//! a single dead-worker-recovery state machine.
//!
//! See `docs/architecture/failover-orchestrator.md` for the full design
//! (state machine, in-flight Forward handling, KV consistency,
//! authentication, split-brain). This module ships the skeleton:
//!
//! - [`Orchestrator`] struct that owns the per-target failover plan:
//!   the dead worker's watchdog handle, a backup [`PeerEndpoint`], the
//!   user-supplied spawn-replacement callback, and a routing-update
//!   sink that re-points rank 0's downstream socket once the swap
//!   commits.
//! - [`SpawnReplacement`] trait — pluggable so unit tests can assert
//!   "the callback fired with the right backup address" without
//!   actually forking a worker process.
//! - [`OrchestratorEvent`] enum — what an external observer (CLI,
//!   metrics endpoint, dashboard) sees as the state machine advances.
//! - [`FailoverState`] — the FSM state, exposed for tests + diagnostics.
//! - [`on_watchdog_dead`] — the entry point iter 094's cadence loop
//!   calls when the watchdog crosses its `max_misses` threshold. iter
//!   094 already exposes the shared [`HeartbeatWatchdog`] on
//!   `SparseMoEEngine::heartbeat_watchdog`; this PR adds the missing
//!   callback hook.
//!
//! What this PR explicitly does NOT ship — see the design doc's
//! "Blockers" section for each:
//!
//! 1. **In-flight Forward handling.** When a worker dies mid-Forward,
//!    rank 0's `forward_one_token_first` is parked in `recv_kind_client`
//!    on the downstream socket. The orchestrator must (a) cancel that
//!    blocked task or (b) wait for its 60 s read timeout to fire — both
//!    are missing. The skeleton's test path uses a synthetic
//!    `Watchdog::Dead` injection that fires when no Forward is in
//!    flight.
//! 2. **Mid-decode KV state.** iter 091's `extract_kv_slab` assumes the
//!    runner is quiesced. Calling it on a runner that just died mid-
//!    Forward (KV half-updated for the in-flight token) returns
//!    inconsistent K vs V slot counts across ranks. The skeleton
//!    `extract_from_survivor` documents this limitation; it does not
//!    fix it. See blocker 2.
//! 3. **Monitor authentication.** Any peer that completes TCP handshake
//!    on the backup's listener could declare itself the backup and
//!    install a forged KV slab. No HMAC, no signature, no orchestrator
//!    token. Same posture as the rest of tahoma's wire protocol;
//!    productization in cascadia-fleet.
//! 4. **Network partition (split brain).** If the orchestrator decides
//!    "primary is dead, spawn backup" but the primary was actually
//!    healthy and the orchestrator's heartbeat link was the broken
//!    one, the backup gets KV state from a surviving rank AND the
//!    primary is still running, producing duplicate Forward responses
//!    upstream. The skeleton has no fencing token mechanism.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tahoma_types::PeerEndpoint;
use tokio::sync::{mpsc, Mutex};

use crate::dist::HeartbeatWatchdog;

/// Snapshot of the FSM state the orchestrator can be in for one
/// (primary, backup) pair. The transitions are linear in v1 — there is
/// no rollback path. A failed migration parks the state machine in
/// `Failed` and the operator must restart the engine.
///
/// ```text
///     Healthy
///        |
///        |  on_watchdog_dead()
///        v
///   DetectedDead
///        |
///        |  spawn_replacement() returns Ok
///        v
///    BackupSpawned
///        |
///        |  extract_kv_from_survivor() returns Ok
///        v
///   KvExtracted
///        |
///        |  install_kv_on_backup() returns Ok
///        v
///   KvInstalled
///        |
///        |  reroute_downstream() returns Ok
///        v
///     Committed
///
/// any step → Failed { stage, reason }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailoverState {
    Healthy,
    DetectedDead {
        consecutive_misses: u32,
        successes_before_death: u64,
    },
    BackupSpawned {
        backup: PeerEndpoint,
        spawned_at: SystemNs,
    },
    KvExtracted {
        layers_extracted: u32,
        bytes: u64,
    },
    KvInstalled {
        layers_installed: u32,
    },
    Committed,
    Failed {
        stage: FailoverStage,
        reason: String,
    },
}

/// Which step of the FSM failed. Useful so a higher-level operator can
/// distinguish "the backup didn't come up" from "we couldn't extract
/// KV" from "the swap committed but no traffic is flowing" — each
/// implies a different remediation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailoverStage {
    SpawnReplacement,
    ExtractKv,
    InstallKv,
    RerouteDownstream,
}

impl fmt::Display for FailoverStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FailoverStage::SpawnReplacement => write!(f, "spawn-replacement"),
            FailoverStage::ExtractKv => write!(f, "extract-kv"),
            FailoverStage::InstallKv => write!(f, "install-kv"),
            FailoverStage::RerouteDownstream => write!(f, "reroute-downstream"),
        }
    }
}

/// Monotonic wall-clock-nanos-since-startup wrapper. Cheaper than
/// chrono / SystemTime + serializable + safe to compare across
/// orchestrator events. Created from `Instant` so the comparisons are
/// monotonic even if the system clock steps backward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemNs(pub u64);

impl SystemNs {
    /// Time since the orchestrator's [`Clock::start`] origin. Tests
    /// inject a fixed origin so the FSM transitions are deterministic.
    pub fn elapsed_since(self, origin: SystemNs) -> u64 {
        self.0.saturating_sub(origin.0)
    }
}

/// Pluggable strategy for actually launching the backup worker
/// process. Production wires this to a re-exec of the worker binary
/// with the dead rank's `--rank`, `--total`, `--model-dir`, `--device`
/// args; tests wire it to a mock that just records the call.
///
/// The trait is async so the production impl can `tokio::spawn` a
/// subprocess and wait for its "listening on :PORT" handshake before
/// returning. The returned `PeerEndpoint` is where the orchestrator
/// connects for the KV install step + the new Forward route.
#[async_trait::async_trait]
pub trait SpawnReplacement: Send + Sync {
    async fn spawn(
        &self,
        dead_primary: &PeerEndpoint,
        rank: u32,
        total: u32,
    ) -> Result<PeerEndpoint, SpawnError>;
}

/// Error from a failed backup-worker spawn. The orchestrator parks in
/// [`FailoverState::Failed`] with this reason on receive.
#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("backup spawn timed out after {0:?}")]
    Timeout(std::time::Duration),
    #[error("backup spawn rejected: {0}")]
    Rejected(String),
    #[error("backup spawn failed: {0}")]
    Other(String),
}

/// External-observable events the orchestrator emits as it advances.
/// Subscribers (CLI status, metrics endpoint, future Dashboard) drain
/// these via a `tokio::sync::mpsc::Receiver` returned at construction
/// time. The orchestrator never blocks on a slow subscriber — drops
/// silently if the channel is full (a follow-up may swap this for a
/// broadcast channel with backpressure).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchestratorEvent {
    /// `on_watchdog_dead` fired. Carries the watchdog snapshot the
    /// orchestrator inspected so a metrics endpoint can correlate
    /// "how many successes did we get before death".
    DeadDetected {
        consecutive_misses: u32,
        successes: u64,
    },
    /// `spawn_replacement` returned Ok with the listed backup
    /// endpoint. The backup is up and listening but holds no KV yet.
    BackupSpawned { backup: PeerEndpoint },
    /// `extract_kv_from_survivor` returned Ok. The slab is in memory
    /// on the orchestrator side, not yet on the backup.
    KvExtracted { layers: u32, bytes: u64 },
    /// `install_kv_on_backup` returned Ok. KV is on the backup but
    /// rank 0 is still routing Forward frames to the (dead) primary.
    KvInstalled { layers: u32 },
    /// `reroute_downstream` returned Ok. From this point on, every
    /// Forward goes to the backup. The state machine is at terminal
    /// `Committed`.
    Committed,
    /// One step failed. State is `Failed { stage, reason }` and the
    /// operator must restart the engine.
    Failed {
        stage: FailoverStage,
        reason: String,
    },
}

/// Receiver side of the orchestrator event channel. Returned from
/// [`Orchestrator::new`] so the caller can wire it to a metrics
/// endpoint, CLI status panel, or test assertion.
pub type EventRx = mpsc::Receiver<OrchestratorEvent>;

/// Bounded sender. 64 is enough for the FSM's lifetime (each state
/// transition emits at most one event, plus the FSM transitions
/// linearly so the upper bound is a single-digit count). Bounded
/// instead of unbounded so a deadlocked subscriber surfaces fast.
const EVENT_CAPACITY: usize = 64;

/// The orchestrator. One instance per (primary, backup) pair.
///
/// **Construction.** Caller hands in the watchdog (shared with iter
/// 094's `run_heartbeat_loop`), the primary's address (for diagnostics
/// and spawn-callback context), the backup's address (for the KV
/// install and routing update), and the `SpawnReplacement` impl.
/// Returns the orchestrator plus an event receiver.
///
/// **Driving the FSM.** iter 094's cadence loop calls
/// [`on_watchdog_dead`] when `HeartbeatOutcome::Dead` fires (this PR
/// adds the wiring; iter 094 only logs). The orchestrator drives the
/// FSM forward async — caller must `tokio::spawn` the
/// [`Orchestrator::run`] task at construction time and keep the
/// returned handle until shutdown.
///
/// **Reset.** v1 has no reset path. Once `run` returns, the
/// orchestrator is one-shot. To survive a second failure on the same
/// rank, the caller constructs a new orchestrator with a new backup.
pub struct Orchestrator {
    state: Mutex<FailoverState>,
    primary: PeerEndpoint,
    /// Address the *user pre-allocated* for the backup, before the
    /// replacement was spawned. Kept on the orchestrator so a follow-
    /// up impl can compare it against the SpawnReplacement-returned
    /// address (a mismatch is a config error worth surfacing). In the
    /// skeleton it's only consulted in `Display` impls + tests.
    #[allow(dead_code)]
    backup_addr: PeerEndpoint,
    rank: u32,
    total: u32,
    watchdog: Arc<Mutex<HeartbeatWatchdog>>,
    spawner: Arc<dyn SpawnReplacement>,
    events: mpsc::Sender<OrchestratorEvent>,
    /// Fires when `on_watchdog_dead` is invoked. The `run` task waits
    /// on this and then walks the FSM. We use a counter instead of a
    /// oneshot so duplicate "Dead" notifications (e.g. a transport
    /// also fires `WireBroken` after the watchdog already escalated)
    /// don't double-drive the FSM.
    trigger: Arc<AtomicU64>,
    start: Instant,
}

impl Orchestrator {
    pub fn new(
        primary: PeerEndpoint,
        backup_addr: PeerEndpoint,
        rank: u32,
        total: u32,
        watchdog: Arc<Mutex<HeartbeatWatchdog>>,
        spawner: Arc<dyn SpawnReplacement>,
    ) -> (Self, EventRx) {
        let (tx, rx) = mpsc::channel(EVENT_CAPACITY);
        let orch = Self {
            state: Mutex::new(FailoverState::Healthy),
            primary,
            backup_addr,
            rank,
            total,
            watchdog,
            spawner,
            events: tx,
            trigger: Arc::new(AtomicU64::new(0)),
            start: Instant::now(),
        };
        (orch, rx)
    }

    /// Snapshot the current state. Lock-bounded; for diagnostics only.
    pub async fn state(&self) -> FailoverState {
        self.state.lock().await.clone()
    }

    /// Trigger handle for iter 094's `run_heartbeat_loop`. The cadence
    /// loop calls this on `HeartbeatOutcome::Dead` / `WireBroken`. The
    /// orchestrator's `run` task observes the bumped counter and
    /// advances the FSM.
    pub fn trigger_handle(&self) -> Arc<AtomicU64> {
        self.trigger.clone()
    }

    /// Synchronous wrapper iter 094 can call without holding async
    /// context. Bumps the trigger counter; the orchestrator's `run`
    /// task picks it up on its next poll. Idempotent — bumping twice
    /// for the same death event just means `run` re-reads the same
    /// terminal state.
    pub fn on_watchdog_dead(trigger: &AtomicU64) {
        trigger.fetch_add(1, Ordering::Release);
    }

    /// Drive the FSM. Spawn this on a `tokio::Runtime` after
    /// constructing the orchestrator. Returns when the FSM hits a
    /// terminal state (`Committed` or `Failed`).
    pub async fn run(&self) -> FailoverState {
        // Phase 1: wait for the trigger to bump.
        self.wait_for_trigger().await;

        // Phase 2: snapshot the watchdog, transition to DetectedDead.
        let (misses, successes) = self.snapshot_watchdog().await;
        self.transition(FailoverState::DetectedDead {
            consecutive_misses: misses,
            successes_before_death: successes,
        })
        .await;
        self.emit(OrchestratorEvent::DeadDetected {
            consecutive_misses: misses,
            successes,
        })
        .await;

        // Phase 3: spawn the replacement.
        let backup = match self
            .spawner
            .spawn(&self.primary, self.rank, self.total)
            .await
        {
            Ok(addr) => addr,
            Err(e) => {
                let reason = format!("{e}");
                self.fail(FailoverStage::SpawnReplacement, reason.clone())
                    .await;
                return FailoverState::Failed {
                    stage: FailoverStage::SpawnReplacement,
                    reason,
                };
            }
        };
        self.transition(FailoverState::BackupSpawned {
            backup: backup.clone(),
            spawned_at: SystemNs(self.start.elapsed().as_nanos() as u64),
        })
        .await;
        self.emit(OrchestratorEvent::BackupSpawned {
            backup: backup.clone(),
        })
        .await;

        // Phase 4: extract KV from a surviving rank.
        //
        // SKELETON: in v1 we have neither a surviving-rank registry
        // nor a connection back to the survivor. The full impl walks
        // tahoma-topology for a rank that mirrors the dead range AND
        // is currently quiesced, then issues `extract_kv_slab` over
        // wire. For now we emit a `Failed { ExtractKv }` event so the
        // FSM advances visibly past this point — a real consumer can
        // see "we got to extract but it's not wired" without grepping
        // logs. iter 091's `KvMigration` FrameKind + Runner API are
        // the building blocks; composing them is the next iter.
        match self.extract_from_survivor().await {
            Ok((layers, bytes)) => {
                self.transition(FailoverState::KvExtracted {
                    layers_extracted: layers,
                    bytes,
                })
                .await;
                self.emit(OrchestratorEvent::KvExtracted { layers, bytes })
                    .await;

                // Phase 5: install on backup. Same skeleton caveat —
                // depends on iter 091's `KvMigration` send helpers
                // being on the same branch (iter 091 + iter 094
                // diverged from iter 010 independently). Composing
                // them is the iter 097 merge.
                match self.install_on_backup(&backup, layers, bytes).await {
                    Ok(installed) => {
                        self.transition(FailoverState::KvInstalled {
                            layers_installed: installed,
                        })
                        .await;
                        self.emit(OrchestratorEvent::KvInstalled { layers: installed })
                            .await;

                        // Phase 6: reroute. Skeleton — requires a hook
                        // into rank 0's `transport.downstream` cell.
                        match self.reroute_downstream(&backup).await {
                            Ok(_) => {
                                self.transition(FailoverState::Committed).await;
                                self.emit(OrchestratorEvent::Committed).await;
                                FailoverState::Committed
                            }
                            Err(e) => {
                                let reason = e;
                                self.fail(FailoverStage::RerouteDownstream, reason.clone())
                                    .await;
                                FailoverState::Failed {
                                    stage: FailoverStage::RerouteDownstream,
                                    reason,
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let reason = e;
                        self.fail(FailoverStage::InstallKv, reason.clone()).await;
                        FailoverState::Failed {
                            stage: FailoverStage::InstallKv,
                            reason,
                        }
                    }
                }
            }
            Err(e) => {
                let reason = e;
                self.fail(FailoverStage::ExtractKv, reason.clone()).await;
                FailoverState::Failed {
                    stage: FailoverStage::ExtractKv,
                    reason,
                }
            }
        }
    }

    /// Wait until the trigger counter goes above zero. Polled (not
    /// `Notify`) so a death signal that arrived before `run` was
    /// spawned is not lost — the counter is checked once before the
    /// first sleep.
    async fn wait_for_trigger(&self) {
        // Fast path: trigger already fired.
        if self.trigger.load(Ordering::Acquire) > 0 {
            return;
        }
        // Slow path: poll. 50 ms is fine — orchestrator wakeup
        // latency is dwarfed by the spawn time (seconds).
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            if self.trigger.load(Ordering::Acquire) > 0 {
                return;
            }
        }
    }

    async fn snapshot_watchdog(&self) -> (u32, u64) {
        let wg = self.watchdog.lock().await;
        (wg.consecutive_misses(), wg.successes())
    }

    async fn transition(&self, new: FailoverState) {
        *self.state.lock().await = new;
    }

    async fn emit(&self, ev: OrchestratorEvent) {
        // `try_send` drops if the receiver is slow. Acceptable for a
        // diagnostic stream — and a `send().await` here would risk
        // wedging the FSM behind a dead subscriber.
        let _ = self.events.try_send(ev);
    }

    async fn fail(&self, stage: FailoverStage, reason: String) {
        self.transition(FailoverState::Failed {
            stage,
            reason: reason.clone(),
        })
        .await;
        self.emit(OrchestratorEvent::Failed { stage, reason }).await;
    }

    /// SKELETON. Extract KV from a survivor rank.
    ///
    /// Production impl:
    /// 1. Walk `tahoma-topology` for a rank that mirrors the dead
    ///    rank's layer range AND has not also died.
    /// 2. Send a `Quiesce` frame so the survivor stops accepting new
    ///    Forward frames (blocker 1: quiesce protocol does not yet
    ///    exist — see design doc).
    /// 3. Call `Runner::extract_kv_slab(layer_start, layer_end)` over
    ///    wire (iter 091 ships the local-side API; the wire helper
    ///    `send_kv_migration` flows the other direction — we need an
    ///    `extract_kv_request` frame).
    /// 4. Return the slab bytes + the layer count + the bytes count
    ///    for the event.
    ///
    /// For now: returns `Err("extract-kv not yet wired")` so the FSM
    /// advances to `Failed { ExtractKv }` visibly.
    async fn extract_from_survivor(&self) -> Result<(u32, u64), String> {
        Err(
            "extract_from_survivor: iter 091 KvMigration wire frame + survivor-rank registry \
             not yet composed (see docs/architecture/failover-orchestrator.md blocker 2)"
                .into(),
        )
    }

    /// SKELETON. Install the slab on the freshly-spawned backup.
    ///
    /// Production impl:
    /// 1. Open a `tahoma-transport::ActivationClient` to the backup.
    /// 2. Issue `dist::send_kv_migration` per layer (iter 091's wire
    ///    helper).
    /// 3. Wait for `KvMigrationAck` per layer (blocker 1 — the ack
    ///    frame doesn't yet exist).
    /// 4. Return the install count for the event.
    async fn install_on_backup(
        &self,
        _backup: &PeerEndpoint,
        _layers: u32,
        _bytes: u64,
    ) -> Result<u32, String> {
        Err("install_on_backup: skeleton — see failover-orchestrator.md blocker 2".into())
    }

    /// SKELETON. Re-point rank 0's downstream socket to the backup.
    ///
    /// Production impl needs a mutable cell on `SparseMoEEngine`
    /// (currently `self.transport.downstream` is `Option<Arc<Mutex<...>>>`
    /// — re-routing means swapping the inner `ActivationClient` while
    /// the FSM holds the lock so no Forward goes to the dead primary
    /// after the swap).
    async fn reroute_downstream(&self, _backup: &PeerEndpoint) -> Result<(), String> {
        Err("reroute_downstream: skeleton — needs a mutable rank-0 routing hook".into())
    }
}

// ----------------------------------------------------------------------
// Test helpers
// ----------------------------------------------------------------------

/// Mock `SpawnReplacement` that records each call and returns a fixed
/// backup address. Public so integration tests in other crates can
/// reuse it.
pub struct MockSpawner {
    pub backup: PeerEndpoint,
    pub calls: Mutex<Vec<MockSpawnCall>>,
    pub force_error: Mutex<Option<SpawnError>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockSpawnCall {
    pub dead_primary: PeerEndpoint,
    pub rank: u32,
    pub total: u32,
}

impl MockSpawner {
    pub fn new(backup: PeerEndpoint) -> Self {
        Self {
            backup,
            calls: Mutex::new(Vec::new()),
            force_error: Mutex::new(None),
        }
    }

    pub fn with_error(backup: PeerEndpoint, err: SpawnError) -> Self {
        Self {
            backup,
            calls: Mutex::new(Vec::new()),
            force_error: Mutex::new(Some(err)),
        }
    }

    pub async fn call_count(&self) -> usize {
        self.calls.lock().await.len()
    }

    pub async fn last_call(&self) -> Option<MockSpawnCall> {
        self.calls.lock().await.last().cloned()
    }
}

#[async_trait::async_trait]
impl SpawnReplacement for MockSpawner {
    async fn spawn(
        &self,
        dead_primary: &PeerEndpoint,
        rank: u32,
        total: u32,
    ) -> Result<PeerEndpoint, SpawnError> {
        self.calls.lock().await.push(MockSpawnCall {
            dead_primary: dead_primary.clone(),
            rank,
            total,
        });
        // Take the error if one was queued — single-use, so a later
        // call to the same mock spawner succeeds.
        if let Some(e) = self.force_error.lock().await.take() {
            return Err(e);
        }
        Ok(self.backup.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn endpoint(host: &str, port: u16) -> PeerEndpoint {
        PeerEndpoint::new(host, port)
    }

    fn watchdog_with(misses: u32, successes: u64) -> Arc<Mutex<HeartbeatWatchdog>> {
        let mut w = HeartbeatWatchdog::default();
        for _ in 0..successes {
            w.record_success();
        }
        for _ in 0..misses {
            // record_miss returns true at threshold but we don't care
            // here — we just want the counter populated for the FSM.
            let _ = w.record_miss();
        }
        Arc::new(Mutex::new(w))
    }

    #[tokio::test]
    async fn watchdog_dead_fires_spawn_replacement_callback() {
        // The synthetic Watchdog::Dead path the task asks for. We
        // construct the orchestrator, spawn its run task, fire the
        // dead trigger, and assert the spawn callback ran exactly
        // once with the dead primary's address.
        let primary = endpoint("primary.lan", 9001);
        let backup = endpoint("backup.lan", 9002);
        let watchdog = watchdog_with(2, 17);
        let spawner = Arc::new(MockSpawner::new(backup.clone()));
        let (orch, mut events) = Orchestrator::new(
            primary.clone(),
            backup.clone(),
            1,
            2,
            watchdog,
            spawner.clone(),
        );

        let orch_arc = Arc::new(orch);
        let trigger = orch_arc.trigger_handle();
        let orch_for_task = orch_arc.clone();
        let run_task = tokio::spawn(async move { orch_for_task.run().await });

        // Fire the synthetic Dead.
        Orchestrator::on_watchdog_dead(&trigger);

        // The FSM should advance: DeadDetected → BackupSpawned →
        // (then KV failure in skeleton). Drain the first two events.
        let dead = tokio::time::timeout(Duration::from_secs(2), events.recv())
            .await
            .expect("DeadDetected event timed out")
            .expect("event channel closed");
        match dead {
            OrchestratorEvent::DeadDetected {
                consecutive_misses,
                successes,
            } => {
                assert_eq!(consecutive_misses, 2);
                assert_eq!(successes, 17);
            }
            other => panic!("expected DeadDetected, got {other:?}"),
        }

        let spawned = tokio::time::timeout(Duration::from_secs(2), events.recv())
            .await
            .expect("BackupSpawned event timed out")
            .expect("event channel closed");
        match spawned {
            OrchestratorEvent::BackupSpawned { backup: b } => assert_eq!(b, backup),
            other => panic!("expected BackupSpawned, got {other:?}"),
        }

        // Verify the spawn callback fired exactly once with the dead
        // primary's address + the right rank.
        assert_eq!(spawner.call_count().await, 1);
        let call = spawner.last_call().await.expect("call recorded");
        assert_eq!(call.dead_primary, primary);
        assert_eq!(call.rank, 1);
        assert_eq!(call.total, 2);

        // Skeleton: the FSM should land in Failed { ExtractKv } since
        // extract_from_survivor is unwired.
        let final_state = tokio::time::timeout(Duration::from_secs(2), run_task)
            .await
            .expect("run task didn't finish")
            .expect("run task panicked");
        match final_state {
            FailoverState::Failed { stage, .. } => {
                assert_eq!(stage, FailoverStage::ExtractKv);
            }
            other => panic!("expected Failed at ExtractKv, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn spawn_callback_failure_halts_fsm_at_spawn_stage() {
        let primary = endpoint("primary.lan", 9001);
        let backup = endpoint("backup.lan", 9002);
        let watchdog = watchdog_with(2, 0);
        let spawner = Arc::new(MockSpawner::with_error(
            backup.clone(),
            SpawnError::Timeout(Duration::from_millis(500)),
        ));
        let (orch, mut events) =
            Orchestrator::new(primary, backup, 1, 2, watchdog, spawner.clone());

        let orch_arc = Arc::new(orch);
        let trigger = orch_arc.trigger_handle();
        let orch_for_task = orch_arc.clone();
        let run_task = tokio::spawn(async move { orch_for_task.run().await });
        Orchestrator::on_watchdog_dead(&trigger);

        // DeadDetected then Failed { SpawnReplacement }.
        let _ = tokio::time::timeout(Duration::from_secs(2), events.recv())
            .await
            .expect("DeadDetected event timed out");
        let failed = tokio::time::timeout(Duration::from_secs(2), events.recv())
            .await
            .expect("Failed event timed out")
            .expect("event channel closed");
        match failed {
            OrchestratorEvent::Failed { stage, .. } => {
                assert_eq!(stage, FailoverStage::SpawnReplacement);
            }
            other => panic!("expected Failed{{SpawnReplacement}}, got {other:?}"),
        }

        let final_state = tokio::time::timeout(Duration::from_secs(2), run_task)
            .await
            .expect("run task didn't finish")
            .expect("run task panicked");
        assert!(matches!(
            final_state,
            FailoverState::Failed {
                stage: FailoverStage::SpawnReplacement,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn trigger_fired_before_run_is_not_lost() {
        // Regression guard: a death signal that arrives before the
        // run task starts polling must still drive the FSM. The fast-
        // path check at the top of `wait_for_trigger` handles this.
        let primary = endpoint("primary.lan", 9001);
        let backup = endpoint("backup.lan", 9002);
        let watchdog = watchdog_with(2, 1);
        let spawner = Arc::new(MockSpawner::new(backup.clone()));
        let (orch, mut events) =
            Orchestrator::new(primary, backup, 0, 2, watchdog, spawner.clone());

        let orch_arc = Arc::new(orch);
        let trigger = orch_arc.trigger_handle();
        // Fire BEFORE spawning the run task.
        Orchestrator::on_watchdog_dead(&trigger);

        let orch_for_task = orch_arc.clone();
        let run_task = tokio::spawn(async move { orch_for_task.run().await });

        // The first event should still arrive.
        let first = tokio::time::timeout(Duration::from_secs(2), events.recv())
            .await
            .expect("DeadDetected event timed out")
            .expect("event channel closed");
        assert!(matches!(first, OrchestratorEvent::DeadDetected { .. }));
        assert_eq!(spawner.call_count().await, 1);
        let _ = run_task.await;
    }

    #[test]
    fn failover_stage_display_strings_are_stable() {
        // Diagnostics depend on these strings — pin them.
        assert_eq!(
            FailoverStage::SpawnReplacement.to_string(),
            "spawn-replacement"
        );
        assert_eq!(FailoverStage::ExtractKv.to_string(), "extract-kv");
        assert_eq!(FailoverStage::InstallKv.to_string(), "install-kv");
        assert_eq!(
            FailoverStage::RerouteDownstream.to_string(),
            "reroute-downstream"
        );
    }

    #[test]
    fn system_ns_elapsed_is_saturating() {
        // If a later snapshot somehow precedes the origin (clock skew
        // in test fakes), `elapsed_since` returns 0 rather than
        // wrapping under.
        let origin = SystemNs(1000);
        let later = SystemNs(500);
        assert_eq!(later.elapsed_since(origin), 0);
        let later2 = SystemNs(1500);
        assert_eq!(later2.elapsed_since(origin), 500);
    }

    #[test]
    fn duplicate_on_watchdog_dead_calls_do_not_panic() {
        // The cadence loop in iter 094 may emit both `Dead` and
        // `WireBroken` for the same death event (the transport
        // sometimes fails its `recv_raw` after the watchdog already
        // tripped). The trigger handle is a counter — multiple bumps
        // are idempotent w.r.t. the FSM, which only checks `> 0`.
        let trigger = Arc::new(AtomicU64::new(0));
        Orchestrator::on_watchdog_dead(&trigger);
        Orchestrator::on_watchdog_dead(&trigger);
        Orchestrator::on_watchdog_dead(&trigger);
        assert_eq!(trigger.load(Ordering::Acquire), 3);
    }
}
