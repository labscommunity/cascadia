# Failover orchestrator

**Status:** (B) skeleton + design doc (iter 096). Composes iter 091
(`perf/kv-migration-091` — KV migration wire frame + Runner extract/
install APIs), iter 092 (`perf/heartbeat-recovery-092` — heartbeat
wire + watchdog), and iter 094 (`perf/heartbeat-driver-094` — rank-0
cadence loop). End-to-end recovery is multi-week and not in this PR.

**Track:** sparse-MoE pipeline-parallel.

## Goal

Detect a dead pipeline worker, spawn a backup, migrate KV state from
a surviving rank, and re-route Forward traffic — all without
restarting the user's generation. Together with iters 091/092/094 this
closes the "what happens when one of N AI PCs in the pipeline dies
mid-decode" gap.

Killer-demo scenario: K2.6 sharded across `alpha` / `beta` / `charlie`.
The user submits a 256-token completion. Mid-decode, `beta`'s AI PC
thermally throttles and the worker process panics. iter 094's cadence
loop on `alpha` (rank 0) declares `beta` dead within ≤ 4 s. iter 096's
orchestrator spawns a replacement worker on a hot-spare AI PC,
migrates the KV slab for `beta`'s layer range from `charlie` (which
mirrors `beta`'s layers for replication, *not* in v1), installs the
slab on the replacement, and re-points rank 0's downstream socket.
The user sees a ~5 s pause in the SSE stream and the next token
arrives normally.

## What this PR ships

**(B) skeleton + design doc.** Specifically:

| Component | File | What |
|---|---|---|
| `Orchestrator` struct | `crates/tahoma-engine-sparse-moe/src/orchestrator.rs` | Owns watchdog handle + backup endpoint + spawn callback; drives a 7-state FSM |
| `FailoverState` enum | same | `Healthy` → `DetectedDead` → `BackupSpawned` → `KvExtracted` → `KvInstalled` → `Committed`; off-path: `Failed { stage }` |
| `SpawnReplacement` trait | same | Async pluggable strategy; production wires to subprocess re-exec, tests wire to `MockSpawner` |
| `OrchestratorEvent` stream | same | `tokio::sync::mpsc::Receiver<OrchestratorEvent>` returned at construction; one event per FSM transition |
| `on_watchdog_dead` trigger | same | The hook iter 094's cadence loop is supposed to call when `HeartbeatOutcome::Dead` fires — iter 094 only logs |
| `MockSpawner` (test-only export) | same | Records every call; configurable to force a `SpawnError` on the first attempt |
| 6 unit tests | same | Synthetic `Watchdog::Dead` end-to-end through FSM (passes); spawn-callback failure halts at `SpawnReplacement`; trigger-before-run is not lost; duplicate dead notifications idempotent; `FailoverStage::Display` strings stable; `SystemNs` saturating arithmetic |
| Design doc | `docs/architecture/failover-orchestrator.md` | (this file) |

## What this PR explicitly does NOT ship

These are the items the task spec asked me to document as blockers.
Each is honest about why it's deferred.

### 1. In-flight Forward handling

**Problem.** When the primary dies, rank 0's
`forward_one_token_first` is parked inside
`block_on(recv_kind_client(&downstream))` on the (now-dead)
downstream socket. The watchdog detects the death within ≤ 4 s, but
the Forward task is *still blocked* — its `recv_raw(4)` call won't
return until `tahoma-transport`'s `DEFAULT_TIMEOUT = 60 s` fires. The
orchestrator's `reroute_downstream` step needs exclusive access to
`SparseMoEEngine::transport.downstream` to swap the socket; the
Forward task is holding the mutex behind that `Arc<TokioMutex>`.

**What's needed.**

- A "task cancel" signal the Forward task awaits *alongside* the
  recv. Rust idiom is `tokio::select! { recv_kind_client(...) =
  ..., cancel.notified() = ... }`. `forward_one_token_first` is
  currently blocking-bridge code (`block_on`), so the cancel has to
  cross the sync/async seam.
- A "Forward retry token" so that once the swap commits, the rank-0
  driver can re-issue the in-flight Forward at the post-migration
  KV state. Today there is no replay mechanism — a dropped Forward
  is a dropped token, and tokens are positional in the decode
  loop, so the whole generation must be cancelled.

**Skeleton scope.** Tests inject `Watchdog::Dead` at a moment when no
Forward is in flight (between decode steps). Production traffic will
trip blocker 1 on every real failure until this is fixed.

### 2. Mid-decode KV state

**Problem.** iter 091's `Runner::extract_kv_slab` is documented as
"caller must serialize against decode" — i.e. the runner is
quiesced. In the failover scenario, the survivor rank is *currently*
quiesced (rank 0 stopped issuing Forward because the dead primary
hangs the pipeline), so extract is fine in *that* direction. The
real problem is shape-symmetric: the *dead* rank's last action might
have been "K updated, V update interrupted by SIGSEGV". If a hot-
spare rank with a stale copy of that range is what we promote, its
K and V slot counts are off by one relative to the upstream history.
Decode resumes producing garbage tokens with no visible error.

**What's needed.**

- A "KV slot generation counter" per layer per rank. The orchestrator
  refuses to install a slab whose generation counter is < the
  upstream history length (i.e. the survivor is behind).
- A "torn-write" detector: extract reads both K and V; if the slot
  counts differ for any layer, the survivor is unreliable for this
  range and the orchestrator must fall back to "drop the request,
  surface a 503-equivalent to the user".

**Skeleton scope.** `extract_from_survivor()` returns
`Err("not yet wired")`. The FSM advances visibly to
`Failed { ExtractKv }` so an operator sees exactly where the missing
piece is.

### 3. Monitor authentication

**Problem.** The orchestrator's "spawn the backup, install KV on it"
flow uses the same un-authenticated TCP wire as every other tahoma
frame. A malicious peer that completes the TCP handshake on the
backup's listener port can:

- Send a forged `KvMigration` frame with a different layer range
  than the one the orchestrator intended, corrupting the backup's KV.
- Claim to be "the backup" to the orchestrator (by listening on the
  expected port before the real backup spawns).
- DoS the orchestrator by reply-flooding the routing-update channel.

**What's needed.**

- An HMAC tag on every `KvMigration` frame, keyed off an
  orchestrator-distributed nonce.
- TLS on the activation socket. Out of scope for the OSS skeleton;
  productization lives in the cascadia-fleet track per the locked
  design decision.

**Skeleton scope.** Same posture as the rest of tahoma's wire
protocol today (heartbeat, KV migration, Forward, etc.) — trusted
LAN only.

### 4. Network partition (split brain)

**Problem.** The orchestrator decides "primary is dead" based on
*its own* heartbeat link to the primary. If that link is broken but
the primary is healthy and still serving upstream Forward frames
(which arrived from rank 0 over a different physical path), then:

1. Orchestrator spawns a backup.
2. Backup installs a KV slab.
3. Orchestrator re-points rank 0's downstream to the backup.
4. Meanwhile, the *original* primary is still alive, still has the
   old socket open to rank 0, and may still be in the middle of
   responding to the last Forward.
5. Rank 0 now has TWO sockets to two "primaries": one to the
   original (now-zombie) primary and one to the new backup. Both
   may emit Token frames for the same Forward; rank 0's
   `recv_kind_client` reads whichever arrives first, and the model
   produces a non-deterministic mix of two parallel decode threads.

**What's needed.**

- A fencing token. The orchestrator's spawn step issues a monotonic
  "generation N+1" token; the backup tags every Token frame with
  this. Rank 0 ignores any Token whose generation is < the current
  generation. The zombie primary's Tokens (generation N) are
  silently dropped.
- A "primary, terminate yourself" signal. The orchestrator's spawn
  step also sends a `Suicide` frame to the (presumed-dead) primary's
  address. If the primary is actually alive, it sees the frame and
  shuts down. If it's dead the frame is dropped — same outcome.
- A separate consensus layer to decide "is the primary actually
  dead, or is the orchestrator's view of the network broken?". This
  is the *hard* part — the cascadia-fleet track's productization
  story includes this.

**Skeleton scope.** None. Documented as a known unsafe behavior.

## State machine

```text
              Healthy
                 |
                 |  on_watchdog_dead()
                 |  (iter 094 cadence loop calls this on threshold cross)
                 v
            DetectedDead { consec_misses, successes_before_death }
                 |
                 |  SpawnReplacement::spawn() → Ok(backup_addr)
                 v
           BackupSpawned { backup, spawned_at }
                 |
                 |  extract_from_survivor() → Ok((layers, bytes))
                 v
             KvExtracted { layers, bytes }
                 |
                 |  install_on_backup() → Ok(installed)
                 v
            KvInstalled { layers_installed }
                 |
                 |  reroute_downstream() → Ok(())
                 v
              Committed

  any step's Err(reason) →  Failed { stage, reason }
```

Three observations:

- **One-shot.** v1 has no reset path. The orchestrator survives one
  failover; a second failure on the same rank needs a new
  orchestrator instance. The follow-up is a "rotating backup pool"
  that pre-allocates N backups per rank.
- **Linear.** No conditional branches inside the FSM. Each step
  either advances or terminally fails — no retry, no fallback, no
  rollback. Acceptable because the failure modes are observable and
  the operator can restart; production hardening adds a retry loop
  per stage.
- **Trigger-counter, not oneshot.** `on_watchdog_dead` bumps an
  `AtomicU64` rather than firing a `oneshot::Sender`. Two reasons:
  (a) iter 094's cadence loop may emit both `Dead` and `WireBroken`
  for the same death event (transport sometimes fails recv after
  the watchdog already tripped); the counter de-duplicates by being
  idempotent w.r.t. "is it > 0?". (b) A trigger fired before
  `Orchestrator::run` is spawned is not lost — the run task checks
  the counter once before its first sleep.

## Public API

```rust
// Construction. Returns the orchestrator + an event receiver.
let (orch, mut events) = Orchestrator::new(
    primary,                  // PeerEndpoint
    backup_addr,              // PeerEndpoint
    rank, total,              // u32, u32
    watchdog,                 // Arc<Mutex<HeartbeatWatchdog>> (from iter 094)
    spawner,                  // Arc<dyn SpawnReplacement>
);

// Driving. tokio::spawn this; it returns when the FSM hits a
// terminal state (Committed or Failed).
let final_state = tokio::spawn(orch.run());

// Triggering. iter 094's cadence loop calls this synchronously
// from the `match outcome { Dead => ... }` arm.
let trigger = orch.trigger_handle(); // Arc<AtomicU64>
Orchestrator::on_watchdog_dead(&trigger);

// Observing. Drain events as the FSM advances.
while let Some(ev) = events.recv().await {
    match ev {
        OrchestratorEvent::DeadDetected { .. } => /* log + metric */,
        OrchestratorEvent::BackupSpawned { backup } => /* log + metric */,
        OrchestratorEvent::KvExtracted { layers, bytes } => /* log */,
        OrchestratorEvent::KvInstalled { layers } => /* log */,
        OrchestratorEvent::Committed => /* success metric */,
        OrchestratorEvent::Failed { stage, reason } => /* alarm */,
    }
}
```

## Wiring to iter 094

iter 094's `Builder::build` already spawns the cadence loop and
exposes the `HeartbeatWatchdog` through
`SparseMoEEngine::heartbeat_watchdog`. The hook iter 094 marked
`FOLLOW-UP-orchestrator` lives in `run_heartbeat_loop`'s terminal
match:

```rust
// In crates/tahoma-engine-sparse-moe/src/engine.rs (iter 094).
match final_outcome {
    HeartbeatOutcome::Dead => {
        tracing::error!(
            "heartbeat: pipeline worker declared DEAD by watchdog \
             (consecutive misses crossed threshold); \
             upper-layer recovery is not yet wired (FOLLOW-UP-orchestrator)"
        );
    }
    // ... other arms
}
```

iter 096 changes this to:

```rust
match final_outcome {
    HeartbeatOutcome::Dead => {
        if let Some(trigger) = orchestrator_trigger.as_ref() {
            Orchestrator::on_watchdog_dead(trigger);
        }
        tracing::error!("heartbeat: pipeline worker declared DEAD; orchestrator notified");
    }
    HeartbeatOutcome::WireBroken => {
        // Same — link death is treated as worker death by the
        // orchestrator (it has no way to distinguish them post-hoc).
        if let Some(trigger) = orchestrator_trigger.as_ref() {
            Orchestrator::on_watchdog_dead(trigger);
        }
        tracing::error!("heartbeat: downstream wire broken; orchestrator notified");
    }
    HeartbeatOutcome::Alive | HeartbeatOutcome::Missed => { /* keep going */ }
}
```

The wiring change is deliberately *not* in this PR — it depends on
`SparseMoEBuilderConfig` learning about a backup endpoint + spawn
callback, which is a separate concern. Filed as the iter 097 merge.

## Test coverage

All 6 orchestrator unit tests pass (`cargo test -p
tahoma-engine-sparse-moe --lib orchestrator`):

| Test | What |
|---|---|
| `watchdog_dead_fires_spawn_replacement_callback` | Synthetic `Watchdog::Dead` → spawn-callback fires with correct `(primary, rank, total)`; FSM advances to `BackupSpawned`; lands in `Failed { ExtractKv }` (skeleton) |
| `spawn_callback_failure_halts_fsm_at_spawn_stage` | `SpawnError::Timeout` from the spawn callback → FSM lands in `Failed { SpawnReplacement }` |
| `trigger_fired_before_run_is_not_lost` | `on_watchdog_dead` called *before* `Orchestrator::run` is spawned — the FSM still advances |
| `duplicate_on_watchdog_dead_calls_do_not_panic` | Multiple bumps to the trigger counter are idempotent w.r.t. the FSM |
| `failover_stage_display_strings_are_stable` | Pins the `Display` strings on `FailoverStage` (used in event reasons + logs) |
| `system_ns_elapsed_is_saturating` | `SystemNs::elapsed_since` underflow-safe under clock skew |

Out of scope (deferred to follow-ups):

- End-to-end test under real worker death. Wants a subprocess spawn
  and `kill -SEGV` on the local AI PC fleet; same gap as iter 094's
  cadence-loop test matrix.
- Race between cadence-loop firing and orchestrator startup. The
  trigger counter handles the happy case; what if the cadence loop
  fires *while* the orchestrator's `run` is mid-`spawn` callback?
  Skeleton test relies on `tokio::spawn` ordering; production needs
  an explicit barrier.
- Concurrent failover (rank 1 AND rank 2 die at the same time).
  Skeleton has one orchestrator per dead rank — two failovers
  contend for the same `transport.downstream` mutex on rank 0.

## Next-step ordering

1. **Compose iter 091 + iter 094 onto one base.** Both are
   currently independent branches off PR #10 (`208104e`). Need a
   merge commit that brings `FrameKind::KvMigration` + Runner's
   `extract_kv_slab` / `install_kv_slab` into the same tree as the
   cadence loop + watchdog. ~30 LOC of conflict resolution; the
   real work is in deciding the FrameKind code-byte ordering
   (`0x30` for KvMigration was claimed in iter 091; heartbeat used
   `0x40 / 0x41` in iter 092 — they don't collide).
2. **Wire `on_watchdog_dead` from the cadence loop.** Replace the
   `tracing::error!` stub with a `trigger` lookup. ~10 LOC.
3. **Implement `extract_from_survivor`.** Open a fresh
   `ActivationClient` to the survivor (requires topology
   awareness — `SparseMoEBuilderConfig` needs a "peers I can ask
   for KV" list), issue `KvMigration` over wire, accumulate slabs.
   Composes with the quiesce protocol (iter 091 blocker 2).
4. **Implement `install_on_backup`.** Symmetric to step 3 but
   client-side; per-layer `send_kv_migration` + (eventual) ack.
5. **Implement `reroute_downstream`.** Needs a mutable hook into
   `SparseMoEEngine::transport.downstream`. The `Arc<TokioMutex>`
   already exists; the orchestrator can `*guard = new_client` once
   it holds the lock — but the Forward task (blocker 1) is
   currently holding it.
6. **Fencing token (blocker 4).** Add a `generation: u32` field to
   `FrameKind::Forward` (bump to `0x53_4D_45_03`); rank 0 rejects
   Token frames whose generation < current.

Refs: PR #10 (pipeline-parallel inference + Rust shells), iter 091
(`perf/kv-migration-091` — KV migration wire frame + Runner APIs),
iter 092 (`perf/heartbeat-recovery-092` — heartbeat wire +
watchdog), iter 094 (`perf/heartbeat-driver-094` — cadence loop).
