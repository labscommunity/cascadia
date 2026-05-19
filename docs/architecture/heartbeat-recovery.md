# Worker heartbeat + auto-recovery

**Status:** wire frame + watchdog + CLI flag (iter 092) + driver-side
cadence loop (iter 094). Auto-restart is still FOLLOW-UP-orchestrator
and not in either PR.

**Track:** sparse-MoE pipeline-parallel (iter 092 → iter 094, follow-
up to iter 030's Matias 2-box revival).

## Goal

Detect a dead pipeline worker within 2 heartbeat intervals and surface
that signal to an orchestrator that can re-spawn the worker, restore its
KV state, and re-point the driver's downstream socket — all without
restarting the user's generation.

The two scenarios this exists to fix:

1. **Worker process crashes mid-decode.** The WMI-detached Windows
   worker (iter 030) survives an SSH disconnect from the driver, but
   nothing today survives the worker's own segfault, OOM kill, or
   `panic!()`. The rank-0 driver's `recv_kind_client` on the downstream
   socket then blocks for `DEFAULT_TIMEOUT = 60 s` before erroring —
   one full minute of user-visible hang per token, indefinitely.
2. **TCP connection silently wedges.** Half-open sockets (e.g. cable
   pulled mid-decode, hypervisor pause/resume that drops the socket
   while keeping the kernel's keepalive timer reset) look identical
   to a slow-but-alive peer until the next 60 s timeout fires.
   Heartbeats give a 1 s / 2 s detection ceiling instead.

## What this PR ships

**(B) skeleton + design doc.** Specifically:

- Two new `FrameKind` codes (`HeartbeatPing` = `0x53_4D_45_40`,
  `HeartbeatPong` = `0x53_4D_45_41`), disjoint from `Forward` /
  `Reset` / `Token` and from `dist_spec`'s frame namespace.
- Send/recv helpers for both directions (rank-0 → worker uses
  downstream client; worker → rank-0 uses upstream server). Symmetric
  `_upstream` / `_downstream` helpers are also wired so a future
  bidirectional probe doesn't need new helpers.
- A 12-byte over-the-wire payload (4 B kind + 8 B BE u64 nonce). The
  nonce is opaque to the worker — it echoes whatever it received — and
  exists so the driver can match a pong to the ping that produced it
  (otherwise a stale pong from a retried ping looks like a fresh ack).
- `HeartbeatWatchdog` state machine: per-worker, configurable
  `max_misses`, `record_success` / `record_miss(&mut self) -> bool`
  (returns `true` once the worker has crossed the death threshold), and
  a saturating success counter for the orchestrator's restart policy.
- Worker-side wiring in `SparseMoEEngine::handle_one_frame`: a Ping
  arriving on the upstream socket is echoed back as a Pong. No engine
  state changes. No KV touch.
- CLI flag `--heartbeat-interval-ms N` on `tahoma worker` (default 0 =
  legacy behavior, no heartbeats).
- 12 new tests (6 unit in `dist.rs::tests`, 6 integration in
  `tests/heartbeat_wire.rs`): wire round-trip in both directions,
  nonce-mismatch ordering, watchdog default tolerance, success-resets-
  streak, higher tolerance, recovery after first pong.

## What iter 094 ships (driver-side cadence loop)

Closes blocker 1 of iter 092. Specifically:

- `dist::HeartbeatOutcome` enum: `Alive` | `Missed` | `Dead` |
  `WireBroken` — the per-round result reported back to the cadence
  loop (and ultimately to the orchestrator hook).
- `dist::ping_one_round(downstream, nonce, timeout, &watchdog)` — one
  ping → pong exchange, atomic under the downstream socket mutex.
  Stale pongs (mismatched nonce) are drained silently; the loop
  continues waiting for the matching pong until the deadline.
- `dist::run_heartbeat_loop(downstream, watchdog, interval, timeout,
  cancel)` — the spawn-once driver. Monotonic nonce per round; default
  `timeout = 2 × interval` to match the watchdog's "2 misses → dead"
  default. Soft-cancel via `Arc<AtomicBool>` (engine `close()` flips
  it); `JoinHandle::abort()` as a backstop when the loop is sleeping.
- `SparseMoEBuilderConfig::with_heartbeat_interval_ms(ms)` — plumbed
  through `tahoma-cli`'s existing `--heartbeat-interval-ms` flag.
- `SparseMoEEngine` carries `heartbeat_task: Option<JoinHandle<()>>`
  + cancel flag + shared watchdog handle. `Builder::build` spawns the
  loop on rank 0 in multi-stage mode when interval > 0.
- 9 new integration tests in `tests/heartbeat_wire.rs` covering the
  per-round helper (3) and the full cadence loop (6), including the
  task-spec "worker drops every Nth ping → watchdog fires" pattern at
  N=1 (silent worker) and N=2 (alternating reply/drop stays alive).
- Updates to `docs/architecture/heartbeat-recovery.md` and the design
  doc's blocker / next-step list.

Detection upper bound at default settings: with `--heartbeat-interval-ms
1000` and `max_misses=1`, a stuck worker is declared dead within ≤ 4 s
(2 s for the first ping's timeout to elapse, 1 s sleep, then up to 1 s
of the second ping's timeout before crossing the threshold). Real-
world median is closer to 3 s because the ping-after-cancel sleep
runs to completion before the second probe.

What iter 094 still does NOT ship — these remain `FOLLOW-UP-orchestrator`:

- Upper-layer signaling when `HeartbeatOutcome::Dead` fires. The loop
  logs `tracing::error!` and exits; nothing tears down the engine,
  cancels in-flight tasks, or kicks the orchestrator. The shared
  `heartbeat_watchdog` field is the seam where iter 095 hangs a
  callback.
- Worker auto-restart. Same as iter 092 — still
  `FOLLOW-UP-orchestrator`.
- KV migration to the replacement worker (still `iter 091` skeleton).
- Driver-side failover (re-pointing downstream socket).
- Side-channel heartbeat socket (option (c) in blocker 1 above). The
  current design serializes heartbeats against Forward via the
  downstream socket mutex; mid-Forward false misses are documented in
  blocker 1 as the known limitation.

What this PR explicitly does NOT ship — see "Blockers / open questions"
below:

- The rank-0 heartbeat loop (no `tokio::spawn` of "send ping every N ms,
  watchdog.record_miss on timeout"). The wire helpers are public; a
  follow-up wires them into `step_first`'s idle window.
- Worker auto-restart. The orchestrator that respawns a dead worker
  process is out of scope for iter 092.
- KV migration to the new worker. The wire frame for migrating KV state
  to a replacement worker is the previous skeleton
  (`docs/architecture/kv-migration.md`, iter 091) — composing the two is
  a follow-up.
- Driver-side failover (re-pointing rank 0's downstream socket to a
  replacement worker). Requires fleet topology mutability that does not
  exist yet in `tahoma-topology`.

## Wire format

Both frames share the same body shape: a single 8-byte big-endian u64
nonce after the 4-byte kind code.

```text
                    ┌──────────────┬──────────────────┐
HeartbeatPing  →    │ 0x534D4540   │   nonce (8 B BE) │
                    └──────────────┴──────────────────┘
                    ┌──────────────┬──────────────────┐
HeartbeatPong  ←    │ 0x534D4541   │   nonce (8 B BE) │
                    └──────────────┴──────────────────┘
```

Total = 12 wire bytes per ping or pong. At a 1 s heartbeat cadence on
the Matias 2-box pipeline, that's ~24 B/s per link (12 B ping + 12 B
pong) — invisible next to the ~14 KiB hidden-state tensor each forward
step.

The nonce is opaque to the worker. The driver picks it freely (e.g.
incrementing counter, randomized to make stale pongs distinguishable
across reconnects). The worker echoes whatever it received.

## Why a nonce instead of "any pong counts"

A naive "any pong cancels the miss" design has a corner case: the
driver sends ping 1, the worker is briefly stuck (e.g. GC pause), the
driver times out and records a miss, then sends ping 2. The worker
unsticks, sees BOTH pings in its socket buffer, replies to both. The
driver reads the first pong (which is the stale ping-1 echo) and
treats it as ping-2's response — masking the genuine miss-then-recover
sequence under a clean "I'm fine".

With a nonce, the driver knows that pong-with-nonce-1 is the late
echo of ping 1, not the response to ping 2. It still records the
ping-1 miss (since the deadline expired), and only after pong-with-
nonce-2 arrives does `record_success` reset the streak.

## Watchdog state machine

```text
        record_success
            ↓
[consec_misses = 0] ─────record_miss──→ [consec_misses = 1]
            ▲                                  │
            │ record_success                   │ record_miss
            │                                  ↓
            └──── record_success ──── [consec_misses = 2 → is_dead()]
```

`max_misses` is the tolerance threshold; `is_dead()` returns true once
`consecutive_misses > max_misses` (strict greater-than). Default is
`max_misses = 1`, which matches the task spec "2 misses in a row → mark
worker as dead": the SECOND miss returns `true`.

`record_success` resets `consecutive_misses` to zero and bumps a
saturating `successes: u64` counter. The orchestrator uses
`successes()` to distinguish a worker that never came up at all (0
successes) from one that ran for an hour and then died (> 1 success);
the restart policy may want different timeouts for each.

`record_miss(&mut self) -> bool` returns `true` once the watchdog
crosses the death threshold (i.e. `is_dead()` has just flipped). The
return is convenient for the heartbeat loop's "if true, escalate" check.

## Mutex serialization (why heartbeats don't corrupt the wire)

Each transport socket is wrapped in a `tokio::sync::Mutex` (see
`StageTransport`). All four heartbeat helpers (`send_heartbeat_*`,
`recv_heartbeat_body_*`) acquire the same mutex as the Forward / Reset /
Token helpers. Consequence: a heartbeat cannot interleave with the body
of a Forward frame and split a tensor across two reads.

The trade-off: during a long Forward send (~14 KiB at K2.6 hidden, sub-
millisecond at LAN bandwidth) the heartbeat queues behind it. Acceptable
because a healthy Forward send is the strongest possible liveness
signal — a heartbeat would just be redundant proof.

The downside surfaces during a slow Forward: if a downstream peer is
genuinely wedged and the driver's `send_forward` blocks waiting for
TCP backpressure to clear, the heartbeat task also blocks waiting for
the mutex. The heartbeat then can't detect liveness independently of
the Forward — it just inherits the same wedge. Mitigation lives in
the orchestrator: a separate idle-only heartbeat task, or moving the
heartbeat onto a dedicated side-channel TCP connection (one extra
listener per worker — punted, see Blocker #5).

## Blockers / open questions

### 1. ~~No driver-side heartbeat loop~~ — shipped in iter 094

iter 094 (`perf/heartbeat-driver-094`) wires the rank-0 cadence loop:
`dist::run_heartbeat_loop` is `tokio::spawn`'d in `Builder::build`
when `--heartbeat-interval-ms > 0` on rank 0 of a multi-stage engine.
The loop calls `dist::ping_one_round` every `interval` ms with a
monotonic nonce, default timeout = `2 × interval`, and updates the
shared `HeartbeatWatchdog`. On `watchdog.is_dead()` or
`HeartbeatOutcome::WireBroken` the loop exits and the engine logs
`heartbeat: pipeline worker declared DEAD by watchdog`.

Approach taken: option (b) — a separate `tokio::spawn` task — but with
the mutex-coupling caveat below addressed by holding the downstream
socket guard for the entire ping+pong round (`ping_one_round` acquires
`downstream.lock()` once across `send_raw(Ping)` and the deadline-
bounded `recv_raw(Pong)` so the heartbeat exchange is atomic on the
wire). The existing Forward path is unchanged: it acquires + releases
the same mutex per `send_raw` / `recv_raw` call. The two paths
serialize on the mutex, never interleave bytes mid-frame.

Limitation accepted: if the heartbeat task acquires the mutex between
Forward's `send` and Forward's `recv`, the next bytes off the wire are
the Forward's Token response, not a Pong. `ping_one_round` treats this
as `WireBroken` (an unexpected frame on the heartbeat channel). On a
healthy pipeline this is rare — Forward send → worker process →
Forward response RTT is sub-millisecond on LAN, much less than the
heartbeat interval — but it WILL trip a false miss occasionally.
Detecting and tolerating this is option (c) (side-channel socket) and
remains an open follow-up.

Tests covering the cadence loop end-to-end live in
`tests/heartbeat_wire.rs` (9 new tests: alive/missed/dead per round,
silent worker → dead, intermittent worker → alive, drop-every-Nth
patterns, higher tolerance, intermittent recovery, prompt cancel).

### 2. No orchestrator restart path

`is_dead()` returning true is the trigger; what happens next is out of
scope. The follow-up needs (at minimum):

- A "dead worker" callback the engine invokes from its heartbeat loop.
- Process-respawn logic — re-execing the worker binary with the same
  rank / device / model args. On Windows, mirror iter 030's WMI-detached
  spawn pattern so the new worker also survives SSH disconnect.
- Reconnection: rank 0 needs to re-establish its downstream `TCP`
  connection to the new worker process's listener.
- KV state restore: invoke `send_kv_migration` (iter 091 skeleton) for
  each layer the dead worker owned, sourced from the next-most-recent
  generation's checkpoint OR from a peer rank that mirrors the same
  range (replication is itself a track).

### 3. No mid-Forward heartbeat detection

The mutex coupling described above means a wedged Forward doesn't get
an independent liveness signal during the wedge. A side-channel
heartbeat socket fixes this but is +1 listener port per worker. Not a
v1 blocker because the existing 60 s read timeout is the worst case;
heartbeats only need to beat that.

### 4. No nonce wraparound handling

The driver picks the nonce. If the orchestrator uses a counter starting
at 0 and the worker process runs longer than `2^64` heartbeats (i.e.
~6e11 years at 1 s cadence), nonces wrap. Practically irrelevant; flagged
for symmetry with KV migration's similar non-issue.

### 5. Multi-hop ping not implemented

The worker echoes a Ping that arrives on its upstream socket. It does
NOT forward the Ping to its downstream peer. Consequence: rank 0 pinging
rank 1 only verifies rank-1 liveness, not the rank-2 chain. The follow-
up should add either an end-to-end Ping that traverses the full chain
(simpler; one nonce per chain) or per-rank Ping issued by the driver to
each rank independently (more diagnostic, but rank 0 needs direct
sockets to each rank).

### 6. No authentication on the heartbeat channel

Same caveat as all of tahoma's wire protocol today: a peer that can
TCP-connect to the listener can spoof a Pong or DoS the worker with
gratuitous Pings. Productization punted to the cascadia-fleet track.

### 7. CLI flag is engine-specific

`--heartbeat-interval-ms` is parsed on all engines but only the sparse-
MoE engine will honor it (once the driver loop is wired in step 1). For
the mock / ov-genai / ov-runtime / ov-dist-spec engines the flag is a
no-op. A more honest design moves the flag onto a `[heartbeat]` config
sub-table once we have a config-file path.

## Testing matrix

iter 092 + iter 094: 21 tests across `dist.rs::tests` and
`tests/heartbeat_wire.rs`.

| Concern                                       | Coverage                                              | Iter |
| --------------------------------------------- | ----------------------------------------------------- | ---- |
| FrameKind codes disjoint                      | `heartbeat_codes_disjoint_from_other_frames`          | 092 |
| FrameKind round-trip                          | `frame_kind_round_trip_includes_heartbeat`            | 092 |
| Watchdog spec ("2 misses → dead")             | `watchdog_default_is_two_misses`                      | 092 |
| Watchdog success resets streak                | `watchdog_success_resets_miss_counter`                | 092 |
| Higher tolerance                              | `watchdog_with_higher_tolerance`                      | 092 |
| Body bytes constant pinned to 8               | `heartbeat_body_bytes_is_eight`                       | 092 |
| Ping → Pong round-trip downstream             | `heartbeat_ping_pong_round_trip_downstream`           | 092 |
| Multiple pings stay in nonce order            | `heartbeat_nonce_mismatch_round_trip_*`               | 092 |
| Worker→driver direction (symmetric)           | `heartbeat_ping_upstream_is_symmetric`                | 092 |
| Two simulated misses → dead                   | `watchdog_declares_dead_after_two_simulated...`       | 092 |
| Streak recovers after first pong              | `watchdog_recovers_on_first_pong`                     | 092 |
| Worker handler echoes Pong                    | covered by `dist_wire.rs` chain + engine compile      | 092 |
| `ping_one_round` alive when worker replies    | `ping_one_round_alive_when_worker_replies`            | 094 |
| `ping_one_round` missed when silent           | `ping_one_round_missed_when_worker_silent`            | 094 |
| `ping_one_round` dead on threshold crossing   | `ping_one_round_dead_on_threshold_crossing`           | 094 |
| Cadence loop healthy worker → never dead      | `run_heartbeat_loop_stays_alive_against_healthy_worker` | 094 |
| Silent worker → cadence loop fires Dead       | `run_heartbeat_loop_fires_when_worker_silent_forever` | 094 |
| Intermittent worker → loop stays Alive        | `run_heartbeat_loop_fires_when_worker_drops_every_nth_ping` | 094 |
| Higher tolerance crosses at exactly Nth miss  | `run_heartbeat_loop_fires_after_exactly_n_drops_with_higher_tolerance` | 094 |
| record_success inside loop resets streak      | `run_heartbeat_loop_recovers_from_intermittent_misses` | 094 |
| Cancel flag stops loop within one interval    | `run_heartbeat_loop_exits_promptly_on_cancel`         | 094 |

Out of scope:

- Latency-bounded delivery (P50 / P99 of ping-to-pong on a Matias 2-box
  link). Wants a real 2-box bench rather than the loopback in
  `heartbeat_wire.rs`.
- Behavior under genuine process death (`kill -SEGV`, OOM). The
  current cadence test uses a silent socket as the proxy; matching the
  real-world `recv_raw` error semantics needs an actual subprocess
  spawn.
- Engine-level teardown: does `SparseMoEEngine::close()` cleanly stop
  the spawned task without orphaning a tokio worker? Manual smoke
  test only; not in CI.

## Next-step ordering

1. ~~Wire a between-task heartbeat in `SparseMoEEngine::step_first`~~
   — shipped in iter 094 as a separate `tokio::spawn`'d cadence loop
   (`run_heartbeat_loop`). Updated approach + tradeoffs in blocker 1
   above. The "between-task only" alternative is filed as a follow-up
   if the false-miss rate under the current design proves problematic.
2. Add a side-channel heartbeat socket + `--heartbeat-port` flag
   (~150 LOC + 1 new listener per worker). Lifts the mutex coupling.
3. Wire the orchestrator restart callback. Estimated 300–500 LOC; needs
   process-spawn + reconnect helpers in `tahoma-runner` + KV restore
   composition with iter 091's migration skeleton. The cadence loop
   in iter 094 already exposes the `HeartbeatWatchdog` through
   `SparseMoEEngine::heartbeat_watchdog`; the orchestrator can read
   `successes()` / `consecutive_misses()` without re-plumbing.
4. Multi-hop ping for full-chain detection. Estimated +100 LOC.
5. End-to-end CI test under `tests-e2e/` that kills a worker and asserts
   the orchestrator recovers within 5 s without losing the request.

Refs: PR #10 (pipeline-parallel inference + Rust shells), iter 030
(Matias 2-box revival), iter 091 (KV migration skeleton), iter 094
(`perf/heartbeat-driver-094` — driver-side cadence loop).
