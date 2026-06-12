# Qwen3.6-35B M4' — cross-node stage pipeline (rev 2)

Status: **DRAFT rev 2** — rewritten after the 4-angle rev-1 review
(adversarial, feasibility-vs-code, scope, external/codex). Successor
milestone to M3' (`qwen36-moe-support.md`). All numbers measured
(spikes 1–3, 2026-06-12, pawan-01/04) unless marked *extrapolated*.

## 0. Rev-1 → rev-2 changes (review disposition)

The owner re-framed the goal, which resolves the review's top finding by
changing the objective rather than defending it:

- **Goal is CAPABILITY PARITY, not a perf win.** `ov-runtime` runs
  per-stage shards cross-machine (`--rank/--total/--next`; the llama-8B
  2-stage ran on these exact nodes); `gemma4` and `ov-dist-spec` are
  multi-stage engines. Qwen3.6 is the only sharded model locked to one
  box. M4' makes it a peer of the others. The ~7 % decode wire tax is
  the accepted cost of the capability, as it was for llama.
- GPU-prefill handoff (rev 1's headline) is demoted to an OPTIONAL
  follow-on milestone with its own gate; its kill-question (single-box
  sequential GPU prefill might be as good) no longer threatens the core.
- Scope cuts applied wholesale from review: M4'-0 collapsed into a
  day-0 probe; `--prefill-device` flag dropped; reset protocol is one
  RESET/RESET_ACK exchange, fail-loud, no retry loop; control frames
  piggyback the existing transport session (no second socket); M4'-1 is
  internal scaffolding until its gate passes; GPU residency defaults to
  recompile-first, resident variant only if measurement demands it.
- Feasibility corrections adopted: position travels as the existing
  i64 position-prefix frame (static-KV mechanism, `runtime.rs:262`), NOT
  `[4,1,n]` mRoPE rows (derivable stage-locally); the stateful path's
  implicit `seq>1` reset heuristic is EXPLICITLY not reusable (chunked
  prefill would re-trigger it mid-task); shim FFI is ~3 Rust methods /
  ~8 C ABI entry points in the existing getter style; the engine
  per-stage refactor is named as the dominant cost.

## 1. Goal

`cascadia worker --rank R --total 2 --engine qwen36-moe` works exactly
like it does for `ov-runtime`: stage0 node runs embeddings + layers
0–19, stage1 node runs layers 20–39 + logits, hidden states cross the
wire per chunk/token, greedy decode is token-faithful to the single-box
engine, and the M3' robustness matrix holds across two processes on two
machines.

Non-goals (v1): GPU prefill (M4'-2, optional), batching, mid-task
failover/state migration, enterprise integration (gossip EngineKind
bump per ADR-001 only after M4'-1 passes), expert parallelism (dead
as-wired: 14.5 ms relay RTT × ~40 hops/token ≈ 580 ms/token).

## 2. Measured priors

- Wire: 8 KB round-trip pawan-01↔04 median 14.55 ms, p95 16.4 ms (DERP
  relay — node subnets do not route directly). Decode budget impact:
  ~7 % on ~210 ms/token → floor ≈ 4.4 tok/s vs 4.8 single-box.
- 2 MB prefill-chunk frames: latency measured only at 8 KB; sustained
  relay throughput UNMEASURED → M4'-1 day-0 probe records 2 MB
  one-way p50/p95/p99 (review finding; bandwidth could serialize
  prefill).
- State is stage-local (M3' §3) — nothing but hidden states, positions
  and token ids cross the wire.
- Full-chain handoff/parity machinery proven by spikes 1–3 (16/16
  token-parity vs the CPU-pure golden after cross-request state import).
- Stage sizes ~9–10 GB; CPU stage compile ~25 s; transport frame
  (`cascadia-transport`, 20-byte header, rank ≤3, 256 MiB cap) fits
  every frame this design sends.

## 3. Design

### 3.1 Topology and frames

```
A (rank 0): embeddings + stage0          B (rank 1): stage1 + logits
prefill:  per 256-chunk: [pos-prefix i64] + [1,n,2048] f32  ->  B
decode:   per token:     [pos-prefix i64] + [1,1,2048] f32  ->  B
                         B -> A: token id (existing token-return path)
control:  RESET / RESET_ACK frames on the SAME transport session
```

- Position-prefix frame per activation frame (the static-KV path
  mechanism). Downstream resets its OV state only at position 0 — the
  stateful `seq>1` heuristic is disabled for qwen36 stages.
- mRoPE rows are built stage-locally from the absolute position (all 4
  rows identical to `t0..t1` in text-only mode, `qwen36.rs:256`).
- Decode token return reuses the ov-runtime token-return path.

### 3.2 Reset protocol (minimal, review-cut)

Task admission on A: A resets local state, sends RESET, B resets and
replies RESET_ACK, A admits. No ack → task fails loud; no retry loop
inside the engine (an unstable wire is invariant-7 territory; the API
caller retries). The admission wait must complete within one `step()`
call or emit progress — the runner closes streams after 3 consecutive
empty steps (`cascadia-runner/src/lib.rs:28`).

### 3.3 Task epochs and stale frames (codex finding)

Every frame carries the task epoch (one u32 prefix or folded into the
position frame). A peer inside a synchronous OV call cannot be
interrupted; when it returns, any frame from an older epoch is dropped.
Cancel/disconnect on either side → epoch bump + RESET exchange before
the next admit. This is the entire distributed-cancel story; nothing
fancier is in scope.

### 3.4 Startup handshake (codex finding)

Before first admit, A→B exchange: manifest hash, stage range, OV
version string, wire dtype. Mismatch = refuse to serve, log both sides.
(State schema never crosses nodes, so state-shape skew is out of scope.)

### 3.5 Failure semantics

Connect-once like ov-runtime today: peer process death mid-task = task
failure; recovery = restart both workers (documented operator action).
Reconnect machinery is explicitly out of scope for M4'-1 — the gate
tests "kill + restart both → next task clean", not live re-pairing
(feasibility finding: no re-dial/re-accept exists anywhere today, and
building it is not parity work).

## 4. Invariants

1–4 of M3' §4.1 carry verbatim (batch=1, greedy-only, position-0 reset
as sole recovery, no ShardSpec changes). New:

5. Task admits only after RESET/RESET_ACK; fail-loud on no-ack.
6. Frames from a stale epoch are dropped silently; state-mutating work
   happens only for the current epoch.
7. Peer loss mid-task = task failure + position-0 re-entry after
   operator restart. No partial-state serving.

## 5. Deltas (sized honestly per the feasibility review)

- **Engine (dominant cost — most of the milestone):** per-stage mode in
  `Qwen36Engine`: stage-role dispatch (first/last for 2 stages),
  rank-aware load (compile only MY stage; embeddings+tokenizer on rank
  0 only), transport plumbing + async bridge (port the
  `send_hidden_downstream`/`recv_hidden_from_upstream`/
  `send_token_to_upstream` pattern from `runtime.rs:696-820`),
  chunked-prefill send/recv with position prefixes, RESET/ACK + epoch
  handling. Template exists and is proven; still days, not hours.
- **CLI/runner (small):** `Qwen36Builder` gains rank/total (stage
  selection) + a real `configure_listen` (today inherits the no-op
  default); `cmd_worker` already passes everything needed
  (`cli/src/lib.rs:793-845`).
- **Transport (verify-only):** existing frames suffice; no crate
  changes expected. Day-0 probe confirms.
- **Shim FFI: NONE for M4'-1.** State get/set is only needed for the
  optional GPU handoff (M4'-2). CPU-only pipeline keeps state inside
  each stage's single infer request.
- **Exporter: none.**

## 6. Milestones

- **Day-0 probe (hours, no engine code):** two Python processes on
  pawan-01+04 speak the proposed frame sequence over TCP through the
  real relay: position-prefix + 2 MB chunk frames ×4, then 64 decode
  frames, RESET/ACK, epoch-stale drop. Records: 2 MB p50/p95/p99,
  decode-frame p95, protocol soundness. Gate to start engine work:
  prefill wire total < 5 s @1K-equivalent AND decode p95 < 25 ms.
- **M4'-1 (cross-node CPU pipeline — THE milestone):**
  Gate, all required:
  1. 64-token greedy through the 2-node pipeline matches the single-box
     engine per the M3' §5 token-agreement protocol on the parity
     prompt set (not raw `==`; near-tie allowance per the established
     criterion), after one hardware-homogeneity check run on pawan-04
     (one CPU decode measurement — all priors are pawan-01).
  2. Decode ≥ 4 tok/s short-ctx.
  3. Robustness matrix, named rows: cancel mid-decode entered at A;
     SSE disconnect at A; kill B mid-decode → A fails task loud; kill A
     mid-decode → B drops stale frames on restart; restart both → next
     task clean; 3 sequential tasks → no state bleed (golden repeat).
  4. Wire histogram during the gate run; decode p95 > 40 ms → M4'
     BLOCKED pending a direct-path ops fix (not a pass-with-caveat).
- **M4'-2 (OPTIONAL, separately justified — GPU prefill handoff):**
  enters only with its own spike first: chunked-256 GPU prefill state
  correctness + rate AND single-box sequential GPU prefill TTFT with
  warm compile cache. If single-box sequential gets within 1.5× of the
  2-node projection, M4'-2 is cut and the single-box trick becomes an
  M3' enhancement instead. Shim state FFI (~3 Rust / ~8 C entries)
  belongs to this milestone only.
- **Exit/stop rule:** any M4'-1 gate failing twice with the day-0 probe
  green → stop, record why in this doc, M3' single-box remains the
  shipped answer. (Probe failing = stop before engine code at all.)

## 7. Risks (re-ranked per review)

1. **Distributed cancel/epoch correctness** — the M3' robustness matrix
   squared. Retired by M4'-1 gate row set (the matrix is named, not a
   label).
2. **Relay bandwidth on 2 MB frames** — unmeasured; retired by the
   day-0 probe BEFORE engine work (was rev-1's mispriced risk).
3. **Protocol mismatch with the stateful reset heuristic** — designed
   out (§3.1 position-prefix + heuristic disabled); day-0 probe
   double-checks framing.
4. **pawan-04 hardware non-homogeneity** — retired by the one-run check
   in gate 1.

## 8. Decisions closed (were §8 open questions in rev 1)

- Control channel: same transport session, RESET/RESET_ACK frames.
- M4'-1 visibility: internal scaffolding until its gate passes.
- GPU residency: recompile-first; resident variant only if M4'-2
  happens and measurement demands it.
- `--prefill-device`: does not exist; GPU handoff (if built) is
  unconditional-when-available behind the M4'-2 gate.
