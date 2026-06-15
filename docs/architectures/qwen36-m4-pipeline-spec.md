# Qwen3.6-35B — cross-node stage pipeline (M4')

Design for running the sharded `qwen36-moe` engine across nodes (peer of
`ov-runtime` / `gemma4`). Successor to M3' (`qwen36-moe-support.md`).
Numbers measured on fleet nodes A/B (spikes 1–3, 2026-06-12) unless
marked *extrapolated*.

## 0. Key decisions

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

- Wire: 8 KB round-trip node-A↔node-B median 14.55 ms, p95 16.4 ms (DERP
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

### 3.2 Reset protocol (minimal)

Task admission on A: A resets local state, sends RESET, B resets and
replies RESET_ACK, A admits. No ack → task fails loud; no retry loop
inside the engine (an unstable wire is invariant-7 territory; the API
caller retries). The admission wait must complete within one `step()`
call or emit progress — the runner closes streams after 3 consecutive
empty steps (`cascadia-runner/src/lib.rs:28`).

### 3.3 Task epochs and stale frames

Every frame carries the task epoch (one u32 prefix or folded into the
position frame). A peer inside a synchronous OV call cannot be
interrupted; when it returns, any frame from an older epoch is dropped.
Cancel/disconnect on either side → epoch bump + RESET exchange before
the next admit. This is the entire distributed-cancel story; nothing
fancier is in scope.

### 3.4 Startup handshake

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

## 5. Deltas

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

- **Day-0 probe — RUN 2026-06-12, PASS (with a variance caveat):**
  the full frame sequence (handshake, RESET/ACK, position-prefix +
  4×2 MB prefill chunks, 64 decode frames, stale-epoch drop) ran
  between node-A↔node-B over the live relay. Protocol sound end-to-end.
  Prefill wire 2.95 s for 4×2 MB (sustained ~2.7 MB/s — the rev-1
  bandwidth risk is retired). Decode: run 1 p50 13.5 ms / p95 39.2 ms
  (relay burst); run 2 p50 13.5 ms / p95 14.2 ms. Gate (prefill < 5 s,
  decode p95 < 25 ms): PASS on run 2; run 1's tail shows the relay is
  BURSTY — the M4'-1 gate's wire histogram must use a long window and
  the p95>40 ms block-rule stands. Engine work is unblocked.
- **M4'-1 GATES RUN 2026-06-12 — ALL PASS** (node A rank 0 ↔
  node B rank 1, CPU both, engine a195fd6, identical exporter trees
  manifest-matched 43066861):
  1. **Token agreement: PARITY_EXACT** — 64-token greedy through the
     2-node pipeline char-identical to the single-box engine on the
     same tree. Reference re-blessed first: the m4 exporter tree
     (logit-slice + inputs_embeds removal) diverges from the M3'-tree
     golden at ~token 30 with an f16 near-tie flip (both coherent —
     the documented §5-allowance case); single-box determinism
     reconfirmed before the pipeline run. The parity run itself
     exercises node B CPU decode (stage1), covering the
     hardware-homogeneity check. Prompt set 6/6
     (`golden/promptset_pipeline-2node.json`).
  2. **Decode 4.1–4.8 tok/s** short-ctx (≥4 required; single-box
     envelope 4.7–8.8 — wire tax visible but within budget).
  3. **Robustness, all rows:** cancel mid-decode PASS (+follow-up
     clean); SSE disconnect PASS; 3 sequential tasks identical (no
     state bleed); kill B mid-decode → A fails task loud (os error
     10054, final marker emitted, server healthy); kill A mid-decode →
     B survives, warns at exactly 500 ms spacing (relay backoff fix
     a195fd6, found when the first boot's dead-peer spin flooded the
     rank-1 log); restart both → next task clean (verified twice).
  4. **Wire histogram (decode, 64 frames): p50 21.2 ms / p95 23.6 ms /
     max 37.9 ms** — under the 40 ms block rule. (RTT minus
     peer-reported infer time; higher than the day-0 probe's 13.5 ms
     p50 — includes serialize + session mutex.)

  Ops notes from the run: node B enterprise cascadia-node service
  uninstalled to free RAM (OVMS held ~19 GB; rank 1 OOM'd at stage
  compile) — restore with `cascadia-node.exe service install`;
  pipeline listens on :9200 (enterprise node owns :9100); the relay
  worker pattern is `cmd /c C:\cascadia\m4_rank{0,1}.bat` held by a
  long-lived ssh session.

- **REGATE 2026-06-12 — N-stage engine + chat-template trees (commit
  1c154bc):** re-ran after the N-stage refactor, chat-template wiring,
  and cancel-tombstone landed; both nodes rebuilt + re-exported
  (exporter now copies `chat_template.jinja`, manifests still
  43066861). Results vs the first gate:
  1. **Parity PARITY_EXACT** again (re-blessed on the new template
     rendering — the API now applies the model chat template for
     qwen36, so the prompt opens the think block and decode starts at
     the reasoning content rather than emitting a literal `<think>`).
  2. **tok/s 4.0–4.2** (≥4 holds).
  3. **Robustness cancel/disconnect/bleed all PASS** (kill rows carried
     from the first gate; backoff fix already in the build).
  4. **Promptset 6/6 — and the legacy echo wart is GONE.** Before:
     `"Parisuser: What is the capital…"`; now clean `"Paris"`, `"42"`,
     `"Tokyo"`. The chat-template wiring is the fix
     (`golden/promptset_pipeline-2node-template.json`).
  - **Gate 4 wire: RELAY-BLOCKED this window.** Decode p50 *improved*
     to ~17 ms (from 21.2), but the DERP-relay p95 tail is variable
     across clean 64-frame runs: 36.7 / 43.2 / 42.5 / 20.0 ms —
     frequently over the 40 ms hard rule (bursty windows with mixed
     control frames hit p95 ~43, max 80+). The engine is not the
     cause (p50 down, parity exact every run); this is the documented
     direct-path limitation (node subnets don't route directly → relay
     through "sea"). Per §6's "p95 > 40 ms → BLOCKED, not
     pass-with-caveat", a reliable pass needs the LAN-route ops fix.
     The first gate caught a good relay window (p95 23.6); this one
     caught a marginal one. **Functional parity is proven; the wire
     SLA is relay-bound, not code-bound.**

- **M4'-1 (cross-node CPU pipeline — THE milestone):**
  Gate, all required:
  1. 64-token greedy through the 2-node pipeline matches the single-box
     engine per the M3' §5 token-agreement protocol on the parity
     prompt set (not raw `==`; near-tie allowance per the established
     criterion), after one hardware-homogeneity check run on node B
     (one CPU decode measurement — all priors are node A).
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

## 7. Risks

1. **Distributed cancel/epoch correctness** — the M3' robustness matrix
   squared. Retired by M4'-1 gate row set (the matrix is named, not a
   label).
2. **Relay bandwidth on 2 MB frames** — unmeasured; retired by the
   day-0 probe BEFORE engine work (was rev-1's mispriced risk).
3. **Protocol mismatch with the stateful reset heuristic** — designed
   out (§3.1 position-prefix + heuristic disabled); day-0 probe
   double-checks framing.
4. **node B hardware non-homogeneity** — retired by the one-run check
   in gate 1.

## 8. Implementation notes (M4'-1 engine, qwen36.rs)

Deviations/decisions made during the build; all functionally within the
rev-2 design:

- **Frame format:** 12-byte BE raw header `[kind][epoch][pos]` + body,
  not §3.1's literal i64 position-prefix tensor. Same information,
  matches the day-0 probe's validated framing; epoch folded into the
  header per §3.3's "or folded into the position frame" option.
- **Handshake (§3.4):** runs in rank-0 `warmup()` (fail-loud at boot;
  retried at first admit if the transport hiccuped). Carries proto
  version, total, wire dtype, and the FULL manifest.json text (compare
  beats hash, no new dependency). OV version string is NOT exchanged —
  the shim exposes no version API and adding FFI is out of M4'-1 scope;
  the manifest compare covers export-level skew. Mismatch poisons both
  sides (admissions fail with the reason).
- **Lockstep prefill:** every FORWARD (including intermediate prefill
  chunks) is answered with the downstream argmax TOKEN; rank 0 discards
  all but the last. Costs one ~15 ms RTT per 256-chunk (noise vs the
  2.95 s measured chunk wire time) and keeps the session quiescent
  between `step()` calls, so cancel never races an in-flight frame.
- **Wire dtype f32** per §3.1 (parity gate compares against the all-f32
  single-box engine; f16 truncation is avoidable risk).
- Stale-epoch FORWARD frames are dropped with no response; the driver's
  60 s recv timeout fails the task loud (lockstep makes this reachable
  only via epoch bugs — belt-and-braces per §3.3).

**N-stage extension (post-gate, 2026-06-12):** the 2-stage cap was
M4'-1 scope, not architecture; the engine now supports `--total N`.
Middle ranks relay: FORWARD → own stage → forward downstream → TOKEN
back upstream; HELLO and RESET chain through (rank 0's original payload
is validated by every rank; a RESET_ACK means everything downstream is
at position 0). TOKEN frames accumulate per-rank infer time so rank 0's
histogram stays the chain's true wire share. Per-hop wire tax stacks
and exporter `--total N` already slices arbitrarily.

**3-node / middle-rank LIVE VALIDATION 2026-06-12 (commit 1c154bc):**
ran a real cross-machine 3-stage chain — rank0 + rank2 co-resident on
node A (separate processes, separate stages; node A had 25 GB free, no
enterprise teardown), **rank1 the MIDDLE on node B**, so both wire hops
are A↔B real network. (A true third physical node was attempted but
the relay couldn't sustain the 7.6 GB stage copy — resets mid-transfer;
co-residency exercises the identical middle-rank code over real
cross-machine frames without the bulk copy.) 3-stage tree:
layers 0–12 / 13–25 / 26–39, manifest 5ab2dad1 matched on both nodes.
Results:
- **Parity PARITY_EXACT** vs the single-box 3-stage golden (blessed on
  node A `--total 1`) → the middle-rank relay is token-faithful.
- **Promptset 6/6**, clean (chat template through 3 ranks).
- **Robustness:** cancel mid-decode PASS (+follow-up clean); 3-task
  no-bleed PASS.
- **Wire (2 hops): p50 ~32 ms** ≈ 2× the single-hop ~17 ms — confirms
  per-hop additivity; p95 36–62 ms (2 hops × bursty relay). Decode-only
  ~3.9 tok/s (the per-hop wire tax stacks as designed).
The middle-rank code (relay role, chained HELLO/RESET, per-rank infer
accounting) is validated. 2-node remains the perf-reference config.

Launch (rank 1 first — its listener must be up before rank 0 dials;
generally start the HIGHEST rank first, then descending — each rank's
listener must be up before its upstream dials):

```
# node B (rank 1: stage1 + logits)
cascadia worker --rank 1 --total 2 --engine qwen36-moe `
  --model C:\cascadia\models\qwen36-shards-2stage --device CPU --listen :9100

# node A (rank 0: embeddings + stage0 + tokenizer + API)
cascadia worker --rank 0 --total 2 --engine qwen36-moe `
  --model C:\cascadia\models\qwen36-shards-2stage --device CPU `
  --next <p04-tailscale-ip>:9100 --api :8000
```

Gate harness: `tools/qwen36_surgery/m4_gate_serving.py` (gates 1–2 +
prompt set; gate 4 read from the rank-0 wire-histogram log line) and
`m4_gate_robustness.py` (gate 3 rows; kill rows operator-driven via its
`longgen` mode). Rank 1 needs only manifest.json, generation_config.json
and stage1/ from the shard tree (~10 GB); embeddings + tokenizer load on
rank 0 only.

## 9. Decisions closed

- Control channel: same transport session, RESET/RESET_ACK frames.
- M4'-1 visibility: internal scaffolding until its gate passes.
- GPU residency: recompile-first; resident variant only if M4'-2
  happens and measurement demands it.
- `--prefill-device`: does not exist; GPU handoff (if built) is
  unconditional-when-available behind the M4'-2 gate.
