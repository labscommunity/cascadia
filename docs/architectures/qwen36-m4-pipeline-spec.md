# Qwen3.6-35B M4' — 2-node layer pipeline with per-node GPU-prefill handoff (rev 1)

Status: **DRAFT rev 1 — awaiting multi-angle review.** Successor
milestone to M3' (`qwen36-moe-support.md`); every number below is
measured, not estimated (spikes 1–3, 2026-06-12, pawan-01/04).

## 1. Value proposition (the part review must attack hardest)

A 2-node pipeline does NOT speed up single-stream decode — stages
serialize, and the wire adds ~14.5 ms/token. What it buys, measured:

1. **TTFT.** Single-box GPU prefill of the full model is impossible
   beside CPU decode (2× ~18 GB weights vs 31.6 GB). Split per node,
   each stage is ~9–10 GB — and **GPU+CPU dual residency per stage
   fits** (spike 1). GPU prefill measured 6–10 s @1K whole-model vs
   ~41 s CPU batched: target TTFT @1K ≤ 15 s, ~3× better than M3'.
2. **Context headroom.** 262K-native model; KV + DeltaNet state plus
   weights squeeze one 32 GB box at long context. Half the weights per
   node ≈ double the state budget.

Anti-scenario honesty (carried from the M4' gate): 35B int4 FITS one
box; if review judges TTFT+context insufficient to justify the build,
the correct outcome is CUT, not build. The killer-demo framing
("model too big for one box") is NOT available for this model.

## 2. Measured priors (all 2026-06-12 unless noted)

- **Spike 1 — state handoff:** all 40 per-stage VariableStates are
  plain f32 tensors; GPU→export 34.1 MB @32-tok ctx in 46 ms, CPU→import
  23 ms. Hidden-level drift rel 4.0e-2 (step 1) → ~1.9e-1 (step 8)
  from mixed GPU/CPU f16 regimes; no-state control 4.3e+1.
- **Spike 2 — wire:** 8 KB ([1,1,2048] f32) round-trip pawan-01↔04
  median **14.55 ms**, p95 16.4 ms, over the tailscale DERP relay — the
  only path (node subnets don't route to each other). Expert
  parallelism (§4.3 of the M3' spec) is dead as-wired: ~40
  round-trips/token ≈ 580 ms/token floor.
- **Spike 3 — full-chain handoff:** GPU prefill both stages → state
  import to CPU stages → 16 greedy tokens **token-identical** to the
  CPU-pure golden. Drift does not flip greedy near-ties.
- **M3' baselines:** CPU 2-stage decode 4.8 tok/s (chain) / 4.7–8.8
  (engine, short ctx); CPU batched prefill ~13.8 tok/s engine-measured;
  GPU whole-model prefill 6–10 s @1K; stage compile: CPU ~25 s,
  GPU ~16 s.

## 3. Topology and dataflow

Two nodes, one stage each (stage0 = node A, stage1 = node B). Node A is
the entry (gateway). Per task:

```
prefill:  A: GPU stage0 prefill (chunked 256)  --hidden[1,T,2048]/chunk--> B: GPU stage1 prefill
          A: state GPU->CPU import                                        B: state GPU->CPU import
decode:   A: CPU stage0 step --hidden[1,1,2048]--> B: CPU stage1 step --token id--> A
```

- State NEVER crosses the wire — DeltaNet/KV state is stage-local
  (M3' spec §3); the handoff is intra-node (GPU request → CPU request).
- Wire frames: prefill chunks `[1,256,2048]` f32 (2 MB/chunk, ~4
  chunks @1K — bandwidth not latency bound); decode `[1,1,2048]` f32
  (8 KB) forward + token id back. Decode budget: 14.5 ms wire on
  ~210 ms compute ≈ 7% tax → floor ~4.4 tok/s vs 4.8 single-box.
- Per-node GPU residency is TRANSIENT: GPU stage compiled for prefill,
  state exported, GPU model FREED before decode begins (spike-3
  sequencing) — steady-state memory = CPU stage only (~10 GB/node).
  Alternative (GPU model kept resident for the next task's prefill)
  costs ~10 GB/node but saves ~16 s recompile; decided at M4'-1 by
  measuring task-to-task latency both ways. OV compile cache may make
  recompile cheap; measure before choosing.

## 4. Invariants (carried + new)

1–4 of M3' §4.1 carry verbatim (batch=1, greedy-only, position-0 reset
as sole recovery, no ShardSpec changes beyond peers). New:

5. **Cross-node reset atomicity:** a task admits only when BOTH nodes
   confirm position-0 (reset ack on the control channel). A node that
   cannot confirm forces the pair into reset-retry; no partial-state
   serving, fail-loud after N attempts.
6. **Per-regime parity:** acceptance compares GPU-prefill+CPU-decode
   runs against their own goldens, not against CPU-pure (spike 1 drift
   is real even when greedy survives it; spike 3 shows survival on the
   reference prompt, not a guarantee).
7. **Wire is hidden-state + position + token only.** No state
   migration, no mid-task failover: peer loss mid-task = task failure +
   position-0 re-entry on a healthy pair (M3' rule extended).

## 5. Deltas required (smallest honest list)

- **shim FFI:** `state_names()/get_state()/set_state()` on Runtime
  (spike used Python; the engine needs it in the C++ shim). New surface,
  ~3 functions.
- **engine:** `Qwen36Builder::connect` accepts upstream/downstream
  peers (today: hard-reject); per-stage mode (run MY stage only);
  prefill-chunk send/recv; decode-step send/recv; GPU-prefill handoff
  path behind a flag (`--prefill-device GPU`).
- **transport:** reuse the existing stage-frame transport
  (`ov-runtime` multi-stage already ships hidden states cross-machine —
  the llama 2-stage mesh ran on these nodes). Delta = qwen36 frame
  carries mRoPE position rows; verify the existing frame fits or extend.
- **runner/CLI:** `--rank/--total/--next` wiring for qwen36-moe
  (exists for ov-runtime; mirror it).
- **exporter:** no changes (stages already per-stage IRs).
- Explicitly NOT in scope: enterprise integration (gossip EngineKind
  bump per ADR-001 happens only after community M4' passes), expert
  parallelism (dead as-wired), state migration/failover, batching.

## 6. Milestones, each gated by measurement

- **M4'-0 (probe, ~day):** two-process SINGLE-BOX rehearsal — stage0
  and stage1 in separate processes on pawan-01, localhost TCP, full
  protocol (chunked prefill frames, decode frames, reset handshake).
  Gate: 64-token greedy == single-process engine output; decode ≥
  4.5 tok/s localhost.
- **M4'-1 (cross-node, ~days):** real 2-node run pawan-01+04, CPU-only
  first (no GPU handoff). Gate: 64-token greedy == M4'-0 output;
  decode ≥ 4 tok/s; reset/cancel/disconnect matrix passes (engine kill
  mid-decode on either node → next task clean).
- **M4'-2 (GPU prefill handoff):** `--prefill-device GPU` on both
  nodes. Gate: TTFT @1K ≤ 15 s; decode unchanged from M4'-1; 64-token
  output coherent + stable vs its own golden (invariant 6); per-node
  peak ≤ 24 GB.
- **Exit:** all three gates green → M4' acceptance = the §1 value prop
  delivered with numbers; any gate failing twice → write down why and
  stop (the M3' single-box engine remains the shipped answer).

## 7. Risks (ranked, with the measurement that retires each)

1. **Reset coordination correctness** under cancel/disconnect/timeouts
   ×2 nodes — the M3' robustness matrix squared. Retired by the M4'-1
   gate matrix. (Highest risk: this is distributed-state correctness,
   not perf.)
2. **mRoPE position frame mismatch** with the existing transport frame.
   Retired at M4'-0 (localhost protocol rehearsal).
3. **GPU prefill chunking semantics** — chunked GPU prefill state
   accumulation has only been probed single-shot @32 tokens (spike 1)
   and unchunked @8-token prompt (spike 3); chunked-at-256 GPU prefill
   state correctness is unmeasured. Retire FIRST in M4'-2 with a
   one-cell probe before any engine code depends on it.
4. **Relay-path variance** — 14.5 ms median is one measurement window;
   DERP under load may spike. M4'-1 records a latency histogram during
   the gate run; p95 > 40 ms forces the direct-LAN ops conversation.

## 8. Open questions for review

- Is §1 enough to justify the build at all (the anti-scenario)?
- Keep-GPU-resident vs recompile-per-task (§3) — right default?
- Should M4'-1 CPU-only cross-node be PUBLIC (a shippable "2-node
  qwen" without GPU handoff) or internal-only scaffolding?
- Control channel: piggyback the existing transport session vs a
  separate TCP control socket (reset acks, token return path).
