# Three-tier {iGPU, NPU, CPU} placement (#41)

Status: **in progress** (branch `feat/three-tier-placement-41`).
Builds on `profile-devices` (#45, step 1) and the now-working CPU (#62)
and NPU (#63) shard-execution paths.

## TL;DR of the measure-first phase

Issue #41 asks for an ILP that places model layers across {iGPU, NPU,
CPU} and **beats `--device GPU` alone by ≥10% steady-state tok/s**. Before
committing an architecture we measured the actual target hardware
(`cascadia profile-devices` + an OV device-property probe) and read the
runtime. The conclusion that drives the design:

> **On a model that *fits* the iGPU, no placement can beat GPU-alone on
> single-stream tok/s.** The iGPU is the fastest single tier, the runtime
> runs pipeline stages *sequentially per token* (no cross-request
> overlap), so moving work to a slower tier only *adds* latency. The
> ≥10% win exists only in the **memory-forced regime**: a model that does
> not fit in the iGPU's memory budget, where tiered placement is the only
> way to run it at all — and where a cost-aware ILP beats OpenVINO's naive
> `HETERO` spill. This is exactly PowerInfer §6.3's offline placement
> problem, adapted to Intel UMA.

## Measured hardware — pawan-01 (the validation box)

`Intel Core Ultra 7 258V` (Lunar Lake), **31.6 GB** unified DDR5 (UMA):

| Tier | OV name | int8 TOPS | fp16 TFLOPS | Memory budget | Streams |
|---|---|---|---|---|---|
| iGPU | Arc 140V | **63.9** | 31.9 | **16.48 GB** (`GPU_DEVICE_TOTAL_MEM_SIZE`) | 1–2 |
| NPU | AI Boost (arch 4000) | **46.7** | 23.3 | **16.0 GB** (`NPU_DEVICE_TOTAL_MEM_SIZE`) | 1–4 |
| CPU | Ultra 7 258V | — | — | system RAM (31.6 GB shared) | 1–8 |

Key facts:
- **Compute ranks GPU > NPU > CPU** (63.9 > 46.7 TOPS int8). The NPU is
  *not* faster per-op than the iGPU — so splitting a fitting model onto
  it is pure loss for single-stream latency. (NPU's value is capacity +
  power efficiency, not peak speed.)
- **iGPU budget ≈ 16.5 GB.** Weights + KV + activations must fit. A dense
  model whose int4 weights exceed ~16 GB (≈ 32B params) **cannot load on
  the iGPU alone** → memory-forced. (Confirmed empirically in the bench
  below.)
- **UMA: the per-device caps overlap — they all draw from ONE 31.6 GB
  pool.** The iGPU and NPU each *report* ~16 GB, but that is an
  addressing limit, not private memory; placing 16 GB on the iGPU **and**
  16 GB on the NPU would need 32 GB > 31.6 GB physical and OOM. So the ILP
  needs **two** memory constraints: a per-device cap (`≤ cap[d]`, the
  addressing limit) **and** a global cap (`Σ over all stages ≤ usable
  system RAM`, the shared pool). A model that forces multi-device on
  Lunar Lake therefore lives in a narrow band: int4 weights in
  `(16.5 GB, ~28 GB]` — bigger than the iGPU can address, small enough to
  fit the pool. ≈ 32B–48B int4.
- `MAX_BATCH_SIZE = 1`, `OPTIMAL_BATCH_SIZE = 1` on the iGPU; the runtime
  has no cross-request pipelining today (see "Why not throughput").
- NPU has **no bf16** (`DEVICE_GOPS[bf16]=0`); its path is int4/int8
  static-shape stateless (the #63 export mode).

## Why not the throughput regime (yet)

A balanced 2-stage GPU+NPU pipeline *could* ~2× aggregate throughput **if**
the runtime overlapped concurrent requests (stage 0 on request B while
stage 1 finishes request A). It does not: `Engine::step()` runs one task
end-to-end and stages synchronize over the TCP transport
(`crates/cascadia-engine*`, `crates/cascadia-runner`). Building
cross-request pipeline overlap is a large, separate runtime change
(tracked as a follow-up); it is **not** required to satisfy #41, and the
memory-forced regime is the honest, hardware-supported win on Lunar Lake.

## The placement problem (ILP)

Per-stage assignment over a multi-stage static export. Inputs come from a
per-(stage, device) profile:

- `lat[s][d]` — single forward latency of stage `s` on device `d`
  (`+∞` if `d` can't compile `s` → op-support gate).
- `mem[s]` — resident memory of stage `s` (≈ its IR weight bytes + KV).
- `cap[d]` — device memory budget (GPU 16.5 GB, NPU queried, CPU = free RAM).

Decision: `x[s][d] ∈ {0,1}`, each stage on exactly one device.

```
minimize    Σ_s Σ_d  lat[s][d] · x[s][d]        (+ transport between tiers)
subject to  Σ_d x[s][d] = 1                       ∀ stage s
            Σ_s mem[s] · x[s][d] ≤ cap[d]          ∀ device d   (addressing limit)
            Σ_s mem[s]            ≤ pool                        (shared UMA pool)
            x[s][d] = 0  where lat[s][d] = +∞      (op-support gate)
```

The `pool` constraint is what makes this a UMA problem rather than N
independent devices: the cheapest-latency assignment that satisfies the
per-device caps can still exceed the physical 31.6 GB, so the solver
rejects any model whose total resident memory exceeds usable system RAM.
*(Status: the v1 solver in `placement.rs` enforces the per-device caps +
op-support gate; the `pool` constraint is the first refinement landing
next, with the per-stage profiler.)*

Problem size is tiny (stages ≤ ~16, devices = 3), so we solve it
**exactly in pure Rust** (no `good_lp`/CBC system dependency — keeps the
single-static-binary, Rust-only invariant). For the fitting case the
optimum is trivially "all stages on GPU" (the tool reports this and the
placed run equals GPU-alone — the correctness/no-regression check). For
the memory-forced case the cap constraint forces overflow, and minimizing
`lat` puts the overflow on NPU before CPU.

## Build plan

1. **`cascadia profile-devices --per-stage <multi-stage-shard>`** — extend
   the existing tool to compile + time each stage IR on each available
   device (incl. the NPU static path) and record `lat`/`mem`/op-support →
   `placement_profile.json`. *(the ILP cost table)*
2. **ILP solver** (`crates/cascadia-cli` or a small `placement` module) —
   exact memory-capped assignment; emits `placement.json`
   (per-stage device list) + the objective + a human summary.
3. **Apply** — launch the heterogeneous multi-stage pipeline from
   `placement.json` (the multi-process ring already supports per-stage
   `--device`; #63 validated 2-stage). Add a launcher that spawns one
   worker per stage with its assigned device.
4. **Validate on pawan-01**:
   - *Correctness:* fitting model → ILP picks all-GPU → placed run matches
     `--device GPU` (no regression).
   - *≥10% win:* a >16.5 GB model (exported on the miner, IR shipped to
     pawan) → `--device GPU` OOMs; the ILP-placed GPU+NPU+CPU run executes
     it and beats a naive uniform/HETERO spill by ≥10% steady-state tok/s.
   - *Graceful degrade:* a stage that fails op-support on a device is
     excluded by the ILP (`lat=+∞`), placement still solves.

## References
- PowerInfer §6.3 (offline ILP placement), PowerInfer-2 §4.1.3 (dynamic
  per-batch graphs — future).
- `docs/perf/DEVICE_PROFILE.md` (#45, the step-1 profiler this extends).
- #37/#63 (NPU static path), #57/#62 (CPU path) — the tiers this places onto.
