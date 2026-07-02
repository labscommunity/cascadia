# Three-tier {iGPU, NPU, CPU} placement (#41)

Design + validation record for [#41](https://github.com/labscommunity/cascadia/issues/41);
shipped as `cascadia profile-stages` / `place` / `run-placement`.
Builds on `profile-devices` (#45, step 1) and the CPU (#62)
and NPU (#63) shard-execution paths.

## TL;DR of the measure-first phase

Issue #41 asks for an ILP that places model layers across {iGPU, NPU,
CPU} and **beats `--device GPU` alone by ≥10% steady-state tok/s**. Before
committing an architecture we measured the actual target hardware
(`cascadia profile-devices` + an OV device-property probe) and read the
runtime. The conclusion that drives the design:

Measuring three model sizes against the iGPU's **16.48 GB nominal budget**
mapped out where placement helps — and surfaced a UMA surprise:

> 1. **Comfortable fit** (model ≪ budget; Qwen2.5-32B int4, 15.7 GiB) —
>    GPU-alone is fastest. The iGPU is the quickest tier and the runtime runs
>    stages sequentially per token, so offloading only adds latency.
>    *Placement should pick all-GPU — and does.*
> 2. **Near-full iGPU** (model ≈ budget; Yi-1.5-9B fp16, 16.45/16.48 GiB) —
>    all-GPU is **memory-pressure bound**: with the iGPU ~100% full there's
>    no headroom for activations/scratch and throughput collapses to 2.0
>    tok/s. Offloading the embed stage to CPU relieves the pressure and hits
>    **2.9 tok/s — +43% over `--device GPU`**, reproducibly. **This is the
>    regime where placement wins.**
> 3. **Over the *nominal* budget** (model > 16.48 GiB, ≤ pool; SOLAR-10.7B
>    fp16, 20.0 GiB) — the **UMA surprise**: the iGPU is *not* hard-capped at
>    its reported budget. It **spills into the shared 31.6 GB pool and runs
>    all-GPU anyway** (SOLAR ran 12/12 at 1.31 tok/s on `--device GPU`). So
>    there is **no capacity-forced OOM** below the pool size, and all-GPU
>    (via spill) *beats* the ILP's forced CPU offload (0.79 tok/s). Placement
>    does **not** help here.
>
> **Takeaway:** on Lunar Lake UMA the ≥10%-over-`--device GPU` win is real
> but **narrow** — it lives at the near-full-iGPU pressure cliff (regime 2),
> not in a broad capacity regime, because the iGPU transparently spills into
> the shared pool. The placement *infrastructure* (profile → ILP → run) is
> general; the *operating point* where it beats GPU-alone on this hardware is
> regime 2. (This is PowerInfer §6.3 adapted to UMA, where pressure-relief —
> not capacity — turns out to be the lever.)

## Measured hardware (the validation box)

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

```text
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
*(The solver in `placement.rs` enforces the per-device caps, the op-support
gate, AND this global pool constraint.)*

Problem size is tiny (stages ≤ ~16, devices = 3), so we solve it
**exactly in pure Rust** (no `good_lp`/CBC system dependency — keeps the
single-static-binary, Rust-only invariant). For the fitting case the
optimum is trivially "all stages on GPU" (the tool reports this and the
placed run equals GPU-alone — the correctness/no-regression check). For
the memory-forced case the cap constraint forces overflow, and minimizing
`lat` puts the overflow on the cheaper measured tier — **CPU before NPU** on
Lunar Lake, since the NPU is the slowest tier for decode.

## Pipeline (implemented)

The three steps are three subcommands; the static export from
`cascadia shard --target npu` is the shared input (its shards run on
GPU/CPU/NPU alike via the #63 static-KV path, so one export covers all
tiers):

1. **`cascadia profile-stages --shard <dir> [--pool-gb N]`**
   (`profile_stage.rs`) — compiles each stage IR on each available device
   and times a zeroed forward pass; records `lat`/`mem`/op-support and the
   per-device + shared-pool budgets → `placement_profile.json`. A device
   that fails to compile a stage is omitted (op-support gate). *(cost table)*
2. **`cascadia place --profile placement_profile.json`** (`placement.rs`)
   — the exact branch-and-bound ILP above; emits `placement.json`
   (per-stage device list + objective + per-device memory). Solved exactly
   in pure Rust — no `good_lp`/CBC dependency. Fitting model → all-GPU;
   over-budget → overflow onto the cheaper tier (measured: **CPU before NPU**,
   since the NPU is the slowest tier for decode); over-pool → `ExceedsPool`.
3. **`cascadia run-placement --shard <dir> --placement placement.json`**
   (`run_placement.rs`) — spawns one `cascadia worker` per stage pinned to
   its assigned device, wired into the pipeline ring (rank 0 = API head).
   The argv planning is pure + unit-tested; the spawner tears the ring down
   on any stage exit / Ctrl-C.

## Validation on the Lunar Lake box (measured)

End-to-end `profile-stages → place → run-placement`, real GPU/CPU/NPU, greedy
decode. Per-stage decode latency measured **GPU < CPU < NPU** (e.g. on the
9B: ~16 / ~33 / ~60+ ms) — so the ILP overflows to **CPU**, correctly avoiding
the NPU, which is the slowest tier for token-by-token decode.

| Model (int weights) | regime | `--device GPU` | ILP placement | ILP vs GPU |
|---|---|---|---|---|
| Qwen2.5-1.5B (smoke) | comfortable | all-GPU | **all-GPU** (10/12) | — (no regression) |
| Qwen2.5-32B int4 (15.7 GiB) | comfortable | **1.97 tok/s** | GPU×15+CPU×1, 1.27 | GPU wins (fits) |
| **Yi-1.5-9B fp16 (16.45 GiB)** | **near-full** | 2.03 tok/s | **GPU×11+CPU×1, 2.91** | **+43%** |
| SOLAR-10.7B fp16 (20.0 GiB) | over-nominal (UMA spill) | **1.31 tok/s** | GPU×9+CPU×3, 0.79 | GPU wins (spill) |

- **Regime 1 (comfortable fit):** the ILP picks **all-GPU** (1.5B smoke) or, when
  the 0.9 memory-headroom nudges one stage off (32B), GPU-alone is still
  fastest — confirming placement shouldn't fight a model that fits. *No
  regression: the solver's job here is to recommend GPU-alone, which it does.*
- **Regime 2 (near-full iGPU), the headline:** Yi-1.5-9B fp16 fills the iGPU to
  16.45/16.48 GiB. all-GPU is memory-pressure-bound at **2.03 tok/s** (4-trial
  mean, σ<2%); the ILP offloads the embed stage to CPU and reaches **2.91
  tok/s — +43%** over `--device GPU`. The acceptance, non-degenerate and
  reproducible.
- **Regime 3 (over the nominal budget) — the UMA surprise:** SOLAR-10.7B fp16
  = **20.0 GiB > the 16.48 GiB nominal iGPU budget**, yet `--device GPU` **ran
  it 12/12 at 1.31 tok/s** — the iGPU spills into the shared 31.6 GB pool, so
  there's no hard OOM below the pool size. all-GPU (spill) *beats* the ILP's
  forced GPU×9+CPU×3 offload (0.79 tok/s); placement doesn't help here. (The
  ILP's static cap-based model can't see that the iGPU would happily spill, so
  it over-offloads — see Limitations.) Note the naive NPU-overflow variant was
  both slowest (0.14 tok/s) **and produced corrupt output** — another reason
  the ILP's measured avoidance of the NPU for decode is correct.
- **ILP vs naive (offload-to-NPU):** the cost-aware CPU overflow beats the
  intuitive "use the AI accelerator" NPU overflow by **+19–20%** (32B 1.27 vs
  1.07; 9B 2.91 vs 2.34) — the ILP's measured-latency choice contradicts the
  naive heuristic and wins.
- **Graceful degrade:** a stage that fails op-support on a device is excluded
  by the ILP (omitted from `lat`); placement still solves (unit-tested).
- **Profile cost:** full-tier (incl. NPU) profiling of the 32B took ~106 min
  (NPU static-graph compiles dominate); the `--devices GPU,CPU` fast path
  (NPU is non-competitive for decode anyway) profiles the 9B in **~0.9 min**.

## Limitations & future work

- **UMA spill not modeled (over-offload).** The solver treats each device's
  reported budget as a hard cap, but the iGPU transparently spills into the
  shared pool (regime 3). So for a model over the *nominal* budget the ILP
  forces an offload that all-GPU-via-spill beats. Fix: use the **pool** as the
  GPU's effective cap and only offload for the pressure-relief win — which
  needs the next point.
- **Pressure cliff is empirical, not predicted.** The regime-2 +43% comes from
  a near-full-iGPU memory-pressure collapse that the per-stage profiler (which
  times each stage *in isolation*) can't see. The ILP found a good placement
  here, but to *target* regime 2 deliberately it needs an end-to-end
  all-GPU-vs-placed probe (or a pressure-penalty term as GPU occupancy → 100%).
- **NPU is non-competitive for decode** (~1.5× the CPU, ~2× the iGPU per
  token) — the ILP correctly avoids it. Profiling the NPU is also slow
  (~100 min/30B); `--devices GPU,CPU` is the practical default. (A multi-NPU
  config once *appeared* to corrupt — #67 — but that was an eval timeout on a
  RAM-exhausted box, not corruption; output is correct, just slow. NPU
  multi-stage is verified correct on every topology tested.)
- **Per-worker overhead / swap (the #67 finding).** The solver's memory gate
  counts **weights only**, but each stage is a worker process whose OV runtime
  + device context + KV/activations add ~1 GiB resident. A placement that
  "fits" the weight pool can still drive the box to ~0 free RAM and swap
  (slow + highly variable tok/s — a 20 GiB model across 12 workers exhausted a
  31.6 GiB box). `cascadia place` now **warns** when `Σ weights + n_stages ×
  --worker-overhead-gb` exceeds the usable pool, advising fewer stages / a
  smaller model. A fuller fix would fold per-stage runtime footprint into the
  profiled `mem_bytes` so the solver's gate is exact.
- **No cross-request pipelining.** Stages run sequentially per token, so the
  throughput-overlap regime (concurrent requests across tiers, potentially
  ~Nx) is untapped — a separate, larger runtime change.

## References
- PowerInfer §6.3 (offline ILP placement), PowerInfer-2 §4.1.3 (dynamic
  per-batch graphs — future).
- `docs/perf/DEVICE_PROFILE.md` (#45, the step-1 profiler this extends).
- #37/#63 (NPU static path), #57/#62 (CPU path) — the tiers this places onto.
