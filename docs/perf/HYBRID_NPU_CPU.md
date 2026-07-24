# Hybrid NPU+CPU execution — chunked prefill on one device, decode on another

**Status:** shipped behind `--prefill-device` / `--static-prefill-seq`
([PR #107](https://github.com/labscommunity/cascadia/pull/107)).
Static-path only (`--engine ov-runtime`, `--target npu` exports).

## What this is

LLM inference has two phases with opposite hardware appetites:

- **Prefill** is compute-bound: one wide forward over the whole prompt. An AI
  PC's NPU is built for exactly this (Lunar Lake NPU4: 48 peak TOPS, a
  136 GB/s fabric port, 12K MACs) while the CPU cluster peaks around 5 TOPS.
- **Decode** is DRAM-bandwidth-bound: one token per forward, weights stream
  from memory every step, arithmetic intensity ~1 MAC/byte. Extra TOPS buy
  nothing; the winner is whoever streams weights with the least overhead. In
  cascadia's static path the CPU has been the better decode device
  (docs/perf/THREE_TIER_PLACEMENT.md measured NPU decode ~1.5× slower than
  CPU per token on these shards).

Until now cascadia's static (NPU) path also prefilled **one token at a time**
(the IR is seq=1 by construction), so prefill both streamed the full stage
weights once per prompt token and ran as GEMVs — the worst of both worlds and
the reason sharded-NPU TTFT was poor.

This feature splits the phases across devices, with the memory traffic
structured deliberately:

1. **Chunked prefill.** `tools/export_shards.py --target npu
   --static-prefill-seq C` emits a second static IR reshaped to a seq=`C`
   query window. The engine consumes the prompt `C` tokens per forward:
   stage weights stream from DRAM **once per chunk instead of once per
   token** (a ~`C`× cut in prefill weight traffic), and the wide matmuls are
   compute-bound — NPU-shaped work.
2. **Per-phase device.** `cascadia worker --device CPU --prefill-device NPU`
   compiles the prefill variant on the NPU and the seq=1 decode variant on
   the CPU.
3. **One KV, in host DRAM.** The exporter pins the prefill variant's past-KV
   length to exactly the decode variant's (`static_context − 1`), so both
   compiled models feed from and absorb into the **same host-side `StaticKv`
   ring**. The prefill→decode handoff costs zero copies beyond the per-input
   memcpy every path already pays; neither device holds device-resident KV.
4. **Phases don't contend.** Prefill (NPU pulling DRAM) and decode (CPU
   pulling DRAM) are temporally exclusive, so the LPDDR5X bus serves one
   phase at a time — measured cross-IP contention on Lunar Lake is mild
   (NPU-victim 1.09–1.17× under CPU load, BIDENT/arXiv 2606.05271), and this
   design sidesteps even that.

The cost: two compiled models hold **two copies of the stage weights**
resident (~2× weight RSS when the devices differ — same trade Intel's NPUW
makes; weight-bank sharing across heterogeneous devices doesn't exist in
OpenVINO today). KV is not duplicated. Budget for it when sizing stages.

Prior art for the pattern: Intel's own NPUW `StaticLLMPipeline` (two static
models — reshaped prefill + kvcache — with host-side KV copies between them),
AMD Ryzen AI "hybrid mode" ("the prefill phase … benefits from running on the
NPU … the decode phase is memory intensive"), llm.npu (arXiv 2407.05858,
22.4× prefill speedup, NPU prefill + CPU decode), PowerInfer-2, HeteroLLM.
Cascadia's twist is doing it **per pipeline stage** in a sharded pipeline,
with the phase boundary riding the existing wire protocol.

## Quickstart

```bash
# 1. Export a static shard WITH the chunked-prefill variant (C=64 here):
python tools/export_shards.py \
  --model unsloth/Llama-3.2-1B-Instruct \
  --output-dir ./l32-1b-npu --num-stages 1 \
  --target npu --static-context 1024 --static-prefill-seq 64
# (or: cascadia shard --target npu --static-prefill-seq 64 ...)

# 2. Serve with prefill on the NPU, decode on the CPU:
cascadia worker --rank 0 --total 1 --engine ov-runtime \
  --model ./l32-1b-npu --device CPU --prefill-device NPU --api :8000

# A/B knobs:
#   --no-chunked-prefill        -> legacy token-at-a-time prefill baseline
#   --prefill-device omitted    -> chunked prefill on --device (amortization only)
```

Parity + phase timing in one shot (the test doubles as the bench):

```bash
CASCADIA_STATIC_SHARDS=./l32-1b-npu CASCADIA_PREFILL_DEVICE=NPU \
  cargo test -p cascadia-engine-openvino --features openvino \
  --test static_prefill_parity --release -- --nocapture
```

## Measured

Lunar Lake AI PC (Core Ultra 7 258V, 32 GB LPDDR5X, Windows), OpenVINO GenAI
2026.1 SDK, Llama-3.2-1B-Instruct INT4 single-stage static shard
(`--static-context 1024 --static-prefill-seq 64`), greedy, 32 new tokens,
release build, 2026-07-14. Every row of that run produced token-identical
output — but the parity check is near-tie-tolerant (see the parity bullet
below), so token-identity is the observed result for these prompts, not a
guarantee: a later 1B + 3B sweep saw near-tie forks as early as token 2.

**Short prompt (~31 tokens → 1 chunk):**

| Config | TTFT | Decode tok/s | TTFT vs tokenwise |
|---|---|---|---|
| tokenwise `[CPU]` (old static path) | 1550.7 ms | 19.21 | 1× |
| chunked `[CPU]` | 265.2 ms | 18.75 | 5.8× |
| **hybrid `[NPU prefill + CPU decode]`** | **201.8 ms** | 18.55 | **7.7×** |
| tokenwise `[NPU]` (old static path) | 1279.6 ms | 24.74 | 1.2× |
| chunked `[NPU]` (all-NPU) | 203.1 ms | 21.49 | 7.6× |

**Long prompt (~435 tokens → 7 chunks), CPU decode** (post-review-fix run —
the skip-unused-logits + buffer-reuse fixes cut hybrid TTFT a further ~27%):

| Config | TTFT | Decode tok/s | TTFT vs tokenwise |
|---|---|---|---|
| tokenwise `[CPU]` | 22,923 ms | 18.89 | 1× |
| chunked `[CPU]` | 1,629 ms | 19.12 | 14.1× |
| **hybrid `[NPU prefill + CPU decode]`** | **687 ms** | 18.80 | **33.4×** |

**Over-window prompt (~1200 tokens > the 1023-slot window)** — the parity-cap
regime: the first 1023 rows chunk, the ~177-token tail steps tokenwise. All
three configs produced identical output in this run (no near-tie fork within
the generated span):

| Config | TTFT | Decode tok/s | TTFT vs tokenwise |
|---|---|---|---|
| tokenwise `[CPU]` | 63,782 ms | 18.02 | 1× |
| chunked `[CPU]` | 13,521 ms | 17.59 | 4.7× |
| **hybrid `[NPU prefill + CPU decode]`** | **11,045 ms** | 17.91 | **5.8×** |

Free RAM on the box moved 26.4 → 26.3 GB across a full three-leg run —
the 1B model's two variants cost ~1.3 GB resident plus transients.

**2-stage pipeline smoke (same box, loopback, both stages
`--device CPU --prefill-device NPU`):** chunked `[1, r, hidden]` frames +
position frames crossing the wire, each stage's NPU prefill absorbing into
its own ring. 40-token prompt through `/v1/chat/completions`: warm request
end-to-end 0.48 s with `prefill_ms=109`, decode ~19 tok/s, correct greedy
output, clean teardown. Warmup now exercises both compiled models on every
stage (including relays), so the first request no longer pays one-time NPU
graph init (0.69 s vs 0.88 s pre-fix; a cold pipeline's first request can
still wait on downstream stages' one-time model COMPILE — health reflects
stage 0 only).

## Device × model-size matrix (2026-07-16)

Same box/SDK/flags; Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B INT4
single-stage static shards (context 1024, prefill chunk 64). TTFT in ms as
short (~31-token) / long (~435-token) prompt. **steady** = request 2+ of a
warm process (`CASCADIA_STATIC_TASKS=2`); a compiled NPU model IMPORTED from
the blob cache defers ~300–430 ms of driver init to its first inference, so
request-1-through-cache TTFTs read that much high (a cold in-process compile
does not pay it — earlier ~500 ms short-prompt hybrid readings were this
artifact, not the prefill).

| Config | 1B short/long | 3B short/long | 8B short/long |
|---|---|---|---|
| tokenwise `[CPU]` | 1,543 / 22,787 | 4,526 / 63,559 | 7,458 / 94,589 |
| chunked `[CPU]` steady | 240 / 1,654 | 649 / 4,360 | 1,582 / 9,161¹ |
| **hybrid `[NPU→CPU]` steady** | **129 / 644** | **321 / 1,582** | — ² |
| tokenwise `[GPU]` | 1,312 / 20,575 | 4,473 / 54,623 | 4,847 / 67,273 |
| chunked `[GPU]` | 74 / 391 | 226 / 1,228 | 357 / 1,410 |
| hybrid `[NPU→GPU]` ³ | 493 / 948 | 734 / 2,009 | — ² |
| tokenwise `[NPU]` | 1,295 / 17,967 | 4,412 / 49,142 | — ² |
| chunked `[NPU]` (all-NPU) | 205 / 1,002³ | 393 / 1,965³ | — ² |

¹ 8B first-request numbers (single runs; steady not probed at 8B).
² Any NPU compile of the 8B stage is infeasible on this 32 GB box: the NPU
compiler's host-side transient exceeded ~23 GB with NOTHING else resident
(free RAM 26.9 → 3.6 GB, run killed by a safety watchdog; the hybrid attempt
with the CPU decode model already resident hit 0.7 GB free). Matches Intel's
">16 GB RAM for >7B models" NPU guidance. The transient scales with stage
weight bytes, so the viable 8B-on-NPU route is pipeline-parallel (2+ stages
≈ 3B-class bytes per stage) — measured working, plus three more routes: see
"Big-model NPU routes" below.
³ First request after a cache import (subtract ~300–430 ms for steady; e.g.
cold-compiled all-NPU long measured 687 ms at 1B).

Decode tok/s across the same runs — the ladder flips with model size:

| Model | NPU | GPU | CPU |
|---|---|---|---|
| 1B | **24.3–25.6** | 20.0–22.6 | 18.4–19.2 |
| 3B | **8.7–9.0** | 7.1–8.2 | 5.4–7.0 |
| 8B | n/a² | **5.8–6.5** | 4.0–4.4 |

Matrix takeaways:

- **GPU is the fastest prefill device at every size** (74–357 ms short,
  391–1,410 ms long) — the phase-split mechanism is device-agnostic, and
  `--prefill-device GPU` (or all-GPU chunked) is the raw-TTFT winner when the
  iGPU is free.
- **Steady NPU prefill beats chunked-CPU prefill 1.9–2.8×**, growing with
  prompt length and model size (1B short 240→129; 3B long 4,360→1,582) — at
  identical decode, since the CPU never sees the prefill.
- **NPU decode stays FASTEST up to 3B on this silicon** (24.5 → 9.0 tok/s
  ladder-topping at 1B and 3B). The "NPU decode ~1.5× slower than CPU" regime
  from three-tier placement (Yi-9B class) is unreachable single-stage on a
  32 GB LNL box — the NPU compile envelope excludes those models first.
- **Greedy parity across kernels/devices**: the chunked/hybrid legs run a
  *different compiled graph* (seq=`C`) from the seq=1 decode graph, so **any**
  leg — same-device CPU/NPU included — can fork at a genuine argmax near-tie
  when the two graphs' floating-point accumulation tips a near-equal top-2
  (both branches coherent, deterministic per config, observed more often as
  model size grows; a 1B + 3B sweep over varied prompts, 2026-07-24, saw forks
  **as early as token 2**, correcting both an earlier "token-exact on CPU/NPU"
  claim and an earlier reading that forks land ~token 30). The host KV state
  itself is byte-identical — proven by the ring-math unit tests
  (`chunked_absorb_matches_sequential` et al.) — so the fork lives in the
  graphs' FP, not the host bookkeeping. The harness therefore tolerates a
  near-tie fork with a loud report (fork index + both texts) and hard-fails
  only if the sequences fork **at the very first decoded token** — the only
  position that reliably points at wrong prefill KV rather than a near-tie
  (near-ties occur too early for a larger prefix guard to hold).
  `CASCADIA_PARITY_SOFT=1` tolerates even that, for pure timing sweeps.

## Big-model NPU routes (2026-07-16, second session)

The matrix's 8B single-stage NPU cells were memory-infeasible: the NPU
compiler's host transient measured **~6–9× the stage's INT4 bytes**
(8B variants: 23.8–37.9 GB; 3B: ~17 GB python-RSS incl. held objects). Four
routes around it — the first three measured on pawan-01 (32 GB LNL):

1. **Pipeline-parallel stages + sequential cache warm (measured, works).**
   A 2-stage 8B export (2.07 GB INT4/stage) compiles per-stage ON-BOX when
   the compiles are serialized: `tests/compile_warm_probe.rs` compiles each
   IR one-at-a-time into `--ov-cache-dir` (~490 s each, floor 7.7 GB free),
   then the pipeline starts all ranks concurrently as pure blob imports.
   Measured warm e2e (24 new tokens): all-NPU **2.0 s** short / **7.9 s**
   long (~435-tok); hybrid NPU→CPU 2.3 s / 8.9 s; steady residency ~12 GB
   (all-NPU) / ~16 GB (hybrid); zero watchdog events. Do NOT serialize
   worker *startup* instead — a non-first rank blocks in transport accept
   before engine load, deadlocking the bring-up; warm the cache.
2. **AOT cross-compile + blob import (measured; removes the on-box spike
   entirely).** The NPU compiler is a userspace library that runs WITHOUT an
   NPU: drop `libnpu_driver_compiler.so` (from the `intel-driver-compiler-npu`
   deb in intel/linux-npu-driver releases) into the OpenVINO package libs as
   `libopenvino_intel_npu_compiler.so`, compile with `NPU_PLATFORM=4000` on
   any big-RAM Linux box, `export_model` (the Python binding needs an
   `io.BytesIO`), ship the blob (~5.8 GB = **1.4×** INT4), import via
   `Runtime::import_blob` (`tests/blob_import_probe.rs`): **6.9–7.5 s import
   on the 32 GB box, free RAM unmoved.** The Linux-VCL→Windows-driver
   handshake passed with OV pinned 2026.1 on both sides. Caveats: CACHE_DIR
   entries are NOT portable (the hash bakes in absolute path + mtime) — use
   export/import; cache *loads* also miss without a device present, so
   export blobs rather than shipping cache dirs; Intel documents offline
   blobs as dev-only (pin versions; `ov::compatibility_check` pre-validates).
3. **NPUW function folding (measured; monolithic 8B compiles on-box).**
   Passing `NPU_USE_NPUW=YES, NPUW_FOLD=YES, NPUW_FUNCALL_FOR_ALL=YES,
   NPUW_ONLINE_PIPELINE=REP, NPUW_DCOFF_TYPE=f16, NPUW_DCOFF_SCALE=YES,
   NPUW_WEIGHTS_BANK=bank0, NPUW_HOST_GATHER=YES` at compile makes NPUW
   detect the 32 repeated decoder blocks and compile ONE function body:
   the monolithic 8B compiled on-box at ~21.5 GB peak (floor 5.4 GB free),
   greedy output matched the baseline in that run (near-tie-tolerant parity),
   chunked NPU prefill TTFT **1.71 s** (short prompt).
   Decode through the folded/DCOFF model is only **1.16–1.19 tok/s** (DCOFF
   expands weights to f16 → 2× decode bytes, plus per-layer funcall
   overhead) — so use folding for PREFILL and decode on CPU/GPU (needs
   per-device plugin-props plumbing, follow-up), or evaluate the
   `NPUW_DQ=YES` INT4-resident path (newer-compiler-gated, unmeasured here).
   These `NPUW_*` keys are internal/unstable — pin the OV version.
4. **Research-validated, unbuilt:** hetero-split prefill (attention on
   CPU/iGPU, FFN-only subgraphs on NPU → only one block ever compiles;
   HeteroLLM measured 8B prefill 247.9 tok/s on a phone-class UMA SoC,
   arXiv 2501.14794); prefix/system-prompt KV caching seeded into the host
   ring (2.2–3.3× TTFT, zero compile cost, chunk-aligned); speculative
   prefill (up to 7.66× TTFT, draft on CPU/iGPU, static shapes preserved,
   arXiv 2502.02789); weightless blobs (`CACHE_MODE=OPTIMIZE_SIZE` +
   `WEIGHTS_PATH` at import — smaller artifact, leaner import, 2025.3+).

Sizing rule: budget **~6–9× a stage's INT4 bytes** of free host RAM for any
on-box NPU compile (or move the compile off-box / behind NPUW folding), and
**~1.4×** for a blob import.

## What the numbers say

1. **Chunked prefill is the first-order win** (5.8–11.7× TTFT on a single
   device): it converts prefill from `P` weight-streaming GEMV passes into
   `⌈P/C⌉` wide GEMM passes. This lands even with no second device.
2. **The NPU compounds it on real prompts.** Steady-state (request 2+ of a
   warm process — see the matrix section's import-init note) the NPU
   prefills **1.9–2.8× faster than the chunked CPU** at both 1B and 3B,
   the edge growing with prompt length and model size, up to **35–40× over
   the shipping tokenwise path** — the compute-bound phase belongs on the
   48-TOPS engine.
3. **Decode is untouched** (18–19 tok/s in every CPU-decode config): the
   phase boundary costs nothing observable — the shared host ring means no
   KV transfer step exists to pay for.
4. **Honest nuance (revised by the device matrix above):** NPU decode is
   the FASTEST decode on this silicon up to 3B, and all-NPU chunked
   (`--device NPU`, no `--prefill-device`) or all-GPU chunked are legitimate
   single-device configs — all-GPU wins raw TTFT outright. What the
   NPU→CPU split uniquely buys is the best CPU-decode TTFT while leaving
   BOTH accelerators idle between phases: prefill borrows the NPU for
   ~130 ms–1.6 s and decode taxes only the CPU — the right shape when the
   iGPU is busy (demos, display, another model), when the NPU must stay
   available (system AI), for multi-tenant boxes, or for battery (the NPU
   is the lowest-power engine for the compute burst). At 8B-class the
   matrix is blunter: no NPU config compiles on 32 GB (envelope note
   above), so the accelerated configs are GPU-chunked or CPU-chunked, and
   the "NPU decode slower at big models" regime (three-tier placement,
   Yi-9B under memory pressure) never arises single-stage. Measure per
   model; the knobs compose every way.

## Constraints & follow-ups

- Static path only. The stateful (`cpu-gpu`) path keeps KV inside OpenVINO
  state, which cannot be shared across two compiled models without new FFI
  (`VariableState::set_state`) — so no stateful phase split yet.
- Greedy sampling only (inherited from the static path).
- Chunks are capped at the KV window: only rows whose absolute position stays
  ≤ `static_context − 1` run chunked; an over-window prompt tail steps
  tokenwise through the decode model (a chunk-wide mask cannot express
  per-token eviction), so the cap adds no divergence beyond the seq=1 path's
  own eviction in any regime (greedy tokens can still fork at an argmax
  near-tie — see `assert_parity`). So the chunked speedup applies to the first
  `static_context − 1` prompt tokens — size `--static-context` to your
  prompts.
- Very short prompts (≲ C/8 tokens) may not gain: they pay one padded C-wide
  forward (incl. C vocab projections on head stages) versus a handful of
  cheap seq=1 steps. Measured 31-token prompts still won 5.8–7.7×; a
  3-token prompt may not. `--no-chunked-prefill` is the lever if a workload
  is dominated by tiny prompts.
- Heterogeneous pipelines degrade gracefully: a stage without the prefill
  variant consumes incoming chunks token-by-token (correct, just
  unamortized), and a stage with a NARROWER window than an upstream sender
  sub-chunks the incoming frame instead of erroring.
- `profile-stages` counts both IR variants' bytes into a stage's memory —
  accurate for the default single-device chunked runtime (both models
  resident), conservative (over-counted) for `--no-chunked-prefill` or the
  hybrid split; the placement pipeline does not yet model the phase split at
  all (see follow-ups).
- Follow-ups: phase-aware placement (per-device prefill_ms + decode_ms in
  `profile-stages`, two-device assignments in `place`), zero-copy shim
  tensors (4 KB-aligned host buffers import into Level-Zero on UMA),
  NPU-verify speculative decoding on the static ring, gemma4 static path.

## References

- Issue #37 / PR #63 — the static-KV NPU path this builds on.
- docs/NPU_SHARDING.md — export + ring + wire-position contract.
- docs/perf/THREE_TIER_PLACEMENT.md — per-stage device economics (#41).
- Intel NPUW static LLM pipeline (OpenVINO `llm_compiled_model.cpp`);
  `NPUW_LLM_PREFILL_CHUNK_SIZE` (OV 2025.3+).
- AMD Ryzen AI hybrid OGA flow (ryzenai.docs.amd.com/en/latest/hybrid_oga.html).
- llm.npu — arXiv 2407.05858; HeteroLLM — arXiv 2501.14794; PowerInfer-2 —
  arXiv 2406.06282; BIDENT — arXiv 2606.05271 (LNL cross-IP contention).
- Intel Lunar Lake AI accelerators deck (NPU4: 48 TOPS, 136 GB/s IP bandwidth).
