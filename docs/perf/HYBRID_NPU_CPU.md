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
release build, 2026-07-14. Every row produced token-identical output
(`static_prefill_parity` asserts it).

**Short prompt (~31 tokens → 1 chunk):**

| Config | TTFT | Decode tok/s | TTFT vs tokenwise |
|---|---|---|---|
| tokenwise `[CPU]` (old static path) | 1550.7 ms | 19.21 | 1× |
| chunked `[CPU]` | 265.2 ms | 18.75 | 5.8× |
| **hybrid `[NPU prefill + CPU decode]`** | **201.8 ms** | 18.55 | **7.7×** |
| tokenwise `[NPU]` (old static path) | 1279.6 ms | 24.74 | 1.2× |
| chunked `[NPU]` (all-NPU) | 203.1 ms | 21.49 | 7.6× |

**Long prompt (~435 tokens → 7 chunks), CPU decode:**

| Config | TTFT | Decode tok/s | TTFT vs tokenwise |
|---|---|---|---|
| tokenwise `[CPU]` | 22,628 ms | 18.95 | 1× |
| chunked `[CPU]` | 1,929 ms | 18.30 | 11.7× |
| **hybrid `[NPU prefill + CPU decode]`** | **942 ms** | 18.76 | **24.0×** |

Free RAM on the box moved 26.4 → 26.3 GB across a full three-leg run —
the 1B model's two variants cost ~1.3 GB resident plus transients.

**2-stage pipeline smoke (same box, loopback, both stages
`--device CPU --prefill-device NPU`):** chunked `[1, r, hidden]` frames +
position frames crossing the wire, each stage's NPU prefill absorbing into
its own ring. 40-token prompt through `/v1/chat/completions`: warm request
end-to-end 0.51 s with `prefill_ms=132`, decode 18.5 tok/s, correct greedy
output, clean teardown. (A pipeline's first-ever request can additionally
wait on downstream stages still finishing their one-time NPU compile —
health reflects stage 0 only; subsequent requests are steady-state.)

## What the numbers say

1. **Chunked prefill is the first-order win** (5.8–11.7× TTFT on a single
   device): it converts prefill from `P` weight-streaming GEMV passes into
   `⌈P/C⌉` wide GEMM passes. This lands even with no second device.
2. **The NPU compounds it on real prompts.** At one chunk the NPU's edge
   over CPU is modest (265 → 202 ms; per-run setup dominates). At 7 chunks
   the NPU prefills **2.05× faster than the CPU** (1929 → 942 ms) and 24×
   faster than the shipping tokenwise path — the compute-bound phase
   belongs on the 48-TOPS engine, and the gap widens with prompt length.
3. **Decode is untouched** (18–19 tok/s in every CPU-decode config): the
   phase boundary costs nothing observable — the shared host ring means no
   KV transfer step exists to pay for.
4. **Honest nuance:** on THIS 1B model the NPU's own decode (24.7 tok/s
   tokenwise) beats CPU decode — small-model decode on NPU4 is fine, and
   all-NPU chunked (`--device NPU`, no `--prefill-device`) is a legitimate
   config at 7.6× TTFT. The CPU-decode split matters for the cases
   three-tier placement measured (larger models under memory pressure,
   NPU decode ~1.5× slower than CPU, THREE_TIER_PLACEMENT.md) and when the
   NPU should stay free between requests. Measure per model; the knobs
   compose either way.

## Constraints & follow-ups

- Static path only. The stateful (`cpu-gpu`) path keeps KV inside OpenVINO
  state, which cannot be shared across two compiled models without new FFI
  (`VariableState::set_state`) — so no stateful phase split yet.
- Greedy sampling only (inherited from the static path).
- Chunk width ≤ `static_context − 1`; the sliding-window semantics under
  prompt overflow match the seq=1 path (unit-tested equivalence).
- Heterogeneous pipelines degrade gracefully: a stage without the prefill
  variant consumes incoming chunks token-by-token (correct, just unamortized).
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
