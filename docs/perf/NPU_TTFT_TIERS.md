# NPU TTFT at scale: the 1B→70B tier benchmarks

**Status:** tiers 1B/3B/8B/14B measured; 32B (3-box) and 70B (6-box) pipeline
tiers in flight — this doc snapshots method + results so far and will be
finalized with the fleet numbers. Companion to
docs/perf/HYBRID_NPU_CPU.md (the device × model-size matrix and the
big-model NPU routes live there).

## The question

Can devices with NPUs give people faster TTFT in real workflows as models
grow — through 14B on one box and 70B-class on a small fleet — and what does
it cost in memory and deployment complexity?

## Method

- Models: Llama-3.2-1B/3B, Llama-3.1-8B, Qwen2.5-14B, Qwen2.5-32B (3-stage),
  Llama-3.3-70B (6-stage); all NNCF sym-INT4 static exports
  (`--target npu --static-context 1024 --static-prefill-seq 64`).
- Every NPU model is **AOT cross-compiled on a big-RAM Linux server**
  (OpenVINO pip wheel + Intel's standalone `intel-driver-compiler-npu`
  library, `NPU_PLATFORM=4000`) and shipped as an `export_model` blob; boxes
  import via the engine's `.blob` sibling path — **no compiler and no
  compile transient ever runs on a 32 GB box**.
- Fleet boxes run a ~900 MB flat runtime dir (SDK DLLs + `cascadia.exe`,
  no install); TTFT measured short (~31-token) / long (~435-token) prompts,
  greedy, request 2+ of a warm process where noted.

## Results so far (Lunar Lake 258V boxes, 32 GB)

| Model / config | TTFT short | TTFT long | decode tok/s |
|---|---|---|---|
| 1B hybrid NPU→CPU (steady) | **129 ms** | **644 ms** | 18–19 |
| 3B hybrid NPU→CPU (steady) | **321 ms** | **1.58 s** | 6.6–6.8 |
| 8B 2-stage PP all-NPU (1 box, warm e2e incl. 24-tok decode) | 2.0 s | 7.9 s | — |
| 14B tokenwise CPU | 14.0 s | 166.7 s | 2.2–2.7 |
| 14B tokenwise GPU | 9.1 s | 114.4 s | 3.4–3.8 |
| **14B chunked GPU** | **603 ms** | **2.79 s** | 3.4–3.5 |
| 14B tokenwise NPU (from AOT blob) | 9.0 s | 125.0 s | 3.5 |
| 32B 3-box all-NPU pipeline | pending | pending | — |
| 70B 6-box all-NPU pipeline | pending | pending | — |

14B speedups: chunked-GPU is **15× short / 41× long** vs the tokenwise
static-graph baseline on the same device.

## The 14B memory-envelope finding

On a 32 GB box, ONE 14B model fits any single device; TWO do not (except on
the iGPU):

- NPU-resident models cost ≈ **blob bytes = 1.4× INT4** in Level-Zero
  allocations → a 14B pair ≈ 21 GB L0 → first inference dies with
  `ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY`.
- CPU pairs OOM during the second plugin compile (silent `bad_alloc` abort).
- iGPU pairs fit (UMA copies stay ≈ 1× INT4): all-GPU chunked is the 14B
  single-box config. The NPU is the cheapest device to compile FOR (AOT)
  and the most expensive to keep models RESIDENT on.

So the single-box ladder is: ≤8B every config; 14B single-model or iGPU
pairs; beyond that, shard across boxes (or 64 GB-class machines).

## Attribution: is the 41× documented elsewhere?

The *mechanism* is prior art; the *measurement direction* is not:

- Sarathi (arXiv 2308.16369, OSDI'24) coined "chunked prefill" and states
  the weight-fetch-amortization analysis explicitly — but in datacenter
  serving the baseline is whole-prompt prefill that already saturates the
  GPU, so chunking is a scheduling TAX there (their chunk-64 measured ~5×
  slower prefill; vLLM docs likewise warn smaller chunks worsen TTFT).
- Client NPUs ship fixed-chunk static prefill as standard practice with no
  published multipliers: OpenVINO NPUW `NPUW_LLM_PREFILL_CHUNK_SIZE`,
  Qualcomm's 128-token "prompt processor" graphs, AMD's
  `hybrid_opt_chunk_context`, llm.npu's 256-token chunks (its 22.4×/43.6×
  is vs other engines, not vs tokenwise).
- The seq=1 static-graph baseline is real shipping practice (NITRO,
  arXiv 2412.11053, on Intel NPUs), not a strawman — but no one publishes
  "chunked vs token-at-a-time on an integrated GPU". Independent
  corroboration: llama.cpp community tables on this exact iGPU (Arc 140V)
  show a ~22–39× implicit gap between batched prompt processing and
  one-token-at-a-time throughput — our 15–41× approaches that physics
  ceiling. Closest cousins: Apple's Core ML Llama write-up (~85× fixing the
  inverse pad-to-max static pathology) and SqueezeBits' ANE-prefill +
  GPU-decode disaggregation on Apple silicon (same phase-split shape).

## Fleet deployment learnings (all bitten, all fixed)

1. **Transfers need end-to-end integrity, not size checks**: a truncated
   blob fails as `Blob is missing NPU metadata!` (metadata trailer lives at
   the file end); a bit-corrupted one can crash the NPU
   (`ZE_RESULT_ERROR_DEVICE_LOST`). Chunk large files (512 MB parts),
   retry per part with SSH keep-alives (a stalled stream without
   ServerAlive* hung 15 h), reassemble, then **sha256-compare** — see
   experiments/fleet-bench/xfer_blob.sh.
2. **Windows workers must be started detached via WMI**
   (`Invoke-CimMethod Win32_Process Create`): processes spawned through a
   transient ssh session die with the session.
3. **Inline multi-line PowerShell over ssh is unreliable** (silently
   produces nothing); ship `.ps1` files and invoke with `-File`.
4. **Cross-compiled blobs are driver-sensitive per graph**: the same VCL
   (1.32.1) produced blobs that execute (8B, 14B) and one that reproducibly
   kills the device (32B stage-0) on a healthy NPU. Validate every blob
   with one real inference at deploy time; keep the compiler and driver
   generations pinned together.
5. Tailscale DERP relays are fine for pipeline hops (16 ms box↔box) but
   slow for bulk (~1–5 MB/s/stream); parallel part-streams aggregate.

## Reproduction

Scripts under experiments/fleet-bench/; the parity/bench harness is
`tests/static_prefill_parity.rs` (env knobs in its docstring); single-box
and pipeline runners are the `run_*.ps1` files. Blob production:
`export_blobs.py` pattern — `core.compile_model(xml, "NPU",
{"NPU_PLATFORM": "4000"})` then `export_model` into an `io.BytesIO`.
