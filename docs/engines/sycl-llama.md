# sycl-llama — external llama.cpp SYCL engine

`sycl-llama` runs GGUF models on Intel GPUs through llama.cpp: instead of
loading weights in-process, cascadia spawns a `llama-server` child on a
loopback port and proxies its OpenAI SSE endpoints through the normal engine
contract. It is single-stage and batch=1 — a way to put GGUF/SYCL models
behind the same `/v1/*` API as every other engine.

## When to use `--elastic`

On this engine `--elastic` maps to `GGML_STREAM_WEIGHTS=1` plus a
resident-weight budget on the child (`--elastic-vram`, default `auto`): a
patched llama.cpp build keeps the layers that fit in VRAM resident and
`pread`s the rest from the GGUF into a fixed slot pool on every forward
pass. When the whole model fits, streaming turns itself off and the model
runs at stock speed — so `--elastic` is safe to leave on.

- One model per card → `--elastic` (default `auto`): stock speed when it
  fits, only the overflow streams when it doesn't.
- Several models on one card, or a model that doesn't fit at all →
  `--elastic` with an explicit `--elastic-vram` budget per instance (`0` =
  maximum packing, every layer streamed).
- Decode drops with the streamed share: ~7 GB/s of streamed weights per
  token. `--elastic-vram 0` is the capacity mode — 0.55 t/s on the 27B.

Measured on a 32 GB Arc Pro B70 (`--elastic-vram 0`, i.e. fully streamed;
see [Choosing `--elastic-vram`](#choosing---elastic-vram) for partial
budgets):

| model | peak VRAM stock | peak VRAM `--elastic` | decode stock | decode `--elastic` | load stock | load `--elastic` |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M | 1.37 GiB | 0.69 GiB | 218.4 t/s | 8.52 t/s | 3.0 s | 4.0 s |
| Qwen3.8-27B Q4_K_S | 15.87 GiB | 2.78 GiB | 19.29 t/s | 0.55 t/s | 18.4 s | 14.9 s |
| Qwen3.6-35B-A3B Q4_K_XL | 21.50 GiB | 2.59 GiB | 78.34 t/s | 0.25 t/s | 27.5 s | 20.4 s |

Decode under full streaming pays the model size per token — usable for
batch/offline work and model co-tenancy; with a partial budget the resident
share decodes at near-stock speed (see the sweep table below).

**Reliability:** 15/15 load/unload cycles succeeded with the default retry
configuration — one cycle needed the in-engine retry, during a real xe engine
reset — and VRAM returned to baseline on every cycle (see
`docs/perf/sycl-elastic/data.json` → `lifecycle_retry`).

## Serving vs parking

What streaming buys depends on whether the GPU has its own memory.

- **Discrete cards** (Arc Pro B70, A770, dGPU class): a partial budget is a
  real VRAM dial. Streamed layers re-read over PCIe each token; resident
  layers decode at stock speed, so `--elastic-vram` trades decode rate for
  how much of the card is left for other work — the serving knob when the
  model doesn't fit or shares the card.
- **UMA / integrated GPUs** (Arc B390, Lunar Lake iGPU): the "device pool"
  and host memory are the same DRAM, so streaming re-reads weights inside
  the memory the GPU already computes from — it buys accounting isolation,
  not bandwidth. Measured by t8 on a B390 iGPU (7B, fully streamed):
  `10.4 -> 0.52 tok/s`, while freeing ~82% of the accounted pool. The
  pinned-host variant (streamed weights mapped in-place) ran `6.3 tok/s`
  but frees nothing on UMA, and that device has no system-USM path.
  Reproduced on a second UMA part, the Arc 140T iGPU (Arrow Lake-H,
  Qwen2.5-7B-Instruct Q4_K_M, `-ctk/-ctv q8_0`, `-fa on`, `--parallel 1`):
  resident `14.97 tok/s` -> half-resident (14/28 layers) `0.66` -> fully
  streamed `0.35 tok/s`, while peak shared-memory usage dropped
  `4628 MiB -> ~1045 MiB`.
  Recommendation: for serving use `auto` — a model that fits takes the
  stock resident path — and treat streaming as a *parking* mode (model
  kept loadable under a small footprint while other work owns the DRAM).
  The build prints `stream-weights: warning: integrated GPU (shared
  memory): ...` whenever streaming actually activates on an iGPU. The
  device is classified with the Level Zero `ZE_DEVICE_PROPERTY_FLAG_INTEGRATED`
  property when the driver reports it, and falls back to the SYCL
  `host_unified_memory` property on stacks where the L0 device-type probe
  is unavailable (OpenCL-adapter Windows builds).
  The stacked PR [#172](https://github.com/labscommunity/cascadia/pull/172)
  explores `SYCL_Host` buffer placement as an alternative on UMA.

## Quickstart

```bash
# 1. build a patched llama-server (clone + patch + oneAPI SYCL build)
scripts/build-llama-stream.sh ~/llama-stream
#    Windows: scripts\build-llama-stream-windows.bat (VS 2022 Build Tools
#    + Intel oneAPI; verified on an Arc B390 iGPU)
export CASCADIA_LLAMA_BIN=~/llama-stream/build/bin/llama-server

# 2. the child needs the oneAPI runtime libs on LD_LIBRARY_PATH
source /opt/intel/oneapi/setvars.sh

# 3. serve a GGUF through cascadia
cascadia run /path/to/model.gguf --engine sycl-llama --elastic --api 127.0.0.1:8080

# 4. talk to it like any cascadia engine. The "model" field is required and
#    must equal the id /v1/models reports (the model path's basename by
#    default, or --served-model-name on the worker):
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"model.gguf","messages":[{"role":"user","content":"hi"}]}'
```

`--device` defaults to `GPU`, which maps to `SYCL0` — no flag needed on a
one-GPU box.

## Flags

| Flag / env | Default | Meaning |
|---|---|---|
| `--engine sycl-llama` | — | Select this engine. |
| `--elastic` | off | `GGML_STREAM_WEIGHTS=1` on the child. Before spawn, the resolved binary (and `libggml-base*` next to it) is probed for the `GGML_STREAM_WEIGHTS` marker — a stock build fails fast with an error instead of silently running resident. The host `--elastic` interposer is NOT activated for this engine and is scrubbed from the child's environment (see below). With the default `--elastic-vram auto`, a model that fits runs at stock speed (streaming turns itself off). |
| `--elastic-vram` | `auto` | Resident-weight VRAM budget in GiB for `--elastic`, passed as `GGML_STREAM_VRAM_MB`. `auto` = free device memory − non-streamed weights − 2× largest layer − `GGML_STREAM_RESERVE_MB`; `0` = stream every layer (maximum packing). If the whole model fits, that model streams nothing; fused ops and SYCL graphs come back on once no loaded model streams. The decision is per load: a draft model (`--llama-args "-md draft.gguf"`) that fits does not change the main model's budget, and the reverse. |
| `--elastic-share` | — | Expected co-tenant count, forwarded as `GGML_STREAM_VRAM_SHARE` when `--elastic` is on and `--elastic-vram` is `auto`. The child then caps its automatic resident-weight budget at `min(free − overhead, (total − overhead) / N)` so N instances loaded in sequence each target a 1/N share of the card. Warns when combined with an explicit `--elastic-vram` or set without `--elastic`; `GGML_STREAM_RESIDENT_LAYERS` still wins. This is a load-time weight cap only — it does not rebalance KV cache, expert caches, or running instances. |
| `--llama-mtp` | off | Speculative decoding with the model's own MTP (nextn) head: `--spec-type draft-mtp` on the child. The draft runs against the already loaded weights (no second model). Needs a GGUF with nextn layers; a model without them fails at load. See "Faster decode and prefill". |
| `--device` | `GPU` (run) / `CPU` (worker) | Device mapping: `GPU` → `SYCL0`, `GPU.N` → `SYCLN`, `CPU` → `--device none` + `-ngl 0`; anything else (`SYCL1`, `Vulkan0`, `SYCL0,SYCL1`) is passed verbatim. |
| `--llama-bin` | auto | `llama-server` path. Resolution: flag > `CASCADIA_LLAMA_BIN` > `llama-server` (`llama-server.exe`) on `PATH`. |
| `CASCADIA_LLAMA_BIN` | — | Env fallback for `--llama-bin`. |
| `--llama-ctx` | `4096` | Context size (`-c`). |
| `--llama-ngl` | `99` | GPU layers (`-ngl`); forced to 0 on `--device CPU`. |
| `--llama-args` | — | Raw args appended verbatim; one value per occurrence, split on spaces, repeatable: `--llama-args "-ctk q8_0 -fa on"`. Reserved flags the engine owns are rejected at startup — `-m`/`--model`/`-mu`/`--model-url`/`-hf`/`-hfr`/`--hf-repo` (use the positional MODEL), `--host`/`--port` (engine picks loopback), `--device`/`-dev` (use `--device`), `-ngl`/`--gpu-layers`/`--n-gpu-layers` (use `--llama-ngl`), `-c`/`--ctx-size` (use `--llama-ctx`). llama-server's parser is last-wins, so without the check a stray `--port` or `-m` would silently shadow the engine's own. |
| `--llama-load-timeout` | auto | Per-attempt `/health` deadline in seconds. Auto = 60 + 8 per GiB of model file. |
| `--llama-load-retries` | `1` | Extra spawn attempts when the child exits or never becomes healthy; each retry uses a fresh port after a 3 s pause. |
| `GGML_STREAM_MIN_KB` | `256` | (child env, patched build) minimum layer-tensor size that is streamed rather than uploaded. |
| `GGML_STREAM_VRAM_MB` | set by `--elastic-vram` | (child env) resident-weight budget in MiB, or `auto`. |
| `GGML_STREAM_RESERVE_MB` | `2048` | (child env) headroom `auto` leaves for KV + compute buffers. |
| `GGML_STREAM_RESIDENT_LAYERS` | — | (child env) keep the first N layers resident directly; overrides the budget. Passed through to the child with a warning when set. |
| `GGML_STREAM_VRAM_SHARE` | set by `--elastic-share` | (child env) co-tenant divisor for the automatic budget (strict decimal, 1..=u32::MAX; invalid values fail the load). With no flag, an ambient value passes through untouched. |
| `GGML_STREAM_READ_THREADS` | `4` | (child env) reader-pool size for streamed fills: file reads happen on worker threads while an in-order-queue host task gates each H2D copy on the fill completing. `0` = inline reads on the dispatch thread (old behavior). The patch clamps the value to `0..16` (negatives to 0, anything > 16 to 16). |
| `GGML_STREAM_STAGING_BUFS` | `8` | (child env) pinned staging-ring slots per device, capped at 64. |
| `CASCADIA_EXPERT_CACHE_MB` | `0` | (cascadia env) forwarded as `GGML_STREAM_EXPERT_CACHE_MB`: hot-expert device cache for router-aware MoE streaming (0002). |

Without `--elastic`, ambient `GGML_STREAM_WEIGHTS`, `GGML_STREAM_VRAM_MB`,
`GGML_STREAM_VRAM_SHARE` and `GGML_STREAM_RESIDENT_LAYERS` are dropped
from the child's environment, so `GGML_STREAM_WEIGHTS=1 cascadia run`
still runs resident.
With `--elastic` the engine sets `GGML_STREAM_WEIGHTS=1` and
`GGML_STREAM_VRAM_MB` itself, and a set `GGML_STREAM_RESIDENT_LAYERS`
passes through with a warning since it overrides the budget.
`GGML_STREAM_VRAM_SHARE` is set from `--elastic-share` when the budget
is automatic, or passes through untouched when the env already carries it. The host
allocator interposer never reaches the child either:
`CASCADIA_ELASTIC_ACTIVE` and `ELASTIC_*` are removed, and `LD_PRELOAD`
is filtered down to non-`libcascadia_elastic.*` entries (a user's own
preloads survive; the var drops only when nothing remains).

If cascadia dies — including by `SIGKILL` — the child dies with it: on
Linux the spawn installs `PR_SET_PDEATHSIG` and the child is spawned from
a dedicated long-lived thread, so a retired blocking-pool thread can't
free the child early; on Windows every child is assigned to a
`KILL_ON_JOB_CLOSE` Job Object. An orphaned llama-server can't hold
VRAM or its loopback port.

Sampling knobs (`top_p`, `top_k`, `seed`, `frequency_penalty`,
`presence_penalty`, `stop`) are forwarded to the child only when they differ
from the defaults.

## Choosing `--elastic-vram`

`--elastic-vram` is the resident-weight budget: the first N layers stay on
the GPU, the rest stream from the GGUF every token. `auto` picks N from the
device's free memory; `0` streams everything.

Measured sweep (32 GB Arc Pro B70, `-c 4096`, KV q8_0, `-fa on`):

| `--elastic-vram` | resident layers | streamed per token | peak VRAM | decode |
|---|---|---|---|---|
| **Qwen3.8-27B** | | | | |
| 0 | 0/65 | 13087 MiB | 2.78 GiB | 0.55 t/s |
| 2 | 10/65 | 11167 MiB | 4.63 GiB | 0.62 t/s |
| 4 | 22/65 | 9051 MiB | 6.66 GiB | 0.76 t/s |
| 8 | 42/65 | 5074 MiB | 10.52 GiB | 1.36 t/s |
| 10 | 53/65 | 2874 MiB | 12.66 GiB | 2.31 t/s |
| 12 | 62/65 | 821 MiB | 14.70 GiB | 8.35 t/s |
| auto / 40 | model fits | 0 | 15.87 GiB | 19.3 t/s (stock) |
| **Qwen3.6-35B-A3B** | | | | |
| 0 | 0/40 | 20279 MiB | 2.59 GiB | 0.25 t/s |
| 4 | 7/40 | 16662 MiB | 6.06 GiB | 0.30 t/s |
| 10 | 20/40 | 10141 MiB | 12.42 GiB | 0.47 t/s |
| 16 | 32/40 | 4118 MiB | 18.31 GiB | 1.12 t/s |
| auto | model fits | 0 | 21.50 GiB | 77.5 t/s (stock) |

Guidance:

- **One model per card** → leave the default `auto`; it keeps as much
  resident as fits and disables streaming entirely when the whole model
  fits.
- **Several models on one card** → give each an explicit budget so the
  budgets plus ~3 GiB per instance (KV, compute buffers, stream slots at
  these settings) stay under the card size. `auto` is
  first-come-first-served: in a 3x 27B test the third instance found no
  room, went fully streamed, and the driver evicted ~11 GiB of the other
  instances' weights to host memory.
- **MoE** → router-aware expert streaming is on when `--elastic` is: only
  the experts the router selects are read per token (detected
  automatically, any MoE). Measured on Qwen3.6-35B-A3B: fully streamed
  decode 0.25 -> 2.60 t/s at 1.89 GiB peak; a partial budget adds resident
  layers on top. Set `CASCADIA_EXPERT_CACHE_MB=<MiB>` to pin a hot-expert
  cache on the device (default 0 keeps the resident-layer budget exact).

## Faster decode and prefill

Measured on the Arc Pro B70 (Qwen3.8-27B UD-Q4_K_S, ctx 4096, KV q8_0,
`-fa on`, 3 chat prompts x 64 tokens; median of 3 fresh server runs, all
runs within 2.5% of the median). Chart: [fig17](../perf/sycl-elastic/fig17_mtp.png).

| `--elastic-vram` | decode | with `--llama-mtp` | peak VRAM cost | MTP tokens accepted |
|---|---|---|---|---|
| resident (fits) | 19.0 t/s | **26.5 t/s** (+39%) | +0.76 GiB | 111/149 |
| 12 (62/65 resident) | 18.6 t/s | 19.3 t/s (+4%) | +0.59 GiB | 111/149 |
| 8 (42/65) | 0.80 t/s | **2.39 t/s** (3.0x) | +1.13 GiB | 111/149 |
| 0 (fully streamed) | 0.67 t/s | **2.02 t/s** (3.0x) | +1.12 GiB | 110/151 |

- **`--llama-mtp` is the largest decode lever under streaming.** Every
  forward pass re-reads the streamed weights; MTP drafts several tokens
  with the model's own nextn head and the target verifies them in one
  pass, so the weight traffic is shared by every accepted token. The gain
  is smallest at 62/65, where the two streamed layers already stay pinned
  in their slots and decode runs at resident speed.
- Output: the target model decides every token, but verification runs as a
  small batch, so greedy text can flip at a near-tie, like any batch-size
  change. In the n=3 sweep, MTP text matched plain decode on 11 of 12
  prompt/arm pairs; the miss was the fully streamed arm.
- MTP also costs VRAM: the draft context's KV and compute buffers add
  0.6-1.1 GiB on top of the budget, so leave that headroom.
- `--llama-args "--spec-type ..."` is rejected next to `--llama-mtp`:
  llama-server appends spec types instead of replacing them, so both modes
  would run. `--spec-draft-n-max` and other draft knobs stay allowed.
- Keeping the MTP layer resident ahead of the budget was measured and
  rejected: no gain at 0 or 8 GiB, and at 12 GiB it costs a regular layer
  (19.8 → 9.7 t/s).
- MoE (Qwen3.6-35B-A3B) ships no nextn layers, so `--llama-mtp` does not
  apply. For **streamed MoE prefill**, a larger micro-batch amortizes the
  expert reads: `--llama-args "-ub 2048"` takes a 2,444-token prompt from
  185 to 265 t/s (+43%) at +0.3 GiB VRAM. Do not raise it for dense
  models: on the 27B it is 12-13% slower (resident 194 → 168 t/s, fully
  streamed 171 → 150 t/s), and resident MoE is 18% slower.

## Limitations

- Single-stage only (`--total 1`), one request at a time (batch=1).
- SYCL backend only; the streaming patch is a SYCL backend feature.
- Fused ops and SYCL graphs are disabled while any loaded model streams
  (correctness requirement of streaming); a process whose models all fit
  keeps them.
- KV cache is still reserved against the full `-c` context on device —
  streaming removes *weight* residency, not KV.
- MoE expert residency is opt-in (`CASCADIA_EXPERT_CACHE_MB`); by default
  selected experts are re-read per token (correct, ~10x faster than
  streaming every expert, still slower than resident).
- Windows: supported. Weight slices are read with a positioned `ReadFile`
  on the file's OS handle (no POSIX `pread`). Verified on two UMA iGPUs:
  an Arc B390 (Panther Lake, oneAPI 2026.0 + MSVC 19.44) and an Arc 140T
  (Arrow Lake-H, same dep-pack binaries). Covered there: build,
  streaming + partial streaming, greedy-token parity, the iGPU warning,
  the Job Object child cleanup (`taskkill /F` and normal exit both reap
  `llama-server.exe`), and OpenAI tool calls end to end — see
  `docs/perf/sycl-elastic/fig9_windows.png`. Pass `-c` explicitly when
  streaming: with streamed weights the default context auto-fit sees the
  freed memory as free and may pick the model's full train context (large
  f32 KV). cascadia always passes `--llama-ctx`. The router-aware MoE path
  uses the same positioned-read primitive and has not yet been exercised on
  Windows.
- Chat turns carry `role` + `content` + tool-call fields (`tool_calls`,
  `tool_call_id`, `name`) and request-level `tools` in the OpenAI wire
  form; the child runs with `--jinja` so tool calls render through the
  model's template (a `--no-jinja` in `--llama-args` overrides). `--jinja`
  is on for *every* request, not just tool requests: every prompt renders
  through the GGUF's embedded jinja chat template, so a model whose template
  is broken fails at load or request time. `--llama-args "--no-jinja"` falls
  back to llama-server's built-in template detection (tool calls are then
  unavailable). `tool_choice` is honored: `"auto"` / `"required"`
  are forwarded verbatim; `"none"` sends the request without `tools`
  (llama-server would still render them into the prompt, and the model
  then writes a call as plain text); a named choice
  (`{"type":"function","function":{"name":N}}`) is sent as `"required"` with
  `tools` narrowed to that one function (llama-server itself parses only the
  string forms, silently treating an object as `"auto"`). llama-server's
  `"required"` is weaker than OpenAI's: its grammar allows free text before
  the call, so a model can spend `max_tokens` on text and return
  `finish_reason: "length"` with no call (seen once with Qwen2.5-1.5B on an
  Arc B390, named choice). An unknown string,
  a malformed object, or a name absent from `tools` is rejected with a 400
  before the request reaches the engine. The child's streamed `delta.tool_calls` fragments
  are re-assembled and emitted as `<tool_call>` text before the final chunk,
  so streaming and non-streaming clients get the same structured
  `tool_calls` the API produces for other engines. Multimodal parts are not
  plumbed; `prompt` remains the fallback for non-chat engines.

## Doctor

`cascadia doctor` has a `sycl-llama` section: it resolves the binary (a
miss is informational, with the build-script hint), runs `--version` with
a 10 s timeout (a `shared libraries` failure points at the missing
oneAPI runtime), reports the weight-streaming preflight result, and lists
`--list-devices` output.

## Troubleshooting

- **`this llama-server build has no weight-streaming support; --elastic
  would not reduce VRAM`** — the preflight found a `libggml-base` beside the
  binary without the `GGML_STREAM_WEIGHTS` marker, i.e. a stock build. Build
  a patched one (`scripts/build-llama-stream.sh`) or drop `--elastic`. If no
  `libggml-base` is found at all (static build, unusual layout) the engine
  warns `could not verify weight-streaming support` and continues.
- **`llama-server: error while loading shared libraries: libsvml.so
  (or libsycl.so / libur_loader.so)`** — the child cannot find the oneAPI
  runtime: `source /opt/intel/oneapi/setvars.sh` in the shell that launches
  cascadia (the engine forwards the child's stderr to yours, so the missing
  library line is visible).
- **`retrying llama-server load ...`** — the child exited or never became
  healthy inside the deadline; the engine respawns on a fresh port
  (`--llama-load-retries`, default 1 extra attempt). Intermittent load hangs
  correlate with xe copy-engine resets: check
  `dmesg | grep -i 'engine reset'`. The final error includes the last ~20
  child stderr lines.
- **intermittent load hang (child never becomes healthy)** — on this xe
  host, model load occasionally stalls at buffer clear
  (`dmesg` shows `Engine memory CAT error ... class=bcs` followed by a
  `guc_exec_queue_timedout_job` reset); seen on stock arms too — 7 of 42
  starts in the sweep. The engine's load timeout + retry recovers; lower
  `--llama-load-timeout` to fail over faster.
- **port already in use** — the engine binds the child to a freshly picked
  loopback port, so this should not happen between retries; only `--api`
  needs a free port you choose.
- **benchmarking on Linux xe hosts** — avoid `echo 2|3 > /proc/sys/vm/
  drop_caches`: dropping inode caches before Intel GPU runtime init leaks
  ~1 GiB of kernel memory per process start (observed kernel 7.0.0-28,
  compute-runtime 26.05). Evict the model file only (`posix_fadvise
  DONTNEED`) for cold-cache runs.
