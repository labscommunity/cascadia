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

## Quickstart

```bash
# 1. build a patched llama-server (clone + patch + oneAPI SYCL build)
scripts/build-llama-stream.sh ~/llama-stream
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
| `--elastic` | off | `GGML_STREAM_WEIGHTS=1` on the child. Before spawn, the resolved binary (and `libggml-base*` next to it) is probed for the `GGML_STREAM_WEIGHTS` marker — a stock build fails fast with an error instead of silently running resident. The host-side `--elastic` interposer still applies to the process as usual. With the default `--elastic-vram auto`, a model that fits runs at stock speed (streaming turns itself off). |
| `--elastic-vram` | `auto` | Resident-weight VRAM budget in GiB for `--elastic`, passed as `GGML_STREAM_VRAM_MB`. `auto` = free device memory − non-streamed weights − 2× largest layer − `GGML_STREAM_RESERVE_MB`; `0` = stream every layer (maximum packing). If the whole model fits, streaming is disabled entirely (stock path, fusion back on). |
| `--device` | `GPU` (run) / `CPU` (worker) | Device mapping: `GPU` → `SYCL0`, `GPU.N` → `SYCLN`, `CPU` → `--device none` + `-ngl 0`; anything else (`SYCL1`, `Vulkan0`, `SYCL0,SYCL1`) is passed verbatim. |
| `--llama-bin` | auto | `llama-server` path. Resolution: flag > `CASCADIA_LLAMA_BIN` > `llama-server` (`llama-server.exe`) on `PATH`. |
| `CASCADIA_LLAMA_BIN` | — | Env fallback for `--llama-bin`. |
| `--llama-ctx` | `4096` | Context size (`-c`). |
| `--llama-ngl` | `99` | GPU layers (`-ngl`); forced to 0 on `--device CPU`. |
| `--llama-args` | — | Raw args appended verbatim; one value per occurrence, split on spaces, repeatable: `--llama-args "-ctk q8_0 -fa on"`. |
| `--llama-load-timeout` | auto | Per-attempt `/health` deadline in seconds. Auto = 60 + 8 per GiB of model file. |
| `--llama-load-retries` | `1` | Extra spawn attempts when the child exits or never becomes healthy; each retry uses a fresh port after a 3 s pause. |
| `GGML_STREAM_MIN_KB` | `256` | (child env, patched build) minimum layer-tensor size that is streamed rather than uploaded. |
| `GGML_STREAM_VRAM_MB` | set by `--elastic-vram` | (child env) resident-weight budget in MiB, or `auto`. |
| `GGML_STREAM_RESERVE_MB` | `2048` | (child env) headroom `auto` leaves for KV + compute buffers. |
| `GGML_STREAM_RESIDENT_LAYERS` | — | (child env) keep the first N layers resident directly; overrides the budget. |

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
- **MoE** → partial budgets help little today: every expert streams per
  token, so decode stays at ~1 t/s until the model is fully resident
  (router-aware streaming is future work).

## Limitations

- Single-stage only (`--total 1`), one request at a time (batch=1).
- SYCL backend only; the streaming patch is a SYCL backend feature.
- Under `--elastic`, fused ops and SYCL graphs are disabled (correctness
  requirement of streaming).
- KV cache is still reserved against the full `-c` context on device —
  streaming removes *weight* residency, not KV.
- MoE models stream *all* experts per token (no router awareness).
- Windows: the fd/pread streaming path is unsupported there.
- Chat turns carry `role` + `content` only (no tool calls / multimodal
  parts); `prompt` remains the fallback for non-chat engines.

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
