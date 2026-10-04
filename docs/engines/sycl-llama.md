# sycl-llama — external llama.cpp SYCL engine

`sycl-llama` runs GGUF models on Intel GPUs through llama.cpp: instead of
loading weights in-process, cascadia spawns a `llama-server` child on a
loopback port and proxies its OpenAI SSE endpoints through the normal engine
contract. It is single-stage and batch=1 — a way to put GGUF/SYCL models
behind the same `/v1/*` API as every other engine.

## When to use `--elastic`

On this engine `--elastic` maps to `GGML_STREAM_WEIGHTS=1` on the child: a
patched llama.cpp build never uploads layer weights to the device — they are
`pread` from the GGUF into a fixed slot pool on every forward pass. That makes
it a **capacity mode**, not a speed mode.

- Model fits VRAM and you care about speed → run stock (no `--elastic`).
- Model doesn't fit, or you want several models resident on one card →
  `--elastic`.

Measured on a 32 GB Arc Pro B70 (mean of 3 cold runs):

| model | peak VRAM stock | peak VRAM `--elastic` | decode stock | decode `--elastic` | load stock | load `--elastic` |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M | 1.46 GiB | 0.79 GiB | 224.6 t/s | 8.87 t/s | 6.5 s | 6.5 s |
| Qwen3.8-27B Q4_K_S | 15.91 GiB | 3.22 GiB | 19.35 t/s | 0.59 t/s | 20.5 s | 16.5 s |
| Qwen3.6-35B-A3B Q4_K_XL | 21.68 GiB | 2.70 GiB | 34.2 t/s | 0.29 t/s | 28.5 s | 22.5 s |

Decode under streaming pays the model size per token (~8 GB/s / model_GB), so
expect ~0.6 t/s on a 27B — usable for batch/offline work and model-co-tenancy,
not for interactive chat.

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
| `--elastic` | off | `GGML_STREAM_WEIGHTS=1` on the child. Before spawn, the resolved binary (and `libggml-base*` next to it) is probed for the `GGML_STREAM_WEIGHTS` marker — a stock build fails fast with an error instead of silently running resident. The host-side `--elastic` interposer still applies to the process as usual. |
| `--device` | `GPU` (run) / `CPU` (worker) | Device mapping: `GPU` → `SYCL0`, `GPU.N` → `SYCLN`, `CPU` → `--device none` + `-ngl 0`; anything else (`SYCL1`, `Vulkan0`, `SYCL0,SYCL1`) is passed verbatim. |
| `--llama-bin` | auto | `llama-server` path. Resolution: flag > `CASCADIA_LLAMA_BIN` > `llama-server` (`llama-server.exe`) on `PATH`. |
| `CASCADIA_LLAMA_BIN` | — | Env fallback for `--llama-bin`. |
| `--llama-ctx` | `4096` | Context size (`-c`). |
| `--llama-ngl` | `99` | GPU layers (`-ngl`); forced to 0 on `--device CPU`. |
| `--llama-args` | — | Raw args appended verbatim; one value per occurrence, split on spaces, repeatable: `--llama-args "-ctk q8_0 -fa on"`. |
| `--llama-load-timeout` | auto | Per-attempt `/health` deadline in seconds. Auto = 60 + 8 per GiB of model file. |
| `--llama-load-retries` | `1` | Extra spawn attempts when the child exits or never becomes healthy; each retry uses a fresh port after a 3 s pause. |
| `GGML_STREAM_MIN_KB` | `256` | (child env, patched build) minimum layer-tensor size that is streamed rather than uploaded. |

Sampling knobs (`top_p`, `top_k`, `seed`, `frequency_penalty`,
`presence_penalty`, `stop`) are forwarded to the child only when they differ
from the defaults.

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
- `usage.prompt_tokens` reports `0` on this engine — the prompt is
  tokenized by the child, not by cascadia (completion tokens are still
  counted).

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
- **port already in use** — the engine binds the child to a freshly picked
  loopback port, so this should not happen between retries; only `--api`
  needs a free port you choose.
