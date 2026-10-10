# sycl-llama engine + `--elastic` device-side mapping — 2026-10-03

## What this is

`crates/cascadia-engine-llamacpp` — a subprocess-backed engine. Cascadia's
existing engines load weights in-process (OpenVINO IRs); this one spawns a
`llama-server` child and proxies its OpenAI `/v1/completions` SSE stream
through the `Engine` trait (`submit`/`step`/`cancel`/`close`).

Purpose: make cascadia's `--elastic` posture meaningful on the **device**
side, which the merged `--elastic` interposer (PR #132) cannot reach —
device allocations bypass malloc, documented "GPU inert" in the PR itself.

## Is it the same as `--elastic`? The paper's own answer

From the elastic-inference paper (`papers/elastic-book-sources/paper`):

> "The obligations are the contract; the mechanisms are free."
> — §"What the protocol does not specify"

> "Device memory bypasses malloc and has no pager, so our mechanisms do not
> transfer; the obligations do. ... the same decoupling applied to weight
> and scratch arenas would be O1/O3. We state this as the protocol's
> portability claim and **have not implemented it**."
> — §"Accelerators: the analogous mechanism"

So `--elastic` means "satisfy the elastic obligations on this substrate,"
not "run the interposer." The mapping:

| engine + device | `--elastic` mechanism | conformance |
|---|---|---|
| OV engines, CPU | `cascadia-elastic` interposer (host pages file-backed) | O1/O3 posture, merged #132 |
| `sycl-llama`, dGPU | `GGML_STREAM_WEIGHTS=1` on child — weights never resident on device, `pread` per layer into a fixed slot pool | **O1 device-side** (weights not resident — stronger than file-backed); O2 open (KV still reserved vs -c); O3 partial (ggml pools) |
| OV engines, GPU/NPU | inert + warning ("device memory has no pager") | none — hard driver/runtime limit |

The warning in `build_builder` states the mapping plainly at launch.

## Usage

```bash
cascadia worker --engine sycl-llama \
  --model /path/to/model.gguf \
  --llama-bin /path/to/llama-server \
  --device SYCL1 --llama-ctx 16384 --llama-ngl 99 \
  --elastic            # -> GGML_STREAM_WEIGHTS=1 on the child
```

The host `--elastic` interposer still activates on the cascadia process as
usual (orthogonal: file-backed host pages). The child inherits `LD_PRELOAD`,
so its host-side allocations get the same posture — the two layers are
additive, not conflicting.

## What's wired vs not

- [x] `EngineKind::SyclLlama` ("sycl-llama"), CLI args `--llama-bin`
      (optional: --llama-bin > $CASCADIA_LLAMA_BIN > PATH),
      `--llama-ctx`, `--llama-ngl`, `--llama-args` (space-split,
      repeatable), `--llama-load-timeout`, `--llama-load-retries`
- [x] `LlamaCppBuilder`: spawn → `/health` poll → `Engine` via SSE proxy;
      child stderr piped + forwarded to our stderr with a 40-line ring
      buffer for error tails; load retry with fresh port (xe reset hangs)
- [x] `--device` mapping: GPU -> SYCL0, GPU.N -> SYCLN, CPU -> `none`
      (-ngl 0), anything else verbatim
- [x] `--elastic` preflight: the resolved binary + `libggml-base*` beside
      it are probed for the `GGML_STREAM_WEIGHTS` marker; a stock build is
      rejected up front
- [x] sampling forwarding: top_p/top_k/seed/penalties/stop sent to the
      child when non-default; SSE socket timeout 300 s
- [x] `cargo check` clean for the crate and `cascadia-cli`; unit + mock
      HTTP engine + spawn-failure tests under `cargo test`
- [x] `cascadia run` passthrough for the llama-* args (RunArgs → worker
      fields wired in cmd_run)
- [x] reproducible build: `scripts/build-llama-stream.sh` (clone +
      `patches/llama.cpp/0001-sycl-stream-weights.patch` + oneAPI icx
      SYCL build), marker verified post-build
- [x] live e2e: `cascadia run --engine sycl-llama --elastic` served
      Qwen3.8-27B correctly through the cascadia API at ~3.2 GB VRAM
      (kernel vram_mm), Qwen2.5-1.5B at ~0.8 GB; child env confirms
      GGML_STREAM_WEIGHTS=1 + inherited LD_PRELOAD
- [x] measured conformance (partial): committed-VRAM floor via kernel
      `vram_mm` (27B: 15.91->3.22 GiB; 1.5B: 1.46->0.79 GiB; 35B-A3B:
      21.68->2.70 GiB); --elastic off runs resident and leaves the child
      env clean; co-tenancy: five 27B elastic instances on one card at
      ~14.9 GiB peak, all correct
- [x] lifecycle: in-engine retry absorbed intermittent child load hangs
      (15/15 cycles, `lifecycle_retry` in docs/perf/sycl-elastic/data.json)
- [ ] `--stream-weights` CLI flag upstream in llama.cpp (env gate today)

## Measured numbers behind this (2026-10-03, lab repo)

15.4 GB Qwen3.8-27B hybrid on a 32 GB B70: VRAM 15.82 -> 2.80 GB
(committed floor ~fixed slot pool), decode 0.59 t/s vs 19.6 resident
(scales as ~8.1 GB/s / model_GB), 64/64 greedy tokens byte-identical vs
unfused stock, 5x 27B instances hot on one card at 26.8 GB total.
Cost disclosure for users: streaming is a capacity posture, decode pays
the model size per token.
