# OpenVINO performance tuning

Cascadia plumbs the OpenVINO plugin properties that materially affect LLM
inference on Intel hardware through the `cascadia worker` CLI and into the
OV-routed engines: `ov-genai`, `ov-runtime`, `gemma4`,
`ov-dist-spec`, and the OV IRs in `sparse-moe`. (`qwen35` is intentionally
excluded — it compiles with a fixed plugin config because some hints, e.g.
`INFERENCE_PRECISION_HINT=f32` / `EXECUTION_MODE_HINT=ACCURACY`, break its
IRs; see `qwen36.rs`.) All flags are opt-in: when unset, the engine builders
behave exactly as before, so tuning never changes default behavior.

> **`sparse-moe`:** general hints reach its OV-compiled experts, but the same
> `f32` / `ACCURACY` hints that break `qwen35` can also fail to compile
> experts backed by f16-only fused kernels. OpenVINO surfaces that as a compile
> error (not silent), so treat those two hints as experimental there and
> benchmark before relying on them.

Each general flag maps to an `ov::AnyMap` entry handed to the OV compile step —
`ov::Core::compile_model` for the raw engines, the `ov::genai::LLMPipeline`
constructor for `ov-genai`. Values are forwarded to OpenVINO, which rejects an
invalid one with its own error message; the two fixed-set flags
(`--ov-performance-mode`, `--ov-execution-mode`) are additionally checked
against their allowed values at the CLI.

## Flags

### General hints

Forwarded to whichever OV-routed engine you select — all except
`qwen35` (setting them there logs a warning and has no effect). `PERFORMANCE_HINT`,
`INFERENCE_PRECISION_HINT`, and `EXECUTION_MODE_HINT` are plugin-agnostic;
`NUM_STREAMS`, `INFERENCE_NUM_THREADS` (CPU-oriented), and
`ALLOW_AUTO_BATCHING` (GPU/AUTO) are only effective on the plugins that own
them — set them for the matching `--device`.

| Flag | OV property | Notes |
|---|---|---|
| `--ov-performance-mode <MODE>` | `PERFORMANCE_HINT` | `LATENCY`, `THROUGHPUT`, or `CUMULATIVE_THROUGHPUT`. `LATENCY` suits single-user decode; `THROUGHPUT` enables `NUM_STREAMS` auto-tuning. |
| `--ov-inference-precision <PREC>` | `INFERENCE_PRECISION_HINT` | `f16`, `bf16`, or `f32`. **Set this on Xe2/Battlemage** — f16/bf16 share XMX throughput, but the default can silently fall back to f32. |
| `--ov-num-streams <N>` | `NUM_STREAMS` | Parallel inference streams. |
| `--ov-num-threads <N>` | `INFERENCE_NUM_THREADS` | Host CPU thread cap. |
| `--ov-allow-auto-batching` | `ALLOW_AUTO_BATCHING` | Enables internal batching on the GPU plugin. |
| `--ov-execution-mode <MODE>` | `EXECUTION_MODE_HINT` | `ACCURACY` or `PERFORMANCE`. `PERFORMANCE` trades a little accuracy for throughput. |

### NPU-only knobs

These are `ov::genai` `LLMPipeline` convenience keys, so they apply **only to
`--engine ov-genai` on an NPU device**. They are silently dropped otherwise —
both when `--device` is not an NPU plugin (case-insensitive
`starts_with("NPU")`, e.g. `NPU` or `NPU.0`) **and** on every non-`ov-genai`
engine (`ov-runtime`, `gemma4`, `ov-dist-spec`, `sparse-moe`), which compile
via raw `ov::Core::compile_model` and cannot consume these keys. Use
`--engine ov-genai` to reach the NPU LLM pipeline that handles static-shape
compilation internally.

| Flag | OV property | Notes |
|---|---|---|
| `--npu-prefill-chunk-size <TOKS>` | `NPUW_LLM_PREFILL_CHUNK_SIZE` | OV 2025.3+. Chunked prefill for dynamic prompt length. |
| `--npu-max-prompt-len <TOKS>` | `MAX_PROMPT_LEN` | Static-shape constraint. |
| `--npu-min-response-len <TOKS>` | `MIN_RESPONSE_LEN` | Static-shape constraint. |

## Recommended settings by hardware

Best-effort starting points — always benchmark on your specific model +
workload.

| Hardware | Recommended flags |
|---|---|
| Xeon CPU-only | `--ov-performance-mode LATENCY --ov-inference-precision bf16` (if AVX-512_BF16) `--ov-num-streams 2` |
| Lunar Lake iGPU (Xe2) | `--ov-performance-mode LATENCY --ov-inference-precision f16 --device GPU` |
| Arc Pro B70 (Battlemage) | `--ov-performance-mode LATENCY --ov-inference-precision f16 --ov-allow-auto-batching --device GPU` |
| NPU 4 (Lunar Lake) | `--engine ov-genai --device NPU --npu-prefill-chunk-size 512` |

## Verifying properties reach the plugin

Set `OPENVINO_LOG_LEVEL=4` to have the OpenVINO plugin log the applied config at
`compile_model` time, then confirm each flag you passed appears in the dumped
property map.

## References

- [OpenVINO — high-level performance hints](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html)
- [OpenVINO — GPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)
- [OpenVINO — NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [OpenVINO 2025.3 release notes — NPU chunked prefill](https://github.com/openvinotoolkit/openvino/releases/tag/2025.3.0)
