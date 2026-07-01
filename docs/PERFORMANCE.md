# OpenVINO performance tuning

Cascadia plumbs the OpenVINO plugin properties that materially affect LLM
inference on Intel hardware through the `cascadia worker` / `cascadia run`
CLI and into every OV-routed engine (`ov-genai`, `ov-runtime`,
`ov-dist-spec`, and the head IR in `sparse-moe`). All flags are opt-in: when
unset, the engine builders behave exactly as before, so tuning never changes
default behavior.

Each flag maps 1:1 to an `ov::AnyMap` entry passed to
`ov::Core::compile_model(model, device, props)`. Cascadia does not validate
values — OpenVINO rejects an invalid one with its own error message.

## Flags

### General hints (valid on every device)

| Flag | OV property | Notes |
|---|---|---|
| `--ov-performance-mode <MODE>` | `PERFORMANCE_HINT` | `LATENCY`, `THROUGHPUT`, or `CUMULATIVE_THROUGHPUT`. `LATENCY` suits single-user decode; `THROUGHPUT` enables `NUM_STREAMS` auto-tuning. |
| `--ov-inference-precision <PREC>` | `INFERENCE_PRECISION_HINT` | `f16`, `bf16`, or `f32`. **Set this on Xe2/Battlemage** — f16/bf16 share XMX throughput, but the default can silently fall back to f32. |
| `--ov-num-streams <N>` | `NUM_STREAMS` | Parallel inference streams. |
| `--ov-num-threads <N>` | `INFERENCE_NUM_THREADS` | Host CPU thread cap. |
| `--ov-allow-auto-batching` | `ALLOW_AUTO_BATCHING` | Enables internal batching on the GPU plugin. |
| `--ov-execution-mode <MODE>` | `EXECUTION_MODE_HINT` | `ACCURACY` or `PERFORMANCE`. `PERFORMANCE` trades a little accuracy for throughput. |

### NPU-only knobs

These are **silently dropped unless `--device` names an NPU plugin**
(a case-insensitive `starts_with("NPU")`, e.g. `NPU` or `NPU.0`). Passing an
NPU property to a non-NPU plugin errors, so the gate keeps GPU/CPU runs safe.

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

Set `OV_LOG_LEVEL=4` to have the OpenVINO plugin log the applied config at
`compile_model` time, then confirm each flag you passed appears in the dumped
property map.

## References

- [OpenVINO — high-level performance hints](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html)
- [OpenVINO — GPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)
- [OpenVINO — NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [OpenVINO 2025.3 release notes — NPU chunked prefill](https://github.com/openvinotoolkit/openvino/releases/tag/2025.3.0)
