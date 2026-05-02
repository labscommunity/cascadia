# c27: GPU plugin property sweep — null result

## Setup
Llama 3.1 8B INT4 on alpha B390 GPU. LLMPipeline plain (no draft), 64-tok
factual prompt, 3 runs per config, best-of-3.

## Results

| Plugin properties | Best tok/s | vs default |
|---|---:|---:|
| (default) | 139.95 | — |
| `GPU_QUEUE_THROTTLE=HIGH` | 139.25 | -0.5% |
| `GPU_HOST_TASK_PRIORITY=HIGH` | 138.79 | -0.8% |
| `GPU_QUEUE_THROTTLE=HIGH;GPU_HOST_TASK_PRIORITY=HIGH` | 135.99 | -2.8% |
| `KV_CACHE_PRECISION=u8;DYNAMIC_QUANTIZATION_GROUP_SIZE=32` | 132.55 | **-5.3%** |
| `KV_CACHE_PRECISION=f16` | 140.17 | +0.2% (noise) |
| `KV_CACHE_PRECISION=u8;DYNAMIC_QUANTIZATION_GROUP_SIZE=64` | 139.40 | -0.4% |
| `INFERENCE_PRECISION_HINT=f16` | 139.41 | -0.4% |
| `NUM_STREAMS=2` | 138.04 | -1.4% |

## Findings

1. **The literature's claim of "GPU_QUEUE_THROTTLE=HIGH + GPU_HOST_TASK_PRIORITY=HIGH gives
   5-15% latency win" is FALSE on Battlemage OV 2026.1.** The combined setting is
   actually 2.8% slower than default, and individually each is within noise.

2. **Explicit `KV_CACHE_PRECISION=u8 + DYNAMIC_QUANTIZATION_GROUP_SIZE=32` is a 5%
   regression**, not the "killer feature" the synthesis billed. Likely interpretation:
   the GPU plugin defaults already use U8 KV cache (since OV 2024.6) but with a
   larger DynQuant group than 32. Setting group=32 explicitly increases per-step
   quantisation overhead for marginal accuracy gain that doesn't help our perf
   metric.

3. **`NUM_STREAMS=2` does nothing for single-user.** It would only help in a
   multi-tenant CB scenario where two requests can be in flight on different
   compute streams — which is what the LLMPipeline scheduler already manages
   internally for batched workloads.

4. **`KV_CACHE_PRECISION=f16` (overriding U8 default) is essentially tied** at
   64-token output. At very long outputs (2K+ tokens) U8 should win on memory
   pressure, but for short factual responses the default wins on simplicity.

## Recommendation
Do not override GPU plugin properties for single-user latency workloads on
Battlemage. The 2026.1 defaults are already optimal. The only knob worth
keeping in tahoma's CLI is `--ov-cache-dir` (kernel-compile cache, not measured
here but quoted as a 5-20s cold-start TTFT win in the synthesis).
