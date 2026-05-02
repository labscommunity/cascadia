# c30: concurrent NPU + GPU multi-model serving on charlie (Lunar Lake)

## Hypothesis
Discovery #4 says NPU runs Llama 3.2 1B at 53-91% of GPU speed. Putting an
8B on the GPU and a 1B on the NPU concurrently should deliver aggregate
throughput close to the sum of solo throughputs, since the two devices
have separate compute pools.

## Setup
- charlie 140V (Lunar Lake) — has GPU and NPU 4 (~48 TOPS).
- 8B target: `C:\cascadia\models\llama-3.1-8b-int4` on GPU.
- 1B target: `C:\cascadia\models\srang992-llama-3.2-1b-int4-ov` on NPU.
- 64-token output, 1 request per device.
- Both `LLMPipeline` instances created and warmed up, then we run either
  one alone (`solo-gpu` / `solo-npu`) or both concurrently in `threading.Thread`.

## Results

| Mode | GPU 8B tok/s | NPU 1B tok/s |
|------|------------:|-------------:|
| solo-gpu (with NPU pipe also loaded but idle) | 86.15 | — |
| solo-npu (with GPU pipe also loaded but idle) | — | 41.84 |
| **concurrent** | **80.48 (-7%)** | **37.59 (-10%)** |

## Reference (single-pipe baselines from c1, c26 — for comparison)
- 8B GPU plain LLMPipeline single-pipe: 91.14 tok/s (c1-2)
- 1B NPU plain LLMPipeline single-pipe: 112.89 tok/s (c26)

## Findings

1. **Loading both pipes has a non-trivial cost EVEN BEFORE running concurrently.**
   - 8B on GPU: 91 → 86 tok/s (-6%) once an NPU pipe is *also loaded but idle*.
   - 1B on NPU: 113 → 42 tok/s (**-63%!**) once a GPU pipe is *also loaded
     but idle*. This is a major surprise. Likely cause: the OV core shares
     resources between plugins (plugin scheduler, shared memory pool, or
     the per-process Python GIL holding) and the NPU plugin is more
     sensitive to this contention than the GPU plugin.
2. **Running both concurrently costs each device an additional 7-10%** on
   top of the both-loaded baseline. The marginal cross-talk during inference
   itself is small.
3. **Aggregate workloads served:**
   - Sequential (serial): 64/86 + 64/42 = 0.74 + 1.52 = 2.26 s
   - Concurrent: max(64/80, 64/38) = max(0.80, 1.68) = 1.68 s
   - **Concurrent wins by ~34% on aggregate throughput** even with the
     both-loaded NPU degradation, because the NPU's wall-clock work is
     hidden under the GPU's.

## Implications

- For tahoma multi-model serving on a single Intel AI PC: **concurrent
  NPU+GPU is a real win over sequential**, but the magnitude depends
  heavily on whether the NPU degradation (113→42) is fundamental or
  fixable.
- Action: investigate whether the NPU drop with GPU-pipe-also-loaded is
  tunable (separate OV core instances? subprocess isolation? OV plugin
  config to disable cross-pipe scheduling?).

## Open follow-ups
1. Repeat on alpha B390 (Battlemage host with NPU) — does the same
   loading penalty apply?
2. Test with separate `ov::Core` instances per pipe (not sure if
   `openvino_genai.LLMPipeline` exposes that).
3. Run the concurrent test multiple times and report variance.
4. Test with target on NPU + draft on GPU (inverse of c24 — mostly an
   academic interest since the small draft on NPU is the wrong mapping).
