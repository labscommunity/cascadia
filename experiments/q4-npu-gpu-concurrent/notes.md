# Q4 — NPU+GPU concurrent on charlie (blocked on dynamic-shape NPU compile)

**Hypothesis:** OV 2024.4+ added Lunar Lake NPU/GPU memory sharing via Level Zero RemoteTensor (zero-copy host-shared). Prior python autolab d4 finding of -75% was OV 2024.x without this path. Test if OV 2026.1 makes NPU+GPU concurrent on a single host viable for "draft on NPU + target on GPU."

**Setup:** charlie (Lunar Lake 140V iGPU + NPU 4). Llama 3.2 1B INT4 on NPU + Llama 3.1 8B INT4 on GPU, run concurrently in separate threads.

## Result

**GPU 8B alone (charlie):** 48-52 ms/token = **19.3-20.7 tok/s**.

This was an unexpected finding — charlie's 140V iGPU on Llama 8B (LLMPipeline path) is 84-90% the speed of alpha's B390 dGPU (23 tok/s). The earlier assumption that charlie was "1.6× slower per layer" came from per-stage v5 IR timings (no LLMPipeline optimizations); the LLMPipeline-class path on the same hardware is much closer to alpha. **This is also a partial debunk of D2 — if charlie's per-stage compute can be brought to LLMPipeline parity (per Q2), the per-stage compute imbalance is much smaller than e10's 27/43 ms suggested.**

**NPU 1B failed to compile** even after `model.reshape({...})` to fix outer dimensions. OV's NPU plugin requires fully-static-shape models throughout — internal `ShapeOf` ops keep dynamic bounds even after reshape on the standard HF-converted IR (`srang992/Llama-3.2-1B-Instruct-ov-INT4`).

To unblock: either (a) re-export Llama 3.2 1B with full static shapes (`optimum-cli export openvino --task text-generation-with-past --batch 1 --sequence-length N`), (b) use openvino-genai's `LLMPipeline(device='NPU', ...)` which handles compile semantics internally, or (c) use Intel-published static-NPU variants (e.g. `OpenVINO/Llama-3.2-1B-Instruct-int4-npu-ov` if it exists).

**No NPU+GPU concurrent measurement landed**, so the L0 RemoteTensor path remains untested in OV 2026.1. Cannot confirm or refute the d4 -75% finding for current OV.

## What I did learn

- charlie's LLMPipeline-on-8B is competitive with alpha — D2's structural ceiling was based on per-stage compute that didn't include the LLMPipeline optimizations. **Q2 (PA at compile time) is the real lever**, even more than I assumed in the first analysis. Charlie at PA-equivalent could match alpha's per-stage in the distributed path, putting the 2-stage PP ceiling at ~25 tok/s — enough to clear the bar.

## Status

- Q4 stuck on NPU compile model availability — needs static-shape Llama 1B IR.
- Q4 was likely the wrong question anyway: if PA on charlie's GPU brings stage_1 down to LLMPipeline-class compute, NPU concurrency is icing not blocker. Re-prioritize Q2 over Q4.
