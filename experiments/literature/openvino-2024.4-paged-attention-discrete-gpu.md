# OpenVINO: PagedAttention + Continuous Batching for discrete GPU (2024.4.0)

**Released:** 2024-09 (2024-09-19)
**What changed:** First release where the GPU plugin gains PagedAttention and continuous batching for *discrete* GPUs (Arc / Flex). Same release introduced XMX systolic-array LLM kernels for Lunar Lake (Series 2) built-in GPUs, and added memory sharing for NPUs on Series 2 (no host-copy overhead).
**Headline perf claim (if any):** "Significant boost in throughput for parallel inferencing when serving LLMs on Intel Arc Graphics" (no number given). Both 1st and 2nd token latency improved on Intel GPU platforms.
**How to use it from optimum-intel / OV runtime:** Continuous batching is enabled by passing a `SchedulerConfig` to `LLMPipeline`:
```python
import openvino_genai as ov_genai
sched = ov_genai.SchedulerConfig()
sched.cache_size = 1                 # GB of KV cache budget; or use num_kv_blocks
sched.dynamic_split_fuse = True      # split prefill, fuse with decode batches
sched.max_num_batched_tokens = 2048  # tokens per scheduling step
pipe = ov_genai.LLMPipeline(model_dir, "GPU", scheduler_config=sched)
```
For optimum: `OVModelForCausalLM` does not directly expose continuous batching; use GenAI `LLMPipeline` for it on GPU.
**Intel GPU applicability:** HIGH for Arc B390 Battlemage (dGPU). MEDIUM for Arc 140V Lunar Lake (the release explicitly calls out "discrete GPUs" for PagedAttention; iGPU got XMX kernels and dynamic quantization separately in 2024.4 / 2024.5).
**Open hypothesis it generates for us:** On charlie (B390) measure tokens/sec for Llama-3-8B-INT4 via `LLMPipeline` with default config vs. with `SchedulerConfig(cache_size=4, dynamic_split_fuse=True, max_num_batched_tokens=4096)` at batch=1, batch=4, batch=8. Hypothesis: continuous batching gives ≥2x throughput at batch=4, ≥4x at batch=8 on B390 vs. naive batching.

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2024.4.0
