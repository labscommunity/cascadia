# OpenVINO: PagedAttention + Continuous Batching DEFAULT on GPU plugin (2025.1.0)

**Released:** 2025-04 (2025-04-10)
**What changed:** PagedAttention and continuous batching are now ENABLED BY DEFAULT in the GPU plugin (no `SchedulerConfig` required). NPU acceleration for text-generation in OV Runtime + OV Model Server. Token Eviction in GenAI to bound KV cache for long generations. New LSTM kernels on GPU. Optimization for Core Ultra 200H 2nd-token latency. Reduced binary size (CPU GEMM kernel removed).
**Headline perf claim (if any):** "Enhanced performance and efficient resource utilization with the implementation of Paged Attention and Continuous Batching by default in the GPU plugin." (no number)
**How to use it from optimum-intel / OV runtime:** Just upgrade — no code change. To explicitly tune:
```python
sched = ov_genai.SchedulerConfig()
sched.cache_size = 4                 # GB
sched.max_num_batched_tokens = 4096
sched.dynamic_split_fuse = True
sched.enable_prefix_caching = True
pipe = ov_genai.LLMPipeline(model_dir, "GPU", scheduler_config=sched)

# Token Eviction
evict = ov_genai.CacheEvictionConfig(start_size=32, recent_size=128, max_cache_size=2048)
sched.cache_eviction_config = evict
```
For optimum: behavior also flips on `OVModelForCausalLM` — your existing code gets PA for free.
**Intel GPU applicability:** HIGH for both Arc 140V and Arc B390. This is the killer "free perf upgrade" release.
**Open hypothesis it generates for us:** Same workload, OV 2025.0 vs 2025.1 on alpha and charlie, Llama-3-8B-INT4, batch=1 streaming + batch=8 concurrent. Hypothesis: 2025.1 default is within 5% of 2024.4-with-explicit-SchedulerConfig at batch=1 (no regression) and ≥2x faster at batch=8.

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2025.1.0
