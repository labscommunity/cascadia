# OpenVINO GPU plugin properties: complete map for LLM tuning

**Released:** Stable surface across OV 2024.x → 2026.x; documented at https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html
**What changed:** This is a synthesis of the GPU plugin properties relevant to LLM perf. Each gets a brief description of what it does, the C++/Python ov::set_property line, and notes on iGPU vs dGPU applicability.

**Headline perf claim (if any):** N/A — these are the dials.

**How to use it from optimum-intel / OV runtime:**

### Hint properties (always use these first)
```python
import openvino as ov, openvino.properties.hint as hint
core = ov.Core()
core.set_property("GPU", {
    hint.performance_mode(): hint.PerformanceMode.LATENCY,   # or THROUGHPUT, CUMULATIVE_THROUGHPUT
    hint.execution_mode(): hint.ExecutionMode.PERFORMANCE,   # or ACCURACY
    hint.inference_precision(): ov.Type.f16,                 # f16 standard for GPU LLM; bf16 also supported on Series 2 / B-series
    hint.num_requests(): 1,                                  # at LATENCY the plugin auto-picks
    hint.priority(): hint.Priority.HIGH,                     # combined with GPU_QUEUE_PRIORITY below
})
```

### Compilation cache (huge TTFT win on cold start)
```python
core.set_property("GPU", {"CACHE_DIR": "./ov_gpu_cache"})
core.set_property("GPU", {"CACHE_MODE": "OPTIMIZE_SPEED"})  # vs OPTIMIZE_SIZE
```
Enabling `CACHE_DIR` shaves model-load time by an order of magnitude after the first run — the GPU plugin compiles cl_cache binaries per device.

### KV cache compression (default U8 since 2024.6 on GPU)
```python
core.set_property("GPU", {"KV_CACHE_PRECISION": "u8"})    # default; u8 / f16 / bf16
```

### Dynamic activation quantization
```python
core.set_property("GPU", {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"})  # 0 disables, 32/64/128 are typical
```
Default 32 on Lunar Lake / B-series since 2025.2. Runs activation → INT8 with tile-grain scaling. Pairs with INT4 weights for the fastest LLM decode path on iGPU/dGPU.

### Concurrency / queue properties
```python
import openvino.properties.intel_gpu.hint as gpu_hint
core.set_property("GPU", {
    gpu_hint.queue_priority(): hint.Priority.HIGH,            # GPU_QUEUE_PRIORITY: LOW/MED/HIGH (CL_QUEUE_PRIORITY_*_KHR)
    gpu_hint.queue_throttle(): gpu_hint.ThrottleLevel.HIGH,   # GPU_QUEUE_THROTTLE: LOW/MED/HIGH energy-vs-perf
    gpu_hint.host_task_priority(): hint.Priority.HIGH,        # GPU_HOST_TASK_PRIORITY: HIGH = pin TBB to BIG cores
})
```
For an inference-only daemon on Lunar Lake (where the iGPU shares the CPU power budget), `queue_throttle=HIGH + host_task_priority=HIGH` typically gives ~5-15% latency improvement at the cost of fan noise / battery life.

### Streams / batching
```python
import openvino.properties as props, openvino.properties.streams as streams
core.set_property("GPU", {streams.num(): 2})            # 2 GPU streams for concurrent inference requests
core.set_property("GPU", {hint.num_requests(): 4})      # batch hint when using THROUGHPUT
```
On dGPU streams=2-4 helps THROUGHPUT mode; on iGPU usually streams=1 is best (fewer conflicts on shared XMX).

### Compile-time control
```python
core.set_property("GPU", {hint.enable_cpu_pinning(): True})   # ov::hint::enable_cpu_pinning replaces the deprecated Affinity API
```

### Other notable properties
- `MAX_NUM_INFER_REQUESTS` — bound on parallel infer reqs on GPU
- `INFERENCE_NUM_THREADS` — number of host helper threads (compile)
- `MODEL_PRIORITY` — `LOW/MED/HIGH` for context priority
- `ENABLE_PROFILING` — turn on per-op timing
- `LOG_LEVEL` — `LOG_NONE/ERROR/WARNING/INFO/DEBUG/TRACE`

### Multi-device / heterogeneous
```python
# Pin layers across iGPU + NPU (LLMs)
ov_model = core.compile_model(model, "HETERO:GPU,NPU", {"HETERO_PRIORITIES": "GPU NPU"})
# Auto plugin (let OV decide)
ov_model = core.compile_model(model, "AUTO:GPU,CPU", {hint.performance_mode(): hint.PerformanceMode.LATENCY})
```

**Intel GPU applicability:** HIGH for both Arc 140V and Arc B390 — every property above is supported on both.
**Open hypothesis it generates for us:** Build a small ablation matrix on charlie (B390): grid over `KV_CACHE_PRECISION ∈ {u8, f16}` × `DYNAMIC_QUANTIZATION_GROUP_SIZE ∈ {0, 32, 64, 128}` × `streams ∈ {1, 2}` × `CACHE_DIR ∈ {set, unset}` for Llama-3-8B-INT4. 32 cells, 5-min budget each. Hypothesis: `u8 / 32 / 1 / set` is within 5% of the global max; `f16 / 0 / 1 / unset` is the worst (the OV ≤2024.5 default).

Sources:
- https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__runtime__ocl__gpu__prop__cpp__api.html
- https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html
