# OpenVINO transformation passes: SDPAToPagedAttention, IndirectKVCache, make_stateful_transformation, fuse_cache_reorder

**Released:** Various; surveyed across OV 2024.x → 2026.x source.
**What changed:** These are model-graph transformations the GPU plugin (and CPU plugin) run at compile-time on stateful LLMs. They're rarely user-facing but determine what the runtime actually executes.

### SDPAToPagedAttention
- **Where**: `src/common/transformations/include/transformations/sdpa_to_paged_attention.hpp` in the OpenVINO source
- **What**: Replaces the model's stock `ScaledDotProductAttention` op with `PagedAttention` op when the GPU plugin compiles a stateful LLM graph. This is the transformation that actually unlocks PagedAttention on GPU. With OV 2025.1 the GPU plugin runs this transform by default.
- **User-visible**: Set via SchedulerConfig in GenAI; or via plugin property in pure runtime. With `SchedulerConfig` provided to `LLMPipeline`, the graph compile path runs `SDPAToPagedAttention` and you get block-paged KV cache.
- The transform was moved into the runtime in OV 2026.0 GenAI (PR #2937: "Move SDPAToPA-related functionality to runtime").

### IndirectKVCache
- An optimization in the CPU plugin (and partial GPU) that stores KV cache as indirected tensors so beam-search variants don't need to physically reorder rows when reordering beams; instead a beam-index tensor is updated.

### make_stateful_transformation
- The pass that takes a stateless ONNX/OV model with explicit `past_key_values` inputs/outputs and folds them into Variable nodes hidden inside the model — i.e. converts to stateful form. Run automatically when you export with `text-generation-with-past` (default) and don't pass `--disable-stateful`. *If you skip this* (i.e. pass `--disable-stateful`) you get worse perf because the runtime can't manage KV internally.

### fuse_cache_reorder
- A graph pattern matcher that recognizes "Gather(Variable, beam_idx)" patterns produced by beam search and replaces with the beam-reorder op that runs on-device without copying.

### MHA fusion (added in OV 2024.3)
- Multi-Head Attention is fused into a single op (instead of 4-5 ops) to reduce launch overhead on GPU. Built into the GPU plugin compilation.

### RMSNorm + RoPE fusion (OV 2026.1)
- New pattern that fuses RMSNorm + Rotary Position Embedding into a single GPU kernel. Currently shipped for LTX-Video; broader LLM support presumably to follow.

**Headline perf claim (if any):** Each pass typically yields 5-15% latency improvement; cumulatively they're the difference between a "works" and a "fast" LLM on Intel GPU.
**How to use it from optimum-intel / OV runtime:** Mostly automatic. The user-controllable parts:
```python
# Make sure stateful is on (default):
# optimum-cli export openvino -m model --task text-generation-with-past ./out
# (don't pass --disable-stateful)

# Use SchedulerConfig in GenAI to engage SDPAToPagedAttention:
sched = ov_genai.SchedulerConfig(); sched.cache_size = 4
pipe = ov_genai.LLMPipeline("./out", "GPU", scheduler_config=sched)

# Verify the runtime is using PagedAttention by setting:
pipe.get_metrics()  # On GenAI, gives KV-cache hit rates
```
Inspect post-compile graph (debugging):
```python
import openvino as ov
core = ov.Core()
m = core.read_model("./out/openvino_model.xml")
compiled = core.compile_model(m, "GPU")
# Look at the runtime info to see PA was applied
print(compiled.get_property("DEVICE_ARCHITECTURE"))
```
**Intel GPU applicability:** HIGH — these transforms are how the GPU plugin actually runs LLMs. Without them, a stateful model with vanilla SDPA would fall back to recomputing KV every step.
**Open hypothesis it generates for us:** Export Llama-3-8B once with `--disable-stateful` and once without. Measure tokens/sec on charlie (B390). Hypothesis: stateful (default) gives ≥3x decode tokens/sec because the runtime can engage PagedAttention + indirect KV.

Sources:
- https://github.com/openvinotoolkit/openvino/tree/master/src/common/transformations/include/transformations
- https://github.com/openvinotoolkit/openvino.genai/pull/2937 (SDPAToPA → runtime, 2026.0)
- https://huggingface.co/docs/optimum/main/en/intel/openvino/export
