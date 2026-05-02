# intel/intel-npu-acceleration-library: EOL — superseded by OpenVINO GenAI on NPU

**Released:** Last release v1.4.0 (2024-11-22); no commits since (as of 2026-04 survey).
**What changed:** Library was Intel's first attempt at a Python-friendly NPU API for AI PCs (Meteor Lake / Lunar Lake / Arrow Lake NPUs). Final release v1.4.0 added:
- Turbo mode (max NPU power state, ~10-20% latency win)
- Phi-3 MLP layer support
- Qwen Math 7B example
- Audio Spectrogram Transformer example
- Ubuntu 24.04 build
- chunk tensor op
- log_softmax, prelu, normalize ops

The library has had **no commits since November 2024**. Intel's NPU LLM story has moved entirely into the OpenVINO 2025.x NPU plugin + OV GenAI's NPU pipeline. New work (FastDraft speculative decoding, NPU LLM acceleration in OV Runtime, Whisper on NPU, NPU dynamic prompts up to 8K, NPU dynamic batch, etc.) is shipping in OV not here.

**Headline perf claim (if any):** Turbo mode ~10-20% latency improvement on Llama-2-7B (community benchmarks).
**How to use it from optimum-intel / OV runtime:** Don't. Use OV NPU plugin instead:
```python
# CORRECT 2025+ NPU path
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("Llama-3-8B-int4-ov", "NPU")
print(pipe.generate("Hi"))

# Validated NPU LLMs (OV ≥2024.5): Llama 3 8B, Llama 2 7B, Mistral-v0.2-7B,
# Qwen2-7B-Instruct, Phi-3 Mini, Phi-4 Mini, Qwen3 (1.7/4/8B)

# Spec decoding on NPU (OV 2026.0): use Phi-3-mini-FastDraft-50M as draft
pipe = ov_genai.LLMPipeline("Phi-3-mini-128k-instruct-int4-ov", "NPU",
    draft_model=ov_genai.draft_model("Phi-3-mini-FastDraft-50M-int8-ov", "NPU"))
```
Pre-converted NPU LLMs are at https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0
**Intel GPU applicability:** N/A directly (this lib is NPU-only). But for the Lunar Lake target it matters indirectly — Arc 140V *also* has an NPU on the same SoC, and HETERO:GPU,NPU dispatch is interesting (LLM head on NPU + body on GPU).
**Open hypothesis it generates for us:** On alpha (Lunar Lake) compare Phi-3-mini-INT4 on NPU-only vs GPU-only vs HETERO:GPU,NPU. Hypothesis: NPU is most power-efficient (lowest watts) but GPU has highest tokens/sec; HETERO is rarely best because of host transfer overhead.

Sources:
- https://github.com/intel/intel-npu-acceleration-library/releases/tag/v1.4.0
- https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0
