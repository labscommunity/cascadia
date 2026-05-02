# OpenVINO: FP8-4BLUT, llama.cpp backend, NPU spec-decoding (2026.0 + 2026.1)

**Released:** 2026-02 (2026.0) and 2026-04 (2026.1)
**What changed:**
- **2026.0**: int4 data-aware weight compression for 3D MatMuls → enables MoE LLMs with reduced memory + improved accuracy. Per-layer / per-group LUT for FP8-4BLUT (codebook quant). Speculative decoding for NPUs (Phi-3-mini FastDraft). NPU compiler-in-plugin (no driver dependency). MoE LLM optimization. GPT-OSS-20B + MiniCPM-V-4_5-8B + MiniCPM-o-2.6 supported on CPU+GPU. CPU plugin requires AVX2 minimum (SSE dropped).
- **2026.1**: Preview OpenVINO BACKEND for llama.cpp (so llama.cpp can dispatch to OV plugins on CPU/GPU/NPU; validated on Llama-3.2-1B GGUF, Phi-3-mini-4k GGUF, Qwen2.5-1.5B GGUF, Mistral-7B-Instruct-v0.3 GGUF). TaylorSeer Lite caching for Flux/SD3/LTX-Video diffusion. RMSNorm + RoPE fusion on GPU for LTX-Video. Prompt Lookup Decoding extended to VLMs. Dynamic LoRA for Qwen3-VL and other VLMs. Smaller runtime: ICU DLL eliminated from tokenizers. Arc Pro B70 with 32GB single-GPU 20-30B inference.
**Headline perf claim (if any):** 2026.0: "MoE LLMs to run with reduced memory, bandwidth, and improved accuracy." 2026.1: "End-to-end acceleration through fusion of RMSNorm and RoPE operators" for LTX-Video on GPU.
**How to use it from optimum-intel / OV runtime:**
```python
# Spec decoding on NPU with FastDraft
pipe = ov_genai.LLMPipeline("Phi-3-mini-128k-instruct-int4-ov", "NPU",
    draft_model=ov_genai.draft_model("Phi-3-mini-FastDraft-50M-int8-ov", "NPU"))

# FP8 LUT compression (cb4 = codebook 4-bit with 16 fp8 values)
# At export:
# optimum-cli export openvino -m model_id --weight-format cb4 ./out

# llama.cpp -> OV backend (preview, 2026.1)
# Build llama.cpp with -DGGML_OPENVINO=ON; runs GGUF on Intel CPU/GPU/NPU
```
**Intel GPU applicability:** HIGH for Arc 140V and Arc B390. The llama.cpp-OV backend is potentially huge: it lets us route every llama.cpp-style model through OV plugin tuning on iGPU/dGPU. FP8 LUT is preview but it's the path to >INT4 accuracy at INT4-ish memory.
**Open hypothesis it generates for us:** On alpha (140V) build llama.cpp with `GGML_OPENVINO=ON` (2026.1) vs vanilla `GGML_SYCL=ON`, run Phi-3-mini-4k Q4_K_M, measure tokens/sec. Hypothesis: OV backend delivers ≥30% faster decode on 140V because of the U8 KV cache + dynamic-quant XMX path that pure SYCL doesn't have.

Sources:
- https://github.com/openvinotoolkit/openvino/releases/tag/2026.0.0
- https://github.com/openvinotoolkit/openvino/releases/tag/2026.1.0
- https://huggingface.co/OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov
