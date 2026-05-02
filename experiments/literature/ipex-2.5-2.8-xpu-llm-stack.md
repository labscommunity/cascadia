# IPEX (Intel Extension for PyTorch) XPU LLM stack: 2.5 → 2.8 (and EOL)

**Released:** 2.5.10+xpu (2024-12) → 2.6.10+xpu (2025-03) → 2.7.10+xpu (2025-04) → 2.8.10+xpu (2025-08)
**What changed:** IPEX XPU evolved from a "patch torch for Intel GPU" library into the kernel provider for vLLM/TGI on Intel Arc + Max — and was then **discontinued** after v2.8 (Aug 2025). Future Intel GPU support moves to upstream torch + torch-xpu-ops.

Highlights by version:
- **2.5.10+xpu (Dec 2024)**: oneDNN v3.6, oneAPI 2025.0.1. Arc B-series Windows preview. Static Cache + Dynamic Cache support (HF format). Speculative decoding kernel support. Sliding window attention. Multi-round chat. Block-format support REMOVED from oneDNN integration (breaking).
- **2.6.10+xpu (Mar 2025)**: oneDNN v3.7. Custom kernels: `rms_norm`, `rotary_embedding`, `paged_attention`, `chunked_prefill`. MoE kernels: `topk_softmax`, `moe_gemm`, `moe_scatter`, `moe_gather`. **INT4 GEMM (GPTQ) ~1.3x faster than 2.5; ~1.5x faster than FP16 in decode, on par with FP16 in prefill.** AWQ algorithm support. NF4 QLoRA preview. Hybrid ATen op layer (delegating to torch-xpu-ops where possible). Arrow Lake-H (mobile) Windows beta.
- **2.7.10+xpu (Apr 2025)**: oneDNN v3.7.1. **Sliding-window support added to `ipex.llm.modules.PagedAttention.flash_attn_varlen_func`** (covers Phi3 + Mistral). NF4 dequantize kernel ~4.4x-5.6x faster. INT8 LoRA finetuning via `_int_mm`. Codegen moved to torch-xpu-ops. Python 3.13t prototype.
- **2.8.10+xpu (Aug 2025)**: oneDNN v3.8.1. Qwen3 added to optimized model list. New custom kernels: `selective_scan_fn`, `causal_conv1d_fn` for Jamba. **PyTorch XCCL adoption for distributed (replaces torch-ccl).** Stops overriding device allocator + most oneMKL/oneDNN ops (uses upstream PyTorch's). **EOL announced.**

**Headline perf claim (if any):** "INT4 GEMM ~1.3x faster than previous release; in decode stage outperforms FP16 by ~1.5x" (2.6). NF4 dequantize 4.4-5.6x speedup (2.7).
**How to use it from optimum-intel / OV runtime:**
```python
# Inference path
import intel_extension_for_pytorch as ipex
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", torch_dtype=torch.float16).to("xpu")
model = ipex.llm.optimize(model, dtype=torch.float16, device="xpu")  # auto-applies fused kernels
# Then standard transformers .generate()

# vLLM on Intel GPU (uses IPEX kernels for PA, chunked prefill, MoE)
# pip install vllm[xpu]
# python -m vllm.entrypoints.openai.api_server --model qwen2.5-7b --device xpu
```
**Intel GPU applicability:**
- Arc B390 Battlemage (dGPU): HIGH — actively validated for vLLM/TGI by Intel.
- Arc 140V Lunar Lake (iGPU): MEDIUM — supported but iGPU validation mostly via Core Ultra Series 2 SKU. Better path on iGPU is OV runtime.
**Open hypothesis it generates for us:** On charlie (B390) install IPEX 2.8.10+xpu + vLLM-xpu and serve Llama-3-8B-INT4-GPTQ; compare against OV+GenAI continuous-batching at batch=8 (50 concurrent prompts). Hypothesis: vLLM-xpu wins by ≥30% on tokens/sec/GPU because of more mature PagedAttention scheduling, even though OV's INT4 kernels are faster per-token.

Sources:
- https://github.com/intel/intel-extension-for-pytorch/releases/tag/v2.5.10%2Bxpu
- https://github.com/intel/intel-extension-for-pytorch/releases/tag/v2.6.10%2Bxpu
- https://github.com/intel/intel-extension-for-pytorch/releases/tag/v2.7.10%2Bxpu
- https://github.com/intel/intel-extension-for-pytorch/releases/tag/v2.8.10%2Bxpu
