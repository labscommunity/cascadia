# optimum-intel: IPEX XPU + custom PagedAttention (v1.21.0)

**Released:** 2024-12 (v1.21.0)
**What changed:** Unified XPU/CPU IPEX modeling with a custom PagedAttention KV cache for LLMs (PR #1009 sywangyi). NF4 data type support for OV weight compression. NNCF 2.14 features integrated. SD3 + Flux + MiniCPMv + NanoLlava + Phi3v VLM support. Layer-wise quantization in INC.
**Headline perf claim (if any):** N/A specific, but PA on XPU is the structural change that lets the IPEX path target Intel GPU under Arc.
**How to use it from optimum-intel / OV runtime:**
```python
# IPEX XPU path (alternative to OV)
from optimum.intel import IPEXModelForCausalLM
import torch

model = IPEXModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B",
    torch_dtype=torch.float16, device_map="xpu")  # routes to Intel GPU via IPEX
# Generation uses IPEX's PagedAttention internally.

# OV path (compare)
from optimum.intel import OVModelForCausalLM
model = OVModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", export=True, device="GPU")
```
**Intel GPU applicability:** MEDIUM for Arc 140V (IPEX XPU validated mostly on B-series + Data Center Max; iGPU support is "best effort"). HIGH for Arc B390 — IPEX has been heavily exercised on B-series for vLLM/TGI.
**Open hypothesis it generates for us:** On charlie (B390), benchmark Llama-3-8B-INT4 via the IPEX XPU path vs OV GPU path. Hypothesis: at batch=1, OV is faster (better INT4 weight kernels); at batch=8, IPEX is competitive because of its custom PagedAttention with vLLM-style block layout.

Sources:
- https://github.com/huggingface/optimum-intel/releases/tag/v1.21.0
- https://github.com/huggingface/optimum-intel/pull/1009
