# Intel Neural Compressor 3.x: FP8 / MXFP4 / NVFP4 / MXFP8 (2024-2025)

**Released:** v3.0 (Aug 2024) → v3.7 (Dec 2025)
**What changed:** Neural Compressor pivoted in 3.x toward FP8 / MX-formats / NVFP4 quantization for LLMs targeting Gaudi + Xeon + Intel GPUs. Survey of relevant features:

- **v3.0 (Aug 2024)**: New 3.x architecture, modular framework. PyTorch-first, deprecation of TF/Keras path.
- **v3.1 (Oct 2024)**: TEQ (Trainable Equivalent Transformation) PTQ algorithm. Layer-wise quantization for memory-constrained quant. Auto-round integration improvements.
- **v3.2 (Dec 2024)**: AutoRound enhancements. Layer-wise quant for LLMs.
- **v3.3 (Mar 2025)**: MXFP8 + MXFP4 PTQ (experimental) on LLMs.
- **v3.4 (May 2025)**: Distillation work, INT4 weight-only.
- **v3.5 (Sep 2025)**: NVFP4 PTQ added.
- **v3.6 (Oct 2025)**: Mixed bits (MXFP4 + MXFP8) autotuning.
- **v3.7 (Dec 2025)**: NVFP4 PTQ on LLMs (experimental); mixed-bits MXFP4/MXFP8 autotuning across Llama 3 series, DeepSeek-R1, Qwen3-235B; **MXFP8 PTQ on video diffusion (FramePack)**; MXFP4 QAT on Llama 3.

**Validated GPU hardware (v3.7)**: Intel Arc B-Series Graphics (B580 and B60), Intel Data Center GPU Max Series (1550). Notably **Arc 140V (Lunar Lake iGPU) is NOT in the validated list** — INC 3.x is dGPU-focused.

**Headline perf claim (if any):** No specific %; gains depend on which quant scheme. The NVFP4 / MXFP4 path is the same compression (4-bit) as INT4 but with floating-point exponents → typically <0.5pp accuracy loss vs INT8.
**How to use it from optimum-intel / OV runtime:**
```python
# WeightOnlyQuant via INC for FP8/MXFP4
from neural_compressor.torch.quantization import RTNConfig, prepare, convert
config = RTNConfig(dtype="mx_fp4", group_size=32, use_sym=True)
model = prepare(model=model, quant_config=config)
model = convert(model)

# AutoRound (best FP4/INT4 accuracy)
from neural_compressor.torch.quantization import AutoRoundConfig
config = AutoRoundConfig(dtype="int4", group_size=128, sym=True, batch_size=8, iters=200)
```
Output models can be loaded by IPEX or exported to ONNX/OV. NNCF (which optimum-intel uses) has its own algorithms (AWQ, GPTQ, scale-estimation) that overlap with INC; for OV deployment NNCF is the canonical path, INC is the canonical path for IPEX/PyTorch deployment.
**Intel GPU applicability:** MEDIUM. INC 3.x targets B-series + Max-series. For Arc 140V (Lunar Lake) NNCF + optimum-intel is the better path. For Arc B390 INC's MXFP4/NVFP4 are interesting but require IPEX runtime (not OV).
**Open hypothesis it generates for us:** On charlie (B390), quantize Llama-3-8B with NNCF AWQ-INT4 (OV path) vs INC AutoRound-INT4 (IPEX path) using the same calibration data; compare lm-eval scores + tokens/sec. Hypothesis: AutoRound gives ≤0.3pp better lm-eval but the OV runtime tokens/sec is ≥1.3x faster, so NNCF+OV wins on Pareto.

Sources:
- https://github.com/intel/neural-compressor/releases/tag/v3.7
- https://github.com/intel/neural-compressor/releases/tag/v3.0
- https://github.com/intel/neural-compressor/releases
