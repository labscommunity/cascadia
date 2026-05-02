# llama.cpp SYCL backend: Intel Arc / Lunar Lake / Battlemage 2025-2026

**Released:** Continuous; key dates below.
**What changed:** llama.cpp's SYCL backend is the *baseline competitor* on Intel GPU. Important inflection points:

- **2025-Q2 (~b3xxx era)**: Reorder framework added Q4_0 to MUL_MAT path → 21-87% speedup on Llama-2-7B-Q4_0 across Intel iGPUs (MTL, ARL-H) and dGPUs (Arc, Flex, PVC).
- **2025-09**: Q8_0 added to the reorder optimization → **3.1x token-generation speedup on Arc Pro B70 (4.88 → 15.24 t/s on Qwen3.5-27B)**.
- **2026-Q1**: Flash-Attention support landed for SYCL backend; perf impact varies by model (positive for long-context attention-bound, ~neutral elsewhere).
- **Outstanding bug (open as of survey)**: `GGML_SYCL_F16=ON` produces corrupted output on Intel Arc Pro B70 / Xe2 unless `GGML_SYCL_DISABLE_OPT=1` is set. Worth knowing for B390 (same Xe2 family).
- **Q8_0 kernel inefficiency on Battlemage**: Q8_0 only achieves 21-24% of theoretical bandwidth on B70/Xe2 vs Q4_K_M at 53-64% — kernel work continues.

**Headline perf claim (if any):** 3.1x on Arc Pro B70 (Qwen3.5-27B Q8_0) with reorder framework. Lunar Lake iGPU prefill ~2x faster than M1 Pro and "nearly on par with M1 Max", but decode TG comparable to M2 (memory-bandwidth limited).
**How to use it from optimum-intel / OV runtime:** Not optimum/OV — but the reference build + run for Intel GPU:
```bash
# Build
git clone https://github.com/ggerganov/llama.cpp; cd llama.cpp
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DGGML_SYCL_F16=ON   # ENABLE WITH CARE on Xe2 / B-series
cmake --build build --config Release -j

# Run on Intel iGPU (Arc 140V) or dGPU (Arc B390)
ZES_ENABLE_SYSMAN=1 GGML_SYCL_DISABLE_OPT=1 \
  ./build/bin/llama-cli -m model.Q4_K_M.gguf -p "Hello" -n 128 -ngl 99 \
  --device SYCL0   # SYCL0 = first Intel GPU
```
Note the OpenVINO 2026.1 release introduces an **OV backend FOR llama.cpp** (`-DGGML_OPENVINO=ON`), which is likely a better path than `-DGGML_SYCL=ON` on Intel hardware once stable.
**Intel GPU applicability:** HIGH for both. This is the bar to beat — anyone running local LLMs on an Intel GPU today is on llama.cpp SYCL.
**Open hypothesis it generates for us:** On charlie (B390), benchmark Qwen2.5-7B Q4_K_M three ways: (a) llama.cpp SYCL b5xxx, (b) llama.cpp with OV backend (2026.1), (c) OV GenAI native IR. Measure tokens/sec at batch=1 and batch=8. Hypothesis: at batch=1 (a)≈(b)≈(c)±15%; at batch=8 (c) wins by ≥2x because llama.cpp lacks continuous batching.

Sources:
- https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
- https://github.com/ggml-org/llama.cpp/issues/21517 (Q8_0 kernel inefficiency)
- https://github.com/ggml-org/llama.cpp/issues/21893 (F16 corruption on Battlemage)
- https://github.com/ggml-org/llama.cpp/discussions/12570 (Arc status)
- https://medium.com/@techhara/local-llm-benchmark-on-intel-lunar-lake-133c39f10455
