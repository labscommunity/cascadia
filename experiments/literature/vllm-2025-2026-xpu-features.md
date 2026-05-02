# vLLM XPU + cross-cutting features 2025-2026 likely to be ported into OpenVINO

**Released:** v0.7.0 (Feb 2025) → v0.20.0 (Apr 2026)
**What changed:** vLLM is the upstream-techniques source — Intel ports many of these into OV. Surveying recent releases for techniques relevant to Intel iGPU/dGPU LLM serving:

- **Intel XPU support**: vLLM-XPU is Intel's official path; backed by IPEX kernels. Recent: torch 2.11 upgrade for XPU (#37947), GDN attention for Qwen3-Next on XPU (#33657), torch.compile for XPU GDN (#39466), XPU MXFP8 + MXFP4 quant ops (#38682, #39857), per-channel FP8 linear (#38316), FP8 KV cache on XPU (#37731), W4A16 AutoRound on XPU (#37986).
- **TurboQuant 2-bit KV cache**: 4× capacity. New attention backend; FA3/FA4 prefill support added later.
- **Per-token-head INT8/FP8 KV cache quantization**: even finer than per-token KV compression.
- **PagedAttention v3**: more efficient scheduling, better long-context perf.
- **Continuous batching with chunked prefill**: standard since v0.6+.
- **Speculative decoding**: Eagle3 (faster than vanilla draft); MTP (multi-token prediction); Eagle3 for Gemma4, MiniMax-M2.
- **Prefix caching**: enabled by default (matches OV 2025.4 with prefix caching for chat).
- **Disaggregated prefill/decode**: split prefill onto one GPU, decode onto another (P/D disaggregation). Enabled by NIXL connector. Relevant to multi-host orchestration!
- **KV offload to CPU/disk**: HMA + LMCache integration.
- **Structured output**: XGrammar (also in OV GenAI 2025.3).
- **Mixture-of-Experts kernels**: TritonExperts, FlashInfer NVFP4 MoE, MoRI for AMD; XPU MXFP8 GEMM + compressed-tensor schema.
- **Quantization formats**: GGUF support added (incl. non-standard like UD-IQ1_S).

**Headline perf claim (if any):** TurboQuant: 4x KV cache capacity. P/D disaggregation gives roughly 1.3-1.7x throughput on benchmarks because prefill and decode have different optimal batch shapes.
**How to use it from optimum-intel / OV runtime:** Most vLLM techniques aren't directly in OV, but the patterns to watch:
```bash
# vLLM XPU on Arc dGPU
pip install vllm[xpu]
VLLM_XPU=1 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --device xpu --dtype float16 \
  --enable-prefix-caching --enable-chunked-prefill \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 32 --gpu-memory-utilization 0.9
```
**Intel GPU applicability:** HIGH for Arc B390 — vLLM-XPU is actively validated on B-series. MEDIUM for Arc 140V (iGPU memory budget restricts what models will fit; not Intel's primary target for vLLM).
**Open hypothesis it generates for us:** Take the vLLM "P/D disaggregation" idea and prototype a 2-host version on alpha + charlie: alpha (140V) does prefill (compute-heavy but small KV) on a system-prompt-heavy chat workload, ships KV via TCP/RDMA to charlie (B390) which does decode (memory-bandwidth-heavy). Hypothesis: even with TCP-only transfer, total tokens/sec exceeds running both phases on either single device because we play to each device's strength. This is a Tahoma differentiator — exo doesn't do this.

Sources:
- https://github.com/vllm-project/vllm/releases/tag/v0.20.0
- https://github.com/vllm-project/vllm/releases (full release index)
