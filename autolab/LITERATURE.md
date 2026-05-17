# Literature synthesis (initial pass, 2026-05-17)

Compiled from 3 parallel research agents covering: (a) K2.6 / DeepSeek-V3
inference landscape, (b) KV/activation precision for MoE, (c) pipeline-
parallel async overlap + transport. Refreshed as the loop discovers new
candidate techniques.

## Headline reframing

> "Your current 0.05–0.11 tok/s suggests you are nowhere near the
> network/compute wall yet — the gain is in pipeline structure, not
> lower-level kernels." — (pipeline-parallel agent)

DRAM bandwidth ceiling on Lunar Lake is ~50 GB/s, putting the K2.6
theoretical CPU ceiling around 3-5 tok/s per box. Reference points from
the literature on the same model class (DeepSeek-V3 ~671B):

| Config | tok/s | Source |
|---|---:|---|
| 2x Xeon Plat 8452Y, 1TB DDR5, +A100, KTransformers V0.3 | **13.69** decode | KT tutorial |
| TR Pro 7965WX, 256GB, +A6000, ik_llama.cpp Q2_K_R4 | **13.13** | GH disc #258 |
| M3 Ultra 512GB, mlx-lm 4-bit | **>20** | VentureBeat |
| EPYC 9654, DDR5-4800 12ch, Q5_K_S llama.cpp | 6.2 | GH disc #11765 |
| EPYC 7K62, DDR4 8ch, Q5_K_S | 4.2 | same |
| 1x 24GB GPU + 256GB RAM, UD-TQ1_0 (1.8-bit) | 1-2 | Unsloth |
| **Tahoma 2-box matias-02/03 (current)** | **0.05** | this repo |

We're ~30-300× below comparable systems. The gap is structural, not
hardware-bound. **DRAM bandwidth dominates; doubling memory channels
doubles tok/s; dual-socket underperforms single-socket due to NUMA.**

## Top moonshots ranked by cross-agent consensus

### Tier S (do first; all three agents converged)

1. **Per-token expert reduction / sigmoid-threshold pruning (A2/A3).** Best-in-class data point: arxiv 2505.03531 "Faster MoE LLM Inference" reports **≥10% throughput / 0% perf loss** on DeepSeek-V3 sigmoid router; up to 50% at low concurrency (matches our CPU-bound regime). KTransformers V0.3 measured 286.55 vs 227 tok/s with `-ser` (K=6 vs K=8, ~25% upside). K2.6's sigmoid router is in the same family. This is the **single biggest decode-speed lever**.

2. **BF16 inter-rank wire (D1).** All three agents say this is essentially free. Halves frame from 28 KB → 14 KB. No published quality loss in production PP inference. Compute in f32, marshal in bf16.

3. **Async pipeline overlap (D4 / no current ID).** PipeInfer (arxiv 2407.11798): 1.5-2.15× over vanilla PP on 8 nodes. SpecPipe (arxiv 2504.04104): 4.19-5.53× TBT improvement on 8 stages. On our 2-stage pipeline expect ~2×. Requires drafter for the verify-while-draft pattern, OR speculative "compute T+1 using last-sampled token" pattern without drafter.

### Tier A (do after Tier S)

4. **DERP relay → direct WireGuard (D9).** 22ms → single-digit ms RTT. Eliminates jitter risk. Largest single network win.

5. **ProMoE-style speculative expert prefetch (C1).** Arxiv 2410.22134: MLP predictor 84.7% accuracy on expert selection, 2.20× TTFT, 2.07× TPOT. ~2M-param predictor trains in 1-2 h. K2.6's 384 experts × top-8 dispatch is a perfect fit.

6. **MLA KV INT8 with Hadamard rotation (A8 variant).** MHA2MLA (arxiv 2502.14837): 92% KV reduction at 0.5% LongBench drop. ik_llama.cpp `-ctk q8_0` is the reference impl. Bare INT8/FP8 without rotation breaks (vLLM Kimi-K2.5 case study).

7. **Layer rebalance based on per-stage profile (E2).** prima.cpp's "Halda" / Hyperion-style profile-driven assignment. Equal 30/30 split is almost certainly wrong on MoE because of expert-loading variance.

### Tier B (worthwhile but defer)

8. **iGPU offload of layer 0 / shared experts (E6).** OpenVINO 2026.0 GA'd MoE for Intel iGPU/NPU. Qwen3-30B-A3B hit 34 tok/s on Core Ultra 9 285H + Arc iGPU.
9. **NPU offload of router/gate (E7).** Lunar Lake has NPU; OV 2026.0 supports it for MoE.
10. **EAQuant per-expert calibration if re-quantizing (A7).** Arxiv 2506.13329: +1.15-13.81% accuracy over naive AWQ.

### Tier C (don't waste time on these)

- ❌ **MXFP4 on Intel CPU.** No Blackwell tensor cores → no compute speedup. RTN MXFP4 is *worse* than INT4 in perplexity (8.23 vs 7.40 on L3-8B). Only useful with Blackwell or with GPTQ/block-rotation calibration.
- ❌ **Tree-spec on current draft architecture.** PR #4 already showed it loses on wire scaling.
- ❌ **TP over Lunar-Lake fabric.** PR #4 already showed -90% throughput.
- ❌ **GPipe-style micro-batching for single-user decode.** No micro-batches when each forward = 1 token.

## Reference implementations to crib from

| Repo / paper | What to lift | Why |
|---|---|---|
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | FlashMLA-3 CPU kernel (PR #273), `-ser` expert reduction, runtime repack, IQ_KT trellis quants | Only widely-deployed OSS MLA CPU kernel. Practical reference for everything CPU-side. |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | AMX prefill kernel (21.3 TFLOPS/socket), expert deferral (+1.45×), NUMA-aware placement (+63%) | Best-published CPU-MoE perf. AMX is post-Lunar Lake but algorithms transfer. |
| [Unsloth dynamic GGUF](https://unsloth.ai/docs/models/tutorials/kimi-k2-thinking-how-to-run-locally) | UD-Q2_K_XL quantization recipe for K2 family | Documented K2 path for low-bit quant. |
| [PipeInfer paper (arxiv 2407.11798)](https://arxiv.org/html/2407.11798v1) | Continuous-speculation pipeline-bubble fill | Practical algorithm for async overlap without GPU TP. |
| [ProMoE paper (arxiv 2410.22134)](https://arxiv.org/html/2410.22134) | MLP-based expert predictor for prefetch | 84.7% hit rate; trains offline; small enough to live in tahoma. |
| [TransMLA / MHA2MLA](https://arxiv.org/abs/2502.07864) | Hadamard rotation before KV quant on MLA latents | Avoids the outlier-driven quality break that vLLM hit on Kimi-K2.5. |
| [OpenVINO 2026.0 MoE GA notes](https://medium.com/openvino-toolkit/openvino-2026-0-new-models-enhanced-genai-and-smarter-compression-bf846a59cda8) | MoE op set for iGPU/NPU | Opens the iGPU offload path. |

## Negative results from literature (avoid re-discovering)

- vLLM FP8 KV on MLA without per-head calibration: Kimi-K2.5 multi-turn coherence drops to 1.07/5. Systematic degradation across all context lengths.
- MXFP4 RTN on dense L3-8B: 8.23 ppl vs INT4 7.40 ppl (worse).
- BF16 reductions on extremely deep models can drift if not handled — keep compute in f32, store/ship in bf16.
- TP over consumer LAN: requires NVLink-class fabric; PR #4 d4 confirmed -90% on our hardware.

## Open questions the loop should keep asking

- Has anyone published K2.6 numbers specifically on Intel AI PC iGPU + NPU? (As of agent 3's search: **no** — green field.)
- What's the actual DDR5-6400 BW measurement on matias? Need to run `mlc` or equivalent.
- Does layer-0 OV IR pin the f16 / f32 numerics? Worth dropping into Rust if so.
- Is the K2.6 router's sigmoid output skewed enough that a 30-50% threshold-prune is near-lossless? Bench this directly.

## Sources for the index

All cited inline above. Master URL list:

- https://github.com/kvcache-ai/ktransformers
- https://github.com/ikawrakow/ik_llama.cpp
- https://github.com/ggml-org/llama.cpp/discussions/11765
- https://unsloth.ai/docs/models/tutorials/kimi-k2-thinking-how-to-run-locally
- https://huggingface.co/moonshotai/Kimi-K2.6
- https://medium.com/openvino-toolkit/openvino-2026-0-new-models-enhanced-genai-and-smarter-compression-bf846a59cda8
- https://arxiv.org/abs/2502.07864    (TransMLA)
- https://arxiv.org/abs/2502.14837    (MHA2MLA)
- https://arxiv.org/html/2410.22134   (ProMoE)
- https://arxiv.org/abs/2511.05814    (MoE caching)
- https://arxiv.org/html/2511.14102   (MoE-SpeQ)
- https://arxiv.org/abs/2602.16052    (MoE-Spec)
- https://arxiv.org/html/2505.03531   (Faster MoE LLM Inference)
- https://arxiv.org/html/2407.11798   (PipeInfer)
- https://arxiv.org/abs/2504.04104    (SpecPipe)
- https://arxiv.org/html/2504.08791   (prima.cpp)
- https://arxiv.org/abs/2509.26182    (Parallax)
- https://arxiv.org/abs/2506.13329    (EAQuant)
- https://arxiv.org/html/2411.09510   (FP5/FP4/FP3 PP compression)
- https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
- https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- https://www.lmsys.org/blog/2025-10-22-KTransformers/
