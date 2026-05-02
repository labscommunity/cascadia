# c6: long-generation perf

## Setup

- Target: Llama 3.1 8B INT4 on alpha B390 GPU, LLMPipeline.
- Draft (when used): `srang992/Llama-3.2-1B-Instruct-ov-INT4`.
- Prompt: "Write a short essay about distributed inference for large language models." (creative, longer-form than c2's factual prompt)
- Output: **256 tokens**

## Results

| ID | Engine | K | Decode tok/s | Decode time | vs plain |
|---|---|---|---|---|---|
| c6-1 | LLMPipeline plain | — | **21.54** | 11.88 s | (baseline) |
| c6-2 | LLMPipeline + draft | 5 | **24.94** | 10.27 s | **+15.8%** |
| c6-3 | LLMPipeline + draft | 10 | 19.41 | 13.19 s | -9.9% |

## Headlines

1. **At 256-token creative output, plain LLMPipeline drops to 21.5 tok/s** — a 4.5× slowdown vs the 96.4 tok/s seen on a 64-token factual response. KV-cache attention dominates per-token cost as the cache grows.
2. **Spec decode K=5 buys +15.8% on long-form output** — much better than the +4% it gave at 64 tokens. The per-spec-round overhead amortises across more decode steps.
3. **K=10 over-speculates on creative content** — same pattern as the on-main K-sweep showed for v5 dist-spec at 256 tokens with creative prompts. Accept rate falls; wasted draft work outweighs savings.

## Recommendations to encode in the engine

- Default `--spec-k 4` is fine for short responses but **K=5 is the better default for `max_tokens >= 256`**.
- For chat workloads where each turn is short (~50-100 tokens), spec decode is mostly a wash — recommend not enabling it by default.
- For long-form generation (essays, code, summaries), spec decode is worth the draft-model load cost.

## Next

- c7: GGUF reader path on LLMPipeline (skip optimum-cli for prototypes).
- c8: SchedulerConfig + prefix caching for multi-turn chat workloads.
- c10: validate the ~10× LLMPipeline win on Phi-3-mini, Qwen 2.5, Gemma to see if it generalises across architectures.
