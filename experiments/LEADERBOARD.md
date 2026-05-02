# Leaderboard

Best measured tok/s per (model, hardware, engine, prompt-class) combination. Rows are replaced when a higher number lands; the previous record stays in the linked experiment dir.

## Llama 3.1 8B Instruct INT4 — single node

| Hardware | Engine | tok/s | Source |
|---|---|---|---|
| **alpha B390 GPU** | **openvino_genai.LLMPipeline (greedy, no config)** | **96.41** | c1-llmpipeline/c1-1 |
| alpha B390 GPU | ov-spec K=4 + 1B INT4 draft | 13.83 | c0-baselines/c0-3 |
| alpha B390 GPU | ov-optimum (OVModelForCausalLM) | 8.89 | c0-baselines/c0-1b |
| charlie 140V GPU | ov-optimum (OVModelForCausalLM) | 10.33 | c0-baselines/c0-2 |
| charlie 140V GPU | LLMPipeline | (n/a) | DLL ABI mismatch — c1-2 |

## Llama 3.1 8B Instruct INT4 — distributed (alpha + charlie via Thunderbolt 4)

| Engine | Config | tok/s | Source |
|---|---|---|---|
| ov-dist-spec | K=4, v5 shards | 17.59 | c0-baselines/c0-5 |

(Now strictly slower than single-node alpha LLMPipeline. Distributed serving for 8B is no longer a perf win — it's a fit win for models that don't fit on one node.)

## Notes

- All numbers above are decode-only tok/s (load + warmup excluded), 64 generated tokens, prompt = "What is the capital of France?".
- The 10.8× LLMPipeline jump on alpha is documented in `DISCOVERIES.md` #1.
