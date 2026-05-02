# Leaderboard

Best measured tok/s per `(model, hardware, engine, prompt-class)` combination.

## Llama 3.1 8B Instruct INT4

| Hardware | Engine | Output | tok/s | Source |
|---|---|---|---|---|
| **alpha B390 GPU** | **LLMPipeline + FastDraft 150M K=5** | **64 tok** | **119.24** | c18-fastdraft/c18-1 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=10 | 64 tok | 118.22 | c18-fastdraft/c18-2 |
| alpha B390 GPU | LLMPipeline + 1B INT4 draft K=10 | 64 tok | 100.90 | c2-llmpipe-spec/c2-4 |
| alpha B390 GPU | LLMPipeline plain | 64 tok | 96.41 | c1-llmpipeline/c1-1 |
| alpha B390 GPU | LLMPipeline + plugin properties | 64 tok | 92.74 | c1-llmpipeline/c1-3 |
| charlie 140V GPU | LLMPipeline plain | 64 tok | 91.14 | c1-llmpipeline/c1-2 |
| alpha B390 GPU | tahoma `--engine ov-genai` | 64 tok | 87.10 | c3-ov-genai-engine/c3-1 |
| charlie 140V GPU | tahoma `--engine ov-genai` | 64 tok | 71.31 | c3-ov-genai-engine/c5-1 |
| **alpha B390 GPU** | **LLMPipeline + FastDraft 150M K=5** | **256 tok** | **24.79** | c18-fastdraft/c18-3 |
| alpha B390 GPU | LLMPipeline + 1B INT4 draft K=5 | 256 tok | 24.94 | c6-long-gen/c6-2 |
| charlie 140V GPU | LLMPipeline plain | 256 tok | 23.18 | c10-other-models/c17 |
| alpha B390 GPU | LLMPipeline plain | 256 tok | 21.54 | c6-long-gen/c6-1 |
| alpha B390 GPU | ov-spec K=4 + 1B INT4 draft | 64 tok | 13.83 | c0-baselines/c0-3 |
| alpha B390 GPU | ov-optimum (OVModelForCausalLM) | 64 tok | 8.89 | c0-baselines/c0-1b |
| charlie 140V GPU | ov-optimum (OVModelForCausalLM) | 64 tok | 10.33 | c0-baselines/c0-2 |

## Llama 3.2 1B Instruct INT4

| Hardware | Engine | Output | tok/s | Source |
|---|---|---|---|---|
| alpha B390 GPU | LLMPipeline plain | 64 tok | 149.47 | c10-other-models/c10-3 |
| alpha B390 GPU | LLMPipeline plain | 256 tok | 81.07 | c10-other-models/c10-2 |

## Llama 3.1 8B Instruct INT4 — distributed (alpha + charlie via Thunderbolt 4)

| Engine | Config | tok/s | Source |
|---|---|---|---|
| ov-dist-spec | K=4, v5 shards | 17.59 | c0-baselines/c0-5 |

(Strictly slower than single-node `ov-genai+FastDraft` (119 tok/s on alpha). Distributed serving for 8B is no longer a perf decision — only a fit decision for models that don't fit on one node.)

## Notes

- All numbers above are decode-only tok/s (load + warmup excluded).
- 64-tok runs use prompt "What is the capital of France?".
- 256-tok runs use prompt "Write a short essay about distributed inference for large language models." (creative writing).
- DISCOVERY #1 in `DISCOVERIES.md` documents the LLMPipeline jump.
- DISCOVERY #2 documents the FastDraft win.
