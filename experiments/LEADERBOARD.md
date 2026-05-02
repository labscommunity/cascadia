# Leaderboard

Best measured tok/s per `(model, hardware, engine, prompt-class)` combination.

## Llama 3.1 8B Instruct INT4 — short output (64 tokens, factual prompt)

| Hardware | Engine | tok/s | Source |
|---|---|---|---|
| **alpha B390 GPU** | **tahoma `ov-genai` + FastDraft K=5** | **134.90** | c18-fastdraft/c18-6 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=5 (raw) | 119.24 | c18-fastdraft/c18-1 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=10 | 118.22 | c18-fastdraft/c18-2 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=7 | 115.67 | c18-fastdraft/c18-k7 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=12 | 115.76 | c18-fastdraft/c18-k12 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=4 | 106.14 | c18-fastdraft/c18-k4 |
| alpha B390 GPU | LLMPipeline + 1B INT4 draft K=10 | 100.90 | c2-llmpipe-spec/c2-4 |
| alpha B390 GPU | LLMPipeline plain | 96.41 | c1-llmpipeline/c1-1 |
| **charlie 140V GPU** | **tahoma `ov-genai` + FastDraft K=5** | **96.04** | c18-fastdraft/c18-7 |
| charlie 140V GPU | LLMPipeline plain | 91.14 | c1-llmpipeline/c1-2 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=3 | 89.15 | c18-fastdraft/c18-k3 |
| alpha B390 GPU | tahoma `ov-genai` plain | 87.10 | c3-ov-genai-engine/c3-1 |
| charlie 140V GPU | tahoma `ov-genai` plain | 71.31 | c3-ov-genai-engine/c5-1 |
| alpha B390 GPU | ov-spec K=4 + 1B INT4 draft (legacy) | 13.83 | c0-baselines/c0-3 |
| alpha B390 GPU | ov-optimum (OVModelForCausalLM) | 8.89 | c0-baselines/c0-1b |
| charlie 140V GPU | ov-optimum (OVModelForCausalLM) | 10.33 | c0-baselines/c0-2 |

## Llama 3.1 8B Instruct INT4 — long output (256 tokens, creative prompt)

| Hardware | Engine | tok/s | Source |
|---|---|---|---|
| **alpha B390 GPU** | **LLMPipeline + FastDraft 150M K=3** | **27.12** | c18-fastdraft/c18-256-k3 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=2 | 26.12 | c18-fastdraft/c18-256-k2 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=4 | 26.10 | c18-fastdraft/c18-256-k4 |
| charlie 140V GPU | LLMPipeline + FastDraft 150M K=5 | 26.90 | c18-fastdraft/c18-8 |
| alpha B390 GPU | LLMPipeline + 1B INT4 draft K=5 | 24.94 | c6-long-gen/c6-2 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=5 | 24.79 | c18-fastdraft/c18-3 |
| charlie 140V GPU | LLMPipeline plain | 23.18 | c10-other-models/c17 |
| alpha B390 GPU | LLMPipeline plain | 21.54 | c6-long-gen/c6-1 |

K-sweep finding: **the optimal K depends on output length AND content type.** For short factual responses, K=5-10 is the sweet spot (high accept rate amortises draft cost). For long creative responses, **K=3 is best** — lower because accept rate falls and over-speculation wastes draft work.

## Phi-3-mini-128k INT4 — 64 tokens

| Engine | tok/s | Source |
|---|---|---|
| LLMPipeline + 50M FastDraft K=5 | 43.90 (+36%) | c18-fastdraft/c18-9 |
| LLMPipeline plain | 32.18 | c18-fastdraft/c18-10 |

## Llama 3.2 1B Instruct INT4

| Hardware | Engine | Output | tok/s | Source |
|---|---|---|---|---|
| alpha B390 GPU | LLMPipeline plain | 64 tok | 149.47 | c10-other-models/c10-3 |
| alpha B390 GPU | LLMPipeline plain | 256 tok | 81.07 | c10-other-models/c10-2 |

## Llama 3.1 8B Instruct INT4 — distributed (alpha + charlie via Thunderbolt 4)

| Engine | Config | tok/s | Source |
|---|---|---|---|
| ov-dist-spec | K=4, v5 shards | 17.59 | c0-baselines/c0-5 |

(Now strictly slower than single-node `ov-genai+FastDraft` at 134.9 / 96.0 tok/s. Distributed serving for 8B is no longer a perf decision.)

## Aggregate speedup vs main branch's `baselines.md`

| Hardware | main best | new best (this branch) | multiplier |
|---|---|---|---|
| alpha B390 GPU 8B INT4 | 16.7 (`ov-optimum`) | **134.90** (`ov-genai + FastDraft K=5`) | **8.1×** |
| alpha B390 GPU 8B INT4 | 35.0 (`ov-spec K=4`) | **134.90** | **3.9×** |
| charlie 140V GPU 8B INT4 | 17.0 (`ov-optimum`) | **96.04** | **5.6×** |
| alpha B390 GPU 8B INT4 256-tok | (n/a) | 27.12 | — |
| 1B INT4 64-tok | (n/a) | 149.47 | — |

## Notes

- All numbers above are decode-only tok/s (load + warmup excluded).
- 64-tok runs use prompt "What is the capital of France?".
- 256-tok runs use prompt "Write a short essay about distributed inference for large language models." (creative writing).
- DISCOVERY #1 in `DISCOVERIES.md` documents the LLMPipeline jump.
- DISCOVERY #2 documents the FastDraft win.
