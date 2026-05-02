# Leaderboard

## ⚠️ CORRECTED NUMBERS (post c57-c59)

The bench scripts in c1-c56 used `tok_s = max_tokens / total_dt` instead of
counting actual generated tokens. For prompts that EOS early (factual chat,
extractive summary), this inflated absolute throughput by 5-14×. For
prompts that fill the cap (creative writing, multi-tenant CB explanation
prompts), the numbers are accurate.

### Verified actual rates (best of N runs)

| Hardware | Workload | Engine | actual tok/s |
|----------|----------|--------|-------------:|
| alpha B390 GPU | Llama 3.1 8B INT4, factual chat | LLMPipeline plain | 17.5 |
| alpha B390 GPU | Llama 3.1 8B INT4, factual chat | LLMPipeline + FastDraft K=5 | **27.2** (+55%) |
| alpha B390 GPU | Llama 3.1 8B INT4, creative 256-out | LLMPipeline + FastDraft K=3 | **28.29** |
| alpha B390 GPU | Llama 3.1 8B INT4, multi-tenant CB b=8 | LLMPipeline + CB | **131** aggregate |
| charlie 140V GPU | Llama 3.2 1B INT4, factual chat | LLMPipeline plain | **55.6** |
| charlie 140V GPU | Llama 3.1 8B INT4, extractive RAG (1K-4K input) | LLMPipeline + PL | **~28** (+40-50%) |

### Verified RELATIVE wins (still valid for engine selection)

| Comparison | Win |
|-----------|----:|
| LLMPipeline vs OVModelForCausalLM (8B INT4 factual) | ~10× |
| LLMPipeline + FastDraft vs LLMPipeline plain (factual) | +55% |
| LLMPipeline + PL vs LLMPipeline plain (extractive RAG) | +40-50% |
| Concurrent NPU+GPU vs sequential same-GPU (b=16 + 1B) | +16% effective |

---

# (LEGACY — original inflated numbers below)


Best measured tok/s per `(model, hardware, engine, workload)` combination.

## Llama 3.1 8B Instruct INT4 — single user

### Short factual (64 tokens, "What is the capital of France?")

| Hardware | Engine | tok/s | Source |
|---|---|---|---|
| **alpha B390 GPU** | **tahoma `ov-genai` + FastDraft 150M K=5** | **134.90** | c18-fastdraft/c18-6 |
| **charlie 140V GPU** | **tahoma `ov-genai` + FastDraft 150M K=5** | **96.04** | c18-fastdraft/c18-7 |
| alpha B390 GPU | LLMPipeline + FastDraft 150M K=5 (raw) | 119.24 | c18-fastdraft/c18-1 |
| alpha B390 GPU | LLMPipeline plain | 96.41 | c1-llmpipeline/c1-1 |
| charlie 140V GPU | LLMPipeline plain | 91.14 | c1-llmpipeline/c1-2 |
| alpha B390 GPU | tahoma `ov-genai` plain | 87.10 | c3-ov-genai-engine/c3-1 |
| alpha B390 GPU | ov-spec K=4 + 1B INT4 draft (legacy) | 13.83 | c0-baselines/c0-3 |
| alpha B390 GPU | ov-optimum (OVModelForCausalLM) | 8.89 | c0-baselines/c0-1b |
| charlie 140V GPU | ov-optimum (OVModelForCausalLM) | 10.33 | c0-baselines/c0-2 |

### Long creative (256 tokens, essay prompt)

| Hardware | Engine | tok/s | Source |
|---|---|---|---|
| **alpha B390 GPU** | **LLMPipeline + FastDraft 150M K=3** | **27.12** | c18-fastdraft/c18-256-k3 |
| charlie 140V GPU | LLMPipeline + FastDraft 150M K=5 | 26.90 | c18-fastdraft/c18-8 |
| alpha B390 GPU | LLMPipeline plain | 21.54 | c6-long-gen/c6-1 |

### RAG / summarization (128 tokens, passage + summarize)

| Hardware | Engine | tok/s | Source |
|---|---|---|---|
| **charlie 140V GPU** | **LLMPipeline + prompt_lookup (n=3, K=5)** | **108.82** | c21-prompt-lookup |
| **alpha B390 GPU** | **LLMPipeline + prompt_lookup (n=3, K=5)** | **91.57** | c21-prompt-lookup |
| charlie 140V GPU | LLMPipeline plain | 66.16 | c21-prompt-lookup |
| alpha B390 GPU | LLMPipeline plain | 57.69 | c21-prompt-lookup |

## Llama 3.1 8B Instruct INT4 — multi-tenant aggregate (LLMPipeline + SchedulerConfig CB)

| Hardware | Batch | Aggregate tok/s | Per-request |
|---|---|---|---|
| alpha B390 GPU | 1 | 134 | 134 |
| alpha B390 GPU | 8 | **138** | 17 |
| alpha B390 GPU | 16 | 274 | 17.1 |
| alpha B390 GPU | 32 | **559** | 17.5 |
| alpha B390 GPU | 64 | 362 (-35%, saturation) | 5.65 |
| charlie 140V GPU | 1 | 143 | 143 |
| charlie 140V GPU | 8 | **149** | 18.6 |

## Phi-3-mini INT4

| Hardware | Engine | tok/s |
|---|---|---|
| **alpha B390 GPU** | **LLMPipeline + 50M FastDraft K=5** | **43.90 (+36%)** |
| charlie 140V GPU | LLMPipeline + 50M FastDraft K=5 | 40.68 (+12%) |
| charlie 140V GPU | LLMPipeline plain | 36.26 |
| alpha B390 GPU | LLMPipeline plain | 32.18 |

## Llama 3.2 1B Instruct INT4 — single user

### 64-token factual

| Hardware | Engine | tok/s |
|---|---|---|
| **charlie 140V GPU** | LLMPipeline plain | **211.39** |
| alpha B390 GPU | LLMPipeline plain | 149.47 |
| alpha host NPU (Battlemage box) | LLMPipeline plain | 135.84 |
| charlie 140V NPU | LLMPipeline plain | 112.89 |

### 256-token

| Hardware | Engine | tok/s |
|---|---|---|
| alpha B390 GPU | LLMPipeline plain | 81.07 |

## Distributed (alpha + charlie via Thunderbolt 4)

| Engine | tok/s |
|---|---|
| ov-dist-spec K=4, v5 shards | 17.59 |

(Strictly slower than single-node `ov-genai+FastDraft` at 134/96 tok/s. Distributed serving for 8B is no longer a perf decision.)

## Aggregate speedup vs main branch's `baselines.md`

| Hardware | main best | new best (this branch) | multiplier |
|---|---|---|---|
| alpha B390 GPU 8B INT4 short | 16.7 (`ov-optimum`) | **134.90** (`ov-genai + FastDraft K=5`) | **8.1×** |
| alpha B390 GPU 8B INT4 short | 35.0 (`ov-spec K=4`) | **134.90** | **3.9×** |
| charlie 140V GPU 8B INT4 short | 17.0 (`ov-optimum`) | **96.04** | **5.6×** |
| alpha B390 GPU 8B INT4 RAG | (n/a) | **91.57** | — |
| charlie 140V GPU 8B INT4 RAG | (n/a) | **108.82** | — |

## Notes

- All numbers above are decode-only tok/s (load + warmup excluded).
- 64-tok runs use prompt "What is the capital of France?".
- 256-tok runs use prompt "Write a short essay about distributed inference for large language models." (creative writing).
- 128-tok RAG runs use a passage + "summarize in 2 sentences" instruction.
- DISCOVERY #1 in `DISCOVERIES.md` documents the LLMPipeline jump.
- DISCOVERY #2 documents the FastDraft win.
- DISCOVERY #3 documents the Prompt Lookup win.

## Llama 3.1 8B INT4 — RAG/extractive workloads (LLMPipeline + prompt_lookup)

| Hardware | Input | Output | Plain | PL n=3 | Δ | Source |
|---|---|---|---|---|---|---|
| **charlie 140V GPU** | 1024 | 256 | 101.88 | **160.64** | **+58%** | c45 |
| charlie 140V GPU | 1024 | 128 | 51.70 | 80.76 | +56% | c45 |
| charlie 140V GPU | 1024 | 64  | 25.65 | 40.20 | +57% | c45 |
| charlie 140V GPU | 2048 | 128 | 52.15 | 89.37 | +71% | c45 |
| charlie 140V GPU |  256 | 128 | 56.94 | 85.13 | +50% | c45 |
| charlie 140V GPU |  256 | 128 | 66.16 | 108.82 | +65% | c21 (original) |

**PL for extractive workloads scales beautifully across both input and output dimensions.**

## Llama 3.1 8B INT4 — RAG extractive cross-platform (LLMPipeline + prompt_lookup)

| Hardware | Input | Output | Plain | PL n=3 | Δ |
|---|---|---|---|---|---|
| **charlie 140V GPU** | 1024 | 256 | 101.88 | **160.64** | +58% |
| **alpha B390 GPU** | 1024 | 256 | 94.88 | **150.71** | +59% |
| charlie 140V GPU | 1024 | 128 | 51.70 | 80.76 | +56% |
| alpha B390 GPU | 1024 | 128 | 47.11 | 75.34 | +60% |
| charlie 140V GPU | 1024 | 64 | 25.65 | 40.20 | +57% |
| alpha B390 GPU | 1024 | 64 | 23.61 | 37.63 | +59% |

**charlie 140V iGPU edges out alpha B390 dGPU for RAG workloads** — likely due to lower iGPU latency to system RAM.

## PEAK PL extractive: 4K input + 256 output

| Hardware | Plain | PL n=3 | Δ |
|---|---|---|---|
| **charlie 140V GPU** | 100.17 | **194.36** | **+94%** |
| **alpha B390 GPU** | 97.21 | **191.53** | **+97%** |

**194 tok/s on a Lunar Lake iGPU for real RAG at 4K input + 256 output.**
Best single-instance speedup in the autolab session (PL nearly doubles plain).

## ABSOLUTE PEAK: PL extractive at 4K input + 512 output

| Hardware | Plain | PL n=3 | Δ |
|---|---|---|---|
| **charlie 140V GPU (Lunar Lake iGPU)** | 199.57 | **388.81** | **+95%** |
| **alpha B390 GPU (Battlemage dGPU)** | 194.57 | **381.54** | **+96%** |

**~388 tok/s on a Lunar Lake iGPU** for real extractive RAG at 4K input
+ 512 output. Pure decode rate ~480 tok/s (excluding ~250ms TTFT prefill).
PL nearly DOUBLES plain throughput. This is the absolute single-instance
peak in the autolab session.
