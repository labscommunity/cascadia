# c60: MAJOR CORRECTION to Discovery #1 — LLMPipeline is NOT 10× faster

## Setup
alpha B390 GPU, Llama 3.1 8B INT4. Same model directory used by both engines.
Same prompt: "What is the capital of France?". 64 max_tokens cap.
Both warmed up before timing.

## Results (with proper actual-token counting)

| Engine | actual tokens generated | dt | actual tok/s |
|--------|-----------------------:|---:|-------------:|
| `optimum.intel.OVModelForCausalLM` | 64 | 3.245s | **19.72** |
| `openvino_genai.LLMPipeline` | 8 (EOS early) | 0.459s | **17.42** |
| Relative | — | — | LLMPipeline is **0.88×** of OVModel |

**LLMPipeline is actually ~12% SLOWER per-token than OVModelForCausalLM in
this test.**

## What went wrong with the original c0/c1 bench

The c0 baseline reported `ov-optimum: 8.89 tok/s`. The c1 LLMPipeline test
reported `96.41 tok/s`. The "10.8× speedup" was the headline Discovery #1.

In reality:
1. **c0 bench's 8.89 was likely cold-start inflated** — the OVModel bench
   didn't warm up properly before timing. Real OVModel rate is ~20 tok/s.
2. **c1 bench's 96 was max-tokens-cap inflated** — LLMPipeline EOSes at 8
   tokens but the bench divided 64 by total dt. Real LLMPipeline rate is
   ~17 tok/s.

## What this means for Discovery #1

The narrative "switch from OVModelForCausalLM to LLMPipeline for 10× speedup"
is **WRONG**. They produce roughly the same per-token rate (~17-20 tok/s).

Reasons users should still prefer LLMPipeline:
1. **Chat template handling**: LLMPipeline applies chat templates correctly.
   OVModelForCausalLM continues generating raw text past natural EOS.
2. **Spec decode hooks**: LLMPipeline supports `draft_model=` (FastDraft)
   and `prompt_lookup=True` (PL). These ARE real speedups (FastDraft +55%,
   PL +40-50% per c58/c57).
3. **Continuous batching**: LLMPipeline has SchedulerConfig CB.
4. **Cleaner API**.

But the **raw decode rate is similar**. The "10× win" was a methodology error.

## Implications for the autolab session

Discoveries #2, #3, #4 are still valid (relative wins within LLMPipeline):
- FastDraft over LLMPipeline plain: real +55% (c58)
- PL over LLMPipeline plain (extractive): real +40-50% (c57)
- NPU concurrent over sequential: real +16% (c52)

Discovery #1 needs to be re-stated:
- LLMPipeline ≈ OVModelForCausalLM for raw per-token rate.
- LLMPipeline is the right engine for production for OTHER reasons.

## Lesson

Always verify cross-engine comparisons by:
1. Counting actual generated tokens.
2. Properly warming both engines before timing.
3. Using the SAME prompt-handling (chat template applied or not).

## Update: Apples-to-apples comparison (c60-fair)

When BOTH engines:
1. Use the same chat template (both apply Llama 3 chat template)
2. Are properly warmed up (multiple warm-up generates)
3. Produce the same output (both generate "The capital of France is Paris." = 8 tokens)

| Engine | actual tokens | dt | actual tok/s |
|--------|--------------:|---:|-------------:|
| `OVModelForCausalLM` | 8 | 0.464s | 17.23 |
| `LLMPipeline` | 8 | 0.464s | 17.24 |
| **Ratio** | — | — | **1.00× (IDENTICAL)** |

## DEFINITIVE finding

OVModelForCausalLM and LLMPipeline have **the SAME per-token throughput**
on Intel GPU for INT4 LLMs in OV 2026.1. Discovery #1's claimed 10×
speedup was 100% a bench artifact.

The original c0/c1 numbers (8.89 vs 96.41) emerged from:
- c0 OVModel bench: cold-start (no warmup → JIT compile included in timing)
- c1 LLMPipeline bench: warmed up + max_tokens-cap inflation

When you fix BOTH bench errors, the engines tie.

## Why LLMPipeline is still right for tahoma

- **Chat template applied correctly** by default.
- **Spec decode hooks** (FastDraft +55%, PL +40-50%).
- **Continuous batching** support.
- **Cleaner Python API** (`pipe.generate(prompt, cfg)` vs setting up tokenizer + generation_config + decode).

These are real engineering wins. Just NOT a raw per-token speed win.
