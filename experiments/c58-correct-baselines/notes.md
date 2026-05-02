# c58: Corrected Discovery #1 + #2 baselines (actual token counting)

## Setup
alpha B390 GPU, Llama 3.1 8B INT4. Greedy decode. 64 max_tokens cap.
Prompt: "What is the capital of France?" — model produces "The capital of
France is Paris." = **8 actual tokens**, then EOS.

## Results

| Mode | dt | actual tokens | actual tok/s | bench claimed |
|------|---:|--------------:|-------------:|--------------:|
| LLMPipeline plain | 0.456s | **8** | **17.54** | 96.41 (5.5× inflated) |
| LLMPipeline + FastDraft K=5 | 0.294s | **8** | **27.19** | 134.90 (5× inflated) |
| Relative win | — | — | **+55%** | +40% (bench-derived) |

## Findings

1. **Real plain LLMPipeline rate at this prompt: ~17.5 tok/s actual** (not 96).
   The 96 bench number was 64-token-cap divided by 0.66s = 96 even though
   only 8 tokens were generated.
2. **Real FastDraft rate: ~27.2 tok/s actual** (not 135). +55% relative.
3. **Discovery #2 magnitude is +55%, not +24%** — both numbers are valid for
   different framings (FastDraft completes the same task 55% faster, OR the
   "tok/s" rate is +40-55% depending on how you count).

## Updated headline numbers (actual)

For short factual chat ("What is the capital of France?", 8 actual tokens):
- alpha B390 LLMPipeline plain: 17.5 tok/s
- alpha B390 LLMPipeline + FastDraft K=5: 27.2 tok/s (+55%)

For extractive RAG (4K input, ~99 actual token summary):
- charlie 140V LLMPipeline plain: ~20 tok/s
- charlie 140V LLMPipeline + PL: ~28 tok/s (+40%)

## Decision Matrix is still valid

The DECISION_MATRIX.md guidance is still correct (FastDraft for short
input, PL for extractive RAG, etc.) — those are RELATIVE recommendations
based on workload patterns. The absolute throughput numbers in the matrix
should be discounted by ~5×.

## Why the bench was wrong

The bench script defaulted to `n = max_tokens` if perf_metrics didn't
return num_generated_tokens. perf_metrics in OV 2026.1 LLMPipeline often
returns this as None or 0 for greedy short outputs, causing the fallback
to inflate. Should always use `tokenizer.encode(text).input_ids.shape[-1]`.
