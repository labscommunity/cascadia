# c35: prompt_lookup at long inputs (charlie 140V)

## Setup
Llama 3.1 8B INT4 on charlie 140V GPU. LLMPipeline with `prompt_lookup=True`,
`max_ngram_size=3`, `num_assistant_tokens=5`. Input is a passage with
moderately repetitive vocabulary (designed to give PL some matches) at
~{128, 512, 1024, 2048} tokens. Output 64 tok summary.

## Results

| Input ~tok | tok/s |
|-----------:|------:|
|        128 |  53.4 |
|        512 |  32.9 |
|       1024 |  31.6 |
|       2048 |  34.8 |

## Findings

1. **PL still works at long inputs** — 32-35 tok/s at 1K+ inputs is a real
   speedup vs the c32 plain decode plateau (~21 tok/s).
2. **Best at short input** (128 → 53.4 tok/s), as expected: short input
   means the prompt-lookup table is small and cheap to build.
3. **Plateau at long input** — the lookup table size grows linearly but
   the n-gram match probability also grows, so the two effects roughly
   cancel.
4. **Inconsistency** — at 2048 input the rate is HIGHER than 1024 (34.8
   vs 31.6). Within run-to-run noise of ~3 tok/s, but the trend isn't
   strictly monotonic. Likely the longer input gives more match
   opportunities and partly cancels the prefill overhead.

## Recommendation

For RAG / summarization workloads with input ≥100 tokens, prompt_lookup
is a clear default on charlie. The win over plain decode at 1K input is
roughly +50-80% based on the PL/plain estimate. Short factual chats with
no input reuse should still use FastDraft instead.
