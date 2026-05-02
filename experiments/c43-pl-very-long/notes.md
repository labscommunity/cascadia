# c43: Prompt Lookup at very long inputs (2K, 4K) on charlie 140V

## Setup
Llama 3.1 8B INT4 + LLMPipeline + `prompt_lookup=True` on charlie 140V GPU.
Input is a synthetic distributed-systems passage at ~{2048, 4096} tokens
+ summary instruction. 64-tok output. Sweep `max_ngram_size ∈ {2, 3, 5}`.

## Results

| Input | n=2 | n=3 | n=5 |
|-------|----:|----:|----:|
|  2048 | 34.3 | 35.5 | 34.0 |
|  4096 | 32.8 | 32.8 | 32.8 |

## Findings

1. **PL works at 4K input** — 33 tok/s end-to-end vs ~21 tok/s plain decode
   estimate from c32 (different platform but similar profile). **~+57%
   over plain at 4K input.**
2. **ngram size 2-5 makes no significant difference** at very long inputs.
   The lookup table size grows with input but the win-per-match also
   grows; effects cancel.
3. **PL win is roughly constant from 1K → 4K input** (~32-35 tok/s
   throughout). The win does NOT scale with input length — once the
   pattern reuse matches a sufficient n-gram, the per-step savings are
   the same.

## Recommendation

Keep `prompt_lookup=True, max_ngram_size=3` as the default for
RAG / summarization workloads at any input length up to at least 4K.
The win is robust and the cost (lookup table) is negligible.

## Open
- Test 8K input — does the win hold? (Would need to validate context
  fits and KV doesn't OOM.)
- Test PL with output ≥ 256 to see if longer outputs amplify or dampen
  the win.
