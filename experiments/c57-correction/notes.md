# c57: MAJOR CORRECTION — PL extractive bench was inflated by EOS-early behavior

## Summary

The bench scripts in c45-c56 reported `tok_s = max_tokens / total_dt` instead
of `actual_generated_tokens / total_dt`. For the **extractive summary prompt**
("Summarize the passage above in 2 short sentences"), the model EOSes at
~99 tokens regardless of the max_tokens cap. So all "256 output", "512 output",
"1024 output" numbers were measuring the SAME ~99-token actual output, just
divided by different inflated max_tokens values.

## Verification

charlie 140V GPU, Llama 3.1 8B INT4, 4K input + 512 max_tokens cap, extractive:

| Mode | total dt | actual tokens | actual tok/s | bench claimed |
|------|---------:|--------------:|-------------:|--------------:|
| plain | 4.97s | **99** | **19.94** | 199.57 (10× inflated) |
| PL | 3.53s | **99** | **28.02** | 388.81 (14× inflated) |

The model hits `<eos>` after generating ~99 tokens (a 2-sentence summary,
plus model preamble). max_tokens=512/1024 is never reached.

## Corrected Discovery #3 numbers

**Discovery #3 RELATIVE win is still real and valid:**
- PL extractive vs plain: **+40% to +50% actual win** (not +94% as previously
  claimed in c54). Both bench runs hit the same EOS-at-99 behavior, so the
  RELATIVE win between modes is preserved (PL takes less time to produce the
  same 99 tokens).

**Discovery #3 ABSOLUTE numbers are inflated:**
- "388 tok/s PEAK" was actually ~28 tok/s (verified above)
- "194 tok/s at 4K + 256" was actually ~38-50 tok/s
- "160 tok/s at 1K + 256" was actually ~50-65 tok/s

The bench ratio is `inflated/actual ≈ max_tokens / actual_generated`.

## What WAS correctly measured

- **Plain LLMPipeline 5-token input + 64-token output factual**: model
  generates ~20-30 tokens before EOS for this short factual prompt.
  Inflation factor ~2x for the c1 96 tok/s baseline → real ~50 tok/s.
- **FastDraft +24% for short factual** (c18): both modes hit same EOS,
  relative win is correct.
- **CB multi-tenant** (c20, c41): each prompt produces ~99-128 tokens
  depending on topic. The aggregate calculation `batch * max_tokens / dt`
  is ALSO inflated proportionally. Relative scaling (batch=8 vs batch=32
  comparison) is correct, absolute throughput overstated.
- **NPU vs GPU concurrent**: relative comparison correct, absolute likely
  ~2x inflated.

## Impact on conclusions

1. **All RELATIVE findings stand**:
   - LLMPipeline 10× over OVModelForCausalLM (DISCOVERY #1) — short-output
     prompts where EOS-early is similar; relative ratio is robust.
   - FastDraft +24% short input chat (DISCOVERY #2) — relative ratio robust.
   - PL +40-50% on extractive workloads (DISCOVERY #3) — REVISED from +94%.
   - NPU concurrent serving (-3% GPU cost) (DISCOVERY #4) — relative ratio
     robust.
2. **All ABSOLUTE throughput numbers need a correction factor** roughly equal
   to `actual_tokens / max_tokens`:
   - Short factual (~30 actual): correction ~0.5x
   - Medium factual (~99 actual at 256 cap): correction ~0.4x
   - "Long" extractive (~99 actual at 512 cap): correction ~0.2x
   - "Very long" extractive (~99 actual at 1024 cap): correction ~0.1x

## Corrected leaderboard (best-effort)

For real chat workloads (not capped at unrealistic max_tokens):

| Workload | Best engine | tok/s (corrected) | Hardware |
|---|---|---|---|
| Short factual chat (~30 actual tokens) | LLMPipeline + FastDraft K=5 | ~67 | alpha B390 |
| Extractive RAG (~100 actual tokens, 1K-4K input) | LLMPipeline + PL | ~28-30 | charlie 140V |
| Extractive RAG plain (~100 actual tokens, 1K-4K input) | LLMPipeline plain | ~20 | charlie 140V |
| Multi-tenant CB batch=32 short factual | LLMPipeline + CB | ~280 aggregate | alpha B390 |
| Concurrent NPU+GPU | mixed | ~50 effective per req | charlie 140V |

## Lessons for future autolab work

1. **ALWAYS count actual generated tokens via tokenizer**, not max_tokens cap.
2. **Use a prompt that doesn't hit EOS** if you want to test long-output perf
   (e.g., "Write a 500-word essay on..." rather than "Summarize in 2 sentences").
3. **perf_metrics.num_generated_tokens is unreliable** — can return None, can
   return 0, can return cap. Always cross-check with tokenizer.
4. **Be skeptical of multi-x improvements that seem too good** — physics says
   8B INT4 on Lunar Lake iGPU caps at ~150-200 tok/s by memory bandwidth.
   Anything claiming 388 tok/s should be verified.
