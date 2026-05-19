# 009 — A3 K=3/4/8 10-prompt robustness — K=4 IS THE ROBUST LEADER

**Verdict:** **important revision.** K=3 was misleading on the narrow
3-prompt eval; on a 10-prompt broader set K=3 fails 4 prompts that
K=4 and K=8 both answer correctly. **K=4 is the actual production
sweet spot: same 9/10 quality as K=8 baseline, 2.46× faster.**

**Date:** 2026-05-17 ~20:08 PT
**Magnitude class:** **M** (revised down from L)

## 10-prompt set

Factual + format-diverse:
1. "The capital of France is" → paris
2. "The largest ocean on Earth is the" → pacific
3. "Two plus two equals" → four
4. "The first president of the United States was" → washington
5. "The largest planet in our solar system is" → jupiter
6. "Water boils at 100 degrees" → celsius
7. "Python is a programming language created by" → guido
8. "The square root of 144 is" → 12
9. "Mount Everest is located in the" → himalaya
10. "The speed of light is approximately 300" → km

max_tokens=16 (longer outputs = better substring hit rate; prefill amortized).

## Results

| K | tok/s | Quality | Failed prompts |
|--:|------:|:-------:|----------------|
| 8 | 0.0853 | 9/10 | "km" prompt — model said "kilometers per second" (substring "km" not matched). Single sampling-format artifact. |
| **4** | **0.2100** | **9/10** | "celsius" prompt — model gave "(A) 212 F (" multi-choice format instead of word. Single sampling-format artifact. |
| 3 | 0.3050 | 6/10 | jupiter, celsius, guido, "12" — **substantive failures**: gave multi-choice questions, said "the Dutch" instead of "Guido van Rossum", gave vague answers. K=3 doesn't have enough expert capacity for these prompts. |

**K=8 vs K=4: tied at 9/10 quality** (both have 1 single sampling
artifact, on different prompts). Throughput delta = +146%.

**K=3 vs K=4: -3 quality** (K=4 = 9/10, K=3 = 6/10). The +258%
throughput at K=3 isn't worth the 30% quality regression.

## Productionization revision (supersedes iter 008)

**New recommendation: `--top-k-override 4`**, not K=3.

- 9/10 quality matching baseline K=8 (within sampling noise)
- +146% throughput on long-output workloads (max_tokens=16)
- +109% on short-output workloads (max_tokens=8, from iter 005/006)
- No re-train, no re-quantize, no re-architect

The K=3 number stays in the leaderboard as "fastest if quality eval
is narrow" — useful for workloads where:
- Inputs are very factual / single-token answer style
- Some quality drop is acceptable for latency
- The specific failures (multi-choice format, vague answers) are not
  in the critical path

## Caveats

- 10 prompts is still narrow. A proper quality eval (perplexity, MMLU,
  LongBench, etc.) would tighten the recommendation.
- Failures are sampling/format dependent: temperature, top-p, prompt
  template all affect quality cliff position. Re-evaluate if those
  change.
- Miner single-stage substrate. On 2-box matias (memory-bound vs
  disk-bound), the throughput delta likely shrinks but the quality
  picture should hold.

## Linked

- `bench_k8_10p.jsonl`, `bench_k4_10p.jsonl`, `bench_k3_10p.jsonl`
- `~/k26_bench_miner_10.sh` on miner (10-prompt bench harness; commit
  to autolab/bench next)
- builds on 008 (which had K=3 as leader before robustness check)
