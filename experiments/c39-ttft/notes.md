# c39: TTFT-aware bench at 1024 input + 32 output (alpha B390)

## Setup
Llama 3.1 8B INT4 on alpha B390. Same seed prompt across all modes.
1024-token input, 32-token output. Best of 3 runs (vs c36's best of 2).

## Results

| Mode | best tok/s | TTFT ms | Total dt s |
|------|-----------:|--------:|-----------:|
| plain | **20.04** | **105** | 1.60 |
| FastDraft K=5 fixed | 18.67 | 118 | 1.71 |
| FastDraft adaptive thr=0.3 | 19.60 | 126 | 1.63 |
| Prompt-lookup n=3 | 20.10 | 128 | 1.59 |

## Findings

1. **plain LLMPipeline is the best (or tied) at long input + short output.**
   tok/s: 20.04 plain vs 20.10 PL (tied), 19.60 adaptive K (-2%), 18.67
   fixed K=5 (-7%). TTFT: plain wins decisively (105 vs 118-128).
2. **C36's claimed +8% adaptive K over plain (21.80 vs 20.24) does NOT
   reproduce with best-of-3.** With more samples the gap closes to within
   noise. Earlier conclusion overstated.
3. **Bench-to-bench variance is large (~10%)** at 1024 input + 32 output.
   Any sub-10% claim needs ≥5 runs to be reliable.

## Methodology lesson

At long input + short output, the prefill dominates total time and small
variations in compile state, KV warm-up, and disk I/O drown out
sub-10% effects. For these workloads:
- Always run ≥3 trials and report best-of-N + variance.
- Beware "first run cold" effects — discard run 1 or use it as warmup.
- Tie-breaks should rely on TTFT (more reliable than total tok/s for
  short outputs).

## REVISED decision matrix

| Input | Output | Best engine (validated) |
|-------|--------|-------------------------|
| <100  | any    | LLMPipeline + FastDraft K=5 (or K=3 long out) |
| 100-1K | any   | LLMPipeline + Prompt Lookup (RAG) / plain |
| 1K+   | <64    | **LLMPipeline plain** (lowest TTFT, ties for tok/s) |
| 1K+   | 64+    | LLMPipeline plain (Discovery #2 caveat: FastDraft tied) |

**Adaptive K thr=0.3 is NOT a reliable win at long input.** The c36
finding was within noise.

## Open follow-ups

- Run same TTFT bench on charlie 140V.
- Test at input=4096 to see if any spec method wins at very long input
  (where compute is even more attention-dominated).
