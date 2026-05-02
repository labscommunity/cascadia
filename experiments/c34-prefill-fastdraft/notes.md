# c34: FastDraft K=5 at long input — CORRECTED — TIED with plain

## Setup
Llama 3.1 8B INT4 + FastDraft 150M K=5 on alpha B390 GPU. Long synthetic
input passage at ~{128, 512, 1024, 2048, 4096} tokens followed by
"Given the passage above, what is one key challenge in distributed systems?
Answer in one sentence."

## Sweep 1 — input length, fixed output=32 (FastDraft only)

| Input ~tok | FastDraft tok/s |
|-----------:|----------------:|
|        128 |            31.0 |
|        512 |            17.3 |
|       1024 |            18.2 |
|       2048 |            17.7 |
|       4096 |            20.1 |

## Sweep 2 — output length, fixed input=1024 (FastDraft vs plain)

| Output | FastDraft tok/s | Plain tok/s | Δ |
|-------:|----------------:|------------:|--:|
|     32 |            18.4 |        18.2 | +1% (tied) |
|     64 |            33.5 |        32.0 | +5% (tied) |
|    128 |            65.7 |        64.8 | +1% (tied) |
|    256 |           125.8 |       127.4 | -1% (tied) |

**FastDraft brings ZERO benefit over plain LLMPipeline at long input.**
Tied within run-to-run variance (~5%).

## Why this revises Discovery #2

The original c18 discovery measured FastDraft +24% on **short input**
("What is the capital of France?", ~5 tokens). For long-input workloads
the +24% disappears.

Two converging reasons:
1. **Per-pass attention compute**: at 1024+ KV, both target and draft are
   bottlenecked by KV-attention. The draft model's smaller MLP/QKV layers
   don't help — the per-step time ratio between draft and target shrinks
   from ~10x (short input) to ~3-5x (long input).
2. **Accept rate**: FastDraft 150M was trained on factual-style data. For
   summarization-of-passage tasks the model output is less predictable
   given the draft's outputs, so accept rate drops. Lower accept rate ×
   higher per-step cost = no net win.

## REVISED decision matrix

| Input | Output | Recommended engine |
|-------|--------|--------------------|
| <100  | any    | LLMPipeline + FastDraft K=5 (short out) / K=3 (long out) |
| 100-1K | <64    | LLMPipeline + Prompt Lookup (if RAG) / plain |
| 100-1K | 64+    | LLMPipeline + Prompt Lookup (if RAG) / FastDraft |
| 1K+   | any    | **LLMPipeline plain** (FastDraft brings nothing) |

The sweet spot for FastDraft is **short input + medium output**.

## Quality cross-check (anecdotal)

The first_text outputs of plain and FastDraft at 1024 input + 256 output
are similar in length and content style — both produce coherent
responses about latency/fault-tolerance/etc. No evidence of FastDraft
breaking quality.

## What we now know about per-token decode rate at 1024 KV

- Pure decode rate (excluding TTFT) at 1024 input:
  - 32 out: ~20 tok/s
  - 256 out: ~142 tok/s
- The increase with output length is suspect — pure-decode should be
  roughly constant per token. Possible explanations:
  - KV grows during decode → later tokens slower → but average across
    32-256 should be similar regardless of output length cap.
  - perf_metrics `ttft` may not represent pure prefill time. Likely it
    accounts for one decode step too. Treat numbers as end-to-end.
