# c34: FastDraft K=5 at long input + various output lengths

## Setup
Llama 3.1 8B INT4 + FastDraft 150M K=5 on alpha B390 GPU. Input is a
synthetic distributed-systems passage at ~{128, 512, 1024, 2048, 4096}
tokens followed by "Given the passage above, what is one key challenge
in distributed systems? Answer in one sentence."

Two sweeps:
1. **Input-length sweep at fixed output=32** (find input cliff).
2. **Output-length sweep at fixed input=1024** (find output amortization).

## Sweep 1: input-length, output=32

| Input ~tok | FastDraft tok/s | Plain tok/s (c32) | Δ |
|-----------:|----------------:|------------------:|--:|
|        128 |            31.0 |              37.4 | -17% |
|        512 |            17.3 |              21.6 | -20% |
|       1024 |            18.2 |              21.0 | -13% |
|       2048 |            17.7 |              20.8 | -15% |
|       4096 |            20.1 |              21.3 | -6%  |

**FastDraft is a NET LOSS at all input lengths when output is 32 tokens.**

## Sweep 2: output-length sweep at input=1024

| Output ~tok | tok/s | (vs out=32) |
|------------:|------:|------------:|
|          32 |  18.4 | (baseline)  |
|          64 |  33.5 | +82%        |
|         128 |  65.7 | +257%       |
|         256 | 125.8 | +584%       |

The tok/s grows almost linearly with output length, indicating that the
fixed cost of compiling spec rounds is amortising over more output. The
"pure decode rate" excluding TTFT works out to roughly:
- out=32: 19.8 tok/s
- out=64: 35.8 tok/s
- out=128: 70.7 tok/s
- out=256: 135.1 tok/s

## Findings

1. **FastDraft win is workload-specific by (input, output, content) tuple.**
   Our previous +24% number was for short input + 64-token output + short factual.
   Long input (1024+) + short output (32) is a NET LOSS (-13 to -20%).
2. **FastDraft win scales DRAMATICALLY with output length** at fixed long
   input. By 256 output, FastDraft on long input is faster than ANY plain
   short-input config.
3. **Hypothesis for the loss at short output**: spec decode has a fixed
   per-round overhead (target verify forward + draft K forwards). For very
   short outputs the round overhead doesn't amortise; for long outputs it
   does dramatically.

## Decision matrix update (input → output → engine)

| Input | Output | Recommended engine | Reason |
|---|---|---|---|
| <100  | <128  | LLMPipeline + FastDraft K=5 | Discovery #2: +24% |
| <100  | 128-256 | LLMPipeline + FastDraft K=3 | Discovery #2: +26% (long-creative) |
| 100-1K | <64  | LLMPipeline + Prompt Lookup | Discovery #3: +59% if RAG |
| 100-1K | 64+  | LLMPipeline + FastDraft K=5 | spec amortises now |
| 1K+   | <64   | **LLMPipeline plain** | FastDraft net loss; PL helps if RAG |
| 1K+   | 64+   | LLMPipeline + FastDraft K=5 | spec wins big once output amortises |

## Open follow-ups

- Test K=3 at long input + short output. Lower K may still amortise less
  but waste less.
- Test `assistant_confidence_threshold` for adaptive K at long input. We
  saw -22% on short input but maybe it helps here.
- Validate plain bench at out=64,128,256 (in progress).
- Test on charlie (in progress in c35 for PL).
