# c36: Adaptive K via assistant_confidence_threshold at long input

## Setup
Llama 3.1 8B INT4 + FastDraft 150M on alpha B390 GPU. Long passage input
(~1024 tokens) + summary instruction, 32-token output. Sweep
`assistant_confidence_threshold` ∈ {0.3, 0.5, 0.7}.

`assistant_confidence_threshold` is mutually exclusive with `num_assistant_tokens`
(set the threshold in GenerationConfig, do not set num_assistant_tokens).

## Results

| threshold | tok/s |
|-----------|------:|
| 0.3       | 21.80 |
| 0.5       | 21.33 |
| 0.7       | 20.96 |

## Comparison to other configs at same input/output

| Engine                                        | tok/s | vs adaptive 0.3 |
|----------------------------------------------|------:|----------------:|
| LLMPipeline + FastDraft K=5 (fixed, c34)     | 18.43 |          -15% |
| LLMPipeline plain (c34)                       | 18.21 |          -16% |
| **LLMPipeline + FastDraft adaptive thr=0.3** | **21.80** | (winner) |

## Findings

1. **Adaptive K with low threshold (0.3) beats fixed K=5 by 18%** at long
   input + short output. Beats plain by 20%.
2. **Lower threshold = more permissive = more draft tokens accepted**.
   At 0.3 the draft model's "low confidence" tokens are still accepted
   (subject to verification), keeping the spec round productive.
3. **This contradicts c18's finding** (adaptive K hurt -22% on short input).
   Why: on short input, fixed K=5 was already highly accept-rate-friendly
   (factual short answer) so adaptive's caution wasted opportunity. On
   long input + summarisation, fixed K=5 over-speculates (low accept rate);
   adaptive at 0.3 lets the draft contribute productively when it's at
   least somewhat aligned.

## Recommendation

For tahoma's `ov-genai` engine, when inputs are large (≥1K tokens), use
`assistant_confidence_threshold=0.3` instead of `num_assistant_tokens=5`.
Need to add a new flag `--ov-spec-threshold` to expose this.

## Limitation

This was tested with a specific summarisation prompt. For other long-input
workloads (e.g., long-input + long-output, or different content patterns)
the optimal threshold may differ.

## Open follow-ups

- Same sweep on charlie 140V.
- Test 0.1, 0.2, 0.4 to find finer-grained sweet spot.
- Test with output=128, 256 to see if win shrinks at long output.
- Update tahoma engine to expose this flag.
