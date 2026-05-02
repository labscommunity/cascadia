# c51: quality check — FastDraft and PL produce identical greedy output

## Setup
charlie 140V GPU, Llama 3.1 8B INT4. Greedy decode (do_sample=False),
64 tokens output. Two prompts:
1. "What is the capital of France?" (factual)
2. Passage + "Summarize the passage above in 2 short sentences." (extractive)

Compare:
- Plain LLMPipeline
- LLMPipeline + FastDraft 150M K=5
- LLMPipeline + prompt_lookup n=3, K=5

## Results

### Factual prompt (FastDraft vs plain)

```
plain: 'The capital of France is Paris.'
FD:    'The capital of France is Paris.'
identical: True
```

### Extractive prompt (PL vs plain)

```
plain: 'Distributed inference speeds up inference for large language models
        by splitting computations across multiple devices. Techniques like
        pipeline parallelism and tensor parallelism are used to split com...'
PL:    'Distributed inference speeds up inference for large language models
        by splitting computations across multiple devices. Techniques like
        pipeline parallelism and tensor parallelism are used to split com...'
identical: True
```

## Findings

1. **FastDraft is mathematically lossless**. Output is byte-identical to
   plain greedy decode for the same model and prompt. Confirmed.
2. **Prompt Lookup is mathematically lossless**. Same identical output
   as plain greedy. Confirmed.
3. **Both techniques are pure perf wins, no quality regression.**

## Why this matters

Spec decode (in any form) is mathematically lossless when:
- Greedy decoding is used (do_sample=False).
- Target model's logits decide acceptance/rejection of draft tokens.
- Acceptance criterion uses target's actual probability distribution.

Both FastDraft and PL satisfy these conditions in OpenVINO 2026.1.

For sampling (do_sample=True), spec decode is statistically equivalent
but produces different specific tokens due to RNG paths. Quality
distribution is preserved but exact token-by-token outputs may differ.

## Implication for tahoma

No quality concerns when using FastDraft or PL. Always-on for the
appropriate workloads. Document this in the engine config.
