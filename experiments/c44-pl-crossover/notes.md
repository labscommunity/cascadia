# c44: PL crossover point — when does PL stop helping?

## Setup
charlie 140V (Lunar Lake) GPU, Llama 3.1 8B INT4. 64-token output.
Synthetic distributed-systems passage at varied input lengths.
Compare plain LLMPipeline vs LLMPipeline + prompt_lookup.

## Results

| Input | Plain tok/s | PL n=3 tok/s | Δ |
|-------|-----------:|-------------:|--:|
|  256 (c21) | 66.16      | 108.82       | **+65%** ← original Discovery #3 finding |
|  512  | 36.35      | 32.94 (c35)  | -9% |
| 1024  | 34.83      | 31.55 (c35)  | -9% |
| 2048  | 33.59      | 35.53 (c43)  | +6% |
| 4096  | 38.38      | 32.79 (c43)  | -15% |

## Findings — IMPORTANT REFINEMENT TO DISCOVERY #3

1. **The c21 +65% PL win was workload-specific.** With a SYNTHETIC
   passage (constructed by repeating sentences), PL is at best tied
   with plain and at worst -15%.
2. **The c21 prompt was a real RAG-style summary** where the answer
   literally quoted the passage's vocabulary, giving high n-gram match
   rate. The synthetic passage in c32/c34/c35/c43/c44 has lower
   match-able vocabulary in the model's natural answer.
3. **PL is HIGHLY content-dependent.** For deployment, A/B testing PL
   on the actual workload is essential.

## REVISED Discovery #3 statement

PL gives a strong win (+50-65%) for **specific RAG/summarization
workloads where the model's output quotes input vocabulary verbatim**.
For other inputs (technical text + open-ended question), PL is at
best tied with plain. The technique is real but the magnitude is
workload-specific.

## Workload-detection heuristic for tahoma

Don't enable PL by default. Provide a `--prompt-lookup` opt-in flag
and document:
- Best for: extractive summarization, code completion-in-context,
  "rewrite this in style X", direct Q-over-passage where model output
  is mostly extractive.
- Not for: open-ended QA, creative writing, code generation from spec,
  anything where model's output diverges significantly from input
  vocabulary.

## Why the c21 prompt worked but synthetics don't

c21 prompt: a passage about distributed inference + "Summarize the
passage above in 2 short sentences." — the model copies key phrases
("distributed inference", "pipeline parallelism", "tensor parallelism",
etc.) directly into its answer. The accept rate is high.

c34 prompt: similar passage + "Given the passage above, what is one
key challenge in distributed systems? Answer in one sentence." — the
model answers in its own words ("Latency is a key challenge..."), not
quoting the passage. Accept rate is much lower.

The summarization VS analytical-question distinction matters more than
input length.
