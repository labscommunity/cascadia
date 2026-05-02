# c21: Prompt Lookup decoding for RAG/summarization

## Setup

- Llama 3.1 8B INT4 on alpha B390 GPU.
- LLMPipeline with `prompt_lookup=True`.
- GenerationConfig: `num_assistant_tokens=5`, `max_ngram_size=3`.
- 128 generated tokens, prompt is a passage about distributed inference
  followed by "Summarize the passage above in 2 short sentences."
  — the model's output naturally reuses ~50% of the input vocabulary.

## Results

| Engine config | tok/s | vs no-lookup |
|---|---|---|
| LLMPipeline plain (no lookup) | 57.69 | (baseline) |
| **LLMPipeline + prompt_lookup** | **91.57** | **+58.7%** |

## Why this is a discovery

Prompt Lookup decoding is "free" speculative decoding — it predicts the
next N tokens by **matching the last N-gram of the generated output
against substrings of the input prompt**. When the output reuses input
text (RAG, summarization, code completion-in-context, instruction
following with quoted material), the match rate is high and the
speculative tokens are accepted.

Compared to FastDraft:

| Workload | FastDraft 150M K=5 | Prompt Lookup |
|---|---|---|
| Short factual (no input reuse) | 119 tok/s (+24%) | (no benefit expected) |
| RAG summarization | (would need to test) | **91.6 tok/s** (+59%) |

The win comes from **zero draft cost**: instead of running a 150M
parameter draft for each spec round, prompt lookup is a constant-time
n-gram lookup against the input. For 1K-2K input prompts the lookup
table builds in microseconds.

## Recommendations to encode

- For RAG / summarization / "rewrite this" workloads: use
  `prompt_lookup=True` + `num_assistant_tokens=5` + `max_ngram_size=3`.
- For pure-generation chat (no input reuse): stick with FastDraft.
- For mixed workloads: TODO — test if prompt_lookup adds value when
  input reuse is low, or if it just adds overhead.

## Open follow-ups

- Try with a longer prompt (4K input → does the lookup table still
  build fast enough?).
- Try `max_ngram_size=4,5` for higher accept rate on long prefix
  matches.
- Test on charlie 140V — does the win hold on Lunar Lake?
- Wire `--prompt-lookup` flag into the tahoma `ov-genai` engine.
