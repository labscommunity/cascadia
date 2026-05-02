# c25: prompt_lookup max_ngram_size sweep on charlie 140V

## Setup
Llama 3.1 8B INT4 on charlie GPU. 128-tok output. RAG passage + summarise
prompt (same as c21). 3 runs each, best-of-3.

## Results (serial — single process per run)
| max_ngram_size | tok/s |
|---|---|
| 2 | 102.41 |
| 3 | 102.63 |
| 4 | 104.26 |
| 5 | 101.58 |

All within ~3% of each other; n choice doesn't matter much in this range.
Default n=3 is fine.

## Anomaly resolved
A first attempt running n=2/4/5 *in parallel* on the same GPU gave 32-36
tok/s for each — a 3× regression. This was GPU contention, not a real
property of the config. **Lesson for future benches: never run multiple
LLMPipeline processes against the same physical GPU concurrently — they
serialise on kernel queue and inflate latency without raising aggregate
throughput.** The continuous-batching path inside one LLMPipeline does
the right thing; multiple OS processes do not.

## Recommendation to encode
Keep `max_ngram_size=3` as the default for prompt_lookup. n=4 is slightly
better in this RAG benchmark but the difference is within run-to-run noise.
