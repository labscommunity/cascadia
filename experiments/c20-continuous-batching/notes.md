# c20: Continuous batching with SchedulerConfig

## Setup

- Llama 3.1 8B INT4 on alpha B390 GPU.
- LLMPipeline with `SchedulerConfig(dynamic_split_fuse=True, cache_size=4 GB, max_num_batched_tokens=8192)`.
- Batched call `pipe.generate([prompts], cfg)` with N independent prompts, 64 tokens each, greedy.

## Results

| Batch | Decode (s) | Aggregate tok/s | Per-request tok/s |
|---|---|---|---|
| 1 | 0.48 | **133.77** | 133.77 |
| 2 | 3.57 | 35.89 | 17.95 |
| 4 | 3.68 | 69.65 | 17.41 |
| 8 | 3.71 | **137.92** | 17.24 |

## Interpretation

- Batch=1 is the no-batching path (fast single forward).
- For batch ≥ 2, per-request throughput drops to ~17.5 tok/s, but
  aggregate throughput scales LINEARLY with batch size up to at
  least batch=8.
- At batch=8, aggregate matches batch=1 — i.e. you can serve 8
  concurrent chat sessions at the same total tok/s as one fast
  session, with the cost paid by each user (each gets 17 vs 134).

## Implications

- **For single-user serving (chatbot, code completion):** stick with
  batch=1 + LLMPipeline + FastDraft. 134 tok/s on alpha B390.
- **For multi-tenant serving (cloud API, classroom):** continuous
  batching gives you ~8× concurrent users at the cost of per-user
  latency. Worthwhile when concurrency demand exceeds 1-2.

## Without SchedulerConfig

For comparison, `pipe.generate([prompts], cfg)` WITHOUT `SchedulerConfig` (c20-1, no-CB path):

| Batch | Aggregate tok/s |
|---|---|
| 1 | 133.0 |
| 2 | 29.0 |
| 4 | 55.1 |

So the SchedulerConfig+CB path gives ~25% MORE aggregate throughput at
batch=2-4 vs the no-scheduler path. CB IS engaged.

## Open follow-up

- Test batch=16, batch=32 — does aggregate keep scaling, or does it
  plateau because GPU compute saturates?
- Test with FastDraft + batched generation. Spec decode in CB mode is
  more complex; LLMPipeline may or may not support it.
- Wire concurrent-request handling into tahoma's `OVGenAIEngine` — the
  CB-mode pipe lets multiple `pipe.generate()` calls overlap, which is
  what real serving needs.
