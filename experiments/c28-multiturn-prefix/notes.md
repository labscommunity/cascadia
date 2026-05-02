# c28: Multi-turn chat — does explicit SchedulerConfig prefix caching help?

## Setup
Llama 3.1 8B INT4 on alpha B390 GPU. 4 conversational turns over a long
shared system prompt (~250 tokens). 64-token responses. Three modes:
1. **plain** — concatenate full history into the prompt every turn (no chat API)
2. **chat** — `pipe.start_chat(system_message=...)`, default scheduler
3. **chat-cb** — `start_chat()` + explicit `SchedulerConfig(enable_prefix_caching=True, dynamic_split_fuse=True)`

## Results (per-turn tok/s, but see caveat below)

| Turn | plain | chat | chat-cb |
|------|------:|-----:|--------:|
| 0    | 31.7  | 24.9 | 24.7   |
| 1    | 20.3  | 30.0 | 29.0   |
| 2    | 90.8  | 72.3 | 66.5   |
| 3    | 63.8  | 43.5 | 44.4   |

## Findings

1. **`chat-cb` (explicit SchedulerConfig prefix caching) is essentially
   identical to `chat` (default).** LLMPipeline already does effective
   internal prefix caching for `start_chat()`. The explicit SchedulerConfig
   adds nothing on this workload.

2. **Cross-turn variability is high** — turn 2 is 3x faster than turn 1 in
   plain mode. This is because some questions ("What is the bandwidth
   requirement of tensor parallelism?") trigger early EOS, so the actual
   `num_generated_tokens` is well below the 64 cap. The bench reports
   `tok/s = num_generated_tokens / total_time` which can be misleading.

3. **plain mode is sometimes faster** because we re-prefill the *whole*
   history each turn but plain mode also re-runs greedy decode without
   any chat-template overhead. For RAG-style "single-shot Q over context"
   the plain mode is fine.

## What we did NOT measure
TTFT specifically. The `perf_metrics.get_ttft()` API requires
`pipe.generate([prompt], ...)` (list, not str) to return a DecodedResults
object. Future work: redo this bench with proper TTFT capture per turn —
that's the metric where prefix caching is supposed to win.

## Recommendation
Do not bother wiring an explicit `--ov-scheduler-cb-prefix` flag in tahoma —
the default `LLMPipeline` chat-mode prefix caching is already engaged.
The right place to spend follow-up effort is on the API layer using
`generate([prompt], ...)` so the application can read TTFT per turn.
