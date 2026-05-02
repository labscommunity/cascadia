# tahoma Inference Engine Decision Matrix (Intel GPUs, OV 2026.1)

Compiled from c0-c38 experiments on alpha (Battlemage B390 dGPU) and
charlie (Lunar Lake 140V iGPU + NPU 4).

## Quick start: which engine for which job

```
                 ┌─────────────────────────────────────────────────────┐
                 │                Input length tokens                 │
input → output   ├──────────────┬─────────────────┬──────────────────┤
                 │   <100        │   100-1000      │   1000+          │
─────────────────┼──────────────┼─────────────────┼──────────────────┤
short out (<64)  │ FastDraft K=5 │ FastDraft K=5   │ Plain LLMPipeline│
                 │ (+24%)        │ or PL (if RAG)  │ or adaptive K    │
                 │               │                  │ (+8% best case)  │
─────────────────┼──────────────┼─────────────────┼──────────────────┤
medium out 64-256│ FastDraft K=5 │ Plain or       │ Plain LLMPipeline│
                 │ K=3 for 256   │ FastDraft       │                  │
─────────────────┼──────────────┼─────────────────┼──────────────────┤
long out 256+    │ FastDraft K=3 │ Plain           │ Plain            │
                 │ (+26%)        │                  │                  │
─────────────────┴──────────────┴─────────────────┴──────────────────┘
```

If RAG/summarization (output reuses input): `prompt_lookup=True, max_ngram_size=3`
gives +50-65% on charlie at 100-1K input. Mutually exclusive with FastDraft.

## Plugin properties (do NOT override on Battlemage)

`KV_CACHE_PRECISION`, `DYNAMIC_QUANTIZATION_GROUP_SIZE`, `GPU_QUEUE_THROTTLE`,
`GPU_HOST_TASK_PRIORITY`, `INFERENCE_PRECISION_HINT`, `NUM_STREAMS`, `PERFORMANCE_HINT`:
all default values are optimal. Explicit overrides are no-op or regressions.

The ONE knob worth setting:
- `CACHE_DIR=<path>` — persists kernel JIT compile cache across loads
  (cold-start TTFT win, see c38 for measurement).

## Multi-tenant continuous batching

Enable explicit `SchedulerConfig(enable_prefix_caching=True, dynamic_split_fuse=True)`
on the LLMPipeline only when serving 4+ concurrent requests. Per-request
throughput drops from 134 → 17 tok/s but aggregate scales to 138-149 tok/s
at batch=8 on both alpha and charlie.

## Concurrent multi-model serving (NEW for Intel)

Llama 3.2 1B INT4 runs on NPU at 53-91% of GPU speed.
Run 8B-on-GPU + 1B-on-NPU concurrently; aggregate throughput beats serial
by ~34%. Note the NPU loading penalty (113→42 tok/s when GPU pipe also
loaded) is not yet diagnosed — open follow-up.

## Cross-platform

| Hardware | LLMPipeline plain 8B INT4 (64 tok) | + FastDraft 150M K=5 |
|---|---|---|
| alpha B390 GPU | 96 tok/s | 119 (+24%) |
| charlie 140V GPU | 91 tok/s | 96 (+5%) |

K=5 is the right K on both platforms. Adaptive K thr=0.3 gives +8% on
alpha and +3% on charlie at long input.

## Long input regression (real-world RAG implication)

Plain LLMPipeline 8B INT4 at long input degrades dramatically vs short
input due to KV-attention overhead:

| Input ~tok | Plain tok/s |
|-----------|------------:|
|         5 | 96 (Discovery #1 baseline) |
|       128 | ~37        |
|       512+ | ~21        |
|       4096 | ~21        |

**Headline 96 tok/s is for short input only.** Real RAG deployments at
1K+ input get 21 tok/s plain decode rate (or +5-8% with adaptive K).
