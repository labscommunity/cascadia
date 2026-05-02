# c52: Real multi-tenant — 8B CB batch=16 + 1B NPU concurrent

## Setup
charlie 140V (Lunar Lake) — GPU + NPU 4. Realistic concurrent workload:
- 8B Llama on GPU running CB scheduler with batch=16 (real multi-tenant chat)
- 1B Llama on NPU running a single classifier-style request

64-tok output. Compare:
1. solo-gpu-cb: only GPU CB runs (NPU pipe loaded but idle)
2. concurrent: both run in threads simultaneously

## Results

| Mode | GPU 8B CB batch=16 agg | GPU per-req | NPU 1B |
|------|----------------------:|------------:|-------:|
| solo-gpu-cb | 208.56 |  13.03 | — |
| **concurrent** | **203.10 (-3%)** | 12.69 | **38.44** |

Effective throughput when concurrent:
- GPU serving 16 chat requests at 13 tok/s each
- NPU serving 1 classifier request at 38 tok/s
- Total effective serving: 203 + 38 = **241 tok/s** vs 208 solo
- **+16% effective throughput** for adding the NPU classifier with -3% cost.

## Findings — VALIDATES Discovery #4 in REAL multi-tenant

1. **NPU concurrency is essentially free** for GPU CB workload. Only -3%
   aggregate hit on GPU for adding a fully-running NPU 1B classifier.
2. **NPU 1B at 38 tok/s** under contention is plenty fast for classifier
   workloads, intent detection, summary, RAG retrieval, etc.
3. **Real-world tahoma deployment** can serve a multi-tenant 8B chat API
   on the GPU AND a always-available 1B classifier on the NPU on a
   single Lunar Lake AI PC.

## Killer demo for tahoma

A single Intel Lunar Lake laptop (charlie's class hardware) can:
- Serve 16 concurrent chat sessions on Llama 3.1 8B at 13 tok/s each
- Serve a Llama 3.2 1B classifier always-on at 38 tok/s
- Total: 17 concurrent users on one laptop

This is the kind of deployment density Intel hardware enables that
NVIDIA consumer cards don't (no NPU equivalent).

## Open follow-ups

- Test 32-concurrent CB + NPU classifier (does NPU still get fair share?)
- Test GPU at PEAK extractive RAG (PL +94%, ~194 tok/s) + NPU classifier
  concurrently. The 194 tok/s is single-instance though, so CB doesn't
  apply directly.
