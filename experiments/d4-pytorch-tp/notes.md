# d4: pytorch-tp tensor parallelism over TB4 — 10× SLOWER than single-node

## Setup
alpha + charlie via TB4 (8.75 Gbps, 0.142ms RTT).
pytorch-tp engine: 2-rank TP (each holds half the attention heads + half MLP).
TCP all-reduce after each o_proj and down_proj (32 all-reduces per token for 16-layer Llama 1B).

Model: Llama 3.2 1B INT4 (unsloth, HF cached snapshot).
64 max_tokens, factual prompt.

## Result

| Metric | Value |
|--------|------:|
| Total wall_clock | 38.18 s |
| Setup (load + slice + warmup) | ~25 s |
| Decode time | ~12 s for 64 tokens |
| **Actual tok/s** | **~5.3** |

### Comparison

| Engine | Llama 3.2 1B INT4 | tok/s |
|--------|-------------------|------:|
| Single-node LLMPipeline plain (c58) | charlie GPU | **55.6** |
| Single-node LLMPipeline plain | alpha GPU | ~50 |
| **pytorch-tp 2-rank over TB4** | **distributed** | **~5.3** |

**TP over TB4 is ~10× SLOWER than single-node.**

## Why TP is so slow over TB4

For Llama 1B with 16 layers, TP requires:
- 16 layers × 2 all-reduces per layer = **32 all-reduces per token**

Each all-reduce is a TCP round-trip (send + recv). With TB4 latency 0.142 ms RTT:
- 32 × 0.142 ms = **4.5 ms of network overhead per token**

Plus:
- TCP serialization overhead per call (~100 µs)
- Python GIL + numpy roundtrip (~50 µs)
- Total network overhead per token: ~10 ms

For comparison, the per-token compute on a fast iGPU at 50 tok/s is ~20 ms. So network is half the wall time before any compute.

Plus the tp_engine uses HF transformers (slow per-step compute) instead of OV. So we're paying both:
- HF transformers overhead vs OV runtime
- TCP all-reduce overhead per layer

## Conclusion

**TP over Thunderbolt 4 is not viable for tahoma's distributed inference.**
TP fundamentally requires very high inter-node bandwidth + low latency — the
NVLink class (~600 GB/s, ~1 µs). TB4 (~5 GB/s, ~150 µs) is several orders
of magnitude short.

For Intel-only distributed deployments, **pipeline parallelism (ov-runtime, ov-dist-spec)** is
the right choice. PP needs only one transfer per token, not 32+.

## Relation to Phase 2 distributed wins

Reinforces the d3 finding: ov-dist-spec K=4 + FastDraft (PP-based) at
38.49 tok/s is the right distributed config. TP is a distraction over TB4.
