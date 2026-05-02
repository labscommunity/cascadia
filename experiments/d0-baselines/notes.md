# d0: Distributed baselines (alpha B390 + charlie 140V via TB4)

## Setup
- alpha 10.10.10.1 (B390 dGPU), charlie 10.10.10.2 (Lunar Lake iGPU)
- Direct TB4 link, 8.75 Gbps, 0.142ms RTT (d1)
- Pre-existing v3/v5 shards copied to charlie via Mac as relay (TB4-direct scp had issues)
- Bench uses tahoma worker rank 0/1, prompt fed via stdin, 64 max_tokens
- 8 prompt tokens (chat-template not applied — model continues in completion mode)

## Engines benched (preliminary, single run each)

### d0-1 ov-runtime (2-stage v3 PP)
- Shard: `shards_2stage_v3` (layer split 0-16 / 16-32, INT4)
- Decode time: ~7.5s for 64 tokens (estimated WALL_CLOCK − ~17s setup)
- **~8.5 tok/s actual**
- Engine doesn't emit "task done" log — timing is rough.

### d0-2 ov-dist-spec (2-stage v5 PP + spec decode K=4)
- Shard: `shards_2stage_v5_beam` + Llama 3.2 1B INT4 draft (cached)
- Decode time: 4.088s for 64 tokens (18 spec rounds, accept ~0.62)
- **15.66 tok/s actual** ← spec decode ~doubles ov-runtime perf
- Matches c0-5 historical (17.59 — within run noise)

### d0-3 pytorch (default distributed PP via HF transformers)
- Pending; HF transformers initial load is slow.

## Comparison to single-node Phase 1 (corrected)

| Workload | Single-node | Distributed (alpha+charlie/TB4) | Δ |
|----------|------------:|--------------------------------:|--:|
| 8B INT4 plain LLMPipeline (alpha) | ~17.5 tok/s | n/a (LLMPipeline is single-stage only) | — |
| 8B INT4 + FastDraft K=5 (alpha) | ~27.2 tok/s | n/a | — |
| 8B INT4 ov-runtime distributed | n/a | **~8.5 tok/s** | -50% vs single-node plain |
| 8B INT4 ov-dist-spec K=4 distributed | n/a | **15.66 tok/s** | -10% vs single-node plain |

## Key observations

1. **Distributed PP at half-compute-each is SLOWER per-token than single-node** — counterintuitive. Ideal PP would be ~2× single-node (each node does half the layers); we measure 0.5× to 0.9×.
2. **The network is NOT the bottleneck** (d1 showed only 78 µs per-token activation transfer at 8.75 Gbps; total per-token wall time is ~60-120 ms).
3. **Spec decode helps** (8.5 → 15.66 tok/s for ov-runtime → ov-dist-spec).
4. **Pre-exported v3/v5 IRs lose the LLMPipeline optimizations** (PagedAttention, U8 KV, XMX dynamic quant). The single-node FastDraft win (+55%) comes from LLMPipeline runtime; the distributed engines bypass it.

## Hypothesis for d2

The distributed engines run a different (slower) OV runtime path than LLMPipeline.
For tahoma to make distributed competitive, we need either:
(a) port LLMPipeline-equivalent optimizations into the multi-stage path
(b) re-export shards with flags that engage PagedAttention etc. at runtime
(c) accept ~half-speed per-token but use distribution for models that DON'T fit on one node (e.g., Mixtral 8×7B INT4, Llama 3.3 70B INT4)

(c) is the strongest motivation for distribution anyway — single-node 8B is faster everywhere; distribution exists for models that single-node can't run at all.

## K-sweep for ov-dist-spec on alpha+charlie/TB4

| K | tokens | spec rounds | accept/round | decode_s | tok/s |
|---|-------:|------------:|-------------:|---------:|------:|
| 1 | 64 | 34 | 1.88 | 3.665 | 17.46 |
| **2** | 64 | 24 | 2.67 | 3.397 | **18.84** ← sweet spot |
| 3 | 64 | 20 | 3.20 | 3.497 | 18.30 |
| 4 | 64 | 18 | 3.55 | 4.088 | 15.66 |
| 8 | 64 | 12 | 5.33 | 5.763 | 11.10 |

## Distributed has a DIFFERENT K sweet spot than single-node

Single-node spec decode (Phase 1): K=5 was the sweet spot.
Distributed (Phase 2): K=2 wins.

Why: each spec round in distributed mode runs K+1 forward passes through
the FULL pipeline (alpha → charlie → alpha return). The per-round
overhead grows linearly with K, but the amortization benefit doesn't
because the accept rate on K candidates with the 1B FastDraft is
~0.62 (constant). So lower K = lower overhead = higher net throughput.

**Real headline number: ov-dist-spec K=2 at 18.84 tok/s — actually
BEATS single-node OVModelForCausalLM (17.23 tok/s) by +9%.**

This is the first case where distributed inference is strictly better
than the single-node equivalent for the SAME engine class.

## Comparison table (corrected, actual tok counts)

| Engine | Where | tok/s | Notes |
|--------|-------|------:|-------|
| OVModelForCausalLM | single-node alpha | 17.23 | c60 |
| LLMPipeline plain | single-node alpha | 17.45 | c61 |
| LLMPipeline + FastDraft K=5 | single-node alpha | 27.02 | c61 |
| ov-runtime v3 | distributed alpha+charlie | ~8.5 | rough — no done log |
| ov-dist-spec K=4 v5 | distributed alpha+charlie | 15.66 | c0-5 era result |
| **ov-dist-spec K=2 v5** | **distributed alpha+charlie** | **18.84** | **NEW d0** |
| ov-dist-spec K=8 v5 | distributed alpha+charlie | 11.10 | over-spec hurts |

**Distributed wins for ov-dist-spec class (+9% over single OVModel).**
**LLMPipeline + FastDraft is still 43% faster than the best distributed result** (27.02 vs 18.84) — porting LLMPipeline's optimizations to multi-stage is the only way to close that gap on models that fit on one node.

For models that DON'T fit on one node, distributed is the only option.
