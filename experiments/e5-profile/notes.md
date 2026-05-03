# e5 — instrument ov-runtime per-task timing (alpha_ms / wire_ms breakdown)

**Goal:** Localize where the per-token cost goes on distributed ov-runtime so subsequent campaigns target the actual bottleneck.

**Change landed:** `crates/tahoma-engine-openvino/src/runtime.rs` — added per-task accumulators `t_alpha_compute` (time inside `run_first`, i.e. stage_0 GPU compute) and `t_wire` (send hidden + wait for charlie's stage_1 + recv next_token). Engine's `task done` log line now emits `alpha_ms`, `wire_ms`, `other_ms` alongside `elapsed_s` and `tok_s`.

**Setup:** Same as e3 (ov-runtime distributed v3 16/16, no spec, 256-tok creative on alpha+charlie/TB4). One trial.

## Result

| metric | value |
|---|---:|
| tokens | 256 |
| elapsed | 22.88 s |
| **tok/s** | **11.19** |
| alpha_ms | 8,733 (38%) |
| wire_ms | 14,096 (62%) |
| other_ms | 52 (0.2%) |

Per-token breakdown:
- alpha (stage_0 compute on B390 GPU): **34 ms/token**
- wire (network + charlie stage_1 compute): **55 ms/token**
- other: 0.2 ms/token

## What this tells us

**Wire is 1.6× alpha** — charlie's iGPU is the bottleneck (network is sub-ms per d1 measurement; nearly all of `wire_ms` is charlie's stage_1 compute).

The implication for the moonshot lineup:
1. **Async overlap of alpha stage_0 with charlie stage_1** would cap per-token at max(34, 55) = 55 ms = 18.2 tok/s — a +63% improvement, just shy of e0's 23 tok/s. Combined with PA shrinking charlie stage_1 by ~30%, we'd land at ~26 tok/s — **at the bar**.
2. Layer rebalance alone can't help much: charlie's per-layer compute (~3.5 ms/layer) is too close to alpha's (~2.1 ms/layer) — moving 6 layers from charlie to alpha shifts ~13 ms in each direction, swapping the bottleneck without reducing it (confirmed in e4).
3. The 0.2% `other_ms` rules out tokenizer/decode/Rust-side overhead. The lever is purely per-stage GPU compute and the serialization between them.

## Next campaigns

- e7: implement async-overlap in dist_spec for the high-accept (factual) workload — confirms the overlap pattern works before generalizing
- e8: paged-attention re-export (`V5_MODE=paged_attention`) + engine support for PA inputs — targets the per-stage compute
- e9: combined PA + async overlap
