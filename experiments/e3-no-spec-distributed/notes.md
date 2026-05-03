# e3 — distributed pure pipeline-parallel without spec decode

**Hypothesis:** When FastDraft acceptance is low (5-15% on creative content), the spec-decode overhead exceeds its amortization gain. A pure no-spec PP baseline (`ov-runtime`, no draft model) might match or beat ov-dist-spec K=1 on this workload.

**Setup:**
- alpha (rank 0, B390) ↔ charlie (rank 1, LL 140V) via TB4
- Engine: `ov-runtime`, v3 shards 16/16 (`shards_2stage_v3`)
- No draft model
- Same 256-tok creative prompt as e0/e1/e2
- 5 trials, charlie restarted between each

## Result

| Trial | tokens | engine elapsed (s) | tok/s |
|------:|-------:|-------------------:|------:|
| 1 | 256 | 21.16 | 12.10 |
| 2 | 256 | 20.96 | 12.21 |
| 3 | 256 | 20.84 | 12.28 |
| 4 | 256 | 23.47 | 10.91 |
| 5 | 256 | 21.07 | 12.15 |

**Median: 12.15 tok/s.** Spread: 10.91 – 12.28 (12.7%). Trial 4 is a low outlier (probably charlie GPU thermal or driver hiccup); the four other trials are within 1.4%.

## Conclusion

- **Pure PP without spec is marginally better than spec-decode K=1** (12.15 vs 11.78 tok/s).
- This confirms e2's read of the K-sweep: on this creative workload, FastDraft 150M's per-round overhead (one extra forward pass + reject-handling) exceeds its acceptance amortization (~15% accepted prefix).
- The new distributed leaderboard high is **12.15 tok/s** = 53% of e0 monolithic.
- Distance to bar (27.6 tok/s): **need 2.27× from current**, no spec.

## Implications for the campaign roadmap

The spec-decode lever has been fully explored on the creative workload — it's a wash or net negative. To beat the bar, the win has to come from one of:

1. **Per-stage compute speedup** — paged-attention re-export (engine surgery to feed PA inputs)
2. **Asymmetric layer split** — give charlie fewer layers since its iGPU is the bottleneck (e4, in progress)
3. **Async stage_0 / stage_1 overlap** — speculative pre-compute of next round's stage_0 while charlie verifies — not a free win for low-spec workloads, but combined with adaptive K could pay off
4. **Within-host TP** — alpha GPU + alpha NPU on one stage, charlie GPU + charlie NPU on another. Doubles per-stage hardware.
5. **A draft model that actually matches creative content** — not FastDraft

Filing follow-ups in this order:
- e4: layer rebalance 22/10 (alpha-heavy)
- e5: detailed target.feed timing breakdown (debug-level logs) — confirm where the per-round 80ms goes
- e6+: engine surgery (paged-attention or async overlap, depending on what e5 reveals)
