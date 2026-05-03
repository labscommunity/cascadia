# e7 — factual workload baseline (single-node + distributed, single trial each)

**Hypothesis:** Creative writing is FastDraft-hostile (5% accept). A factual / technical prompt should give higher acceptance and let spec-decode amortize. Establish factual baseline numbers before sweeping K (e8) and committing engine surgery.

**Setup:**
- Prompt: a 35-token factual technical question about RISC vs CISC processors (`experiments/e7-factual-baseline/logs/prompt.txt`). Output continues to ~256 tokens.
- max_tokens=256
- Single-node: alpha + ov-genai + FastDraft K=5
- Distributed: alpha+charlie/TB4 + ov-dist-spec K=3 + FastDraft

## Result (single trial each)

| topology | engine | accept | tok/s |
|---|---|---:|---:|
| single-node | ov-genai + FastDraft K=5 | (not logged by this engine) | **23.30** |
| distributed alpha+charlie/TB4 | ov-dist-spec K=3 + FastDraft | 0.205 | **15.30** |

## Conclusion

- **Factual** is much better for spec-decode than creative: accept jumps from 0.054 → 0.205 (4×). Throughput gains in distributed: 9.88 → 15.30 (+55%).
- But **distributed (15.30) is still 66% of single-node (23.30)**. Even with workload-favorable spec, distributed is not yet at parity.
- Distributed K=3 may not be optimal for factual either — e8 sweeps K∈{1,2,4,5} on this workload to find the local maximum.

The bar (e0 single-node creative × 1.20 = 27.6 tok/s) is workload-specific. For this factual workload the corresponding single-node bar is 23.30 × 1.20 = **27.96 tok/s**. Distributed needs ~+82% over current factual baseline to clear it.

## Implications

- Workload-aware bar tracking will be useful — different workloads have different single-node ceilings and different distributed gaps.
- The campaign roadmap should include factual + creative measurements throughout (e8 = factual K-sweep; e9+ = workload-mix).
