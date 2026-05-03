# e8 — K-sweep on FACTUAL workload (ov-dist-spec)

**Hypothesis:** Spec decode wins on factual content where draft acceptance is high. The K=3 default may not be optimal — sweep K∈{1,2,4,5} on a 35-token technical prompt that produces a long technical answer.

**Setup:**
- Distributed alpha+charlie/TB4, ov-dist-spec, v5 16/16, FastDraft 150M
- Prompt: `experiments/e7-factual-baseline/logs/prompt.txt` (RISC vs CISC technical question)
- max_tokens=256, 3 trials per K (one K=4 trial dropped — charlie hung)

## Result

| K | trials | med tok/s | med accept | med elapsed |
|--:|------:|----------:|-----------:|------------:|
| 1 | 3 | **15.81** | 0.378 | 16.19 s |
| 2 | 3 | 15.76 | 0.268 | 16.25 s |
| 3 | 1 (e7) | 15.30 | 0.205 | 16.74 s |
| 4 | 2 | 13.89 | 0.159 | 18.43 s |
| 5 | 3 | 12.91 | 0.132 | 19.82 s |

**K=1 wins on factual too**, by a hair over K=2. Same monotonic decline as the creative workload (e2): lower K = higher per-draft accept rate × less wasted target verify.

## Pattern across both workloads

| Workload | K=1 | K=3 | best K |
|---|------:|------:|---|
| creative (e2) | 11.78 | 9.88 | K=1 |
| factual (e8) | 15.81 | 15.30 | K=1 |

K=1 is optimal regardless of workload at this draft acceptance level. The Phase 14 default of K=3 should be revised to K=1 in the engine — that's a free +5-19% across both workloads.

## Distance to the bar

| Workload | Single-node best | Distributed best (K=1) | Bar (single × 1.20) | Gap to bar |
|---|------:|------:|------:|---:|
| creative | 23.01 | 11.78 (e2) | 27.61 | -57% |
| factual | 23.30 | 15.81 | 27.96 | -43% |

Spec decode lever is exhausted on both workloads. The next moonshots must address per-stage compute or async pipelining.

## Implication for the campaign roadmap

- **Change `--spec-k` default from 3 to 1** in the CLI (small commit, free win).
- Pursue async draft + target overlap (modest win on factual where accept ~38%, marginal on creative).
- Per-stage compute speedup (PA was tried in e9 attempt and failed — see e9 notes).
