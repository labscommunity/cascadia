# JOURNAL — autolab/k26-perf

Append-only. Newest at top. One entry per moonshot iteration.

Entry template:
```
## NNN — <title> (YYYY-MM-DD HH:MMZ)

**Hypothesis:** ...
**Bucket / candidate:** ...
**Literature:** (one-paragraph synthesis with [[refs]])
**Campaign:** `campaigns/NNN_*.yaml`
**Design choice:** ...
**Result:** <win | neutral | negative>; tok/s = ...; quality = ...
**Learning:** ...
**Next:** (spawned sub-questions, follow-up moonshots, or "park")
```

---

## 001 — Baseline established (2026-05-17 ~13:36 PT)

**Hypothesis:** 2-box matias-02+03 K2.6 pipeline on main @ 208104e
delivers ~0.05 tok/s steady-state (3-prompt aggregate), 3/3 on the
Paris/Pacific/four quality eval. Matches the PR #9 / PR #10 numbers
from memory.

**Bucket / candidate:** baseline (not a moonshot — reference anchor)
**Literature:** none required for baseline. See [[LITERATURE]] for the
horizon: 0.05 tok/s is ~200x below comparable systems in the literature
(KTransformers on Xeon+A100 = 13.69, ik_llama on TR Pro+A6000 = 13.13,
mlx-lm on M3 Ultra = >20). The 30-300x gap is structural, not
hardware-bound, per the pipeline-parallel research agent's read.

**Campaign:** `campaigns/000_baseline_main.yaml`
**Design choice:** Clean restart of matias-02 (rank 0) + matias-03 (rank 1)
via the new `start_workers.sh` / `start_rank{0,1}.ps1` wrappers. Bench
script `k26_3prompt_eval.ps1` polls API readiness then runs 3 prompts at
max_tokens=8 temp=0.

**Result:** **baseline** (anchor for downstream moonshots)
- Paris    : 8 tok / 123.06 s = 0.0650 tok/s ✓ "Paris"
- Pacific  : 8 tok / 170.09 s = 0.0470 tok/s ✓ "Pacific"
- four     : 8 tok / 140.93 s = 0.0568 tok/s ✓ "four"
- **AGG**  : 24 tok / 434.09 s = **0.0553 tok/s**, **3/3 quality**

**Learning:**
- Variance is large (0.047-0.065 across prompts). Pacific (longest output
  context window after prompt + 8 tokens) was slowest; Paris shortest
  was fastest. Per-token latency creeps up with KV size in the
  shells (O(N) with our pre-allocated KV but the attention dot product
  is still per-token).
- The bench script's per-prompt `completion_tokens` is 0 due to a
  PS 5.1 / Invoke-RestMethod auto-parse quirk with snake_case JSON
  fields. tok/s in the rank-0 internal log is correct; bench's tok/s
  computation needs a max_tokens fallback. Filing as a bench-harness
  improvement, not a baseline-blocker.
- Cold-cache restart took ~5 min for the workers to come ready
  (warmer than the historical ~40 min in memory — likely OS page
  cache still warm from the morning's stale rank-0 process).

**Next (iteration 002):**
- Implement q1 (instrumentation): add per-stage timing (layer0 / shells
  attention / experts dispatch / wire / head) so we can attribute the
  17-20 s/tok across stages and rank moonshots by which stage they
  actually attack.
- After q1, the next real moonshot is Tier-S #1: per-token expert
  reduction (A2/A3). Lit says +10-50% with <1% quality cost on
  DeepSeek-V3 sigmoid router family (K2.6 is in that family).

## 000 — Scaffold (2026-05-17 ~12:45 PT)

Branch `autolab/k26-perf` cut from `origin/main @ 208104e`. Autolab
artifact tree created. PRIOR_ART synthesized from PRs #1/#4/#5/#7/#9/#10.
60 moonshot candidates enumerated in MOONSHOTS.md across 7 buckets
(quant, KV/attn, dispatch, wire, topo, sched, algo). 7 research
questions decomposed in research_plan.yaml. PR #11 opened as draft
(long-lived, will not merge). 3 parallel lit-research agents converged
on Tier-S moonshots (A2/A3 expert reduction, D1 BF16 wire, D4 async
overlap).
