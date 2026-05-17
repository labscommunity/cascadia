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

## 000 — Scaffold (2026-05-17)

Branch `autolab/k26-perf` cut from `origin/main @ 208104e`. Autolab
artifact tree created. PRIOR_ART synthesized from PRs #1/#4/#5/#7/#9/#10.
60 moonshot candidates enumerated in MOONSHOTS.md across 7 buckets
(quant, KV/attn, dispatch, wire, topo, sched, algo). 7 research
questions decomposed in research_plan.yaml.

No measurement yet — this is the setup commit. Iteration counter
remains at 0; first moonshot is q1 (per-token time breakdown), which
turns into the baseline we'll re-measure for everything downstream.
