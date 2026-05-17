# DISCOVERIES — autolab/k26-perf

Verified novel findings promoted from JOURNAL. Inclusion criteria:

1. Reproducible across 3+ independent runs on real K2.6 pipeline
2. Quality eval passes (substring + coherent on Paris/Pacific/four)
3. Prior-art search shows the finding is novel for K2.6-class
   sparse-MoE on Intel CPU pipeline-parallel inference (cite refs)
4. Magnitude class ≥ S (≥5% delta), OR L+ negative result that
   closes a major open question

Each entry includes a prior-art search recording what was checked and
what's actually novel about the finding.

Entry template:
```
## DNN — <title> (YYYY-MM-DD)

**Claim:** ...
**Magnitude:** S / M / L / XL — measured delta
**Reproducibility:** N independent runs, σ = ...
**Prior art search:**
  - Searched: <queries>
  - Found: <papers / blogs / repos> with summaries
  - Novel because: ...
**Productionizable as:** PR #NNN against main (or "research-only")
**Campaign:** `campaigns/NNN_*.yaml`
**Linked moonshot:** MN in MOONSHOTS.md
```

---

(empty — first entries land here once moonshots execute)
