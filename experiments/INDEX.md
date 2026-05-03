# Campaign index

Chronological. Status legend: ✓ WIN (≥20%), ⚠ NEUTRAL (<10%), ✗ LOSS, ◌ ERROR/INCOMPLETE, 📊 BASELINE.

| Iter | Campaign | ID | Hypothesis | HW | Result | Δ vs baseline |
|------|----------|----|------------|----|--------|---------------|
| 1 | e0-baseline-single-node | e0 | re-measure single-node ov-genai + FastDraft K=5 baseline | alpha | 📊 23.01 tok/s (median of 5, 256-tok creative) | bar = ×1.20 = 27.6 tok/s |
| 2 | e1-baseline-distributed | e1 | re-measure distributed ov-dist-spec K=3 + FastDraft baseline | alpha+charlie | 📊 ✗ 9.88 tok/s (median of 5, accept=5.4%) | -57% vs e0; -64% vs bar |
| 3 | e2-k-sweep | e2 | sweep K∈{1,2,4,5,6} on creative workload | alpha+charlie | ⚠ K=1 wins at 11.78 tok/s (+19% over K=3) | -49% vs e0; -57% vs bar |
| 4 | e3-no-spec-distributed | e3 | pure PP without spec decode (ov-runtime) on creative | alpha+charlie | ⚠ 12.15 tok/s (slightly better than K=1 spec) | -47% vs e0; -56% vs bar |
| 5 | e4-layer-rebalance | e4 | 22/10 alpha-heavy split (bottleneck=charlie hypothesis) | alpha+charlie | ✗ 12.23 tok/s (in noise vs 16/16) — bottleneck is per-step OV overhead | -47% vs e0; -56% vs bar |
