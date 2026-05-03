# Campaign index

Chronological. Status legend: ✓ WIN (≥20%), ⚠ NEUTRAL (<10%), ✗ LOSS, ◌ ERROR/INCOMPLETE, 📊 BASELINE.

| Iter | Campaign | ID | Hypothesis | HW | Result | Δ vs baseline |
|------|----------|----|------------|----|--------|---------------|
| 1 | e0-baseline-single-node | e0 | re-measure single-node ov-genai + FastDraft K=5 baseline | alpha | 📊 23.01 tok/s (median of 5, 256-tok creative) | bar = ×1.20 = 27.6 tok/s |
| 2 | e1-baseline-distributed | e1 | re-measure distributed ov-dist-spec K=3 + FastDraft baseline | alpha+charlie | 📊 ✗ 9.88 tok/s (median of 5, accept=5.4%) | -57% vs e0; -64% vs bar |
