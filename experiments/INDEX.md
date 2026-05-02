# Experiment index

Chronological list of every experiment run on this branch. Status legend: ✓ WIN, ⚠ NEUTRAL (within noise), ✗ LOSS, ◌ ERROR/INCOMPLETE, 📊 BASELINE.

| Iter | Campaign | ID | Hypothesis | HW | Result | Δ vs baseline |
|------|----------|----|------------|----|--------|---------------|
| 1    | c0-baselines | c0-1   | re-measure ov-optimum on alpha B390 | alpha | 📊 8.85 tok/s | -47% vs main 16.7 |
| 2    | c0-baselines | c0-1b  | confirm c0-1 reproducibility | alpha | 📊 8.89 tok/s | identical to c0-1 |
| 3    | c0-baselines | c0-2   | re-measure ov-optimum on charlie 140V | charlie | 📊 10.33 tok/s | -39% vs main 17.0 |
| 4    | c0-baselines | c0-3   | re-measure ov-spec K=4 on alpha | alpha | 📊 13.83 tok/s, accept 0.50 | -60% vs main 35.0; accept fell from 0.91 |
| 5    | c0-baselines | c0-4   | re-measure ov-runtime on TB (v5 shards) | dist | ◌ engine v3-only, shape mismatch | — |
| 6    | c0-baselines | c0-5   | re-measure ov-dist-spec on TB | dist | 📊 17.59 tok/s, accept 0.62 | matches main 17.36 |
