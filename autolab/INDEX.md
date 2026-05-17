# Campaign INDEX — autolab/k26-perf

One row per campaign in launch order. See `campaigns/NNN_*.yaml` for
definitions, `experiments/NNN_*/` for raw data, `JOURNAL.md` for
narrative.

| NNN | Date | Moonshot | Hypothesis (one-liner) | Result | tok/s | Quality |
|----:|------|---------|------------------------|--------|------:|--------|
| -   | 2026-05-17 | (scaffold) | — | — | 0.05 (baseline) | 3/3 |

Format note: `result` is one of `win` / `neutral` / `negative` / `running`.
`tok/s` is steady-state on the 2-box matias pipeline unless noted.
`quality` is the 3-prompt eval pass count (3/3 expected).
