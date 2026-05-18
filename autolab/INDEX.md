# Campaign INDEX — autolab/k26-perf

One row per campaign in launch order. See `campaigns/NNN_*.yaml` for
definitions, `experiments/NNN_*/` for raw data, `JOURNAL.md` for
narrative.

| NNN | Date | Moonshot | Hypothesis (one-liner) | Result | tok/s | Quality |
|----:|------|---------|------------------------|--------|------:|--------|
| 000 | 2026-05-17 | (baseline) | Establish 2-box matias baseline on main @ 208104e | baseline | 0.0553 | 3/3 |
| 003 | 2026-05-17 | q1-instrumentation | Per-stage timing; expert dispatch >60% predicted | **verified** (82%!) | 0.0550 | 3/3 |
| 004 | 2026-05-17 | A3 top-K | K=8→K=6 on 2-box matias | parked (infra: Tailscale broken) | — | — |
| 005 | 2026-05-17 | A3 top-K (miner) | K=8→K=6 on miner single-stage | **WIN +40%** | **0.1116** | 3/3 |
| 006 | 2026-05-17 | A3 Pareto sweep | K=4, K=2 sweep on miner | **WIN +109% @ K=4**; K=2 quality cliff | **0.1667** | 3/3 (K=4) |
| 007 | 2026-05-17 | A2 routing-threshold | sigmoid-weight expert pruning | neutral (A3 K=4 dominates Pareto) | 0.1043 (thr=0.2) | 3/3 |
| 008 | 2026-05-17 | A3 full Pareto | K=3 + K=5 to complete the K-sweep | **WIN +208% @ K=3** (new leader) | **0.2455** | 3/3 (K=3) |
| 009 | 2026-05-17 | A3 robustness | 10-prompt eval at K=3, K=4, K=8 | **REVISION**: K=3 fails 6/10; K=4 is real leader (+146%, 9/10) | **0.2100** | 9/10 (K=4) |
| 010 | 2026-05-17 | F4 rayon heads | parallel SDPA (rayon over 64 heads) | neutral on miner (-2.7%, I/O-bound) | 0.2044 (K=4+F4) | 9/10 |
| 011 | 2026-05-17 | A3 K=4 long-context | max_tokens=64 sustained-throughput | **CONFIRMS**: 0.3253 tok/s, peak 0.45, 9/10 | **0.3253** | 9/10 |
| 012 | 2026-05-17 | A3 K=4 code prompts | 5 code/programming prompts at K=4 | 4/5 (consistent ~80-90% across prompt classes) | 0.2298 | 4/5 |
| 013 | 2026-05-17 | K=4 vs K=8 apples-to-apples mt=64 | Direct head-to-head at long context | **K=4 +210% AND higher quality (9 vs 8/10)** | 0.3253 (K=4) | 9/10 (K=4) vs 8/10 (K=8) |
| 014 | 2026-05-17 | spinout PR #29 | productionize K=4 win on main | **SHIPPED** as PR #29 (perf/a3-topk-override) | — | — |
| 015 | 2026-05-17 | A3+A2 compose | K=4 + threshold=0.3 (adaptive per-token K) | Pareto-incomparable: +33% tok/s but -1 quality (Guido) | 0.2792 | 8/10 |
| 016 | 2026-05-17 | A3+A2 thr=0.1 | K=4 + lower threshold | **WIN +11%** at same 9/10 quality (Guido prompt preserved) | **0.2336** | 9/10 |
| 019 | 2026-05-18 | K=6 at temp=0.7 | Fill K×temp curve | **WIN**: K=6=8/10 matches K=8, +75% tps | 0.1489 | 8/10 |
| 017 | 2026-05-18 | compose longctx | K=4 + thr=0.1 at mt=64 | NEUTRAL: filter is short-context-only (-3% at long ctx) | 0.3150 | 9/10 |
| 018 | 2026-05-18 | K × temperature | K=4 vs K=8 at temp=0.7 | **CAVEAT**: K=4 fragile at temp=0.7 (5/10); K=8 robust (8/10) | 0.2400 (K=4) | 5/10 (K=4) vs 8/10 (K=8) |
| 020 | 2026-05-18 | K=5 at temp=0.7 | Find exact temp-robust K cliff | borderline 6/10 (closer to K=4 fragility) | 0.1851 | 6/10 |
| 021 | 2026-05-18 | K=6 long-context | mt=64 sustained-throughput | **PERFECT 10/10 quality + +51% vs K=8** (Pareto-dominant) | **0.1587** | **10/10** |
| 022 | 2026-05-18 | K=6+thr=0.1 compose | composed flag at K=6 | 10/10 quality, slight tps edge | 0.1482 | 10/10 |
| 023 | 2026-05-18 | K=6 on code prompts | quality on programming questions | 4/5 (matches K=4; same format failure on x=5+3) | 0.1197 | 4/5 |

Format note: `result` is one of `win` / `neutral` / `negative` / `running`.
`tok/s` is steady-state on the 2-box matias pipeline unless noted.
`quality` is the 3-prompt eval pass count (3/3 expected).
