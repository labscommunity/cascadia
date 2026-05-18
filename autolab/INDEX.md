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
| 024 | 2026-05-18 | K=6 mt=128 sustained | very-long context | 3/3, 0.1713 tok/s (sustained K=6 production number) | **0.1713** | 3/3 |
| 025 | 2026-05-18 | K=4 mt=128 sustained | throughput-max long-context | 3/3 narrow, 0.3209 tok/s (+87% vs K=6 mt=128) | **0.3209** | 3/3 |
| 026 | 2026-05-18 | system prompt | "Answer concisely" on code prompts | neutral: shifts which prompt fails, same 4/5 rate | 0.1624 | 4/5 |
| 027 | 2026-05-18 | K=6 + sys 10p | completion-style sys prompt | slightly negative (-1 quality, prefill cost) | 0.1162 | 9/10 |
| 028 | 2026-05-18 | K=6 temp ladder | K=6 quality across temp=0.3, 0.5 | **WIN** (insight): K=6 gentle curve (10/10/9/8) vs K=4 cliff (9→5); K=6 dominates temp>0 | 0.1457 (t=0.3) / 0.1539 (t=0.5) | 10/10 (t=0.3), 9/10 (t=0.5) |
| 029 | 2026-05-18 | F5 sliding-window | windowed attention via --attention-window W flag | **IMPL DONE** (perf/f5-sparse-attention-029 @ 769c8b0), 8 unit tests pass, bench deferred for sibling-agent miner contention | (bench pending) | (bench pending) |
| 030 | 2026-05-18 | matias 2-box | revive 2-box pipeline-parallel via SSH-tunnel chain | **PIPELINE ALIVE** (infra/matias-2box-revival-029 @ 61778ef): SSH-tunnel pivot (Tailscale stayed dead), 117ms RTT, WMI-detached workers, 0.0770 tok/s @ K=8 10p mt=32 | 0.0770 | 9/10 (substring) |
| 031 | 2026-05-18 | A1 int2 experts | int4→int2 expert weight quantization | **KERNEL WORKS** (perf/a1-int2-experts-029 @ ae617a6): 1.80× smaller, 2.89× faster in clean run, cosine 0.61, 4/4 quality preserved at layer-30 swap; pipeline bench contaminated by 3-worker contention | 1.722ms (int2) vs 4.977ms (int4) per-expert | 4/4 (substring) |
| 032 | 2026-05-18 | A8 KV bf16 | KV cache fp32→bf16 + inline bf16→f32 in SDPA | **VERIFIED WIN** (perf/a8-kv-bf16-029 @ ebd8ac4): ~2.1× attn kernel speedup (687ms vs 1456ms f32), KV mem halved (2.4 vs 5 MB/tok), 3/3 quality | (kernel: 687ms attn, contention-depressed e2e) | 3/3 |
| 033 | 2026-05-18 | C1 expert prefetch | madvise(WILLNEED) on next-token experts via background thread | **VERIFIED WIN** (perf/c1-expert-prefetch-029 @ eb57a9e): +26.8% tok/s A/B under contention (0.0683 → 0.0866), drops=0, same-as-last predictor | 0.0866 (A/B vs 0.0683) | n/a (substring not tested in A/B) |
| 034 | 2026-05-18 | A8+C1 combined bench | merge + bench on live 2-box matias | **MERGE LANDED** (perf/a8-c1-combined-bench-034 @ 8713929), Windows C1 gap discovered (memmap2 Unix-only), bench incomplete (agent died); needs Windows port OR Linux re-bench | (incomplete) | (incomplete) |
| 037 | 2026-05-18 | F5 bench retry | long-context windowed-attention bench (miner) | (in flight) | (pending) | (pending) |
| 038 | 2026-05-18 | C1 Windows port | PrefetchVirtualMemory via windows-sys | **CODE READY** (perf/c1-windows-port-038 @ 77650ea): Mac + Windows MSVC + Windows GNU all build clean, 7/7 unit tests pass; bench gated on matias source-sync (~300 MB scp + rebuild) | (no bench) | (no bench) |
| 035 | 2026-05-18 | F5 bench | long-context windowed-attention bench | **FAILED** (API Error: Overloaded, 450 tokens before crash); will retry | — | — |
| 036 | 2026-05-18 | spec-decode skeleton | n-gram Prompt-Lookup draft + accept/rewind in sparse-moe | **FOUNDATION** (perf/spec-decode-skeleton-034 @ acd21bd): 38 unit tests pass, simulation = bit-identical to sequential greedy; throughput win waits for ForwardBatch(K) wire frame | (no bench) | bit-identical to greedy (proven by sim test) |

Format note: `result` is one of `win` / `neutral` / `negative` / `running`.
`tok/s` is steady-state on the 2-box matias pipeline unless noted.
`quality` is the 3-prompt eval pass count (3/3 expected).
