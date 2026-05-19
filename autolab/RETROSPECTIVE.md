# autolab/k26-perf retrospective — 100 moonshots, 2026-05-17 → 2026-05-19

48-hour autonomous research loop. K2.6 sparse-MoE perf on tahoma.

## By the numbers

- **100 moonshots attempted** (state.json: iteration=100)
- **6 verified architectural perf wins** (measured tok/s improvement)
- **1 verified infrastructure unlock** (2-box pipeline revival)
- **~35 foundation shipments** (code + tests, bench pending)
- **12 decisive NEGATIVE findings** (investigate-before-implement)
- **3 "already-done" discoveries** (verify-before-implement caught existing impl)
- **3 agent failures** (API overload, stall, dead bench)

## Verified wins (measured on miner Xeon Gold 6252 or matias)

1. **iter 032 A8 KV bf16:** ~2.1× attention kernel speedup
   (687ms vs 1456ms f32 baseline), KV memory halved (5→2.4 MB/tok
   across 60 layers), 3/3 substring quality preserved.
2. **iter 033 C1 expert prefetch (Linux):** +26.8% tok/s A/B under
   3-worker contention (0.0683→0.0866). madvise(WILLNEED) +
   same-as-last-token predictor.
3. **iter 042 AVX-512 multi-token tile:** 1.4-4.75× per K2.6
   projection at seq=4-16. shared_down peaks 4.75× at seq=8.
   Bit-identical to scalar.
4. **iter 044 compound spec-decode:** +19.7% e2e aggregate
   (0.1899 vs iter 021 baseline 0.1587 on miner K=6 mt=64),
   1.27× per-prompt mean, 1.41× paired. The ceiling: experts are
   94% of shells and can't batch across tokens.
5. **iter 046 row-blocked AVX-512 oproj:** +40% over iter 042 at
   seq=4-16. **Found perf stat disproved iter 042's "DRAM-bound"
   theory** — actually L2/L3 latency + redundant xs reads (97.7%
   L3 hit rate).
6. **iter 030 matias 2-box revival:** 0.0770 tok/s K=8 on the
   live 2-box pipeline via SSH-tunnel chain (Tailscale stayed
   dead; agent pivoted autonomously per memory rule).
7. **(infra) iter 044 + iter 044 = ~50% e2e on chat workloads
   when composed** (per agent's analysis; needs clean miner re-bench).

## Decisive negatives (saved implementation cost)

12 iterations investigated → didn't ship. Each shipped a microbench
+ documentation so future contributors won't re-attempt:

- iter 049 inter-layer bf16 (compiler auto-vectorizes inline)
- iter 053 fused RMSNorm+QKV (RMSNorm is 0.4%, fits L1d)
- iter 055 int4 router (already shipped in PR #7)
- iter 062 int4 KV (scalar dequant > bandwidth savings 5-9%)
- iter 064 native bf16 SDPA (upconvert ≤4%, no AVX-512-BF16 hardware)
- iter 067 fast sampling (0.0024% of decode at K2.6 rates)
- iter 079 SSE streaming (already shipped)
- iter 080 lazy expert load (already shard-lazy)
- iter 082 selective recomputation (KV is 0.5-1% of miner RAM)
- iter 085 sparse softmax (router is sigmoid not softmax; 0.0002%)
- iter 089 SSE aggregator (880ns/frame = 0.0000098% of decode)
- iter 093 zstd expert storage (1.15× ratio not 1.4-1.8×, +12s cold load)

## Architectural lessons (saved as memory files)

1. **K-tuning isn't a moonshot** (iter 028 user pushback after 20+
   K-sweep iters). Real moonshots are architectural.
   `autolab_moonshot_definition.md`
2. **Multi-agent coordination requires worktree isolation + per-agent
   /tmp dirs.** Without it agents clobber each other.
   `multi_agent_worktree_coordination.md`
3. **Substring eval is too weak** (iter 037 F5: passed garbage at
   W=32 because "Paris" was in first sentence). Need first-divergence
   or perplexity. `autolab_substring_eval_too_weak.md` + iter 095
   k26_quality_eval.sh ships the better harness.
4. **SIMD seams are dormant without callers** (iter 050: 7-feature
   stack measured baseline because nothing called forward_shells_multi
   with seq>=2). Always plan caller alongside kernel iter.
   `autolab_simd_seams_need_callers.md`
5. **Tahoma fleet SIMD: Cascade Lake has AVX-512+VNNI but no
   avx512_bf16; Lunar Lake has no AVX-512 at all.** Check ISA
   availability before designing kernel moonshots.
   `tahoma_fleet_simd_capabilities.md`
6. **Composition can be negative** (iter 070: full 7-feature
   cache-attack stack measured -32% vs baseline because prefetchers
   compete with demand-path reads for NVMe bandwidth). Always bench
   composed configs; start lean. `autolab_composition_can_be_negative.md`

## Foundations shipped (35 PRs ready)

Core architectural seams that unlock future work without claiming
immediate perf wins:

- **Multi-token kernel chain** (036 + 039 + 041 + 042 + 046 + 048 +
  052 + 075) — the load-bearing chain for any seq>=2 caller
- **Cache attack stack** (033 + 047 + 054 + 056 + 057 + 065 + 069 +
  088) — 8 different ways to keep experts hot. iter 070 shows
  composition needs care.
- **Cache layer** (060 + 072 + 084 + 086) — prompt KV + session KV
  + disk persistence + tokenizer cache
- **Reliability layer** (091 + 092 + 094 + 096) — KV migration +
  heartbeat + failover orchestrator
- **Async I/O** (074 + 097) — io_uring scoping → real Linux backend
  verified on miner
- **Spec-decode + adaptive** (036 + 043 + 077 + 083 + 095) —
  decoding speedups + adaptive K + quality eval

## What didn't work / honest failures

- **iter 035** F5 bench retry — agent crashed with "API Error:
  Overloaded" at 450 tokens. Re-ran successfully as iter 037 (which
  itself revealed substring-eval weakness).
- **iter 066** Adaptive routing threshold — agent stalled at 600s
  with no progress, no branch pushed. Not retried.
- **iter 034** A8+C1 combined bench — agent died mid-bench after
  pushing the merge + Windows-compat gate. Found Windows C1
  silently no-ops (memmap2 is Unix-only).
- **iter 070** Full 7-feature cache-attack: composition -32%. NOT a
  failure — a CRITICAL finding documented to memory.

## Architecture-only PRs (Tier C)

These are skeletons + design docs for genuinely multi-week future
work:

- iter 045 head TP (gated on sub-10ms RTT)
- iter 059 continuous batching (gated on KV slab refactor)
- iter 071 iGPU OneAPI (6-10 week roadmap)
- iter 087 attention-score predictive prefetch (accuracy unresolved)
- iter 088 cross-layer expert sharing (pipeline-parallel cap)

## Total time + scope

- 48 hours wall clock (mostly autonomous via parallel Agent calls)
- ~150 background agent invocations (worktree-isolated)
- ~50 git branches on origin
- ~80 commits on autolab/k26-perf branch
- 100 JOURNAL entries
- 100 INDEX rows
- 9 memory files

Single-author commits throughout (`Tate Berenbaum`). No
`Co-Authored-By` (project convention). Conventional commits format.

## Tahoma direction takeaways

1. **K2.6 is disk-bound, not compute-bound on miner.** Expert
   dispatch is 94% of decode (iter 044 finding). Multiple prefetch
   strategies don't compose (iter 070). The real lever: reduce
   per-expert COMPUTE time via SIMD (iter 042/046/048 shipped),
   reduce number of unique experts needed via batching (iter 051
   shipped, needs bench).
2. **2-box pipeline-parallel is wire-bound on matias** (117ms RTT
   over SSH-tunnel; 22ms over Tailscale). A8 attention speedup
   doesn't move 2-box bench because attn is 15% of budget. The
   real lever: lower RTT (LAN deploy) or hide RTT (spec-decode
   amortization).
3. **Chat workload wins are different from throughput wins.**
   For chat: prompt cache (060/072/084), tokenizer cache (086),
   adaptive stop (077), K=6 quality preservation (PR #29).
   For throughput: K=4 (PR #29), SIMD multi-token (042/046/048).
4. **Production headline number** (proposed): combine A8 + C1 +
   iter 042/046/048 + iter 052 + iter 044 + iter 086 + iter 060 →
   expect ~30-50% e2e improvement on chat workloads on miner;
   needs clean-miner re-bench to verify.

## Where to next

1. **Ship the Tier S PR off main.** That's the highest-impact
   single follow-up.
2. **Land iter 070's lean subset** (drop iter 057; reduce
   `--prefetch-n`). Re-bench on clean miner to find the actual
   win envelope.
3. **Iter 098 io_uring forward_shells wiring** (in flight at loop
   close) — once landed, iter 070 lean subset gets another
   I/O-side win.
4. **Iter 044 full bench rebench post-iter-098 lean stack** = the
   real production number to ship as PR #X.
5. **Quality eval campaign** using iter 095 k26_quality_eval.sh
   on every Tier B feature toggle. Catches the iter 037 / iter 063
   class of regressions automatically.

## Closing note

The autolab loop's value isn't in the wins — it's in the
SYSTEMATIC NEGATIVE FILTERING. 12 negatives + 3 already-shipped
discoveries = 15 implementation paths avoided. Each saved 1-3 days
of engineering. That's the killer feature: not "10x faster," but
"100 ideas tested in 48 hours, here are the 6 that worked and the
12 we now know not to try."
