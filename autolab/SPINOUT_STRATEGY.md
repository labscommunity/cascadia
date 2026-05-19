# Spinout PR Strategy — autolab/k26-perf (post-100 moonshots)

This document distills the autolab/k26-perf 100-moonshot loop into a
shippable PR roadmap off main. The autolab branch itself is the
long-lived research log (`PR #11`, not for merging); production wins
spin out as small focused PRs off main.

Branches in this loop: ~50+. Verified wins: 6 measured perf + 1
infra revival. Foundation shipments: ~35. Decisive negatives: 12.

## Tier S — Verified perf wins, ship soonest

Bundle as one cohesive "K2.6 Linux production" PR off main, default
flags set to safe values.

| Iter | Branch | Headline measurement |
|------|--------|----------------------|
| 030 | infra/matias-2box-revival-029 | SSH-tunnel chain revival of 2-box; 0.0770 tok/s; WMI-detached spawn |
| 032 | perf/a8-kv-bf16-029 | ~2.1× attention kernel + KV mem halved |
| 033 | perf/c1-expert-prefetch-029 | +26.8% A/B under contention (Linux) |
| 038 | perf/c1-windows-port-038 | Win32 PrefetchVirtualMemory (cross-platform) |
| 042 | perf/int4-multi-token-avx-vnni-042 | 1.4-4.75× per projection at seq=4-16 |
| 044 | perf/spec-decode-compound-044 | +19.7% e2e on miner K=6 mt=64 |
| 046 | perf/oproj-amx-or-avx512-blocked-046 | +40% over iter 042 at seq>=4 (oproj/shared_down) |
| 048 | perf/wire-simd-multi-tiles-048 | ProjShape dispatcher; engine seam |
| 052 | perf/layer0-multi-simd-052 | layer-0 SIMD lift (8 projections) |
| 075 | perf/extend-simd-dispatch-075 | LargeShape variant + load-bearing bf16-KV merge fix |

**Order:** 075 is the load-bearing merge — must come first. Then
the rest can land sequentially via cherry-pick.

## Tier A — Foundations + correctness fixes

Ship as small focused PRs in dependency order.

| Iter | Branch | Notes |
|------|--------|-------|
| 036 | perf/spec-decode-skeleton-034 | n-gram Prompt-Lookup draft + 38 tests |
| 039 | perf/forward-batch-spec-decode-039 | ForwardBatch(K) wire frame + 58 tests; bug discovered in 043 |
| 043 | fix/spec-decode-reconcile-off-by-one-043 | Off-by-one fix (single-stage spec-decode now works past round 1) |
| 040 | perf/chunked-prefill-040 | Chunked prefill seam (no perf change today; foundation) |
| 041 | perf/int4-multi-token-041 | Multi-token int4 seam (bit-identical) |
| 060 | perf/static-prompt-cache-060 | LRU prefix KV cache (in-memory) |
| 072 | perf/session-kv-cache-072 | Per-session KV cache for chat |
| 084 | perf/persistent-kv-cache-084 | Disk persistence for iter 060 |
| 086 | perf/tokenizer-cache-086 | LRU prompt→tokens cache |
| 077 | perf/adaptive-stop-077 | EOS + stop seqs + repetition detection |
| 090 | perf/warmup-profiling-090 | tracing spans for startup |

## Tier B — Code shipped, bench pending

Default OFF; include flag for opt-in. The iter 070 finding
(composition can be -32%) means cache-attack-stack features should
NEVER all be on at maxed settings. Lean subset recommended:
033 + 054 + 056 + 069 (per iter 070 agent's analysis).

| Iter | Branch | Status |
|------|--------|--------|
| 047 | perf/c1-better-predictor-047 | top-N pre-softmax; instrumentation for future bench |
| 054 | perf/expert-pinning-054 | mlock hot-set; needs RLIMIT_MEMLOCK |
| 056 | perf/cache-aware-dispatch-056 | 3-phase split; bit-identical |
| 057 | perf/async-kernel-sched-057 | Speculative cross-layer prefetch; **iter 070 identified as most I/O-greedy — drop from lean subset** |
| 065 | perf/prefill-hint-schedule-065 | Prefill observations seed expert_hits |
| 069 | perf/hot-expert-buffer-069 | Contiguous packing of top-N hot experts |
| 051 | perf/expert-batching-051 | Expert dispatch batching (the keystone breaker) |
| 058 | perf/int4-embedding-058 | Embedding bf16 → int4 (660 MB vs 2.34 GB) |
| 045 | perf/head-tp-045 | Head TP plumbing (gated on sub-10ms RTT) |
| 076 | perf/cpu-affinity-076 | Thread pinning |
| 081 | perf/rank-balance-081 | --layer-range flag |
| 083 | perf/dynamic-spec-k-083 | AdaptiveK controller |
| 088 | perf/cross-layer-expert-share-088 | Cross-layer L3 sharing |
| 097 | perf/io-uring-milestone1-097 | Real Linux io_uring backend (10ms read on miner) |
| 098 | perf/io-uring-forward-shells-098 | Wire io_uring into forward_shells (in flight) |

## Tier C — Skeletons + design docs

Land separately as architecture-only PRs; productionization is
multi-week work.

| Iter | Branch | What |
|------|--------|------|
| 059 | perf/continuous-batching-059 | ContinuousBatcher skeleton |
| 078 | perf/continuous-batching-wiring-078 | Blocker 3 (per-request sampling state) addressed |
| 091 | perf/kv-migration-091 | FrameKind::KvMigration + Runner API |
| 092 | perf/heartbeat-recovery-092 | Wire frames + watchdog |
| 094 | perf/heartbeat-driver-094 | Cadence loop |
| 096 | perf/failover-orchestrator-096 | FSM composing 091+092+094 |
| 071 | perf/igpu-oneapi-scoping-071 | iGPU OneAPI scoping (6-10 week roadmap) |
| 074 | perf/io-uring-prefetch-074 | io_uring 282-line scoping doc |
| 087 | perf/attn-predict-prefetch-087 | Shadow router skeleton + bench |

## Tier D — Quality eval infra

| Iter | Branch | What |
|------|--------|------|
| 095 | autolab/quality-eval-095 | k26_quality_eval.sh A/B harness (addresses iter 037 substring weakness) |
| 003 | (on autolab) | Per-stage instrumentation (already on autolab branch) |

## Iters that are now no-ops (already in production)

These iters discovered the feature was already shipped — no code
change needed beyond regression tests:

- iter 055: K2.6 router IS already int4 (since PR #7)
- iter 079: SSE streaming IS already implemented
- iter 080: SafetensorsExpertSource IS already shard-lazy

Their regression tests (iter 055 router top-K stability, iter 079 SSE
format, iter 080 mmap_profile bin) can ship as a single "regression
coverage" PR.

## Decisive NEGATIVES (do NOT ship; docs explain why)

These are research docs only. Their value: prevent future attempts.

| Iter | Negative finding |
|------|------------------|
| 049 | bf16 inter-layer hidden states 3.4× slower (compiler auto-vectorizes inline) |
| 053 | Fused RMSNorm+QKV: RMSNorm is 0.4% of unfused; fits L1d already |
| 062 | int4 KV: scalar dequant cost > bandwidth savings (5-9% slower); needs SIMD first |
| 064 | Native bf16 SDPA: upconvert is ≤4%, AVX-512-BF16 not in fleet hardware |
| 067 | Fast sampling: 0.0024% of decode (vocab is f32 + already partial-sort) |
| 082 | Selective recomputation: KV is 0.5-1% of miner RAM; not bottleneck |
| 085 | Sparse softmax: router is sigmoid not softmax; sigmoid is 0.0002% of routed path |
| 089 | SSE aggregator: 880ns/frame = 0.0000098% of decode at K2.6 rates |
| 093 | zstd expert storage: 1.15× ratio (not 1.4-1.8×), break-even at 62 MB/s vs miner's 3 GB/s NVMe |

## Critical infrastructure ALSO ship

- The 9 memory files in `/Users/tatef/.claude/projects/-Users-tatef-Workspaces-tahoma/memory/`:
  - `autolab_loop_autonomy.md` (don't wait, kill+retry)
  - `tahoma_git_conventions.md` (no Co-Authored-By, conventional commits)
  - `user_workstyle.md`
  - `autolab_moonshot_definition.md` (K-tuning is not a moonshot)
  - `multi_agent_worktree_coordination.md` (worktree isolation, per-agent /tmp dirs)
  - `autolab_substring_eval_too_weak.md` (substring eval is insufficient)
  - `autolab_simd_seams_need_callers.md` (kernel speedups are dormant without callers)
  - `tahoma_fleet_simd_capabilities.md` (Cascade Lake has AVX-512+VNNI but no AVX-512-BF16; Lunar Lake has no AVX-512)
  - `autolab_composition_can_be_negative.md` (iter 070's -32% finding)

These should be referenced in CONTRIBUTING.md or similar.
