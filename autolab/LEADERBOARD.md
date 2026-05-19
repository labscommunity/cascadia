# LEADERBOARD — autolab/k26-perf

Updated at iter 100 close (2026-05-19).

## Verified perf wins (ranked by measured tok/s impact)

| Rank | Iter | Win | tok/s impact | Quality | Branch |
|-----:|-----:|-----|-------------|---------|--------|
| 1 | 044 | Compound spec-decode | **+19.7% e2e** (0.1899 vs 0.1587 baseline), 1.27× per-prompt, 1.41× paired | 9/10 substring | perf/spec-decode-compound-044 |
| 2 | 046 | Row-blocked AVX-512 oproj | +40% over iter 042 at seq=4-16 (1.41× at seq=4 = 1.84× scalar) | bit-identical | perf/oproj-amx-or-avx512-blocked-046 |
| 3 | 042 | AVX-512 multi-token tile | 1.4-4.75× per K2.6 projection at seq=4-16 (peak 4.75× shared_down) | bit-identical | perf/int4-multi-token-avx-vnni-042 |
| 4 | 033 | C1 expert prefetch (Linux) | +26.8% A/B under 3-worker contention | n/a (no substring eval in A/B) | perf/c1-expert-prefetch-029 |
| 5 | 032 | A8 KV bf16 | ~2.1× attention kernel (687ms vs 1456ms), KV mem halved | 3/3 substring | perf/a8-kv-bf16-029 |
| 6 | 030 | Matias 2-box revival | 0.0770 tok/s K=8 (vs iter 000 baseline 0.0553) | 9/10 substring | infra/matias-2box-revival-029 |

## K-tuning wins (early iters, pre-pivot; productionized as PR #29)

| Iter | Setting | tok/s | Quality | Notes |
|-----:|---------|------:|--------|-------|
| 011 | K=4 mt=64 chat | 0.3253 | 9/10 | Throughput-max single-stage |
| 021 | K=6 mt=64 chat | 0.1587 | 10/10 | Universal default (PR #29) |
| 024 | K=6 mt=128 | 0.1713 | 3/3 | Sustained long-context |
| 025 | K=4 mt=128 | 0.3209 | 3/3 | Long-context throughput-max |

## Current per-substrate leader (chat-quality-preserving)

- **miner single-stage:** iter 044 compound = 0.202 tok/s per-prompt
  mean (K=6, mt=64, --prompt-lookup 3 --spec-k 4)
- **miner single-stage throughput-max:** iter 025 K=4 mt=128 =
  0.3209 tok/s (3/3 quality, deterministic chat)
- **matias 2-box:** iter 030 = 0.0770 tok/s (K=8, mt=32, SSH-tunnel
  chain). Tailscale stayed dead; revivable via tunnel.

## Verified architectural negatives (decisive skip)

| Iter | What | Why |
|-----:|------|-----|
| 049 | bf16 inter-layer hidden | Compiler auto-vectorizes inline; 0.00007% theoretical saving |
| 053 | Fused RMSNorm+QKV | RMSNorm is 0.4% of unfused; fits L1d already |
| 055 | int4 router | Already shipped in PR #7 (verify-before-implement) |
| 062 | int4 KV (scalar) | Dequant cost > bandwidth savings; 5-9% SLOWER |
| 064 | Native bf16 SDPA | Upconvert ≤4%; AVX-512-BF16 not in fleet |
| 067 | Fast sampling | 0.0024% of decode at K2.6 rates |
| 079 | SSE streaming | Already shipped (verify-before-implement) |
| 080 | Lazy expert load | Already shard-lazy (verify-before-implement) |
| 082 | Selective recomputation | KV is 0.5-1% of miner RAM |
| 085 | Sparse softmax | Router is sigmoid not softmax; 0.0002% |
| 089 | SSE aggregator | 880ns/frame = 0.0000098% of decode |
| 093 | zstd expert storage | 1.15× ratio (not 1.4-1.8×); +12s cold load |

## Composition findings

- iter 070: **Full 7-feature cache-attack stack = -32% on miner.**
  Prefetch contention + RAM pressure on demand-path. Lean subset
  (drop iter 057, lower prefetch-n) recommended.
- iter 063: Lookahead decoding (iter 061) = -1.6% on factoid prompts
  (per-prompt 7/10 wins, but aggregate dragged by anomaly).
  Workload-dependent.

## Class counts (state.json)

| Class | Count |
|-------|------:|
| win (verified perf) | 6 architectural + early K-tuning |
| neutral / foundation | ~35 |
| negative (decisive) | 12 |
| already-shipped discovery | 3 |
| agent failure | 3 |
| **total** | 100 |
