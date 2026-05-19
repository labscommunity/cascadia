# LEADERBOARD — autolab/k26-perf

Best result per topology. Updated whenever a verified win lands.

## 1-box (miner, Xeon Gold 6252, 133 GB RAM)

| Rank | tok/s | Config | Date | Campaign |
|-----:|------:|--------|------|----------|
| 1 | **0.3253** | **A3: --top-k-override 4, max_tokens=64** (9/10 quality, peak prompt 0.45 tok/s, ~3-5× K=8 in production-realistic workloads) | 2026-05-17 | [011_a3_k4_longcontext](experiments/011_a3_k4_longcontext/) |
| 2 | 0.2100 | A3: --top-k-override 4, max_tokens=16 (9/10 quality, +146% vs K=8) | 2026-05-17 | [009](experiments/009_a3_robustness_10prompt/) |
| 2 (narrow) | 0.3050 | A3: --top-k-override 3 — **6/10 on 10-prompt** (was 3/3 on narrow eval, +258% raw); demoted from leader after robustness check | 2026-05-17 | 009 |
| 3 | 0.1667 | A3: --top-k-override 4, 3-prompt narrow (3/3) | 2026-05-17 | [006](experiments/006_a3_topk_sweep/) |
| 3 | 0.1547 | A3: --top-k-override 5 (3/3 quality, +94% vs K=8) | 2026-05-17 | 008 |
| 4 | 0.1116 | A3: --top-k-override 6 (3/3 quality, +40% vs K=8) | 2026-05-17 | [005](experiments/005_a3_topk_miner/) |
| 5 | 0.0797 | K2.6 top-8 baseline (3/3 quality) | 2026-05-17 | 005 K=8 reference |
| (excl) | 0.2716 | A3 K=2 — +241% but 2/3 quality (substring fail on "four") | 2026-05-17 | 006 K=2 cliff probe |
| - | ~0.11 | (historical) main @ 208104e single-stage from PR #7 | 2026-05-16 | (older reference) |

## 2-box (matias-02 + matias-03, Lunar Lake 258V × 2, Tailscale DERP)

| Rank | tok/s | Config | Date | Campaign |
|-----:|------:|--------|------|----------|
| 1 | **0.0553** | main @ 208104e, 30/30 split, fp32 KV, K=1, F32 hidden wire | 2026-05-17 | [000_baseline_main](campaigns/000_baseline_main.yaml) |

### Per-prompt detail (campaign 000)

| Prompt | wall (s) | tok | tok/s | quality |
|--------|---------:|----:|------:|---------|
| Paris       | 123.06 | 8 | **0.065** | ✓ paris   |
| Pacific     | 170.09 | 8 | **0.047** | ✓ pacific |
| four        | 140.93 | 8 | **0.057** | ✓ four    |
| **AGG**     | **434.09** | **24** | **0.055** | **3/3** |

## 3-box (matias-02 + 03 + extra cascadia box) — staging required

(no entries yet)

## Hardware ceilings (empirical, for context)

| Hardware | Read BW | K2.6 ceiling | Source |
|----------|--------:|-------------:|--------|
| Miner DDR4-2133 5-ch | 58 GB/s | ~3.5 tok/s | [[PRIOR_ART]] |
| Matias LPDDR5-6400 | ~50 GB/s* | ~3.0 tok/s* | *estimated, needs measurement |
| Sapphire Rapids HBM | ~800 GB/s | ~48 tok/s | spec |
| Gaudi 3 / Gaudi 2 | ~3.7 TB/s | reframes problem | spec |

(`*` = needs explicit measurement campaign — add as moonshot if not yet done)
