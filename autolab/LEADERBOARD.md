# LEADERBOARD — autolab/k26-perf

Best result per topology. Updated whenever a verified win lands.

## 1-box (miner, Xeon Gold 6252, 133 GB RAM)

| Rank | tok/s | Config | Date | Campaign |
|-----:|------:|--------|------|----------|
| 1 | **0.1116** | **A3: K2.6 single-stage, --top-k-override 6** (3/3 quality) | 2026-05-17 | [005_a3_topk_miner](experiments/005_a3_topk_miner/) |
| 2 | 0.0797 | K2.6 single-stage, top-8 baseline (3/3 quality) | 2026-05-17 | 005 K=8 reference run |
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
