# LEADERBOARD — autolab/k26-perf

Best result per topology. Updated whenever a verified win lands.

## 1-box (miner, Xeon Gold 6252, 133 GB RAM)

| Rank | tok/s | Config | Date | Campaign |
|-----:|------:|--------|------|----------|
| 1 | ~0.11 | main @ 208104e, K2.6 single-stage, top-8, disk-bound | 2026-05-16 | (baseline from PR #7) |

## 2-box (matias-02 + matias-03, Lunar Lake 258V × 2, Tailscale DERP)

| Rank | tok/s | Config | Date | Campaign |
|-----:|------:|--------|------|----------|
| 1 | ~0.05 | main @ 208104e, 30/30 split, fp32 KV, K=1, F32 hidden wire | 2026-05-17 | (baseline from PR #9) |

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
