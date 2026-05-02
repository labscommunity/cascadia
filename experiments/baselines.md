# Baselines

These numbers are the comparison reference for every experiment in this autolab session. Re-measured at the start of campaign 0; whenever the cluster state drifts, the new numbers replace these and any "Δ vs baseline" claims downstream get re-anchored.

**Last full re-measure:** 2026-05-02 (campaign 0 — see `experiments/c0-baselines/notes.md`).

## Hardware

| Node | CPU | iGPU | dGPU | Memory |
|---|---|---|---|---|
| alpha | Panther Lake | Xe2 (Arc 140V class) | **Arc B390 Battlemage** | 32 GB |
| charlie | Lunar Lake | Arc 140V (Xe2) | — | 32 GB |
| beta | Lunar Lake | Arc 140V (Xe2) | — | 32 GB |

Network: alpha ↔ charlie via Thunderbolt 4 at 10.10.10.0/24.

## Software stack (last verified)

| Package | Version |
|---|---|
| OpenVINO | 2026.2.0 |
| optimum-intel | 1.27 |
| transformers | 4.57.6 |
| nncf | 3.1.0 |

## Reference numbers — measured 2026-05-02

### Llama 3.1 8B Instruct INT4

| Hardware | Engine | Config | Decode tok/s | Notes |
|---|---|---|---|---|
| alpha B390 GPU | ov-optimum | greedy 64 | **8.85** | Cold load 26.8s; warm 23.6s |
| charlie 140V GPU | ov-optimum | greedy 64 | **10.33** | Cold load 29.3s |
| alpha B390 GPU | ov-spec K=4 | + Llama-3.2-1B-Instruct INT4 draft | **13.83** | 21 steps, accept 0.50 |
| alpha+charlie TB | ov-runtime | v5 shards | (broken — see c0-4) | engine doesn't pass attention_mask |
| alpha+charlie TB | ov-dist-spec K=4 | v5 shards | **17.59** | 18 steps, accept 0.62 |

### Llama 3.2 1B Instruct fp16 (PyTorch path)

| Hardware | Engine | tok/s | Notes |
|---|---|---|---|
| alpha CPU | pytorch | ~10.7 | tied embeddings; fixed in main |
| alpha+charlie TB CPU | pytorch-tp tp_size=2 | ~6.4 | network-dominated for small model |

## Reference numbers from main (carried over for context — NOT used as baseline)

These are what `main` claims; they don't match what we measured today. If a future experiment beats THESE numbers, we have a real win; if it just beats the lower today-baseline, we're recovering ground rather than gaining new perf.

| Hardware | Engine | Config | tok/s (main) |
|---|---|---|---|
| alpha B390 GPU | ov-optimum | greedy | 16.7 |
| charlie 140V GPU | ov-optimum | greedy | 17.0 |
| alpha B390 GPU | ov-spec K=4 | | 35.0 |
| alpha+charlie / TB | ov-runtime (v3) | 2-stage | 12.1 |
| alpha+charlie / TB on AC | ov-runtime (v3) | 2-stage | 15.4 |
| alpha+charlie / TB | ov-dist-spec (v5) K=4 | | 17.36 |
