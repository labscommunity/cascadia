# Baselines

These numbers are the comparison reference for every experiment in this autolab session. They get re-measured at the start of each session; if the cluster state has drifted, the new numbers replace these and any "Δ vs baseline" claims downstream get re-anchored.

Last full re-measure: (pending — campaign 0).

## Hardware

| Node | CPU | iGPU | dGPU | Memory |
|---|---|---|---|---|
| alpha | Panther Lake | Xe2 (Arc 140V class) | **Arc B390 Battlemage** | 32 GB |
| charlie | Lunar Lake | Arc 140V (Xe2) | — | 32 GB |
| beta | Lunar Lake | Arc 140V (Xe2) | — | 32 GB |

Network: alpha ↔ charlie via Thunderbolt 4 at 10.10.10.0/24 (15.4 tok/s plain dist measured baseline).

## Software stack (last verified)

| Package | Version |
|---|---|
| OpenVINO | 2026.2.0 |
| optimum-intel | 1.27 |
| transformers | 4.57.6 |
| torch | 2.11.0+xpu (charlie/alpha) |
| nncf | 3.1.0 |

## Reference numbers (carried over from main; will be re-confirmed)

### Llama 3.1 8B Instruct INT4

| Hardware | Engine | Config | tok/s |
|---|---|---|---|
| alpha B390 GPU | ov-optimum | greedy | 16.7 |
| charlie 140V GPU | ov-optimum | greedy | 17.0 |
| **alpha B390 GPU** | **ov-spec** | K=4, draft=Llama-3.2-1B-Instruct INT4 | **35.0** |
| alpha+charlie / TB | ov-runtime (v3 shards) | 2-stage GPU | 12.1 |
| alpha+charlie / TB on AC | ov-runtime (v3 shards) | 2-stage GPU | 15.4 |
| alpha+charlie / TB | ov-dist-spec (v3 shards) | K=4, K=4 | 15.77 |
| alpha+charlie / TB | ov-dist-spec (v5 shards) | K=4 | 17.36 |

### Llama 3.2 1B Instruct fp16

| Hardware | Engine | tok/s |
|---|---|---|
| alpha CPU | pytorch | ~10.7 (load 18s, decode 220 ms/token) |
| alpha+charlie TB CPU | pytorch-tp tp_size=2 | ~6.4 (decode 320 ms/token) |
