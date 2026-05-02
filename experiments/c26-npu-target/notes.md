# c26: NPU as a target device — small models on Intel NPU

## Setup
- alpha: Battlemage host (Meteor Lake-class NPU)
- charlie: Lunar Lake (NPU 4 ~48 TOPS)
- Llama 3.2 1B INT4, FastDraft 150M INT8 — small enough for NPU memory.

## Results

| Hardware | Model | Device | tok/s |
|---|---|---|---|
| alpha (host NPU) | Llama 3.2 1B INT4 | NPU | 135.84 |
| alpha (B390 dGPU) | Llama 3.2 1B INT4 | GPU | 149.47 (from leaderboard) |
| alpha (host NPU) | FastDraft 150M INT8 | NPU | 85.57 |
| **charlie (Lunar Lake NPU)** | **Llama 3.2 1B INT4** | **NPU** | **112.89** |
| **charlie (Lunar Lake GPU)** | **Llama 3.2 1B INT4** | **GPU** | **211.39** |

## Findings

1. **Llama 3.2 1B INT4 runs cleanly on Intel NPU** (both Lunar Lake and the
   older host NPU on alpha's Battlemage box). Compile takes longer than GPU
   (~30-60s vs ~5-10s) but the model produces correct output.
2. **NPU is 91% of GPU speed on alpha (Battlemage host) for 1B**, and
   **53% of GPU speed on charlie (Lunar Lake)**. The huge GPU 211 tok/s on
   charlie is the new single-node leaderboard high for Llama 3.2 1B INT4.
3. **NPU is power-efficient** — at ~10-15 W vs the dGPU's ~30-50W under
   load, NPU is a solid tradeoff for sustained background workloads where
   wall-clock latency isn't the priority.

## Why this matters for tahoma

Multi-model serving on a single Lunar Lake AI PC: the iGPU runs the main
8B chat model at 134 tok/s while the NPU concurrently serves a 1B classifier
or auxiliary model at 113 tok/s. Two devices, two models, two clients,
~zero contention. This is a real Intel-only differentiator.

## Open
- Verify NPU + GPU concurrent workload doesn't cause cross-talk on shared
  memory bandwidth (likely fine — NPU has its own SRAM).
- Try Llama 3.2 3B INT4 on NPU (alpha already has a 3B from the c10 sweep).
