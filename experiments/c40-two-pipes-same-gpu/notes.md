# c40: Two LLMPipeline instances on same GPU vs cross-device (NPU+GPU)

## Setup
charlie 140V (Lunar Lake): GPU + NPU 4. Compare:
1. **Same-GPU concurrent**: 8B INT4 + 1B INT4, both on GPU.
2. **Cross-device concurrent (c30)**: 8B INT4 on GPU + 1B INT4 on NPU.

64-tok output each, 1 request per pipe, simultaneous via `threading.Thread`.

## Results

### Same-GPU concurrent (c40)

| Mode | 8B GPU tok/s | 1B GPU tok/s |
|------|------------:|-------------:|
| solo (B pipe loaded but idle) | 81.5 | — |
| concurrent | 70.1 (-13%) | 45.7 |

Reference: 1B alone single-pipe on charlie GPU = 211.39 tok/s (c26).
Loading the 8B pipe and running 1B drops it to 45.7 — **78% drop**.

### Cross-device NPU+GPU (c30)

| Mode | 8B GPU tok/s | 1B NPU tok/s |
|------|------------:|-------------:|
| solo (other pipe loaded but idle) | 86.2 | 41.8 |
| concurrent | 80.5 | 37.6 |

### Aggregate comparison

For "serve one 8B request + one 1B request":

| Strategy | Total wall-clock |
|----------|----------------:|
| Sequential, same-GPU (8B then 1B) | 64/91 + 64/211 = 1.00 s |
| Concurrent same-GPU | max(64/70, 64/46) = 1.40 s |
| Concurrent NPU+GPU | max(64/80, 64/38) = 1.68 s |
| **Sequential same-GPU is the actual winner** for short outputs |

Hmm — at short output, sequential wins because each request finishes fast.

For longer-output workloads (256 tok each):
| Strategy | Total wall-clock |
|----------|----------------:|
| Sequential same-GPU | 256/91 + 256/211 = 4.0 s |
| Concurrent same-GPU | max(256/70, 256/46) = 5.6 s |
| Concurrent NPU+GPU | max(256/80, 256/38) = 6.7 s |
| **Sequential still wins** because 1B runs 2-3x faster on GPU. |

## Findings

1. **Loading 2 pipes on same GPU costs the active pipe ~10%** even when
   only one runs. (8B drops 91 → 81.)
2. **Concurrent same-GPU is a 13% / 78% loss** vs solo for the larger /
   smaller pipe respectively.
3. **For 1B workloads, just run sequentially on GPU.** Sequential GPU
   processing (8B then 1B) is faster overall than splitting devices,
   because 1B runs 2-3x faster on charlie GPU than NPU.
4. **The NPU pays off in different scenarios** — e.g., when GPU is
   committed to a long-running 8B job and you need to serve a 1B
   classifier *concurrently* without preempting. Or when you want to
   keep GPU idle for power savings on a laptop.

## Revision to Discovery #4

NPU concurrent serving is **NOT** a throughput win for transient short
workloads. It is a win for:
- Long-running GPU workloads where you can't afford to preempt.
- Power-constrained scenarios where NPU's lower wattage matters.
- Multi-tenant where one tenant has SLA on always-available second-model
  inference (e.g., classifier always on NPU).

For pure "best aggregate throughput", sequential same-GPU is optimal
for charlie's mix of 8B + 1B at short outputs.
