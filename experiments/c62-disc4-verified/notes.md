# c62: Discovery #4 (NPU concurrent serving) verified with actual tokens

## Setup
charlie 140V (Lunar Lake), Llama 3.1 8B INT4 on GPU + Llama 3.2 1B INT4 on NPU.
GPU runs CB batch=8 (8 concurrent chat-style requests).
NPU runs 1 single classifier-style request.
All counts are actual tokens via tokenizer (not max_tokens cap).

## Results

| Mode | GPU 8B (CB b=8) | NPU 1B |
|------|----------------:|-------:|
| solo-gpu-cb (NPU pipe loaded but idle) | 137.64 agg / 17.20 per-req | — |
| **concurrent** | 127.47 agg / 15.93 per-req (-7%) | 30.31 |

### Effective serving comparison

For the workload "8 chat sessions + 1 classifier on Lunar Lake":
- Solo (only GPU runs CB): 137.64 tok/s aggregate (8 chat sessions)
- Concurrent (GPU CB + NPU 1B): 127.47 + 30.31 = **157.78 tok/s** total effective
- **Effective uplift: +14.6%** for adding the NPU classifier with -7% GPU cost.

## Findings — VERIFIED

1. **Discovery #4 is REAL** — adding an always-on classifier on the NPU
   while serving 8 concurrent chat sessions on the GPU costs the GPU only
   -7% throughput.
2. **Effective concurrent throughput uplift: +14.6%** (close to the
   originally-reported +16%, within 2% noise).
3. **Actual per-request tok/s on GPU CB**: 17.20 solo, 15.93 concurrent.
   Per-user latency under concurrent serving stays in the 16-17 tok/s
   range — usable for chat.

## Cross-NPU note

NPU 1B running concurrently produces 30.31 actual tok/s. Solo NPU 1B
(c26) was 112.89 tok/s — the loading penalty when GPU pipe is also
loaded is real (113 → 30 = 73% drop). But the 30 tok/s is enough for
classifier workloads.

## Final discovery status (post-correction, all verified)

| Discovery | Verified result |
|-----------|-----------------|
| #1 LLMPipeline vs OVModel | 1.00× (DEBUNKED — c60) |
| #2 FastDraft +24% | +55% (CONFIRMED+ — c58, c61) |
| #3 PL +59-65% | +40-50% extractive (CONFIRMED- — c57) |
| #4 NPU concurrent serving | +14.6% effective (CONFIRMED — c62) |

Real autolab outcome: 3 verified discoveries + 1 debunked + comprehensive
methodology lessons across 62 campaigns.
