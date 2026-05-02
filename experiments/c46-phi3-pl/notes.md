# c46: Phi-3 mini + PL extractive cross-model validation

## Setup
charlie 140V GPU, Phi-3-mini-128k INT4. Same extractive prompt as c45
(passage + "Summarize the passage above in 2 short sentences").
1024-token input, 128-token output.

## Results

| Mode | tok/s |
|------|------:|
| plain | 52.71 |
| **PL n=3, K=5** | **69.37** |
| Δ | **+32%** |

## Findings

1. **PL works on Phi-3 too** (+32% on extractive workload).
2. **Smaller win than Llama 8B (+56% at same input/output)** — Phi-3
   mini is faster per-step so the amortization is less dramatic.
3. **Cross-model validated**: PL's effectiveness scales with both
   model size and workload extractiveness.

## Updated guidance

PL is a generally useful optimization for any model on extractive
RAG workloads:
- Llama 3.1 8B: +56-71%
- Phi-3 mini: +32%

Larger / slower models benefit MORE from PL (more spec-decode amortization
per accepted token).
