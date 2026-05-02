# c50: Phi-3 mini + PL at peak config (4K input + 256 output extractive)

## Setup
charlie 140V GPU, Phi-3-mini-128k INT4. 4K input + 256 output extractive
(passage + "Summarize..."). Best of 3 trials.

## Results

| Mode | tok/s |
|------|------:|
| plain | 182.48 |
| PL n=3 | **190.51** |
| Δ | **+4.4%** |

## Comparison to Llama 8B at same config

| Model | Plain | PL | Δ |
|-------|------:|---:|--:|
| Llama 3.1 8B INT4 | 100.17 | 194.36 | **+94%** |
| Phi-3 mini INT4 (~3.8B) | 182.48 | 190.51 | +4% |

## Findings

1. **Phi-3 + PL win at 4K is only +4%** vs Llama 8B's +94%.
2. **Phi-3 is already 1.8× faster than Llama 8B at plain 4K input** (182
   vs 100 tok/s) due to its smaller size.
3. **PL win scales DRAMATICALLY with model size**:
   - Phi-3 mini (~3.8B): +4-32% (depending on input)
   - Llama 3.1 8B (~7B effective): +50-97% (depending on input)
4. **Mechanism**: PL saves target-model forward passes. The savings are
   proportional to per-step compute cost. Smaller models have lower
   per-step cost so the savings don't amortize.

## Updated guidance for tahoma

PL is a no-brainer for **larger models on extractive RAG**. For smaller
models, the win shrinks dramatically and may not be worth the slight
overhead at very small models.

Default policy:
- Llama 3.1 8B / Llama 3.2 3B / Mistral 7B-class: enable PL by default
  for extractive workloads.
- Phi-3 mini and smaller: enable PL only if input is short (<1K) where
  the win is more meaningful (c46 showed +32% at 1K + 128 out).
- Llama 3.2 1B and below: don't bother with PL.

## Final cross-model PL summary

| Model | Best PL win (extractive) | Best abs tok/s |
|-------|--------------------------|---------------:|
| Llama 3.1 8B INT4 | +97% (4K input) | 194 |
| Phi-3 mini INT4 | +32% (1K input + 128 out) | ~70 |
| Llama 3.2 1B INT4 | (untested — likely small) | — |

PL effectiveness is model-size dependent. For tahoma's 8B-class
default models, PL is the killer optimization for RAG.
