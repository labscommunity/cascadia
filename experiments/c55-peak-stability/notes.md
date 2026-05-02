# c55: PEAK stability check — 388 tok/s confirmed reliable

## Setup
charlie 140V GPU, Llama 3.1 8B INT4. PEAK config: 4K input + 512 output
extractive PL. 3 sequential best-of-3 runs to validate stability.

## Results

| Run | Best tok/s | Variance |
|-----|-----------:|---------:|
| 1 | 389.58 | — |
| 2 | 388.28 | -0.3% |
| 3 | 388.81 | -0.2% |

## Findings

- **PEAK is stable across runs** within 0.3% variance.
- All "first-run cold" within each best-of-3 hits 137-140 tok/s
  (consistent — model JIT compile + cold cache).
- Best-of-3 reliably reaches 388-389 tok/s on charlie iGPU.

## Conclusion

The 388 tok/s peak number is reproducible. This is the validated
absolute single-instance throughput high for the autolab session:
**~388 tok/s on a Lunar Lake iGPU for Llama 3.1 8B INT4 extractive RAG
at 4K input + 512 output**.
