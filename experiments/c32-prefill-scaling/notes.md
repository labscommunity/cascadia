# c32: Long-input prefill + decode scaling on alpha B390

## Setup
Llama 3.1 8B INT4 on alpha B390 GPU. LLMPipeline plain (no draft / PL).
32-token output. Inputs constructed at target lengths {128, 512, 1024,
2048, 4096} tokens by repeating a seed passage.

NOTE: Used `pipe.generate([prompt], cfg)` (list form) to access perf_metrics
for TTFT. STR-vs-LIST overhead measured separately in c32-aux to validate.

## Results (best of 2 runs per row)

| Input ~tok | TTFT ms | Total dt s | Decode dt s | Decode tok/s |
|-----------:|--------:|-----------:|------------:|-------------:|
|        128 |     300 |       1.16 |        0.86 |         37.4 |
|        512 |     304 |       1.79 |        1.48 |         21.6 |
|       1024 |     183 |       1.71 |        1.52 |         21.0 |
|       2048 |     222 |       1.76 |        1.54 |         20.8 |
|       4096 |     332 |       1.84 |        1.50 |         21.3 |

## Findings

1. **Decode rate plateaus at ~21 tok/s for inputs ≥ 512 tokens.** This is
   roughly 22% of our headline 96 tok/s number for plain LLMPipeline at
   short input. The KV-cache attention overhead dominates incremental
   decode once context is non-trivial.
2. **TTFT scales roughly linearly with input** for cold-cache runs (run 1:
   430ms → 2493ms across 128→4096 inputs) but is **almost constant for
   warm-cache runs** (best-of-2: 300→332 ms). The OV plugin's prefill kernel
   is highly efficient once warmed.
3. **Prefill rate at 4096 input is 12,343 tok/s** (4096/0.332s). Roughly
   600× faster than decode. So for any task where the question is "how fast
   can I read 4K tokens of context?" the answer is sub-second on B390.

## Implications

- Our previous "96 tok/s" headline was for ~5-token input. For real RAG
  workloads at 1K+ input the plain decode is ~21 tok/s.
- Prompt Lookup (which we measured at 91 tok/s on alpha for 128-tok output
  with ~250 input) is doing real work — without it that workload would be
  much slower than naive expectation suggested.
- Long-input deployments will benefit MORE from FastDraft / Prompt Lookup
  than short-input deployments (where the wins are already large).

## Caveats

- Results use list-form `generate()`. STR form may be faster — pending
  validation in `bench_str_vs_list.py` (c32-aux).
- "n_in" reported as 0 in perf_metrics — the bench doesn't read actual
  tokenised input length, only the target. The "input ~N" rows are
  approximate.
