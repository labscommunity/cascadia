# Experiment index

Chronological list of every experiment run on this branch. Status legend: ✓ WIN, ⚠ NEUTRAL (within noise), ✗ LOSS, ◌ ERROR/INCOMPLETE, 📊 BASELINE.

| Iter | Campaign | ID | Hypothesis | HW | Result | Δ vs baseline |
|------|----------|----|------------|----|--------|---------------|
| 1    | c0-baselines | c0-1   | re-measure ov-optimum on alpha B390 | alpha | 📊 8.85 tok/s | -47% vs main 16.7 |
| 2    | c0-baselines | c0-1b  | confirm c0-1 reproducibility | alpha | 📊 8.89 tok/s | identical to c0-1 |
| 3    | c0-baselines | c0-2   | re-measure ov-optimum on charlie 140V | charlie | 📊 10.33 tok/s | -39% vs main 17.0 |
| 4    | c0-baselines | c0-3   | re-measure ov-spec K=4 on alpha | alpha | 📊 13.83 tok/s, accept 0.50 | -60% vs main; accept fell from 0.91 |
| 5    | c0-baselines | c0-4   | re-measure ov-runtime on TB (v5 shards) | dist | ◌ engine v3-only, shape mismatch | — |
| 6    | c0-baselines | c0-5   | re-measure ov-dist-spec on TB | dist | 📊 17.59 tok/s, accept 0.62 | matches main 17.36 |
| 7    | c1-llmpipeline | c1-1 | LLMPipeline (no plugin config) on alpha | alpha | **✓ 96.41 tok/s** | **+10.8×** vs c0-1b |
| 8    | c1-llmpipeline | c1-2 | LLMPipeline on charlie (after pkg fix) | charlie | **✓ 91.14 tok/s** | **+8.83×** vs c0-2 |
| 9    | c1-llmpipeline | c1-3 | LLMPipeline + CACHE_DIR/KV_u8/DynQuant=32 | alpha | ⚠ 92.74 tok/s | within noise of c1-1; defaults engaged |
| 10   | c2-llmpipe-spec | c2-1 | LLMPipeline + draft_model K=5 | alpha | ◌ draft IR has beam_idx/attn_mask params LLMPipeline rejects | — |
| 11   | c3-ov-genai-engine | c3-1 | wire LLMPipeline as tahoma engine `ov-genai` | alpha | **✓ 87.1 tok/s** | -10% vs raw c1-1; **+9.8×** vs ov-optimum |
