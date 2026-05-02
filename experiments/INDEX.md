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
| 10   | c2-llmpipe-spec | c2-1 | LLMPipeline + draft_model K=5 (auto-exported draft) | alpha | ◌ draft IR has beam_idx/attn_mask params LLMPipeline rejects | — |
| 11   | c3-ov-genai-engine | c3-1 | wire LLMPipeline as tahoma engine `ov-genai` | alpha | **✓ 87.1 tok/s** | -10% vs raw c1-1; **+9.8×** vs ov-optimum |
| 12   | c1-llmpipeline | c5-1 | tahoma `--engine ov-genai` on charlie | charlie | **✓ 71.3 tok/s** | **+6.9×** vs c0-2 |
| 13-17| c2-llmpipe-spec | c2-1b..5 | LLMPipeline + HF-published 1B draft, K-sweep | alpha | ⚠ +4-5% peak (K=10 → 100.9) | spec decode marginal at 64 tok with 1B draft |
| 18   | c6-long-gen | c6-1 | plain LLMPipeline at 256 tokens | alpha | 📊 21.54 tok/s | 4.5× slowdown from 64-tok 96 |
| 19   | c6-long-gen | c6-2 | LLMPipeline + draft K=5 at 256 tokens | alpha | **✓ 24.94 tok/s** | **+15.8%** over c6-1 |
| 20   | c6-long-gen | c6-3 | LLMPipeline + draft K=10 at 256 tokens | alpha | ✗ 19.41 tok/s | -10% — over-speculates on creative content |
| 21   | c10-other-models | c10-1 | Qwen 2.5 1.5B INT4 | alpha | ◌ no openvino_tokenizer.xml in IR | — |
| 22   | c10-other-models | c10-2 | Llama 3.2 1B INT4 at 256 tok | alpha | **✓ 81.07 tok/s** | (1B leaderboard) |
| 23   | c10-other-models | c10-3 | Llama 3.2 1B INT4 at 64 tok | alpha | **✓ 149.47 tok/s** | (highest tok/s for 1B INT4) |
| 24   | c8-scheduler-prefix | c8-1 | SchedulerConfig (no prefix caching) baseline | alpha | 📊 turn1 1.92s / turn2 1.57s | warm-cache effect |
| 25   | c8-scheduler-prefix | c8-2 | SchedulerConfig + prefix_caching=True | alpha | ⚠ no win; turn 2 slightly slower | needs deeper investigation |
| 26   | c11-chat-prefix | c11-1 | start_chat() + prefix_caching + 2300-tok prefix | alpha | ⚠ no win; turn 2/3 slightly slower | API form may be wrong |
| 27   | c12-kv-eviction | c12-1/2 | CacheEvictionConfig max_cache_size=512 | alpha | ⚠ no win; eviction never triggered | seq < cap |
| 28   | c10-other-models | c15-1 | gemma4-26b-MoE INT4 LLMPipeline | alpha | ◌ "Port for tensor name attention_mask was not found" | IR incompat |
| 29   | c10-other-models | c17-1 | charlie 8B 256-tok plain LLMPipeline | charlie | ⚠ 23.18 tok/s | confirms 8B long-gen is architectural |
| 30-31| c18-fastdraft | c18-1/2 | LLMPipeline + Intel FastDraft 150M K=5/10 (raw) | alpha | **✓ 119.24 / 118.22 tok/s** | **+24% over plain** — DISCOVERY #2 |
| 32-33| c18-fastdraft | c18-3/4 | FastDraft at 256 tok K=5/10 | alpha | ⚠ 24.79 / 18.88 | tied with 1B draft long-gen; K=10 over-spec |
| 34   | c18-fastdraft | c18-5 | tahoma `--engine ov-genai --draft-model fastdraft` | alpha | ✗ 86.2 tok/s — spec didn't fire (warmup config bug) | bug in engine |
| 35   | c18-fastdraft | c18-6 | engine bug fixed (set num_assistant_tokens in warmup) | alpha | **✓ 134.90 tok/s** | **+15.2× over ov-optimum baseline** |
| 36   | c18-fastdraft | c18-7 | tahoma `--engine ov-genai + FastDraft` on charlie | charlie | **✓ 96.04 tok/s** | **+9.3× over ov-optimum baseline** |
