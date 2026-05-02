# c10: how the LLMPipeline win generalises across models + sizes

## Results — alpha B390 GPU, plain LLMPipeline, INT4

| Model | Output tokens | Decode tok/s | Per-token (ms) |
|---|---|---|---|
| Qwen 2.5 1.5B | 64 | (skipped — IR has no `openvino_tokenizer.xml`) | — |
| Llama 3.2 1B | **64** | **149.47** | **6.7** |
| Llama 3.2 1B | **256** | **81.07** | **12.3** |
| Llama 3.1 8B | **64** | **96.41** | **10.4** |
| Llama 3.1 8B | **256** | **21.54** | **46.5** |

## Headlines

1. **Llama 3.2 1B INT4 hits 149 tok/s** at 64-token output on alpha B390 — a new tahoma leaderboard high. The 1B is so small that single-token decode is mostly memory-bandwidth-bound on KV cache, not compute-bound.
2. **The 1B's long-gen degradation is mild (1.8×) vs the 8B's (4.5×)** going 64 → 256 tokens. Per-token cost balloons faster on the larger model because each layer's KV-attention does more work per step.
3. **Models without `openvino_tokenizer.xml` don't work with LLMPipeline** out of the box. The OV-format tokenizer files have to be either bundled by `optimum-cli export openvino` (which they are by default since OV 2024.5+ when `openvino-tokenizers` is installed) or generated separately via `convert_tokenizer`. Our older Qwen/Gemma INT4 dirs were exported before this and need re-export or manual tokenizer-conversion before they work with LLMPipeline.

## Implications

- A real serving stack with the LLMPipeline path needs an export-time check that the tokenizer XML is present. Users with old IRs will hit this error otherwise.
- The 8B's 4.5× long-gen degradation is **the next big problem to attack**. Sparse attention prefill (SnapKV, OV 2025.3) doesn't apply (short prompt). KV eviction (CacheEvictionConfig) does — at the cost of some output quality. Another follow-up.

## Open questions

- Does the 10× win over OVModelForCausalLM hold for non-Llama architectures (Qwen, Phi-3, Gemma)? Need re-exported IRs to confirm. Architectural differences (e.g. Qwen's tied embeddings, Gemma's unique attention) could shift the multiplier.
- Does the long-gen degradation pattern (`per_token_cost ≈ const × kv_cache_size`) hold for the 1B too? Looks like yes (1.8× over the 4× cache size growth).
