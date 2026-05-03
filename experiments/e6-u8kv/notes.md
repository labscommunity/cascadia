# e6 — U8 KV cache + dynamic-quant-group plugin properties (regression)

**Hypothesis:** LLMPipeline runs with `KV_CACHE_PRECISION=u8` + `DYNAMIC_QUANTIZATION_GROUP_SIZE=32` by default and gets a big perf win on Intel GPU. Setting these explicitly in `ov-runtime`'s plugin config might give us a free speedup.

**Setup:** Same as e3 (ov-runtime distributed v3 16/16, no spec, 256-tok creative) but with `--ov-kv-precision u8 --ov-dyn-quant-group 32` on both alpha + charlie.

## Result (single trial, killed early)

- elapsed: 24.25 s
- **tok/s: 10.56** vs e3 baseline 12.15 (-13%)
- alpha_ms: 8767 (vs 8733 baseline — same)
- wire_ms: 15428 (vs 14096 baseline — +9% slower)

Wire (charlie's stage_1 + network) is the slowdown. Killed campaign after one trial — clear regression.

## Conclusion

Confirms the prior python autolab finding: explicit u8 KV precision is a regression on these v3 IRs. The SDPAToPagedAttention pass that LLMPipeline applies is what makes u8 KV win there; without PA, u8 quantization adds dequantize cost on every read for no compute saving.

This won't be fixed by tweaking quant groups — it requires PA in the IR, which requires either:
- Re-export with `V5_MODE=paged_attention` (engine support for PA inputs needed)
- Or the engine applies SDPAToPagedAttention at compile time

Either way, plugin-config alone can't reclaim the LLMPipeline win on multi-stage. Defer the lever; resume with engine surgery.
