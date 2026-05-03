# e10 — full per-stage timing breakdown on dist_spec K=1 (factual)

**Setup:** Same as e8 K=1 trial (factual prompt, 256 tokens, alpha+charlie/TB4). One trial with the new INFO-level `spec_decode timing` line that splits target.feed into alpha-side (setup + stage_0 infer + output read) vs wire (network + charlie stage_1 + recv).

## Result

```
spec_decode timing
    target_ms = 12876            (80% of total — distributed verify)
    draft_ms = 3214              (20% of total — alpha GPU draft)
    other_ms = 71                (~0)
    total_ms = 16162
    target_alpha_setup_ms  = 7      (set_input cost)
    target_alpha_infer_ms  = 4980   (alpha B390 stage_0 GPU compute) — 31% of total
    target_alpha_output_ms = 2      (read stage_0 output)
    target_wire_ms         = 7872   (network + charlie stage_1 + recv) — 49% of total
```

185 spec rounds (K=1), 1.38 tokens/round.

## Per-round breakdown

- alpha stage_0 compute: 4980 / 185 = **27 ms**
- wire (charlie + net): 7872 / 185 = **43 ms**
- draft.feed (incl. rejection rewinds): 3214 / 185 = ~17 ms
- other: ~7 ms

Per-round wall ≈ 94 ms ⇒ 14.7 tok/s (matches measured 15.84 within noise).

## Implications for the perf bar

This data localizes the structural ceiling:

- **Optimal async overlap** would let alpha draft DURING the 43 ms charlie wait. Best case: per-round = 27 (alpha stage_0) + max(17, 43) (drafts overlapped with wait) + 7 (reconcile) = **77 ms** ⇒ **17.9 tok/s on factual K=1**.
- That is +13% over current 15.81. Worth shipping but **NOT enough to reach the bar (28 tok/s)**.
- The bar requires either:
  1. **Parallel work that doesn't depend on autoregressive token sequence** (none exists for single-user sequential without speculation)
  2. **Speculation that reliably succeeds** (FastDraft accepts 38% on factual, much less on creative — not enough)
  3. **Per-stage compute speedup** (PA re-export is dead-end; LLMPipeline-class optimizations are not retrofittable to per-stage IRs in OV 2026.1)
  4. **Hardware concurrency within a host** (NPU+GPU TP — major engineering)
  5. **Models that don't fit single-node** — distributed wins by default because single-node OOMs (Mixtral 8x7B, Llama 70B)

## The fundamental limit (filed as a discovery)

For Llama 3.1 8B INT4 on alpha (B390 dGPU) + charlie (LL 140V iGPU), 2-stage PP for single-user sequential decoding is **structurally bounded at ~17-18 tok/s** with optimal engineering, vs single-node alpha at 23 tok/s (e0). Beating single-node is not possible without changing one of: model, hardware concurrency model, or speculation regime.

→ See DISCOVERIES.md D2.

## Pivot

- Implement bounded async overlap for the +13% gain it offers (real, just modest).
- Pivot the moonshot search to: (a) Mixtral/Llama-70B "doesn't-fit-single-node" wins; (b) within-host TP via NPU concurrency.
