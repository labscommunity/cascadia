# M3 step 1 — embed-as-pseudo-head feasibility test (FAILED)

**Hypothesis:** Project the v3 stage_0 hidden state (output of layer 16 of Llama 3.1 8B) through `embed_tokens.weight^T` to get pseudo-logits. If pseudo_token = argmax(pseudo_logits) agrees with real_token = argmax(stage_1 lm_head output) ≥ 40% of the time, the early-exit pseudo-head is a viable speculation source for the multi-day engineering investment.

**Setup:**
- alpha (Battlemage Arc B390 dGPU)
- v3 stage_0 IR (`shards_2stage_v3/stage_0/openvino_model.xml`) — 16 layers + embed
- Reference: full Llama 3.1 8B INT4 IR (`models/llama-3.1-8b-int4/openvino_model.xml`) via OV directly (not LLMPipeline)
- embed_tokens.weight loaded from `models/llama-3.1-8b-src/` safetensors via torch (bf16 → f16)
- Prompt: 10-token RISC-vs-CISC technical seed; iterative greedy decode of 32 tokens
- Both models fed identical input + position; compared per-position pseudo_token vs real_token

**Code:** `logs/m3_pseudo_head.py`

## Result

**Per-step agreement: 0/32 = 0.0%.**

Inspecting the pattern, pseudo-token at position N usually equals real-token at position N-1 — the layer-16 hidden state is mostly representing the INPUT token's identity, not the NEXT token's. The 16-layer prefix hasn't yet evolved the residual stream enough to predict the next token; the embedding projection essentially re-extracts the input.

```
 pos real_tok pseudo_tok  match
   0      271      63092      n
   1      791      41148      n
   2     1401        791      n     <- pseudo at pos 2 == real at pos 1 (off-by-one shift)
   3    12062       1401      n     <- same pattern
   4     1990      12062      n
   5     1595       1990      n
   ...
```

## Conclusion — M3 is non-viable as proposed

The early-exit pseudo-head moonshot is **dead** at the cheapest validation step. Multi-day engineering would have produced a speculation source with 0% accept rate — strictly negative win.

**Why this fails:** the pseudo-head is doing prediction at depth 16/32. For Llama 3.1 (32-layer model), the residual stream at layer 16 still primarily encodes the INPUT token's lexical identity. The "switch" to predicting the NEXT token happens in the deeper layers (typically layers 24-32 for 32-layer models, per published interpretability work).

**Variants worth trying** (lower priority):
- Pseudo-head at deeper layers (e.g. layer 24 or 28). Would require re-exporting stage_0 with more layers.
- A learned linear head trained to predict next token from layer-16 hidden state. Multi-week training effort.
- Cross-model speculation: a smaller model (Llama 3.2 1B) as the speculator runs on alpha NPU; same family so embedding spaces align. The d4 python autolab tested this and saw -75% due to OV cross-device sync overhead.

## What this tells us about the structural limit

D2 (the 17-18 tok/s ceiling for 2-stage PP) holds even more strongly: there is no cheap "fake stage_1" we can run on alpha to break the autoregressive serialization. The lm_head at the end of stage_1 is doing real work that can't be skipped or approximated by an earlier projection.

Filing as DISCOVERIES D3.
