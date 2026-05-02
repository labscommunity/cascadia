# e0: paged-attention engineering — research + validation

## Research summary

The `paged_attention_transformation` Python API exists in
`openvino._offline_transformations`:

```python
from openvino._offline_transformations import paged_attention_transformation
paged_attention_transformation(
    model,
    use_block_indices_inputs=False,
    use_score_outputs=False,
    allow_score_aggregation=False,
    allow_cache_rotation=False,
    allow_xattention=False,
    allow_adaptive_rkv=False,
)  # mutates in place
```

It rewrites SDPA → PagedAttentionExtension, replaces stateful KV variables
with key_cache.N / value_cache.N parameters, removes attention_mask and
beam_idx, and adds (max_context_len, past_lens, subsequence_begins,
block_indices_begins, block_indices) inputs.

LLMPipeline pairs it with `apply_gather_before_matmul_transformation`
(internal C++ helper, not exposed in Python).

## Validation result — pass works on optimum-cli models, FAILS on v5 shards

### `C:\cascadia\models\llama-3.1-8b-int4` (optimum-cli export, single-shard)

- 32 SDPA → 32 PagedAttentionExtension ✓
- Inputs after: `[input_ids, position_ids, key_cache.0..31, value_cache.0..31, past_lens, subsequence_begins, block_indices, block_indices_begins, max_context_len]` ✓
- attention_mask and beam_idx removed cleanly ✓
- The transform succeeds.

### `C:\cascadia\shards_1stage_v5_beam` (rainier-exported v5 shard)

- 32 SDPA STAYS as SDPA (transform fails to replace) ✗
- 64 stateful variables stay as ReadValue/Assign ✗
- attention_mask + beam_idx are removed from inputs but STILL referenced in graph (dangling refs) ✗
- `Model references undeclared parameters: beam_idx, attention_mask`

The rainier-exported v5 shards have an SDPA op variant that the
`SDPAToPagedAttention` pass does not recognize. Likely the export uses
torch.jit.trace + nncf, which produces a slightly different SDPA opset
than what optimum-cli's HF export does.

## Validation result — single-node PA gives no perf win (c60 retrospective)

From Phase 1 campaign c60 with proper measurement:

| Engine | tok/s (8B INT4 factual) |
|--------|------------------------:|
| OVModelForCausalLM (NO PA) | 17.23 |
| LLMPipeline (DOES apply PA) | 17.24 |
| ratio | 1.00× |

PA does NOT change per-token throughput for the single-node 8B INT4
case. The c60 conclusion was that LLMPipeline ≈ OVModel at raw rate.

This means: even if we could engineer the v5 shards to engage PA in
multi-stage, the perf gain would be ~0%.

## Conclusion: e0-e2 engineering path is BLOCKED

Two compounding blockers:

1. **The PA transform fails on rainier-exported v5 shards.** Would require
   either re-exporting the shards via optimum-cli style (which doesn't
   support multi-stage), or modifying rainier's export to produce
   PA-compatible SDPA ops, or writing a custom transform pass for v5.
2. **Even if blocker 1 were fixed, single-node PA shows ~0% perf win.**
   Per c60's apples-to-apples measurement, PA is not the source of
   LLMPipeline's perf advantage — that's spec decode amortization.

## Where the perf actually is for distributed

The d3 finding (38.49 tok/s) is the demonstrated distributed peak. Its
sources:
- FastDraft 150M companion (small per-round draft compute)
- K=4 (right amount of speculative tokens for distributed amortization)
- Long output (256-1024 tok) so spec round overhead amortizes
- ov-dist-spec engine's mask-based KV-cache rewind (efficient validation)

These are all in the spec-decode + scheduling layer. The OV runtime
optimizations (PA, U8 KV, XMX) appear to be fully engaged via the
underlying compile_model + stateful KV path on Battlemage / Lunar Lake
in OV 2026.1.

The engine code is essentially already at the OV runtime ceiling for
this hardware class. Further perf would require:

1. **Different model** — running something that better matches the
   compute/memory ratio of 2 nodes' aggregate (capacity argument).
2. **Multi-tenant / batched** — where continuous batching kicks in. Not
   relevant to single-prompt streaming.
3. **Faster fabric** (NVLink-class) — irrelevant; we're not bandwidth-
   or latency-bound on TB4.

## Recommendation

**Do not pursue the e0-e2 engineering path.** The d3 winning config
(ov-dist-spec K=4 + FastDraft + long output = 38.49 tok/s) is the
demonstrated answer for distributed perf on Llama 3.1 8B INT4 across
alpha+charlie/TB4.

For further perf gains, target:
- The capacity/MOE story (run models too big for one node)
- The multi-tenant story (CB scheduling across distributed)
- Both are larger engineering projects with payoffs that aren't
  per-token-tok/s on 8B.

## Saved scripts

- `validate_pa.py` — initial validation
- `validate_pa3.py` — proper introspection of `_offline_transformations`
- `validate_pa4.py` — inspect dangling references after failed transform
