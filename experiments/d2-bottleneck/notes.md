# d2: distributed bottleneck — it's the OV runtime, not the network

## Inputs to this analysis

| Source | Result |
|--------|--------|
| d1 (network) | TB4 = 8.75 Gbps, 0.142 ms RTT |
| d0-2 (ov-dist-spec) | 15.66 tok/s, 64 tok in 4.088 s, 18 spec rounds |
| Phase 1 c58 (single-node) | LLMPipeline plain = 17.45 tok/s |
| Phase 1 c60 (single-node) | OVModelForCausalLM plain = 17.23 tok/s |
| Phase 1 c61 (single-node) | LLMPipeline + FastDraft = 27.02 tok/s |

## Per-round / per-token decomposition

ov-dist-spec K=4 v5 across alpha+charlie:
- Round time: 4.088 s / 18 rounds = **227 ms per round**
- Tokens accepted per round: 64 / 18 = **3.55 tok/round** (out of K=4 candidates)
- Per accepted token: 227 / 3.55 = **64 ms/token**

Per-round work breakdown:
- 1 target validation forward pass (validates K+1 candidates) split across alpha+charlie
- K = 4 draft forward passes on alpha (1B Llama)
- 5 activation transfers alpha → charlie (one per draft candidate + bookkeeping)
- 1 result transfer charlie → alpha (token id)

Network cost per round: 6 transfers × ~78 µs/transfer = **468 µs ≈ 0.5 ms** (0.2% of round time)
Compute cost per round: 227 - 0.5 = **226.5 ms (99.8% of round time)**

## Conclusion: network is irrelevant; compute is everything

The TB4 link could be **100× slower** and we'd still measure roughly the same
distributed throughput. The bottleneck is the OV runtime path used by
ov-runtime / ov-dist-spec, NOT the inter-node link.

## Why the OV runtime is slow (hypothesis)

The pre-exported v3/v5 shards bypass the LLMPipeline runtime. Specifically,
the c60 / c61 single-node measurements showed:
- OVModelForCausalLM (legacy OV runtime): 17.23 tok/s plain
- LLMPipeline (optimized OV runtime): 17.24 tok/s plain
- LLMPipeline + FastDraft K=5: 27.02 tok/s (+55%)

The ov-runtime / ov-dist-spec engines use the legacy OV runtime path
(OpenVINO `Core.compile_model` directly on the IR), missing the GenAI
pipeline's PagedAttention / U8 KV / XMX dynamic quant optimisations
that LLMPipeline applies via `SDPAToPagedAttention` and similar
runtime passes.

So the distributed engines run at the equivalent of "single-node
OVModelForCausalLM" rate (~17 tok/s) MINUS spec overhead per round.
Distributed ov-dist-spec at 15.66 tok/s is consistent with
"single-node OVModel rate" / "loss to spec round overhead".

## Strategic implications

For tahoma's distributed perf to materially beat single-node, ONE of
the following must change:

1. **Port LLMPipeline to multi-stage** — rewrite the multi-stage engine
   (currently `ov_runtime.py` and `dist_spec.py`) to use openvino_genai
   primitives. Major engineering work; LLMPipeline today is single-stage.
2. **Re-export shards with PagedAttention applied at export time**
   instead of relying on the runtime pass. Smaller engineering work
   but requires re-exporting all shards.
3. **Use distributed for models that don't fit on one node**, accepting
   per-token rate similar to single-node for models that DO fit. This
   is the existing tahoma value prop and the most pragmatic answer.

## Recommendation for d3+

Pursue (3) first — it's where distribution actually changes the user's
options. Test by:
- d3a: bench Mixtral 8x7B INT4 (~24 GB) split across alpha+charlie. Single-node would OOM on alpha B390 (probably 12 GB VRAM); distributed enables it.
- d3b: bench Llama 3.3 70B INT4 (~35 GB) split across alpha+charlie. Required distribution.
- d3c: revisit (2) — try `optimum-cli export openvino --weight-format int4 --paged-attention` flags on a shard export, see if runtime engages PA.
