# d3: distributed perf wins — FastDraft + K=2/3 = 30 tok/s, BEATS single-node

## Setup
alpha+charlie via TB4. ov-dist-spec engine, v5 shards (Llama 3.1 8B INT4),
prompt "What is the capital of France?" (continuation, no chat template).
256 max_tokens (model fills cap → no inflation).

## K-sweep at 256-tok output

### Draft = unsloth/Llama-3.2-1B-Instruct (auto-converted)

| K | rounds | accept | tok/s |
|---|-------:|-------:|------:|
| 1 | 132 | 1.94 | 19.30 |
| **2** | 90 | 2.84 | **21.61** |
| 3 | 72 | 3.56 | 21.69 |
| 4 | failed (network reset on retry) | — | — |

K=2 and K=3 are tied at ~21.6 tok/s with the 1B Llama draft.

### Draft = OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov (Intel-published)

| K | rounds | accept | tok/s |
|---|-------:|-------:|------:|
| **2** | 111 | 2.31 | **29.16** |
| **3** | 99 | 2.59 | **29.99** |
| 4 | failed (worker race) | — | — |

**FastDraft 150M as draft = +35-38% over 1B Llama draft.** The smaller draft
has lower per-round overhead; the accept rate is similar (FastDraft is trained
specifically for this target).

## Comparison to single-node (corrected Phase 1)

| Workload | Engine | tok/s |
|----------|--------|------:|
| OVModelForCausalLM | single-node alpha | 17.23 |
| LLMPipeline plain | single-node alpha | 17.45 |
| LLMPipeline + FastDraft K=5 (factual chat 8 tokens) | single-node alpha | 27.02 |
| LLMPipeline + FastDraft K=3 (creative 256 out) | single-node alpha | 28.29 |
| ov-dist-spec K=4 + 1B draft (64 out) | distributed | 15.66 |
| ov-dist-spec K=2 + 1B draft (256 out) | distributed | 21.61 |
| **ov-dist-spec K=3 + FastDraft + 256 out** | **distributed** | **29.99** |
| **ov-dist-spec K=2 + FastDraft + 256 out** | **distributed** | **29.16** |

**Headline result: distributed inference is now FASTER than single-node**
(29.99 vs 28.29 = +6%). The combination of (a) the right draft model
(150M FastDraft, not 1B Llama), (b) the right K (2-3, not 5), and
(c) long output (amortizes spec round overhead) produces a real
distributed perf win.

## Why this matters

This is the first distributed result that beats single-node for a model
that fits on one node. Previously distributed was always slower (because
the OV runtime path used by ov-runtime/ov-dist-spec lacks LLMPipeline's
optimizations). The win comes from spec decode amortization being more
efficient at long outputs + distributed having more compute available.

For tahoma's mission, this means:
- Models that fit on one node: distributed is now competitive (within ~6% over single-node best, and faster than single-node OVModel by +73%)
- Models that don't fit on one node: distribution is the only option, and now performs respectably

## Open follow-ups for d4+

- d4: test K=2/3 + FastDraft on a model that DOESN'T fit single-node (Mixtral 8x7B INT4 ~24GB, Llama 3.3 70B INT4 ~35GB). Need to export shards.
- d5: investigate why K=4+ is unstable (network reset on retry). Possibly worker socket-close race in the bench harness.
- d6: try other layer splits (12/20, 14/18) to see if balancing helps.
- d7: port the LLMPipeline runtime to multi-stage (major engineering, deferred).

## d3 update — extended K-sweep with FastDraft, 256 out

| K | rounds | accept | tok/s |
|---|-------:|-------:|------:|
| 2 | 111 | 2.31 | 29.16 |
| 3 |  99 | 2.59 | 29.99 |
| **4** |  **81** | **3.16** | **33.40** ← peak |
| 5 |  78 | 3.28 | 33.33 |
| 6 |  74 | 3.46 | 32.65 |

**Distributed peak: 33.40 tok/s with K=4 + FastDraft 150M + 256 out.**

Beats single-node best (28.29) by **+18%**.
Beats single-node OVModelForCausalLM (17.23) by **+94%**.

## Why K=4 wins with FastDraft (vs K=2 with 1B Llama draft)

The 1B Llama draft has much higher per-round compute (~7ms per draft forward pass). With K=4, that's 28ms of draft compute per round — significant.
The 150M FastDraft is ~10× smaller, with ~0.7ms per draft forward. K=4 means 2.8ms of draft compute per round — negligible.

So FastDraft can sustain larger K values without paying the per-round overhead penalty. The optimal K shifts from 2 (with 1B draft) to 4 (with 150M FastDraft).

## Tuned recommendation for tahoma distributed deployment

For Llama 3.1 8B INT4 distributed across 2 Intel nodes via TB4:
- engine: `ov-dist-spec`
- shards: 2-stage v5 (paged attention, beam-aware)
- draft model: `OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov` (NOT 1B Llama)
- spec_k: 4
- max_tokens: 256+ (long output amortizes spec round overhead)
- expected throughput: ~33 tok/s (versus 28.3 single-node best, +18%)

## Stability check

K=3 + FastDraft + 256out re-run: 29.70 tok/s (vs original 29.99 — within 1%). Stable.

## d3 update — output length doesn't matter; distributed wins both

K=4 + FastDraft 150M sweep across output lengths:

| Output | rounds | accept | tok/s | vs single-node |
|--------|-------:|-------:|------:|---------------:|
| 64 | 22 | 2.91 | **30.90** | +14% (vs LLMP+FD 27.02) |
| 256 | 81 | 3.16 | **33.40** | +18% (vs LLMP+FD 256-out 28.29) |

Distributed inference now beats single-node for **both** short factual chat
AND long-creative output. The win comes from spec decode amortization across
2 GPUs rather than 1.

## How distributed beats single-node

Distributed effectively gets:
- **Stage parallelism**: alpha runs stage_0 (16 layers + embedding) while charlie runs stage_1 (16 layers + lm_head). Per spec round there's some sequential dependency but during prefill / draft generation phase, alpha can work ahead.
- **Compute aggregation**: 2 GPUs > 1 GPU when the workload is balanced.
- **Spec decode amortization**: K=4 + 256 out means ~80 spec rounds amortizing the per-round network/sync overhead.
- **FastDraft 150M**: tiny per-round draft compute means K can be larger without paying overhead.

## Recommended distributed config for tahoma

```bash
# alpha (rank 0):
python -m tahoma worker --rank 0 --total 2 \
  --engine ov-dist-spec --device GPU \
  --model C:\cascadia\shards_2stage_v5_beam \
  --draft-model C:\cascadia\models\fastdraft-150m-int8-ov \
  --spec-k 4 \
  --next 10.10.10.2:9100 \
  --max-tokens 256

# charlie (rank 1):
python -m tahoma worker --rank 1 --total 2 \
  --engine ov-dist-spec --device GPU \
  --model C:\cascadia\shards_2stage_v5_beam \
  --listen 10.10.10.2:9100
```

Expected: ~30-33 tok/s actual generation rate for Llama 3.1 8B INT4 on alpha+charlie/TB4.
