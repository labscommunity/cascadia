# d5: layer split rebalance — uneven splits hurt

ov-runtime engine (v3 shards), 256 max_tokens:

| Split (alpha/charlie) | tok/s |
|-----------------------|------:|
| 16/16 (`shards_2stage_v3`) | ~17 |
| 14/18 (`shards_2s_14_18`)  | ~12.8 (-25%) |

Charlie's iGPU is slower per-layer than alpha's dGPU. Loading charlie
with MORE layers makes the PP bottleneck stage even slower.

What we'd want is the OPPOSITE split (e.g., 20/12 — alpha heavier).
Don't have those shards. Re-export needed.

Pragmatic answer: stick with d3 winning config (ov-dist-spec K=4 + FastDraft
16/16 v5 = 38.49 tok/s).
