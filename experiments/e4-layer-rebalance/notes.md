# e4 — layer rebalance 22/10 (alpha-heavy)

**Hypothesis:** Charlie's LL 140V iGPU is ~2× slower per layer than alpha's B390 dGPU. Re-exporting v3 shards with an asymmetric 22/10 split (alpha 22 layers / charlie 10 layers) should rebalance per-stage compute and increase throughput.

**Setup:**
- Re-exported via `rainier/scripts/export_cached_shards_v3.py` on alpha with `--layer-split 22`
- Output: `C:\cascadia\shards_2stage_v3_22_10\` (~4 GB INT4)
- Stage_1 + tokenizer + config.json zipped + scp'd to charlie
- ov-runtime no-spec, same 256-tok creative prompt
- 5 trials, charlie restarted between each

## Result

| Trial | tokens | engine elapsed (s) | tok/s |
|------:|-------:|-------------------:|------:|
| 1 | 256 | 20.93 | 12.23 |
| 2 | 256 | 20.99 | 12.20 |
| 3 | 256 | 20.80 | 12.31 |
| 4 | 256 | 22.54 | 11.36 |
| 5 | 256 | 20.52 | 12.48 |

**Median: 12.23 tok/s** vs e3's 16/16 split at 12.15 tok/s = **+0.7%, in noise**.

## Conclusion — the rebalance hypothesis was wrong

Layer split is essentially a no-op. The bottleneck didn't move much because:

- 16/16: alpha ~25ms compute, charlie ~40ms compute, network ~17ms = 82ms/token
- 22/10: alpha ~33ms compute, charlie ~25ms compute, network ~17ms = 75ms/token (predicted)

The PREDICTED 8.5% improvement didn't materialize. Best explanation: per-step OpenVINO runtime overhead (set_input × 4 + infer + output_copy) is significant relative to per-layer compute, so reducing layer count on charlie doesn't reduce its time proportionally.

Either:
- (a) per-layer compute is faster than 2.5ms/layer for charlie (the 16/16 was bottlenecked by something else), or
- (b) per-step OV overhead is large (~20ms), and reducing layer count from 16 to 10 only saves ~15ms while increasing alpha by ~9ms = net ~+6ms which we then lose to noise

Either way, **layer split is not the lever**. Need to attack:

1. Per-step OV overhead — possibly via paged-attention re-export (KV cache pre-allocated, fewer set_input calls, internal block-table)
2. Async stage overlap — start stage_0 of round N+1 while stage_1 of N is computing
3. Per-stage compute — INT4 group size, U8 KV cache plugin property

## Pivot

Stopping the layer-split exploration. The single-trial 20/12 export was queued but its data point is unlikely to differ. Focus shifts to engine surgery:

- e5: profile per-step internals at INFO level (currently debug only) so we can localize overhead
- e6+: engine modifications for paged-attention or async overlap

Negative result, but valuable — confirmed layer split is not the answer.
