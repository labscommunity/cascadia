# 045: expert-parallel serving preparation

018 measured that this 1 GbE LAN can support one stream's expert traffic,
not the fifteen-stream workload. Keep this a distinct, lone-stream study;
no aggregate-throughput claim or replacement of the ordinary pipeline.

The existing EP bank is stateless and preserves gate-order accumulation.
A new shared-bank constructor permits independent driver sessions to reuse
one resident bank; the current CLI/topology remains unchanged. Twenty EP
tests passed, including simultaneous callers, a complete invalid request,
one peer disconnecting, survivor traffic and a replacement session using
the same bank. This is CPU fixture parity, not fleet f16 GPU parity.

Remaining work: bounded listeners/session lifecycle and pipeline-driver
configuration; preserve individual expert outputs and gate-order summation;
prepare and verify compact GPU shards (about 43 GB per box), using only the
signed deployment channel and bounded read-only model transfer. Capacity,
route concentration from 040, full f16-path parity and a reversible one-layer
cost canary must precede any whole-fleet reshard/serving test. The entry/API
door is currently unreachable, so no transfer or deployment is authorized
by this preparation alone until normal settle and gate checks are possible.

Prediction to qualify a full serving run: measured EP layer p95 including
transfer below its sequential local layer cost, exact outputs on matching
GPU numerics, and no CPU fallback/OOM. Kill on worse layer latency, lost
exactness or insufficient shard headroom. 018's network-only result is not
a serving benchmark; this item is not complete.
