# 040: measure routed-expert concentration on real fleet traffic

Count routed selections in every MoE layer with
CASCADIA_INKLING_EXPERT_COUNTS=1 (default off). Counts include prompt and
decode routing work, including speculative or fallback evaluations. Shared
experts are excluded. Every 1024 routing rows, emit cumulative top-16/32/64
shares, maximum share and distinct count using a changing per-layer tag.

Run the twelve prompt families at fifteen streams for at least ten minutes.
Uniform routing gives top-64 share 25%; the hot-replica proposal requires
over 60% at most of the 64 MoE layers. Collect summaries from all layers,
not only the head or the latest rank. If that concentration is absent, close
the hot-replica premise with the measured distribution.

Prediction: counting overhead is below 1% and exact outputs are unchanged.
Counter unit tests and four pipeline/capture/reconnect tests with counters
enabled passed. Kill the enabled diagnostic after measurement, or sooner
if gates fail or it measurably slows serving.
