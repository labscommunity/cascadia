# Attention-based predictive expert prefetch — investigation + skeleton

**Branch:** `perf/attn-predict-prefetch-087`
**Verdict (C):** Cost half of the investigation says **PROCEED** — a
shadow router GEMV is 0.10 % of routed-path wall time. The accuracy
half is **unresolved without a real K2.6 trace on miner**; the
skeleton ships pure helpers so the accuracy bench can run against a
live model when a slot opens.

## What the proposal asked

> iter 047 prefetches based on hit-frequency history. iter 057
> prefetches next layer's hot experts. **Better predictor: use
> attention output to predict next-layer routing.** The hidden state
> after attention is the input to the next router; we could run a
> cheap "shadow router" forecast.

Concretely: while layer i's expert MLPs are still running (~150 ms
per layer on miner — the 99 %-of-decode bucket), use *layer i's
post-attention hidden state* as a proxy for layer i+1's router input,
multiply by layer i+1's router weights, take the top-K, and hand
those IDs to the existing iter 029 prefetcher as a hint for which
expert weight slices to page in next.

This is the same family as ProMoE (arxiv 2410.22134, 84.7 % hit rate
on DeepSeek-V2's router with an MLP predictor) and MoE-SpeQ (arxiv
2511.14102, more recent), except instead of training an offline MLP
we re-use the next layer's own router as a one-step-shifted forecast.

## Why this is *not* the same as iter 057

iter 057's `speculative_prefetch_expert_ids` consults the
`expert_hits[i + 1]` fire-count histogram (the iter 054 source of
truth for which experts have *historically* fired on layer i+1). The
prediction is "the experts that fired most often in the past will
fire again", with no per-token adjustment. On steady-state generation
that's fine; on prompts that hit a fresh corner of the distribution
it's the same as same-as-last-token prefetch with extra steps.

The shadow-router prediction is per-token: each layer makes a fresh
forecast using *this token's* attention output. The bet is that the
intra-token signal beats the cross-token average.

## Cost — `bench_shadow_router`

The bin at `crates/tahoma-int4-gemm/src/bin/bench_shadow_router.rs`
times one shadow router GEMV against the routed-path baseline, sized
exactly to the K2.6 layer (`HIDDEN=7168`, `N_ROUTED_EXPERTS=384`,
`INTERMEDIATE_DENSE=18432`, `TOPK=8`).

```sh
cargo run --release -p tahoma-int4-gemm --bin bench_shadow_router
```

Numbers on the dev box (Apple M1, scalar fallback for int4 GEMV
since no AVX-512):

| Stage                              |  Median ns/layer |    % of routed path |
| ---------------------------------- | ---------------: | ------------------: |
| Router GEMV `int4 [384, 7168]`     |          216 659 |              0.0988 |
| **Shadow GEMV** (same shape)       |      **220 486** |          **0.1005** |
| Shadow sigmoid (384 `exp()`)       |              633 |              0.0003 |
| Shadow top-K argsort (384 → 8)     |              868 |              0.0004 |
| 8 × expert MLP (gate+silu+up+down) |      219 118 600 |             99.8990 |
| **Routed path total**              |      219 336 761 |              100.00 |
| **Shadow overhead (GEMV+σ+topk)**  |      **221 988** |          **0.1012** |

Per layer the shadow router costs ~222 µs on M1 — the same order as
the real router GEMV, because it *is* the same kernel call. Across
all 60 routed layers per decode step that's ~13 ms / ~99 ms = 0.1 %
of decode wall time. On the Xeon Gold 6252 miner the GEMV runs
5–10 × faster (AVX-512 vs scalar fallback) but the expert MLPs do
not — they're bandwidth-bound on disk reads, not compute-bound —
so the ratio holds or improves.

The proposal cited 0.11 % per layer from iter 085; the measured
0.10 % here is the same number arrived at differently (iter 085
measured router GEMV alone in isolation; this bench measures the
full shadow pipeline against the full routed path). Doubling the
router GEMV cost is essentially free.

### Cost verdict

The cost-side concern raised by the proposal ("Cost-benefit unclear
— shadow router GEMV per layer is significant compute") is **not
borne out by measurement**. The shadow router is cheap. **Cost is
not the bottleneck for this technique.**

## Accuracy — what we cannot measure here

The shadow router's value depends on whether the predicted top-K
intersects the actually-fired top-K substantially better than the
existing predictors (iter 057's hit-frequency, or naive
same-as-last-token).

The premise is:

```text
post_norm_{i+1}(h_{i+1}) ≈ post_norm_{i+1}(attn_residual_i)
                          where h_{i+1} = attn_residual_i + shared_out_i + moe_out_i
```

This holds if `shared_out_i + moe_out_i` is small relative to
`attn_residual_i` — i.e., if MoE output is residual-like in
magnitude. On K2.6 the `ROUTED_SCALING_FACTOR=2.827` weights the
expert sum so that it is *not* trivially small (the per-expert
`topk_w / sum(topk_w)` sums to 1; multiplied by 2.827, the expert
contribution to the next residual has the same order of magnitude
as the residual stream itself). So a-priori the proxy is lossy in
a way that could push top-K predictions far enough to miss the
actually-fired experts.

But the actually-fired top-K is also forgiving: K2.6 fires 8 of 384,
and recent loop measurements (iter 054 hot-expert histograms) show
a heavy-tailed firing distribution where 30–50 of the 384 cover
most decode tokens. Predicting top-12 to top-32 with the shadow
router could give a high enough hit rate to be useful even with a
lossy proxy.

There are **three unknowns** that only a real K2.6 trace on miner
can resolve. None can be answered with synthetic random data
because the answers depend on activation statistics specific to a
trained model on real prompts:

1. **Hit rate of shadow-top-K vs actually-fired top-K**, for K in
   {8, 12, 16, 24, 32, 64}. Need a per-layer recall@K curve. The
   wire-up is the same as iter 047's `prefetch_hits / prefetch_chances`
   counter pair — drop it onto the shadow-router output instead of
   the histogram, increment under the same `lid` accumulator.

2. **Cross-layer correlation of the prediction error.** If the
   shadow router is systematically wrong on the *same* token across
   *most* layers (i.e. the prompt is out-of-distribution for the
   model's attention head agreement), then per-layer prefetching
   doesn't compose and the win caps at iter 057. If the errors are
   uncorrelated across layers, prefetching on the layers where the
   shadow router *is* accurate is the right strategy and the win is
   additive in N_layers.

3. **Composition with iter 047 / 054 / 056 / 057 / 065.** The
   prefetcher channel is bounded; if shadow-router predictions
   collide with the hit-frequency predictions on the same expert
   IDs, the marginal value of the new signal is zero. Need the
   diff-vs-iter-057-baseline (extra IDs the shadow router suggests
   that the histogram doesn't).

## What this branch ships

- `crates/tahoma-int4-gemm/src/bin/bench_shadow_router.rs` — the
  microbench used to back the numbers above. Self-contained, no
  model on disk required, deterministic synthetic data. Run it on
  any platform to update the table.
- `crates/tahoma-int4-gemm/src/shell_int4.rs` —
  `shadow_router_predict_topn` pure helper: given a post-attention
  proxy hidden state, the next layer's int4 router weights, scales,
  bias, and N, returns the top-N expert IDs by `sigmoid(GEMV) + bias`.
  Same arithmetic order as the production router. No runtime wiring
  — this is the leaf the prefetcher hook will call when iter 057's
  branch is rebased on this work. Unit tests cover the contract
  (top-K invariant, deterministic for fixed weights, sane behavior
  on N=0, N=TOPK, N=N_ROUTED_EXPERTS).
- This doc.

No runtime CLI flag. No engine plumbing. No changes to the
`Runner` (the prefetcher and `expert_hits` map both live on iter
057's branch, not on main — wiring a hint with no consumer would
be dead code).

## Bench plan for the accuracy half (deferred)

When a miner slot opens (and the parent loop has iter 047/054/056/057
stacked or merged), the path forward is mechanical:

1. Rebase this branch onto the iter 057 base (or whatever is the
   current stack head).
2. Add a `shadow_router_n: Option<u32>` to
   `SparseMoEBuilderConfig` mirroring the existing
   `speculative_prefetch_n` plumbing. `--shadow-router-n N` on the
   CLI; `None` keeps iter 057 behavior.
3. In `runner.rs::forward_shells`, after each layer i's shell
   forward but before the dispatch loop, when `i + 1 < n_layers`
   and `Some(n)`:
   - Call `shadow_router_predict_topn(&outs.attn_out_post_norm,
     &self.layers[i + 1].int4_shell, n)`.
   - For each predicted ID, `prefetcher.try_submit(next_lid, eid)`.
   - Increment a `shadow_router_hits / shadow_router_chances` pair
     by comparing the prediction to the *next* layer's actual
     `outs.routing_ids` after it runs.
4. Wire the counters into the per-token stage timing log line
   alongside the existing `prefetch_hits / prefetch_chances` (which
   iter 047 already emits).
5. Bench plan: one bench per N ∈ {8, 12, 16, 24}, three prompts,
   three runs each. Compare:
   - Hit rate (shadow vs iter 057's histogram)
   - tok/s (shadow + iter 057 vs iter 057 alone vs main)
   - Cumulative prefetch bandwidth (drops/submits ratio at the
     prefetcher channel)

Two outcomes worth shipping:

- **If shadow recall @ K matches or beats iter 057 on disjoint
  experts**, ship it as an additional signal: union the shadow IDs
  with the histogram IDs in `speculative_prefetch_expert_ids`.
- **If shadow recall is comparable but covers different prompts**
  (e.g., wins on novel prompts, loses on repetitive ones), gate
  per-token: emit shadow only when the same-as-last prediction
  agrees with itself across layers (low-confidence regime).

Either result deletes the skeleton; neither result reverts the
microbench (it stays as a cost regression test).

## Conditions under which this is *not* worth re-asking

This investigation's cost analysis is decisive — a shadow GEMV is
cheap and stays cheap as long as the routed path stays dominated by
expert MLPs. The investigation should be reopened only if the
accuracy bench (above) hits one of:

- **Shadow recall @ K below the same-as-last baseline.** The proxy
  is too lossy for K2.6's router distribution. No amount of
  prefetcher cleverness saves it.
- **Shadow recall @ K matches iter 057 exactly, on the same IDs.**
  No new information; the histogram already captures everything the
  shadow router can predict. Costs 0.1 % of decode for zero hit-rate
  lift over iter 057.
- **Shadow recall causes iter 057 prefetcher channel saturation**
  (drops/submits > 50 %). The signal is good but the integration
  point is wrong — the right fix would be a higher-bandwidth backend
  (iter 074 io_uring) before re-running this bench.

## References

- iter 047 (`perf/c1-better-predictor-047`): top-N from router scores
  predictor — the cross-token analogue.
- iter 054 (`perf/expert-pinning-054`): hot-set histogram used by
  iter 057.
- iter 057 (`perf/async-kernel-sched-057`): speculative next-layer
  prefetch, using histogram source.
- iter 074 (`perf/io_uring_prefetch.md`): higher-bandwidth
  prefetcher backend, dependency-free knob for accuracy half.
- iter 085 (`perf/sparse-softmax-085`): companion investigation
  using the same `bench_router_sigmoid` cost framework — confirms
  the routed path is 99 %+ expert MLPs.
- ProMoE (arxiv 2410.22134): offline MLP predictor for next-layer
  routing on DeepSeek-V2. The shadow router is the same idea with
  the model's own router weights as the predictor.
- MoE-SpeQ (arxiv 2511.14102): more recent, broader survey of
  predictive prefetch in the MoE literature.
