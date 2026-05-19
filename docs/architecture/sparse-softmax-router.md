# Sparse softmax for the K2.6 router — investigation, not shipped

**Branch:** `perf/sparse-softmax-085`
**Verdict:** Investigation only. Implementation **skipped**. The
"softmax" the proposal wants to make sparse is in fact a per-element
**sigmoid**, and the full sigmoid pass costs ~400 ns out of
~220 ms per layer on the routed path — six orders of magnitude
below anything worth threading a threshold parameter through the
runner / CLI / API for.

## What the proposal asked

> Router output: 384 expert scores per layer. Top-K selection picks
> top 8. Softmax is computed over all 384 (current behavior) then
> top-K filtered. Sparse softmax: apply a threshold BEFORE softmax
> to drop low-score experts entirely, saving the `exp()` cost on
> those.

The hypothesis is that the inner-loop `exp()` on 384 logits per
layer is hot enough to be worth a pre-norm filter, composing with
the existing post-softmax `routing_threshold` (filters AFTER softmax)
and the better-predictor work.

## What K2.6 actually does

K2.6's router is **not softmax**. From
`crates/tahoma-int4-gemm/src/shell.rs` (the bf16 reference path)
and `crates/tahoma-int4-gemm/src/shell_int4.rs` (the production
int4 path), the routing block is:

```text
router_logits = router_weight @ post_norm           // int4 GEMV, [384]
scores[i]     = 1 / (1 + exp(-router_logits[i]))    // per-element sigmoid
scores_for_choice[i] = scores[i] + bias[i]          // noaux_tc bias
topk_ids      = argsort_desc(scores_for_choice)[:8]
topk_w[k]     = scores[topk_ids[k]]                 // ORIGINAL sigmoid score
topk_w        = (topk_w / sum(topk_w)) * 2.827      // normalize + scale
```

Sigmoid is **per-element**: each of the 384 scores is independent
of the others. There is no normalization-by-sum over all 384 that
a "drop the small ones before computing it" trick could shortcut.
The proposal's premise — that we are computing a sum-of-exps over
384 entries and then top-K filtering — doesn't match the model.

The closest analogue is: skip the `exp(-x)` on entries whose
**logits** are far below the running top-K threshold, on the
intuition that an entry with logit `≪ 0` will have a tiny sigmoid
and almost certainly not make the top 8. That's a coherent
optimization. The question is whether it's worth shipping.

## Measurement — `bench_router_sigmoid`

The bin at `crates/tahoma-int4-gemm/src/bin/bench_router_sigmoid.rs`
times each piece of the routed path with synthetic data sized
exactly to the K2.6 layer (`HIDDEN=7168`, `N_ROUTED_EXPERTS=384`,
`INTERMEDIATE_DENSE=18432`, `TOPK=8`).

```sh
cargo run --release -p tahoma-int4-gemm --bin bench_router_sigmoid
```

Numbers on the dev box (Apple M1, scalar fallback for int4 GEMV
since no AVX-512):

| Stage                           | Median ns/layer |   % of routed path |
| ------------------------------- | --------------: | -----------------: |
| Router sigmoid (384 `exp()`)    |             417 |             0.0002 |
| Router GEMV `int4 [384, 7168]`  |         246 298 |             0.1124 |
| Top-K argsort `(384 → 8)`       |           2 217 |             0.0010 |
| 8 × expert MLP (gate+silu+up+down) | 218 951 167 |            99.8864 |
| **Routed path total**           |     219 200 099 |             100.00 |

The sigmoid loop is **0.0002 %** of the routed path on this hardware.
Even isolated to "everything in the router stage that is not GEMV,"
sigmoid is 16 % of `sigmoid + topk` together — and `sigmoid + topk`
together are 1.1 % of the router stage, which is 0.1 % of the routed
path.

### Where sigmoid sits on real target hardware

Apple M1 hits the int4 GEMV with the scalar fallback. On the actual
Xeon Gold 6252 miner (AVX-512 BW + VL) the GEMV is bandwidth-bound
and runs roughly 5–10 × faster, while the 384-element sigmoid loop
is the same on either chip (scalar `expf`, ~2 ns/elem on x86 with
a tight loop, ~1 ns/elem on M1 with libm). So on the real target:

- Router GEMV: ~25–50 µs/layer
- Router sigmoid: ~800 ns/layer
- 8 × expert MLP: still tens of ms/layer (disk-bound on cold pages
  on the miner — iter 080 measurement)

The sigmoid never crosses **0.01 %** of layer time. The proposal's
intuition that "exp() on 384 values isn't huge but it's the inner
loop" is correct for a model where the inner loop is something
else — say, a softmax over a 128 K vocab. Here the inner loop is
the 8 expert MLPs.

## Why pre-sigmoid threshold doesn't save anything elsewhere

The proposal mentions composition with two other knobs. Both
operate on **already-computed** scores, so dropping the sigmoid
work would not change either of their inputs:

- `routing_threshold` (iter 015 / experiments/007_a2): filters
  scores after sigmoid + bias, before dispatching to experts. The
  expert dispatch is what makes it valuable; the sigmoid cost is
  noise. Saving sigmoid work on dropped entries doesn't unlock any
  additional expert skipping that wasn't already possible.
- "Better predictor" (iter 047): replaces the router scores with a
  cheaper proxy when correlation with the full router is high.
  That's the **GEMV** end of the cost — replacing `int4 @ post_norm`
  with something cheaper. Independent of how we then normalize.

## Three conditions under which this is worth re-asking

1. **Router becomes vectorized softmax.** If a future model variant
   uses softmax with normalization (or the router moves to an MLP
   head with a softmax tail), the inner loop changes character —
   then pre-norm thresholding does reduce sum-of-exps work.
2. **Number of experts grows by 1–2 orders of magnitude.** At
   `N_ROUTED_EXPERTS ≈ 30 000` the sigmoid loop crosses ~30 µs,
   which is the same order as today's router GEMV. Even then the
   GEMV would also blow up proportionally.
3. **Expert MLPs become trivial relative to the router.** If
   somebody finds a way to drop the per-expert work by 100 × (e.g.
   tiny LoRA-style sparse adapters instead of full MLPs), the
   sigmoid's share of layer time rises by 100 × — to 0.02 %. Still
   not worth a flag.

## What did ship in this branch

- `crates/tahoma-int4-gemm/src/bin/bench_router_sigmoid.rs` —
  the microbench used to back the numbers above. Self-contained,
  no model on disk required, deterministic synthetic data. Run it
  on any platform to update the table.
- This doc.

No runtime code changes. The router sigmoid is left as it is.
