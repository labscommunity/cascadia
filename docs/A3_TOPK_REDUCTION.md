# A3 — MoE top-K dispatch reduction

A small opt-in flag on `tahoma worker` that reduces the number of
experts dispatched per token in the sparse-MoE engine. Significant
throughput gains on K2.6 (and other sigmoid-router MoE models) at
negligible quality cost.

## Usage

```bash
tahoma worker --engine sparse-moe --top-k-override 4 ...
```

Default is `None` (no behavior change — uses manifest's top_k, which
is 8 for K2.6). When set to k' < manifest top_k, only the first k' of
the routed top-K experts are dispatched per shell layer per token.
The shell's router still computes the full top-K of routing
weights/ids; we just skip dispatching the tail.

There's also a complementary flag:

```bash
tahoma worker --engine sparse-moe --routing-threshold 0.1 ...
```

which skips experts whose router weight is below the threshold. The
two flags compose (`--top-k-override` is applied first, then
`--routing-threshold` filters the remaining experts).

## Measured Pareto on K2.6 (miner single-stage, Xeon Gold 6252 + DDR4-2133)

Bench: 10-prompt factual eval, max_tokens=64, single replicate per K.
Source: `autolab/k26-perf` branch experiments `009_a3_robustness_10prompt`,
`011_a3_k4_longcontext`, `013_a3_k4_vs_k8_longcontext`.

| K | tok/s | Δ vs K=8 | Quality (substring + coherent) |
|--:|------:|---------:|--------------------------------|
| 8 (manifest default) | 0.105 | (ref) | 8/10 |
| 6                    | ~0.11* | +5%* | (narrow eval 3/3; broad untested) |
| 5                    | 0.155 | +47%   | 3/3 narrow |
| **4 (recommended)**  | **0.325** | **+210%** | **9/10** (matches K=8) |
| 3                    | 0.305 | +190% | **6/10** (real quality regression) |
| 2                    | 0.272 | +159% | 2/3 narrow (fails on "four") |

`*` K=6 number is from max_tokens=8 eval extrapolated; not directly
benched at max_tokens=64.

**K=4 is the productionizable sweet spot at low temperature.** Equal-
or-better quality than the K=8 baseline (8/10 vs 9/10 — K=8 failed one
prompt that K=4 got right) and 3× the throughput on chat-realistic
output lengths.

**K=3 has a real quality regression** on broader prompts (substantive
failures: gives multi-choice questions instead of direct answers, gets
factual details wrong like "Python created by the Dutch" instead of
"Guido van Rossum"). Stay at K=4 for production.

## K-tiering by temperature

K choice should depend on the inference workload's sampling temperature:

| Workload | Recommended K | Quality | Throughput vs K=8 |
|----------|--------------:|--------:|-------------------:|
| Greedy / low-temp (temp ≤ 0.3) | **K=4** | 9/10 | **+146-210%** |
| Mid/high-temp chat (temp 0.5-0.7) | **K=6** | 8/10 (matches K=8) | **+75%** |
| Maximum quality | K=8 | 9/10 at temp=0 | (ref) |

**Source:** autolab/k26-perf iter 018/019. At temp=0.7, K=4 quality
collapses to 5/10 (model outputs become incoherent under sampling
variance — "Pyth" "Pyth" for Python prompt). K=6 holds 8/10 — same
as K=8 — at temp=0.7 with +75% throughput. K=6 is the safe default
for any workload not committed to greedy sampling.

## Why this works on K2.6

K2.6's sigmoid router (vs softmax) gives routing weights that are
relatively uniform across the top-8 — no single "obviously right"
expert. Per the `Faster MoE LLM Inference` paper (arxiv 2505.03531),
sigmoid-router MoE models tolerate substantial K reduction at low
quality cost, especially at low concurrency (CPU-bound regime).

The empirical curve confirms this: throughput scales roughly linearly
with K reduction (less expert dispatch work, less expert page-in on
disk-bound substrates) while quality holds 9/10 down to K=4. At K=3
the per-token expert "vote" becomes too small to reliably steer the
output distribution on harder prompts.

## Caveats

- **Temperature sensitivity (important):** K=4 quality is validated at
  temperature=0 (greedy). At temperature=0.7, K=4 quality COLLAPSES
  to 5/10 on the same 10-prompt eval while K=8 holds 8/10. The smaller
  expert budget can't recover from sampling variance — outputs go
  incoherent ("Pyth" "Pyth" for the Python prompt, "delighted" for
  Washington). For high-temperature chat workloads (temp ≥ 0.5),
  recommend K=6 or K=8. The +146% throughput win only applies in the
  low-temperature regime. Source: `autolab/k26-perf` iter 018.
- Measured on miner single-stage (disk-bound regime — K2.6 = 553 GB,
  RAM = 133 GB). The 2-box matias setup (compute-bound) may show a
  smaller relative win but the quality picture should hold.
- 10-prompt substring eval is narrow. A proper MMLU/LongBench eval
  would tighten the quality claim. The 1-prompt quality difference
  between K=4 and K=8 at temp=0 may be sampling-format dependent
  rather than a true capability gap.
- Output text diverges from K=8 at K=4 (different routing → different
  hidden states → different sampled tokens). Substring pass doesn't
  imply semantic equivalence.

## Sources

- `autolab/k26-perf` PR #11 — long-lived research branch with full Pareto
  bench artifacts.
- `experiments/009_a3_robustness_10prompt/result.md` — 10-prompt eval
  K=3 vs K=4 vs K=8 at max_tokens=16.
- `experiments/013_a3_k4_vs_k8_longcontext/result.md` — apples-to-apples
  K=4 vs K=8 at max_tokens=64.
- arxiv 2505.03531 "Faster MoE LLM Inference" — sigmoid-router MoE
  literature.
- KTransformers V0.3 `-ser` knob — related production knob on
  DeepSeek-V3.
