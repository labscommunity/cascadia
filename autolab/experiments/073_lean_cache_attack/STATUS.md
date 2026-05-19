# 073 lean cache-attack subset bench

Follow-up to iter 070, which composed the full 7-feature cache-attack
stack at maxed-out settings (`--prefetch-n 16 --pin-top-n 16
--speculative-prefetch 16 --hot-expert-buffer-n 8 --prefill-hint-weight
0.5`) and measured **-32% vs the iter 021 K=6 baseline** on miner
(0.1108 vs 0.1627 tok/s on prompt 1). iter 070's analysis: the chain
submitted ~1700 madvise(WILLNEED)/token and held 34 GB of RAM, and on
the single-NVMe miner the speculative prefetches competed with the
actual expert-dispatch demand reads — net cold-cache latency went up.

This iter measured a series of **lean subsets** to see whether any
configuration beats the K=6 baseline on the single-NVMe miner substrate.

## Bench harness

10-prompt × `max_tokens=64` × `temp=0` factual eval (identical to iter
021 and iter 070). curl timeout 1800 s to absorb the cold-cache prompts.
Aggregate `tok/s` = `(sum completion_tokens × 1000) / sum wall_ms`.

## Worker config — shared common knobs
All configs run on miner under `sudo + ulimit -l unlimited` (iter 054
needs ~24 GB of mlock at this `--pin-top-n`):

```
sudo -n nohup bash -c "ulimit -l unlimited && \
  source /home/tatef/openvino_2026.1.0/setupvars.sh && \
  ./target/release/tahoma worker --engine sparse-moe --rank 0 --total 1 \
    --device CPU --model /tmp/k26-model-miner --max-tokens 64 \
    --top-k-override 6 \
    --pin-top-n 16 --pin-after-tokens 8 \
    <CONFIG-SPECIFIC-FLAGS> \
    --api :8000"
```

`--top-k-override 6` enables iter 033 prefetch by default (`prefetch_n`
defaults to manifest `top_k=8` when iter 033 is on — the conservative
"same-as-last-token" predictor, half of iter 070's `--prefetch-n 16`).
`--pin-top-n 16 --pin-after-tokens 8` is iter 054 pinning fired right
after an 8-token warmup window.

## Config-specific flags

| Config | Extra flags |
|---|---|
| A | (none) — pin only (iter 054 + iter 033 default prefetch) |
| B | `--cache-aware-dispatch` |
| C | `--cache-aware-dispatch --hot-expert-buffer-n 8 --hot-expert-warmup-dispatches 1500` |
| D | C + `--prefetch-n 8` (intended "lower than iter 070's 16" but 8 is also TOPK so equiv to C) |
| E | C + `--prefill-hint-weight 0.5` |

## Results

| Config | tok/s   | quality | vs iter 021 | notes |
|---|---|---|---|---|
| iter 021 baseline (K=6, no opt-ins)               | 0.1587 | 10/10 | —      | reference |
| iter 070 full 7-stack (prompt 1 only, abandoned)  | 0.1108 | pass  | -30.2% | the regression we want to avoid |
| **073-A pin only**                                | **0.1492** | 10/10 | **-6.0%** | **best of the lean subset — still slightly negative** |
| 073-B pin + cache-aware dispatch                  | 0.1477 | 10/10 | -6.9%  | reorder neutral, same hit-rate 43.5% as A |
| 073-C pin + dispatch + hot-buffer                 | 0.1386 | 10/10 | -12.7% | hot-buffer 11.3 GB + pin 22.7 GB = 34 GB held; cold-tail page-cache budget shrinks |
| 073-D pin + dispatch + hot-buffer + light prefetch | 0.1368 | 10/10 | -13.8% | `--prefetch-n 8` collapses to default TOPK; A/A check vs C |
| 073-E pin + dispatch + hot-buffer + prefill-hint  | 0.1379 | 10/10 | -13.1% | prefill-hint neutral on top of hot-buffer cost |

All five configs **pass 10/10 quality** — no correctness regressions.
But **no config beats the iter 021 baseline** on this single-NVMe substrate.

## Per-feature attribution

Reading deltas between adjacent configs isolates each feature's impact:

| Feature added | Configs | Δ vs prior | Verdict on miner |
|---|---|---|---|
| iter 054 pinning + iter 033 default prefetch  | baseline → A | -6.0%  | slightly negative |
| iter 056 cache-aware dispatch reorder         | A → B        | -1.0%  | neutral (within noise) |
| iter 069 hot-expert buffer (11.3 GB)          | B → C        | -6.2%  | **clear negative** (RAM crowds page cache) |
| `--prefetch-n 8` (==TOPK == iter 033 default) | C → D        | -1.3%  | no-op (spec flaw — flag below default) |
| iter 065 prefill-hint W=0.5                   | C → E        | -0.5%  | neutral (within noise) |

## Recommendation

**On miner (single-NVMe, 24-thread Xeon Gold 6252, 133 GB RAM, 553 GB
model): do NOT compose the cache-attack chain.** The best lean config
(Config A — pin + default prefetch, 0.1492 tok/s) is still -6.0% vs
the iter 021 K=6 plain baseline (0.1587 tok/s). Two specific dropouts:

- **Drop iter 069 hot-expert buffer.** Costs 11.3 GB of OWNED memory
  with no observable benefit — Configs C/D/E all land 6-7% below A/B.
  The hot-buffer's intended L3-residency win needs L3 ≥ 50 MiB per
  socket and ≤ 25% L3-share-per-thread. Xeon Gold 6252 has 27.5 MiB
  L3 / 24 threads = no fit. Wait for Sapphire Rapids HBM nodes or
  AI-PC iGPU sharing patterns.
- **Drop iter 056 cache-aware dispatch reorder.** Neutral effect
  (A → B = -1.0%, within run-to-run noise on a 10-prompt eval).
  Same reasoning: reorder hopes for L3-warm hot-experts, but no expert
  fits in L3.

**What MIGHT still be worth keeping** (slightly negative but maybe
positive in higher-RAM / lower-disk-pressure regimes):
- iter 054 pinning. Costs 22.7 GB RAM but at least covers the heavy-
  tail head with zero re-paging. Config A's -6% is consistent with
  "pinning shrinks page-cache budget by 22.7 GB → slightly more
  cold-tail misses". On miner this is a wash; on a 256+ GB AI PC it
  could turn net-positive.

## What this iter ruled out

- **The hypothesis "drop iter 057 speculative prefetch and the rest of
  the chain will turn positive"** (the iter 070 recommendation). It
  does turn positive vs iter 070's -32%, but it doesn't reach baseline.
  The chain is wrong on this hardware, not just the most-aggressive
  knob.
- **The hypothesis "iter 047 default prefetch + iter 054 pinning is
  the safe minimal chain"**. Config A (which IS that chain) is -6%, not
  net-positive. The premise that "pinning protects the hot-set, prefetch
  hides cold-tail page-in" is empirically false on miner — the cold-tail
  is large enough that even the pin'd 22.7 GB + the same-as-last-token
  prefetch can't hide it.

## What's next

- **Different prefetch_n.** Config D should be re-run with `--prefetch-n
  12` or `16` to actually test the iter 047 widened predictor (the
  spec's `--prefetch-n 8` is a no-op vs the iter 033 default). But
  iter 070 already tested `--prefetch-n 16` in the full stack and it
  was -32%; testing the lean variant of just pin + prefetch-n=16
  would be informative.
- **Smaller pin_top_n.** All configs here use `--pin-top-n 16` (22.7
  GB). Testing `--pin-top-n 8` (~11.4 GB) or `--pin-top-n 4` (~5.7 GB)
  would tell us if there's a sweet spot where pinning's protection
  exceeds the page-cache budget cost. Worth a follow-up iter.
- **Pipeline parallelism (PR #9 / #10 merged).** The mission line is
  "model that doesn't fit on one Intel laptop run across two-three".
  On 2 boxes the per-box active footprint drops ~50%, which should
  pull the per-box working set below the disk-bound regime entirely.
  The cache-attack chain might land net-positive there, where it
  cannot on single-box miner.

## Files

- `STATUS.md` — this document
- `k26_bench_073.sh` — 10-prompt × mt=64 × temp=0 harness (curl -m 1800)
- `launch_worker.sh` — per-config worker launcher with sudo + ulimit
- `run_configs_BCDE.sh` — sequential runner (waits for prior aggregate,
  kills worker, launches next)
- `run_all_configs.sh` — same but A..E
- `compare.sh` — aggregate markdown table generator
- `bench_073_{A,B,C,D,E}.jsonl` — raw per-prompt + aggregate per config
- `instrumentation_073_{A,B,C,D,E}.log` — feature-set events + per-task
  done events per config
