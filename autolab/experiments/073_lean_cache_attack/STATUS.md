# 073 lean cache-attack subset bench

Follow-up to iter 070, which composed the full 7-feature cache-attack
stack at maxed-out settings (`--prefetch-n 16 --pin-top-n 16
--speculative-prefetch 16 --hot-expert-buffer-n 8 --prefill-hint-weight
0.5`) and measured **-32% vs the iter 021 K=6 baseline** on miner
(0.1108 vs 0.1627 tok/s on prompt 1). iter 070's analysis: the chain
submitted ~1700 madvise(WILLNEED)/token and held 34 GB of RAM, and on
the single-NVMe miner the speculative prefetches competed with the
actual expert-dispatch demand reads — net cold-cache latency went up.

This iter measures a series of **lean subsets** to see whether any
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
| D | C + `--prefetch-n 8` (iter 070 used `--prefetch-n 16`; halved here) |
| E | C + `--prefill-hint-weight 0.5` |

## Results (placeholder)

| Config | tok/s | quality | vs iter 021 | notes |
|---|---|---|---|---|
| iter 021 baseline (K=6, no opt-ins)               | 0.1587 | 10/10 | —      | reference |
| iter 070 full 7-stack (prompt 1 only, abandoned)  | 0.1108 | pass  | -30.1% | the regression we want to avoid |
| 073-A pin only                                    | 0.1492 | 10/10 | -6.0%  | hit-rate 43.5%, pinned 22.7 GB |
| 073-B pin + cache-aware dispatch                  | 0.1477 | 10/10 | -6.9%  | reorder neutral, same hit-rate 43.5% as A |
| 073-C pin + dispatch + hot-buffer                 | 0.1386 | 10/10 | -12.7% | hot-buffer 11.3 GB + pin 22.7 GB = 34 GB held; cold-tail page-cache budget shrinks |
| 073-D pin + dispatch + hot-buffer + light prefetch | 0.1368 | 10/10 | -13.8% | --prefetch-n 8 == default (TOPK), so effectively same as C; -1.3% noise |
| 073-E pin + dispatch + hot-buffer + prefill-hint  | TBD    | TBD   | TBD    | |

## Conclusion

(filled in once benches complete)
