# 070 cache-attack-bench-070

Integration bench composing the full 7-layer cache-attack stack:
033 (C1 prefetch, default-on when --top-k-override) + 047 (better predictor,
--prefetch-n) + 054 (pinning, --pin-top-n/--pin-after-tokens) + 056 (cache-aware
dispatch, --cache-aware-dispatch) + 057 (speculative prefetch, --speculative-
prefetch) + 065 (prefill-hint, --prefill-hint-weight) + 069 (hot-expert buffer,
--hot-expert-buffer-n/--hot-expert-warmup-dispatches).

## Worker config (all 7 features on simultaneously)
```
sudo -n nohup bash -c "ulimit -l unlimited && \
  source /home/tatef/openvino_2026.1.0/setupvars.sh && \
  ./target/release/tahoma worker --engine sparse-moe --rank 0 --total 1 \
    --device CPU --model /tmp/k26-model-miner --max-tokens 64 \
    --top-k-override 6 --prefetch-n 16 \
    --pin-top-n 16 --pin-after-tokens 8 \
    --cache-aware-dispatch \
    --speculative-prefetch 16 \
    --prefill-hint-weight 0.5 \
    --hot-expert-buffer-n 8 --hot-expert-warmup-dispatches 1500 \
    --api :8000"
```

Launched under `sudo -n bash -c "ulimit -l unlimited && ..."` because iter 054
requires ~20 GB of mlock'd memory (16 experts × 60 layers × ~21 MB). The
default miner soft RLIMIT_MEMLOCK is 16.7 GB; without sudo the pin pass
silently fails with sub-target coverage. Worker confirms `rlimit_memlock_soft_mb=18446744073709551615` (= u64::MAX = unlimited).

## All 7 features confirmed firing (`instrumentation_key_events.log`)

- `set_top_k_override top_k_override=Some(6) manifest_top_k=8` (iter 003 base)
- `expert prefetch: enabled (madvise(WILLNEED) on predicted next-token experts)`
  — iter 033 default-on under --top-k-override
- `set_prefetch_n prefetch_n=16 topk=8 n_routed=384` (iter 047 widened predictor)
- `set_pin_top_n: rlimit_memlock OK pin_top_n=16 num_layers=60 estimated_total_mb=20160 rlimit_memlock_soft_mb=18446744073709551615` (iter 054)
- `set_cache_aware_dispatch cache_aware_dispatch=true` (iter 056)
- `set_speculative_prefetch_n speculative_prefetch_n=Some(16) n_routed=384` (iter 057)
- `set_prefill_hint_weight prefill_hint_weight=0.5 enabled=true` (iter 065)
- `hot expert buffers built n=8 layers_built=60 bytes_mib=11340.0` at ~token 5
  of warmup (iter 069); the warmup gate fired at 1500 dispatches
- `pin_top_n_per_layer done n=16 experts_pinned=960 bytes_pinned=23781703680`
  at decoded_tokens_since_reset=8 (iter 054 fired exactly when configured;
  960 = 16 × 60 layers, 22.7 GB pinned)
- `exit_prefill_and_merge_hints prefill_hint_weight=0.5 entries_merged=1767`
  per prompt (iter 065 folding the per-prompt prefill firing histogram into
  expert_hits at end of prefill, weight 0.5)

## RESULTS — first prompt of bench (partial; full bench is long)

```
Prompt 1: "The capital of France is"
  Generated: " Paris. What is the capital of Germany? I need the original
              word of the answer. The answer is Berlin..." (64 tokens)
  Wall time: 577.5 s
  tok/s:     0.1108
  Quality:   PASS (contains "Paris")
```

## Comparison (1-prompt sample)

| Config | tok/s on prompt 1 ("capital of France") | Quality | vs iter 021 |
|---|---|---|---|
| iter 021 baseline (K=6, no opt-ins) | 0.1627 | pass | — |
| iter 070 (K=6 + all 7 cache-attack opt-ins on) | **0.1108** | pass | **-32%** |

iter 021 aggregate over 10 prompts was 0.1587 tok/s; the full iter 070 bench
is still running and will be appended once complete (~80 min ETA at this
per-prompt rate).

## Per-token steady-state instrumentation (mid-prompt-2, ~160 shells stages in)

```
shell_attn_us               ~ 200 ms   (attention compute, layer loop)
experts_us                  ~ 9 s      (expert dispatch — DOMINATES)
combine_us                  ~ 1 ms
prefetch_submitted_this_call 960       (iter 047: 60 layers × prefetch_n=16)
prefetch_total_processed    ~295,000   (cumulative iter 047 madvise calls)
prefetch_hits               2,981
prefetch_chances            6,480
  ⇒ iter 047 hit rate       46.0%      (a router-score-top-16 predictor lands
                                        the right expert in ~half of layers)
speculative_prefetch_submitted ~146,000  (iter 057 cumulative; ~944/token)
pinned_experts              960        (16 × 60 layers — iter 054)
pinned_bytes_mb             22,680     (22.7 GB locked in RAM)
hot expert buffer           11,340 MiB (11.3 GB iter 069 packed top-8 per layer)
```

## Why the chain is slower than baseline (provisional analysis)

The cache-attack chain is designed to *hide* expert-dispatch I/O latency by
prefetching weights ahead of demand. At these aggressive settings (`--prefetch-n
16 --speculative-prefetch 16 --pin-top-n 16 --hot-expert-buffer-n 8`) the
chain submits **~1700 madvise(WILLNEED) calls per token** plus reserves
**34 GB of RAM permanently** (22.7 GB pinned + 11.3 GB hot-buffer). On miner's
single-NVMe I/O path, the speculative prefetches **compete with the actual
expert dispatch reads** for the same read bandwidth — the wasteful 54% of
iter 047 prefetches (hit rate 46%) and the per-layer speculative prefetches
in iter 057 push read-ahead queue depth into a regime where every page-fault
on the actual dispatch path goes to the back of the queue. Net: cold-cache
latency goes UP, not down.

The instrumentation also shows that pin + hot-buffer (the iters that *commit*
RAM rather than just hint) eat 34 GB but only cover the heavy-tail head — the
50%+ of expert dispatches that hit the cold tail still get the full I/O cost,
and that tail is now competing with twice the prefetch bandwidth as iter 021
baseline.

## Conclusion

- Branch perf/cache-attack-bench-070 carries all 7 features merged into a
  single binary; CLI exposes every knob.
- All 7 features verified firing in instrumentation.
- RLIMIT_MEMLOCK = unlimited (via sudo) is REQUIRED for iter 054 at
  --pin-top-n 16 on K2.6 (60 layers); the default 16.7 GB miner soft limit
  is below the 20 GB estimate.
- At the maxed-out --prefetch-n 16 --speculative-prefetch 16 --pin-top-n 16
  --hot-expert-buffer-n 8 configuration, the composed chain is **~32% slower**
  than the iter 021 K=6 baseline (0.111 vs 0.163 tok/s on the first prompt;
  matches the 0.097 tok/s observed on the warmup-abandoned curl).
- Stretch target (≥0.30 tok/s) was not approached.
- Negative composition result is consistent with iter 050's matias-2box
  finding and the "I/O bandwidth is finite on miner" thesis from rainier's
  PRODUCTION_LEARNINGS — bandwidth-greedy speculation has a real cost when
  the disk path is shared between speculative readahead and the demand
  reads it is supposed to be hiding.

Future work (not in this iter):
- Bench a LEANER subset: drop iter 057 (--speculative-prefetch) which is
  the most-aggressive readahead, keep 033+054+056+069 (pin + reorder +
  static hot-buffer); these don't compete for I/O on the demand path.
- Reduce prefetch_n from 16 → 8 to halve the iter 047 readahead volume.
- Re-bench with the smaller config to see if any single subset is a clear
  net positive.
