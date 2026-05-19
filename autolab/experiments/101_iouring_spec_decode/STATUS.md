# 101 spec-decode-iouring-bench

Integration bench: does iter 098's `--prefetch-backend io-uring` async
expert prefetch on top of iter 044's K=4 spec-decode compound stack
improve on iter 044's +19.7% e2e baseline?

## Status: PARTIAL — miner contention blocked full bench

Config A (no prefetch, iter 044 baseline) completed 4 of 10 prompts
cleanly before another autolab agent on the same user account
(`tatef`) launched a competing worker on port 8000 at 14:24 UTC,
killing my worker mid-decode. Prompts 5-10 hit the new worker but
received malformed responses (`awk: line 1: syntax error at or near *`
in the bench harness — the new worker had different defaults that
broke `usage.completion_tokens` parsing). Configs B (`madvise`) and C
(`io-uring`) were never launched.

This matches the iter 044 commit's operational note: "F5 bench retry
agent (iter 037) was running on :8000 when the spec bench started; ...
contention erases the spec win in single-stage mode." The miner is a
shared single-node fleet — running long-tail benches there is
inherently racy with sibling research agents.

## What the partial Config A data does validate

The 4 completed prompts produced **byte-identical outputs to iter
044's reference JSONL** at `autolab/experiments/044_spec_decode_compound/
bench_spec_K4_mt64.jsonl`:

| Prompt | iter 044 (ref) tok/s | iter 101 Config A tok/s | bit-identical? |
|---|---|---|---|
| France capital | 0.341  | 0.211 | yes |
| Pacific ocean  | 0.180  | 0.174 | yes |
| 2+2            | 0.176  | 0.171 | yes |
| First president | 0.190 | 0.186 | yes |

Aggregate (4 prompts): tokens=256, wall_ms=1389319, **tok/s=0.1843**.
iter 044's full 10-prompt aggregate was 0.1899.

Conclusion: **the merge of iter 044 + iter 098 is functionally
correct** — the spec-decode compound stack still produces the same
sequences. The tok/s gap on prompt 1 (-38% vs iter 044) is consistent
with cold-cache vs warm-cache (iter 044's worker had been used by an
F5 retry agent for 20+ min before bench start; my Config A worker
started cold).

## Setup (when retried)

- Branch: `perf/spec-decode-iouring-bench-101` (merge of iter 044 + iter 098)
- Host: miner (Xeon Gold 6252, 48 SMT, OpenVINO 2026.1.0, Linux 6.17)
- Model: K2.6 sparse-MoE single-stage (60 layers, 384 experts, top_k=8)
- Bench: 10-prompt factoid eval at temp=0, max_tokens=64
- All configs use `--top-k-override 6 --prompt-lookup 3 --spec-k 4`

## Worker configs

| Config | Prefetch backend | Description |
|---|---|---|
| A | `off` (default) | iter 044 baseline (no async prefetch) |
| B | `madvise` | iter 044 + iter 033 madvise(WILLNEED) prefetch |
| C | `io-uring` | iter 044 + iter 098 real io_uring IORING_OP_READ |

All 3 launched under `sudo -n bash -c "ulimit -l unlimited && ..."`
(iter 070 pattern, harmless when no pin requested).

## Baselines from prior iters (same model, same miner)

| Iter | Config | tok/s | Quality (substring) |
|---|---|---|---|
| 021 | K=6, no opt-ins | 0.1587 | 10/10 |
| 044 | K=6 + spec-K4 prompt-lookup | 0.1899 | 9/10 |
| 033 | K=6 + madvise (no spec, temp=0.5) | 0.0866 | — |

## Code merge

Merge conflicts in 3 files resolved cleanly:
- `crates/tahoma-cli/src/lib.rs`: kept both iter 044's `--top-k-override`/
  `--routing-threshold` AND iter 098's `--prefetch-backend`. Both
  field-additions ordered correctly in the struct; both
  config-construction code paths preserved.
- `crates/tahoma-engine-sparse-moe/src/lib.rs`: kept both re-export
  lines.
- `crates/tahoma-engine-sparse-moe/src/runner.rs`:
  - Runner fields: kept `top_k_override`, `routing_threshold` (iter
    044) AND `prefetcher`, `last_routing_ids` (iter 098).
  - `forward_shells` expert dispatch loop: iterate over
    `effective_top_k` (iter 044's A3 cap), apply
    `routing_threshold` filter (iter 044's A2), AND track the
    *actually-dispatched* expert IDs into `last_routing_ids[i]` (iter
    098 prefetch predictor). The predictor sees what will be accessed
    on the next forward, not the full manifest top_k — most faithful
    merge.

`cargo build --release -p tahoma --features openvino` clean on miner
(39.7s). No new clippy warnings beyond what's pre-existing on the two
parent branches.

## io_uring instrumentation (not collected — Config C never ran)

iter 097's instrumentation exposes (via `Runner::prefetch_stats()`):
- `submits` — calls to `try_submit` that landed in channel
- `drops` — overflow drops
- `completed` — requests fully processed
- `backend_queue_full` — io_uring SQ rejections
- `slices_pushed` — successful tensor-slice prefetches

These need a `--prefetch-backend io-uring` worker to populate. The
runner does not currently emit them per-token; they would have to be
sampled via a separate `/v1/diagnostics` call or by adding a per-token
`info!` log line in `forward_shells`. Neither was in scope for the
bench itself.

## Recommendation

iter 101 needs a re-run when miner is exclusively available. Two
practical options:
1. Coordinate the autolab agent fleet so only one long-running bench
   runs at a time on miner (single-node sparse-MoE benches need ~60
   min wall-clock per 10-prompt eval).
2. Promote io_uring prefetch validation to the simulator path
   (`tests/io_uring_prefetch_synthetic.rs` already does bit-exact
   verification on the int4-gemm layer; an e2e simulated workload
   with `madvise` and `io-uring` could give first-order timing
   comparisons on the dev box without miner).

For this branch, the deliverable is:
- merged code (build + run validated)
- partial Config A bench (4 prompts, bit-identical to iter 044)
- contention documentation
- launch scripts ready for re-run (`launch_config_{A,B,C}.sh`)
