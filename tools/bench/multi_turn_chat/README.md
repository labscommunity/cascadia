# multi_turn_chat — bench for KV-prefix / per-session / persistent KV cache features

Drives `/v1/chat/completions` against a running tahoma worker to put real
numbers on the KV-cache feature stack from autolab iters 060/072/084. The
existing single-turn substring eval couldn't measure caches because each
prompt was unique and short-lived.

## What it measures

For each request the harness records `total_seconds`, `completion_tokens`,
and the prompt size. By running each workload with `max_tokens=1` and again
with `max_tokens=N`, the prefill cost can be isolated from decode:

  prefill_cost ≈ total_seconds(max_tokens=1)
  decode_cost  ≈ total_seconds(max_tokens=N) − total_seconds(max_tokens=1)

KV caches only affect prefill. A cache hit drops the prefill cost to roughly
the cost of prefilling just the NEW tokens (or zero, if no new prefix).

## Workloads

- `single_turn_repeat` — N identical single-turn requests. With iter 060's
  static prompt cache, requests 2..N should skip prefill of the shared
  prefix.
- `multi_turn_chat` — M conversations of N turns each. Each turn includes
  all prior messages. With iter 072's session cache (`X-Session-Id`
  header), turns 2..N prefill only the new user message rather than the
  whole history.

## Quick run (one config, against a running worker)

```
python3 bench.py \
  --url http://127.0.0.1:18000 \
  --mode both \
  --repeats 5 --conversations 3 --turns 5 \
  --use-session-id \
  --summary \
  --out results.jsonl
```

## Full A/B/C/D run (orchestrates the worker for you)

```
TAHOMA_BIN=/path/to/tahoma \
MODEL=/path/to/k26/manifest-dir \
PORT=18000 \
./run_abcd.sh /tmp/kv-cache-bench-results
```

On the miner box, this looks like:

```
source ~/openvino_2026.1.0/setupvars.sh
TAHOMA_BIN=/tmp/tahoma/target/release/tahoma \
MODEL=/tmp/k26-model-miner \
PORT=18000 \
./run_abcd.sh /tmp/kv-cache-bench-results
```

Configs:

| Config | Flags on worker                                                                                       | What we're isolating              |
| ------ | ----------------------------------------------------------------------------------------------------- | --------------------------------- |
| A      | (none)                                                                                                | Baseline — cache off              |
| B      | `--kv-prefix-cache-size 4`                                                                            | iter 060 only                     |
| C      | `--kv-prefix-cache-size 4 --session-cache-size-mb 256`                                                | iter 060 + iter 072               |
| D_cold | `--kv-prefix-cache-size 4 --session-cache-size-mb 256 --kv-prefix-cache-path /tmp/tahoma-kv-cache.bin` | first start, empty disk file      |
| D_warm | same as D_cold, but disk file populated from D_cold's shutdown                                        | iter 084 warm-restart benefit     |

The script starts a fresh worker per config (so each config sees the same
cold-start conditions), waits for `/health`, runs the bench, then SIGTERMs
the worker before moving on.

## Reading the output

Each JSONL line is one HTTP request. The fields most useful for comparison:

- `mode` — `prefill` (max_tokens=1) or `e2e` (max_tokens=N)
- `workload` — `single_turn_repeat` or `multi_turn_chat`
- `turn_index` — 0-indexed turn within a conversation
- `total_seconds` — wall-clock for this request
- `completion_tokens` — tokens the server reports it produced
- `prompt_chars` / `prompt_messages` — proxy for prefill size

For the cache wins, compare:

- Config A vs Config B `single_turn_repeat` `mode=prefill` turn 1+ — Cache
  hit on the static prompt should drop turn-1+ prefill time.
- Config A vs Config C `multi_turn_chat` `mode=prefill` turn 1+ — Session
  cache should drop turn-1+ prefill time (only new user message prefills).
- Config D_cold vs Config D_warm `single_turn_repeat` `mode=prefill` turn 0
  — Persistent cache should drop turn-0 prefill time on the warm restart.

Pass `--summary` to print an aggregate table per (workload, mode, turn).

## Limitations

- Single-stage only. iter 060/072/084 docs all note "multi-stage no-op";
  this harness assumes one worker.
- Total wall-clock includes HTTP framing. Sub-millisecond noise is
  irrelevant at K2.6 latency (>100 ms per token), but for shorter models
  you'd want server-side instrumentation instead.
- `prompt_tokens` is not reported by the server today (`Usage.prompt_tokens
  = 0` in `tahoma-api`), so prompt size is approximated with `prompt_chars`.
