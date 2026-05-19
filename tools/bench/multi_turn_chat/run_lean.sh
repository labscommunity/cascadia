#!/usr/bin/env bash
# Lean A/C/D bench — drops the redundant Config B (subsumed by C) and the
# e2e batch (caches only affect prefill). Tuned for substrates where
# per-token cost is ~seconds (e.g. K2.6 sparse-MoE single-stage on a 24-core
# Xeon: ~6 s per token).
#
# Configs:
#   A — cache OFF
#   C — static prompt + per-session
#   D_cold — above + persistent path (empty disk file at start)
#   D_warm — above with disk file populated from D_cold's shutdown
#
# Each config: 3 single-turn repeats + 1 multi-turn conversation × 3 turns
#   = 6 requests × prefill mode only (max_tokens=1)
#
# Expected wall-clock with K2.6 on miner (~6 s / token, ~30-50 token prompts):
#   A:       6 × ~5 min cold = ~30 min
#   C:       first request cold ~5 min, rest cache-hit ~10-20s each = ~7 min
#   D_cold:  same as C ~7 min
#   D_warm:  ALL cache-hit (persistence) = ~2 min
#   + 4 worker boots × ~2-5 min = ~15 min
#   Total: ~60-75 min
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <results_dir>" >&2
  exit 2
fi
results_dir=$1
mkdir -p "$results_dir"

TAHOMA_BIN=${TAHOMA_BIN:-tahoma}
MODEL=${MODEL:?MODEL=/path/to/k26/manifest-dir is required}
PORT=${PORT:-18000}
URL="http://127.0.0.1:${PORT}"
PERSIST_PATH=${PERSIST_PATH:-/tmp/tahoma-kv-cache.bin}
DEVICE=${DEVICE:-CPU}
TOP_K_OVERRIDE=${TOP_K_OVERRIDE:-6}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-360}

COMMON_ARGS=(
  worker --engine sparse-moe --model "$MODEL" --api ":${PORT}"
  --rank 0 --total 1 --device "$DEVICE" --top-k-override "$TOP_K_OVERRIDE"
)

bench=$(dirname "$(readlink -f "$0")")/bench.py

start_worker() {
  local label=$1
  shift
  local logf="${results_dir}/${label}_worker.log"
  echo "[$(date -Iseconds)] starting worker for ${label} → $logf" >&2
  "${TAHOMA_BIN}" "${COMMON_ARGS[@]}" "$@" >"$logf" 2>&1 &
  echo $!
}

wait_for_health() {
  local pid=$1
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$(date -Iseconds)] worker pid $pid died before health came up" >&2
      return 1
    fi
    if curl -fsS "$URL/health" >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] health OK (pid $pid)" >&2
      return 0
    fi
    sleep 2
  done
  echo "[$(date -Iseconds)] health timeout after ${HEALTH_TIMEOUT}s" >&2
  return 1
}

stop_worker() {
  local pid=$1
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  fi
}

run_bench() {
  local label=$1
  local use_session=$2
  local out="${results_dir}/${label}.jsonl"
  local extra=()
  if [[ "$use_session" == "1" ]]; then extra+=(--use-session-id); fi
  echo "[$(date -Iseconds)] bench ${label} → $out" >&2
  python3 "$bench" \
    --url "$URL" \
    --mode both \
    --skip-e2e \
    --repeats 3 \
    --conversations 1 \
    --turns 3 \
    "${extra[@]}" \
    --summary \
    --out "$out"
}

trap 'echo "[$(date -Iseconds)] aborting"; jobs -p | xargs -r kill 2>/dev/null || true' INT TERM EXIT

# === A: baseline (cache off) ===
pid=$(start_worker A)
wait_for_health "$pid"
run_bench A 0
stop_worker "$pid"

# === C: static prompt + per-session cache ===
pid=$(start_worker C --kv-prefix-cache-size 4 --session-cache-size-mb 256)
wait_for_health "$pid"
run_bench C 1
stop_worker "$pid"

# === D_cold: above + persistent disk (empty file at start) ===
rm -f "$PERSIST_PATH"
pid=$(start_worker D_cold --kv-prefix-cache-size 4 --session-cache-size-mb 256 --kv-prefix-cache-path "$PERSIST_PATH")
wait_for_health "$pid"
run_bench D_cold 1
echo "[$(date -Iseconds)] checking persist file before shutdown:" >&2
ls -la "$PERSIST_PATH" 2>&1 >&2 || true
stop_worker "$pid"
echo "[$(date -Iseconds)] persist file after shutdown:" >&2
ls -la "$PERSIST_PATH" 2>&1 >&2 || true

# === D_warm: restart with same disk file present ===
pid=$(start_worker D_warm --kv-prefix-cache-size 4 --session-cache-size-mb 256 --kv-prefix-cache-path "$PERSIST_PATH")
wait_for_health "$pid"
run_bench D_warm 1
stop_worker "$pid"

trap - INT TERM EXIT
echo "[$(date -Iseconds)] all configs complete → $results_dir" >&2
