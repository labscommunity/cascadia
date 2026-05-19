#!/usr/bin/env bash
# Orchestrate a 4-config A/B/C/D KV-cache bench on a single host.
#
# Configs:
#   A — cache OFF (baseline)
#   B — static prompt cache only            (--kv-prefix-cache-size 4)
#   C — static + per-session cache          (--kv-prefix-cache-size 4 --session-cache-size-mb 256)
#   D — static + session + persistent disk  (--kv-prefix-cache-size 4 --session-cache-size-mb 256 --kv-prefix-cache-path /tmp/tahoma-kv-cache.bin)
#
# For each config:
#   1. Start a fresh worker with the appropriate flags
#   2. Wait for /health to return 200
#   3. Run bench.py twice: single_turn_repeat + multi_turn_chat (use_session_id when applicable)
#   4. Kill the worker
#
# Config D runs the bench TWICE in sequence, killing+restarting the worker
# between runs, so we can compare cold-start (run 1) vs warm-restart (run 2).
#
# Requires:
#   - tahoma binary on $PATH or set TAHOMA_BIN
#   - manifest path set in MANIFEST (the K2.6 manifest dir)
#   - python3 on $PATH (only the stdlib is needed)
#
# Usage:
#   ./run_abcd.sh /tmp/kv-cache-bench-results [config-suffix]
#
# Output goes to <results_dir>/<config>.jsonl  +  <results_dir>/<config>_worker.log

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <results_dir> [config-suffix]" >&2
  exit 2
fi
results_dir=$1
suffix=${2:-}
mkdir -p "$results_dir"

TAHOMA_BIN=${TAHOMA_BIN:-tahoma}
MODEL=${MODEL:?MODEL=/path/to/k26/manifest-dir is required (e.g. /tmp/k26-model-miner)}
PORT=${PORT:-18000}
URL="http://127.0.0.1:${PORT}"
PERSIST_PATH=${PERSIST_PATH:-/tmp/tahoma-kv-cache.bin}
DEVICE=${DEVICE:-CPU}
TOP_K_OVERRIDE=${TOP_K_OVERRIDE:-6}
COMMON_ARGS=(
  worker
  --engine sparse-moe
  --model "$MODEL"
  --api ":${PORT}"
  --rank 0 --total 1
  --device "$DEVICE"
  --top-k-override "$TOP_K_OVERRIDE"
)
# A cushion of 300s before we declare the worker dead at startup; K2.6 cold-
# load is ~5min on miner.
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-360}

bench=$(dirname "$(readlink -f "$0")")/bench.py

start_worker() {
  local label=$1
  shift
  local logf="${results_dir}/${label}${suffix}_worker.log"
  echo "[$(date -Iseconds)] starting worker for ${label} → $logf" >&2
  "${TAHOMA_BIN}" "${COMMON_ARGS[@]}" "$@" >"$logf" 2>&1 &
  echo $!
}

wait_for_health() {
  local pid=$1
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "worker pid $pid died before health came up" >&2
      return 1
    fi
    if curl -fsS "$URL/health" >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] health OK (pid $pid)" >&2
      return 0
    fi
    sleep 2
  done
  echo "health timeout after ${HEALTH_TIMEOUT}s" >&2
  return 1
}

stop_worker() {
  local pid=$1
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
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
  local out="${results_dir}/${label}${suffix}.jsonl"
  local extra=()
  if [[ "$use_session" == "1" ]]; then
    extra+=(--use-session-id)
  fi
  echo "[$(date -Iseconds)] bench ${label} → $out" >&2
  python3 "$bench" \
    --url "$URL" \
    --mode both \
    --repeats 5 \
    --conversations 3 \
    --turns 5 \
    --max-tokens-decode 32 \
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

# === B: static prompt cache only ===
pid=$(start_worker B --kv-prefix-cache-size 4)
wait_for_health "$pid"
run_bench B 0
stop_worker "$pid"

# === C: static + session ===
pid=$(start_worker C --kv-prefix-cache-size 4 --session-cache-size-mb 256)
wait_for_health "$pid"
run_bench C 1
stop_worker "$pid"

# === D: static + session + persistent (two-run: cold then warm) ===
rm -f "$PERSIST_PATH"
pid=$(start_worker D_cold --kv-prefix-cache-size 4 --session-cache-size-mb 256 --kv-prefix-cache-path "$PERSIST_PATH")
wait_for_health "$pid"
run_bench D_cold 1
stop_worker "$pid"

pid=$(start_worker D_warm --kv-prefix-cache-size 4 --session-cache-size-mb 256 --kv-prefix-cache-path "$PERSIST_PATH")
wait_for_health "$pid"
run_bench D_warm 1
stop_worker "$pid"

trap - INT TERM EXIT
echo "[$(date -Iseconds)] all configs complete → $results_dir" >&2
