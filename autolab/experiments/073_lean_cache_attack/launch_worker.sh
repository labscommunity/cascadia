#!/bin/bash
# Launch a tahoma worker on the miner under sudo+ulimit -l unlimited
# with the lean-cache-attack feature flags for one of configs A..E.
#
# Usage: launch_worker.sh <config_letter>
#
# Side effects:
# - kills any existing /tmp/tahoma/target/release/tahoma workers
# - launches a fresh worker in the background with logs at
#   /tmp/tahoma-073-<config>.log
# - prints the pid of the launched bash wrapper to stdout
#
# Caller is responsible for polling /health and / waiting for warmup
# before benching.
set -euo pipefail

CFG=${1:-A}
TAHOMA=/tmp/tahoma/target/release/tahoma
MODEL=/tmp/k26-model-miner
OV_SETUP=/home/tatef/openvino_2026.1.0/setupvars.sh
LOG=/tmp/tahoma-073-${CFG}.log

# Kill any prior worker
sudo -n pkill -9 -f 'target/release/tahoma' 2>/dev/null || true
sleep 2

# Pick config flags. Common to all: --top-k-override 6 --pin-top-n 16
# --pin-after-tokens 8.
case "$CFG" in
    A)
        # Pinning only (iter 054)
        EXTRA=""
        ;;
    B)
        # Pin + cache-aware dispatch
        EXTRA="--cache-aware-dispatch"
        ;;
    C)
        # Pin + dispatch + hot-buffer
        EXTRA="--cache-aware-dispatch --hot-expert-buffer-n 8 --hot-expert-warmup-dispatches 1500"
        ;;
    D)
        # Pin + dispatch + hot-buffer + light prefetch (n=8, half of iter 070)
        EXTRA="--cache-aware-dispatch --hot-expert-buffer-n 8 --hot-expert-warmup-dispatches 1500 --prefetch-n 8"
        ;;
    E)
        # Pin + dispatch + hot-buffer + prefill-hint
        EXTRA="--cache-aware-dispatch --hot-expert-buffer-n 8 --hot-expert-warmup-dispatches 1500 --prefill-hint-weight 0.5"
        ;;
    *)
        echo "unknown config: $CFG" >&2
        exit 1
        ;;
esac

# IMPORTANT: ulimit -l must be set inside the sudo bash subshell because
# rlimits are per-process and don't propagate from a non-sudo shell.
sudo -n nohup bash -c "ulimit -l unlimited && \
  source $OV_SETUP && \
  $TAHOMA worker --engine sparse-moe --rank 0 --total 1 \
    --device CPU --model $MODEL --max-tokens 64 \
    --top-k-override 6 \
    --pin-top-n 16 --pin-after-tokens 8 \
    $EXTRA \
    --api :8000" > "$LOG" 2>&1 &

PID=$!
echo "launched config=$CFG pid=$PID log=$LOG"
echo "extra-flags: $EXTRA"
