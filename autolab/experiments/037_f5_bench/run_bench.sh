#!/bin/bash
# F5 bench driver: restarts worker per W, runs single-prompt bench at fixed mt.
# Usage: run_bench.sh <W> <max_tokens> <out.jsonl> <worker_log>
set -eu
W=${1:?need W}
MT=${2:?need max_tokens}
OUT=${3:?need out.jsonl}
WLOG=${4:-/tmp/tahoma-f5-w${W}-mt${MT}.log}

echo "[$(date -u +%T)] killing old worker"
ssh miner "pkill -9 tahoma 2>/dev/null; sleep 3; pkill -9 -f setupvars 2>/dev/null"
sleep 2
echo "[$(date -u +%T)] starting W=$W worker"
ssh miner "source /home/tatef/openvino_2026.1.0/setupvars.sh > /dev/null 2>&1; cd /tmp/tahoma && nohup ./target/release/tahoma worker --engine sparse-moe --rank 0 --total 1 --device CPU --model /tmp/k26-model-miner --max-tokens 256 --attention-window $W --api :8000 > $WLOG 2>&1 < /dev/null &" > /dev/null
echo "[$(date -u +%T)] waiting for ready"
ssh miner "until curl -sf -m 3 http://127.0.0.1:8000/health > /dev/null 2>&1; do sleep 5; done"
echo "[$(date -u +%T)] ready - starting bench (mt=$MT)"
ssh miner "/tmp/k26_bench_f5.sh http://127.0.0.1:8000 $OUT $MT 1"
echo "[$(date -u +%T)] bench done"
