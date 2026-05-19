#!/bin/bash
# Config A — iter 044 baseline (no prefetch)
# Defaults: --prefetch-backend off
set -u
cd /tmp/tahoma
sudo -n nohup bash -c "ulimit -l unlimited && \
  source /home/tatef/openvino_2026.1.0/setupvars.sh && \
  ./target/release/tahoma worker --engine sparse-moe --rank 0 --total 1 \
    --device CPU --model /tmp/k26-model-miner \
    --top-k-override 6 --prompt-lookup 3 --spec-k 4 --max-tokens 64 \
    --api :8000" > /tmp/tahoma-101-A.log 2>&1 &
echo worker_pid=$!
