#!/bin/bash
# Wait until Config A's bench has its aggregate line, then run B, C, D, E
# sequentially. Designed to be launched with nohup once Config A's bench
# is in flight.
set -u

DIR=$(cd "$(dirname "$0")" && pwd)

# Wait for Config A aggregate to appear (Config A's bench writes to
# /tmp/k26-bench-073-A.jsonl).
echo "$(date) waiting for Config A aggregate"
until grep -q 'aggregate' /tmp/k26-bench-073-A.jsonl 2>/dev/null; do
    sleep 30
done
echo "$(date) Config A aggregate found"
tail -1 /tmp/k26-bench-073-A.jsonl
sudo -n pkill -9 -f 'target/release/tahoma' 2>/dev/null || true
sleep 3

for CFG in B C D E; do
    echo "=== iter 073 config $CFG === $(date)"
    bash "$DIR/launch_worker.sh" "$CFG"
    # Wait for runner ready
    until grep -q 'runner ready' /tmp/tahoma-073-$CFG.log 2>/dev/null; do
        sleep 10
    done
    echo "config $CFG: worker ready, running bench at $(date)"
    bash "$DIR/k26_bench_073.sh" http://127.0.0.1:8000 0.0 /tmp/k26-bench-073-$CFG.jsonl 64 \
        > /tmp/bench-073-$CFG-progress.log 2>&1 || true
    echo "config $CFG: bench done at $(date)"
    tail -1 /tmp/k26-bench-073-$CFG.jsonl
    sudo -n pkill -9 -f 'target/release/tahoma' 2>/dev/null || true
    sleep 5
done

echo "=== all configs done $(date) ==="
