#!/bin/bash
# Run the lean-cache-attack bench across configs A..E sequentially.
# Each config: launch worker, wait for runner ready, run 10-prompt bench
# at mt=64 temp=0, capture instrumentation snapshot, kill worker, loop.
set -u

DIR=$(cd "$(dirname "$0")" && pwd)
CONFIGS=${1:-"A B C D E"}

for CFG in $CONFIGS; do
    echo "=== iter 073 config $CFG ==="
    bash "$DIR/launch_worker.sh" "$CFG"
    # Wait for runner ready
    until grep -q 'runner ready' /tmp/tahoma-073-$CFG.log 2>/dev/null; do
        sleep 5
    done
    echo "config $CFG: worker ready, running bench"
    bash "$DIR/k26_bench_073.sh" http://127.0.0.1:8000 0.0 /tmp/k26-bench-073-$CFG.jsonl 64 \
        > /tmp/bench-073-$CFG-progress.log 2>&1 || true
    echo "config $CFG: bench done"
    tail -1 /tmp/k26-bench-073-$CFG.jsonl
    # Snapshot last instrumentation event for this config
    grep -E 'task done|pin_top_n_per_layer done|hot expert buffers built|exit_prefill|set_(top_k|prefetch|pin|cache_aware|speculative|prefill_hint)' \
        /tmp/tahoma-073-$CFG.log > "$DIR/instrumentation_073_${CFG}.log" 2>&1 || true
    # Kill before next config
    sudo -n pkill -9 -f 'target/release/tahoma' 2>/dev/null || true
    sleep 3
done

echo "=== all configs done ==="
bash "$DIR/compare.sh"
