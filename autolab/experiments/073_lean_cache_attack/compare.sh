#!/bin/bash
# Aggregate iter 073 lean-cache-attack bench results vs the iter 021
# K=6 baseline and the iter 070 full 7-feature stack. Prints a markdown
# table to stdout.
set -u

DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd "$DIR/../.." && pwd)"

ITER021="$ROOT/experiments/021_k6_longcontext/bench_k6_mt64.jsonl"

# iter 070 reference (per its STATUS.md / partial bench): 0.1108 tok/s on prompt 1
# Full 10-prompt iter 070 was abandoned (-32% sub-baseline regression).
ITER070_TOKS="0.1108"
ITER070_QUAL="pass(1/1)"

agg_line() {
    jq -c 'select(.aggregate==true)' "$1" 2>/dev/null | head -1
}

a021=$(agg_line "$ITER021")
t021=$(echo "$a021" | jq -r '.tok_per_sec')
q021=$(echo "$a021" | jq -r '.quality_pass')

if [ -z "$a021" ]; then
    echo "ERROR: missing iter 021 baseline at $ITER021" 1>&2
    exit 1
fi

cat <<EOF
| Config | tok/s | quality | vs iter 021 |
|---|---|---|---|
| iter 021 baseline (K=6, no opt-ins) | $t021 | $q021 | - |
| iter 070 full 7-stack (prompt 1 only) | $ITER070_TOKS | $ITER070_QUAL | $(awk "BEGIN { printf \"%+.1f%%\", (($ITER070_TOKS-$t021)/$t021)*100 }") |
EOF

for cfg in A B C D E; do
    f="$DIR/bench_073_${cfg}.jsonl"
    [ -f "$f" ] || continue
    a=$(agg_line "$f")
    [ -z "$a" ] && continue
    t=$(echo "$a" | jq -r '.tok_per_sec')
    q=$(echo "$a" | jq -r '.quality_pass')
    vs=$(awk "BEGIN { if ($t021>0) printf \"%+.1f%%\", (($t-$t021)/$t021)*100; else print \"NA\" }")
    case "$cfg" in
        A) name="A: pin only (iter 054)";;
        B) name="B: pin + cache-aware dispatch";;
        C) name="C: pin + dispatch + hot-buffer";;
        D) name="D: pin + dispatch + hot-buffer + light prefetch (n=8)";;
        E) name="E: pin + dispatch + hot-buffer + prefill-hint (W=0.5)";;
    esac
    echo "| iter 073 $name | $t | $q | $vs |"
done
