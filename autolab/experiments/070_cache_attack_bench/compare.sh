#!/bin/bash
# Compare iter 070 cache-attack-bench results vs iter 021 baseline and
# iter 044 spec-decode baseline. Prints a markdown table to stdout.
set -u

DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$(cd "$DIR/../.." && pwd)"

ITER070="$DIR/bench_cache_attack_mt64_temp0.jsonl"
ITER021="$ROOT/experiments/021_k6_longcontext/bench_k6_mt64.jsonl"
# iter 044 may or may not have a bench JSONL we can reference; if not,
# we fall back to the journal-recorded numbers.

agg_line() {
    jq -c 'select(.aggregate==true)' "$1" 2>/dev/null | head -1
}

a070=$(agg_line "$ITER070")
a021=$(agg_line "$ITER021")

if [ -z "$a070" ]; then
    echo "WARN: $ITER070 has no aggregate line yet" 1>&2
fi
if [ -z "$a021" ]; then
    echo "ERROR: missing iter 021 baseline at $ITER021" 1>&2
    exit 1
fi

t021=$(echo "$a021" | jq -r '.tok_per_sec')
q021=$(echo "$a021" | jq -r '.quality_pass')
t070=$(echo "$a070" | jq -r '.tok_per_sec')
q070=$(echo "$a070" | jq -r '.quality_pass')

# iter 044 reference (per task description): 0.1899 tok/s aggregate, 9/10
t044="0.1899"
q044="9/10"

vs021=$(awk "BEGIN { if ($t021>0) printf \"%+.1f%%\", (($t070-$t021)/$t021)*100; else print \"NA\" }")
vs044=$(awk "BEGIN { if ($t044>0) printf \"%+.1f%%\", (($t070-$t044)/$t044)*100; else print \"NA\" }")

cat <<EOF
| Config | tok/s | quality | vs iter 021 |
|---|---|---|---|
| iter 021 baseline (K=6, no opt-ins) | $t021 | $q021 | - |
| iter 044 (K=6 + spec-decode) | $t044 | $q044 | $(awk "BEGIN { printf \"%+.1f%%\", (($t044-$t021)/$t021)*100 }") |
| iter 070 (K=6 + full 7-feature cache-attack stack) | $t070 | $q070 | $vs021 (vs iter 044: $vs044) |
EOF
