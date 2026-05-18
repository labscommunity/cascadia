#!/bin/bash
# Compare F5 bench jsonl files. Produces a markdown comparison table.
# Usage: compare.sh <bench_w0.jsonl> [bench_w32.jsonl ...]
set -u
cd "$(dirname "$0")"

printf "| W   | mt  | tok/s   | n_tok | wall_s | quality | content (first 80 chars)\n"
printf "|-----|-----|---------|-------|--------|---------|-------------------------\n"

for f in bench_*.jsonl; do
    [ -f "$f" ] || continue
    # Filename: bench_w<W>_mt<MT>.jsonl
    fn=${f#bench_}
    fn=${fn%.jsonl}
    w=${fn%%_*}; w=${w#w}
    mt=${fn##*_}; mt=${mt#mt}
    agg=$(tail -1 "$f" 2>/dev/null)
    n_tok=$(echo "$agg" | jq -r .total_tokens)
    wall_ms=$(echo "$agg" | jq -r .total_wall_ms)
    tok_s=$(echo "$agg" | jq -r .tok_per_sec)
    quality=$(echo "$agg" | jq -r .quality_pass)
    wall_s=$(awk "BEGIN {printf \"%.1f\", $wall_ms/1000}")
    # First prompt's content
    content=$(head -1 "$f" 2>/dev/null | jq -r .content | head -c 80 | tr '\n' ' ')
    printf "| %-3s | %-3s | %-7s | %-5s | %-6s | %-7s | %s\n" "$w" "$mt" "$tok_s" "$n_tok" "$wall_s" "$quality" "$content"
done
