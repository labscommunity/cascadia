#!/bin/bash
# K2.6 10-prompt robustness bench on 2-box matias via SSH-tunnel chain.
# Run from Mac; assumes Mac:18000 is forwarded to matias-02:8000.
# Mirrors k26_bench_miner_10.sh; the only difference is the default API URL.
set -u
API=${1:-http://127.0.0.1:18000}
OUTFILE=${2:-/tmp/k26-bench-2box-matias.jsonl}
MAX_TOKENS=${3:-32}

prompts=(
    "The capital of France is|paris"
    "The largest ocean on Earth is the|pacific"
    "Two plus two equals|four"
    "The first president of the United States was|washington"
    "The largest planet in our solar system is|jupiter"
    "Water boils at 100 degrees|celsius"
    "Python is a programming language created by|guido"
    "The square root of 144 is|12"
    "Mount Everest is located in the|himalaya"
    "The speed of light is approximately 300|km"
)

> "$OUTFILE"
total_tokens=0; total_ms=0; pass=0; n=${#prompts[@]}
for pair in "${prompts[@]}"; do
    prompt="${pair%%|*}"
    substr="${pair##*|}"
    body=$(jq -nc --arg p "$prompt" --argjson mt $MAX_TOKENS '{model:"k26",messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0}')
    t0=$(date +%s%N)
    r=$(curl -s -m 1200 -X POST "$API/v1/chat/completions" -H "Content-Type: application/json" -d "$body")
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    content=$(echo "$r" | jq -r '.choices[0].message.content // ""')
    ct=$(echo "$r" | jq -r '.usage.completion_tokens // 0')
    [ "$ct" = "0" ] && ct=$MAX_TOKENS
    quality_pass=false
    lc_content=$(echo "$content" | tr '[:upper:]' '[:lower:]')
    if echo "$lc_content" | grep -qF "$substr"; then quality_pass=true; pass=$((pass+1)); fi
    tok_s=$(awk "BEGIN { printf \"%.4f\", ($ct*1000)/$wall_ms }")
    total_tokens=$(( total_tokens + ct )); total_ms=$(( total_ms + wall_ms ))
    line=$(jq -nc \
        --arg prompt "$prompt" --arg substr "$substr" --arg content "$content" \
        --argjson ct $ct --argjson wall_ms $wall_ms --argjson tok_s "$tok_s" --argjson qp $quality_pass \
        '{prompt:$prompt,substr:$substr,content:$content,completion_tokens:$ct,wall_ms:$wall_ms,tok_per_sec:$tok_s,quality_pass:$qp}')
    echo "$line"
    echo "$line" >> "$OUTFILE"
done
agg_tok_s=$(awk "BEGIN { printf \"%.4f\", ($total_tokens*1000)/$total_ms }")
agg=$(jq -nc --argjson tt $total_tokens --argjson tw $total_ms --argjson ts $agg_tok_s --arg qp "$pass/$n" --arg ts_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "{aggregate:true,n_prompts:$n,total_tokens:\$tt,total_wall_ms:\$tw,tok_per_sec:\$ts,quality_pass:\$qp,timestamp_utc:\$ts_utc}")
echo "AGG: $agg"
echo "$agg" >> "$OUTFILE"
