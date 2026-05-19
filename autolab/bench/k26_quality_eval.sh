#!/bin/bash
# K2.6 quality eval harness — iter 095 follow-up to iter 037's
# substring-eval-too-weak finding.
#
# Why this exists:
#   iter 037 measured +80% TPS at W=32 but the 10-prompt substring eval
#   passed garbage outputs because 'paris' was in the first sentence
#   while the rest of the generation degenerated to 'Question? Question?
#   ...' loops. The classic eval is silent on coherence past ~25 tokens.
#
# What this script does instead:
#   * Generates the FULL output for each prompt under TWO configs
#     (baseline + feature) and dumps both verbatim to JSONL.
#   * Computes first-divergence position (token-ish via whitespace + char)
#     between baseline and feature responses for the same prompt.
#   * Computes total bytes per response (output-length proxy — short
#     responses for the feature config raise an EOS-on-garbage flag).
#   * Prints a side-by-side comparison table at the end.
#   * Returns non-zero if quality regression triggers (configurable).
#
# What this script intentionally does NOT do (yet):
#   * Perplexity vs baseline-logits — requires the API to expose
#     per-token logprobs. /v1/chat/completions on tahoma-api currently
#     returns content only. Adding logprobs is a separate piece of work
#     (TODO: see crates/tahoma-api/src/lib.rs `ChatCompletionChoice`).
#     When that lands, plumb the logprobs through and add a perplexity
#     column. Left as a stub in the per-prompt JSONL row
#     (`logprob_baseline`/`logprob_feature` fields kept null).
#   * Eyeball-eval CLI helper — see `k26_eyeball_eval.sh` in this dir;
#     this script writes JSONL that eyeball loads side-by-side.
#
# Usage modes:
#
#   (1) Two endpoints live at once (parallel-host eval — fastest):
#     k26_quality_eval.sh \
#       --baseline-api http://miner-a:8000 \
#       --feature-api  http://miner-b:8000 \
#       --max-tokens 64 \
#       --out-dir /tmp/k26-qeval
#
#   (2) One endpoint, sequential (you re-launch worker between runs):
#     # baseline (W=0)
#     k26_quality_eval.sh --api http://127.0.0.1:8000 \
#         --label baseline --max-tokens 64 \
#         --out-dir /tmp/k26-qeval
#     # feature (W=32) — after restarting worker with --attention-window 32
#     k26_quality_eval.sh --api http://127.0.0.1:8000 \
#         --label feature --max-tokens 64 \
#         --out-dir /tmp/k26-qeval
#     # then compare
#     k26_quality_eval.sh --compare /tmp/k26-qeval
#
# Outputs (mode 1, both endpoints):
#   $OUT_DIR/baseline.jsonl    # one row per prompt, full content
#   $OUT_DIR/feature.jsonl     # one row per prompt, full content
#   $OUT_DIR/comparison.jsonl  # per-prompt divergence + length-delta
#   $OUT_DIR/summary.txt       # human-readable side-by-side table
#
# Exit codes:
#   0  — eval completed, no regressions tripped
#   1  — bad usage / missing flags
#   2  — API unreachable / probe failed
#   3  — eval ran, but at least one regression flag tripped (EOS-on-
#        garbage suspected or first-divergence < min-divergence-threshold)

set -u

# -------- defaults --------
BASELINE_API=""
FEATURE_API=""
SINGLE_API=""
LABEL=""
OUT_DIR="/tmp/k26-qeval"
MAX_TOKENS=64
TEMPERATURE=0
MODEL="k26"
TIMEOUT_S=900
EOS_GARBAGE_FRAC="0.5"   # feature_bytes < 0.5 * baseline_bytes => flag
MIN_DIVERGENCE_TOK=8     # divergence before token 8 => flag
COMPARE_ONLY=""
PROMPTS_FILE=""

# Same 10 prompts as k26_bench_miner_10.sh so this slots into existing
# experiment dirs. Each line is a single prompt (no substr — we don't
# trust substring eval, that's the whole point).
DEFAULT_PROMPTS=(
    "The capital of France is"
    "The largest ocean on Earth is the"
    "Two plus two equals"
    "The first president of the United States was"
    "The largest planet in our solar system is"
    "Water boils at 100 degrees"
    "Python is a programming language created by"
    "The square root of 144 is"
    "Mount Everest is located in the"
    "The speed of light is approximately 300"
)

usage() {
    sed -n '2,50p' "$0"
    exit 1
}

# -------- arg parsing --------
while [ $# -gt 0 ]; do
    case "$1" in
        --baseline-api)  BASELINE_API="$2"; shift 2;;
        --feature-api)   FEATURE_API="$2";  shift 2;;
        --api)           SINGLE_API="$2";   shift 2;;
        --label)         LABEL="$2";        shift 2;;
        --out-dir)       OUT_DIR="$2";      shift 2;;
        --max-tokens)    MAX_TOKENS="$2";   shift 2;;
        --temperature)   TEMPERATURE="$2";  shift 2;;
        --model)         MODEL="$2";        shift 2;;
        --timeout)       TIMEOUT_S="$2";    shift 2;;
        --eos-garbage-frac)    EOS_GARBAGE_FRAC="$2";    shift 2;;
        --min-divergence-tok)  MIN_DIVERGENCE_TOK="$2";  shift 2;;
        --prompts-file)  PROMPTS_FILE="$2"; shift 2;;
        --compare)       COMPARE_ONLY="$2"; shift 2;;
        -h|--help)       usage;;
        *) echo "unknown flag: $1" >&2; exit 1;;
    esac
done

# -------- prompts loader --------
load_prompts() {
    if [ -n "$PROMPTS_FILE" ]; then
        # one prompt per line
        prompts=()
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            case "$line" in '#'*) continue;; esac
            prompts+=("$line")
        done < "$PROMPTS_FILE"
    else
        prompts=("${DEFAULT_PROMPTS[@]}")
    fi
}

# -------- helpers --------

# Issue a single completion request, return content + completion_tokens
# + wall_ms via globals: g_content, g_ct, g_wall_ms.
gen_one() {
    local api="$1"
    local prompt="$2"
    local body
    body=$(jq -nc --arg p "$prompt" --argjson mt "$MAX_TOKENS" --argjson tmp "$TEMPERATURE" --arg m "$MODEL" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:$tmp}')
    local t0 t1 r
    t0=$(date +%s%N)
    r=$(curl -s -m "$TIMEOUT_S" -X POST "$api/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body")
    t1=$(date +%s%N)
    g_wall_ms=$(( (t1 - t0) / 1000000 ))
    g_content=$(echo "$r" | jq -r '.choices[0].message.content // ""')
    g_ct=$(echo "$r" | jq -r '.usage.completion_tokens // 0')
    [ "$g_ct" = "0" ] && g_ct="$MAX_TOKENS"
}

# Probe API health + tiny inference. Exits 2 on failure.
probe_api() {
    local api="$1"
    local name="$2"
    echo "probing $name @ $api ..." >&2
    if ! curl -s -m 10 -f "$api/health" >/dev/null 2>&1; then
        echo "ERROR: $name /health failed at $api" >&2
        exit 2
    fi
    local body
    body=$(jq -nc --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"Hi"}],max_tokens:1,temperature:0}')
    if ! curl -s -m "$TIMEOUT_S" -f -X POST "$api/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body" >/dev/null 2>&1; then
        echo "ERROR: $name tiny inference probe failed at $api" >&2
        exit 2
    fi
    echo "  $name OK" >&2
}

# Drive one config (baseline OR feature) end-to-end, write $OUT_DIR/$label.jsonl.
run_config() {
    local api="$1"
    local label="$2"
    local outfile="$OUT_DIR/${label}.jsonl"
    : > "$outfile"
    local i=0
    for prompt in "${prompts[@]}"; do
        gen_one "$api" "$prompt"
        local bytes=${#g_content}
        # word count = whitespace-split (cheap proxy for token count
        # without a tokenizer dep — good enough for "did the output
        # truncate?" decisions).
        local words
        words=$(printf '%s' "$g_content" | wc -w | tr -d ' ')
        local tok_s
        if [ "$g_wall_ms" -gt 0 ]; then
            tok_s=$(awk "BEGIN { printf \"%.4f\", ($g_ct*1000)/$g_wall_ms }")
        else
            tok_s="0"
        fi
        local row
        row=$(jq -nc \
            --argjson idx "$i" \
            --arg prompt "$prompt" \
            --arg content "$g_content" \
            --argjson ct "$g_ct" \
            --argjson wall_ms "$g_wall_ms" \
            --argjson tok_s "$tok_s" \
            --argjson bytes "$bytes" \
            --argjson words "$words" \
            --arg label "$label" \
            '{idx:$idx, label:$label, prompt:$prompt, content:$content,
              completion_tokens:$ct, wall_ms:$wall_ms, tok_per_sec:$tok_s,
              bytes:$bytes, words:$words,
              logprob_baseline:null, logprob_feature:null}')
        echo "$row" >> "$outfile"
        echo "[$label] idx=$i ct=$g_ct wall_ms=$g_wall_ms tok/s=$tok_s bytes=$bytes" >&2
        i=$((i + 1))
    done
    echo "wrote $outfile ($i rows)" >&2
}

# Compute first-divergence position (whitespace-split word index) between
# two strings. Echoes the divergence position; -1 means identical, or the
# length of the shorter string if one is a prefix of the other.
first_divergence_words() {
    local a="$1"
    local b="$2"
    # Use python for safe word-split (handles unicode, multi-space).
    # python3 is available everywhere we run miner / matias.
    python3 - "$a" "$b" <<'PYEOF'
import sys
a = sys.argv[1].split()
b = sys.argv[2].split()
n = min(len(a), len(b))
for i in range(n):
    if a[i] != b[i]:
        print(i)
        sys.exit(0)
if len(a) == len(b):
    print(-1)
else:
    print(n)
PYEOF
}

# Compute first-divergence position in BYTES (a tighter signal — catches
# casing / punctuation differences a word-split eats).
first_divergence_bytes() {
    local a="$1"
    local b="$2"
    python3 - "$a" "$b" <<'PYEOF'
import sys
a = sys.argv[1].encode('utf-8')
b = sys.argv[2].encode('utf-8')
n = min(len(a), len(b))
for i in range(n):
    if a[i] != b[i]:
        print(i)
        sys.exit(0)
if len(a) == len(b):
    print(-1)
else:
    print(n)
PYEOF
}

# Compare two JSONL files (baseline + feature), produce comparison.jsonl
# + summary.txt. Sets g_regressions count.
compare_runs() {
    local dir="$1"
    local base="$dir/baseline.jsonl"
    local feat="$dir/feature.jsonl"
    local cmp="$dir/comparison.jsonl"
    local summary="$dir/summary.txt"
    if [ ! -f "$base" ] || [ ! -f "$feat" ]; then
        echo "ERROR: $base and/or $feat missing — need both to compare" >&2
        exit 2
    fi
    : > "$cmp"
    g_regressions=0
    {
        printf '%s\n' "K2.6 quality eval — baseline vs feature"
        printf '%s\n' "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\n' "baseline:  $base"
        printf '%s\n' "feature:   $feat"
        printf '%s\n' "eos-garbage-frac threshold: $EOS_GARBAGE_FRAC"
        printf '%s\n' "min-divergence-tok threshold: $MIN_DIVERGENCE_TOK"
        printf '\n'
        printf '%-3s | %-8s | %-8s | %-7s | %-7s | %-7s | %-8s | %s\n' \
            'idx' 'base_b' 'feat_b' 'div_wrd' 'div_byt' 'len_frac' 'flag' 'prompt'
        printf '%s\n' '----+----------+----------+---------+---------+---------+----------+--------------------'
    } > "$summary"

    local n
    n=$(wc -l < "$base" | tr -d ' ')
    local i=0
    while [ "$i" -lt "$n" ]; do
        local brow frow
        brow=$(sed -n "$((i + 1))p" "$base")
        frow=$(sed -n "$((i + 1))p" "$feat")
        local bprompt bcontent bbytes bwords
        local fcontent fbytes fwords
        bprompt=$(echo "$brow" | jq -r '.prompt')
        bcontent=$(echo "$brow" | jq -r '.content')
        bbytes=$(echo "$brow" | jq -r '.bytes')
        bwords=$(echo "$brow" | jq -r '.words')
        fcontent=$(echo "$frow" | jq -r '.content')
        fbytes=$(echo "$frow" | jq -r '.bytes')
        fwords=$(echo "$frow" | jq -r '.words')

        local div_words div_bytes len_frac
        div_words=$(first_divergence_words "$bcontent" "$fcontent")
        div_bytes=$(first_divergence_bytes "$bcontent" "$fcontent")
        if [ "$bbytes" -gt 0 ]; then
            len_frac=$(awk "BEGIN { printf \"%.3f\", $fbytes/$bbytes }")
        else
            len_frac="nan"
        fi

        # Flag regressions:
        #   * feature shrank by > (1 - eos-garbage-frac) — possible
        #     EOS-on-garbage early-stop.
        #   * outputs diverged at word index < min-divergence-tok and
        #     feature is shorter — likely quality cliff.
        local flag="ok"
        local short_flag long_div_flag
        short_flag=$(awk "BEGIN { print ($fbytes < $EOS_GARBAGE_FRAC * $bbytes) ? 1 : 0 }")
        if [ "$short_flag" = "1" ] && [ "$bbytes" -gt 0 ]; then
            flag="SHORT"
            g_regressions=$((g_regressions + 1))
        fi
        if [ "$div_words" -ge 0 ] && [ "$div_words" -lt "$MIN_DIVERGENCE_TOK" ]; then
            if [ "$flag" = "ok" ]; then
                flag="EARLYDIV"
                g_regressions=$((g_regressions + 1))
            else
                flag="${flag}+EARLYDIV"
            fi
        fi

        local crow
        crow=$(jq -nc \
            --argjson idx "$i" \
            --arg prompt "$bprompt" \
            --arg baseline_content "$bcontent" \
            --arg feature_content "$fcontent" \
            --argjson baseline_bytes "$bbytes" \
            --argjson feature_bytes "$fbytes" \
            --argjson baseline_words "$bwords" \
            --argjson feature_words "$fwords" \
            --argjson first_divergence_words "$div_words" \
            --argjson first_divergence_bytes "$div_bytes" \
            --arg length_fraction "$len_frac" \
            --arg flag "$flag" \
            '{idx:$idx, prompt:$prompt,
              baseline_bytes:$baseline_bytes, feature_bytes:$feature_bytes,
              baseline_words:$baseline_words, feature_words:$feature_words,
              first_divergence_words:$first_divergence_words,
              first_divergence_bytes:$first_divergence_bytes,
              length_fraction:($length_fraction|tonumber? // null),
              flag:$flag,
              baseline_content:$baseline_content,
              feature_content:$feature_content}')
        echo "$crow" >> "$cmp"

        printf '%-3d | %-8d | %-8d | %-7s | %-7s | %-7s | %-8s | %s\n' \
            "$i" "$bbytes" "$fbytes" "$div_words" "$div_bytes" "$len_frac" "$flag" \
            "$(printf '%s' "$bprompt" | cut -c1-40)" \
            >> "$summary"

        i=$((i + 1))
    done

    {
        printf '\n'
        printf 'regressions tripped: %d / %d\n' "$g_regressions" "$n"
        printf '\n'
        printf 'Legend:\n'
        printf '  base_b / feat_b   bytes of completion content (baseline / feature)\n'
        printf '  div_wrd / div_byt first-divergence index (word / byte); -1 = identical\n'
        printf '  len_frac          feature_bytes / baseline_bytes\n'
        printf '  flag              ok = no regression\n'
        printf '                    SHORT = feature shrank > (1 - eos-garbage-frac); possible EOS-on-garbage\n'
        printf '                    EARLYDIV = diverged before min-divergence-tok words\n'
        printf '\n'
        printf 'For full content side-by-side, run:\n'
        printf '  bash autolab/bench/k26_eyeball_eval.sh %s\n' "$dir"
    } >> "$summary"

    cat "$summary"
    return $g_regressions
}

# -------- main --------

mkdir -p "$OUT_DIR"
load_prompts

if [ -n "$COMPARE_ONLY" ]; then
    OUT_DIR="$COMPARE_ONLY"
    compare_runs "$OUT_DIR"
    rc=$?
    [ $rc -gt 0 ] && exit 3
    exit 0
fi

if [ -n "$BASELINE_API" ] && [ -n "$FEATURE_API" ]; then
    # Mode 1: both endpoints live
    probe_api "$BASELINE_API" "baseline"
    probe_api "$FEATURE_API"  "feature"
    run_config "$BASELINE_API" "baseline"
    run_config "$FEATURE_API"  "feature"
    compare_runs "$OUT_DIR"
    rc=$?
    [ $rc -gt 0 ] && exit 3
    exit 0
fi

if [ -n "$SINGLE_API" ] && [ -n "$LABEL" ]; then
    # Mode 2: one endpoint, sequential — user runs twice
    case "$LABEL" in
        baseline|feature) ;;
        *) echo "ERROR: --label must be 'baseline' or 'feature'" >&2; exit 1;;
    esac
    probe_api "$SINGLE_API" "$LABEL"
    run_config "$SINGLE_API" "$LABEL"
    echo "wrote $OUT_DIR/${LABEL}.jsonl" >&2
    echo "when both runs are done, run: $0 --compare $OUT_DIR" >&2
    exit 0
fi

echo "ERROR: need either (--baseline-api + --feature-api), or (--api + --label), or --compare DIR" >&2
usage
