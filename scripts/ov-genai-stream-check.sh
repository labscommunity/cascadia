#!/usr/bin/env bash
# shellcheck disable=SC2015  # verdict returns 0, so `A && verdict ok || verdict bad` is if-else
# Streaming check of the ov-genai default (stateful) path against ONE already-running
# OpenAI-compatible cascadia worker (`cascadia worker --engine ov-genai ...`, which streams
# each pipeline write). Run it from any machine that reaches the worker's port, or on the
# worker's own machine over loopback (the timing rows are only meaningful there).
#
# usage: scripts/ov-genai-stream-check.sh --url <base-url> [options]
#   --url URL              the worker's base URL, e.g. http://127.0.0.1:8000 (required)
#   --model ID             model name to send (default: first id of /v1/models)
#   --node-log SRC         the worker's log: a local path, or ssh:HOST:PATH (read with ssh + grep/tail).
#                          Adds the divergence row and prints the engine's ttft_ms per first-byte run
#   --first-byte-max S     first-byte bar, seconds, median of the runs (0.12)
#   --ttft-factor F        instead: first byte <= F x the median ttft_ms the engine logged for the
#                          same runs (an NPU row: 2); needs --node-log
#   --tok-s-min N          tok-s bar, completion tokens / total seconds, median (41)
#   --tokens-per-delta N   at most N completion tokens per content delta, median (8)
#   --speculative K        the worker runs --draft-model / --prompt-lookup with K tokens a round:
#                          at most K+1 tokens per delta (overrides --tokens-per-delta)
#   --runs N               timed streamed runs for first-byte / deltas / tok-s (3)
#   --max-tokens N         max_tokens of the timed runs and of cb-gap's streamed request (256)
#   --long-tokens N        max_tokens of the long request in queued-headers / cb-gap (1500)
#   --headers-max S        queued-headers bar, seconds (1.0)
#   --gap-ratio R          cb-gap bar: largest inter-delta gap <= R x the median gap (3)
#   --cb-gap               add the cb-gap row (the worker runs --cb)
#   --only LIST / --skip LIST   comma list of: first-byte deltas tok-s parity queued-headers cb-gap divergence
#   --timeout S            client limit on one request, seconds (600)
#   --commit SHA           the commit the worker was built from, printed with the run (default: this
#                          checkout's HEAD); the script cannot ask the worker
#   --no-thinking-flag     don't send enable_thinking: false
#   --dry-run              print the plan and exit
# Every request is a chat completion at temperature 0 (greedy). The timed runs ask the model to count
# in words, so every token is visible text.
# Needs bash, curl and python3; runs from Linux, macOS, Git Bash or WSL.
# Prints one PASS/FAIL (or SKIP) line per row and "OV-GENAI STREAM CHECK: ALL PASS" or
# "OV-GENAI STREAM CHECK: <n> FAILED" (exit 1).
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
URL=""; MODEL=""; NODE_LOG=""; FB_MAX=0.12; TTFT_FACTOR=""; TOKS_MIN=41; PER=8; SPEC=""
RUNS=3; MAX_TOKENS=256; LONG_TOKENS=1500; HDR_MAX=1.0; GAP_RATIO=3; CB_GAP=0; ONLY=""; SKIPS=""
TIMEOUT=600; COMMIT=""; THINK=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL=$2; shift ;;
    --model) MODEL=$2; shift ;;
    --node-log) NODE_LOG=$2; shift ;;
    --first-byte-max) FB_MAX=$2; shift ;;
    --ttft-factor) TTFT_FACTOR=$2; shift ;;
    --tok-s-min) TOKS_MIN=$2; shift ;;
    --tokens-per-delta) PER=$2; shift ;;
    --speculative) SPEC=$2; shift ;;
    --runs) RUNS=$2; shift ;;
    --max-tokens) MAX_TOKENS=$2; shift ;;
    --long-tokens) LONG_TOKENS=$2; shift ;;
    --headers-max) HDR_MAX=$2; shift ;;
    --gap-ratio) GAP_RATIO=$2; shift ;;
    --cb-gap) CB_GAP=1 ;;
    --only) ONLY=$2; shift ;;
    --skip) SKIPS=$2; shift ;;
    --timeout) TIMEOUT=$2; shift ;;
    --commit) COMMIT=$2; shift ;;
    --no-thinking-flag) THINK=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)"; exit 2 ;;
  esac
  shift
done
URL=${URL%/}
[ -n "$SPEC" ] && PER=$((SPEC + 1))
[ -n "$COMMIT" ] || COMMIT=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

DIVERGENCE_LINE=ov_genai_stream_divergence
ALL="first-byte deltas tok-s parity queued-headers"
[ "$CB_GAP" = 1 ] && ALL="$ALL cb-gap"
[ -n "$NODE_LOG" ] && ALL="$ALL divergence"
want() { # row name -> 0 when it runs
  local c=$1
  [[ " $ALL " == *" $c "* ]] || return 1
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$c,"* ]]; then return 1; fi
  [[ ",$SKIPS," == *",$c,"* ]] && return 1
  return 0
}

# Every effective argument, so the line reruns exactly this check.
reproduce() {
  local r=(scripts/ov-genai-stream-check.sh --url "${URL:-<url>}")
  [ -n "$MODEL" ] && r+=(--model "$MODEL")
  [ -n "$NODE_LOG" ] && r+=(--node-log "$NODE_LOG")
  if [ -n "$TTFT_FACTOR" ]; then r+=(--ttft-factor "$TTFT_FACTOR"); else r+=(--first-byte-max "$FB_MAX"); fi
  r+=(--tok-s-min "$TOKS_MIN")
  if [ -n "$SPEC" ]; then r+=(--speculative "$SPEC"); else r+=(--tokens-per-delta "$PER"); fi
  r+=(--runs "$RUNS" --max-tokens "$MAX_TOKENS" --long-tokens "$LONG_TOKENS" --headers-max "$HDR_MAX"
      --gap-ratio "$GAP_RATIO" --timeout "$TIMEOUT" --commit "$COMMIT")
  [ "$CB_GAP" = 1 ] && r+=(--cb-gap)
  [ -n "$ONLY" ] && r+=(--only "$ONLY")
  [ -n "$SKIPS" ] && r+=(--skip "$SKIPS")
  [ "$THINK" = 0 ] && r+=(--no-thinking-flag)
  printf 'reproduce:'; printf ' %q' "${r[@]}"; printf '\n'
}

if [ "$DRY" = 1 ]; then
  if [ -n "$TTFT_FACTOR" ]; then fbbar="<= ${TTFT_FACTOR} x the median ttft_ms the engine logged for those runs"
  else fbbar="<= ${FB_MAX}s"; fi
  cat <<EOF
OV-GENAI STREAM CHECK plan (dry run; nothing is sent)
  worker: ${URL:-<--url unset>}  model: ${MODEL:-<first of /v1/models>}  log: ${NODE_LOG:-none}
  commit: $COMMIT  run: $STAMP
  COUNT = chat "Count from 1 to 100000 in words, separated by commas.", temperature 0
  warm-up: a short streamed chat (8 tokens), then COUNT max_tokens $MAX_TOKENS stream:false once (not measured;
           a GPU's first full-length answer after a load can differ from every later one).
  rows:
EOF
  { want first-byte || want deltas || want tok-s; } && echo "   timed      COUNT streamed, max_tokens $MAX_TOKENS, $RUNS runs one after another; each run's first content
              delta, delta count, completion_tokens (usage) and total time are recorded."
  want first-byte && echo "   first-byte median time from sending to the first content delta $fbbar."
  want deltas && echo "   deltas     median completion tokens per content delta <= $PER (>= 1 delta per $PER tokens${SPEC:+; --speculative $SPEC})."
  want tok-s && echo "   tok-s      median completion_tokens / total seconds >= $TOKS_MIN."
  want parity && echo "   parity     COUNT max_tokens $MAX_TOKENS with stream:false, twice; the first timed run's streamed content
              joined must equal a stream:false answer byte for byte. When the two stream:false answers differ
              from each other and the stream equals neither, the device varies: SKIP (unverifiable)."
  want queued-headers && echo "   queued-headers  A = COUNT max_tokens $LONG_TOKENS, stream:false; B = a short streamed chat sent 0.5 s
              after A: B's status line and headers (HTTP 200, text/event-stream) within ${HDR_MAX}s of sending it."
  want cb-gap && echo "   cb-gap     A = COUNT max_tokens $LONG_TOKENS, stream:false; B = COUNT max_tokens $MAX_TOKENS streamed, sent
              0.5 s after A: B's largest gap between content deltas <= $GAP_RATIO x its median gap."
  want divergence && echo "   divergence the worker log gains no '$DIVERGENCE_LINE' line over the whole run."
  [ -n "$TTFT_FACTOR" ] && [ -z "$NODE_LOG" ] && echo "   note: --ttft-factor needs --node-log; first-byte will FAIL without it."
  reproduce
  exit 0
fi

[ -n "$URL" ] || { echo "--url is required (see --help)"; exit 2; }
PY=${PYTHON:-}
if [ -z "$PY" ]; then for p in python3 python; do "$p" -c 'import json' >/dev/null 2>&1 && { PY=$p; break; }; done; fi
[ -n "$PY" ] || { echo "python3 is required"; exit 2; }
command -v curl >/dev/null || { echo "curl is required"; exit 2; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t ogsc)
trap 'rm -rf "$WORK"' EXIT
FAILS=0; SKIPN=0
verdict() { if [ "$2" = ok ]; then echo "PASS $1"; else echo "FAIL $1 (${3:-})"; FAILS=$((FAILS+1)); fi; }
skip() { echo "SKIP $1"; SKIPN=$((SKIPN+1)); }
py() { "$PY" -I "$HERE/ov_genai_stream_check.py" "$@"; }
calc() { awk "BEGIN { printf \"%.3f\", ($1) }"; }

# --- the worker log: a local file or ssh:HOST:PATH ---
log_cmd() { # args -> runs the command on the log's machine
  if [[ "$NODE_LOG" == ssh:* ]]; then
    local rest=${NODE_LOG#ssh:} host path q
    host=${rest%%:*}; path=${rest#*:}
    printf -v q '%q ' "$@"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$q $(printf '%q' "$path")" 2>/dev/null
  else
    "$@" "$NODE_LOG" 2>/dev/null
  fi
}
log_lines() { log_cmd wc -l | awk '{ print $1 + 0 }'; }
log_since() { log_cmd tail -n "+$(( $1 + 1 ))" | tr -d '\r'; } # line offset -> the lines after it
div_count() { log_cmd grep -c "$DIVERGENCE_LINE" | awk '{ print $1 + 0 }'; }

# --- requests ---
# send <name> <kind> <max_tokens> <stream 0|1>  -> $WORK/<name>.json (timings + content)
send() {
  py body "$2" "$3" "$4" "$THINK" "$MODEL" > "$WORK/$1.body"
  py send "$URL" "$WORK/$1.body" "$WORK/$1.json" "$TIMEOUT"
}
brief() { py summary "$WORK/$1.json"; } # one line about a record

# --- run ---
code=$(curl -s -o "$WORK/models" --max-time 10 -w '%{http_code}' "$URL/v1/models")
[ "$code" = 200 ] || { echo "worker at $URL does not answer /v1/models (HTTP $code)"; echo "OV-GENAI STREAM CHECK: 1 FAILED"; exit 1; }
[ -n "$MODEL" ] || MODEL=$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$WORK/models" 2>/dev/null)
echo "OV-GENAI STREAM CHECK run $STAMP commit $COMMIT"
echo "worker $URL model $MODEL log ${NODE_LOG:-none}"
reproduce
DIV0=""
if [ -n "$NODE_LOG" ]; then
  DIV0=$(div_count); L0=$(log_lines)
  [ "${L0:-0}" -gt 0 ] || echo "warning: the worker log ${NODE_LOG} is empty or unreadable"
fi

# Warm-up, so no measured run is the worker's first. A short prompt is not enough on a GPU: the first full COUNT
# answer after a load can differ from every later one, streamed or not, which would fail parity for a reason
# that has nothing to do with streaming. So the warm-up also runs COUNT itself once at full length.
send warm short 8 1
send warm-count count "$MAX_TOKENS" 0
echo "warm-up: COUNT stream:false $(brief warm-count)"

if want first-byte || want deltas || want tok-s; then
  recs=()
  OFF=0; [ -n "$NODE_LOG" ] && OFF=$(log_lines)
  for i in $(seq 1 "$RUNS"); do
    send "t$i" count "$MAX_TOKENS" 1
    recs+=("$WORK/t$i.json")
    echo "timed run $i: $(brief "t$i")"
  done
  TTFTS=""; [ -n "$NODE_LOG" ] && { log_since "$OFF" > "$WORK/log.timed"; TTFTS=$(py ttft "$WORK/log.timed"); }
  [ -n "$NODE_LOG" ] && echo "INFO engine ttft_ms of the timed runs (task done lines): ${TTFTS:-none logged}"
  if want first-byte; then
    bar=$FB_MAX; how="${FB_MAX}s"
    if [ -n "$TTFT_FACTOR" ]; then
      if [ -n "$TTFTS" ]; then
        # shellcheck disable=SC2086  # one argument per value
        mt=$(py median $TTFTS); bar=$(calc "$TTFT_FACTOR * $mt / 1000"); how="${TTFT_FACTOR} x engine ttft ${mt} ms = ${bar}s"
      else
        bar=-1; how="${TTFT_FACTOR} x engine ttft (none logged; give --node-log)"
      fi
    fi
    read -r v m xs <<<"$(py first-byte "$bar" "${recs[@]}")"
    [ "$v" = ok ] && verdict "first-byte: median first content delta ${m}s <= $how (runs $xs)" ok \
      || verdict "first-byte: median first content delta <= $how" bad "median ${m}s, runs $xs"
  fi
  if want deltas; then
    read -r v m xs <<<"$(py deltas "$PER" "${recs[@]}")"
    [ "$v" = ok ] && verdict "deltas: median ${m} completion tokens per content delta <= $PER (runs $xs)" ok \
      || verdict "deltas: >= 1 content delta per $PER completion tokens" bad "median ${m} tokens/delta, runs $xs"
  fi
  if want tok-s; then
    read -r v m xs <<<"$(py tok-s "$TOKS_MIN" "${recs[@]}")"
    [ "$v" = ok ] && verdict "tok-s: median ${m} completion tokens/s >= $TOKS_MIN (runs $xs)" ok \
      || verdict "tok-s: >= $TOKS_MIN completion tokens/s" bad "median ${m}, runs $xs"
  fi
fi

if want parity; then
  [ -e "$WORK/t1.json" ] || send t1 count "$MAX_TOKENS" 1
  send ref1 count "$MAX_TOKENS" 0
  send ref2 count "$MAX_TOKENS" 0
  echo "parity: stream:false $(brief ref1); again $(brief ref2)"
  read -r v detail <<<"$(py parity "$WORK/t1.json" "$WORK/ref1.json" "$WORK/ref2.json")"
  case "$v" in
    ok) verdict "parity: the streamed deltas join to the stream:false answer byte for byte ($detail)" ok ;;
    varies) skip "parity: unverifiable, the two stream:false answers differ from each other and the stream equals neither ($detail)" ;;
    *) verdict "parity: the streamed deltas join to the stream:false answer byte for byte" bad "$detail" ;;
  esac
fi

# pair <a> <b> <b kind> <b max_tokens>: A (stream:false, LONG_TOKENS) in the background, then B
# streamed 0.5 s later; waits for both.
pair() {
  local pa
  py body count "$LONG_TOKENS" 0 "$THINK" "$MODEL" > "$WORK/$1.body"
  py send "$URL" "$WORK/$1.body" "$WORK/$1.json" "$TIMEOUT" &
  pa=$!
  sleep 0.5
  send "$2" "$3" "$4" 1
  wait "$pa"
}

if want queued-headers; then
  pair qa qb short 16
  echo "queued-headers: A $(brief qa); B $(brief qb)"
  read -r v t detail <<<"$(py headers "$HDR_MAX" "$WORK/qb.json")"
  [ "$v" = ok ] && verdict "queued-headers: B's headers came ${t}s after it was sent, <= ${HDR_MAX}s ($detail)" ok \
    || verdict "queued-headers: B's headers within ${HDR_MAX}s" bad "${t}s, $detail"
fi

if want cb-gap; then
  pair ca cb count "$MAX_TOKENS"
  echo "cb-gap: A $(brief ca); B $(brief cb)"
  read -r v mx md n <<<"$(py gap "$GAP_RATIO" "$WORK/cb.json")"
  [ "$v" = ok ] && verdict "cb-gap: B's largest inter-delta gap ${mx}s <= $GAP_RATIO x median ${md}s ($n gaps)" ok \
    || verdict "cb-gap: B's largest inter-delta gap <= $GAP_RATIO x its median" bad "max ${mx}s, median ${md}s, $n gaps"
fi

if want divergence; then
  DIV1=$(div_count)
  if [ -z "$DIV0" ] || [ -z "$DIV1" ]; then
    verdict "divergence: the worker log is readable" bad "could not count lines in $NODE_LOG"
  else
    added=$((DIV1 - DIV0))
    [ "$added" = 0 ] && verdict "divergence: no '$DIVERGENCE_LINE' line added to the worker log during the run" ok \
      || verdict "divergence: no '$DIVERGENCE_LINE' line added" bad "$added line(s): $(log_since "${L0:-0}" | grep -m1 "$DIVERGENCE_LINE")"
  fi
fi

[ "$SKIPN" -gt 0 ] && echo "INFO $SKIPN row(s) skipped"
if [ "$FAILS" = 0 ]; then echo "OV-GENAI STREAM CHECK: ALL PASS"; else echo "OV-GENAI STREAM CHECK: $FAILS FAILED"; exit 1; fi
