#!/usr/bin/env bash
# shellcheck disable=SC2015,SC2028  # verdict returns 0, so `A && verdict ok || verdict bad` is if-else; the plan prints "\\n" as is
# In-flight cancel check (M18) against ONE already-running OpenAI-compatible cascadia node.
# A client hang-up, a `stop` match and /v1/cancel must end a running one-shot generation
# (stateful ov-genai, single-stage sparse-moe), and the engine must be free for the next request
# at once. The engines log one line per task (M18 D7), which this script reads:
#   task cancelled tokens=N elapsed_s=S | task stopped at stop string tokens=N | task done ...
# and it FAILs on any "engine stopped on a stop the returned text lacks (D8 invariant)" line.
#
# usage: scripts/inflight-cancel-check.sh --url <base-url> (--log <file> | --log-cmd <cmd>) [options]
#   --url URL            node base URL, e.g. http://127.0.0.1:8671          (env CHECK_URL)
#   --model ID           model name to send (default: first id of /v1/models) (env CHECK_MODEL)
#   --log FILE           the node's log file, read as it grows                (env CHECK_LOG)
#   --log-cmd CMD        a command that prints the node's whole log so far    (env CHECK_LOG_CMD)
#   --ab URL             a second node (the base build, same model): adds case 6 and H6 (tok/s A/B)
#                        and compares every stop case's content with it (H4)  (env CHECK_AB_URL)
#   --only LIST / --skip LIST   comma list of: baseline stream nonstream stop cancel h1 h2 h7 h8 ab h6
#   --round-tokens K     tokens one engine round may add after a cancel (1; draft / prompt-lookup: k;
#                        ov-genai CB-adapter backend: 2)
#   --eps S              latency allowance for the hang-up to reach the node, seconds (0.5)
#   --n-tol PCT          tolerance on the cancelled task's token count vs D*R, percent (25)
#   --long-max-tokens N  max_tokens of the long request (default: none, the server's window-long default)
#   --long-timeout S     client limit on a long request that should end early (stop cases) (300)
#   --settle S           client limit on a request queued behind a long one (600)
#   --log-wait S         how long to wait for a task line to show in the log (20)
#   --ref-max-tokens N   max_tokens of the reference count run the stop cases compare with (200)
#   --ab-runs N          timed runs per node in the A/B (3); --ab-bar PCT  tip >= PCT % of base (97)
#   --no-task-log        the engine logs no per-task lines (--cb, multi-stage): skip those log checks
#   --no-thinking-flag   don't send enable_thinking: false on chat requests
#   --dry-run            print the plan and exit
# Needs bash, curl, python3 (or python) and standard tools; runs from Linux, macOS, Git Bash or WSL.
# Prints one PASS/FAIL (or SKIP) line per claim and
# "INFLIGHT CANCEL CHECK: ALL PASS" or "INFLIGHT CANCEL CHECK: <n> FAILED" (exit 1).
set -uo pipefail

URL=${CHECK_URL:-}; MODEL=${CHECK_MODEL:-}; LOG_FILE=${CHECK_LOG:-}; LOG_CMD=${CHECK_LOG_CMD:-}
AB_URL=${CHECK_AB_URL:-}; AB_MODEL=""
ONLY=""; SKIPS=""; ROUND=1; EPS=0.5; NTOL=25; LONG_MAX=""; LONG_TIMEOUT=300; SETTLE=600
LOG_WAIT=20; REF_MAX=200; T0_MAX=24; AB_RUNS=3; AB_BAR=97; NO_TASK_LOG=0; THINK=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL=$2; shift ;;
    --model) MODEL=$2; shift ;;
    --log) LOG_FILE=$2; shift ;;
    --log-cmd) LOG_CMD=$2; shift ;;
    --ab) AB_URL=$2; shift ;;
    --only) ONLY=$2; shift ;;
    --skip) SKIPS=$2; shift ;;
    --round-tokens) ROUND=$2; shift ;;
    --eps) EPS=$2; shift ;;
    --n-tol) NTOL=$2; shift ;;
    --long-max-tokens) LONG_MAX=$2; shift ;;
    --long-timeout) LONG_TIMEOUT=$2; shift ;;
    --settle) SETTLE=$2; shift ;;
    --log-wait) LOG_WAIT=$2; shift ;;
    --ref-max-tokens) REF_MAX=$2; shift ;;
    --ab-runs) AB_RUNS=$2; shift ;;
    --ab-bar) AB_BAR=$2; shift ;;
    --no-task-log) NO_TASK_LOG=1 ;;
    --no-thinking-flag) THINK=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '3,33p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)"; exit 2 ;;
  esac
  shift
done
URL=${URL%/}; AB_URL=${AB_URL%/}

ALL="baseline stream nonstream stop cancel h1 h2 h7 h8"
[ -n "$AB_URL" ] && ALL="$ALL ab h6"
want() { # case name -> 0 when it runs
  local c=$1
  [[ " $ALL " == *" $c "* ]] || return 1
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$c,"* ]]; then return 1; fi
  [[ ",$SKIPS," == *",$c,"* ]] && return 1
  return 0
}
# The baseline (T0, R, the reference text) feeds every case but the A/B ones.
NEED_BASE=0; for c in stream nonstream stop cancel h1 h2 h7 h8; do want "$c" && NEED_BASE=1; done
want baseline && NEED_BASE=1

if [ "$DRY" = 1 ]; then
  cat <<EOF
INFLIGHT CANCEL CHECK plan (dry run; nothing is sent)
  node: ${URL:-<--url unset>}  model: ${MODEL:-<first of /v1/models>}  log: ${LOG_FILE:-${LOG_CMD:-<unset>}}
  A/B base node: ${AB_URL:-none}   round tokens K=$ROUND  eps=${EPS}s  long max_tokens: ${LONG_MAX:-server default}
  T0 = chat "Write one sentence about the sea.", temperature 0, max_tokens $T0_MAX, streamed
  LONG = chat "Count from 1 to 100000 in words, separated by commas.", temperature 0
  cases:
EOF
  want baseline || [ "$NEED_BASE" = 1 ] && cat <<EOF
   baseline   T0 twice (must be equal); LONG max_tokens 128 -> R = tok_s of its 'task done' line
              (else completion_tokens / time); LONG max_tokens $REF_MAX with a never-matching stop -> T.
              D = max(3 s, 20/R s); one token time = 1/R.
EOF
  want stream && echo "   stream     case 2: LONG streamed, client hangs up at D (curl --max-time D); T0 sent 0.1 s later.
              PASS: curl timed out (the task was running); the next task line in the log is 'task cancelled'
              for the long task (not 'task done'); its elapsed_s <= D + K/R + eps; |tokens - R*D| <= K +
              max(2, ${NTOL}% of R*D); the T0 follow-up's first byte <= T0's own + (K+1)/R + eps; its text = T0."
  want nonstream && echo "   nonstream  case 3: the same with LONG not streamed."
  want stop && echo "   stop       case 4: LONG with stop [\"twenty\"]. PASS: finish stop, text = T up to the first 'twenty';
              the log's task line is 'task stopped at stop string' with tokens < 40; the next T0 text = T0."
  want cancel && echo "   cancel     case 5: LONG streamed; the task id is read from the first streamed frame, then
              POST /v1/cancel/<id> at D; the case 2 checks. A one-shot engine sends no frame before its final
              chunk, so no client can learn the id while it runs: then SKIP (the hang-up still ends the task)."
  want h1 && echo "   h1         stops [\"<|end|>\"], [\"<|assistant|>\"], [\"</s>\"] on LONG max_tokens $REF_MAX: each the stop rule below."
  want h2 && echo "   h2         stops [\"\\nThirty\"], [\" thirty\"] on LONG max_tokens $REF_MAX: the stop rule below."
  want h7 && echo "   h7         a CJK answer with stop [\"\\u3002\"] and the near miss [\"\\u3003\"]; an emoji answer with
              its first emoji as the stop and the next code point as a near miss: the stop rule below."
  want h8 && echo "   h8         /v1/completions, empty prompt, stop [\"\\n\"], max_tokens 32 -> text \"\", finish stop
              (and = base with --ab)."
  echo "              stop rule: a reference run with a never-matching stop gives Tr. If Tr holds the stop: finish
              stop, text = Tr up to the first stop, log 'task stopped at stop string' with tokens <= 1.25 x
              (Tr's tokens up to the stop) + K + 2. Else: no stop, log 'task done', text and finish = Tr.
              With --ab, the content and finish also equal the base node's (H4)."
  want ab && echo "   ab         case 6: LONG max_tokens 512, $AB_RUNS timed runs per node alternating; median tok/s
              (completion_tokens / time); PASS tip >= ${AB_BAR}% of base."
  want h6 && echo "   h6         the A/B with stops [\"zzzz-never\", <256-byte never-matching string>]."
  echo "   always     the node log holds no 'D8 invariant' line."
  exit 0
fi

[ -n "$URL" ] || { echo "--url is required (see --help)"; exit 2; }
[ -n "$LOG_FILE$LOG_CMD" ] || { echo "--log or --log-cmd is required (see --help)"; exit 2; }
PY=${PYTHON:-}
if [ -z "$PY" ]; then for p in python3 python; do "$p" -c 'import json' >/dev/null 2>&1 && { PY=$p; break; }; done; fi
[ -n "$PY" ] || { echo "python3 is required"; exit 2; }
command -v curl >/dev/null || { echo "curl is required"; exit 2; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t icc)
trap 'rm -rf "$WORK"' EXIT
FAILS=0; SEQ=0
verdict() { if [ "$2" = ok ]; then echo "PASS $1"; else echo "FAIL $1 (${3:-})"; FAILS=$((FAILS+1)); fi; }
skip() { echo "SKIP $1"; }

# --- JSON, SSE and text helpers (python; every argument and output is ASCII) ---
read -r -d '' PYH <<'PYEOF'
import json, math, os, sys
PROMPTS = {
    "t0": "Write one sentence about the sea.",
    "count": "Count from 1 to 100000 in words, separated by commas.",
    # "Write three sentences about the sea, in Chinese."
    "cjk": "\u7528\u4e2d\u6587\u5199\u4e09\u53e5\u8bdd\u4ecb\u7ecd\u5927\u6d77\u3002",
    "emoji": "Reply with five sea-themed emoji separated by spaces, then the word done.",
    "empty": "",
}
def stops(s):
    if s == "none": return None
    if s == "never": return ["zzzz-never"]
    if s == "never256": return ["zzzz-never", "zq" * 128]
    return json.loads(s)
def load(p):
    with open(p) as f: return json.load(f)
def short(t, n=90):
    return json.dumps(t)[:n]
cmd, a = sys.argv[1], sys.argv[2:]
if cmd == "body":
    kind, key, mx, stream, st = a
    b = {"model": os.environ.get("CK_MODEL", "")}
    if kind == "chat":
        b["messages"] = [{"role": "user", "content": PROMPTS[key]}]
        if os.environ.get("CK_THINK", "1") == "1": b["enable_thinking"] = False
    else:
        b["prompt"] = PROMPTS[key]
    b["temperature"] = 0
    if mx: b["max_tokens"] = int(mx)
    if stream == "1": b["stream"] = True; b["stream_options"] = {"include_usage": True}
    s = stops(st)
    if s is not None: b["stop"] = s
    print(json.dumps(b))
elif cmd == "parse":
    raw_p, stream, out = a
    try: raw = open(raw_p, "rb").read().decode("utf-8", "replace")
    except OSError: raw = ""
    r = {"text": "", "finish": None, "id": None, "completion_tokens": None, "done": False, "error": None, "frames": 0}
    def take(j):
        r["id"] = r["id"] or j.get("id")
        if j.get("error"): r["error"] = j["error"]
        for c in j.get("choices") or []:
            d = c.get("delta") or {}
            r["text"] += d.get("content") or ""
            m = c.get("message") or {}
            r["text"] += m.get("content") or ""
            r["text"] += c.get("text") or ""
            r["finish"] = c.get("finish_reason") or r["finish"]
        u = j.get("usage")
        if u: r["completion_tokens"] = u.get("completion_tokens")
    if stream == "1":
        for line in raw.splitlines():
            if not line.startswith("data: "): continue
            d = line[6:].strip()
            if d == "[DONE]": r["done"] = True; continue
            try: j = json.loads(d)
            except ValueError: continue
            r["frames"] += 1; take(j)
    else:
        try: take(json.loads(raw))
        except ValueError: r["error"] = "not JSON: " + short(raw, 160)
    with open(out, "w") as f: json.dump(r, f)
elif cmd == "get":
    v = load(a[0]).get(a[1])
    print("" if v is None else (int(v) if isinstance(v, bool) else v))
elif cmd == "show":
    print(short(load(a[0])["text"], int(a[1]) if len(a) > 1 else 90))
elif cmd == "same":
    x, y = load(a[0]), load(a[1])
    sys.exit(0 if (x["text"], x["finish"]) == (y["text"], y["finish"]) else 1)
elif cmd == "cut":
    ref, got, st = load(a[0]), load(a[1]), stops(a[2])
    t = ref["text"]; hits = [t.find(s) for s in st if s and t.find(s) >= 0]
    if hits:
        i = min(hits); want = t[:i]
        ok = got["text"] == want and got["finish"] == "stop"
        frac = (i + max(len(s) for s in st)) / max(1, len(t))
        est = math.ceil((ref.get("completion_tokens") or 0) * frac)
        print("ok" if ok else "bad", "match", est,
              "got=%s finish=%s want=...%s" % (short(got["text"][-60:], 70), got["finish"], short(want[-60:], 70)))
    else:
        ok = got["text"] == t and got["finish"] == ref["finish"]
        print("ok" if ok else "bad", "nomatch", 0,
              "got=%s finish=%s want=%s finish=%s" % (short(got["text"], 70), got["finish"], short(t, 70), ref["finish"]))
elif cmd == "emoji":
    t = load(a[0])["text"]
    c = next((ch for ch in t if ord(ch) >= 0x1F000 or 0x2600 <= ord(ch) <= 0x27BF), None)
    print("" if c is None else json.dumps([c]) + " " + json.dumps([chr(ord(c) + 1)]))
elif cmd == "model":
    try: print(json.load(open(a[0]))["data"][0]["id"])
    except Exception: print("")
elif cmd == "median":
    v = sorted(float(x) for x in a); n = len(v)
    print("%.3f" % ((v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2) if n else 0))
PYEOF
py() { "$PY" -I -c "$PYH" "$@"; }

# --- float math (awk) and time ---
calc() { awk "BEGIN { printf \"%.3f\", ($1) }"; }
fle() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 <= b + 0) }'; }
now() { if [ -n "${EPOCHREALTIME:-}" ]; then echo "${EPOCHREALTIME/,/.}"; else "$PY" -c 'import time; print("%.3f" % time.time())'; fi; }

# --- the node log ---
ESC=$(printf '\033')
log_dump() {
  if [ -n "$LOG_FILE" ]; then cat -- "$LOG_FILE" 2>/dev/null; else bash -c "$LOG_CMD" 2>/dev/null; fi \
    | tr -d '\r' | sed "s/${ESC}\[[0-9;]*m//g"
}
log_lines() { log_dump | wc -l | tr -d ' '; }
TERM_RE='task (cancelled|done|stopped at stop string)'
term_since() { log_dump | tail -n +"$(( $1 + 1 ))" | grep -E "$TERM_RE"; }
wait_term() { # offset count timeout_s -> the task lines since offset (waits for <count> of them)
  local out="" t_end=$((SECONDS + $3))
  while :; do
    out=$(term_since "$1")
    [ "$(printf '%s' "$out" | grep -c .)" -ge "$2" ] && break
    [ "$SECONDS" -ge "$t_end" ] && break
    sleep 0.5
  done
  printf '%s\n' "$out"
}
field() { printf '%s\n' "$1" | sed -nE "s/.*[[:space:]]$2=([^[:space:]]+).*/\1/p" | head -1; }
kind_of() {
  case "$1" in
    *"task cancelled"*) echo cancelled ;;
    *"task stopped at stop string"*) echo stopped ;;
    *"task done"*) echo "done" ;;
    *) echo none ;;
  esac
}

# --- requests ---
# req <name> <base-url> <chat|cmpl> <prompt key> <max_tokens|""> <stream 0|1> <stops> <client limit s>
# writes $WORK/<name>.{body,raw,w,json}; .w = "ttfb total http_code curl_exit"
req() {
  local n=$1 base=$2 kind=$3 key=$4 mx=$5 st=$6 sp=$7 mt=$8 ep=/v1/chat/completions m=$MODEL
  [ "$kind" = cmpl ] && ep=/v1/completions
  [ "$base" = "$AB_URL" ] && m=$AB_MODEL
  CK_MODEL=$m CK_THINK=$THINK py body "$kind" "$key" "$mx" "$st" "$sp" > "$WORK/$n.body"
  curl -s -N --max-time "$mt" -H 'content-type: application/json' --data-binary @- \
    -o "$WORK/$n.raw" -w '%{time_starttransfer} %{time_total} %{http_code}' "$base$ep" \
    < "$WORK/$n.body" > "$WORK/$n.w"
  echo " $?" >> "$WORK/$n.w"
  py parse "$WORK/$n.raw" "$st" "$WORK/$n.json"
}
wget_() { local _a _b _c _d; read -r _a _b _c _d < "$WORK/$1.w"; case "$2" in ttfb) echo "$_a" ;; total) echo "$_b" ;; code) echo "$_c" ;; rc) echo "$_d" ;; esac; }
nx() { SEQ=$((SEQ + 1)); NX="r$SEQ"; }  # sets NX (no subshell, so SEQ advances)

health() { curl -s -o "$WORK/models.$2" --max-time 10 "$1/v1/models" -w '%{http_code}'; }

# --- the shared cancel checks ---
check_cancel_log() { # label offset follow-up-name
  local lab=$1 off=$2 fol=$3 lines first fid k tid el n nexp tol bound
  if [ "$NO_TASK_LOG" = 1 ]; then skip "$lab: engine task-line checks (--no-task-log)"; return; fi
  lines=$(wait_term "$off" 2 "$LOG_WAIT"); first=$(printf '%s\n' "$lines" | head -1)
  fid=$(py get "$WORK/$fol.json" id); k=$(kind_of "$first"); tid=$(field "$first" task)
  if [ "$k" = cancelled ] && [ "$tid" != "$fid" ]; then
    verdict "$lab: the long task's log line is 'task cancelled' (not 'task done')" ok
  else
    verdict "$lab: the long task's log line is 'task cancelled'" bad "next task line: ${first:-none within ${LOG_WAIT}s}"
    return
  fi
  el=$(field "$first" elapsed_s); n=$(field "$first" tokens)
  bound=$(calc "$D + $ROUND / $R + $EPS")
  fle "${el:-1e9}" "$bound" \
    && verdict "$lab: cancelled within $ROUND token time(s) of the hang-up (elapsed_s=$el <= D + $ROUND/R + eps = $bound)" ok \
    || verdict "$lab: cancelled within $ROUND token time(s) of the hang-up" bad "elapsed_s=${el:-?} > $bound"
  nexp=$(calc "$R * $D"); tol=$(calc "$ROUND + (2 > $NTOL / 100 * $nexp ? 2 : $NTOL / 100 * $nexp)")
  awk -v n="${n:-x}" -v e="$nexp" -v t="$tol" 'BEGIN { d = n - e; if (d < 0) d = -d; exit !(n ~ /^[0-9]+$/ && d <= t) }' \
    && verdict "$lab: tokens=$n ~ D*R = $nexp (+-$tol)" ok \
    || verdict "$lab: tokens ~ D*R" bad "tokens=${n:-?} vs $nexp +-$tol"
}
check_followup() { # label follow-up-name
  local lab=$1 fol=$2 tt code bound
  tt=$(wget_ "$fol" ttfb); code=$(wget_ "$fol" code)
  bound=$(calc "$TTFB0 + ($ROUND + 1) / $R + $EPS")
  [ "$code" = 200 ] && fle "$tt" "$bound" \
    && verdict "$lab: the next request's first byte at ${tt}s <= T0's ${TTFB0}s + $((ROUND + 1)) token times + eps = ${bound}s" ok \
    || verdict "$lab: the next request starts at once" bad "HTTP $code, first byte ${tt}s > ${bound}s"
  py same "$WORK/T0.json" "$WORK/$fol.json" \
    && verdict "$lab: the next request's text = T0" ok \
    || verdict "$lab: the next request's text = T0" bad "got $(py show "$WORK/$fol.json") want $(py show "$WORK/T0.json")"
}

# --- cases ---
case_baseline() {
  local tps tl ct line off
  req T0a "$URL" chat t0 "$T0_MAX" 1 none "$SETTLE"
  req T0 "$URL" chat t0 "$T0_MAX" 1 none "$SETTLE"
  py same "$WORK/T0a.json" "$WORK/T0.json" && [ -n "$(py get "$WORK/T0.json" text)" ] \
    && verdict "baseline: T0 is the same twice at temperature 0 ($(py show "$WORK/T0.json" 60))" ok \
    || verdict "baseline: T0 is deterministic" bad "$(py show "$WORK/T0a.json") vs $(py show "$WORK/T0.json")"
  TTFB0=$(wget_ T0 ttfb)
  off=$(log_lines)
  req Rrun "$URL" chat count 128 0 none "$SETTLE"
  line=""; [ "$NO_TASK_LOG" = 1 ] || line=$(wait_term "$off" 1 "$LOG_WAIT" | head -1)
  tps=$(field "$line" tok_s)
  if [ -n "$tps" ] && fle 0.01 "$tps"; then
    R=$tps; echo "baseline: R = $R tok/s (log: ${line#*INFO })"
  else
    ct=$(py get "$WORK/Rrun.json" completion_tokens); tl=$(wget_ Rrun total)
    R=$(calc "${ct:-0} / ($tl > 0 ? $tl : 1)"); echo "baseline: R = $R tok/s ($ct tokens / ${tl}s client time; no tok_s in the log)"
  fi
  fle 0.01 "$R" || { verdict "baseline: a decode rate" bad "R=$R"; R=1; }
  D=$(calc "(20 / $R > 3) ? 20 / $R : 3")
  echo "baseline: D = max(3, 20/R) = ${D}s, one token time = $(calc "1 / $R")s, T0 first byte ${TTFB0}s"
  req T "$URL" chat count "$REF_MAX" 0 never "$SETTLE"
  echo "baseline: reference T ($(py get "$WORK/T.json" completion_tokens) tokens, $(py get "$WORK/T.json" finish)): $(py show "$WORK/T.json" 100)"
}

case_disconnect() { # label stream(0|1)
  local lab=$1 st=$2 off d f rc
  nx; d=$NX; nx; f=$NX
  off=$(log_lines)
  req "$d" "$URL" chat count "$LONG_MAX" "$st" none "$D"
  rc=$(wget_ "$d" rc)
  sleep 0.1
  req "$f" "$URL" chat t0 "$T0_MAX" 1 none "$SETTLE"
  [ "$rc" = 28 ] && verdict "$lab: the client hung up at D=${D}s while the long request ran" ok \
    || verdict "$lab: the client hung up mid-generation" bad "curl exit $rc (28 = timed out at D); the long request ended before D or failed"
  check_cancel_log "$lab" "$off" "$f"
  check_followup "$lab" "$f"
}

case_stop() {
  local lab="case 4 stop" s f off v mode est detail line k n
  nx; s=$NX; nx; f=$NX
  off=$(log_lines)
  req "$s" "$URL" chat count "$LONG_MAX" 0 '["twenty"]' "$LONG_TIMEOUT"
  read -r v mode est detail <<<"$(py cut "$WORK/T.json" "$WORK/$s.json" '["twenty"]')"
  [ "$v" = ok ] && [ "$mode" = match ] \
    && verdict "$lab: finish stop, the text ends right before 'twenty' (= T up to it)" ok \
    || verdict "$lab: finish stop, text ends before 'twenty'" bad "$mode $detail; curl exit $(wget_ "$s" rc)"
  if [ "$NO_TASK_LOG" = 1 ]; then skip "$lab: engine task-line check (--no-task-log)"; else
    line=$(wait_term "$off" 1 "$LOG_WAIT" | head -1); k=$(kind_of "$line"); n=$(field "$line" tokens)
    [ "$k" = stopped ] && [ -n "$n" ] && [ "$n" -lt 40 ] 2>/dev/null \
      && verdict "$lab: the log shows 'task stopped at stop string' tokens=$n < 40" ok \
      || verdict "$lab: 'task stopped at stop string' with tokens < 40" bad "next task line: ${line:-none}"
  fi
  req "$f" "$URL" chat t0 "$T0_MAX" 1 none "$SETTLE"
  py same "$WORK/T0.json" "$WORK/$f.json" && verdict "$lab: the next request's text = T0" ok \
    || verdict "$lab: the next request's text = T0" bad "got $(py show "$WORK/$f.json")"
}

case_cancel() {
  local lab="case 5 /v1/cancel" c f off cpid t0 id="" rem code t_end
  nx; c=$NX; nx; f=$NX
  off=$(log_lines)
  CK_MODEL=$MODEL CK_THINK=$THINK py body chat count "$LONG_MAX" 1 none > "$WORK/$c.body"
  curl -s -N --max-time "$SETTLE" -H 'content-type: application/json' --data-binary @- \
    -o "$WORK/$c.raw" "$URL/v1/chat/completions" < "$WORK/$c.body" &
  cpid=$!; t0=$(now)
  while :; do
    if grep -q '^data: {' "$WORK/$c.raw" 2>/dev/null; then
      py parse "$WORK/$c.raw" 1 "$WORK/$c.json"; id=$(py get "$WORK/$c.json" id); [ -n "$id" ] && break
    fi
    fle "$D" "$(calc "$(now) - $t0")" && break
    kill -0 "$cpid" 2>/dev/null || break
    sleep 0.2
  done
  if [ -z "$id" ]; then
    kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
    skip "$lab: no streamed frame within D=${D}s, so a client cannot learn the task id (a one-shot engine sends its id only in its final chunk; /v1/cancel/:task_id needs it). The hang-up ends the task instead."
    [ "$NO_TASK_LOG" = 1 ] || wait_term "$off" 1 "$SETTLE" > /dev/null
    return
  fi
  rem=$(calc "$D - ($(now) - $t0)"); fle 0.01 "$rem" && sleep "$rem"
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST --max-time 10 "$URL/v1/cancel/$id")
  sleep 0.1
  req "$f" "$URL" chat t0 "$T0_MAX" 1 none "$SETTLE"
  t_end=$((SECONDS + LOG_WAIT))
  while kill -0 "$cpid" 2>/dev/null && [ "$SECONDS" -lt "$t_end" ]; do sleep 0.2; done
  if kill -0 "$cpid" 2>/dev/null; then
    kill "$cpid" 2>/dev/null; verdict "$lab: the cancelled stream ends" bad "still open ${LOG_WAIT}s after the cancel"
  else
    verdict "$lab: POST /v1/cancel/$id -> HTTP $code and the stream ended" "$([ "$code" = 200 ] && echo ok || echo bad)" "HTTP $code"
  fi
  wait "$cpid" 2>/dev/null
  check_cancel_log "$lab" "$off" "$f"
  check_followup "$lab" "$f"
}

stop_case() { # label kind prompt-key max_tokens stops ref-name
  local lab=$1 kind=$2 key=$3 mx=$4 sp=$5 ref=$6 s b off v mode est detail line k n lim
  nx; s=$NX
  off=$(log_lines)
  req "$s" "$URL" "$kind" "$key" "$mx" 0 "$sp" "$LONG_TIMEOUT"
  read -r v mode est detail <<<"$(py cut "$WORK/$ref.json" "$WORK/$s.json" "$sp")"
  if [ "$mode" = match ]; then
    verdict "$lab: finish stop, text = the reference up to the stop" "$v" "$detail"
  else
    verdict "$lab: the reference has no such stop: no stop fired, text and finish = the reference" "$v" "$detail"
  fi
  if [ "$NO_TASK_LOG" != 1 ]; then
    line=$(wait_term "$off" 1 "$LOG_WAIT" | head -1); k=$(kind_of "$line"); n=$(field "$line" tokens)
    if [ "$mode" = match ]; then
      lim=$(calc "1.25 * $est + $ROUND + 2")
      [ "$k" = stopped ] && fle "${n:-1e9}" "$lim" \
        && verdict "$lab: log 'task stopped at stop string' tokens=$n <= $lim" ok \
        || verdict "$lab: log 'task stopped at stop string' with small tokens" bad "next task line: ${line:-none}; limit $lim"
    else
      [ "$k" = "done" ] && verdict "$lab: log 'task done' (no engine stop)" ok \
        || verdict "$lab: log 'task done' (no engine stop)" bad "next task line: ${line:-none}"
    fi
  fi
  if [ -n "$AB_URL" ]; then
    nx; b=$NX; req "$b" "$AB_URL" "$kind" "$key" "$mx" 0 "$sp" "$LONG_TIMEOUT"
    py same "$WORK/$s.json" "$WORK/$b.json" && verdict "H4 $lab: content and finish byte-identical to the base node" ok \
      || verdict "H4 $lab: same as the base node" bad "tip $(py show "$WORK/$s.json") base $(py show "$WORK/$b.json")"
  fi
}

case_h1() {
  local s; for s in '["<|end|>"]' '["<|assistant|>"]' '["</s>"]'; do
    stop_case "H1 special-token stop $s" chat count "$REF_MAX" "$s" T
  done
}
case_h2() {
  local s; for s in '["\nThirty"]' '[" thirty"]'; do
    stop_case "H2 SentencePiece stop $s" chat count "$REF_MAX" "$s" T
  done
}
case_h7() {
  local pick st near
  req Tcjk "$URL" chat cjk 96 0 never "$SETTLE"
  echo "H7 CJK reference: $(py show "$WORK/Tcjk.json" 100)"
  stop_case 'H7 CJK stop ["\u3002"]' chat cjk 96 '["\u3002"]' Tcjk
  stop_case 'H7 CJK near-miss stop ["\u3003"]' chat cjk 96 '["\u3003"]' Tcjk
  req Temo "$URL" chat emoji 64 0 never "$SETTLE"
  echo "H7 emoji reference: $(py show "$WORK/Temo.json" 100)"
  pick=$(py emoji "$WORK/Temo.json")
  if [ -z "$pick" ]; then skip "H7 emoji stop: the reference answer holds no emoji"; return; fi
  st=${pick%% *}; near=${pick#* }
  stop_case "H7 emoji stop $st" chat emoji 64 "$st" Temo
  stop_case "H7 emoji near-miss stop $near" chat emoji 64 "$near" Temo
}
case_h8() {
  local s b txt fin
  nx; s=$NX; req "$s" "$URL" cmpl empty 32 0 '["\n"]' "$LONG_TIMEOUT"
  txt=$(py show "$WORK/$s.json"); fin=$(py get "$WORK/$s.json" finish)
  [ "$txt" = '""' ] && [ "$fin" = stop ] && [ "$(wget_ "$s" code)" = 200 ] \
    && verdict 'H8 completions, empty prompt, stop ["\n"]: text "" and finish stop' ok \
    || verdict 'H8 completions, empty prompt, stop ["\n"]: text "" and finish stop' bad "HTTP $(wget_ "$s" code) text $txt finish $fin"
  if [ -n "$AB_URL" ]; then
    nx; b=$NX; req "$b" "$AB_URL" cmpl empty 32 0 '["\n"]' "$LONG_TIMEOUT"
    py same "$WORK/$s.json" "$WORK/$b.json" && verdict "H8: same as the base node" ok \
      || verdict "H8: same as the base node" bad "tip $txt base $(py show "$WORK/$b.json")"
  fi
}

ab_case() { # label stops
  local lab=$1 sp=$2 i n tt tb ct tl mt mb ratio
  req abw1 "$URL" chat count 64 0 "$sp" "$SETTLE"; req abw2 "$AB_URL" chat count 64 0 "$sp" "$SETTLE"
  tt=(); tb=()
  for i in $(seq 1 "$AB_RUNS"); do
    for n in tip base; do
      if [ "$n" = tip ]; then req "ab$n$i" "$URL" chat count 512 0 "$sp" "$SETTLE"
      else req "ab$n$i" "$AB_URL" chat count 512 0 "$sp" "$SETTLE"; fi
      ct=$(py get "$WORK/ab$n$i.json" completion_tokens); tl=$(wget_ "ab$n$i" total)
      if [ "$n" = tip ]; then tt+=("$(calc "${ct:-0} / ($tl > 0 ? $tl : 1)")"); else tb+=("$(calc "${ct:-0} / ($tl > 0 ? $tl : 1)")"); fi
      echo "$lab run $i $n: ${ct:-?} tokens in ${tl}s"
    done
  done
  mt=$(py median "${tt[@]}"); mb=$(py median "${tb[@]}")
  ratio=$(calc "($mb > 0) ? 100 * $mt / $mb : 0")
  fle "$AB_BAR" "$ratio" \
    && verdict "$lab: tip ${mt} tok/s >= ${AB_BAR}% of base ${mb} tok/s (${ratio}%)" ok \
    || verdict "$lab: tip >= ${AB_BAR}% of base" bad "tip ${mt} base ${mb} = ${ratio}% (runs tip ${tt[*]} base ${tb[*]})"
}

# --- run ---
[ "$(health "$URL" tip)" = 200 ] || { echo "node at $URL does not answer /v1/models"; echo "INFLIGHT CANCEL CHECK: 1 FAILED"; exit 1; }
[ -n "$MODEL" ] || MODEL=$(py model "$WORK/models.tip")
if [ -n "$AB_URL" ]; then
  [ "$(health "$AB_URL" base)" = 200 ] || { echo "base node at $AB_URL does not answer /v1/models"; echo "INFLIGHT CANCEL CHECK: 1 FAILED"; exit 1; }
  AB_MODEL=$(py model "$WORK/models.base")
fi
log_dump | grep -q . || echo "warning: the node log is empty so far (${LOG_FILE:-$LOG_CMD})"
echo "node $URL model $MODEL${AB_URL:+ ; base node $AB_URL model $AB_MODEL}; K=$ROUND eps=${EPS}s long max_tokens=${LONG_MAX:-server default}"
startup=$(log_dump | grep -m1 'request defaults'); [ -n "$startup" ] && echo "startup: ${startup#*INFO }"

R=1; D=3; TTFB0=0
[ "$NEED_BASE" = 1 ] && case_baseline
want stream && case_disconnect "case 2 streamed disconnect" 1
want nonstream && case_disconnect "case 3 non-streamed disconnect" 0
want stop && case_stop
want cancel && case_cancel
want h1 && case_h1
want h2 && case_h2
want h7 && case_h7
want h8 && case_h8
want ab && ab_case "case 6 A/B tok/s" none
want h6 && ab_case "H6 A/B tok/s with stops active" never256

N8=$(log_dump | grep -c 'D8 invariant')
[ "$N8" = 0 ] && verdict "the node log has no 'D8 invariant' line" ok \
  || verdict "the node log has no 'D8 invariant' line" bad "$N8 line(s): $(log_dump | grep -m1 'D8 invariant')"

if [ "$FAILS" = 0 ]; then echo "INFLIGHT CANCEL CHECK: ALL PASS"; else echo "INFLIGHT CANCEL CHECK: $FAILS FAILED"; exit 1; fi
