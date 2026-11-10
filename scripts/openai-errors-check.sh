#!/usr/bin/env bash
# OpenAI error responses against the mock engine: the error envelope on bad bodies, unknown routes and
# wrong methods, the body limit, the capacity 503, refused and out-of-range fields with `param`, the
# over-window 400 (`context_length_exceeded`) on every route, and an engine failure mid-stream.
# usage: scripts/openai-errors-check.sh <path-to-cascadia-binary> [port]   (needs curl, jq, python3)
# Prints one verdict line per claim (PASS/FAIL) and "OPENAI ERRORS CHECK: ALL PASS" or "OPENAI ERRORS CHECK: N FAILED" (exit 1).
# shellcheck disable=SC2015  # `test && verdict ok || verdict bad`: verdict ok always succeeds.
set -uo pipefail
BIN=${1:?cascadia binary}
PORT=${2:-18481}
WORK=$(mktemp -d)
FAILS=0
PID=""
HOLD=""
cleanup() {
  [ -n "$HOLD" ] && kill "$HOLD" 2>/dev/null
  [ -n "$PID" ] && { kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
  rm -rf "$WORK"
}
trap cleanup EXIT
verdict() { if [ "$2" = ok ]; then echo "PASS $1"; else echo "FAIL $1 ($3)"; FAILS=$((FAILS+1)); fi; }

# A model dir: a large window (so the capacity check's long streams keep running) and a minimal chat template.
M="$WORK/model"; mkdir -p "$M/tokenizer"
echo '{"max_position_embeddings": 1000000}' > "$M/config.json"
cat > "$M/tokenizer/tokenizer_config.json" <<'EOF'
{"chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}", "bos_token": "", "eos_token": ""}
EOF

start() { # $@ = extra flags
  "$BIN" run "$M" --engine mock --api "127.0.0.1:$PORT" "$@" > "$WORK/server.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 100); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return 0; sleep 0.1; done
  echo "server did not start"; cat "$WORK/server.log"; exit 1
}
stop() { kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=""; }

# req METHOD PATH [BODY] [extra curl args...]: sets $CODE, $BODY (response body) and $HDRS (response headers).
# Every response whose body has an "error" key is kept for the envelope check at the end.
N=0
req() {
  local method=$1 path=$2 data=${3-}; shift 3 2>/dev/null || shift $#
  N=$((N+1))
  local args=(-s -X "$method" -D "$WORK/h$N" -o "$WORK/b$N" -w '%{http_code}')
  [ -n "$data" ] && args+=(--data-binary "$data")
  CODE=$(curl "${args[@]}" "$@" "127.0.0.1:$PORT$path")
  BODY=$(cat "$WORK/b$N"); HDRS=$(tr -d '\r' < "$WORK/h$N")
  jq -e 'type == "object" and has("error")' <<<"$BODY" >/dev/null 2>&1 && echo "$N $method $path" >> "$WORK/errors"
}
post() { req POST "$1" "$2" -H 'content-type: application/json'; }
ctype_json() { grep -qi '^content-type: application/json' <<<"$HDRS"; }
etype() { jq -r '.error.type' <<<"$BODY"; }
eparam() { jq -r '.error.param' <<<"$BODY"; }
ecode() { jq -r '.error.code' <<<"$BODY"; }
emsg() { jq -r '.error.message' <<<"$BODY"; }
short() { printf '%s' "HTTP $CODE $(head -c 300 <<<"$BODY")"; }
# is_err STATUS TYPE: status, JSON content type and the error type.
is_err() { [ "$CODE" = "$1" ] && ctype_json && [ "$(etype)" = "$2" ]; }
chat() { printf '{"model":"m","messages":[{"role":"user","content":"%s"}]%s}' "$1" "${2-}"; }

start --api-max-body-mb 1

# --- Malformed requests --------------------------------------------------------------------------
post /v1/chat/completions '{"model":"m","messages":['
{ is_err 400 invalid_request_error && [[ "$(emsg)" == "request body is not valid JSON"* ]]; } \
  && verdict "bad JSON -> 400 envelope, 'request body is not valid JSON: ...'" ok || verdict "bad JSON" bad "$(short)"

req POST /v1/chat/completions "$(chat a)"
{ is_err 400 invalid_request_error && [[ "$(emsg)" == *'Content-Type: application/json'* ]]; } \
  && verdict "missing Content-Type -> 400 envelope naming Content-Type: application/json" ok || verdict "missing Content-Type" bad "$(short)"

post /v1/chat/completions "$(chat a ',"temperature":"hot"')"
{ is_err 400 invalid_request_error && [[ "$(emsg)" == *temperature* ]] && [[ "$(emsg)" != *"Failed to deserialize"* ]]; } \
  && verdict "wrong field type (temperature: \"hot\") -> 400 envelope, serde message without the axum wrapper" ok || verdict "wrong field type" bad "$(short)"

python3 -I -c "import json; print(json.dumps({'model':'m','messages':[{'role':'user','content':'x'*(2*1024*1024)}]}))" > "$WORK/2mib.json"
req POST /v1/chat/completions "@$WORK/2mib.json" -H 'content-type: application/json'
is_err 413 invalid_request_error \
  && verdict "2 MiB body over --api-max-body-mb 1 -> 413 envelope" ok || verdict "body over the limit" bad "$(short)"

post /v1/embeddings '{"model":"m","input":"a"}'
{ is_err 404 invalid_request_error && [ "$(emsg)" = "Invalid URL (POST /v1/embeddings)" ]; } \
  && verdict "POST /v1/embeddings -> 404 'Invalid URL (POST /v1/embeddings)'" ok || verdict "unknown route" bad "$(short)"

req GET /v1/chat/completions ""
{ is_err 405 invalid_request_error && [ "$(emsg)" = "Method GET is not allowed for /v1/chat/completions" ] && grep -qi '^allow:.*POST' <<<"$HDRS"; } \
  && verdict "GET /v1/chat/completions -> 405 'Method GET is not allowed for /v1/chat/completions', Allow: POST" ok || verdict "wrong method" bad "$(short) allow=$(grep -i '^allow:' <<<"$HDRS")"

# --- Capacity --------------------------------------------------------------------------------------
# `run` has no flag for the in-flight cap, so fill it: 16 streamed requests (the API's default cap) on
# a 100000-word prompt whose clients never read, so their streams stay open and hold every slot.
python3 -I -c "import json; print(json.dumps({'model':'m','stream':True,'messages':[{'role':'user','content':' '.join('w%d' % i for i in range(100000))}]}))" > "$WORK/long.json"
python3 -I - "$PORT" 16 "$WORK/long.json" <<'EOF' > "$WORK/hold.log" 2>&1 &
import socket, sys, time
port, n, body = int(sys.argv[1]), int(sys.argv[2]), open(sys.argv[3], "rb").read()
held = []
for _ in range(n):
    s = socket.create_connection(("127.0.0.1", port))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(body) + body)
    held.append(s)
print("holding", n, flush=True)
time.sleep(120)
EOF
HOLD=$!
for _ in $(seq 1 50); do grep -q holding "$WORK/hold.log" && break; sleep 0.1; done
sleep 1
post /v1/chat/completions "$(chat a)"
CAP_CODE=$CODE; CAP_BODY=$BODY; CAP_RA=$(grep -i '^retry-after:' <<<"$HDRS" | tr -d ' ' | cut -d: -f2)
{ is_err 503 server_error && [ "$CAP_RA" = 1 ] && [[ "$(emsg)" == "engine at capacity"* ]]; } && CAP=ok || CAP=bad
H=$(curl -s -o /dev/null -w '%{http_code}' "127.0.0.1:$PORT/health")
kill "$HOLD" 2>/dev/null; wait "$HOLD" 2>/dev/null; HOLD=""
sleep 1
H2=$(curl -s -o /dev/null -w '%{http_code}' "127.0.0.1:$PORT/health")
{ [ "$CAP" = ok ] && [ "$H" = 200 ] && [ "$H2" = 200 ]; } \
  && verdict "at capacity -> 503 server_error + Retry-After: 1; /health stays 200 (during and after)" ok \
  || verdict "capacity 503" bad "HTTP $CAP_CODE retry-after=$CAP_RA $CAP_BODY health=$H/$H2"

# --- Refused and out-of-range fields ---------------------------------------------------------------
while IFS='|' read -r extra param; do
  post /v1/chat/completions "$(chat a ",$extra")"
  { is_err 400 invalid_request_error && [ "$(eparam)" = "$param" ]; } \
    && verdict "chat $extra -> 400 param $param" ok || verdict "chat $extra" bad "$(short)"
done <<'EOF'
"temperature":2.5|temperature
"top_p":1.5|top_p
"frequency_penalty":3|frequency_penalty
"n":2|n
"top_logprobs":21|top_logprobs
"reasoning_effort":"turbo"|reasoning_effort
"logit_bias":{"50256":-100}|logit_bias
"modalities":["text","audio"]|modalities
"audio":{}|audio
"web_search_options":{}|web_search_options
"tools":[{"type":"custom","custom":{"name":"grammar"}}]|tools
EOF

post /v1/chat/completions "$(chat a ',"temperature":2.5,"stream":true')"
{ is_err 400 invalid_request_error && [ "$(eparam)" = temperature ] && ! grep -q '^data:' <<<"$BODY"; } \
  && verdict "streamed request with temperature 2.5 -> JSON 400 param temperature (not SSE)" ok || verdict "streamed refusal" bad "$(short)"

# --- Accepted no-op forms --------------------------------------------------------------------------
while IFS= read -r extra; do
  post /v1/chat/completions "$(chat a ",$extra")"
  { [ "$CODE" = 200 ] && [ "$(jq -r .object <<<"$BODY")" = chat.completion ]; } \
    && verdict "chat $extra -> 200" ok || verdict "chat $extra" bad "$(short)"
done <<'EOF'
"n":1
"logit_bias":{}
"modalities":["text"]
"audio":null
"response_format":{"type":"json_object"}
"reasoning_effort":"none"
"reasoning_effort":"minimal"
"reasoning_effort":"low"
"reasoning_effort":"medium"
"reasoning_effort":"high"
"reasoning_effort":"xhigh"
"reasoning_effort":"max"
EOF
post /v1/chat/completions "$(chat a ',"stream":true,"stream_options":{"include_usage":true}')"
{ [ "$CODE" = 200 ] && grep -qi '^content-type: text/event-stream' <<<"$HDRS" && grep -q 'data: \[DONE\]' <<<"$BODY"; } \
  && verdict "chat \"stream\":true,\"stream_options\":{\"include_usage\":true} -> 200 stream" ok || verdict "stream_options" bad "$(short)"

# Many stop sequences are accepted (editors send several fill-in-the-middle stop tokens).
STOP9='"stop":["s1","s2","s3","s4","s5","s6","s7","s8","s9"]'
post /v1/chat/completions "$(chat a ",$STOP9")"
{ [ "$CODE" = 200 ] && [ "$(jq -r .object <<<"$BODY")" = chat.completion ]; } \
  && verdict "chat with 9 stop strings -> 200" ok || verdict "chat with 9 stop strings" bad "$(short)"
post /v1/completions "{\"model\":\"m\",\"prompt\":\"a\",$STOP9}"
{ [ "$CODE" = 200 ] && [ "$(jq -r .object <<<"$BODY")" = text_completion ]; } \
  && verdict "/v1/completions with 9 stop strings -> 200" ok || verdict "/v1/completions with 9 stop strings" bad "$(short)"

# --- Other routes ----------------------------------------------------------------------------------
post /v1/completions '{"model":"m","prompt":"a","temperature":3}'
{ is_err 400 invalid_request_error && [ "$(eparam)" = temperature ]; } \
  && verdict "/v1/completions temperature 3 -> 400 param temperature" ok || verdict "/v1/completions temperature 3" bad "$(short)"
post /v1/responses '{"model":"m","input":"a","reasoning":{"effort":"turbo"}}'
{ is_err 400 invalid_request_error && [ "$(eparam)" = reasoning.effort ]; } \
  && verdict "/v1/responses reasoning.effort turbo -> 400 param reasoning.effort" ok || verdict "/v1/responses reasoning.effort" bad "$(short)"

# --- Over the window (the mock refuses a prompt containing __prompt_too_long__ as 9000 tokens in 8192) ---
for stream in false true; do
  for route in "/v1/chat/completions|messages|$(chat __prompt_too_long__ ",\"stream\":$stream")" \
               "/v1/completions|prompt|{\"model\":\"m\",\"prompt\":\"__prompt_too_long__\",\"stream\":$stream}" \
               "/v1/responses|input|{\"model\":\"m\",\"input\":\"__prompt_too_long__\",\"stream\":$stream}"; do
    IFS='|' read -r path noun body <<<"$route"
    post "$path" "$body"
    want="This model's maximum context length is 8192 tokens. However, your $noun resulted in 9000 tokens."
    { is_err 400 invalid_request_error && [ "$(ecode)" = context_length_exceeded ] && [ "$(eparam)" = "$noun" ] && [[ "$(emsg)" == "$want"* ]]; } \
      && verdict "over-window $path stream=$stream -> 400 context_length_exceeded, param $noun, OpenAI's sentence" ok \
      || verdict "over-window $path stream=$stream" bad "$(short)"
  done
done

# --- Engine failure (the mock fails a prompt containing __engine_error__ at its first step) ----------
post /v1/chat/completions "$(chat __engine_error__ ',"stream":true')"
FRAMES=$(grep '^data: ' <<<"$BODY" | sed 's/^data: //')
ERRF=$(grep -m1 '"error"' <<<"$FRAMES")
LAST=$(tail -1 <<<"$FRAMES")
{ [ "$CODE" = 200 ] && jq -e '(.id | type) == "string" and .object == "error" and .error.type == "server_error"
      and (.error.message | type) == "string" and .error.param == null and .error.code == null
      and (.error | has("param") and has("code"))' <<<"$ERRF" >/dev/null && [ "$LAST" = "[DONE]" ]; } \
  && verdict "engine failure mid-stream -> one error frame (object error, server_error, param/code null), then [DONE]" ok \
  || verdict "engine failure mid-stream" bad "HTTP $CODE $(head -c 400 <<<"$BODY")"
[ -n "$ERRF" ] && jq -c '{error}' <<<"$ERRF" > "$WORK/b_stream_error" && echo "stream_error frame" >> "$WORK/errors"

post /v1/chat/completions "$(chat __engine_error__)"
is_err 503 server_error \
  && verdict "engine failure, not streamed -> 503 server_error" ok || verdict "engine failure, not streamed" bad "$(short)"

stop

# --- Every error body above is OpenAI's envelope -----------------------------------------------------
BAD=""; COUNT=0
while read -r n rest; do
  f="$WORK/b$n"; [ "$n" = stream_error ] && f="$WORK/b_stream_error"
  COUNT=$((COUNT+1))
  jq -e '(keys == ["error"]) and (.error | type == "object" and has("message") and has("type") and has("param") and has("code"))
         and (.error.message | type) == "string" and (.error.type | type) == "string"' "$f" >/dev/null 2>&1 || BAD="$BAD [$rest: $(head -c 200 "$f")]"
done < "$WORK/errors"
[ -z "$BAD" ] && [ "$COUNT" -gt 0 ] \
  && verdict "all $COUNT error bodies are {\"error\":{message,type,param,code}} with all four keys" ok || verdict "error envelope shape" bad "$BAD"

if [ "$FAILS" = 0 ]; then echo "OPENAI ERRORS CHECK: ALL PASS"; else echo "OPENAI ERRORS CHECK: $FAILS FAILED"; exit 1; fi
