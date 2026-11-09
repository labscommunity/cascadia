#!/usr/bin/env bash
# OpenAI request handling against the mock engine: max_completion_tokens, request defaults and the
# restore flags, accepted-and-ignored fields, deprecated functions, early stop on `stop`.
# usage: scripts/openai-requests-check.sh <path-to-cascadia-binary> [port]   (needs curl, jq, python3)
# Prints one verdict line per claim (PASS/FAIL) and "OPENAI REQUESTS CHECK: ALL PASS" or "OPENAI REQUESTS CHECK: N FAILED" (exit 1).
set -uo pipefail
BIN=${1:?cascadia binary}
PORT=${2:-18471}
WORK=$(mktemp -d)
FAILS=0
verdict() { if [ "$2" = ok ]; then echo "PASS $1"; else echo "FAIL $1 ($3)"; FAILS=$((FAILS+1)); fi; }

# A model dir: 300-token window, temperature 0.7 / top_p 0.9, and a minimal chat template.
M="$WORK/model"; mkdir -p "$M"
echo '{"max_position_embeddings": 300}' > "$M/config.json"
echo '{"temperature": 0.7, "top_p": 0.9, "do_sample": true}' > "$M/generation_config.json"
mkdir -p "$M/tokenizer"; cat > "$M/tokenizer/tokenizer_config.json" <<'EOF'
{"chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}", "bos_token": "", "eos_token": ""}
EOF

start() { # $@ = extra flags
  "$BIN" run "$M" --engine mock --api "127.0.0.1:$PORT" "$@" > "$WORK/server.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 100); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && return 0; sleep 0.1; done
  echo "server did not start"; cat "$WORK/server.log"; exit 1
}
stop() { kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
post() { curl -s -H 'content-type: application/json' "127.0.0.1:$PORT$1" -d "$2"; }
words() { python3 -I -c "print(' '.join('w%d' % i for i in range($1)))"; }

start
LINE=$(sed 's/\x1b\[[0-9;]*m//g' "$WORK/server.log" | grep -m1 'max_tokens=')
echo "startup: $LINE"
[[ "$LINE" == *"max_tokens=300"* && "$LINE" == *"config.json"* && "$LINE" == *"temperature=0.69"* && "$LINE" == *"top_p=0.89"* && "$LINE" == *"generation_config.json"* ]] \
  && verdict "defaults: window 300 from config.json, temperature 0.7 / top_p 0.9 from generation_config.json" ok \
  || verdict "defaults from the model's files" bad "$LINE"

R=$(post /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"alpha bravo charlie delta echo"}],"max_completion_tokens":2}')
[ "$(jq -r '.usage.completion_tokens' <<<"$R")" = 2 ] && [ "$(jq -r '.choices[0].finish_reason' <<<"$R")" = length ] \
  && verdict "max_completion_tokens: 2 -> 2 completion tokens, length" ok || verdict "max_completion_tokens: 2" bad "$R"

R=$(post /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"alpha bravo charlie delta echo"}],"max_tokens":2}')
[ "$(jq -r '.usage.completion_tokens' <<<"$R")" = 2 ] && verdict "max_tokens: 2 still works" ok || verdict "max_tokens: 2" bad "$R"

R=$(post /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"a b"}],"max_tokens":4,"user":"u","safety_identifier":"s","prompt_cache_key":"k","prompt_cache_retention":"24h","metadata":{"k":"v"},"store":true,"service_tier":"auto","verbosity":"low","prediction":{"type":"content","content":"x"},"stream_options":{"include_obfuscation":false}}')
[ "$(jq -r '.object' <<<"$R")" = chat.completion ] && verdict "harmless fields incl. store: true -> 200" ok || verdict "harmless fields" bad "$R"

# The mock reports its prompt's words as prompt_tokens and echoes them: a 200-word prompt would
# answer 200 tokens, but the prompt plus the answer reach the 300-token window after 100.
R=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$(words 200)\"}]}")
[ "$(jq -r '.usage.prompt_tokens + .usage.completion_tokens' <<<"$R")" = 300 ] && [ "$(jq -r '.usage.total_tokens' <<<"$R")" = 300 ] && [ "$(jq -r '.choices[0].finish_reason' <<<"$R")" = length ] \
  && verdict "no limit: the answer runs until prompt + answer fill the 300-token window (length)" ok || verdict "no limit ends at the window" bad "$(jq -c '.usage,.choices[0].finish_reason' <<<"$R")"

R=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$(words 50)\"}],\"max_tokens\":100000}")
[ "$(jq -r '.usage.completion_tokens' <<<"$R")" = 50 ] && verdict "explicit max_tokens 100000 accepted (clamped to the window)" ok || verdict "explicit large limit" bad "$R"

TC='<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>'
R=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$TC\"}],\"functions\":[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}]}")
[ "$(jq -r '.choices[0].message.function_call.name' <<<"$R")" = get_weather ] && [ "$(jq -r '.choices[0].finish_reason' <<<"$R")" = function_call ] \
  && verdict "deprecated functions -> legacy message.function_call, finish_reason function_call" ok || verdict "functions mapping" bad "$R"

R=$(curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' "127.0.0.1:$PORT/v1/chat/completions" -d '{"model":"m","messages":[{"role":"user","content":"a"}],"functions":[{"name":"f"}],"tools":[{"type":"function","function":{"name":"f"}}]}')
[ "$R" = 400 ] && verdict "functions + tools -> 400" ok || verdict "functions + tools" bad "HTTP $R"

R=$(post /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"a b c STOP d e f"}],"stop":["STOP"]}')
[ "$(jq -r '.choices[0].message.content' <<<"$R")" = "a b c " ] && [ "$(jq -r '.choices[0].finish_reason' <<<"$R")" = stop ] && [ "$(jq -r '.usage.prompt_tokens' <<<"$R")" -gt 0 ] \
  && verdict "stop ends early: content cut, finish stop, prompt_tokens > 0" ok || verdict "early stop (non-stream)" bad "$R"

R=$(post /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"a b c STOP d e f"}],"stop":["STOP"],"stream":true,"stream_options":{"include_usage":true}}')
U=$(grep '^data: {' <<<"$R" | sed 's/^data: //' | jq -c 'select(.usage != null) | .usage' | tail -1)
{ grep -q 'data: \[DONE\]' <<<"$R" && [ "$(jq -r '.prompt_tokens' <<<"$U")" -gt 0 ] && ! grep -q 'STOP' <<<"$R"; } \
  && verdict "streamed stop: no stop text sent, usage prompt_tokens > 0, [DONE]" ok || verdict "early stop (stream)" bad "$R"

R=$(post /v1/completions "{\"model\":\"m\",\"prompt\":\"$(words 200)\"}")
[ "$(jq -r '.usage.prompt_tokens + .usage.completion_tokens' <<<"$R")" = 300 ] && [ "$(jq -r '.choices[0].finish_reason' <<<"$R")" = length ] \
  && verdict "/v1/completions: same default, ends when prompt + answer fill the window (length)" ok || verdict "/v1/completions default" bad "$(jq -c '.usage,.choices[0].finish_reason' <<<"$R")"
stop

# A 1000-token window here, so a 400-word prompt plus a 256-token answer fit and the 256 cap shows.
echo '{"max_position_embeddings": 1000}' > "$M/config.json"
start --default-max-tokens 256 --default-temperature 0 --default-top-p 1
LINE=$(sed 's/\x1b\[[0-9;]*m//g' "$WORK/server.log" | grep -m1 'max_tokens=')
echo "startup (restore flags): $LINE"
[[ "$LINE" == *"max_tokens=256"* && "$LINE" == *"temperature=0.0 "* && "$LINE" == *"top_p=1.0 "* && "$LINE" == *'max_tokens_source="--default-max-tokens"'* ]] \
  && verdict "restore flags: 256 / 0 / 1 from the flags" ok || verdict "restore flags" bad "$LINE"
R=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$(words 400)\"}]}")
[ "$(jq -r '.usage.completion_tokens' <<<"$R")" = 256 ] && [ "$(jq -r '.choices[0].finish_reason' <<<"$R")" = length ] \
  && verdict "restore flags: no limit -> 256 tokens, length (the old output)" ok || verdict "restore flags 256" bad "$(jq -c .usage <<<"$R")"
stop
rm -rf "$WORK"

if [ "$FAILS" = 0 ]; then echo "OPENAI REQUESTS CHECK: ALL PASS"; else echo "OPENAI REQUESTS CHECK: $FAILS FAILED"; exit 1; fi
