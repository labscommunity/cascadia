#!/usr/bin/env bash
# responses-clients-check.sh — live check of Cascadia's POST /v1/responses with real clients.
#
# Starts Cascadia on the mock engine (tests/fixtures/mock-tools-model, no model weights) and runs each client's
# tool loop against it (see docs/RESPONSES.md "Testing" for what the mock does):
#   agents  OpenAI Agents SDK (openai-agents 0.24.0) tool agent          AGENTS_SDK_TOOL_LOOP
#   aisdk   @ai-sdk/openai 4.0.111 multi-step loop + item_reference     AISDK_TOOL_LOOP_AND_ITEM_REFERENCE
#   n8n     n8n 2.46.6 AI Agent + OpenAI Chat Model, Responses API on   N8N_RESPONSES_AGENT_TOOL_LOOP
#   codex   Codex CLI 0.166.0 (codex on PATH) exec_command round trip   CODEX_MOCK_TOOL_LOOP (SKIP if not 0.166.0)
# Prints one verdict line per check, then RESPONSES_CLIENTS: ALL PASS (every check ran and passed, exit 0),
# RESPONSES_CLIENTS: PASS (skipped: <names>) (exit 0) or RESPONSES_CLIENTS: FAIL <names> (exit 1).
#
# Everything installed or written lives under WORK (default: a new temp dir; set WORK=... to reuse one and skip
# reinstalls): the Python venv, node_modules, the n8n home (N8N_USER_FOLDER, so ~/.n8n is never touched) and
# CODEX_HOME (so ~/.codex is never touched). Package-manager download caches are the user's defaults.
# scripts/responses_clients/max_tokens_proxy.py is for the hardware Codex run against a real model; this
# script does not use it.
#
# --url BASE checks an already-running server instead (one serving the same mock script; --bin is then unused),
# --api-key-env NAME sends the value of $NAME as every client's API key (Bearer; passed through the environment,
# never printed or put on a command line), and --model ID picks the model id (default mock-tools-model).
#
# Usage: scripts/responses-clients-check.sh [--bin target/debug/cascadia] [--port 18080]
#                                           [--url http://HOST:PORT] [--api-key-env NAME] [--model ID]
#                                           [--only agents,aisdk,n8n,codex] [--dry-run]
# Env: WORK (work dir), PYTHON (interpreter for the venv, default python3).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$REPO/scripts/responses_clients"
BIN="target/debug/cascadia"
PORT=18080
ONLY="agents,aisdk,n8n,codex"
DRY_RUN=0
URL=""
KEY_ENV=""
MODEL="mock-tools-model"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --api-key-env) KEY_ENV="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for c in ${ONLY//,/ }; do
  case "$c" in agents|aisdk|n8n|codex) ;; *) echo "unknown check in --only: $c" >&2; exit 2 ;; esac
done
case "$BIN" in /*) ;; *) BIN="$REPO/$BIN" ;; esac
if [[ -n "$KEY_ENV" ]]; then
  [[ "$KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "--api-key-env takes a variable name: $KEY_ENV" >&2; exit 2; }
  if [[ $DRY_RUN == 0 && -z "${!KEY_ENV:-}" ]]; then echo "\$$KEY_ENV is empty or unset" >&2; exit 2; fi
  # The helpers and Codex read the key from this variable by name; only the name is ever passed or printed.
  export "${KEY_ENV?}"
  export RESPONSES_CHECK_KEY_ENV="$KEY_ENV"
else
  # The helpers send a key whenever this is set, so one inherited from the caller must not reach a keyless run.
  unset RESPONSES_CHECK_KEY_ENV
fi

PYTHON="${PYTHON:-python3}"
AGENTS_VER="openai-agents==0.24.0"
AISDK_PKGS=(ai@7.0.165 @ai-sdk/openai@4.0.111 zod@4.6.7)
N8N_PKG="n8n@2.46.6"
CODEX_VER="codex-cli 0.166.0"
if [[ -n "$URL" ]]; then
  URL="${URL%/}"; URL="${URL%/v1}"
  BASE="$URL/v1"
else
  BASE="http://127.0.0.1:$PORT/v1"
fi

if [[ $DRY_RUN == 1 ]]; then WORK="${WORK:-<mktemp -d>}"; else WORK="${WORK:-$(mktemp -d)}"; fi

# Print a step; run it unless --dry-run.
step() { printf '+ %s\n' "$*"; if [[ $DRY_RUN == 0 ]]; then "$@"; fi; }
note() { printf '== %s\n' "$*"; }

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

start_server() {
  if [[ -n "$URL" ]]; then note "use the running server at $BASE"; return 0; fi
  note "start Cascadia mock on 127.0.0.1:$PORT (log: $WORK/cascadia.log)"
  printf '+ %s run tests/fixtures/mock-tools-model --engine mock --api 127.0.0.1:%s\n' "$BIN" "$PORT"
  [[ $DRY_RUN == 1 ]] && { echo "+ wait for $BASE/models"; return 0; }
  [[ -x "$BIN" ]] || { echo "cascadia binary not found: $BIN (build it: cargo build -p cascadia)" >&2; exit 1; }
  if curl -fsS -o /dev/null "$BASE/models" 2>/dev/null; then
    echo "port $PORT already serves $BASE/models; stop that server or pass --port" >&2; exit 1
  fi
  (cd "$REPO" && exec "$BIN" run tests/fixtures/mock-tools-model --engine mock --api "127.0.0.1:$PORT") \
    > "$WORK/cascadia.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    if curl -fsS -o /dev/null "$BASE/models" 2>/dev/null; then echo "server up"; return 0; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "cascadia exited; tail of its log:" >&2; tail -20 "$WORK/cascadia.log" >&2; exit 1; }
    sleep 0.5
  done
  echo "cascadia did not answer $BASE/models within 60s" >&2; tail -20 "$WORK/cascadia.log" >&2; exit 1
}

check_agents() {
  local venv="$WORK/agents-venv"
  if [[ ! -x "$venv/bin/python" ]]; then
    step "$PYTHON" -m venv "$venv"
    step "$venv/bin/python" -m pip install -q --disable-pip-version-check "$AGENTS_VER"
  fi
  step "$venv/bin/python" -I "$HERE/agents_sdk_check.py" "$BASE" "$MODEL"
}

check_aisdk() {
  local dir="$WORK/aisdk"
  step mkdir -p "$dir"
  if [[ ! -d "$dir/node_modules/@ai-sdk/openai" ]]; then
    step npm install --prefix "$dir" --no-audit --no-fund --loglevel=error "${AISDK_PKGS[@]}"
  fi
  # Node resolves imports from the script's own directory, so run a copy that sits next to node_modules.
  step cp "$HERE/aisdk_check.mjs" "$dir/aisdk_check.mjs"
  step node "$dir/aisdk_check.mjs" "$BASE" "$MODEL"
}

n8n_cmd() {
  N8N_USER_FOLDER="$WORK/n8n-home" N8N_DIAGNOSTICS_ENABLED=false N8N_RUNNERS_ENABLED=true \
    N8N_BLOCK_ENV_ACCESS_IN_NODE=false "$WORK/n8n/node_modules/.bin/n8n" "$@"
}

n8n_execute() { n8n_cmd execute --id v3n8nagent --rawOutput > "$WORK/n8n/execute.out" 2>"$WORK/n8n/execute.err" || true; }

check_n8n() {
  local dir="$WORK/n8n"
  step mkdir -p "$dir" "$WORK/n8n-home"
  if [[ ! -x "$dir/node_modules/.bin/n8n" ]]; then
    note "installing $N8N_PKG (takes a few minutes)"
    step env N8N_USER_FOLDER="$WORK/n8n-home" npm install --prefix "$dir" --no-audit --no-fund --loglevel=error "$N8N_PKG"
  fi
  step "$PYTHON" -I "$HERE/n8n_workflow.py" gen "$BASE" "$dir/creds.json" "$dir/wf.json" "$MODEL"
  printf '+ n8n import:credentials / import:workflow / execute  (N8N_USER_FOLDER=%s)\n' "$WORK/n8n-home"
  if [[ $DRY_RUN == 0 ]]; then
    n8n_cmd import:credentials --input "$dir/creds.json" > "$dir/import.log" 2>&1
    n8n_cmd import:workflow --input "$dir/wf.json" >> "$dir/import.log" 2>&1
    n8n_execute
  fi
  step "$PYTHON" -I "$HERE/n8n_workflow.py" verify "$dir/execute.out"
}

codex_provider() {
  local env_key=""
  [[ -n "$KEY_ENV" ]] && env_key=',env_key="'"$KEY_ENV"'"'
  printf 'model_providers.cascadia={name="cascadia",base_url="%s",wire_api="responses"%s}' "$BASE" "$env_key"
}

codex_run() {
  (cd "$WORK/codex-work" && CODEX_HOME="$WORK/codex-home" codex exec --skip-git-repo-check \
    -c "$(codex_provider)" \
    -c model_provider=cascadia -m "$MODEL" "__mock_tool_call__" < /dev/null > "$WORK/codex-out.txt" 2>&1) || true
}

check_codex() {
  # Turn 1: the mock calls Codex's exec_command; Codex runs it. Turn 2: Codex replays reasoning, function_call
  # and function_call_output; the mock answers "mock: got ...".
  local have=""
  if command -v codex >/dev/null 2>&1; then have="$(codex --version 2>/dev/null || true)"; fi
  if [[ $DRY_RUN == 0 && "$have" != "$CODEX_VER" ]]; then
    echo "CODEX_MOCK_TOOL_LOOP: SKIP ($CODEX_VER not on PATH)"; return 0
  fi
  step mkdir -p "$WORK/codex-work" "$WORK/codex-home"
  printf '+ (cd %s && CODEX_HOME=%s codex exec --skip-git-repo-check -c %s -c model_provider=cascadia -m %s "__mock_tool_call__")\n' \
    "$WORK/codex-work" "$WORK/codex-home" "$(codex_provider)" "$MODEL"
  [[ $DRY_RUN == 1 ]] && { echo "+ require \"mock: got\" in its output"; return 0; }
  codex_run
  grep -n "exec\|mock" "$WORK/codex-out.txt" | head -12 || true
  if grep -q "mock: got" "$WORK/codex-out.txt"; then echo "CODEX_MOCK_TOOL_LOOP: PASS"; else echo "CODEX_MOCK_TOOL_LOOP: FAIL"; return 1; fi
}

verdict_name() {
  case "$1" in
    agents) echo AGENTS_SDK_TOOL_LOOP ;; aisdk) echo AISDK_TOOL_LOOP_AND_ITEM_REFERENCE ;;
    n8n) echo N8N_RESPONSES_AGENT_TOOL_LOOP ;; codex) echo CODEX_MOCK_TOOL_LOOP ;;
  esac
}

note "WORK=$WORK"
[[ $DRY_RUN == 1 ]] || mkdir -p "$WORK/logs"
start_server

VERDICTS=(); FAILED=(); SKIPPED=()
for c in ${ONLY//,/ }; do
  note "check: $c"
  name="$(verdict_name "$c")"
  if [[ $DRY_RUN == 1 ]]; then "check_$c"; continue; fi
  log="$WORK/logs/$c.log"
  set +e; "check_$c" 2>&1 | tee "$log"; set -e
  line="$(grep -E "^$name: (PASS|FAIL|SKIP)" "$log" | tail -1 || true)"
  [[ -n "$line" ]] || line="$name: FAIL (no verdict; see $log)"
  VERDICTS+=("$line")
  case "$line" in *": PASS"*) ;; *": SKIP"*) SKIPPED+=("$c") ;; *) FAILED+=("$c") ;; esac
done

[[ $DRY_RUN == 1 ]] && { note "dry run: nothing was run"; exit 0; }
echo
note "verdicts (logs in $WORK/logs)"
for v in "${VERDICTS[@]}"; do echo "$v"; done
if [[ ${#FAILED[@]} -eq 0 && ${#SKIPPED[@]} -eq 0 ]]; then
  echo "RESPONSES_CLIENTS: ALL PASS"
elif [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "RESPONSES_CLIENTS: PASS (skipped: ${SKIPPED[*]})"
else
  echo "RESPONSES_CLIENTS: FAIL ${FAILED[*]}"; exit 1
fi
