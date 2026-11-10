#!/usr/bin/env bash
# Real client libraries against the mock engine: each must classify the server's over-window 400 as
# context overflow by its own detector, and openai-python must retry a capacity 503 using Retry-After.
# Clients: openai-python, LiteLLM, pydantic-ai, crewAI (each in its own venv, at the versions pinned
# below) and Cline's error classifier (its source at a pinned commit, run by node; skipped without node).
# usage: scripts/openai-errors-clients.sh <path-to-cascadia-binary> [port]
#   needs python3 (uv if present, which also fetches a Python crewAI supports; else python3 -m venv)
#   and network for pip/npm. Everything is installed into a temp dir that is removed on exit.
# Prints one verdict line per claim (PASS/FAIL/SKIP) and "OPENAI ERRORS CLIENTS: ALL PASS" or
# "OPENAI ERRORS CLIENTS: N FAILED" (exit 1). SKIP lines don't fail.
set -uo pipefail
BIN=$(cd "$(dirname "${1:?cascadia binary}")" && pwd)/$(basename "$1")
PORT=${2:-18491}
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
PID=""
cleanup() {
  [ -n "$PID" ] && { kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
  rm -rf "$WORK"
}
trap cleanup EXIT

# Pinned client versions; bump deliberately. openai-python comes from the conformance suite's pin.
# Each library resolves its own openai dependency, so each gets its own venv; the openai pins
# beside them are the versions those resolved to when this was written.
PY_VERSION=3.12
LITELLM="litellm==1.106.6 openai==2.54.0"
PYDANTIC_AI="pydantic-ai-slim[openai]==2.63.0 openai==3.49.0"
CREWAI="crewai==1.15.33 openai==2.54.0"
CLINE_COMMIT=94eecbbe1a2f899c66b118d13656651b0a884e1f
CLINE_NPM="ai@7.0.155 @ai-sdk/openai-compatible@3.0.80 jsonrepair@3.13.2"

OUT="$WORK/verdicts"; : > "$OUT"
say() { echo "$1" | tee -a "$OUT"; }

# A model dir: a large window (so the capacity check's long streams keep running) and a minimal chat template.
M="$WORK/model"; mkdir -p "$M/tokenizer"
echo '{"max_position_embeddings": 1000000}' > "$M/config.json"
cat > "$M/tokenizer/tokenizer_config.json" <<'EOF'
{"chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}", "bos_token": "", "eos_token": ""}
EOF
"$BIN" run "$M" --engine mock --api "127.0.0.1:$PORT" > "$WORK/server.log" 2>&1 &
PID=$!
for _ in $(seq 1 100); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.1; done
curl -sf "127.0.0.1:$PORT/health" >/dev/null || { echo "server did not start"; cat "$WORK/server.log"; exit 1; }
BASE="http://127.0.0.1:$PORT/v1"
export BASE PORT
# LiteLLM fetches its model price list from the network at import unless told to use its bundled copy.
export LITELLM_LOCAL_MODEL_COST_MAP=True

# venv NAME PACKAGES...: a venv at $WORK/NAME with the packages installed; prints nothing on success.
venv() {
  local dir="$WORK/$1"; shift
  if command -v uv >/dev/null; then
    uv venv -q --python "$PY_VERSION" "$dir" && uv pip install -q --python "$dir/bin/python" "$@"
  else
    python3 -m venv "$dir" && "$dir/bin/pip" install -q "$@"
  fi
}
# client NAME: run the Python check below for that client in its venv; it prints its own verdict lines.
# Only the verdict lines are shown: LiteLLM and crewAI print their own banners for the error.
client() { "$WORK/$1/bin/python" -I "$WORK/clients.py" "$1" 2>"$WORK/$1.err" | grep -E '^(PASS|FAIL|SKIP) ' | tee -a "$OUT"
  [ "${PIPESTATUS[0]}" = 0 ] || say "FAIL $1 (check crashed: $(tail -3 "$WORK/$1.err" | tr '\n' ' '))"; }

cat > "$WORK/clients.py" <<'EOF'
import os, socket, sys, threading, time

BASE, PORT = os.environ["BASE"], int(os.environ["PORT"])
LONG = "__prompt_too_long__"
# The mock refuses a prompt containing __prompt_too_long__ as 9000 tokens in an 8192-token window.
SENTENCE = "This model's maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens."
MSGS = [{"role": "user", "content": LONG}]


def verdict(name, ok, why=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" ({why})"), flush=True)


def hold_capacity(n=16):
    """Fill the in-flight cap (16 by default): streamed requests on a long prompt whose clients never read."""
    import json
    body = json.dumps({"model": "m", "stream": True, "messages": [
        {"role": "user", "content": " ".join("w%d" % i for i in range(100000))}]}).encode()
    held = []
    for _ in range(n):
        s = socket.create_connection(("127.0.0.1", PORT))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % len(body) + body)
        held.append(s)
    time.sleep(1)
    return held


def check_openai():
    import openai
    c = openai.OpenAI(base_url=BASE, api_key="x", max_retries=0)
    try:
        c.chat.completions.create(model="m", messages=MSGS)
        verdict("openai-python: over-window chat raises BadRequestError", False, "no error raised")
    except openai.BadRequestError as e:
        ok = e.status_code == 400 and e.code == "context_length_exceeded" and e.param == "messages" and SENTENCE in e.message
        verdict(f"openai-python {openai.__version__}: over-window chat raises BadRequestError, code context_length_exceeded, param messages, OpenAI's sentence",
                ok, f"code={e.code} param={e.param} message={e.message}")

    try:
        stream = c.chat.completions.create(model="m", messages=MSGS, stream=True)
        list(stream)
        verdict("openai-python: over-window streamed chat raises BadRequestError", False, "no error raised")
    except openai.BadRequestError as e:
        verdict("openai-python: over-window streamed chat raises BadRequestError, code context_length_exceeded",
                e.code == "context_length_exceeded", f"code={e.code} message={e.message}")

    # Count the server's 503s on the chat route from its /metrics, before and after.
    def served_503():
        import re, urllib.request
        text = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics").read().decode()
        m = re.search(r'cascadia_http_requests_total\{endpoint="/v1/chat/completions",status="503"\} (\d+)', text)
        return int(m.group(1)) if m else 0

    c = openai.OpenAI(base_url=BASE, api_key="x", max_retries=2)
    held = hold_capacity()
    before, t0 = served_503(), time.monotonic()
    try:
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "a"}])
        verdict("openai-python: capacity 503 retried, then APIStatusError 503", False, "no error raised")
    except openai.APIStatusError as e:
        took, n = time.monotonic() - t0, served_503() - before
        ok = e.status_code == 503 and n == 3 and took >= 1.9
        verdict(f"openai-python: capacity 503 retried twice after Retry-After: 1 (server answered {n} 503s in {took:.1f}s), then APIStatusError 503",
                ok, f"status={e.status_code} 503s={n} took={took:.2f}s")
    finally:
        for s in held:
            s.close()
    time.sleep(1)


def check_litellm():
    import litellm
    from importlib.metadata import version
    try:
        litellm.completion(model="openai/m", api_base=BASE, api_key="x", messages=MSGS, num_retries=0)
        verdict("LiteLLM: over-window raises ContextWindowExceededError", False, "no error raised")
    except litellm.ContextWindowExceededError as e:
        verdict(f"LiteLLM {version('litellm')}: over-window raises litellm.ContextWindowExceededError", SENTENCE in str(e), str(e))
    except Exception as e:
        verdict("LiteLLM: over-window raises ContextWindowExceededError", False, f"{type(e).__name__}: {e}")


def check_pydantic_ai():
    from importlib.metadata import version
    from pydantic_ai import Agent, ModelContextWindowExceededError
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    model = OpenAIChatModel("m", provider=OpenAIProvider(base_url=BASE, api_key="x"))
    agent = Agent(model)
    try:
        agent.run_sync(LONG)
        verdict("pydantic-ai: over-window raises ModelContextWindowExceededError", False, "no error raised")
    except ModelContextWindowExceededError as e:
        verdict(f"pydantic-ai {version('pydantic-ai-slim')}: Agent + OpenAIChatModel over-window raises ModelContextWindowExceededError", True)
    except Exception as e:
        verdict("pydantic-ai: over-window raises ModelContextWindowExceededError", False, f"{type(e).__name__}: {e}")


def check_crewai():
    from importlib.metadata import version
    from crewai import LLM
    from crewai.utilities.exceptions.context_window_exceeding_exception import LLMContextLengthExceededError
    llm = LLM(model="openai/m", base_url=BASE, api_key="x", max_retries=0)
    try:
        llm.call(LONG)
        verdict("crewAI: over-window raises LLMContextLengthExceededError", False, "no error raised")
    except LLMContextLengthExceededError as e:
        verdict(f"crewAI {version('crewai')}: LLM.call over-window raises LLMContextLengthExceededError", SENTENCE in e.original_error_message, e.original_error_message)
        verdict("crewAI: its detector (_is_context_length_exceeded_error) is true on the server's message",
                LLMContextLengthExceededError._is_context_length_exceeded_error(e.original_error_message) is True)
    except Exception as e:
        verdict("crewAI: over-window raises LLMContextLengthExceededError", False, f"{type(e).__name__}: {e}")


{"openai": check_openai, "litellm": check_litellm, "pydantic-ai": check_pydantic_ai, "crewai": check_crewai}[sys.argv[1]]()
EOF

if venv openai -r "$REPO/tests/conformance/requirements-openai-sdk.txt" 2>"$WORK/i.err"; then client openai; else say "FAIL openai install ($(tail -2 "$WORK/i.err" | tr '\n' ' '))"; fi
# shellcheck disable=SC2086
if venv litellm $LITELLM 2>"$WORK/i.err"; then client litellm; else say "FAIL litellm install ($(tail -2 "$WORK/i.err" | tr '\n' ' '))"; fi
# shellcheck disable=SC2086
if venv pydantic-ai $PYDANTIC_AI 2>"$WORK/i.err"; then client pydantic-ai; else say "FAIL pydantic-ai install ($(tail -2 "$WORK/i.err" | tr '\n' ' '))"; fi
# shellcheck disable=SC2086
if venv crewai $CREWAI 2>"$WORK/i.err"; then client crewai; else say "FAIL crewai install ($(tail -2 "$WORK/i.err" | tr '\n' ' '))"; fi

# Cline: its own classifier (sdk/packages/llms/src/providers/error-classification.ts) run on the error
# the Vercel AI SDK's openai-compatible provider (what Cline's openai-compatible provider uses) raises.
NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)
if [ "$NODE_MAJOR" = 0 ]; then
  say "SKIP cline (node not found)"
elif [ "$NODE_MAJOR" -lt 23 ]; then
  say "SKIP cline (node $NODE_MAJOR; running Cline's TypeScript source needs node 23 or newer)"
else
  C="$WORK/cline"; mkdir -p "$C"
  RAW="https://raw.githubusercontent.com/cline/cline/$CLINE_COMMIT/sdk/packages"
  echo '{"type": "module", "private": true}' > "$C/package.json"
  # shellcheck disable=SC2086
  if curl -sfL "$RAW/llms/src/providers/error-classification.ts" -o "$C/error-classification.ts" \
     && curl -sfL "$RAW/shared/src/parse/json.ts" -o "$C/json.ts" \
     && (cd "$C" && npm install -s --no-audit --no-fund $CLINE_NPM >"$WORK/npm.log" 2>&1); then
    # Its only workspace import is safeJsonParse from @cline/shared, which lives in json.ts.
    sed -i.orig 's#from "@cline/shared"#from "./json.ts"#' "$C/error-classification.ts"
    cat > "$C/check.mjs" <<'EOF'
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText, streamText } from "ai";
import { classifyProviderError, isRetryableProviderError } from "./error-classification.ts";
const p = createOpenAICompatible({ name: "cascadia", baseURL: process.env.BASE, apiKey: "x", includeUsage: true });
const model = p.chatModel("m");
const out = (name, ok, why) => console.log((ok ? "PASS " : "FAIL ") + name + (ok ? "" : ` (${why})`));
try {
  await generateText({ model, prompt: "__prompt_too_long__", maxRetries: 0 });
  out("cline: over-window classified context_window_exceeded", false, "no error raised");
} catch (e) {
  const v = classifyProviderError(e);
  out(`cline ${process.env.CLINE_COMMIT.slice(0, 8)}: classifyProviderError on the over-window error -> context_window_exceeded, not retryable`,
      v === "context_window_exceeded" && !isRetryableProviderError(e), `${v} retryable=${isRetryableProviderError(e)} ${e?.message}`);
}
let streamErr;
const r = streamText({ model, prompt: "__prompt_too_long__", maxRetries: 0, onError: ({ error }) => { streamErr = error; } });
try { for await (const _ of r.textStream) {} } catch (e) { streamErr ??= e; }
const sv = streamErr && classifyProviderError(streamErr);
out("cline: streamed over-window classified context_window_exceeded", sv === "context_window_exceeded", `${sv} ${streamErr?.message}`);
EOF
    (cd "$C" && CLINE_COMMIT=$CLINE_COMMIT node --experimental-strip-types --no-warnings check.mjs 2>"$WORK/cline.err") | tee -a "$OUT"
    [ "${PIPESTATUS[0]}" = 0 ] || say "FAIL cline (check crashed: $(tail -3 "$WORK/cline.err" | tr '\n' ' '))"
  else
    say "FAIL cline setup (download or npm install failed: $(tail -3 "$WORK/npm.log" 2>/dev/null | tr '\n' ' '))"
  fi
fi

FAILS=$(grep -c '^FAIL' "$OUT")
if [ "$FAILS" = 0 ]; then echo "OPENAI ERRORS CLIENTS: ALL PASS"; else echo "OPENAI ERRORS CLIENTS: $FAILS FAILED"; exit 1; fi
