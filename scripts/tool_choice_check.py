#!/usr/bin/env python3
"""Live check of `tool_choice`, `allowed_tools`, custom tools and `parallel_tool_calls`.

Sends a fixed set of chat requests to an OpenAI-compatible base URL and prints one
`VERDICT <case> PASS|FAIL|INFO …` line per check, then `TOOL CHOICE CHECK: ALL PASS` or
`TOOL CHOICE CHECK: FAILED (<n>: <names>)`. Exit code 0 only when every check passes
(1 on a failure, 2 on bad arguments). Uses only the standard library (Python 3.10+).

Usage:
    python3 scripts/tool_choice_check.py http://127.0.0.1:8000 [--mock] [--gemma] [--expect-fallback]
                                         [--concurrency N] [--cases a,b,…] [--dry-run]

Cases (`--cases` takes these names; "+stream" means a streamed variant runs too):
    required          tool_choice "required" → at least one call to a declared tool,
                      finish "tool_calls" (+stream: every delta carries `index`, the first
                      delta of each call carries id, type and function.name)
    named_function    a named function on an unrelated prompt → exactly that one call,
                      finish "stop" (+stream)
    named_custom      a named custom tool → one {"id","type":"custom","custom":{"name","input"}}
                      with no `function` key, finish "stop" (+stream: one delta in that shape plus `index`)
    custom_replay     a second turn replaying a custom call and its tool result → HTTP 200
    allowed_required  allowed_tools mode "required" over two of three tools → a call, never
                      to the tool left out
    allowed_auto      allowed_tools mode "auto" → HTTP 200, never a call to the tool left out
    allowed_bad_name  allowed_tools listing an undeclared name → HTTP 400 naming tool_choice
    parallel_false    parallel_tool_calls false on a two-call prompt → exactly one call (+stream:
                      one distinct index); a control with true must make two or more
    json_union        response_format json_schema with callable tools, tool_choice auto: a
                      tool prompt → a call or schema-valid content; "Say hello." → schema-valid
                      content (a call is INFO)
    plain_unchanged   tools without tool_choice → 200, content or a function call, no custom objects
    concurrency       (with --concurrency N > 1) N `required` requests at once, each returns a call
    gemma             (with --gemma) a Gemma 4 model under auto → get_weather with a "city"
                      argument, and no Gemma tool-call markers left in the content (+stream)

Every request sends "max_tokens": 512, "temperature": 0 and a top-level "enable_thinking": false.

`--mock` is for `cascadia run mock-model --engine mock`. The mock answers a constrained request
with the minimal instance of the grammar (the first `anyOf` branch, so the first tool, and
`minItems` calls) and echoes the prompt otherwise. Forced cases (required, named_function,
named_custom, custom_replay, allowed_*, parallel_false's false half, plain_unchanged,
concurrency) must pass. These report INFO there because the mock can't produce them:
parallel_false's true control (one call is the minimal instance), json_union's "Say hello."
half (the first branch of the union is the call array, so it answers with a call) and gemma (no
grammar under auto, so the mock echoes the prompt). Tools need a chat template: a mock model
without one answers every tools request with a 400, which fails the run.

`--expect-fallback` is for a node that drops the forced-call grammar and answers unconstrained
(ov-genai on NPU, or started with a draft model). Every request must still answer HTTP 200 (the
bad-name 400 aside) and parse; a check that fails only because the output wasn't constrained
(no call, a call count, a tool name or finish_reason the grammar would have fixed, content that
isn't the schema's JSON) reports INFO "not enforced on this node (expected)". Malformed call
objects, broken stream deltas, calls to a tool that allowed_tools left out and a second call
under parallel_tool_calls false (the server trims those itself) still FAIL.

`--dry-run` prints every request body instead of sending it and checks nothing.
"""

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

BASE = ""
MOCK = False
FALLBACK = False
DRY_RUN = False
DRY_COUNT = 0
FAILED = []
NOT_ENFORCED = "not enforced on this node (expected)"


def fn_tool(name, desc, props=("city",)):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in props},
                "required": list(props),
            },
        },
    }


GET_WEATHER = fn_tool("get_weather", "Get the current weather for a city")
GET_TIME = fn_tool("get_time", "Get the current local time in a city")
GET_STOCK = fn_tool("get_stock_price", "Get the latest stock price for a ticker", ("ticker",))
APPLY_PATCH = {"type": "custom", "custom": {"name": "apply_patch", "description": "Apply a patch"}}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
GEMMA_MARKERS = ("<|tool_call>", "<tool_call|>", '<|"|>', "call:get_weather")


# ---------------------------------------------------------------- HTTP


def post(body, timeout=900):
    """Return (status, raw body, seconds). status 0 = no HTTP answer (connection error)."""
    global DRY_COUNT
    if DRY_RUN:
        DRY_COUNT += 1
        print(f"DRY-RUN POST {BASE.rstrip('/')}/v1/chat/completions {json.dumps(body)}", flush=True)
        if body.get("stream"):
            return 200, "data: [DONE]\n", 0.0
        return 200, json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}), 0.0
    req = urllib.request.Request(
        BASE.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read().decode(errors="replace"), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(errors="replace"), e.code
    except (urllib.error.URLError, OSError) as e:
        raw, status = f"connection error: {e}", 0
    return status, raw, time.time() - t0


def chat(prompt, tools, stream=False, messages=None, **extra):
    body = {
        "model": "m",
        "messages": messages or [{"role": "user", "content": prompt}],
        "tools": tools,
        "max_tokens": 512,
        "temperature": 0,
        "enable_thinking": False,
    }
    if stream:
        body["stream"] = True
    body.update(extra)
    return body


# ---------------------------------------------------------------- parsing


class Resp:
    def __init__(self, content, calls, finish, problems, dt):
        self.content = content
        self.calls = calls  # message-shaped tool calls, in index order
        self.finish = finish
        self.problems = problems  # stream delta shape problems
        self.dt = dt


def parse_message(raw, dt):
    choice = json.loads(raw)["choices"][0]
    msg = choice["message"]
    return Resp(msg.get("content") or "", msg.get("tool_calls") or [], choice.get("finish_reason"), [], dt)


def parse_stream(raw, dt):
    text, finish, problems = [], None, []
    calls = {}  # index -> message-shaped call
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        for ch in chunk.get("choices", []):
            delta = ch.get("delta") or {}
            if delta.get("content"):
                text.append(delta["content"])
            for d in delta.get("tool_calls") or []:
                if "index" not in d:
                    problems.append(f"tool_calls delta without index: {json.dumps(d)[:80]}")
                    continue
                i = d["index"]
                if i not in calls:
                    kind = d.get("type")
                    inner = d.get(kind) if kind in ("function", "custom") else None
                    if not d.get("id") or not isinstance(inner, dict) or not inner.get("name"):
                        problems.append(f"first delta of index {i} lacks id/type/name: {json.dumps(d)[:120]}")
                        kind = kind or "function"
                    calls[i] = {"id": d.get("id"), "type": kind}
                    if kind == "custom":
                        calls[i]["custom"] = {"name": (inner or {}).get("name"), "input": ""}
                    else:
                        calls[i]["function"] = {"name": (inner or {}).get("name"), "arguments": ""}
                    if "function" in d and kind == "custom":
                        problems.append(f"custom delta of index {i} carries a function key")
                call = calls[i]
                if call["type"] == "custom":
                    piece = (d.get("custom") or {}).get("input")
                    if isinstance(piece, str):
                        call["custom"]["input"] += piece
                else:
                    piece = (d.get("function") or {}).get("arguments")
                    if isinstance(piece, str):
                        call["function"]["arguments"] += piece
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return Resp("".join(text), [calls[i] for i in sorted(calls)], finish, problems, dt)


def send(case, body):
    """Send, check HTTP 200 and parse. Returns a Resp, or None after a FAIL verdict."""
    status, raw, dt = post(body)
    if status != 200:
        hint = ""
        if MOCK and status == 400 and "no chat template" in raw:
            hint = " (the mock model needs a chat template for tools)"
        verdict(case, "FAIL", f"HTTP {status}{hint}: {raw[:200]}")
        return None
    try:
        return parse_stream(raw, dt) if body.get("stream") else parse_message(raw, dt)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        verdict(case, "FAIL", f"unparseable response ({type(e).__name__}: {e}): {raw[:200]}")
        return None


def call_name(c):
    inner = c.get(c.get("type")) if isinstance(c, dict) else None
    return inner.get("name") if isinstance(inner, dict) else None


def function_call_problems(c):
    """Shape problems of one function call (always a FAIL, enforced or not)."""
    if not isinstance(c, dict):
        return [f"call is not an object: {c!r:.80}"]
    out = []
    if c.get("type") != "function":
        out.append(f"type={c.get('type')!r} (want function)")
        return out
    if not c.get("id"):
        out.append("call without id")
    fn = c.get("function")
    if not isinstance(fn, dict):
        return out + ["no function object"]
    try:
        args = json.loads(fn.get("arguments") or "")
        if not isinstance(args, dict):
            out.append(f"{fn.get('name')} arguments are not a JSON object: {fn.get('arguments')!r:.80}")
    except (ValueError, TypeError):
        out.append(f"{fn.get('name')} arguments are not JSON: {fn.get('arguments')!r:.80}")
    return out


def custom_call_problems(c, name):
    if not isinstance(c, dict):
        return [f"call is not an object: {c!r:.80}"]
    out = []
    if c.get("type") != "custom":
        out.append(f"type={c.get('type')!r} (want custom)")
    if "function" in c:
        out.append("custom call carries a function key")
    if not c.get("id"):
        out.append("call without id")
    cu = c.get("custom")
    if not isinstance(cu, dict):
        return out + ["no custom object"]
    if cu.get("name") != name:
        out.append(f"custom.name={cu.get('name')!r} (want {name})")
    if not isinstance(cu.get("input"), str):
        out.append(f"custom.input is not a string: {cu.get('input')!r:.80}")
    return out


def summary(r):
    names = [call_name(c) for c in r.calls]
    text = r.content[:80].replace("\n", " ")
    return f"{r.dt:.2f}s finish={r.finish} calls={names} content={text!r}"


def schema_valid(text):
    """Hand check of ANSWER_SCHEMA: an object with exactly one string key `answer`."""
    try:
        v = json.loads(text)
    except ValueError:
        return False
    return isinstance(v, dict) and set(v) == {"answer"} and isinstance(v["answer"], str)


# ---------------------------------------------------------------- verdicts


def verdict(case, tag, detail):
    if DRY_RUN:
        return
    print(f"VERDICT {case} {tag} {detail}", flush=True)
    if tag == "FAIL":
        FAILED.append(case)


def judge(case, r, shape, enforce, info=None):
    """shape: problems that always fail. enforce: problems a constraint would have prevented."""
    shape = list(r.problems) + list(shape)
    if shape:
        verdict(case, "FAIL", f"{'; '.join(shape)} | {summary(r)}")
    elif enforce:
        if FALLBACK:
            verdict(case, "INFO", f"{NOT_ENFORCED}: {'; '.join(enforce)} | {summary(r)}")
        elif info:
            verdict(case, "INFO", f"{info}: {'; '.join(enforce)} | {summary(r)}")
        else:
            verdict(case, "FAIL", f"{'; '.join(enforce)} | {summary(r)}")
    else:
        verdict(case, "PASS", summary(r))


def function_calls_check(r, allowed, finish, count=None, at_least=1):
    """Shared checks for forced function calls; returns (shape, enforce)."""
    shape, enforce = [], []
    for c in r.calls:
        shape += function_call_problems(c)
    n = len(r.calls)
    if count is not None and n != count:
        enforce.append(f"{n} calls (want exactly {count})")
    elif n < at_least:
        enforce.append(f"{n} calls (want at least {at_least})")
    bad = [call_name(c) for c in r.calls if call_name(c) not in allowed]
    if bad:
        enforce.append(f"calls to {bad} (allowed {sorted(allowed)})")
    if finish and r.finish != finish:
        enforce.append(f"finish_reason={r.finish} (want {finish})")
    return shape, enforce


# ---------------------------------------------------------------- cases


def case_required():
    for stream in (False, True):
        case = "required_stream" if stream else "required"
        r = send(case, chat("Tell me about Paris.", [GET_WEATHER, GET_TIME], stream, tool_choice="required"))
        if r:
            judge(case, r, *function_calls_check(r, {"get_weather", "get_time"}, "tool_calls"))


def case_named_function():
    choice = {"type": "function", "function": {"name": "get_weather"}}
    for stream in (False, True):
        case = "named_function_stream" if stream else "named_function"
        r = send(case, chat("Tell me a joke.", [GET_WEATHER, GET_TIME], stream, tool_choice=choice))
        if r:
            judge(case, r, *function_calls_check(r, {"get_weather"}, "stop", count=1))


def case_named_custom():
    choice = {"type": "custom", "custom": {"name": "apply_patch"}}
    for stream in (False, True):
        case = "named_custom_stream" if stream else "named_custom"
        body = chat("Fix the typo in README.md.", [GET_WEATHER, APPLY_PATCH], stream, tool_choice=choice)
        r = send(case, body)
        if not r:
            continue
        shape, enforce = [], []
        if len(r.calls) != 1:
            enforce.append(f"{len(r.calls)} calls (want exactly 1)")
        for c in r.calls:
            if call_name(c) == "apply_patch" or (isinstance(c, dict) and c.get("type") == "custom"):
                shape += custom_call_problems(c, "apply_patch")
            else:
                enforce.append(f"call to {call_name(c)!r} (want apply_patch)")
        if r.finish != "stop":
            enforce.append(f"finish_reason={r.finish} (want stop)")
        judge(case, r, shape, enforce)


def case_custom_replay():
    messages = [
        {"role": "user", "content": "Fix the typo in README.md."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "custom", "custom": {"name": "apply_patch", "input": "x"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Patch applied."},
    ]
    r = send("custom_replay", chat(None, [GET_WEATHER, APPLY_PATCH], messages=messages))
    if r:
        verdict("custom_replay", "PASS", f"HTTP 200 {summary(r)}")


def allowed_choice(mode, names):
    return {
        "type": "allowed_tools",
        "allowed_tools": {"mode": mode, "tools": [{"type": "function", "function": {"name": n}} for n in names]},
    }


def case_allowed_required():
    tools = [GET_WEATHER, GET_TIME, GET_STOCK]
    choice = allowed_choice("required", ["get_weather", "get_time"])
    r = send("allowed_required", chat("What is the stock price of ACME?", tools, tool_choice=choice))
    if not r:
        return
    shape, enforce = function_calls_check(r, {"get_weather", "get_time"}, "tool_calls")
    # The API drops calls outside the allowed list whatever the engine did, so this one always fails.
    leaked = [c for c in r.calls if call_name(c) == "get_stock_price"]
    if leaked:
        shape.append("a call to get_stock_price, which allowed_tools left out")
        enforce = [e for e in enforce if not e.startswith("calls to")]
    judge("allowed_required", r, shape, enforce)


def case_allowed_auto():
    tools = [GET_WEATHER, GET_TIME, GET_STOCK]
    choice = allowed_choice("auto", ["get_weather", "get_time"])
    r = send("allowed_auto", chat("What is the stock price of ACME?", tools, tool_choice=choice))
    if not r:
        return
    shape = []
    for c in r.calls:
        shape += function_call_problems(c)
    if any(call_name(c) == "get_stock_price" for c in r.calls):
        shape.append("a call to get_stock_price, which allowed_tools left out")
    judge("allowed_auto", r, shape, [])


def case_allowed_bad_name():
    choice = allowed_choice("required", ["nope"])
    status, raw, _ = post(chat("Tell me about Paris.", [GET_WEATHER, GET_TIME], tool_choice=choice))
    ok = status == 400 and "tool_choice" in raw
    verdict("allowed_bad_name", "PASS" if ok else "FAIL", f"HTTP {status}: {raw[:200]}")


PARALLEL_PROMPT = "What is the weather in Paris and what time is it in Tokyo?"


def case_parallel_false():
    tools = [GET_WEATHER, GET_TIME]
    names = {"get_weather", "get_time"}
    for stream in (False, True):
        case = "parallel_false_stream" if stream else "parallel_false"
        body = chat(PARALLEL_PROMPT, tools, stream, tool_choice="required", parallel_tool_calls=False)
        r = send(case, body)
        if r:
            shape, enforce = function_calls_check(r, names, "tool_calls", count=1)
            if len(r.calls) > 1:
                # Without a grammar the API still keeps only the first parsed call, so a second
                # call fails on every node.
                shape.append(f"{len(r.calls)} calls with parallel_tool_calls false")
                enforce = [e for e in enforce if "want exactly" not in e]
            judge(case, r, shape, enforce)
    # Control: the same prompt with parallel calls allowed must be able to make two, or the
    # false case above proves nothing. A model that makes only one call even when two are
    # allowed is a limit of the model, not of the server: the control is then inconclusive.
    case = "parallel_true_control"
    r = send(case, chat(PARALLEL_PROMPT, tools, tool_choice="required", parallel_tool_calls=True))
    if r:
        shape, enforce = function_calls_check(r, names, "tool_calls", at_least=2)
        if len(r.calls) < 2 and not shape:
            verdict(case, "INFO", f"inconclusive (model made <2 calls) | {summary(r)}")
        else:
            judge(case, r, shape, enforce)


def case_json_union():
    rf = {"type": "json_schema", "json_schema": {"name": "reply", "schema": ANSWER_SCHEMA, "strict": True}}

    def one(case, prompt):
        r = send(case, chat(prompt, [GET_WEATHER], tool_choice="auto", response_format=rf))
        if not r:
            return None, None
        shape = []
        for c in r.calls:
            shape += function_call_problems(c)
        if r.calls:
            bad = [call_name(c) for c in r.calls if call_name(c) != "get_weather"]
            if bad:
                shape.append(f"calls to undeclared {bad}")
            return r, ("call", shape)
        return r, ("content" if schema_valid(r.content) else "neither", shape)

    r, res = one("json_union_tool_prompt", "What's the weather in Paris? Use the tool.")
    if r:
        kind, shape = res
        if kind == "neither":
            judge("json_union_tool_prompt", r, shape, ["neither a call nor content matching the schema"])
        else:
            judge("json_union_tool_prompt", r, shape, [])
            if not shape:
                verdict("json_union_tool_prompt_answer", "INFO", f"answered with {kind}")

    r, res = one("json_union_hello", "Say hello.")
    if r:
        kind, shape = res
        if kind == "content":
            judge("json_union_hello", r, shape, [])
        elif kind == "call":
            if shape:
                judge("json_union_hello", r, shape, [])
            else:
                why = "the mock answers the union's first branch" if MOCK else "answered with a call"
                verdict("json_union_hello", "INFO", f"{why} | {summary(r)}")
        else:
            judge("json_union_hello", r, shape, ["content does not match the schema"])


def case_plain_unchanged():
    r = send("plain_unchanged", chat("Say hello.", [GET_WEATHER]))
    if not r:
        return
    shape = []
    if any(isinstance(c, dict) and (c.get("type") == "custom" or "custom" in c) for c in r.calls):
        shape.append("custom call objects in a function-only request")
    for c in r.calls:
        shape += function_call_problems(c)
    if not r.calls and not r.content.strip():
        shape.append("neither content nor a call")
    judge("plain_unchanged", r, shape, [])


def case_gemma():
    for stream in (False, True):
        case = "gemma_stream" if stream else "gemma"
        r = send(case, chat("What's the weather in Paris?", [GET_WEATHER], stream, tool_choice="auto"))
        if not r:
            continue
        shape, enforce = [], []
        leaked = [m for m in GEMMA_MARKERS if m in r.content]
        if leaked:
            shape.append(f"content keeps Gemma markers {leaked}")
        for c in r.calls:
            shape += function_call_problems(c)
        if not r.calls or call_name(r.calls[0]) != "get_weather":
            enforce.append("tool_calls[0] is not get_weather")
        elif not shape:
            args = json.loads(r.calls[0]["function"]["arguments"])
            if "city" not in args:
                enforce.append(f"arguments without city: {args}")
        judge(case, r, shape, enforce, info="the mock echoes the prompt under auto" if MOCK else None)


def case_concurrency(n):
    def one(i):
        body = chat(f"Tell me about city number {i}, Paris.", [GET_WEATHER, GET_TIME], tool_choice="required")
        status, raw, _ = post(body)
        if status != 200:
            return "fail", f"#{i} HTTP {status}: {raw[:80]}"
        try:
            r = parse_message(raw, 0)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            return "fail", f"#{i} unparseable ({e})"
        shape, enforce = function_calls_check(r, {"get_weather", "get_time"}, "tool_calls")
        if shape:
            return "fail", f"#{i} {'; '.join(shape)}"
        if enforce:
            return ("info" if FALLBACK else "fail"), f"#{i} {'; '.join(enforce)}"
        return "pass", f"#{i} {[call_name(c) for c in r.calls]}"

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(one, range(n)))
    kinds = {k for k, _ in results}
    detail = f"{time.time() - t0:.2f}s " + "; ".join(d for _, d in results)
    case = f"concurrency_{n}"
    if "fail" in kinds:
        verdict(case, "FAIL", detail)
    elif "info" in kinds:
        verdict(case, "INFO", f"{NOT_ENFORCED}: {detail}")
    else:
        verdict(case, "PASS", detail)


CASES = {
    "required": case_required,
    "named_function": case_named_function,
    "named_custom": case_named_custom,
    "custom_replay": case_custom_replay,
    "allowed_required": case_allowed_required,
    "allowed_auto": case_allowed_auto,
    "allowed_bad_name": case_allowed_bad_name,
    "parallel_false": case_parallel_false,
    "json_union": case_json_union,
    "plain_unchanged": case_plain_unchanged,
}


def main():
    ap = argparse.ArgumentParser(
        description="Live check of tool_choice, allowed_tools, custom tools and parallel_tool_calls.",
        epilog="See the module docstring for the cases and what --mock / --expect-fallback relax.",
    )
    ap.add_argument("base", help="server base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--cases", default=",".join(CASES), help="comma-separated subset of: " + ",".join(CASES))
    ap.add_argument("--concurrency", type=int, default=0, help="also send N `required` requests at once (N > 1)")
    ap.add_argument("--mock", action="store_true", help="the server runs the mock engine")
    ap.add_argument("--gemma", action="store_true", help="the server runs a Gemma 4 model: add the gemma case")
    ap.add_argument("--expect-fallback", action="store_true",
                    help="the node drops the forced-call grammar (ov-genai NPU / draft model)")
    ap.add_argument("--dry-run", action="store_true", help="print the request bodies, send nothing")
    args = ap.parse_args()

    global BASE, MOCK, FALLBACK, DRY_RUN
    BASE, MOCK, FALLBACK, DRY_RUN = args.base, args.mock, args.expect_fallback, args.dry_run
    names = [n for n in args.cases.split(",") if n]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; known: {','.join(CASES)}")

    for name in names:
        CASES[name]()
    if args.gemma:
        case_gemma()
    if args.concurrency > 1:
        case_concurrency(args.concurrency)

    if DRY_RUN:
        print(f"TOOL CHOICE CHECK: DRY RUN ({DRY_COUNT} requests)", flush=True)
        sys.exit(0)
    if FAILED:
        print(f"TOOL CHOICE CHECK: FAILED ({len(FAILED)}: {', '.join(FAILED)})", flush=True)
        sys.exit(1)
    print("TOOL CHOICE CHECK: ALL PASS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
