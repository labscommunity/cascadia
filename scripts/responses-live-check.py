#!/usr/bin/env python3
"""Live /v1/responses check against a running Cascadia server (any engine).

Sends what Codex and the agent SDKs send and checks the answers' shape:

  text            plain input; a thinking model's reasoning is a `reasoning`
                  item, never `<think>` text in `output_text`
  tool_call       one function tool; the model's call is a `function_call`
                  with JSON arguments
  tool_stream     the same, streamed: output_item.added, function_call_arguments
                  delta/done, response.completed carrying the call
  tool_result     the call and its output sent back; a text answer follows
  codex_turn1     Codex CLI 0.166.0's first request (functions, a namespace,
                  web_search, include, reasoning.summary), streamed
  codex_turn2     Codex's replay (reasoning, function_call, its output), streamed
  codex_late_dev  turn 2 plus a later `developer` message and a new user
                  message, as Codex sends after a permission change
  late_dev        instructions, a user and an assistant turn, then a
                  `developer` message and a user message, with no tools (the
                  turns a template that only takes a leading system turn
                  refuses unless the server folds them)

Usage:
  python3 scripts/responses-live-check.py http://127.0.0.1:18100 [--max-output-tokens 1024]
          [--only text,tool_call] [--timeout 1800] [--mock] [--expect-reasoning]

`--mock` adds the mock engine's `__mock_tool_call__` marker to the tool prompt,
so `cascadia run tests/fixtures/mock-tools-model --engine mock` scripts a call.

Every response body and every streamed event is validated against the pinned
OpenAI spec subset (`crates/cascadia-api/tests/fixtures/openai-spec-subset.json`:
`Response`, `ResponseStreamEvent`), no `<think>`/`</think>` may appear in any
output text or delta, and `usage.output_tokens_details.reasoning_tokens` must be
present and at most `output_tokens`. `--expect-reasoning` (a thinking model)
also requires the `text` check to return a `reasoning` item.

Every check prints `<NAME>: PASS` or `<NAME>: FAIL <why>`; the last line is
`RESPONSES_LIVE: ALL PASS` only when all ran and passed. Stdlib only.
"""

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "crates" / "cascadia-api" / "tests" / "fixtures"
SCHEMAS = json.loads((FIXTURES / "openai-spec-subset.json").read_text())["components"]["schemas"]
MARKERS = ("<think>", "</think>")
TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def spec_errors(schema, v, at="$", depth=0):
    """The ways `v` breaks `schema` (the JSON-schema keywords the pinned
    subset uses); empty when it validates."""
    if depth > 64:
        return [f"{at}: schema too deep"]
    if "$ref" in schema:
        return spec_errors(SCHEMAS[schema["$ref"].rsplit("/", 1)[-1]], v, at, depth + 1)
    errs = []
    for sub in schema.get("allOf", []):
        errs += spec_errors(sub, v, at, depth + 1)
    if "not" in schema and not spec_errors(schema["not"], v, at, depth + 1):
        errs.append(f"{at}: matches a `not` schema")
    for key, need_one in (("anyOf", False), ("oneOf", True)):
        if key in schema:
            branches = [spec_errors(b, v, at, depth + 1) for b in schema[key]]
            ok = sum(1 for b in branches if not b)
            if ok == 0:
                # Name the branch whose `type` const matches, else the shortest.
                typed = [b for b in branches if not any(e.startswith(f"{at}.type:") for e in b)]
                best = min(typed or branches, key=len)
                errs.append(f"{at}: matches no {key} branch (closest: {best[:2]})")
            elif need_one and ok > 1:
                errs.append(f"{at}: matches {ok} oneOf branches")
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(TYPES[x](v) for x in ts if x in TYPES):
            return errs + [f"{at}: {type(v).__name__} is not {t}"]
    if "enum" in schema and v not in schema["enum"]:
        errs.append(f"{at}: {v!r} not in {schema['enum'][:6]}")
    if isinstance(v, str):
        if len(v) < schema.get("minLength", 0) or len(v) > schema.get("maxLength", 1 << 62):
            errs.append(f"{at}: string length {len(v)}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v < schema.get("minimum", float("-inf")) or v > schema.get("maximum", float("inf")):
            errs.append(f"{at}: {v} out of range")
    if isinstance(v, dict):
        for k in schema.get("required", []):
            if k not in v:
                errs.append(f"{at}: missing required {k!r}")
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in v:
                errs += spec_errors(sub, v[k], f"{at}.{k}", depth + 1)
        extra = schema.get("additionalProperties")
        if extra is False:
            errs += [f"{at}: unexpected {k!r}" for k in v if k not in props]
        elif isinstance(extra, dict):
            for k in v:
                if k not in props:
                    errs += spec_errors(extra, v[k], f"{at}.{k}", depth + 1)
    if isinstance(v, list):
        if len(v) < schema.get("minItems", 0):
            errs.append(f"{at}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, x in enumerate(v):
                errs += spec_errors(schema["items"], x, f"{at}[{i}]", depth + 1)
    return errs


def conform(name, v, what):
    errs = spec_errors({"$ref": "#/components/schemas/" + name}, v)
    if errs:
        raise Fail(f"{what} breaks the spec's {name}: {errs[:3]}")


def check_usage(resp):
    u = resp.get("usage") or {}
    rt = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
    if not isinstance(rt, int) or rt > u.get("output_tokens", 0):
        raise Fail(f"usage reasoning_tokens {rt!r} vs output_tokens {u.get('output_tokens')!r}")

WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}
TOOL_PROMPT = (
    "What is the weather in Paris right now? You cannot know it yourself: "
    "call the get_weather tool with the city."
)


class Fail(Exception):
    pass


def post(base, body, timeout):
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise Fail(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")
    return resp


def call(base, body, timeout):
    with post(base, body, timeout) as r:
        resp = json.loads(r.read())
    conform("Response", resp, "the response body")
    if resp.get("usage") is not None:
        check_usage(resp)
    no_think_markers(resp)
    return resp


def stream(base, body, timeout):
    """The stream's events, in order (`data:` payloads)."""
    events = []
    with post(base, dict(body, stream=True), timeout) as r:
        for raw in r:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    events.append(json.loads(data))
    for e in events:
        conform("ResponseStreamEvent", e, f"event {e.get('type')} #{e.get('sequence_number')}")
        if e.get("type") in ("response.output_text.delta", "response.output_text.done"):
            t = e.get("delta", e.get("text", ""))
            if any(m in t for m in MARKERS):
                raise Fail(f"think marker in {e['type']}: {t[:120]!r}")
    if events and isinstance(events[-1].get("response"), dict):
        final = events[-1]["response"]
        no_think_markers(final)
        if final.get("usage") is not None:
            check_usage(final)
    return events


def texts(resp):
    return [
        part.get("text", "")
        for item in resp.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]


def check_finished(resp, allow_incomplete=True):
    status = resp.get("status")
    ok = {"completed", "incomplete"} if allow_incomplete else {"completed"}
    if status not in ok:
        raise Fail(f"status {status!r}, error {resp.get('error')}")


def no_think_markers(resp):
    for t in texts(resp):
        if any(m in t for m in MARKERS):
            raise Fail(f"think marker in output_text: {t[:200]!r}")
    for item in resp.get("output", []):
        if item.get("type") == "reasoning":
            for part in item.get("content") or []:
                if any(m in part.get("text", "") for m in MARKERS):
                    raise Fail(f"think marker in reasoning text: {part.get('text', '')[:120]!r}")


def summary(resp):
    kinds = [i.get("type") for i in resp.get("output", [])]
    return f"status={resp.get('status')} items={kinds} usage={resp.get('usage')}"


def the_call(resp):
    calls = [i for i in resp.get("output", []) if i.get("type") == "function_call"]
    if not calls:
        raise Fail(f"no function_call: {summary(resp)} text={texts(resp)[:1]}")
    c = calls[0]
    if c.get("name") != "get_weather":
        raise Fail(f"called {c.get('name')!r}")
    try:
        args = json.loads(c.get("arguments", ""))
    except ValueError:
        raise Fail(f"arguments are not JSON: {c.get('arguments')!r}")
    if not isinstance(args, dict) or "city" not in args:
        raise Fail(f"arguments {args!r}")
    return c


def run(args):
    global TOOL_PROMPT
    if args.mock:
        TOOL_PROMPT += " __mock_tool_call__"
    base, mot, to = args.url, args.max_output_tokens, args.timeout
    try:
        model = call(base, {"model": "x", "input": "hi", "max_output_tokens": 1}, to)["model"]
    except Exception as e:  # noqa: BLE001 - the verdict line must still come
        print(f"SERVER: FAIL {type(e).__name__}: {e}")
        print("RESPONSES_LIVE: FAIL (server)")
        return 1
    state = {}

    def text():
        r = call(base, {"model": model, "input": "What is 17 times 3? Answer with the number.",
                        "max_output_tokens": mot}, to)
        check_finished(r)
        if not "".join(texts(r)).strip() and r.get("status") == "completed":
            raise Fail(f"empty answer: {summary(r)}")
        if args.expect_reasoning and not any(i.get("type") == "reasoning" for i in r["output"]):
            raise Fail(f"--expect-reasoning: no reasoning item: {summary(r)}")
        return summary(r)

    def tool_call():
        r = call(base, {"model": model, "input": TOOL_PROMPT, "tools": [WEATHER],
                        "max_output_tokens": mot}, to)
        check_finished(r)
        no_think_markers(r)
        state["call"] = the_call(r)
        state["output"] = r["output"]
        return summary(r)

    def tool_stream():
        ev = stream(base, {"model": model, "input": TOOL_PROMPT, "tools": [WEATHER],
                           "max_output_tokens": mot}, to)
        types = [e.get("type") for e in ev]
        if types[:2] != ["response.created", "response.in_progress"]:
            raise Fail(f"first events {types[:2]}")
        last = ev[-1]
        if last.get("type") not in ("response.completed", "response.incomplete"):
            raise Fail(f"last event {last.get('type')}: {json.dumps(last)[:300]}")
        if "response.function_call_arguments.done" not in types:
            raise Fail(f"no function_call_arguments.done in {sorted(set(types))}")
        added = [e for e in ev if e.get("type") == "response.output_item.added"
                 and e["item"].get("type") == "function_call"]
        if not added:
            raise Fail("no output_item.added for the call")
        resp = last["response"]
        the_call(resp)
        seq = [e.get("sequence_number") for e in ev]
        if seq != sorted(seq):
            raise Fail("sequence_number not increasing")
        return f"{len(ev)} events; {summary(resp)}"

    def tool_result():
        if "call" not in state:
            raise Fail("needs tool_call to pass first")
        c = state["call"]
        history = [{"role": "user", "content": TOOL_PROMPT}] + [
            i for i in state["output"] if i.get("type") in ("reasoning", "function_call")
        ] + [{"type": "function_call_output", "call_id": c["call_id"],
              "output": json.dumps({"city": "Paris", "temperature_c": 18, "sky": "clear"})}]
        r = call(base, {"model": model, "input": history, "tools": [WEATHER],
                        "store": False, "max_output_tokens": mot}, to)
        check_finished(r)
        no_think_markers(r)
        if not "".join(texts(r)).strip() and r.get("status") == "completed":
            raise Fail(f"no answer after the tool output: {summary(r)}")
        return summary(r) + f" answer={''.join(texts(r))[:120]!r}"

    def codex(name, extra=None):
        body = json.loads((FIXTURES / name).read_text())
        body["model"] = model
        body["max_output_tokens"] = mot
        if extra:
            body["input"] = body["input"] + extra
        ev = stream(base, body, to)
        last = ev[-1] if ev else {}
        if last.get("type") not in ("response.completed", "response.incomplete"):
            raise Fail(f"last event {last.get('type')}: {json.dumps(last)[:400]}")
        no_think_markers(last["response"])
        return f"{len(ev)} events; {summary(last['response'])}"

    late_dev = [
        {"type": "message", "role": "developer",
         "content": [{"type": "input_text", "text": "<permissions instructions>Commands now run "
                      "without approval.</permissions instructions>"}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "Now say done."}]},
    ]
    def late_dev_plain():
        r = call(base, {"model": model, "instructions": "You are terse.", "input": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
            {"role": "developer", "content": "From now on answer in one word."},
            {"role": "user", "content": "Say done."},
        ], "max_output_tokens": mot, "store": False}, to)
        check_finished(r)
        if not "".join(texts(r)).strip() and r.get("status") == "completed":
            raise Fail(f"empty answer: {summary(r)}")
        return summary(r) + f" answer={''.join(texts(r))[:60]!r}"

    checks = [
        ("text", text),
        ("tool_call", tool_call),
        ("tool_stream", tool_stream),
        ("tool_result", tool_result),
        ("codex_turn1", lambda: codex("codex-0.166.0-turn1.json")),
        ("codex_turn2", lambda: codex("codex-0.166.0-turn2.json")),
        ("codex_late_dev", lambda: codex("codex-0.166.0-turn2.json", late_dev)),
        ("late_dev", late_dev_plain),
    ]
    only = set(args.only.split(",")) if args.only else None
    unknown = (only or set()) - {name for name, _ in checks}
    if unknown:
        print(f"RESPONSES_LIVE: FAIL (unknown checks: {', '.join(sorted(unknown))})")
        return 2
    failed, ran = [], 0
    print(f"model: {model}")
    for name, fn in checks:
        if only and name not in only:
            continue
        ran += 1
        try:
            detail = fn()
            print(f"{name.upper()}: PASS  {detail}", flush=True)
        except Exception as e:  # noqa: BLE001 - every failure is a verdict line
            failed.append(name)
            why = str(e) if isinstance(e, Fail) else f"{type(e).__name__}: {e}"
            print(f"{name.upper()}: FAIL {why}", flush=True)
    if failed:
        print(f"RESPONSES_LIVE: FAIL ({', '.join(failed)})")
        return 1
    print("RESPONSES_LIVE: ALL PASS" if not only else f"RESPONSES_LIVE: PASS ({ran} selected)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("url")
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--only", default="")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--expect-reasoning", action="store_true")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
