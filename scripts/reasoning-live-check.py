#!/usr/bin/env python3
"""Live check of a thinking model's reasoning on /v1/chat/completions.

Runs against a running Cascadia server (any engine) and checks what the README's
"Reasoning (thinking) models" section promises:

  nonstream_split            thinking on (the default): the reasoning comes back in
                             `message.reasoning_content`, `content` holds no
                             `<think>`/`</think>`, and
                             `usage.completion_tokens_details.reasoning_tokens` is
                             counted (> 0, at most `completion_tokens`)
  stream_split               the same, streamed with `stream_options.include_usage`:
                             reasoning arrives as `delta.reasoning_content` frames,
                             never in the same delta as answer text; the last choice
                             frame has `finish_reason`; the usage chunk has
                             `choices: []` and `reasoning_tokens`; the body ends with
                             `data: [DONE]`
  stream_matches_nonstream   (--mock only, where output is deterministic) the
                             streamed reasoning and answer equal the non-streamed ones
  switch_thinking_disabled   `thinking: {type: disabled}` turns thinking off: no
                             `reasoning_content`, no `completion_tokens_details`
  switch_kwargs_off          the same with `chat_template_kwargs.enable_thinking: false`
  switch_effort_none         the same with `reasoning: {effort: "none"}`
  replayed_reasoning_accepted  a tool loop whose assistant turn carries its earlier
                             reasoning (as `reasoning_content`, then as `reasoning`)
                             is accepted and answered
  odd_shapes_accepted        unknown switch shapes (`reasoning: "high"`,
                             `thinking: true`) are ignored, not rejected
  format_none_think_in_content  (--format-none, replaces the two split checks) a
                             server started with `--reasoning-format none` keeps
                             `<think>...</think>` in `content`, sends no
                             `reasoning_content` (also not in any stream delta), and
                             still reports `reasoning_tokens`

For a real model, thinking-off replies must also have no think tags in `content`.
With `--mock` (server started with `--engine mock`, which echoes the last user
message word by word) every prompt is `<think> plan more </think> the answer`, so
thinking-on replies carry reasoning and thinking-off replies keep the text as is,
`<think>` included.

Usage:
  python3 scripts/reasoning-live-check.py BASE_URL [--model M] [--mock]
          [--format-none] [--timeout S]

Without `--model`, the first id from `/v1/models` is used. Every check prints
`<name>: PASS` or `<name>: FAIL <why>`; the last line is
`REASONING_LIVE: ALL PASS` (exit 0) or `REASONING_LIVE: FAIL (n)` (exit 1).
Python 3 stdlib only.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

TAGS = ("<think>", "</think>")
MOCK_PROMPT = "<think> plan more </think> the answer"
REAL_PROMPT = "What is 17*23? Think step by step, then answer with the number only."
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class Fail(Exception):
    pass


def http(base, path, body, timeout):
    """(status, raw body bytes) of a GET (body None) or a JSON POST."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        raise Fail(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")
    except urllib.error.URLError as e:
        raise Fail(f"cannot reach {base}: {e.reason}")


class Client:
    def __init__(self, args):
        self.base, self.timeout, self.mock = args.url, args.timeout, args.mock
        self.model = args.model
        self.max_tokens = 256 if args.mock else 2048

    def body(self, extra=None, messages=None):
        prompt = MOCK_PROMPT if self.mock else REAL_PROMPT
        b = {
            "model": self.model,
            "messages": messages or [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        b.update(extra or {})
        return b

    def chat(self, extra=None, messages=None):
        status, raw = http(self.base, "/v1/chat/completions", self.body(extra, messages), self.timeout)
        if status != 200:
            raise Fail(f"HTTP {status}")
        return json.loads(raw)

    def stream(self, extra=None):
        """(chunks, raw text) of a streamed request, parsing the SSE body here."""
        b = self.body(dict(extra or {}, stream=True, stream_options={"include_usage": True}))
        _, raw = http(self.base, "/v1/chat/completions", b, self.timeout)
        text = raw.decode()
        chunks = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    chunks.append(json.loads(data))
        return chunks, text


def message(resp):
    try:
        return resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise Fail(f"no choices[0].message in {json.dumps(resp)[:300]}")


def no_tags(text, where):
    if any(t in (text or "") for t in TAGS):
        raise Fail(f"think tag in {where}: {text[:160]!r}")


def reasoning_tokens(usage, where):
    u = usage or {}
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if not isinstance(rt, int) or isinstance(rt, bool):
        raise Fail(f"{where}: reasoning_tokens {rt!r} is not an int (usage {u})")
    return rt, u.get("completion_tokens")


def joined(chunks):
    """Joined reasoning and content deltas; fails on a delta carrying both."""
    reasoning, content = [], []
    for c in chunks:
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            r, t = d.get("reasoning_content"), d.get("content")
            if r and t:
                raise Fail(f"one delta has both reasoning_content and content: {d}")
            reasoning.append(r or "")
            content.append(t or "")
    return "".join(reasoning), "".join(content)


def check_nonstream_split(cl, state):
    resp = cl.chat()
    m = message(resp)
    r = m.get("reasoning_content")
    if not isinstance(r, str) or not r:
        raise Fail(f"reasoning_content is {r!r}")
    no_tags(m.get("content"), "content")
    rt, ct = reasoning_tokens(resp.get("usage"), "usage")
    if not (0 < rt <= (ct or 0)):
        raise Fail(f"reasoning_tokens {rt} not in 1..completion_tokens {ct}")
    state["nonstream"] = (r, m.get("content") or "")


def check_stream_split(cl, state):
    chunks, text = cl.stream()
    reasoning, content = joined(chunks)
    if not reasoning:
        raise Fail("no delta.reasoning_content in the stream")
    no_tags(content, "streamed content")
    framed = [c for c in chunks if c.get("choices")]
    if not framed or not framed[-1]["choices"][0].get("finish_reason"):
        raise Fail("the last choice frame has no finish_reason")
    usage = [c for c in chunks if c.get("usage")]
    if not usage or usage[-1].get("choices") != []:
        raise Fail(f"no usage chunk with choices: [] ({usage[-1:] or 'none'})")
    reasoning_tokens(usage[-1]["usage"], "usage chunk")
    if not text.rstrip().endswith("data: [DONE]"):
        raise Fail(f"body does not end with data: [DONE]: {text[-80:]!r}")
    state["stream"] = (reasoning, content)


def check_stream_matches_nonstream(cl, state):
    if "nonstream" not in state or "stream" not in state:
        raise Fail("needs nonstream_split and stream_split to pass first")
    if state["nonstream"] != state["stream"]:
        raise Fail(f"non-stream {state['nonstream']!r} != stream {state['stream']!r}")


def thinking_off(cl, extra):
    resp = cl.chat(extra)
    m = message(resp)
    if m.get("reasoning_content"):
        raise Fail(f"reasoning_content with thinking off: {m['reasoning_content'][:120]!r}")
    if "completion_tokens_details" in (resp.get("usage") or {}):
        raise Fail(f"completion_tokens_details with thinking off: {resp['usage']}")
    content = m.get("content") or ""
    if cl.mock:
        if "<think>" not in content:
            raise Fail(f"mock thinking-off text was changed: {content!r}")
    else:
        no_tags(content, "content")


def check_replayed_reasoning(cl, state):
    call = {"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
    user = MOCK_PROMPT if cl.mock else "What is the weather in Paris? Use the tool."
    for key in ("reasoning_content", "reasoning"):
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", key: "I need the weather first.",
             "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 18, "sky": "clear"}'},
        ]
        m = message(cl.chat({"tools": [WEATHER]}, msgs))
        if not (m.get("content") or m.get("tool_calls")):
            raise Fail(f"replay with `{key}`: empty answer and no tool call: {m}")


def check_odd_shapes(cl, state):
    message(cl.chat({"reasoning": "high", "thinking": True}))


def check_format_none(cl, state):
    resp = cl.chat()
    m = message(resp)
    if "</think>" not in (m.get("content") or ""):
        raise Fail(f"no </think> in content: {(m.get('content') or '')[:160]!r}")
    if m.get("reasoning_content"):
        raise Fail(f"reasoning_content present: {m['reasoning_content'][:120]!r}")
    reasoning_tokens(resp.get("usage"), "usage")
    chunks, _ = cl.stream()
    for c in chunks:
        for ch in c.get("choices") or []:
            if "reasoning_content" in (ch.get("delta") or {}):
                raise Fail(f"stream delta has reasoning_content: {ch['delta']}")


def model_id(args):
    _, raw = http(args.url, "/v1/models", None, args.timeout)
    data = json.loads(raw).get("data") or []
    if not data:
        raise Fail("/v1/models lists no model")
    return data[0].get("id")


def reproduce(args):
    cmd = ["python3", "scripts/reasoning-live-check.py", args.url]
    if args.model:
        cmd += ["--model", args.model]
    if args.mock:
        cmd.append("--mock")
    if args.format_none:
        cmd.append("--format-none")
    cmd += ["--timeout", str(args.timeout)]
    return " ".join(cmd)


def run(args):
    try:
        served = model_id(args)
    except Fail as e:
        print(f"models: FAIL {e}")
        print(f"reproduce: {reproduce(args)}")
        print("REASONING_LIVE: FAIL (1)")
        return 1
    print(f"model: {served}")
    cl = Client(args)
    cl.model = args.model or served

    if args.format_none:
        checks = [("format_none_think_in_content", check_format_none)]
    else:
        checks = [("nonstream_split", check_nonstream_split),
                  ("stream_split", check_stream_split)]
        if args.mock:
            checks.append(("stream_matches_nonstream", check_stream_matches_nonstream))
    checks += [
        ("switch_thinking_disabled", lambda c, s: thinking_off(c, {"thinking": {"type": "disabled"}})),
        ("switch_kwargs_off",
         lambda c, s: thinking_off(c, {"chat_template_kwargs": {"enable_thinking": False}})),
        ("switch_effort_none", lambda c, s: thinking_off(c, {"reasoning": {"effort": "none"}})),
        ("replayed_reasoning_accepted", check_replayed_reasoning),
        ("odd_shapes_accepted", check_odd_shapes),
    ]

    state, failed = {}, 0
    for name, fn in checks:
        try:
            fn(cl, state)
            print(f"{name}: PASS")
        except Fail as e:
            failed += 1
            print(f"{name}: FAIL {e}")
        except Exception as e:  # a malformed body or a dropped connection
            failed += 1
            print(f"{name}: FAIL {type(e).__name__}: {e}")
        sys.stdout.flush()

    print(f"reproduce: {reproduce(args)}")
    if failed:
        print(f"REASONING_LIVE: FAIL ({failed})")
        return 1
    print("REASONING_LIVE: ALL PASS")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("url", metavar="BASE_URL", help="e.g. http://127.0.0.1:8080")
    p.add_argument("--model", default="", help="model id to send (default: first from /v1/models)")
    p.add_argument("--mock", action="store_true",
                   help="server runs --engine mock (echoes the last user message)")
    p.add_argument("--format-none", action="store_true",
                   help="server runs --reasoning-format none")
    p.add_argument("--timeout", type=int, default=600, help="seconds per request")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
