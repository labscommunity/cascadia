#!/usr/bin/env python3
"""Live check of the OpenAI response and stream shape of a running cascadia server.

Sends real requests to ``/v1/chat/completions`` and ``/v1/completions`` and checks
the wire shape a stock OpenAI client relies on (see the "Response and stream
shape" section of docs/CLI.md): SSE framing and ``[DONE]``, ``delta.role`` on
the first chunk only, no ``n_tokens`` / ``completion_tokens`` on chunks, a
constant ``system_fingerprint``, ``usage`` with ``include_usage`` and
``continuous_usage_stats``, ``logprobs`` keys, and, with ``--tools``, tool calls
streamed as a header delta followed by argument fragments.

Prints one ``PASS <name>`` or ``FAIL <name>: <why>`` line per check, then
``OPENAI STREAM CHECK: ALL PASS`` (exit 0) or ``OPENAI STREAM CHECK: FAILED (n)``
(exit 1). Python 3.8+, standard library only.

Usage, Linux / macOS:

    python3 scripts/openai_stream_check.py --base http://127.0.0.1:8676
    python3 scripts/openai_stream_check.py --base http://127.0.0.1:8676 --tools
    python3 scripts/openai_stream_check.py --base http://127.0.0.1:8800 --tools --tool-model
    python3 scripts/openai_stream_check.py --self-test

Windows PowerShell (same flags):

    python scripts\\openai_stream_check.py --base http://127.0.0.1:8676 --tools

Modes:

* default: the plain chat, completions and non-stream checks. Works on any
  engine; ``--model`` defaults to the first id of ``/v1/models``.
* ``--tools``: adds the tool-call stream checks, aimed at the mock engine
  (``cascadia run <dir> --engine mock``), which echoes its prompt word by word.
  The mock model directory needs a chat template that accepts ``tools`` and
  renders only the user text, so the echo is exactly the user message (a
  ``<tool_call>`` the check sends). Create it once:

      mkdir -p mock-tools-model/tokenizer
      cat > mock-tools-model/tokenizer/tokenizer_config.json <<'JSON'
      {"chat_template": "{%- if tools %}{%- endif %}{%- for m in messages %}{%- if m.role == 'user' %}{{ m.content }}{%- endif %}{%- endfor %}",
       "eos_token": "<|im_end|>", "bos_token": null}
      JSON
      ./target/debug/cascadia run ./mock-tools-model --engine mock --api 127.0.0.1:8676

  (A full tools template such as Qwen's also renders the system prompt and the
  tool list, which the mock would echo ahead of the call.)
* ``--tools --tool-model``: the same tool checks against a real model that can
  call tools (e.g. a Qwen2.5 Instruct). It is asked "What's the weather in
  Paris? Use the tool." at temperature 0 instead of being fed the call.

``--self-test`` needs no server: it feeds canned bad streams (``n_tokens`` on a
chunk, the role on every delta, no ``[DONE]``, a changing fingerprint, a tool
call sent whole) through the same checks and confirms each one fails.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

FP_RE = re.compile(r"^fp_[0-9a-f]{10}$")
TIMEOUT = 300
TOOL_ARGS = {"city": "Paris"}
# Spaces split the mock's echo into several words, so the arguments arrive as several fragments.
MOCK_TOOL_CALL = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
REAL_TOOL_PROMPT = "What's the weather in Paris? Use the tool."
WEATHER_PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": WEATHER_PARAMS}}]
FUNCTIONS = [{"name": "get_weather", "parameters": WEATHER_PARAMS}]


def short(value, limit=300):
    text = value if isinstance(value, str) else json.dumps(value)
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------- transport


def http_post(base, path, body):
    """POST ``body`` as JSON; return (status, raw text)."""
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def first_model(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"][0]["id"]
    except Exception:  # noqa: BLE001 - any failure falls back to a placeholder id
        return "default"


# ---------------------------------------------------------------- SSE parsing


def parse_sse(text):
    """Split an SSE body into its ``data:`` payloads; return (payloads, problems).

    Each event must be exactly one ``data:`` line; the only other lines allowed
    are ``:`` comments (keep-alives) and the blank separators.
    """
    problems = []
    payloads = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        lines = [line for line in block.split("\n") if line != ""]
        data_lines = []
        for line in lines:
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
            else:
                problems.append("non-data SSE line %s" % short(line, 80))
        if len(data_lines) > 1:
            problems.append("an event with %d data: lines" % len(data_lines))
        payloads.extend(data_lines)
    if not payloads or payloads[-1] != "[DONE]":
        problems.append("stream does not end with data: [DONE]")
    elif payloads.count("[DONE]") != 1:
        problems.append("[DONE] sent more than once")
    return payloads, problems


def stream_chunks(text):
    """(chunks, problems): the JSON chunks of an SSE body, framing problems first."""
    payloads, problems = parse_sse(text)
    chunks = []
    for p in payloads:
        if p == "[DONE]":
            continue
        try:
            chunk = json.loads(p)
        except ValueError:
            problems.append("data is not JSON: %s" % short(p, 80))
            continue
        if not isinstance(chunk, dict):
            problems.append("data is not an object: %s" % short(p, 80))
            continue
        if chunk.get("object") == "error" or "error" in chunk:
            problems.append("error event: %s" % short(chunk))
            continue
        chunks.append(chunk)
    if not chunks:
        problems.append("no chunks")
    return chunks, problems


# ---------------------------------------------------------------- shape checks


def forbidden_keys(node, path=""):
    """Paths of ``n_tokens`` / ``completion_tokens`` / ``token_id(s)`` outside ``usage``."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = path + "." + key
            if key == "usage":
                continue
            if key in ("n_tokens", "completion_tokens", "token_id", "token_ids"):
                found.append(here)
            found.extend(forbidden_keys(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(forbidden_keys(value, "%s[%d]" % (path, i)))
    return found


def usage_problems(usage, where):
    if not isinstance(usage, dict):
        return ["%s: usage is not an object: %s" % (where, short(usage))]
    out = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(usage.get(key), int):
            out.append("%s: usage.%s is not an integer" % (where, key))
    for key in ("prompt_tokens_details", "completion_tokens_details"):
        if not isinstance(usage.get(key), dict):
            out.append("%s: usage.%s is not an object" % (where, key))
    return out


def logprobs_problems(choice, where):
    """``logprobs`` must be present, and null or ``{content: [...]}`` of OpenAI items."""
    if "logprobs" not in choice:
        return ["%s: choice has no logprobs key" % where]
    lp = choice["logprobs"]
    if lp is None:
        return []
    if not isinstance(lp, dict):
        return ["%s: logprobs is neither null nor an object" % where]
    out = []
    for item in lp.get("content") or []:
        if not isinstance(item, dict) or not {"token", "logprob", "bytes", "top_logprobs"} <= set(item):
            out.append("%s: logprobs item is not {token, logprob, bytes, top_logprobs}: %s" % (where, short(item, 120)))
            break
    return out


def check_stream(text, *, kind="chat", include_usage, continuous=False, finishes=("stop", "length")):
    """Every shape rule for one stream. Returns (problems, chunks)."""
    chunks, problems = stream_chunks(text)
    if not chunks:
        return problems, chunks
    want_object = "chat.completion.chunk" if kind == "chat" else "text_completion"
    fps = set()
    for i, c in enumerate(chunks):
        where = "chunk %d" % i
        if c.get("object") != want_object:
            problems.append("%s: object %s, want %s" % (where, short(c.get("object")), want_object))
        fp = c.get("system_fingerprint")
        if not isinstance(fp, str) or not FP_RE.match(fp):
            problems.append("%s: system_fingerprint %s is not fp_ + 10 hex" % (where, short(fp)))
        fps.add(json.dumps(fp))
        bad = forbidden_keys(c)
        if bad:
            problems.append("%s: carries %s" % (where, ", ".join(bad)))
        for choice in c.get("choices") or []:
            problems.extend(logprobs_problems(choice, where))
    if len(fps) > 1:
        problems.append("system_fingerprint changes within the stream: %s" % ", ".join(sorted(fps)))
    for key in ("id", "created"):
        values = {json.dumps(c.get(key)) for c in chunks}
        if len(values) != 1:
            problems.append("%s is not constant: %s" % (key, ", ".join(sorted(values))))

    body = chunks
    if include_usage:
        last = chunks[-1]
        if last.get("choices") != []:
            problems.append("last chunk is not the usage chunk (choices: [])")
        problems.extend(usage_problems(last.get("usage"), "usage chunk"))
        body = chunks[:-1]
        prev = -1
        for i, c in enumerate(body):
            if "usage" not in c:
                problems.append("chunk %d: no usage key with include_usage" % i)
            elif continuous:
                u = c["usage"]
                if not isinstance(u, dict):
                    problems.append("chunk %d: usage %s, want an object (continuous_usage_stats)" % (i, short(u)))
                    continue
                n = u.get("completion_tokens")
                if not isinstance(n, int) or n < prev:
                    problems.append("chunk %d: running completion_tokens %s after %s" % (i, short(n), prev))
                else:
                    prev = n
            elif c["usage"] is not None:
                problems.append("chunk %d: usage %s before the usage chunk, want null" % (i, short(c["usage"])))
    else:
        for i, c in enumerate(chunks):
            if "usage" in c:
                problems.append("chunk %d: usage key without include_usage" % i)

    with_choices = [c for c in body if c.get("choices")]
    if not with_choices:
        problems.append("no chunk with a choice")
    if kind == "chat":
        for i, c in enumerate(with_choices):
            delta = (c["choices"][0] or {}).get("delta")
            if not isinstance(delta, dict):
                problems.append("choice chunk %d: no delta object" % i)
                continue
            if i == 0 and delta.get("role") != "assistant":
                problems.append('first delta has no role "assistant": %s' % short(delta, 120))
            if i > 0 and "role" in delta:
                problems.append("choice chunk %d repeats delta.role" % i)
    found = [
        (i, ch.get("finish_reason"))
        for i, c in enumerate(with_choices)
        for ch in c["choices"]
        if ch.get("finish_reason") is not None
    ]
    if len(found) != 1:
        problems.append("%d non-null finish_reason values, want 1: %s" % (len(found), short(found)))
    else:
        i, reason = found[0]
        if reason not in finishes:
            problems.append("finish_reason %s, want one of %s" % (short(reason), "/".join(finishes)))
        if i != len(with_choices) - 1:
            problems.append("finish_reason is not on the last choice chunk")
    for c in with_choices:
        for ch in c["choices"]:
            if "finish_reason" not in ch:
                problems.append("a choice has no finish_reason key (want null until the end)")
                break
    return problems, chunks


def check_nonstream_chat(text):
    try:
        resp = json.loads(text)
    except ValueError:
        return ["not JSON: %s" % short(text)]
    out = []
    fp = resp.get("system_fingerprint")
    if not isinstance(fp, str) or not FP_RE.match(fp):
        out.append("system_fingerprint %s is not fp_ + 10 hex" % short(fp))
    out.extend(usage_problems(resp.get("usage"), "response"))
    choices = resp.get("choices") or []
    if not choices:
        out.append("no choices")
    else:
        out.extend(logprobs_problems(choices[0], "choices[0]"))
    bad = forbidden_keys(resp)
    if bad:
        out.append("carries %s" % ", ".join(bad))
    return out


def tool_deltas(chunks, legacy):
    """The tool-call deltas of a stream, in order, as (index, delta) pairs."""
    out = []
    for c in chunks:
        for ch in c.get("choices") or []:
            delta = ch.get("delta") or {}
            if legacy:
                if "function_call" in delta:
                    out.append((0, delta["function_call"]))
            else:
                for tc in delta.get("tool_calls") or []:
                    out.append((tc.get("index") if isinstance(tc, dict) else None, tc))
    return out


def check_tool_stream(chunks, legacy, want_args):
    """Header delta, then argument-only fragments; arguments equal ``want_args``."""
    problems = []
    deltas = tool_deltas(chunks, legacy)
    if not deltas:
        return ["no %s delta in the stream" % ("function_call" if legacy else "tool_calls")]
    args = ""
    seen_header = False
    for n, (index, d) in enumerate(deltas):
        if not isinstance(d, dict):
            problems.append("delta %d is not an object" % n)
            continue
        fn = d if legacy else d.get("function")
        if not isinstance(fn, dict):
            problems.append("delta %d has no function object: %s" % (n, short(d, 120)))
            continue
        if index != 0:
            continue
        if not seen_header:
            seen_header = True
            if legacy:
                if fn.get("name") != "get_weather" or fn.get("arguments") != "":
                    problems.append('first function_call delta is not {name: "get_weather", arguments: ""}: %s' % short(d, 160))
            else:
                if not d.get("id") or d.get("type") != "function" or fn.get("name") != "get_weather" or fn.get("arguments") != "":
                    problems.append(
                        'first tool delta is not {index, id, type: "function", function: {name, arguments: ""}}: %s'
                        % short(d, 160)
                    )
            continue
        want_keys = {"arguments"} if legacy else {"index", "function"}
        if set(d) != want_keys or (not legacy and set(fn) != {"arguments"}):
            problems.append("fragment %d carries more than the arguments: %s" % (n, short(d, 160)))
        if isinstance(fn.get("arguments"), str):
            args += fn["arguments"]
    if len(deltas) < 2:
        problems.append("the call came as one delta, not a header plus fragments")
    try:
        parsed = json.loads(args)
    except ValueError:
        problems.append("concatenated arguments are not JSON: %s" % short(args))
    else:
        if parsed != want_args:
            problems.append("arguments %s, the non-stream reply has %s" % (short(parsed), short(want_args)))
    return problems


# ---------------------------------------------------------------- the live run


class Run:
    def __init__(self):
        self.fails = 0

    def verdict(self, name, problems):
        if problems:
            self.fails += 1
            print("FAIL %s: %s" % (name, "; ".join(problems[:4]) + (" (+%d more)" % (len(problems) - 4) if len(problems) > 4 else "")))
        else:
            print("PASS %s" % name)
        sys.stdout.flush()


def post_checked(base, path, body):
    """(text, problems): a non-200 is itself the problem."""
    status, text = http_post(base, path, body)
    if status != 200:
        return text, ["HTTP %s: %s" % (status, short(text))]
    return text, []


def live(args):
    base = args.base
    model = args.model or first_model(base)
    print("server %s, model %s" % (base, model))
    run = Run()
    msgs = [{"role": "user", "content": "alpha bravo charlie delta echo"}]
    chat = {"model": model, "messages": msgs, "max_tokens": 16, "temperature": 0}

    def stream_case(name, path, body, **kw):
        text, problems = post_checked(base, path, body)
        chunks = []
        if not problems:
            more, chunks = check_stream(text, **kw)
            problems = more
        run.verdict(name, problems)
        return chunks

    stream_case(
        "chat stream, include_usage: framing, role once, fingerprint, ids, logprobs, usage chunk, one finish",
        "/v1/chat/completions",
        dict(chat, stream=True, stream_options={"include_usage": True}),
        include_usage=True,
    )
    stream_case(
        "chat stream without include_usage: no usage key on any chunk",
        "/v1/chat/completions",
        dict(chat, stream=True),
        include_usage=False,
    )
    stream_case(
        "chat stream, continuous_usage_stats: usage on every chunk, running count never drops",
        "/v1/chat/completions",
        dict(chat, stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True}),
        include_usage=True,
        continuous=True,
    )
    stream_case(
        "completions stream: text_completion chunks, logprobs key, usage chunk",
        "/v1/completions",
        {"model": model, "prompt": "alpha bravo charlie delta echo", "max_tokens": 16, "temperature": 0,
         "stream": True, "stream_options": {"include_usage": True}},
        kind="completions",
        include_usage=True,
    )
    text, problems = post_checked(base, "/v1/chat/completions", chat)
    run.verdict(
        "non-stream chat: system_fingerprint, usage details objects, choices[0].logprobs key",
        problems or check_nonstream_chat(text),
    )
    chunks = stream_case(
        "chat stream with logprobs: true: logprobs object or null on every choice",
        "/v1/chat/completions",
        dict(chat, stream=True, logprobs=True, top_logprobs=2),
        include_usage=False,
    )
    text, problems = post_checked(base, "/v1/chat/completions", dict(chat, logprobs=True, top_logprobs=2))
    if not problems:
        problems = check_nonstream_chat(text) + ["non-stream carries %s" % p for p in forbidden_keys(json.loads(text))]
    run.verdict("non-stream chat with logprobs: true: logprobs key, no token_id anywhere", problems)

    if args.tools:
        tool_msgs = [{"role": "user", "content": REAL_TOOL_PROMPT if args.tool_model else MOCK_TOOL_CALL}]
        base_req = {"model": model, "messages": tool_msgs, "temperature": 0, "max_tokens": 128}
        for legacy in (False, True):
            label = "functions (legacy)" if legacy else "tools"
            req = dict(base_req, functions=FUNCTIONS) if legacy else dict(base_req, tools=TOOLS)
            text, problems = post_checked(base, "/v1/chat/completions", req)
            want = None
            if not problems:
                problems = check_nonstream_chat(text)
                msg = ((json.loads(text).get("choices") or [{}])[0]).get("message") or {}
                call = msg.get("function_call") if legacy else (msg.get("tool_calls") or [{}])[0].get("function")
                try:
                    want = json.loads(call["arguments"])
                except (TypeError, KeyError, ValueError):
                    problems.append("non-stream reply has no parsable call: %s" % short(msg))
            run.verdict("%s non-stream: a get_weather call with JSON arguments" % label, problems)

            finish = "function_call" if legacy else "tool_calls"
            text, problems = post_checked(
                base, "/v1/chat/completions", dict(req, stream=True, stream_options={"include_usage": True})
            )
            chunks = []
            if not problems:
                problems, chunks = check_stream(text, include_usage=True, finishes=(finish,))
            run.verdict("%s stream: general shape, finish_reason %s" % (label, finish), problems)
            if chunks:
                problems = check_tool_stream(chunks, legacy, want if want is not None else TOOL_ARGS)
                if not args.tool_model and want is not None and want != TOOL_ARGS:
                    problems.append("non-stream arguments %s, want %s" % (short(want), short(TOOL_ARGS)))
                run.verdict(
                    "%s stream: header delta then argument-only fragments, arguments equal the non-stream reply"
                    % label,
                    problems,
                )
            else:
                run.verdict("%s stream: header delta then argument-only fragments" % label, ["no chunks to inspect"])

    if run.fails:
        print("OPENAI STREAM CHECK: FAILED (%d)" % run.fails)
        return 1
    print("OPENAI STREAM CHECK: ALL PASS")
    return 0


# ---------------------------------------------------------------- self-test


def sse(*events, done=True):
    body = "".join("data: %s\n\n" % json.dumps(e) for e in events)
    return body + ("data: [DONE]\n\n" if done else "")


def good_chat_events(**override):
    fp = "fp_0123456789"
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "system_fingerprint": fp, "usage": None}
    first = dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": "a "},
                                 "logprobs": None, "finish_reason": None}])
    mid = dict(base, choices=[{"index": 0, "delta": {"content": "b "}, "logprobs": None, "finish_reason": None}])
    end = dict(base, choices=[{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}])
    usage = dict(base, choices=[], usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4,
                                          "prompt_tokens_details": {}, "completion_tokens_details": {}})
    events = [first, mid, end, usage]
    for i, patch in override.items():
        events[int(i[1:])] = dict(events[int(i[1:])], **patch)
    return events


def tool_chunks(deltas):
    base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "system_fingerprint": "fp_0123456789"}
    return [dict(base, choices=[{"index": 0, "delta": {"tool_calls": [d]}, "logprobs": None, "finish_reason": None}])
            for d in deltas]


def self_test():
    fails = 0

    def expect(name, problems, want_fail):
        nonlocal fails
        ok = bool(problems) == want_fail
        if not ok:
            fails += 1
        print("%s %s%s" % ("PASS" if ok else "FAIL", name, (": " + "; ".join(problems[:2])) if problems else ""))

    good = sse(*good_chat_events())
    expect("self-test: a well-formed stream passes", check_stream(good, include_usage=True)[0], False)
    ev = good_chat_events(e1={"n_tokens": 1, "completion_tokens": 2})
    expect("self-test: a chunk with n_tokens / completion_tokens fails", check_stream(sse(*ev), include_usage=True)[0], True)
    ev = good_chat_events()
    ev[1]["choices"][0]["delta"]["role"] = "assistant"
    expect("self-test: delta.role on every delta fails", check_stream(sse(*ev), include_usage=True)[0], True)
    expect("self-test: no [DONE] fails", check_stream(sse(*good_chat_events(), done=False), include_usage=True)[0], True)
    expect("self-test: two data: lines in one event fail",
           check_stream(good.replace("data: [DONE]", "data: {}\ndata: [DONE]"), include_usage=True)[0], True)
    expect("self-test: an event: line fails", check_stream("event: x\n" + good, include_usage=True)[0], True)
    expect("self-test: a : keep-alive comment passes", check_stream(": keep-alive\n\n" + good, include_usage=True)[0], False)
    ev = good_chat_events(e2={"system_fingerprint": "fp_aaaaaaaaaa"})
    expect("self-test: a changing system_fingerprint fails", check_stream(sse(*ev), include_usage=True)[0], True)
    ev = good_chat_events(e0={"system_fingerprint": "fp_xyz"}, e1={"system_fingerprint": "fp_xyz"},
                          e2={"system_fingerprint": "fp_xyz"}, e3={"system_fingerprint": "fp_xyz"})
    expect("self-test: a malformed fingerprint fails", check_stream(sse(*ev), include_usage=True)[0], True)
    ev = good_chat_events(e1={"created": 2})
    expect("self-test: a changing created fails", check_stream(sse(*ev), include_usage=True)[0], True)
    ev = good_chat_events()
    del ev[1]["choices"][0]["logprobs"]
    expect("self-test: a choice without logprobs fails", check_stream(sse(*ev), include_usage=True)[0], True)
    ev = good_chat_events()
    ev[3]["usage"] = {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    expect("self-test: usage without the details objects fails", check_stream(sse(*ev), include_usage=True)[0], True)
    ev = good_chat_events()
    del ev[0]["usage"]
    expect("self-test: a chunk with no usage key under include_usage fails",
           check_stream(sse(*ev), include_usage=True)[0], True)
    expect("self-test: usage keys without include_usage fail", check_stream(good, include_usage=False)[0], True)
    ev = good_chat_events()
    ev[1]["choices"][0]["finish_reason"] = "stop"
    expect("self-test: two finish_reason values fail", check_stream(sse(*ev), include_usage=True)[0], True)
    ev = good_chat_events()
    ev[0]["usage"] = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    ev[1]["usage"] = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    ev[2]["usage"] = {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    expect("self-test: a running completion count that drops fails",
           check_stream(sse(*ev), include_usage=True, continuous=True)[0], True)
    expect("self-test: error event fails",
           check_stream(sse({"id": "x", "object": "error", "error": {"message": "boom"}}), include_usage=False)[0], True)

    header = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
    frags = [{"index": 0, "function": {"arguments": '{"city": '}}, {"index": 0, "function": {"arguments": '"Paris"}'}}]
    expect("self-test: header then argument fragments pass", check_tool_stream(tool_chunks([header] + frags), False, TOOL_ARGS), False)
    whole = dict(header, function={"name": "get_weather", "arguments": '{"city": "Paris"}'})
    expect("self-test: a call sent whole in one delta fails", check_tool_stream(tool_chunks([whole]), False, TOOL_ARGS), True)
    frag_with_id = dict(frags[0], id="call_1", type="function")
    expect("self-test: a fragment repeating id/type fails",
           check_tool_stream(tool_chunks([header, frag_with_id, frags[1]]), False, TOOL_ARGS), True)
    expect("self-test: a tool delta without a function object fails",
           check_tool_stream(tool_chunks([header, {"index": 0}] + frags), False, TOOL_ARGS), True)
    expect("self-test: arguments that differ from the non-stream reply fail",
           check_tool_stream(tool_chunks([header] + frags), False, {"city": "Rome"}), True)
    expect("self-test: a token_id in logprobs fails",
           ["x"] if forbidden_keys({"choices": [{"logprobs": {"content": [{"token_id": 3}]}}]}) else [], True)

    if fails:
        print("OPENAI STREAM CHECK: FAILED (%d)" % fails)
        return 1
    print("OPENAI STREAM CHECK: ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Live check of the OpenAI stream shape (stdlib only).")
    ap.add_argument("--base", help="server base URL, e.g. http://127.0.0.1:8676")
    ap.add_argument("--model", help="model id to send (default: the first of /v1/models)")
    ap.add_argument("--tools", action="store_true", help="also check streamed tool calls")
    ap.add_argument("--tool-model", action="store_true",
                    help="with --tools: a real model is asked to call the tool (default: the mock echoes the call)")
    ap.add_argument("--self-test", action="store_true", help="check the checks on canned bad streams; no server")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.base:
        ap.error("--base is required (or --self-test)")
    return live(args)


if __name__ == "__main__":
    sys.exit(main())
