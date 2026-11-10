#!/usr/bin/env python3
"""Live check of `response_format` (structured output) against a running server.

Sends a fixed set of chat requests to an OpenAI-compatible base URL and prints one
`VERDICT <case> PASS|FAIL|INFO …` line per case, then `STRUCTURED OUTPUT CHECK: ALL PASS`
or `STRUCTURED OUTPUT CHECK: <n> FAILED`. Exit code 0 only when every case passes.

Usage:
    python3 scripts/structured_output_check.py http://127.0.0.1:8000 [--mock] [--cases a,b,…] [--concurrency N]
    python3 scripts/structured_output_check.py http://127.0.0.1:8000 --expect-unavailable

`--expect-unavailable` is for an ov-genai server that reports structured output unavailable
(started with `--draft-model`, or on NPU): JSON requests must get the 400 that says so, and a
plain request must still answer 200. It replaces the normal cases.

`--mock` is for `cascadia run mock-model --engine mock`: the mock answers any schema with a
minimal instance and has no chat template, so the two cases only a real backend can pass
(a schema the backend can't build → 400; tools rendered by the model's template) report INFO.

Answers are validated with the `jsonschema` package (`pip install jsonschema`). Without it
the run fails (`VERDICT jsonschema_installed FAIL`), except with `--mock`, where answers are
only parsed. Uses only the standard library otherwise. Schemas come from
`crates/cascadia-engine-mock/tests/fixtures/` (browser-use and pydantic-ai shapes).
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "crates", "cascadia-engine-mock", "tests", "fixtures")

try:
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover - depends on the environment
    jsonschema = None

PERSON = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def load_fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def post(base, body, timeout=900):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        status = e.code
    return status, raw, time.time() - t0


def chat(messages, **extra):
    body = {"model": "m", "messages": messages, "temperature": 0}
    body.update(extra)
    return body


def rf_schema(schema, name="out"):
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


def content_of(raw):
    obj = json.loads(raw)
    choice = obj["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason")


def stream_content(raw):
    text, finish = [], None
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        for ch in chunk.get("choices", []):
            delta = ch.get("delta", {})
            if delta.get("content"):
                text.append(delta["content"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return "".join(text), finish


def check_json(text, schema):
    """Return (ok, detail)."""
    try:
        value = json.loads(text)
    except ValueError as e:
        return False, f"not JSON ({e}): {text[:120]!r}"
    if schema is None:
        return True, f"json={json.dumps(value)[:120]}"
    if jsonschema is None:
        return True, "parsed (jsonschema not installed: schema NOT checked)"
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as e:
        return False, f"schema violation: {e.message}"
    return True, f"valid json={json.dumps(value)[:120]}"


RESULTS = []
MOCK = False


def verdict(case, ok, detail, info=False):
    tag = "INFO" if info else ("PASS" if ok else "FAIL")
    print(f"VERDICT {case} {tag} {detail}", flush=True)
    if not info:
        RESULTS.append(ok)


def run_json_case(base, case, schema, rf, messages, stream=False, info=False, **extra):
    body = chat(messages, response_format=rf, stream=stream, **extra)
    status, raw, dt = post(base, body)
    if status != 200:
        verdict(case, False, f"HTTP {status}: {raw[:200]}")
        return None
    text, finish = stream_content(raw) if stream else content_of(raw)
    ok, detail = check_json(text, schema)
    if ok and finish != "stop":
        ok, detail = False, f"finish_reason={finish} (want stop); {detail}"
    verdict(case, ok, f"{dt:.2f}s finish={finish} {detail}", info=info)
    return text if ok else None


def case_json_object(base):
    body = chat(
        [{"role": "user", "content": "Describe a cat as a short JSON object with three keys."}],
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    status, raw, dt = post(base, body)
    if status != 200:
        return verdict("json_object", False, f"HTTP {status}: {raw[:200]}")
    text, finish = content_of(raw)
    ok, detail = check_json(text, {"type": "object"})
    verdict("json_object", ok and finish == "stop", f"{dt:.2f}s finish={finish} {detail}")
    try:
        verdict("json_object_keys", True, f"keys={len(json.loads(text))}", info=True)
    except ValueError:
        pass


def case_prose_prompt(base):
    run_json_case(
        base,
        "prose_prompt_still_json",
        PERSON,
        rf_schema(PERSON, "person"),
        [{"role": "user", "content": "Write one sentence about cats."}],
        max_tokens=256,
    )


def case_person(base):
    run_json_case(
        base,
        "json_schema_person",
        PERSON,
        rf_schema(PERSON, "person"),
        [{"role": "user", "content": "Ada Lovelace was 36. Return her name and age."}],
        max_tokens=256,
    )


def case_person_stream(base):
    run_json_case(
        base,
        "json_schema_stream",
        PERSON,
        rf_schema(PERSON, "person"),
        [{"role": "user", "content": "Ada Lovelace was 36. Return her name and age."}],
        stream=True,
        max_tokens=256,
    )


def case_browser_use(base):
    schema = load_fixture("browser_use_schema.json")
    text = run_json_case(
        base,
        "browser_use_schema",
        schema,
        rf_schema(schema, "AgentOutput"),
        [
            {"role": "system", "content": "You are a browser agent. Reply with the next action as JSON."},
            {"role": "user", "content": "Page: [3]<button>Sign in</button>. Task: sign in."},
        ],
        max_tokens=4096,
    )
    if text is None:
        return
    # The answer's keys must come in the schema's order (thinking first, action last): an agent
    # reads its fields in that order. The mock emits sorted keys, so this is INFO there.
    keys = list(json.loads(text))
    want = [k for k in schema["properties"] if k in keys]
    verdict("browser_use_key_order", keys == want, f"keys={keys}", info=MOCK)


def case_browser_use_exact(base):
    """browser-use's own sampling and limit field (max_completion_tokens only)."""
    schema = load_fixture("browser_use_schema.json")
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "You are a browser agent. Reply with the next action as JSON."},
            {"role": "user", "content": "Page: [3]<button>Sign in</button>. Task: sign in."},
        ],
        "temperature": 0.2,
        "frequency_penalty": 0.3,
        "max_completion_tokens": 4096,
        "response_format": rf_schema(schema, "AgentOutput"),
    }
    status, raw, dt = post(base, body)
    if status != 200:
        return verdict("browser_use_exact_body", False, f"HTTP {status}: {raw[:200]}", info=True)
    text, finish = content_of(raw)
    ok, detail = check_json(text, schema)
    # Informational: a server without max_completion_tokens support caps this at its default limit.
    verdict("browser_use_exact_body", ok, f"{dt:.2f}s finish={finish} {detail}", info=True)


BOUNDED = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "n": {"type": "integer", "minimum": 0, "maximum": 9},
        "color": {"enum": ["red", "green", "blue"]},
    },
    "required": ["ok", "n", "color"],
    "additionalProperties": False,
}


def case_bounded(base):
    # Only finite choices: any model, even one with random weights, must close this object within
    # a few dozen tokens, so a pass shows the grammar alone forces the shape.
    run_json_case(
        base,
        "json_schema_bounded",
        BOUNDED,
        rf_schema(BOUNDED, "choice"),
        [{"role": "user", "content": "Pick a color and a digit."}],
        max_tokens=64,
    )


ZOD_EMAIL = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "email": {
            "type": "string",
            "format": "email",
            # zod 4 (before 4.6) emits this look-ahead pattern for z.email().
            "pattern": "^(?!\\.)(?!.*\\.\\.)([A-Za-z0-9_'+\\-\\.]*)[A-Za-z0-9_+-]@([A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$",
        }
    },
    "required": ["email"],
    "additionalProperties": False,
}


def case_zod_email(base):
    # A pattern the grammar backend can't enforce must be accepted (and logged), not refused.
    run_json_case(
        base,
        "zod_email_pattern",
        ZOD_EMAIL,
        rf_schema(ZOD_EMAIL, "contact"),
        [{"role": "user", "content": "Ada's address is ada@example.com. Return it."}],
        max_tokens=128,
        # The mock's minimal instance ignores `pattern` (an empty string).
        info=MOCK,
    )


def case_pattern_props(base):
    # The largest patternProperties fan-out the server's caps allow (64 keys in each of two objects): the
    # grammar compile must stay quick, whatever the answer.
    def obj(prefix):
        return {"type": "object", "additionalProperties": False,
                "patternProperties": {f"^{prefix}{i}[a-z]*$": {"type": "integer"} for i in range(64)}}
    schema = {"type": "object", "properties": {"a": obj("k"), "b": obj("m")}, "required": ["a", "b"],
              "additionalProperties": False}
    body = chat([{"role": "user", "content": "Return two empty objects a and b."}],
                response_format=rf_schema(schema, "pp"), max_tokens=32)
    status, raw, dt = post(base, body)
    verdict("pattern_properties_compile_fast", status in (200, 400) and dt < 15, f"HTTP {status} {dt:.2f}s {raw[:120]}")


def case_pydantic(base):
    schema = load_fixture("pydantic_output_schema.json")
    run_json_case(
        base,
        "pydantic_native_output",
        schema,
        rf_schema(schema, "Person"),
        [{"role": "user", "content": "Ada Lovelace, 36, lives at 1 Main St, Kathmandu, NP. No email. Tags: math."}],
        max_tokens=1024,
    )


def case_thinking_default(base):
    body = chat(
        [{"role": "user", "content": "Ada Lovelace was 36. Return her name and age."}],
        response_format=rf_schema(PERSON, "person"),
        max_tokens=256,
    )
    body.pop("temperature")  # default sampling, default thinking
    status, raw, dt = post(base, body)
    if status != 200:
        return verdict("thinking_default_off", False, f"HTTP {status}: {raw[:200]}")
    text, finish = content_of(raw)
    ok, detail = check_json(text, PERSON)
    ok = ok and "<think>" not in text
    verdict("thinking_default_off", ok, f"{dt:.2f}s finish={finish} {detail}")


def case_malformed(base):
    body = chat([{"role": "user", "content": "hi"}], response_format={"type": "xml"})
    status, raw, _ = post(base, body)
    ok = status == 400 and "invalid response_format" in raw
    verdict("malformed_400", ok, f"HTTP {status}: {raw[:160]}")


def case_bad_schema(base):
    body = chat(
        [{"role": "user", "content": "hi"}],
        response_format=rf_schema({"type": 42}),
    )
    status, raw, _ = post(base, body)
    ok = status == 400
    verdict("unbuildable_schema_400", ok, f"HTTP {status}: {raw[:200]}", info=MOCK)


def case_cap(base):
    schema = {"type": "object", "properties": {"s": {"type": "string", "maxLength": 20000}}}
    body = chat([{"role": "user", "content": "hi"}], response_format=rf_schema(schema))
    status, raw, _ = post(base, body)
    ok = status == 400 and "exceeds 10000" in raw
    verdict("schema_cap_400", ok, f"HTTP {status}: {raw[:160]}")


def case_ref_depth(base):
    # A long $ref chain used to overflow the grammar converter's stack and kill the worker.
    n = 2000
    defs = {f"d{k}": {"$ref": f"#/$defs/d{k + 1}"} for k in range(n)}
    defs[f"d{n}"] = {"type": "string"}
    body = chat([{"role": "user", "content": "hi"}],
                response_format=rf_schema({"$defs": defs, "$ref": "#/$defs/d0"}), max_tokens=8)
    status, raw, _ = post(base, body)
    verdict("ref_depth_400", status == 400 and "through $ref" in raw, f"HTTP {status}: {raw[:160]}")
    status, raw, dt = post(base, chat([{"role": "user", "content": "Say hello."}], max_tokens=8))
    verdict("server_alive_after_ref_depth", status == 200, f"HTTP {status} {dt:.2f}s")



def case_unsatisfiable_bounds(base):
    # Bounds no value meets: a grammar built from them can never finish, and a backend that
    # builds it anyway decodes until killed. The server must refuse it fast.
    for kind, schema in (
        ("string", {"type": "string", "minLength": 5, "maxLength": 2}),
        ("array", {"type": "array", "minItems": 5, "maxItems": 2}),
    ):
        body = chat([{"role": "user", "content": "hi"}], response_format=rf_schema(schema), max_tokens=8)
        status, raw, dt = post(base, body, timeout=30)
        ok = status == 400 and "invalid response_format" in raw and "no value can satisfy both" in raw and dt < 5
        verdict(f"unsatisfiable_bounds_400_{kind}", ok, f"HTTP {status} {dt:.2f}s: {raw[:200]}")
    status, raw, dt = post(base, chat([{"role": "user", "content": "Say hello."}], max_tokens=8))
    verdict("server_alive_after_unsatisfiable_bounds", status == 200, f"HTTP {status} {dt:.2f}s")

def case_optional_properties(base):
    # 6000 optional properties (65 KB, inside the size cap): llama-server's grammar converter takes
    # time cubic in an object's optional properties (minutes here). The server must answer fast,
    # and stay up.
    schema = {"type": "object", "properties": {f"p{i}": {} for i in range(6000)}}
    body = chat([{"role": "user", "content": "hi"}], response_format=rf_schema(schema), max_tokens=8)
    def timed(b):
        t0 = time.time()
        try:
            return post(base, b, timeout=60)
        except OSError as e:  # a timeout: the converter is still running
            return 0, f"no answer: {e}", time.time() - t0
    status, raw, dt = timed(body)
    a_status, _, a_dt = timed(chat([{"role": "user", "content": "Say hello."}], max_tokens=8))
    ok = status in (200, 400) and dt < 15 and a_status == 200
    verdict("optional_properties_compile_fast", ok,
            f"HTTP {status} {dt:.2f}s {raw[:200]}; then plain HTTP {a_status} {a_dt:.2f}s")


def case_text(base):
    body = chat(
        [{"role": "user", "content": "Say hello."}],
        response_format={"type": "text"},
        max_tokens=16,
    )
    status, raw, dt = post(base, body)
    verdict("text_format_unchanged", status == 200, f"HTTP {status} {dt:.2f}s")


def case_tools(base):
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up a person",
                                                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    body = chat(
        [{"role": "user", "content": "Ada Lovelace was 36. Return her name and age."}],
        response_format=rf_schema(PERSON, "person"),
        tools=tools,
        max_tokens=256,
    )
    status, raw, dt = post(base, body)
    # With callable tools the constraint is not applied (tools keep working): 200 is the pass.
    # A server without a chat template can't render tools at all (its own 400): INFO on the mock.
    verdict("json_with_tools_not_enforced_200", status == 200, f"HTTP {status} {dt:.2f}s {raw[:120]}",
            info=MOCK and status == 400 and "no chat template" in raw)


def case_concurrent(base, n):
    def one(i):
        if i % 2 == 0:
            body = chat([{"role": "user", "content": f"Person number {i} is Ada, 36. Return name and age."}],
                        response_format=rf_schema(PERSON, "person"), max_tokens=256)
            status, raw, _ = post(base, body)
            if status != 200:
                return False, f"#{i} HTTP {status}"
            ok, detail = check_json(content_of(raw)[0], PERSON)
            return ok, f"#{i} json {detail[:40]}"
        body = chat([{"role": "user", "content": f"Count to {i + 3}."}], max_tokens=32)
        status, _, _ = post(base, body)
        return status == 200, f"#{i} plain HTTP {status}"

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(one, range(n)))
    ok = all(r[0] for r in results)
    verdict(f"concurrent_{n}_mixed", ok, f"{time.time() - t0:.2f}s " + "; ".join(r[1] for r in results))


def case_unavailable(base):
    for kind, rf in (("json_object", {"type": "json_object"}), ("json_schema", rf_schema(PERSON, "person"))):
        body = chat([{"role": "user", "content": "hi"}], response_format=rf, max_tokens=16)
        status, raw, _ = post(base, body)
        ok = status == 400 and "structured output is not available" in raw
        verdict(f"unavailable_{kind}_400", ok, f"HTTP {status}: {raw[:240]}")
    body = chat([{"role": "user", "content": "Say hello."}], max_tokens=16)
    status, raw, dt = post(base, body)
    verdict("plain_still_works", status == 200, f"HTTP {status} {dt:.2f}s {raw[:120]}")


CASES = {
    "json_object": case_json_object,
    "prose": case_prose_prompt,
    "person": case_person,
    "stream": case_person_stream,
    "browser_use": case_browser_use,
    "browser_use_exact": case_browser_use_exact,
    "pydantic": case_pydantic,
    "bounded": case_bounded,
    "zod_email": case_zod_email,
    "pattern_props": case_pattern_props,
    "thinking": case_thinking_default,
    "malformed": case_malformed,
    "bad_schema": case_bad_schema,
    "cap": case_cap,
    "ref_depth": case_ref_depth,
    "unsatisfiable_bounds": case_unsatisfiable_bounds,
    "optional_properties": case_optional_properties,
    "text": case_text,
    "tools": case_tools,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--concurrency", type=int, default=0, help="also run N mixed requests at once")
    ap.add_argument("--mock", action="store_true", help="the server runs the mock engine (see the module doc)")
    ap.add_argument("--expect-unavailable", action="store_true",
                    help="the server reports structured output unavailable (see the module doc)")
    args = ap.parse_args()
    global MOCK
    MOCK = args.mock
    if jsonschema is None:
        if MOCK:
            print("WARN jsonschema not installed: answers are parsed but not checked against their schema",
                  flush=True)
        else:
            verdict("jsonschema_installed", False, "install it with pip install jsonschema")
    if args.expect_unavailable:
        case_unavailable(args.base)
    else:
        for name in args.cases.split(","):
            CASES[name](args.base)
    if args.concurrency and not args.expect_unavailable:
        case_concurrent(args.base, args.concurrency)
    failed = RESULTS.count(False)
    print("STRUCTURED OUTPUT CHECK: ALL PASS" if failed == 0 else f"STRUCTURED OUTPUT CHECK: {failed} FAILED", flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
