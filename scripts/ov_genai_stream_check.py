#!/usr/bin/env python3
"""The client and the verdict rules of scripts/ov-genai-stream-check.sh.

The shell script runs the checks; this module sends the requests (timing the
response headers and every content delta on a monotonic clock) and holds the
pure rules each verdict is judged by, so they can be unit-tested without a
node (scripts/test_ov_genai_stream_check.py).

Subcommands, as the shell script calls them:

    body KIND MAX_TOKENS STREAM THINK MODEL       -> the request JSON on stdout
    send URL BODY_FILE OUT_JSON TIMEOUT_S         -> one request, its timings in OUT_JSON
    get FILE KEY                                  -> one field of a record
    summary REC                                   -> one line about a record
    first-byte MAX_S REC...                       -> "ok|bad <median> <runs>"
    deltas PER REC...                             -> "ok|bad <median tokens/delta> <runs>"
    tok-s MIN REC...                              -> "ok|bad <median> <runs>"
    parity STREAMED REF...                        -> "ok|bad|varies <detail>"
    headers MAX_S REC                             -> "ok|bad <seconds> <detail>"
    gap RATIO REC                                 -> "ok|bad <max> <median> <n gaps>"
    ttft LOG_FILE                                 -> the ttft_ms values of 'task done' lines
    divergence LOG_FILE                           -> the count of divergence lines
    median X...                                   -> the median

Only the standard library; Python 3.8+.
"""

import http.client
import json
import re
import sys
import time
import urllib.parse

PROMPTS = {
    # A long, regular answer: every token is visible text, so the delta and
    # tok/s rules measure the stream itself.
    "count": "Count from 1 to 100000 in words, separated by commas.",
    "short": "Reply with the single word ok.",
}

DIVERGENCE = "ov_genai_stream_divergence"


# --- pure rules ---


def median(xs):
    v = sorted(float(x) for x in xs)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def parse_sse(events):
    """Fold timed SSE lines into a record.

    `events` is a list of (t, line) with t seconds after the request was
    sent. Returns the content deltas as [t, text] (only non-empty content),
    the joined content, the finish reason, completion_tokens from the usage
    block, whether [DONE] came, and the first error object.
    """
    r = {
        "deltas": [],
        "content": "",
        "finish": None,
        "completion_tokens": None,
        "done": False,
        "error": None,
        "frames": 0,
    }
    for t, line in events:
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            r["done"] = True
            continue
        try:
            j = json.loads(d)
        except ValueError:
            continue
        r["frames"] += 1
        take(r, j, t)
    return r


def take(r, j, t=None):
    """Add one response object (a stream frame or a whole answer) to `r`."""
    if j.get("error") and r["error"] is None:
        r["error"] = j["error"]
    for c in j.get("choices") or []:
        text = ((c.get("delta") or {}).get("content") or "") + (
            (c.get("message") or {}).get("content") or ""
        )
        if text:
            r["content"] += text
            if t is not None:
                r["deltas"].append([t, text])
        r["finish"] = c.get("finish_reason") or r["finish"]
    u = j.get("usage")
    if u and u.get("completion_tokens") is not None:
        r["completion_tokens"] = u["completion_tokens"]


def first_byte(rec):
    """Seconds from sending to the first content delta (None: no delta)."""
    return rec["deltas"][0][0] if rec.get("deltas") else None


def first_byte_verdict(recs, max_s):
    """Median first content delta over the runs <= max_s; a run with no
    delta counts as infinitely late."""
    xs = [first_byte(r) for r in recs]
    xs = [float("inf") if x is None else x for x in xs]
    m = median(xs) if xs else float("inf")
    return m <= max_s, m, xs


def tokens_per_delta(rec):
    n = len(rec.get("deltas") or [])
    tok = rec.get("completion_tokens") or 0
    if tok == 0:
        return 0.0
    return float("inf") if n == 0 else tok / n


def deltas_verdict(recs, per):
    """At least one content delta per `per` completion tokens: the median
    tokens per delta over the runs <= per. A run without usage counts as
    failing (the rule can't be judged)."""
    xs = []
    for r in recs:
        xs.append(float("inf") if r.get("completion_tokens") is None else tokens_per_delta(r))
    m = median(xs) if xs else float("inf")
    return m <= per, m, xs


def tok_s(rec):
    tok = rec.get("completion_tokens") or 0
    total = rec.get("t_end") or 0
    return tok / total if total > 0 else 0.0


def tok_s_verdict(recs, min_tok_s):
    xs = [tok_s(r) for r in recs]
    m = median(xs) if xs else 0.0
    return m >= min_tok_s, m, xs


def first_diff(a, b):
    """Index of the first differing character (len of the shorter when one
    is a prefix of the other), or -1 when equal."""
    if a == b:
        return -1
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def parity_verdict(streamed, refs):
    """The streamed content equals a stream:false answer byte for byte.

    `refs` are the non-streamed answers of the same request. ok when it
    equals any of them; bad when it equals none and they all agree (the
    device is reproducible, so the stream changed the text); varies when it
    equals none and they disagree among themselves (unverifiable here).
    """
    s = streamed["content"]
    texts = [r["content"] for r in refs]
    if any(s == t for t in texts):
        return "ok", "%d bytes" % len(s.encode("utf-8"))
    t0 = texts[0] if texts else ""
    i = first_diff(s, t0)
    detail = "first difference at char %d: streamed %s vs stream:false %s" % (
        i,
        json.dumps(s[max(0, i - 20) : i + 40]),
        json.dumps(t0[max(0, i - 20) : i + 40]),
    )
    if len(set(texts)) > 1:
        return "varies", detail
    return "bad", detail


def headers_verdict(rec, max_s):
    """The response's status line and headers came <= max_s after it was
    sent, with HTTP 200 and an event stream."""
    t = rec.get("t_headers")
    ctype = rec.get("content_type") or ""
    status = rec.get("status")
    if t is None:
        return False, None, "no response headers (%s)" % (rec.get("error") or "?")
    ok = t <= max_s and status == 200 and ctype.startswith("text/event-stream")
    return ok, t, "HTTP %s content-type %s" % (status, ctype or "-")


def gaps(times):
    return [b - a for a, b in zip(times, times[1:])]


def gap_verdict(rec, ratio):
    """The largest gap between consecutive content deltas <= ratio x their
    median gap. Fewer than two gaps can't be judged (bad)."""
    g = gaps([t for t, _ in rec.get("deltas") or []])
    if len(g) < 2:
        return False, 0.0, 0.0, len(g)
    mx, md = max(g), median(g)
    return mx <= ratio * md, mx, md, len(g)


TTFT_RE = re.compile(r"\bttft_ms=([0-9.]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def ttft_values(log_text):
    """The ttft_ms of every 'task done' line (ov-genai's own time to first
    token, which the engine logs when the shim reports it)."""
    out = []
    for line in ANSI_RE.sub("", log_text).splitlines():
        if "task done" not in line:
            continue
        m = TTFT_RE.search(line)
        if m:
            out.append(float(m.group(1)))
    return out


def count_divergence(log_text):
    return sum(1 for line in log_text.splitlines() if DIVERGENCE in line)


# --- the client ---


def body(kind, max_tokens, stream, think, model):
    b = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[kind]}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
    }
    if think:
        b["enable_thinking"] = False
    if stream:
        b["stream"] = True
        b["stream_options"] = {"include_usage": True}
    return b


def send(url, payload, timeout_s):
    """POST /v1/chat/completions; time the headers, every SSE line and the
    end on a monotonic clock, all relative to the moment the request was
    written. Never raises: a failure is the record's `error`."""
    rec = {
        "t_headers": None,
        "t_end": None,
        "status": None,
        "content_type": None,
        "deltas": [],
        "content": "",
        "finish": None,
        "completion_tokens": None,
        "done": False,
        "error": None,
        "frames": 0,
        "sent_at": time.time(),
    }
    u = urllib.parse.urlsplit(url)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    stream = bool(payload.get("stream"))
    data = json.dumps(payload).encode("utf-8")
    path = (u.path.rstrip("/") or "") + "/v1/chat/completions"
    t0 = time.monotonic()
    try:
        conn = conn_cls(u.hostname, u.port, timeout=timeout_s)
        conn.request("POST", path, body=data, headers={"content-type": "application/json"})
        t0 = time.monotonic()
        resp = conn.getresponse()
        rec["t_headers"] = time.monotonic() - t0
        rec["status"] = resp.status
        rec["content_type"] = resp.getheader("content-type")
        if stream and resp.status == 200:
            events = []
            while True:
                line = resp.readline()
                if not line:
                    break
                events.append((time.monotonic() - t0, line.decode("utf-8", "replace").rstrip("\r\n")))
            rec.update(parse_sse(events))
        else:
            raw = resp.read().decode("utf-8", "replace")
            try:
                take(rec, json.loads(raw))
            except ValueError:
                rec["error"] = "not JSON: " + raw[:160]
        rec["t_end"] = time.monotonic() - t0
        conn.close()
    except Exception as e:  # noqa: BLE001 - the record carries any failure
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        rec["t_end"] = time.monotonic() - t0
    return rec


def summary(rec):
    """One line about a record, for the run's log."""
    def num(x, f="%.3f"):
        return "-" if x is None else f % x
    fb = first_byte(rec)
    tpd = tokens_per_delta(rec)
    streamed = (rec.get("content_type") or "").startswith("text/event-stream")
    parts = ["HTTP %s" % (rec.get("status") or "-"), "headers %ss" % num(rec.get("t_headers"))]
    if streamed:
        parts += [
            "first delta %ss" % num(fb),
            "%d deltas" % len(rec.get("deltas") or []),
        ]
    parts.append("%s tokens" % num(rec.get("completion_tokens"), "%d"))
    if streamed:
        parts.append("%s tokens/delta" % fmt(tpd))
    parts += [
        "finish %s" % (rec.get("finish") or "-"),
        "total %ss" % num(rec.get("t_end")),
        "%.2f tok/s" % tok_s(rec),
    ]
    if rec.get("error"):
        parts.append("error %s" % json.dumps(rec["error"])[:200])
    return ", ".join(parts)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(x):
    return "inf" if x == float("inf") else "%.3f" % x


def main(argv):
    cmd, a = argv[0], argv[1:]
    if cmd == "body":
        kind, mx, stream, think, model = a
        print(json.dumps(body(kind, mx, stream == "1", think == "1", model)))
    elif cmd == "send":
        url, body_p, out, timeout_s = a
        rec = send(url, load(body_p), float(timeout_s))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rec, f)
    elif cmd == "summary":
        print(summary(load(a[0])))
    elif cmd == "get":
        v = load(a[0]).get(a[1])
        print("" if v is None else v)
    elif cmd == "first-byte":
        ok, m, xs = first_byte_verdict([load(p) for p in a[1:]], float(a[0]))
        print("ok" if ok else "bad", fmt(m), ",".join(fmt(x) for x in xs))
    elif cmd == "deltas":
        ok, m, xs = deltas_verdict([load(p) for p in a[1:]], float(a[0]))
        print("ok" if ok else "bad", fmt(m), ",".join(fmt(x) for x in xs))
    elif cmd == "tok-s":
        ok, m, xs = tok_s_verdict([load(p) for p in a[1:]], float(a[0]))
        print("ok" if ok else "bad", fmt(m), ",".join(fmt(x) for x in xs))
    elif cmd == "parity":
        v, detail = parity_verdict(load(a[0]), [load(p) for p in a[1:]])
        print(v, detail)
    elif cmd == "headers":
        ok, t, detail = headers_verdict(load(a[1]), float(a[0]))
        print("ok" if ok else "bad", "-" if t is None else fmt(t), detail)
    elif cmd == "gap":
        ok, mx, md, n = gap_verdict(load(a[1]), float(a[0]))
        print("ok" if ok else "bad", fmt(mx), fmt(md), n)
    elif cmd == "ttft":
        with open(a[0], encoding="utf-8", errors="replace") as f:
            print(" ".join("%.1f" % x for x in ttft_values(f.read())))
    elif cmd == "divergence":
        with open(a[0], encoding="utf-8", errors="replace") as f:
            print(count_divergence(f.read()))
    elif cmd == "median":
        print(fmt(median(a)))
    else:
        print("unknown command: %s" % cmd, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
