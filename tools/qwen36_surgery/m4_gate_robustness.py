"""Robustness matrix gate for the staged qwen36-moe engine (M4'-1 gate 3;
see docs/architectures/qwen36-moe-support.md).

Automated rows:
  cancel     - cancel mid-decode entered at A (/v1/cancel/:id), then a
               follow-up task must be clean
  disconnect - SSE client disconnect mid-decode, follow-up clean
  bleed      - 3 sequential greedy tasks, identical outputs (no state
               bleed across RESET/epoch cycles)
  clean      - one short task, asserts coherent output (used as the
               "next task clean" check after the operator kill rows)

Operator rows (run `longgen`, kill the process mid-decode, then `clean`
after restarting):
  kill B mid-decode -> A must fail the task loud (stream terminates,
      "pipeline decode failed" in rank-0 log)
  kill A mid-decode -> B logs socket error; restart both -> `clean`

Usage: python m4_gate_robustness.py http://127.0.0.1:8000 <row>
"""
import json
import sys
import time
import urllib.request
from urllib.parse import urlparse

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ROW = sys.argv[2] if len(sys.argv) > 2 else "clean"
U = urlparse(BASE)


def sse_start(content, max_tokens):
    """Open a streaming chat; returns (http.client conn, response)."""
    import http.client

    conn = http.client.HTTPConnection(U.hostname, U.port or 80, timeout=600)
    body = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "enable_thinking": True,
            "stream": True,
        }
    )
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    return conn, conn.getresponse()


def sse_read_chunks(resp, n):
    """Read up to n data chunks; returns (task_id, texts, finished)."""
    task_id, texts, buf = None, [], b""
    while len(texts) < n:
        b = resp.read1(4096)
        if not b:
            return task_id, texts, True
        buf += b
        while b"\n\n" in buf:
            line, buf = buf.split(b"\n\n", 1)
            for part in line.split(b"\n"):
                if not part.startswith(b"data: "):
                    continue
                data = part[6:]
                if data == b"[DONE]":
                    return task_id, texts, True
                d = json.loads(data)
                task_id = task_id or d.get("id")
                delta = d["choices"][0].get("delta", {}).get("content")
                if delta:
                    texts.append(delta)
    return task_id, texts, False


def chat(content, max_tokens=32):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(
            {
                "model": "default",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "enable_thinking": False,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def clean(tag="CLEAN"):
    text = chat("What is the capital of Japan? One word.")
    ok = "tokyo" in text.lower()
    print(f"{tag}_{'PASS' if ok else 'FAIL'}: {json.dumps(text[:60])}", flush=True)
    return ok


if ROW == "cancel":
    conn, resp = sse_start("Write a long story about the ocean.", 200)
    task_id, texts, fin = sse_read_chunks(resp, 8)
    print(f"streamed {len(texts)} chunks, task {task_id}; cancelling...", flush=True)
    urllib.request.urlopen(
        urllib.request.Request(BASE + f"/v1/cancel/{task_id}", method="POST"), timeout=30
    )
    t0 = time.perf_counter()
    _, more, _ = sse_read_chunks(resp, 500)
    conn.close()
    print(
        f"stream ended {time.perf_counter() - t0:.1f}s after cancel with {len(more)} more chunks",
        flush=True,
    )
    print("CANCEL_PASS" if len(more) < 30 else "CANCEL_FAIL (kept generating)", flush=True)
    clean("FOLLOWUP")
elif ROW == "disconnect":
    conn, resp = sse_start("Write a long story about mountains.", 200)
    _, texts, _ = sse_read_chunks(resp, 8)
    conn.close()  # hard disconnect; ChunkStream::Drop cancels at A
    print(f"disconnected after {len(texts)} chunks", flush=True)
    time.sleep(3)
    clean("FOLLOWUP")
elif ROW == "bleed":
    outs = [chat("Describe a rainbow in two sentences.", 48) for _ in range(3)]
    same = outs[0] == outs[1] == outs[2]
    print("BLEED_PASS (3 identical)" if same else "BLEED_FAIL:", flush=True)
    if not same:
        for i, o in enumerate(outs):
            print(f"  run{i}: {json.dumps(o[:100])}", flush=True)
elif ROW == "longgen":
    # For the operator kill rows: stream a long generation and report
    # how/when the stream terminates.
    conn, resp = sse_start("Write a very long essay about rivers.", 400)
    task_id, texts, _ = sse_read_chunks(resp, 4)
    print(f"task {task_id} decoding; KILL THE TARGET RANK NOW", flush=True)
    t0 = time.perf_counter()
    try:
        _, more, fin = sse_read_chunks(resp, 10_000)
        print(
            f"stream ended after {len(more)} more chunks, {time.perf_counter() - t0:.1f}s, done={fin}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 - report any termination mode
        print(f"stream raised after {time.perf_counter() - t0:.1f}s: {e!r}", flush=True)
    conn.close()
else:
    clean()
