"""M4'-1 serving gates (run on/against the rank-0 node API).

Gates covered (qwen36-m4-pipeline-spec.md §6):
  1. token agreement vs the blessed single-box golden
     (golden/qwen36_parity_64.json) — text-prefix compare; on divergence
     report the index + both continuations for the M3' §5 near-tie
     coherence judgment (not raw ==).
  2. decode >= 4 tok/s short-ctx (completion_tokens / decode wall;
     cross-check against the engine's `tok_s` log line).
  promptset: the 6-prompt #77 set (expect-substring assertions), saved
     as golden/promptset_<label>.json for cross-path comparison.

Usage:
  python m4_gate_serving.py http://127.0.0.1:8000 parity
  python m4_gate_serving.py http://127.0.0.1:8000 promptset pipeline-2node

Gate 4 (wire histogram) is read from the rank-0 log: the engine emits
"qwen36 pipeline decode wire histogram" (p50/p95/max ms) per task.
"""
import json
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODE = sys.argv[2] if len(sys.argv) > 2 else "parity"


def chat(content, max_tokens, thinking):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "enable_thinking": thinking,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    wall = time.perf_counter() - t0
    text = out["choices"][0]["message"]["content"]
    return text, out.get("usage", {}), wall


def parity():
    g = json.load(open(HERE / "golden" / "qwen36_parity_64.json"))
    # golden prompt is the rendered "user: ..." form; send the raw user
    # content — the engine renders the same legacy form.
    user = g["prompt"].removeprefix("user: ")
    text, usage, wall = chat(user, g["max_tokens"], True)
    ref = g["text"]
    div = next(
        (i for i, (a, b) in enumerate(zip(text, ref)) if a != b),
        min(len(text), len(ref)),
    )
    full_match = text == ref
    ct = usage.get("completion_tokens", 0)
    print(f"completion_tokens={ct} wall={wall:.1f}s tok/s={ct / wall:.2f} (incl. prefill; see engine tok_s log for decode-only)", flush=True)
    if full_match:
        print(f"PARITY_EXACT len={len(text)}", flush=True)
    else:
        print(f"PARITY_DIVERGES at char {div} of {len(ref)} (S5 near-tie judgment needed):", flush=True)
        print("  golden  :", json.dumps(ref[max(0, div - 40):div + 80]), flush=True)
        print("  pipeline:", json.dumps(text[max(0, div - 40):div + 80]), flush=True)
    print("FULL_TEXT_JSON:", json.dumps(text), flush=True)


def promptset(label):
    ps = json.load(open(HERE / "promptset.json"))
    results, passed = [], 0
    for p in ps["prompts"]:
        text, usage, wall = chat(p["content"], p["max_tokens"], p["thinking"])
        ok = p["expect"].lower() in text.lower()
        passed += ok
        results.append({"id": p["id"], "ok": ok, "content": text[:120], "usage": usage})
        print(f"{p['id']}: {'PASS' if ok else 'FAIL'} ({wall:.1f}s) {json.dumps(text[:80])}", flush=True)
    out = {"label": label, "passed": passed, "total": len(results), "results": results}
    path = HERE / "golden" / f"promptset_{label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"PROMPTSET {passed}/{len(results)} -> {path}", flush=True)


if MODE == "parity":
    parity()
else:
    promptset(sys.argv[3] if len(sys.argv) > 3 else "pipeline-2node")
