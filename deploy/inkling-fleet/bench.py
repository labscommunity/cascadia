#!/usr/bin/env python3
"""Throughput demo: N simultaneous streamed chat requests against the pipeline head.

    python3 bench.py http://192.168.50.10:8000 --streams 16 --tokens 64 [--rounds 2]

Per request: wall time, time to first token, tokens, steady-state tokens/s.
Per round: aggregate tokens/s (all generated tokens over the round's wall time)
and the sum of the per-stream steady-state rates.
"""
import argparse, json, sys, time, threading, urllib.error, urllib.request

PROMPTS = ["Explain in three sentences why the sky is blue.", "What is the capital of France? Answer in one word.",
           "Write two sentences about the Pacific Ocean.", "List three prime numbers and say why they are prime.",
           "Describe a cat in two sentences.", "What is 6 times 7? Explain briefly.", "Name two planets and one fact about each.",
           "Give one tip for sleeping better, in two sentences.", "Why is the ocean salty? Two sentences.",
           "Summarise photosynthesis in two sentences.", "What does a compiler do? Two sentences.", "Describe rain in one sentence.",
           "What is a haiku? Give one.", "Explain gravity to a child in two sentences.", "Name a famous painting and its painter.",
           "What is the boiling point of water? Explain."]


# The pipeline is on the LAN: never through a web proxy (boxes with http_proxy set would send the fleet's
# names to it and get HTTP 504).
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))


FILLER = ("The harbour town woke slowly that morning: gulls over the quay, a baker carrying trays, two "
          "fishermen arguing about the tide, and a child counting the boats as they left one by one. ")


def one(base, i, tokens, out, prompt_words=0):
    prompt = PROMPTS[i % len(PROMPTS)]
    if prompt_words:  # a long prompt, to time prefill: filler text, then the question
        words = (FILLER * (prompt_words // len(FILLER.split()) + 1)).split()[:prompt_words]
        prompt = "Read this, then answer the question after it.\n\n" + " ".join(words) + "\n\n" + prompt
    body = json.dumps({"model": "inkling", "stream": True, "max_tokens": tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; n = 0; text = []
    try:
        r = None
        for attempt in range(20):  # the API answers 503 while at capacity; back off and retry
            try:
                r = urllib.request.urlopen(req, timeout=36000)
                break
            except urllib.error.HTTPError as e:
                if e.code != 503 or attempt == 19:
                    raise
                time.sleep(0.5 + 0.25 * attempt)
        with r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    v = json.loads(data)
                except Exception:
                    continue
                if "error" in v and not v.get("choices"):
                    raise RuntimeError(f"server error: {str(v['error'])[:120]}")
                ch = v.get("choices", [{}])[0]
                d = ch.get("delta") or {}
                if ch.get("finish_reason") is None and d != {}:
                    now = time.time(); first = first or now; last = now; n += 1
                    if d.get("content"):
                        text.append(d["content"])
        wall = time.time() - t0
        out[i] = dict(i=i, wall_s=round(wall, 2), ttft_s=round((first or t0) - t0, 2), tokens=n,
                      stream_tok_s=round((n - 1) / (last - first), 2) if n > 1 and last > first else None,
                      text="".join(text)[:80])
    except Exception as e:
        out[i] = dict(i=i, wall_s=round(time.time() - t0, 2), error=str(e)[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://192.168.50.10:8000")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1, help="untimed rounds first (fills the expert caches)")
    ap.add_argument("--prompt-words", type=int, default=0, help="pad every prompt with this many words of text (times prefill)")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    for r in range(a.warmup + a.rounds):
        out = [None] * a.streams
        ths = [threading.Thread(target=one, args=(base, i, a.tokens, out, a.prompt_words)) for i in range(a.streams)]
        t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
        ok = [x for x in out if x and "error" not in x]
        tot = sum(x["tokens"] for x in ok)
        tag = "warm-up" if r < a.warmup else f"round {r - a.warmup + 1}"
        print(f"[{tag}] streams={a.streams} completed={len(ok)}/{a.streams} tokens={tot} wall={wall:.1f}s "
              f"aggregate={tot / wall:.2f} tok/s  sum_of_streams={sum(x['stream_tok_s'] or 0 for x in ok):.2f} tok/s  "
              f"ttft mean={sum(x['ttft_s'] for x in ok) / max(1, len(ok)):.1f}s max={max([x['ttft_s'] for x in ok] or [0]):.1f}s")
        for x in out:
            if x and "error" in x:
                print("   error:", x["error"])
        if r >= a.warmup:
            for x in ok[:3]:
                print(f"   stream {x['i']}: {x['stream_tok_s']} tok/s, ttft {x['ttft_s']} s: {x['text']!r}")


if __name__ == "__main__":
    sys.exit(main())
