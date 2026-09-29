#!/usr/bin/env python3
"""context_scan.py EXP [--sizes 1024,4096,12288,32768] [--budget-s 720] [--new-tokens 32] [--streams 1]

The fleet's performance against context length, with REAL prompts, inside a fixed time budget (autolab 034):
for each size, one request whose prompt is natural text of about that many tokens (the fleet's own responses,
013 corpus, then Dolly instructions), with a "needle" sentence near the start and a question at the end that asks
for it. Measured per size: exact prompt tokens (the API's `usage`), time to first token = the prefill (from which
prefill tokens per second), decode tokens per second AT that context, and whether the answer contains the needle
(a light correctness check of attention at that length: the window is 512, the global layers see everything).

Sizes run in ascending order. Before each one the harness predicts its prefill time from the rates measured so
far (or 40 tok/s before any) and SKIPS it if that would pass the budget: the run ends on time whatever the fleet
does, and the skipped sizes are recorded as such. A phase cap of `--cap` seconds guards each request.

Writes experiments/EXP/context_scan.json and prints a table. Standard library only; no fleet state changes.
"""
import argparse, json, os, random, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
API = os.environ.get("INKLING_API", "http://localhost:18000")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHARS_PER_TOKEN = 4.4   # the model's own text measured 4.57; Dolly's a little denser; the API's usage gives the truth


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def text_pool():
    """Natural text in a fixed order (deterministic prompts across runs)."""
    parts = []
    p = os.path.join(LAB, "experiments", "013_acceptance_study", "corpus.jsonl")
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l)
            t = d.get("text", "").replace("<think>", "").replace("</think>", "\n")
            if len(t) > 200:
                parts.append(t.strip())
    p = os.path.join(LAB, "experiments", "016_distill_data", "dolly_corpus.jsonl")
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l)
            t = (d.get("prompt", "") + "\n\n" + d.get("text", "")).strip()
            if len(t) > 200:
                parts.append(t)
    if not parts:
        sys.exit("no text sources found (experiments/013 corpus.jsonl, experiments/016 dolly_corpus.jsonl)")
    return parts


def build_prompt(pool, tokens, seed):
    """About `tokens` tokens of documents, a needle after the first document, the question at the end."""
    rng = random.Random(seed)
    code = "%s-%d" % (rng.choice(["amber", "cobalt", "quartz", "sienna", "umber", "indigo"]), rng.randint(1000, 9999))
    needle = "\n\n[Note for the reader: the access code for this collection is %s. It appears only here.]\n\n" % code
    head = "Below is a collection of documents separated by lines of dashes. Read all of them.\n\n"
    tail = ("\n\n-----\n\nThat is the end of the collection. Answer in one line: what is the access code stated in the note "
            "near the beginning of the collection? Reply with the code only.")
    budget = int(tokens * CHARS_PER_TOKEN) - len(head) - len(needle) - len(tail)
    body, i = [], 0
    n = 0
    while n < budget:
        doc = pool[i % len(pool)]
        body.append(doc)
        n += len(doc) + 9
        i += 1
    docs = "\n\n-----\n\n".join(body)
    # the needle goes after the first document; trim the body to the budget from the end
    first, rest = body[0], "\n\n-----\n\n".join(body[1:])
    rest = rest[: max(0, budget - len(first))]
    return head + first + needle + rest + tail, code


def chat(prompt, new_tokens, cap):
    body = json.dumps({"model": "inkling", "stream": True, "max_tokens": new_tokens, "temperature": 0,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; n = 0; text = []; usage = None; err = None
    try:
        with OPENER.open(req, timeout=cap) as r:
            for line in r:
                if time.time() - t0 > cap:
                    err = "cap"; break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    v = json.loads(data)
                except ValueError:
                    continue
                if v.get("usage"):
                    usage = v["usage"]
                if "error" in v and not v.get("choices"):
                    err = str(v["error"])[:200]; break
                ch = (v.get("choices") or [{}])[0]; d = ch.get("delta") or {}
                if ch.get("finish_reason") is None and d != {}:
                    now = time.time(); first = first or now; last = now; n += 1
                    text.append(d.get("content") or d.get("reasoning_content") or d.get("reasoning") or "")
    except Exception as e:  # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, str(e)[:160])
    return dict(wall_s=round(time.time() - t0, 2), ttft_s=round((first - t0), 2) if first else None, tokens=n,
                decode_tok_s=round((n - 1) / (last - first), 3) if n > 1 and last > first else None,
                prompt_tokens=(usage or {}).get("prompt_tokens"), text="".join(text), error=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--sizes", default="1024,4096,12288,32768")
    ap.add_argument("--budget-s", type=int, default=720, help="wall-clock budget for the whole scan")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--cap", type=int, default=900, help="cap per request")
    ap.add_argument("--streams", type=int, default=1, help="parallel identical requests per size (1 = one stream)")
    a = ap.parse_args()
    sizes = sorted({int(s) for s in a.sizes.split(",") if s.strip()})
    pool = text_pool()
    out_dir = os.path.join(LAB, "experiments", a.exp)
    os.makedirs(out_dir, exist_ok=True)
    results, t_start = [], time.time()
    rates = []   # measured prefill tok/s
    out_path = os.path.join(out_dir, "context_scan.json")

    def save(note=""):
        # after EVERY size: if a box dies mid-scan, what was measured before it is on disk already
        json.dump(dict(exp=a.exp, sizes=sizes, budget_s=a.budget_s, new_tokens=a.new_tokens, results=results,
                       wall_s=round(time.time() - t_start), note=note), open(out_path + ".tmp", "w"), indent=1)
        os.replace(out_path + ".tmp", out_path)

    save("started")
    for size in sizes:
        left = a.budget_s - (time.time() - t_start)
        rate = min(rates) if rates else 40.0
        predicted = size / rate + a.new_tokens / 2.0 + 5
        if predicted > left:
            log("size %d: predicted %.0f s at %.0f tok/s of prefill, %.0f s left: skipped" % (size, predicted, rate, left))
            results.append(dict(size=size, skipped=True, predicted_s=round(predicted), left_s=round(left)))
            save()
            continue
        prompt, code = build_prompt(pool, size, seed=size)
        log("size %d: %d chars, sending (%d parallel), predicted %.0f s" % (size, len(prompt), a.streams, predicted))
        results.append(dict(size=size, in_flight=True, started_at=time.strftime("%H:%M:%S")))
        save("size %d in flight" % size)
        results.pop()
        if a.streams > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(a.streams) as ex:
                rs = list(ex.map(lambda _: chat(prompt, a.new_tokens, min(a.cap, max(60, left))), range(a.streams)))
        else:
            rs = [chat(prompt, a.new_tokens, min(a.cap, max(60, left)))]
        r = rs[0]
        found = code in (r.get("text") or "")
        pt = r.get("prompt_tokens")
        prefill_rate = round(pt / r["ttft_s"], 1) if pt and r.get("ttft_s") else None
        if prefill_rate:
            rates.append(prefill_rate)
        if r.get("error") and not r.get("ttft_s"):
            log("size %d: no first token after %s s (%s)" % (size, r.get("wall_s"), r["error"]))
        row = dict(size=size, prompt_tokens=pt, ttft_s=r.get("ttft_s"), prefill_tok_s=prefill_rate,
                   decode_tok_s=r.get("decode_tok_s"), tokens=r.get("tokens"), wall_s=r.get("wall_s"),
                   needle_found=found, error=r.get("error"), streams=a.streams,
                   answer=(r.get("text") or "")[-120:],
                   others=[dict(ttft_s=x.get("ttft_s"), decode_tok_s=x.get("decode_tok_s"), error=x.get("error")) for x in rs[1:]])
        results.append(row)
        save()
        log("size %d: prompt %s tok, first token %s s (%s tok/s prefill), decode %s tok/s, needle %s%s" % (
            size, pt, r.get("ttft_s"), prefill_rate, r.get("decode_tok_s"), "FOUND" if found else "missing",
            (", error: " + r["error"]) if r.get("error") else ""))
        if r.get("error"):
            log("stopping the scan after an error")
            break
    save("finished")
    print("\n| context (tokens) | first token (s) | prefill tok/s | decode tok/s | needle | note |")
    print("|---|---|---|---|---|---|")
    for r in results:
        if r.get("skipped"):
            print("| %d | - | - | - | - | skipped: %d s predicted, %d s left |" % (r["size"], r["predicted_s"], r["left_s"]))
        else:
            print("| %s | %s | %s | %s | %s | %s |" % (r["prompt_tokens"] or r["size"], r["ttft_s"], r["prefill_tok_s"], r["decode_tok_s"],
                                                     "yes" if r["needle_found"] else "NO", r.get("error") or ""))
    print("total %d s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
