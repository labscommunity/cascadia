#!/usr/bin/env python3
"""How often would drafter X have named the model's next token? Offline, over text the fleet really wrote.

    drafter_study.py CORPUS.jsonl --out RESULT.json [--gguf NAME=PATH:TEMPLATE ...] [--positions 4000] [--ngram]

Needs tiktoken (o200k_base = the model's tokenizer, verified in autolab/bench/tokenizer_probe.py) and, for the
neural drafters, llama.cpp's `llama-server` on PATH. Run it with the scratch venv's python.

The measure is teacher-forced and per position of the MODEL's tokenization: the drafter sees the conversation and
the model's own text so far, continues it greedily, and scores a hit when its continuation starts with the text of
the model's next token. That is what a drafter with another vocabulary can deliver at best (its text re-tokenised
into the model's vocabulary), and it is the `a` of `1 / (a*T + (1-a)*L)` because a pipelined drafter only ever
extends a prefix that is right so far.

`--ngram` replays the fleet's present drafter (request history + a table learned from the other responses).
"""
import argparse, collections, json, os, subprocess, sys, time, urllib.request

TEMPLATES = {
    "qwen3": "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "plain": "Question: {p}\n\nAnswer: ",
}


def post(port, path, body, timeout=120):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as r:
        return json.loads(r.read())


def serve(path, port):
    p = subprocess.Popen(["llama-server", "-m", path, "--port", str(port), "-c", "4096", "-ngl", "99", "--parallel", "1",
                          "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return p
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("llama-server did not come up for " + path)


def positions(corpus, enc, limit):
    """(doc index, prefix text, next-token text, index in the response) at the model's token boundaries."""
    out = []
    for di, d in enumerate(corpus):
        ids = enc.encode(d["text"], disallowed_special=())
        pieces = [enc.decode_single_token_bytes(t) for t in ids]
        acc = b""
        for k, piece in enumerate(pieces):
            try:
                prefix = acc.decode("utf-8"); nxt = piece.decode("utf-8")
            except UnicodeDecodeError:
                acc += piece
                continue  # a token that splits a UTF-8 sequence: not scoreable as text
            if k > 0:
                out.append((di, prefix, nxt, k))
            acc += piece
    if limit and len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def study_gguf(name, path, template, corpus, pos, port):
    srv = serve(path, port)
    hits = 0; n = 0; t0 = time.time(); by_k = collections.defaultdict(lambda: [0, 0]); vec = []
    try:
        for di, prefix, nxt, k in pos:
            prompt = TEMPLATES[template].format(p=corpus[di]["prompt"]) + prefix
            r = post(port, "/completion", dict(prompt=prompt, n_predict=6, temperature=0, cache_prompt=True, stream=False))
            d = r.get("content", "")
            hit = d.startswith(nxt)
            if not hit and nxt.startswith(d) and len(d) < len(nxt):  # too short to tell: look further
                r = post(port, "/completion", dict(prompt=prompt, n_predict=16, temperature=0, cache_prompt=True, stream=False))
                hit = r.get("content", "").startswith(nxt)
            hits += hit; n += 1; vec.append(int(hit))
            b = by_k[min(k // 32, 4)]; b[0] += hit; b[1] += 1
            if n % 500 == 0:
                print("  %s: %d/%d positions, a = %.3f, %.0f s" % (name, n, len(pos), hits / n, time.time() - t0), flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=20)
        except subprocess.TimeoutExpired:
            srv.kill()
    return dict(drafter=name, positions=n, hits=hits, a=round(hits / max(1, n), 4), seconds=round(time.time() - t0),
                by_32_tokens={str(k * 32): round(v[0] / max(1, v[1]), 4) for k, v in sorted(by_k.items())},
                runs=run_lengths(pos, vec), hit_vector="".join(map(str, vec)))


def run_lengths(pos, vec):
    """Mean number of consecutive hits from a random start inside a response: what a chain of guesses is worth."""
    runs = []; cur = 0; last = None
    for (di, _p, _n, k), h in zip(pos, vec):
        if last is not None and (di != last[0] or k != last[1] + 1):
            cur = 0
        cur = cur + 1 if h else 0
        runs.append(cur); last = (di, k)
    return round(sum(runs) / max(1, len(runs)), 3)


def study_ngram(corpus, enc, max_ctx=3, min_conf=0.125, min_seen=2):
    """The fleet's drafter: the request's own history first (longest earlier match of the last 4..2 tokens), then a table
    of next-token counts keyed by the last 3/2/1 tokens, learned from the other responses (leave-one-out)."""
    docs = [(enc.encode(d["prompt"], disallowed_special=()), enc.encode(d["text"], disallowed_special=())) for d in corpus]
    res = {}
    for label, use_shared in (("ngram_history_only", False), ("ngram_history_plus_table", True)):
        hits = prop = n = 0
        for di, (p, t) in enumerate(docs):
            table = collections.defaultdict(collections.Counter)
            if use_shared:
                for dj, (p2, t2) in enumerate(docs):
                    if dj == di:
                        continue
                    seq = p2 + t2
                    for i in range(1, len(seq)):
                        for c in range(1, max_ctx + 1):
                            if i - c >= 0:
                                table[tuple(seq[i - c:i])][seq[i]] += 1
            hist = list(p)
            for k, tok in enumerate(t):
                guess = None
                for c in (4, 3, 2):  # own history, longest match, most recent occurrence
                    if len(hist) > c:
                        key = hist[-c:]
                        for s in range(len(hist) - c - 1, -1, -1):
                            if hist[s:s + c] == key:
                                guess = hist[s + c]; break
                    if guess is not None:
                        break
                if guess is None and use_shared:
                    for c in range(max_ctx, 0, -1):
                        cnt = table.get(tuple(hist[-c:]))
                        if cnt:
                            g, m = cnt.most_common(1)[0]; tot = sum(cnt.values())
                            if tot >= min_seen and m / tot >= min_conf:
                                guess = g; break
                if k > 0:
                    n += 1; prop += guess is not None; hits += guess == tok
                hist.append(tok)
            if use_shared and di >= 47:
                break  # leave-one-out tables are slow to build: 48 documents are enough
        res[label] = dict(drafter=label, positions=n, proposed=prop, hits=hits, a=round(hits / max(1, n), 4),
                          precision=round(hits / max(1, prop), 4))
    return list(res.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("--out", required=True); ap.add_argument("--gguf", nargs="*", default=[])
    ap.add_argument("--positions", type=int, default=4000); ap.add_argument("--ngram", action="store_true")
    ap.add_argument("--port", type=int, default=18231)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    corpus = [json.loads(l) for l in open(a.corpus) if l.strip()]
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    if a.ngram:
        for r in study_ngram(corpus, enc):
            print(json.dumps(r), flush=True); results.append(r)
    pos = positions(corpus, enc, a.positions)
    for spec in a.gguf:
        name, rest = spec.split("=", 1); path, template = rest.rsplit(":", 1)
        r = study_gguf(name, path, template, corpus, pos, a.port)
        print(json.dumps(r), flush=True); results.append(r)
        json.dump(results, open(a.out, "w"), indent=1)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
