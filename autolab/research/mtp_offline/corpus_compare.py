#!/usr/bin/env python3
"""Compare a CPU-reference-path greedy text (033: dumped on the Mac Pro) with the fleet's corpus text: first-divergence token index.

    corpus_compare.py gen.jsonl corpus.jsonl tokenizer.json
"""
import json, sys
from tokenizers import Tokenizer
gen = [json.loads(l) for l in open(sys.argv[1])]
corpus = {r["i"]: r for r in (json.loads(l) for l in open(sys.argv[2]))}
tok = Tokenizer.from_file(sys.argv[3])
FAM = ["explain", "code", "arithmetic", "story", "lists", "tables", "rewriting", "facts", "poem", "how-to", "translation", "true/false"]
same = 0; firsts = []
for g in gen:
    ids = g["gen_ids"]; c = corpus[g["i"]]["text"]
    # API text: <|content_thinking|> is shown as "<think>", the closing <|end_message|> of the thinking block as "</think>"
    pieces = []
    for k in range(len(ids)):
        pieces.append(tok.decode(ids[:k + 1], skip_special_tokens=False))
    norm = lambda s: s.replace("<|content_thinking|>", "<think>")
    first = None
    for k, p in enumerate(pieces):
        p = norm(p)
        if p.endswith("�"):
            continue                       # a partial multi-byte character: decide at the next token
        if not c.startswith(p):
            # the API may render later special tokens differently; only count a divergence in ordinary text
            first = k; break
    if first is None:
        same += 1
    firsts.append(first)
    print("i=%3d %-11s gen=%3d corpus_tokens=%s first_divergence=%s%s" % (g["i"], FAM[g["family"]], len(ids), corpus[g["i"]]["tokens"],
          first, "" if first is None else "  | mine: %r | fleet: %r" % (norm(pieces[first])[-40:], c[max(0, len(norm(pieces[first - 1])) - 25 if first else 0):][:40])))
d = [f for f in firsts if f is not None]
print("\n%d sequences; identical over the whole generated text: %d; diverged: %d (first-divergence token index: min %s, median %s, max %s)" % (
    len(gen), same, len(d), min(d) if d else None, sorted(d)[len(d) // 2] if d else None, max(d) if d else None))
tot = sum(len(g["gen_ids"]) for g in gen); agree = sum((f if f is not None else len(g["gen_ids"])) for f, g in zip(firsts, gen))
print("tokens before the first divergence: %d of %d (%.1f %%)" % (agree, tot, 100.0 * agree / tot))
