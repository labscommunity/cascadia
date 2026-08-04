"""Characterise the packed-vs-baseline divergences.

Exact match between two DIFFERENT compiled graphs under greedy decode is a very
strict bar: argmax is discontinuous, so an fp16 rounding difference of ~1e-5 on
a near-tied pair of logits flips one token and the sequences separate from
there. This distinguishes that benign case from an actual masking/KV bug, which
would diverge EARLY and produce incoherent text.
"""

import json
import sys

A = json.load(open(sys.argv[1] + ".json"))
B = json.load(open(sys.argv[2] + ".json"))

print(f"{'prompt':<46} {'identical prefix':>17} {'lenA':>5} {'lenB':>5}")
print("-" * 78)
tot_chars = same_chars = 0
late = []
for p in A["solo"]:
    ta, tb = A["solo"][p]["text"], B["solo"][p]["text"]
    n = min(len(ta), len(tb))
    i = next((k for k in range(n) if ta[k] != tb[k]), n)
    tot_chars += max(len(ta), len(tb))
    same_chars += i
    frac = i / max(len(ta), len(tb))
    mark = "exact" if ta == tb else f"{frac:6.1%}"
    print(f"{p[:44]:<46} {mark:>17} {len(ta):>5} {len(tb):>5}")
    if ta != tb:
        late.append((p, i, len(ta), frac, ta, tb))

print(f"\ncharacter-level agreement across the suite: {same_chars}/{tot_chars} "
      f"= {same_chars/tot_chars:.2%}")

print("\n--- divergence detail ---")
for p, i, la, frac, ta, tb in late:
    print(f"\n{p}")
    print(f"  first divergence at char {i} ({frac:.1%} of the way through)")
    print(f"  common prefix ends: ...{ta[max(0,i-60):i]!r}")
    print(f"  A continues:        {ta[i:i+70]!r}")
    print(f"  B continues:        {tb[i:i+70]!r}")
    # a masking/KV bug degenerates: repetition, truncation, or word salad
    def health(t):
        w = t.split()
        rep = max((w.count(x) for x in set(w)), default=0)
        return f"words={len(w)} max_word_repeat={rep} ends_mid_word={not t.rstrip().endswith(('.', '!', '?', '`', ':'))}"
    print(f"  A health: {health(ta)}")
    print(f"  B health: {health(tb)}")
