"""Scored LONG-FORM accuracy — the regime where drift actually shows.

The short-answer benchmark can't detect degradation: 32-token answers don't
accumulate enough fp16 drift to flip a near-tie. These prompts generate 150-200
tokens each and are scored on ORDERED ATOM RECALL — every expected item must
appear, in the right order. That penalises exactly what degradation looks like
(omission, reordering, repetition loops, early truncation), rather than
rewarding a correct first sentence.

Budget note: at --packed-slots 4 the per-slot region is 255 tokens, so prompt +
generation is kept under that. Otherwise this would measure context eviction,
not drift.

    python longform.py <base_url> <label>
    python longform.py score <labelA> <labelB>
"""

import json
import sys
import urllib.request

MAX_TOKENS = 200

ONES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty"]
PRIMES = ["2", "3", "5", "7", "11", "13", "17", "19", "23", "29", "31", "37"]
PLANETS = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
ALPHA = list("abcdefghijklmnopqrstuvwxyz")
CAPITALS = ["paris", "tokyo", "rome", "madrid", "ottawa", "berlin", "lisbon", "vienna"]
SQUARES = ["1", "4", "9", "16", "25", "36", "49", "64", "81", "100"]

SUITE = [
    ("Count from one to twenty in words, separated by commas. Output only the list.", ONES),
    ("List the first twelve prime numbers, separated by commas. Only the list.", PRIMES),
    ("List the eight planets in order from the Sun, separated by commas. Only the list.", PLANETS),
    ("List the seven days of the week in order, then the twelve months in order.", DAYS + MONTHS),
    ("Write the alphabet from a to z, letters separated by spaces. Only the letters.", ALPHA),
    ("Give the capital city of France, Japan, Italy, Spain, Canada, Germany, "
     "Portugal and Austria, in that order, one per line.", CAPITALS),
    ("List the squares of the numbers 1 through 10 in order, separated by commas.", SQUARES),
]


def ask(base, q):
    body = {"model": "default", "messages": [{"role": "user", "content": q}],
            "max_tokens": MAX_TOKENS, "temperature": 0.0}
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return {"text": out["choices"][0]["message"]["content"],
            "usage": out.get("usage", {})}


def capture(base, label):
    rec = {}
    for q, _ in SUITE:
        rec[q] = ask(base, q)
        n = rec[q]["usage"].get("completion_tokens", 0)
        print(f"  [{label}] {n:>4} tok  {q[:46]}", flush=True)
    json.dump(rec, open(f"{label}.json", "w"), indent=1)


def ordered_recall(text, atoms):
    """Fraction of expected atoms found IN ORDER — a single forward scan, so an
    omission or a reordering both cost."""
    t = text.lower()
    idx = found = 0
    for a in atoms:
        j = t.find(a, idx)
        if j >= 0:
            found += 1
            idx = j + len(a)
    return found / len(atoms)


def repetition(text):
    w = text.lower().split()
    return max((w.count(x) for x in set(w)), default=0) / max(len(w), 1)


def score(la, lb):
    A = json.load(open(f"{la}.json"))
    B = json.load(open(f"{lb}.json"))
    print(f"{'task':<46} {'atoms':>6} {la:>9} {lb:>9} {'tokA':>5} {'tokB':>5}")
    print("-" * 86)
    ta = tb = 0.0
    ident = 0
    for q, atoms in SUITE:
        ra, rb = ordered_recall(A[q]["text"], atoms), ordered_recall(B[q]["text"], atoms)
        ta += ra
        tb += rb
        ident += A[q]["text"] == B[q]["text"]
        flag = "" if abs(ra - rb) < 1e-9 else "  <-- DIFFERS"
        print(f"{q[:44]:<46} {len(atoms):>6} {ra:>8.0%} {rb:>8.0%} "
              f"{A[q]['usage'].get('completion_tokens',0):>5} "
              f"{B[q]['usage'].get('completion_tokens',0):>5}{flag}")
    n = len(SUITE)
    print("-" * 86)
    print(f"mean ordered recall  {la}: {ta/n:.1%}   {lb}: {tb/n:.1%}   "
          f"delta {100*(tb-ta)/n:+.1f} pts")
    print(f"identical text on {ident}/{n}")
    ra_rep = sum(repetition(A[q]["text"]) for q, _ in SUITE) / n
    rb_rep = sum(repetition(B[q]["text"]) for q, _ in SUITE) / n
    print(f"mean max-word-repetition  {la}: {ra_rep:.3f}   {lb}: {rb_rep:.3f} "
          f"(a degeneration loop would spike this)")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        capture(sys.argv[1], sys.argv[2])
