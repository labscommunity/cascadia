"""Scored task accuracy — measures QUALITY, not just equality.

The parity suites answer "does packed produce the same text as baseline". They
cannot answer "is packed as CORRECT as baseline", because a divergence is only
reported, never judged. This scores both configs against ground truth so an
accuracy claim rests on a measurement.

    python scored.py <base_url> <label>
    python scored.py score <labelA> <labelB>
"""

import json
import sys
import urllib.request

# Unambiguous factual questions; `any` alternative counts as correct.
QA = [
    ("What is the capital of France? Answer with just the city name.", ["paris"]),
    ("What is the capital of Japan? Answer with just the city name.", ["tokyo"]),
    ("What is the capital of Italy? Answer with just the city name.", ["rome", "roma"]),
    ("What is the capital of Spain? Answer with just the city name.", ["madrid"]),
    ("What is the capital of Canada? Answer with just the city name.", ["ottawa"]),
    ("What is the largest ocean on Earth? Answer with just the name.", ["pacific"]),
    ("What planet is closest to the Sun? Answer with just the name.", ["mercury"]),
    ("What is the chemical symbol for water? Answer with just the symbol.", ["h2o"]),
    ("How many days are in a leap year? Answer with just the number.", ["366"]),
    ("What is 15 plus 27? Answer with just the number.", ["42"]),
    ("What is 9 times 8? Answer with just the number.", ["72"]),
    ("How many continents are there? Answer with just the number.", ["7", "seven"]),
    ("What gas do humans exhale? Answer with just the gas name.", ["carbon dioxide", "co2"]),
    ("What is the freezing point of water in Celsius? Just the number.", ["0", "zero"]),
    ("Who wrote Romeo and Juliet? Answer with just the surname.", ["shakespeare"]),
    ("What is the largest planet in our solar system? Just the name.", ["jupiter"]),
    ("How many sides does a hexagon have? Just the number.", ["6", "six"]),
    ("What is the currency of the United Kingdom? Just the name.", ["pound", "sterling", "gbp"]),
    ("What colour do you get mixing blue and yellow? Just the colour.", ["green"]),
    ("What is the square root of 81? Just the number.", ["9", "nine"]),
]


def ask(base, q):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return out["choices"][0]["message"]["content"]


def capture(base, label):
    rec = {}
    for q, _ in QA:
        rec[q] = ask(base, q)
    json.dump(rec, open(f"{label}.json", "w"), indent=1)
    print(f"[{label}] {len(QA)} answers -> {label}.json")


def grade(text, accepts):
    t = text.lower()
    return any(a in t for a in accepts)


def score(la, lb):
    A = json.load(open(f"{la}.json"))
    B = json.load(open(f"{lb}.json"))
    sa = sb = agree = 0
    print(f"{'question':<52} {la:>10} {lb:>10}")
    print("-" * 76)
    for q, acc in QA:
        ga, gb = grade(A[q], acc), grade(B[q], acc)
        sa += ga
        sb += gb
        agree += (A[q] == B[q])
        flag = "" if ga == gb else "   <-- DIFFERS IN CORRECTNESS"
        print(f"{q[:50]:<52} {'ok' if ga else 'WRONG':>10} {'ok' if gb else 'WRONG':>10}{flag}")
    n = len(QA)
    print("-" * 76)
    print(f"{la}: {sa}/{n} correct ({sa/n:.0%})")
    print(f"{lb}: {sb}/{n} correct ({sb/n:.0%})")
    print(f"identical text on {agree}/{n}")
    print(f"\nACCURACY DELTA: {sb - sa:+d} answers")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        capture(sys.argv[1], sys.argv[2])
