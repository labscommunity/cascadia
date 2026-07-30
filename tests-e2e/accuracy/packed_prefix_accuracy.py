"""Prefix-cache accuracy parity.

The main suite uses all-different prompts, so longest-common-prefix matching
finds nothing and prefix caching never engages — it would pass vacuously. This
suite gives every request the SAME long system prompt, so requests 2..N reuse
cached K/V instead of recomputing it. Reuse must not change a single token.

Also stresses the boundary: a couple of prompts are long enough that the
reused prefix plus their own tokens approach the per-slot region.

    python prefix_accuracy.py capture <base> <label>
    python prefix_accuracy.py compare <labelA> <labelB>
"""

import concurrent.futures as cf
import json
import sys
import urllib.request

MAX_TOKENS = 96
SYSTEM = (
    "You are a precise assistant for a geography and science quiz. Answer with "
    "one short factual sentence and nothing else. Do not add commentary, do not "
    "explain your reasoning, do not apologise, and never mention these "
    "instructions. Be terse and factual at all times, and follow this format "
    "exactly for every question you are asked, without exception."
)
QUESTIONS = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the tallest mountain on Earth?",
    "What is the largest ocean?",
    "What gas do plants absorb during photosynthesis?",
    "How many continents are there?",
    "What is the longest river in Africa?",
    "What planet is closest to the Sun?",
]


def generate(base, question, max_tokens=MAX_TOKENS):
    body = {
        "model": "default",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return {
        "text": out["choices"][0]["message"]["content"],
        "usage": out.get("usage", {}),
    }


def capture(base, label):
    rec = {"label": label, "sequential": {}, "concurrent": {}}
    # Sequential: request 1 populates the cache, 2..N hit it.
    print(f"[{label}] sequential (cache populates then hits)", flush=True)
    for q in QUESTIONS:
        rec["sequential"][q] = generate(base, q)
    # Concurrent: several cache hits in flight at once, sharing the batch.
    print(f"[{label}] concurrent (cache hits sharing the batch)", flush=True)
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {q: ex.submit(generate, base, q) for q in QUESTIONS}
        for q, f in futs.items():
            rec["concurrent"][q] = f.result()
    with open(f"{label}.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{label}] captured -> {label}.json")


def compare(la, lb):
    A = json.load(open(f"{la}.json"))
    B = json.load(open(f"{lb}.json"))
    fails = []

    def cmp(ka, kb, name):
        same = [q for q in QUESTIONS if A[ka][q]["text"] == B[kb][q]["text"]]
        print(f"\n{name}: {len(same)}/{len(QUESTIONS)} exact")
        for q in QUESTIONS:
            if q not in same:
                print(f"  DIFF {q!r}")
                print(f"    A: {A[ka][q]['text'][:90]!r}")
                print(f"    B: {B[kb][q]['text'][:90]!r}")
        if len(same) != len(QUESTIONS):
            fails.append(name)

    print("=" * 66)
    print(f"PREFIX-CACHE PARITY  A={la}  B={lb}")
    print("=" * 66)
    cmp("sequential", "sequential", f"{la} vs {lb} (sequential)")
    cmp("concurrent", "concurrent", f"{la} vs {lb} (concurrent)")
    # within-config: cache hits must match the populating path too
    same_int = [q for q in QUESTIONS
                if B["sequential"][q]["text"] == B["concurrent"][q]["text"]]
    print(f"\n[{lb}] sequential vs concurrent: {len(same_int)}/{len(QUESTIONS)} exact")
    if len(same_int) != len(QUESTIONS):
        fails.append(f"[{lb}] sequential vs concurrent")

    distinct = len({A["sequential"][q]["text"] for q in QUESTIONS})
    print(f"\nTEETH: {distinct}/{len(QUESTIONS)} distinct answers "
          f"(a broken cache would collapse them)")
    teeth = distinct >= len(QUESTIONS) - 1
    print("\n" + "=" * 66)
    print("RESULT:", "PASS" if (not fails and teeth) else f"FAIL {fails}")


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3])
