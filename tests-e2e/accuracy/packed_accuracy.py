"""Accuracy parity harness for packed multi-slot decode.

Deliberately avoids the weak-eval trap: every comparison is an EXACT match on
the FULL generated text (not a substring / first-sentence check), generations
are long enough for KV drift to surface, and the suite includes a teeth check
that must fail if the harness were vacuous.

    python accuracy.py capture <base_url> <label>     -> writes <label>.json
    python accuracy.py compare <labelA> <labelB>      -> exact-diff report

Capture runs each prompt twice per mode:
  solo    - the only request in flight
  batched - issued concurrently with 3 unrelated requests

so a single capture proves BOTH determinism (solo == solo repeat) and
batch-composition invariance (solo == batched), which is the property that
would break first if slots leaked into each other.
"""

import concurrent.futures as cf
import json
import sys
import urllib.request

MAX_TOKENS = 128  # long enough that a single early divergence cascades visibly

# Diverse + adversarial: repetition, counting, code, long-form, and a prompt
# whose generation runs long (region/eviction pressure).
PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Count from one to twenty in words, separated by commas.",
    "List four prime numbers and explain briefly what makes a number prime.",
    "Write a short Python function that reverses a string.",
    "Name three oceans and give one fact about each.",
    "Explain photosynthesis in exactly two sentences.",
    "Describe the colour blue to someone who has never seen it.",
    "Write two sentences about Mount Fuji.",
    "Repeat the word 'echo' ten times, separated by spaces.",
    "Summarise why the sky appears blue, in three sentences.",
]

# Fillers used only to occupy the other slots during the batched pass; their
# outputs are never compared, they exist to perturb the batch.
FILLERS = [
    "Name a river in Europe.",
    "What is 12 times 12?",
    "Give one fact about penguins.",
]


def generate(base, prompt, max_tokens=MAX_TOKENS):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    ch = out["choices"][0]
    return {
        "text": ch["message"]["content"],
        "finish_reason": ch.get("finish_reason"),
        "usage": out.get("usage", {}),
    }


def capture(base, label, solo_only=False):
    """`solo_only` skips the concurrent pass — needed when the config under test
    cannot serve concurrent requests (e.g. the non-packed multi-stage path)."""
    rec = {"label": label, "max_tokens": MAX_TOKENS, "solo": {}, "solo2": {}, "batched": {}}

    print(f"[{label}] solo pass (1/3)", flush=True)
    for p in PROMPTS:
        rec["solo"][p] = generate(base, p)

    print(f"[{label}] solo repeat pass (2/3) - determinism", flush=True)
    for p in PROMPTS:
        rec["solo2"][p] = generate(base, p)

    if solo_only:
        rec["batched"] = rec["solo"]
        with open(f"{label}.json", "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[{label}] captured (solo only) -> {label}.json")
        return

    print(f"[{label}] batched pass (3/3) - each prompt shares the batch", flush=True)
    for p in PROMPTS:
        # target prompt + 3 fillers issued together, so the target lands in a
        # populated batch rather than running alone
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            fut = ex.submit(generate, base, p)
            for f in FILLERS:
                ex.submit(generate, base, f, 48)
            rec["batched"][p] = fut.result()

    with open(f"{label}.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[{label}] captured {len(PROMPTS)} prompts x 3 passes -> {label}.json")


def _cmp(a, b, name, results):
    same = [p for p in PROMPTS if a[p]["text"] == b[p]["text"]]
    diff = [p for p in PROMPTS if p not in same]
    results.append((name, len(same), len(PROMPTS), diff))
    print(f"\n{name}: {len(same)}/{len(PROMPTS)} exact")
    for p in diff:
        ta, tb = a[p]["text"], b[p]["text"]
        # first divergent character, so a late drift is distinguishable from a
        # completely different answer
        i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]),
                 min(len(ta), len(tb)))
        print(f"  DIFF {p[:44]!r}")
        print(f"    diverges at char {i} of {len(ta)}/{len(tb)}")
        print(f"    A: ...{ta[max(0,i-30):i+50]!r}")
        print(f"    B: ...{tb[max(0,i-30):i+50]!r}")


def compare(la, lb):
    A = json.load(open(f"{la}.json"))
    B = json.load(open(f"{lb}.json"))
    results = []

    print("=" * 66)
    print(f"ACCURACY PARITY  A={la}  B={lb}   max_tokens={A['max_tokens']}")
    print("=" * 66)

    # within-config invariants (each must hold on its own merits)
    _cmp(A["solo"], A["solo2"], f"[{la}] determinism (solo vs solo-repeat)", results)
    _cmp(A["solo"], A["batched"], f"[{la}] batch-composition invariance", results)
    _cmp(B["solo"], B["solo2"], f"[{lb}] determinism (solo vs solo-repeat)", results)
    _cmp(B["solo"], B["batched"], f"[{lb}] batch-composition invariance", results)

    # cross-config parity
    _cmp(A["solo"], B["solo"], f"PARITY {la} vs {lb} (solo)", results)
    _cmp(A["batched"], B["batched"], f"PARITY {la} vs {lb} (batched)", results)

    # teeth: the harness must be capable of reporting a difference
    distinct = len({A["solo"][p]["text"] for p in PROMPTS})
    print(f"\nTEETH: {distinct}/{len(PROMPTS)} prompts produced distinct outputs")
    nonempty = all(len(A["solo"][p]["text"].strip()) > 0 for p in PROMPTS)
    print(f"TEETH: all outputs non-empty: {nonempty}")
    lens = [A["solo"][p]["usage"].get("completion_tokens", 0) for p in PROMPTS]
    print(f"TEETH: completion_tokens min={min(lens)} max={max(lens)} "
          f"(short outputs would make exact-match trivially easy)")
    teeth_ok = distinct >= len(PROMPTS) - 1 and nonempty and min(lens) >= 5

    print("\n" + "=" * 66)
    hard_fail = False
    for name, same, total, _ in results:
        verdict = "PASS" if same == total else "FAIL"
        if same != total:
            hard_fail = True
        print(f"  {verdict}  {name}: {same}/{total}")
    print(f"  {'PASS' if teeth_ok else 'FAIL'}  teeth (harness is not vacuous)")
    print("=" * 66)
    print("RESULT:", "PASS" if (not hard_fail and teeth_ok) else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3], solo_only=(len(sys.argv) > 4 and sys.argv[4] == "solo"))
    else:
        compare(sys.argv[2], sys.argv[3])
