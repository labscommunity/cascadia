"""Agreement of a running decisions server (SystemOne route, /v1/systemone) with a full-precision reference.

  python tools/decision/parity_check.py --url http://127.0.0.1:8080                       # only drive the server (dump)
  python tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref ref.json        # gate: raw >= 96% and no flip with reference margin >= 0.05

REF.json maps "<record id>.<question id>" to the reference's probabilities in API option order
(noul [false, true]; choice in request order; score by level index), as answer_probs flattens them.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict

from records import records


def answer_probs(ans):
    """A /v1/systemone answer -> its probabilities in API option order."""
    if ans["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    return list(ans["probabilities"].values())


def margin(p):
    s = sorted(p, reverse=True)
    return s[0] - (s[1] if len(s) > 1 else 0.0)


def gate(flip_margins, agree, total, min_agree, confident):
    """Pass iff raw agreement >= min_agree AND no flip where the reference itself was confident."""
    sure = [m for m in flip_margins if m >= confident]
    if sure:
        return False, f"{len(sure)} confident flip(s) (reference margin >= {confident}): {[round(m, 3) for m in sure]}"
    rate = agree / max(1, total)
    if rate < min_agree:
        return False, f"agreement {rate:.3f} < {min_agree}"
    return True, f"agreement {agree}/{total} = {rate:.3f}, every flip had reference margin < {confident}"


def systemone(url, rec):
    """POST one record to /v1/systemone; a non-2xx reply exits with the server's message."""
    body = json.dumps({"model": "m", "state": rec["state"], "questions": rec["questions"]}).encode()
    req = urllib.request.Request(url + "/v1/systemone", data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ERROR {rec['id']}: HTTP {e.code} {e.read().decode(errors='replace')}")


def tally(ref, results):
    """Compare [(record, /v1/systemone response)] with REF over EVERY reference entry.
    -> (per_type {kind: [agree, n]}, agree, total, [(key, kind, ref margin, got, want)] flips).
    A missing or extra answer, a question without a reference entry, an unused reference entry or
    an option-count mismatch fails: partial coverage must never be scored."""
    per_type = defaultdict(lambda: [0, 0])
    agree = total = 0
    flips, used = [], set()
    for rec, resp in results:
        if list(resp["answers"]) != list(rec["questions"]):
            raise SystemExit(f"ERROR {rec['id']}: server answered {list(resp['answers'])}, asked {list(rec['questions'])}")
        for qid, ans in resp["answers"].items():
            key = f"{rec['id']}.{qid}"
            if key not in ref:
                raise SystemExit(f"ERROR {key}: no reference entry")
            used.add(key)
            got, want = answer_probs(ans), ref[key]
            if len(got) != len(want):
                raise SystemExit(f"ERROR {key}: server has {len(got)} options, reference {len(want)}")
            same = max(range(len(got)), key=got.__getitem__) == max(range(len(want)), key=want.__getitem__)
            kind = rec["questions"][qid]["type"]
            per_type[kind][0] += same
            per_type[kind][1] += 1
            agree += same
            total += 1
            if not same:
                flips.append((key, kind, margin(want), got, want))
    if set(ref) - used:
        raise SystemExit(f"ERROR reference entries never answered: {sorted(set(ref) - used)[:10]}")
    if total != len(ref):
        raise SystemExit(f"ERROR compared {total} answers but the reference has {len(ref)} entries")
    return per_type, agree, total, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref", default=None, help="omit to only drive the server (it writes its --decision-dump)")
    ap.add_argument("--min-agree", type=float, default=0.96, help="raw top-1 agreement floor (0.96 = at most 3 flips of 82)")
    ap.add_argument("--confident", type=float, default=0.05, help="a flip where the reference margin is >= this fails the gate")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    results = [(rec, systemone(url, rec)) for rec in records()]
    if not a.ref:
        print("requests sent (no reference given)")
        return
    with open(a.ref) as f:
        ref = json.load(f)
    per_type, agree, total, flips = tally(ref, results)
    for key, kind, m, got, want in flips:
        print(f"FLIP {key} {kind} ref_margin={m:.3f} got={[round(x, 3) for x in got]} ref={[round(x, 3) for x in want]}")
    for kind, (g, n) in sorted(per_type.items()):
        print(f"  {kind}: {g}/{n}")
    ok, why = gate([f[2] for f in flips], agree, total, a.min_agree, a.confident)
    print(("PASS " if ok else "FAIL ") + why)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
