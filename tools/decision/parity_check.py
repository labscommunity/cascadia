"""Agreement of a running decisions server (SystemOne route, /v1/systemone) with a full-precision reference.

  python tools/decision/parity_check.py --url http://127.0.0.1:8080 --dump d.jsonl        # drive the server, slice its dump
  python tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref ref.json        # gate: raw >= 96% and no flip with reference margin >= 0.05
  python tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref ref.json --answers-out single.json   # also save this run's answers
  python tools/decision/parity_check.py --url http://127.0.0.1:8080 --ref single.json --min-agree 1 --confident 0  # top-1 identical to that run

Without --ref the server must run with `--decision-dump DUMP` and have no other client: this run's
passes (the bytes the dump grew by while the 50 records were posted) are written to
`<DUMP minus .jsonl>.parity.jsonl` (mode 0600, never overwritten without --force) and checked to
align with the 50 records. That slice is what reference_stream.py --dump takes; the whole dump also
holds golden, duplicate-request and earlier runs' passes.

REF.json maps "<record id>.<question id>" to the reference's probabilities in API option order
(noul [false, true]; choice in request order; score by level index), as answer_probs flattens them.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from codec_golden import align, write_private
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


def parity_slice(dump):
    """Where a no-ref run writes its own passes: d.jsonl -> d.parity.jsonl."""
    return Path(dump).with_suffix(".parity.jsonl")


def answers_ref(results):
    """[(record, /v1/systemone response)] -> {"<record id>.<question id>": probabilities}, the REF
    format, so one server's answers can gate another's (a pipeline tree against its single box)."""
    out = {}
    for rec, resp in results:
        if list(resp["answers"]) != list(rec["questions"]):
            raise SystemExit(f"ERROR {rec['id']}: server answered {list(resp['answers'])}, asked {list(rec['questions'])}")
        for qid, ans in resp["answers"].items():
            out[f"{rec['id']}.{qid}"] = answer_probs(ans)
    return out


def save_answers(path, results, force=False):
    """Write this run's answers as a REF file (mode 0600, never overwritten without --force)."""
    ref = answers_ref(results)
    write_private(path, json.dumps(ref, indent=0) + "\n", force=force)
    print(f"saved {len(ref)} answers to {path}")


def drive_and_slice(url, dump, force=False, answers_out=None):
    """POST the 50 records and write exactly the dump bytes they added to parity_slice(dump),
    then check the slice aligns with the records (one run, nothing else)."""
    out = parity_slice(dump)
    if out.exists() and not force:
        raise SystemExit(f"{out} exists; pass --force to overwrite it")
    if answers_out and Path(answers_out).exists() and not force:
        raise SystemExit(f"{answers_out} exists; pass --force to overwrite it")
    if not Path(dump).exists():
        raise SystemExit(f"{dump} does not exist: start the server with --decision-dump {dump}")
    start = Path(dump).stat().st_size
    recs = records()
    results = [(rec, systemone(url, rec)) for rec in recs]
    write_private(out, Path(dump).read_bytes()[start:].decode(), force=force)
    passes = sum(len(ps) for _, ps in align(str(out), recs))
    print(f"wrote {passes} passes for {len(recs)} records to {out}")
    if answers_out:
        save_answers(answers_out, results, force=force)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref", default=None, help="omit to only drive the server and slice its --decision-dump (needs --dump)")
    ap.add_argument("--dump", default=None, help="the server's --decision-dump file (same machine); required without --ref")
    ap.add_argument("--force", action="store_true", help="overwrite an existing <dump>.parity.jsonl")
    ap.add_argument("--min-agree", type=float, default=0.96, help="raw top-1 agreement floor (0.96 = at most 3 flips of 82)")
    ap.add_argument("--confident", type=float, default=0.05, help="a flip where the reference margin is >= this fails the gate")
    ap.add_argument("--answers-out", default=None,
                    help="also save this run's answers as a REF file (0600), e.g. a single-box run that a pipeline "
                         "run of the same model is then gated against with --min-agree 1 --confident 0")
    a = ap.parse_args(argv)
    url = a.url.rstrip("/")
    if not a.ref:
        if not a.dump:
            ap.error("without --ref, pass --dump (the server's --decision-dump) so this run's passes are sliced out")
        drive_and_slice(url, a.dump, a.force, a.answers_out)
        return
    if a.answers_out and Path(a.answers_out).exists() and not a.force:
        raise SystemExit(f"{a.answers_out} exists; pass --force to overwrite it")
    with open(a.ref) as f:
        ref = json.load(f)
    results = [(rec, systemone(url, rec)) for rec in records()]
    if a.answers_out:
        save_answers(a.answers_out, results, force=a.force)
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
