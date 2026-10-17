"""Typed-decisions latency benchmark and run-to-run parity for a running decisions server.

  python tools/decision/bench_cases.py cases td-test.json td40.jsonl [--per-workflow 10]
  python tools/decision/bench_cases.py run --url http://127.0.0.1:8080 --model kev-9b \\
      --cases td40.jsonl --out kev-base.jsonl [--per-workflow 10] [--timeout 600]
  python tools/decision/bench_cases.py compare kev-base.jsonl kev-new.jsonl [--confident 0.05] [--max-dp 0.05]
  python tools/decision/bench_cases.py logsum server.log [--event "qwen36 decision pass done"]

cases converts a typed-decisions split (a JSON array or JSONL of rows with "id", "workflow" and
"state"/"questions" as JSON strings or objects) into CASES, keeping the first N rows of each workflow
in file order and interleaving them round-robin over the sorted workflow names: case i of every
workflow, then case i+1. That is the order the 2026-10-17 baseline client (td_perf.py) posted, so
runs of this tool compare against its perf-*.jsonl answers.

CASES is JSONL, one case per line: {"id": str, "workflow": str, "body": {...}}, where body is a
/v1/systemone request without "model" (run sets it). Cases are posted one at a time, in file order;
--per-workflow N keeps the first N of each workflow. OUT is JSONL, one line per case (mode 0600, never
overwritten). compare matches cases by id and exits 1 when max |dp| > --max-dp, a flip happens where
the first run's margin is >= --confident, the two runs hold different cases or questions, no case
was answered in both, or a case errored in either run (--allow-errors reports those without
failing). It also reads the
baseline client's lines ({"id", "latency_s", "probs"}: a noul as {"false": 1-p, "true": p}, a choice
or score as its probabilities). logsum prints the median and p90 of every numeric key=value field of
the matching log lines.

CASES and OUT hold request content (states, questions, answers), like --decision-dump: keep them
private.
"""
import argparse
import http.client
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict, defaultdict

from parity_check import answer_probs, margin


def select(cases, per_workflow):
    if per_workflow is None:
        return list(cases)
    seen, out = defaultdict(int), []
    for c in cases:
        if seen[c["workflow"]] < per_workflow:
            seen[c["workflow"]] += 1
            out.append(c)
    return out


def _json_field(v):
    return json.loads(v) if isinstance(v, str) else v


def round_robin(rows, per_workflow):
    """The baseline client's order: `by[w]` lists in file order, then
    `[by[w][i] for i in range(per_workflow) for w in sorted(by)]` (a workflow with fewer rows is
    skipped once it runs out)."""
    by = defaultdict(list)
    for r in rows:
        by[r["workflow"]].append(r)
    return [by[w][i] for i in range(per_workflow) for w in sorted(by) if i < len(by[w])]


def to_cases(rows, per_workflow):
    return [{"id": str(r["id"]), "workflow": r["workflow"],
             "body": {"state": _json_field(r["state"]), "questions": _json_field(r["questions"])}}
            for r in round_robin(rows, per_workflow)]


def load_rows(path):
    """A JSON array (or {"rows": [...]}) or JSONL of dataset rows."""
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    if isinstance(data, dict):
        rows = data.get("rows", data.get("data"))
        if not isinstance(rows, list):
            sys.exit(f"{path}: a JSON object needs a 'rows' or 'data' list")
        data = [r.get("row", r) for r in rows]
    return data


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode(errors="replace")[:500]
        except (http.client.HTTPException, OSError) as e2:
            body = f"(body unreadable: {e2})"
        return e.code, time.perf_counter() - t0, None, body
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as e:
        return 0, time.perf_counter() - t0, None, f"{type(e).__name__}: {e}"
    lat = time.perf_counter() - t0
    try:
        return status, lat, json.loads(raw), None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # Not a success: the error field makes `ok` reject it, whatever the status.
        return status, lat, None, f"invalid JSON in the response: {e}"[:500]


def pct(xs, q):
    """Linear-interpolated percentile (numpy's default), q in [0, 100]."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ok(r):
    """This tool's lines carry status; the baseline client's carry none (a 200) or an error."""
    return r.get("status", 200) == 200 and not r.get("error")


def probs_of(r):
    """Question id -> probabilities in API option order, from either line format."""
    if "answers" in r:
        return {q: answer_probs(a) for q, a in r["answers"].items()}
    out = {}
    for q, d in (r.get("probs") or {}).items():
        # A baseline noul is {"false": 1-p, "true": p}: answer_probs order, whatever the key order.
        out[q] = [d["false"], d["true"]] if set(d) == {"false", "true"} else list(d.values())
    return out


def summary(results):
    by = OrderedDict()
    for r in results:
        by.setdefault(r.get("model"), []).append(r)
    out = {}
    for m, rs in by.items():
        good = [r["latency_s"] for r in rs if ok(r)]
        out[m] = {"n": len(good), "errors": len(rs) - len(good), "p50": pct(good, 50),
                  "mean": statistics.fmean(good) if good else float("nan"), "p90": pct(good, 90)}
    return out


def compare(a, b, confident, max_dp, allow_errors=False):
    """Fails on cases in only one file, a question in only one run's answer (or with a different
    option count), max |dp| > max_dp, a flip where a's margin is >= confident, no case compared,
    or (unless allow_errors) a case that errored in either run."""
    ia = {r["id"]: probs_of(r) for r in a if ok(r)}
    ib = {r["id"]: probs_of(r) for r in b if ok(r)}
    errors_a = sorted(r["id"] for r in a if not ok(r))
    errors_b = sorted(r["id"] for r in b if not ok(r))
    missing = sorted({r["id"] for r in a} ^ {r["id"] for r in b})
    worst, flips, sure, compared = 0.0, 0, 0, 0
    for cid in sorted(set(ia) & set(ib)):
        compared += 1
        qa, qb = ia[cid], ib[cid]
        for qid in list(qa) + [q for q in qb if q not in qa]:
            pa, pb = qa.get(qid), qb.get(qid)
            if pa is None or pb is None or len(pb) != len(pa):
                missing.append(f"{cid}.{qid}")
                continue
            worst = max(worst, max(abs(x - y) for x, y in zip(pa, pb)))
            if pa.index(max(pa)) != pb.index(max(pb)):
                flips += 1
                sure += margin(pa) >= confident
    errored = bool(errors_a or errors_b) and not allow_errors
    ok_ = not missing and worst <= max_dp and sure == 0 and compared > 0 and not errored
    return {"max_dp": worst, "flips": flips, "confident_flips": sure, "missing": missing,
            "errors_a": errors_a, "errors_b": errors_b, "compared": compared, "ok": ok_}


def model_pairs(sa, sb):
    """Models to set side by side: the shared ones, or the only one of each run (a baseline line
    carries no model)."""
    shared = [m for m in sa if m in sb]
    if shared:
        return [(m, m) for m in shared]
    if len(sa) == 1 and len(sb) == 1:
        return [(next(iter(sa)), next(iter(sb)))]
    return []


ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r'(\w+)=("[^"]*"|\S+)')


def logsum(lines, event):
    vals = defaultdict(list)
    for line in lines:
        line = ANSI.sub("", line)
        if event not in line:
            continue
        for k, v in FIELD.findall(line.split(event, 1)[1]):
            try:
                vals[k].append(float(v))
            except ValueError:
                pass
    return {k: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90)} for k, v in vals.items()}


def open_private(path):
    """A new file, mode 0600; refuses an existing one."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(fd, "w")


def write_private(path, lines):
    with open_private(path) as f:
        for x in lines:
            f.write(json.dumps(x) + "\n")


def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("cases")
    k.add_argument("split")
    k.add_argument("out")
    k.add_argument("--per-workflow", type=int, default=10)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--per-workflow", type=int)
    r.add_argument("--timeout", type=float, default=600.0)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--confident", type=float, default=0.05)
    c.add_argument("--max-dp", type=float, default=0.05)
    c.add_argument("--allow-errors", action="store_true",
                   help="do not fail on cases that errored in either run (they are still reported)")
    s = sub.add_parser("logsum")
    s.add_argument("log")
    s.add_argument("--event", default="qwen36 decision pass done")
    a = ap.parse_args(argv)
    if a.cmd == "cases":
        if os.path.exists(a.out):
            sys.exit(f"{a.out} exists; pick a new name")
        cases = to_cases(load_rows(a.split), a.per_workflow)
        write_private(a.out, cases)
        n = len({x["workflow"] for x in cases})
        print(f"{len(cases)} cases over {n} workflows -> {a.out}")
        return 0
    if a.cmd == "run":
        if os.path.exists(a.out):
            sys.exit(f"{a.out} exists; pick a new name")
        cases = select(load(a.cases), a.per_workflow)
        results = []
        # Written line by line, so a run cut short keeps the cases it finished.
        with open_private(a.out) as out:
            for case in cases:
                body = dict(case["body"], model=a.model)
                status, lat, data, err = post(a.url.rstrip("/"), body, a.timeout)
                rec = {"id": case["id"], "workflow": case["workflow"], "model": a.model,
                       "status": status, "latency_s": lat,
                       "answers": (data or {}).get("answers", {}),
                       "usage": (data or {}).get("usage", {}), "error": err}
                results.append(rec)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"{case['id']} {status} {lat:.3f}s", flush=True)
        for m, v in summary(results).items():
            print(f"{m}: p50 {v['p50']:.2f} s, mean {v['mean']:.2f} s, p90 {v['p90']:.2f} s, "
                  f"n {v['n']}, errors {v['errors']}")
        return 0
    if a.cmd == "compare":
        ra, rb = load(a.a), load(a.b)
        res = compare(ra, rb, a.confident, a.max_dp, a.allow_errors)
        sa, sb = summary(ra), summary(rb)
        for ma, mb in model_pairs(sa, sb):
            d = (sb[mb]["p50"] - sa[ma]["p50"]) / sa[ma]["p50"] * 100
            label = ma if ma == mb else f"{ma or a.a} -> {mb or a.b}"
            print(f"{label}: p50 {sa[ma]['p50']:.2f} -> {sb[mb]['p50']:.2f} s ({d:+.1f}%), "
                  f"p90 {sa[ma]['p90']:.2f} -> {sb[mb]['p90']:.2f} s, "
                  f"errors {sa[ma]['errors']} -> {sb[mb]['errors']}")
        print(f"compared {res['compared']} cases; errored: {len(res['errors_a'])} in {a.a}, "
              f"{len(res['errors_b'])} in {a.b}" + (" (allowed)" if a.allow_errors else ""))
        for side, errs in ((a.a, res["errors_a"]), (a.b, res["errors_b"])):
            if errs:
                print(f"errored in {side}: {errs}")
        if res["compared"] == 0:
            print("no case answered in both runs: nothing compared")
        print(f"max |dp| = {res['max_dp']:.4f}")
        print(f"flips = {res['flips']} (confident {res['confident_flips']})")
        if res["missing"]:
            print(f"cases or questions in only one run: {res['missing']}")
        print("PASS" if res["ok"] else "FAIL")
        return 0 if res["ok"] else 1
    with open(a.log, errors="replace") as f:
        for key, v in sorted(logsum(f, a.event).items()):
            print(f"{key}: p50 {v['p50']:.1f}, p90 {v['p90']:.1f}, n {v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
