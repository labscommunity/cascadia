"""/v1/decisions (OpenAI format) must answer exactly like /v1/systemone on the 50 records.

  python tools/decision/openai_equivalence.py --url http://127.0.0.1:8080 [--all-states]

The records' instructions are strings, their noul questions carry no criteria overrides, and their
score levels are plain strings, so both routes build byte-identical prompts: any difference in
probabilities, choice/score or confidence is an adapter bug (crates/cascadia-api/src/decisions/openai.rs).
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

from records import records

KIND = {"noul": "predicate", "choice": "choice", "score": "score"}


def to_openai(rec):
    """SystemOne record -> OpenAI Decisions request. String states pass through; object states go as
    their json.dumps text, which is what the JEV codec renders for objects anyway. Kev and Clef render
    object states structurally, so for them the equivalence check runs on string-state records only."""
    st = rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"], ensure_ascii=False)
    qs = []
    for qid, q in rec["questions"].items():
        o = {"type": KIND[q["type"]], "name": qid, "instructions": q.get("instructions") or qid}
        if q["type"] == "choice":
            o["choices"] = [{"value": k, "description": v} for k, v in q["criteria"].items()]
        elif q["type"] == "score":
            o["levels"] = [{"label": lvl} for lvl in q["criteria"]]
        qs.append(o)
    return {"model": "m", "input": st, "questions": qs}


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ERROR {url}: HTTP {e.code} {e.read().decode(errors='replace')}")


def choice_key(v):
    """The server's internal key for an OpenAI choice value (choice_key in openai.rs): the string
    itself, or Python's str() of a number (2 -> "2", 1e5 -> "100000.0")."""
    return v if isinstance(v, str) else str(v)


def flat_openai(a):
    if a["type"] == "predicate":
        return {"p": [1 - a["probability"], a["probability"]]}
    out = {"p": [x["probability"] for x in a["probabilities"]], "c": a["confidence"], "s": a.get("score")}
    if a["type"] == "choice":
        out["choice"] = choice_key(a["choice"])
        out["names"] = [choice_key(x["value"]) for x in a["probabilities"]]
    else:
        out["choice"] = None
        out["names"] = [str(x["value"]) for x in a["probabilities"]]
    return out


def flat_systemone(a):
    if a["type"] == "noul":
        return {"p": [1 - a["noul"], a["noul"]]}
    return {"p": list(a["probabilities"].values()), "c": a["confidence"], "s": a.get("score"), "choice": a.get("choice"),
            "names": list(a["probabilities"])}


def compare_answer(rec, oa, so):
    """Problems between one /v1/decisions answer and the /v1/systemone answer for the same question:
    probabilities, confidence, score and choice, plus the option NAMES (choice values, score level
    indices) and the score labels against what the record sent, not just positions."""
    q = rec["questions"][oa["name"]]
    problems = []
    if oa["type"] != KIND[q["type"]]:
        problems.append(f"type {oa['type']} for a {q['type']} question")
        return problems
    got, want = flat_openai(oa), flat_systemone(so)
    if got != want:
        problems.append(f"openai={got} systemone={want}")
    if q["type"] == "choice" and got["names"] != list(q["criteria"]):
        problems.append(f"choice values {got['names']} != request {list(q['criteria'])}")
    if q["type"] == "score":
        labels = [x.get("label") for x in oa["probabilities"]]
        if labels != list(q["criteria"]):
            problems.append(f"score labels {labels} != request {list(q['criteria'])}")
        if [x["value"] for x in oa["probabilities"]] != list(range(len(q["criteria"]))):
            problems.append(f"score values {[x['value'] for x in oa['probabilities']]} are not 0..{len(q['criteria']) - 1}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--all-states", action="store_true", help="also object-state records (JEV only: identical rendering)")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    n = bad = 0
    for rec in records():
        if not a.all_states and not isinstance(rec["state"], str):
            continue
        oa = post(url + "/v1/decisions", to_openai(rec))
        so = post(url + "/v1/systemone", {"model": "m", "state": rec["state"], "questions": rec["questions"]})
        if [x["name"] for x in oa["answers"]] != list(so["answers"]):
            n += 1
            bad += 1
            print(f"DIFF {rec['id']}: answer names/order differ: openai={[x['name'] for x in oa['answers']]} "
                  f"systemone={list(so['answers'])}")
            continue
        for ans in oa["answers"]:
            problems = compare_answer(rec, ans, so["answers"][ans["name"]])
            n += 1
            if problems:
                bad += 1
                print(f"DIFF {rec['id']}.{ans['name']}: " + "; ".join(problems))
    print(f"{n - bad}/{n} questions identical across /v1/decisions and /v1/systemone")
    sys.exit(1 if bad or n == 0 else 0)


if __name__ == "__main__":
    main()
