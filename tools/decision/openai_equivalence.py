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


def flat_openai(a):
    if a["type"] == "predicate":
        return {"p": [1 - a["probability"], a["probability"]]}
    return {"p": [x["probability"] for x in a["probabilities"]], "c": a["confidence"], "s": a.get("score"), "choice": a.get("choice")}


def flat_systemone(a):
    if a["type"] == "noul":
        return {"p": [1 - a["noul"], a["noul"]]}
    return {"p": list(a["probabilities"].values()), "c": a["confidence"], "s": a.get("score"), "choice": a.get("choice")}


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
            got, want = flat_openai(ans), flat_systemone(so["answers"][ans["name"]])
            n += 1
            if got != want:
                bad += 1
                print(f"DIFF {rec['id']}.{ans['name']}: openai={got} systemone={want}")
    print(f"{n - bad}/{n} questions identical across /v1/decisions and /v1/systemone")
    sys.exit(1 if bad or n == 0 else 0)


if __name__ == "__main__":
    main()
