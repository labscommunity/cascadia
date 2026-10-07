"""Golden check: the server's decision passes vs independent reference encoders, and its answers vs the
spike's saved full-precision references. Catches codec bugs that int8-vs-fp32 parity cannot see.

  python tools/decision/codec_golden.py kev  --url http://127.0.0.1:8080 --dump kev.jsonl  --tokenizer ~/models/kev9b [--answers]
  python tools/decision/codec_golden.py clef --url http://127.0.0.1:8080 --dump clef.jsonl --tokenizer ~/models/clef-flash [--answers]
  python tools/decision/codec_golden.py jev  --url http://127.0.0.1:8080 --dump jev.jsonl  --tokenizer ~/models/jev9b --adapter-dir ~/models/jev9b/adapter_vllm [--answers]

The server must run with `--decision-dump DUMP`. Each dump line is one decision pass, written by
crates/cascadia-api/src/decisions/handler.rs: {"parts": [[question_id, logit_start, logit_end], ...],
"readout": {"prompt_ids": [...], "head": {"kind": "verbalizer"|"pointer"|"joint", ...}, "cache_prefix"?: n}}.

Reference encoders come from the spike sources under $SPIKE_DIR (imported at run time, never vendored):
Kev `kev-spike/kev.py` questions(), Clef `clef-spike/flash/joint_schema_model.py` encode_record(),
and for JEV the serve_decide layout reproduced below (jev_text).
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from records import records

SPIKE_DIR = Path(os.environ.get("SPIKE_DIR", "/Users/blokchainaholic/Desktop/clabs/tahoma/experiments"))


def align(dump_path, recs):
    """Yield (record, [pass, ...]) by consuming dump lines until every question of the record is covered.
    A pass naming a question the record does not have, or lines left over at the end, mean the dump
    and the records are out of step (another client hit the server, or the codec split differently)."""
    with open(dump_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    i = 0
    for rec in recs:
        need, got = set(rec["questions"]), []
        while need:
            if i >= len(lines):
                raise SystemExit(f"MISMATCH dump ends before record {rec['id']} is covered")
            qids = {p[0] for p in lines[i]["parts"]}
            if not qids <= need:
                raise SystemExit(f"MISMATCH dump line {i + 1} covers {sorted(qids - need)}, not open questions of {rec['id']}")
            got.append(lines[i])
            need -= qids
            i += 1
        yield rec, got
    if i != len(lines):
        raise SystemExit(f"MISMATCH dump has {len(lines) - i} pass(es) beyond the {len(recs)} records sent")


def fail(msg):
    print(f"MISMATCH {msg}")
    sys.exit(1)


# Golden-only records: every rendering branch the 50 parity records never reach. No saved
# reference exists for them, so they are compared on prompts/positions only, never on answers.
SYNTHETIC = [
    {"id": "s00", "state": ["first turn. ", {"type": "text", "text": "second turn."}, {"k": 1.5}],
     "questions": {"u": {"type": "noul", "instructions": "Is this resolved?",
                         "criteria": {"true": "Yes, closed.", "false": "No, still open."}}}},
    {"id": "s01", "state": {"items": [{"sku": "a1", "price": 1e-05}, {"sku": "b2", "price": 1e16}], "total": 2.5},
     "questions": {"c": {"type": "choice", "instructions": "",
                         "criteria": {"refund": {"when": "charged twice", "max": 500}, "ignore": "no action"}}}},
    {"id": "s02", "state": "Plain text state.",
     "questions": {"s": {"type": "score", "instructions": None, "criteria": ["none", "minor", "major", "critical"]}}},
]


def branches(rec):
    out = set()
    if isinstance(rec["state"], list):
        out.add("list_state")
    if "1e" in json.dumps(rec["state"]) or ".5" in json.dumps(rec["state"]):
        out.add("float")
    for q in rec["questions"].values():
        if q["type"] == "noul" and q.get("criteria"):
            out.add("noul_criteria")
        if q["type"] == "choice" and any(isinstance(v, dict) for v in q["criteria"].values()):
            out.add("dict_description")
        if q.get("instructions") == "":
            out.add("empty_instructions")
        if "instructions" in q and q["instructions"] is None:
            out.add("null_instructions")
    return out


def golden_records():
    return records() + SYNTHETIC


def jev_state(state):
    """serve_decide _parts: str verbatim, dict json.dumps, list concatenated part by part."""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "".join(p if isinstance(p, str) else jev_value(p.get("text")) if isinstance(p, dict) and p.get("type") == "text"
                       else json.dumps(p, ensure_ascii=False) for p in state)
    return json.dumps(state, ensure_ascii=False)


def jev_value(v):
    return "" if v is None else v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def jev_text(rec, qid, labels):
    """serve_decide s1_pass layout, caller order; mirrors the server codec's rendering rules.
    The server also escapes special-token text (<|...|>) in user strings; none of the golden
    records contain any, so that step is the identity here."""
    q = rec["questions"][qid]
    kind = q["type"]
    if kind == "choice":
        lines = [f"{labels[i]}) {n}: {jev_value(d)}" if jev_value(d) else f"{labels[i]}) {n}" for i, (n, d) in enumerate(q["criteria"].items())]
    elif kind == "noul":
        lines = ["false", "true"]
    else:
        lines = [str(i) for i in range(6)]
    question = jev_value(q["instructions"]) if q.get("instructions") is not None else qid
    return f"[kind] {kind}\n[state] {jev_state(rec['state'])}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def check_jev(dump, tok, adapter_dir, recs):
    head = json.loads((Path(adapter_dir) / "decision_head.json").read_text())
    ranges = head["slots"]["ranges"]
    labels = head["slots"]["verbalizers"][ranges["choice"][0]:ranges["choice"][1]]
    n = 0
    for rec, passes in align(dump, recs):
        for p in passes:
            qid = p["parts"][0][0]
            q = rec["questions"][qid]
            want = tok(jev_text(rec, qid, labels), add_special_tokens=False).input_ids
            if p["readout"]["prompt_ids"] != want:
                fail(f"jev {rec['id']}.{qid}: prompt_ids differ")
            lo, hi = ranges[q["type"]]
            if q["type"] == "choice":
                hi = lo + len(q["criteria"])
            if p["readout"]["head"]["token_ids"] != head["verbalizer_ids"][lo:hi]:
                fail(f"jev {rec['id']}.{qid}: verbalizer token_ids differ from decision_head.json")
            n += 1
    print(f"jev: {n} passes match the serve_decide layout and decision_head.json")


def check_kev(dump, tok, adapter_dir, recs):
    sys.path.insert(0, str(SPIKE_DIR / "kev-spike"))
    import kev
    kev.records = lambda: recs  # kev.questions iterates kev.records(); point it at the golden set
    want = {qid: (row, d, ends) for qid, kind, row, d, ends, keys in kev.questions(tok)}
    n = 0
    for rec, passes in align(dump, recs):
        for p in passes:
            qid = p["parts"][0][0]
            row, d, ends = want[f"{rec['id']}.{qid}"]
            r = p["readout"]
            if r["prompt_ids"] != list(row) or r["head"]["decide_pos"] != d or r["head"]["option_pos"] != list(ends):
                fail(f"kev {rec['id']}.{qid}: row/positions differ")
            n += 1
    print(f"kev: {n} passes match kev.questions")


_CLEF_REF = None


def clef_reference():
    """reference_flash.pt (~200 MB of saved hidden states), memory-mapped and loaded once."""
    global _CLEF_REF
    if _CLEF_REF is None:
        import torch
        _CLEF_REF = torch.load(SPIKE_DIR / "clef-spike" / "reference_flash.pt", map_location="cpu",
                               weights_only=True, mmap=True)
    return _CLEF_REF


def check_clef(dump, tok, adapter_dir, recs):
    sys.path.insert(0, str(SPIKE_DIR / "clef-spike" / "flash"))
    import joint_schema_model as jsm
    saved = {e["id"]: e["input_ids"] for e in clef_reference()}
    n = 0
    for rec, passes in align(dump, recs):
        if len(passes) != 1:
            fail(f"clef {rec['id']}: {len(passes)} passes, expected one joint pass")
        (p,) = passes
        enc = jsm.encode_record(tok, rec, max_length=16384)
        r = p["readout"]
        if r["prompt_ids"] != list(enc.input_ids):
            fail(f"clef {rec['id']}: prompt_ids differ")
        if rec["id"] in saved and r["prompt_ids"] != list(saved[rec["id"]]):
            fail(f"clef {rec['id']}: prompt_ids differ from reference_flash.pt input_ids")
        if len(r["head"]["questions"]) != len(enc.questions):
            fail(f"clef {rec['id']}: {len(r['head']['questions'])} joint questions, encode_record has {len(enc.questions)}")
        for q, eq in zip(r["head"]["questions"], enc.questions):
            if q["qtype"] != eq.question_type or tuple(q["span"]) != tuple(eq.question_span) or \
                    [tuple(o["span"]) for o in q["options"]] != [tuple(s) for s in eq.option_spans]:
                fail(f"clef {rec['id']}.{eq.question_id}: spans/type differ")
        n += 1
    print(f"clef: {n} records match encode_record")


def check_answers(family, url):
    import torch
    if family == "clef":
        # reference_flash.pt is per RECORD: {id, input_ids, hidden, logits=[one tensor per question]}
        recs = {r["id"]: r for r in records()}
        ref = {f"{e['id']}.{qid}": {"probs": torch.as_tensor(lg).float().softmax(-1)}
               for e in clef_reference() for qid, lg in zip(recs[e["id"]]["questions"], e["logits"])}
    else:
        ref_file = {"kev": SPIKE_DIR / "kev-spike" / "reference_kev.pt", "jev": SPIKE_DIR / "jev-spike" / "reference_jev.pt"}[family]
        ref = {r["qid"]: r for r in torch.load(ref_file, map_location="cpu", weights_only=True)}
    agree = total = 0
    for rec in records():
        resp = systemone(url, rec)
        for qid, ans in resp["answers"].items():
            r = ref.get(f"{rec['id']}.{qid}")
            if r is None:
                continue
            q = rec["questions"][qid]
            if family == "jev" and q["type"] == "choice":
                print(f"  (jev choice, informational) {rec['id']}.{qid}: server={top1(ans)} reference={reference_top1(family, q, r)}")
                continue
            if family == "jev" and q["type"] == "score":
                r = {"probs": torch.tensor(rescale_score(r["probs"].tolist(), len(q["criteria"])))}
            want = reference_top1(family, q, r)
            got = top1(ans)
            agree += got == want
            total += 1
            if got != want:
                print(f"ANSWER {rec['id']}.{qid}: server={got} reference={want}")
    print(f"{family}: server answers agree with the saved reference on {agree}/{total}")
    if total == 0 or agree < 0.98 * total:
        sys.exit(1)


def rescale_score(p6, levels):
    """Same mapping as the JEV codec's rescale_score (crates/cascadia-api/src/decisions/jev.rs)."""
    out = [0.0] * levels
    for j, p in enumerate(p6):
        x = j * (levels - 1) / (len(p6) - 1)
        lo = int(x)
        w = x - lo
        out[lo] += p * (1 - w)
        if w > 0:
            out[lo + 1] += p * w
    return out


def systemone(url, rec):
    """POST one record to /v1/systemone; a non-2xx reply is a golden failure with the server's message."""
    body = json.dumps({"model": "m", "state": rec["state"], "questions": rec["questions"]}).encode()
    req = urllib.request.Request(url + "/v1/systemone", data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        fail(f"{rec['id']}: HTTP {e.code} {e.read().decode(errors='replace')}")


def post_all(url, recs):
    for rec in recs:
        systemone(url, rec)


def top1(ans):
    if ans["type"] == "noul":
        return "true" if ans["noul"] >= 0.5 else "false"
    if ans["type"] == "choice":
        return ans["choice"]
    p = list(ans["probabilities"].values())
    return str(max(range(len(p)), key=p.__getitem__))


def reference_top1(family, q, r):
    """Name of the reference's top option in API terms. The saved references carry no option names,
    so this is each spike's stored order: Clef noul [true, false] and sorted choices
    (joint_schema_model.question_options); Kev noul [false, true] and caller-order choices
    (kev.questions); JEV noul [false, true] and sorted choices (jev.questions)."""
    i = int(r["probs"].argmax())
    kind = q["type"]
    if kind == "noul":
        order = ["true", "false"] if family == "clef" else ["false", "true"]
        return order[i]
    if kind == "choice":
        names = sorted(q["criteria"]) if family in ("clef", "jev") else list(q["criteria"])
        return names[i]
    return str(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["jev", "kev", "clef"])
    ap.add_argument("--url", required=True, help="the running server (started with --decision-dump DUMP)")
    ap.add_argument("--dump", required=True, help="the server's --decision-dump file (same machine)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--adapter-dir", default=None, help="jev: dir with decision_head.json")
    ap.add_argument("--answers", action="store_true", help="also compare top-1 answers with the saved spike reference")
    a = ap.parse_args()
    if a.family == "jev" and not a.adapter_dir:
        raise SystemExit("jev needs --adapter-dir (decision_head.json)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    url = a.url.rstrip("/")
    # Compare only what THIS run appends: earlier parity runs may already be in the dump.
    start = Path(a.dump).stat().st_size if Path(a.dump).exists() else 0
    recs = golden_records()
    post_all(url, recs)
    tail = Path(a.dump).read_bytes()[start:].decode()
    fresh = Path(a.dump).with_suffix(".golden.jsonl")
    fresh.write_text(tail)
    {"jev": check_jev, "kev": check_kev, "clef": check_clef}[a.family](str(fresh), tok, a.adapter_dir, recs)
    if a.answers:
        check_answers(a.family, url)


if __name__ == "__main__":
    main()
