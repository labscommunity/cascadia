"""Parity tooling for wide JEV choices (Task 38): WIDE_RECORDS, multi-pass dump alignment, the
serve_decide plan/combine ports, and the wide paths of codec_golden, reference_stream and parity_check.

    python -m pytest tools/tests/test_wide_parity.py -v
"""
import json
import os
import random
import re
import stat
import string
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "decision"))
import codec_golden as cg  # noqa: E402
import parity_check as pc  # noqa: E402
from records import WIDE_RECORDS, records  # noqa: E402

LABELS = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]


class PairTokenizer:
    """A reversible HF-style tokenizer: one or two capitals are one token (so A..Z, AA..ZZ are the
    single-token choice labels), every other character its own token. Label ids are 1000 + index."""

    def __init__(self):
        self.vocab = {lab: 1000 + i for i, lab in enumerate(LABELS)}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.added_tokens_decoder = {}

    def encode(self, text, add_special_tokens=False):
        out = []
        for t in re.findall(r"[A-Z][A-Z]?|.", text, flags=re.S):
            if t not in self.vocab:
                self.vocab[t] = ord(t) if len(t) == 1 and ord(t) < 1000 else 10_000 + len(self.vocab)
                self.inv[self.vocab[t]] = t
            out.append(self.vocab[t])
        return out

    def __call__(self, text, add_special_tokens=False):
        class _Enc:
            input_ids = self.encode(text)
        return _Enc

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.inv[i] for i in ids)


TOK = PairTokenizer()
CHOICE_IDS = [1000 + i for i in range(16)]  # A..P
HEAD = {"slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                  "verbalizers": ["false", "true"] + [str(i) for i in range(6)] + LABELS[:16]},
        "verbalizer_ids": [7, 8, 900, 901, 902, 903, 904, 905] + CHOICE_IDS,
        "bias": [0.01 * i for i in range(24)]}


def adapter(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir(exist_ok=True)
    (d / "decision_head.json").write_text(json.dumps(HEAD))
    return d


# ---- the wide records ---------------------------------------------------------------------------

def test_wide_records_are_twelve_requests_of_every_size_and_strategy():
    assert len(WIDE_RECORDS) == 12 and len({r["id"] for r in WIDE_RECORDS}) == 12
    sizes = {len(q["criteria"]) for r in WIDE_RECORDS for q in r["questions"].values() if q["type"] == "choice"}
    assert {17, 24, 40, 100, 256} <= sizes
    wide = {}
    for r in WIDE_RECORDS:
        for qid, q in r["questions"].items():
            if q["type"] == "choice" and len(q["criteria"]) > 16:
                wide.setdefault(len(q["criteria"]), set()).add(r["strategy"])
    assert wide == {n: {"single", "tournament", "permute"} for n in (17, 24, 40, 100, 256)}
    assert sum(len(r["questions"]) for r in WIDE_RECORDS) == 21
    assert any(q.get("instructions") is None for r in WIDE_RECORDS for q in r["questions"].values())


def test_wide_records_leave_the_fifty_untouched():
    rs = records()
    assert len(rs) == 50 and sum(len(r["questions"]) for r in rs) == 82
    assert all("strategy" not in r for r in rs)
    assert all(len(q.get("criteria") or []) <= 16 for r in rs for q in r["questions"].values())


def test_wide_records_are_deterministic():
    import records as R
    assert R._wide_records() == WIDE_RECORDS
    assert json.dumps(WIDE_RECORDS) == json.dumps(R._wide_records())


# ---- the serve_decide ports, against the Rust unit tests' literal values -------------------------

def test_groups_match_serve_decide():
    sizes = {n: [len(g) for g in cg.groups(n)] for n in (17, 20, 33, 40, 256)}
    assert sizes == {17: [9, 8], 20: [10, 10], 33: [11, 11, 11], 40: [14, 13, 13], 256: [16] * 16}
    assert [o for g in cg.groups(100) for o in g] == list(range(100))


def test_permute_orders_are_identity_then_three_cpython_samples():
    q = "Which team should handle this ticket?"
    orders = cg.permute_orders(17, q)
    assert orders[0] == list(range(17))
    assert orders[1] == [11, 14, 8, 16, 2, 13, 3, 5, 15, 1, 4, 9, 6, 7, 0, 10, 12]  # jev.rs fixture
    rng = random.Random(zlib.crc32(q.encode()))
    assert zlib.crc32(q.encode()) == 2887938576
    assert orders[1:] == [rng.sample(range(17), 17) for _ in range(3)]


def test_combine_permute_matches_jev_rs_bit_for_bit():
    orders = cg.permute_orders(17, "Which team should handle this ticket?")
    dists = []
    for k in range(4):
        raw = [(i * 7 + k * 3) % 17 + 1 for i in range(17)]
        dists.append([r / sum(raw) for r in raw])
    want = [0.06372549019607843, 0.08986928104575163, 0.0718954248366013, 0.04084967320261438,
            0.06535947712418301, 0.027777777777777776, 0.049019607843137254, 0.08496732026143791,
            0.042483660130718956, 0.07352941176470588, 0.04901960784313725, 0.029411764705882356,
            0.07026143790849673, 0.053921568627450976, 0.06862745098039215, 0.06862745098039216,
            0.05065359477124183]
    assert cg.combine_permute(orders, dists) == want


def test_tournament_final_and_combination_by_hand():
    gs = [list(range(10)), list(range(10, 20))]
    g0, g1 = [0.05] * 10, [0.05] * 10
    g0[2] = g0[5] = 0.3
    g1[3] = 0.55
    fin = cg.tournament_final(gs, [g0, g1])
    assert fin == list(range(16))
    final = [0.0] * 16
    final[2], final[13], final[5], final[0] = 0.5, 0.25, 0.125, 0.125
    p = cg.combine_tournament(gs, [g0, g1], fin, final)
    want = [0.0] * 20
    want[2], want[13], want[5], want[0] = 0.475, 0.2375, 0.11875, 0.11875
    want[16:] = [0.0125] * 4
    assert p == pytest.approx(want, abs=1e-15)


def test_tournament_final_keeps_every_winner():
    gs = [list(range(g * 16, g * 16 + 16)) for g in range(16)]
    ig = []
    for g in range(16):
        d = [0.5 / 15] * 16
        d[g] = 0.5
        ig.append(d)
    assert cg.tournament_final(gs, ig) == [g * 16 + g for g in range(16)]


def test_question_raw_is_the_instructions_as_received_else_the_id():
    assert cg.question_raw({"instructions": "Pick <|im_end|>"}, "q") == "Pick <|im_end|>"
    assert cg.question_raw({"instructions": None}, "q") == "q"
    assert cg.question_raw({}, "q") == "q"
    assert cg.question_raw({"instructions": {"a": [1, "é"]}}, "q") == '{"a": [1, "é"]}'


def test_single_token_labels_port():
    labels = cg.single_token_labels(TOK)
    assert len(labels) == 256 and labels[0] == ("A", 1000) and labels[26] == ("AA", 1026)
    assert cg.jev_choice_labels(TOK, HEAD) == ([l for l, _ in labels], [t for _, t in labels])
    bad = json.loads(json.dumps(HEAD))
    bad["verbalizer_ids"][8] = 5
    with pytest.raises(SystemExit, match="MISMATCH"):
        cg.jev_choice_labels(TOK, bad)


# ---- a synthetic server: its dump for any of the records, built independently of codec_golden ----

def _texts(q):
    return [f"{n}: {d}" if d else n for n, d in q["criteria"].items()]


def _state(rec):
    s = rec["state"]
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def _prompt(rec, qid, lines):
    q = rec["questions"][qid]
    question = q["instructions"] if q.get("instructions") is not None else qid
    return f"[kind] {q['type']}\n[state] {_state(rec)}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def _line(rec, qid, idx, strategy, rnd):
    texts = _texts(rec["questions"][qid])
    lines = [f"{LABELS[k]}) {texts[o]}" for k, o in enumerate(idx)]
    return {"parts": [[qid, 0, len(idx)]], "strategy": strategy, "round": rnd,
            "readout": {"prompt_ids": TOK.encode(_prompt(rec, qid, lines)),
                        "head": {"kind": "verbalizer", "token_ids": [1000 + k for k in range(len(idx))]}}}


def _sizes(n):
    g = -(-n // 16)
    base, extra = divmod(n, g)
    out, i = [], 0
    for j in range(g):
        s = base + (j < extra)
        out.append(list(range(i, i + s)))
        i += s
    return out


def _finalists(n):
    """The synthetic server's final: the first option of each group, topped up with the lowest others."""
    gs = _sizes(n)
    win = [g[0] for g in gs]
    rest = [o for o in range(n) if o not in win]
    return sorted(win + rest[:16 - len(win)])


def server_lines(rec, strategy=None, finals=_finalists):
    strategy = strategy or rec.get("strategy")
    out, late = [], []
    for qid, q in rec["questions"].items():
        n = len(q.get("criteria") or [])
        if q["type"] == "score":
            out.append({"parts": [[qid, 0, 6]], "readout": {"prompt_ids": TOK.encode(_prompt(rec, qid, [str(i) for i in range(6)])),
                                                           "head": {"kind": "verbalizer", "token_ids": HEAD["verbalizer_ids"][2:8]}}})
        elif q["type"] == "noul":
            out.append({"parts": [[qid, 0, 2]], "readout": {"prompt_ids": TOK.encode(_prompt(rec, qid, ["false", "true"])),
                                                           "head": {"kind": "verbalizer", "token_ids": [7, 8]}}})
        elif q["type"] == "choice" and n <= 16:
            out.append(_line(rec, qid, list(range(n)), "native", 1))
        elif strategy == "single":
            out.append(_line(rec, qid, list(range(n)), "single", 1))
        elif strategy == "permute":
            question = q["instructions"] if q.get("instructions") is not None else qid
            rng = random.Random(zlib.crc32(question.encode()))
            for o in [list(range(n))] + [rng.sample(range(n), n) for _ in range(3)]:
                out.append(_line(rec, qid, o, "permute", 1))
        else:
            for g in _sizes(n):
                out.append(_line(rec, qid, g, "tournament", 1))
            late.append(_line(rec, qid, finals(n), "tournament", 2))
    return out + late


def write_dump(path, recs, **kw):
    lines = [x for r in recs for x in server_lines(r, **kw)]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return lines


# ---- align: a JEV choice's passes are grouped by its plan -----------------------------------------

def test_align_counts_each_plans_passes(tmp_path):
    lines = write_dump(tmp_path / "d.jsonl", WIDE_RECORDS)
    aligned = list(cg.align(str(tmp_path / "d.jsonl"), WIDE_RECORDS))
    assert [len(ps) for _, ps in aligned] == [len(server_lines(r)) for r in WIDE_RECORDS]
    assert sum(len(ps) for _, ps in aligned) == len(lines)
    by_id = {r["id"]: ps for r, ps in aligned}
    assert len(by_id["w3-tournament"]) == 17 and len(by_id["w3-permute"]) == 4 and len(by_id["w3-single"]) == 1
    # w2: 3 + 7 groups, then both finals once round 1 is over
    rounds = [(p["parts"][0][0], p["round"]) for p in by_id["w2-tournament"]]
    assert rounds[-2:] == [("country", 2), ("language", 2)] and len(rounds) == 12
    qp = cg.question_passes(by_id["w2-tournament"])
    assert list(qp) == ["country", "language"] and [len(v) for v in qp.values()] == [4, 8]


@pytest.mark.parametrize("cut", ["drop_final", "drop_permute", "extra_group"])
def test_align_refuses_a_plan_with_the_wrong_number_of_passes(cut):
    rec = next(r for r in WIDE_RECORDS if r["id"] == ("w0-permute" if cut == "drop_permute" else "w0-tournament"))
    lines = server_lines(rec)
    if cut == "drop_final":
        lines = lines[:-1]
    elif cut == "drop_permute":
        lines = lines[:-1]
    else:
        lines = lines[:2] + [lines[2]] + lines[2:]
    with pytest.raises(SystemExit, match="MISMATCH"):
        list(cg.align_lines(lines, [rec]))


def test_align_refuses_a_final_before_its_groups_or_a_changed_strategy():
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w0-tournament")
    lines = server_lines(rec)  # team, product g0, product g1, product final
    early = [lines[0], lines[1], lines[3], lines[2]]
    with pytest.raises(SystemExit, match="round"):
        list(cg.align_lines(early, [rec]))
    changed = [dict(x) for x in lines]
    changed[2]["strategy"] = "permute"
    with pytest.raises(SystemExit, match="strategy"):
        list(cg.align_lines(changed, [rec]))


def test_align_refuses_a_plan_pass_with_several_parts():
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w0-single")
    lines = server_lines(rec)
    lines[1] = {**lines[1], "parts": [["product", 0, 17], ["team", 17, 23]]}
    with pytest.raises(SystemExit, match="MISMATCH"):
        list(cg.align_lines(lines[1:], [rec]))


# ---- codec_golden check_jev over wide dumps -----------------------------------------------------

def test_check_jev_accepts_every_strategy(tmp_path, capsys):
    lines = write_dump(tmp_path / "d.jsonl", WIDE_RECORDS)
    cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), WIDE_RECORDS)
    out = capsys.readouterr().out
    assert f"jev: {len(lines)} passes" in out


def test_check_jev_still_checks_the_fifty_native_records(tmp_path, capsys):
    recs = records()[:6]
    lines = []
    for r in recs:
        for qid, q in r["questions"].items():
            if q["type"] == "score":
                lines.append({"parts": [[qid, 0, 6]], "readout": {
                    "prompt_ids": TOK.encode(_prompt(r, qid, [str(i) for i in range(6)])),
                    "head": {"kind": "verbalizer", "token_ids": HEAD["verbalizer_ids"][2:8]}}})
            elif q["type"] == "noul":
                lines.append({"parts": [[qid, 0, 2]], "readout": {
                    "prompt_ids": TOK.encode(_prompt(r, qid, ["false", "true"])),
                    "head": {"kind": "verbalizer", "token_ids": [7, 8]}}})
            else:
                ln = _line(r, qid, list(range(len(q["criteria"]))), "native", 1)
                lines.append(ln)
    (tmp_path / "d.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), recs)
    assert f"jev: {len(lines)} passes" in capsys.readouterr().out


def _mutate(tmp_path, rid, fn, **kw):
    rec = next(r for r in WIDE_RECORDS if r["id"] == rid)
    lines = server_lines(rec, **kw)
    fn(lines)
    (tmp_path / "d.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return rec


def test_check_jev_refuses_a_permute_pass_in_the_wrong_order(tmp_path, capsys):
    def swap(lines):
        lines[1], lines[2] = lines[2], lines[1]
    rec = _mutate(tmp_path, "w0-permute", swap)
    with pytest.raises(SystemExit):
        cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), [rec])
    assert "MISMATCH jev w0-permute.product" in capsys.readouterr().out


def test_check_jev_refuses_wrong_extended_label_ids(tmp_path, capsys):
    def relabel(lines):
        lines[1]["readout"]["head"]["token_ids"][16] = 4242
    rec = _mutate(tmp_path, "w0-single", relabel)
    with pytest.raises(SystemExit):
        cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), [rec])
    assert "token_ids" in capsys.readouterr().out


def test_check_jev_refuses_a_final_without_every_group(tmp_path, capsys):
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w2-tournament")
    lines = server_lines(rec, finals=lambda n: list(range(16)))  # group 0 only for country (40)
    (tmp_path / "d.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(SystemExit):
        cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), [rec])
    assert "final" in capsys.readouterr().out


def test_check_jev_refuses_a_final_whose_prompt_is_not_the_layout(tmp_path, capsys):
    def corrupt(lines):
        lines[-1]["readout"]["prompt_ids"] = lines[-1]["readout"]["prompt_ids"][:-1]
    rec = _mutate(tmp_path, "w0-tournament", corrupt)
    with pytest.raises(SystemExit):
        cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), [rec])
    assert "MISMATCH" in capsys.readouterr().out


def test_check_jev_refuses_a_plan_other_than_the_requested_strategy(tmp_path, capsys):
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w0-single")
    lines = server_lines(rec, strategy="permute")
    (tmp_path / "d.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(SystemExit):
        cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), [rec])
    assert "strategy" in capsys.readouterr().out


def test_check_jev_refuses_a_plan_within_the_trained_slots(tmp_path, capsys):
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w0-single")
    lines = server_lines(rec)
    lines[0]["strategy"] = "single"  # team has 6 options: must be native
    (tmp_path / "d.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(SystemExit):
        cg.check_jev(str(tmp_path / "d.jsonl"), TOK, adapter(tmp_path), [rec])
    assert "native" in capsys.readouterr().out


def test_final_indices_reads_the_finalists_back():
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w3-tournament")
    fin = [o * 16 + o for o in range(16)]
    ln = _line(rec, "queue", fin, "tournament", 2)
    assert cg.final_indices(rec, "queue", ln["readout"]["prompt_ids"], TOK, LABELS) == fin


def test_jev_plan_probs_combines_by_strategy():
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w0-permute")
    q = rec["questions"]["product"]
    orders = cg.permute_orders(17, q["instructions"])
    dists = [[1.0 if i == 0 else 0.0 for i in range(17)] for _ in range(4)]
    p = cg.jev_plan_probs(rec, "product", "permute", dists)
    want = [0.0] * 17
    for o in orders:
        want[o[0]] += 0.25
    assert p == want
    assert cg.jev_plan_probs(rec, "product", "single", [list(range(17))]) == list(range(17))
    gs = cg.groups(17)
    ig = [[1 / len(g)] * len(g) for g in gs]
    fin = cg.tournament_final(gs, ig)
    final = [1 / 16] * 16
    assert cg.jev_plan_probs(rec, "product", "tournament", ig + [final], fin) == \
        cg.combine_tournament(gs, ig, fin, final)
    with pytest.raises(SystemExit, match="MISMATCH"):
        cg.jev_plan_probs(rec, "product", "permute", dists[:3])


# ---- the request body carries the record's strategy ---------------------------------------------

class _Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.body).encode()


@pytest.mark.parametrize("mod", [cg, pc])
def test_systemone_sends_the_records_strategy(mod, monkeypatch):
    sent = []

    def urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return _Reply({"answers": {}})
    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    mod.systemone("http://x", WIDE_RECORDS[1])
    mod.systemone("http://x", records()[0])
    assert sent[0]["strategy"] == WIDE_RECORDS[1]["strategy"] == "tournament"
    assert "strategy" not in sent[1] and set(sent[1]) == {"model", "state", "questions"}


# ---- parity_check --wide ------------------------------------------------------------------------

def _wide_server(monkeypatch, dump, posted):
    def systemone(url, rec):
        posted.append(rec["id"])
        with open(dump, "a") as f:
            for x in server_lines(rec):
                f.write(json.dumps(x) + "\n")
        answers = {}
        for qid, q in rec["questions"].items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.8}
            else:
                n = len(q["criteria"])
                answers[qid] = {"type": q["type"], "probabilities": {k: (0.5 if i == 0 else 0.5 / (n - 1))
                                                                  for i, k in enumerate(q["criteria"])}}
        return {"answers": answers}
    monkeypatch.setattr(pc, "systemone", systemone)


def test_parity_wide_slices_its_own_file(tmp_path, monkeypatch):
    dump = tmp_path / "jev9b.jsonl"
    dump.write_text("")
    posted = []
    _wide_server(monkeypatch, dump, posted)
    pc.main(["--url", "http://x", "--dump", str(dump), "--wide"])
    out = tmp_path / "jev9b.wide.parity.jsonl"
    assert posted == [r["id"] for r in WIDE_RECORDS]
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert not (tmp_path / "jev9b.parity.jsonl").exists(), "the 50-record slice is a different file"
    assert [len(ps) for _, ps in cg.align(str(out), WIDE_RECORDS)] == [len(server_lines(r)) for r in WIDE_RECORDS]


def test_parity_wide_gates_against_a_wide_reference(tmp_path, monkeypatch, capsys):
    _wide_server(monkeypatch, tmp_path / "d.jsonl", [])
    ref = {f"{r['id']}.{qid}": ([0.2, 0.8] if q["type"] == "noul" else
                                [0.5 if i == 0 else 0.5 / (len(q["criteria"]) - 1) for i in range(len(q["criteria"]))])
           for r in WIDE_RECORDS for qid, q in r["questions"].items()}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(tmp_path / "ref.json"), "--wide"])
    assert e.value.code == 0 and "PASS agreement 21/21" in capsys.readouterr().out
    # the 50-record gate refuses the wide reference: its entries are never answered
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(tmp_path / "ref.json")])
    assert e.value.code != 0


# ---- reference_stream: a question's passes combined by its plan ---------------------------------

def _jev_tree(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir(exist_ok=True)
    prof = {"family": "jev", "jev": {"verbalizers": HEAD["slots"]["verbalizers"], "verbalizer_ids": HEAD["verbalizer_ids"],
                                     "bias": HEAD["bias"], "slots": HEAD["slots"]["ranges"],
                                     "temperature": {"noul": 1.0, "choice": 2.0, "score": 0.5},
                                     "choice_labels": LABELS[:256], "choice_label_ids": [1000 + i for i in range(256)]}}
    (tree / "decision.json").write_text(json.dumps(prof))
    return tree, prof["jev"]


def test_reference_stream_combines_each_plan(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import reference_stream as rs
    tree, j = _jev_tree(tmp_path)
    hidden_size = 4
    gen = torch.Generator().manual_seed(0)
    table = {}

    def lm_rows(weights_dir, ids):
        for i in ids:
            if int(i) not in table:
                table[int(i)] = torch.randn(hidden_size, generator=gen)
        return torch.stack([table[int(i)] for i in ids])
    monkeypatch.setattr(rs, "lm_rows", lm_rows)
    recs = [r for r in WIDE_RECORDS if r["id"].startswith("w0-")]
    for rec in recs:
        lines = server_lines(rec)
        hs = [torch.randn(3, hidden_size, generator=gen) for _ in lines]
        probs = rs.family_probs("jev", rec, lines, hs, str(tree), "w", None, None, None, tok=TOK)
        assert set(probs) == {"team", "product"} and len(probs["product"]) == 17
        assert sum(probs["product"]) == pytest.approx(1.0)

        def dist(line, h):
            vids = line["readout"]["head"]["token_ids"]
            logits = lm_rows("w", vids) @ h[-1]
            z = [(float(l) + (j["bias"][8 + i] if i < 16 else 0.0)) / 2.0 for i, l in enumerate(logits)]
            return rs.softmax(z)
        wide = [(ln, h) for ln, h in zip(lines, hs) if ln["parts"][0][0] == "product"]
        dists = [dist(ln, h) for ln, h in wide]
        if rec["strategy"] == "single":
            want = dists[0]
        elif rec["strategy"] == "permute":
            want = [0.0] * 17
            for o, d in zip(cg.permute_orders(17, rec["questions"]["product"]["instructions"]), dists):
                for i, v in zip(o, d):
                    want[i] += v / 4
        else:
            want = cg.combine_tournament(cg.groups(17), dists[:-1], _finalists(17), dists[-1])
        assert probs["product"] == pytest.approx(want, abs=1e-12), rec["id"]


def test_reference_stream_tournament_needs_a_tokenizer(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import reference_stream as rs
    tree, _ = _jev_tree(tmp_path)
    monkeypatch.setattr(rs, "lm_rows", lambda w, ids: torch.ones(len(ids), 2))
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w0-tournament")
    lines = server_lines(rec)
    with pytest.raises(SystemExit, match="tokenizer"):
        rs.family_probs("jev", rec, lines, [torch.ones(2, 2) for _ in lines], str(tree), "w", None, None, None)


def test_reference_stream_wide_flag_uses_the_wide_records(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import reference_stream as rs
    seen = {}

    def build_ref(family, weights, dump, recs, tree, lora, scale, clef_dir, threads=4, tok=None):
        seen.update(recs=recs, tok=tok)
        return {"x": [1.0]}
    monkeypatch.setattr(rs, "build_ref", build_ref)
    monkeypatch.setattr(rs, "tree_tokenizer", lambda tree: "TOK")
    tree, _ = _jev_tree(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rs", "jev", "--weights", "w", "--tree", str(tree), "--dump", "d", "--out",
                                      str(tmp_path / "ref.json"), "--wide"])
    rs.main()
    assert seen["recs"] == WIDE_RECORDS and seen["tok"] == "TOK"


# ---- the real JEV tokenizer (skipped when the spike files are absent) ---------------------------

SPIKE = Path(os.environ.get("SPIKE_DIR", "/Users/blokchainaholic/Desktop/clabs/tahoma/experiments"))


def test_real_jev_tokenizer_labels_extend_the_trained_head_and_finals_parse():
    tj = SPIKE / "jev-spike" / "jev9b" / "tokenizer.json"
    hj = SPIKE / "jev-spike" / "jev9b" / "adapter_vllm" / "decision_head.json"
    if not (tj.exists() and hj.exists()):
        pytest.skip("jev9b tokenizer/head absent")
    pytest.importorskip("tokenizers")
    tok = cg.TokenizerFile(str(tj))
    head = json.loads(hj.read_text())
    labels, ids = cg.jev_choice_labels(tok, head)
    assert len(labels) == 256 and labels[:16] == list("ABCDEFGHIJKLMNOP")
    _, added = cg.tokenizer_added(tok)
    rec = next(r for r in WIDE_RECORDS if r["id"] == "w2-tournament")
    for qid in ("country", "language"):
        n = len(rec["questions"][qid]["criteria"])
        fin = [g[-1] for g in cg.groups(n)]
        fin = sorted(fin + [o for o in range(n) if o not in fin][:16 - len(fin)])
        p = tok(cg.jev_pass_text(rec, qid, fin, labels, added), add_special_tokens=False).input_ids
        assert cg.final_indices(rec, qid, p, tok, labels, added) == fin
