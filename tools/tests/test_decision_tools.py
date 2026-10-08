"""Decision parity tools: answer flattening, margins, dump alignment.

    python -m pytest tools/tests/test_decision_tools.py -v
"""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "decision"))
import codec_golden as cg  # noqa: E402
import parity_check as pc  # noqa: E402
from records import records  # noqa: E402


def test_records_are_the_fifty_spike_records():
    rs = records()
    assert len(rs) == 50 and sum(len(r["questions"]) for r in rs) == 82


def test_answer_probs_flattens_every_type():
    assert pc.answer_probs({"type": "noul", "noul": 0.75}) == [0.25, 0.75]
    assert pc.answer_probs({"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}}) == [0.1, 0.9]
    assert pc.answer_probs({"type": "score", "score": 1.2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}) == [0.2, 0.4, 0.4]


def test_margin_is_top1_minus_top2():
    assert abs(pc.margin([0.501, 0.499]) - 0.002) < 1e-12
    assert pc.margin([1.0]) == 1.0


def test_gate_tolerates_coin_flips_but_not_confident_flips():
    assert pc.gate([0.002, 0.01, 0.04], 79, 82, 0.96, 0.05)[0], "3 undecided flips: pass"
    assert not pc.gate([0.002, 0.01, 0.02, 0.03], 78, 82, 0.96, 0.05)[0], "4 flips: under 96%"
    ok, why = pc.gate([0.06], 81, 82, 0.96, 0.05)
    assert not ok and "confident" in why, "one flip the reference was sure about fails"


def test_synthetic_records_cover_the_risky_branches():
    kinds = {k for r in cg.SYNTHETIC for k in cg.branches(r)}
    assert {"list_state", "noul_criteria", "dict_description", "empty_instructions", "null_instructions", "float",
            "nested_list_of_dicts", "float:1e-05", "float:1e+16", "float:2.5", "special_token_text"} <= kinds


def test_records_convert_to_openai_requests():
    import openai_equivalence as oe
    r = records()[0]
    o = oe.to_openai(r)
    assert [q["name"] for q in o["questions"]] == list(r["questions"])
    assert o["input"] == (r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False))
    kinds = {q["type"] for rr in records() for q in oe.to_openai(rr)["questions"]}
    assert kinds == {"predicate", "choice", "score"}


def test_dump_passes_align_to_records(tmp_path):
    # JEV/Kev: one pass per question, in record then question order
    lines = []
    for r in records()[:2]:
        for qid in r["questions"]:
            lines.append({"parts": [[qid, 0, 2]], "readout": {"prompt_ids": [1, 2], "head": {"kind": "verbalizer", "token_ids": [1, 2]}}})
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    aligned = list(cg.align(str(p), records()[:2]))
    assert [len(passes) for _, passes in aligned] == [len(r["questions"]) for r in records()[:2]]
    assert aligned[0][0]["id"] == records()[0]["id"]


# ---- BLD-1: backbone_diff matches across checkpoint layouts and always ends with a verdict ---------

_LAYER_SUFFIXES = ["self_attn.q_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight",
                   "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
                   "post_attention_layernorm.weight"]


def _hdr(prefix, layers=3, changed=(), dtype="BF16", extra=(), drop=()):
    """A synthetic safetensors header: canonical name -> entry; the entry's url carries a content tag
    so the fake sha sees a difference exactly where `changed` says."""
    out = {}
    names = [f"layers.{i}.{s}" for i in range(layers) for s in _LAYER_SUFFIXES] + ["embed_tokens.weight", "norm.weight"]
    for n in names:
        if n in drop:
            continue
        out[prefix + n] = (("ft:" if n in changed else "stock:") + n, dtype, (4, 4), 0, 32)
    out["lm_head.weight"] = (("ft:" if "lm_head.weight" in changed else "stock:") + "lm_head", dtype, (8, 4), 0, 64)
    for n in extra:
        out[n] = ("x:" + n, dtype, (2,), 0, 4)
    return out


def _fake_sha(entry):
    return entry[0]


def _run(a, b, **kw):
    import backbone_diff as bd
    lines = []
    v, code = bd.compare(a, b, sha=_fake_sha, log=lines.append, **kw)
    assert lines[-1].startswith(f"VERDICT: {v}")
    return v, code, lines


def test_backbone_diff_matches_text_only_and_vlm_layouts():
    # JEV-9B style (Qwen3_5ForCausalLM) vs a Qwen VLM repo with a vision tower and MTP head
    a = _hdr("model.")
    b = _hdr("model.language_model.", extra=("model.visual.blocks.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight"))
    v, code, lines = _run(a, b, per_layer=True)
    assert (v, code) == ("STOCK", 0)
    assert "common 24" in lines[0]  # 3 layers x 7 + embed + norm + lm_head; vision and MTP ignored
    assert [l for l in lines if l.startswith("layer ")] == [f"layer {i}: SAME (7 tensors)" for i in range(3)]


def test_backbone_diff_sees_a_change_outside_gate_proj():
    for changed in ("layers.1.self_attn.o_proj.weight", "layers.2.input_layernorm.weight", "embed_tokens.weight",
                    "lm_head.weight", "norm.weight"):
        v, code, lines = _run(_hdr("model."), _hdr("model.language_model.", changed={changed}), per_layer=True)
        assert (v, code) == ("FINETUNED", 0), changed
    v, _, lines = _run(_hdr("model."), _hdr("model.", changed={"layers.1.self_attn.o_proj.weight"}), per_layer=True)
    assert "layer 1: DIFF (7 tensors, differing: self_attn.o_proj.weight)" in lines


def test_backbone_diff_is_unknown_when_nothing_is_comparable():
    only_lm_head = {"lm_head.weight": ("stock:lm_head", "BF16", (8, 4), 0, 64)}
    assert _run(_hdr("model."), only_lm_head)[:2] == ("UNKNOWN", 2)
    assert _run(_hdr("model.", layers=3), _hdr("model.", layers=4))[:2] == ("UNKNOWN", 2)
    assert _run(_hdr("model."), _hdr("model.", dtype="F32"))[:2] == ("UNKNOWN", 2)
    assert _run(_hdr("model."), _hdr("model.", drop={"layers.0.mlp.up_proj.weight"}))[:2] == ("UNKNOWN", 2)


def test_backbone_diff_main_exits_nonzero_on_unknown(monkeypatch, capsys):
    import backbone_diff as bd
    hdrs = {"a": _hdr("model."), "b": {"lm_head.weight": ("stock:lm_head", "BF16", (8, 4), 0, 64)}}
    monkeypatch.setattr(bd, "headers", hdrs.__getitem__)
    with pytest.raises(SystemExit) as e:
        bd.main(["a", "b"])
    assert e.value.code == 2
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("VERDICT: UNKNOWN")


def test_backbone_diff_tensor_sha_hashes_both_ends(monkeypatch):
    import backbone_diff as bd
    blob = bytes(range(256)) * 64  # 16 KiB
    monkeypatch.setattr(bd, "get", lambda url, rng: blob[rng[0]:rng[1] + 1])
    base = bd.tensor_sha(("u", "BF16", (1,), 0, len(blob)), cap=1024)
    tail_changed = blob[:-1] + b"\x00"
    monkeypatch.setattr(bd, "get", lambda url, rng: tail_changed[rng[0]:rng[1] + 1])
    assert bd.tensor_sha(("u", "BF16", (1,), 0, len(blob)), cap=1024) != base


# ---- Fix round 1: every partial-coverage path fails loudly -------------------------------------

def _server_answers(rec):
    """A fake /v1/systemone response whose top-1 is always the first option."""
    out = {}
    for qid, q in rec["questions"].items():
        if q["type"] == "noul":
            out[qid] = {"type": "noul", "noul": 0.2}
        elif q["type"] == "choice":
            ks = list(q["criteria"])
            out[qid] = {"type": "choice", "choice": ks[0], "probabilities": {k: (0.9 if i == 0 else 0.01) for i, k in enumerate(ks)}}
        else:
            out[qid] = {"type": "score", "score": 0.1, "probabilities": {str(i): (0.9 if i == 0 else 0.01) for i in range(len(q["criteria"]))}}
    return {"answers": out}


def _kev_ref(recs):
    # Kev order: noul [false, true], caller-order choices, levels by index -> index 0 matches the fake server
    return {f"{r['id']}.{qid}": {"probs": [0.9] + [0.01] * (len(q.get("criteria") or [0, 0]) - 1)}
            for r in recs for qid, q in r["questions"].items()}


def test_score_answers_covers_every_question():
    recs = records()
    resps = [_server_answers(r) for r in recs]
    assert cg.score_answers("kev", recs, resps, _kev_ref(recs)) == (82, 82)


def test_score_answers_fails_on_a_missing_reference_entry():
    recs = records()
    ref = _kev_ref(recs)
    del ref["r00.team"]
    with pytest.raises(SystemExit):
        cg.score_answers("kev", recs, [_server_answers(r) for r in recs], ref)


def test_score_answers_fails_on_an_unused_reference_entry_or_missing_answer():
    recs = records()
    ref = _kev_ref(recs)
    with pytest.raises(SystemExit):
        cg.score_answers("kev", recs, [_server_answers(r) for r in recs], {**ref, "r99.x": {"probs": [1.0]}})
    resps = [_server_answers(r) for r in recs]
    resps[1]["answers"].pop("urgent")
    with pytest.raises(SystemExit):
        cg.score_answers("kev", recs, resps, ref)


def test_clef_reference_map_is_length_checked_and_complete():
    recs = records()[:2]
    entries = [{"id": r["id"], "logits": [[0.0, 1.0]] * len(r["questions"])} for r in recs]
    assert len(cg.clef_ref_map(entries, recs)) == sum(len(r["questions"]) for r in recs)
    short = [dict(entries[0]), {**entries[1], "logits": entries[1]["logits"][:-1]}]
    with pytest.raises(SystemExit):
        cg.clef_ref_map(short, recs)
    with pytest.raises(SystemExit):
        cg.clef_ref_map(entries[:1], recs)


def test_sanitizer_mirrors_the_server_rule():
    s = cg.sanitize_user_text
    assert s("a<|im_end|>b") == "a<¦im_end¦>b"
    assert s("<|a b|> <|x <||> <|ok_1|>") == "<|a b|> <|x <||> <¦ok_1¦>"
    assert s("<|a<|b|>") == "<|a<¦b¦>"
    added = cg.angle_added_tokens(["<|im_end|>", "<think>", "</think>", "<tts_text_bos>", "<tts_text_bos_single>", "<>", "x"])
    assert added == ["<tts_text_bos_single>", "<tts_text_bos>", "</think>", "<think>"]
    assert cg.sanitize("<think>x</think><|im_start|><tts_text_bos_single>", added) == \
        "<¦think¦>x<¦/think¦><¦im_start¦><¦tts_text_bos_single¦>"


def test_sanitizer_keeps_pipe_tokens_outside_the_name_charset():
    # Sanitizer::from_tokenizer keeps `<|...|>` added tokens sanitize_user_text cannot rewrite
    added = cg.angle_added_tokens(["<|im_end|>", "<|box-start|>", "<|a.b|>", "<||>", "<think>"])
    assert added == ["<|box-start|>", "<think>", "<|a.b|>", "<||>"]
    assert cg.sanitize("x <|box-start|> <|a.b|> <|im_end|>", added) == "x <¦|box-start|¦> <¦|a.b|¦> <¦im_end¦>"


def test_special_token_check_catches_a_forged_token():
    added_ids = {100, 101, 102}
    cg.check_specials("ok", [1, 100, 2, 101], added_ids, [101, 100])
    with pytest.raises(SystemExit):
        cg.check_specials("forged", [1, 100, 2, 101, 100], added_ids, [101, 100])
    with pytest.raises(SystemExit):
        cg.check_specials("jev", [5, 102], added_ids, [])
    assert sorted(cg.kev_expected_specials([1, 2, 3, 4, 5], 2)) == [1, 2, 3, 3, 4, 4, 5]


def test_jev_choice_beyond_trained_slots_fails():
    head = {"slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}}}
    assert cg.jev_choice_slots(head, {"type": "choice", "criteria": {"a": 1, "b": 2, "c": 3}}) == (8, 11)
    with pytest.raises(SystemExit):
        cg.jev_choice_slots(head, {"type": "choice", "criteria": {f"o{i}": "" for i in range(17)}})


def test_golden_copy_is_private_and_never_silently_overwritten(tmp_path):
    p = tmp_path / "d.golden.jsonl"
    cg.write_private(p, "a\n")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and p.read_text() == "a\n"
    with pytest.raises(SystemExit):
        cg.write_private(p, "b\n")
    cg.write_private(p, "b\n", force=True)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and p.read_text() == "b\n"


def test_parity_tally_requires_every_reference_entry():
    recs = records()
    ref = {f"{r['id']}.{qid}": pc.answer_probs(a) for r in recs for qid, a in _server_answers(r)["answers"].items()}
    results = [(r, _server_answers(r)) for r in recs]
    _, agree, total, flips = pc.tally(ref, results)
    assert (agree, total, flips) == (82, 82, [])
    broken = [(r, _server_answers(r)) for r in recs]
    broken[1][1]["answers"].pop("urgent")
    with pytest.raises(SystemExit):
        pc.tally(ref, broken)
    with pytest.raises(SystemExit):
        pc.tally({**ref, "r99.x": [1.0, 0.0]}, results)


def test_openai_equivalence_checks_names_and_labels_not_just_positions():
    import openai_equivalence as oe
    rec = {"id": "x", "state": "s", "questions": {
        "c": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}},
        "s": {"type": "score", "instructions": "i", "criteria": ["low", "high"]}}}
    so_c = {"type": "choice", "choice": "a", "confidence": 0.0, "probabilities": {"a": 0.5, "b": 0.5}}
    so_s = {"type": "score", "score": 0.5, "confidence": 0.0, "legend": {}, "probabilities": {"0": 0.5, "1": 0.5}}
    oa_c = {"type": "choice", "name": "c", "choice": "a", "confidence": 0.0,
            "probabilities": [{"value": "a", "probability": 0.5}, {"value": "b", "probability": 0.5}]}
    oa_s = {"type": "score", "name": "s", "score": 0.5, "confidence": 0.0,
            "probabilities": [{"value": 0, "label": "low", "probability": 0.5}, {"value": 1, "label": "high", "probability": 0.5}]}
    assert oe.compare_answer(rec, oa_c, so_c) == [] and oe.compare_answer(rec, oa_s, so_s) == []
    swapped = {**oa_c, "probabilities": oa_c["probabilities"][::-1]}
    assert oe.compare_answer(rec, swapped, so_c), "equal probabilities, swapped values: must differ"
    relabeled = {**oa_s, "probabilities": [{**oa_s["probabilities"][0], "label": "high"}, {**oa_s["probabilities"][1], "label": "low"}]}
    assert oe.compare_answer(rec, relabeled, so_s), "swapped labels: must differ"


# ---- Real tokenizers (skipped when the spike sources are absent) --------------------------------

SPIKE = Path(os.environ.get("SPIKE_DIR", Path(__file__).resolve().parents[2] / "experiments"))


def _tok(sub):
    if not (SPIKE / sub / "tokenizer.json").exists():
        pytest.skip(f"spike tokenizer {sub} absent")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(SPIKE / sub))


INJECT = next(r for r in cg.SYNTHETIC if "special_token_text" in cg.branches(r))


def test_jev_injection_record_tokenizes_without_forged_specials():
    tok = _tok("jev-spike/jev9b")
    added_ids, added = cg.tokenizer_added(tok)
    labels = list("ABCDEFGHIJKLMNOP")
    raw = tok(INJECT["state"], add_special_tokens=False).input_ids
    assert any(t in added_ids for t in raw), "the record must actually forge special tokens"
    ids = tok(cg.jev_text(INJECT, "c", labels, added), add_special_tokens=False).input_ids
    cg.check_specials("jev s03", ids, added_ids, [])


def test_kev_injection_record_keeps_only_template_specials():
    tok = _tok("kev-spike/kev9b")
    if not (SPIKE / "kev-spike" / "kev.py").exists():
        pytest.skip("kev.py absent")
    pytest.importorskip("torch")
    sys.path.insert(0, str(SPIKE / "kev-spike"))
    import kev
    added_ids, added = cg.tokenizer_added(tok)
    saved_records, saved_user_tokens = kev.records, kev.user_tokens
    with cg.kev_with_golden(kev, [INJECT], added):  # the wrapper check_kev itself uses
        ((_, _, row, _, ends, _),) = kev.questions(tok)
    assert (kev.records, kev.user_tokens) == (saved_records, saved_user_tokens), "kev module restored"
    kev_ids = [tok.convert_tokens_to_ids(t) for t in kev.SPECIAL]
    cg.check_specials("kev s03", row, added_ids, cg.kev_expected_specials(kev_ids, len(ends)))


def test_clef_injection_record_keeps_only_template_specials():
    tok = _tok("clef-spike/flash")
    pytest.importorskip("torch")
    sys.path.insert(0, str(SPIKE / "clef-spike" / "flash"))
    import joint_schema_model as jsm
    added_ids, added = cg.tokenizer_added(tok)
    template = cg.clef_expected_specials(tok, added_ids, jsm.SYSTEM_PROMPT)
    with pytest.raises(SystemExit):  # unsanitized, the forged tokens get through
        cg.check_specials("clef raw", jsm.encode_record(tok, INJECT).input_ids, added_ids, template)
    cg.check_specials("clef s03", jsm.encode_record(tok, cg.clef_sanitized(INJECT, added)).input_ids, added_ids, template)


# ---- DRIFT-1 / BLD-2: a no-ref parity run writes its own slice of the dump ------------------------

def _pass_line(qid):
    return json.dumps({"parts": [[qid, 0, 2]], "readout": {"prompt_ids": [1], "head": {"kind": "verbalizer", "token_ids": [1, 2]}}})


def _fake_server(monkeypatch, dump, posted, extra_after=None):
    """pc.systemone replaced by a server that appends one pass per question to `dump`."""
    def systemone(url, rec):
        posted.append(rec["id"])
        with open(dump, "a") as f:
            for qid in rec["questions"]:
                f.write(_pass_line(qid) + "\n")
            if rec["id"] == extra_after:
                f.write(_pass_line("intruder") + "\n")
        return {"answers": {}}
    monkeypatch.setattr(pc, "systemone", systemone)


def test_parity_run_slices_only_its_own_passes(tmp_path, monkeypatch):
    dump = tmp_path / "jev9b.jsonl"
    dump.write_text(_pass_line("earlier") + "\n" + _pass_line("golden") + "\n")  # earlier runs' passes
    posted = []
    _fake_server(monkeypatch, dump, posted)
    pc.main(["--url", "http://x", "--dump", str(dump)])
    out = tmp_path / "jev9b.parity.jsonl"
    assert len(posted) == 50
    assert len(out.read_text().splitlines()) == 82
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert [len(ps) for _, ps in cg.align(str(out), records())] == [len(r["questions"]) for r in records()]
    # passes appended after the run (golden, duplicate request) never reach the slice
    with open(dump, "a") as f:
        f.write(_pass_line("later") + "\n")
    assert len(out.read_text().splitlines()) == 82


def test_parity_run_refuses_to_overwrite_its_slice_without_force(tmp_path, monkeypatch):
    dump = tmp_path / "d.jsonl"
    dump.write_text("")
    (tmp_path / "d.parity.jsonl").write_text("old")
    posted = []
    _fake_server(monkeypatch, dump, posted)
    with pytest.raises(SystemExit, match="--force"):
        pc.main(["--url", "http://x", "--dump", str(dump)])
    assert posted == [], "refused before posting anything"
    pc.main(["--url", "http://x", "--dump", str(dump), "--force"])
    assert len((tmp_path / "d.parity.jsonl").read_text().splitlines()) == 82


def test_parity_run_fails_when_another_client_hits_the_server(tmp_path, monkeypatch):
    dump = tmp_path / "d.jsonl"
    dump.write_text("")
    _fake_server(monkeypatch, dump, [], extra_after="r10")
    with pytest.raises(SystemExit, match="MISMATCH"):
        pc.main(["--url", "http://x", "--dump", str(dump)])


def test_parity_run_without_ref_needs_dump(capsys):
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x"])
    assert e.value.code == 2 and "--dump" in capsys.readouterr().err


# ---- QUA-11: check_kev's kev patch is one context manager that restores the module ----------------

def test_kev_with_golden_patches_and_restores():
    import types
    seen = []
    kev = types.SimpleNamespace(records=lambda: ["orig"], user_tokens=lambda t, text: seen.append(text) or text)
    orig = (kev.records, kev.user_tokens)
    with cg.kev_with_golden(kev, ["golden"], ["<think>"]):
        assert kev.records() == ["golden"]
        kev.user_tokens(None, "a<think>b")
    assert seen == ["a<¦think¦>b"], "angle added tokens escaped before kev's own pipe escaping"
    assert (kev.records, kev.user_tokens) == orig
    with pytest.raises(RuntimeError):
        with cg.kev_with_golden(kev, ["golden"], []):
            raise RuntimeError
    assert (kev.records, kev.user_tokens) == orig, "restored on error too"


# ---- FID-3: JEV choice answers are gated, on jev.py's own (sorted) prompts --------------------------

def _jev_ref(recs):
    """reference_jev.pt-shaped entries: noul [false, true], choices sorted, 6 score levels; top-1 index 0."""
    out = {}
    for r in recs:
        for qid, q in r["questions"].items():
            n = 2 if q["type"] == "noul" else len(q["criteria"]) if q["type"] == "choice" else 6
            out[f"{r['id']}.{qid}"] = {"probs": [0.9] + [0.01] * (n - 1)}
    return out


def test_jev_answers_gate_choices_sent_sorted():
    recs = [cg.jev_sorted(r) for r in records()]
    n_choice = sum(q["type"] == "choice" for r in recs for q in r["questions"].values())
    assert n_choice == 33 and all(list(q["criteria"]) == sorted(q["criteria"])
                                  for r in recs for q in r["questions"].values() if q["type"] == "choice")
    resps = [_server_answers(r) for r in recs]  # top-1 = first option = first sorted name = reference index 0
    assert cg.score_answers("jev", recs, resps, _jev_ref(recs)) == (82, 82)
    # a choice flip is now counted against the gate, not printed and skipped
    rid, qid = next((r["id"], qid) for r in recs for qid, q in r["questions"].items() if q["type"] == "choice")
    flipped = [_server_answers(r) for r in recs]
    i = [r["id"] for r in recs].index(rid)
    flipped[i]["answers"][qid]["choice"] = list(recs[i]["questions"][qid]["criteria"])[1]
    assert cg.score_answers("jev", recs, flipped, _jev_ref(recs)) == (81, 82)


def test_jev_answers_refuse_unsorted_choices():
    recs = records()
    assert any(list(q["criteria"]) != sorted(q["criteria"]) for r in recs for q in r["questions"].values()
               if q["type"] == "choice"), "the records do have unsorted choices"
    with pytest.raises(SystemExit):
        cg.score_answers("jev", recs, [_server_answers(r) for r in recs], _jev_ref(recs))


def test_jev_answer_run_prompts_must_equal_jev_py():
    recs = [cg.jev_sorted(r) for r in records()[:3]]
    ref_q, lines = [], []
    for r in recs:
        for qid in r["questions"]:
            ids = [len(lines), 7, 9]
            ref_q.append((f"{r['id']}.{qid}", r["questions"][qid]["type"], ids, [1, 2], [0.0, 0.0], ["x", "y"]))
            lines.append({"parts": [[qid, 0, 2]], "readout": {"prompt_ids": list(ids), "head": {"kind": "verbalizer", "token_ids": [1, 2]}}})
    assert cg.check_jev_reference_prompts(lines, recs, ref_q) == len(lines)
    bad = json.loads(json.dumps(lines))
    bad[2]["readout"]["prompt_ids"][1] = 8
    with pytest.raises(SystemExit):
        cg.check_jev_reference_prompts(bad, recs, ref_q)
    bad = json.loads(json.dumps(lines))
    bad[0]["readout"]["head"]["token_ids"] = [2, 1]
    with pytest.raises(SystemExit):
        cg.check_jev_reference_prompts(bad, recs, ref_q)
    with pytest.raises(SystemExit):  # jev.py has a question the server never answered
        cg.check_jev_reference_prompts(lines, recs, ref_q + [("r99.z", "noul", [1], [1, 2], [0, 0], ["a", "b"])])


def test_jev_sorted_prompt_equals_jev_py_text():
    """jev_text over a jev_sorted record is jev.py's own text for the 50 records (string level)."""
    jev_py = SPIKE / "jev-spike" / "jev.py"
    if not jev_py.exists():
        pytest.skip("jev.py absent")
    labels = list("ABCDEFGHIJKLMNOP")
    for r in records():
        rs_ = cg.jev_sorted(r)
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        for qid, q in r["questions"].items():
            kind = q["type"]
            if kind == "choice":
                names = sorted(q["criteria"])
                lines = [f"{l}) {n}: {q['criteria'][n]}" for l, n in zip(labels, names)]
            else:
                lines = ["false", "true"] if kind == "noul" else [str(i) for i in range(6)]
            want = (f"[kind] {kind}\n[state] {state}\n[question] {q['instructions']}\n[options]\n"
                    + "\n".join(lines) + "\n[decision]:")  # jev.py questions(), verbatim
            assert cg.jev_text(rs_, qid, labels) == want, f"{r['id']}.{qid}"


def test_jev_sorted_server_layout_tokenizes_to_jev_py_questions():
    """With the real JEV tokenizer: the server-codec mirror (jev_text, sanitizer included) over
    jev_sorted records gives jev.py questions()' exact prompt ids for all 82 questions."""
    tok = _tok("jev-spike/jev9b")
    if not (SPIKE / "jev-spike" / "jev9b" / "adapter_vllm" / "decision_head.json").exists():
        pytest.skip("jev decision_head.json absent")
    sys.path.insert(0, str(SPIKE / "jev-spike"))
    import jev
    _, added = cg.tokenizer_added(tok)
    labels = [l for l, _ in jev.choice_label_ids(tok, 16)]
    want = {key: list(ids) for key, _, ids, _, _, _ in jev.questions(tok)}
    got = {f"{r['id']}.{qid}": tok(cg.jev_text(cg.jev_sorted(r), qid, labels, added), add_special_tokens=False).input_ids
           for r in records() for qid in r["questions"]}
    assert got == want


# ---- Fix round 2: backbone_diff (P2-BLD-1, P2-BLD-2, P2-BLD-4) -----------------------------------

DUMMY_TOKEN = "hf_dummy_test_value_not_a_secret"


@pytest.mark.parametrize("name", ["lm_head.weight", "embed_tokens.weight", "norm.weight"])
def test_backbone_diff_top_level_tensor_on_one_side_is_unknown(name):
    full = _hdr("model.")
    trimmed = {k: v for k, v in _hdr("model.language_model.").items()
               if k not in (name, "model.language_model." + name)}
    for a, b in ((full, trimmed), (trimmed, full)):
        v, code, lines = _run(a, b)
        assert (v, code) == ("UNKNOWN", 2), name
        assert name in lines[-1]


_TARGETS = {f"layers.{i}.{s}" for i in range(3) for s in ("self_attn.q_proj.weight", "mlp.down_proj.weight")} \
    | {"lm_head.weight"}


def test_backbone_diff_merged_when_every_difference_is_an_adapter_target():
    changed = {"layers.0.self_attn.q_proj.weight", "layers.2.mlp.down_proj.weight", "lm_head.weight"}
    seen = []

    def proved(names, a, b):
        seen.append(sorted(names))
        return "MERGED", "checked: do not graft --lora again"
    v, code, lines = _run(_hdr("model."), _hdr("model.", changed=changed), adapter=_TARGETS.__contains__,
                          merge_check=proved)
    assert (v, code) == ("MERGED", 3)
    assert "--lora" in lines[-1]
    assert seen == [sorted(changed)]
    # P3-BLD-5: the all-on-target pattern alone is not a proof
    v, code, lines = _run(_hdr("model."), _hdr("model.", changed=changed), adapter=_TARGETS.__contains__)
    assert (v, code) == ("UNKNOWN", 2)
    for verdict, code in (("FINETUNED", 0), ("UNKNOWN", 2)):
        got = _run(_hdr("model."), _hdr("model.", changed=changed), adapter=_TARGETS.__contains__,
                   merge_check=lambda n, a, b, v=verdict: (v, "check said so"))[:2]
        assert got == (verdict, code)


def test_backbone_diff_finetuned_when_a_difference_is_off_target():
    changed = {"layers.0.self_attn.q_proj.weight", "layers.1.self_attn.o_proj.weight"}
    v, code, lines = _run(_hdr("model."), _hdr("model.", changed=changed), adapter=_TARGETS.__contains__)
    assert (v, code) == ("FINETUNED", 0)
    assert any("outside the adapter" in l and "layers.1.self_attn.o_proj.weight" in l for l in lines)
    # without an adapter the verdict is unchanged
    assert _run(_hdr("model."), _hdr("model.", changed={"layers.0.self_attn.q_proj.weight"}))[:2] == ("FINETUNED", 0)


def _save_st(path, names, value=0.0):
    import numpy as np
    from safetensors.numpy import save_file
    save_file({n: np.full((4, 4), value if not callable(value) else value(n), dtype=np.float32) for n in names}, str(path))


def test_backbone_diff_adapter_targets_from_adapter_file_and_config(tmp_path):
    import backbone_diff as bd
    ad = tmp_path / "adapter_vllm"
    ad.mkdir()
    _save_st(ad / "adapter_model.safetensors", [
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.lm_head.lora_A.weight", "base_model.model.lm_head.lora_B.weight"])
    on = bd.adapter_targets(str(ad))
    assert on("layers.0.self_attn.q_proj.weight") and on("lm_head.weight")
    assert not on("layers.1.self_attn.q_proj.weight") and not on("layers.0.self_attn.q_proj.bias")
    assert on is not None and bd.adapter_targets(str(ad / "adapter_model.safetensors"))("lm_head.weight")
    only_cfg = tmp_path / "cfg"
    only_cfg.mkdir()
    (only_cfg / "adapter_config.json").write_text(json.dumps({"target_modules": ["q_proj", "down_proj"]}))
    on = bd.adapter_targets(str(only_cfg))
    assert on("layers.7.self_attn.q_proj.weight") and on("layers.0.mlp.down_proj.weight")
    assert not on("layers.0.mlp.up_proj.weight") and not on("lm_head.weight")
    (only_cfg / "adapter_config.json").write_text(json.dumps({"target_modules": r".*\.(q_proj|lm_head)"}))
    on = bd.adapter_targets(str(only_cfg))
    assert on("layers.0.self_attn.q_proj.weight") and not on("layers.0.mlp.up_proj.weight")


def _local_repo(root, prefix="model.", changed=(), index=True):
    root.mkdir()
    names = [f"{prefix}layers.{i}.{s}" for i in range(2) for s in _LAYER_SUFFIXES] + \
        [f"{prefix}embed_tokens.weight", f"{prefix}norm.weight", "lm_head.weight"]
    if index:
        half = len(names) // 2
        _save_st(root / "model-00001-of-00002.safetensors", names[:half], lambda n: 1.0 + (n in changed))
        _save_st(root / "model-00002-of-00002.safetensors", names[half:], lambda n: 1.0 + (n in changed))
        wm = {n: ("model-00001-of-00002.safetensors" if k < half else "model-00002-of-00002.safetensors")
              for k, n in enumerate(names)}
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    else:
        _save_st(root / "model.safetensors", names, lambda n: 1.0 + (n in changed))
    return root


def test_backbone_diff_reads_local_snapshot_dirs(tmp_path, monkeypatch, capsys):
    import backbone_diff as bd
    monkeypatch.setattr(bd, "get", lambda *a, **k: pytest.fail("a local dir must not touch the network"))
    a = _local_repo(tmp_path / "a")
    b = _local_repo(tmp_path / "b", prefix="model.language_model.", index=False)
    with pytest.raises(SystemExit) as e:
        bd.main([str(a), str(b), "--slice", "16"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("VERDICT: STOCK")
    c = _local_repo(tmp_path / "c", changed={"model.layers.1.mlp.up_proj.weight"})
    with pytest.raises(SystemExit) as e:
        bd.main([str(a), str(c), "--slice", "16"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("VERDICT: FINETUNED")


def test_backbone_diff_config_only_adapter_cannot_prove_merged(tmp_path, monkeypatch, capsys):
    """P3-BLD-5: with only adapter_config.json there are no lora tensors to verify a merge: UNKNOWN."""
    import backbone_diff as bd
    ad = tmp_path / "cfg"
    ad.mkdir()
    (ad / "adapter_config.json").write_text(json.dumps({"target_modules": ["q_proj"]}))
    a = _local_repo(tmp_path / "a")
    b = _local_repo(tmp_path / "b", changed={"model.layers.0.self_attn.q_proj.weight"})
    with pytest.raises(SystemExit) as e:
        bd.main([str(a), str(b), "--adapter", str(ad)])
    assert e.value.code == 2
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("VERDICT: UNKNOWN") and "adapter_model.safetensors" in last


class _Resp:
    def __init__(self, status, body):
        self.status, self.body = status, body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_backbone_diff_sends_hf_token_only_as_an_unredirected_header(monkeypatch):
    import backbone_diff as bd
    seen = []
    monkeypatch.setattr(bd, "_open", lambda req: seen.append(req) or _Resp(206, b"abcd"))
    monkeypatch.setenv("HF_TOKEN", DUMMY_TOKEN)
    assert bd.get("https://huggingface.co/x/resolve/main/f", (0, 3)) == b"abcd"
    req = seen[0]
    assert req.unredirected_hdrs.get("Authorization") == f"Bearer {DUMMY_TOKEN}"
    assert "Authorization" not in req.headers  # never copied onto a redirect by urllib itself
    monkeypatch.delenv("HF_TOKEN")
    bd.get("https://huggingface.co/x/resolve/main/f", (0, 3))
    assert "Authorization" not in seen[1].unredirected_hdrs


def test_backbone_diff_redirect_keeps_token_on_hf_only(monkeypatch):
    import urllib.request
    import backbone_diff as bd
    req = urllib.request.Request("https://huggingface.co/x/resolve/main/f")
    req.add_header("Range", "bytes=0-3")
    req.add_unredirected_header("Authorization", f"Bearer {DUMMY_TOKEN}")
    h = bd._Redirect()
    same = h.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/y/resolve/main/f")
    cdn = h.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.hf.co/abc?sig=1")
    assert same.unredirected_hdrs.get("Authorization") == f"Bearer {DUMMY_TOKEN}"
    assert "Authorization" not in cdn.unredirected_hdrs and "Authorization" not in cdn.headers
    assert cdn.headers.get("Range") == "bytes=0-3"


def test_backbone_diff_ranged_read_must_be_206_with_the_exact_length(monkeypatch):
    import backbone_diff as bd
    for resp in (_Resp(200, b"abcd"), _Resp(206, b"abc")):
        monkeypatch.setattr(bd, "_open", lambda req, r=resp: r)
        with pytest.raises(SystemExit, match="206"):
            bd.get("https://huggingface.co/x/resolve/main/f", (0, 3))


def test_backbone_diff_http_error_never_shows_the_token(monkeypatch):
    import io
    import urllib.error
    import backbone_diff as bd

    def deny(req):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b""))
    monkeypatch.setattr(bd, "_open", deny)
    monkeypatch.setenv("HF_TOKEN", DUMMY_TOKEN)
    with pytest.raises(SystemExit) as e:
        bd.get("https://huggingface.co/x/resolve/main/f", (0, 3))
    assert "401" in str(e.value) and DUMMY_TOKEN not in str(e.value)


# ---- Fix round 3: HF_TOKEN hygiene (P3-BLD-2) ----------------------------------------------------

CRLF_DUMMY = "hf_dummy_crlf_not_a_secret\r\n"
INNER_DUMMY = "hf_dummy_inner\r\nnot_a_secret"


def test_backbone_diff_token_is_stripped_of_surrounding_whitespace(monkeypatch):
    import backbone_diff as bd
    seen = []
    monkeypatch.setattr(bd, "_open", lambda req: seen.append(req) or _Resp(206, b"abcd"))
    monkeypatch.setenv("HF_TOKEN", CRLF_DUMMY)
    bd.get("https://huggingface.co/x/resolve/main/f", (0, 3))
    assert seen[0].unredirected_hdrs["Authorization"] == "Bearer " + CRLF_DUMMY.strip()
    monkeypatch.setenv("HF_TOKEN", " \t\r\n")
    bd.get("https://huggingface.co/x/resolve/main/f", (0, 3))
    assert "Authorization" not in seen[1].unredirected_hdrs


@pytest.mark.parametrize("bad", [INNER_DUMMY, "hf_dummy has_space", "hf_dummy\x07bel", "hf_dummy_é"])
def test_backbone_diff_malformed_token_is_refused_without_showing_it(monkeypatch, bad):
    import backbone_diff as bd
    monkeypatch.setattr(bd, "_open", lambda req: pytest.fail("a malformed token must fail before any request"))
    monkeypatch.setenv("HF_TOKEN", bad)
    with pytest.raises(SystemExit) as e:
        bd.get("https://huggingface.co/x/resolve/main/f", (0, 3))
    assert "HF_TOKEN is malformed" in str(e.value)
    for part in bad.split():
        assert part.strip() not in str(e.value)


@pytest.mark.parametrize("exc", ["ValueError", "URLError", "HTTPException", "OSError", "RuntimeError"])
def test_backbone_diff_request_errors_never_show_the_token(monkeypatch, capsys, exc):
    import http.client
    import urllib.error
    import backbone_diff as bd
    tok = CRLF_DUMMY.strip()

    def boom(req):
        # mimic http.client / urllib errors that quote the header value
        msg = f"Invalid header value {req.unredirected_hdrs.get('Authorization')!r}"
        raise {"ValueError": ValueError(msg), "URLError": urllib.error.URLError(msg),
               "HTTPException": http.client.HTTPException(msg), "OSError": OSError(msg),
               "RuntimeError": RuntimeError(msg)}[exc]
    monkeypatch.setattr(bd, "_open", boom)
    monkeypatch.setenv("HF_TOKEN", CRLF_DUMMY)
    url = "https://huggingface.co/x/resolve/main/f"
    with pytest.raises(SystemExit) as e:
        bd.get(url, (0, 3))
    text = str(e.value)
    assert url in text and tok not in text
    assert e.value.__suppress_context__ and e.value.__cause__ is None
    out = capsys.readouterr()
    assert tok not in out.out and tok not in out.err


def test_backbone_diff_cli_with_a_malformed_token_prints_no_token(tmp_path):
    """End to end in a subprocess whose environment holds only PATH and a dummy HF_TOKEN."""
    import subprocess
    script = Path(__file__).resolve().parent.parent / "decision" / "backbone_diff.py"
    env = {"PATH": os.environ.get("PATH", ""), "HF_TOKEN": INNER_DUMMY}
    r = subprocess.run([sys.executable, str(script), "org/a-repo", "org/b-repo"], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert "HF_TOKEN is malformed" in r.stderr and "Traceback" not in r.stderr
    for part in ("hf_dummy_inner", "not_a_secret"):
        assert part not in r.stdout and part not in r.stderr


# ---- Fix round 3: config-only target_modules (P3-BLD-4) ------------------------------------------

@pytest.mark.parametrize("tm", ["all-linear", ["all-linear"], "ALL-LINEAR"])
def test_backbone_diff_all_linear_target_modules(tmp_path, tm):
    import backbone_diff as bd
    (tmp_path / "adapter_config.json").write_text(json.dumps({"target_modules": tm}))
    on = bd.adapter_targets(str(tmp_path))
    for n in ("layers.0.self_attn.q_proj.weight", "layers.3.mlp.down_proj.weight",
              "layers.1.linear_attn.in_proj_qkvz.weight", "layers.1.linear_attn.out_proj.weight"):
        assert on(n), n
    for n in ("layers.0.input_layernorm.weight", "layers.1.linear_attn.norm.weight",
              "layers.1.linear_attn.conv1d.weight", "layers.1.linear_attn.A_log", "embed_tokens.weight",
              "norm.weight", "lm_head.weight", "layers.0.self_attn.q_norm.weight"):
        assert not on(n), n
    # a merged all-linear upload no longer reads FINETUNED: it reaches the merge check
    changed = {"layers.0.self_attn.q_proj.weight", "layers.1.mlp.up_proj.weight"}
    v, code, lines = _run(_hdr("model."), _hdr("model.", changed=changed), adapter=on)
    assert (v, code) == ("UNKNOWN", 2) and "no merge check" in lines[-1]


def test_backbone_diff_targets_matching_nothing_are_unknown(tmp_path):
    import backbone_diff as bd
    for tm in ("nothing_proj", ["nothing_proj"]):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"target_modules": tm}))
        on = bd.adapter_targets(str(tmp_path))
        v, code, lines = _run(_hdr("model."), _hdr("model.", changed={"layers.0.self_attn.q_proj.weight"}), adapter=on)
        assert (v, code) == ("UNKNOWN", 2) and "match none" in lines[-1]


# ---- Fix round 3: MERGED must be proved (P3-BLD-5) -----------------------------------------------

def _bf16_bits(x):
    """float32 array -> bf16 bit patterns (round to nearest even)."""
    import numpy as np
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16(x):
    import numpy as np
    return (_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def _write_st(path, tensors):
    """Raw safetensors writer: {name: float32 array}, stored as BF16 when name is in `bf16` (all)."""
    import struct as st
    import numpy as np
    hdr, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        raw = _bf16_bits(arr).tobytes()
        hdr[name] = {"dtype": "BF16", "shape": list(np.shape(arr)), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    Path(path).write_bytes(st.pack("<Q", len(h)) + h + b"".join(blobs))


def _merge_case(seed=0, out=48, inp=32, r=4, alpha=8.0):
    import numpy as np
    rng = np.random.default_rng(seed)
    w = _bf16(rng.normal(0, 0.02, (out, inp)).astype(np.float32))
    la = rng.normal(0, 0.05, (r, inp)).astype(np.float32)
    lb = rng.normal(0, 0.05, (out, r)).astype(np.float32)
    la, lb = _bf16(la), _bf16(lb)
    scale = alpha / r
    delta = scale * (lb.astype(np.float64) @ la.astype(np.float64))
    return w, la, lb, scale, delta


def test_backbone_diff_merge_evidence_on_synthetic_tensors():
    import numpy as np
    import backbone_diff as bd
    w, la, lb, scale, delta = _merge_case()
    up = lambda x: x.astype(np.float64)
    merged = _bf16((w + delta).astype(np.float32))
    assert bd.merge_evidence(up(w), up(merged), up(la), up(lb), scale, "BF16")[0] == "MERGED"
    # either argument order (A = fine-tune, B = stock) proves it too
    assert bd.merge_evidence(up(merged), up(w), up(la), up(lb), scale, "BF16")[0] == "MERGED"
    # an independent fine-tune of the same module, of the same size: unrelated to B@A
    other = np.random.default_rng(7).normal(0, float(np.std(delta)), w.shape)
    indep = _bf16((w + other).astype(np.float32))
    assert bd.merge_evidence(up(w), up(indep), up(la), up(lb), scale, "BF16")[0] == "FINETUNED"
    # the adapter's direction at the wrong scale: neither proved nor refuted
    twice = _bf16((w + 2 * delta).astype(np.float32))
    assert bd.merge_evidence(up(w), up(twice), up(la), up(lb), scale, "BF16")[0] == "UNKNOWN"
    # a merge done in bf16 arithmetic (delta rounded to bf16 first, as PEFT does for bf16 weights)
    merged16 = _bf16((w + _bf16(delta.astype(np.float32))).astype(np.float32))
    assert bd.merge_evidence(up(w), up(merged16), up(la), up(lb), scale, "BF16")[0] == "MERGED"
    # a delta far below bf16's resolution proves nothing, even though the residual is tiny
    tiny = np.random.default_rng(9).normal(0, 1e-7, w.shape)
    assert bd.merge_evidence(up(w), up(w + tiny), up(la), up(lb), 1e-5, "BF16")[0] != "MERGED"
    # shapes that do not compose
    assert bd.merge_evidence(up(w), up(merged), up(la), up(lb.T), scale, "BF16")[0] == "UNKNOWN"


def _merge_repos(tmp_path, kind, cfg=None):
    """Stock repo A, fine-tune repo B (q_proj of layers 0 and 1 changed per `kind`) and a PEFT adapter."""
    import numpy as np
    rng = np.random.default_rng(3)
    names = [f"model.layers.{i}.{s}" for i in range(2) for s in _LAYER_SUFFIXES] + \
        ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    stock = {n: _bf16(rng.normal(0, 0.02, (48, 32)).astype(np.float32)) for n in names}
    tuned, lora = dict(stock), {}
    for i in range(2):
        n = f"model.layers.{i}.self_attn.q_proj.weight"
        _, la, lb, scale, delta = _merge_case(seed=10 + i)
        lora[f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj.lora_A.weight"] = la
        lora[f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj.lora_B.weight"] = lb
        if kind == "merged":
            add = delta
        else:
            add = np.random.default_rng(20 + i).normal(0, float(np.std(delta)), delta.shape)
        tuned[n] = _bf16((stock[n] + add).astype(np.float32))
    for side, ts in (("a", stock), ("b", tuned)):
        (tmp_path / side).mkdir()
        _write_st(tmp_path / side / "model.safetensors", ts)
    ad = tmp_path / "adapter_vllm"
    ad.mkdir()
    _write_st(ad / "adapter_model.safetensors", lora)
    (ad / "adapter_config.json").write_text(json.dumps(cfg or {"r": 4, "lora_alpha": 8.0,
                                                               "target_modules": ["q_proj"]}))
    return tmp_path / "a", tmp_path / "b", ad


@pytest.mark.parametrize("kind,code,verdict", [("merged", 3, "MERGED"), ("independent", 0, "FINETUNED")])
def test_backbone_diff_main_verifies_the_merge(tmp_path, monkeypatch, capsys, kind, code, verdict):
    import backbone_diff as bd
    monkeypatch.setattr(bd, "get", lambda *a, **k: pytest.fail("local dirs must not touch the network"))
    a, b, ad = _merge_repos(tmp_path, kind)
    with pytest.raises(SystemExit) as e:
        bd.main([str(a), str(b), "--adapter", str(ad)])
    out = capsys.readouterr().out.strip().splitlines()
    assert e.value.code == code, out
    assert out[-1].startswith(f"VERDICT: {verdict}")
    assert sum("merge check layers." in l for l in out) == 2


def test_backbone_diff_merge_check_unknown_when_it_cannot_run(tmp_path, monkeypatch, capsys):
    import backbone_diff as bd
    a, b, ad = _merge_repos(tmp_path, "merged", cfg={"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj"],
                                                     "use_dora": True})
    with pytest.raises(SystemExit) as e:
        bd.main([str(a), str(b), "--adapter", str(ad)])
    assert e.value.code == 2 and "use_dora" in capsys.readouterr().out.strip().splitlines()[-1]
    (ad / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj"]}))
    with pytest.raises(SystemExit) as e:  # every target larger than the read cap
        bd.main([str(a), str(b), "--adapter", str(ad), "--merge-check-bytes", "16"])
    assert e.value.code == 2 and "at most 16 bytes" in capsys.readouterr().out.strip().splitlines()[-1]


# ---- Fix round 4: FINETUNED needs a residual beyond rounding (P4-BLD-2) ---------------------------

def _reviewer_merge(dstd, seed=0, out=1024, inp=4096, r=32, scale=2.0):
    """P4-BLD-2's reproduction: a 1024x4096 bf16 weight of std 0.02 and an r=32, scale-2 adapter whose
    delta has std `dstd`, merged in bf16. -> (stock, merged, lora_A, lora_B, scale, rng), all float64."""
    import numpy as np
    rng = np.random.default_rng(seed)
    w = _bf16(rng.normal(0, 0.02, (out, inp)).astype(np.float32)).astype(np.float64)
    s = float(np.sqrt(dstd / (scale * np.sqrt(r))))
    la = _bf16(rng.normal(0, s, (r, inp)).astype(np.float32)).astype(np.float64)
    lb = _bf16(rng.normal(0, s, (out, r)).astype(np.float32)).astype(np.float64)
    merged = _bf16((w + scale * (lb @ la)).astype(np.float32)).astype(np.float64)
    return w, merged, la, lb, scale, rng


def test_backbone_diff_merge_below_resolution_is_unknown_not_finetuned():
    import numpy as np
    import backbone_diff as bd
    # a genuine merge whose delta (std 1e-5, ~0.1 ulp) mostly vanishes in bf16: small cosine, but the
    # residual is far inside rounding, so it is no evidence of an independent fine-tune
    w, merged, la, lb, scale, rng = _reviewer_merge(1e-5)
    v, why = bd.merge_evidence(w, merged, la, lb, scale, "BF16")
    assert v == "UNKNOWN" and "resolution" in why, why
    w2, merged2, la2, lb2, _, _ = _reviewer_merge(2e-6, seed=1)
    assert bd.merge_evidence(w2, merged2, la2, lb2, scale, "BF16")[0] == "UNKNOWN"
    # independent changes on the same weight, from about one ulp down to the delta's own size: their
    # residual RMS is no larger than rounding's, but many elements sit outside a merge's rounding
    # bound (a merge puts none there) -> FINETUNED
    for istd in (1e-4, 1e-5):
        indep = _bf16((w + rng.normal(0, istd, w.shape)).astype(np.float32)).astype(np.float64)
        v, why = bd.merge_evidence(w, indep, la, lb, scale, "BF16")
        assert v == "FINETUNED", (istd, why)
    assert "0.0% of elements beyond rounding" in bd.merge_evidence(w, merged, la, lb, scale, "BF16")[1]
    # and a merge of std 1e-4 is still proved
    w3, merged3, la3, lb3, _, _ = _reviewer_merge(1e-4, seed=2)
    assert bd.merge_evidence(w3, merged3, la3, lb3, scale, "BF16")[0] == "MERGED"


# ---- Fix round 4: config-only regex target_modules see the VLM prefix (P4-BLD-3) ------------------

@pytest.mark.parametrize("rx", [
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
])
def test_backbone_diff_config_regex_matches_vlm_module_names(tmp_path, rx):
    import backbone_diff as bd
    (tmp_path / "adapter_config.json").write_text(json.dumps({"target_modules": rx}))
    on = bd.adapter_targets(str(tmp_path))
    for n in ("layers.3.self_attn.q_proj.weight", "layers.0.mlp.up_proj.weight", "layers.5.mlp.down_proj.weight"):
        assert on(n), n
    for n in ("layers.0.input_layernorm.weight", "embed_tokens.weight", "lm_head.weight",
              "layers.0.self_attn.q_proj.bias", "layers.1.linear_attn.in_proj_qkvz.weight"):
        assert not on(n), n
    # so a change on those modules reaches the merge stage instead of "targets match nothing"
    v, code, lines = _run(_hdr("model."), _hdr("model.language_model.", changed={"layers.0.self_attn.q_proj.weight"}),
                          adapter=on)
    assert (v, code) == ("UNKNOWN", 2) and "no merge check" in lines[-1], lines[-1]


# ---- Fix round 4: a run's answers can be saved and gated against (P4-DRIFT-3) --------------------

def _answering_server(monkeypatch, dump, flip=None):
    """pc.systemone replaced by a server that appends one pass per question and answers every question
    (noul 0.8, choice/score: the first option 0.7); `flip` turns that record's first answer around."""
    def systemone(url, rec):
        answers = {}
        with open(dump, "a") as f:
            for i, (qid, q) in enumerate(rec["questions"].items()):
                f.write(_pass_line(qid) + "\n")
                hi = 0.05 if (rec["id"] == flip and i == 0) else 0.8
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": hi}
                else:
                    c = q["criteria"]
                    opts = list(c) if isinstance(c, dict) else [str(k) for k in range(len(c))]
                    rest = (1 - hi) / (len(opts) - 1)
                    answers[qid] = {"type": q["type"], "probabilities": {o: (hi if k == 0 else rest) for k, o in enumerate(opts)}}
        return {"answers": answers}
    monkeypatch.setattr(pc, "systemone", systemone)


def test_parity_answers_out_saves_a_reference_the_gate_reads(tmp_path, monkeypatch, capsys):
    dump = tmp_path / "kev9b.jsonl"
    dump.write_text("")
    _answering_server(monkeypatch, dump)
    saved = tmp_path / "kev9b.answers.json"
    pc.main(["--url", "http://x", "--dump", str(dump), "--answers-out", str(saved)])
    ans = json.loads(saved.read_text())
    assert len(ans) == 82 and stat.S_IMODE(saved.stat().st_mode) == 0o600
    with pytest.raises(SystemExit, match="--force"):  # never silently overwritten
        pc.main(["--url", "http://x", "--ref", str(saved), "--answers-out", str(saved)])
    # the same answers gate at 100% with every flip counted (the pipeline-vs-single-box check)
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(saved), "--min-agree", "1", "--confident", "0"])
    assert e.value.code == 0 and "PASS agreement 82/82" in capsys.readouterr().out
    # one flipped top-1 fails it
    _answering_server(monkeypatch, tmp_path / "other.jsonl", flip="r01")
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(saved), "--min-agree", "1", "--confident", "0"])
    assert e.value.code == 1 and "FAIL" in capsys.readouterr().out


# ---- P5-RUN-3 / P5-RUN-5: near-tie flips against a saved run; --answers-out only after a PASS ------

def _saved_answers(tmp_path, monkeypatch):
    dump = tmp_path / "single.jsonl"
    dump.write_text("")
    _answering_server(monkeypatch, dump)
    saved = tmp_path / "single.answers.json"
    pc.main(["--url", "http://x", "--dump", str(dump), "--answers-out", str(saved)])
    return saved


def test_parity_near_tie_flip_against_saved_answers_is_informational(tmp_path, monkeypatch, capsys):
    saved = _saved_answers(tmp_path, monkeypatch)
    ref = json.loads(saved.read_text())
    key = next(k for k in ref if k.startswith("r01."))
    n, top = len(ref[key]), max(range(len(ref[key])), key=ref[key].__getitem__)
    # the saved (single-box) run was a near-tie on this question, leaning away from the server's top-1
    near = [0.0] * n
    near[top], near[(top + 1) % n] = 0.49, 0.51
    ref[key] = near
    saved.write_text(json.dumps(ref))
    _answering_server(monkeypatch, tmp_path / "pipe.jsonl")
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(saved), "--min-agree", "0", "--confident", "0.05"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "PASS" in out, out
    assert f"FLIP {key}" in out and "(near-tie, informational)" in out
    assert "max |dp| =" in out
    # the strict form still fails on it
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(saved), "--min-agree", "1", "--confident", "0"])
    assert e.value.code == 1


def test_parity_confident_flip_against_saved_answers_still_fails(tmp_path, monkeypatch, capsys):
    saved = _saved_answers(tmp_path, monkeypatch)
    _answering_server(monkeypatch, tmp_path / "pipe.jsonl", flip="r01")   # saved margin 0.6 on the flipped question
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(saved), "--min-agree", "0", "--confident", "0.05"])
    out = capsys.readouterr().out
    assert e.value.code == 1 and "FAIL 1 confident flip" in out and "near-tie" not in out


def test_parity_answers_out_is_written_only_after_a_pass(tmp_path, monkeypatch, capsys):
    saved = _saved_answers(tmp_path, monkeypatch)
    # PASS: the verdict prints first, then the save
    _answering_server(monkeypatch, tmp_path / "a.jsonl")
    capsys.readouterr()
    ok_out = tmp_path / "pass.answers.json"
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(saved), "--answers-out", str(ok_out)])
    out = capsys.readouterr().out
    assert e.value.code == 0 and ok_out.exists()
    assert out.index("PASS") < out.index(f"saved 82 answers to {ok_out}")
    # FAIL: nothing is written
    _answering_server(monkeypatch, tmp_path / "b.jsonl", flip="r01")
    bad_out = tmp_path / "fail.answers.json"
    with pytest.raises(SystemExit) as e:
        pc.main(["--url", "http://x", "--ref", str(saved), "--min-agree", "1", "--confident", "0",
                 "--answers-out", str(bad_out)])
    out = capsys.readouterr().out
    assert e.value.code == 1 and not bad_out.exists()
    assert "FAIL" in out and "answers not saved" in out


def test_max_abs_delta():
    rec = {"id": "r", "questions": {"q": {"type": "noul"}}}
    assert pc.max_abs_delta({"r.q": [0.3, 0.7]}, [(rec, {"answers": {"q": {"type": "noul", "noul": 0.6}}})]) == pytest.approx(0.1)
