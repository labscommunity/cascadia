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

SPIKE = Path(os.environ.get("SPIKE_DIR", "/Users/blokchainaholic/Desktop/clabs/tahoma/experiments"))


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
    try:
        kev.records = lambda: [INJECT]
        kev.user_tokens = lambda t, text: saved_user_tokens(t, cg.escape_added(text, added))
        ((_, _, row, _, ends, _),) = kev.questions(tok)
    finally:
        kev.records, kev.user_tokens = saved_records, saved_user_tokens
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
