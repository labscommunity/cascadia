"""Exporter decision options and decision.json/assets writers.

    python -m pytest tools/tests/test_decision_export.py -v
"""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import export_qwen36_moe as ex  # noqa: E402


def test_manifest_records_final_hidden():
    spec = ex.ModelSpec(model_type="qwen3_5", hidden=8, num_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    m = ex.build_manifest(spec, "src", final_hidden=True)
    assert m["final_hidden_output"] is True
    assert list(m)[-2:] == ["final_hidden_output", "stages"]
    assert "final_hidden_output" not in ex.build_manifest(spec, "src")


def test_chat_manifest_is_byte_identical_to_the_pre_decision_exporter():
    """Ranks compare manifest.json byte for byte at HELLO and kv_fingerprint hashes it, so a chat
    export must not grow a `final_hidden_output: false` key a tree from main does not have."""
    spec = ex.ModelSpec(model_type="qwen3_5", hidden=8, num_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    main_manifest = {"arch": "qwen3_5", "family": spec.family, "hidden_size": 8, "num_layers": 4,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"], "source": "src",
                     "last_logits_only": True, "stages": []}
    assert json.dumps(ex.build_manifest(spec, "src"), indent=2) == json.dumps(main_manifest, indent=2)


# ---------------------------------------------------------------------------
# run_export wiring, with the IR work stubbed out (no model download, no real IR)
# ---------------------------------------------------------------------------

LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]


class _FakeStage:
    def get_parameters(self):
        return []

    def get_sinks(self):
        return []


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """A model dir with config.json + placeholder IR files; extract_stage, ov.save_model and the
    weight_swap entry points are replaced with recorders."""
    ov = pytest.importorskip("openvino")
    import weight_swap

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "hidden_size": 8, "num_hidden_layers": 8,
                        "layer_types": LAYER_TYPES, "vocab_size": 16},
    }))
    for aux in ("openvino_language_model.xml", "openvino_text_embeddings_model.xml",
                "openvino_text_embeddings_model.bin", "tokenizer.json"):
        (model / aux).write_text("stock " + aux)
    calls = []
    monkeypatch.setattr(ex, "extract_stage", lambda xml, spec, a, b, first, last: _FakeStage())
    monkeypatch.setattr(ov, "save_model", lambda *a, **kw: calls.append(("save",)))
    monkeypatch.setattr(weight_swap, "Weights", lambda d: ("W", d))
    reports = []

    def swap_stage(stage, w):
        calls.append(("swap", w))
        reports.append(object())  # per-call sentinel: check_swap must get this stage's report
        return reports[-1]

    def check_swap(report, layers, lt, last):
        assert report is reports[-1]
        assert lt == LAYER_TYPES
        calls.append(("check", list(layers), last))
    monkeypatch.setattr(weight_swap, "swap_stage", swap_stage)
    monkeypatch.setattr(weight_swap, "check_swap", check_swap)
    monkeypatch.setattr(weight_swap, "swap_embeddings",
                        lambda xml, w, out: calls.append(("emb", os.path.basename(xml), w)))
    monkeypatch.setattr(weight_swap, "add_final_hidden_output", lambda stage: calls.append(("fh",)))
    return model, tmp_path / "out", calls, weight_swap, monkeypatch


def _lora(monkeypatch, ws, calls, per_stage, targets, cfg_scale=2.0):
    """graft_lora returns `per_stage` (the modules it grafted) on every stage; None = the targets
    of that stage's layers (layer index in the module name), so the coverage check passes."""
    monkeypatch.setattr(ws, "lora_scale_from_config", lambda f: cfg_scale)
    monkeypatch.setattr(ws, "lora_targets", lambda f: targets)

    def graft(stage, f, s, layers=None, lm_head=True):
        calls.append(("graft", s))
        if per_stage is not None:
            return per_stage
        return [t for t in targets if int(t.split(".")[1]) in layers]
    monkeypatch.setattr(ws, "graft_lora", graft)


def test_chat_export_unchanged_without_decision_flags(stubbed):
    model, out, calls, _, _ = stubbed
    ex.run_export(str(model), str(out), 2)
    assert [c[0] for c in calls] == ["save", "save"]
    m = json.loads((out / "manifest.json").read_text())
    assert "final_hidden_output" not in m
    assert (out / "openvino_text_embeddings_model.bin").read_text() == "stock openvino_text_embeddings_model.bin"


def test_weights_swaps_and_checks_every_stage_and_embeddings(stubbed, tmp_path):
    model, out, calls, _, _ = stubbed
    wdir = tmp_path / "ft"
    wdir.mkdir()
    (wdir / "tokenizer.json").write_text("fine-tune tokenizer")
    ft_cfg = (model / "config.json").read_text().replace('"vocab_size": 16', '"vocab_size": 16, "ft": 1')
    (wdir / "config.json").write_text(ft_cfg)
    ex.run_export(str(model), str(out), 2, weights=str(wdir))
    assert [c[0] for c in calls if c[0] in ("swap", "check")] == ["swap", "check", "swap", "check"]
    checks = [c for c in calls if c[0] == "check"]
    assert checks == [("check", [0, 1, 2, 3], False), ("check", [4, 5, 6, 7], True)]
    assert ("emb", "openvino_text_embeddings_model.xml", ("W", str(wdir))) in calls
    assert not (out / "openvino_text_embeddings_model.bin").exists()  # not copied from the stock IR
    assert (out / "tokenizer.json").read_text() == "fine-tune tokenizer"
    assert (out / "config.json").read_text() == ft_cfg
    assert (out / "manifest.json").exists()


def test_weights_with_other_architecture_refused_before_any_work(stubbed, tmp_path):
    model, out, calls, _, _ = stubbed
    wdir = tmp_path / "ft"
    wdir.mkdir()
    cfg = json.loads((model / "config.json").read_text())
    cfg["text_config"]["hidden_size"] = 16
    (wdir / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="not the IR's architecture"):
        ex.run_export(str(model), str(out), 2, weights=str(wdir))
    assert calls == []


@pytest.mark.parametrize("flag", ["weights", "lora"])
def test_validate_refused_with_weights_or_lora_before_any_work(stubbed, tmp_path, flag):
    model, out, calls, _, mp = stubbed
    mp.setattr(ex, "extract_stage", lambda *a: pytest.fail("extract_stage ran"))
    with pytest.raises(SystemExit, match="--validate"):
        ex.run_export(str(model), str(out), 2, validate=True, **{flag: str(tmp_path / "x")})
    assert calls == [] and not out.exists()


def test_lora_graft_must_cover_every_adapter_module(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=["layers.0.a"], targets=["layers.0.a", "layers.4.b"])
    out.mkdir()
    (out / "manifest.json").write_text("{}")  # stale manifest from an earlier run in the same dir
    with pytest.raises(SystemExit, match=r"grafted more than once: \['layers\.0\.a'\]; "
                                         r"not grafted: \['layers\.4\.b'\]"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "adapter_model.safetensors"))
    assert not (out / "manifest.json").exists()


def test_lora_key_matching_two_matmuls_cannot_cancel_one_matching_none(stubbed, tmp_path):
    # 2 grafts for 2 targets passed the old count check
    model, out, calls, ws, mp = stubbed
    stages = iter([["layers.0.a", "layers.0.a"], []])
    _lora(mp, ws, calls, per_stage=None, targets=["layers.0.a", "layers.0.b"])
    mp.setattr(ws, "graft_lora", lambda stage, f, s, layers=None, lm_head=True: next(stages))
    with pytest.raises(SystemExit, match=r"not grafted: \['layers\.0\.b'\]"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "adapter_model.safetensors"))
    assert not (out / "manifest.json").exists()


def test_lora_audit_adapter_refused_before_any_work(stubbed, tmp_path):
    # limits-audit B2: embedding LoRA + modules_to_save ["norm"]; the real weight_swap checks run
    model, out, calls, ws, mp = stubbed
    sys.path.insert(0, os.path.dirname(__file__))
    from test_weight_swap import save_adapter, write_audit_adapter
    mp.setattr(ex, "extract_stage", lambda *a: pytest.fail("extract_stage ran"))
    ad = tmp_path / "ad"
    ad.mkdir()
    path = write_audit_adapter(ad)
    with pytest.raises(ValueError, match="modules_to_save"):
        ex.run_export(str(model), str(out), 2, lora=path)
    # without adapter_config.json (explicit --lora-scale) the key check still refuses
    (ad / "adapter_config.json").unlink()
    with pytest.raises(ValueError, match="lora_embedding"):
        ex.run_export(str(model), str(out), 2, lora=path, lora_scale=1.0)
    p = "base_model.model.model.layers.0.mlp.down_proj"
    path = save_adapter(ad / "saved.safetensors", [f"{p}.lora_A.weight", f"{p}.lora_B.weight",
                                                   "base_model.model.model.norm.weight"])
    with pytest.raises(ValueError, match=r"model\.norm\.weight"):
        ex.run_export(str(model), str(out), 2, lora=path, lora_scale=1.0)
    assert calls == []


def test_embeddings_swap_failure_leaves_no_manifest(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed

    def boom(*a):
        raise ValueError("no embedding Constant")
    mp.setattr(ws, "swap_embeddings", boom)
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        ex.run_export(str(model), str(out), 2, weights=str(tmp_path / "ft"))
    assert not (out / "manifest.json").exists()


def test_lora_scale_comes_from_adapter_config(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=None, targets=["layers.0.a", "layers.4.a"], cfg_scale=2.0)
    ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "adapter_model.safetensors"))
    assert [c for c in calls if c[0] == "graft"] == [("graft", 2.0), ("graft", 2.0)]


def test_lora_scale_override_warns(stubbed, tmp_path, capsys):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=None, targets=["layers.0.a", "layers.4.a"], cfg_scale=2.0)
    ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "a.safetensors"), lora_scale=0.5)
    assert [c for c in calls if c[0] == "graft"] == [("graft", 0.5), ("graft", 0.5)]
    assert "WARNING: --lora-scale 0.5 overrides adapter_config.json's 2.0" in capsys.readouterr().out


def test_lora_config_errors_propagate(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed

    def refuse(f):
        raise ValueError("adapter_config.json has use_dora: true")
    mp.setattr(ws, "lora_scale_from_config", refuse)
    with pytest.raises(ValueError, match="use_dora"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "a.safetensors"))


def test_missing_adapter_config_needs_explicit_scale(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed

    def missing(f):
        raise FileNotFoundError(f)
    mp.setattr(ws, "lora_scale_from_config", missing)
    with pytest.raises(SystemExit, match="pass --lora-scale"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "a.safetensors"))


def test_final_hidden_only_on_last_stage_and_in_manifest(stubbed):
    model, out, calls, _, _ = stubbed
    ex.run_export(str(model), str(out), 2, final_hidden=True)
    assert [c[0] for c in calls] == ["save", "fh", "save"]
    assert json.loads((out / "manifest.json").read_text())["final_hidden_output"] is True


def test_cli_flags_reach_run_export(monkeypatch):
    seen = {}
    monkeypatch.setattr(ex, "run_export", lambda *a, **kw: seen.update(a=a, kw=kw))
    monkeypatch.setattr(sys, "argv", ["x", "--model", "m", "--out", "o", "--total", "8",
                                      "--weights", "w", "--lora", "l.safetensors",
                                      "--lora-scale", "0.25", "--final-hidden"])
    ex.main()
    assert seen["a"] == ("m", "o", 8, False, None)
    assert seen["kw"] == {"weights": "w", "lora": "l.safetensors", "lora_scale": 0.25, "final_hidden": True}


# ---------------------------------------------------------------------------
# decision_export.py: decision.json profile + decision/ head assets
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

import decision_export as de  # noqa: E402


def test_jev_profile_from_adapter_files(tmp_path):
    adapter = tmp_path / "adapter_vllm"
    adapter.mkdir()
    ids = list(range(100, 124))
    (adapter / "decision_head.json").write_text(json.dumps({
        "bias": [0.1] * 24, "verbalizer_ids": ids,
        "slots": {"num_slots": 24, "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                  "verbalizers": ["false", "true"] + [str(i) for i in range(6)] + [chr(65 + i) for i in range(16)]}}))
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"per_kind": {"noul": 1.0, "choice": 0.98, "score": 1.01}}))
    out = tmp_path / "tree"
    de.write_jev_profile(str(adapter), str(cal), str(out), max_tokens=4096)
    d = json.loads((out / "decision.json").read_text())
    assert d["family"] == "jev" and d["max_tokens"] == 4096
    assert d["jev"]["verbalizer_ids"] == ids and d["jev"]["slots"]["choice"] == [8, 24]
    assert d["jev"]["temperature"]["choice"] == 0.98
    assert json.loads((out / "decision" / "head.json").read_text()) == {"kind": "verbalizer"}


def test_kev_profile_writes_pointer_weights(tmp_path):
    torch = pytest.importorskip("torch")
    head = {"head": {"q.weight": torch.ones(256, 8), "q.bias": torch.zeros(256),
                     "k.weight": torch.full((256, 8), 2.0), "k.bias": torch.ones(256)},
            "temperature": 2.19}
    pt = tmp_path / "head.pt"
    torch.save(head, pt)
    out = tmp_path / "tree"
    de.write_kev_profile(str(pt), str(out), max_tokens=73728)
    d = json.loads((out / "decision.json").read_text())
    assert d == {"family": "kev", "max_tokens": 73728, "kev": {"temperature": 2.19}}
    kw = np.fromfile(out / "decision" / "pointer_k_w.f32", dtype="<f4")
    assert kw.shape == (256 * 8,) and kw[0] == 2.0
    assert json.loads((out / "decision" / "head.json").read_text()) == {"kind": "pointer", "dp": 256}


def test_kev_pointer_weights_are_row_major_out_by_in(tmp_path):
    # decision.rs `affine` reads w[o * hidden + i]: torch Linear weight (out, in), row-major.
    torch = pytest.importorskip("torch")
    qw = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    head = {"head": {"q.weight": qw, "q.bias": torch.tensor([7.0, 8.0, 9.0]),
                     "k.weight": qw.bfloat16(), "k.bias": torch.zeros(3)}, "temperature": 1.5}
    pt = tmp_path / "head.pt"
    torch.save(head, pt)
    out = tmp_path / "tree"
    de.write_kev_profile(str(pt), str(out))
    d = out / "decision"
    assert np.fromfile(d / "pointer_q_w.f32", dtype="<f4").tolist() == [0, 1, 2, 3, 4, 5]
    assert np.fromfile(d / "pointer_q_b.f32", dtype="<f4").tolist() == [7, 8, 9]
    assert np.fromfile(d / "pointer_k_w.f32", dtype="<f4").tolist() == [0, 1, 2, 3, 4, 5]
    assert json.loads((d / "head.json").read_text()) == {"kind": "pointer", "dp": 3}


class _Pickled:  # any non-allowlisted class is refused by torch.load(weights_only=True)
    pass


def test_kev_head_pt_is_loaded_weights_only(tmp_path):
    torch = pytest.importorskip("torch")
    pt = tmp_path / "head.pt"
    torch.save({"head": {}, "temperature": 1.0, "x": _Pickled()}, pt)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        de.write_kev_profile(str(pt), str(tmp_path / "tree"))


def _clef_checkpoint(tmp_path, vocab=5, hidden=3, key="lm_head.weight", tied=False):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    ck = tmp_path / "clef"
    ck.mkdir()
    lm = torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden).bfloat16()
    save_file({key: lm}, str(ck / "model-00001.safetensors"))
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model-00001.safetensors"}}))
    (ck / "config.json").write_text(json.dumps({"tie_word_embeddings": tied, "text_config": {}}))
    return ck, lm


def test_clef_profile_writes_lm_head_f16_and_joint_head(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # clef_head imports torch at module level
    from decision import clef_head
    ck, lm = _clef_checkpoint(tmp_path)
    seen = []
    monkeypatch.setattr(clef_head, "export_joint_head",
                        lambda d, xml: seen.append((d, xml)) or open(xml, "w").close())
    out = tmp_path / "tree"
    de.write_clef_profile(str(ck), str(out), max_tokens=16384, chunk=2)  # chunk < vocab: streamed
    d = out / "decision"
    assert json.loads((out / "decision.json").read_text()) == {"family": "clef", "max_tokens": 16384, "clef": {}}
    assert json.loads((d / "head.json").read_text()) == {"kind": "joint", "vocab": 5}
    assert seen == [(str(ck), str(d / "joint_head.xml"))]
    got = np.fromfile(d / "lm_head.f16", dtype="<f2")
    assert got.shape == (5 * 3,) and got.tolist() == lm.float().flatten().tolist()


def test_clef_tied_embeddings_use_embed_tokens(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # clef_head imports torch at module level
    from decision import clef_head
    ck, lm = _clef_checkpoint(tmp_path, key="model.language_model.embed_tokens.weight", tied=True)
    monkeypatch.setattr(clef_head, "export_joint_head", lambda d, xml: open(xml, "w").close())
    out = tmp_path / "tree"
    de.write_clef_profile(str(ck), str(out))
    assert np.fromfile(out / "decision" / "lm_head.f16", dtype="<f2").tolist() == lm.float().flatten().tolist()


def test_clef_missing_lm_head_refused_without_partial_head_json(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # clef_head imports torch at module level
    from decision import clef_head
    ck, _ = _clef_checkpoint(tmp_path, key="model.language_model.embed_tokens.weight", tied=False)
    monkeypatch.setattr(clef_head, "export_joint_head", lambda d, xml: open(xml, "w").close())
    out = tmp_path / "tree"
    with pytest.raises(SystemExit, match="lm_head.weight"):
        de.write_clef_profile(str(ck), str(out))
    assert not (out / "decision" / "head.json").exists()


# CLI wiring: run_export stubbed (no IR), profile writers stubbed or synthetic.

@pytest.fixture
def cli(monkeypatch, tmp_path):
    calls = []

    def fake_run_export(*a, **kw):
        # decision assets must already be in place when the manifest (commit marker) is written
        calls.append(("export", a, kw, os.path.exists(os.path.join(a[1], "decision.json"))))
    monkeypatch.setattr(ex, "run_export", fake_run_export)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144, "hidden_size": 8}}))

    # JEV derives its choice labels from the tokenizer.json the tree ships (the base IR's here)
    (base / "tokenizer.json").write_text("base tokenizer")

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["decision_export.py", *argv])
        de.main()
    run.tokenizers = []
    monkeypatch.setattr(de, "load_tokenizer", lambda path: run.tokenizers.append(path) or _jev_tok())
    return run, calls, base, tmp_path


def test_cli_jev_lora_from_adapter_dir_and_no_validate(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "write_jev_profile",
                        lambda ad, cal, out, max_tokens, tokenizer=None: seen.update(ad=ad, cal=cal, mt=max_tokens)
                        or de._write(out, {"family": "jev"}, {"kind": "verbalizer"}))
    ad = tmp / "jev" / "adapter_vllm"
    run("jev", "--base-ir", str(base), "--adapter-dir", str(ad), "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a == (str(base), str(tmp / "t"), 1, False, None)  # Task 58: a single box is one stage
    assert kw == {"weights": None, "lora": os.path.join(str(ad), "adapter_model.safetensors"),
                  "lora_scale": None, "final_hidden": False}
    assert assets_first
    assert seen == {"ad": str(ad), "cal": str(tmp / "jev" / "calibration.json"), "mt": 32768}


def test_cli_kev_final_hidden_weights_lora(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "kev_head_meta", lambda pt: {"base": None, "weights": None})
    monkeypatch.setattr(de, "write_kev_profile",
                        lambda pt, out, hidden, max_tokens=None: seen.update(pt=pt, hidden=hidden) or de._write(out, {"family": "kev"}, {"kind": "pointer"}))
    pt = str(_kev_dir(tmp) / "H.pt")
    run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors", "--lora-scale", "2",
        "--head-pt", pt, "--stages", "2", "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a == (str(base), str(tmp / "t"), 2, False, None)
    assert kw == {"weights": "W", "lora": "L.safetensors", "lora_scale": 2.0, "final_hidden": True,
                  "tokenizer": str(tmp / "kev" / "tokenizer.json")}
    assert assets_first and seen == {"pt": pt, "hidden": 8}


def test_cli_clef_weights_default_to_clef_dir(cli, monkeypatch):
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "write_clef_profile",
                        lambda cd, out, max_tokens=None: de._write(out, {"family": "clef"}, {"kind": "joint"}))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a[3] is False and kw == {"weights": "C", "lora": None, "lora_scale": None, "final_hidden": True}
    assert assets_first


@pytest.mark.parametrize("flags", [["--lora", "L.safetensors"], ["--lora-scale", "1.0"]])
def test_cli_clef_refuses_a_lora(cli, monkeypatch, capsys, flags):
    # H36: Clef's joint head reads its own full checkpoint; a grafted adapter would move it
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "write_clef_profile", lambda *a, **kw: pytest.fail("clef profile written"))
    with pytest.raises(SystemExit) as e:
        run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"), *flags)
    assert e.value.code == 2 and "clef takes no --lora/--lora-scale" in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()


@pytest.mark.parametrize("family,missing", [("jev", "--adapter-dir"), ("kev", "--head-pt"), ("clef", "--clef-dir")])
def test_cli_family_inputs_required_before_export(cli, family, missing, capsys):
    run, calls, base, tmp = cli
    with pytest.raises(SystemExit):
        run(family, "--base-ir", str(base), "--out", str(tmp / "t"))
    assert missing in capsys.readouterr().err and calls == []


# Fix round 1: stale manifest, writer-side checks, config source for max_tokens.

def _jev_adapter(tmp_path, n_bias=24, choice=(8, 24), per_kind=None):
    adapter = tmp_path / "jev" / "adapter_vllm"
    adapter.mkdir(parents=True)
    (adapter / "decision_head.json").write_text(json.dumps({
        "bias": [0.0] * n_bias, "verbalizer_ids": list(range(24)),
        "slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": list(choice)},
                  "verbalizers": [str(i) for i in range(24)]}}))
    cal = tmp_path / "jev" / "calibration.json"
    cal.write_text(json.dumps({"per_kind": per_kind or {"noul": 1.0, "choice": 1.0, "score": 1.0}}))
    return adapter, cal


def test_cli_removes_stale_manifest_before_writing_assets(cli, monkeypatch):
    run, calls, base, tmp = cli
    out = tmp / "t"
    out.mkdir()
    (out / "manifest.json").write_text("{}")  # earlier export in the same tree

    def early_failure(*a, **kw):  # e.g. --weights architecture mismatch, before run_export's own removal
        raise SystemExit("--weights W is not the IR's architecture")
    monkeypatch.setattr(ex, "run_export", early_failure)
    adapter, _ = _jev_adapter(tmp)
    with pytest.raises(SystemExit, match="architecture"):
        run("jev", "--base-ir", str(base), "--adapter-dir", str(adapter), "--out", str(out))
    assert not (out / "manifest.json").exists()
    assert (out / "decision.json").exists()  # new assets, but no loadable tree


def test_cli_stale_manifest_removed_when_asset_writer_fails(cli, monkeypatch):
    run, calls, base, tmp = cli
    out = tmp / "t"
    out.mkdir()
    (out / "manifest.json").write_text("{}")

    def boom(*a, **kw):
        raise ValueError("bad head.pt")
    monkeypatch.setattr(de, "kev_head_meta", lambda pt: {"base": None, "weights": None})
    monkeypatch.setattr(de, "write_kev_profile", boom)
    with pytest.raises(ValueError):
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(_kev_dir(tmp) / "H.pt"), "--out", str(out))
    assert not (out / "manifest.json").exists() and calls == []


def test_jev_calibration_missing_kind_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path, per_kind={"noul": 1.0, "choice": 0.98})
    with pytest.raises(ValueError, match="score"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_jev_length_mismatch_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path, n_bias=23)
    with pytest.raises(ValueError, match="lengths differ: 24/24/23"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_jev_slot_range_out_of_bounds_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path, choice=(8, 25))
    with pytest.raises(ValueError, match="choice"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


@pytest.mark.parametrize("kind,rng,match", [
    ("noul", (0, 3), "noul .* must be 2 wide"),
    ("score", (2, 7), "score .* must be 6 wide"),
    ("choice", (9, 8), "choice"),
])
def test_jev_slot_widths_mirror_the_server(tmp_path, kind, rng, match):
    # JevCodec::new refuses these at load; the exporter must not write a profile the server rejects
    adapter, cal = _jev_adapter(tmp_path)
    h = json.loads((adapter / "decision_head.json").read_text())
    h["slots"]["ranges"][kind] = list(rng)
    (adapter / "decision_head.json").write_text(json.dumps(h))
    with pytest.raises(ValueError, match=match):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_jev_missing_slot_range_raises(tmp_path):
    adapter, cal = _jev_adapter(tmp_path)
    h = json.loads((adapter / "decision_head.json").read_text())
    del h["slots"]["ranges"]["score"]
    (adapter / "decision_head.json").write_text(json.dumps(h))
    with pytest.raises(ValueError, match="score"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))


def test_kev_hidden_mismatch_raises(tmp_path):
    torch = pytest.importorskip("torch")
    pt = tmp_path / "head.pt"
    torch.save({"head": {"q.weight": torch.ones(4, 8), "q.bias": torch.zeros(4),
                         "k.weight": torch.ones(4, 8), "k.bias": torch.zeros(4)}, "temperature": 1.0}, pt)
    with pytest.raises(ValueError, match="q.weight has hidden dim 8 but the base IR's hidden_size is 16"):
        de.write_kev_profile(str(pt), str(tmp_path / "tree"), hidden=16)
    de.write_kev_profile(str(pt), str(tmp_path / "ok"), hidden=8)


def test_cli_jev_max_tokens_from_weights_config(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "write_jev_profile",
                        lambda ad, cal, out, max_tokens, tokenizer=None: seen.update(mt=max_tokens))
    w = tmp / "ft"
    w.mkdir()
    (w / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 4096}}))
    run("jev", "--base-ir", str(base), "--weights", str(w), "--weights-not-merged", "--adapter-dir", str(tmp / "a"),
        "--out", str(tmp / "t"))
    assert seen == {"mt": 4096}  # the fine-tune's config (shipped in the tree), not the base IR's 262144


# Scoped-review fixes: Kev ships Kev's own tokenizer, and a kev export needs its backbone + adapter.

def test_tokenizer_override_replaces_the_weights_tokenizer_before_the_manifest(stubbed, tmp_path):
    model, out, calls, _, monkeypatch = stubbed
    wdir = tmp_path / "ft"
    wdir.mkdir()
    (wdir / "tokenizer.json").write_text("base tokenizer")
    (wdir / "config.json").write_text((model / "config.json").read_text())
    kev_tok = tmp_path / "kev" / "tokenizer.json"
    kev_tok.parent.mkdir()
    kev_tok.write_text("kev tokenizer")
    import shutil
    real_copy = shutil.copy2
    seen = []

    def copy2(src, dst):
        if os.path.basename(dst) == "tokenizer.json":
            seen.append((str(src), (out / "manifest.json").exists()))
        return real_copy(src, dst)
    monkeypatch.setattr(shutil, "copy2", copy2)
    ex.run_export(str(model), str(out), 2, weights=str(wdir), tokenizer=str(kev_tok))
    assert (out / "tokenizer.json").read_text() == "kev tokenizer"
    assert seen[-1] == (str(kev_tok), False)  # the override lands before the manifest (commit marker)
    assert (out / "manifest.json").exists()


def test_missing_tokenizer_override_refused_before_any_work(stubbed, tmp_path):
    model, out, calls, _, _ = stubbed
    with pytest.raises(SystemExit, match="tokenizer"):
        ex.run_export(str(model), str(out), 2, tokenizer=str(tmp_path / "nope" / "tokenizer.json"))
    assert calls == [] and not (out / "manifest.json").exists()


def _kev_dir(tmp, tokenizer=True):
    d = tmp / "kev"
    d.mkdir(exist_ok=True)
    if tokenizer:
        (d / "tokenizer.json").write_text("kev tokenizer")
    return d


def _stub_kev(monkeypatch, meta=None):
    monkeypatch.setattr(de, "kev_head_meta", lambda pt: meta or {"base": "Qwen/Qwen3.5-9B-Base", "weights": "lora"})
    monkeypatch.setattr(de, "write_kev_profile",
                        lambda pt, out, hidden, max_tokens=None: de._write(out, {"family": "kev"}, {"kind": "pointer"}))


def test_cli_kev_ships_the_tokenizer_beside_head_pt(cli, monkeypatch):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp)
    run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
        "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    (_, a, kw, _), = calls
    assert kw["tokenizer"] == str(kd / "tokenizer.json")


def test_cli_tokenizer_flag_overrides(cli, monkeypatch):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp, tokenizer=False)
    other = tmp / "other.json"
    other.write_text("t")
    run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
        "--head-pt", str(kd / "head.pt"), "--tokenizer", str(other), "--out", str(tmp / "t"))
    (_, a, kw, _), = calls
    assert kw["tokenizer"] == str(other)


def test_cli_kev_without_its_tokenizer_refused_before_any_write(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp, tokenizer=False)
    with pytest.raises(SystemExit):
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    assert "tokenizer.json" in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()


@pytest.mark.parametrize("drop", ["--weights", "--lora"])
def test_cli_kev_needs_weights_and_lora(cli, monkeypatch, capsys, drop):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch)
    kd = _kev_dir(tmp)
    argv = {"--weights": "W", "--lora": "L.safetensors"}
    del argv[drop]
    with pytest.raises(SystemExit):
        run("kev", "--base-ir", str(base), *[x for kv in argv.items() for x in kv],
            "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    err = capsys.readouterr().err
    assert drop in err and "Qwen/Qwen3.5-9B-Base" in err
    assert calls == [] and not (tmp / "t").exists()


def test_cli_kev_head_pt_recording_full_weights_refused(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    _stub_kev(monkeypatch, meta={"base": "B", "weights": "full"})
    kd = _kev_dir(tmp)
    with pytest.raises(SystemExit):
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(kd / "head.pt"), "--out", str(tmp / "t"))
    assert "'full'" in capsys.readouterr().err and calls == []


def test_kev_head_meta_reads_base_and_weights_mode(tmp_path):
    torch = pytest.importorskip("torch")
    pt = tmp_path / "head.pt"
    head = {k: torch.zeros(1) for k in ("q.weight", "q.bias", "k.weight", "k.bias")}
    torch.save({"head": head, "temperature": 1.0, "base": "Qwen/Qwen3.5-9B-Base", "weights": "lora"}, pt)
    assert de.kev_head_meta(str(pt)) == {"base": "Qwen/Qwen3.5-9B-Base", "weights": "lora", "unsupported": []}
    torch.save({"head": head, "temperature": 1.0}, pt)
    assert de.kev_head_meta(str(pt)) == {"base": None, "weights": None, "unsupported": []}
    torch.save({"head": {}, "temperature": 1.0, "x": _Pickled()}, pt)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        de.kev_head_meta(str(pt))


def test_cli_jev_and_clef_pass_no_tokenizer_by_default(cli, monkeypatch):
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "write_clef_profile",
                        lambda cd, out, max_tokens=None: de._write(out, {"family": "clef"}, {"kind": "joint"}))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"))
    (_, a, kw, _), = calls
    assert "tokenizer" not in kw


@pytest.mark.parametrize("content", [None, b"not a torch file", "list"])
def test_cli_bad_head_pt_is_a_clean_error(cli, monkeypatch, capsys, content):
    # A missing, unreadable or non-dict head.pt must exit with a message, not a traceback.
    run, calls, base, tmp = cli
    kd = _kev_dir(tmp)
    pt = kd / "head.pt"
    if content == "list":
        torch = pytest.importorskip("torch")
        torch.save([1, 2], pt)
    elif content is not None:
        pt.write_bytes(content)
    with pytest.raises(SystemExit) as e:
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", str(pt), "--out", str(tmp / "t"))
    assert e.value.code == 2
    assert str(pt) in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()


def test_cli_jev_weights_need_the_not_merged_statement(cli, monkeypatch, capsys):
    """P2-BLD-1: --weights plus the grafted --lora applies the adapter twice when the shards already
    hold it merged; the operator must state they do not (backbone_diff said FINETUNED, not MERGED)."""
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "write_jev_profile",
                        lambda ad, cal, out, max_tokens, tokenizer=None: de._write(out, {"family": "jev"}, {"kind": "verbalizer"}))
    ad = tmp / "jev" / "adapter_vllm"
    with pytest.raises(SystemExit) as e:
        run("jev", "--base-ir", str(base), "--adapter-dir", str(ad), "--weights", str(base), "--out", str(tmp / "t"))
    assert e.value.code == 2 and "--weights-not-merged" in capsys.readouterr().err
    assert calls == []
    run("jev", "--base-ir", str(base), "--adapter-dir", str(ad), "--weights", str(base), "--weights-not-merged",
        "--out", str(tmp / "t"))
    (_, _, kw, _), = calls
    assert kw["weights"] == str(base)


# Task 23 (limits-audit B3): head.pt features the server's pointer head does not implement.

def _kev_head(pt, **meta):
    """A head.pt shaped like jaredpalmer/kev-9b's (kev.checkpoint.Meta fields + training args)."""
    torch = pytest.importorskip("torch")
    d = {"head": {"q.weight": torch.ones(4, 8), "q.bias": torch.zeros(4),
                  "k.weight": torch.ones(4, 8), "k.bias": torch.zeros(4)},
         "temperature": 2.19, "base": "Qwen/Qwen3.5-9B-Base", "base_revision": "rev", "lora": 16,
         "head_dim": 4, "option_isolation": False, "special_embeddings": False, "weights_dtype": "fp32",
         "holdout": [], "weights": "lora", "lora_placement": "full",
         "args": {"option_isolation": 0, "special_embeddings": 0, "lora_targets": "all"}}
    d.update(meta)
    torch.save({k: v for k, v in d.items() if v is not _DROP}, pt)
    return str(pt)


_DROP = object()


def test_kev_shipped_head_shape_exports(tmp_path):
    pt = _kev_head(tmp_path / "head.pt")
    de.write_kev_profile(pt, str(tmp_path / "tree"), hidden=8)
    assert json.loads((tmp_path / "tree" / "decision" / "head.json").read_text()) == {"kind": "pointer", "dp": 4}
    # older checkpoints without the flags take kev.checkpoint.Meta's defaults
    pt = _kev_head(tmp_path / "old.pt", head_dim=_DROP, option_isolation=_DROP, special_embeddings=_DROP)
    de.write_kev_profile(pt, str(tmp_path / "old"), hidden=8)


@pytest.mark.parametrize("meta,match", [
    ({"option_isolation": True}, "option_isolation"),
    ({"special_embeddings": True}, "special_embeddings"),
    ({"head_dim": 256}, "head_dim 256"),
    ({"weights": "full"}, "weights 'full'"),
    ({"head": None}, "head tensors None"),  # Meta's default: no head saved (was a TypeError)
    ({"head_dim": None}, "head_dim None"),  # was int(None): a misleading "not a readable head.pt"
])
def test_kev_head_flags_the_server_does_not_implement_refused(tmp_path, meta, match):
    pt = _kev_head(tmp_path / "head.pt", **meta)
    with pytest.raises(ValueError, match=match):
        de.write_kev_profile(pt, str(tmp_path / "tree"), hidden=8)
    assert not (tmp_path / "tree").exists()


@pytest.mark.parametrize("meta,match", [
    ({}, "head tensors None"),
    ({"head": None}, "head tensors None"),
    ({"head": [1, 2]}, r"head tensors \[1, 2\]"),
    ({"head": dict.fromkeys(de.KEV_HEAD_TENSORS), "head_dim": "4"}, "head_dim '4' .an integer"),
    ({"head": dict.fromkeys(de.KEV_HEAD_TENSORS), "head_dim": True}, "head_dim True .an integer"),
])
def test_kev_head_without_a_usable_head_is_a_refusal_not_a_crash(meta, match):
    # torch-free: kev_head_unsupported only reads the dict, so CI covers it without torch
    (msg,) = de.kev_head_unsupported(meta)
    assert re.search(match, msg), msg


def test_cli_kev_head_pt_without_a_head_is_a_clean_error(cli, capsys):
    run, calls, base, tmp = cli
    kd = _kev_dir(tmp)
    pt = _kev_head(kd / "head.pt", head=None)
    with pytest.raises(SystemExit) as e:
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", pt, "--out", str(tmp / "t"))
    assert e.value.code == 2 and "head tensors None" in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()


def test_kev_head_extra_head_tensor_refused(tmp_path):
    torch = pytest.importorskip("torch")
    head = {"q.weight": torch.ones(4, 8), "q.bias": torch.zeros(4), "k.weight": torch.ones(4, 8),
            "k.bias": torch.zeros(4), "gate.weight": torch.ones(1, 8)}
    pt = _kev_head(tmp_path / "head.pt", head=head)
    with pytest.raises(ValueError, match="gate.weight"):
        de.write_kev_profile(pt, str(tmp_path / "tree"), hidden=8)
    assert not (tmp_path / "tree").exists()


@pytest.mark.parametrize("flag", ["option_isolation", "special_embeddings"])
def test_cli_kev_unsupported_head_flag_is_a_clean_error(cli, capsys, flag):
    run, calls, base, tmp = cli
    kd = _kev_dir(tmp)
    pt = _kev_head(kd / "head.pt", **{flag: True})
    with pytest.raises(SystemExit) as e:
        run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors",
            "--head-pt", pt, "--out", str(tmp / "t"))
    assert e.value.code == 2
    assert f"{flag}=True" in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()


# Task 30 (limits-audit L6): --max-tokens for every family, capped like JEV by the checkpoint's
# max_position_embeddings; calibration bounds mirrored from DecisionProfile::load.

def test_max_tokens_defaults_are_capped_by_max_position_embeddings():
    assert de.resolve_max_tokens("jev", None, None) == 32768
    assert de.resolve_max_tokens("jev", None, 262144) == 32768
    assert de.resolve_max_tokens("jev", None, 4096) == 4096
    assert de.resolve_max_tokens("kev", None, None) == 73728
    assert de.resolve_max_tokens("kev", None, 262144) == 73728
    assert de.resolve_max_tokens("kev", None, 40960) == 40960
    assert de.resolve_max_tokens("clef", None, None) == 16384
    assert de.resolve_max_tokens("clef", None, 8192) == 8192


@pytest.mark.parametrize("family", ["jev", "kev", "clef"])
def test_max_tokens_override_is_used_as_given(family):
    assert de.resolve_max_tokens(family, 2048, 262144) == 2048
    assert de.resolve_max_tokens(family, 2048, None) == 2048
    assert de.resolve_max_tokens(family, 4096, 4096) == 4096


@pytest.mark.parametrize("family", ["jev", "kev", "clef"])
def test_max_tokens_override_above_max_position_embeddings_refused(family):
    with pytest.raises(ValueError, match="--max-tokens 4097 exceeds the max_position_embeddings 4096 of the checkpoint"):
        de.resolve_max_tokens(family, 4097, 4096)


@pytest.mark.parametrize("family,cap,name", [("jev", 131072, "MAX_DECISION_PROMPT"),
                                             ("kev", 131072, "MAX_DECISION_PROMPT"),
                                             ("clef", 16384, "MAX_JOINT_PROMPT")])
def test_max_tokens_override_above_the_family_hard_cap_refused(family, cap, name):
    assert de.resolve_max_tokens(family, cap, None) == cap
    with pytest.raises(ValueError, match=f"--max-tokens {cap + 1} exceeds {family}'s {cap} \\({name}"):
        de.resolve_max_tokens(family, cap + 1, None)
    with pytest.raises(ValueError, match=name):
        de.resolve_max_tokens(family, cap + 1, 10 ** 7)


def _seen_writers(monkeypatch, seen):
    monkeypatch.setattr(de, "write_jev_profile", lambda ad, cal, out, max_tokens, tokenizer=None: seen.update(mt=max_tokens))
    monkeypatch.setattr(de, "kev_head_meta", lambda pt: {"base": None, "weights": None})
    monkeypatch.setattr(de, "write_kev_profile",
                        lambda pt, out, hidden, max_tokens=None: seen.update(mt=max_tokens))
    monkeypatch.setattr(de, "write_clef_profile", lambda cd, out, max_tokens=None: seen.update(mt=max_tokens))


def _family_argv(family, base, tmp):
    return {"jev": ["--adapter-dir", str(tmp / "jev" / "adapter_vllm")],
            "kev": ["--weights", "W", "--lora", "L.safetensors", "--head-pt", str(_kev_dir(tmp) / "head.pt")],
            "clef": ["--clef-dir", "C"]}[family] + ["--base-ir", str(base), "--out", str(tmp / "t")]


@pytest.mark.parametrize("family,default", [("jev", 32768), ("kev", 73728), ("clef", 16384)])
def test_cli_max_tokens_reaches_every_family(cli, monkeypatch, family, default):
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    run(family, *_family_argv(family, base, tmp))
    assert seen == {"mt": default}  # base config's 262144 does not bind
    run(family, *_family_argv(family, base, tmp), "--max-tokens", "1024")
    assert seen == {"mt": 1024}


def test_cli_kev_default_capped_by_the_weights_config(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    w = tmp / "ft"
    w.mkdir()
    (w / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 40960}}))
    run("kev", "--base-ir", str(base), "--weights", str(w), "--lora", "L.safetensors",
        "--head-pt", str(_kev_dir(tmp) / "head.pt"), "--out", str(tmp / "t"))
    assert seen == {"mt": 40960}


def test_cli_clef_default_capped_by_the_clef_dir_config(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    cd = tmp / "clef"
    cd.mkdir()
    (cd / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    run("clef", "--base-ir", str(base), "--clef-dir", str(cd), "--out", str(tmp / "t"))
    assert seen == {"mt": 8192}


@pytest.mark.parametrize("family", ["jev", "kev", "clef"])
def test_cli_max_tokens_above_max_position_embeddings_is_a_clean_error(cli, monkeypatch, capsys, family):
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    (base / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 4096, "hidden_size": 8}}))
    out = tmp / "t"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        run(family, *_family_argv(family, base, tmp), "--max-tokens", "8192")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--max-tokens 8192" in err and "max_position_embeddings 4096" in err
    # refused before anything is touched: no writer ran, the earlier tree is left as it was
    assert seen == {} and calls == [] and (out / "manifest.json").exists()


def test_cli_clef_max_tokens_above_the_joint_cap_is_a_clean_error(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    with pytest.raises(SystemExit) as e:
        run("clef", *_family_argv("clef", base, tmp), "--max-tokens", "16385")
    assert e.value.code == 2 and "MAX_JOINT_PROMPT" in capsys.readouterr().err
    assert seen == {} and calls == []


def test_clef_cap_follows_the_model_width_like_joint_prompt_cap():
    # T30 M1: clef.rs clamps max_tokens to joint_prompt_cap(hidden); refuse here instead
    assert [de.joint_prompt_cap(h) for h in (0, 4096, 5120, 6144, 8192)] == [16384, 16384, 16384, 13653, 10240]
    assert de.resolve_max_tokens("clef", None, None, hidden=5120) == 16384
    assert de.resolve_max_tokens("clef", None, None, hidden=8192) == 10240
    assert de.resolve_max_tokens("clef", None, 8192, hidden=6144) == 8192
    assert de.resolve_max_tokens("clef", 10240, None, hidden=8192) == 10240
    with pytest.raises(ValueError, match=r"--max-tokens 10241 exceeds clef's 10240 \(joint_prompt_cap\(8192\)"):
        de.resolve_max_tokens("clef", 10241, None, hidden=8192)
    # the width never touches jev/kev
    assert de.resolve_max_tokens("kev", None, None, hidden=8192) == 73728


def test_cli_clef_cap_reads_the_base_ir_width(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    (base / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144, "hidden_size": 8192}}))
    run("clef", *_family_argv("clef", base, tmp))
    assert seen == {"mt": 10240}
    seen.clear()
    with pytest.raises(SystemExit) as e:
        run("clef", *_family_argv("clef", base, tmp), "--max-tokens", "16384")
    assert e.value.code == 2 and "joint_prompt_cap(8192)" in capsys.readouterr().err
    assert seen == {}


@pytest.mark.parametrize("family,default", [("jev", 32768), ("clef", 16384)])
def test_cli_without_any_config_json_uses_the_family_default(cli, monkeypatch, family, default):
    # T30 M4: neither --weights/--clef-dir nor the base IR has config.json: no window is known,
    # so the family default is written (this used to be a FileNotFoundError)
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    (base / "config.json").unlink()
    run(family, *_family_argv(family, base, tmp))
    assert seen == {"mt": default} and [c[0] for c in calls] == ["export"]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "lots", ""])
def test_cli_max_tokens_must_be_a_positive_integer(cli, monkeypatch, capsys, value):
    run, calls, base, tmp = cli
    seen = {}
    _seen_writers(monkeypatch, seen)
    with pytest.raises(SystemExit) as e:
        run("kev", *_family_argv("kev", base, tmp), f"--max-tokens={value}")
    assert e.value.code == 2 and "--max-tokens" in capsys.readouterr().err
    assert seen == {} and calls == []


@pytest.mark.parametrize("per_kind,field", [
    ({"noul": 1.0, "choice": 0.0009, "score": 1.0}, "jev.temperature.choice"),
    ({"noul": 1.0, "choice": 1.0, "score": 1000.5}, "jev.temperature.score"),
    ({"noul": 0, "choice": 1.0, "score": 1.0}, "jev.temperature.noul"),
    ({"noul": float("nan"), "choice": 1.0, "score": 1.0}, "jev.temperature.noul"),
    ({"noul": "1.0", "choice": 1.0, "score": 1.0}, "jev.temperature.noul"),
])
def test_jev_temperature_outside_the_server_bounds_refused(tmp_path, per_kind, field):
    adapter, cal = _jev_adapter(tmp_path, per_kind=per_kind)
    out = tmp_path / "tree"
    with pytest.raises(ValueError, match=f"{field} must be in \\[0.001, 1000\\]"):
        de.write_jev_profile(str(adapter), str(cal), str(out))
    assert not (out / "decision.json").exists()


@pytest.mark.parametrize("bias", [10000.5, -10001, float("inf"), float("nan")])
def test_jev_bias_outside_the_server_bounds_refused(tmp_path, bias):
    adapter, cal = _jev_adapter(tmp_path)
    h = json.loads((adapter / "decision_head.json").read_text())
    h["bias"][3] = bias
    (adapter / "decision_head.json").write_text(json.dumps(h))
    out = tmp_path / "tree"
    with pytest.raises(ValueError, match=r"jev.bias\[3\] must be in \[-10000, 10000\]"):
        de.write_jev_profile(str(adapter), str(cal), str(out))
    assert not (out / "decision.json").exists()


def test_jev_calibration_at_the_server_bounds_exports(tmp_path):
    adapter, cal = _jev_adapter(tmp_path, per_kind={"noul": 0.001, "choice": 1000, "score": 1.0})
    h = json.loads((adapter / "decision_head.json").read_text())
    h["bias"][0], h["bias"][1] = 10000, -10000
    (adapter / "decision_head.json").write_text(json.dumps(h))
    de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))
    assert json.loads((tmp_path / "tree" / "decision.json").read_text())["jev"]["bias"][:2] == [10000, -10000]


@pytest.mark.parametrize("t", [0.0009, 1001.0, 0.0])
def test_kev_temperature_outside_the_server_bounds_refused_before_any_asset(tmp_path, t):
    pt = _kev_head(tmp_path / "head.pt", temperature=t)
    with pytest.raises(ValueError, match=r"kev.temperature must be in \[0.001, 1000\]"):
        de.write_kev_profile(pt, str(tmp_path / "tree"), hidden=8)
    assert not (tmp_path / "tree").exists()
    for ok in (0.001, 1000.0):
        de.write_kev_profile(_kev_head(tmp_path / "ok.pt", temperature=ok), str(tmp_path / "ok"), hidden=8)


def _rust_consts(rel, root=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)):
    """`pub` (or `pub(crate)` ...) consts of `rel` whose value is a number or a product of numbers
    (`320 * 1024 * 1024`): a reformatted constant still parses, only a changed value fails."""
    with open(os.path.join(root, rel)) as f:
        src = f.read()
    out = {}
    for k, expr in re.findall(r"pub(?:\([\w:]+\))? const (\w+): \w+ = ([0-9_.e+\- *]+);", src):
        v = 1.0
        for f in expr.split("*"):
            v *= float(f.strip().replace("_", ""))
        out[k] = v
    return out


def test_rust_consts_parser_reads_products_and_restricted_visibility(tmp_path):
    (tmp_path / "x.rs").write_text("pub const A: usize = 16_384;\npub(crate) const B: usize = 16 * 1024;\n"
                                   "pub const C: f64 = 1e-3;\npub const D: usize = 320 * 1024 * 1024;\n")
    assert _rust_consts("x.rs", root=str(tmp_path)) == {"A": 16384, "B": 16384, "C": 1e-3, "D": 320 * 1024 * 1024}


def test_bounds_mirror_the_rust_constants():
    """The exporter refuses what the server would: these must stay equal to the Rust constants."""
    prof = _rust_consts("crates/cascadia-api/src/decisions/profile.rs")
    assert (de.MIN_TEMPERATURE, de.MAX_TEMPERATURE, de.MAX_ABS_BIAS) == \
        (prof["MIN_TEMPERATURE"], prof["MAX_TEMPERATURE"], prof["MAX_ABS_BIAS"])
    # Task 36: the server refuses a profile with more choice labels than the exporter writes.
    assert de.MAX_CHOICE_LABELS == prof["MAX_CHOICE_LABELS"]
    types = _rust_consts("crates/cascadia-types/src/decision.rs")
    assert (de.MAX_DECISION_PROMPT, de.MAX_JOINT_PROMPT, de.MAX_HEAD_INPUT_BYTES) == \
        (types["MAX_DECISION_PROMPT"], types["MAX_JOINT_PROMPT"], types["MAX_HEAD_INPUT_BYTES"])


# Task 35: JEV's extended single-token choice labels (serve_decide.py single_token_labels).

class _FakeTok:
    """Greedy longest-match tokenizer over `vocab` (str -> id); characters outside it get 1000+ord."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.longest = max(map(len, vocab))

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False  # the reference encodes without BOS/EOS
        ids, i = [], 0
        while i < len(text):
            for n in range(min(self.longest, len(text) - i), 0, -1):
                if text[i:i + n] in self.vocab:
                    ids.append(self.vocab[text[i:i + n]])
                    i += n
                    break
            else:
                ids.append(1000 + ord(text[i]))
                i += 1
        return ids


_LETTERS = [chr(65 + i) for i in range(26)]


def _jev_tok(extra=None, trained=range(8, 24)):
    """A..P are ids `trained` (the _jev_adapter choice ids); Q..Z follow from 500."""
    vocab = {lab: i for lab, i in zip(_LETTERS, list(trained) + list(range(500, 526)))}
    vocab.update(extra or {})
    return _FakeTok(vocab)


def test_single_token_labels_skip_multi_token_labels():
    # (a) AB is two tokens (A, B), AC is one: AB skipped, AC kept
    labels = de.single_token_labels(_jev_tok({"AC": 900}))
    assert labels[:26] == list(zip(_LETTERS, list(range(8, 24)) + list(range(500, 510))))
    assert labels[26:] == [("AC", 900)]


def test_single_token_labels_skip_a_label_the_option_line_merges():
    # (b) "Q)" merges into one token in "x\nQ) y": Q's own token never reaches the prompt
    labels = de.single_token_labels(_jev_tok({"Q)": 901}))
    assert "Q" not in [lab for lab, _ in labels]
    assert [lab for lab, _ in labels][:17] == _LETTERS[:16] + ["R"]


def test_single_token_labels_cap_at_256():
    # (d) every one- and two-letter label is one token: 702 candidates, the first 256 kept
    two = {a + b: 2000 + 26 * i + j for i, a in enumerate(_LETTERS) for j, b in enumerate(_LETTERS)}
    labels = de.single_token_labels(_jev_tok(two))
    assert len(labels) == de.MAX_CHOICE_LABELS == 256
    assert [lab for lab, _ in labels[:28]] == _LETTERS + ["AA", "AB"]  # outer letter first
    assert labels[-1][0] == "IV"  # 26 + 8*26 + 22


def test_jev_profile_writes_extended_choice_labels(tmp_path):
    adapter, cal = _jev_adapter(tmp_path)
    out = tmp_path / "tree"
    de.write_jev_profile(str(adapter), str(cal), str(out), tokenizer=_jev_tok({"AC": 900}))
    jev = json.loads((out / "decision.json").read_text())["jev"]
    assert jev["choice_labels"] == _LETTERS + ["AC"]
    assert jev["choice_label_ids"] == list(range(8, 24)) + list(range(500, 510)) + [900]
    lo, hi = jev["slots"]["choice"]
    assert jev["choice_label_ids"][:hi - lo] == jev["verbalizer_ids"][lo:hi]


def test_jev_profile_without_tokenizer_has_no_label_fields(tmp_path):
    # old trees (and Kev/Clef) carry neither field; the server reads absence as trained slots only
    adapter, cal = _jev_adapter(tmp_path)
    de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"))
    jev = json.loads((tmp_path / "tree" / "decision.json").read_text())["jev"]
    assert "choice_labels" not in jev and "choice_label_ids" not in jev


def test_jev_labels_not_matching_the_trained_ids_refused(tmp_path):
    # (c) the tokenizer's first 16 labels are not the trained choice ids: wrong tokenizer.json
    adapter, cal = _jev_adapter(tmp_path)
    with pytest.raises(ValueError) as e:
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"), tokenizer=_jev_tok(trained=range(30, 46)))
    msg = str(e.value)
    assert str(list(range(30, 46))) in msg and str(list(range(8, 24))) in msg  # both lists named
    assert not (tmp_path / "tree").exists()


def test_jev_too_few_labels_for_the_trained_slots_refused(tmp_path):
    # "\nP" merges in the option line: only 15 of the 16 trained labels survive
    adapter, cal = _jev_adapter(tmp_path)
    with pytest.raises(ValueError, match="verbalizer_ids"):
        de.write_jev_profile(str(adapter), str(cal), str(tmp_path / "tree"), tokenizer=_jev_tok({"\nP": 902}))
    assert not (tmp_path / "tree").exists()


def test_load_tokenizer_encodes_without_special_tokens(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, processors
    t = tokenizers.Tokenizer(models.WordLevel({"<s>": 0, "A": 1, "AB": 2, "[UNK]": 3}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    t.save(str(tmp_path / "tokenizer.json"))
    tok = de.load_tokenizer(str(tmp_path / "tokenizer.json"))
    assert tok.encode("A AB", add_special_tokens=False) == [1, 2]


def test_cli_jev_labels_from_the_tokenizer_the_tree_ships(cli, monkeypatch):
    run, calls, base, tmp = cli
    adapter, _ = _jev_adapter(tmp)
    run("jev", "--base-ir", str(base), "--adapter-dir", str(adapter), "--out", str(tmp / "t"))
    assert run.tokenizers == [str(base / "tokenizer.json")]
    jev = json.loads((tmp / "t" / "decision.json").read_text())["jev"]
    assert jev["choice_labels"] == _LETTERS and jev["choice_label_ids"][:16] == list(range(8, 24))
    # --weights' tokenizer.json wins over the IR's, as run_export copies it; --tokenizer over both
    w = tmp / "ft"
    w.mkdir()
    (w / "tokenizer.json").write_text("ft tokenizer")
    run("jev", "--base-ir", str(base), "--weights", str(w), "--weights-not-merged", "--adapter-dir", str(adapter),
        "--out", str(tmp / "t2"))
    other = tmp / "other.json"
    other.write_text("t")
    run("jev", "--base-ir", str(base), "--weights", str(w), "--weights-not-merged", "--adapter-dir", str(adapter),
        "--tokenizer", str(other), "--out", str(tmp / "t3"))
    assert run.tokenizers[1:] == [str(w / "tokenizer.json"), str(other)]


def test_cli_jev_without_a_tokenizer_refused_before_any_write(cli, capsys):
    run, calls, base, tmp = cli
    (base / "tokenizer.json").unlink()
    adapter, _ = _jev_adapter(tmp)
    with pytest.raises(SystemExit) as e:
        run("jev", "--base-ir", str(base), "--adapter-dir", str(adapter), "--out", str(tmp / "t"))
    assert e.value.code == 2 and "tokenizer.json" in capsys.readouterr().err
    assert calls == [] and not (tmp / "t").exists()


# ---------------------------------------------------------------------------
# Task 58: single-box trees are one stage; the export's peak RAM is checked first
# ---------------------------------------------------------------------------

GiB = 1 << 30
# The int8 IRs' weights on pawan-01 (language model + text embeddings bins), and its RAM.
IR_9B = (7.398 + 0.948) * GiB
IR_27B = (23.88 + 1.185) * GiB
PAWAN01_RAM = 31.58 * GiB


def test_ir_weight_bytes_sums_the_language_model_and_embeddings_bins(tmp_path):
    (tmp_path / "openvino_language_model.bin").write_bytes(b"x" * 300)
    (tmp_path / "openvino_text_embeddings_model.bin").write_bytes(b"x" * 20)
    (tmp_path / "openvino_vision_embeddings_model.bin").write_bytes(b"x" * 5000)  # not exported
    assert de.ir_weight_bytes(str(tmp_path)) == 320
    assert de.ir_weight_bytes(str(tmp_path / "missing")) == 0


@pytest.mark.parametrize("stages,swap,measured", [
    (1, False, 8.25), (1, True, 20.3), (1, True, 20.6),  # GiB. JEV graft; Kev, Clef --weights swap
    (8, False, 3.1), (8, True, 5.1), (8, True, 5.3),
])
def test_peak_estimate_covers_the_measured_peaks(monkeypatch, stages, swap, measured):
    # pawan-01, OpenVINO 2026.4.1: exporter process-tree working set, 1 vs 8 stages. The estimate
    # may overshoot (a refusal names --skip-memory-check) but must not undershoot by much.
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    est = de.export_peak_bytes("ir", stages, swap) / GiB
    assert measured <= est <= measured * 1.6, est


def test_the_documented_27b_two_stage_exports_fit_a_32_gb_host(monkeypatch):
    # Not measured by the 9B calibration: Clef 27B (--weights) at --stages 2 estimates 31.3 GiB,
    # just under pawan-01's 31.58 GiB; JEV 27B (graft) 12.5 GiB.
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_27B)
    assert de.export_peak_bytes("ir", 2, True) / GiB == pytest.approx(31.33, abs=0.01)
    assert de.memory_refusal("ir", 2, True, PAWAN01_RAM) is None
    assert de.memory_refusal("ir", 2, False, PAWAN01_RAM) is None
    assert de.memory_refusal("ir", 1, True, PAWAN01_RAM) is not None


def test_peak_estimate_shrinks_with_the_stage_count(monkeypatch):
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    peaks = [de.export_peak_bytes("ir", n, True) for n in (1, 2, 4, 8)]
    assert peaks == sorted(peaks, reverse=True) and peaks[0] == int(IR_9B * 2.5)


def test_physical_ram_is_read_on_this_machine():
    ram = de.physical_ram_bytes()
    assert isinstance(ram, int) and ram > 2**30


def test_memory_refusal_names_the_estimate_the_ram_and_the_fallback(monkeypatch):
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    why = de.memory_refusal("ir", 1, True, 16 * GiB)
    assert why.startswith("the export needs about 20.9 GiB of RAM at --stages 1 (the base IR's 8.3 GiB "
                          "of int8 weights, x2.5 for the --weights swap), but this machine has 16.0 GiB")
    assert "--stages 8" in why and "--skip-memory-check" in why
    assert de.memory_refusal("ir", 1, True, PAWAN01_RAM) is None
    assert de.memory_refusal("ir", 1, False, 16 * GiB) is None  # JEV's graft: about 8.3 GiB
    assert de.memory_refusal("ir", 1, True, None) is None  # RAM unknown: not refused


def _clef(monkeypatch):
    monkeypatch.setattr(de, "write_clef_profile",
                        lambda cd, out, max_tokens=None: de._write(out, {"family": "clef"}, {"kind": "joint"}))


def test_cli_refuses_an_export_larger_than_ram_before_any_write(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    monkeypatch.setattr(de, "physical_ram_bytes", lambda: int(16 * GiB))
    monkeypatch.setattr(de, "write_clef_profile", lambda *a, **kw: pytest.fail("clef profile written"))
    out = tmp / "t"
    out.mkdir()
    (out / "manifest.json").write_text("{}")  # an earlier tree stays as it was
    with pytest.raises(SystemExit) as e:
        run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(out))
    err = capsys.readouterr().err
    assert e.value.code == 2
    assert "needs about 20.9 GiB of RAM at --stages 1" in err and "this machine has 16.0 GiB" in err
    assert "--stages 8" in err
    assert calls == [] and (out / "manifest.json").exists() and not (out / "decision.json").exists()


def test_cli_exports_when_the_estimate_fits(cli, monkeypatch):
    run, calls, base, tmp = cli
    _clef(monkeypatch)
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    monkeypatch.setattr(de, "physical_ram_bytes", lambda: int(PAWAN01_RAM))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"))
    (_, a, _, _), = calls
    assert a[2] == 1


def test_cli_skip_memory_check_overrides_the_refusal(cli, monkeypatch):
    run, calls, base, tmp = cli
    _clef(monkeypatch)
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    monkeypatch.setattr(de, "physical_ram_bytes", lambda: pytest.fail("RAM read with the check skipped"))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"), "--skip-memory-check")
    (_, a, _, _), = calls
    assert a[2] == 1


@pytest.mark.parametrize("stages", [2, 3, 8])
def test_cli_stages_n_reaches_run_export_and_shrinks_the_estimate(cli, monkeypatch, stages):
    # A pipeline passes one stage per rank, and --stages 8 is the low-memory single box: both
    # unchanged, and each fits where the 1-stage export (20.9 GiB) does not.
    run, calls, base, tmp = cli
    _clef(monkeypatch)
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    monkeypatch.setattr(de, "physical_ram_bytes", lambda: int(16 * GiB))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"), "--stages", str(stages))
    (_, a, _, _), = calls
    assert a[2] == stages


def test_cli_jev_graft_is_estimated_without_the_swap_factor(cli, monkeypatch):
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    monkeypatch.setattr(de, "physical_ram_bytes", lambda: int(16 * GiB))
    monkeypatch.setattr(de, "write_jev_profile",
                        lambda ad, cal, out, max_tokens, tokenizer=None: de._write(out, {"family": "jev"}, {"kind": "verbalizer"}))
    run("jev", "--base-ir", str(base), "--adapter-dir", str(tmp / "jev" / "adapter_vllm"), "--out", str(tmp / "t"))
    assert len(calls) == 1


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_stages_must_be_positive(cli, capsys, value):
    run, calls, base, tmp = cli
    with pytest.raises(SystemExit) as e:
        run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"), "--stages", value)
    assert e.value.code == 2 and "--stages" in capsys.readouterr().err and calls == []


# Fix round 1: no --stages 8 advice at 8 or more stages; an uneven --layer-split is sized by its
# largest stage.

def _ir_with_layers(tmp_path, layers=32):
    ir = tmp_path / "ir"
    ir.mkdir()
    (ir / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": layers}}))
    return str(ir)


@pytest.mark.parametrize("stages", [8, 16])
def test_refusal_at_8_or_more_stages_does_not_suggest_stages_8(monkeypatch, stages):
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    why = de.memory_refusal("ir", stages, True, 4 * GiB)
    assert why.startswith(f"the export needs about 7.9 GiB of RAM at --stages {stages} ")
    fix = why.split("GiB: ", 1)[1]
    assert "--stages 8" not in fix
    assert f"more stages than {stages}" in fix and "--skip-memory-check" in why


def test_refusal_below_8_stages_suggests_stages_8(monkeypatch):
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    assert "export with --stages 8 instead" in de.memory_refusal("ir", 2, True, 4 * GiB)


def test_largest_stage_share_follows_the_layer_split(tmp_path):
    ir = _ir_with_layers(tmp_path)
    assert de.largest_stage_share(ir, 2) == 0.5
    assert de.largest_stage_share(ir, 2, "24") == 0.75  # layers 0-23 | 24-31
    assert de.largest_stage_share(ir, 8, "4,8,12,16,20,24,28") == 0.125
    assert de.largest_stage_share(ir, 3) == 12 / 32  # 10 + 10 + 12: the last stage takes the rest
    # No config.json or an invalid split: 1/N (run_export refuses a bad split itself)
    assert de.largest_stage_share(str(tmp_path / "none"), 4) == 0.25
    assert de.largest_stage_share(ir, 2, "40") == 0.5


def test_uneven_layer_split_is_estimated_from_its_largest_stage(tmp_path, monkeypatch):
    ir = _ir_with_layers(tmp_path)
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    even = de.export_peak_bytes(ir, 2, True)
    uneven = de.export_peak_bytes(ir, 2, True, "24")
    assert uneven == int(IR_9B * 2.5 * 0.75) and uneven > even == int(IR_9B * 2.5 * 0.5)
    why = de.memory_refusal(ir, 2, True, 14 * GiB, "24")
    assert why.startswith("the export needs about 15.6 GiB of RAM at --stages 2 --layer-split 24 ")
    assert de.memory_refusal(ir, 2, True, 14 * GiB) is None  # the even split fits (10.4 GiB)


def test_cli_passes_the_layer_split_to_the_check(cli, monkeypatch, capsys):
    run, calls, base, tmp = cli
    (base / "config.json").write_text(json.dumps({"text_config": {
        "max_position_embeddings": 262144, "hidden_size": 8, "num_hidden_layers": 32}}))
    monkeypatch.setattr(de, "ir_weight_bytes", lambda d: IR_9B)
    monkeypatch.setattr(de, "physical_ram_bytes", lambda: int(14 * GiB))
    with pytest.raises(SystemExit):
        run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"),
            "--stages", "2", "--layer-split", "24")
    assert "--layer-split 24" in capsys.readouterr().err and calls == []
