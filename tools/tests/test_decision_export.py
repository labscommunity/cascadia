"""Exporter decision options and decision.json/assets writers.

    python -m pytest tools/tests/test_decision_export.py -v
"""
import inspect
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import export_qwen36_moe as ex  # noqa: E402


def test_run_export_accepts_decision_options():
    params = inspect.signature(ex.run_export).parameters
    for name in ("weights", "lora", "lora_scale", "final_hidden"):
        assert name in params, name
    assert params["final_hidden"].default is False


def test_lora_scale_defaults_to_adapter_config():
    assert inspect.signature(ex.run_export).parameters["lora_scale"].default is None


def test_manifest_records_final_hidden():
    spec = ex.ModelSpec(model_type="qwen3_5", hidden=8, num_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    m = ex.build_manifest(spec, "src", final_hidden=True)
    assert m["final_hidden_output"] is True
    assert ex.build_manifest(spec, "src")["final_hidden_output"] is False


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
    monkeypatch.setattr(ws, "lora_scale_from_config", lambda f: cfg_scale)
    monkeypatch.setattr(ws, "lora_targets", lambda f: targets)
    monkeypatch.setattr(ws, "graft_lora", lambda stage, f, s: calls.append(("graft", s)) or per_stage)


def test_chat_export_unchanged_without_decision_flags(stubbed):
    model, out, calls, _, _ = stubbed
    ex.run_export(str(model), str(out), 2)
    assert [c[0] for c in calls] == ["save", "save"]
    m = json.loads((out / "manifest.json").read_text())
    assert m["final_hidden_output"] is False
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


def test_lora_graft_total_must_match_adapter_targets(stubbed, tmp_path):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=3, targets=7)
    out.mkdir()
    (out / "manifest.json").write_text("{}")  # stale manifest from an earlier run in the same dir
    with pytest.raises(SystemExit, match="grafted 6 LoRA branches but the adapter has 7 targets"):
        ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "adapter_model.safetensors"))
    assert not (out / "manifest.json").exists()


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
    _lora(mp, ws, calls, per_stage=3, targets=6, cfg_scale=2.0)
    ex.run_export(str(model), str(out), 2, lora=str(tmp_path / "adapter_model.safetensors"))
    assert [c for c in calls if c[0] == "graft"] == [("graft", 2.0), ("graft", 2.0)]


def test_lora_scale_override_warns(stubbed, tmp_path, capsys):
    model, out, calls, ws, mp = stubbed
    _lora(mp, ws, calls, per_stage=3, targets=6, cfg_scale=2.0)
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
    from decision import clef_head
    ck, lm = _clef_checkpoint(tmp_path, key="model.language_model.embed_tokens.weight", tied=True)
    monkeypatch.setattr(clef_head, "export_joint_head", lambda d, xml: open(xml, "w").close())
    out = tmp_path / "tree"
    de.write_clef_profile(str(ck), str(out))
    assert np.fromfile(out / "decision" / "lm_head.f16", dtype="<f2").tolist() == lm.float().flatten().tolist()


def test_clef_missing_lm_head_refused_without_partial_head_json(tmp_path, monkeypatch):
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
    (base / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["decision_export.py", *argv])
        de.main()
    return run, calls, base, tmp_path


def test_cli_jev_lora_from_adapter_dir_and_no_validate(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "write_jev_profile",
                        lambda ad, cal, out, max_tokens: seen.update(ad=ad, cal=cal, mt=max_tokens)
                        or de._write(out, {"family": "jev"}, {"kind": "verbalizer"}))
    ad = tmp / "jev" / "adapter_vllm"
    run("jev", "--base-ir", str(base), "--adapter-dir", str(ad), "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a == (str(base), str(tmp / "t"), 8, False, None)
    assert kw == {"weights": None, "lora": os.path.join(str(ad), "adapter_model.safetensors"),
                  "lora_scale": None, "final_hidden": False}
    assert assets_first
    assert seen == {"ad": str(ad), "cal": str(tmp / "jev" / "calibration.json"), "mt": 32768}


def test_cli_kev_final_hidden_weights_lora(cli, monkeypatch):
    run, calls, base, tmp = cli
    seen = {}
    monkeypatch.setattr(de, "write_kev_profile",
                        lambda pt, out: seen.update(pt=pt) or de._write(out, {"family": "kev"}, {"kind": "pointer"}))
    run("kev", "--base-ir", str(base), "--weights", "W", "--lora", "L.safetensors", "--lora-scale", "2",
        "--head-pt", "H.pt", "--stages", "2", "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a == (str(base), str(tmp / "t"), 2, False, None)
    assert kw == {"weights": "W", "lora": "L.safetensors", "lora_scale": 2.0, "final_hidden": True}
    assert assets_first and seen == {"pt": "H.pt"}


def test_cli_clef_weights_default_to_clef_dir(cli, monkeypatch):
    run, calls, base, tmp = cli
    monkeypatch.setattr(de, "write_clef_profile",
                        lambda cd, out: de._write(out, {"family": "clef"}, {"kind": "joint"}))
    run("clef", "--base-ir", str(base), "--clef-dir", "C", "--out", str(tmp / "t"))
    (_, a, kw, assets_first), = calls
    assert a[3] is False and kw == {"weights": "C", "lora": None, "lora_scale": None, "final_hidden": True}
    assert assets_first


@pytest.mark.parametrize("family,missing", [("jev", "--adapter-dir"), ("kev", "--head-pt"), ("clef", "--clef-dir")])
def test_cli_family_inputs_required_before_export(cli, family, missing, capsys):
    run, calls, base, tmp = cli
    with pytest.raises(SystemExit):
        run(family, "--base-ir", str(base), "--out", str(tmp / "t"))
    assert missing in capsys.readouterr().err and calls == []
