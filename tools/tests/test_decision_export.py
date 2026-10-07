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
