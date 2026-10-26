"""Unit tests for the gemma-4 OpenVINO-IR dispatch fail-fasts in export_shards.

Run from the repo root with:

    python -m pytest tools/tests/test_gemma4_dispatch.py -v

The dispatch guards (--target npu, --layer-split, unreadable config.json)
fire BEFORE the surgery module is imported, so these tests need neither
openvino nor a real IR — a stub dir with a gemma-4 config.json and an empty
openvino_language_model.xml is enough. They exist because losing a guard is
silent: --target npu would emit a stateful/dynamic shard the NPU compiler
rejects on-device, and --layer-split would be silently replaced by an even
split.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Add tools/ to sys.path so we can import export_shards as a module.
_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

try:
    import export_shards
except ImportError as exc:
    if "torch" in str(exc) or "numpy" in str(exc):
        pytest.skip(
            f"export_shards.py needs torch/numpy at import time: {exc}",
            allow_module_level=True,
        )
    raise


def _stub_gemma4_ir(dirpath):
    """A dir export_shards recognizes as an exported gemma-4 OV VLM IR."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    (dirpath / "openvino_language_model.xml").write_text("")
    return dirpath


@pytest.fixture(autouse=True)
def _restore_torch_default_dtype():
    """`export_shards.main()` sets torch's default dtype (fp16 by default)
    before the config-first dispatch exits; undo it so the rotary tests in
    test_arch_detection.py are not poisoned by test ordering."""
    torch = pytest.importorskip("torch")
    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


def _argv(model, out, *extra):
    return [
        "export_shards.py",
        "--model", str(model),
        "--output-dir", str(out),
        "--num-stages", "2",
        *extra,
    ]


def test_gemma4_ir_dispatch_rejects_npu(tmp_path, monkeypatch):
    model = _stub_gemma4_ir(tmp_path / "ir")
    monkeypatch.setattr(
        sys, "argv", _argv(model, tmp_path / "out", "--target", "npu"))
    with pytest.raises(SystemExit, match="npu"):
        export_shards.main()


def test_gemma4_ir_dispatch_rejects_layer_split(tmp_path, monkeypatch):
    model = _stub_gemma4_ir(tmp_path / "ir")
    monkeypatch.setattr(
        sys, "argv", _argv(model, tmp_path / "out", "--layer-split", "8"))
    with pytest.raises(SystemExit, match="layer-split"):
        export_shards.main()


def test_gemma4_ir_unreadable_config_fails_loudly(tmp_path, monkeypatch):
    # An OV VLM IR dir is only recognizable via config.json: a corrupt one
    # must fail with context, not fall through to the generic/torch path.
    model = tmp_path / "ir"
    model.mkdir()
    (model / "openvino_language_model.xml").write_text("")
    (model / "config.json").write_text("{not json")
    monkeypatch.setattr(sys, "argv", _argv(model, tmp_path / "out"))
    with pytest.raises(SystemExit, match="config.json"):
        export_shards.main()


E2B_TEXT = {"model_type": "gemma4_text", "hidden_size_per_layer_input": 256}
BIG_TEXT = {"model_type": "gemma4_text", "hidden_size_per_layer_input": 0}


def _stub_ir_with(dirpath, text_config):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(
        {"model_type": "gemma4", "text_config": text_config, "vision_config": {}}))
    (dirpath / "openvino_language_model.xml").write_text("")
    return dirpath


def _surgery_calls(monkeypatch):
    import types

    calls = []
    stub = types.ModuleType("export_gemma4_text")
    stub.run_export = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "export_gemma4_text", stub)
    return calls


def test_e2b_e4b_export_is_pointed_at_the_torch_exporter(tmp_path, monkeypatch):
    """The IR surgery cannot cut an E2B / E4B vision tree (per-layer
    embeddings); `shard` names the exporter that can instead of failing in
    the surgery."""
    calls = _surgery_calls(monkeypatch)
    model = _stub_ir_with(tmp_path / "ir", E2B_TEXT)
    monkeypatch.setattr(sys, "argv", _argv(model, tmp_path / "out"))
    with pytest.raises(SystemExit) as e:
        export_shards.main()
    assert "tools/export_gemma4.py" in str(e.value) and "--text-only" in str(e.value)
    assert calls == []


def test_e2b_e4b_text_only_gets_the_text_tree(tmp_path, monkeypatch):
    calls = _surgery_calls(monkeypatch)
    model = _stub_ir_with(tmp_path / "ir", E2B_TEXT)
    monkeypatch.setattr(sys, "argv", _argv(model, tmp_path / "out", "--text-only"))
    export_shards.main()
    assert [c["vision"] for c in calls] == [False]


def test_26b_31b_export_gets_the_vision_tree(tmp_path, monkeypatch):
    calls = _surgery_calls(monkeypatch)
    model = _stub_ir_with(tmp_path / "ir", BIG_TEXT)
    monkeypatch.setattr(sys, "argv", _argv(model, tmp_path / "out"))
    export_shards.main()
    assert [c["vision"] for c in calls] == [True]


@pytest.mark.parametrize("text,needle", [(E2B_TEXT, "tools/export_gemma4.py"),
                                         (BIG_TEXT, "and shard <dir>")])
def test_raw_gemma4_refusal_names_the_working_path(tmp_path, monkeypatch, text, needle):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "config.json").write_text(json.dumps(
        {"model_type": "gemma4", "text_config": text, "vision_config": {}}))
    monkeypatch.setattr(sys, "argv", _argv(raw, tmp_path / "out"))
    with pytest.raises(SystemExit, match="raw Gemma 4") as e:
        export_shards.main()
    assert needle in str(e.value)
