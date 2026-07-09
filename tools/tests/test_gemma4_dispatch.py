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
