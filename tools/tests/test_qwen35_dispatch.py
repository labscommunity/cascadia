"""Unit tests for the Qwen3.5-family OpenVINO-IR dispatch in export_shards.

Run from the repo root with:

    python -m pytest tools/tests/test_qwen35_dispatch.py -v

Both `qwen3_5_moe` (Qwen3.5/3.6) and dense `qwen3_5` (Qwen3.8) must route
config-first to the IR-surgery exporter; the `--layer-split` guard fires
BEFORE the surgery module is imported, so a stub dir with a config.json and
an empty openvino_language_model.xml is enough (no openvino needed). A
dense model that slipped past the dispatch would otherwise be silently
mis-exported through the generic Qwen3 path ("qwen3_5" contains "qwen3").
"""

from __future__ import annotations

import json
import os
import sys

import pytest

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


def _stub_ir(dirpath, model_type, inner=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": model_type}
    if inner:
        cfg["text_config"] = {"model_type": inner}
    (dirpath / "config.json").write_text(json.dumps(cfg))
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


@pytest.mark.parametrize(
    "outer,inner",
    [
        ("qwen3_5", "qwen3_5_text"),  # Qwen3.8-27B (dense)
        ("qwen3_5_moe", "qwen3_5_moe_text"),  # Qwen3.6-35B-A3B
    ],
)
def test_qwen35_family_dispatches_to_surgery(tmp_path, monkeypatch, capsys, outer, inner):
    model = _stub_ir(tmp_path / "ir", outer, inner)
    monkeypatch.setattr(
        sys, "argv", _argv(model, tmp_path / "out", "--layer-split", "8"))
    with pytest.raises(SystemExit) as excinfo:
        export_shards.main()
    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "IR-surgery" in out, out
    assert outer in out, out
    assert "--layer-split" in out, out
