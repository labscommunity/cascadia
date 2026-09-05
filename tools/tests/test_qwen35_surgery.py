"""Unit tests for the Qwen3.5-family surgery exporter's shape helpers.

Run from the repo root with:

    python -m pytest tools/tests/test_qwen35_surgery.py -v

`export_qwen36_moe.py` imports openvino lazily inside the functions that
touch IRs, so the pure helpers under test here (config → ModelSpec, the
layer-type-sequenced state-variable ids, stage ranges) need only numpy.
They pin the invariants the 27B dense export relies on: hidden / layer
count come from config.json, and the `past.{conv,ssm}.N` / `past.{key,
value}.N` numbering walks `layer_types` rather than a 40-layer constant.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_SURGERY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery")
)
if _SURGERY_DIR not in sys.path:
    sys.path.insert(0, _SURGERY_DIR)

pytest.importorskip("numpy")
import export_qwen36_moe as sx  # noqa: E402


def _layer_types(n, interval=4):
    return [sx.FULL if (i + 1) % interval == 0 else sx.LINEAR for i in range(n)]


def _qwen38_cfg():
    """Shape of Qwen/Qwen3.8-27B's config.json (outer VLM wrapper)."""
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "layer_types": _layer_types(64),
            "full_attention_interval": 4,
            "vocab_size": 248320,
        },
    }


def _qwen36_cfg():
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": _layer_types(40),
            "vocab_size": 248320,
        },
    }


# ---------------------------------------------------------------------------
# spec_from_config / read_model_spec
# ---------------------------------------------------------------------------


def test_spec_reads_dense_qwen38_from_nested_text_config():
    spec = sx.spec_from_config(_qwen38_cfg())
    assert spec.model_type == "qwen3_5"
    assert spec.family == "qwen3_5"
    assert (spec.hidden, spec.num_layers, spec.vocab) == (5120, 64, 248320)
    assert spec.layer_types.count(sx.FULL) == 16
    assert spec.layer_types.count(sx.LINEAR) == 48


def test_spec_reads_qwen36_moe():
    spec = sx.spec_from_config(_qwen36_cfg())
    assert spec.model_type == "qwen3_5_moe"
    assert (spec.hidden, spec.num_layers) == (2048, 40)


def test_spec_synthesises_layer_types_from_interval():
    cfg = _qwen38_cfg()
    del cfg["text_config"]["layer_types"]
    spec = sx.spec_from_config(cfg)
    assert spec.layer_types == _layer_types(64)


def test_spec_bare_text_config_strips_text_suffix():
    cfg = _qwen38_cfg()["text_config"]
    spec = sx.spec_from_config(cfg)
    assert spec.model_type == "qwen3_5"
    assert spec.hidden == 5120


@pytest.mark.parametrize("mt", ["llama", "qwen3", "qwen3_moe", "gemma4"])
def test_spec_rejects_non_family(mt):
    with pytest.raises(ValueError, match="qwen3_5"):
        sx.spec_from_config({"model_type": mt, "hidden_size": 8, "num_hidden_layers": 4})


def test_spec_rejects_unknown_layer_type():
    cfg = _qwen38_cfg()
    cfg["text_config"]["layer_types"][0] = "sliding_attention"
    with pytest.raises(ValueError, match="layer_types"):
        sx.spec_from_config(cfg)


def test_read_model_spec_from_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen38_cfg()))
    spec = sx.read_model_spec(str(tmp_path))
    assert spec.hidden == 5120


def test_read_model_spec_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        sx.read_model_spec(str(tmp_path))


# ---------------------------------------------------------------------------
# layer_state_vids — numbered by layer-type sequence
# ---------------------------------------------------------------------------


def test_state_vids_walk_layer_types_on_64_layers():
    lt = _layer_types(64)
    assert sx.layer_state_vids(lt, 0) == ["past.conv.0cache", "past.ssm.0cache"]
    assert sx.layer_state_vids(lt, 3) == ["past.key.0cache", "past.value.0cache"]
    assert sx.layer_state_vids(lt, 4) == ["past.conv.3cache", "past.ssm.3cache"]
    assert sx.layer_state_vids(lt, 62) == ["past.conv.47cache", "past.ssm.47cache"]
    assert sx.layer_state_vids(lt, 63) == ["past.key.15cache", "past.value.15cache"]


def test_state_vids_match_legacy_40_layer_formula():
    """The 35B exporter derived ids arithmetically from the 3:1 interval;
    the layer_types walk must reproduce it exactly (regression)."""
    lt = _layer_types(40)
    for g in range(40):
        if (g + 1) % 4 == 0:
            k = g // 4
            want = [f"past.key.{k}cache", f"past.value.{k}cache"]
        else:
            m = g - (g + 1) // 4
            want = [f"past.conv.{m}cache", f"past.ssm.{m}cache"]
        assert sx.layer_state_vids(lt, g) == want, g


def test_state_vids_cover_every_index_once():
    lt = _layer_types(64)
    seen = [v for g in range(64) for v in sx.layer_state_vids(lt, g)]
    assert len(seen) == len(set(seen)) == 128


# ---------------------------------------------------------------------------
# stage_ranges / check_stage_ranges
# ---------------------------------------------------------------------------


def test_stage_ranges_even_split():
    assert sx.stage_ranges(64, 2) == [(0, 31), (32, 63)]
    assert sx.stage_ranges(40, 2) == [(0, 19), (20, 39)]
    assert sx.stage_ranges(64, 1) == [(0, 63)]


def test_stage_ranges_remainder_goes_to_last_stage():
    assert sx.stage_ranges(64, 3) == [(0, 20), (21, 41), (42, 63)]


def test_stage_ranges_rejects_bad_total():
    with pytest.raises(ValueError):
        sx.stage_ranges(64, 0)
    with pytest.raises(ValueError):
        sx.stage_ranges(64, 65)


def test_check_stage_ranges_requires_an_attention_layer_per_stage():
    spec = sx.spec_from_config(_qwen38_cfg())
    sx.check_stage_ranges(spec, sx.stage_ranges(64, 16))  # one attn layer each
    with pytest.raises(ValueError, match="full_attention"):
        sx.check_stage_ranges(spec, sx.stage_ranges(64, 32))  # 2-layer stages
