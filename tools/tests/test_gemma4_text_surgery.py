"""Unit tests for the pure seams of the gemma-4 IR-surgery exporter.

Run from the repo root with:

    python -m pytest tools/tests/test_gemma4_text_surgery.py -v

export_gemma4_text imports openvino at import time; an env without it skips
the whole file. The surgery itself needs real IRs and is gated on-node via
--validate — these tests pin only the pure helpers whose contracts have
bitten before: the inclusive stage_ranges vs half-open layer_end write, and
the sink-attribution regexes (the key_values-substring trap).
"""

from __future__ import annotations

import os
import sys

import pytest

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SURGERY_DIR = os.path.join(_TOOLS_DIR, "gemma4_surgery")
for _d in (_TOOLS_DIR, _SURGERY_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

pytest.importorskip(
    "openvino", reason="export_gemma4_text imports openvino at import time")

import export_gemma4_text  # noqa: E402


# --- stage_ranges: inclusive tuples feeding the half-open layer_end write ---

def test_stage_ranges_inclusive_contiguous_full_cover():
    assert export_gemma4_text.stage_ranges(4, 48) == [
        (0, 11), (12, 23), (24, 35), (36, 47)]


def test_stage_ranges_remainder_folds_into_last_stage():
    r = export_gemma4_text.stage_ranges(4, 50)
    assert r[0] == (0, 11)
    assert r[-1] == (36, 49)


@pytest.mark.parametrize(
    "total,num_layers", [(1, 48), (2, 32), (3, 32), (4, 50), (5, 62)])
def test_stage_ranges_half_open_sum_equals_num_layers(total, num_layers):
    r = export_gemma4_text.stage_ranges(total, num_layers)
    assert r[0][0] == 0
    assert r[-1][1] == num_layers - 1
    assert all(r[i + 1][0] == r[i][1] + 1 for i in range(len(r) - 1))
    # stage_config.json writes the HALF-OPEN layer_end = b + 1 and
    # cascadia-types computes num_layers = layer_end - layer_start, so the
    # per-stage (b + 1) - a must sum to the model's total. If stage_ranges
    # ever returns half-open ends, the b + 1 write double-counts and this
    # catches it (this exact off-by-one shipped mid-PR once).
    assert sum((b + 1) - a for a, b in r) == num_layers


# --- sink attribution: variable_id parsing --------------------------------

@pytest.mark.parametrize(
    "vid,layer,kind",
    [
        ("past_key_values.3.value", 3, "value"),
        ("past_key_values.17.key", 17, "key"),
        ("present.0.key", 0, "key"),
        ("model.layers.5.conv_cache", 5, "conv"),
    ],
)
def test_sink_attribution(vid, layer, kind):
    assert export_gemma4_text.sink_layer_index(vid) == layer
    assert export_gemma4_text.sink_kind(vid) == kind


def test_kind_never_misreads_key_values_substring():
    # EVERY KV variable_id contains the literal "key_values"; the kind must
    # come from the delimited token AFTER the layer index, so this is a
    # value cache, not a key cache.
    assert export_gemma4_text.sink_kind("past_key_values.0.value") == "value"


def test_layer_index_digit_fallback():
    assert export_gemma4_text.sink_layer_index("some_cache_7") == 7
    assert export_gemma4_text.sink_layer_index("no_digits_here") is None


# --- run_export input guards (fire before any OV work) ---------------------

def test_run_export_rejects_model_equals_output_dir(tmp_path):
    # flatten_config / regenerate_tokenizer_bos rewrite files in place:
    # losing this guard means the tool destroys the user's source IR.
    d = tmp_path / "ir"
    d.mkdir()
    with pytest.raises(SystemExit, match="output-dir"):
        export_gemma4_text.run_export(
            model=str(d), output_dir=str(d), num_stages=1)


def test_run_export_rejects_nonpositive_num_stages(tmp_path):
    with pytest.raises(SystemExit, match="num-stages"):
        export_gemma4_text.run_export(
            model=str(tmp_path / "ir"), output_dir=str(tmp_path / "out"),
            num_stages=0)
