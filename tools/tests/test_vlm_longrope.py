"""longrope.py: the long-RoPE switch for Phi-3 su/longrope language models.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_longrope.py -v

A toy rotary block stands in for optimum's trace (a `[1, rot/2, 1]`
inv_freq constant times the positions); the switch must keep the short
factors while a pass's positions stay under the original window and take the
long ones from it on, as Hugging Face does. The cut and packed guards are
checked on their own.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")
from openvino import opset13 as ops  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_HERE, os.pardir, "vlm_surgery"), os.path.join(_HERE, os.pardir), _HERE):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import longrope  # noqa: E402

SHORT = [1.0, 1.5, 2.0, 2.5]
LONG = [3.0, 4.0, 5.0, 6.0]
TEXT = {"hidden_size": 16, "num_attention_heads": 2, "rope_theta": 10000.0,
        "max_position_embeddings": 64, "original_max_position_embeddings": 16,
        "rope_scaling": {"type": "longrope", "short_factor": SHORT, "long_factor": LONG}}


def inv(factors):
    expo = np.arange(0, 8, 2, dtype=np.float64) / 8
    return (1.0 / (np.asarray(factors) * 10000.0 ** expo)).astype(np.float32)


def toy_rotary():
    """position_ids [1, S] -> freqs [1, 4, S], as optimum traces Phi-3's rotary."""
    pos = ops.parameter([1, -1], ov.Type.i64, name="position_ids")
    pos.output(0).get_tensor().set_names({"position_ids"})
    inv_freq = ops.constant(inv(SHORT).reshape(1, 4, 1))
    p = ops.unsqueeze(ops.convert(pos, ov.Type.f32), ops.constant(np.int64(1)))
    freqs = ops.multiply(inv_freq, p)
    return ov.Model([freqs], [pos], "toy_rotary")


def run(model, positions):
    # f32 inference: ARM CPUs otherwise compute in f16.
    out = ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(
        [np.array([positions], np.int64)])[0]
    return np.asarray(out)


def test_config_frequencies_match_hugging_face():
    short, long, original = longrope.long_rope(TEXT)
    assert original == 16
    np.testing.assert_allclose(short, inv(SHORT), rtol=1e-6)
    np.testing.assert_allclose(long, inv(LONG), rtol=1e-6)
    # InternVL nests the Phi-3 config.
    assert longrope.text_config({"llm_config": TEXT}) is TEXT


def test_switch_takes_long_factors_once_a_pass_reaches_the_window():
    model = toy_rotary()
    assert longrope.add_long_rope_switch(model, TEXT) == 1
    pos = np.arange(10, 16)  # max 15: under the window
    np.testing.assert_allclose(run(model, pos), inv(SHORT)[None, :, None] * pos, rtol=1e-6)
    pos = np.arange(14, 20)  # max 19: the whole pass rotates with the long factors
    np.testing.assert_allclose(run(model, pos), inv(LONG)[None, :, None] * pos, rtol=1e-6)
    np.testing.assert_allclose(run(model, [16]), inv(LONG)[None, :, None] * 16, rtol=1e-6)
    np.testing.assert_allclose(run(model, [15]), inv(SHORT)[None, :, None] * 15, rtol=1e-6)


def test_other_ropes_are_left_alone_and_a_missing_constant_is_an_error():
    model = toy_rotary()
    plain = {k: v for k, v in TEXT.items() if k != "rope_scaling"}
    assert longrope.add_long_rope_switch(model, plain) == 0
    other = dict(TEXT, rope_scaling={"type": "longrope", "short_factor": [9.0] * 4,
                                     "long_factor": LONG})
    with pytest.raises(ValueError, match="no short-factor inv_freq constant"):
        longrope.add_long_rope_switch(toy_rotary(), other)


def test_patch_export_writes_a_switched_language_model(tmp_path):
    src, out = tmp_path / "export", tmp_path / "switched"
    src.mkdir()
    ov.save_model(toy_rotary(), str(src / longrope.LM_XML), compress_to_fp16=False)
    (src / "config.json").write_text(json.dumps({"model_type": "internvl_chat",
                                                 "llm_config": TEXT}))
    (src / "tokenizer.json").write_text("{}")
    assert longrope.patch_export(str(src), str(out)) == 1
    assert (out / "tokenizer.json").read_text() == "{}"
    model = ov.Core().read_model(str(out / longrope.LM_XML))
    np.testing.assert_allclose(run(model, [20]), inv(LONG)[None, :, None] * 20, rtol=1e-6)
    # The source export is untouched.
    model = ov.Core().read_model(str(src / longrope.LM_XML))
    np.testing.assert_allclose(run(model, [20]), inv(SHORT)[None, :, None] * 20, rtol=1e-6)


def test_cut_refuses_a_longrope_export_whose_rotary_it_cannot_find(tmp_path):
    from test_vlm_cut_phi_internvl import export, phi3v_config
    import cut_vlm_lm

    src = tmp_path / "phi"
    cfg = phi3v_config()
    cfg["rope_scaling"] = TEXT["rope_scaling"]
    cfg["head_dim"] = 8  # four frequencies, as SHORT / LONG list
    export(src, cfg, extra=("openvino_vision_projection_model.xml",))
    with pytest.raises(SystemExit, match="stage_0: no short-factor inv_freq constant"):
        cut_vlm_lm.run(str(src), str(tmp_path / "tree"), 1, "phi3_v",
                       cut_vlm_lm.g.DEFAULT_LAYERNORM_SUFFIX, None)


def test_packed_slots_must_not_reach_the_switch(tmp_path):
    import packed_variant

    stage = tmp_path / "stage_0"
    stage.mkdir()
    (stage / "stage_config.json").write_text(json.dumps({
        "stateful": False, "static_seq": 1, "static_context": 8193,
        "long_rope_switch": 4096}))
    with pytest.raises(RuntimeError, match="per pass, not per slot; use at least 2 slots"):
        packed_variant.write_packed_variant(str(stage), slots=1)
    # Two slots of 4096 never reach position 4096; the guard lets them through
    # (the build then fails on the missing IR, which this stage dir lacks).
    with pytest.raises(Exception) as e:
        packed_variant.write_packed_variant(str(stage), slots=2)
    assert "per pass" not in str(e.value)
