"""cut_vlm_lm.py's Phi-3.5-vision and InternVL2 families on a toy export.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_cut_families.py -v

A one-layer stateful toy language model stands in for the real one; each
family's config layout (phi3_v at the top level with its separator rows,
internvl_chat under `llm_config`) must come out as a flat text config, the
original kept as config.vlm.json, and a `vision` block naming the family and
its soft-token id. Phi-3.5 trees also stop on `<|end|>`.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SURGERY = os.path.abspath(os.path.join(_HERE, os.pardir, "vlm_surgery"))
for p in (_SURGERY, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_vlm_distspec import HIDDEN, KVH, VOCAB, toy_embeddings, toy_lm  # noqa: E402

TOKENIZER = {"added_tokens": [
    {"id": 2, "content": "</s>"}, {"id": 30, "content": "<|end|>"},
    {"id": 31, "content": "<|image|>"}, {"id": 29, "content": "<IMG_CONTEXT>"}]}
TEXT = {"num_hidden_layers": 1, "hidden_size": HIDDEN, "num_attention_heads": KVH,
        "num_key_value_heads": KVH, "vocab_size": VOCAB, "eos_token_id": 2,
        "original_max_position_embeddings": 4096}


def export(src, cfg, extra=()):
    src.mkdir()
    ov.save_model(toy_lm(), str(src / "openvino_language_model.xml"), compress_to_fp16=False)
    for fn in ("openvino_text_embeddings_model.xml", "openvino_vision_embeddings_model.xml", *extra):
        ov.save_model(toy_embeddings(), str(src / fn), compress_to_fp16=False)
    (src / "config.json").write_text(json.dumps(cfg))
    (src / "tokenizer.json").write_text(json.dumps(TOKENIZER))
    (src / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
    (src / "preprocessor_config.json").write_text(json.dumps({"num_crops": 4}))


def phi3v_config():
    return {"model_type": "phi3_v", **TEXT,
            "img_processor": {"name": "clip_vision_model", "image_dim_out": 2,
                              "num_img_tokens": 144},
            "embd_layer": {"use_hd_transform": True}, "glb_GN": [[[0.5] * 8]],
            "sub_GN": [[[[0.25] * 8]]], "auto_map": {}}


def test_phi3v_cut_flattens_config_and_names_image_token(tmp_path):
    import cut_vlm_lm as cut

    src, out = tmp_path / "vlm", tmp_path / "tree"
    export(src, phi3v_config(), extra=("openvino_vision_projection_model.xml",))
    cut.run(str(src), str(out), 1, "phi3_v", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)

    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["vision"]["family"] == "phi3_v"
    assert pipe["vision"]["image_token_id"] == 31
    assert pipe["vision"]["token_type_ids"] is False
    flat = json.loads((out / "config.json").read_text())
    assert flat["num_hidden_layers"] == 1 and "glb_GN" not in flat and "img_processor" not in flat
    vlm = json.loads((out / "config.vlm.json").read_text())
    assert vlm["sub_GN"] == [[[[0.25] * 8]]], "rank 0 reads the separators from here"
    assert (out / "openvino_vision_projection_model.xml").exists()
    for d in (out, out / "tokenizer"):
        assert json.loads((d / "generation_config.json").read_text())["eos_token_id"] == [2, 30]
    # The source export is untouched.
    assert json.loads((src / "generation_config.json").read_text())["eos_token_id"] == 2


def test_phi3v_cut_requires_the_projection_ir(tmp_path):
    import cut_vlm_lm as cut

    src = tmp_path / "vlm"
    export(src, phi3v_config())
    with pytest.raises(SystemExit, match="openvino_vision_projection_model.xml"):
        cut.run(str(src), str(tmp_path / "tree"), 1, "phi3_v", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)


def test_internvl_cut_takes_llm_config(tmp_path):
    import cut_vlm_lm as cut

    src, out = tmp_path / "vlm", tmp_path / "tree"
    export(src, {"model_type": "internvl_chat", "img_context_token_id": 29,
                 "vision_config": {"model_type": "intern_vit_6b"}, "llm_config": dict(TEXT),
                 "max_dynamic_patch": 12})
    cut.run(str(src), str(out), 1, "internvl_chat", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)

    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["vision"]["family"] == "internvl_chat"
    assert pipe["vision"]["image_token_id"] == 29
    flat = json.loads((out / "config.json").read_text())
    assert flat["hidden_size"] == HIDDEN and flat["eos_token_id"] == 2
    assert json.loads((out / "generation_config.json").read_text())["eos_token_id"] == 2


def test_family_is_checked_against_the_export(tmp_path):
    import cut_vlm_lm as cut

    src = tmp_path / "vlm"
    export(src, phi3v_config(), extra=("openvino_vision_projection_model.xml",))
    with pytest.raises(SystemExit, match="--family internvl_chat"):
        cut.run(str(src), str(tmp_path / "t"), 1, "internvl_chat",
                cut.g.DEFAULT_LAYERNORM_SUFFIX, None)
