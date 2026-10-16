"""`cut_vlm_lm.py --family llava_next` on a toy export.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_cut_llava_next.py -v

The cut must name the family and image token in the tree's `vision` block,
carry `image_newline` to rank 0 (in `config.vlm.json`) and the image
processor's settings, and refuse an export rank 0 could not serve.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_HERE = os.path.dirname(os.path.abspath(__file__))
for d in (_HERE, os.path.join(_HERE, os.pardir, "vlm_surgery")):
    d = os.path.abspath(d)
    if d not in sys.path:
        sys.path.insert(0, d)

import cut_vlm_lm as cut  # noqa: E402
from test_vlm_distspec import HIDDEN, KVH, VOCAB, toy_embeddings, toy_lm  # noqa: E402


def toy_export(root, **over):
    root.mkdir()
    ov.save_model(toy_lm(), str(root / "openvino_language_model.xml"), compress_to_fp16=False)
    for fn in ("openvino_text_embeddings_model.xml", "openvino_vision_embeddings_model.xml"):
        ov.save_model(toy_embeddings(), str(root / fn), compress_to_fp16=False)
    cfg = {
        "model_type": "llava_next", "image_token_index": 30,
        "image_grid_pinpoints": [[336, 672], [672, 336]],
        "image_newline": [0.5] * HIDDEN,
        "vision_feature_select_strategy": "default",
        "vision_config": {"model_type": "clip_vision_model", "image_size": 336, "patch_size": 14},
        "text_config": {"num_hidden_layers": 1, "hidden_size": HIDDEN,
                        "num_attention_heads": KVH, "num_key_value_heads": KVH,
                        "vocab_size": VOCAB},
    }
    cfg.update(over)
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "preprocessor_config.json").write_text(json.dumps({
        "image_grid_pinpoints": [[336, 672], [672, 336]], "crop_size": {"height": 336, "width": 336},
        "size": {"shortest_edge": 336}}))
    (root / "processor_config.json").write_text(json.dumps({"patch_size": 14}))
    (root / "tokenizer.json").write_text("{}")
    return root


def test_cut_writes_a_llava_next_tree(tmp_path):
    src, out = toy_export(tmp_path / "vlm"), tmp_path / "tree"
    cut.run(str(src), str(out), 1, "llava_next", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)
    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["vision"]["family"] == "llava_next"
    assert pipe["vision"]["image_token_id"] == 30
    assert pipe["vision"]["token_type_ids"] is False
    vlm = json.loads((out / "config.vlm.json").read_text())
    assert len(vlm["image_newline"]) == HIDDEN
    for fn in ("preprocessor_config.json", "processor_config.json",
               "openvino_vision_embeddings_model.xml", "openvino_text_embeddings_model.xml"):
        assert (out / fn).exists(), fn
    m = ov.Core().read_model(str(out / "stage_0" / "openvino_model.xml"))
    assert "inputs_embeds" in [p.get_friendly_name() for p in m.get_parameters()]


@pytest.mark.parametrize("over, match", [
    ({"image_newline": None}, "image_newline"),
    ({"image_newline": [0.0] * (HIDDEN - 1)}, "image_newline"),
    ({"vision_feature_select_strategy": "full"}, "vision_feature_select_strategy"),
    ({"vision_config": {"model_type": "siglip_vision_model"}}, "--family llava_next"),
])
def test_cut_refuses_what_rank_0_cannot_serve(tmp_path, over, match):
    src = toy_export(tmp_path / "vlm", **over)
    with pytest.raises(SystemExit, match=match):
        cut.run(str(src), str(tmp_path / "tree"), 1, "llava_next",
                cut.g.DEFAULT_LAYERNORM_SUFFIX, None)


def test_cut_requires_the_image_processor_config(tmp_path):
    src = toy_export(tmp_path / "vlm")
    os.remove(src / "preprocessor_config.json")
    with pytest.raises(SystemExit, match="preprocessor_config.json"):
        cut.run(str(src), str(tmp_path / "tree"), 1, "llava_next",
                cut.g.DEFAULT_LAYERNORM_SUFFIX, None)
