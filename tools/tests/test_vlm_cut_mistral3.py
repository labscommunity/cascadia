"""`cut_vlm_lm.py --family mistral3` on a toy export.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_cut_mistral3.py -v

Mistral 3 is the Pixtral layout plus a 2x2 patch merger that rank 0 runs
(`openvino_multi_modal_projector_model.xml`) at the `spatial_merge_size` it
reads from `processor_config.json`. Without either file the tree would serve
as plain Pixtral and fail only at image time, so the cut refuses the export;
with them both land in the tree root beside the stages, and the `vision`
block names the family (causal attention: no token_type_ids frame).
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

PROJECTOR = "openvino_multi_modal_projector_model.xml"
IMAGE_TOKEN = 10


def toy_export(root, skip=()):
    root.mkdir()
    ov.save_model(toy_lm(), str(root / "openvino_language_model.xml"), compress_to_fp16=False)
    for fn in ("openvino_text_embeddings_model.xml", "openvino_vision_embeddings_model.xml",
               PROJECTOR):
        if fn not in skip:
            ov.save_model(toy_embeddings(), str(root / fn), compress_to_fp16=False)
    (root / "config.json").write_text(json.dumps({
        "model_type": "mistral3", "image_token_index": IMAGE_TOKEN,
        "spatial_merge_size": 2, "eos_token_id": 2,
        "vision_config": {"model_type": "pixtral", "patch_size": 14},
        "text_config": {"num_hidden_layers": 1, "hidden_size": HIDDEN,
                        "num_attention_heads": KVH, "num_key_value_heads": KVH,
                        "vocab_size": VOCAB},
    }))
    if "processor_config.json" not in skip:
        (root / "processor_config.json").write_text(json.dumps(
            {"patch_size": 14, "spatial_merge_size": 2}))
    (root / "preprocessor_config.json").write_text(json.dumps({"patch_size": 14}))
    (root / "tokenizer.json").write_text("{}")
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
    return root


@pytest.mark.parametrize("missing", [PROJECTOR, "processor_config.json"])
def test_cut_refuses_a_mistral3_export_without_the_merger_inputs(tmp_path, missing):
    src, out = toy_export(tmp_path / "vlm", skip=(missing,)), tmp_path / "tree"
    with pytest.raises(SystemExit, match=f"lacks {missing}; not an OpenVINO mistral3"):
        cut.run(str(src), str(out), 1, "mistral3", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)
    # Refused before anything is written.
    assert not out.exists()


def test_cut_writes_a_mistral3_tree(tmp_path):
    src, out = toy_export(tmp_path / "vlm"), tmp_path / "tree"
    cut.run(str(src), str(out), 1, "mistral3", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)

    root = {p.name for p in out.iterdir()}
    assert root == {
        "pipeline_config.json", "config.json", "config.vlm.json", "stage_0", "tokenizer",
        "openvino_text_embeddings_model.xml", "openvino_text_embeddings_model.bin",
        "openvino_vision_embeddings_model.xml", "openvino_vision_embeddings_model.bin",
        "openvino_multi_modal_projector_model.xml", "openvino_multi_modal_projector_model.bin",
        "processor_config.json", "preprocessor_config.json", "generation_config.json",
        "tokenizer.json",
    }
    assert {p.name for p in (out / "tokenizer").iterdir()} == {
        "tokenizer.json", "generation_config.json", "config.json"}
    assert json.loads((out / "processor_config.json").read_text())["spatial_merge_size"] == 2

    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["vision"] == {
        "family": "mistral3",
        "text_embeddings": "openvino_text_embeddings_model.xml",
        "vision_embeddings": "openvino_vision_embeddings_model.xml",
        "token_type_ids": False,
        "image_token_id": IMAGE_TOKEN,
    }
    stage = json.loads((out / "stage_0" / "stage_config.json").read_text())
    assert stage["has_embed"] and stage["has_head"]
    assert "position_rows" not in stage
    assert "inputs_embeds" in stage["inputs"] and "token_type_ids" not in stage["inputs"]
    flat = json.loads((out / "config.json").read_text())
    assert flat["hidden_size"] == HIDDEN and flat["eos_token_id"] == 2
    assert json.loads((out / "config.vlm.json").read_text())["model_type"] == "mistral3"
