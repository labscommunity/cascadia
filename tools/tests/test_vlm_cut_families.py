"""cut_vlm_lm.py's family table: which exports each --family accepts and
whether its stages keep token_type_ids (the third wire frame)."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("openvino")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "vlm_surgery"))
import cut_vlm_lm as cut  # noqa: E402

GEMMA3 = {
    "model_type": "gemma3",
    "image_token_index": 262144,
    "eos_token_id": [1, 106],
    "mm_tokens_per_image": 256,
    "text_config": {"model_type": "gemma3_text", "num_hidden_layers": 48,
                    "hidden_size": 3840, "num_attention_heads": 16, "head_dim": 256,
                    "sliding_window": 1024,
                    "layer_types": ["sliding_attention"] * 5 + ["full_attention"]},
    "vision_config": {"model_type": "siglip_vision_model"},
}


def test_gemma3_is_accepted_and_keeps_token_types():
    cut.check_family(GEMMA3, "gemma3")
    assert cut.FAMILIES["gemma3"]["token_type_ids"] is True
    assert cut.FAMILIES["pixtral"]["token_type_ids"] is False


def test_family_mismatch_is_refused():
    with pytest.raises(SystemExit):
        cut.check_family(GEMMA3, "pixtral")
    with pytest.raises(SystemExit):
        cut.check_family({**GEMMA3, "vision_config": {"model_type": "pixtral"}}, "gemma3")


def test_flat_config_keeps_window_layer_types_and_eos():
    flat = cut.flat_text_config(GEMMA3)
    # static_variant.py caps context by the window; packed_variant.py tells
    # the two mask producers apart by layer_types; ov-runtime reads eos.
    assert flat["sliding_window"] == 1024
    assert flat["layer_types"][5] == "full_attention"
    assert flat["eos_token_id"] == [1, 106]
