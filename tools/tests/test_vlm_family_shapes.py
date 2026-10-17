"""Every family's stage inputs through static_variant.pin_shapes and
packed_variant.build_packed_model.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_family_shapes.py -v

Each family has its own branch in both shape pinners (Gemma's
token_type_ids, Qwen's M-RoPE rows, Qwen3-VL's DeepStack inputs, the
embed-vs-later-stage main input). One table runs every family's stage-input
set through both, so a change made for one family cannot drop another's
branch unnoticed (Gemma's token_type_ids pin once went missing exactly so).
The stages are test_packed_variant.py's synthetic static stages, the way
static_variant.py leaves them before packed_variant.py runs.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.abspath(os.path.join(_HERE, os.pardir))
for p in (_TOOLS, os.path.join(_TOOLS, "vlm_surgery"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import packed_variant as pv  # noqa: E402
import static_variant as sv  # noqa: E402
from test_packed_variant import (  # noqa: E402
    HD, HEADS, HIDDEN, LAYER_TYPES, PAST, retarget_main, toy_qwen_stage, toy_stage)

DEEPSTACK = 3
CANON = ("input_ids", "inputs_embeds", "hidden_states", "attention_mask", "position_ids",
         "token_type_ids", "visual_pos_masks", "deepstack_visual_embeds")


def _plain():
    return toy_stage(1)


# family -> (stage builder, has_embed, sliding layer types?, extra inputs)
# The extra inputs name the family branches: "types" (token_type_ids kept
# after packing), "types-dropped" (only the masks read it), "mrope",
# "deepstack". The main input is read off the built stage.
FAMILIES = {
    "gemma3": (lambda: toy_stage(2), True, True, {"types-dropped"}),
    "gemma3-later": (lambda: retarget_main(toy_stage(2), "hidden_states"), False, True,
                     {"types-dropped"}),
    # The Gemma 4 surgery's stage 0 port is grafted_inputs_embeds (alias
    # inputs_embeds); here token_type_ids also feeds the residual stream.
    "gemma4": (lambda: toy_stage(2, grafted=True, extra_types_use=True), True, True,
               {"types"}),
    "gemma4-later": (lambda: retarget_main(toy_stage(2, extra_types_use=True),
                                           "hidden_states"), False, True, {"types"}),
    "qwen2_5_vl": (lambda: toy_qwen_stage(0), True, False, {"mrope"}),
    "qwen2_5_vl-later": (lambda: retarget_main(toy_qwen_stage(0), "hidden_states"), False,
                         False, {"mrope"}),
    "qwen3_vl": (lambda: toy_qwen_stage(DEEPSTACK), True, False, {"mrope", "deepstack"}),
    "pixtral": (_plain, True, False, set()),
    "mistral3": (_plain, True, False, set()),
    "phi3_v": (_plain, True, False, set()),
    "internvl_chat": (_plain, True, False, set()),
    "llava_next": (_plain, True, False, set()),
    "plain-later": (lambda: retarget_main(toy_stage(1), "hidden_states"), False, False, set()),
    "text-ids": (lambda: retarget_main(toy_stage(1), "input_ids"), True, False, set()),
}


def _canon(port) -> str:
    return next((c for c in CANON if c in port.get_names()), port.get_any_name())


def _shapes(model) -> dict:
    return {_canon(i): list(i.get_shape()) for i in model.inputs}


def _main(model) -> str:
    return next(n for n in ("input_ids", "inputs_embeds", "hidden_states")
                if n in {_canon(i) for i in model.inputs})


def _row(main: str, s: int):
    return [1, s] if main == "input_ids" else [1, s, HIDDEN]


def _past():
    return {f"past_key_values.{li}.{k}": [1, HEADS, PAST, HD]
            for li in range(3) for k in ("key", "value")}


def _expected_pinned(main: str, extras: set, s: int) -> dict:
    want = {main: _row(main, s), "attention_mask": [1, PAST + s],
            "position_ids": [3, 1, s] if "mrope" in extras else [1, s], **_past()}
    if extras & {"types", "types-dropped"}:
        want["token_type_ids"] = [1, s]
    if "deepstack" in extras:
        want["visual_pos_masks"] = [1, s]
        want["deepstack_visual_embeds"] = [DEEPSTACK, s, HIDDEN]
    return want


def _expected_packed(main: str, extras: set, sliding: bool, s: int) -> dict:
    want = _expected_pinned(main, extras, s)
    del want["attention_mask"]
    want["attn_mask_4d"] = [1, 1, s, PAST + s]
    if sliding:
        want["attn_mask_sliding_4d"] = [1, 1, s, PAST + s]
    if "types-dropped" in extras:
        del want["token_type_ids"]
    return want


def _pin(family: str, seq: int):
    build, _, _, extras = FAMILIES[family]
    m = build()
    sv.pin_shapes(m, seq, PAST + seq,
                  deepstack_layers=DEEPSTACK if "deepstack" in extras else 0)
    return m


@pytest.mark.parametrize("seq", [1, 4], ids=["decode", "prefill"])
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_pin_shapes_per_family(family, seq):
    m = _pin(family, seq)
    assert _shapes(m) == _expected_pinned(_main(m), FAMILIES[family][3], seq)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_packed_variant_per_family(family):
    _, has_embed, sliding, extras = FAMILIES[family]
    m = _pin(family, 1)  # the static decode IR packed_variant re-reads
    main = _main(m)
    seq = 4
    info = {}
    packed, ctx = pv.build_packed_model(
        m, seq, has_embed, HIDDEN,
        layer_types=LAYER_TYPES if sliding else None, layer_start=0, num_layers=3,
        info=info, deepstack_layers=DEEPSTACK if "deepstack" in extras else 0)
    assert ctx == PAST + seq
    assert _shapes(packed) == _expected_packed(main, extras, sliding, seq)
    if sliding:
        assert info["packed_masks"] == {
            "attn_mask_4d": "full_attention", "attn_mask_sliding_4d": "sliding_attention"}


def test_the_table_covers_every_cut_family():
    import cut_vlm_lm as cut

    assert set(cut.FAMILIES) <= set(FAMILIES), "a new cut family needs a row here"
