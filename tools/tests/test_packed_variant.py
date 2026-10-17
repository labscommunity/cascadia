"""Unit tests for tools/packed_variant.py on synthetic static stages.

Run from the repo root with:

    python -m pytest tools/tests/test_packed_variant.py -v

The toy stages have the static contract (`inputs_embeds`, `attention_mask`,
`position_ids`, `past_key_values.{i}.*` in, `present.{i}.*` out) and real
ScaledDotProductAttention nodes. One toy builds a single shared mask (every
family before Gemma 4); the other builds two, Gemma 4 style: a sliding-window
mask and a full-attention one, both reading `token_type_ids`, named after the
first layer that built them. The tests check the port contract and that the
packed model fed the masks the producers would have built gives the original
model's output.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import packed_variant as pv  # noqa: E402
from openvino import opset13 as ops  # noqa: E402

HIDDEN, HEADS, HD, PAST = 8, 2, 4, 6
# Global layer types of the toy model; the stage holds layers 0..2.
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention"]


def _const(v, dt=np.int64):
    return ops.constant(np.array(v, dtype=dt))


def _mask_producer(mask, types, x, name, use_types, weight=-0.5):
    """[1,1,S,T] additive f32 mask from the 2D mask (and the token types),
    ending in a Slice named like the optimum export's. The two Gemma-style
    producers must differ (`weight`), or OpenVINO merges them on reshape."""
    m = ops.convert(mask, ov.Type.f32)
    m = ops.multiply(ops.subtract(m, ops.constant(np.float32(1))), ops.constant(np.float32(1e4)))
    if use_types:
        t = ops.convert(ops.reduce_sum(types, _const([1]), True), ov.Type.f32)
        m = ops.add(m, ops.multiply(t, ops.constant(np.float32(weight))))
    m = ops.unsqueeze(ops.unsqueeze(m, _const([1])), _const([1]))  # [1,1,1,T]
    s = ops.gather(ops.shape_of(x), _const([1]), _const(0))
    t_len = ops.gather(ops.shape_of(mask), _const([1]), _const(0))
    target = ops.concat([_const([1, 1]), s, t_len], 0)
    b = ops.broadcast(m, target)
    out = ops.slice(b, _const([0]), _const([np.iinfo(np.int64).max]), _const([1]), _const([3]))
    out.set_friendly_name(name)
    return out


def toy_stage(producers: int, seed: int = 0, extra_types_use: bool = False, outputs=None,
              grafted: bool = False):
    """Three attention layers over a static past of PAST tokens.

    `producers == 1`: one mask feeds every SDPA. `producers == 2`: layers 0, 1
    take the sliding producer (`layers.0...`), layer 2 the full one
    (`layers.2...`). `outputs`, when a list, receives the producer nodes so a
    reference model can expose them.
    """
    rng = np.random.default_rng(seed)
    x = ops.parameter([1, -1, HIDDEN], ov.Type.f32, name="inputs_embeds")
    # The Gemma surgery's stage 0 port is `grafted_inputs_embeds`, aliased.
    x.output(0).set_names({"grafted_inputs_embeds", "inputs_embeds"} if grafted
                          else {"inputs_embeds"})
    mask = ops.parameter([1, -1], ov.Type.i64, name="attention_mask")
    mask.output(0).set_names({"attention_mask"})
    pos = ops.parameter([1, -1], ov.Type.i64, name="position_ids")
    pos.output(0).set_names({"position_ids"})
    params = [x, mask, pos]
    types = None
    if producers == 2:
        types = ops.parameter([1, -1], ov.Type.i64, name="token_type_ids")
        types.output(0).set_names({"token_type_ids"})
        params.append(types)
        sliding = _mask_producer(
            mask, types, x, "__module.layers.0.self_attn/aten::sdpa/Slice_11", True)
        full = _mask_producer(
            mask, types, x, "__module.layers.2.self_attn/aten::sdpa/Slice_11", True, -0.25)
        per_layer = [sliding, sliding, full]
    else:
        shared = _mask_producer(mask, None, x, "__module.model/aten::mask/Slice_11", False)
        per_layer = [shared] * 3
    if outputs is not None:
        outputs.extend(dict.fromkeys(per_layer))
    h = x.output(0)
    if extra_types_use:
        bias = ops.convert(ops.reduce_sum(types, _const([1]), True), ov.Type.f32)
        h = ops.add(h, ops.unsqueeze(bias, _const([2]))).output(0)
    p = ops.unsqueeze(ops.convert(pos, ov.Type.f32), _const([2]))
    h = ops.add(h, ops.multiply(p, ops.constant(np.float32(0.01)))).output(0)
    results = []
    for li in range(3):
        def proj():
            w = ops.constant(rng.standard_normal((HIDDEN, HEADS * HD)).astype(np.float32) * 0.3)
            y = ops.reshape(ops.matmul(h, w, False, False), _const([0, 0, HEADS, HD]), True)
            return ops.transpose(y, _const([0, 2, 1, 3]))
        q, k_new, v_new = proj(), proj(), proj()
        kv = {}
        for kind, new in (("key", k_new), ("value", v_new)):
            name = f"past_key_values.{li}.{kind}"
            past = ops.parameter([1, HEADS, PAST, HD], ov.Type.f16, name=name)
            past.output(0).set_names({name})
            params.append(past)
            cat = ops.concat([ops.convert(past, ov.Type.f32), new], 2)
            r = ops.result(ops.convert(cat, ov.Type.f16))
            r.output(0).set_names({f"present.{li}.{kind}"})
            results.append(r)
            kv[kind] = cat
        att = ops.scaled_dot_product_attention(q, kv["key"], kv["value"], per_layer[li])
        att = ops.reshape(ops.transpose(att, _const([0, 2, 1, 3])), _const([0, 0, -1]), True)
        wo = ops.constant(rng.standard_normal((HEADS * HD, HIDDEN)).astype(np.float32) * 0.3)
        h = ops.add(h, ops.matmul(att, wo, False, False)).output(0)
    out = ops.result(h)
    out.output(0).set_names({"hidden_states"})
    return ov.Model([out] + results, params, "toy_static_stage")


F32 = {"INFERENCE_PRECISION_HINT": "f32"}


def pinned(m, seq=1):
    """Pin the toy to the static decode shape, as static exports are."""
    embeds = next(i.get_any_name() for i in m.inputs if "inputs_embeds" in i.get_names())
    shapes = {embeds: [1, seq, HIDDEN], "attention_mask": [1, PAST + seq],
              "position_ids": [1, seq]}
    if any(i.get_any_name() == "token_type_ids" for i in m.inputs):
        shapes["token_type_ids"] = [1, seq]
    m.reshape(shapes)
    return m


def _feed(seq, rng, types=True):
    feed = {
        "inputs_embeds": rng.standard_normal((1, seq, HIDDEN)).astype(np.float32),
        "attention_mask": np.ones((1, PAST + seq), np.int64),
        "position_ids": np.arange(seq, dtype=np.int64)[None] + PAST,
    }
    feed["attention_mask"][0, :2] = 0
    if types:
        feed["token_type_ids"] = np.array([[0] + [1] * (seq - 1)], np.int64)
    for li in range(3):
        for kind in ("key", "value"):
            feed[f"past_key_values.{li}.{kind}"] = (
                rng.standard_normal((1, HEADS, PAST, HD)).astype(np.float16))
    return feed


def test_classify_two_producers_from_names_and_order():
    m = toy_stage(2)
    sdpa = [o for o in m.get_ordered_ops() if o.get_type_name() == "ScaledDotProductAttention"]
    kinds = pv.classify_mask_producers(sdpa, LAYER_TYPES, 0, 3)
    assert sorted(kinds.values()) == ["full_attention", "sliding_attention"]
    assert kinds["__module.layers.0.self_attn/aten::sdpa/Slice_11"] == "sliding_attention"
    # The name decides alone when the SDPA count does not match the layers.
    assert pv.classify_mask_producers(sdpa, LAYER_TYPES, 0, None) == kinds
    # A name that contradicts the layers it feeds is refused.
    shifted = ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    with pytest.raises(RuntimeError, match="its name says"):
        pv.classify_mask_producers(sdpa, shifted, 1, 3)
    # No sliding layers anywhere: every producer is full attention.
    assert set(pv.classify_mask_producers(sdpa, None).values()) == {"full_attention"}


def test_two_producers_become_two_mask_ports():
    seq = 3
    info = {}
    packed, ctx = pv.build_packed_model(
        pinned(toy_stage(2)), seq, has_embed=True, hidden_size=HIDDEN,
        layer_types=LAYER_TYPES, layer_start=0, num_layers=3, info=info)
    assert ctx == PAST + seq
    names = [i.get_any_name() for i in packed.inputs]
    assert "attn_mask_4d" in names and "attn_mask_sliding_4d" in names
    assert "attention_mask" not in names and "token_type_ids" not in names
    assert info["packed_masks"] == {
        "attn_mask_4d": "full_attention", "attn_mask_sliding_4d": "sliding_attention"}
    assert sorted(info["packed_dropped_inputs"]) == ["attention_mask", "token_type_ids"]
    shapes = {i.get_any_name(): list(i.get_shape()) for i in packed.inputs}
    assert shapes["attn_mask_4d"] == [1, 1, seq, ctx]
    assert shapes["attn_mask_sliding_4d"] == [1, 1, seq, ctx]
    assert shapes["inputs_embeds"] == [1, seq, HIDDEN]


def test_packed_model_fed_the_producer_masks_matches_the_original():
    seq = 3
    core = ov.Core()
    rng = np.random.default_rng(7)
    feed = _feed(seq, rng)
    # Reference: the original stage, with its two mask producers exposed.
    prods = []
    ref = toy_stage(2, outputs=prods)
    ref.add_results([ops.result(p) for p in prods])
    got = core.compile_model(ref, "CPU", F32).create_infer_request().infer(feed)
    outs = list(got.values())
    want_h = outs[0]
    sliding_mask, full_mask = outs[-2], outs[-1]

    packed, _ = pv.build_packed_model(
        pinned(toy_stage(2)), seq, True, HIDDEN, layer_types=LAYER_TYPES, num_layers=3)
    pfeed = {k: v for k, v in feed.items() if k not in ("attention_mask", "token_type_ids")}
    pfeed["attn_mask_sliding_4d"] = sliding_mask
    pfeed["attn_mask_4d"] = full_mask
    res = core.compile_model(packed, "CPU", F32).create_infer_request().infer(pfeed)
    np.testing.assert_allclose(res["hidden_states"], want_h, rtol=1e-4, atol=1e-4)
    # And each port reaches its own layers: swapping the masks changes the output.
    pfeed["attn_mask_sliding_4d"], pfeed["attn_mask_4d"] = full_mask * 0, sliding_mask
    swapped = core.compile_model(packed, "CPU", F32).create_infer_request().infer(pfeed)
    assert not np.allclose(swapped["hidden_states"], want_h, atol=1e-4)


def test_token_type_ids_kept_when_something_else_reads_it():
    info = {}
    packed, _ = pv.build_packed_model(
        pinned(toy_stage(2, extra_types_use=True)), 3, True, HIDDEN,
        layer_types=LAYER_TYPES, num_layers=3, info=info)
    shapes = {i.get_any_name(): list(i.get_shape()) for i in packed.inputs}
    assert shapes["token_type_ids"] == [1, 3]
    assert "attention_mask" not in shapes
    assert info["packed_dropped_inputs"] == ["attention_mask"]


def test_two_producers_without_sliding_layer_types_are_refused():
    with pytest.raises(RuntimeError, match="ONE shared mask producer"):
        pv.build_packed_model(pinned(toy_stage(2)), 3, True, HIDDEN)


def test_single_producer_keeps_the_original_edit():
    info = {}
    packed, ctx = pv.build_packed_model(pinned(toy_stage(1)), 2, True, HIDDEN, info=info)
    names = [i.get_any_name() for i in packed.inputs]
    assert names[-1] == "attn_mask_4d"
    assert "attention_mask" not in names and "attn_mask_sliding_4d" not in names
    assert info == {}, "the single-mask path records nothing new"
    # Layer types with sliding layers but one FULL producer: same edit.
    packed2, _ = pv.build_packed_model(
        pinned(toy_stage(1)), 2, True, HIDDEN, layer_types=["full_attention"] * 3, num_layers=3)
    assert [i.get_any_name() for i in packed2.inputs] == names


def test_write_packed_variant_records_mask_ports(tmp_path):
    stage = tmp_path / "stage_0"
    stage.mkdir()
    ov.save_model(pinned(toy_stage(2)), str(stage / "openvino_model.xml"), compress_to_fp16=False)
    (stage / "stage_config.json").write_text(json.dumps({
        "stateful": False, "has_embed": True, "static_seq": 1,
        "static_context": PAST + 1, "layer_start": 0, "layer_end": 3}))
    (tmp_path / "config.json").write_text(json.dumps({
        "layer_types": LAYER_TYPES, "sliding_window": 1024}))
    cfg = pv.write_packed_variant(str(stage), slots=2, packed_seq=3)
    on_disk = json.loads((stage / "stage_config.json").read_text())
    assert on_disk == cfg
    assert cfg["packed_masks"] == {
        "attn_mask_4d": "full_attention", "attn_mask_sliding_4d": "sliding_attention"}
    assert cfg["packed_sliding_window"] == 1024
    assert cfg["packed_context"] == PAST + 3
    names = {i.get_any_name() for i in ov.Core().read_model(
        str(stage / "openvino_packed_model.xml")).inputs}
    assert {"attn_mask_4d", "attn_mask_sliding_4d"} <= names
    # Without a window the sliding port cannot be fed.
    (tmp_path / "config.json").write_text(json.dumps({"layer_types": LAYER_TYPES}))
    with pytest.raises(RuntimeError, match="sliding_window"):
        pv.write_packed_variant(str(stage), slots=2, packed_seq=3)


def test_gemma_embed_stage_without_hidden_size(tmp_path):
    # Gemma 4 vision surgery stage 0: no hidden_size in stage_config, and its
    # embedding port is `grafted_inputs_embeds` (alias `inputs_embeds`).
    stage = tmp_path / "stage_0"
    stage.mkdir()
    m = pinned(toy_stage(2, grafted=True))
    assert m.inputs[0].get_any_name() == "grafted_inputs_embeds"
    ov.save_model(m, str(stage / "openvino_model.xml"), compress_to_fp16=False)
    (stage / "stage_config.json").write_text(json.dumps({
        "stateful": False, "has_embed": True, "static_seq": 1,
        "static_context": PAST + 1, "layer_start": 0, "layer_end": 3,
        "export_version": "gemma4_vision_surgery_v1"}))
    (tmp_path / "config.json").write_text(json.dumps({
        "layer_types": LAYER_TYPES, "sliding_window": 1024}))
    assert pv.stage_hidden_size(m, {}, str(stage)) == HIDDEN
    pv.write_packed_variant(str(stage), slots=2, packed_seq=3)
    packed = ov.Core().read_model(str(stage / "openvino_packed_model.xml"))
    port = next(i for i in packed.inputs if "inputs_embeds" in i.get_names())
    assert list(port.get_shape()) == [1, 3, HIDDEN]


def toy_qwen_stage(deepstack_layers: int = 0):
    """A causal stage like Qwen VL's: one full-attention mask producer, M-RoPE
    `position_ids [3, ?, ?]`, and (Qwen3-VL) the DeepStack inputs."""
    rng = np.random.default_rng(3)
    x = ops.parameter([1, -1, HIDDEN], ov.Type.f32, name="inputs_embeds")
    x.output(0).set_names({"inputs_embeds"})
    mask = ops.parameter([1, -1], ov.Type.i64, name="attention_mask")
    mask.output(0).set_names({"attention_mask"})
    pos = ops.parameter([3, -1, -1], ov.Type.i64, name="position_ids")
    pos.output(0).set_names({"position_ids"})
    params = [x, mask, pos]
    shared = _mask_producer(mask, None, x, "__module.model/aten::mask/Slice_11", False)
    # Fold the 3 position rows back in so the rank-3 port is reachable:
    # sum axis 0 -> [1, seq], trailing unsqueeze -> [1, seq, 1] broadcasts.
    pbias = ops.convert(ops.reduce_sum(pos, _const([0]), False), ov.Type.f32)
    pbias = ops.unsqueeze(pbias, _const([2]))
    h = ops.add(x.output(0), ops.multiply(pbias, ops.constant(np.float32(0.01)))).output(0)
    if deepstack_layers:
        vpm = ops.parameter([1, -1], ov.Type.i64, name="visual_pos_masks")
        vpm.output(0).set_names({"visual_pos_masks"})
        dse = ops.parameter([-1, -1, HIDDEN], ov.Type.f32, name="deepstack_visual_embeds")
        dse.output(0).set_names({"deepstack_visual_embeds"})
        params += [vpm, dse]
        cum = ops.cumsum(ops.reshape(vpm, _const([-1]), False), _const(0))
        idx = ops.maximum(ops.subtract(cum, _const(1)), _const(0))
        layer0 = ops.gather(dse, _const(0), _const(0))
        rows = ops.gather(layer0, idx, _const(0))
        h = ops.add(h, ops.reshape(rows, _const([1, -1, HIDDEN]), False)).output(0)
    results = []
    for li in range(3):
        def proj():
            w = ops.constant(rng.standard_normal((HIDDEN, HEADS * HD)).astype(np.float32) * 0.3)
            y = ops.reshape(ops.matmul(h, w, False, False), _const([0, 0, HEADS, HD]), True)
            return ops.transpose(y, _const([0, 2, 1, 3]))
        q, k_new, v_new = proj(), proj(), proj()
        kv = {}
        for kind, new in (("key", k_new), ("value", v_new)):
            name = f"past_key_values.{li}.{kind}"
            past = ops.parameter([1, HEADS, PAST, HD], ov.Type.f16, name=name)
            past.output(0).set_names({name})
            params.append(past)
            cat = ops.concat([ops.convert(past, ov.Type.f32), new], 2)
            r = ops.result(ops.convert(cat, ov.Type.f16))
            r.output(0).set_names({f"present.{li}.{kind}"})
            results.append(r)
            kv[kind] = cat
        att = ops.scaled_dot_product_attention(q, kv["key"], kv["value"], shared)
        att = ops.reshape(ops.transpose(att, _const([0, 2, 1, 3])), _const([0, 0, -1]), True)
        wo = ops.constant(rng.standard_normal((HEADS * HD, HIDDEN)).astype(np.float32) * 0.3)
        h = ops.add(h, ops.matmul(att, wo, False, False)).output(0)
    out = ops.result(h)
    out.output(0).set_names({"hidden_states"})
    return ov.Model([out] + results, params, "toy_qwen_static")


def _pin_qwen(m, seq=1, deepstack_layers=0):
    shapes = {"inputs_embeds": [1, seq, HIDDEN], "attention_mask": [1, PAST + seq],
              "position_ids": [3, 1, seq]}
    if deepstack_layers:
        shapes["visual_pos_masks"] = [1, seq]
        shapes["deepstack_visual_embeds"] = [deepstack_layers, seq, HIDDEN]
    m.reshape(shapes)
    return m


def test_packed_mrope_position_ids_pinned_to_three_rows():
    seq = 4
    packed, ctx = pv.build_packed_model(_pin_qwen(toy_qwen_stage()), seq, True, HIDDEN)
    port = next(i for i in packed.inputs if "position_ids" in i.get_names())
    assert list(port.get_shape()) == [3, 1, seq]
    assert ctx == PAST + seq


def test_packed_deepstack_inputs_pinned_to_seq_width():
    seq = 4
    packed, _ = pv.build_packed_model(
        _pin_qwen(toy_qwen_stage(3), seq, 3), seq, True, HIDDEN, deepstack_layers=3)
    shapes = {i.get_any_name(): list(i.get_shape()) for i in packed.inputs}
    assert shapes["position_ids"] == [3, 1, seq]
    assert shapes["visual_pos_masks"] == [1, seq]
    assert shapes["deepstack_visual_embeds"] == [3, seq, HIDDEN]
    # The 2D attention_mask is replaced by the 4D per-row mask as usual.
    assert "attn_mask_4d" in shapes and "attention_mask" not in shapes


def test_packed_deepstack_layer_count_from_tree_config(tmp_path):
    stage = tmp_path / "stage_0"
    stage.mkdir()
    ov.save_model(_pin_qwen(toy_qwen_stage(3), 1, 3),
                  str(stage / "openvino_model.xml"), compress_to_fp16=False)
    (stage / "stage_config.json").write_text(json.dumps({
        "stateful": False, "has_embed": True, "static_seq": 1,
        "static_context": PAST + 1, "layer_start": 0, "layer_end": 3,
        "position_rows": 3, "hidden_size": HIDDEN}))
    (tmp_path / "pipeline_config.json").write_text(json.dumps({
        "vision": {"family": "qwen3_vl", "mrope_rows": 3, "deepstack_layers": 3}}))
    assert pv.tree_deepstack_layers(str(stage)) == 3
    cfg = pv.write_packed_variant(str(stage), slots=2, packed_seq=4)
    assert cfg["packed_seq"] == 4
    packed = ov.Core().read_model(str(stage / "openvino_packed_model.xml"))
    shapes = {i.get_any_name(): list(i.get_shape()) for i in packed.inputs}
    assert shapes["deepstack_visual_embeds"] == [3, 4, HIDDEN]
    assert shapes["position_ids"] == [3, 1, 4]


def test_hidden_size_falls_back_to_the_tree_config(tmp_path):
    stage = tmp_path / "stage_0"
    stage.mkdir()
    m = toy_stage(1)
    # An embedding port without a static width: config.json decides.
    x = ops.parameter([1, -1, -1], ov.Type.f32, name="inputs_embeds")
    x.output(0).set_names({"inputs_embeds"})
    dyn = ov.Model([ops.result(x)], [x], "dyn")
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 640}}))
    assert pv.stage_hidden_size(dyn, {}, str(stage)) == 640
    assert pv.stage_hidden_size(m, {"hidden_size": 12}, str(stage)) == 12
