"""Unit tests for tools/vlm_surgery/cut_vlm_lm.py on toy Qwen VL language models.

Run from the repo root with:

    python -m pytest tools/tests/test_vlm_qwen_cut.py -v

The toy has the export's interface: `inputs_embeds`, `attention_mask`,
`position_ids [3, B, T]`, `beam_idx`, stateful KV per layer with the
optimum layer scopes, and (Qwen3-VL) the DeepStack inputs added after the
first layers the way the OpenVINO export does it (`cumsum` of the mask
gathers rows of `deepstack_visual_embeds[k]`). The tests cut it and check
the stage contract ov-runtime relies on, and that the cut chain (with the
i64 mask) computes what the uncut model does.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")
ov = pytest.importorskip("openvino")

_SURGERY = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "vlm_surgery"))
if _SURGERY not in sys.path:
    sys.path.insert(0, _SURGERY)

import cut_vlm_lm as cut  # noqa: E402
from openvino import opset13 as ops  # noqa: E402

H, KVH, HD, VOCAB = 8, 2, 4, 11
SCOPE = "__module.model.language_model.layers"


def named(node, name):
    node.set_friendly_name(name)
    return node


def toy_lm(layers: int, deepstack: int, seed: int = 0) -> ov.Model:
    rng = np.random.default_rng(seed)
    x = ops.parameter([-1, -1, H], ov.Type.f32, name="inputs_embeds")
    mask = ops.parameter([-1, -1], ov.Type.i64, name="attention_mask")
    pos = ops.parameter([3, -1, -1], ov.Type.i64, name="position_ids")
    beam = ops.parameter([-1], ov.Type.i32, name="beam_idx")
    params = [x, mask, pos, beam]
    if deepstack:
        vmask = ops.parameter([-1, -1], ov.Type.boolean, name="visual_pos_masks")
        dsv = ops.parameter([-1, -1, H], ov.Type.f32, name="deepstack_visual_embeds")
        params[2:2] = [vmask, dsv]
    for p in params:
        p.output(0).set_names({p.get_friendly_name()})
    # The rotary stand-in every layer reads: all three rows matter.
    rows = ops.convert(pos, ov.Type.f32)
    w3 = ops.constant(np.array([1.0, 0.5, 0.25], np.float32).reshape(3, 1, 1))
    rot = ops.unsqueeze(ops.reduce_sum(ops.multiply(rows, w3), ops.constant(np.int64(0)), False),
                        ops.constant(np.int64(2)))
    # The mask stand-in every layer reads too (the export's causal mask).
    msum = ops.convert(ops.reduce_sum(mask, ops.constant(np.array([1], np.int64)), True),
                       ov.Type.f32)
    rot = ops.add(rot, ops.reshape(ops.multiply(msum, ops.constant(np.float32(0.1))),
                                   ops.constant(np.array([-1, 1, 1], np.int64)), False))
    h = x.output(0)
    sinks = []
    for layer in range(layers):
        sc = f"{SCOPE}.{layer}"
        pw = named(ops.power(h, ops.constant(np.float32(2))), f"{sc}.input_layernorm/aten::pow/Power")
        n = ops.multiply(h, ops.add(ops.reduce_mean(pw, ops.constant(np.int64(-1)), True),
                                    ops.constant(np.float32(1))))
        outs = {}
        for kind in ("key", "value"):
            w = ops.constant(rng.standard_normal((H, KVH * HD)).astype(np.float32) * 0.1)
            new = ops.reshape(ops.matmul(n, w, False, False),
                              ops.constant(np.array([0, 0, KVH, HD], np.int64)), True)
            new = ops.transpose(new, ops.constant(np.array([0, 2, 1, 3], np.int64)))
            info = ov.op.util.VariableInfo()
            info.variable_id = f"past_key_values.{layer}.{kind}present.{layer}.{kind}"
            info.data_shape = ov.PartialShape([-1, KVH, -1, HD])
            info.data_type = ov.Type.f32
            var = ov.op.util.Variable(info)
            init = ops.broadcast(ops.constant(np.float32(0)),
                                 ops.constant(np.array([1, KVH, 0, HD], np.int64)))
            rv = ops.read_value(init, var)
            past = ops.gather(rv, beam, ops.constant(np.int64(0)))
            cat = named(ops.concat([past, new], 2), f"{sc}.self_attn/aten::cat/Concat_{kind}")
            sinks.append(ops.assign(cat, var))
            outs[kind] = cat
        kv = ops.reduce_mean(ops.add(outs["key"], outs["value"]),
                             ops.constant(np.array([1, 2, 3], np.int64)), True)
        kv = ops.reshape(kv, ops.constant(np.array([-1, 1, 1], np.int64)), False)
        wo = ops.constant(rng.standard_normal((H, H)).astype(np.float32) * 0.1)
        h = named(ops.add(ops.add(h, ops.matmul(n, wo, False, False)),
                          ops.add(ops.multiply(rot, ops.constant(np.float32(0.01))), kv)),
                  f"{sc}/aten::add/Add").output(0)
        if layer < deepstack:
            m = ops.reshape(vmask, ops.constant(np.array([-1], np.int64)), False)
            idx = ops.maximum(ops.subtract(ops.cumsum(ops.convert(m, ov.Type.i64),
                                                      ops.constant(np.int64(0))),
                                           ops.constant(np.int64(1))),
                              ops.constant(np.int64(0)))
            rows_k = ops.gather(dsv, ops.constant(np.int64(layer)), ops.constant(np.int64(0)))
            g = ops.reshape(ops.gather(rows_k, idx, ops.constant(np.int64(0))),
                            ops.shape_of(h), False)
            mf = ops.reshape(ops.convert(m, ov.Type.f32), ops.shape_of(mask), False)
            mf = ops.unsqueeze(mf, ops.constant(np.int64(2)))
            h = ops.add(h, ops.multiply(g, mf)).output(0)
    head = ops.constant(rng.standard_normal((H, VOCAB)).astype(np.float32))
    logits = ops.matmul(h, head, False, False)
    logits.output(0).set_names({"logits"})
    return ov.Model([ops.result(logits)], sinks, params, "toy_qwen_vl_lm")


def export_dir(tmp_path, family: str, layers: int, deepstack: int):
    src = tmp_path / f"src_{family}"
    src.mkdir()
    ov.save_model(toy_lm(layers, deepstack), str(src / "openvino_language_model.xml"),
                  compress_to_fp16=False)
    vision = {"model_type": family, "spatial_merge_size": 2}
    if deepstack:
        vision["deepstack_visual_indexes"] = list(range(5, 5 + 6 * deepstack, 6))
    cfg = {"model_type": family, "image_token_id": 151655, "video_token_id": 151656,
           "vision_start_token_id": 151652, "vision_end_token_id": 151653,
           "vision_config": vision,
           "text_config": {"num_hidden_layers": layers, "hidden_size": H,
                           "num_attention_heads": 2, "num_key_value_heads": KVH,
                           "head_dim": HD, "eos_token_id": 1}}
    (src / "config.json").write_text(json.dumps(cfg))
    for fn in ("openvino_text_embeddings_model.xml", "openvino_vision_embeddings_model.xml",
               "openvino_vision_embeddings_merger_model.xml",
               "openvino_vision_embeddings_pos_model.xml"):
        (src / fn).write_text("<net/>")
    (src / "preprocessor_config.json").write_text("{}")
    (src / "tokenizer.json").write_text("{}")
    return src


def stage_params(path):
    m = ov.Core().read_model(str(path))
    return {p.get_friendly_name(): p for p in m.get_parameters()}, m


def test_qwen3_vl_cut_keeps_deepstack_in_stage_0(tmp_path):
    src = export_dir(tmp_path, "qwen3_vl", layers=6, deepstack=3)
    out = tmp_path / "tree"
    cut.run(str(src), str(out), 2, "qwen3_vl", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)
    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["vision"]["family"] == "qwen3_vl"
    assert pipe["vision"]["mrope_rows"] == 3
    assert pipe["vision"]["deepstack_layers"] == 3
    assert pipe["vision"]["image_token_id"] == 151655
    for fn in ("openvino_vision_embeddings_merger_model.xml",
               "openvino_vision_embeddings_pos_model.xml", "config.vlm.json"):
        assert (out / fn).exists(), fn

    p0, _ = stage_params(out / "stage_0" / "openvino_model.xml")
    p1, _ = stage_params(out / "stage_1" / "openvino_model.xml")
    assert p0["visual_pos_masks"].get_element_type() == ov.Type.i64
    assert "deepstack_visual_embeds" in p0
    assert not {"visual_pos_masks", "deepstack_visual_embeds"} & set(p1)
    for p in (p0, p1):
        assert str(p["position_ids"].get_partial_shape()) == "[3,?,?]"
    for i in (0, 1):
        sc = json.loads((out / f"stage_{i}" / "stage_config.json").read_text())
        assert sc["position_rows"] == 3

    # The cut chain with the i64 mask computes what the uncut model does.
    core = ov.Core()
    T = 5
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, T, H)).astype(np.float32)
    pos = np.stack([np.array([0, 1, 1, 1, 3]), np.array([0, 1, 1, 2, 3]),
                    np.array([0, 1, 2, 1, 3])])[:, None, :].astype(np.int64)
    vm = np.array([[0, 1, 1, 1, 0]], dtype=bool)
    ds = rng.standard_normal((3, 3, H)).astype(np.float32)
    feed = {"inputs_embeds": x, "attention_mask": np.ones((1, T), np.int64),
            "position_ids": pos, "beam_idx": np.array([0], np.int32)}
    whole = core.compile_model(toy_lm(6, 3), "CPU").create_infer_request()
    want = whole.infer({**feed, "visual_pos_masks": vm, "deepstack_visual_embeds": ds})[0]
    s0 = core.compile_model(str(out / "stage_0" / "openvino_model.xml"), "CPU")
    s1 = core.compile_model(str(out / "stage_1" / "openvino_model.xml"), "CPU")
    hid = s0.create_infer_request().infer(
        {**feed, "visual_pos_masks": vm.astype(np.int64), "deepstack_visual_embeds": ds})[0]
    got = s1.create_infer_request().infer(
        {"hidden_states": hid.astype(np.float16), "attention_mask": feed["attention_mask"],
         "position_ids": pos, "beam_idx": feed["beam_idx"]})[0]
    np.testing.assert_allclose(got, want[:, -1:, :], rtol=2e-2, atol=2e-2)


def test_qwen3_vl_refuses_a_stage_0_without_the_deepstack_layers(tmp_path):
    src = export_dir(tmp_path, "qwen3_vl", layers=4, deepstack=3)
    with pytest.raises(SystemExit, match="DeepStack"):
        cut.run(str(src), str(tmp_path / "tree"), 2, "qwen3_vl",
                cut.g.DEFAULT_LAYERNORM_SUFFIX, None)


def test_qwen2_5_vl_cut_records_three_position_rows(tmp_path):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=4, deepstack=0)
    (src / "openvino_vision_embeddings_pos_model.xml").unlink()
    out = tmp_path / "tree"
    cut.run(str(src), str(out), 2, "qwen2_5_vl", cut.g.DEFAULT_LAYERNORM_SUFFIX, None)
    pipe = json.loads((out / "pipeline_config.json").read_text())
    assert pipe["vision"]["mrope_rows"] == 3
    assert "deepstack_layers" not in pipe["vision"]
    assert pipe["vision"]["token_type_ids"] is False
    for i in (0, 1):
        sc = json.loads((out / f"stage_{i}" / "stage_config.json").read_text())
        assert sc["position_rows"] == 3
        params, _ = stage_params(out / f"stage_{i}" / "openvino_model.xml")
        assert str(params["position_ids"].get_partial_shape()) == "[3,?,?]"
    assert (out / "openvino_vision_embeddings_merger_model.xml").exists()


def test_qwen2_5_vl_requires_the_merger_ir(tmp_path):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=4, deepstack=0)
    (src / "openvino_vision_embeddings_merger_model.xml").unlink()
    with pytest.raises(SystemExit, match="merger"):
        cut.run(str(src), str(tmp_path / "tree"), 2, "qwen2_5_vl",
                cut.g.DEFAULT_LAYERNORM_SUFFIX, None)


def test_family_must_match_the_export(tmp_path):
    src = export_dir(tmp_path, "qwen2_5_vl", layers=4, deepstack=0)
    with pytest.raises(SystemExit, match="expects model_type"):
        cut.run(str(src), str(tmp_path / "tree"), 2, "qwen3_vl",
                cut.g.DEFAULT_LAYERNORM_SUFFIX, None)
