"""weight_swap: quantizers, transform table, LoRA graft and final-hidden output.

    python -m pytest tools/tests/test_weight_swap.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery"))
import weight_swap as ws  # noqa: E402


def test_int8_asym_per_channel_formula():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 64)).astype(np.float32)
    q, zp, s = ws.q_asym(w, 8, axis=-1)
    assert q.dtype == np.uint8 and zp.shape == (4, 1) and s.dtype == np.float16
    lo, hi = w.min(-1, keepdims=True), w.max(-1, keepdims=True)
    exp_s = ((hi - lo) / 255).astype(np.float16)
    np.testing.assert_array_equal(s, exp_s)
    deq = (q.astype(np.float32) - zp) * s.astype(np.float32)
    assert np.abs(deq - w).max() <= float(s.astype(np.float32).max())


def test_int8_asym_single_signed_rows_keep_zero_in_range():
    # NNCF's asymmetric range is [min(w, 0), max(w, 0)]. A 4-tap linear-attention conv1d channel is
    # often all one sign (row 10 of Qwen3.8-27B layer 0's conv1d below); a [min, max] range would
    # clamp the zero point to 0/255 and saturate the row ([186, 255, 255, 255] instead of Intel's
    # [107, 178, 255, 173]).
    w = np.array([[0.03540039, 0.05859375, 0.08398438, 0.05688477],
                  [-0.01452637, -0.10058594, -0.07861328, -0.06591797]], dtype=np.float32)
    q, zp, s = ws.q_asym(w, 8, axis=-1)
    np.testing.assert_array_equal(q[0], [107, 178, 255, 173])
    np.testing.assert_array_equal(zp[:, 0], [0, 255])
    span = np.maximum(w.max(-1, keepdims=True), 0) - np.minimum(w.min(-1, keepdims=True), 0)
    np.testing.assert_array_equal(s, (span / 255.0).astype(np.float16))
    deq = (q.astype(np.float32) - zp) * s.astype(np.float32)
    assert np.abs(deq - w).max() <= float(s.astype(np.float32).max())


def test_sym8_rows():
    w = np.array([[1.0, -2.0], [0.0, 0.0]], dtype=np.float32)
    q, s = ws.q_sym8(w)
    np.testing.assert_array_equal(q[0], [64, -127])
    assert s[1, 0] > 0  # zero row keeps a positive scale


def test_check_swap_requires_every_layer_kind():
    full = {"proj": {0: sorted(ws.LAYER_PROJ["linear_attention"]), 1: sorted(ws.LAYER_PROJ["full_attention"])},
            "conv": {0: 1},
            "float": {0: ["input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.A_log",
                          "linear_attn.dt_bias", "linear_attn.norm.weight"],
                      1: ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                          "self_attn.k_norm.weight"],
                      "final": ["norm.weight"]}}
    ws.check_swap(full, range(0, 2), ["linear_attention", "full_attention"], last=True)
    missing_conv = {**full, "conv": {}}
    with pytest.raises(ValueError, match="conv"):
        ws.check_swap(missing_conv, range(0, 2), ["linear_attention", "full_attention"], last=True)
    no_final = {**full, "float": {k: v for k, v in full["float"].items() if k != "final"}}
    with pytest.raises(ValueError, match="norm"):
        ws.check_swap(no_final, range(0, 2), ["linear_attention", "full_attention"], last=True)


def test_check_swap_names_a_single_missing_projection():
    assert len(ws.LAYER_PROJ["linear_attention"]) == 8 and len(ws.LAYER_PROJ["full_attention"]) == 7
    floats = {0: sorted(ws.LAYER_FLOATS["linear_attention"]), 1: sorted(ws.LAYER_FLOATS["full_attention"])}
    for layer, kind in ((0, "linear_attention"), (1, "full_attention")):
        for drop in sorted(ws.LAYER_PROJ[kind]):
            proj = {0: sorted(ws.LAYER_PROJ["linear_attention"]), 1: sorted(ws.LAYER_PROJ["full_attention"])}
            proj[layer] = [p for p in proj[layer] if p != drop]
            report = {"proj": proj, "conv": {0: 1}, "float": floats}
            with pytest.raises(ValueError, match=rf"layer {layer} .*{drop}"):
                ws.check_swap(report, range(0, 2), ["linear_attention", "full_attention"], last=False)


def test_lora_scale_comes_from_adapter_config(tmp_path):
    import json
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64, "rank_pattern": {}, "alpha_pattern": {}}))
    assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 2.0
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "use_rslora": True}))
    assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 8.0
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "rank_pattern": {"q_proj": 8}}))
    with pytest.raises(ValueError, match="rank_pattern"):
        ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "use_dora": True}))
    with pytest.raises(ValueError, match="use_dora"):
        ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "lora_bias": True}))
    with pytest.raises(ValueError, match="lora_bias"):
        ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "use_dora": False, "lora_bias": False}))
    assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 2.0


def test_float_transform_table():
    w = np.array([0.5, -0.25], dtype=np.float32)
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["input_layernorm.weight"]](w), 1 + w)
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["linear_attn.A_log"]](w), np.exp(w))
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["linear_attn.dt_bias"]](w), w)


@pytest.mark.parametrize("prefix", ["", "language_model."])
def test_graft_lora_adds_scaled_branch(tmp_path, prefix):
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    x = ops.parameter([1, 3], np.float32, name="x")
    w = ops.constant(np.eye(3, dtype=np.float32))
    mm = ops.matmul(x, w, False, True)
    mm.set_friendly_name("__module.model.model.language_model.layers.0.mlp.down_proj/ov_ext::linear/MatMul")
    model = ov.Model([ops.result(mm)], [x], "t")
    a = torch.ones(1, 3)
    b = torch.full((3, 1), 0.5)
    path = str(tmp_path / "adapter_model.safetensors")
    save_file({f"base_model.model.model.{prefix}layers.0.mlp.down_proj.lora_A.weight": a,
               f"base_model.model.model.{prefix}layers.0.mlp.down_proj.lora_B.weight": b}, path)
    assert ws.lora_targets(path) == 1
    assert ws.graft_lora(model, path, 2.0) == 1
    out = ov.Core().compile_model(model, "CPU")([np.array([[1.0, 2.0, 3.0]], np.float32)])[0]
    # base = x ; lora = 2 * (x·1)·0.5 = sum(x) = 6
    np.testing.assert_allclose(out, [[7.0, 8.0, 9.0]], rtol=1e-6)


def test_graft_lora_only_touches_the_stage_layers(tmp_path):
    """A stage > 0 keeps shape-only copies of earlier layers (mask/position bookkeeping); grafting
    them inflates the count and fails the exporter's sum check, so only the stage's own layers
    (and lm_head on the last stage) are grafted."""
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    x = ops.parameter([1, 3], np.float32, name="x")
    outs, keys = [], {}
    for name in ("layers.0.mlp.down_proj", "layers.5.mlp.down_proj", "lm_head"):
        mm = ops.matmul(x, ops.constant(np.eye(3, dtype=np.float32)), False, True)
        mm.set_friendly_name(f"__module.model.model.language_model.{name}/ov_ext::linear/MatMul")
        outs.append(ops.result(mm))
        p = "base_model.model." + ("" if name == "lm_head" else "model.") + name
        keys[f"{p}.lora_A.weight"] = torch.ones(1, 3)
        keys[f"{p}.lora_B.weight"] = torch.full((3, 1), 0.5)
    path = str(tmp_path / "adapter_model.safetensors")
    save_file(keys, path)
    assert ws.graft_lora(ov.Model(outs, [x], "t"), path, 1.0, layers=range(4, 8), lm_head=False) == 1
    assert ws.graft_lora(ov.Model(outs, [x], "t"), path, 1.0, layers=range(4, 8), lm_head=True) == 2
    assert ws.graft_lora(ov.Model(outs, [x], "t"), path, 1.0) == 3


def test_add_final_hidden_output_exposes_lm_head_input():
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    x = ops.parameter([1, 2, 4], np.float32, name="x")
    norm = ops.multiply(x, ops.constant(np.float32(2.0)))
    mm = ops.matmul(norm, ops.constant(np.ones((5, 4), np.float32)), False, True)
    mm.set_friendly_name("__module.model.lm_head/ov_ext::linear/MatMul")
    model = ov.Model([ops.result(mm)], [x], "t")
    ws.add_final_hidden_output(model)
    assert len(model.outputs) == 2
    assert "final_hidden" in model.outputs[1].get_names()
    with pytest.raises(ValueError, match="1 output"):
        ws.add_final_hidden_output(model)  # already has two outputs


def test_add_final_hidden_output_without_lm_head_raises_value_error():
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    x = ops.parameter([1, 4], np.float32, name="x")
    model = ov.Model([ops.result(ops.multiply(x, ops.constant(np.float32(2.0))))], [x], "t")
    with pytest.raises(ValueError, match="lm_head"):
        ws.add_final_hidden_output(model)


class _FakeWeights:
    """dict-backed stand-in for ws.Weights (HF names relative to the layer stack)."""

    def __init__(self, d):
        self.d = d

    def get(self, name):
        return self.d[name]

    def rows(self, name, a, b):
        return self.d[name][a:b]


_PFX = "self.model.model.language_model."


def _build_stage(ov, ops, layer_types, rng, skip=()):
    """Tiny graph whose Constants are named like the stock IR's, plus matching fake HF weights."""
    hid, out_f, k = 8, 6, 4
    x = ops.parameter([1, hid], np.float32, name="x")
    results, hf = [], {}

    def compressed(name, u8_shape):
        w = ops.constant(np.full(u8_shape, 128, np.uint8), dtype=ov.Type.u8)
        w.set_friendly_name(name)
        zp = ops.constant(np.full([u8_shape[0]] + [1] * (len(u8_shape) - 1), 128, np.uint8), dtype=ov.Type.u8)
        zp.set_friendly_name(name + "/zero_point")
        s = ops.constant(np.ones([u8_shape[0]] + [1] * (len(u8_shape) - 1), np.float16), dtype=ov.Type.f16)
        s.set_friendly_name(name + "/scale")
        f = lambda n: ops.convert(n, ov.Type.f32)  # noqa: E731
        return ops.multiply(ops.subtract(f(w), f(zp)), f(s))

    def float_const(consumer, hf_name):
        c = ops.constant(np.zeros(hid, np.float32))
        mul = ops.multiply(x, c)
        mul.set_friendly_name(consumer)
        results.append(ops.result(mul))
        hf[hf_name] = rng.normal(size=hid).astype(np.float32)

    for i, t in enumerate(layer_types):
        for path in sorted(ws.LAYER_PROJ[t]):
            if (i, path) in skip:
                continue
            deq = compressed(f"{_PFX}layers.{i}.{path}._openvino_orig_weight", [out_f, hid])
            mm = ops.matmul(x, deq, False, True)
            mm.set_friendly_name(f"__module.model.model.language_model.layers.{i}.{path}/ov_ext::linear/MatMul")
            results.append(ops.result(mm))
            hf[f"layers.{i}.{path}.weight"] = rng.normal(size=(out_f, hid)).astype(np.float32)
        base = f"{_PFX}layers.{i}."
        float_const(base + "input_layernorm/aten::mul/Multiply_1", f"layers.{i}.input_layernorm.weight")
        float_const(base + "post_attention_layernorm/aten::mul/Multiply_1", f"layers.{i}.post_attention_layernorm.weight")
        if t == "linear_attention":
            results.append(ops.result(compressed(f"{base}linear_attn/aten::_convolution/Reshape", [hid, 1, k])))
            hf[f"layers.{i}.linear_attn.conv1d.weight"] = rng.normal(size=(hid, 1, k)).astype(np.float32)
            float_const(base + "linear_attn.norm/aten::mul/Multiply_1", f"layers.{i}.linear_attn.norm.weight")
            float_const(base + "linear_attn/aten::add/Add_2", f"layers.{i}.linear_attn.dt_bias")
            float_const(base + "linear_attn/aten::neg/Multiply_1", f"layers.{i}.linear_attn.A_log")
        else:
            float_const(base + "self_attn.q_norm/aten::mul/Multiply_1", f"layers.{i}.self_attn.q_norm.weight")
            float_const(base + "self_attn.k_norm/aten::mul/Multiply_1", f"layers.{i}.self_attn.k_norm.weight")
    float_const(f"{_PFX}norm/aten::mul/Multiply_1", "norm.weight")
    return ov.Model(results, [x], "stage"), _FakeWeights(hf)


def _const(model, name):
    return next(op for op in model.get_ops() if op.get_type_name() == "Constant" and op.get_friendly_name() == name)


def test_swap_stage_round_trips_through_check_swap():
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    types = ["linear_attention", "full_attention"]
    model, weights = _build_stage(ov, ops, types, np.random.default_rng(1))
    report = ws.swap_stage(model, weights)
    assert report["proj"] == {0: sorted(ws.LAYER_PROJ["linear_attention"]), 1: sorted(ws.LAYER_PROJ["full_attention"])}
    assert report["conv"] == {0: 1}
    assert set(report["float"][0]) == ws.LAYER_FLOATS["linear_attention"]
    assert set(report["float"][1]) == ws.LAYER_FLOATS["full_attention"]
    assert report["float"]["final"] == ["norm.weight"]
    ws.check_swap(report, range(0, 2), types, last=True)
    # projection constants now hold the int8 re-quantization of the fake HF tensor
    name = f"{_PFX}layers.1.self_attn.q_proj._openvino_orig_weight"
    w = weights.get("layers.1.self_attn.q_proj.weight")
    q, zp, s = (_const(model, name + sfx).get_data() for sfx in ("", "/zero_point", "/scale"))
    eq, ezp, es = ws.q_asym(w, 8, axis=-1)
    np.testing.assert_array_equal(q, eq)
    np.testing.assert_array_equal(zp, ezp)
    np.testing.assert_array_equal(s, es)
    # conv keeps its 3-D shape; floats go through their transform
    assert list(_const(model, f"{_PFX}layers.0.linear_attn/aten::_convolution/Reshape").get_output_shape(0)) == [8, 1, 4]
    a_log = next(op for op in model.get_ops() if op.get_friendly_name() == f"{_PFX}layers.0.linear_attn/aten::neg/Multiply_1")
    np.testing.assert_allclose(a_log.input_value(1).get_node().get_data(), np.exp(weights.get("layers.0.linear_attn.A_log")),
                               rtol=1e-6)
    out = ov.Core().compile_model(model, "CPU")([np.ones((1, 8), np.float32)])
    assert all(np.isfinite(v).all() for v in out.values())


def test_swap_stage_missing_projection_fails_check_swap():
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    types = ["linear_attention", "full_attention"]
    model, weights = _build_stage(ov, ops, types, np.random.default_rng(2), skip={(0, "linear_attn.in_proj_a")})
    report = ws.swap_stage(model, weights)
    with pytest.raises(ValueError, match=r"layer 0 .*in_proj_a"):
        ws.check_swap(report, range(0, 2), types, last=True)


# ---- BLD-4: swap_embeddings refuses an embeddings layout it would silently corrupt ---------------

def _embeddings_xml(ov, ops, path, asym=False, scale_rows=None):
    rows, hid = 6, 4
    name = "self.embed_tokens.weight_openvino_orig_weight"
    w = ops.constant(np.full((rows, hid), 128 if asym else 1, np.uint8 if asym else np.int8),
                     dtype=ov.Type.u8 if asym else ov.Type.i8)
    w.set_friendly_name(name)
    deq = ops.convert(w, ov.Type.f32)
    if asym:
        zp = ops.constant(np.full((rows, 1), 128, np.uint8), dtype=ov.Type.u8)
        zp.set_friendly_name(name + "/zero_point")
        deq = ops.subtract(deq, ops.convert(zp, ov.Type.f32))
    s = ops.constant(np.ones((scale_rows or rows, 1), np.float16), dtype=ov.Type.f16)
    s.set_friendly_name(name + "/scale")
    deq = ops.multiply(deq, ops.convert(s, ov.Type.f32))
    ids = ops.parameter([1, 2], np.int64, name="input_ids")
    out = ops.gather(deq, ids, 0)
    ov.save_model(ov.Model([ops.result(out)], [ids], "emb"), str(path))
    return rows, hid


class _RowWeights:
    def __init__(self, arr):
        self.arr = arr

    def rows(self, name, a, b):
        assert name == "embed_tokens.weight"
        return self.arr[a:b]


def test_check_embedding_layout_names_each_problem():
    ws.check_embedding_layout("e.xml", "i8", False, [6, 1], 6)
    with pytest.raises(ValueError, match=r"element type u8.*zero point.*scale shape \[1, 1\]"):
        ws.check_embedding_layout("e.xml", "u8", True, [1, 1], 6)


def test_swap_embeddings_refuses_u8_asymmetric_embeddings(tmp_path):
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    xml = tmp_path / "emb.xml"
    rows, hid = _embeddings_xml(ov, ops, xml, asym=True)
    with pytest.raises(ValueError, match="not i8 symmetric"):
        ws.swap_embeddings(str(xml), _RowWeights(np.ones((rows, hid), np.float32)), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_swap_embeddings_writes_i8_symmetric_rows(tmp_path):
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    xml = tmp_path / "emb.xml"
    rows, hid = _embeddings_xml(ov, ops, xml)
    w = np.random.default_rng(3).normal(size=(rows, hid)).astype(np.float32)
    ws.swap_embeddings(str(xml), _RowWeights(w), str(tmp_path / "out"))
    m = ov.Core().read_model(str(tmp_path / "out" / "openvino_text_embeddings_model.xml"))
    q = _const(m, "self.embed_tokens.weight_openvino_orig_weight").get_data()
    eq, _ = ws.q_sym8(w)
    np.testing.assert_array_equal(q, eq)
