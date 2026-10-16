"""weight_swap: quantizers, transform table, LoRA graft and final-hidden output.

    python -m pytest tools/tests/test_weight_swap.py -v
"""
import os
import sys
from pathlib import Path

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


def test_q_asym_zero_and_constant_rows():
    # H32: a zero row keeps a positive scale and dequantizes to exactly 0; a constant row's range
    # is [0, c] (or [c, 0]), so it lands on the top (bottom) code and dequantizes to ~c.
    w = np.array([[0.0] * 4, [0.25] * 4, [-0.5] * 4], dtype=np.float32)
    q, zp, s = ws.q_asym(w, 8, axis=-1)
    assert s[0, 0] > 0 and (q[0] == zp[0]).all()
    np.testing.assert_array_equal(q[1], [255] * 4)
    np.testing.assert_array_equal(zp[:, 0], [0, 0, 255])
    np.testing.assert_array_equal(q[2], [0] * 4)
    deq = (q.astype(np.float32) - zp) * s.astype(np.float32)
    np.testing.assert_array_equal(deq[0], [0.0] * 4)
    np.testing.assert_allclose(deq[1:], w[1:], rtol=1e-3)


def test_q_asym_u4_groups():
    # H32: the u4 path quantizes [out, G, 128] per group (axis -1) into 0..15
    rng = np.random.default_rng(3)
    w = rng.normal(size=(2, 3, 128)).astype(np.float32)
    q, zp, s = ws.q_asym(w, 4, axis=-1)
    assert q.shape == w.shape and zp.shape == s.shape == (2, 3, 1)
    assert q.dtype == zp.dtype == np.uint8 and s.dtype == np.float16
    assert q.max() <= 15 and zp.max() <= 15
    span = np.maximum(w.max(-1, keepdims=True), 0) - np.minimum(w.min(-1, keepdims=True), 0)
    np.testing.assert_array_equal(s, (span / 15.0).astype(np.float16))
    deq = (q.astype(np.float32) - zp) * s.astype(np.float32)
    assert np.abs(deq - w).max() <= float(s.astype(np.float32).max())


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_q_asym_output_dtypes_do_not_follow_the_input(dtype):
    # H32: whatever the checkpoint dtype, the codes are u8 and the scale f16 (the stock IR's types)
    w = np.linspace(-1, 2, 16, dtype=dtype).reshape(2, 8)
    q, zp, s = ws.q_asym(w, 8, axis=-1)
    assert (q.dtype, zp.dtype, s.dtype) == (np.uint8, np.uint8, np.float16)
    deq = (q.astype(np.float32) - zp) * s.astype(np.float32)
    assert np.abs(deq - w.astype(np.float32)).max() <= float(s.astype(np.float32).max())


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


def write_audit_adapter(d):
    """limits-audit B2 repro: one decoder LoRA plus an embedding LoRA, with `norm` in modules_to_save
    (PEFT saves the trained norm as a plain tensor). A count-only check saw 1 target and 1 graft."""
    import json
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    path = str(d / "adapter_model.safetensors")
    save_file({"base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.ones(1, 3),
               "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.ones(3, 1),
               "base_model.model.model.embed_tokens.lora_embedding_A": torch.ones(1, 16),
               "base_model.model.model.embed_tokens.lora_embedding_B": torch.ones(3, 1),
               "base_model.model.model.norm.weight": torch.ones(3)}, path)
    (d / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1, "modules_to_save": ["norm"]}))
    return path


def save_adapter(path, keys):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    save_file({k: torch.ones(1, 3) for k in keys}, str(path))
    return str(path)


def test_lora_targets_refuses_embedding_lora_and_saved_modules(tmp_path):
    path = write_audit_adapter(tmp_path)
    with pytest.raises(ValueError, match=r"embedding LoRA keys.*embed_tokens\.lora_embedding_A"):
        ws.lora_targets(path)
    p = "base_model.model.model.layers.0.mlp.down_proj"
    path = save_adapter(tmp_path / "saved.safetensors", [f"{p}.lora_A.weight", f"{p}.lora_B.weight",
                                                         "base_model.model.model.norm.weight"])
    with pytest.raises(ValueError, match=r"model\.norm\.weight"):
        ws.lora_targets(path)


def test_lora_targets_refuses_unpaired_and_duplicate_modules(tmp_path):
    p = "base_model.model.model.layers.0.mlp.down_proj"
    with pytest.raises(ValueError, match="lora_B"):
        ws.lora_targets(save_adapter(tmp_path / "a.safetensors", [f"{p}.lora_A.weight"]))
    with pytest.raises(ValueError, match="lora_A"):
        ws.lora_targets(save_adapter(tmp_path / "b.safetensors", [f"{p}.lora_B.weight"]))
    # the same module named twice (with and without the "model." prefix): the graft would apply one
    q = "base_model.model.layers.0.mlp.down_proj"
    with pytest.raises(ValueError, match=r"layers\.0\.mlp\.down_proj"):
        ws.lora_targets(save_adapter(tmp_path / "c.safetensors", [f"{p}.lora_A.weight", f"{p}.lora_B.weight",
                                                                  f"{q}.lora_A.weight", f"{q}.lora_B.weight"]))
    ok = save_adapter(tmp_path / "d.safetensors", [f"{p}.lora_A.weight", f"{p}.lora_B.weight",
                                                   "base_model.model.lm_head.lora_A.weight",
                                                   "base_model.model.lm_head.lora_B.weight"])
    assert ws.lora_targets(ok) == ["layers.0.mlp.down_proj", "lm_head"]


def test_lora_config_refuses_modules_to_save(tmp_path):
    import json
    write_audit_adapter(tmp_path)
    with pytest.raises(ValueError, match=r"modules_to_save.*norm"):
        ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors"))
    for empty in (None, []):
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "modules_to_save": empty}))
        assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 2.0


def test_graft_lora_refuses_the_audit_adapter(tmp_path):
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    path = write_audit_adapter(tmp_path)
    x = ops.parameter([1, 3], np.float32, name="x")
    mm = ops.matmul(x, ops.constant(np.eye(3, dtype=np.float32)), False, True)
    mm.set_friendly_name("__module.model.model.language_model.layers.0.mlp.down_proj/ov_ext::linear/MatMul")
    with pytest.raises(ValueError, match="lora_embedding"):
        ws.graft_lora(ov.Model([ops.result(mm)], [x], "t"), path, 1.0)


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
    assert ws.lora_targets(path) == ["layers.0.mlp.down_proj"]
    assert ws.graft_lora(model, path, 2.0) == ["layers.0.mlp.down_proj"]
    out = ov.Core().compile_model(model, "CPU")([np.array([[1.0, 2.0, 3.0]], np.float32)])[0]
    # base = x ; lora = 2 * (x·1)·0.5 = sum(x) = 6
    np.testing.assert_allclose(out, [[7.0, 8.0, 9.0]], rtol=1e-6)


def test_graft_lora_only_touches_the_stage_layers(tmp_path):
    """A stage > 0 keeps shape-only copies of earlier layers (mask/position bookkeeping); grafting
    them grafts a module twice and fails the exporter's coverage check, so only the stage's own layers
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
    assert ws.graft_lora(ov.Model(outs, [x], "t"), path, 1.0, layers=range(4, 8), lm_head=False) == \
        ["layers.5.mlp.down_proj"]
    assert sorted(ws.graft_lora(ov.Model(outs, [x], "t"), path, 1.0, layers=range(4, 8), lm_head=True)) == \
        ["layers.5.mlp.down_proj", "lm_head"]
    assert sorted(ws.graft_lora(ov.Model(outs, [x], "t"), path, 1.0)) == \
        ["layers.0.mlp.down_proj", "layers.5.mlp.down_proj", "lm_head"]


def test_graft_lora_reports_a_module_once_per_matching_matmul(tmp_path):
    """Two MatMuls carrying the same module name are each grafted and each reported; a module with no
    MatMul is not reported. The exporter's coverage check (test_decision_export's two-MatMuls case,
    which stubs graft_lora) relies on exactly this."""
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    x = ops.parameter([1, 3], np.float32, name="x")
    outs = []
    for _ in range(2):
        mm = ops.matmul(x, ops.constant(np.eye(3, dtype=np.float32)), False, True)
        mm.set_friendly_name("__module.model.model.language_model.layers.0.mlp.down_proj/ov_ext::linear/MatMul")
        outs.append(ops.result(mm))
    keys = {}
    for m in ("layers.0.mlp.down_proj", "layers.0.mlp.up_proj"):
        keys[f"base_model.model.model.{m}.lora_A.weight"] = torch.ones(1, 3)
        keys[f"base_model.model.model.{m}.lora_B.weight"] = torch.full((3, 1), 0.5)
    path = str(tmp_path / "adapter_model.safetensors")
    save_file(keys, path)
    model = ov.Model(outs, [x], "t")
    assert ws.graft_lora(model, path, 2.0) == ["layers.0.mlp.down_proj", "layers.0.mlp.down_proj"]
    got = ov.Core().compile_model(model, "CPU")([np.array([[1.0, 2.0, 3.0]], np.float32)])
    for out in got.values():  # both branches grafted: x + 2 * sum(x) * 0.5
        np.testing.assert_allclose(out, [[7.0, 8.0, 9.0]], rtol=1e-6)


def _two_matmul_model(tmp_path):
    """x -> layers.0.mlp.up_proj -> layers.1.mlp.down_proj, each with its own random A/B (r=2), and
    the adapter for both. Returns (build() -> fresh ov.Model, adapter path)."""
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    rng = np.random.default_rng(41)
    w0, w1 = (rng.standard_normal((4, 4)).astype(np.float32) for _ in range(2))
    keys = {}
    for m in ("layers.0.mlp.up_proj", "layers.1.mlp.down_proj"):
        keys[f"base_model.model.model.{m}.lora_A.weight"] = torch.from_numpy(rng.standard_normal((2, 4)).astype(np.float32))
        keys[f"base_model.model.model.{m}.lora_B.weight"] = torch.from_numpy(rng.standard_normal((4, 2)).astype(np.float32))
    path = str(tmp_path / "adapter_model.safetensors")
    save_file(keys, path)

    def build():
        x = ops.parameter([2, 4], np.float32, name="x")
        h = ops.matmul(x, ops.constant(w0), False, True)
        h.set_friendly_name("__module.model.model.language_model.layers.0.mlp.up_proj/ov_ext::linear/MatMul")
        y = ops.matmul(h, ops.constant(w1), False, True)
        y.set_friendly_name("__module.model.model.language_model.layers.1.mlp.down_proj/ov_ext::linear/MatMul")
        return ov.Model([ops.result(y)], [x], "t")
    return build, path


def _run(model, x, scale=None):
    import openvino as ov
    feeds = {"x": x} if scale is None else {"x": x, "lora_scale": np.array(scale, np.float32)}
    # f32 execution: the CPU plugin may otherwise run in f16/bf16 and blur the comparison
    return ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(feeds)[0]


def test_switchable_graft_is_the_baked_graft_at_its_scale_and_the_base_model_at_zero(tmp_path):
    """System 2 runs the same tree as the unmodified base model by feeding lora_scale = 0; System 1
    feeds the trained scale and must get exactly what the baked graft computes."""
    build, path = _two_matmul_model(tmp_path)
    x = np.random.default_rng(7).standard_normal((2, 4)).astype(np.float32)
    base = _run(build(), x)
    baked = build()
    assert ws.graft_lora(baked, path, 2.0) == ["layers.0.mlp.up_proj", "layers.1.mlp.down_proj"]
    sw = build()
    assert ws.graft_lora(sw, path, 2.0, switchable=True) == ["layers.0.mlp.up_proj", "layers.1.mlp.down_proj"]
    assert not np.allclose(_run(baked, x), base)  # the adapter does move the output
    assert np.allclose(_run(sw, x, 2.0), _run(baked, x), rtol=1e-5, atol=1e-5)
    assert np.allclose(_run(sw, x, 0.0), base, rtol=1e-6, atol=1e-6)


def test_switchable_graft_adds_one_shared_scalar_lora_scale_input(tmp_path):
    build, path = _two_matmul_model(tmp_path)
    sw = build()
    ws.graft_lora(sw, path, 2.0, switchable=True)
    scales = [p for p in sw.get_parameters() if p.get_friendly_name() == "lora_scale"]
    assert len(scales) == 1 and len(sw.get_parameters()) == 2
    p, = scales
    assert p.get_element_type().get_type_name() == "f32" and list(p.get_partial_shape().to_shape()) == []
    assert "lora_scale" in p.output(0).get_names()
    # every branch multiplies by that one input, after (x A^T) B^T, LoRA term first in the Add
    muls = [t.get_node() for t in p.output(0).get_target_inputs()]
    assert len(muls) == 2 and all(m.get_type_name() == "Multiply" for m in muls)
    for m in muls:
        add, = [t.get_node() for t in m.output(0).get_target_inputs()]
        assert add.get_type_name() == "Add" and add.input_value(0).get_node() is m
    # B is no longer pre-multiplied by the scale: no Constant on any branch depends on it
    from safetensors.torch import load_file
    bs = [v.numpy() for k, v in load_file(path).items() if k.endswith("lora_B.weight")]
    consts = [op.get_data() for op in sw.get_ops() if op.get_type_name() == "Constant"]
    for b in bs:
        assert any(c.shape == b.T.shape and np.array_equal(c, b.T) for c in consts)


def test_switchable_graft_leaves_a_stage_without_branches_alone(tmp_path):
    """Only a stage that carries a grafted branch gains the input (a stage outside `layers` has none)."""
    build, path = _two_matmul_model(tmp_path)
    m = build()
    assert ws.graft_lora(m, path, 2.0, layers=range(4, 8), lm_head=False, switchable=True) == []
    assert [p.get_friendly_name() for p in m.get_parameters()] == ["x"]


# The graft as it was before the switchable option (Task 41), kept verbatim as the reference the
# default graft must still serialize to byte for byte.
_PRE_SWITCHABLE_GRAFT = r'''
def graft_lora(model, adapter_file, scale, layers=None, lm_head=True):
    from openvino import opset13 as ops
    from safetensors.torch import load_file
    tensors = load_file(adapter_file)
    pairs = {mod: {ab: tensors[k].float().numpy() for ab, k in side.items()}
             for mod, side in ws.lora_modules(tensors, adapter_file).items()}
    grafted = []
    for op in list(model.get_ordered_ops()):
        if op.get_type_name() != "MatMul":
            continue
        m = re.search(r"__module\.(?:model\.)*(?:language_model\.)?((?:layers\.\d+\.[\w.]+)|lm_head)/(?:aten|ov_ext)::linear/MatMul$",
                      op.get_friendly_name())
        if not m or m.group(1) not in pairs:
            continue
        if m.group(1) == "lm_head":
            if not lm_head:
                continue
        elif layers is not None and int(m.group(1).split(".")[1]) not in layers:
            continue
        a, b = pairs[m.group(1)]["A"], pairs[m.group(1)]["B"]
        x = op.input_value(0)
        xa = ops.matmul(x, ops.constant(np.ascontiguousarray(a.T, dtype=np.float32)), False, False)
        xab = ops.matmul(xa, ops.constant(np.ascontiguousarray(b.T * scale, dtype=np.float32)), False, False)
        consumers = list(op.output(0).get_target_inputs())
        added = ops.add(xab, op.output(0))
        for c in consumers:
            c.replace_source_output(added.output(0))
        grafted.append(m.group(1))
    model.validate_nodes_and_infer_types()
    return grafted
'''

_SAVE_GRAFTED = r'''
import re, sys
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops
sys.path.insert(0, {surgery!r})
import weight_swap as ws
{reference}
w = np.arange(16, dtype=np.float32).reshape(4, 4) / 7
outs = []
for name in ("layers.0.mlp.up_proj", "layers.1.mlp.down_proj", "lm_head"):
    x = ops.parameter([2, 4], np.float32, name="x" + str(len(outs)))
    mm = ops.matmul(x, ops.constant(w), False, True)
    mm.set_friendly_name("__module.model.model.language_model." + name + "/ov_ext::linear/MatMul")
    outs.append((ops.result(mm), x))
model = ov.Model([r for r, _ in outs], [x for _, x in outs], "t")
graft = graft_lora if {use_reference!r} else ws.graft_lora
graft(model, {adapter!r}, 2.0, layers=range(0, 2), lm_head=True)
ov.save_model(model, {out!r}, compress_to_fp16=False)
'''


def test_default_graft_serializes_byte_for_byte_as_before(tmp_path):
    """Default exports must not change: the default graft builds the same graph, node for node, as
    the graft before `switchable` (each run in a fresh interpreter, so auto-generated node names
    count from the same start)."""
    pytest.importorskip("openvino")
    torch = pytest.importorskip("torch")
    import subprocess
    from safetensors.torch import save_file
    rng = np.random.default_rng(3)
    keys = {}
    for m in ("model.layers.0.mlp.up_proj", "model.layers.1.mlp.down_proj", "lm_head"):
        keys[f"base_model.model.{m}.lora_A.weight"] = torch.from_numpy(rng.standard_normal((2, 4)).astype(np.float32))
        keys[f"base_model.model.{m}.lora_B.weight"] = torch.from_numpy(rng.standard_normal((4, 2)).astype(np.float32))
    adapter = str(tmp_path / "adapter_model.safetensors")
    save_file(keys, adapter)
    surgery = os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery")
    files = {}
    for tag, use_reference in (("before", True), ("now", False)):
        out = str(tmp_path / tag / "m.xml")
        script = _SAVE_GRAFTED.format(surgery=os.path.abspath(surgery), adapter=adapter, out=out,
                                      use_reference=use_reference,
                                      reference=_PRE_SWITCHABLE_GRAFT if use_reference else "")
        subprocess.run([sys.executable, "-c", script], check=True)
        files[tag] = (Path(out).read_bytes(), Path(out).with_suffix(".bin").read_bytes())
    assert b"Add" in files["now"][0] and b"Multiply" not in files["now"][0]
    assert files["now"] == files["before"]


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


def test_swap_stage_leaves_other_consumers_of_a_shared_float_constant_alone():
    """H31: a float constant also read by an op that is not one of FLOAT_KINDS keeps feeding that op
    the stock value; only the matched consumer gets the swapped weights."""
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    x = ops.parameter([1, 4], np.float32, name="x")
    c = ops.constant(np.full(4, 3.0, np.float32))
    norm = ops.multiply(x, c)
    norm.set_friendly_name(f"{_PFX}layers.0.input_layernorm/aten::mul/Multiply_1")
    other = ops.multiply(x, c)
    other.set_friendly_name("unrelated/aten::mul/Multiply")
    model = ov.Model([ops.result(norm), ops.result(other)], [x], "t")
    hf = np.array([0.5, -0.5, 1.0, 0.0], np.float32)
    report = ws.swap_stage(model, _FakeWeights({"layers.0.input_layernorm.weight": hf}))
    assert report["float"] == {0: ["input_layernorm.weight"]}
    np.testing.assert_array_equal(norm.input_value(1).get_node().get_data(), 1.0 + hf)
    np.testing.assert_array_equal(other.input_value(1).get_node().get_data(), [3.0] * 4)


def test_swap_stage_u4_projection_round_trips():
    """H32: a u4 [out, G, 128] projection is re-quantized per group with u4 codes and zero points."""
    ov = pytest.importorskip("openvino")
    from openvino import opset13 as ops
    out_f, g = 2, 2
    x = ops.parameter([1, g * 128], np.float32, name="x")
    name = f"{_PFX}layers.0.mlp.down_proj._openvino_orig_weight"
    w = ops.constant(np.full((out_f, g, 128), 8, np.uint8), dtype=ov.Type.u4)
    w.set_friendly_name(name)
    zp = ops.constant(np.full((out_f, g, 1), 8, np.uint8), dtype=ov.Type.u4)
    zp.set_friendly_name(name + "/zero_point")
    sc = ops.constant(np.ones((out_f, g, 1), np.float16), dtype=ov.Type.f16)
    sc.set_friendly_name(name + "/scale")
    f = lambda n: ops.convert(n, ov.Type.f32)  # noqa: E731
    deq = ops.reshape(ops.multiply(ops.subtract(f(w), f(zp)), f(sc)), ops.constant(np.array([out_f, g * 128])), False)
    model = ov.Model([ops.result(ops.matmul(x, deq, False, True))], [x], "t")
    hf = np.random.default_rng(4).normal(size=(out_f, g * 128)).astype(np.float32)
    report = ws.swap_stage(model, _FakeWeights({"layers.0.mlp.down_proj.weight": hf}))
    assert report["proj"] == {0: ["mlp.down_proj"]}
    q, zq, s = (_const(model, name + sfx) for sfx in ("", "/zero_point", "/scale"))
    assert q.get_element_type() == zq.get_element_type() == ov.Type.u4
    eq, ezp, es = ws.q_asym(hf.reshape(out_f, g, 128), 4, axis=-1)

    def unpack(c):  # u4 constants hold two codes per byte, low nibble first
        d = c.get_data().reshape(-1)
        return np.stack([d & 0xF, d >> 4], -1).reshape(-1)
    np.testing.assert_array_equal(unpack(q), eq.reshape(-1))
    np.testing.assert_array_equal(unpack(zq), ezp.reshape(-1))
    np.testing.assert_array_equal(s.get_data(), es)
    got = ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})([np.ones((1, g * 128), np.float32)])[0]
    want = ((eq.astype(np.float32) - ezp) * es.astype(np.float32)).reshape(out_f, -1).sum(-1)
    np.testing.assert_allclose(got[0], want, rtol=1e-4, atol=1e-3)


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
