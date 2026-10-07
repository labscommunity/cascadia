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


def test_sym8_rows():
    w = np.array([[1.0, -2.0], [0.0, 0.0]], dtype=np.float32)
    q, s = ws.q_sym8(w)
    np.testing.assert_array_equal(q[0], [64, -127])
    assert s[1, 0] > 0  # zero row keeps a positive scale


def test_check_swap_requires_every_layer_kind():
    full = {"proj": {0: 4, 1: 7}, "conv": {0: 1},
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


def test_lora_scale_comes_from_adapter_config(tmp_path):
    import json
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64, "rank_pattern": {}, "alpha_pattern": {}}))
    assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 2.0
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "use_rslora": True}))
    assert ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors")) == 8.0
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "rank_pattern": {"q_proj": 8}}))
    with pytest.raises(ValueError, match="rank_pattern"):
        ws.lora_scale_from_config(str(tmp_path / "adapter_model.safetensors"))


def test_float_transform_table():
    w = np.array([0.5, -0.25], dtype=np.float32)
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["input_layernorm.weight"]](w), 1 + w)
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["linear_attn.A_log"]](w), np.exp(w))
    assert np.allclose(ws.TRANSFORMS[ws.FLOAT_KIND_TRANSFORM["linear_attn.dt_bias"]](w), w)


def test_graft_lora_adds_scaled_branch():
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
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "lora_test.safetensors")
    save_file({"base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": a,
               "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": b}, path)
    assert ws.graft_lora(model, path, 2.0) == 1
    out = ov.Core().compile_model(model, "CPU")([np.array([[1.0, 2.0, 3.0]], np.float32)])[0]
    # base = x ; lora = 2 * (x·1)·0.5 = sum(x) = 6
    np.testing.assert_allclose(out, [[7.0, 8.0, 9.0]], rtol=1e-6)


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
