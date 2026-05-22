"""Unit tests for cascadia shard's architecture detection + quirk gates.

Run from the repo root with:

    python -m pytest tools/tests/test_arch_detection.py -v

These tests do NOT import torch / openvino / transformers — they exercise
just the pure-Python detection logic in tools/export_shards.py against
a small `_FakeConfig` that mimics the relevant `transformers` config
attributes.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

# Add tools/ to sys.path so we can import export_shards as a module.
_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

# We import the module directly. It needs `numpy`/`torch` at import time
# for the rotary class, so the test envs that lack them will skip.
try:
    import export_shards  # noqa: F401
except ImportError as exc:
    if "torch" in str(exc) or "numpy" in str(exc):
        pytest.skip(
            f"export_shards.py needs torch/numpy at import time: {exc}",
            allow_module_level=True,
        )
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConfig(types.SimpleNamespace):
    """Stand-in for a transformers PretrainedConfig.

    `SimpleNamespace` gives attribute access for arbitrary keys, which
    matches how detect_architecture / check_export_quirks use getattr.
    """

    def __init__(self, **kw):
        super().__init__()
        for k, v in kw.items():
            setattr(self, k, v)


def _multimodal_wrapper(**inner_kw):
    """Build a config that mimics Gemma 3/4 / Llama 4 / Mistral 3.x —
    outer wrapper with model_type and text_config inner.
    """
    inner = _FakeConfig(**inner_kw)
    outer_kw = {"text_config": inner}
    # If the caller provided outer-specific fields they will already be
    # in inner_kw; tests use _FakeConfig directly for outer-only configs.
    return _FakeConfig(**outer_kw)


# ---------------------------------------------------------------------------
# detect_architecture — accept cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,arch_first,expected",
    [
        ("llama", "LlamaForCausalLM", "llama"),
        ("llama", "MetaLlamaForCausalLM", "llama"),
        ("mistral", "MistralForCausalLM", "mistral"),
        ("qwen2", "Qwen2ForCausalLM", "qwen2"),
        ("qwen2", "Qwen2_5ForCausalLM", "qwen2"),
        ("qwen3", "Qwen3ForCausalLM", "qwen3"),
        ("phi3", "Phi3ForCausalLM", "phi"),
        ("gemma", "GemmaForCausalLM", "gemma"),
        ("gemma2", "Gemma2ForCausalLM", "gemma"),
        ("gemma3_text", "Gemma3ForCausalLM", "gemma"),
        # DeepSeek-R1-Distill-Qwen-7B reports as Qwen2 (the base).
        ("qwen2", "Qwen2ForCausalLM", "qwen2"),
        # DeepSeek-V2-Lite has its own model_type but rides the llama
        # path via fall-through (the warning string is checked elsewhere).
    ],
)
def test_detect_arch_accepts_known_families(model_type, arch_first, expected):
    cfg = _FakeConfig(model_type=model_type, architectures=[arch_first])
    assert export_shards.detect_architecture(cfg) == expected


def test_qwen3_detected_before_qwen2():
    """Ordering matters: qwen3 contains 'qwen' but must hit the qwen3
    branch first (it has q_norm/k_norm handling that qwen2 lacks)."""
    cfg = _FakeConfig(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    assert export_shards.detect_architecture(cfg) == "qwen3"


def test_multimodal_wrapper_unwraps_text_config():
    """Configs like Llama4ForConditionalGeneration nest the text-tower
    config under text_config. detect_architecture should dispatch off
    the inner model_type when the outer has no useful one — except for
    families where the wrapper itself is the rejection trigger (llama4,
    mistral3, gemma4). The inner config below is a plain llama, so the
    wrapper should NOT trigger an llama4 rejection."""
    cfg = _FakeConfig(
        model_type="some_wrapper",
        architectures=["SomeWrapperForCausalLM"],
        text_config=_FakeConfig(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
        ),
    )
    assert export_shards.detect_architecture(cfg) == "llama"


# ---------------------------------------------------------------------------
# detect_architecture — explicit rejection cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,arch_first,expected_keyword",
    [
        ("llama4", "Llama4ForConditionalGeneration", "Llama 4"),
        ("llama4_text", "Llama4TextForCausalLM", "Llama 4"),
        ("gemma4", "Gemma4ForConditionalGeneration", "Gemma 4"),
        ("gemma4_text", "Gemma4TextForCausalLM", "Gemma 4"),
        ("qwen3_moe", "Qwen3MoeForCausalLM", "Qwen3-MoE"),
        ("mixtral", "MixtralForCausalLM", "Mixtral"),
        ("gpt_oss", "GptOssForCausalLM", "gpt-oss"),
        ("deepseek_v3", "DeepseekV3ForCausalLM", "DeepSeek-V3"),
        ("jamba", "JambaForCausalLM", "Mamba"),
        ("falcon_mamba", "FalconMambaForCausalLM", "Mamba"),
    ],
)
def test_detect_arch_rejects_known_unsupported_families(
    model_type, arch_first, expected_keyword
):
    cfg = _FakeConfig(model_type=model_type, architectures=[arch_first])
    with pytest.raises(export_shards.UnsupportedModelError) as exc:
        export_shards.detect_architecture(cfg)
    assert expected_keyword in str(exc.value)


def test_gemma4_rejected_even_through_multimodal_wrapper():
    """Inner config has model_type='gemma4_text', outer is 'gemma4'.
    Detection should fire on either."""
    cfg = _FakeConfig(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=_FakeConfig(
            model_type="gemma4_text",
            architectures=[],
        ),
    )
    with pytest.raises(export_shards.UnsupportedModelError):
        export_shards.detect_architecture(cfg)


def test_mistral3_wrapper_does_not_reject_text_path():
    """Mistral 3.x is a multimodal wrapper; the text inner is 'mistral'
    which we DO support. Detection should return 'mistral', not reject."""
    cfg = _FakeConfig(
        model_type="mistral3",
        architectures=["Mistral3ForConditionalGeneration"],
        text_config=_FakeConfig(
            model_type="mistral",
            architectures=["MistralForCausalLM"],
        ),
    )
    assert export_shards.detect_architecture(cfg) == "mistral"


def test_deepseek_v2_lite_is_not_rejected():
    """DeepSeek-V2-Lite is standard attention + RoPE (not MLA). The
    rejection for deepseek_v2 explicitly excludes 'lite'."""
    cfg = _FakeConfig(
        model_type="deepseek_v2_lite",
        architectures=["DeepseekV2LiteForCausalLM"],
    )
    # Falls through to the warning + Llama fallback (no exception).
    assert export_shards.detect_architecture(cfg) == "llama"


def test_unknown_family_falls_through_to_llama_with_warning(capsys):
    cfg = _FakeConfig(
        model_type="brand_new_model_2027",
        architectures=["BrandNewModel2027ForCausalLM"],
    )
    assert export_shards.detect_architecture(cfg) == "llama"
    captured = capsys.readouterr()
    assert "unknown model_type" in captured.out
    assert "brand_new_model_2027" in captured.out


# ---------------------------------------------------------------------------
# check_export_quirks
# ---------------------------------------------------------------------------


def test_quirks_flag_moe_via_num_local_experts():
    cfg = _FakeConfig(
        model_type="custom_moe",
        architectures=["CustomMoEForCausalLM"],
        num_local_experts=8,
        num_experts_per_tok=2,
    )
    quirks = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert any("MoE" in q for q in quirks)


def test_quirks_flag_longrope_scaling():
    cfg = _FakeConfig(
        model_type="phi3",
        rope_scaling={"type": "longrope", "short_factor": [1.0], "long_factor": [2.0]},
    )
    quirks = export_shards.check_export_quirks(cfg, arch_tag="phi")
    assert any("longrope" in q.lower() for q in quirks)


def test_quirks_flag_yarn_scaling():
    cfg = _FakeConfig(
        model_type="gpt_oss",
        rope_scaling={"type": "yarn", "factor": 32.0},
    )
    quirks = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert any("yarn" in q.lower() for q in quirks)


def test_quirks_flag_softcap():
    cfg = _FakeConfig(
        model_type="gemma2",
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    )
    quirks = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("softcap" in q.lower() for q in quirks)


def test_quirks_flag_mixed_layer_types():
    cfg = _FakeConfig(
        model_type="gemma3_text",
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
    )
    quirks = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("layer_types" in q for q in quirks)


def test_quirks_flag_asymmetric_head_dim():
    cfg = _FakeConfig(
        model_type="gemma4_text", head_dim=256, global_head_dim=512
    )
    quirks = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("asymmetric head_dim" in q for q in quirks)


def test_quirks_flag_pli():
    cfg = _FakeConfig(model_type="gemma4_text", hidden_size_per_layer_input=256)
    quirks = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("hidden_size_per_layer_input" in q for q in quirks)


def test_quirks_flag_kv_sharing():
    cfg = _FakeConfig(model_type="gemma4_text", num_kv_shared_layers=20)
    quirks = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("num_kv_shared_layers" in q for q in quirks)


def test_quirks_flag_qk_norm_on_non_qwen3():
    """use_qk_norm=True on a non-Qwen3 path means the q_norm/k_norm
    won't be applied (only the qwen3 branch checks for them)."""
    cfg = _FakeConfig(model_type="llama", use_qk_norm=True)
    quirks = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert any("use_qk_norm" in q for q in quirks)


def test_quirks_partial_rotary_does_NOT_warn_anymore():
    """partial_rotary_factor < 1.0 is now handled by
    TracedRotaryEmbedding (it pads inv_freq with zeros). No warning
    should fire."""
    cfg = _FakeConfig(model_type="phi3", partial_rotary_factor=0.75)
    quirks = export_shards.check_export_quirks(cfg, arch_tag="phi")
    assert not any("partial_rotary_factor" in q for q in quirks)


def test_quirks_clean_for_well_supported_model():
    """Llama 3.1 8B has none of the gates we check."""
    cfg = _FakeConfig(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    quirks = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert quirks == []


# ---------------------------------------------------------------------------
# Partial rotary in TracedRotaryEmbedding
# ---------------------------------------------------------------------------


def test_traced_rotary_partial_pads_inv_freq_with_zeros():
    """For partial_rotary_factor=0.75 and head_dim=96, the leading
    36 / 48 inv_freq entries should be the standard formula, and the
    last 12 should be zero (so cos=1, sin=0 — no rotation)."""
    try:
        import torch  # noqa
    except ImportError:
        pytest.skip("torch not available")
    head_dim = 96
    rope_theta = 10_000.0
    rotary = export_shards.TracedRotaryEmbedding(
        head_dim, rope_theta=rope_theta, partial_rotary_factor=0.75
    )
    inv_freq = rotary.inv_freq
    assert inv_freq.shape[0] == head_dim // 2  # 48
    # Last 12 entries (the un-rotated dims) should be zero.
    rot_pairs = int(0.75 * head_dim) // 2  # 36
    nope_pairs = head_dim // 2 - rot_pairs  # 12
    assert (inv_freq[-nope_pairs:] == 0).all()
    assert (inv_freq[:rot_pairs] != 0).all()


def test_traced_rotary_full_rotary_unchanged_default():
    """partial_rotary_factor=1.0 (default) should be identical to the
    pre-existing behaviour: standard inv_freq across all dims."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    head_dim = 128
    rope_theta = 500_000.0
    rotary = export_shards.TracedRotaryEmbedding(
        head_dim, rope_theta=rope_theta
    )
    expected = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    assert torch.allclose(rotary.inv_freq, expected)


def test_traced_rotary_rejects_zero_rotated_dims():
    """partial_rotary_factor too small to leave any rotated dims should
    error out, not silently return all-zero rotation."""
    try:
        import torch  # noqa
    except ImportError:
        pytest.skip("torch not available")
    with pytest.raises(ValueError):
        export_shards.TracedRotaryEmbedding(
            head_dim=4, rope_theta=10_000.0, partial_rotary_factor=0.1
        )
