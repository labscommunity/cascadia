"""Unit tests for cascadia shard's architecture detection + quirk gates.

Run from the repo root with:

    python -m pytest tools/tests/test_arch_detection.py -v

These tests exercise the detection logic in tools/export_shards.py against
a small `_FakeConfig` that mimics the relevant `transformers` config
attributes. The module imports torch/numpy at import time (for the rotary
class), so an env without them skips the whole file.
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
    """Stand-in for a transformers PretrainedConfig. SimpleNamespace gives
    attribute access for arbitrary keys, matching how detect_architecture /
    is_moe_config / check_export_quirks read fields via getattr."""

    def __init__(self, **kw):
        super().__init__()
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# detect_architecture — accept cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,arch_first,expected",
    [
        ("llama", "LlamaForCausalLM", "llama"),
        ("mistral", "MistralForCausalLM", "mistral"),
        ("qwen2", "Qwen2ForCausalLM", "qwen2"),
        ("qwen2", "Qwen2_5ForCausalLM", "qwen2"),
        ("qwen3", "Qwen3ForCausalLM", "qwen3"),
        ("phi3", "Phi3ForCausalLM", "phi"),
        # Gemma-1 (2-norm) and Gemma-2 (4-norm + softcap) must map to
        # DISTINCT tags — they have different decoder structures (#61).
        ("gemma", "GemmaForCausalLM", "gemma"),
        ("gemma2", "Gemma2ForCausalLM", "gemma2"),
        # DeepSeek-R1-Distill-Qwen reports the base model_type and rides it.
        ("qwen2", "Qwen2ForCausalLM", "qwen2"),
    ],
)
def test_detect_arch_accepts_known_families(model_type, arch_first, expected):
    cfg = _FakeConfig(model_type=model_type, architectures=[arch_first])
    assert export_shards.detect_architecture(cfg) == expected


def test_gemma2_is_distinct_from_gemma1():
    """Regression guard for #61: Gemma-2 must NOT collapse to the Gemma-1
    'gemma' tag (Gemma-2 uses Gemma2DecoderLayer with 4-norm + softcap)."""
    cfg = _FakeConfig(model_type="gemma2", architectures=["Gemma2ForCausalLM"])
    assert export_shards.detect_architecture(cfg) == "gemma2"


def test_qwen3_detected_before_qwen2():
    """qwen3 contains 'qwen' but must hit the qwen3 branch first (it has
    q_norm/k_norm handling qwen2 lacks)."""
    cfg = _FakeConfig(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    assert export_shards.detect_architecture(cfg) == "qwen3"


def test_multimodal_wrapper_unwraps_text_config():
    """A wrapper config with a plain-llama text_config should detect as
    'llama' off the inner config, not the opaque outer model_type."""
    cfg = _FakeConfig(
        model_type="some_wrapper",
        architectures=["SomeWrapperForCausalLM"],
        text_config=_FakeConfig(
            model_type="llama", architectures=["LlamaForCausalLM"]
        ),
    )
    assert export_shards.detect_architecture(cfg) == "llama"


def test_mistral3_wrapper_does_not_reject_text_path():
    """Mistral 3.x is a multimodal wrapper; the text inner is 'mistral'
    which we support. Detection should return 'mistral', not reject."""
    cfg = _FakeConfig(
        model_type="mistral3",
        architectures=["Mistral3ForConditionalGeneration"],
        text_config=_FakeConfig(
            model_type="mistral", architectures=["MistralForCausalLM"]
        ),
    )
    assert export_shards.detect_architecture(cfg) == "mistral"


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
# detect_architecture — explicit (non-MoE) rejection cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,arch_first,expected_keyword",
    [
        ("gemma3", "Gemma3ForCausalLM", "Gemma 3"),
        ("gemma3_text", "Gemma3ForConditionalGeneration", "Gemma 3"),
        ("gemma4", "Gemma4ForConditionalGeneration", "Gemma 4"),
        ("gemma4_text", "Gemma4TextForCausalLM", "Gemma 4"),
        ("gpt_oss", "GptOssForCausalLM", "gpt-oss"),
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


def test_gemma4_not_misdetected_as_gemma1():
    """The bug this PR fixes: 'gemma4' CONTAINS 'gemma', so without an
    explicit reject it falls through to the Gemma-1 path and is silently
    mis-exported. It must raise, never return 'gemma'."""
    cfg = _FakeConfig(model_type="gemma4", architectures=["Gemma4ForCausalLM"])
    with pytest.raises(export_shards.UnsupportedModelError):
        export_shards.detect_architecture(cfg)


def test_gemma4_rejected_through_multimodal_wrapper():
    """Inner text_config model_type='gemma4_text', outer 'gemma4' — detection
    inspects both, so either triggers the rejection."""
    cfg = _FakeConfig(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=_FakeConfig(model_type="gemma4_text", architectures=[]),
    )
    with pytest.raises(export_shards.UnsupportedModelError):
        export_shards.detect_architecture(cfg)


@pytest.mark.parametrize(
    "model_type,arch_first",
    [
        ("mixtral", "MixtralForCausalLM"),
        ("llama4", "Llama4ForConditionalGeneration"),
        ("qwen3_moe", "Qwen3MoeForCausalLM"),
    ],
)
def test_detect_arch_does_not_itself_reject_moe(model_type, arch_first):
    """detect_architecture is NOT the MoE gate — is_moe_config() is (#60).
    It returns a tag (or 'llama' fallback) for MoE families; the export
    flow rejects them via is_moe_config first. This guards against
    re-introducing the now-redundant MoE rejection here."""
    cfg = _FakeConfig(model_type=model_type, architectures=[arch_first])
    # Must not raise.
    tag = export_shards.detect_architecture(cfg)
    assert isinstance(tag, str)
    # ...but is_moe_config DOES reject them.
    assert export_shards.is_moe_config(cfg) is True


# ---------------------------------------------------------------------------
# is_moe_config — the MoE gate (#60)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,arch_first",
    [
        ("mixtral", "MixtralForCausalLM"),
        ("llama4", "Llama4ForConditionalGeneration"),
        ("qwen3_moe", "Qwen3MoeForCausalLM"),
        ("qwen2_moe", "Qwen2MoeForCausalLM"),
        ("deepseek_v3", "DeepseekV3ForCausalLM"),
        ("jamba", "JambaForCausalLM"),
        ("dbrx", "DbrxForCausalLM"),
    ],
)
def test_is_moe_config_rejects_moe_model_types(model_type, arch_first):
    cfg = _FakeConfig(model_type=model_type, architectures=[arch_first])
    assert export_shards.is_moe_config(cfg) is True


def test_is_moe_config_rejects_via_expert_count():
    """gpt-oss-style configs carry an explicit expert count even if the
    model_type isn't in the known set."""
    cfg = _FakeConfig(
        model_type="gpt_oss",
        architectures=["GptOssForCausalLM"],
        num_local_experts=128,
        num_experts_per_tok=4,
    )
    assert export_shards.is_moe_config(cfg) is True


def test_is_moe_config_rejects_dbrx_ffn_config():
    cfg = _FakeConfig(
        model_type="custom",
        architectures=["CustomForCausalLM"],
        ffn_config={"moe_num_experts": 16},
    )
    assert export_shards.is_moe_config(cfg) is True


@pytest.mark.parametrize(
    "model_type,arch_first",
    [
        ("llama", "LlamaForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
    ],
)
def test_is_moe_config_accepts_dense_families(model_type, arch_first):
    cfg = _FakeConfig(model_type=model_type, architectures=[arch_first])
    assert export_shards.is_moe_config(cfg) is False


# ---------------------------------------------------------------------------
# check_export_quirks — returns (hard, soft)
# ---------------------------------------------------------------------------


def test_quirks_moe_is_hard():
    cfg = _FakeConfig(model_type="custom_moe", num_local_experts=8)
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert any("mixture-of-experts" in q for q in hard)


def test_quirks_longrope_is_soft():
    cfg = _FakeConfig(
        model_type="phi3",
        rope_scaling={"type": "longrope", "short_factor": [1.0], "long_factor": [2.0]},
    )
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="phi")
    assert any("longrope" in q.lower() for q in soft)
    assert hard == []


def test_quirks_yarn_is_soft():
    cfg = _FakeConfig(model_type="x", rope_scaling={"rope_type": "yarn", "factor": 32.0})
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert any("yarn" in q.lower() for q in soft)
    assert hard == []


def test_quirks_softcap_on_gemma1_is_hard():
    """A model carrying softcap fields on the Gemma-1 path (arch_tag
    'gemma') is unsupported — Gemma-1 has no softcap handling."""
    cfg = _FakeConfig(
        model_type="gemma",
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    )
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("softcap" in q.lower() for q in hard)


def test_quirks_softcap_on_gemma2_is_NOT_flagged():
    """Regression guard for #61: the gemma2 path DOES apply attn + final
    softcap, so Gemma-2's softcap fields must produce no quirk. Flagging it
    as hard would (wrongly) abort the supported Gemma-2 export."""
    cfg = _FakeConfig(
        model_type="gemma2",
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
    )
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="gemma2")
    assert not any("softcap" in q.lower() for q in hard + soft)


def test_quirks_mixed_layer_types_is_soft():
    cfg = _FakeConfig(
        model_type="x",
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        sliding_window=4096,
    )
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="gemma2")
    assert any("layer_types" in q for q in soft)
    assert hard == []


def test_quirks_asymmetric_head_dim_is_hard():
    cfg = _FakeConfig(model_type="gemma4_text", head_dim=256, global_head_dim=512)
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("asymmetric head_dim" in q for q in hard)


def test_quirks_per_layer_embeddings_is_hard():
    cfg = _FakeConfig(model_type="gemma4_text", hidden_size_per_layer_input=256)
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("hidden_size_per_layer_input" in q for q in hard)


def test_quirks_kv_sharing_is_hard():
    cfg = _FakeConfig(model_type="gemma4_text", num_kv_shared_layers=20)
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="gemma")
    assert any("num_kv_shared_layers" in q for q in hard)


def test_quirks_qk_norm_on_non_qwen3_is_hard():
    cfg = _FakeConfig(model_type="llama", use_qk_norm=True)
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert any("use_qk_norm" in q for q in hard)


def test_quirks_qk_norm_on_qwen3_is_NOT_flagged():
    """The qwen3 path applies q_norm/k_norm, so use_qk_norm on qwen3 is fine."""
    cfg = _FakeConfig(model_type="qwen3", use_qk_norm=True)
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="qwen3")
    assert not any("use_qk_norm" in q for q in hard + soft)


def test_quirks_partial_rotary_is_not_flagged():
    """partial_rotary_factor < 1.0 is now supported (TracedRotaryEmbedding
    zero-pads inv_freq) — it must produce no quirk."""
    cfg = _FakeConfig(model_type="phi", partial_rotary_factor=0.4)
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="phi")
    assert not any("partial_rotary" in q for q in hard + soft)


def test_quirks_clean_for_well_supported_model():
    """Llama 3.1 8B (llama3 rope scaling) trips none of the gates."""
    cfg = _FakeConfig(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
        },
    )
    hard, soft = export_shards.check_export_quirks(cfg, arch_tag="llama")
    assert hard == [] and soft == []


# ---------------------------------------------------------------------------
# Partial rotary in TracedRotaryEmbedding (needs torch)
# ---------------------------------------------------------------------------


def test_traced_rotary_partial_inv_freq_width_and_denominator():
    """Partial rotary emits inv_freq of width rotary_dim/2 (NO zero padding)
    with the ROTARY dim as the denominator — matching transformers Phi/Phi3.
    Guards against the head_dim-denominator bug (which matched cos/sin VALUES
    but, applied with full-head rotate_half, corrupted ~every dim)."""
    import torch

    head_dim, prf, theta = 96, 0.75, 10_000.0
    rotary = export_shards.TracedRotaryEmbedding(
        head_dim, rope_theta=theta, partial_rotary_factor=prf
    )
    rotary_dim = int(prf * head_dim)  # 72
    assert rotary.rotary_dim == rotary_dim
    inv_freq = rotary.inv_freq
    assert inv_freq.shape[0] == rotary_dim // 2  # 36, NOT head_dim//2=48
    assert (inv_freq != 0).all()  # no zero padding
    expected = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    assert torch.allclose(inv_freq, expected, atol=1e-6)
    # The old (wrong) head_dim denominator must NOT match.
    wrong = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / head_dim))
    assert not torch.allclose(inv_freq, wrong, atol=1e-4)


@pytest.mark.parametrize(
    "head_dim,prf",
    [
        (128, 1.0),   # full rotary (regression guard — must stay byte-exact)
        (128, 0.75),  # Phi-4-mini
        (80, 0.4),    # Phi-2
        (64, 0.5),    # Phi-1.5
        (64, 0.25),   # StableLM-2
    ],
)
def test_partial_rotary_APPLIED_matches_hf_phi3(head_dim, prf):
    """THE decisive correctness check: apply cascadia's rotary to q/k and
    compare the RESULT (not just cos/sin) to HF's apply_rotary_pos_emb. The
    earlier cos/sin-only comparison passed while the applied rotation was wrong
    on ~88% of dims."""
    import torch

    try:
        from transformers import Phi3Config
        from transformers.models.phi3 import modeling_phi3 as m
    except Exception:  # pragma: no cover
        pytest.skip("transformers Phi3 not available")

    theta = 10_000.0
    cfg = Phi3Config(
        hidden_size=head_dim,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_hidden_layers=2,
        partial_rotary_factor=prf,
        rope_theta=theta,
        max_position_embeddings=512,
        rope_scaling=None,
    )
    torch.manual_seed(0)
    pos = torch.arange(8).unsqueeze(0)
    q = torch.randn(1, 2, 8, head_dim)
    k = torch.randn(1, 2, 8, head_dim)

    hf_cos, hf_sin = m.Phi3RotaryEmbedding(config=cfg)(q, pos)
    q_hf, k_hf = m.apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    rotary = export_shards.TracedRotaryEmbedding(
        head_dim, rope_theta=theta, partial_rotary_factor=prf
    )
    cos, sin = rotary(pos, torch.float32)
    q_c, k_c = export_shards.apply_rotary(q, k, cos, sin)

    assert torch.allclose(q_hf, q_c, atol=1e-4), (q_hf - q_c).abs().max().item()
    assert torch.allclose(k_hf, k_c, atol=1e-4), (k_hf - k_c).abs().max().item()


def test_partial_rotary_passes_through_contiguous_tail():
    """Without transformers: apply_rotary rotates the leading rotary_dim slice
    and leaves the CONTIGUOUS tail [rotary_dim:] untouched (HF layout) — not
    the split-across-both-halves layout the buggy zero-pad version produced."""
    import torch

    head_dim, prf = 64, 0.5
    rotary_dim = int(prf * head_dim)  # 32
    rotary = export_shards.TracedRotaryEmbedding(
        head_dim, rope_theta=10_000.0, partial_rotary_factor=prf
    )
    cos, sin = rotary(torch.arange(3).unsqueeze(0), torch.float32)
    assert cos.shape[-1] == rotary_dim  # cos spans only the rotary dims
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, head_dim)
    k = torch.randn(1, 2, 3, head_dim)
    q_rot, k_rot = export_shards.apply_rotary(q, k, cos, sin)
    # Contiguous tail [rotary_dim:] is passed through byte-for-byte.
    assert torch.equal(q_rot[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_rot[..., rotary_dim:], k[..., rotary_dim:])
    # Leading rotary_dim block DOES rotate at non-zero positions.
    assert not torch.allclose(
        q_rot[:, :, 1:, :rotary_dim], q[:, :, 1:, :rotary_dim], atol=1e-3
    )


def test_traced_rotary_full_rotary_unchanged_default():
    import torch

    head_dim = 128
    rope_theta = 500_000.0
    rotary = export_shards.TracedRotaryEmbedding(head_dim, rope_theta=rope_theta)
    assert rotary.rotary_dim == head_dim
    expected = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    assert torch.allclose(rotary.inv_freq, expected)


def test_traced_rotary_rejects_zero_rotated_dims():
    # partial_rotary_factor too small to leave any rotated dims.
    with pytest.raises(ValueError):
        export_shards.TracedRotaryEmbedding(
            head_dim=4, rope_theta=10_000.0, partial_rotary_factor=0.1
        )


def test_traced_rotary_rejects_nope_zero_factor():
    # partial_rotary_factor=0.0 (NoPE) -> rotary_dim=0 -> reject (NOT silently
    # coerced to full rotary).
    with pytest.raises(ValueError):
        export_shards.TracedRotaryEmbedding(
            head_dim=64, rope_theta=10_000.0, partial_rotary_factor=0.0
        )


def test_traced_rotary_rejects_odd_rotary_dim():
    # int(prf*head_dim) odd -> can't split into pairs -> reject rather than
    # silently diverge from HF by a dim.
    with pytest.raises(ValueError):
        export_shards.TracedRotaryEmbedding(
            head_dim=10, rope_theta=10_000.0, partial_rotary_factor=0.5  # int(5)=5 odd
        )


def test_is_moe_config_rejects_qwen35_moe_model_type():
    """Qwen3.5/Qwen3.6 (qwen3_5_moe) is MoE even though its architecture
    class ends in ForConditionalGeneration, not MoeForCausalLM (#77)."""
    cfg = _FakeConfig(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
    )
    assert export_shards.is_moe_config(cfg) is True


def test_is_moe_config_rejects_nested_text_config_experts():
    """Qwen3.6-35B-A3B is a VLM whose expert fields live under text_config;
    the outer config carries no expert counts (#77). The gate must unwrap."""
    cfg = _FakeConfig(
        model_type="some_future_vlm",
        architectures=["FutureVlmForConditionalGeneration"],
        text_config=_FakeConfig(
            model_type="some_future_vlm_text",
            num_experts=256,
            num_experts_per_tok=8,
        ),
    )
    assert export_shards.is_moe_config(cfg) is True


def test_is_moe_config_accepts_qwen35_dense():
    """Qwen3.8-27B is the DENSE member of the qwen3_5 family: outer
    model_type qwen3_5, nested text_config qwen3_5_text, no expert fields.
    It must not be mistaken for the MoE sibling."""
    cfg = _FakeConfig(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=_FakeConfig(model_type="qwen3_5_text", hidden_size=5120),
    )
    assert export_shards.is_moe_config(cfg) is False


def test_detect_arch_rejects_qwen35_dense_hybrid():
    """"qwen3_5" contains "qwen3": without an explicit guard the dense
    hybrid would fall through to the Qwen3 decoder path and be silently
    mis-exported. The rejection must point at the surgery route."""
    cfg = _FakeConfig(
        model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"]
    )
    with pytest.raises(export_shards.UnsupportedModelError, match="surgery|int4-ov"):
        export_shards.detect_architecture(cfg)


def test_detect_arch_rejects_qwen35_text_inner_type():
    cfg = _FakeConfig(model_type="qwen3_5_text", architectures=["Qwen3_5ForCausalLM"])
    with pytest.raises(export_shards.UnsupportedModelError):
        export_shards.detect_architecture(cfg)


def test_quirks_linear_attention_is_hard():
    """Gated DeltaNet layers are a different model, not a lossy quirk."""
    cfg = _FakeConfig(
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        head_dim=256,
    )
    hard, soft = export_shards.check_export_quirks(cfg, "qwen3")
    assert any("linear_attention" in h for h in hard), (hard, soft)
    assert not any("mixed layer_types" in s for s in soft), soft
