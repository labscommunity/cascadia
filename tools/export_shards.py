#!/usr/bin/env python3
"""cascadia standalone model exporter.

Takes a HuggingFace model (Llama, Mistral, Qwen2, or any decoder-only LLM
that follows the same conventions) and produces a cascadia shard directory:

    shards/
      pipeline_config.json
      tokenizer/
        tokenizer.json + config.json + special_tokens_map.json
      stage_0/
        openvino_model.xml + .bin
        stage_config.json
      stage_1/
        openvino_model.xml + .bin
        stage_config.json
      ...

This script is invoked by `cascadia shard`. It is also runnable standalone
for users who want fine-grained control:

    python export_shards.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --output-dir ~/cascadia-shards/llama-8b-2stage \
        --num-stages 2 \
        --quantization int4

Architecture support: The script uses HuggingFace's AutoModelForCausalLM
to load layers, so any model whose decoder layers expose the standard
attribute names (`self_attn.{q_proj,k_proj,v_proj,o_proj}`,
`mlp`, `input_layernorm`, `post_attention_layernorm`) and uses RoPE
rotary will work. This covers:
  - Llama 1/2/3/3.1/3.2 family
  - Mistral 7B / Mixtral (single-expert per shard only)
  - Qwen2 / Qwen2.5
  - Some Phi-3 variants (those with standard attention)
  - DeepSeek-V2-Lite

Models with non-standard architectures (sliding-window attention with
custom kernels, MoE routing per layer, multimodal) may export but won't
match HF reference outputs; benchmark before trusting the result.
"""

import argparse
import gc
import glob
import json
import math
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture detection + pre-export config sanity
# ---------------------------------------------------------------------------


class UnsupportedModelError(RuntimeError):
    """Raised when the model needs a feature the exporter doesn't honour.

    Distinct from generic RuntimeError so callers can catch it and emit a
    cleaner user-facing message.
    """


def _text_config(config):
    """Unwrap a multimodal wrapper if present.

    Models like Gemma 3, Gemma 4, Llama 4, Mistral 3.x ship a multimodal
    config with the text backbone under `config.text_config`. We dispatch
    detection off the inner text config when present.
    """
    inner = getattr(config, "text_config", None)
    if inner is not None and hasattr(inner, "model_type"):
        return inner
    return config


class _RawConfigNS:
    """Lightweight stand-in for a transformers config built directly from
    raw config.json.

    Pre-flight detection needs to inspect model_type and architectures
    BEFORE asking transformers' AutoConfig to load the model (because
    AutoConfig raises ValueError on unknown model_types for very-new
    families like gemma4, blocking our specific rejection message).
    This class wraps the raw JSON dict + recurses into text_config so
    detect_architecture sees the same shape it would on a real config.
    """

    def __init__(self, cfg_dict):
        self._cfg = dict(cfg_dict)
        if isinstance(cfg_dict.get("text_config"), dict):
            self.text_config = _RawConfigNS(cfg_dict["text_config"])

    def __getattr__(self, name):
        try:
            return self._cfg[name]
        except KeyError:
            raise AttributeError(name)


def _print_unsupported(exc: "UnsupportedModelError") -> None:
    """Single chokepoint for the user-facing rejection block."""
    print("", flush=True)
    print("=" * 60, flush=True)
    print("ERROR: model architecture not supported by cascadia shard.", flush=True)
    print("=" * 60, flush=True)
    print(str(exc), flush=True)
    print("", flush=True)
    print(
        "If you really want to attempt the export anyway (e.g. for "
        "experimental work), set CASCADIA_ALLOW_LOSSY_EXPORT=1.",
        flush=True,
    )


def detect_architecture(config) -> str:
    """Return a short tag identifying the model family. Used to pick the
    right decoder-layer class.

    Order matters: more-specific matches (qwen3, mistral3, gemma4)
    are checked before generic substrings (qwen, mistral, gemma).

    Raises UnsupportedModelError for families we know about but can't
    honour today (MoE, Gemma 4, Llama 4, gpt-oss, MLA-based DeepSeek
    full). The error message names the family and points at a doc.

    For genuinely unknown model_types, returns "llama" with a verbose
    warning, since most modern decoder-only LMs have a Llama-shaped
    layer interface and load via load_state_dict(strict=False).
    """
    inner = _text_config(config)
    model_type = getattr(inner, "model_type", "").lower()
    arch_list = getattr(inner, "architectures", []) or []
    arch_first = (arch_list[0] if arch_list else "").lower()
    # Also inspect the OUTER wrapper for multimodal-only model_types,
    # because some configs (e.g. Pixtral) put the family-identifying
    # type on the outer, not the inner text_config.
    outer_type = getattr(config, "model_type", "").lower()
    outer_arch_list = getattr(config, "architectures", []) or []
    outer_arch_first = (outer_arch_list[0] if outer_arch_list else "").lower()

    def _has(needle):
        return (
            needle in model_type
            or needle in arch_first
            or needle in outer_type
            or needle in outer_arch_first
        )

    # ---- Explicit reject list — known families we don't yet support ----

    # Llama 4: MoE + iRoPE (NoPE every 4 layers) + QK-norm + chunked attn.
    if _has("llama4") or "llama-4" in (outer_type + model_type):
        raise UnsupportedModelError(
            "Llama 4 (Scout/Maverick — model_type 'llama4' / "
            "'llama4_text') is MoE with iRoPE (NoPE every 4 layers), "
            "QK-norm, and chunked attention. The generic exporter does "
            "not yet support any of these. See "
            "docs/architectures/moe.md and the Llama 4 row of "
            "docs/SHARDING.md."
        )

    # Gemma 4: per-layer-type asymmetric attention (head_dim,
    # num_kv_heads), per-layer-type RoPE incl. 'proportional' scaling,
    # KV-shared layers, per-layer embeddings, restored softcap.
    if _has("gemma4") or _has("gemma_4") or _has("gemma-4"):
        raise UnsupportedModelError(
            "Gemma 4 (model_type 'gemma4' / 'gemma4_text', April 2026) "
            "requires per-layer-type asymmetric attention, per-layer-type "
            "RoPE (incl. the new 'proportional' scaling), KV-shared "
            "layers (E2B/E4B), per-layer embeddings (E2B/E4B), and "
            "restored final logit softcap. The generic exporter cannot "
            "honour these — see docs/architectures/gemma4-support.md "
            "for the port plan and rainier's working Python prototype."
        )

    # Qwen3-MoE — 128 experts, top-8 routing, no shared expert.
    if _has("qwen3_moe") or _has("qwen3moe") or "qwen3-moe" in (
        outer_type + model_type
    ):
        raise UnsupportedModelError(
            "Qwen3-MoE (model_type 'qwen3_moe') routes 8 of 128 experts "
            "per token. cascadia's generic sharder does not yet support "
            "MoE — see docs/architectures/moe.md. For a one-off MoE "
            "demo, see cascadia-engine-sparse-moe (Kimi K2.6 only)."
        )

    # Mixtral — would otherwise fall through to 'mistral' and silently
    # miswire block_sparse_moe as a dense MLP.
    if _has("mixtral"):
        raise UnsupportedModelError(
            "Mixtral (model_type 'mixtral') is MoE (8 experts top-2 / "
            "8x22B). The generic exporter would silently treat the "
            "block_sparse_moe layer as a dense MLP and produce garbage. "
            "See docs/architectures/moe.md. For a one-off MoE demo, "
            "see cascadia-engine-sparse-moe."
        )

    # gpt-oss — OpenAI MoE (Aug 2025) with sigmoid-routed top-k +
    # alternating sliding/full + YARN + MXFP4 quant.
    if _has("gpt_oss") or _has("gpt-oss") or "gptoss" in model_type:
        raise UnsupportedModelError(
            "gpt-oss (model_type 'gpt_oss', August 2025) is OpenAI's "
            "open-weight MoE. Sigmoid-routed top-k experts, alternating "
            "sliding/full attention per layer, YARN RoPE, MXFP4 native "
            "quant. Multiple features the generic exporter does not "
            "support. See docs/architectures/moe.md."
        )

    # DeepSeek-V2/V3/R1 (the full models — MLA-based, distinct from
    # R1-Distill-{Llama,Qwen} which ride the existing paths).
    if _has("deepseek_v3") or _has("deepseek-v3") or _has("deepseekv3"):
        raise UnsupportedModelError(
            "DeepSeek-V3 / R1 (model_type 'deepseek_v3') uses Multi-head "
            "Latent Attention (MLA — q_lora_rank, kv_lora_rank, split "
            "qk_nope_head_dim / qk_rope_head_dim). This is a full "
            "attention-block rewrite. The R1-Distill-{Qwen,Llama} models "
            "report their base model_type and ride those paths instead "
            "— use those if you want R1-style reasoning at AI-PC sizes."
        )
    if _has("deepseek_v2") and not _has("lite"):
        raise UnsupportedModelError(
            "DeepSeek-V2 (full, not Lite) uses MLA. Use "
            "DeepSeek-V2-Lite (which is standard attention + RoPE) or "
            "the R1-Distill variants."
        )

    # Jamba / Falcon-Mamba / Granite 4 / Nemotron-H — hybrid Mamba.
    if _has("jamba") or _has("falcon_mamba") or _has("mamba"):
        raise UnsupportedModelError(
            "Mamba / hybrid Mamba-Transformer architectures "
            "(Jamba, Falcon-Mamba, Granite 4, Nemotron-H) need a Mamba "
            "kernel. Out of scope for OpenVINO IR export."
        )

    # ---- Accept list — families that load and run ----

    if _has("llama"):
        return "llama"
    # Mistral 3.x is a multimodal wrapper around a Mistral text backbone;
    # the text inner config reports model_type "mistral" which the next
    # branch catches. Outer "mistral3" was inspected above by _has().
    if _has("mistral"):
        return "mistral"
    if _has("qwen3"):
        return "qwen3"
    if _has("qwen2"):
        return "qwen2"
    if _has("phi"):
        return "phi"
    # Gemma 1, 2, 3 all match "gemma". Gemma 4 was rejected above.
    if _has("gemma"):
        return "gemma"

    print(
        f"  warning: unknown model_type={model_type!r}, "
        f"architectures={arch_list!r}; falling back to Llama "
        "decoder layer. This works for ~80% of post-Llama-2 dense "
        "decoder-only LMs. Common failure modes if your model is "
        "atypical: (a) non-RoPE rotary, (b) QK-norm before RoPE "
        "(silently dropped — see Qwen3 path for the right detection), "
        "(c) MoE routing, (d) sliding-window attention, (e) per-layer "
        "embeddings. Set CASCADIA_DEBUG_ARCH=1 for the full config "
        "dump.",
        flush=True,
    )
    if os.environ.get("CASCADIA_DEBUG_ARCH"):
        keys = sorted(vars(inner).keys()) if hasattr(inner, "__dict__") else []
        print(f"  config keys: {keys}", flush=True)
    return "llama"


def check_export_quirks(config, arch_tag: str) -> list[str]:
    """Inspect the (text-inner) config for features the exporter
    drops silently and return a list of human-readable warnings.

    The caller decides whether to: print and continue (default),
    print and abort (default unless CASCADIA_ALLOW_LOSSY_EXPORT=1),
    or silently allow.

    Detection covers:
    - MoE configs that slipped past detect_architecture (defensive)
    - partial_rotary_factor < 1.0 (Phi-4-mini etc.)
    - rope_scaling.type in {longrope, yarn, dynamic} (not implemented)
    - attn_logit_softcapping / final_logit_softcapping (Gemma 2)
    - layer_types with mixed sliding/full (Gemma 3+, Cohere2, gpt-oss)
    - asymmetric head_dim (global_head_dim != head_dim) — Gemma 4
    - sqrt(hidden) embed scaling families that need it
    - QK-Norm in non-Qwen3 paths (silently dropped)
    """
    cfg = _text_config(config)
    warnings: list[str] = []

    # MoE — last-line-of-defence catch.
    moe_fields = [
        ("num_local_experts", 0),
        ("num_experts", 0),
        ("num_routed_experts", 0),
        ("n_routed_experts", 0),
    ]
    for field, default in moe_fields:
        v = getattr(cfg, field, default)
        if isinstance(v, int) and v > 1:
            warnings.append(
                f"config.{field}={v} indicates MoE; the exporter will "
                "treat layer.mlp as a dense MLP. See "
                "docs/architectures/moe.md."
            )
            break

    # Partial rotary: handled in TracedRotaryEmbedding by padding
    # the back of inv_freq with zeros (so the trailing dims get cos=1
    # / sin=0 and pass through unchanged). We log it for transparency
    # but don't gate on it.

    # RoPE scaling — only "llama3" and default are honoured in runtime.
    rope_scaling = getattr(cfg, "rope_scaling", None) or {}
    rope_type = (
        rope_scaling.get("type") or rope_scaling.get("rope_type") or ""
    ).lower() if isinstance(rope_scaling, dict) else ""
    if rope_type and rope_type not in ("llama3", "default"):
        warnings.append(
            f"rope_scaling.type={rope_type!r} not supported in the Rust "
            "runtime (only 'llama3' and 'default' are honoured). Long-"
            "context outputs will degrade past "
            f"original_max_position_embeddings="
            f"{rope_scaling.get('original_max_position_embeddings', '?')}."
        )

    # rope_parameters dict form — Gemma 4 style (rejected) or per-layer.
    rope_params = getattr(cfg, "rope_parameters", None)
    if isinstance(rope_params, dict):
        types = set(rope_params.keys())
        # Per-layer-type rope_parameters is the Gemma 4 quirk — but
        # detect_architecture should have already rejected Gemma 4.
        if "full_attention" in types or "sliding_attention" in types:
            warnings.append(
                "rope_parameters is per-layer-type "
                f"(keys={sorted(types)}). The exporter applies one "
                "rotary across all layers in a stage; outputs on "
                "sliding-window layers will use the wrong base."
            )

    # Softcap.
    attn_cap = getattr(cfg, "attn_logit_softcapping", None)
    final_cap = getattr(cfg, "final_logit_softcapping", None)
    if attn_cap or final_cap:
        warnings.append(
            f"softcap detected (attn={attn_cap}, final={final_cap}) — "
            "the SDPA-based forward we use does not apply attn-softcap, "
            "and the head stage does not apply final-softcap. Top-k "
            "sampling will be too sharp."
        )

    # Per-layer-type sliding window list.
    layer_types = getattr(cfg, "layer_types", None)
    if isinstance(layer_types, list) and len(set(layer_types)) > 1:
        warnings.append(
            f"layer_types is mixed ({sorted(set(layer_types))}). The "
            "exporter treats every layer as full causal — "
            "sliding-window layers attend to too much context."
        )

    # Asymmetric head_dim — only Gemma 4 so far.
    global_hd = getattr(cfg, "global_head_dim", None)
    head_dim = getattr(cfg, "head_dim", None)
    if global_hd is not None and head_dim is not None and global_hd != head_dim:
        warnings.append(
            f"asymmetric head_dim (head_dim={head_dim}, "
            f"global_head_dim={global_hd}) — the exporter assumes a "
            "single head_dim per stage and will fail to load weights."
        )

    # Per-layer embeddings (Gemma 4 E2B/E4B).
    pli = getattr(cfg, "hidden_size_per_layer_input", 0) or 0
    if pli > 0:
        warnings.append(
            f"hidden_size_per_layer_input={pli} — model uses per-layer "
            "embeddings as a side channel; not wired through the "
            "exporter or transport."
        )

    # KV sharing across layers.
    kv_shared = getattr(cfg, "num_kv_shared_layers", 0) or 0
    if kv_shared > 0:
        warnings.append(
            f"num_kv_shared_layers={kv_shared} — model reuses KV "
            "cache across layers; the exporter allocates a fresh cache "
            "per layer."
        )

    # QK-norm on non-Qwen3 architectures (the only path that handles it).
    use_qk_norm = getattr(cfg, "use_qk_norm", False)
    if use_qk_norm and arch_tag != "qwen3":
        warnings.append(
            "config.use_qk_norm=True but the chosen arch_tag is "
            f"'{arch_tag}' (only the 'qwen3' path applies q_norm/k_norm "
            "before RoPE). Output will collapse to repetition on long "
            "prompts."
        )

    return warnings


def get_decoder_layer_cls(arch_tag: str):
    """Return the decoder layer class to instantiate per-layer."""
    if arch_tag == "llama":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        return LlamaDecoderLayer
    if arch_tag == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        return MistralDecoderLayer
    if arch_tag == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        return Qwen2DecoderLayer
    if arch_tag == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        return Qwen3DecoderLayer
    if arch_tag == "phi":
        # Phi-3 lives in transformers.models.phi3 in newer transformers
        try:
            from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

            return Phi3DecoderLayer
        except ImportError:
            from transformers.models.phi.modeling_phi import PhiDecoderLayer

            return PhiDecoderLayer
    if arch_tag == "gemma":
        # Prefer Gemma3 → Gemma2 → Gemma. Each layer class is
        # backward-compatible enough that loading a Gemma 1 checkpoint
        # via Gemma3DecoderLayer with strict=False produces a viable
        # subset, but the right class avoids missing-weight surprises.
        try:
            from transformers.models.gemma3.modeling_gemma3 import (
                Gemma3DecoderLayer,
            )

            return Gemma3DecoderLayer
        except ImportError:
            pass
        try:
            from transformers.models.gemma2.modeling_gemma2 import (
                Gemma2DecoderLayer,
            )

            return Gemma2DecoderLayer
        except ImportError:
            from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

            return GemmaDecoderLayer
    raise ValueError(f"unsupported arch tag: {arch_tag}")


def get_norm_cls(arch_tag: str):
    """Return the RMSNorm class for the architecture."""
    if arch_tag == "llama":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        return LlamaRMSNorm
    if arch_tag == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralRMSNorm

        return MistralRMSNorm
    if arch_tag == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        return Qwen2RMSNorm
    if arch_tag == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        return Qwen3RMSNorm
    if arch_tag == "gemma":
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

            return Gemma3RMSNorm
        except ImportError:
            pass
        try:
            from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

            return Gemma2RMSNorm
        except ImportError:
            from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

            return GemmaRMSNorm
    # Phi3 uses RMSNorm too
    if arch_tag == "phi":
        try:
            from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm

            return Phi3RMSNorm
        except ImportError:
            # Fall back to Llama's — same math
            from transformers.models.llama.modeling_llama import LlamaRMSNorm

            return LlamaRMSNorm
    # Default
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    return LlamaRMSNorm


# ---------------------------------------------------------------------------
# Stage planning
# ---------------------------------------------------------------------------


def compute_stage_plan(num_layers: int, num_stages: int, layer_split: str | None):
    """Build per-stage (layer_start, layer_end, has_embed, has_head) tuples.

    `layer_split` lets the caller override the uniform split with explicit
    boundaries: e.g. "16,24" means stage 0 = [0,16), stage 1 = [16,24),
    stage 2 = [24, num_layers).
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if num_stages > num_layers:
        raise ValueError(
            f"num_stages ({num_stages}) cannot exceed num_layers ({num_layers})"
        )

    if layer_split:
        bounds = [int(x.strip()) for x in layer_split.split(",") if x.strip()]
        if len(bounds) != num_stages - 1:
            raise ValueError(
                f"--layer-split has {len(bounds)} boundaries; needs {num_stages - 1}"
                f" for {num_stages} stages"
            )
        bounds = [0, *bounds, num_layers]
    else:
        base = num_layers // num_stages
        rem = num_layers % num_stages
        bounds = [0]
        cursor = 0
        for i in range(num_stages):
            cnt = base + (1 if i < rem else 0)
            cursor += cnt
            bounds.append(cursor)

    plan = []
    for i in range(num_stages):
        plan.append(
            {
                "stage": i,
                "layer_start": bounds[i],
                "layer_end": bounds[i + 1],
                "has_embed": (i == 0),
                "has_head": (i == num_stages - 1),
            }
        )
    return plan


# ---------------------------------------------------------------------------
# Internal rotary (computed from position_ids — canonical genai convention)
# ---------------------------------------------------------------------------


class TracedRotaryEmbedding(nn.Module):
    """RoPE from position_ids, traced into a graph OV's RoPEFusion can match.

    Supports partial rotary: when ``partial_rotary_factor < 1.0`` (Phi-3
    Mini 128k, Phi-4-mini, StableLM-2, Gemma 4 global), only the leading
    ``partial_rotary_factor * head_dim`` dims of each head are rotated;
    the trailing dims pass through. We model this by padding the back of
    ``inv_freq`` with zeros so the rotation angles for those dims are
    always 0, leaving cos=1 and sin=0 (i.e. multiplying the trailing
    dims by 1 and adding zero — a no-op).
    """

    def __init__(self, head_dim, rope_theta=500000.0, partial_rotary_factor=1.0):
        super().__init__()
        self.head_dim = head_dim
        self.partial_rotary_factor = float(partial_rotary_factor)
        if self.partial_rotary_factor >= 1.0:
            inv_freq = 1.0 / (
                rope_theta
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
        else:
            # Number of dim-pairs that get rotated.
            rot_pairs = int(self.partial_rotary_factor * head_dim) // 2
            if rot_pairs <= 0:
                raise ValueError(
                    f"partial_rotary_factor={self.partial_rotary_factor} "
                    f"with head_dim={head_dim} leaves 0 rotated dims"
                )
            # Important: the exponent for the rotated dims must use the
            # ORIGINAL head_dim as the denominator (matches HF's
            # _compute_proportional_rope_parameters in transformers ≥
            # 5.x). Using `2*rot_pairs` instead of `head_dim` would
            # change the per-dim frequencies and break parity with HF.
            inv_freq_rot = 1.0 / (
                rope_theta
                ** (
                    torch.arange(0, 2 * rot_pairs, 2, dtype=torch.float32)
                    / head_dim
                )
            )
            zero_pad = (head_dim // 2) - rot_pairs
            inv_freq = torch.cat(
                [inv_freq_rot, torch.zeros(zero_pad, dtype=torch.float32)], dim=0
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, target_dtype):
        bsz, seq_len = position_ids.shape
        inv_freq_expanded = self.inv_freq[None, None, :].expand(bsz, seq_len, -1)
        freqs = position_ids[:, :, None].float() * inv_freq_expanded
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(target_dtype)
        sin = emb.sin().to(target_dtype)
        return cos, sin


def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = q.shape[-1] // 2

    def rotate_half(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# SDPA-based decoder forward with KV-cached attention (model-agnostic)
# ---------------------------------------------------------------------------


def cached_layer_forward_sdpa(
    layer,
    hidden_states,
    cos,
    sin,
    causal_mask,
    past_key,
    past_value,
    num_heads,
    num_kv_heads,
    head_dim,
):
    """One decoder layer forward, attention via SDPA, KV-cached.

    Works for any decoder layer that exposes the conventional projection
    names (`self_attn.q_proj/k_proj/v_proj/o_proj`, `input_layernorm`,
    `post_attention_layernorm`, `mlp`).
    """
    bsz, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    q = layer.self_attn.q_proj(hidden_states)
    k = layer.self_attn.k_proj(hidden_states)
    v = layer.self_attn.v_proj(hidden_states)

    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Some architectures (e.g. Qwen3) apply RMSNorm to Q/K before RoPE.
    # RMSNorm operates on the last dim (head_dim), so post-transpose is equivalent.
    if hasattr(layer.self_attn, "q_norm"):
        q = layer.self_attn.q_norm(q)
    if hasattr(layer.self_attn, "k_norm"):
        k = layer.self_attn.k_norm(k)

    q, k = apply_rotary(q, k, cos, sin)

    # Append to cache.
    k = torch.cat([past_key, k], dim=2)
    v = torch.cat([past_value, v], dim=2)

    # Expand KV for GQA.
    k_exp = k[:, :, None, :, :].expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    k_exp = k_exp.reshape(bsz, num_heads, -1, head_dim)
    v_exp = v[:, :, None, :, :].expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    v_exp = v_exp.reshape(bsz, num_heads, -1, head_dim)

    attn_output = F.scaled_dot_product_attention(
        q,
        k_exp,
        v_exp,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(head_dim),
    )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output)

    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, k, v


# ---------------------------------------------------------------------------
# Stage wrappers
# ---------------------------------------------------------------------------


class _BaseStage(nn.Module):
    def __init__(
        self,
        layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor=1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary = TracedRotaryEmbedding(
            head_dim, rope_theta, partial_rotary_factor=partial_rotary_factor
        )

    def _build_causal_mask(self, attention_mask, seq_len, past_kv_len, dtype):
        """attention_mask: [bsz, past_kv_len + seq_len] with 1=allowed, 0=masked.
        Returns [bsz, 1, seq_len, past_kv_len + seq_len] additive (0/-inf)."""
        full_seq_len = past_kv_len + seq_len
        q_pos = torch.arange(
            seq_len, device=attention_mask.device
        ).unsqueeze(-1) + past_kv_len
        k_pos = torch.arange(full_seq_len, device=attention_mask.device).unsqueeze(0)
        causal_allow = (k_pos <= q_pos).to(dtype)
        pad_allow = attention_mask.unsqueeze(1).to(dtype)
        allow = causal_allow.unsqueeze(0) * pad_allow
        mask = (1.0 - allow) * torch.finfo(dtype).min
        return mask.unsqueeze(1)

    def _run_layers(self, hidden_states, attention_mask, position_ids, past_kv):
        cos, sin = self.rotary(position_ids, hidden_states.dtype)
        bsz, seq_len = position_ids.shape
        past_kv_len = past_kv[0].shape[2]
        causal_mask = self._build_causal_mask(
            attention_mask, seq_len, past_kv_len, hidden_states.dtype
        )
        present_kv = []
        for idx, layer in enumerate(self.layers):
            hidden_states, pk, pv = cached_layer_forward_sdpa(
                layer,
                hidden_states,
                cos,
                sin,
                causal_mask,
                past_kv[idx * 2],
                past_kv[idx * 2 + 1],
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            present_kv.extend([pk, pv])
        return hidden_states, present_kv


class CachedEmbedStageWrapper(_BaseStage):
    def __init__(
        self,
        embed_tokens,
        layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor=1.0,
    ):
        super().__init__(
            layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.embed_tokens = embed_tokens

    def forward(self, input_ids, attention_mask, position_ids, *past_kv):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states, present_kv = self._run_layers(
            hidden_states, attention_mask, position_ids, past_kv
        )
        return (hidden_states, *present_kv)


class CachedMiddleStageWrapper(_BaseStage):
    def __init__(
        self,
        layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor=1.0,
    ):
        super().__init__(
            layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )

    def forward(self, hidden_states, attention_mask, position_ids, *past_kv):
        hidden_states, present_kv = self._run_layers(
            hidden_states, attention_mask, position_ids, past_kv
        )
        return (hidden_states, *present_kv)


class CachedHeadStageWrapper(_BaseStage):
    def __init__(
        self,
        layers,
        norm,
        lm_head,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor=1.0,
    ):
        super().__init__(
            layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states, attention_mask, position_ids, *past_kv):
        hidden_states, present_kv = self._run_layers(
            hidden_states, attention_mask, position_ids, past_kv
        )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return (logits, *present_kv)


class CachedFullStageWrapper(_BaseStage):
    def __init__(
        self,
        embed_tokens,
        layers,
        norm,
        lm_head,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor=1.0,
    ):
        super().__init__(
            layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, *past_kv):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states, present_kv = self._run_layers(
            hidden_states, attention_mask, position_ids, past_kv
        )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return (logits, *present_kv)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_stage_weights(model_dir, layer_start, layer_end, has_embed, has_head, tied_embeddings):
    """Selectively load only the weights needed for this stage.

    If `tied_embeddings` is True and `has_head` is True (but not `has_embed`),
    we also pull `model.embed_tokens.weight` so the head stage can wire it
    into its lm_head. Wastes ~vocab*hidden bytes but avoids re-opening safetensors.
    """
    from safetensors import safe_open

    needed = []
    for i in range(layer_start, layer_end):
        needed.append(f"model.layers.{i}.")
    if has_embed:
        needed.append("model.embed_tokens.")
    if has_head:
        needed.append("model.norm.")
        needed.append("lm_head.")
        if tied_embeddings and not has_embed:
            needed.append("model.embed_tokens.")
    state_dict = {}
    for sf in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(key.startswith(p) for p in needed):
                    state_dict[key] = f.get_tensor(key)
    return state_dict


def build_wrapper(
    config,
    state_dict,
    layer_start,
    layer_end,
    has_embed,
    has_head,
    rope_theta,
    arch_tag,
    partial_rotary_factor=1.0,
):
    """Construct the appropriate stage wrapper for this rank."""
    DecoderLayer = get_decoder_layer_cls(arch_tag)
    NormCls = get_norm_cls(arch_tag)
    config._attn_implementation = "eager"

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    # Most LLMs have head_dim = hidden_size / num_heads, but Qwen3 (and some
    # others) decouple them — config.head_dim is the source of truth when set.
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)

    layers = []
    for i in range(layer_start, layer_end):
        layer = DecoderLayer(config, layer_idx=i)
        prefix = f"model.layers.{i}."
        layer_keys = [k for k in list(state_dict.keys()) if k.startswith(prefix)]
        layer_sd = {k.removeprefix(prefix): state_dict[k] for k in layer_keys}
        layer.load_state_dict(layer_sd, strict=False)
        layer.eval()
        for k in layer_keys:
            del state_dict[k]
        del layer_sd
        gc.collect()
        layers.append(layer)
        if (i - layer_start) % 4 == 3:
            print(
                f"    built layer {i} (+{(i - layer_start + 1)}/{layer_end - layer_start})",
                flush=True,
            )

    rms_eps = getattr(config, "rms_norm_eps", 1e-6)
    if has_embed and has_head:
        embed_w = state_dict["model.embed_tokens.weight"]
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        embed.load_state_dict({"weight": embed_w})
        del state_dict["model.embed_tokens.weight"]
        norm = NormCls(config.hidden_size, eps=rms_eps)
        norm.load_state_dict({"weight": state_dict["model.norm.weight"]})
        del state_dict["model.norm.weight"]
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if "lm_head.weight" in state_dict:
            lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]})
            del state_dict["lm_head.weight"]
        else:
            print(
                "  tied embeddings: reusing embed weight for lm_head", flush=True
            )
            lm_head.load_state_dict({"weight": embed_w})
        return CachedFullStageWrapper(
            embed,
            layers,
            norm,
            lm_head,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
    if has_embed:
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        embed.load_state_dict({"weight": state_dict["model.embed_tokens.weight"]})
        del state_dict["model.embed_tokens.weight"]
        return CachedEmbedStageWrapper(
            embed,
            layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
    if has_head:
        norm = NormCls(config.hidden_size, eps=rms_eps)
        norm.load_state_dict({"weight": state_dict["model.norm.weight"]})
        del state_dict["model.norm.weight"]
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if "lm_head.weight" in state_dict:
            lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]})
            del state_dict["lm_head.weight"]
        elif "model.embed_tokens.weight" in state_dict:
            # Tied embeddings + head-only stage: load_stage_weights pulled
            # the embed weight specifically for this case.
            print(
                "  tied embeddings + head-only stage: reusing embed for lm_head",
                flush=True,
            )
            lm_head.load_state_dict(
                {"weight": state_dict["model.embed_tokens.weight"]}
            )
            del state_dict["model.embed_tokens.weight"]
        else:
            raise RuntimeError(
                "head-only stage missing both lm_head.weight and embed_tokens.weight"
                " — tied_embeddings detection failed?"
            )
        return CachedHeadStageWrapper(
            layers,
            norm,
            lm_head,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )
    return CachedMiddleStageWrapper(
        layers,
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor=partial_rotary_factor,
    )


# ---------------------------------------------------------------------------
# Per-stage export
# ---------------------------------------------------------------------------


def export_single_stage(
    model_dir,
    output_dir,
    stage_plan,
    config,
    quantization,
    rope_theta,
    arch_tag,
    partial_rotary_factor=1.0,
):
    import openvino as ov

    stage_idx = stage_plan["stage"]
    layer_start = stage_plan["layer_start"]
    layer_end = stage_plan["layer_end"]
    has_embed = stage_plan["has_embed"]
    has_head = stage_plan["has_head"]
    num_layers = layer_end - layer_start
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)

    print(f"\n{'=' * 60}", flush=True)
    print(
        f"STAGE {stage_idx}: layers [{layer_start}, {layer_end})"
        f" | embed={has_embed} | head={has_head}",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)

    tied_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    # 1. Load weights selectively for this stage.
    print("  Loading weights...", flush=True)
    state_dict = load_stage_weights(
        model_dir, layer_start, layer_end, has_embed, has_head, tied_embeddings
    )
    weight_mb = sum(t.nbytes for t in state_dict.values()) / 1e6
    print(f"  {len(state_dict)} tensors ({weight_mb:.0f} MB)", flush=True)

    # 2. Build per-stage module wrapper.
    print("  Building wrapper...", flush=True)
    wrapper = build_wrapper(
        config,
        state_dict,
        layer_start,
        layer_end,
        has_embed,
        has_head,
        rope_theta,
        arch_tag,
        partial_rotary_factor=partial_rotary_factor,
    )
    del state_dict
    gc.collect()

    # 3. Example inputs (past_seq=1 to avoid zero-dim).
    seq_len = 4
    past_seq = 1
    full_seq_len = past_seq + seq_len
    if has_embed:
        main_input = torch.randint(0, config.vocab_size, (1, seq_len))
    else:
        main_input = torch.randn(1, seq_len, config.hidden_size)
    attention_mask = torch.ones(1, full_seq_len, dtype=torch.long)
    position_ids = torch.arange(
        past_seq, past_seq + seq_len, dtype=torch.long
    ).unsqueeze(0)
    past_kv_tensors = []
    for _ in range(num_layers):
        past_kv_tensors.append(torch.randn(1, num_kv_heads, past_seq, head_dim))
        past_kv_tensors.append(torch.randn(1, num_kv_heads, past_seq, head_dim))
    example_inputs = (main_input, attention_mask, position_ids, *past_kv_tensors)

    # 4. Trace + convert to OpenVINO.
    print("  torch.jit.trace...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs)
    print("  Trace OK", flush=True)
    del wrapper
    gc.collect()

    print("  ov.convert_model...", flush=True)
    ov_model = ov.convert_model(traced, example_input=example_inputs)
    del traced
    gc.collect()

    # 5. Name inputs/outputs and set dynamic shapes.
    for i, inp in enumerate(ov_model.inputs):
        shape = inp.partial_shape
        if i == 0:
            name = "input_ids" if has_embed else "hidden_states"
            if len(shape) >= 2:
                shape[1] = -1
        elif i == 1:
            name = "attention_mask"
            if len(shape) >= 2:
                shape[1] = -1
        elif i == 2:
            name = "position_ids"
            if len(shape) >= 2:
                shape[1] = -1
        else:
            kv_idx = i - 3
            layer_local = kv_idx // 2
            is_value = kv_idx % 2 == 1
            kv_type = "value" if is_value else "key"
            name = f"past_key_values.{layer_local}.{kv_type}"
            if len(shape) >= 3:
                shape[2] = -1
        inp.node.set_partial_shape(shape)
        inp.set_names({name})

    for i, out in enumerate(ov_model.outputs):
        if i == 0:
            name = "logits" if has_head else "hidden_states"
        else:
            kv_idx = i - 1
            layer_local = kv_idx // 2
            is_value = kv_idx % 2 == 1
            kv_type = "value" if is_value else "key"
            name = f"present.{layer_local}.{kv_type}"
        out.set_names({name})

    # 6. Make stateful (KV cache via ReadValue/Assign instead of input/output).
    print(f"  apply_make_stateful_transformation ({num_layers} KV pairs)...", flush=True)
    pairs = []
    for layer_local in range(num_layers):
        pairs.append(
            (
                f"past_key_values.{layer_local}.key",
                f"present.{layer_local}.key",
            )
        )
        pairs.append(
            (
                f"past_key_values.{layer_local}.value",
                f"present.{layer_local}.value",
            )
        )
    pair_map = dict(pairs)

    from openvino._offline_transformations import apply_make_stateful_transformation

    apply_make_stateful_transformation(ov_model, pair_map)

    # 7. fuse_cache_reorder: add beam_idx + Gather on each ReadValue.
    # This is what optimum-intel does — required for OV GPU's IndirectKVCache.
    try:
        import openvino.opset13 as opset
        from openvino import PartialShape, Type

        beam_idx_param = opset.parameter(PartialShape([-1]), Type.i32, name="beam_idx")
        beam_idx_param.set_friendly_name("beam_idx")
        beam_idx_param.output(0).set_names({"beam_idx"})
        read_values = [n for n in ov_model.get_ops() if n.get_type_name() == "ReadValue"]
        axis_const = opset.constant(0, Type.i32)
        for rv in read_values:
            rv_out = rv.output(0)
            gather_node = opset.gather(rv_out, beam_idx_param, axis_const)
            gather_out = gather_node.output(0)
            for target_input in list(rv_out.get_target_inputs()):
                if target_input.get_node() is gather_node:
                    continue
                target_input.replace_source_output(gather_out)
        ov_model.add_parameters([beam_idx_param])
        ov_model.validate_nodes_and_infer_types()
        print(
            f"  fuse_cache_reorder: beam_idx + Gather on {len(read_values)} ReadValue ops",
            flush=True,
        )
    except Exception as e:
        print(f"  fuse_cache_reorder FAILED: {e}", flush=True)
        import traceback

        traceback.print_exc()

    # 8. Compress weights.
    if quantization in ("int4", "int4_asym"):
        try:
            import nncf

            mode = (
                nncf.CompressWeightsMode.INT4_SYM
                if quantization == "int4"
                else nncf.CompressWeightsMode.INT4_ASYM
            )
            print(f"  nncf {quantization} compression...", flush=True)
            ov_model = nncf.compress_weights(
                ov_model, mode=mode, group_size=128, ratio=1.0, all_layers=True
            )
            print("  Quantization OK", flush=True)
        except Exception as e:
            print(
                f"  WARNING: INT4 compression failed ({e}); falling back to FP16",
                flush=True,
            )
    elif quantization == "int8":
        try:
            import nncf

            print("  nncf int8 compression...", flush=True)
            ov_model = nncf.compress_weights(
                ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM
            )
        except Exception as e:
            print(
                f"  WARNING: INT8 compression failed ({e}); falling back to FP16",
                flush=True,
            )

    # 9. Save.
    stage_dir = os.path.join(output_dir, f"stage_{stage_idx}")
    os.makedirs(stage_dir, exist_ok=True)
    xml_path = os.path.join(stage_dir, "openvino_model.xml")
    print("  Saving model...", flush=True)
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    bin_size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / 1e6
    print(f"  Saved: {stage_dir} ({bin_size_mb:.0f} MB)", flush=True)

    # 10. Per-stage metadata.
    meta = {
        "stage": stage_idx,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "has_embed": has_embed,
        "has_head": has_head,
        "quantization": quantization,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "num_layers_total": config.num_hidden_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "stateful": True,
        "rope_theta": rope_theta,
        "partial_rotary_factor": partial_rotary_factor,
        "arch_tag": arch_tag,
        "inputs": (
            "input_ids/hidden_states, attention_mask, position_ids, beam_idx "
            "(KV cache is internal state)"
        ),
        "export_version": "v5_canonical_inputs",
    }
    with open(os.path.join(stage_dir, "stage_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return bin_size_mb


# ---------------------------------------------------------------------------
# Top-level export pipeline
# ---------------------------------------------------------------------------


def fetch_config_json(model_id_or_path: str) -> str:
    """Return a local path to config.json for ``model_id_or_path``.

    For a local directory, just point at the file. For an HF repo id,
    download only ``config.json`` (a few KB) without fetching tokenizer
    or safetensors. Used by the pre-flight architecture check so a
    multi-GB model isn't pulled before detect_architecture can reject
    it.
    """
    if os.path.isdir(model_id_or_path):
        path = os.path.join(model_id_or_path, "config.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{model_id_or_path} is a directory but has no config.json"
            )
        return path
    from huggingface_hub import hf_hub_download

    cache_root = os.path.expanduser("~/.cache/cascadia/models")
    safe_id = model_id_or_path.replace("/", "--")
    local_dir = os.path.join(cache_root, safe_id)
    os.makedirs(local_dir, exist_ok=True)
    print(f"Pre-fetching config.json for {model_id_or_path}", flush=True)
    return hf_hub_download(
        repo_id=model_id_or_path,
        filename="config.json",
        local_dir=local_dir,
    )


def maybe_download(model_id_or_path: str) -> str:
    """If `model_id_or_path` is an existing local directory, return it.
    Otherwise treat as an HF repo id and snapshot_download. Returns the
    local model directory containing config.json + safetensors + tokenizer.
    """
    if os.path.isdir(model_id_or_path):
        # Local path: ensure config.json present.
        if not os.path.exists(os.path.join(model_id_or_path, "config.json")):
            raise FileNotFoundError(
                f"{model_id_or_path} is a directory but has no config.json"
            )
        return model_id_or_path
    # HF id — download.
    from huggingface_hub import snapshot_download

    cache_root = os.path.expanduser("~/.cache/cascadia/models")
    safe_id = model_id_or_path.replace("/", "--")
    local_dir = os.path.join(cache_root, safe_id)
    print(f"Downloading {model_id_or_path} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=model_id_or_path,
        local_dir=local_dir,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer.*",
            "*.model",
            "special_tokens_map.json",
        ],
        max_workers=8,
    )
    return local_dir


def copy_tokenizer(model_dir: str, output_dir: str) -> None:
    """Copy tokenizer files into the shard's tokenizer/ subdir so the
    runtime engines can find them without falling back to HF cache."""
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    wanted = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "config.json",
        "generation_config.json",
        "added_tokens.json",
    )
    copied = 0
    for fname in wanted:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tok_dir, fname))
            copied += 1
    print(f"Copied {copied} tokenizer files to {tok_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export an HF causal-LM model as cascadia per-stage shards"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id (e.g. unsloth/Meta-Llama-3.1-8B-Instruct) "
        "or path to a local directory with safetensors + config.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output shard directory (will be created)",
    )
    parser.add_argument(
        "--num-stages", type=int, required=True, help="Pipeline stages to split into"
    )
    parser.add_argument(
        "--quantization",
        default="int4",
        choices=["fp16", "int4", "int4_asym", "int8"],
        help="Weight precision (default: int4)",
    )
    parser.add_argument(
        "--layer-split",
        default=None,
        help='Explicit layer boundaries, comma-separated (e.g. "16,24" for'
        " 3-stage 32-layer split). If unset, layers split uniformly.",
    )
    parser.add_argument(
        "--default-dtype",
        default="fp16",
        choices=["fp16", "fp32"],
        help="torch default dtype during the export (fp16 reduces memory)",
    )
    parser.add_argument(
        "--stage", type=int, default=None, help="Export only this stage index (debug)"
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=None,
        help=(
            "Override rope_theta (default: read from config.rope_theta, "
            "falling back to a per-family baseline). The exporter bakes "
            "this into a TracedRotaryEmbedding inv_freq buffer; if the "
            "config is missing or wrong, output is garbage. Use this to "
            "patch missing-theta configs without editing them."
        ),
    )
    parser.add_argument(
        "--partial-rotary-factor",
        type=float,
        default=None,
        help=(
            "Override partial_rotary_factor (default: read from "
            "config.partial_rotary_factor, falling back to 1.0). Phi-4-"
            "mini is 0.75; only the leading 75% of each head's RoPE "
            "dims should be rotated. Wrong value silently produces "
            "garbage output."
        ),
    )
    args = parser.parse_args()

    if args.default_dtype == "fp16":
        torch.set_default_dtype(torch.float16)

    from transformers import AutoConfig

    # Pre-flight: pull ONLY config.json (a few KB) and run our
    # architecture-detection logic against it before committing to the
    # multi-GB snapshot download. Without this, both the snapshot
    # download AND transformers' AutoConfig.from_pretrained would run
    # for several minutes against models we're about to reject anyway.
    cfg_json_path = fetch_config_json(args.model)
    cfg_json = {}
    try:
        with open(cfg_json_path) as f:
            cfg_json = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  warning: could not read config.json: {e}", flush=True)
    if cfg_json:
        raw_ns = _RawConfigNS(cfg_json)
        try:
            detect_architecture(raw_ns)
        except UnsupportedModelError as exc:
            _print_unsupported(exc)
            if not os.environ.get("CASCADIA_ALLOW_LOSSY_EXPORT"):
                sys.exit(2)
            print(
                "  CASCADIA_ALLOW_LOSSY_EXPORT=1 set — attempting load "
                "anyway. transformers may still reject.",
                flush=True,
            )

    # Architecture passed the pre-flight — fetch the full snapshot.
    model_dir = maybe_download(args.model)

    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    except ValueError as exc:
        # Often transformers' "does not recognize this architecture" for
        # very-new model types. Try trust_remote_code=True as a last
        # resort (model repos like google/gemma-4-* often ship their
        # own modeling code). If that also fails, give up.
        msg = str(exc)
        if "does not recognize this architecture" in msg:
            print(
                "  AutoConfig.from_pretrained failed without "
                "trust_remote_code; retrying with trust_remote_code=True.",
                flush=True,
            )
            try:
                config = AutoConfig.from_pretrained(
                    model_dir, trust_remote_code=True
                )
            except Exception:
                print(
                    "  Still failed. Upgrade transformers or pin a "
                    "compatible version.",
                    flush=True,
                )
                raise
        else:
            raise
    # For multimodal wrappers (Gemma 3/4 ConditionalGeneration,
    # Llama4ForConditionalGeneration, Mistral3, etc.), the text-tower
    # config nests under config.text_config. Most downstream code wants
    # the text-tower fields, not the wrapper's empty ones.
    text_cfg = _text_config(config)

    try:
        arch_tag = detect_architecture(config)
    except UnsupportedModelError as exc:
        _print_unsupported(exc)
        if not os.environ.get("CASCADIA_ALLOW_LOSSY_EXPORT"):
            sys.exit(2)
        print(
            "  CASCADIA_ALLOW_LOSSY_EXPORT=1 set — proceeding via Llama "
            "fallback.",
            flush=True,
        )
        arch_tag = "llama"

    # Run pre-export sanity checks that catch quirks the architecture
    # tag alone misses (e.g. partial_rotary_factor on phi3, MoE in a
    # newly-released variant, mixed layer_types).
    quirks = check_export_quirks(config, arch_tag)
    if quirks:
        print("", flush=True)
        print("=" * 60, flush=True)
        print("EXPORT WILL DROP THE FOLLOWING FEATURES:", flush=True)
        print("=" * 60, flush=True)
        for w in quirks:
            print(f"  - {w}", flush=True)
        if not os.environ.get("CASCADIA_ALLOW_LOSSY_EXPORT"):
            print("", flush=True)
            print(
                "Set CASCADIA_ALLOW_LOSSY_EXPORT=1 to continue anyway. "
                "Output WILL diverge from the HF reference in ways "
                "predicted above. See docs/SHARDING.md.",
                flush=True,
            )
            sys.exit(2)
        print(
            "  CASCADIA_ALLOW_LOSSY_EXPORT=1 set — proceeding.",
            flush=True,
        )

    # transformers 4.x exposes rope_theta directly; 5.x moved it under
    # config.rope_parameters = {"rope_theta": ..., "rope_type": ...}.
    # Handle both — wrong rope_theta gets silently baked into the traced
    # rotary inv_freq buffer and produces garbage output at inference.
    rope_theta_raw = getattr(text_cfg, "rope_theta", None)
    if rope_theta_raw is None:
        rope_params = getattr(text_cfg, "rope_parameters", None) or {}
        if isinstance(rope_params, dict):
            # Some new families use a per-layer-type dict here; reject was
            # done earlier, this branch is the simple-dict case.
            rope_theta_raw = rope_params.get("rope_theta")
    rope_theta_defaulted = rope_theta_raw is None
    if rope_theta_raw is None:
        # The 500k default was Llama-3's choice. Llama-2 and below used
        # 10000.0; Qwen2 uses 1e6; Phi-3 uses 10000.0. Wrong baseline
        # produces garbage. Pick the per-family default if we have one.
        if arch_tag in ("llama",):
            rope_theta_raw = 500_000.0
        elif arch_tag in ("qwen2", "qwen3"):
            rope_theta_raw = 1_000_000.0
        else:
            rope_theta_raw = 10_000.0
    rope_theta = float(rope_theta_raw)
    if rope_theta_defaulted:
        print(
            f"  warning: config.rope_theta not set; defaulted to "
            f"{rope_theta} based on arch_tag={arch_tag!r}. Override with "
            "--rope-theta if this is wrong (symptom: garbage output).",
            flush=True,
        )

    if args.rope_theta is not None:
        print(
            f"  rope_theta override: {rope_theta} -> {args.rope_theta}",
            flush=True,
        )
        rope_theta = float(args.rope_theta)

    # Partial rotary factor (Phi-3 Mini 128k, Phi-4-mini, StableLM-2).
    # Default 1.0 (rotate all head_dim positions). Values < 1.0 mean
    # only the leading fraction of dims is rotated.
    partial_rotary_factor = float(
        getattr(text_cfg, "partial_rotary_factor", 1.0) or 1.0
    )
    if args.partial_rotary_factor is not None:
        print(
            f"  partial_rotary_factor override: {partial_rotary_factor} "
            f"-> {args.partial_rotary_factor}",
            flush=True,
        )
        partial_rotary_factor = float(args.partial_rotary_factor)
    if partial_rotary_factor != 1.0:
        print(
            f"  partial_rotary_factor={partial_rotary_factor}: "
            f"rotating the leading {partial_rotary_factor*100:.0f}% of "
            "each head's dims; trailing dims pass through.",
            flush=True,
        )

    print(
        f"\nModel: {text_cfg.num_hidden_layers} layers,"
        f" hidden={text_cfg.hidden_size},"
        f" kv_heads={getattr(text_cfg, 'num_key_value_heads', text_cfg.num_attention_heads)},"
        f" rope_theta={rope_theta},"
        f" arch={arch_tag},"
        f" partial_rotary_factor={partial_rotary_factor}",
        flush=True,
    )

    plan = compute_stage_plan(
        text_cfg.num_hidden_layers, args.num_stages, args.layer_split
    )

    print(f"\nStage plan ({args.num_stages} stages):", flush=True)
    for s in plan:
        parts = []
        if s["has_embed"]:
            parts.append("embed")
        parts.append(f"layers {s['layer_start']}-{s['layer_end'] - 1}")
        if s["has_head"]:
            parts.append("norm+head")
        print(f"  Stage {s['stage']}: {' + '.join(parts)}", flush=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    copy_tokenizer(model_dir, output_dir)

    pipeline_meta = {
        "model_id": args.model,
        "num_stages": args.num_stages,
        "num_layers": text_cfg.num_hidden_layers,
        "hidden_size": text_cfg.hidden_size,
        "num_attention_heads": text_cfg.num_attention_heads,
        "num_key_value_heads": getattr(
            text_cfg, "num_key_value_heads", text_cfg.num_attention_heads
        ),
        "vocab_size": text_cfg.vocab_size,
        "rope_theta": rope_theta,
        "partial_rotary_factor": partial_rotary_factor,
        "arch_tag": arch_tag,
        "quantization": args.quantization,
        "export_version": "v5_canonical_inputs",
    }
    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump(pipeline_meta, f, indent=2)

    total_mb = 0.0
    for s in plan:
        if args.stage is not None and s["stage"] != args.stage:
            continue
        size = export_single_stage(
            model_dir,
            output_dir,
            s,
            text_cfg,
            args.quantization,
            rope_theta,
            arch_tag,
            partial_rotary_factor=partial_rotary_factor,
        )
        total_mb += size
        gc.collect()

    print("\n" + "=" * 60, flush=True)
    print("EXPORT COMPLETE", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Total: {total_mb:.0f} MB", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
