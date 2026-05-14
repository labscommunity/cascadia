#!/usr/bin/env python3
"""tahoma standalone model exporter.

Takes a HuggingFace model (Llama, Mistral, Qwen2, or any decoder-only LLM
that follows the same conventions) and produces a tahoma shard directory:

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

This script is invoked by `tahoma shard`. It is also runnable standalone
for users who want fine-grained control:

    python export_shards.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --output-dir ~/tahoma-shards/llama-8b-2stage \
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
# Architecture detection
# ---------------------------------------------------------------------------


def detect_architecture(config) -> str:
    """Return a short tag identifying the model family. Used to pick the
    right decoder-layer class. Falls back to 'llama' for unknown models
    (most modern decoder-only LLMs have a Llama-compatible layer shape).
    """
    model_type = getattr(config, "model_type", "").lower()
    arch_list = getattr(config, "architectures", []) or []
    arch_first = (arch_list[0] if arch_list else "").lower()

    if "llama" in model_type or "llama" in arch_first:
        return "llama"
    if "mistral" in model_type or "mistral" in arch_first:
        return "mistral"
    if "qwen3" in model_type or "qwen3" in arch_first:
        return "qwen3"
    if "qwen2" in model_type or "qwen2" in arch_first:
        return "qwen2"
    if "phi" in model_type or "phi" in arch_first:
        return "phi"
    if "gemma" in model_type or "gemma" in arch_first:
        return "gemma"
    print(
        f"  warning: unknown model_type={model_type!r}, architectures={arch_list!r};"
        " falling back to Llama decoder layer (may break for non-Llama-compat models)",
        flush=True,
    )
    return "llama"


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
        try:
            from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

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
    """RoPE from position_ids, traced into a graph OV's RoPEFusion can match."""

    def __init__(self, head_dim, rope_theta=500000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
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
    def __init__(self, layers, num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary = TracedRotaryEmbedding(head_dim, rope_theta)

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
        self, embed_tokens, layers, num_heads, num_kv_heads, head_dim, rope_theta
    ):
        super().__init__(layers, num_heads, num_kv_heads, head_dim, rope_theta)
        self.embed_tokens = embed_tokens

    def forward(self, input_ids, attention_mask, position_ids, *past_kv):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states, present_kv = self._run_layers(
            hidden_states, attention_mask, position_ids, past_kv
        )
        return (hidden_states, *present_kv)


class CachedMiddleStageWrapper(_BaseStage):
    def forward(self, hidden_states, attention_mask, position_ids, *past_kv):
        hidden_states, present_kv = self._run_layers(
            hidden_states, attention_mask, position_ids, past_kv
        )
        return (hidden_states, *present_kv)


class CachedHeadStageWrapper(_BaseStage):
    def __init__(
        self, layers, norm, lm_head, num_heads, num_kv_heads, head_dim, rope_theta
    ):
        super().__init__(layers, num_heads, num_kv_heads, head_dim, rope_theta)
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
    ):
        super().__init__(layers, num_heads, num_kv_heads, head_dim, rope_theta)
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
        )
    if has_embed:
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        embed.load_state_dict({"weight": state_dict["model.embed_tokens.weight"]})
        del state_dict["model.embed_tokens.weight"]
        return CachedEmbedStageWrapper(
            embed, layers, num_heads, num_kv_heads, head_dim, rope_theta
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
            layers, norm, lm_head, num_heads, num_kv_heads, head_dim, rope_theta
        )
    return CachedMiddleStageWrapper(
        layers, num_heads, num_kv_heads, head_dim, rope_theta
    )


# ---------------------------------------------------------------------------
# Per-stage export
# ---------------------------------------------------------------------------


def export_single_stage(
    model_dir, output_dir, stage_plan, config, quantization, rope_theta, arch_tag
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

    cache_root = os.path.expanduser("~/.cache/tahoma/models")
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
        description="Export an HF causal-LM model as tahoma per-stage shards"
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
    args = parser.parse_args()

    if args.default_dtype == "fp16":
        torch.set_default_dtype(torch.float16)

    from transformers import AutoConfig

    model_dir = maybe_download(args.model)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    arch_tag = detect_architecture(config)
    # transformers 4.x exposes rope_theta directly; 5.x moved it under
    # config.rope_parameters = {"rope_theta": ..., "rope_type": ...}.
    # Handle both — wrong rope_theta gets silently baked into the traced
    # rotary inv_freq buffer and produces garbage output at inference.
    rope_theta_raw = getattr(config, "rope_theta", None)
    if rope_theta_raw is None:
        rope_params = getattr(config, "rope_parameters", None) or {}
        rope_theta_raw = rope_params.get("rope_theta")
    if rope_theta_raw is None:
        rope_theta_raw = 500000.0
    rope_theta = float(rope_theta_raw)

    print(
        f"\nModel: {config.num_hidden_layers} layers,"
        f" hidden={config.hidden_size},"
        f" kv_heads={getattr(config, 'num_key_value_heads', config.num_attention_heads)},"
        f" rope_theta={rope_theta},"
        f" arch={arch_tag}",
        flush=True,
    )

    plan = compute_stage_plan(
        config.num_hidden_layers, args.num_stages, args.layer_split
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
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(
            config, "num_key_value_heads", config.num_attention_heads
        ),
        "vocab_size": config.vocab_size,
        "rope_theta": rope_theta,
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
            config,
            args.quantization,
            rope_theta,
            arch_tag,
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
