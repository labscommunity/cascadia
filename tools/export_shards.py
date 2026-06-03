#!/usr/bin/env python3
"""cascadia standalone model exporter.

Takes a HuggingFace dense decoder-only model (Llama, Mistral, Qwen2/3, Phi-3,
Gemma 1/2 — see the architecture-support note below; mixture-of-experts models
are detected and rejected) and produces a cascadia shard directory:

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
attribute names (`self_attn.{q_proj,k_proj,v_proj,o_proj}` or a fused
`qkv_proj`, `mlp`, `input_layernorm`, `post_attention_layernorm`) and uses
RoPE rotary will work. This covers:
  - Llama 1/2/3/3.1/3.2 family (+ Llama-arch models like deepseek-llm)
  - Mistral 7B
  - Qwen2 / Qwen2.5 / Qwen3
  - Phi-3 (fused qkv_proj — #58)
  - Gemma 1 (scaled embeddings) and Gemma-2 (+ attention/final-logit
    softcapping + the 4-norm decoder structure — #61)

Weights load from *.safetensors or, as a fallback, legacy pytorch_model*.bin
(e.g. deepseek-llm — #59).

Mixture-of-experts models (Mixtral, DeepSeek-V2/V3, Qwen*-MoE, Llama-4, …)
are detected and rejected up front (#60): the exporter builds dense decoder
layers and does not implement MoE routing / per-expert MLPs. Other
non-standard architectures (custom sliding-window kernels, multimodal) may
export but won't match HF reference outputs; benchmark before trusting.
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
    # Distinguish Gemma generations — they have different decoder structures
    # (Gemma-1 2-norm, Gemma-2 4-norm + softcapping, Gemma-3 different again).
    # Check the more specific tags first since each name contains the prior.
    if "gemma3" in model_type or "gemma3" in arch_first:
        return "gemma3"
    if "gemma2" in model_type or "gemma2" in arch_first:
        return "gemma2"
    if "gemma" in model_type or "gemma" in arch_first:
        return "gemma"
    print(
        f"  warning: unknown model_type={model_type!r}, architectures={arch_list!r};"
        " falling back to Llama decoder layer (may break for non-Llama-compat models)",
        flush=True,
    )
    return "llama"


def is_moe_config(config) -> bool:
    """True if `config` describes a mixture-of-experts model.

    cascadia's exporter builds dense decoder layers (one MLP per layer). MoE
    models route each token through a subset of many expert MLPs per layer,
    which this exporter does not implement. Detecting them lets the caller
    reject early with a clear message instead of silently falling back to a
    dense Llama layer and emitting garbage (#60).
    """
    # 1. Top-level expert-count fields — definitive (dense models lack these).
    for field in ("num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"):
        v = getattr(config, field, None)
        if isinstance(v, int) and v > 1:
            return True
    # 2. Nested expert config — DBRX puts the count under config.ffn_config.
    ffn = getattr(config, "ffn_config", None)
    if ffn is not None:
        n = ffn.get("moe_num_experts") if isinstance(ffn, dict) else getattr(ffn, "moe_num_experts", None)
        if isinstance(n, int) and n > 1:
            return True
    # 3. A per-token expert router count is a dense-model-free MoE signal.
    if isinstance(getattr(config, "num_experts_per_tok", None), int):
        return True
    # 4. Known MoE model_types (exact) + architecture class names. Exact
    #    model_type avoids false-positives from a dense model merely named
    #    "*moe*"; class names ending in MoEForCausalLM are reliably MoE.
    model_type = getattr(config, "model_type", "").lower()
    arch_first = ((getattr(config, "architectures", []) or [""])[0]).lower()
    moe_types = {
        "mixtral", "dbrx", "deepseek_v2", "deepseek_v3", "qwen2_moe", "qwen3_moe",
        "phimoe", "jamba", "granitemoe", "olmoe", "grok", "grok-1", "llama4",
    }
    if model_type in moe_types:
        return True
    return arch_first.endswith("moeforcausallm") or "mixtral" in arch_first or "grok" in arch_first


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
    if arch_tag == "gemma2":
        from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

        return Gemma2DecoderLayer
    if arch_tag == "gemma":
        # Gemma-1 (2-norm). Must NOT use Gemma2DecoderLayer — its attention
        # reads config.query_pre_attn_scalar, which GemmaConfig lacks.
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
    if arch_tag == "gemma2":
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

        return Gemma2RMSNorm
    if arch_tag == "gemma":
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
    attn_softcap=None,
    query_scale=None,
    gemma2_norms=False,
):
    """One decoder layer forward, attention via SDPA, KV-cached.

    Works for any decoder layer that exposes the conventional projection
    names (`self_attn.q_proj/k_proj/v_proj/o_proj` or a fused `qkv_proj`,
    `input_layernorm`, `post_attention_layernorm`, `mlp`).

    Gemma-2 needs three deviations (#61), gated by the optional args:
      - `query_scale`: attention scale = 1/sqrt(query_pre_attn_scalar) instead
        of 1/sqrt(head_dim);
      - `attn_softcap`: tanh-softcap the attention logits (SDPA can't, so we do
        attention manually when set);
      - `gemma2_norms`: the 4-norm residual structure (post_attention_layernorm
        AFTER attn + pre/post_feedforward_layernorm around the MLP).
    """
    bsz, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    if hasattr(layer.self_attn, "qkv_proj"):
        # Fused QKV projection (e.g. Phi-3, #58): one matmul, split by size into
        # q (num_heads), k/v (num_kv_heads), each * head_dim.
        qkv = layer.self_attn.qkv_proj(hidden_states)
        q_sz = num_heads * head_dim
        kv_sz = num_kv_heads * head_dim
        if qkv.shape[-1] != q_sz + 2 * kv_sz:
            raise ValueError(
                f"fused qkv_proj width {qkv.shape[-1]} != q({q_sz})+k({kv_sz})+v({kv_sz}); "
                f"this arch packs qkv differently than the assumed [Q,K,V] contiguous layout"
            )
        q = qkv[..., :q_sz]
        k = qkv[..., q_sz : q_sz + kv_sz]
        v = qkv[..., q_sz + kv_sz : q_sz + 2 * kv_sz]
    else:
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

    scale = query_scale if query_scale is not None else (1.0 / math.sqrt(head_dim))
    if attn_softcap is not None:
        # SDPA cannot tanh-softcap the logits, so do attention by hand
        # (Gemma-2): scores = softcap * tanh((q·kᵀ)·scale / softcap) + mask.
        scores = torch.matmul(q, k_exp.transpose(2, 3)) * scale
        scores = attn_softcap * torch.tanh(scores / attn_softcap)
        scores = scores + causal_mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(probs, v_exp)
    else:
        attn_output = F.scaled_dot_product_attention(
            q,
            k_exp,
            v_exp,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output)

    if gemma2_norms:
        # Gemma-2: norm AFTER attn (pre-residual) + pre/post norms around MLP.
        attn_output = layer.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
    else:
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    return hidden_states, k, v


class ArchSpec:
    """Per-architecture export deviations. Defaults are the Llama-family
    behaviour; Gemma needs scaled embeddings, and Gemma-2 additionally needs
    attention/final-logit softcapping, a query pre-attn scalar, and the 4-norm
    decoder structure (#61)."""

    def __init__(
        self,
        embed_scale=1.0,
        attn_softcap=None,
        final_softcap=None,
        query_scale=None,
        gemma2_norms=False,
    ):
        self.embed_scale = embed_scale
        self.attn_softcap = attn_softcap
        self.final_softcap = final_softcap
        self.query_scale = query_scale  # None → 1/sqrt(head_dim) in the layer fn
        self.gemma2_norms = gemma2_norms


def arch_spec_from_config(config, arch_tag, head_dim) -> "ArchSpec":
    """Build the ArchSpec for a model from its HF config (keyed off the
    architecture tag from detect_architecture)."""
    if arch_tag == "gemma2":
        # Scaled embeddings + attention/final-logit softcapping + query
        # pre-attn scalar + the 4-norm decoder structure.
        qpas = getattr(config, "query_pre_attn_scalar", None) or head_dim
        return ArchSpec(
            embed_scale=float(config.hidden_size) ** 0.5,
            attn_softcap=getattr(config, "attn_logit_softcapping", None),
            final_softcap=getattr(config, "final_logit_softcapping", None),
            query_scale=1.0 / math.sqrt(qpas),
            gemma2_norms=True,
        )
    if arch_tag == "gemma":
        # Gemma-1: scaled embeddings only (standard 2-norm layer, no softcap).
        return ArchSpec(embed_scale=float(config.hidden_size) ** 0.5)
    return ArchSpec()


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
        # Per-architecture deviations; build_wrapper overrides for Gemma(-2).
        self.arch = ArchSpec()

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
                attn_softcap=self.arch.attn_softcap,
                query_scale=self.arch.query_scale,
                gemma2_norms=self.arch.gemma2_norms,
            )
            present_kv.extend([pk, pv])
        return hidden_states, present_kv

    def _scale_embeds(self, hidden_states):
        """Gemma scales token embeddings by sqrt(hidden_size) (#61); no-op
        otherwise. Single source of truth so embed/full stages can't drift."""
        if self.arch.embed_scale != 1.0:
            return hidden_states * self.arch.embed_scale
        return hidden_states

    def _softcap_logits(self, logits):
        """Gemma-2 tanh-softcaps the final logits (#61); no-op otherwise."""
        if self.arch.final_softcap is not None:
            return self.arch.final_softcap * torch.tanh(logits / self.arch.final_softcap)
        return logits


class CachedEmbedStageWrapper(_BaseStage):
    def __init__(
        self, embed_tokens, layers, num_heads, num_kv_heads, head_dim, rope_theta
    ):
        super().__init__(layers, num_heads, num_kv_heads, head_dim, rope_theta)
        self.embed_tokens = embed_tokens

    def forward(self, input_ids, attention_mask, position_ids, *past_kv):
        hidden_states = self._scale_embeds(self.embed_tokens(input_ids))
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
        logits = self._softcap_logits(self.lm_head(hidden_states))
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
        hidden_states = self._scale_embeds(self.embed_tokens(input_ids))
        hidden_states, present_kv = self._run_layers(
            hidden_states, attention_mask, position_ids, past_kv
        )
        hidden_states = self.norm(hidden_states)
        logits = self._softcap_logits(self.lm_head(hidden_states))
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

    def wanted(key):
        return any(key.startswith(p) for p in needed)

    state_dict = {}
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if safetensor_files:
        from safetensors import safe_open

        for sf in safetensor_files:
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if wanted(key):
                        state_dict[key] = f.get_tensor(key)
        return state_dict

    # Fallback: legacy PyTorch .bin checkpoints (e.g. deepseek-llm-7b ships
    # only pytorch_model*.bin, no safetensors) — #59. Match ONLY weight-shard
    # filename patterns (never bare *.bin, which would pick up training_args.bin
    # / optimizer.bin and choke torch.load).
    bin_files = []
    for pat in ("pytorch_model*.bin", "model*.bin", "consolidated*.bin"):
        bin_files += glob.glob(os.path.join(model_dir, pat))
    bin_files = sorted(set(bin_files))
    if not bin_files:
        raise FileNotFoundError(
            f"no *.safetensors or weight *.bin (pytorch_model*/model*/consolidated*) "
            f"in {model_dir}"
        )
    for bf in bin_files:
        try:
            # mmap=True lazy-maps the shard, so we materialize only the few
            # layers this stage keeps (via .clone()) instead of loading the
            # whole multi-GB shard into RAM.
            shard = torch.load(bf, map_location="cpu", weights_only=True, mmap=True)
        except Exception as e:
            # Older torch (no mmap kwarg) or a checkpoint weights_only can't
            # unpickle — fall back loudly for trusted local/HF weights.
            print(
                f"  torch.load(weights_only,mmap) failed ({e}); retrying full unpickle",
                flush=True,
            )
            shard = torch.load(bf, map_location="cpu", weights_only=False)
        for key, tensor in shard.items():
            if wanted(key):
                state_dict[key] = tensor.clone() if hasattr(tensor, "clone") else tensor
        del shard
        gc.collect()
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


def make_stateful_with_init(ov_model, add_reorder=True):
    """Make the (stateless) past_kv-in / present-out model stateful, building
    ReadValue/Assign nodes WITH explicit inits.

    `apply_make_stateful_transformation` emits init-less ReadValues (0 inputs),
    which the OpenVINO CPU plugin rejects ("Node ReadValue ... contains less
    parent edges than 0", #57). Constructing the nodes directly lets us attach
    a zero-length init of the right shape/element-type, so the IR loads on the
    CPU plugin AND keeps the beam_idx + Gather IndirectKVCache the GPU needs for
    speed (`add_reorder`). A zero-length, batch-1 init also gives the first-step
    KV Concat a correct {1,h,0,d} state instead of an empty {0,h,0,d} one.
    """
    import numpy as np
    import openvino as ov
    import openvino.opset13 as ops
    from openvino import PartialShape, Type, Tensor
    from openvino.op.util import Variable, VariableInfo

    beam_idx = axis0 = None
    if add_reorder:
        beam_idx = ops.parameter(PartialShape([-1]), Type.i32)
        beam_idx.set_friendly_name("beam_idx")
        beam_idx.output(0).set_names({"beam_idx"})
        axis0 = ops.constant(np.array(0, dtype=np.int32))

    # present.N.{key,value} sources, mapped by the index-based naming convention.
    present_src = {}
    keep_results = []
    for i, r in enumerate(ov_model.get_results()):
        if i == 0:
            keep_results.append(r)  # logits / hidden_states
            continue
        kv_idx = i - 1
        kv = "value" if kv_idx % 2 == 1 else "key"
        present_src[f"present.{kv_idx // 2}.{kv}"] = r.input_value(0)

    sinks, keep_params = [], []
    for p in ov_model.get_parameters():
        nm = p.output(0).get_any_name()
        if not nm.startswith("past_key_values."):
            keep_params.append(p)
            continue
        src = present_src[nm.replace("past_key_values.", "present.", 1)]
        et = p.get_element_type()
        ps = p.get_partial_shape()
        # The zero-length init below assumes the canonical static
        # [batch, kv_heads, seq, head_dim] KV layout (only seq is dynamic).
        # Fail loudly on anything else rather than mis-init silently.
        if len(ps) != 4 or not ps[1].is_static or not ps[3].is_static:
            raise RuntimeError(
                f"{nm}: expected static [batch, kv_heads, seq, head_dim] KV "
                f"shape, got {ps}"
            )
        dims = [
            (d.get_length() if d.is_static else (0 if idx >= 2 else 1))
            for idx, d in enumerate(ps)
        ]
        vi = VariableInfo()
        vi.data_shape = ps
        vi.data_type = et
        vi.variable_id = nm
        var = Variable(vi)
        rv = ops.read_value(ops.constant(Tensor(et, dims)), var)
        feed = rv.output(0)
        if add_reorder:
            feed = ops.gather(rv.output(0), beam_idx, axis0).output(0)
        p.output(0).replace(feed)
        # Assign takes a Node (not an Output); the present-KV source is the
        # sole output of its producing node, so output index must be 0.
        if src.get_index() != 0:
            raise RuntimeError(
                f"present source for {nm} is output #{src.get_index()}; "
                "assign needs a single-output node"
            )
        sinks.append(ops.assign(src.get_node(), var))

    new_params = keep_params + ([beam_idx] if add_reorder else [])
    new_model = ov.Model(keep_results, sinks, new_params)
    new_model.validate_nodes_and_infer_types()
    return new_model


def export_single_stage(
    model_dir, output_dir, stage_plan, config, quantization, rope_theta, arch_tag,
    cache_reorder=True,
    target="cpu-gpu", static_seq=1, static_context=1024,
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
    # Attach per-architecture export deviations (Gemma embed scale, Gemma-2
    # softcapping / 4-norm structure — #61). Default ArchSpec is a no-op for
    # Llama-family models.
    wrapper.arch = arch_spec_from_config(
        config,
        arch_tag,
        getattr(config, "head_dim", None)
        or (config.hidden_size // config.num_attention_heads),
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

    # 5. Name inputs/outputs and set shapes. NPU requires fully STATIC shapes
    # (it cannot compile dynamic dims — #37); cpu-gpu keeps dynamic seq/KV.
    npu = target == "npu"
    past_len = max(static_context - static_seq, 1)
    for i, inp in enumerate(ov_model.inputs):
        shape = inp.partial_shape
        if i == 0:
            name = "input_ids" if has_embed else "hidden_states"
            if len(shape) >= 2:
                shape[1] = static_seq if npu else -1
            # Relay/last stages take hidden_states [batch, seq, hidden]; the
            # hidden dim is dynamic out of the tracer, so pin it for NPU too
            # (the seq pin above is not enough — the NPU compiler rejects the
            # dynamic last dim).
            if npu and not has_embed and len(shape) >= 3:
                shape[2] = config.hidden_size
        elif i == 1:
            name = "attention_mask"
            if len(shape) >= 2:
                shape[1] = static_context if npu else -1
        elif i == 2:
            name = "position_ids"
            if len(shape) >= 2:
                shape[1] = static_seq if npu else -1
        else:
            kv_idx = i - 3
            layer_local = kv_idx // 2
            is_value = kv_idx % 2 == 1
            kv_type = "value" if is_value else "key"
            name = f"past_key_values.{layer_local}.{kv_type}"
            if len(shape) >= 3:
                # Static batch=1 so the cpu-gpu stateful KV variable
                # initialises to batch 1, not 0 — otherwise the first-step KV
                # Concat fails on the CPU plugin ({0,h,0,d} vs {1,h,S,d}).
                # cascadia always runs a single sequence (batch 1). For NPU the
                # past length is also pinned static (context - seq).
                shape[0] = 1
                shape[2] = past_len if npu else -1
        if npu and len(shape) >= 1:
            # NPU: batch must be static (=1) on EVERY input, not just KV —
            # a dynamic batch leaves unbounded upper bounds the compiler rejects.
            shape[0] = 1
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

    # 6+7. Make stateful with init-bearing ReadValues (+ beam_idx cache-reorder).
    # NPU mode skips this entirely: the NPU stack does not support state
    # variables, so the model stays STATELESS (explicit past_kv in / present_kv
    # out) with static shapes — #37. The cpu-gpu path builds the stateful nodes
    # directly (init-bearing ReadValues) instead of
    # apply_make_stateful_transformation, whose init-less ReadValues the
    # OpenVINO CPU plugin rejects (#57) — the same IR then loads on CPU AND
    # keeps the IndirectKVCache (cache_reorder) the GPU needs for speed.
    if not npu:
        ov_model = make_stateful_with_init(ov_model, add_reorder=cache_reorder)
        print(
            f"  stateful: init-bearing ReadValue/Assign ({num_layers} KV pairs)"
            + (" + cache_reorder (beam_idx Gather)" if cache_reorder else ""),
            flush=True,
        )
    else:
        print(
            "  NPU mode: stateless (explicit KV in/out) + static shapes — "
            "skipping make_stateful + cache_reorder (#37)",
            flush=True,
        )

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

    # NPU: EVERY dim (inputs AND outputs) must be static or the compiler
    # rejects the IR (#37). Run this on the FINAL graph — after make_stateful is
    # skipped AND after nncf weight compression — so a dynamic dim introduced by
    # a decompression subgraph is also caught, not just the post-trace shape. We
    # pinned the inputs earlier (incl. the relay hidden dim); re-infer so the
    # outputs pick up static shapes, then verify. Catching it here fails the
    # export instead of producing an IR that only blows up at NPU load (and that
    # CPU/GPU would happily accept).
    if npu:
        ov_model.validate_nodes_and_infer_types()

        def _dynamic(ports):
            out = []
            for idx, p in enumerate(ports):
                ps = p.partial_shape
                if not ps.is_static:
                    try:
                        nm = p.get_any_name()
                    except Exception:
                        nm = f"#{idx}"
                    out.append(f"{nm}{ps}")
            return out

        bad_in = _dynamic(ov_model.inputs)
        bad_out = _dynamic(ov_model.outputs)
        if bad_in or bad_out:
            raise RuntimeError(
                f"NPU export (stage {stage_idx}): shapes still dynamic after static pinning "
                f"+ compression — the NPU compiler will reject this IR (#37). Pin or fold these "
                f"dims. dynamic inputs={bad_in} dynamic outputs={bad_out}"
            )
        print(
            f"  NPU static-shape check OK: all {len(ov_model.inputs)} inputs + "
            f"{len(ov_model.outputs)} outputs static",
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
        "stateful": not npu,
        "rope_theta": rope_theta,
        "arch_tag": arch_tag,
        "target": target,
        "static_seq": static_seq if npu else None,
        "static_context": static_context if npu else None,
        "inputs": (
            "input_ids/hidden_states, attention_mask, position_ids, "
            "past_key_values.* (explicit KV in/out)"
            if npu
            else "input_ids/hidden_states, attention_mask, position_ids"
            + (", beam_idx" if cache_reorder else "")
            + " (KV cache is internal state)"
        ),
        "cache_reorder": cache_reorder,
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

    cache_root = os.path.expanduser("~/.cache/cascadia/models")
    safe_id = model_id_or_path.replace("/", "--")
    local_dir = os.path.join(cache_root, safe_id)
    print(f"Downloading {model_id_or_path} -> {local_dir}", flush=True)
    common = ["*.json", "tokenizer.*", "*.model", "special_tokens_map.json"]
    # Prefer safetensors. Only fetch legacy .bin shards if the repo has no
    # safetensors — avoids pulling a redundant multi-GB .bin for the many repos
    # that ship both, and only matches weight-shard names (not training_args.bin).
    snapshot_download(
        repo_id=model_id_or_path,
        local_dir=local_dir,
        allow_patterns=["*.safetensors", *common],
        max_workers=8,
    )
    if not glob.glob(os.path.join(local_dir, "*.safetensors")):
        print("  no safetensors in repo; fetching legacy .bin weight shards", flush=True)
        snapshot_download(
            repo_id=model_id_or_path,
            local_dir=local_dir,
            allow_patterns=[
                "pytorch_model*.bin",
                "model*.bin",
                "consolidated*.bin",
                *common,
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
        "--target",
        choices=["cpu-gpu", "npu"],
        default="cpu-gpu",
        help="Deployment target. 'npu' emits a STATELESS, STATIC-shape IR "
        "(no make_stateful, fixed seq/KV) that the NPU compiler accepts (#37).",
    )
    parser.add_argument(
        "--static-seq",
        type=int,
        default=1,
        help="NPU only: fixed query-window length (default 1 = decode step).",
    )
    parser.add_argument(
        "--static-context",
        type=int,
        default=1024,
        help="NPU only: fixed total context; past-KV length = context - seq.",
    )
    args = parser.parse_args()

    # The NPU runtime decodes one token per step and derives past-KV length as
    # static_context - 1, so reject configs it cannot load (fail fast — the
    # export takes minutes). chunked prefill (static_seq > 1) is not yet wired
    # into the runtime.
    if args.target == "npu":
        if args.static_seq != 1:
            parser.error(
                "--static-seq must be 1 for --target npu (the runtime decodes one token "
                "per step; chunked prefill is not implemented yet)"
            )
        if args.static_context <= args.static_seq:
            parser.error(
                f"--static-context ({args.static_context}) must be > --static-seq "
                f"({args.static_seq}) so past-KV length >= 1"
            )
        if args.default_dtype != "fp16":
            parser.error(
                f"--default-dtype must be fp16 for --target npu (the runtime feeds KV as "
                f"f16; --default-dtype {args.default_dtype} would emit f32 KV ports and fail "
                f"the present-shape check at first inference)"
            )
    elif args.static_seq != 1 or args.static_context != 1024:
        print(
            "WARNING: --static-seq/--static-context are ignored without --target npu "
            "(the cpu-gpu export is dynamic-shape)",
            flush=True,
        )

    if args.default_dtype == "fp16":
        torch.set_default_dtype(torch.float16)

    from transformers import AutoConfig

    # Read the config FIRST (cheap — just config.json, no weights) so we can
    # reject unsupported models before downloading tens-to-hundreds of GB.
    # AutoConfig.from_pretrained works on an HF id (fetches only config.json)
    # or a local dir.
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    if is_moe_config(config):
        sys.exit(
            f"ERROR: {args.model} is a mixture-of-experts (MoE) model, which the "
            f"cascadia exporter does not support (#60). It builds dense decoder "
            f"layers (one MLP per layer); MoE routing + per-expert MLPs are not "
            f"implemented, and falling back to a dense layer would silently emit "
            f"garbage. Aborting rather than producing a broken shard."
        )
    # The traced RoPE rotates the FULL head_dim with plain inv_freq. Reject
    # models needing partial rotary (rotating only a head_dim slice, e.g.
    # phi-2) — they would silently emit garbage. Long-context rope_scaling
    # (llama3/yarn/longrope) is identity-ish within the original context, so
    # short generations match; warn rather than reject those.
    prf = getattr(config, "partial_rotary_factor", None)
    if isinstance(prf, (int, float)) and prf < 1.0:
        sys.exit(
            f"ERROR: {args.model} uses partial rotary (partial_rotary_factor={prf}); the "
            f"exporter applies RoPE to the full head_dim and would emit garbage. Not supported."
        )
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rtype = (rope_scaling.get("rope_type") or rope_scaling.get("type") or "").lower()
        if rtype in ("longrope", "su", "yarn"):
            print(
                f"  WARNING: rope_scaling type {rtype!r} is not modeled (plain RoPE baked); "
                f"output will diverge from reference beyond the original context length.",
                flush=True,
            )
    arch_tag = detect_architecture(config)
    # Accepted — now fetch the weights.
    model_dir = maybe_download(args.model)
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
            target=args.target,
            static_seq=args.static_seq,
            static_context=args.static_context,
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
