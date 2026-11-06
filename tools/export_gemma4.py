#!/usr/bin/env python3
"""cascadia Gemma 4 exporter — per-stage stateful OpenVINO IR shards.

Ported from the reference Python prototype (2026-04-24) that produced
working multi-stage Gemma 4 E2B shards for a 2-stage worker /
coordinator pipeline. The algorithm handles every Gemma 4-specific
quirk the generic exporter in `export_shards.py` cannot:

* Per-layer-type asymmetric attention — `head_dim=256` for
  sliding-attention layers vs `global_head_dim=512` for
  full-attention layers.
* Per-layer-type RoPE config — sliding uses theta=10k with default
  rope_type; global uses theta=1M with `rope_type="proportional"` and
  `partial_rotary_factor=0.25` (only the first 25% of each head's
  dims are rotated; trailing dims pass through).
* KV-sharing across layers — for E2B (20 of 35 layers reuse a
  previous layer's KV cache) and E4B (18 of 42). Cross-stage KV
  passing is handled via explicit external_kv/cross_kv I/O.
* Per-layer embeddings (PLI) — E2B / E4B have a side-channel
  `embed_tokens_per_layer` that modulates every layer's MLP output;
  the per-layer slice flows alongside `hidden_states` between
  pipeline stages.
* Restored `final_logit_softcapping=30.0` — Gemma 3 dropped softcap;
  Gemma 4 brought back the final variant. Applied in the head stage.
* Q/K/V `RMSNorm` per head before rotary.
* Sliding-window attention — `sliding_attention` layers are masked to
  `text_config.sliding_window` (512 on E2B/E4B, 1024 on 31B) while
  `full_attention` layers see the whole prefix. The window is recorded
  as `sliding_window` in stage/pipeline config, under
  `export_version = gemma4_cached_v1.1`; v1 trees lack the band and let
  sliding layers attend past their window.

Variants covered:

| Variant | Layers | head_dim (l/g) | num_kv_heads (l/g) | PLI | KV-sharing | Status |
|---------|-------:|----------------|--------------------|-----|------------|--------|
| E2B-it  |     35 | 256 / 512      | 1 / —              | 256 | 20         | tested |
| E4B-it  |     42 | 256 / 512      | 2 / —              | 256 | 18         | tested |
| 31B-it  |     60 | 256 / 512      | 16 / 4             |   0 |  0         | tested |
| 26B-A4B-it (MoE) | 48 | — | — | — | — | NOT SUPPORTED (MoE block) |

INT4 quantisation: optional via `--quantization int4` (calls into
nncf, same path as the generic exporter).
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import shutil
import sys
import time
from typing import Any


# Files `--vision` copies from the vision directory into the tree root.
VISION_FILES = (
    "openvino_vision_embeddings_model.xml",
    "openvino_vision_embeddings_model.bin",
    "preprocessor_config.json",
    "processor_config.json",
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# OV/torch patches required by the Gemma 4 trace path
# ---------------------------------------------------------------------------


def apply_patches() -> None:
    """Apply the two patches the Gemma 4 trace path needs:

    1. ``openvino.frontend.pytorch.utils.torch_tensor_to_ov_const`` chokes
       on 0-dim scalar buffers (Gemma 4 has per-layer scalar weights
       like ``layer_scalar``). Reshape to (1,) before constancy.

    2. After ``AutoModelForCausalLM.from_pretrained``, walk all
       0-dim buffers in the model and reshape to (1,) so the same
       issue doesn't surface during trace via ``register_buffer``.

    These mirror the reference exporter's patches. Idempotent; safe
    to call multiple times.
    """
    import openvino.frontend.pytorch.utils as ov_utils

    if getattr(ov_utils, "_cascadia_patched", False):
        return
    _orig = ov_utils.torch_tensor_to_ov_const

    def _patched(tensor, shared_memory=False):
        if hasattr(tensor, "dim") and tensor.dim() == 0:
            tensor = tensor.reshape(1)
        return _orig(tensor, shared_memory)

    ov_utils.torch_tensor_to_ov_const = _patched
    ov_utils._cascadia_patched = True


def fix_zero_dim_buffers(model) -> int:
    """Reshape every 0-dim buffer on the model to shape (1,). Returns
    the count of patched buffers (for logging)."""
    fixed = 0
    for n, b in model.named_buffers():
        if b.dim() == 0:
            parts = n.split(".")
            obj = model
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], b.reshape(1))
            fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Custom rotary — handles Gemma 4's "proportional" rope_type
# ---------------------------------------------------------------------------


def rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary on (b, heads, seq, head_dim) tensors. cos/sin are
    (b, seq, head_dim)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


def _make_traced_rotary_class():
    """Return the Gemma-4-flavoured TracedRotaryEmbedding nn.Module.

    Returned lazily so this file can be imported in a context that
    lacks torch (e.g. running unit tests on a host without torch).
    """
    import torch

    class GemmaTracedRotaryEmbedding(torch.nn.Module):
        """Trace-friendly rotary for one Gemma 4 layer type.

        Holds inv_freq as a buffer; forward computes cos/sin in FP32
        and casts to ``target_dtype`` at the end so the entire output
        graph has a consistent dtype. (HF's
        ``Gemma3nTextRotaryEmbedding`` internally mixes FP16/FP32, and
        torch.jit.trace captures the mismatch into the OV IR;
        opset1::MatMul on OV 2026.1 GPU rejects mismatched element
        types — a regression vs 2026.0 which auto-promoted.)

        Two rope_types are honoured per Gemma 4:

        * ``"default"`` (sliding_attention): full head_dim rotated.
        * ``"proportional"`` (full_attention with
          ``partial_rotary_factor=0.25`` on Gemma 4): only the first
          ``partial_rotary_factor * head_dim`` dims of each head are
          rotated; the trailing dims pass through. We model this by
          padding the back of ``inv_freq`` with zeros (cos=1 / sin=0
          on those dims is a no-op).
        """

        def __init__(self, head_dim, rope_theta, partial_rotary_factor=1.0):
            super().__init__()
            self.head_dim = head_dim
            self.partial_rotary_factor = float(partial_rotary_factor)
            if partial_rotary_factor >= 1.0:
                inv_freq = 1.0 / (
                    rope_theta
                    ** (
                        torch.arange(0, head_dim, 2, dtype=torch.float32)
                        / head_dim
                    )
                )
            else:
                rope_angles = int(partial_rotary_factor * head_dim // 2)
                if rope_angles <= 0:
                    raise ValueError(
                        f"partial_rotary_factor={partial_rotary_factor} "
                        f"with head_dim={head_dim} leaves 0 rotated dims"
                    )
                inv_freq_rotated = 1.0 / (
                    rope_theta
                    ** (
                        torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                        / head_dim
                    )
                )
                nope_angles = head_dim // 2 - rope_angles
                if nope_angles > 0:
                    inv_freq = torch.cat(
                        [
                            inv_freq_rotated,
                            torch.zeros(nope_angles, dtype=torch.float32),
                        ],
                        dim=0,
                    )
                else:
                    inv_freq = inv_freq_rotated
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, position_ids, target_dtype):
            bsz, seq_len = position_ids.shape
            inv_freq_expanded = self.inv_freq[None, None, :].expand(
                bsz, seq_len, -1
            )
            freqs = position_ids[:, :, None].float() * inv_freq_expanded
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(target_dtype)
            sin = emb.sin().to(target_dtype)
            return cos, sin

    return GemmaTracedRotaryEmbedding


# ---------------------------------------------------------------------------
# Per-layer cached forward
# ---------------------------------------------------------------------------


# Shard format stamped into stage_config.json / pipeline_config.json.
# gemma4_cached_v1 (pre-fix) built a plain causal mask for every layer, so
# sliding_attention layers attended past their window; v1.1 bakes the window
# band and records it under "sliding_window". Readers match on the
# "gemma4" prefix (genai.rs), so bumping here never changes routing.
EXPORT_VERSION = "gemma4_cached_v1.1"

SLIDING_LAYER = "sliding_attention"
FULL_LAYER = "full_attention"
KNOWN_LAYER_TYPES = frozenset({SLIDING_LAYER, FULL_LAYER})


def resolve_sliding_window(text_config, layer_types=None):
    """Window to bake into `sliding_attention` layers, or None for no bound.

    Read config-first (before the multi-minute model load) so a bad config
    fails fast. `text_config.sliding_window` absent, None or 0 means no bound
    — acceptable only when `layer_types` has no `sliding_attention` layer
    (HF itself refuses to build a sliding mask without a window (None); a
    window of 0 it does not refuse, so here the exporter is stricter), so
    callers pass the model's `layer_types` and a windowless config with sliding
    layers is rejected instead of exporting the pre-v1.1 unmasked behaviour
    under a v1.1 stamp. A negative window is a configuration error rather
    than something to pass through: `build_attention_mask` would mask every
    key and softmax would return NaN, and the export would still succeed.
    Bool / non-integral values (only reachable through `trust_remote_code`
    configs) are rejected rather than coerced.
    """
    try:
        from transformers.integrations.heterogeneity.configuration_utils import (
            AmbiguousGlobalPerLayerAttributeError as _PerLayerError,
        )
    except ImportError:
        # transformers < 5.14 has no per-layer overrides; `except ()` catches
        # nothing, so every read error keeps its own message.
        _PerLayerError = ()
    try:
        raw = getattr(text_config, "sliding_window", None)
    except _PerLayerError as e:  # transformers >= 5.14 per-layer overrides
        raise ValueError(
            "text_config.sliding_window is overridden per layer "
            f"(per_layer_config); the exporter bakes one window per stage: {e}"
        ) from e
    window = None
    if raw is not None:
        try:
            integral = not isinstance(raw, bool) and int(raw) == raw
        except (TypeError, ValueError, OverflowError):
            # A per-layer list, nan, inf or a non-numeric string: the bare
            # conversion error carries no hint of which config key is at
            # fault, so report it like any other non-integer.
            integral = False
        if not integral:
            raise ValueError(
                f"text_config.sliding_window must be an integer (got {raw!r})"
            )
        window = int(raw)
        if window < 0:
            raise ValueError(
                f"text_config.sliding_window must be >= 0 (got {window}); "
                "0 or absent means no window"
            )
        window = window or None
    if window is None and layer_types is not None and SLIDING_LAYER in layer_types:
        raise ValueError(
            "text_config has sliding_attention layers but no positive "
            "sliding_window; refusing to export them as full-causal (that is "
            "the gemma4_cached_v1 bug this exporter fixes). Check the config "
            "class exposes `sliding_window`."
        )
    return window


def build_attention_mask(seq_len, full_seq_len, sliding_window, device, dtype):
    """Additive attention mask for one Gemma 4 decoder layer.

    Rows are queries (the `seq_len` new tokens), columns are keys (the whole
    `full_seq_len` cache including the new tokens), so query `i` sits at
    absolute position `past + i` where `past = full_seq_len - seq_len`.

    Always causal: query `i` may not see key `j > past + i`.

    `sliding_window` additionally bounds how far back a query may look, which
    is what `sliding_attention` layers do (`full_attention` layers pass None).
    HF's `create_sliding_window_causal_mask` keeps key `j` when
    `0 <= (past + i) - j < sliding_window`, so the extra masked band is
    `j - i <= past - sliding_window`. Without it every layer runs as full
    attention, which agrees with HF only while the sequence is shorter than
    the window -- so short prompts look correct and long ones silently drift.

    Raises `ValueError` for `sliding_window <= 0`: a zero or negative band
    would cover the diagonal, every row would be fully masked, and softmax
    would return NaN.
    """
    import torch

    if sliding_window is not None and sliding_window < 1:
        raise ValueError(
            f"sliding_window must be >= 1 (got {sliding_window}); "
            "pass None for full attention"
        )
    past = full_seq_len - seq_len
    # One -inf plane; triu/tril are out-of-place, so both bands share it.
    neg = torch.full((seq_len, full_seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(neg, diagonal=past + 1)
    if sliding_window is not None:
        mask = mask + torch.tril(neg, diagonal=past - sliding_window)
    return mask


def cached_gemma4_layer_forward(
    layer,
    hidden_states,
    cos,
    sin,
    past_key,
    past_value,
    num_heads,
    num_kv_heads,
    head_dim,
    per_layer_input,
    sliding_window=None,
    causal_mask=None,
):
    """Manual cached forward for one Gemma 4 decoder layer.

    `causal_mask` (rows = this step's queries, cols = the whole cache) is the
    wrapper's per-forward mask shared by every layer of the same type; when
    None the mask is built here from `sliding_window`.

    Handles: input_layernorm, Q/K/V with norms + rotary, KV cache
    concat, GQA expansion, attention (scaling=1.0 — Q/K norms handle
    magnitude), o_proj, post_attention_layernorm, pre/post
    _feedforward_layernorm, MLP, optional PLI modulation, optional
    layer_scalar.
    """
    import torch
    import torch.nn.functional as F

    bsz, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    q = layer.self_attn.q_proj(hidden_states)
    q = q.view(bsz, seq_len, num_heads, head_dim)
    q = layer.self_attn.q_norm(q)
    q = q.transpose(1, 2)

    k = layer.self_attn.k_proj(hidden_states)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim)
    # k_eq_v (Gemma 4 31B global/full_attention layers): v_proj is None and V
    # derives from the RAW k_proj output — HF sets value_states = key_states
    # BEFORE k_norm/rope are applied. Capture v from the pre-norm k here.
    if layer.self_attn.v_proj is not None:
        v = layer.self_attn.v_proj(hidden_states)
        v = v.view(bsz, seq_len, num_kv_heads, head_dim)
    else:
        v = k
    k = layer.self_attn.k_norm(k)
    k = k.transpose(1, 2)

    v = layer.self_attn.v_norm(v)
    v = v.transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)

    k = torch.cat([past_key, k], dim=2)
    v = torch.cat([past_value, v], dim=2)

    k_exp = k.unsqueeze(2).expand(
        bsz, num_kv_heads, num_kv_groups, -1, head_dim
    )
    k_exp = k_exp.reshape(bsz, num_heads, -1, head_dim)
    v_exp = v.unsqueeze(2).expand(
        bsz, num_kv_heads, num_kv_groups, -1, head_dim
    )
    v_exp = v_exp.reshape(bsz, num_heads, -1, head_dim)

    attn_weights = torch.matmul(q, k_exp.transpose(2, 3))

    if causal_mask is None:
        causal_mask = build_attention_mask(
            seq_len, k.shape[2], sliding_window, q.device, q.dtype
        )
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    attn_output = torch.matmul(attn_weights, v_exp)

    attn_output = (
        attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
    )
    attn_output = layer.self_attn.o_proj(attn_output)

    hidden_states = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if per_layer_input is not None and getattr(
        layer, "hidden_size_per_layer_input", 0
    ):
        residual = hidden_states
        hidden_states = layer.per_layer_input_gate(hidden_states)
        hidden_states = layer.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = layer.per_layer_projection(hidden_states)
        hidden_states = layer.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    if hasattr(layer, "layer_scalar"):
        hidden_states = hidden_states * layer.layer_scalar

    return hidden_states, k, v


def cached_gemma4_shared_layer_forward(
    layer,
    hidden_states,
    cos,
    sin,
    source_key,
    source_value,
    num_heads,
    num_kv_heads,
    head_dim,
    per_layer_input,
    sliding_window=None,
    causal_mask=None,
):
    """Forward for KV-shared layers (Gemma 4 E2B/E4B). Uses the source
    layer's K/V instead of computing its own. Only Q projection runs.
    `causal_mask`: see `cached_gemma4_layer_forward`.
    """
    import torch
    import torch.nn.functional as F

    bsz, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    q = layer.self_attn.q_proj(hidden_states)
    q = q.view(bsz, seq_len, num_heads, head_dim)
    q = layer.self_attn.q_norm(q)
    q = q.transpose(1, 2)
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    q = q * cos_u + rotate_half(q) * sin_u

    k = source_key
    v = source_value

    k_exp = k.unsqueeze(2).expand(
        bsz, num_kv_heads, num_kv_groups, -1, head_dim
    )
    k_exp = k_exp.reshape(bsz, num_heads, -1, head_dim)
    v_exp = v.unsqueeze(2).expand(
        bsz, num_kv_heads, num_kv_groups, -1, head_dim
    )
    v_exp = v_exp.reshape(bsz, num_heads, -1, head_dim)

    attn_weights = torch.matmul(q, k_exp.transpose(2, 3))
    if causal_mask is None:
        causal_mask = build_attention_mask(
            seq_len, k.shape[2], sliding_window, q.device, q.dtype
        )
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    attn_output = torch.matmul(attn_weights, v_exp)

    attn_output = (
        attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
    )
    attn_output = layer.self_attn.o_proj(attn_output)

    hidden_states = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if per_layer_input is not None and getattr(
        layer, "hidden_size_per_layer_input", 0
    ):
        residual = hidden_states
        hidden_states = layer.per_layer_input_gate(hidden_states)
        hidden_states = layer.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = layer.per_layer_projection(hidden_states)
        hidden_states = layer.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    if hasattr(layer, "layer_scalar"):
        hidden_states = hidden_states * layer.layer_scalar
    return hidden_states


# ---------------------------------------------------------------------------
# Stage planning + KV-share resolution
# ---------------------------------------------------------------------------


def compute_stage_plan(num_layers: int, num_stages: int) -> list[dict]:
    """Uniform layer split. For uneven layer counts, the first
    `(num_layers % num_stages)` stages get one extra layer each."""
    if num_stages > num_layers:
        raise ValueError(
            f"num_stages ({num_stages}) cannot exceed num_layers "
            f"({num_layers})"
        )
    base, rem = divmod(num_layers, num_stages)
    plan, offset = [], 0
    for i in range(num_stages):
        count = base + (1 if i < rem else 0)
        plan.append(
            {
                "stage": i,
                "layer_start": offset,
                "layer_end": offset + count,
                "has_embed": (i == 0),
                "has_head": (i == num_stages - 1),
            }
        )
        offset += count
    return plan


def resolve_kv_sharing(text_config, stage_plan):
    """Compute per-layer KV-sharing metadata for one stage.

    Returns a dict with:
        is_shared: list[bool], length=num_layers_in_stage
        source_local: list[int]
            non-negative -> source layer's local index in this stage
            negative     -> -(external_idx + 1) for KV received from
                            a previous stage
        cross_stage_sources: list[int]
            local indices in this stage whose KV is consumed by a later
            stage; their KV stays as regular output (not stateful).
        external_shared_sources: list[(src_global_idx, layer_type, head_dim)]
            sources from an earlier stage that this stage needs.

    For models with num_kv_shared_layers == 0 (31B), is_shared is all
    False and the cross/external lists are empty (zero-cost path).
    """
    ls, le = stage_plan["layer_start"], stage_plan["layer_end"]
    num_layers = le - ls
    num_total = text_config.num_hidden_layers
    num_kv_shared = text_config.num_kv_shared_layers or 0
    first_non_shared = num_total - num_kv_shared
    layer_types = text_config.layer_types
    prev_types = layer_types[:first_non_shared]

    is_shared: list[bool] = []
    source_local: list[int] = []
    for i in range(num_layers):
        global_idx = ls + i
        if global_idx >= first_non_shared:
            lt = layer_types[global_idx]
            try:
                src_global = (
                    len(prev_types) - 1 - prev_types[::-1].index(lt)
                )
            except ValueError:
                raise RuntimeError(
                    f"Gemma 4 KV-share: layer {global_idx} type={lt!r} "
                    "has no earlier source layer of matching type"
                )
            src_local = src_global - ls
            is_shared.append(True)
            source_local.append(
                src_local if 0 <= src_local < num_layers else -1
            )
        else:
            is_shared.append(False)
            source_local.append(-1)

    # Cross-stage sources: layers in THIS stage whose KV later stages need.
    cross_stage_sources = set()
    for gi in range(le, num_total):
        if gi >= first_non_shared:
            lt = layer_types[gi]
            src_g = len(prev_types) - 1 - prev_types[::-1].index(lt)
            if ls <= src_g < le:
                cross_stage_sources.add(src_g - ls)

    # External shared KV: sources from a PREVIOUS stage that we need.
    external_shared_sources: list[tuple[int, str, int]] = []
    for i in range(num_layers):
        if is_shared[i] and source_local[i] == -1:
            lt = layer_types[ls + i]
            hd = (
                getattr(text_config, "global_head_dim", text_config.head_dim)
                if lt == "full_attention"
                else text_config.head_dim
            )
            src_global = (
                len(prev_types) - 1 - prev_types[::-1].index(lt)
            )
            key = (src_global, lt, hd)
            if key not in external_shared_sources:
                external_shared_sources.append(key)
            ext_idx = next(
                j
                for j, s in enumerate(external_shared_sources)
                if s[0] == src_global
            )
            source_local[i] = -(ext_idx + 1)

    return {
        "is_shared": is_shared,
        "source_local": source_local,
        "cross_stage_sources": sorted(cross_stage_sources),
        "external_shared_sources": external_shared_sources,
    }


# ---------------------------------------------------------------------------
# Wrapper construction
# ---------------------------------------------------------------------------


def splice_image_rows(h, is_img, image_features):
    """Traceable equivalent of HF's ``h.masked_scatter(is_img[..., None],
    image_features)`` for batch 1: row k of ``image_features`` [M, hidden]
    replaces the k-th image position of ``h`` [1, T, hidden]. A zero row is
    appended so a step without images (M = 0) still gathers from a
    non-empty tensor; positions that are not images keep ``h``."""
    import torch as _t

    feats = _t.cat([image_features.to(h.dtype), _t.zeros_like(h[0, :1, :])], dim=0)
    slot = (_t.cumsum(is_img.to(_t.int64), dim=1) - 1).clamp(min=0)
    rows = feats[slot[0]].unsqueeze(0)
    return _t.where(is_img.unsqueeze(-1), rows, h)


def build_cached_wrapper(model, text_config, stage_plan, sliding_window=None, vision_ids=None):
    """Build a CachedStageWrapper for one stage. Returns (wrapper,
    head_dims_per_layer, kv_share_info). `sliding_window` is the value
    `run_export` resolved config-first (None = sliding layers stay unbounded;
    `run_export` only passes None when the model has no sliding layers).

    `vision_ids` = (image_token_id, pad_token_id) makes the embed stage take
    a trailing `image_features` [M, hidden] input and splice it in the way
    HF's Gemma4Model.forward does: image tokens embed (and look up their
    per-layer embedding) as the pad id, then their rows of the main
    embedding are replaced, in order, by the image features, and the
    per-layer projection reads the spliced embedding."""
    import torch
    import torch.nn as nn

    GemmaTracedRotaryEmbedding = _make_traced_rotary_class()

    ls, le = stage_plan["layer_start"], stage_plan["layer_end"]
    has_embed, has_head = stage_plan["has_embed"], stage_plan["has_head"]
    num_total = text_config.num_hidden_layers
    hidden_dim = text_config.hidden_size
    pli_dim = text_config.hidden_size_per_layer_input or 0
    num_layers = le - ls

    # Gemma 4 wraps the language model under model.model.language_model
    # (Gemma4ForConditionalGeneration's text backbone).
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        tm = model.model.language_model
        lm_head_ref = model.lm_head
    elif hasattr(model, "language_model"):
        tm = model.language_model
        lm_head_ref = model.lm_head
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Text-only Gemma4ForCausalLM variant: text backbone at model.model
        tm = model.model
        lm_head_ref = model.lm_head
    else:
        raise RuntimeError(
            "Unable to locate Gemma 4 text backbone on model "
            f"(type={type(model).__name__}); expected "
            "model.model.language_model or model.language_model or "
            "model.model.layers."
        )

    stage_layers = list(tm.layers)[ls:le]
    stage_layer_types = text_config.layer_types[ls:le]

    share = resolve_kv_sharing(text_config, stage_plan)
    is_shared = share["is_shared"]
    source_local = share["source_local"]
    cross_stage_sources = share["cross_stage_sources"]
    external_shared_sources = share["external_shared_sources"]

    n_shared = sum(is_shared)
    n_own_kv = sum(not s for s in is_shared)
    log(
        f"  KV sharing: {n_own_kv} own + {n_shared} shared, "
        f"{len(cross_stage_sources)} cross-stage sources out, "
        f"{len(external_shared_sources)} external sources in"
    )

    head_dim_local = text_config.head_dim
    head_dim_global = getattr(
        text_config, "global_head_dim", head_dim_local
    )

    head_dims = [
        (head_dim_global if lt == "full_attention" else head_dim_local)
        for lt in stage_layer_types
    ]

    num_heads = text_config.num_attention_heads
    num_kv_heads_local = text_config.num_key_value_heads
    # Gemma 4 31B uses asymmetric KV heads per attention type: full_attention
    # (global) layers have fewer KV heads (num_global_key_value_heads) than
    # sliding_attention (local) layers, alongside the asymmetric head_dim. E2B/E4B
    # lack the global field, so this falls back to num_key_value_heads (unchanged).
    # The trailing `or` also covers configs that carry the field but leave it
    # None/0, which getattr would otherwise pass through as a bad head count.
    num_kv_heads_global = getattr(
        text_config, "num_global_key_value_heads", num_kv_heads_local
    ) or num_kv_heads_local
    num_kv_heads_per_layer = [
        (num_kv_heads_global if lt == "full_attention" else num_kv_heads_local)
        for lt in stage_layer_types
    ]
    downstream_pli_count = num_total - le if not has_head else 0

    rope_params = getattr(text_config, "rope_parameters", None)
    if rope_params and isinstance(rope_params, dict):
        rp_local = rope_params.get("sliding_attention", {}) or {}
        rp_global = rope_params.get("full_attention", {}) or {}
        rope_theta_local = float(rp_local.get("rope_theta", 10_000.0))
        rope_theta_global = float(rp_global.get("rope_theta", rope_theta_local))
        prf_local = float(rp_local.get("partial_rotary_factor", 1.0))
        prf_global = float(rp_global.get("partial_rotary_factor", 1.0))
    else:
        rope_theta_global = float(
            getattr(text_config, "rope_theta", 1_000_000.0)
        )
        rope_theta_local = float(
            getattr(text_config, "rope_local_base_freq", rope_theta_global)
        )
        prf_local = prf_global = 1.0
    log(
        f"  Rotary: local hd={head_dim_local} theta={rope_theta_local} prf={prf_local} | "
        f"global hd={head_dim_global} theta={rope_theta_global} prf={prf_global}"
    )

    final_softcap = float(
        getattr(text_config, "final_logit_softcapping", 0.0) or 0.0
    )
    if has_head and final_softcap:
        log(f"  Head stage will apply final_logit_softcapping={final_softcap}")

    splice_images = has_embed and vision_ids is not None
    image_token_id, vision_pad_id = vision_ids if splice_images else (None, None)

    class CachedStageWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            if has_embed:
                self.embed = tm.embed_tokens
                # PLI side channel — only present on E2B/E4B (pli_dim>0).
                if pli_dim > 0:
                    self.embed_pli = tm.embed_tokens_per_layer
                    self.pli_proj = tm.per_layer_model_projection
                    self.pli_sv = tm.per_layer_model_projection_scale
                    self.pli_norm = tm.per_layer_projection_norm
                    self.pli_is = tm.per_layer_input_scale
            self.layers = nn.ModuleList(stage_layers)
            self.rotary_local = GemmaTracedRotaryEmbedding(
                head_dim_local,
                rope_theta_local,
                partial_rotary_factor=prf_local,
            )
            self.rotary_global = GemmaTracedRotaryEmbedding(
                head_dim_global,
                rope_theta_global,
                partial_rotary_factor=prf_global,
            )
            if has_head:
                self.norm = tm.norm
                self.lm_head = lm_head_ref
            # Stored attrs for trace fidelity.
            self._head_dims = head_dims
            self._num_kv_heads = num_kv_heads_per_layer
            self._layer_types = list(stage_layer_types)
            self._unique_types = list(set(stage_layer_types))
            self._is_shared = is_shared
            self._source_local = source_local
            self._cross_stage_sources = cross_stage_sources
            self._n_external = len(external_shared_sources)
            self._final_softcap = final_softcap
            # Per-layer window (None = full causal), resolved like head_dims
            # and num_kv_heads: sliding_attention layers get the bound.
            self._layer_windows = [
                sliding_window if lt == SLIDING_LAYER else None
                for lt in stage_layer_types
            ]

        def forward(self, main_input, position_ids, *args):
            # args layout: [*external_shared_kv, *own_past_kv(, image_features)]
            import torch as _t

            image_features = None
            if splice_images:
                image_features, args = args[-1], args[:-1]
            n_ext = self._n_external
            ext_kv = args[: n_ext * 2] if n_ext > 0 else ()
            past_kv = args[n_ext * 2 :]

            if has_embed:
                if splice_images:
                    is_img = main_input == image_token_id
                    main_input = _t.where(
                        is_img, _t.full_like(main_input, vision_pad_id), main_input
                    )
                h = self.embed(main_input)
                if splice_images:
                    h = splice_image_rows(h, is_img, image_features)
                if pli_dim > 0:
                    raw_pli = self.embed_pli(main_input).reshape(
                        main_input.shape[0],
                        main_input.shape[1],
                        num_total,
                        pli_dim,
                    )
                    pj = self.pli_proj(h) * self.pli_sv
                    pj = pj.reshape(h.shape[0], h.shape[1], num_total, pli_dim)
                    pj = self.pli_norm(pj)
                    all_pli = (pj + raw_pli) * self.pli_is
                    stage_pli = all_pli[:, :, ls:le, :]
                    if downstream_pli_count > 0:
                        downstream_pli = all_pli[:, :, le:, :]
                else:
                    stage_pli = None
                    downstream_pli = None
            else:
                if pli_dim > 0:
                    h = main_input[:, :, :hidden_dim]
                    pli_flat = main_input[:, :, hidden_dim:]
                    n_pli = num_layers + downstream_pli_count
                    pli_all = pli_flat.reshape(
                        h.shape[0], h.shape[1], n_pli, pli_dim
                    )
                    stage_pli = pli_all[:, :, :num_layers, :]
                    if downstream_pli_count > 0:
                        downstream_pli = pli_all[:, :, num_layers:, :]
                else:
                    h = main_input
                    stage_pli = None
                    downstream_pli = None

            rotary_cache = {}
            for lt in self._unique_types:
                if lt == "full_attention":
                    rotary_cache[lt] = self.rotary_global(
                        position_ids, h.dtype
                    )
                else:
                    rotary_cache[lt] = self.rotary_local(
                        position_ids, h.dtype
                    )

            # One additive mask per (window, cache source) per forward instead
            # of one per layer: every own-KV layer of a stage sees the same
            # (seq_len, past + seq_len) cache, and a shared layer borrows a
            # full-length source, so the masks are identical across layers of
            # a type. Keyed by the K length's origin so a traced graph keeps a
            # separate ShapeOf chain for external (wire) sources.
            seq_len = h.shape[1]
            mask_cache = {}

            def mask_for(window, k_len, key):
                mk = (window, key)
                if mk not in mask_cache:
                    mask_cache[mk] = build_attention_mask(
                        seq_len, k_len, window, h.device, h.dtype
                    )
                return mask_cache[mk]

            present_kv = []
            present_kv_by_local = {}
            kv_input_idx = 0
            for i, layer in enumerate(self.layers):
                lt = self._layer_types[i]
                hd = self._head_dims[i]
                nkv = self._num_kv_heads[i]
                sw = self._layer_windows[i]
                cos, sin = rotary_cache[lt]
                pli_slice = stage_pli[:, :, i, :] if stage_pli is not None else None

                if self._is_shared[i]:
                    src = self._source_local[i]
                    if src >= 0:
                        sk, sv = present_kv_by_local[src]
                    else:
                        ext_idx = -(src + 1)
                        sk = ext_kv[ext_idx * 2]
                        sv = ext_kv[ext_idx * 2 + 1]
                    mask = mask_for(
                        sw, sk.shape[2], "own" if src >= 0 else "ext"
                    )
                    h = cached_gemma4_shared_layer_forward(
                        layer,
                        h,
                        cos,
                        sin,
                        sk,
                        sv,
                        num_heads,
                        nkv,
                        hd,
                        per_layer_input=pli_slice,
                        sliding_window=sw,
                        causal_mask=mask,
                    )
                else:
                    pk_in = past_kv[kv_input_idx * 2]
                    mask = mask_for(sw, pk_in.shape[2] + seq_len, "own")
                    h, pk, pv = cached_gemma4_layer_forward(
                        layer,
                        h,
                        cos,
                        sin,
                        pk_in,
                        past_kv[kv_input_idx * 2 + 1],
                        num_heads,
                        nkv,
                        hd,
                        per_layer_input=pli_slice,
                        sliding_window=sw,
                        causal_mask=mask,
                    )
                    present_kv.extend([pk, pv])
                    present_kv_by_local[i] = (pk, pv)
                    kv_input_idx += 1

            cross_kv = []
            for src_local in self._cross_stage_sources:
                sk, sv = present_kv_by_local[src_local]
                cross_kv.extend([sk, sv])

            if has_head:
                h = self.norm(h)
                logits = self.lm_head(h)
                if self._final_softcap:
                    sc = self._final_softcap
                    logits = sc * _t.tanh(logits / sc)
                return (logits, *cross_kv, *present_kv)
            else:
                if downstream_pli_count > 0 and downstream_pli is not None:
                    pli_out = downstream_pli.reshape(
                        h.shape[0], h.shape[1], -1
                    )
                    out = _t.cat([h, pli_out], dim=-1)
                else:
                    out = h
                return (out, *cross_kv, *present_kv)

    wrapper = CachedStageWrapper()
    wrapper.eval()
    return wrapper, head_dims, num_kv_heads_per_layer, share


# ---------------------------------------------------------------------------
# Stage export — trace, convert to OV, name I/O, make stateful, save
# ---------------------------------------------------------------------------


def make_stateful_with_init_gemma4(ov_model):
    """Fold gemma4 own-KV (``present.N`` -> ``past_key_values.N``) into stateful
    ReadValue/Assign Variables WITH an explicit zero-length, batch-1 init.

    Unlike export_shards.make_stateful_with_init this maps present<->past by
    NAME (the gemma4 graph interleaves ``cross_kv.*`` outputs and
    ``external_kv.*`` inputs, so index-based folding does not apply), keeps
    every non-``present.*`` result (logits/hidden_states + cross_kv.*) and
    non-``past_key_values.*`` parameter (input_ids/hidden_states, position_ids,
    external_kv.*), and adds NO beam_idx/cache-reorder (gemma4 has none).

    The init matters: ``apply_make_stateful_transformation`` emits init-less
    ReadValues that the OV CPU plugin rejects (#57), and even where they load,
    ``reset_state()`` yields a {0,h,0,d} (batch-0) state whose KV Concat fails
    against the {1,h,seq,d} new K/V. A {1,h,0,d} init makes ``reset_state()``
    work uniformly across stages (same approach as the v5 exporter, #62).
    """
    import openvino as ov
    import openvino.opset13 as ops
    from openvino import Tensor
    from openvino.op.util import Variable, VariableInfo

    # present.N.{key,value} source outputs, keyed by their port name.
    present_src = {}
    keep_results = []
    for r in ov_model.get_results():
        name = next(iter(r.output(0).get_names()), "")
        if name.startswith("present."):
            present_src[name] = r.input_value(0)
        else:
            keep_results.append(r)  # logits / hidden_states + cross_kv.*

    sinks = []
    keep_params = []
    for p in ov_model.get_parameters():
        nm = p.output(0).get_any_name()
        if not nm.startswith("past_key_values."):
            keep_params.append(p)  # input_ids/hidden, position_ids, external_kv.*
            continue
        src = present_src[nm.replace("past_key_values.", "present.", 1)]
        et = p.get_element_type()
        ps = p.get_partial_shape()
        # Canonical [batch, kv_heads, seq, head_dim]: only seq (axis 2) is
        # dynamic; batch defaults to 1, seq to 0 for the zero-length init.
        if len(ps) != 4 or not ps[1].is_static or not ps[3].is_static:
            raise RuntimeError(
                f"{nm}: expected static [batch, kv_heads, seq, head_dim], got {ps}"
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
        p.output(0).replace(rv.output(0))
        if src.get_index() != 0:
            raise RuntimeError(
                f"present source for {nm} is output #{src.get_index()}; "
                "assign needs a single-output node"
            )
        sinks.append(ops.assign(src.get_node(), var))

    new_model = ov.Model(keep_results, sinks, keep_params)
    new_model.validate_nodes_and_infer_types()
    return new_model


def name_stage_ports(ov_model, has_embed, has_head, splice_images, n_external, n_cross):
    """Name a converted stage's ports and free its sequence dims.

    Inputs, in trace order: `input_ids` (embed stage) or `hidden_states`,
    `position_ids`, `external_kv.{i}.key|value` (KV shared from an earlier
    stage), `past_key_values.{i}.key|value`, then, on an embed stage that
    splices images, `image_features` [M, hidden] with M dynamic (0 on a step
    without images). Outputs: `logits` or `hidden_states`, then
    `cross_kv.{i}.key|value` (KV a later stage reads) and
    `present.{i}.key|value`."""
    for i, inp in enumerate(ov_model.inputs):
        shape = inp.partial_shape
        if splice_images and i == len(ov_model.inputs) - 1:
            shape[0] = -1
            inp.node.set_partial_shape(shape)
            inp.set_names({"image_features"})
            continue
        if i == 0:
            name = "input_ids" if has_embed else "hidden_states"
            if len(shape) >= 2:
                shape[1] = -1
        elif i == 1:
            name = "position_ids"
            if len(shape) >= 2:
                shape[1] = -1
        elif i < 2 + n_external * 2:
            ei = (i - 2) // 2
            kt = "value" if (i - 2) % 2 else "key"
            name = f"external_kv.{ei}.{kt}"
            if len(shape) >= 3:
                shape[2] = -1
        else:
            oi = (i - 2 - n_external * 2) // 2
            kt = "value" if (i - 2 - n_external * 2) % 2 else "key"
            name = f"past_key_values.{oi}.{kt}"
            if len(shape) >= 3:
                shape[2] = -1
        inp.node.set_partial_shape(shape)
        inp.set_names({name})

    for i, out in enumerate(ov_model.outputs):
        if i == 0:
            name = "logits" if has_head else "hidden_states"
        elif i < 1 + n_cross * 2:
            ci = (i - 1) // 2
            kt = "value" if (i - 1) % 2 else "key"
            name = f"cross_kv.{ci}.{kt}"
        else:
            oi = (i - 1 - n_cross * 2) // 2
            kt = "value" if (i - 1 - n_cross * 2) % 2 else "key"
            name = f"present.{oi}.{kt}"
        out.set_names({name})


def export_single_stage(
    model,
    text_config,
    stage_plan,
    output_dir,
    quantization,
    device_verify="CPU",
    sliding_window=None,
    vision_ids=None,
):
    """Trace the Gemma 4 wrapper for one stage, convert to OV, make
    KV state stateful, optionally compress, save under
    ``output_dir/stage_N/``."""
    import torch
    import openvino as ov

    idx = stage_plan["stage"]
    ls, le = stage_plan["layer_start"], stage_plan["layer_end"]
    has_embed, has_head = stage_plan["has_embed"], stage_plan["has_head"]
    num_layers = le - ls
    num_total = text_config.num_hidden_layers
    hidden_dim = text_config.hidden_size
    pli_dim = text_config.hidden_size_per_layer_input or 0
    num_kv_heads = text_config.num_key_value_heads
    # `or num_kv_heads` guards a present-but-None/0 global field (see
    # build_cached_wrapper); absent → getattr default already handles E2B/E4B.
    num_kv_heads_global = getattr(
        text_config, "num_global_key_value_heads", num_kv_heads
    ) or num_kv_heads
    downstream_pli_count = num_total - le if not has_head else 0

    log(f"\n{'=' * 60}")
    log(
        f"STAGE {idx}: layers [{ls}, {le}) | embed={has_embed} | head={has_head}"
    )
    log("=" * 60)

    splice_images = has_embed and vision_ids is not None
    wrapper, head_dims, num_kv_heads_per_layer, share = build_cached_wrapper(
        model, text_config, stage_plan, sliding_window=sliding_window,
        vision_ids=vision_ids,
    )
    log("  Wrapper built")

    is_shared = share["is_shared"]
    cross_stage_sources = share["cross_stage_sources"]
    external_shared_sources = share["external_shared_sources"]

    own_kv_head_dims = [hd for hd, sh in zip(head_dims, is_shared) if not sh]
    own_kv_num_heads = [
        nkv for nkv, sh in zip(num_kv_heads_per_layer, is_shared) if not sh
    ]
    non_shared_local_indices = [
        i for i, sh in enumerate(is_shared) if not sh
    ]
    n_own_kv = len(own_kv_head_dims)
    n_cross = len(cross_stage_sources)
    n_external = len(external_shared_sources)
    log(
        f"  KV: {n_own_kv} own (stateful for all), {n_cross} cross-stage "
        f"out, {n_external} external in"
    )

    # Example inputs for trace.
    seq_len = 4
    past_seq = 1

    if has_embed:
        main_input = torch.randint(0, text_config.vocab_size, (1, seq_len))
        if splice_images:
            # Two image tokens so the trace exercises the gather.
            main_input[0, 1:3] = vision_ids[0]
    else:
        if pli_dim > 0:
            input_dim = hidden_dim + (num_layers + downstream_pli_count) * pli_dim
        else:
            input_dim = hidden_dim
        main_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    ext_kv = []
    for _, lt, hd in external_shared_sources:
        nkv = num_kv_heads_global if lt == "full_attention" else num_kv_heads
        ext_kv.append(torch.randn(1, nkv, past_seq, hd))
        ext_kv.append(torch.randn(1, nkv, past_seq, hd))

    own_past_kv = []
    for nkv, hd in zip(own_kv_num_heads, own_kv_head_dims):
        own_past_kv.append(torch.randn(1, nkv, past_seq, hd))
        own_past_kv.append(torch.randn(1, nkv, past_seq, hd))

    example_inputs = (main_input, position_ids, *ext_kv, *own_past_kv)
    if splice_images:
        example_inputs += (torch.randn(2, hidden_dim, dtype=torch.float32),)

    log("  Tracing...")
    t_trace = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
    log(f"  Trace OK ({time.time() - t_trace:.0f}s)")
    del wrapper
    gc.collect()

    log("  ov.convert_model...")
    t_conv = time.time()
    ov_model = ov.convert_model(traced, example_input=example_inputs)
    del traced
    gc.collect()
    log(
        f"  Converted ({time.time() - t_conv:.0f}s); "
        f"{len(ov_model.inputs)} inputs, {len(ov_model.outputs)} outputs"
    )

    name_stage_ports(ov_model, has_embed, has_head, splice_images, n_external, n_cross)
    ov_model.validate_nodes_and_infer_types()

    # Make own KV stateful WITH explicit batch-1 zero-length inits (cross_kv
    # stays a regular output; external_kv stays a regular input). Init-bearing
    # ReadValues are required so the OV CPU plugin loads the shard AND so
    # reset_state() yields a {1,h,0,d} state at the runtime (an init-less
    # ReadValue gives {0,h,0,d}, whose KV Concat fails against the new K/V).
    log(f"  make_stateful_with_init ({n_own_kv} own-KV pairs)...")
    if n_own_kv > 0:
        ov_model = make_stateful_with_init_gemma4(ov_model)
    ov_model.validate_nodes_and_infer_types()
    log(
        f"  Stateful: {len(ov_model.inputs)} inputs, "
        f"{len(ov_model.outputs)} outputs"
    )

    # Optional INT4/INT8 compression.
    if quantization in ("int4", "int4_asym"):
        try:
            import nncf

            mode = (
                nncf.CompressWeightsMode.INT4_SYM
                if quantization == "int4"
                else nncf.CompressWeightsMode.INT4_ASYM
            )
            log(f"  nncf {quantization} compression...")
            ov_model = nncf.compress_weights(
                ov_model, mode=mode, group_size=128, ratio=1.0, all_layers=True
            )
        except Exception as e:
            log(
                f"  WARNING: INT4 compression failed ({e}); leaving FP32. "
                "Re-run with --quantization fp32 to skip the attempt."
            )
    elif quantization == "int8":
        try:
            import nncf

            log("  nncf int8 compression...")
            ov_model = nncf.compress_weights(
                ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM
            )
        except Exception as e:
            log(f"  WARNING: INT8 compression failed ({e}); leaving FP32.")

    # Save.
    stage_dir = os.path.join(output_dir, f"stage_{idx}")
    os.makedirs(stage_dir, exist_ok=True)
    xml_path = os.path.join(stage_dir, "openvino_model.xml")
    log("  Saving...")
    ov.save_model(ov_model, xml_path)
    bin_path = os.path.join(stage_dir, "openvino_model.bin")
    size_mb = (
        os.path.getsize(bin_path) / (1024**2) if os.path.exists(bin_path) else 0
    )
    log(f"  Saved: {stage_dir} ({size_mb:.0f} MB)")

    # Stage metadata — extends the v5 schema with Gemma 4 fields.
    final_softcap = float(
        getattr(text_config, "final_logit_softcapping", 0.0) or 0.0
    )
    meta = {
        "stage": idx,
        "layer_start": ls,
        "layer_end": le,
        "has_embed": has_embed,
        "has_head": has_head,
        "quantization": quantization,
        "num_layers_total": num_total,
        "hidden_size": hidden_dim,
        "vocab_size": text_config.vocab_size,
        "num_attention_heads": text_config.num_attention_heads,
        "num_kv_heads": num_kv_heads,
        # Per-layer (Gemma 4-specific) — list length == num_layers_in_stage.
        "head_dims": head_dims,
        "layer_types": list(text_config.layer_types[ls:le]),
        # Window compiled into this stage's sliding_attention layers (None =
        # no sliding layers). Exports before gemma4_cached_v1.1 lack this key
        # and let those layers attend past the window; the gemma4 engine
        # warns at load when a sliding layer has no window.
        "sliding_window": sliding_window,
        "is_shared": is_shared,
        "own_kv_head_dims": own_kv_head_dims,
        "cross_stage_sources_local": cross_stage_sources,
        "external_shared_sources": [
            {"src_global_layer": sg, "layer_type": lt, "head_dim": hd}
            for sg, lt, hd in external_shared_sources
        ],
        "pli_dim": pli_dim,
        "downstream_pli_count": downstream_pli_count,
        "final_logit_softcapping": final_softcap,
        "arch_tag": "gemma4",
        "stateful": True,
        "export_version": EXPORT_VERSION,
        "inputs": (
            "input_ids/hidden_states+downstream_pli, position_ids, "
            "external_kv.*.key/value (n=n_external_kv); KV cache is "
            "internal state"
        ),
    }
    with open(os.path.join(stage_dir, "stage_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Self-verify on CPU (fast). A failure means the stage on disk is broken
    # — a non-finite output is a mis-built mask — so it aborts the export
    # instead of leaving a v1.1-stamped NaN shard behind an exit code of 0.
    if device_verify:
        try:
            _verify_stage(
                ov_model,
                stage_idx=idx,
                has_embed=has_embed,
                num_kv_heads=num_kv_heads,
                pli_dim=pli_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                downstream_pli_count=downstream_pli_count,
                external_shared_sources=external_shared_sources,
                device=device_verify,
                sliding_window=sliding_window,
                splice_images=splice_images,
            )
        except Exception as e:
            log(f"  Self-verify FAILED for stage {idx} ({str(e)[:200]})")
            raise

    del ov_model
    gc.collect()
    return size_mb


def _verify_stage(
    ov_model,
    stage_idx,
    has_embed,
    num_kv_heads,
    pli_dim,
    hidden_dim,
    num_layers,
    downstream_pli_count,
    external_shared_sources,
    device,
    sliding_window=None,
    splice_images=False,
):
    """Compile the stage and run a prefill + one decode step.

    With a window, the prefill spans `sliding_window + 4` tokens so the
    exported `tril` band (whose diagonal must stay dynamic through
    torch.jit.trace -> ov.convert_model) is actually exercised — a 3-token
    prefill never reaches it. Every output must be finite: a broken band
    masks whole rows and surfaces as NaN, and the RuntimeError raised for
    that propagates out of `export_single_stage` and fails the export —
    a NaN stage must not stay on disk under a v1.1 stamp.
    """
    import numpy as np
    import openvino as ov

    log(f"  Self-verify on {device}...")
    core = ov.Core()
    comp = core.compile_model(ov_model, device)
    req = comp.create_infer_request()
    req.reset_state()
    for sv in req.query_state():
        shape = list(sv.state.shape)
        shape[0] = 1
        shape[2] = 0
        sv.state = ov.Tensor(np.zeros(shape, dtype=np.float32))

    pf_seq = (sliding_window + 4) if sliding_window else 3
    inputs = {}
    inp_idx = 0
    if pli_dim > 0:
        input_dim = hidden_dim + (num_layers + downstream_pli_count) * pli_dim
    else:
        input_dim = hidden_dim
    if has_embed:
        inputs[inp_idx] = (np.arange(pf_seq, dtype=np.int64) % 1000 + 1)[None, :]
    else:
        inputs[inp_idx] = np.random.randn(1, pf_seq, input_dim).astype(np.float32)
    inp_idx += 1
    inputs[inp_idx] = np.arange(pf_seq, dtype=np.int64)[None, :]
    inp_idx += 1
    for _, _, hd in external_shared_sources:
        inputs[inp_idx] = np.zeros(
            (1, num_kv_heads, pf_seq, hd), dtype=np.float32
        )
        inp_idx += 1
        inputs[inp_idx] = np.zeros(
            (1, num_kv_heads, pf_seq, hd), dtype=np.float32
        )
        inp_idx += 1
    if splice_images:
        # No image rows: exercises the port and the appended zero row that
        # every text-only step (and every decode step) relies on.
        inputs[comp.input("image_features")] = np.zeros((0, hidden_dim), dtype=np.float32)
    result = req.infer(inputs)
    out = result[comp.output(0)]
    if not np.isfinite(out).all():
        raise RuntimeError(
            f"non-finite prefill output at {pf_seq} tokens "
            f"(sliding_window={sliding_window}): masked-out rows?"
        )
    log(
        f"    Prefill OK: shape={out.shape} ({pf_seq} tokens"
        + (f", crosses window {sliding_window}" if sliding_window else "")
        + ")"
    )

    # Decode step (1 token).
    inputs2 = {}
    inp_idx = 0
    if has_embed:
        inputs2[inp_idx] = np.array([[4]], dtype=np.int64)
    else:
        inputs2[inp_idx] = np.random.randn(1, 1, input_dim).astype(np.float32)
    inp_idx += 1
    inputs2[inp_idx] = np.array([[pf_seq]], dtype=np.int64)
    inp_idx += 1
    for _, _, hd in external_shared_sources:
        inputs2[inp_idx] = np.zeros(
            (1, num_kv_heads, pf_seq + 1, hd), dtype=np.float32
        )
        inp_idx += 1
        inputs2[inp_idx] = np.zeros(
            (1, num_kv_heads, pf_seq + 1, hd), dtype=np.float32
        )
        inp_idx += 1
    if splice_images:
        inputs2[comp.input("image_features")] = np.zeros((0, hidden_dim), dtype=np.float32)
    result2 = req.infer(inputs2)
    out2 = result2[comp.output(0)]
    if not np.isfinite(out2).all():
        raise RuntimeError("non-finite decode output after the windowed prefill")
    log(f"    Decode OK: shape={out2.shape}")


# ---------------------------------------------------------------------------
# Top-level entry — used both standalone and from export_shards.main()
# ---------------------------------------------------------------------------


def run_export(
    model_id: str,
    output_dir: str,
    num_stages: int,
    quantization: str = "fp16",
    stage: int | None = None,
    device_verify: str | None = None,
    vision_dir: str | None = None,
):
    """Export a Gemma 4 model end-to-end.

    `vision_dir` (``--vision``): a directory holding the model's OpenVINO
    vision encoder (`openvino_vision_embeddings_model.{xml,bin}`, e.g. from
    OpenVINO/gemma-4-E2B-it-fp16-ov) and its processor configs. Stage 0
    then takes `image_features` (see build_cached_wrapper) and the tree
    carries the encoder for rank 0.

    Caller is responsible for ensuring `model_id` resolves to a local
    snapshot with `config.json` and `*.safetensors`. We use HF's
    `AutoModelForCausalLM` with `trust_remote_code=True` so older
    `transformers` (< 5.5) still load Gemma 4 via the model repo's
    bundled Python code.

    In the other direction, from `transformers` 5.14 a heterogeneous config
    (31B's `head_dim=256` / `global_head_dim=512`) registers those as
    per-layer attributes, so the exporter's global `text_config.head_dim`
    reads raise `AmbiguousGlobalPerLayerAttributeError` unless the config
    sets `allow_global_per_layer_attribute_access` or the bundled remote
    code is used.
    """
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    # The generic export_shards.main() sets torch default dtype to
    # float16 for memory; that propagates into our example_inputs and
    # into newly-allocated buffers inside the model, then mismatches
    # the model weights (which we load explicitly as fp32) at the
    # make_stateful step ("Input type: f32 Variable type: f16"). Reset
    # to float32 for the entire Gemma 4 export — the per-stage IRs are
    # saved fp16 either way after `ov.save_model(..., compress_to_fp16)`.
    torch.set_default_dtype(torch.float32)
    apply_patches()

    log("Loading config...")
    full_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_config = getattr(full_config, "text_config", None) or full_config
    if not hasattr(text_config, "layer_types"):
        raise RuntimeError(
            f"Gemma 4 text_config missing 'layer_types' "
            f"(model_id={model_id!r}). Check trust_remote_code support."
        )
    if getattr(text_config, "enable_moe_block", False):
        raise RuntimeError(
            "Gemma 4 MoE variant (26B-A4B) is not supported — MoE "
            "block routing requires a separate exporter. See "
            "docs/architectures/moe.md."
        )
    unknown_types = sorted(set(text_config.layer_types) - KNOWN_LAYER_TYPES)
    if unknown_types:
        raise RuntimeError(
            f"Gemma 4 layer_types {unknown_types} are not supported; the "
            f"exporter knows {sorted(KNOWN_LAYER_TYPES)} (HF would fail on "
            "these too)."
        )
    # Every stage slices layer_types by layer index, so without this a short
    # list raised IndexError minutes into the export and a long one dropped
    # its tail unnoticed.
    if len(text_config.layer_types) != text_config.num_hidden_layers:
        raise RuntimeError(
            f"Gemma 4 layer_types has {len(text_config.layer_types)} entries "
            f"for {text_config.num_hidden_layers} layers; they must match."
        )
    if getattr(text_config, "use_bidirectional_attention", None) == "all" or (
        getattr(text_config, "is_causal", True) is False
    ):
        raise RuntimeError(
            "Gemma 4 bidirectional-attention (embedding) variants are not "
            "supported: the exporter builds causal masks, and HF rewrites "
            "`sliding_window` to `window // 2 + 1` for that mode."
        )
    # Config-first: fails before the multi-minute model load, and a config
    # with sliding layers but no window is rejected instead of silently
    # exporting the pre-v1.1 unmasked behaviour.
    sliding_window = resolve_sliding_window(text_config, text_config.layer_types)

    # A --stage re-export touches one stage but rewrites pipeline_config.json,
    # so re-exporting into a tree written by another version leaves the root
    # claiming this version while the untouched stages keep the old behaviour
    # (v1: sliding layers with no window). A full export rewrites everything,
    # so it may still overwrite.
    prev_pipeline = os.path.join(output_dir, "pipeline_config.json")
    if stage is not None and os.path.exists(prev_pipeline):
        with open(prev_pipeline) as f:
            prev_version = json.load(f).get("export_version")
        if prev_version != EXPORT_VERSION:
            raise RuntimeError(
                f"{prev_pipeline} was written by export_version "
                f"{prev_version!r}, but this exporter writes "
                f"{EXPORT_VERSION!r}; re-export every stage (drop --stage) "
                "rather than mixing versions in one tree."
            )

    log(
        f"  {text_config.num_hidden_layers} layers, "
        f"hidden={text_config.hidden_size}, "
        f"heads={text_config.num_attention_heads}, "
        f"kv_heads={text_config.num_key_value_heads}, "
        f"head_dim={text_config.head_dim}, "
        f"global_head_dim={getattr(text_config, 'global_head_dim', 'n/a')}, "
        f"pli_dim={text_config.hidden_size_per_layer_input or 0}, "
        f"num_kv_shared={text_config.num_kv_shared_layers or 0}, "
        f"softcap={getattr(text_config, 'final_logit_softcapping', None)}, "
        f"sliding_window={sliding_window}"
    )

    plan = compute_stage_plan(text_config.num_hidden_layers, num_stages)
    if stage is not None:
        # The untouched stages must also share this plan's layer split, or
        # the tree skips or repeats layers.
        _here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_here, "gemma4_surgery"))
        sys.path.insert(0, _here)
        from export_gemma4_text import check_reused_tree

        check_reused_tree(
            output_dir, num_stages, text_config.num_hidden_layers, EXPORT_VERSION,
            [(s["layer_start"], s["layer_end"] - 1) for s in plan],
            text_config.hidden_size,
        )
    log(f"\nStage plan ({num_stages} stages):")
    for s in plan:
        log(
            f"  Stage {s['stage']}: layers "
            f"[{s['layer_start']}, {s['layer_end']})"
            + (" + embed" if s["has_embed"] else "")
            + (" + norm/head" if s["has_head"] else "")
        )

    log("\nLoading model (float32)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    fixed = fix_zero_dim_buffers(model)
    log(f"  Loaded + patched {fixed} scalar buffers in {time.time() - t0:.0f}s")

    os.makedirs(output_dir, exist_ok=True)

    # Tokenizer.
    tok_dir = os.path.join(output_dir, "tokenizer")
    if not os.path.exists(tok_dir):
        os.makedirs(tok_dir, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tok.save_pretrained(tok_dir)
        tc_path = os.path.join(tok_dir, "tokenizer_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as ff:
                tc_json = json.load(ff)
            if isinstance(tc_json.get("extra_special_tokens"), list):
                tc_json["extra_special_tokens"] = {}
                with open(tc_path, "w") as ff:
                    json.dump(tc_json, ff, indent=2)
        log(f"  Tokenizer -> {tok_dir}")

    # Copy text config so the Rust runtime can load Rotary params.
    cfg_dst = os.path.join(output_dir, "tokenizer", "config.json")
    if not os.path.exists(cfg_dst):
        try:
            full_config.to_json_file(cfg_dst)
        except Exception as e:
            log(f"  warning: could not write {cfg_dst}: {e}")

    # Pipeline metadata.
    pipeline_meta = {
        "model_id": model_id,
        "num_stages": num_stages,
        "num_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "vocab_size": text_config.vocab_size,
        "head_dim": text_config.head_dim,
        "global_head_dim": getattr(text_config, "global_head_dim", None),
        "pli_dim": text_config.hidden_size_per_layer_input or 0,
        "num_kv_shared_layers": text_config.num_kv_shared_layers or 0,
        "layer_types": list(text_config.layer_types),
        # See the stage_config.json comment in export_single_stage.
        "sliding_window": sliding_window,
        "final_logit_softcapping": float(
            getattr(text_config, "final_logit_softcapping", 0.0) or 0.0
        ),
        "rope_parameters": getattr(text_config, "rope_parameters", None),
        "arch_tag": "gemma4",
        "quantization": quantization,
        "export_version": EXPORT_VERSION,
    }
    vision_ids = None
    if vision_dir:
        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
            raise SystemExit(
                "--vision: this Gemma 4 attends bidirectionally inside images "
                "(use_bidirectional_attention='vision', 26B-A4B/31B), which the "
                "gemma4 engine's causal mask does not model. Serve its images "
                "with tools/gemma4_surgery/export_gemma4_text.py --vision on "
                "--engine ov-runtime."
            )
        ids = {}
        for k in ("image_token_id", "boi_token_id", "eoi_token_id"):
            ids[k] = getattr(full_config, k, None)
        ids["pad_token_id"] = getattr(text_config, "pad_token_id", None)
        missing = [k for k, v in ids.items() if v is None]
        if missing:
            raise SystemExit(f"--vision: config lacks {missing}")
        for fn in VISION_FILES:
            src = os.path.join(vision_dir, fn)
            if not os.path.exists(src):
                raise SystemExit(f"--vision needs {fn} in {vision_dir}")
            shutil.copy2(src, os.path.join(output_dir, fn))
        vision_ids = (ids["image_token_id"], ids["pad_token_id"])
        pipeline_meta["vision"] = {
            "family": "gemma4",
            "vision_embeddings": "openvino_vision_embeddings_model.xml",
            "stage0_input": "image_features",
            **ids,
        }
        log(f"  vision: stage 0 splices image_features; ids {ids}")
    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump(pipeline_meta, f, indent=2)

    total_mb = 0.0
    for sp in plan:
        if stage is not None and sp["stage"] != stage:
            continue
        size = export_single_stage(
            model, text_config, sp, output_dir, quantization,
            device_verify=device_verify,
            sliding_window=sliding_window,
            vision_ids=vision_ids,
        )
        total_mb += size
        gc.collect()

    log("\n" + "=" * 60)
    log("GEMMA 4 EXPORT COMPLETE")
    log(f"  Output: {output_dir}")
    log(f"  Total: {total_mb:.0f} MB")
    log("=" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        required=True,
        help="HF repo id (e.g. google/gemma-4-E2B-it) or local snapshot dir.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-stages", type=int, default=2)
    p.add_argument(
        "--quantization",
        default="fp16",
        choices=["fp16", "fp32", "int4", "int4_asym", "int8"],
        help=(
            "Weight precision. fp16 is the default for Gemma 4 because "
            "INT4 compression sometimes fails on the per-layer scalar "
            "buffers and the per-layer-projection norms; if it works "
            "for your variant, prefer int4 for size."
        ),
    )
    p.add_argument(
        "--stage", type=int, default=None, help="Export only this stage."
    )
    p.add_argument(
        "--verify-device",
        default="CPU",
        help="OV device to self-verify each stage on. 'none' to skip.",
    )
    p.add_argument(
        "--vision",
        default=None,
        metavar="DIR",
        help=(
            "Keep image input: DIR holds the model's OpenVINO vision encoder "
            "(openvino_vision_embeddings_model.xml/.bin) and processor configs, "
            "e.g. a download of OpenVINO/gemma-4-E2B-it-fp16-ov. E2B/E4B only."
        ),
    )
    args = p.parse_args()

    device_verify = None if args.verify_device.lower() == "none" else args.verify_device
    run_export(
        model_id=args.model,
        output_dir=args.output_dir,
        num_stages=args.num_stages,
        quantization=args.quantization,
        stage=args.stage,
        device_verify=device_verify,
        vision_dir=args.vision,
    )


if __name__ == "__main__":
    main()
