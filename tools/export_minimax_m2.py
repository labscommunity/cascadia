#!/usr/bin/env python3
"""Export MiniMax-M2 into Cascadia's sparse-MoE OV-IR layout.

MiniMax-M2 is a 62-layer, all-MoE transformer: standard GQA attention
(48 q-heads / 8 kv-heads, head_dim 128) with full-width QK-norm and
*partial* rotary (first 64 of 128 dims), top-8 routing over 256 experts
with **sigmoid** scoring + an additive `e_score_correction_bias` used for
selection only, and a SwiGLU expert FFN (w1=gate, w3=up, w2=down,
intermediate 1536). No shared expert, no dense layers, no MTP weights in
the published checkpoint. Native weights are block-FP8 (e4m3, [128,128]
`weight_scale_inv`); we dequant to fp32 then re-quantize to INT4 via NNCF.

Unlike Kimi K2.6 (MLA, run by a hardcoded Rust kernel), every M2
architecture detail lives in the *traced OpenVINO graph* here, so the
Cascadia runtime stays architecture-agnostic and correctness reduces to
"does OV faithfully run what PyTorch traced" — which we check with the
companion harness `test_minimax_m2_pipeline.py` against the canonical HF
model.

Output layout (consumed by crates/cascadia-engine-sparse-moe):

    <out>/manifest.json
    <out>/layer0/openvino_model.{xml,bin}              # embed_tokens (ids -> hidden)
    <out>/head/openvino_model.{xml,bin}                # final RMSNorm + lm_head
    <out>/shells/layer_NN/openvino_model.{xml,bin}     # attn + routing, 7-tensor contract
    <out>/experts/layer_NN/expert_EEE/openvino_model.{xml,bin}

Shell I/O contract (decode, seq=1):
    in:  x[1,1,H] bf16, past_k[1,KV,P,D] f32, past_v[1,KV,P,D] f32, past_seq_len i64 scalar
    out: attn_out_post_norm[1,1,H], attn_residual[1,1,H], shared_expert_out[1,1,H](zeros),
         routing_ids[K] i64, routing_weights[K] f32, present_k[1,KV,1,D], present_v[1,KV,1,D]

Usage:
    # tiny synthetic model for the correctness test (no download needed):
    python export_minimax_m2.py --tiny --out /tmp/m2_tiny

    # full model from a local FP8 checkout:
    python export_minimax_m2.py --model /path/to/MiniMax-M2 --out /path/to/export \
        [--layers 0-3] [--experts-int8] [--no-quant]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# OpenVINO 2026.x: `openvino.runtime` was removed; opsets live under `openvino`.
import openvino as ov
from openvino import opset15 as opset

try:
    import nncf
except Exception:  # nncf optional for --no-quant / tiny debugging
    nncf = None


# --------------------------------------------------------------------------
# FP8 block dequant (DeepSeek-V3 / vLLM convention, [128,128] blocks)
# --------------------------------------------------------------------------
def dequant_fp8_block(weight_fp8: torch.Tensor, scale_inv: torch.Tensor,
                      block=(128, 128)) -> torch.Tensor:
    """W_f32[i,j] = float(W_fp8[i,j]) * scale_inv[i//128, j//128]."""
    w = weight_fp8.to(torch.float32)
    out, inn = w.shape
    bo, bi = block
    # Expand the per-block scale up to the full weight shape.
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(bo, dim=0)[:out].repeat_interleave(bi, dim=1)[:, :inn]
    return w * s


def materialize_linear(mod: nn.Module) -> nn.Linear:
    """Return an fp32 nn.Linear for `mod`, dequantizing block-FP8 if needed.

    HF stores FP8 linears as the raw e4m3 `weight` plus a `weight_scale_inv`
    buffer. A plain bf16/fp32 Linear has no scale and is returned as-is
    (cast to fp32)."""
    weight = mod.weight
    out_f, in_f = weight.shape
    bias = getattr(mod, "bias", None)
    scale_inv = getattr(mod, "weight_scale_inv", None)
    if scale_inv is None:
        # also tolerate a stashed attribute name some loaders use
        scale_inv = getattr(mod, "weight_scale", None)
    if scale_inv is not None and weight.dtype in (torch.float8_e4m3fn,):
        w = dequant_fp8_block(weight, scale_inv)
    else:
        w = weight.to(torch.float32)
    lin = nn.Linear(in_f, out_f, bias=bias is not None)
    with torch.no_grad():
        lin.weight.copy_(w)
        if bias is not None:
            lin.bias.copy_(bias.to(torch.float32))
    return lin


def rms_weight(mod: nn.Module) -> torch.Tensor:
    return mod.weight.detach().to(torch.float32)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x.to(dt)).to(dt)


# --------------------------------------------------------------------------
# Rotary (partial, NeoX rotate_half), matching HF default rope
# --------------------------------------------------------------------------
def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class PartialRotary(nn.Module):
    """cos/sin for a single absolute position `pos` (scalar i64 tensor).

    rotary_dim of the head_dim is rotated; the rest passes through. inv_freq
    has rotary_dim/2 entries. attention_scaling is 1.0 for default rope."""

    def __init__(self, rotary_dim: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.rotary_dim = rotary_dim

    def forward(self, pos):  # pos: scalar i64 tensor (current token position)
        p = pos.to(torch.float32).reshape(1)             # [1]
        freqs = p[:, None] * self.inv_freq[None, :]      # [1, rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)          # [1, rotary_dim]
        return emb.cos(), emb.sin()                      # each [1, rotary_dim]

    def apply(self, q, k, cos, sin):
        # q: [1, Hq, 1, D], k: [1, Hkv, 1, D]; rotate first rotary_dim dims.
        rd = self.rotary_dim
        cos = cos[None, None, :, :]                      # [1,1,1,rd]
        sin = sin[None, None, :, :]
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        k_rot = k_rot * cos + rotate_half(k_rot) * sin
        return torch.cat((q_rot, q_pass), -1), torch.cat((k_rot, k_pass), -1)


# --------------------------------------------------------------------------
# Traceable wrappers
# --------------------------------------------------------------------------
class ShellWrapper(nn.Module):
    """One M2 decoder layer minus the routed-expert combine, as the
    stateless 7-tensor shell. Reuses the reference layer's leaf modules
    (Linears / RMSNorms / gate / e_score_correction_bias); reimplements
    the attention + partial-rope + sigmoid-routing glue with explicit KV
    in/out so it traces cleanly to OV."""

    def __init__(self, layer, cfg):
        super().__init__()
        attn = layer.self_attn
        self.q_proj = materialize_linear(attn.q_proj)
        self.k_proj = materialize_linear(attn.k_proj)
        self.v_proj = materialize_linear(attn.v_proj)
        self.o_proj = materialize_linear(attn.o_proj)
        self.q_norm_w = rms_weight(attn.q_norm)
        self.k_norm_w = rms_weight(attn.k_norm)
        self.in_ln_w = rms_weight(layer.input_layernorm)
        self.post_ln_w = rms_weight(layer.post_attention_layernorm)
        # In transformers the MoE block is `layer.mlp` (the checkpoint's
        # `block_sparse_moe.*` keys are remapped on load). The router
        # `gate` is a MiniMaxM2TopKRouter whose `.weight` [E,H] is applied
        # as F.linear (== a bias-free Linear); e_score_correction_bias is a
        # buffer on the block, added to scores for *selection* only.
        moe = layer.mlp
        self.gate = materialize_linear(moe.gate)          # F32 [E, H]
        self.e_bias = moe.e_score_correction_bias.detach().to(torch.float32)
        self.rotary = PartialRotary(cfg["partial_rotary_dim"], cfg["rope_theta"])

        self.H = cfg["hidden_size"]
        self.Hq = cfg["num_q_heads"]
        self.Hkv = cfg["num_kv_heads"]
        self.D = cfg["head_dim"]
        self.K = cfg["top_k"]
        self.eps = cfg["rms_norm_eps"]
        self.scale = self.D ** -0.5
        self.groups = self.Hq // self.Hkv

    def forward(self, x, past_k, past_v, past_seq_len):
        # x: [1,1,H]
        residual = x
        h = rmsnorm(x, self.in_ln_w, self.eps)
        q = self.q_proj(h)                                # [1,1,Hq*D]
        k = self.k_proj(h)                                # [1,1,Hkv*D]
        v = self.v_proj(h)
        q = rmsnorm(q, self.q_norm_w, self.eps)           # full-width QK-norm
        k = rmsnorm(k, self.k_norm_w, self.eps)
        q = q.view(1, 1, self.Hq, self.D).transpose(1, 2)   # [1,Hq,1,D]
        k = k.view(1, 1, self.Hkv, self.D).transpose(1, 2)  # [1,Hkv,1,D]
        v = v.view(1, 1, self.Hkv, self.D).transpose(1, 2)

        cos, sin = self.rotary(past_seq_len)
        q, k = self.rotary.apply(q, k, cos, sin)

        present_k, present_v = k, v                        # this token only
        k_full = torch.cat((past_k, k), dim=2)             # [1,Hkv,P+1,D]
        v_full = torch.cat((past_v, v), dim=2)

        # GQA expand
        kf = k_full.repeat_interleave(self.groups, dim=1)  # [1,Hq,P+1,D]
        vf = v_full.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(q, kf.transpose(2, 3)) * self.scale  # [1,Hq,1,P+1]
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
        attn = torch.matmul(probs, vf)                     # [1,Hq,1,D]
        attn = attn.transpose(1, 2).reshape(1, 1, self.Hq * self.D)
        attn = self.o_proj(attn)                           # [1,1,H]

        attn_residual = residual + attn
        attn_out_post_norm = rmsnorm(attn_residual, self.post_ln_w, self.eps)

        # sigmoid routing: bias added for selection only; weights = gathered
        # raw sigmoid, renormalized to sum 1. No routed_scaling_factor.
        logits = self.gate(attn_out_post_norm.to(torch.float32)).reshape(-1)  # [E]
        rw = torch.sigmoid(logits)
        sel = rw + self.e_bias
        _, ids = torch.topk(sel, self.K, dim=-1, sorted=False)
        w = rw.gather(-1, ids)
        w = w / w.sum(-1, keepdim=True)

        shared = torch.zeros_like(attn_residual)
        return (attn_out_post_norm, attn_residual, shared,
                ids.to(torch.int64), w.to(torch.float32), present_k, present_v)


def _maybe_dequant_3d(slice_w, scale_inv):
    """fp32 view of a (possibly block-FP8) 2D weight slice from a 3D param."""
    if scale_inv is not None and slice_w.dtype == torch.float8_e4m3fn:
        return dequant_fp8_block(slice_w, scale_inv)
    return slice_w.to(torch.float32)


class ExpertWrapper(nn.Module):
    """One expert's SwiGLU FFN, sliced from the fused 3D expert params.

    transformers stores all experts of a layer as two 3D tensors:
      experts.gate_up_proj : [E, 2*inter, hidden]  (gate = first half, up = second)
      experts.down_proj    : [E, hidden, inter]
    Forward (matches MiniMaxM2Experts): down( silu(gate(x)) * up(x) )."""

    def __init__(self, experts, ei):
        super().__init__()
        gup = experts.gate_up_proj[ei]                 # [2*inter, hidden]
        dwn = experts.down_proj[ei]                    # [hidden, inter]
        gup_s = getattr(experts, "gate_up_proj_scale_inv", None)
        dwn_s = getattr(experts, "down_proj_scale_inv", None)
        gup = _maybe_dequant_3d(gup, gup_s[ei] if gup_s is not None else None)
        dwn = _maybe_dequant_3d(dwn, dwn_s[ei] if dwn_s is not None else None)
        inter = dwn.shape[1]
        H = dwn.shape[0]
        self.gate = nn.Linear(H, inter, bias=False)
        self.up = nn.Linear(H, inter, bias=False)
        self.down = nn.Linear(inter, H, bias=False)
        with torch.no_grad():
            self.gate.weight.copy_(gup[:inter])
            self.up.weight.copy_(gup[inter:])
            self.down.weight.copy_(dwn)

    def forward(self, x):                          # x: [1,1,H]
        return self.down(F.silu(self.gate(x)) * self.up(x))


class EmbedWrapper(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = nn.Embedding(embed.num_embeddings, embed.embedding_dim)
        with torch.no_grad():
            self.embed.weight.copy_(embed.weight.to(torch.float32))

    def forward(self, ids):                        # ids: [1,1] i64
        return self.embed(ids)


class HeadWrapper(nn.Module):
    def __init__(self, norm, lm_head, eps):
        super().__init__()
        self.norm_w = rms_weight(norm)
        self.lm_head = materialize_linear(lm_head)
        self.eps = eps

    def forward(self, h):                          # h: [1,1,H]
        return self.lm_head(rmsnorm(h, self.norm_w, self.eps))


# --------------------------------------------------------------------------
# OV conversion / quantization helpers
# --------------------------------------------------------------------------
def convert_and_save(wrapper, example_inputs, out_xml: Path, dyn_axes,
                     in_names, out_names, quant_mode="int4", group_size=128):
    """Trace -> OV -> set dynamic axes / names -> NNCF compress -> save.

    Resumable: if a non-empty IR already exists at `out_xml`, skip it. This
    lets a long full-model export (15k+ expert IRs) restart where it left
    off after an interruption."""
    if out_xml.exists() and out_xml.stat().st_size > 0:
        bin_path = out_xml.with_suffix(".bin")
        if bin_path.exists():
            return
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
    m = ov.convert_model(traced, example_input=example_inputs)

    # name + dynamic axes for each input
    for i, inp in enumerate(m.inputs):
        if i < len(in_names):
            inp.get_tensor().set_names({in_names[i]})
        ps = inp.partial_shape
        for ax in dyn_axes.get(i, []):
            if ax < len(ps):
                ps[ax] = -1
        inp.node.set_partial_shape(ps)
    m.validate_nodes_and_infer_types()
    for i, o in enumerate(m.outputs):
        if i < len(out_names):
            o.get_tensor().set_names({out_names[i]})

    if quant_mode and quant_mode != "none" and nncf is not None:
        if quant_mode == "int4":
            m = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT4_SYM,
                                      group_size=group_size, ratio=1.0)
        elif quant_mode == "int8":
            m = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_SYM)

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(m, str(out_xml), compress_to_fp16=False)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def tiny_config():
    """Small but architecturally-faithful M2 config for the correctness test."""
    from transformers import MiniMaxM2Config
    return MiniMaxM2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,          # expert FFN dim
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_local_experts=8,
        num_experts_per_tok=2,
        rotary_dim=16,                 # partial: half of head_dim
        rope_theta=5000000.0,
        rms_norm_eps=1e-6,
        scoring_func="sigmoid",
        use_qk_norm=True,
        qk_norm_type="per_layer",
        shared_intermediate_size=0,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        use_mtp=False,
    )


def load_reference(args):
    """Return (model, cfg_dict). model is an HF MiniMaxM2ForCausalLM in fp32."""
    from transformers import AutoModelForCausalLM
    if args.tiny:
        from transformers import MiniMaxM2ForCausalLM
        torch.manual_seed(0)
        cfg = tiny_config()
        model = MiniMaxM2ForCausalLM(cfg).to(torch.float32).eval()
        hf = model.config
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.float32,
            low_cpu_mem_usage=True)
        model.eval()
        hf = model.config

    # Rope lives under `rope_parameters` in transformers 5.x. The model
    # derives the rotary dim as head_dim * partial_rotary_factor (default
    # 1.0 = full rotary); a top-level `rotary_dim` is ignored by the code.
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    rope_params = getattr(hf, "rope_parameters", None) or {}
    rope_theta = float(rope_params.get("rope_theta", getattr(hf, "rope_theta", 10000.0)))
    prf = float(rope_params.get("partial_rotary_factor",
                                getattr(hf, "partial_rotary_factor", 1.0)))
    rotary_dim = int(head_dim * prf)
    cfg = dict(
        arch="minimax_m2",
        hidden_size=hf.hidden_size,
        num_q_heads=hf.num_attention_heads,
        num_kv_heads=hf.num_key_value_heads,
        head_dim=head_dim,
        num_layers=hf.num_hidden_layers,
        num_experts=hf.num_local_experts,
        top_k=hf.num_experts_per_tok,
        expert_intermediate=int(getattr(hf, "intermediate_size", 0)),
        partial_rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        rms_norm_eps=float(hf.rms_norm_eps),
        vocab_size=hf.vocab_size,
    )
    return model, cfg


def parse_layers(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < n]


def _ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


# --------------------------------------------------------------------------
# int4_bin expert quantization (matches cascadia-int4-gemm's dequant:
# per-32-col-group symmetric int4, value = (nibble - 8) * scale, nibbles
# packed two-per-byte with the even column in the low nibble, bf16 LE
# per-group scales). Validated against the kernel by the Rust unit test
# `int4_bin_expert_matches_fp32_within_tolerance`.
# --------------------------------------------------------------------------
_INT4_GROUP = 32


def _pack_int4_grouped(w: np.ndarray):
    """Pack `[out, in]` fp32 -> (packed u8 [out, in/2], scale bf16-LE [out, in/32])."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    out, inn = w.shape
    g = _INT4_GROUP
    assert inn % g == 0, f"in_dim {inn} not divisible by group {g}"
    ng = inn // g
    wg = w.reshape(out, ng, g)
    max_abs = np.abs(wg).max(axis=2)
    s = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float32)  # [out, ng]
    q = np.clip(np.round(wg / s[:, :, None]), -8, 7).astype(np.int32)
    nib = (q + 8).astype(np.uint8).reshape(out, inn)
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    u = s.view(np.uint32)
    bf = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")  # bf16 round-to-nearest-even
    return packed.tobytes(), bf.tobytes()


def export_expert_bin(w1: np.ndarray, w3: np.ndarray, w2: np.ndarray, path: Path):
    """Write one expert's flat int4 binary: gate/up/down each as
    packed-nibbles followed by bf16 scales (the layout OvMoeRunner's
    int4_bin backend slices). w1=gate, w3=up, w2=down."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for w in (w1, w3, w2):  # gate, up, down
            packed, scale = _pack_int4_grouped(w)
            f.write(packed)
            f.write(scale)


class StReader:
    """Reads weights from a sharded safetensors checkpoint one tensor at a
    time, dequantizing block-FP8 (`<base>.weight` + `<base>.weight_scale_inv`)
    to fp32. Used by the streaming full-model export so we never hold the
    whole 230B model in RAM."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open  # noqa: F401
        self._safe_open = safe_open
        idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.wm = idx["weight_map"]
        self.dir = model_dir
        self._open = {}

    def _handle(self, shard: str):
        h = self._open.get(shard)
        if h is None:
            h = self._safe_open(str(self.dir / shard), framework="pt")
            self._open[shard] = h
        return h

    def _raw(self, name: str):
        return self._handle(self.wm[name]).get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.wm

    def weight(self, base: str) -> torch.Tensor:
        """fp32 weight for `<base>.weight`, block-FP8-dequantized if a
        `<base>.weight_scale_inv` sibling exists."""
        w = self._raw(base + ".weight")
        sk = base + ".weight_scale_inv"
        if self.has(sk):
            return dequant_fp8_block(w, self._raw(sk))
        return w.to(torch.float32)

    def tensor(self, name: str) -> torch.Tensor:
        return self._raw(name).to(torch.float32)


def export_full_streaming(args, out: Path):
    """Export the full MiniMax-M2 from a local FP8 checkout WITHOUT loading
    the whole model into RAM. Reads each component's weights from the
    safetensors shards on demand, builds the same validated wrappers via
    lightweight namespaces, traces+quantizes, and frees before the next."""
    md = Path(args.model)
    # Load via the canonical config so rotary is derived exactly as the
    # model does it: MiniMax-M2's config.json carries a top-level
    # `rotary_dim` (64) which transformers converts to
    # partial_rotary_factor = rotary_dim/head_dim = 0.5. Reading the raw
    # JSON would miss this (no rope_parameters/partial_rotary_factor) and
    # wrongly default to full rotary.
    from transformers import AutoConfig
    hf = AutoConfig.from_pretrained(md, trust_remote_code=True)
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    prf = getattr(hf, "partial_rotary_factor", None)
    if prf is None:
        prf = (getattr(hf, "rope_parameters", None) or {}).get("partial_rotary_factor", 1.0)
    rope_theta = getattr(hf, "rope_theta", None)
    if rope_theta is None:
        rope_theta = (getattr(hf, "rope_parameters", None) or {}).get("rope_theta", 1e4)
    cfg = dict(
        arch="minimax_m2",
        hidden_size=hf.hidden_size,
        num_q_heads=hf.num_attention_heads,
        num_kv_heads=hf.num_key_value_heads,
        head_dim=head_dim,
        num_layers=hf.num_hidden_layers,
        num_experts=hf.num_local_experts,
        top_k=hf.num_experts_per_tok,
        expert_intermediate=int(hf.intermediate_size),
        partial_rotary_dim=int(head_dim * float(prf)),
        rope_theta=float(rope_theta),
        rms_norm_eps=float(hf.rms_norm_eps),
        vocab_size=hf.vocab_size,
    )
    print(f"[cfg] {json.dumps(cfg)}", flush=True)
    reader = StReader(md)
    H = cfg["hidden_size"]
    KV, D = cfg["num_kv_heads"], cfg["head_dim"]
    qmode = "none" if args.no_quant else "int4"
    expert_q = "int8" if args.experts_int8 else qmode
    # Routing + lm_head are precision-sensitive (MiniMax-M2 keeps gate /
    # e_score_correction_bias / lm_head full-precision); the shell graph
    # carries the router gate, so default shells + head to fp16.
    shell_qmode = "none" if args.no_quant else args.shell_quant
    head_qmode = "none" if args.no_quant else args.head_quant
    layer_ids = parse_layers(args.layers, cfg["num_layers"])

    # embed (layer0) — embed_tokens is not FP8-quantized
    print("[embed] layer0", flush=True)
    ew = reader.weight("model.embed_tokens")
    embed = _ns(num_embeddings=ew.shape[0], embedding_dim=ew.shape[1], weight=ew)
    convert_and_save(EmbedWrapper(embed), (torch.zeros((1, 1), dtype=torch.int64),),
                     out / "layer0" / "openvino_model.xml", {0: [1]},
                     ["input_ids"], ["hidden"], "none" if args.no_quant else "int8")
    del ew, embed

    # head
    print("[head] norm + lm_head", flush=True)
    head = HeadWrapper(_ns(weight=reader.weight("model.norm")),
                       _ns(weight=reader.weight("lm_head")), cfg["rms_norm_eps"])
    convert_and_save(head, (torch.zeros((1, 1, H)),), out / "head" / "openvino_model.xml",
                     {0: [1]}, ["x"], ["logits"], head_qmode)
    del head

    shell_in_names = ["x", "past_k", "past_v", "past_seq_len"]
    shell_out_names = ["attn_out_post_norm", "attn_residual", "shared_expert_out",
                       "routing_ids", "routing_weights", "present_k", "present_v"]
    for li in layer_ids:
        b = f"model.layers.{li}"
        a = b + ".self_attn"
        m = b + ".block_sparse_moe"
        attn = _ns(
            q_proj=_ns(weight=reader.weight(a + ".q_proj")),
            k_proj=_ns(weight=reader.weight(a + ".k_proj")),
            v_proj=_ns(weight=reader.weight(a + ".v_proj")),
            o_proj=_ns(weight=reader.weight(a + ".o_proj")),
            q_norm=_ns(weight=reader.weight(a + ".q_norm")),
            k_norm=_ns(weight=reader.weight(a + ".k_norm")),
        )
        moe = _ns(gate=_ns(weight=reader.weight(m + ".gate")),
                  e_score_correction_bias=reader.tensor(m + ".e_score_correction_bias"))
        layer = _ns(self_attn=attn, mlp=moe,
                    input_layernorm=_ns(weight=reader.weight(b + ".input_layernorm")),
                    post_attention_layernorm=_ns(weight=reader.weight(b + ".post_attention_layernorm")))
        ex = (torch.zeros((1, 1, H)), torch.zeros((1, KV, 3, D)),
              torch.zeros((1, KV, 3, D)), torch.tensor(3, dtype=torch.int64))
        convert_and_save(ShellWrapper(layer, cfg), ex,
                         out / "shells" / f"layer_{li:02d}" / "openvino_model.xml",
                         {1: [2], 2: [2]}, shell_in_names, shell_out_names, shell_qmode)
        del attn, moe, layer

        edir = out / "experts" / f"layer_{li:02d}"
        for ei in range(cfg["num_experts"]):
            eb = f"{m}.experts.{ei}"
            if args.experts == "int4_bin":
                binp = edir / f"expert_{ei:03d}.bin"
                if binp.exists() and binp.stat().st_size > 0:
                    continue  # resume
                w1 = reader.weight(eb + ".w1").numpy()
                w3 = reader.weight(eb + ".w3").numpy()
                w2 = reader.weight(eb + ".w2").numpy()
                export_expert_bin(w1, w3, w2, binp)
                del w1, w3, w2
            else:
                w1 = reader.weight(eb + ".w1")   # gate
                w3 = reader.weight(eb + ".w3")   # up
                w2 = reader.weight(eb + ".w2")   # down
                experts_mod = _ns(gate_up_proj=[torch.cat([w1, w3], dim=0)], down_proj=[w2])
                convert_and_save(ExpertWrapper(experts_mod, 0), (torch.zeros((1, 1, H)),),
                                 edir / f"expert_{ei:03d}" / "openvino_model.xml",
                                 {0: [1]}, ["x"], ["y"], expert_q)
                del w1, w3, w2, experts_mod
        print(f"[layer {li}] shell + {cfg['num_experts']} experts done", flush=True)

    # eos: prefer generation_config.json, else tokenizer/config, else last id
    eos = []
    gc = md / "generation_config.json"
    if gc.exists():
        g = json.loads(gc.read_text())
        e = g.get("eos_token_id")
        eos = e if isinstance(e, list) else ([e] if isinstance(e, int) else [])
    if not eos:
        e = getattr(hf, "eos_token_id", None)
        eos = e if isinstance(e, list) else ([e] if isinstance(e, int) else [cfg["vocab_size"] - 1])
    manifest = {
        "arch": "minimax_m2",
        "num_layers": cfg["num_layers"],
        "dense_layers": [],
        "num_experts": cfg["num_experts"],
        "top_k": cfg["top_k"],
        "hidden_size": cfg["hidden_size"],
        "num_kv_heads": cfg["num_kv_heads"],
        "qk_head_dim": cfg["head_dim"],
        "v_head_dim": cfg["head_dim"],
        "vocab_size": cfg["vocab_size"],
        "eos_token_ids": eos,
        "experts_format": args.experts,
        "expert_intermediate": cfg["expert_intermediate"],
        "shell_backend": "ov_ir",
        "embed_is_layer0": True,
        "num_q_heads": cfg["num_q_heads"],
        "head_dim": cfg["head_dim"],
        "partial_rotary_dim": cfg["partial_rotary_dim"],
        "rope_theta": cfg["rope_theta"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "routing": "sigmoid",
        "use_qk_norm": True,
        "has_shared_expert": False,
        "exported_layers": layer_ids,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for fn in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
               "special_tokens_map.json", "vocab.json", "merges.txt"):
        src = md / fn
        if src.exists():
            shutil.copy(src, out / fn)
    print(f"[manifest] wrote {out/'manifest.json'} (eos={eos})", flush=True)
    print("[done streaming export]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path/hub id of MiniMax-M2 (FP8)")
    ap.add_argument("--tiny", action="store_true", help="export a synthetic small M2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="subset e.g. 0-3 (debug); default all")
    ap.add_argument("--no-quant", action="store_true", help="skip INT4 (fp32 weights)")
    ap.add_argument("--experts-int8", action="store_true",
                    help="INT8 experts instead of INT4 (debug numerics)")
    ap.add_argument("--experts", choices=["ov_ir", "int4_bin"], default="ov_ir",
                    help="full-model expert backend: per-expert OV IR (default) or "
                         "flat int4 binaries for the AVX-512 kernel (faster decode)")
    ap.add_argument("--shell-quant", choices=["none", "int8", "int4"], default="int4",
                    help="precision for the per-layer shell IR (attention + router gate). "
                         "'none' (fp16) keeps routing precise — M2 keeps the gate full-precision")
    ap.add_argument("--head-quant", choices=["none", "int8", "int4"], default="int4",
                    help="precision for the head IR (final norm + lm_head); M2 keeps lm_head full-precision")
    ap.add_argument("--ref-prompt", default="1,2,3,4,5",
                    help="comma token ids for the --tiny reference forward")
    ap.add_argument("--ref-gen", type=int, default=6, help="reference greedy tokens")
    args = ap.parse_args()

    if not args.tiny and not args.model:
        ap.error("need --model or --tiny")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Full model: stream per-layer from safetensors (the model is far too
    # large to load into RAM via from_pretrained). --tiny uses the
    # in-memory path below (and writes a reference.json for the test).
    if args.model and not args.tiny:
        export_full_streaming(args, out)
        return

    qmode = "none" if args.no_quant else "int4"
    expert_q = "int8" if args.experts_int8 else qmode

    print("[load] reference model ...", flush=True)
    model, cfg = load_reference(args)
    H = cfg["hidden_size"]
    base = model.model  # MiniMaxM2Model
    layer_ids = parse_layers(args.layers, cfg["num_layers"])
    print(f"[cfg] {json.dumps(cfg)}", flush=True)
    print(f"[plan] exporting layers {layer_ids[0]}..{layer_ids[-1]} "
          f"({len(layer_ids)} layers x {cfg['num_experts']} experts)", flush=True)

    # --- embed (layer0) ---
    print("[embed] exporting layer0 (embed_tokens)", flush=True)
    ex_ids = torch.zeros((1, 1), dtype=torch.int64)
    convert_and_save(EmbedWrapper(base.embed_tokens), (ex_ids,),
                     out / "layer0" / "openvino_model.xml",
                     dyn_axes={0: [1]}, in_names=["input_ids"], out_names=["hidden"],
                     quant_mode=("none" if args.no_quant else "int8"))

    # --- head ---
    print("[head] exporting head (norm + lm_head)", flush=True)
    ex_h = torch.zeros((1, 1, H), dtype=torch.float32)
    convert_and_save(HeadWrapper(base.norm, model.lm_head, cfg["rms_norm_eps"]), (ex_h,),
                     out / "head" / "openvino_model.xml",
                     dyn_axes={0: [1]}, in_names=["x"], out_names=["logits"],
                     quant_mode=qmode)

    # --- shells + experts ---
    KV, D = cfg["num_kv_heads"], cfg["head_dim"]
    shell_in_names = ["x", "past_k", "past_v", "past_seq_len"]
    shell_out_names = ["attn_out_post_norm", "attn_residual", "shared_expert_out",
                       "routing_ids", "routing_weights", "present_k", "present_v"]
    for li in layer_ids:
        layer = base.layers[li]
        print(f"[shell] layer {li}", flush=True)
        ex = (torch.zeros((1, 1, H)),
              torch.zeros((1, KV, 3, D)),   # past_seq=3 example; axis 2 dynamic
              torch.zeros((1, KV, 3, D)),
              torch.tensor(3, dtype=torch.int64))
        convert_and_save(ShellWrapper(layer, cfg), ex,
                         out / "shells" / f"layer_{li:02d}" / "openvino_model.xml",
                         dyn_axes={1: [2], 2: [2]}, in_names=shell_in_names,
                         out_names=shell_out_names, quant_mode=qmode)
        edir = out / "experts" / f"layer_{li:02d}"
        experts_mod = layer.mlp.experts
        for ei in range(cfg["num_experts"]):
            convert_and_save(ExpertWrapper(experts_mod, ei),
                             (torch.zeros((1, 1, H)),),
                             edir / f"expert_{ei:03d}" / "openvino_model.xml",
                             dyn_axes={0: [1]}, in_names=["x"], out_names=["y"],
                             quant_mode=expert_q)
        print(f"[shell] layer {li} done ({cfg['num_experts']} experts)", flush=True)

    # --- manifest ---
    eos = []
    try:
        eos_id = getattr(model.config, "eos_token_id", None)
        if isinstance(eos_id, int):
            eos = [eos_id]
        elif isinstance(eos_id, list):
            eos = eos_id
    except Exception:
        pass
    if not eos:
        eos = [cfg["vocab_size"] - 1]
    manifest = {
        "arch": "minimax_m2",
        "num_layers": cfg["num_layers"],
        "dense_layers": [],                 # all layers are MoE
        "num_experts": cfg["num_experts"],
        "top_k": cfg["top_k"],
        "hidden_size": cfg["hidden_size"],
        "num_kv_heads": cfg["num_kv_heads"],
        "qk_head_dim": cfg["head_dim"],
        "v_head_dim": cfg["head_dim"],
        "vocab_size": cfg["vocab_size"],
        "eos_token_ids": eos,
        "expert_intermediate": cfg.get("expert_intermediate", 0),
        "experts_format": "ov_ir",
        # M2-specific runtime hints (read by the OV-IR shell backend):
        "shell_backend": "ov_ir",
        "embed_is_layer0": True,
        "num_q_heads": cfg["num_q_heads"],
        "head_dim": cfg["head_dim"],
        "partial_rotary_dim": cfg["partial_rotary_dim"],
        "rope_theta": cfg["rope_theta"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "routing": "sigmoid",
        "use_qk_norm": True,
        "has_shared_expert": False,
        "exported_layers": layer_ids,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {out/'manifest.json'}", flush=True)

    # --- reference (tiny): greedy token sequence from the canonical model ---
    if args.tiny:
        prompt = [int(x) for x in args.ref_prompt.split(",")]
        ids = torch.tensor([prompt], dtype=torch.int64)
        gen = list(prompt)
        with torch.no_grad():
            cur = ids
            for _ in range(args.ref_gen):
                logits = model(cur).logits[0, -1]
                nxt = int(torch.argmax(logits))
                gen.append(nxt)
                cur = torch.tensor([gen], dtype=torch.int64)
        # also save first-step next-token logits for a tighter check
        with torch.no_grad():
            first_logits = model(ids).logits[0, -1].to(torch.float32).numpy()
        ref = {
            "prompt_ids": prompt,
            "greedy_tokens": gen,
            "first_next_token": gen[len(prompt)],
            "first_logits_top8": np.argsort(-first_logits)[:8].tolist(),
        }
        (out / "reference.json").write_text(json.dumps(ref, indent=2))
        # copy a tokenizer-free note; tiny has no tokenizer
        print(f"[reference] greedy_tokens={gen}", flush=True)

    # copy tokenizer for full model
    if not args.tiny:
        for fn in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                   "special_tokens_map.json", "vocab.json", "merges.txt"):
            src = Path(args.model) / fn
            if src.exists():
                shutil.copy(src, out / fn)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
