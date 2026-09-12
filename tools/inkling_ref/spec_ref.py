"""Pure-torch f32 implementation of PORT_SPEC.md §3 over CHECKPOINT-named tensors.

This is the spec the Rust port is written from, executed literally (per-position loops, no
bf16 write-back). `gen_fixtures.py` runs it next to transformers' `InklingForCausalLM` so any
place where the spec and HF disagree shows up as a numeric mismatch BEFORE the Rust side is
debugged against the fixtures.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def sconv(w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """w [C, K], u [T, C]. out[p, c] = sum_j w[c, j] * u[p - (K-1) + j, c] with u[<0] = 0
    (kernel tap K-1 multiplies the CURRENT token, tap 0 the token K-1 back); returns out + u
    (the residual lives inside the module)."""
    T, C = u.shape
    K = w.shape[1]
    up = torch.cat([torch.zeros(K - 1, C, dtype=u.dtype), u], dim=0)
    out = torch.zeros_like(u)
    for j in range(K):
        out = out + w[:, j][None, :] * up[j:j + T]
    return out + u


def relpos_bias(r: torch.Tensor, proj: torch.Tensor, q_pos: int, kv_pos) -> torch.Tensor:
    """r [Hq, d_rel] (one query), proj [d_rel, extent] -> bias [Hq, len(kv_pos)];
    bias = r_h . proj[:, dist] for 0 <= dist < extent, else 0."""
    extent = proj.shape[1]
    out = torch.zeros(r.shape[0], len(kv_pos))
    for j, kp in enumerate(kv_pos):
        d = q_pos - kp
        if 0 <= d < extent:
            out[:, j] = r @ proj[:, d]
    return out


def gate(logits: torch.Tensor, bias: torch.Tensor, top_k: int, n_shared: int,
         route_scale: float, global_scale: float):
    """logits [E + n_shared] f32 -> (idx [top_k] i64 in canonical order = selection score
    descending, ties toward the lower id; w [top_k]; gammas [n_shared])."""
    E = logits.shape[0] - n_shared
    s = torch.sigmoid(logits.float())
    choice = s[:E] + bias.float()
    order = sorted(range(E), key=lambda i: (-float(choice[i]), i))[:top_k]
    den = sum(float(s[i]) for i in order) + float(s[E:].sum())
    w = torch.tensor([float(s[i]) / den * route_scale * global_scale for i in order], dtype=torch.float32)
    gam = (s[E:] / den * route_scale * global_scale).float()
    return torch.tensor(order, dtype=torch.int64), w, gam


def attention(h: torch.Tensor, W: dict, *, heads: int, kv_heads: int, head_dim: int, d_rel: int,
              window, n_floor, alpha: float, eps: float) -> torch.Tensor:
    """Prefill of the spec's attention on h [T, H] (already attn_norm'ed). W holds one layer's
    checkpoint-suffix-named f32 tensors (convs as [C, K]). Returns o = Wo concat_h(a_h) [T, H]
    — BEFORE attn_sconv, which the decoder layer applies. `window=None` -> global layer."""
    T = h.shape[0]
    q = h @ W["attn.wq_du.weight"].T
    kr = h @ W["attn.wk_dv.weight"].T
    vr = h @ W["attn.wv_dv.weight"].T
    r = (h @ W["attn.wr_du.weight"].T).view(T, heads, d_rel)
    kc = sconv(W["attn.k_sconv.weight"], kr)
    v = sconv(W["attn.v_sconv.weight"], vr).view(T, kv_heads, head_dim)
    q = rmsnorm(q.view(T, heads, head_dim), W["attn.q_norm.weight"], eps)
    k = rmsnorm(kc.view(T, kv_heads, head_dim), W["attn.k_norm.weight"], eps)
    proj = W["attn.rel_logits_proj.proj"]
    grp = heads // kv_heads
    out = torch.zeros(T, heads, head_dim)
    for p in range(T):
        tau = 1.0
        if window is None and n_floor is not None:
            tau = 1.0 + alpha * math.log(max((p + 1) / n_floor, 1.0))
        qp = q[p] * tau
        js = [j for j in range(p + 1) if window is None or p - j < window]
        bias = relpos_bias(r[p], proj, p, js) * tau
        for hh in range(heads):
            kh = k[js, hh // grp]
            score = (kh @ qp[hh]) / head_dim + bias[hh]
            a = torch.softmax(score, dim=0)
            out[p, hh] = a @ v[js, hh // grp]
    return out.reshape(T, heads * head_dim) @ W["attn.wo_ud.weight"].T


def swiglu(gate_w, up_w, down_w, h):
    return (F.silu(h @ gate_w.T) * (h @ up_w.T)) @ down_w.T


def deinterleave_rows(w13: torch.Tensor):
    """checkpoint w13 rows: 2i = gate_i, 2i+1 = up_i."""
    return w13[0::2], w13[1::2]


def dense_mlp(W: dict, h2: torch.Tensor) -> torch.Tensor:
    g, u = deinterleave_rows(W["mlp.w13_dn.weight"])
    return swiglu(g, u, W["mlp.w2_md.weight"], h2) * W["mlp.global_scale"]


def moe(W: dict, h2: torch.Tensor, *, top_k: int, n_shared: int, route_scale: float) -> torch.Tensor:
    Wg, bias, gs = W["mlp.gate.weight"], W["mlp.gate.bias"], float(W["mlp.gate.global_scale"])
    w13, w2 = W["mlp.experts.w13_weight"], W["mlp.experts.w2_weight"]
    sw13, sw2 = W["mlp.shared_experts.shared_w13_weight"], W["mlp.shared_experts.shared_w2_weight"]
    out = torch.zeros_like(h2)
    for t in range(h2.shape[0]):
        idx, w, gam = gate(Wg @ h2[t], bias, top_k, n_shared, route_scale, gs)
        acc = torch.zeros(h2.shape[1])
        for i, wi in zip(idx.tolist(), w.tolist()):
            g, u = deinterleave_rows(w13[i])
            acc = acc + wi * swiglu(g, u, w2[i], h2[t])
        for s in range(n_shared):
            g, u = deinterleave_rows(sw13[s])
            acc = acc + float(gam[s]) * swiglu(g, u, sw2[s], h2[t])
        out[t] = acc
    return out


def layer_weights(ckpt: dict, li: int) -> dict:
    """One layer's tensors keyed by checkpoint suffix, f32, convs reshaped [C, 1, K] -> [C, K]."""
    p = f"model.llm.layers.{li}."
    W = {}
    for k, v in ckpt.items():
        if k.startswith(p):
            t = v.float()
            if k.endswith("sconv.weight"):
                t = t.reshape(t.shape[0], t.shape[-1])
            W[k[len(p):]] = t
    return W


def layer_dims(man: dict, li: int) -> dict:
    sliding = man["layer_types"][li] == "sliding"
    return dict(
        heads=man["swa_num_attention_heads"] if sliding else man["num_attention_heads"],
        kv_heads=man["swa_num_kv_heads"] if sliding else man["num_kv_heads"],
        head_dim=man["swa_head_dim"] if sliding else man["head_dim"],
        d_rel=man["d_rel"],
        window=man["sliding_window"] if sliding else None,
        n_floor=man["log_scaling_n_floor"],
        alpha=man["log_scaling_alpha"],
        eps=man["rms_norm_eps"],
    )


def layer_forward(x: torch.Tensor, W: dict, man: dict, li: int) -> torch.Tensor:
    eps = man["rms_norm_eps"]
    h = rmsnorm(x, W["attn_norm.weight"], eps)
    o = attention(h, W, **layer_dims(man, li))
    o = sconv(W["attn_sconv.weight"], o)
    x = x + o
    h2 = rmsnorm(x, W["mlp_norm.weight"], eps)
    if li in set(man["dense_layers"]):
        m = dense_mlp(W, h2)
    else:
        m = moe(W, h2, top_k=man["top_k"], n_shared=man["n_shared_experts"], route_scale=man["route_scale"])
    m = sconv(W["mlp_sconv.weight"], m)
    return x + m


def model_forward(ids: list[int], ckpt: dict, man: dict):
    """Prefill: returns (per-layer outputs [T, H], logits [T, unpadded_vocab])."""
    eps = man["rms_norm_eps"]
    x = ckpt["model.llm.embed.weight"].float()[torch.tensor(ids)]
    x = rmsnorm(x, ckpt["model.llm.embed_norm.weight"].float(), eps)
    outs = []
    for li in range(man["num_layers"]):
        x = layer_forward(x, layer_weights(ckpt, li), man, li)
        outs.append(x.clone())
    y = rmsnorm(x, ckpt["model.llm.norm.weight"].float(), eps) / man["logits_mup_width_multiplier"]
    logits = (y @ ckpt["model.llm.unembed.weight"].float().T)[:, :man["unpadded_vocab_size"]]
    return outs, logits
