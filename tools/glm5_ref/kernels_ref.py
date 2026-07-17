"""Pure-torch CPU primitives for the GLM-5.2 shell — the readable spec + ground
truth the Rust port in `src/glm/` is validated against.

Semantics are transcribed from GLM-5.2 / DeepSeek-V3 modeling code. Computation
is fp32 (the router runs in fp32 upstream); bit-exactness with any CUDA kernel
is not the goal — the Rust shell is validated against *this*.
"""
import math
from typing import Dict, Tuple

import torch


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Round f32 -> bf16 -> f32 (RNE), matching `dsv4::math::to_bf16`."""
    return t.to(torch.bfloat16).to(torch.float32)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: f32 accumulation, bf16-rounded output — the dsv4 convention
    (`dsv4::math::rmsnorm`). Weight only, no bias."""
    ms = (x.to(torch.float32) ** 2).mean(dim=-1, keepdim=True)
    return _bf16(x.to(torch.float32) * torch.rsqrt(ms + eps) * w.to(torch.float32))


def _lin(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w^T with bf16-rounded output (`dsv4::math::linear_bf16_w`).
    `w` is bf16-valued [O, I]; f32 accumulation."""
    return _bf16(x.to(torch.float32) @ w.to(torch.float32).t())


def precompute_freqs_cis(dim: int, seqlen: int, base: float) -> torch.Tensor:
    """GLM-5.2 rope frequency table. `rope_type="default"` -> NO YaRN, so this
    is the plain inverse-frequency table (unlike the V4 reference which blends
    YaRN when original_seq_len > 0).

    freqs[i] = base^-(2i/dim); returns the complex cis of shape
    [seqlen, dim/2]. Computed in fp32 to match the Rust `precompute_freqs`.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    ang = torch.outer(t, freqs)                          # [seqlen, dim/2]
    return torch.polar(torch.ones_like(ang), ang)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Interleaved partial RoPE (`rope_interleave=True`): adjacent (even, odd)
    pairs of the last dim are rotated as one complex number by `freqs_cis` —
    NOT rotate-half. `x[..., -1]` must be `2 * freqs_cis.shape[-1]`.

    Output is bf16-rounded to match the Rust `apply_rope_row` write-back.
    """
    shape = x.shape
    xc = torch.view_as_complex(x.to(torch.float32).reshape(*shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis).reshape(shape)
    return out.to(torch.bfloat16)


def moe_gate(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    scale: float,
    norm_topk: bool = True,
    eps: float = 1e-20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GLM-5.2 / DeepSeek-V3 `noaux_tc` router gate (n_group=1, topk_group=1 ->
    plain top-k, no group masking).

    logits: [T, E] fp32 router logits (x @ W_gate^T).
    bias:   [E]    fp32 `e_score_correction_bias` (selection-only).

    - scores  = sigmoid(logits)                          # scoring_func="sigmoid"
    - choice  = scores + bias                            # noaux_tc selection score
    - select top_k experts by `choice`, DESC, ties broken toward the LOWER id
      (a fully deterministic contract the Rust side mirrors exactly — avoids
      torch.topk's unspecified tie order)
    - weight  = scores[selected]                         # ORIGINAL scores, not choice
    - if norm_topk and top_k>1: weight /= weight.sum() + eps
    - weight *= scale                                    # routed_scaling_factor

    Returns (idx:[T, top_k] int32, weight:[T, top_k] fp32) in the canonical
    selection order.
    """
    logits = logits.to(torch.float32)
    bias = bias.to(torch.float32)
    scores = torch.sigmoid(logits)                       # [T, E]
    choice = scores + bias                               # [T, E]
    t, e = scores.shape
    assert top_k <= e, (top_k, e)
    idx = torch.empty(t, top_k, dtype=torch.int32)
    weight = torch.empty(t, top_k, dtype=torch.float32)
    for r in range(t):
        ch = choice[r]
        order = sorted(range(e), key=lambda i: (-float(ch[i]), i))[:top_k]
        sel = torch.tensor(order, dtype=torch.long)
        w = scores[r, sel].clone()
        if norm_topk and top_k > 1:
            w = w / (w.sum() + eps)
        w = w * scale
        idx[r] = torch.tensor(order, dtype=torch.int32)
        weight[r] = w
    return idx, weight


def attention_ref(x: torch.Tensor, w: Dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    """GLM-5.2 MLA attention, NAIVE (materialized) full-causal form — the
    obviously-correct ground truth the Rust *absorbed* path is validated
    against. Absorbed == naive by linearity; they differ only by f32
    accumulation order (ULP), which the golden tolerance absorbs.

    x: [S, hidden] fp32 (bf16-valued). `w` keys: wq_a[qlora,hidden],
    q_a_ln[qlora], wq_b[H*qk,qlora], wkv_a[kvl+rope,hidden], kv_a_ln[kvl],
    wkv_b[H*(nope+vh),kvl], wo[hidden,H*vh] (all bf16-valued). cfg dims:
    n_heads, hidden, qk_nope, qk_rope, v_head, kv_lora, q_lora, eps, theta.

    Numeric contract (matches the Rust shell): bf16 rounding after each
    linear / RMSNorm / rope; the kv_b up-projection and the score/softmax/
    context core stay in f32 (never materialized+rounded in the absorbed
    path, so the reference must not round them either). Softmax scale =
    qk_head^-0.5 = (nope+rope)^-0.5. No mscale, no sink, no bias.
    """
    H, nope, rope = cfg["n_heads"], cfg["qk_nope"], cfg["qk_rope"]
    vh, kvl, eps, theta = cfg["v_head"], cfg["kv_lora"], cfg["eps"], cfg["theta"]
    qk = nope + rope
    S = x.shape[0]
    scale = 1.0 / math.sqrt(qk)
    fc = precompute_freqs_cis(rope, S, theta)                    # [S, rope/2]

    qr = _rms(_lin(x, w["wq_a"]), w["q_a_ln"], eps)              # [S, qlora]
    q = _lin(qr, w["wq_b"]).reshape(S, H, qk)                    # [S,H,qk]
    comp = _lin(x, w["wkv_a"])                                   # [S, kvl+rope]
    lat = _rms(comp[:, :kvl], w["kv_a_ln"], eps)                 # [S, kvl]  (Lc)
    kpe = comp[:, kvl:]                                          # [S, rope]

    # rope q_pe (per head) and k_pe (shared) — interleaved, bf16-rounded.
    qpe = apply_rotary_emb(q[:, :, nope:qk], fc.unsqueeze(1))    # [S,H,rope]
    qnope = q[:, :, :nope]                                       # [S,H,nope]  (bf16)
    kpe = apply_rotary_emb(kpe, fc)                             # [S, rope]  (Rc)

    # kv_b up-projection, f32 core (NO bf16 rounding of k_nope / value).
    kvb = (lat @ w["wkv_b"].to(torch.float32).t()).reshape(S, H, nope + vh)
    knope = kvb[:, :, :nope]                                     # [S,H,nope]
    value = kvb[:, :, nope:]                                     # [S,H,vh]

    ctx = torch.zeros(S, H, vh, dtype=torch.float32)
    for s in range(S):
        for h in range(H):
            sc = torch.empty(s + 1, dtype=torch.float32)
            for t in range(s + 1):
                sn = torch.dot(qnope[s, h], knope[t, h])
                sp = torch.dot(qpe[s, h], kpe[t])
                sc[t] = (sn + sp) * scale
            p = torch.softmax(sc, dim=0)
            ctx[s, h] = (p.unsqueeze(-1) * value[: s + 1, h]).sum(0)

    return _lin(ctx.reshape(S, H * vh), w["wo"])                 # [S, hidden]
