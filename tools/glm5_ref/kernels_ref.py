"""Pure-torch CPU primitives for the GLM-5.2 shell — the readable spec + ground
truth the Rust port in `src/glm/` is validated against.

Semantics are transcribed from GLM-5.2 / DeepSeek-V3 modeling code. Computation
is fp32 (the router runs in fp32 upstream); bit-exactness with any CUDA kernel
is not the goal — the Rust shell is validated against *this*.
"""
from typing import Tuple

import torch


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
