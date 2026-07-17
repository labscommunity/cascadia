"""Pure-torch CPU primitives for the GLM-5.2 shell — the readable spec + ground
truth the Rust port in `src/glm/` is validated against.

Semantics are transcribed from GLM-5.2 / DeepSeek-V3 modeling code. Computation
is fp32 (the router runs in fp32 upstream); bit-exactness with any CUDA kernel
is not the goal — the Rust shell is validated against *this*.
"""
from typing import Tuple

import torch


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
