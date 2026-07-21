"""Pure-PyTorch (CPU-friendly) replacements for DeepSeek-V4-Flash's tilelang
kernels (`inference/kernel.py` in the deepseek-ai/DeepSeek-V4-Flash repo).

These exist so the vendored reference model (`model.py` here) runs on CPU
without CUDA/tilelang/fast_hadamard_transform, giving cascadia a local ground
truth for the tiny-model export test AND a readable spec for the Rust shell
port.

Semantics are transcribed 1:1 from the upstream kernels:

- act_quant       block-wise FP8 (e4m3, amax/448) quant; `inplace=True` is the
                  QAT-style fused quant+dequant used pervasively at inference.
                  When `scale_fmt` is set, scales round UP to a power of two
                  (2^ceil(log2(amax/448)) — `fast_round_scale` bit-trick).
- fp4_act_quant   block-32 FP4 (e2m1, amax/6) with power-of-2 scales; RNE onto
                  the e2m1 grid {0,.5,1,1.5,2,3,4,6} (ties -> even mantissa).
- fp8_gemm        C = A_fp8 @ B_fp8^T with per-[1,128] act scales and
                  per-[128,128] weight scales (dequant-then-matmul here).
- fp4_gemm        C = A_fp8 @ B_fp4^T; B packed 2-per-byte along K (low nibble
                  = even element, per upstream inference/convert.py), per-32
                  e8m0 weight scales.
- sparse_attn     gather top-k KV rows by index (-1 = masked), softmax with a
                  per-head learnable sink logit that joins the denominator
                  only, then prob-weighted sum of the same gathered rows.
- hc_split_sinkhorn  Hyper-Connections mixing coefficients: sigmoid pre/post
                  gates + a Sinkhorn-normalised (row/col, `iters` rounds)
                  softmax combination matrix.
- hadamard_transform  standard Walsh-Hadamard butterfly (replaces the
                  fast_hadamard_transform CUDA extension).

Numerical notes: computations are done in fp32 like the upstream kernels
(which accumulate in fp32). Bit-exactness with the CUDA kernels is NOT
guaranteed (different reduction orders); this module is cascadia's reference
— the exporter and the Rust shell are validated against *it*.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0

# e2m1 magnitude grid and the midpoints between adjacent entries.
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# For a value exactly on midpoint k (between grid k and k+1), RNE picks the
# entry with even mantissa: [0, 2, 2, 4, 4, 6, 6][k].
_E2M1_TIE_IDX = torch.tensor([0, 2, 2, 4, 4, 6, 6])


def e8m0_to_f32(s: torch.Tensor) -> torch.Tensor:
    """Decode float8_e8m0fnu (biased power-of-two exponent byte) to fp32."""
    if s.dtype == torch.float8_e8m0fnu:
        bits = s.view(torch.uint8).to(torch.int32)
        return torch.exp2((bits - 127).to(torch.float32))
    return s.to(torch.float32)


def _pow2_round_up(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) for positive x — upstream `fast_round_scale`."""
    return torch.exp2(torch.ceil(torch.log2(x)))


def _quantize_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the e2m1 grid. `v` already clamped to ±6."""
    a = v.abs()
    grid = _E2M1_GRID.to(v.dtype)
    mid = _E2M1_MID.to(v.dtype)
    idx = torch.bucketize(a, mid, right=False)  # a < mid[k] -> k
    # Exact midpoints: bucketize(right=False) sends a == mid[k] to k+1;
    # RNE wants the even-mantissa neighbour instead.
    for k in range(mid.numel()):
        tie = a == mid[k]
        if tie.any():
            idx = torch.where(tie, _E2M1_TIE_IDX[k].to(idx.dtype), idx)
    return torch.sign(v) * grid[idx]


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
):
    """Block-wise FP8 quantization along the last dim.

    inplace=True: fused quant+dequant written back into `x` (returns x).
    inplace=False: returns (y_e4m3, scales) with scales stored as
    `scale_dtype` (fp32 or e8m0)."""
    N = x.size(-1)
    assert N % block_size == 0, (N, block_size)
    xf = x.to(torch.float32).reshape(-1, N // block_size, block_size)
    amax = xf.abs().amax(dim=-1).clamp_min(1e-4)          # [M, G]
    if scale_fmt is not None:
        s = _pow2_round_up(amax * (1.0 / FP8_MAX))
    else:
        s = amax * (1.0 / FP8_MAX)
    q = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    q8 = q.to(torch.float8_e4m3fn)
    if inplace:
        y = (q8.to(torch.float32) * s.unsqueeze(-1)).reshape(x.shape).to(x.dtype)
        x.copy_(y)
        return x
    y = q8.reshape(x.shape)
    s_out = s.reshape(*x.shape[:-1], N // block_size)
    if scale_dtype == torch.float8_e8m0fnu:
        s_out = s_out.to(torch.float8_e8m0fnu)
    return y, s_out


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """Block-wise FP4 (e2m1) quantization with power-of-2 scales."""
    N = x.size(-1)
    assert N % block_size == 0, (N, block_size)
    xf = x.to(torch.float32).reshape(-1, N // block_size, block_size)
    amax = xf.abs().amax(dim=-1).clamp_min(6.0 * 2.0 ** -126)
    s = _pow2_round_up(amax * (1.0 / FP4_MAX))
    q = (xf / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)
    q = _quantize_e2m1(q)
    if inplace:
        y = (q * s.unsqueeze(-1)).reshape(x.shape).to(x.dtype)
        x.copy_(y)
        return x
    # Non-inplace packed output is unused by the reference model at inference;
    # return dequantized values + scales for completeness.
    return q.reshape(x.shape), s.reshape(*x.shape[:-1], N // block_size)


def _dequant_fp8_weight(b: torch.Tensor, b_s: torch.Tensor, block: int = 128) -> torch.Tensor:
    """[N,K] e4m3 weight * per-[128,128]-block scale -> fp32."""
    n, k = b.shape
    s = e8m0_to_f32(b_s)
    s = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return b.to(torch.float32) * s


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T, per-[1,128] act / [128,128] weight scales."""
    K = a.size(-1)
    af = a.to(torch.float32).reshape(-1, K)
    asf = e8m0_to_f32(a_s).reshape(af.size(0), -1)          # [M, K/128]
    af = af * asf.repeat_interleave(K // asf.size(-1), dim=-1)
    bf = _dequant_fp8_weight(b, b_s)
    c = af @ bf.t()
    return c.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def dequant_fp4_weight(b: torch.Tensor, b_s: torch.Tensor,
                       group: int = 32) -> torch.Tensor:
    """Unpack [N, K/2] float4_e2m1fn_x2 (or int8/uint8 view of it) + per-32
    e8m0 scales -> fp32 [N, K]. Low nibble = even element (convert.py)."""
    if b.dtype == getattr(torch, "float4_e2m1fn_x2", None):
        raw = b.view(torch.uint8)
    else:
        raw = b.view(torch.uint8) if b.dtype == torch.int8 else b
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    vals = torch.stack([_FP4_LUT[low.long()], _FP4_LUT[high.long()]], dim=-1)
    vals = vals.flatten(-2)                                  # [N, K]
    s = e8m0_to_f32(b_s).repeat_interleave(group, dim=-1)    # [N, K]
    return vals * s


def fp4_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
             scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T (per-128 act, per-32 e8m0 weight scales)."""
    K = a.size(-1)
    af = a.to(torch.float32).reshape(-1, K)
    asf = e8m0_to_f32(a_s).reshape(af.size(0), -1)
    af = af * asf.repeat_interleave(K // asf.size(-1), dim=-1)
    bf = dequant_fp4_weight(b, b_s)
    c = af @ bf.t()
    return c.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    """q:[b,s,h,d], kv:[b,n,d], attn_sink:[h] fp32, topk_idxs:[b,s,topk] i32
    (-1 = masked slot). The sink contributes exp(sink) to the softmax
    denominator but no value row. Returns o:[b,s,h,d] in q.dtype."""
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0                                          # [b,s,t]
    gathered = kv.gather(
        1, idx.clamp_min(0).reshape(b, -1, 1).expand(-1, -1, d)
    ).reshape(b, s, -1, d).to(torch.float32)                  # [b,s,t,d]
    qf = q.to(torch.float32)
    scores = torch.einsum("bshd,bstd->bsht", qf, gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.to(torch.float32).view(1, 1, h, 1)
    m = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    e = torch.exp(scores - m)                                 # [b,s,h,t]
    denom = e.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
    probs = e / denom
    o = torch.einsum("bsht,bstd->bshd", probs, gathered)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor,
                      hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes:[b,s,(2+hc)*hc] fp32 -> (pre:[b,s,hc], post:[b,s,hc],
    comb:[b,s,hc,hc]). Matches hc_split_sinkhorn_kernel exactly:
      pre  = sigmoid(m[:hc]  * scale[0] + base[:hc]) + eps
      post = 2*sigmoid(m[hc:2hc] * scale[1] + base[hc:2hc])
      comb = softmax(m[2hc:] * scale[2] + base[2hc:], rows) + eps, then
             col-normalise, then (iters-1) x [row-normalise, col-normalise]
             (each normalise divides by sum + eps)."""
    hc = hc_mult
    m = mixes.to(torch.float32)
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = m[..., 2 * hc:].reshape(*m.shape[:-1], hc, hc) * hc_scale[2] \
        + hc_base[2 * hc:].reshape(hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Walsh-Hadamard transform along the last dim (power-of-2 size),
    multiplied by `scale`. Replaces the fast_hadamard_transform extension."""
    d = x.size(-1)
    assert d & (d - 1) == 0, f"hadamard dim {d} not a power of 2"
    orig_dtype = x.dtype
    y = x.to(torch.float32).clone()
    h = 1
    while h < d:
        y = y.unflatten(-1, (-1, 2, h))
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).flatten(-3)
        h *= 2
    return (y * scale).to(orig_dtype)
