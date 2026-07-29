"""Pure-torch CPU primitives for the Kimi-K3 shell — the readable spec + ground
truth the Rust port in `src/k3/` is validated against.

Semantics are transcribed from the vendored upstream sources under
`upstream/`:
  - `modeling_kimi_linear.py`  (HF `moonshotai/Kimi-K3`)
  - `kda_naive.py`, `kda_gate.py`  (`fla-org/flash-linear-attention`, MIT)

Computation is fp32 with bf16 rounding after each linear / norm — the dsv4 /
glm5 convention (`dsv4::math`). Cores that upstream explicitly runs in fp32
stay fp32 here: the KDA recurrence, the AttnRes softmax mixture, the router,
the attention softmax, and the SiTU activation.

Bit-exactness with any CUDA/Triton kernel is not the goal — the Rust shell is
validated against *this*.
"""
import math

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# dtype helpers (mirror `dsv4::math`)
# --------------------------------------------------------------------------


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Round f32 -> bf16 -> f32 (RNE), matching `dsv4::math::to_bf16`."""
    return t.to(torch.bfloat16).to(torch.float32)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with weight: f32 accumulation, bf16-rounded output."""
    ms = (x.to(torch.float32) ** 2).mean(dim=-1, keepdim=True)
    return _bf16(x.to(torch.float32) * torch.rsqrt(ms + eps) * w.to(torch.float32))


def _lin(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w^T with bf16-rounded output. `w` is [O, I]; f32 accumulation."""
    return _bf16(x.to(torch.float32) @ w.to(torch.float32).t())


# --------------------------------------------------------------------------
# SiTU activation  (upstream `SituAndMul`)
# --------------------------------------------------------------------------


def situ(gate_up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """SiTU: `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`.

    `gate_up` is the concatenation `[gate | up]` along the last dim. Upstream
    computes this in f32 and casts back, so we keep the whole thing f32.
    K3 config: beta=4.0, linear_beta=25.0.
    """
    d = gate_up.shape[-1] // 2
    g = gate_up[..., :d].to(torch.float32)
    u = gate_up[..., d:].to(torch.float32)
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return a * u


def mlp_ref(x, w1, w3, w2, beta: float, linear_beta: float | None):
    """SiTU MLP: `w2( SiTU(cat[w1(x), w3(x)]) )`.

    w1 = gate, w3 = up, w2 = down — three separate matrices in the checkpoint
    (upstream concatenates only at runtime). Used by routed experts, the shared
    expert, and the dense layer-0 MLP.
    """
    gate_up = torch.cat([_lin(x, w1), _lin(x, w3)], dim=-1)
    return _lin(situ(gate_up, beta, linear_beta), w2)


# --------------------------------------------------------------------------
# MoE router  (upstream `KimiMoEGate`) — identical in shape to glm5's
# --------------------------------------------------------------------------


def moe_gate(logits, bias, top_k: int, scale: float, renormalize: bool):
    """sigmoid scoring + `noaux_tc` selection bias, norm-topk, then `* scale`.

    Selection is by `(scores + bias)`; the returned weights are the ORIGINAL
    sigmoid scores of the selected experts. Upstream calls `torch.topk(...,
    sorted=False)`, whose tie order is unspecified — we define a deterministic
    order (selection-score descending, ties toward the LOWER expert id) so the
    Rust port has something exact to match. This mirrors
    `glm::gate::moe_gate`, which K3 reuses unchanged (K3 just has scale=1.0).

    `logits` is [T, E] (computed in f32 upstream), `bias` is [E].
    """
    scores = torch.sigmoid(logits.to(torch.float32))
    choice = scores + bias.to(torch.float32).unsqueeze(0)

    t, e = scores.shape
    # deterministic top-k: sort by choice desc, ties -> lower id
    order = torch.argsort(
        choice - torch.arange(e, dtype=torch.float32, device=scores.device) * 1e-30,
        dim=-1, descending=True, stable=True,
    )[:, :top_k]

    weight = torch.gather(scores, 1, order)
    if renormalize and top_k > 1:
        weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)
    return order, weight * scale


# --------------------------------------------------------------------------
# Attention Residuals  (upstream `_apply_attn_res`)
# --------------------------------------------------------------------------


def apply_attn_res(prefix_sum, block_residual, proj_w, norm_w, eps: float):
    """Learned softmax mixture over the block-residual stack + the prefix sum.

    prefix_sum:     [T, H]
    block_residual: [T, nb, H]
    proj_w:         [1, H]   (nn.Linear(H, 1, bias=False).weight)
    norm_w:         [H]

    Note this is NOT a carried anchor: the mixture attends over EVERY prior
    block, which is why a pipeline rank boundary cannot avoid shipping the
    whole stack. Applied twice per layer (pre-attention and pre-MLP) with
    independent (proj, norm) pairs.
    """
    v = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)  # [T, nb+1, H]
    vf = v.to(torch.float32)
    var = vf.pow(2).mean(-1, keepdim=True)
    k = vf * torch.rsqrt(var + eps)
    score_w = norm_w.to(torch.float32) * proj_w.squeeze(0).to(torch.float32)
    scores = (k * score_w).sum(-1)                       # [T, nb+1]
    probs = scores.softmax(-1).unsqueeze(1)              # [T, 1, nb+1]
    return torch.matmul(probs, vf).squeeze(1)            # [T, H]


# --------------------------------------------------------------------------
# KDA — Kimi Delta Attention
# --------------------------------------------------------------------------


def short_conv(x, weight, state=None):
    """Causal depthwise conv1d (kernel K) + silu, with a running state.

    x:      [T, D]
    weight: [D, K]   (fla `ShortConvolution`, groups=D, no bias here)
    state:  [D, K-1] previous inputs, or None (zeros)

    Returns (y [T, D], new_state [D, K-1]).

    NOTE — state layout differs from fla on purpose. fla's `ShortConvolution`
    caches `[D, W]` with `W == kernel_size`, i.e. a rolling window that
    INCLUDES the current token (`cache[..., -1] = x`; `y = (cache*w).sum(-1)`).
    We carry the `K-1` PREVIOUS inputs instead and left-pad. The convolution
    result is identical; only the buffer convention differs. The Rust port owns
    its own layout and is validated against this reference, not against fla's
    cache tensor — do not assume the two buffers are interchangeable.
    """
    t, d = x.shape
    kk = weight.shape[-1]
    if state is None:
        state = x.new_zeros(d, kk - 1)
    # left-pad with the carried state so position 0 sees the previous tokens
    padded = torch.cat([state.t(), x.to(torch.float32)], dim=0)   # [K-1+T, D]
    out = x.new_zeros(t, d, dtype=torch.float32)
    for i in range(kk):
        out += padded[i:i + t] * weight[:, i].to(torch.float32).unsqueeze(0)
    new_state = padded[-(kk - 1):].t().contiguous() if kk > 1 else state
    return _bf16(F.silu(out)), new_state


def kda_gate(g_raw, a_log, dt_bias, lower_bound: float | None):
    """KDA decay gate (`kda_gate.py`).

    g_raw:   [T, H, D]  (from f_b_proj(f_a_proj(x)))
    a_log:   [H]
    dt_bias: [H*D]
    K3 sets `gate_lower_bound = -5.0`, which selects the lower-bound branch:
        g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
    The default branch (no lower bound) would be
        g = -exp(A_log) * softplus(g_raw + dt_bias)
    """
    t, h, d = g_raw.shape
    g = g_raw.to(torch.float32) + dt_bias.to(torch.float32).view(h, d).unsqueeze(0)
    a = a_log.to(torch.float32).exp().view(1, h, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * g)
    return -a * F.softplus(g)


def kda_recurrence(q, k, v, g, beta, state=None):
    """Gated delta rule (`kda_naive.naive_recurrent_kda`), fp32.

    q, k: [T, H, K] (already L2-normalised; q already scaled)
    v:    [T, H, V]
    g:    [T, H, K] log-space per-dim decay
    beta: [T, H]
    state:[H, K, V] or None

    Returns (o [T, H, V], final_state [H, K, V]).
    """
    t, h, kd = q.shape
    vd = v.shape[-1]
    q, k, v, g, beta = (z.to(torch.float32) for z in (q, k, v, g, beta))
    s = q.new_zeros(h, kd, vd) if state is None else state.to(torch.float32).clone()
    o = q.new_zeros(t, h, vd)
    for i in range(t):
        q_i, k_i, v_i, g_i, b_i = q[i], k[i], v[i], g[i], beta[i]
        s = s * g_i.unsqueeze(-1).exp()                       # [H, K, V]
        # delta rule: S += (beta*k) (x) (v - S^T k)
        sk = (k_i.unsqueeze(-1) * s).sum(-2)                  # [H, V]  = S^T k
        s = s + torch.einsum("hk,hv->hkv", b_i.unsqueeze(-1) * k_i, v_i - sk)
        o[i] = torch.einsum("hk,hkv->hv", q_i, s)
    return o, s


def kda_ref(x, w, cfg, state=None, conv_state=None):
    """One KDA layer. `w` is a dict of weights, `cfg` of dims.

    Returns (out [T, hidden], new_state, new_conv_state).
    """
    nh, hd, eps = cfg["num_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    t = x.shape[0]

    q = _lin(x, w["q_proj"])
    k = _lin(x, w["k_proj"])
    v = _lin(x, w["v_proj"])
    cs = conv_state or {}
    q, csq = short_conv(q, w["q_conv1d"], cs.get("q"))
    k, csk = short_conv(k, w["k_conv1d"], cs.get("k"))
    v, csv = short_conv(v, w["v_conv1d"], cs.get("v"))

    g_raw = _lin(_lin(x, w["f_a_proj"]), w["f_b_proj"]).view(t, nh, hd)
    g = kda_gate(g_raw, w["A_log"], w["dt_bias"], cfg.get("gate_lower_bound"))
    beta = torch.sigmoid(_lin(x, w["b_proj"]).to(torch.float32))      # [T, H]

    q = q.view(t, nh, hd)
    k = k.view(t, nh, hd)
    v = v.view(t, nh, hd)
    # use_qk_l2norm_in_kernel=True, then the kernel's scale = K**-0.5
    q = F.normalize(q.to(torch.float32), p=2, dim=-1) * (hd ** -0.5)
    k = F.normalize(k.to(torch.float32), p=2, dim=-1)

    o, new_state = kda_recurrence(q, k, v, g, beta, state)

    # FusedRMSNormGated(head_dim, activation='sigmoid'): rmsnorm(o)*w * sigmoid(g)
    g_out = _lin(x, w["g_proj"]).view(t, nh, hd)
    o = _rms(o, w["o_norm"], eps) * torch.sigmoid(g_out.to(torch.float32))

    out = _lin(_bf16(o).reshape(t, nh * hd), w["o_proj"])
    return out, new_state, {"q": csq, "k": csk, "v": csv}


# --------------------------------------------------------------------------
# Gated NoPE MLA
# --------------------------------------------------------------------------


def mla_ref(x, w, cfg, kv_cache=None):
    """One full-attention (MLA) layer — NoPE, with an output gate.

    `mla_use_nope=True` and `rotary_emb is None`: the `qk_rope_head_dim` slice
    still exists dimensionally but passes through UNROTATED, and its key half
    is MQA-shared (one copy per token, broadcast over all heads).

    This is the naive (materialised k/v) form, matching upstream. The Rust port
    uses glm5-style absorbed-latent decode (576 f/token) instead, which is
    mathematically equivalent and the only memory-feasible form at long ctx.

    kv_cache: dict with 'k' [P, H, qk_head] and 'v' [P, H, v_head], or None.
    Returns (out [T, hidden], new_cache).
    """
    nh = cfg["num_heads"]
    nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
    vhd, kv_lora, eps = cfg["v_head_dim"], cfg["kv_lora_rank"], cfg["rms_norm_eps"]
    qh = nope + rope
    t = x.shape[0]

    q = _lin(_rms(_lin(x, w["q_a_proj"]), w["q_a_layernorm"], eps), w["q_b_proj"])
    q = q.view(t, nh, qh)

    ckv = _lin(x, w["kv_a_proj_with_mqa"])                 # [T, kv_lora + rope]
    k_lat, k_rot = ckv[:, :kv_lora], ckv[:, kv_lora:]
    kp = _lin(_rms(k_lat, w["kv_a_layernorm"], eps), w["kv_b_proj"])
    kp = kp.view(t, nh, nope + vhd)
    k_nope, v = kp[..., :nope], kp[..., nope:]
    # k_rot is MQA-shared: [T, rope] broadcast across every head
    k = torch.cat([k_nope, k_rot.unsqueeze(1).expand(t, nh, rope)], dim=-1)

    if kv_cache is not None and kv_cache["k"].shape[0] > 0:
        k = torch.cat([kv_cache["k"], k], dim=0)
        v = torch.cat([kv_cache["v"], v], dim=0)
    new_cache = {"k": k, "v": v}

    p = k.shape[0]
    scale = qh ** -0.5
    # [H, T, P]
    scores = torch.einsum("thd,phd->htp", q.to(torch.float32), k.to(torch.float32)) * scale
    # causal: query i (absolute p-t+i) may see keys 0..p-t+i
    qpos = torch.arange(p - t, p, device=x.device).unsqueeze(-1)
    mask = torch.arange(p, device=x.device).unsqueeze(0) > qpos
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    probs = scores.softmax(-1)
    ctx = torch.einsum("htp,phd->thd", probs, v.to(torch.float32))   # [T, H, v_head]

    o = _bf16(ctx).reshape(t, nh * vhd)
    o = o * torch.sigmoid(_lin(x, w["g_proj"]).to(torch.float32))    # output gate
    return _lin(_bf16(o), w["o_proj"]), new_cache


# --------------------------------------------------------------------------
# LatentMoE block  (upstream `KimiSparseMoeBlock`)
# --------------------------------------------------------------------------


def latent_moe_ref(x, w, cfg):
    """Routed experts in a 3584-dim latent + 2 shared experts on hidden.

    The gate reads HIDDEN (7168), not the latent. The RMSNorm is applied to the
    COMBINED routed output, before the up-projection.
    """
    beta, lbeta = cfg["situ_beta"], cfg["situ_linear_beta"]
    eps = cfg["rms_norm_eps"]

    logits = x.to(torch.float32) @ w["gate_weight"].to(torch.float32).t()
    idx, wt = moe_gate(logits, w["e_score_correction_bias"],
                       cfg["top_k"], cfg["routed_scaling_factor"],
                       cfg["moe_renormalize"])

    x_lat = _lin(x, w["routed_expert_down_proj"])          # 7168 -> 3584
    y = torch.zeros_like(x_lat, dtype=torch.float32)
    for i in range(x.shape[0]):
        for j in range(idx.shape[1]):
            e = int(idx[i, j])
            ew = w["experts"][e]
            y[i] += wt[i, j] * mlp_ref(x_lat[i:i + 1], ew["w1"], ew["w3"], ew["w2"],
                                       beta, lbeta)[0].to(torch.float32)
    y = _bf16(y)
    if cfg.get("latent_moe_use_norm", True):
        y = _rms(y, w["routed_expert_norm"], eps)
    y = _lin(y, w["routed_expert_up_proj"])                # 3584 -> 7168

    shared = mlp_ref(x, w["shared_w1"], w["shared_w3"], w["shared_w2"], beta, lbeta)
    return _bf16(y.to(torch.float32) + shared.to(torch.float32))


# --------------------------------------------------------------------------
# Decoder layer + full model
# --------------------------------------------------------------------------


def is_kda_layer(cfg, idx: int) -> bool:
    """Layer 0 appears in neither `kda_layers` nor `full_attn_layers`, so it
    falls through to MLA. (`full_attn_layers` also lists 93, which is out of
    range for `num_hidden_layers=93` and simply unused.) 69 KDA + 24 MLA = 93.
    """
    return idx in cfg["kda_layers"]


def layer_ref(x, block_residual, w, cfg, idx, state=None, conv_state=None,
              kv_cache=None):
    """One decoder layer with Attention Residuals (`_forward_attn_residual`).

    Returns (prefix_sum [T,H], block_residual [T,nb,H], state, conv_state, kv).
    Note the layer's output IS the prefix sum, not a plain residual stream.
    """
    eps = cfg["rms_norm_eps"]
    beta, lbeta = cfg["situ_beta"], cfg["situ_linear_beta"]
    prefix_sum = x

    if block_residual.shape[1] > 0:
        h = apply_attn_res(prefix_sum, block_residual,
                           w["attn_res_proj"], w["attn_res_norm"], eps)
    else:
        h = x

    if idx % cfg["attn_res_block_size"] == 0:
        block_residual = torch.cat(
            [block_residual, prefix_sum.unsqueeze(1)], dim=1)
        prefix_sum = None

    h = _rms(h, w["input_layernorm"], eps)
    if is_kda_layer(cfg, idx):
        h, state, conv_state = kda_ref(h, w["attn"], cfg, state, conv_state)
    else:
        h, kv_cache = mla_ref(h, w["attn"], cfg, kv_cache)

    prefix_sum = h if prefix_sum is None else _bf16(
        prefix_sum.to(torch.float32) + h.to(torch.float32))

    h = apply_attn_res(prefix_sum, block_residual,
                       w["mlp_res_proj"], w["mlp_res_norm"], eps)
    h = _rms(h, w["post_attention_layernorm"], eps)
    if idx >= cfg["first_k_dense_replace"]:
        h = latent_moe_ref(h, w["moe"], cfg)
    else:
        h = mlp_ref(h, w["w1"], w["w3"], w["w2"], beta, lbeta)

    prefix_sum = _bf16(prefix_sum.to(torch.float32) + h.to(torch.float32))
    return prefix_sum, block_residual, state, conv_state, kv_cache


def model_ref(tokens, w, cfg, caches=None):
    """embed -> layers -> output AttnRes -> final norm -> lm_head.

    `caches` is an optional per-layer dict for incremental decode; pass None
    for a fresh sequence. Returns (logits [T, vocab], caches).
    """
    eps = cfg["rms_norm_eps"]
    n = cfg["num_hidden_layers"]
    x = w["embed"][tokens].to(torch.float32)
    t, h = x.shape
    block_residual = x.new_zeros(t, 0, h)
    caches = caches or [{} for _ in range(n)]

    for i in range(n):
        c = caches[i]
        x, block_residual, c["state"], c["conv"], c["kv"] = layer_ref(
            x, block_residual, w["layers"][i], cfg, i,
            c.get("state"), c.get("conv"), c.get("kv"))

    # one final mixture over the block stack, then the output norm
    x = apply_attn_res(x, block_residual,
                       w["output_attn_res_proj"], w["output_attn_res_norm"], eps)
    x = _rms(x, w["norm"], eps)
    return _lin(x, w["lm_head"]), caches
