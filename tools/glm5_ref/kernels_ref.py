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


def swiglu_ref(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor,
               wd: torch.Tensor) -> torch.Tensor:
    """GLM/DeepSeek SwiGLU FFN: down(silu(gate·x) * up·x), no bias — used by
    routed experts, the shared expert, and the dense first-k layers
    (`silu(g)·u` then down). bf16 after each linear; the
    silu*up product stays f32."""
    g = _lin(x, wg)
    u = _lin(x, wu)
    h = torch.nn.functional.silu(g) * u
    return _lin(h, wd)


def _lin_f32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w^T with f32 output (no bf16 round) — the router path (a plain
    f32 matmul for the gate logits, `dsv4::math::linear_f32`)."""
    return x.to(torch.float32) @ w.to(torch.float32).t()


def moe_ref(x, rw, rb, experts, shared, top_k, scale):
    """GLM MoE block: out = sum_{i in topk} w_i * expert_i(x) + shared(x).
    Router logits are f32 (no bf16), gate is sigmoid + noaux_tc (`moe_gate`);
    routed weights already carry `routed_scaling_factor`. Accumulation order:
    routed experts in gate order, then the shared expert.

    x:[S,hidden]; rw:[E,hidden]; rb:[E]; experts: list of (wg,wu,wd);
    shared: (wg,wu,wd). Returns [S,hidden]."""
    S = x.shape[0]
    logits = _lin_f32(x, rw)                                 # [S, E]
    idx, w = moe_gate(logits, rb, top_k, scale, norm_topk=True)
    out = torch.zeros_like(x.to(torch.float32))
    for s in range(S):
        acc = torch.zeros(x.shape[1], dtype=torch.float32)
        for j in range(top_k):
            e = int(idx[s, j])
            acc = acc + float(w[s, j]) * swiglu_ref(x[s], *experts[e])
        acc = acc + swiglu_ref(x[s], *shared)
        out[s] = acc
    return out


def layer_ref(x, in_ln, post_ln, aw, acfg, moe, mcfg, eps):
    """One GLM transformer layer (pre-norm):
        h   = x + attention(rmsnorm(x, input_layernorm))
        out = h + mlp(rmsnorm(h, post_attention_layernorm))
    `mlp` is the MoE block here. `moe = (router_w, router_bias, experts,
    shared)`, `mcfg = (top_k, scale)`. RMSNorm eps = rms_norm_eps."""
    nrm = _rms(x, in_ln, eps)
    h = x.to(torch.float32) + attention_ref_absorbed(nrm, aw, acfg)
    nrm2 = _rms(h, post_ln, eps)
    rw, rb, experts, shared = moe
    top_k, scale = mcfg
    out = h + moe_ref(nrm2, rw, rb, experts, shared, top_k, scale)
    return out, h  # h = post-attention residual (diagnostic)


def model_ref(cfg, w, prompt, n_gen):
    """Full GLM-5.2 tiny model, greedy generation — the parity target for the
    Rust `GlmModel`. Stateless full-sequence recompute per step (equivalent to
    incremental KV decode since absorbed attention == naive causal).

    cfg: attention dims (n_heads, qk_nope, qk_rope, v_head, kv_lora, q_lora,
    eps, theta) + top_k, scale. w: {'embed'[V,H], 'final_norm'[H], 'lm_head'
    [V,H], 'layers': [ {in_ln, post_ln, attn(dict), kind:'dense'|'moe', mlp} ]}
    where dense mlp = (wg,wu,wd) and moe mlp = (router_w, router_bias, experts,
    shared). lm_head logits are f32 (argmax). Returns the `n_gen` greedy ids."""
    eps, top_k, scale = cfg["eps"], cfg["top_k"], cfg["scale"]

    def forward_seq(tokens):
        x = w["embed"][torch.tensor(tokens, dtype=torch.long)].to(torch.float32)  # [S,H]
        for L in w["layers"]:
            nrm = _rms(x, L["in_ln"], eps)
            x = x + attention_ref_absorbed(nrm, L["attn"], cfg)
            nrm2 = _rms(x, L["post_ln"], eps)
            if L["kind"] == "dense":
                wg, wu, wd = L["mlp"]
                x = x + swiglu_ref(nrm2, wg, wu, wd)
            else:
                rw, rb, experts, shared = L["mlp"]
                x = x + moe_ref(nrm2, rw, rb, experts, shared, top_k, scale)
        x = _rms(x, w["final_norm"], eps)
        return _lin_f32(x, w["lm_head"])                     # [S, vocab]

    tokens = list(prompt)
    gen = []
    for _ in range(n_gen):
        logits = forward_seq(tokens)
        nxt = int(logits[-1].argmax())
        tokens.append(nxt)
        gen.append(nxt)
    return gen


def mtp_draft_ref(hlast, next_tok, G, w, cfg):
    """GLM-5.2 MTP (multi-token-prediction) head — DeepSeek-V3 draft chain.
    Given the true pre-norm hidden `hlast` at the last position
    and the just-emitted `next_tok`, propose `G` greedy draft tokens:

        x  = rmsnorm(embed[tok], enorm)
        h  = rmsnorm(final_norm(hlast) if g==0 else h_prev, hnorm)
        hx = eh_proj @ [x ; h]                    # [2H] -> [H], block input
        hx = MTP_block(hx)                        # full transformer layer, own causal KV
        draft = argmax(lm_head @ rmsnorm(hx, mtp_norm)); tok = draft; h = hx

    `w` keys: embed[V,H], enorm[H], hnorm[H], final_norm[H], mtp_norm[H],
    eh_proj[H,2H], lm_head[V,H], and `mtp_block` = {in_ln, post_ln, attn(dict),
    rw, rb, experts, shared}. `cfg` = attention dims + top_k, scale, eps.
    The block is recomputed over the growing draft sequence (causal-equivalent
    to the incremental KV the Rust head uses). Returns the `G` draft ids."""
    eps, top_k, scale = cfg["eps"], cfg["top_k"], cfg["scale"]
    blk = w["mtp_block"]
    h = hlast.to(torch.float32).clone()
    tok = next_tok
    drafts, hx_seq = [], []
    for g in range(G):
        x = _rms(w["embed"][tok].to(torch.float32).unsqueeze(0), w["enorm"], eps).squeeze(0)
        if g == 0:
            h = _rms(h.unsqueeze(0), w["final_norm"], eps).squeeze(0)
        h = _rms(h.unsqueeze(0), w["hnorm"], eps).squeeze(0)
        hx = _lin(torch.cat([x, h]).unsqueeze(0), w["eh_proj"]).squeeze(0)   # [H]
        hx_seq.append(hx)
        out, _ = layer_ref(torch.stack(hx_seq), blk["in_ln"], blk["post_ln"],
                           blk["attn"], cfg, (blk["rw"], blk["rb"], blk["experts"], blk["shared"]),
                           (top_k, scale), eps)
        hx_post = out[-1]
        row = _rms(hx_post.unsqueeze(0), w["mtp_norm"], eps).squeeze(0)
        logit = _lin_f32(row.unsqueeze(0), w["lm_head"]).squeeze(0)
        d = int(logit.argmax())
        drafts.append(d)
        tok = d
        h = hx_post
    return drafts


def _layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """Classic LayerNorm (mean + population variance, weight + bias),
    bf16-rounded output. Used by the DSA indexer's k-norm (eps 1e-6)."""
    mu = x.to(torch.float32).mean(dim=-1, keepdim=True)
    var = ((x.to(torch.float32) - mu) ** 2).mean(dim=-1, keepdim=True)
    return _bf16((x.to(torch.float32) - mu) * torch.rsqrt(var + eps) * w + b)


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


def attention_ref_absorbed(x: torch.Tensor, w: Dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    """GLM-5.2 MLA attention, ABSORBED-latent form — mirrors the Rust shell's
    decode path operand-for-operand (differs from `attention_ref` only in
    accumulation order, ULP). Use this when composing a full layer so the
    attention output matches the shell tightly and doesn't perturb the router;
    `attention_ref` (naive) remains the independent check that absorbed==naive.
    """
    H, nope, rope = cfg["n_heads"], cfg["qk_nope"], cfg["qk_rope"]
    vh, kvl, eps, theta = cfg["v_head"], cfg["kv_lora"], cfg["eps"], cfg["theta"]
    qk = nope + rope
    S = x.shape[0]
    scale = 1.0 / math.sqrt(qk)
    fc = precompute_freqs_cis(rope, S, theta)

    qr = _rms(_lin(x, w["wq_a"]), w["q_a_ln"], eps)
    q = _lin(qr, w["wq_b"]).reshape(S, H, qk)
    comp = _lin(x, w["wkv_a"])
    lat = _rms(comp[:, :kvl], w["kv_a_ln"], eps)                 # Lc [S,kvl]
    kpe = apply_rotary_emb(comp[:, kvl:], fc)                    # Rc [S,rope]
    qpe = apply_rotary_emb(q[:, :, nope:qk], fc.unsqueeze(1))    # [S,H,rope]
    qnope = q[:, :, :nope]                                       # [S,H,nope]
    wkv_b = w["wkv_b"].to(torch.float32)                         # [H*(nope+vh), kvl]

    ctx = torch.zeros(S, H, vh, dtype=torch.float32)
    for s in range(S):
        for h in range(H):
            rbase = h * (nope + vh)
            w_uk = wkv_b[rbase:rbase + nope, :]                  # [nope, kvl]
            w_uv = wkv_b[rbase + nope:rbase + nope + vh, :]      # [vh, kvl]
            qabs = qnope[s, h] @ w_uk                            # [kvl]
            sc = torch.empty(s + 1, dtype=torch.float32)
            for t in range(s + 1):
                sc[t] = (torch.dot(qabs, lat[t]) + torch.dot(qpe[s, h], kpe[t])) * scale
            p = torch.softmax(sc, dim=0)
            clat = (p.unsqueeze(-1) * lat[: s + 1]).sum(0)       # [kvl]
            ctx[s, h] = w_uv @ clat                              # [vh]

    return _lin(ctx.reshape(S, H * vh), w["wo"])


def _rope_first(v: torch.Tensor, fc_pos: torch.Tensor, rd: int) -> torch.Tensor:
    """Rope only the FIRST `rd` dims of `v`'s last axis (the indexer ropes the
    qk_rope prefix of each head's index_head_dim), leaving the rest untouched.
    `fc_pos` is the [rd/2] complex table row for this position."""
    head = apply_rotary_emb(v[..., :rd], fc_pos)
    return torch.cat([head, v[..., rd:]], dim=-1)


def indexer_ref(x, qr_q, w, cfg, query_pos: int, topk: int):
    """GLM-5.2 DSA lightning indexer — score all causal keys for one query and
    return the top-`keep` selection (raw positions).

    Per key t: kd = rope_first(layernorm(ix_wk·x[t], eps=1e-6), qk_rope).
    Query: qi = rope_first(ix_wq·qr_q, qk_rope) per index head; w32 = ix_wp·x[q].
    Score: isc[t] = (1/sqrt(nh))·Σ_h w32[h]·ReLU((1/sqrt(hd))·qi[h]·kd[t]).
    Select keep = min(query_pos+1, topk): threshold = keep-th largest, emit t in
    ASCENDING position order (>thr first, then ==thr) — the two-pass form.

    x:[S,hidden], qr_q:[q_lora] (the RMSNorm'd q_a residual at the query pos).
    Numeric contract matches the Rust shell: bf16 after each linear/LN/rope; the
    score dots/ReLU/scale stay f32. Returns (isc:[query_pos+1], sel:[keep] i32).
    """
    nh, hd, rd = cfg["index_nh"], cfg["index_hd"], cfg["qk_rope"]
    eps, theta = cfg["ln_eps"], cfg["theta"]
    S = x.shape[0]
    fc = precompute_freqs_cis(rd, S, theta)                      # [S, rd/2]

    ic = torch.zeros(S, hd, dtype=torch.float32)
    for t in range(S):
        kd = _layernorm(_lin(x[t], w["ix_wk"]), w["k_norm_w"], w["k_norm_b"], eps)
        ic[t] = _rope_first(kd, fc[t], rd)

    qi = _rope_first(_lin(qr_q, w["ix_wq"]).reshape(nh, hd), fc[query_pos], rd)  # [nh,hd]
    w32 = _lin(x[query_pos], w["ix_wp"])                         # [nh]

    nk = query_pos + 1
    wsc, rs = 1.0 / math.sqrt(nh), 1.0 / math.sqrt(hd)
    isc = torch.zeros(nk, dtype=torch.float32)
    for t in range(nk):
        a = 0.0
        for h in range(nh):
            d0 = float(torch.dot(qi[h], ic[t])) * rs
            if d0 > 0.0:
                a += float(w32[h]) * d0
        isc[t] = a * wsc

    keep = min(nk, topk)
    thr = torch.sort(isc, descending=True).values[keep - 1]
    sel = []
    for t in range(nk):
        if isc[t] > thr and len(sel) < keep:
            sel.append(t)
    for t in range(nk):
        if isc[t] == thr and len(sel) < keep:
            sel.append(t)
    return isc, torch.tensor(sel, dtype=torch.int32)
