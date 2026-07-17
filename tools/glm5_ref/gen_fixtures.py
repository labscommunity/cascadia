#!/usr/bin/env python3
"""Generate golden fixtures for the Rust GLM-5.2 shell primitives.

Mirrors `tools/deepseek_v4_ref/gen_fixtures.py`: deterministic synthetic inputs
+ reference outputs, dumped as fixtures.safetensors + fixtures_meta.json under
crates/cascadia-engine-sparse-moe/tests/fixtures/glm5/ (gitignored; the Rust
goldens SKIP when absent).

Run:
  python tools/glm5_ref/gen_fixtures.py \
      --out crates/cascadia-engine-sparse-moe/tests/fixtures/glm5
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm5_ref.kernels_ref import (apply_rotary_emb, attention_ref, indexer_ref,
                                  layer_ref, model_ref, moe_gate, moe_ref,
                                  mtp_draft_ref, precompute_freqs_cis, swiglu_ref)

FX = {}
META = {}


def put(name, t):
    FX[name] = t.detach().to(torch.float32).contiguous()


def put_i32(name, t):
    FX[name] = t.detach().to(torch.int32).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(42)

    # ---- 1. MoE router gate: sigmoid + noaux_tc, top-8 of 32, scale 2.5 ----
    # (GLM-5.2 real dims are 256 experts / top-8; 32 keeps the fixture tiny
    #  while exercising the identical selection/normalise/scale contract.)
    T, E, TOP_K, SCALE = 5, 32, 8, 2.5
    logits = torch.randn(T, E, generator=g)
    bias = 0.3 * torch.randn(E, generator=g)             # e_score_correction_bias
    idx, weight = moe_gate(logits, bias, TOP_K, SCALE, norm_topk=True)
    put("gate.logits", logits)
    put("gate.bias", bias)
    put_i32("gate.idx", idx)
    put("gate.weight", weight)
    META["gate"] = {"T": T, "E": E, "top_k": TOP_K, "scale": SCALE, "norm_topk": True}

    # ---- 2. rope: interleaved partial RoPE, dim 64, theta 8e6, no YaRN ----
    # (GLM's qk_rope_head_dim = 64; rope_theta = 8e6; rope_type "default".)
    ROPE_DIM, ROPE_THETA = 64, 8.0e6
    SEQ, HEADS = 16, 2
    fc = precompute_freqs_cis(ROPE_DIM, SEQ, ROPE_THETA)      # [SEQ, 32] complex
    put("rope.freqs", torch.view_as_real(fc))                # [SEQ, 32, 2]
    xr = (0.5 * torch.randn(6, HEADS, ROPE_DIM, generator=g)).to(torch.bfloat16)
    put("rope.apply.in", xr)
    yr = apply_rotary_emb(xr, fc[:6].unsqueeze(1))           # broadcast over heads
    put("rope.apply.out", yr)
    META["rope"] = {"dim": ROPE_DIM, "theta": ROPE_THETA, "seq": SEQ, "heads": HEADS}

    # ---- 3. MLA attention: tiny but structurally faithful (nope/pe/v split,
    #         q/kv LoRA, kv_a/q_a RMSNorms, absorbed-vs-naive equivalence) ----
    acfg = {
        "hidden": 32, "n_heads": 3, "qk_nope": 6, "qk_rope": 4, "v_head": 6,
        "kv_lora": 8, "q_lora": 16, "eps": 1e-5, "theta": 8.0e6,
    }
    H, hid, nope, rp = acfg["n_heads"], acfg["hidden"], acfg["qk_nope"], acfg["qk_rope"]
    vh, kvl, ql = acfg["v_head"], acfg["kv_lora"], acfg["q_lora"]
    qk = nope + rp
    NS = 6  # sequence length (prefill + decode positions)

    def wbf(*shape, s=0.3):
        return (s * torch.randn(*shape, generator=g)).to(torch.bfloat16).to(torch.float32)

    aw = {
        "wq_a": wbf(ql, hid),
        "q_a_ln": (1.0 + 0.05 * torch.randn(ql, generator=g)).to(torch.bfloat16).to(torch.float32),
        "wq_b": wbf(H * qk, ql),
        "wkv_a": wbf(kvl + rp, hid),
        "kv_a_ln": (1.0 + 0.05 * torch.randn(kvl, generator=g)).to(torch.bfloat16).to(torch.float32),
        "wkv_b": wbf(H * (nope + vh), kvl),
        "wo": wbf(hid, H * vh),
    }
    ax = (0.5 * torch.randn(NS, hid, generator=g)).to(torch.bfloat16).to(torch.float32)
    aout = attention_ref(ax, aw, acfg)
    for k, v in aw.items():
        put(f"attn.{k}", v)
    put("attn.x", ax)
    put("attn.out", aout)
    META["attn"] = {**{k: acfg[k] for k in acfg}, "seq": NS}

    # ---- 4. DSA lightning indexer: score causal keys, select top-k ----
    icfg = {"hidden": 32, "q_lora": 16, "index_nh": 2, "index_hd": 8,
            "qk_rope": 4, "ln_eps": 1e-6, "theta": 8.0e6}
    inh, ihd, ird, iql, ihid = (icfg["index_nh"], icfg["index_hd"], icfg["qk_rope"],
                                icfg["q_lora"], icfg["hidden"])
    ISEQ, IQPOS, ITOPK = 6, 5, 3  # nk=6 > topk=3 -> selection active
    iw = {
        "ix_wq": wbf(inh * ihd, iql),
        "ix_wk": wbf(ihd, ihid),
        "ix_wp": wbf(inh, ihid),
        "k_norm_w": (1.0 + 0.05 * torch.randn(ihd, generator=g)).to(torch.bfloat16).to(torch.float32),
        "k_norm_b": (0.1 * torch.randn(ihd, generator=g)).to(torch.bfloat16).to(torch.float32),
    }
    ix = (0.5 * torch.randn(ISEQ, ihid, generator=g)).to(torch.bfloat16).to(torch.float32)
    iqr = (0.5 * torch.randn(ISEQ, iql, generator=g)).to(torch.bfloat16).to(torch.float32)
    isc, isel = indexer_ref(ix, iqr[IQPOS], iw, icfg, IQPOS, ITOPK)
    for k, v in iw.items():
        put(f"idx.{k}", v)
    put("idx.x", ix)
    put("idx.qr", iqr)
    put("idx.isc", isc)
    put_i32("idx.sel", isel)
    META["idx"] = {**{k: icfg[k] for k in icfg}, "seq": ISEQ, "query_pos": IQPOS, "topk": ITOPK}

    # ---- 5. SwiGLU FFN (expert / shared / dense MLP building block) ----
    FHID, FINT, FROWS = 32, 20, 4
    fwg, fwu = wbf(FINT, FHID), wbf(FINT, FHID)
    fwd = wbf(FHID, FINT)
    fx = (0.5 * torch.randn(FROWS, FHID, generator=g)).to(torch.bfloat16).to(torch.float32)
    fy = swiglu_ref(fx, fwg, fwu, fwd)
    put("ffn.wg", fwg)
    put("ffn.wu", fwu)
    put("ffn.wd", fwd)
    put("ffn.x", fx)
    put("ffn.out", fy)
    META["ffn"] = {"hidden": FHID, "inter": FINT, "rows": FROWS}

    # ---- 6. MoE block: router (sigmoid+noaux_tc) + top-k experts + shared ----
    MHID, MEXP, MTOPK, MINT, MSCALE, MROWS = 32, 4, 2, 10, 2.5, 3
    m_rw = wbf(MEXP, MHID)                                    # router weight
    m_rb = (0.3 * torch.randn(MEXP, generator=g)).to(torch.bfloat16).to(torch.float32)

    def expert_w():
        return (wbf(MINT, MHID), wbf(MINT, MHID), wbf(MHID, MINT))

    m_experts = [expert_w() for _ in range(MEXP)]
    m_shared = expert_w()
    mx = (0.5 * torch.randn(MROWS, MHID, generator=g)).to(torch.bfloat16).to(torch.float32)
    m_out = moe_ref(mx, m_rw, m_rb, m_experts, m_shared, MTOPK, MSCALE)

    put("moe.router_w", m_rw)
    put("moe.router_bias", m_rb)
    for e in range(MEXP):
        put(f"moe.e{e}.wg", m_experts[e][0])
        put(f"moe.e{e}.wu", m_experts[e][1])
        put(f"moe.e{e}.wd", m_experts[e][2])
    put("moe.sh.wg", m_shared[0])
    put("moe.sh.wu", m_shared[1])
    put("moe.sh.wd", m_shared[2])
    put("moe.x", mx)
    put("moe.out", m_out)
    META["moe"] = {"hidden": MHID, "n_experts": MEXP, "top_k": MTOPK,
                   "moe_inter": MINT, "scale": MSCALE, "rows": MROWS}

    # ---- 7. full transformer layer: norms + attn + residual + MoE + residual ----
    lcfg = {"hidden": 32, "n_heads": 3, "qk_nope": 6, "qk_rope": 4, "v_head": 6,
            "kv_lora": 8, "q_lora": 16, "eps": 1e-5, "theta": 8.0e6}
    lH, lhid, lnope, lrp = lcfg["n_heads"], lcfg["hidden"], lcfg["qk_nope"], lcfg["qk_rope"]
    lvh, lkvl, lql = lcfg["v_head"], lcfg["kv_lora"], lcfg["q_lora"]
    lqk = lnope + lrp
    lE, lTOPK, lINT, lSCALE, lSEQ = 4, 2, 10, 2.5, 5

    def norm_w(n):
        return (1.0 + 0.05 * torch.randn(n, generator=g)).to(torch.bfloat16).to(torch.float32)

    l_in_ln, l_post_ln = norm_w(lhid), norm_w(lhid)
    l_aw = {
        "wq_a": wbf(lql, lhid), "q_a_ln": norm_w(lql), "wq_b": wbf(lH * lqk, lql),
        "wkv_a": wbf(lkvl + lrp, lhid), "kv_a_ln": norm_w(lkvl),
        "wkv_b": wbf(lH * (lnope + lvh), lkvl), "wo": wbf(lhid, lH * lvh),
    }
    l_rw, l_rb = wbf(lE, lhid), (0.3 * torch.randn(lE, generator=g)).to(torch.bfloat16).to(torch.float32)
    l_experts = [(wbf(lINT, lhid), wbf(lINT, lhid), wbf(lhid, lINT)) for _ in range(lE)]
    l_shared = (wbf(lINT, lhid), wbf(lINT, lhid), wbf(lhid, lINT))
    lx = (0.5 * torch.randn(lSEQ, lhid, generator=g)).to(torch.bfloat16).to(torch.float32)
    l_out, l_hmid = layer_ref(lx, l_in_ln, l_post_ln, l_aw, lcfg,
                              (l_rw, l_rb, l_experts, l_shared), (lTOPK, lSCALE), lcfg["eps"])
    put("layer.h_mid", l_hmid)

    put("layer.in_ln", l_in_ln)
    put("layer.post_ln", l_post_ln)
    for k, v in l_aw.items():
        put(f"layer.attn.{k}", v)
    put("layer.moe.router_w", l_rw)
    put("layer.moe.router_bias", l_rb)
    for e in range(lE):
        put(f"layer.moe.e{e}.wg", l_experts[e][0])
        put(f"layer.moe.e{e}.wu", l_experts[e][1])
        put(f"layer.moe.e{e}.wd", l_experts[e][2])
    put("layer.moe.sh.wg", l_shared[0])
    put("layer.moe.sh.wu", l_shared[1])
    put("layer.moe.sh.wd", l_shared[2])
    put("layer.x", lx)
    put("layer.out", l_out)
    META["layer"] = {**{k: lcfg[k] for k in lcfg}, "n_experts": lE, "top_k": lTOPK,
                     "moe_inter": lINT, "scale": lSCALE, "seq": lSEQ}

    # ---- 8. full tiny model: embed -> layers (dense + MoE) -> norm -> lm_head,
    #         greedy generation (token-exact parity target) ----
    mcfg = {"n_heads": 3, "qk_nope": 6, "qk_rope": 4, "v_head": 6, "kv_lora": 8,
            "q_lora": 16, "eps": 1e-5, "theta": 8.0e6, "top_k": 2, "scale": 2.5}
    mH, mhH = mcfg["n_heads"], 32                            # heads, hidden
    mnope, mrp, mvh = mcfg["qk_nope"], mcfg["qk_rope"], mcfg["v_head"]
    mkvl, mql = mcfg["kv_lora"], mcfg["q_lora"]
    mqk = mnope + mrp
    VOCAB, NLAYERS, FIRST_DENSE, DINT = 16, 3, 1, 12
    mE, mINT = 4, 10
    PROMPT, NGEN = [1, 2, 3, 4], 4

    def attn_w():
        return {
            "wq_a": wbf(mql, mhH), "q_a_ln": norm_w(mql), "wq_b": wbf(mH * mqk, mql),
            "wkv_a": wbf(mkvl + mrp, mhH), "kv_a_ln": norm_w(mkvl),
            "wkv_b": wbf(mH * (mnope + mvh), mkvl), "wo": wbf(mhH, mH * mvh),
        }

    def dense_w():
        return (wbf(DINT, mhH), wbf(DINT, mhH), wbf(mhH, DINT))

    def moe_w():
        rw = wbf(mE, mhH)
        rb = (0.3 * torch.randn(mE, generator=g)).to(torch.bfloat16).to(torch.float32)
        exps = [(wbf(mINT, mhH), wbf(mINT, mhH), wbf(mhH, mINT)) for _ in range(mE)]
        sh = (wbf(mINT, mhH), wbf(mINT, mhH), wbf(mhH, mINT))
        return (rw, rb, exps, sh)

    m_embed = wbf(VOCAB, mhH)
    m_final_norm = norm_w(mhH)
    m_lm_head = wbf(VOCAB, mhH)
    m_layers = []
    for li in range(NLAYERS):
        aw = attn_w()
        if li < FIRST_DENSE:
            m_layers.append({"in_ln": norm_w(mhH), "post_ln": norm_w(mhH),
                             "attn": aw, "kind": "dense", "mlp": dense_w()})
        else:
            m_layers.append({"in_ln": norm_w(mhH), "post_ln": norm_w(mhH),
                             "attn": aw, "kind": "moe", "mlp": moe_w()})
    greedy = model_ref(mcfg,
                       {"embed": m_embed, "final_norm": m_final_norm,
                        "lm_head": m_lm_head, "layers": m_layers},
                       PROMPT, NGEN)

    put("model.embed", m_embed)
    put("model.final_norm", m_final_norm)
    put("model.lm_head", m_lm_head)
    for li, L in enumerate(m_layers):
        put(f"model.L{li}.in_ln", L["in_ln"])
        put(f"model.L{li}.post_ln", L["post_ln"])
        for k, v in L["attn"].items():
            put(f"model.L{li}.attn.{k}", v)
        if L["kind"] == "dense":
            wg, wu, wd = L["mlp"]
            put(f"model.L{li}.dense.wg", wg)
            put(f"model.L{li}.dense.wu", wu)
            put(f"model.L{li}.dense.wd", wd)
        else:
            rw, rb, exps, sh = L["mlp"]
            put(f"model.L{li}.moe.router_w", rw)
            put(f"model.L{li}.moe.router_bias", rb)
            for e in range(mE):
                put(f"model.L{li}.moe.e{e}.wg", exps[e][0])
                put(f"model.L{li}.moe.e{e}.wu", exps[e][1])
                put(f"model.L{li}.moe.e{e}.wd", exps[e][2])
            put(f"model.L{li}.moe.sh.wg", sh[0])
            put(f"model.L{li}.moe.sh.wu", sh[1])
            put(f"model.L{li}.moe.sh.wd", sh[2])
    META["model"] = {**{k: mcfg[k] for k in mcfg}, "vocab": VOCAB, "hidden": mhH,
                     "n_layers": NLAYERS, "first_dense": FIRST_DENSE, "dense_inter": DINT,
                     "n_experts": mE, "moe_inter": mINT, "prompt": PROMPT,
                     "n_gen": NGEN, "greedy": greedy}

    # ---- 9. MTP head: DeepSeek-V3 draft chain (embed+norms+eh_proj+block+lm_head) ----
    G_MTP, MTP_TOK = 3, 5
    mtp_hlast = (0.5 * torch.randn(mhH, generator=g)).to(torch.bfloat16).to(torch.float32)
    mtp_embed = wbf(VOCAB, mhH)
    mtp_lm_head = wbf(VOCAB, mhH)
    mtp_eh_proj = wbf(mhH, 2 * mhH)
    mtp_enorm, mtp_hnorm = norm_w(mhH), norm_w(mhH)
    mtp_final_norm, mtp_mtp_norm = norm_w(mhH), norm_w(mhH)
    mtp_attn = attn_w()
    mtp_rw = wbf(mE, mhH)
    mtp_rb = (0.3 * torch.randn(mE, generator=g)).to(torch.bfloat16).to(torch.float32)
    mtp_exps = [(wbf(mINT, mhH), wbf(mINT, mhH), wbf(mhH, mINT)) for _ in range(mE)]
    mtp_sh = (wbf(mINT, mhH), wbf(mINT, mhH), wbf(mhH, mINT))
    mtp_in_ln, mtp_post_ln = norm_w(mhH), norm_w(mhH)
    mtp_block = {"in_ln": mtp_in_ln, "post_ln": mtp_post_ln, "attn": mtp_attn,
                 "rw": mtp_rw, "rb": mtp_rb, "experts": mtp_exps, "shared": mtp_sh}
    mtp_w = {"embed": mtp_embed, "enorm": mtp_enorm, "hnorm": mtp_hnorm,
             "final_norm": mtp_final_norm, "mtp_norm": mtp_mtp_norm,
             "eh_proj": mtp_eh_proj, "lm_head": mtp_lm_head, "mtp_block": mtp_block}
    mtp_drafts = mtp_draft_ref(mtp_hlast, MTP_TOK, G_MTP, mtp_w, mcfg)

    put("mtp.hlast", mtp_hlast)
    put("mtp.embed", mtp_embed)
    put("mtp.lm_head", mtp_lm_head)
    put("mtp.eh_proj", mtp_eh_proj)
    put("mtp.enorm", mtp_enorm)
    put("mtp.hnorm", mtp_hnorm)
    put("mtp.final_norm", mtp_final_norm)
    put("mtp.mtp_norm", mtp_mtp_norm)
    put("mtp.block.in_ln", mtp_in_ln)
    put("mtp.block.post_ln", mtp_post_ln)
    for k, v in mtp_attn.items():
        put(f"mtp.block.attn.{k}", v)
    put("mtp.block.moe.router_w", mtp_rw)
    put("mtp.block.moe.router_bias", mtp_rb)
    for e in range(mE):
        put(f"mtp.block.moe.e{e}.wg", mtp_exps[e][0])
        put(f"mtp.block.moe.e{e}.wu", mtp_exps[e][1])
        put(f"mtp.block.moe.e{e}.wd", mtp_exps[e][2])
    put("mtp.block.moe.sh.wg", mtp_sh[0])
    put("mtp.block.moe.sh.wu", mtp_sh[1])
    put("mtp.block.moe.sh.wd", mtp_sh[2])
    META["mtp"] = {"next_tok": MTP_TOK, "g": G_MTP, "drafts": mtp_drafts,
                   "vocab": VOCAB, "hidden": mhH}

    from safetensors.torch import save_file
    save_file(FX, str(out / "fixtures.safetensors"))
    (out / "fixtures_meta.json").write_text(json.dumps(META, indent=2))
    total = sum(t.numel() * t.element_size() for t in FX.values())
    print(f"[glm5 fixtures] {len(FX)} tensors, {total/1e6:.3f} MB -> {out}")

    # ---- 10. loader round-trip: write a tiny export, generate via the export
    #          reader (the Rust glm::loader must reproduce these greedy tokens) ----
    import export_glm5
    from glm5_ref.load_export import load_export_and_generate
    export_dir = out.parent / "glm5_export"
    export_glm5.export_tiny(export_dir)
    rt_prompt, rt_ngen = [1, 2, 3, 4], 4
    rt_greedy = load_export_and_generate(export_dir, rt_prompt, rt_ngen)
    print(f"[loader round-trip] export {export_dir} prompt {rt_prompt} greedy {rt_greedy}")


if __name__ == "__main__":
    main()
