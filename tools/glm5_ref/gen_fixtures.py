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
                                  moe_gate, moe_ref, precompute_freqs_cis,
                                  swiglu_ref)

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

    from safetensors.torch import save_file
    save_file(FX, str(out / "fixtures.safetensors"))
    (out / "fixtures_meta.json").write_text(json.dumps(META, indent=2))
    total = sum(t.numel() * t.element_size() for t in FX.values())
    print(f"[glm5 fixtures] {len(FX)} tensors, {total/1e6:.3f} MB -> {out}")


if __name__ == "__main__":
    main()
