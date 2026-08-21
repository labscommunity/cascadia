#!/usr/bin/env python3
"""Generate golden fixtures for the Rust Kimi-K3 shell primitives.

Mirrors `tools/glm5_ref/gen_fixtures.py`: deterministic synthetic inputs +
reference outputs, dumped as fixtures.safetensors + fixtures_meta.json under
crates/cascadia-engine-sparse-moe/tests/fixtures/kimi_k3/ (gitignored; the Rust
goldens SKIP when absent).

Run:
  python tools/kimi_k3_ref/gen_fixtures.py \
      --out crates/cascadia-engine-sparse-moe/tests/fixtures/kimi_k3
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kimi_k3_ref.kernels_ref import (apply_attn_res, kda_gate, kda_recurrence,
                                     latent_moe_ref, mla_ref, model_ref,
                                     moe_gate, short_conv, situ)
from kimi_k3_ref.tiny import tiny_cfg, tiny_weights

FX, META = {}, {}


def put(name, t):
    FX[name] = t.detach().to(torch.float32).contiguous()


def put_i32(name, t):
    FX[name] = t.detach().to(torch.int32).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)

    # ---- 1. SiTU -----------------------------------------------------
    B, D = 9, 48
    gate_up = torch.randn(B, 2 * D) * 3.0
    put("situ.in", gate_up)
    put("situ.out", situ(gate_up, 4.0, 25.0))
    META["situ"] = {"rows": B, "dim": D, "beta": 4.0, "linear_beta": 25.0}

    # ---- 2. MoE gate (sigmoid + noaux_tc) ----------------------------
    T, E, TK, SCALE = 7, 32, 6, 1.0
    logits = torch.randn(T, E)
    bias = torch.randn(E) * 0.1
    idx, wt = moe_gate(logits, bias, TK, SCALE, True)
    put("gate.logits", logits)
    put("gate.bias", bias)
    put_i32("gate.idx", idx)
    put("gate.weight", wt)
    META["gate"] = {"T": T, "E": E, "top_k": TK, "scale": SCALE,
                    "renormalize": True}

    # ---- 3. AttnRes mixture ------------------------------------------
    H, NB, EPS = 64, 4, 1e-5
    ps = torch.randn(T, H)
    br = torch.randn(T, NB, H)
    pw, nw = torch.randn(1, H), torch.randn(H)
    put("attnres.prefix_sum", ps)
    put("attnres.block_residual", br)
    put("attnres.proj_w", pw)
    put("attnres.norm_w", nw)
    put("attnres.out", apply_attn_res(ps, br, pw, nw, EPS))
    META["attnres"] = {"T": T, "H": H, "n_blocks": NB, "eps": EPS}

    # ---- 4. short conv (fresh + carried state) -----------------------
    CD, CK = 24, 4
    cx = torch.randn(T, CD)
    cw = torch.randn(CD, CK)
    y0, s0 = short_conv(cx, cw)
    y1, s1 = short_conv(cx, cw, s0)
    put("conv.x", cx)
    put("conv.weight", cw)
    put("conv.y0", y0)
    put("conv.state0", s0)
    put("conv.y1", y1)
    put("conv.state1", s1)
    META["conv"] = {"T": T, "D": CD, "K": CK}

    # ---- 5. KDA gate (lower-bound branch, as K3 uses) ----------------
    GH, GD, LB = 6, 32, -5.0
    g_raw = torch.randn(T, GH, GD)
    a_log = torch.rand(GH).uniform_(1, 16).log()
    dt_bias = torch.randn(GH * GD)
    put("kdagate.g_raw", g_raw)
    put("kdagate.a_log", a_log)
    put("kdagate.dt_bias", dt_bias)
    put("kdagate.out", kda_gate(g_raw, a_log, dt_bias, LB))
    META["kdagate"] = {"T": T, "H": GH, "D": GD, "lower_bound": LB}

    # ---- 6. KDA recurrence (gated delta rule) ------------------------
    RH, RK, RV = 4, 16, 16
    q = F.normalize(torch.randn(T, RH, RK), p=2, dim=-1) * (RK ** -0.5)
    k = F.normalize(torch.randn(T, RH, RK), p=2, dim=-1)
    v = torch.randn(T, RH, RV)
    g = -torch.rand(T, RH, RK)
    beta = torch.rand(T, RH)
    st_in = torch.randn(RH, RK, RV) * 0.1
    o, st_out = kda_recurrence(q, k, v, g, beta, st_in.clone())
    for n, t in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
                 ("state_in", st_in), ("o", o), ("state_out", st_out)]:
        put(f"kda.{n}", t)
    META["kda"] = {"T": T, "H": RH, "K": RK, "V": RV,
                   "note": "q is pre-scaled by K**-0.5; q/k pre-L2-normalised"}

    # ---- 7. end-to-end tiny model (token ids -> logits) --------------
    cfg = tiny_cfg()
    w = tiny_weights(cfg)
    toks = torch.tensor([3, 17, 5, 28, 11, 2, 19])
    logits_out, _ = model_ref(toks, w, cfg)
    put_i32("model.tokens", toks)
    put("model.logits", logits_out)
    put_i32("model.argmax", logits_out.argmax(-1))
    # the tiny weights are regenerated from the seed on the Rust side via the
    # same layout, so we record the shape contract rather than every tensor
    META["model"] = {k: (sorted(v) if isinstance(v, set) else v)
                     for k, v in cfg.items()}
    META["model"]["seed"] = 0
    META["model"]["n_tokens"] = len(toks)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(FX, str(out / "fixtures.safetensors"))
    (out / "fixtures_meta.json").write_text(json.dumps(META, indent=2) + "\n")
    print(f"wrote {len(FX)} tensors -> {out}/fixtures.safetensors")
    print(f"wrote meta            -> {out}/fixtures_meta.json")


if __name__ == "__main__":
    main()
