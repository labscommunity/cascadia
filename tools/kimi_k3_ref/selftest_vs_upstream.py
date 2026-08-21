#!/usr/bin/env python3
"""Validate `kernels_ref.py` against the VENDORED UPSTREAM sources.

This is the check that makes the CPU reference trustworthy: rather than
comparing against hand-written expectations (which would just re-encode my
reading of the code), it extracts the real upstream functions out of
`upstream/*.py` and diffs our transcription against them numerically.

Upstream can't be imported directly — `modeling_kimi_linear.py` needs
transformers + fla, and `kda_gate.py` needs triton. So we pull the specific
pure-torch functions/classes out by AST and exec them with a minimal namespace.

Run:  python tools/kimi_k3_ref/selftest_vs_upstream.py
"""
import ast
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kimi_k3_ref.kernels_ref import (apply_attn_res, kda_gate, kda_recurrence,
                                     moe_gate, situ)

UP = HERE / "upstream"


def extract(path: Path, names):
    """Exec just the named top-level defs/classes from `path`."""
    tree = ast.parse(path.read_text())
    want = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    got = {n.name for n in want}
    missing = set(names) - got
    if missing:
        raise SystemExit(f"could not find {missing} in {path.name}")
    ns = {"torch": torch, "F": F, "nn": nn, "math": __import__("math")}
    exec(compile(ast.Module(body=want, type_ignores=[]), str(path), "exec"), ns)
    return ns


def check(name, a, b, tol=1e-5):
    a, b = a.to(torch.float64), b.to(torch.float64)
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    d = (a - b).abs().max().item()
    ok = d <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|diff| = {d:.3e}")
    return ok


def main():
    torch.manual_seed(0)
    results = []

    # ---- 1. SiTU vs upstream SituAndMul -------------------------------
    print("SiTU vs upstream SituAndMul")
    up = extract(UP / "modeling_kimi_linear.py", ["SituAndMul"])
    ref = up["SituAndMul"](beta=4.0, linear_beta=25.0)
    x = torch.randn(7, 256, dtype=torch.float32) * 3.0
    results.append(check("situ", situ(x, 4.0, 25.0), ref(x)))
    # also the no-linear_beta path
    ref2 = up["SituAndMul"](beta=4.0, linear_beta=None)
    results.append(check("situ (no linear_beta)", situ(x, 4.0, None), ref2(x)))

    # ---- 2. AttnRes vs upstream _apply_attn_res -----------------------
    print("AttnRes vs upstream _apply_attn_res")
    up = extract(UP / "modeling_kimi_linear.py", ["_apply_attn_res"])
    T, H, NB, EPS = 5, 64, 3, 1e-5
    ps = torch.randn(T, H)
    br = torch.randn(T, NB, H)

    class _P:  # stand-ins for nn.Linear(H,1,bias=False) / KimiRMSNorm(H)
        pass
    proj, norm = _P(), _P()
    proj.weight = torch.randn(1, H)
    norm.weight = torch.randn(H)
    norm.variance_epsilon = EPS
    results.append(check(
        "apply_attn_res",
        apply_attn_res(ps, br, proj.weight, norm.weight, EPS),
        up["_apply_attn_res"](ps, br, proj, norm),
    ))

    # ---- 3. KDA gate vs upstream naive gates --------------------------
    print("KDA gate vs upstream naive_kda_*_gate")
    up = extract(UP / "kda_gate.py", ["naive_kda_gate", "naive_kda_lowerbound_gate"])
    Hh, D = 6, 32
    g_raw = torch.randn(4, Hh, D)
    a_log = torch.rand(Hh).log()
    dt_bias = torch.randn(Hh * D)
    # K3 uses the lower-bound branch (gate_lower_bound = -5.0)
    results.append(check(
        "kda_gate (lower_bound=-5)",
        kda_gate(g_raw, a_log, dt_bias, -5.0),
        up["naive_kda_lowerbound_gate"](g_raw.clone(), a_log, dt_bias, -5.0),
    ))
    results.append(check(
        "kda_gate (default branch)",
        kda_gate(g_raw, a_log, dt_bias, None),
        up["naive_kda_gate"](g_raw.clone(), a_log, dt_bias),
    ))

    # ---- 4. KDA recurrence vs upstream naive_recurrent_kda ------------
    print("KDA recurrence vs upstream naive_recurrent_kda")
    up = extract(UP / "kda_naive.py", ["naive_recurrent_kda"])
    B, T, Hh, K, V = 1, 9, 4, 16, 16
    q = F.normalize(torch.randn(B, T, Hh, K), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, Hh, K), p=2, dim=-1)
    v = torch.randn(B, T, Hh, V)
    g = -torch.rand(B, T, Hh, K)          # log-space decay, negative
    beta = torch.rand(B, T, Hh)           # already sigmoid'd
    s0 = torch.randn(B, Hh, K, V) * 0.1

    o_up, s_up = up["naive_recurrent_kda"](
        q, k, v, g, beta, initial_state=s0.clone(), output_final_state=True)
    # ours: no batch dim, and q pre-scaled by K**-0.5 (upstream scales inside)
    o_ours, s_ours = kda_recurrence(
        q[0] * (K ** -0.5), k[0], v[0], g[0], beta[0], state=s0[0].clone())
    results.append(check("kda_recurrence o", o_ours, o_up[0], tol=1e-4))
    results.append(check("kda_recurrence state", s_ours, s_up[0], tol=1e-4))

    # ---- 5. moe_gate vs upstream KimiMoEGate semantics ----------------
    # Upstream uses torch.topk(sorted=False) whose tie order is unspecified;
    # we compare the SELECTED SET and the resulting weights-by-expert, not the
    # emission order (the Rust port pins a deterministic order of its own).
    print("moe_gate vs upstream KimiMoEGate")
    E, TK = 32, 6
    logits = torch.randn(11, E)
    bias = torch.randn(E) * 0.1
    idx, wt = moe_gate(logits, bias, TK, 1.0, True)

    scores = torch.sigmoid(logits.float())
    up_idx = torch.topk(scores + bias.unsqueeze(0), k=TK, dim=-1, sorted=False)[1]
    up_wt = scores.gather(1, up_idx)
    up_wt = up_wt / (up_wt.sum(-1, keepdim=True) + 1e-20)

    set_ok = all(set(idx[i].tolist()) == set(up_idx[i].tolist()) for i in range(11))
    print(f"  {'ok  ' if set_ok else 'FAIL'} moe_gate selected set matches")
    results.append(set_ok)
    # weight attached to each expert must agree regardless of order
    dense_ours = torch.zeros(11, E).scatter_(1, idx, wt)
    dense_up = torch.zeros(11, E).scatter_(1, up_idx, up_wt)
    results.append(check("moe_gate weights (by expert)", dense_ours, dense_up))

    # ---- 6. short_conv vs torch's own causal depthwise conv1d ---------
    # fla's ShortConvolution is a depthwise conv1d (groups=D) with causal
    # padding K-1, truncated to T, then silu. We validate against F.conv1d
    # rather than fla (which needs triton), covering both the fresh-state and
    # the carried-state (streaming) paths.
    print("short_conv vs F.conv1d (causal depthwise + silu)")
    from kimi_k3_ref.kernels_ref import short_conv
    D, KK, T2 = 12, 4, 10
    xs = torch.randn(T2, D)
    wc = torch.randn(D, KK)

    def conv_ref(seq):
        z = seq.t().unsqueeze(0)                                  # [1, D, T]
        z = F.pad(z, (KK - 1, 0))
        y = F.conv1d(z, wc.unsqueeze(1), groups=D)                # [1, D, T]
        return F.silu(y.squeeze(0).t())

    y_ours, st = short_conv(xs, wc)
    results.append(check("short_conv (fresh state)", y_ours, conv_ref(xs), tol=1e-2))

    # streaming: first 6 tokens, then the remaining 4 carrying state, must
    # equal running all 10 at once
    y_a, st_a = short_conv(xs[:6], wc)
    y_b, _ = short_conv(xs[6:], wc, st_a)
    results.append(check("short_conv (carried state)",
                         torch.cat([y_a, y_b], 0), conv_ref(xs), tol=1e-2))

    print()
    if all(results):
        print(f"PASS — {len(results)}/{len(results)} checks agree with upstream")
        return 0
    print(f"FAIL — {sum(results)}/{len(results)} checks passed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
