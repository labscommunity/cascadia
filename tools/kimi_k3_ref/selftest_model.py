#!/usr/bin/env python3
"""Self-consistency checks for the assembled K3 reference model.

The composed model cannot be diffed against upstream (that needs transformers +
fla + triton), so we check the properties that catch the realistic bug classes:

  1. prefill/decode equivalence — running T tokens as one batch must equal
     feeding them one at a time through the caches. This exercises every piece
     of carried state at once: the KDA recurrent state, the short-conv windows,
     the MLA KV cache, and the AttnRes block stack. It is the check most likely
     to catch a state-plumbing bug in either this reference or the Rust port.
  2. causality — changing token t must not alter the logits at positions < t.
  3. AttnRes stack growth — one entry per block boundary.
  4. determinism.

Run:  python tools/kimi_k3_ref/selftest_model.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kimi_k3_ref.kernels_ref import model_ref
from kimi_k3_ref.tiny import tiny_cfg, tiny_weights


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{detail}")
    return ok


def main():
    cfg = tiny_cfg()
    w = tiny_weights(cfg)
    toks = torch.tensor([3, 17, 5, 28, 11, 2, 19])
    results = []

    print("prefill vs incremental decode")
    full, _ = model_ref(toks, w, cfg)
    caches = None
    step_rows = []
    for i in range(len(toks)):
        out, caches = model_ref(toks[i:i + 1], w, cfg, caches)
        step_rows.append(out[0])
    step = torch.stack(step_rows)
    d = (full - step).abs().max().item()
    results.append(check("logits agree", d <= 2e-2, f": max|diff| = {d:.3e}"))
    results.append(check("argmax agrees", bool((full.argmax(-1) == step.argmax(-1)).all())))

    print("causality")
    alt = toks.clone()
    alt[-1] = (int(alt[-1]) + 7) % cfg["vocab_size"]
    full_alt, _ = model_ref(alt, w, cfg)
    d_prefix = (full[:-1] - full_alt[:-1]).abs().max().item()
    results.append(check("earlier positions unchanged", d_prefix == 0.0,
                         f": max|diff| = {d_prefix:.3e}"))

    print("AttnRes block stack")
    # boundaries at layers 0, 2, 4 for block_size=2 over 6 layers -> 3 entries
    expect = sum(1 for i in range(cfg["num_hidden_layers"])
                 if i % cfg["attn_res_block_size"] == 0)
    from kimi_k3_ref.kernels_ref import layer_ref
    br = torch.zeros(len(toks), 0, cfg["hidden_size"])
    x = w["embed"][toks].float()
    st = cv = kv = None
    for i in range(cfg["num_hidden_layers"]):
        x, br, st, cv, kv = layer_ref(x, br, w["layers"][i], cfg, i, st, cv, kv)
        st = cv = kv = None  # fresh per layer; we only care about the stack here
    results.append(check("stack grew once per block", br.shape[1] == expect,
                         f": {br.shape[1]} entries (expected {expect})"))

    print("determinism")
    again, _ = model_ref(toks, w, cfg)
    results.append(check("repeat run identical", bool((full == again).all())))

    print()
    if all(results):
        print(f"PASS — {len(results)}/{len(results)} checks")
        return 0
    print(f"FAIL — {sum(results)}/{len(results)} checks passed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
