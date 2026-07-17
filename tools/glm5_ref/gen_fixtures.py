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
from glm5_ref.kernels_ref import apply_rotary_emb, moe_gate, precompute_freqs_cis

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

    from safetensors.torch import save_file
    save_file(FX, str(out / "fixtures.safetensors"))
    (out / "fixtures_meta.json").write_text(json.dumps(META, indent=2))
    total = sum(t.numel() * t.element_size() for t in FX.values())
    print(f"[glm5 fixtures] {len(FX)} tensors, {total/1e6:.3f} MB -> {out}")


if __name__ == "__main__":
    main()
