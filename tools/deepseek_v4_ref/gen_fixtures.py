#!/usr/bin/env python3
"""Generate golden fixtures for the Rust dsv4 shell (Phase 2).

Dumps two kinds of goldens from the deterministic tiny reference model:

1. component-level: exact in/out pairs for every primitive the Rust side
   must replicate (hadamard, FP8/FP4 QAT sims, sinkhorn HC, rmsnorm, YaRN
   rope, sparse_attn) with synthetic inputs;
2. integration-level: hook-captured per-layer attention/MoE inputs+outputs
   and hidden states from one prefill + two decode steps of the tiny model,
   plus logits — the targets the assembled Rust pipeline must hit.

Output: fixtures.safetensors + fixtures_meta.json, intended to be committed
under crates/cascadia-engine-sparse-moe/tests/fixtures/dsv4/ (total < 2 MB).

Run:  python gen_fixtures.py --out ../../crates/cascadia-engine-sparse-moe/tests/fixtures/dsv4
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepseek_v4_ref.kernels_ref import (act_quant, fp4_act_quant,
                                         hadamard_transform, hc_split_sinkhorn,
                                         sparse_attn)
from deepseek_v4_ref.model import (RMSNorm, apply_rotary_emb,
                                   precompute_freqs_cis)
from export_deepseek_v4 import TINY_KW, build_tiny

FX = {}
META = {"tiny": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TINY_KW.items()}}


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

    def rnd(*shape, s=0.5):
        return (s * torch.randn(*shape, generator=g)).to(torch.bfloat16)

    # ---- 1. hadamard (dim 32, the indexer head_dim in tiny) ----
    x = rnd(4, 32)
    put("hadamard.in", x)
    put("hadamard.out", hadamard_transform(x.clone(), scale=32 ** -0.5))

    # ---- 2. act_quant inplace sim (block 64, pow2 scales — attention kv) ----
    x = rnd(3, 128)
    put("act_quant.in", x)
    y = x.clone()
    act_quant(y, 64, "ue8m0", torch.float8_e8m0fnu, True)
    put("act_quant.out", y)

    # ---- 3. fp4_act_quant inplace sim (block 32 — indexer q/kv) ----
    x = rnd(3, 64)
    put("fp4_act_quant.in", x)
    y = x.clone()
    fp4_act_quant(y, 32, True)
    put("fp4_act_quant.out", y)

    # ---- 4. sinkhorn HC coefficients ----
    hc = TINY_KW["hc_mult"]
    mix_hc = (2 + hc) * hc
    mixes = torch.randn(2, 3, mix_hc, generator=g)
    scale = 0.3 * torch.randn(3, generator=g)
    base = 0.3 * torch.randn(mix_hc, generator=g)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc,
                                        TINY_KW["hc_sinkhorn_iters"],
                                        TINY_KW["hc_eps"])
    put("sinkhorn.mixes", mixes)
    put("sinkhorn.scale", scale)
    put("sinkhorn.base", base)
    put("sinkhorn.pre", pre)
    put("sinkhorn.post", post)
    put("sinkhorn.comb", comb)

    # ---- 5. rmsnorm ----
    x = rnd(2, 5, 64)
    norm = RMSNorm(64, 1e-6)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.05 * torch.randn(64, generator=g))
    put("rmsnorm.in", x)
    put("rmsnorm.w", norm.weight)
    put("rmsnorm.out", norm(x))

    # ---- 6. rope freqs: YaRN and plain (rope_head_dim 16) ----
    fy = precompute_freqs_cis(16, 64, 32, 160000.0, 16.0, 32, 1)
    put("rope.freqs_yarn", torch.view_as_real(fy))          # [64, 8, 2]
    fp = precompute_freqs_cis(16, 64, 0, 10000.0, 16.0, 32, 1)
    put("rope.freqs_plain", torch.view_as_real(fp))

    # ---- 7. rope apply: 4-d path, forward + inverse ----
    x = rnd(1, 6, 4, 16)
    put("rope.apply4.in", x)
    y = x.clone()
    apply_rotary_emb(y, fp[0:6])
    put("rope.apply4.out", y)
    yi = x.clone()
    apply_rotary_emb(yi, fp[0:6], inverse=True)
    put("rope.apply4.inv_out", yi)
    x3 = rnd(1, 6, 16)                                       # 3-d path (compressor kv)
    put("rope.apply3.in", x3)
    y3 = x3.clone()
    apply_rotary_emb(y3, fy[0:6])
    put("rope.apply3.out", y3)

    # ---- 8. sparse_attn ----
    q = rnd(1, 3, 4, 16)
    kv = rnd(1, 10, 16)
    sink = 0.5 * torch.randn(4, generator=g)
    idxs = torch.tensor([[[0, 1, 2, -1, -1, -1],
                          [1, 2, 3, 4, -1, -1],
                          [2, 3, 4, 5, 9, -1]]], dtype=torch.int32)
    put("sparse_attn.q", q)
    put("sparse_attn.kv", kv)
    put("sparse_attn.sink", sink)
    put_i32("sparse_attn.idxs", idxs)
    put("sparse_attn.out", sparse_attn(q, kv, sink, idxs, 16 ** -0.5))

    # ---- 8b. compressor + indexer component fixtures (milestone 2) ----
    # Fresh generator so sections 1-8 stay byte-identical across regenerations.
    from deepseek_v4_ref.model import Compressor, Indexer, ModelArgs
    a = ModelArgs(**TINY_KW)
    # real model constructs under bf16 default (Transformer sets it); mirror
    # that so Indexer's wq_b / kv_cache get the right dtypes. Also mirror the
    # module globals Transformer.__init__ sets (scale_dtype="fp8" default ->
    # power-of-2 ue8m0 scales in act_quant) — standalone components would
    # otherwise run with the file defaults (scale_fmt=None, no pow2).
    torch.set_default_dtype(torch.bfloat16)
    import deepseek_v4_ref.model as ref_model
    ref_model.scale_fmt = "ue8m0"
    ref_model.scale_dtype = torch.float8_e8m0fnu
    g2 = torch.Generator().manual_seed(77)

    def rnd2(*shape, s=0.5):
        return (s * torch.randn(*shape, generator=g2)).to(torch.bfloat16)

    def det_init(mod):
        with torch.no_grad():
            for name, p in mod.named_parameters():
                if "norm" in name and p.ndim == 1:
                    p.copy_(1.0 + 0.05 * torch.randn(p.shape, generator=g2,
                                                     dtype=torch.float32).to(p.dtype))
                else:
                    p.copy_((0.3 * torch.randn(p.shape, generator=g2,
                                               dtype=torch.float32)).to(p.dtype))
        return mod

    yarn_freqs = precompute_freqs_cis(a.rope_head_dim, a.max_seq_len,
                                      a.original_seq_len, a.compress_rope_theta,
                                      a.rope_factor, a.beta_fast, a.beta_slow)
    comp_meta = {}

    def comp_fixture(tag, ratio, head_dim, rotate, n_dec):
        comp = det_init(Compressor(a, ratio, head_dim, rotate))
        comp.kv_cache = torch.zeros(a.max_batch_size, a.max_seq_len // ratio,
                                    head_dim, dtype=torch.bfloat16)
        comp.freqs_cis = yarn_freqs
        put(f"{tag}.wkv_w", comp.wkv.weight)
        put(f"{tag}.wgate_w", comp.wgate.weight)
        put(f"{tag}.ape", comp.ape)
        put(f"{tag}.norm_w", comp.norm.weight)
        x = rnd2(1, 18, a.dim)
        put(f"{tag}.x_prefill", x)
        fired = []
        with torch.no_grad():
            ret = comp(x, 0)
            if ret is not None:
                put(f"{tag}.prefill_ret", ret)
                fired.append(("prefill", int(ret.shape[1])))
            xd = rnd2(n_dec, a.dim)
            put(f"{tag}.x_dec", xd)
            for i in range(n_dec):
                pos = 18 + i
                r = comp(xd[i:i + 1].unsqueeze(0), pos)
                if r is not None:
                    put(f"{tag}.dec_ret_{pos}", r)
                    fired.append((f"pos{pos}", 1))
        put(f"{tag}.cache_final", comp.kv_cache)
        comp_meta[tag] = fired

    comp_fixture("comp4", 4, TINY_KW["head_dim"], False, 8)
    comp_fixture("comp16", 16, TINY_KW["head_dim"], False, 14)
    comp_fixture("compidx", 4, TINY_KW["index_head_dim"], True, 8)
    META["compress_fired"] = comp_meta

    # full Indexer: topk outputs per phase (offset=0 for the fixture)
    idx = det_init(Indexer(a, 4))
    idx.freqs_cis = yarn_freqs
    put("idx.wq_b_w", idx.wq_b.weight)
    put("idx.weights_proj_w", idx.weights_proj.weight)
    put("idx.comp_wkv_w", idx.compressor.wkv.weight)
    put("idx.comp_wgate_w", idx.compressor.wgate.weight)
    put("idx.comp_ape", idx.compressor.ape)
    put("idx.comp_norm_w", idx.compressor.norm.weight)
    xi = rnd2(1, 18, a.dim)
    qri = rnd2(1, 18, a.q_lora_rank)
    put("idx.x_prefill", xi)
    put("idx.qr_prefill", qri)
    with torch.no_grad():
        tk = idx(xi, qri, 0, 0)
        put_i32("idx.topk_prefill", tk)
        xd = rnd2(8, a.dim)
        qd = rnd2(8, a.q_lora_rank)
        put("idx.x_dec", xd)
        put("idx.qr_dec", qd)
        dec_tks = []
        for i in range(8):
            pos = 18 + i
            tk = idx(xd[i:i + 1].unsqueeze(0), qd[i:i + 1].unsqueeze(0), pos, 0)
            dec_tks.append(tk.reshape(-1))
        put_i32("idx.topk_dec", torch.stack(dec_tks))
    put("idx.cache_final", idx.kv_cache)

    # ---- 9. integration: tiny model, prefill + 2 decode steps, hooks ----
    model = build_tiny()
    prompt = list(range(1, 19))
    META["prompt"] = prompt
    captures = {}

    def attn_hook(li):
        def h(mod, inp, out):
            key = f"L{li}.attn.{captures['phase']}"
            captures[key + ".in"] = inp[0].detach().clone()
            captures[key + ".out"] = out.detach().clone()
        return h

    def ffn_hook(li):
        def h(mod, inp, out):
            key = f"L{li}.ffn.{captures['phase']}"
            captures[key + ".in"] = inp[0].detach().clone()
            captures[key + ".out"] = out.detach().clone()
        return h

    def gate_hook(li):
        def h(mod, inp, out):
            key = f"L{li}.gate.{captures['phase']}"
            weights, indices = out
            captures[key + ".weights"] = weights.detach().to(torch.float32).clone()
            captures[key + ".indices"] = indices.detach().to(torch.int32).clone()
        return h

    hooks = []
    for li, layer in enumerate(model.layers):
        hooks.append(layer.attn.register_forward_hook(attn_hook(li)))
        hooks.append(layer.ffn.register_forward_hook(ffn_hook(li)))
        hooks.append(layer.ffn.gate.register_forward_hook(gate_hook(li)))

    # capture per-layer hidden states by stepping the loop manually
    with torch.no_grad():
        captures["phase"] = "prefill"
        ids = torch.tensor([prompt])
        h = model.embed(ids)
        put("int.embed_out", h)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for li, layer in enumerate(model.layers):
            h = layer(h, 0, ids)
            put(f"int.prefill.h_after_L{li}", h)
        logits = model.head(h, model.hc_head_fn, model.hc_head_scale,
                            model.hc_head_base, model.norm)
        put("int.prefill.logits", logits)
        toks = [int(logits[0].argmax())]

        for step in range(2):
            captures["phase"] = f"decode{step}"
            pos = len(prompt) + step
            ids_step = torch.tensor([[toks[-1]]])
            h = model.embed(ids_step)
            h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
            for li, layer in enumerate(model.layers):
                h = layer(h, pos, ids_step)
                put(f"int.decode{step}.h_after_L{li}", h)
            logits = model.head(h, model.hc_head_fn, model.hc_head_scale,
                                model.hc_head_base, model.norm)
            put(f"int.decode{step}.logits", logits)
            toks.append(int(logits[0].argmax()))
    for hk in hooks:
        hk.remove()
    for k, v in captures.items():
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.int32, torch.int64):
                put_i32("int." + k, v)
            else:
                put("int." + k, v)

    # 9b. per-layer weights so Rust tests can build the layers themselves
    sd = model.state_dict()
    for name, t in sd.items():
        if name.startswith("mtp."):
            continue
        if name.startswith(("layers.", "embed.", "head.", "norm.", "hc_head")):
            if t.dtype in (torch.int32, torch.int64):
                put_i32(f"int.w.{name}", t)
            else:
                put(f"int.w.{name}", t)
    META["greedy_tokens"] = toks
    META["note"] = ("integration hiddens are [b,s,hc,dim] (prefill s=18, decode "
                    "s=1); attn hook in/out are the normed input / attn output "
                    "of Attention.forward for each layer+phase")

    from safetensors.torch import save_file
    save_file(FX, str(out / "fixtures.safetensors"))
    (out / "fixtures_meta.json").write_text(json.dumps(META, indent=2))
    total = sum(t.numel() * t.element_size() for t in FX.values())
    print(f"[fixtures] {len(FX)} tensors, {total/1e6:.2f} MB -> {out}")
    print(f"[greedy] {toks}")


if __name__ == "__main__":
    main()
