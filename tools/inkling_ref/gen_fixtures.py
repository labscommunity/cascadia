#!/usr/bin/env python3
"""Golden fixtures for the Rust Inkling port (PORT_SPEC.md §4). transformers' `InklingForCausalLM`
(native `models/inkling`, >= 5.16) is the oracle; every golden is produced by calling HF modules
directly. The same tensors are also cross-checked against `spec_ref.py` (the spec executed
literally), so a spec-vs-HF disagreement fails HERE, not in the Rust debugger.

Run from the repo root:
    python tools/inkling_ref/gen_fixtures.py \
        --out crates/cascadia-engine-sparse-moe/tests/fixtures/inkling

Writes fixtures.safetensors (+ fixtures.json sidecar for humans):
  prompt_ids i64 [12], greedy_ids i64 [8]            HF generate(do_sample=False), cross-checked
                                                       against a no-cache re-prefill loop
  layer{L}_out f32 [12, 64]                            output of decoder layer L on the prompt prefill
  final_logits f32 [12, 120]                           after norm, mup, unembed, vocab slice
  sconv_in [8, 6], sconv_w [8, 4], sconv_out [8, 6]    InklingShortConvolution (stored [C, T])
  relpos_r [4, 4], relpos_proj [4, 8], relpos_bias [4, 10]   InklingRelativeLogits, query at 9, keys 0..9
  gate_x [1, 64], gate_logits [10], gate_bias [8], gate_global_scale [1],
  gate_idx i64 [2], gate_w [2], gate_gamma [2]         InklingTopkRouter of layer 1 (canonical order)
  attn_x [6, 64] -> attn_out [6, 64] (layer 3: global + log scaling active, n_floor 4)
                    attn_out_sliding [6, 64] (layer 0: window 4)   -- o_proj output, BEFORE attn_sconv
  moe_x [1, 64] -> moe_out [1, 64]                     InklingMoE of layer 1
  model.llm.* (every weight, checkpoint names)         gate/up RE-INTERLEAVED into the w13 layout

Sliding-window rule observed in transformers 5.16.1 (`masking_utils.sliding_window_overlay`,
AND-ed with the causal mask by `sliding_window_causal_mask_function`):
    allowed(q_idx, kv_idx) = (kv_idx <= q_idx) and (kv_idx > q_idx - sliding_window)
i.e. `q_idx - kv_idx < sliding_window`: the window holds exactly `sliding_window` keys INCLUDING
the current token (window 4, q=5 -> keys 2,3,4,5). Verified below by materializing
`create_sliding_window_causal_mask` and comparing it with that formula.

sconv convention (torch conv1d, padding K-1, then [:T], residual added inside the module):
    out[p, c] = sum_{j=0..3} w[c, j] * u[p - 3 + j, c]   (u[<0] = 0)   ; module returns out + u
so kernel tap w[:, 3] multiplies the CURRENT token and w[:, 0] the token 3 back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import export_inkling  # noqa: E402
from inkling_ref import (MIN_ARGMAX_MARGIN, N_GEN, TINY_CONFIG, TINY_SEED, UNEMBED_STD,  # noqa: E402
                         build_tiny_model, greedy_reference, hf_state_to_checkpoint, select_prompt)
from inkling_ref import spec_ref  # noqa: E402

DEFAULT_OUT = _TOOLS.parent / "crates" / "cascadia-engine-sparse-moe" / "tests" / "fixtures" / "inkling"

FX: dict[str, torch.Tensor] = {}
CHECKS: dict[str, float] = {}


def put(name, t):
    FX[name] = t.detach().to(torch.float32).contiguous().clone()


def put_i64(name, t):
    FX[name] = torch.as_tensor(t).to(torch.int64).contiguous().clone()


def check(name, a, b, tol=1e-4):
    """Record max |a-b| (relative to max |b|) between HF and the spec reference; fail above tol."""
    a, b = a.detach().float(), b.detach().float()
    err = float((a - b).abs().max())
    rel = err / max(float(b.abs().max()), 1e-12)
    CHECKS[name] = rel
    if rel > tol:
        raise AssertionError(f"[spec-vs-HF] {name}: rel err {rel:.3e} > {tol:.1e} (abs {err:.3e})")


def canonical_gate_order(logits, bias, top_k):
    """Selection order the Rust gate uses: (sigmoid(logit) + bias) descending, ties -> lower id."""
    s = torch.sigmoid(logits.float())[: bias.shape[0]] + bias.float()
    return sorted(range(bias.shape[0]), key=lambda i: (-float(s[i]), i))[:top_k]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import transformers
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.inkling.modeling_inkling import (InklingRelativeLogits,
                                                              InklingShortConvolution)

    man = export_inkling.load_and_validate_config(TINY_CONFIG)
    model = build_tiny_model(man, TINY_SEED)
    cfg = model.config
    ckpt = hf_state_to_checkpoint(model.state_dict())
    H, eps = man["hidden_size"], man["rms_norm_eps"]
    g = torch.Generator().manual_seed(TINY_SEED + 1)

    # ---- 0. sliding-window rule: materialize HF's mask and compare with the formula ----
    T6 = 6
    dummy = torch.zeros(1, T6, H)
    sm = create_sliding_window_causal_mask(config=cfg, inputs_embeds=dummy, attention_mask=None,
                                           past_key_values=None, position_ids=None)
    cm = create_causal_mask(config=cfg, inputs_embeds=dummy, attention_mask=None,
                            past_key_values=None, position_ids=None)
    assert sm is not None and cm is not None, "eager masks must be materialized"
    W = man["sliding_window"]
    want = torch.tensor([[(j <= i) and (i - j < W) for j in range(T6)] for i in range(T6)])
    assert torch.equal(sm[0, 0] == 0, want), f"sliding mask rule mismatch:\n{(sm[0, 0] == 0).int()}"
    assert torch.equal(cm[0, 0] == 0, torch.tril(torch.ones(T6, T6, dtype=torch.bool)))

    # ---- 1. prompt prefill: per-layer outputs + final logits; greedy continuation. The prompt is
    #         the first candidate seed whose top-2 logit gap is >= MIN_ARGMAX_MARGIN at all 12 prompt
    #         positions and 8 greedy steps, for the f32 model AND its int4 round-trip (--tiny uses
    #         the same prompt), so "argmax EXACT" survives bf16 write-back on the Rust side.
    prompt, prompt_seed, prompt_margin = select_prompt(man, model)
    ids = torch.tensor([prompt], dtype=torch.long)
    layer_outs = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, li=li: layer_outs.__setitem__(li, o.detach().clone()))
             for li, layer in enumerate(model.model.layers)]
    o = model(ids, use_cache=False, output_hidden_states=True)
    for h in hooks:
        h.remove()
    for li in range(man["num_layers"] - 1):  # HF replaces the last entry with the normed state
        assert torch.equal(o.hidden_states[li + 1], layer_outs[li])
    ref, margins = greedy_reference(model, prompt, N_GEN)
    put_i64("prompt_ids", prompt)
    put_i64("greedy_ids", ref["greedy_ids"])
    for li in range(man["num_layers"]):
        put(f"layer{li}_out", layer_outs[li][0])
    put("final_logits", o.logits[0])
    assert o.logits.shape[-1] == man["unpadded_vocab_size"]
    assert o.logits[0].argmax(-1).tolist() == ref["first_logits_argmax"]

    spec_outs, spec_logits = spec_ref.model_forward(prompt, ckpt, man)
    for li in range(man["num_layers"]):
        check(f"layer{li}_out", layer_outs[li][0], spec_outs[li])
    check("final_logits", o.logits[0], spec_logits)

    # ---- 2. sconv: InklingShortConvolution on [1, T, C], no cache ----
    C, Tc = 8, 6
    sc = InklingShortConvolution(C, man["conv_kernel_size"], layer_idx=0, conv_idx=0).float()
    sconv_w = (0.05 * torch.randn(C, man["conv_kernel_size"], generator=g)).to(torch.bfloat16).float()
    sc.conv1d.weight.copy_(sconv_w[:, None, :])
    sconv_in = (torch.randn(C, Tc, generator=g)).to(torch.bfloat16).float()  # stored [C, T]
    sconv_out = sc(sconv_in.T[None])[0].T  # module wants [B, T, C]
    put("sconv_in", sconv_in)
    put("sconv_w", sconv_w)
    put("sconv_out", sconv_out)
    check("sconv_out", sconv_out, spec_ref.sconv(sconv_w, sconv_in.T).T)

    # ---- 3. relpos: InklingRelativeLogits, query at position 9 vs keys 0..9 (extent 8 -> keys 0,1 get 0) ----
    Hq, d_rel, extent, KV = man["num_attention_heads"], man["d_rel"], man["rel_extent"], 10
    rl = InklingRelativeLogits(d_rel, extent).float()
    relpos_proj = (0.05 * torch.randn(d_rel, extent, generator=g)).to(torch.bfloat16).float()
    rl.proj.copy_(relpos_proj)
    relpos_r = torch.randn(Hq, d_rel, generator=g).to(torch.bfloat16).float()
    relpos_bias = rl(relpos_r[None, None], torch.tensor([9]), torch.arange(KV))[0, :, 0, :]  # [Hq, KV]
    put("relpos_r", relpos_r)
    put("relpos_proj", relpos_proj)
    put("relpos_bias", relpos_bias)
    check("relpos_bias", relpos_bias, spec_ref.relpos_bias(relpos_r, relpos_proj, 9, list(range(KV))))
    assert torch.all(relpos_bias[:, :KV - extent] == 0)

    # ---- 4. gate: InklingTopkRouter of layer 1 (the first MoE layer). Pick an input where the
    #         e_score_correction_bias actually changes the selection, so the Rust gate must apply it.
    router = model.model.layers[1].mlp.gate
    top_k, n_shared = man["top_k"], man["n_shared_experts"]
    gate_x = None
    for _ in range(256):
        cand = torch.randn(1, H, generator=g).to(torch.bfloat16).float()
        lg = cand @ router.weight.T
        with_bias = set(canonical_gate_order(lg[0], router.e_score_correction_bias, top_k))
        without = set(canonical_gate_order(lg[0], torch.zeros_like(router.e_score_correction_bias), top_k))
        gate_x = cand
        if with_bias != without:
            break
    else:
        print("[gate] WARNING: no input found where the bias flips the selection", flush=True)
    gate_logits = (gate_x @ router.weight.T)[0]
    _, tw, ti, gam = router(gate_x)
    order = canonical_gate_order(gate_logits, router.e_score_correction_bias, top_k)
    hf_pos = {int(i): p for p, i in enumerate(ti[0].tolist())}
    assert set(hf_pos) == set(order), f"HF top-k {sorted(hf_pos)} != canonical {sorted(order)}"
    gate_w = torch.tensor([float(tw[0, hf_pos[i]]) for i in order])
    put("gate_x", gate_x)
    put("gate_logits", gate_logits)
    put("gate_bias", router.e_score_correction_bias)
    put("gate_global_scale", router.global_scale)
    put_i64("gate_idx", order)
    put("gate_w", gate_w)
    put("gate_gamma", gam[0])
    s_idx, s_w, s_gam = spec_ref.gate(gate_logits, router.e_score_correction_bias, top_k, n_shared,
                                      man["route_scale"], float(router.global_scale))
    assert s_idx.tolist() == order
    check("gate_w", gate_w, s_w)
    check("gate_gamma", gam[0], s_gam)

    # ---- 5. attention: InklingAttention.forward on [1, T, H], no cache. Layer 3 = global with log
    #         scaling active (n_floor 4 -> tau > 1 at positions 4, 5); layer 0 = sliding (window 4).
    Ta = 6
    attn_x = torch.randn(Ta, H, generator=g).to(torch.bfloat16).float()
    masks = {"global": create_causal_mask(config=cfg, inputs_embeds=attn_x[None], attention_mask=None,
                                          past_key_values=None, position_ids=None),
             "sliding": create_sliding_window_causal_mask(config=cfg, inputs_embeds=attn_x[None],
                                                          attention_mask=None, past_key_values=None,
                                                          position_ids=None)}
    for li, name in ((3, "attn_out"), (0, "attn_out_sliding")):
        assert man["layer_types"][li] == ("global" if li == 3 else "sliding")
        attn = model.model.layers[li].self_attn
        out_hf, _ = attn(hidden_states=attn_x[None], attention_mask=masks[man["layer_types"][li]])
        put(name, out_hf[0])
        Wl = spec_ref.layer_weights(ckpt, li)
        check(name, out_hf[0], spec_ref.attention(attn_x, Wl, **spec_ref.layer_dims(man, li)))
    put("attn_x", attn_x)

    # ---- 6. MoE: InklingMoE of layer 1 on one token ----
    moe_x = torch.randn(1, H, generator=g).to(torch.bfloat16).float()
    moe_out = model.model.layers[1].mlp(moe_x[None])[0]
    put("moe_x", moe_x)
    put("moe_out", moe_out)
    check("moe_out", moe_out, spec_ref.moe(spec_ref.layer_weights(ckpt, 1), moe_x, top_k=top_k,
                                           n_shared=n_shared, route_scale=man["route_scale"]))

    # ---- 7. every weight under its checkpoint name (f32, bf16-exact) ----
    for k, v in ckpt.items():
        put(k, v)

    from safetensors.torch import save_file
    save_file(FX, str(out / "fixtures.safetensors"))
    sidecar = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": TINY_SEED,
        "weight_init": f"N(0, 0.05) bf16-rounded; norms/global_scale 1 + N(0, 0.05); unembed N(0, {UNEMBED_STD})",
        "tiny_config": TINY_CONFIG,
        "manifest": man,
        "prompt_ids": prompt,
        "prompt_seed": prompt_seed,
        "greedy_ids": ref["greedy_ids"],
        "first_logits_argmax": ref["first_logits_argmax"],
        "argmax_top2_logit_margins": margins,
        "argmax_margin_floor": {"required": MIN_ARGMAX_MARGIN, "f32_and_int4_roundtrip": prompt_margin},
        "gate": {"layer": 1, "idx_order": "selection score (sigmoid(logit)+bias) descending, ties -> lower id",
                 "logits": "[num_experts + n_shared]: routed first, shared last"},
        "conventions": {
            "sliding_window": "allowed iff q_idx - kv_idx < sliding_window (window keys incl. current)",
            "sconv": "out[p,c] = sum_j w[c,j]*u[p-3+j,c], u[<0]=0, returns out+u; w[:,3] hits the current token",
            "attn_out": "o_proj output (before attn_sconv), positions 0..5, layer 3 log-scaled (tau at p>=4)",
            "layer_out": "decoder-layer output (after mlp_sconv + residual), i.e. the next layer's input",
            "final_logits": "rmsnorm(x, norm)/mup @ unembed^T, sliced to unpadded_vocab_size",
            "weights": "checkpoint names; w13 rows interleaved (2i = gate_i, 2i+1 = up_i); convs [C,1,K]",
            "dtypes": "all floats f32 (bf16-exact weights); prompt_ids/greedy_ids/gate_idx i64",
        },
        "spec_vs_hf_max_rel_err": CHECKS,
        "tensors": {k: [list(v.shape), str(v.dtype).replace("torch.", "")] for k, v in FX.items()},
    }
    (out / "fixtures.json").write_text(json.dumps(sidecar, indent=2))
    total = sum(t.numel() * t.element_size() for t in FX.values())
    print(f"[inkling fixtures] {len(FX)} tensors, {total / 1e6:.3f} MB -> {out}", flush=True)
    print(f"[inkling fixtures] prompt seed {prompt_seed} {prompt} greedy {ref['greedy_ids']} "
          f"(min top-2 logit margin {min(min(margins['prompt']), min(margins['greedy'])):.3f}, "
          f"floor {prompt_margin:.3f} incl. int4 round-trip)", flush=True)
    worst = max(CHECKS.items(), key=lambda kv: kv[1])
    print(f"[spec-vs-HF] all {len(CHECKS)} checks within tolerance; worst {worst[0]} rel {worst[1]:.2e}", flush=True)


if __name__ == "__main__":
    main()
