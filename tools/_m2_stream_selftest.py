#!/usr/bin/env python3
"""Self-test for the streaming full-model export path.

Builds the tiny MiniMax-M2, writes it out in the *published* checkpoint
format (sharded safetensors with `block_sparse_moe.experts.E.w1/w2/w3`
naming + index.json + config.json), runs `export_minimax_m2.export_full_streaming`
on it, and checks the result still reproduces the canonical HF greedy
output. Validates the safetensors reader, fake-namespace wiring, and the
w1/w3->gate_up mapping without needing the 230B download. (FP8 dequant is
not exercised here — the tiny weights are fp32 — and is checked separately
on a real shard.)

    python tools/_m2_stream_selftest.py --work /tmp/m2_stream
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

import export_minimax_m2 as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    src = work / "published"
    out = work / "export"
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import MiniMaxM2ForCausalLM
    torch.manual_seed(0)
    cfg = E.tiny_config()
    model = MiniMaxM2ForCausalLM(cfg).to(torch.float32).eval()
    base = model.model

    # --- dump weights in published w1/w2/w3 format ---
    sd = {}
    sd["model.embed_tokens.weight"] = base.embed_tokens.weight.detach().clone()
    sd["model.norm.weight"] = base.norm.weight.detach().clone()
    sd["lm_head.weight"] = model.lm_head.weight.detach().clone()
    for i, layer in enumerate(base.layers):
        p = f"model.layers.{i}"
        a = layer.self_attn
        sd[f"{p}.self_attn.q_proj.weight"] = a.q_proj.weight.detach().clone()
        sd[f"{p}.self_attn.k_proj.weight"] = a.k_proj.weight.detach().clone()
        sd[f"{p}.self_attn.v_proj.weight"] = a.v_proj.weight.detach().clone()
        sd[f"{p}.self_attn.o_proj.weight"] = a.o_proj.weight.detach().clone()
        sd[f"{p}.self_attn.q_norm.weight"] = a.q_norm.weight.detach().clone()
        sd[f"{p}.self_attn.k_norm.weight"] = a.k_norm.weight.detach().clone()
        sd[f"{p}.input_layernorm.weight"] = layer.input_layernorm.weight.detach().clone()
        sd[f"{p}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.detach().clone()
        moe = layer.mlp
        sd[f"{p}.block_sparse_moe.gate.weight"] = moe.gate.weight.detach().clone()
        sd[f"{p}.block_sparse_moe.e_score_correction_bias"] = moe.e_score_correction_bias.detach().clone()
        gup = moe.experts.gate_up_proj  # [E, 2*inter, H]
        dwn = moe.experts.down_proj      # [E, H, inter]
        inter = dwn.shape[2]
        for e in range(gup.shape[0]):
            sd[f"{p}.block_sparse_moe.experts.{e}.w1.weight"] = gup[e, :inter].detach().clone()
            sd[f"{p}.block_sparse_moe.experts.{e}.w3.weight"] = gup[e, inter:].detach().clone()
            sd[f"{p}.block_sparse_moe.experts.{e}.w2.weight"] = dwn[e].detach().clone()

    save_file(sd, str(src / "model.safetensors"))
    index = {"metadata": {"total_size": 0},
             "weight_map": {k: "model.safetensors" for k in sd}}
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    (src / "config.json").write_text(json.dumps(cfg.to_dict()))
    print(f"[selftest] wrote published-format checkpoint: {len(sd)} tensors")

    # --- reference greedy from the canonical model ---
    prompt = [1, 2, 3, 4, 5]
    gen = list(prompt)
    with torch.no_grad():
        cur = torch.tensor([gen], dtype=torch.int64)
        for _ in range(6):
            nxt = int(torch.argmax(model(cur).logits[0, -1]))
            gen.append(nxt)
            cur = torch.tensor([gen], dtype=torch.int64)
    reference = {"prompt_ids": prompt, "greedy_tokens": gen,
                 "first_next_token": gen[len(prompt)]}

    # --- run the streaming exporter ---
    sargs = argparse.Namespace(model=str(src), tiny=False, out=str(out),
                               layers="", no_quant=True, experts_int8=False)
    E.export_full_streaming(sargs, out)
    (out / "reference.json").write_text(json.dumps(reference))

    # --- pipeline check ---
    rc = subprocess.call([sys.executable, str(Path(__file__).parent / "test_minimax_m2_pipeline.py"),
                          "--model-dir", str(out)])
    sys.exit(rc)


if __name__ == "__main__":
    main()
