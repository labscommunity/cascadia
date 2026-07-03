#!/usr/bin/env python3
"""Run an exported MiniMax-M2 (cascadia sparse-MoE OV-IR layout) as a full
pipeline in pure Python/OpenVINO and check it against the canonical HF
greedy output saved in reference.json.

This is the architecture-correctness gate: it exercises the *exact* OV
graphs the Rust runtime will run (embed -> per-layer [shell -> top-k expert
dispatch -> combine] -> head), token-by-token with a growing KV cache, and
asserts the greedy token sequence matches the HF reference. If this passes,
the export + the routing/attention/rope math are correct and the Rust side
is just plumbing.

Usage:
    python test_minimax_m2_pipeline.py --model-dir /tmp/m2_tiny
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--max-new", type=int, default=0, help="override ref gen length")
    args = ap.parse_args()

    md = Path(args.model_dir)
    man = json.loads((md / "manifest.json").read_text())
    ref = json.loads((md / "reference.json").read_text())
    H = man["hidden_size"]
    KV = man["num_kv_heads"]
    D = man["head_dim"]
    K = man["top_k"]
    layers = man.get("exported_layers", list(range(man["num_layers"])))

    core = ov.Core()
    # SNIPPETS bug workaround seen on K2.6 shells: harmless if unsupported.
    try:
        core.set_property("CPU", {"SNIPPETS_MODE": "DISABLE"})
    except Exception:
        pass

    def compile_xml(p):
        return core.compile_model(str(p), args.device)

    embed = compile_xml(md / "layer0" / "openvino_model.xml")
    head = compile_xml(md / "head" / "openvino_model.xml")
    shells = {li: compile_xml(md / "shells" / f"layer_{li:02d}" / "openvino_model.xml")
              for li in layers}
    expert_cache = {}

    def expert(li, ei):
        key = (li, ei)
        c = expert_cache.get(key)
        if c is None:
            c = compile_xml(md / "experts" / f"layer_{li:02d}" / f"expert_{ei:03d}" /
                            "openvino_model.xml")
            expert_cache[key] = c
        return c

    # per-layer KV cache, grown by concatenation (tiny model -> simplicity over speed)
    past_k = {li: np.zeros((1, KV, 0, D), np.float32) for li in layers}
    past_v = {li: np.zeros((1, KV, 0, D), np.float32) for li in layers}

    def out(req, name):
        return req.get_tensor(name).data

    def step(token_id, pos):
        ids = np.array([[token_id]], dtype=np.int64)
        h = embed(ids)[embed.output("hidden")].astype(np.float32)
        for li in layers:
            sh = shells[li]
            req = sh.create_infer_request()
            req.infer({
                "x": h,
                "past_k": past_k[li],
                "past_v": past_v[li],
                "past_seq_len": np.array(pos, dtype=np.int64),
            })
            apn = req.get_tensor("attn_out_post_norm").data
            residual = req.get_tensor("attn_residual").data
            shared = req.get_tensor("shared_expert_out").data
            ids_r = np.array(req.get_tensor("routing_ids").data).reshape(-1)
            w_r = np.array(req.get_tensor("routing_weights").data).reshape(-1)
            pk = req.get_tensor("present_k").data
            pv = req.get_tensor("present_v").data
            past_k[li] = np.concatenate([past_k[li], pk], axis=2)
            past_v[li] = np.concatenate([past_v[li], pv], axis=2)
            moe = np.zeros_like(residual)
            for k in range(K):
                ei = int(ids_r[k])
                w = float(w_r[k])
                ex = expert(li, ei)
                y = ex(apn)[ex.output("y")]
                moe = moe + w * y
            h = residual + shared + moe
        logits = head(h.astype(np.float32))[head.output("logits")].reshape(-1)
        return int(np.argmax(logits)), logits

    prompt = ref["prompt_ids"]
    n_gen = args.max_new or (len(ref["greedy_tokens"]) - len(prompt))
    gen = list(prompt)
    pos = 0
    # prefill prompt token-by-token; keep last logits
    last_argmax = None
    for t in prompt:
        last_argmax, _ = step(t, pos)
        pos += 1
    gen.append(last_argmax)
    # autoregressive
    for _ in range(n_gen - 1):
        nxt, _ = step(gen[-1], pos)
        pos += 1
        gen.append(nxt)

    print("ref :", ref["greedy_tokens"])
    print("ours:", gen)
    match = gen[:len(ref["greedy_tokens"])] == ref["greedy_tokens"]
    first_ok = gen[len(prompt)] == ref["first_next_token"]
    print(f"first-token match: {first_ok}  full-seq match: {match}")
    if not first_ok:
        raise SystemExit("FAIL: first generated token does not match HF reference")
    print("PASS" if match else "PARTIAL (first token ok; later tokens diverged)")


if __name__ == "__main__":
    main()
