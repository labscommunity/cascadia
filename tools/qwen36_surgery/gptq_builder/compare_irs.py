#!/usr/bin/env python
"""Compare two Qwen3.5-MoE VLM-layout OpenVINO IR dirs (text only): prompt logits + greedy decode agreement.
usage: compare_irs.py <dir_A> <dir_B> [--tokens N] [--device CPU] [--prompt "..."]
Runs the language model directly (openvino runtime, stateful), embedding tokens through each dir's text-embeddings IR,
so no GenAI pipeline / chat template is involved. Reports last-position logits agreement and greedy token parity."""
import argparse, json, os, sys, time
import numpy as np
import openvino as ov


def load(core, d, device):
    emb = core.compile_model(core.read_model(os.path.join(d, "openvino_text_embeddings_model.xml")), "CPU")
    lm_model = core.read_model(os.path.join(d, "openvino_language_model.xml"))
    names = {i.get_any_name(): i for i in lm_model.inputs}
    hidden = lm_model.input("inputs_embeds").get_partial_shape()[-1].get_length()
    lm = core.compile_model(lm_model, device)
    return emb, lm, names, hidden


def embed(emb, ids):
    r = emb.create_infer_request()
    out = r.infer({emb.inputs[0].get_any_name(): np.array(ids, dtype=np.int64).reshape(1, -1)})
    return out[emb.outputs[0]].astype(np.float32)


def feeds_for(names, embeds, pos0):
    T = embeds.shape[1]
    f = {"inputs_embeds": embeds}
    if "attention_mask" in names:
        f["attention_mask"] = np.ones((1, pos0 + T), dtype=np.int64)
    if "position_ids" in names:
        ps = names["position_ids"].get_partial_shape()
        rows = ps[0].get_length() if ps[0].is_static else 3
        pos = np.arange(pos0, pos0 + T, dtype=np.int64)
        f["position_ids"] = np.broadcast_to(pos, (rows, 1, T)).copy()
    if "beam_idx" in names:
        f["beam_idx"] = np.zeros((1,), dtype=np.int32)
    return f


def run(d, core, ids, n_dec, device, label):
    emb, lm, names, hidden = load(core, d, device)
    req = lm.create_infer_request()
    req.reset_state()
    t0 = time.time()
    out = req.infer(feeds_for(names, embed(emb, ids), 0))
    logits = out[lm.outputs[0]].astype(np.float32)
    last = logits[0, -1] if logits.ndim == 3 else logits.reshape(-1)
    t_prefill = time.time() - t0
    gen = []
    pos = len(ids)
    t0 = time.time()
    cur = int(last.argmax())
    for _ in range(n_dec):
        gen.append(cur)
        o = req.infer(feeds_for(names, embed(emb, [cur]), pos))
        lg = o[lm.outputs[0]].astype(np.float32)
        cur = int((lg[0, -1] if lg.ndim == 3 else lg.reshape(-1)).argmax())
        pos += 1
    t_dec = time.time() - t0
    print(f"[{label}] hidden={hidden} prefill {len(ids)} tok in {t_prefill:.1f}s, {n_dec} greedy tokens in {t_dec:.1f}s ({n_dec/max(t_dec,1e-9):.2f} tok/s)", flush=True)
    return last, gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()
    core = ov.Core()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.a)
    ids = tok(args.prompt)["input_ids"]
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)
    la, ga = run(args.a, core, ids, args.tokens, args.device, "A")
    lb, gb = run(args.b, core, ids, args.tokens, args.device, "B")
    d = float(np.abs(la - lb).max()); n = float(np.abs(lb).max()) + 1e-9
    top1 = int(la.argmax()) == int(lb.argmax())
    k = 5
    ov5 = len(set(np.argsort(-la)[:k].tolist()) & set(np.argsort(-lb)[:k].tolist()))
    agree = sum(1 for x, y in zip(ga, gb) if x == y)
    print(f"LOGITS last-pos: max_abs={d:.3e} rel={d/n:.3e} top1_match={top1} top5_overlap={ov5}/{k}")
    print(f"GREEDY {args.tokens}: agree={agree}/{args.tokens}")
    print("A:", repr(tok.decode(ga)))
    print("B:", repr(tok.decode(gb)))
    print("RESULT", "PASS" if (top1 and ov5 >= 4 and agree >= int(0.75 * args.tokens)) else "FAIL")


if __name__ == "__main__":
    main()
