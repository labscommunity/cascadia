#!/usr/bin/env python
"""Standalone greedy generation on one Qwen3.5-MoE VLM-layout IR dir (text only), to check coherence.
usage: gen_check.py <ir_dir> [--tokens N] [--prompt ...] [--device CPU]"""
import argparse, os, time, numpy as np, openvino as ov
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("--tokens", type=int, default=8); ap.add_argument("--prompt", default="The capital of France is"); ap.add_argument("--device", default="CPU")
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.d); ids = tok(a.prompt)["input_ids"]
core = ov.Core()
emb = core.compile_model(core.read_model(os.path.join(a.d, "openvino_text_embeddings_model.xml")), "CPU")
lm_model = core.read_model(os.path.join(a.d, "openvino_language_model.xml")); names = {i.get_any_name(): i for i in lm_model.inputs}
t0 = time.time(); lm = core.compile_model(lm_model, a.device); print(f"compiled in {time.time()-t0:.0f}s", flush=True)
def embed(x): return emb.create_infer_request().infer({emb.inputs[0].get_any_name(): np.array(x, dtype=np.int64).reshape(1, -1)})[emb.outputs[0]].astype(np.float32)
def feeds(e, pos0):
    T = e.shape[1]; f = {"inputs_embeds": e}
    if "attention_mask" in names: f["attention_mask"] = np.ones((1, pos0 + T), dtype=np.int64)
    if "position_ids" in names:
        ps = names["position_ids"].get_partial_shape(); rows = ps[0].get_length() if ps[0].is_static else 3
        f["position_ids"] = np.broadcast_to(np.arange(pos0, pos0 + T, dtype=np.int64), (rows, 1, T)).copy()
    if "beam_idx" in names: f["beam_idx"] = np.zeros((1,), dtype=np.int32)
    return f
req = lm.create_infer_request(); req.reset_state()
t0 = time.time(); out = req.infer(feeds(embed(ids), 0)); lg = out[lm.outputs[0]].astype(np.float32); last = lg[0, -1] if lg.ndim == 3 else lg.reshape(-1)
print(f"prefill {len(ids)} tok {time.time()-t0:.1f}s; top5 first token: {[(tok.decode([int(i)]), round(float(last[i]),2)) for i in np.argsort(-last)[:5]]}", flush=True)
gen = []; cur = int(last.argmax()); pos = len(ids); t0 = time.time()
for _ in range(a.tokens):
    gen.append(cur); o = req.infer(feeds(embed([cur]), pos)); l2 = o[lm.outputs[0]].astype(np.float32); cur = int((l2[0, -1] if l2.ndim == 3 else l2.reshape(-1)).argmax()); pos += 1
print(f"greedy {a.tokens} tokens in {time.time()-t0:.1f}s: {tok.decode(gen)!r}")
print("NAN_CHECK:", "nan/inf in logits" if not np.isfinite(last).all() else "finite")
