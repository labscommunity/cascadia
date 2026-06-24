"""Validate the TP OpenVINO export by running the segmented chain in Python
(local all-reduce = sum of rank partials) and comparing to the HF fp32 reference.

If the fp16 export matches the reference greedy tokens, the segmentation + slicing
+ stateful-KV export is correct and the Rust tp-runtime engine just has to
replicate this chain with a NETWORK all-reduce instead of a local sum.
"""
import json, os, sys
import numpy as np
import openvino as ov
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SHARD = sys.argv[1]
DEV = sys.argv[2] if len(sys.argv) > 2 else "CPU"
NGEN = int(sys.argv[3]) if len(sys.argv) > 3 else 20

pc = json.load(open(os.path.join(SHARD, "pipeline_config.json")))
TP, L = pc["tp_size"], pc["num_layers"]
tok = AutoTokenizer.from_pretrained(os.path.join(SHARD, "tokenizer"))
print(f"shard tp={TP} L={L} dev={DEV}")

core = ov.Core()
def compile_seg(rank, name):
    cm = core.compile_model(os.path.join(SHARD, f"rank_{rank}", name, "openvino_model.xml"), DEV)
    return cm.create_infer_request(), [p.get_any_name() for p in cm.inputs]

# compile all segments
reqs = {}  # (rank,name) -> (infer_request, input_names)
for r in range(TP):
    rc = json.load(open(os.path.join(SHARD, f"rank_{r}", "rank_config.json")))
    for s in rc["segments"]:
        reqs[(r, s["name"])] = compile_seg(r, s["name"])
print(f"compiled {len(reqs)} segments")

def run(rank, name, feeds):
    ir, innames = reqs[(rank, name)]
    fd = {}
    for n in innames:
        if n in feeds:
            fd[n] = feeds[n]
        elif n == "beam_idx":
            fd[n] = np.zeros((1,), dtype=np.int32)
    ir.infer(fd)
    return ir.get_output_tensor(0).data.copy()

def reset_all():
    for (r, name), (ir, _) in reqs.items():
        if name.startswith("attn_"):
            ir.reset_state()

def tp_forward(input_ids, position):
    seq = input_ids.shape[1]
    total = position + seq
    ids = input_ids.astype(np.int64)
    mask = np.ones((1, total), dtype=np.int64)
    pos = np.arange(position, position + seq, dtype=np.int64)[None, :]
    # embed (rank 0; replicated so identical) -> hidden f32
    hidden = run(0, "embed", {"input_ids": ids}).astype(np.float32)
    for i in range(L):
        pa = np.zeros_like(hidden)
        for r in range(TP):
            pr = run(r, f"attn_{i}", {"hidden_states": hidden.astype(np.float32),
                                      "attention_mask": mask, "position_ids": pos})
            pa = pa + pr.astype(np.float32)          # all-reduce(attn)
        hidden = hidden + pa
        pm = np.zeros_like(hidden)
        for r in range(TP):
            pr = run(r, f"mlp_{i}", {"hidden_states": hidden.astype(np.float32)})
            pm = pm + pr.astype(np.float32)          # all-reduce(mlp)
        hidden = hidden + pm
    logits = run(0, "head", {"hidden_states": hidden.astype(np.float32)})
    return logits

def greedy_tp(prompt_ids, n):
    reset_all()
    out = list(prompt_ids)
    logits = tp_forward(np.array([prompt_ids]), 0)     # prefill
    nxt = int(np.argmax(logits[0, -1])); out.append(nxt)
    pos = len(prompt_ids)
    for _ in range(n - 1):
        logits = tp_forward(np.array([[nxt]]), pos)     # decode
        nxt = int(np.argmax(logits[0, -1])); out.append(nxt); pos += 1
    return out[len(prompt_ids):]

# reference
print("loading HF fp32 reference...")
model = AutoModelForCausalLM.from_pretrained(pc["model_id"], torch_dtype=torch.float32).eval()
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids
ref = model.generate(ids, max_new_tokens=NGEN, do_sample=False)[0, ids.shape[1]:].tolist()

with torch.no_grad():
    tp = greedy_tp(ids[0].tolist(), NGEN)
match = sum(1 for a, b in zip(ref, tp) if a == b)
print(f"REF: {ref}")
print(f"TP : {tp}")
print(f"match {match}/{NGEN}")
print(f"TP text: {tok.decode(tp)!r}")
ok = match >= NGEN - 1   # allow 1 late divergence from fp rounding
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
