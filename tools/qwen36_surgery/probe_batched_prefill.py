"""GPU-prefill-split design probe, part 1 (CPU): do the exported stage
IRs accept T>1, and how much TTFT does batched prefill buy vs the
engine's T=1 loop? Also: does decode continued from batched-prefill
states match decode continued from sequential-prefill states (greedy)?

Run with the cascadia server STOPPED (RAM).
"""
import json
import time

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

DIR = r"C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov"
SH = r"C:\cascadia\models\qwen36-shards-2stage"
HIDDEN = 2048
N_DECODE = 8  # continuation tokens to compare

tok = Tokenizer.from_file(SH + r"\tokenizer.json")
base = (
    "user: Summarize the history of distributed computing, covering "
    "mainframes, clusters, grids, clouds, and modern edge meshes. "
)
prompt_ids = tok.encode(base * 12).ids[:512]
T = len(prompt_ids)
print(f"prompt tokens: {T}", flush=True)

core = ov.Core()
emb = core.compile_model(DIR + r"\openvino_text_embeddings_model.xml", "CPU")
emb_req = emb.create_infer_request()

def embed_seq(ids):
    a = np.array([ids], dtype=np.int64)  # [1, T]
    return emb_req.infer({emb.inputs[0].get_any_name(): a})[emb.outputs[0]].astype(np.float32)

stages = [core.compile_model(core.read_model(p), "CPU")
          for p in (SH + r"\stage0\stage.xml", SH + r"\stage1\stage.xml")]
for j, c in enumerate(stages):
    print(f"stage{j} inputs:", {i.get_any_name(): str(i.get_partial_shape()) for i in c.inputs}, flush=True)

def feeds(model, t0, t1, hidden=None, embeds=None):
    """One pass covering absolute positions [t0, t1): T = t1-t0, mask length t1."""
    n = t1 - t0
    f = {}
    for inp in model.inputs:
        nm = inp.get_any_name()
        et = inp.get_element_type().to_dtype()
        if nm == "stage_hidden":
            f[nm] = hidden.astype(et)
        elif "embed" in nm:
            f[nm] = (embeds if embeds is not None
                     else np.zeros((1, n, HIDDEN))).astype(et)
        elif "attention_mask" in nm:
            f[nm] = np.ones([1, t1], dtype=et)
        elif "position" in nm:
            f[nm] = np.tile(np.arange(t0, t1, dtype=et).reshape(1, 1, n), (4, 1, 1))
        else:
            f[nm] = np.zeros([1], dtype=et)
    return f

def run_pass(reqs, t0, t1, embeds):
    h = embeds
    for j, (c, r) in enumerate(reqs):
        first = j == 0
        f = feeds(c, t0, t1, hidden=None if first else h, embeds=h if first else None)
        h = r.infer(f)[c.outputs[0]].astype(np.float32)
    return h  # last stage output [1, T, vocab]

def decode_from(reqs, start_step, first_logits, n):
    gen = []
    logits = first_logits
    for _ in range(n):
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        e = embed_seq([nxt])
        out = run_pass(reqs, start_step + len(gen) - 1, start_step + len(gen), e)
        logits = out.reshape(out.shape[1], -1)[-1]
    return gen

E = embed_seq(prompt_ids)  # [1, T, 2048]

# --- A: sequential T=1 prefill (engine's current behavior) ---
reqs_a = [(c, c.create_infer_request()) for c in stages]
t0 = time.perf_counter()
logits = None
for i in range(T):
    out = run_pass(reqs_a, i, i + 1, E[:, i:i + 1, :])
    logits = out.reshape(-1)
seq_wall = time.perf_counter() - t0
gen_a = decode_from(reqs_a, T, logits, N_DECODE)
print(f"SEQUENTIAL prefill: {seq_wall:.1f}s ({T/seq_wall:.1f} tok/s)", flush=True)
print("seq decode:", gen_a, flush=True)

# --- B: batched single-pass prefill ---
reqs_b = [(c, c.create_infer_request()) for c in stages]
t0 = time.perf_counter()
out = run_pass(reqs_b, 0, T, E)
bat_wall = time.perf_counter() - t0
logits_b = out.reshape(out.shape[1], -1)[-1]
gen_b = decode_from(reqs_b, T, logits_b, N_DECODE)
print(f"BATCHED prefill: {bat_wall:.1f}s ({T/bat_wall:.1f} tok/s)", flush=True)
print("bat decode:", gen_b, flush=True)

match = sum(1 for a, b in zip(gen_a, gen_b) if a == b)
print(f"DECODE PARITY batched-vs-sequential: {match}/{N_DECODE}", flush=True)
print(f"TTFT_SPEEDUP {seq_wall/bat_wall:.1f}x", flush=True)
