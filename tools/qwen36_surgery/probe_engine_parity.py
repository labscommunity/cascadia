"""M3' acceptance: whole-model greedy reference for engine parity.

Mirrors the engine exactly: legacy prompt render ("role: content"),
tokenizer.json with special tokens, T=1 prefill, greedy decode with
EOS-stop-before-emit, decode(skip_special_tokens). Prints the reference
text as JSON for comparison against the engine's API output.
"""
import json
import time

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

DIR = r"C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov"
SH = r"C:\cascadia\models\qwen36-shards-2stage"
HIDDEN, N_TOK = 2048, 64
USER = "Explain how rainbows form."
PROMPT = "user: " + USER

tok = Tokenizer.from_file(SH + r"\tokenizer.json")
prompt_ids = tok.encode(PROMPT).ids
eos = json.load(open(SH + r"\generation_config.json"))["eos_token_id"]
if isinstance(eos, list):
    eos = eos[0]
print("prompt_tokens:", len(prompt_ids), "eos:", eos, flush=True)

core = ov.Core()
emb = core.compile_model(DIR + r"\openvino_text_embeddings_model.xml", "CPU")
emb_req = emb.create_infer_request()

def embed(t):
    a = np.array([[t]], dtype=np.int64)
    return emb_req.infer({emb.inputs[0].get_any_name(): a})[emb.outputs[0]].astype(np.float32).reshape(1, 1, HIDDEN)

full = core.compile_model(DIR + r"\openvino_language_model.xml", "CPU")
req = full.create_infer_request()

def feeds(step):
    f = {}
    for inp in full.inputs:
        nm = inp.get_any_name()
        ps = inp.get_partial_shape()
        dims = [(d.get_length() if d.is_static else 1) for d in ps]
        et = inp.get_element_type().to_dtype()
        if "embed" in nm:
            continue  # set separately
        elif "attention_mask" in nm:
            f[nm] = np.ones([1, step + 1], dtype=et)
        elif "position" in nm:
            f[nm] = np.full([dims[0], 1, 1], step, dtype=et)
        else:
            f[nm] = np.zeros(dims, dtype=et)
    return f

emb_name = next(i.get_any_name() for i in full.inputs if "embed" in i.get_any_name())

def run(t, step):
    f = feeds(step)
    f[emb_name] = embed(t)
    return req.infer(f)[full.outputs[0]].astype(np.float32).reshape(-1)

step = 0
logits = None
t0 = time.perf_counter()
for t in prompt_ids:
    logits = run(t, step)
    step += 1
gen = []
for _ in range(N_TOK):
    nxt = int(np.argmax(logits))
    if nxt == eos:
        break
    gen.append(nxt)
    logits = run(nxt, step)
    step += 1
dt = time.perf_counter() - t0

text = tok.decode(gen, skip_special_tokens=True)
print("ref_tokens:", len(gen), f"wall {dt:.1f}s", flush=True)
print("REF_TEXT_JSON:", json.dumps(text), flush=True)
print("REF_IDS:", gen, flush=True)
