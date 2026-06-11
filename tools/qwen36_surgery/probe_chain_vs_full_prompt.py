"""3-way discriminator for an engine/full greedy divergence: run the
Python 2-stage chain on the SAME prompt the engine served. chain==engine
and chain!=full at the same token => f16 fusion-order noise (inherent to
the sharding); chain==full => engine bug."""
import json

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

DIR = r"C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov"
SH = r"C:\cascadia\models\qwen36-shards-2stage"
HIDDEN, N_TOK = 2048, 64
PROMPT = "user: Explain how rainbows form."

tok = Tokenizer.from_file(SH + r"\tokenizer.json")
prompt_ids = tok.encode(PROMPT).ids
eos = json.load(open(SH + r"\generation_config.json"))["eos_token_id"]
if isinstance(eos, list):
    eos = eos[0]
print("prompt_ids:", prompt_ids, flush=True)

core = ov.Core()
emb = core.compile_model(DIR + r"\openvino_text_embeddings_model.xml", "CPU")
emb_req = emb.create_infer_request()

def embed(t):
    a = np.array([[t]], dtype=np.int64)
    return emb_req.infer({emb.inputs[0].get_any_name(): a})[emb.outputs[0]].astype(np.float32).reshape(1, 1, HIDDEN)

def mk_feeds(model, step, hidden=None, embeds=None):
    f = {}
    for inp in model.inputs:
        nm = inp.get_any_name()
        ps = inp.get_partial_shape()
        dims = [(d.get_length() if d.is_static else 1) for d in ps]
        et = inp.get_element_type().to_dtype()
        if nm == "stage_hidden":
            f[nm] = hidden.astype(et)
        elif "embed" in nm:
            dims[-1] = HIDDEN
            f[nm] = (embeds if embeds is not None else np.zeros(dims)).astype(et)
        elif "attention_mask" in nm:
            f[nm] = np.ones([1, step + 1], dtype=et)
        elif "position" in nm:
            f[nm] = np.full([dims[0], 1, 1], step, dtype=et)
        else:
            f[nm] = np.zeros(dims, dtype=et)
    return f

def run_chain(stage_paths, label):
    comps = [core.compile_model(core.read_model(p), "CPU") for p in stage_paths]
    reqs = [(c, c.create_infer_request()) for c in comps]
    step = 0
    logits = None
    def one(t, step):
        h = embed(t)
        out = None
        for j, (c, r) in enumerate(reqs):
            first = j == 0
            f = mk_feeds(c, step, hidden=None if first else out, embeds=h if first else None)
            out = r.infer(f)[c.outputs[0]].astype(np.float32)
        return out.reshape(-1)
    for t in prompt_ids:
        logits = one(t, step)
        step += 1
    gen = []
    for _ in range(N_TOK):
        nxt = int(np.argmax(logits))
        if nxt == eos:
            break
        gen.append(nxt)
        logits = one(nxt, step)
        step += 1
    print(f"{label}_IDS:", gen, flush=True)
    print(f"{label}_TEXT_JSON:", json.dumps(tok.decode(gen, skip_special_tokens=True)), flush=True)
    return gen

chain = run_chain([SH + r"\stage0\stage.xml", SH + r"\stage1\stage.xml"], "CHAIN")
full = run_chain([DIR + r"\openvino_language_model.xml"], "FULL")
match = sum(1 for a, b in zip(chain, full) if a == b)
first_div = next((i for i, (a, b) in enumerate(zip(chain, full)) if a != b), min(len(chain), len(full)))
print(f"CHAIN_VS_FULL: {match}/{min(len(chain), len(full))} match, first divergence at {first_div}", flush=True)
