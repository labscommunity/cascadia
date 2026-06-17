"""M4' spike 3 (final pre-engine gate): full 2-stage GREEDY decode after
GPU-prefill state handoff, vs the blessed CPU-pure golden.

Memory-sequenced for a 31.6 GB node (run with cascadia-node STOPPED):
  1. stage0 on GPU: prefill prompt, export states + hidden; free GPU model
  2. stage1 on GPU: prefill from saved hidden, export states; free
  3. stage0+stage1 on CPU: import states, greedy decode 16 tokens
  4. compare token-level vs golden/qwen36_parity_64.json (engine ≡ chain
     CPU-pure reference)

Gate: coherent continuation + early-token agreement. GPU/CPU f16 regime
mixing makes full parity impossible; report the match prefix length.
"""
import gc
import json

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

SH = r"C:\cascadia\models\qwen36-shards-2stage"
GOLDEN = r"C:\Users\devcloud\Workspaces\cascadia-oss\tools\qwen36_surgery\golden\qwen36_parity_64.json"
HIDDEN, N_DEC = 2048, 16

tok = Tokenizer.from_file(SH + r"\tokenizer.json")
g = json.load(open(GOLDEN))
prompt_ids = tok.encode(g["prompt"]).ids
golden_ids = [int(x) for x in g["ids"][:N_DEC]]
P = len(prompt_ids)
print(f"prompt {P} tokens; golden head: {golden_ids[:8]}", flush=True)

core = ov.Core()
emb = core.compile_model(SH + r"\openvino_text_embeddings_model.xml", "CPU")
emb_req = emb.create_infer_request()

def embed(ids):
    a = np.array([ids], dtype=np.int64)
    return emb_req.infer({emb.inputs[0].get_any_name(): a})[emb.outputs[0]].astype(np.float32)

def feeds(comp, t0, t1, embeds=None, hidden=None):
    n = t1 - t0
    f = {}
    for inp in comp.inputs:
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

def export_states(req):
    return {s.name: np.array(s.state.data, copy=True) for s in req.query_state()}

E = embed(prompt_ids)

# 1. stage0 GPU prefill
print("stage0 GPU prefill...", flush=True)
s0g = core.compile_model(core.read_model(SH + r"\stage0\stage.xml"), "GPU")
r0g = s0g.create_infer_request()
h0 = r0g.infer(feeds(s0g, 0, P, embeds=E))[s0g.outputs[0]].astype(np.float32)
st0 = export_states(r0g)
del r0g, s0g
gc.collect()

# 2. stage1 GPU prefill (from stage0's GPU hidden)
print("stage1 GPU prefill...", flush=True)
s1g = core.compile_model(core.read_model(SH + r"\stage1\stage.xml"), "GPU")
r1g = s1g.create_infer_request()
out = r1g.infer(feeds(s1g, 0, P, hidden=h0))[s1g.outputs[0]].astype(np.float32)
st1 = export_states(r1g)
logits = out.reshape(out.shape[1], -1)[-1]
del r1g, s1g
gc.collect()

# 3. CPU stages, import states, greedy decode
print("compiling CPU stages...", flush=True)
s0c = core.compile_model(core.read_model(SH + r"\stage0\stage.xml"), "CPU")
s1c = core.compile_model(core.read_model(SH + r"\stage1\stage.xml"), "CPU")
r0 = s0c.create_infer_request()
r1 = s1c.create_infer_request()
r0.reset_state()
r1.reset_state()
for s in r0.query_state():
    s.state = ov.Tensor(st0[s.name])
for s in r1.query_state():
    s.state = ov.Tensor(st1[s.name])
print("states imported; decoding...", flush=True)

gen = []
for i in range(N_DEC):
    nxt = int(np.argmax(logits))
    gen.append(nxt)
    e = embed([nxt])
    h = r0.infer(feeds(s0c, P + i, P + i + 1, embeds=e))[s0c.outputs[0]].astype(np.float32)
    o = r1.infer(feeds(s1c, P + i, P + i + 1, hidden=h))[s1c.outputs[0]].astype(np.float32)
    logits = o.reshape(o.shape[1], -1)[-1]

match = sum(1 for a, b in zip(gen, golden_ids) if a == b)
prefix = next((i for i, (a, b) in enumerate(zip(gen, golden_ids)) if a != b), N_DEC)
print("handoff:", gen, flush=True)
print("golden :", golden_ids, flush=True)
print("text   :", json.dumps(tok.decode(gen, skip_special_tokens=True)), flush=True)
print(f"FULL_CHAIN_HANDOFF match {match}/{N_DEC}, agree-prefix {prefix}", flush=True)
