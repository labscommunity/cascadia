# Prototype: greedy 64-token decode over the 2-stage shard chain vs
# the whole model. Outputs: token-sequence parity + per-token timing.
# Also decides the staged architecture: if dense-MoE stages are fast enough on
# CPU, monolithic stages suffice; else MoE must be externalized (routed).
import time
import numpy as np
import openvino as ov

DIR = r"C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov"
SH = r"C:\cascadia\models\qwen36-shards-2stage"
HIDDEN, N_TOK = 2048, 64
PROMPT_IDS = [9707, 11, 1246, 525, 498, 30]  # arbitrary real token ids
core = ov.Core()

emb = core.compile_model(DIR + r"\openvino_text_embeddings_model.xml", "CPU")
emb_req = emb.create_infer_request()
def embed(tok):
    t = np.array([[tok]], dtype=np.int64)
    return emb_req.infer({emb.inputs[0].get_any_name(): t})[emb.outputs[0]].astype(np.float32).reshape(1, 1, HIDDEN)

def mk_feeds(model, step, total_len, hidden=None, embeds=None):
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
            f[nm] = np.ones([1, total_len], dtype=et)
        elif "position" in nm:
            arr = np.full([dims[0], 1, 1], step, dtype=et)
            f[nm] = arr
        else:
            f[nm] = np.zeros(dims, dtype=et)
    return f

# simpler explicit implementation
def run_chain(stage_models, label):
    reqs = [(core.compile_model(core.read_model(p), "CPU")) for p in stage_models]
    reqs = [(c, c.create_infer_request()) for c in reqs]
    toks = list(PROMPT_IDS)
    gen = []
    times = []
    step = 0
    logits = None
    # prefill token-by-token (T=1 steps keep state simple), then decode
    for phase in ("prefill", "decode"):
        seq = toks if phase == "prefill" else range(N_TOK)
        for item in seq:
            tok = item if phase == "prefill" else int(np.argmax(logits))
            if phase == "decode":
                gen.append(tok)
            e = embed(tok)
            t0 = time.perf_counter()
            h = e
            for j, (c, r) in enumerate(reqs):
                first = (j == 0)
                feeds = mk_feeds(c, step, step + 1,
                                 hidden=None if first else h,
                                 embeds=h if first else None)
                res = r.infer(feeds)
                h = res[c.outputs[0]].astype(np.float32)
            logits = h.reshape(-1)
            times.append(time.perf_counter() - t0)
            step += 1
    dec = times[len(toks):]
    med = sorted(dec)[len(dec)//2]
    print(f"{label}: {len(gen)} tokens, median decode {med*1000:.0f} ms/tok -> {1/med:.2f} tok/s", flush=True)
    return gen

chain_toks = run_chain([SH + r"\stage0\stage.xml", SH + r"\stage1\stage.xml"], "CHAIN(2-stage)")
full_toks = run_chain([DIR + r"\openvino_language_model.xml"], "FULL")

match = sum(1 for a, b in zip(chain_toks, full_toks) if a == b)
first_div = next((i for i, (a, b) in enumerate(zip(chain_toks, full_toks)) if a != b), len(chain_toks))
print(f"GREEDY PARITY: {match}/{len(full_toks)} tokens match, first divergence at {first_div}", flush=True)
print("M3_PROTO_OK" if first_div >= 32 else "M3_PROTO_DIVERGES_EARLY", flush=True)
print("chain:", chain_toks[:16], flush=True)
print("full :", full_toks[:16], flush=True)
