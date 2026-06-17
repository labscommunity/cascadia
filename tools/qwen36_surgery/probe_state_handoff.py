import numpy as np, openvino as ov
SH = r"C:\cascadia\models\qwen36-shards-2stage"
PREFILL, STEPS = 32, 8
core = ov.Core()
emb = core.compile_model(SH + r"\openvino_text_embeddings_model.xml", "CPU")
emb_req = emb.create_infer_request()
def embed(ids):
    a = np.array([ids], dtype=np.int64)
    return emb_req.infer({emb.inputs[0].get_any_name(): a})[emb.outputs[0]].astype(np.float32)
PROMPT = (np.arange(PREFILL) % 1000 + 100).tolist()
CONT = [555, 777, 222, 888, 111, 999, 444, 666]  # teacher-forced continuation
def feeds(comp, t0, t1, embeds):
    n = t1 - t0; f = {}
    for inp in comp.inputs:
        nm = inp.get_any_name(); et = inp.get_element_type().to_dtype()
        if "embed" in nm: f[nm] = embeds.astype(et)
        elif "attention_mask" in nm: f[nm] = np.ones([1, t1], dtype=et)
        elif "position" in nm: f[nm] = np.tile(np.arange(t0, t1, dtype=et).reshape(1,1,n), (4,1,1))
        else: f[nm] = np.zeros([1], dtype=et)
    return f
print("compiling stage0 CPU + GPU...", flush=True)
cpu = core.compile_model(core.read_model(SH + r"\stage0\stage.xml"), "CPU")
gpu = core.compile_model(core.read_model(SH + r"\stage0\stage.xml"), "GPU")
E = embed(PROMPT)
def steps(req, comp):
    outs = []
    for i, t in enumerate(CONT):
        h = req.infer(feeds(comp, PREFILL + i, PREFILL + i + 1, embed([t])))[comp.outputs[0]].astype(np.float32)
        outs.append(h.reshape(-1))
    return outs
# A: cpu prefill -> cpu steps
ra = cpu.create_infer_request(); ra.infer(feeds(cpu, 0, PREFILL, E)); A = steps(ra, cpu)
# B: gpu prefill -> export -> import -> cpu steps
rg = gpu.create_infer_request(); rg.infer(feeds(gpu, 0, PREFILL, E))
exported = {s.name: np.array(s.state.data, copy=True) for s in rg.query_state()}
rb = cpu.create_infer_request(); rb.reset_state()
for s in rb.query_state():
    s.state = ov.Tensor(exported[s.name])
B = steps(rb, cpu)
# D: NO state -> cpu steps (negative control)
rd = cpu.create_infer_request(); rd.reset_state(); D = steps(rd, cpu)
def rel(x, y): return float(np.abs(x - y).max() / (np.abs(x).max() + 1e-9))
relAB = [rel(a, b) for a, b in zip(A, B)]
relAD = [rel(a, d) for a, d in zip(A, D)]
print("rel A-vs-B per step:", [f"{r:.1e}" for r in relAB], flush=True)
print("rel A-vs-D per step:", [f"{r:.1e}" for r in relAD], flush=True)
ok = max(relAB) < 0.05 and min(relAD) > 10 * max(relAB)
print(f"STATE_HANDOFF_{'OK' if ok else 'FAIL'} (B tracks A: {max(relAB):.1e}; D diverges: {min(relAD):.1e})", flush=True)
