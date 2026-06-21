"""Probe: does VariableState.reset() actually restore position-0 behavior
on a surgically-extracted stage IR? Repro for the engine's empty-response
bug (request 1 coherent, requests 2..N empty)."""
import numpy as np
import openvino as ov

D = r"C:\cascadia\models\qwen36-shards-2stage"

core = ov.Core()
comp = core.compile_model(D + r"\stage0\stage.xml", "CPU")
req = comp.create_infer_request()
print("num states:", len(req.query_state()))

emb = core.compile_model(D + r"\openvino_text_embeddings_model.xml", "CPU")
e = emb(np.array([[1000]], dtype=np.int64))[emb.outputs[0]]

def feed(step, hidden):
    feeds = {}
    for inp in comp.inputs:
        n = inp.get_any_name()
        if n == "stage_hidden":
            feeds[n] = np.asarray(hidden, dtype=np.float32).reshape(1, 1, -1)
        elif "embed" in n:
            feeds[n] = np.asarray(hidden, dtype=np.float32).reshape(1, 1, -1)
        elif "attention_mask" in n:
            feeds[n] = np.ones((1, step + 1), dtype=np.int64)
        elif "position" in n:
            feeds[n] = np.full((4, 1, 1), step, dtype=np.int64)
        elif "beam" in n:
            feeds[n] = np.zeros((1,), dtype=np.int32)
        else:
            raise SystemExit(f"unexpected input {n}")
    return feeds

req.infer(feed(0, e))
out1 = req.get_output_tensor(0).data.copy()
req.infer(feed(1, e))
out2 = req.get_output_tensor(0).data.copy()

for s in req.query_state():
    s.reset()

req.infer(feed(0, e))
out3 = req.get_output_tensor(0).data.copy()

print("step0 rerun-after-reset max|diff| vs fresh:", float(np.abs(out1 - out3).max()))
print("step1 (state engaged) max|diff| vs step0:  ", float(np.abs(out1 - out2).max()))
print("RESET", "OK" if np.abs(out1 - out3).max() < 1e-5 else "BROKEN")
