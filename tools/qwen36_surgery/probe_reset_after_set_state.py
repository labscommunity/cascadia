"""Probe: does VariableState.reset() scrub state that was installed by set_state()?

probe_reset_state.py already proves reset-after-Assign-driven-inference is bit-exact. The
issue-34 warm-pull path exercises a sequence nothing covers: set_state(donor tensors) ->
reset() -> infer at position 0. On the rig, a qwen36 head that has served ONE cross-chain
warm resume returns deterministic garbage for every later request — including prompts that
provably take the cold path — until the process restarts. llama-8b is unaffected, and the
suspected reason is shape: qwen36-moe carries fixed-shape DeltaNet recurrent states (ssm,
conv) with no sequence dim to collapse to zero, whereas llama is all dynamic-seq-dim KV.

Verdict:
  BROKEN  -> reset() does not undo set_state(); the engine's recreate_request fix is justified.
  OK      -> hypothesis dead. Do NOT ship the fix; re-triage.

Step (f) is the one that validates the fix's mechanism: it checks a fresh InferRequest off the
same CompiledModel is clean, i.e. that a recompile is not required.
"""
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
        if n == "stage_hidden" or "embed" in n:
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


def classify(name):
    """Fixed-shape recurrent (DeltaNet) vs dynamic-seq KV — the suspected discriminator."""
    low = name.lower()
    if "ssm" in low or "conv" in low or "recurrent" in low:
        return "recurrent(fixed)"
    if "key" in low or "value" in low or "present" in low or "past" in low:
        return "kv(dynamic)"
    return "other"


# Baseline: pristine position-0 logits.
req.infer(feed(0, e))
out1 = req.get_output_tensor(0).data.copy()

# Advance so the states hold something non-trivial, then snapshot them —
# this is what get_state_blob serializes on the donor node.
req.infer(feed(1, e))
export = {s.get_name(): np.array(s.state.data, copy=True) for s in req.query_state()}
print("captured states:", len(export))

# (b) Re-apply the snapshot — what set_state_blob does on the acceptor.
for s in req.query_state():
    s.state = ov.Tensor(export[s.get_name()])

# (c) Reset, exactly as reset_all() does on the next cold admission.
for s in req.query_state():
    s.reset()

# (e) Anything still holding data after reset is the leak. Report by class.
residue = {}
for s in req.query_state():
    kind = classify(s.get_name())
    m = float(np.abs(np.array(s.state.data, copy=False)).max()) if s.state.data.size else 0.0
    prev = residue.get(kind, (0.0, 0))
    residue[kind] = (max(prev[0], m), prev[1] + 1)
for kind, (m, n) in sorted(residue.items()):
    print(f"  post-reset residue  {kind:<18} n={n:<4} max|state|={m:.6g}")

# (d) Does a position-0 infer match the pristine baseline?
req.infer(feed(0, e))
out_after = req.get_output_tensor(0).data.copy()
d_after = float(np.abs(out1 - out_after).max())

# (f) Does a FRESH InferRequest off the same CompiledModel come back clean?
#     This is what the engine fix does instead of restarting the process.
req2 = comp.create_infer_request()
req2.infer(feed(0, e))
out_fresh = req2.get_output_tensor(0).data.copy()
d_fresh = float(np.abs(out1 - out_fresh).max())

print()
print("set_state -> reset -> step0   max|diff| vs pristine:", d_after)
print("fresh InferRequest    step0   max|diff| vs pristine:", d_fresh)
print()
print("RESET-AFTER-SET_STATE:", "OK" if d_after < 1e-5 else "BROKEN")
print("FRESH-REQUEST-SUFFICES:", "YES" if d_fresh < 1e-5 else "NO (needs recompile, not just a new request)")
