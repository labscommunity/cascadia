"""Q2 step 3: measure per-token speedup of PA-transformed v5 stage_0 vs plain v5.

This is the CRITICAL test. If PA gives ≥30% per-token speedup, the engine
work is worth it. If <10%, abandon PA path.
"""
import time
import numpy as np
import openvino as ov

V5_STAGE0_XML = r"C:\cascadia\shards_2stage_v5_beam\stage_0\openvino_model.xml"
N_DECODE = 64  # decode this many tokens
PROMPT = np.array([[128000, 70869, 279, 1401, 12062, 1990, 36821]], dtype=np.int64)

core = ov.Core()

print("=== PLAIN v5 stage_0 (no PA) ===", flush=True)
plain = core.compile_model(V5_STAGE0_XML, "GPU")
plain_req = plain.create_infer_request()

n0 = PROMPT.shape[1]
beam = np.zeros(1, dtype=np.int32)

# Warmup
plain_req.reset_state()
plain_req.set_input_tensor(0, ov.Tensor(PROMPT))
plain_req.set_input_tensor(1, ov.Tensor(np.ones((1, n0), dtype=np.int64)))
plain_req.set_input_tensor(2, ov.Tensor(np.arange(n0, dtype=np.int64).reshape(1, n0)))
plain_req.set_input_tensor(3, ov.Tensor(beam))
plain_req.infer()
for i in range(3):
    plain_req.set_input_tensor(0, ov.Tensor(np.array([[100]], dtype=np.int64)))
    plain_req.set_input_tensor(1, ov.Tensor(np.ones((1, n0+1+i), dtype=np.int64)))
    plain_req.set_input_tensor(2, ov.Tensor(np.array([[n0+i]], dtype=np.int64)))
    plain_req.set_input_tensor(3, ov.Tensor(beam))
    plain_req.infer()

# Time plain decode
plain_req.reset_state()
plain_req.set_input_tensor(0, ov.Tensor(PROMPT))
plain_req.set_input_tensor(1, ov.Tensor(np.ones((1, n0), dtype=np.int64)))
plain_req.set_input_tensor(2, ov.Tensor(np.arange(n0, dtype=np.int64).reshape(1, n0)))
plain_req.set_input_tensor(3, ov.Tensor(beam))
plain_req.infer()
t0 = time.perf_counter()
for i in range(N_DECODE):
    plain_req.set_input_tensor(0, ov.Tensor(np.array([[100]], dtype=np.int64)))
    plain_req.set_input_tensor(1, ov.Tensor(np.ones((1, n0+1+i), dtype=np.int64)))
    plain_req.set_input_tensor(2, ov.Tensor(np.array([[n0+i]], dtype=np.int64)))
    plain_req.set_input_tensor(3, ov.Tensor(beam))
    plain_req.infer()
plain_dt = time.perf_counter() - t0
print(f"  {N_DECODE} decode steps: {plain_dt*1000:.1f} ms ({plain_dt/N_DECODE*1000:.2f} ms/token)", flush=True)

print("\n=== PA-transformed v5 stage_0 ===", flush=True)
model = core.read_model(V5_STAGE0_XML)
from openvino._offline_transformations import paged_attention_transformation
try:
    paged_attention_transformation(model)
except RuntimeError:
    pass
unregistered = [n for n in model.get_ops() if n.get_type_name() == "Parameter" and n not in model.get_parameters()]
if unregistered:
    model.add_parameters(unregistered)
    model.validate_nodes_and_infer_types()
pa = core.compile_model(model, "CPU")  # CPU avoids GPU OOM during PA cache preallocation
pa_req = pa.create_infer_request()

block_size = 32  # PA default
def pa_inputs(seq_len_so_far, n_new):
    total = seq_len_so_far + n_new
    needed_blocks = (total + block_size - 1) // block_size
    return {
        "past_lens": np.array([seq_len_so_far], dtype=np.int32),
        "subsequence_begins": np.array([0, n_new], dtype=np.int32),
        "block_indices": np.arange(needed_blocks, dtype=np.int32),
        "block_indices_begins": np.array([0, needed_blocks], dtype=np.int32),
        "max_context_len": np.array(total, dtype=np.int32),
    }

def feed(name, arr):
    pa_req.set_tensor(name, ov.Tensor(arr))

# Warmup PA
feed("input_ids", PROMPT)
feed("position_ids", np.arange(n0, dtype=np.int64))
feed("attention_mask", np.ones((1, n0), dtype=np.int64))
feed("beam_idx", beam)
for k, v in pa_inputs(0, n0).items():
    feed(k, v)
pa_req.infer()
for i in range(3):
    cur = n0 + i
    feed("input_ids", np.array([[100]], dtype=np.int64))
    feed("position_ids", np.array([cur], dtype=np.int64))
    feed("attention_mask", np.ones((1, cur+1), dtype=np.int64))
    feed("beam_idx", beam)
    for k, v in pa_inputs(cur, 1).items():
        feed(k, v)
    pa_req.infer()

# Time PA decode
# Reset state via re-prefill
feed("input_ids", PROMPT)
feed("position_ids", np.arange(n0, dtype=np.int64))
feed("attention_mask", np.ones((1, n0), dtype=np.int64))
feed("beam_idx", beam)
for k, v in pa_inputs(0, n0).items():
    feed(k, v)
pa_req.infer()
t0 = time.perf_counter()
for i in range(N_DECODE):
    cur = n0 + i
    feed("input_ids", np.array([[100]], dtype=np.int64))
    feed("position_ids", np.array([cur], dtype=np.int64))
    feed("attention_mask", np.ones((1, cur+1), dtype=np.int64))
    feed("beam_idx", beam)
    for k, v in pa_inputs(cur, 1).items():
        feed(k, v)
    pa_req.infer()
pa_dt = time.perf_counter() - t0
print(f"  {N_DECODE} decode steps: {pa_dt*1000:.1f} ms ({pa_dt/N_DECODE*1000:.2f} ms/token)", flush=True)

speedup = (plain_dt - pa_dt) / plain_dt * 100
print(f"\n=== RESULT ===", flush=True)
print(f"  Plain: {plain_dt/N_DECODE*1000:.2f} ms/token", flush=True)
print(f"  PA:    {pa_dt/N_DECODE*1000:.2f} ms/token", flush=True)
print(f"  PA is {speedup:+.1f}% faster than plain", flush=True)
