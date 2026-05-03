"""Q2 step 2: actually run inference on the PA-transformed model.

Confirms we can wire PA inputs end-to-end. Compares output to plain v5 IR
to verify correctness.
"""
import numpy as np
import openvino as ov

V5_STAGE0_XML = r"C:\cascadia\shards_2stage_v5_beam\stage_0\openvino_model.xml"

core = ov.Core()
print("=== PLAIN v5 stage_0 (no PA) ===", flush=True)
plain = core.compile_model(V5_STAGE0_XML, "GPU")
plain_req = plain.create_infer_request()
print(f"  inputs: {[sorted(p.get_names())[0] for p in plain.inputs]}", flush=True)

# 7-token prompt
prompt = np.array([[128000, 70869, 279, 1401, 12062, 1990, 36821]], dtype=np.int64)  # [1, 7]
n = prompt.shape[1]
attn = np.ones((1, n), dtype=np.int64)
pos = np.arange(n, dtype=np.int64).reshape(1, n)
beam = np.zeros(1, dtype=np.int32)

plain_req.set_input_tensor(0, ov.Tensor(prompt))
plain_req.set_input_tensor(1, ov.Tensor(attn))
plain_req.set_input_tensor(2, ov.Tensor(pos))
plain_req.set_input_tensor(3, ov.Tensor(beam))
plain_req.infer()
plain_out = plain_req.get_output_tensor(0).data.copy()
print(f"  output shape: {plain_out.shape}, sum={plain_out.sum():.4f}", flush=True)

print("\n=== PA-transformed v5 stage_0 ===", flush=True)
model = core.read_model(V5_STAGE0_XML)
from openvino._offline_transformations import paged_attention_transformation
try:
    paged_attention_transformation(model)
except RuntimeError as e:
    # Expected — register dangling parameters and continue
    print(f"  caught (expected): {str(e)[:100]}", flush=True)
unregistered = [n for n in model.get_ops() if n.get_type_name() == "Parameter" and n not in model.get_parameters()]
if unregistered:
    model.add_parameters(unregistered)
    model.validate_nodes_and_infer_types()
pa = core.compile_model(model, "GPU")
pa_req = pa.create_infer_request()
print(f"  inputs: {[sorted(p.get_names())[0] for p in pa.inputs]}", flush=True)

# Build PA inputs for a 7-token prefill (1 sequence)
# past_lens [1] = 0 (cache is empty for prefill)
past_lens = np.array([0], dtype=np.int32)
# subsequence_begins [2] = [0, 7]
sub_begins = np.array([0, n], dtype=np.int32)
# block_indices: each block holds 32 tokens by default. For 7 tokens we need 1 block.
block_size = 32
needed_blocks = (n + block_size - 1) // block_size
block_indices = np.arange(needed_blocks, dtype=np.int32)
block_indices_begins = np.array([0, needed_blocks], dtype=np.int32)
max_context_len = np.array(n, dtype=np.int32)  # scalar

input_map = {p.get_names().pop(): p for p in pa.inputs}
def feed(name, arr):
    pa_req.set_tensor(name, ov.Tensor(arr))

feed("input_ids", prompt)
feed("position_ids", pos.flatten())  # [?]: flat
feed("past_lens", past_lens)
feed("subsequence_begins", sub_begins)
feed("block_indices", block_indices)
feed("block_indices_begins", block_indices_begins)
feed("max_context_len", max_context_len)
# Dangling but compiled — provide stub values
feed("attention_mask", attn)
feed("beam_idx", beam)

try:
    pa_req.infer()
    pa_out = pa_req.get_output_tensor(0).data.copy()
    print(f"  output shape: {pa_out.shape}, sum={pa_out.sum():.4f}", flush=True)
    # Compare last token's hidden state
    diff = np.abs(plain_out[0, -1, :] - pa_out[0, -1, :]).mean()
    print(f"  mean abs diff (last token hidden): {diff:.6f}", flush=True)
    if diff < 0.1:
        print(f"  AGREEMENT: PA path matches plain path within tolerance", flush=True)
    else:
        print(f"  WARNING: PA path differs from plain (might be OK due to PA quantization)", flush=True)
except Exception as e:
    print(f"  INFER FAILED: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()
