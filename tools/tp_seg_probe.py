"""Per-infer cost of one TP segment on the GPU (clean wall + GPU-busy).
Tells us the per-segment OV dispatch floor: 2L+2 segments * this = the TP token cost
that segment-collapse (megakernel/graph-replay) would have to beat."""
import sys, time
import numpy as np
import openvino as ov

xml = sys.argv[1]
H = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
core = ov.Core()

def mk_feed(cm):
    feed = {}
    for p in cm.inputs:
        nm = p.get_any_name()
        et = p.get_element_type().get_type_name()
        if nm == "hidden_states":
            feed[p] = ov.Tensor(np.zeros((1, 1, H), dtype=np.float32))
        elif nm == "attention_mask":
            feed[p] = ov.Tensor(np.ones((1, 1), dtype=np.int64))
        elif nm == "position_ids":
            feed[p] = ov.Tensor(np.zeros((1, 1), dtype=np.int64))
        elif nm == "input_ids":
            feed[p] = ov.Tensor(np.zeros((1, 1), dtype=np.int64))
        elif nm == "beam_idx":
            feed[p] = ov.Tensor(np.zeros((1,), dtype=np.int32))
        else:  # past_key_values.* (decode: seq dim 1)
            shape = [(d.get_length() if d.is_static else 1) for d in p.get_partial_shape()]
            feed[p] = ov.Tensor(np.zeros(shape, dtype=np.float16 if "f16" in et else np.float32))
    return feed

# clean wall (no profiling)
cm = core.compile_model(xml, "GPU")
ir = cm.create_infer_request()
feed = mk_feed(cm)
for _ in range(30):
    ir.infer(feed)
t0 = time.perf_counter()
for _ in range(N):
    ir.infer(feed)
wall = (time.perf_counter() - t0) / N

# gpu-busy via profiling pass
cm2 = core.compile_model(xml, "GPU", {"PERF_COUNT": True})
ir2 = cm2.create_infer_request()
feed2 = mk_feed(cm2)
for _ in range(10):
    ir2.infer(feed2)
busy = sum(p.real_time.total_seconds() for p in ir2.get_profiling_info() if "NOT_RUN" not in str(p.status))
print(f"RESULT {xml.split(chr(92))[-2]} per_infer_wall_us={wall*1e6:.1f} gpu_busy_us={busy*1e6:.1f}")
