"""Validate: does applying paged_attention_transformation actually speed up?"""
import openvino as ov, time, sys, json
from openvino._offline_transformations import paged_attention_transformation

xml = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else "GPU"

# Path A: raw compile_model (current ov-runtime path)
print(f"=== Path A: raw compile_model ===", flush=True)
core_a = ov.Core()
m_a = core_a.read_model(xml)
print(f"  inputs: {[p.any_name for p in m_a.inputs]}")
print(f"  SDPA ops: {sum(1 for op in m_a.get_ops() if 'ScaledDotProduct' in op.get_type_name())}")
t0 = time.perf_counter()
compiled_a = core_a.compile_model(m_a, device)
print(f"  compile_a: {time.perf_counter()-t0:.2f}s", flush=True)

# Path B: apply PA transform first, then compile
print(f"=== Path B: PA transform applied ===", flush=True)
core_b = ov.Core()
m_b = core_b.read_model(xml)
try:
    paged_attention_transformation(m_b, allow_score_aggregation=True)
    print(f"  PA transform applied OK", flush=True)
    print(f"  inputs after: {[p.any_name for p in m_b.inputs]}")
    print(f"  PA ops: {sum(1 for op in m_b.get_ops() if 'PagedAttention' in op.get_type_name())}")
    print(f"  SDPA ops: {sum(1 for op in m_b.get_ops() if 'ScaledDotProduct' in op.get_type_name())}")
    print(f"  outputs: {[p.any_name for p in m_b.outputs]}")
    t0 = time.perf_counter()
    compiled_b = core_b.compile_model(m_b, device)
    print(f"  compile_b: {time.perf_counter()-t0:.2f}s", flush=True)
except Exception as e:
    print(f"  PA transform FAILED: {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)

print("=== summary ===")
print(json.dumps({
    "path_a_inputs": [p.any_name for p in m_a.inputs],
    "path_b_inputs": [p.any_name for p in m_b.inputs],
    "path_b_outputs": [p.any_name for p in m_b.outputs],
}, indent=2))
