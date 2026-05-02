"""Inspect what's left referencing beam_idx/attention_mask after transform."""
import openvino as ov, sys
import openvino._offline_transformations as ot

xml = sys.argv[1]
m = ov.Core().read_model(xml)

print("Before transform:")
for inp in m.inputs:
    print(f"  input: {inp.any_name}, consumers: {[c.get_node().get_friendly_name() + ':' + c.get_node().get_type_name() for c in inp.get_target_inputs()][:5]}")

# Try the transform but catch the error and inspect
try:
    ot.paged_attention_transformation(m)
except Exception as e:
    pass

# Now look at what's still referencing the removed params
print("\nAfter (attempted) transform:")
for op in m.get_ops():
    if op.get_friendly_name() in ("beam_idx", "attention_mask"):
        print(f"  Still in graph: {op.get_friendly_name()} ({op.get_type_name()})")
        consumers = []
        for out in op.outputs():
            for ti in out.get_target_inputs():
                consumers.append(ti.get_node().get_friendly_name() + ':' + ti.get_node().get_type_name())
        print(f"    consumed by: {consumers[:10]}")

# Also list the model's params after transform
print(f"\nModel parameters after: {[p.any_name for p in m.inputs]}")
print(f"Model variables after: {len(m.get_variables())}")
print(f"Op types after:")
from collections import Counter
op_types = Counter(op.get_type_name() for op in m.get_ops())
for k, v in sorted(op_types.items()):
    if 'Attention' in k or 'Paged' in k or 'Variable' in k or 'ReadValue' in k or 'Assign' in k:
        print(f"  {k}: {v}")
