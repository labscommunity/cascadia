"""Try defaults + introspect what's available."""
import openvino as ov, sys
import openvino._offline_transformations as ot

# List functions
print("Available _offline_transformations:")
for n in sorted(dir(ot)):
    if not n.startswith('_'):
        print(f"  {n}")
print()

# Get docstring
print("paged_attention_transformation help:")
print(ot.paged_attention_transformation.__doc__)

xml = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else "GPU"

# Try with all defaults
print("\n=== Test 1: defaults ===")
m1 = ov.Core().read_model(xml)
print(f"  before: {[p.any_name for p in m1.inputs]}")
try:
    ot.paged_attention_transformation(m1)
    print(f"  after: inputs={[p.any_name for p in m1.inputs]}")
    print(f"  outputs: {[p.any_name for p in m1.outputs]}")
    pa = sum(1 for op in m1.get_ops() if 'PagedAttention' in op.get_type_name())
    sdpa = sum(1 for op in m1.get_ops() if 'ScaledDotProduct' in op.get_type_name())
    print(f"  PA={pa}, SDPA={sdpa}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
