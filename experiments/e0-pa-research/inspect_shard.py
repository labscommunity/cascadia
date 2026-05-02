"""Inspect a v5 stage shard to see if PagedAttention is baked in."""
import openvino as ov, sys
core = ov.Core()
xml = sys.argv[1]
print(f"=== Inspecting {xml} ===")
m = core.read_model(xml)
print("\nINPUTS:")
for p in m.inputs:
    name = p.any_name
    shape = p.partial_shape
    print(f"  {name}: shape={shape}, type={p.element_type}")
print(f"\nTotal inputs: {len(m.inputs)}")
print("\nOUTPUTS:")
for p in m.outputs:
    name = p.any_name
    print(f"  {name}: shape={p.partial_shape}")
print(f"\nVariables (stateful KV): {len(m.get_variables())}")
for v in m.get_variables()[:3]:
    print(f"  {v.get_info()}")
print(f"  ... ({len(m.get_variables())} total)")

# Look for PagedAttention op
ops = [op.get_type_name() for op in m.get_ops()]
pa_count = sum(1 for op in ops if "PagedAttention" in op)
sdpa_count = sum(1 for op in ops if "ScaledDotProduct" in op or "SDPA" in op)
print(f"\nGraph ops: {len(ops)} total")
print(f"  PagedAttention*: {pa_count}")
print(f"  SDPA / ScaledDotProductAttention: {sdpa_count}")

# Check for paged attention specific input names
pa_inputs = ["max_context_len", "past_lens", "subsequence_begins", "block_indices", "block_indices_begins"]
input_names = {p.any_name for p in m.inputs}
have_pa_inputs = [n for n in pa_inputs if n in input_names]
print(f"\nPA-specific inputs present: {have_pa_inputs}")
print(f"\nVERDICT: {'PA already baked in' if pa_count > 0 else 'NOT paged — SDPA still present'}")
