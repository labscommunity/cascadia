"""Q2 v2: try PA pass on V5 stage_0 IR (canonical SDPA inputs).

Also try via openvino_genai's higher-level API since the C++ pass isn't bound to Python.
"""
import sys
import openvino as ov

V5_STAGE0_XML = r"C:\cascadia\shards_2stage_v5_beam\stage_0\openvino_model.xml"
print(f"Loading: {V5_STAGE0_XML}", flush=True)
core = ov.Core()
model = core.read_model(V5_STAGE0_XML)

print(f"\nBEFORE — inputs ({len(model.inputs)}):", flush=True)
for p in model.inputs:
    print(f"  {sorted(p.get_names())} shape={p.get_partial_shape()} dtype={p.get_element_type()}", flush=True)

# Try via the offline_transformations API used in v5 export
try:
    from openvino._offline_transformations import paged_attention_transformation
    print(f"\nApplying paged_attention_transformation (offline_transformations)...", flush=True)
    paged_attention_transformation(model)
    print("  OK", flush=True)
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

print(f"\nAFTER — inputs ({len(model.inputs)}):", flush=True)
for p in model.inputs:
    print(f"  {sorted(p.get_names())} shape={p.get_partial_shape()} dtype={p.get_element_type()}", flush=True)

# Hunt for unregistered Parameters that the pass added but didn't register
unregistered = [n for n in model.get_ops() if n.get_type_name() == "Parameter" and n not in model.get_parameters()]
print(f"  Unregistered Parameter ops: {len(unregistered)}", flush=True)
for u in unregistered[:10]:
    print(f"    - {u.get_friendly_name()} shape={u.get_partial_shape()}", flush=True)
if unregistered:
    model.add_parameters(unregistered)
    model.validate_nodes_and_infer_types()
    print(f"  Registered them. Now inputs: {len(model.inputs)}", flush=True)
    for p in model.inputs:
        print(f"  {sorted(p.get_names())} shape={p.get_partial_shape()} dtype={p.get_element_type()}", flush=True)

print(f"\nCompile on GPU...", flush=True)
try:
    compiled = core.compile_model(model, "GPU")
    print(f"  GPU compile OK, inputs={len(compiled.inputs)}, outputs={len(compiled.outputs)}", flush=True)
    for p in compiled.inputs:
        print(f"    in: {sorted(p.get_names())} shape={p.get_partial_shape()}", flush=True)
except Exception as e:
    print(f"  GPU compile FAILED: {type(e).__name__}: {e}", flush=True)
