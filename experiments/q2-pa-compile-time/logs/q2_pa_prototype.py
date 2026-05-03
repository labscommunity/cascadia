"""Q2 prototype: apply SDPAToPagedAttention at compile time on a v3 stage_0 IR.

If this works, the engine can do the same in C++ via ov::pass::SDPAToPagedAttention.
After the pass runs, the model's input list will gain PA inputs we need to bind.
"""
import sys
import openvino as ov
from openvino import passes as ov_pass

STAGE0_XML = sys.argv[1] if len(sys.argv) > 1 else r"C:\cascadia\shards_2stage_v3\stage_0\openvino_model.xml"

print(f"Loading: {STAGE0_XML}", flush=True)
core = ov.Core()
model = core.read_model(STAGE0_XML)

print(f"\nBEFORE pass — inputs ({len(model.inputs)}):", flush=True)
for p in model.inputs:
    print(f"  {sorted(p.get_names())} shape={p.get_partial_shape()}", flush=True)
print(f"BEFORE pass — outputs ({len(model.outputs)}):", flush=True)
for p in model.outputs:
    print(f"  {sorted(p.get_names())} shape={p.get_partial_shape()}", flush=True)

# Apply the pass
print(f"\nApplying ov::pass::SDPAToPagedAttention...", flush=True)
try:
    pa_pass = ov_pass.SDPAToPagedAttention(False, False, False, False, False, False)
    success = pa_pass.run_on_model(model)
    print(f"  pass returned: {success}", flush=True)
except Exception as e:
    print(f"  PASS FAILED: {type(e).__name__}: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

print(f"\nAFTER pass — inputs ({len(model.inputs)}):", flush=True)
for p in model.inputs:
    print(f"  {sorted(p.get_names())} shape={p.get_partial_shape()} dtype={p.get_element_type()}", flush=True)
print(f"AFTER pass — outputs ({len(model.outputs)}):", flush=True)
for p in model.outputs:
    print(f"  {sorted(p.get_names())} shape={p.get_partial_shape()}", flush=True)

print(f"\nAttempting compile on GPU...", flush=True)
try:
    compiled = core.compile_model(model, "GPU")
    print(f"  GPU compile OK, {len(compiled.inputs)} inputs, {len(compiled.outputs)} outputs", flush=True)
except Exception as e:
    print(f"  GPU compile FAILED: {type(e).__name__}: {e}", flush=True)

print(f"\nAttempting compile on CPU as fallback...", flush=True)
try:
    compiled_cpu = core.compile_model(model, "CPU")
    print(f"  CPU compile OK", flush=True)
except Exception as e:
    print(f"  CPU compile FAILED: {type(e).__name__}: {e}", flush=True)
