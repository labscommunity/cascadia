# M2' brick 3: SHELL EXTRACTION prototype. Cut layer 0 (a DeltaNet layer)
# out of the official IR as a standalone stage model: new hidden-state
# Parameter at the layer boundary, layer-0 state (conv.0/ssm.0) preserved
# as ReadValue/Assign, validated against the full model's layer-0 output.
import numpy as np
import openvino as ov
from openvino import opset13 as ops

DIR = r"C:\cascadia\models\Qwen3.6-35B-A3B-int4-ov"
XML = DIR + r"\openvino_language_model.xml"
HIDDEN = 2048
L = "layers.3"  # first full-attention layer (3:1 pattern)
core = ov.Core()

def build_feeds(model, embeds=None):
    feeds = {}
    for inp in model.inputs:
        nm = inp.get_any_name()
        ps = inp.get_partial_shape()
        dims = [(d.get_length() if d.is_static else 1) for d in ps]
        et = inp.get_element_type().to_dtype()
        if "embed" in nm or nm == "stage_hidden":
            dims[-1] = HIDDEN
            feeds[nm] = embeds.astype(et) if embeds is not None else ((np.random.rand(*dims) - 0.5) * 0.05).astype(et)
        elif "attention_mask" in nm:
            feeds[nm] = np.ones(dims, dtype=et)
        else:
            feeds[nm] = np.zeros(dims, dtype=et)
    return feeds

# ---------- pass 1: full model, tap layer-0 input + output ----------
model = core.read_model(XML)
boundary_in = boundary_out = None
for op in model.get_ops():
    n = op.get_friendly_name()
    if n.endswith(f"{L}.input_layernorm/aten::pow/Power"):
        boundary_in = op.input_value(0)          # layer-0 input hidden
    elif n.endswith(f"{L}/aten::add/Add_1"):
        boundary_out = op.output(0)              # layer-0 output hidden
assert boundary_in is not None and boundary_out is not None, "boundary not found"
model.add_outputs([boundary_in, boundary_out])
comp_full = core.compile_model(model, "CPU")
req_full = comp_full.create_infer_request()
embeds = ((np.random.rand(1, 1, HIDDEN) - 0.5) * 0.05).astype(np.float32)
res = req_full.infer(build_feeds(model, embeds))
x_in = res[comp_full.outputs[-2]].astype(np.float32)
y_ref = res[comp_full.outputs[-1]].astype(np.float32)
print("tap shapes:", x_in.shape, y_ref.shape, "y_norm:", float(np.abs(y_ref).max()), flush=True)
del comp_full, req_full

# ---------- pass 2: extract layer 0 as standalone stage ----------
model2 = core.read_model(XML)
b_in = b_out = None
for op in model2.get_ops():
    n = op.get_friendly_name()
    if n.endswith(f"{L}.input_layernorm/aten::pow/Power"):
        b_in = op.input_value(0)
    elif n.endswith(f"{L}/aten::add/Add_1"):
        b_out = op.output(0)

# new Parameter for the stage's hidden input; rewire ALL consumers of the
# boundary tensor (input_layernorm chain + residual add)
param = ops.parameter(b_in.get_partial_shape(), b_in.get_element_type(), name="stage_hidden")
param.output(0).set_names({"stage_hidden"})
for target_in in list(b_in.get_target_inputs()):
    target_in.replace_source_output(param.output(0))

# collect layer-0 state sinks (Assign ops whose variable_id mentions .0 conv/ssm)
sinks = []
for op in model2.get_ops():
    if op.get_type_name() == "Assign":
        vid = op.get_variable_id()
        if ".key.0c" in vid or ".value.0c" in vid or vid.endswith("key.0") or vid.endswith("value.0"):
            sinks.append(op)
print("layer-0 assigns:", [a.get_variable_id()[:60] for a in sinks], flush=True)

# determine which original Parameters remain upstream of the stage
result = ops.result(b_out)
needed_params = [param]
import collections
seen = set()
stack = [result] + sinks
reach_params = set()
while stack:
    node = stack.pop()
    if node.get_instance_id() in seen:
        continue
    seen.add(node.get_instance_id())
    for iv in node.input_values():
        src = iv.get_node()
        if src.get_type_name() == "Parameter" and src.get_friendly_name() != "stage_hidden":
            reach_params.add(src)
        stack.append(src)
orig_params = [p for p in model2.get_parameters() if p in reach_params]
print("reachable original params:", [p.get_friendly_name() for p in orig_params], flush=True)

stage = ov.Model([result], sinks, [param] + orig_params, "qwen36_stage_l3")
print("stage built: inputs:", [i.get_any_name() for i in stage.inputs], flush=True)

ov.save_model(stage, r"C:\cascadia\models\qwen36-stage-l3\stage.xml", compress_to_fp16=False)
print("stage saved", flush=True)

comp_s = core.compile_model(stage, "CPU")
req_s = comp_s.create_infer_request()
feeds = build_feeds(stage, x_in)
y_stage = req_s.infer(feeds)[comp_s.outputs[0]].astype(np.float32)

d = float(np.abs(y_stage - y_ref).max()); n = float(np.abs(y_ref).max()) + 1e-9
print(f"STAGE PARITY max_abs={d:.3e} rel={d/n:.3e}", flush=True)
print("SHELL_EXTRACT_OK" if d/n < 1e-2 else "SHELL_EXTRACT_FAIL", flush=True)
