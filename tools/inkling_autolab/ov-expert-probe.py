"""Explore CPU/iGPU execution of one full-sized synthetic Inkling expert.

This is an exploratory component measurement, NOT full-model tokens/s. The
reference uses the same int4 grid and bf16 activation boundaries, with f64 dots
as an independent numerical oracle. No checkpoint or production backend changes.
"""
import argparse
import ctypes
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset15 as ops

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--device', choices=['CPU', 'GPU'], required=True)
ap.add_argument('--root', type=Path, default=Path('C:/Users/devcloud/inkling-autolab'))
args = ap.parse_args()
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
spec = importlib.util.spec_from_file_location('export', args.root/'repo/tools/glm5_expert_ov.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
hidden, inter = 6144, 3072
path = args.root/'synthetic-experts/synthetic_0.bin'
buf = path.read_bytes()
x = ((np.arange(hidden) * 17 % 113 - 56) / 56).astype(np.float32).reshape(1, 1, hidden)

def bf16(a):
    bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)

wg, off = export._deq(buf, 0, inter, hidden)
wu, off = export._deq(buf, off, inter, hidden)
wd, _ = export._deq(buf, off, hidden, inter)
g = bf16(x.astype(np.float64) @ wg.T.astype(np.float64))
u = bf16(x.astype(np.float64) @ wu.T.astype(np.float64))
h = (g / (1 + np.exp(-g))) * u
reference = bf16(h.astype(np.float64) @ wd.T.astype(np.float64))
del wg, wu, wd, g, u, h

wg, off = export._int4_section(buf, 0, inter, hidden)
wu, off = export._int4_section(buf, off, inter, hidden)
wd, _ = export._int4_section(buf, off, hidden, inter)
inp = ops.parameter([1, 1, hidden], ov.Type.f32, name='x')
def rounded(node):
    return ops.convert(ops.convert(node, ov.Type.bf16), ov.Type.f32)
g = rounded(ops.matmul(inp, wg, False, True))
u = rounded(ops.matmul(inp, wu, False, True))
h = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)
y = rounded(ops.matmul(h, wd, False, True))
model = ov.Model([y], [inp], 'inkling_expert_probe')
core = ov.Core()
print('openvino_version=' + ov.__version__, flush=True)
print('device_name=' + core.get_property(args.device, 'FULL_DEVICE_NAME'), flush=True)
config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'}
if args.device == 'CPU':
    config['INFERENCE_NUM_THREADS'] = 16
start = time.perf_counter()
compiled = core.compile_model(model, args.device, config)
print('compile_seconds=' + str(time.perf_counter()-start), flush=True)
request = compiled.create_infer_request()
for _ in range(10):
    request.infer({0: x})
measurements = []
for _ in range(7):
    start = time.perf_counter()
    for _ in range(64):
        output = request.infer({0: x})[0]
    measurements.append((time.perf_counter()-start)*1000/64)
got = np.asarray(output).reshape(-1).astype(np.float64)
ref = reference.reshape(-1).astype(np.float64)
relative_rms = float(np.linalg.norm(got-ref) / max(np.linalg.norm(ref), 1e-30))
max_scaled = float(np.max(np.abs(got-ref))/max(np.sqrt(np.mean(ref*ref)), 1e-30))
valid = bool(np.isfinite(got).all() and relative_rms <= 0.005 and max_scaled <= 0.03)
print('scope=synthetic_single_expert_numerical_oracle')
print('expert_ms=' + str(float(np.median(measurements))))
print('relative_rms=' + str(relative_rms))
print('max_error_over_rms=' + str(max_scaled))
print('validation_passed=' + str(int(valid)))
print('timing_samples_json=' + json.dumps(measurements))
if not valid:
    raise SystemExit('Independent numerical oracle rejected the component result')
