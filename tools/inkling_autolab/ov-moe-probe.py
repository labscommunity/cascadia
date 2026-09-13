"""Eight distinct full-size resident experts on CPU/iGPU; component probe only.

No routing/attention/paging or production backend substitution is included.
Every expert is checked independently at two inputs, before timed execution.
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
ap.add_argument('--schedule', choices=['fused', 'serial', 'async'], required=True)
ap.add_argument('--root', type=Path, default=Path('C:/Users/devcloud/inkling-autolab'))
args = ap.parse_args()
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
spec = importlib.util.spec_from_file_location('export', args.root/'repo/tools/glm5_expert_ov.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
hidden, inter, experts = 6144, 3072, 8
x0 = ((np.arange(hidden) * 17 % 113 - 56) / 56).astype(np.float32).reshape(1, 1, hidden)
inputs = [x0, -x0[:, :, ::-1].copy()]

def bf16(a):
    bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)

def rounded(node):
    return ops.convert(ops.convert(node, ov.Type.bf16), ov.Type.f32)

core = ov.Core()
print('openvino_version=' + ov.__version__, flush=True)
print('device_name=' + core.get_property(args.device, 'FULL_DEVICE_NAME'), flush=True)
config = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'}
if args.device == 'CPU':
    config['INFERENCE_NUM_THREADS'] = 16
parameters, outputs, references, compiled = [], [], [], []
fused_input = ops.parameter([1, 1, hidden], ov.Type.f32, name='x')
compile_seconds = 0.0
for i in range(experts):
    buf = (args.root/f'synthetic-experts/synthetic_{i}.bin').read_bytes()
    wg, off = export._deq(buf, 0, inter, hidden)
    wu, off = export._deq(buf, off, inter, hidden)
    wd, _ = export._deq(buf, off, hidden, inter)
    refs = []
    for x in inputs:
        g = bf16(x.astype(np.float64) @ wg.T.astype(np.float64))
        u = bf16(x.astype(np.float64) @ wu.T.astype(np.float64))
        h = (g / (1 + np.exp(-g))) * u
        refs.append(bf16(h.astype(np.float64) @ wd.T.astype(np.float64)))
    references.append(refs)
    del wg, wu, wd, g, u, h
    wg, off = export._int4_section(buf, 0, inter, hidden)
    wu, off = export._int4_section(buf, off, inter, hidden)
    wd, _ = export._int4_section(buf, off, hidden, inter)
    inp = fused_input if args.schedule == 'fused' else ops.parameter([1, 1, hidden], ov.Type.f32, name='x')
    g = rounded(ops.matmul(inp, wg, False, True))
    u = rounded(ops.matmul(inp, wu, False, True))
    h = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)
    y = rounded(ops.matmul(h, wd, False, True))
    if args.schedule == 'fused':
        outputs.append(y)
    else:
        start = time.perf_counter()
        compiled.append(core.compile_model(ov.Model([y], [inp], f'expert_{i}'), args.device, config))
        compile_seconds += time.perf_counter() - start
if args.schedule == 'fused':
    start = time.perf_counter()
    compiled.append(core.compile_model(ov.Model(outputs, [fused_input], 'eight_experts'), args.device, config))
    compile_seconds += time.perf_counter() - start
print('compile_seconds=' + str(compile_seconds), flush=True)
requests = [c.create_infer_request() for c in compiled]

def infer(x):
    if args.schedule == 'fused':
        requests[0].infer({0: x})
        return [requests[0].get_output_tensor(i).data for i in range(experts)]
    if args.schedule == 'serial':
        for r in requests:
            r.infer({0: x})
    else:
        for r in requests:
            r.start_async({0: x})
        for r in requests:
            r.wait()
    return [r.get_output_tensor(0).data for r in requests]

errors = []
for j, x in enumerate(inputs):
    for i, output in enumerate(infer(x)):
        got = np.asarray(output).reshape(-1).astype(np.float64)
        ref = references[i][j].reshape(-1).astype(np.float64)
        relative_rms = float(np.linalg.norm(got-ref) / max(np.linalg.norm(ref), 1e-30))
        max_scaled = float(np.max(np.abs(got-ref))/max(np.sqrt(np.mean(ref*ref)), 1e-30))
        errors.append({'expert': i, 'input': j, 'relative_rms': relative_rms,
                       'max_error_over_rms': max_scaled, 'finite': bool(np.isfinite(got).all())})
valid = all(e['finite'] and e['relative_rms'] <= 0.005 and e['max_error_over_rms'] <= 0.03 for e in errors)
print('scope=synthetic_eight_resident_experts_numerical_oracle')
print('validation_passed=' + str(int(valid)))
print('oracle_errors_json=' + json.dumps(errors), flush=True)
if not valid:
    raise SystemExit('Independent per-expert numerical oracle rejected result')
for i in range(10):
    infer(inputs[i % 2])
measurements = []
for _ in range(7):
    start = time.perf_counter()
    for i in range(32):
        infer(inputs[i % 2])
    measurements.append((time.perf_counter()-start)*1000/32)
print('experts_ms=' + str(float(np.median(measurements))))
print('timing_samples_json=' + json.dumps(measurements))
