#!/usr/bin/env python3
"""dsv4 expert -> OV int4 IR. Builds the EXACT dsv4 SwiGLU
(silu(min(w1 x, L)) * clamp(w3 x, +/-L) -> w2), NNCF-int4-compresses, saves an
OV IR whose input is 'x' [1,1,dim] and single output [1,1,dim] (the Rust
dispatch_expert contract). `--validate` compares fp16 & int4 outputs to a numpy
reference of the true dsv4 forward for one expert.

Usage:
  python dsv4_expert_ov.py --bin <expert.bin> --out <dir> [--dim 4096 --inter 2048 --limit 10.0]
  python dsv4_expert_ov.py --bin <expert.bin> --validate
"""
import argparse, os
import numpy as np
import openvino as ov
from openvino import Model, Type
from openvino import opset15 as ops
import nncf, logging
nncf.set_log_level(logging.CRITICAL)

GROUP = 32

def b16(u): return (u.astype(np.uint32) << 16).view(np.float32)

def deq(buf, off, out, inn):
    nb = out * inn // 2; ng = inn // GROUP
    p = np.frombuffer(buf, np.uint8, nb, off).reshape(out, inn // 2)
    s = b16(np.frombuffer(buf, "<u2", out * ng, off + nb).reshape(out, ng))
    q = np.empty((out, inn), np.int32); q[:, 0::2] = (p & 0xF); q[:, 1::2] = (p >> 4); q -= 8
    return (q.astype(np.float32).reshape(out, ng, GROUP) * s[:, :, None]).reshape(out, inn), off + nb + out * ng * 2

def load(binp, dim, inter):
    buf = open(binp, "rb").read()
    w1, o = deq(buf, 0, inter, dim)
    w3, o = deq(buf, o, inter, dim)
    w2, o = deq(buf, o, dim, inter)
    return w1, w3, w2

def ref_forward(w1, w3, w2, x, limit):
    """exact dsv4 Expert::forward math in f32 (no bf16 rounding)."""
    gate = x @ w1.T; up = x @ w3.T
    g = np.minimum(gate, limit)
    u = np.clip(up, -limit, limit)
    silu = g / (1.0 + np.exp(-g))
    h = silu * u
    return h @ w2.T

def build(w1, w3, w2, dim, limit):
    x = ops.parameter([1, 1, dim], Type.f32, name="x")
    def lin(a, w):
        return ops.matmul(a, ops.convert(ops.constant(w.astype(np.float16)), Type.f32), False, True)
    gate = lin(x, w1); up = lin(x, w3)
    g = ops.minimum(gate, ops.constant(np.float32(limit)))
    u = ops.clamp(up, -float(limit), float(limit))
    silu = ops.multiply(g, ops.sigmoid(g))
    h = ops.multiply(silu, u)
    out = lin(h, w2)
    m = Model([out], [x], "dsv4_expert")
    m.outputs[0].tensor.set_names({"out"})
    return m

def to_int4(m):
    return nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT4_SYM, group_size=GROUP, ratio=1.0)

def convert_one(binp, outdir, dim, inter, limit):
    w1, w3, w2 = load(binp, dim, inter)
    os.makedirs(outdir, exist_ok=True)
    ov.save_model(to_int4(build(w1, w3, w2, dim, limit)),
                  os.path.join(outdir, "openvino_model.xml"), compress_to_fp16=False)

if __name__ == "__main__":
    import glob, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin")
    ap.add_argument("--out")
    ap.add_argument("--layer-dir", help="convert every expert_*.bin here")
    ap.add_argument("--out-root", help="dest: <out-root>/expert_EEE/openvino_model.xml")
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--limit", type=float, default=10.0)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    if a.layer_dir:
        bins = sorted(glob.glob(os.path.join(a.layer_dir, "expert_*.bin")))
        t0 = time.perf_counter()
        for i, b in enumerate(bins):
            name = os.path.splitext(os.path.basename(b))[0]  # expert_000 / expert_shared
            convert_one(b, os.path.join(a.out_root, name), a.dim, a.inter, a.limit)
            if (i + 1) % 32 == 0:
                print(f"  {i+1}/{len(bins)} ({(time.perf_counter()-t0):.0f}s)", flush=True)
        print(f"converted {len(bins)} experts -> {a.out_root} in {time.perf_counter()-t0:.0f}s")
        raise SystemExit(0)

    w1, w3, w2 = load(a.bin, a.dim, a.inter)
    m = build(w1, w3, w2, a.dim, a.limit)

    if a.validate:
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((1, 1, a.dim)).astype(np.float32)) * 0.1
        ref = ref_forward(w1, w3, w2, x.reshape(1, a.dim), a.limit).reshape(-1)
        core = ov.Core()
        def run(model):
            r = core.compile_model(model, "CPU", {"SNIPPETS_MODE": "DISABLE"}).create_infer_request()
            return np.array(r.infer({"x": x})[0]).reshape(-1)
        of16 = run(m)
        oi4 = run(to_int4(m))
        def stats(name, got):
            d = np.abs(got - ref); rel = d / (np.abs(ref) + 1e-6)
            print(f"  {name}: max_abs={d.max():.5f} mean_abs={d.mean():.5f} "
                  f"max_rel={rel.max():.3f} cos={np.dot(got,ref)/(np.linalg.norm(got)*np.linalg.norm(ref)+1e-9):.5f}")
        print(f"ref |mean|={np.abs(ref).mean():.4f} max={np.abs(ref).max():.4f}")
        stats("fp16 ", of16)
        stats("int4 ", oi4)
    else:
        os.makedirs(a.out, exist_ok=True)
        ov.save_model(to_int4(m), os.path.join(a.out, "openvino_model.xml"), compress_to_fp16=False)
        print("saved", a.out)
