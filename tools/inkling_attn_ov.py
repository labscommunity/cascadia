#!/usr/bin/env python3
"""Inkling attention projections -> per-layer OpenVINO IRs for the iGPU.

Per layer two graphs, both `x [1, rows, in] f32` in, f32 out:
  qkvr: x[.., hidden] -> q [.., Hq*D], k [.., Hkv*D], v [.., Hkv*D], r [.., Hq*d_rel]
        (`attn.wq_du`, `attn.wk_dv`, `attn.wv_dv`, `attn.wr_du`)
  o:    ctx[.., Hq*D] -> [.., hidden]                                  (`attn.wo_ud`)
The head norms, relative-position bias, softmax, KV cache and the short
convolutions stay in Rust (`inkling/attn.rs`); only the five GEMVs move.

Weights (`--weights`, default `int8` = per-row symmetric u8, ~132 MB per layer;
`int4` = group-32 on the experts' grid, ~66 MB, faster but ~10% weight-relative
error against int8's ~1.2%; `--dir-name` keeps variants side by side): the bf16
shells are the ~264 MB per
layer the CPU already streams at ~80 GB/s, so a f16 copy on the iGPU gains
nothing — the gain is in bytes. `int4` quantises each projection on the same
grid as the experts (symmetric, per-row groups of 32, scale = max|w|/7 rounded
to bf16, nibble = q + 8) and writes it in the plugin's compressed-FC layout
(`u4 [out, in/32, 32]` + zero point 8 + f16 scales), ~66 MB per layer.
`f16` keeps the shells' values (bf16 -> f16, exact for the 7 mantissa bits;
tiny values become f16 subnormals) — the exact-ish reference, no byte saving.

`--validate` compiles one layer's `qkvr` on a device and compares with numpy on
the same weights (so it measures the plugin, not the quantisation); the
quantisation error itself is what the layer dump's CPU-vs-iGPU comparison and
`tools/inkling_ref/real_layer_parity.py` measure.

`--head` writes the unembed head instead: `x [1, rows, hidden] -> logits`, the
2.46 GB bf16 GEMV every token pays on the last rank (~31 ms on this CPU).

Layout written (opt-in for the runtime: `CASCADIA_INKLING_OV_ATTN=1`,
`CASCADIA_INKLING_OV_HEAD=1`):
  <out>/attn_ov/layer_NN/qkvr/openvino_model.{xml,bin}
  <out>/attn_ov/layer_NN/o/openvino_model.{xml,bin}
  <out>/head_ov/openvino_model.{xml,bin}

Usage:
  python tools/inkling_attn_ov.py --src /data/inkling-int4 --layers 0,1,2,3
  python tools/inkling_attn_ov.py --src /data/inkling-int4 --layers 2 --validate --validate-device GPU
"""
import argparse
import json
import os
import time

import numpy as np
import openvino as ov
from openvino import Model, PartialShape, Type
from openvino import opset15 as ops
import json as _json
import struct

GROUP = 32


class RawSafetensors:
    """Minimal safetensors reader returning raw bytes per tensor (the numpy
    framework of `safetensors` refuses bf16)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = _json.loads(self.f.read(n))
        self.base = 8 + n

    def get(self, name):
        meta = self.header[name]
        a, b = meta["data_offsets"]
        self.f.seek(self.base + a)
        return meta["dtype"], meta["shape"], self.f.read(b - a)
PROJ = [("q", "attn.wq_du.weight"), ("k", "attn.wk_dv.weight"), ("v", "attn.wv_dv.weight"), ("r", "attn.wr_du.weight")]


def bf16_round(x):
    b = x.astype(np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def load_bf16(st, name):
    """bf16 (or f32) tensor -> f32 numpy [out, in]."""
    dtype, shape, raw = st.get(name)
    if dtype == "BF16":
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, np.float32).reshape(shape)
    raise SystemExit(f"{name}: unexpected dtype {dtype}")


def raw_const(t, dims, raw):
    ten = ov.Tensor(t, ov.Shape(dims))
    view = ten.data if isinstance(ten.data, np.ndarray) else np.frombuffer(ten.data, np.uint8)
    view = view.reshape(-1).view(np.uint8)
    src = np.frombuffer(raw, np.uint8)
    assert view.size == src.size, f"{t} {dims}: {view.size} vs {src.size}"
    view[:] = src
    return ov.op.Constant(ten, shared_memory=False)


def quant_int4(w):
    """f32 [out, in] -> (nibbles u8 [out, in/32, 32] (q+8), scales f32 [out, in/32]) on the experts' grid."""
    out, inn = w.shape
    ng = inn // GROUP
    wg = w.reshape(out, ng, GROUP)
    mx = np.abs(wg).max(-1)
    s = bf16_round(np.where(mx > 0, mx / 7.0, 1.0).astype(np.float32))
    q = np.clip(np.round(wg / s[..., None]), -8, 7).astype(np.int16) + 8
    return q.astype(np.uint8), s


def quant_int8(w):
    """f32 [out, in] -> (u8 [out, in] (q+128), scales f32 [out]) per-row symmetric int8."""
    mx = np.abs(w).max(-1)
    s = bf16_round(np.where(mx > 0, mx / 127.0, 1.0).astype(np.float32))
    q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int16) + 128
    return q.astype(np.uint8), s


def weight_node(w, weights):
    out, inn = w.shape
    if weights == "f16":
        return ops.convert(ops.constant(w.astype(np.float16)), Type.f32)
    if weights == "int8":
        q, s = quant_int8(w)
        wc = ops.constant(q)                                             # u8 [out, in]
        zp = ops.constant(np.full((out, 1), 128, np.uint8))
        x = ops.subtract(ops.convert(wc, Type.f16), ops.convert(zp, Type.f16))
        x = ops.multiply(x, ops.constant(s.reshape(out, 1).astype(np.float16)))
        return ops.convert(x, Type.f32)
    q, s = quant_int4(w)
    ng = inn // GROUP
    n = q.reshape(-1)
    packed = (n[0::2] | (n[1::2] << 4)).astype(np.uint8).tobytes()
    wc = raw_const(Type.u4, [out, ng, GROUP], packed)
    zp = raw_const(Type.u4, [out, ng, 1], np.full((out * ng + 1) // 2, 0x88, np.uint8).tobytes())
    x = ops.subtract(ops.convert(wc, Type.f16), ops.convert(zp, Type.f16))
    x = ops.multiply(x, ops.constant(s.reshape(out, ng, 1).astype(np.float16)))
    x = ops.reshape(x, ops.constant(np.array([out, inn], np.int64)), False)
    return ops.convert(x, Type.f32)


def build_qkvr(ws, hidden, weights):
    x = ops.parameter(PartialShape([1, -1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    outs = []
    for name, w in ws:
        y = ops.matmul(x, weight_node(w, weights), False, True)
        y.get_output_tensor(0).set_names({name})
        outs.append(y)
    return Model(outs, [x], "inkling_attn_qkvr")


def build_o(wo, weights):
    hidden, hqd = wo.shape
    x = ops.parameter(PartialShape([1, -1, hqd]), Type.f32, name="ctx")
    x.get_output_tensor(0).set_names({"ctx"})
    y = ops.matmul(x, weight_node(wo, weights), False, True)
    y.get_output_tensor(0).set_names({"out"})
    return Model([y], [x], "inkling_attn_o")


def head_model(src, weights):
    """unembed [vocab, hidden] -> one FC IR: x [1, rows, hidden] -> logits [1, rows, vocab].
    The RMSNorm and the mup divide stay in Rust; only the 2.46 GB bf16 GEMV moves."""
    st = RawSafetensors(os.path.join(src, "head.safetensors"))
    w = load_bf16(st, "unembed.weight")
    vocab, hidden = w.shape
    x = ops.parameter(PartialShape([1, -1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    y = ops.matmul(x, weight_node(w, weights), False, True)
    y.get_output_tensor(0).set_names({"logits"})
    m = Model([y], [x], "inkling_head")
    return m, w


def layer_models(src, lid, weights):
    st = RawSafetensors(os.path.join(src, "shells", f"layer_{lid:02d}.safetensors"))
    ws = [(n, load_bf16(st, t)) for n, t in PROJ]
    wo = load_bf16(st, "attn.wo_ud.weight")
    hidden = ws[0][1].shape[1]
    return build_qkvr(ws, hidden, weights), build_o(wo, weights), ws, wo


def validate(src, lid, weights, device):
    mq, mo, ws, wo = layer_models(src, lid, weights)
    hidden = ws[0][1].shape[1]
    core = ov.Core()
    cfg = {"INFERENCE_PRECISION_HINT": "f16"}
    t0 = time.time()
    cq = core.compile_model(mq, device, cfg)
    co = core.compile_model(mo, device, cfg)
    print(f"layer {lid}: compiled qkvr + o on {device} in {time.time()-t0:.1f}s ({weights})")
    types = {}
    for op in cq.get_runtime_model().get_ordered_ops():
        lt = op.get_rt_info()["layerType"].astype(str) if "layerType" in op.get_rt_info() else op.get_type_name()
        types[lt] = types.get(lt, 0) + 1
    print("  qkvr runtime layers:", types)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 2, hidden)).astype(np.float32)
    rq = cq.create_infer_request()
    res = rq.infer({"x": x})
    for (name, w), out in zip(ws, res.values()):
        if weights == "int4":
            q, s = quant_int4(w)
            wref = ((q.astype(np.float32) - 8.0) * s[..., None]).reshape(w.shape)
        elif weights == "int8":
            q, s = quant_int8(w)
            wref = (q.astype(np.float32) - 128.0) * s[:, None]
        else:
            wref = w.astype(np.float16).astype(np.float32)
        dq = np.abs(wref - w)
        print(f"  {name} quantisation vs bf16 weights: rel_rms={float(np.sqrt(np.mean(dq*dq))/(np.sqrt(np.mean(w*w))+1e-12)):.3e}")
        ref = x[0] @ wref.T
        d = np.abs(np.array(out).reshape(ref.shape) - ref)
        print(f"  {name}: max_abs={d.max():.3e} rel_rms={float(np.sqrt(np.mean(d*d))/(np.sqrt(np.mean(ref*ref))+1e-12)):.3e}")
    for _ in range(3):
        rq.infer({"x": x})
    n = 30
    t0 = time.time()
    for _ in range(n):
        rq.infer({"x": x})
    tq = 1e3 * (time.time() - t0) / n
    ro = co.create_infer_request()
    ctx = rng.standard_normal((1, 2, wo.shape[1])).astype(np.float32)
    for _ in range(3):
        ro.infer({"ctx": ctx})
    t0 = time.time()
    for _ in range(n):
        ro.infer({"ctx": ctx})
    to = 1e3 * (time.time() - t0) / n
    print(f"  steady 2-row calls: qkvr {tq:.2f} ms, o {to:.2f} ms (sum {tq+to:.2f} ms per layer)")


def validate_head(src, weights, device):
    m, w = head_model(src, weights)
    core = ov.Core()
    t0 = time.time()
    cm = core.compile_model(m, device, {"INFERENCE_PRECISION_HINT": "f16"})
    print(f"head: compiled on {device} in {time.time()-t0:.1f}s ({weights}), unembed {w.shape}")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, w.shape[1])).astype(np.float32) * 0.1
    r = cm.create_infer_request()
    got = np.array(r.infer({"x": x})[0]).reshape(-1)
    if weights == "int8":
        q, s = quant_int8(w); wref = (q.astype(np.float32) - 128.0) * s[:, None]
    elif weights == "int4":
        q, s = quant_int4(w); wref = ((q.astype(np.float32) - 8.0) * s[..., None]).reshape(w.shape)
    else:
        wref = w.astype(np.float16).astype(np.float32)
    ref = (x[0, 0] @ wref.T)
    d = np.abs(got - ref)
    print(f"  vs numpy on the same weights: max_abs={d.max():.3e} rel_rms={float(np.sqrt(np.mean(d*d))/(np.sqrt(np.mean(ref*ref))+1e-12)):.3e}")
    print(f"  argmax match: {int(np.argmax(got)) == int(np.argmax(ref))} (got {int(np.argmax(got))}, ref {int(np.argmax(ref))})")
    dq = np.abs(wref - w)
    print(f"  quantisation vs bf16 weights: rel_rms={float(np.sqrt(np.mean(dq*dq))/(np.sqrt(np.mean(w*w))+1e-12)):.3e}")
    for _ in range(3):
        r.infer({"x": x})
    n = 20; t0 = time.time()
    for _ in range(n):
        r.infer({"x": x})
    print(f"  steady 1-row call: {1e3*(time.time()-t0)/n:.2f} ms")


def main():
    ap = argparse.ArgumentParser(description="Inkling attention projections -> per-layer OpenVINO IRs")
    ap.add_argument("--src", required=True)
    ap.add_argument("--layers", default="", help="comma list of layer indices (not needed with --head)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", choices=["int4", "int8", "f16"], default="int8")
    ap.add_argument("--dir-name", default="attn_ov", help="output subdirectory under --out (runtime: CASCADIA_INKLING_OV_ATTN_DIR)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-device", default="GPU")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--head", action="store_true", help="write (or --validate) the unembed head IR instead of layers")
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.src, "manifest.json")))
    assert man.get("arch") == "inkling", man.get("arch")
    out = args.out or args.src
    if args.head:
        if args.validate:
            validate_head(args.src, args.weights, args.validate_device)
            return
        dst = os.path.join(out, "head_ov")
        if args.skip_existing and os.path.exists(os.path.join(dst, "openvino_model.xml")):
            print("head: exists, skipped")
            return
        t0 = time.time()
        m, _ = head_model(args.src, args.weights)
        os.makedirs(dst, exist_ok=True)
        ov.save_model(m, os.path.join(dst, "openvino_model.xml"), compress_to_fp16=False)
        print(f"head: written ({args.weights}) in {time.time()-t0:.0f}s")
        return
    for lid in [int(v) for v in args.layers.split(",")]:
        if args.validate:
            validate(args.src, lid, args.weights, args.validate_device)
            continue
        dst = os.path.join(out, args.dir_name, f"layer_{lid:02d}")
        if args.skip_existing and os.path.exists(os.path.join(dst, "o", "openvino_model.xml")):
            print(f"layer {lid}: exists, skipped")
            continue
        t0 = time.time()
        mq, mo, _, _ = layer_models(args.src, lid, args.weights)
        os.makedirs(os.path.join(dst, "qkvr"), exist_ok=True)
        os.makedirs(os.path.join(dst, "o"), exist_ok=True)
        ov.save_model(mq, os.path.join(dst, "qkvr", "openvino_model.xml"), compress_to_fp16=False)
        ov.save_model(mo, os.path.join(dst, "o", "openvino_model.xml"), compress_to_fp16=False)
        print(f"layer {lid}: written ({args.weights}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
