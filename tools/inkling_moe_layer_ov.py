#!/usr/bin/env python3
"""Inkling MoE layer -> one OpenVINO IR per layer that the Intel GPU plugin fuses into its
`moe_3gemm_fused_compressed` kernel (OpenVINO 2026.3+).

The graph is the "tiled 3-GEMM MoE block" the plugin's `ConvertTiledMoeBlockToGatherMatmuls`
pass matches (the same shape optimum-intel exports for Qwen3-MoE): Tile the rows over the
experts, three batched MatMuls against expert-major constants, SwiGLU, and a routing map
built by scattering the top-k weights into a zero [rows, experts] tensor. Two things differ
from an HF export, deliberately:

* the router is NOT in the graph: `topk_indices [rows, K] i32` and `routing_weights
  [rows, K] f32` are Parameters, fed by the Rust `inkling_gate` (sigmoid + bias top-k,
  normalisation over selected + shared, route_scale, global_scale) — Inkling's routing is
  not one of the plugin's fused router types, and keeping it in Rust keeps it exact;
* the two shared experts are the last two experts of the stack (ids `num_experts + s`),
  always selected with their gammas as weights, so K = top_k + n_shared and the op's
  `num_shared_expert` is 0.

Weights: the bins' own int4 nibbles and bf16 group-32 scales, re-expressed as the plugin's
compressed layout — by default `u4` constants `[E, out, in/32, 32]` verbatim with an explicit
zero point of 8 (so every weight constant sits in the IR .bin with an offset, which the
plugin's offload path needs), or `--layout i4` (signed nibbles, no zero point) — and `f16`
scales `[E, out, in/32, 1]` (bf16 -> f16 is exact for the 7 mantissa bits; scales below
6.1e-5 lose precision as f16 subnormals — reported by `--validate`). Gate/up run in the
plugin's f16 path; there is no f32 mode for this kernel.

Layout written (opt-in for the runtime: `CASCADIA_INKLING_OV_MOE=1`):
  <out>/moe_ov/layer_NN/openvino_model.{xml,bin}      (MoE layers only; ~8.3 GB each incl. 4 dummy experts)

Known plugin behaviour on the Arc B390 (driver 32.0.101.8860, OV 2026.3.1): the batched-GEMV
decode kernel crashes the process; the runtime sets OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=0 so
decode takes the grouped-GEMM path, and a single-row call still crashes there, so the
runtime pads decode to two rows (the second row a copy with zero weights).

Usage:
  python tools/inkling_moe_layer_ov.py --src /data/inkling-int4 --layers 2,3
  python tools/inkling_moe_layer_ov.py --src /data/inkling-int4 --layers 2 --validate   # compile + numpy check
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import openvino as ov
from openvino import Model, PartialShape, Type
from openvino import opset15 as ops

try:
    from openvino.utils.node_factory import NodeFactory
except ImportError:  # older layout
    from openvino.runtime.utils.node_factory import NodeFactory

GROUP = 32


def _bf16_to_f16(u16):
    """bf16 bit patterns -> f16 values (via f32; exact unless the value is out of f16's range)."""
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float16)


def read_bin_sections(path, hidden, inter):
    """One int4_bin -> ((nibbles u8 [out, in/2], scales bf16 u16 [out, in/32]) for gate, up, down)."""
    buf = np.fromfile(path, np.uint8)
    out = []
    off = 0
    for (o, i) in ((inter, hidden), (inter, hidden), (hidden, inter)):
        nb = o * i // 2
        ng = i // GROUP
        packed = buf[off:off + nb].reshape(o, i // 2)
        off += nb
        scales = buf[off:off + o * ng * 2].view("<u2").reshape(o, ng)
        off += o * ng * 2
        out.append((packed, scales))
    return out


def raw_const(t, dims, raw):
    ten = ov.Tensor(t, ov.Shape(dims))
    view = ten.data if isinstance(ten.data, np.ndarray) else np.frombuffer(ten.data, np.uint8)
    view = view.reshape(-1).view(np.uint8)
    src = np.frombuffer(raw, np.uint8)
    assert view.size == src.size, f"{t} {dims}: {view.size} vs {src.size}"
    view[:] = src
    return ov.op.Constant(ten, shared_memory=False)


def stacked_weight(packed_list, scale_list, out, inn, layout="u4zp"):
    """Expert-major compressed constant + f16 scales -> the plugin's decompression chain -> [E, out, in] f32.

    `u4zp` (default): the bins' nibbles verbatim as `u4` with an explicit zero point of 8 —
    every weight constant then lives in the IR .bin with an offset, which the plugin's
    offload path (OFFLOAD_RATIO / WEIGHTS_PATH) requires. `i4`: signed nibbles (bins' bytes
    with the high bit of each nibble flipped), no zero point — the same values, the layout
    optimum-intel's `--sym` export uses; the plugin cannot offload it (2026.3.1 asks the
    zero-point placeholder for a bin offset)."""
    e = len(packed_list)
    ng = inn // GROUP
    raw = np.concatenate([p.reshape(-1) for p in packed_list])
    sc = np.stack([_bf16_to_f16(s) for s in scale_list]).reshape(e, out, ng, 1)
    if layout == "i4":
        w = raw_const(Type.i4, [e, out, ng, GROUP], (raw ^ np.uint8(0x88)).tobytes())
        w = ops.convert(w, Type.f16)
    else:
        w = raw_const(Type.u4, [e, out, ng, GROUP], raw.tobytes())
        zp = raw_const(Type.u4, [e, out, ng, 1], np.full((e * out * ng + 1) // 2, 0x88, np.uint8).tobytes())
        w = ops.subtract(ops.convert(w, Type.f16), ops.convert(zp, Type.f16))
    w = ops.multiply(w, ops.constant(sc))
    w = ops.reshape(w, ops.constant(np.array([e, out, inn], np.int64)), False)
    return ops.convert(w, Type.f32)


def build_layer(gate_w, up_w, down_w, hidden, inter, n_total, k):
    """The tiled 3-GEMM MoE block with Parameter-fed routing; `n_total` experts, `k` selected per row."""
    x = ops.parameter(PartialShape([1, -1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    idx = ops.parameter(PartialShape([-1, k]), Type.i32, name="topk_indices")
    idx.get_output_tensor(0).set_names({"topk_indices"})
    rw = ops.parameter(PartialShape([-1, k]), Type.f32, name="routing_weights")
    rw.get_output_tensor(0).set_names({"routing_weights"})

    xr = ops.reshape(x, ops.constant(np.array([-1, hidden], np.int64)), False)          # [T, H]
    tile = ops.tile(xr, ops.constant(np.array([n_total, 1], np.int64)))                 # [E*T, H]
    xt = ops.reshape(tile, ops.constant(np.array([n_total, -1, hidden], np.int64)), False)  # [E, T, H]
    g = ops.matmul(xt, gate_w, False, True)                                             # [E, T, I]
    u = ops.matmul(xt, up_w, False, True)
    sw = NodeFactory("opset4").create("Swish", [g.output(0)], {})                       # single-input Swish (matcher requirement)
    h = ops.multiply(sw, u)
    d = ops.matmul(h, down_w, False, True)                                              # [E, T, H]
    d3 = ops.reshape(d, ops.constant(np.array([n_total, 0, hidden], np.int64)), True)   # [E, T, H]
    ishape = ops.shape_of(idx, output_type="i64")
    tdim = ops.gather(ishape, ops.constant(np.array([0], np.int64)), ops.constant(np.array(0, np.int64)))
    zshape = ops.concat([tdim, ops.constant(np.array([n_total], np.int64))], 0)         # [T, E]
    zeros = ops.broadcast(ops.constant(np.array(0.0, np.float32)), zshape)
    routing = ops.scatter_elements_update(zeros, idx, rw, ops.constant(np.array(1, np.int64)))  # [T, E]
    rt = ops.transpose(routing, ops.constant(np.array([1, 0], np.int32)))               # [E, T]
    rr = ops.reshape(rt, ops.constant(np.array([n_total, 0], np.int64)), True)
    ru = ops.unsqueeze(rr, ops.constant(np.array(2, np.int64)))                          # [E, T, 1]
    y = ops.reduce_sum(ops.multiply(d3, ru), ops.constant(np.array([0], np.int64)), False)  # [T, H]
    y = ops.reshape(y, ops.constant(np.array([1, -1, hidden], np.int64)), False)        # [1, T, H]
    m = Model([y], [x, idx, rw], "inkling_moe_layer")
    m.outputs[0].tensor.set_names({"y"})
    return m


def layer_model(src, lid, man, layout="u4zp", pad_experts=4):
    hidden, inter = man["hidden_size"], man["moe_intermediate"]
    n_exp, n_sh, k = man["num_experts"], man["n_shared_experts"], man["top_k"]
    edir = os.path.join(src, "experts", f"layer_{lid:02d}")
    paths = [os.path.join(edir, f"expert_{e:03d}.bin") for e in range(n_exp)] + \
            [os.path.join(edir, f"expert_shared{s}.bin") for s in range(n_sh)]
    gates, ups, downs = [], [], []
    for p in paths:
        (gp, gs), (up_, us), (dp, ds) = read_bin_sections(p, hidden, inter)
        gates.append((gp, gs)); ups.append((up_, us)); downs.append((dp, ds))
    # Dummy experts (value 0, scale 0) after the real ones: OpenVINO 2026.3.1
    # compiles a saved fused-MoE IR only through its offload path, whose
    # smallest ratio (1%) leaves floor(0.99 * E) resident slots; with E padded
    # so that floor(0.99 * E') >= E every real expert fits in a slot at once
    # and steady-state decode never streams. 4 dummies cover 258 (259 slots).
    for _ in range(pad_experts):
        gates.append((np.full_like(gates[0][0], 0x88), np.zeros_like(gates[0][1])))
        ups.append((np.full_like(ups[0][0], 0x88), np.zeros_like(ups[0][1])))
        downs.append((np.full_like(downs[0][0], 0x88), np.zeros_like(downs[0][1])))
    n_total = n_exp + n_sh + pad_experts
    assert (n_total * 99) // 100 >= n_exp + n_sh, "pad_experts too small for the 1% offload floor"
    gate_w = stacked_weight([g[0] for g in gates], [g[1] for g in gates], inter, hidden, layout)
    up_w = stacked_weight([u[0] for u in ups], [u[1] for u in ups], inter, hidden, layout)
    down_w = stacked_weight([d[0] for d in downs], [d[1] for d in downs], hidden, inter, layout)
    return build_layer(gate_w, up_w, down_w, hidden, inter, n_total, k + n_sh)


def validate(src, lid, man, device="GPU", layout="u4zp", pad_experts=4):
    """Compile the layer and compare one 2-row call against a numpy reference on the bins' grid."""
    from glm5_expert_ov import _load  # noqa: E402  (dequantised gate/up/down of one bin)
    hidden, inter = man["hidden_size"], man["moe_intermediate"]
    n_exp, n_sh, k = man["num_experts"], man["n_shared_experts"], man["top_k"]
    m = layer_model(src, lid, man, layout, pad_experts)
    core = ov.Core()
    t0 = time.time()
    cm = core.compile_model(m, device, {"INFERENCE_PRECISION_HINT": "f16"})
    print(f"layer {lid}: compiled on {device} in {time.time()-t0:.1f}s")
    types = {}
    for op in cm.get_runtime_model().get_ordered_ops():
        lt = op.get_rt_info()["layerType"].astype(str) if "layerType" in op.get_rt_info() else op.get_type_name()
        types[lt] = types.get(lt, 0) + 1
    fused = any("moe" in t.lower() for t in types)
    print(f"  runtime layers: {types}\n  FUSED: {fused}")
    if not fused:
        raise SystemExit("the plugin did not fuse the layer")
    rng = np.random.default_rng(0)
    rows = 2
    x = (rng.standard_normal((1, rows, hidden)).astype(np.float32)) * 0.5
    K = k + n_sh
    sel = np.zeros((rows, K), np.int32); wts = np.zeros((rows, K), np.float32)
    for r in range(rows):
        sel[r, :k] = rng.choice(n_exp, k, replace=False); sel[r, k:] = n_exp + np.arange(n_sh)
        wts[r] = rng.uniform(0.2, 2.0, K)
    req = cm.create_infer_request()
    out = np.array(req.infer({"x": x, "topk_indices": sel, "routing_weights": wts})[0]).reshape(rows, hidden)
    edir = os.path.join(src, "experts", f"layer_{lid:02d}")
    ref = np.zeros((rows, hidden), np.float32)
    for r in range(rows):
        for e, w in zip(sel[r], wts[r]):
            p = os.path.join(edir, f"expert_{e:03d}.bin" if e < n_exp else f"expert_shared{e - n_exp}.bin")
            wg, wu, wd = _load(p, hidden, inter)
            gg = x[0, r] @ wg.T; uu = x[0, r] @ wu.T
            ref[r] += w * (((gg / (1 + np.exp(-gg))) * uu) @ wd.T)
    d = np.abs(out - ref)
    rms = float(np.sqrt(np.mean(d * d)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))
    print(f"  vs numpy grid reference ({rows} rows, {K} experts each): max_abs={d.max():.3e} rel_rms={rms:.3e} ref|max|={np.abs(ref).max():.3e}")
    for _ in range(3):
        req.infer({"x": x, "topk_indices": sel, "routing_weights": wts})
    n = 20; t0 = time.time()
    for _ in range(n):
        req.infer({"x": x, "topk_indices": sel, "routing_weights": wts})
    print(f"  steady 2-row call: {1e3*(time.time()-t0)/n:.2f} ms")


def main():
    ap = argparse.ArgumentParser(description="Inkling MoE layers -> per-layer fused-MoE OpenVINO IRs")
    ap.add_argument("--src", required=True)
    ap.add_argument("--layers", required=True, help="comma list of MoE layer indices")
    ap.add_argument("--out", default=None, help="destination root (default: --src); writes <out>/moe_ov/layer_NN/")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-device", default="GPU")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--layout", choices=["u4zp", "i4"], default="u4zp")
    ap.add_argument("--pad-experts", type=int, default=4, help="dummy experts appended so the 1%% offload floor keeps every real expert resident")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    man = json.load(open(os.path.join(args.src, "manifest.json")))
    assert man.get("arch") == "inkling", man.get("arch")
    dense = set(man["dense_layers"])
    out = args.out or args.src
    for lid in [int(v) for v in args.layers.split(",")]:
        if lid in dense:
            print(f"layer {lid}: dense, skipped (the dense MLP keeps the per-expert IR)")
            continue
        if args.validate:
            validate(args.src, lid, man, args.validate_device, args.layout, args.pad_experts)
            continue
        dst = os.path.join(out, "moe_ov", f"layer_{lid:02d}")
        xml = os.path.join(dst, "openvino_model.xml")
        if args.skip_existing and os.path.exists(xml):
            print(f"layer {lid}: exists, skipped")
            continue
        t0 = time.time()
        m = layer_model(args.src, lid, man, args.layout, args.pad_experts)
        os.makedirs(dst, exist_ok=True)
        ov.save_model(m, xml, compress_to_fp16=False)
        print(f"layer {lid}: written {xml} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
