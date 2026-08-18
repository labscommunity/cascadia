#!/usr/bin/env python3
"""GLM-5.2 (`glm5`) expert int4 bins -> per-expert OpenVINO int4 IRs.

Converts a glm5 int4_bin expert export (produced by `tools/export_glm5.py`) into
per-expert OpenVINO SwiGLU IRs the Rust runtime (`glm/ov_expert.rs`) runs on
iGPU / NPU / CPU, instead of the Rust int4 mmap GEMV kernel. Each routed and
shared expert becomes a tiny `x[1,1,hidden] -> down(silu(gate·x) * up(x)) ->
y[1,1,hidden]` graph — the SAME activation the Rust glm5 SwiGLU applies
(silu(gate)·up, **no clamp**; the routing weight is applied by the runtime, not
baked in).

Weight dtype (`--dtype`, default **int4**):
  * `int4` — the bins' OWN nibbles and bf16 group scales, packed verbatim into a
    `u4` constant + the standard OV decompression subgraph
    (`u4 -> Convert(f32) -> Subtract(8) -> Multiply(bf16 scale -> f32)`).
    The IR sits on the EXACT quantization grid the Rust mmap kernel reads —
    no NNCF, no float round-trip, ~0.56 B/weight. (The previous version
    dequantized to fp32, built fp16 constants, and let NNCF re-quantize on a
    fresh grid: int4 -> fp32 -> fp16 -> new-int4, cos ~0.9968 per expert vs
    the source grid. See OV-INT4-REQUANT-ISSUE.)
  * `fp16` — the uncompressed fp16-constant graph (2 B/weight, ~3.6x larger).
    Escape hatch for a box with spare RAM or debugging. NOT grid-exact (fp16
    rounds the dequantized values).

`--validate` compiles the fp16 and int4 IRs for one expert and compares both to
a numpy reference of the true glm SwiGLU on the bins' grid. int4 must match to
f32 accumulation-order noise (~1e-6); it prints full-precision error stats and
a scale ratio so gain errors can't hide in rounded output.

The IRs are entirely **opt-in**: the runtime loads them only when
`CASCADIA_GLM5_OV_EXPERTS=1` and `<model>/experts_ov` exists. Absent -> the Rust
int4 mmap path over `experts/layer_NN/expert_*.bin` is used unchanged. This tool
never touches the int4 bins, shells, or manifest; it only writes `experts_ov/`.

int4_bin byte layout (byte-identical to `dsv4/expert_mmap.rs` `MmapExpert` and
`export_glm5.py::_pack_int4_grouped`): per section gate `[inter,hidden]`, up
`[inter,hidden]`, down `[hidden,inter]` — packed nibbles (`out*in/2` u8, low
nibble = even col, high = odd col, value = nibble-8) then bf16-LE per-32-column
group scales. GROUP=32.

Requires `openvino` only (NNCF is no longer used).

Usage:
  python tools/glm5_expert_ov.py --src /media/.../glm5/export
  python tools/glm5_expert_ov.py --src .../export --layers 3,4,5 --dtype int4
  python tools/glm5_expert_ov.py --src .../export --validate
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
from openvino import Model, PartialShape, Type
from openvino import opset15 as ops

GROUP = 32  # int4_bin group size (cols per bf16 scale) — export_glm5.py contract


def _b16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def _deq(buf, off, out, inn):
    """int4_bin section at `buf[off:]` -> (fp32 [out, inn], next offset)."""
    nb = out * inn // 2
    ng = inn // GROUP
    p = np.frombuffer(buf, np.uint8, nb, off).reshape(out, inn // 2)
    s = _b16(np.frombuffer(buf, "<u2", out * ng, off + nb).reshape(out, ng))
    q = np.empty((out, inn), np.int32)
    q[:, 0::2] = p & 0x0F          # even cols = low nibble
    q[:, 1::2] = p >> 4            # odd cols = high nibble
    q -= 8
    w = (q.astype(np.float32).reshape(out, ng, GROUP) * s[:, :, None]).reshape(out, inn)
    return w, off + nb + out * ng * 2


def _load(binp, hidden, inter):
    """gate/up [inter,hidden], down [hidden,inter] as fp32 (dequantized)."""
    buf = open(binp, "rb").read()
    wg, o = _deq(buf, 0, inter, hidden)
    wu, o = _deq(buf, o, inter, hidden)
    wd, o = _deq(buf, o, hidden, inter)
    return wg, wu, wd


def _ref_forward(wg, wu, wd, x):
    """Exact glm SwiGLU in f32 (no clamp): down(silu(gate·x) * up·x)."""
    g = x @ wg.T
    u = x @ wu.T
    silu = g / (1.0 + np.exp(-g))
    return (silu * u) @ wd.T


def _build(wg, wu, wd, hidden):
    """fp16-constant SwiGLU graph: x[1,1,H] -> down(silu(gate(x))*up(x)) [1,1,H].
    No clamp — matches glm `ffn::swiglu`. Weights are fp16 constants converted to
    f32 for the matmul; `--dtype int4` then NNCF-compresses them in place."""
    x = ops.parameter(PartialShape([1, 1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})

    def lin(a, w):  # a @ w.T with a fp16 weight constant
        return ops.matmul(a, ops.convert(ops.constant(w.astype(np.float16)), Type.f32), False, True)

    g = lin(x, wg)                                   # [1,1,inter]
    u = lin(x, wu)
    h = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)  # silu(gate)·up
    y = lin(h, wd)                                   # [1,1,hidden]
    m = Model([y], [x], "glm5_expert")
    m.outputs[0].tensor.set_names({"y"})
    return m


def _raw_const(t, dims, raw):
    """Constant of low-precision Type `t` built from raw LE bytes, verbatim.
    The python Constant ctor has no bytes overload; go through a Tensor whose
    u8 byte view we fill. `shared_memory=False` copies, so the Tensor and the
    source buffer may die."""
    ten = ov.Tensor(t, ov.Shape(dims))
    view = ten.data if isinstance(ten.data, np.ndarray) else np.frombuffer(ten.data, np.uint8)
    view = view.reshape(-1).view(np.uint8)
    src = np.frombuffer(raw, np.uint8)
    assert view.size == src.size, f"{t} tensor {dims}: {view.size}B buffer vs {src.size}B raw"
    view[:] = src
    return ov.op.Constant(ten, shared_memory=False)


def _int4_section(buf, off, out, inn):
    """One int4_bin section -> (f32 weight node on the bins' EXACT grid, next off).

    The packed nibble bytes are a valid OV `u4` buffer AS-IS: the bin stores the
    even column in the low nibble, which is also OV's u4 element order (verified
    by probe — nibble-swapping breaks it). The bf16 scales likewise go in
    verbatim as a `bf16` constant. Dequant runs in f32 (`(q-8) * f32(scale)`),
    the same arithmetic as `expert_mmap.rs::dequant_row_dot`'s grid, so the IR
    and the Rust kernel see bit-identical weights."""
    nb = out * inn // 2
    ng = inn // GROUP
    nibbles = _raw_const(Type.u4, [out, ng, GROUP], buf[off:off + nb])
    scales = _raw_const(Type.bf16, [out, ng, 1], buf[off + nb:off + nb + out * ng * 2])
    w = ops.convert(nibbles, Type.f32)
    w = ops.subtract(w, ops.constant(np.float32(8.0)))
    w = ops.multiply(w, ops.convert(scales, Type.f32))
    w = ops.reshape(w, ops.constant(np.array([out, inn], np.int64)), False)
    return w, off + nb + out * ng * 2


def _build_int4(binp, hidden, inter):
    """SwiGLU graph whose weights are the bins' own nibbles + scales (u4
    constants, zero re-quantization). Same topology as `_build`."""
    buf = open(binp, "rb").read()
    wg, o = _int4_section(buf, 0, inter, hidden)
    wu, o = _int4_section(buf, o, inter, hidden)
    wd, _ = _int4_section(buf, o, hidden, inter)

    x = ops.parameter(PartialShape([1, 1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    g = ops.matmul(x, wg, False, True)
    u = ops.matmul(x, wu, False, True)
    h = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)
    y = ops.matmul(h, wd, False, True)
    m = Model([y], [x], "glm5_expert")
    m.outputs[0].tensor.set_names({"y"})
    return m


def _save_expert(binp, hidden, inter, dtype, edst):
    if dtype == "int4":
        m = _build_int4(binp, hidden, inter)
    else:
        wg, wu, wd = _load(binp, hidden, inter)
        m = _build(wg, wu, wd, hidden)
    os.makedirs(edst, exist_ok=True)
    ov.save_model(m, os.path.join(edst, "openvino_model.xml"), compress_to_fp16=False)


def _validate(binp, hidden, inter):
    """Compile fp16 + int4 IRs for one expert and compare both to the numpy
    reference of the true glm SwiGLU on the bins' own grid. int4 must match to
    f32 accumulation-order noise; anything worse means the exporter left the
    grid. Full-float stats + a scale ratio so a uniform gain error (invisible
    to cosine) and sub-1e-5 deviations (invisible at 5 decimals) surface."""
    wg, wu, wd = _load(binp, hidden, inter)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 1, hidden)).astype(np.float32)) * 0.1
    ref = _ref_forward(wg, wu, wd, x.reshape(1, hidden)).reshape(-1)
    core = ov.Core()

    def run(model):
        # Pin f32 so the numbers measure the WEIGHT GRID, not the plugin's
        # inference-precision default (f16 on some hosts -> ~1e-3 noise).
        cfg = {"SNIPPETS_MODE": "DISABLE", "INFERENCE_PRECISION_HINT": "f32"}
        r = core.compile_model(model, "CPU", cfg).create_infer_request()
        return np.array(r.infer({"x": x})[0]).reshape(-1)

    def stats(name, got):
        d = np.abs(got - ref)
        rel = d / (np.abs(ref) + 1e-6)
        cos = np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9)
        gain = float(np.dot(got, ref) / (np.dot(ref, ref) + 1e-12))
        print(
            f"  {name}: exact={np.array_equal(got, ref)} max_abs={d.max():.3e} "
            f"mean_abs={d.mean():.3e} max_rel={rel.max():.3e} cos={cos!r} gain={gain!r}"
        )
        return float(d.max())

    print(f"validate {binp}  ref |mean|={np.abs(ref).mean():.4f} max={np.abs(ref).max():.4f}")
    stats("fp16", run(_build(wg, wu, wd, hidden)))
    int4_err = stats("int4", run(_build_int4(binp, hidden, inter)))
    # Grid gate: f32 accumulation reorder tops out orders of magnitude below
    # any quantization step. 1e-4 abs on ~0.1-magnitude outputs is a loose
    # ceiling for "same grid" and far below the old exporter's ~1e-3.
    if int4_err > 1e-4:
        raise SystemExit(f"int4 IR left the source grid: max_abs={int4_err:.3e} (expected ~1e-6)")
    print("  int4 IR is on the source quantization grid (residual = f32 accumulation order)")


def main():
    ap = argparse.ArgumentParser(description="glm5 int4 experts -> per-expert OpenVINO int4/fp16 SwiGLU IRs")
    ap.add_argument("--src", required=True, help="glm5 int4_bin export dir (has manifest.json + experts/)")
    ap.add_argument("--layers", default="all",
                    help="comma layer ids to convert, e.g. 3,4,5 (default 'all' MoE layers)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: --src; IRs land at <out>/experts_ov/...)")
    ap.add_argument("--dtype", choices=["int4", "fp16"], default="int4",
                    help="IR weight dtype: int4 (NNCF-compressed, default, RAM-tight) or fp16 (~4x larger)")
    ap.add_argument("--validate", action="store_true",
                    help="compare fp16 & int4 vs a numpy reference for one expert, then exit")
    args = ap.parse_args()

    out = args.out or args.src
    man = json.load(open(os.path.join(args.src, "manifest.json")))
    # glm5 manifest keys (export_glm5.py::build_manifest / loader.rs::GlmManifest):
    #   num_layers   (NOT num_hidden_layers)
    #   dense_layers (NOT first_k_dense_replace) == list(range(first_k_dense_replace))
    hidden = man["hidden_size"]
    inter = man["expert_intermediate"]
    nexp = man["num_experts"]
    n_shared = man.get("n_shared_experts", 1)
    num_layers = man["num_layers"]
    dense = set(man.get("dense_layers", []))
    inter_shared = inter * n_shared  # shared_experts MLP is n_shared merged FFNs

    moe_layers = [li for li in range(num_layers) if li not in dense]

    if args.validate:
        lid = moe_layers[0]
        _validate(os.path.join(args.src, "experts", f"layer_{lid:02d}", "expert_000.bin"), hidden, inter)
        return

    if args.layers.strip().lower() == "all":
        layers = moe_layers
    else:
        layers = []
        for x in args.layers.split(","):
            if x.strip() == "":
                continue
            li = int(x)
            if li < 0 or li >= num_layers:
                print(f"skip layer {li}: out of range [0,{num_layers})", flush=True)
            elif li in dense:
                print(f"skip layer {li}: dense (first_k_dense_replace) — no routed experts", flush=True)
            else:
                layers.append(li)

    print(f"glm5 experts -> OpenVINO IRs  dtype={args.dtype}  layers={len(layers)}  experts={nexp}+shared", flush=True)
    for lid in layers:
        edir = os.path.join(args.src, "experts", f"layer_{lid:02d}")
        odir = os.path.join(out, "experts_ov", f"layer_{lid:02d}")
        for eid in range(nexp):
            _save_expert(os.path.join(edir, f"expert_{eid:03d}.bin"),
                         hidden, inter, args.dtype, os.path.join(odir, f"expert_{eid:03d}"))
        _save_expert(os.path.join(edir, "expert_shared.bin"),
                     hidden, inter_shared, args.dtype, os.path.join(odir, "expert_shared"))
        print(f"layer {lid}: converted {nexp} experts + shared", flush=True)

    print(f"done -> {os.path.join(out, 'experts_ov')}  "
          f"({len(layers)} layers, {nexp} experts each + shared, dtype={args.dtype})", flush=True)


if __name__ == "__main__":
    main()
