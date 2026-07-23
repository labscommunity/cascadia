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
  * `int4` — **NNCF `INT4_SYM` weight-compression** (`group_size=32`, `ratio=1.0`)
    on the fp16-constant graph, so the IR stores int4-compressed weights
    (~0.56 B/weight, matching the int4 bins). This is the same proven path the
    dsv4 fleet tool (`dsv4/dsv4_expert_ov.py`) uses and validated on-device; it
    avoids hand-packing a `u4` constant. It is the ONLY dtype that fits a
    RAM-tight shard, which is the whole reason the offload exists.
  * `fp16` — the uncompressed fp16-constant graph (2 B/weight, ~3.6x larger).
    Escape hatch for a box with spare RAM or debugging.

`--validate` compiles the fp16 and int4 IRs for one expert and compares both to a
numpy reference of the true glm SwiGLU (max/mean abs error + cosine), so the
quantization error is visible before a fleet run.

The IRs are entirely **opt-in**: the runtime loads them only when
`CASCADIA_GLM5_OV_EXPERTS=1` and `<model>/experts_ov` exists. Absent -> the Rust
int4 mmap path over `experts/layer_NN/expert_*.bin` is used unchanged. This tool
never touches the int4 bins, shells, or manifest; it only writes `experts_ov/`.

int4_bin byte layout (byte-identical to `dsv4/expert_mmap.rs` `MmapExpert` and
`export_glm5.py::_pack_int4_grouped`): per section gate `[inter,hidden]`, up
`[inter,hidden]`, down `[hidden,inter]` — packed nibbles (`out*in/2` u8, low
nibble = even col, high = odd col, value = nibble-8) then bf16-LE per-32-column
group scales. GROUP=32.

Requires `openvino` and `nncf` (the fleet export environment has both).

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


def _to_int4(m):
    """NNCF symmetric int4 weight-compression (group_size=32), matching the
    int4_bin grouping. Same call the dsv4 fleet tool uses."""
    import logging

    import nncf
    nncf.set_log_level(logging.CRITICAL)
    return nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT4_SYM, group_size=GROUP, ratio=1.0)


def _save_expert(binp, hidden, inter, dtype, edst):
    wg, wu, wd = _load(binp, hidden, inter)
    m = _build(wg, wu, wd, hidden)
    if dtype == "int4":
        m = _to_int4(m)
    os.makedirs(edst, exist_ok=True)
    ov.save_model(m, os.path.join(edst, "openvino_model.xml"), compress_to_fp16=False)


def _validate(binp, hidden, inter):
    """Compile fp16 + int4 IRs for one expert and compare both to the numpy
    reference of the true glm SwiGLU."""
    wg, wu, wd = _load(binp, hidden, inter)
    m = _build(wg, wu, wd, hidden)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 1, hidden)).astype(np.float32)) * 0.1
    ref = _ref_forward(wg, wu, wd, x.reshape(1, hidden)).reshape(-1)
    core = ov.Core()

    def run(model):
        r = core.compile_model(model, "CPU", {"SNIPPETS_MODE": "DISABLE"}).create_infer_request()
        return np.array(r.infer({"x": x})[0]).reshape(-1)

    def stats(name, got):
        d = np.abs(got - ref)
        rel = d / (np.abs(ref) + 1e-6)
        cos = np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9)
        print(f"  {name}: max_abs={d.max():.5f} mean_abs={d.mean():.5f} max_rel={rel.max():.3f} cos={cos:.5f}")

    print(f"validate {binp}  ref |mean|={np.abs(ref).mean():.4f} max={np.abs(ref).max():.4f}")
    stats("fp16", run(m))
    stats("int4", run(_to_int4(_build(wg, wu, wd, hidden))))


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
