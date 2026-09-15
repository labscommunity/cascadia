#!/usr/bin/env python3
"""Inkling int4 expert bins -> per-expert OpenVINO int4 IRs (iGPU / NPU / CPU).

The Inkling counterpart of `tools/glm5_expert_ov.py`, reusing its graph
builders: the int4_bin byte layout is shared (`dsv4/expert_mmap.rs`
`MmapExpert`, GROUP=32, gate/up `[inter,hidden]` then down `[hidden,inter]`,
packed nibbles + bf16-LE group scales). Every routed expert, both shared
experts and the two dense layers' MLP become a `x[1,1,hidden] ->
down(silu(gate·x) * up·x) -> y[1,1,hidden]` graph whose weights are the bins'
OWN nibbles and scales (u4 constants + the standard decompression subgraph):
the IR sits on the exact quantisation grid the Rust kernel reads.

Rounding: the Rust kernel (`MmapExpert::swiglu_from`) rounds the gate and up
GEMV outputs to bf16 (round-to-nearest-even) before the SiLU product and
rounds the down output to bf16. The IR keeps gate/up in f32 and the runtime
rounds the output: OpenVINO's f32 -> bf16 `Convert` TRUNCATES on both the CPU
and the GPU plugin (measured on the B390: a 0.37% gain loss per expert with
the rounding in the graph, 6e-7 relative rms without), so reproducing the two
inner rounding points in-graph is not possible and `--rounding rust` exists
only to demonstrate that. What remains between the two paths is therefore the
Rust kernel's inner bf16 rounding (unbiased, <= half a bf16 ULP on gate/up)
plus f32 accumulation order; 99.9% of bf16-rounded outputs are identical.

Layout written (opt-in for the runtime: `CASCADIA_INKLING_OV_EXPERTS=1`):
  <out>/experts_ov/layer_NN/expert_EEE/openvino_model.{xml,bin}
  <out>/experts_ov/layer_NN/expert_sharedS/...        (S = 0, 1)
  <out>/experts_ov/layer_NN/dense/...                  (dense layers only)

`--validate` compiles one MoE expert's int4 IR on CPU and compares it with a
numpy reference of the same SwiGLU on the bins' grid (with the same bf16
rounding points): the residual must be f32 accumulation-order noise.
This tool never touches the bins, shells or manifest.

Usage:
  python tools/inkling_expert_ov.py --src /data/inkling-int4 --layers 0,1,2,3
  python tools/inkling_expert_ov.py --src /data/inkling-int4 --layers 2 --validate
"""
import argparse
import json
import os
import sys

import numpy as np
import openvino as ov
from openvino import Model, PartialShape, Type
from openvino import opset15 as ops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm5_expert_ov import _int4_section, _load, _raw_const  # noqa: E402

GROUP = 32


def _bf16_round(x):
    """Round-to-nearest-even f32 -> bf16 -> f32, the Rust `to_bf16`."""
    b = x.astype(np.float32).view(np.uint32)
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return b.view(np.float32)


def _ref_forward(wg, wu, wd, x, rounding):
    g = x @ wg.T
    u = x @ wu.T
    if rounding == "rust":
        g, u = _bf16_round(g), _bf16_round(u)
    silu = g / (1.0 + np.exp(-g))
    return (silu * u) @ wd.T


def build_int4(binp, hidden, inter, rounding):
    """SwiGLU graph on the bins' own nibbles + scales."""
    buf = open(binp, "rb").read()
    wg, o = _int4_section(buf, 0, inter, hidden)
    wu, o = _int4_section(buf, o, inter, hidden)
    wd, _ = _int4_section(buf, o, hidden, inter)

    x = ops.parameter(PartialShape([1, 1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    g = ops.matmul(x, wg, False, True)
    u = ops.matmul(x, wu, False, True)
    if rounding == "rust":
        g = ops.convert(ops.convert(g, Type.bf16), Type.f32)
        u = ops.convert(ops.convert(u, Type.bf16), Type.f32)
    h = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)
    y = ops.matmul(h, wd, False, True)
    m = Model([y], [x], "inkling_expert")
    m.outputs[0].tensor.set_names({"y"})
    return m


def save_expert(binp, hidden, inter, rounding, edst):
    m = build_int4(binp, hidden, inter, rounding)
    os.makedirs(edst, exist_ok=True)
    ov.save_model(m, os.path.join(edst, "openvino_model.xml"), compress_to_fp16=False)


def plugin_config(device, precision, dq_group):
    """Compile properties that decide the numerics: `INFERENCE_PRECISION_HINT`
    (f32 = exact grid arithmetic; f16 = the iGPU's native fast path) and
    `DYNAMIC_QUANTIZATION_GROUP_SIZE` (0 = no int8 activation quantisation;
    the plugins' default quantises activations of int4-weight matmuls, which
    costs ~1e-2 relative error). The runtime (`inkling/ov_expert.rs`) sets the
    same two from `CASCADIA_INKLING_OV_PRECISION` / `_OV_DQ_GROUP`."""
    cfg = {"INFERENCE_PRECISION_HINT": precision, "DYNAMIC_QUANTIZATION_GROUP_SIZE": str(dq_group)}
    if device == "CPU":
        cfg["SNIPPETS_MODE"] = "DISABLE"
    return cfg


def validate(binp, hidden, inter, rounding, device="CPU", precision="f32", dq_group=0, strict=True):
    wg, wu, wd = _load(binp, hidden, inter)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 1, hidden)).astype(np.float32)) * 0.1
    ref = _ref_forward(wg, wu, wd, x.reshape(1, hidden), rounding).reshape(-1)
    core = ov.Core()
    cfg = plugin_config(device, precision, dq_group)
    r = core.compile_model(build_int4(binp, hidden, inter, rounding), device, cfg).create_infer_request()
    got = np.array(r.infer({"x": x})[0]).reshape(-1)
    d = np.abs(got - ref)
    cos = np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-9)
    gain = float(np.dot(got, ref) / (np.dot(ref, ref) + 1e-12))
    # bf16-level agreement of the OUTPUT the runtime will round.
    same_bf16 = float(np.mean(_bf16_round(got) == _bf16_round(ref)))
    rms = float(np.sqrt(np.mean(d * d)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))
    print(
        f"validate {binp} on {device} ({precision}, dq_group={dq_group}): "
        f"ref |mean|={np.abs(ref).mean():.4f} max={np.abs(ref).max():.4f}\n"
        f"  int4 IR ({rounding} rounding): max_abs={d.max():.3e} mean_abs={d.mean():.3e} rel_rms={rms:.3e} "
        f"cos={cos!r} gain={gain!r} bf16-identical outputs={same_bf16:.4%}"
    )
    if not strict:
        return
    # On the grid the only f32 residual is accumulation order (~1e-6
    # relative); the in-graph rounding variant fails this by design.
    if rms > 1e-4:
        raise SystemExit(f"int4 IR left the source grid: rel_rms={rms:.3e} (expected ~1e-6)")
    print("  int4 IR is on the source quantisation grid")


def main():
    ap = argparse.ArgumentParser(description="Inkling int4 experts -> per-expert OpenVINO int4 SwiGLU IRs")
    ap.add_argument("--src", required=True, help="Inkling int4_bin export dir (manifest.json + experts/)")
    ap.add_argument("--layers", default="all", help="comma list of layer indices, or 'all'")
    ap.add_argument("--out", default=None, help="destination root (default: --src); writes <out>/experts_ov/")
    ap.add_argument("--rounding", choices=["none", "rust"], default="none",
                    help="'rust' inserts bf16 Converts after gate/up; OV truncates there, so it is wrong "
                         "by construction (kept for the demonstration). Default 'none'.")
    ap.add_argument("--validate", action="store_true", help="compile one expert on CPU and compare with numpy")
    ap.add_argument("--validate-device", default="CPU")
    ap.add_argument("--validate-precision", default="f32", help="INFERENCE_PRECISION_HINT for --validate")
    ap.add_argument("--validate-dq-group", type=int, default=0, help="DYNAMIC_QUANTIZATION_GROUP_SIZE for --validate")
    ap.add_argument("--validate-report-only", action="store_true", help="print the numerics without the grid gate")
    ap.add_argument("--skip-existing", action="store_true", help="skip experts whose IR xml already exists")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.src, "manifest.json")))
    if man.get("arch") != "inkling":
        raise SystemExit(f"manifest arch {man.get('arch')!r} is not inkling")
    hidden = man["hidden_size"]
    inter = man["moe_intermediate"]
    dense_inter = man["dense_intermediate"]
    n_exp = man["num_experts"]
    n_shared = man["n_shared_experts"]
    dense_layers = set(man["dense_layers"])
    n_layers = man["num_layers"]
    layers = list(range(n_layers)) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    out = args.out or args.src

    if args.validate:
        lid = next((l for l in layers if l not in dense_layers), None)
        if lid is None:
            raise SystemExit("--validate needs a MoE layer in --layers")
        validate(os.path.join(args.src, "experts", f"layer_{lid:02d}", "expert_000.bin"),
                 hidden, inter, args.rounding, args.validate_device, args.validate_precision,
                 args.validate_dq_group, strict=not args.validate_report_only)
        return

    for lid in layers:
        edir = os.path.join(args.src, "experts", f"layer_{lid:02d}")
        odir = os.path.join(out, "experts_ov", f"layer_{lid:02d}")
        if lid in dense_layers:
            dst = os.path.join(odir, "dense")
            if not (args.skip_existing and os.path.exists(os.path.join(dst, "openvino_model.xml"))):
                save_expert(os.path.join(edir, "dense.bin"), hidden, dense_inter, args.rounding, dst)
            print(f"layer {lid:02d}: dense MLP written")
            continue
        n_done = 0
        for eid in range(n_exp):
            dst = os.path.join(odir, f"expert_{eid:03d}")
            if args.skip_existing and os.path.exists(os.path.join(dst, "openvino_model.xml")):
                continue
            save_expert(os.path.join(edir, f"expert_{eid:03d}.bin"), hidden, inter, args.rounding, dst)
            n_done += 1
        for s in range(n_shared):
            dst = os.path.join(odir, f"expert_shared{s}")
            if args.skip_existing and os.path.exists(os.path.join(dst, "openvino_model.xml")):
                continue
            save_expert(os.path.join(edir, f"expert_shared{s}.bin"), hidden, inter, args.rounding, dst)
            n_done += 1
        print(f"layer {lid:02d}: {n_done} expert IRs written ({n_exp} routed + {n_shared} shared)")


if __name__ == "__main__":
    main()
