#!/usr/bin/env python3
"""GLM-5.2 (`glm5`) counterpart of `tools/int4bin_to_ovir.py`.

Converts a glm5 int4_bin expert export (produced by `tools/export_glm5.py`) into
per-expert OpenVINO SwiGLU IRs that the Rust runtime can run on iGPU / NPU / CPU
via OpenVINO, instead of the Rust int4 mmap GEMV kernel. Each routed and shared
expert is baked into a tiny `x[1,1,hidden] -> down(silu(gate·x) * up(x)) ->
y[1,1,hidden]` graph — the SAME activation the Rust glm5 SwiGLU applies
(silu(gate)·up, **no clamp**; the routing weight is applied to the output by the
runtime, not baked in).

Weight dtype (`--dtype`, default **int4**):
  * `int4` — weights stay **u4-compressed** in the IR (a `Constant(u4)` +
    `Convert(f16) -> Subtract(8) -> Multiply(group scale)` dequant subgraph that
    OpenVINO fuses into its int4-weight GEMV). The IR's resident + on-disk
    footprint then matches the int4 bins (~0.56 B/weight). This is the ONLY
    dtype that fits a RAM-tight shard — the reason the offload exists — and
    matches dsv4's stated intent ("int4 keeps the weights small enough to fit
    RAM, unlike an fp16 re-export").
  * `fp16` — dequantize to fp16 constants (2 B/weight, ~3.6x larger). Simpler
    graph (a plain `Constant`), kept as an escape hatch for a box with spare RAM
    or for debugging; it re-quadruples the residency bar, so avoid it at scale.

The IRs are entirely **opt-in**: the runtime loads them only when
`CASCADIA_GLM5_OV_EXPERTS=1` is set *and* `<model>/experts_ov` exists (see
`glm/ov_expert.rs`, mirroring `CASCADIA_DSV4_OV_EXPERTS`). When the env is unset
or the dir is absent, the default Rust int4 mmap path over
`experts/layer_NN/expert_*.bin` is used unchanged. This tool never touches the
int4 bins, the shells, or the manifest; it only writes a new `experts_ov/` tree.

The int4_bin byte layout read here is byte-identical to what the Rust
`MmapExpert` (`dsv4/expert_mmap.rs`) parses and what `export_glm5.py`
(`_pack_int4_grouped`) writes: for each of gate `[inter,hidden]`, up
`[inter,hidden]`, down `[hidden,inter]`, packed nibbles (`out*in/2` u8, low
nibble = even col, high = odd col, value = nibble-8) followed by bf16-LE
per-32-column group scales. GROUP=32. That low-nibble-first packing matches
OpenVINO's u4 element order, so the packed bytes feed a `Constant(u4)` directly.

NOTE: the `int4` graph (u4 `Constant` + dequant) is the memory-correct path but
is **UNVALIDATED on-device** here (no OpenVINO / checkpoint in CI). If a given
OpenVINO build rejects the u4 `Constant` construction, fall back to `--dtype
fp16` (a plain fp16 `Constant`, the shape the mirrored M2 tool ships) and fix the
u4 API for that version. On-device compile+infer must be validated at scale-out
either way.

Usage:
  python tools/glm5_expert_ov.py --src /media/.../glm5/export
  python tools/glm5_expert_ov.py --src .../export --layers 3,4,5 --dtype int4
  python tools/glm5_expert_ov.py --src .../export --dtype fp16 --out /tmp/glm5_ov
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
from openvino import Model, PartialShape, Shape, Type, op
from openvino import opset15 as ops

GROUP = 32  # int4_bin group size (cols per bf16 scale) — export_glm5.py contract


def _bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _read_sections(path, hidden, inter):
    """One glm5 expert bin -> [(packed_u8, scale_f32[out,ng], out, inn)] for
    gate [inter,hidden], up [inter,hidden], down [hidden,inter], in that order.
    Sizing + order match MmapExpert::open and export_glm5.py::export_expert_bin
    (per section: packed nibbles then bf16 group scales)."""
    b = np.fromfile(path, dtype=np.uint8)
    o = 0

    def take(n):
        nonlocal o
        s = b[o:o + n]
        o += n
        return s

    secs = []
    for out, inn in ((inter, hidden), (inter, hidden), (hidden, inter)):
        ng = inn // GROUP
        packed = np.array(take(out * (inn // 2)), dtype=np.uint8)          # out*in/2 u8
        scale = _bf16_bits_to_f32(
            np.frombuffer(take(out * ng * 2).tobytes(), dtype="<u2")
        ).reshape(out, ng)                                                  # [out, ng] f32
        secs.append((packed, scale, out, inn))
    return secs


def _dequant(packed, scale, out, inn):
    """u4 packed nibbles + bf16 group scales -> fp32 [out, in] (for --dtype fp16)."""
    ng = inn // GROUP
    p = packed.reshape(out, inn // 2)
    q = np.empty((out, inn), dtype=np.int32)
    q[:, 0::2] = (p & 0x0F).astype(np.int32)   # even cols = low nibble
    q[:, 1::2] = (p >> 4).astype(np.int32)     # odd cols = high nibble
    q -= 8
    w = q.astype(np.float32).reshape(out, ng, GROUP) * scale[:, :, None]
    return w.reshape(out, inn)


def _weight_fp16(packed, scale, out, inn):
    """Dequantized fp16 weight constant [out, inn] (2 B/weight)."""
    return ops.constant(_dequant(packed, scale, out, inn).astype(np.float16))


def _weight_int4(packed, scale, out, inn):
    """u4-compressed weight -> dequantized f16 node [out, inn] (0.5 B/weight in
    the IR). Constant(u4) holds the raw nibbles; OpenVINO fuses the
    Convert->Subtract->Multiply into its int4-weight GEMV. The int4_bin low-
    nibble-first packing matches OV's u4 element order, so `packed` feeds the u4
    Constant unchanged."""
    ng = inn // GROUP
    t = ov.Tensor(Type.u4, Shape([out, ng, GROUP]))
    # `t.data` is a uint8 view of the packed buffer (numel/2 bytes for u4);
    # copy the int4_bin's packed nibbles straight in.
    dst = np.asarray(t.data).reshape(-1)
    src = packed.reshape(-1)
    assert dst.size == src.size, f"u4 buffer {dst.size} != packed {src.size}"
    dst[:] = src
    w = ops.convert(op.Constant(t), Type.f16)
    w = ops.subtract(w, ops.constant(np.float16(8.0)))                 # zero point
    w = ops.multiply(w, ops.constant(scale[:, :, None].astype(np.float16)))  # per-group scale
    w = ops.reshape(w, ops.constant(np.array([out, inn], dtype=np.int64)), False)
    return w  # f16 [out, inn]


def _expert_model(secs, hidden, inter, dtype):
    """SwiGLU expert IR: x[1,1,H] -> down(silu(gate(x))*up(x)) [1,1,H].
    No clamp — matches the glm5 Rust SwiGLU (ffn::swiglu_mmap / MmapExpert)."""
    mk = _weight_int4 if dtype == "int4" else _weight_fp16
    (gp, gs, go, gi), (up, us, uo, ui), (dp, ds, do_, di) = secs
    x = ops.parameter(PartialShape([1, 1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    xb = ops.convert(x, Type.f16)
    gW = mk(gp, gs, go, gi)   # [inter, hidden]
    uW = mk(up, us, uo, ui)   # [inter, hidden]
    dW = mk(dp, ds, do_, di)  # [hidden, inter]
    g = ops.matmul(xb, gW, False, True)          # [1,1,inter]
    u = ops.matmul(xb, uW, False, True)
    act = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)
    y = ops.convert(ops.matmul(act, dW, False, True), Type.f32)  # [1,1,hidden]
    y.get_output_tensor(0).set_names({"y"})
    return Model([y], [x])


def _save_expert(path, hidden, inter, dtype, edst):
    os.makedirs(edst, exist_ok=True)
    secs = _read_sections(path, hidden, inter)
    ov.save_model(_expert_model(secs, hidden, inter, dtype),
                  os.path.join(edst, "openvino_model.xml"), compress_to_fp16=False)


def main():
    ap = argparse.ArgumentParser(description="glm5 int4 experts -> per-expert OpenVINO SwiGLU IRs")
    ap.add_argument("--src", required=True, help="glm5 int4_bin export dir (has manifest.json + experts/)")
    ap.add_argument("--layers", default="all",
                    help="comma layer ids to convert, e.g. 3,4,5 (default 'all' MoE layers)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: --src; IRs land at <out>/experts_ov/...)")
    ap.add_argument("--dtype", choices=["int4", "fp16"], default="int4",
                    help="IR weight dtype: int4 (u4-compressed, default, RAM-tight) or fp16 (~4x larger)")
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
