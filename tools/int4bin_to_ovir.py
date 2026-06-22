#!/usr/bin/env python3
"""Build a per-node M2 model dir whose experts are fp16 ov_ir (iGPU-runnable).

The full MiniMax-M2 export uses `int4_bin` experts — a flat int4 layout fed to
the cascadia-int4-gemm AVX-512 CPU kernel. That kernel can't run on a NUC
(no AVX-512), and the iGPU needs OpenVINO graphs anyway. This tool converts a
chosen subset of layers' int4_bin experts into fp16 `ov_ir` expert IRs
(dequantizing the int4 nibbles, no FP8 source required) and assembles a node
model dir: ov_ir manifest + the (copied) shell IRs + converted experts + the
head (for the last pipeline rank).

Pair with `tools/split_m2_shells.py` (run on the output dir) to produce the
shell_core/router split the iGPU path needs, then point a `cascadia worker`
rank at this dir with `CASCADIA_M2_LAYER_RANGE=<start>:<end>` matching the
layers converted here.

Usage:
  python tools/int4bin_to_ovir.py --src /media/.../m2/export \
      --layers 60,61 --out /tmp/m2_nuc_charlie --head
"""
import argparse
import json
import os
import shutil

import numpy as np
import openvino as ov
from openvino import Model, Type, PartialShape
from openvino import opset15 as ops

GROUP = 32  # int4_bin group size (cols per bf16 scale)


def _bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _dequant(packed: bytes, scale: bytes, out: int, inn: int) -> np.ndarray:
    """int4_bin packed (out*in/2 u8) + bf16 group scales -> fp32 [out, in]."""
    ng = inn // GROUP
    p = np.frombuffer(packed, dtype=np.uint8).reshape(out, inn // 2)
    s = _bf16_bits_to_f32(np.frombuffer(scale, dtype="<u2").reshape(out, ng))
    q = np.empty((out, inn), dtype=np.int32)
    q[:, 0::2] = (p & 0x0F).astype(np.int32)   # even cols = low nibble
    q[:, 1::2] = (p >> 4).astype(np.int32)     # odd cols = high nibble
    q -= 8
    w = q.astype(np.float32).reshape(out, ng, GROUP) * s[:, :, None]
    return w.reshape(out, inn)


def _read_expert_bin(path, hidden, inter):
    b = np.fromfile(path, dtype=np.uint8)
    gp = inter * (hidden // 2)
    gs = inter * (hidden // GROUP) * 2
    dp = hidden * (inter // 2)
    ds = hidden * (inter // GROUP) * 2
    o = 0
    def take(n):
        nonlocal o
        s = b[o:o + n].tobytes(); o += n; return s
    gate = _dequant(take(gp), take(gs), inter, hidden)   # [inter, hidden]
    up   = _dequant(take(gp), take(gs), inter, hidden)   # [inter, hidden]
    down = _dequant(take(dp), take(ds), hidden, inter)   # [hidden, inter]
    return gate, up, down


def _expert_model(gate, up, down, hidden, inter):
    """fp16 SwiGLU expert: x[1,1,H] -> down(silu(gate(x))*up(x)) [1,1,H]."""
    x = ops.parameter(PartialShape([1, 1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    xb = ops.convert(x, Type.f16)
    gW = ops.constant(gate.astype(np.float16))   # [inter, hidden]
    uW = ops.constant(up.astype(np.float16))
    dW = ops.constant(down.astype(np.float16))   # [hidden, inter]
    g = ops.matmul(xb, gW, False, True)          # [1,1,inter]
    u = ops.matmul(xb, uW, False, True)
    act = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)
    y = ops.convert(ops.matmul(act, dW, False, True), Type.f32)  # [1,1,hidden]
    y.get_output_tensor(0).set_names({"y"})
    return Model([y], [x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="full int4_bin M2 export dir")
    ap.add_argument("--layers", required=True, help="comma layer ids to convert, e.g. 60,61")
    ap.add_argument("--out", required=True, help="output node model dir")
    ap.add_argument("--head", action="store_true", help="copy head/ (last pipeline rank)")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.src, "manifest.json")))
    hidden = man["hidden_size"]
    inter = man["expert_intermediate"]
    nexp = man["num_experts"]
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    os.makedirs(args.out, exist_ok=True)

    # ov_ir manifest (same dims/arch; experts are now OV graphs)
    man["experts_format"] = "ov_ir"
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"))

    for lid in layers:
        # shell IR (already OV; split_m2_shells.py runs on this dir afterwards)
        sdst = os.path.join(args.out, "shells", f"layer_{lid:02d}")
        os.makedirs(sdst, exist_ok=True)
        for ext in ("xml", "bin"):
            src = os.path.join(args.src, "shells", f"layer_{lid:02d}", f"openvino_model.{ext}")
            if os.path.exists(src):
                shutil.copy(src, sdst)
        # experts: int4_bin .bin -> fp16 ov_ir IR
        for eid in range(nexp):
            binp = os.path.join(args.src, "experts", f"layer_{lid:02d}", f"expert_{eid:03d}.bin")
            gate, up, down = _read_expert_bin(binp, hidden, inter)
            edst = os.path.join(args.out, "experts", f"layer_{lid:02d}", f"expert_{eid:03d}")
            os.makedirs(edst, exist_ok=True)
            ov.save_model(_expert_model(gate, up, down, hidden, inter),
                          os.path.join(edst, "openvino_model.xml"), compress_to_fp16=False)
        print(f"layer {lid}: converted {nexp} experts + copied shell", flush=True)

    if args.head:
        hsrc = os.path.join(args.src, "head")
        if os.path.isdir(hsrc):
            shutil.copytree(hsrc, os.path.join(args.out, "head"), dirs_exist_ok=True)
            print("copied head/", flush=True)
    print(f"done -> {args.out}  (now run split_m2_shells.py on it)", flush=True)


if __name__ == "__main__":
    main()
