#!/usr/bin/env python3
"""GLM-5.2 (`glm5`) counterpart of `tools/int4bin_to_ovir.py`.

Converts a glm5 int4_bin expert export (produced by `tools/export_glm5.py`) into
per-expert OpenVINO **fp16 SwiGLU IRs** that the Rust runtime can run on iGPU /
NPU / CPU via OpenVINO, instead of the Rust int4 mmap GEMV kernel. Each routed
and shared expert's int4 nibbles are dequantized (no FP8 source needed) and baked
into a tiny `x[1,1,hidden] -> down(silu(gate·x) * up(x)) -> y[1,1,hidden]` graph —
the SAME activation the Rust glm5 SwiGLU applies (silu(gate)·up, **no clamp**, the
routing weight is applied to the output by the runtime, not baked in).

The IRs are entirely **opt-in**: the runtime loads them only when
`CASCADIA_GLM5_OV_EXPERTS=1` is set *and* `<model>/experts_ov` exists (mirroring
`CASCADIA_DSV4_OV_EXPERTS` for dsv4 in `dsv4/ov_expert.rs`). When the env is unset
or the `experts_ov` dir is absent, the default engine path — the Rust int4 mmap
kernel over `experts/layer_NN/expert_*.bin` — is used unchanged. This tool never
touches the int4 bins, the shells, or the manifest; it only writes a new
`experts_ov/` tree next to them.

The int4_bin byte layout read here is byte-identical to what the Rust
`MmapExpert` (`dsv4/expert_mmap.rs`) parses and what `export_glm5.py`
(`_pack_int4_grouped`) writes: for each of gate `[inter,hidden]`, up
`[inter,hidden]`, down `[hidden,inter]`, packed nibbles (`out*in/2` u8, low nibble
= even col, high = odd col, value = nibble-8) followed by bf16-LE per-32-column
group scales. GROUP=32.

Usage:
  python tools/glm5_expert_ov.py --src /media/.../glm5/export
  python tools/glm5_expert_ov.py --src /media/.../glm5/export --layers 3,4,5
  python tools/glm5_expert_ov.py --src .../export --layers all --out /tmp/glm5_ov
"""
import argparse
import json
import os

import numpy as np
import openvino as ov
from openvino import Model, Type, PartialShape
from openvino import opset15 as ops

GROUP = 32  # int4_bin group size (cols per bf16 scale) — export_glm5.py contract


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
    """Read one glm5 expert bin: gate/up [inter,hidden], down [hidden,inter].
    Order + sizing match MmapExpert::open and export_glm5.py::export_expert_bin
    (per section: packed nibbles then bf16 scales; sections gate, up, down)."""
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
    """fp16 SwiGLU expert: x[1,1,H] -> down(silu(gate(x))*up(x)) [1,1,H].
    No clamp — matches the glm5 Rust SwiGLU (ffn::swiglu_mmap / MmapExpert)."""
    x = ops.parameter(PartialShape([1, 1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    xb = ops.convert(x, Type.f16)
    gW = ops.constant(gate.astype(np.float16))   # [inter, hidden]
    uW = ops.constant(up.astype(np.float16))
    dW = ops.constant(down.astype(np.float16))    # [hidden, inter]
    g = ops.matmul(xb, gW, False, True)          # [1,1,inter]
    u = ops.matmul(xb, uW, False, True)
    act = ops.multiply(ops.multiply(g, ops.sigmoid(g)), u)
    y = ops.convert(ops.matmul(act, dW, False, True), Type.f32)  # [1,1,hidden]
    y.get_output_tensor(0).set_names({"y"})
    return Model([y], [x])


def _save_expert(gate, up, down, hidden, inter, edst):
    os.makedirs(edst, exist_ok=True)
    ov.save_model(_expert_model(gate, up, down, hidden, inter),
                  os.path.join(edst, "openvino_model.xml"), compress_to_fp16=False)


def main():
    ap = argparse.ArgumentParser(description="glm5 int4 experts -> per-expert fp16 OpenVINO SwiGLU IRs")
    ap.add_argument("--src", required=True, help="glm5 int4_bin export dir (has manifest.json + experts/)")
    ap.add_argument("--layers", default="all",
                    help="comma layer ids to convert, e.g. 3,4,5 (default 'all' MoE layers)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: --src; IRs land at <out>/experts_ov/...)")
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

    for lid in layers:
        edir = os.path.join(args.src, "experts", f"layer_{lid:02d}")
        odir = os.path.join(out, "experts_ov", f"layer_{lid:02d}")
        # routed experts: int4_bin .bin -> fp16 ov_ir IR
        for eid in range(nexp):
            binp = os.path.join(edir, f"expert_{eid:03d}.bin")
            gate, up, down = _read_expert_bin(binp, hidden, inter)
            _save_expert(gate, up, down, hidden, inter,
                         os.path.join(odir, f"expert_{eid:03d}"))
        # shared expert (inter = expert_intermediate * n_shared_experts)
        sgate, sup, sdown = _read_expert_bin(os.path.join(edir, "expert_shared.bin"),
                                             hidden, inter_shared)
        _save_expert(sgate, sup, sdown, hidden, inter_shared,
                     os.path.join(odir, "expert_shared"))
        print(f"layer {lid}: converted {nexp} experts + shared", flush=True)

    print(f"done -> {os.path.join(out, 'experts_ov')}  "
          f"({len(layers)} layers, {nexp} experts each + shared)", flush=True)


if __name__ == "__main__":
    main()
