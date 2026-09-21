#!/bin/bash
# Runs this box's Inkling rank (started by the cascadia-inkling systemd unit).
set -u
PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$PREFIX/rank.env"; set +a
# ---- Role swap: two boxes exchange their PIPELINE roles (rank, layers, API); everything that belongs to the BOX stays
# where it is: the beacon's rank and the host names it keeps (<fleet>-rank-N), the updater's source, and on the entry
# box the operator tunnel, the release poller and the file server (their keys were made on that box). Why: the entry
# box lost power three times under rank 0's load (API, scheduler, draft model, the highest package power of the 25 W
# boxes); a box without that platform limit takes the role, the entry box keeps a middle rank and relays :8000.
# ROLE_SWAP="A B" (box ranks) or empty; empty in the repository: autolab/bench/build_role_swap.py writes the run.sh
# of a swap release. A box of the pair changes role ONLY if it holds the other role's data,
# verified (role-swap/ready-<role>, written by the sync block of the overrides); every other box only learns where
# its next rank lives now. BOX_RANK is this box's installed rank, RANK the role it plays.
ROLE_SWAP=""
BOX_RANK="$RANK"
role_swap() {
  local a b per base r nr nb port
  set -- $ROLE_SWAP; a="${1:-}"; b="${2:-}"
  case "$a$b" in ''|*[!0-9]*) return 0 ;; esac
  [ "$a" != "$b" ] && [ "$a" -lt "$TOTAL" ] && [ "$b" -lt "$TOTAL" ] || return 0
  # the last rank also holds the output head: not part of this mechanism
  [ "$a" -lt "$((TOTAL-1))" ] && [ "$b" -lt "$((TOTAL-1))" ] || return 0
  per=$((LAYER_END - LAYER_START)); [ "$per" -gt 0 ] || return 0
  r="$BOX_RANK"
  if [ "$BOX_RANK" = "$a" ] || [ "$BOX_RANK" = "$b" ]; then
    [ "$BOX_RANK" = "$a" ] && r="$b" || r="$a"
    if [ ! -e "$PREFIX/role-swap/ready-$r" ]; then
      echo "role swap: no verified data for role $r on this box (box rank $BOX_RANK): keeping the installed role"
      r="$BOX_RANK"
    fi
  fi
  if [ "$r" != "$BOX_RANK" ]; then
    RANK="$r"; LAYER_START=$((r * per)); LAYER_END=$((r * per + per))
    echo "role swap: box rank $BOX_RANK plays pipeline rank $RANK, layers [$LAYER_START,$LAYER_END)"
  fi
  # where the next pipeline rank lives (rank.env: NEXT=<fleet>-rank-<n>:<port>, empty on the last rank)
  if [ -n "${NEXT:-}" ]; then
    nr=$((RANK + 1)); nb="$nr"
    [ "$nr" = "$a" ] && nb="$b"; [ "$nr" = "$b" ] && nb="$a"
    port="${NEXT##*:}"; base=$((port - BOX_RANK - 1))
    case "$NEXT" in "$FLEET-rank-"*) NEXT="$FLEET-rank-$nb:$((base + nr))" ;; esac
  fi
}
role_swap
export BOX_RANK ROLE_SWAP
# fleet-wide settings pushed from rank 0 by the updater win over this box's own (same on every box by construction)
if [ -f "$PREFIX/fleet-overrides.env" ]; then set -a; source "$PREFIX/fleet-overrides.env"; set +a; fi
# Fused MoE IRs for layers that have none yet (CASCADIA_FUSE_LAYERS="36,37,..." from the overrides). The installer
# generated three per box and left the generator on the install medium, so it travels in this script. One layer at
# a time into a scratch folder, moved into place only when complete: a box that loses power mid-way never finds
# half an IR. Whatever fails here, the worker still starts (a layer without an IR simply stays on the CPU path).
gen_fused_irs() {
  # $1 = layers, $2 = int4 group of the weights written (32 = the bins verbatim, into moe_ov/). A second call with
  # CASCADIA_FUSE_DECODE_LAYERS and group 64|128 writes those layers ALSO re-quantised to the wider group, into their
  # own folder (moe_ov_g64): the GPU plugin's MoE decode kernels refuse group 32 on Xe2 and newer. The engine picks
  # that folder per layer with CASCADIA_INKLING_OV_MOE_DECODE_DIR=moe_ov_g64; the group-32 layers stay where they are.
  local want="${1:-}" py l nn dst tmp free_gb shift have pair grp sub mark
  [ -n "$want" ] || return 0
  grp="${2:-32}"; case "$grp" in 64|128) sub="moe_ov_g$grp" ;; *) grp=32; sub="moe_ov" ;; esac
  py=$(command -v python3) || return 0
  [ -d "$PREFIX/pylib" ] || { echo "fused IRs: no $PREFIX/pylib on this box"; return 0; }
  mkdir -p "$PREFIX/tools" "$PREFIX/logs"
  tmp="$PREFIX/model/.fuse-tmp"
  for l in ${want//,/ }; do
    case "$l" in ''|*[!0-9]*) continue ;; esac
    nn=$(printf %02d "$l"); dst="$PREFIX/model/$sub/layer_$nn"; mark="$PREFIX/logs/fused-ir.failed-$nn"
    [ "$sub" = "moe_ov" ] || mark="$PREFIX/logs/fused-ir.failed-$sub-$nn"
    # CASCADIA_FUSE_UP_SHIFT="8:4 40:4": those layers are (re)generated with their up-projection scales x 2^-N (an expert
    # whose own output passes f16's range on the device); a layer counts as present only with the asked exponent.
    shift=0; for pair in ${CASCADIA_FUSE_UP_SHIFT:-}; do [ "${pair%%:*}" = "$l" ] && shift="${pair##*:}"; done
    case "$shift" in ''|*[!0-9]*) shift=0 ;; esac
    have=$(sed -n 's/.*"up_scale_exponent"[^0-9]*\([0-9][0-9]*\).*/\1/p' "$dst/cascadia_moe.json" 2>/dev/null | head -1)
    [ -s "$dst/openvino_model.xml" ] && [ -s "$dst/openvino_model.bin" ] && [ "${have:-0}" = "$shift" ] && continue
    # a layer that failed once is not retried at every start (each try can take minutes): CASCADIA_FUSE_RETRY=1 asks again
    [ -e "$mark" ] && [ "${CASCADIA_FUSE_RETRY:-0}" != 1 ] && continue
    free_gb=$(df -Pk "$PREFIX/model" | awk 'NR==2 {print int($4/1048576)}')
    [ "${free_gb:-0}" -ge 20 ] || { echo "fused IRs: only ${free_gb} GB free, layer $l not generated"; return 0; }
    write_generator "$PREFIX/tools/inkling_moe_layer_ov.py"
    rm -rf "$tmp"; mkdir -p "$tmp"
    echo "fused IRs: generating layer $l (about a minute, 8.4 GB; re-quantising to a wider group takes about six)"
    if PYTHONPATH="$PREFIX/pylib" timeout "$([ "$grp" = 32 ] && echo 900 || echo 2400)" "$py" "$PREFIX/tools/inkling_moe_layer_ov.py" --src "$PREFIX/model" --out "$tmp" \
         --layers "$l" --layout u4zp --pad-experts 4 --up-scale-exponent "$shift" --group "$grp" --moe-dir "$sub" \
         >> "$PREFIX/logs/fused-ir.log" 2>&1 \
       && [ -s "$tmp/$sub/layer_$nn/openvino_model.bin" ]; then
      mkdir -p "$PREFIX/model/$sub"; rm -rf "$dst"; mv "$tmp/$sub/layer_$nn" "$dst" && echo "fused IRs: layer $l ready ($sub)"
      rm -f "$mark"
    else
      echo "fused IRs: layer $l failed (see $PREFIX/logs/fused-ir.log); it stays on the CPU path"
      : > "$mark"
    fi
    rm -rf "$tmp"
  done
}
write_generator() {
cat > "$1" <<'PYGEN_EOF'
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


def stacked_weight(packed_list, scale_list, out, inn, layout="u4zp", scale_shift=0):
    """Expert-major compressed constant + f16 scales -> the plugin's decompression chain -> [E, out, in] f32.

    `u4zp` (default): the bins' nibbles verbatim as `u4` with an explicit zero point of 8 —
    every weight constant then lives in the IR .bin with an offset, which the plugin's
    offload path (OFFLOAD_RATIO / WEIGHTS_PATH) requires. `i4`: signed nibbles (bins' bytes
    with the high bit of each nibble flipped), no zero point — the same values, the layout
    optimum-intel's `--sym` export uses; the plugin cannot offload it (2026.3.1 asks the
    zero-point placeholder for a bin offset)."""
    e = len(packed_list)
    if OUT_GROUP[0] != GROUP:
        return regrouped_weight(packed_list, scale_list, out, inn, OUT_GROUP[0], scale_shift)
    ng = inn // GROUP
    raw = np.concatenate([p.reshape(-1) for p in packed_list])
    sc = np.stack([_bf16_to_f16(s) for s in scale_list]).reshape(e, out, ng, 1)
    if scale_shift:
        # A power of two moves only the exponent: exact unless the scale drops below f16's smallest normal (6.1e-5),
        # where it keeps fewer mantissa bits. Reported so the caller can judge (see --up-scale-exponent).
        f32 = sc.astype(np.float32) * np.float32(2.0 ** -scale_shift)
        sub = int(np.count_nonzero((np.abs(f32) < 6.1e-5) & (f32 != 0)))
        print(f"  scales x 2^-{scale_shift}: {sub} of {f32.size} ({100.0 * sub / f32.size:.2f} %) become f16 subnormals")
        sc = f32.astype(np.float16)
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


# Group size of the weights WRITTEN (the bins are always GROUP = 32). 32 = the bins' nibbles and scales verbatim.
# 64 / 128 (`--group`): the GPU plugin's MoE DECODE kernels (batched GEMV, the path built for a few rows) refuse
# group 32 on Xe2 and newer (they need group >= 2 x sub-group size = 64; the refusal is swallowed at compile time and
# surfaces as "Unable to cast reference from base to derived type" at infer). Regrouping re-quantises: each wider
# group gets its own scale and zero point (asymmetric u4), fitted to the dequantised 32-group weights.
OUT_GROUP = [GROUP]


def _bf16_bits_to_f32(u16):
    """bfloat16 bit patterns -> float32, exactly (same exponent width: shift into the top 16 bits)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def regrouped_weight(packed_list, scale_list, out, inn, group, scale_shift=0):
    """As `stacked_weight` (u4 + zero point + f16 scale -> [E, out, in] f32), with `group`-wide groups."""
    assert group % GROUP == 0 and inn % group == 0, (group, inn)
    e, ng = len(packed_list), inn // group
    q_all = np.empty((e, out, ng, group), np.uint8)
    s_all = np.empty((e, out, ng, 1), np.float16)
    z_all = np.empty((e, out, ng, 1), np.uint8)
    err_num = err_den = 0.0
    for i, (p, sc) in enumerate(zip(packed_list, scale_list)):
        v = np.empty((out, inn), np.float32)
        v[:, 0::2] = (p & 0x0F).astype(np.float32) - 8.0
        v[:, 1::2] = (p >> 4).astype(np.float32) - 8.0
        w = (v.reshape(out, inn // GROUP, GROUP) * _bf16_bits_to_f32(sc).reshape(out, inn // GROUP, 1)).reshape(out, ng, group)
        mn = np.minimum(w.min(axis=2, keepdims=True), 0.0)
        mx = np.maximum(w.max(axis=2, keepdims=True), 0.0)
        s = ((mx - mn) / 15.0).astype(np.float16).astype(np.float32)
        s[s == 0] = 1.0
        z = np.clip(np.rint(-mn / s), 0, 15)
        q = np.clip(np.rint(w / s) + z, 0, 15)
        err_num += float((((q - z) * s - w) ** 2).sum()); err_den += float((w ** 2).sum())
        q_all[i] = q.astype(np.uint8); z_all[i] = z.astype(np.uint8)
        s_all[i] = (s * np.float32(2.0 ** -scale_shift)).astype(np.float16)
    print(f"  regrouped {e} x [{out}, {inn}] to group {group}: weight error {100.0 * (err_num / max(err_den, 1e-30)) ** 0.5:.2f} % rms of the group-32 weights")
    flat = q_all.reshape(-1)
    raw = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    zf = z_all.reshape(-1)
    if zf.size % 2:
        zf = np.append(zf, np.uint8(8))
    zraw = (zf[0::2] | (zf[1::2] << 4)).astype(np.uint8)
    wq = raw_const(Type.u4, [e, out, ng, group], raw.tobytes())
    zp = raw_const(Type.u4, [e, out, ng, 1], zraw.tobytes())
    w = ops.subtract(ops.convert(wq, Type.f16), ops.convert(zp, Type.f16))
    w = ops.multiply(w, ops.constant(s_all))
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


def layer_model(src, lid, man, layout="u4zp", pad_experts=4, up_shift=0):
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
    up_w = stacked_weight([u[0] for u in ups], [u[1] for u in ups], inter, hidden, layout, up_shift)
    down_w = stacked_weight([d[0] for d in downs], [d[1] for d in downs], hidden, inter, layout)
    return build_layer(gate_w, up_w, down_w, hidden, inter, n_total, k + n_sh)


def dense_model(src, lid, man, layout="u4zp"):
    """The dense MLP of one of the first layers as three compressed MatMuls with a dynamic row count:
    x [1, rows, H] -> down(silu(gate x) * up x) -> y [1, rows, H] (the runtime applies mlp.global_scale)."""
    hidden, inter = man["hidden_size"], man["dense_intermediate"]
    (gp, gs), (up_, us), (dp, ds) = read_bin_sections(os.path.join(src, "experts", f"layer_{lid:02d}", "dense.bin"), hidden, inter)
    sq = ops.constant(np.array([0], np.int64))
    gate_w = ops.squeeze(stacked_weight([gp], [gs], inter, hidden, layout), sq)     # [I, H]
    up_w = ops.squeeze(stacked_weight([up_], [us], inter, hidden, layout), sq)
    down_w = ops.squeeze(stacked_weight([dp], [ds], hidden, inter, layout), sq)     # [H, I]
    x = ops.parameter(PartialShape([1, -1, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    g = ops.matmul(x, gate_w, False, True)
    u = ops.matmul(x, up_w, False, True)
    sw = NodeFactory("opset4").create("Swish", [g.output(0)], {})
    y = ops.matmul(ops.multiply(sw, u), down_w, False, True)
    m = Model([y], [x], "inkling_dense_mlp")
    m.outputs[0].tensor.set_names({"y"})
    return m


def dense_as_moe_model(src, lid, man, layout="u4zp", pad_experts=4):
    """The dense MLP as an all-experts-active MoE layer: its inner neurons cut into slices as wide as a routed expert
    (24,576 / 3,072 = 8 on Inkling), each slice an "expert" of the fused block, every row selecting all of them with
    weight 1. `down(silu(gate x) * up x)` is a sum over inner neurons, so the slices' outputs add up to the same
    vector; gate/up are cut by rows and down by columns on the bins' own 32-weight groups, so no weight is requantised.
    Why: on the Arc B390 three compressed MatMuls 24,576 wide read their 4-bit weights at ~26 GB/s, the fused
    3-GEMM op at 80-130 GB/s, and rank 0's two dense layers made it the slowest stage of the pipeline."""
    hidden, inter, ei = man["hidden_size"], man["dense_intermediate"], man["moe_intermediate"]
    assert inter % ei == 0 and ei % GROUP == 0 and hidden % GROUP == 0, (inter, ei)
    n = inter // ei
    (gp, gs), (up_, us), (dp, ds) = read_bin_sections(os.path.join(src, "experts", f"layer_{lid:02d}", "dense.bin"), hidden, inter)
    gates = [(gp[e * ei:(e + 1) * ei], gs[e * ei:(e + 1) * ei]) for e in range(n)]
    ups = [(up_[e * ei:(e + 1) * ei], us[e * ei:(e + 1) * ei]) for e in range(n)]
    downs = [(np.ascontiguousarray(dp[:, e * ei // 2:(e + 1) * ei // 2]),
              np.ascontiguousarray(ds[:, e * ei // GROUP:(e + 1) * ei // GROUP])) for e in range(n)]
    for _ in range(pad_experts):
        gates.append((np.full_like(gates[0][0], 0x88), np.zeros_like(gates[0][1])))
        ups.append((np.full_like(ups[0][0], 0x88), np.zeros_like(ups[0][1])))
        downs.append((np.full_like(downs[0][0], 0x88), np.zeros_like(downs[0][1])))
    gate_w = stacked_weight([g[0] for g in gates], [g[1] for g in gates], ei, hidden, layout)
    up_w = stacked_weight([u[0] for u in ups], [u[1] for u in ups], ei, hidden, layout)
    down_w = stacked_weight([d[0] for d in downs], [d[1] for d in downs], hidden, ei, layout)
    return build_layer(gate_w, up_w, down_w, hidden, ei, n + pad_experts, n)


def validate_dense_as_moe(src, lid, man, device="CPU", layout="u4zp", pad_experts=4):
    """The all-slices-active MoE form against the three-MatMul form of the same dense layer, on `device`."""
    core = ov.Core()
    a = core.compile_model(dense_model(src, lid, man, layout), device)
    b = core.compile_model(dense_as_moe_model(src, lid, man, layout, pad_experts), device)
    n = man["dense_intermediate"] // man["moe_intermediate"]
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((1, 3, man["hidden_size"])) * 0.05).astype(np.float32)
    ids = np.tile(np.arange(n, dtype=np.int32), (3, 1))
    w = np.ones((3, n), np.float32)
    ya = a({"x": x})[a.outputs[0]]
    yb = b({"x": x, "topk_indices": ids, "routing_weights": w})[b.outputs[0]]
    rel = float(np.sqrt(((ya - yb) ** 2).mean()) / (np.sqrt((ya ** 2).mean()) + 1e-30))
    print(f"layer {lid}: dense-as-MoE vs three MatMuls on {device}: rel RMS {rel:.3e}, max |y| {float(np.abs(ya).max()):.4g}")
    return rel


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
    ap.add_argument("--up-scale-exponent", type=int, default=0,
                    help="multiply every up-projection scale by 2^-N; the layer then returns y * 2^-N and the runtime multiplies "
                         "it back (it reads cascadia_moe.json next to the IR). For layers whose expert output itself passes "
                         "f16's 65504 on the device (Inkling layer 8's shared expert reaches -94909); 4 is the tested value")
    ap.add_argument("--dense", action="store_true",
                    help="write the DENSE layers among --layers instead (their MLP as <out>/dense_ov/layer_NN/), skip the MoE ones")
    ap.add_argument("--group", type=int, default=GROUP, choices=[32, 64, 128],
                    help="group size of the weights written (the bins are group 32). 64 or 128 re-quantise (asymmetric u4 per "
                         "group) so that the GPU plugin's MoE decode kernels accept the layer on Xe2+ (they refuse group 32)")
    ap.add_argument("--moe-dir", default="moe_ov", help="folder name under --out for the MoE layers (default moe_ov)")
    ap.add_argument("--dense-as-moe", action="store_true",
                    help="write the DENSE layers among --layers as all-slices-active fused-MoE IRs (<out>/dense_moe_ov/layer_NN/): "
                         "the same sum through the fused 3-GEMM op, which reads 4-bit weights about three times faster on the iGPU")
    args = ap.parse_args()
    OUT_GROUP[0] = args.group
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    man = json.load(open(os.path.join(args.src, "manifest.json")))
    assert man.get("arch") == "inkling", man.get("arch")
    dense = set(man["dense_layers"])
    out = args.out or args.src
    for lid in [int(v) for v in args.layers.split(",")]:
        if args.dense_as_moe:
            if lid not in dense:
                print(f"layer {lid}: not a dense layer, skipped")
                continue
            if args.validate:
                validate_dense_as_moe(args.src, lid, man, args.validate_device, args.layout, args.pad_experts)
                continue
            dst = os.path.join(out, "dense_moe_ov", f"layer_{lid:02d}")
            xml = os.path.join(dst, "openvino_model.xml")
            if args.skip_existing and os.path.exists(xml):
                print(f"layer {lid}: exists, skipped")
                continue
            t0 = time.time()
            os.makedirs(dst, exist_ok=True)
            ov.save_model(dense_as_moe_model(args.src, lid, man, args.layout, args.pad_experts), xml, compress_to_fp16=False)
            with open(os.path.join(dst, "cascadia_moe.json"), "w") as f:
                json.dump({"up_scale_exponent": 0, "dense_slices": man["dense_intermediate"] // man["moe_intermediate"]}, f)
            print(f"layer {lid}: dense MLP as MoE written {xml} in {time.time()-t0:.0f}s")
            continue
        if args.dense:
            if lid not in dense:
                print(f"layer {lid}: not a dense layer, skipped")
                continue
            dst = os.path.join(out, "dense_ov", f"layer_{lid:02d}")
            xml = os.path.join(dst, "openvino_model.xml")
            if args.skip_existing and os.path.exists(xml):
                print(f"layer {lid}: exists, skipped")
                continue
            t0 = time.time()
            os.makedirs(dst, exist_ok=True)
            ov.save_model(dense_model(args.src, lid, man, args.layout), xml, compress_to_fp16=False)
            print(f"layer {lid}: dense MLP written {xml} in {time.time()-t0:.0f}s")
            continue
        if lid in dense:
            print(f"layer {lid}: dense, skipped (the dense MLP keeps the per-expert IR)")
            continue
        if args.validate:
            validate(args.src, lid, man, args.validate_device, args.layout, args.pad_experts)
            continue
        dst = os.path.join(out, args.moe_dir, f"layer_{lid:02d}")
        xml = os.path.join(dst, "openvino_model.xml")
        if args.skip_existing and os.path.exists(xml):
            print(f"layer {lid}: exists, skipped")
            continue
        t0 = time.time()
        m = layer_model(args.src, lid, man, args.layout, args.pad_experts, args.up_scale_exponent)
        os.makedirs(dst, exist_ok=True)
        ov.save_model(m, xml, compress_to_fp16=False)
        with open(os.path.join(dst, "cascadia_moe.json"), "w") as f:
            json.dump({"up_scale_exponent": args.up_scale_exponent, "group": args.group}, f)
        print(f"layer {lid}: written {xml} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
PYGEN_EOF
}
# The iGPU may hold more than half of RAM when the overrides say so (CASCADIA_GPU_PAGES_GIB): the kernel's default
# limit for driver-owned system memory is 50 %, which is three fused layers of 8.3 GB on a 61 GiB box. Not persistent.
if [ -n "${CASCADIA_GPU_PAGES_GIB:-}" ] && [ -w /sys/module/ttm/parameters/pages_limit ]; then
  case "$CASCADIA_GPU_PAGES_GIB" in ''|*[!0-9]*) ;; *) echo $((CASCADIA_GPU_PAGES_GIB * 262144)) > /sys/module/ttm/parameters/pages_limit 2>/dev/null || true ;; esac
fi
gen_fused_irs "${CASCADIA_FUSE_LAYERS:-}" 32 || true
gen_fused_irs "${CASCADIA_FUSE_DECODE_LAYERS:-}" "${CASCADIA_FUSE_GROUP:-64}" || true
# The dense layers' MLP (rank 0) as one device call for all rows of a frame (CASCADIA_FUSE_DENSE="0,1"): same
# generator, same rules (scratch folder, moved into place when complete, a failure is marked and tolerated).
# CASCADIA_FUSE_DENSE_MOE="0,1" writes the same MLP once more as an all-slices-active fused-experts layer
# (dense_moe_ov/): the form in which this GPU reads 4-bit weights three to four times faster (autolab 027).
gen_dense_irs() {
  # $1 = layers, $2 = generator flag, $3 = folder under model/
  local want="${1:-}" flag="$2" sub="$3" py l nn dst tmp mark
  [ -n "$want" ] || return 0
  py=$(command -v python3) || return 0
  [ -d "$PREFIX/pylib" ] || return 0
  mkdir -p "$PREFIX/tools" "$PREFIX/logs"
  tmp="$PREFIX/model/.fuse-tmp"
  for l in ${want//,/ }; do
    case "$l" in ''|*[!0-9]*) continue ;; esac
    [ "$l" -ge "${LAYER_START:-0}" ] && [ "$l" -lt "${LAYER_END:-0}" ] || continue
    nn=$(printf %02d "$l"); dst="$PREFIX/model/$sub/layer_$nn"; mark="$PREFIX/logs/dense-ir.failed-$nn"
    [ "$sub" = "dense_ov" ] || mark="$PREFIX/logs/dense-ir.failed-$sub-$nn"
    [ -s "$dst/openvino_model.xml" ] && [ -s "$dst/openvino_model.bin" ] && continue
    [ -e "$mark" ] && [ "${CASCADIA_FUSE_RETRY:-0}" != 1 ] && continue
    write_generator "$PREFIX/tools/inkling_moe_layer_ov.py"
    rm -rf "$tmp"; mkdir -p "$tmp"
    echo "dense IRs: generating layer $l ($sub)"
    if PYTHONPATH="$PREFIX/pylib" timeout 600 "$py" "$PREFIX/tools/inkling_moe_layer_ov.py" --src "$PREFIX/model" --out "$tmp" \
         --layers "$l" "$flag" >> "$PREFIX/logs/fused-ir.log" 2>&1 && [ -s "$tmp/$sub/layer_$nn/openvino_model.bin" ]; then
      mkdir -p "$PREFIX/model/$sub"; rm -rf "$dst"; mv "$tmp/$sub/layer_$nn" "$dst" && echo "dense IRs: layer $l ready ($sub)"
    else
      echo "dense IRs: layer $l failed (see $PREFIX/logs/fused-ir.log); it keeps its previous path"; : > "$mark"
    fi
    rm -rf "$tmp"
  done
}
gen_dense_irs "${CASCADIA_FUSE_DENSE:-}" --dense dense_ov || true
gen_dense_irs "${CASCADIA_FUSE_DENSE_MOE:-}" --dense-as-moe dense_moe_ov || true
# side-by-side OpenVINO runtime for this process only
if [ -n "${OVDIR:-}" ] && [ -f "$OVDIR/setupvars.sh" ]; then set +u; source "$OVDIR/setupvars.sh" > /dev/null; set -u; fi
args=(worker --rank "$RANK" --total "$TOTAL" --engine sparse-moe --device CPU --model "$PREFIX/model"
      --layer-start "$LAYER_START" --layer-end "$LAYER_END")
[ "$RANK" -gt 0 ] && args+=(--listen ":$((9100 + RANK))")
[ -n "${NEXT:-}" ] && args+=(--next "$NEXT")
[ "$RANK" = 0 ] && args+=(--api ":8000")
# What the dashboard shows for this box. --device stays CPU (the engine's own device: routing and the CPU-side
# expert layers); these two flags only label the card, and only with what this rank is really set up to use.
if [ "${CASCADIA_INKLING_OV_ATTN:-0}" = 1 ]; then
  args+=(--advertise-device iGPU)
  [ -n "${CASCADIA_INKLING_OV_MOE_LAYERS:-}" ] && args+=(--advertise-engines "sparse-moe,fused-moe")
fi
# Rank 0 is the only rank that dials without first waiting for anyone, and it starts instantly. After a
# "systemctl restart" it would reach rank 1 while that process is still exiting (its listener is open for a
# moment longer), then sit on a dead link until the first request makes it re-dial - and until then the ranks
# behind it wait unloaded. Give them time to come back first (they restart 5 s after they notice).
[ "$RANK" = 0 ] && sleep 8
exec "$PREFIX/cascadia" "${args[@]}"
