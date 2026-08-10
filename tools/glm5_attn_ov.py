#!/usr/bin/env python3
"""GLM-5.2 MLA attention: OpenVINO exporter and numerics validation harness.

Builds the per-layer OpenVINO graphs a glm5 rank uses to run prefill attention
on an Intel iGPU instead of the Rust path, and the self-checks that prove those
graphs reproduce the Rust engine's numerics.

Modes:
  --validate-ops     per-op harness (`linear`, `rmsnorm`, `rope`, the bf16
                     round-trip subgraph, and the rope-table exact gate).
                     Self-contained: no model dir, synthetic weights.
  --validate-graph   the same discipline on a whole assembled attention graph
                     vs a numpy reference. Self-contained; needs a rope dump.
  --export MODEL_DIR write MODEL_DIR/attn_ov/{layer_NN.xml,.bin,
                     export_stamp.json}. Needs manifest.json +
                     shells/layer_NN.safetensors, and a rope dump.
  --canary           compile the shipped bf16-rounding subgraph on --device
                     and assert its raw output bits. Run this on a node before
                     trusting its GPU.
  --bench MODEL_DIR  time one layer's graph against the Rust baseline. Needs a
                     real model dir on the target machine.

Exit codes: 0 all checks ran and passed; 1 a check failed, or a compile drifted
off the required config, or a quantized kernel appeared; 2 everything that ran
passed but a safety-critical check was SKIPPED (e.g. no rope-table dump).

Numeric contract (see `dsv4/math.rs`, `dsv4/rope.rs`, `glm/attn.rs` header):
  * every linear / RMSNorm / rope output is rounded to bf16 (round-to-nearest-
    even); everything else accumulates in f32.
  * the graph's `x` input is the RAW residual: the graph applies
    `rmsnorm(input_layernorm)` itself, as its first op. The caller must not
    pre-normalize.
  * `in_ln` uses the manifest's `rms_norm_eps` (1e-5); `q_a_ln`/`kv_a_ln` use
    `glm::attn::MLA_LATENT_EPS` (1e-6) — two DIFFERENT epsilons, so `rmsnorm`
    takes eps as a real graph input (not a baked constant) and validation
    proves 1e-5 vs 1e-6 diverge.
  * dot-product reduction order differs across numpy / OV / Rust (`math.rs`
    documents this as tolerated ULP-level divergence), so exact equality is
    NOT the gate for `linear`/`rmsnorm`/`rope`'s arithmetic — a bounded
    bf16-ULP tolerance is. The one thing that MUST be exact is the rope
    cos/sin table itself: a silently-wrong-basis table is a decode-breaking
    bug that short-prompt parity would not catch. The table is never computed
    here — it is loaded from a bit-for-bit dump of the Rust `Freqs` struct and
    verified against the sha256 in that dump's sidecar. Produce one with:
      cargo test -p cascadia-engine-sparse-moe --lib dump_real_dims_freqs

Usage:
  python3 tools/glm5_attn_ov.py --validate-ops
  python3 tools/glm5_attn_ov.py --validate-graph --rope-table-dump /path/dump.bin
  python3 tools/glm5_attn_ov.py --export /models/glm5 --rope-table-dump /path/dump.bin
  python3 tools/glm5_attn_ov.py --canary --device GPU
"""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import statistics
import tempfile
import time

import numpy as np
import openvino as ov
from openvino import Model, PartialShape, Type
from openvino import opset15 as ops

# Rust's f32::cos/sin/powf lower to the platform's libm cosf/sinf/powf.
# numpy's vectorized cos/sin/power on float32 arrays do NOT call those same
# symbols (their own SIMD polynomial approximations), so they can land 1 ULP
# off Rust's table. Calling libm directly via ctypes uses the IDENTICAL
# symbols Rust calls on this host, which is what actually gets the rope table
# to bit-exact match. Falls back to numpy (with a printed warning) if libm
# can't be loaded — the exact-match check then reports NOT EXACT instead of
# silently accepting numpy's approximation.
try:
    _libm_name = ctypes.util.find_library("m")
    if _libm_name is None and os.name == "nt":
        # Windows has no libm: C99 float math lives in the UCRT, which is
        # where Rust std lowers f32::cos/sin/powf on windows-msvc — so this
        # keeps the same-symbols property the bit-exact gate depends on.
        _libm_name = "ucrtbase"
    if _libm_name is None:
        raise OSError("no system libm found")
    _libm = ctypes.CDLL(_libm_name)
    _libm.cosf.restype = ctypes.c_float
    _libm.cosf.argtypes = [ctypes.c_float]
    _libm.sinf.restype = ctypes.c_float
    _libm.sinf.argtypes = [ctypes.c_float]
    _libm.powf.restype = ctypes.c_float
    _libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
except (OSError, TypeError, AttributeError):
    _libm = None
    print("WARNING: could not load system libm via ctypes; rope table will use "
          "numpy's cos/sin/power, which do not bit-match Rust's libm calls "
          "(the rope-table exact gate will report NOT EXACT).")

COMPILE_CFG = {"INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
DEVICE = "CPU"  # no GPU on this dev machine; the graphs are device-agnostic

# The binding contract: every compile must land on these EFFECTIVE values, not
# just request them — a hint is not a guarantee. `compile_checked` reads both
# back from every compiled model and fails the run if either drifted.
REQUIRED_PRECISION_HINT = Type.f32
REQUIRED_DYN_QUANT_GROUP_SIZE = 0

# GLM-5.2 real export dims — order-of-magnitude placeholders until a real
# manifest supplies exact values (this harness is model-dir-free by design).
REAL_HIDDEN = 4096
REAL_Q_LORA = 1536
REAL_KV_LORA = 512
REAL_QK_ROPE = 64
REAL_ROPE_THETA = 8_000_000.0  # export_glm5.py default; loader.rs passes original_seq_len=0 (no YaRN)

RMS_NORM_EPS = 1e-5   # manifest's rms_norm_eps, applies to in_ln
MLA_LATENT_EPS = 1e-6  # glm::attn::MLA_LATENT_EPS, applies to q_a_ln / kv_a_ln


# --------------------------------------------------------------------------
# bf16 rounding: numpy reference ops
# --------------------------------------------------------------------------

def bf16_round(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 (round-to-nearest-even) -> f32, matching
    `half::bf16::from_f32(v).to_f32()` (`dsv4/math.rs::to_bf16`).

    Uses the round-to-nearest-even bit trick
    `((bits + 0x7FFF + ((bits>>16)&1)) >> 16) << 16`. Verified against
    `half::bf16::from_f32` on 20 hand-picked cases including exact ties
    (`0x..._8000`, both retained-LSB parities) and a mantissa-carry-into-
    exponent tie — see `crates/cascadia-engine-sparse-moe/src/dsv4/math.rs`'s
    `bf16_ties::matches_half_crate` test, which is the ground truth this was
    checked against.

    FINDING: the brief's trick formula does NOT reproduce `half`'s NaN
    handling (`half` forces the qNaN bit via `(bits>>16)|0x0040`; the trick
    formula's carry can instead corrupt the exponent). NaN is special-cased
    below. Activations in this contract are never NaN in practice, but a
    literal port of the brief's formula would silently mishandle it.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    bits = x.view(np.uint32)
    is_nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)) << np.uint32(16)
    nan_bits = ((bits >> np.uint32(16)) | np.uint32(0x40)) << np.uint32(16)
    out_bits = np.where(is_nan, nan_bits, rounded).astype(np.uint32)
    return out_bits.view(np.float32).copy()


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """The bf16-rounded value's top-16 bits, as uint16 (the actual bf16 bit
    pattern `half::bf16::from_f32(v).to_bits()` would produce)."""
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def ref_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Port of `dsv4/math.rs::rmsnorm`: y = bf16(w * (x / sqrt(mean(x^2)+eps))),
    f32 internal math, one bf16 rounding at the end. `x`: [..., dim],
    `w`/`eps` broadcast over the leading dims."""
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    ms = np.mean((x * x).astype(np.float32), axis=-1, keepdims=True, dtype=np.float32)
    ms = ms.astype(np.float32)
    r = np.float32(1.0) / np.sqrt((ms + np.float32(eps)).astype(np.float32))
    return bf16_round(x * r.astype(np.float32) * w)


def ref_linear_bf16(x: np.ndarray, w_bits: np.ndarray) -> np.ndarray:
    """Port of `dsv4/math.rs::linear_bf16_w`: y[o] = bf16(dot(bf16_widen(w[o]), x)).
    `w_bits`: [out_dim, in_dim] uint16 bf16 bit patterns (the Rust weight
    store's own dtype). `x`: [in_dim] f32."""
    x = np.asarray(x, dtype=np.float32)
    w_bits = np.asarray(w_bits, dtype=np.uint16)
    w = (w_bits.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(w_bits.shape)
    y = (w.astype(np.float32) @ x.astype(np.float32)).astype(np.float32)
    return bf16_round(y)


def _rope_cos_sin(theta: float, rot_dims: int, pos: int) -> np.ndarray:
    """[half, 2] (cos, sin) table for one position — plain (non-YaRN) rope,
    matching `dsv4/rope.rs::precompute_freqs` at GLM-5.2's real call site
    (`glm/loader.rs` always passes `original_seq_len=0`, so the YaRN
    correction-range branch never executes for this model).

    Computed via the system's libm cosf/sinf/powf (ctypes) rather than
    numpy's vectorized cos/sin/power — see the `_libm` comment at module
    scope for why that's what makes the table bit-exact against Rust."""
    half = rot_dims // 2
    theta_f = ctypes.c_float(theta)
    out = np.empty((half, 2), dtype=np.float32)
    for i in range(half):
        # single f32 division `(2*i)/dim`, matching Rust's `(2*i) as f32 / dim
        # as f32` exactly — dividing in python float (f64) first and only
        # then truncating to f32 would double-round and could disagree.
        exp = np.float32(2 * i) / np.float32(rot_dims)
        if _libm is not None:
            base = _libm.powf(theta_f, ctypes.c_float(float(exp)))
            freq = np.float32(1.0) / np.float32(base)
            ang = np.float32(np.float32(pos) * freq)
            out[i, 0] = _libm.cosf(ctypes.c_float(float(ang)))
            out[i, 1] = _libm.sinf(ctypes.c_float(float(ang)))
        else:
            freq = np.float32(1.0) / np.power(np.float32(theta), exp)
            ang = np.float32(np.float32(pos) * freq)
            out[i, 0] = np.cos(ang)
            out[i, 1] = np.sin(ang)
    return out


def precompute_freqs_table(rot_dims: int, seqlen: int, theta: float) -> np.ndarray:
    """Flat `[seqlen * half * 2]` f32 table, position-major, interleaved
    (cos, sin) — same layout as Rust's `Freqs.data`
    (`dsv4/rope.rs::freqs_dump::dump_real_dims_freqs` dumps the byte-for-byte
    comparable table for GLM-5.2's real dims)."""
    return np.stack(
        [_rope_cos_sin(theta, rot_dims, t) for t in range(seqlen)]
    ).reshape(-1).astype(np.float32)


def ref_rope_interleaved(x: np.ndarray, pos: int, theta: float, rot_dims: int) -> np.ndarray:
    """Port of `dsv4/rope.rs::apply_rope_row` (non-inverse): rotates the LAST
    `rot_dims` elements of each row of `x` as adjacent (even, odd) complex
    pairs, bf16-rounding each output element. `x`: [..., row_dim] f32."""
    x = np.asarray(x, dtype=np.float32)
    row_dim = x.shape[-1]
    start = row_dim - rot_dims
    cs = _rope_cos_sin(theta, rot_dims, pos)  # [half, 2]
    c, s = cs[:, 0], cs[:, 1]
    out = x.copy()
    a = x[..., start:row_dim:2]
    b = x[..., start + 1:row_dim:2]
    out[..., start:row_dim:2] = bf16_round(a * c - b * s)
    out[..., start + 1:row_dim:2] = bf16_round(a * s + b * c)
    return out


# --------------------------------------------------------------------------
# 20 hand-picked bf16 tie cases — SAME list as
# `crates/cascadia-engine-sparse-moe/src/dsv4/math.rs::bf16_ties::CASES`.
# Used to cross-check the `_bf16_roundtrip` subgraph against `bf16_round` on
# exactly the cases verified against `half::bf16::from_f32`.
# --------------------------------------------------------------------------
BF16_TIE_CASES = [
    (0x3F808000, 0x3F80), (0x3F818000, 0x3F82), (0x3F808001, 0x3F81),
    (0x3F817FFF, 0x3F81), (0xBF808000, 0xBF80), (0xBF818000, 0xBF82),
    (0x40FF8000, 0x4100), (0x3F800000, 0x3F80), (0x00000000, 0x0000),
    (0x80000000, 0x8000), (0x7F800000, 0x7F80), (0xFF800000, 0xFF80),
    (0x7FC00000, 0x7FC0), (0x7F800001, 0x7FC0), (0x00000001, 0x0000),
    (0x3F7FFFFF, 0x3F80), (0x42F60000, 0x42F6), (0x42F68000, 0x42F6),
    (0x42F78000, 0x42F8), (0xC2F68000, 0xC2F6),
]


# --------------------------------------------------------------------------
# OV graph builders
# --------------------------------------------------------------------------

def _raw_const(t: Type, dims: list, raw: bytes):
    """Constant of low-precision Type `t` built from raw LE bytes, verbatim
    (same pattern as `tools/glm5_expert_ov.py::_raw_const`: the python
    Constant ctor has no bytes overload, so fill a Tensor's byte view)."""
    ten = ov.Tensor(t, ov.Shape(dims))
    view = ten.data if isinstance(ten.data, np.ndarray) else np.frombuffer(ten.data, np.uint8)
    view = view.reshape(-1).view(np.uint8)
    src = np.frombuffer(raw, np.uint8)
    assert view.size == src.size, f"{t} tensor {dims}: {view.size}B buffer vs {src.size}B raw"
    view[:] = src
    return ov.op.Constant(ten, shared_memory=False)


# Strictly decreasing powers of two, sum=127. Every `2**step` (and its
# reciprocal) used below is a NORMAL float32 (>= 2**-64, far above the
# subnormal threshold 2**-126) -- deliberately narrower than the full f32
# exponent range so no intermediate ever touches a subnormal, where FTZ/DAZ
# SIMD execution can silently flush a value to zero and corrupt the search
# (this bit us during development at the full [-127,127] range). Bounds the
# supported magnitude to roughly [2**-64, 2**63], comfortably beyond any
# realistic bf16-native activation.
_BF16_BISECT_STEPS = (64, 32, 16, 8, 4, 2, 1)


def _bf16_roundtrip(node):
    """Round `node` (f32) to the bf16 grid (round-to-nearest-even) and back to
    f32, matching `half::bf16::from_f32(v).to_f32()` / `dsv4/math.rs::to_bf16`
    -- the trailing rounding the Rust contract applies after every linear /
    RMSNorm / rope.

    NOT implemented as `Convert(f32->bf16)->Convert(bf16->f32)`: that pair is
    silently defeated under `INFERENCE_PRECISION_HINT=f32` (the exact config
    this whole exporter is required to compile under) -- the CPU plugin
    treats it as a "compressed-weight precision hint" and removes it as a
    redundant identity (verified: the compiled exec graph literally drops
    both Converts, `out == x` bit-for-bit; the plugin can ALSO be caught
    doing plain truncation instead of round-to-nearest when the pair isn't
    fully eliminated). The `--validate-ops` ULP gate never caught this
    because it re-rounds BOTH sides (ref and graph output) before comparing,
    which mostly still lands in the same bf16 bucket -- the stricter
    requirement (`lc`/`rc` must land EXACTLY on the bf16 grid, since
    Rust caches literal bf16 bits) is what exposed it.

    Implemented instead as genuine arithmetic reproducing `bf16_round`'s
    round-to-nearest-even bit trick WITHOUT bit access (OpenVINO's available
    opsets have no bitcast/reinterpret op): find `e` with `2**e <= |x| <
    2**(e+1)` via exact power-of-two bisection (comparisons and multiplies by
    powers of two are exact in IEEE754, so this introduces no ULP-level noise
    of its own), then `round(x / 2**(e-7), half_to_even) * 2**(e-7)` --
    `ops.round`'s `half_to_even` mode was verified directly against known
    tie cases (0.5->0, 1.5->2, 2.5->2, -0.5->-0, -1.5->-2). `x==0` and
    `is_nan(x)` are special-cased (the bisection has no valid exponent for
    either); NaN is canonicalized to `0x7FC0` widened, matching every
    `BF16_TIE_CASES` NaN expectation, though NOT `half`'s payload- or
    SIGN-preserving behavior (a negative NaN loses its sign bit here,
    e.g. `0xffc00000 -> 0x7fc00000`) -- unreachable in practice, matching
    `bf16_round`'s own doc note that activations are never NaN.

    The bisection floor is `2**-64`, not the full f32
    subnormal range (`_BF16_BISECT_STEPS`'s comment explains why -- a
    starting constant at the true f32 floor, `2**-127`, is itself a
    SUBNORMAL and got silently flushed to zero during development, corrupting
    the search). The precise, verified consequence: every genuine f32
    SUBNORMAL, and every normal value with magnitude below `2**-64`
    (~5.4e-20), rounds to exactly `0.0` here -- NOT just "very extreme"
    inputs like `1e-30`. `bf16_round` (the numpy reference) rounds those
    correctly (down to the true bf16 subnormal boundary near `2**-133`).
    Practical impact is nil for this contract's activations (bf16-native,
    never remotely that small), but a later reader of this file should
    trust the bound as stated here, not a narrower one."""
    ax = ops.abs(node)
    e = ops.constant(np.float32(-64.0))
    p = ops.constant(np.float32(2.0 ** -64))
    for step in _BF16_BISECT_STEPS:
        cand_e = ops.add(e, ops.constant(np.float32(step)))
        cand_p = ops.multiply(p, ops.constant(np.float32(2.0 ** step)))
        take = ops.greater_equal(ax, cand_p)
        e = ops.select(take, cand_e, e)
        p = ops.select(take, cand_p, p)
    ulp = ops.multiply(p, ops.constant(np.float32(2.0 ** -7)))
    rounded = ops.round(ops.divide(node, ulp), "half_to_even")
    result = ops.multiply(rounded, ulp)
    is_zero = ops.equal(node, ops.constant(np.float32(0.0)))
    result = ops.select(is_zero, node, result)  # preserve signed zero exactly
    canonical_nan = ops.constant(np.uint32(0x7FC00000).view(np.float32).copy())
    return ops.select(ops.is_nan(node), canonical_nan, result)


def build_op_graph(op_name: str, dims: dict) -> Model:
    """One tiny OV graph per contract op, WITH its trailing bf16 round-trip.

    `dims` carries both shapes and the concrete numpy arrays to bake as
    constants (weights / rmsnorm gain), so the caller can feed the SAME
    arrays into the numpy reference op and get a graph baking those exact
    values — the arithmetic gap under test is reduction order, not different
    inputs.
      - 'linear':  {"out_dim", "in_dim", "w_bits" (uint16 [out,in])}
      - 'rmsnorm': {"dim", "w" (f32 [dim])}                  (eps is a graph input)
      - 'rope':    {"rot_dims", "theta", "pos"}
    """
    if op_name == "linear":
        out_dim, in_dim, w_bits = dims["out_dim"], dims["in_dim"], dims["w_bits"]
        x = ops.parameter(PartialShape([1, in_dim]), Type.f32, name="x")
        x.get_output_tensor(0).set_names({"x"})
        w_const = _raw_const(Type.bf16, [out_dim, in_dim], np.ascontiguousarray(w_bits).tobytes())
        y = ops.matmul(x, ops.convert(w_const, Type.f32), False, True)  # x @ w.T -> [1, out_dim]
        y = _bf16_roundtrip(y)
        m = Model([y], [x], "linear_bf16")
        m.outputs[0].tensor.set_names({"y"})
        return m

    if op_name == "rmsnorm":
        dim, w = dims["dim"], dims["w"]
        x = ops.parameter(PartialShape([1, dim]), Type.f32, name="x")
        x.get_output_tensor(0).set_names({"x"})
        eps_in = ops.parameter(PartialShape([]), Type.f32, name="eps")
        eps_in.get_output_tensor(0).set_names({"eps"})
        w_const = ops.constant(np.asarray(w, dtype=np.float32))
        ms = ops.reduce_mean(ops.multiply(x, x), ops.constant(np.array([1], np.int64)), True)
        denom = ops.sqrt(ops.add(ms, eps_in))
        r = ops.divide(ops.constant(np.float32(1.0)), denom)
        y = ops.multiply(ops.multiply(x, r), w_const)
        y = _bf16_roundtrip(y)
        m = Model([y], [x, eps_in], "rmsnorm")
        m.outputs[0].tensor.set_names({"y"})
        return m

    if op_name == "rope":
        rot_dims, theta, pos = dims["rot_dims"], dims["theta"], dims["pos"]
        half = rot_dims // 2
        # cos/sin as precomputed f32 constants (the design intent: no
        # transcendental op runs in-graph, so OV's sin/cos accuracy vs
        # numpy's is a non-issue for the arithmetic under test).
        cs = _rope_cos_sin(theta, rot_dims, pos)
        c_const = ops.constant(cs[:, 0].reshape(1, half).astype(np.float32))
        s_const = ops.constant(cs[:, 1].reshape(1, half).astype(np.float32))
        x = ops.parameter(PartialShape([1, rot_dims]), Type.f32, name="x")
        x.get_output_tensor(0).set_names({"x"})
        start0 = ops.constant(np.array([0], np.int64))
        start1 = ops.constant(np.array([1], np.int64))
        stop = ops.constant(np.array([rot_dims], np.int64))
        step2 = ops.constant(np.array([2], np.int64))
        axis1 = ops.constant(np.array([1], np.int64))
        a = ops.slice(x, start0, stop, step2, axis1)  # even indices -> [1, half]
        b = ops.slice(x, start1, stop, step2, axis1)  # odd indices -> [1, half]
        re = ops.subtract(ops.multiply(a, c_const), ops.multiply(b, s_const))
        im = ops.add(ops.multiply(a, s_const), ops.multiply(b, c_const))
        re_u = ops.unsqueeze(re, ops.constant(np.array([2], np.int64)))
        im_u = ops.unsqueeze(im, ops.constant(np.array([2], np.int64)))
        inter = ops.concat([re_u, im_u], axis=2)  # [1, half, 2] -> interleaved on reshape
        y = ops.reshape(inter, ops.constant(np.array([1, rot_dims], np.int64)), False)
        y = _bf16_roundtrip(y)
        m = Model([y], [x], "rope_interleaved")
        m.outputs[0].tensor.set_names({"y"})
        return m

    raise ValueError(f"unknown op {op_name!r}")


def _bf16_roundtrip_graph(n: int) -> Model:
    """Standalone `x[n] -> _bf16_roundtrip -> y` graph, used only to
    cross-check the bf16 rounding subgraph against `bf16_round` on the tie
    cases (a narrower question than the per-op ULP gate). Originally built
    around a bare `Convert(f32->bf16)->Convert(bf16->f32)` pair; that pair
    gets silently eliminated as an identity by the CPU plugin under
    `INFERENCE_PRECISION_HINT=f32` (see `_bf16_roundtrip`'s docstring).
    `_bf16_roundtrip` was replaced with an exact-arithmetic implementation
    that does NOT suffer that elision — but that alone did NOT make the check
    below meaningful again: `validate_bf16_roundtrip`'s comparison was
    SEPARATELY tautological (re-rounds the graph's output via `bf16_round`
    before comparing, which passes even for an untouched/identity output —
    reverting `_bf16_roundtrip` to the broken Convert pair left it green, per
    two independent reviews). Both bugs needed fixing; see
    `validate_bf16_roundtrip`'s docstring for the second one."""
    x = ops.parameter(PartialShape([n]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    y = _bf16_roundtrip(x)
    m = Model([y], [x], "bf16_roundtrip")
    m.outputs[0].tensor.set_names({"y"})
    return m


# --------------------------------------------------------------------------
# ULP comparison
# --------------------------------------------------------------------------

def _bf16_ulp_key(bits16: np.ndarray) -> np.ndarray:
    """Monotonic integer key for bf16 bit patterns: key(a) < key(b) iff
    float(a) < float(b) (standard IEEE-754 total-order trick), so |key
    difference| is exact ULP distance on the bf16 grid."""
    b = bits16.astype(np.int32)
    sign = (b & 0x8000) != 0
    return np.where(sign, -(b & 0x7FFF), b)


def ulp_stats(ref: np.ndarray, got: np.ndarray):
    """(max_ulp_diff, flip_rate, worst_index) between two f32 arrays, compared
    on the bf16 grid both are already rounded onto."""
    rb, gb = bf16_bits(ref), bf16_bits(got)
    diff = np.abs(_bf16_ulp_key(rb) - _bf16_ulp_key(gb))
    worst = int(np.argmax(diff))
    return int(diff.max()), float(np.mean(diff == 1)), worst, rb, gb


def on_grid(x: np.ndarray) -> bool:
    """`x == bf16_round(x)` — a STRUCTURAL check, independent of `ulp_stats`.

    The ULP check alone re-rounds both sides
    before comparing (see `validate_bf16_roundtrip`'s docstring for why that
    class of comparison is elision-blind on its own). Every bf16-contract op
    graph must be checked with BOTH assertions, not just one: `on_grid`
    catches "rounding never happened" (a graph that silently passed its
    input through would still often land within 1 ULP of the CORRECTLY
    rounded reference after re-rounding, but its raw output would almost
    never already BE on the bf16 grid); `ulp_stats`'s `max_ulp<=1` tolerance
    covers only the reduction-order noise numpy/OV/Rust legitimately differ
    on. Neither assertion subsumes the other."""
    x = np.asarray(x, dtype=np.float32)
    return bool(np.array_equal(x, bf16_round(x)))


_effective_config_total = 0
_effective_config_bad: list = []


def compile_checked(core, model: Model, name: str, device: str = None):
    """`core.compile_model` + effective-config readback. `INFERENCE_PRECISION_HINT`
    and `DYNAMIC_QUANTIZATION_GROUP_SIZE` are compile HINTS, not guarantees —
    the plugin can silently land elsewhere. Every compile in this harness
    (~17 of them) feeds the numerics every later task builds on, so each one
    reads both properties back and records a failure if either isn't exactly
    the required value, instead of trusting the requested config dict.
    `device` defaults to the module's `DEVICE` (CPU, for numerics validation);
    `--bench` (Step 6b) is the one caller that overrides it to target a real
    accelerator."""
    global _effective_config_total
    compiled = core.compile_model(model, device or DEVICE, COMPILE_CFG)
    _effective_config_total += 1
    for prop, required in (
        ("INFERENCE_PRECISION_HINT", REQUIRED_PRECISION_HINT),
        ("DYNAMIC_QUANTIZATION_GROUP_SIZE", REQUIRED_DYN_QUANT_GROUP_SIZE),
    ):
        effective = compiled.get_property(prop)
        if effective != required:
            _effective_config_bad.append(
                f"[{name}] {prop}: requested={COMPILE_CFG[prop]!r} "
                f"effective={effective!r} (required {required!r})"
            )
    return compiled


# `compile_checked` recorded drift into
# `_effective_config_bad`, but only main()'s `--validate-ops`/`--validate-graph`
# branch ever inspected it -- `--export` and `--bench` both `return`ed before
# that inline check ran, so a real drift on those two paths (the one that
# produces what ships, and the one that runs on the GPU plugin where
# precision/dyn-quant drift is the documented risk) was silently swallowed.
# Reproduced live: `--inject-bad-config` + `--export` wrote a complete
# `attn_ov/` tree and exited 0 with 3 unread mismatches recorded.
#
# `_config_drift_ok()` is idempotent (prints once, caches the verdict) so it
# can be called both inline (validate mode's existing report position) and
# from `_exit_after_drift_check()`, which `main()` calls from a `finally` --
# a single choke point every exit path passes through, so a future mode
# can't forget to wire this in.
_config_drift_report_done = False
_config_drift_ok_cached = True


def _config_drift_ok() -> bool:
    global _config_drift_report_done, _config_drift_ok_cached
    if not _config_drift_report_done:
        _config_drift_report_done = True
        print("\n== effective-config readback ==")
        if _effective_config_bad:
            for msg in _effective_config_bad:
                print(f"  FAIL: {msg}")
            print(f"  {len(_effective_config_bad)} mismatch(es) across {_effective_config_total} compiles")
            _config_drift_ok_cached = False
        elif not _effective_config_total:
            # An "OK ... on all 0 compiles" line reads as evidence when it is
            # the absence of evidence.
            print("  (nothing compiled — no effective config to confirm)")
        else:
            print(f"  OK: INFERENCE_PRECISION_HINT={REQUIRED_PRECISION_HINT} and "
                  f"DYNAMIC_QUANTIZATION_GROUP_SIZE={REQUIRED_DYN_QUANT_GROUP_SIZE} "
                  f"confirmed effective on all {_effective_config_total} compiles")
    return _config_drift_ok_cached


_kernel_scan_report_done = False
_kernel_scan_ok_cached = True


def _kernel_scan_ok() -> bool:
    """Spec Sec3.1's "no int8 / dyn-quant kernels" gate. Idempotent in the same
    way `_config_drift_ok` is, so validate mode can print it in its report
    position and `_exit_after_contract_checks` can still enforce it for
    `--export`/`--bench`/`--canary`, which never had it before."""
    global _kernel_scan_report_done, _kernel_scan_ok_cached
    if _kernel_scan_report_done:
        return _kernel_scan_ok_cached
    _kernel_scan_report_done = True
    print("\n== exec-graph kernels seen ==")
    if not _kernel_names_seen:
        print("  (nothing inspected — no graph was compiled)")
        return _kernel_scan_ok_cached
    for k in sorted(_kernel_names_seen):
        print(f"  {k}")
    bad = sorted(k for k in _kernel_names_seen if _looks_quantized(k))
    if bad:
        print(f"  FAIL: quantized kernel(s)/precision(s) present: {bad}")
        _kernel_scan_ok_cached = False
    else:
        print(f"  OK: no int8 / uint8 / dyn-quant kernels across "
              f"{_effective_config_total} compile(s)")
    return _kernel_scan_ok_cached


def _exit_after_contract_checks():
    """The single choke point: `main()` calls this from a `finally`, after
    EVERY mode (validate-ops, validate-graph, export, bench, canary). No-ops if
    nothing was compiled (e.g. `--help`); otherwise raises `SystemExit(1)` on
    effective-config drift or a quantized kernel regardless of what else
    passed, and never overrides a clean exit's own code."""
    if not _effective_config_total:
        return
    drift_ok = _config_drift_ok()
    kernels_ok = _kernel_scan_ok()
    if not (drift_ok and kernels_ok):
        raise SystemExit(1)


# Every exec-graph entry any compile in this process reported.
_kernel_names_seen: set = set()

# The quantization signal lives in `runtimePrecision` and `execType`, NOT in
# `layerType`: `layerType` is a structural op name (`FullyConnected`, `Const`,
# `Eltwise`, `Subgraph`, ...) and is byte-identical with dynamic quantization
# on and off, so a scan over it can never fire.
#
# Signed/sub-byte precisions are unambiguous — nothing in a clean f32 graph
# reports them. `u8` is NOT: a fused eltwise `Subgraph` legitimately reports
# `runtimePrecision=u8` for the BOOLEAN intermediates of `_bf16_roundtrip`'s
# comparison-based bisection, and every clean run of this harness prints
# exactly that. So `u8` only counts on an op that can actually be quantized.
_QUANT_PRECISIONS_ANY_OP = ("int8", "i8", "int4", "i4", "uint4", "u4")
_QUANT_PRECISIONS_COMPUTE_ONLY = ("uint8", "u8")
_QUANTIZABLE_OPS = ("fullyconnected", "matmul", "gemm", "convolution",
                    "deconvolution", "innerproduct")

_QUANT_TAG = "quantized:"


def _looks_quantized(entry: str) -> bool:
    return entry.startswith(_QUANT_TAG)


def _rt_str(node, key: str):
    try:
        v = node.get_rt_info()[key]
    except Exception:  # noqa: BLE001 — best-effort diagnostic
        return None
    s = v.astype(str) if hasattr(v, "astype") else str(v)
    return s or None


def _quant_hit(layer_type: str, rp: str, et: str) -> bool:
    fields = (rp, *et.split("_"))
    if any(p in fields for p in _QUANT_PRECISIONS_ANY_OP):
        return True
    if any(op in layer_type for op in _QUANTIZABLE_OPS) and any(
            p in fields for p in _QUANT_PRECISIONS_COMPUTE_ONLY):
        return True
    return "dynquant" in (rp + et).replace("_", "")


def _kernel_names(compiled) -> set:
    """Best-effort exec-graph inventory: the structural `layerType` (context
    for a human reading this tool's output) PLUS `runtimePrecision` and
    `execType`. A node whose precision says it is quantized also emits a
    `quantized:` entry, which is what `_kernel_scan_ok` fails on (spec
    Sec3.1). Note `execType` is empty on some CPU plugin builds — the check
    must not depend on it alone."""
    names = set()
    try:
        for node in compiled.get_runtime_model().get_ops():
            t = (_rt_str(node, "layerType") or node.get_type_name()).lower()
            names.add(t)
            rp = (_rt_str(node, "runtimePrecision") or "").lower()
            et = (_rt_str(node, "execType") or "").lower()
            if rp:
                names.add(f"runtimeprecision={rp}")
            if et:
                names.add(f"exectype={et}")
            if _quant_hit(t, rp, et):
                names.add(f"{_QUANT_TAG}{t} runtimePrecision={rp or '?'} execType={et or '?'}")
    except Exception as e:  # noqa: BLE001 — best-effort diagnostic
        names.add(f"<unavailable: {type(e).__name__}>")
    return names


# --------------------------------------------------------------------------
# --validate-ops
# --------------------------------------------------------------------------

def _linear_cases():
    cases = [{"seed": 0, "out_dim": REAL_Q_LORA, "in_dim": REAL_HIDDEN, "note": "real dims (wq_a: hidden->q_lora)"}]
    for seed in range(1, 5):
        r = np.random.default_rng(seed)
        cases.append({"seed": seed, "out_dim": int(r.integers(1, 96)), "in_dim": int(r.integers(1, 256)), "note": "random"})
    return cases


def _rmsnorm_cases():
    # x_scale ~0.7 gives mean(x^2) ~0.5 — eps (1e-5/1e-6) is 5 orders of
    # magnitude below that, so for THOSE cases the two epsilons landing on
    # the same bf16 output is expected, not a bug (bf16 has ~3 decimal
    # digits). The two-eps guard the brief asks for ("1e-5 vs 1e-6 must
    # differ") only has teeth where eps is comparable to mean(x^2) — the
    # near-zero-activation regime eps exists to protect in the first place —
    # so that's a dedicated `guard=True` case with a correspondingly tiny
    # x_scale, and only that case's divergence is required to hold.
    cases = [{"seed": 0, "dim": REAL_Q_LORA, "x_scale": 0.7, "guard": False, "note": "real dims (q_a_ln)"}]
    for seed in range(1, 4):
        r = np.random.default_rng(seed + 100)
        cases.append({"seed": seed, "dim": int(r.integers(1, 256)), "x_scale": 0.7, "guard": False, "note": "random"})
    cases.append({"seed": 4, "dim": 64, "x_scale": 2e-3, "guard": True, "note": "near-zero activation (two-eps guard)"})
    return cases


def _rope_cases():
    cases = [{"seed": 0, "rot_dims": REAL_QK_ROPE, "theta": REAL_ROPE_THETA, "pos": 10, "note": "real dims (qk_rope)"}]
    for seed in range(1, 5):
        r = np.random.default_rng(seed + 200)
        rd = int(r.integers(1, 32)) * 2  # must be even
        cases.append({
            "seed": seed, "rot_dims": rd,
            "theta": float(r.uniform(1e3, 1e7)),
            "pos": int(r.integers(0, 128)),
            "note": "random",
        })
    return cases


def validate_linear(core) -> bool:
    print("\n== linear (linear_bf16_w) ==")
    ok = True
    for case in _linear_cases():
        rng = np.random.default_rng(case["seed"])
        out_dim, in_dim = case["out_dim"], case["in_dim"]
        w_f32 = (rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.02)
        w_bits = (w_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)  # bf16-truncate to build the Rust-shape store
        x = rng.standard_normal((in_dim,)).astype(np.float32) * 0.5

        ref = ref_linear_bf16(x, w_bits)
        model = build_op_graph("linear", {"out_dim": out_dim, "in_dim": in_dim, "w_bits": w_bits})
        compiled = compile_checked(core, model, f"linear seed={case['seed']}")
        got = np.array(compiled.create_infer_request().infer({"x": x.reshape(1, in_dim)})[0]).reshape(-1)

        max_ulp, flip, worst, rb, gb = ulp_stats(ref, got)
        grid = on_grid(got)
        _kernel_names_seen.update(_kernel_names(compiled))
        passed = max_ulp <= 1 and flip <= 0.01 and grid
        ok &= passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] seed={case['seed']:2d} out={out_dim:5d} in={in_dim:5d} "
              f"({case['note']}) max_ulp_diff={max_ulp} flip_rate={flip:.4f} on_grid={grid}")
        if not passed:
            print(f"    worst idx={worst} ref=0x{rb[worst]:04x}({ref[worst]!r}) "
                  f"ov=0x{gb[worst]:04x}({got[worst]!r})")
    return ok


def validate_rmsnorm(core) -> bool:
    print("\n== rmsnorm ==")
    ok = True
    for case in _rmsnorm_cases():
        rng = np.random.default_rng(case["seed"] + 100)
        dim = case["dim"]
        w = rng.standard_normal((dim,)).astype(np.float32) * 0.1 + 1.0
        x = rng.standard_normal((dim,)).astype(np.float32) * case["x_scale"]

        model = build_op_graph("rmsnorm", {"dim": dim, "w": w})
        compiled = compile_checked(core, model, f"rmsnorm seed={case['seed']}")
        req = compiled.create_infer_request()
        _kernel_names_seen.update(_kernel_names(compiled))

        outs = {}
        for eps, tag in ((RMS_NORM_EPS, "1e-5"), (MLA_LATENT_EPS, "1e-6")):
            ref = ref_rmsnorm(x, w, eps)
            got = np.array(req.infer({"x": x.reshape(1, dim), "eps": np.float32(eps)})[0]).reshape(-1)
            max_ulp, flip, worst, rb, gb = ulp_stats(ref, got)
            grid = on_grid(got)
            outs[tag] = got
            passed = max_ulp <= 1 and flip <= 0.01 and grid
            ok &= passed
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] seed={case['seed']:2d} dim={dim:5d} eps={tag} ({case['note']}) "
                  f"max_ulp_diff={max_ulp} flip_rate={flip:.4f} on_grid={grid}")
            if not passed:
                print(f"    worst idx={worst} ref=0x{rb[worst]:04x}({ref[worst]!r}) "
                      f"ov=0x{gb[worst]:04x}({got[worst]!r})")

        diverges = not np.array_equal(bf16_bits(outs["1e-5"]), bf16_bits(outs["1e-6"]))
        if case["guard"]:
            ok &= diverges
            label = "DIFFER (ok)" if diverges else "IDENTICAL (FAIL — hardcoded eps?)"
        else:
            # Not a failure either way here — see _rmsnorm_cases' comment.
            label = "differ" if diverges else "identical (expected: eps << mean(x^2) at this scale)"
        print(f"    two-eps: 1e-5 vs 1e-6 {label}")
    return ok


def validate_rope(core) -> bool:
    print("\n== rope (apply_rope_row, interleaved) ==")
    ok = True
    for case in _rope_cases():
        rng = np.random.default_rng(case["seed"] + 200)
        rd, theta, pos = case["rot_dims"], case["theta"], case["pos"]
        x = rng.standard_normal((rd,)).astype(np.float32) * 0.5

        ref = ref_rope_interleaved(x, pos, theta, rd)
        model = build_op_graph("rope", {"rot_dims": rd, "theta": theta, "pos": pos})
        compiled = compile_checked(core, model, f"rope seed={case['seed']}")
        got = np.array(compiled.create_infer_request().infer({"x": x.reshape(1, rd)})[0]).reshape(-1)
        _kernel_names_seen.update(_kernel_names(compiled))

        max_ulp, flip, worst, rb, gb = ulp_stats(ref, got)
        grid = on_grid(got)
        passed = max_ulp <= 1 and flip <= 0.01 and grid
        ok &= passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] seed={case['seed']:2d} rot_dims={rd:3d} theta={theta:.3e} pos={pos:4d} "
              f"({case['note']}) max_ulp_diff={max_ulp} flip_rate={flip:.4f} on_grid={grid}")
        if not passed:
            print(f"    worst idx={worst} ref=0x{rb[worst]:04x}({ref[worst]!r}) "
                  f"ov=0x{gb[worst]:04x}({got[worst]!r})")
    return ok


def validate_bf16_roundtrip(core) -> bool:
    """The `_bf16_roundtrip` subgraph vs `bf16_round`, on the SAME 20 tie
    cases verified against `half::bf16::from_f32` in Rust.

    The previous version compared
    `bf16_bits(core_out)` — which itself calls `bf16_round(core_out)` before
    extracting bits — against the expected bf16 bits. Since `bf16_round` is
    idempotent, that comparison is TAUTOLOGICAL for any `core_out` close
    enough to re-round to the same bucket: an identity graph (the exact bug
    `_bf16_roundtrip` had under `INFERENCE_PRECISION_HINT=f32`) still "passes" because `bf16_round(xin)` re-derives the
    same expected value the test table already encodes. Two independent
    reviews caught this the same way: reverting `_bf16_roundtrip` to the
    broken Convert pair left this check GREEN.

    Fixed by comparing the RAW f32 bit pattern of `core_out` — no re-rounding
    — against `want << 16` (the bf16-rounded value widened to f32, i.e. top
    16 bits = the expected bf16 bits, low 16 bits = exactly zero). An
    identity/untouched output fails this by construction: for every
    non-already-on-grid tie case, `xin`'s raw bits differ from `want << 16`
    in the low 16 bits. Comparing as `uint32` (not `float32 ==`) also makes
    the two NaN tie cases correct for free — `_bf16_roundtrip` canonicalizes
    NaN to a specific bit pattern (`0x7FC00000`), and integer equality on
    that canonical pattern isn't subject to IEEE's `NaN != NaN`."""
    print("\n== bf16 Convert round-trip (OV vs numpy, tie cases) ==")
    xin = np.array([np.uint32(b) for b, _ in BF16_TIE_CASES], dtype=np.uint32).view(np.float32)
    model = _bf16_roundtrip_graph(len(BF16_TIE_CASES))
    compiled = compile_checked(core, model, "bf16_roundtrip")
    core_out = np.array(compiled.create_infer_request().infer({"x": xin})[0]).reshape(-1)
    ov_bits_raw = core_out.view(np.uint32)  # RAW bits -- NOT re-rounded via bf16_bits/bf16_round
    np_bits = bf16_bits(bf16_round(xin))
    mismatches = [
        (i, hex(bits), f"expect=0x{want:04x}", f"numpy=0x{np_bits[i]:04x}", f"ov_raw=0x{ov_bits_raw[i]:08x}")
        for i, (bits, want) in enumerate(BF16_TIE_CASES)
        if ov_bits_raw[i] != (np.uint32(want) << np.uint32(16)) or np_bits[i] != want
    ]
    if mismatches:
        print(f"  {len(mismatches)}/{len(BF16_TIE_CASES)} mismatches vs half::bf16::from_f32 ground truth:")
        for m in mismatches:
            print(f"    {m}")
    else:
        print(f"  all {len(BF16_TIE_CASES)} tie cases: numpy and OV round-trip both match "
              "half::bf16::from_f32 exactly (raw-bits check, not re-rounded)")
    return len(mismatches) == 0


# --------------------------------------------------------------------------
# Device canary.
#
# Everything above (`--validate-ops`, `--validate-graph`) compiles on
# `DEVICE="CPU"` — a deliberate choice for numerics validation (see `DEVICE`'s
# own comment: the graphs are device-agnostic), but
# the GPU plugin is a DIFFERENT compiler we cannot exercise here, with two
# named defeat vectors specific to it: the `_bf16_roundtrip` subgraph
# silently executing in f16 despite `INFERENCE_PRECISION_HINT=f32` (would
# corrupt the exponent-bisection constants), and its `Round` kernel's tie
# mode differing from CPU's verified `half_to_even` (would only show up at
# an exact tie, i.e. the f=.5 cases below — f=.25/.75 cases can't
# distinguish `half_to_even` from `half_away_from_zero`, only from
# truncation/identity). Property readback (`compile_checked`) is NOT
# sufficient on its own: on this stack a self-reported property has already
# been observed to lie (`query_model` claiming kernels present that exist on
# no device) — this canary asserts actual RAW OUTPUT BITS instead.
# --------------------------------------------------------------------------

def _bf16_ulp_at(x: float) -> float:
    """The bf16 ULP (spacing between adjacent bf16-representable values)
    at the magnitude of `x`. Pure python/`math.frexp` — this is host-side
    battery GENERATION, not part of the numerics under test, so it doesn't
    need to dodge the subnormal/overflow concerns `_bf16_roundtrip`'s
    in-graph bisection does."""
    _, e = math.frexp(abs(x))  # x = m * 2**e, 0.5 <= |m| < 1
    return 2.0 ** (e - 8)  # 7 explicit + 1 implicit mantissa bit = 8 significant bits


def _rnd1(x: float) -> float:
    return float(bf16_round(np.array([x], dtype=np.float32))[0])


# Bases deliberately NOT at a power-of-two magnitude boundary (±1.0, ±8.0,
# ±0.015625 were tried first and rejected: at an exact power of two, the
# NEXT bf16 grid point going toward zero crosses into a FINER-spaced
# exponent bucket, so `g_lo + _bf16_ulp_at(g_lo)` silently computes the
# wrong g_hi). Mix of even- and odd-bf16-mantissa `g_lo` (so the f=.5 tie
# cases exercise RNE rounding BOTH directions, not just "always down") and
# both signs (so truncation's round-toward-zero direction, which flips with
# sign, is exercised both ways too).
_CANARY_BASES = (1.5, 1.0234375, -1.5, -1.0234375,
                  12.0, 12.1875, -12.0, -12.1875,
                  100.0, -100.0, 0.0234375, -0.0234375)


def canary_cases() -> list:
    """Battery of `{x, rne, trunc, half_away, identity, label}` — the four
    hypotheses predict DIFFERENT raw bits at least at the f=.5 (tie) point,
    and RNE/half_away diverge from truncation/identity at f=.25/.75.
    `rne` (ground truth) is `bf16_round`, itself verified bit-exact against
    `half::bf16::from_f32` in Rust (the engine crate's `bf16_ties` test).
    """
    cases = []
    for base in _CANARY_BASES:
        g_lo = _rnd1(base)
        ulp = _bf16_ulp_at(g_lo)
        g_hi = _rnd1(g_lo + ulp)
        for frac, tag in ((0.25, "f=.25"), (0.5, "f=.5(tie)"), (0.75, "f=.75")):
            x = np.float32(g_lo + frac * ulp)
            rne = _rnd1(float(x))
            # Truncation rounds toward ZERO, so it lands on whichever of
            # g_lo/g_hi has the smaller magnitude — and since g_hi = g_lo +
            # ulp, that is g_lo for a positive base but g_hi for a NEGATIVE
            # one. The direction flips with the sign at every frac, tie
            # included. (Predicting g_lo unconditionally mislabels every
            # negative-base case, so a real device truncation would report
            # "unknown" instead of naming itself.)
            trunc = g_lo if base > 0 else g_hi
            # half-away-from-zero is plain nearest away from the tie; only the
            # exact tie differs from RNE, breaking toward larger magnitude
            # (g_hi for base>0; g_lo, the more-negative one, for base<0).
            if frac < 0.5:
                half_away = g_lo
            elif frac > 0.5:
                half_away = g_hi
            else:
                half_away = g_hi if base > 0 else g_lo
            cases.append({
                "label": f"base={base:g} {tag}",
                "x": x,
                "rne": np.float32(rne),
                "trunc": np.float32(trunc),
                "half_away": np.float32(half_away),
                "identity": x,
            })
    return cases


def run_canary(core, device: str) -> bool:
    """Compiles the SHIPPED `_bf16_roundtrip` subgraph (the same one every
    op/layer graph in this file uses) on the ACTUAL target `device` with the
    ACTUAL compile config (`compile_checked`, so a config drift on this
    device is caught by the same choke point as everything else), feeds
    `canary_cases()` plus the 20 `BF16_TIE_CASES`, and asserts RAW f32 bit
    patterns against the RNE-correct expectation exactly. On a mismatch,
    reports which of the OTHER three hypotheses (if any) the observed output
    matches, so a real GPU failure is diagnosable without re-deriving the
    battery by hand."""
    print(f"\n== device canary ({device}) ==")
    battery = canary_cases()
    tie_x = np.array([np.uint32(b) for b, _ in BF16_TIE_CASES], dtype=np.uint32).view(np.float32)
    xin = np.concatenate([np.array([c["x"] for c in battery], dtype=np.float32), tie_x])
    model = _bf16_roundtrip_graph(len(xin))
    compiled = compile_checked(core, model, f"canary({device})", device=device)
    out = np.array(compiled.create_infer_request().infer({"x": xin})[0]).reshape(-1)
    out_bits = out.view(np.uint32)

    mismatches = []
    for i, c in enumerate(battery):
        want_bits = np.float32(c["rne"]).view(np.uint32)
        if out_bits[i] != want_bits:
            hyps = {k: np.float32(c[k]).view(np.uint32) for k in ("trunc", "half_away", "identity")}
            matched = next((k for k, v in hyps.items() if v == out_bits[i]), "unknown")
            mismatches.append(f"    [{c['label']}] x={c['x']!r} want(RNE)=0x{want_bits:08x} "
                               f"got=0x{out_bits[i]:08x} (matches: {matched})")
    n_battery = len(battery)
    for j, (bits, want16) in enumerate(BF16_TIE_CASES):
        i = n_battery + j
        want_bits = np.uint32(want16) << np.uint32(16)
        if out_bits[i] != want_bits:
            mismatches.append(f"    [tie_case idx={j}] x_bits=0x{bits:08x} want=0x{want_bits:08x} "
                               f"got=0x{out_bits[i]:08x}")

    if mismatches:
        print(f"  {len(mismatches)}/{len(xin)} mismatches (battery={n_battery}, tie_cases={len(BF16_TIE_CASES)}):")
        for m in mismatches:
            print(m)
    else:
        print(f"  all {len(xin)} cases ({n_battery} battery + {len(BF16_TIE_CASES)} tie) "
              f"exact-match RNE on {device}")
    return len(mismatches) == 0


def validate_rope_table_exact(rope_dump_path: str) -> bool:
    """The ONE exact gate. Diffs the exporter's precomputed cos/sin
    table against a bit-for-bit dump of Rust's `Freqs.data` at GLM-5.2's real
    rope dims. `numpy`'s `cos`/`powf` need not agree with Rust's libm — a
    silently-different table is a wrong-basis bug that short-prompt parity
    would not catch, so this must be exact, not ULP-tolerant."""
    print("\n== rope table EXACT gate ==")
    if not os.path.exists(rope_dump_path):
        print(f"  SKIPPED: no dump at {rope_dump_path}")
        print("  regenerate with: cargo test -p cascadia-engine-sparse-moe --lib "
              "dump_real_dims_freqs -- --nocapture")
        return False

    dim, seqlen, theta = 64, 16, REAL_ROPE_THETA
    rust_table = np.fromfile(rope_dump_path, dtype="<f4")
    py_table = precompute_freqs_table(dim, seqlen, theta)
    if rust_table.shape != py_table.shape:
        print(f"  SHAPE MISMATCH: rust={rust_table.shape} python={py_table.shape}")
        return False

    exact = np.array_equal(rust_table.view(np.uint32), py_table.view(np.uint32))
    if exact:
        print(f"  EXACT: {py_table.size} f32 values bit-for-bit identical (dim={dim} seqlen={seqlen} theta={theta:g})")
        return True

    diff_idx = np.flatnonzero(rust_table.view(np.uint32) != py_table.view(np.uint32))
    ulp = np.abs(rust_table[diff_idx] - py_table[diff_idx])
    print(f"  NOT EXACT: {diff_idx.size}/{py_table.size} values differ "
          f"(max abs diff={ulp.max():.3e}) — numpy transcendentals vs Rust libm, as anticipated")
    for i in diff_idx[:5]:
        print(f"    idx={i} rust={rust_table[i]!r} (0x{rust_table.view(np.uint32)[i]:08x}) "
              f"python={py_table[i]!r} (0x{py_table.view(np.uint32)[i]:08x})")
    return False


# ============================================================================
# Full per-layer attention graph + export pipeline + stamp.
#
# Weight source: `<model>/shells/layer_NN.safetensors`, tensor names written by
# `tools/export_glm5.py`'s `attn_shell()` (--tiny) / `export_real`'s per-layer
# `shell` dict — `input_layernorm.weight`, `self_attn.{wq_a,q_a_layernorm,
# wq_b,wkv_a,kv_a_layernorm,wkv_b,o_proj}.weight` — read the SAME way
# `glm/loader.rs::load_layer` reads them (`g`/`gb` there), via a small
# safetensors reader mirroring `dsv4/st.rs::StFile` (see `SafeTensorsFile`
# below) rather than a torch dependency.
# ============================================================================

MAX_BATCH_COUNT = 256  # dist.rs MAX_BATCH_COUNT (spec Sec2/Sec14 windowing
# constant — NOT a manifest dim). If this wire constant ever changes, the
# window math here (P_max = W - MAX_BATCH_COUNT) must be revisited (spec
# Sec14's own cross-reference note).

NO_INDEXER_W = 2048  # spec Sec4: dense-exact domain cap when the manifest has
# no DSA indexer (index_n_heads == 0) — never unbounded, never 0.

# Bump whenever the exported graph's numerics change. The engine compares this
# against its own compiled-in EXPECTED_EXPORTER_VERSION (ov_attn.rs) and
# refuses an IR set from a generation it makes no claim about.
# "2": bf16 round-trip + layer-graph numerics corrections since "1".
EXPORTER_VERSION = "2"

# Additive-mask "invalid" sentinel. Finite (not literal -inf): avoids any
# inf-vs-inf edge case inside the fused SDPA kernel while sitting far enough
# below realistic score magnitudes that its softmax weight rounds to exactly
# 0. `_MASK_VALID_THRESHOLD` is what in-graph/host code uses to classify a
# mask entry as "valid" vs "masked" when deriving past_len from the mask.
MASK_NEG = np.float32(-1e30)
_MASK_VALID_THRESHOLD = np.float32(-1e29)


# ----------------------------------------------------------------------------
# Minimal safetensors reader — mirrors `dsv4/st.rs::StFile` (whole-file read,
# header parse, dtype-aware decode) rather than depending on torch/ml_dtypes
# just to get at raw bf16 bits (numpy has no native bf16 dtype). Shell files
# are documented there as "a few hundred MB at most", so a full read is fine.
# ----------------------------------------------------------------------------
class SafeTensorsFile:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            self.header = json.loads(f.read(hlen))
            self.data_start = 8 + hlen
            f.seek(0)
            self.data = f.read()

    def _bytes(self, name: str):
        if name not in self.header:
            raise KeyError(f"tensor {name!r} not found in safetensors file")
        info = self.header[name]
        s, e = info["data_offsets"]
        return info, self.data[self.data_start + s:self.data_start + e]

    def bf16_bits(self, name: str) -> np.ndarray:
        """Raw bf16 bit pattern (u16), shape as declared. The tensor must be
        BF16-dtype in the file — true of every attention projection weight
        `export_glm5.py` writes (this module's docstring above)."""
        info, b = self._bytes(name)
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {info['dtype']!r}")
        return np.frombuffer(b, dtype="<u2").reshape(info["shape"]).copy()

    def f32(self, name: str) -> np.ndarray:
        """f32-widened tensor (accepts F32 or BF16), matching `StFile::f32`."""
        info, b = self._bytes(name)
        shape = info["shape"]
        if info["dtype"] == "F32":
            return np.frombuffer(b, dtype="<f4").reshape(shape).astype(np.float32).copy()
        if info["dtype"] == "BF16":
            bits = np.frombuffer(b, dtype="<u2").reshape(shape)
            return ((bits.astype(np.uint32) << np.uint32(16)).view(np.float32)).copy()
        raise ValueError(f"{name}: unsupported dtype {info['dtype']!r}")


def read_manifest(model_dir: str) -> dict:
    path = os.path.join(model_dir, "manifest.json")
    with open(path) as f:
        m = json.load(f)
    if m.get("arch") != "glm5":
        raise SystemExit(f"{path}: arch={m.get('arch')!r}, expected 'glm5'")
    return m


def derive_w(manifest: dict) -> int:
    """`W` per spec Sec4 — mirrors `glm/loader.rs::load_layer`'s indexer
    clamp, EXCEPT it fails loudly instead of the loader's `usize::MAX` when an
    indexer is present with `index_topk == 0`: the loader's clamp is a
    decode-time convenience (unbounded causal attention has no static shape
    problem there) that a compiled graph's fixed `P_max` cannot inherit."""
    index_n_heads = int(manifest.get("index_n_heads", 0))
    index_topk = int(manifest.get("index_topk", 0))
    if index_n_heads > 0:
        if index_topk == 0:
            raise SystemExit(
                f"manifest has an indexer (index_n_heads={index_n_heads}) but "
                "index_topk == 0 — the loader clamps this to unbounded "
                "(usize::MAX) at decode time, which has no static graph "
                "shape; export cannot proceed. Re-export with a real "
                "index_topk, or without an indexer."
            )
        return index_topk
    return NO_INDEXER_W


# ----------------------------------------------------------------------------
# Rope table sourcing (Step 0): a bit-for-bit Rust dump, never recomputed in
# Python. Extends `dsv4/rope.rs::freqs_dump::dump_real_dims_freqs`
# (now parameterized via env vars, defaults unchanged) rather than inventing a
# second mechanism.
# ----------------------------------------------------------------------------
def _rope_regen_cmd(rot_dims: int, seqlen: int, theta: float, dump_path: str) -> str:
    return (
        f"GLM5_ROPE_DUMP_ROT_DIMS={rot_dims} GLM5_ROPE_DUMP_SEQLEN={seqlen} "
        f"GLM5_ROPE_DUMP_THETA={theta!r} GLM5_ROPE_FREQS_DUMP={dump_path} "
        "cargo test -p cascadia-engine-sparse-moe --lib dump_real_dims_freqs -- --nocapture"
    )


def load_rope_table(dump_path: str, rot_dims: int, seqlen: int, theta: float, *, hard_fail: bool):
    """Loads the `[seqlen, rot_dims//2]` (cos, sin) table from a Rust `Freqs`
    dump, verifying its `.meta.json` sidecar's dims against what THIS
    export/test actually needs — never recomputing the table in Python (a
    wrong-basis table is a decode-breaking bug short-prompt parity would not
    catch).

    `hard_fail=True` (export mode): missing/mismatched dump raises
    `SystemExit`. `hard_fail=False` (`--validate-graph`): a MISSING dump
    returns `None` (caller treats it as a loud, non-fatal SKIPPED — the SAME
    discipline `--validate-ops` established for its exact gate); a PRESENT-but-
    mismatched dump still raises regardless of `hard_fail`.
    """
    half = rot_dims // 2
    regen = _rope_regen_cmd(rot_dims, seqlen, theta, dump_path)
    if not os.path.exists(dump_path):
        if hard_fail:
            raise SystemExit(f"rope table dump missing at {dump_path}\n  regenerate with: {regen}")
        print(f"  SKIPPED: no rope table dump at {dump_path}")
        print(f"  regenerate with: {regen}")
        return None

    meta_path = dump_path + ".meta.json"
    if not os.path.exists(meta_path):
        raise SystemExit(f"rope table dump at {dump_path} has no .meta.json sidecar "
                          f"(stale/hand-made dump?)\n  regenerate with: {regen}")
    with open(meta_path) as f:
        meta = json.load(f)
    if (int(meta["rot_dims"]) != rot_dims or int(meta["seqlen"]) != seqlen
            or not math.isclose(float(meta["theta"]), float(theta), rel_tol=1e-6)):
        raise SystemExit(
            f"rope table dump at {dump_path} was built for rot_dims={meta['rot_dims']} "
            f"seqlen={meta['seqlen']} theta={meta['theta']}, but this export/test needs "
            f"rot_dims={rot_dims} seqlen={seqlen} theta={theta}.\n  regenerate with: {regen}"
        )
    # Provenance, not just dims: the sidecar carries the sha256 of the bytes
    # the Rust dump test wrote. Without this check a table Python generated
    # itself passes, and numpy's cos/sin land ~1 ULP off Rust's libm on a
    # minority of entries — a wrong-basis table survives short-prompt parity
    # and breaks decode.
    want_sha = meta.get("sha256")
    if not want_sha:
        raise SystemExit(
            f"rope table dump at {dump_path}: sidecar has no sha256 (written by an "
            f"older dump test, or hand-made)\n  regenerate with: {regen}"
        )
    with open(dump_path, "rb") as f:
        got_sha = hashlib.sha256(f.read()).hexdigest()
    if got_sha != want_sha:
        raise SystemExit(
            f"rope table dump at {dump_path}: sha256 {got_sha} does not match the "
            f"sidecar's {want_sha} — the table was modified or regenerated outside "
            f"the Rust dump test\n  regenerate with: {regen}"
        )

    raw = np.fromfile(dump_path, dtype="<f4")
    if raw.size != seqlen * half * 2:
        raise SystemExit(f"rope table dump at {dump_path}: {raw.size} f32 values, "
                          f"expected {seqlen * half * 2} (seqlen={seqlen} half={half})")
    raw = raw.reshape(seqlen, half, 2)
    return raw[:, :, 0].copy(), raw[:, :, 1].copy()


def _scale_for_qk(qk_head: int) -> np.float32:
    """`(qk_head as f32).powf(-0.5)` — `attn.rs`'s `scale` field, computed via
    the SAME system-libm ctypes call `_rope_cos_sin` uses above, for
    bit-exactness with Rust's `f32::powf` (numpy's vectorized `power` can land
    1 ULP off)."""
    base = np.float32(qk_head)
    if _libm is not None:
        return np.float32(_libm.powf(ctypes.c_float(float(base)), ctypes.c_float(-0.5)))
    return np.power(base, np.float32(-0.5), dtype=np.float32)


def load_layer_weights(model_dir: str, li: int, manifest: dict) -> dict:
    """One layer's attention weights from `<model_dir>/shells/layer_NN.safetensors`
    — see this section's header comment for the exact source tensor names."""
    hidden = manifest["hidden_size"]
    h = manifest["num_attention_heads"]
    nope = manifest["qk_nope_head_dim"]
    rope_d = manifest["qk_rope_head_dim"]
    vh = manifest["v_head_dim"]
    kvl = manifest["kv_lora_rank"]
    ql = manifest["q_lora_rank"]
    qk = nope + rope_d

    st = SafeTensorsFile(os.path.join(model_dir, "shells", f"layer_{li:02d}.safetensors"))
    w = {
        "in_ln": st.f32("input_layernorm.weight"),
        "wq_a_bits": st.bf16_bits("self_attn.wq_a.weight"),
        "q_a_ln": st.f32("self_attn.q_a_layernorm.weight"),
        "wq_b_bits": st.bf16_bits("self_attn.wq_b.weight"),
        "wkv_a_bits": st.bf16_bits("self_attn.wkv_a.weight"),
        "kv_a_ln": st.f32("self_attn.kv_a_layernorm.weight"),
        "wkv_b_bits": st.bf16_bits("self_attn.wkv_b.weight"),
        "wo_bits": st.bf16_bits("self_attn.o_proj.weight"),
    }
    # Mirror AttentionLayer::new's shape asserts — catch manifest/shell drift
    # loudly instead of a confusing shape error deep in graph construction.
    checks = [
        ("in_ln", (hidden,)), ("wq_a_bits", (ql, hidden)), ("q_a_ln", (ql,)),
        ("wq_b_bits", (h * qk, ql)), ("wkv_a_bits", (kvl + rope_d, hidden)),
        ("kv_a_ln", (kvl,)), ("wkv_b_bits", (h * (nope + vh), kvl)),
        ("wo_bits", (hidden, h * vh)),
    ]
    for name, shape in checks:
        got = tuple(int(d) for d in w[name].shape)
        if got != shape:
            raise SystemExit(f"layer {li}: {name} shape {got} != expected {shape} "
                              "(manifest/shell drift?)")
    return w


def layer_dims(manifest: dict, w: int, rows: int) -> dict:
    return {
        "hidden": manifest["hidden_size"],
        "h": manifest["num_attention_heads"],
        "qk_nope": manifest["qk_nope_head_dim"],
        "qk_rope": manifest["qk_rope_head_dim"],
        "v_head": manifest["v_head_dim"],
        "kv_lora": manifest["kv_lora_rank"],
        "q_lora": manifest["q_lora_rank"],
        "rms_norm_eps": manifest["rms_norm_eps"],
        "rope_theta": manifest["rope_theta"],
        "rows": rows,
        "p_max": w - rows,
    }


# --------------------------------------------------------------------------
# ref_attention — non-absorbed MLA prefill reference (spec Sec3 dataflow)
# --------------------------------------------------------------------------

def ref_attention(weights: dict, x_rows: np.ndarray, past_lc: np.ndarray,
                   past_rc: np.ndarray, dims: dict):
    """Full numpy MLA-attention reference, built from the per-op refs above
    (`ref_rmsnorm`/`ref_linear_bf16`/`ref_rope_interleaved`) called per row so
    it is literally THEIR math under test, not a batched re-derivation.
    `past_lc`/`past_rc` are the TRUE (unpadded) past — `[P, kv_lora]` /
    `[P, qk_rope]` for whatever `P` the caller has; padding to a fixed graph
    shape is a `--validate-graph`-only concern layered on top of this
    function (this is also why the tiny-model parity harness reuses it with no window
    concept at all).

    Returns `(attn_out[R,hidden], lc[R,kv_lora], rc[R,qk_rope])`,
    `R = x_rows.shape[0]`. `scores`/softmax/context stay f32 throughout (no
    bf16 rounding) — spec Sec3.1's "NOT inside the SDPA core", matching
    `attn.rs::forward_token`'s f32 absorb core (`smax`/`denom`/`clat` are all
    `f32` there) exactly, including the per-row sequential accumulation.
    """
    R = x_rows.shape[0]
    P = past_lc.shape[0]
    hidden, h = dims["hidden"], dims["h"]
    nope, rope_d, vh = dims["qk_nope"], dims["qk_rope"], dims["v_head"]
    kvl = dims["kv_lora"]
    qk = nope + rope_d
    theta = dims["rope_theta"]
    scale = _scale_for_qk(qk)

    lc_out = np.zeros((R, kvl), np.float32)
    rc_out = np.zeros((R, rope_d), np.float32)
    q_all = np.zeros((R, h, qk), np.float32)
    for r in range(R):
        pos = P + r
        nrm = ref_rmsnorm(x_rows[r], weights["in_ln"], dims["rms_norm_eps"])
        qr = ref_linear_bf16(nrm, weights["wq_a_bits"])
        qr = ref_rmsnorm(qr, weights["q_a_ln"], MLA_LATENT_EPS)
        q_flat = ref_linear_bf16(qr, weights["wq_b_bits"]).reshape(h, qk)
        for hi in range(h):
            q_all[r, hi] = ref_rope_interleaved(q_flat[hi], pos, theta, rope_d)

        comp = ref_linear_bf16(nrm, weights["wkv_a_bits"])
        latent, kpe = comp[:kvl], comp[kvl:]
        lc_out[r] = ref_rmsnorm(latent, weights["kv_a_ln"], MLA_LATENT_EPS)
        rc_out[r] = ref_rope_interleaved(kpe, pos, theta, rope_d)

    full_lc = np.concatenate([past_lc.astype(np.float32), lc_out], axis=0)  # [T,kvl]
    full_rc = np.concatenate([past_rc.astype(np.float32), rc_out], axis=0)  # [T,rope_d]
    T = P + R

    ctx = np.zeros((R, h, vh), np.float32)
    wkv_b_bits = weights["wkv_b_bits"]
    for hi in range(h):
        rbase = hi * (nope + vh)
        w_uk = ((wkv_b_bits[rbase:rbase + nope].astype(np.uint32) << np.uint32(16))
                .view(np.float32))
        w_uv = ((wkv_b_bits[rbase + nope:rbase + nope + vh].astype(np.uint32) << np.uint32(16))
                .view(np.float32))
        K_nope = (full_lc @ w_uk.T).astype(np.float32)          # [T,nope]  f32 core, no bf16 round
        V_h = (full_lc @ w_uv.T).astype(np.float32)              # [T,vh]
        K_h = np.concatenate([K_nope, full_rc], axis=-1)         # [T,qk]
        Qh = q_all[:, hi, :]                                     # [R,qk]
        scores = ((Qh @ K_h.T) * scale).astype(np.float32)       # [R,T]
        for r in range(R):
            cut = P + r + 1  # causal: row r (absolute pos P+r) sees columns [0, P+r]
            if cut < T:
                scores[r, cut:] = -np.inf
        smax = scores.max(axis=-1, keepdims=True).astype(np.float32)
        expv = np.exp((scores - smax).astype(np.float32)).astype(np.float32)
        denom = expv.sum(axis=-1, keepdims=True).astype(np.float32)
        p = (expv / denom).astype(np.float32)
        ctx[:, hi, :] = (p @ V_h).astype(np.float32)

    ctx_flat = ctx.reshape(R, h * vh)
    attn_out = np.zeros((R, hidden), np.float32)
    for r in range(R):
        attn_out[r] = ref_linear_bf16(ctx_flat[r], weights["wo_bits"])
    return attn_out, lc_out, rc_out


# --------------------------------------------------------------------------
# build_layer_graph — the OV attention IR (spec Sec3)
# --------------------------------------------------------------------------

def _i64(*vals) -> "ov.Node":
    return ops.constant(np.array(vals, dtype=np.int64))


def _i64_scalar(v: int) -> "ov.Node":
    return ops.constant(np.array(v, dtype=np.int64))


def build_layer_graph(weights: dict, dims: dict) -> Model:
    """One MLA-attention OV graph for one transformer layer — the
    NON-ABSORBED prefill form (spec Sec3): K/V are expanded per head from the
    `kv_lora` latent via `wkv_b`'s per-head row slices, rather than
    `attn.rs::forward_token`'s absorbed DECODE form (which folds the
    up-projection into the query/output and never materializes K/V) —
    mathematically equivalent by associativity of matmul (`q_nope·(W_UK·Lc) ==
    (q_nope·W_UK)·Lc`), but `scaled_dot_product_attention` needs literal K/V.

    ALL dims are read from `dims` (no literals): `rows`, `p_max`, `hidden`,
    `h`, `qk_nope`, `qk_rope`, `v_head`, `kv_lora`, `q_lora`, `rms_norm_eps`,
    `rope_theta`. `weights` carries the per-layer bf16-bit projection tensors
    (`load_layer_weights`'s output, or a synthetic dict for `--validate-graph`)
    PLUS the Rust-dump-sourced `rope_cos`/`rope_sin` tables
    (`[p_max+rows, qk_rope//2]` each, Step 0).

    Inputs: `x[rows,hidden]`, `past_lc[p_max,kv_lora]`, `past_rc[p_max,qk_rope]`,
    `mask[rows,p_max+rows]` (additive f32) — EXACTLY spec Sec3's four tensors;
    no fifth "past_len" input. The window's true past length is derived
    IN-GRAPH from the mask itself (spec Sec3.2: "the true past length enters
    through the mask input... not through the graph shape") via a
    ReduceSum-over-valid-columns on row 0's past segment (row 0 is always a
    real row — windows are trailing-zero-padded, spec Sec3), then used to
    `Gather` the matching `[rows, half]` slice of the rope table. This
    derivation is load-bearing, not cosmetic: row i's true rope position is
    `past_len + i`, and RoPE's relative-position identity
    (`dot(rope(q,pi),rope(k,pj))` is a function of `pj-pi` alone) gives no
    freedom to substitute a different, compile-time-fixed `pi` for q/rc's
    rotation — `past_rc`'s cached rows are already rotated at their TRUE
    absolute positions (committed by a past window) and can't be cheaply
    re-rotated in-graph, so matching them requires knowing the actual
    `past_len`.

    Outputs: `attn_out[rows,hidden]`, `lc[rows,kv_lora]`, `rc[rows,qk_rope]`.

    Op set used: `ops.scaled_dot_product_attention` (fused), NOT explicit
    MatMul+softmax — verified during development against a numpy reference
    (max abs diff ~1e-7, expected float reduction-order noise) with an
    explicit additive mask + explicit scale, per spec Sec13's fallback
    guidance. The one wrinkle: SDPA requires rank >= 3 (`Query input rank
    length must be at least 3 or more`), so each head's Q/K/V/mask gets a
    leading size-1 dim added before the call and squeezed after — no
    multi-head batching, one SDPA op per head — which is what makes the
    exported graph's node count scale with head count.
    """
    rows, p_max = dims["rows"], dims["p_max"]
    hidden, h = dims["hidden"], dims["h"]
    nope, rope_d, vh = dims["qk_nope"], dims["qk_rope"], dims["v_head"]
    kvl, qlora = dims["kv_lora"], dims["q_lora"]
    qk = nope + rope_d
    W = p_max + rows
    eps_in_ln = np.float32(dims["rms_norm_eps"])
    eps_latent = np.float32(MLA_LATENT_EPS)
    scale = _scale_for_qk(qk)

    x = ops.parameter(PartialShape([rows, hidden]), Type.f32, name="x")
    x.get_output_tensor(0).set_names({"x"})
    past_lc = ops.parameter(PartialShape([p_max, kvl]), Type.f32, name="past_lc")
    past_lc.get_output_tensor(0).set_names({"past_lc"})
    past_rc = ops.parameter(PartialShape([p_max, rope_d]), Type.f32, name="past_rc")
    past_rc.get_output_tensor(0).set_names({"past_rc"})
    mask = ops.parameter(PartialShape([rows, W]), Type.f32, name="mask")
    mask.get_output_tensor(0).set_names({"mask"})

    def const_bf16(bits, shape):
        return _raw_const(Type.bf16, list(shape), np.ascontiguousarray(bits).tobytes())

    def linear_bf16(inp, w_bits, out_dim, in_dim):
        w_const = const_bf16(w_bits, [out_dim, in_dim])
        y = ops.matmul(inp, ops.convert(w_const, Type.f32), False, True)
        return _bf16_roundtrip(y)

    def rmsnorm_bf16(inp, w_f32, eps_val):
        w_const = ops.constant(np.asarray(w_f32, np.float32))
        ms = ops.reduce_mean(ops.multiply(inp, inp), _i64(1), True)
        denom = ops.sqrt(ops.add(ms, ops.constant(np.float32(eps_val))))
        r = ops.divide(ops.constant(np.float32(1.0)), denom)
        y = ops.multiply(ops.multiply(inp, r), w_const)
        return _bf16_roundtrip(y)

    # past_len, derived from the mask (see docstring), and the rope-table
    # gather shared by q-rope and rc-rope for this window's new rows.
    row0_past = ops.slice(mask, _i64(0, 0), _i64(1, p_max), _i64(1, 1), _i64(0, 1))
    valid = ops.greater(row0_past, ops.constant(_MASK_VALID_THRESHOLD))
    past_len = ops.reduce_sum(ops.convert(valid, Type.i64), _i64(0, 1), False)
    idx = ops.range(past_len, ops.add(past_len, _i64_scalar(rows)), _i64_scalar(1), output_type="i64")
    cos_table = ops.constant(np.asarray(weights["rope_cos"], np.float32))
    sin_table = ops.constant(np.asarray(weights["rope_sin"], np.float32))
    c_rows = ops.gather(cos_table, idx, _i64_scalar(0))  # [rows, half]
    s_rows = ops.gather(sin_table, idx, _i64_scalar(0))

    def apply_rope(vec, dim_total):
        """Rotate the LAST `rope_d` dims of `vec[rows, dim_total]` as
        adjacent (even, odd) complex pairs — same slice/concat/reshape
        pattern as `build_op_graph('rope', ...)`, generalized to a
        batch of rows sharing one gathered `[rows, half]` cos/sin table."""
        start = dim_total - rope_d
        a = ops.slice(vec, _i64(0, start), _i64(rows, dim_total), _i64(1, 2), _i64(0, 1))
        b = ops.slice(vec, _i64(0, start + 1), _i64(rows, dim_total), _i64(1, 2), _i64(0, 1))
        re = ops.subtract(ops.multiply(a, c_rows), ops.multiply(b, s_rows))
        im = ops.add(ops.multiply(a, s_rows), ops.multiply(b, c_rows))
        re_u = ops.unsqueeze(re, _i64(2))
        im_u = ops.unsqueeze(im, _i64(2))
        inter = ops.concat([re_u, im_u], axis=2)
        rotated = ops.reshape(inter, _i64(rows, rope_d), False)
        if start == 0:
            return rotated
        head = ops.slice(vec, _i64(0, 0), _i64(rows, start), _i64(1, 1), _i64(0, 1))
        return ops.concat([head, rotated], axis=1)

    nrm = rmsnorm_bf16(x, weights["in_ln"], eps_in_ln)                             # [rows,hidden]

    qr = linear_bf16(nrm, weights["wq_a_bits"], qlora, hidden)                     # [rows,qlora]
    qr = rmsnorm_bf16(qr, weights["q_a_ln"], eps_latent)                           # [rows,qlora]
    q_flat = linear_bf16(qr, weights["wq_b_bits"], h * qk, qlora)                  # [rows,h*qk]

    comp = linear_bf16(nrm, weights["wkv_a_bits"], kvl + rope_d, hidden)           # [rows,kvl+rope_d]
    latent = ops.slice(comp, _i64(0, 0), _i64(rows, kvl), _i64(1, 1), _i64(0, 1))
    kpe = ops.slice(comp, _i64(0, kvl), _i64(rows, kvl + rope_d), _i64(1, 1), _i64(0, 1))
    lc = rmsnorm_bf16(latent, weights["kv_a_ln"], eps_latent)                       # OUTPUT lc [rows,kvl]
    rc = _bf16_roundtrip(apply_rope(kpe, rope_d))                                   # OUTPUT rc [rows,rope_d]

    q_heads = []
    for hi in range(h):
        qh = ops.slice(q_flat, _i64(0, hi * qk), _i64(rows, (hi + 1) * qk), _i64(1, 1), _i64(0, 1))
        qh = _bf16_roundtrip(apply_rope(qh, qk))
        q_heads.append(qh)

    full_lc = ops.concat([past_lc, lc], axis=0)   # [W,kvl]
    full_rc = ops.concat([past_rc, rc], axis=0)   # [W,rope_d]

    # ONE shared bf16 constant + ONE f32 convert for wkv_b — per-head slices
    # below are views into it, so the per-head loop does NOT duplicate weight
    # bytes into separate Constant nodes (the same "no duplicated weights"
    # discipline spec Sec3.2 mandates at the bucket level, applied here too).
    wkv_b_const = const_bf16(weights["wkv_b_bits"], [h * (nope + vh), kvl])
    wkv_b_f32 = ops.convert(wkv_b_const, Type.f32)

    scale_const = ops.constant(np.float32(scale))
    m3 = ops.unsqueeze(mask, _i64(0))  # [1,rows,W], shared across heads

    ctx_heads = []
    for hi in range(h):
        rbase = hi * (nope + vh)
        w_uk = ops.slice(wkv_b_f32, _i64(rbase, 0), _i64(rbase + nope, kvl), _i64(1, 1), _i64(0, 1))
        w_uv = ops.slice(wkv_b_f32, _i64(rbase + nope, 0), _i64(rbase + nope + vh, kvl), _i64(1, 1), _i64(0, 1))
        K_nope = ops.matmul(full_lc, w_uk, False, True)   # [W,nope]
        V_h = ops.matmul(full_lc, w_uv, False, True)       # [W,vh]
        K_h = ops.concat([K_nope, full_rc], axis=1)         # [W,qk]

        q3 = ops.unsqueeze(q_heads[hi], _i64(0))
        k3 = ops.unsqueeze(K_h, _i64(0))
        v3 = ops.unsqueeze(V_h, _i64(0))
        ctx3 = ops.scaled_dot_product_attention(q3, k3, v3, m3, scale_const, False)
        ctx_heads.append(ops.squeeze(ctx3, _i64(0)))

    ctx = ops.concat(ctx_heads, axis=1)                        # [rows,h*vh]
    attn_out = linear_bf16(ctx, weights["wo_bits"], hidden, h * vh)

    m = Model([attn_out, lc, rc], [x, past_lc, past_rc, mask], "glm5_attn_layer")
    m.outputs[0].tensor.set_names({"attn_out"})
    m.outputs[1].tensor.set_names({"lc"})
    m.outputs[2].tensor.set_names({"rc"})
    return m


def build_window_mask(rows: int, p_max: int, past_len: int) -> np.ndarray:
    """Host-side additive mask builder matching spec Sec3.1's convention: real
    past occupies columns `[0,past_len)`, padding `[past_len,p_max)`; window
    columns `[p_max,p_max+rows)` are causal (row i sees window columns
    `p_max..p_max+i]`). Applied uniformly to ALL `rows` (real and padding) —
    a padding row still sees its own diagonal (never fully masked, per spec
    Sec3.1's NaN-avoidance note), since its output is discarded by the caller
    regardless."""
    W = p_max + rows
    m = np.full((rows, W), MASK_NEG, dtype=np.float32)
    m[:, :past_len] = 0.0
    for i in range(rows):
        m[i, p_max:p_max + i + 1] = 0.0
    return m


def expected_bf16_layer_bytes(dims: dict) -> dict:
    """Expected `.bin` size for one layer: bf16 attention weight bytes
    (2 B/elem) plus the small f32 rope table + norm weights — Step 5's
    no-fp16/f32-blowup sanity check."""
    hidden, h = dims["hidden"], dims["h"]
    nope, rope_d, vh = dims["qk_nope"], dims["qk_rope"], dims["v_head"]
    kvl, ql = dims["kv_lora"], dims["q_lora"]
    rows, p_max = dims["rows"], dims["p_max"]
    qk = nope + rope_d
    W = p_max + rows
    half = rope_d // 2
    bf16_elems = (ql * hidden + h * qk * ql + (kvl + rope_d) * hidden
                  + h * (nope + vh) * kvl + hidden * h * vh)
    f32_elems = hidden + ql + kvl + 2 * W * half
    return {"bf16_bytes": bf16_elems * 2, "f32_bytes": f32_elems * 4,
            "total": bf16_elems * 2 + f32_elems * 4}


# --------------------------------------------------------------------------
# --validate-graph (Step 3) — synthetic dims small enough to compile/run fast;
# `qk_rope=REAL_QK_ROPE`/`theta=REAL_ROPE_THETA`/`p_max+rows=16` deliberately
# match the DEFAULT rope-table dump, so the common case needs no extra
# `cargo test` invocation beyond what `--validate-ops` already asked for.
# --------------------------------------------------------------------------

VG_HIDDEN, VG_H, VG_QK_NOPE, VG_V_HEAD, VG_KV_LORA, VG_Q_LORA = 48, 3, 12, 10, 8, 16
VG_QK_ROPE = REAL_QK_ROPE   # 64 -- matches the default rope dump (seqlen=16)
VG_P_MAX, VG_ROWS_MAX = 8, 8  # p_max+rows_max = 16, matches that same dump exactly


def _vg_synthetic_weights(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    h, nope, rope_d, vh = VG_H, VG_QK_NOPE, VG_QK_ROPE, VG_V_HEAD
    kvl, ql, hidden = VG_KV_LORA, VG_Q_LORA, VG_HIDDEN
    qk = nope + rope_d

    def bf16bits(shape, scale=0.02):
        f = rng.standard_normal(shape).astype(np.float32) * scale
        return (f.view(np.uint32) >> np.uint32(16)).astype(np.uint16)

    def normw(n):
        return (1.0 + 0.05 * rng.standard_normal(n)).astype(np.float32)

    return {
        "in_ln": normw(hidden),
        "wq_a_bits": bf16bits((ql, hidden)),
        "q_a_ln": normw(ql),
        "wq_b_bits": bf16bits((h * qk, ql)),
        "wkv_a_bits": bf16bits((kvl + rope_d, hidden)),
        "kv_a_ln": normw(kvl),
        "wkv_b_bits": bf16bits((h * (nope + vh), kvl)),
        "wo_bits": bf16bits((hidden, h * vh)),
    }


def validate_graph(core, rope_dump_path: str):
    """Returns True (pass), False (fail), or None (rope dump SKIPPED — same
    non-fatal-but-loud discipline as `--validate-ops`'s exact gate)."""
    print("\n== validate-graph (Step 3) ==")
    dims = {
        "hidden": VG_HIDDEN, "h": VG_H, "qk_nope": VG_QK_NOPE, "qk_rope": VG_QK_ROPE,
        "v_head": VG_V_HEAD, "kv_lora": VG_KV_LORA, "q_lora": VG_Q_LORA,
        "rms_norm_eps": RMS_NORM_EPS, "rope_theta": REAL_ROPE_THETA,
        "rows": VG_ROWS_MAX, "p_max": VG_P_MAX,
    }
    p_max, rows = dims["p_max"], dims["rows"]
    W = p_max + rows
    table = load_rope_table(rope_dump_path, VG_QK_ROPE, W, REAL_ROPE_THETA, hard_fail=False)
    if table is None:
        return None
    weights = dict(_vg_synthetic_weights(seed=42), rope_cos=table[0], rope_sin=table[1])

    model = build_layer_graph(weights, dims)
    node_count = len(model.get_ordered_ops())
    flag = "  ** > 50k nodes -- flag per brief **" if node_count > 50_000 else ""
    print(f"  graph node count: {node_count}{flag}")
    compiled = compile_checked(core, model, "validate-graph")
    _kernel_names_seen.update(_kernel_names(compiled))
    req = compiled.create_infer_request()

    rng = np.random.default_rng(7)
    x_full = (rng.standard_normal((rows, dims["hidden"])).astype(np.float32) * 0.3)
    past_lc_full = bf16_round(rng.standard_normal((p_max, dims["kv_lora"])).astype(np.float32) * 0.3)
    past_rc_full = bf16_round(rng.standard_normal((p_max, dims["qk_rope"])).astype(np.float32) * 0.3)

    past_lens = sorted({0, 1, p_max // 2, p_max})
    row_counts = sorted({1, 5, rows})
    print(f"  past_len cases: {past_lens}  rows cases: {row_counts}")

    ok = True
    save_case = None  # (inputs, before-outputs) for the Step 5 round-trip check
    for past_len in past_lens:
        past_lc_in = np.zeros_like(past_lc_full)
        past_rc_in = np.zeros_like(past_rc_full)
        past_lc_in[:past_len] = past_lc_full[:past_len]
        past_rc_in[:past_len] = past_rc_full[:past_len]
        true_past_lc = past_lc_full[:past_len]
        true_past_rc = past_rc_full[:past_len]
        mask_in = build_window_mask(rows, p_max, past_len)

        for real_rows in row_counts:
            x_in = np.zeros_like(x_full)
            x_in[:real_rows] = x_full[:real_rows]

            ref_out, ref_lc, ref_rc = ref_attention(
                weights, x_full[:real_rows], true_past_lc, true_past_rc, dims)

            inputs = {"x": x_in, "past_lc": past_lc_in, "past_rc": past_rc_in, "mask": mask_in}
            got = req.infer(inputs)
            got_attn = np.array(got[0])[:real_rows]
            got_lc = np.array(got[1])[:real_rows]
            got_rc = np.array(got[2])[:real_rows]

            max_ulp_a, flip_a, worst_a, rb_a, gb_a = ulp_stats(ref_out, got_attn)
            max_ulp_lc, _, _, _, _ = ulp_stats(ref_lc, got_lc)
            max_ulp_rc, _, _, _, _ = ulp_stats(ref_rc, got_rc)
            # attn_out is `linear_bf16_w(ctx, wo)` in Rust,
            # which ALWAYS bf16-rounds its output same as lc/rc's rmsnorm/rope
            # — it was missing the on-grid structural check, leaving it
            # elision-blind on its own (ULP alone re-rounds both sides).
            on_grid_a = on_grid(got_attn)
            on_grid_lc = on_grid(got_lc)
            on_grid_rc = on_grid(got_rc)

            passed = (max_ulp_a <= 1 and flip_a <= 0.01 and on_grid_a
                      and max_ulp_lc <= 1 and on_grid_lc
                      and max_ulp_rc <= 1 and on_grid_rc)
            ok &= passed
            status = "PASS" if passed else "FAIL"
            note = "  <- P=0 case" if past_len == 0 else ""
            print(f"  [{status}] past_len={past_len:2d} rows={real_rows:2d} "
                  f"attn(ulp={max_ulp_a},flip={flip_a:.4f},grid={on_grid_a}) "
                  f"lc(ulp={max_ulp_lc},grid={on_grid_lc}) rc(ulp={max_ulp_rc},grid={on_grid_rc}){note}")
            if not passed:
                print(f"    worst attn idx={worst_a} ref=0x{rb_a.flat[worst_a]:04x} ov=0x{gb_a.flat[worst_a]:04x}")
            if past_len == p_max // 2 and real_rows == min(5, rows) and save_case is None:
                # Unsliced (full [rows,...]) outputs -- Step 5 re-infers the
                # SAME `inputs` on the reloaded model and compares the FULL
                # tensors, not just the real-row prefix.
                save_case = (inputs, np.array(got[0]).copy(), np.array(got[1]).copy(), np.array(got[2]).copy())

    # Step 5: save -> core.read_model -> re-run one case, assert the round
    # trip preserves the baked constants (bf16 weights + rope table) exactly.
    print("  -- Step 5: save/reload round-trip --")
    tmp_dir = tempfile.mkdtemp(prefix="glm5_attn_vg_")
    xml_path = os.path.join(tmp_dir, "layer.xml")
    bin_path = os.path.join(tmp_dir, "layer.bin")
    ov.save_model(model, xml_path, compress_to_fp16=False)
    reloaded = core.read_model(xml_path)
    compiled2 = compile_checked(core, reloaded, "validate-graph (reloaded)")
    req2 = compiled2.create_infer_request()
    inputs, before_attn, before_lc, before_rc = save_case
    after = req2.infer(inputs)
    roundtrip_ok = (np.array_equal(np.array(after[0]), before_attn)
                     and np.array_equal(np.array(after[1]), before_lc)
                     and np.array_equal(np.array(after[2]), before_rc))
    sizes = expected_bf16_layer_bytes(dims)
    bin_size = os.path.getsize(bin_path)
    size_ok = bin_size < 1.5 * sizes["total"]
    print(f"    round-trip exact match (attn_out/lc/rc): {roundtrip_ok}")
    print(f"    .bin={bin_size}B  expected~{sizes['total']}B "
          f"(bf16={sizes['bf16_bytes']}B f32={sizes['f32_bytes']}B)  size_ok={size_ok}")
    ok &= roundtrip_ok and size_ok
    return ok


# --------------------------------------------------------------------------
# Export mode (Step 4) + stamp
# --------------------------------------------------------------------------

def export_all(model_dir: str, out_dir: str, rope_dump_path: str, layers_filter=None,
                rows_cap: int = MAX_BATCH_COUNT):
    """`rows_cap` overrides the graph's baked row capacity (default
    MAX_BATCH_COUNT, the production wire's window size). A small `W` model
    (e.g. the `--tiny` DSA fixture, `index_topk=8`) cannot fit MAX_BATCH_COUNT
    at all -- every window would need zero past capacity -- so the tiny-model
    parity harness passes a small `rows_cap` here to get a graph
    whose shape actually fits the tiny budget. Production exports never pass
    this (the default reproduces the exact prior behavior)."""
    manifest = read_manifest(model_dir)
    w = derive_w(manifest)
    rows = rows_cap
    if w <= rows:
        raise SystemExit(f"derived W={w} <= rows={rows} (rows_cap) — every "
                          "window would have zero past capacity; refusing to export")
    p_max = w - rows

    hidden = manifest["hidden_size"]
    kv_lora = manifest["kv_lora_rank"]
    qk_rope = manifest["qk_rope_head_dim"]
    rope_theta = manifest["rope_theta"]
    cos, sin = load_rope_table(rope_dump_path, qk_rope, w, rope_theta, hard_fail=True)

    layers = list(range(manifest["num_layers"]))
    if layers_filter is not None:
        layers = [li for li in layers if li in layers_filter]
    if not layers:
        raise SystemExit("no layers selected (--layers filter matched nothing)")

    attn_dir = os.path.join(out_dir, "attn_ov")
    os.makedirs(attn_dir, exist_ok=True)

    # Drop any prior stamp BEFORE writing a single .xml/.bin. A crash halfway
    # through would otherwise leave last run's valid-looking stamp beside a
    # mixed-generation IR set, which the engine would accept.
    stamp_path = os.path.join(attn_dir, "export_stamp.json")
    if os.path.exists(stamp_path):
        os.remove(stamp_path)

    core = ov.Core()

    manifest_path = os.path.join(model_dir, "manifest.json")
    with open(manifest_path, "rb") as f:
        manifest_sha256 = hashlib.sha256(f.read()).hexdigest()

    dims = layer_dims(manifest, w, rows)
    per_layer_digest = {}
    ir_digest = {}
    total_bytes = 0
    for li in layers:
        shell_path = os.path.join(model_dir, "shells", f"layer_{li:02d}.safetensors")
        with open(shell_path, "rb") as f:
            per_layer_digest[f"{li:02d}"] = hashlib.sha256(f.read()).hexdigest()

        w_layer = load_layer_weights(model_dir, li, manifest)
        w_layer["rope_cos"], w_layer["rope_sin"] = cos, sin
        model = build_layer_graph(w_layer, dims)
        # Compile once before committing to disk -- catches a broken export
        # before it becomes a stale-looking .xml/.bin pair. The compiles that
        # produce what SHIPS are inspected for quantized kernels too; they
        # used to be the only ones that weren't.
        compiled = compile_checked(core, model, f"export layer_{li:02d}")
        _kernel_names_seen.update(_kernel_names(compiled))

        xml_path = os.path.join(attn_dir, f"layer_{li:02d}.xml")
        bin_path = os.path.join(attn_dir, f"layer_{li:02d}.bin")
        ov.save_model(model, xml_path, compress_to_fp16=False)
        bin_size = os.path.getsize(bin_path)
        total_bytes += bin_size + os.path.getsize(xml_path)
        # One digest over xml||bin, byte-for-byte what `ov_attn.rs::
        # verify_artifact_digests` re-derives at load.
        h = hashlib.sha256()
        for p in (xml_path, bin_path):
            with open(p, "rb") as f:
                for chunk in iter(lambda f=f: f.read(1 << 20), b""):
                    h.update(chunk)
        ir_digest[f"{li:02d}"] = h.hexdigest()
        print(f"  layer_{li:02d}: {bin_size} bytes .bin")

    stamp = {
        "manifest_sha256": manifest_sha256,
        "per_layer_digest": per_layer_digest,
        "ir_digest": ir_digest,
        "exporter_version": EXPORTER_VERSION,
        "w": w,
        "p_max": p_max,
        "rows": rows,
        "hidden": hidden,
        "kv_lora": kv_lora,
        "qk_rope": qk_rope,
    }
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"  wrote {stamp_path}")
    print(f"  total attn_ov bytes: {total_bytes} across {len(layers)} layer(s)")
    return stamp


# --------------------------------------------------------------------------
# Step 6b — f32 throughput tripwire. Needs a fleet node + a real manifest;
# this only BUILDS the harness (it prints the single command for an operator
# to run) and never fabricates a number.
# --------------------------------------------------------------------------

def bench_layer(model_dir: str, layer: int, rope_dump_path: str, device: str, iters: int):
    manifest = read_manifest(model_dir)
    w = derive_w(manifest)
    rows = MAX_BATCH_COUNT
    p_max = w - rows
    dims = layer_dims(manifest, w, rows)
    cos, sin = load_rope_table(rope_dump_path, manifest["qk_rope_head_dim"], w,
                                manifest["rope_theta"], hard_fail=True)
    weights = load_layer_weights(model_dir, layer, manifest)
    weights["rope_cos"], weights["rope_sin"] = cos, sin

    model = build_layer_graph(weights, dims)
    print(f"openvino {ov.__version__}  device={device}  cfg={COMPILE_CFG}")
    print(f"shapes: hidden={dims['hidden']} h={dims['h']} kv_lora={dims['kv_lora']} "
          f"qk_rope={dims['qk_rope']} v_head={dims['v_head']} W={w} p_max={p_max} rows={rows}")
    print(f"graph node count: {len(model.get_ordered_ops())}")

    core = ov.Core()
    compiled = compile_checked(core, model, f"bench layer_{layer:02d}", device=device)
    req = compiled.create_infer_request()

    rng = np.random.default_rng(0)
    feed = {
        "x": rng.standard_normal((rows, dims["hidden"])).astype(np.float32),
        "past_lc": bf16_round(rng.standard_normal((p_max, dims["kv_lora"])).astype(np.float32)),
        "past_rc": bf16_round(rng.standard_normal((p_max, dims["qk_rope"])).astype(np.float32)),
        "mask": build_window_mask(rows, p_max, p_max),  # steady-state: past fully populated
    }
    for _ in range(5):
        req.infer(feed)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        req.infer(feed)
        ts.append((time.perf_counter() - t0) * 1e3)
    med, p95 = statistics.median(ts), sorted(ts)[int(len(ts) * 0.95) - 1]
    print(f"OV layer_{layer:02d} (rows={rows}, past=p_max steady-state): "
          f"median={med:.3f}ms p95={p95:.3f}ms over {iters} iters")
    # glm5_attn_bench.rs's default batch bucket list used
    # to be [1, 512, 1024, 2048] -- no 256, so this command couldn't produce
    # the "batch=256" numerator the line below asks for. 256 is now in its
    # default list (dist.rs MAX_BATCH_COUNT = `rows` above), but the batch is
    # still passed explicitly here so this stays correct even if that
    # default list changes again.
    print(f"\nRust baseline (five-projection sum, comparable dims): "
          f"cargo run --release -p cascadia-engine-sparse-moe --example glm5_attn_bench "
          f"-- {model_dir} {iters} {rows}")
    print(f"speedup = (Rust batch={rows} five-projection median ms) / "
          f"(this script's median ms = {med:.3f})")
    print("Step 6b gate: speedup >= ~20x to proceed with Tasks 3-9 as scoped "
          "(spec Sec13); gates survive down to ~5x.")
    return med, p95


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-ops", action="store_true",
                     help="run the self-contained per-op numerics harness (no model dir needed)")
    ap.add_argument("--validate-graph", action="store_true",
                     help="run the self-contained full-attention-graph harness (Step 3, no model dir needed)")
    ap.add_argument("--export", metavar="MODEL_DIR",
                     help="export attn_ov/{layer_NN.xml,.bin,export_stamp.json} for a real model dir "
                          "(manifest.json + shells/layer_NN.safetensors)")
    ap.add_argument("--out", metavar="OUT_DIR",
                     help="--export only: output dir (default: MODEL_DIR itself)")
    ap.add_argument("--layers", default=None,
                     help="--export only: comma-separated global layer indices to export (dev filter; "
                          "default: all layers, including first_k_dense_replace dense-MLP layers)")
    ap.add_argument("--rows-cap", type=int, default=MAX_BATCH_COUNT,
                     help="--export only: override the graph's baked row capacity (default "
                          "MAX_BATCH_COUNT=256, the production wire's window size). Needed for a "
                          "small-W model (e.g. --tiny's index_topk=8) where MAX_BATCH_COUNT can't "
                          "fit at all -- the tiny-model parity harness uses this.")
    ap.add_argument("--bench", metavar="MODEL_DIR",
                     help="Step 6b throughput tripwire: time one layer's graph under the f32 contract "
                          "against the Rust five-projection baseline (needs a fleet node + a real "
                          "manifest.json/shells; it cannot run on a machine without one)")
    ap.add_argument("--bench-layer", type=int, default=0, help="--bench: which global layer index to time")
    ap.add_argument("--bench-device", default="GPU", help="--bench: OV device (default GPU)")
    ap.add_argument("--bench-iters", type=int, default=50, help="--bench: timed iterations")
    ap.add_argument("--canary", action="store_true",
                     help="device canary -- compiles the shipped bf16-rounding "
                          "subgraph on --device with the real compile config and asserts raw bits "
                          "(no model dir needed; run this on any device before trusting it)")
    ap.add_argument("--device", default="CPU", help="--canary: OV device to canary (default CPU)")
    ap.add_argument("--rope-table-dump", default=os.path.join(tempfile.gettempdir(), "glm5_rope_freqs_dump.bin"),
                     help="path to the Rust Freqs dump for the rope-table exact gate / --validate-graph / "
                          "--export / --bench (default matches dsv4/rope.rs::freqs_dump's default output path)")
    ap.add_argument("--inject-bad-config", action="store_true", help=argparse.SUPPRESS)
    # ^ debug-only: deliberately requests DYNAMIC_QUANTIZATION_GROUP_SIZE=32
    # (a value the CPU plugin WILL honor) so the effective-config readback
    # gate has a real mismatch to catch, proving it fails loudly instead of
    # passing vacuously. Not for normal use.
    args = ap.parse_args()

    if args.inject_bad_config:
        COMPILE_CFG["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "32"
        print("*** --inject-bad-config: requesting DYNAMIC_QUANTIZATION_GROUP_SIZE=32 "
              "(contract requires 0) to prove the readback gate fires ***")
    # ^ Deliberately ahead of ALL mode dispatch (it was previously
    # only wired for --validate-ops/--validate-graph) so this debug flag can
    # prove the drift gate fires on --export/--bench/--canary too, now that
    # `_exit_after_drift_check()` enforces it on every one of them.

    # EVERY exit path below is wrapped by this `finally` --
    # `_exit_after_drift_check()` is the single choke point that makes
    # `--export`/`--bench` (which `return` well before validate mode's own
    # inline checks) fail nonzero on effective-config drift too. See its
    # docstring / the comment above `_config_drift_ok` for why this was a
    # live bug, not a hypothetical one.
    try:
        if args.export:
            out_dir = args.out or args.export
            layers_filter = {int(x) for x in args.layers.split(",")} if args.layers else None
            stamp = export_all(args.export, out_dir, args.rope_table_dump, layers_filter,
                                rows_cap=args.rows_cap)
            print(f"\nexport_stamp: {json.dumps(stamp, indent=2)}")
            return

        if args.bench:
            bench_layer(args.bench, args.bench_layer, args.rope_table_dump, args.bench_device, args.bench_iters)
            return

        if args.canary:
            core = ov.Core()
            ok = run_canary(core, args.device)
            if not ok:
                print("\nFAILURES ABOVE")
                raise SystemExit(1)
            print("\nALL PASS")
            raise SystemExit(0)

        if not (args.validate_ops or args.validate_graph):
            ap.print_help()
            return

        print(f"openvino {ov.__version__}  device={DEVICE}  cfg={COMPILE_CFG}")
        core = ov.Core()

        ok = True
        critical_skips = []

        if args.validate_ops:
            ok &= validate_linear(core)
            ok &= validate_rmsnorm(core)
            ok &= validate_rope(core)
            ok &= validate_bf16_roundtrip(core)
            rope_table_dump_present = os.path.exists(args.rope_table_dump)
            rope_table_exact = validate_rope_table_exact(args.rope_table_dump)
            # A present-but-mismatched table IS a hard failure. A MISSING dump is
            # kept non-fatal (a fresh checkout legitimately hasn't run the Rust
            # test yet) but is tracked separately so the final summary can't read
            # as "ALL PASS" while the one safety-critical exact-match gate never
            # ran.
            if rope_table_dump_present:
                ok &= rope_table_exact
            else:
                critical_skips.append(
                    "the rope-table EXACT gate did NOT run (no Rust dump found at "
                    f"{args.rope_table_dump}) — the one bit-for-bit-required check in "
                    "this harness was skipped, not passed"
                )

        if args.validate_graph:
            vg_result = validate_graph(core, args.rope_table_dump)
            # Same discipline as the rope-table gate above: a missing dump is a loud,
            # non-fatal SKIPPED, not a silent pass.
            if vg_result is None:
                critical_skips.append(
                    "--validate-graph did NOT run (no Rust rope-table dump found at "
                    f"{args.rope_table_dump}) — regenerate per the message printed above"
                )
            else:
                ok &= vg_result

        ok &= _kernel_scan_ok()
        ok &= _config_drift_ok()

        if not ok:
            print("\nFAILURES ABOVE")
            raise SystemExit(1)
        if critical_skips:
            print("\n" + "!" * 78)
            print("!! PASSED, BUT CRITICAL CHECK(S) SKIPPED — THIS IS NOT A FULL VALIDATION RUN !!")
            for s in critical_skips:
                print(f"!!   {s}")
            print("!" * 78)
            # Exit 2, not 0: a wrapper or CI job reading $? must not see green
            # for a run where a safety-critical gate never executed. Distinct
            # from 1 so "skipped" and "failed" stay tellable apart.
            raise SystemExit(2)
        print("\nALL PASS")
        raise SystemExit(0)
    finally:
        _exit_after_contract_checks()


if __name__ == "__main__":
    main()
