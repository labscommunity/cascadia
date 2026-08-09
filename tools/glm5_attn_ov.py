#!/usr/bin/env python3
"""GLM-5.2 MLA attention exporter — per-op numerics validation harness.

Gate 1c derisking for offloading GLM-5.2 prefill attention to an OpenVINO
device: before any attention graph exists, prove that tiny OV graphs can
reproduce the Rust engine's bf16 activation contract for the three ops that
compose MLA attention (`linear`, `rmsnorm`, `rope`). No model dir, no
attention graph — `--validate-ops` is entirely self-contained (synthetic
weights).

Numeric contract (see `dsv4/math.rs`, `dsv4/rope.rs`, `glm/attn.rs` header):
  * every linear / RMSNorm / rope output is rounded to bf16 (round-to-nearest-
    even); everything else accumulates in f32.
  * `in_ln` uses the manifest's `rms_norm_eps` (1e-5); `q_a_ln`/`kv_a_ln` use
    `glm::attn::MLA_LATENT_EPS` (1e-6) — two DIFFERENT epsilons, so `rmsnorm`
    takes eps as a real graph input (not a baked constant) and validation
    proves 1e-5 vs 1e-6 diverge.
  * dot-product reduction order differs across numpy / OV / Rust (`math.rs`
    documents this as tolerated ULP-level divergence), so exact equality is
    NOT the gate for `linear`/`rmsnorm`/`rope`'s arithmetic — a bounded
    bf16-ULP tolerance is. The one thing that MUST be exact is the rope
    cos/sin table itself: a silently-wrong-basis table is a decode-breaking
    bug that short-prompt parity would not catch. `--validate-ops` diffs the
    table this exporter would embed against a bit-for-bit dump of the Rust
    `Freqs` struct (see `dsv4/rope.rs::freqs_dump::dump_real_dims_freqs`).

Usage:
  python3 tools/glm5_attn_ov.py --validate-ops
  python3 tools/glm5_attn_ov.py --validate-ops --rope-table-dump /path/to/dump.bin
"""
import argparse
import ctypes
import ctypes.util
import os
import tempfile

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
    _libm = ctypes.CDLL(ctypes.util.find_library("m"))
    _libm.cosf.restype = ctypes.c_float
    _libm.cosf.argtypes = [ctypes.c_float]
    _libm.sinf.restype = ctypes.c_float
    _libm.sinf.argtypes = [ctypes.c_float]
    _libm.powf.restype = ctypes.c_float
    _libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
except OSError:
    _libm = None
    print("WARNING: could not load system libm via ctypes; rope table will use "
          "numpy's cos/sin/power, which do not bit-match Rust's libm calls "
          "(Step 3b's exact gate will report NOT EXACT).")

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
# Used to cross-check OV's own f32->bf16->f32 Convert round-trip against
# `bf16_round` on exactly the cases verified against `half::bf16::from_f32`.
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


def _bf16_roundtrip(node):
    """Explicit `Convert f32->bf16->f32`, reproducing the trailing bf16 round
    the Rust contract applies after every linear / RMSNorm / rope."""
    return ops.convert(ops.convert(node, Type.bf16), Type.f32)


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
    """Standalone `x[n] -> Convert(bf16) -> Convert(f32)` graph, used only to
    cross-check OV's own bf16 Convert rounding against `bf16_round` on the
    tie cases (a narrower question than the per-op ULP gate)."""
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


_effective_config_total = 0
_effective_config_bad: list = []


def compile_checked(core, model: Model, name: str):
    """`core.compile_model` + effective-config readback. `INFERENCE_PRECISION_HINT`
    and `DYNAMIC_QUANTIZATION_GROUP_SIZE` are compile HINTS, not guarantees —
    the plugin can silently land elsewhere. Every compile in this harness
    (~17 of them) feeds the numerics every later task builds on, so each one
    reads both properties back and records a failure if either isn't exactly
    the required value, instead of trusting the requested config dict."""
    global _effective_config_total
    compiled = core.compile_model(model, DEVICE, COMPILE_CFG)
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


def _kernel_names(compiled) -> set:
    """Best-effort exec-graph op kind set (mirrors
    `glm5_attn_ov_probe.py::compiled_precisions`), used only to assert no
    int8/dyn-quant kernel snuck into the compiled graph (spec Sec3.1)."""
    names = set()
    try:
        for node in compiled.get_runtime_model().get_ops():
            t = node.get_type_name()
            try:
                any_v = node.get_rt_info()["layerType"]
                t = any_v.astype(str) if hasattr(any_v, "astype") else str(any_v)
            except Exception:  # noqa: BLE001 — best-effort diagnostic
                pass
            names.add(t.lower())
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
        _kernel_names_seen.update(_kernel_names(compiled))
        passed = max_ulp <= 1 and flip <= 0.01
        ok &= passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] seed={case['seed']:2d} out={out_dim:5d} in={in_dim:5d} "
              f"({case['note']}) max_ulp_diff={max_ulp} flip_rate={flip:.4f}")
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
            outs[tag] = got
            passed = max_ulp <= 1 and flip <= 0.01
            ok &= passed
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] seed={case['seed']:2d} dim={dim:5d} eps={tag} ({case['note']}) "
                  f"max_ulp_diff={max_ulp} flip_rate={flip:.4f}")
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
        passed = max_ulp <= 1 and flip <= 0.01
        ok &= passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] seed={case['seed']:2d} rot_dims={rd:3d} theta={theta:.3e} pos={pos:4d} "
              f"({case['note']}) max_ulp_diff={max_ulp} flip_rate={flip:.4f}")
        if not passed:
            print(f"    worst idx={worst} ref=0x{rb[worst]:04x}({ref[worst]!r}) "
                  f"ov=0x{gb[worst]:04x}({got[worst]!r})")
    return ok


def validate_bf16_roundtrip(core) -> bool:
    """OV's own `Convert f32->bf16->f32` vs `bf16_round`, on the SAME 20 tie
    cases verified against `half::bf16::from_f32` in Rust. Not an op under
    the ULP gate (it IS the rounding primitive) — reported separately."""
    print("\n== bf16 Convert round-trip (OV vs numpy, tie cases) ==")
    xin = np.array([np.uint32(b) for b, _ in BF16_TIE_CASES], dtype=np.uint32).view(np.float32)
    model = _bf16_roundtrip_graph(len(BF16_TIE_CASES))
    compiled = compile_checked(core, model, "bf16_roundtrip")
    core_out = np.array(compiled.create_infer_request().infer({"x": xin})[0]).reshape(-1)
    ov_bits = bf16_bits(core_out)
    np_bits = bf16_bits(bf16_round(xin))
    mismatches = [
        (i, hex(bits), f"expect=0x{want:04x}", f"numpy=0x{np_bits[i]:04x}", f"ov=0x{ov_bits[i]:04x}")
        for i, (bits, want) in enumerate(BF16_TIE_CASES)
        if ov_bits[i] != want or np_bits[i] != want
    ]
    if mismatches:
        print(f"  {len(mismatches)}/{len(BF16_TIE_CASES)} mismatches vs half::bf16::from_f32 ground truth:")
        for m in mismatches:
            print(f"    {m}")
    else:
        print(f"  all {len(BF16_TIE_CASES)} tie cases: numpy and OV Convert both match half::bf16::from_f32 exactly")
    return len(mismatches) == 0


def validate_rope_table_exact(rope_dump_path: str) -> bool:
    """Step 3b: the ONE exact gate. Diffs the exporter's precomputed cos/sin
    table against a bit-for-bit dump of Rust's `Freqs.data` at GLM-5.2's real
    rope dims. `numpy`'s `cos`/`powf` need not agree with Rust's libm — a
    silently-different table is a wrong-basis bug that short-prompt parity
    would not catch, so this must be exact, not ULP-tolerant."""
    print("\n== rope table EXACT gate (Step 3b) ==")
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


_kernel_names_seen: set = set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-ops", action="store_true",
                     help="run the self-contained per-op numerics harness (no model dir needed)")
    ap.add_argument("--rope-table-dump", default=os.path.join(tempfile.gettempdir(), "glm5_rope_freqs_dump.bin"),
                     help="path to the Rust Freqs dump for Step 3b's exact gate "
                          "(default matches dsv4/rope.rs::freqs_dump's default output path)")
    ap.add_argument("--inject-bad-config", action="store_true", help=argparse.SUPPRESS)
    # ^ debug-only: deliberately requests DYNAMIC_QUANTIZATION_GROUP_SIZE=32
    # (a value the CPU plugin WILL honor) so the effective-config readback
    # gate has a real mismatch to catch, proving it fails loudly instead of
    # passing vacuously. Not for normal use.
    args = ap.parse_args()

    if not args.validate_ops:
        ap.print_help()
        return

    if args.inject_bad_config:
        COMPILE_CFG["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "32"
        print("*** --inject-bad-config: requesting DYNAMIC_QUANTIZATION_GROUP_SIZE=32 "
              "(contract requires 0) to prove the readback gate fires ***")

    print(f"openvino {ov.__version__}  device={DEVICE}  cfg={COMPILE_CFG}")
    core = ov.Core()

    ok = True
    ok &= validate_linear(core)
    ok &= validate_rmsnorm(core)
    ok &= validate_rope(core)
    ok &= validate_bf16_roundtrip(core)
    rope_table_dump_present = os.path.exists(args.rope_table_dump)
    rope_table_exact = validate_rope_table_exact(args.rope_table_dump)
    # A present-but-mismatched table IS a hard failure. A MISSING dump is
    # kept non-fatal (a fresh checkout legitimately hasn't run the Rust test
    # yet) but is tracked separately below so the final summary can't read as
    # "ALL PASS" while the one safety-critical exact-match gate never ran.
    critical_skips = []
    if rope_table_dump_present:
        ok &= rope_table_exact
    else:
        critical_skips.append(
            "Step 3b rope-table EXACT gate did NOT run (no Rust dump found at "
            f"{args.rope_table_dump}) — the one bit-for-bit-required check in "
            "this harness was skipped, not passed"
        )

    print("\n== exec-graph kernels seen ==")
    for k in sorted(_kernel_names_seen):
        print(f"  {k}")
    bad = [k for k in _kernel_names_seen if "int8" in k or "dynquant" in k.replace("_", "") or "dyn_quant" in k]
    if bad:
        print(f"  FAIL: dyn-quant/int8 kernel(s) present: {bad}")
        ok = False
    else:
        print("  OK: no int8 / dyn-quant kernels")

    print("\n== effective-config readback ==")
    if _effective_config_bad:
        for msg in _effective_config_bad:
            print(f"  FAIL: {msg}")
        print(f"  {len(_effective_config_bad)} mismatch(es) across {_effective_config_total} compiles")
        ok = False
    else:
        print(f"  OK: INFERENCE_PRECISION_HINT={REQUIRED_PRECISION_HINT} and "
              f"DYNAMIC_QUANTIZATION_GROUP_SIZE={REQUIRED_DYN_QUANT_GROUP_SIZE} "
              f"confirmed effective on all {_effective_config_total} compiles")

    if not ok:
        print("\nFAILURES ABOVE")
        raise SystemExit(1)
    if critical_skips:
        print("\n" + "!" * 78)
        print("!! PASSED, BUT CRITICAL CHECK(S) SKIPPED — THIS IS NOT A FULL VALIDATION RUN !!")
        for s in critical_skips:
            print(f"!!   {s}")
        print("!" * 78)
        raise SystemExit(0)
    print("\nALL PASS")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
