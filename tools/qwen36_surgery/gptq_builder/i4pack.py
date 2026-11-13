"""Signed-INT4 repacking of GPTQ nibbles for the OpenVINO GPU plugin (`gptq_to_ov.py --expert-sym`).

GPTQ sym (zp=8) stores each weight as an unsigned nibble q in [0, 15] meaning (q - 8) * scale.
OpenVINO's u4 + zero-point form (Convert -> Subtract(zp) -> Multiply) says exactly that, but the
GPU plugin's MoE fusion (MOECompressed, OpenVINO 2026.4.1) rejects the uniform-zero-point variant
for the expert stacks. The signed i4 form (Convert -> Multiply, no Subtract — NNCF's INT4_SYM
pattern) carries the same values: the two's-complement nibble of (q - 8) is q ^ 0x8, so a packed
byte stream (two nibbles per byte, low nibble first) is the u4 stream XOR 0x88.

Pure Python here so the transform is unit-testable without numpy/openvino
(`tools/tests/test_gptq_expert_sym.py`); the builder applies the numpy variant to whole expert
stacks.
"""

from __future__ import annotations

# Both nibbles of a byte flip their sign bit: 0x8 in the low nibble, 0x80 in the high one.
U4_TO_I4_XOR = 0x88


def i4_nibble_value(nib: int) -> int:
    """Value of a 4-bit two's-complement nibble (0x0..0xF -> -8..7)."""
    if not 0 <= nib <= 0xF:
        raise ValueError(f"not a nibble: {nib!r}")
    return nib - 16 if nib >= 8 else nib


def u4_to_i4_nibble(q: int, zp: int = 8) -> int:
    """GPTQ nibble q with zero point zp=8 -> the i4 nibble whose signed value is q - 8."""
    if zp != 8:
        raise ValueError("the XOR identity holds for zp=8 only")
    if not 0 <= q <= 0xF:
        raise ValueError(f"not a nibble: {q!r}")
    return q ^ 0x8


def u4_to_i4_bytes(packed: bytes) -> bytes:
    """Packed u4 pairs (low nibble first) -> packed i4 pairs with the same per-weight values."""
    return bytes(b ^ U4_TO_I4_XOR for b in packed)


def unpack_nibbles(packed: bytes) -> list[int]:
    """Packed pairs -> nibbles, low nibble first (OpenVINO's u4/i4 element order)."""
    out: list[int] = []
    for b in packed:
        out.append(b & 0xF)
        out.append(b >> 4)
    return out


def u4_to_i4_packed(packed, zp: int = 8):
    """numpy variant for whole expert stacks: uint8 array of packed u4 pairs -> contiguous uint8
    array of packed i4 pairs (same shape). The builder keeps the result alive next to the Constant."""
    if zp != 8:
        raise ValueError("the XOR identity holds for zp=8 only")
    import numpy as np

    return np.ascontiguousarray(np.asarray(packed, dtype=np.uint8) ^ np.uint8(U4_TO_I4_XOR))
