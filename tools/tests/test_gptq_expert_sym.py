"""Unit tests for the GPTQ builder's `--expert-sym` nibble transform (`gptq_builder/i4pack.py`).

Run from the repo root with:

    python -m pytest tools/tests/test_gptq_expert_sym.py -v

The transform is pure Python (no numpy, no openvino), so these run anywhere; the one numpy
cross-check skips when numpy is absent. They pin the identity the GPU-plugin fix relies on: a GPTQ
sym nibble q with zero point 8 and the i4 nibble q ^ 0x8 denote the same weight (q - 8) * scale,
so the packed byte stream of an expert stack is the u4 stream XOR 0x88, in either nibble order.
"""

from __future__ import annotations

import os
import sys

import pytest

_BUILDER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "qwen36_surgery", "gptq_builder")
)
if _BUILDER_DIR not in sys.path:
    sys.path.insert(0, _BUILDER_DIR)

import i4pack  # noqa: E402


def _pack_pairs(nibbles):
    """Low nibble first, like the builder's pack_u4_pairs."""
    assert len(nibbles) % 2 == 0
    return bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2))


def test_every_nibble_keeps_its_value_under_the_sign_flip():
    for q in range(16):
        assert i4pack.i4_nibble_value(i4pack.u4_to_i4_nibble(q)) == q - 8


def test_packed_stream_is_u4_xor_0x88_for_every_byte():
    # Every (low, high) nibble pair -> both weights keep their values after the XOR.
    for lo in range(16):
        for hi in range(16):
            u4 = _pack_pairs([lo, hi])
            i4 = i4pack.u4_to_i4_bytes(u4)
            assert i4 == bytes([u4[0] ^ 0x88])
            lo_i, hi_i = i4pack.unpack_nibbles(i4)
            assert i4pack.i4_nibble_value(lo_i) == lo - 8
            assert i4pack.i4_nibble_value(hi_i) == hi - 8


def test_round_trip_and_length_preserved():
    nibbles = [(i * 7 + 3) % 16 for i in range(256)]
    u4 = _pack_pairs(nibbles)
    i4 = i4pack.u4_to_i4_bytes(u4)
    assert len(i4) == len(u4) == 128
    assert i4pack.u4_to_i4_bytes(i4) == u4, "XOR is its own inverse"
    assert [i4pack.i4_nibble_value(n) for n in i4pack.unpack_nibbles(i4)] == [q - 8 for q in nibbles]


def test_only_the_gptq_zero_point_8_is_accepted():
    with pytest.raises(ValueError):
        i4pack.u4_to_i4_nibble(3, zp=7)
    with pytest.raises(ValueError):
        i4pack.u4_to_i4_packed(b"\x00", zp=0)
    with pytest.raises(ValueError):
        i4pack.i4_nibble_value(16)


def test_numpy_variant_matches_pure_python():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, size=(2, 3, 4, 64), dtype=np.uint8)
    out = i4pack.u4_to_i4_packed(packed)
    assert out.shape == packed.shape and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    assert out.tobytes() == i4pack.u4_to_i4_bytes(packed.tobytes())
