"""Frame-level round-trip tests for dist_spec_protocol."""

from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from tahoma.worker.engines.openvino.dist_spec_protocol import (
    FrameKind,
    recv_forward_body,
    recv_kind,
    recv_logits_body,
    send_forward,
    send_logits,
    send_reset,
)


def _pair() -> tuple[socket.socket, socket.socket]:
    a, b = socket.socketpair()
    return a, b


def test_forward_roundtrip() -> None:
    a, b = _pair()
    try:
        attn = np.array([[1, 1, 1, 0, 1]], dtype=np.int64)
        hidden = (np.random.randn(1, 2, 8) * 10).astype(np.float16)

        def writer() -> None:
            send_forward(a, logical_pos_start=42, attention_mask=attn, hidden_states=hidden)

        t = threading.Thread(target=writer, daemon=True)
        t.start()

        kind = recv_kind(b)
        assert kind == FrameKind.FORWARD

        pos, attn_rx, hs_rx = recv_forward_body(b)
        assert pos == 42
        # attention_mask is normalized back to rank-2.
        np.testing.assert_array_equal(attn_rx, attn)
        np.testing.assert_array_equal(hs_rx, hidden)

        t.join(timeout=2.0)
    finally:
        a.close()
        b.close()


def test_reset_kind_only() -> None:
    a, b = _pair()
    try:
        send_reset(a)
        kind = recv_kind(b)
        assert kind == FrameKind.RESET
    finally:
        a.close()
        b.close()


def test_logits_roundtrip() -> None:
    a, b = _pair()
    try:
        logits = (np.random.randn(1, 3, 100) * 5).astype(np.float16)

        def writer() -> None:
            send_logits(a, logits)

        t = threading.Thread(target=writer, daemon=True)
        t.start()

        assert recv_kind(b) == FrameKind.LOGITS_RESPONSE
        rx = recv_logits_body(b)
        np.testing.assert_array_equal(rx, logits)
        t.join(timeout=2.0)
    finally:
        a.close()
        b.close()


def test_unknown_kind_int_raises() -> None:
    """A frame kind outside the IntEnum should fail when constructed."""
    with pytest.raises(ValueError):
        FrameKind(99)
