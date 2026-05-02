"""Smoke tests for tahoma.worker.transport."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from tahoma.worker.transport import (
    ActivationClient,
    ActivationServer,
    send_tensor,
)


def _free_port() -> int:
    """Pick an OS-assigned port (best-effort; small race with re-bind)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_once(server: ActivationServer, received: dict) -> None:
    server.accept()
    tensor, stats = server.recv()
    received["tensor"] = tensor
    received["stats"] = stats


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 8, 16), np.float16),
        ((1, 4, 32), np.float32),
        ((2, 3, 5), np.int8),
        ((1, 1, 100), np.int32),
        ((1, 1, 64), np.int64),  # used by attention_mask in dist_spec_protocol
    ],
)
def test_roundtrip(shape: tuple[int, ...], dtype: type) -> None:
    server = ActivationServer("127.0.0.1", 0)
    server.start()

    payload = (np.random.randn(*shape) * 100).astype(dtype)
    received: dict = {}

    server_thread = threading.Thread(
        target=_serve_once, args=(server, received), daemon=True,
    )
    server_thread.start()
    time.sleep(0.05)  # let the server reach accept()

    client = ActivationClient("127.0.0.1", server.port)
    client.connect(timeout=2.0)
    send_stats = client.send(payload)

    server_thread.join(timeout=2.0)
    assert not server_thread.is_alive()

    np.testing.assert_array_equal(received["tensor"].reshape(shape), payload)
    assert received["stats"].bytes == send_stats.bytes
    assert send_stats.bytes >= payload.nbytes

    client.close()
    server.close()


def test_connect_timeout_raises() -> None:
    """ActivationClient raises TimeoutError when nothing is listening."""
    client = ActivationClient("127.0.0.1", _free_port())
    with pytest.raises(TimeoutError):
        client.connect(timeout=0.5)


def test_send_rank_too_high_raises() -> None:
    """send_tensor refuses tensors with rank > 3."""
    bad = np.zeros((1, 2, 3, 4), dtype=np.float32)
    a, b = socket.socketpair()
    try:
        with pytest.raises(ValueError, match="rank"):
            send_tensor(a, bad)
    finally:
        a.close()
        b.close()
