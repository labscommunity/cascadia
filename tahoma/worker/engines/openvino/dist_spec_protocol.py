"""Control-frame wire protocol for distributed speculative decoding.

Adds a thin command layer on top of `transport.py`'s `send_tensor` /
`recv_tensor`. The driver issues commands to each worker stage; the worker
responds (for FORWARD) or applies state changes (REWIND, RESET).

Frame format::

    [4B kind BE] [4B int_arg BE] [4B has_tensor BE]
    [if has_tensor: send_tensor format (24B header + payload)]

`kind`:
    1 = FORWARD          payload = activation tensor
    2 = REWIND           int_arg = count of cache positions to drop
    3 = RESET            (no args)
    4 = LOGITS_RESPONSE  payload = logits tensor (worker → driver)

Tensor payloads reuse the existing `send_tensor` / `recv_tensor` format so we
don't duplicate dtype + shape encoding.
"""

from __future__ import annotations

import socket
import struct
from enum import IntEnum

import numpy as np

from tahoma.worker.transport import (
    _recv_exact,
    recv_tensor,
    send_tensor,
)


class FrameKind(IntEnum):
    FORWARD = 1
    REWIND = 2
    RESET = 3
    LOGITS_RESPONSE = 4


_FRAME_HEADER_FMT = ">III"
_FRAME_HEADER_SIZE = 12


def send_frame(
    sock: socket.socket,
    kind: FrameKind,
    *,
    int_arg: int = 0,
    tensor: np.ndarray | None = None,
) -> None:
    """Send a control frame. `tensor` is optional payload."""
    has_tensor = 1 if tensor is not None else 0
    header = struct.pack(_FRAME_HEADER_FMT, int(kind), int_arg, has_tensor)
    sock.sendall(header)
    if tensor is not None:
        send_tensor(sock, tensor)


def recv_frame(sock: socket.socket) -> tuple[FrameKind, int, np.ndarray | None]:
    """Receive one control frame. Returns (kind, int_arg, tensor_or_None)."""
    header = _recv_exact(sock, _FRAME_HEADER_SIZE)
    kind_int, int_arg, has_tensor = struct.unpack(_FRAME_HEADER_FMT, header)
    tensor: np.ndarray | None = None
    if has_tensor:
        tensor, _stats = recv_tensor(sock)
    return FrameKind(kind_int), int_arg, tensor
