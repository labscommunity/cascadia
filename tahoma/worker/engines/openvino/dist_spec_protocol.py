"""Control-frame wire protocol for distributed speculative decoding (v5).

v5 shards use canonical optimum-intel-style inputs:
``(input_ids|hidden_states, attention_mask, position_ids, beam_idx)``. KV cache
is internal stateful storage. Rewind is mask-based: the driver flips bits in
its `valid_mask` and the worker just sees an updated `attention_mask` on the
next FORWARD — no `query_state`/`set_state` round-trip needed.

Frames
------

FORWARD (driver -> worker, also worker -> next worker for chains > 2 stages)
    [4B kind=1 BE][4B logical_pos_start BE]
    + send_tensor(attention_mask)   # int64 [1, total_seq_len]
    + send_tensor(hidden_states)    # float16 [1, new_tokens, hidden_size]

RESET (driver -> worker, propagated downstream)
    [4B kind=3 BE]

LOGITS_RESPONSE (worker -> upstream)
    [4B kind=4 BE]
    + send_tensor(logits)           # float16 [1, new_tokens, vocab_size]

`logical_pos_start` is the position id of the first new token (i.e. the
driver's `logical_pos`). The worker derives `position_ids =
arange(logical_pos_start, logical_pos_start + new_tokens)`. `attention_mask`
covers the full `total_seq_len = past_in_cache + new_tokens` and is what
encodes any rewinds the driver has applied.

Tensor payloads reuse `transport.send_tensor` / `recv_tensor` so dtype + shape
encoding stays in one place.
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
    RESET = 3
    LOGITS_RESPONSE = 4


_KIND_FMT = ">I"
_KIND_SIZE = 4
_FORWARD_HEADER_FMT = ">II"  # kind + logical_pos_start
_FORWARD_HEADER_SIZE = 8


def send_forward(
    sock: socket.socket,
    logical_pos_start: int,
    attention_mask: np.ndarray,
    hidden_states: np.ndarray,
) -> None:
    """Driver/relay -> next stage. Sends an attention-aware activation frame."""
    header = struct.pack(_FORWARD_HEADER_FMT, int(FrameKind.FORWARD), int(logical_pos_start))
    sock.sendall(header)
    send_tensor(sock, attention_mask)
    send_tensor(sock, hidden_states)


def send_reset(sock: socket.socket) -> None:
    sock.sendall(struct.pack(_KIND_FMT, int(FrameKind.RESET)))


def send_logits(sock: socket.socket, logits: np.ndarray) -> None:
    sock.sendall(struct.pack(_KIND_FMT, int(FrameKind.LOGITS_RESPONSE)))
    send_tensor(sock, logits)


def recv_kind(sock: socket.socket) -> FrameKind:
    """Read just the kind. Useful for dispatching in the worker loop."""
    raw = _recv_exact(sock, _KIND_SIZE)
    (kind,) = struct.unpack(_KIND_FMT, raw)
    return FrameKind(kind)


def recv_forward_body(
    sock: socket.socket,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Read the rest of a FORWARD frame after `recv_kind` returned FORWARD.

    Returns (logical_pos_start, attention_mask, hidden_states). Tensor ranks
    are normalized to their canonical shapes (attention_mask: [1, total],
    hidden_states: [1, new_tokens, hidden_size]) — `transport.send_tensor`
    pads to MAX_RANK=3 on the wire.
    """
    raw = _recv_exact(sock, _FORWARD_HEADER_SIZE - _KIND_SIZE)  # 4 more bytes
    (logical_pos_start,) = struct.unpack(">I", raw)
    attention_mask, _ = recv_tensor(sock)
    if attention_mask.ndim == 3 and attention_mask.shape[0] == 1:
        attention_mask = attention_mask[0]
    hidden_states, _ = recv_tensor(sock)
    return int(logical_pos_start), attention_mask, hidden_states


def recv_logits_body(sock: socket.socket) -> np.ndarray:
    logits, _ = recv_tensor(sock)
    # Logits are canonical [1, new_tokens, vocab_size]; recv returns same.
    return logits
