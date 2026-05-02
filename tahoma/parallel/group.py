"""Ring-based all-reduce over TCP for a small TP group.

Each TP rank owns one outbound connection to ``(rank + 1) mod tp_size`` and
one inbound from ``(rank - 1) mod tp_size``. all-reduce(sum) runs as
ring-reduce-scatter followed by ring-all-gather, the standard implementation
that minimises bytes-on-the-wire for a given message size.

For tp_size == 2 this degenerates to a single round trip (rank 0 sends its
half, rank 1 sends its half, both sum). For tp_size in {3..8} the ring
implementation is bandwidth-optimal.

Invariants
----------
- Every rank in the group must call collectives in the same order.
- Tensors must be the same shape + dtype on every rank.
- Only ``np.float16`` and ``np.float32`` are supported (the dtypes that
  show up in attention / MLP outputs).
"""

from __future__ import annotations

import logging
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

from tahoma.worker.transport import (
    DEFAULT_SOCKET_TIMEOUT,
    _recv_exact,
)

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {np.float16, np.float32}


@dataclass(frozen=True)
class TPPeer:
    """Address of one peer in a TP group."""

    tp_rank: int
    host: str
    port: int


def _send_chunk(sock: socket.socket, chunk: np.ndarray) -> None:
    """Send a numpy chunk: [4B byte_count BE][raw bytes]."""
    data = chunk.tobytes()
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_chunk(sock: socket.socket, dtype: type, count: int) -> np.ndarray:
    raw = _recv_exact(sock, 4)
    (n,) = struct.unpack(">I", raw)
    payload = _recv_exact(sock, n)
    arr = np.frombuffer(payload, dtype=dtype)
    if arr.size != count:
        raise RuntimeError(
            f"TP collective shape mismatch: peer sent {arr.size} elements, "
            f"expected {count}",
        )
    return arr


class TPGroup:
    """A TP group owns N-1 peer connections and runs collectives over them.

    Lifecycle::

        group = TPGroup(tp_rank=1, tp_size=4, peers=[...])
        group.start_listener(listen_host, listen_port)  # accept inbound
        group.connect_peers()                             # connect outbound
        ...
        group.all_reduce_sum_inplace(tensor)
        ...
        group.close()

    The split between ``start_listener`` (accept) and ``connect_peers``
    (connect) lets every rank in the group come up before anyone tries to
    talk; once both halves are bound the ring is complete.
    """

    def __init__(self, *, tp_rank: int, tp_size: int, peers: list[TPPeer]):
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(f"tp_rank {tp_rank} out of range for tp_size {tp_size}")
        if tp_size > 1 and len(peers) != tp_size - 1:
            raise ValueError(
                f"TPGroup expected {tp_size - 1} peers, got {len(peers)}",
            )
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._peers = {p.tp_rank: p for p in peers}
        self._listener: socket.socket | None = None
        self._inbound: socket.socket | None = None     # from (tp_rank - 1) mod N
        self._outbound: socket.socket | None = None    # to (tp_rank + 1) mod N

    # --------- lifecycle ------------------------------------------------

    def start_listener(self, host: str, port: int) -> None:
        """Bind+listen for the inbound peer connection."""
        if self.tp_size == 1:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.bind((host, port))
        sock.listen(1)
        self._listener = sock
        logger.info("TPGroup tp_rank=%d listening on %s:%d", self.tp_rank, host, port)

    def accept_inbound(self) -> None:
        if self.tp_size == 1:
            return
        if self._listener is None:
            raise RuntimeError("call start_listener() before accept_inbound()")
        sock, addr = self._listener.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(DEFAULT_SOCKET_TIMEOUT)
        self._inbound = sock
        logger.info("TPGroup tp_rank=%d inbound connected from %s", self.tp_rank, addr)

    def connect_outbound(self, *, retry_for: float = 30.0) -> None:
        """Connect to (tp_rank + 1) mod tp_size."""
        if self.tp_size == 1:
            return
        next_rank = (self.tp_rank + 1) % self.tp_size
        peer = self._peers.get(next_rank)
        if peer is None:
            raise RuntimeError(f"missing peer for tp_rank={next_rank}")
        deadline = time.monotonic() + retry_for
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(max(0.5, deadline - time.monotonic()))
            try:
                sock.connect((peer.host, peer.port))
                sock.settimeout(DEFAULT_SOCKET_TIMEOUT)
                self._outbound = sock
                logger.info(
                    "TPGroup tp_rank=%d outbound -> %s:%d",
                    self.tp_rank, peer.host, peer.port,
                )
                return
            except (ConnectionRefusedError, OSError) as err:
                last_err = err
                sock.close()
                time.sleep(0.5)
        raise TimeoutError(
            f"TPGroup tp_rank={self.tp_rank} could not connect to "
            f"tp_rank={next_rank} at {peer.host}:{peer.port} within "
            f"{retry_for}s",
        ) from last_err

    def close(self) -> None:
        for s in (self._inbound, self._outbound, self._listener):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._inbound = self._outbound = self._listener = None

    # --------- collectives ---------------------------------------------

    def all_reduce_sum_inplace(self, tensor: np.ndarray) -> np.ndarray:
        """Sum ``tensor`` across all TP ranks; result lives on every rank.

        Implementation: ring reduce-scatter then ring all-gather. For
        tp_size == 1 this is a no-op (caller may pass through).
        """
        if self.tp_size == 1:
            return tensor
        if tensor.dtype.type not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"TPGroup all_reduce only supports float16/float32, got {tensor.dtype}",
            )
        if self._inbound is None or self._outbound is None:
            raise RuntimeError("TPGroup ring not connected; call accept/connect first")

        flat = np.ascontiguousarray(tensor).reshape(-1)
        n = flat.size
        chunks = _split_chunks(n, self.tp_size)

        # ---- reduce-scatter: tp_size - 1 rounds ----
        # At round r, send chunk owned by (rank - r) mod N to next; receive
        # chunk owned by (rank - r - 1) mod N from prev; sum into local.
        for r in range(self.tp_size - 1):
            send_idx = (self.tp_rank - r) % self.tp_size
            recv_idx = (self.tp_rank - r - 1) % self.tp_size
            send_slice = flat[chunks[send_idx][0]:chunks[send_idx][1]]
            recv_count = chunks[recv_idx][1] - chunks[recv_idx][0]
            _send_chunk(self._outbound, send_slice)
            inc = _recv_chunk(self._inbound, flat.dtype.type, recv_count)
            flat[chunks[recv_idx][0]:chunks[recv_idx][1]] += inc

        # After reduce-scatter, rank `r` holds the final value of chunk
        # `(r + 1) mod N` (the chunk it received-and-summed in the last round).
        # For tp_size == 2 the math gives chunk owned by (rank - 1) mod 2 ==
        # the OTHER rank, which is correct.

        # ---- all-gather: tp_size - 1 rounds ----
        for r in range(self.tp_size - 1):
            send_idx = (self.tp_rank - r + 1) % self.tp_size
            recv_idx = (self.tp_rank - r) % self.tp_size
            _send_chunk(self._outbound, flat[chunks[send_idx][0]:chunks[send_idx][1]])
            recv_count = chunks[recv_idx][1] - chunks[recv_idx][0]
            inc = _recv_chunk(self._inbound, flat.dtype.type, recv_count)
            flat[chunks[recv_idx][0]:chunks[recv_idx][1]] = inc

        return flat.reshape(tensor.shape)


def _split_chunks(n: int, tp_size: int) -> list[tuple[int, int]]:
    """Even split of n elements into tp_size chunks; trailing chunk takes the
    remainder. Returns [(start, end), ...]."""
    base = n // tp_size
    chunks: list[tuple[int, int]] = []
    cursor = 0
    for i in range(tp_size):
        size = base if i < tp_size - 1 else n - cursor
        chunks.append((cursor, cursor + size))
        cursor += size
    return chunks


def all_reduce_sum_inplace(group: TPGroup, tensor: np.ndarray) -> np.ndarray:
    """Functional alias for ``group.all_reduce_sum_inplace(tensor)``."""
    return group.all_reduce_sum_inplace(tensor)
