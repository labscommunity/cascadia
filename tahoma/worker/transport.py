"""TCP-based activation tensor relay between pipeline stages.

Wire format::

    [4B payload_len BE] [4B dtype_code BE] [4B dim0 BE] [4B dim1 BE] [4B dim2 BE]
    [payload_len bytes raw tensor data]

Tensors up to 3D are supported. Lower-rank tensors are padded with leading-1
dimensions; the receiver delivers the wire-encoded shape.

This is intentionally simple — raw TCP, point-to-point, blocking. It is the
data plane between adjacent pipeline stages. Control-plane messages travel
over the routing module.
"""

from __future__ import annotations

import logging
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DTYPE_MAP: dict[int, type] = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.int32,
    4: np.int64,
}
DTYPE_REVERSE: dict[type, int] = {v: k for k, v in DTYPE_MAP.items()}

HEADER_SIZE = 20  # 5 × uint32_be
RECV_BUFFER = 65_536
MAX_RANK = 3

# Default timeout for blocking send / recv after a connection is established.
# A single tensor transfer over LAN is sub-second; over a degraded link,
# 60 s gives the peer plenty of time to flush. Beyond that we assume the peer
# is dead and surface a TimeoutError so the caller can clean up rather than
# hanging the pipeline indefinitely.
DEFAULT_SOCKET_TIMEOUT = 60.0


@dataclass
class TransferStats:
    """Timing and byte count for a single send or recv."""

    elapsed_ms: float = 0.0
    bytes: int = 0


def send_tensor(sock: socket.socket, tensor: np.ndarray) -> TransferStats:
    """Send an activation tensor over a connected TCP socket."""
    if tensor.ndim > MAX_RANK:
        raise ValueError(
            f"tensor rank > {MAX_RANK} not supported (got shape {tensor.shape})"
        )

    tensor = np.ascontiguousarray(tensor)
    dtype_code = DTYPE_REVERSE.get(tensor.dtype.type, 0)
    data = tensor.tobytes()

    shape = list(tensor.shape)
    while len(shape) < MAX_RANK:
        shape.insert(0, 1)

    header = struct.pack(">IIIII", len(data), dtype_code, shape[0], shape[1], shape[2])

    start = time.perf_counter()
    sock.sendall(header + data)
    return TransferStats(
        elapsed_ms=(time.perf_counter() - start) * 1000,
        bytes=len(header) + len(data),
    )


def recv_tensor(sock: socket.socket) -> tuple[np.ndarray, TransferStats]:
    """Receive an activation tensor from a connected TCP socket."""
    start = time.perf_counter()
    header = _recv_exact(sock, HEADER_SIZE)
    payload_len, dtype_code, d0, d1, d2 = struct.unpack(">IIIII", header)
    payload = _recv_exact(sock, payload_len)

    elapsed_ms = (time.perf_counter() - start) * 1000
    dtype = DTYPE_MAP.get(dtype_code, np.float32)
    tensor = np.frombuffer(payload, dtype=dtype).reshape(d0, d1, d2)
    return tensor, TransferStats(elapsed_ms=elapsed_ms, bytes=HEADER_SIZE + payload_len)


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < num_bytes:
        chunk = sock.recv(min(RECV_BUFFER, num_bytes - received))
        if not chunk:
            raise ConnectionError("socket closed during recv")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class ActivationServer:
    """TCP server that receives activations from the upstream pipeline stage."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9100):
        self.host = host
        self.port = port
        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        # Update self.port with the OS-assigned port if caller passed 0.
        self.port = sock.getsockname()[1]
        self._server_sock = sock
        logger.info("ActivationServer listening on %s:%d", self.host, self.port)

    def accept(self) -> None:
        if self._server_sock is None:
            raise RuntimeError("call start() before accept()")
        sock, addr = self._server_sock.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(DEFAULT_SOCKET_TIMEOUT)
        self._client_sock = sock
        logger.info("ActivationServer accepted connection from %s", addr)

    def recv(self) -> tuple[np.ndarray, TransferStats]:
        if self._client_sock is None:
            raise RuntimeError("call accept() before recv()")
        return recv_tensor(self._client_sock)

    def send(self, tensor: np.ndarray) -> TransferStats:
        """Send a tensor back upstream (used by the last stage to return tokens)."""
        if self._client_sock is None:
            raise RuntimeError("call accept() before send()")
        return send_tensor(self._client_sock, tensor)

    def close(self) -> None:
        if self._client_sock is not None:
            self._client_sock.close()
            self._client_sock = None
        if self._server_sock is not None:
            self._server_sock.close()
            self._server_sock = None


class ActivationClient:
    """TCP client that sends activations to the downstream pipeline stage."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None

    def connect(self, timeout: float = 30.0) -> None:
        """Connect to the downstream stage. Retries until `timeout` elapses."""
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None

        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(max(0.5, deadline - time.monotonic()))
            try:
                sock.connect((self.host, self.port))
                sock.settimeout(DEFAULT_SOCKET_TIMEOUT)
                self._sock = sock
                logger.info("ActivationClient connected to %s:%d", self.host, self.port)
                return
            except (ConnectionRefusedError, OSError) as err:
                last_err = err
                sock.close()
                time.sleep(0.5)

        raise TimeoutError(
            f"could not connect to {self.host}:{self.port} within {timeout}s"
        ) from last_err

    def send(self, tensor: np.ndarray) -> TransferStats:
        if self._sock is None:
            raise RuntimeError("call connect() before send()")
        return send_tensor(self._sock, tensor)

    def recv(self) -> tuple[np.ndarray, TransferStats]:
        """Receive a tensor back from downstream (last-stage token return)."""
        if self._sock is None:
            raise RuntimeError("call connect() before recv()")
        return recv_tensor(self._sock)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
