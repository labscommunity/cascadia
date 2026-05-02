"""TPGroup ring all-reduce — verify correctness against numpy ground truth."""

from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from tahoma.parallel import TPGroup
from tahoma.parallel.group import TPPeer, _split_chunks


def _peer_addrs(tp_size: int) -> list[int]:
    """Allocate `tp_size` unique ephemeral ports.

    Sequentially closing one ephemeral socket before opening the next
    lets the kernel hand the same port to two consecutive calls under
    load (and once intermittently caused EADDRINUSE in CI py3.12).
    Hold every discovery socket open simultaneously so the kernel
    guarantees uniqueness, then close them all together. The race
    window between close-and-rebind still exists but is dramatically
    narrowed and applies only to inter-process collisions.
    """
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(tp_size)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        ports = [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()
    return ports


def _peers_for(rank: int, tp_size: int, ports: list[int]) -> list[TPPeer]:
    """Every rank knows the address of every OTHER rank."""
    return [
        TPPeer(tp_rank=r, host="127.0.0.1", port=ports[r])
        for r in range(tp_size) if r != rank
    ]


def _run_group(rank: int, tp_size: int, ports: list[int],
               tensors: list[np.ndarray], results: dict, errors: dict) -> None:
    """Per-rank ring lifecycle + a single all_reduce_sum_inplace."""
    g = TPGroup(
        tp_rank=rank, tp_size=tp_size,
        peers=_peers_for(rank, tp_size, ports),
    )
    try:
        g.start_listener("127.0.0.1", ports[rank])
        # Symmetric handshake: every rank accepts inbound and connects
        # outbound concurrently. We launch the accept on its own thread so
        # connect_outbound() can proceed in parallel.
        accept_thread = threading.Thread(target=g.accept_inbound, daemon=True)
        accept_thread.start()
        g.connect_outbound(retry_for=10.0)
        accept_thread.join(timeout=10.0)
        assert not accept_thread.is_alive(), f"rank {rank} accept stalled"

        out = g.all_reduce_sum_inplace(tensors[rank].copy())
        results[rank] = out
    except Exception as err:  # noqa: BLE001
        errors[rank] = err
    finally:
        g.close()


@pytest.mark.parametrize("tp_size", [2, 3, 4])
def test_all_reduce_sum_matches_numpy(tp_size: int) -> None:
    n = 64
    ports = _peer_addrs(tp_size)
    rng = np.random.default_rng(seed=tp_size)
    tensors = [rng.standard_normal(n).astype(np.float32) for _ in range(tp_size)]
    expected = sum(tensors)

    results: dict = {}
    errors: dict = {}
    threads = [
        threading.Thread(
            target=_run_group,
            args=(rank, tp_size, ports, tensors, results, errors),
            daemon=True,
        )
        for rank in range(tp_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert not errors, f"rank errors: {errors}"
    for rank in range(tp_size):
        np.testing.assert_allclose(
            results[rank], expected, rtol=1e-5, atol=1e-5,
            err_msg=f"rank {rank} all-reduce result wrong",
        )


def test_all_reduce_fp16_is_supported() -> None:
    tp_size = 2
    n = 16
    ports = _peer_addrs(tp_size)
    tensors = [
        np.full(n, 1.0, dtype=np.float16),
        np.full(n, 2.0, dtype=np.float16),
    ]
    expected = np.full(n, 3.0, dtype=np.float16)
    results: dict = {}
    errors: dict = {}
    threads = [
        threading.Thread(
            target=_run_group,
            args=(rank, tp_size, ports, tensors, results, errors),
            daemon=True,
        )
        for rank in range(tp_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)
    assert not errors
    for rank in range(tp_size):
        np.testing.assert_array_equal(results[rank], expected)


def test_tp_size_1_is_no_op() -> None:
    g = TPGroup(tp_rank=0, tp_size=1, peers=[])
    arr = np.arange(8, dtype=np.float32)
    out = g.all_reduce_sum_inplace(arr)
    np.testing.assert_array_equal(out, arr)


def test_unsupported_dtype_raises() -> None:
    # Two ranks; build a real ring first, then fail on the int call.
    tp_size = 2
    ports = _peer_addrs(tp_size)
    rngs = [np.array([1, 2, 3], dtype=np.int32) for _ in range(tp_size)]

    def runner(rank: int, ports: list[int], errors: dict) -> None:
        g = TPGroup(tp_rank=rank, tp_size=tp_size,
                    peers=_peers_for(rank, tp_size, ports))
        try:
            g.start_listener("127.0.0.1", ports[rank])
            t = threading.Thread(target=g.accept_inbound, daemon=True)
            t.start()
            g.connect_outbound(retry_for=10.0)
            t.join(timeout=10.0)
            with pytest.raises(TypeError, match="float16/float32"):
                g.all_reduce_sum_inplace(rngs[rank])
        except Exception as err:  # noqa: BLE001
            errors[rank] = err
        finally:
            g.close()

    errors: dict = {}
    threads = [
        threading.Thread(target=runner, args=(rank, ports, errors), daemon=True)
        for rank in range(tp_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
    assert not errors


def test_split_chunks_even_and_remainder() -> None:
    assert _split_chunks(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    # 9 elements / 3 ranks → 3 each
    assert _split_chunks(9, 3) == [(0, 3), (3, 6), (6, 9)]
    # 10 elements / 3 ranks → 3, 3, 4 (last takes remainder)
    assert _split_chunks(10, 3) == [(0, 3), (3, 6), (6, 10)]


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        TPGroup(tp_rank=0, tp_size=0, peers=[])
    with pytest.raises(ValueError, match="tp_rank"):
        TPGroup(tp_rank=2, tp_size=2, peers=[])
    with pytest.raises(ValueError, match="expected"):
        TPGroup(tp_rank=0, tp_size=4, peers=[])  # need 3 peers
