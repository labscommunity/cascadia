"""Tensor-parallel engine: weight slicing + all-reduce wiring.

These tests exercise the building blocks (column/row slicing, attention/MLP
patching, end-to-end correctness on a tiny model) on CPU. The full multi-rank
benchmark with real Llama weights runs separately on the alpha+charlie cluster.
"""

from __future__ import annotations

import socket
import threading

import numpy as np
import pytest
import torch
import torch.nn as nn

from tahoma.parallel import TPGroup
from tahoma.parallel.group import TPPeer
from tahoma.worker.engines.openvino.tp_engine import (
    _slice_linear_columns,
    _slice_linear_rows,
    _wrap_forward_with_allreduce,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Pure-python slicing
# ---------------------------------------------------------------------------


def test_column_split_concatenated_equals_original() -> None:
    full = nn.Linear(4, 8, bias=True)
    full.weight.data = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    full.bias.data = torch.arange(8, dtype=torch.float32)

    rank0 = _slice_linear_columns(full, tp_rank=0, tp_size=2)
    rank1 = _slice_linear_columns(full, tp_rank=1, tp_size=2)
    assert rank0.weight.shape == (4, 4)
    assert rank1.weight.shape == (4, 4)
    # Concatenating the per-rank weight slices reproduces the original.
    cat = torch.cat([rank0.weight.data, rank1.weight.data], dim=0)
    torch.testing.assert_close(cat, full.weight.data)
    cat_b = torch.cat([rank0.bias.data, rank1.bias.data], dim=0)
    torch.testing.assert_close(cat_b, full.bias.data)


def test_row_split_partial_outputs_sum_to_original() -> None:
    full = nn.Linear(8, 4, bias=False)
    full.weight.data = torch.randn(4, 8)
    x = torch.randn(2, 8)
    expected = full(x)

    rank0 = _slice_linear_rows(full, tp_rank=0, tp_size=2)
    rank1 = _slice_linear_rows(full, tp_rank=1, tp_size=2)
    # Each rank receives only its portion of the input dim. Sum reproduces.
    out0 = rank0(x[:, :4])
    out1 = rank1(x[:, 4:])
    torch.testing.assert_close(out0 + out1, expected, atol=1e-5, rtol=1e-5)


def test_row_split_bias_only_on_rank_zero() -> None:
    full = nn.Linear(4, 4, bias=True)
    full.bias.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rank0 = _slice_linear_rows(full, tp_rank=0, tp_size=2)
    rank1 = _slice_linear_rows(full, tp_rank=1, tp_size=2)
    torch.testing.assert_close(rank0.bias.data, full.bias.data)
    torch.testing.assert_close(rank1.bias.data, torch.zeros(4))


def test_column_split_rejects_non_divisible_dim() -> None:
    full = nn.Linear(4, 5, bias=False)
    with pytest.raises(ValueError, match="not divisible"):
        _slice_linear_columns(full, tp_rank=0, tp_size=2)


def test_row_split_rejects_non_divisible_dim() -> None:
    full = nn.Linear(5, 4, bias=False)
    with pytest.raises(ValueError, match="not divisible"):
        _slice_linear_rows(full, tp_rank=0, tp_size=2)


# ---------------------------------------------------------------------------
# All-reduce hook on a real torch module — end-to-end correctness for a
# tiny "TP-row-split" Linear layer driven by a real TP group.
# ---------------------------------------------------------------------------


def _run_one_rank(
    rank: int, ports: list[int],
    weight_full: torch.Tensor, x: torch.Tensor,
    results: dict, errors: dict,
) -> None:
    tp_size = len(ports)
    peers = [TPPeer(tp_rank=r, host="127.0.0.1", port=ports[r])
             for r in range(tp_size) if r != rank]
    g = TPGroup(tp_rank=rank, tp_size=tp_size, peers=peers)
    try:
        g.start_listener("127.0.0.1", ports[rank])
        t = threading.Thread(target=g.accept_inbound, daemon=True)
        t.start()
        g.connect_outbound(retry_for=10.0)
        t.join(timeout=10.0)
        assert not t.is_alive()

        # Build a row-split Linear for THIS rank.
        out_dim, in_dim = weight_full.shape
        chunk = in_dim // tp_size
        my = nn.Linear(chunk, out_dim, bias=False)
        my.weight.data = weight_full[:, rank * chunk:(rank + 1) * chunk].clone()
        _wrap_forward_with_allreduce(my, g)

        # Each rank only sees its slice of the input.
        my_x = x[:, rank * chunk:(rank + 1) * chunk]
        with torch.no_grad():
            out = my(my_x)
        results[rank] = out.detach().cpu().numpy()
    except Exception as err:  # noqa: BLE001
        errors[rank] = err
    finally:
        g.close()


def test_allreduce_hook_reproduces_full_linear() -> None:
    """A row-split Linear with the all-reduce hook should match the full Linear."""
    tp_size = 2
    in_dim = 16
    out_dim = 8
    weight_full = torch.randn(out_dim, in_dim, dtype=torch.float16)
    full = nn.Linear(in_dim, out_dim, bias=False)
    full.weight.data = weight_full.clone()
    x = torch.randn(2, in_dim, dtype=torch.float16)
    with torch.no_grad():
        expected = full(x)

    ports = [_free_port() for _ in range(tp_size)]
    results: dict = {}
    errors: dict = {}
    threads = [
        threading.Thread(
            target=_run_one_rank,
            args=(rank, ports, weight_full, x, results, errors),
            daemon=True,
        )
        for rank in range(tp_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert not errors, f"rank errors: {errors}"
    # Every rank must end up with the SAME (all-reduced) value, equal to the
    # full Linear's output.
    expected_np = expected.detach().cpu().numpy().astype(np.float16)
    for rank in range(tp_size):
        np.testing.assert_allclose(
            results[rank], expected_np, rtol=2e-3, atol=2e-3,
            err_msg=f"rank {rank} mismatch",
        )
