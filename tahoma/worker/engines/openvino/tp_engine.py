"""Tensor-parallel PyTorch engine.

Each TP rank holds 1/tp_size of the attention heads and 1/tp_size of every
MLP projection. After the attention output projection and after the MLP
down projection we run an all-reduce-sum across the TP group to combine
partial outputs.

Layout (per Llama-family decoder layer):

    Q/K/V proj: column-split  → each rank holds ``num_heads / tp_size`` heads
    o_proj:    row-split     → each rank's output is a partial; sum to
                                reconstruct the full hidden_states
    gate/up:   column-split  → each rank holds intermediate / tp_size cols
    down_proj: row-split     → each rank's output is a partial; sum to
                                reconstruct hidden_states

Constraints
-----------
- ``num_attention_heads`` and ``num_key_value_heads`` (GQA) must both be
  divisible by ``tp_size``. For Llama 3.1 8B: 32 attn / 8 kv → tp_size in
  {2, 4, 8}. For Llama 3.2 1B: 32 attn / 8 kv → same.
- ``intermediate_size`` must be divisible by ``tp_size``.
- We use TCP collectives (:mod:`tahoma.parallel`); the cost dominates on
  slow networks. A 10 GbE / Thunderbolt link is the floor for TP to be
  worth it on small models; for large MLPs it amortises over compute.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from tahoma.parallel.group import TPGroup, TPPeer
from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import LoadProgress
from tahoma.worker.engines.base import Builder, Engine
from tahoma.worker.engines.openvino.engine import OpenVINOEngine, _ActiveTask
from tahoma.worker.engines.openvino.loader import ModelShard, _torch_device
from tahoma.worker.transport import ActivationClient, ActivationServer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight slicing + module patching
# ---------------------------------------------------------------------------


def _slice_linear_columns(linear: nn.Linear, tp_rank: int, tp_size: int) -> nn.Linear:
    """Column-parallel: keep slice [tp_rank*chunk : (tp_rank+1)*chunk] of OUT dim."""
    out, in_ = linear.weight.shape
    if out % tp_size != 0:
        raise ValueError(
            f"column-split: output dim {out} not divisible by tp_size {tp_size}",
        )
    chunk = out // tp_size
    new = nn.Linear(in_, chunk, bias=linear.bias is not None)
    new.weight.data = linear.weight.data[tp_rank * chunk:(tp_rank + 1) * chunk, :].clone()
    if linear.bias is not None:
        new.bias.data = linear.bias.data[tp_rank * chunk:(tp_rank + 1) * chunk].clone()
    return new


def _slice_linear_rows(linear: nn.Linear, tp_rank: int, tp_size: int) -> nn.Linear:
    """Row-parallel: keep slice [tp_rank*chunk : (tp_rank+1)*chunk] of IN dim.

    Bias is NOT split — bias adds once per rank, so we keep the full bias on
    rank 0 only and zero it elsewhere (so all-reduce sums to the original).
    Most decoder LMs have bias=False here so it rarely matters in practice.
    """
    out, in_ = linear.weight.shape
    if in_ % tp_size != 0:
        raise ValueError(
            f"row-split: input dim {in_} not divisible by tp_size {tp_size}",
        )
    chunk = in_ // tp_size
    new = nn.Linear(chunk, out, bias=linear.bias is not None)
    new.weight.data = linear.weight.data[:, tp_rank * chunk:(tp_rank + 1) * chunk].clone()
    if linear.bias is not None:
        if tp_rank == 0:
            new.bias.data = linear.bias.data.clone()
        else:
            new.bias.data.zero_()
    return new


def _wrap_forward_with_allreduce(linear: nn.Linear, tp_group: TPGroup) -> None:
    """Wrap ``linear.forward`` so the all-reduce sum runs on its output.

    Forward hooks were unreliable for this codepath — PyTorch can ignore the
    hook's return value when the consuming op was already inlined (transformers
    eager_attention_forward stores the o_proj result in a local before
    returning, which captures the pre-hook value). Monkey-patching the
    instance's ``forward`` is the unambiguous fix: callers see the all-reduced
    tensor as the natural return of ``linear(x)``.
    """
    original_forward = linear.forward

    def patched(x: torch.Tensor) -> torch.Tensor:
        out = original_forward(x)
        device = out.device
        dtype = out.dtype
        np_out = out.detach().to(torch.float16).cpu().contiguous().numpy()
        # Copy so we don't write into a buffer that the next op might still hold.
        np_out = np.ascontiguousarray(np_out).copy()
        reduced = tp_group.all_reduce_sum_inplace(np_out)
        return torch.from_numpy(reduced.copy()).to(device=device, dtype=dtype)

    linear.forward = patched  # type: ignore[method-assign]


def _patch_attention(attn: nn.Module, tp_group: TPGroup, tp_rank: int, tp_size: int) -> None:
    """In-place patch: slice Q/K/V/O projections + register all-reduce hook.

    Updates the attention module's ``num_heads``/``num_key_value_heads``
    attributes so the existing forward path reshapes correctly after slicing.
    """
    full_heads = attn.config.num_attention_heads
    full_kv_heads = getattr(attn.config, "num_key_value_heads", full_heads)
    if full_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads ({full_heads}) not divisible by tp_size ({tp_size})",
        )
    if full_kv_heads % tp_size != 0:
        raise ValueError(
            f"num_key_value_heads ({full_kv_heads}) not divisible by tp_size ({tp_size})",
        )

    attn.q_proj = _slice_linear_columns(attn.q_proj, tp_rank, tp_size)
    attn.k_proj = _slice_linear_columns(attn.k_proj, tp_rank, tp_size)
    attn.v_proj = _slice_linear_columns(attn.v_proj, tp_rank, tp_size)
    attn.o_proj = _slice_linear_rows(attn.o_proj, tp_rank, tp_size)
    # Inform the attention module that it now owns fewer heads. Without this
    # the reshape inside forward would explode.
    if hasattr(attn, "num_heads"):
        attn.num_heads = full_heads // tp_size
    if hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = full_kv_heads // tp_size
    if hasattr(attn, "hidden_size"):
        attn.hidden_size = (full_heads // tp_size) * attn.head_dim
    _wrap_forward_with_allreduce(attn.o_proj, tp_group)


def _patch_mlp(mlp: nn.Module, tp_group: TPGroup, tp_rank: int, tp_size: int) -> None:
    """In-place patch: column-split gate/up, row-split down + all-reduce hook."""
    if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj")):
        # Fallback for unusual MLP shapes — we don't try to be clever.
        raise RuntimeError(
            f"TP MLP patching requires gate/up/down_proj on the MLP "
            f"(got {type(mlp).__name__})",
        )
    mlp.gate_proj = _slice_linear_columns(mlp.gate_proj, tp_rank, tp_size)
    mlp.up_proj = _slice_linear_columns(mlp.up_proj, tp_rank, tp_size)
    mlp.down_proj = _slice_linear_rows(mlp.down_proj, tp_rank, tp_size)
    _wrap_forward_with_allreduce(mlp.down_proj, tp_group)


# ---------------------------------------------------------------------------
# TP-aware ModelShard
# ---------------------------------------------------------------------------


class TPModelShard(ModelShard):
    """Variant of ``ModelShard`` that slices each decoder layer's attention
    and MLP weights to this rank's portion and registers all-reduce hooks.

    Only the layers we own are sliced. Embedding + lm_head are NOT split
    (they're held identically on every rank — the tradeoff is correctness
    vs an extra ``vocab/tp_size`` shard per rank, which we don't bother with
    for a v0 TP implementation).
    """

    def __init__(self, spec: ShardSpec, model_path: str, tp_group: TPGroup):
        super().__init__(spec, model_path)
        self._tp_group = tp_group

    def _build_components(self, state_dict: dict[str, torch.Tensor]) -> None:
        super()._build_components(state_dict)
        if self._tp_group.tp_size == 1 or self._layers is None:
            return
        torch_dev = _torch_device(self.spec.device)
        for layer in self._layers:
            _patch_attention(
                layer.self_attn, self._tp_group,
                self.spec.tp_rank, self.spec.tp_size,
            )
            _patch_mlp(
                layer.mlp, self._tp_group,
                self.spec.tp_rank, self.spec.tp_size,
            )
            layer.eval()
            layer.half()
            layer.to(torch_dev)
        logger.info(
            "TP: sliced %d layers for tp_rank=%d/%d",
            len(self._layers), self.spec.tp_rank, self.spec.tp_size,
        )


# ---------------------------------------------------------------------------
# Engine + Builder
# ---------------------------------------------------------------------------


class PyTorchTPEngine(OpenVINOEngine):
    """OpenVINOEngine variant that owns a TPGroup.

    Inherits the entire pipeline-stage step machinery from ``OpenVINOEngine``;
    the TP work happens inside the patched modules, transparent to the
    surrounding orchestration code.
    """

    def __init__(
        self,
        spec: ShardSpec,
        shard: TPModelShard,
        tp_group: TPGroup,
        upstream_server: ActivationServer | None,
        downstream_client: ActivationClient | None,
    ):
        super().__init__(
            spec=spec, shard=shard,
            upstream_server=upstream_server,
            downstream_client=downstream_client,
        )
        self._tp_group = tp_group

    def close(self) -> None:
        super().close()
        if self._tp_group is not None:
            self._tp_group.close()


class PyTorchTPBuilder(Builder):
    """Pipeline + TP. ``--tp-size N`` requires N peers in the TP group, each
    owning a 1/N slice of every weight matrix in this stage.

    The builder owns three lifecycles: pipeline upstream/downstream, TP-group
    inbound/outbound, and the model shard itself. Each opens in the order
    that avoids deadlock — pipeline first (because the rest of the cluster
    expects it bound), TP next (peers might still be coming up).
    """

    def __init__(
        self,
        model_path: str,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_listen: tuple[str, int] | None = None,
        tp_peers: list[TPPeer] | None = None,
    ):
        self._model_path = model_path
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._tp_listen = tp_listen
        self._tp_peers = tp_peers or []
        self._spec: ShardSpec | None = None
        self._shard: TPModelShard | None = None
        self._tp_group: TPGroup | None = None
        self._upstream: ActivationServer | None = None
        self._downstream: ActivationClient | None = None
        self._listen_host = "0.0.0.0"
        self._listen_port: int | None = None

    def configure_listen(self, host: str, port: int) -> None:
        self._listen_host = host
        self._listen_port = port

    def connect(self, peers: PeerLayout) -> None:
        # Pipeline transport (same logic as OpenVINOBuilder).
        if peers.upstream is not None:
            if self._listen_port is None:
                raise RuntimeError("configure_listen() required for non-first stages")
            server = ActivationServer(self._listen_host, self._listen_port)
            server.start()
            self._upstream = server
        if peers.downstream is not None:
            client = ActivationClient(peers.downstream.host, peers.downstream.port)
            client.connect()
            self._downstream = client
        if self._upstream is not None:
            self._upstream.accept()

        # TP transport (when tp_size > 1).
        if self._tp_size > 1:
            if self._tp_listen is None:
                raise RuntimeError(
                    "TP enabled but no --tp-listen address provided",
                )
            if len(self._tp_peers) != self._tp_size - 1:
                raise RuntimeError(
                    f"TP enabled (tp_size={self._tp_size}) but only "
                    f"{len(self._tp_peers)} peer(s) supplied; "
                    f"expected {self._tp_size - 1}",
                )
            self._tp_group = TPGroup(
                tp_rank=self._tp_rank,
                tp_size=self._tp_size,
                peers=self._tp_peers,
            )
            host, port = self._tp_listen
            self._tp_group.start_listener(host, port)
            # Concurrent accept + connect so every rank can come up in parallel.
            import threading
            t = threading.Thread(target=self._tp_group.accept_inbound, daemon=True)
            t.start()
            self._tp_group.connect_outbound(retry_for=30.0)
            t.join(timeout=30.0)
            if t.is_alive():
                raise RuntimeError("TP accept_inbound stalled — peer never connected")
            logger.info(
                "TP group ready: tp_rank=%d/%d listening on %s:%d",
                self._tp_rank, self._tp_size, host, port,
            )
        else:
            # tp_size == 1: degenerate group, no collectives.
            self._tp_group = TPGroup(tp_rank=0, tp_size=1, peers=[])

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        # Inject TP coordinates into the spec so downstream code can read them.
        if shard.tp_size != self._tp_size or shard.tp_rank != self._tp_rank:
            shard = ShardSpec(
                model_id=shard.model_id,
                layer_start=shard.layer_start,
                layer_end=shard.layer_end,
                total_layers=shard.total_layers,
                device=shard.device,
                is_first_stage=shard.is_first_stage,
                is_last_stage=shard.is_last_stage,
                tp_size=self._tp_size,
                tp_rank=self._tp_rank,
            )
        self._spec = shard
        if self._tp_group is None:
            raise RuntimeError("call connect() before load()")
        yield LoadProgress(0, None, "starting")
        self._shard = TPModelShard(spec=shard, model_path=self._model_path, tp_group=self._tp_group)
        yield LoadProgress(0, None, "loading + slicing weights")
        self._shard.load()
        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._shard is None or self._spec is None or self._tp_group is None:
            raise RuntimeError("call connect() and load() before build()")
        return PyTorchTPEngine(
            spec=self._spec,
            shard=self._shard,
            tp_group=self._tp_group,
            upstream_server=self._upstream,
            downstream_client=self._downstream,
        )

    def close(self) -> None:
        if self._upstream is not None:
            self._upstream.close()
            self._upstream = None
        if self._downstream is not None:
            self._downstream.close()
            self._downstream = None
        if self._tp_group is not None:
            self._tp_group.close()
            self._tp_group = None


# Re-export for tests / external users.
__all__ = [
    "PyTorchTPBuilder",
    "PyTorchTPEngine",
    "TPModelShard",
    "_patch_attention",
    "_patch_mlp",
    "_slice_linear_columns",
    "_slice_linear_rows",
]


# Module-level reference so the engine registry can import this without a
# circular import (registry → tp_engine → engine).
_ = _ActiveTask  # noqa: B015 (keep the import alive for linters)
