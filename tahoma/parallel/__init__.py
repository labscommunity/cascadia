"""Collective primitives for tensor parallelism.

Pipeline parallelism uses point-to-point sends over ``worker.transport``.
Tensor parallelism needs collectives — every TP peer must see every other
peer's contribution to a partial result before the next layer can run.

This module ships a ring-based all-reduce (sum) over TCP. It's intentionally
minimal: a single :class:`TPGroup` that owns ``tp_size - 1`` peer connections
and exposes ``all_reduce_sum`` for fp16 / fp32 tensors. No backend selection,
no autotuning — the goal is *correctness*, plus a baseline for the per-engine
TP integration that follows.

Production-quality TP for the OpenVINO INT4 stateful shards needs three
things this commit does NOT yet do:

1. **Re-exported shards** with column-parallel attention (``q_proj``,
   ``k_proj``, ``v_proj``, MLP gate/up split along output dim) and
   row-parallel attention/MLP outputs (split along input dim). Today's v5
   shards are not TP-split.
2. **Engine integration** that calls :meth:`TPGroup.all_reduce_sum` after
   each attention and MLP block. This is per-engine work and is gated on
   the new shards above.
3. **KV-cache sharding** by attention head across TP ranks (so each rank's
   stateful storage is 1/tp_size the size of the unsharded equivalent).

See ``docs/architecture/tensor-parallelism.md`` for the export plan.
"""

from __future__ import annotations

from tahoma.parallel.group import TPGroup, all_reduce_sum_inplace

__all__ = ["TPGroup", "all_reduce_sum_inplace"]
