"""Speculative-decoding engine using rainier's mask-based-rewind path.

Bypasses optimum-intel and transformers' `_assisted_decoding` (which is
incompatible with optimum-intel 1.27's tuple-format past_key_values), going
directly to OpenVINO Runtime + the manual `spec_decode_greedy` loop.

Single-stage. Loads two pre-exported OV IR models — a target (the model
you actually want output from) and a small draft. They must share a
tokenizer.

Per rainier DISCOVERY #20: 1.36x at K=3, 1.55x at long context.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId
from tahoma.worker.engines.base import Builder, Engine
from tahoma.worker.engines.openvino.optimum_engine import resolve_or_export_ov_ir
from tahoma.worker.engines.openvino.spec_decode import (
    MaskedReq,
    make_masked_req,
    spec_decode_greedy,
)

logger = logging.getLogger(__name__)


@dataclass
class _Active:
    task: GenerationTask
    tokens: list[int]
    stats: Any


class OVSpecDecodeEngine(Engine):
    """Single-stage spec-decode engine.

    Greedy only. Yields one chunk with the full completion (not per-token
    streaming) — since spec_decode_greedy runs to max_tokens before returning,
    streaming would require a non-trivial refactor.
    """

    def __init__(
        self,
        target: MaskedReq,
        draft: MaskedReq,
        tokenizer: Any,
        k: int = 3,
    ):
        self._target = target
        self._draft = draft
        self._tokenizer = tokenizer
        self._k = k
        self._active: dict[TaskId, _Active] = {}
        self._pending: list[GenerationTask] = []

    def warmup(self) -> None:
        try:
            ids = self._tokenizer("Hi", return_tensors="np").input_ids.astype("int64")
            spec_decode_greedy(self._target, self._draft, ids, max_tokens=2, k=self._k)
            logger.info("warmup ok (spec_decode K=%d)", self._k)
        except Exception as err:  # noqa: BLE001
            logger.warning("warmup failed: %s", err)

    def submit(self, task: GenerationTask) -> None:
        if task.task_id in self._active or any(t.task_id == task.task_id for t in self._pending):
            return
        self._pending.append(task)

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        if not self._active and self._pending:
            task = self._pending.pop(0)
            ids = self._tokenizer(task.prompt, return_tensors="np").input_ids.astype("int64")
            tokens, stats = spec_decode_greedy(
                self._target, self._draft, ids,
                max_tokens=task.max_tokens, k=self._k,
            )
            text = self._tokenizer.decode(tokens, skip_special_tokens=True)
            logger.info(
                "task %s: %d tokens, %d steps, accept_rate=%.2f",
                task.task_id[:8], len(tokens), stats.n_steps, stats.accept_rate,
            )
            yield task.task_id, Chunk(
                task_id=task.task_id,
                token_id=tokens[-1] if tokens else 0,
                text=text,
                is_final=True,
            )

    def close(self) -> None:
        pass


class OVSpecDecodeBuilder(Builder):
    """Builder for `OVSpecDecodeEngine`. Loads target + draft + tokenizer."""

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        device: str = "GPU",
        weight_format: str = "int4",
        draft_weight_format: str = "int4",
        k: int = 3,
    ):
        self._model_path = model_path
        self._draft_model_path = draft_model_path
        self._device = device
        self._weight_format = weight_format
        self._draft_weight_format = draft_weight_format
        self._k = k
        self._target: MaskedReq | None = None
        self._draft: MaskedReq | None = None
        self._tokenizer: Any = None

    def connect(self, peers: PeerLayout) -> None:
        if peers.upstream is not None or peers.downstream is not None:
            raise RuntimeError(
                "OVSpecDecodeEngine is single-stage only; do not configure peers"
            )

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        import openvino as ov  # type: ignore[import-untyped]
        from transformers import AutoTokenizer

        if not (shard.is_first_stage and shard.is_last_stage):
            raise RuntimeError("OVSpecDecodeEngine requires --total 1")

        yield LoadProgress(0, None, f"resolving target {self._model_path}")
        target_path = resolve_or_export_ov_ir(
            self._model_path, weight_format=self._weight_format,
        )

        yield LoadProgress(0, None, f"resolving draft {self._draft_model_path}")
        draft_path = resolve_or_export_ov_ir(
            self._draft_model_path, weight_format=self._draft_weight_format,
        )

        core = ov.Core()

        yield LoadProgress(0, None, "compiling target IR")
        target_compiled = core.compile_model(f"{target_path}/openvino_model.xml", self._device)
        self._target = make_masked_req(target_compiled)

        yield LoadProgress(0, None, "compiling draft IR")
        draft_compiled = core.compile_model(f"{draft_path}/openvino_model.xml", self._device)
        self._draft = make_masked_req(draft_compiled)

        yield LoadProgress(0, None, "loading tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(target_path)

        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._target is None or self._draft is None or self._tokenizer is None:
            raise RuntimeError("call load() before build()")
        return OVSpecDecodeEngine(self._target, self._draft, self._tokenizer, k=self._k)

    def close(self) -> None:
        self._target = None
        self._draft = None
        self._tokenizer = None
