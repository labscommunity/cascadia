"""Single-stage OpenVINO Runtime engine via optimum-intel.

Wraps `optimum.intel.OVModelForCausalLM` for fast 1-stage inference using
pre-exported OpenVINO IR (typically INT4 quantized via
`optimum-cli export openvino --weight-format int4 ...`).

Limitations:
- **Single-stage only.** The model runs in one process; no cross-node
  pipeline parallelism. Use the PyTorch `OpenVINOEngine` for distributed.
- One chunk per task (not per token). The chunk's `text` carries the full
  completion; `is_final=True` on the only chunk. Streaming is a follow-up.

Why this exists: OV Runtime + INT4 + OV's KV cache + iGPU graph compilation
benchmarks ~4× faster than the PyTorch fp16 + manual KV cache path on the
same hardware (15.6 tok/s vs 4.0 tok/s on Lunar Lake Arc 140V for Llama-3.1-8B).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId
from tahoma.worker.engines.base import Builder, Engine

logger = logging.getLogger(__name__)


class OptimumOVEngine(Engine):
    """1-stage engine wrapping `optimum.intel.OVModelForCausalLM`."""

    def __init__(self, model: Any, tokenizer: Any):
        self._model = model
        self._tokenizer = tokenizer
        self._active: dict[TaskId, GenerationTask] = {}
        self._pending: list[GenerationTask] = []

    def warmup(self) -> None:
        try:
            ids = self._tokenizer("Hi", return_tensors="pt")
            self._model.generate(
                **ids,
                max_new_tokens=1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            logger.info("warmup ok")
        except Exception as err:  # noqa: BLE001
            logger.warning("warmup failed: %s", err)

    def submit(self, task: GenerationTask) -> None:
        if task.task_id in self._active or any(t.task_id == task.task_id for t in self._pending):
            return
        self._pending.append(task)

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        if not self._active and self._pending:
            task = self._pending.pop(0)
            self._active[task.task_id] = task
            logger.info("task %s active", task.task_id[:8])

        if not self._active:
            return

        task_id, task = next(iter(self._active.items()))
        ids = self._tokenizer(task.prompt, return_tensors="pt")
        prompt_len = ids.input_ids.shape[1]

        do_sample = task.temperature > 0
        out = self._model.generate(
            **ids,
            max_new_tokens=task.max_tokens,
            do_sample=do_sample,
            temperature=task.temperature if do_sample else 1.0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_ids = out[0][prompt_len:].tolist()
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        last_token = int(new_ids[-1]) if new_ids else 0
        yield task_id, Chunk(
            task_id=task_id,
            token_id=last_token,
            text=text,
            is_final=True,
        )
        del self._active[task_id]

    def close(self) -> None:
        pass


class OptimumOVBuilder(Builder):
    """Builder for `OptimumOVEngine`. Loads pre-exported OV IR + tokenizer."""

    def __init__(self, model_path: str, device: str = "GPU"):
        self._model_path = model_path
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def connect(self, peers: PeerLayout) -> None:
        if peers.upstream is not None or peers.downstream is not None:
            raise RuntimeError(
                "OptimumOVEngine is single-stage only; do not configure peers"
            )

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        from optimum.intel import OVModelForCausalLM  # type: ignore[import-untyped]
        from transformers import AutoTokenizer

        if not (shard.is_first_stage and shard.is_last_stage):
            raise RuntimeError(
                "OptimumOVEngine requires a single-stage shard "
                "(set --total 1)"
            )

        yield LoadProgress(0, None, "compiling OV model")
        self._model = OVModelForCausalLM.from_pretrained(
            self._model_path,
            device=self._device,
            export=False,
        )
        yield LoadProgress(0, None, "loading tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("call load() before build()")
        return OptimumOVEngine(self._model, self._tokenizer)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
