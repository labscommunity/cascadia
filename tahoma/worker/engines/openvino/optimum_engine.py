"""Single-stage OpenVINO Runtime engine via optimum-intel.

Wraps `optimum.intel.OVModelForCausalLM` for fast 1-stage inference using
pre-exported OpenVINO IR (typically INT4 quantized via
`optimum-cli export openvino --weight-format int4 ...`).

Limitations:
- **Single-stage only.** The model runs in one process; no cross-node
  pipeline parallelism. Use the PyTorch `OpenVINOEngine` for distributed.

Why this exists: OV Runtime + INT4 + OV's KV cache + iGPU graph compilation
benchmarks ~4x faster than the PyTorch fp16 + manual KV cache path on the
same hardware (~17 tok/s vs ~4 tok/s on Lunar Lake Arc 140V for Llama-3.1-8B).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Thread
from typing import Any

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId
from tahoma.worker.engines.base import Builder, Engine

logger = logging.getLogger(__name__)


@dataclass
class _ActiveTask:
    task: GenerationTask
    streamer: Any                  # transformers.TextIteratorStreamer
    iterator: Any                  # iter(streamer) — keep position across step() calls
    thread: Thread
    token_count: int = 0
    finished: bool = False


class OptimumOVEngine(Engine):
    """1-stage engine wrapping `optimum.intel.OVModelForCausalLM`.

    Streams chunks one per token using `TextIteratorStreamer` running in a
    background thread. Each `step()` pulls one chunk from the streamer's
    queue (blocking briefly on the next token) and yields it.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self._model = model
        self._tokenizer = tokenizer
        self._active: dict[TaskId, _ActiveTask] = {}
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

    def _start_task(self, task: GenerationTask) -> _ActiveTask:
        from transformers import TextIteratorStreamer

        ids = self._tokenizer(task.prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        do_sample = task.temperature > 0
        kwargs = dict(
            input_ids=ids.input_ids,
            attention_mask=ids.attention_mask,
            max_new_tokens=task.max_tokens,
            do_sample=do_sample,
            temperature=task.temperature if do_sample else 1.0,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self._model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        return _ActiveTask(
            task=task, streamer=streamer, iterator=iter(streamer), thread=thread,
        )

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        # Activate next pending task if no task is active.
        if not self._active and self._pending:
            task = self._pending.pop(0)
            active = self._start_task(task)
            self._active[task.task_id] = active
            logger.info("task %s active", task.task_id[:8])

        if not self._active:
            return

        task_id, active = next(iter(self._active.items()))

        # Pull next text chunk from the streamer (blocks until next token).
        try:
            text = next(active.iterator)
        except StopIteration:
            text = ""
            active.finished = True

        if text:
            active.token_count += 1

        is_final = active.finished

        yield task_id, Chunk(
            task_id=task_id,
            token_id=0,
            text=text,
            is_final=is_final,
        )

        if is_final:
            active.thread.join(timeout=5.0)
            del self._active[task_id]

    def close(self) -> None:
        for active in self._active.values():
            for _ in active.iterator:
                pass
            active.thread.join(timeout=5.0)
        self._active.clear()


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
