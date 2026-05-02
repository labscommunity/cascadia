"""openvino_genai.LLMPipeline engine.

DISCOVERY #1 (campaign c1 in this branch's experiments/) showed that
``openvino_genai.LLMPipeline`` is **~10x** faster than the optimum-intel
``OVModelForCausalLM`` path on Intel GPU for INT4 LLMs. This engine wraps
LLMPipeline so tahoma users get that speedup for free.

What you give up vs the optimum-intel engine
--------------------------------------------
- HuggingFace-style ``model.generate`` introspection.
- Direct access to the underlying ``optimum.intel`` model object for
  custom quantisation passes.

What you get
------------
- ~10x decode speedup on Intel iGPU/dGPU for INT4 LLMs.
- Built-in PagedAttention, U8 KV cache, XMX dynamic quant — all the
  GPU-default optimisations Intel ships in OV 2024.6 → 2026.1.
- Hooks for SchedulerConfig (continuous batching, prefix caching, KV
  eviction, sparse attention) — exposed in a follow-up.
- Hooks for speculative decoding via ``draft_model=`` — exposed in a
  follow-up once we have an LLMPipeline-compatible draft IR.

Streaming
---------
This v0 yields ONE chunk per task with the full text + ``is_final=True``,
matching the existing ``ov-optimum`` engine's contract. Per-token streaming
via the LLMPipeline streamer callback is straightforward to add in a
follow-up campaign.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId
from tahoma.worker.engines.base import Builder, Engine

logger = logging.getLogger(__name__)


@dataclass
class _ActiveTask:
    task: GenerationTask
    submitted_at: float = field(default_factory=time.time)


class OVGenAIEngine(Engine):
    """Single-stage LLMPipeline-backed engine.

    Tasks are processed serially: each ``step()`` activates the next pending
    task (if no task is active), runs ``pipe.generate(...)`` to completion,
    and yields one chunk with the full text. This matches the existing
    ``OptimumOVEngine`` semantics so the API layer doesn't need to know
    about the underlying engine.
    """

    def __init__(
        self,
        pipe: Any,
        max_tokens_default: int = 256,
    ):
        self._pipe = pipe
        self._max_tokens_default = max_tokens_default
        self._pending: list[GenerationTask] = []
        # We don't track an "active" set the way streaming engines do — the
        # generate() call below is blocking, so the task is "active" only
        # for the duration of step().

    def warmup(self) -> None:
        """One-token forward to compile kernels and warm caches."""
        try:
            import openvino_genai as ov_genai
            cfg = ov_genai.GenerationConfig()
            cfg.max_new_tokens = 1
            cfg.do_sample = False
            self._pipe.generate("Hi", cfg)
            logger.info("ov-genai warmup ok")
        except Exception as err:  # noqa: BLE001
            logger.warning("ov-genai warmup failed: %s", err)

    def submit(self, task: GenerationTask) -> None:
        if any(t.task_id == task.task_id for t in self._pending):
            return
        self._pending.append(task)

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        if not self._pending:
            return
        task = self._pending.pop(0)
        import openvino_genai as ov_genai

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = task.max_tokens or self._max_tokens_default
        cfg.do_sample = task.temperature > 0.0
        if task.temperature > 0.0:
            cfg.temperature = task.temperature

        t0 = time.perf_counter()
        result = self._pipe.generate(task.prompt, cfg)
        elapsed = time.perf_counter() - t0

        text = str(result)
        # Strip the prompt if LLMPipeline echoed it (some templates do).
        if text.startswith(task.prompt):
            text = text[len(task.prompt):].lstrip()

        # Best-effort token count from perf_metrics.
        tokens = cfg.max_new_tokens
        metrics = getattr(result, "perf_metrics", None)
        if metrics is not None:
            v = getattr(metrics, "num_generated_tokens", None)
            if v is not None:
                try:
                    tokens = int(v.mean if hasattr(v, "mean") else v)
                except (TypeError, ValueError):
                    pass

        tok_s = tokens / elapsed if elapsed > 0 else 0.0
        logger.info(
            "task %s done: %d tokens in %.3fs (%.1f tok/s)",
            task.task_id[:8], tokens, elapsed, tok_s,
        )
        yield task.task_id, Chunk(
            task_id=task.task_id, token_id=0,
            text=text, is_final=True,
        )

    def close(self) -> None:
        # Nothing to release — LLMPipeline handles its own teardown via __del__.
        pass


class OVGenAIBuilder(Builder):
    """Builder for ``OVGenAIEngine``.

    Loads an OV IR directory via ``openvino_genai.LLMPipeline``. Single-stage
    only; pipeline parallelism is not (yet) supported through GenAI.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "GPU",
        cache_dir: str | None = None,
        kv_cache_precision: str | None = None,
        dyn_quant_group: str | None = None,
    ):
        self._model_path = model_path
        self._device = device
        self._cache_dir = cache_dir
        self._kv_cache_precision = kv_cache_precision
        self._dyn_quant_group = dyn_quant_group
        self._pipe: Any = None

    def configure_listen(self, host: str, port: int) -> None:
        # Single-stage only — listen address is meaningless. Stored for
        # interface parity with multi-stage builders.
        del host, port

    def connect(self, peers: PeerLayout) -> None:
        if peers.upstream is not None or peers.downstream is not None:
            raise RuntimeError(
                "OVGenAIEngine is single-stage only; do not configure peers",
            )

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        if not (shard.is_first_stage and shard.is_last_stage):
            raise RuntimeError("OVGenAIEngine requires --total 1")

        yield LoadProgress(0, None, f"locating IR at {self._model_path}")
        if not Path(self._model_path).exists():
            raise RuntimeError(
                f"OV IR not found at {self._model_path}. "
                f"Pre-export with: optimum-cli export openvino "
                f"--model <hf-id> --weight-format int4 --task text-generation-with-past "
                f"{self._model_path}",
            )

        plugin_config: dict[str, str] = {}
        if self._cache_dir:
            plugin_config["CACHE_DIR"] = self._cache_dir
        if self._kv_cache_precision:
            plugin_config["KV_CACHE_PRECISION"] = self._kv_cache_precision
        if self._dyn_quant_group:
            plugin_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = self._dyn_quant_group

        yield LoadProgress(0, None, f"compiling LLMPipeline on {self._device}")
        try:
            import openvino_genai as ov_genai
        except ImportError as err:
            raise RuntimeError(
                "openvino-genai is required for the ov-genai engine; "
                "install with `pip install openvino-genai==<matching openvino version>`",
            ) from err
        self._pipe = ov_genai.LLMPipeline(self._model_path, self._device, **plugin_config)
        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._pipe is None:
            raise RuntimeError("call load() before build()")
        return OVGenAIEngine(pipe=self._pipe)

    def close(self) -> None:
        self._pipe = None


__all__ = ["OVGenAIBuilder", "OVGenAIEngine"]
