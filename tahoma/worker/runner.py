"""Runner — owns a `Builder`/`Engine` for a single pipeline stage.

Lifecycle:
    1. start(peers, shard) — connect transports, load weights, build engine, warm up.
    2a. (first stage)  submit(task), then iterate generate() for chunks.
    2b. (other stages) run_relay_loop() — forwards activations indefinitely.
    3. close().
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, TaskId
from tahoma.worker.engines.base import Builder, Engine

logger = logging.getLogger(__name__)


class Runner:
    """Per-stage process owner."""

    def __init__(self, builder: Builder):
        self._builder = builder
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("call start() first")
        return self._engine

    def start(self, peers: PeerLayout, shard: ShardSpec) -> None:
        logger.info(
            "connect: upstream=%s downstream=%s", peers.upstream, peers.downstream,
        )
        self._builder.connect(peers)
        for progress in self._builder.load(shard):
            logger.info("load: %s", progress.message)
        logger.info("build engine")
        self._engine = self._builder.build()
        logger.info("warmup")
        self._engine.warmup()
        logger.info("runner ready")

    def submit(self, task: GenerationTask) -> None:
        self.engine.submit(task)

    def step_once(self) -> dict[TaskId, Chunk]:
        return {tid: chunk for tid, chunk in self.engine.step()}

    def generate(self, task: GenerationTask) -> Iterator[Chunk]:
        """Submit a task and yield chunks as they arrive (first stage only)."""
        self.submit(task)
        while True:
            chunks = self.step_once()
            chunk = chunks.get(task.task_id)
            if chunk is None:
                # No chunk produced this step (queue empty / task already finished).
                return
            yield chunk
            if chunk.is_final:
                return

    def run_relay_loop(self) -> None:
        """Step the engine forever; exits when transport closes (non-first stages)."""
        engine = self.engine
        try:
            while True:
                # Drain step's iterable; non-first stages produce no chunks.
                for _ in engine.step():
                    pass
        except (ConnectionError, BrokenPipeError, OSError) as err:
            logger.info("transport closed: %s", err)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._builder.close()
