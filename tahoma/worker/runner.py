"""Runner — owns a `Builder`/`Engine` for a single pipeline stage.

Lifecycle:
    1. start(peers, shard) — connect transports, load weights, build engine, warm up.
    2a. (first stage)  submit(task), then iterate generate() for chunks.
    2b. (other stages) run_relay_loop() — forwards activations indefinitely.
    3. close().

Concurrency
-----------
``generate()`` is safe to call from multiple threads concurrently. Each call
shares the engine through a single lock around ``engine.step()``. Chunks
emitted by the engine are routed to per-task buffers, so caller A's loop
sees chunks for caller A's task even if caller B is also running.

This is round-robin, *not* batched compute — every step still produces at
most one chunk per active task per round. Real batched forward needs the
underlying engine to call into a batched primitive; that's per-engine work.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from collections.abc import Iterator

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, TaskId
from tahoma.worker.engines.base import Builder, Engine

logger = logging.getLogger(__name__)


# When ``generate()`` sees this many consecutive empty engine steps with no
# new chunks for *any* task, it returns. This is the "engine has gone idle
# and never produced anything for me" case — without it the caller would
# block forever on a misbehaving engine.
_MAX_CONSECUTIVE_EMPTY_STEPS = 3


class Runner:
    """Per-stage process owner."""

    def __init__(self, builder: Builder):
        self._builder = builder
        self._engine: Engine | None = None
        # Lock serialises engine.step() across concurrent generate() callers.
        self._step_lock = threading.Lock()
        # Per-task chunk buffer; appended to by whichever generator's step()
        # call produced the chunk, drained by the owning task's generator.
        self._chunk_buffers: dict[TaskId, deque[Chunk]] = defaultdict(deque)
        self._cancelled: set[TaskId] = set()

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("call start() first")
        return self._engine

    # ----- lifecycle ------------------------------------------------------

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

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._builder.close()

    # ----- submit / generate ---------------------------------------------

    def submit(self, task: GenerationTask) -> None:
        self.engine.submit(task)

    def step_once(self) -> dict[TaskId, Chunk]:
        """Run one engine step. Convenience wrapper for tests / debug."""
        return {tid: chunk for tid, chunk in self.engine.step()}

    def cancel(self, task_id: TaskId) -> None:
        """Cooperatively stop the named task. The next ``generate()`` poll
        for that task returns; any chunks already buffered are dropped."""
        with self._step_lock:
            self._cancelled.add(task_id)
            self._chunk_buffers.pop(task_id, None)

    def generate(self, task: GenerationTask) -> Iterator[Chunk]:
        """Submit a task and yield chunks as they arrive (first stage only).

        Safe to call concurrently with other ``generate()`` calls — chunks
        for other tasks produced during *our* step() turns are buffered for
        their owners, and vice versa.
        """
        self.submit(task)
        consecutive_empty = 0
        try:
            while True:
                if task.task_id in self._cancelled:
                    return

                # Drain any chunks already buffered for us by other generators.
                buf = self._chunk_buffers.get(task.task_id)
                while buf:
                    chunk = buf.popleft()
                    yield chunk
                    if chunk.is_final:
                        return

                # Take our turn driving the engine.
                with self._step_lock:
                    produced = list(self.engine.step())
                    for tid, chunk in produced:
                        if tid not in self._cancelled:
                            self._chunk_buffers[tid].append(chunk)

                if produced:
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
                    if consecutive_empty >= _MAX_CONSECUTIVE_EMPTY_STEPS:
                        # Engine isn't making progress. Bail rather than hang.
                        return
        finally:
            with self._step_lock:
                self._chunk_buffers.pop(task.task_id, None)
                self._cancelled.discard(task.task_id)

    # ----- relay loop (non-first stages) ---------------------------------

    def run_relay_loop(self) -> None:
        """Step the engine forever; exits when transport closes (non-first stages)."""
        engine = self.engine
        try:
            while True:
                for _ in engine.step():
                    pass
        except (ConnectionError, BrokenPipeError, OSError) as err:
            logger.info("transport closed: %s", err)
