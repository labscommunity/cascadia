"""Runner orchestrates Builder + Engine. Tests use fakes — no real models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pytest

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId
from tahoma.worker.engines.base import Builder, Engine
from tahoma.worker.runner import Runner


def _shard() -> ShardSpec:
    return ShardSpec(
        model_id="fake", layer_start=0, layer_end=0, total_layers=0,
        device="CPU", is_first_stage=True, is_last_stage=True,
    )


def _task(prompt: str = "hi", max_tokens: int = 3) -> GenerationTask:
    return GenerationTask(task_id="t1", prompt=prompt, max_tokens=max_tokens)


@dataclass
class FakeEngine(Engine):
    chunks_to_yield: list[Chunk] = field(default_factory=list)
    submitted: list[GenerationTask] = field(default_factory=list)
    warmed: bool = False
    closed: bool = False
    _idx: int = 0

    def warmup(self) -> None:
        self.warmed = True

    def submit(self, task: GenerationTask) -> None:
        self.submitted.append(task)

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        if self._idx < len(self.chunks_to_yield):
            chunk = self.chunks_to_yield[self._idx]
            self._idx += 1
            yield (chunk.task_id, chunk)

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeBuilder(Builder):
    engine: FakeEngine = field(default_factory=FakeEngine)
    connected: bool = False
    progress_events: list[LoadProgress] = field(
        default_factory=lambda: [LoadProgress(0, 1, "loading")],
    )
    closed: bool = False
    raise_on_load: Exception | None = None

    def connect(self, peers: PeerLayout) -> None:
        self.connected = True

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        if self.raise_on_load is not None:
            raise self.raise_on_load
        yield from self.progress_events

    def build(self) -> Engine:
        return self.engine

    def close(self) -> None:
        self.closed = True


def test_start_invokes_each_phase_in_order() -> None:
    engine = FakeEngine()
    builder = FakeBuilder(engine=engine)
    runner = Runner(builder)

    runner.start(PeerLayout(upstream=None, downstream=None), _shard())

    assert builder.connected
    assert engine.warmed
    assert runner.engine is engine


def test_engine_property_before_start_raises() -> None:
    runner = Runner(FakeBuilder())
    with pytest.raises(RuntimeError, match="call start"):
        _ = runner.engine


def test_close_tears_down_engine_and_builder() -> None:
    engine = FakeEngine()
    builder = FakeBuilder(engine=engine)
    runner = Runner(builder)
    runner.start(PeerLayout(upstream=None, downstream=None), _shard())

    runner.close()

    assert engine.closed
    assert builder.closed


def test_close_without_start_does_not_raise() -> None:
    builder = FakeBuilder()
    runner = Runner(builder)
    runner.close()  # No start() ever ran; should still close the builder.
    assert builder.closed


def test_generate_yields_until_final_chunk() -> None:
    chunks = [
        Chunk(task_id="t1", token_id=1, text="A"),
        Chunk(task_id="t1", token_id=2, text="B"),
        Chunk(task_id="t1", token_id=3, text="!", is_final=True),
        # Sentinel that should never be yielded — generator returns at is_final.
        Chunk(task_id="t1", token_id=999, text="X"),
    ]
    engine = FakeEngine(chunks_to_yield=chunks)
    builder = FakeBuilder(engine=engine)
    runner = Runner(builder)
    runner.start(PeerLayout(upstream=None, downstream=None), _shard())

    out = list(runner.generate(_task()))
    assert [c.text for c in out] == ["A", "B", "!"]
    assert engine.submitted == [_task()]


def test_generate_returns_when_engine_emits_nothing_for_task() -> None:
    """If step() yields no chunk for the task_id, generate() exits cleanly."""
    other = Chunk(task_id="OTHER", token_id=1, text="foo")
    engine = FakeEngine(chunks_to_yield=[other])
    builder = FakeBuilder(engine=engine)
    runner = Runner(builder)
    runner.start(PeerLayout(upstream=None, downstream=None), _shard())

    # generate() will see one step that produces a chunk for a different task,
    # then a step that produces nothing — and exit.
    out = list(runner.generate(_task()))
    assert out == []


def test_relay_loop_exits_on_connection_error() -> None:
    @dataclass
    class CrashingEngine(FakeEngine):
        def step(self) -> Iterable[tuple[TaskId, Chunk]]:
            raise ConnectionError("peer disconnected")
            yield  # unreachable; makes this a generator

    engine = CrashingEngine()
    builder = FakeBuilder(engine=engine)
    runner = Runner(builder)
    runner.start(PeerLayout(upstream=None, downstream=None), _shard())

    # Should not raise; relay loop catches ConnectionError/OSError.
    runner.run_relay_loop()


def test_load_failure_propagates() -> None:
    builder = FakeBuilder(raise_on_load=ValueError("bad shard"))
    runner = Runner(builder)
    with pytest.raises(ValueError, match="bad shard"):
        runner.start(PeerLayout(upstream=None, downstream=None), _shard())


def test_concurrent_generate_routes_chunks_per_task() -> None:
    """Two tasks submitted concurrently each receive their own chunks.

    The fake engine emits chunks for both tasks interleaved; the runner
    must route each chunk to the right caller.
    """
    import threading

    @dataclass
    class TwoTaskEngine(FakeEngine):
        # Step yields one chunk for whichever task is next in the queue.
        # We hard-code an interleaving so the ordering is deterministic.
        plan: list[tuple[str, str, bool]] = field(default_factory=lambda: [
            ("a", "A1", False), ("b", "B1", False),
            ("a", "A2", False), ("b", "B2", False),
            ("a", "A3", True),  ("b", "B3", True),
        ])
        cursor: int = 0

        def step(self) -> Iterable[tuple[TaskId, Chunk]]:
            if self.cursor >= len(self.plan):
                return
            tid, text, final = self.plan[self.cursor]
            self.cursor += 1
            yield (tid, Chunk(task_id=tid, token_id=self.cursor, text=text, is_final=final))

    engine = TwoTaskEngine()
    builder = FakeBuilder(engine=engine)
    runner = Runner(builder)
    runner.start(PeerLayout(upstream=None, downstream=None), _shard())

    out_a: list[Chunk] = []
    out_b: list[Chunk] = []

    def run(task_id: str, sink: list[Chunk]) -> None:
        task = GenerationTask(task_id=task_id, prompt="x", max_tokens=10)
        sink.extend(runner.generate(task))

    ta = threading.Thread(target=run, args=("a", out_a), daemon=True)
    tb = threading.Thread(target=run, args=("b", out_b), daemon=True)
    ta.start()
    tb.start()
    ta.join(timeout=5.0)
    tb.join(timeout=5.0)
    assert not ta.is_alive() and not tb.is_alive()

    assert [c.text for c in out_a] == ["A1", "A2", "A3"]
    assert [c.text for c in out_b] == ["B1", "B2", "B3"]


def test_cancel_terminates_generate_early() -> None:
    """cancel() pushes the named task into a cancelled set; the next
    generate() poll for that task returns."""
    chunks = [Chunk(task_id="t1", token_id=i, text=str(i)) for i in range(20)]
    engine = FakeEngine(chunks_to_yield=chunks)
    builder = FakeBuilder(engine=engine)
    runner = Runner(builder)
    runner.start(PeerLayout(upstream=None, downstream=None), _shard())

    # Run generate in a thread; cancel after one chunk.
    import threading
    received: list[Chunk] = []
    def run() -> None:
        for c in runner.generate(_task()):
            received.append(c)
            if len(received) == 1:
                runner.cancel("t1")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive()
    # Should have received at least one chunk and then exited well before
    # the 20-chunk plan finished naturally.
    assert 1 <= len(received) < 20
