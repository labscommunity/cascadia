"""Engine and Builder ABCs.

A `Builder` is the per-shard setup phase: connect to upstream/downstream peers
and load model weights. Once both finish, `build()` returns an `Engine` that
produces chunks via `submit()` + `step()`.

This split mirrors exo (https://github.com/exo-explore/exo). The shape lets
multiple inference backends (OpenVINO today; IPEX-LLM, llama.cpp later) plug
in behind a stable interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId


class Engine(ABC):
    """A loaded inference engine for a single pipeline stage.

    Lifecycle: `warmup` → repeated (`submit`, `step`) → `close`.
    """

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy forward pass to compile graphs and warm caches."""

    @abstractmethod
    def submit(self, task: GenerationTask) -> None:
        """Queue a generation task. Idempotent on duplicate task_id."""

    @abstractmethod
    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        """Run one forward pass.

        Yields `(task_id, chunk)` for any task that produced output this step.
        Pipeline stages other than the last typically yield nothing — output
        activations flow over the transport rather than back to the caller.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources (sockets, GPU contexts, file handles)."""


class Builder(ABC):
    """Per-shard setup phase that yields an `Engine`.

    Lifecycle: `connect` → `load` (yields progress) → `build` → (`Engine`).
    `close()` is safe to call at any point and tears down whatever has been
    set up so far.
    """

    @abstractmethod
    def connect(self, peers: PeerLayout) -> None:
        """Open transport to the upstream and/or downstream peers."""

    @abstractmethod
    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        """Load this node's shard. Yields progress events for the caller."""

    @abstractmethod
    def build(self) -> Engine:
        """Construct the Engine. Call after `connect()` and `load()` complete."""

    @abstractmethod
    def close(self) -> None:
        """Tear down transports and partially-loaded weights."""
