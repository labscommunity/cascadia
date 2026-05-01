"""Common runtime types passed between modules.

Engine-agnostic dataclasses for tasks, chunks, and load progress. The API
layer produces these; engine implementations consume them.
"""

from __future__ import annotations

from dataclasses import dataclass

TaskId = str


@dataclass(frozen=True)
class GenerationTask:
    """A single generation request, end-to-end."""

    task_id: TaskId
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass(frozen=True)
class Chunk:
    """A single token (or final marker) produced by the engine for a task."""

    task_id: TaskId
    token_id: int
    text: str
    is_final: bool = False


@dataclass(frozen=True)
class LoadProgress:
    """A progress event emitted while a shard is loading."""

    bytes_loaded: int
    bytes_total: int | None
    message: str
