"""Common runtime types passed between modules.

Engine-agnostic dataclasses for tasks, chunks, and load progress. The API
layer produces these; engine implementations consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TaskId = str


@dataclass(frozen=True)
class TopLogprob:
    """Single (token, logprob) pair in a top-k logprobs response."""

    token: str
    token_id: int
    logprob: float


@dataclass(frozen=True)
class TokenLogprobs:
    """Logprob payload for one emitted token, OpenAI-compatible."""

    token: str
    token_id: int
    logprob: float
    top_logprobs: list[TopLogprob] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationTask:
    """A single generation request, end-to-end.

    Engines that don't support a given option should ignore it silently — the
    API layer is responsible for telling the user when their request was
    accepted but degraded (e.g. ``logprobs=true`` against an engine that
    doesn't expose logits).
    """

    task_id: TaskId
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    # Number of top logprobs to return per token. 0 = disabled. The token's
    # own logprob is always returned when this is > 0.
    logprobs: int = 0
    # When true, the engine should expose chain-of-thought / reasoning tokens
    # if the model supports it (DeepSeek V3.1, Qwen3, GLM-4.7, etc.). Engines
    # that don't recognise this leave it as a no-op.
    enable_thinking: bool = False
    # When true, the engine may execute code from the model repository (e.g.
    # custom tokenizer / attention modules). Off by default; opt in via
    # `--trust-remote-code` on the worker. Treat as a security boundary.
    trust_remote_code: bool = False


@dataclass(frozen=True)
class Chunk:
    """A single token (or final marker) produced by the engine for a task."""

    task_id: TaskId
    token_id: int
    text: str
    is_final: bool = False
    # Filled in when the engine supports it AND the task asked for it.
    logprobs: TokenLogprobs | None = None


@dataclass(frozen=True)
class LoadProgress:
    """A progress event emitted while a shard is loading."""

    bytes_loaded: int
    bytes_total: int | None
    message: str
