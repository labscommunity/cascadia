"""HTTP API contract tests using FastAPI's TestClient.

The Runner is faked so tests don't need a real engine — they exercise the
request/response shape, error paths, and the wiring between runner.generate
and the OpenAI-compatible response format.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tahoma.api.server import make_app  # noqa: E402
from tahoma.shared.types import Chunk, GenerationTask  # noqa: E402


@dataclass
class FakeRunner:
    """Implements the Runner.generate(task) -> Iterator[Chunk] surface."""

    text_chunks: list[str] = field(default_factory=lambda: ["Hello", " ", "world"])
    submitted: list[GenerationTask] = field(default_factory=list)

    def generate(self, task: GenerationTask) -> Iterator[Chunk]:
        self.submitted.append(task)
        for i, t in enumerate(self.text_chunks):
            yield Chunk(
                task_id=task.task_id,
                token_id=i,
                text=t,
                is_final=(i == len(self.text_chunks) - 1),
            )


def _client(runner: FakeRunner | None = None, model_id: str = "fake-model") -> TestClient:
    return TestClient(make_app(runner or FakeRunner(), model_id=model_id))


def test_health_returns_ok() -> None:
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_lists_configured_id() -> None:
    r = _client(model_id="my-llama").get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "my-llama"
    assert body["data"][0]["owned_by"] == "tahoma"


def test_chat_completion_returns_concatenated_text() -> None:
    runner = FakeRunner(text_chunks=["The ", "quick ", "fox"])
    r = _client(runner).post(
        "/v1/chat/completions",
        json={
            "model": "any",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "The quick fox"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] == 3
    assert len(runner.submitted) == 1
    submitted = runner.submitted[0]
    assert submitted.max_tokens == 32
    assert "user: hi" in submitted.prompt


def test_chat_completion_finish_reason_stop_when_under_max() -> None:
    runner = FakeRunner(text_chunks=["a", "b"])
    r = _client(runner).post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "u", "content": "p"}], "max_tokens": 10},
    )
    assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_finish_reason_length_at_max() -> None:
    runner = FakeRunner(text_chunks=["a", "b"])
    r = _client(runner).post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "u", "content": "p"}], "max_tokens": 2},
    )
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_streaming_request_returns_501() -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "x", "messages": [{"role": "u", "content": "p"}],
            "stream": True,
        },
    )
    assert r.status_code == 501
    assert "streaming" in r.json()["detail"]


def test_chat_completion_validates_request_shape() -> None:
    r = _client().post("/v1/chat/completions", json={"model": "x"})  # missing messages
    assert r.status_code == 422


def test_chat_completion_id_starts_with_chatcmpl() -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "u", "content": "p"}]},
    )
    assert r.json()["id"].startswith("chatcmpl-")
