"""Ollama-dialect API surface — version, tags, show, generate, chat (NDJSON + non-streaming)."""

from __future__ import annotations

import json
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
    text_chunks: list[str] = field(default_factory=lambda: ["The ", "answer ", "is ", "42."])
    submitted: list[GenerationTask] = field(default_factory=list)

    def generate(self, task: GenerationTask) -> Iterator[Chunk]:
        self.submitted.append(task)
        for i, t in enumerate(self.text_chunks):
            yield Chunk(
                task_id=task.task_id, token_id=i, text=t,
                is_final=(i == len(self.text_chunks) - 1),
            )


def _client(runner: FakeRunner | None = None, model_id: str = "llama-test") -> TestClient:
    return TestClient(make_app(runner or FakeRunner(), model_id=model_id))


def test_api_version_ok() -> None:
    r = _client().get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_api_tags_lists_served_model() -> None:
    r = _client(model_id="my-model").get("/api/tags")
    assert r.status_code == 200
    body = r.json()
    assert body["models"][0]["name"] == "my-model"
    assert body["models"][0]["model"] == "my-model"
    assert body["models"][0]["details"]["format"] == "tahoma"


def test_api_show_returns_metadata() -> None:
    r = _client(model_id="abc").post("/api/show", json={"name": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert body["details"]["format"] == "tahoma"
    assert "abc" in body["modelfile"]


def test_api_generate_blocking_returns_full_response() -> None:
    runner = FakeRunner(text_chunks=["Hello", " ", "world"])
    r = _client(runner).post(
        "/api/generate",
        json={"model": "x", "prompt": "Hi", "stream": False, "options": {"num_predict": 32}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "Hello world"
    assert body["done"] is True
    assert body["eval_count"] == 3
    # The blocking response also carries duration metadata, never zero.
    assert body["total_duration"] >= 0


def test_api_chat_blocking_returns_message() -> None:
    runner = FakeRunner(text_chunks=["A", "B"])
    r = _client(runner).post(
        "/api/chat",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "AB"
    assert body["done"] is True


def test_api_generate_streaming_emits_ndjson_then_done() -> None:
    runner = FakeRunner(text_chunks=["One", "Two", "Three"])
    r = _client(runner).post(
        "/api/generate",
        json={"model": "x", "prompt": "Count", "stream": True},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")

    lines = [line for line in r.text.splitlines() if line.strip()]
    assert len(lines) >= 4  # 3 chunk events + 1 final

    for line in lines:
        json.loads(line)  # every line must be valid JSON

    final = json.loads(lines[-1])
    assert final["done"] is True
    assert final["done_reason"] == "stop"
    assert final["eval_count"] >= 3

    # Concatenating the per-chunk responses reconstructs the text.
    chunks = [json.loads(line) for line in lines[:-1]]
    assert "".join(c["response"] for c in chunks) == "OneTwoThree"


def test_api_chat_streaming_emits_message_chunks() -> None:
    runner = FakeRunner(text_chunks=["foo", "bar"])
    r = _client(runner).post(
        "/api/chat",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "p"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    lines = [line for line in r.text.splitlines() if line.strip()]
    assert len(lines) >= 3

    chunks = [json.loads(line) for line in lines]
    # Every non-final chunk has message.content with one piece of text.
    assert chunks[0]["message"]["role"] == "assistant"
    assert "".join(c["message"]["content"] for c in chunks[:-1]) == "foobar"
    assert chunks[-1]["done"] is True


def test_api_generate_options_num_predict_overrides_max_tokens() -> None:
    runner = FakeRunner(text_chunks=["x"])
    _client(runner).post(
        "/api/generate",
        json={"model": "x", "prompt": "p", "stream": False, "options": {"num_predict": 7}},
    )
    assert runner.submitted[0].max_tokens == 7


def test_api_chat_accepts_tools_field_without_crashing() -> None:
    """Tools are accepted (and currently ignored by the engine plumbing)."""
    runner = FakeRunner()
    r = _client(runner).post(
        "/api/chat",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "p"}],
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "noop"}}],
        },
    )
    assert r.status_code == 200


def test_openai_chat_accepts_tools_and_returns_empty_tool_calls() -> None:
    """tools accepted; engines that don't generate tool_calls return []."""
    runner = FakeRunner(text_chunks=["ok"])
    r = _client(runner).post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "p"}],
            "tools": [{"type": "function", "function": {"name": "noop"}}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["tool_calls"] == []
