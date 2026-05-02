"""Ollama-compatible API surface (subset).

Implements the routes that OpenWebUI and other Ollama clients depend on:

- ``GET  /api/version``  → ``{"version": "..."}``
- ``GET  /api/tags``     → list of installed models, Ollama format
- ``POST /api/show``     → model metadata (returns the configured model id only)
- ``POST /api/generate`` → text-completion style; supports streaming + non-streaming
- ``POST /api/chat``     → chat-completion style; supports streaming + non-streaming

Differences vs Ollama proper:

- Streaming uses NDJSON (one JSON object per line, terminated by a final
  ``"done": true`` object). This matches Ollama's wire format.
- We don't implement ``/api/pull``, ``/api/push``, or ``/api/embeddings``
  here. Pull happens through the model registry (separate route group);
  push and embeddings are out of scope until we have a real embedding model.
- ``/api/tags`` returns one model — the one the worker is serving — until
  the registry lands and can enumerate locally-cached models.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tahoma.shared.types import GenerationTask
from tahoma.worker.runner import Runner

logger = logging.getLogger(__name__)

# Same major.minor as the Ollama version OpenWebUI checks against. Bump in
# step with our actual feature surface.
OLLAMA_API_VERSION = "0.1.32"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class OllamaMessage(BaseModel):
    role: str
    content: str
    # Ollama also accepts `images: [base64]`; we drop them on the floor for
    # now (text-only models).


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    options: dict[str, Any] | None = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = True
    options: dict[str, Any] | None = None
    # Tools are accepted but currently passed through only as prompt text;
    # see /v1/chat/completions for the OpenAI-format tool_calls path.
    tools: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Ollama uses RFC 3339 UTC timestamps."""
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def _ndjson(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode()


def _format_ollama_chat_prompt(messages: list[OllamaMessage]) -> str:
    """Apply the same minimal templating we use for /v1/chat/completions."""
    parts = [f"{m.role}: {m.content}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def _max_tokens_from_options(options: dict[str, Any] | None, default: int = 256) -> int:
    """Ollama exposes max-tokens as ``num_predict``. Both names are accepted."""
    if not options:
        return default
    if "num_predict" in options:
        return int(options["num_predict"])
    if "max_tokens" in options:
        return int(options["max_tokens"])
    return default


def _temperature_from_options(options: dict[str, Any] | None) -> float:
    if options and "temperature" in options:
        return float(options["temperature"])
    return 0.0


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register(app: FastAPI, runner: Runner, model_id: str) -> None:
    """Wire the Ollama routes into ``app``. Called from ``make_app``."""

    @app.get("/api/version")
    def api_version() -> dict:
        return {"version": OLLAMA_API_VERSION}

    @app.get("/api/tags")
    def api_tags() -> dict:
        return {
            "models": [
                {
                    "name": model_id,
                    "model": model_id,
                    "modified_at": _now_iso(),
                    "size": 0,
                    "digest": "",
                    "details": {
                        "parent_model": "",
                        "format": "tahoma",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                },
            ],
        }

    @app.post("/api/show")
    def api_show(req: dict) -> dict:
        name = req.get("name") or req.get("model") or model_id
        return {
            "modelfile": f"# Tahoma-served model: {name}\n",
            "parameters": "",
            "template": "",
            "details": {
                "parent_model": "",
                "format": "tahoma",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "",
                "quantization_level": "",
            },
        }

    def _make_task(prompt: str, max_tokens: int, temperature: float) -> GenerationTask:
        return GenerationTask(
            task_id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @app.post("/api/generate")
    async def api_generate(req: OllamaGenerateRequest, http_req: Request) -> Any:
        max_tokens = _max_tokens_from_options(req.options)
        temperature = _temperature_from_options(req.options)
        task = _make_task(req.prompt, max_tokens, temperature)

        if req.stream:
            return StreamingResponse(
                _stream_generate(runner, task, req.model, http_req),
                media_type="application/x-ndjson",
            )
        return await asyncio.to_thread(_run_generate_blocking, runner, task, req.model)

    @app.post("/api/chat")
    async def api_chat(req: OllamaChatRequest, http_req: Request) -> Any:
        prompt = _format_ollama_chat_prompt(req.messages)
        max_tokens = _max_tokens_from_options(req.options)
        temperature = _temperature_from_options(req.options)
        task = _make_task(prompt, max_tokens, temperature)

        if req.stream:
            return StreamingResponse(
                _stream_chat(runner, task, req.model, http_req),
                media_type="application/x-ndjson",
            )
        return await asyncio.to_thread(_run_chat_blocking, runner, task, req.model)


# ---------------------------------------------------------------------------
# Blocking + streaming runners
# ---------------------------------------------------------------------------


def _run_generate_blocking(runner: Runner, task: GenerationTask, model: str) -> dict:
    text_parts: list[str] = []
    started = time.time()
    for chunk in runner.generate(task):
        text_parts.append(chunk.text)
        if chunk.is_final:
            break
    elapsed_ns = int((time.time() - started) * 1e9)
    return {
        "model": model,
        "created_at": _now_iso(),
        "response": "".join(text_parts),
        "done": True,
        "done_reason": "stop",
        "context": [],
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": len(text_parts),
        "eval_duration": elapsed_ns,
    }


def _run_chat_blocking(runner: Runner, task: GenerationTask, model: str) -> dict:
    text_parts: list[str] = []
    started = time.time()
    for chunk in runner.generate(task):
        text_parts.append(chunk.text)
        if chunk.is_final:
            break
    elapsed_ns = int((time.time() - started) * 1e9)
    return {
        "model": model,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": "".join(text_parts)},
        "done": True,
        "done_reason": "stop",
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": len(text_parts),
        "eval_duration": elapsed_ns,
    }


async def _stream_generate(
    runner: Runner, task: GenerationTask, model: str, http_req: Request,
) -> AsyncIterator[bytes]:
    started = time.time()
    eval_count = 0
    finish_reason = "stop"
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    SENTINEL = object()

    def producer() -> None:
        try:
            for chunk in runner.generate(task):
                queue.put_nowait(chunk)
        except Exception as err:  # noqa: BLE001
            queue.put_nowait(err)
        finally:
            queue.put_nowait(SENTINEL)

    p = asyncio.create_task(asyncio.to_thread(producer))
    try:
        while True:
            if await http_req.is_disconnected():
                finish_reason = "cancelled"
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                yield _ndjson({"error": str(item)})
                finish_reason = "error"
                break
            chunk = item
            eval_count += 1
            yield _ndjson({
                "model": model,
                "created_at": _now_iso(),
                "response": chunk.text,
                "done": False,
            })
            if chunk.is_final:
                break
    finally:
        with contextlib.suppress(Exception):
            await p

    elapsed_ns = int((time.time() - started) * 1e9)
    yield _ndjson({
        "model": model,
        "created_at": _now_iso(),
        "response": "",
        "done": True,
        "done_reason": finish_reason,
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": eval_count,
        "eval_duration": elapsed_ns,
    })


async def _stream_chat(
    runner: Runner, task: GenerationTask, model: str, http_req: Request,
) -> AsyncIterator[bytes]:
    started = time.time()
    eval_count = 0
    finish_reason = "stop"
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    SENTINEL = object()

    def producer() -> None:
        try:
            for chunk in runner.generate(task):
                queue.put_nowait(chunk)
        except Exception as err:  # noqa: BLE001
            queue.put_nowait(err)
        finally:
            queue.put_nowait(SENTINEL)

    p = asyncio.create_task(asyncio.to_thread(producer))
    try:
        while True:
            if await http_req.is_disconnected():
                finish_reason = "cancelled"
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                yield _ndjson({"error": str(item)})
                finish_reason = "error"
                break
            chunk = item
            eval_count += 1
            yield _ndjson({
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": chunk.text},
                "done": False,
            })
            if chunk.is_final:
                break
    finally:
        with contextlib.suppress(Exception):
            await p

    elapsed_ns = int((time.time() - started) * 1e9)
    yield _ndjson({
        "model": model,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": finish_reason,
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": eval_count,
        "eval_duration": elapsed_ns,
    })


# Suppress an unused-import warning if HTTPException is later needed for /api/pull etc.
_ = HTTPException
