"""OpenAI-compatible HTTP server.

Endpoints:
- ``GET /health`` — liveness probe.
- ``GET /v1/models`` — single-model registry advertising the served model id.
- ``POST /v1/chat/completions`` — non-streaming and SSE streaming, with
  optional ``logprobs``, ``enable_thinking``, and request cancellation.
- ``POST /v1/cancel/{task_id}`` — explicit cancellation.
- ``GET /state`` — runner + engine state snapshot for ops.
- ``GET /events`` — server-sent stream of state-change events (task accepted,
  done, cancelled).
- ``GET /node_id`` — this worker's persistent id.

Tracing: when ``TAHOMA_TRACING_ENABLED=1`` is in the environment, every log
line and every event includes the originating ``task_id`` so a single chat
completion can be followed end-to-end.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tahoma.api import ollama as ollama_routes
from tahoma.download import (
    discover_local_snapshots,
    get_model,
    list_models,
    pull,
    register as registry_register,
    search_hf,
    unregister,
)
from tahoma.master import (
    StageRequirement,
    elect_master,
    is_master,
    propose_placements,
)
from tahoma.shared.topology import Topology
from tahoma.shared.types import GenerationTask
from tahoma.worker.runner import Runner

logger = logging.getLogger(__name__)


def _node_id() -> str:
    """Stable per-process node id; persisted to /tmp on first call."""
    path = os.path.join(
        os.environ.get("TAHOMA_NODE_ID_DIR", "/tmp"), "tahoma-node-id",
    )
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        nid = str(uuid.uuid4())
        try:
            with open(path, "w") as f:
                f.write(nid)
        except OSError:
            pass  # ephemeral if we can't persist; fine
        return nid


def _tracing_enabled() -> bool:
    return os.environ.get("TAHOMA_TRACING_ENABLED", "").lower() in ("1", "true", "yes")


def _namespace() -> str:
    return os.environ.get("TAHOMA_NAMESPACE", "default")


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ToolFunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolSpec(BaseModel):
    type: str = "function"
    function: ToolFunctionSpec


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = False
    # Number of top logprobs per token. OpenAI accepts True/False for
    # `logprobs` and a separate int for `top_logprobs`; we collapse to a
    # single int where 0 = disabled.
    logprobs: bool = False
    top_logprobs: int = Field(0, ge=0, le=20)
    # Pass-through for reasoning models (DeepSeek V3.1, Qwen3, GLM-4.7, ...).
    enable_thinking: bool = False
    # OpenAI-compatible tool definitions. Tahoma accepts and forwards them
    # into the prompt; engines that lack a grammar-constrained sampling path
    # can emit `tool_calls: null` in the response. Engines that DO know how
    # to call tools should look at `task.tools` (currently empty — the
    # plumbing is in place; engine integration is a follow-up).
    tools: list[ToolSpec] | None = None
    tool_choice: str | dict[str, Any] | None = None


class TopLogprobOut(BaseModel):
    token: str
    bytes: list[int] | None = None
    logprob: float


class TokenLogprobsOut(BaseModel):
    token: str
    bytes: list[int] | None = None
    logprob: float
    top_logprobs: list[TopLogprobOut] = []


class ChoiceLogprobsOut(BaseModel):
    content: list[TokenLogprobsOut] = []


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded args, per OpenAI spec


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: ChoiceLogprobsOut | None = None
    # Empty list when the engine doesn't emit tool calls. Always present so
    # OpenAI clients have a stable shape.
    tool_calls: list[ToolCall] = []


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


def _format_prompt(messages: list[ChatMessage]) -> str:
    """Minimal chat-template formatting. Replace with the model's template later."""
    parts = [f"{m.role}: {m.content}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Server-side bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _ServerState:
    """Hot-path state carried alongside the FastAPI app."""

    node_id: str
    namespace: str
    started_at: float
    in_flight: dict[str, float] = field(default_factory=dict)
    cancel_flags: dict[str, bool] = field(default_factory=dict)
    completed: int = 0
    cancelled: int = 0
    event_subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)

    def emit(self, event: dict[str, Any]) -> None:
        """Push an event to every /events subscriber. Drops if a queue is full."""
        for q in list(self.event_subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: drop rather than block the request thread.
                pass


def make_app(
    runner: Runner,
    model_id: str,
    *,
    topology: Topology | None = None,
) -> FastAPI:
    state = _ServerState(
        node_id=_node_id(),
        namespace=_namespace(),
        started_at=time.time(),
    )
    tracing = _tracing_enabled()
    topo = topology if topology is not None else Topology()

    app = FastAPI(title="Tahoma", version="0.0.1")

    # Ollama-compatible endpoints (/api/version, /api/tags, /api/show,
    # /api/generate, /api/chat) registered alongside the OpenAI surface.
    # OpenWebUI and other Ollama clients just work.
    ollama_routes.register(app, runner, model_id)

    # ----- liveness + ops -------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/node_id")
    def node_id() -> dict:
        return {"node_id": state.node_id, "namespace": state.namespace}

    @app.get("/state")
    def get_state() -> dict:
        return {
            "node_id": state.node_id,
            "namespace": state.namespace,
            "model": model_id,
            "uptime_s": round(time.time() - state.started_at, 2),
            "in_flight": list(state.in_flight.keys()),
            "completed": state.completed,
            "cancelled": state.cancelled,
            "tracing": tracing,
            "peers": [
                {
                    "node_id": n.node_id, "host": n.host, "port": n.port,
                    "device": n.device, "memory_mb": n.memory_mb,
                    "engines": n.engines,
                }
                for n in topo.nodes.values()
            ],
            "is_master": is_master(topo, state.node_id, state.namespace),
            "master": elect_master(topo, state.namespace),
        }

    @app.post("/instance/previews")
    def previews(spec: dict) -> dict:
        """Return suggested pipeline placements for a model.

        Request body::

            {"requirements": [
                {"rank": 0, "min_memory_mb": 8000, "needs_engines": ["ov-runtime"]},
                {"rank": 1, "min_memory_mb": 8000, "needs_engines": ["ov-runtime"]}
            ]}
        """
        try:
            reqs = [
                StageRequirement(
                    rank=int(r["rank"]),
                    min_memory_mb=int(r.get("min_memory_mb", 0)),
                    needs_engines=tuple(r.get("needs_engines", [])),
                )
                for r in spec.get("requirements", [])
            ]
        except (KeyError, TypeError, ValueError) as err:
            raise HTTPException(status_code=400, detail=f"invalid requirements: {err}") from err
        proposals = propose_placements(topo, reqs, namespace=state.namespace)
        return {
            "namespace": state.namespace,
            "proposals": [p.as_dict() for p in proposals],
        }

    @app.get("/events")
    async def events() -> StreamingResponse:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)
        state.event_subscribers.append(queue)

        async def stream() -> AsyncIterator[bytes]:
            try:
                # Open with a hello so clients know the stream is live.
                yield _sse({"type": "hello", "node_id": state.node_id})
                while True:
                    evt = await queue.get()
                    yield _sse(evt)
            finally:
                with contextlib.suppress(ValueError):
                    state.event_subscribers.remove(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    # ----- model listing --------------------------------------------------

    @app.get("/v1/models")
    def list_v1_models() -> dict:
        # Currently-served model + everything we have in the registry.
        seen: set[str] = {model_id}
        data = [{"id": model_id, "object": "model", "owned_by": "tahoma"}]
        for entry in list_models():
            if entry.id in seen:
                continue
            seen.add(entry.id)
            data.append(entry.to_openai())
        return {"object": "list", "data": data}

    # ----- model registry -----------------------------------------------

    @app.get("/models")
    def reg_list_models() -> dict:
        return {"models": [
            {
                "id": e.id, "source": e.source, "local_path": e.local_path,
                "size_bytes": e.size_bytes, "pulled_at": e.pulled_at,
                "revision": e.revision, "tags": e.tags,
            }
            for e in list_models()
        ]}

    @app.get("/models/{model_id:path}")
    def reg_get_model(model_id: str) -> dict:
        entry = get_model(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown model {model_id!r}")
        return {
            "id": entry.id, "source": entry.source,
            "local_path": entry.local_path, "size_bytes": entry.size_bytes,
            "pulled_at": entry.pulled_at, "revision": entry.revision,
            "tags": entry.tags,
        }

    @app.delete("/models/{model_id:path}")
    def reg_delete_model(model_id: str) -> dict:
        if not unregister(model_id):
            raise HTTPException(status_code=404, detail=f"unknown model {model_id!r}")
        return {"unregistered": model_id}

    @app.post("/models/discover")
    def reg_discover() -> dict:
        """Walk the HF cache and register everything that's already there."""
        added: list[str] = []
        for entry in discover_local_snapshots():
            if get_model(entry.id) is None:
                registry_register(entry)
                added.append(entry.id)
        return {"discovered": added}

    @app.get("/models/search/{query:path}")
    def reg_search(query: str, limit: int = 20) -> dict:
        return {"results": search_hf(query, limit=limit)}

    @app.post("/models/pull")
    async def reg_pull(spec: dict) -> Any:
        model_id_in = spec.get("model")
        if not model_id_in:
            raise HTTPException(status_code=400, detail="missing 'model' field")
        revision = spec.get("revision")
        stream = spec.get("stream", True)

        if not stream:
            # Block until done; return final event.
            final: dict[str, Any] = {}
            for event in pull(model_id_in, revision=revision):
                final = {
                    "status": event.status, "progress_bytes": event.progress_bytes,
                    "total_bytes": event.total_bytes, "file": event.file,
                    "error": event.error,
                }
            return final

        async def stream_pull() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[Any] = asyncio.Queue()
            SENTINEL = object()

            def producer() -> None:
                try:
                    for event in pull(model_id_in, revision=revision):
                        queue.put_nowait(event)
                finally:
                    queue.put_nowait(SENTINEL)

            # Fire the producer; do NOT await — it must run concurrently.
            producer_task = asyncio.create_task(asyncio.to_thread(producer))
            try:
                while True:
                    item = await queue.get()
                    if item is SENTINEL:
                        break
                    yield (json.dumps({
                        "status": item.status, "progress_bytes": item.progress_bytes,
                        "total_bytes": item.total_bytes, "file": item.file,
                        "error": item.error,
                    }) + "\n").encode()
            finally:
                with contextlib.suppress(Exception):
                    await producer_task

        return StreamingResponse(stream_pull(), media_type="application/x-ndjson")

    # ----- cancellation ---------------------------------------------------

    @app.post("/v1/cancel/{task_id}")
    def cancel(task_id: str) -> dict:
        if task_id not in state.in_flight:
            raise HTTPException(status_code=404, detail=f"unknown task {task_id}")
        state.cancel_flags[task_id] = True
        state.emit({"type": "task.cancelled", "task_id": task_id})
        logger.info("cancel requested for task=%s", task_id)
        return {"status": "cancelling", "task_id": task_id}

    # ----- chat completions -----------------------------------------------

    def _make_task(req: ChatCompletionRequest) -> GenerationTask:
        prompt = _format_prompt(req.messages)
        wants_logprobs = req.logprobs and req.top_logprobs > 0
        return GenerationTask(
            task_id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            logprobs=req.top_logprobs if wants_logprobs else 0,
            enable_thinking=req.enable_thinking,
        )

    def _trace(msg: str, **kw: Any) -> None:
        if tracing:
            logger.info("trace: %s %s", msg, json.dumps(kw, default=str))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, http_req: Request) -> Any:
        task = _make_task(req)
        state.in_flight[task.task_id] = time.time()
        state.emit({"type": "task.accepted", "task_id": task.task_id, "model": req.model})
        _trace("task.accepted", task_id=task.task_id, model=req.model, stream=req.stream)

        try:
            if req.stream:
                return StreamingResponse(
                    _sse_chat(runner, task, req, http_req, state, tracing),
                    media_type="text/event-stream",
                )
            return await asyncio.to_thread(
                _run_chat_blocking, runner, task, req, state, tracing,
            )
        finally:
            # _sse_chat / _run_chat_blocking own their own per-task cleanup;
            # here we only ensure the cancel flag is gone.
            state.cancel_flags.pop(task.task_id, None)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _logprobs_to_out(lp: Any) -> TokenLogprobsOut | None:
    if lp is None:
        return None
    return TokenLogprobsOut(
        token=lp.token,
        logprob=lp.logprob,
        top_logprobs=[
            TopLogprobOut(token=tp.token, logprob=tp.logprob)
            for tp in lp.top_logprobs
        ],
    )


def _run_chat_blocking(
    runner: Runner,
    task: GenerationTask,
    req: ChatCompletionRequest,
    state: _ServerState,
    tracing: bool,
) -> ChatCompletionResponse:
    text_parts: list[str] = []
    completion_tokens = 0
    finish_reason = "length"
    logprob_items: list[TokenLogprobsOut] = []
    start = time.perf_counter()

    try:
        for chunk in runner.generate(task):
            if state.cancel_flags.get(task.task_id):
                finish_reason = "cancelled"
                break
            text_parts.append(chunk.text)
            completion_tokens += 1
            lp = _logprobs_to_out(chunk.logprobs)
            if lp is not None:
                logprob_items.append(lp)
            if chunk.is_final:
                finish_reason = "stop" if completion_tokens < req.max_tokens else "length"
    finally:
        state.in_flight.pop(task.task_id, None)
        if finish_reason == "cancelled":
            state.cancelled += 1
        else:
            state.completed += 1
        state.emit({
            "type": "task.done",
            "task_id": task.task_id,
            "finish_reason": finish_reason,
            "tokens": completion_tokens,
        })

    elapsed = time.perf_counter() - start
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    if tracing:
        logger.info(
            "trace: task.done task=%s tokens=%d elapsed=%.2fs tps=%.1f finish=%s",
            task.task_id, completion_tokens, elapsed, tps, finish_reason,
        )
    else:
        logger.info(
            "completed %d tokens in %.2fs (%.1f tok/s) task=%s",
            completion_tokens, elapsed, tps, task.task_id[:8],
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{task.task_id[:8]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="".join(text_parts)),
                finish_reason=finish_reason,
                logprobs=ChoiceLogprobsOut(content=logprob_items) if logprob_items else None,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        ),
    )


async def _sse_chat(
    runner: Runner,
    task: GenerationTask,
    req: ChatCompletionRequest,
    http_req: Request,
    state: _ServerState,
    tracing: bool,
) -> AsyncIterator[bytes]:
    """OpenAI-style SSE: chunked deltas terminated by ``data: [DONE]``."""

    chatcmpl_id = f"chatcmpl-{task.task_id[:8]}"
    created = int(time.time())
    completion_tokens = 0
    finish_reason = "length"

    def _delta(text: str, *, role: str | None = None,
               finish: str | None = None,
               lp: TokenLogprobsOut | None = None) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        if role is not None:
            delta["role"] = role
        if text:
            delta["content"] = text
        choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish}
        if lp is not None:
            choice["logprobs"] = {"content": [lp.model_dump()]}
        return {
            "id": chatcmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [choice],
        }

    # Initial role chunk (matches OpenAI behaviour).
    yield _sse(_delta("", role="assistant"))

    # Run the blocking generator in a thread and bridge it to async via a queue.
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    SENTINEL = object()

    def _produce() -> None:
        try:
            for chunk in runner.generate(task):
                queue.put_nowait(chunk)
        except Exception as err:  # noqa: BLE001
            queue.put_nowait(err)
        finally:
            queue.put_nowait(SENTINEL)

    producer = asyncio.create_task(asyncio.to_thread(_produce))

    try:
        while True:
            # Cooperative cancellation: check once per token slot.
            if await http_req.is_disconnected():
                state.cancel_flags[task.task_id] = True
            if state.cancel_flags.get(task.task_id):
                finish_reason = "cancelled"
                break

            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                logger.exception("engine error in stream", exc_info=item)
                yield _sse({"error": str(item)})
                finish_reason = "error"
                break

            chunk = item
            completion_tokens += 1
            lp = _logprobs_to_out(chunk.logprobs)
            yield _sse(_delta(chunk.text, lp=lp))
            if chunk.is_final:
                finish_reason = "stop" if completion_tokens < req.max_tokens else "length"
                break
    finally:
        # Make sure the producer thread isn't left hanging.
        with contextlib.suppress(Exception):
            await producer
        state.in_flight.pop(task.task_id, None)
        if finish_reason == "cancelled":
            state.cancelled += 1
        else:
            state.completed += 1
        state.emit({
            "type": "task.done",
            "task_id": task.task_id,
            "finish_reason": finish_reason,
            "tokens": completion_tokens,
        })
        if tracing:
            logger.info(
                "trace: task.done(stream) task=%s tokens=%d finish=%s",
                task.task_id, completion_tokens, finish_reason,
            )

    yield _sse(_delta("", finish=finish_reason))
    yield b"data: [DONE]\n\n"
