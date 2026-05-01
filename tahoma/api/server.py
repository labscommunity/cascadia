"""OpenAI-compatible HTTP server (MVP, non-streaming).

Implements `/v1/chat/completions` (non-streaming), `/v1/models`, and
`/health`. Streaming and tool calling are roadmap.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tahoma.shared.types import GenerationTask
from tahoma.worker.runner import Runner

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


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


def make_app(runner: Runner, model_id: str) -> FastAPI:
    app = FastAPI(title="Tahoma", version="0.0.1")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "tahoma"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        if req.stream:
            raise HTTPException(status_code=501, detail="streaming not implemented in MVP")

        prompt = _format_prompt(req.messages)
        task = GenerationTask(
            task_id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        text_parts: list[str] = []
        completion_tokens = 0
        finish_reason = "length"
        start = time.perf_counter()

        for chunk in runner.generate(task):
            text_parts.append(chunk.text)
            completion_tokens += 1
            if chunk.is_final:
                # If we stopped due to max_tokens, finish_reason="length"; else "stop".
                finish_reason = "stop" if completion_tokens < req.max_tokens else "length"

        elapsed = time.perf_counter() - start
        tps = completion_tokens / elapsed if elapsed > 0 else 0.0
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
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=0,
                completion_tokens=completion_tokens,
                total_tokens=completion_tokens,
            ),
        )

    return app
