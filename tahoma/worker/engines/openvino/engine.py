"""OpenVINO engine for a single pipeline stage.

Wraps the selective safetensors `ModelShard` (loader.py) and the TCP activation
transport into the `Engine` / `Builder` ABC.

Stage roles:

- **first stage**: tokenize → embed → forward → send activation downstream;
  receive token back; yield as a `Chunk` to the caller.
- **middle stage**: receive activation upstream → forward → send downstream;
  receive token from downstream; relay back upstream.
- **last stage**: receive activation upstream → forward → lm_head → sample;
  send token back upstream.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId
from tahoma.worker.engines.base import Builder, Engine
from tahoma.worker.engines.openvino.loader import ModelShard
from tahoma.worker.transport import ActivationClient, ActivationServer

logger = logging.getLogger(__name__)


@dataclass
class _ActiveTask:
    task: GenerationTask
    current_ids: np.ndarray  # [1, current_seq_len]
    generated: list[int] = field(default_factory=list)


class OpenVINOEngine(Engine):
    """Per-stage inference engine, dispatched by `ShardSpec.is_first/last_stage`."""

    def __init__(
        self,
        spec: ShardSpec,
        shard: ModelShard,
        upstream_server: ActivationServer | None,
        downstream_client: ActivationClient | None,
    ):
        self.spec = spec
        self.shard = shard
        self._upstream = upstream_server
        self._downstream = downstream_client
        self._active: dict[TaskId, _ActiveTask] = {}
        self._pending: list[GenerationTask] = []

    def warmup(self) -> None:
        """First stage runs a 1-token forward to warm graphs.

        Other stages cannot warm without a feed; they simply log.
        """
        if self.spec.is_first_stage:
            try:
                bos = getattr(self.shard.tokenizer, "bos_token_id", None) or 1
                dummy_ids = np.array([[bos]], dtype=np.int64)
                hs = self.shard.embed(dummy_ids)
                self.shard.forward_layers(hs)
                logger.info("warmup ok")
            except Exception as err:  # noqa: BLE001
                logger.warning("warmup failed: %s", err)
        else:
            logger.info("warmup skipped on non-first stage")

    def submit(self, task: GenerationTask) -> None:
        if not self.spec.is_first_stage:
            raise RuntimeError("submit() is only valid on the first stage")
        if task.task_id in self._active or any(t.task_id == task.task_id for t in self._pending):
            return  # idempotent
        self._pending.append(task)

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        if self.spec.is_first_stage:
            yield from self._step_first()
            return
        if self.spec.is_last_stage:
            self._step_last()
            return
        self._step_middle()

    # --------------- first stage ---------------

    def _step_first(self) -> Iterable[tuple[TaskId, Chunk]]:
        # Activate the next pending task if no task is currently active. MVP
        # serves one task at a time.
        if not self._active and self._pending:
            task = self._pending.pop(0)
            input_ids = self.shard.tokenizer.encode(task.prompt, return_tensors="np")
            self._active[task.task_id] = _ActiveTask(task=task, current_ids=input_ids)
            logger.info(
                "task %s active: prompt_tokens=%d", task.task_id[:8], input_ids.shape[1],
            )

        if not self._active:
            return

        task_id = next(iter(self._active))
        active = self._active[task_id]

        # Stage-0 forward
        hs = self.shard.embed(active.current_ids)
        hs = self.shard.forward_layers(hs)

        if self.spec.is_last_stage:
            # 1-stage pipeline (single node)
            logits = self.shard.lm_head(hs)
            next_token = int(np.argmax(logits[0, -1, :]))
        else:
            assert self._downstream is not None, "non-last first stage needs downstream"
            self._downstream.send(hs)
            token_array, _ = self._downstream.recv()
            next_token = int(token_array[0])

        active.generated.append(next_token)
        active.current_ids = np.append(active.current_ids, [[next_token]], axis=1)

        text = self.shard.tokenizer.decode([next_token], skip_special_tokens=True)
        eos = getattr(self.shard.tokenizer, "eos_token_id", None)
        is_final = (
            len(active.generated) >= active.task.max_tokens
            or (eos is not None and next_token == eos)
        )

        yield task_id, Chunk(
            task_id=task_id,
            token_id=next_token,
            text=text,
            is_final=is_final,
        )

        if is_final:
            del self._active[task_id]

    # --------------- middle stage ---------------

    def _step_middle(self) -> None:
        assert self._upstream is not None and self._downstream is not None
        activation, _ = self._upstream.recv()
        output = self.shard.forward_layers(activation)
        self._downstream.send(output)
        token_array, _ = self._downstream.recv()
        self._upstream.send(token_array)

    # --------------- last stage ---------------

    def _step_last(self) -> None:
        assert self._upstream is not None
        activation, _ = self._upstream.recv()
        hs = self.shard.forward_layers(activation)
        logits = self.shard.lm_head(hs)
        next_token = int(np.argmax(logits[0, -1, :]))
        token_array = np.array([next_token], dtype=np.int32)
        self._upstream.send(token_array)

    def close(self) -> None:
        if self._upstream is not None:
            self._upstream.close()
        if self._downstream is not None:
            self._downstream.close()


class OpenVINOBuilder(Builder):
    """Build an OpenVINOEngine for one stage.

    Lifecycle: `configure_listen` → `connect(peers)` → `load(shard)` → `build()`.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._spec: ShardSpec | None = None
        self._shard: ModelShard | None = None
        self._upstream: ActivationServer | None = None
        self._downstream: ActivationClient | None = None
        self._listen_host: str = "0.0.0.0"
        self._listen_port: int | None = None

    def configure_listen(self, host: str, port: int) -> None:
        """Bind address used when this stage is not the first."""
        self._listen_host = host
        self._listen_port = port

    def connect(self, peers: PeerLayout) -> None:
        # Three-step ordering avoids deadlock for any pipeline length:
        # 1. bind/listen (so upstream peer can connect to us)
        # 2. connect outbound (with retry, so we tolerate downstream not-yet-ready)
        # 3. accept inbound (blocks until upstream connects)
        if peers.upstream is not None:
            if self._listen_port is None:
                raise RuntimeError(
                    "configure_listen() must be called before connect() "
                    "for stages that have an upstream peer"
                )
            server = ActivationServer(self._listen_host, self._listen_port)
            server.start()
            self._upstream = server

        if peers.downstream is not None:
            client = ActivationClient(peers.downstream.host, peers.downstream.port)
            client.connect()
            self._downstream = client

        if self._upstream is not None:
            self._upstream.accept()

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        self._spec = shard
        yield LoadProgress(0, None, "starting")
        self._shard = ModelShard(spec=shard, model_path=self._model_path)
        yield LoadProgress(0, None, "loading weights")
        self._shard.load()
        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._shard is None or self._spec is None:
            raise RuntimeError("call load() before build()")
        return OpenVINOEngine(
            spec=self._spec,
            shard=self._shard,
            upstream_server=self._upstream,
            downstream_client=self._downstream,
        )

    def close(self) -> None:
        if self._upstream is not None:
            self._upstream.close()
            self._upstream = None
        if self._downstream is not None:
            self._downstream.close()
            self._downstream = None
