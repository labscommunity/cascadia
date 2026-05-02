"""Distributed speculative decoding engine.

Architecture:
- **Driver** (rank 0) runs the spec decode loop: holds a local draft model
  (small, fast) and a `DistributedMaskedReq` that wraps the target model
  spread across N stages.
- **Workers** (rank 1..N-1) run a per-stage portion of the target model and
  respond to control-frame commands (FORWARD / REWIND / RESET) from upstream.
  Last stage applies lm_head and returns logits.

Activation flow per spec round:
1. Driver runs draft locally → K candidate token IDs.
2. Driver runs stage-0 forward on (prev_correction + K drafts) → hidden_states.
3. Driver sends FORWARD frame downstream.
4. Each worker runs its layers on the activations, forwards downstream.
5. Last stage runs lm_head → logits, sends LOGITS_RESPONSE back upstream.
6. Logits propagate back to the driver.
7. Driver runs the standard spec_decode_greedy_stream accept logic.
8. On reject, driver sends REWIND(N) downstream so each stage trims its
   physical KV cache.

Cache rewind uses OV's `query_state` / `set_state` to slice out the last K
positions of each layer's KV cache. This is "physical trim" — slower than
mask-based rewind (~40 ms per call on Intel iGPU per rainier #20) but
required because the v3 stateful shards don't accept attention_mask.

For 8B that fits on one GPU, this engine is **slower** than monolithic
`ov-spec` (because of network + physical-trim cost). The motivation is the
70B story — a model that doesn't fit on any one node.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tahoma.shared.shard import ShardSpec
from tahoma.shared.topology import PeerLayout
from tahoma.shared.types import Chunk, GenerationTask, LoadProgress, TaskId
from tahoma.worker.engines.base import Builder, Engine
from tahoma.worker.engines.openvino.dist_spec_protocol import (
    FrameKind,
    recv_frame,
    send_frame,
)
from tahoma.worker.engines.openvino.ov_runtime import (
    _build_rotary,
    _compute_cos_sin,
    _load_pipeline_config,
    _load_stage_config,
)
from tahoma.worker.engines.openvino.spec_decode import (
    SpecDecodeStats,
    spec_decode_greedy_stream,
)
from tahoma.worker.transport import ActivationClient, ActivationServer

logger = logging.getLogger(__name__)


def _physical_rewind(req: Any, k: int) -> None:
    """Trim the last `k` positions from each KV cache state via OV state API.

    Assumes seq_len is the second-to-last dim (canonical KV layout
    [batch, num_kv_heads, seq_len, head_dim]). Costs ~40 ms per call on
    Intel iGPU per rainier DISCOVERY #20.
    """
    import openvino as ov  # type: ignore[import-untyped]

    if k <= 0:
        return
    for var_state in req.query_state():
        data = var_state.state.data
        seq_len = data.shape[-2]
        if seq_len <= k:
            # Trim to empty (rewind everything).
            new_data = data[..., :0, :].copy()
        else:
            new_data = data[..., :-k, :].copy()
        var_state.state = ov.Tensor(new_data)


# ---------------------------------------------------------------------------
# Driver side: DistributedMaskedReq + OVDistributedSpecEngine + Builder
# ---------------------------------------------------------------------------


class DistributedMaskedReq:
    """Wraps a multi-stage target as a single `MaskedReq`-shaped object.

    Implements `reset()`, `feed(input_ids) -> logits`, `rewind(k)`. The
    driver-local stage runs in-process; downstream stages receive control
    frames and reply.
    """

    __slots__ = (
        "local_req", "local_inputs", "local_outputs", "rotary", "hidden_size",
        "downstream_sock", "position",
    )

    def __init__(
        self,
        local_req: Any,
        local_inputs: list[str],
        local_outputs: list[str],
        rotary: Any,
        hidden_size: int,
        downstream_sock: Any,
    ):
        self.local_req = local_req
        self.local_inputs = local_inputs
        self.local_outputs = local_outputs
        self.rotary = rotary
        self.hidden_size = hidden_size
        self.downstream_sock = downstream_sock
        self.position = 0

    def reset(self) -> None:
        self.local_req.reset_state()
        send_frame(self.downstream_sock, FrameKind.RESET)
        self.position = 0

    def feed(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward input_ids through stage-0 then the rest of the pipeline.

        Returns logits (1, seq_len, vocab) from the last stage.
        """
        seq_len = input_ids.shape[1]
        positions = np.arange(
            self.position, self.position + seq_len, dtype=np.int64,
        ).reshape(1, seq_len)
        cos, sin = _compute_cos_sin(self.rotary, self.hidden_size, positions)

        # Stage-0 local forward: v3 IR signature (input_ids|hs, cos, sin).
        feed = {
            self.local_inputs[0]: input_ids.astype(np.int64),
            self.local_inputs[1]: cos,
            self.local_inputs[2]: sin,
        }
        self.local_req.infer(feed)
        hs = self.local_req.get_output_tensor(0).data

        # Send hidden_states downstream as a FORWARD frame.
        send_frame(
            self.downstream_sock,
            FrameKind.FORWARD,
            tensor=hs.astype(np.float16),
        )

        # Wait for LOGITS_RESPONSE bouncing back through the chain.
        kind, _, logits = recv_frame(self.downstream_sock)
        if kind != FrameKind.LOGITS_RESPONSE or logits is None:
            raise RuntimeError(f"expected LOGITS_RESPONSE, got {kind}")

        self.position += seq_len
        return logits

    def rewind(self, k: int) -> None:
        if k <= 0:
            return
        _physical_rewind(self.local_req, k)
        send_frame(self.downstream_sock, FrameKind.REWIND, int_arg=k)
        self.position -= k


@dataclass
class _DistSpecActive:
    task: GenerationTask
    iterator: Any                 # generator yielding token ids
    stats: SpecDecodeStats
    emitted: int = 0
    pending_ids: list[int] = field(default_factory=list)
    last_text: str = ""


class OVDistributedSpecEngine(Engine):
    """Driver-side engine: runs spec decode with a distributed target."""

    def __init__(
        self,
        local_draft: Any,           # MaskedReq for the (small) draft model
        distributed_target: DistributedMaskedReq,
        tokenizer: Any,
        k: int = 4,
    ):
        self._draft = local_draft
        self._target = distributed_target
        self._tokenizer = tokenizer
        self._k = k
        self._active: dict[TaskId, _DistSpecActive] = {}
        self._pending: list[GenerationTask] = []

    def warmup(self) -> None:
        try:
            ids = self._tokenizer("Hi", return_tensors="np").input_ids.astype("int64")
            stats = SpecDecodeStats()
            list(spec_decode_greedy_stream(
                self._target, self._draft, ids, max_tokens=2, k=self._k, stats=stats,
            ))
            logger.info("warmup ok (dist-spec K=%d)", self._k)
        except Exception as err:  # noqa: BLE001
            logger.warning("warmup failed: %s", err)

    def submit(self, task: GenerationTask) -> None:
        if task.task_id in self._active or any(t.task_id == task.task_id for t in self._pending):
            return
        self._pending.append(task)

    def _start(self, task: GenerationTask) -> _DistSpecActive:
        ids = self._tokenizer(task.prompt, return_tensors="np").input_ids.astype("int64")
        stats = SpecDecodeStats()
        gen = spec_decode_greedy_stream(
            self._target, self._draft, ids,
            max_tokens=task.max_tokens, k=self._k, stats=stats,
        )
        return _DistSpecActive(task=task, iterator=gen, stats=stats)

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        if not self._active and self._pending:
            task = self._pending.pop(0)
            self._active[task.task_id] = self._start(task)
            logger.info("task %s active (dist-spec K=%d)", task.task_id[:8], self._k)

        if not self._active:
            return

        task_id, active = next(iter(self._active.items()))

        try:
            tok = next(active.iterator)
        except StopIteration:
            logger.info(
                "task %s done: %d tokens, %d steps, accept=%.2f",
                task_id[:8], active.emitted, active.stats.n_steps,
                active.stats.accept_rate,
            )
            yield task_id, Chunk(task_id=task_id, token_id=0, text="", is_final=True)
            del self._active[task_id]
            return

        active.emitted += 1
        active.pending_ids.append(int(tok))
        full_text = self._tokenizer.decode(active.pending_ids, skip_special_tokens=True)
        delta = full_text[len(active.last_text):]
        active.last_text = full_text

        is_final = active.emitted >= active.task.max_tokens
        yield task_id, Chunk(
            task_id=task_id, token_id=int(tok), text=delta, is_final=is_final,
        )

        if is_final:
            logger.info(
                "task %s done: %d tokens, %d steps, accept=%.2f",
                task_id[:8], active.emitted, active.stats.n_steps,
                active.stats.accept_rate,
            )
            del self._active[task_id]

    def close(self) -> None:
        self._active.clear()


class OVDistributedSpecBuilder(Builder):
    """Driver-side builder: loads draft + local stage-0 + connects downstream."""

    def __init__(
        self,
        pipeline_dir: str,
        draft_model_path: str,
        device: str = "GPU",
        weight_format: str = "int4",
        k: int = 4,
    ):
        self._pipeline_dir = Path(pipeline_dir)
        self._draft_model_path = draft_model_path
        self._device = device
        self._weight_format = weight_format
        self._k = k
        self._local_req: Any = None
        self._local_inputs: list[str] = []
        self._local_outputs: list[str] = []
        self._rotary: Any = None
        self._hidden_size: int = 0
        self._draft_req: Any = None
        self._tokenizer: Any = None
        self._downstream: ActivationClient | None = None

    def connect(self, peers: PeerLayout) -> None:
        if peers.upstream is not None:
            raise RuntimeError("driver should not have an upstream")
        if peers.downstream is None:
            raise RuntimeError(
                "driver requires a downstream peer (use --next host:port)"
            )
        client = ActivationClient(peers.downstream.host, peers.downstream.port)
        client.connect()
        self._downstream = client

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        import openvino as ov  # type: ignore[import-untyped]
        from transformers import AutoTokenizer

        from tahoma.worker.engines.openvino.optimum_engine import resolve_or_export_ov_ir
        from tahoma.worker.engines.openvino.spec_decode import MaskedReq

        # Load pipeline metadata for the (rainier-style) target.
        pipeline_cfg = _load_pipeline_config(self._pipeline_dir)
        stage_dir = self._pipeline_dir / "stage_0"
        stage_cfg = _load_stage_config(stage_dir)
        if not stage_cfg.get("has_embed", False):
            raise RuntimeError(
                "driver expects stage 0 to have embed (has_embed=true)"
            )
        self._hidden_size = pipeline_cfg["hidden_size"]
        model_id = pipeline_cfg["model_id"]

        yield LoadProgress(0, None, "compiling target stage 0")
        core = ov.Core()
        target_compiled = core.compile_model(str(stage_dir / "openvino_model.xml"), self._device)
        self._local_req = target_compiled.create_infer_request()
        self._local_inputs = [list(inp.names)[0] for inp in target_compiled.inputs]
        self._local_outputs = [list(out.names)[0] for out in target_compiled.outputs]

        yield LoadProgress(0, None, "loading rotary + tokenizer")
        self._rotary, _ = _build_rotary(model_id)
        tok_dir = self._pipeline_dir / "tokenizer"
        if tok_dir.is_dir():
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
            except (ValueError, KeyError, ImportError):
                self._tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

        yield LoadProgress(0, None, f"resolving draft {self._draft_model_path}")
        draft_path = resolve_or_export_ov_ir(
            self._draft_model_path, weight_format=self._weight_format,
        )

        yield LoadProgress(0, None, "compiling draft")
        draft_compiled = core.compile_model(f"{draft_path}/openvino_model.xml", self._device)
        draft_req = draft_compiled.create_infer_request()
        has_beam = any(
            any("beam_idx" in n for n in inp.get_names()) for inp in draft_compiled.inputs
        )
        self._draft_req = MaskedReq(draft_req, has_beam)

        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._local_req is None or self._draft_req is None or self._tokenizer is None:
            raise RuntimeError("call load() before build()")
        if self._downstream is None:
            raise RuntimeError("call connect() before build()")
        target = DistributedMaskedReq(
            local_req=self._local_req,
            local_inputs=self._local_inputs,
            local_outputs=self._local_outputs,
            rotary=self._rotary,
            hidden_size=self._hidden_size,
            downstream_sock=self._downstream._sock,
        )
        return OVDistributedSpecEngine(
            local_draft=self._draft_req,
            distributed_target=target,
            tokenizer=self._tokenizer,
            k=self._k,
        )

    def close(self) -> None:
        if self._downstream is not None:
            self._downstream.close()
            self._downstream = None


# ---------------------------------------------------------------------------
# Worker side: OVDistSpecWorkerEngine + OVDistSpecWorkerBuilder
# ---------------------------------------------------------------------------


class OVDistSpecWorkerEngine(Engine):
    """Per-stage worker: handles FORWARD / REWIND / RESET frames in a loop."""

    def __init__(
        self,
        rank: int,
        is_last: bool,
        local_req: Any,
        local_inputs: list[str],
        rotary: Any,
        hidden_size: int,
        upstream: ActivationServer,
        downstream: ActivationClient | None,
    ):
        self.rank = rank
        self.is_last = is_last
        self._req = local_req
        self._inputs = local_inputs
        self._rotary = rotary
        self._hidden_size = hidden_size
        self._upstream = upstream
        self._downstream = downstream
        self._position = 0

    def warmup(self) -> None:
        logger.info("worker warmup skipped (driven by driver)")

    def submit(self, task: GenerationTask) -> None:
        raise RuntimeError("workers cannot accept tasks directly")

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        self._handle_one_frame()
        return ()  # workers don't yield chunks; they service frames

    def _handle_one_frame(self) -> None:
        kind, arg, tensor = recv_frame(self._upstream._client_sock)

        if kind == FrameKind.FORWARD:
            assert tensor is not None
            seq_len = tensor.shape[1]
            positions = np.arange(
                self._position, self._position + seq_len, dtype=np.int64,
            ).reshape(1, seq_len)
            cos, sin = _compute_cos_sin(self._rotary, self._hidden_size, positions)
            feed = {
                self._inputs[0]: tensor.astype(np.float32),  # hidden_states from upstream
                self._inputs[1]: cos,
                self._inputs[2]: sin,
            }
            self._req.infer(feed)
            out = self._req.get_output_tensor(0).data
            self._position += seq_len

            if self.is_last:
                # Stage's IR includes lm_head; output IS logits.
                send_frame(
                    self._upstream._client_sock,
                    FrameKind.LOGITS_RESPONSE,
                    tensor=out.astype(np.float16),
                )
            else:
                # Forward downstream, await LOGITS_RESPONSE, relay back.
                assert self._downstream is not None
                send_frame(
                    self._downstream._sock,
                    FrameKind.FORWARD,
                    tensor=out.astype(np.float16),
                )
                kind2, _, logits = recv_frame(self._downstream._sock)
                if kind2 != FrameKind.LOGITS_RESPONSE or logits is None:
                    raise RuntimeError(f"expected LOGITS_RESPONSE, got {kind2}")
                send_frame(
                    self._upstream._client_sock,
                    FrameKind.LOGITS_RESPONSE,
                    tensor=logits,
                )
            return

        if kind == FrameKind.REWIND:
            _physical_rewind(self._req, arg)
            self._position -= arg
            if not self.is_last:
                assert self._downstream is not None
                send_frame(self._downstream._sock, FrameKind.REWIND, int_arg=arg)
            return

        if kind == FrameKind.RESET:
            self._req.reset_state()
            self._position = 0
            if not self.is_last:
                assert self._downstream is not None
                send_frame(self._downstream._sock, FrameKind.RESET)
            return

        raise RuntimeError(f"unexpected frame kind {kind}")

    def close(self) -> None:
        if self._upstream is not None:
            self._upstream.close()
        if self._downstream is not None:
            self._downstream.close()


class OVDistSpecWorkerBuilder(Builder):
    """Worker-side builder: loads stage_<rank> IR + opens upstream/downstream."""

    def __init__(
        self,
        pipeline_dir: str,
        rank: int,
        total: int,
        device: str = "GPU",
    ):
        self._pipeline_dir = Path(pipeline_dir)
        self._rank = rank
        self._total = total
        self._device = device
        self._req: Any = None
        self._inputs: list[str] = []
        self._rotary: Any = None
        self._hidden_size: int = 0
        self._upstream: ActivationServer | None = None
        self._downstream: ActivationClient | None = None
        self._listen_host = "0.0.0.0"
        self._listen_port: int | None = None

    def configure_listen(self, host: str, port: int) -> None:
        self._listen_host = host
        self._listen_port = port

    def connect(self, peers: PeerLayout) -> None:
        if peers.upstream is None:
            raise RuntimeError("worker requires an upstream peer")
        if self._listen_port is None:
            raise RuntimeError("configure_listen() required for worker")
        server = ActivationServer(self._listen_host, self._listen_port)
        server.start()
        self._upstream = server

        if peers.downstream is not None:
            client = ActivationClient(peers.downstream.host, peers.downstream.port)
            client.connect()
            self._downstream = client

        self._upstream.accept()

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        import openvino as ov  # type: ignore[import-untyped]

        pipeline_cfg = _load_pipeline_config(self._pipeline_dir)
        stage_dir = self._pipeline_dir / f"stage_{self._rank}"
        stage_cfg = _load_stage_config(stage_dir)
        is_last = (self._rank == self._total - 1)
        if is_last and not stage_cfg.get("has_head", False):
            raise RuntimeError(
                f"last stage (rank {self._rank}) expected has_head=true"
            )
        self._hidden_size = pipeline_cfg["hidden_size"]
        model_id = pipeline_cfg["model_id"]

        yield LoadProgress(0, None, f"compiling stage {self._rank}")
        core = ov.Core()
        compiled = core.compile_model(str(stage_dir / "openvino_model.xml"), self._device)
        self._req = compiled.create_infer_request()
        self._inputs = [list(inp.names)[0] for inp in compiled.inputs]

        yield LoadProgress(0, None, "loading rotary")
        self._rotary, _ = _build_rotary(model_id)

        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._req is None or self._upstream is None:
            raise RuntimeError("call connect() and load() before build()")
        is_last = (self._rank == self._total - 1)
        return OVDistSpecWorkerEngine(
            rank=self._rank,
            is_last=is_last,
            local_req=self._req,
            local_inputs=self._inputs,
            rotary=self._rotary,
            hidden_size=self._hidden_size,
            upstream=self._upstream,
            downstream=self._downstream,
        )

    def close(self) -> None:
        if self._upstream is not None:
            self._upstream.close()
            self._upstream = None
        if self._downstream is not None:
            self._downstream.close()
            self._downstream = None
