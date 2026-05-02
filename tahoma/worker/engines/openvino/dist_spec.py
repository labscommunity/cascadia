"""Distributed speculative decoding engine (v5 shards, mask-based rewind).

Architecture
------------
- **Driver** (rank 0) runs the spec decode loop: holds a local draft model
  (small, fast) and a `DistributedMaskedReq` that wraps the target model
  spread across N stages.
- **Workers** (rank 1..N-1) run a per-stage portion of the target model and
  service FORWARD / RESET frames from upstream. Last stage applies lm_head
  and returns logits.

Activation flow per spec round
------------------------------
1. Driver runs draft locally → K candidate token IDs.
2. Driver runs stage-0 forward on (prev_correction + K drafts) using the
   v5 input convention (input_ids, attention_mask, position_ids, beam_idx).
   The attention_mask covers the full ``cache_len + new_tokens`` history,
   with already-rewound positions zeroed.
3. Driver sends FORWARD(logical_pos_start, attention_mask, hidden_states)
   downstream.
4. Each worker runs its layers with the same attention_mask + a derived
   position_ids, forwards downstream.
5. Last stage runs lm_head → logits, sends LOGITS_RESPONSE back upstream.
6. Logits propagate back to the driver.
7. Driver runs the standard spec_decode_greedy_stream accept logic.
8. On reject, driver flips bits in `valid_mask` (free) and decrements
   `logical_pos`. No round-trip — workers see the new mask on the next
   FORWARD frame.

Why this is faster than the previous physical-rewind protocol
-------------------------------------------------------------
v3 shards exposed only ``(input_ids|hs, cos, sin)`` and rewind required
``query_state``/``set_state`` round-trips against each layer (~40 ms per
call on Intel iGPU per rainier DISCOVERY #20). v5 shards accept an
explicit attention_mask, so rewind becomes pure driver-side bookkeeping
and the per-step network footprint is the same as plain dist inference
plus one FORWARD per spec round of K+1 verification tokens.
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
    recv_forward_body,
    recv_kind,
    recv_logits_body,
    send_forward,
    send_logits,
    send_reset,
)
from tahoma.worker.engines.openvino.spec_decode import (
    SpecDecodeStats,
    spec_decode_greedy_stream,
)
from tahoma.worker.transport import ActivationClient, ActivationServer

logger = logging.getLogger(__name__)


def _load_pipeline_config(pipeline_dir: Path) -> dict:
    with (pipeline_dir / "pipeline_config.json").open() as f:
        return json.load(f)


def _load_stage_config(stage_dir: Path) -> dict:
    with (stage_dir / "stage_config.json").open() as f:
        return json.load(f)


def _v5_inputs(compiled: Any) -> dict[str, str]:
    """Map canonical v5 input names to their actual tensor name on this IR."""
    by_name: dict[str, str] = {}
    for inp in compiled.inputs:
        names = list(inp.names)
        for canonical in ("input_ids", "hidden_states", "attention_mask",
                          "position_ids", "beam_idx"):
            if any(canonical == n or canonical in n for n in names):
                by_name[canonical] = names[0]
                break
    return by_name


# ---------------------------------------------------------------------------
# Driver side
# ---------------------------------------------------------------------------


class DistributedMaskedReq:
    """Wraps a multi-stage target as a single MaskedReq-shaped object.

    Mirrors `mini_coord_spec_mbatch.DistributedTargetMaskedReq` from rainier
    but speaks the tahoma framed protocol.
    """

    __slots__ = (
        "stage0", "stage0_inputs", "downstream_sock",
        "valid_mask", "cache_len", "logical_pos",
    )

    def __init__(
        self,
        stage0: Any,
        stage0_inputs: dict[str, str],
        downstream_sock: Any,
    ):
        self.stage0 = stage0
        self.stage0_inputs = stage0_inputs
        self.downstream_sock = downstream_sock
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self) -> None:
        self.stage0.reset_state()
        send_reset(self.downstream_sock)
        self.valid_mask[:] = 1
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids: np.ndarray) -> np.ndarray:
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new_size = max(total * 2, len(self.valid_mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[: len(self.valid_mask)] = self.valid_mask
            self.valid_mask = new_mask

        attn = np.empty((1, total), dtype=np.int64)
        attn[0, : self.cache_len] = self.valid_mask[: self.cache_len]
        attn[0, self.cache_len :] = 1
        pos = np.arange(
            self.logical_pos, self.logical_pos + n, dtype=np.int64,
        ).reshape(1, n)

        feed = {
            self.stage0_inputs["input_ids"]: input_ids.astype(np.int64),
            self.stage0_inputs["attention_mask"]: attn,
            self.stage0_inputs["position_ids"]: pos,
            self.stage0_inputs["beam_idx"]: np.zeros(1, dtype=np.int32),
        }
        self.stage0.infer(feed)
        hidden = self.stage0.get_output_tensor(0).data.copy()

        send_forward(
            self.downstream_sock,
            logical_pos_start=self.logical_pos,
            attention_mask=attn,
            hidden_states=hidden.astype(np.float16),
        )
        kind = recv_kind(self.downstream_sock)
        if kind != FrameKind.LOGITS_RESPONSE:
            raise RuntimeError(f"expected LOGITS_RESPONSE, got {kind}")
        logits = recv_logits_body(self.downstream_sock)

        self.cache_len += n
        self.logical_pos += n
        return logits

    def rewind(self, k: int) -> None:
        if k <= 0:
            return
        self.valid_mask[self.cache_len - k : self.cache_len] = 0
        self.logical_pos -= k


@dataclass
class _DistSpecActive:
    task: GenerationTask
    iterator: Any
    stats: SpecDecodeStats
    emitted: int = 0
    pending_ids: list[int] = field(default_factory=list)
    last_text: str = ""


class OVDistributedSpecEngine(Engine):
    """Driver-side engine: spec decode against a distributed target."""

    def __init__(
        self,
        local_draft: Any,
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
            logger.info("warmup ok (dist-spec v5 K=%d)", self._k)
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
            logger.info("task %s active (dist-spec v5 K=%d)", task.task_id[:8], self._k)

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
        cache_dir: str | None = None,
        kv_cache_precision: str | None = None,
        dyn_quant_group: str | None = None,
    ):
        self._pipeline_dir = Path(pipeline_dir)
        self._draft_model_path = draft_model_path
        self._device = device
        self._weight_format = weight_format
        self._k = k
        self._cache_dir = cache_dir
        self._kv_cache_precision = kv_cache_precision
        self._dyn_quant_group = dyn_quant_group
        self._stage0: Any = None
        self._stage0_inputs: dict[str, str] = {}
        self._draft_req: Any = None
        self._tokenizer: Any = None
        self._downstream: ActivationClient | None = None

    def connect(self, peers: PeerLayout) -> None:
        if peers.upstream is not None:
            raise RuntimeError("driver should not have an upstream")
        if peers.downstream is None:
            raise RuntimeError("driver requires a downstream peer (--next host:port)")
        client = ActivationClient(peers.downstream.host, peers.downstream.port)
        client.connect()
        self._downstream = client

    def load(self, shard: ShardSpec) -> Iterable[LoadProgress]:
        # Validate before importing openvino so failures are cheap and the
        # error message is the actual problem rather than an ImportError.
        pipeline_cfg = _load_pipeline_config(self._pipeline_dir)
        stage_dir = self._pipeline_dir / "stage_0"
        stage_cfg = _load_stage_config(stage_dir)
        if not stage_cfg.get("has_embed", False):
            raise RuntimeError("driver expects stage 0 to have embed (has_embed=true)")
        if pipeline_cfg.get("export_version", "").startswith("v3"):
            raise RuntimeError(
                "driver requires v5 shards (canonical inputs); got v3. "
                "Re-export with scripts/export_cached_shards_v5.py."
            )
        model_id = pipeline_cfg["model_id"]

        import openvino as ov  # type: ignore[import-untyped]

        from tahoma.worker.engines.openvino._hub import force_offline, load_tokenizer
        from tahoma.worker.engines.openvino.optimum_engine import resolve_or_export_ov_ir
        from tahoma.worker.engines.openvino.spec_decode import MaskedReq

        force_offline()

        from tahoma.worker.engines.openvino._plugin import build_plugin_config
        plugin_config = build_plugin_config(
            self._cache_dir, self._kv_cache_precision, self._dyn_quant_group,
        )

        yield LoadProgress(0, None, "compiling target stage 0")
        core = ov.Core()
        target_compiled = core.compile_model(
            str(stage_dir / "openvino_model.xml"), self._device, plugin_config,
        )
        self._stage0 = target_compiled.create_infer_request()
        self._stage0_inputs = _v5_inputs(target_compiled)
        missing = {"input_ids", "attention_mask", "position_ids", "beam_idx"} - set(self._stage0_inputs)
        if missing:
            raise RuntimeError(f"stage 0 IR is missing v5 inputs: {sorted(missing)}")

        yield LoadProgress(0, None, "loading tokenizer")
        self._tokenizer = load_tokenizer(model_id)

        yield LoadProgress(0, None, f"resolving draft {self._draft_model_path}")
        draft_path = resolve_or_export_ov_ir(
            self._draft_model_path, weight_format=self._weight_format,
        )

        yield LoadProgress(0, None, "compiling draft")
        draft_compiled = core.compile_model(
            f"{draft_path}/openvino_model.xml", self._device, plugin_config,
        )
        draft_req = draft_compiled.create_infer_request()
        has_beam = any(
            any("beam_idx" in n for n in inp.get_names()) for inp in draft_compiled.inputs
        )
        self._draft_req = MaskedReq(draft_req, has_beam)

        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._stage0 is None or self._draft_req is None or self._tokenizer is None:
            raise RuntimeError("call load() before build()")
        if self._downstream is None:
            raise RuntimeError("call connect() before build()")
        target = DistributedMaskedReq(
            stage0=self._stage0,
            stage0_inputs=self._stage0_inputs,
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
# Worker side
# ---------------------------------------------------------------------------


class OVDistSpecWorkerEngine(Engine):
    """Per-stage worker: handles FORWARD / RESET frames in a loop."""

    def __init__(
        self,
        rank: int,
        is_last: bool,
        stage_req: Any,
        stage_inputs: dict[str, str],
        upstream: ActivationServer,
        downstream: ActivationClient | None,
    ):
        self.rank = rank
        self.is_last = is_last
        self._req = stage_req
        self._inputs = stage_inputs
        self._upstream = upstream
        self._downstream = downstream

    def warmup(self) -> None:
        logger.info("worker warmup skipped (driven by driver)")

    def submit(self, task: GenerationTask) -> None:
        raise RuntimeError("workers cannot accept tasks directly")

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        self._handle_one_frame()
        return ()

    def _handle_one_frame(self) -> None:
        kind = recv_kind(self._upstream._client_sock)

        if kind == FrameKind.FORWARD:
            logical_pos_start, attn, hidden = recv_forward_body(self._upstream._client_sock)
            new_tokens = hidden.shape[1]
            pos = np.arange(
                logical_pos_start, logical_pos_start + new_tokens, dtype=np.int64,
            ).reshape(1, new_tokens)

            feed = {
                self._inputs["hidden_states"]: hidden.astype(np.float32),
                self._inputs["attention_mask"]: attn,
                self._inputs["position_ids"]: pos,
                self._inputs["beam_idx"]: np.zeros(1, dtype=np.int32),
            }
            self._req.infer(feed)
            out = self._req.get_output_tensor(0).data

            if self.is_last:
                send_logits(self._upstream._client_sock, out.astype(np.float16))
            else:
                assert self._downstream is not None
                send_forward(
                    self._downstream._sock,
                    logical_pos_start=logical_pos_start,
                    attention_mask=attn,
                    hidden_states=out.astype(np.float16),
                )
                kind2 = recv_kind(self._downstream._sock)
                if kind2 != FrameKind.LOGITS_RESPONSE:
                    raise RuntimeError(f"expected LOGITS_RESPONSE, got {kind2}")
                logits = recv_logits_body(self._downstream._sock)
                send_logits(self._upstream._client_sock, logits)
            return

        if kind == FrameKind.RESET:
            self._req.reset_state()
            if not self.is_last:
                assert self._downstream is not None
                send_reset(self._downstream._sock)
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
        cache_dir: str | None = None,
        kv_cache_precision: str | None = None,
        dyn_quant_group: str | None = None,
    ):
        self._pipeline_dir = Path(pipeline_dir)
        self._rank = rank
        self._total = total
        self._device = device
        self._cache_dir = cache_dir
        self._kv_cache_precision = kv_cache_precision
        self._dyn_quant_group = dyn_quant_group
        self._req: Any = None
        self._inputs: dict[str, str] = {}
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
        # Validate before importing openvino (cheap-failure ordering).
        # The pipeline_config may not exist on a worker that only has its own
        # stage subdir copied (saving disk). Fall back to stage_config alone.
        stage_dir = self._pipeline_dir / f"stage_{self._rank}"
        if not stage_dir.is_dir():
            # Allow passing the stage dir directly (rainier-style worker layout).
            stage_dir = self._pipeline_dir
        stage_cfg = _load_stage_config(stage_dir)
        is_last = (self._rank == self._total - 1)
        if is_last and not stage_cfg.get("has_head", False):
            raise RuntimeError(f"last stage (rank {self._rank}) expected has_head=true")
        if stage_cfg.get("export_version", "").startswith("v3"):
            raise RuntimeError(
                "worker requires v5 shards (canonical inputs); got v3. "
                "Re-export with scripts/export_cached_shards_v5.py."
            )

        import openvino as ov  # type: ignore[import-untyped]

        from tahoma.worker.engines.openvino._hub import force_offline

        force_offline()

        from tahoma.worker.engines.openvino._plugin import build_plugin_config
        plugin_config = build_plugin_config(
            self._cache_dir, self._kv_cache_precision, self._dyn_quant_group,
        )

        yield LoadProgress(0, None, f"compiling stage {self._rank}")
        core = ov.Core()
        compiled = core.compile_model(
            str(stage_dir / "openvino_model.xml"), self._device, plugin_config,
        )
        self._req = compiled.create_infer_request()
        self._inputs = _v5_inputs(compiled)
        missing = {"hidden_states", "attention_mask", "position_ids", "beam_idx"} - set(self._inputs)
        if missing:
            raise RuntimeError(f"stage {self._rank} IR is missing v5 inputs: {sorted(missing)}")

        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._req is None or self._upstream is None:
            raise RuntimeError("call connect() and load() before build()")
        is_last = (self._rank == self._total - 1)
        return OVDistSpecWorkerEngine(
            rank=self._rank,
            is_last=is_last,
            stage_req=self._req,
            stage_inputs=self._inputs,
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
