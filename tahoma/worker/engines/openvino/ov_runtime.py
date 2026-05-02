"""Multi-stage OpenVINO Runtime engine.

Loads pre-exported per-stage stateful OV IR shards (rainier v3+ format) and
runs them across the existing TCP transport. Each stage owns one
`InferRequest`; the KV cache is internal state on the IR (`ReadValue`/`Assign`
ops) and accumulates automatically across decode calls. `reset_state()` runs
between independent generation tasks.

Pipeline directory layout (matches rainier's exporter):

    <pipeline-dir>/
        pipeline_config.json    # model_id, num_stages, rope_theta, ...
        tokenizer/              # HF tokenizer dump
        stage_0/openvino_model.{xml,bin}, stage_config.json
        stage_1/...
        stage_N/...

Wire format between stages: hidden_states only. Each stage tracks its own
absolute-position counter so it can compute cos/sin locally without sending
position metadata across the wire. Counter resets when an activation with
seq_len > 1 arrives (signals a new prefill on relay/last stages) or when the
first stage starts a new task.

Limitations of this initial port:
- IR auto-export is **not** wired here (use rainier's `export_cached_shards.py`
  to produce the shards, or arrange for them to be staged on each node).
- Greedy sampling only.
- One task at a time per pipeline.
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
from tahoma.worker.transport import ActivationClient, ActivationServer

logger = logging.getLogger(__name__)


def _load_pipeline_config(pipeline_dir: Path) -> dict[str, Any]:
    return json.loads((pipeline_dir / "pipeline_config.json").read_text())


def _load_stage_config(stage_dir: Path) -> dict[str, Any]:
    return json.loads((stage_dir / "stage_config.json").read_text())


def _build_rotary(model_id: str) -> tuple[Any, Any]:
    """Return (rotary_module, hf_config) for cos/sin computation.

    The rotary uses transformers' canonical Llama implementation, which
    handles `rope_scaling = {type: "llama3", ...}` correctly.
    """
    import torch
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    rotary = LlamaRotaryEmbedding(config=config)
    rotary.eval()
    for p in rotary.parameters():
        p.requires_grad_(False)
    return rotary, config


def _compute_cos_sin(rotary: Any, hidden_size: int, position_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the rotary module to produce (cos, sin) numpy float32 arrays."""
    import torch

    seq_len = position_ids.shape[1]
    dummy = torch.zeros(1, seq_len, hidden_size, dtype=torch.float32)
    pos_t = torch.tensor(position_ids, dtype=torch.long)
    with torch.no_grad():
        cos, sin = rotary(dummy, pos_t)
    return cos.float().cpu().numpy(), sin.float().cpu().numpy()


@dataclass
class _Shard:
    """One pipeline stage's compiled OV IR + per-task state."""

    spec_meta: dict[str, Any]
    request: Any                # ov.InferRequest
    input_names: list[str]
    output_names: list[str]
    position: int = 0           # absolute position of next token to be ingested

    def reset(self) -> None:
        self.request.reset_state()
        self.position = 0


@dataclass
class _ActiveTask:
    task: GenerationTask
    prompt_ids: np.ndarray
    generated: list[int] = field(default_factory=list)
    prefilled: bool = False
    last_token: int | None = None


class OVRuntimeEngine(Engine):
    """Per-stage OV Runtime engine with stateful KV cache shards."""

    def __init__(
        self,
        spec: ShardSpec,
        shard: _Shard,
        rotary: Any,
        hidden_size: int,
        tokenizer: Any | None,
        upstream_server: ActivationServer | None,
        downstream_client: ActivationClient | None,
    ):
        self.spec = spec
        self._shard = shard
        self._rotary = rotary
        self._hidden_size = hidden_size
        self._tokenizer = tokenizer
        self._upstream = upstream_server
        self._downstream = downstream_client
        self._active: dict[TaskId, _ActiveTask] = {}
        self._pending: list[GenerationTask] = []

    def warmup(self) -> None:
        if self.spec.is_first_stage and self._tokenizer is not None:
            try:
                ids = self._tokenizer("Hi", return_tensors="np").input_ids
                self._infer_first(ids, position=0)
                self._shard.reset()
                logger.info("warmup ok")
            except Exception as err:  # noqa: BLE001
                logger.warning("warmup failed: %s", err)
        else:
            logger.info("warmup skipped on non-first stage")

    def submit(self, task: GenerationTask) -> None:
        if not self.spec.is_first_stage:
            raise RuntimeError("submit() is only valid on the first stage")
        if task.task_id in self._active or any(t.task_id == task.task_id for t in self._pending):
            return
        self._pending.append(task)

    # ----------------------- inference primitives -----------------------

    def _infer_first(self, input_ids: np.ndarray, position: int) -> np.ndarray:
        seq_len = input_ids.shape[1]
        positions = np.arange(position, position + seq_len, dtype=np.int64).reshape(1, seq_len)
        cos, sin = _compute_cos_sin(self._rotary, self._hidden_size, positions)

        feed = self._build_feed(
            x=input_ids.astype(np.int64),
            cos=cos,
            sin=sin,
        )
        out = self._shard.request.infer(feed)
        return out[self._shard.output_names[0]]

    def _infer_relay(self, hidden_states: np.ndarray, position: int) -> np.ndarray:
        seq_len = hidden_states.shape[1]
        positions = np.arange(position, position + seq_len, dtype=np.int64).reshape(1, seq_len)
        cos, sin = _compute_cos_sin(self._rotary, self._hidden_size, positions)

        feed = self._build_feed(
            x=hidden_states.astype(np.float32),
            cos=cos,
            sin=sin,
        )
        out = self._shard.request.infer(feed)
        return out[self._shard.output_names[0]]

    def _build_feed(self, *, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> dict[str, np.ndarray]:
        # Inputs in the v3 format are (input_ids|hidden_states, cos, sin).
        # Use ordinal order — names vary across exports.
        names = self._shard.input_names
        if len(names) < 3:
            raise RuntimeError(
                f"shard expected >=3 inputs, got {len(names)}: {names}"
            )
        return {names[0]: x, names[1]: cos, names[2]: sin}

    # ----------------------- step dispatch -----------------------

    def step(self) -> Iterable[tuple[TaskId, Chunk]]:
        if self.spec.is_first_stage:
            yield from self._step_first()
            return
        if self.spec.is_last_stage:
            self._step_last()
            return
        self._step_middle()

    def _step_first(self) -> Iterable[tuple[TaskId, Chunk]]:
        if not self._active and self._pending:
            task = self._pending.pop(0)
            assert self._tokenizer is not None
            prompt_ids = self._tokenizer(task.prompt, return_tensors="np").input_ids
            self._shard.reset()
            self._active[task.task_id] = _ActiveTask(task=task, prompt_ids=prompt_ids)
            logger.info("task %s active: prompt_tokens=%d", task.task_id[:8], prompt_ids.shape[1])

        if not self._active:
            return

        task_id = next(iter(self._active))
        active = self._active[task_id]

        if not active.prefilled:
            input_ids = active.prompt_ids.astype(np.int64)
            active.prefilled = True
        else:
            input_ids = np.array([[active.last_token]], dtype=np.int64)

        hs = self._infer_first(input_ids, position=self._shard.position)
        self._shard.position += input_ids.shape[1]

        if self.spec.is_last_stage:
            # 1-stage: the IR also produced logits.
            next_token = int(np.argmax(hs[0, -1, :]))
        else:
            assert self._downstream is not None
            self._downstream.send(hs.astype(np.float16))
            token_array, _ = self._downstream.recv()
            next_token = int(token_array.flat[0])

        active.last_token = next_token
        active.generated.append(next_token)

        eos_id = getattr(self._tokenizer, "eos_token_id", None) if self._tokenizer else None
        text = self._tokenizer.decode([next_token], skip_special_tokens=True) if self._tokenizer else ""
        is_final = (
            len(active.generated) >= active.task.max_tokens
            or (eos_id is not None and next_token == eos_id)
        )

        yield task_id, Chunk(
            task_id=task_id,
            token_id=next_token,
            text=text,
            is_final=is_final,
        )

        if is_final:
            del self._active[task_id]

    def _step_last(self) -> None:
        assert self._upstream is not None
        activation, _ = self._upstream.recv()
        if activation.shape[1] > 1:
            self._shard.reset()  # new prefill arrived → discard any cached state
        out = self._infer_relay(activation, position=self._shard.position)
        self._shard.position += activation.shape[1]

        # Last stage IR already includes norm + lm_head, so out is logits.
        next_token = int(np.argmax(out[0, -1, :]))
        token_array = np.array([next_token], dtype=np.int32)
        self._upstream.send(token_array)

    def _step_middle(self) -> None:
        assert self._upstream is not None and self._downstream is not None
        activation, _ = self._upstream.recv()
        if activation.shape[1] > 1:
            self._shard.reset()
        out = self._infer_relay(activation, position=self._shard.position)
        self._shard.position += activation.shape[1]
        self._downstream.send(out.astype(np.float16))
        token_array, _ = self._downstream.recv()
        self._upstream.send(token_array)

    def close(self) -> None:
        if self._upstream is not None:
            self._upstream.close()
        if self._downstream is not None:
            self._downstream.close()


class OVRuntimeBuilder(Builder):
    """Builder for `OVRuntimeEngine`. Loads a per-stage stateful OV IR shard."""

    def __init__(self, pipeline_dir: str, rank: int, total: int, device: str = "GPU"):
        self._pipeline_dir = Path(pipeline_dir)
        self._rank = rank
        self._total = total
        self._device = device
        self._shard: _Shard | None = None
        self._spec: ShardSpec | None = None
        self._rotary: Any = None
        self._hidden_size: int | None = None
        self._tokenizer: Any | None = None
        self._upstream: ActivationServer | None = None
        self._downstream: ActivationClient | None = None
        self._listen_host: str = "0.0.0.0"
        self._listen_port: int | None = None

    def configure_listen(self, host: str, port: int) -> None:
        self._listen_host = host
        self._listen_port = port

    def connect(self, peers: PeerLayout) -> None:
        # Same connect-order as the PyTorch path: bind → connect-out → accept-in.
        if peers.upstream is not None:
            if self._listen_port is None:
                raise RuntimeError("configure_listen() required when upstream is set")
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
        import openvino as ov  # type: ignore[import-untyped]
        from transformers import AutoTokenizer

        yield LoadProgress(0, None, f"reading {self._pipeline_dir}")
        pipeline_cfg = _load_pipeline_config(self._pipeline_dir)
        if pipeline_cfg["num_stages"] != self._total:
            raise RuntimeError(
                f"--total ({self._total}) does not match pipeline_config "
                f"num_stages ({pipeline_cfg['num_stages']})"
            )

        stage_dir = self._pipeline_dir / f"stage_{self._rank}"
        stage_cfg = _load_stage_config(stage_dir)

        layer_start = stage_cfg["layer_start"]
        layer_end = stage_cfg["layer_end"]
        is_first = bool(stage_cfg.get("has_embed", False))
        is_last = bool(stage_cfg.get("has_head", False))
        self._spec = ShardSpec(
            model_id=pipeline_cfg["model_id"],
            layer_start=layer_start,
            layer_end=layer_end,
            total_layers=pipeline_cfg["num_layers"],
            device=self._device,
            is_first_stage=is_first,
            is_last_stage=is_last,
        )

        self._hidden_size = pipeline_cfg["hidden_size"]
        model_id = pipeline_cfg["model_id"]

        yield LoadProgress(0, None, "compiling OV shard")
        core = ov.Core()
        model = core.read_model(str(stage_dir / "openvino_model.xml"))
        compiled = core.compile_model(model, self._device)
        request = compiled.create_infer_request()

        input_names = [list(inp.names)[0] for inp in compiled.inputs]
        output_names = [list(out.names)[0] for out in compiled.outputs]
        self._shard = _Shard(
            spec_meta=stage_cfg,
            request=request,
            input_names=input_names,
            output_names=output_names,
        )

        yield LoadProgress(0, None, "loading rotary + tokenizer")
        self._rotary, _ = _build_rotary(model_id)
        if is_first:
            tokenizer_dir = self._pipeline_dir / "tokenizer"
            if tokenizer_dir.is_dir():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        yield LoadProgress(1, 1, "ready")

    def build(self) -> Engine:
        if self._shard is None or self._spec is None:
            raise RuntimeError("call load() before build()")
        return OVRuntimeEngine(
            spec=self._spec,
            shard=self._shard,
            rotary=self._rotary,
            hidden_size=self._hidden_size or 0,
            tokenizer=self._tokenizer,
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
        self._shard = None
