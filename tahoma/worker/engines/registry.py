"""Engine registry — central wiring of CLI ``--engine`` choices to builders.

Each entry binds a name (``ov-runtime``, ``ov-spec``, ...) to the validation,
shard-spec construction, and builder construction needed to start the engine.
``cli.cmd_worker`` looks up the entry once and calls the three callbacks; no
per-engine if/elif chain.

Add a new engine in three steps:
1. Implement an ``Engine`` + ``Builder`` (see ``engines/base.py``).
2. Register an ``EngineSpec`` here.
3. Add the name to the CLI ``--engine`` choices in ``cli.py``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass

from tahoma.shared.shard import ShardPlan, ShardSpec
from tahoma.worker.engines.base import Builder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSpec:
    """Wiring metadata for one engine implementation."""

    name: str
    description: str
    validate: Callable[[argparse.Namespace], None]
    build_shard_spec: Callable[[argparse.Namespace], ShardSpec]
    build_builder: Callable[[argparse.Namespace, str, int], Builder]


_REGISTRY: dict[str, EngineSpec] = {}


def register(engine: EngineSpec) -> None:
    if engine.name in _REGISTRY:
        raise ValueError(f"engine {engine.name!r} already registered")
    _REGISTRY[engine.name] = engine


def get(name: str) -> EngineSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown engine {name!r}; choices: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in engine registrations
# ---------------------------------------------------------------------------

def _stub_shard_spec(args: argparse.Namespace) -> ShardSpec:
    """ShardSpec for engines that build their own internal layer plan from
    a pipeline_config.json or that are single-stage."""
    return ShardSpec(
        model_id=args.model,
        layer_start=0,
        layer_end=0,
        total_layers=0,
        device=args.device,
        is_first_stage=(args.rank == 0),
        is_last_stage=(args.rank == args.total - 1),
    )


def _require_single_stage(name: str) -> Callable[[argparse.Namespace], None]:
    def check(args: argparse.Namespace) -> None:
        if args.total != 1:
            raise SystemExit(f"--engine {name} is single-stage only; use --total 1")
    return check


def _require_draft(name: str) -> Callable[[argparse.Namespace], None]:
    def check(args: argparse.Namespace) -> None:
        if not args.draft_model:
            raise SystemExit(f"--engine {name} requires --draft-model")
    return check


def _and(*checks: Callable[[argparse.Namespace], None]) -> Callable[[argparse.Namespace], None]:
    def check(args: argparse.Namespace) -> None:
        for c in checks:
            c(args)
    return check


# pytorch (default, distributed)
def _pytorch_spec(args: argparse.Namespace) -> ShardSpec:
    plan = ShardPlan.from_hf_model_id(
        args.model, num_stages=args.total, devices=[args.device] * args.total,
    )
    return plan.stages[args.rank]


def _pytorch_builder(args: argparse.Namespace, host: str, port: int) -> Builder:
    from tahoma.worker.engines.openvino import OpenVINOBuilder
    builder = OpenVINOBuilder(model_path=args.model)
    builder.configure_listen(host, port)
    return builder


register(EngineSpec(
    name="pytorch",
    description="distributed PyTorch (default)",
    validate=lambda _args: None,
    build_shard_spec=_pytorch_spec,
    build_builder=_pytorch_builder,
))


# pytorch-tp (PyTorch with tensor parallelism inside one or more PP stages)
def _pytorch_tp_validate(args: argparse.Namespace) -> None:
    if args.tp_size < 2:
        raise SystemExit(
            "--engine pytorch-tp requires --tp-size >= 2; for tp_size=1 use --engine pytorch",
        )
    if not args.tp_listen:
        raise SystemExit("--engine pytorch-tp requires --tp-listen host:port")
    if len(args.tp_peer) != args.tp_size - 1:
        raise SystemExit(
            f"--engine pytorch-tp requires --tp-peer for every other rank "
            f"({args.tp_size - 1} entries; got {len(args.tp_peer)})",
        )


def _pytorch_tp_spec(args: argparse.Namespace) -> ShardSpec:
    base = _pytorch_spec(args)
    return ShardSpec(
        model_id=base.model_id, layer_start=base.layer_start,
        layer_end=base.layer_end, total_layers=base.total_layers,
        device=base.device,
        is_first_stage=base.is_first_stage,
        is_last_stage=base.is_last_stage,
        tp_size=args.tp_size, tp_rank=args.tp_rank,
    )


def _pytorch_tp_builder(args: argparse.Namespace, host: str, port: int) -> Builder:
    from tahoma.parallel.group import TPPeer
    from tahoma.worker.engines.openvino.tp_engine import PyTorchTPBuilder

    tp_listen_host, tp_listen_port = _parse_addr(args.tp_listen)
    peers: list[TPPeer] = []
    for spec in args.tp_peer:
        rank_str, addr = spec.split("@", 1)
        peer_host, peer_port = _parse_addr(addr)
        peers.append(TPPeer(tp_rank=int(rank_str), host=peer_host, port=peer_port))

    builder = PyTorchTPBuilder(
        model_path=args.model,
        tp_rank=args.tp_rank, tp_size=args.tp_size,
        tp_listen=(tp_listen_host, tp_listen_port),
        tp_peers=peers,
    )
    builder.configure_listen(host, port)
    return builder


def _parse_addr(s: str, default_host: str = "0.0.0.0") -> tuple[str, int]:
    if s.startswith(":"):
        return default_host, int(s[1:])
    h, p = s.rsplit(":", 1)
    return h, int(p)


register(EngineSpec(
    name="pytorch-tp",
    description="PyTorch with tensor parallelism (column/row-split + ring all-reduce)",
    validate=_pytorch_tp_validate,
    build_shard_spec=_pytorch_tp_spec,
    build_builder=_pytorch_tp_builder,
))


# ov-optimum
def _ov_optimum_builder(args: argparse.Namespace, _host: str, _port: int) -> Builder:
    from tahoma.worker.engines.openvino.optimum_engine import OptimumOVBuilder
    return OptimumOVBuilder(
        model_path=args.model,
        device=args.device,
        weight_format=args.ov_weight_format,
        draft_model_path=args.draft_model,
        draft_weight_format=args.ov_weight_format,
    )


register(EngineSpec(
    name="ov-optimum",
    description="single-stage OV via optimum-intel; auto-export",
    validate=_require_single_stage("ov-optimum"),
    build_shard_spec=lambda args: ShardSpec(
        model_id=args.model, layer_start=0, layer_end=0, total_layers=0,
        device=args.device, is_first_stage=True, is_last_stage=True,
    ),
    build_builder=_ov_optimum_builder,
))


# ov-runtime
def _ov_runtime_builder(args: argparse.Namespace, host: str, port: int) -> Builder:
    from tahoma.worker.engines.openvino.ov_runtime import OVRuntimeBuilder
    builder = OVRuntimeBuilder(
        pipeline_dir=args.model, rank=args.rank, total=args.total, device=args.device,
        cache_dir=getattr(args, "ov_cache_dir", None),
        kv_cache_precision=getattr(args, "ov_kv_precision", None),
        dyn_quant_group=getattr(args, "ov_dyn_quant_group", None),
    )
    builder.configure_listen(host, port)
    return builder


register(EngineSpec(
    name="ov-runtime",
    description="multi-stage OV with stateful KV cache; pre-exported pipeline dir",
    validate=lambda _args: None,
    build_shard_spec=_stub_shard_spec,
    build_builder=_ov_runtime_builder,
))


# ov-spec
def _ov_spec_builder(args: argparse.Namespace, _host: str, _port: int) -> Builder:
    from tahoma.worker.engines.openvino.spec_decode_engine import OVSpecDecodeBuilder
    return OVSpecDecodeBuilder(
        model_path=args.model,
        draft_model_path=args.draft_model,
        device=args.device,
        weight_format=args.ov_weight_format,
        draft_weight_format=args.ov_weight_format,
        k=args.spec_k,
    )


register(EngineSpec(
    name="ov-spec",
    description="single-stage OV spec decode (legacy; prefer ov-genai)",
    validate=_and(_require_single_stage("ov-spec"), _require_draft("ov-spec")),
    build_shard_spec=lambda args: ShardSpec(
        model_id=args.model, layer_start=0, layer_end=0, total_layers=0,
        device=args.device, is_first_stage=True, is_last_stage=True,
    ),
    build_builder=_ov_spec_builder,
))


# ov-genai
def _ov_genai_validate(args: argparse.Namespace) -> None:
    _require_single_stage("ov-genai")(args)
    if args.draft_model and args.prompt_lookup > 0:
        raise SystemExit(
            "--engine ov-genai: --draft-model and --prompt-lookup are mutually "
            "exclusive (both set GenerationConfig.num_assistant_tokens). Pick one.",
        )


def _ov_genai_builder(args: argparse.Namespace, _host: str, _port: int) -> Builder:
    from tahoma.worker.engines.openvino.genai_engine import OVGenAIBuilder
    return OVGenAIBuilder(
        model_path=args.model,
        device=args.device,
        cache_dir=getattr(args, "ov_cache_dir", None),
        kv_cache_precision=getattr(args, "ov_kv_precision", None),
        dyn_quant_group=getattr(args, "ov_dyn_quant_group", None),
        draft_model_path=getattr(args, "draft_model", None),
        draft_device=getattr(args, "draft_device", None) or args.device,
        speculative_k=getattr(args, "spec_k", 5),
        prompt_lookup_ngram=getattr(args, "prompt_lookup", 0),
    )


register(EngineSpec(
    name="ov-genai",
    description="single-stage openvino_genai.LLMPipeline; FastDraft + Prompt Lookup",
    validate=_ov_genai_validate,
    build_shard_spec=lambda args: ShardSpec(
        model_id=args.model, layer_start=0, layer_end=0, total_layers=0,
        device=args.device, is_first_stage=True, is_last_stage=True,
    ),
    build_builder=_ov_genai_builder,
))


# ov-dist-spec
def _ov_dist_spec_validate(args: argparse.Namespace) -> None:
    if args.total < 2:
        raise SystemExit("--engine ov-dist-spec requires --total >= 2")
    if args.rank == 0 and not args.draft_model:
        raise SystemExit("--engine ov-dist-spec rank 0 requires --draft-model")


def _ov_dist_spec_builder(args: argparse.Namespace, host: str, port: int) -> Builder:
    if args.rank == 0:
        from tahoma.worker.engines.openvino.dist_spec import OVDistributedSpecBuilder
        return OVDistributedSpecBuilder(
            pipeline_dir=args.model,
            draft_model_path=args.draft_model,
            device=args.device,
            weight_format=args.ov_weight_format,
            k=args.spec_k,
            cache_dir=getattr(args, "ov_cache_dir", None),
            kv_cache_precision=getattr(args, "ov_kv_precision", None),
            dyn_quant_group=getattr(args, "ov_dyn_quant_group", None),
        )
    from tahoma.worker.engines.openvino.dist_spec import OVDistSpecWorkerBuilder
    builder = OVDistSpecWorkerBuilder(
        pipeline_dir=args.model, rank=args.rank, total=args.total, device=args.device,
        cache_dir=getattr(args, "ov_cache_dir", None),
        kv_cache_precision=getattr(args, "ov_kv_precision", None),
        dyn_quant_group=getattr(args, "ov_dyn_quant_group", None),
    )
    builder.configure_listen(host, port)
    return builder


register(EngineSpec(
    name="ov-dist-spec",
    description="multi-stage OV spec decode with mask-based rewind; v5 shards",
    validate=_ov_dist_spec_validate,
    build_shard_spec=_stub_shard_spec,
    build_builder=_ov_dist_spec_builder,
))
