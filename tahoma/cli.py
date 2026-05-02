"""Tahoma CLI.

Examples:
    # Last stage on rank 1 (listens on :9100):
    tahoma worker --rank 1 --total 2 --model /path/to/model --listen :9100

    # First stage on rank 0 (sends to next-stage host:9100, serves API on :8000):
    tahoma worker --rank 0 --total 2 --model /path/to/model \
        --next 192.168.1.50:9100 --api :8000

For non-API single-stage interactive use, omit `--api` and pipe prompts on stdin.
"""

from __future__ import annotations

import argparse
import logging
import sys

from tahoma.shared.shard import ShardPlan
from tahoma.shared.topology import PeerEndpoint, PeerLayout
from tahoma.shared.types import GenerationTask
from tahoma.worker.engines.base import Builder
from tahoma.worker.engines.openvino import OpenVINOBuilder
from tahoma.worker.runner import Runner

logger = logging.getLogger(__name__)


def parse_addr(s: str, default_host: str = "0.0.0.0") -> tuple[str, int]:
    """Parse `host:port` or `:port`."""
    if s.startswith(":"):
        return default_host, int(s[1:])
    host, port = s.rsplit(":", 1)
    return host, int(port)


def cmd_worker(args: argparse.Namespace) -> int:
    if args.rank < 0 or args.rank >= args.total:
        sys.exit(f"--rank must be in [0, {args.total}); got {args.rank}")

    if args.engine == "ov-optimum":
        if args.total != 1:
            sys.exit("--engine ov-optimum is single-stage only; use --total 1")
        from tahoma.shared.shard import ShardSpec

        spec = ShardSpec(
            model_id=args.model, layer_start=0, layer_end=0,
            total_layers=0, device=args.device,
            is_first_stage=True, is_last_stage=True,
        )
        logger.info(
            "engine=ov-optimum rank=0/1 device=%s model=%s",
            args.device, args.model,
        )
    elif args.engine == "ov-runtime":
        # ShardSpec is built inside the builder from pipeline_config.json.
        from tahoma.shared.shard import ShardSpec

        spec = ShardSpec(
            model_id=args.model, layer_start=0, layer_end=0,
            total_layers=0, device=args.device,
            is_first_stage=(args.rank == 0),
            is_last_stage=(args.rank == args.total - 1),
        )
        logger.info(
            "engine=ov-runtime rank=%d/%d device=%s pipeline=%s",
            args.rank, args.total, args.device, args.model,
        )
    else:
        plan = ShardPlan.from_hf_model_id(
            args.model, num_stages=args.total, devices=[args.device] * args.total,
        )
        spec = plan.stages[args.rank]
        logger.info(
            "engine=pytorch rank=%d/%d layers=[%d,%d) device=%s first=%s last=%s",
            args.rank, args.total, spec.layer_start, spec.layer_end, spec.device,
            spec.is_first_stage, spec.is_last_stage,
        )

    listen_host, listen_port = parse_addr(args.listen)

    builder: Builder
    if args.engine == "ov-optimum":
        from tahoma.worker.engines.openvino.optimum_engine import OptimumOVBuilder

        builder = OptimumOVBuilder(
            model_path=args.model,
            device=args.device,
            weight_format=args.ov_weight_format,
        )
    elif args.engine == "ov-runtime":
        from tahoma.worker.engines.openvino.ov_runtime import OVRuntimeBuilder

        ov_runtime_builder = OVRuntimeBuilder(
            pipeline_dir=args.model,
            rank=args.rank,
            total=args.total,
            device=args.device,
        )
        ov_runtime_builder.configure_listen(listen_host, listen_port)
        builder = ov_runtime_builder
    else:
        ov_builder = OpenVINOBuilder(model_path=args.model)
        ov_builder.configure_listen(listen_host, listen_port)
        builder = ov_builder

    upstream: PeerEndpoint | None = None
    if not spec.is_first_stage:
        # We listen for the upstream peer; PeerEndpoint here just signals
        # "we have an upstream" to the builder.
        upstream = PeerEndpoint(host=listen_host, port=listen_port)

    downstream: PeerEndpoint | None = None
    if not spec.is_last_stage:
        if not args.next:
            sys.exit("--next is required for non-last stages")
        host, port = parse_addr(args.next, default_host="127.0.0.1")
        downstream = PeerEndpoint(host=host, port=port)

    peers = PeerLayout(upstream=upstream, downstream=downstream)
    runner = Runner(builder)

    try:
        runner.start(peers, spec)

        if not spec.is_first_stage:
            logger.info("entering relay loop")
            runner.run_relay_loop()
            return 0

        # First stage: API or stdin.
        if args.api:
            from tahoma.api import make_app
            import uvicorn  # type: ignore[import-untyped]

            api_host, api_port = parse_addr(args.api)
            app = make_app(runner, model_id=args.model)
            logger.info("API serving on %s:%d", api_host, api_port)
            uvicorn.run(app, host=api_host, port=api_port, log_level="info")
            return 0

        # No API → stdin loop for quick CLI testing.
        logger.info("stdin mode: type a prompt and press enter")
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line:
                continue
            task = GenerationTask(
                task_id=f"stdin-{len(line)}",
                prompt=line,
                max_tokens=args.max_tokens,
            )
            for chunk in runner.generate(task):
                print(chunk.text, end="", flush=True)
                if chunk.is_final:
                    print()
        return 0
    finally:
        runner.close()


def main() -> int:
    parser = argparse.ArgumentParser(prog="tahoma")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("worker", help="run a pipeline-stage worker")
    pw.add_argument("--rank", type=int, required=True, help="0-based stage index")
    pw.add_argument("--total", type=int, required=True, help="total number of stages")
    pw.add_argument(
        "--model", required=True, help="HF model id or local model directory",
    )
    pw.add_argument(
        "--listen", default=":9100",
        help="bind address for the upstream-receiving socket (default :9100)",
    )
    pw.add_argument(
        "--next", help="downstream peer (host:port) — required for non-last stages",
    )
    pw.add_argument(
        "--api", help="API bind address (e.g. :8000) — only for rank 0",
    )
    pw.add_argument(
        "--device", default="CPU", help="device hint: CPU / GPU / NPU (default CPU)",
    )
    pw.add_argument(
        "--engine", default="pytorch",
        choices=["pytorch", "ov-optimum", "ov-runtime"],
        help=(
            "inference engine: "
            "'pytorch' (default, distributed) | "
            "'ov-optimum' (single-stage OV via optimum-intel; auto-export) | "
            "'ov-runtime' (multi-stage OV with stateful KV cache; expects a "
            "pre-exported pipeline directory at --model with stage_<i>/ subdirs)"
        ),
    )
    pw.add_argument(
        "--ov-weight-format", default="int4",
        choices=["int4", "int8", "fp16", "fp32"],
        help="weight format for OV IR auto-export (only used by --engine ov-optimum)",
    )
    pw.add_argument(
        "--max-tokens", type=int, default=64, help="max new tokens for stdin mode",
    )
    pw.set_defaults(func=cmd_worker)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
