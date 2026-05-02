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
import os
import signal
import sys
from pathlib import Path
from types import FrameType

from tahoma.shared.topology import PeerEndpoint, PeerLayout
from tahoma.shared.types import GenerationTask
from tahoma.worker.engines import registry
from tahoma.worker.runner import Runner

logger = logging.getLogger(__name__)


def parse_addr(s: str, default_host: str = "0.0.0.0") -> tuple[str, int]:
    """Parse `host:port` or `:port`."""
    if s.startswith(":"):
        return default_host, int(s[1:])
    host, port = s.rsplit(":", 1)
    return host, int(port)


def _write_pid_file(path: Path) -> None:
    """Write our PID and ensure removal on normal exit. Used by supervisors
    (systemd Type=simple, NSSM, supervisord) that probe liveness via PID file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n")
    import atexit
    atexit.register(lambda: path.unlink(missing_ok=True))


def cmd_engines(_args: argparse.Namespace) -> int:
    for name in registry.names():
        spec = registry.get(name)
        print(f"  {name:<14}  {spec.description}")
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    if args.rank < 0 or args.rank >= args.total:
        sys.exit(f"--rank must be in [0, {args.total}); got {args.rank}")

    engine = registry.get(args.engine)
    engine.validate(args)
    spec = engine.build_shard_spec(args)
    logger.info(
        "engine=%s rank=%d/%d device=%s model=%s",
        engine.name, args.rank, args.total, args.device, args.model,
    )

    listen_host, listen_port = parse_addr(args.listen)
    builder = engine.build_builder(args, listen_host, listen_port)

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

    if args.pid_file:
        _write_pid_file(Path(args.pid_file))

    def _shutdown(signum: int, _frame: FrameType | None) -> None:
        logger.info("received signal %d, shutting down", signum)
        runner.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

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
        choices=registry.names(),
        help=(
            "inference engine. Run `tahoma engines` to see descriptions. "
            "Add new engines via `tahoma.worker.engines.registry.register(...)`."
        ),
    )
    pw.add_argument(
        "--ov-weight-format", default="int4",
        choices=["int4", "int8", "fp16", "fp32"],
        help="weight format for OV IR auto-export (only used by --engine ov-optimum)",
    )
    pw.add_argument(
        "--draft-model",
        help=(
            "speculative-decoding draft model — small model id or path that "
            "shares the target's tokenizer (e.g. unsloth/Llama-3.2-1B-Instruct "
            "for a Llama-3.1 target). Used by --engine ov-optimum (best effort, "
            "may fall back) and required by --engine ov-spec."
        ),
    )
    pw.add_argument(
        "--spec-k", type=int, default=4,
        help=(
            "speculative-decoding draft length per round (default 4). "
            "Tuned on Llama-3.1-8B target + Llama-3.2-1B draft on Arc B390: "
            "K=3 → 24.6 tok/s, K=4 → 35 tok/s (91% accept), K=5 → 28 tok/s, "
            "K=7 → 26 tok/s. Sweet spot depends on draft acceptance rate."
        ),
    )
    pw.add_argument(
        "--max-tokens", type=int, default=64, help="max new tokens for stdin mode",
    )
    pw.add_argument(
        "--pid-file",
        help="write the worker's PID here for systemd / NSSM / supervisord integration",
    )
    pw.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging level (default INFO)",
    )
    pw.set_defaults(func=cmd_worker)

    pe = sub.add_parser("engines", help="list registered inference engines")
    pe.set_defaults(func=cmd_engines)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(args, "log_level", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
