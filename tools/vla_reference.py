#!/usr/bin/env python3
"""Fixture generator for the ov-policy engine (docs/design/2026-11-15-vla-policy-engine.md).

Python runs only when fixtures are regenerated, never in CI.

    uv run --with openpi-client python tools/vla_reference.py msgpack \
        --out crates/cascadia-api/tests/fixtures/policy

`msgpack` packs sample requests and replies with the real `openpi_client.msgpack_numpy`,
so the Rust codec is pinned to the bytes openpi clients send and accept. The values here
are repeated in crates/cascadia-api/tests/policy_codec.rs; change both together.
"""
import argparse
from pathlib import Path

import numpy as np
from openpi_client import msgpack_numpy

# The spec the codec tests use: cameras "image" and "wrist" at 4x4, state 3,
# chunk [2, 3], noise input [1, 2, 3].
H, W = 4, 4


def frames():
    a = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    return {"image": a, "wrist": (255 - a).astype(np.uint8)}


def cases():
    yield "request.msgpack", {
        "images": frames(),
        "state": np.array([0.5, -1.0, 2.25], dtype=np.float32),
        "prompt": "pick up the cube",
        "seed": 7,
        "noise": np.array([[[0.0, 1.0, -1.0], [0.5, -0.5, 2.0]]], dtype=np.float32),
    }
    # numpy's default float64 state, a numpy scalar seed, no noise.
    yield "request-f8-state.msgpack", {
        "images": frames(),
        "state": np.array([0.5, -1.0, 2.25]),
        "prompt": "pick up the cube",
        "seed": np.int64(7),
    }
    yield "reply.msgpack", {
        "actions": np.array([[0.5, -1.25, 3.0], [0.001, 2.0, -0.0]], dtype=np.float32),
        "server_timing": {"infer_ms": 12.5},
    }


def cmd_msgpack(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in cases():
        data = msgpack_numpy.packb(obj)
        (out / name).write_bytes(data)
        print(f"{name}: {len(data)} bytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("msgpack", help="golden msgpack bytes from openpi_client")
    mp.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "msgpack":
        cmd_msgpack(args.out)


if __name__ == "__main__":
    main()
