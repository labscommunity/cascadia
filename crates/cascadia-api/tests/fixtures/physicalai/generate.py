#!/usr/bin/env python3
"""Fixture generator for the physicalai remote-inference codec (tests/physicalai_codec.rs).

Python runs only when fixtures are regenerated, never in CI. Every byte comes from
physicalai's own protocol functions (`physicalai.inference.remote._protocol` and
`physicalai.transport`), so the Rust codec is pinned to what its client sends and accepts.

    python -m venv .venv && .venv/bin/pip install --no-deps <physicalai checkout> \
        eclipse-zenoh==1.9.0 msgpack==1.2.3 numpy opencv-python-headless loguru pydantic pyyaml \
        'jsonargparse[signatures]>=4.50,<5'
    .venv/bin/python -I generate.py --out .

The values here are repeated in tests/physicalai_codec.rs; change both together.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from physicalai.inference.remote import _protocol as p
from physicalai.transport import derive_port

# The test spec: cameras "image" and "wrist" at 16x16, state 3, chunk [2, 3], noise [1, 2, 3].
H, W = 16, 16
SERVER_ID = "0123456789abcdef0123456789abcdef"
MANIFEST_SHA256 = "ab" * 32
# max_payload_bytes for the test spec: 2*16*16*3 + 3*8 + 2*3*4 + 64 + 65536.
MAX_REQUEST_BYTES = 67184


def frame(invert=False):
    y, x = np.mgrid[0:H, 0:W]
    a = np.stack([x * 16, y * 16, (x + y) * 8], axis=-1).astype(np.uint8)
    return (255 - a).astype(np.uint8) if invert else a


def inputs():
    return {
        "state": np.array([[0.5, -1.0, 2.25]], dtype=np.float32),
        "task": ["pick up the cube"],
        "images.image": frame()[np.newaxis],
        "images.wrist": frame(True)[np.newaxis],
    }


def predict(values, codec, seq=3, budget_ms=2000):
    return p.encode_predict_request(
        values, seq=seq, budget_ms=budget_ms, image_codec=codec, jpeg_quality=90, max_image_side=None
    )


def cases():
    yield "metadata-request.msgpack", p.encode_message({"protocol_version": 1})
    yield "metadata.msgpack", p.encode_message({
        "protocol_version": 1,
        "server_id": SERVER_ID,
        "name": "cell1",
        "policy": {"name": "smolvla", "manifest_sha256": MANIFEST_SHA256, "backend": "openvino", "device": "CPU"},
        "chunk_size": 2,
        "rtc": {"supported": False, "chunk_size": None},
        "image_codecs": ["jpeg", "raw"],
        "max_request_bytes": MAX_REQUEST_BYTES,
    })
    yield "predict-raw.msgpack", predict(inputs(), "raw")
    yield "predict-jpeg.msgpack", predict(inputs(), "jpeg")
    # Unbatched frames, float64 state, and noise for an input-mode package.
    values = inputs()
    values["state"] = np.array([0.5, -1.0, 2.25])
    values["images.image"] = frame()
    values["noise"] = np.array([[[0.0, 1.0, -1.0], [0.5, -0.5, 2.0]]], dtype=np.float32)
    yield "predict-unbatched-noise.msgpack", predict(values, "raw", seq=4)
    # physicalai's PolicySource sends a lone camera under the bare `images` key.
    yield "predict-single-camera.msgpack", predict(
        {"state": inputs()["state"], "task": ["pick up the cube"], "images": frame()[np.newaxis]}, "jpeg", seq=5
    )
    action = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32)
    yield "reply.msgpack", p.encode_predict_reply(
        {"action": action},
        seq=3,
        server_id=SERVER_ID,
        server_queue_s=0.001,
        server_compute_s=0.25,
        server_decode_s=0.0005,
    )
    yield "error-invalid-request.msgpack", p.encode_message({"protocol_version": 1, "code": "invalid_request", "seq": 3})
    yield "error-unsupported-version.msgpack", p.encode_message(
        {"protocol_version": 1, "code": "unsupported_version", "seq": -1}
    )
    yield "reset-ok.msgpack", p.encode_message({"protocol_version": 1, "ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    for name, data in cases():
        (out / name).write_bytes(data)
    # What physicalai's own decoder makes of the JPEG frames (OpenCV), to bound our decoder's
    # difference from it.
    _, decoded = p.decode_predict_request((out / "predict-jpeg.msgpack").read_bytes())
    (out / "predict-jpeg.decoded.rgb").write_bytes(
        decoded["images.image"].tobytes() + decoded["images.wrist"].tobytes()
    )
    (out / "frames.rgb").write_bytes(frame().tobytes() + frame(True).tobytes())
    (out / "ports.json").write_text(
        json.dumps({n: derive_port("inference", n) for n in ("cell1", "smolvla", "a.b-c_d")}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
