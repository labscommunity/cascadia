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

import cv2
import numpy as np
from physicalai.inference.preprocessors.pi05 import Pi05Preprocessor
from physicalai.inference.preprocessors.smolvla import ResizeSmolVLA
from physicalai.inference.preprocessors.vla_jepa import _area_weights, _resize
from physicalai.inference.preprocessors.xr0 import _resize_image
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


def native(h, w, seed):
    """A smooth frame with some texture, as a camera would send at its own resolution."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(seed)
    base = np.stack([x / w * 255, y / h * 255, 128 + 100 * np.sin(x / 3) * np.cos(y / 5)], axis=-1)
    return np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)


def resizes():
    """Each convention's resize as physicalai's server runs it, on float32 [0, 1] frames, x255.

    top_left: ResizeSmolVLA._resize_with_pad; center: Pi05Preprocessor._resize_with_pad;
    stretch: cv2.resize INTER_LINEAR on float32 (ResizePreprocessor's stretch arithmetic).
    """
    for name, (h, w) in {"down": (48, 64), "tall": (40, 18), "up": (6, 10)}.items():
        img = native(h, w, len(name))
        f = img.astype(np.float32)[np.newaxis] / 255.0
        top_left = ResizeSmolVLA._resize_with_pad(np.transpose(f, (0, 3, 1, 2)), W, H, pad_value=0)
        center = Pi05Preprocessor._resize_with_pad(f, W, H, pad_value=0)
        stretch = cv2.resize(f[0], (W, H), interpolation=cv2.INTER_LINEAR)
        outs = {
            "top_left": np.transpose(top_left[0], (1, 2, 0)),
            "center": center[0],
            "stretch": stretch,
        }
        yield f"resize-{name}.rgb", img.tobytes()
        for conv, out in outs.items():
            assert out.shape == (H, W, 3), (conv, out.shape)
            yield f"resize-{name}-{conv}.f32", (out.astype(np.float32) * 255.0).astype("<f4").tobytes()


def lerobot_resize_with_pad(images, height, width):
    """LeRobot 0.6.0 `policies/pi0/modeling_pi0.py` `resize_with_pad_torch` (openpi's), float32
    channels-last path, as LeRobot's pi0 runs it on [0, 1] frames. Copied, since importing LeRobot
    pulls its whole stack."""
    import torch
    import torch.nn.functional as F

    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    _, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = F.interpolate(images, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    resized = resized.clamp(0.0, 1.0)
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    padded = F.pad(resized, (pad_w0, pad_w0 + remainder_w, pad_h0, pad_h0 + remainder_h), mode="constant", value=0.0)
    return padded.permute(0, 2, 3, 1).numpy()


def kernels():
    """Package-specific resizes, from the preprocessors that run them.

    pi0: LeRobot's resize_with_pad (above). VLA-JEPA: physicalai's first, area step
    (`vla_jepa._area_weights` / `_resize`). XR0: physicalai's `xr0._resize_image` (Qwen
    smart_resize, OpenCV INTER_CUBIC on uint8), with factor 8 and 300 max pixels so small
    frames reach 16x16.
    """
    for name, (h, w) in {"down": (48, 64), "tall": (40, 18), "up": (6, 10)}.items():
        img = native(h, w, len(name))
        f = img.astype(np.float32)[np.newaxis] / 255.0
        pi0 = lerobot_resize_with_pad(f, H, W)[0]
        yield f"resize-{name}-pi0.f32", (pi0.astype(np.float32) * 255.0).astype("<f4").tobytes()
        chw = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255
        area = _resize(chw, (_area_weights(h, H), _area_weights(w, W))).astype(np.float32)
        yield f"resize-{name}-area.f32", (np.transpose(area, (1, 2, 0)) * 255.0).astype("<f4").tobytes()
    for h, w in ((20, 20), (40, 40), (12, 12), (44, 52), (30, 20)):
        img = native(h, w, h + w)
        out = _resize_image(img, 8, 300)
        yield f"xr0-{h}x{w}.rgb", img.tobytes()
        yield f"xr0-{h}x{w}.out-{out.shape[0]}x{out.shape[1]}.rgb", np.ascontiguousarray(out).tobytes()
    # Rows wide enough for OpenCV's vector paths: 70x90 with 3000 max pixels.
    img = native(70, 90, 160)
    out = _resize_image(img, 8, 3000)
    yield "xr0-70x90.rgb", img.tobytes()
    yield f"xr0-70x90.out-{out.shape[0]}x{out.shape[1]}.rgb", np.ascontiguousarray(out).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    for name, data in cases():
        (out / name).write_bytes(data)
    for name, data in resizes():
        (out / name).write_bytes(data)
    for name, data in kernels():
        (out / name).write_bytes(data)
    # An unchanged client at a camera's native 480x640, JPEG.
    big = inputs()
    big["images.image"] = native(480, 640, 1)[np.newaxis]
    big["images.wrist"] = native(480, 640, 2)
    (out / "predict-native-480x640.msgpack").write_bytes(predict(big, "jpeg", seq=6))
    # What physicalai's server feeds SmolVLA from it: OpenCV's decode, then the top-left letterbox.
    _, decoded = p.decode_predict_request((out / "predict-native-480x640.msgpack").read_bytes())
    golden = b""
    for key in ("images.image", "images.wrist"):
        f = decoded[key].astype(np.float32).reshape(1, 480, 640, 3) / 255.0
        r = ResizeSmolVLA._resize_with_pad(np.transpose(f, (0, 3, 1, 2)), W, H, pad_value=0)
        golden += (np.transpose(r[0], (1, 2, 0)).astype(np.float32) * 255.0).astype("<f4").tobytes()
    (out / "predict-native-480x640.top_left.f32").write_bytes(golden)
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
