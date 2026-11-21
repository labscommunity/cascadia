#!/usr/bin/env python3
"""Serving check for a live `cascadia run <model> --engine ov-policy` server.

Sends one reference observation through openpi's own WebSocket client and compares the served
actions with the reference's, so a pass shows the package loads, serves the openpi wire format,
and computes what the reference model computes.

    python tools/vla_serve_check.py --url ws://127.0.0.1:19430/v1/policy/ws \
        --ref pkg.ref.npz --kind pi05_droid

Each kind reads its reference the way its parity test in
crates/cascadia-engine-openvino/tests/policy_openvino.rs does:

- `pi05_droid`: the package's `ref.npz` (LeRobot PI05Policy, f32): frame `--frame` of `base` and
  `wrist`, the 8-wide `state`, `task`, and that frame's `noise` sent as the request's noise;
  the first 8 of each 32-wide action row are compared, as the test does.
- `molmoact2`: `molmoact2-e2e.npz` (PyTorch f32 pipeline): `obs_images.<camera>` float CHW
  frames turned into uint8 HWC, `obs_state`, `task`; the noise is the package's constant.
- `xr1`: `xr1.ref.npz` (Xiaomi's model, f32): the four `crop_<camera>` frames per camera as one
  `[4, H, W, 3]` array (history 4, oldest first), the `[4, 14]` `state14`, `task`, `noise1`;
  expected is `ref1` cut to its first 12 dims.

The default bound is the engine test's own for what the server runs, read from its metadata:
the `precision` it compiled at (`f32` on CPU or GPU at f32, `f16` on GPU by default) and, for
MolmoAct2, the checkpoint its cameras identify (LIBERO, DROID or SO-100/101). An unknown
precision or checkpoint fails the check; `--bound` overrides the default.

Prints `SERVE_CHECK OK <kind> max_abs=... shape=...`, or `SERVE_CHECK FAILED: <why>` and exits 1.
`--dry-run` builds the request and prints its keys, shapes and dtypes without connecting.

Needs numpy, and `openpi-client` for a live check. On Python 3.14 openpi-client's `dm-tree`
dependency does not build; install it without its dependencies and add the ones it uses
(`typing_extensions` too, which it does not declare; numpy 2.x works):

    pip install --no-deps openpi-client
    pip install numpy msgpack "websockets>=11" typing_extensions
"""
import argparse
import sys

import numpy as np

KINDS = ("molmoact2", "pi05_droid", "xr1")

PI05_DROID_F16_BOUND = 2e-2

# The engine tests' bounds (crates/cascadia-engine-openvino/tests/policy_openvino.rs), by kind
# (and MolmoAct2 checkpoint) and compiled precision.
BOUNDS = {
    "pi05_droid": {"f32": 5e-3, "f16": PI05_DROID_F16_BOUND},
    "molmoact2 LIBERO": {"f32": 1e-3, "f16": 1e-2},
    "molmoact2 DROID": {"f32": 1e-3, "f16": 3e-3},
    "molmoact2 SO-100/101": {"f32": 1e-2, "f16": 5e-2},  # degrees
    "xr1": {"f32": 1e-3, "f16": 1e-2},
}

# A MolmoAct2 checkpoint by the cameras it serves.
MOLMOACT2_CHECKPOINTS = {
    ("image", "wrist_image"): "LIBERO",
    ("exterior_1_left", "exterior_2_left", "wrist_left"): "DROID",
    ("side", "top"): "SO-100/101",
}

XR1_CAMERAS = {
    "agentview_left": "robot0_agentview_left",
    "agentview_right": "robot0_agentview_right",
    "eye_in_hand": "robot0_eye_in_hand",
}


class CheckError(Exception):
    pass


def molmo_frame(a):
    """A float `[1, 3, H, W]` frame as uint8 HWC: Intel's `_resize`, `clip(x * 255, 0, 255)`
    truncated (scaled only when the frame is in [0, 1]), as the Rust test's `molmo_frame`."""
    a = np.asarray(a, dtype=np.float32)
    scale = np.float32(255.0) if a.max() <= 1.0 else np.float32(1.0)
    hwc = np.clip(a[0] * scale, 0.0, 255.0).astype(np.uint8).transpose(1, 2, 0)
    return np.ascontiguousarray(hwc)


def build(kind, z, frame=0):
    """The request for one reference observation, the expected actions, and the slice of the
    served `[T, D]` actions they are compared with (`None`: all of it)."""
    task = str(z["task"].item())
    if kind == "pi05_droid":
        n = z["base"].shape[0]
        if not 0 <= frame < n:
            raise CheckError(f"--frame {frame}: the reference has {n} frames")
        req = {
            "images": {
                "base_0_rgb": np.ascontiguousarray(z["base"][frame]),
                "left_wrist_0_rgb": np.ascontiguousarray(z["wrist"][frame]),
            },
            "state": np.asarray(z["state"][frame], dtype=np.float32),
            "prompt": task,
            "noise": np.asarray(z["noise"][frame][None], dtype=np.float32),
        }
        expected = np.asarray(z["actions"][frame], dtype=np.float32)
        return req, expected[:, :8], 8
    if kind == "molmoact2":
        prefix = "obs_images."
        cams = [k[len(prefix):] for k in z.files if k.startswith(prefix)]
        if not cams:
            raise CheckError(f"no {prefix}<camera> frames in the reference")
        req = {
            "images": {c: molmo_frame(z[prefix + c]) for c in cams},
            "state": np.asarray(z["obs_state"], dtype=np.float32).reshape(-1),
            "prompt": task,
        }
        expected = np.asarray(z["actions"], dtype=np.float32)
        return req, expected.reshape(expected.shape[-2:]), None
    if kind == "xr1":
        req = {
            "images": {c: np.ascontiguousarray(z[f"crop_{k}"]) for c, k in XR1_CAMERAS.items()},
            "state": np.asarray(z["state14"], dtype=np.float32),
            "prompt": task,
            "noise": np.asarray(z["noise1"], dtype=np.float32),
        }
        ref = np.asarray(z["ref1"], dtype=np.float32)
        return req, ref.reshape(ref.shape[-2:])[:, :12], None
    raise CheckError(f"unknown kind {kind!r}")


def describe(req):
    lines = []
    for key, v in req.items():
        if key == "images":
            for cam, img in v.items():
                lines.append(f"images.{cam}: {img.dtype} {list(img.shape)}")
        elif isinstance(v, np.ndarray):
            lines.append(f"{key}: {v.dtype} {list(v.shape)}")
        else:
            lines.append(f"{key}: {type(v).__name__} {v!r}")
    return lines


def check_metadata(meta, req):
    """The server's spec must take this request, so a mismatch reads as one, not as a diff."""
    cams = sorted(req["images"])
    if sorted(meta["cameras"]) != cams:
        raise CheckError(f"server cameras {meta['cameras']}; the reference has {cams}")
    history = meta.get("history", 1)
    frame = next(iter(req["images"].values())).shape
    hw = frame[-3:-1]
    if (meta["height"], meta["width"]) != tuple(hw):
        raise CheckError(f"server frames {meta['height']}x{meta['width']}; the reference's {hw}")
    if (history > 1) != (len(frame) == 4) or (history > 1 and frame[0] != history):
        raise CheckError(f"server history {history}; the reference frames are {list(frame)}")
    if meta["state_len"] != req["state"].shape[-1]:
        raise CheckError(f"server state_len {meta['state_len']}; the reference's {req['state'].shape}")
    if "noise" in req and meta["noise"] != "input":
        raise CheckError(f"server noise mode {meta['noise']}; this check sends the reference's noise")


def choose_bound(kind, meta):
    """The engine test's bound for what the server runs, and why it was chosen."""
    precision = meta.get("precision") or ""
    key = kind
    if kind == "molmoact2":
        ckpt = MOLMOACT2_CHECKPOINTS.get(tuple(sorted(meta.get("cameras", []))))
        if ckpt is None:
            raise CheckError(f"unknown molmoact2 checkpoint (cameras {meta.get('cameras')}); "
                             "pass --bound")
        key = f"molmoact2 {ckpt}"
    bound = BOUNDS[key].get(precision)
    if bound is None:
        raise CheckError(f"unknown precision {precision!r} in the server metadata; pass --bound")
    return bound, f"{key} at {precision} on {meta.get('device')}"


def compare(actions, expected, width):
    """Max abs diff of the served actions against the expected ones."""
    actions = np.asarray(actions, dtype=np.float32)
    if not np.all(np.isfinite(actions)):
        raise CheckError(f"non-finite actions, shape {list(actions.shape)}")
    got = actions[:, :width] if width is not None else actions
    if got.shape != expected.shape:
        raise CheckError(f"actions shape {list(actions.shape)}; expected {list(expected.shape)}")
    return float(np.max(np.abs(got - expected))), list(actions.shape)


def run(args):
    with np.load(args.ref) as z:
        req, expected, width = build(args.kind, z, args.frame)
    if args.dry_run:
        print(f"dry run {args.kind} ({args.ref}); expected actions {list(expected.shape)}")
        for line in describe(req):
            print(f"  {line}")
        return
    from openpi_client import websocket_client_policy

    # openpi uses `host` as the whole URI when it starts with "ws"; a port would follow the path.
    client = websocket_client_policy.WebsocketClientPolicy(host=args.url, port=None)
    meta = client.get_server_metadata()
    print(f"server model={meta.get('model')} device={meta.get('device')} "
          f"precision={meta.get('precision')} cameras={meta['cameras']} chunk={meta['chunk']}",
          flush=True)
    check_metadata(meta, req)
    if args.bound is None:
        args.bound, why = choose_bound(args.kind, meta)
        print(f"bound {args.bound:.1e}: the engine test's for {why}", flush=True)
    else:
        print(f"bound {args.bound:.1e}: --bound", flush=True)
    try:
        reply = client.infer(req)
    except RuntimeError as e:  # an error frame from the server
        raise CheckError(f"server refused the request: {e}") from e
    max_abs, shape = compare(reply["actions"], expected, width)
    infer_ms = reply.get("server_timing", {}).get("infer_ms")
    if max_abs > args.bound:
        raise CheckError(f"{args.kind} max_abs={max_abs:.3e} > bound {args.bound:.1e} shape={shape}")
    print(f"SERVE_CHECK OK {args.kind} max_abs={max_abs:.3e} shape={shape} "
          f"bound={args.bound:.1e} infer_ms={infer_ms}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", help="the policy WebSocket, e.g. ws://127.0.0.1:19430/v1/policy/ws")
    ap.add_argument("--ref", required=True, help="the kind's reference .npz")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--bound", type=float,
                    help="max abs diff allowed (default: the engine test's for the server's "
                         "precision and checkpoint)")
    ap.add_argument("--frame", type=int, default=0, help="pi05_droid: the reference frame to send")
    ap.add_argument("--dry-run", action="store_true", help="print the request and do not connect")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.url:
        ap.error("--url is required unless --dry-run")
    try:
        run(args)
    except CheckError as e:
        print(f"SERVE_CHECK FAILED: {e}")
        return 1
    except Exception as e:  # connection refused, a closed socket, a malformed reference
        print(f"SERVE_CHECK FAILED: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
