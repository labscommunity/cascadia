#!/usr/bin/env python3
"""LIBERO accuracy client for the ov-policy server (docs/design/2026-11-15-vla-policy-engine.md).

Runs LIBERO episodes in the simulator and asks a `cascadia run --engine ov-policy` server for
each action, through openpi's own WebSocket client, so a run is also the interoperability test.

    MUJOCO_GL=egl python tools/vla_libero_client.py --url ws://127.0.0.1:8000/v1/policy/ws \
        --suite libero_spatial --episodes-per-task 5 --seed 7 --out spatial.jsonl

Needs `openpi-client`, a LIBERO install (`hf-libero`, which brings robosuite and MuJoCo), torch
(LIBERO loads its initial states with it) and pillow. Headless machines render with
`MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`.

The episode loop, the observation keys and the state follow openpi's `examples/libero/main.py`
(Physical-Intelligence/openpi at 215abfb2): 10 settling steps with the dummy action, both
cameras rotated 180 degrees, state = eef position + axis-angle of the eef quaternion + the two
gripper joint positions, the suite's step limit, `env.seed(seed)` and init state `episode`.
LeRobot's LIBERO integration, which produced the training data of `HuggingFaceVLA/smolvla_libero`,
builds the same observation. Images are resized to the size and padding the server's metadata
states, not openpi's 224 centred pad (see "Cameras" in the design doc).
"""
import argparse
import json
import math
import pathlib
import statistics
import sys
import time

import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution the training data was rendered at
NUM_STEPS_WAIT = 10  # the simulator drops objects at the start; wait for them to settle
MAX_STEPS = {  # openpi's limits: the longest training demo of each suite, plus a margin
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
# Camera names the server advertises for LIBERO packages, and the simulator image behind each.
CAMERA_SOURCES = {
    "image": "agentview_image",
    "image2": "robot0_eye_in_hand_image",
    "base": "agentview_image",  # XR0
    "wrist_left": "robot0_eye_in_hand_image",
}


def quat2axisangle(quat):
    """robosuite's `quat2axisangle` (xyzw), as copied into openpi's LIBERO example."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_with_pad(img, height, width, convention):
    """Bilinear resize keeping the aspect ratio, padded with 0: centred, or on the top and left;
    or a plain stretch."""
    if convention == "stretch":
        return np.asarray(Image.fromarray(img).resize((width, height), resample=Image.BILINEAR))
    h, w = img.shape[:2]
    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)
    resized = np.asarray(Image.fromarray(img).resize((rw, rh), resample=Image.BILINEAR))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if convention == "center":
        top, left = (height - rh) // 2, (width - rw) // 2
    elif convention == "top_left":
        top, left = height - rh, width - rw
    else:
        raise ValueError(f"unknown resize convention {convention!r}")
    out[top : top + rh, left : left + rw] = resized
    return out


def observation(obs, task, meta, seed, flip=True):
    """`flip=False` sends the frames as the simulator renders them, for packages that rotate
    them on the server (an `image_flip` step)."""
    images = {}
    for cam in meta["cameras"]:
        img = obs[CAMERA_SOURCES[cam]]
        img = np.ascontiguousarray(img[::-1, ::-1] if flip else img)
        images[cam] = resize_with_pad(img, meta["height"], meta["width"], meta["resize"])
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    return {"images": images, "state": state, "prompt": str(task), "seed": int(seed)}


def check_metadata(meta):
    unknown = [c for c in meta["cameras"] if c not in CAMERA_SOURCES]
    if unknown:
        sys.exit(f"server cameras {meta['cameras']}: no LIBERO image for {unknown}")
    if meta["state_len"] != 8:
        sys.exit(f"server state_len {meta['state_len']}, LIBERO state is 8")
    if meta["chunk"][1] != 7:
        sys.exit(f"server action dim {meta['chunk'][1]}, LIBERO actions are 7")


def run_episode(client, env, init_state, task, meta, max_steps, seed_base, flip=True, replan=None):
    env.reset()
    obs = env.set_init_state(init_state)
    plan, infer_ms, rtt_ms, done, t, error = [], [], [], False, 0, None
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        if not plan:
            # A per-call seed makes both precisions draw the same noise on the same call.
            req = observation(obs, task, meta, seed_base + t, flip)
            t0 = time.perf_counter()
            try:
                reply = client.infer(req)
            except RuntimeError as e:  # an error frame from the server
                error = str(e)
                break
            rtt_ms.append((time.perf_counter() - t0) * 1000)
            infer_ms.append(float(reply["server_timing"]["infer_ms"]))
            plan.extend(np.asarray(reply["actions"])[:replan])
        obs, _, done, _ = env.step(plan.pop(0).tolist())
        if done:
            break
        t += 1
    return done, t, infer_ms, rtt_ms, error


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True, help="ws://<host>:<port>/v1/policy/ws")
    ap.add_argument("--suite", default="libero_spatial", choices=sorted(MAX_STEPS))
    ap.add_argument("--tasks", default="all", help="comma-separated task ids, or all")
    ap.add_argument("--episodes-per-task", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=None, help="default: the suite's limit")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="JSONL, one line per episode")
    ap.add_argument("--flip", choices=["both", "none"], default="both",
                    help="rotate both cameras 180 degrees (openpi, LeRobot), or send them as rendered "
                         "(packages whose manifest flips on the server)")
    ap.add_argument("--replan", type=int, default=None,
                    help="execute at most this many actions of each chunk (default: all)")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    tasks = range(suite.n_tasks) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
    max_steps = args.max_steps or MAX_STEPS[args.suite]

    # openpi uses `host` as the whole URI when it starts with "ws"; a port would follow the path.
    client = websocket_client_policy.WebsocketClientPolicy(host=args.url, port=None)
    meta = client.get_server_metadata()
    check_metadata(meta)
    print(f"server model={meta['model']} device={meta['device']} cameras={meta['cameras']} "
          f"{meta['height']}x{meta['width']} resize={meta['resize']} chunk={meta['chunk']} "
          f"noise={meta['noise']} flip={args.flip} replan={args.replan}", flush=True)

    out = args.out.open("w") if args.out else None
    n = k = errors = 0
    all_infer = []
    start = time.perf_counter()
    for task_id in tasks:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        if args.episodes_per_task > len(init_states):
            sys.exit(f"task {task_id} has {len(init_states)} init states")
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)  # openpi: the seed moves objects even with a fixed init state
        for ep in range(args.episodes_per_task):
            t0 = time.perf_counter()
            seed_base = args.seed * 1_000_000 + task_id * 10_000 + ep * 1_000
            done, steps, infer_ms, rtt_ms, error = run_episode(
                client, env, init_states[ep], task.language, meta, max_steps, seed_base,
                args.flip == "both", args.replan,
            )
            wall = time.perf_counter() - t0
            n += 1
            k += bool(done)
            errors += error is not None
            all_infer += infer_ms
            rec = {
                "suite": args.suite, "task_id": task_id, "task": task.language, "episode": ep,
                "seed": args.seed, "device": meta["device"], "success": bool(done), "steps": steps,
                "calls": len(infer_ms), "wall_s": round(wall, 2),
                "infer_ms_mean": round(statistics.fmean(infer_ms), 2) if infer_ms else None,
                "infer_ms_p50": round(statistics.median(infer_ms), 2) if infer_ms else None,
                "rtt_ms_mean": round(statistics.fmean(rtt_ms), 2) if rtt_ms else None,
                "error": error,
            }
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            print(f"episode task={task_id} ep={ep} success={done} steps={steps} "
                  f"calls={len(infer_ms)} infer_ms={rec['infer_ms_mean']} rtt_ms={rec['rtt_ms_mean']} "
                  f"wall_s={rec['wall_s']}" + (f" error={error!r}" if error else ""), flush=True)
        env.close()
    if out:
        out.close()
    mean = statistics.fmean(all_infer) if all_infer else float("nan")
    p50 = statistics.median(all_infer) if all_infer else float("nan")
    print(f"LIBERO-RESULT suite={args.suite} episodes={n} success={k} rate={k / max(n, 1):.3f} "
          f"mean_infer_ms={mean:.1f} p50_infer_ms={p50:.1f} errors={errors} "
          f"wall_s={time.perf_counter() - start:.0f}", flush=True)


if __name__ == "__main__":
    main()
