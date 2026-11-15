"""Unit tests for tools/vla_libero_client.py (needs numpy, pillow and openpi-client)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("openpi_client")
pytest.importorskip("PIL")

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import vla_libero_client as c  # noqa: E402


def _obs():
    rng = np.random.default_rng(0)
    return {
        "agentview_image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }


def _meta(cameras):
    return {"cameras": cameras, "height": 8, "width": 8, "resize": "stretch"}


def test_flip_rotates_both_cameras_and_none_sends_them_as_rendered():
    obs = _obs()
    flipped = c.observation(obs, "t", _meta(["image", "image2"]), 0)
    raw = c.observation(obs, "t", _meta(["image", "image2"]), 0, flip=False)
    np.testing.assert_array_equal(flipped["images"]["image"], obs["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(flipped["images"]["image2"], obs["robot0_eye_in_hand_image"][::-1, ::-1])
    np.testing.assert_array_equal(raw["images"]["image"], obs["agentview_image"])
    np.testing.assert_array_equal(raw["images"]["image2"], obs["robot0_eye_in_hand_image"])


def test_xr0_camera_names_map_to_the_libero_cameras():
    obs = _obs()
    req = c.observation(obs, "t", _meta(["base", "wrist_left"]), 0)
    np.testing.assert_array_equal(req["images"]["base"], obs["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(req["images"]["wrist_left"], obs["robot0_eye_in_hand_image"][::-1, ::-1])
    c.check_metadata({"cameras": ["base", "wrist_left"], "state_len": 8, "chunk": [30, 7]})


class _Env:
    def __init__(self):
        self.steps = 0

    def reset(self):
        pass

    def set_init_state(self, _):
        return _obs()

    def step(self, _):
        self.steps += 1
        return _obs(), 0.0, False, {}


class _Client:
    def __init__(self, chunk):
        self.calls, self.chunk = 0, chunk

    def infer(self, _):
        self.calls += 1
        return {"actions": np.zeros((self.chunk, 7), np.float32), "server_timing": {"infer_ms": 1.0}}


@pytest.mark.parametrize("replan,calls", [(None, 1), (10, 3)])
def test_replan_executes_at_most_that_many_actions_per_chunk(replan, calls):
    client = _Client(chunk=30)
    _, _, infer_ms, _, _ = c.run_episode(
        client, _Env(), None, "t", _meta(["image"]), 30, 0, replan=replan
    )
    assert client.calls == calls == len(infer_ms)


def test_wrist_image_maps_to_the_eye_in_hand_camera():
    obs = _obs()
    req = c.observation(obs, "t", _meta(["image", "wrist_image"]), 0, flip=False)
    np.testing.assert_array_equal(req["images"]["wrist_image"], obs["robot0_eye_in_hand_image"])


def test_invert_gripper_binarizes_and_inverts_only_the_last_dim():
    a = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2]], np.float32)
    out = c.invert_gripper(a)
    np.testing.assert_array_equal(out[:, :6], a[:, :6])
    np.testing.assert_array_equal(out[:, 6], [-1.0, 1.0])
