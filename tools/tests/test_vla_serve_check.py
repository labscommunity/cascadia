"""Unit tests for tools/vla_serve_check.py's request building (needs numpy only).

The synthetic cases run anywhere. The real references are checked when their path is set:
`CASCADIA_REF_PI05_DROID` (the package's `ref.npz`), `CASCADIA_REF_MOLMOACT2` (a DROID
checkpoint's `molmoact2-e2e.npz`) and `CASCADIA_REF_XR1` (`xr1.ref.npz`); the shapes asserted are
the ones the Rust parity tests build.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

_TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

if np is not None:
    import vla_serve_check as sc


def _npz(**arrays):
    f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    f.close()
    np.savez(f.name, **arrays)
    return f.name


class _Npz(dict):
    """An `NpzFile`'s arrays, with its `files`."""

    @property
    def files(self):
        return list(self)


def _shapes(req):
    out = {f"images.{c}": (str(v.dtype), v.shape) for c, v in req["images"].items()}
    for k in ("state", "noise"):
        if k in req:
            out[k] = (str(req[k].dtype), req[k].shape)
    return out


@unittest.skipIf(np is None, "needs numpy")
class Synthetic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.paths = []

    def tearDown(self):
        for p in self.paths:
            os.unlink(p)

    def load(self, **arrays):
        p = _npz(**arrays)
        self.paths.append(p)
        # Read and closed, so tearDown can delete it on Windows too.
        with np.load(p) as z:
            return _Npz({k: z[k] for k in z.files})

    def test_pi05_droid_sends_one_frame_with_its_noise(self):
        z = self.load(
            base=self.rng.integers(0, 255, (3, 4, 4, 3), dtype=np.uint8),
            wrist=self.rng.integers(0, 255, (3, 4, 4, 3), dtype=np.uint8),
            state=self.rng.standard_normal((3, 8), dtype=np.float32),
            task=np.array("pick"),
            noise=self.rng.standard_normal((3, 15, 32), dtype=np.float32),
            actions=self.rng.standard_normal((3, 15, 32), dtype=np.float32),
        )
        req, want, width = sc.build("pi05_droid", z, frame=1)
        np.testing.assert_array_equal(req["images"]["base_0_rgb"], z["base"][1])
        np.testing.assert_array_equal(req["images"]["left_wrist_0_rgb"], z["wrist"][1])
        np.testing.assert_array_equal(req["noise"][0], z["noise"][1])
        self.assertEqual(req["noise"].shape, (1, 15, 32))
        self.assertEqual((req["prompt"], width, want.shape), ("pick", 8, (15, 8)))
        # The served chunk is 32 wide; only DROID's 8 dims are compared.
        served = np.concatenate([want, np.full((15, 24), 9.0, np.float32)], axis=1)
        self.assertEqual(sc.compare(served, want, width), (0.0, [15, 32]))
        with self.assertRaises(sc.CheckError):
            sc.build("pi05_droid", z, frame=3)

    def test_molmo_frames_are_truncated_uint8_hwc(self):
        img = np.zeros((1, 3, 2, 2), np.float32)
        img[0, 0, 0, 0], img[0, 1, 0, 1], img[0, 2, 1, 1] = 0.5, 1.0, 0.999
        z = self.load(
            **{"obs_images.top": img, "obs_images.side": img},
            obs_state=np.zeros((1, 6), np.float32),
            task=np.array("t"),
            actions=np.zeros((1, 30, 6), np.float32),
        )
        req, want, width = sc.build("molmoact2", z)
        self.assertEqual(sorted(req["images"]), ["side", "top"])
        top = req["images"]["top"]
        self.assertEqual((top.dtype, top.shape), (np.uint8, (2, 2, 3)))
        self.assertEqual((top[0, 0, 0], top[0, 1, 1], top[1, 1, 2]), (127, 255, 254))
        self.assertNotIn("noise", req)
        self.assertEqual((req["state"].shape, want.shape, width), ((6,), (30, 6), None))
        # Frames already in [0, 255] are not scaled again.
        self.assertEqual(sc.molmo_frame(img * 255)[0, 0, 0], 127)

    def test_xr1_sends_the_four_frame_history(self):
        crops = {
            f"crop_{k}": self.rng.integers(0, 255, (4, 4, 4, 3), dtype=np.uint8)
            for k in sc.XR1_CAMERAS.values()
        }
        ref = self.rng.standard_normal((1, 16, 60), dtype=np.float32)
        z = self.load(
            **crops,
            state14=np.zeros((4, 14), np.float32),
            task=np.array("t"),
            noise1=np.zeros((1, 16, 60), np.float32),
            ref1=ref,
        )
        req, want, _ = sc.build("xr1", z)
        np.testing.assert_array_equal(
            req["images"]["eye_in_hand"], crops["crop_robot0_eye_in_hand"]
        )
        np.testing.assert_array_equal(want, ref[0, :, :12])
        meta = {"cameras": list(sc.XR1_CAMERAS), "height": 4, "width": 4, "history": 4,
                "state_len": 14, "noise": "input"}
        sc.check_metadata(meta, req)
        for bad in ({"history": 1}, {"state_len": 8}, {"noise": "constant"}, {"cameras": ["a"]}):
            with self.assertRaises(sc.CheckError, msg=bad):
                sc.check_metadata({**meta, **bad}, req)

    def test_the_default_bound_follows_the_precision_and_checkpoint(self):
        def bound(kind, precision, cameras=(), device="GPU"):
            meta = {"precision": precision, "cameras": list(cameras), "device": device}
            return sc.choose_bound(kind, meta)[0]

        self.assertEqual(bound("pi05_droid", "f16"), sc.PI05_DROID_F16_BOUND)
        self.assertEqual(bound("pi05_droid", "f32", device="CPU"), 5e-3)
        self.assertEqual((bound("xr1", "f16"), bound("xr1", "f32")), (1e-2, 1e-3))
        libero, droid = ["wrist_image", "image"], ["exterior_1_left", "exterior_2_left", "wrist_left"]
        self.assertEqual((bound("molmoact2", "f16", libero), bound("molmoact2", "f32", libero)),
                         (1e-2, 1e-3))
        self.assertEqual((bound("molmoact2", "f16", droid), bound("molmoact2", "f32", droid)),
                         (3e-3, 1e-3))
        self.assertEqual((bound("molmoact2", "f16", ["top", "side"]),
                          bound("molmoact2", "f32", ["top", "side"])), (5e-2, 1e-2))
        _, why = sc.choose_bound("molmoact2", {"precision": "f16", "cameras": droid, "device": "GPU"})
        self.assertEqual(why, "molmoact2 DROID at f16 on GPU")
        for kind, precision, cameras in (
            ("molmoact2", "f16", ["image"]),  # unknown checkpoint
            ("xr1", "", ()),  # no precision stated
            ("pi05_droid", "int8", ()),
        ):
            with self.assertRaisesRegex(sc.CheckError, "pass --bound"):
                bound(kind, precision, cameras)

    def test_a_wrong_shape_or_non_finite_reply_fails(self):
        want = np.zeros((2, 3), np.float32)
        with self.assertRaises(sc.CheckError):
            sc.compare(np.zeros((3, 3)), want, None)
        with self.assertRaises(sc.CheckError):
            sc.compare(np.full((2, 3), np.nan), want, None)
        self.assertAlmostEqual(sc.compare(np.full((2, 3), 0.25), want, None)[0], 0.25)


def _real(var):
    path = os.environ.get(var)
    return path if path and os.path.exists(path) else None


@unittest.skipIf(np is None, "needs numpy")
class RealReferences(unittest.TestCase):
    def build(self, var, kind):
        path = _real(var)
        if path is None:
            self.skipTest(f"{var} is unset or missing")
        with np.load(path) as z:
            return sc.build(kind, z)

    def test_pi05_droid(self):
        req, want, _ = self.build("CASCADIA_REF_PI05_DROID", "pi05_droid")
        self.assertEqual(_shapes(req), {
            "images.base_0_rgb": ("uint8", (224, 224, 3)),
            "images.left_wrist_0_rgb": ("uint8", (224, 224, 3)),
            "state": ("float32", (8,)),
            "noise": ("float32", (1, 15, 32)),
        })
        self.assertEqual(want.shape, (15, 8))

    def test_molmoact2_droid(self):
        req, want, _ = self.build("CASCADIA_REF_MOLMOACT2", "molmoact2")
        cams = ["exterior_1_left", "exterior_2_left", "wrist_left"]
        self.assertEqual(sorted(req["images"]), cams)
        for c in cams:
            self.assertEqual(_shapes(req)[f"images.{c}"], ("uint8", (378, 378, 3)))
        self.assertEqual(_shapes(req)["state"], ("float32", (8,)))
        self.assertNotIn("noise", req)
        self.assertEqual(want.shape, (15, 8))

    def test_xr1(self):
        req, want, _ = self.build("CASCADIA_REF_XR1", "xr1")
        self.assertEqual(_shapes(req), {
            "images.agentview_left": ("uint8", (4, 256, 256, 3)),
            "images.agentview_right": ("uint8", (4, 256, 256, 3)),
            "images.eye_in_hand": ("uint8", (4, 256, 256, 3)),
            "state": ("float32", (4, 14)),
            "noise": ("float32", (1, 16, 60)),
        })
        self.assertEqual(want.shape, (16, 12))


if __name__ == "__main__":
    unittest.main()
