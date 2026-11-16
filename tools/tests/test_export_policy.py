"""Unit tests for the helpers in tools/export_policy/export_policy.py (standard library only)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "export_policy"))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import export_policy as ep  # noqa: E402


def _normalizer(features, norm=("MEAN_STD", "MEAN_STD"), state_file="pre_step.safetensors"):
    step = {
        "registry_name": "normalizer_processor",
        "config": {"features": features, "norm_map": {"STATE": norm[0], "ACTION": norm[1]}},
    }
    if state_file:
        step["state_file"] = state_file
    return {"steps": [{"registry_name": "to_batch_processor", "config": {}}, step]}


def _unnormalizer(features, state_file="post_step.safetensors"):
    step = {
        "registry_name": "unnormalizer_processor",
        "config": {"features": features, "norm_map": {"ACTION": "MEAN_STD"}},
    }
    if state_file:
        step["state_file"] = state_file
    return {"steps": [step]}


STATE = {"observation.state": {"type": "STATE", "shape": [8]}}
ACTION = {"action": {"type": "ACTION", "shape": [7]}}
FILES = {"config.json", "pre_step.safetensors", "post_step.safetensors"}


class StatsProblem(unittest.TestCase):
    def test_present(self):
        self.assertIsNone(ep.stats_problem(_normalizer(STATE), _unnormalizer(ACTION), FILES))
        # offline: the listing is unknown, the JSON still decides
        self.assertIsNone(ep.stats_problem(_normalizer(STATE), _unnormalizer(ACTION), None))

    def test_droid_empty_features(self):
        why = ep.stats_problem(_normalizer({}), _unnormalizer({}), FILES)
        self.assertIn("no STATE feature", why)

    def test_old_format(self):
        self.assertIn("older LeRobot format", ep.stats_problem(None, None, FILES))

    def test_missing_state_file(self):
        why = ep.stats_problem(_normalizer(STATE), _unnormalizer(ACTION), {"config.json"})
        self.assertIn("pre_step.safetensors", why)
        why = ep.stats_problem(_normalizer(STATE, state_file=None), _unnormalizer(ACTION), FILES)
        self.assertIn("names no state_file", why)

    def test_missing_action_stats(self):
        why = ep.stats_problem(_normalizer(STATE), _unnormalizer({}), FILES)
        self.assertIn("ACTION", why)
        self.assertIn("policy_postprocessor.json", ep.stats_problem(_normalizer(STATE), None, FILES))

    def test_identity_state_needs_no_stats(self):
        pre = _normalizer({}, norm=("IDENTITY", "MEAN_STD"), state_file=None)
        self.assertIsNone(ep.stats_problem(pre, _unnormalizer(ACTION), FILES))


class LicenceFiles(unittest.TestCase):
    def test_root_licence_files_only(self):
        files = {"LICENSE", "LICENCE.md", "COPYING", "docs/LICENSE", "README.md", "NOTICE"}
        self.assertEqual(ep.license_files(files), ["COPYING", "LICENCE.md", "LICENSE"])
        self.assertEqual(ep.license_files(None), [])


def _manifest(mode="mean_std", width=8, rtc=False):
    blank = {k: [0.0] * width for k in ("mean", "std", "q01", "q99", "min", "max")}
    pre = {"state": {"name": "state", "shape": [width], "type": "STATE", **blank}}
    if rtc:
        pre["prev_chunk_left_over"] = {"name": "prev_chunk_left_over", "shape": [7], **blank}
    return {
        "model": {
            "preprocessors": [{"type": "noise"}, {"type": "normalize", "mode": mode, "stats": pre}],
            "postprocessors": [
                {
                    "type": "denormalize",
                    "mode": mode,
                    "stats": {"action": {"name": "action", "shape": [7], **blank}},
                }
            ],
        }
    }


class ApplyStats(unittest.TestCase):
    def test_mean_std_with_rtc_leftover(self):
        m = _manifest(rtc=True)
        stats = {"state": {"mean": [1.0] * 8, "std": [2.0] * 8}, "action": {"mean": [3] * 7, "std": [4] * 7}}
        done = ep.apply_stats(m, stats)
        self.assertEqual(done, ["normalize.state", "normalize.prev_chunk_left_over", "denormalize.action"])
        pre = m["model"]["preprocessors"][1]["stats"]
        self.assertEqual(pre["state"]["mean"], [1.0] * 8)
        self.assertEqual(pre["prev_chunk_left_over"]["std"], [4.0] * 7)
        self.assertEqual(m["model"]["postprocessors"][0]["stats"]["action"]["mean"], [3.0] * 7)

    def test_mode_keys_required(self):
        m = _manifest(mode="quantiles")
        stats = {"state": {"mean": [1.0] * 8, "std": [1.0] * 8}, "action": {"q01": [0] * 7, "q99": [1] * 7}}
        with self.assertRaisesRegex(ValueError, "q01, q99"):
            ep.apply_stats(m, stats)

    def test_width_must_match(self):
        stats = {"state": {"mean": [1.0] * 32, "std": [1.0] * 32}, "action": {"mean": [0] * 7, "std": [1] * 7}}
        with self.assertRaisesRegex(ValueError, "32 values; the package's state has 8"):
            ep.apply_stats(_manifest(), stats)

    def test_needs_a_normalize_step(self):
        with self.assertRaisesRegex(ValueError, "no normalize"):
            ep.apply_stats({"model": {"preprocessors": [], "postprocessors": []}}, {"state": {}, "action": {}})


class LoadStats(unittest.TestCase):
    def _write(self, doc):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(doc, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return Path(f.name)

    def test_accepts_lerobot_state_key(self):
        doc = ep.load_stats(self._write({"observation.state": {"mean": [0]}, "action": {"q01": [0]}}))
        self.assertIn("state", doc)

    def test_rejects_bad_files(self):
        for bad, msg in (
            ([1], "JSON object"),
            ({"state": {"mean": [0]}}, "\"action\""),
            ({"state": {"mean": [float("nan")]}, "action": {"mean": [0]}}, "finite"),
            ({"state": {"mean": "x"}, "action": {"mean": [0]}}, "list of finite"),
        ):
            with self.assertRaisesRegex(ValueError, msg):
                ep.load_stats(self._write(bad))


if __name__ == "__main__":
    unittest.main()
