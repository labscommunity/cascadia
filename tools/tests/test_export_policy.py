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
import index_guard as ig  # noqa: E402


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


class LocalCheckpoint(unittest.TestCase):
    def test_card_license_reads_the_front_matter(self):
        self.assertEqual(ep.card_license("---\nlicense: apache-2.0\ntags: [x]\n---\n# m"), "apache-2.0")
        self.assertEqual(ep.card_license("---\nlicense: 'gemma'\n---\n"), "gemma")
        self.assertIsNone(ep.card_license("# no front matter\nlicense: mit"))
        self.assertIsNone(ep.card_license(""))

    def _checkpoint(self, root: Path, kind: str) -> None:
        (root / "config.json").write_text(json.dumps({"type": kind, "chunk_size": 50}))
        (root / "policy_preprocessor.json").write_text(json.dumps(_normalizer(STATE)))
        (root / "policy_postprocessor.json").write_text(json.dumps(_unnormalizer(ACTION)))
        for name in ("pre_step.safetensors", "post_step.safetensors", "LICENSE"):
            (root / name).write_text("x")
        (root / "README.md").write_text("---\nlicense: apache-2.0\n---\n")

    def test_a_local_directory_is_read_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            self._checkpoint(Path(d), "smolvla")
            out = ep.preflight(d)
        self.assertIsNone(out["error"])
        self.assertEqual(out["revision"], ep.LOCAL_REVISION)
        self.assertEqual(out["license"], "apache-2.0")
        self.assertEqual(out["license_files"], ["LICENSE"])
        self.assertEqual(out["policy_type"], "smolvla")
        self.assertTrue(out["stats"]["ok"], out["stats"])
        self.assertEqual(out["notes"], [])

    def test_a_local_pi05_skips_the_gated_check_with_a_note(self):
        with tempfile.TemporaryDirectory() as d:
            self._checkpoint(Path(d), "pi05")
            out = ep.preflight(d)
        self.assertIsNone(out["gated"])
        self.assertEqual(len(out["notes"]), 1)
        self.assertIn("google/paligemma-3b-pt-224", out["notes"][0])

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_a_link_out_of_the_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as outside, tempfile.TemporaryDirectory() as d:
            secret = Path(outside) / "id_key"
            secret.write_text("private")
            self._checkpoint(Path(d), "smolvla")
            (Path(d) / "LICENSE").unlink()
            (Path(d) / "LICENSE").symlink_to(secret)
            out = ep.preflight(d)
            self.assertIn("leaves the checkpoint directory", out["error"])
            self.assertIn("LICENSE", out["error"])
            # Even a link that stays inside is not a licence file to copy.
            (Path(d) / "LICENSE").unlink()
            (Path(d) / "LICENSE").symlink_to(Path(d) / "README.md")
            self.assertNotIn("LICENSE", ep.local_files(Path(d)))
            with self.assertRaises(ValueError):
                ep.local_file(Path(d), "LICENSE")
            # A weights file linked from outside is refused the same way.
            (Path(d) / "LICENSE").unlink()
            (Path(d) / "model.safetensors").symlink_to(secret)
            self.assertEqual(ep.escaping_link(Path(d)), "model.safetensors")

    def test_a_local_directory_without_config_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = ep.preflight(d)
        self.assertIn("no config.json", out["error"])


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

    def test_wider_than_the_package_is_refused(self):
        stats = {"state": {"mean": [1.0] * 32, "std": [1.0] * 32}, "action": {"mean": [0] * 7, "std": [1] * 7}}
        with self.assertRaisesRegex(ValueError, "32 values; the package's state has 8"):
            ep.apply_stats(_manifest(), stats)

    def test_narrower_stats_pad_with_the_identity(self):
        # pi0.5 DROID: the package's state and action are 32 wide, real stats 8 wide.
        for mode, keys, pad in (
            ("mean_std", ("mean", "std"), (0.0, 1.0)),
            ("quantiles", ("q01", "q99"), (-1.0, 1.0)),
            ("min_max", ("min", "max"), (-1.0, 1.0)),
        ):
            m = _manifest(mode=mode, width=32)
            real = {k: [5.0] * 8 for k in keys}
            ep.apply_stats(m, {"state": real, "action": {k: [5.0] * 7 for k in keys}})
            state = m["model"]["preprocessors"][1]["stats"]["state"]
            for k, v in zip(keys, pad):
                self.assertEqual(state[k], [5.0] * 8 + [v] * 24, (mode, k))

    def test_a_step_without_mode_is_mean_std(self):
        m = _manifest()
        for step in m["model"]["preprocessors"][1:] + m["model"]["postprocessors"]:
            del step["mode"]
        stats = {"state": {"q01": [0] * 8, "q99": [1] * 8}, "action": {"q01": [0] * 7, "q99": [1] * 7}}
        with self.assertRaisesRegex(ValueError, "mean_std"):
            ep.apply_stats(m, stats)

    def test_needs_a_normalize_step(self):
        with self.assertRaisesRegex(ValueError, "no normalize"):
            ep.apply_stats({"model": {"preprocessors": [], "postprocessors": []}}, {"state": {}, "action": {}})


class PackageMode(unittest.TestCase):
    def test_smolvla_is_always_mean_std(self):
        self.assertEqual(ep.package_mode("smolvla", _normalizer(STATE, norm=("QUANTILES", "QUANTILES")), {}), "mean_std")

    def test_pi05_follows_the_checkpoint_or_defaults_to_quantiles(self):
        self.assertEqual(ep.package_mode("pi05", _normalizer(STATE), {}), "mean_std")
        self.assertEqual(ep.package_mode("pi05", _normalizer(STATE, norm=("QUANTILES", "QUANTILES")), {}), "quantiles")
        mixed = _normalizer(STATE, norm=("MEAN_STD", "QUANTILES"))
        self.assertEqual(ep.package_mode("pi05", mixed, {}), "quantiles")
        self.assertEqual(ep.package_mode("pi05", mixed, {"normalization_mode": "MEAN_STD"}), "mean_std")
        self.assertEqual(ep.package_mode("pi05", None, {}), "quantiles")


class Widths(unittest.TestCase):
    def test_from_config_features(self):
        config = {
            "input_features": {"observation.state": {"type": "STATE", "shape": [8]},
                               "observation.images.a": {"type": "VISUAL", "shape": [3, 2, 2]}},
            "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        }
        self.assertEqual(ep.feature_width(config, "input_features", "STATE"), 8)
        self.assertEqual(ep.feature_width(config, "output_features", "ACTION"), 7)
        self.assertIsNone(ep.feature_width({}, "output_features", "ACTION"))


class CopyPackage(unittest.TestCase):
    def test_copies_subdirectories_and_skips_a_nested_ir(self):
        with tempfile.TemporaryDirectory() as t:
            src, dst = Path(t, "src"), Path(t, "dst")
            (src / "ir").mkdir(parents=True)
            (src / "assets").mkdir()
            for rel in ("manifest.json", "ir/model.xml", "ir/model.bin", "assets/a.txt", "tokenizer.xml"):
                (src / rel).write_text(rel)
            ep.copy_package(src, dst, "ir/model.xml")
            got = sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
            self.assertEqual(got, ["assets/a.txt", "manifest.json", "tokenizer.xml"])

    def test_refuses_an_ir_outside_the_package(self):
        with tempfile.TemporaryDirectory() as t:
            for bad in ("../x.xml", "/abs/x.xml", ""):
                with self.assertRaises(ValueError):
                    ep.copy_package(Path(t), Path(t, "o"), bad)


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


LOCK = """torchcodec==0.11.1 ; sys_platform == 'darwin' \\
    --hash=sha256:%s
    # via lerobot
physicalai @ git+https://example.invalid/physicalai@abc
torch==2.11.0+cpu ; sys_platform != 'darwin' \\
    --hash=sha256:%s
""" % ("a" * 64, "c" * 64)


class IndexGuard(unittest.TestCase):
    def test_parse_lock(self):
        pins, hashes = ig.parse_lock(LOCK)
        self.assertEqual(pins, [("torchcodec", "0.11.1"), ("torch", "2.11.0+cpu")])
        self.assertEqual(hashes, {"torchcodec": {"a" * 64}, "torch": {"c" * 64}})

    def test_local_version_shadow_is_flagged(self):
        pins, hashes = ig.parse_lock(LOCK)
        html = '<a href="/whl/cpu/torchcodec-0.11.1%%2Bcpu-cp312-cp312-win_amd64.whl#sha256=%s">' % (
            "b" * 64
        )
        links = {"torchcodec": ig.wheel_links(html)}
        self.assertEqual(links["torchcodec"][0][0], "torchcodec-0.11.1+cpu-cp312-cp312-win_amd64.whl")
        found = ig.shadows(pins, hashes, links)
        self.assertEqual(len(found), 1)
        self.assertIn("torchcodec==0.11.1", found[0])
        hashes["torchcodec"].add("b" * 64)
        self.assertEqual(ig.shadows(pins, hashes, links), [])

    def test_unreachable_wheels_are_ignored(self):
        pins, hashes = ig.parse_lock(LOCK)
        links = {
            "torchcodec": [
                ("torchcodec-0.11.1+cpu-cp310-cp310-win_amd64.whl", "d" * 64),
                ("torchcodec-0.11.2+cpu-cp312-cp312-win_amd64.whl", "d" * 64),
                ("torchcodec-0.11.10-cp312-cp312-win_amd64.whl", "d" * 64),
            ],
            # a local pin matches only itself
            "torch": [("torch-2.11.0-cp312-cp312-macosx_11_0_arm64.whl", "d" * 64)],
        }
        self.assertEqual(ig.shadows(pins, hashes, links), [])

    def test_python_tags(self):
        for name, ok in (
            ("x-1-cp312-cp312-win_amd64.whl", True),
            ("x-1-py3-none-any.whl", True),
            ("x-1-cp39-abi3-manylinux_2_17_x86_64.whl", True),
            ("x-1-cp313-abi3-manylinux_2_17_x86_64.whl", False),
            ("x-1-cp311-cp311-win_amd64.whl", False),
        ):
            self.assertEqual(ig.cp312_compatible(name), ok, name)

    def test_committed_lock_takes_torchcodec_cpu_from_the_torch_index(self):
        with open(os.path.join(_DIR, "requirements.txt"), encoding="utf-8") as f:
            pins, _ = ig.parse_lock(f.read())
        self.assertIn(("torchcodec", "0.11.1+cpu"), pins)


if __name__ == "__main__":
    unittest.main()
