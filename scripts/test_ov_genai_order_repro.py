"""Offline tests for scripts/ov_genai_order_repro.py (the order and the verdict
rules; no model and no openvino_genai needed).

    python3 -m unittest scripts/test_ov_genai_order_repro.py
"""

import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ov_genai_order_repro as r  # noqa: E402


class Sequence(unittest.TestCase):
    def test_order(self):
        self.assertEqual(r.sequence(), ["A", "B", "A", "A", "B", "B", "A"])

    def test_every_prompt_known(self):
        self.assertTrue(all(p in r.PROMPTS for p in r.sequence()))


class Group(unittest.TestCase):
    def test_groups_in_first_seen_order(self):
        g = r.group([("A", "a1"), ("B", "b1"), ("A", "a2"), ("B", "b1")])
        self.assertEqual(list(g), ["A", "B"])
        self.assertEqual(g["A"], ["a1", "a2"])
        self.assertEqual(g["B"], ["b1", "b1"])

    def test_lines(self):
        self.assertEqual(
            r.group_lines([("A", "x"), ("B", "y"), ("A", "x")]),
            ["prompt A: x x", "prompt B: y"],
        )


class Verdict(unittest.TestCase):
    def test_stable(self):
        ans = [(p, "h" + p) for p in r.sequence()]
        self.assertEqual(r.verdict(ans), "stable")

    def test_depends_on_previous(self):
        ans = [(p, "h" + p) for p in r.sequence()]
        ans[3] = ("A", "other")  # A right after A
        self.assertEqual(r.verdict(ans), "depends-on-previous")

    def test_empty_is_stable(self):
        self.assertEqual(r.verdict([]), "stable")


class Lines(unittest.TestCase):
    def test_run_line(self):
        self.assertEqual(r.run_line(2, "A", "0123456789ab", 900), "run 2 prompt A sha 0123456789ab chars 900")

    def test_sha(self):
        self.assertEqual(r.sha(""), "e3b0c44298fc")


class DryRun(unittest.TestCase):
    def test_prints_plan_without_model(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(r.main(["/nonexistent", "GPU", "--dry-run"]), 0)
        text = out.getvalue()
        self.assertIn("order A B A A B B A", text)
        self.assertIn("device GPU", text)


class Import(unittest.TestCase):
    def test_import_needs_no_openvino_genai(self):
        # A None entry in sys.modules makes `import openvino_genai` fail.
        code = (
            "import sys; sys.modules['openvino_genai'] = None; sys.path.insert(0, %r); "
            "import ov_genai_order_repro as m; print(m.verdict([]))" % HERE
        )
        out = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, check=True)
        self.assertEqual(out.stdout.decode().strip(), "stable")


if __name__ == "__main__":
    unittest.main()
