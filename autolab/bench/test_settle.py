import unittest
from unittest.mock import patch

import lab


class SettleTests(unittest.TestCase):
    def run_reports(self, reports):
        def rows(report):
            result = {rank: dict(state="active", phase="serving", restarts=0, files="same")
                      for rank in range(11)}
            result[8].update(report.get("head", {}))
            return result

        with patch.object(lab, "status", side_effect=reports) as status, \
             patch.object(lab, "fleet_rows", side_effect=rows), \
             patch.object(lab.time, "sleep"), patch.object(lab, "log"):
            self.assertTrue(lab.settle({"cascadia": "expected"}, cap=30))
        return status.call_count

    def report(self, timestamp, phase="pipeline chain ready; accepting requests total=11", **head):
        return dict(time=timestamp, fleet=dict(files={"cascadia": "expected"}),
                    head=dict(phase=phase, **head))

    def test_idle_ready_head_needs_three_advancing_stable_reports(self):
        reports = [self.report(t) for t in [1, 1, 2, 2, 3, 4]]
        self.assertEqual(self.run_reports(reports), 6)

    def test_loading_wrong_chain_and_inactive_head_do_not_count_as_ready(self):
        invalid = [self.report(1, "loading the model"),
                   self.report(2, "pipeline chain ready; accepting requests total=10"),
                   self.report(3, state="failed"),
                   self.report(4, "pipeline readiness probe failed; requests remain refused")]
        self.assertEqual(self.run_reports(invalid + [self.report(t) for t in [5, 6, 7]]), 7)

    def test_restart_resets_stability(self):
        reports = [self.report(1), self.report(2)]
        reports += [self.report(t, restarts=1) for t in [3, 4, 5, 6]]
        self.assertEqual(self.run_reports(reports), 6)


if __name__ == "__main__":
    unittest.main()
