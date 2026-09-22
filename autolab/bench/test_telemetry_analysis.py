import unittest
from telemetry_analysis import normalize_records


class TelemetryAnalysisTest(unittest.TestCase):
    def test_swapped_roles_skewed_clock_backlog_and_repeated_windows(self):
        # The worker is a day ahead; the entry receive clock is 600 seconds
        # ahead. An old window appears for the first time in the first poll.
        old = dict(rank=0, at='2099-01-01T00:00:00Z', rt=600, window_ms=10000, frames=100)
        fresh = dict(rank=0, at='2099-01-01T00:00:10Z', rt=1699, window_ms=10000, frames=200)
        records = [dict(rank=8, lt=1100, rt=1700, profs=[old, fresh]),
                   dict(rank=8, lt=1102, rt=1702, profs=[fresh]),
                   dict(rank=0, lt=1100, rt=1700, profs=[dict(rank=8, at='tag', probe=1)]),
                   dict(rank=0, lt=1110, rt=1710, profs=[dict(rank=8, at='later', rt=1710, window_ms=10000)])]
        out = normalize_records(records)
        self.assertEqual((out[0]['rank'], out[0]['box']), (0, 8))
        self.assertEqual((out[2]['rank'], out[2]['box']), (8, 0))
        self.assertEqual([p['lt'] for p in out[0]['profs']], [0, 1099])
        self.assertEqual(out[1]['profs'], [])
        self.assertEqual(out[2]['profs'], [])
        self.assertNotIn('lt', fresh)  # raw evidence stays untouched

    def test_legacy_records_without_receive_time(self):
        out = normalize_records([dict(rank=3, lt=20, profs=[dict(window_ms=1000, frames=5)])])
        self.assertEqual(out[0]['rank'], 3)
        self.assertEqual(out[0]['profs'][0]['lt'], 20)


if __name__ == '__main__':
    unittest.main()
