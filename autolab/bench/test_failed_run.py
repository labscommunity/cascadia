import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import lab


class FailedRunTests(unittest.TestCase):
    def test_incomplete_phase_is_saved_and_stops_following_phases(self):
        for failed in [dict(errors=['timeout'], completed=0), dict(errors=[], completed=1)]:
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                telemetry = MagicMock()
                args = SimpleNamespace(exp='fixture', phases=['first:2:32','never:2:32'], cap=3, prompt_words=0)
                with patch.object(lab, 'LAB', directory), patch.object(lab,'Telemetry',return_value=telemetry), \
                     patch.object(lab,'raw_telemetry_path',return_value=str(Path(directory)/'raw.jsonl')), \
                     patch.object(lab,'run_phase',return_value=failed) as phase, patch.object(lab.time,'sleep') as sleep:
                    self.assertEqual(lab.cmd_bench(args),5)
                phase.assert_called_once();sleep.assert_not_called()
                telemetry.stop_ev.set.assert_called_once()
                self.assertEqual(json.loads((Path(directory)/'experiments/fixture/phases.json').read_text()),[failed])

    def test_failed_warmup_never_enters_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(exp='fixture',files=[],warm_streams=15,cap=3,warm=0)
            with patch.object(lab,'LAB',directory),patch.object(lab,'run_phase',return_value=dict(errors=['timeout'],completed=0)), \
                 patch.object(lab,'gate') as gate:
                self.assertEqual(lab.cmd_run(args),5)
            gate.assert_not_called()

    def test_transport_failure_stops_serial_gate_requests(self):
        with patch.object(lab,'chat',return_value=dict(error='timeout',tokens=0,text='')) as chat:
            ok,rows=lab.gate()
        self.assertFalse(ok);self.assertEqual(len(rows),1);chat.assert_called_once()


if __name__=='__main__':unittest.main()
