import math
from pathlib import Path
import struct
import tempfile
import unittest
from capture_read import read_capture


class CaptureFormatTests(unittest.TestCase):
    def test_f32_preserves_large_residuals_and_f16_files_remain_readable(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'capture.bin'
            for magic, dtype, values in [(b'INKCAP01', 'e', (1., -2.)),
                                          (b'INKCAP02', 'f', (100000.125, -900000.5))]:
                header = magic + struct.pack('<IIIIQ', 2, 10, 0, 1, 7)
                path.write_bytes(header + struct.pack('<q2' + dtype, 42, *values))
                result = read_capture(path)
                self.assertEqual(result['tokens'], [42])
                self.assertEqual(result['states'], [values])
                self.assertTrue(all(math.isfinite(x) for x in result['states'][0]))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                read_capture(path)


if __name__ == '__main__':
    unittest.main()
