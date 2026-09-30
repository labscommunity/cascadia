import tempfile
from pathlib import Path
import unittest
from ngram_merge import merge, read, write, MAX_U32


class NgramMergeTest(unittest.TestCase):
    def test_merge_keeps_both_histories_and_engine_limits(self):
        a = {1: (10, [(11, 8), (12, 2)]), 2: (MAX_U32, [(20, MAX_U32)])}
        b = {1: (20, [(11, 10), (13, 4), (14, 3), (15, 3)]), 2: (1, [(20, 1)]), 3: (1, [(30, 1)])}
        result = merge(a, b)
        self.assertEqual(result[1], (30, [(14, 3), (15, 3), (13, 4), (11, 18)]))
        self.assertEqual(result[2], (MAX_U32, [(20, MAX_U32)]))
        self.assertEqual(result[3], b[3])
        self.assertEqual(a[1][0], 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'table.bin'
            write(path, result)
            self.assertEqual(read(path), result)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(Exception):
                read(path)


if __name__ == '__main__':
    unittest.main()
