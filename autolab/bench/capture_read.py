#!/usr/bin/env python3
"""Read an INKCAP01 capture without external dependencies (f16 -> Python float)."""
import argparse
import json
import struct
from pathlib import Path


def read_capture(path):
    data = Path(path).read_bytes()
    if len(data) < 32 or data[:8] != b'INKCAP01':
        raise ValueError('not an INKCAP01 capture')
    hidden, rank, slot, prompt_rows, sequence = struct.unpack_from('<IIIIQ', data, 8)
    if not 1 <= hidden <= 65536:
        raise ValueError('invalid hidden width')
    size = 8 + 2 * hidden
    if (len(data) - 32) % size:
        raise ValueError('incomplete residual record')
    tokens, states = [], []
    for offset in range(32, len(data), size):
        tokens.append(struct.unpack_from('<q', data, offset)[0])
        states.append(struct.unpack_from('<%de' % hidden, data, offset + 8))
    return dict(hidden=hidden, rank=rank, slot=slot, prompt_rows=prompt_rows,
                sequence=sequence, tokens=tokens, states=states)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path')
    a = ap.parse_args()
    result = read_capture(a.path)
    result['positions'] = len(result.pop('states'))
    print(json.dumps(result))
