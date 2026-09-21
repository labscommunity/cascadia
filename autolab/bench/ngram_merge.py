#!/usr/bin/env python3
"""Merge CSNGRAM1 phrase tables, preserving counts and the four strongest followers."""
import argparse
from pathlib import Path
import struct

MAGIC = b'CSNGRAM1'
MAX_CONTEXTS = 2_000_000
MAX_U32 = (1 << 32) - 1


def read(path):
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:8] != MAGIC:
        raise ValueError('not a CSNGRAM1 table')
    count = struct.unpack_from('<Q', data, 8)[0]
    if count > MAX_CONTEXTS:
        raise ValueError('table exceeds the engine context limit')
    rows, offset = {}, 16
    for _ in range(count):
        key, seen, n = struct.unpack_from('<QIB', data, offset)
        offset += 13
        if n > 4 or key in rows:
            raise ValueError('invalid follower count or repeated context')
        followers = []
        for _ in range(n):
            followers.append(struct.unpack_from('<qI', data, offset))
            offset += 12
        rows[key] = (seen, followers)
    if offset != len(data):
        raise ValueError('trailing bytes')
    return rows


def merge(current, seed):
    rows = dict(current)
    for key, (seen, followers) in seed.items():
        if key not in rows:
            if len(rows) < MAX_CONTEXTS:
                rows[key] = (seen, followers)
            continue
        have_seen, have_followers = rows[key]
        counts = dict(have_followers)
        for token, count in followers:
            counts[token] = min(MAX_U32, counts.get(token, 0) + count)
        # Rust chooses the LAST equal maximum, so keep larger counts last.
        best = sorted(counts.items(), key=lambda x: (x[1], x[0]))[-4:]
        rows[key] = (min(MAX_U32, have_seen + seen), best)
    return rows


def write(path, rows):
    path = Path(path)
    temp = path.with_suffix('.merge-part')
    with temp.open('wb') as f:
        f.write(MAGIC + struct.pack('<Q', len(rows)))
        for key, (seen, followers) in rows.items():
            f.write(struct.pack('<QIB', key, seen, len(followers)))
            for token, count in followers:
                f.write(struct.pack('<qI', token, count))
    temp.replace(path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('current'); ap.add_argument('seed'); ap.add_argument('out')
    a = ap.parse_args()
    current = read(a.current) if Path(a.current).exists() else {}
    seeded = merge(current, read(a.seed))
    write(a.out, seeded)
    print('phrase table contexts:', len(current), '->', len(seeded))
