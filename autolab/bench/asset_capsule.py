#!/usr/bin/env python3
"""Carry a bounded model-asset archive through the existing signed ELF release.

The executable remains an ELF; the fleet updater verifies its ordinary full
file hash. Extraction additionally requires the archive hash from the signed
overrides. No publisher, updater, file-server or tunnel changes are needed.
"""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import tarfile
import tempfile

MAGIC = b'CSMTP-CAPSULE1\0\0'
FOOTER = struct.Struct('<16sQ32s')
MAX_ARCHIVE = 700 * 1024**2
MAX_EXTRACTED = 3 * 1024**3
MAX_FILES = 256
assert len(MAGIC) == 16


class SliceReader:
    def __init__(self, source, size):
        self.source, self.left = source, size

    def read(self, size=-1):
        size = self.left if size < 0 else min(self.left, size)
        data = self.source.read(size)
        self.left -= len(data)
        return data


def make(binary, archive, out):
    binary, archive, out = map(Path, (binary, archive, out))
    if out.exists() or out.resolve() in (binary.resolve(), archive.resolve()):
        raise ValueError('output must be a new, separate file')
    with binary.open('rb') as f:
        if f.read(5) != b'\x7fELF\x02':
            raise ValueError('expected a 64-bit ELF executable')
    size = archive.stat().st_size
    if not 0 < size <= MAX_ARCHIVE:
        raise ValueError('split archives larger than 700 MiB across planned releases')
    digest = hashlib.sha256()
    with out.open('xb') as dest:
        with binary.open('rb') as source:
            shutil.copyfileobj(source, dest, 1024**2)
        with archive.open('rb') as source:
            for chunk in iter(lambda: source.read(1024**2), b''):
                digest.update(chunk); dest.write(chunk)
        dest.write(FOOTER.pack(MAGIC, size, digest.digest()))
    out.chmod(0o755)
    return digest.hexdigest()


def extract(binary, target, expected):
    binary, target = map(Path, (binary, target))
    target = target.resolve()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('expected a lowercase SHA-256 digest from the signed release')
    marker = target / '.capsules' / expected
    if marker.is_file():
        return False
    size = binary.stat().st_size
    if size < FOOTER.size:
        raise ValueError('executable has no asset capsule')
    with binary.open('rb') as source:
        source.seek(-FOOTER.size, 2)
        magic, count, digest = FOOTER.unpack(source.read(FOOTER.size))
        if magic != MAGIC or not 0 < count <= MAX_ARCHIVE or count + FOOTER.size >= size:
            raise ValueError('invalid asset capsule footer')
        if digest.hex() != expected:
            raise ValueError('capsule does not match this signed extraction request')
        offset = size - FOOTER.size - count
        source.seek(offset)
        hashed = hashlib.sha256()
        left = count
        while left:
            chunk = source.read(min(left, 1024**2))
            if not chunk:
                raise ValueError('truncated capsule')
            hashed.update(chunk); left -= len(chunk)
        if hashed.digest() != digest:
            raise ValueError('asset capsule checksum mismatch')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.capsule-stage-', dir=target) as scratch:
            source.seek(offset)
            total, files = 0, []
            with tarfile.open(fileobj=SliceReader(source, count), mode='r|xz') as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts or '.capsules' in path.parts:
                        raise ValueError('unsafe archive path')
                    if member.isdir():
                        continue
                    if not member.isfile() or not path.parts:
                        raise ValueError('capsules contain regular files only')
                    if path in files:
                        raise ValueError('duplicate archive path')
                    total += member.size
                    if total > MAX_EXTRACTED or len(files) >= MAX_FILES:
                        raise ValueError('asset extraction limit exceeded')
                    dest = Path(scratch).joinpath(*path.parts)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as inp, dest.open('xb') as out:
                        shutil.copyfileobj(inp, out, 1024**2)
                        out.flush(); os.fsync(out.fileno())
                    if dest.stat().st_size != member.size:
                        raise ValueError('truncated archive member')
                    files.append(path)
            if not files:
                raise ValueError('empty asset archive')
            # Files are installed atomically only after the whole archive is
            # checked. The runtime requires every planned capsule marker.
            for path in files:
                dest = target.joinpath(*path.parts)
                if any(p.is_symlink() for p in (dest, *dest.parents)):
                    raise ValueError('refusing a symlink in the asset destination')
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(Path(scratch).joinpath(*path.parts), dest)
            marker.parent.mkdir(exist_ok=True)
            marker.write_text('%d files, %d bytes\n' % (len(files), total))
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    m = sub.add_parser('make'); m.add_argument('binary'); m.add_argument('archive'); m.add_argument('out')
    e = sub.add_parser('extract'); e.add_argument('binary'); e.add_argument('target'); e.add_argument('sha256')
    a = ap.parse_args()
    if a.command == 'make':
        print(make(a.binary, a.archive, a.out))
    else:
        print('asset capsule installed:', extract(a.binary, a.target, a.sha256))
