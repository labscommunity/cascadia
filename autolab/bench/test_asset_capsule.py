import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from asset_capsule import make, extract, FOOTER


class CapsuleTests(unittest.TestCase):
    def capsule(self, root, name='model/weights.bin', link=False):
        binary, archive, out = (root / n for n in ('binary', 'archive.tar.xz', 'capsule'))
        binary.write_bytes(b'\x7fELF\x02' + b'fixture executable' * 10)
        with tarfile.open(archive, 'w:xz') as tar:
            entry = tarfile.TarInfo(name)
            if link:
                entry.type = tarfile.SYMTYPE; entry.linkname = '/etc/passwd'
                tar.addfile(entry)
            else:
                entry.size = 4; tar.addfile(entry, io.BytesIO(b'data'))
        return out, make(binary, archive, out)

    def test_checked_install_and_idempotent_plain_binary_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, digest = self.capsule(root)
            target = root / 'installed'
            self.assertTrue(extract(source, target, digest))
            self.assertEqual((target/'model/weights.bin').read_bytes(), b'data')
            # Later ordinary binaries need not contain the already-installed capsule.
            self.assertFalse(extract(root/'binary', target, digest))
            self.assertEqual(digest, hashlib.sha256((root/'archive.tar.xz').read_bytes()).hexdigest())

    def test_checksum_and_expected_release_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, digest = self.capsule(root)
            with self.assertRaisesRegex(ValueError, 'signed extraction'):
                extract(source, root/'installed', '0'*64)
            data = bytearray(source.read_bytes()); data[-FOOTER.size-1] ^= 1
            source.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                extract(source, root/'installed', digest)
            self.assertFalse((root/'installed').exists())

    def test_archive_paths_and_links_cannot_escape(self):
        for name, link in [('../outside', False), ('/outside', False), ('model/link', True), ('.capsules/fake', False)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source, digest = self.capsule(root, name, link)
                with self.assertRaises(ValueError):
                    extract(source, root/'installed', digest)
                self.assertFalse((root/'outside').exists())


if __name__ == '__main__':
    unittest.main()
