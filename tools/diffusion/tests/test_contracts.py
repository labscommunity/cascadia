import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from cascadia_diffusion.contracts import decode, encode, parse_json, validate_manifest, verify_bundle

FIXTURES = Path(__file__).resolve().parents[3] / 'crates/cascadia-diffusion/tests/fixtures'

class Contracts(unittest.TestCase):
    def test_rust_shared_fixture_and_truncation(self):
        manifest = parse_json((FIXTURES/'manifest.json').read_bytes())
        mid = validate_manifest(manifest)
        self.assertEqual(mid, (FIXTURES/'manifest.sha256').read_text().strip())
        raw = (FIXTURES/'frame.bin').read_bytes()
        header, tensors = decode(raw)
        self.assertEqual(header['manifest'], mid)
        self.assertEqual(bytes(tensors[0]), struct.pack('<ff', 1., -2.))
        self.assertEqual(encode(header, tensors), raw)
        for end in range(len(raw)):
            with self.assertRaises(ValueError): decode(raw[:end])
        for changed in (raw+b'\x00', raw[:-4]+struct.pack('<f',float('nan')), b'CDF1'+b'\xff'*4+raw[8:]):
            with self.assertRaises(ValueError): decode(changed)

    def test_invalid_schemas(self):
        header, tensors = decode((FIXTURES/'frame.bin').read_bytes())
        for shape in ([0], [2**64,2], [1024]*9, [True]):
            changed = copy.deepcopy(header)
            changed['tensors'][0]['shape'] = shape
            with self.assertRaises(ValueError): encode(changed,tensors)
        with self.assertRaises(ValueError): decode((FIXTURES/'frame.bin').read_bytes(), [])
        with self.assertRaises(ValueError): parse_json('{"a":1,"a":2}')
        with self.assertRaises(ValueError): parse_json('{"a":NaN}')

    def test_bundle_tamper_and_unlisted_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            shutil.copy(FIXTURES/'manifest.json',root/'manifest.json')
            (root/'unet').mkdir()
            (root/'unet/model.xml').write_bytes(b'xml')
            (root/'unet/model.bin').write_bytes(b'bin')
            mid=(FIXTURES/'manifest.sha256').read_text().strip()
            verify_bundle(root,mid,['unet'])
            (root/'tokenizer_config.json').write_text('{}')
            with self.assertRaises(ValueError): verify_bundle(root,mid,['unet'])
            (root/'tokenizer_config.json').unlink()
            (root/'unet/model.bin').write_bytes(b'bad')
            with self.assertRaises(ValueError): verify_bundle(root,mid,['unet'])

if __name__ == '__main__': unittest.main()
