import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from cascadia_diffusion.contracts import decode, encode, parse_json, validate_manifest, validate_sdxl_metadata, verify_bundle

FIXTURES = Path(__file__).resolve().parents[3] / 'crates/cascadia-diffusion/tests/fixtures'

class Contracts(unittest.TestCase):
    def test_unsupported_pipeline_settings_fail_before_export_or_engine_import(self):
        from cascadia_diffusion.export import export_sdxl
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = {
                '_class_name': 'StableDiffusionXLPipeline',
                'scheduler': ['diffusers', 'EulerDiscreteScheduler'],
                'tokenizer': ['transformers', 'CLIPTokenizer'],
                'tokenizer_2': ['transformers', 'CLIPTokenizer'],
            }
            for name in ('tokenizer', 'tokenizer_2'):
                (root/name).mkdir()
                (root/name/'tokenizer_config.json').write_text('{"model_max_length":77}')
            (root/'scheduler').mkdir()
            (root/'model_index.json').write_text(json.dumps(index))
            scheduler = root/'scheduler/scheduler_config.json'
            scheduler.write_text('{"_class_name":"EulerDiscreteScheduler"}')
            validate_sdxl_metadata(root)
            for changed in (index | {'force_zeros_for_empty_prompt': False},
                            index | {'requires_aesthetics_score': True},
                            index | {'tokenizer': ['transformers', 'CLIPTokenizerFast']}):
                (root/'model_index.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    validate_sdxl_metadata(root)
            (root/'model_index.json').write_text(json.dumps(index))
            scheduler.write_text('{"_class_name":"DPMSolverMultistepScheduler"}')
            with self.assertRaisesRegex(ValueError, 'scheduler substitution'):
                export_sdxl(root, root/'output', 'a'*40, 512)
            self.assertFalse((root/'output').exists())

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
