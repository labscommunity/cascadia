"""Pipeline.generate request validation, checked before any inference runs."""
import sys
import unittest
from unittest import mock

from cascadia_diffusion.runtime import Pipeline


class Generate(unittest.TestCase):
    def setUp(self):
        # Construction needs the inference stack; validation needs only the
        # manifest, so build the object bare and fail loudly if it infers.
        self.pipeline=object.__new__(Pipeline)
        self.pipeline.manifest={'width':512,'height':512}
        self.pipeline.pipe=mock.Mock(side_effect=AssertionError('validation let the request through'))
        modules=mock.patch.dict(sys.modules,{'torch':mock.MagicMock()})
        modules.start()
        self.addCleanup(modules.stop)

    def test_rejects_invalid_request_fields(self):
        cases=[
            ([], 'invalid generation fields'),
            ({'prompt':'x','seed':1}, 'invalid generation fields'),
            ({}, 'invalid prompt'),
            ({'prompt':'   '}, 'invalid prompt'),
            ({'prompt':'x'*16385}, 'invalid prompt'),
            ({'prompt':'x','negative_prompt':1}, 'invalid negative prompt'),
            ({'prompt':'x','negative_prompt':'x'*16385}, 'invalid negative prompt'),
            ({'prompt':'x','num_inference_steps':0}, 'integer out of range'),
            ({'prompt':'x','num_inference_steps':51}, 'integer out of range'),
            ({'prompt':'x','num_inference_steps':True}, 'integer out of range'),
            ({'prompt':'x','rng_seed':-1}, 'integer out of range'),
            ({'prompt':'x','rng_seed':2**32}, 'integer out of range'),
            ({'prompt':'x','guidance_scale':1}, 'SDXL fixed batch requires guidance in (1,30]'),
            ({'prompt':'x','guidance_scale':30.5}, 'SDXL fixed batch requires guidance in (1,30]'),
            ({'prompt':'x','guidance_scale':'7'}, 'SDXL fixed batch requires guidance in (1,30]'),
            ({'prompt':'x','size':'1024x1024'}, 'size differs from artifact'),
            ({'prompt':'x','n':2}, 'one base64 PNG is supported'),
            ({'prompt':'x','response_format':'url'}, 'one base64 PNG is supported'),
        ]
        for body,message in cases:
            with self.subTest(body=body):
                with self.assertRaises(ValueError) as caught:self.pipeline.generate(body)
                self.assertEqual(str(caught.exception),message)
        self.pipeline.pipe.assert_not_called()
        # A valid request reaches inference, so the cases above failed on their field.
        with self.assertRaises(AssertionError):self.pipeline.generate({'prompt':'x','negative_prompt':'','size':'512x512','n':1,'num_inference_steps':20,'guidance_scale':7.5,'rng_seed':0,'response_format':'b64_json'})


if __name__=='__main__':unittest.main()
