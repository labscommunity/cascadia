#!/usr/bin/env python3
"""OpenAI-compatibility checks with the unmodified `openai` Python SDK.

Cascadia promises OpenAI compatibility, and the SDK is the client most callers
use, so these checks drive a real `cascadia` binary (mock engine, no model
files) through `openai.OpenAI` rather than raw HTTP:

  * /v1/models lists the served id, with `input_modalities` ["text"]
  * chat with string content and with a content-part array
  * streaming deltas and a final finish_reason
  * errors raise the SDK's own classes with OpenAI's
    {"error": {message, type, param, code}} body:
      - image_url sent to a text-only model -> BadRequestError
      - an unsupported content part -> BadRequestError, reason only
      - legacy completions with a prompt list -> BadRequestError

Run:  python3 tests/conformance/test_openai_sdk.py --bin target/debug/cascadia
Requires the pinned SDK: pip install -r tests/conformance/requirements-openai-sdk.txt
"""
import argparse
import socket
import subprocess
import sys
import time
import unittest
import urllib.request

import openai

BIN = "target/debug/cascadia"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OpenAISdkConformance(unittest.TestCase):
    proc = None
    client = None

    @classmethod
    def setUpClass(cls):
        port = free_port()
        cls.proc = subprocess.Popen(
            [BIN, "worker", "--rank", "0", "--total", "1", "--engine", "mock",
             "--model", "mock-model", "--api", f"127.0.0.1:{port}", "--log-level", "warn"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError(f"cascadia exited early with {cls.proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            cls.proc.kill()
            raise RuntimeError("cascadia never became healthy")
        cls.client = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1",
                                   api_key="unused", max_retries=0, timeout=60)

    @classmethod
    def tearDownClass(cls):
        if cls.proc:
            cls.proc.kill()
            cls.proc.wait()

    def chat(self, content, **kw):
        return self.client.chat.completions.create(
            model="mock-model", max_tokens=8,
            messages=[{"role": "user", "content": content}], **kw)

    def assert_bad_request(self, call, message):
        with self.assertRaises(openai.BadRequestError) as ctx:
            call()
        err = ctx.exception
        self.assertEqual(err.status_code, 400)
        self.assertEqual(err.body["type"], "invalid_request_error")
        self.assertIsNone(err.body["param"])
        self.assertIn(message, err.body["message"])
        return err.body["message"]

    def test_models_list(self):
        models = self.client.models.list().data
        self.assertEqual([m.id for m in models], ["mock-model"])
        self.assertEqual(models[0].model_extra.get("input_modalities"), ["text"])

    def test_string_and_array_content(self):
        for content in ("hello there friend",
                        [{"type": "text", "text": "hello"}, {"type": "text", "text": "there"}]):
            r = self.chat(content)
            self.assertEqual(r.object, "chat.completion")
            self.assertEqual(r.choices[0].message.role, "assistant")
            self.assertTrue(r.choices[0].message.content)
            self.assertGreater(r.usage.completion_tokens, 0)

    def test_streaming(self):
        chunks = list(self.chat("one two three four", stream=True))
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        self.assertTrue(text)
        finishes = [c.choices[0].finish_reason for c in chunks if c.choices]
        self.assertIn(finishes[-1], ("stop", "length"))

    def test_image_on_text_only_model_is_bad_request(self):
        self.assert_bad_request(
            lambda: self.chat([{"type": "text", "text": "what is this?"},
                               {"type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]),
            "does not accept image input")

    def test_unsupported_part_reports_reason_only(self):
        msg = self.assert_bad_request(
            lambda: self.chat([{"type": "input_audio",
                                "input_audio": {"data": "", "format": "wav"}}]),
            "content part type `input_audio` is not supported")
        self.assertNotIn("Failed to deserialize", msg)
        self.assertNotIn(" at line ", msg)

    def test_legacy_completions_prompt_list(self):
        self.assert_bad_request(
            lambda: self.client.completions.create(model="mock-model", prompt=["a", "b"],
                                                   max_tokens=4),
            "multi-prompt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=BIN, help="cascadia binary (default: %(default)s)")
    args, rest = ap.parse_known_args()
    BIN = args.bin
    print(f"openai SDK {openai.__version__}, binary {BIN}")
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
