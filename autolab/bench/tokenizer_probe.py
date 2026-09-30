#!/usr/bin/env python3
"""Does the fleet's tokenizer agree with o200k_base on ordinary text? (decides whether a gpt-oss class model could draft)

Sends raw /v1/completions requests with max_tokens=1 and compares usage.prompt_tokens with tiktoken's count.
Run with a Python that has tiktoken (scratchpad venv). Only when the fleet is idle: each probe is one prefill.
"""
import json, sys, urllib.request
import tiktoken

API = "http://localhost:18000"
enc = tiktoken.get_encoding("o200k_base")
TESTS = ["Hello world", "The quick brown fox jumps over the lazy dog.",
         "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
         "Überraschung! 日本語のテキスト 12345.6789 %$#@", "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
         "SELECT id, name FROM users WHERE created_at > '2026-01-01' ORDER BY id DESC LIMIT 10;",
         "   leading spaces\tand\ttabs\n\n\nand blank lines   ", "🙂🚀 emoji and ∑∫√ math symbols ≤ ≥ ≠"]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
same = 0
for t in TESTS:
    body = json.dumps({"model": "inkling", "prompt": t, "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(API + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=300) as r:
        v = json.loads(r.read().decode())
    fleet, ref = v["usage"]["prompt_tokens"], len(enc.encode(t))
    same += fleet == ref
    print("%-3s fleet %3d  o200k %3d  %r" % ("ok" if fleet == ref else "DIFF", fleet, ref, t[:50]))
print("%d of %d counts agree" % (same, len(TESTS)))
