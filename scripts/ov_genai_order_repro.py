#!/usr/bin/env python3
"""Does a greedy answer from plain OpenVINO GenAI depend on the previous request?

No Cascadia code runs here: one openvino_genai.LLMPipeline answers two prompts
in the order A B A A B B A (greedy, 256 new tokens, chat template on), and the
script prints a short sha256 of every answer, the answers grouped per prompt,
and one verdict line:

    REPRO depends-on-previous   some prompt got two different answers
    REPRO stable                every prompt got one answer

Usage:

    python3 scripts/ov_genai_order_repro.py <model-dir> <device> [--cache-dir DIR]
    python3 scripts/ov_genai_order_repro.py <model-dir> GPU --dry-run

--cache-dir sets the pipeline's CACHE_DIR (compiled kernels). --dry-run prints
the plan without loading a model. Needs `pip install openvino-genai` (except
for --dry-run). The pure rules are unit-tested in
scripts/test_ov_genai_order_repro.py.

Only the standard library and openvino_genai; Python 3.8+.
"""

import argparse
import hashlib
import sys

PROMPTS = {
    "A": "Count from 1 to 100,000 in words, separated by commas.",
    "B": "Name the eight planets of the solar system in order from the Sun, with two sentences about each one.",
}

ORDER = "ABAABBA"
MAX_NEW_TOKENS = 256


# --- pure rules ---


def sequence():
    """The prompt keys in the order they are sent."""
    return list(ORDER)


def sha(text):
    """The 12-hex-digit sha256 prefix printed for an answer."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def group(answers):
    """Group (prompt, sha) pairs per prompt, in first-seen order.

    Returns a dict of prompt -> list of shas in the order they came.
    """
    seen = {}
    for p, h in answers:
        seen.setdefault(p, []).append(h)
    return seen


def verdict(answers):
    """Return depends-on-previous if some prompt got two different answers, else stable."""
    seen = group(answers)
    return "depends-on-previous" if any(len(set(v)) > 1 for v in seen.values()) else "stable"


def run_line(i, p, h, chars):
    return "run %d prompt %s sha %s chars %d" % (i, p, h, chars)


def group_lines(answers):
    return ["prompt %s: %s" % (p, " ".join(hs)) for p, hs in group(answers).items()]


def plan(model, device, cache_dir):
    lines = [
        "model %s device %s cache-dir %s" % (model, device, cache_dir or "-"),
        "greedy, max_new_tokens %d, chat template on" % MAX_NEW_TOKENS,
        "order %s" % " ".join(sequence()),
    ]
    for k in sorted(PROMPTS):
        lines.append("prompt %s: %s" % (k, PROMPTS[k]))
    return lines


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Does a greedy OpenVINO GenAI answer depend on the previous prompt?")
    ap.add_argument("model", help="the OpenVINO model directory")
    ap.add_argument("device", help="CPU, GPU, NPU, ...")
    ap.add_argument("--cache-dir", help="the pipeline's CACHE_DIR")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without loading a model")
    return ap.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.dry_run:
        for line in plan(args.model, args.device, args.cache_dir):
            print(line)
        return 0

    import openvino_genai as ov_genai

    props = {"CACHE_DIR": args.cache_dir} if args.cache_dir else {}
    pipe = ov_genai.LLMPipeline(args.model, args.device, **props)
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW_TOKENS
    cfg.do_sample = False
    cfg.apply_chat_template = True
    print("openvino_genai %s device %s model %s" % (ov_genai.__version__, args.device, args.model), flush=True)
    answers = []
    for i, p in enumerate(sequence()):
        text = str(pipe.generate(PROMPTS[p], cfg))
        h = sha(text)
        answers.append((p, h))
        print(run_line(i, p, h, len(text)), flush=True)
    for line in group_lines(answers):
        print(line)
    print("REPRO " + verdict(answers))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
