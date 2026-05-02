"""LLMPipeline + speculative decoding bench.

Usage:
    python bench_llmpipe_spec.py --model <dir> --draft <dir> [--k 5] [--threshold 0.4]
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--device", default="GPU")
    p.add_argument("--draft-device", default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--k", type=int, default=5, help="num_assistant_tokens")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="assistant_confidence_threshold; 0 disables")
    p.add_argument("--cache-dir", default="")
    args = p.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    import openvino_genai as ov_genai

    plugin_config: dict[str, str] = {}
    if args.cache_dir:
        plugin_config["CACHE_DIR"] = args.cache_dir

    draft_dev = args.draft_device or args.device
    t_load_start = time.time()
    draft = ov_genai.draft_model(args.draft, draft_dev)
    pipe = ov_genai.LLMPipeline(
        args.model, args.device, draft_model=draft, **plugin_config,
    )
    t_load_end = time.time()

    cfg_warm = ov_genai.GenerationConfig()
    cfg_warm.max_new_tokens = 4
    cfg_warm.do_sample = False
    cfg_warm.num_assistant_tokens = args.k
    pipe.generate(args.prompt, cfg_warm)
    t_warm_end = time.time()

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.max_tokens
    cfg.do_sample = False
    cfg.num_assistant_tokens = args.k
    if args.threshold > 0:
        cfg.assistant_confidence_threshold = args.threshold

    t_gen_start = time.time()
    res = pipe.generate(args.prompt, cfg)
    t_gen_end = time.time()

    text = str(res)
    decode_s = t_gen_end - t_gen_start
    tokens = args.max_tokens
    perf_blob: dict = {}
    metrics = getattr(res, "perf_metrics", None)
    if metrics is not None:
        for attr in (
            "num_generated_tokens", "mean_tok_per_sec",
            "num_input_tokens", "load_time", "ttft",
            "throughput", "mean_decode_latency",
        ):
            v = getattr(metrics, attr, None)
            if v is None:
                continue
            try:
                perf_blob[attr] = {"mean": v.mean, "std": getattr(v, "std", None)}
            except AttributeError:
                perf_blob[attr] = float(v) if not isinstance(v, (int, float)) else v
    if perf_blob.get("num_generated_tokens"):
        v = perf_blob["num_generated_tokens"]
        tokens = int(v["mean"] if isinstance(v, dict) else v)

    tok_s = tokens / decode_s if decode_s else None

    out = {
        "model": args.model, "draft": args.draft, "device": args.device,
        "k": args.k, "threshold": args.threshold,
        "tokens": tokens, "decode_s": decode_s, "tok_s": tok_s,
        "load_s": t_load_end - t_load_start,
        "warmup_s": t_warm_end - t_load_end,
        "plugin_config": plugin_config,
        "perf_metrics": perf_blob,
        "first_text": text[:80],
    }
    print(f"RESULT={json.dumps(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
