"""Benchmark openvino_genai.LLMPipeline against the optimum-intel path.

Usage (on the worker node):
    python bench_llmpipeline.py --model <dir> --device GPU --max-tokens 64

Emits a final JSON line like:
    RESULT={"tok_s": 14.21, "decode_s": 4.503, "tokens": 64, ...}

We measure decode-only tok/s (excludes prompt prefill + warmup).
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="GPU")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--kv-precision", default="")  # e.g. "u8"
    p.add_argument("--dyn-quant-group", default="")  # e.g. "32"
    args = p.parse_args()

    # Force HF offline so transformers/optimum don't ping the hub at startup.
    os.environ["HF_HUB_OFFLINE"] = "1"

    import openvino_genai as ov_genai

    plugin_config: dict[str, str] = {}
    if args.cache_dir:
        plugin_config["CACHE_DIR"] = args.cache_dir
    if args.kv_precision:
        plugin_config["KV_CACHE_PRECISION"] = args.kv_precision
    if args.dyn_quant_group:
        plugin_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = args.dyn_quant_group

    t_load_start = time.time()
    pipe = ov_genai.LLMPipeline(args.model, args.device, **plugin_config)
    t_load_end = time.time()

    # Warmup: 4 tokens.
    cfg_warm = ov_genai.GenerationConfig()
    cfg_warm.max_new_tokens = 4
    cfg_warm.do_sample = False
    pipe.generate(args.prompt, cfg_warm)
    t_warm_end = time.time()

    # Real run.
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.max_tokens
    cfg.do_sample = False
    t_gen_start = time.time()
    res = pipe.generate(args.prompt, cfg)
    t_gen_end = time.time()

    # GenAI returns a DecodedResults object with .perf_metrics + .texts.
    text = str(res)
    decode_s = t_gen_end - t_gen_start
    # Generated token count: derive from perf_metrics if available, else
    # assume max_new_tokens (greedy without EOS won't terminate early on this prompt).
    tokens = args.max_tokens
    perf_blob: dict = {}
    metrics = getattr(res, "perf_metrics", None)
    if metrics is not None:
        # Newer GenAI versions expose num_generated_tokens, mean_tok_per_sec, etc.
        for attr in (
            "num_generated_tokens", "mean_tok_per_sec",
            "num_input_tokens", "load_time", "ttft",
            "throughput", "mean_decode_latency",
        ):
            v = getattr(metrics, attr, None)
            if v is None:
                continue
            try:
                # Some return a Mean()/Stddev() pair object.
                perf_blob[attr] = {"mean": v.mean, "std": getattr(v, "std", None)}
            except AttributeError:
                perf_blob[attr] = float(v) if not isinstance(v, (int, float)) else v

    if perf_blob.get("num_generated_tokens"):
        tokens = int(perf_blob["num_generated_tokens"]["mean"]
                     if isinstance(perf_blob["num_generated_tokens"], dict)
                     else perf_blob["num_generated_tokens"])

    tok_s = tokens / decode_s if decode_s else None

    out = {
        "model": args.model, "device": args.device, "tokens": tokens,
        "decode_s": decode_s, "tok_s": tok_s,
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
