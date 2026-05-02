"""Bench LLMPipeline with SchedulerConfig (continuous batching, prefix caching).

Two-turn chat: same system prompt, different user questions. Turn 2 should
be much faster on TTFT if prefix caching is engaged.
"""
from __future__ import annotations
import argparse, json, os, time

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="GPU")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prefix-caching", action="store_true")
    args = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    import openvino_genai as ov_genai

    sched = ov_genai.SchedulerConfig()
    sched.enable_prefix_caching = bool(args.prefix_caching)
    sched.dynamic_split_fuse = True
    sched.cache_size = 4
    sched.max_num_batched_tokens = 4096

    t0 = time.time()
    pipe = ov_genai.LLMPipeline(args.model, args.device, scheduler_config=sched)
    t_load = time.time() - t0

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.max_tokens
    cfg.do_sample = False

    SYSTEM = ("You are a helpful and concise assistant. Always answer in 2 sentences. "
              "Be polite. Cite a source if relevant. Do not invent facts. "
              "If you don't know, say so. Use plain language. ") * 4
    Q1 = "What is the capital of France?"
    Q2 = "What is the largest mountain?"

    pipe.generate("Hi", ov_genai.GenerationConfig())  # warmup

    def turn(prompt: str):
        t = time.time()
        r = pipe.generate(SYSTEM + "\n\n" + prompt, cfg)
        return time.time() - t, str(r)

    d1, t1 = turn(Q1)
    d2, t2 = turn(Q2)
    print(f"RESULT={json.dumps({'load_s': t_load, 'turn1_s': d1, 'turn2_s': d2, 'prefix_caching': args.prefix_caching, 'system_chars': len(SYSTEM)})}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
