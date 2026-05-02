# c1: LLMPipeline (openvino_genai) beats OVModelForCausalLM

**Campaign:** c1-llmpipeline

**Hypothesis:** `openvino_genai.LLMPipeline` is 1.4-2.0× faster than `optimum.intel.OVModelForCausalLM` on the same Llama 3.1 8B INT4 model on Intel GPUs. The synthesis (`literature/_intel_synthesis.md` #1) cites this as the single highest-leverage knob: GenAI bundles SchedulerConfig (continuous batching), prefix caching, KV eviction, and is what Intel actively perf-tunes.

**Falsification:** if LLMPipeline yields ≤1.05× over the c0-1b baseline of 8.89 tok/s on alpha, reject. (1.05× is within run-to-run noise.)

**Predicted outcome:** 13-18 tok/s on alpha, 14-20 tok/s on charlie.

**Comparison baselines:**
- c0-1b: alpha B390 ov-optimum 8.89 tok/s
- c0-2:  charlie 140V ov-optimum 10.33 tok/s

**Setup variables to hold constant:** same INT4 model dir, same prompt, same max_tokens (64), same node, greedy decoding (no sampling).

**Risk:**
- LLMPipeline expects a directory containing the OV IR + tokenizer; our cached model dirs are optimum-cli outputs which already have that layout.
- If openvino_genai version is too old for the model, may need `convert_tokenizer` step.

**Plan:**
1. Run `bench_llmpipeline.py` on alpha with `--device GPU --model C:\cascadia\models\llama-3.1-8b-int4`.
2. Same on charlie.
3. Compare `tok_s` to c0-1b and c0-2.
4. If big win, ablate scheduler config / cache_dir / kv_cache_precision in c1.x sub-experiments.
