# optimum-cli export openvino: complete flag map (2024.01 → 2026.04)

**Released:** Current as of optimum-intel v1.27.0 (2025-12) — most flags landed v1.13–v1.23.
**What changed:** The CLI is the canonical entry point: it converts a HF model to OpenVINO IR, with quantization choices baked in. Below is the flag inventory most relevant to GPU LLM perf — extracted from the live `--help`.

**Headline perf claim (if any):** N/A — these are the exact knobs that drive every other experiment.

**How to use it from optimum-intel / OV runtime:**

### Tasks (--task)
For decoder LLMs, *always* use the `-with-past` variant unless you have a reason not to:
- `text-generation-with-past`     ← LLM with KV cache (default for decoder models, stateful)
- `text-generation`               ← LLM without past (stateless, slower)
- `text2text-generation-with-past`
- `automatic-speech-recognition-with-past`
The `-with-past` flag means past keys/values are reused → no recomputation per token. With `--disable-stateful` you also separate KV from the model graph (worse perf, only use for legacy code).

### Weight formats (--weight-format)
`fp32 | fp16 | int8 | int4 | mxfp4 | nf4 | cb4`
- `int8` is the default for models > 1B params.
- `int4` enables INT4 weight-only — typical 3-4× memory reduction, often <1pp accuracy hit.
- `cb4` is codebook 4-bit with 16 fp8 (E4M3) values — newest format from 2026.0.

### Group size (--group-size)
**Recommended: 128.** Default in optimum is `-1` if not specified, but the typical CLI invocation `--weight-format int4` resolves a per-model default config (often 128, sometimes 64 for smaller models). `-1` means per-column (largest, best accuracy but lowest speed and worst memory). Group sizes 32 / 64 / 128 trade more accuracy per smaller scale-block for slightly more storage. Group=32 → finest scales, ~5-10% extra storage vs g128, often <0.5pp better accuracy. Group=128 → standard sweet spot. The `--group-size-fallback` (`error|ignore|adjust`) was added later to handle nodes that can't honor the chosen group.

### Mixed precision (--ratio)
A value <1.0 keeps part of the model in INT8 to recover accuracy. `--ratio 0.8` = 80% INT4 + 20% INT8. Only takes effect with `--dataset` for data-aware mixed assignment. Default 1.0 (all INT4).

### Data-aware compression algorithms (need `--dataset`)
- `--awq` — Activation-aware Weight Quantization. Improves INT4 quality. With dataset: data-aware AWQ (slower). Without: data-free.
- `--scale-estimation` — minimizes L2 error between original and compressed (needs dataset).
- `--gptq` — layer-wise activation-minimization (needs dataset, takes more time).
- `--lora-correction` — adds low-rank adapters to recover accuracy at small inference cost.
- `--smooth-quant-alpha` — for INT8 quant of activations (--quant-mode).
- `--sensitivity-metric` — picks which layers stay INT8 in mixed mode: `weight_quantization_error | hessian_input_activation | mean_activation_variance | max_activation_variance | mean_activation_magnitude`.
- `--all-layers` — also compress embeddings + last MatMul (default keeps them INT8).
- `--backup-precision` — `none | int8_sym | int8_asym` for unsupported nodes (default int8_asym).

### Full quantization (activations + weights) (--quant-mode)
`int8 | f8e4m3 | f8e5m2 | cb4_f8e4m3 | int4_f8e4m3 | int4_f8e5m2`
Activates SmoothQuant for INT8. New cb4_f8e4m3 / int4_f8e4m3 came in v1.23 (May 2025).

### Stateful (--disable-stateful)
DON'T pass it for LLMs. Stateful models hide KV cache inside the model graph → much faster GPU exec.

### NO --paged-attention flag exists
Surveyed exhaustively — there is no `--paged-attention` CLI option. Paged attention is enabled at *runtime* by the GPU plugin (default in OV ≥2025.1) and via `SchedulerConfig` in GenAI `LLMPipeline`. Stateful models export → SDPA → SDPAToPagedAttention transformation runs at compile time on GPU device.

### Default int4 configs per model
optimum-intel ships `optimum/intel/openvino/configuration.py` with per-model defaults — e.g. Llama-3.1-8B uses g=128, ratio=1.0, sym=False; Phi-4-mini uses g=64; Llama-3.2-1B uses g=64; Qwen3-4B/-8B has its own defaults. Just running `--weight-format int4` (no other flag) hits this map.

**Intel GPU applicability:** HIGH for everything. These flags determine what you ship to alpha and charlie.
**Open hypothesis it generates for us:** Export Llama-3-8B 6 ways: g=32, g=64, g=128 each with `--awq`. Measure (a) export time, (b) file size, (c) tokens/sec on B390, (d) lm-eval suite. Hypothesis: g=64+AWQ wins the Pareto: <2% perplexity loss vs g=32, ~5% smaller, ~1.05x faster decode.

Sources:
- https://huggingface.co/docs/optimum/main/en/intel/openvino/export
- https://github.com/huggingface/optimum-intel/blob/main/docs/source/openvino/export.mdx
- https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/configuration.py
