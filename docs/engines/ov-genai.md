# `ov-genai` — single-stage `openvino_genai.LLMPipeline`

Wraps Intel's `openvino_genai.LLMPipeline` to expose **FastDraft speculative decode** and **Prompt Lookup decoding**. Both are mathematically lossless under greedy decoding.

## When to use

- Single-node Intel deployment (Battlemage, Lunar Lake, Arrow Lake, Panther Lake).
- You want speculative decoding via Intel's prebuilt FastDraft companion models, **or** you have an extractive workload (RAG, summarisation, code-in-context) where Prompt Lookup wins.
- For multi-stage pipeline parallelism, use `ov-runtime` or `ov-dist-spec` instead — `LLMPipeline` is single-stage only.

## Shard format

A single-directory OpenVINO IR: the model graph (`openvino_model.xml`/`.bin`, or `openvino_language_model.xml` for the VLM-style exports of Qwen3.5/3.6 and Gemma 4) **plus** `openvino_tokenizer.xml` and `openvino_detokenizer.xml`. `ov::genai::LLMPipeline` loads the tokenizer as an OpenVINO model, so an IR missing those two starts and then generates empty strings (`completion_tokens: 0`) rather than failing loudly.

`cascadia shard` does not produce this layout — it emits per-stage IRs for `ov-runtime`. Either download a pre-exported IR (Intel publishes INT4 IRs for many models under the [OpenVINO](https://huggingface.co/OpenVINO) org) or build one with Intel's exporter:

```bash
# The [openvino] extra pulls openvino, nncf, AND openvino-tokenizers — without
# openvino-tokenizers the export silently omits the two tokenizer IRs.
pip install "optimum-intel[openvino]"

optimum-cli export openvino \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --weight-format int4 \
    --task text-generation-with-past \
    /models/llama-3.1-8b-int4-ov
```

(For draft models, prefer Intel's prebuilt FastDraft IRs from HF — e.g. `OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov` — they're trained for the corresponding target family.)

## Examples

### Plain LLMPipeline

```bash
cascadia worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model /models/llama-3.1-8b-int4-ov \
              --ov-cache-dir ~/.cache/cascadia/ov_kernel_cache \
              --api :8000
```

### FastDraft speculative decode (short-input chat)

```bash
cascadia worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model /models/llama-3.1-8b-int4-ov \
              --draft-model /models/llama-3.1-fastdraft-150m-int8-ov \
              --spec-k 5 \
              --ov-cache-dir ~/.cache/cascadia/ov_kernel_cache \
              --api :8000
```

### Prompt Lookup decoding (extractive workloads)

```bash
cascadia worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model /models/llama-3.1-8b-int4-ov \
              --prompt-lookup 3 \
              --ov-cache-dir ~/.cache/cascadia/ov_kernel_cache \
              --api :8000
```

## Workload guidance

| Workload                                | Recommended config                          |
|-----------------------------------------|---------------------------------------------|
| Short factual chat (<100 tok output)    | `--draft-model FASTDRAFT --spec-k 5`        |
| Long creative writing (256+ tok)        | `--draft-model FASTDRAFT --spec-k 3`        |
| Extractive RAG / summarisation          | `--prompt-lookup 3`                         |
| Open-ended QA over long context         | plain (no spec)                             |

## Picking K (FastDraft)

Measured on Arc B390 with Llama-3.1-8B INT4 + Intel's FastDraft 150M INT8:

| K | tok/s | accept |
|---|-------|--------|
| 3 | 30.8  | 0.88   |
| **5** | **35.1** | **0.86** |
| 7 | 32.4  | 0.78   |

Sweet spot is K=5 for short factual prompts; drop to K=3 for long-creative outputs (acceptance falls as the draft drifts).

## Plugin tuning

- `--ov-cache-dir <path>` — persists kernel JIT compile results across runs. Cuts cold-start by ~62% on second+ launches. **Recommended for any non-throwaway deployment.**
- `--ov-kv-precision {u8,f16}` and `--ov-dyn-quant-group <N>` — exposed for debugging only; defaults are already optimal on Battlemage / Lunar Lake.

## Continuous batching (`--cb`, #20)

`--cb` swaps the `LLMPipeline` for ov-genai's `ContinuousBatchingPipeline`:
concurrent requests join one paged-attention batch, each `step()` advances the
batch by one scheduler iteration, and each request streams incremental text
deltas (unlike the default engine's one-chunk-per-task). Which requests run in
a given iteration is the scheduler's choice, bounded by `--cb-max-num-seqs` /
`--cb-max-batched-tokens` and KV-cache pressure.
`cancel()` aborts a single request mid-generation without touching the rest
of the batch. Tune with `--cb-cache-size`, `--cb-max-num-seqs`,
`--cb-max-batched-tokens`, `--cb-dynamic-split-fuse`, `--cb-prefix-caching`
(zeros/unset keep ov-genai defaults).

Device note: paged attention is a CPU/GPU-plugin capability. On NPU,
ov-genai serves the static NPUW pipeline — `--cb` will fail at compile
there; run NPU workers without `--cb` (requests queue sequentially).

```bash
cascadia worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model /models/qwen3-8b-int4-ov \
              --cb --cb-cache-size 4 --cb-max-num-seqs 32 \
              --cb-prefix-caching true \
              --api :8000
```

### When `--cb` helps — and when it hurts

**`--cb` is not a free win. Measure before enabling it.** It trades
single-request latency for batch throughput, and on some shapes that trade is
very bad. Aggregate tokens/s, Phi-3.5-mini-int4 and Qwen3-8B-int4 on a Lunar
Lake box (Arc 140V iGPU + CPU), `max_tokens=64`, warm kernel cache:

| workload | device | without `--cb` | with `--cb` |
|---|---|---|---|
| short prompt, 1 request | GPU | 39.9 | 36.9 (**0.92×**) |
| short prompt, 8 concurrent | GPU | 39.8 | 141.8 (**3.6×**) |
| short prompt, 16 concurrent | GPU | 36.2 | 176.3 (**4.9×**) |
| short prompt, 8 concurrent | CPU | 22.7 | 36.8 (1.6×) |
| ~1200-token prompt, 8 concurrent | GPU | 33.4 | 20.7 (**0.6×**) |
| ~1200-token prompt, 4 concurrent | CPU | 15.9 | 3.3 (**0.2×**) |

Three rules follow:

- **Enable it for short-prompt concurrent serving, preferably on GPU.** That is
  where the 3–5× lives. The win grows with concurrency and needs at least 4
  in-flight requests to appear at all.
- **Do not enable it for a single-user worker.** Every configuration measured
  costs 8–15% at concurrency 1 — paged-attention bookkeeping with no batch to
  amortise it over.
- **Do not enable it on CPU for long-context workloads.** A ~1200-token prompt
  collapses to ~0.2× — a five-fold throughput loss, reproduced on both models,
  and *not* recoverable by raising `--cb-max-batched-tokens`. RAG-style traffic
  on a CPU worker is the worst case for this flag.

On GPU, long prompts are model-dependent rather than uniformly bad: Qwen3-8B
gained (1.6× at 8 concurrent) where Phi-3.5-mini lost (0.6×). If your prompts
are long, benchmark your own model. Raising `--cb-max-batched-tokens` above the
prompt length recovers much of the GPU loss (Phi-3.5-mini went 0.6× → 1.1× at 4
concurrent) because it stops splitting the prefill; it does nothing on CPU.

## Limitations

- **Single-stage only.** No pipeline parallelism (`--total 1` enforced).
- **Streaming**: the default path yields one chunk per task with `is_final=True`; `--cb` streams per-iteration text deltas.
- **Draft / target tokeniser must match.** FastDraft companions are trained per target family; mixing across families won't work.
- **`--draft-model` and `--prompt-lookup` are mutually exclusive.** Both set `GenerationConfig.num_assistant_tokens`; the validator rejects the combination.
- **`--cb` is incompatible with `--draft-model` / `--prompt-lookup` and with VLM-layout exports.** The CB scheduler owns batch composition; speculative CB is a follow-up.
- **`--cb` is a throughput/latency trade, not a strict upgrade.** It costs ~10% at concurrency 1 and up to 5x on CPU with long prompts — see [When `--cb` helps](#when---cb-helps--and-when-it-hurts).
