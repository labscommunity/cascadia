# `ov-genai` — single-stage `openvino_genai.LLMPipeline`

Wraps Intel's `openvino_genai.LLMPipeline` to expose **FastDraft speculative decode** and **Prompt Lookup decoding**. Both are mathematically lossless under greedy decoding, which a request gets
with `temperature: 0`. A request that omits `temperature` gets the server
default (the model's `generation_config.json`, else 1), so it samples; start
the worker with `--default-temperature 0` to keep greedy decoding by default
(see [Request defaults](../CLI.md#request-defaults)).

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

When the batch needs more KV cache than `--cb-cache-size` holds, OpenVINO
drops a request. A request that had already produced output keeps it: it ends
with `finish_reason: length` and its text, like an answer that hit its limit
(it used to be an HTTP 503 or an SSE error event, which clients retry; a
`length` answer is not retried). A request dropped before it produced anything
still fails. Both log a warning naming the knobs, and both count in
`cascadia_kv_cache_exhausted_total{outcome="length"|"error"}` (see
[METRICS.md](../METRICS.md)). A rising count means the cache is too small for
`--cb-max-num-seqs` and the answers asked for: raise `--cb-cache-size` or
lower `--cb-max-num-seqs`.

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

## Checking streaming on hardware

Two checks cover the streamed default path on a real device; neither runs in CI.

**Engine parity test.** Runs the engine itself on a real IR, once with streaming
off (twice, to see whether the device repeats itself at temperature 0) and once
with it on, over six prompts (ASCII, code, CJK, emoji, a counting list, an
answer that starts with whitespace), greedy, 256 new tokens. It fails when the
streamed text plus the final chunk differs from the non-streamed answer by a
single byte, when the engine logs `ov_genai_stream_divergence`, or when there
is less than one delta per 8 tokens. It prints each prompt's first delta,
total time, tok/s and the engine's `ttft_ms`.

```bash
CASCADIA_STREAM_PARITY_MODEL=./phi-3.5-mini-int4-ov CASCADIA_STREAM_PARITY_DEVICE=GPU \
  cargo test -p cascadia-engine-openvino --features openvino \
  --test stateful_stream_parity -- --nocapture
```

It skips when `CASCADIA_STREAM_PARITY_MODEL` is unset. `CASCADIA_STREAM_PARITY_DEVICE`
defaults to `GPU`; `CASCADIA_STREAM_PARITY_DRAFT` (with `CASCADIA_STREAM_PARITY_SPEC_K`)
or `CASCADIA_STREAM_PARITY_PROMPT_LOOKUP` test a speculative pipeline, and then
`CASCADIA_STREAM_PARITY_TOKENS_PER_DELTA` should be k+1.

**Live check against a running worker.** `scripts/ov-genai-stream-check.sh` needs
only bash, curl and python3 and talks to the worker's OpenAI endpoint. Run it on
the worker's own machine over loopback for the timing rows to mean anything:

```bash
cascadia worker --rank 0 --total 1 --engine ov-genai --device GPU \
              --model ./phi-3.5-mini-int4-ov --api 127.0.0.1:8000 > node.log 2>&1 &
scripts/ov-genai-stream-check.sh --url http://127.0.0.1:8000 --node-log node.log
```

Each row prints PASS or FAIL, and the run ends with
`OV-GENAI STREAM CHECK: ALL PASS` (exit 0) or `<n> FAILED` (exit 1):

| Row | Passes when |
|---|---|
| `first-byte` | the median of 3 streamed 256-token runs reaches its first content delta within `--first-byte-max` (0.12 s); `--ttft-factor 2` compares with 2× the engine's logged `ttft_ms` instead (NPU) |
| `deltas` | at least one content delta per 8 completion tokens (per k+1 with `--speculative k`) |
| `tok-s` | completion tokens / total time ≥ `--tok-s-min` (41) |
| `parity` | the streamed content equals the `stream: false` answer to the same request byte for byte |
| `queued-headers` | a streamed request sent 0.5 s after a 1500-token one gets HTTP 200 and `text/event-stream` headers within 1 s |
| `cb-gap` (`--cb-gap`) | a streamed request beside a running non-streamed one has no gap between deltas over 3× its median gap (`--cb` workers) |
| `divergence` (`--node-log`) | the worker log gains no `ov_genai_stream_divergence` line during the run |

Before any row, the check sends two unmeasured warm-up requests: a short streamed chat, then the
256-token count prompt with `stream: false`. On a GPU the first full-length answer after a worker loads
can differ from every later answer to the same request, streamed or not. Without the second warm-up,
the `parity` row would compare that cold answer with warm ones.

On an Intel GPU, OpenVINO's greedy answer to a request can also depend on the previous request's
prompt: the same prompt sent right after itself gets a different text than after another prompt or
first after a load. CPU does not show it, and it happens in plain OpenVINO GenAI too, so it does not
come from Cascadia. To see it, run `python3 scripts/ov_genai_order_repro.py <model-dir> GPU` (needs
`pip install openvino-genai`).

`--node-log` also takes `ssh:HOST:PATH` to read a remote worker's log. `--dry-run`
prints the plan and the full `reproduce:` command line without sending
anything; `--help` lists every option. The helpers' unit tests run with
`python3 -m unittest scripts/test_ov_genai_stream_check.py`.

## Limitations

- **Single-stage only.** No pipeline parallelism (`--total 1` enforced).
- **Streaming.** The default (stateful, non-`--cb`) path streams too: each time the pipeline writes a token (or a whole speculative round with `--draft-model` / `--prompt-lookup`), new text goes to the client, one write late: a write sends only the text its decode and the previous write's decode agree on, so a tokenizer cleanup rule that rewrites text because of the newest token (`it '` then `it's`) never sends a wrong byte, and the final chunk sends the last write's text. That costs about one token of delay per delta (the first byte comes at the second token). The streamed deltas join to exactly the text a non-streamed request returns: an echoed prompt and the whitespace after it are held back and dropped, a character split across tokens goes out whole, and the final chunk carries the rest and the answer's token count, so `usage` is the same as non-streamed. A non-streamed request (`stream: false` or unset) does not take this route: it gets the pipeline's own answer in one final chunk, exactly as before streaming existed, with no per-write decode, so a rewrite can never reach it. A rewrite of text already sent to a streaming client is logged as `ov_genai_stream_divergence` (the line carries byte counts only: the bytes sent and the answer's bytes, never the text) and counted in `cascadia_ov_genai_stream_divergence_total{model}`. One generation still runs at a time; while its prompt is still being read, the node keeps the stream alive and a queued request's headers go out at once. Each write re-decodes the answer so far, so the per-token cost grows with the answer's length: at 256 tokens streaming runs at 98.7% of the non-streamed speed (42.63 vs 43.18 tok/s, Phi-3.5-mini int4 on an Arc 140V iGPU, OpenVINO GenAI 2026.4.1). `--cb` streams per-iteration text deltas. An embedder that builds `OvGenaiBuilder` itself gets one final chunk per task unless it calls `stream_writes(true)`; the `cascadia worker` command turns it on.
- **Cancel and `stop` on the default path.** The default (non-`--cb`) path makes the whole answer in one `generate()` call, and that call now ends early on a client disconnect, on `/v1/cancel/{task_id}`, or at a `stop` match. It ends within one token, or within one decode round with `--draft-model` / `--prompt-lookup`, which deliver several tokens at a time. The first 4 `stop` sequences that are non-empty, at most 256 bytes and contain no U+FFFD are checked in the engine; all others are still cut by the API once the answer is complete. For models whose chat template frames output with special tokens, the engine's stop check is off (the API still cuts), and cancel still works. The cost is one callback per generated token; with streaming off (an embedder's default) also a short decode of the newest tokens while a `stop` is set. With streaming on, the check runs on the streamed text itself, so a stop split across tokens is found where it completes. `--cb` already cancels per request.
  - The prompt's prefill is one call that can't be interrupted, so a cancel during a long prompt's prefill waits for it to finish.
  - When ov-genai runs the streamer on its own worker thread, a cancel or stop lands one step later. That happens with the paged-attention / continuous-batching backend a plain pipeline can pick, and with the speculative pipelines. The node logs which one it got once, at the first request: `ov-genai streamer thread threaded_streamer=true`.
  - When the answer starts with text the server strips (an echoed prompt and the whitespace after it; an empty prompt counts as echoed), the streaming path never checks that text and checks the answer from its first visible character. With streaming off, the engine's stop check starts one short window after that text: about the longest stop's length + 16 tokens, plus one speculative round with `--draft-model` / `--prompt-lookup`. If the stripped text itself runs past that window, the check is skipped for the request. A stop the engine doesn't catch is cut by the API at the end, so the answer is the same either way; only the latency differs. A chat answer that merely starts with whitespace is checked at once.
  - A non-streaming client that half-closes its socket after sending the request (`shutdown(SHUT_WR)`: raw sockets, `nc -N`, some proxies) counts as a disconnect, and its request is cancelled. curl and the OpenAI SDKs don't half-close.
- **Draft / target tokeniser must match.** FastDraft companions are trained per target family; mixing across families won't work.
- **`--draft-model` and `--prompt-lookup` are mutually exclusive.** Both set `GenerationConfig.num_assistant_tokens`; the validator rejects the combination.
- **`--cb` is incompatible with `--draft-model` / `--prompt-lookup` and with VLM-layout exports.** The CB scheduler owns batch composition; speculative CB is a follow-up.
- **`--cb` is a throughput/latency trade, not a strict upgrade.** It costs ~10% at concurrency 1 and up to 5x on CPU with long prompts — see [When `--cb` helps](#when---cb-helps--and-when-it-hurts).
