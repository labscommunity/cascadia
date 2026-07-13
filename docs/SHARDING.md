# Sharding models for cascadia

`cascadia shard` slices a HuggingFace causal-LM into per-stage OpenVINO IRs
that the worker engines (`ov-runtime`, `ov-dist-spec`) can load. Each
stage holds a contiguous slab of decoder layers + (on the first stage)
the embedding + (on the last stage) the final norm + lm_head.

> Running shards on the **Intel NPU** needs a static-shape, stateless export
> (`--target npu`) and a host-side KV ring — see [NPU_SHARDING.md](NPU_SHARDING.md).

## Quick start

```bash
# One-time pip install (export-time only, not needed at runtime). From a source
# checkout; from a release bundle run `cascadia doctor`, which prints the pins:
pip install -r tools/requirements.txt

# Two-stage shard from HF:
cascadia shard --model unsloth/Meta-Llama-3.1-8B-Instruct \
             --output-dir ~/cascadia/llama-8b-2stage \
             --num-stages 2 \
             --quantization int4
```

Output:

```text
~/cascadia/llama-8b-2stage/
  pipeline_config.json    # global metadata: model_id, layer count, vocab, etc.
  tokenizer/              # tokenizer.json + config.json + special tokens
  stage_0/
    openvino_model.xml    # OpenVINO IR (FP16 graph + INT4 weights)
    openvino_model.bin    # quantized weights
    stage_config.json     # per-stage metadata: layer_start, layer_end, has_embed, has_head
  stage_1/
    openvino_model.xml
    openvino_model.bin
    stage_config.json
```

This directory is portable. Copy it to every worker node (or re-run
`cascadia shard` separately on each, depending on which is faster on your
network).

## Flags

| Flag | Meaning |
|------|---------|
| `--model` | HF repo id (`unsloth/Meta-Llama-3.1-8B-Instruct`), a local dir with `config.json` + `*.safetensors`, or (Gemma-4 / Qwen3.6 only) an exported OpenVINO IR dir. HF repos auto-download into `~/.cache/cascadia/models/`. |
| `-o`, `--output-dir` | Where to write the shard tree. |
| `--num-stages N` | Pipeline stages to split into. 2 is the common case for 2-machine setups; use 3+ for larger clusters. |
| `--quantization` | `int4` (default — typical), `int4-asym`, `int8`, or `fp16`. INT4 needs nncf installed. |
| `--layer-split a,b,...` | Override the uniform split. With `--num-stages 3 --layer-split 16,24` on a 32-layer model: stage 0 = layers 0..15, stage 1 = 16..23, stage 2 = 24..31. Useful for asymmetric hardware (e.g. give the bigger node more layers). |
| `--stage N` | Re-export only stage N (debugging). |
| `--target cpu-gpu\|npu` | Deployment target. `npu` emits a stateless static-shape shard — see [NPU_SHARDING.md](NPU_SHARDING.md). |
| `--static-seq N` / `--static-context N` | NPU only: fixed query window (must be 1) and total context (default 1024). |
| `--default-dtype fp16\|fp32` | torch dtype during export. `fp16` (default) is required for `--target npu`. |
| `--python /path/to/python` | Override Python interpreter. By default cascadia tries `python3` then `python` and picks the first that can import the export deps — so a bare `python3` stub (common on Windows) doesn't shadow the real install. |
| `--skip-check` | Skip the dependency probe — pass if you know your env is already set up. |

## Supported architectures

Cascadia's exporter uses HuggingFace's standard decoder-layer classes,
so anything whose layer exposes the conventional names
(`self_attn.{q_proj,k_proj,v_proj,o_proj}`, `mlp`, `input_layernorm`,
`post_attention_layernorm`) and uses RoPE rotary embeddings works.

This table tracks what `tools/export_shards.py` accepts today. For each
family, **Status** is one of:

- ✅ **tested** — exercised on hardware in CI or by hand; produces tokens that
  match HF reference closely (single-token-level agreement on greedy decode
  for short prompts).
- ⚠️ **best-effort** — loads and traces, but either has no automated quality
  check, or a SOFT quirk (long-context RoPE / sliding window) means it is exact
  only within the original context window.
- 🚧 **rejected** — a recognised `model_type` (or config feature) the exporter
  cannot honour; rejected config-first, before download (#60/#69), with a clear
  error and a pointer to the right tool. No shards produced unless
  `CASCADIA_ALLOW_LOSSY_EXPORT=1`.
- ❌ **unknown** — not in any accept/reject list; `detect_architecture` falls
  back to the Llama path with a stderr warning. May work (most post-Llama-2
  dense decoder-only LMs do) or produce garbage — benchmark before trusting.

| Family | Status | Notes |
|--------|--------|-------|
| **Llama** (1, 2, 3, 3.1, 3.2, 3.3) | ✅ tested | Reference path. Llama 3.2 1B/3B + 3.3 have tied embeddings — handled. Llama-3 RoPE scaling (`rope_type: "llama3"`) is recomputed in the Rust runtime. |
| **Mistral** (7B v0.1+, NeMo 12B) | ✅ tested | Sliding-window attention is disabled in the export (uses standard SDPA across the full KV cache). NeMo and Small 3 (`model_type: "mistral"`) work out of the box; Pixtral / Mistral Small 3.1 (`model_type: "mistral3"`) are wrapper configs — see "Multimodal text-only path" below. |
| **Qwen2 / Qwen2.5** | ✅ tested | Uses Qwen2-specific decoder layer with bias on q/k/v projections. |
| **Qwen3 dense** | ✅ tested | Dispatches to `Qwen3DecoderLayer`; `q_norm` / `k_norm` (RMSNorm on Q/K before RoPE) are detected and applied automatically. `head_dim` is read from `config.head_dim` (Qwen3 decouples it from `hidden_size / num_heads`). |
| **DeepSeek R1 Distills** (Qwen / Llama 7B–70B) | ✅ supported | Distills inherit their base config (`model_type: "qwen2"` or `"llama"`), so they ride the existing paths with no special handling. Short aliases (`r1-distill-qwen-7b`, …) resolve via `tools/model_aliases.py` (#49). |
| **DeepSeek-V2-Lite** | ⚠️ best-effort / MoE | Standard attention + RoPE on the Llama path, BUT V2-Lite is itself MoE (`n_routed_experts`), so `is_moe_config` now rejects it (#60). Use the dense R1-Distills instead. |
| **Phi-3** (Mini 3.8B, Medium 14B) | ✅ supported | `Phi3DecoderLayer` loads; fused `qkv_proj` is split (#58). **Partial rotary** (`partial_rotary_factor < 1.0`) is honoured (#69) — only the leading `factor·head_dim` dims rotate, the rest pass through. |
| **Phi-3-Small** (7B) | 🚧 rejected | A distinct **blocksparse** architecture (`Phi3SmallForCausalLM`, tiktoken tokenizer, `trust_remote_code`) — not the `Phi3DecoderLayer` path. Unrunnable on any OpenVINO stack today: no prebuilt int4-ov IR exists, `cascadia shard` fails at config-parse, and `optimum-cli export openvino` rejects `phi3small` as a custom architecture. |
| **Phi-4** (14B, Phi-4-mini, Phi-4-reasoning) | ✅ supported (short context) | Reports `model_type: "phi3"`. **Phi-4-mini** has `partial_rotary_factor=0.75` (honoured, #69) and **LongRoPE** scaling (not modeled — soft-warned). Exact within the original context window; long-context degrades. |
| **Gemma 1 / Gemma 2** | ✅ supported (short context) | Gemma 1 (2-norm) and Gemma 2 (4-norm + attention/final **logit softcapping** + sqrt(hidden) embed scaling) are applied (#61). Sliding-window attention (Gemma 2) is treated as full-causal — exact within the window. |
| **Gemma 3** (1B / 4B / 12B / 27B) | 🚧 rejected | Interleaved local/global sliding-window, QK-norm, and per-layer-type RoPE bases are not modeled; `detect_architecture` rejects it up front (#69) rather than mis-exporting through the `gemma` path. |
| **Gemma 4** (E2B / E4B / 31B, April 2026) | ✅ via dedicated exporter | `cascadia shard` auto-dispatches Gemma 4 to `tools/export_gemma4.py` (#48), which handles per-layer-type asymmetric `head_dim` (256/512), per-layer-type/proportional RoPE, KV-sharing, per-layer embeddings, and `final_logit_softcapping=30.0`. **Exporter-only today**: the shards run via OpenVINO Core; multi-stage on the Rust `ov-runtime` engine is a tracked follow-up (see `docs/architectures/gemma4-support.md`, Phase B). The 26B-A4B MoE variant is rejected. |
| **Llama 4** (Scout / Maverick, April 2025) | 🚧 rejected | MoE with iRoPE (NoPE every 4 layers), QK-norm, chunked attention. Rejected by `is_moe_config` (#60); would need MoE infra + custom rotary. |
| **Qwen3 MoE** (30B-A3B, 235B-A22B) | 🚧 rejected | `model_type: "qwen3_moe"`. 128 experts, top-8 routing, no shared expert. Rejected by `is_moe_config` (#60). Note: `cascadia-engine-sparse-moe` (Kimi K2.6) demonstrates the routing pattern but is not wired into the generic `cascadia shard` path. |
| **Qwen3.5 / Qwen3.6** (35B-A3B, hybrid Gated-DeltaNet MoE) | ✅ single-stage via `ov-genai` (OV ≥ 2026.2); ✅ staged via `cascadia shard` + `--engine qwen36-moe` | `model_type: "qwen3_5_moe"` (expert fields nested under `text_config`). 256 experts top-8 + 1 shared expert, 3:1 linear:full hybrid attention, mRoPE. OV 2026.2 compiles it natively: `--engine ov-genai --model <optimum-int4-ir>` serves it whole-model, and `cascadia shard --model <int4-ov dir>` cuts the official IR into stage shards for the in-process `qwen36-moe` staged engine (IR surgery, no re-quantization — #77, `docs/architectures/qwen3.6.md`, `docs/architectures/qwen36-moe-support.md`). |
| **gpt-oss** (20b, 120b, August 2025) | 🚧 rejected | OpenAI's open-weight MoE. Alternating sliding/full per-layer, YARN RoPE, sigmoid-routed top-k, MXFP4 native quant. Rejected by `is_moe_config` (#60). |
| **Mixtral 8x7B / 8x22B** | 🚧 rejected | `is_moe_config` rejects it up front (#60) rather than silently miswiring `block_sparse_moe` as a dense MLP. For a one-off MoE demo, see `cascadia-engine-sparse-moe` (Kimi K2.6). |
| **DeepSeek-V3 / R1** (full 671B) | 🚧 rejected | MoE + Multi-head Latent Attention (`q_lora_rank`, `kv_lora_rank`, split `qk_nope_head_dim` / `qk_rope_head_dim`) — a full attention rewrite. Rejected by `is_moe_config` (#60); use R1-Distill-Qwen/Llama instead. |
| **Jamba / Falcon-Mamba / Granite 4 / Nemotron-H** | 🚧 rejected | Hybrid Transformer-Mamba (SSM) architectures need a Mamba kernel; `detect_architecture` rejects them up front (#69). Out of scope for OpenVINO IR export. |
| **Multimodal (Llava, Llama 4 multimodal, Gemma 3/4 multimodal)** | ❌ not supported | Vision / audio encoders are not exported. For multimodal configs with a `text_config` sub-config (Gemma 3, Llama 4, Mistral 3.x), the Rust runtime's RoPE loader unwraps `text_config` so a text-only IR may load — but `cascadia shard` does not yet auto-extract just the text tower; you have to slice the model yourself first. |

For unknown architectures the script falls back to `LlamaDecoderLayer`
and warns on stderr. If the resulting shard's outputs match the HF
reference, it's good. (To compare against a reference, serve an
`optimum-cli`-exported IR of the same model through `--engine ov-genai`;
ov-genai cannot read a HuggingFace checkout or a shard tree.) If they diverge, file an
issue with the model's `model_type` and `architectures` from `config.json`.

### Architecture quirks — handled, dropped, or rejected

The exporter recomputes RoPE locally in the OV IR using a `theta` baked
from `config.rope_theta` (or `config.rope_parameters.rope_theta` on
transformers ≥ 5). How it treats the per-family extras, as of #69 (+ #48
for Gemma 4):

| Feature | Affected families | Status |
|---------|-------------------|--------|
| **Partial rotary** (`partial_rotary_factor < 1.0`) | Phi-1/2, Phi-3/4 Mini, StableLM-2, Persimmon | ✅ **handled** (#69) — only the leading `factor·head_dim` dims rotate; the rest pass through (byte-parity with HF Phi/Phi3). |
| **Logit softcapping** (`final_logit_softcapping`, `attn_logit_softcapping`) | Gemma 2, Gemma 4 | ✅ **handled** — Gemma 2 via the gemma2 path (#61), Gemma 4 via `export_gemma4.py` (#48). On any other family `check_export_quirks` treats it as a HARD quirk. |
| **Embed scaling by sqrt(hidden_size)** | Gemma 1, 2, 4 | ✅ **handled** (#61) — the embed stage applies `arch.embed_scale`. |
| **Tied embeddings** | Llama 3.2 small, Granite, Cohere2, SmolLM3, Gemma | ✅ **handled** via `tie_word_embeddings`. |
| **QK-Norm** (RMSNorm on Q/K before RoPE) | Qwen3 (✅), OLMo 2, Llama 4 | ✅ on the qwen3 path; on any other family `check_export_quirks` flags it HARD (rejected) — without it output collapses to repetition. |
| **Asymmetric per-layer-type `head_dim`**, **per-layer-type RoPE**, **KV-sharing**, **per-layer embeddings** | Gemma 4 | ✅ **handled** by the dedicated `tools/export_gemma4.py` (#48). The generic builder assumes one `head_dim`, so a *non*-Gemma-4 model with these is rejected by `check_export_quirks`. |
| **LongRoPE** (`rope_type: "longrope"`) | Phi-3 Mini 128k, Phi-4 Mini long-context | ⚠️ **dropped (soft)** — plain RoPE baked; exact within the original context window, degrades beyond it (warn-and-proceed). |
| **YARN** (`rope_type: "yarn"`) | Qwen2.5 long-context, gpt-oss, DeepSeek-V3 | ⚠️ **dropped (soft)** — same as LongRoPE. |
| **NTK-by-parts / "dynamic"** scaling | Older Llama-2 long-context derivatives | ⚠️ dropped (soft). |
| **Per-layer-type sliding window** (`layer_types`) | Gemma 2/3, gpt-oss, Cohere2 | ⚠️ **dropped (soft)** — treated as full-causal; exact within the window, over-attends beyond it. |
| **MLA** (Multi-head Latent Attention) | DeepSeek-V2/V3/R1 (full) | 🚧 **rejected** — full attention-block rewrite. |
| **NoPE layers** (no rotary every N) | Llama 4, SmolLM3 | 🚧 rejected (Llama 4 is MoE; SmolLM3 unsupported). |
| **MoE routing** | Mixtral, Qwen3-MoE, Llama 4, gpt-oss, GraniteMoE, DeepSeek-V2/V3, Gemma 4 26B-A4B | 🚧 **rejected** by `is_moe_config` (#60) — the generic builder cannot route experts. |

HARD quirks (those that corrupt output even on short prompts, or won't
load) abort the export with a clear error; SOFT quirks (correct within the
original context window) warn and proceed. To force an export past a HARD
quirk anyway, set `CASCADIA_ALLOW_LOSSY_EXPORT=1` (output WILL diverge from
the HF reference).

## Picking `--num-stages`

The right answer is determined by **memory** more than throughput. A
shard's GPU memory footprint at runtime is roughly:

```text
shard_size ≈ (layers_in_shard / total_layers) × model_size_int4
           + KV_cache_per_token × max_context_len
           + workspace (~500 MB OV plugin)
```

For Llama 3.1 8B INT4 (~4 GB) on a 12 GB Battlemage Arc B390 with
4096-token max context: a 16/16 2-stage split fits comfortably.

For Llama 3.1 70B INT4 (~35 GB) on the same B390: you'd need 4–5 stages
to keep each shard under 8 GB of weights + leave room for KV cache.
Practical layout: `--num-stages 4 --layer-split 20,40,60` (uneven, with
the most layers on the GPU/CPU node that has the most free RAM).


There's a per-stage overhead (network round-trip + OV plugin init), so
**don't** over-shard a model that fits on fewer nodes. 2 stages costs
~1ms / token of network latency on TB4, ~5ms on 1 GbE.

## Picking `--quantization`

| Mode | Weight bits | Quality loss | Speed | Recommended for |
|------|------------:|--------------|-------|-----------------|
| `int4` | 4 (sym, group=128) | ~1-2% on standard benchmarks | fastest | **Default.** Almost always the right choice. |
| `int4-asym` | 4 (asym, group=128) | similar to int4 | similar | When INT4 sym shows numerical issues on a specific model. |
| `int8` | 8 (asym, per-channel) | ~0.5% | slower than int4 | When int4 quality is unacceptable. |
| `fp16` | 16 | none | slowest, biggest | Debugging or when nncf isn't installed. |

INT4 with group_size=128 is the sweet spot on Intel Arc / iGPU because
the weight unpacking happens in the same SIMD pass as the matmul.

## Re-exporting one stage

If you change a config knob (rope_theta, quantization mode) and want to
avoid the full multi-minute re-export:

```bash
cascadia shard --model <same args> --stage 1
```

This re-runs only stage 1 against the same source weights. The
`pipeline_config.json` is overwritten each invocation but should be
identical.

## Troubleshooting

**"NNCF not installed" warning on INT4 export**: the script falls back to
FP16 weights silently. Install nncf (`pip install "nncf>=2.18"`) and re-run.

**"tied embeddings" log line on Llama 3.2 1B/3B**: not an error. Llama
3.2 small models share `lm_head.weight` with `embed_tokens.weight`. The
exporter detects `config.tie_word_embeddings == True` and reuses the
same tensor.

**Export hangs after "Quantization OK"**: NNCF on tied-weight models
sometimes loops in an internal scan. Use `--quantization fp16` and let
the runtime do dynamic int4 unpacking instead, or use `--quantization
int8` which doesn't hit the same path.

**"unknown model_type, falling back to Llama" warning**: the model
isn't in the explicit support list. The export will produce a working
IR for any model whose layer interface matches Llama's, but you should
verify the outputs (serve an `optimum-cli`-exported IR of the same model
through `--engine ov-genai` and compare the first 10 generated tokens).

**OOM during INT4 quantization**: NNCF holds the full FP16 model in
RAM during compression. For 70B+ models, run on a machine with ≥ 64 GB
RAM, or use `--quantization fp16` and quantize per-shard later via
`nncf.compress_weights` directly.
