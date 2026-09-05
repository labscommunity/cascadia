# Qwen3.8-27B (dense `qwen3_5`) — and fine-tunes like Qwopus3.8-27B-Flash

Status: **served two ways, hardware-validated on a Panther Lake AI PC
(2026-09-04)**: single-stage through `ov-genai`, and staged through the
`qwen35` engine (the Qwen3.6 IR-surgery machinery, made config-driven).

## The model

`Qwen/Qwen3.8-27B` (Apache-2.0, 2026-08-14) is `Qwen3_5ForConditionalGeneration`,
`model_type: qwen3_5` — the **dense** member of the Qwen3.5 architecture
family whose MoE member (`qwen3_5_moe`, Qwen3.5/3.6-35B-A3B) Cascadia already
serves ([qwen3.6.md](./qwen3.6.md), [qwen36-moe-support.md](./qwen36-moe-support.md)).
There is no `qwen3_8` model type anywhere (HF, optimum-intel, llama.cpp).

- 64 layers in the 3:1 pattern: 48× Gated DeltaNet (linear attention —
  recurrent `ssm` + `conv` state, **no KV cache**; 16 K-heads / 48 V-heads ×
  128, conv kernel 4) + 16× gated full attention (GQA 24/4, head_dim 256,
  `attn_output_gate`, partial-rotary 0.25, interleaved mRoPE `[11,11,10]`,
  `rope_theta 1e7`). Full attention at layers 4k+3.
- `hidden_size 5120`, `intermediate_size 17408` (dense SwiGLU MLP — no
  experts), `vocab_size 248320`, untied embeddings, 262K native context,
  `mtp_num_hidden_layers 1`, 27-layer vision tower (text-only here).
- The `state_dict` also carries `mtp.*` (one MTP layer + `fc`) and
  `visual.*`. Fine-tunes such as `Jackrong/Qwopus3.8-27B-Flash` keep the
  identical config and tensor set, so the same export recipe applies.

**GGUF is not the way in.** `Qwopus3.8-27B-Flash-GGUF` (llama.cpp arch
`qwen35`, MTP layer stored as `blk.64.*`) is rejected by OpenVINO GenAI's
GGUF reader (`llama` / `qwen2` / `qwen3` only) and by
`transformers(gguf_file=)`; OpenVINO core's native GGUF frontend lists
`qwen35` only in 2026.4 nightlies (greedy, batch 1). Export from the
safetensors sibling repo instead — the fine-tune author publishes both.

## Serving path A — single-stage `ov-genai` (whole-model IR)

Any `*-int4-ov`-layout export (Intel's `OpenVINO/Qwen3.8-27B-int4-ov`, or your
own — below) is VLM-layout (`openvino_language_model.xml` + text-embeddings +
vision IRs, no `openvino_model.xml`); the `ov-genai` engine auto-detects that
and serves text-only through `VLMPipeline`, exactly as for Qwen3.6:

```bash
cascadia run OpenVINO/Qwen3.8-27B-int4-ov --engine ov-genai --device GPU --api :8000
# alias: cascadia run qwen3.8-27b --engine ov-genai --device GPU
```

Nothing in Cascadia inspects `model_type` on this path; support is OpenVINO
GenAI's (2026.2+ has the fused GatedDeltaNet op and the Qwen3.5 VLM
pipeline).

## Serving path B — staged `qwen35` (IR surgery, 1–16 stages)

`cascadia shard` recognises `qwen3_5` / `qwen3_5_moe` config-first and cuts
the official IR at decoder-layer boundaries (no re-export, no
re-quantisation; stages inherit the int4 weights byte-for-byte):

```bash
cascadia shard --model /path/to/Qwen3.8-27B-int4-ov \
    --output-dir ./qwen38-2stage --num-stages 2
cascadia run ./qwen38-2stage --engine qwen35 --device GPU --api :8000
```

What changed versus the Qwen3.6-only exporter (`tools/qwen36_surgery/export_qwen36_moe.py`):

- hidden size, layer count and per-layer attention type are read from the
  model dir's `config.json` (`text_config`); the 27B's `past.{conv,ssm}.0..47`
  / `past.{key,value}.0..15` state ids are found by walking `layer_types`
  instead of the 40-layer interval formula.
- every stage must own at least one full-attention layer (the orphan-state
  rewire needs a same-kind cache to redirect global mask/past-length reads
  onto) ⇒ **at most 16 stages** for the 64-layer 27B; the exporter refuses
  finer splits up front.
- the manifest records `arch` (`qwen3_5`), `family`, `hidden_size`,
  `num_layers`, `layer_types`; the engine sizes its activation frames from
  `hidden_size` (5120 here; 2048 default for Qwen3.6-era manifests without it).
  A 256-token prefill chunk is a 5 MiB wire frame — far under the 256 MiB
  transport cap.

The MoE block was always opaque to the cut, so the engine, the pipeline
frames, DeltaNet reset semantics and the greedy-only / batch=1 invariants
(qwen36-moe-support.md §4.1) carry over unchanged.

## Exporting a fine-tune (Qwopus3.8-27B-Flash recipe)

Toolchain that worked (isolated venv): `transformers==5.2.0` (optimum-intel
pins `5.2.x` for `qwen3_5`; 5.0.0 fails with
`cannot import name 'Qwen3_5DynamicCache'`), `optimum-intel` main
(≥ 2.2.0.dev0 — includes the MTP head export), `nncf 3.3.0`,
`openvino 2026.3.1`, torch CPU. The task **must** be `image-text-to-text`
(`text-generation-with-past` is rejected for `qwen3_5`; the vision IRs it
emits are small and ignored at serve time).

```bash
optimum-cli export openvino --model Jackrong/Qwopus3.8-27B-Flash \
    --task image-text-to-text --weight-format int4 --group-size 128 \
    --ratio 1.0 --group-size-fallback ignore ./Qwopus3.8-27B-Flash-int4-ov
```

That is Intel's published recipe (INT4_ASYM, g128, ratio 1.0). Notes from the
run on a 64 GB box:

- the bf16 checkpoint is 52 GB and loads whole; export peaks right at a
  64 GB box's commit limit (auto-managed pagefile grew 67 → 70 GB). Stop
  anything else memory-heavy first.
- fine-tune repos may lack `preprocessor_config.json` /
  `video_preprocessor_config.json` — copy them from `Qwen/Qwen3.8-27B`.
- `main_export()` from Python writes the **bf16** IR (51 GB) and does not
  compress; the CLI compresses as a second step. Equivalent by hand:
  `OVModelForVisualCausalLM.from_pretrained(bf16_dir, quantization_config=
  OVWeightQuantizationConfig(bits=4, sym=False, group_size=128, ratio=1.0,
  group_size_fallback="ignore")).save_pretrained(int4_dir)` (2.6 min).
  The bf16 IR is worth keeping as a lossless on-device reference.
- output matches Intel's layout byte-for-byte in size: 13.93 GB language
  model, 1.27 GB text embeddings, 0.26 GB MTP head, 0.46 GB vision merger.

## Validation record — tate-07 (Intel Core Ultra X7 358H, Arc B390, 64 GB)

Panther Lake: 16 cores (4P+8E+4LPE), 64 GB LPDDR5X-8533 (~136 GB/s), Arc
B390 iGPU (12 Xe3 cores, 33.5 GiB UMA visible to OpenVINO), NPU 5; Windows
11; driver 32.0.101.8860. Runtime: OpenVINO 2026.3.x Python for the raw-IR
probes, GenAI 2026.2.1 SDK for the `cascadia` binary (MSVC build — the
MinGW toolchain cannot link the C++ GenAI API). Raw-IR numbers use the
stateful IR directly (`inputs_embeds` / `attention_mask` / `position_ids
[4,1,T]` all mRoPE rows equal / `beam_idx`), chunked prefill, T=1 greedy
decode — the same contract the staged engine feeds.

**Q1 — does it export properly?** Yes. Intel's `Qwen3.8-27B-int4-ov` and our
Qwopus int4 export both load and decode correctly on CPU and GPU, and the
2-stage surgery of the Qwopus IR validates (`--validate`: top-1 match,
top-5 overlap 5/5, 8/8 multi-token greedy parity chain vs whole model,
relative logit drift 1.6e-2 from stage-boundary f16 fusion order).

| IR | device | decode tok/s | notes |
|---|---|---|---|
| Intel `Qwen3.8-27B-int4-ov` | GPU (B390) | **6.3–6.4** (3 runs × 63 tok) | compile 60 s, 16 GB resident, TTFT 0.23–0.26 s at 5-token prompt |
| Intel `Qwen3.8-27B-int4-ov` | CPU | 3.67 | first run (20–28 s JIT warm-up on the first inference), 28 GB resident |
| Qwopus int4 (ours) | GPU (B390) | **6.4–6.7** (six prompts) | identical answers to CPU; 391 for 17×23, primes 101/103, haiku, `<think>` on raw prompts |
| Qwopus int4 (ours) | CPU | 3.8 | 28 GB resident |

Greedy text is identical between CPU and GPU for the same prompt on both
IRs. The bandwidth ceiling for ~14 GB of int4 weights at 136 GB/s is ~9–10
tok/s; the stateful path's GatedDeltaNet reference kernel on CPU
(openvino #37845: only the PagedAttention path has the optimised GDN kernel)
explains most of the CPU gap.

**HF-reference parity (Qwopus).** Reference: `transformers 5.2.0` bf16
greedy on a 28-core Mac Pro (1.5 TB RAM; `Qwen3_5ForCausalLM`, MTP and
vision tensors dropped, ~2 s/token). Compared token-for-token over 32
greedy tokens against the OpenVINO exports on tate-07:

| prompt | OV **bf16** IR (CPU) | OV **int4** IR (GPU = CPU) |
|---|---|---|
| raw `The capital of France is` | 32/32 | 32/32 |
| raw `user: Explain how rainbows form.` (the parity-test prompt) | 32/32 | 32/32 |
| chat `What is 17 * 23? …` | 4/4 (`391`) | 4/4 (`391`) |
| chat `List three prime numbers greater than 100.` | 32/32 | diverges at token 15 (`1.  **101**` vs `1. **101**`: a spacing token; content identical) |
| chat `Write a haiku about mountains.` | diverges at token 0 | diverges at token 2 |

The haiku's first step is an **exact bf16 logit tie** in the reference
(`Stone` vs the alternative, both 18.5), so both OpenVINO results are the
other side of a coin flip, not an error. Net: the bf16 IR reproduces the
HF reference exactly wherever the reference is not tied — the export is
correct — and the int4 IR's residual differences are quantisation
near-ties, with every factual answer intact.

**Q2 — throughput through the real entry point.** `cascadia run … --api`
then `POST /v1/chat/completions` (`enable_thinking: false`, `max_tokens 96`,
greedy). `completion_tps` = completion tokens / whole-request wall time, so
it includes prefill of the 19-token prompt and HTTP.

| path | binary / SDK | device | load | 96-token request | 17×23 |
|---|---|---|---|---|---|
| `--engine qwen35`, Qwopus 2-stage chain (`cascadia shard --num-stages 2`) | MSVC, GenAI 2026.2.1 | GPU (B390) | 41 s (both stages) | **15.0–15.6 s → 6.2–6.4 tok/s** | `391` |
| `--engine qwen35`, same chain | MSVC, GenAI 2026.2.1 | CPU | ~80 s | 32.2 s → 3.0 tok/s | `391` |
| `--engine ov-genai`, Qwopus int4 (whole IR, `VLMPipeline` text-only) | MSVC, GenAI **2026.3.0** | GPU (B390) | ~60 s | **13.2 s → 7.3 tok/s** | `391` |
| `--engine ov-genai`, Qwopus int4 | MSVC, GenAI 2026.2.1 | CPU | ~40 s | serves (`391`; throughput not measured on CPU) | `391` |
| `--engine ov-genai`, Intel `Qwen3.8-27B-int4-ov` as published | MSVC, GenAI 2026.2.1 **and** 2026.3.0 | GPU | — | `pipeline_create_vlm` throws (tokenizer IRs, see below) | — |
| `--engine ov-genai`, Intel IR after `convert_tokenizer` | MSVC, GenAI 2026.2.1 and 2026.3.0 | CPU | ~40 s | serves | `391` |

So on this box the single-stage GenAI path is the fastest (its PagedAttention
backend has the optimised GatedDeltaNet kernel) and works on both the
2026.2.1 SDK the repo pins and 2026.3; the staged path is ~15 % behind on
GPU.

**Why Intel's published IR throws, and the fix.** Its `openvino_tokenizer`
/ `openvino_detokenizer` IRs were built with openvino-tokenizers **2026.4
nightly** (stateful `ReadValue`/`Assign` ops and a bare `Truncate` op that
the 2026.2/2026.3 tokenizer extension cannot load); GenAI's `VLMPipeline`
constructor loads them and throws. Isolated with hard-linked variants of
the Intel directory served by the same binary:

| variant of `OpenVINO/Qwen3.8-27B-int4-ov` | `ov-genai` |
|---|---|
| as published | throws in `pipeline_create_vlm` |
| + our `chat_template.jinja` only | throws |
| + tokenizer/detokenizer IRs from a 2026.3.1 export (Intel's template kept) | **serves** (`391`) |
| + tokenizer/detokenizer IRs regenerated in place with `convert_tokenizer` 2026.3.1 | **serves** (`391`) |
| as published, GenAI **2026.5 nightly** Python `VLMPipeline` | creates in 9 s, generates correctly (`LLMPipeline` still rejects the template's `is undefined`) |

The chat template is not the problem on Cascadia's path (the API renders it
itself). So, until the shim is built against a 2026.4+ GenAI SDK, serve
Intel's IR after one command with the *installed* openvino-tokenizers:

```bash
convert_tokenizer /path/to/Qwen3.8-27B-int4-ov --with-detokenizer -o /path/to/Qwen3.8-27B-int4-ov
```

Exports made with the 2026.3 toolchain (the recipe above) need nothing.

**Q3 — context capacity** (weights resident, growing synthetic prompt).
Qwopus int4 on the B390 via the raw stateful IR (512-token prefill chunks,
16 greedy tokens after the prompt), one run per point, fresh process each
(RSS = host-visible resident set incl. the iGPU's UMA allocations; the
OVMS node was paused so the box was otherwise idle):

| prompt tokens | TTFT | prefill tok/s | decode tok/s | resident |
|---|---|---|---|---|
| 1 K | 2.4 s | 425 | 6.1 | 17.1 GB |
| 4 K | 8.1 s | 505 | 6.4 | 18.1 GB |
| 8 K | 15.5 s | 527 | 6.1 | 19.1 GB |
| 16 K | 32.8 s | 500 | 6.2 | 19.5 GB |
| 32 K | 78.5 s | 417 | 5.1 | 20.8 GB |
| 64 K | 219 s | 299 | 4.0 | 21.9 GB |
| 128 K | 667 s | 196 | 3.1 | 25.1 GB |
| 256 K (native max) | 3256 s (54 min) | 81 | 2.5 | 31.9 GB at decode (~43 GB peak during prefill) |

Memory is not the limit on a 64 GB box: state grows ~60–120 KB per
context token (16 attention layers × 4 KV heads × 256 × f16 = 64 KB/token
of KV; the 48 DeltaNet layers hold fixed-size recurrent state), so even
the full 262 K window costs well under 32 GB on top of the 16 GB of
weights. What degrades is time: prefill falls from ~500 tok/s to ~200
tok/s as attention over the growing KV dominates, and decode from 6.4 to
3.1 tok/s by 128 K. Practical guidance on this hardware: ≤32 K tokens
stays interactive (TTFT ≲ 80 s, decode ≥ 5 tok/s); 64–128 K works but
TTFT is 4–11 minutes; the full 262 K window fits (peak ~43 GB of 64 GB)
but costs 54 minutes of prefill, so it is a capacity fact, not a usable
setting on an iGPU. (The 256 K prompt was 4 tokens over the model's
`max_position_embeddings`; mRoPE extrapolated without error.)

**Regression — Qwen3.6-35B-A3B on the same exporter and engine** (tate-07,
B390). The config-driven exporter cut `OpenVINO/Qwen3.6-35B-A3B-int4-ov`
exactly as before (`qwen3_5_moe`, 40 layers = 30 linear + 10 full, 40 state
variables per 20-layer stage; `--validate`: top-1 match, top-5 5/5, 8/8
greedy chain-vs-whole). Served through `cascadia run --engine qwen35 --device
GPU --api`: `391` for 17×23 and a correct 64-token rainbow answer at
**19–20 tok/s** end-to-end (the MoE reads ~1.7 GB per token; the Lunar Lake
numbers in qwen36-moe-support.md were 4.7–8.8). The same tree with its
manifest stripped to the Qwen3.6-era keys (`arch`, `source`,
`last_logits_only`, `stages` — no `hidden_size`) served identically through
the `--engine qwen36-moe` alias, so existing shard trees and scripts keep
working.

## Limits and follow-ups

- **Greedy-only, batch=1** on the staged path (DeltaNet state cannot be
  rolled back; position-0 reset is the only recovery — qwen36-moe-support.md §4.1).
- **NPU**: out of scope (dynamic-shape, 14 GB int4).
- **MTP head** (`openvino_mtp_model.xml`, 264 MB, one full-attention layer
  fed with `hidden_states` + `inputs_embeds`) is exported but unused.
  Qwopus's whole point is 80.7 % MTP draft acceptance; GenAI grew MTP
  drafting for this family in the 2026.4 nightlies (ContinuousBatching /
  PagedAttention only). For the staged engine the missing piece is a
  DeltaNet state snapshot/restore around each verify step (~40 MB/step
  at hidden 5120).
- **Vision** input is not supported on either path (text-only).
- Long-context OpenVINO-side knobs (`KV_CACHE_PRECISION`, PagedAttention)
  only reach the `ov-genai` path; the staged path's caches are graph-level
  `ReadValue`/`Assign` variables.
