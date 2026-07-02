# Qwen3.6-35B-A3B (and Qwen3.5) — hybrid Gated-DeltaNet MoE

Tracking issue: #77. Status: **single-stage supported and
hardware-validated; the generic multi-stage exporter rejects it — see
[qwen36-moe-support.md](qwen36-moe-support.md) for the dedicated
sharded (IR-surgery) path.**

## The model

`Qwen/Qwen3.6-35B-A3B` — `model_type: qwen3_5_moe` (shares the Qwen3.5
architecture class; the expert/MoE fields live under the nested
`text_config`). Apache-2.0. 35B total / ~3B active.

- 40 layers: repeating 3× Gated DeltaNet (linear attention, recurrent
  matrix + conv state, **no KV cache**) + 1× gated full attention
  (GQA 16/2, head_dim 256, `attn_output_gate`, partial-rotary 0.25,
  **mRoPE** `mrope_section [11,11,10]` interleaved, `rope_theta 1e7`).
- MoE per layer: 256 experts, top-8 routed + 1 shared expert (sigmoid
  gate), `moe_intermediate_size 512`.
- `vocab_size 248320`, `hidden_size 2048`, 262K native context,
  `mtp_num_hidden_layers: 1` (MTP head), 27-layer vision tower (it is a
  VLM; text-only here).

## How to run it: single-stage `ov-genai` (OV GenAI ≥ 2026.2)

OpenVINO 2026.2 natively compiles `qwen3_5_moe` (fused GatedDeltaNet op,
CPU & GPU, tool-calling included). One cascadia-side requirement the
issue didn't anticipate: the published export is **VLM-layout**
(`openvino_language_model.xml` + separate embeddings/vision IRs — no
`openvino_model.xml`), which `LLMPipeline` cannot open ("Port for tensor
name input_ids was not found"). The `ov-genai` engine auto-detects this
layout and uses `VLMPipeline` (text-only) via the shim's
`cascadia_pipeline_create_vlm`:

```bash
# Pre-exported IR (preferred):
#   https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov
cascadia run --engine ov-genai --model /path/to/Qwen3.6-35B-A3B-int4-ov --device GPU
```

Or export yourself with Optimum-Intel on the 2026.2 toolchain
(`tools/requirements.txt` pins the floors):

```bash
optimum-cli export openvino -m Qwen/Qwen3.6-35B-A3B \
    --weight-format int4 --task text-generation-with-past <out-dir>
```

INT4 weights are ~18–20 GB — fits a single 32 GB Intel AI PC via UMA.
NPU is out of scope (35B does not fit). Vision input is not supported on
this path (text-only).

## Validation record (Intel Lunar Lake node)

Hardware: Core Ultra 7 258V (Lunar Lake), 32 GB UMA, Windows; SDK: OV
GenAI 2026.2 (build 21894); model: `OpenVINO/Qwen3.6-35B-A3B-int4-ov`
(18.3 GB on disk); toolchain probe: transformers 5.4.0 loads the config
(`norm_topk_prob` is not a config parameter in this family).

- **Served end-to-end**: `cascadia run <model-dir> --device GPU` →
  `/v1/chat/completions` returned coherent OpenAI-format completions
  (2/2 requests + `/v1/models`), over the mesh network from a remote
  client.
- **Throughput (GPU, ~1K-token prompt, 128 new tokens)**: greedy
  1.27 tok/s (TPOT 785 ms, TTFT 10.3 s); with prompt-lookup
  (`num_assistant_tokens=5`, `max_ngram_size=3`) 2.46 tok/s (TPOT
  406 ms). Lunar Lake serves this model *correctly but slowly*; usable
  interactive serving wants an Arc/B60-class GPU (the 2026.2 release
  notes' optimization target for Qwen3 MoE).
- **Crash fixed during bring-up**: `ov::genai::StreamerVariant{}`
  default-constructs an empty `std::function` → fast-fail `0xc0000409`
  mid-generate; the shim passes `std::monostate{}` explicitly.
- **Caution**: prompt-lookup output diverged textually from greedy on
  the same prompt (GPU numerics under batched verification) — HF-parity
  checks must compare per decode mode, applied output not intermediates.

Follow-ups (validated on the same node):

- **CPU device**: coherent completion through the API (`--device CPU`).
- **2026.2 engine regression smoke** (ov-genai, dense Qwen3-1.7B-int4-ov):
  CPU ✅ and GPU ✅ with textually identical outputs; NPU rejected that
  export (`vpux-compiler ... 16 duplicated names`) — **not** a 2026.2
  regression: the NPU-proven `llama-3.1-8b-instruct-npu-ov` export served
  fine on NPU. Dynamic-shape `int4-ov` exports are CPU/GPU artifacts; NPU
  needs its dedicated static-shape exports, as before.

**Tool-calling validated** on BOTH serving paths through
`/v1/chat/completions` (CPU): a hermes-format tool schema in
the system message yields a well-formed
`<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>`
from the nonsharded `ov-genai` path (reasoned, then called) and from the
staged `qwen36-moe` engine (empty think, then called). Prompt-level
validation per #77 — an OpenAI `tools`/`tool_calls` API surface is a
separate feature, not in #77 scope.

**`usage` token counts fixed**: real `prompt_tokens` ride
the engine's final chunk; `total = prompt + completion`.
**Thinking-mode**: `enable_thinking: false` on the request prefills the
template's empty think block (staged engine); default unchanged.

**Fixed prompt set validated**: 6/6 factual-correctness
PASS on BOTH paths (`tools/qwen36_surgery/promptset.json`; outputs in
`tools/qwen36_surgery/golden/promptset_*.json`). Staged engine ran with
`enable_thinking: false` and small budgets; the `ov-genai` path thinks
unconditionally (the pipeline applies the chat template internally, so
prompt-level think suppression doesn't reach the assistant position —
that approach was tried and reverted; passing template kwargs at the
GenAI level is the follow-up) and passed with 256-token budgets. True bf16 HF-reference
parity is NOT claimable on the hardware tested (35B bf16 needs ~70 GB;
a 32 GB node cannot hold it) — the gate is factual correctness +
cross-path consistency on
the official int4 artifact — we state the actual gate used rather
than claiming full HF-reference parity.

Known limitation: the tool-calling probe and prompt set have been
validated on CPU only, not yet repeated on iGPU.

## Why `cascadia shard` rejects it

The multi-stage exporter builds dense decoder layers with one SDPA + KV
cache per layer. This architecture has recurrent DeltaNet state on 30 of
40 layers, a 256-expert router + shared expert, mRoPE, and an MTP head —
none of which the generic export path models. `is_moe_config()` rejects it
config-first (nested `text_config` unwrap + `qwen3_5_moe` model_type,
\#77) with a pointer to the `ov-genai` path above.

The pipeline-parallel design (the actual too-big-for-one-box story) is
documented separately in [qwen36-moe-support.md](qwen36-moe-support.md).
