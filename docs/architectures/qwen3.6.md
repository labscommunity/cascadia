# Qwen3.6-35B-A3B (and Qwen3.5) — hybrid Gated-DeltaNet MoE

Tracking issue: #77. Status: **single-stage supported (pending hardware
validation); multi-stage rejected — see
[qwen36-moe-support.md](qwen36-moe-support.md) for the deferred sharded
design.**

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
CPU & GPU, tool-calling included). The `ov-genai` engine hands a model
directory to `openvino_genai::LLMPipeline`, so no cascadia engine code is
involved:

```
# Pre-exported IR (preferred):
#   https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov
cascadia run --engine ov-genai --model /path/to/Qwen3.6-35B-A3B-int4-ov --device GPU
```

Or export yourself with Optimum-Intel on the 2026.2 toolchain
(`tools/requirements.txt` pins the floors):

```
optimum-cli export openvino -m Qwen/Qwen3.6-35B-A3B \
    --weight-format int4 --task text-generation-with-past <out-dir>
```

INT4 weights are ~18–20 GB — fits a single 32 GB Intel AI PC via UMA.
NPU is out of scope (35B does not fit). Vision input is not supported on
this path (text-only).

TODO(#77): record the exact verified `optimum-intel` + `nncf` versions and
HF-parity validation results (CPU + iGPU, tool-calling) once run on
hardware; until then treat this page as the recipe, not a validation
record.

## Why `cascadia shard` rejects it

The multi-stage exporter builds dense decoder layers with one SDPA + KV
cache per layer. This architecture has recurrent DeltaNet state on 30 of
40 layers, a 256-expert router + shared expert, mRoPE, and an MTP head —
none of which the generic export path models. `is_moe_config()` rejects it
config-first (nested `text_config` unwrap + `qwen3_5_moe` model_type,
\#77) with a pointer to the `ov-genai` path above.

The pipeline-parallel design (the actual too-big-for-one-box story) is
specced separately in [qwen36-moe-support.md](qwen36-moe-support.md) and
deferred behind its M1 feasibility measurements.
