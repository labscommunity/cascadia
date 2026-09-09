# Architecture support

Status of decoder-only causal LMs in Cascadia's `cascadia shard` exporter
and the OpenVINO Rust runtime, as of #69 (partial rotary + config-first
rejection) and #48 (Gemma 4 exporter).

The authoritative table — what works, what regresses, what is rejected
— lives in [`../SHARDING.md`](../SHARDING.md#supported-architectures).
This directory holds per-family deep-dives for non-trivial cases.

## Per-family notes

- [`gemma4-support.md`](./gemma4-support.md) — Gemma 4 (April 2026):
  what makes it different from Gemma 3, the dedicated
  `tools/export_gemma4.py` exporter (#48) that `cascadia shard`
  auto-dispatches to, and the `gemma4` engine that serves the shards.
- [`phi.md`](./phi.md) — Phi-3 / Phi-4 family: partial rotary (handled,
  #69), LongRoPE (soft-dropped), and sliding window.
- [`moe.md`](./moe.md) — Mixtral, Qwen3-MoE, Llama 4, gpt-oss,
  GraniteMoE, Hunyuan: why the generic exporter can't ship them,
  and how `cascadia-engine-sparse-moe` (Kimi K2.6) hints at the
  shape of a fix.
- [`mistral.md`](./mistral.md) — Mistral 7B / NeMo / Small 3.x: the
  `mistral` path, sliding-window handling, and the `text_config` unwrap
  for `mistral3` multimodal wrappers.
- [`r1-distill.md`](./r1-distill.md) — DeepSeek R1 Distills (Qwen / Llama):
  they ride the base qwen2 / llama paths; pipeline-parallel recipe.
- [`qwen3.6.md`](./qwen3.6.md) — Qwen3.6-35B-A3B single-stage support:
  hybrid GatedDeltaNet + MoE facts, serving paths, hardware validation.
- [`qwen36-moe-support.md`](./qwen36-moe-support.md) — Qwen3.6 staged
  serving: IR-surgery exporter, the `qwen35` engine (formerly
  `qwen36-moe`), and multi-node pipeline mode with acceptance gates.
- [`qwen3.8.md`](./qwen3.8.md) — Qwen3.8-27B (dense `qwen3_5`): the same
  surgery + staged engine, the `ov-genai` single-stage path, Panther Lake
  iGPU/CPU throughput and context-capacity measurements, fine-tune
  (Qwopus) export recipe.
- [`minimax-m2.md`](./minimax-m2.md) — MiniMax-M2 on the sparse-MoE
  engine: export pipeline, quantization configs, measured throughput.

## How to add a new family

1. Find the model's `config.json` (HuggingFace download or `cat` on a
   local snapshot). Note `model_type`, `architectures`, and any
   `rope_scaling` / `rope_parameters` / `layer_types` fields.
2. Check whether `transformers.models.<model_type>.modeling_<model_type>`
   has a `<Foo>DecoderLayer` class. If so, the change to
   `tools/export_shards.py::get_decoder_layer_cls()` is one branch.
3. Trace through the quirks table in `SHARDING.md` to see which apply.
   For each dropped or rejected feature the new family relies on, the
   exporter needs an explicit branch (and `check_export_quirks` / `is_moe_config`
   may need updating) — and the Rust runtime
   (`crates/cascadia-engine-openvino/`) may need a matching change.
4. Add an e2e test: `cascadia shard --model <hf-id>
   --output-dir /tmp/shards --num-stages 2` on an export host, then
   `cascadia worker --engine ov-runtime --rank 0` on that host and
   `--rank 1` on an AI PC. Hit `/v1/chat/completions` and inspect a
   short generation.

## Reference

- Cascadia's exporter is `tools/export_shards.py`, driven by
  `cascadia shard`. Architecture-specific exporters (e.g.
  `tools/export_gemma4.py`) live beside it and are auto-dispatched
  on `model_type` — use them as the template when porting a new
  family that doesn't fit the generic path.
- [OpenVINO GenAI supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)
  is the upstream list of what `openvino_genai::LLMPipeline` can load
  via `optimum-cli`. Anything on that list is a candidate for the
  `ov-genai` single-stage engine without needing changes to our
  exporter.
