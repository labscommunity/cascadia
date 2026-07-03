# Engine deep-dives

Cascadia ships seven engines (`cascadia engines` lists them). The three
OpenVINO pipeline engines are documented here:

- [`ov-genai.md`](./ov-genai.md) — single-stage
  `openvino_genai.LLMPipeline`; FastDraft + Prompt Lookup decoding.
- [`ov-runtime.md`](./ov-runtime.md) — multi-stage stateful KV cache
  over pre-exported per-stage shards.
- [`ov-dist-spec.md`](./ov-dist-spec.md) — multi-stage distributed
  speculative decoding with mask-based KV rewind.

The remaining engines are documented with their model families under
[`../architectures/`](../architectures/):

- `gemma4` — [`../architectures/gemma4-support.md`](../architectures/gemma4-support.md)
- `sparse-moe` — [`../architectures/minimax-m2.md`](../architectures/minimax-m2.md)
  (MiniMax-M2 pipeline) and [`../architectures/moe.md`](../architectures/moe.md)
  (MoE family background)
- `qwen36-moe` — [`../architectures/qwen36-moe-support.md`](../architectures/qwen36-moe-support.md)

`mock` (deterministic word-echo test engine) needs no deep-dive; see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).
