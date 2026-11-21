# Engine deep-dives

Cascadia ships nine engines (`cascadia engines` lists them). The
OpenVINO pipeline engines, the robot-policy engine and the external
llama.cpp engine are documented here:

- [`ov-genai.md`](./ov-genai.md) — single-stage
  `openvino_genai.LLMPipeline`; FastDraft + Prompt Lookup decoding.
- [`ov-runtime.md`](./ov-runtime.md) — multi-stage stateful KV cache
  over pre-exported per-stage shards.
- [`ov-dist-spec.md`](./ov-dist-spec.md) — multi-stage distributed
  speculative decoding with mask-based KV rewind.
- [`sycl-llama.md`](./sycl-llama.md) — external llama.cpp SYCL engine
  (GGUF via a spawned `llama-server`); `--elastic` = device-side weight
  streaming.
- [`ov-policy.md`](./ov-policy.md) — robot models (VLA policies): one
  model per process, `cascadia run` only, behind the sibling
  `PolicyEngine` trait rather than `Engine`; serves
  `/v1/policy/metadata`, the openpi WebSocket `/v1/policy/ws` and,
  with `--zenoh`, Intel's physicalai protocol.

The remaining engines are documented with their model families under
[`../architectures/`](../architectures/):

- `gemma4` — [`../architectures/gemma4-support.md`](../architectures/gemma4-support.md)
- `sparse-moe` — [`../architectures/minimax-m2.md`](../architectures/minimax-m2.md)
  (MiniMax-M2 pipeline) and [`../architectures/moe.md`](../architectures/moe.md)
  (MoE family background)
- `qwen35` (alias `qwen36-moe`) — [`../architectures/qwen36-moe-support.md`](../architectures/qwen36-moe-support.md), [`../architectures/qwen3.8.md`](../architectures/qwen3.8.md)

`mock` (deterministic word-echo test engine) needs no deep-dive; see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).
