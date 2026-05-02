# c3: ov-genai engine in tahoma

## Result

| ID | Path | Decode tok/s | vs raw c1-1 (96.41) |
|---|---|---|---|
| c3-1 | `tahoma worker --engine ov-genai` on alpha B390 | **87.1** | -10% |

The 10% wrapping cost is from:
- The CLI startup overhead for the first request (still JIT-compiling kernels — c1-1 was a fresh process too, so this isn't new).
- Going through `Runner.generate` → `Engine.step` → `pipe.generate` instead of calling `pipe.generate` directly.
- A small first-token detokenisation overhead in the engine's full-text return.

87 tok/s is still **9.8× over the ov-optimum baseline of 8.89**. The discovery is now consumable through tahoma's normal CLI:

    tahoma worker --rank 0 --total 1 --engine ov-genai --device GPU \
                  --model C:\cascadia\models\llama-3.1-8b-int4

## What landed in tahoma

- New file: `tahoma/worker/engines/openvino/genai_engine.py` (no edits to existing engines).
- Registry entry `ov-genai` in `engines/registry.py`.
- New CLI flags `--ov-cache-dir`, `--ov-kv-precision`, `--ov-dyn-quant-group` (used by ov-genai only).

## Open follow-ups

- **Per-token streaming** via the LLMPipeline streamer callback — currently we yield ONE chunk with the full text. SSE on the API would benefit from per-token chunks.
- **SchedulerConfig** for continuous batching + prefix caching (multi-turn chat win).
- **Speculative decoding** once we have an LLMPipeline-compatible draft IR (c2 blocked on this).
