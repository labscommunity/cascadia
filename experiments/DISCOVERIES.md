# Discoveries

Novel / surprising / undocumented findings worth saving forever. Format: cite the experiment that produced the evidence, give the surprise plainly, explain why it's saveable.

## D1 — OV 2026.1 PagedAttention transform requires optimum-cli-shaped IRs and is not retrofittable to per-stage trace exports

**Setup:** OpenVINO 2026.1.0 + openvino-genai 2026.1.0 on alpha (Battlemage Arc B390). Llama 3.1 8B Instruct INT4, exported per-stage via rainier's `scripts/export_cached_shards_v5.py` (torch.jit.trace + nncf + apply_make_stateful_transformation). Tried to apply `openvino._offline_transformations.paged_attention_transformation` at the end of the export to engage LLMPipeline-class GPU-plugin optimizations on multi-stage IRs.

**Finding:** Two stacked failures:

1. The transformation's first check is `!model->get_variables().empty()` — it requires a stateful model. The v5 script branched away from `apply_make_stateful_transformation` for the PA path (commented as "PA expects pre-stateful"). That branch was wrong for OV 2026.1.
2. After patching to apply stateful first, the next check is `unregistered_parameters.str().empty()` — the transform deletes `attention_mask` from the parameter list (PA absorbs masking) but leaves a graph reference, failing validation.

The OV error suggests `optimum-cli export openvino --task text-generation-with-past`. optimum-cli's exporter produces a model shape PA accepts — but it is monolithic, no per-stage support exists.

**Why this matters:** D7 in the prior python autolab ([branch autolab/intel-gpu-perf](https://github.com/labscommunity/tahoma/tree/autolab/intel-gpu-perf)) hoped PA was a config flag. It is not. Reclaiming the LLMPipeline per-stage win on multi-stage requires either custom torch.export flows or forking optimum-cli — multi-week. **For the current Llama-style trace-based per-stage export, runtime PA optimizations are unreachable.** This re-prioritizes the moonshot stack toward async pipelining and per-host TP, away from per-stage compute optimization via PA.

**Source experiments:** `experiments/e9-paged-attention/`.
