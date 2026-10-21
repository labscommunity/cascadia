# gptq_builder — Qwen3.5-MoE GPTQ-Int4 checkpoint → OpenVINO VLM-layout IR, without a float model

Builds the same IR layout Intel publishes (`OpenVINO/Qwen3.5-*-int4-ov`: `openvino_language_model.xml`
+ text-embeddings + vision/merger IRs + tokenizer IRs) straight from a `Qwen/Qwen3.5-*-GPTQ-Int4`
checkpoint. Peak RAM ≈ 0.6 bytes/param (the int4 constants) instead of the ~6 bytes/param the stock
`optimum-cli export openvino` path needs (bf16 load + fp16 intermediate + nncf). The 397B-A17B fits in a
400 GiB machine; the official pipeline needs ~2.4 TB.

Why not the stock path: transformers 5.2 + gptqmodel load GPTQ checkpoints of fused-MoE Qwen3.5 as
RANDOM weights (per-expert `qweight/scales` are "unexpected", fused `experts.gate_up_proj` is "missing"),
silently. See docs/architectures/qwen35-gptq-ir-builder.md.

## Steps

1. **Structural template** (needs torch + transformers==5.2.0 + optimum-intel main; ~45 GB RAM for 397B):

       python template_export.py <gptq_ckpt_dir> <template_dir> 16

   Exports the real config with `moe_intermediate_size` shrunk to 16, `mtp_num_hidden_layers=0`, every
   parameter randomised (literal 0/1 parameters make OpenVINO fold `x*1` away and the graph degenerates),
   fp16 constants, no compression. Only the graph structure is used.

2. **Build** (openvino + numpy + nncf only; no torch):

       python gptq_to_ov.py --template <template_dir> --ckpt <gptq_ckpt_dir> --out <ir_dir> --compress int4

   Every constant is replaced: expert weights are repacked from GPTQ (sym, g128, zp=8) into
   `u4 [E,out,groups,128] → Convert → Subtract(zp) → Multiply(scale) → Reshape` (Intel's exact pattern,
   no re-quantisation); all other weights are copied bf16→f16 (derived constants handled: conv1d 4-D
   reshape, `exp(A_log)`, `dt_bias`, zero-centred norms `1+w`, gated norm `w`, vision biases/LayerNorm
   from consumer module paths) and then compressed by nncf (int4 asym g128, Intel's recipe).
   `--skip-nonexpert --skip-aux --layers 0,1` with an official IR as `--template` splices GPTQ experts
   into Intel's IR for A/B tests. `--dry-map` reports the constant mapping only.

3. **Check**: `gen_check.py <ir_dir>` (greedy text), `compare_irs.py <ir_A> <ir_B>` (logits + greedy parity),
   or OpenVINO GenAI `VLMPipeline` for image+text. Then `../export_qwen36_moe.py --model <ir_dir> --out
   <stages> --total N --validate` cuts stage shards for `cascadia run --engine qwen36-moe`.

Validated 2026-10-08 on Qwen3.5-35B-A3B: text output identical to the official Intel IR
(" Paris.\nThe capital of France is Paris.\nThe"), image+text coherent through VLMPipeline.
