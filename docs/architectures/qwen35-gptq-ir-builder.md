# Qwen3.5-MoE: building the OpenVINO IR from a GPTQ-Int4 checkpoint (397B-class without a float model)

Status (2026-10-09): 35B-A3B validated end to end (text identical to Intel's official IR, image+text
coherent). **397B-A17B built and validated**: 204 GB language-model IR (+1 GB embeddings, +0.9 GB
vision/merger) in 74 min on a 400 GiB guest, peak ~160 GB RAM; `gen_check` on the full model (CPU, 8 vCPU):
compile 435 s, 5-token prefill 16.6 s, top-1 ` Paris` (17.9 vs 14.8 runner-up), 12 greedy tokens in 8.6 s =
` Paris.<|im_end|>\n<|im_start|>assistant\n<think>\nThinking Process:` (dense-bmm over all 512 experts).
Six-stage cut with `export_qwen36_moe.py --total 6 --validate`: chain-vs-full logits rel 4.1e-3, top-1 match,
top-5 5/5, 8-token greedy parity 8/8; shards 32 GB + 5 × 42 GB. For 64 GB CPU ranks cut ~10 stages (the CPU
plugin holds ~2× the IR in RSS) or place stages on the iGPU. Tools: `tools/qwen36_surgery/gptq_builder/`.

## Why

No OpenVINO IR exists for any ≥100B VLM. The stock exporter (`optimum-cli export openvino`) loads the bf16
model in torch (2 B/param), traces it, writes a full fp16 intermediate to `$TMPDIR`, then re-quantises with
nncf: for Qwen3.5-397B-A17B that is ~800 GB to load, ~2.4 TB peak RAM, ~800 GB of scratch disk. The
pre-quantised `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` checkpoint (236 GB) does not help: transformers 5.2 +
gptqmodel cannot map per-expert GPTQ tensors into the fused `Qwen3_5MoeExperts` parameters, drops them as
"unexpected", leaves the fused params "missing" (random), and also converts attention / shared-expert
layers to quant-linear shape so their bf16 weights are dropped too. The export "succeeds" and generates
`!!!!…`. Always validate an exported IR by generation.

## How

1. A *structural template*: optimum-intel export of the real config with `moe_intermediate_size=16`,
   MTP off, random fp32 weights (**all** parameters randomised: Qwen3.5's zero-centred RMSNorm
   `(1+w)·x` with `w=0` folds to `x·1`, which OpenVINO eliminates, so the un-randomised template loses
   the norm multiplies; the gated norm `w·x` with `w=1` likewise), fp16 constants, no compression.
2. `gptq_to_ov.py` replaces every constant of the template:
   - experts: GPTQ `qweight [in/8,out]` (8 nibbles/word, low nibble first), `scales [in/128,out]`,
     `qzeros=0x88888888` (zp=8), trivial `g_idx` → transposed to `[out,in]`, grouped `[E,out,in/128,128]`,
     packed u4 (OpenVINO: low nibble = first element), zero point u4 8, scales f16 `[E,out,groups,1]`,
     `Reshape [E,out,in]` — exactly Intel's chain (`…/VariadicSplit.0` = gate, `.1` = up, `experts.down_proj`).
     Verified bit-exact against the GPTQ reference dequantisation; gate/up/down correlate 0.98–0.99 with
     Intel's independently quantised experts.
   - everything else: bf16→f16 1:1 by name (template names carry a `_compressed` suffix), plus derived
     constants: conv1d weight reshaped to `[C,1,1,4]`, `exp(A_log)` (the negation stays a graph op),
     `dt_bias` `[1,1,H]` (consumer `linear_attn/aten::add`), decoder/attention/final norms `1+w`,
     `linear_attn.norm` plain `w`, vision linear biases / LayerNorm weight+bias / patch-embed conv bias
     resolved from the consumer's `__module.<path>/aten::<op>` name; `self.weight` of the embeddings IR →
     `embed_tokens.weight`; `pos_embed`, `patch_embed` prefixes. The rotary `inv_freq` table is computed
     from config and left alone.
   - nncf `INT4_ASYM`, group 128, ratio 1.0 on the remaining f16 MatMul weights (Intel's recipe); the
     already-integer expert constants are untouched.
3. Tokenizer IRs via `openvino_tokenizers`; configs copied (quantization_config removed, MTP 0).
4. Stage shards for `--engine qwen36-moe`: `tools/qwen36_surgery/export_qwen36_moe.py` (config-driven).

## Facts worth knowing

- **Dead expert neurons.** Qwen3.5-MoE has a large fraction of dead intermediate units: 41% of
  layer-0 gate/up rows sit at nncf's clamp floor in Intel's own bf16-derived 35B IR; the GPTQ checkpoints
  show 55–59% (35B) and 78–92% (397B) all-zero scales, with the matching `down_proj` columns exactly at
  the zero point. Zero scales dequantise to exactly 0, so the build is faithful; a sparse per-expert
  engine could skip these rows (effective expert width ≈ 10–45%).
- Version matrix for the template export: transformers **5.2.0** exactly (optimum-intel main imports
  `Qwen3_5DynamicCache`, removed in 5.6; its Qwen3.5 configs cap at 5.2.*), optimum-intel git main,
  openvino ≥ 2026.4. gptqmodel is not needed by the builder.
- `ov.op.Constant(ov.Tensor(packed_bytes, shape, u4))` shares the numpy buffer by default; keep the
  buffers alive or pass `shared_memory=False`, or the saved zero points are garbage.
- The GPU plugin's `MOECompressed` pass rejected a build whose zero points were garbage; retest on GPU
  after that fix is pending.
- The staged engine executes the official dense-bmm MoE graph (every token runs all experts): ~1 tok/s
  single-stream projected for the 397B across five 64 GB ranks; a sparse per-expert engine is the known
  follow-up and slices the same `[E,…]` constants.
