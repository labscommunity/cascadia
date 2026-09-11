# Inkling (Thinking Machines Lab) on the sparse-MoE engine

Status: **engine + exporter + tests landed on `feat/inkling`; real-model
export and hardware validation in progress** (this page is updated as the
numbers land).

Inkling is Thinking Machines Lab's open-weights (Apache-2.0) mixture-of-experts
family, released 2026-07-15. Cascadia runs the text model of both sizes through
the sparse-MoE engine's Rust-shell path (`--engine sparse-moe`, family selected
by `manifest.json` `arch: "inkling"`), beside GLM-5 and DeepSeek-V4.

| | **Inkling** | **Inkling-Small** |
|---|---|---|
| HF repo | `thinkingmachines/Inkling` | `thinkingmachines/Inkling-Small` |
| total / active params | 975B / 41B | 276B / 12B |
| decoder layers | 66 | 42 |
| hidden | 6144 | 4096 |
| attention (global layers) | 64 q-heads / 8 kv-heads, head_dim 128 | 32 / 8, 128 |
| attention (sliding layers) | 64 / 16, window 512 | 32 / 8, window 512 |
| layer pattern | 55 sliding + 11 global (`local_layer_ids`; every 6th global) | 35 + 7 |
| MLP | layers 0–1 dense SwiGLU (24576); layers 2+ MoE | dense 16384; MoE |
| MoE | 256 routed experts, top-6, + 2 shared; expert width 3072 | 256, top-6, +2; 2048 |
| positional scheme | **no RoPE** — learned relative-position bias (`d_rel` 16, extent 1024 global / 512 sliding) + log-scaled queries on global layers | same |
| convolutions | causal depthwise short convs (kernel 4) on k, v, attention output, MLP output | same |
| vocab | 201024 (logits sliced to 200058) | same |
| context | 1,048,576 | same |
| checkpoint | 109 bf16 safetensors, 1.905 TB (+ `mtp.safetensors`) | 33 files, 532 GB |
| Cascadia int4 artifact | ~525 GB (routed experts int4 group-32 ≈ 490 GB; shells, shared experts, dense, embed/unembed bf16 ≈ 36 GB) | ~150 GB |

Multimodal encoders (`model.visual.*` hMLP image patches, `model.audio.*` dMel
audio) and the 8-layer MTP draft head (`model.mtp.*`) are dropped by the
exporter: text-only, greedy/sampled decode.

## What the port implements

`crates/cascadia-engine-sparse-moe/src/inkling/` — semantics from transformers'
native `modeling_inkling.py` (5.16+), which is also the test oracle:

- `conv.rs` — `ShortConv`: torch `conv1d(padding=K-1, groups=C)[:seq]` with the
  residual added inside; f32; per-channel history ring for decode; `prefill`
  bit-identical to sequential `decode`.
- `relpos.rs` — `RelPos`: `bias(h, dist) = r_h · proj[:, dist]` for
  `0 <= dist < extent`, else 0 (`r = W_r h`, one `d_rel` vector per head).
- `attn.rs` — GQA with per-head q/k RMSNorm, k/v short convs, `1/D` scaling
  (q/k are RMS-normalised, hence not `1/√D`), the relative bias, sliding
  (`p - j < 512`, a `window + rewind` ring per kv head) or global keys
  (`max_seq` rows), log scaling `τ = 1 + α·ln(max((p+1)/n_floor, 1))` on q and
  bias for global layers. bf16 projection weights, f32 accumulate, bf16
  write-back after each linear (the GLM convention); softmax/accumulate f32.
- `gate.rs` — `inkling_gate`: sigmoid scores; top-k chosen on `σ + bias`
  (lower expert id wins ties); weights `σ_i / (Σ_selected σ + Σ_shared σ) ·
  route_scale · global_scale`, and the same normalisation gives each shared
  expert its gamma.
- `moe.rs` — `MoeLayer` (routed experts + the two shared experts as separate
  `AnyExpert`s with their gammas; batch-union prefill visits each expert once)
  and `DenseMlp` (× `global_scale`). Experts reuse the glm/dsv4 int4 group-32
  mmap kernels (AVX-512 → AVX2 → scalar dispatch).
- `model.rs` — `Layer` (pre-norm → attention → attn conv → residual; post-norm →
  MLP → mlp conv → residual), `Model` (embed → embed RMSNorm → layers → norm →
  `/ logits_mup_width_multiplier` → unembed → slice to `unpadded_vocab_size`).
- `loader.rs` / `stage.rs` — the `export_inkling.py` layout and the
  `StagedRunner` (even layer split, batched prefill, position lock-step) that
  the generic `PipelineEngine` drives across N ranks.

Every piece of sequence state (KV rings, four conv histories per layer)
supports `reset`, `truncate` (speculative-decode rewind; `DEFAULT_REWIND = 32`
rows of slack on the sliding rings) and `snapshot`/`restore` (prefix cache).

## Export layout (`tools/export_inkling.py`)

```text
<out>/
  manifest.json                 arch "inkling" + the fields in loader.rs::InklingManifest
  embed.safetensors             embed.weight (bf16), embed_norm.weight (f32)
  head.safetensors              unembed.weight (bf16), norm.weight (f32)
  shells/layer_NN.safetensors   attn.{wq_du,wk_dv,wv_dv,wr_du,wo_ud}.weight (bf16),
                                attn.{q_norm,k_norm}.weight, attn.{k,v}_sconv.weight [C,4],
                                attn.rel_logits_proj.proj [16, extent], attn_norm.weight,
                                attn_sconv.weight, mlp_norm.weight, mlp_sconv.weight,
                                MoE: mlp.gate.{weight [258,H], bias [256], global_scale}
                                dense: mlp.global_scale                          (all f32)
  experts/layer_NN/expert_EEE.bin       int4 group-32 (gate, up, down) — w13 de-interleaved
  experts/layer_NN/expert_shared{0,1}.bin
  experts/layer_NN/dense.bin            layers 0-1
```

Tensor names are the checkpoint's own with the `model.llm.[layers.N.]` prefix
stripped. The fused `w13` tensors interleave gate/up rows (row `2i` = gate,
`2i+1` = up — transformers' `Interleave(dim=1)`); the exporter de-interleaves.

Modes: `--validate config.json` (contract check, prints the derived manifest),
`--tiny OUT` (synthetic tiny model built with transformers' own
`InklingForCausalLM`, exported, plus `reference.json` = HF's greedy tokens on
the int4-dequantised weights), `--model DIR --out OUT` for a checkpoint. The
real export streams: `--skip-missing-shards` processes whatever shards have
downloaded, `--delete-consumed-shards` frees a shard once every tensor it holds
is written, re-runs are idempotent (per-tensor temp-then-rename, per-layer done
markers), `--layers-done-check` asserts completeness. That is how the 1.9 TB
checkpoint is converted on a box with 900 GB of scratch.

## Validation

Tier 1–4 of the family test ladder (all synthetic, no downloads, `cargo test
-p cascadia-engine-sparse-moe --test 'inkling_*'`):

1. **Primitive goldens vs HF** (`inkling_conv/relpos/gate/attn`): element-wise
   1e-4 on the f32 paths; attention layers (sliding layer 0, global layer 3
   with log scaling active) within bf16 write-back tolerance.
2. **Model parity** (`inkling_model`): per-layer hidden states ≤ 0.1 % of row
   scale vs HF float32 on the tiny model; 8/8 greedy ids exact.
3. **Loader round-trip** (`inkling_loader`): `load_model` on the `--tiny` export
   reproduces HF's greedy ids computed on the same int4-dequantised weights;
   the staged runner (token-by-token and batched prefill) matches.
4. **Wire** (`inkling_wire`): a 2-rank chain over the real loopback
   `cascadia-transport` matches the single-process reference.

Python: `tools/tests/test_inkling_export.py` (contract, de-interleave vs
transformers' op, tiny round-trip).

Real model (miner, Xeon Gold 6252 48T / 172 GB / SATA SSD scratch): _pending —
export streaming in progress; per-layer parity against checkpoint slices and
factual prompts end to end follow._

## Serving

```bash
# export (streams the HF download through int4 quantisation)
python tools/export_inkling.py --model /path/to/Inkling --out /data/inkling-int4 \
    --skip-missing-shards --delete-consumed-shards
cp /path/to/Inkling/tokenizer.json /data/inkling-int4/

# single box
cascadia run /data/inkling-int4 --engine sparse-moe --api :8000
# N ranks: start the last rank first, as for glm5
cascadia worker --rank 1 --total 2 --engine sparse-moe --model /data/inkling-int4 --listen :9100
cascadia worker --rank 0 --total 2 --engine sparse-moe --model /data/inkling-int4 --next host:9100 --api :8000
```

Knobs: `CASCADIA_INKLING_MAX_SEQ` (global-layer KV rows; default 4096 — the
sliding layers' rings are fixed at 512 + 32), `CASCADIA_INKLING_EXPERTS=eager|mmap`
(default mmap for real-sized expert sets), `--max-seq`, `--experts-mode`.

## Sizing and the hardware honesty note

Per generated token the engine touches ~41B active parameters: ~21.7B routed
expert weights (int4, ~11 GB of reads), ~7.2B shared-expert weights, ~8.7B
attention weights and the dense/embed/unembed tables (bf16, ~36 GB resident).
RAM-resident at 100 GB/s that is roughly 0.3–0.5 s/token; with the 490 GB of
routed experts paged from disk the routed reads dominate and the SATA SSD on
the validation box caps decode at well under 0.1 tok/s. A "record" run needs
the int4 artifact (~525 GB) resident: a ≥640 GB-RAM box, or an N-rank pipeline
whose ranks together hold it.

## Open follow-ups

- Per-rank KV-prefix cache and the qwen35-style in-process prefix cache (TTFT).
- MTP draft head (exported? no — dropped) / n-gram speculative decode: the
  rewind slack is in place.
- Vision / audio inputs (encoders dropped).
- Hot/cold expert residency (`CASCADIA_GLM5_HOTCOLD` port) for paged runs.
