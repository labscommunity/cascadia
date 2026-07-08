# DeepSeek-V4-Flash (`deepseek_v4`) — sharding scope

Goal: export a full DeepSeek-V4-Flash checkpoint on a big-RAM Linux box, shard
it across 4× 32 GB Intel AI-PCs, and run it distributed via the sparse-moe
engine pattern.

Ground truth used (2026-07-07): `deepseek-ai/DeepSeek-V4-Flash` —
`config.json`, `model.safetensors.index.json` (69,187 tensors, 46 shards,
~160 GB), and the repo's reference implementation
(`inference/model.py`, 827 lines; `inference/kernel.py`, 536 lines of
tilelang kernels). There is **no HF-transformers modeling file** — the repo
ships a standalone reference impl, so `AutoModelForCausalLM` does not load it
(transformers 4.57 has no `deepseek_v4`).

## Shape summary

| Param | Value |
|---|---|
| layers | 43 (+1 MTP block) |
| hidden | 4096, vocab 129,280 |
| experts | **256 routed (FP4 e2m1!) + 1 shared**, top-6, `moe_inter=2048` |
| routing | `sqrtsoftplus` scores, `noaux_tc` bias, norm-topk, scale 1.5; **first 3 layers hash-routed by token id** (`gate.tid2eid[vocab, 6]`) |
| attention | MLA-flavored **MQA-on-latent**: one 512-dim KV latent/token (`wkv: 4096→512`), 64 q-heads × 512, rope on last 64 dims only |
| q path | `wq_a 4096→1024` → RMSNorm → `wq_b 1024→64×512`, per-head RMS on q |
| o path | **grouped low-rank**: 8 groups, `wo_a` per-group 4096→1024 (FP8) einsum → `wo_b 8192→4096`; **inverse-rope applied to attn output** |
| KV regime | **sliding window 128** raw + **learned KV compression** beyond it; per-layer `compress_ratios = [0,0,4,128,4,128,…,4,0]` |
| sparse attn | ratio-4 layers use a **DSA Indexer** (64 idx-heads × 128, Hadamard-rotated FP4-sim scoring) → top-512 compressed positions; `attn_sink` per head |
| residual | **Hyper-Connections**: hidden state is `[b, s, 4, 4096]` (4 copies); every sublayer does Sinkhorn-normalised mix (20 iters) pre/post |
| rope | YaRN ×16 (theta 160000 on compressed layers, 10000 on pure-window layers), 1M context |
| quant | FP8 e4m3 block-128×128 (attn/shared), **FP4 e2m1 experts**, ue8m0 scales |
| extras | `swiglu_limit=10` clamps, MTP head, custom tokenizer encoding dir |

Weight budget: experts ≈ 256·40·3·(4096·2048) FP4 ≈ **129 GB** of the
160 GB; attention/shared/embed ≈ 31 GB FP8. Active/token ≈ 6+1 experts
→ genuinely streamable.

## Why this fits the sparse-moe strategy

The cascadia sparse-moe design (per-(layer,expert) artefacts + Rust shell,
OpenVINO only for expert MLPs or bypassed entirely via the int4 GEMM kernel)
fits this model well, same as Kimi K2.6:

- OpenVINO/optimum **cannot** export `deepseek_v4` whole — irrelevant, the
  shell never goes through OV.
- Experts are plain SwiGLU (with clamp) — trivially exportable per-expert
  (FP4→f32 dequant → int4 or OV IR), 6-of-256 streamed → ~30 GB/node disk,
  tiny RAM. The 4-way distribution math works (43 layers ÷ 4 ≈ 11/rank).
- Sliding-window-128 + compressed KV means **tiny KV caches** — great fit
  for 32 GB nodes.

## Why this is NOT "adapt rust_k26" — component delta

`rust_k26` implements classic V3-MLA (kv_lora absorbed attention, full
causal). V4-Flash's shell shares almost nothing with it:

| V4 component | Reference | In cascadia today | Work |
|---|---|---|---|
| FP8 block-128 dequant | `kernel.py act_quant/fp8_gemm` | ✅ `export_minimax_m2.py` has the same e4m3 block scheme | reuse |
| **FP4 e2m1 expert dequant** | `fp4_gemm` | ❌ (int4-gemm is signed-int4, not e2m1) | exporter dequant f32 → requant int4 (small) |
| per-expert export/stream/top-k dispatch | — | ✅ sparse-moe engine + M2 exporter | reuse/adapt |
| Gate: sqrtsoftplus + noaux_tc bias + **hash tid2eid** | `Gate` | ❌ (engine has sigmoid/softmax routers) | new, small — but **input_ids must flow to every stage** (wire change) |
| MQA-on-latent attention + q/kv RMSNorms + partial rope + **inverse-rope on output** | `Attention` | ❌ | new Rust kernel (moderate) |
| **attn_sink** + windowed **sparse gather attention** (online softmax over top-k idxs) | `sparse_attn_kernel` | ❌ | new Rust kernel (moderate) |
| **Compressor** — learned gated pooling (ratio 4 overlap / 128), APE, **stateful incremental decode compression** | `Compressor` | ❌ | new, intricate state machine (hard) |
| **Indexer (DSA)** — own compressor + Hadamard rotate + FP4-sim scoring + top-512 | `Indexer` | ❌ (needs a Hadamard transform too) | new (hard) |
| **Hyper-Connections** — hidden `[b,s,4,d]`, Sinkhorn(20) mixing every sublayer | `Block.hc_pre/hc_post`, `hc_split_sinkhorn_kernel` | ❌ | new (moderate math, **4× activation wire width**) |
| grouped low-rank O (`wo_a`/`wo_b`, o_groups=8) | `Attention` | ❌ | new (small) |
| YaRN dual-theta per layer-type | `precompute_freqs_cis` | partial (YaRN soft-dropped in generic exporter) | new (small) |
| MTP block | `MTPBlock` | ❌ | **skip** (greedy path needs main head only) |
| custom tokenizer encoding | `encoding/` | ❌ | evaluate; tokenizer.json exists so likely standard-loadable |

Pipeline-parallel wire implications (vs the generic f16 `hidden_states` path):
1. inter-stage activation is `[b, s, 4, 4096]` (HC copies) — **4× wider**;
2. `input_ids` must accompany activations to all stages (hash gate reads
   raw token ids at layers 0–2 — cheap but a protocol addition).

## Implementation layout

- **Exporter** (`tools/export_deepseek_v4.py`): FP4/FP8 dequant, 69k-tensor
  walk, per-expert int4 + shell safetensors artefacts. Synthetic
  `--tiny`/`--med`/`--big` modes build deterministic random-weight models
  (with an int4-round-tripped reference) for the correctness harness — no
  download or GPU needed.
- **CPU reference** (`tools/deepseek_v4_ref/`): the upstream `inference/`
  model with tilelang kernels replaced by pure-torch equivalents, so it runs
  on CPU and gives a local ground truth; incremental decode matches
  fresh-prefill token-for-token (the compressor decode state machine is
  consistent) and is deterministic across runs.
- **Rust shell** (`crates/cascadia-engine-sparse-moe/src/dsv4/`): HC-Sinkhorn
  mixing, windowed + learned-compression KV, the DSA indexer, attention
  sinks, and grouped low-rank O — none of which is OpenVINO-traceable, so it
  is implemented directly in Rust, mirroring the CPU reference 1:1 and
  golden-tested against fixtures it generates.
- **Pipeline** (`src/engine.rs`, `src/dist.rs`): `Dsv4Engine` drives the
  token-by-token distributed loop; rank 0 embeds + drives, mid ranks relay
  the HC-copy hidden state, the last rank runs the head + sampler. The
  hash-gate token ids stay on rank 0 (all hash layers fall in stage 0).

## Validation

- **FP8 / FP4 dequant**: byte-exact against DeepSeek's official
  `inference/convert.py` + `kernel.py` on a real shard (maxdiff 0.0 for both
  the FP8 shell scheme and per-expert FP4 → int4).
- **Rust shell math**: the synthetic `--med`/`--big` harnesses (real
  structural dims: head_dim=512, o_groups=8, `nope!=rope`, YaRN, ratio-128
  compressor, hash layers) match the CPU reference greedy, sharing the same
  int4-round-tripped experts so only shell logic is under test.
- **Wire / pipeline**: 2-, 3-, and 4-rank chains over a real loopback
  `cascadia-transport` match the single-process reference exactly, including
  the two-mid-relay 4-rank topology and the 64 KB real-width (hc·dim = 16384)
  forward frame; mmap and eager expert dispatch are bitwise identical.
- **Sampling**: the distributed driver must not sample on prefill-intermediate
  steps — the last rank only samples on the final prompt token and each decode
  step, so discarded prefill tokens never enter its repetition-penalty history
  (`FrameKind::ForwardNoSample`).
