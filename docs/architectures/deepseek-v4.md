# DeepSeek-V4-Flash (`deepseek_v4`)

DeepSeek-V4-Flash (~250B MoE, ~160 GB) exported and run distributed across
4× 32 GB Intel AI-PCs via the sparse-moe engine — one pipeline stage per node.
The checkpoint ships a standalone reference implementation (no HF-transformers
modeling file), so the shell is implemented directly against DeepSeek's
`inference/` reference.

**Status:** implemented and running distributed; shell, exporter, and pipeline
are complete and golden-tested. See [Performance](#performance).

## Architecture

| Param | Value |
|---|---|
| layers | 43 (+1 MTP block) |
| hidden | 4096, vocab 129,280 |
| experts | 256 routed (FP4 e2m1) + 1 shared, top-6, `moe_inter=2048`; first 3 layers hash-routed by token id |
| routing | `sqrtsoftplus` scores, `noaux_tc` bias, norm-topk, scale 1.5 |
| attention | MLA-style MQA-on-latent: one 512-dim KV latent/token (`wkv 4096→512`), 64 q-heads × 512, rope on the last 64 dims |
| q / o path | `wq_a 4096→1024` → RMSNorm → `wq_b 1024→64×512`; grouped low-rank output (8 groups, `wo_a`→`wo_b`), inverse-rope on the attn output |
| KV regime | sliding window 128 raw + learned KV compression beyond it (per-layer `compress_ratios`) |
| sparse attn | ratio-4 layers: DSA Indexer (Hadamard-rotated FP4-sim scoring) → top-512 positions; per-head `attn_sink` |
| residual | Hyper-Connections: hidden is `[b, s, 4, 4096]` (4 copies), Sinkhorn-normalised mix (20 iters) per sublayer |
| rope / quant | YaRN ×16 (theta 160000 compressed / 10000 window), 1M ctx; FP8 e4m3 block-128 shell, FP4 e2m1 experts, ue8m0 scales |

Weights ≈ 129 GB FP4 experts + ~31 GB FP8 attention/shared/embed; ~6+1 experts
active/token → streamable at ~30 GB/node.

**Pipeline-parallel wire specifics:** the inter-stage activation is
`[b, s, 4, 4096]` (HC copies — 4× the usual width), and `input_ids` must
accompany activations to every stage (the hash gate reads raw token ids at
layers 0–2).

## Implementation

- **Exporter** (`tools/export_deepseek_v4.py`) — FP4/FP8 dequant + per-expert
  int4 and shell safetensors; `--tiny/--med/--big` synthetic modes build
  deterministic models for the correctness harness.
- **CPU reference** (`tools/deepseek_v4_ref/`) — the upstream `inference/`
  model with pure-torch kernels; a local, deterministic ground truth.
- **Rust shell** (`crates/cascadia-engine-sparse-moe/src/dsv4/`) — HC-Sinkhorn
  mixing, windowed + learned-compression KV, DSA indexer, attention sinks, and
  grouped low-rank O. None is OpenVINO-traceable, so it is implemented directly
  in Rust, mirroring the CPU reference 1:1. Experts run as mmap int4 (Rust,
  default) or optional per-expert OpenVINO IR.
- **Pipeline** (`src/engine.rs`, `src/dist.rs`) — rank 0 embeds + drives, mid
  ranks relay the HC-copy hidden state, the last rank runs the head + sampler.

## Performance

Steady-state 4-node decode is **compute-bound** — experts ~50%, attention
`proj` ~33%, inter-node transport ~16–30% (a relayed link the engine can't
shorten). The default (Rust mmap expert) decode path was optimized ~2×:

- **Fused AVX2 int4 expert kernel** — the int4 nibble decode is fused straight
  into the dot product (SIMD unpack → FMA, no f32 scratch row) instead of a
  scalar unpack followed by a separate dot. Experts are the dominant decode
  cost; ~1.8× on its own.
- **Parallel grouped `o_proj`** — the `wo_a` mid-GEMV ran serially on one core;
  its independent rows are spread across cores (bit-identical). ~+10%.
- **Prefill is per-token, not streamed** — `Dsv4Engine` forwards the prompt one
  token per frame across the ranks, each a blocking round-trip with a bounded
  reply deadline (the same wire path decode uses); a mid-prefill peer death
  fails fast rather than leaving a silent KV hole. The one-way streamed
  `ForwardPrefill` frame belongs to the K2.6 sparse-MoE engine and is **not**
  wired into dsv4. The largest prefill win would be a direct node↔node LAN
  rather than a relayed link.

Evaluated and **rejected with measurement** (recorded so they aren't re-tried):
FP8-e4m3 shell (bit-exact, but `proj` is compute- not bandwidth-bound after the
parallelization → no gain), MTP / n-gram speculative decode (≤~1.1× at this
compute/transport split), and a batched-shell kernel (no benefit to
single-stream autoregressive decode). The largest remaining win is a direct LAN
between nodes — a deployment change, not engine work.

## Validation

- **Dequant** — byte-exact against DeepSeek's `inference/convert.py` +
  `kernel.py` on a real shard (FP8 shell and per-expert FP4 → int4).
- **Rust shell** — the synthetic `--med/--big` harnesses (real structural dims:
  head_dim 512, o_groups 8, YaRN, ratio-128 compressor, hash layers) match the
  CPU reference greedy, isolating shell logic from the experts.
- **Wire / pipeline** — 2-, 3-, and 4-rank chains over real `cascadia-transport`
  match the single-process reference exactly (including the two-mid-relay
  4-rank topology); mmap and eager expert dispatch agree.
