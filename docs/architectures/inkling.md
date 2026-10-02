# Inkling (Thinking Machines Lab) on the sparse-MoE engine

Status: **engine + exporter + tests on `feat/inkling` (PR #154); the 975B
checkpoint is exported (512 GB int4) and validated layer-for-layer against
transformers on real weights; end-to-end serving measured on the miner** (see
Validation).

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
  mmap kernels (AVX-512 → AVX2 → scalar dispatch). After routing, a token's
  routed + shared experts run **concurrently** (each GEMV row-parallel on its
  own; the sum stays in gate order, so the result is bit-identical to a serial
  visit — `CASCADIA_INKLING_SERIAL_EXPERTS=1` restores that schedule), and the
  prefill block visits its unique experts the same way. An expert's bin is
  streamed whole into an owned buffer first (glm's overlapped read) only when
  its pages are **not** resident (a 64-page `mincore` sample per token);
  a resident expert is computed straight off the mapping, where the copy
  would cost ~2× (miner, warm: 28 → 14 ms per MoE layer).
  `CASCADIA_INKLING_SEQ_READS=1` forces the direct path everywhere.
- `model.rs` — `Layer` (pre-norm → attention → attn conv → residual; post-norm →
  MLP → mlp conv → residual), `Model` (embed → embed RMSNorm → layers → norm →
  `/ logits_mup_width_multiplier` → unembed → slice to `unpadded_vocab_size`).
- `loader.rs` / `stage.rs` — the `export_inkling.py` layout and the
  `StagedRunner` (even layer split, batched prefill, position lock-step) that
  the generic `PipelineEngine` drives across N ranks.

Every piece of sequence state (KV rings, four conv histories per layer)
supports `reset` (O(1): every position read is below `len` and was written
since the last reset or restore), `truncate` (speculative-decode rewind;
`DEFAULT_REWIND = 32` rows of slack on the sliding rings, bounded against a
write high-water mark so consecutive rewinds cannot creep past it) and
`snapshot`/`restore` (prefix cache).

**Serving through the API.** Inkling frames its output with *special* tokens
(`<|message_model|>`, `<|content_thinking|>`, `<|content_text|>`,
`<|content_invoke_tool_json|>`, `<|end_message|>`) that engines strip when
decoding. `cascadia-api` reads the marker ids off each chunk
(`MarkerDialect`, ids from `tokenizer_config.json`) and re-inserts the
textual delimiters every other served template uses — `<think>…</think>`
around the scratchpad and a Hermes `<tool_call>{"name","arguments"}</tool_call>`
for tool calls — in the non-streaming and streaming paths. The chat template
turns `reasoning_effort` into a numeric thinking level (`none` = 0 switches
thinking off, `high` = 0.9 is the default); the API probes at load that the
template distinguishes the OpenAI words and passes the caller's own word
through (an explicit `enable_thinking: true` beats a `none`). Its tool
declarations render with `tojson(sort_keys=true, separators=(",", ":"))`,
which the API's Python-compatible `tojson` honours.

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
the int4-dequantised weights; refuses to wipe a directory that is not its own
unless `--force`), `--model DIR --out OUT` for a checkpoint. The real export
streams: `--skip-missing-shards` processes whatever shards have downloaded,
every tensor is converted the moment its shard is readable (a layer's tensors
span shards 1..108, so per-unit consumption would need ~1.5 TB resident), a
unit whose shards are all present is written once directly while one with a
pending shard is staged and assembled later, `--delete-consumed-shards` frees
a shard once every tensor it holds is converted, re-runs are idempotent
(temp-then-rename, done markers), `--layers-done-check` asserts completeness.
That is how the 1.9 TB checkpoint is converted on a box with 900 GB of
scratch.

## Validation

Tier 1–4 of the family test ladder (all synthetic, no downloads, `cargo test
-p cascadia-engine-sparse-moe --test 'inkling_*'`):

1. **Primitive goldens vs HF** (`inkling_conv/relpos/gate/attn`): element-wise
   1e-4 on the f32 paths; attention layers (sliding layer 0, global layer 3
   with log scaling active) within bf16 write-back tolerance. The fixtures'
   relative-position bias is load-bearing (rms |bias| / rms |q·k/D| of 1.3–2.3
   per layer): zeroing it moves the hidden states 4–9 % of row scale against a
   2 % tolerance and the attention goldens by 6–13 %.
2. **Model parity** (`inkling_model`): per-layer hidden states ≤ 0.1 % of row
   scale vs HF float32 on the tiny model; 8/8 greedy ids exact.
3. **Loader round-trip** (`inkling_loader`): `load_model` on the `--tiny` export
   reproduces HF's greedy ids computed on the same int4-dequantised weights;
   the staged runner (token-by-token and batched prefill) matches.
4. **Wire** (`inkling_wire`): a 2-rank chain over the real loopback
   `cascadia-transport` matches the single-process reference.

Python: `tools/tests/test_inkling_export.py` (contract, de-interleave vs
transformers' op, tiny round-trip).

**Real-weight per-layer parity** (`examples/inkling_layer_dump.rs` +
`tools/inkling_ref/real_layer_parity.py`, exercised on the tiny export by
`tools/tests/test_inkling_real_parity.py`): the Rust example loads the first
`K` layers of an export, runs a token list through the decode path
(`forward_token`) and the batched prefill path and dumps every residual-stream
tensor; the Python side builds a `K`-layer `InklingForCausalLM` from the same
int4-dequantised weights (meta-device init, no head unless `K == num_layers`)
and reports max |diff| / row RMS, rms(diff) / rms and cosine per layer and
path. This is how the 975B export is validated layer by layer on a box that
cannot run HF end to end (`K = 3`, float32 ≈ 67 GB: layers 0–1 dense + the
first MoE layer). Tiny export, float32: worst element ≤ 0.26 % of its row RMS,
rms(diff)/rms ≤ 0.06 %, argmax 19/19, decode == prefill bit for bit.

**Real model — 975B export on the miner** (Xeon Gold 6252, 48 threads,
172 GB RAM, artifact on a SATA SSD; `hf download` 1.905 TB in 16-shard
batches at ~350 MB/s streamed through `export_inkling.py --skip-missing-shards
--delete-consumed-shards`, ~1.2 s per expert with 8 workers, whole export ≈ 2 h,
512 GB on disk):

`parity_run.sh 3` — prompt `The capital of France is` (5 tokens), Rust
`inkling_layer_dump` of layers 0–2 (two dense, one MoE; mmap experts; decode and
batched-prefill paths bit-identical) vs transformers on the same
int4-dequantised weights:

| layer | vs HF float32: rms(diff)/rms | max\|diff\| / max\|row\| | min cos | vs HF **bfloat16** (the model's native dtype): rms(diff)/rms |
|---|---|---|---|---|
| embed + embed_norm | 0.0000 | 0.0000 | 1.000000 | 0.0019 |
| 0 (dense, sliding) | 0.0031 | 0.0049 | 1.000000 | 0.0016 |
| 1 (dense, sliding) | 0.0016 | 0.0024 | 1.000000 | 0.0034 |
| 2 (MoE, sliding) | 0.0013 | 0.0012 | 1.000000 | **0.0311** (cos 0.99982) |

Verdict PASS against float32 (criteria: rms ≤ 1 %, max|diff| ≤ 1 % of the
row's max). The absolute worst diffs (4.1 / 2.5 / 9.6) sit on the residual
stream's massive-activation dims (row max ≈ 830 where row RMS ≈ 11) at one
bf16 ULP — the shell's bf16 write-back. The last column is the yardstick: the
model's own bfloat16 execution deviates from float32 by 3.1 % rms on the MoE
layer, ~25× more than the Rust shell does.

**End to end** (`smoke.sh`: `cascadia run out --engine sparse-moe --api :8010`,
mmap experts, `reasoning_effort: "none"` so the template emits `Thinking effort
level: 0`, greedy; the server loads in 91 s — 36 GB of bf16 shells and the
edge tables read from the SSD, experts mmap'd):

| prompt (tokens) | answer | tokens | wall |
|---|---|---|---|
| "What is the capital of France? Answer in one word." (25) — TTFT probe, cold page cache | — | 1 | 176 s |
| same, warm | `Paris` | 4 | 33 s |
| "What is 17 + 25? Answer with just the number." (27) | `42` | 4 | 83 s |
| "Which ocean is the largest on Earth? One sentence." (24) | `The Pacific Ocean is the largest on Earth, covering about 63 million square` | 16 | 174–180 s |
| "What is the capital of Italy? One word." (25), **thinking on** (`reasoning_effort: "low"`) | `<think>The user asks for the capital of Italy, requesting one word. The answer is Rome.</think>Rome` | 25 | 205 s |

The thinking-on row is the special-token framing translated by the API
(`<|content_thinking|>…<|end_message|>` → `<think>…</think>`, then the
`<|content_text|>` answer). A first run with thinking left on by accident
(the API's GLM effort mapping had escalated `"none"` to `"high"` — fixed in
the same PR) produced coherent reasoning openings at 12 tokens per 256–299 s;
with the overlapped expert reads the warmup fell from 28 s to 13 s and a
cold 25-token prefill from 176 s to 107–155 s depending on page-cache state. Decode is 8–25 s/token depending on how many of a
token's ~8 experts × 64 layers are already in the page cache: the routed
experts (490 GB) page from a SATA SSD into 172 GB of RAM, so this box is a
correctness platform, not a throughput one (next section).

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
sliding layers' rings are fixed at 512 + 32) and `CASCADIA_INKLING_EXPERTS=eager|mmap`
(default mmap for real-sized expert sets; `eager` is the dequantised dev
path — 4× the bytes, measured 2.5× slower per layer than a page-cache-resident
mmap, so it is not the "resident" mode); in-process hosts set the same two
through `SparseMoEBuilderConfig::{max_seq, experts_mode}`. Schedule knobs,
both bit-identical either way: `CASCADIA_INKLING_SERIAL_EXPERTS=1` (visit a
token's experts one after another) and `CASCADIA_INKLING_SEQ_READS=1` (never
copy a paged-out expert's bin before its GEMV). `CASCADIA_INKLING_PIN_EXPERTS=1`
is the RAM-resident mode: every expert bin (and the dense MLPs) is `mlock`'d
as it is opened, so the export is wired in memory and the GEMVs never fault —
for a box whose RAM holds the export (512 GB int4), on macOS in particular,
which re-faults cached file pages on every touch. Best-effort: the first
refused lock is reported and the rest stay page-cache backed. The engine
sizes rayon's pool to the **physical** cores unless `RAYON_NUM_THREADS` is
set: the row-parallel GEMVs gain nothing from a core's second hyperthread
and lose 3× to it on macOS (Mac Pro 28c/56t, pinned: 33 → 10.4 ms per MoE
layer); Linux is indifferent (miner 24c/48t: 14.2 → 13.4).

## Sizing and the hardware honesty note

Per generated token the engine touches ~41B active parameters: ~21.7B routed
expert weights (int4, ~11 GB of reads), ~7.2B shared-expert weights, ~8.7B
attention weights and the dense/embed/unembed tables (bf16, ~36 GB resident).
With the 490 GB of routed experts paged from the miner's SATA SSD into 172 GB
of RAM, measured decode is 8–25 s/token (0.05–0.13 tok/s) and a cold 25-token
prefill takes ~3 minutes — the SSD, not the shell, is the clock there.

**RAM-resident, measured (2019 Mac Pro, Xeon W-3275M 28c/56t, 1.5 TB, macOS
12.7, the whole export `mlock`'d, 28 threads; same binary and prompts as the
miner runs, answers byte-identical):**

| | port as first merged (mmap, 56 threads) | + concurrent experts, pin, physical cores |
|---|---|---|
| load (wire 490 GB) | 70 s + 328 s first touch | 812 s (≈0.6 GB/s) |
| TTFT, 25-token prompt | 39.6 s | **12.5 s** |
| 64-token answer, wall | 227 s | **53.9 s** |
| decode | 3.0 s/token, 0.33 tok/s | **0.66 s/token, 1.5 tok/s** |
| prefill | 1.6 s/token | 0.5 s/token (per token, like decode: no row batching yet) |

The first resident run showed the shell, not memory, was the clock (3 s/token
against a ~0.4 s bandwidth floor): the experts ran serially, the per-token
bin copy cost 2×, macOS re-faulted cached pages, and 56 hyperthreads tripled
the GEMV time. The four schedule changes (all bit-identical, byte-compared
on an 8-layer dump) are what the second column measures; the remaining gap
to the bandwidth floor is the bf16 attention GEMVs, per-row dequant and the
unbatched prefill — the follow-ups below. A "record" run therefore wants the
int4 artifact (512 GB) resident — a ≥640 GB-RAM box with the pin, or an
N-rank pipeline whose ranks together hold it (e.g. 4 × 160 GB) — plus those
follow-ups.

### OpenVINO expert backend (iGPU / NPU / CPU)

A box whose CPU has no wide SIMD — Panther Lake is AVX2-only, so the int4
kernels take their scalar/AVX2 paths — can run the experts on its Xe3 iGPU
through OpenVINO instead: `CASCADIA_INKLING_OV_EXPERTS=1` with a
`<model>/experts_ov/` tree makes every routed / shared expert and the two
dense MLPs a compiled OV model (`inkling/ov_expert.rs`, the glm5 backend's
design with Inkling's naming: `layer_NN/expert_EEE`, `expert_sharedS`,
`dense`). The IRs come from `tools/inkling_expert_ov.py`, which packs the
bins' own nibbles and bf16 group scales into `u4`/`bf16` constants — no
re-quantisation, the IR sits on the exact grid the Rust kernel reads — and
`--validate` compares one expert on CPU or GPU against a numpy reference of
that grid. What differs from the Rust kernel is its inner bf16 rounding of
gate/up (OpenVINO's `Convert` to bf16 truncates, so the graph cannot
reproduce it) plus f32 accumulation order: measured on the Arc B390,
relative rms 1.5e-6 per expert at f32 with 99.9% of bf16 outputs
identical; `CASCADIA_INKLING_OV_PRECISION=f16` is the iGPU's native, inexact
fast mode (7e-4). `CASCADIA_INKLING_OV_DEVICE` (default `GPU`),
`_OV_CACHE` / `_OV_CACHE_MB` bound the LRU of compiled models by count and
estimated device bytes (a resident benchmark of a few layers needs
~13 GiB per MoE layer), `_OV_CACHE_DIR` persists the compiled blobs, and
`_OV_DQ_GROUP` (default 0) keeps the plugins' int8 activation quantisation
off. A missing or uncompilable IR falls back to the Rust kernel per expert
(warned once); GPU resource exhaustion disables the backend for the
process. `inkling_layer_dump --warm-ov` compiles every loaded layer's
experts before timing and the read-out counts device hits, compiles and
fallbacks, so a run that silently fell back cannot pass as a device
measurement. Whole-model runs on a paged host are out of scope for this
design (an IR per expert means a compile per first touch and one
compiled model per resident expert), which is the glm5 backend's finding
too; it exists to measure the iGPU's per-layer decode and prefill against
the CPU kernel on real layers.

Measured on tate-07 (Core Ultra X7 358H, 4P+8E+4LPE, 64 GB LPDDR5X, Arc
B390 iGPU, Windows 11, OpenVINO GenAI 2026.2.1 runtime), real layers of the
975B export, 23-token prompt, page cache warm, `inkling_layer_dump`, one run
each; the same binary with and without `CASCADIA_INKLING_OV_EXPERTS=1`:

| layer | CPU kernel (AVX2, 16 threads) | iGPU experts, f32 (exact) | iGPU experts, f16 |
|---|---|---|---|
| dense (0, 1), decode | 8.3 ms/token | 6.3 ms/token | 6.5 ms/token |
| MoE (2), decode | 23.7 ms/token | **7.6 ms/token (3.1×)** | 7.1 ms/token (3.3×) |
| MoE (2), prefill of 23 tokens | 210 ms | 152 ms | 129 ms |
| expert call on the device | — | 4.4 ms mean, 8 in flight | 3.5 ms mean |
| residual stream vs CPU after 3 layers | — | rel rms 5.0e-4, routing identical | 5.9e-4 |

The attention shell stays on the CPU (~3 ms/layer of bf16 GEMV), so the
MoE layer's remaining 4–5 ms is the eight concurrent expert calls; the
int4 weights stream at ~55–60 GB/s aggregate on the iGPU against the
CPU kernel's ~13 GB/s equivalent. f16 buys 7%: the calls are weight-stream
and call-overhead bound, not compute bound. Whole model, resident:
64 × 7.6 + 2 × 6.3 ≈ 0.5 s/token on this class of box versus 1.5 s/token on
its CPU kernel — a 3× that a 64 GB box cannot cash on its own (the
975B export pages from NVMe there; the experts' 16 GB/token of reads are the
clock), but which every rank of a resident pipeline would see. Warming the
260 IRs of three layers takes ~30 s from the blob cache (~13 GiB device).

Resident-set limit of this design on a 64 GB box: a second MoE layer on the
device (518 compiled models, ~26 GiB of driver allocations on top of the
bins' page cache) pushes the box into memory pressure — the 4-layer run
kept layer 3 at 7.7 ms/token (f32) / 6.8 (f16) but the first-allocated
layer 2 fell to 16.8 / 21.7 ms/token, and the CPU pass of that sequence ran
at 31–32 ms/token instead of 23.7 (one run each, the box hot from the
previous sequence). Numerics and routing stayed as above (rel rms 5.5e-4
after four layers). One MoE layer per 64 GB is the clean comparison; a
resident pipeline rank holds one or two layers anyway.

### OpenVINO fused-MoE backend (one compiled model per layer)

OpenVINO 2026.3's GPU plugin fuses a whole MoE layer into its
`moe_3gemm_fused_compressed` kernel (all experts one expert-major compressed
constant, a token's k experts in one launch, rows grouped per expert for
prefill, optional on-disk expert streaming). `tools/inkling_moe_layer_ov.py`
writes that layer graph for Inkling — the "tiled 3-GEMM block" the plugin's
`ConvertTiledMoeBlockToGatherMatmuls` pass matches, with the bins' own
nibbles as `u4` constants (zero point 8, bf16 scales as f16) and the routing
as inputs (`topk_indices`, `routing_weights`) fed by the Rust gate, so
Inkling's routing stays exact and the two shared experts ride as the last
two ids with their gammas — into `<model>/moe_ov/layer_NN/` (~8.2 GB, ~30 s
per layer). `inkling/ov_moe.rs` runs them: `CASCADIA_INKLING_OV_MOE=1`
(`_OV_MOE_DEVICE`, `_OV_MOE_CACHE_DIR`, `_OV_MOE_OFFLOAD` = the plugin's
`OFFLOAD_RATIO`), one call per layer per token or per prefill batch,
precedence over the per-expert backend for layers that have an IR, per-layer
fallback otherwise.

Findings on the Arc B390 (driver 32.0.101.8860, OpenVINO 2026.3.1 and the
2026.5 nightly), all reproduced on Intel's own optimum-intel Qwen3-MoE
export before any Inkling graph was involved:

- the plugin's batched-GEMV decode kernel crashes the process (access
  violation in `openvino_intel_gpu_plugin.dll`); the backend sets
  `OV_GPU_MOE_BATCHED_GEMV_THRESHOLD=0` so decode takes the grouped-GEMM
  path, and a single-row call still crashes there, so decode is padded to
  two rows (the second a copy with zero weights) — 2.9–3.0 ms per MoE layer
  at Inkling's shape (8 experts, 256 MB, ~85 GB/s) versus 4.4 ms for the
  per-expert backend's eight concurrent calls;
- prefill: 23 rows in 30 ms per layer (unique experts read once, ~60 GB/s),
  128 rows in 34 ms — versus 152 ms for the per-expert backend's per-row
  calls;
- numerics: the kernel is f16-only; the real layer-2 IR matches the numpy
  grid reference at relative rms 6.4e-4 (the per-expert f32 path: 1.5e-6);
- on-disk offload (`OFFLOAD_RATIO` + `WEIGHTS_PATH`) needs the asymmetric
  layout (the symmetric `i4` export's zero-point placeholder has no bin
  offset) and streams non-resident experts at ~1 GB/s even from a warm page
  cache — an order of magnitude under the NVMe and the Rust mmap path, so it
  cannot serve a paged whole model on this box;
- the matcher wants the single-input `Swish` (the Python helper's default
  adds a beta constant and silently prevents fusion);
- a saved IR's file-backed constants cannot be built into the fused op at
  all (the plugin's reorder insertion fails) unless the offload path is on;
  the shim therefore materialises the IR's constants in memory before
  compiling (`CASCADIA_MATERIALIZE_CONSTANTS=1`, the backend's default),
  which is the form that measures 3.4 ms per padded decode row — the
  offload path (`CASCADIA_INKLING_OV_MOE_OFFLOAD=N`) works too but starts
  its slot cache empty (every first touch streams at ~1 GB/s, so the
  backend sweeps all experts at warm-up), needs the IR padded with dummy
  experts so its 1% floor still holds every real expert (`--pad-experts`),
  and costs 5.5 ms per decode row and 55 ms per 23-row prefill;
- each new row count pays a first-call cost (~55 ms, and ~1 s at the
  32-row kernel boundary), so a serving loop should bucket prefill rows;
- the plugin reads its knobs through the C runtime's environment, which a
  Rust `set_var` does not update on Windows; the backend uses `_putenv_s`
  there.

Measured on tate-07 with the fused backend in the engine (same dump, real
layer 2, 23-token prompt, one run each, all with zero fallbacks):

| path | MoE layer decode | prefill, 23 tokens | fused call mean | residual vs CPU |
|---|---|---|---|---|
| CPU kernel | 30.1 ms/token | 207 ms | — | exact |
| per-expert iGPU, f32 | 7.4 ms/token | 153 ms | 4.5 ms × 8 in flight | rel rms 5.0e-4 |
| fused iGPU, materialised | 6.8 ms/token | 148 ms | 6.7 ms (≈4.5 decode, 55 prefill) | rel rms 1.9e-3 |
| fused iGPU, offload 1% | 6.7 ms/token | 148 ms | 6.7 ms | rel rms 1.9e-3 |

The ~3 ms of each layer that remain are the bf16 attention GEMVs on the
CPU (already at ~80 GB/s), which is why the next step is the int4
attention-projection backend below rather than a device copy of bf16.

### OpenVINO attention-projection backend (int4 projections on the iGPU)

`tools/inkling_attn_ov.py` re-quantises a layer's five attention GEMVs
(`q`, `k`, `v`, `r`, `o`; ~264 MB bf16) to int4 on the experts' grid
(~66 MB) and writes two IRs per layer (`attn_ov/layer_NN/{qkvr,o}`);
`inkling/ov_attn.rs` runs them (`CASCADIA_INKLING_OV_ATTN=1`,
`_OV_ATTN_DEVICE`, `_OV_ATTN_DIR` to point at an alternate-precision dir —
the tool writes int8 into `attn_ov` by default, int4 into `attn_ov_int4`)
with outputs rounded to bf16 like the Rust kernel. The head norms, position
bias, softmax, KV cache and convolutions stay in Rust; prefill projects all
rows in one call and applies the output projection once per batch. Device
probe on the B390: int4 qkvr 0.60 ms + o 0.32 ms per layer against ~3 ms on
the CPU, while a f16 copy of the same weights takes 2.45 ms — the byte count
is the lever, not the device.

With everything on the iGPU (dense MLPs and the fused MoE as above, plus
these projections), the same dump on tate-07:

| path | dense layer | MoE layer decode | MoE prefill, 23 tokens | residual vs CPU after 3 layers |
|---|---|---|---|---|
| CPU kernel | 9.7 ms/token | 30.1 ms/token | 207 ms | exact |
| fused MoE on iGPU, attention on CPU | 6.2 ms/token | 6.9 ms/token | 148 ms | rel rms 1.9e-3 |
| + int4 attention on iGPU (`--weights int4`) | 3.6 ms/token | 4.5 ms/token | 88 ms | rel rms 1.5e-2 |
| **+ int8 attention on iGPU (`--weights int8`, default)** | **4.2 ms/token** | **5.1 ms/token** | **89 ms** | rel rms 6.5e-3 |

That is 5.9× (int8) to 6.6× (int4) the CPU kernel per MoE layer, ~0.33 s
per token for the whole model on resident ranks (~3 tok/s single stream on
a 12-rank pipeline). The attention weights' round-to-nearest quantisation
is what sets the residual: int4 (group 32) carries ~10% weight-relative
error and moves the residual stream to 1.5e-2, int8 (per row) ~1.2% and
6.5e-3; the plugin's own error on either set of weights is 2e-4. int8 is
the default; int4 is the speed option for a demo that tolerates it. With
int4 the device's f16 GEMV and GEMM paths also stopped being bit-identical,
so decode and prefill differed by 3e-4 relative at layer 2 (the dump's
`dec-vs-pre` column); with int8 they agree to the bit again.

### OpenVINO head backend, and releasing the Rust copies

`tools/inkling_attn_ov.py --head --weights int8` writes the unembed table
(`[201024, 6144]`, 2.46 GB of bf16 that every token reads on the last
rank, ~31 ms on this CPU) as one compressed-FC IR under `head_ov/`;
`CASCADIA_INKLING_OV_HEAD=1` (`_OV_HEAD_DEVICE`, default `GPU`) runs it
through `inkling/ov_head.rs`. The RMSNorm, the mup divide and the slice to
`unpadded_vocab_size` stay in Rust. On tate-07 the int8 head validates at
2.1e-4 relative to the quantised grid with the same argmax, and a 1-row
call takes 11.3 ms on the B390.

`CASCADIA_INKLING_OV_ATTN_DROP_RUST=1` compiles each layer's attention IR
at load and frees the five bf16 projection tables (264 MB per layer,
16.4 GB for the model) that would otherwise sit next to the int8 device
copy in unified memory. A layer whose IR fails to compile keeps its
tables; after a release a refused backend call is fatal and says so. On a
64 GB box that RAM is what the expert cache lives on
(`CASCADIA_INKLING_EXPERT_CACHE_MIB`, now allowed up to 1024 per layer),
which is the point. `CASCADIA_INKLING_OV_MOE_LAYERS=2,3,4` restricts the
fused-MoE backend to a subset of the layers that have IRs (the Windows
rank budget below).

### Expert-parallel dispatch (star topology)

Beside the layer pipeline, the family can run as a **driver + expert
workers**: the driver (`--ep-workers a:p,b:p,…`, a single stage) holds every
layer's attention, norms, convs, router, dense MLPs, embed and head, and no
expert weights; worker `N` of `W` (`--ep-worker-index N --ep-worker-count W
--listen :port`) holds, for every MoE layer, the experts with `id % W == N`
(the two shared experts are dispatched like routed ones, with their gammas as
weights) and nothing else. Per MoE layer the driver routes locally, sends each
involved worker the rows' hidden states plus the expert ids it must serve
(`ExpertDispatch`, all workers in flight together), receives each expert's raw
output (`ExpertResult`), applies the weights and sums in gate order — so the
result is **bit-identical** to the single-process `MoeLayer::forward`
(asserted by `tests/inkling_ep.rs` over loopback TCP). Workers are stateless:
no reset, no prefix frames, no KV.

```bash
# workers first (each holds ~1/W of the 490 GB of routed experts, mmap'd)
cascadia worker --rank 0 --total 1 --engine sparse-moe --model /data/inkling-int4 --ep-worker-index 0 --ep-worker-count 3 --listen :9200
cascadia worker --rank 0 --total 1 --engine sparse-moe --model /data/inkling-int4 --ep-worker-index 1 --ep-worker-count 3 --listen :9201
cascadia worker --rank 0 --total 1 --engine sparse-moe --model /data/inkling-int4 --ep-worker-index 2 --ep-worker-count 3 --listen :9202
# then the driver
cascadia run /data/inkling-int4 --engine sparse-moe --ep-workers hostA:9200,hostB:9201,hostC:9202 --api :8000
```

Measured on the miner, all four processes on the one box over loopback TCP
(3 workers, mmap experts paging from the same SATA SSD; page cache warm from
the earlier runs): the same four prompts answer identically to the
single-process runs (`Paris`, `42`, the Pacific sentence); cold TTFT 42 s
(single-process runs: 107–176 s), 16-token answer 121 s vs 174–180 s, i.e.
0.13 tok/s vs 0.09. That is the parallel-expert-read effect of the scaling
note's RAM-starved regime showing up even on one box (three processes read
their expert bins concurrently); the network round per MoE layer costs
~0.15–0.2 ms on loopback. When it pays and when it does not across real
boxes — one LAN round per MoE layer, the driver's own attention reads as the
serial floor — is worked through in the scaling note.

Resident (Mac Pro, export pinned, driver on 28 threads + 3 workers on 9
each, loopback): same byte-identical answers; TTFT 13.6 s vs 12.5 s single
process, 64-token answer 59.9 s vs 53.9 s, decode 0.74 vs 0.66 s/token — the
star's whole cost on one box is ~1.2 ms per MoE layer (frame round + the
thread split), which is what a cabled LAN has to stay under to break even
per §4 of the scaling note.

Topology, bandwidth ceilings and what expert-level routing across boxes would
buy: [`../perf/INKLING_SCALING.md`](../perf/INKLING_SCALING.md).

### Serving with the iGPU backends (end to end)

`cascadia run <model> --engine sparse-moe --api :8011` on tate-07 with the
three backends enabled and IRs for layers 0–7 (the remaining 58 layers on
the CPU, paged from NVMe): the same three prompts as the miner and Mac Pro
runs answer byte-identically — `Paris`, `42`, the Pacific sentence — with
int8 attention projections and the fused MoE on the iGPU. Wall times there
are the paged CPU layers' (5–22 minutes per prompt), so this is the
correctness gate for the device paths inside the real serving loop, not a
speed measurement. In fact that run was *paging*: six fused MoE layers are
50 GB of unified memory on top of the ~40 GB the whole-model process needs,
which is what those wall times were. On a single 64 GB box the fused
layers must stay off and the iGPU carries the attention projections and
the head only; `docs/perf/INKLING_SINGLE_BOX_BENCH.md` has the whole-model
single-stream measurements (CPU control, ours on the iGPU, OpenVINO's own
MoE offload) under the autolab's campaign-129 protocol.

**Rank budget.** The limit on a 64 GB Windows box is not RAM but the
iGPU's shared-memory budget, which Windows sets to half the RAM: OpenVINO
reports `GPU_DEVICE_TOTAL_MEM_SIZE` = 33.5 GiB on tate-07. A fused MoE
layer is 8.3 GB of device allocations, so three layers (25 GB) run at the
per-layer numbers above, four (33.2 GB) sit at the cap and decode falls to
50–145 ms per MoE layer, and five or six page every call (250–430 ms).
So a 64 GB Windows rank carries **three** fused MoE layers on the iGPU
(the two dense layers are small) — 22 such ranks for the 975B model — a
96 GB box five and a 128 GB box seven. Confirmed with the 5-layer dump
(2 dense + 3 fused MoE, int8 attention): 4.5–4.6 ms per dense and
5.4–5.6 ms per MoE layer decode, 52–91 ms prefill, residual 5.9e-3 after
five layers — the single-layer numbers hold across a full rank. Linux ranks are not under the 50%
policy (Level Zero shared allocations can use most of the RAM) and are
the way to get five or six layers per 64 GB; that is unmeasured here.

## Open follow-ups

- **Multi-stream decode (aggregate throughput).** Done: per-stream
  sequence slots in the layers, `Layer::forward_rows` over `(slot, token)`
  rows, a single-stage scheduler (`CASCADIA_STREAMS=N`) and a pipeline wire
  with G micro-batches in flight (`CASCADIA_STREAMS_INFLIGHT`) — see
  [`../perf/INKLING_MULTISTREAM.md`](../perf/INKLING_MULTISTREAM.md).
  Still open: real-model aggregate numbers on hardware, prefill/decode
  mixing inside one micro-batch (a prefill currently stalls the other
  streams for its duration), and per-stream prefix caching.