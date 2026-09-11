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
  mmap kernels (AVX-512 → AVX2 → scalar dispatch). After routing, the selected
  bins are prefetched and read concurrently (glm's overlapped path) instead of
  page-faulting one at a time inside their GEMVs; `CASCADIA_INKLING_SEQ_READS=1`
  restores serial reads.
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
| "Which ocean is the largest on Earth? One sentence." (24) | `The Pacific Ocean is the largest on Earth, covering about 63 million square` | 16 | 180 s |

A first run with thinking left on (the API's GLM effort mapping had escalated
`"none"` to `"high"` — fixed in the same PR) produced coherent reasoning
openings ("The user is asking for the capital of France and wants") at 12
tokens per 256–299 s. Decode is 8–25 s/token depending on how many of a
token's ~48 experts × 64 layers are already in the page cache: the routed
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
(default mmap for real-sized expert sets); in-process hosts set the same two
through `SparseMoEBuilderConfig::{max_seq, experts_mode}`.

## Sizing and the hardware honesty note

Per generated token the engine touches ~41B active parameters: ~21.7B routed
expert weights (int4, ~11 GB of reads), ~7.2B shared-expert weights, ~8.7B
attention weights and the dense/embed/unembed tables (bf16, ~36 GB resident).
RAM-resident at 100 GB/s that is roughly 0.3–0.5 s/token. With the 490 GB of
routed experts paged from the miner's SATA SSD into 172 GB of RAM, measured
decode is 8–25 s/token (0.05–0.12 tok/s) and a cold 25-token prefill takes
~3 minutes — the SSD, not the shell, is the clock. A "record" run needs the
int4 artifact (512 GB) resident: a ≥640 GB-RAM box, or an N-rank pipeline
whose ranks together hold it (e.g. 4 × 160 GB), plus the prefix cache and
expert-residency follow-ups below.

## Open follow-ups

- Per-rank KV-prefix cache and the qwen35-style in-process prefix cache (TTFT).
- MTP draft head (exported? no — dropped) / n-gram speculative decode: the
  rewind slack is in place.
- Vision / audio inputs (encoders dropped).
- Hot/cold expert residency (`CASCADIA_GLM5_HOTCOLD` port) for paged runs.
