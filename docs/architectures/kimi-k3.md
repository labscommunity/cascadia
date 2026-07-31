# Kimi-K3 (`kimi_k3`)

Moonshot **Kimi-K3** (`moonshotai/Kimi-K3`, ~2.8T MoE, 1M ctx) analysed against the
`cascadia-engine-sparse-moe` engine.

**Status: running on real weights.** The full 1.56 TB export is built and
verified, and the model generates correct text — `"The capital of France is"` ->
`" Paris. The E"`. The shell (`crates/cascadia-engine-sparse-moe/src/k3/`), the
exporter (`tools/export_kimi_k3.py`) and the CPU reference (`tools/kimi_k3_ref/`)
are golden-tested, and `{1,2,3,4,6}`-rank pipelines are bit-identical to a single
process.

Multi-rank is validated too: a 4-rank chain over the real transport returns
`" Paris. The"`, identical to the single process, with no wire errors. That path
had only ever run in-process before, so the widened AttnRes wire had never
crossed a socket.

**Any multi-rank K3 run must raise `CASCADIA_ACTIVATION_TIMEOUT_SECS`.** The
transport's default reply deadline is 60 s, and a K3 decode step takes minutes —
the wire reads a slow rank as a dead peer, drops the task after the first token
and reports it as a clean short completion, which looks like an early stop rather
than a failure. This will bite the fleet identically.

What remains is throughput, not correctness. Decode is dominated by fetching
expert weights; see [Measured on that host](#measured-on-that-host) for the real
numbers and [Where the remaining speed is](#where-the-remaining-speed-is) for what
is left.

K3 **does** run on the 4× 32 GB AI-PC fleet that dsv4 and glm5 target: the bf16
shell is ~29 GB per node of 32, and the routed experts stream from NVMe as
GLM-5.2's do. What it does not get there is speed — ~0.15% expert residency, so
roughly 0.05–0.1 tok/s. More nodes improve that steeply. See
[Feasibility](#feasibility).

## Architecture

From the real `config.json`. The checkpoint is multimodal
(`KimiK3ForConditionalGeneration`, 27-layer ViT); we drop the vision tower and
serve `text_config` only (`model_type: kimi_linear`).

| Param | Value |
|---|---|
| layers | 93 (`first_k_dense_replace=1` dense + 92 MoE, `moe_layer_freq=1`) |
| hidden / vocab | 7168 / 163,840; dense FFN `intermediate_size=33792`; `rms_norm_eps=1e-5` |
| experts | **896 routed + 2 shared, top-16**, `moe_intermediate_size=3072` |
| **LatentMoE** | routed experts run in a **3584-dim latent**, not hidden (`routed_expert_hidden_size=3584`, `latent_moe_use_norm=true`) — per-layer 7168↔3584 projections + RMSNorm |
| routing | `sigmoid` + `noaux_tc` bias, norm-topk, `routed_scaling_factor=1.0`, `num_expert_group=1, topk_group=1` (no group masking) |
| activation | **SiTU** (`hidden_act="situ"`, β=4.0, linear β=25.0) — model-global, NOT SwiGLU |
| attention | **hybrid 3:1** — 69 KDA layers : 24 full-attention layers |
| ├ KDA | Kimi Delta Attention (gated delta-rule linear attn): 96 heads × 128, `short_conv_kernel_size=4`, `gate_lower_bound=-5.0`, `use_full_rank_gate=true`. Fixed-size recurrent state, **no KV** |
| └ full attn | MLA: `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope=128` + `qk_rope=64`, `v_head_dim=128`, 96 heads. **`mla_use_nope=true`** (no RoPE at all) + **`mla_use_output_gate=true`** |
| residual | **AttnRes** — attention residual carried across 12-layer blocks (`attn_res_block_size=12`) |
| rope | none on full-attn layers (NoPE); `max_position_embeddings=1,048,576` |
| quant | **`mxfp4-pack-quantized`** (compressed-tensors): FP4 **e2m1** values + **E8M0 u8** group-32 scales, symmetric. Only routed experts are quantized — `self_attn`, `shared_experts`, dense MLP and `lm_head` are bf16 |

## Feasibility

### Routed experts (streamed)

```
per expert    = 3 * 3584 * 3072            =  33.0M params
routed total  = 92 layers * 896 * 33.0M    =  2.72T params   (matches the "2.8T" claim)
on disk @fp4  = 2.72T * 0.5 B * 1.0625     =  ~1.45 TB       (E8M0 u8 scales = +6.25%)
active/token  = 16 * 92 * 33.0M            =  48.6B params   = ~24 GB streamed/token at 0% hit
```

### Always-resident bf16 shell

Everything below is in the quantization `ignore` list, so it is bf16 and must be
RAM-resident before a single routed expert is pinned:

| Component | Derivation | bf16 |
|---|---|---:|
| KDA attention | 69 × (4×7168×12288 qkvo + 7168×12288 full-rank gate) = 30.4B | **60.7 GB** |
| shared experts | 92 × 2 × 3 × 7168 × 3072 = 12.2B | **24.3 GB** |
| gated MLA | 24 × ~232M = 5.6B | 11.1 GB |
| LatentMoE projections | 92 × 2 × 7168 × 3584 = 4.7B | 9.4 GB |
| embed + lm_head | 2 × 163,840 × 7168 = 2.35B | 4.7 GB |
| dense layer 0 | 3 × 7168 × 33,792 = 727M | 1.5 GB |
| | | **~112 GB** |

Confirmed from the modeling source: `shared_experts` is built with no
`hidden_size` override, so it runs at 7168 rather than in the 3584 latent —
the more expensive of the two possibilities.

### Verdict

The shell is split across ranks with the layers, so what matters is the PER-NODE
figure, not a fleet total. A KDA+MoE layer costs ~1.25 GB of bf16 shell, an
MLA+MoE layer ~0.83 GB; embed and head add 4.7 GB spread over the two edge ranks.
Allowing ~2.5 GB per node for the OS and the KV/recurrent state:

| Nodes (32 GB each) | layers/node | shell/node | left for expert cache | resident |
|---|---:|---:|---:|---:|
| **4** | ~23 | 29.0 GB | 0.5 GB | ~0.15% |
| 6 | ~16 | 20.1 GB | 9.4 GB | ~3.9% |
| 8 | ~12 | 15.7 GB | 13.8 GB | ~7.6% |
| ~53 | ~2 | ~2 GB | ~27 GB | ~100% |

**K3 runs on 4 nodes.** It fits — 29 GB of 32 — and the routed experts stream
from NVMe exactly as GLM-5.2's do. Expect roughly **0.05–0.1 tok/s**: each rank
streams ~6 GB per token and the pipeline is serial across ranks. Correctness is
fully testable there; the throughput number will be poor but real.

Two caveats at N=4. 29 GB of 32 leaves little for the OS, so whether it stays
stable or starts swapping is empirical, not something the arithmetic settles. And
with ~0.5 GB of page cache against a 1.45 TB expert set, `CASCADIA_K3_AUTOPIN` is
doing real work rather than a marginal optimisation.

More nodes help steeply, because the shell shrinks as the cache grows: N=6 and
N=8 reach ~3.9% and ~7.6% residency. The split is parameterised by `total`, so
no code change is needed.

| Other targets | Result |
|---|---|
| **Xeon bench host** (~172 GB RAM, ~1.6 TB free) | export fits with `--free-source-shards`. Predicted ~0.1 tok/s at ~4% resident; MEASURED 363 s/token there, ~36x worse — see [Measured on that host](#measured-on-that-host) |
| single host, ≥768 GB RAM + ~2 TB striped NVMe ≥10 GB/s | 45–60% residency → ~0.5–1 tok/s |

For scale: GLM-5.2 is ~386 GB int4 and reaches 0.4–0.6 tok/s at N=4 with ~35%
residency. K3 is ~3.8× larger with 2× the active experts per token, so at the
same node count it sits two orders of magnitude lower on residency. That is the
real difference between them — not whether K3 fits.

#### On the Xeon bench host

Source and export cannot coexist on one filesystem there, and no single mount is
large enough for the export alone:

```
source checkpoint  1.561 TB   (already downloaded and byte-verified)
export output      1.560 TB   (1.446 TB experts + 116 GB shells; no shrink —
                               the experts are native MXFP4, so the repack is
                               byte-for-byte and there is nothing to requantise)
largest single mount  ~567 GB
```

`--expert-roots` spreads the expert bins over several filesystems and symlinks
them back into `<out>/experts/`, so nothing has to be deleted:

```
shells (116 GB)  ->  a mount with ~150 GB spare
33 / 32 / 32 expert layers  ->  three ~550 GB mounts
capacity 97 layers vs 92 needed
```

Note the shells need their own mount. Three mounts is 2 layers SHORT once the
116 GB of shells takes a bite out of one of them — four is the working plan.

The exporter enforces both halves of that on its own rather than trusting the
operator to have done the arithmetic: the pre-flight sizes `<out>` for the shells
only when `--expert-roots` is given (the bins are not landing there), and the
root planner holds back the shell bytes on whichever root shares a filesystem
with `<out>`, plus budgets per filesystem rather than per directory so two roots
on one mount cannot each claim its free space.

### Measured on that host

The numbers below are from a real run of the full export, not arithmetic. An
earlier revision of this section predicted `45 s/token` from a seek-plus-transfer
model and credited it to spreading the roots over three spindles; both were
wrong, and the run is recorded here in their place.

The first run used the defaults as they shipped; the second is after the fixes
below. Same host, same prompt, `routed` bytes byte-identical in both, so the
comparison is like-for-like.

```
                          as-shipped      corrected      note
load                        1010.8 s       1088.7 s      unchanged control
tok 1 forward                768.5 s        182.6 s      4.22x, EXCLUDES prefill
decode, steady state         737.8 s        148.6 s      4.96x
per token, end to end       1278.3 s        363.0 s      3.52x
experts, share of wall           99%            46%      same strategy both sides
eff                        204 MB/s      1836 MB/s      both overstated ~5x
page-cache hit                 4.7%           4.8%
routed                     154.98 GB      154.98 GB      identical work
```

Two rows above are not what their old labels claimed, because the profiler was
blind to prefill: it recorded prefill's routed bytes but none of its time.

The `tok 1` row was labelled "prefill + tok 1" and is the decode forward alone.
And `eff` is routed bytes over the EXPERT-BUCKET time, a bucket prefill also
never contributed to — so it divided prefill's bytes by decode's time and
overstated fetch throughput about 5x.

Both columns are wrong the same way and the wall-clock rows are measured
directly, so every ratio above still stands. Re-measured after the fix, on the
same host and prompt:

```
prefill                      757.6 s      72% of a 3-token run, previously unbilled
tok 1 forward (in the above)     ~147 s
decode, steady state         147.5 s      matches the 148.6 s column
eff, prefill                306 MB/s      vs 1836 MB/s reported before
accounted for                 99.98%      905.1 + 147.5 vs 1052.8 s end to end
```

Two changes account for it: `madvise(MADV_WILLNEED)` over a layer's routed
experts right after routing, and an AVX2 fp4 kernel. They are not independent —
AVX2 is compute-only, yet measured *fetch* throughput rose 745 -> 1836 MB/s,
because a faster kernel returns to issue the next prefetch sooner and keeps the
queue deeper. Modelling the expert bucket as serial `compute + I/O` does not even
balance against these numbers; they compound.

Prompt `"The capital of France is"` -> `" Paris. The"`, unchanged throughout.

Two things that model got wrong:

**Spreading layers across mounts buys no read parallelism.** A layer's 16 expert
reads all live in one bin on one spindle, and layers are visited serially, so the
concurrency is one disk deep no matter how many roots there are. Aggregate
throughput peaked at 204 MB/s — about what a single one of these drives does. The
spread is still worth having for *capacity*, which is why it exists, and it lets
a later prefetch overlap layer N+1 with layer N; it is not a bandwidth multiplier
today.

**Compute is free.** KDA, MLA, AttnRes and the router together are ~1% of wall
time. Nothing on the CPU side is worth optimising until the I/O is fixed; the
whole cost is fetching expert weights.

The ~4x amplification is readahead over-fetching past each 17.6 MB slice inside a
15.7 GB mapping — the expert set is scattered, not streamed. Suppressing it with
`MADV_RANDOM` was tried and removed: it cut the amplification and lost on both
storage classes anyway, because it also kills readahead WITHIN each contiguous
slice. See the fetch-strategy section.

This is a correctness harness, not a throughput benchmark. Use
`examples/k3_run.rs`, which reports which devices are backing the experts and
works identically for a single-directory or a split export:

```
cargo run --release --example k3_run -- <export> "The capital of France is" 20
CASCADIA_K3_PROFILE=1 CASCADIA_K3_AUTOPIN=1 …
```

## Implementation

Follows the established sibling-shell pattern (`src/dsv4/`, `src/glm/`) — a Rust
shell validated against a Python CPU reference, not OpenVINO-traced graphs.

### Reuse map

| Reuse as-is | Adapt | Net-new |
|---|---|---|
| `dsv4::math` (bf16/linear/dot/rmsnorm), `staged::StagedRunner`, `sampling`, the dsv4 TCP pipeline wire, `glm::gate::moe_gate` — sigmoid + `noaux_tc` + norm-topk is an **exact** match for K3's router | `glm::attn` **absorbed-decode structure only** (`qabs = W_UKᵀ·q`; `score = qabs·Lc`; `ctx = W_UV·clat`) — rewritten, not flagged: NoPE deletes the whole `Rc`/`k_pe` path, and the output gate has no hook | KDA layer (short conv + gated delta recurrence + full-rank gate), SiTU, LatentMoE block + projections, AttnRes block carry, **fp4 e2m1 expert kernel** |

### Corrections to the obvious-but-wrong approach

1. **`dsv4::expert_mmap` cannot decode K3 experts.** It is a *linear symmetric*
   grid — `Σ (nibble−8) · bf16_scale · x` (`expert_mmap.rs:438`). FP4 e2m1 is
   *nonlinear* (`{0, .5, 1, 1.5, 2, 3, 4, 6}±`) with u8 E8M0 scales. Add a new
   fp4 bin format + a 16-entry-LUT SIMD kernel (a mechanical variant of
   `dequant_row_dot_avx2/512`). Do **not** have the exporter regrind mxfp4 →
   linear int4: re-quantizing an already-4-bit grid whose levels (0.5/1.5/3)
   aren't on the linear grid is an unevaluated quality risk, and it costs
   +12.5% disk for bf16 scales instead of +6.25%.
2. **Batched prefill is mandatory from day one, not an optimization.**
   Per-token prefill streams the full active set per position: a 4k prompt at
   ~8% residency is ~4000 × 22 GB ≈ 80 TB of reads. Batch-union saturates
   essentially all 896 experts per layer in one pass ≈ 1.36 TB — a ~60×
   difference. KDA's sequential recurrence does **not** conflict with this:
   `glm/stage.rs:601` already runs per-position attention inside the batched
   layer loop, then unions the MoE across rows. K3 does the same, walking the
   recurrent state row by row (~1.6M MACs/position — noise next to the MoE).
   Chunked-scan (parallel delta rule) KDA stays deferred.
3. **AttnRes: widen the wire — boundary-snapping does not work.**
   *(Corrected after reading `modeling_kimi_linear.py`; an earlier revision of
   this doc said the opposite.)* AttnRes is **not** a carried anchor. Each layer
   holds a growing **stack** of per-block residuals and mixes over all of them
   with a learned softmax (`_apply_attn_res`, applied **twice per layer** — once
   before attention, once before the MLP). Because the mixture attends over
   *every* prior block, snapping rank starts to block boundaries saves nothing:
   the stack still has to cross the wire. The inter-stage activation is
   therefore `prefix_sum [H]` **+** `block_residual [n_blocks, H]` — up to
   9 × 7168 f32 ≈ 258 KB/token. This is the dsv4 Hyper-Connections situation
   (`dsv4/stage.rs:75`, `hidden = hc_mult * hidden_size`), not the glm5
   `index_aligned_split` one. Use an even layer split and widen the wire.
4. **Keep the sibling module.** 69 of 93 layers are linear attention — there is
   no MLA-shaped core to generalize dsv4/glm5 into. Cross-importing leaves is
   already the in-tree precedent (`glm/attn.rs` uses `dsv4::math`; `glm/stage.rs`
   uses `dsv4::stage::even_layer_split`).

### What is built

| | |
|---|---|
| `tools/kimi_k3_ref/` | CPU reference; 11/11 checks against the vendored upstream, 7 bit-exact |
| `tools/export_kimi_k3.py` | config contract, fp4 repack, streaming + resumable, `--check-index`, `--free-source-shards`, tokenizer.json |
| `tools/kimi_k3_tokenizer.py` | tiktoken → `tokenizer.json`, validated identical to reference tiktoken |
| `src/k3/` | SiTU, AttnRes, KDA, absorbed gated-NoPE MLA, LatentMoE, fp4 kernel, loader, stage, profiler, residency |
| tests | e2e argmax-exact vs the reference; `{1,2,3,4,6}`-rank bit-identical; batched prefill bit-exact |

The export and first-token bring-up are done — see the measured run above.

### How the routed experts are fetched

Three strategies over K3's real access pattern — 16 scattered 17.6 MB slices per
layer — with the page cache dropped between runs. Useful MB/s counts only bytes
the model asked for, so over-fetching is penalised:

```
                  NVMe    rotational
mmap+willneed     2117           141
mmap+random        112            75
pread x16         3456           136
```

That benchmark drops the cache before every run, so it only describes a *cold*
fetch. Decode is not cold — by the second token most routed experts are already
resident, and the strategies stop being equivalent there: the mapping hands back
a pointer, an explicit read still copies the whole 17.6 MB slice.

Measured on the real model, two runs per side, same binary, cache dropped
identically, this flag the only difference:

```
                  prefill+tok1        steady decode      3 tokens
mmap        905.1s  897.8s  (901.5)  147.5s 140.3s (143.9)  1045.6s
pread x16   914.2s  917.2s  (915.7)  131.9s 131.5s (131.7)  1047.6s
```

Explicit reads are 8.5% faster at steady decode, 1.6% slower at prefill, and a
wash over three tokens. Prefill is paid once and decode per token, so anything
generating more than a handful comes out ahead — hence the default. The pread
pair agrees to 0.3%, the mmap pair spreads 5%.

**Both phases must use the same strategy.** An earlier build ran mmap prefill
into pread decode and steady decode cost 193.0s — 47% worse than either
consistent choice, with identical decode code. Prefill decides what the page
cache holds when decode starts; mixing leaves it suiting neither. One flag covers
both phases today, and a per-phase split has to rule that combination out.

That mixed build is also where a reported "26% steady-state regression" came
from: it was compared against an mmap run, so binary and flag both differed, and
the result was read as a property of `pread`. It took a controlled repeat to
show the opposite sign. Two variables, one conclusion, wrong.

The first default was set on the cold benchmark alone, which
is the second time on this model that a microbenchmark predicted the opposite of
the workload (see `MADV_RANDOM` below). A fetch strategy is only settled once it
has run against real decode.

`MADV_RANDOM` was removed rather than kept behind a flag. It did what it claimed
— read amplification fell ~4x to ~1x — but it loses on *both* storage classes,
by 19x on NVMe and 1.9x on rotational, because it suppresses readahead *within*
each contiguous slice. It was briefly the default, which cost 5.8x on the
rotational host until it was measured. Kept in git history, not in the code.

### Devices: CPU, iGPU, NPU — measured, not assumed

Benchmarked on a Core Ultra 7 258V AI-PC (Arc 140V iGPU, AI Boost NPU) with a 4-bit
decompress+MatMul at K3's real expert dims, via OpenVINO. GFLOP/s:

```
w1/w3 [3072,3584]
 batch    CPU     GPU     NPU
     1  293.1   125.7    18.9      decode  -> CPU wins 2.3x
     8  871.6   496.7    73.1      CPU still wins
    32 1019.0  1799.2   288.9      prefill -> GPU wins 1.8x
```

**Decode belongs on the CPU.** At batch 1 the iGPU cannot fill its occupancy on a
single-row GEMV. Note this is NOT the shared-bus explanation that has been passed
around: at batch 1 the CPU moves 82 GB/s and the GPU 35 GB/s, both far below the
~137 GB/s bus, so bandwidth is not the limiter for either. The conclusion (use the
CPU) was right; the reason usually given for it is not.

**Prefill is a real iGPU opportunity.** The crossover lands by batch 32, and prefill
is minutes of first-token latency on this model. Unbuilt.

**The NPU is out.** 15x slower than CPU at batch 1 and still 3.5x slower at batch
32, before accounting for the static-shape work needed to compile a dynamic MoE at
all. Not worth pursuing.

**CPU compute is not free once I/O is fixed.** fp4 weights give ~3.8 flop/byte,
above this machine's ~1.5 flop/byte balance, so a resident deployment is compute
bound rather than bandwidth bound. That is why `expert_fp4.rs` has an AVX2 path:
the AI-PCs have no AVX-512, so every AVX-512 kernel in the tree falls back to
scalar there.

### Where the remaining speed is

Decode is 99% expert I/O, so every worthwhile change is an I/O change. Ranked by
measured impact rather than by how interesting the code is.

One caveat that cost most of a day: three entries below were configured, reported
themselves enabled, and did nothing. The prefix cache sized its key index from an
unrelated env var and evicted every entry on the line after recording it; autopin
needs more tokens than any test run produces; explicit reads were judged from a
comparison that moved two variables. None failed loudly. Before trusting a row
here, check that the feature was observed DOING something — a `routed=` that
dropped, a `syscr` that moved — not merely that it was switched on.

| | Status | Expected |
|---|---|---|
| `madvise(MADV_WILLNEED)` after routing | done | **measured 2.46x**: tok 1 forward (excl. prefill) 768s -> 312s, `eff` 204 -> 745 MB/s, `routed` bytes identical |
| AVX2 fp4 expert kernel | done | **measured 1.79x** at real dims on x86 |
| explicit concurrent reads | done, **default** (`CASCADIA_K3_READ=0` opts out) | **measured +8.5%** steady-state decode, -1.6% prefill, 2 runs per side. Both phases must use the same strategy — see the fetch section |
| `madvise(MADV_RANDOM)` | **removed** | lost on both storage classes — see below |
| autopin (`CASCADIA_K3_AUTOPIN=1`) | built, never exercised, **needs a long run** | prior art finds static hot-set pinning helps cold start and loses in steady state. Two gotchas before measuring: the histogram is only persisted when autopin is enabled, so the FIRST enabled run always reports `pinned=0` and merely records; and the confidence ramp counts selections, of which K3 makes 92 layers x 16 = 1472 per token, so nothing pins below ~3.4 tokens and full confidence needs ~136. A 3-token run produces 4416 selections and stays under the floor — this cannot be evaluated at that scale |
| prefix cache | working, **on by default** (5% of free RAM; `CASCADIA_K3_PREFIX_CACHE=<bytes>` overrides, `=0` disables), any rank count | **measured 2.45x at 2 ranks and 2.60x at 1**, the same -64% of prefill bytes either way, so the saving comes from the reuse fraction rather than the topology. At 2 ranks: prefill bytes 142.06 -> 51.66 GB and prefill 649.4 -> 228.3 s at `reused=7 prompt=10`, saving in proportion to the reuse fraction. Byte-bounded LRU over the post-prefill layer states. It was reachable only from the pipeline path at first, so a single rank accepted the budget and ignored it; `step_single_stage` now takes the same route. Reuse needs a STRICT prefix, so resending an identical prompt never hits — the case it serves is the next turn, which resends the reply too |
| lane-lazy expert reads | **dropped, measured** | 29.1% of lanes are dead at the most aggressive threshold, but only 5.7% of `w3` PAGES, and 0.0% losslessly. The sparsity is real and too scattered to skip a page. `CASCADIA_K3_CHESS_PROBE=1` re-measures |
| n-gram speculative decode | research | bounded by expert-set overlap; measured reuse is ~33%, so expect ~1.2-1.4x, not 2x |

`ngram_draft.rs` and the `spec_decode.rs` primitives are already generic and pure,
but K3 rejection has to rewind *both* the KDA recurrent state and the MLA cache,
where K2.6 rewinds KV slots only. The per-channel FFN sparsity work in
`cascadia-int4-gemm` (CHESS) is a compute win there, and was expected to be an
I/O win here — the more valuable half — until it was measured; see below.

Chunked-scan KDA prefill stays deferred: the recurrence is 1% of wall time, so
there is nothing to win.

### Two ideas the measurements killed

Both were promising on paper, both had a threshold written down before any data,
and both came in far under it. Each was settled by a probe rather than by
building the feature — the probes are still in the tree, so either number can be
re-checked on other hardware or another model.

**Cross-layer gate prediction — dropped.** Score a layer's router against the
previous layer's hidden and see how much of the real top-16 lands in the top-`M`
(`CASCADIA_K3_GATE_PROBE=1`):

```
top16=41.2%  top20=46.2%  top24=50.5%  top32=57.2%   (n=2912)
```

The bar was 80% at `M = 24`. At the measured 50.5%, prefetching 24 experts catches
`0.505 x 16` = 8.1 of them and the other 7.9 still miss — about 32 fetches where 16
were needed. Every width loses, and `M = 32` loses hardest. The published results
come from models selecting a much larger fraction than 16 of 896, which is
exactly why the numbers had to be taken here rather than assumed.

**Lane-lazy expert reads (CHESS) — dropped.** Count dead lanes and, separately,
`w3` pages with no live lane on them (`CASCADIA_K3_CHESS_PROBE=1`):

```
t=0      lanes=0.0%   pages=0.0%
t=0.001  lanes=0.4%   pages=0.0%
t=0.01   lanes=3.4%   pages=0.0%
t=0.1    lanes=29.1%  pages=5.7%
```

The bar was 20% of pages. Losslessly nothing is skippable at all, and even at a
threshold that costs real precision only 5.7% of pages clear. The lane column is
the idea's promise and the page column is what the storage stack would deliver:
29.1% of lanes dead frees 5.7% of pages, because the dead ones are scattered and
a page with one live lane on it is read in full. Reporting only the lane figure
would have justified building something that saves nearly nothing.

## Resolved math

Extracted from the real upstream sources, vendored under
`tools/kimi_k3_ref/upstream/` (`modeling_kimi_linear.py` from the HF repo;
`kda_naive.py` + `kda_gate.py` from `fla-org/flash-linear-attention`, MIT). These
are the load-bearing details `config.json` alone could not answer.

### SiTU (`SituAndMul`) — model-global activation

```
gate, up = split(x, 2)                      # computed in f32, cast back
situ_a   = β · tanh(gate/β) · sigmoid(gate)          β = 4.0
up'      = linear_β · tanh(up/linear_β)              linear_β = 25.0
out      = situ_a · up'
```

MLP is `w2( SiTU(cat[w1(x), w3(x)]) )` — w1=gate, w3=up, w2=down. Three separate
matrices in the checkpoint (concatenated only at runtime), so the existing
gate/up/down section layout in `expert_mmap` still applies.

### AttnRes — learned mixture over a growing block stack

```python
_apply_attn_res(prefix_sum, block_residual, proj, norm):
    v       = cat([block_residual (T,nb,H), prefix_sum (T,1,H)], dim=1)
    k       = v * rsqrt(mean(v², -1) + eps)          # RMS-normalise, weight not yet applied
    score_w = norm.weight * proj.weight              # proj: Linear(H, 1, bias=False)
    probs   = softmax((k * score_w).sum(-1), -1)
    return probs @ v
```

Per layer (`_forward_attn_residual`), with two independent (proj, norm) pairs:

```
prefix_sum = hidden_in
if block_residual non-empty:
    hidden = _apply_attn_res(prefix_sum, block_residual, self_attention_res_proj, self_attention_res_norm)
if layer_idx % 12 == 0:
    block_residual = cat([block_residual, prefix_sum])   # grow the stack
    prefix_sum     = None
hidden     = self_attn(input_layernorm(hidden))
prefix_sum = (prefix_sum + hidden) if prefix_sum is not None else hidden
hidden     = _apply_attn_res(prefix_sum, block_residual, mlp_res_proj, mlp_res_norm)
hidden     = moe_or_mlp(post_attention_layernorm(hidden))
prefix_sum = prefix_sum + hidden
return prefix_sum, block_residual
```

Appends occur at layers 0, 12, …, 84 → **8 stack entries** over 93 layers.

### LatentMoE

```
topk_idx, topk_w = gate(x)                  # gate reads HIDDEN (7168), not the latent
x_lat = routed_expert_down_proj(x)          # 7168 -> 3584
y     = Σ_k w_k · expert_k(x_lat)           # experts in 3584, moe_inter 3072
y     = routed_expert_norm(y)               # RMSNorm(3584), applied to the COMBINED output
y     = routed_expert_up_proj(y)            # 3584 -> 7168
out   = y + shared_experts(x)               # shared on HIDDEN 7168, inter = 3072 × 2
```

Shared experts take **no `hidden_size` override** → they run at 7168, confirming
the 24.3 GB line in the shell table (an earlier revision flagged 3584 as
possible; it is not).

### Layer indexing — `linear_attn_config` is 1-indexed

`kda_layers` and `full_attn_layers` list layers **1-indexed**. Subtracting 1
yields an exact partition of `0..92` (69 KDA + 24 MLA), which the checkpoint's
tensor index confirms directly: layer 0 carries `self_attn.A_log` (KDA) and
layer 3 carries `self_attn.kv_b_proj` (MLA).

Read as 0-indexed the lists look wrong in two ways — layer 0 appears in neither
and `full_attn_layers` ends at 93, out of range — which is exactly the shape of
an off-by-one. The exporter shifts on load, so `manifest.json` is 0-indexed.

### KDA (Kimi Delta Attention)

```
q, k = q_proj(x), k_proj(x)                 # 7168 -> 96×128
v    = v_proj(x)                            # 7168 -> 96×128
q, k, v = silu(shortconv_k4(·))             # per-tensor causal depthwise conv + own conv state
g_raw = f_b_proj(f_a_proj(x))               # 7168 -> 128 -> 12288  (low-rank)
g     = -5.0 · sigmoid( exp(A_log)[h] · (g_raw + dt_bias) )      # lower-bound gate variant
β     = sigmoid(b_proj(x))                  # 7168 -> 96, per head
q, k  = l2norm(q), l2norm(k);  q *= 128^-0.5

# per head, state S is [K=128, V=128]:
S = S · exp(g)[:, None]
S = S + (β·k) ⊗ (v − Sᵀk)
o = Sᵀ q

o = FusedRMSNormGated(o, g_proj(x), act=sigmoid)   # full-rank gate, per head_dim
o = o_proj(o)                                       # 12288 -> 7168
```

`gate_lower_bound = -5.0` selects the `lower_bound · sigmoid(exp(A_log)·g)`
branch, **not** the `-exp(A_log) · softplus(g)` default
(`kda_gate.py`, `USE_LOWER_BOUND`).

### Gated NoPE MLA (24 layers)

`rotary_emb = None` and `assert use_nope` — **no rotation anywhere**. The
`qk_rope_head_dim=64` slice still exists dimensionally but passes through
unrotated, and `k_rot` is MQA-shared (`[B,1,T,64]`, broadcast over all 96 heads).
`scaling = q_head_dim^-0.5 = 192^-0.5`. Output gate:

```
g = sigmoid(g_proj(x))                      # 7168 -> 12288, full rank
attn_out = attn_out * g                     # after head-concat, BEFORE o_proj
attn_out = o_proj(attn_out)
```

The HF reference caches expanded k/v; we use glm5's absorbed-latent decode
instead (**576 floats/token** = 512 latent + 64 shared rot), which is
mathematically equivalent and the only memory-feasible form at long context.

## Findings from checkpoint verification

Both were found by validating against the real checkpoint's metadata (the
tensor index and the safetensors headers — ~1.6 MB fetched, no weights).

### 1. `A_log` is zero-padded in the checkpoint — RESOLVED

The released weights ship `A_log` as `[128]` on every KDA layer, while
`modeling_kimi_linear.py` declares `torch.empty(num_heads)` = **96** and fla's
gate does `A_log.view(H, 1)` with `H = g.shape[-2] = 96`. vLLM's Kimi-Linear
implementation also stores it per head. `view(96, 1)` cannot take 128 elements,
so on the face of it the published modeling file cannot run the published
weights.

Reading the actual 512 bytes settles it — the tensor is **96 real values
zero-padded to `head_dim`**:

```
idx 0..95  : nonzero 96/96, exp(A_log) in [0.471, 11.776]   (init: log(uniform(1,16)))
idx 96..127: nonzero 0/32,  all exactly 0.0
```

So the decay is per HEAD, as every implementation says, and the shell is
correct as written. The loader drops the padding.

Dropping it is not cosmetic: `exp(0) = 1` is *no decay*, so consuming the tail
would leave 32 heads' recurrent state never decaying — output that looks
plausible and degrades as context grows. `kda.rs` has a test pinning that
rationale so the truncation is not "simplified away" later.

### 2. `tokenizer.json` — resolved

K3 ships `tiktoken.model` plus a custom `TikTokenTokenizer`, and **no chat
template**, while every engine here loads `tokenizer.json` via the HF
`tokenizers` crate. The exporter now builds one and validates it token-for-token
against the reference tiktoken, failing the export on a mismatch.

The `pat_str` carries across unchanged rather than being translated: it uses
Java/ICU character-class intersection (`&&` against `\p{Han}`), which the
`tokenizers` build in use accepts verbatim. `tests/k3_tokenizer_pattern.rs` pins
that, because rewriting it by hand is what would silently mis-split text and
present as a model quality problem. The missing chat template still means
`/v1/chat/completions` falls back to legacy formatting.

## Open risks

- **Residency (highest)** — at N=4 there is ~0.5 GB of page cache for a 1.45 TB
  expert set. Throughput follows residency, so more nodes and `CASCADIA_K3_AUTOPIN`
  are the levers; the arithmetic cannot predict whether 29 GB of 32 stays stable
  under load.
- **Thin real-weight coverage.** The full export runs and generates correct text,
  but the automated suites still exercise a 6-layer synthetic model; every
  real-weight result in this document is a hand-run measurement, mostly n=1.
- **Export margin** — ~20 GB spare on a 1.6 TB disk, and only with
  `--free-source-shards`, which is destructive: a freed layer cannot be
  re-exported without re-downloading.
- **Linear-attention state cannot rewind.** The append-only generate loop is
  fine, but any future spec-decode / MTP accept-reject needs *state
  checkpointing*, not KV truncation.
- **Prefix cache** — a K3 snapshot must carry the KDA recurrent state
  (96×128×128 f32 × 69 layers ≈ 430 MB) plus conv windows, not just KV.
- `max_seq` sizing is *better* than glm5's: only the 24 full-attn layers scale
  with context (576 floats/token).
