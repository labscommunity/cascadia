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

**And they help far more than those residency figures suggest, because routing
is not uniform.** Measured over 104,880 real selections (`k3_skew` on a
`.k3_usage` from four prefills at the full top-16, 1.27 observations per slot so
it is not a sparsity artefact):

```
experts/layer   coverage   uniform   ratio
       1 (N=4)      4.3%      0.1%    38.8x
      35 (N=6)     44.6%      3.9%    11.4x
      68 (N=8)     61.9%      7.6%     8.2x
```

`coverage` is the share of routed reads the hottest experts of each layer
account for — the hit rate pinning them buys, and so the fraction the I/O term
falls by. Cache converts to hit rate at 8-39x the uniform rate, so the residency
column above UNDERSTATES what a node costs and buys: N=6 is not 3.9% of reads
served from RAM, it is ~45%.

At N=4 the distribution is just as skewed (38.8x) but the budget is ~1 expert
per LAYER — 0.5 GB is 28 experts spread over 23 layers — so only 4.3% is
capturable. The hot set exists; there is nowhere to put it. That is a budget
problem, not a routing one, and it is why `CASCADIA_K3_AUTOPIN` is worth little
at four nodes and a great deal at six.

Adding nodes is close to free on the wire here, which is worth stating because
the opposite is true for tensor-parallel serving: K3 is PIPELINE-parallel, so a
hop carries one activation — `(1 + max_blocks) * 7168` f32, ~258 KB — not a
per-layer all-reduce. Three hops at N=4 is under 2 ms against a 9 s token.

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

### The test suite was reporting green on tests it never ran

Eleven K3 integration tests skip themselves when their fixture is absent, and the
fixtures are gitignored. They exist on the machine they were generated on and
nowhere else — so on every remote host, "50 suites passed" meant those eleven
returned early without executing a line. Every green Linux run cited in this
document before that was found was, for those paths, vacuous.

`CASCADIA_REQUIRE_FIXTURES=1` now turns a missing fixture into a failure instead
of a skip, and the fixtures a plain checkout cannot produce are generated rather
than assumed. CI and any verification run should set it; without it a skip is
indistinguishable from a pass in the summary line.

This is the same failure as the three features in the table below that reported
themselves enabled and did nothing. A skip that prints like a pass and a flag
that sets without engaging are the same bug in different clothing: the signal
says yes and nothing happened. Prefer checks that fail loudly over checks that
decline quietly.

A related one in the exporter: its memory headroom was computed as a fraction of
the per-shard size with no ceiling, so a large shard could reserve tens of GB and
the export would thrash or die on a smaller box. It is now clamped to a fixed
upper bound.

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

Benchmarked on a Core Ultra 7 258V AI-PC (Arc 140V iGPU, AI Boost NPU) at K3's
real expert dims via OpenVINO.

This section was measured twice and wrong both times before it was right. The
first pass measured OpenVINO's `u4`, a LINEAR grid — K3's mxfp4 is the NONLINEAR
e2m1 grid, so those numbers described a kernel K3 cannot use. The second pass
fixed the grid but held the weights as an OpenVINO `Constant`, and every number
it produced was constant folding:

```
batch 32, HELD AS CONSTANT — these numbers are void
        u4 1374    e2m1 1306    f16 1218  GFLOP/s
```

`f16` does no decode at all, so it cannot be the slowest of the three. All three
landing within 12% is what folding looks like: OV decoded each variant once at
compile time and then ran the same pre-decoded f16 matmul. "e2m1 keeps 90% of
the `u4` bound" measured the compiler, not a kernel.

**A constant is also the wrong shape for K3.** There are 896 x 92 experts and
nothing stays resident; a real backend compiles ONE graph and streams weights
through it as an input. Re-measured that way, so nothing can fold:

```
w1/w3 [3072,3584], weights as a runtime INPUT
 batch    CPU     GPU     NPU
     1     4.3     5.4    n/a
    32    99.5   127.6    n/a     (f16 ceiling: GPU 280.4)
```

Three findings follow, and they close the device question rather than open it.

**Native MXFP4 exists in OpenVINO 2026.2 and K3 cannot reach it.** `f4e2m1` and
`f8e8m0` are real element types — exactly K3's format — and a
`Convert -> Multiply -> MatMul` over them compiles on CPU and GPU. But the fused
dequant+matmul op (`FullyConnectedCompressed`) pattern-matches only on
`v0::Constant` weights *and* scales, so a streamed weight can never select it, on
any device or version. And `f4e2m1` as a *Parameter* is rejected outright by the
GPU plugin. Native MXFP4 there is a resident-weight compression feature; K3 has
no resident weights.

**The iGPU has no MXFP4 kernel at all.** Intel gates MXFP4 dequant marking on
`arch >= xe3p`; Arc 140V is `xe2`. There are no e2m1 entries anywhere in the GPU
kernel selector or its OpenCL kernels. Corroborating: Intel ships gpt-oss-20b —
natively an MXFP4 model — as INT4-MIXED rather than running it native.

**The NPU is out on documented grounds, not measured ones.** Static shapes only,
and the supported inference types are F32/F16/U8. There is no 4-bit float weight
type to compile against, and a data-dependent gather into a data-dependent weight
tensor is not expressible under a compiler that fixes all shapes at compile time.

So for the ROUTED EXPERTS the CPU is not one option among three — it is the only
one.

**That verdict covers the experts and nothing else, which is narrower than the
question it was asked to answer.** Every row of it — the rejected `f4e2m1`
parameter, the constant-only fusion, the xe3p gate, the NPU's missing 4-bit type
and its static-shape rule — is a property of the streamed fp4 path. The
always-resident bf16 shell is the opposite workload on every axis that decided
those rows: bf16 not fp4, static shapes not a data-dependent gather, plain
supported matmuls not a compression path, and — the one that matters most — it is
loaded ONCE rather than costing 25.8 GB of transfer per token. The shell is also
roughly half the wall clock (`experts, share of wall: 46%` above).

None of it has been measured on a device. `cascadia profile-stages` / `place` /
`run-placement` already exist for exactly this question (see
`docs/perf/THREE_TIER_PLACEMENT.md`) and have never been pointed at K3. Two
things temper it in advance: at 4 nodes the shell is 29 GB per node against an
iGPU budget of ~16.5 GB, so only part of it could move; and #41 measured the
near-full-iGPU regime collapsing under memory pressure. KDA's recurrence stays on
the CPU in any case — only the projections feeding it are placeable.

**The kernel was the bottleneck here, and was fixed.** Measured in isolation on
the same machine (`simd_gemv_bench`, release), 3072x3584:

```
before  3.340 ms   6.6 GFLOP/s
after   0.572 ms  38.5 GFLOP/s   (58x over scalar)
```

Two stalls of similar size, neither visible alone: the AVX2 loop chained four
FMAs into one accumulator (~16 cycles of latency per group where the hardware
retires in ~2), and `e8m0_to_f32` used `powi` with a runtime exponent once per
group. Fixing only the powi gives 3.367 ms; only the accumulators, 3.599 ms.
Both, 0.572 ms — the FMA chain left idle cycles that hid the powi.

That verification was run against the real exported weights, not synthetic
sections, so the scale reassociation and the `from_bits` conversion are confirmed
on the actual e8m0 distribution rather than on a generator's.

Against the unfolded numbers above, the fixed kernel at 38.5 GFLOP/s is **9x
faster than OpenVINO's own streamed CPU path** at batch 1 and **7x faster than
the iGPU's**. The earlier claim that prefill was "a real iGPU opportunity, 3.35x"
came from the folded table and is withdrawn — and neither figure charges the
25.8 GB/token that would have to reach the device.

**A caveat on the CPU comparison.** 4.3 GFLOP/s is OpenVINO's *unfused* path,
which is what a streamed weight is guaranteed to get. It is a floor, not OV's
ceiling. The comparison is still the right one for K3, because K3 cannot use
constants — but it is not a statement that our kernel beats OpenVINO in general.

**The roofline says the kernel is not done.** At 38.5 GFLOP/s a 3072x3584 GEMV
moves 5.85 MB in 572 us = 10.2 GB/s effective, against ~99.5 GB/s measured on
this class of part. The working set also fits inside Lunar Lake's 8 MB
memory-side cache, so a hot kernel is not even reaching DRAM. Roughly 10x of
instruction-side headroom remains — though see the balance table below for why
that headroom is not where the wall-clock is.

### Which side of the balance K3 sits on

Per token K3 streams 25.8 GB. The compute column below counts only the ROUTED
EXPERTS — `2 x 16 x 92 x 33M` = 97.2 GFLOP — and that is the table's defect:

| | I/O | expert compute | ratio as published |
|---|---|---|---|
| bench host, rotational ~200 MB/s | ~129 s | 14.7 s | I/O, 9:1 |
| AI-PC, NVMe 3566 MB/s | **7.2 s** | **2.5 s** | I/O, 2.9:1 |

**The bf16 shell is missing from it, and the shell is the LARGER half.** ~112 GB
of bf16 is 56B parameters, all of them dense-used every token: 112 GFLOP against
the experts' 97.2. Being resident it costs no NVMe, but it still has to cross
DRAM — 112 GB per token, which at the ~99.5 GB/s this class of part measures is a
**1.13 s/token floor** before any inefficiency in the kernel reading it.

Fold in just that floor and the balance moves:

```
as published   7.20 / 2.50 = 2.9:1
+ shell floor  7.20 / 3.63 = 2.0:1     and 2.0:1 is the OPTIMISTIC end —
                                        it assumes the bf16 GEMV achieves full
                                        memory bandwidth, which the fp4 kernel
                                        did not until it was fixed
```

So the "compute is worth at most 1.34x end-to-end" conclusion that was drawn from
the 2.9:1 figure does not hold; at 2.0:1 it is 1.5x or better. Two rounds of
expert-kernel work were deprioritised on the strength of a ratio that had left
out more than half the arithmetic.

**That 1.13 s is a floor the kernel already reaches — measured, not assumed.**
`dot_bf16w_avx2` accumulates into two registers, and the fp4 kernel's 5.84x came
from exactly that shape of defect, so the analogy was worth testing. It does not
hold. `bf16_accumulator_sweep` (`dsv4/math.rs`) measures 2, 4 and 8 accumulators
against the shipped kernel at real shell shapes:

```
                       shipped   2acc    4acc    8acc     parallel (48 thr)
kda qkvo [12288,7168]  6.6       6.6     6.6     6.9  GB/s   59.3 GB/s
shared   [ 6144,7168]  7.4       7.4     7.4     7.5         76.1
up_proj  [ 7168,3584]  8.0       7.7     7.9     8.3        104.0
```

Every variant ties — the loop is not FMA-latency-bound, so the accumulator count
is not the lever it was on fp4. And the shipped path (`linear_bf16_w`, rayon over
rows) reaches 59-104 GB/s, which on a 6-channel DDR4-2933 host is at the memory
roof. **The bf16 shell is bandwidth-bound and there is no kernel win in it.**

The single-thread columns are the trap this nearly fell into: 6-8 GB/s next to a
~100 GB/s roof reads like enormous headroom, and means nothing, because the
production caller is parallel. Measure the entry point that actually runs.

The NVMe figure is measured at K3's access pattern — 16 scattered 17.5 MB slices,
16-way concurrent, scratch larger than RAM since Windows has no `drop_caches`
(`bench_fetch_win.py`). Stable at 2x and 3.8x RAM (3376 / 3566 MB/s), and it
agrees with the Linux NVMe number taken with real cache drops.

**A longer run on an AI-PC gets less: 2858 MB/s**, not 3566. `nvme_readbench`
over 2800 synthetic bins (49 GB, RAM is 32) reading 44.92 GB in one pass:

```
explicit pread x16   2858 MB/s    98.2 ms/token
mmap + touch         1386 MB/s   202.5 ms/token    2.06x apart
```

Order-independent — running the phases in either order moves each by ~2%,
because a 45 GB working set is far enough past RAM that the page cache cannot
skew it. A SHORT run is a different story: reading 3.37 GB of a freshly written
49 GB set reported 6303 MB/s, more than double, because the bins were still
cached. Any fetch number from a run that does not exceed RAM is measuring the
page cache.

Take 2858 MB/s as the sustained figure — it is the one that matches decode,
which streams continuously rather than in bursts — and it makes I/O **9.0
s/token**, not 7.2. The disk was 82% full, which is also the realistic state.

"Decode is 99% expert I/O" describes the rotational host. On NVMe the two are the
same order, and which one binds moved with the kernel fix above. Re-measure the
split on an AI-PC billing shell, experts and I/O SEPARATELY before trusting any
ratio in this section — the profiler bucket that produced the table never
attributed shell time.

### Where the remaining speed is

What dominates depends on the storage (see above): ~9:1 I/O bound on rotational,
and on NVMe no better than ~2:1 once the shell is counted.

**The one lever with headroom well past any of this is `top_k`.** Routed bytes
are exactly `top_k * moe_layers * expert_bytes`, so they fall in direct
proportion. Everything else here competes for the ~1.5x on the compute side;
this competes for the other half.

`--top-k-override` is wired (`K3Manifest::effective_top_k`). The quality screen
has now run on real weights — `k3_topk_probe`, four prompts chosen to span easy
recall through contested continuation, comparing next-token logits at k=4
against the same prompt at k=16:

```
prompt          k=16 predicts   k=4 argmax   top5   KL(nats)
factual_easy    " Rome"         same         4/5    0.073
open_ended      " write"        same         5/5    0.096
narrative       " the"          same         5/5    0.062
reasoning       " fade"         same         4/5    0.365
```

Nothing was ruled out: the top token survives at a QUARTER of the bytes on every
prompt, including ones where the reference was a weak hedge (" the") rather than
an overdetermined answer.

**Read the last row before acting on the rest.** `reasoning` is the only prompt
needing a contextual chain carried across 92 layers (roses -> flowers -> fade),
and its distribution moved 4-6x further than any other. The top token held, but
by much less margin. A single-step screen is least trustworthy exactly there,
because that is the error that compounds over a generation — and K2.6 found a
real cliff (K=4 fine, K=3 a regression) that a screen like this would have
walked straight past.

So: promising, not settled. `top_k` stays at the manifest value until a
generation eval on the survivors, because unlike autopin this is the one change
that alters output.

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
| AVX2 fp4 expert kernel | done | **58x over scalar**, 0.572 ms per 3072x3584 GEMV = 38.5 GFLOP/s. A previous "1.79x" here was an end-to-end token delta, not kernel throughput |
| explicit concurrent reads | done, **default** (`CASCADIA_K3_READ=0` opts out) | **measured +8.5%** steady-state decode, -1.6% prefill, 2 runs per side. Both phases must use the same strategy — see the fetch section |
| `madvise(MADV_RANDOM)` | **removed** | lost on both storage classes — see below |
| autopin (`CASCADIA_K3_AUTOPIN=1`) | built, never exercised, **warms over ~136 tokens** | prior art finds static hot-set pinning helps cold start and loses in steady state. Two gotchas before measuring: the histogram is only persisted when autopin is enabled, so the FIRST enabled run always reports `pinned=0` and merely records; and the confidence ramp counts selections, of which K3 makes 92 layers x 16 = 1472 per token, so nothing pins below ~3.4 tokens and full confidence needs ~136. A 3-token run produces 4416 selections and stays under the floor. The histogram MERGES on load, though, so the ~136 tokens accumulate across runs rather than needing one long session: any sequence of runs with the flag set warms it, and a long-lived worker warms itself |
| prefix cache | working, **on by default** (5% of free RAM; `CASCADIA_K3_PREFIX_CACHE=<bytes>` overrides, `=0` disables), any rank count | **measured 2.45x at 2 ranks and 2.60x at 1**, the same -64% of prefill bytes either way, so the saving comes from the reuse fraction rather than the topology. The derived default was checked with no env set at all: 555 s against 548 s for a hand-set budget, and the same 103.32 GB prefill, so it behaves as the tuned value. At 2 ranks: prefill bytes 142.06 -> 51.66 GB and prefill 649.4 -> 228.3 s at `reused=7 prompt=10`, saving in proportion to the reuse fraction. Byte-bounded LRU over the post-prefill layer states. It was reachable only from the pipeline path at first, so a single rank accepted the budget and ignored it; `step_single_stage` now takes the same route. Reuse needs a STRICT prefix, so resending an identical prompt never hits — the case it serves is the next turn, which resends the reply too |
| close the CPU kernel gap | **done, measured 5.8x** | 0.572 ms per 3072x3584 GEMV, 38.5 GFLOP/s. Per-token compute 14.7s -> 2.5s, which puts an AI-PC node back to I/O bound |
| batched GEMM for prefill | **done, measured 1.0-1.4x**, far under forecast | prefill routes several rows to one expert and each was re-decoding the whole 17.5 MB section; `gemm` now decodes a weight row once and dots it against every row. Peaks at 1.42x around 8 rows, ~1.0x at 2. The forecast was 2.6x, taken from OV's batch-32 number, and it was the wrong comparison: OV does a real blocked GEMM that reuses activations in registers and cuts FLOPs per row, while this only hoists the decode. Bit-identical to the per-row path by construction, not merely close — prefill batches, decode does not, and the prefix cache hands one to the other. `CASCADIA_K3_GEMM=0` opts out |
| int8 VNNI expert kernel | built, **off by default** (`CASCADIA_K3_VNNI=1`), **measured 1.57-1.60x on an AI-PC** | `vpdpbusd` multiplies bytes and the doubled e2m1 grid `{0,1,2,3,4,6,8,12}` already fits in `i8`, so the fp4 decode feeds it directly: one `vpdpbusd` and a convert per 32 columns against ~25 ops widening to f32. Off by default because it is the only inexact path here — activations quantise to int8 per group of 32. That costs **0.027% of accumulated magnitude** on uniform random activations, which is the BEST case and not evidence about a real model: the quantiser takes an amax per group, so one large value costs the other 31 their resolution, and LLM activations carry exactly that. `vnni_accuracy_under_activation_outliers` plants one outlier per group and sweeps it — worst-element error scales linearly with the ratio, and the dot error peaks near **0.46%** at 100x before falling again (a large enough outlier dominates the sum and is itself represented exactly). Still small, but 17x the figure usually quoted. The question that decides the default is not single-dot error but whether token predictions change over 92 layers x 16 experts, which `k3_topk_probe` already measures and which needs real weights. Three encodings are built and chosen at runtime. Lunar Lake reports `avxvnniint8`, so it takes `vpdpbssd` — signed x signed, which needs neither of the two `vpsignb` the unsigned form does: 1.48x -> **1.57-1.60x** against a 38.8 GFLOP/s f32 baseline. `vpdpbusd` (VEX) and the AVX512-VNNI form remain for parts without it; the latter is what lets the kernel be verified off-target, since no AI-PC has AVX-512. A Xeon 6252 measured 1.31-1.34x, and the prediction that an AI-PC would be LOWER (faster f32 baseline, same memory system) was wrong in the useful direction. Not yet wired into the batched `gemm`, so prefill and decode cannot both benefit |
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
  Until `CASCADIA_REQUIRE_FIXTURES=1` existed the coverage was thinner still than
  that sentence implied — see above.
- **The device verdict rests on OpenVINO's current behaviour, not on silicon.**
  The iGPU is ruled out by an architecture gate (`xe3p`) and by a graph pattern
  that requires constant weights. Both are software, and Intel is actively moving
  in this area — a newer runtime added an offload-to-disk MoE path with an LRU
  device cache, gated to `u4`/`i4` and reachable on this hardware. If the iGPU is
  ever revisited, requantising mxfp4 -> u4 offline is the entry point, and the
  cost is an accuracy delta that has not been measured.
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
