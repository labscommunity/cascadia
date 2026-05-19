# zstd compression for K2.6 expert safetensors at rest — investigation, not shipped

**Branch:** `perf/zstd-expert-storage-093`
**Verdict:** Investigation only. Implementation **skipped**. The
compression ratio is dominated by the format K2.6 ships in (packed
int4, near-entropy) and the wall-clock arithmetic is decisively
negative on every storage tier we target.

## Premise

K2.6 is 553 GB on disk in safetensors form. The task brief
hypothesised that `zstd-3` on f16/int4 weights would compress
1.4-1.8× and shrink the on-disk footprint to ~330 GB, with
decompression "worth tens of seconds added to cold start." It also
flagged the obvious risks: cannot mmap compressed bytes (loses the
iter 080 shard-lazy property of `SafetensorsExpertSource`),
decompressed bytes consume RAM (eating into the iter 054 expert
`mlock` headroom on the miner), and the win may not survive contact
with measured throughput.

This investigation answers two questions concretely:

1. What ratio does `zstd` actually achieve on the on-disk format
   K2.6 uses today?
2. At what storage tier (if any) does the wall-clock saving from
   smaller reads beat the wall-clock cost of decompression?

## On-disk layout of K2.6 experts

Per `crates/tahoma-int4-gemm/src/safetensors_source.rs` the expert
tensors that consume the bulk of the 553 GB are stored in
`compressed-tensors` int4 group-32 form. Each expert has three
projections × two tensor kinds:

| Tensor kind        | dtype      | What it is                                          |
|--------------------|------------|-----------------------------------------------------|
| `weight_packed`    | I32        | int4 weights, eight nibbles packed per `i32` lane   |
| `weight_scale`     | BF16       | per-group-32 fp scale factor                        |
| `weight_shape`     | I32 (8 B)  | original [out, in] dimensions                       |

Measured on the first MoE shard (`model-00010-of-000064.safetensors`,
9.5 GB on disk, all of layer 9's 32 experts):

```
packed bytes (I32, int4):  8.456 GB    (88.9 % of shard)
scale  bytes (BF16):       1.057 GB    (11.1 % of shard)
shape  bytes (I32):        0.000 GB    (~1 KB total)
```

The packed-int4 stream is what dominates. It is **not** raw float
data — it is dense 4-bit symbols already at high entropy.

## Compression ratios (zstd 1.5.5 on miner Xeon Gold 6252)

Single layer of experts (32 experts × 6 tensors = 192 expert tensors)
extracted from `model-00010-of-000064.safetensors` then compressed in
two streams (packed-int4 vs bf16 scales):

| Stream  | Level | Original    | Compressed  | Ratio    | Compress wall | Throughput |
|---------|------:|------------:|------------:|---------:|--------------:|-----------:|
| packed  |     1 |   8455.7 MB |   ~7670 MB  |   1.10×  |        47.4 s |  ~178 MB/s |
| packed  |     3 |   8455.7 MB |    7688.1 MB|   1.10×  |        52.2 s |  ~162 MB/s |
| packed  |     6 |   8455.7 MB |   ~7670 MB  |   1.10×  |        61.2 s |  ~138 MB/s |
| packed  |     9 |   8455.7 MB |   ~7670 MB  |   1.10×  |        72.5 s |  ~117 MB/s |
| packed  |    19 |   8455.7 MB |    7690.1 MB|   1.10×  |          slow |       slow |
| packed  | 22+LR |   8455.7 MB |    aborted  |        – |  >74 min CPU  |          – |
| scale   |     1 |   1057.0 MB |    ~620 MB  |   1.70×  |         6.8 s |  ~156 MB/s |
| scale   |     3 |   1057.0 MB |     569.2 MB|   1.86×  |        17.2 s |   ~62 MB/s |
| scale   |     6 |   1057.0 MB |    ~538 MB  |   1.96×  |        38.3 s |   ~28 MB/s |

Trained-dictionary test (`zstd --train` on 64 expert samples,
128 KB dict, `zstd -3 -D dict`):

| Stream  | With dict | Improvement vs no-dict |
|---------|-----------|------------------------|
| packed  | 1.10×     | none (90.92 % → identical) |
| scale   | 1.86×     | none |

**Combined per-shard ratio: 9.51 GB → 8.26 GB = 1.15× (saves 1.26 GB).**

Extrapolating across 64 shards:

```
uncompressed: 608.8 GB     (64 × 9.51 GB; matches 553 GB ls -l reality
                            after accounting for small + dense layer 0/63 shards)
compressed:   528.5 GB
saving:        80.3 GB     (~13 %)
```

The 1.4-1.8× hypothesis in the task brief is **wrong for K2.6
experts**. It applies to *raw* bf16/f16 weight tensors — we
verified this against the dense layer-0 `up_proj` (bf16, 252 MiB →
197 MiB = 1.28×), and against the bf16 scale tensors above (1.86×).
But the 89 % of the model that is packed int4 is already entropy-
dense and effectively incompressible.

## Decompression cost vs raw mmap

Per-shard wall-clock on miner (NVMe-backed `/mnt/external_ssd`,
measured at ~3 GB/s sequential):

| Operation                                          | Wall clock | CPU usage |
|----------------------------------------------------|-----------:|----------:|
| Cold-cache `cat packed.bin > /dev/null` (raw read) |    2.84 s |       63 % |
| Cold-cache `zstd -dc -T0 packed.zst > /dev/null`   |   15.26 s |      140 % |
| Warm-cache `zstd -d -T0` (CPU-bound)               |   11.98 s |      124 % |
| Warm-cache `zstd -d -T1` (single-threaded)         |   17.74 s |      126 % |

Decompression throughput (warm, multi-thread): ~682 MB/s of *output*.

Single-frame `.zst` doesn't usefully parallelise — `-T0` reaches
~1.4 cores. Multi-frame would help but adds tooling complexity and
breaks compatibility with the standard `zstd` CLI workflow.

## Break-even SSD bandwidth

If we compressed and read from disk:

```
wall_raw      = orig_bytes / disk_BW
wall_compr    = comp_bytes / disk_BW + orig_bytes / decomp_BW
wall_compr < wall_raw  iff  disk_BW < (orig - comp) * decomp_BW / orig
```

Plugging in the packed numbers (`orig=8456 MB`, `comp=7688 MB`,
`decomp_BW=682 MB/s`):

```
break-even disk BW = (8456-7688) * 682 / 8456 = 62 MB/s
```

**Compression saves wall-clock only when disk bandwidth is below
62 MB/s.**

Reference points:

| Storage class                          | Typical bandwidth | Verdict |
|----------------------------------------|------------------:|---------|
| Miner SSD (NVMe, measured)             |          3.0 GB/s | loses 12 s/shard |
| Consumer NVMe (e.g. Samsung 980)       |        2-3 GB/s   | loses ~5-10 s/shard |
| Consumer SATA SSD (e.g. 870 EVO)       |          500 MB/s | loses ~8 s/shard |
| Lunar Lake AI PC eMMC / cheap NVMe     |       400 MB/s+   | loses ~7 s/shard |
| USB 3.0 external drive                 |       200 MB/s+   | loses ~4 s/shard |
| Gigabit Ethernet NFS (single stream)   |          100 MB/s | loses ~2 s/shard |
| 10/100 Mbps Ethernet                   |        ~10 MB/s   | wins ~30 s/shard |
| Spinning rust HDD                      |          100 MB/s | loses ~2 s/shard |

The matias/AI-PC scenario the task brief asked about: matias is a
Lunar Lake box with built-in NVMe at GB/s class. Same verdict as
miner — loses wall clock.

## The "RAM headroom" tax — quantified

Even setting wall clock aside, compression forces the loader to
hold decompressed bytes in process memory because `mmap` of
compressed data is meaningless. Today on miner:

- `SafetensorsExpertSource` opens lazily (iter 080 discovery):
  shard mmaps are created only when a tensor in them is first
  requested. At start-up, VMA cost is zero.
- The OS pages in expert tensor pages on demand. Cold experts
  consume zero RAM until the router picks them.
- Iter 054 pins the **active** experts via `mlock` to keep them
  off the LRU eviction list under memory pressure.

A zstd-compressed path would have to either:

1. **Decompress eagerly at load** — costs `~13 min` total (64
   shards × 12.4 s extra) plus holds the full ~525 GB of
   decompressed expert bytes resident. Miner has 133 GB RAM.
   Catastrophic.
2. **Decompress one expert at a time on demand** — each fault
   into a cold expert now costs an additional ~30-50 ms (per-
   expert decompression at 682 MB/s × ~24 MB/expert), serialised
   on a CPU core that the dispatch hot path needs. This collides
   with the iter 057 async kernel scheduler attempt and the iter
   054 expert pinning logic, neither of which assumes a CPU-bound
   decode in the critical path.
3. **Per-frame compressed shards with seekable index** — engineering
   work to ship a custom format. Saves 80 GB at the cost of
   building, validating, and maintaining a new on-disk format that
   does not interop with HuggingFace `safetensors`. Not worth the
   80 GB.

## Why this is different from "zstd usually wins on tensor data"

The "zstd on tensor data" rule of thumb (1.4-1.8×) assumes:

- The tensors are stored in their original numerical dtype
  (f32/f16/bf16), where there is real redundancy in the
  exponent + sign + low mantissa bits.
- The exponent distribution is non-uniform (weights cluster near
  zero), so entropy coding wins.

K2.6's expert weights have already been compressed by the
`compressed-tensors` quantisation step:

- The 4-bit symbols span the full int4 range fairly uniformly
  (quantisation distributed across the per-32 group dynamic range).
- The I32 lanes pack eight independent 4-bit symbols. Adjacent
  bytes are uncorrelated.
- That's the *whole point* of quantising the model — you converted
  redundant fp bits into dense binary symbols. Re-running an
  entropy coder on dense binary symbols saves nothing.

The 1.86× we measured on the bf16 *scale* stream is consistent
with the rule of thumb — those are still real fp numbers and they
do have exploitable redundancy. But scales are only 11 % of the
on-disk footprint of an expert shard, so even with scales fully
optimised, the combined ratio caps at ~1.15×.

## Where this might be worth re-asking

Compression at rest becomes interesting only if:

1. **The model ships in raw fp form (not pre-quantised).** Tahoma
   does intend to support `bf16` or `f16` source weights for
   non-quantised models (Llama, Mistral, Qwen via `tahoma shard`).
   For those, the 1.4-1.8× hypothesis is correct and the math
   might re-open. The point matters for the K2.6 path only.
2. **Network-backed storage at < ~50 MB/s.** Loading from a remote
   NFS or HTTP origin over a slow link. Not the target hardware
   profile (Intel AI PCs ship with NVMe).
3. **A format that supports random-access decode per expert (e.g.
   per-expert zstd frame with seekable index).** Would defer the
   decompression cost to first-touch per expert and avoid the
   "decompress 525 GB into RAM" antipattern. Engineering work that
   pays back only the 80 GB; not on the critical path.

## What this branch ships

- This document.
- No change to `tahoma-int4-gemm`, `SafetensorsExpertSource`, the
  runner, or the on-disk layout.
- No new dependency (`zstd` crate **not** added to `Cargo.toml`).

That is the entire deliverable. Adding compression infrastructure
that the math says nothing should use would just be code rot, and
worse, it would create a tempting wrong default for future model
ports.

## Pointers

- `crates/tahoma-int4-gemm/src/safetensors_source.rs` — the
  `SafetensorsExpertSource` we would have had to teach about
  decompression. Currently mmap-only.
- `docs/architecture/selective-recomputation.md` — sister
  investigation in the same "don't add a code path nothing will
  exercise" spirit.
- Iter 054 (`perf/expert-pinning-054`) — the `mlock` headroom
  this change would consume.
- Iter 080 (`perf/lazy-expert-load-080`) — the shard-lazy property
  this change would defeat.
- `model-00010-of-000064.safetensors` — the shard we measured. Any
  MoE shard between `model-00002-of-000064.safetensors` and
  `model-00063-of-000064.safetensors` would give the same answer.
