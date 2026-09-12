# Inkling on a fleet: where the tokens per second come from

A sizing note for the Inkling 975B port ([`../architectures/inkling.md`](../architectures/inkling.md)):
what bounds single-stream decode on Intel AI-PC fleets, why "160 GB/s ÷ 20 GB × 12
boxes ≈ 80 tok/s" does not follow, and what expert-level routing across boxes
would and would not buy. Every constant below is measured on this branch or in
[`../TENSOR_PARALLEL.md`](../TENSOR_PARALLEL.md); the rest is arithmetic.

## 1. Bytes per token (the invariant)

Decode at batch 1 is memory-bound: a token costs the bytes of every weight it
touches, once, wherever those weights live. From the export's dimensions:

| component (per token) | params touched | today's export | all-int4 |
|---|---|---|---|
| attention, all 66 layers | 8.58 B | 17.2 GB (bf16) | 4.8 GB |
| routed experts, 6 of 256 × 64 layers | 21.7 B | 12.2 GB | 12.2 GB |
| shared experts, 2 × 64 layers | 7.25 B | 4.1 GB | 4.1 GB |
| dense MLP, layers 0–1 | 0.91 B | 0.5 GB | 0.5 GB |
| unembed | 1.24 B | 2.5 GB (bf16) | 0.7 GB |
| **total** | **~39 B active** | **36.5 GB** | **22.3 GB** |

One routed expert is 56.6 M parameters = **31.85 MB** at int4 group-32 (+ scales).
"41 B active × 0.5 byte ≈ 20 GB" is the traffic for one **token across all 66
layers**, not per expert.

Bandwidth-bound ceiling, single stream, weights resident (no disk), whole
pipeline:

| memory bandwidth | today (36.5 GB/tok) | all-int4 (22.3 GB/tok) |
|---|---|---|
| 58 GB/s (miner, Cascade Lake, measured) | 1.6 tok/s | 2.6 tok/s |
| ~100 GB/s (Lunar/Panther Lake LPDDR5X sustained) | 2.7 tok/s | 4.5 tok/s |
| 136–154 GB/s (LPDDR5X-8533/9600 theoretical) | 3.7–4.2 tok/s | 6.1–6.9 tok/s |
| 160 GB/s (the estimate's figure) | 4.4 tok/s | 7.2 tok/s |

So 8 tok/s is the **ceiling for the entire model with everything at int4 at peak
bandwidth**, and it is one stream's number no matter how many boxes share the
weights (§3).

## 2. Measured per layer (`inkling_layer_dump`, 23-token prompt)

Miner (Xeon Gold 6252, 24c/48t, 192 GB, experts paging from SATA SSD), 24
threads, after the concurrent-expert schedule and residency-adaptive reads:

| layer | cold (experts paging from SATA) | page-cache warm (RAM-resident) |
|---|---|---|
| dense (0, 1) | 12–13 ms/token | 12 ms/token |
| MoE (2–7) | 72–95 ms/token | **13.4–14.2 ms/token** |
| batched prefill, MoE layer | ~250 ms per 23 tokens (11 ms/token) | 237 ms (10.3 ms/token) |

Mac Pro 2019 (Xeon W-3275M, 28c/56t, 1.5 TB, macOS 12), the whole 512 GB
export **wired in RAM** (`CASCADIA_INKLING_PIN_EXPERTS=1`), same prompt —
the per-layer numbers behind the resident run in §2a:

| config (all bit-identical) | MoE ms/token | dense ms/token | prefill ms per 23 tokens (MoE layer) |
|---|---|---|---|
| serial experts, per-token copy, 56 threads (the port as first merged) | 37 | 20 | 950 |
| concurrent experts, direct off the mapping, 56 threads | 45 | 22 | 600 |
| concurrent experts, pinned, 56 threads | 33 | 22 | 650 |
| concurrent experts, pinned, **28 threads (physical cores)** | **10.4** | **9.0** | **173** |

Three things the resident box taught: (1) macOS re-faults file-backed pages
it has already cached on every touch, so computing off an unpinned mapping is
*slower* there than copying the bin first (the opposite of Linux) — hence the
pin mode; (2) the row-parallel GEMVs lose 3× to hyperthreads on macOS (56 →
28 threads: 33 → 10.4 ms) while Linux does not care (48 → 24: 14.2 → 13.4),
so the engine now sizes rayon to the physical cores; (3) the "eager" expert
mode is the dequantised dev path (4× the bytes, 2.5× slower) — resident int4
means a mapping the OS keeps, or the pin.

A resident MoE layer moves 264 MB of bf16 attention + 8 × 31.85 MB of experts =
519 MB per token. At the Mac Pro's 10.4 ms that is ~50 GB/s against a 6-channel
DDR4-2933 ceiling of ~140 GB/s, i.e. the shell runs at ~35 % of bandwidth
(bf16 attention GEMV + per-row dequant, no SIMD work on the shell yet).
Whole-model, resident, on this box: 2 × 9 + 64 × 10.4 ≈ 0.68 s + head ≈
**0.7–0.75 s/token, ~1.4 tok/s**. The miner's serving numbers (0.05–0.13
tok/s) are the SSD paging ~6× on top of its own ~0.9 s/token resident layer
sum; they say nothing about resident performance.

### 2a. Resident, end to end (Mac Pro, export pinned, 28 threads)

Same binary, prompts and greedy answers as the miner runs (byte-identical
text). Wiring the 490 GB at load took 812 s (~0.6 GB/s, page cache cold).

| request | wall | per token |
|---|---|---|
| TTFT, 25-token prompt | **12.5 s** (39.6 s before the schedule fixes) | 0.5 s/token prefill — per token like decode; row-batched prefill is a follow-up |
| 26-token answer | 28.7 s | 0.65 s/token |
| 64-token answer | **53.9 s** (227 s before) | **0.66 s/token = 1.5 tok/s** (0.33 before) |
| thinking on, 25 tokens | 28.4 s | 0.66 s/token |
| driver + 3 workers, loopback, 64 tokens | 59.9 s | 0.74 s/token: the star costs ~1.2 ms per MoE layer on one box |

So the resident single-stream number for today's export on a 1.5 TB
Cascade Lake box is **1.5 tok/s**, 4.5× the first resident run and ~2.6× off
the bandwidth floor of §1 — the shell items in §5 are the rest.

## 3. Why a pipeline does not multiply per-box bandwidth

Layer sharding (what `--engine sparse-moe` does across ranks) gives box *i* a
contiguous slice of layers with **all 256 experts of those layers**. A token
visits the boxes in order; each reads only the ~1/N of the per-token bytes that
its slice needs. The per-token time is the **sum** over boxes, not the minimum:

```
t_token = Σ_i (bytes_i / bw_i) + hops  = 36.5 GB / 100 GB/s + 12 × ~0.5 ms ≈ 0.37 s
```

Twelve Panther Lake boxes therefore decode one stream at the *same* 2.7 tok/s a
single 512 GB-RAM box with the same bandwidth would, plus hop latency. What N
boxes buy is (a) the weights fitting in RAM at all — 12 × 64 GB = 768 GB holds
the 512 GB artifact, 8 × 32 GB Lunar Lake does not — and (b) **throughput**:
while box 3 works on stream A's token, box 2 can work on stream B's. With ~12
streams in flight the aggregate approaches 12 × 2.7 ≈ 30 tok/s (all-int4: ~50);
"80 tok/s" is reachable only as that kind of aggregate, never as one stream.

No box in a layer pipeline is ever "routed through without the needed
experts": every layer's attention must run on every token, and the experts a
layer selects are on the box that owns the layer. The 250 unselected experts per
layer are never read.

## 4. What expert-level coordination across boxes would change

Two things the pipeline cannot do: read one token's expert set **in parallel on
several boxes**, and read the attention weights in parallel (tensor parallelism).
Both need per-layer network rounds; both are latency-bound by the LAN.

Measured constants: 225 µs per all-reduce round-trip on the cabled 192.168.0.x
LAN (`TENSOR_PARALLEL.md`, 32 rounds = 7.2 ms/token); 22 ms p50 / 102 ms p99 over
the Tailscale DERP relay. A hidden state is 24 KB (f32) — bandwidth is not the
issue, round trips are.

| topology (12 boxes, 64 GB each, all-int4 export, 100 GB/s) | network rounds / token | est. s/token | est. tok/s (1 stream) |
|---|---|---|---|
| **layer pipeline (today)** | 13 one-way hops | 0.22 + 0.006 | **~4.5** |
| expert parallel (experts spread over boxes, attention on the driver) | 64 layers × 2 (dispatch + gather) | attention 0.05 + shared 0.04 + experts 64 × (0.3 ms read + 0.45 ms RTT) ≈ 0.14 | ~7 |
| expert parallel + tensor-parallel attention | 128 all-reduces + 128 dispatch/gather | 0.04 + 0.03 + 0.06 ≈ 0.13, ~0.25 with p99 tails | ~4–8 |
| same over the DERP relay | 256 rounds × 22 ms | ~5.6 | 0.2 |

Built (`--ep-workers` / `--ep-worker-index`, see `inkling.md`) and measured on the
miner with driver + 3 workers on the one box over loopback TCP, real export, SSD-paged:
answers identical to single-process, cold 25-token TTFT 42 s (vs 107–176 s), 16-token
answer 121 s vs 174–180 s (0.13 vs 0.09 tok/s) — the parallel-expert-read effect of
the paged regime, with a ~0.15–0.2 ms loopback round per MoE layer. Page-cache state
differed between the runs, so treat the ratio as indicative until a cold-cache A/B.

Read across: in the **RAM-resident** regime the best expert-parallel design is
~1.5–2× the pipeline, not an order of magnitude, and only on a switched LAN with
sub-millisecond tails; over a relay it is 20× *slower* than the pipeline. The
bytes per token are the same in every row; the rows differ only in how many
boxes read them at once, and 200+ network rounds per token eat most of that.

Where expert-level coordination is a large win is the **RAM-starved** regime —
boxes that cannot hold their share, so experts page from NVMe (this is what
OpenVINO 2026.3's MoE weight offloading, `OFFLOAD_RATIO` + LRU expert cache, does
inside one box). A paged pipeline reads each box's experts *serially* (5.5 layers
× 6 × 32 MB ≈ 1 GB per box per token at ~5 GB/s NVMe ≈ 0.2 s, × 12 boxes in
sequence ≈ 2.4 s/token); expert-parallel dispatch reads a layer's 6 experts on
up to 6 boxes' NVMes at once (≈ 6 ms per layer + RTT ≈ 0.5 s/token) — about
**5×**. That is the honest shape of the idea: a way to make 8 × 32 GB Lunar Lake
boxes serve a 512 GB model at ~2 tok/s instead of ~0.4, not a way to get 80.
Expert popularity in a 256-expert model is flat enough that an LRU of "hot"
experts does not change this much (each token needs ~6 of 256 per layer, nearly
uniformly over a conversation); locality comes from batching, not from caching.

## 5. What actually moves the single-stream number, in order

1. **Resident weights.** The miner's 0.05–0.13 tok/s is paging. A 12 × 64 GB
   pipeline or any ≥ 640 GB host gets the resident baseline for free — measured
   **~1.4 tok/s** on the Mac Pro (§2, §2a) with the schedule fixes of
   2026-09-11 (concurrent experts, residency-adaptive reads, pin mode,
   physical-core pool: 3 s → ~0.75 s per token on that box).
2. **int4 attention, shared experts, unembed** in the export: 36.5 → 22.3 GB per
   token, **1.6×**. The shell's bf16 GEMV path becomes the same int4 GEMV the
   experts use.
3. **Shell efficiency**: 55 % → ~85 % of bandwidth (fused q/k/v/r GEMV, the
   AVX-512/AVX2 int4 kernels for the projections, no per-row bf16 round trips)
   ≈ **1.5×**.
4. **Speculative decoding** with the MTP head the checkpoint ships (8 draft
   layers) or the n-gram drafter: a K-token verify pass reads the attention /
   shared / dense / unembed weights once for K tokens (the routed experts still
   scale with K); at K = 4 and ~75 % acceptance ≈ **1.3–1.6×**.
5. **Continuous batching** for aggregate throughput: linear in concurrent
   streams until compute-bound (the pipeline supports it as is).
6. **Expert-parallel dispatch**: **1.5–2×** on a resident cabled LAN, **~5×** in
   the RAM-starved paged regime, a new runtime (per-layer dispatcher, expert
   placement, all-reduce), and the paper.

Stacked, 1–4 put one stream on a 12-box Panther Lake pipeline at roughly
**6–9 tok/s**; 6 could push that towards 12–15 on a cabled LAN. Aggregate
throughput on top of that scales with streams.
