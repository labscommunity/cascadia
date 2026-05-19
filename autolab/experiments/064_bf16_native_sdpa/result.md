# 064 — bf16 native SDPA — NEGATIVE (upconvert is already cheap)

**Verdict:** **negative** — do not ship a hand-rolled bf16 SDPA intrinsic.
**Date:** 2026-05-18
**Magnitude class:** **XS** (≤4% of attention compute)

## Hypothesis

Iter 032 (`perf/a8-kv-bf16-029` @ ebd8ac4) stores KV as `Vec<u16>` (bf16
bit pattern) and upconverts to f32 inline at each dot-product element:

```rust
for i in 0..QK_HEAD_DIM {
    let kf = f32::from_bits((k_row[i] as u32) << 16);
    s += q_h[i] * kf;
}
```

The conjecture for iter 064 was that the inline upconvert costs
meaningful cycles in the hot loop, and a native bf16 SDPA — either using
`VDPBF16PS` (AVX-512 BF16) or a hand-restructured split-pass loop — could
recover them.

## Method

`crates/tahoma-int4-gemm/src/bin/bench_bf16_upconvert.rs` —
single-head scalar microbench at K2.6 attention shapes
(QK_HEAD_DIM = 192, V_HEAD_DIM = 128) decomposed into five variants:

| variant            | what it measures                                            |
|--------------------|-------------------------------------------------------------|
| `f32 SDPA`         | pre-iter-032 reference (no cvt)                             |
| `bf16 inline`      | iter 032 baseline (cvt fused inside dot/accum)              |
| `bf16 split`       | cvt one row to scratch f32 first, then f32 dot              |
| `bf16 upcvt only`  | just the cvt pass (k+v) — isolated dequant cost             |
| `f32 dot only`     | same f32 math, with no cvt — subtract to isolate cvt cost   |

Run on the actual benchmark host: **miner (Xeon Gold 6252, Cascade
Lake)**, single-core (`taskset -c 0`), `RUSTFLAGS=-C target-cpu=native`.
Also captured Mac M-series ARM64 for contrast.

## Results — miner (Xeon Gold 6252)

| past_seq_len | f32 SDPA | bf16 inline | bf16 split | bf16 upcvt only | f32 dot only | upconvert overhead (inline - dot) | % of bf16 inline |
|-------------:|---------:|------------:|-----------:|----------------:|-------------:|----------------------------------:|-----------------:|
|           16 |  5.76 us |     5.83 us |    6.53 us |         0.63 us |      5.83 us |                          0.003 us |             0.0% |
|           64 | 22.48 us |    22.98 us |   24.95 us |         2.01 us |     22.49 us |                          0.491 us |             2.1% |
|          256 | 86.50 us |    90.08 us |   97.18 us |         7.76 us |     86.58 us |                          3.494 us |             3.9% |
|         1024 |  357 us  |     357 us  |     403 us |          31 us  |       367 us |                            ≈0 us  |             0.0% |
|         4096 | 1479 us  |    1422 us  |    1501 us |         153 us  |      1477 us |                            ≈0 us  |             0.0% |

**Headline:** the inline upconvert costs **≤ 4% of total SDPA time** at
any working-set size, and is **statistically zero** at the long-context
sizes (1024, 4096) that dominate decode wall-clock. The `bf16 inline`
column matches or beats the `f32 dot only` column at 1024 and 4096
because the bf16 cache is half the bytes and gets a free win from L2/L3
residency — exactly iter 032's bandwidth thesis.

**Split-pass restructure is empirically slower at every size**, by
6–13%. Materialising the cvt to a scratch buffer wastes the
register-resident value that the fused loop keeps live.

## Hardware availability of the AVX-512 BF16 path

The proposed iter 064 acceleration was `VDPBF16PS` (AVX-512 BF16):
multiply two bf16 inputs and FMA-accumulate into f32 in one µop. It
requires the `avx512_bf16` CPUID feature, available on Cooper Lake and
later.

- **miner = Cascade Lake (family 6, model 85, stepping 7).** CPUID:
  `avx512f`, `avx512bw`, `avx512cd`, `avx512dq`, `avx512vl`,
  `avx512_vnni`. **No `avx512_bf16`.**
- **AI PC fleet = Lunar Lake (model 189).** **No AVX-512 at all** —
  Intel dropped the entire AVX-512 ISA from consumer chips after Tiger
  Lake.

So even if the upconvert were 30% of SDPA, `VDPBF16PS` would have no
target hardware to run on within tahoma's hardware scope (Lunar Lake,
Arrow Lake, Panther Lake AI PCs; Xeon CPU-only servers). The only iter
064 angle that could have shipped was the scalar restructure — which
this bench shows is slower, not faster.

## Decision

**Drop the iter 064 line of investigation.** Iter 032 already extracts
the bf16 win — the inline upconvert IS the right shape on both the
miner and on consumer AVX2-only Intel parts. There is no win to chase.

The microbench (`bench_bf16_upconvert`) is the artifact — future
iterations comparing KV formats should re-use it instead of
re-litigating the upconvert overhead question from scratch.

## Reproduce

```sh
cargo build --release --bin bench_bf16_upconvert -p tahoma-int4-gemm
taskset -c 0 ./target/release/bench_bf16_upconvert
```

Raw output: `bench_bf16_upconvert_miner_xeon_gold_6252.txt`,
`bench_bf16_upconvert_mac_arm64.txt`.
