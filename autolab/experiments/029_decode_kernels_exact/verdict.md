# 029 verdict: the exact route works and is worth nothing; neither is the sleeping GPU wait. Both removed.

Overrides only (binary 8baebd3c). Head sharing off. 15 streams: 24.77 / 24.72 tok/s (027: 24.89 / 24.69). Gates pass.

## a) Decode kernels on the exact group-32 layers (rank 5, layers 30-31, six-byte sub-group patch)

The patch applied (sha256 8c4bf294...), all six layers of rank 5 passed the load check on the highest expert ids at
the exact bar (cosine 0.999994-0.999995 against the host kernels), no restart, gates exact. So the plugin's MoE
kernels run correctly with sub-group 16 on Xe3, prefill-path kernels included.

| one call, rank 5 idle (µs, median of 33) | 1 row | 2 rows | 3 rows (padded to 8) |
|---|---|---|---|
| layer 30, decode kernels, UNPADDED one-row call | 3211 | 4821 | 7236 |
| layer 31, decode kernels, UNPADDED one-row call | 2963 | 5052 | 7564 |
| layers 32-35, prefill path (one row padded to two) | 3113-3272 | 5187-6384 | 7416-7915 |

**No gain.** 026's estimate (2.1-2.2 ms for one row) came from a two-row call whose second row repeated the first
row's experts (3.76 ms); fitted on distinct rows (4.9 ms at two, 15.3 ms at eight) the decode path is
`1.4 ms + 1.7 ms per row`: 3.1 ms for one row, which is what it measures. The ~1.0-1.4 ms every MoE call costs
before its bytes is common to both paths (small kernels: gather, top-k bookkeeping, scatter; launches and one wait),
not something the decode path avoids. At 15 streams rank 5 ran 40.6 ms a frame between neighbours at 39.2-41.1.
Closed: the decode kernels are for many rows on a GPU where the prefill path is slow, which is not this one.
The patcher stays in `bench/` (it is how one gets these kernels for group 32 on Xe2+ if that ever matters).

## b) GPU queue throttle LOW (rank 6)

| 15 streams | rank 6 (LOW) | ranks 5 / 7 |
|---|---|---|
| busy CPU cores | **0.52** | 0.96 / 0.97 |
| package power | 16.6 W | 18.2 / 17.8 W |
| ms per frame | **42.6** | 40.6 / 40.0 |
| GPU attention / experts per frame | 11.3 / 26.2 ms | 9.8 / 24.8, 9.6 / 24.3 ms |

The completion wait does sleep (half a core, 1.5 W less), and every wake-up costs: +2.3 ms a frame. The watts did
not come back as GPU speed. Negative.

## What the evening's four attempts say together (027, 028, 029a, 029b)

A frame of r rows costs a stage `16 + 19 r` ms and nothing on the device-call side moves those constants: the 16 is
0.78 GB of int8 attention read once (8-9 ms) plus ~1.2 ms of fixed cost in each of six expert calls; the 19 is one
row's 1.57 GB of experts at the bus limit plus 5 ms of CPU work. With 15 rows in eleven frames the ideal round is
11 x T(1.36) = 461 ms = 32.5 tok/s; the fleet does 545-556 ms (two-row frames take 593 ms to go round by service
time alone, one-row frames queue at rank 10 behind them) = 27 tok/s raw, 24.8 as the sum of stream rates.
What is left that is exact and larger than a few per cent: guess rows from a drafter that is right >= 0.8 of the
time (the shipped MTP head: two rows per stream amortise the 16 ms, +14 % at 15 streams, ~2x per stream at 3-8
streams). Not exact: int4 attention (if an int4 path faster than 31 GB/s exists on this GPU), fewer experts per
token. Hardware: the 25 W platform limit of ranks 0-7 (ranks 8-10 are 12 % faster), and which boxes sit at rank 0
and rank 10 (the two positions with extra work are not the fast boxes).

Note: rank 10's clock is one day ahead since about 19:30 CDT (its records carry 2026-09-22); `analyze.py` then
counts its frames twice (its per-frame columns in 028 and 029 are halved; the raw records say 36.6 + 11.7 ms a
frame, as in 027). Inference is unaffected.
