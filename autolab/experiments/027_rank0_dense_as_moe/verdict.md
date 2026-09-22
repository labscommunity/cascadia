# 027 verdict: rank 0 is no longer the slowest stage; the output head is, and the round barely moved

Binary 491351d9, overrides = 026 + `CASCADIA_FUSE_DENSE_MOE="0,1"`, `CASCADIA_INKLING_OV_DENSE_MOE=1` on rank 0.

| load check (rank 0) | cosine | rel. error | fused-experts form, 1 / 2 rows | three MatMuls, 1 / 2 rows |
|---|---|---|---|---|
| layer 0 | 0.999999 | 5.7e-4 | 4.51 / 4.47 ms | 8.15 / 8.41 ms |
| layer 1 | 0.999999 | 5.9e-4 | 4.45 / 4.48 ms | 8.11 / 8.41 ms |

(5.8e-4 is two f16 device paths against each other; the generator's f32 comparison on a CPU was 7e-7.)

| 15 streams x 128 tokens | 026 | 027 |
|---|---|---|
| rank 0 ms per frame / busy | 51.9 / 95.8 % | **43.7 / 82.4 %** |
| rank 10 ms per frame + head / busy | 35.1 + 11.6 / 85 % | 35.9 + 11.6 / **88.5 % (96 % with the head's send)** |
| rank 0's round trip | 549 ms | 544 ms |
| steady tok/s (two phases) | 24.58 / 24.69 | **24.89 / 24.69** |

Both gates pass; one fresh stream 4.9-5.7 tok/s (64 tokens).

The fix does what it says (-8.2 ms on rank 0, less than the predicted 12: the fused op takes 4.5 ms for these eight
slices against 3.0 ms for a routed layer's eight experts) and the fleet gains 0.5 %: the round was already paced by
rank 10 as much as by rank 0 (47.5 ms with the head; a ring of eleven frames turns at eleven times its slowest
stage: 11 x 47.5 + hops = 535-545 ms). Kept: it is exact, and it is the precondition for the head work (028) to show.
