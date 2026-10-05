# Where the weights live vs decode speed on a UMA iGPU (Arc B390, Linux)

Measured 2026-10-05 on a Core Ultra X7 358H (Panther Lake, Arc B390 iGPU,
61 GB LPDDR5x, Ubuntu 26.04, kernel 7.0.0-31 `xe`, compute-runtime 26.05,
Level Zero 1.14, oneAPI 2026.1 icpx), with `llama-bench -p 0 -n 64 -r 2
-ngl 99` on builds of the pinned base `1692f9e50`: stock, stock + patches
0001/0002 (this PR's streaming), and stock + patch 0003 (the host-buffer
change below). The "device pool" column is the per-process
`drm-resident-gtt` peak from the driver's fdinfo: on this iGPU every
GPU-visible allocation — device USM *and* pinned host USM — is accounted
in that one pool. The box is shared; baselines are medians of three
(single-run outliers of 12, 16 and 20 tok/s were seen on the 7B stock
arm).

## Qwen2.5-7B-Instruct Q4_K_M (4.4 GB file, 3740 MiB of layer weights)

| placement | decode tok/s | device pool (MiB) |
|---|---|---|
| stock, weights on device | 10.4 | 4884 |
| 0001/0002, `GGML_STREAM_VRAM_MB=auto` (fits → streaming off) | 10.5 | 4818 |
| 0001/0002, 17/28 layers resident, 1495 MiB streamed per token | 1.19 | 3548 |
| 0001/0002, fully streamed, 3740 MiB per token | 0.52 | 865 |
| 0003, all layer weights in pinned host memory, GPU computes in place | 6.3 | 5058 |
| 0003, half the layers in pinned host memory | 9.0 | 4679 |
| stock, layer weights on CPU buffers (the CPU computes them) | 4.3 | 1046 |

With the `--elastic` host interposer (#132) preloaded into the child, the
way `cascadia --elastic` launches it: stock 10.4, fully streamed 0.51,
half 1.24 — speed-neutral.

## Qwen2.5-1.5B-Instruct Q4_K_M

| placement | decode tok/s | device pool (MiB) |
|---|---|---|
| stock, device | 38.9 | 1305 |
| 0001/0002, fully streamed, 743 MiB per token | 1.9 | 577 |
| 0001/0002, 18/28 resident | 5.0 | 1069 |
| 0003, pinned host, in place, all layers | 26.1 | 1407 |
| 0003, pinned host, half the layers | 34.6 | 1427 |
| stock, CPU buffers | 18.4 | 527 |

## What it says

- **Streaming on UMA moves the model within one DRAM.** 3.7 GB per token
  at about 1.9 GB/s (page cache → staging buffer → Level Zero memcpy,
  synchronous, per tensor): 20× slower to free 82 % of the pool; the
  half-resident point is 9× slower to free 27 %. The PR's Windows B390
  figure (6.3 → 0.4 tok/s on a 27B) is the same effect.
- **Pinned host memory read in place is 11–13× faster than streaming and
  frees nothing.** The driver counts host USM in the GTT pool, so the
  device-pool number does not move. The 0.57× factor is the cost of
  coherent host memory versus cached device memory on an iGPU — not
  bandwidth, which is the same bus.
- **So there is no cheap pool reduction on this device.** The elastic
  tier the CPU posture relies on — pageable, file-backed memory the
  consumer reads directly — needs system USM (SVM). The B390 exposes
  host, device and shared USM but not system USM (`sycl-ls --verbose`:
  no `usm_system_allocations`; `clinfo`: "Shared System USM: n/a"), so
  the GPU cannot read page-cache pages. On iGPUs `auto` is the serving
  mode and streaming is a parking mode that costs the whole model per
  token.
- **For discrete cards the 0003 change is the experiment.** With
  `-ot '<layers>=SYCL_Host'` the overflow is read over the link in place:
  stable pointers (fused ops and graphs stay on), no per-token
  file → staging → copy, and the budget keeps its meaning. On the B70 the
  deficit law at a practical 50 GB/s predicts about 14 tok/s for 62/65
  resident (measured with streaming: 8.35) and about 3 tok/s with every
  layer on the host (measured: 0.55). To measure:

  ```bash
  scripts/build-llama-stream.sh ~/llama-stream     # now applies 0003 too
  B=~/llama-stream/build/bin/llama-bench; M=Qwen3.8-27B-UD-Q4_K_S.gguf
  $B -m $M -p 0 -n 64 -r 3 -ngl 99                                   # resident
  $B -m $M -p 0 -n 64 -r 3 -ngl 99 --load-mode read -ot 'blk\.6[2-4]\..*=SYCL_Host'  # 3 layers on the host
  $B -m $M -p 0 -n 64 -r 3 -ngl 99 --load-mode read -ot 'blk\..*=SYCL_Host'         # everything on the host
  GGML_STREAM_WEIGHTS=1 GGML_STREAM_VRAM_MB=12288 $B -m $M -p 0 -n 64 -r 3 -ngl 99   # streaming, same split (12 GiB budget = 62/65)
  ```

  and, separately, the driver-managed oversubscription the PR's
  three-model `auto` arm hit (3.3 + 3.2 + 0.49 tok/s against 0.52
  combined for three streamed copies): several resident instances over
  the card's size with streaming off, per-instance tok/s and `vram_mm`.

## Patch 0003

`patches/llama.cpp/0003-sycl-host-buffer-compute.patch`, 14 lines on the
stock tree: `ggml_backend_sycl_device_supports_buft` also accepts the
backend's own host buffer type, so the scheduler keeps the op on the GPU
instead of handing it to the CPU backend; and the `-ot` parsers
(llama-bench, `common/arg.cpp`) register each device's host buffer type,
so `SYCL_Host` is a valid target. Greedy parity against the resident run
was not part of the llama-bench sweep above; the server-level parity
harness in this PR applies unchanged.
