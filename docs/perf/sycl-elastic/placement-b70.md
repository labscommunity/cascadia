# Placement study: streaming vs pinned-host vs driver oversubscription (B70)

Measured 2026-10-06 on the B70 host (2x Intel Arc Pro B70 32 GB, xe driver,
oneAPI 2026.0) against `feat/sycl-llama-elastic-stack` (`82aa11d`) with
`llama.cpp` patches 0001+0002+0003 built at pinned base `1692f9e50`
(`scripts/build-llama-stream.sh`, markers verified). Raw data:
`experiments/2026-10-05-placement-b70/` (not committed).

## Deviation from the recipe

The recipe's card (SYCL1 / PCI 0000:0f:00.0) was occupied by an unrelated
`vllm serve` (~14.7 GiB) for the whole study, so **all measurements run on
SYCL0 = PCI 0000:0b:00.0** instead: `-dev SYCL0` / `--device SYCL0`, VRAM from
`/sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm` ("usage:" line, ~3 Hz
sampling, peak reported). The 0f card read `usage: 14675709952` bytes before
*and* after the study - untouched. Absolute t/s on 0b run ~5-10% below the
campaign's 0f numbers (same silicon, different slot/power envelope); all
comparisons below are within-card so relative conclusions hold.

Other deviations:
- `~/llama-stack` is replaced by
  `/mnt/nvme-wd-sn770-500gb-data/placement-b70/llama-stack` (root fs was
  nearly full); the build script ran unchanged.
- The recipe's `--load-mode read` is not a valid value in this llama.cpp
  (`auto|none|mmap|mlock|mmap+mlock|dio`); the host arms used
  `--load-mode none`, which is what the patch intends (mmap off -> the
  loader reads weights into the host buffer; cascadia's
  `--llama-host-layers` emits the equivalent `--no-mmap`).
- The stack's 0001/0002 differ from the PR-171 head's regenerated pair
  (ours adds the slot-pool teardown in 0001, the expert-arena cleanup in
  0002, and the `host_unified_memory` iGPU-classification fallback). None
  of that affects B70 measurements, so section A was still run on the
  stack build as specified.

## Section A - regression of the fixed patches (27B, `cascadia run`, SYCL0)

Median of 9 samples (3 runs x 3 prompts), decode = child llama-server
`timings.predicted_per_second`, ctx 4096, `-ctk q8_0 -fa on`, temp 0, 64
tokens.

| arm | decode t/s median (range) | expected | stream-weights line |
|---|---|---|---|
| resident (stack, no `--elastic`) | 17.58 (17.49-17.67) | ~19.3 on 0f | - |
| `--elastic-vram 12` | **8.33 (8.25-8.36)** | ~8.35 | `resident layers 62/65 (12265 MiB resident, 821 MiB streamed per token)` |
| `--elastic-vram 0` | **0.55 (0.54-0.55)** | ~0.55 | `resident layers 0/65 (0 MiB resident, 13087 MiB streamed per token)` |
| `--elastic` (auto) | 17.49 (17.33-17.62) | stock speed | `model fits (13087 MiB), streaming disabled` |
| MoE resident | 74.22 (73.85-74.55) | - | - |
| MoE `--elastic-vram 0` | **2.50 (2.48-2.51)** | ~2.61 | `resident layers 0/40` + `moe experts: router-aware expert streaming active (120 routed tensors)` |
| MoE `--elastic-vram 10` | **4.72 (4.59-4.80)** | ~4.91 | `resident layers 20/40 (10137 MiB resident, 10141 MiB streamed per token)` |

All targets hit within noise once the 0f->0b card delta is accounted for.

Process hygiene on the fully-streamed instance:
`fd | grep -c gguf` = **1**; child env has **0** `LD_PRELOAD` (parent had
`libcascadia_elastic.<pid>.so` + a user preload), only
`GGML_STREAM_WEIGHTS=1`/`GGML_STREAM_VRAM_MB=0` set; `kill -9` of cascadia
-> child gone in <1 s (measured at the 3 s check), VRAM back to idle.

Greedy parity (3 fixed prompts, 64 tokens): streamed arms (v12, v0) are
**byte-identical to the unpatched upstream build** at 1692f9e50 on all
three. The stack's own *fused* resident path flips one greedy near-tie at
char 109 of the first prompt - `GGML_SYCL_ENABLE_FUSION=0` resident output
equals upstream stock and streamed output exactly, so the streamed path is
numerically clean; the single diff is confined to the fused fast path
(a tie-break, not corruption; identical behaviour was seen in #171's own
campaign data).

## Section B - pinned-host in-place (patch 0003) vs streaming (27B)

`llama-bench` from the stack build: `-p 0 -n 64 -r 3 -ngl 99 -fa 1 -ctk
q8_0 -ctv q8_0 -dev SYCL0`; tg64 t/s, median of 3 outer runs. Streaming
budgets tuned so the resident-layer count matches the `-ot` regex's
resident set (N = layers kept on device).

| N resident | stream t/s | host-in-place t/s | vram peak (stream / host) |
|---|---|---|---|
| 65 (refs) | stack resident 17.22, auto 17.17 | upstream stock ~17.5 | 17.3 GiB |
| 62 | 8.25 | **6.36** | 16.1 / 16.8 GiB |
| 55 | 2.80 | **3.29** | 15.4 / 15.0 GiB |
| 34 / 33 (see note) | 1.10 | **1.46** | 10.0 / 10.4 GiB |
| 0 | **0.54** | 0.45 | 4.0 / 3.7 GiB |

Engine-path check: `cascadia run ... --llama-host-layers 'blk\.6[2-4]\..*'`
emits `--no-mmap -ot ...=SYCL_Host`, decodes at **6.25 t/s** (bench: 6.36),
output byte-identical to the resident arm.

Note on N=34/33: the streaming budget that came closest gave 34/65
resident; the recipe's host regex `blk\.(3[3-9]|[4-6][0-9])` keeps 33.
The host arm therefore reads one extra layer (~201 MiB) per token; the
rate column below accounts for it.

**The link on this host is PCIe 4.0 x8, not the ~50 GB/s the deficit law
assumed.** Each B70's upstream switch port (`0000:09:00.0`,
`0000:0d:00.0`) trains at 16 GT/s x8 (capable of 32 GT/s x16; the AM4
platform splits its Gen4 lanes x8/x8). Theoretical ceiling is ~15.75
GB/s per card (the GPU endpoints' own `2.5 GT/s x1` is the card-internal
virtual link and is not the bottleneck). Effective host-read rate per arm,
from `bytes_off_device / (1/tps - 1/17.22)`:

| N | off-device GB/token | stream GB/s | host-in-place GB/s |
|---|---|---|---|
| 62 | 0.86 | **13.6** | 8.7 |
| 55 | 2.57 | 8.6 | **10.5** |
| 34 / 33 | 6.98 / 7.19 | 8.2 | **11.5** |
| 0 | 13.72 | **7.7** | 6.3 |

At N=62 streaming runs at ~86% of the x8 Gen4 ceiling, and the deficit
law with a practical 13 GB/s predicts 8.05 t/s vs 8.25 measured. Toward
full streaming both mechanisms fall to ~6-8 GB/s (the law at 13 GB/s
gives 0.90 t/s at N=0 vs 0.54/0.45 measured), so another cost besides
the link dominates there: per-tensor synchronous copies for streaming,
in-place reads for host placement. The ~14 t/s (N=62) and ~3 t/s (N=0)
host predictions assumed a ~50 GB/s x16 Gen5 link that this host does not
have, so they are not reachable here. Winner is non-monotonic: streaming
wins the extremes (N=62: +30%, N=0: +20%), host-in-place wins the middle
(N=55: +18%, N=34/33: +33%).

## Section C - three 27B instances on one card, all generating

Direct `llama-server` (campaign method), `GGML_STREAM_*` via env, 64-token
completions fired concurrently, per-instance `predicted_per_second`.

| arm | per-instance t/s | combined t/s | peak vram_mm usage |
|---|---|---|---|
| `auto` x3 (splits: fits / 63/65 / 0/65) | 3.49, 5.40, 0.50 | **9.4** | 31.89 GiB* |
| `--elastic-vram 12,12,0` (62/65, 62/65, 0/65) | 1.88, 1.87, 0.48 | **4.2** | 31.89 GiB* |
| no `--elastic` (driver oversubscription) | 3.21, 4.22, 4.15 | **11.6** | 31.70 GiB* |
| 2 resident + 1 all-host (`-ot 'blk\..*'=SYCL_Host`) | 1.18, 1.18, 0.40 | **2.8** | 31.84 GiB* |
| 3 x one-third on host | 0.54, 0.54, 0.54 | **1.6** | 31.71 GiB* |

*vram_mm's `size` on this card is 34,242,297,856 bytes = 31.89 GiB. Every
C arm peaked at, not above, the card's capacity: usage never exceeds
`size`. Three 27B instances (~3 x 15 GiB resident in the no-`--elastic`
arm) cannot all be in VRAM at once, so the driver must be keeping part
of them in host memory. That is inferred from the capacity, not read off
the counter. In the `auto` arm, instance 3 also printed `warning: 1913 MiB
free, fully streamed needs ~4267 MiB` and loaded fully streamed. Host RAM never went below ~30 GiB
available; no instance failed to load; no xe resets observed.

## Tate's three questions, answered from this data

1. **Does in-place host memory beat streaming at the same residency?**
   Only in the middle. At N=55/34 host placement wins by ~20-30%; at the
   extremes streaming wins (N=62: 8.25 vs 6.36; N=0: 0.54 vs 0.45). The
   link on this host is PCIe 4.0 x8 (~15.75 GB/s), not the ~50 GB/s
   the predictions assumed. Streaming reaches ~13.6 GB/s at N=62 (near
   the ceiling) and host-in-place peaks at ~11.5 GB/s mid-range; both fall
   to ~6-8 GB/s at N=0. On a Gen5 x16 host the ranking could change.
   Retest there before generalising.
   Recommendation: for a discrete card, keep weight *streaming* as the
   `--elastic` mechanism; `SYCL_Host` placement is a better fit for UMA
   (same DRAM either way) or possibly for mid-range resident splits if a
   cheaper implementation of host-read wins over staging+memcpy.
2. **Does driver oversubscription beat either for 3 instances?** Yes:
   plain no-`--elastic` instances reached 11.6 combined t/s vs 9.4 for
   auto-streaming and 4.2 for explicit budgets. The driver's eviction is
   gentler than per-token re-read. Note the caveat: oversubscription gives
   no residency guarantees - the third instance's pages migrate on
   pressure, so decode rate is whatever the driver decides.
3. **Which should `--elastic` run on a discrete card?** Streaming (the
   current 0001 path), with `auto` as default: it gives a guaranteed,
   bounded device footprint (the property co-tenancy actually wants even
   though raw throughput favours oversubscription), it is fastest exactly
   where parking lives (N=0), and it avoids the mid-range where host
   placement happens to win but a smaller resident budget still serves.
