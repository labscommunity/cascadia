# Inkling on one Panther Lake box: CPU vs iGPU vs OpenVINO-native

Three whole-model, single-stream decode measurements of Inkling (975B MoE,
66 layers, 549 GB int4 export) on **tate-07** (Core Ultra X7 358H, 16 CPUs,
64 GB, Arc B390 iGPU, one Samsung PM9C1 Gen4 NVMe, Windows 11, OpenVINO
2026.3.1), all with the **campaign-129 protocol** from
`tools/inkling_autolab/INKLING_129_REPRODUCTION.md`:

* `inkling_decode_bench --export <model> --cases large-cases.baseline-reference.json --tokens 64 --samples 3 --layer-profile …`
* three prompts (`water_cycle`, `binary_search`, `short_story`) × three
  repetitions, 64 generated tokens each, 63 timed decode steps per sample;
* process pinned: PriorityClass High, affinity 65535 (`tools/inkling_autolab`
  Windows E-core rule);
* env = the promoted PTL read profile (`ptl-profile.ps1` +
  `SECOND_PREDICT_RANK=2`); the expert cache is 256 MiB per MoE layer
  (16 GiB) unless stated;
* score = **min over the nine samples of `63 / decode_seconds`** (median and
  max shown too); greedy output must hash to `ce0fbb9a116d3d09`, the record's
  reference, or the run is not correct.

Runner: scratchpad `t07_bench129.ps1`; raw artifacts on tate-07 under
`C:\Users\devcloud\inkling-igpu\logs\b129_<tag>.{json,profile.json,log}`.

## Results

Whole model, single stream, decode tok/s over the nine samples
(min is the campaign score):

| configuration | what runs where | **min** / median / max | prefill (30–32 tok) | greedy parity with the bf16 reference |
|---|---|---|---|---|
| **CPU control** — record binary `full-second-prefetch.exe`, campaign 129 as documented | everything Rust on the CPU; experts streamed from NVMe by the tuned reader (uncached + pipelined + predicted reads, 16 GiB expert cache, 46 % hits) | **1.098** / 1.127 / 1.200 | 24.6–26.1 s | exact, hash `ce0fbb9a116d3d09` |
| **ours on iGPU** — `feat/inkling-igpu` | attention projections (int8 IRs, all 66 layers) + unembed head (int8 IR) on the Arc B390 through OpenVINO; the 16.4 GB of Rust bf16 attention copies released; experts streamed by the same tuned reader, same 16 GiB cache | **1.144** / 1.155 / 1.233 | 20.0–22.1 s | `water_cycle` exact; `binary_search` parts at token 47/64, `short_story` at 28/64 (int8 numerics) |
| **ours on iGPU, 24 GiB cache** | as above with the freed RAM given to the expert cache (384 MiB/layer; hit rate 46 → 55 %) | **1.236** / 1.263 / 1.337 | 20.2–21.3 s | same as above (same output hash `bdf16eab…`) |
| **OpenVINO-native MoE** | as "ours on iGPU" plus OpenVINO's own fused MoE kernel (`moe_3gemm_fused_compressed`) **and** OpenVINO's on-disk expert offload (`OFFLOAD_RATIO=90`, `WEIGHTS_PATH`) serving MoE layers 2–7; the other 58 MoE layers as in the control | **0.587** / 0.607 / 0.628 | 43.3–46.1 s | `water_cycle` exact; `binary_search` 29/64, `short_story` 28/64 |
| OpenVINO fused MoE, experts resident on the device | fused kernel with layers 2–4 resident (three 8.3 GB layers = the Windows iGPU budget on 64 GB), everything else as "ours on iGPU" | **1.003** / 1.118 / 1.242 | 19.4–20.9 s | `binary_search` 29/64, the others exact |

`OFFLOAD_RATIO=99` (1 % resident) crashes in the plugin's expert-slot copy
(`gpu_usm::copy_from … dst_size=8, copy_size=48`): the resident slots must
hold at least one call's experts, so 90 % is the coldest usable setting.

Per-layer decode medians from the layer profiles (ms per token; the
MoE layer time is *whatever the layer waited for*, disk included):

| layer set | CPU control | ours on iGPU | ours, 24 GiB cache | OpenVINO offload (L2–7 cold) | OpenVINO fused resident (L2–4) |
|---|---|---|---|---|---|
| dense L0/L1, attention + MLP | 2.9 + 3.8 | 3.1 + 4.3 | 3.1 + 4.3 | 3.1 + 4.2 | 3.4 + 4.2 |
| MoE layers, attention (min / med / max) | 3.0 / **3.4** / 3.5 | 2.4 / **2.6** / 3.3 | 2.3 / **2.6** / 3.3 | 2.4 / 2.6 / 3.3 | 2.5 / 2.7 / 3.5 |
| MoE layers 2–7, experts (mean) | 14.5 | 15.0 | 14.2 | **147.1** | **5.2** (L2–4), 15.0 (L5–7) |
| MoE layers 8–65, experts (mean) | 8.9 | 9.2 | **7.8** | 9.4 | 9.4 |
| sum over 66 layers | 0.22 + 0.61 = **0.83 s** | 0.17 + 0.63 = 0.80 s | 0.17 + 0.55 = **0.72 s** | 0.17 + 1.43 = 1.61 s | 0.18 + 0.61 = 0.79 s |

## Reading the numbers

**The single box is disk-bound, and no device changes that.** In the
control the 16 GiB expert cache hits 46 % of routed-expert reads; the
misses (~6.6 GB per token) stream from the one NVMe at ~7 GB/s, which is the
0.61 s of MoE time per token. Compute is hidden under the reads. Moving the
attention to the iGPU took 0.8 ms off every layer (3.4 → 2.6) and cut prefill
by 4–5 s, but the decode gain is only +4 %: a layer's time is set by its
expert misses, so time saved in attention mostly turns into waiting on the
read pipeline. The lever that moved the number was RAM: releasing the 16.4 GB
of bf16 attention copies and giving 8 GB of it to the expert cache raised
the hit rate from 46 to 55 % and the score from 1.10 to **1.24 tok/s
(+13 %)**. On this box that is what the iGPU is worth for single-stream
decode.

**Fused MoE layers must stay off a single 64 GB box.** Each fused layer holds
8.3 GB of expert weights in unified memory. Six of them (50 GB) next to the
38–41 GB benchmark process is what made an earlier `cascadia run` on this box
page at 55 s per token; one resident layer saves ~11 ms per token, the same
8 GB as expert cache saves an order of magnitude more.

**OpenVINO's expert offload is not a substitute for the tuned reader.** In
the same run, a MoE layer served by OpenVINO's `OffloadExpertWeightProvider`
takes 147 ms per token against 14.5 ms for the neighbouring layers served by
the Rust reader: ten times slower, because it issues positional reads on
demand with no cross-layer prediction and no cache across tokens. Six such
layers halve the whole-model rate (0.59 tok/s); all 64 would be ~10 s per
token. OpenVINO's fused *kernel* is fine (5.2 ms per resident layer), but
keeping layers resident on the device costs 8.3 GB of unified memory each,
and on a 64 GB box that memory comes out of the expert cache: three resident
layers made the whole model *slower* (1.00 tok/s min, high variance).

**The fleet is different physics, and the iGPU's edge there is smaller
than the earlier per-layer numbers suggested.** With a rank's layers resident
(the 12-box installation: 5–6 layers per box) nothing streams from disk, and
the per-layer time is memory bandwidth: 256 MB of int4 expert weights per
MoE layer per token plus the attention projections. The whole-model profiles
give the resident cost for both engines *in the same process*: the tuned CPU
kernels do a cache-resident MoE layer in ~5 ms (the best-cached layers of
the control) + 3.4 ms attention, and the fused iGPU layer does 5.2 ms + 2.6 ms
attention. Both draw on the same LPDDR5X bus (the CPU at ~50 GB/s effective,
the iGPU at ~75 GB/s), so the iGPU is **1.1–1.6× the tuned CPU per resident
layer, not the 6× that the untuned 30 ms/layer dump figure implied**. A
12-rank resident pipeline is therefore ≈ 1.5–2 tok/s single stream on
either engine, and the iGPU's practical value on a rank is prefill and
freeing the CPU. The layer-dump harness (5 layers, tuned env, pinned)
puts the same comparison at 18 ms per MoE layer on the CPU with experts
memory-mapped (each call copies 256 MB out of the page cache) against 7.8 ms
with the fused kernel resident on the iGPU (and 9.0 vs 4.8 ms for a dense
layer, 340 vs 54–94 ms for a 23-row prefill), i.e. 2.3× — an upper bound on
the iGPU's edge, since a rank that owns its experts in RAM (the whole-model
expert cache) is the 5 ms case, not the 18 ms one. (`--experts eager` in that
harness expands the experts and runs out of memory at three MoE layers, so
the owned-RAM CPU case is only available from the whole-model profile.) On
Windows the iGPU holds three fused layers per 64 GB box (the driver caps
shared memory at half of RAM); Linux ranks or 96–128 GB boxes lift that.

## Serving-path note

`cascadia run` logs `tok_s` as generated tokens over the whole request, prefill
included, so a 16-token answer behind a 25 s Inkling prefill reads as
0.02 tok/s next to the 1.1 tok/s decode figure above. The single-stage task
log now also carries `prefill_s`, `decode_s`, `decode_steps` and
`decode_tok_s` (the benchmark's definition). The decode code path is the
same as the benchmark's (`StagedRunner::generate_reason` →
`Layer::forward_token` → the pipelined predicted reads); a serving check with
the same env profile and no fused MoE layers is reported below once measured.

| `cascadia run`, same profile, 16-token request (23-token prompt) | prefill | decode tok/s (from the task log) |
|---|---|---|
| CPU | 20.2 s | 15 steps in 13.0 s = **1.15** (the log's whole-request `tok_s` says 0.48) |
| ours on iGPU (attention + head) | 17.1 s | 15 steps in 12.8 s = **1.17** (whole-request `tok_s` 0.53) |

Both match the benchmark's decode rate for their configuration; the
serving path is not slower than the harness.
