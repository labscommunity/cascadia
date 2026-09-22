# Candidates, ranked by expected effect on the two targets

Tier S = structural, could move a target by 2x or more. Tier A = 1.2-2x. Tier B = < 1.2x or enabling.

## Aggregate throughput (target 60 tok/s)

| id | tier | idea | expected | cost | status |
|---|---|---|---|---|---|
| M1 | S | balanced groups: new streams join the emptiest in-flight group | util 30 % -> 60-80 %: ~2x | done (0f339e55) | **exp 001: steady 9.7 -> 14.7 tok/s @48** |
| M2 | S | expert sharing across rows: each distinct expert read ONCE per frame, all its rows through a multi-row int4 kernel; shared experts as one GEMM over all rows | t_row 33 -> ~10 ms at 32 rows/frame | kernel + MoE block path | kernel being built |
| M3 | S | fused iGPU MoE for large frames and prefill (compute-bound regime), f16 made safe | removes the CPU GEMM compute wall at >= 16 rows | IR rescale + f16 fix | research running |
| M4 | A | more frames in flight than ranks (`CASCADIA_STREAMS_INFLIGHT` 16-33): a closed loop of N frames over M servers with variable service keeps about N/(N+M-1) busy | util +15-25 % | env only | exp 002 |
| M5 | A | token replies go straight from the last rank to rank 0 (today relayed through 9 ranks, each only when it is not computing: up to 9 x half a frame time per reply) | cycle -10-25 % at high util | done (3c2ef930) | exp 002a |
| M6 | A | hundreds of slots: KV in f16 and/or smaller MAX_SEQ so 350 streams fit | enables M2's batch sizes | memory work | todo |
| M7 | A | rank 0 is the slowest stage (dense layers run row by row as GEMV, plus embed/emit/admission): batch the dense MLP | rank 0 frame -30 % | small | todo |
| M8 | B | workers merge frames queued at their input into one micro-batch | fewer fixed costs per row | engine | todo |
| M9 | B | duplicate page-cache + anon copy of experts: mlock/own one copy, free ~10 GiB per box for slots | enables M6 | loader | todo |

## Single stream (target 10 tok/s)

| id | tier | idea | expected | cost | status |
|---|---|---|---|---|---|
| S1 | S | pipelined speculation: draft tokens enter the pipeline behind the last real one, so up to 11 positions of one stream are in flight on 11 memory buses; reject = StreamRewind | 1.4-1.9x with n-gram drafts (a = 0.3-0.5), 5.5x with a = 0.9 | done (72861323) | exp 002b |
| S2 | S | a draft that is right 9 times in 10: a small model on rank 0's iGPU/NPU sharing the tokenizer (needs a file channel larger than the six names, or an embedded blob) | makes S1 reach 10 tok/s | large | open question |
| S3 | A | 25 W platform limit on ranks 0-7 (53 vs 38 ms per stage) | single 1.7 -> ~2.2 | BIOS/RAPL, user decision | reported |
| S4 | A | LAN: 2-3.5 ms ping on cdc_ncm; per token 10 hops + 10 reply hops | -30-50 ms of 560 | NIC tuning via run.sh | todo, risk |
| S5 | B | int4 attention projections (7.9 -> 4 GB/token) | -60 ms | needs IR regeneration on the boxes | todo |
| S6 | B | fused iGPU MoE at f16 on the 25 W boxes (GPU moves more bytes per watt than 16 throttled cores?) | unknown | after M3 | todo |
| S7 | B | cold start: warm the resident expert copy at startup | first request 34 s -> 10 s | small | todo |

## Time to first token

| id | tier | idea | expected | status |
|---|---|---|---|---|
| T1 | S | windowed prefill: long prompts travel as 128-row windows back to back (also fixes the >256-token crash) | 530 tokens: 242 s (crash) -> ~80 s | **exp 001: 508 tokens, TTFT 65 s, no outage** |
| T2 | S | prefill experts on the iGPU (compute-bound: 530 rows x 8 experts) | 22 s/rank -> 2-3 s | with M3 |
| T3 | A | batch several waiting prompts into one prefill frame (rows of several slots share expert reads) | burst TTFT 88 s -> ~30 s | done (d7927f9e), next binary |

## Added 2026-09-20 (from the per-rank data and the fused-MoE study)

| id | tier | idea | expected | status |
|---|---|---|---|---|
| X1 | A | under the 25 W cap, fewer rayon threads / no low-power E cores: cores stalled on DRAM burn the budget the memory controller needs | unknown, per-rank A/B | exp 003 |
| X2 | B | PM QoS (hold /dev/cpu_dma_latency at 0): frame receive is 3.5 ms on the capped boxes, 0.8 ms on the others | -25 ms per single-stream token | exp 003 |
| X3 | A | fused iGPU MoE at f16 with the power-of-two weight rescale + non-finite fallback, compiled at load | f16 about 2x f32 on the device; first question is whether it is finite on layers 36-38 | exp 003 (rank 6 only) |
| X4 | A | `up`-scale attenuation (2^-4) in the shim's constant copy for the sporadic overflow inside an expert (layer 8: -94909) | removes the remaining fallbacks | after X3's fallback count |
| X5 | B | cross-request n-gram table on rank 0 (stock reasoning phrases repeat across requests) | acceptance +0.1-0.2 | todo |
| X6 | A | rebalance groups as streams finish; more frames in flight than ranks | util +10-20 % | exp 004 (env) |

## Added 2026-09-20 afternoon (after 012-015): single stream, what is left and what each is worth

Measured: prose 3.1-3.7 tok/s (a = 0.42, L = 410 ms), structured tasks 6-11, memorised 10-12.

1. **Drafter distilled on the fleet's own outputs.** Qwen3-0.6B is right 0.41 on explanations and 0.78 on arithmetic;
   the gap is style, which fine-tuning on a few million of this model's tokens closes in part (literature: +10 to
   +45 % relative). The fleet writes ~200k tokens an hour at 176 streams; the miner's 4060 Ti can tune 0.6B in an
   hour or two. Expected: a 0.42 -> 0.5-0.55 on prose = 3.5 -> 4.3 tok/s. Exact.
2. **Expert parallelism for the lone row, as a mode** (`PHYSICS.md`). Every box holds 1/11 of every layer's experts
   (CPU-resident, 47.8 GB) and serves any stage; a stage fans a layer's six routed experts out and computes the two
   shared ones itself. L 410 -> ~290 ms. Needs: expert servers that accept several drivers, the staged loader with
   remote experts (the engine has both halves, the CLI forbids the combination), a re-shard of 43 GB per box over
   the LAN driven from the overrides, junk guesses kept off the expert servers. Costs the multi-stream mode while on.
3. **int4 attention projections generated on the boxes** (-5 ms per stage, L -55 ms). Changes numerics: needs a
   wider quality gate than twelve questions.
4. **Hot-expert replicas** (keeps both modes): the 4-7 GB each box has left hold the most-used experts of OTHER
   ranks' layers; a lone row offloads those while its own iGPU reads the rest. Worth ~-50 ms if usage is as skewed
   as in other MoEs (unmeasured here: count expert ids per layer first).
5. **Not exact, so only ever opt-in:** keep a guess the model itself finds likely (its probability within a factor of
   the top token's). With a model drafter that is a ~0.7 on prose: ~6 tok/s today, ~10 with 2 and 3.

## Added 2026-09-20 evening: the 15-stream roster (goal: interactive speed for up to 15 streams, ideally 60 tok/s)

Where a 15-stream round goes (026, per frame of 1.36 rows): rank 0 **51.9 ms, 96 % busy**; ranks 1-7 39-42 ms and
23-28 % idle; ranks 8-9 35.5 ms and 35 % idle; rank 10 35.1 + 11.6 ms of output head, 85 % busy. 24.6 tok/s steady.
Inside a middle rank's 41 ms: fused experts 24.7 (six calls; 2.98 ms for one row's eight experts, 4.83 ms for two
rows' fourteen), attention projections on the GPU 10.3 (int8, 0.78 GB per frame whatever the row count), CPU 3.5.

| id | tier | idea | expected at 15 streams | exact? | status |
|---|---|---|---|---|---|
| F1 | A | rank 0's dense layers through the fused-experts op (eight slices, all selected) | rank 0 51.9 -> ~39 ms; pace set by rank 10 (46.7): **+11 %** | yes (7e-7) | **done (027): rank 0 51.9 -> 43.7 ms, fleet +0.5 %** (the last rank paces the ring) |
| F2 | A | rank 10's output head as an int4 proxy + top-k on the device, the k candidates rescored against the bf16 rows on the host (greedy rows only; sampled rows keep the full head) | 11.6 -> ~6 ms; pace 46.7 -> ~42: **+11 %** | as exact as today's int8 head or better | **dropped**: this GPU reads int4 through the generic FC path 3x slower per byte than int8 (8.1 ms for 255 MB), so an int4 head would be slower; sharing head calls instead lost 2.5 % (028) |
| F3 | A | the plugin's MoE DECODE kernels for one-row frames (7 of 11 frames at 15 streams): they read at the bus limit (133-136 GB/s against 85-92 on the prefill path) | -0.9 ms per layer per one-row frame = **-3.4 ms per average frame (+8 %)**, single stream L -59 ms | group 64: NO (12 % of a block's output); group 32 needs a rebuilt plugin (sub-group 16) | **closed (026, 029)**: works with group 64 (not exact) and on exact group 32 with a six-byte plugin patch, but a one-row call costs 3.0-3.2 ms on either path (`1.4 + 1.7 r` ms): no gain |
| F4 | B | int4 attention projections (0.78 -> 0.4 GB per frame) | -4 ms per frame (+10 %) | no (wider quality gate first) | todo |
| F5 | B | one frame's CPU work (attention core, routers) while the GPU runs the previous call | <= -3 ms | yes | todo |
| F6 | S | **speculation for every stream, not only a lone one**, with a drafter that is right >= 0.8 of the time: the model's own shipped MTP head (eight dense draft blocks, 10.5 GB bf16, `mtp.safetensors`, dropped by the exporter; found on the miner). A guess row costs a full set of expert reads (19 ms of a stage), so it pays only at high acceptance: **with the measured a1 = 0.73: about +7 % at 15 streams, ~2x per stream at 3-8 streams, 3.4-4.1 -> 5-6 tok/s alone** | yes (verified like today's guesses) | **offline study done (033): a1 = 0.726 (0.63-0.69 prose, 0.87 arithmetic)**, deeper modules need their own context kept current; passes the plan's bar; fleet wiring is the next big build |

The byte ceiling stands: 15 rows x 6 layers x 8 experts x 31.85 MB + 11 frames x 0.78 GB of attention = 31.5 GB per
round per stage = 232 ms at the bus limit = 65 tok/s with nothing else in a frame; rank 10 adds 11 head reads
(13.6 GB) per round. F1-F5 together land at 35-40 tok/s; 60 at 15 streams is not reachable with exact int4 experts
on eleven buses of 136 GB/s.

### The teammate's hidden-state drafter plan (`spec.md`, E1-E5), slotted

| their id | what | here | order |
|---|---|---|---|
| E4 | the shipped MTP head as-is, scored on this model's own text | = F6's first question. Structure read from the file today: 8 modules, each `embed_norm`, `hidden_norm`, `input_proj [6144, 12288]` and one DENSE Inkling block (attention + MLP 24,576 wide): DeepSeek-V3 style chained depths, 1.3 GB bf16 each (~0.39 GB at int4 MLP + int8 attention = ~3.5 ms per draft on one bus, plus the unembed) | **done (033)**: a1 0.726, a2-a8 0.69-0.85 given the chain so far with context upkeep, 15.4 ms per draft (7.9 with a 65k-row unembed prefix) |
| E1 | residual dump over the 013 corpus (ran on the Mac Pro's CPU path; **from now on: the fleet state capture**, queue) | reduced to what E4 and E2 need: final state at every position + the ten rank-boundary residuals at generated positions (not all 66 layers) | **done**: 36 prompts on the Mac Pro, 77 min; tool `examples/inkling_spec_dump.rs` |
| E2 | logit lens per rank boundary (the "guess later from a deeper rank" idea) | measured on the same dump; their own prediction is that it fails its bar | **done, closed**: raw lens 0.000 through rank 4 (0.49 at rank 9), tuned lens 0.34-0.39, all below their bars |
| E3 | a new EAGLE-style head trained on the final state | only if E4 fails its bar (a1 >= 0.7 at <= 25 ms per draft): the shipped head is already trained on the real distribution | not needed for now: the shipped head passes its bar; revisit only if fleet-captured states show it lower |
| E5 | fleet wiring: the reply carries the final state (12 kB f16), a head drafts on rank 0 | after E4/E3 pass. For F6 it also needs guess rows for MANY streams in the scheduler (today speculation is a lone-stream mode) and F1 first (the head's cost lands on rank 0, the stage everyone waits for) | later |
| appendix | expert-usage skew | rides on E1 if the dump records routed ids | optional |

### Status after 033 and the role swap (2026-09-20 night): what is left, in order

Measurements run on the fleet from here on (the owner's instruction): hidden states through the fleet state capture,
expert usage through counters on the ranks, text through the API. The Mac Pro stays a CPU reference for parity work
and any-tensor inspection, nothing else (`OPERATING.md`, section 6).

1. **Reliability first.** The entry box's power (brick / outlet / unit): owner's hands. Requests arriving while the
   chain assembles should be refused, not wedge it (small binary change, queue).
2. **MTP drafts on the fleet** (F6 / the teammate's E5): first the fleet state capture (it is E5a's frame extension
   and it re-scores the head on the fleet's own states), then export the eight dense MTP blocks (int4 MLP, int8 attention),
   run them on the box that plays rank 0 with their own KV and conv state, carry the final state on the reply link
   (12 kB a row), guess rows for MANY streams in the scheduler (today speculation is a lone-stream mode), a 65k-row
   draft unembed. Start with module 0 only (a1 0.73, 8 ms a draft), measure, then depth.
3. **Small and exact:** carry the memorised-phrase table over to the new rank 0 (true/false 9.4 -> 4.3 since the swap);
   one frame's CPU work overlapped with the previous device call (3.5 ms of 41).
4. **Owner's decision (not exact):** int4 attention, if an int4 path faster than this GPU's 26-31 GB/s exists; fewer
   experts per token. 60 tok/s at 15 streams needs one of these AND speculation; exact int4 experts top out near 32.
