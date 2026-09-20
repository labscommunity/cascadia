# Candidates, ranked by expected effect on the two targets

Tier S = structural, could move a target by 2x or more. Tier A = 1.2-2x. Tier B = < 1.2x or enabling.

## Aggregate throughput (target 60 tok/s)

| id | tier | idea | expected | cost | status |
|---|---|---|---|---|---|
| M1 | S | balanced groups: new streams join the emptiest in-flight group | util 30 % -> 60-80 %: ~2x | done (0f339e55) | exp 001 |
| M2 | S | expert sharing across rows: each distinct expert read ONCE per frame, all its rows through a multi-row int4 kernel; shared experts as one GEMM over all rows | t_row 33 -> ~10 ms at 32 rows/frame | kernel + MoE block path | todo |
| M3 | S | fused iGPU MoE for large frames and prefill (compute-bound regime), f16 made safe | removes the CPU GEMM compute wall at >= 16 rows | IR rescale + f16 fix | research running |
| M4 | A | more frames in flight than ranks (`CASCADIA_STREAMS_INFLIGHT` 16-33): a closed loop of N frames over M servers with variable service keeps about N/(N+M-1) busy | util +15-25 % | env only | exp 002 |
| M5 | A | token replies go straight from the last rank to rank 0 (today relayed through 9 ranks, each only when it is not computing: up to 9 x half a frame time per reply) | cycle -10-25 % at high util | new link or relay thread | todo |
| M6 | A | hundreds of slots: KV in f16 and/or smaller MAX_SEQ so 350 streams fit | enables M2's batch sizes | memory work | todo |
| M7 | A | rank 0 is the slowest stage (dense layers run row by row as GEMV, plus embed/emit/admission): batch the dense MLP | rank 0 frame -30 % | small | todo |
| M8 | B | workers merge frames queued at their input into one micro-batch | fewer fixed costs per row | engine | todo |
| M9 | B | duplicate page-cache + anon copy of experts: mlock/own one copy, free ~10 GiB per box for slots | enables M6 | loader | todo |

## Single stream (target 10 tok/s)

| id | tier | idea | expected | cost | status |
|---|---|---|---|---|---|
| S1 | S | pipelined speculation: draft tokens enter the pipeline behind the last real one, so up to 11 positions of one stream are in flight on 11 memory buses; reject = rewind KV (the conv rings already have rewind slack) | 1.7-2.5x with n-gram drafts (a = 0.3-0.5), 6x with a = 0.9 | engine protocol | todo |
| S2 | S | a draft that is right 9 times in 10: a small model on rank 0's iGPU/NPU sharing the tokenizer (needs a file channel larger than the six names, or an embedded blob) | makes S1 reach 10 tok/s | large | open question |
| S3 | A | 25 W platform limit on ranks 0-7 (53 vs 38 ms per stage) | single 1.7 -> ~2.2 | BIOS/RAPL, user decision | reported |
| S4 | A | LAN: 2-3.5 ms ping on cdc_ncm; per token 10 hops + 10 reply hops | -30-50 ms of 560 | NIC tuning via run.sh | todo, risk |
| S5 | B | int4 attention projections (7.9 -> 4 GB/token) | -60 ms | needs IR regeneration on the boxes | todo |
| S6 | B | fused iGPU MoE at f16 on the 25 W boxes (GPU moves more bytes per watt than 16 throttled cores?) | unknown | after M3 | todo |
| S7 | B | cold start: warm the resident expert copy at startup | first request 34 s -> 10 s | small | todo |

## Time to first token

| id | tier | idea | expected | status |
|---|---|---|---|---|
| T1 | S | windowed prefill: long prompts travel as 128-row windows back to back (also fixes the >256-token crash) | 530 tokens: 242 s (crash) -> ~80 s | exp 001 |
| T2 | S | prefill experts on the iGPU (compute-bound: 530 rows x 8 experts) | 22 s/rank -> 2-3 s | with M3 |
| T3 | A | batch several waiting prompts into one prefill frame (rows of several slots share expert reads) | burst TTFT 88 s -> ~30 s | todo |
