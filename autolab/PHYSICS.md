# What the hardware allows

Model (from the export manifest): 66 layers, hidden 6144, layers 0-1 dense
(intermediate 24576), 64 MoE layers with 256 routed experts (top-6) + 2 shared, expert
intermediate 3072, int4 experts 31.85 MB each, int8 attention projections, vocab 201k.

Bytes that must cross the memory bus for ONE token of ONE stream:

| part | bytes/token (file sizes of the export, checked 2026-09-20) |
|---|---|
| experts: 64 layers x 8 experts x 32.74 MB as the iGPU reads them (u4 + zero points + f16 scales; 31.85 MB in the CPU layout) | 16.76 GB |
| attention projections, int8: 66 layers x 130.1 MB (`attn_ov` = 8191 MiB) | 8.59 GB |
| dense layers 0-1, int4: 2 x 262 MB | 0.52 GB |
| output head, int8 (`head_ov` = 1,235,694,528 B) | 1.24 GB |
| **total** | **~27.1 GB** |

> **Correction (2026-09-20, the user asked whether 60-65 GB/s was not low for this chip: it was).** This paragraph
> used to say "one box moves about 60-65 GB/s (LPDDR5x-8533)" and derived a 2.4 tok/s sequential ceiling from it.
> That number was never a measurement of the memory: it was what the CPU expert kernel achieved
> (31.85 MB / 0.46 ms = 69 GB/s on the 60 W boxes), mislabelled as the bus. What is known:
>
> - **Spec:** Core Ultra X7 358H, LPDDR5x up to 9600 MT/s, 2 channels, **128-bit** (Intel ARK). The identical box
>   tate-07 reports 8 x 8 GB Samsung parts at 8533 MT/s: 8533 x 128 / 8 = **136.5 GB/s peak**. The venue boxes are
>   the same SKU with the same 64 GB; their DIMM table has not been read (no shell there), so "same" is an assumption.
> - **Seen on the venue boxes** (bytes a kernel must read / its time, 015c): output head **108 GB/s** (1.236 GB in
>   11.5 ms, rank 10); attention projections 67-69 GB/s on the 25 W ranks, 81-85 on the 42-48 W ranks; fused experts
>   72-75 and 76-80; rank 0 next to the drafter model 63-64; rank 0's dense layers **26 GB/s**.
> - **Seen on tate-07:** a dense 27B int4 model decodes at 6.3-6.7 tok/s on the same iGPU: 13.9 GB x 6.5 = ~90 GB/s.
>
> So the memory delivers at least 108 GB/s to the iGPU, and the per-layer paths of this engine run at 60-80 % of
> that. They are dominated by memory traffic, but NOT pinned at a hardware limit: about a quarter of a token's trip
> is recoverable without touching the topology (see "Where one token's time goes").

Historic note, kept for the record: at "62 GB/s" the sequential ceiling was 26 GB / 62 = 0.42 s = 2.4 tok/s, and
0.56 s was measured on the CPU path. With the corrected figures the sequential floor is 27.1 GB / 108 GB/s = 0.25 s
(4 tok/s without any speculation), against 0.42 s of stage work measured today.

Consequences:

- **> 10 tok/s single stream needs several memory buses working on the same stream at once.**
  With layers sharded by box, the only way is to have several positions of the stream in flight
  at different ranks: pipelined speculative decoding. Throughput is about
  `1 / (T * (1 + (1-a) * (D-1)))` for stage time T, depth D = 11, draft acceptance a:
  a = 0.5 -> 3.7 tok/s, 0.8 -> 7, 0.9 -> 10.6 (T = 45 ms). 10 tok/s needs a draft that is right
  nine times in ten; n-gram drafts are right two to four times in ten on prose.
- **Aggregate throughput is one row per `t_row` of the slowest stage.** Today t_row = 37-40 ms
  on the 25 W boxes, 27 ms on the three 60 W boxes, flat in batch size because every row reads
  its own experts. At 100 % utilization that is 25-27 tok/s; the fleet delivers 7-10 because
  the ranks idle 65-70 % of the time.
- **> 60 tok/s needs t_row < 16 ms, which only expert sharing between rows gives.** R rows in
  a frame touch `256 * (1 - (250/256)^R) + 2` distinct experts: R = 16 -> 83, 32 -> 138,
  64 -> 202 (of 258). Reading each distinct expert once costs, per frame and rank,
  `distinct * 31.85 MB * 6 / 62 GB/s`: R = 16 -> 0.26 s (62 rows/s), 32 -> 0.43 s (75 rows/s),
  64 -> 0.62 s (103 rows/s), before attention (2-5 ms/row) and compute. At those batch sizes
  the int4 GEMM is compute-bound on the CPU (29 GMAC per 64-row layer), which is where the iGPU
  earns its place. 60 tok/s therefore needs about 11 x 32 = 350 concurrent streams, a batched
  expert kernel that reads each expert once, enough KV memory for the slots, and a full pipeline.

## Single stream, second pass (measured 2026-09-20, experiments 012-015)

`time per token = a*T + (1-a)*L`, all three measured:

- **L, one token's trip through all eleven ranks: ~410 ms of round trip + rank 0's own 53 ms.** ~424 ms is device
  work dominated by memory traffic: per rank ~11 ms of int8 attention projections (0.78 GB) and ~21 ms of expert
  reads (6 layers x 8 experts x 32.74 MB = 1.57 GB), plus rank 0's dense layers (~20 ms) and the head (11.5 ms). Hops
  were another ~35 ms until the NIC driver's aggregation timer was switched off (014: 3.3 ms -> 0.25 ms per round
  trip). (An earlier version of this line said "every millisecond at the speed of a memory bus, no kernel makes L
  shorter". Wrong: the head on the same iGPU reads at 108 GB/s, these paths at 64-85. See the next section.)
- **T, the slowest stage: ~50 ms** (rank 0, with the drafter model's threads beside it).
- **a, the share of right guesses: a property of the text.** For the same 0.6B drafter: 0.39 (story), 0.41
  (explanation), 0.59 (code), 0.71 (rewriting), 0.78 (arithmetic); tables that have seen the text: 0.9.

| what | a | L ms | tok/s |
|---|---|---|---|
| 011 (n-gram tables) | 0.30 | 478 | 2.9 measured |
| 015c prose / structured / memorised | 0.42 / 0.75 / 0.9 | 410 | 3.1-3.7 / 6-8 / 10-12 measured |
| + the weight-reading paths at the rate the head already reaches on this iGPU (fewer, larger GPU calls; rank 0's dense kernel; see "Where one token's time goes") | 0.42 | ~340 | ~4.3 (estimate) |
| + drafter distilled on this model's outputs | 0.52 prose | 410 | ~4.5 |
| + expert parallelism for the lone row (six routed experts on six buses, 12 kB each way per box) | 0.52 | ~290 | ~5.9 |
| + int4 attention projections (changes numerics) | 0.52 | ~240 | ~7 |
| + an acceptance rule that keeps a guess the model finds likely (not exact) | ~0.7 | ~240 | ~10 |

So on open prose **10 tok/s with exact output is not within reach of this fleet today**: it would need a >= 0.85 at
today's L, or L <= 130 ms at today's a. Even with every lever above stacked (kernels at the head's rate, expert
parallelism, a distilled drafter) the estimate is ~7 tok/s; attention alone is 66 layers x 1.2 ms = 80 ms on one
box at 108 GB/s before a single expert is read. It is reached today on text the drafter predicts well (true/false + justification 10.9, memorised
prompts 10.7, copying 12.8), and everything in between is the table above.

Expert parallelism, re-assessed: latency no longer forbids it (16 ms per token at 0.25 ms per round trip), the 1 GbE
wire still taxes it (72 kB out and back per layer through the driver's port = 1.2 ms of a 1.5-1.9 ms layer, against
3.45 ms on one bus), and it needs every box to hold 1/11 of every layer instead of all of six layers: 43 GB to move
per box, and the 64 tok/s multi-stream mode is lost while it is active (EP moves ~24 MB per token through one
port). It is a mode, not an upgrade.

## Where one token's time goes (measured, 015c, one stream, 128 tokens, story prompt, 3.38 tok/s)

Source: every rank's stage profile over the phase (`experiments/015c_ensemble/analysis.json`, phase `fam3_story`);
11 ranks x 300-550 frames each. "iGPU" = time inside the OpenVINO GPU call, "CPU" = Rust code around it.

### 1. One trip through the fleet: 466 ms (rank 0 starts the frame -> rank 0 reads the token it produced)

| where the time goes | ms | % of the trip | bytes read | effective rate |
|---|---|---|---|---|
| **iGPU, fused expert layers** (64 layers x 8 experts, one call per layer) | 226.7 | **48.6 %** | 16.76 GB | 72-75 GB/s (25 W ranks), 76-80 (42-48 W ranks), 63 (rank 0, beside the drafter) |
| **iGPU, attention projections** (int8 q/k/v/r + o, two calls per layer) | 120.3 | **25.8 %** | 8.59 GB | 67-69 / 81-85 / 64 GB/s |
| CPU, rest of attention (KV attention over the cache, convolutions, norms, RoPE) | 25.0 | 5.4 % | small | - |
| iGPU, rank 0's two dense layers (+ rank 0's four routers, ~1.4 ms) | 21.2 | 4.5 % | 0.52 GB | **26 GB/s: three times slower per byte than the expert path** |
| CPU, routers + MoE glue, ranks 1-10 (top-6 of 256, weight rescale, tensor in/out) | 18.9 | 4.0 % | small | - |
| iGPU, output head on rank 10 (int8, 201,024 x 6144) | 11.5 | 2.5 % | 1.24 GB | **108 GB/s** |
| CPU, everything else inside the stages (embedding, residuals, frame encode/decode) | 0.4 | 0.1 % | - | - |
| **not compute** (round trip 413.2 ms minus ranks 1-10's 370.9 ms of work) | 42.4 | **9.1 %** | | |
| &nbsp;&nbsp;LAN: 10 forward hops of one hidden state (~25 kB, ~0.45 ms each) + the reply | ~5 | ~1.1 % | | estimate from 014's 12 kB pings |
| &nbsp;&nbsp;rank 0 reads the reply late: it checks the link only between guess frames (53 ms each, busy 70 %) | ~18 | ~4 % | | estimate, not yet counted directly |
| &nbsp;&nbsp;frame pick-up on ten ranks that had gone idle (idle-link pings cost +0.2-0.4 ms) | ~3 | ~0.7 % | | estimate |
| &nbsp;&nbsp;not attributed (rewind frames, runtime outside the timed regions, scheduling) | ~16 | ~3.5 % | | |
| **total** | **466.3** | 100 % | 27.1 GB | |

By device: **iGPU 81.5 %** (380 ms), CPU 9.5 % (44 ms), network + waiting 9.1 % (42 ms). By rank: rank 0 53.1 ms
(11.4 %, the slowest stage: dense layers + the drafter's threads), ranks 1-7 36.8-38.1 ms each (8 %), ranks 8-9
33.0-33.7 ms (7 %, same kernels, higher power limit), rank 10 31.9 + 11.5 ms head (9.3 %).

### 2. One output token of prose: 296 ms = 0.41 x 53 ms + 0.59 x 466 ms

52 of 128 tokens had a right guess behind them and cost one rank-0 frame; 71 had a wrong guess and cost a whole
trip (5 had none). Model 298 ms, measured 296 ms. Per output token:

| | ms | % |
|---|---|---|
| iGPU expert layers | 141.4 | 47.4 % |
| iGPU attention projections | 76.4 | 25.6 % |
| iGPU rank 0 dense layers (every token pays rank 0's frame, right guess or not) | 21.2 | 7.1 % |
| CPU attention | 16.0 | 5.4 % |
| CPU routers + glue | 11.2 | 3.8 % |
| iGPU head | 6.8 | 2.3 % |
| network + waiting + unattributed | 25.2 | 8.4 % |

Seen as wall time: **93 % of a token's time is waiting for wrong-guess trips, 7 % is the cadence of right guesses.**
Seen as fleet capacity (11 boxes x 296 ms): 13 % useful frames, 43 % guess frames that are thrown away (3.7 guesses
sent per token, 0.41 right, and nothing cancels a wrong one before it has crossed all eleven ranks), 44 % idle.
On arithmetic (a = 0.78, 8.2 tok/s, 122 ms per token) the same trip is paid 22 % of the time.

### 3. What this says about the levers

If every weight-reading path ran at the rate the head already reaches on the same iGPU (108 GB/s): experts 226.7 ->
155 ms, attention projections 120.3 -> 80 ms, dense 20 -> 5 ms: **a trip of ~340 ms instead of 466 (-27 %)**, the
same size as expert parallelism and without re-sharding anything. Where the gap comes from, most likely first:

1. **Many small GPU calls.** Attention is 12 calls per frame of ~0.9 ms each, experts 6 calls of ~3.5 ms; the head is
   ONE call over 1.24 GB. A fixed 0.3-0.5 ms per call would explain most of the attention gap (12 x 0.35 = 4.2 ms of
   11.4). Test: fuse a rank's q/k/v/r (and o) IRs so a frame makes 2-6 calls, or chain them asynchronously.
2. **The 25 W platform limit** on ranks 0-7: the same kernels run 18 % (attention) and 7 % (experts) faster on the
   42-48 W ranks: ~28 ms of the trip.
3. **int4 experts cost more arithmetic per byte than the int8 head**, and eight experts scattered in an 8.4 GB buffer
   defeat prefetching. Test: time one fused layer with 8 adjacent experts against 8 scattered ones.
4. **Rank 0's dense layers at 26 GB/s** are a kernel problem (compressed MatMul with a dynamic row count, 24,576
   wide): ~15 ms, and rank 0 is the stage that sets the cadence of right guesses.
5. Rank 0 as coordinator AND stage: ~18 ms per trip read late. A drafter-aware rank 0 that stops sending guesses
   nobody expects to be right would be free to read at once.

None of this was visible while the doc claimed the bus was saturated at 62 GB/s.

## Where the weights live decides how many memory buses serve one token (2026-09-20, the user's question)

"Eleven boxes with everything in RAM only slightly outperform one box for a single stream: why?" Because of how the
weights are PARTITIONED, not how they are encoded. The fleet's RAM is 78 % full (535 GB of 690 GB), so every layout
is a partition (each weight stored once); the partitions differ in how many boxes can work on one token's layer.

One token reads 27.1 GB, in 66 strictly sequential layers; inside a layer the eight experts are independent of each
other. One box alone reads most of that from RAM and the cold experts from NVMe: ~1.1 tok/s. The fleet, cut BY LAYER
(six layers per box), reads all of it from RAM but still through one memory system at a time: 466 ms = 2.1 tok/s
before speculation. **Pipeline parallelism multiplies throughput (65 tok/s, ~30x one box); for one stream it is
one PTL box that never misses RAM.** Speculation is how the idle ten boxes were put to work on a lone stream so far.

The three ways to cut the same bytes (lone row, measured kernel rates, this LAN: 1 GbE, 0.25 ms round trip, a hidden
state is 12.3 kB at f16):

| cut | boxes reading one token's layer | network per layer | expert part of a layer | trip | multi-stream wire cap | output |
|---|---|---|---|---|---|---|
| by layer (today) | 1 | none (1 hop per 6 layers) | 3.6 ms | 466 ms | none (65 tok/s compute-bound) | exact |
| by expert: each box holds 1/11 of every layer's 256 experts (+ the two shared ones everywhere, 4.2 GB) | ~5.5 of 11 (two of the six routed experts share a box in 81 % of layers) | 74 kB out, 74 kB back | ~1.9 ms | ~360 ms (-23 %) | ~150 tok/s | exact |
| by slice: every expert's 3072 inner neurons cut in 11 (slices of 288/192, multiples of the 32-weight groups); every box holds a slice of all 16,512 experts and returns ONE summed vector per layer | 11 of 11, perfectly balanced, independent of routing | 12 kB to ten boxes (123 kB unicast, 12 kB if multicast), 123 kB back | 1.5-2.0 ms | ~350 ms (-25 %) | ~53 tok/s | not bit-identical (the sum order changes) |

On THIS LAN both cuts land at about -25 %, because 1.2 of the ~1.9 ms is the 1 GbE port pushing six to ten copies
of a 12 kB vector out and as many back. What the same cuts would give as the other terms move:

| | trip | prose (a = 0.42) | structured (a = 0.78) |
|---|---|---|---|
| today | 466 ms | 3.4 tok/s | 8 |
| + weights cut inside the layer, this 1 GbE | ~350 | 4.4 | 10 |
| + GPU paths at the rate the head already shows (call fusion, rank 0's dense kernel) | ~290 | 5.2 | 12 |
| + a faster fabric (2.5 GbE dongles, or Thunderbolt between boxes: 0.1 ms, 20 Gb/s) | ~230 | 6.4 | 14 |
| + int4 attention (attention on the driving box is then the floor: 66 x 0.7 ms) | ~180 | 8 | 17 |

The floor of the whole idea: 27.1 GB / (11 x 100 GB/s) = 25 ms if all buses always worked, plus one network exchange
per sequential layer (66 x >= 0.4 ms): ~50-60 ms, 16-20 tok/s before speculation. Nothing on 1 GbE gets near it.

Encoding (fewer bytes per token) is the other axis, and it is independent of the cut: experts at ~3 bits instead of
4.6 (-25 to -35 % of 16.8 GB; needs a new export of 522 GB, calibration data, and an iGPU kernel that does not
exist in OpenVINO), attention at int4 (-4.3 GB). Both change the numerics. Entropy-coding int4 (3.5 bits of entropy)
cannot be decoded at 100 GB/s. Skipping low-weight experts or inactive neurons is lossy.

## The 15-stream regime (goal re-set by the user on 2026-09-20: interactive speed for up to 15 streams, ideally 60 tok/s aggregate)

Measured baseline (019, config of 015c, 15 mixed-task streams x 128 tokens, all arriving at once): **22.5-23.4 tok/s
steady = 1.4-1.6 tok/s per stream, first token after 31 s (median)**. 11 streams: 20.8; 22: 32.1; 33: 37.7.

Eleven stages want eleven frames in flight, so 15 streams means 1.36 rows per frame, and one round (every stream
one token) is the sum of eleven frame times at the pace of the slowest stage. Frame time of a middle rank (25 W):

| rows per frame | 1.00 | 1.36 | 1.92 | 2.90 | 14.4 |
|---|---|---|---|---|---|
| ms per frame | 35.1 | 40.8 | 50.6 | 70.8 | 154 |
| GPU attention projections | 9.7 | 9.6 | 10.2 | 12.4 | 14.4 |
| GPU fused experts | 21.2 | 25.0 | 31.2 | 44.3 | 120 |
| CPU (attention core, routers, glue) | 4.2 | 5.9 | 9.3 | 14.1 | ~20 |

`T(r) = 16 + 19 r ms` for r <= 3 (not the `28 + 9 r` of large batches): a second row reads its own eight experts
per layer (1.57 GB per stage, 12 ms: the marginal row moves at ~129 GB/s, the bus at work), while the FIRST row pays
~13 ms that are not bytes: 12 attention calls at ~0.3 ms and 6 fused-MoE calls at ~1.5 ms of fixed cost each.

What follows from that, in the order it matters at 15 streams:

1. **Two stages set the pace.** Rank 0 takes 48.3 ms per 1.36-row frame (its two dense layers run at 26 GB/s: 18 ms)
   and rank 10 takes 36.5 + 11.7 ms (the output head); everyone else 37-41 ms and waits 25-33 % of the time.
   Balanced, the same kernels give +18 %.
2. **The GPU is idle for a third of every frame** (CPU glue + per-call host work, during which no kernel runs) if
   the fixed cost per call is host time, which OpenVINO's dynamic-shape flow suggests (shape inference, kernel
   selection, argument setting per primitive per call). Two frames in flight INSIDE a box (its six layers cut in two
   halves with disjoint state, one thread each, sharing the iGPU) would fill those gaps: up to ~1.6x at one or two
   rows per frame. To be confirmed by measuring GPU time against wall time per call before building it.
3. **The byte ceiling.** Per round a stage reads 11 x 0.78 GB of attention + 15 x 1.57 GB of experts = 32 GB. At
   the 108-129 GB/s this memory has shown: 250-300 ms per round = **50-60 tok/s is the ceiling of this layout with
   today's encodings**; int4 attention would lift it by ~15 %. 60 tok/s is at the edge of physics, 40-50 is the
   engineering target.
4. **Speculation does not help here.** A guess row costs a full set of expert reads (19 ms of a 41 ms frame); it pays
   only above ~0.6-0.75 acceptance, which only text the tables have memorised reaches. It stays a lone-stream tool.
5. **First token.** A prompt travels as ONE frame through eleven stages in series (5 s for 45 tokens alone, 31 s when
   fifteen arrive together and are admitted in fat frames that also stall everyone's decoding). Windows of <= 32 rows
   sent back to back pipeline one prompt across the stages, and stay on the GPU plugin's small-batch MoE path.

### The 15-stream regime, as measured (026-033; supersedes the estimates in items 1, 2 and 4 above)

**The constants held.** A frame of r rows costs a 25 W stage `16 + 19 r` ms. The 16: 0.78 GB of int8 attention read
once per frame (8-9 ms at ~90 GB/s) plus about 1.2 ms of fixed cost in each of six fused-expert calls. The 19: one
row's 1.57 GB of experts (12 ms, at the bus limit) plus ~5 ms of CPU work and glue. Nothing on the device-call side
moved them: the plugin's decode kernels cost `1.4 + 1.7 r` ms per call against the prefill path's 3.0 ms for one row
and 4.8 for two (026, 029: no gain, and they share no expert between rows); two threads of device calls overlap
1.04-1.10x (023); letting the completion wait sleep costs 2.3 ms a frame (029).

**What turns the ring.** F frames circulate over eleven stages. A round (every stream one token) takes the larger of
* the busiest stage's work per round: the last rank, `F x (13.5 + 11.7 head) + 17 x 15 rows` = **25 F + 255 ms**, and
* one frame's trip through all stages: `11 x (16 + 19 x 15/F)` + hops = **179 + 3135/F ms**.

| frames F | 8 | 9 | 10 | **11** | 12 | 15 |
|---|---|---|---|---|---|---|
| last rank's work per round | 457 | 482 | 507 | **532** | 557 | 630 |
| one frame's trip | 571 | 527 | 492 | **464** | 440 | 388 |
| round = the larger | 571 | 527 | 507 | **532** | 557 | 630 |
| tok/s (15 / round) | 26.3 | 28.5 | 29.6 | **28.2** | 26.9 | 23.8 |

Measured at F = 11: rounds of 544-556 ms, 27 tok/s raw, 24-25 as the sum of the streams' own rates. F = 10 might be
worth 5 % (`CASCADIA_STREAMS_INFLIGHT=10`, untested); nothing else in the grouping is. This is why making rank 0
faster (027: -8 ms) returned 0.5 %, and why sharing head calls, which makes a reply wait for another frame's layers,
lost 2.5 % (028): the ring is a closed loop, and a late reply is a late next frame.

**Ideal of this layout at 15 streams: about 32 tok/s** (every stage at the middle ranks' `11 x T(1.36)` = 461 ms, no
head penalty). The fleet is at 77 % of that. What could still move it, exactly:

| lever | worth at 15 streams | note |
|---|---|---|
| frames in flight 11 -> 10 | up to +5 % | one environment variable on rank 0; untested |
| the last rank's head cheaper per call | up to +8 % (then rank 0 paces) | int8 at 107 GB/s is already the bus; int4 through this GPU's generic path is slower per byte; no exact idea left |
| guess rows from the model's own MTP head (a1 = 0.73 measured, 033) | **about +7 %**: two rows per stream = 1.73 tokens for 1.62x the stage time | the same head is worth ~2x per stream at 3-8 streams, where the ring has idle room, and 3.4-4.1 -> ~5-6 tok/s for one stream |
| not exact: int4 attention (if a fast int4 path exists), top-4 instead of top-6 experts | +10 %, +20 % | owner's decision; changes the model's output |

60 tok/s at 15 streams would need 15 rows x 1.57 GB x 11 stages + attention + head = ~34 GB per round per stage in
250 ms = 136 GB/s sustained with nothing else in a frame: the bus's rated peak. It is not there with exact int4
experts on these boxes.

**The boxes are not equal, and one is odd.** Ranks 8-10 (no 25 W platform limit) run a stage in 35-36 ms against
39-42 ms. The entry box, doing a middle rank's work since the role swap (032), needs 44 ms and 20.0 W where its seven
identical siblings need 39-41 ms and 17-19 W: a unit or power-supply problem, the same box that lost all power three
times in one day.
