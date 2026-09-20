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
