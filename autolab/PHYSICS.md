# What the hardware allows

Model (from the export manifest): 66 layers, hidden 6144, layers 0-1 dense
(intermediate 24576), 64 MoE layers with 256 routed experts (top-6) + 2 shared, expert
intermediate 3072, int4 experts 31.85 MB each, int8 attention projections, vocab 201k.

Bytes that must cross the memory bus for ONE token of ONE stream:

| part | bytes/token |
|---|---|
| experts: 64 layers x 8 experts x 31.85 MB | 16.3 GB |
| attention projections, int8, 66 layers x ~120 MB | 7.9 GB |
| dense layers 0-1 | ~0.9 GB |
| output head, int8 | 1.2 GB |
| **total** | **~26 GB** |

One box moves about 60-65 GB/s (LPDDR5x-8533, shared by CPU and iGPU). A pipeline runs the
layers one after another, so a single stream is served by ONE memory bus at a time:
26 GB / 62 GB/s = 0.42 s, a ceiling of **~2.4 tok/s**. Measured: 0.56 s (1.6-1.7 tok/s), of
which 35-50 ms is LAN latency. The CPU path is already within ~25 % of the sequential ceiling.
The iGPU shares the same bus: it cannot lift this ceiling for one row.

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
