# Journal

## 2026-09-20, iteration 000: instrument first

Question: where does the time go? Nothing on the fleet could answer it (every rank says
"serving"). Built a per-rank stage profile into the engine and a telemetry packet into the
beacon, rolled both out, ran 48/1/11/16/48 streams plus long prompts.

Found: (1) the ranks idle 65-70 % at 48 streams because streams pile into the first groups;
(2) experts are not shared between rows at all; (3) eight boxes are power-capped at 25 W;
(4) prompts over 256 tokens crash the chain; (5) LAN ping is 2-3.5 ms; (6) prefill costs as much
per row as decode. The long-prompt phases took the chain down four times; Tailscale on rank 0
went away mid-run, so the recording of the last phases stayed on rank 0.

Reasoning for the next step: the two targets need different things. Aggregate: fill the
pipeline (M1, M4), then make rows share expert reads (M2) with the iGPU doing the GEMM (M3).
Single stream: the sequential bandwidth ceiling is 2.4 tok/s; only several positions in flight
(S1) can pass it. First release batches M1, T1, the telemetry door and the profile fix.

## 2026-09-20, iteration 001: fill the pipeline, stop the long-prompt outage

Release channel: the connectivity session's signed channel (`release.py publish` -> rank 0
poller -> fleet updater) is now this loop's only door; Tailscale on rank 0 is off. Lock taken,
baseline binaries and overrides kept in `~/inkling-release/baseline/`.

Built while the fleet measured: (a) emptiest-group admission, (b) StreamFeed windows for long
prompts, (c) stage profile reports a request's last window at once, (d) fused MoE made safe at
f16 (power-of-two weight rescale, non-finite fallback, compile at load), (e) pipelined
speculation for a lone stream with StreamRewind, (f) direct reply link last rank -> rank 0.
(a)-(d) are in binary 71973619 (exp 001); (e), (f) follow in exp 002.

Research notes behind (e) and (f):

- Single stream is served by one memory bus at a time (PHYSICS.md): the only way past ~2.4 tok/s
  with layers sharded by box is to have several positions of the stream in flight. A wrong
  guess costs at most one stage time (the corrected frame queues behind one dropped frame at
  rank 1, then follows it down the pipe in lockstep), a right guess saves a whole round trip:
  time per token = a*T + (1-a)*(D*T + ~T/2). The draft decides everything: a = 0.3 gives 1.4x,
  0.5 gives 1.85x, 0.9 gives 5.5x.
- With balanced groups 11 and 16 streams both settle near 9-10 tok/s of steady decode although
  the stage times allow ~16 at 11 streams: the group turn on rank 0 takes ~110-150 ms where a
  stage takes 53-90. Replies are relayed up through nine ranks that each forward only between
  two of their own frames; with every rank busy a reply waits at most of them. A direct
  connection from the last rank removes all nine waits.
- The fused-MoE study (research/fused_moe_f16.md) found the systematic overflow site: routing
  weights sum to 8 x global_scale (about 100 per weight at layer 40), so the weighted sum leaves
  f16. Dividing a row's weights by a power of two and multiplying the output back is exact.
  It also raised a doubt to settle on the fleet: the f32 hint may never have compiled the fused
  kernel, in which case "fused f32 = 1.66 tok/s" was the CPU path plus overhead. The stage
  profile now carries ov_moe_calls / ov_moe_fallbacks / ov_moe_nonfinite to tell.

## 2026-09-20, iterations 002a/002b: the reply path and the first speculation on real hardware

002a (direct reply link): 11 streams 8.9 -> 12.9 tok/s steady, 48 streams 14.7 -> 18.4. First
per-rank view through /api/fleet/telemetry: at 48 streams rank 0 is 96 % busy, ranks 1-7 84-90 %,
the three 60 W boxes 62-71 %. Fleet mean 82 % (exp 000: 30 %). The scheduling losses are gone;
what is left is time per row on the slowest stage, which is rank 0 (46.7 ms/row at 3.9 rows per
frame) because its two dense layers run row by row, then the 25 W boxes (40-43), then the 60 W
boxes (30).

002b (speculation): output character-identical; +5-30 % on free-form reasoning, 2.79 tok/s on a
copy task where 47 of 96 tokens were right guesses. The time model
`T (a + (1 - a) D)` predicted 2.8 for that acceptance. So the mechanism delivers exactly what the
draft's acceptance allows, and everything about the single-stream target is now a question about
drafts: a = 0.5 -> 2.9 tok/s, 0.7 -> 4.6, 0.9 -> 8.6 at T = 58 ms (the 25 W limit lives in T:
at the 60 W boxes' 41 ms the same acceptances give 4.1 / 6.5 / 12).

Reasoning for 004: with the pipeline full, per-row cost is the only lever. Rows must stop
re-reading what they share: the shared experts (2 of every row's 8), the dense layers (all of
rank 0's rows), and routed experts once frames carry enough rows to collide (8.7 rows/frame at
96 streams: 70 routed pairs on ~50 distinct experts). The multi-row int4 kernel is bit-identical
per row, so the gate stays character-exact; it now also runs 8 requests side by side, because a
lone request never enters a multi-row kernel. Batched admission attacks the burst TTFT (91 s mean
at 48): ten 24-token prompts touch ~250 experts together, not 10 x 135. Pre-warm makes every
experiment start from the same state (002a's single-stream number was taken with the expert
cache at 60-90 % and misses up to 1.4 %).

Measurement hygiene learned: compare steady decode (sum of per-stream rates) across experiments,
not aggregate (it includes the admission ramp, a third of a 64-token phase at 48 streams); warm
with many streams after every restart, or pre-warm.
