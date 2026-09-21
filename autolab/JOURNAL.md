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

## 2026-09-20, iterations 004-006: rows share reads, admission stops blocking, the power limit is measured

004 (multi-row int4 kernel, batched dense MLP, batched admission, pre-warm): rank 0 left the
bottleneck (181.7 -> 129.6 ms/frame), the 25 W ranks took it over at 95-99 % busy. Steady decode
20.7 tok/s at 48 streams, 24.5 at 96; short-prompt TTFT 9.6 -> 5.8 s; prefill per prompt row 2.5x
cheaper. The frame-time model `17 ms + attention(R) + 6 U(R) t_expert` fits R = 1, 4, 6.4, 14
and 19 within 10 %: decode is one memory read per distinct expert.

005/006 (short engine steps, admission while waiting): two different reasons admissions crawled.
First the engine lock: a step was a whole round, submits waited for it and arrived a few per round.
Then the reply wait: rank 0 blocked on a prefill frame's reply (up to 30 s through 11 ranks) with
most of a burst still unadmitted. 48-request burst TTFT 80 -> 73 -> 44 s, aggregate 12.6 -> 18.7.
264 streams: 34.9 tok/s steady. Platform watts confirm the limiter: ranks 0-7 sit at psys 24-25 W
(PL1 = 25 W) with clocks at 1.8-2.0 GHz; ranks 8-10 draw 42-48 W.

A number worth keeping: once the drafter table was on disk the gate prompts (seen many times) ran
at 8.5, 6.5 and 5.3 tok/s single-stream with exact output. It says nothing about unseen prompts
(1.9-2.0 tok/s), but it is the measured top of pipelined speculation on this fleet and it matches
the time model at a ~ 0.9.

## 2026-09-20, iteration 003: what the 25 W boxes do best

Seven identical boxes, one run, one variable each. Fewer threads lose (12: +14 %, 8: +43 % stage
time), also with the low-power E cores fenced off: all 16 cores earn their watts. Holding the CPUs
out of deep C-states is worth 8 % of a stage for a lone stream and nothing under load. The iGPU is
the find: three of six MoE layers fused at f16 (exact since the power-of-two weight rescale; zero
fallbacks in 1300 frames) cut the stage time by 24 % at 4 rows and 19 % at 14, at 5 W LESS package
power. A fused layer takes 23.7 ms at 15 rows where a CPU layer on the same box takes ~46: under a
power limit the GPU moves experts for fewer joules than 16 throttled cores. Why only three layers:
the kernel gives driver-owned system memory half of RAM by default, and the installer generated
three IRs per box to match.

Reasoning for 007/008: fuse everywhere (007), then lift the half-of-RAM limit and generate the
other three IRs on the boxes (008; run.sh now carries the generator). Memory is not the obstacle:
a fused layer costs 8.3 GB of device memory, a resident CPU layer 7.8 GB, so six fused layers are
~55 GB either way; the obstacle is the 8.4 GB host copy the shim makes while compiling each layer,
which at the sixth layer leaves about 3 GB free. 008 therefore starts on one rank.

## 2026-09-20, iterations 007-008: the iGPU takes every expert layer

007 (overrides only): each rank's three IR layers fused at f16: 45.7 tok/s steady at 264 streams,
unseen-prompt single stream 2.6-3.0 tok/s, answers 12/12; one gate prompt departs from the CPU
reference at character 81 with an equivalent phrase (half precision on the device is not
bit-identical, so an answer-level check joined the gates). Rank 1 showed the second overflow site
the f16 study had predicted: layer 8's shared expert passes 65504 inside the expert, 2.8 % of its
fused calls fell back, and the fallbacks filled the CPU cache until the box swapped.

008a/b: why only three layers per box? The kernel lets a driver own half of RAM (ttm pages_limit),
and the installer generated three IRs to match. But a fused layer (8.3 GB on the device) replaces a
resident CPU layer (7.8 GB): six fused layers need about the memory the box was already using.
run.sh now carries the IR generator (one layer at a time, moved into place when complete), raises
pages_limit for the boot when the overrides ask, and regenerates a layer with attenuated up scales
(layer 8: x 2^-4, multiplied back on the host; exact on the fixture). One rank first (rank 6:
169 ms/frame at 14 rows against 230-260 for its neighbours, memory healthy), then all:
**55.6 tok/s steady at 176 streams, 3.0-3.4 tok/s on unseen prompts, 12.3 tok/s on a memorised
one**, CPU expert cache 0 MiB on every rank, the 25 W boxes as fast as the 60 W ones. The platform
power limit stopped mattering the moment the experts left the CPU.

What limits now: rank 0 (its two dense layers still on the CPU: 60 ms of a 230 ms frame, 96.8 %
busy), swap-ins of driver-owned pages at swappiness 60, and a client-side descriptor limit that made
every phase above ~250 streams meaningless (the Mac's tunnel agent has 256 descriptors).

## 2026-09-20, iterations 009-011: the last stage imbalances

009: rank 0's dense layers onto the iGPU (229 -> 187 ms per frame): 57.9 tok/s steady at 176
streams. More streams stopped helping (352: 54.9): on the device a row costs about 10 ms per rank
whatever the batch, so extra streams only add KV memory, and at 352 streams five ranks swap.
010: 30k tokens of varied traffic into the cross-request drafter: unseen prompts stay at ~3.3
tok/s (a ~ 0.3). Word n-grams saturate; past that a draft has to understand the text.
011: with the experts on the iGPU the CPUs idle, and the per-row part of attention (convs, head
norms, softmax over the row's own cache) still ran row after row: 20-30 ms of a 190 ms frame.
Rows are independent sequences, so they now run concurrently, bit-identical per row:
**64.2 tok/s steady at 176 streams.**

Where the two targets stand, and why:
- aggregate > 60 tok/s: met in steady decode (64.2 at 176 streams, 60.9 at 264); from 9.7 this
  morning. Aggregate over a whole short phase is lower (37-38) because a third of a 32-token phase
  is admitting 176 prompts.
- single stream > 10 tok/s: met only on prompts the drafter has seen (12 tok/s, exact output);
  3.3 tok/s on unseen prompts, from 1.6. A stage is ~40 ms now (24 ms of expert reads at the memory
  bus limit + 13 ms of int8 attention projections, also bus-bound), eleven stages in series are
  440 ms, and only right guesses shorten that: `T (a + (1 - a) D)`. a = 0.3 today. 10 tok/s needs
  a ~ 0.85: a trained draft head on the last rank's hidden states (EAGLE-style), not an n-gram table.

## 2026-09-20, iterations 012-015: the single-stream equation, term by term

The user: "keep going until we hit 10 tok/s single stream. think from first principles and outside the box."
So each term of `a*T + (1-a)*L` got measured instead of assumed.

012 (anatomy): L = 408 ms of bus-bound work + hops + 45 ms that rank 0 wasted computing guess frames before
reading the reply that mattered; a = 0.30. With a fixed near 0.3 the fleet already ran within 10 % of what the
topology allows: the target needs both a and L to move.
013 (offline, the fleet's own text): how right can a drafter be? n-gram tables 30-38 %, Qwen3-0.6B 54 %, 4B 61 %,
an oracle over two drafters 66 %, and by task 0.39 (story) to 0.78 (arithmetic). Nothing reaches the 0.85 that
10 tok/s needs at this L.
014 (the user pointed at their poll-mode NIC driver project): the 3.3 ms LAN round trip was `cdc_ncm` holding small
frames for up to 1.2 ms per direction. `tx_timer_usecs = 0`: 0.2-0.3 ms. Latency no longer rules expert parallelism
out; the 1 GbE wire and a 43 GB re-shard per box remain its price.
015: a 0.6B drafter model beside rank 0 (text in, the target's tokens out, any vocabulary), a rank 0 that reads
replies between guess frames, tables where they are sure. First release broke every lone request for a few minutes
(a waiting round surfaced as an empty engine step; rolled back, fixed, and the test now fails the old code).
Result: prose 3.0 -> 3.1-3.7, structured tasks 5.9-10.9, memorised 10-12, 176 streams 65.5. Exact.

Also fixed: seven raw telemetry files (they name lab hosts) had been force-added with experiment folders and pushed.
The branch was rewritten without them and the harness now writes raw telemetry outside the work tree.

Next, in order of return per effort: distil the drafter on this model's outputs (a +0.1 on prose), expert
parallelism for the lone row as a serving MODE (L -27 %), int4 attention (L -13 %, changes numerics).

## 2026-09-20 evening, iterations 019-029: the goal is now 15 streams

The user re-set the goal: interactive speed for up to 15 streams, ideally 60 tok/s aggregate at 15. Baseline 22.5-23.4
(019). What the numbers say (PHYSICS.md, "The 15-stream regime"; MOONSHOTS.md, F1-F6): eleven frames of 1.36 rows
turn around a ring at eleven times its slowest stage; a row costs 19 ms of a stage, the frame itself 16; the byte
ceiling of this layout is 50-65 tok/s with nothing else in a frame. 60 is not there with exact int4 experts;
30-35 exact, ~40 with lossy options.

* 024 prompts as 8-row windows (config): first token alone 5 -> 2 s, fifteen at once 31 -> 7 s; 24.2-24.6 tok/s.
* 020/023: GPU calls from two threads overlap 1.04-1.10x: the per-call fixed cost is on the device; splitting a
  box's layers between two frames is dead.
* 021/022/025/026: the plugin's MoE decode kernels. They never crashed: `infer()` throws because group 32 is
  refused on Xe2+ (swallowed assert), the engine fell back to host experts, the host cache filled the box, the OOM
  killer did the rest (cache now capped at 1 GiB fleet-wide). With group 64 they work and read at the bus limit,
  but share no expert between rows: worth +8 % at 15 streams, and group 64 is not exact. 029 goes after the exact
  route: six compiled `32`s in the plugin become `16`s (the pre-Xe2 configuration of the same code).
* 027: rank 0's dense layers through the fused-experts op: 51.9 -> 43.7 ms a frame, exact (cosine 0.999999 against
  the MatMul form); the ring gained 0.5 % because rank 10's head (11.6 ms per CALL) paces it just as much.
* 028: the last rank shares one head call between decode frames that are already waiting. **Operator error on its
  first rollout:** the gate and a 15-stream phase were started before the settle check had finished (a build
  task's "exited with code 0" was misread as the publish task's), requests reached a half-built chain, which then
  wedged (streams admitted, no replies, ranks 1-2 "waiting for the previous rank") until the next release. Rolled
  back to 027 within ten minutes, gates pass, re-run pending. Rule: no traffic before "steady 3/3" is on screen.
  Also worth a fix of its own: requests that arrive while the chain is assembling should be refused, not wedge it.

## 2026-09-20 late evening, iterations 029-033: two negatives, a dying entry box, a role swap

* **029, both negative.** The decode kernels run on the exact group-32 weights once six compiled `32`s in the GPU plugin
  become `16`s (the pre-Xe2 configuration of the same code; found by disassembling the byte-identical library on the
  build host; every layer of the canary rank passed the load check at 0.999994). A one-row call then takes 3.0-3.2 ms,
  the same as the prefill path: 026's "2.1 ms" was extrapolated from a call whose second row repeated the first row's
  experts. Fitted on distinct rows the decode path is `1.4 ms + 1.7 ms per row`. The ~1-1.4 ms every MoE call costs
  before its bytes is common to both paths. Queue throttle LOW halves the busy core and saves 1.5 W, and costs 2.3 ms
  a frame in wake-ups. Both removed (030).
* **What paces 15 streams, corrected.** 028's verdict called it a convoy behind the two-row frames. The better reading:
  a ring of F frames turns at the LARGER of (a) the busiest stage's work per round and (b) one frame's trip through
  eleven stages. The last rank does 11 x (36.6 + 11.7) = 531 ms per round (measured round 544-556). Fewer frames
  lengthen the trip (`179 + 3135/F` ms), more frames add fixed cost and head calls (`25 F + 255` ms): F = 11 is
  within 5 % of the optimum. So neither regrouping nor head sharing (which delays a reply by one frame's layers and
  lost 2.5 %) helps; only a cheaper frame or cheaper rows do.
* **The entry box lost all power three times** (13:39, ~20:38 one minute after a reload at idle, ~20:57 one minute
  into a 15-stream phase), its clock back to the firmware date each time. Two lessons that cost time: its reset clock
  blocks every release (the fleet manifest is stamped with it and updaters refuse older manifests; now self-healed in
  the overrides), and a fleet that is "11/11 serving" is not settled until `steady 3/3` (traffic into a half-built
  chain wedged it for ten minutes).
* **032 role swap**, on the user's request: the box installed as rank 8 plays rank 0; the entry box keeps the door,
  plays rank 8 and relays :8000. Built as two releases (verified LAN copy of each other's role data, then a `run.sh`
  whose `ROLE_SWAP` names the pair), tested in two containers named like the boxes before it touched the fleet,
  reversible in one release. Gates exact; 15 streams unchanged (23.6 / 24.6); rank 0's stage 44.2 -> 39.5 ms. The
  entry box, now doing exactly what seven identical boxes do, is 8-10 % slower and draws more: the fault is in the
  unit or its supply, not in the role.
* **033 (offline, the teammate's plan):** the checkpoint's own MTP head is right 0.726 of the time on this model's
  text (0.63-0.69 on prose); its deeper modules need their own attention and conv state kept current (0.11 without).
  One stream ~5-6 tok/s if wired; the logit-lens / delayed-guess idea is closed (0.00 raw through rank 4).
* **For whoever continues:** `OPERATING.md` (how to deploy without closing the only door; `lab.py publish` now refuses
  the releases that would), `QUEUE.md` (every experiment and its status; anyone may add items), the README's status.


## 2026-09-21: continuation, 034-037

The owner explicitly handed publishing over from tahoma-6d to
`autolab-continuation-20260921`. The initial fleet was settled, all eleven
workers served the recorded role swap, and the published binary/run/overrides
hashes matched the documented files.

034: ten frames in flight was negative. Paired fifteen-stream phases were
24.580/24.679 with eleven frames, 24.596/24.444 with ten; 176 streams fell
67.974 -> 64.846. Both output gates passed and every request completed. The
simple round-time approximation overstated the benefit of fewer frames.
Eleven frames are restored in 035.

035 adds an opt-in idle handshake through every worker before rank 0 admits
requests. A delayed-final-worker integration test proves TCP connectivity
alone does not open admission, and that the gate recovers without requests.
Nine pipeline/reconnect/speculation tests passed. The release settled and
both fleet gates passed. Paired fifteen-stream phases: 24.212 / 24.631 tok/s,
all requests complete, first token 6.61 / 6.10 s. Kept for reliability.

036 is kept (harness only): pipeline-role and installed-box identities are
separate; old profile windows are placed by their receive age and duplicate
windows excluded. The role-0 summary now reads installed box 8. The harness
also accepts `--prompt-tag` to compare identical prompts in separate records.

037 is built and tested, now rolling out: bounded f16 residual capture in the
runner, sampled token IDs from the actual last-rank sampler, rewind handling,
and completed-file downloads through a restricted route on the existing
entry relay. Capture, readiness, direct-return, head-sharing and speculation
tests passed. The original 36-prompt corpus and checkpoint are available for
re-scoring the shipped MTP head on fleet states.

037's first deployed capture format was insufficient: raw residuals before
the final RMSNorm exceed f16's range. Gates passed (24.28 / 24.42 tok/s at
15 streams), but downloaded-state validation found infinities in the dump.
Stopped before collecting the study corpus. INKCAP02 stores original f32
residuals; regression test includes values outside f16 range. Any future
12 kB MTP reply must carry normalized states, not raw final residuals.
