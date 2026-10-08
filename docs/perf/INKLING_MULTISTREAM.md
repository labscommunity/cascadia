# Inkling multi-stream decode (continuous batching across a pipeline)

The sparse-MoE pipeline engine served one request at a time: rank 0 popped a
task, prefilled it, then drove one token per step through every rank while
every other rank sat idle waiting for that one frame. This note describes the
multi-stream scheduler that replaces it for Inkling, what it is built on, how
it was validated, and what it means for a multi-box pipeline.

## What changed

**Per-stream sequence slots in the layers.** `ShortConv` and `AttentionLayer`
keep a pool of parked sequence states (the conv history ring, the KV cache,
their cursors). `select(slot)` makes one of them live by swapping buffers —
O(1), no copying — so one set of weights serves many sequences.
`Layer::forward_rows(xs, rows, slots)` decodes one token per stream: the
attention projections run once for all rows (the weights are read once per
step), attention and the convs run per row on that row's slot, and the MoE
runs all rows as one batch-union (each expert is read once for every stream
that chose it — the aggregate-throughput lever). Per row the op sequence is
`forward_token`'s, so a stream decoded in a batch is bit-identical to the same
stream decoded alone on the CPU kernels (`tests/inkling_streams.rs`).

**Runner surface.** `StagedRunner` gains `configure_streams`, `open_stream`,
`open_stream_at`, `close_stream`, `stream_pos`, `prefill_stream`,
`decode_streams`, `head_logits_rows` (all default to "unsupported", so dsv4 /
glm5 / OpenVINO runners are untouched). `stream_pos` returns `None` for a
slot that is not open. The Inkling runner implements them; the OpenVINO
head takes all rows in one call. `CASCADIA_STAGE_PROFILE_SECS=<n>` (unset
or 0 = off) makes each multi-stream rank log a `stage profile` line every
n seconds while frames flow: time spent waiting, receiving, computing,
prefilling, in the head, sending, relaying and emitting, rank 0's frame
round trip, frame and row counts, the runner's attention and MLP times,
and the expert cache's hits, misses and retained MiB. Use it to find the
slowest stage of a pipeline.

**Single-stage scheduler** (`CASCADIA_STREAMS=N`). Each `step`: admit up to
`CASCADIA_STREAMS_ADMIT` (default 1) pending tasks — tokenize, take a slot,
prefill, sample the first token; emit every active stream's pending token
(one `Chunk::token` per stream per step; the runner fans them out by task
id); retire finished streams; run one batched forward for the survivors and
sample each stream's next token with its own history and rng. A forward
panic fails the batch's tasks, not the process. Aggregate tok/s is logged
every 16 steps.

**Pipeline wire.** Five appended frame kinds: `StreamOpen` (prefill a slot on
every rank; the last rank seeds a per-slot sampler and replies the first
token), `StreamFeed` (one window of a prompt longer than
`CASCADIA_STREAMS_PREFILL_WINDOW` rows, default 128, at most
`MAX_STREAM_ROWS` = 256; the first window opens the slot and only the last
one is sampled and answered; a receiver refuses a `StreamOpen` of more
than `MAX_STREAM_ROWS` rows), `StreamDecode` (one row per stream with `(slot, pos)`; every rank
decodes them as one batch on its own slots; the last rank samples each row
with its slot's sampler), `StreamClose` (free the slot everywhere),
`StreamTokens` (the reply). Every rank sets the same `CASCADIA_STREAMS`;
rank 0 picks slot ids, workers open the same ids.

**Groups in flight.** Rank 0 splits its streams into G groups
(`CASCADIA_STREAMS_INFLIGHT`, default = the rank count). One step serves
every group in turn. In a group's turn, rank 0 receives that group's
outstanding replies — the oldest frames on the wire, so the single reply
FIFO stays ordered — admits new streams into it, emits its ready tokens,
retires finished streams and sends one decode micro-batch.
Mid ranks wait on readiness of both sockets (`wait_readable` on each
activation stream, which is cancel-safe and consumes nothing) and treat
a frame from upstream and a reply from downstream as independent events, so
G frames are in flight and every rank is busy on a different group's rows.
The last rank stays sequential.

## Why groups pay: the cost model

A resident rank's cost per micro-batch is a fixed part (the attention
weights, kernel launches) plus a part that grows with rows (experts touched
grows sub-linearly: 8 distinct experts for 1 row, ~45 for 8, ~137 for 32, all
258 past ~100 rows; attention per row). With one frame through R ranks in
series, a step costs `R · cost(S)` for S tokens. With G groups of S/G rows in
flight, a step costs `max(cost(S/G), R · cost(S/G) / G)` for S/G tokens: for
G ≥ R every rank is busy and the throughput is `(S/G) / cost(S/G)` — better
than the serial `S / (R · cost(S))` exactly because smaller frames are cheaper
per row. If the cost were constant per frame the two would be equal; the gain
is real because it is not. `tests/inkling_streams_overlap.rs` charges
`10 ms + 5 ms/row` per micro-batch on a 4-rank loopback pipeline and measures
one group vs four: same tokens, about 1.6–1.9× less wall time on an idle
machine.

## Validation

| test | what it shows |
|---|---|
| `inkling_streams.rs` | streams decoded in a batch are bit-identical (logits and greedy ids) to each stream alone, including a stream admitted mid-flight and a slot reused after close; the single-sequence path is unchanged and still reproduces the HF reference ids |
| `inkling_streams_wire.rs` | 3-rank loopback pipeline, 5 tasks over 3 slots (admission, finish, reuse): every task's tokens equal the single-stage engine's |
| `inkling_streams_overlap.rs` | slow runner (10 ms + 5 ms/row per micro-batch), 4-rank loopback: with one group in flight exactly one rank decodes at a time; with four groups all four ranks must decode at once (asserted); four streams that arrive one per round go one row per group (wall time about 1.6–1.9× shorter, reported, not asserted) |
| `inkling_streams_single_stage.rs` | the single-stage scheduler: greedy tasks decoded together (admitted mid-flight, slots reused) give the one-task path's tokens; a seeded sampled task gives the same tokens alone and in a batch; a stream cancelled during decode frees its slot |
| `inkling_streams_long_prompt.rs` | prompts longer than one window (4 rows here) go as `StreamFeed` windows and give the single-stage tokens; a long prompt cancelled while its windows go down leaves no stream on rank 0; a lone prompt of more than three rounds of windows finishes through the runner |
| `inkling_streams_cache.rs` | the batched MoE path with the expert cache on: the hit pass is bit-identical to the miss pass, the tokens equal the eager reference, the second pass hits the cache, and a prefill never grows the cache |
| `inkling_streams_local_failure.rs` | a panic in rank 0's `decode_streams` with three groups in flight: each task of that round gets an error chunk, the owed replies are drained, and the next round gives the reference tokens on the same link with no re-dial and no worker exit |
| `inkling_streams_idle.rs` | a stream-mode pipeline idles past the frame idle ceiling, then serves a request with the same tokens; no worker exits |
| `inkling_streams_wire.rs::link_loss_mid_round_fails_each_stream_once` | the link to rank 1 dies with frames in flight: each task gets exactly one error chunk and nothing after it; rank 0 dials again and serves the next round |
| `inkling_streams_wire.rs::three_rank_cancel_during_decode_frees_the_slot` | a stream cancelled with its frame in flight retires on every rank; the others and the task that takes its slot get the single-stage tokens |
| `inkling_streams_wire.rs::rank_with_fewer_slots_fails_fast` | a rank with fewer slots than rank 0 exits with an error, and rank 0's task gets an error chunk soon, not at the reply deadline |
| `inkling_streams_wire.rs::over_budget_stream_finishes_with_length` | a stream that asks for more tokens than the context budget ends with `FinishReason::Length`, with the same tokens on a single stage and on the pipeline |
| `inkling_streams_wire.rs::bad_stream_decode_frame_is_rejected_without_panic`, `failed_stream_frame_closes_the_link_at_once` | a decode frame for a free slot or a repeated slot, or a frame the last rank cannot serve, makes the worker exit with an error at once instead of a panic or a silent wait |
| `cascadia-transport` `{tcp,uds}_wait_readable_peeks_without_consuming` | `wait_readable` on TCP and Unix sockets: a wait cancelled by a timeout loses nothing, a ready byte stays on the wire, EOF gives `SocketClosed` |
| `inkling_streams_wire.rs::last_rank_exits_after_upstream_reset_{one_task,streams}` | a last rank whose upstream dies hard (TCP reset) exits its step loop for the supervisor instead of spinning on `NotConnected`, on the one-task path and in stream mode (fails without the fix) |
| `inkling_streams_wire.rs::rank0_redials_after_downstream_restart` | a TCP forwarder cuts the rank 0 link the way a dying neighbour does: neighbours restart while rank 0 idles → no request fails, same tokens; neighbours down → fast error; back → served again (fails with the probe disabled, and with the re-dial disabled; passes on macOS and Linux) |
| local API run (`cascadia run`, fixture, `CASCADIA_STREAMS=4`) | four concurrent `/v1/completions` return exactly what the one-task path returns |

Measured on hardware below (four boxes). Not yet run: a longer pipeline
and a Linux iGPU rank (no Linux Panther Lake box was reachable).

## Measured on four boxes (2026-09-18)

Test bed: delta (192.168.0.122, 1 GbE) as rank 0 and the NUCs alpha, beta,
charlie (2.5 GbE) as ranks 1–3 — all Core Ultra X7 358H, 32 GB, Windows 11
— on the home LAN, CPU path only (no OpenVINO on the boxes), the real
export sliced per rank (pushed from the miner over ssh), a manifest
truncated to 11 layers. No rank loads layers 5 and 8, so the pipeline runs
9 real layers (2 dense + 7 MoE). Its words mean nothing, but its per-layer
cost, wire and batching are the real thing. Ranks hold `[0,3) [3,5) [6,8) [9,11)` (rank 0: the two
dense layers + one MoE + embed; two MoE layers per NUC; head on rank 3): a
32 GB box cannot hold three MoE layers (23 GB) next to Windows.
`CASCADIA_STREAMS=16`, four groups in flight, 32-token answers, load from
a client on the LAN, rates from rank 0's log.

**Plain memory-mapped experts (the OS page cache holds the slice):**

| streams | per-stream tok/s | sum | client aggregate incl. TTFT | mean TTFT |
|---|---|---|---|---|
| 1 | 4.17 | 4.2 | 4.0 | 2.6 s |
| 2 | 2.52 | 5.0 | 4.5 | 4.1 s |
| 4 | 2.40 | 9.6 | 8.0 | 6.2 s |
| 8 | 1.44 | 11.5 | 9.0 | 9.5 s |
| 16 | 0.89 | 14.2 | 10.7 | 15.3 s |

Steady windows reached 16.3 tok/s at 8 streams (225 ms per round of four
groups). One stream costs 240 ms per token over 7 MoE + 2 dense layers,
i.e. ~30 ms per MoE layer: the untuned mmap kernel regime (the same 30 ms
the tate-07 layer dump measured for the CPU kernels). The overlap is real:
16 streams deliver 3.4× the single stream's tokens.

**The autolab's tuned read profile on the same ranks** first measured
*slower* (1.5 tok/s single, 6.9 tok/s sum at 16 streams, TTFT 6–33 s):
its unbuffered reads bypass the page cache and, until this branch, the
batched MoE path — which every multi-stream decode step uses — never
consulted the expert cache, so every step re-read every expert from NVMe.
The batched path now looks up its unique experts once and admits misses
after compute (`inkling_streams_cache.rs`: the hit pass is bit-identical to
the miss pass, the tokens equal the eager reference, the second pass hits
and a prefill never grows the cache). With that fix the tuned profile is the CPU
configuration to deploy:

| streams | per-stream tok/s | sum | client aggregate incl. TTFT | mean TTFT |
|---|---|---|---|---|
| 1 | 8.79 | 8.8 | 6.0 | 1.8 s |
| 2 | 3.47 | 6.9 | 6.1 | 2.9 s |
| 4 | 3.22 | 12.9 | 10.3 | 5.2 s |
| 8 | 2.03 | 16.3 | 13.3 | 6.5 s |
| 16 | 1.07 | 17.1 | 13.4 | 11.8 s |

Steady windows reached 19–21 tok/s. One stream costs 114 ms per token over
the 9 layers, ~14 ms per MoE layer including the hops — 2.1× the mmap
regime, and about the 12–15 ms the tate-07 whole-model profile showed for
its cache-resident layers. TTFT halves as well (the prefill also runs from
the cache).

**Rank 0 on the iGPU** (delta's Arc B390 through the side-by-side OpenVINO
2026.3.1 runtime: int8 attention IRs on its three layers with the Rust
copies released, the int8 head IR, the fused MoE IR for layer 2 generated
on the box in 52 s; the NUC ranks unchanged): 8.4 tok/s single, 17.8 sum at
16 streams, windows to 21.5, zero fallbacks. Rank 0 owns one MoE layer, so
the pipeline's number barely moves; the point of the run is that the whole
iGPU path — IR generation, fused kernel, multi-stream frames — works on a
Windows box.

**What a paged three-layer rank looks like**, for contrast (the first
attempt, three MoE layers per NUC with the tuned profile at 8 GB of cache
per layer, RAM oversubscribed): 1.25 tok/s single stream — 85 ms per MoE
layer, exactly the 256 MB of expert bytes per token at the NVMe's 3 GB/s
— and 5.6 tok/s sum at 16 streams. Residency is everything.

## What to expect on a 12-rank pipeline

From the per-layer numbers in `INKLING_SINGLE_BOX_BENCH.md` (resident
ranks, 5–6 layers per box):

| streams in flight | per-stream tok/s (CPU / iGPU) | aggregate tok/s (CPU / iGPU) |
|---|---|---|
| 12 (one per rank) | 1.7 / 2.7 | 20 / 33 |
| 96 (8 per rank) | 0.45 / 1.0 | 43 / 96 |
| 384 (32 per rank) | 0.15 / 0.35 | 59 / 135 |

Rank RAM per stream slot: about 10 MB per layer at `CASCADIA_INKLING_MAX_SEQ`
1024 (33 MB for a global-attention layer at 4096), so 32 slots on a 6-layer
rank cost ~2 GB. Windows keeps the iGPU at three fused MoE layers per 64 GB
box; the CPU column needs no OpenVINO on the boxes at all.

## Where expert-parallel fits

The expert-parallel star (PR #156) moves every token's hidden state to up to
eight workers per MoE layer and their expert outputs back: ~24 MB per token
through the driver's one NIC. On 2.5 GbE that caps the star at roughly 12
tok/s aggregate however many streams are batched (~25 with FP16 both ways);
the pipeline moves 288 KB per token. Expert-parallel is therefore the wrong
topology for aggregate throughput. Its place is (a) single-stream latency on
a switched LAN with sub-millisecond round trips — the scaling note puts it at
~1.5–2× the pipeline, unmeasured — and (b) the RAM-starved regime where boxes
cannot hold their layers and reading a token's experts on several NVMes at
once is worth 64 network rounds. For the installation, run the pipeline with
streams; keep the star as a fallback if boxes turn out smaller than 64 GB.

## Restarts

A worker rank exits when a neighbour goes away, by design: its listener
accepts exactly once, so a fresh process is the only clean reconnect. Run
every worker rank under a supervisor that starts it again. Rank 0 is the
exception: it is a client of rank 1, so it keeps its process and its API
and dials again in place. Before it admits a request on an idle link it
probes the socket (the downstream never sends unsolicited bytes, so EOF, an
error or data means dead or out of sync). While no request is in progress,
a background keeper does the same probe every 2 s and dials again when the
link is dead. A latched wire failure aborts the open streams and dials
again with a 2 s budget at most every 3 s.

What that looks like from outside, measured on the four-box bed:

- a box restarts: the ranks behind rank 0 restart once (about five seconds
  plus load time); the next request is served on a fresh connection and
  does not fail (rank 0's log: `downstream link found dead while idle;
  re-dialing`, reconnected 22 ms later);
- a request made while a rank is still down waits on rank 1, which is itself
  waiting for its neighbour: it completes if the box comes back within rank
  1's 300 s connect budget and fails otherwise; requests after that fail
  fast until the box is back, then are served again with no manual step;
- an idle pipeline stays up. It did not before: the last rank waited for its
  next frame in a receive the transport bounds at 900 s, so every 15 minutes
  of silence the pipeline rebuilt itself. The last rank now waits with the
  same readiness peek the middle ranks use. For binaries built before that
  change, set `CASCADIA_FRAME_IDLE_CEILING_SECS=0` on every rank.

Bugs found on the bed along the way, all fixed in this change: rank 0 kept
serving a dead socket's error until it was restarted by hand (now it dials
again in place); the last rank spun forever on `worker recv_kind failed:
not connected` after its upstream reset, so the restarted middle rank could
never reconnect (now it exits for its supervisor); the 15-minute idle
teardown described above.
