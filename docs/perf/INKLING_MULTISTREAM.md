# Inkling multi-stream decode (continuous batching across a pipeline)

The sparse-MoE pipeline engine served one request at a time: rank 0 popped a
task, prefilled it, then drove one token per step through every rank while
every other rank sat idle waiting for that one frame. This note describes the
multi-stream scheduler that replaces it for Inkling, what it is built on, how
it was validated, and what it means for a 12-box installation.

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
glm5 / OpenVINO runners are untouched). The Inkling runner implements them;
the OpenVINO head takes all rows in one call.

**Single-stage scheduler** (`CASCADIA_STREAMS=N`). Each `step`: admit up to
`CASCADIA_STREAMS_ADMIT` (default 1) pending tasks — tokenize, take a slot,
prefill, sample the first token; emit every active stream's pending token
(one `Chunk::token` per stream per step; the runner fans them out by task
id); retire finished streams; run one batched forward for the survivors and
sample each stream's next token with its own history and rng. A forward
panic fails the batch's tasks, not the process. Aggregate tok/s is logged
every 16 steps.

**Pipeline wire.** Four appended frame kinds: `StreamOpen` (prefill a slot on
every rank; the last rank seeds a per-slot sampler and replies the first
token), `StreamDecode` (one row per stream with `(slot, pos)`; every rank
decodes them as one batch on its own slots; the last rank samples each row
with its slot's sampler), `StreamClose` (free the slot everywhere),
`StreamTokens` (the reply). Every rank sets the same `CASCADIA_STREAMS`;
rank 0 picks slot ids, workers open the same ids.

**Groups in flight.** Rank 0 splits its streams into G groups
(`CASCADIA_STREAMS_INFLIGHT`, default = the rank count) and serves one group
per step: receive that group's outstanding replies — the oldest frames on
the wire, so the single reply FIFO stays ordered — admit new streams into it,
emit its ready tokens, retire finished streams, send one decode micro-batch.
Mid ranks wait on readiness of both sockets (a cancel-safe `peek`) and treat
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
one group vs four: same tokens, 1.5–2× less wall time.

## Validation

| test | what it shows |
|---|---|
| `inkling_streams.rs` | streams decoded in a batch are bit-identical (logits and greedy ids) to each stream alone, including a stream admitted mid-flight and a slot reused after close; the single-sequence path is unchanged and still reproduces the HF reference ids |
| `inkling_streams_wire.rs` | 3-rank loopback pipeline, 5 tasks over 3 slots (admission, finish, reuse): every task's tokens equal the single-stage engine's |
| `inkling_streams_overlap.rs` | slow runner (10 ms + 5 ms/row per micro-batch), 4-rank loopback: with one group in flight exactly one rank decodes at a time, with four groups all four are observed decoding at once (wall time 1.6-1.9x shorter, reported, not asserted) |
| `inkling_streams_wire.rs::last_rank_exits_after_upstream_reset` | a last rank whose upstream dies hard (TCP reset) exits its step loop for the supervisor instead of spinning on `NotConnected` (fails without the fix) |
| `inkling_streams_wire.rs::rank0_redials_after_downstream_restart` | a TCP forwarder cuts the rank 0 link the way a dying neighbour does: neighbours restart while rank 0 idles → no request fails, same tokens; neighbours down → fast error; back → served again (fails with the probe disabled, and with the re-dial disabled; passes on macOS and Linux) |
| local API run (`cascadia run`, fixture, `CASCADIA_STREAMS=4`) | four concurrent `/v1/completions` return exactly what the one-task path returns |
| the crate's 439 tests | no regression |

Measured on hardware below (four boxes). Not yet run: the 12-box fleet
itself and a Linux iGPU rank (no Linux Panther Lake box was reachable; the
installer's iGPU path is the same tools and IRs that ran on Windows).

## Measured on four boxes (2026-09-18)

Test bed: delta (192.168.0.122, 1 GbE) as rank 0 and the NUCs alpha, beta,
charlie (2.5 GbE) as ranks 1–3 — all Core Ultra X7 358H, 32 GB, Windows 11
— on the home LAN, CPU path only (no OpenVINO on the boxes), the real
export sliced per rank (pushed from the miner over ssh), a manifest
truncated to 11 layers so the pipeline is a complete model of layers 0–10
whose words mean nothing but whose per-layer cost, wire and batching are
the real thing. Ranks hold `[0,3) [3,5) [6,8) [9,11)` (rank 0: the two
dense layers + one MoE + embed; two MoE layers per NUC; head on rank 3): a
32 GB box cannot hold three MoE layers (23 GB) next to Windows.
`CASCADIA_STREAMS=16`, four groups in flight, 32-token answers, load from
the miner over the LAN (`lan_load.py`), rates from rank 0's log.

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
after compute (`inkling_streams_cache.rs`: bit-identical to the eager
reference, second pass hits). With that fix the tuned profile is the CPU
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

**The installed pipeline** (`deploy/inkling-fleet`: `install.ps1 -Rank 0`
on delta from the SSD tree — side-by-side runtime, private Python, fused IR
generated on the box in 27 s, scheduled task — with the three NUC ranks
under a restart loop; `bench.py` from another box): 8.5 tok/s for one
stream, 10.8 summed at 8 streams, 19.1 summed at 16 (13.5 tok/s
aggregate including the 11 s mean time to first token). Same numbers as
the hand-launched pipeline, from a box installed by the one command the
venue will use.

**Rank 0 on the iGPU** (delta's Arc B390 through the side-by-side OpenVINO
2026.3.1 runtime: int8 attention IRs on its three layers with the Rust
copies released, the int8 head IR, the fused MoE IR for layer 2 generated
on the box in 52 s; the NUC ranks unchanged): 8.4 tok/s single, 17.8 sum at
16 streams, windows to 21.5, zero fallbacks. Rank 0 owns one MoE layer, so
the pipeline's number barely moves; the point of the run is that the whole
iGPU path — runtime install, IR generation, fused kernel, multi-stream
frames — works on a Windows box that had nothing on it.

**What a paged three-layer rank looks like**, for contrast (the first
attempt, three MoE layers per NUC with the tuned profile at 8 GB of cache
per layer, RAM oversubscribed): 1.25 tok/s single stream — 85 ms per MoE
layer, exactly the 256 MB of expert bytes per token at the NVMe's 3 GB/s
— and 5.6 tok/s sum at 16 streams. Residency is everything.

## What to expect on the 12-box pipeline

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

## Deploying 12 boxes offline

Rank `r` of 12 needs only its layer slice: `manifest.json`, the tokenizer
files, `shells/layer_NN.safetensors`, `experts/layer_NN/`, optionally
`attn_ov/layer_NN/`, plus `embed.safetensors` on rank 0 and
`head.safetensors` (+ `head_ov/`) on rank 11 — 36 to 48 GB per box, from the
export on the miner's portable SSD. Each box runs

```
cascadia worker --rank r --total 12 --engine sparse-moe --model <slice dir> \
  --listen :91<r> --next <ip of r+1>:91<r+1> [--api :8000 on rank 0]
```

with the promoted CPU read profile (`tools/inkling_autolab/ptl-profile.ps1`),
`CASCADIA_STREAMS=<slots>` identical on every rank, and, where the iGPU is
used, `CASCADIA_INKLING_OV_ATTN=1 CASCADIA_INKLING_OV_ATTN_DIR=attn_ov_int8
CASCADIA_INKLING_OV_ATTN_DROP_RUST=1 CASCADIA_INKLING_OV_HEAD=1` (last rank).
Start the last rank first, rank 0 last (or in any order under a supervisor:
ranks retry their downstream until it accepts).

**Restarts.** A worker rank exits when a neighbour goes away, by design (its
listener accepts exactly once, so a fresh process is the only clean
reconnect), so every rank runs under a supervisor that relaunches it: the
installers use systemd `Restart=always` and a scheduled-task loop. Rank 0 is
the exception: it is a client of rank 1, so it keeps its process and its API
and dials again in place. Before admitting a request on an idle link it
probes the socket (the downstream never sends unsolicited bytes, so EOF, an
error or data means dead or out of sync), and a latched wire failure aborts
the open streams and re-dials with a 2 s budget at most every 3 s.

What that looks like from outside, measured on the four-box bed:

- a box restarts or is re-installed: the ranks behind rank 0 restart once
  (about five seconds plus load time); the next request is served on a fresh
  connection and does not fail (rank 0's log: `downstream link found dead
  while idle; re-dialing`, reconnected 22 ms later);
- a request made while a rank is still down waits on rank 1, which is itself
  waiting for its neighbour: it completes if the box comes back within rank
  1's 300 s connect budget and fails otherwise; requests after that fail
  fast until the box is back, then are served again with no manual step;
- an idle pipeline stays up. It did not before: the last rank waited for its
  next frame in a receive the transport bounds at 900 s, so every 15 minutes
  of silence the pipeline rebuilt itself. The last rank now waits with the
  same peek the middle ranks use, and the installers also set
  `CASCADIA_FRAME_IDLE_CEILING_SECS=0` (which is what protects binaries
  built before that change).

Bugs found on the bed along the way, all fixed on this branch: rank 0 kept
serving a dead socket's error until restarted by hand (first made to exit,
`b7336add`, then replaced by the in-place re-dial, `ea7a54ed`); the last rank
spun forever on `worker recv_kind failed: not connected` after its upstream
reset, so the restarted middle rank could never reconnect (`44f6b909`); the
Windows installer unregistered the previous task without ending its
`run.ps1` loop, which relaunched an orphan that held port 8000 (`9a33b412`);
the 15-minute idle teardown (`5bf7f9b1`, `ea7a54ed`).

The `cascadia-array` control plane
does exactly this for a ring of Windows boxes (bundled DHCP + mDNS, USB
enrollment, artifact pull over LAN HTTP, reverse-order start, health polls);
what it lacks for Inkling is the per-rank slice packaging, an env profile in
the plan, and a concurrent load generator — see its `docs/INKLING.md`.
