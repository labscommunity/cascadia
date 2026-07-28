# Continuous batching on the NPU: packed multi-slot decode ("seq-as-batch")

The NPU is the one device where cascadia cannot borrow continuous batching from
OpenVINO: paged attention lives in the CPU/GPU plugins, and the NPU servable is
stateful and sequential. This documents the route that does work, and the
measurements behind it. All numbers are Lunar Lake (Core Ultra 7 258V, Intel AI
Boost), OpenVINO 2026.2.1, `qwen25-1.5b-2stage-npu` stage 0.

## Why not batch

Reshaping a stateless static export's batch dimension from 1 to N is rejected by
the NPU compiler, identically on both graph shapes tested (embed+layers and
layers+lm_head), at N = 2, 4 and 8:

```
Failed Pass ConvertBatchedLayerTo1N
failed to legalize operation 'IE.Convolution' that was explicitly marked illegal
  @ aten::cat/Concat ["branches_concat_to_conv"]
```

Batch-1 compiles fine on the same graph, so it is the batch axis specifically.
The pass name is the useful part: the compiler's only strategy for batch > 1 is
unrolling into N batch-1 operations, so even a legalized batch axis would not
amortize the weight stream.

## Why the sequence axis instead

The same export compiles happily at seq > 1 — the chunked-prefill variant has
relied on that since #107. And the sequence axis is where the amortization is,
because decode is weight-bound, not KV-bound:

| | bytes per token |
|---|---|
| weights (int4 stage) | 458 MB |
| KV traffic (in + out) | 29.3 MB |

A **16:1 ratio**. One weight stream can therefore serve many query rows almost
for free. Measured with a shared mask (one sequence, S tokens):

| seq | ms/iter | ms/token |
|---|---|---|
| 1 | 10.57 | 10.57 |
| 2 | 18.78 | 9.39 |
| 4 | 19.70 | 4.92 |
| 8 | 21.70 | 2.71 |
| 16 | 22.81 | 1.43 |

From S=2 to S=16 — eight times the tokens — latency rises 18.78 → 22.81 ms.
Fixed cost ~18.5 ms, marginal cost ~0.29 ms per token. (S=1 sits oddly low
relative to S=2; the seq=1 graph likely takes a GEMV path. Read the trend from
S >= 2.)

## The mechanism

Pack N independent requests into the sequence dimension and isolate them with a
**block-diagonal attention mask**. Batch stays 1, so `ConvertBatchedLayerTo1N`
never runs.

The stock export cannot express this: it takes a 2D `attention_mask [1, T]` that
every query row shares — the same property that caps chunked-prefill parity at
the KV window. But it builds its 4D additive mask in a **single node** whose
output feeds every SDPA's mask input, and SDPA's mask slot already carries a
query dimension. Replacing that one node with a `[1, 1, S, T]` Parameter hands
mask construction to the host, where per-slot policy belongs.

`tools/packed_variant.py` performs exactly that edit — no HF model, no torch, no
re-trace — and works on an already-exported stage:

```bash
python tools/packed_variant.py <stage_dir> --slots 8
# or at export time
python tools/export_shards.py ... --target npu --packed-slots 8
```

It emits `openvino_packed_model.xml` beside the decode IR and records
`packed_slots` / `packed_seq` / `packed_context` / `packed_region` in
`stage_config.json`.

The host side lives in `crates/cascadia-engine-openvino/src/packed.rs`. The KV
window is partitioned into `region = past_len / slots` per slot; slot `s` owns
`[s*region, (s+1)*region)` in every layer and head. A `PackedPlan` maps each
query row to a slot, and one mask writer covers every case: open each row on its
own slot's occupied region plus the query columns of same-slot rows at or before
it. One row per slot gives block-diagonal decode; many rows in one slot gives a
causal prefill chunk; mixing them is dynamic-split-fuse, for free.

Idle rows are opened on their own query column only — never fully blocked, which
would produce NaN out of softmax and poison the live rows sharing the inference.

## Measured with real isolation (N independent sequences)

| slots | ms/iter | ms/token | vs seq=1 | per-slot context |
|---|---|---|---|---|
| 1 | 10.57 | 10.57 | 1.00x | 1023 |
| 4 | 25.58 | 6.40 | 1.65x | 255 |
| 8 | 27.17 | 3.40 | 3.11x | 127 |
| 16 | 28.10 | 1.76 | **6.01x** | 63 |

Correctness at every point:

- **Isolation** — slot 0 held fixed while all other slots were randomized twice:
  row 0 bit-identical, `max|delta| = 0`.
- **Teeth** — changing slot 0's own token moves row 0 by ~1.6e3, so the
  isolation result is not vacuous.
- **Equivalence** — a packed slot versus the untouched seq=1 graph running that
  same sequence alone: `max|delta| = 3.05e-05` on values of magnitude ~5806,
  cosine `1.00000000`. Packing is numerically correct, not merely leak-free.

## Sizing

Per-slot context is `(static_context - 1) / slots`, so slots and context trade
directly. The table above partitions a 1023-slot window, which is why per-slot
context shrinks as slots rise; a deployment wanting 8 x 1024 should export with
`--static-context 8193`. KV traffic then scales with the slot count, but there is
room: 8 x 29.3 MB = 234 MB still sits under the 458 MB weight stream.

## End-to-end

`--packed-slots N` on the ov-runtime engine turns a static single-stage export
into a slot-served worker: admission tokenizes into a free slot, one inference
prefills a chunk or decodes a row per ready slot, and a finished slot retires
immediately — freeing its KV region for the next admission rather than waiting
for the batch.

Validated on Llama-3.2-1B-Instruct (single-stage int4 static export), on both
devices that run the stateless static path:

**NPU** (Intel AI Boost), 4 slots:

```
packed, 4 slots   first tokens 2.67-2.85 s  all complete  8.68 s  overlap yes
baseline (off)    first tokens 0.32-7.48 s  all complete 11.17 s  overlap no
```

**CPU**, 4 slots:

```
packed, 4 slots   first tokens 2.12-2.93 s   all complete  9.32 s   overlap yes
baseline (off)    first tokens 0.57-12.53 s  all complete 18.34 s   overlap no
```

Both devices return the same text for the same prompt ("The capital of France is
Paris.", usage 42 + 15), so the packed path is device-independent as expected —
it is the same stateless static IR, only the compile target differs.

Baseline serializes on both: each request's first token lands only after the
previous one finishes (worst case 7.48 s on NPU, 12.53 s on CPU). Packed admits
all four at once and the early finishers retire while the others run on —
continuous, not batch-synchronous. Worst-case time-to-first-token improves
7.48 s -> 2.85 s on NPU (2.6x) and 12.53 s -> 2.93 s on CPU (4.3x); wall clock
1.3x and 2.0x respectively.

These end-to-end ratios at 4 slots measure SCHEDULING, and are bounded by it:
the win grows with slot count, and the weight-amortization ceiling (3.1x at 8
slots, 6.0x at 16) is the graph-level NPU property measured in the table above.
A deployment wanting that ceiling exports with more slots and a wider
`--static-context`.

Note also that packed mode skips compiling the chunked-prefill variant
entirely — a packed plan whose rows all belong to one slot IS a causal chunk —
saving a full NPU compile and a second resident weight copy.

### Multi-stage

Packing works across a pipeline. Stage 0 ships an I64 `[1, 2, S]` **plan frame**
(slot id per row, `-1` for idle; absolute position per row) ahead of the
`[1, S, hidden]` block; relay and head stages decode it, re-derive same-slot
causal order, run their own packed inference over their own per-slot rings, and
the tail replies with one token per row. A row at position 0 resets its slot —
the same in-band new-sequence signal the single-task static path already uses,
so no separate admission message is needed. Every stage must be started with the
same `--packed-slots`, since the slot count is baked into each stage's IR shape.

Validated on TinyLlama 2-stage static (CPU, both ranks on one box over
loopback): `"The capital of France is Paris."`, usage `35 + 15`; and 4
concurrent requests across the pipeline with first tokens at 2.09-2.26 s, slot 2
retiring at 5.9 s while the rest ran to 7.5 s.

### Prefix caching (`--packed-prefix N`)

Paged attention is what usually buys prefix sharing, and the NPU has none. But
sharing does not actually need paging — it needs a *mask that can open the same
columns for several rows*, which this design already has. So reserve the first
`N` columns of the KV window as a read-only **shared prefix** and let every
slot's mask open them:

```
[ shared prefix (N) ][ slot 0 region ][ slot 1 region ] ... [ slot S-1 region ]
       read-only,          private          private              private
    any slot may attend
```

The first admitted request populates it (its first `N` tokens' K/V are copied to
the front of the window, together with their token ids). Later requests are
matched by longest common prefix against those ids; a request reusing `k` tokens
starts its sequence at absolute position `k` and never prefills them again.
RoPE stays correct precisely because the cached K/V were computed at their true
absolute positions `0..k`, and attention over past KV is a read — so many rows
sharing those columns needs no copy and no reference counting.

Measured on NPU, 4 slots, four sequential requests sharing a ~96-token system
prompt (Llama-3.2-1B, 105-token prompts):

| | first request | later requests | speedup |
|---|---|---|---|
| `--packed-prefix 96` | 2.26 s | **0.38 s** | **5.95x** |
| off | 2.81 s | 2.09 s | 1.34x (warmup only) |

All four answers stayed correct (Paris / Tokyo / Rome / Madrid), which is the
check that matters: reuse must not corrupt attention.

The cost is honest and visible: the shared region is taken from the same fixed
window, so `region = (static_context - 1 - N) / slots`. Prefix capacity trades
directly against per-slot context.

Limits versus real paged prefix caching: one cache entry (not an LRU of many),
populated by whichever request arrives first; no block-level dedup between
partially-overlapping prompts beyond that single shared run; and the cached
prefix persists for the worker's lifetime rather than being evicted by pressure.

### Per-slot cancel

A disconnecting client's slot is retired and its KV region returned to the free
pool immediately, leaving the other in-flight slots untouched — the single-task
path can only drop *queued* tasks, never reclaim a running one. Observed in the
2-stage run: four disconnects released slots one at a time
(`in_flight=3, 2, 1, 0`).

Side benefit worth remembering: at ~0.21-0.29 ms marginal per row, speculative
decode verification is nearly free on the NPU.
