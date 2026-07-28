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

## Status

Shipped here: the IR variant (hardware-verified — the emitted graph compiles on
NPU and passes isolation + teeth) and the host-side primitives (unit-tested for
mask layout, region-local scatter, in-region slide, and per-slot reset).

Not yet wired: the engine execution path — slot admission/eviction in
`OvRuntimeEngine`, per-slot sampling and chunk emission, and the multi-stage wire
carrying `[1, S, hidden]` plus per-slot positions.

Side benefit worth remembering: at ~0.21-0.29 ms marginal per row, speculative
decode verification is nearly free on the NPU.
