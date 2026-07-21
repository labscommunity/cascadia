# Sharding on the Intel NPU

How `cascadia` exports and runs **sharded (pipeline-parallel) models on the
Intel NPU**. Companion to [SHARDING.md](SHARDING.md), which covers the general
`cpu-gpu` path. Code cited inline: `tools/export_shards.py` (export) and
`crates/cascadia-engine-openvino/src/runtime.rs` (runtime).

## Why the NPU needs a different shard

The OpenVINO NPU plugin rejects two things a normal cascadia shard relies on:

1. **Dynamic shapes.** A standard (`--target cpu-gpu`) shard has a dynamic
   sequence length and a growing KV-cache dim. The NPU compiler cannot compile
   any dynamic dim — the `vpux-compiler StopLocationVerifierPass: Found N
   duplicated names` failure (issue #37) that off-the-shelf dynamic exports hit.
2. **OpenVINO state variables.** The normal KV cache is an *internal* OV state
   (`ReadValue`/`Assign` nodes added by `make_stateful`). The NPU stack has no
   concept of state variables.

So an NPU shard must be **static-shape** *and* **stateless**. cascadia produces
exactly that at export time, and rebuilds the KV cache + position bookkeeping on
the host at runtime.

## Half 1 — Export: static, stateless, per-stage IRs

`cascadia shard --target npu` emits one IR per stage (the CLI forwards
`--target` / `--static-seq` / `--static-context` to the bundled exporter, which
needs `torch`, `transformers`, `nncf`, `safetensors`,
`openvino`). Per stage (`export_shards.py` sections 5-10, ~lines 1348-1538):

- **Layer split.** Stage 0 owns the embedding + first layer slice
  (`has_embed = true`); the last stage owns the final layers + LM head
  (`has_head = true`); middle stages are pure relay. Uniform by default, or
  `--layer-split`.
- **Every shape pinned static** (the NPU requirement):
  - first input (`input_ids` on the embed stage, else `hidden_states`) → seq
    dim = `static_seq` (1); relay stages also pin the otherwise-dynamic
    `hidden` dim to `config.hidden_size`.
  - `attention_mask` → length `static_context` (default 1024).
  - `past_key_values.{layer}.{key,value}` → past length = `static_context -
    static_seq`, batch = 1.
  - **batch = 1 on every input** — a dynamic batch leaves unbounded upper
    bounds the compiler rejects.
- **Stateless KV** (`export_shards.py` ~lines 1404-1424). `make_stateful` is
  skipped; the IR stays a pure function
  `(input, attention_mask, position_ids, past_key_values.*) -> (logits|hidden_states, present.*)`
  — KV in and KV out are explicit graph ports, not hidden state.
- **Post-compression static check** (~lines 1460-1495). After NNCF INT4
  weight-compression the graph is re-inferred and **every input AND output dim
  is asserted static**; a leftover dynamic dim (e.g. from an int4 decompression
  subgraph) *fails the export* rather than producing an IR that only blows up at
  NPU load (and that CPU/GPU would happily accept).
- **Weights**: INT4 channel-wise (NNCF `group_size=128`), saved FP16.
- **Metadata**: each stage gets `stage_config.json` with
  `stateful=false, target="npu", static_seq, static_context, layer_start/end,
  has_embed, has_head, num_kv_heads, head_dim, ...`; plus the shared
  `pipeline_config.json` and `tokenizer/`.

## Half 2 — Runtime: a host-side KV ring per stage

Because the IR is stateless and `seq=1`, the `ov-runtime` engine reconstructs the
KV cache and position window on the host — the `StaticKv` bounded ring
(`runtime.rs` ~lines 284-401). It holds the most-recent `valid` real tokens' K/V
left-aligned in `past_len`-slot buffers. Per decoded token:

1. `begin_token(position)` — `valid = min(position, past_len)`; the ring resets
   when `position == 0` (start of a new sequence).
2. `write_mask_bytes()` — rebuilds the fixed-length `attention_mask`: `1` for the
   `valid` real past slots (left-aligned) plus the current-token slot
   (`past_len`), `0` for the padding between.
3. Feed the ring (`past_key_values.*`) + mask + `position_ids` into the compiled
   IR and run it.
4. `absorb_layer()` — copy the new token's K/V from `present[past_len]` into the
   ring: append at `valid`, or slide the window (drop oldest) once full.

`seq=1` means there is no chunked prefill — the prompt is fed one token at a time
through this same path.

## The sharding part — keeping pipeline stages in lockstep

What makes it *sharded* NPU rather than single-device NPU (`runtime.rs`
~lines 17-28, 256-283):

- **One ring per stage**, each over its own layer range, on its own device.
- **Inter-stage wire = `hidden_states` (f16)** relayed stage k -> k+1 over TCP.
- **Position carried on the wire (the key trick).** Stateful (`cpu-gpu`) shards
  each track their own absolute-position counter locally and reset it when a
  `seq>1` prefill activation arrives, so they need no position on the wire.
  Static (NPU) shards are `seq=1`, so that prefill signal does not exist.
  Instead, **stage 0 sends the absolute `position` as its own 8-byte framed
  tensor (I64 `[1,1,1]`) immediately before each hidden activation**
  (`encode_wire_position`). Downstream stages read it (`decode_wire_position`),
  reset their ring at `position == 0`, and derive their visible-past count from
  it — keeping **every stage's ring in lockstep**. The frame is strictly
  validated: a stateful/static pipeline mismatch is a hard error, not a silently
  mis-padded position.
- **Loop close.** The last stage (`has_head`) emits logits -> sample -> the token
  is fed back to stage 0 for the next step.

## Placement and heterogeneous deployment

Stages can land on different device tiers (the point of three-tier placement,
issue #41):

1. `cascadia profile-stages --shard <dir> --devices NPU,GPU,CPU` -> per-(stage,
   device) latency + memory + **op-support**, writing `placement_profile.json`.
   (Also the simplest "does each stage run on the NPU?" check: a stage with an
   NPU latency compiled and ran there.)
2. `cascadia place` -> ILP over that cost table -> `placement.json`.
3. `cascadia run-placement` -> one worker per stage, each pinned to its assigned
   device -> a heterogeneous pipeline (e.g. stage 0 on the iGPU, stage 1 on the
   NPU).

## Constraints

- **Context fixed at export time** (`--static-context`, default 1024); usable
  past = `context - 1`. Re-export for a longer window.
- **`seq=1` only** — one token per forward pass; no chunked prefill yet.
- **FP16 KV** on the ring and the wire; `--default-dtype` must be `fp16` for
  `--target npu`.
- **batch = 1** (single sequence).
- Per-token host overhead is the mask rewrite + ring copies — small relative to
  the NPU forward pass.

## Reproduce

```bash
# 1. Export a 2-stage static NPU shard:
cascadia shard \
  --model unsloth/Llama-3.2-1B-Instruct \
  --output-dir ./llama1b-npu-2stage \
  --num-stages 2 --quantization int4 \
  --target npu --static-seq 1 --static-context 1024

# 2. Profile each stage on each device (latency + memory + op-support):
cascadia profile-stages --shard ./llama1b-npu-2stage --devices NPU,GPU,CPU

# 3. (optional) Solve + launch a heterogeneous pipeline:
cascadia place --profile placement_profile.json --output placement.json
cascadia run-placement --shard ./llama1b-npu-2stage --placement placement.json
```

A 2-stage Llama-3.2-1B NPU shard has been run end-to-end on a Meteor Lake AI PC
with both stages compiling and executing on the NPU.

## Code map

| Concern | Location |
|---|---|
| NPU static-shape export + stateless KV | `tools/export_shards.py` sections 5-10 (~1348-1538) |
| Static-shape validation after compression | `tools/export_shards.py` (~1460-1495) |
| Host-side KV ring (`StaticKv`) | `crates/cascadia-engine-openvino/src/runtime.rs` (~284-401) |
| Wire format + position lockstep | `runtime.rs` (~17-28; `encode/decode_wire_position` ~256-283) |
| Per-stage profiler / placement / launch | `cascadia profile-stages`, `cascadia place`, `cascadia run-placement` |
