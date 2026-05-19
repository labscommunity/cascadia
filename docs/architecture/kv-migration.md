# KV state migration between pipeline ranks

**Status:** skeleton (wire frame + Runner API only). End-to-end driver
not yet wired.

**Track:** sparse-MoE pipeline-parallel (PR #10 follow-up).

## Goal

Move a slab of per-layer KV state from one rank to another **mid-decode**,
without restarting the generation.

Two concrete use cases motivate this:

1. **Failover.** Rank A dies mid-generation. Rank B takes over its
   layer range, with the KV state intact, and the rank-0 driver
   re-points its downstream socket. The user sees a transient pause,
   not a dropped request.
2. **Hot rebalance.** The first 30 decode steps reveal that rank 0 is
   spending 2x longer per token than rank 1 (e.g. its iGPU is
   thermally throttled). The orchestrator shifts five layers from
   rank 0 to rank 1 — KV slabs for those layers migrate over TCP, the
   layer-range bookkeeping on both ranks updates, and decode resumes
   at the next token boundary.

Today this is impossible: each rank's KV state lives in `LayerState`
buffers local to its `Runner`. Failover requires `Reset` + re-prefill;
rebalance requires a full restart.

## What this PR ships

**(B) skeleton + design doc.** Specifically:

| Component | File | What |
|---|---|---|
| New frame kind | `crates/tahoma-engine-sparse-moe/src/dist.rs` | `FrameKind::KvMigration` (0x53_4D_45_30), wire helpers `send_kv_migration` / `recv_kv_migration_body_{server,client}` |
| Runner API | `crates/tahoma-engine-sparse-moe/src/runner.rs` | `extract_kv_slab(layer_start, layer_end) -> Vec<u8>`, `install_kv_slab(layer_start, layer_end, &[u8]) -> usize` |
| Skeleton handler | `crates/tahoma-engine-sparse-moe/src/engine.rs` | Worker accepts `KvMigration` frames and installs them (no ACK, no quiesce — caller is responsible) |
| Tests | `crates/tahoma-engine-sparse-moe/tests/kv_migration_wire.rs` | Wire round-trips, multi-layer sequences, length-mismatch rejection |
|  | (inside `runner.rs`) | Pure-Rust install round-trip without model artifacts |

What's **NOT** in this PR — the blockers section below covers each:

- Atomic-swap orchestrator (when does the migration "commit"?).
- Pause-token protocol (how do we stop the rank-0 driver from issuing
  new `Forward` frames during the swap?).
- Recovery on partial install (sender keeps its copy; receiver
  rejects).
- Layer-0 migration (rank 0 currently owns the dense layer 0 plus its
  KV cache; not movable in v1).
- Slot-chunked KvMigration variant for `past_seq_len > ~2k` (per-layer
  body would exceed `MAX_TENSOR_BYTES`).
- CLI / API endpoint to *trigger* a migration.

## Wire format

```
[4B BE FrameKind::KvMigration = 0x53_4D_45_30]
[20B per-layer header:
   4B BE u32 lid           # 1-based MoE layer id, matching manifest
   4B BE u32 past_seq_len  # populated KV slots (== generation step count - 1 on rank N)
   4B BE u32 num_heads     # K2.6 = 64
   4B BE u32 qk_head_dim   # K2.6 = 192
   4B BE u32 v_head_dim    # K2.6 = 128
]
[I8 tensor shape=[1, 1, N], where N = num_heads * past_seq_len * (qk_head_dim + v_head_dim) * 4
   body = K block (num_heads * past_seq_len * qk_head_dim * f32 LE)
        | V block (num_heads * past_seq_len * v_head_dim  * f32 LE)
]
```

The I8 carrier tensor uses the existing `tahoma-transport` length-
prefixed wire (`recv_tensor`), so we get its 256 MiB cap, shape-vs-
byte sanity check, and 60 s read timeout for free.

**One frame = one layer.** Multi-layer migration = N consecutive
frames. This keeps each frame well under the 256 MiB tensor cap for
realistic context lengths:

| past_seq_len | per-layer body | inside 256 MiB cap? |
|---|---|---|
| 256  | 20 MiB  | yes |
| 1024 | 82 MiB  | yes |
| 2048 | 164 MiB | yes |
| 4096 | 328 MiB | **no** — needs slot chunking (see blockers) |

## Runner API

```rust
impl Runner {
    /// Read-only snapshot. Caller must serialize against decode.
    pub fn extract_kv_slab(
        &self,
        layer_start: u32,
        layer_end: u32,
    ) -> Result<Vec<u8>, RunnerError>;

    /// Overwrite owned layers in the range. Caller must quiesce.
    /// Returns number of layers installed.
    pub fn install_kv_slab(
        &mut self,
        layer_start: u32,
        layer_end: u32,
        kv_bytes: &[u8],
    ) -> Result<usize, RunnerError>;

    /// Validates pre-install consistency from the engine layer.
    pub fn past_seq_len_for(&self, lid: u32) -> Option<usize>;
}
```

The extract output and install input share a layout (see wire format
above), so a roundtrip is `install_kv_slab(install_kv_slab_args,
&extract_kv_slab(extract_args)?)`. The `KvMigrationLayer` wire payload
exposes `into_install_slab()` to assemble the same bytes from a single
received frame, so the worker handler can call `install_kv_slab` with
a single-layer range without re-encoding.

## Blockers (why this is (B) and not a full implementation)

### 1. Atomicity & consistency

KV state at layer L is only meaningful *relative to* the token
sequence the upstream ranks have already pushed through. Concretely:
the layer-L K/V at slot `s` is what attention produced when token `s`
arrived with the hidden state that ranks 0..L produced. If you ship
layer L's slab from rank A to rank B but rank B's other layers were
populated by a *different* generation, the cache is incoherent and the
model will produce garbage.

The migration must therefore either:

- **(a)** move *every* layer the receiver needs, in lock-step (full
  rank takeover), OR
- **(b)** only re-balance layers within an existing decode, ensuring
  the rank-0 driver keeps issuing Forward frames in the same order
  through whichever rank now owns each layer.

The skeleton supports per-layer slabs but **does not** enforce either
invariant. The orchestrator (not in this PR) must.

### 2. In-flight requests during the swap

The current pipeline is fully synchronous: rank 0's
`forward_one_token_first` blocks on the round-trip downstream. If a
`KvMigration` frame arrives while a worker is in the middle of
processing a `Forward`, the worker would have to either:

- finish the in-flight `Forward` first, then install (and re-route the
  TOKEN response somewhere — possibly stale), OR
- abort the in-flight `Forward`, install, then NAK the original
  request.

Neither is implemented. The skeleton's worker handler only services
frames serially (`handle_one_frame` is called from `step_worker` one
at a time), so de-facto we're picking option (a) — but there's no
ACK back to the *sender* of the `Forward`, so the rank-0 driver sees
no signal that the response it gets is from the post-migration state.

**Proposed protocol** (not in this PR):

```
1. orchestrator -> rank 0   : SWAP_PREPARE { layers_to_move }
2. rank 0       -> rank A   : QUIESCE { layers }
3. rank A       -> rank 0   : QUIESCE_ACK { last_forward_consumed }
4. rank 0       -> rank A   : KV_MIGRATION_REQUEST { layers, destination }
5. rank A       -> rank B   : KvMigration frame(s)
6. rank B       -> rank A   : KV_MIGRATION_ACK
7. rank A       -> rank 0   : KV_MIGRATION_DONE
8. rank 0 updates routing table; next Forward uses new ranges.
```

Three new frame kinds (`Quiesce`, `QuiesceAck`, `KvMigrationAck`) and
a rank-0 orchestrator are required. Estimated +400-600 LOC.

### 3. Layer-0 migration

Rank 0 currently owns the dense layer 0 *and* its KV cache (the
embed_tokens table + post-attention dense MLP). The
`Layer0State` is structurally similar to `LayerState` but has a
different field layout (embed mmap, no `lid`, no `Int4Shell`), and the
"layer 0 is implicit at rank 0" assumption is baked into
`SparseMoEBuilderConfig::is_first` and `forward_layer0_step`.

For v1 we **explicitly reject** layer-0 migration in
`extract_kv_slab` / `install_kv_slab` (`layer_start == 0` returns
`RunnerError::Internal`). Failover of rank 0 is therefore not
supported in this skeleton.

To unblock: extract the embed pin + layer-0 weights into a movable
artifact, add a `KvMigration { layer == 0 }` sentinel that carries
both KV state and embed/weight handles, and teach
`SparseMoEBuilderConfig` to defer the "is_first" decision until after
runtime configuration.

### 4. Long context > MAX_TENSOR_BYTES

At `past_seq_len > ~2k`, a single layer's KV exceeds the 256 MiB
transport cap. `send_kv_migration` will reject the frame at that
point. To unblock: add a slot-chunked variant
(`KvMigrationChunk { slot_start, slot_end }`) so a layer's KV ships
across N frames. Out of scope for v1 — K2.6 evals run at ≤1k context.

### 5. Cross-architecture compatibility

The wire format hard-codes `num_heads`, `qk_head_dim`, `v_head_dim` in
the per-layer header. Receivers cross-check against compile-time
constants from `tahoma-int4-gemm` (`NUM_HEADS`, `QK_HEAD_DIM`,
`V_HEAD_DIM` — K2.6 values). A different architecture (Llama, Qwen)
would trip the shape mismatch in `install_kv_slab` even if the body
were structurally similar.

This is acceptable for v1 (the sparse-MoE engine is K2.6-only). When
the engine generalizes (Phase 13?), the shape sanity check becomes a
soft compatibility hint rather than a hard error.

### 6. Authentication / authorization

Today any peer that can complete the TCP handshake on the activation
port can issue a `KvMigration` frame. In a single-LAN AI PC fleet
this is fine — the broader threat model has us on a trusted Tahoma
mesh — but a hardened deployment (cascadia-fleet) will want the
orchestrator to gate migration with a signed token. Out of scope for
the OSS skeleton; productization in cascadia-fleet.

## Testing matrix

The skeleton ships unit + integration tests but **not** end-to-end
quality tests (no model artifacts on CI). What's covered:

| Test | File | What |
|---|---|---|
| `install_layer_kv_basic_round_trip` | runner.rs `tests` | f32 K/V stamps round-trip through `install_layer_kv` |
| `install_layer_kv_rejects_wrong_size` | runner.rs `tests` | size mismatch yields `RunnerError::Internal` |
| `install_layer_kv_grows_capacity_when_needed` | runner.rs `tests` | install with past_seq_len > initial cap triggers geometric grow |
| `kv_migration_frame_round_trips_tiny_layer` | tests/kv_migration_wire.rs | full Frame → tensor → re-decode loop |
| `kv_migration_frame_round_trips_zero_past_seq_len` | tests/kv_migration_wire.rs | empty-body edge case |
| `kv_migration_two_layers_in_sequence` | tests/kv_migration_wire.rs | back-to-back layer frames don't bleed state |
| `kv_migration_rejects_wrong_body_length` | tests/kv_migration_wire.rs | sender bails on body / header mismatch before any bytes go on wire |
| `kv_migration_client_side_recv` | tests/kv_migration_wire.rs | symmetric upstream-bound path works |

End-to-end test (deferred to v2): on the local AI PC fleet
(`beta`/`charlie`), shard K2.6 across 2 ranks, generate 10 tokens,
then trigger a manual migration of 5 layers from rank 1 to rank 0 and
verify the next 10 tokens decode to the same logits as the
single-rank baseline. Requires the orchestrator from blocker #2.

## Next steps (rough order)

1. **Pause-token protocol** (blocker #2) — adds `Quiesce`,
   `QuiesceAck`, `KvMigrationAck` frames; rank-0 orchestrator.
2. **Rebalance trigger from rank 0** — observed per-rank latency drift
   from the existing transport stats → SWAP_PREPARE.
3. **End-to-end test on local fleet** — small K2.6 quant, 2 ranks,
   manually trigger migration via a CLI flag, compare logits.
4. **Failover (rank dies)** — needs `tahoma-discovery` to surface
   rank-loss events; the orchestrator then promotes a hot-spare rank
   already loaded with the same layer ranges.
5. **Layer-0 migration** (blocker #3) — required for rank-0 failover.
6. **Slot-chunked variant** (blocker #4) — required for >2k context.
