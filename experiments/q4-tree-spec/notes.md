# Q4 — Tree-spec implementation

User asked for tree-spec. Built it end-to-end (v6 export with 4D mask, Rust
tree mask + verifier + KV pruning). The implementation is **algorithmically
correct** but does **not improve throughput** for this hardware/model setup
because the draft model (Llama 3.2 1B) is the bottleneck and tree-spec
roughly doubles draft work.

## Path

1. **v6 export** — copied `export_cached_shards_v5.py` to v6, swapped 2D
   `attention_mask` (i64 pad mask) for 4D additive f16 mask
   `[1, 1, q, total]`. `_build_causal_mask` becomes a passthrough; the
   driver builds the topology-aware mask client-side. Re-exported Llama 3.1
   8B INT4 as `shards_2stage_v6_treemask` (~4.1 GB, 2 stages × 16 layers).

2. **Wire protocol** — added `FrameKind::ForwardV6 = 5`. v6 frame carries
   f16 4D mask + i64 position_ids + f16 hidden_states. Worker detects v6
   from `stage_config.json` `export_version` and routes to
   `handle_forward_v6`.

3. **Driver-side** — `DistributedMaskedReq.is_v6` flag toggles between
   chain mask (v5/v6) and tree mask (v6 only). `feed`, `feed_send_async`,
   `feed_tree_send_async` all branch on `is_v6`. `feed_tree_send_async`
   takes (input_ids, position_ids, parents) — caller builds the tree.

4. **Tree topology** — `tree_preset = 1` is "width-2 at root":
   ```
        prev_correction
        /           \
       L_0          R_0  (= top-1, top-2 of draft at root)
       |             |
       L_1          R_1  (= top-1 chain from each)
       ...           ...
       L_{k-1}      R_{k-1}
   ```
   Tree size = 2K + 1.

5. **Verifier** — walks both chains greedily (target_argmax must match
   draft token at each depth), picks the longer accepted path.
   `confirm_tree_path` invalidates the non-winning entries via
   `valid_mask` so subsequent rounds attend only to the chosen path.

## Bug found and fixed

Initial implementation gave **accept = 0.07** on chain-via-tree-code
(should match chain spec's 0.51). Off-by-one in `draft.rewind(k)` —
chain spec rewinds `k` because its build loop does `k-1` feeds plus
1 speculative feed; tree code's build loop only does `k-1` feeds (no
speculative). The over-rewind invalidated 1 PAST entry, corrupting
draft context. Fix: rewind exactly the number of entries added
(`left_feeds = k - 1` for left, `right_feeds = k` for right).

After fix: tree-preset 99 (chain-only via tree code) hits accept=0.51,
matching chain spec. ✓

## Final results — width-2-root tree (preset 1)

| Config | Tokens | Steps | tok/step | accept | tok/s |
|--------|-------:|------:|---------:|-------:|------:|
| Chain K=5 (v6, baseline) | 256 | 72 | 3.55 | 0.51 | **18.46** |
| Tree-via-chain K=5 (preset 99) | 256 | 72 | 3.55 | 0.51 | 15.66 |
| Tree K=3 (preset 1) | 256 | 82 | 3.12 | 0.35 | 12.77 |
| Tree K=5 (preset 1) | 256 | 67 | 3.82 | 0.28 | 7.81 |
| Tree K=5 (preset 1) | 1024 | 227 | 4.51 | 0.35 | 9.41 |

**Tree K=5 gives +8% tokens-per-step over chain K=5 (3.82 vs 3.55), but
-58% throughput (7.81 vs 18.46 tok/s).**

## Why tree-spec loses for this setup

Per-step time breakdown (K=5, 256-tok):

| | Chain | Tree |
|---|------:|-----:|
| draft_ms / step | 128 | 248 (1.94×) |
| target_alpha_infer_ms / step | 28 | 32 (1.14×) |
| target_wire_ms / step | 34 | 205 (6.0×) |

Two cost multipliers:

1. **Draft cost ~doubles**. We generate two K-chains sequentially. A
   1B Llama draft can only generate one token at a time, so
   per-position cost is the same and total cost scales with token
   count. (EAGLE-2 dodges this with a single-layer draft head that
   produces many candidates per forward — we don't have one.)

2. **Wire cost grows much faster than data**. 4D additive f16 mask is
   `query_len × total_keys × 2` bytes. Tree's `query_len = 11` vs
   chain's `query_len = 6` is 1.83× data, but wire time grew 6×.
   Likely a combination of TCP framing, charlie's stage_1 attention
   compute scaling super-linearly with seq_len, and OV mask handling
   overhead. Did not fully diagnose.

## Algorithmic note: tree-spec helps low-accept regimes

For a width-2-root tree, the win comes ONLY when the draft top-1 at
position 0 is rejected AND the top-2 alt is accepted. With our 0.88
factual / 0.51 short-prompt accept rates, the draft top-1 is already
very good, so the alt rarely fires usefully:

  E[extra tokens from tree] ≈ P(L_0 reject) × P(R_0 == target_argmax | L_0 reject) × E[R_chain]
                            ≈ 0.12 × 0.5 × 4 ≈ 0.24 tokens/step (factual long-gen)

That's a 5–8% throughput ceiling at our accept rate, which the doubled
draft cost obliterates.

Tree-spec tends to win in the literature for setups with:
- Single-layer draft heads (e.g. EAGLE) that draft many candidates
  per forward — draft cost stays flat as tree grows.
- Lower base accept rates (~0.5) where wider trees recover meaningful
  acceptance.
- Batched serving (multiple sequences amortize per-tree overhead).

We have none of those: full transformer draft, 0.5–0.88 accept,
single-user sequential.

## What would make tree-spec win for us

1. **Re-export the draft as v6** and tree-feed it in one batched
   forward. Halves draft cost. Estimated work: ~2 hours (download 1B
   Llama raw, run v6 export, plumb 4D mask through `MaskedReq`).
   Likely gets tree to ~13–15 tok/s — closer to chain but probably
   still under.

2. **EAGLE-2 dynamic tree pruning** with a draft head. Substantial
   research effort — train an EAGLE head for Llama 8B INT4 on Intel
   GPU. Multi-day work.

3. **Batched serving mode** so per-tree fixed costs are amortized.
   Out of scope for the single-user serving mission.

## Code paths

- v6 export: `rainier/scripts/export_cached_shards_v6_treemask.py`
- 4D mask helpers: `tahoma-engine-openvino/src/dist_spec.rs`
  - `build_chain_mask_f16` — chain spec mask in additive f16
  - `build_tree_mask_f16` — tree spec mask using parents array
- ForwardV6 frame: `dist_spec.rs` `FrameKind::ForwardV6`
- Driver: `feed_send_async` (chain) + `feed_tree_send_async` (tree)
- Worker: `handle_forward_v6` (single fn handles both chain and tree)
- Driver loop: `spec_decode_greedy_tree` (tree_preset 1)
- Tree-preset 99 = chain via tree code path (debug / sanity)

## Q5 — option 1 attempted: parallel draft via v6 4D mask

Re-exported Llama 3.2 1B Instruct as a v6 IR (single stage, embed + 16
layers + tied lm_head). Two patches were needed:

- **Tied embeddings**: 1B/3B Llamas share `embed_tokens.weight` with
  `lm_head.weight`. The v5 export script only handled separate weights;
  added detection + reuse. (Works for INT8/FP16 but the INT4 path hangs
  in NNCF compression for tied weights — likely a known NNCF interaction.
  Used FP16 quantization instead, ~2.5 GB vs ~700 MB INT4. Trade-off:
  bigger memory footprint + slower per-token compute, but the v6 mask
  path works.)
- **`MaskedReq.feed_pair` + `build_pair_mask_f16`**: new draft API that
  processes `[L_i, R_i]` in a single batched forward, with a per-query
  mask that isolates each chain from its sibling. Requires v6 draft.
- **`MaskedReq.invalidate_recent`**: rewind without touching `logical_pos`
  (needed because `feed_pair` writes 2 cache entries per logical-position
  step, so the standard `rewind(k)` would corrupt position tracking).
- **`spec_decode_greedy_tree` preset 2**: uses `feed_pair` instead of
  sequentially building LEFT then RIGHT. K-1 batched 2-token feeds per
  round instead of 2K-1 single-token feeds.

### Result

| Config | tok/s | tokens/step | accept | draft_ms/step |
|--------|------:|------------:|-------:|--------------:|
| Chain K=5 (v6, baseline) | **18.46** | 3.55 | 0.51 | 128 |
| Tree K=5 sequential (preset 1, INT4 1B draft) | 7.81 | 3.82 | 0.28 | 248 |
| **Tree K=5 parallel (preset 2, FP16 1B v6 draft)** | **11.20** | 3.82 | 0.28 | 186 |

Parallel draft saves **25% on draft cost** (248 → 186 ms/step), giving
**+43% throughput vs sequential tree** (7.81 → 11.20 tok/s). Generation
is correct (verified via round traces — chain spec K=5 final corrections
match).

### Why it still loses to chain spec

The wire/charlie cost is the dominant bottleneck:

| | Chain K=5 (v6) | Tree K=5 parallel (preset 2) |
|---|--------------:|----------------------------:|
| target_alpha_infer_ms / step | 28 | 33 |
| target_wire_ms / step | 34 | 118 (3.5×) |
| draft_ms / step | 128 | 186 (1.45×) |
| total ms / step | 192 | 341 |

Tree's wire jumps **3.5×** for only ~2× more data. Looking at the
rough math: tree sends `[1, 11, total]` mask vs chain's `[1, 6, total]`
(1.83× bigger), tree hidden is `[1, 11, 4096]` vs `[1, 6, 4096]`
(1.83×), tree logits are `[1, 11, 128k]` vs `[1, 6, 128k]` (1.83×).
At TB4 ~10 Gbps the extra ~1.3 MB/round should add ~1 ms; the actual
~80 ms extra per step is mostly charlie's stage_1 attention compute
plus OV's per-call overhead growing super-linearly with seq_len.

Net per-step time: tree 341 ms vs chain 192 ms. Tree's +8% tokens-per-step
(3.82 vs 3.55) is dwarfed by the +78% step time.

### Two further wins available, both deferred

1. **Re-export 1B draft as INT4 v6** — would cut draft cost ~3× (memory
   bandwidth-bound), giving tree a draft cost of ~60 ms/step. Per-step
   becomes max(60, 152) = 152 ms ≈ chain step time. Throughput parity
   with chain (~18 tok/s), still not a win.
   - Blocked on the NNCF tied-embeddings hang. Would need to either patch
     NNCF, fork the script to skip lm_head compression, or train an EAGLE
     head whose lm_head isn't tied.

2. **Reduce wire/charlie scaling** — the 3.5× wire growth for 1.83×
   data is the real killer. Possible mitigations:
   - Bit-packed boolean mask instead of f16 additive (16× smaller).
   - Per-stage PA — would need the OV plugin OOM workaround we deferred
     in Q2.1.
   - Fixed mask-shape preallocation in OV to avoid per-call shape rebinding.

   These are multi-day investigations. Not pursued.

## Recommendation

Do **not** enable `--spec-tree 1` for production benchmarks on this
configuration. Chain spec K=5 with v5 IR remains the best at 29.47
tok/s factual long-gen.

Keep the tree-spec code: it's ready to use when (a) we add a draft
head, or (b) we re-export the draft as v6 to enable parallel chain
generation.
