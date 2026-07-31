//! Packed multi-slot decode ("seq-as-batch") for the stateless static-shape
//! (NPU) path — continuous batching on a device that refuses batch > 1.
//!
//! The NPU compiler rejects a batch dimension outright (`ConvertBatchedLayerTo1N`
//! fails to legalize), but it compiles seq > 1 happily — that is what the
//! chunked-prefill variant already relies on. So we pack N independent requests
//! into the SEQUENCE dimension of one inference and isolate them with a
//! block-diagonal attention mask. Batch stays 1; the rejected pass never runs.
//!
//! Layout. The IR's `past_len` KV window is partitioned into `slots` equal
//! regions of `region = past_len / slots` slots; slot `s` owns
//! `[s*region, (s+1)*region)` in every layer's ring, in every head. A packed
//! inference presents `packed_seq` query rows; a [`PackedPlan`] says which slot
//! (if any) each row belongs to. The mask then opens, for each row, exactly its
//! own slot's occupied past region plus the query columns of same-slot rows at
//! or before it — which yields block-diagonal masking for decode (one row per
//! slot) and causal masking for a prefill chunk (many rows, one slot), from one
//! writer. Mixed decode+prefill steps fall out for free.
//!
//! Measured on Lunar Lake (Qwen2.5-1.5B stage 0, OV 2026.2.1): row isolation is
//! bit-exact, a packed slot matches the same sequence run alone to fp16 rounding
//! noise, and throughput is 3.1x at 8 slots / 6.0x at 16 vs seq=1 — because
//! weights outweigh KV traffic 16:1, so the per-iteration weight stream amortizes
//! across slots. See docs/perf/NPU_PACKED_SLOTS.md.

use cascadia_ov_genai_shim::DType as ShimDType;

/// Additive-mask "blocked" value. f16's most-negative finite (-65504) and an
/// f32 equivalent — finite rather than -inf so a fully-blocked row can never
/// produce NaN out of softmax.
const NEG_F16_BITS: u16 = 0xFBFF;
const NEG_F32: f32 = -3.0e38;

/// Leading KV columns of each slot's region that eviction must never drop —
/// attention sinks. Transformer heads park a large share of their softmax mass
/// on the first few tokens of a sequence; slide those out and the mass is
/// forced onto ordinary content, which collapses generation into degenerate
/// repetition rather than truncating it gracefully. Four is the width the
/// StreamingLLM result found sufficient.
const PACKED_SINK: usize = 4;

/// Which slot a query row belongs to, and its ordinal among that slot's rows in
/// this same inference (0 for decode; 0..n for a prefill chunk). `order` drives
/// causal masking within a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedRow {
    pub slot: usize,
    pub order: usize,
}

/// Row assignment for one packed inference. `rows.len() == packed_seq`; `None`
/// marks an idle row (no request occupies it this step).
#[derive(Clone, Debug, Default)]
pub struct PackedPlan {
    pub rows: Vec<Option<PackedRow>>,
}

impl PackedPlan {
    pub fn idle(packed_seq: usize) -> Self {
        Self {
            rows: vec![None; packed_seq],
        }
    }

    /// One row per listed slot, in order — the decode shape.
    pub fn decode(packed_seq: usize, slots: &[usize]) -> Self {
        let mut rows = vec![None; packed_seq];
        for (r, &slot) in slots.iter().take(packed_seq).enumerate() {
            rows[r] = Some(PackedRow { slot, order: 0 });
        }
        Self { rows }
    }

    /// `n` consecutive rows all belonging to `slot` — the prefill-chunk shape.
    pub fn chunk(packed_seq: usize, slot: usize, n: usize) -> Self {
        let mut rows = vec![None; packed_seq];
        for (order, row) in rows.iter_mut().take(n.min(packed_seq)).enumerate() {
            *row = Some(PackedRow { slot, order });
        }
        Self { rows }
    }

    pub fn active_rows(&self) -> impl Iterator<Item = (usize, PackedRow)> + '_ {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(r, pr)| pr.map(|pr| (r, pr)))
    }

    pub fn is_empty(&self) -> bool {
        self.rows.iter().all(|r| r.is_none())
    }
}

/// Host-side KV for `slots` independent sequences sharing one static IR window.
pub struct PackedKv {
    pub slots: usize,
    /// Read-only shared prefix at the FRONT of the window, `[0, prefix_capacity)`.
    /// Any number of slots may open these columns at once — attention over past
    /// KV is a read, so sharing needs no copy and no paging. 0 disables it.
    pub prefix_capacity: usize,
    /// How much of the shared prefix currently holds real K/V.
    pub prefix_valid: usize,
    /// Past KV slots owned by each packed slot (after the shared prefix).
    pub region: usize,
    /// Leading columns of each slot's region that the slide never evicts (see
    /// [`PACKED_SINK`]). Capped at half the region so pinning can never crowd
    /// out the sliding window itself.
    sink: usize,
    pub past_len: usize,
    /// Query rows per inference (the IR's static seq length).
    pub packed_seq: usize,
    /// `present.*` length = `past_len + packed_seq`.
    pub context: usize,
    kv_heads: usize,
    head_dim: usize,
    elem_bytes: usize,
    key_buf: Vec<Vec<u8>>,
    val_buf: Vec<Vec<u8>>,
    /// Absolute tokens consumed by each slot's sequence (drives RoPE position).
    /// A slot reusing `k` shared-prefix tokens starts at `pos = k`, because the
    /// cached K/V were computed at absolute positions `0..k` — which is exactly
    /// what makes reuse RoPE-correct.
    pos: Vec<usize>,
    /// Shared-prefix columns each slot may read (its reuse length).
    shared_use: Vec<usize>,
}

impl PackedKv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slots: usize,
        past_len: usize,
        packed_seq: usize,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_dtype: ShimDType,
        prefix_capacity: usize,
    ) -> Self {
        let elem_bytes = match kv_dtype {
            ShimDType::F32 | ShimDType::I32 => 4,
            ShimDType::I64 => 8,
            _ => 2,
        };
        let per_layer = kv_heads * past_len * head_dim * elem_bytes;
        let prefix_capacity = prefix_capacity.min(past_len.saturating_sub(slots));
        let region = (past_len - prefix_capacity) / slots;
        Self {
            slots,
            prefix_capacity,
            prefix_valid: 0,
            region,
            sink: PACKED_SINK.min(region / 2),
            past_len,
            packed_seq,
            context: past_len + packed_seq,
            kv_heads,
            head_dim,
            elem_bytes,
            key_buf: vec![vec![0u8; per_layer]; layers],
            val_buf: vec![vec![0u8; per_layer]; layers],
            pos: vec![0; slots],
            shared_use: vec![0; slots],
        }
    }

    /// First KV column owned by `s` (its region sits after the shared prefix).
    fn region_start(&self, s: usize) -> usize {
        self.prefix_capacity + s * self.region
    }

    /// Shared-prefix columns slot `s` is reusing.
    pub fn shared_use(&self, s: usize) -> usize {
        self.shared_use[s]
    }

    /// Real past tokens currently visible to slot `s` (its region is a bounded
    /// window, so this saturates at `region`).
    pub fn valid(&self, s: usize) -> usize {
        self.pos[s]
            .saturating_sub(self.shared_use[s])
            .min(self.region)
    }

    /// Absolute position of the next token for slot `s` (RoPE input).
    pub fn position(&self, s: usize) -> usize {
        self.pos[s]
    }

    /// Clear one slot without disturbing its neighbours — an admitted request
    /// starts from an empty region.
    pub fn reset_slot(&mut self, s: usize) {
        self.reset_slot_reusing(s, 0);
    }

    /// Clear slot `s` and start it reusing `shared` cached prefix tokens: its
    /// sequence then begins at absolute position `shared`, and its mask opens
    /// the first `shared` shared-prefix columns.
    pub fn reset_slot_reusing(&mut self, s: usize, shared: usize) {
        let shared = shared.min(self.prefix_valid);
        let (region, kv_heads, past_len) = (self.region, self.kv_heads, self.past_len);
        let slot_bytes = self.head_dim * self.elem_bytes;
        let buf_row = past_len * slot_bytes;
        let start = self.region_start(s) * slot_bytes;
        for buf in self.key_buf.iter_mut().chain(self.val_buf.iter_mut()) {
            for h in 0..kv_heads {
                let base = h * buf_row + start;
                buf[base..base + region * slot_bytes].fill(0);
            }
        }
        self.pos[s] = shared;
        self.shared_use[s] = shared;
    }

    /// Copy the first `n` tokens of `src` slot's region into the shared prefix,
    /// making them reusable by every slot. Called once, after the request that
    /// populates the cache has prefilled that far.
    pub fn populate_prefix_from(&mut self, src: usize, n: usize) {
        let n = n.min(self.prefix_capacity).min(self.valid(src));
        if n == 0 || self.prefix_valid > 0 {
            return;
        }
        let slot_bytes = self.head_dim * self.elem_bytes;
        let buf_row = self.past_len * slot_bytes;
        let src_start = self.region_start(src) * slot_bytes;
        let kv_heads = self.kv_heads;
        for buf in self.key_buf.iter_mut().chain(self.val_buf.iter_mut()) {
            for h in 0..kv_heads {
                let base = h * buf_row;
                let from = base + src_start;
                let to = base;
                // Non-overlapping by construction: the shared region is
                // [0, prefix_capacity) and every slot region starts at or after
                // prefix_capacity.
                buf.copy_within(from..from + n * slot_bytes, to);
            }
        }
        self.prefix_valid = n;
        // The populating slot keeps its own copy; it does not retroactively
        // "reuse" what it already computed.
    }

    pub fn key_bytes(&self, li: usize) -> &[u8] {
        &self.key_buf[li]
    }

    pub fn val_bytes(&self, li: usize) -> &[u8] {
        &self.val_buf[li]
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn present_layer_bytes(&self) -> usize {
        self.kv_heads * self.context * self.head_dim * self.elem_bytes
    }

    /// Advance a slot's absolute position after its row(s) were absorbed.
    pub fn advance(&mut self, s: usize, by: usize) {
        self.pos[s] += by;
    }

    /// Absorb query row `row`'s K/V (which sits at `present` index
    /// `past_len + row`) into `slot`'s region, appending at `at` or sliding the
    /// region's oldest entry out when it is full. `at` is the caller's running
    /// occupancy for that slot within this step, so a multi-row prefill chunk
    /// lands consecutively.
    ///
    /// The slide is strictly bounded to `[slot*region, (slot+1)*region)` — a
    /// neighbouring slot's KV is never read or written.
    pub fn absorb_row(
        &mut self,
        li: usize,
        is_value: bool,
        present: &[u8],
        row: usize,
        slot: usize,
        at: usize,
    ) {
        let slot_bytes = self.head_dim * self.elem_bytes;
        let present_row = self.context * slot_bytes;
        let buf_row = self.past_len * slot_bytes;
        let region = self.region;
        let sink = self.sink;
        let kv_heads = self.kv_heads;
        let src_slot = self.past_len + row;
        let full = at >= region;
        let region_start = self.region_start(slot) * slot_bytes;
        let buf: &mut [u8] = if is_value {
            &mut self.val_buf[li]
        } else {
            &mut self.key_buf[li]
        };
        for h in 0..kv_heads {
            let src = h * present_row + src_slot * slot_bytes;
            let new = &present[src..src + slot_bytes];
            let base = h * buf_row + region_start;
            if full {
                // Drop this slot's oldest EVICTABLE entry — the leading `sink`
                // columns are pinned, so the shift starts past them. The copy
                // stays inside the region.
                buf.copy_within(
                    base + (sink + 1) * slot_bytes..base + region * slot_bytes,
                    base + sink * slot_bytes,
                );
                let dst = base + (region - 1) * slot_bytes;
                buf[dst..dst + slot_bytes].copy_from_slice(new);
            } else {
                let dst = base + at * slot_bytes;
                buf[dst..dst + slot_bytes].copy_from_slice(new);
            }
        }
    }

    /// Write the additive attention mask for `plan` into `buf` as a
    /// `[1, 1, packed_seq, context]` tensor of `dtype`.
    ///
    /// Row `r` is opened on exactly: its slot's occupied past region, and the
    /// query columns of same-slot rows with `order <= its order`. Everything
    /// else is blocked. An idle row is opened on its own query column only —
    /// never fully blocked, which would make softmax produce NaN and poison the
    /// whole inference including live rows.
    pub fn fill_mask(&self, plan: &PackedPlan, buf: &mut Vec<u8>, dtype: ShimDType) {
        let elem = match dtype {
            ShimDType::F32 => 4,
            _ => 2,
        };
        let (s, ctx, region, past_len) =
            (self.packed_seq, self.context, self.region, self.past_len);
        buf.clear();
        buf.resize(s * ctx * elem, 0);
        let blocked: [u8; 4] = {
            let mut b = [0u8; 4];
            match dtype {
                ShimDType::F32 => b.copy_from_slice(&NEG_F32.to_le_bytes()),
                _ => b[..2].copy_from_slice(&NEG_F16_BITS.to_le_bytes()),
            }
            b
        };
        for r in 0..s {
            let row_base = r * ctx * elem;
            let me = plan.rows.get(r).copied().flatten();
            for c in 0..ctx {
                let allow = match me {
                    None => c == past_len + r, // idle: self only, keeps softmax finite
                    Some(pr) => {
                        if c < past_len {
                            // shared prefix this slot is reusing ...
                            if c < self.shared_use[pr.slot] {
                                true
                            } else {
                                // ... plus its own occupied region
                                let start = self.region_start(pr.slot);
                                c >= start && c < start + self.valid(pr.slot)
                            }
                        } else {
                            let q = c - past_len;
                            match plan.rows.get(q).copied().flatten() {
                                Some(other) => other.slot == pr.slot && other.order <= pr.order,
                                None => false,
                            }
                        }
                    }
                };
                if !allow {
                    let off = row_base + c * elem;
                    buf[off..off + elem].copy_from_slice(&blocked[..elem]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(slots: usize, past_len: usize, packed_seq: usize) -> PackedKv {
        PackedKv::new(slots, past_len, packed_seq, 1, 1, 2, ShimDType::F16, 0)
    }

    fn kv_prefix(slots: usize, past_len: usize, packed_seq: usize, prefix: usize) -> PackedKv {
        PackedKv::new(slots, past_len, packed_seq, 1, 1, 2, ShimDType::F16, prefix)
    }

    fn mask_allows(buf: &[u8], ctx: usize, r: usize, c: usize) -> bool {
        let off = (r * ctx + c) * 2;
        u16::from_le_bytes([buf[off], buf[off + 1]]) != NEG_F16_BITS
    }

    #[test]
    fn decode_mask_is_block_diagonal() {
        let mut k = kv(4, 16, 4); // region = 4
        for s in 0..4 {
            k.advance(s, 3); // 3 real tokens each
        }
        let plan = PackedPlan::decode(4, &[0, 1, 2, 3]);
        let mut buf = Vec::new();
        k.fill_mask(&plan, &mut buf, ShimDType::F16);
        for r in 0..4 {
            for c in 0..k.context {
                let want = if c < 16 {
                    // only its own region's first 3 (occupied) slots
                    c >= r * 4 && c < r * 4 + 3
                } else {
                    c == 16 + r // its own query column
                };
                assert_eq!(mask_allows(&buf, k.context, r, c), want, "row {r} col {c}");
            }
        }
    }

    #[test]
    fn a_row_never_sees_another_slots_past_or_query() {
        let mut k = kv(2, 8, 2); // region = 4
        k.advance(0, 4);
        k.advance(1, 4);
        let plan = PackedPlan::decode(2, &[0, 1]);
        let mut buf = Vec::new();
        k.fill_mask(&plan, &mut buf, ShimDType::F16);
        // row 0 must be blocked on slot 1's region (cols 4..8) and on row 1's query col
        for c in 4..8 {
            assert!(!mask_allows(&buf, k.context, 0, c));
        }
        assert!(!mask_allows(&buf, k.context, 0, 8 + 1));
        for c in 0..4 {
            assert!(!mask_allows(&buf, k.context, 1, c));
        }
        assert!(!mask_allows(&buf, k.context, 1, 8));
    }

    #[test]
    fn unoccupied_past_slots_stay_blocked() {
        let mut k = kv(2, 8, 2);
        k.advance(0, 1); // only 1 real token
        let plan = PackedPlan::decode(2, &[0]);
        let mut buf = Vec::new();
        k.fill_mask(&plan, &mut buf, ShimDType::F16);
        assert!(mask_allows(&buf, k.context, 0, 0));
        for c in 1..4 {
            assert!(!mask_allows(&buf, k.context, 0, c), "col {c} unoccupied");
        }
    }

    #[test]
    fn idle_row_is_never_fully_blocked() {
        let k = kv(4, 16, 4);
        let plan = PackedPlan::decode(4, &[0]); // rows 1..3 idle
        let mut buf = Vec::new();
        k.fill_mask(&plan, &mut buf, ShimDType::F16);
        for r in 1..4 {
            let open: Vec<usize> = (0..k.context)
                .filter(|&c| mask_allows(&buf, k.context, r, c))
                .collect();
            assert_eq!(open, vec![16 + r], "idle row {r} must open exactly itself");
        }
    }

    #[test]
    fn chunk_plan_masks_causally_within_one_slot() {
        let mut k = kv(2, 8, 4); // region 4, 4 query rows
        k.advance(1, 2);
        let plan = PackedPlan::chunk(4, 1, 3); // 3 prompt tokens into slot 1
        let mut buf = Vec::new();
        k.fill_mask(&plan, &mut buf, ShimDType::F16);
        for r in 0..3 {
            // own region occupancy
            for c in 4..6 {
                assert!(mask_allows(&buf, k.context, r, c));
            }
            // causal over the chunk's own query columns
            for q in 0..4 {
                let want = q <= r && q < 3;
                assert_eq!(mask_allows(&buf, k.context, r, 8 + q), want, "r{r} q{q}");
            }
            // never the other slot
            for c in 0..4 {
                assert!(!mask_allows(&buf, k.context, r, c));
            }
        }
    }

    /// Row bytes must land inside the owning slot's region, and absorbing into
    /// one slot must not perturb another's bytes.
    #[test]
    fn scatter_is_region_local() {
        let mut k = kv(2, 8, 2); // region 4, head_dim 2, f16 -> 4 bytes/slot
        let present = {
            // present is [heads=1][context=10][head_dim=2] f16; make row r's
            // bytes recognisable: value (r+1) repeated.
            let mut p = vec![0u8; k.present_layer_bytes()];
            for c in 0..k.context {
                for e in 0..2 {
                    let off = (c * 2 + e) * 2;
                    p[off..off + 2].copy_from_slice(&((c as u16) + 100).to_le_bytes());
                }
            }
            p
        };
        k.absorb_row(0, false, &present, 0, 0, 0); // row 0 -> slot 0 @0
        let before_slot1 = k.key_bytes(0)[16..32].to_vec();
        k.absorb_row(0, false, &present, 1, 1, 0); // row 1 -> slot 1 @0
                                                   // slot 0 occupancy 0 holds present row (past_len+0)=8 -> value 108
        let v0 = u16::from_le_bytes([k.key_bytes(0)[0], k.key_bytes(0)[1]]);
        assert_eq!(v0, 108);
        // slot 1 region starts at slot index 4 -> byte 16; holds present row 9
        let v1 = u16::from_le_bytes([k.key_bytes(0)[16], k.key_bytes(0)[17]]);
        assert_eq!(v1, 109);
        assert_ne!(before_slot1, k.key_bytes(0)[16..32].to_vec());
        // slot 0's bytes untouched by the slot-1 write
        assert_eq!(
            u16::from_le_bytes([k.key_bytes(0)[0], k.key_bytes(0)[1]]),
            108
        );
    }

    /// A full region slides within itself and must not consume a neighbour.
    /// With region 4 the sink is capped at 2, so tags 11 and 12 are pinned and
    /// the slide turns over only the last two columns.
    #[test]
    fn full_region_slides_without_crossing_into_the_next_slot() {
        let mut k = kv(2, 8, 1); // region 4, sink 2, packed_seq 1
                                 // Snapshot geometry first: the closure must not hold a borrow of `k`
                                 // across the &mut self absorb calls below.
        let (plb, ctx) = (k.present_layer_bytes(), k.context);
        let mk = move |tag: u16| {
            let mut p = vec![0u8; plb];
            for c in 0..ctx {
                for e in 0..2 {
                    let off = (c * 2 + e) * 2;
                    p[off..off + 2].copy_from_slice(&tag.to_le_bytes());
                }
            }
            p
        };
        // fill slot 1 with a sentinel so we can prove it survives
        k.absorb_row(0, false, &mk(999), 0, 1, 0);
        let slot1_before = k.key_bytes(0)[16..20].to_vec();
        for (i, tag) in [11u16, 12, 13, 14].iter().enumerate() {
            k.absorb_row(0, false, &mk(*tag), 0, 0, i);
        }
        // region now full; one more slides the oldest EVICTABLE tag (13) out
        k.absorb_row(0, false, &mk(15), 0, 0, 4);
        let read = |slot_idx: usize| {
            let off = slot_idx * 4;
            u16::from_le_bytes([k.key_bytes(0)[off], k.key_bytes(0)[off + 1]])
        };
        assert_eq!([read(0), read(1), read(2), read(3)], [11, 12, 14, 15]);
        assert_eq!(
            k.key_bytes(0)[16..20].to_vec(),
            slot1_before,
            "slot 1 must be untouched by slot 0's slide"
        );
    }

    #[test]
    fn reset_slot_clears_only_that_slot() {
        let mut k = kv(2, 8, 1);
        let mut p = vec![0u8; k.present_layer_bytes()];
        p.iter_mut().for_each(|b| *b = 0xAB);
        k.absorb_row(0, false, &p, 0, 0, 0);
        k.absorb_row(0, false, &p, 0, 1, 0);
        k.advance(0, 1);
        k.advance(1, 1);
        k.reset_slot(0);
        assert_eq!(k.position(0), 0);
        assert_eq!(k.position(1), 1);
        assert!(k.key_bytes(0)[0..4].iter().all(|&b| b == 0));
        assert!(k.key_bytes(0)[16..20].iter().any(|&b| b != 0));
    }

    #[test]
    fn valid_saturates_at_region_length() {
        let mut k = kv(4, 16, 4); // region 4
        k.advance(2, 99);
        assert_eq!(k.valid(2), 4);
        assert_eq!(k.position(2), 99);
    }

    /// A relay stage rebuilds `order` from row arrival order; same-slot rows
    /// must come out causally masked exactly as the sender masked them.
    #[test]
    fn rebuilt_chunk_order_masks_identically_to_the_sender() {
        let mut k = kv(2, 8, 4);
        k.advance(1, 2);
        let sender = PackedPlan::chunk(4, 1, 3);
        let mut a = Vec::new();
        k.fill_mask(&sender, &mut a, ShimDType::F16);

        // What a relay reconstructs from the wire: slots only, order re-derived.
        let mut rebuilt = PackedPlan {
            rows: sender
                .rows
                .iter()
                .map(|r| {
                    r.map(|p| PackedRow {
                        slot: p.slot,
                        order: 0,
                    })
                })
                .collect(),
        };
        let mut seen = std::collections::HashMap::new();
        for row in rebuilt.rows.iter_mut().flatten() {
            let n = seen.entry(row.slot).or_insert(0usize);
            row.order = *n;
            *n += 1;
        }
        let mut b = Vec::new();
        k.fill_mask(&rebuilt, &mut b, ShimDType::F16);
        assert_eq!(a, b, "relay-rebuilt plan must mask identically");
    }

    /// Two slots reusing the same cached prefix must both see those columns,
    /// while still being blind to each other's own regions.
    #[test]
    fn shared_prefix_is_readable_by_every_reusing_slot() {
        let mut k = kv_prefix(2, 12, 2, 4); // prefix 4, region (12-4)/2 = 4
        assert_eq!(k.region, 4);
        k.prefix_valid = 3; // 3 cached tokens
        k.reset_slot_reusing(0, 3);
        k.reset_slot_reusing(1, 3);
        k.advance(0, 1); // one own token each
        k.advance(1, 1);
        let plan = PackedPlan::decode(2, &[0, 1]);
        let mut buf = Vec::new();
        k.fill_mask(&plan, &mut buf, ShimDType::F16);
        for r in 0..2 {
            // shared prefix columns 0..3 open for BOTH rows
            for c in 0..3 {
                assert!(mask_allows(&buf, k.context, r, c), "row {r} shared col {c}");
            }
            assert!(
                !mask_allows(&buf, k.context, r, 3),
                "unpopulated prefix col"
            );
        }
        // row 0 sees its own region (4..5) but not slot 1's (8..9)
        assert!(mask_allows(&buf, k.context, 0, 4));
        assert!(!mask_allows(&buf, k.context, 0, 8));
        assert!(mask_allows(&buf, k.context, 1, 8));
        assert!(!mask_allows(&buf, k.context, 1, 4));
    }

    /// Reuse means the slot's own token count excludes the shared tokens, and
    /// its absolute position starts after them (so RoPE stays correct).
    #[test]
    fn reuse_shifts_position_and_own_valid() {
        let mut k = kv_prefix(2, 12, 2, 4);
        k.prefix_valid = 4;
        k.reset_slot_reusing(0, 4);
        assert_eq!(k.position(0), 4, "sequence resumes after the cached prefix");
        assert_eq!(k.valid(0), 0, "no OWN tokens yet");
        k.advance(0, 2);
        assert_eq!(k.position(0), 6);
        assert_eq!(k.valid(0), 2);
        assert_eq!(k.shared_use(0), 4);
    }

    /// Populating the cache copies a slot's first tokens to the front of the
    /// window without touching any other slot's region.
    #[test]
    fn populate_prefix_copies_out_of_the_owning_slot() {
        let mut k = kv_prefix(2, 12, 1, 4);
        let mk = |tag: u16, plb: usize, ctx: usize| {
            let mut p = vec![0u8; plb];
            for c in 0..ctx {
                for e in 0..2 {
                    let off = (c * 2 + e) * 2;
                    p[off..off + 2].copy_from_slice(&tag.to_le_bytes());
                }
            }
            p
        };
        let (plb, ctx) = (k.present_layer_bytes(), k.context);
        // slot 1 sentinel must survive
        k.absorb_row(0, false, &mk(999, plb, ctx), 0, 1, 0);
        let slot1 = k.key_bytes(0)[(4 + 4) * 4..(4 + 4) * 4 + 4].to_vec();
        for (i, tag) in [21u16, 22].iter().enumerate() {
            k.absorb_row(0, false, &mk(*tag, plb, ctx), 0, 0, i);
        }
        k.advance(0, 2);
        k.populate_prefix_from(0, 2);
        assert_eq!(k.prefix_valid, 2);
        let read = |slot_idx: usize| {
            let off = slot_idx * 4;
            u16::from_le_bytes([k.key_bytes(0)[off], k.key_bytes(0)[off + 1]])
        };
        assert_eq!(
            [read(0), read(1)],
            [21, 22],
            "prefix holds the copied tokens"
        );
        assert_eq!(
            k.key_bytes(0)[(4 + 4) * 4..(4 + 4) * 4 + 4].to_vec(),
            slot1,
            "another slot's region must be untouched"
        );
    }

    #[test]
    fn prefix_capacity_shrinks_per_slot_region() {
        let plain = kv(4, 1024, 4);
        let with_prefix = kv_prefix(4, 1024, 4, 256);
        assert_eq!(plain.region, 256);
        assert_eq!(with_prefix.region, 192); // (1024-256)/4
    }

    #[test]
    fn f32_mask_uses_four_byte_elements() {
        let k = kv(2, 8, 2);
        let mut buf = Vec::new();
        k.fill_mask(&PackedPlan::idle(2), &mut buf, ShimDType::F32);
        assert_eq!(buf.len(), 2 * k.context * 4);
    }

    /// Feed `n` single-row tokens into `slot`, tagging each one's K/V with its
    /// own byte so the surviving entries are identifiable afterwards.
    fn feed_tokens(k: &mut PackedKv, slot: usize, n: u8) {
        let slot_bytes = k.head_dim * k.elem_bytes;
        for t in 1..=n {
            let mut present = vec![0u8; k.context * slot_bytes];
            let src = k.past_len * slot_bytes; // query row 0
            present[src..src + slot_bytes].fill(t);
            let at = k.valid(slot);
            k.absorb_row(0, false, &present, 0, slot, at);
            k.advance(slot, 1);
        }
    }

    fn region_bytes(k: &PackedKv, slot: usize) -> Vec<u8> {
        let slot_bytes = k.head_dim * k.elem_bytes;
        let start = k.region_start(slot) * slot_bytes;
        (0..k.region)
            .map(|i| k.key_buf[0][start + i * slot_bytes])
            .collect()
    }

    /// Eviction must never drop a slot's FIRST columns. Transformer heads park
    /// a large share of their softmax mass on the earliest tokens; sliding them
    /// out collapses generation into degenerate repetition instead of degrading
    /// it gracefully. Feeds 12 tokens through a region-6 slot: the four sinks
    /// must survive while only the tail slides.
    #[test]
    fn slide_preserves_the_attention_sink() {
        let mut k = kv(2, 20, 2); // region = 10, sink = 4
        assert_eq!((k.region, k.sink), (10, 4));
        feed_tokens(&mut k, 0, 16);
        assert_eq!(
            region_bytes(&k, 0),
            vec![1, 2, 3, 4, 11, 12, 13, 14, 15, 16],
            "sinks 1-4 pinned; only the six evictable columns slide"
        );
    }

    /// A neighbouring slot's region must stay untouched by the sink-aware slide.
    #[test]
    fn sink_slide_stays_inside_its_own_region() {
        let mut k = kv(2, 20, 2);
        feed_tokens(&mut k, 1, 16);
        assert_eq!(region_bytes(&k, 0), vec![0; 10], "slot 0 never written");
        assert_eq!(
            region_bytes(&k, 1),
            vec![1, 2, 3, 4, 11, 12, 13, 14, 15, 16]
        );
    }

    /// Pinning must never crowd out the sliding window: on a region too small
    /// for the full sink it is capped at half, so the slot keeps turning over.
    #[test]
    fn sink_is_capped_at_half_the_region() {
        let mut k = kv(4, 16, 2); // region = 4 -> sink capped to 2
        assert_eq!((k.region, k.sink), (4, 2));
        feed_tokens(&mut k, 0, 9);
        assert_eq!(region_bytes(&k, 0), vec![1, 2, 8, 9]);
    }
}
