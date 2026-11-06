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
//!
//! Gemma 4 adds three things, all off by default so the other families mask
//! exactly as before:
//! - **Per-layer KV geometry.** Sliding layers are 8 KV heads x 256 and global
//!   layers 2 x 512; each layer's ring is sized by its own `(kv_heads,
//!   head_dim)` ([`PackedKv::with_layers`], from `stage_config.kv_layers`).
//! - **A sliding-window mask.** The packed IR then has a second mask port fed
//!   by [`PackedKv::fill_mask_window`]: the same mask with every column whose
//!   absolute position is `window` or more behind the query row blocked. A
//!   column's position is known from the slot's layout (shared prefix, the
//!   pinned sink columns, then the slid window), see `column_position`.
//! - **Image blocks.** Rows of one image ([`PackedRow::image`]) attend to each
//!   other both ways, on top of causal order, in both masks. An image must
//!   therefore sit whole inside one inference ([`image_safe_take`]).

use cascadia_ov_genai_shim::DType as ShimDType;

/// Additive-mask "blocked" value: f16's most-negative finite, -65504, in
/// both dtypes — finite rather than -inf so a fully-blocked row can never
/// produce NaN out of softmax. The f32 value must stay inside f16 range: the
/// GPU plugin runs the IR in f16 by default and converts an f32 mask input to
/// f16, where anything below -65504 (e.g. f32's -3e38) becomes -inf, and its
/// f16 SDPA turns -inf mask entries into wrong-but-finite attention (garbage
/// tokens). -65504 still underflows softmax to exactly 0 in f32, so CPU output
/// is unchanged.
const NEG_F16_BITS: u16 = 0xFBFF;
const NEG_F32: f32 = -65504.0;

/// Leading KV columns a bounded ring must never evict — attention sinks.
/// Transformer heads park a large share of their softmax mass on the first few
/// tokens of a sequence; slide those out and the mass is forced onto ordinary
/// content, which collapses generation into degenerate repetition rather than
/// truncating it gracefully. Four is the width the StreamingLLM result found
/// sufficient. Shared with the single-task ring in `runtime.rs`, which has the
/// same bounded-window shape at the full `past_len`.
pub(crate) const KV_SINK: usize = 4;

/// Which slot a query row belongs to, and its ordinal among that slot's rows in
/// this same inference (0 for decode; 0..n for a prefill chunk). `order` drives
/// causal masking within a slot.
///
/// `image` names the image a row belongs to (Gemma 4: rows with
/// `token_type_ids == 1`, one id per contiguous run in the prompt). Same-slot
/// rows of the same image attend to each other regardless of order. `None` on
/// text rows and on every family without bidirectional image attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedRow {
    pub slot: usize,
    pub order: usize,
    pub image: Option<u32>,
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
            rows[r] = Some(PackedRow {
                slot,
                order: 0,
                image: None,
            });
        }
        Self { rows }
    }

    /// `n` consecutive rows all belonging to `slot` — the prefill-chunk shape.
    pub fn chunk(packed_seq: usize, slot: usize, n: usize) -> Self {
        let mut rows = vec![None; packed_seq];
        for (order, row) in rows.iter_mut().take(n.min(packed_seq)).enumerate() {
            *row = Some(PackedRow {
                slot,
                order,
                image: None,
            });
        }
        Self { rows }
    }

    /// A prefill chunk whose rows carry their image ids (`images.len()` rows,
    /// `None` for text) — the Gemma 4 shape of [`PackedPlan::chunk`].
    pub fn chunk_with_images(packed_seq: usize, slot: usize, images: &[Option<u32>]) -> Self {
        let mut plan = Self::chunk(packed_seq, slot, images.len());
        for (row, &image) in plan.rows.iter_mut().flatten().zip(images) {
            row.image = image;
        }
        plan
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
    /// [`KV_SINK`]). Capped at half the region so pinning can never crowd
    /// out the sliding window itself.
    sink: usize,
    pub past_len: usize,
    /// Query rows of the widest inference (a prefill chunk). A decode step
    /// may run narrower (see [`PackedKv::context_at`]).
    pub packed_seq: usize,
    /// `present.*` length of the widest inference, `past_len + packed_seq`.
    pub context: usize,
    /// `(kv_heads, head_dim)` per layer, in port order. Uniform on most
    /// families; Gemma 4 mixes sliding (8 x 256) and global (2 x 512) layers.
    geom: Vec<(usize, usize)>,
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
    /// Every layer `kv_heads` x `head_dim`.
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
        Self::with_layers(
            slots,
            past_len,
            packed_seq,
            vec![(kv_heads, head_dim); layers],
            kv_dtype,
            prefix_capacity,
        )
    }

    /// Each layer with its own `(kv_heads, head_dim)` (`stage_config.kv_layers`).
    pub fn with_layers(
        slots: usize,
        past_len: usize,
        packed_seq: usize,
        geom: Vec<(usize, usize)>,
        kv_dtype: ShimDType,
        prefix_capacity: usize,
    ) -> Self {
        let elem_bytes = match kv_dtype {
            ShimDType::F32 | ShimDType::I32 => 4,
            ShimDType::I64 => 8,
            _ => 2,
        };
        let key_buf: Vec<Vec<u8>> = geom
            .iter()
            .map(|&(kv_heads, head_dim)| vec![0u8; kv_heads * past_len * head_dim * elem_bytes])
            .collect();
        let prefix_capacity = prefix_capacity.min(past_len.saturating_sub(slots));
        let region = (past_len - prefix_capacity) / slots;
        Self {
            slots,
            prefix_capacity,
            prefix_valid: 0,
            region,
            sink: KV_SINK.min(region / 2),
            past_len,
            packed_seq,
            context: past_len + packed_seq,
            geom,
            elem_bytes,
            val_buf: key_buf.clone(),
            key_buf,
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
        let (region, past_len) = (self.region, self.past_len);
        let start_col = self.region_start(s);
        for (li, &(kv_heads, head_dim)) in self.geom.iter().enumerate() {
            let slot_bytes = head_dim * self.elem_bytes;
            let buf_row = past_len * slot_bytes;
            let start = start_col * slot_bytes;
            for buf in [&mut self.key_buf[li], &mut self.val_buf[li]] {
                for h in 0..kv_heads {
                    let base = h * buf_row + start;
                    buf[base..base + region * slot_bytes].fill(0);
                }
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
        let src_col = self.region_start(src);
        for (li, &(kv_heads, head_dim)) in self.geom.iter().enumerate() {
            let slot_bytes = head_dim * self.elem_bytes;
            let buf_row = self.past_len * slot_bytes;
            let src_start = src_col * slot_bytes;
            for buf in [&mut self.key_buf[li], &mut self.val_buf[li]] {
                for h in 0..kv_heads {
                    let base = h * buf_row;
                    let from = base + src_start;
                    let to = base;
                    // Non-overlapping by construction: the shared region is
                    // [0, prefix_capacity) and every slot region starts at or
                    // after prefix_capacity.
                    buf.copy_within(from..from + n * slot_bytes, to);
                }
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

    /// KV heads of layer `li`.
    pub fn kv_heads(&self, li: usize) -> usize {
        self.geom[li].0
    }

    /// Head width of layer `li`.
    pub fn head_dim(&self, li: usize) -> usize {
        self.geom[li].1
    }

    /// `(kv_heads, head_dim)` of every layer, in port order.
    pub fn geometry(&self) -> &[(usize, usize)] {
        &self.geom
    }

    /// `present.*` length of an inference `width` query rows wide: the past
    /// window plus its own rows. Every width shares the one past window, so
    /// the host ring serves a narrow decode inference and a wide prefill
    /// chunk alike.
    pub fn context_at(&self, width: usize) -> usize {
        self.past_len + width
    }

    /// Bytes of layer `li`'s `present.*` output for an inference `width` rows
    /// wide (`[1, kv_heads, past_len + width, head_dim]`).
    pub fn present_layer_bytes(&self, li: usize, width: usize) -> usize {
        let (kv_heads, head_dim) = self.geom[li];
        kv_heads * self.context_at(width) * head_dim * self.elem_bytes
    }

    /// Advance a slot's absolute position after its row(s) were absorbed.
    pub fn advance(&mut self, s: usize, by: usize) {
        self.pos[s] += by;
    }

    /// Absorb query row `row`'s K/V (which sits at `present` index
    /// `past_len + row` of an inference `width` rows wide) into `slot`'s
    /// region, appending at `at` or sliding the region's oldest entry out when
    /// it is full. `at` is the caller's running occupancy for that slot within
    /// this step, so a multi-row prefill chunk lands consecutively.
    ///
    /// The slide is strictly bounded to `[slot*region, (slot+1)*region)` — a
    /// neighbouring slot's KV is never read or written.
    #[allow(clippy::too_many_arguments)]
    pub fn absorb_row(
        &mut self,
        li: usize,
        is_value: bool,
        present: &[u8],
        width: usize,
        row: usize,
        slot: usize,
        at: usize,
    ) {
        let (kv_heads, head_dim) = self.geom[li];
        let slot_bytes = head_dim * self.elem_bytes;
        let present_row = self.context_at(width) * slot_bytes;
        let buf_row = self.past_len * slot_bytes;
        let region = self.region;
        let sink = self.sink;
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

    /// Absolute position of the token held in past column `c` for `slot`, or
    /// `None` when the column holds nothing `slot` may read.
    ///
    /// Layout: the shared prefix `[0, shared_use)` holds positions `0..`; the
    /// slot's region holds its own tokens from position `shared_use` on. Once
    /// more own tokens than `region` went through it, the region has slid: its
    /// first `sink` columns stay pinned to the first own tokens, and the rest
    /// hold the most recent `region - sink` tokens in order, the last at
    /// `pos - 1`.
    fn column_position(&self, slot: usize, c: usize) -> Option<usize> {
        let shared = self.shared_use[slot];
        if c < shared {
            return Some(c);
        }
        let start = self.region_start(slot);
        if c < start {
            return None;
        }
        let i = c - start;
        if i >= self.valid(slot) {
            return None;
        }
        let own = self.pos[slot] - shared;
        Some(if own <= self.region || i < self.sink {
            shared + i
        } else {
            shared + own - self.region + i
        })
    }

    /// Write the additive attention mask for `plan` into `buf` as a
    /// `[1, 1, S, past_len + S]` tensor of `dtype`, `S = plan.rows.len()`
    /// (the inference's width: `packed_seq` for a prefill chunk, possibly
    /// fewer rows for a decode step).
    ///
    /// Row `r` is opened on exactly: its slot's occupied past region, and the
    /// query columns of same-slot rows with `order <= its order`, or of the
    /// same image (see [`PackedRow::image`]). Everything else is blocked. An
    /// idle row is opened on its own query column only — never fully blocked,
    /// which would make softmax produce NaN and poison the whole inference
    /// including live rows.
    pub fn fill_mask(&self, plan: &PackedPlan, buf: &mut Vec<u8>, dtype: ShimDType) {
        self.fill_mask_window(plan, buf, dtype, None);
    }

    /// [`PackedKv::fill_mask`] for a sliding-window layer when `window` is
    /// set: additionally block every column whose token sits `window` or more
    /// positions behind the query row (HF: visible iff `0 <= q - k < window`).
    /// Positions are absolute, so the region's slide and the pinned sink
    /// columns are accounted for (`column_position`). A query row's position
    /// is its slot's next position plus its `order`. Image blocks stay open:
    /// an image never spans a window.
    pub fn fill_mask_window(
        &self,
        plan: &PackedPlan,
        buf: &mut Vec<u8>,
        dtype: ShimDType,
        window: Option<usize>,
    ) {
        self.fill_mask_blocks(plan, buf, dtype, window, true);
    }

    /// [`PackedKv::fill_mask_window`], opening image blocks only when
    /// `blocks`: HF's Gemma 4 keeps its full-attention layers causal (only
    /// the sliding layers see each image both ways), Gemma 3 opens both.
    pub fn fill_mask_blocks(
        &self,
        plan: &PackedPlan,
        buf: &mut Vec<u8>,
        dtype: ShimDType,
        window: Option<usize>,
        blocks: bool,
    ) {
        let elem = match dtype {
            ShimDType::F32 => 4,
            _ => 2,
        };
        let s = plan.rows.len();
        let (ctx, past_len) = (self.context_at(s), self.past_len);
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
        let in_window = |q: usize, k: usize| window.is_none_or(|w| q >= k && q - k < w);
        for r in 0..s {
            let row_base = r * ctx * elem;
            let me = plan.rows.get(r).copied().flatten();
            let my_pos = me.map(|pr| self.pos[pr.slot] + pr.order);
            for c in 0..ctx {
                let allow = match me {
                    None => c == past_len + r, // idle: self only, keeps softmax finite
                    Some(pr) => {
                        let q_pos = my_pos.unwrap_or(0);
                        if c < past_len {
                            // the shared prefix this slot reuses, plus its own
                            // occupied region
                            match self.column_position(pr.slot, c) {
                                Some(k) => window.is_none() || in_window(q_pos, k),
                                None => false,
                            }
                        } else {
                            let q = c - past_len;
                            match plan.rows.get(q).copied().flatten() {
                                Some(other) if other.slot == pr.slot => {
                                    let same_image =
                                        blocks && pr.image.is_some() && other.image == pr.image;
                                    same_image
                                        || (other.order <= pr.order
                                            && in_window(q_pos, self.pos[pr.slot] + other.order))
                                }
                                _ => false,
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

/// Rows of the next prefill chunk starting at prompt index `idx`, at most
/// `take`, never ending inside an image (`images`: one entry per prompt token,
/// the image id or `None`). Err when an image starting at `idx` is wider than
/// `take`. Empty `images` (a tree without image blocks) leaves `take` as is.
pub fn image_safe_take(images: &[Option<u32>], idx: usize, take: usize) -> Result<usize, String> {
    let end = idx + take;
    if images.is_empty() || take == 0 || end >= images.len() {
        return Ok(take);
    }
    let (last, next) = (images[end - 1], images[end]);
    if last.is_none() || last != next {
        return Ok(take);
    }
    let mut start = end - 1;
    while start > idx && images[start - 1] == last {
        start -= 1;
    }
    if start > idx {
        return Ok(start - idx);
    }
    let run = images[idx..].iter().take_while(|&&t| t == last).count();
    Err(format!(
        "an image of {run} tokens does not fit one packed inference of {take} rows; \
         rebuild the packed variant with a --packed-seq at least that wide"
    ))
}

/// Image id per prompt token: each contiguous run of `image_token` gets the
/// next id (0, 1, ...), text tokens `None`. The ids only need to tell one
/// image from another inside one slot's chunk.
pub fn prompt_image_ids(prompt_ids: &[i64], image_token: i64) -> Vec<Option<u32>> {
    let mut next = 0u32;
    let mut prev_image = false;
    prompt_ids
        .iter()
        .map(|&t| {
            let is_image = t == image_token;
            let id = if is_image {
                if !prev_image {
                    next += 1;
                }
                Some(next - 1)
            } else {
                None
            };
            prev_image = is_image;
            id
        })
        .collect()
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

    /// A decode step run narrower (one row per live slot) masks each live row
    /// exactly as the wide inference did: the same past columns, the same own
    /// query column. The wide one's extra query columns were blocked for every
    /// live row, so dropping them changes nothing a live row attends to.
    #[test]
    fn narrow_decode_mask_matches_the_wide_one_on_live_rows() {
        let mut k = kv(4, 16, 8); // region 4, widest inference 8 rows
        for (s, n) in [(0, 3), (1, 1), (2, 4), (3, 2)] {
            k.advance(s, n);
        }
        let live = [0usize, 2, 3];
        let (mut wide, mut narrow) = (Vec::new(), Vec::new());
        k.fill_mask(&PackedPlan::decode(8, &live), &mut wide, ShimDType::F16);
        k.fill_mask(&PackedPlan::decode(3, &live), &mut narrow, ShimDType::F16);
        let (wctx, nctx) = (k.context_at(8), k.context_at(3));
        assert_eq!(narrow.len(), 3 * nctx * 2);
        for r in 0..3 {
            for c in 0..16 {
                assert_eq!(
                    mask_allows(&narrow, nctx, r, c),
                    mask_allows(&wide, wctx, r, c),
                    "row {r} past col {c}"
                );
            }
            for q in 0..8 {
                let w = mask_allows(&wide, wctx, r, 16 + q);
                assert_eq!(w, q == r, "wide row {r} query col {q}");
                if q < 3 {
                    assert_eq!(mask_allows(&narrow, nctx, r, 16 + q), w);
                }
            }
        }
    }

    /// The row K/V of an inference `width` rows wide sits at present column
    /// `past_len + row` of a `past_len + width` long present: absorbing from
    /// a narrow inference lands the same bytes the wide one would.
    #[test]
    fn absorb_reads_the_row_at_the_inference_width() {
        let present = |k: &PackedKv, width: usize| {
            let mut p = vec![0u8; k.present_layer_bytes(0, width)];
            for c in 0..k.context_at(width) {
                let off = c * 4;
                p[off..off + 2].copy_from_slice(&((c as u16) + 100).to_le_bytes());
            }
            p
        };
        let mut wide = kv(2, 8, 6);
        let mut narrow = kv(2, 8, 6);
        let (pw, pn) = (present(&wide, 6), present(&narrow, 2));
        assert_eq!(pn.len(), 10 * 4);
        wide.absorb_row(0, false, &pw, 6, 1, 1, 0);
        narrow.absorb_row(0, false, &pn, 2, 1, 1, 0);
        assert_eq!(wide.key_bytes(0), narrow.key_bytes(0));
        // slot 1's region starts at column 4 (byte 16): present row 1 = col 9
        assert_eq!(
            u16::from_le_bytes([narrow.key_bytes(0)[16], narrow.key_bytes(0)[17]]),
            109
        );
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
            let mut p = vec![0u8; k.present_layer_bytes(0, k.packed_seq)];
            for c in 0..k.context {
                for e in 0..2 {
                    let off = (c * 2 + e) * 2;
                    p[off..off + 2].copy_from_slice(&((c as u16) + 100).to_le_bytes());
                }
            }
            p
        };
        k.absorb_row(0, false, &present, k.packed_seq, 0, 0, 0); // row 0 -> slot 0 @0
        let before_slot1 = k.key_bytes(0)[16..32].to_vec();
        k.absorb_row(0, false, &present, k.packed_seq, 1, 1, 0); // row 1 -> slot 1 @0
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
        let (plb, ctx) = (k.present_layer_bytes(0, k.packed_seq), k.context);
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
        k.absorb_row(0, false, &mk(999), k.packed_seq, 0, 1, 0);
        let slot1_before = k.key_bytes(0)[16..20].to_vec();
        for (i, tag) in [11u16, 12, 13, 14].iter().enumerate() {
            k.absorb_row(0, false, &mk(*tag), k.packed_seq, 0, 0, i);
        }
        // region now full; one more slides the oldest EVICTABLE tag (13) out
        k.absorb_row(0, false, &mk(15), k.packed_seq, 0, 0, 4);
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
        let mut p = vec![0u8; k.present_layer_bytes(0, k.packed_seq)];
        p.iter_mut().for_each(|b| *b = 0xAB);
        k.absorb_row(0, false, &p, k.packed_seq, 0, 0, 0);
        k.absorb_row(0, false, &p, k.packed_seq, 0, 1, 0);
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
                        image: p.image,
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
        let (plb, ctx) = (k.present_layer_bytes(0, k.packed_seq), k.context);
        // slot 1 sentinel must survive
        k.absorb_row(0, false, &mk(999, plb, ctx), k.packed_seq, 0, 1, 0);
        let slot1 = k.key_bytes(0)[(4 + 4) * 4..(4 + 4) * 4 + 4].to_vec();
        for (i, tag) in [21u16, 22].iter().enumerate() {
            k.absorb_row(0, false, &mk(*tag, plb, ctx), k.packed_seq, 0, 0, i);
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

    /// The f32 mask's blocked value survives the f16 conversion the GPU
    /// plugin applies to it: finite, and the same -65504 the f16 mask writes.
    #[test]
    fn f32_mask_blocked_value_is_f16_representable() {
        let k = kv(2, 8, 2);
        let (mut b32, mut b16) = (Vec::new(), Vec::new());
        k.fill_mask(&PackedPlan::idle(2), &mut b32, ShimDType::F32);
        k.fill_mask(&PackedPlan::idle(2), &mut b16, ShimDType::F16);
        let v32: Vec<f32> = b32
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let blocked: Vec<f32> = v32.iter().copied().filter(|&v| v != 0.0).collect();
        assert!(!blocked.is_empty());
        assert!(blocked.iter().all(|&v| v == -65504.0));
        assert_eq!(f32::from(half::f16::from_bits(NEG_F16_BITS)), NEG_F32);
        // Same open/blocked pattern in both dtypes.
        for (i, &v) in v32.iter().enumerate() {
            let h = u16::from_le_bytes([b16[2 * i], b16[2 * i + 1]]);
            assert_eq!(v == 0.0, h == 0, "element {i}");
        }
    }

    /// Feed `n` single-row tokens into `slot`, tagging each one's K/V with its
    /// own byte so the surviving entries are identifiable afterwards.
    fn feed_tokens(k: &mut PackedKv, slot: usize, n: u8) {
        let slot_bytes = k.head_dim(0) * k.elem_bytes;
        for t in 1..=n {
            let mut present = vec![0u8; k.context * slot_bytes];
            let src = k.past_len * slot_bytes; // query row 0
            present[src..src + slot_bytes].fill(t);
            let at = k.valid(slot);
            k.absorb_row(0, false, &present, k.packed_seq, 0, slot, at);
            k.advance(slot, 1);
        }
    }

    fn region_bytes(k: &PackedKv, slot: usize) -> Vec<u8> {
        let slot_bytes = k.head_dim(0) * k.elem_bytes;
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

    // ---- Gemma 4: sliding window, image blocks, mixed KV geometry ----

    fn allowed_cols(buf: &[u8], ctx: usize, r: usize) -> Vec<usize> {
        (0..ctx).filter(|&c| mask_allows(buf, ctx, r, c)).collect()
    }

    /// Without a window the windowed writer is the plain mask, byte for byte.
    #[test]
    fn no_window_is_the_plain_mask() {
        let mut k = kv(2, 8, 4);
        k.advance(0, 3);
        k.advance(1, 2);
        let plan = PackedPlan::chunk(4, 1, 3);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        k.fill_mask(&plan, &mut a, ShimDType::F16);
        k.fill_mask_window(&plan, &mut b, ShimDType::F16, None);
        assert_eq!(a, b);
    }

    /// A decode row sees only the last `window` positions, itself included.
    #[test]
    fn sliding_window_blocks_columns_window_or_more_behind() {
        let mut k = kv(2, 20, 2); // region 10
        k.advance(0, 6); // positions 0..5 in cols 0..5; next row is position 6
        let plan = PackedPlan::decode(2, &[0]);
        let mut full = Vec::new();
        k.fill_mask(&plan, &mut full, ShimDType::F16);
        assert_eq!(
            allowed_cols(&full, k.context, 0),
            vec![0, 1, 2, 3, 4, 5, 20]
        );
        let mut sw = Vec::new();
        k.fill_mask_window(&plan, &mut sw, ShimDType::F16, Some(3));
        // 6 - k < 3 -> positions 4, 5 (cols 4, 5) and itself.
        assert_eq!(allowed_cols(&sw, k.context, 0), vec![4, 5, 20]);
    }

    /// Within a prefill chunk the window applies between query rows too.
    #[test]
    fn sliding_window_applies_inside_a_chunk() {
        let mut k = kv(2, 8, 4); // region 4
        k.advance(1, 1); // slot 1: one past token at position 0 (col 4)
        let plan = PackedPlan::chunk(4, 1, 4); // positions 1..4
        let mut sw = Vec::new();
        k.fill_mask_window(&plan, &mut sw, ShimDType::F16, Some(2));
        let ctx = k.context;
        assert_eq!(allowed_cols(&sw, ctx, 0), vec![4, 8]); // pos 1 sees 0 and 1
        assert_eq!(allowed_cols(&sw, ctx, 1), vec![8, 9]); // pos 2 sees 1, 2
        assert_eq!(allowed_cols(&sw, ctx, 3), vec![10, 11]); // pos 4 sees 3, 4
    }

    /// Once a region has slid, its columns are no longer in position order:
    /// the pinned sink columns keep the slot's first tokens and the rest hold
    /// the most recent ones. The window must judge each by its real position.
    #[test]
    fn sliding_window_accounts_for_the_slide_and_the_sinks() {
        let mut k = kv(2, 20, 2); // region 10, sink 4
        feed_tokens(&mut k, 0, 16); // cols hold positions [0,1,2,3, 10..=15]
        assert_eq!(k.position(0), 16);
        let plan = PackedPlan::decode(2, &[0]);
        let mut full = Vec::new();
        k.fill_mask(&plan, &mut full, ShimDType::F16);
        assert_eq!(
            allowed_cols(&full, k.context, 0),
            (0..10).chain([20]).collect::<Vec<_>>()
        );
        let mut sw = Vec::new();
        k.fill_mask_window(&plan, &mut sw, ShimDType::F16, Some(5));
        // 16 - k < 5 -> positions 12..=15 = cols 6..=9; the sinks (0..=3) are
        // far behind and blocked, though they never slide out.
        assert_eq!(allowed_cols(&sw, k.context, 0), vec![6, 7, 8, 9, 20]);
        // A window wider than everything opens exactly what the full mask does.
        let mut wide = Vec::new();
        k.fill_mask_window(&plan, &mut wide, ShimDType::F16, Some(100));
        assert_eq!(wide, full);
    }

    /// Shared-prefix columns sit at positions 0.. and are windowed as such.
    #[test]
    fn sliding_window_covers_the_shared_prefix() {
        let mut k = kv_prefix(2, 12, 2, 4); // prefix 4, region 4
        k.prefix_valid = 4;
        k.reset_slot_reusing(0, 4); // positions 0..3 in the prefix
        k.advance(0, 2); // own positions 4, 5 in cols 4, 5; next is 6
        let plan = PackedPlan::decode(2, &[0]);
        let mut sw = Vec::new();
        k.fill_mask_window(&plan, &mut sw, ShimDType::F16, Some(4));
        // 6 - k < 4 -> positions 3..=5: prefix col 3, own cols 4, 5, itself.
        assert_eq!(allowed_cols(&sw, k.context, 0), vec![3, 4, 5, 12]);
    }

    fn image_plan(packed_seq: usize, slot: usize, images: &[Option<u32>]) -> PackedPlan {
        PackedPlan::chunk_with_images(packed_seq, slot, images)
    }

    /// Rows of one image see each other both ways; text keeps causal order,
    /// and text before an image never sees it.
    #[test]
    fn image_rows_attend_to_each_other_both_ways() {
        let k = kv(2, 8, 6); // region 4, nothing in the past
        let imgs = [None, Some(0), Some(0), Some(0), None];
        let plan = image_plan(6, 0, &imgs);
        for window in [None, Some(1024)] {
            let mut buf = Vec::new();
            k.fill_mask_window(&plan, &mut buf, ShimDType::F16, window);
            let q = |r: usize| -> Vec<usize> {
                allowed_cols(&buf, k.context, r)
                    .iter()
                    .map(|c| c - 8)
                    .collect()
            };
            assert_eq!(q(0), vec![0], "text before the image ({window:?})");
            for r in 1..4 {
                assert_eq!(q(r), vec![0, 1, 2, 3], "image row {r} ({window:?})");
            }
            assert_eq!(q(4), vec![0, 1, 2, 3, 4], "text after the image");
            assert_eq!(q(5), vec![5], "idle row: itself only");
        }
    }

    /// Without blocks (Gemma 4's full-attention layers) image rows stay
    /// causal; the sliding mask of the same plan opens them.
    #[test]
    fn full_mask_without_blocks_keeps_images_causal() {
        let k = kv(2, 8, 6);
        let plan = image_plan(6, 0, &[None, Some(0), Some(0), Some(0), None]);
        let mut buf = Vec::new();
        k.fill_mask_blocks(&plan, &mut buf, ShimDType::F16, None, false);
        let q = |buf: &[u8], r: usize| -> Vec<usize> {
            allowed_cols(buf, k.context, r)
                .iter()
                .map(|c| c - 8)
                .collect()
        };
        for r in 0..5 {
            assert_eq!(q(&buf, r), (0..=r).collect::<Vec<_>>(), "row {r}");
        }
        k.fill_mask_blocks(&plan, &mut buf, ShimDType::F16, Some(1024), true);
        assert_eq!(q(&buf, 1), vec![0, 1, 2, 3]);
    }

    /// Two images in one chunk stay separate blocks.
    #[test]
    fn different_images_do_not_see_each_other_ahead() {
        let k = kv(2, 8, 4);
        let plan = image_plan(4, 0, &[Some(0), Some(0), Some(1), Some(1)]);
        let mut buf = Vec::new();
        k.fill_mask(&plan, &mut buf, ShimDType::F16);
        let q = |r: usize| -> Vec<usize> {
            allowed_cols(&buf, k.context, r)
                .iter()
                .map(|c| c - 8)
                .collect()
        };
        assert_eq!(q(0), vec![0, 1]);
        assert_eq!(q(1), vec![0, 1]);
        assert_eq!(
            q(2),
            vec![0, 1, 2, 3],
            "causal to image 0, block to image 1"
        );
        assert_eq!(q(3), vec![0, 1, 2, 3]);
    }

    /// Image blocks never cross slots, even when both slots' rows carry the
    /// same image id (ids are per prompt).
    #[test]
    fn image_blocks_stay_inside_their_slot() {
        let mut k = kv(2, 8, 4);
        k.advance(0, 2);
        k.advance(1, 3);
        let row = |slot, order| {
            Some(PackedRow {
                slot,
                order,
                image: Some(0),
            })
        };
        let plan = PackedPlan {
            rows: vec![row(0, 0), row(0, 1), row(1, 0), row(1, 1)],
        };
        for window in [None, Some(3)] {
            let mut buf = Vec::new();
            k.fill_mask_window(&plan, &mut buf, ShimDType::F16, window);
            for r in 0..4 {
                let slot = r / 2;
                for c in allowed_cols(&buf, k.context, r) {
                    let owner = if c < 8 { c / 4 } else { (c - 8) / 2 };
                    assert_eq!(owner, slot, "row {r} opened col {c} of slot {owner}");
                }
            }
        }
    }

    /// Mixed geometry: each layer's ring has its own heads and width, and
    /// absorb / reset / populate keep every layer inside its own layout.
    #[test]
    fn mixed_geometry_absorbs_each_layer_by_its_own_shape() {
        // layer 0: 1 head x 2; layer 1: 2 heads x 4 (f16)
        let mut k = PackedKv::with_layers(2, 8, 2, vec![(1, 2), (2, 4)], ShimDType::F16, 0);
        assert_eq!(k.key_bytes(0).len(), 8 * 2 * 2);
        assert_eq!(k.key_bytes(1).len(), 2 * 8 * 4 * 2);
        assert_eq!(k.present_layer_bytes(0, k.packed_seq), 10 * 2 * 2);
        assert_eq!(k.present_layer_bytes(1, k.packed_seq), 2 * 10 * 4 * 2);
        assert_eq!(k.geometry(), &[(1, 2), (2, 4)]);
        // present for layer 1: [heads 2][ctx 10][hd 4]; tag each (head, col).
        let mut p1 = vec![0u8; k.present_layer_bytes(1, k.packed_seq)];
        for h in 0..2 {
            for c in 0..10 {
                for e in 0..4 {
                    let off = ((h * 10 + c) * 4 + e) * 2;
                    let tag = (h * 100 + c) as u16;
                    p1[off..off + 2].copy_from_slice(&tag.to_le_bytes());
                }
            }
        }
        let p0 = vec![0x11u8; k.present_layer_bytes(0, k.packed_seq)];
        k.absorb_row(0, false, &p0, k.packed_seq, 1, 1, 0);
        k.absorb_row(1, false, &p1, k.packed_seq, 1, 1, 0); // row 1 -> slot 1 (cols 4..8) @0
        let rd = |buf: &[u8], h: usize, col: usize| {
            let off = (h * 8 + col) * 4 * 2;
            u16::from_le_bytes([buf[off], buf[off + 1]])
        };
        // present row 1 sits at present column past_len + 1 = 9.
        assert_eq!(rd(k.key_bytes(1), 0, 4), 9);
        assert_eq!(
            rd(k.key_bytes(1), 1, 4),
            109,
            "head 1 lands in head 1's row"
        );
        assert!(
            k.key_bytes(1)[..4 * 4 * 2].iter().all(|&b| b == 0),
            "slot 0 untouched"
        );
        assert_eq!(k.key_bytes(0)[4 * 2 * 2], 0x11, "layer 0 by its own width");
        k.advance(1, 1);
        k.reset_slot(1);
        assert!(k.key_bytes(0).iter().all(|&b| b == 0));
        assert!(k.key_bytes(1).iter().all(|&b| b == 0));
    }

    #[test]
    fn mixed_geometry_populates_the_prefix_per_layer() {
        let mut k = PackedKv::with_layers(2, 12, 1, vec![(1, 2), (2, 4)], ShimDType::F16, 4);
        let mut p1 = vec![0u8; k.present_layer_bytes(1, k.packed_seq)];
        let col = k.past_len; // query row 0
        for h in 0..2 {
            let off = (h * k.context + col) * 4 * 2;
            p1[off..off + 8].fill(0x40 + h as u8);
        }
        k.absorb_row(1, false, &p1, k.packed_seq, 0, 0, 0);
        k.advance(0, 1);
        k.populate_prefix_from(0, 1);
        assert_eq!(k.prefix_valid, 1);
        // prefix col 0 of each head holds that head's row
        assert_eq!(k.key_bytes(1)[0], 0x40);
        assert_eq!(k.key_bytes(1)[12 * 4 * 2], 0x41);
    }

    #[test]
    fn image_ids_number_each_run() {
        let ids = [5, 9, 9, 9, 6, 7, 9, 9, 8];
        assert_eq!(
            prompt_image_ids(&ids, 9),
            vec![
                None,
                Some(0),
                Some(0),
                Some(0),
                None,
                None,
                Some(1),
                Some(1),
                None
            ]
        );
        assert!(prompt_image_ids(&[1, 2], 9).iter().all(Option::is_none));
    }

    #[test]
    fn packed_chunks_never_split_an_image() {
        let imgs = prompt_image_ids(&[1, 9, 9, 9, 2, 3], 9);
        assert_eq!(image_safe_take(&imgs, 0, 3), Ok(1)); // would split 1..4
        assert_eq!(image_safe_take(&imgs, 0, 4), Ok(4)); // ends after it
        assert_eq!(image_safe_take(&imgs, 1, 3), Ok(3)); // exactly the image
        assert!(image_safe_take(&imgs, 1, 2).is_err()); // wider than the chunk
        assert_eq!(image_safe_take(&imgs, 4, 2), Ok(2));
        assert_eq!(image_safe_take(&[], 0, 4), Ok(4)); // no image blocks
                                                       // two adjacent images (different ids) may be cut between them
        let two = [Some(0), Some(0), Some(1), Some(1)];
        assert_eq!(image_safe_take(&two, 0, 2), Ok(2));
        assert_eq!(image_safe_take(&two, 0, 3), Ok(2));
    }
}
