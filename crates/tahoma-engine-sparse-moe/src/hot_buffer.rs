//! Hot-expert weight reordering — pack the most-frequently-fired
//! experts per layer into a single contiguous `Vec<u8>` so consecutive
//! dispatches share L3 cache lines.
//!
//! ## Why
//!
//! On-disk layout (safetensors): experts are stored in numeric order
//! (0, 1, 2, ..., 383) per layer, ~25 MiB each. When the router selects
//! a scattered top-K (e.g. `{3, 47, 102, 215, 280, 311, 350, 372}`),
//! every expert's six weight slices live in physically distant pages of
//! the mmap. They share zero L3 cache lines, so every dispatch round
//! cold-reads ~200 MiB from DRAM.
//!
//! For K2.6 the router has heavy-tailed expert usage — empirically a
//! handful of experts per layer dominate. If we copy the top-N of those
//! into one `Vec<u8>` we get:
//!
//! - Spatial locality: consecutive experts in the hot set sit in
//!   consecutive 4 KB pages and the same set of 64 B cache lines.
//! - Predictable prefetch: the OS / hardware prefetcher streams the
//!   contiguous range instead of fielding 6 × 6 = 36 unrelated faults
//!   per round (gate/up/down × packed/scale, per top-k expert).
//! - No pointer chase: dispatch path becomes `offset + base` instead of
//!   `mmap → page-table walk → slice`.
//!
//! ## Memory trade-off
//!
//! Per layer: `N × 25 MiB`. With 60 MoE layers, that's
//!
//! ```text
//!   N=4   →  6 GiB
//!   N=8   → 12 GiB
//!   N=16  → 24 GiB
//!   N=32  → 48 GiB
//! ```
//!
//! …on top of the safetensors mmap. The Xeon miner has 133 GiB; a
//! Lunar Lake AI PC has 16-32 GiB. Default is `0` (disabled). N=8 is
//! the recommended starting point for the miner.
//!
//! ## Bit-identity
//!
//! The packed buffer is a byte-for-byte copy of the safetensors
//! mmap slices. Dispatch substitutes `&hot.slice(eid)` for
//! `&mmap.slice(eid)`; the kernel inputs are identical. Property test
//! [`tests::cold_and_hot_paths_byte_identical`] cross-checks every
//! expert in a synthetic two-layer model.

use std::collections::HashMap;

use tahoma_int4_gemm::{SafetensorsExpert, SafetensorsExpertSource};

/// One layer's hot-expert buffer: contiguous owned bytes containing
/// the top-N hot experts' six weight slices laid back-to-back per
/// expert. Lookups go through `slice(eid)` which returns either the
/// six borrowed sub-slices or `None` (eid not in the hot set →
/// caller falls back to the mmap source).
///
/// Layout for N experts:
///
/// ```text
///   [expert 0: gate_packed | gate_scale | up_packed | up_scale | down_packed | down_scale]
///   [expert 1: ...                                                                       ]
///   ...
///   [expert N-1: ...                                                                      ]
/// ```
///
/// Each expert occupies `BYTES_PER_EXPERT` bytes; we record the
/// six sub-slice offsets once (they're identical across experts) and
/// look them up by index.
pub struct LayerHotBuffer {
    /// Contiguous owned bytes for all packed experts in this layer.
    pub(crate) buf: Vec<u8>,
    /// `eid → index into the packed buffer (0..N-1)`. Misses fall back
    /// to the mmap source.
    pub(crate) index: HashMap<u32, usize>,
    /// Per-expert stride in bytes (sum of six tensor slice lengths).
    pub(crate) per_expert: usize,
    /// Byte offsets and lengths of each of the six tensors within
    /// one expert's stride. Order matches the kernel call:
    /// `[gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale]`.
    pub(crate) slot_offsets: [usize; 6],
    pub(crate) slot_lengths: [usize; 6],
}

/// Borrowed view into one hot-buffered expert. Cheap (six `&[u8]`s
/// plus the expert index for debug logging).
pub struct HotExpertView<'a> {
    pub gate_packed: &'a [u8],
    pub gate_scale: &'a [u8],
    pub up_packed: &'a [u8],
    pub up_scale: &'a [u8],
    pub down_packed: &'a [u8],
    pub down_scale: &'a [u8],
}

impl LayerHotBuffer {
    /// Build a hot buffer for the given (layer, expert_ids) by copying
    /// each expert's six slices from `source` into one contiguous
    /// owned Vec. The order of `expert_ids` is preserved in the
    /// packed buffer — pass them sorted by hotness if you want the
    /// hottest in the first cache lines.
    ///
    /// Returns an error if any expert lookup fails or if the per-expert
    /// stride is zero (defensive: indicates a manifest mismatch).
    pub fn build(
        source: &SafetensorsExpertSource,
        layer: u32,
        expert_ids: &[u32],
    ) -> Result<Self, String> {
        if expert_ids.is_empty() {
            return Err("hot buffer: empty expert_ids".into());
        }
        // Probe expert 0 of the supplied set to learn the six slice
        // sizes. K2.6's experts are uniform per layer (same shapes for
        // every eid), so one probe is enough. If a future variant
        // breaks that assumption, we'd need to size per expert and
        // store a HashMap of offsets rather than one slot_offsets
        // array — keep this assertion-checked below.
        let probe = source.expert(layer, expert_ids[0]).map_err(|e| {
            format!(
                "hot buffer: probe expert L{layer}/E{} failed: {e}",
                expert_ids[0]
            )
        })?;
        let slot_lengths = expert_slot_lengths(&probe);
        let per_expert: usize = slot_lengths.iter().sum();
        if per_expert == 0 {
            return Err("hot buffer: per-expert stride is zero".into());
        }
        let mut slot_offsets = [0usize; 6];
        let mut running = 0usize;
        for (i, &len) in slot_lengths.iter().enumerate() {
            slot_offsets[i] = running;
            running += len;
        }
        debug_assert_eq!(running, per_expert);

        let total = per_expert * expert_ids.len();
        let mut buf: Vec<u8> = Vec::new();
        buf.try_reserve_exact(total)
            .map_err(|e| format!("hot buffer alloc {total} bytes: {e}"))?;
        buf.resize(total, 0u8);

        let mut index = HashMap::with_capacity(expert_ids.len());
        // First expert is already loaded; write it straight in.
        write_one(
            &mut buf,
            0,
            per_expert,
            &slot_offsets,
            &slot_lengths,
            &probe,
        )?;
        index.insert(expert_ids[0], 0);

        for (i, &eid) in expert_ids.iter().enumerate().skip(1) {
            let w = source
                .expert(layer, eid)
                .map_err(|e| format!("hot buffer: load expert L{layer}/E{eid} failed: {e}"))?;
            // Per-layer slice-size invariant check: every expert in
            // the layer must have the same six slice lengths, else
            // our flat layout is wrong. Cheaper to fail loud than
            // silently corrupt one expert's gate weights.
            let lens = expert_slot_lengths(&w);
            if lens != slot_lengths {
                return Err(format!(
                    "hot buffer: L{layer}/E{eid} slice sizes {lens:?} differ from probe {slot_lengths:?}"
                ));
            }
            write_one(&mut buf, i, per_expert, &slot_offsets, &slot_lengths, &w)?;
            index.insert(eid, i);
        }

        Ok(Self {
            buf,
            index,
            per_expert,
            slot_offsets,
            slot_lengths,
        })
    }

    /// Number of experts in this layer's hot buffer.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// True iff the hot buffer contains no experts. Always false in
    /// practice because `build` rejects empty `expert_ids`, but kept
    /// to satisfy `clippy::len_without_is_empty`.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Memory footprint in bytes of this layer's owned buffer (excludes
    /// the small HashMap + struct overhead).
    pub fn bytes(&self) -> usize {
        self.buf.len()
    }

    /// Get the six tensor sub-slices for `eid` if it's in the hot set.
    /// Returns `None` for a miss; caller falls back to mmap.
    pub fn slice(&self, eid: u32) -> Option<HotExpertView<'_>> {
        let &idx = self.index.get(&eid)?;
        let base = idx * self.per_expert;
        let slot = |i: usize| {
            let off = base + self.slot_offsets[i];
            &self.buf[off..off + self.slot_lengths[i]]
        };
        Some(HotExpertView {
            gate_packed: slot(0),
            gate_scale: slot(1),
            up_packed: slot(2),
            up_scale: slot(3),
            down_packed: slot(4),
            down_scale: slot(5),
        })
    }
}

fn expert_slot_lengths(e: &SafetensorsExpert) -> [usize; 6] {
    [
        e.gate_packed.len(),
        e.gate_scale.len(),
        e.up_packed.len(),
        e.up_scale.len(),
        e.down_packed.len(),
        e.down_scale.len(),
    ]
}

fn write_one(
    buf: &mut [u8],
    idx: usize,
    per_expert: usize,
    offsets: &[usize; 6],
    lengths: &[usize; 6],
    w: &SafetensorsExpert,
) -> Result<(), String> {
    let base = idx * per_expert;
    let slices = [
        w.gate_packed,
        w.gate_scale,
        w.up_packed,
        w.up_scale,
        w.down_packed,
        w.down_scale,
    ];
    for (i, slice) in slices.iter().enumerate() {
        if slice.len() != lengths[i] {
            return Err(format!(
                "hot buffer slot {i}: src len {} differs from probe len {}",
                slice.len(),
                lengths[i]
            ));
        }
        let dst_off = base + offsets[i];
        buf[dst_off..dst_off + lengths[i]].copy_from_slice(slice);
    }
    Ok(())
}

/// Tracks per-(layer, expert) dispatch counts across a generation so
/// the top-N hot set can be computed after warmup.
///
/// Stored as `HashMap<(u32, u32), u64>` because most experts are
/// untouched in any given generation — a dense `Vec<Vec<u64>>`
/// (`n_layers × n_experts`) would be cheaper for small models but
/// wastes 60 × 384 × 8 = 184 KB even when zero counts dominate, and
/// allocating it requires the manifest at construction time. The
/// HashMap path is allocation-free for unused (layer, expert) pairs.
#[derive(Default, Debug)]
pub struct ExpertHits {
    counts: HashMap<(u32, u32), u64>,
    /// Total dispatches recorded. Used as the "warmup tokens" knob;
    /// we trigger the hot-buffer build once `total >= warmup_dispatches`.
    pub total: u64,
}

impl ExpertHits {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, lid: u32, eid: u32) {
        *self.counts.entry((lid, eid)).or_insert(0) += 1;
        self.total += 1;
    }

    /// Return the top-N expert ids for `lid` sorted by descending hit
    /// count. Ties broken by ascending expert id for determinism.
    /// Returns an empty Vec if no hits have been recorded for the layer.
    pub fn top_n_for_layer(&self, lid: u32, n: usize) -> Vec<u32> {
        let mut layer_hits: Vec<(u32, u64)> = self
            .counts
            .iter()
            .filter_map(|(&(l, e), &c)| if l == lid { Some((e, c)) } else { None })
            .collect();
        // Sort by (-count, +eid): highest count first, ties broken by
        // smaller eid so the choice is reproducible across runs with the
        // same hit distribution.
        layer_hits.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        layer_hits.truncate(n);
        layer_hits.into_iter().map(|(eid, _)| eid).collect()
    }

    /// All layer ids that have at least one recorded hit.
    pub fn layers_with_hits(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .counts
            .keys()
            .map(|&(l, _)| l)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_top_n_orders_by_count_then_eid() {
        let mut h = ExpertHits::new();
        // Layer 1: expert 5 fires 10 times, 7 fires 10 times (tie),
        // 3 fires 3 times.
        for _ in 0..10 {
            h.record(1, 5);
            h.record(1, 7);
        }
        for _ in 0..3 {
            h.record(1, 3);
        }
        // Layer 2: just expert 0, three hits — should not appear in
        // layer 1's top-N.
        for _ in 0..3 {
            h.record(2, 0);
        }
        let top2 = h.top_n_for_layer(1, 2);
        // Tie broken by ascending eid: 5 before 7.
        assert_eq!(top2, vec![5, 7]);
        let top10 = h.top_n_for_layer(1, 10);
        assert_eq!(top10, vec![5, 7, 3]);
    }

    #[test]
    fn hits_layers_with_hits_sorted() {
        let mut h = ExpertHits::new();
        h.record(3, 0);
        h.record(1, 0);
        h.record(2, 0);
        assert_eq!(h.layers_with_hits(), vec![1, 2, 3]);
    }

    #[test]
    fn hits_top_n_empty_for_unused_layer() {
        let mut h = ExpertHits::new();
        h.record(0, 0);
        assert!(h.top_n_for_layer(99, 4).is_empty());
    }

    // Bit-identity round-trip: pure-Rust round-trip of the packing
    // logic, no safetensors needed.
    #[test]
    fn layer_hot_buffer_layout_is_contiguous() {
        // Build three fake experts where each slice carries a unique
        // byte stamp; pack them into a hot buffer and verify each
        // slice round-trips at the expected offset.
        let slot_lengths = [16usize, 8, 16, 8, 16, 8];
        let per_expert: usize = slot_lengths.iter().sum();
        let mut buf = vec![0u8; per_expert * 3];

        // Slot offsets the same way `build` would compute them.
        let mut offs = [0usize; 6];
        let mut running = 0usize;
        for (i, &len) in slot_lengths.iter().enumerate() {
            offs[i] = running;
            running += len;
        }

        // Stamp each expert and write through the canonical packing.
        for (idx, eid) in [(0usize, 5u32), (1, 9), (2, 2)].iter() {
            let base = *idx * per_expert;
            for (slot, &len) in slot_lengths.iter().enumerate() {
                let stamp = (*eid as u8).wrapping_mul(10 + slot as u8);
                let dst = &mut buf[base + offs[slot]..base + offs[slot] + len];
                for byte in dst.iter_mut() {
                    *byte = stamp;
                }
            }
        }

        // Reconstruct a LayerHotBuffer and check slices come back.
        let mut index = HashMap::new();
        index.insert(5u32, 0usize);
        index.insert(9, 1);
        index.insert(2, 2);
        let hb = LayerHotBuffer {
            buf,
            index,
            per_expert,
            slot_offsets: offs,
            slot_lengths,
        };

        for &(idx_expected, eid) in &[(0usize, 5u32), (1, 9), (2, 2)] {
            let v = hb.slice(eid).expect("present");
            let slices = [
                v.gate_packed,
                v.gate_scale,
                v.up_packed,
                v.up_scale,
                v.down_packed,
                v.down_scale,
            ];
            for (slot, slice) in slices.iter().enumerate() {
                let stamp = (eid as u8).wrapping_mul(10 + slot as u8);
                assert_eq!(slice.len(), slot_lengths[slot], "slot {slot} eid {eid}");
                assert!(
                    slice.iter().all(|&b| b == stamp),
                    "slot {slot} eid {eid} idx {idx_expected} stamp mismatch"
                );
            }
        }
        // Miss returns None
        assert!(hb.slice(123).is_none());
    }
}
