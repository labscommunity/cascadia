//! Optional model-owned packed routed-weight cache. Admission is driven mainly
//! by past routing frequency, with recency (last-seen request) breaking
//! frequency ties on eviction and an opt-in recency-admission mode
//! (`CASCADIA_INKLING_CACHE_RECENT_TIES`) that admits on a frequency tie.
//! Entries are immutable while leased; no lock covers I/O or GEMV.
use std::sync::{Arc, Mutex};

use super::read_buffers::ReadBuffer;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ExpertCacheStats {
    pub capacity_bytes: usize,
    pub retained_bytes: usize,
    pub hits: u64,
    pub hit_bytes: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub history_resets: u64,
    pub frequency_decays: u64,
    pub recent_tie_admissions: u64,
}

impl ExpertCacheStats {
    pub(crate) fn add(&mut self, other: Self) {
        self.capacity_bytes += other.capacity_bytes;
        self.retained_bytes += other.retained_bytes;
        self.hits += other.hits;
        self.hit_bytes += other.hit_bytes;
        self.misses += other.misses;
        self.admissions += other.admissions;
        self.evictions += other.evictions;
        self.history_resets += other.history_resets;
        self.frequency_decays += other.frequency_decays;
        self.recent_tie_admissions += other.recent_tie_admissions;
    }
}

struct Entry {
    expert: usize,
    bytes: Arc<ReadBuffer>,
}

struct State {
    frequency: Vec<u64>,
    last: Vec<u64>,
    clock: u64,
    decay_requests: u64,
    recent_ties: bool,
    entries: Vec<Entry>,
    stats: ExpertCacheStats,
}

pub(super) struct ExpertCache(Mutex<State>);

impl ExpertCache {
    pub fn new(experts: usize, capacity: usize) -> Self {
        let decay_requests = std::env::var("CASCADIA_INKLING_CACHE_DECAY_REQUESTS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| (4..=65536).contains(n) && n.is_power_of_two())
            .unwrap_or(4096);
        Self::with_policy(
            experts,
            capacity,
            decay_requests,
            super::env_flag("CASCADIA_INKLING_CACHE_RECENT_TIES"),
        )
    }

    #[cfg(test)]
    fn with_decay(experts: usize, capacity: usize, decay_requests: u64) -> Self {
        Self::with_policy(experts, capacity, decay_requests, false)
    }

    fn with_policy(
        experts: usize,
        capacity: usize,
        decay_requests: u64,
        recent_ties: bool,
    ) -> Self {
        Self(Mutex::new(State {
            frequency: vec![0; experts],
            last: vec![0; experts],
            clock: 0,
            decay_requests,
            recent_ties,
            entries: Vec::new(),
            stats: ExpertCacheStats {
                capacity_bytes: capacity,
                ..ExpertCacheStats::default()
            },
        }))
    }

    /// Per-MoE-layer MiB; default zero. Invalid or >16384 MiB settings
    /// disable it (256 MiB × 64 layers = the 16 GiB the campaign-129 profile
    /// runs on a paged 64 GB box; a resident pipeline rank sets it at or
    /// above a layer's 7.7 GiB of experts so its slice stays in RAM).
    /// Allocations grow only on successful decode reads, never during
    /// prefill.
    pub fn configured_bytes() -> usize {
        std::env::var("CASCADIA_INKLING_EXPERT_CACHE_MIB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&mib| mib <= 16384)
            .unwrap_or(0)
            * 1024
            * 1024
    }

    pub fn stats(&self) -> ExpertCacheStats {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).stats
    }

    /// Select a prediction using only current cache membership. Unlike lookup,
    /// this neither observes routing nor changes admission history/counters.
    pub fn first_uncached(&self, predictions: &[usize]) -> Option<usize> {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.stats.capacity_bytes == 0 {
            return None;
        }
        predictions.iter().copied().find(|&expert| {
            expert < state.frequency.len() && !state.entries.iter().any(|e| e.expert == expert)
        })
    }

    /// Keep the first uncached prediction, adding the second only when its
    /// original gate rank is within the supplied ceiling. Inspect membership once,
    /// before actual lookup, without changing cache history or counters.
    pub fn selective_uncached(
        &self,
        predictions: &[usize],
        second_rank_ceiling: usize,
    ) -> [Option<usize>; 2] {
        let [first, second, _] = self.predicted_uncached(predictions, second_rank_ceiling, false);
        [first, second]
    }

    /// The third read is eligible only at original predicted rank2: all three
    /// leading predictions must be distinct, valid and currently uncached.
    pub fn predicted_uncached(
        &self,
        predictions: &[usize],
        second_rank_ceiling: usize,
        third: bool,
    ) -> [Option<usize>; 3] {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut selected = [None, None, None];
        if state.stats.capacity_bytes == 0 {
            return selected;
        }
        for (rank, &expert) in predictions.iter().enumerate() {
            if expert >= state.frequency.len()
                || state.entries.iter().any(|entry| entry.expert == expert)
                || selected.contains(&Some(expert))
            {
                continue;
            }
            if selected[0].is_none() {
                selected[0] = Some(expert);
            } else if selected[1].is_none() {
                if rank > second_rank_ceiling {
                    break;
                }
                selected[1] = Some(expert);
                if !third {
                    break;
                }
            } else {
                if rank <= 2 {
                    selected[2] = Some(expert);
                }
                break;
            }
        }
        selected
    }

    /// Forget a previous sequence's admission preferences, retaining its valid
    /// weight allocations. Counters remain cumulative for workload accounting.
    pub fn reset_history(&self) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.stats.capacity_bytes != 0 {
            state.frequency.fill(0);
            state.last.fill(0);
            state.clock = 0;
            state.stats.history_resets += 1;
        }
    }

    /// Called in gate order before parallel compute, making admission history
    /// independent of read completion order. Return None for the disabled path.
    pub fn lookup(&self, experts: &[usize]) -> Option<Vec<Option<Arc<ReadBuffer>>>> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.stats.capacity_bytes == 0 {
            return None;
        }
        let mut found = Vec::with_capacity(experts.len());
        for &expert in experts {
            assert!(expert < state.frequency.len());
            state.clock = state.clock.saturating_add(1);
            // Bounded history adapts to changing requests without future routing.
            if state.clock.is_multiple_of(state.decay_requests) {
                for count in &mut state.frequency {
                    *count = count.div_ceil(2);
                }
                state.stats.frequency_decays += 1;
            }
            state.frequency[expert] = state.frequency[expert].saturating_add(1);
            state.last[expert] = state.clock;
            let bytes = state
                .entries
                .iter()
                .find(|e| e.expert == expert)
                .map(|e| e.bytes.clone());
            if let Some(bytes) = &bytes {
                state.stats.hits += 1;
                state.stats.hit_bytes += bytes.as_slice().len() as u64;
            } else {
                state.stats.misses += 1;
            }
            found.push(bytes);
        }
        Some(found)
    }

    /// Transfer a fully read allocation after its kernel has completed. An
    /// evicted allocation returns to the caller's scratch lease for reuse.
    /// Never evict an entry with an outstanding lease: retained plus leased
    /// expert allocations therefore stay within this layer's capacity.
    /// Put `expert`'s bytes in the cache before it was ever routed to (a rank
    /// filling its resident copy at startup). Takes free capacity only: never
    /// evicts, never counts as a hit, miss or admission. `false` when it did
    /// not fit or was already there.
    pub fn preload(&self, expert: usize, bytes: &mut ReadBuffer) -> bool {
        let size = bytes.allocated_bytes();
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if bytes.as_slice().is_empty()
            || expert >= state.frequency.len()
            || size > state.stats.capacity_bytes - state.stats.retained_bytes
            || state.entries.iter().any(|e| e.expert == expert)
        {
            return false;
        }
        let incoming = std::mem::take(bytes);
        state.stats.retained_bytes += size;
        state.entries.push(Entry {
            expert,
            bytes: Arc::new(incoming),
        });
        true
    }

    pub fn retain(&self, expert: usize, bytes: &mut ReadBuffer) {
        let size = bytes.allocated_bytes();
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if bytes.as_slice().is_empty()
            || size > state.stats.capacity_bytes
            || expert >= state.frequency.len()
            || state.frequency[expert] == 0
            || state.entries.iter().any(|e| e.expert == expert)
        {
            return;
        }
        let mut replacement = ReadBuffer::default();
        if size > state.stats.capacity_bytes - state.stats.retained_bytes {
            let victim = state
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| Arc::strong_count(&e.bytes) == 1)
                .filter(|(_, e)| {
                    size <= state.stats.capacity_bytes - state.stats.retained_bytes
                        + e.bytes.allocated_bytes()
                })
                .min_by_key(|(_, e)| (state.frequency[e.expert], state.last[e.expert]))
                .map(|(i, _)| i);
            let Some(victim) = victim else { return };
            let old_expert = state.entries[victim].expert;
            let incoming_frequency = state.frequency[expert];
            let old_frequency = state.frequency[old_expert];
            let recent_tie = state.recent_ties
                && incoming_frequency == old_frequency
                && state.last[expert] > state.last[old_expert];
            if incoming_frequency <= old_frequency && !recent_tie {
                return;
            }
            let evicted = state.entries.swap_remove(victim);
            // lookup is serialized by this mutex; no new lease can appear.
            replacement = Arc::try_unwrap(evicted.bytes).unwrap_or_else(|_| unreachable!());
            state.stats.retained_bytes -= replacement.allocated_bytes();
            state.stats.evictions += 1;
            state.stats.recent_tie_admissions += u64::from(recent_tie);
        }
        let incoming = std::mem::replace(bytes, replacement);
        state.stats.retained_bytes += size;
        state.stats.admissions += 1;
        state.entries.push(Entry {
            expert,
            bytes: Arc::new(incoming),
        });
        debug_assert!(state.stats.retained_bytes <= state.stats.capacity_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn bytes(value: u8, length: usize) -> ReadBuffer {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![value; length]).unwrap();
        let mut buffer = ReadBuffer::default();
        buffer.read(file.path(), length).unwrap();
        buffer
    }

    #[test]
    fn disabled_cache_and_unobserved_or_invalid_reads_retain_nothing() {
        let off = ExpertCache::new(2, 0);
        assert!(off.lookup(&[0]).is_none());
        let cache = ExpertCache::new(2, 32);
        cache.retain(0, &mut bytes(17, 32));
        assert_eq!(cache.stats().retained_bytes, 0);
        drop(cache.lookup(&[0]));
        cache.retain(0, &mut ReadBuffer::default());
        cache.retain(0, &mut bytes(17, 64));
        assert_eq!(cache.stats().retained_bytes, 0);
    }

    #[test]
    fn preload_fills_free_capacity_and_is_served_as_hits() {
        let cache = ExpertCache::new(4, 64); // room for two 32-byte experts
        assert!(
            cache.preload(1, &mut bytes(11, 32)),
            "never routed to, still loaded"
        );
        assert!(cache.preload(3, &mut bytes(33, 32)));
        assert!(
            !cache.preload(0, &mut bytes(7, 32)),
            "full: preload never evicts"
        );
        assert!(!cache.preload(1, &mut bytes(11, 32)), "already there");
        assert!(!cache.preload(9, &mut bytes(9, 32)), "out of range");
        let st = cache.stats();
        assert_eq!((st.retained_bytes, st.admissions, st.evictions), (64, 0, 0));
        let hits = cache.lookup(&[1, 0, 3]).expect("cache on");
        assert_eq!(hits[0].as_ref().map(|b| b.as_slice()[0]), Some(11));
        assert!(hits[1].is_none());
        assert_eq!(hits[2].as_ref().map(|b| b.as_slice()[0]), Some(33));
        assert_eq!((cache.stats().hits, cache.stats().misses), (2, 1));
    }

    #[test]
    fn retain_declines_out_of_range_expert_ids_without_panicking() {
        let cache = ExpertCache::new(2, 32);
        drop(cache.lookup(&[0, 1]));
        // An id at or beyond n_routed (a router/n_routed divergence, or a
        // future gate emitting shared-expert ids) must be declined like the
        // prediction siblings skip it, never used to index frequency/last.
        cache.retain(2, &mut bytes(17, 32));
        cache.retain(9, &mut bytes(93, 32));
        assert_eq!(cache.stats().retained_bytes, 0);
        assert_eq!(cache.stats().admissions, 0);
        // A valid id still admits, proving the guard did not close the path.
        cache.retain(0, &mut bytes(61, 32));
        assert_eq!(cache.stats().retained_bytes, 32);
        assert_eq!(cache.stats().admissions, 1);
    }

    #[test]
    fn predicted_membership_query_preserves_history_and_actual_counters() {
        let cache = ExpertCache::with_policy(3, 32, 4, true);
        drop(cache.lookup(&[1]));
        cache.retain(1, &mut bytes(17, 32));
        let before = serde_json::to_value(cache.stats()).unwrap();
        let history = {
            let s = cache.0.lock().unwrap();
            (s.frequency.clone(), s.last.clone(), s.clock)
        };
        assert_eq!(cache.first_uncached(&[1, 2, 0]), Some(2));
        assert_eq!(cache.first_uncached(&[1]), None);
        assert_eq!(cache.first_uncached(&[9, 0]), Some(0));
        assert_eq!(serde_json::to_value(cache.stats()).unwrap(), before);
        let s = cache.0.lock().unwrap();
        assert_eq!((s.frequency.clone(), s.last.clone(), s.clock), history);
        assert_eq!(
            ExpertCache::with_policy(3, 0, 4, true).first_uncached(&[0]),
            None
        );
    }

    #[test]
    fn selective_prediction_respects_original_rank_and_preserves_history() {
        let cache = ExpertCache::with_policy(6, 96, 4, true);
        drop(cache.lookup(&[0, 1, 2]));
        for expert in 0..3 {
            cache.retain(expert, &mut bytes(17, 32));
        }
        let before = serde_json::to_value(cache.stats()).unwrap();
        let history = {
            let s = cache.0.lock().unwrap();
            (s.frequency.clone(), s.last.clone(), s.clock)
        };
        assert_eq!(cache.selective_uncached(&[3, 4, 5], 2), [Some(3), Some(4)]);
        assert_eq!(
            cache.selective_uncached(&[0, 3, 4, 5], 2),
            [Some(3), Some(4)]
        );
        assert_eq!(
            cache.selective_uncached(&[3, 0, 4, 5], 2),
            [Some(3), Some(4)]
        );
        assert_eq!(cache.selective_uncached(&[0, 1, 3, 4], 2), [Some(3), None]);
        assert_eq!(
            cache.selective_uncached(&[0, 1, 2, 3, 4], 2),
            [Some(3), None]
        );
        assert_eq!(cache.selective_uncached(&[9, 3, 4], 2), [Some(3), Some(4)]);
        assert_eq!(cache.selective_uncached(&[3, 3, 4], 2), [Some(3), Some(4)]);
        assert_eq!(cache.selective_uncached(&[3, 3, 3, 4], 2), [Some(3), None]);
        assert_eq!(cache.selective_uncached(&[0, 1, 2, 9], 2), [None, None]);
        assert_eq!(cache.selective_uncached(&[], 2), [None, None]);
        assert_eq!(serde_json::to_value(cache.stats()).unwrap(), before);
        let s = cache.0.lock().unwrap();
        assert_eq!((s.frequency.clone(), s.last.clone(), s.clock), history);
        assert_eq!(
            ExpertCache::new(6, 0).selective_uncached(&[3, 4], 2),
            [None, None]
        );
    }

    #[test]
    fn second_rank_ceiling_changes_only_the_second_prediction() {
        let cache = ExpertCache::with_policy(6, 96, 4, true);
        drop(cache.lookup(&[0, 1, 2]));
        for expert in 0..3 {
            cache.retain(expert, &mut bytes(17, 32));
        }
        assert_eq!(cache.selective_uncached(&[0, 3, 4], 1), [Some(3), None]);
        assert_eq!(cache.selective_uncached(&[0, 3, 4], 2), [Some(3), Some(4)]);
        assert_eq!(cache.selective_uncached(&[0, 1, 3, 4], 2), [Some(3), None]);
        assert_eq!(
            cache.selective_uncached(&[0, 1, 3, 4], 3),
            [Some(3), Some(4)]
        );
        assert_eq!(
            cache.selective_uncached(&[0, 1, 2, 3, 3, 4], 4),
            [Some(3), None]
        );
        assert_eq!(
            cache.selective_uncached(&[0, 1, 2, 3, 3, 4], 5),
            [Some(3), Some(4)]
        );
        for ceiling in 1..=5 {
            assert_eq!(
                cache.selective_uncached(&[0, 1, 2, 3], ceiling),
                [Some(3), None]
            );
        }
    }

    #[test]
    fn third_prediction_requires_three_leading_uncached_distinct_experts() {
        let cache = ExpertCache::with_policy(8, 32, 4, true);
        drop(cache.lookup(&[0]));
        cache.retain(0, &mut bytes(17, 32));
        let before = serde_json::to_value(cache.stats()).unwrap();
        let history = {
            let s = cache.0.lock().unwrap();
            (s.frequency.clone(), s.last.clone(), s.clock)
        };
        assert_eq!(
            cache.predicted_uncached(&[1, 2, 3], 2, true),
            [Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            cache.predicted_uncached(&[1, 2, 3], 1, true),
            [Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            cache.predicted_uncached(&[1, 2, 3], 2, false),
            [Some(1), Some(2), None]
        );
        assert_eq!(
            cache.predicted_uncached(&[0, 1, 2, 3], 2, true),
            [Some(1), Some(2), None]
        );
        assert_eq!(
            cache.predicted_uncached(&[1, 1, 2, 3], 5, true),
            [Some(1), Some(2), None]
        );
        assert_eq!(
            cache.predicted_uncached(&[9, 1, 2, 3], 5, true),
            [Some(1), Some(2), None]
        );
        assert_eq!(
            cache.predicted_uncached(&[0, 1, 2, 3], 1, true),
            [Some(1), None, None]
        );
        assert_eq!(
            ExpertCache::new(8, 0).predicted_uncached(&[1, 2, 3], 2, true),
            [None, None, None]
        );
        assert_eq!(serde_json::to_value(cache.stats()).unwrap(), before);
        let s = cache.0.lock().unwrap();
        assert_eq!((s.frequency.clone(), s.last.clone(), s.clock), history);
    }

    #[test]
    fn lfu_retains_hot_expert_and_recycles_evicted_allocation() {
        let cache = ExpertCache::new(3, 64);
        drop(cache.lookup(&[0, 1]));
        cache.retain(0, &mut bytes(17, 32));
        cache.retain(1, &mut bytes(93, 32));
        drop(cache.lookup(&[0, 0, 2, 2]));
        let mut incoming = bytes(61, 32);
        cache.retain(2, &mut incoming);
        assert_eq!(incoming.as_slice(), [93; 32]);
        let hit = cache.lookup(&[0, 1, 2]).unwrap();
        assert_eq!(hit[0].as_ref().unwrap().as_slice(), [17; 32]);
        assert!(hit[1].is_none());
        assert_eq!(hit[2].as_ref().unwrap().as_slice(), [61; 32]);
        assert_eq!(cache.stats().retained_bytes, 64);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn short_frequency_history_adapts_within_a_sequence() {
        let adaptive = ExpertCache::with_decay(2, 32, 4);
        let long_history = ExpertCache::with_decay(2, 32, 4096);
        for cache in [&adaptive, &long_history] {
            drop(cache.lookup(&[0; 64]));
            cache.retain(0, &mut bytes(17, 32));
            drop(cache.lookup(&[1; 8]));
            cache.retain(1, &mut bytes(93, 32));
        }
        assert_eq!(adaptive.stats().frequency_decays, 18);
        assert_eq!(long_history.stats().frequency_decays, 0);
        let adapted = adaptive.lookup(&[0, 1]).unwrap();
        assert!(adapted[0].is_none());
        assert_eq!(adapted[1].as_ref().unwrap().as_slice(), [93; 32]);
        let old = long_history.lookup(&[0, 1]).unwrap();
        assert_eq!(old[0].as_ref().unwrap().as_slice(), [17; 32]);
        assert!(old[1].is_none());
        adaptive.reset_history();
        assert_eq!(adaptive.stats().frequency_decays, 18);
        assert_eq!(adaptive.stats().retained_bytes, 32);
    }

    #[test]
    fn recent_ties_replace_stale_entries_but_preserve_later_cohort_hits() {
        for recent in [false, true] {
            let cache = ExpertCache::with_policy(3, 32, 32, recent);
            drop(cache.lookup(&[0]));
            cache.retain(0, &mut bytes(17, 32));
            drop(cache.lookup(&[1]));
            let mut incoming = bytes(93, 32);
            cache.retain(1, &mut incoming);
            assert_eq!(cache.stats().recent_tie_admissions, u64::from(recent));
            let state = cache.0.lock().unwrap();
            assert_eq!(state.entries[0].expert, usize::from(recent));
            drop(state);
            if recent {
                assert_eq!(incoming.as_slice(), [17; 32]);
                cache.reset_history();
                // Both frequencies are one; the existing hit was used later
                // in this same cohort, so earlier missing expert2 cannot win.
                drop(cache.lookup(&[2, 1]));
                cache.retain(2, &mut bytes(61, 32));
                assert_eq!(cache.0.lock().unwrap().entries[0].expert, 1);
                assert_eq!(cache.stats().recent_tie_admissions, 1);
            }
        }
    }

    #[test]
    fn recent_ties_cannot_evict_a_leased_buffer_or_exceed_capacity() {
        let cache = ExpertCache::with_policy(2, 32, 32, true);
        drop(cache.lookup(&[0]));
        cache.retain(0, &mut bytes(17, 32));
        let held = cache.lookup(&[0]).unwrap();
        drop(cache.lookup(&[1, 1]));
        let mut incoming = bytes(93, 32);
        cache.retain(1, &mut incoming);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(held[0].as_ref().unwrap().as_slice(), [17; 32]);
        drop(held);
        cache.retain(1, &mut incoming);
        assert_eq!(cache.stats().recent_tie_admissions, 1);
        assert_eq!(cache.stats().retained_bytes, 32);
        assert_eq!(incoming.as_slice(), [17; 32]);
    }

    #[test]
    fn recent_tie_evictions_preserve_real_packed_kernel_bytes() {
        use crate::dsv4::expert_mmap::MmapExpert;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling_export/experts/layer_01");
        let experts: Vec<_> = ["expert_000.bin", "expert_001.bin"]
            .iter()
            .map(|name| MmapExpert::open(&directory.join(name), 64, 32).unwrap())
            .collect();
        let mut scratch = ReadBuffer::default();
        scratch
            .read(experts[0].bin_path(), experts[0].bin_len())
            .unwrap();
        let cache = ExpertCache::with_policy(2, scratch.allocated_bytes(), 32, true);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 7.0) * 0.03125).collect();
        for id in [0, 1, 0, 1, 0, 1, 0, 1] {
            let expert = &experts[id];
            let hit = cache.lookup(&[id]).unwrap();
            let expected = crate::glm::ffn::swiglu_mmap(expert, &x);
            let actual = if let Some(bytes) = &hit[0] {
                expert.swiglu_from(bytes.as_slice(), &x)
            } else {
                scratch.read(expert.bin_path(), expert.bin_len()).unwrap();
                expert.swiglu_from(scratch.as_slice(), &x)
            };
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let missing = hit[0].is_none();
            drop(hit);
            if missing {
                cache.retain(id, &mut scratch);
            }
        }
        assert_eq!(cache.stats().recent_tie_admissions, 4);
        assert_eq!(cache.stats().evictions, 7);
    }

    #[test]
    fn outstanding_leases_prevent_eviction_and_keep_capacity_bounded() {
        let cache = Arc::new(ExpertCache::new(2, 32));
        drop(cache.lookup(&[0]));
        cache.retain(0, &mut bytes(17, 32));
        let held = cache.lookup(&[0]).unwrap();
        let writer = cache.clone();
        std::thread::spawn(move || {
            drop(writer.lookup(&[1, 1, 1]));
            writer.retain(1, &mut bytes(93, 32));
        })
        .join()
        .unwrap();
        assert_eq!(held[0].as_ref().unwrap().as_slice(), [17; 32]);
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().retained_bytes, 32);
        drop(held);
        cache.retain(1, &mut bytes(93, 32));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn independent_models_never_share_same_numbered_expert_bytes() {
        let first = ExpertCache::new(1, 32);
        let second = ExpertCache::new(1, 32);
        drop(first.lookup(&[0]));
        drop(second.lookup(&[0]));
        first.retain(0, &mut bytes(17, 32));
        second.retain(0, &mut bytes(93, 32));
        assert_eq!(
            first.lookup(&[0]).unwrap()[0].as_ref().unwrap().as_slice(),
            [17; 32]
        );
        assert_eq!(
            second.lookup(&[0]).unwrap()[0].as_ref().unwrap().as_slice(),
            [93; 32]
        );
    }

    #[test]
    fn new_sequence_can_replace_stale_favorites_without_discarding_warm_weights() {
        let cache = ExpertCache::new(3, 64);
        drop(cache.lookup(&[0, 1]));
        cache.retain(0, &mut bytes(17, 32));
        cache.retain(1, &mut bytes(93, 32));
        for _ in 0..20 {
            drop(cache.lookup(&[0, 1]));
        }
        let hits_before = cache.stats().hits;
        cache.reset_history();
        assert_eq!(cache.stats().retained_bytes, 64);
        assert_eq!(cache.stats().hits, hits_before);
        let warm = cache.lookup(&[0]).unwrap();
        assert_eq!(warm[0].as_ref().unwrap().as_slice(), [17; 32]);
        drop(warm);
        drop(cache.lookup(&[2]));
        cache.retain(2, &mut bytes(61, 32));
        let hit = cache.lookup(&[0, 1, 2]).unwrap();
        assert!(hit[0].is_some() && hit[1].is_none() && hit[2].is_some());
        assert_eq!(cache.stats().history_resets, 1);
    }

    #[test]
    fn real_int4_kernel_remains_exact_across_cache_hits_and_evictions() {
        use crate::dsv4::expert_mmap::MmapExpert;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling_export/experts/layer_01");
        let experts: Vec<_> = ["expert_000.bin", "expert_001.bin"]
            .iter()
            .map(|name| MmapExpert::open(&directory.join(name), 64, 32).unwrap())
            .collect();
        let mut scratch = ReadBuffer::default();
        scratch
            .read(experts[0].bin_path(), experts[0].bin_len())
            .unwrap();
        let cache = ExpertCache::new(2, scratch.allocated_bytes());
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 7.0) * 0.03125).collect();
        for id in [0, 1, 1, 0, 0, 0, 1, 1, 1, 1] {
            let expert = &experts[id];
            let hit = cache.lookup(&[id]).unwrap();
            let expected = crate::glm::ffn::swiglu_mmap(expert, &x);
            let actual = if let Some(bytes) = &hit[0] {
                expert.swiglu_from(bytes.as_slice(), &x)
            } else {
                scratch.read(expert.bin_path(), expert.bin_len()).unwrap();
                expert.swiglu_from(scratch.as_slice(), &x)
            };
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let missed = hit[0].is_none();
            drop(hit);
            if missed {
                cache.retain(id, &mut scratch);
            }
        }
        assert!(cache.stats().hits > 0 && cache.stats().evictions >= 3);
        let before = cache.stats().admissions;
        assert!(scratch
            .read(&directory.join("missing.bin"), experts[0].bin_len())
            .is_err());
        cache.retain(0, &mut scratch);
        assert_eq!(cache.stats().admissions, before);
    }
}
