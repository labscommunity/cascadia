//! Prompt-prefix KV cache for GLM-5.2 (single-process).
//!
//! Keyed by prompt token prefix, stores the per-layer KV snapshot ([`AttnKv`])
//! taken after prefilling that prefix. A later request that shares the prefix
//! restores it and prefills only the new suffix — skipping the expert-weight
//! streaming a full re-prefill repeats, which a profiled decode shows is ~78% of
//! the time on this NVMe-bound workload. Agentic loops are the target: the
//! system prompt + accumulated history is re-sent every step.
//!
//! Bounded LRU (MRU at the back). Snapshots cost RAM — but GLM's MLA latent KV
//! is compressed, so a cached prefix is far cheaper than a full-KV model would
//! be. This is the single-process store; the cross-rank per-stage snapshot
//! exchange (the `PipelineEngine` deployment) is layered on top.

use super::attn::AttnKv;

pub struct KvPrefixCache {
    /// `(prefix tokens, KV snapshot for [0, tokens.len()))`. LRU: MRU at the back.
    entries: Vec<(Vec<u32>, Vec<AttnKv>)>,
    cap: usize,
}

impl KvPrefixCache {
    /// A cache holding at most `cap` prefixes (>= 1).
    pub fn new(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap: cap.max(1),
        }
    }

    /// The longest cached prefix of `prompt` and its KV snapshot, or `None`.
    /// Only returns a *strict-or-equal* token prefix, so restoring it and
    /// prefilling `prompt[k..]` is bit-identical to a full prefill.
    pub fn longest_prefix(&self, prompt: &[u32]) -> Option<(usize, &Vec<AttnKv>)> {
        let mut best: Option<(usize, &Vec<AttnKv>)> = None;
        for (toks, snap) in &self.entries {
            let k = toks.len();
            if k <= prompt.len() && prompt[..k] == toks[..] && best.map_or(true, |(b, _)| k > b) {
                best = Some((k, snap));
            }
        }
        best
    }

    /// Insert (or refresh to MRU) the snapshot for `tokens`. Evicts the LRU entry
    /// when over capacity.
    pub fn insert(&mut self, tokens: Vec<u32>, snap: Vec<AttnKv>) {
        if let Some(i) = self.entries.iter().position(|(t, _)| *t == tokens) {
            self.entries.remove(i);
        }
        self.entries.push((tokens, snap));
        while self.entries.len() > self.cap {
            self.entries.remove(0);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Per-rank slice KV cache for the distributed (pipeline) path — keyed by a
/// `u64` prefix key that rank 0 assigns (a hash of the prompt prefix) so every
/// rank caches/restores the same prefix in lockstep. Stores this rank's
/// layer-slice KV snapshot. LRU-bounded; `cap == 0` disables it (always misses).
pub struct SliceKvCache {
    entries: Vec<(u64, Vec<AttnKv>)>,
    cap: usize,
}

impl SliceKvCache {
    pub fn new(cap: usize) -> Self {
        Self {
            entries: Vec::new(),
            cap,
        }
    }

    /// Whether caching is on (`cap > 0`).
    pub fn enabled(&self) -> bool {
        self.cap > 0
    }

    /// This rank's cached slice snapshot for `key`, if present.
    pub fn get(&self, key: u64) -> Option<&Vec<AttnKv>> {
        self.entries.iter().find(|(k, _)| *k == key).map(|(_, s)| s)
    }

    /// Remove and return the snapshot for `key` (lets the caller restore without
    /// holding a borrow of the cache, then re-`insert`).
    pub fn take(&mut self, key: u64) -> Option<Vec<AttnKv>> {
        self.entries
            .iter()
            .position(|(k, _)| *k == key)
            .map(|i| self.entries.remove(i).1)
    }

    /// Insert (or refresh to MRU) the snapshot for `key`. Evicts the LRU entry
    /// over capacity; a no-op store when `cap == 0`.
    pub fn insert(&mut self, key: u64, snap: Vec<AttnKv>) {
        if let Some(i) = self.entries.iter().position(|(k, _)| *k == key) {
            self.entries.remove(i);
        }
        self.entries.push((key, snap));
        while self.entries.len() > self.cap {
            self.entries.remove(0);
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(len: usize) -> Vec<AttnKv> {
        // A one-layer snapshot is enough to exercise the cache bookkeeping.
        vec![AttnKv::empty(len)]
    }

    #[test]
    fn slice_cache_get_take_lru() {
        let mut c = SliceKvCache::new(2);
        assert!(c.get(1).is_none());
        c.insert(1, kv(3));
        c.insert(2, kv(4));
        assert_eq!(c.get(1).unwrap()[0].len(), 3);
        // exceed cap -> evict LRU (key 1)
        c.insert(3, kv(5));
        assert!(c.get(1).is_none(), "key 1 should be evicted");
        assert!(c.get(2).is_some() && c.get(3).is_some());
        // take removes
        let t = c.take(2).unwrap();
        assert_eq!(t[0].len(), 4);
        assert!(c.get(2).is_none());
    }

    #[test]
    fn slice_cache_disabled_when_cap_zero() {
        let mut c = SliceKvCache::new(0);
        assert!(!c.enabled());
        c.insert(1, kv(3));
        assert!(c.get(1).is_none(), "cap 0 stores nothing");
    }

    #[test]
    fn prefix_cache_longest_match() {
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], Vec::new());
        c.insert(vec![1, 2], Vec::new());
        // longest prefix of [1,2,3,4] is [1,2,3]
        assert_eq!(c.longest_prefix(&[1, 2, 3, 4]).unwrap().0, 3);
        // [1,2,9] only matches [1,2]
        assert_eq!(c.longest_prefix(&[1, 2, 9]).unwrap().0, 2);
        // [5,6] matches nothing
        assert!(c.longest_prefix(&[5, 6]).is_none());
    }
}
