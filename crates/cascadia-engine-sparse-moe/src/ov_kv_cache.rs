//! f32-native KV snapshot + prefix cache for the OV-IR MoE engine (MiniMax-M2).
//!
//! The K2.6 [`crate::kv_prefix_cache::KvSnapshot`] stores KV as bf16 (`u16`)
//! sized to K2.6 head constants, so it can't hold [`crate::ov_moe::OvMoeRunner`]'s
//! host-held `f32` KV (`ov_moe::LayerKv { k/v: Vec<f32> }`). This module is the
//! OvMoe-native parallel: a snapshot clones the runner's `f32` KV verbatim
//! (lossless → a warm resume is byte-identical to a cold prefill), and the cache
//! is a small content-keyed LRU mirroring `KvPrefixCache`'s strict-prefix
//! contract. Local (multi-stage) use only — the cross-chain wire plane (which
//! serializes KV as bf16) is not wired for this engine yet.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

use crate::kv_prefix_cache::ModelFingerprint;

/// One owned layer's KV slice, `f32` verbatim from the runner
/// (`[kv_heads, seq, head_dim]` row-major, as `OvMoeRunner` holds it).
#[derive(Clone)]
pub struct OvLayerKvSlice {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub seq: usize,
}

/// A rank's KV state captured at `past_seq_len`. Holds only this rank's
/// owned layers (its pipeline-parallel slice), in `layer_ids` order.
#[derive(Clone)]
pub struct OvMoeKvSnapshot {
    pub past_seq_len: usize,
    pub layers: Vec<OvLayerKvSlice>,
}

impl OvMoeKvSnapshot {
    /// Total cached bytes — the `f32` K/V buffers only. Underestimate: no `Vec`/map overhead.
    /// Mirrors [`crate::kv_prefix_cache::KvSnapshot::approx_bytes`].
    pub fn approx_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.k.len() + l.v.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

/// Cache key: model fingerprint digest + a hash of the token-id prefix.
/// The prefix is re-compared in full on lookup, so a prefix-hash collision
/// cannot return a wrong entry.
#[derive(Clone, PartialEq, Eq)]
struct CacheKey {
    model_digest: u64,
    prefix_hash: u64,
}

struct Entry {
    prefix: Vec<i64>,
    snapshot: OvMoeKvSnapshot,
}

/// LRU KV-prefix cache for the OV-IR engine. `capacity == 0` disables it
/// (every `lookup` returns `None`, every `insert` is a no-op). Dedicated
/// rather than genericizing the shipped K2.6 `KvPrefixCache` so the
/// rig-validated cross-chain path is untouched.
pub struct OvMoeKvPrefixCache {
    capacity: usize,
    /// `front` = most-recently-used, `back` = least-recently-used.
    entries: VecDeque<(CacheKey, Entry)>,
    hits: u64,
    misses: u64,
}

impl OvMoeKvPrefixCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity.max(1)),
            hits: 0,
            misses: 0,
        }
    }

    /// True if the cache stores anything. `capacity == 0` returns false.
    pub fn enabled(&self) -> bool {
        self.capacity > 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Longest cached prefix that is a strict prefix of `prompt` (leaving
    /// at least one tail token for the generate loop) and shares `prompt`'s
    /// model fingerprint. Returns the snapshot; its `past_seq_len` is the
    /// matched length. On a hit the entry moves to MRU.
    pub fn lookup(&mut self, prompt: &[i64], fingerprint: &ModelFingerprint) -> Option<OvMoeKvSnapshot> {
        if !self.enabled() {
            return None;
        }
        let model_digest = fingerprint.digest();
        let mut best_idx: Option<usize> = None;
        let mut best_len = 0usize;
        for (i, (key, entry)) in self.entries.iter().enumerate() {
            if key.model_digest != model_digest {
                continue;
            }
            let plen = entry.prefix.len();
            // Strict prefix: must leave >=1 tail token for prefill to sample
            // from, else the first-token logic has nothing to drive.
            if plen >= prompt.len() {
                continue;
            }
            if plen > best_len && prompt.starts_with(&entry.prefix) {
                best_len = plen;
                best_idx = Some(i);
            }
        }
        match best_idx {
            None => {
                self.misses += 1;
                None
            }
            Some(i) => {
                self.hits += 1;
                let entry = self.entries.remove(i).expect("index from enumerate must be valid");
                let snapshot = entry.1.snapshot.clone();
                self.entries.push_front(entry);
                Some(snapshot)
            }
        }
    }

    /// Insert a snapshot keyed by `prefix` + fingerprint. Replaces an
    /// exact-key entry in place; evicts LRU entries until under capacity.
    pub fn insert(&mut self, prefix: Vec<i64>, fingerprint: &ModelFingerprint, snapshot: OvMoeKvSnapshot) {
        if !self.enabled() {
            return;
        }
        debug_assert_eq!(prefix.len(), snapshot.past_seq_len);
        let key = CacheKey {
            model_digest: fingerprint.digest(),
            prefix_hash: hash_prefix(&prefix),
        };
        if let Some(pos) = self
            .entries
            .iter()
            .position(|(k, e)| *k == key && e.prefix == prefix)
        {
            self.entries.remove(pos);
        }
        while self.entries.len() >= self.capacity {
            if self.entries.pop_back().is_none() {
                break;
            }
        }
        self.entries.push_front((key, Entry { prefix, snapshot }));
    }
}

fn hash_prefix(prefix: &[i64]) -> u64 {
    let mut h = DefaultHasher::new();
    prefix.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp() -> ModelFingerprint {
        ModelFingerprint {
            arch: "minimax_m2".into(),
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            hidden_size: 8,
            num_kv_heads: 2,
            qk_head_dim: 2,
            v_head_dim: 2,
            vocab_size: 256,
            layer_start: 0,
            layer_end: 2,
            is_first: true,
            is_last: true,
        }
    }

    fn snap(past_seq_len: usize, fill: f32) -> OvMoeKvSnapshot {
        OvMoeKvSnapshot {
            past_seq_len,
            layers: vec![OvLayerKvSlice {
                k: vec![fill; 2 * past_seq_len * 2],
                v: vec![fill; 2 * past_seq_len * 2],
                seq: past_seq_len,
            }],
        }
    }

    #[test]
    fn insert_then_lookup_returns_same_snapshot() {
        let mut c = OvMoeKvPrefixCache::new(4);
        let prefix = vec![1i64, 2, 3];
        c.insert(prefix.clone(), &fp(), snap(3, 7.0));
        let prompt = vec![1i64, 2, 3, 4];
        let got = c.lookup(&prompt, &fp()).expect("hit");
        assert_eq!(got.past_seq_len, 3);
        assert_eq!(got.layers[0].k[0], 7.0);
    }

    #[test]
    fn full_prompt_match_is_rejected_strict_prefix() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert(vec![1i64, 2, 3], &fp(), snap(3, 1.0));
        // prompt == cached prefix: plen >= prompt.len() → no hit.
        assert!(c.lookup(&[1i64, 2, 3], &fp()).is_none());
    }

    #[test]
    fn lookup_returns_longest_matching_prefix() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert(vec![1i64, 2, 3], &fp(), snap(3, 3.0));
        c.insert(vec![1i64, 2, 3, 4, 5], &fp(), snap(5, 5.0));
        let got = c.lookup(&[1i64, 2, 3, 4, 5, 6], &fp()).expect("hit");
        assert_eq!(got.past_seq_len, 5);
    }

    #[test]
    fn disabled_cache_never_hits() {
        let mut c = OvMoeKvPrefixCache::new(0);
        c.insert(vec![1i64, 2], &fp(), snap(2, 1.0));
        assert!(c.lookup(&[1i64, 2, 3], &fp()).is_none());
        assert_eq!(c.len(), 0);
    }
}
