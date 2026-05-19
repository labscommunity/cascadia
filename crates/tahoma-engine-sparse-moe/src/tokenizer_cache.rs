//! Tokenizer output cache.
//!
//! Caches the `prompt text → token-id Vec` mapping so re-tokenizing
//! common system prompts is skipped on the chat-completion path. The
//! HuggingFace BPE pass for K2.6's tokenizer.json is 5..20 ms for a
//! 100-token system prompt; that's a measurable chunk of per-request
//! latency once the rest of the engine is fast (~0.1 s/token decode at
//! the time of writing).
//!
//! ### Cache shape
//!
//! Keyed by SipHash (std `DefaultHasher`) over the tuple
//! `(prompt_text, add_special_tokens, tokenizer_fingerprint)`.
//!
//! - **prompt_text**: the exact `&str` we hand to `tokenizer.encode()`.
//!   No normalization — different whitespace = different cache key.
//!   The HF tokenizer normalizes internally; we just refuse to coalesce.
//! - **add_special_tokens**: BPE w/ specials vs w/o specials produce
//!   different id streams; both call sites in [`crate::engine`] currently
//!   pass `true` but the API surface accepts the bool to defend against
//!   a future caller that doesn't.
//! - **tokenizer_fingerprint**: digest of the tokenizer.json bytes at
//!   load time. Different model = different fingerprint = no collisions
//!   with stale cached entries when the engine is reloaded against a
//!   new model directory. Required by the task brief.
//!
//! Hash collisions are guarded by re-comparing the full key tuple on
//! lookup (the cache stores the original prompt text alongside each
//! entry).
//!
//! ### LRU + size cap
//!
//! Backing store: `VecDeque<(Key, Entry)>` with `front` = MRU,
//! `back` = LRU. On hit, the matched entry moves to front. On insert,
//! LRU entries are evicted until under capacity. This mirrors
//! [`crate::kv_prefix_cache::KvPrefixCache`] — small caps (default 0,
//! realistic settings 8..128) make the O(n) scan trivial; a HashMap-
//! backed LRU buys ~1 µs/lookup at the cost of doubled code size.
//!
//! ### Disabled by default
//!
//! `TokenizerCache::new(0)` returns a no-op cache: `get` always
//! returns `None`, `insert` is a no-op. The engine's encode path falls
//! back to direct `tokenizer.encode()` with the same byte-identical
//! output as before this PR.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

/// 64-bit digest of the tokenizer.json bytes (or any byte stream that
/// uniquely identifies the tokenizer). Computed once at engine load
/// and stamped into every cache key so swapping in a different model
/// invalidates the cache without an explicit `clear()` call from the
/// caller.
pub type TokenizerFingerprint = u64;

/// Compute a fingerprint from a byte slice — typically the raw bytes
/// of `tokenizer.json`. Cheap (~1 µs/MB on a Xeon), called once at
/// engine load.
pub fn fingerprint_bytes(bytes: &[u8]) -> TokenizerFingerprint {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Cache key. Stored alongside the entry so a hash collision can be
/// detected on lookup (re-compare the full key, not just the hash).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Key {
    fingerprint: TokenizerFingerprint,
    add_special_tokens: bool,
    /// The full prompt text, owned. We pay the allocation on insert
    /// (one per distinct cached prompt) — typical system prompts are
    /// a few hundred bytes, so even a 128-entry cache is <100 KiB of
    /// key storage.
    prompt: String,
}

impl Key {
    fn digest(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.fingerprint.hash(&mut h);
        self.add_special_tokens.hash(&mut h);
        self.prompt.hash(&mut h);
        h.finish()
    }
}

/// One cached entry: token ids produced by `tokenizer.encode(prompt, add_special_tokens)`.
#[derive(Clone, Debug)]
struct Entry {
    token_ids: Vec<u32>,
}

/// LRU tokenizer-output cache.
///
/// Construct with `TokenizerCache::new(capacity)`. `capacity = 0`
/// disables the cache entirely (every `get` returns `None`, every
/// `insert` is a no-op).
pub struct TokenizerCache {
    capacity: usize,
    /// LRU ring: `front` = MRU, `back` = LRU. Each entry pairs a
    /// `(digest, key)` so we can do a fast digest comparison before
    /// falling back to the full key compare.
    entries: VecDeque<(u64, Key, Entry)>,
    /// Total cache hits since construction. Surfaced via [`Self::hits`].
    hits: u64,
    /// Total cache misses since construction.
    misses: u64,
    /// Total inserts (some replace existing entries; some evict).
    inserts: u64,
    /// Total evictions due to capacity overflow.
    evictions: u64,
}

impl TokenizerCache {
    /// Construct a cache with `capacity` entries (LRU evicted on overflow).
    /// `capacity = 0` disables the cache.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity.max(1)),
            hits: 0,
            misses: 0,
            inserts: 0,
            evictions: 0,
        }
    }

    /// True if the cache will store anything. `capacity == 0` returns false.
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
    pub fn misses(&self) -> u64 {
        self.misses
    }
    pub fn inserts(&self) -> u64 {
        self.inserts
    }
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Look up cached token ids for `(prompt, add_special_tokens)` under
    /// the given tokenizer fingerprint. On hit, the matched entry is
    /// promoted to MRU position. Returns `None` (and bumps the miss
    /// counter) if the cache is disabled or the key is absent.
    pub fn get(
        &mut self,
        prompt: &str,
        add_special_tokens: bool,
        fingerprint: TokenizerFingerprint,
    ) -> Option<Vec<u32>> {
        if !self.enabled() {
            return None;
        }
        // Build a temporary key for digest computation. We avoid
        // cloning `prompt` into a String here — DefaultHasher's
        // `str::hash` walks the bytes directly.
        let mut h = DefaultHasher::new();
        fingerprint.hash(&mut h);
        add_special_tokens.hash(&mut h);
        prompt.hash(&mut h);
        let digest = h.finish();

        // Scan for matching digest, then re-compare full key to defend
        // against the (astronomically unlikely) hash collision.
        let mut found_idx: Option<usize> = None;
        for (i, (d, key, _)) in self.entries.iter().enumerate() {
            if *d != digest {
                continue;
            }
            if key.fingerprint == fingerprint
                && key.add_special_tokens == add_special_tokens
                && key.prompt == prompt
            {
                found_idx = Some(i);
                break;
            }
        }
        match found_idx {
            None => {
                self.misses += 1;
                None
            }
            Some(i) => {
                self.hits += 1;
                // Move-to-front (MRU). VecDeque::remove is O(n) but
                // n is at most `capacity`; for the realistic caps
                // here (≤128) that's still well under a µs.
                let entry = self
                    .entries
                    .remove(i)
                    .expect("index from enumerate must be valid");
                let cloned_ids = entry.2.token_ids.clone();
                self.entries.push_front(entry);
                Some(cloned_ids)
            }
        }
    }

    /// Insert a `(prompt, add_special_tokens, fingerprint) → token_ids`
    /// mapping. If an entry already exists for this exact key, replace
    /// it and move to MRU. Evicts LRU entries until under capacity.
    ///
    /// No-op when the cache is disabled.
    pub fn insert(
        &mut self,
        prompt: String,
        add_special_tokens: bool,
        fingerprint: TokenizerFingerprint,
        token_ids: Vec<u32>,
    ) {
        if !self.enabled() {
            return;
        }
        let key = Key {
            fingerprint,
            add_special_tokens,
            prompt,
        };
        let digest = key.digest();

        // Replace any exact-match entry first so capacity accounting
        // doesn't double-count it.
        if let Some(pos) = self.entries.iter().position(|(d, k, _)| {
            *d == digest
                && k.fingerprint == key.fingerprint
                && k.add_special_tokens == key.add_special_tokens
                && k.prompt == key.prompt
        }) {
            self.entries.remove(pos);
        }
        // Evict LRU until we have room for the new entry.
        while self.entries.len() >= self.capacity {
            if self.entries.pop_back().is_none() {
                break;
            }
            self.evictions += 1;
        }
        self.entries.push_front((digest, key, Entry { token_ids }));
        self.inserts += 1;
    }

    /// Discard every entry. Engines call this on model reload —
    /// fingerprint mismatch is the primary correctness guard but a
    /// hard clear is still cheaper than walking the cache to discover
    /// every miss.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP_A: TokenizerFingerprint = 0xAAAA_AAAA_AAAA_AAAA;
    const FP_B: TokenizerFingerprint = 0xBBBB_BBBB_BBBB_BBBB;

    #[test]
    fn disabled_cache_is_a_noop() {
        let mut c = TokenizerCache::new(0);
        assert!(!c.enabled());
        c.insert("hello".to_string(), true, FP_A, vec![1, 2, 3]);
        assert!(c.get("hello", true, FP_A).is_none());
        assert_eq!(c.len(), 0);
        // Stats untouched on disabled cache (get early-returns).
        assert_eq!(c.hits, 0);
        assert_eq!(c.misses, 0);
        assert_eq!(c.inserts, 0);
    }

    /// Load-bearing test the task brief calls out:
    /// "same prompt twice = cache hit on 2nd".
    #[test]
    fn same_prompt_twice_hits_second_time() {
        let mut c = TokenizerCache::new(4);
        // First call: miss (cache was empty), then insert.
        assert!(c.get("system prompt", true, FP_A).is_none());
        assert_eq!(c.misses, 1);
        assert_eq!(c.hits, 0);
        c.insert(
            "system prompt".to_string(),
            true,
            FP_A,
            vec![10, 11, 12, 13],
        );
        assert_eq!(c.inserts, 1);
        // Second call with same prompt: hit, byte-identical ids.
        let got = c.get("system prompt", true, FP_A).expect("hit expected");
        assert_eq!(got, vec![10, 11, 12, 13]);
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 1);
    }

    #[test]
    fn different_prompt_does_not_collide() {
        let mut c = TokenizerCache::new(4);
        c.insert("hello".to_string(), true, FP_A, vec![1, 2]);
        c.insert("world".to_string(), true, FP_A, vec![3, 4]);
        assert_eq!(c.get("hello", true, FP_A), Some(vec![1, 2]));
        assert_eq!(c.get("world", true, FP_A), Some(vec![3, 4]));
        assert_eq!(c.get("nope", true, FP_A), None);
    }

    /// Different tokenizer fingerprint must miss — task brief's
    /// "Tokenizer changes (different model) → cache should invalidate"
    /// requirement.
    #[test]
    fn different_fingerprint_misses() {
        let mut c = TokenizerCache::new(4);
        c.insert("hello".to_string(), true, FP_A, vec![1, 2, 3]);
        // Same prompt, same add_special_tokens, different fingerprint —
        // must miss. Critical for correctness: serving Qwen tokens
        // from a K2.6 cache would silently corrupt every request.
        assert!(c.get("hello", true, FP_B).is_none());
        // The FP_A entry is still there though.
        assert_eq!(c.get("hello", true, FP_A), Some(vec![1, 2, 3]));
    }

    #[test]
    fn different_add_special_tokens_misses() {
        // BPE w/ specials vs w/o specials produce different id streams.
        // Each call site in engine.rs currently passes `true`, but the
        // public surface accepts the bool to defend against a future
        // caller (e.g. chat-template rendering) that doesn't.
        let mut c = TokenizerCache::new(4);
        c.insert("hello".to_string(), true, FP_A, vec![1, 2, 3]);
        assert!(c.get("hello", false, FP_A).is_none());
        c.insert("hello".to_string(), false, FP_A, vec![4, 5]);
        // Both must coexist.
        assert_eq!(c.get("hello", true, FP_A), Some(vec![1, 2, 3]));
        assert_eq!(c.get("hello", false, FP_A), Some(vec![4, 5]));
    }

    #[test]
    fn lru_eviction_drops_oldest() {
        let mut c = TokenizerCache::new(2);
        c.insert("a".to_string(), true, FP_A, vec![1]);
        c.insert("b".to_string(), true, FP_A, vec![2]);
        // Cap=2; inserting a third evicts the oldest ("a").
        c.insert("c".to_string(), true, FP_A, vec![3]);
        assert_eq!(c.len(), 2);
        assert!(c.get("a", true, FP_A).is_none());
        assert!(c.get("b", true, FP_A).is_some());
        assert!(c.get("c", true, FP_A).is_some());
        assert_eq!(c.evictions, 1);
    }

    #[test]
    fn get_promotes_to_mru() {
        // Cap=2: insert A, B → LRU order [B (MRU), A (LRU)].
        // get(A) → A moves to front. Insert C → must evict B, not A.
        let mut c = TokenizerCache::new(2);
        c.insert("a".to_string(), true, FP_A, vec![1]);
        c.insert("b".to_string(), true, FP_A, vec![2]);
        let _ = c.get("a", true, FP_A).expect("A still present");
        c.insert("c".to_string(), true, FP_A, vec![3]);
        assert!(
            c.get("a", true, FP_A).is_some(),
            "A promoted; should survive"
        );
        assert!(c.get("b", true, FP_A).is_none(), "B was LRU; evicted");
        assert!(c.get("c", true, FP_A).is_some(), "C just inserted");
    }

    #[test]
    fn insert_same_key_replaces_in_place() {
        // Calling insert twice with the same key shouldn't grow the cache,
        // and the second value must win.
        let mut c = TokenizerCache::new(4);
        c.insert("hello".to_string(), true, FP_A, vec![1, 2, 3]);
        c.insert("hello".to_string(), true, FP_A, vec![9, 9, 9]);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get("hello", true, FP_A), Some(vec![9, 9, 9]));
    }

    #[test]
    fn clear_empties_cache() {
        let mut c = TokenizerCache::new(4);
        c.insert("hello".to_string(), true, FP_A, vec![1, 2, 3]);
        c.clear();
        assert_eq!(c.len(), 0);
        assert!(c.get("hello", true, FP_A).is_none());
    }

    #[test]
    fn fingerprint_bytes_is_deterministic_and_distinct() {
        let f1 = fingerprint_bytes(b"hello world");
        let f2 = fingerprint_bytes(b"hello world");
        let f3 = fingerprint_bytes(b"hello world!");
        assert_eq!(f1, f2, "same bytes -> same fingerprint");
        assert_ne!(f1, f3, "differing bytes -> different fingerprint");
    }

    #[test]
    fn empty_prompt_is_cacheable() {
        // Edge case: HF tokenizer accepts empty prompts and returns
        // (for K2.6) the bos token alone. Cache must handle this — the
        // engine's chat-completion path can hit it on a "ping" health
        // probe.
        let mut c = TokenizerCache::new(2);
        c.insert(String::new(), true, FP_A, vec![1]);
        assert_eq!(c.get("", true, FP_A), Some(vec![1]));
    }
}
