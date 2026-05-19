//! Multi-turn chat session KV cache.
//!
//! Sibling of [`crate::kv_prefix_cache`]. Both cache the post-prefill
//! KV state and let the runner skip re-prefilling a known prefix —
//! but they target different workloads:
//!
//! - [`crate::kv_prefix_cache::KvPrefixCache`] (iter 060) keys on the
//!   token-id prefix itself; great when many requests share a system
//!   prompt but each request is independent.
//! - [`KvSessionCache`] (this module, iter 072) keys on a caller-
//!   supplied **session id** (UUID from the API request header). Each
//!   session owns one slot. Snapshots are taken at the END of every
//!   generation so they include the assistant's reply, letting the
//!   NEXT turn of the same conversation skip the entire prior
//!   `system + user₁ + asst₁ + user₂ + asst₂ + …` history.
//!
//! ### Why a separate cache (and not just reuse the prefix cache)?
//!
//! The prefix cache is keyed on the prompt token sequence and only
//! snapshots after PREFILL — it never sees the model's own output. A
//! multi-turn chat would either (a) require N entries per session
//! (one per turn) wasting memory, or (b) require explicit "snapshot
//! after generation" semantics the prefix-cache API doesn't expose.
//! Better to express the new behaviour as a distinct cache that the
//! Runner consults separately. The two caches are layered: session
//! lookup runs first (it has the longest possible hit), prefix lookup
//! runs second as a fallback.
//!
//! ### Cache shape
//!
//! `IndexMap<session_id, SessionEntry>` where `SessionEntry` holds the
//! full token sequence so far (prompt + generated) plus a packed
//! [`crate::kv_prefix_cache::KvSnapshot`]. The LRU ring is maintained
//! by reordering the IndexMap (`move_index` on hit, push on insert).
//!
//! ### Byte-budget eviction
//!
//! Unlike the prefix cache (which counts entries), this cache counts
//! BYTES. A session that has accumulated 4k tokens of context is much
//! bigger than a session that has accumulated 200, so a "max N
//! sessions" budget would silently OOM at the long tail. Bytes are
//! summed from each entry's [`KvSnapshot::approx_bytes`].
//!
//! `capacity_bytes == 0` disables the cache: every lookup is None,
//! every insert is a no-op. That's the default and preserves
//! byte-identical behaviour vs the pre-iter-072 build.
//!
//! ### Bit-identity contract
//!
//! The same one the prefix cache promises (iter 060): on a hit, the
//! restored KV bits are identical to what `forward_shells` would have
//! written if every prefix token had been re-prefilled. The Runner's
//! `restore_kv` does the heavy lifting; this module just owns
//! storage + LRU bookkeeping.

use std::collections::VecDeque;

use crate::kv_prefix_cache::{KvSnapshot, ModelFingerprint};

/// One entry in the session cache.
struct SessionEntry {
    /// Caller-supplied session id (e.g. UUID from an `X-Session-Id`
    /// header). Treated as opaque — the cache never parses it. Two
    /// requests with different session ids never share a snapshot,
    /// even if their prompts are identical.
    session_id: String,
    /// The full token sequence captured by the snapshot. On a lookup,
    /// the caller passes the CURRENT prompt; we only return a hit if
    /// `tokens` is a strict prefix of that prompt (so the runner has
    /// at least one suffix token to drive through prefill — same
    /// invariant the prefix cache enforces).
    tokens: Vec<i64>,
    /// Model fingerprint at snapshot time. A model reload (different
    /// fingerprint) invalidates the entry — restoring a snapshot from
    /// one model into another would either segfault on a shape
    /// mismatch or silently corrupt KV bits.
    model_digest: u64,
    /// The packed KV snapshot. Same shape contract as the prefix
    /// cache's entries — the runner's `restore_kv` consumes either.
    snapshot: KvSnapshot,
}

/// LRU cache keyed by session id, sized in bytes.
///
/// Construct with `KvSessionCache::new(capacity_bytes)`.
/// `capacity_bytes == 0` returns a disabled cache.
///
/// The cache is NOT thread-safe — wrap in a `Mutex` at the engine
/// layer. The engine already does this for the prefix cache so the
/// access pattern is established.
pub struct KvSessionCache {
    capacity_bytes: usize,
    /// LRU ring: `front` = most-recently-used, `back` = LRU.
    /// VecDeque suffices because session counts are realistically
    /// O(10²); a HashMap-backed LRU would save a few µs per lookup
    /// but cost ~80 lines of unsafe to maintain pointer stability.
    entries: VecDeque<SessionEntry>,
    /// Running sum of `entry.snapshot.approx_bytes()`. Maintained
    /// incrementally so `total_bytes()` is O(1) — the eviction loop
    /// would otherwise be O(n²) when many small entries pile up.
    total_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub evictions: u64,
}

/// What [`KvSessionCache::lookup`] returns on a hit. The runner uses
/// `matched_len` to know how many prompt tokens are already in KV so
/// it can skip them in prefill.
#[derive(Clone, Debug)]
pub struct SessionHit {
    pub matched_len: usize,
    pub snapshot: KvSnapshot,
}

impl KvSessionCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            entries: VecDeque::new(),
            total_bytes: 0,
            hits: 0,
            misses: 0,
            inserts: 0,
            evictions: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.capacity_bytes > 0
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a session by id. Returns Some(hit) when:
    /// - the session id exists,
    /// - the entry's model_digest matches `fingerprint` (no
    ///   cross-model bleed after a reload),
    /// - the entry's cached tokens are a STRICT prefix of `prompt`
    ///   (the new prompt extends the previous history; at least one
    ///   suffix token is left for prefill to produce logits from).
    ///
    /// On a hit, the entry is promoted to MRU.
    /// On a miss (any reason), `misses` is bumped and `None` returned.
    pub fn lookup(
        &mut self,
        session_id: &str,
        prompt: &[i64],
        fingerprint: &ModelFingerprint,
    ) -> Option<SessionHit> {
        if !self.enabled() {
            return None;
        }
        let model_digest = fingerprint.digest();
        // O(n) scan; n is the number of active sessions and
        // realistically ≤ a few hundred. A HashMap+linked-list LRU
        // would be O(1) here but for our cap range the constant
        // factor of the simple scan wins.
        let mut found_idx: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.session_id == session_id {
                found_idx = Some(i);
                break;
            }
        }
        let Some(i) = found_idx else {
            self.misses += 1;
            return None;
        };
        // Validate the find before promoting.
        let e = &self.entries[i];
        if e.model_digest != model_digest {
            // Same session id but the model has been reloaded since
            // the snapshot was taken. Drop the entry: it would fail
            // restore_kv with a shape error, and even if it didn't,
            // the cached KV bits belong to a different model.
            // Counting this as a miss + eviction is the cleanest
            // signal in the logs.
            let removed = self.entries.remove(i).expect("index from scan");
            self.total_bytes = self
                .total_bytes
                .saturating_sub(removed.snapshot.approx_bytes());
            self.evictions += 1;
            self.misses += 1;
            return None;
        }
        let cached_len = e.tokens.len();
        if cached_len >= prompt.len() {
            // Cached history is >= the new prompt → either the same
            // turn re-submitted, or a roll-back. Either way there's
            // no suffix to prefill, so no useful hit. Treat as miss
            // but keep the entry (the caller may roll forward next).
            self.misses += 1;
            return None;
        }
        if prompt[..cached_len] != e.tokens[..] {
            // Same session id but the prompt diverges from the
            // cached history. Could be a client bug, prompt
            // rewriting, or template change. Evict the stale entry
            // — the cache is useless for this session until the
            // client converges.
            let removed = self.entries.remove(i).expect("index from scan");
            self.total_bytes = self
                .total_bytes
                .saturating_sub(removed.snapshot.approx_bytes());
            self.evictions += 1;
            self.misses += 1;
            return None;
        }
        // Real hit. Promote to MRU and return.
        self.hits += 1;
        let entry = self.entries.remove(i).expect("index from scan");
        let hit = SessionHit {
            matched_len: cached_len,
            snapshot: entry.snapshot.clone(),
        };
        self.entries.push_front(entry);
        Some(hit)
    }

    /// Insert (or replace) the entry for `session_id`. `tokens` should
    /// be the FULL token sequence the snapshot represents — typically
    /// `prompt + generated_tokens` captured at end of generation.
    ///
    /// Replaces any existing entry for the same session id (the most
    /// common path on every turn after the first). Evicts LRU entries
    /// until the byte budget fits. Returns the number of evicted
    /// entries (replacement of the same session id is not counted as
    /// an eviction).
    pub fn insert(
        &mut self,
        session_id: String,
        tokens: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: KvSnapshot,
    ) -> usize {
        if !self.enabled() {
            return 0;
        }
        debug_assert_eq!(
            tokens.len(),
            snapshot.past_seq_len,
            "session-cache insert: token count {} != snapshot past_seq_len {}",
            tokens.len(),
            snapshot.past_seq_len
        );
        // Drop any existing entry for this session id first; the new
        // snapshot supersedes it. This also accounts for the byte
        // delta correctly when the new snapshot is bigger.
        if let Some(pos) = self.entries.iter().position(|e| e.session_id == session_id) {
            let prev = self.entries.remove(pos).expect("index from scan");
            self.total_bytes = self
                .total_bytes
                .saturating_sub(prev.snapshot.approx_bytes());
        }
        let new_bytes = snapshot.approx_bytes();
        // If the single new entry exceeds the entire cap, we still
        // insert it but evict everything else (cap=0 case is already
        // returned above). Caller can decide whether to clear()
        // explicitly to avoid this churn.
        let mut evicted = 0;
        while self.total_bytes + new_bytes > self.capacity_bytes && !self.entries.is_empty() {
            if let Some(victim) = self.entries.pop_back() {
                self.total_bytes = self
                    .total_bytes
                    .saturating_sub(victim.snapshot.approx_bytes());
                evicted += 1;
            }
        }
        self.evictions += evicted as u64;
        let entry = SessionEntry {
            session_id,
            tokens,
            model_digest: fingerprint.digest(),
            snapshot,
        };
        self.total_bytes += new_bytes;
        self.entries.push_front(entry);
        self.inserts += 1;
        evicted
    }

    /// Forget a specific session. Used by future API endpoints
    /// (`DELETE /v1/sessions/:id`) to give callers explicit control.
    /// Returns true if the session was present.
    pub fn forget(&mut self, session_id: &str) -> bool {
        let Some(pos) = self.entries.iter().position(|e| e.session_id == session_id) else {
            return false;
        };
        let removed = self.entries.remove(pos).expect("index from scan");
        self.total_bytes = self
            .total_bytes
            .saturating_sub(removed.snapshot.approx_bytes());
        true
    }

    /// Discard every entry. Called by the engine when the model is
    /// reloaded — every session's KV bits are model-bound.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_prefix_cache::{KvSnapshot, LayerKvSlice};

    fn fp_a() -> ModelFingerprint {
        ModelFingerprint {
            arch: "kimi_k2.6".into(),
            num_layers: 61,
            num_experts: 384,
            top_k: 8,
            hidden_size: 7168,
            num_kv_heads: 64,
            qk_head_dim: 192,
            v_head_dim: 128,
            vocab_size: 163840,
            layer_start: 0,
            layer_end: u32::MAX,
            is_first: true,
            is_last: true,
        }
    }

    fn fp_b() -> ModelFingerprint {
        ModelFingerprint {
            arch: "qwen3".into(),
            num_layers: 32,
            ..fp_a()
        }
    }

    /// Minimal snapshot for tests: 2 heads, head_dim 2, n_slots == past_seq_len.
    /// Mirrors the helper in kv_prefix_cache::tests so the two cache test
    /// suites use the same fixture geometry — eyeballing byte counts and
    /// asserting "this snapshot is X bytes" stays consistent.
    fn mk_snapshot(past_seq_len: usize, fill: u16) -> KvSnapshot {
        let n = 2 * past_seq_len * 2;
        KvSnapshot {
            past_seq_len,
            num_heads: 2,
            qk_head_dim: 2,
            v_head_dim: 2,
            layer0: Some(LayerKvSlice {
                lid: 0,
                past_k: vec![fill; n],
                past_v: vec![fill; n],
            }),
            shells: vec![LayerKvSlice {
                lid: 1,
                past_k: vec![fill.wrapping_mul(2); n],
                past_v: vec![fill.wrapping_mul(2); n],
            }],
        }
    }

    #[test]
    fn disabled_cache_is_a_noop() {
        let mut c = KvSessionCache::new(0);
        assert!(!c.enabled());
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        assert!(c.lookup("s1", &[1, 2, 3, 4], &fp_a()).is_none());
        assert_eq!(c.len(), 0);
        // Disabled lookups don't even bump misses (early return before
        // the counter). Insert doesn't bump inserts either. This makes
        // it trivial to see "feature off" in the logs.
        assert_eq!(c.hits, 0);
        assert_eq!(c.misses, 0);
        assert_eq!(c.inserts, 0);
    }

    #[test]
    fn insert_then_lookup_returns_hit_with_matched_len() {
        let cap = 1 << 30; // 1 GiB — comfortably above any test snapshot.
        let mut c = KvSessionCache::new(cap);
        let snap = mk_snapshot(3, 7);
        c.insert("s1".into(), vec![10, 20, 30], &fp_a(), snap.clone());
        let hit = c.lookup("s1", &[10, 20, 30, 40, 50], &fp_a()).expect("hit");
        assert_eq!(hit.matched_len, 3);
        assert_eq!(hit.snapshot.past_seq_len, snap.past_seq_len);
        // Byte-identity at the snapshot level — the load-bearing
        // invariant the runner relies on.
        assert_eq!(
            hit.snapshot.layer0.as_ref().unwrap().past_k,
            snap.layer0.as_ref().unwrap().past_k
        );
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 0);
    }

    #[test]
    fn lookup_misses_on_unknown_session_id() {
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        assert!(c.lookup("s2", &[1, 2, 3, 4], &fp_a()).is_none());
        assert_eq!(c.misses, 1);
    }

    #[test]
    fn lookup_misses_when_prompt_does_not_extend_cached_history() {
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        // Prompt diverges at position 1 — the cached snapshot's KV
        // bits assume token 2 at that position. Restoring would give
        // wrong outputs. Must miss AND evict (cache is now stale for
        // this session).
        assert!(c.lookup("s1", &[1, 99, 30, 4], &fp_a()).is_none());
        assert_eq!(c.misses, 1);
        assert_eq!(c.evictions, 1);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn lookup_misses_when_prompt_equals_cached_history() {
        // No suffix to drive prefill = no useful hit (mirrors the
        // prefix-cache invariant — first-token logits need at least
        // one new token to be sampled from).
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        assert!(c.lookup("s1", &[1, 2, 3], &fp_a()).is_none());
        assert_eq!(c.misses, 1);
        // Entry should still be there for the next turn (no eviction).
        assert_eq!(c.len(), 1);
        assert_eq!(c.evictions, 0);
    }

    #[test]
    fn lookup_misses_on_fingerprint_change_and_evicts() {
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        // Same session, but the model was reloaded between requests.
        // The cached KV bits belong to fp_a's model architecture;
        // restoring them under fp_b would either segfault on shape
        // mismatch or silently corrupt. Must evict.
        assert!(c.lookup("s1", &[1, 2, 3, 4], &fp_b()).is_none());
        assert_eq!(c.misses, 1);
        assert_eq!(c.evictions, 1);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn insert_replaces_existing_session_in_place() {
        // The steady-state path: every turn replaces the previous
        // turn's snapshot. The cache should NOT grow — same slot,
        // new snapshot.
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        let bytes_after_one = c.total_bytes();
        c.insert("s1".into(), vec![1, 2, 3, 4, 5], &fp_a(), mk_snapshot(5, 9));
        assert_eq!(c.len(), 1);
        // Newer snapshot has more tokens → larger bytes.
        assert!(c.total_bytes() > bytes_after_one);
        let hit = c.lookup("s1", &[1, 2, 3, 4, 5, 6], &fp_a()).expect("hit");
        assert_eq!(hit.matched_len, 5);
        // Value from the second insert (fill=9).
        assert_eq!(hit.snapshot.layer0.as_ref().unwrap().past_k[0], 9);
    }

    #[test]
    fn byte_budget_evicts_lru() {
        // Each mk_snapshot(N, _) is 2 * N * 2 * 2 * 2 * 2 = 32 * N bytes (bf16/u16).
        // mk_snapshot(3, _) = 96 bytes. Cap = 96*2 = 192 → 2 entries fit;
        // a third must evict.
        let bytes_per = 96;
        let mut c = KvSessionCache::new(bytes_per * 2);
        c.insert("s1".into(), vec![1, 1, 1], &fp_a(), mk_snapshot(3, 1));
        c.insert("s2".into(), vec![2, 2, 2], &fp_a(), mk_snapshot(3, 2));
        assert_eq!(c.len(), 2);
        c.insert("s3".into(), vec![3, 3, 3], &fp_a(), mk_snapshot(3, 3));
        assert_eq!(c.len(), 2);
        assert_eq!(c.evictions, 1);
        // s1 was LRU → evicted. s2, s3 remain.
        assert!(c.lookup("s1", &[1, 1, 1, 9], &fp_a()).is_none());
        assert!(c.lookup("s2", &[2, 2, 2, 9], &fp_a()).is_some());
        assert!(c.lookup("s3", &[3, 3, 3, 9], &fp_a()).is_some());
    }

    #[test]
    fn lookup_promotes_session_to_mru() {
        // Insert s1, s2. Lookup s1 (promotes). Insert s3 — must evict
        // s2 (now LRU), not s1.
        let bytes_per = 96;
        let mut c = KvSessionCache::new(bytes_per * 2);
        c.insert("s1".into(), vec![1, 1, 1], &fp_a(), mk_snapshot(3, 1));
        c.insert("s2".into(), vec![2, 2, 2], &fp_a(), mk_snapshot(3, 2));
        c.lookup("s1", &[1, 1, 1, 9], &fp_a()).expect("s1 hit");
        c.insert("s3".into(), vec![3, 3, 3], &fp_a(), mk_snapshot(3, 3));
        assert!(
            c.lookup("s1", &[1, 1, 1, 9], &fp_a()).is_some(),
            "s1 was promoted; should survive eviction"
        );
        assert!(
            c.lookup("s2", &[2, 2, 2, 9], &fp_a()).is_none(),
            "s2 was LRU after the lookup; should be evicted"
        );
    }

    #[test]
    fn forget_removes_specific_session() {
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.insert("s2".into(), vec![4, 5, 6], &fp_a(), mk_snapshot(3, 2));
        assert!(c.forget("s1"));
        assert!(!c.forget("s1")); // already gone
        assert_eq!(c.len(), 1);
        assert!(c.lookup("s1", &[1, 2, 3, 9], &fp_a()).is_none());
        assert!(c.lookup("s2", &[4, 5, 6, 9], &fp_a()).is_some());
    }

    #[test]
    fn clear_empties_cache() {
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.insert("s2".into(), vec![4, 5, 6], &fp_a(), mk_snapshot(3, 2));
        c.clear();
        assert_eq!(c.len(), 0);
        assert_eq!(c.total_bytes(), 0);
        assert!(c.lookup("s1", &[1, 2, 3, 9], &fp_a()).is_none());
    }

    #[test]
    fn total_bytes_tracks_correctly_across_replace() {
        // Insert small → check bytes. Replace with larger → check bytes
        // grew by the delta, not by the full new snapshot's size.
        let mut c = KvSessionCache::new(1 << 20);
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        let small = c.total_bytes();
        c.insert(
            "s1".into(),
            vec![1, 2, 3, 4, 5, 6, 7],
            &fp_a(),
            mk_snapshot(7, 2),
        );
        let big = c.total_bytes();
        // 7-token snapshot should be ~7/3 times the 3-token one.
        // Exact: bytes_per = 32 * past_seq_len → 32*7 vs 32*3 (bf16/u16).
        assert_eq!(big, 32 * 7);
        assert_eq!(small, 32 * 3);
    }

    #[test]
    fn oversized_insert_evicts_everything_and_still_lands() {
        // A single new snapshot bigger than the cap: keep the new one,
        // evict everything else, accept that total_bytes will briefly
        // exceed the cap (we don't refuse the insert).
        let mut c = KvSessionCache::new(96); // exactly one small entry (bf16/u16)
        c.insert("s1".into(), vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        // mk_snapshot(10, _) = 320 bytes, > 96 cap.
        c.insert("s2".into(), vec![0; 10], &fp_a(), mk_snapshot(10, 2));
        // s1 evicted; s2 present (cap exceeded but entry retained).
        assert!(c.lookup("s1", &[1, 2, 3, 9], &fp_a()).is_none());
        assert!(c.lookup("s2", &[0; 11], &fp_a()).is_some());
    }
}
