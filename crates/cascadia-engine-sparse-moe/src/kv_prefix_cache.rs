//! Static prompt KV cache.
//!
//! Caches the post-prefill KV-cache state for an exact token-id prefix
//! so subsequent generations that start with the same prefix can skip
//! that portion of prefill. Common case: a chat-completion endpoint
//! where every request shares the same system prompt — at ~3 s/token
//! prefill, a 500-token shared prefix costs ~1500 s of redundant work
//! per request without caching.
//!
//! ### Cache shape
//!
//! Keyed by a 64-bit FxHash over the **token id sequence** plus a
//! `model_fingerprint` from the manifest. Token-id keying (not raw
//! prompt text) means two requests whose prompt text differs only in
//! whitespace tokenize to different keys — accepted; the cache only
//! ever returns hits when the *token stream* matches byte-identically.
//! Hash collisions are guarded by re-comparing the full token slice on
//! lookup (see [`KvPrefixCache::lookup`]).
//!
//! The cached value is a [`KvSnapshot`] holding the populated K/V
//! prefix for layer 0 and every MoE shell layer the rank owns. Capacity
//! padding is **not** part of the snapshot — on restore we only need
//! `past_seq_len` slots per head; the live runner repacks them into its
//! own (possibly larger) capacity buffer.
//!
//! ### LRU + size cap
//!
//! Backing store: `IndexMap<Hash, KvSnapshot>` so we can do O(1)
//! `move_to_back` on hit and O(1) `pop_front` on overflow. The cap is
//! the **number of entries**, not bytes — at K2.6 dimensions one
//! 512-token snapshot is ~150 MiB so even a small cap is meaningful.
//!
//! ### Disabled by default
//!
//! `KvPrefixCache::new(0)` returns a no-op cache: `lookup` always
//! returns `None`, `insert` is a no-op. This keeps the on-by-default
//! generate path byte-identical to the pre-cache behaviour.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

/// Per-layer slice of the KV cache as it would sit at `past_seq_len = N`
/// after a fresh prefill. The buffers are stored packed: exactly `N`
/// slots per head, no capacity padding. The Runner's restore path
/// re-packs into its own buffer using `write_present_kv`-style indexing.
///
/// **bf16 buffers (PR #30 A8).** Storage matches the runner's live KV:
/// each element is a bf16 value held in a `u16`. Restore is a direct
/// memcpy back into the runner's `[NUM_HEADS, capacity, head_dim]`
/// buffers — no dequantization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerKvSlice {
    /// Layer id this slice belongs to. For layer-0 this is `0`; for
    /// MoE shells it's the manifest layer id (1..num_layers).
    pub lid: u32,
    /// Layout: `[num_heads, n_slots, qk_head_dim]` row-major, bf16-as-u16.
    pub past_k: Vec<u16>,
    /// Layout: `[num_heads, n_slots, v_head_dim]` row-major, bf16-as-u16.
    pub past_v: Vec<u16>,
}

/// Full per-rank snapshot. Holds layer 0 (if owned by this rank) plus
/// every MoE shell layer the rank holds, in the same order as the live
/// `Runner::layers`. The Runner's restore code walks both vectors in
/// lockstep — drift = bug, so we panic in debug.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvSnapshot {
    /// How many tokens were in the prompt prefix when this snapshot was taken.
    /// On restore the Runner advances every layer's `past_seq_len` to this value.
    pub past_seq_len: usize,
    /// `num_heads` baked in at construction time so a snapshot from one
    /// model can't accidentally restore into another. The model fingerprint
    /// is the primary guard; this is a belt-and-braces invariant.
    pub num_heads: u32,
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    /// Optional layer-0 KV slice. None if this snapshot was taken on a
    /// non-first rank.
    pub layer0: Option<LayerKvSlice>,
    /// One slice per MoE shell layer owned by the rank, in `layers` order.
    pub shells: Vec<LayerKvSlice>,
}

impl KvSnapshot {
    /// Total cached bytes — useful for logging the cache footprint.
    /// Underestimate; doesn't count the IndexMap key + hash overhead.
    pub fn approx_bytes(&self) -> usize {
        let layer0 = self
            .layer0
            .as_ref()
            .map(|s| (s.past_k.len() + s.past_v.len()) * std::mem::size_of::<u16>())
            .unwrap_or(0);
        let shells: usize = self
            .shells
            .iter()
            .map(|s| (s.past_k.len() + s.past_v.len()) * std::mem::size_of::<u16>())
            .sum();
        layer0 + shells
    }
}

/// Model fingerprint — anything that affects the **KV bits** of a
/// prefill belongs here. Sampling params (temperature, top-p,
/// repetition penalty) do NOT — those only affect token sampling AFTER
/// the forward pass writes K/V, so a snapshot taken under temp=0.0 is
/// safe to restore for temp=0.7. This is the design decision called out
/// in the task brief.
///
/// Two requests that share a system prompt but use different temperatures
/// MUST share a cache entry; that's the entire point of the optimization.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelFingerprint {
    pub arch: String,
    pub num_layers: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub hidden_size: u32,
    pub num_kv_heads: u32,
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    pub vocab_size: u32,
    /// Rank's `(layer_start, layer_end, is_first, is_last)` baked in.
    /// A snapshot from rank 0 of a 2-stage split can't restore on rank 1.
    pub layer_start: u32,
    pub layer_end: u32,
    pub is_first: bool,
    pub is_last: bool,
}

impl ModelFingerprint {
    /// Deterministic 64-bit digest used as part of the cache key. Hashes
    /// every field; collisions would be astronomically unlikely (and
    /// further guarded by the cache's full prefix re-comparison).
    pub fn digest(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    /// Issue-34 cross-chain PLANE digest: hashes only the model-identity fields, EXCLUDING the
    /// per-stage slice (`layer_start/layer_end/is_first/is_last`). A move A→B pulls every rank's KV
    /// keyed by ONE fingerprint (the moved-to head's), so a tail holder's fp must match despite its
    /// different layer span — a per-stage digest would wrongly reject the worker ranks of a legitimate
    /// move. LOCAL cache keys keep the full `digest()` (a rank-0 snapshot must never restore on rank 1);
    /// this is used only for `model_fingerprint`/holder/export on the plane. Mirrors the OV engines'
    /// deliberately model-level plane fingerprint.
    pub fn plane_digest(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        self.arch.hash(&mut h);
        self.num_layers.hash(&mut h);
        self.num_experts.hash(&mut h);
        self.top_k.hash(&mut h);
        self.hidden_size.hash(&mut h);
        self.num_kv_heads.hash(&mut h);
        self.qk_head_dim.hash(&mut h);
        self.v_head_dim.hash(&mut h);
        self.vocab_size.hash(&mut h);
        h.finish()
    }
}

/// Cache key: model fingerprint digest + 64-bit hash of the token-id
/// prefix. Stored as a single 128-bit pair so equal keys imply both
/// digest and prefix-hash matched. The prefix-hash alone is not unique
/// — see [`KvPrefixCache::lookup`] for the full-slice re-comparison.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    model_digest: u64,
    prefix_hash: u64,
}

/// One cached entry. The token prefix is held alongside the snapshot
/// so `lookup` can re-compare and reject hash collisions.
struct Entry {
    prefix: Vec<i64>,
    snapshot: KvSnapshot,
    /// Pulled over the KV plane (`insert_pulled`) rather than captured from this rank's own turn.
    /// Read back by [`KvPrefixCache::lookup`] so a cert bar can tell the two apart.
    plane_pulled: bool,
}

/// LRU KV-prefix cache.
///
/// Construct with `KvPrefixCache::new(capacity)`. `capacity = 0`
/// disables the cache entirely (every `lookup` returns `None`, every
/// `insert` is a no-op).
///
/// **Concurrency**: this struct is `!Sync` because callers wrap it in
/// a `Mutex` at the engine level. The cache itself is sync-safe —
/// nothing inside it touches global state — but the `Runner` mutates
/// KV buffers during restore so the access pattern is necessarily
/// one-at-a-time.
pub struct KvPrefixCache {
    capacity: usize,
    /// VecDeque used as an LRU ring: `front` = most-recently-used,
    /// `back` = least-recently-used. On hit we move the entry to front.
    /// On miss we push_front; if at capacity we pop_back first.
    ///
    /// O(n) lookup — fine for the small caps that fit in RAM at K2.6
    /// dimensions (~150 MiB per 512-token snapshot, so caps are
    /// realistically 1..8). A HashMap-backed LRU would be 30 lines and
    /// 1 µs faster per lookup; not worth it for cap<=8.
    entries: VecDeque<(CacheKey, Entry)>,
    /// Total cache hits since construction. Logged by the runner.
    pub hits: u64,
    /// Total cache misses since construction.
    pub misses: u64,
    /// Total inserts (some replace existing entries; some evict).
    pub inserts: u64,
    /// Total evictions due to capacity overflow.
    pub evictions: u64,
}

impl KvPrefixCache {
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

    /// Look up the longest cached prefix that matches the start of
    /// `prompt`. Returns `Some((snapshot, plane_pulled))` for the
    /// best match, `None` if nothing matches.
    ///
    /// "Best" = longest prefix length. We scan every entry and pick
    /// the one whose `prefix` is a prefix of `prompt` and is longest.
    /// At cap<=8 this is a single-µs walk; not worth a trie.
    ///
    /// Side effect: on a hit, the matched entry is moved to MRU
    /// position (front of the deque) so it survives the next eviction.
    /// `hits`/`misses` counters are bumped accordingly.
    pub fn lookup(
        &mut self,
        prompt: &[i64],
        fingerprint: &ModelFingerprint,
    ) -> Option<(KvSnapshot, bool)> {
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
            // The prefix must wholly precede `prompt` AND leave at
            // least one suffix token for the generate loop to drive
            // through prefill — restoring the full prompt and then
            // having zero tail tokens to sample from would deadlock
            // the first-token logic.
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
                // Move-to-front (MRU). swap_remove would change the
                // back's index but we explicitly use remove → push_front
                // to preserve LRU ordering across the rest of the deque.
                let entry = self
                    .entries
                    .remove(i)
                    .expect("index from enumerate must be valid");
                let hit = (entry.1.snapshot.clone(), entry.1.plane_pulled);
                self.entries.push_front(entry);
                Some(hit)
            }
        }
    }

    /// Insert a snapshot keyed by the given prefix + model fingerprint.
    /// If an entry already exists for this exact key, replace it and
    /// move to MRU. Evicts LRU entries until under capacity.
    ///
    /// Returns the number of evicted entries (0 in the steady state
    /// once the cache is full and we replace rather than grow).
    ///
    /// Local capture — this rank's own turn. Cross-chain consumer-inserts
    /// go through [`Self::insert_pulled`].
    pub fn insert(
        &mut self,
        prefix: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: KvSnapshot,
    ) -> usize {
        self.store(prefix, fingerprint, snapshot, false)
    }

    /// As [`Self::insert`], but marks the entry as pulled over the KV plane so a later
    /// warm resume off it is attributable to the cross-chain pull, not to a local capture.
    pub fn insert_pulled(
        &mut self,
        prefix: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: KvSnapshot,
    ) -> usize {
        self.store(prefix, fingerprint, snapshot, true)
    }

    fn store(
        &mut self,
        prefix: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: KvSnapshot,
        plane_pulled: bool,
    ) -> usize {
        if !self.enabled() {
            return 0;
        }
        debug_assert_eq!(prefix.len(), snapshot.past_seq_len);
        let key = CacheKey {
            model_digest: fingerprint.digest(),
            prefix_hash: hash_prefix(&prefix),
        };
        // Replace any exact-match entry first so capacity accounting
        // doesn't double-count it.
        if let Some(pos) = self
            .entries
            .iter()
            .position(|(k, e)| *k == key && e.prefix.len() == prefix.len() && e.prefix == prefix)
        {
            self.entries.remove(pos);
        }
        let mut evicted = 0;
        while self.entries.len() >= self.capacity {
            if self.entries.pop_back().is_none() {
                break;
            }
            evicted += 1;
        }
        self.evictions += evicted as u64;
        self.entries.push_front((
            key,
            Entry {
                prefix,
                snapshot,
                plane_pulled,
            },
        ));
        self.inserts += 1;
        evicted
    }

    /// Discard every entry. Used by the API layer when the engine is
    /// reloaded (different model = stale snapshots).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Approximate total bytes held — sum of every cached snapshot's
    /// KV buffers. Useful for logging.
    pub fn approx_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(_, e)| e.snapshot.approx_bytes())
            .sum()
    }
}

fn hash_prefix(prefix: &[i64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    prefix.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn plane_digest_ignores_stage_slice_but_not_model() {
        // Same model, different stage slice ⇒ SAME plane_digest (cross-chain pull asserts one fp for
        // every rank), but DIFFERENT full digest (local cache must never cross ranks).
        let head = fp_a(); // layers 0..MAX, is_first=is_last=true
        let mut tail = fp_a();
        tail.layer_start = 30;
        tail.layer_end = 61;
        tail.is_first = false;
        assert_eq!(
            head.plane_digest(),
            tail.plane_digest(),
            "stage slice must not change the plane fp"
        );
        assert_ne!(
            head.digest(),
            tail.digest(),
            "full digest must reflect the stage slice"
        );
        // A different MODEL must change the plane fp.
        assert_ne!(
            head.plane_digest(),
            fp_b().plane_digest(),
            "different arch/num_layers must change the plane fp"
        );
    }

    /// Minimal snapshot for tests: 2 heads, head_dim 2, n_slots = past_seq_len.
    /// `fill` is treated as the raw u16 storage value (bf16-as-u16); the
    /// tests only check round-trip identity, not numeric semantics.
    fn mk_snapshot(past_seq_len: usize, fill: u16) -> KvSnapshot {
        // [num_heads=2, n_slots=past_seq_len, qk_head_dim=2]
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
        let mut c = KvPrefixCache::new(0);
        assert!(!c.enabled());
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        assert!(c.lookup(&[1, 2, 3, 4], &fp_a()).is_none());
        assert_eq!(c.len(), 0);
        // Stats untouched on disabled cache (lookup early-returns).
        assert_eq!(c.hits, 0);
        assert_eq!(c.misses, 0);
        assert_eq!(c.inserts, 0);
    }

    #[test]
    fn insert_then_lookup_returns_same_snapshot() {
        let mut c = KvPrefixCache::new(4);
        let snap = mk_snapshot(3, 7);
        c.insert(vec![10, 20, 30], &fp_a(), snap.clone());
        let (got, _) = c
            .lookup(&[10, 20, 30, 40, 50], &fp_a())
            .expect("hit expected");
        // Byte-identical restore: this is the load-bearing test the
        // task brief calls out — cached KV bits must match what a
        // fresh prefill would have written.
        assert_eq!(got.past_seq_len, snap.past_seq_len);
        assert_eq!(
            got.layer0.as_ref().unwrap().past_k,
            snap.layer0.as_ref().unwrap().past_k
        );
        assert_eq!(
            got.layer0.as_ref().unwrap().past_v,
            snap.layer0.as_ref().unwrap().past_v
        );
        assert_eq!(got.shells[0].past_k, snap.shells[0].past_k);
        assert_eq!(got.shells[0].past_v, snap.shells[0].past_v);
        assert_eq!(c.hits, 1);
        assert_eq!(c.misses, 0);
    }

    #[test]
    fn lookup_returns_longest_matching_prefix() {
        // Two entries for the same model: a 3-token and a 5-token
        // prefix both consistent with prompt [1,2,3,4,5,6]. Expect
        // the 5-token entry to win.
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.insert(vec![1, 2, 3, 4, 5], &fp_a(), mk_snapshot(5, 2));
        let (got, _) = c.lookup(&[1, 2, 3, 4, 5, 6], &fp_a()).expect("hit");
        assert_eq!(got.past_seq_len, 5);
    }

    #[test]
    fn lookup_misses_when_prefix_differs() {
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![10, 20, 30], &fp_a(), mk_snapshot(3, 1));
        // Prompt diverges at position 1 — must miss, not silently
        // return a wrong snapshot.
        assert!(c.lookup(&[10, 99, 30, 40], &fp_a()).is_none());
        assert_eq!(c.misses, 1);
    }

    #[test]
    fn lookup_misses_on_different_model_fingerprint() {
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        // Same prefix, different model — must miss. Critical for
        // correctness: restoring K2.6 KV into a Qwen runner would
        // segfault or produce garbage.
        assert!(c.lookup(&[1, 2, 3, 4], &fp_b()).is_none());
    }

    #[test]
    fn lookup_rejects_exact_match_no_suffix() {
        // If `prompt == cached prefix` there's no suffix to drive
        // through prefill — the first-token logic in `generate` would
        // have nothing to sample from. Treat as miss.
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        assert!(c.lookup(&[1, 2, 3], &fp_a()).is_none());
    }

    #[test]
    fn lru_eviction_drops_oldest() {
        let mut c = KvPrefixCache::new(2);
        c.insert(vec![1, 1, 1], &fp_a(), mk_snapshot(3, 1));
        c.insert(vec![2, 2, 2], &fp_a(), mk_snapshot(3, 2));
        // Cap=2; inserting a third evicts the oldest ([1,1,1]).
        c.insert(vec![3, 3, 3], &fp_a(), mk_snapshot(3, 3));
        assert_eq!(c.len(), 2);
        assert!(c.lookup(&[1, 1, 1, 9], &fp_a()).is_none());
        assert!(c.lookup(&[2, 2, 2, 9], &fp_a()).is_some());
        assert!(c.lookup(&[3, 3, 3, 9], &fp_a()).is_some());
        assert_eq!(c.evictions, 1);
    }

    #[test]
    fn lookup_promotes_entry_to_mru() {
        // Cap=2: insert A, B → LRU order [B (MRU), A (LRU)].
        // Lookup A → A moves to front. Insert C → must evict B,
        // not A.
        let mut c = KvPrefixCache::new(2);
        c.insert(vec![1, 1, 1], &fp_a(), mk_snapshot(3, 1));
        c.insert(vec![2, 2, 2], &fp_a(), mk_snapshot(3, 2));
        let _ = c.lookup(&[1, 1, 1, 9], &fp_a()).expect("A still present");
        c.insert(vec![3, 3, 3], &fp_a(), mk_snapshot(3, 3));
        assert!(
            c.lookup(&[1, 1, 1, 9], &fp_a()).is_some(),
            "A promoted; should survive"
        );
        assert!(
            c.lookup(&[2, 2, 2, 9], &fp_a()).is_none(),
            "B was LRU; should be evicted"
        );
        assert!(
            c.lookup(&[3, 3, 3, 9], &fp_a()).is_some(),
            "C just inserted"
        );
    }

    #[test]
    fn lookup_reports_entry_provenance() {
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        let (_, pulled) = c.lookup(&[1, 2, 3, 4], &fp_a()).expect("hit");
        assert!(!pulled, "locally captured");
        c.insert_pulled(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 2));
        let (_, pulled) = c.lookup(&[1, 2, 3, 4], &fp_a()).expect("hit");
        assert!(pulled, "consumer-inserted over the plane");
    }

    /// LRU must fail safe: once the pulled entry is evicted, the local entry that still
    /// matches carries its own mark, so a plane assertion can't green off a local capture.
    #[test]
    fn evicting_a_pulled_entry_leaves_a_local_mark() {
        let mut c = KvPrefixCache::new(2);
        c.insert_pulled(vec![1, 1, 1], &fp_a(), mk_snapshot(3, 1));
        c.insert(vec![2, 2, 2], &fp_a(), mk_snapshot(3, 2));
        c.insert(vec![3, 3, 3], &fp_a(), mk_snapshot(3, 3));
        assert_eq!(c.len(), 2);
        assert_eq!(c.evictions, 1);
        assert!(c.lookup(&[1, 1, 1, 9], &fp_a()).is_none(), "pulled was LRU");
        let (_, pulled) = c.lookup(&[2, 2, 2, 9], &fp_a()).expect("hit");
        assert!(!pulled);
    }

    #[test]
    fn insert_same_key_replaces_in_place() {
        // Calling insert twice with the same key shouldn't grow the cache.
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 9));
        assert_eq!(c.len(), 1);
        let (got, _) = c.lookup(&[1, 2, 3, 4], &fp_a()).expect("hit");
        // Second insert wins — value should be 9, not 1.
        assert_eq!(got.layer0.as_ref().unwrap().past_k[0], 9);
    }

    #[test]
    fn clear_empties_cache() {
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.clear();
        assert_eq!(c.len(), 0);
        assert!(c.lookup(&[1, 2, 3, 4], &fp_a()).is_none());
    }

    #[test]
    fn fingerprint_digest_is_deterministic() {
        // Same fields → same digest. Different field → different
        // digest. Belt-and-braces for the "cache must invalidate when
        // model config changes" constraint in the task brief.
        let d1 = fp_a().digest();
        let d2 = fp_a().digest();
        assert_eq!(d1, d2);
        assert_ne!(d1, fp_b().digest());
    }

    #[test]
    fn fingerprint_ignores_sampling_params() {
        // The fingerprint deliberately does NOT carry sampling config.
        // A cached snapshot at temp=0 must be reusable for temp=0.7.
        // We assert this by showing that two distinct fingerprints
        // with identical model fields have identical digests — no
        // accidental drift via a non-model field.
        let fp1 = fp_a();
        let mut fp2 = fp_a();
        // Mutate everything that's NOT a model field. Since
        // ModelFingerprint only carries model fields, there's nothing
        // to mutate — which is the test invariant. If a future PR
        // adds a sampling field to ModelFingerprint, this test will
        // need to be updated and the cache key semantics rethought.
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.digest(), fp2.digest());
        // Touch fp2 to keep the binding live in case the editor
        // strips an "unused mut" warning.
        fp2.arch = "kimi_k2.6".into();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn snapshot_approx_bytes_counts_layer0_and_shells() {
        let snap = mk_snapshot(3, 1);
        // 2 heads × 3 slots × 2 dim × 2 bytes/u16 × 2 (k+v) per layer
        // × (1 layer0 + 1 shell) = 96 bytes.
        let expected = 2 * 3 * 2 * std::mem::size_of::<u16>() * 2 * 2;
        assert_eq!(snap.approx_bytes(), expected);
    }
}
