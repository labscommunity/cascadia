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

use crate::kv_prefix_cache::{ModelFingerprint, LOCAL_NS};

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
    /// Issue-34 H.1a: the namespace this entry is visible in over the wire. `lookup_local` (local
    /// resume) does not filter on this; `lookup_ns` (cross-chain NEGOTIATE) does. Mirrors `KvPrefixCache`.
    partner: String,
    prefix: Vec<i64>,
    snapshot: OvMoeKvSnapshot,
    /// Pulled over the KV plane (`insert_pulled`) rather than captured from this rank's own turn.
    /// Read back by [`OvMoeKvPrefixCache::lookup_local`] so a cert bar can tell the two apart.
    plane_pulled: bool,
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
    /// model fingerprint. Returns `(snapshot, plane_pulled)`; the snapshot's
    /// `past_seq_len` is the matched length. On a hit the entry moves to MRU.
    ///
    /// LOCAL RESUME ONLY — not tenant-namespaced. Wire-facing paths MUST use [`Self::lookup_ns`].
    pub fn lookup_local(
        &mut self,
        prompt: &[i64],
        fingerprint: &ModelFingerprint,
    ) -> Option<(OvMoeKvSnapshot, bool)> {
        self.lookup_impl(None, prompt, fingerprint)
    }

    /// As [`Self::lookup_local`], but confined to entries tagged `partner` (issue-34 H.1a). Backs the
    /// cross-chain wire plane; the namespace filter runs BEFORE the longest-prefix scan so a
    /// cross-tenant probe reads as an empty cache, never a truncated length. [`Self::lookup_local`]
    /// itself stays unconfined (local resume — H.1 §5 non-goal). Mirrors `KvPrefixCache::lookup_ns`.
    pub fn lookup_ns(
        &mut self,
        partner: &str,
        prompt: &[i64],
        fingerprint: &ModelFingerprint,
    ) -> Option<(OvMoeKvSnapshot, bool)> {
        self.lookup_impl(Some(partner), prompt, fingerprint)
    }

    fn lookup_impl(
        &mut self,
        partner_filter: Option<&str>,
        prompt: &[i64],
        fingerprint: &ModelFingerprint,
    ) -> Option<(OvMoeKvSnapshot, bool)> {
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
            if let Some(partner) = partner_filter {
                if entry.partner != partner {
                    continue;
                }
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

    /// Insert a snapshot keyed by `prefix` + fingerprint. Replaces an
    /// exact-key entry in place; evicts LRU entries until under capacity.
    ///
    /// Local capture — this rank's own turn. Cross-chain consumer-inserts go
    /// through [`Self::insert_pulled`].
    pub fn insert(
        &mut self,
        prefix: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: OvMoeKvSnapshot,
    ) {
        self.store(LOCAL_NS, prefix, fingerprint, snapshot, false);
    }

    /// As [`Self::insert`], but marks the entry as pulled over the KV plane so a later
    /// warm resume off it is attributable to the cross-chain pull, not to a local capture, and tags
    /// it with `partner` — the tenant this pull was served to (issue-34 H.1a) — so
    /// [`Self::lookup_ns`] can confine a later NEGOTIATE to it.
    pub fn insert_pulled(
        &mut self,
        partner: &str,
        prefix: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: OvMoeKvSnapshot,
    ) {
        self.store(partner, prefix, fingerprint, snapshot, true);
    }

    fn store(
        &mut self,
        partner: &str,
        prefix: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: OvMoeKvSnapshot,
        plane_pulled: bool,
    ) {
        if !self.enabled() {
            return;
        }
        debug_assert_eq!(prefix.len(), snapshot.past_seq_len);
        let key = CacheKey {
            model_digest: fingerprint.digest(),
            prefix_hash: hash_prefix(&prefix),
        };
        // De-dup within (partner, key, prefix) only — see `KvPrefixCache::store` for why.
        if let Some(pos) = self
            .entries
            .iter()
            .position(|(k, e)| *k == key && e.partner == partner && e.prefix == prefix)
        {
            self.entries.remove(pos);
        }
        while self.entries.len() >= self.capacity {
            if self.entries.pop_back().is_none() {
                break;
            }
        }
        self.entries.push_front((
            key,
            Entry {
                partner: partner.to_string(),
                prefix,
                snapshot,
                plane_pulled,
            },
        ));
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
        let (got, _) = c.lookup_local(&prompt, &fp()).expect("hit");
        assert_eq!(got.past_seq_len, 3);
        assert_eq!(got.layers[0].k[0], 7.0);
    }

    #[test]
    fn lookup_reports_entry_provenance() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert(vec![1i64, 2, 3], &fp(), snap(3, 1.0));
        let (_, pulled) = c.lookup_local(&[1i64, 2, 3, 4], &fp()).expect("hit");
        assert!(!pulled, "locally captured");
        c.insert_pulled("peer", vec![1i64, 2, 3], &fp(), snap(3, 2.0));
        let (_, pulled) = c.lookup_local(&[1i64, 2, 3, 4], &fp()).expect("hit");
        assert!(pulled, "consumer-inserted over the plane");
    }

    /// LRU must fail safe: once the pulled entry is evicted, the local entry that still
    /// matches carries its own mark, so a plane assertion can't green off a local capture.
    #[test]
    fn evicting_a_pulled_entry_leaves_a_local_mark() {
        let mut c = OvMoeKvPrefixCache::new(2);
        c.insert_pulled("peer", vec![1i64, 1, 1], &fp(), snap(3, 1.0));
        c.insert(vec![2i64, 2, 2], &fp(), snap(3, 2.0));
        c.insert(vec![3i64, 3, 3], &fp(), snap(3, 3.0));
        assert_eq!(c.len(), 2);
        let evicted = c.lookup_local(&[1i64, 1, 1, 9], &fp());
        assert!(evicted.is_none(), "pulled was LRU");
        let (_, pulled) = c.lookup_local(&[2i64, 2, 2, 9], &fp()).expect("hit");
        assert!(!pulled);
    }

    #[test]
    fn full_prompt_match_is_rejected_strict_prefix() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert(vec![1i64, 2, 3], &fp(), snap(3, 1.0));
        // prompt == cached prefix: plen >= prompt.len() → no hit.
        assert!(c.lookup_local(&[1i64, 2, 3], &fp()).is_none());
    }

    #[test]
    fn lookup_returns_longest_matching_prefix() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert(vec![1i64, 2, 3], &fp(), snap(3, 3.0));
        c.insert(vec![1i64, 2, 3, 4, 5], &fp(), snap(5, 5.0));
        let (got, _) = c.lookup_local(&[1i64, 2, 3, 4, 5, 6], &fp()).expect("hit");
        assert_eq!(got.past_seq_len, 5);
    }

    /// Issue-34 H.1a: `lookup_ns` is the cross-chain wire path — a cross-tenant probe must read as
    /// an empty cache (`None`), never a truncated length.
    #[test]
    fn lookup_ns_is_confined_to_the_callers_namespace() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert_pulled("tenant-a", vec![11i64, 22, 33], &fp(), snap(3, 1.0));
        assert!(
            c.lookup_ns("tenant-b", &[11i64, 22, 33, 44], &fp())
                .is_none(),
            "cross-tenant probe must miss"
        );
        assert_eq!(
            c.lookup_ns("tenant-a", &[11i64, 22, 33, 44], &fp())
                .map(|(s, _)| s.past_seq_len),
            Some(3),
            "the owner still hits"
        );
    }

    /// The namespace filter must run BEFORE the longest-prefix scan: a coincidentally-longer entry
    /// belonging to another tenant must not mask the caller's own legitimate (shorter) hit.
    #[test]
    fn lookup_ns_does_not_let_a_longer_other_tenant_entry_mask_the_callers_own_hit() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert_pulled("tenant-a", vec![1i64, 2, 3, 4, 5], &fp(), snap(5, 9.0));
        c.insert_pulled("tenant-b", vec![1i64, 2, 3], &fp(), snap(3, 7.0));
        let (snap, _) = c
            .lookup_ns("tenant-b", &[1i64, 2, 3, 4, 5, 6], &fp())
            .expect("tenant-b's own shorter entry must still hit");
        assert_eq!(snap.past_seq_len, 3);
    }

    /// De-dup on insert must be scoped to `(partner, tokens)`, not tokens alone.
    #[test]
    fn insert_dedup_is_scoped_to_partner() {
        let mut c = OvMoeKvPrefixCache::new(4);
        c.insert_pulled("tenant-a", vec![1i64, 2, 3], &fp(), snap(3, 1.0));
        c.insert_pulled("tenant-b", vec![1i64, 2, 3], &fp(), snap(3, 2.0));
        assert_eq!(c.len(), 2, "both tenants' entries survive");
        assert_eq!(
            c.lookup_ns("tenant-a", &[1i64, 2, 3, 9], &fp())
                .map(|(s, _)| s.layers[0].k[0]),
            Some(1.0),
            "tenant-a's entry must be intact, not overwritten by tenant-b's insert"
        );
    }

    #[test]
    fn disabled_cache_never_hits() {
        let mut c = OvMoeKvPrefixCache::new(0);
        c.insert(vec![1i64, 2], &fp(), snap(2, 1.0));
        assert!(c.lookup_local(&[1i64, 2, 3], &fp()).is_none());
        assert_eq!(c.len(), 0);
    }
}
