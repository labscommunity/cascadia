//! Always-on prefix cache for the Qwen3.5-family staged engine (`qwen35`).
//!
//! The engine's turns are stateful OpenVINO requests: 16 attention layers' KV
//! plus 48 fixed-size Gated-DeltaNet recurrent states. Linear state cannot be
//! trimmed, so a prefix cache on this family is snapshot-at-boundary: the
//! engine snapshots the whole chain's state (`get_state_blob` per stage,
//! framed by [`frame_blobs`]) at positions the NEXT prompt will share
//! verbatim, and a later prompt that starts with those exact tokens restores
//! the blob and prefills only the tail.
//!
//! Which positions? Chat templates in this family render the history
//! assistant turn WITHOUT the `<think>` block the live generation prompt
//! carries, so a snapshot keyed on the previous turn's full sequence never
//! matches the next request. The reusable prefix is everything before the
//! prompt's last `<|im_start|>` (the generation prompt), and the end of the
//! system block before it (shared across conversations) — [`chat_boundaries`]
//! — and the engine splits its prefill chunk there to snapshot exactly at
//! those positions, on every turn that prefills past them.
//!
//! [`PrefixCache`] is a byte-bounded LRU keyed by the exact token sequence;
//! lookups are longest-strict-prefix and non-consuming (a shared system
//! prompt serves every conversation that starts with it). Single-process
//! only: in pipeline mode (`--total > 1`) the downstream ranks' state lives
//! elsewhere, which the `kv_coord` coordination plane handles with its
//! CAPTURE/RESTORE frames; this cache stays idle there.
//!
//! The blob helpers below are shared with the `kv_coord` plane (which
//! re-exports them) so the two paths read and write the same framing.

use std::sync::Arc;

/// Restored KV depth (max `shape[2]` over rank≥3 states) from a `get_state_blob` blob — `[u32 count]`
/// then per state `[u32 name_len][name][u8 dtype][u8 rank][u64×rank shape][u64 nb][data]` (LE).
///
/// Warm-resume drives position/mask off this, not the matched token count. For the kv_coord plane's
/// turn-end keys a turn's last sampled token is never fed back, so KV depth = matched_len-1 and using
/// the token count overshoots the mask by one (the attention `Add` fails on shape); prefix-cache keys
/// are snapshotted at a fold boundary, so depth == key length there. `None` if unparseable (callers
/// fall back to the token count, clamped with `.min(len)`).
pub(crate) fn kv_seq_from_blob(blob: &[u8]) -> Option<usize> {
    fn u32_at(b: &[u8], p: usize) -> Option<u32> {
        Some(u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?))
    }
    fn u64_at(b: &[u8], p: usize) -> Option<u64> {
        Some(u64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?))
    }
    let mut p = 0usize;
    let count = u32_at(blob, p)?;
    p += 4;
    let mut seq = 0usize;
    for _ in 0..count {
        let name_len = u32_at(blob, p)? as usize;
        let name_at = p.checked_add(4)?;
        // Hybrid models (qwen36) mix attention KV — whose shape[2] IS the fold depth — with fixed-shape
        // DeltaNet/SSM recurrent states (conv/ssm) whose shape[2] is a constant (e.g. 128) that would
        // poison the depth max. Only attention states carry the true resume depth; skip the recurrent
        // ones. Pure-attention models (llama/dist-spec) have no conv/ssm names ⇒ unchanged.
        let is_recurrent = blob
            .get(name_at..name_at.checked_add(name_len)?)
            .map(|b| {
                let n = String::from_utf8_lossy(b);
                n.contains("conv") || n.contains("ssm")
            })
            .unwrap_or(false);
        p = name_at.checked_add(name_len)?; // skip name_len + name
        let _dtype = *blob.get(p)?;
        let rank = *blob.get(p.checked_add(1)?)? as usize;
        p = p.checked_add(2)?;
        let mut seq_dim = 0usize;
        for i in 0..rank {
            let d = u64_at(blob, p)? as usize;
            p = p.checked_add(8)?;
            if i == 2 {
                seq_dim = d;
            }
        }
        if rank >= 3 && !is_recurrent {
            seq = seq.max(seq_dim);
        }
        let nb = u64_at(blob, p)? as usize;
        p = p.checked_add(8)?.checked_add(nb)?; // skip nbytes + data
    }
    (seq > 0).then_some(seq)
}

/// [`kv_seq_from_blob`] over per-stage parts: max depth over them (stages share the sequence length,
/// `max` is a safe tie-break). `None` if every part is unparseable.
pub(crate) fn kv_seq_from_parts(parts: &[Vec<u8>]) -> Option<usize> {
    parts.iter().filter_map(|p| kv_seq_from_blob(p)).max()
}

/// [`kv_seq_from_parts`] for a framed multi-stage blob (`frame_blobs`). For qwen36 plane captures /
/// dist-spec draft+target; raw single-stage blobs use [`kv_seq_from_blob`] directly.
pub(crate) fn kv_seq_from_framed_blob(blob: &[u8]) -> Option<usize> {
    kv_seq_from_parts(&unframe_blobs(blob)?)
}

/// Frame N opaque per-stage blobs into one: `u32 count | (u32 len | bytes)×count`. A rank that holds
/// several local stages (qwen36 `stages`, dist-spec target+draft) snapshots each and ships the bundle
/// as a single opaque blob — `OvKvCache` and the wire treat it as one payload. The u32 length is a
/// wire-format fact (plane blobs are capped at 256 MiB); the in-process prefix cache keeps the parts
/// unframed, so multi-GB snapshots never go through here.
pub(crate) fn frame_blobs(blobs: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = 4 + blobs.iter().map(|b| 4 + b.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
    for b in blobs {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

/// Inverse of [`frame_blobs`]. `None` on truncation / over-bound count (forged or corrupt bundle).
pub(crate) fn unframe_blobs(b: &[u8]) -> Option<Vec<Vec<u8>>> {
    if b.len() < 4 {
        return None;
    }
    let count = u32::from_le_bytes(b[0..4].try_into().ok()?) as usize;
    // A rank holds at most a model's worth of stages; cap defensively.
    if count > 1024 {
        return None;
    }
    let mut off = 4;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if off + 4 > b.len() {
            return None;
        }
        let len = u32::from_le_bytes(b[off..off + 4].try_into().ok()?) as usize;
        off += 4;
        if off + len > b.len() {
            return None;
        }
        out.push(b[off..off + len].to_vec());
        off += len;
    }
    if off != b.len() {
        return None; // trailing junk
    }
    Some(out)
}

/// Default byte budget for the snapshot LRU. A Qwen3.8-27B snapshot is
/// ~64 KB per context token (KV) + ~150 MB (DeltaNet state): 2.2 GB at 32 K
/// tokens, 8.5 GB at 128 K. 16 GiB keeps several long-context turns hot
/// without competing with the ~16 GB of int4 weights on a 64 GB box; raise
/// it with `--prefix-cache-gb` for 128 K-class contexts.
pub const DEFAULT_PREFIX_CACHE_BYTES: usize = 16 << 30;

/// Snapshots shorter than this are not worth a `get_state_blob` copy.
pub const MIN_PREFIX_TOKENS: usize = 16;

struct Entry {
    tokens: Vec<u32>,
    /// One `get_state_blob` per stage, unframed (no u32 length field, no copy on hit).
    parts: Arc<Vec<Vec<u8>>>,
    bytes: usize,
    last_used: u64,
}

fn parts_bytes(parts: &[Vec<u8>]) -> usize {
    parts.iter().map(Vec::len).sum()
}

/// Byte-bounded LRU of state snapshots keyed by exact token prefix.
pub struct PrefixCache {
    entries: Vec<Entry>,
    budget: usize,
    live: usize,
    tick: u64,
    hits: u64,
    misses: u64,
    /// Shortest key length whose blob `insert` refused for exceeding the
    /// budget. See [`Self::worth_copying`].
    refused_len: Option<usize>,
    /// The over-budget refusal has been reported once; the condition holds for
    /// every later turn, so warning again would bury the line.
    warned: bool,
}

impl PrefixCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget: budget_bytes,
            live: 0,
            tick: 0,
            hits: 0,
            misses: 0,
            refused_len: None,
            warned: false,
        }
    }

    pub fn enabled(&self) -> bool {
        self.budget > 0
    }

    pub fn live_bytes(&self) -> usize {
        self.live
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// True if an entry keyed by exactly `tokens` is cached (so the caller
    /// can skip the snapshot copy).
    pub fn contains(&self, tokens: &[u32]) -> bool {
        self.entries.iter().any(|e| e.tokens == tokens)
    }

    /// Would a snapshot of `bytes` under a `tokens`-long key fit the budget at
    /// all? [`insert`](Self::insert) gates on this AFTER the copy is paid;
    /// [`worth_copying`](Self::worth_copying) is the gate that comes before it.
    pub fn accepts(&self, tokens: usize, bytes: usize) -> bool {
        self.enabled() && tokens >= MIN_PREFIX_TOKENS && bytes <= self.budget
    }

    /// Is a snapshot at `len` tokens still worth the `get_state_blob` copy? A
    /// snapshot's size is monotone non-decreasing in key length (fixed DeltaNet
    /// state plus KV that only grows), so once a key of length `r` has been
    /// refused for exceeding the budget, no key ≥ `r` can ever fit and the
    /// multi-second copy is pure loss. Without this the copy was paid on every
    /// turn and dropped by [`insert`](Self::insert) every time, which made a
    /// too-small budget strictly slower than `--prefix-cache-gb 0`, forever, at
    /// a 0% hit rate. The threshold is measured, not modelled: the waste is one
    /// refused copy per distinct size class.
    pub fn worth_copying(&self, len: usize) -> bool {
        self.refused_len.is_none_or(|r| len < r)
    }

    /// Cache the per-stage `parts` under `tokens`, replacing an entry with the
    /// same key and evicting least-recently-used entries until it fits. A
    /// snapshot larger than the whole budget (or a key shorter than
    /// [`MIN_PREFIX_TOKENS`]) is dropped; returns whether it was stored. An
    /// over-budget blob also records its key length as the
    /// [`worth_copying`](Self::worth_copying) cut-off, so the next turn does
    /// not pay the same copy again.
    pub fn insert(&mut self, tokens: Vec<u32>, parts: Vec<Vec<u8>>) -> bool {
        let bytes = parts_bytes(&parts);
        if self.enabled() && bytes > self.budget {
            self.refused_len = Some(
                self.refused_len
                    .map_or(tokens.len(), |r| r.min(tokens.len())),
            );
            if !self.warned {
                self.warned = true;
                tracing::warn!(
                    key_len = tokens.len(),
                    bytes,
                    budget = self.budget,
                    "qwen35 prefix-cache: snapshot does not fit the budget and was dropped; \
                     no boundary this long or longer will be copied again. Raise \
                     --prefix-cache-gb, or set it to 0 to turn the cache off."
                );
            }
        }
        if !self.accepts(tokens.len(), bytes) {
            return false;
        }
        if let Some(i) = self.entries.iter().position(|e| e.tokens == tokens) {
            let old = self.entries.remove(i);
            self.live -= old.bytes;
        }
        while self.live + bytes > self.budget {
            // Least recently used first; among entries used at the same tick (a prompt marks
            // every prefix it extends), the LONGEST goes first — it is the most specific one
            // (one conversation's boundary) and a shorter one is shared by more prompts.
            let Some((i, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| (e.last_used, std::cmp::Reverse(e.tokens.len())))
            else {
                break;
            };
            let old = self.entries.remove(i);
            self.live -= old.bytes;
            tracing::info!(
                key_len = old.tokens.len(),
                mib = old.bytes >> 20,
                "qwen35 prefix-cache evicted"
            );
        }
        self.tick += 1;
        self.live += bytes;
        self.entries.push(Entry {
            tokens,
            parts: Arc::new(parts),
            bytes,
            last_used: self.tick,
        });
        // `live` is what bounds a multi-GB cache; every add, replace and evict has to
        // agree with the entries it accounts for.
        debug_assert_eq!(
            self.live,
            self.entries.iter().map(|e| e.bytes).sum::<usize>()
        );
        true
    }

    /// Longest cached entry whose key is a STRICT prefix of `prompt`
    /// (`key.len() < prompt.len()`): the per-stage parts and the matched token
    /// count. Non-consuming (an `Arc` clone, no byte copy). EVERY entry the
    /// prompt extends is marked most-recently-used, not only the longest: a
    /// shared system-block entry is in use by every conversation on it even
    /// though those conversations always match their own, longer boundary
    /// entry — under the old rule it was the LRU victim of its own
    /// conversations' refreshes (measured: a 16 GiB budget at 32 K holds three
    /// 4.45 GB snapshots; the fourth evicted the system block and the next
    /// conversation on that prompt went cold, 110 s).
    pub fn longest_prefix(&mut self, prompt: &[u32]) -> Option<(Arc<Vec<Vec<u8>>>, usize)> {
        self.tick += 1;
        let mut best: Option<(usize, usize)> = None; // (index, key len)
        for (i, e) in self.entries.iter_mut().enumerate() {
            if e.tokens.len() < prompt.len() && prompt.starts_with(&e.tokens) {
                e.last_used = self.tick;
                if best.is_none_or(|(_, len)| len < e.tokens.len()) {
                    best = Some((i, e.tokens.len()));
                }
            }
        }
        match best {
            Some((i, len)) => {
                self.hits += 1;
                Some((Arc::clone(&self.entries[i].parts), len))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }
}

/// The two measured rates the warm-turn refresh rule needs. Both come from
/// the engine's own timings on this box and model, so the rule has no
/// device-dependent constant: an iGPU that prefills at 450 tok/s and copies
/// state at 0.7 GB/s and a CPU that prefills at 60 tok/s and copies at
/// memcpy speed each get their own break-even.
#[derive(Debug, Default, Clone, Copy)]
pub struct RefreshCosts {
    /// Prompt tokens prefilled per second (restore and snapshot time excluded).
    pub prefill_tok_s: Option<f64>,
    /// Seconds per byte of the last snapshot copy.
    pub snapshot_s_per_byte: Option<f64>,
}

impl RefreshCosts {
    /// Should a warm turn refresh the boundary snapshot? A cold turn always
    /// snapshots; on a warm turn a snapshot is a *refresh*: it costs one state
    /// copy now and saves re-prefilling `tail` (the tokens between the
    /// restored prefix and the boundary) on every later turn of the
    /// conversation. Taken once re-prefilling the tail would cost at least as
    /// much as the copy of `state_bytes`, so short follow-ups pay restore +
    /// tail only, the tail can never grow past one copy's worth of prefill
    /// before a refresh, and a conversation never goes cold. Unknown rates
    /// refresh.
    pub fn refresh_pays(&self, tail: usize, state_bytes: usize) -> bool {
        match (self.prefill_tok_s, self.snapshot_s_per_byte) {
            (Some(tok_s), Some(s_per_b)) if tok_s > 0.0 => {
                tail as f64 / tok_s >= state_bytes as f64 * s_per_b
            }
            _ => true,
        }
    }
}

/// Snapshot positions for a chat prompt, ascending: the end of the leading
/// system block (the position before the SECOND `im_start`, which a new
/// conversation on the same system prompt re-sends verbatim) and the
/// chat boundary (before the last `im_start`, which begins the generation
/// prompt and is what the next turn of this conversation re-sends verbatim).
/// De-duplicated; each ≥ [`MIN_PREFIX_TOKENS`].
pub fn chat_boundaries(prompt: &[u32], im_start: u32) -> Vec<usize> {
    let marks: Vec<usize> = prompt
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == im_start)
        .map(|(i, _)| i)
        .collect();
    let mut out = Vec::with_capacity(2);
    if marks.len() >= 2 && marks[1] >= MIN_PREFIX_TOKENS {
        out.push(marks[1]);
    }
    if let Some(&last) = marks.last() {
        if last >= MIN_PREFIX_TOKENS && Some(&last) != out.last() {
            out.push(last);
        }
    }
    out
}

/// Which of `boundaries` (from [`chat_boundaries`], ascending) this turn should
/// snapshot. A boundary is taken when it lies past the restored prefix
/// (`warm_prefix`, 0 on a cold turn), the cache does not already hold that
/// exact key, and — on a warm turn — the tail it would save has grown to cost
/// as much as the copy of `state_bytes` ([`RefreshCosts::refresh_pays`]).
///
/// A boundary the budget has already proved it cannot hold
/// ([`PrefixCache::worth_copying`]) is dropped here, before the multi-second
/// copy, rather than by [`PrefixCache::insert`] after it.
pub fn plan_snapshots(
    prompt: &[u32],
    boundaries: &[usize],
    warm_prefix: usize,
    cache: &PrefixCache,
    costs: &RefreshCosts,
    state_bytes: usize,
) -> Vec<usize> {
    boundaries
        .iter()
        .copied()
        .filter(|&b| {
            b > warm_prefix
                && cache.worth_copying(b)
                && !cache.contains(&prompt[..b])
                && (warm_prefix == 0 || costs.refresh_pays(b - warm_prefix, state_bytes))
        })
        .collect()
}

/// End of the next prefill span starting at `idx`: `chunk` tokens, clamped
/// to the prompt length and to the first snapshot position past `idx` so a
/// span ends exactly on it (the chain state is captured right after that
/// span). `snapshot_at` is ascending.
pub fn next_prefill_end(idx: usize, len: usize, chunk: usize, snapshot_at: &[usize]) -> usize {
    // The search below takes the first entry past `idx` in LIST order, which is the
    // nearest boundary only while the list ascends.
    debug_assert!(snapshot_at.windows(2).all(|w| w[0] < w[1]));
    let mut end = (idx + chunk.max(1)).min(len);
    if let Some(&b) = snapshot_at.iter().find(|&&b| b > idx) {
        if b < end {
            end = b;
        }
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(n: usize) -> Vec<Vec<u8>> {
        vec![vec![7u8; n]]
    }

    fn key(len: usize, seed: u32) -> Vec<u32> {
        (0..len as u32).map(|i| i + seed).collect()
    }

    #[test]
    fn longest_strict_prefix_wins_and_is_non_consuming() {
        let mut c = PrefixCache::new(1 << 20);
        let p = key(100, 0);
        assert!(c.insert(p[..20].to_vec(), blob(10)));
        assert!(c.insert(p[..60].to_vec(), blob(10)));
        assert!(c.insert(key(60, 1), blob(10)), "unrelated key");
        let (parts, len) = c.longest_prefix(&p).expect("hit");
        assert_eq!((parts[0].len(), len), (10, 60));
        // Non-consuming: the same lookup hits again.
        assert_eq!(c.longest_prefix(&p).map(|(_, l)| l), Some(60));
        // An exact match is NOT a strict prefix (nothing left to prefill), so
        // the 60-token prompt falls back to the shorter 20-token entry.
        assert_eq!(c.longest_prefix(&p[..60]).map(|(_, l)| l), Some(20));
        assert!(
            c.longest_prefix(&p[..20]).is_none(),
            "nothing strictly shorter"
        );
        assert!(c.longest_prefix(&key(100, 7)).is_none(), "unrelated prompt");
        assert_eq!(c.stats(), (3, 2));
    }

    #[test]
    fn evicts_least_recently_used_to_fit_budget() {
        let mut c = PrefixCache::new(100);
        assert!(c.insert(key(20, 0), blob(40)));
        assert!(c.insert(key(20, 100), blob(40)));
        // Touch the first entry so the second is the LRU victim.
        assert!(c.longest_prefix(&key(30, 0)).is_some());
        assert!(c.insert(key(20, 200), blob(40)));
        assert_eq!(c.len(), 2);
        assert_eq!(c.live_bytes(), 80, "one victim was enough for 40 + 40");
        assert!(c.contains(&key(20, 0)), "recently used survives");
        assert!(!c.contains(&key(20, 100)), "LRU evicted");
        assert!(c.contains(&key(20, 200)));
    }

    #[test]
    fn evicts_as_many_entries_as_the_new_one_needs() {
        // Three 30-byte entries fill 90 of 100; a 70-byte snapshot needs two victims.
        let mut c = PrefixCache::new(100);
        for seed in [0, 100, 200] {
            assert!(c.insert(key(20, seed), blob(30)));
        }
        assert_eq!((c.len(), c.live_bytes()), (3, 90));
        assert!(c.insert(key(20, 300), blob(70)));
        assert_eq!((c.len(), c.live_bytes()), (2, 100));
        assert!(
            !c.contains(&key(20, 0)) && !c.contains(&key(20, 100)),
            "the two oldest go"
        );
        assert!(c.contains(&key(20, 200)) && c.contains(&key(20, 300)));
    }

    #[test]
    fn oversize_short_and_disabled_are_refused() {
        let mut c = PrefixCache::new(50);
        assert!(!c.accepts(20, 51));
        assert!(c.accepts(20, 50));
        assert!(!c.insert(key(20, 0), blob(51)), "blob over budget");
        assert!(
            !c.insert(key(MIN_PREFIX_TOKENS - 1, 0), blob(1)),
            "key too short"
        );
        assert!(c.is_empty());
        let mut off = PrefixCache::new(0);
        assert!(!off.enabled());
        assert!(!off.insert(key(20, 0), blob(1)));
    }

    #[test]
    fn shared_system_block_survives_its_conversations_refreshes() {
        // Budget holds three 30-byte snapshots. sys (20 tokens) is the shared system block;
        // A1 (40) is conversation A's boundary; a refresh A2 (60) must evict A1, not sys.
        let mut c = PrefixCache::new(100);
        assert!(c.insert(key(20, 0), blob(30)), "sys");
        assert!(c.insert(key(40, 0), blob(30)), "A1");
        // Conversation A's next turn: matches A1, and marks sys as used too.
        assert_eq!(c.longest_prefix(&key(50, 0)).unwrap().1, 40);
        assert!(c.insert(key(60, 0), blob(30)), "A2 refresh (fits: 90)");
        assert!(
            c.insert(key(70, 0), blob(30)),
            "A3 refresh (120 > 100: evicts)"
        );
        assert_eq!(c.len(), 3);
        // sys is still there: a new conversation on the same system prompt starts warm.
        assert_eq!(c.longest_prefix(&key(30, 0)).unwrap().1, 20);
        // And the victim was A1 (the oldest boundary), not sys and not the newer refreshes.
        assert!(!c.contains(&key(40, 0)));
        assert!(c.contains(&key(60, 0)) && c.contains(&key(70, 0)));
    }

    #[test]
    fn refresh_rule_is_the_measured_break_even() {
        // Unknown rates: always refresh (never a cold conversation).
        assert!(RefreshCosts::default().refresh_pays(1, 1 << 30));
        let c = measured_costs();
        let bytes = 1_200_000_000;
        assert!(
            !c.refresh_pays(50, bytes),
            "a one-line follow-up: restore + tail only"
        );
        assert!(!c.refresh_pays(1000, bytes), "2.2 s of tail < 2.6 s copy");
        assert!(c.refresh_pays(1300, bytes), "2.9 s of tail >= 2.6 s copy");
        // Same tail, a CPU box that copies at memcpy speed but prefills slowly: refresh sooner.
        let cpu = RefreshCosts {
            prefill_tok_s: Some(60.0),
            snapshot_s_per_byte: Some(0.4 / 1.2e9),
        };
        assert!(cpu.refresh_pays(50, bytes));
    }

    /// tate-07 at 8 K: 455 tok/s prefill, 1.2 GB copied in 2.6 s.
    fn measured_costs() -> RefreshCosts {
        RefreshCosts {
            prefill_tok_s: Some(455.0),
            snapshot_s_per_byte: Some(2.6 / 1.2e9),
        }
    }

    #[test]
    fn plan_snapshots_follows_the_cold_warm_and_cached_rules() {
        let p = key(4000, 0);
        let mut c = PrefixCache::new(1 << 20);
        let costs = measured_costs();
        let bytes = 1_200_000_000;
        // Cold turn (nothing restored): every boundary.
        assert_eq!(
            plan_snapshots(&p, &[31, 1500], 0, &c, &costs, bytes),
            vec![31, 1500]
        );
        // Warm turn, short tail: re-prefilling 80 tokens is far cheaper than the copy.
        assert!(plan_snapshots(&p, &[1500], 1420, &c, &costs, bytes).is_empty());
        // Warm turn, long tail: 1400 tokens of prefill >= one copy, so refresh.
        assert_eq!(
            plan_snapshots(&p, &[1500], 100, &c, &costs, bytes),
            vec![1500]
        );
        // Behind the restored prefix: nothing to snapshot there.
        assert!(plan_snapshots(&p, &[1500], 1500, &c, &costs, bytes).is_empty());
        // Already cached under that exact key (a shared system block): skipped.
        assert!(c.insert(p[..31].to_vec(), blob(10)));
        assert_eq!(
            plan_snapshots(&p, &[31, 1500], 0, &c, &costs, bytes),
            vec![1500]
        );
    }

    #[test]
    fn plan_snapshots_stops_at_the_refused_length() {
        let p = key(200, 0);
        let costs = RefreshCosts::default();
        // Nothing refused yet: the copy is worth attempting at any length.
        let mut c = PrefixCache::new(100);
        assert_eq!(
            plan_snapshots(&p, &[20, 40], 0, &c, &costs, 0),
            vec![20, 40]
        );
        // A 40-token key overruns the budget. That refusal is the threshold:
        // a snapshot only grows with the key, so 40 and beyond are hopeless.
        assert!(!c.insert(key(40, 0), blob(101)));
        assert!(c.worth_copying(39) && !c.worth_copying(40) && !c.worth_copying(4000));
        assert_eq!(
            plan_snapshots(&p, &[20, 40, 120], 0, &c, &costs, 0),
            vec![20]
        );
        // A shorter refusal lowers the threshold; a longer one leaves it alone.
        assert!(!c.insert(key(60, 0), blob(101)));
        assert!(!c.worth_copying(40) && c.worth_copying(39));
        assert!(!c.insert(key(20, 0), blob(101)));
        assert!(!c.worth_copying(20) && c.worth_copying(19));
        assert!(plan_snapshots(&p, &[20, 40, 120], 0, &c, &costs, 0).is_empty());
    }

    #[test]
    fn multi_part_snapshots_account_every_stage() {
        let mut c = PrefixCache::new(100);
        assert!(c.insert(key(20, 0), vec![vec![1u8; 30], vec![2u8; 30]]));
        assert_eq!(c.live_bytes(), 60);
        assert!(
            !c.insert(key(20, 1), vec![vec![1u8; 60], vec![2u8; 60]]),
            "120 > budget"
        );
        let (parts, _) = c.longest_prefix(&key(30, 0)).unwrap();
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn same_key_replaces_and_reaccounts() {
        let mut c = PrefixCache::new(100);
        assert!(c.insert(key(20, 0), blob(30)));
        assert!(c.insert(key(20, 0), blob(60)));
        assert_eq!((c.len(), c.live_bytes()), (1, 60));
    }

    #[test]
    fn prefill_span_ends_on_the_snapshot_boundary() {
        assert_eq!(next_prefill_end(0, 1000, 256, &[]), 256);
        assert_eq!(next_prefill_end(900, 1000, 256, &[]), 1000);
        assert_eq!(next_prefill_end(0, 1000, 256, &[100]), 100);
        assert_eq!(
            next_prefill_end(100, 1000, 256, &[100]),
            356,
            "boundary behind"
        );
        assert_eq!(
            next_prefill_end(0, 1000, 256, &[256]),
            256,
            "boundary on the edge"
        );
        assert_eq!(
            next_prefill_end(0, 1000, 256, &[2000]),
            256,
            "boundary past the end"
        );
        assert_eq!(
            next_prefill_end(0, 1000, 256, &[100, 300]),
            100,
            "first boundary first"
        );
        assert_eq!(
            next_prefill_end(100, 1000, 256, &[100, 300]),
            300,
            "then the next"
        );
        assert_eq!(
            next_prefill_end(300, 1000, 256, &[100, 300]),
            556,
            "none left"
        );
        assert_eq!(next_prefill_end(0, 10, 0, &[]), 1, "chunk never zero");
    }

    #[test]
    fn chat_boundaries_cover_system_block_and_last_turn() {
        let im = 248045u32;
        // [im system(30)] [im user(10)] [im assistant...]
        let mut p = vec![im];
        p.extend(std::iter::repeat_n(5u32, 30));
        p.push(im); // index 31: end of the system block
        p.extend(std::iter::repeat_n(6u32, 10));
        p.push(im); // index 42: generation prompt
        p.extend([7u32, 8, 9]);
        assert_eq!(chat_boundaries(&p, im), vec![31, 42]);
        // Single-turn prompt without a system block: one boundary.
        let mut q = vec![im];
        q.extend(std::iter::repeat_n(5u32, 30));
        q.push(im);
        q.extend([7u32, 8]);
        assert_eq!(chat_boundaries(&q, im), vec![31]);
        // Short system block is skipped; the last boundary still counts.
        let mut r = vec![im, 1, 2, im];
        r.extend(std::iter::repeat_n(6u32, 30));
        r.push(im);
        assert_eq!(chat_boundaries(&r, im), vec![34]);
        assert!(chat_boundaries(&[1, 2, 3], im).is_empty());
    }

    #[test]
    fn frame_roundtrip_and_depth() {
        let parts = vec![vec![1u8, 2, 3], vec![], vec![9u8; 5]];
        let framed = frame_blobs(&parts);
        assert_eq!(unframe_blobs(&framed).unwrap(), parts);
        assert!(
            unframe_blobs(&framed[..framed.len() - 1]).is_none(),
            "truncated"
        );
        assert!(
            kv_seq_from_framed_blob(&framed).is_none(),
            "not state blobs"
        );
        assert!(kv_seq_from_parts(&parts).is_none());
    }
}
