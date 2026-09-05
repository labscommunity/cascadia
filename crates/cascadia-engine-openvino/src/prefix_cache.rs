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
//! prompt's last `<|im_start|>` (the generation prompt) — [`chat_boundary`] —
//! and the engine splits its prefill chunk there to snapshot exactly at that
//! position. It also snapshots at the end of the turn (prompt + generated),
//! which pays off whenever a template does preserve history verbatim.
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
/// Warm-resume drives position/mask off this, not the matched token count: a turn's last sampled token
/// is never fed back, so KV depth = matched_len-1; using the token count overshoots the mask by one and
/// the attention `Add` fails on shape. `None` if unparseable (caller falls back to the token count).
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

/// [`kv_seq_from_blob`] for a framed multi-stage blob (`frame_blobs`): max depth over its parts (stages
/// share the sequence length, `max` is a safe tie-break). For qwen36 stages / dist-spec draft+target;
/// raw single-stage blobs use [`kv_seq_from_blob`] directly. `None` if unparseable.
pub(crate) fn kv_seq_from_framed_blob(blob: &[u8]) -> Option<usize> {
    let parts = unframe_blobs(blob)?;
    parts.iter().filter_map(|p| kv_seq_from_blob(p)).max()
}

/// Frame N opaque per-stage blobs into one: `u32 count | (u32 len | bytes)×count`. A rank that holds
/// several local stages (qwen36 `stages`, dist-spec target+draft) snapshots each and ships the bundle
/// as a single opaque blob — `OvKvCache` and the wire treat it as one payload.
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

/// Longest tail (prompt tokens past the cached prefix) a warm turn prefills
/// before the engine prefers a cold prefill. Snapshots come only from cold
/// turns (a request's attention KV reads back shallow after a restore), so
/// a conversation's tail grows by one turn per warm turn; at ~400 tok/s on
/// an iGPU this bounds the warm-turn prefill to ~10 s and makes the cold
/// refresh amortise over several turns.
pub const MAX_WARM_TAIL: usize = 4096;

struct Entry {
    tokens: Vec<u32>,
    blob: Arc<Vec<u8>>,
    last_used: u64,
}

/// Byte-bounded LRU of state snapshots keyed by exact token prefix.
pub struct PrefixCache {
    entries: Vec<Entry>,
    budget: usize,
    live: usize,
    tick: u64,
    hits: u64,
    misses: u64,
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
        }
    }

    pub fn enabled(&self) -> bool {
        self.budget > 0
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget
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

    /// Cache `blob` under `tokens`, replacing an entry with the same key and
    /// evicting least-recently-used entries until it fits. A blob larger than
    /// the whole budget (or a key shorter than [`MIN_PREFIX_TOKENS`]) is
    /// dropped; returns whether it was stored.
    pub fn insert(&mut self, tokens: Vec<u32>, blob: Vec<u8>) -> bool {
        if !self.enabled() || tokens.len() < MIN_PREFIX_TOKENS || blob.len() > self.budget {
            return false;
        }
        if let Some(i) = self.entries.iter().position(|e| e.tokens == tokens) {
            let old = self.entries.remove(i);
            self.live -= old.blob.len();
        }
        while self.live + blob.len() > self.budget {
            let Some((i, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
            else {
                break;
            };
            let old = self.entries.remove(i);
            self.live -= old.blob.len();
        }
        self.tick += 1;
        self.live += blob.len();
        self.entries.push(Entry {
            tokens,
            blob: Arc::new(blob),
            last_used: self.tick,
        });
        true
    }

    /// Longest cached entry whose key is a STRICT prefix of `prompt`
    /// (`key.len() < prompt.len()`): the blob and the matched token count.
    /// Non-consuming; marks the entry most-recently-used.
    pub fn longest_prefix(&mut self, prompt: &[u32]) -> Option<(Arc<Vec<u8>>, usize)> {
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tokens.len() < prompt.len() && prompt.starts_with(&e.tokens))
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i);
        match best {
            Some(i) => {
                self.tick += 1;
                self.hits += 1;
                let e = &mut self.entries[i];
                e.last_used = self.tick;
                Some((Arc::clone(&e.blob), e.tokens.len()))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }
}

/// Position of the reusable chat boundary in `prompt`: the number of tokens
/// before the LAST `im_start` token, which begins the generation prompt
/// (`<|im_start|>assistant\n…`). Everything before it is re-sent verbatim by
/// the next turn of the conversation. `None` when the marker is absent
/// (legacy prompts), leads the prompt, or the prefix is too short to be
/// worth a snapshot.
pub fn chat_boundary(prompt: &[u32], im_start: u32) -> Option<usize> {
    let pos = prompt.iter().rposition(|&t| t == im_start)?;
    (pos >= MIN_PREFIX_TOKENS).then_some(pos)
}

/// Snapshot positions for a chat prompt, ascending: the end of the leading
/// system block (the position before the SECOND `im_start`, which a new
/// conversation on the same system prompt re-sends verbatim) and the
/// [`chat_boundary`] (before the last `im_start`, which the next turn of
/// this conversation re-sends). De-duplicated; each ≥ [`MIN_PREFIX_TOKENS`].
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

/// End of the next prefill span starting at `idx`: `chunk` tokens, clamped
/// to the prompt length and to the first snapshot position past `idx` so a
/// span ends exactly on it (the chain state is captured right after that
/// span). `snapshot_at` is ascending.
pub fn next_prefill_end(idx: usize, len: usize, chunk: usize, snapshot_at: &[usize]) -> usize {
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

    fn blob(n: usize) -> Vec<u8> {
        vec![7u8; n]
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
        let (b, len) = c.longest_prefix(&p).expect("hit");
        assert_eq!((b.len(), len), (10, 60));
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
        assert!(c.live_bytes() <= 100);
        assert!(c.contains(&key(20, 0)), "recently used survives");
        assert!(!c.contains(&key(20, 100)), "LRU evicted");
        assert!(c.contains(&key(20, 200)));
    }

    #[test]
    fn oversize_short_and_disabled_are_refused() {
        let mut c = PrefixCache::new(50);
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
    fn same_key_replaces_and_reaccounts() {
        let mut c = PrefixCache::new(100);
        assert!(c.insert(key(20, 0), blob(30)));
        assert!(c.insert(key(20, 0), blob(60)));
        assert_eq!((c.len(), c.live_bytes()), (1, 60));
    }

    #[test]
    fn chat_boundary_is_before_last_im_start() {
        let im = 248045u32;
        // [im, system..., im, user..., im, assistant, think...]
        let mut p = vec![im];
        p.extend(std::iter::repeat_n(5u32, 30));
        p.push(im);
        p.extend(std::iter::repeat_n(6u32, 10));
        p.push(im);
        p.extend([7u32, 8, 9]);
        assert_eq!(chat_boundary(&p, im), Some(42));
        assert_eq!(chat_boundary(&[1, 2, 3], im), None, "no marker");
        assert_eq!(chat_boundary(&[im, 1, 2], im), None, "leading marker only");
        let mut short = vec![im];
        short.extend([1u32; 5]);
        short.push(im);
        assert_eq!(chat_boundary(&short, im), None, "too short to snapshot");
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
    }
}
