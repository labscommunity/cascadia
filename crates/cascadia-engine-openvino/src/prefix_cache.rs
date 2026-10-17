//! Always-on prefix cache for the Qwen3.5-family staged engine (`qwen35`).
//!
//! The engine's turns are stateful OpenVINO requests: attention layers' KV
//! plus fixed-size Gated-DeltaNet recurrent states (16 + 48 on the 27B,
//! 10 + 30 of 40 on the Qwen3.6-35B-A3B this cache also serves). Linear state
//! cannot be trimmed, so a prefix cache on this family is
//! snapshot-at-boundary: the engine snapshots the whole chain's state (one
//! `get_state_blob` per stage, kept unframed — [`frame_blobs`] is the
//! `kv_coord` wire format, not this) at positions the NEXT prompt will share
//! verbatim, and a later prompt that starts with those exact tokens restores
//! the parts and prefills only the tail.
//!
//! Which positions? Chat templates in this family render the history
//! assistant turn WITHOUT the `<think>` block the live generation prompt
//! carries, so a snapshot keyed on the previous turn's full sequence never
//! matches the next request. The reusable prefix is everything before the
//! prompt's last `<|im_start|>` (the generation prompt), and the end of the
//! system block before it (shared across conversations) — [`chat_boundaries`]
//! — and the engine splits its prefill chunk there to snapshot exactly at
//! those positions. Which of them a given turn takes is [`plan_snapshots`]:
//! every boundary on a cold turn; on a warm turn only as a *refresh*, once
//! re-prefilling the tail would cost as much as the copy
//! ([`RefreshCosts::refresh_pays`]); never a key the cache already holds, and
//! never one at least as long as a key the budget has already refused
//! ([`PrefixCache::worth_copying`]).
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

/// Exact snapshot size model `(fixed, per_token)` read from one snapshot's per-stage
/// `get_state_blob` parts (same layout as [`kv_seq_from_blob`]): a snapshot at `len` tokens is
/// `fixed + per_token × len` bytes. Each attention KV state (rank ≥ 3, not `conv`/`ssm`) holds
/// `nb / shape[2]` bytes per token at its own precision; everything else (recurrent state,
/// names, shapes, counts) is fixed. One blob at any depth is enough, so the engine's warmup
/// state gives the slope before the first request (P4-CE-1). `None` when a part does not parse
/// exactly, a KV state's bytes are not a whole multiple of its depth, or no KV state is found.
pub(crate) fn state_size_model(parts: &[Vec<u8>]) -> Option<(usize, usize)> {
    fn u32_at(b: &[u8], p: usize) -> Option<u32> {
        Some(u32::from_le_bytes(
            b.get(p..p.checked_add(4)?)?.try_into().ok()?,
        ))
    }
    fn u64_at(b: &[u8], p: usize) -> Option<u64> {
        Some(u64::from_le_bytes(
            b.get(p..p.checked_add(8)?)?.try_into().ok()?,
        ))
    }
    let (mut fixed, mut per_token, mut kv_states) = (0usize, 0usize, 0usize);
    for blob in parts {
        let mut p = 0usize;
        let count = u32_at(blob, p)?;
        p += 4;
        let mut kv_bytes = 0usize;
        for _ in 0..count {
            let name_len = u32_at(blob, p)? as usize;
            let name_at = p.checked_add(4)?;
            let name = blob.get(name_at..name_at.checked_add(name_len)?)?;
            let n = String::from_utf8_lossy(name);
            let recurrent = n.contains("conv") || n.contains("ssm");
            p = name_at.checked_add(name_len)?;
            let rank = *blob.get(p.checked_add(1)?)? as usize;
            p = p.checked_add(2)?;
            let mut seq = 0usize;
            for i in 0..rank {
                let d = u64_at(blob, p)? as usize;
                p = p.checked_add(8)?;
                if i == 2 {
                    seq = d;
                }
            }
            let nb = u64_at(blob, p)? as usize;
            p = p.checked_add(8)?.checked_add(nb)?;
            if p > blob.len() {
                return None;
            }
            if rank >= 3 && !recurrent {
                if seq == 0 || nb % seq != 0 {
                    return None;
                }
                per_token = per_token.checked_add(nb / seq)?;
                kv_bytes = kv_bytes.checked_add(nb)?;
                kv_states += 1;
            }
        }
        if p != blob.len() {
            return None;
        }
        fixed = fixed.checked_add(blob.len() - kv_bytes)?;
    }
    (kv_states > 0).then_some((fixed, per_token))
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
/// ~150 MB of fixed DeltaNet state plus ~130 KB per context token as
/// serialised by `get_state_blob` — about twice the f16 KV footprint —
/// measured on the B390 at 1.2 GB for 8 K and 4.45 GB for 32 K, so ~17 GB at
/// 128 K: a 128 K snapshot needs a budget above this default. Note the fixed
/// term: bytes-per-token read off a short snapshot does not extrapolate.
/// 16 GiB keeps several long-context turns hot without competing
/// with the ~16 GB of int4 weights on a 64 GB box; raise it with
/// `--prefix-cache-gb` for 128 K-class contexts. The CLI clamps this default
/// to a quarter of physical RAM (`resolve_prefix_cache_bytes`), so a 32 GB
/// box gets 8 GiB of it.
pub const DEFAULT_PREFIX_CACHE_BYTES: usize = 16 << 30;

/// Snapshots shorter than this are not worth a `get_state_blob` copy (shared with the API, which
/// never counts a reuse discount below it).
pub use cascadia_types::MIN_PREFIX_TOKENS;

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
    /// Shortest and longest key length `insert` has seen, with their snapshot
    /// bytes: the measured size model behind [`Self::estimated_bytes`].
    size_obs: Option<((usize, usize), (usize, usize))>,
    /// Lower bound on snapshot bytes seeded at load, `(fixed, per_token)`: see
    /// [`Self::seed_floor`].
    floor: Option<(usize, usize)>,
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
            size_obs: None,
            floor: None,
        }
    }

    /// Seed a lower bound on snapshot bytes, `fixed + per_token × len`, known before any snapshot
    /// is taken (the engine derives it from the tree's config.json at load, P3-CE-1). With it,
    /// [`Self::estimated_bytes`] is never unknown, so a snapshot that can never fit is refused
    /// before the first request's prefill. A lower bound never refuses a snapshot that fits.
    pub fn seed_floor(&mut self, fixed: usize, per_token: usize) {
        self.floor = Some((fixed, per_token));
    }

    /// Record a measured snapshot size without caching anything (the engine's warmup state, so
    /// the size model starts from a real blob). No entry, no [`Self::worth_copying`] cut-off.
    pub fn observe(&mut self, len: usize, bytes: usize) {
        let obs = (len, bytes);
        self.size_obs = Some(match self.size_obs {
            None => (obs, obs),
            Some((lo, hi)) => (
                if obs.0 < lo.0 { obs } else { lo },
                if obs.0 >= hi.0 { obs } else { hi },
            ),
        });
    }

    /// Record a measured snapshot's per-stage parts as the size model, without caching anything
    /// (the engine's warmup state, P4-CE-1). When the parts parse ([`state_size_model`]) the
    /// model's line is recorded exactly, as the points `(0, fixed)` and `(1, fixed + per_token)`,
    /// and returned; otherwise the total is recorded at the parts' KV depth (else one token), a
    /// single point that only bounds longer keys from below.
    pub fn observe_state(&mut self, parts: &[Vec<u8>]) -> Option<(usize, usize)> {
        match state_size_model(parts) {
            Some((fixed, per_token)) => {
                self.observe(0, fixed);
                self.observe(1, fixed.saturating_add(per_token));
                Some((fixed, per_token))
            }
            None => {
                let len = kv_seq_from_parts(parts).unwrap_or(1);
                self.observe(len, parts_bytes(parts));
                None
            }
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

    /// Estimated snapshot bytes for a `len`-token key, from the snapshots
    /// `insert` (and [`Self::observe`]) measured: a fixed (DeltaNet) part plus KV
    /// linear in the key, fitted through the shortest and longest key seen. With
    /// one length measured, a key at least that long is at least that large
    /// (monotone). Never below the seeded floor ([`Self::seed_floor`]); `None`
    /// only when neither a measurement nor a floor bounds it.
    pub fn estimated_bytes(&self, len: usize) -> Option<usize> {
        let measured = self.size_obs.and_then(|((l1, b1), (l2, b2))| {
            if l2 > l1 && b2 >= b1 {
                let per_token = (b2 - b1) as f64 / (l2 - l1) as f64;
                Some((b1 as f64 + per_token * (len as f64 - l1 as f64)).max(0.0) as usize)
            } else {
                (len >= l2).then_some(b2)
            }
        });
        let floor = self
            .floor
            .map(|(fixed, per)| fixed.saturating_add(per.saturating_mul(len)));
        measured.max(floor)
    }

    /// Can the cache structurally hold a snapshot at `len` tokens: enabled,
    /// at least [`MIN_PREFIX_TOKENS`], under the measured over-budget
    /// threshold ([`Self::worth_copying`]) and an estimated size within the
    /// budget ([`Self::estimated_bytes`]; unknown counts as fitting, which the
    /// engine's load-time floor rules out, P3-CE-1). Says
    /// nothing about eviction: a snapshot that fits can still be evicted later.
    pub fn can_hold(&self, len: usize) -> bool {
        self.worth_copying(len) && self.accepts(len, self.estimated_bytes(len).unwrap_or(0))
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
        self.observe(tokens.len(), bytes);
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
/// state at ~0.5 GB/s (measured on the B390: 1.2 GB in 2.5 s, 4.45 GB in 9 s)
/// and a CPU that prefills at 60 tok/s and copies at memcpy speed each get
/// their own break-even.
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

/// Snapshot positions for a chat prompt, ascending: the end of the first
/// `<|im_start|>` block (the system block when the prompt has one — only
/// `im_start` occurrences are counted, no role is read), which is the
/// position before the SECOND `im_start` and what a new conversation on the
/// same system prompt re-sends verbatim; and the chat boundary (before the
/// last `im_start`, which begins the generation prompt and is what the next
/// turn of this conversation re-sends verbatim).
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

/// A lower bound on a snapshot's bytes from the tree's config.json (`text_config` when nested),
/// `(fixed, per_token)`: the linear-attention layers' recurrent state
/// (`linear_num_value_heads × linear_key_head_dim × linear_value_head_dim` each) and the
/// full-attention layers' K and V (`2 × num_key_value_heads × head_dim` per token each), at half
/// a byte per element, the smallest OpenVINO state precision (u4 KV). Conv state and framing are
/// left out. Which layers are which comes from `layer_types`, else `full_attention_interval`
/// over `num_hidden_layers`. `None` when a dimension is missing.
pub fn config_state_floor(cfg: &serde_json::Value) -> Option<(usize, usize)> {
    let c = if cfg.get("text_config").is_some() {
        &cfg["text_config"]
    } else {
        cfg
    };
    let dim = |k: &str| c[k].as_u64().map(|v| v as usize);
    let (full, linear) = match c["layer_types"].as_array() {
        Some(types) => {
            let count = |t: &str| types.iter().filter(|x| x.as_str() == Some(t)).count();
            (count("full_attention"), count("linear_attention"))
        }
        None => {
            let (n, every) = (dim("num_hidden_layers")?, dim("full_attention_interval")?);
            if every == 0 {
                return None;
            }
            (n / every, n - n / every)
        }
    };
    let fixed = linear
        .checked_mul(dim("linear_num_value_heads")?)?
        .checked_mul(dim("linear_key_head_dim")?)?
        .checked_mul(dim("linear_value_head_dim")?)?
        / 2;
    let per_token = full
        .checked_mul(2)?
        .checked_mul(dim("num_key_value_heads")?)?
        .checked_mul(dim("head_dim")?)?
        / 2;
    Some((fixed, per_token))
}

/// Which decision `boundaries` (the request's shared prefix, `cache_prefix`)
/// this pass snapshots: every one past the restored prefix that the cache can
/// hold ([`PrefixCache::can_hold`]) and does not already hold. Unlike
/// [`plan_snapshots`] there is no refresh rule: a request whose later passes
/// were admitted on the shared-prefix discount must find this exact key, even
/// when the pass restored a shorter entry (another request's state that this
/// one extends) and the chat rule would judge the copy not worth it.
pub fn plan_decision_snapshots(
    prompt: &[u32],
    boundaries: &[usize],
    warm_prefix: usize,
    cache: &PrefixCache,
) -> Vec<usize> {
    boundaries
        .iter()
        .copied()
        .filter(|&b| b > warm_prefix && cache.can_hold(b) && !cache.contains(&prompt[..b]))
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
    fn a_decision_prefix_is_snapshotted_whatever_the_refresh_rule() {
        // P2-CE-1: request A cached its Kev state (key 40); request B's state extends it (its
        // cache_prefix is 60). B's first pass restores A's entry, so the chat refresh rule sees a
        // 20-token tail against a 1.2 GB copy and skips the boundary. A decision boundary is
        // taken anyway: B's later passes must find it.
        let p = key(200, 0);
        let mut c = PrefixCache::new(1 << 20);
        assert!(c.insert(p[..40].to_vec(), blob(10)));
        let costs = measured_costs();
        let bytes = 1_200_000_000;
        assert!(
            plan_snapshots(&p, &[60], 40, &c, &costs, bytes).is_empty(),
            "chat behaviour unchanged"
        );
        assert_eq!(plan_decision_snapshots(&p, &[60], 40, &c), vec![60]);
        assert_eq!(plan_decision_snapshots(&p, &[60], 0, &c), vec![60], "cold");
        // Not past the restored prefix, or already cached: nothing to copy.
        assert!(plan_decision_snapshots(&p, &[60], 60, &c).is_empty());
        assert!(c.insert(p[..60].to_vec(), blob(10)));
        assert!(plan_decision_snapshots(&p, &[60], 40, &c).is_empty());
        // B's later passes then restore B's own (longer) entry, not A's.
        assert_eq!(c.longest_prefix(&p[..61]).unwrap().1, 60);
        // A boundary the cache cannot hold is not copied at all.
        let mut tiny = PrefixCache::new(1 << 20);
        assert!(plan_decision_snapshots(&p, &[MIN_PREFIX_TOKENS - 1], 0, &tiny).is_empty());
        assert!(!tiny.insert(key(40, 0), blob((1 << 20) + 1)));
        assert!(plan_decision_snapshots(&p, &[60], 0, &tiny).is_empty());
    }

    #[test]
    fn can_hold_is_the_structural_check_before_any_prefill() {
        // Disabled, or a key too short to store: never.
        assert!(!PrefixCache::new(0).can_hold(1000));
        let mut c = PrefixCache::new(1000);
        assert!(!c.can_hold(MIN_PREFIX_TOKENS - 1));
        assert!(c.can_hold(MIN_PREFIX_TOKENS));
        // Nothing measured or seeded: the size is unknown, so the snapshot is attempted. The
        // engine seeds a floor at load, so this only holds for a bare cache.
        assert_eq!(c.estimated_bytes(10_000), None);
        assert!(c.can_hold(10_000));
        // One snapshot measured: a longer key is at least as large (monotone).
        assert!(c.insert(key(100, 0), blob(300)));
        assert_eq!(c.estimated_bytes(100), Some(300));
        assert_eq!(c.estimated_bytes(500), Some(300));
        assert_eq!(c.estimated_bytes(50), None);
        // Two lengths measured: fixed state + per-token KV, extrapolated.
        assert!(c.insert(key(200, 0), blob(500)));
        assert_eq!(c.estimated_bytes(300), Some(700));
        assert_eq!(c.estimated_bytes(150), Some(400));
        assert!(c.can_hold(450), "100 + 2 * 450 = 1000 fits");
        assert!(!c.can_hold(500), "100 + 2 * 500 = 1100 cannot ever fit");
        // A measured refusal stays authoritative.
        assert!(!c.insert(key(300, 0), blob(1001)));
        assert!(!c.can_hold(300) && c.can_hold(299));
    }

    #[test]
    fn a_seeded_size_floor_refuses_before_any_snapshot_was_measured() {
        // P3-CE-1: on a fresh server nothing has been measured. The floor seeded at load (the
        // config lower bound, or the warmup's measured blob) must make a snapshot that can never
        // fit refuse up front, not count as fitting.
        let mut c = PrefixCache::new(1 << 20);
        assert!(c.can_hold(9001), "unseeded: unknown");
        c.seed_floor(25 << 20, 0);
        assert_eq!(c.estimated_bytes(MIN_PREFIX_TOKENS), Some(25 << 20));
        assert!(!c.can_hold(MIN_PREFIX_TOKENS) && !c.can_hold(9001));
        // Fixed bytes plus a per-token term.
        let mut c = PrefixCache::new(1000);
        c.seed_floor(100, 2);
        assert_eq!(c.estimated_bytes(450), Some(1000));
        assert!(c.can_hold(450) && !c.can_hold(451));
        // A measured observation recorded without caching anything: no entry, no refusal.
        c.observe(1, 350);
        assert!(c.is_empty() && c.live_bytes() == 0 && c.worth_copying(1_000_000));
        assert_eq!(
            c.estimated_bytes(100),
            Some(350),
            "measured above the floor"
        );
        assert_eq!(
            c.estimated_bytes(400),
            Some(900),
            "the floor still bounds it from below"
        );
        assert!(!c.can_hold(451));
        // The measured line through (1, 350) and a real snapshot extrapolates above the floor.
        assert!(c.insert(key(101, 0), blob(600)));
        assert_eq!(c.estimated_bytes(201), Some(850));
        assert!(c.can_hold(250), "350 + 2.5 * 249 fits");
        assert!(
            !c.can_hold(300),
            "350 + 2.5 * 299 > 1000, though the floor alone is 700"
        );
    }

    /// A `get_state_blob` blob in the shim's layout: `(name, dtype, shape, element bytes)` per state.
    fn state_blob(states: &[(&str, &[u64], u64)]) -> Vec<u8> {
        let mut b = (states.len() as u32).to_le_bytes().to_vec();
        for (name, shape, elem) in states {
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.push(1); // dtype tag; the size model reads only nb
            b.push(shape.len() as u8);
            for d in *shape {
                b.extend_from_slice(&d.to_le_bytes());
            }
            let nb = shape.iter().product::<u64>() * elem;
            b.extend_from_slice(&nb.to_le_bytes());
            b.extend(std::iter::repeat_n(0u8, nb as usize));
        }
        b
    }

    /// A small hybrid stage at `seq` tokens: recurrent conv/ssm state (fixed) and f32 KV.
    fn hybrid_stage(seq: u64) -> Vec<u8> {
        state_blob(&[
            ("layer0.conv_state", &[1, 48, 4], 4),
            ("layer0.ssm_state", &[1, 4, 8, 8], 4),
            ("layer3.key_cache", &[1, 2, seq, 16], 4),
            ("layer3.value_cache", &[1, 2, seq, 16], 4),
            ("layer7.key_cache", &[1, 2, seq, 16], 2),
            ("layer7.value_cache", &[1, 2, seq, 16], 2),
        ])
    }

    #[test]
    fn the_state_size_model_read_from_one_blob_predicts_every_length() {
        // P4-CE-1: the warmup blob alone gives the exact per-token slope (each KV state's bytes
        // over its sequence depth) and the fixed part (recurrent state plus framing), so the size
        // estimate is exact from the first request, not a lower bound.
        let warm = vec![hybrid_stage(1), hybrid_stage(1)];
        let (fixed, per_token) = state_size_model(&warm).unwrap();
        assert_eq!(per_token, 2 * (2 * 2 * 16 * 4 + 2 * 2 * 16 * 2));
        for seq in [1u64, 3, 17, 1000] {
            let at = hybrid_stage(seq).len() * 2;
            assert_eq!(fixed + per_token * seq as usize, at, "seq {seq}");
        }
        // Read at any depth, the model is the same.
        assert_eq!(
            state_size_model(&[hybrid_stage(5), hybrid_stage(5)]),
            Some((fixed, per_token))
        );
        // No KV state, or an unparseable blob: no model (the caller falls back).
        let recurrent_only = state_blob(&[("layer0.ssm_state", &[1, 4, 8, 8], 4)]);
        assert_eq!(state_size_model(&[recurrent_only]), None);
        assert_eq!(state_size_model(&[vec![1, 2, 3]]), None);
        assert_eq!(state_size_model(&[]), None);
        let mut torn = hybrid_stage(1);
        torn.pop();
        assert_eq!(state_size_model(&[torn]), None);
        let mut trailing = hybrid_stage(1);
        trailing.push(0);
        assert_eq!(state_size_model(&[trailing]), None);
    }

    #[test]
    fn the_warmup_blob_makes_the_estimate_exact_from_the_first_request() {
        // P4-CE-1 trace, scaled down: the config floor (half a byte per element) admits a key
        // whose real snapshot is over the budget; the warmup blob's size model refuses it before
        // any prefill, and still admits one that fits.
        let warm = vec![hybrid_stage(1)];
        let (fixed, per_token) = state_size_model(&warm).unwrap();
        let real = |len: usize| fixed + per_token * len;
        let budget = real(400);
        let mut c = PrefixCache::new(budget);
        c.seed_floor(fixed / 8, per_token / 8);
        assert!(c.can_hold(1000), "the floor alone admits a 1000-token key");
        assert_eq!(c.observe_state(&warm), Some((fixed, per_token)));
        assert!(c.is_empty() && c.worth_copying(1_000_000));
        assert_eq!(c.estimated_bytes(1000), Some(real(1000)));
        assert!(c.can_hold(400) && !c.can_hold(401) && !c.can_hold(1000));
        // A real snapshot later keeps the line exact.
        assert!(c.insert(key(300, 0), vec![vec![0; real(300)]]));
        assert_eq!(c.estimated_bytes(401), Some(real(401)));
        // No KV state in the blob: its total is recorded as the one-token point, as before.
        let mut c = PrefixCache::new(budget);
        let recurrent_only = vec![state_blob(&[("layer0.ssm_state", &[1, 4, 8, 8], 4)])];
        assert_eq!(c.observe_state(&recurrent_only), None);
        assert_eq!(c.estimated_bytes(1), Some(recurrent_only[0].len()));
    }

    #[test]
    fn config_state_floor_is_a_lower_bound_from_the_model_dims() {
        // Kev-9B (Qwen3.5-9B text config): 24 linear-attention layers of 32 x 128 x 128 recurrent
        // state, 8 full-attention layers of 4 KV heads x 256, at half a byte per element (u4, the
        // smallest OpenVINO state precision).
        let mut types = Vec::new();
        for i in 0..32 {
            types.push(if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            });
        }
        let dims = serde_json::json!({
            "linear_num_value_heads": 32, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "num_key_value_heads": 4, "head_dim": 256,
            "num_hidden_layers": 32,
        });
        let mut listed = dims.clone();
        listed["layer_types"] = serde_json::json!(types);
        let floor = Some((24 * 32 * 128 * 128 / 2, 8 * 2 * 4 * 256 / 2));
        assert_eq!(config_state_floor(&listed), floor);
        // Nested under text_config, as the multimodal config ships it.
        assert_eq!(
            config_state_floor(&serde_json::json!({ "text_config": listed })),
            floor
        );
        // No layer_types: full_attention_interval says which layers are full.
        let mut interval = dims.clone();
        interval["full_attention_interval"] = serde_json::json!(4);
        assert_eq!(config_state_floor(&interval), floor);
        // Missing dims: no bound.
        assert_eq!(
            config_state_floor(&serde_json::json!({"vocab_size": 10})),
            None
        );
        assert_eq!(
            config_state_floor(&dims),
            None,
            "which layers are linear is unknown"
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
        // First turn: the system-block end and the chat boundary are the same
        // position (two `im_start`s), so the two rules de-duplicate to one.
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
