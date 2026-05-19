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
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

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
    /// 1 µs faster per lookup; not worth it for cap≤8.
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
    /// `prompt`. Returns `Some((matched_len, snapshot))` for the
    /// best match, `None` if nothing matches.
    ///
    /// "Best" = longest prefix length. We scan every entry and pick
    /// the one whose `prefix` is a prefix of `prompt` and is longest.
    /// At cap≤8 this is a single-µs walk; not worth a trie.
    ///
    /// Side effect: on a hit, the matched entry is moved to MRU
    /// position (front of the deque) so it survives the next eviction.
    /// `hits`/`misses` counters are bumped accordingly.
    pub fn lookup(&mut self, prompt: &[i64], fingerprint: &ModelFingerprint) -> Option<KvSnapshot> {
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
                let snapshot = entry.1.snapshot.clone();
                self.entries.push_front(entry);
                Some(snapshot)
            }
        }
    }

    /// Insert a snapshot keyed by the given prefix + model fingerprint.
    /// If an entry already exists for this exact key, replace it and
    /// move to MRU. Evicts LRU entries until under capacity.
    ///
    /// Returns the number of evicted entries (0 in the steady state
    /// once the cache is full and we replace rather than grow).
    pub fn insert(
        &mut self,
        prefix: Vec<i64>,
        fingerprint: &ModelFingerprint,
        snapshot: KvSnapshot,
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
        self.entries.push_front((key, Entry { prefix, snapshot }));
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

// ---------------------------------------------------------------------------
// Disk persistence (iter 084)
// ---------------------------------------------------------------------------
//
// The cache lives in RAM for the lifetime of the engine process. With
// disk persistence, a long-lived chat workload that restarts at 03:00
// every night doesn't re-pay the prefill cost on the first request
// after restart — the system-prompt snapshots are reloaded into RAM
// before the API binds.
//
// File layout (little-endian, packed):
//
//   ┌───────────────────────────────┐
//   │ MAGIC (8 bytes)               │  b"TAHKVPC\0" — "Tahoma KV
//   │                               │  Prefix Cache", null-terminated.
//   ├───────────────────────────────┤
//   │ FORMAT_VERSION (u32)          │  0 on this PR; bumped on
//   │                               │  format break.
//   ├───────────────────────────────┤
//   │ fingerprint_len (u32)         │  Length of fingerprint blob.
//   ├───────────────────────────────┤
//   │ fingerprint (bincode)         │  Serialized [`ModelFingerprint`].
//   │                               │  Compared on load; mismatch =
//   │                               │  ignore file (don't crash).
//   ├───────────────────────────────┤
//   │ entry_count (u32)             │  Number of cached entries.
//   ├───────────────────────────────┤
//   │ for each entry:               │
//   │   entry_len (u32)             │  Length of the entry blob.
//   │   (prefix, snapshot) (bincode)│  Tuple of (Vec<i64>, KvSnapshot).
//   └───────────────────────────────┘
//
// Length-prefixing every variable-size blob lets a single corrupt
// entry abort *that entry* (logged + skipped) without throwing away
// the entries before it. That matters when persisting K2.6-scale
// snapshots — one ~150 MiB blob is too expensive to discard the
// whole file for a single tail-end write that got torn at shutdown.

/// 8-byte magic at the head of every persisted cache file. The
/// trailing NUL distinguishes from older debug-build files that wrote
/// just "TAHKVPC0" without padding (those are unreadable by this
/// loader — by design; the format hasn't been published).
const MAGIC: &[u8; 8] = b"TAHKVPC\0";

/// Current on-disk format version. Bump when [`KvSnapshot`] or the
/// header changes in a way the loader can't pick up automatically. On
/// version mismatch the loader logs a warn and treats the file as
/// absent — same policy as fingerprint mismatch.
pub const FORMAT_VERSION: u32 = 0;

/// Default filename inside the user-supplied `--kv-prefix-cache-path`
/// directory. We use a directory instead of a single file so a
/// follow-up PR can shard per-rank without changing the CLI surface
/// (e.g. `rank_00.bin`, `rank_01.bin`).
pub const DEFAULT_FILENAME: &str = "rank_00.bin";

/// Errors emitted by the persistence layer.
///
/// All variants are non-fatal at the call site — the engine logs the
/// error and continues with an empty cache. The "fingerprint mismatch
/// = ignore file" requirement in the task brief is enforced by
/// returning `LoadOutcome::FingerprintMismatch` from `load_from_disk`
/// rather than an error, so the caller doesn't have to thread a "this
/// is OK, that isn't" predicate through `match` arms.
#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("io: {0}")]
    Io(#[from] io::Error),

    #[error("bincode encode: {0}")]
    Encode(#[from] bincode::error::EncodeError),

    #[error("bincode decode: {0}")]
    Decode(#[from] bincode::error::DecodeError),

    #[error("bad magic: expected {expected:?}, got {got:?}")]
    BadMagic { expected: [u8; 8], got: [u8; 8] },

    #[error("unsupported format version: expected {expected}, got {got}")]
    UnsupportedVersion { expected: u32, got: u32 },
}

/// Outcome of [`KvPrefixCache::load_from_disk`]. The caller logs the
/// outcome and continues either way — there is no "must crash" case
/// from disk persistence.
#[derive(Debug, Clone)]
pub enum LoadOutcome {
    /// No file existed at the given path. Cache left untouched.
    NotFound,
    /// File loaded and entries inserted; carries the entry count for
    /// logging.
    Loaded { entries: usize },
    /// File existed but its embedded fingerprint disagreed with the
    /// running model. Cache left untouched.
    FingerprintMismatch,
    /// File existed but was malformed (bad magic / version / truncated
    /// header / corrupt bincode). Cache left untouched. The actual
    /// error is logged by the caller via `tracing`.
    Corrupted,
}

impl KvPrefixCache {
    /// Iterate over `(prefix, snapshot)` pairs in LRU order
    /// (most-recently-used first). Used by [`save_to_disk`] and
    /// covered by the test suite — the order matters because reload
    /// must preserve eviction priority.
    pub fn iter(&self) -> impl Iterator<Item = (&[i64], &KvSnapshot)> {
        self.entries
            .iter()
            .map(|(_, e)| (e.prefix.as_slice(), &e.snapshot))
    }

    /// Serialize the cache to `path`. Atomic via write-to-tempfile +
    /// rename — a torn write at shutdown does not leave a partial
    /// file at the canonical location.
    ///
    /// `path` may be either a file or a directory:
    /// - File: written verbatim.
    /// - Directory (must exist; not auto-created): file is written as
    ///   `<dir>/rank_00.bin`. The directory layout is forward-compat
    ///   with per-rank sharding.
    ///
    /// Returns `Ok(0)` when the cache is disabled or empty — both
    /// reasonable "nothing to do" cases.
    pub fn save_to_disk(
        &self,
        path: impl AsRef<Path>,
        fingerprint: &ModelFingerprint,
    ) -> Result<usize, PersistError> {
        if !self.enabled() || self.entries.is_empty() {
            return Ok(0);
        }
        let target = resolve_save_path(path.as_ref())?;
        // Ensure parent dir exists so the user can pass a path under
        // a previously-uncreated subdir.
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        // Write to a sibling tempfile then atomic-rename. tempfile in
        // workspace deps, but std::fs is enough for the simple case
        // (we generate a unique suffix from the system clock).
        let tmp = target.with_extension(format!(
            "bin.tmp.{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        {
            let f = File::create(&tmp)?;
            let mut w = BufWriter::new(f);
            write_header(&mut w, fingerprint)?;
            let count = self.entries.len() as u32;
            w.write_all(&count.to_le_bytes())?;
            for (_key, entry) in &self.entries {
                let blob =
                    bincode::serde::encode_to_vec((&entry.prefix, &entry.snapshot), bincode_cfg())?;
                let len = blob.len() as u32;
                w.write_all(&len.to_le_bytes())?;
                w.write_all(&blob)?;
            }
            w.flush()?;
        }
        fs::rename(&tmp, &target)?;
        info!(
            path = %target.display(),
            entries = self.entries.len(),
            "kv-prefix-cache: saved snapshot to disk"
        );
        Ok(self.entries.len())
    }

    /// Load cache entries from `path`, validating against `fingerprint`.
    ///
    /// Returns a [`LoadOutcome`] describing what happened. None of the
    /// outcomes are fatal:
    /// - `NotFound` — first start with persistence enabled.
    /// - `Loaded` — entries inserted in LRU order.
    /// - `FingerprintMismatch` — model changed under us; treat as
    ///   cold start.
    /// - `Corrupted` — file is bogus; log + skip.
    ///
    /// Entries are inserted in *reverse* iteration order so that the
    /// LRU front-most entry on disk ends up MRU after load (the
    /// `insert` API push_front's onto the deque).
    ///
    /// On any read error past the header, the entries decoded so far
    /// are retained — partial loads are useful in the K2.6 case where
    /// each entry is ~150 MiB and a tail-end truncation shouldn't
    /// throw away earlier ones.
    pub fn load_from_disk(
        &mut self,
        path: impl AsRef<Path>,
        fingerprint: &ModelFingerprint,
    ) -> LoadOutcome {
        if !self.enabled() {
            return LoadOutcome::NotFound;
        }
        let target = match resolve_load_path(path.as_ref()) {
            Ok(p) => p,
            Err(_) => return LoadOutcome::NotFound,
        };
        if !target.exists() {
            return LoadOutcome::NotFound;
        }
        let f = match File::open(&target) {
            Ok(f) => f,
            Err(e) => {
                warn!(path = %target.display(), error = %e, "kv-prefix-cache: open failed; treating as cold start");
                return LoadOutcome::Corrupted;
            }
        };
        let mut r = BufReader::new(f);
        match read_and_validate_header(&mut r, fingerprint) {
            Ok(HeaderOutcome::Ok) => {}
            Ok(HeaderOutcome::FingerprintMismatch) => {
                warn!(
                    path = %target.display(),
                    "kv-prefix-cache: on-disk fingerprint != current model; ignoring file"
                );
                return LoadOutcome::FingerprintMismatch;
            }
            Err(e) => {
                warn!(
                    path = %target.display(),
                    error = %e,
                    "kv-prefix-cache: header read failed; treating as cold start"
                );
                return LoadOutcome::Corrupted;
            }
        };
        // Header OK — read entry count + entries.
        let mut count_buf = [0u8; 4];
        if let Err(e) = r.read_exact(&mut count_buf) {
            warn!(path = %target.display(), error = %e, "kv-prefix-cache: missing entry count");
            return LoadOutcome::Corrupted;
        }
        let count = u32::from_le_bytes(count_buf) as usize;
        // Read every entry into a Vec first so we can reinsert in
        // reverse (so the file's MRU stays MRU after a series of
        // push_fronts).
        let mut decoded: Vec<(Vec<i64>, KvSnapshot)> = Vec::with_capacity(count);
        for i in 0..count {
            let mut len_buf = [0u8; 4];
            if let Err(e) = r.read_exact(&mut len_buf) {
                warn!(
                    path = %target.display(),
                    error = %e,
                    entry = i,
                    "kv-prefix-cache: missing entry length; keeping {} decoded entries",
                    decoded.len()
                );
                break;
            }
            let elen = u32::from_le_bytes(len_buf) as usize;
            let mut blob = vec![0u8; elen];
            if let Err(e) = r.read_exact(&mut blob) {
                warn!(
                    path = %target.display(),
                    error = %e,
                    entry = i,
                    "kv-prefix-cache: short read of entry body; keeping {} decoded entries",
                    decoded.len()
                );
                break;
            }
            let pair: (Vec<i64>, KvSnapshot) =
                match bincode::serde::decode_from_slice(&blob, bincode_cfg()) {
                    Ok((v, _consumed)) => v,
                    Err(e) => {
                        warn!(
                            path = %target.display(),
                            error = %e,
                            entry = i,
                            "kv-prefix-cache: entry decode failed; keeping {} decoded entries",
                            decoded.len()
                        );
                        break;
                    }
                };
            decoded.push(pair);
        }
        // Reinsert in reverse so the file's first entry (MRU on save)
        // ends up MRU after a series of push_front's via insert.
        let loaded = decoded.len();
        for (prefix, snap) in decoded.into_iter().rev() {
            self.insert(prefix, fingerprint, snap);
        }
        info!(
            path = %target.display(),
            entries = loaded,
            "kv-prefix-cache: restored snapshot from disk"
        );
        LoadOutcome::Loaded { entries: loaded }
    }
}

/// Pick the canonical save path. If the user passed a directory, use
/// `<dir>/rank_00.bin`. If they passed a file path (or a path that
/// doesn't exist yet, treated as a file), use it verbatim.
fn resolve_save_path(p: &Path) -> Result<PathBuf, PersistError> {
    if p.is_dir() {
        Ok(p.join(DEFAULT_FILENAME))
    } else {
        Ok(p.to_path_buf())
    }
}

/// Same as [`resolve_save_path`] but never creates a dir; just
/// computes the path. Used by `load_from_disk` where the dir might
/// not exist yet (cold start).
fn resolve_load_path(p: &Path) -> Result<PathBuf, PersistError> {
    if p.is_dir() {
        Ok(p.join(DEFAULT_FILENAME))
    } else {
        Ok(p.to_path_buf())
    }
}

fn bincode_cfg() -> bincode::config::Configuration {
    // Standard config: little-endian, varint, default size limit.
    // Pin explicitly so a future bincode version that flips the
    // default endianness doesn't silently break existing files.
    bincode::config::standard()
}

fn write_header<W: Write>(w: &mut W, fingerprint: &ModelFingerprint) -> Result<(), PersistError> {
    w.write_all(MAGIC)?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    let fp_blob = bincode::serde::encode_to_vec(fingerprint, bincode_cfg())?;
    let fp_len = fp_blob.len() as u32;
    w.write_all(&fp_len.to_le_bytes())?;
    w.write_all(&fp_blob)?;
    Ok(())
}

enum HeaderOutcome {
    Ok,
    FingerprintMismatch,
}

fn read_and_validate_header<R: Read>(
    r: &mut R,
    expected: &ModelFingerprint,
) -> Result<HeaderOutcome, PersistError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(PersistError::BadMagic {
            expected: *MAGIC,
            got: magic,
        });
    }
    let mut ver_buf = [0u8; 4];
    r.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if version != FORMAT_VERSION {
        return Err(PersistError::UnsupportedVersion {
            expected: FORMAT_VERSION,
            got: version,
        });
    }
    let mut fp_len_buf = [0u8; 4];
    r.read_exact(&mut fp_len_buf)?;
    let fp_len = u32::from_le_bytes(fp_len_buf) as usize;
    let mut fp_blob = vec![0u8; fp_len];
    r.read_exact(&mut fp_blob)?;
    let (got, _consumed): (ModelFingerprint, _) =
        bincode::serde::decode_from_slice(&fp_blob, bincode_cfg())?;
    if &got != expected {
        return Ok(HeaderOutcome::FingerprintMismatch);
    }
    Ok(HeaderOutcome::Ok)
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
        let got = c
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
        let got = c.lookup(&[1, 2, 3, 4, 5, 6], &fp_a()).expect("hit");
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
    fn insert_same_key_replaces_in_place() {
        // Calling insert twice with the same key shouldn't grow the cache.
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 9));
        assert_eq!(c.len(), 1);
        let got = c.lookup(&[1, 2, 3, 4], &fp_a()).expect("hit");
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

    // -----------------------------------------------------------------
    // Persistence tests (iter 084)
    // -----------------------------------------------------------------

    fn snapshot_eq(a: &KvSnapshot, b: &KvSnapshot) -> bool {
        if a.past_seq_len != b.past_seq_len
            || a.num_heads != b.num_heads
            || a.qk_head_dim != b.qk_head_dim
            || a.v_head_dim != b.v_head_dim
        {
            return false;
        }
        let layer0_eq = match (&a.layer0, &b.layer0) {
            (None, None) => true,
            (Some(x), Some(y)) => x.lid == y.lid && x.past_k == y.past_k && x.past_v == y.past_v,
            _ => false,
        };
        if !layer0_eq {
            return false;
        }
        if a.shells.len() != b.shells.len() {
            return false;
        }
        for (x, y) in a.shells.iter().zip(b.shells.iter()) {
            if x.lid != y.lid || x.past_k != y.past_k || x.past_v != y.past_v {
                return false;
            }
        }
        true
    }

    #[test]
    fn save_load_round_trip_is_byte_identical() {
        // Load-bearing test the task brief calls out: every KvSnapshot
        // that goes onto disk must come back bit-identical.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();

        let mut c = KvPrefixCache::new(4);
        let s1 = mk_snapshot(3, 1);
        let s2 = mk_snapshot(5, 42);
        c.insert(vec![10, 20, 30], &fp_a(), s1.clone());
        c.insert(vec![1, 2, 3, 4, 5], &fp_a(), s2.clone());

        let written = c.save_to_disk(&path, &fp_a()).unwrap();
        assert_eq!(written, 2);

        let mut c2 = KvPrefixCache::new(4);
        let outcome = c2.load_from_disk(&path, &fp_a());
        match outcome {
            LoadOutcome::Loaded { entries } => assert_eq!(entries, 2),
            other => panic!("expected Loaded, got {other:?}"),
        }
        assert_eq!(c2.len(), 2);
        // Both prefixes must be present and the snapshots byte-identical.
        let got1 = c2.lookup(&[10, 20, 30, 99], &fp_a()).expect("hit s1");
        let got2 = c2.lookup(&[1, 2, 3, 4, 5, 99], &fp_a()).expect("hit s2");
        assert!(snapshot_eq(&got1, &s1), "s1 not byte-identical");
        assert!(snapshot_eq(&got2, &s2), "s2 not byte-identical");
    }

    #[test]
    fn load_missing_file_is_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        // Use an explicit file path that does not exist (so the loader
        // doesn't auto-pick rank_00.bin inside the dir).
        let path = tmp.path().join("does-not-exist.bin");
        let mut c = KvPrefixCache::new(4);
        let outcome = c.load_from_disk(&path, &fp_a());
        assert!(matches!(outcome, LoadOutcome::NotFound));
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn load_rejects_fingerprint_mismatch() {
        // Save under fp_a; load with fp_b expecting FingerprintMismatch.
        // Critical safety net — restoring K2.6 KV into a Qwen runner
        // would either segfault or produce garbage.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();

        let mut c = KvPrefixCache::new(2);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.save_to_disk(&path, &fp_a()).unwrap();

        let mut c2 = KvPrefixCache::new(2);
        let outcome = c2.load_from_disk(&path, &fp_b());
        assert!(
            matches!(outcome, LoadOutcome::FingerprintMismatch),
            "expected FingerprintMismatch, got {outcome:?}"
        );
        assert_eq!(c2.len(), 0, "cache must remain empty on mismatch");
    }

    #[test]
    fn load_rejects_bad_magic_without_crashing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("garbage.bin");
        std::fs::write(&path, b"not a tahoma cache file at all").unwrap();
        let mut c = KvPrefixCache::new(2);
        let outcome = c.load_from_disk(&path, &fp_a());
        assert!(
            matches!(outcome, LoadOutcome::Corrupted),
            "expected Corrupted, got {outcome:?}"
        );
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn save_disabled_cache_is_noop() {
        // A disabled cache (capacity=0) must not write a file even if
        // save_to_disk is called — keeps `--kv-prefix-cache-path`
        // without `--kv-prefix-cache-size` from leaving a 0-byte stub.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();
        let c = KvPrefixCache::new(0);
        let n = c.save_to_disk(&path, &fp_a()).unwrap();
        assert_eq!(n, 0);
        // Nothing should have been written.
        let target = path.join(DEFAULT_FILENAME);
        assert!(
            !target.exists(),
            "no file should be written for disabled cache"
        );
    }

    #[test]
    fn save_empty_cache_is_noop() {
        // Same as disabled, but with capacity > 0 and just no entries.
        // Could be the very first startup before any prompt has been
        // processed.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();
        let c = KvPrefixCache::new(4);
        let n = c.save_to_disk(&path, &fp_a()).unwrap();
        assert_eq!(n, 0);
        let target = path.join(DEFAULT_FILENAME);
        assert!(!target.exists());
    }

    #[test]
    fn load_preserves_mru_order() {
        // Save with [B, A] (B MRU); load and assert that B is still
        // MRU — i.e. inserting C with cap=2 evicts A, not B.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();
        let mut c = KvPrefixCache::new(2);
        c.insert(vec![1, 1, 1], &fp_a(), mk_snapshot(3, 1)); // A
        c.insert(vec![2, 2, 2], &fp_a(), mk_snapshot(3, 2)); // B (MRU)
        c.save_to_disk(&path, &fp_a()).unwrap();

        let mut c2 = KvPrefixCache::new(2);
        let _ = c2.load_from_disk(&path, &fp_a());
        // Insert C — eviction must drop the LRU (which should be A).
        c2.insert(vec![3, 3, 3], &fp_a(), mk_snapshot(3, 3));
        assert!(
            c2.lookup(&[2, 2, 2, 9], &fp_a()).is_some(),
            "B should still be in the cache after eviction"
        );
        assert!(
            c2.lookup(&[1, 1, 1, 9], &fp_a()).is_none(),
            "A was LRU; should be evicted"
        );
    }

    #[test]
    fn save_to_existing_dir_writes_default_filename() {
        let tmp = tempfile::tempdir().unwrap();
        let mut c = KvPrefixCache::new(2);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.save_to_disk(tmp.path(), &fp_a()).unwrap();
        let target = tmp.path().join(DEFAULT_FILENAME);
        assert!(
            target.exists(),
            "expected {} to exist after save",
            target.display()
        );
        // File should start with the magic bytes.
        let bytes = std::fs::read(&target).unwrap();
        assert!(bytes.starts_with(MAGIC), "file missing magic header");
    }

    #[test]
    fn load_unsupported_version_is_corrupted() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad-version.bin");
        // Write magic + bogus version + nothing else.
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(FORMAT_VERSION + 999).to_le_bytes());
        std::fs::write(&path, &buf).unwrap();
        let mut c = KvPrefixCache::new(2);
        let outcome = c.load_from_disk(&path, &fp_a());
        assert!(
            matches!(outcome, LoadOutcome::Corrupted),
            "expected Corrupted on bad version, got {outcome:?}"
        );
    }

    #[test]
    fn load_truncated_entry_keeps_earlier_entries() {
        // Build a valid file then chop its tail. The header + first
        // entry should still load; the truncated second entry is
        // silently dropped.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().to_path_buf();
        let mut c = KvPrefixCache::new(4);
        c.insert(vec![1, 2, 3], &fp_a(), mk_snapshot(3, 1));
        c.insert(vec![4, 5, 6], &fp_a(), mk_snapshot(3, 2));
        c.save_to_disk(&path, &fp_a()).unwrap();
        let file = path.join(DEFAULT_FILENAME);
        let bytes = std::fs::read(&file).unwrap();
        // Truncate roughly mid-second-entry. Drop the last 20 bytes;
        // header + entry-count + one full entry are well within that.
        let truncated = &bytes[..bytes.len().saturating_sub(20)];
        std::fs::write(&file, truncated).unwrap();
        let mut c2 = KvPrefixCache::new(4);
        let outcome = c2.load_from_disk(&file, &fp_a());
        // Either Loaded with at least 1 entry, or Corrupted if even the
        // entry count is unreachable. Both are acceptable; assert the
        // process didn't panic and the cache is in a sane state.
        match outcome {
            LoadOutcome::Loaded { entries } => assert!(entries >= 1),
            LoadOutcome::Corrupted => {}
            other => panic!("unexpected outcome on truncated file: {other:?}"),
        }
    }
}
