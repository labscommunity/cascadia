//! Issue-34 Option C: `KvCoordination` for the OV stateful engines (`ov-runtime`, and — via the same
//! helper — qwen36 / dist-spec). Bridges the OpenVINO KV cache (held inside the C++ runtime as
//! `VariableState`) to the wire plane's NEGOTIATE/GET/INSERT (`cascadia_kv_wire`).
//!
//! **Why opaque.** Unlike the sparse-MoE engine (host-side `Vec<u16>` per layer, sliceable to any
//! prefix), the OV KV lives behind `query_state()`/`get_state()`/`set_state()` as engine-owned
//! tensors whose layout we don't model. So a snapshot is **one self-describing blob** (the shim's
//! `get_state_blob`), shipped under `OPAQUE_KV_LAYOUT` — the codec checks length + crc, not per-layer
//! shape, and the producer alone restores it via `set_state_blob` on the same model.
//!
//! **Coarser than sparse-MoE.** A blob is the WHOLE captured sequence's state, not sliceable. So
//! NEGOTIATE matches the longest *cached full sequence that is a prefix of the request* and serves
//! exactly that length — which is precisely the session-resume case the issue targets (the client
//! resends history, so next turn's prefix == this turn's full sequence).
//!
//! **Rig gate.** The blob bytes only exist with `--features openvino` (the C++ shim); off-rig
//! `get_state_blob`/`set_state_blob` return `Error::Stub` and capture/restore degrade to no-ops
//! (cold reprefill — today's behaviour). Warm==cold fidelity is certified on hardware.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use cascadia_engine::{KvCoordination, KvSnapshotHolder};
use cascadia_kv_wire::{
    KvSnapshotCodec, LayerMeta, Manifest, PartnerId, OPAQUE_KV_LAYOUT, SCHEMA_VERSION,
};

use crate::runtime::OvRuntimeEngine;

/// KV codec/engine revision — bump on any change to the blob *envelope* (the shim's
/// `get_state_blob` framing). Producer (export) and consumer (`consumer_engine_rev`) both read this.
pub const KV_ENGINE_REV: u64 = 1;

/// Captured blobs are large (full KV state); keep only a few most-recent turns.
const KV_MAX_ENTRIES: usize = 8;
/// Cap on stashed unconsumed offers (NEGOTIATE without a paired GET).
const KV_MAX_OFFERS: usize = 32;
/// Byte ceiling on `offers`, PER CACHE. Sized for a handful of concurrent NEGOTIATE→GET round trips
/// at rig blob scale (~35 MB, from the slice-2 measurements); an OV engine holds two of these caches
/// (`kv` and its `kv_share` holder mirror), so the node ceiling is twice this. NOT derived from
/// `KV_MAX_ENTRIES` — an offer is cloned from an entry but outlives it (`take_warm` removes the entry
/// it serves; the LRU drops the rest), so up to `KV_MAX_OFFERS` distinct offers can be held with no
/// surviving source entry. Shrunk under `cfg(test)` so the eviction tests cost megabytes, not gigabytes.
const KV_MAX_OFFER_BYTES: usize = if cfg!(test) { 4 << 20 } else { 256 << 20 };

pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Content-derived epoch over the negotiated prefix tokens. Sound for the serve-time `rank_honors`
/// check (same content ⇒ same epoch ⇒ honor; foreign content ⇒ different epoch ⇒ reject).
pub(crate) fn synth_epoch(prefix: &[i32]) -> u64 {
    let mut buf = Vec::with_capacity(prefix.len() * 4);
    for &t in prefix {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    fnv1a64(&buf)
}

/// FNV-1a digest of raw bytes — identity only, for cross-rank / cross-mode comparison in the logs.
///
/// Deliberately over RAW BYTES, never over lengths or manifest fields: an earlier probe compared the
/// manifest's token COUNT on both sides, found 98 == 98, and read that as confirmation when the two
/// numbers were trivially equal by construction. Hash what actually gets applied.
pub(crate) fn byte_digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Same digest over a token slice, so a rank's token identity can be compared without dumping it.
pub(crate) fn tokens_digest(tokens: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tokens {
        for &b in &t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Per-tensor dump of a state blob: `(name, rank, shape[2], nbytes, digest)` for every state.
///
/// For the qwen36 bar-#1 divergence. qwen36 is byte-identical single-box but diverges on the SHARDED
/// cross-chain move, and four blind fixes (key-name mapping, attention depth off-by-one, T=1 prefill,
/// reset-vs-recreate) have already been spent on it. Whole-blob digests only say "differs"; this says
/// WHICH tensor differs, which discriminates the live hypotheses in one run:
///   - mismatch confined to `conv`/`ssm` names ⇒ fixed-size recurrent state is being sliced as if it
///     were sequence-addressable at the shard boundary (fits single-box-clean / sharded-diverges)
///   - mismatch in attention KV only ⇒ rank/layer layout mapping on the sharded path
///   - no mismatch at all ⇒ the divergence is post-restore numerics, not the transfer
///
/// Off by default — `CASCADIA_KV_TENSOR_DUMP=1`. One line per tensor is far too loud for the certified
/// path, and the certified path must stay byte-identical to what ships.
pub(crate) fn log_blob_tensors(tag: &str, epoch: u64, blob: &[u8]) {
    if std::env::var("CASCADIA_KV_TENSOR_DUMP").ok().as_deref() != Some("1") {
        return;
    }
    fn u32_at(b: &[u8], p: usize) -> Option<u32> {
        Some(u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?))
    }
    fn u64_at(b: &[u8], p: usize) -> Option<u64> {
        Some(u64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?))
    }
    let mut p = 0usize;
    let Some(count) = u32_at(blob, p) else { return };
    p += 4;
    for _ in 0..count {
        let Some(name_len) = u32_at(blob, p).map(|v| v as usize) else {
            return;
        };
        let Some(name_at) = p.checked_add(4) else { return };
        let name = blob
            .get(name_at..name_at.saturating_add(name_len))
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        let Some(np) = name_at.checked_add(name_len) else {
            return;
        };
        p = np;
        let Some(&_dtype) = blob.get(p) else { return };
        let Some(rank) = blob.get(p + 1).map(|r| *r as usize) else {
            return;
        };
        p += 2;
        let mut seq_dim = 0usize;
        for i in 0..rank {
            let Some(d) = u64_at(blob, p).map(|v| v as usize) else {
                return;
            };
            p += 8;
            if i == 2 {
                seq_dim = d;
            }
        }
        let Some(nb) = u64_at(blob, p).map(|v| v as usize) else {
            return;
        };
        p += 8;
        let data = blob.get(p..p.saturating_add(nb)).unwrap_or(&[]);
        tracing::info!(target: "cascadia::kv", event = "kv_tensor_dump",
            tag, epoch, name = %name, rank, seq = seq_dim, nbytes = nb,
            digest = byte_digest(data));
        let Some(np) = p.checked_add(nb) else { return };
        p = np;
    }
}

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

/// Compact a `get_state_blob` blob along the seq dim (index 2): keep position `i` only where
/// `valid[i] != 0` (positions past `valid.len()` are kept), packed in order, rewriting each rank≥3
/// state's `shape[2]` + data to the kept count. Rank<3 **and recurrent (conv/ssm)** states copy
/// verbatim.
///
/// The conv/ssm exclusion is the same one `kv_seq_from_blob` makes, for the same reason: on a hybrid
/// model index 2 is a fixed constant (conv window, ssm state width), not a fold position, so gathering
/// it would silently reorder the recurrent state. Today every caller is dist-spec, which is pure
/// attention — so this is safety by construction rather than by caller discipline. It matters because
/// nothing downstream would catch the mistake: `set_state_blob` rebuilds the tensor from the blob's
/// OWN rewritten shape and only checks `byte_size == nb`, which stays self-consistent after a bad
/// gather, so a corrupted state restores "successfully".
///
/// Spec-decode leaves the KV padded with proposed-then-rejected positions, masked out host-side via
/// `valid_mask` — which the blob does NOT carry. Compacting at capture makes the blob self-describing
/// (restorable with no external mask), like every other engine's. Identity (verbatim clone) when
/// nothing is rejected. `None` if the blob is unparseable or a state's byte layout is inconsistent
/// (caller then skips capture → cold).
pub(crate) fn kv_compact_blob(blob: &[u8], valid: &[i64]) -> Option<Vec<u8>> {
    fn u32_at(b: &[u8], p: usize) -> Option<u32> {
        Some(u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?))
    }
    fn u64_at(b: &[u8], p: usize) -> Option<u64> {
        Some(u64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?))
    }
    let mut p = 0usize;
    let count = u32_at(blob, p)?;
    p += 4;
    let mut out = Vec::with_capacity(blob.len());
    out.extend_from_slice(&count.to_le_bytes());
    for _ in 0..count {
        let state_start = p;
        let name_len = u32_at(blob, p)? as usize;
        let name = blob.get(p.checked_add(4)?..p.checked_add(4)?.checked_add(name_len)?)?;
        p = p.checked_add(4)?.checked_add(name_len)?;
        let dtype = *blob.get(p)?;
        let rank = *blob.get(p.checked_add(1)?)? as usize;
        p = p.checked_add(2)?;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(u64_at(blob, p)? as usize);
            p = p.checked_add(8)?;
        }
        let nb = u64_at(blob, p)? as usize;
        p = p.checked_add(8)?;
        let data_start = p;
        let data_end = data_start.checked_add(nb)?;
        let data = blob.get(data_start..data_end)?;
        // rank<3 (no seq dim), recurrent (conv/ssm — index 2 is not a fold position), or fully-valid
        // ⇒ copy state verbatim. Same name test as kv_seq_from_blob; keep the two in step.
        let is_recurrent = {
            let n = String::from_utf8_lossy(name);
            n.contains("conv") || n.contains("ssm")
        };
        let seq = if rank >= 3 && !is_recurrent {
            shape[2]
        } else {
            0
        };
        let kept: Vec<usize> = (0..seq)
            .filter(|&i| i >= valid.len() || valid[i] != 0)
            .collect();
        if rank < 3 || is_recurrent || kept.len() == seq {
            out.extend_from_slice(blob.get(state_start..data_end)?);
            p = data_end;
            continue;
        }
        // outer = prod(shape[0..2]); cell = bytes per (outer, seq) step (folds heads + trailing dims).
        let outer: usize = shape[..2].iter().product();
        if outer == 0 || seq == 0 || nb % (outer * seq) != 0 {
            return None;
        }
        let cell = nb / (outer * seq);
        let new_seq = kept.len();
        out.extend_from_slice(&(name_len as u32).to_le_bytes());
        out.extend_from_slice(name);
        out.push(dtype);
        out.push(rank as u8);
        for (i, &d) in shape.iter().enumerate() {
            let v = if i == 2 { new_seq as u64 } else { d as u64 };
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&((outer * new_seq * cell) as u64).to_le_bytes());
        for o in 0..outer {
            for &s in &kept {
                let off = (o * seq + s) * cell;
                out.extend_from_slice(data.get(off..off + cell)?);
            }
        }
        p = data_end;
    }
    Some(out)
}

/// Max tokens in a CAPTURE frame body — DoS bound so a forged frame can't allocate unbounded.
pub(crate) const MAX_CAPTURE_TOKENS: usize = 1 << 20;

/// §8 CAPTURE frame BODY (transport-agnostic): `u64 epoch | u32 ntok | ntok × i32 (LE)`. The head
/// broadcasts this after a turn; each engine wraps it in its OWN frame header (qwen36 `frame_header`,
/// ov-runtime `WireTensor`, dist-spec `FrameKind`). Workers parse it, blob their slice, and ACK.
pub(crate) fn capture_body_bytes(epoch: u64, tokens: &[i32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(12 + tokens.len() * 4);
    b.extend_from_slice(&epoch.to_le_bytes());
    b.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for &t in tokens {
        b.extend_from_slice(&t.to_le_bytes());
    }
    b
}

/// Parse a CAPTURE body. `None` on truncation or an over-bound token count (forged/corrupt frame).
pub(crate) fn parse_capture_body(b: &[u8]) -> Option<(u64, Vec<i32>)> {
    if b.len() < 12 {
        return None;
    }
    let epoch = u64::from_le_bytes(b[0..8].try_into().ok()?);
    let ntok = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
    if ntok > MAX_CAPTURE_TOKENS || b.len() != 12 + ntok * 4 {
        return None;
    }
    let mut tokens = Vec::with_capacity(ntok);
    for c in b[12..].chunks_exact(4) {
        tokens.push(i32::from_le_bytes(c.try_into().ok()?));
    }
    Some((epoch, tokens))
}

/// [`capture_body_bytes`] plus the host `valid_mask` (dist-spec only): base body ++ `u32 mask_len |
/// mask_len × u8` (1=valid, 0=rejected draft). Workers carry no mask of their own, so the driver
/// ships it down the CAPTURE chain and each rank compacts its blob with it ([`kv_compact_blob`]).
pub(crate) fn capture_body_bytes_masked(epoch: u64, tokens: &[i32], valid: &[i64]) -> Vec<u8> {
    let mut b = capture_body_bytes(epoch, tokens);
    b.extend_from_slice(&(valid.len() as u32).to_le_bytes());
    b.reserve(valid.len());
    for &m in valid {
        b.push(u8::from(m != 0));
    }
    b
}

/// Inverse of [`capture_body_bytes_masked`]. `None` on truncation / over-bound counts.
pub(crate) fn parse_capture_body_masked(b: &[u8]) -> Option<(u64, Vec<i32>, Vec<i64>)> {
    if b.len() < 12 {
        return None;
    }
    let epoch = u64::from_le_bytes(b[0..8].try_into().ok()?);
    let ntok = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
    if ntok > MAX_CAPTURE_TOKENS {
        return None;
    }
    let tok_end = 12usize.checked_add(ntok.checked_mul(4)?)?;
    let mut tokens = Vec::with_capacity(ntok);
    for c in b.get(12..tok_end)?.chunks_exact(4) {
        tokens.push(i32::from_le_bytes(c.try_into().ok()?));
    }
    let mask_len = u32::from_le_bytes(b.get(tok_end..tok_end + 4)?.try_into().ok()?) as usize;
    if mask_len > MAX_CAPTURE_TOKENS {
        return None;
    }
    let mask_end = (tok_end + 4).checked_add(mask_len)?;
    if b.len() != mask_end {
        return None;
    }
    let valid = b[tok_end + 4..mask_end]
        .iter()
        .map(|&x| i64::from(x != 0))
        .collect();
    Some((epoch, tokens, valid))
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

/// One captured turn: the full token sequence and its opaque KV blob.
struct OvKvEntry {
    tokens: Vec<i32>,
    blob: Vec<u8>,
    /// Pulled over the KV plane (`insert_both`), not captured from this rank's own turn.
    plane_pulled: bool,
}

/// Per-engine KV blob cache + NEGOTIATE→GET offers. Lives in [`OvRuntimeEngine`] behind `kv_coord`.
#[derive(Default)]
pub(crate) struct OvKvCache {
    /// Captured full-sequence blobs, most-recent first, bounded (LRU). Head/single-stage path
    /// (token-keyed: this rank knows the tokens).
    entries: Vec<OvKvEntry>,
    /// `epoch → (tokens, blob)` stashed at NEGOTIATE for the paired GET (short-lived, single-use).
    offers: HashMap<u64, (Vec<i32>, Vec<u8>)>,
    /// `offers` epochs in insertion order, oldest first.
    offer_order: VecDeque<u64>,
    /// Live byte total of `offers`, against `KV_MAX_OFFER_BYTES`.
    offer_bytes: usize,
    /// Offers evicted before their paired GET, cumulative — the flood signal that survives suppression.
    offers_evicted: u64,
    /// A flood evicts once per NEGOTIATE, and one log line each is how a node's chain.log reached
    /// 770 GB (see `warn_limit`); bound the eviction log the same way.
    evict_log: crate::warn_limit::StepWarnLimiter,
    /// §8 multi-stage worker stash: `epoch → (tokens, blob)`. A worker rank has no tokens of its
    /// own, so the head's `CAPTURE(epoch, tokens)` frame carries them; the rank blobs its slice and
    /// stashes here. Served by `export` for repeat/later per-rank GETs (clone, not remove). Bounded.
    captures: HashMap<u64, (Vec<i32>, Vec<u8>)>,
    /// Issue-34 multi-stage cross-chain warm-resume: `(epoch, rank) → that rank's pulled blob`. The head
    /// pulls every rank's KV but can't use a downstream rank's slice locally, so it stashes it here and
    /// ships it inline in the `RESTORE(epoch)` frame to that rank (which `set_state`s it).
    ///
    /// Keyed by (epoch, rank), NOT epoch alone: `pull_on_miss` fans out `for rank in 0..num_ranks` and
    /// every rank shares one content epoch, so an epoch-only key made each rank's blob overwrite the
    /// previous one and the head shipped the LAST rank's KV to rank 1 — silently wrong state rather
    /// than a cold fallback. Invisible at 2 stages (exactly one downstream rank, nothing to overwrite),
    /// which is why the rig certs never caught it. Bounded.
    downstream: HashMap<(u64, u16), Vec<u8>>,
}

impl OvKvCache {
    /// Producer: stash a captured turn keyed by its full token sequence. Bounded LRU (oldest drop).
    pub(crate) fn capture(&mut self, tokens: Vec<i32>, blob: Vec<u8>) {
        self.insert_entry(tokens, blob, false);
    }

    fn insert_entry(&mut self, tokens: Vec<i32>, blob: Vec<u8>, plane_pulled: bool) {
        if tokens.is_empty() || blob.is_empty() {
            return;
        }
        self.entries.retain(|e| e.tokens != tokens); // de-dup exact key (refresh to front)
        self.entries.insert(
            0,
            OvKvEntry {
                tokens,
                blob,
                plane_pulled,
            },
        );
        self.entries.truncate(KV_MAX_ENTRIES);
    }

    /// §8 worker: stash this rank's blob under the head-broadcast `(epoch, tokens)`. Bounded.
    pub(crate) fn capture_under_epoch(&mut self, epoch: u64, tokens: Vec<i32>, blob: Vec<u8>) {
        if blob.is_empty() {
            return;
        }
        if self.captures.len() >= KV_MAX_ENTRIES && !self.captures.contains_key(&epoch) {
            if let Some(k) = self.captures.keys().next().copied() {
                self.captures.remove(&k);
            }
        }
        tracing::info!(target: "cascadia::kv", event = "kv_cap_under_epoch",
            epoch, tokens = tokens.len(), blob = blob.len());
        self.captures.insert(epoch, (tokens, blob));
    }

    /// Consumer INSERT (pulled, validated blob): stash for BOTH restore paths — token-keyed
    /// `entries` (the head warm-resumes via `take_warm` by prompt prefix) and epoch-keyed `captures`
    /// (a worker rank warm-resumes via the head's RESTORE(epoch), having no tokens of its own).
    pub(crate) fn insert_both(&mut self, tokens: Vec<i32>, blob: Vec<u8>) {
        if tokens.is_empty() || blob.is_empty() {
            return;
        }
        self.capture_under_epoch(synth_epoch(&tokens), tokens.clone(), blob.clone());
        self.insert_entry(tokens, blob, true); // pulled over the plane, not captured locally
    }

    /// Worker RESTORE: take the blob stashed under `epoch` (from INSERT/CAPTURE) so the rank can
    /// `set_state` it. Removed on take (one restore per inserted turn).
    pub(crate) fn take_capture(&mut self, epoch: u64) -> Option<(Vec<i32>, Vec<u8>)> {
        self.captures.remove(&epoch)
    }

    /// Head: stash a pulled DOWNSTREAM rank's blob for inline delivery in `RESTORE(epoch)`. Bounded.
    pub(crate) fn stash_downstream(&mut self, epoch: u64, rank: u16, blob: Vec<u8>) {
        if blob.is_empty() {
            return;
        }
        let key = (epoch, rank);
        if self.downstream.len() >= KV_MAX_ENTRIES && !self.downstream.contains_key(&key) {
            if let Some(k) = self.downstream.keys().next().copied() {
                self.downstream.remove(&k);
            }
        }
        tracing::info!(target: "cascadia::kv", event = "kv_stash_downstream",
            epoch, rank, blob = blob.len(), n = self.downstream.len() + 1);
        self.downstream.insert(key, blob);
    }

    /// Head: take a specific downstream RANK's blob to ship in its `RESTORE(epoch)`. Removed on take.
    /// `rank` must be the recipient of that frame — taking by epoch alone returned whichever rank was
    /// stashed last, which is the wrong tensor set for any chain deeper than 2 stages.
    pub(crate) fn take_downstream(&mut self, epoch: u64, rank: u16) -> Option<Vec<u8>> {
        self.downstream.remove(&(epoch, rank))
    }

    /// Count of stashed downstream blobs (diagnostic + single-slot fallback guard).
    pub(crate) fn downstream_len(&self) -> usize {
        self.downstream.len()
    }

    /// Head: epoch-agnostic fallback for the ONE stashed blob belonging to `rank`. Covers stash/restore
    /// epoch-key drift (the stash keys by the pulled rank's manifest tokens, restore by the head's warm
    /// prefix). Returns None unless exactly one blob is stashed AND it is that rank's, so a deeper chain
    /// can never recover the wrong rank's tensors through this path — it goes cold instead.
    pub(crate) fn take_downstream_single(&mut self, rank: u16) -> Option<Vec<u8>> {
        if self.downstream.len() != 1 {
            return None;
        }
        let k = *self.downstream.keys().next()?;
        if k.1 != rank {
            return None;
        }
        self.downstream.remove(&k)
    }

    /// Serve the snapshot asserted by `(epoch, len)` — `offers` first (head NEGOTIATE→GET, single
    /// use), then `captures` (worker stash, repeat-serve). `None` if absent or the length drifted.
    pub(crate) fn serve(&mut self, epoch: u64, len: u32) -> Option<(Vec<i32>, Vec<u8>)> {
        tracing::info!(target: "cascadia::kv", event = "kv_serve", epoch, want_len = len,
            in_offers = self.offers.contains_key(&epoch), in_captures = self.captures.contains_key(&epoch),
            n_offers = self.offers.len(), n_captures = self.captures.len(),
            cap_epochs = ?self.captures.keys().copied().collect::<Vec<_>>());
        let (tokens, blob) = if let Some(off) = self.take_offer(epoch) {
            off
        } else if let Some(cap) = self.captures.get(&epoch) {
            cap.clone()
        } else {
            return None;
        };
        // Probe A (serve side): identity of exactly what this holder is about to hand out, plus which
        // store it came from. NOTE the check below is LENGTH-only — it never compares tokens — so a
        // capture stored under a colliding synthesized epoch with the same length but different tokens
        // would serve silently. `tok_digest` is what makes that visible.
        tracing::info!(target: "cascadia::kv", event = "kv_serve_digest",
            epoch, len, blob_digest = byte_digest(&blob), tok_digest = tokens_digest(&tokens),
            n_tokens = tokens.len(), blob_len = blob.len());
        log_blob_tensors("serve", epoch, &blob);
        // Head/offers path carries tokens ⇒ length must match what was negotiated. Worker captures
        // also carry the head-broadcast tokens, so the same check holds for both.
        if tokens.len() as u32 != len {
            tracing::info!(target: "cascadia::kv", event = "kv_serve_len_mismatch",
                epoch, have = tokens.len(), want = len);
            return None;
        }
        Some((tokens, blob))
    }

    /// NEGOTIATE: longest cached full-sequence that is a prefix of `token_ids`; stash it as an offer
    /// under its content epoch for the paired GET. Returns `(epoch, prefix_len)`. Engine-agnostic.
    pub(crate) fn lookup(&mut self, token_ids: &[i32]) -> Option<(u64, u32)> {
        let (prefix, blob) = {
            let Some(e) = self.longest_prefix(token_ids) else {
                self.log_prefix_miss(token_ids);
                return None;
            };
            (e.tokens.clone(), e.blob.clone())
        };
        let len = prefix.len() as u32;
        let epoch = synth_epoch(&prefix);
        self.stash_offer(epoch, prefix, blob);
        Some((epoch, len))
    }

    /// Stash an offer for its paired GET, evicting until it fits both bounds.
    fn stash_offer(&mut self, epoch: u64, tokens: Vec<i32>, blob: Vec<u8>) {
        // A re-NEGOTIATE of the same prefix replaces its offer instead of double-counting its bytes.
        let _ = self.take_offer(epoch);
        // Loops on the order queue, not the map: if the two ever desync, evicting stops making
        // progress and this spins forever holding the engine lock. Stops at empty so a blob over the
        // whole budget is stashed alone, not evicted to death.
        while !self.offer_order.is_empty()
            && (self.offers.len() >= KV_MAX_OFFERS
                || self.offer_bytes + blob.len() > KV_MAX_OFFER_BYTES)
        {
            self.evict_oldest_offer();
        }
        self.offer_bytes += blob.len();
        self.offer_order.push_back(epoch);
        self.offers.insert(epoch, (tokens, blob));
        debug_assert_eq!(self.offer_order.len(), self.offers.len());
    }

    /// Oldest first: arbitrary `HashMap` order can drop the offer whose paired GET is in flight and
    /// keep a stale one, turning a warm resume cold for no gain.
    fn evict_oldest_offer(&mut self) {
        let Some(epoch) = self.offer_order.pop_front() else {
            return;
        };
        let Some((_, blob)) = self.offers.remove(&epoch) else {
            return;
        };
        self.offer_bytes = self.offer_bytes.saturating_sub(blob.len());
        self.offers_evicted += 1;
        // No `on_success` counterpart: an eviction streak has no natural close, and one line per
        // 30 s is the bound wanted. `suppressed` carries what that interval swallowed.
        let suppressed = match self.evict_log.on_failure(std::time::Instant::now()) {
            Some(crate::warn_limit::StepWarn::First) => 0,
            Some(crate::warn_limit::StepWarn::StillFailing { suppressed }) => suppressed,
            None => return,
        };
        tracing::info!(target: "cascadia::kv", event = "kv_offer_evicted_unserved",
            epoch, blob = blob.len(), suppressed, evicted_total = self.offers_evicted,
            n_offers = self.offers.len(), held = self.offer_bytes);
    }

    /// Remove an offer, keeping `offer_order` and `offer_bytes` in step.
    fn take_offer(&mut self, epoch: u64) -> Option<(Vec<i32>, Vec<u8>)> {
        let off = self.offers.remove(&epoch)?;
        self.offer_order.retain(|&e| e != epoch);
        self.offer_bytes = self.offer_bytes.saturating_sub(off.1.len());
        Some(off)
    }

    /// Longest cached entry whose `tokens` is a prefix of `req`. The blob is whole-sequence
    /// (opaque, not sliceable), so the served length == that entry's token count.
    fn longest_prefix(&self, req: &[i32]) -> Option<&OvKvEntry> {
        self.entries
            .iter()
            .filter(|e| !e.tokens.is_empty() && req.starts_with(&e.tokens))
            .max_by_key(|e| e.tokens.len())
    }

    /// NEGOTIATE-miss diagnostic. A bare `None` reads the same whether nothing was captured or a
    /// captured turn diverges from the request; telling those apart previously cost a rig run.
    fn log_prefix_miss(&self, req: &[i32]) {
        let Some((diverge_at, entry)) = self
            .entries
            .iter()
            .filter(|e| !e.tokens.is_empty())
            .map(|e| {
                (
                    e.tokens.iter().zip(req).take_while(|(a, b)| a == b).count(),
                    e,
                )
            })
            .max_by_key(|&(d, _)| d)
        else {
            tracing::info!(target: "cascadia::kv", event = "kv_negotiate_miss",
                reason = "no_entries", req_len = req.len());
            return;
        };
        tracing::info!(target: "cascadia::kv", event = "kv_negotiate_miss",
            reason = "prefix_diverged", req_len = req.len(), n_entries = self.entries.len(),
            entry_len = entry.tokens.len(), diverge_at,
            entry_tok = ?entry.tokens.get(diverge_at), req_tok = ?req.get(diverge_at));
    }

    /// Consumer: take a cached blob covering a **strict** prefix of `prompt`, for warm-resume at
    /// task start. Strict (`tokens.len() < prompt.len()`) guarantees ≥1 token left to prefill — the
    /// model needs a forward pass to produce the next token, and re-feeding tokens already in the
    /// restored state would double-count. Returns `(blob, prefix_len, plane_pulled)`, removed on
    /// take; `plane_pulled` scopes the chain verdict — see [`chain_verdict`].
    pub(crate) fn take_warm(&mut self, prompt: &[i32]) -> Option<(Vec<u8>, usize, bool)> {
        let idx = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                !e.tokens.is_empty()
                    && e.tokens.len() < prompt.len()
                    && prompt.starts_with(&e.tokens)
            })
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)?;
        let e = self.entries.remove(idx);
        Some((e.blob, e.tokens.len(), e.plane_pulled))
    }
}

/// Chain-restore verdict for a warm head. `down` — every downstream rank confirmed — is binding,
/// except on a turn the KV plane armed, where the ranks arm themselves out-of-band and `false` only
/// means "no CAPTURE under the donor chain's epoch".
///
/// `plane_pulled` scopes that override. The mode flag alone made every turn advisory, so a
/// same-chain turn whose tail had no CAPTURE under the donor epoch still warmed the head —
/// head-warm/tail-cold, a silent corrupt serve.
///
/// Only for engines whose plane ranks still arm out-of-band (qwen36; ov-runtime arms in-band from its
/// `OPCODE_RESTORE` handler, so it takes `down` unmasked). Delete this with issue-34 Task 2.3.
///
/// Residual (accepted): the mark rides a CONTENT-keyed entry, not a request. A later prompt starting
/// with an earlier plane-armed prefix consumes that entry and inherits the override. Closing it needs
/// a request id carried across the node/engine boundary, which cannot be verified off-rig.
pub(crate) fn chain_verdict(down: bool, plane_mode: bool, plane_pulled: bool) -> bool {
    down || (plane_mode && plane_pulled)
}

/// Opaque blob → wire `Manifest` + single-payload `(blob, [])`. K carries the blob; V is empty.
pub(crate) fn blob_to_wire(
    prefix: &[i32],
    blob: &[u8],
    partner: &str,
    model_fingerprint: u64,
    epoch: u64,
) -> (Manifest, Vec<(Vec<u8>, Vec<u8>)>) {
    let meta = LayerMeta {
        layer_index: 0,
        k_shape: vec![], // ignored under OPAQUE_KV_LAYOUT
        v_shape: vec![],
        k_byte_len: blob.len() as u64,
        v_byte_len: 0,
        k_crc32: crc32fast::hash(blob),
        v_crc32: crc32fast::hash(&[]),
    };
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        kv_layout_version: OPAQUE_KV_LAYOUT,
        engine_rev: KV_ENGINE_REV,
        partner: PartnerId(partner.to_string()),
        model_fingerprint,
        prefix_token_hash: synth_epoch(prefix),
        prefix_token_len: prefix.len() as u32,
        snapshot_epoch: epoch,
        num_layers: 1,
        layers: vec![meta],
        token_ids: prefix.to_vec(),
    };
    (manifest, vec![(blob.to_vec(), Vec::new())])
}

/// Wire `Manifest` + payloads → `(tokens, blob)`. `None` on a malformed pair (codec already
/// validated length + crc, so this only guards the opaque-shape contract: exactly one K payload).
pub(crate) fn wire_to_blob(
    manifest: &Manifest,
    payloads: &[(Vec<u8>, Vec<u8>)],
) -> Option<(Vec<i32>, Vec<u8>)> {
    if manifest.kv_layout_version != OPAQUE_KV_LAYOUT || payloads.len() != 1 {
        return None;
    }
    let blob = payloads[0].0.clone();
    if blob.is_empty() {
        return None;
    }
    Some((manifest.token_ids.clone(), blob))
}

impl OvRuntimeEngine {
    /// MODEL-level fingerprint (`model_id` + `total_layers`), NOT per-stage. A cross-chain pull
    /// asserts ONE fingerprint — the entry head's — for EVERY rank's GET, so all ranks of a model
    /// must share it; a per-stage layer span (`layer_start`/`layer_end`) would wrongly reject the
    /// worker ranks of a legitimate move (the tail's stage span differs from the head's). Per-rank
    /// stage selection is by the dial INDEX (rank N → that rank's holder + slice); a sharding
    /// mismatch degrades safely (the opaque blob's `set_state` size-rejects ⇒ cold), so the layer
    /// span is not needed as a guard here.
    pub(crate) fn kv_model_fingerprint(&self) -> u64 {
        let s = self.shard_spec();
        let mut buf = s.model_id.clone().into_bytes();
        buf.extend_from_slice(&s.total_layers.to_le_bytes());
        fnv1a64(&buf)
    }
}

impl KvCoordination for OvRuntimeEngine {
    fn model_fingerprint(&self) -> u64 {
        self.kv_model_fingerprint()
    }

    fn layout_version(&self) -> u16 {
        OPAQUE_KV_LAYOUT
    }

    fn engine_rev(&self) -> u64 {
        KV_ENGINE_REV
    }

    fn tokenize(&self, text: &str) -> Option<Vec<i32>> {
        // add_special_tokens=false mirrors the prefill encode (step_first encodes with `false`), so
        // NEGOTIATE tokens equal what keys the capture.
        let enc = self.tokenizer_ref()?.encode(text, false).ok()?;
        Some(enc.get_ids().iter().map(|&u| u as i32).collect())
    }

    fn lookup(&mut self, _partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        self.kv_cache_mut().lookup(token_ids)
    }

    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let model_fp = self.kv_model_fingerprint();
        // offers (head NEGOTIATE→GET) OR captures (§8 worker stash) — a worker rank has no
        // NEGOTIATE, so its slice is served from the head-broadcast epoch.
        let (prefix, blob) = self.kv_cache_mut().serve(expected_epoch, expected_len)?;
        Some(blob_to_wire(
            &prefix,
            &blob,
            partner,
            model_fp,
            expected_epoch,
        ))
    }

    fn insert(&mut self, manifest: &Manifest, payloads: &[(Vec<u8>, Vec<u8>)]) -> Result<(), ()> {
        let (tokens, blob) = wire_to_blob(manifest, payloads).ok_or(())?;
        // Stage the blob; the next prefill warm-resumes via `OvKvCache::take_warm` (rig-certified).
        self.kv_cache_mut().insert_both(tokens, blob);
        Ok(())
    }

    fn stash_downstream_rank(
        &mut self,
        rank: u16,
        manifest: &Manifest,
        payloads: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ()> {
        // Issue-34 multi-stage: a DOWNSTREAM rank's pulled blob can't be used by the head locally;
        // stash it under the content epoch so `send_restore_downstream` ships it inline to that rank.
        let (_tokens, blob) = wire_to_blob(manifest, payloads).ok_or(())?;
        let epoch = synth_epoch(&manifest.token_ids);
        self.kv_cache_mut().stash_downstream(epoch, rank, blob);
        Ok(())
    }

    fn apply_warm_resume(&mut self, epoch: u64) -> bool {
        // The pull staged this rank's slice under `epoch` (capture cache); set_state it now (plane path).
        let blob = match self.kv_cache_mut().take_capture(epoch) {
            Some((_tokens, blob)) => blob,
            None => return false,
        };
        self.apply_warm_resume_blob(&blob)
    }

    fn abort_warm_resume(&mut self, epoch: u64) {
        // Two cases, both handled: the slice is still STAGED (trigger ran, no commit) ⇒ drop it from the
        // capture cache so a later commit can't resurrect it; or it was already APPLIED (legacy/raced
        // commit) ⇒ scrub the engine back to cold. Safe for an epoch this rank never armed.
        let _ = self.kv_cache_mut().take_capture(epoch);
        self.abort_warm_resume_local();
    }
}

/// Shared handle to a captured-snapshot cache. The engine mirrors its captures here; the holder reads
/// it without ever taking the engine lock (so a busy node can still answer a pull).
pub(crate) type SharedKvCache = Arc<Mutex<OvKvCache>>;

/// A plane-pulled slice parked for the engine to apply itself.
pub struct KvHandoffSlot {
    pub epoch: u64,
    pub manifest: Manifest,
    pub payloads: Vec<(Vec<u8>, Vec<u8>)>,
}

/// Why [`handoff_decision`] refused a parked slice. Variants are 1:1 with the `kv_handoff_*` events
/// the drain logs; the cert greps two of them, so keep the mapping.
#[derive(Debug, PartialEq)]
pub(crate) enum HandoffReject {
    Validate,
    Decode,
    /// The slice's KV depth, which this rank is already past.
    TooLate(i64),
}

/// The pure part of `OvRuntimeEngine::drain_kv_handoff` — structural validation, opaque decode, depth
/// guard — split out because the engine needs a compiled IR, so this is the only part of the drain a
/// unit test can reach. `Ok` carries the blob to `set_state`.
///
/// Validation MUST happen on this path: the driver-loop apply it replaced ran the codec before the
/// consumer insert, which the hand-off skips entirely, so nothing else checks layout / engine_rev /
/// fingerprint and a slice from a drifted build or another model would be `set_state`d silently. The
/// holder's serve check is LENGTH-only (`tokens.len() == len`, never token equality), so it is not
/// that bind either.
pub(crate) fn handoff_decision(
    slot: &KvHandoffSlot,
    model_fp: u64,
    position: i64,
) -> Result<Vec<u8>, HandoffReject> {
    let refs: Vec<(&[u8], &[u8])> = slot
        .payloads
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    if KvSnapshotCodec::validate(
        &slot.manifest,
        &refs,
        OPAQUE_KV_LAYOUT,
        KV_ENGINE_REV,
        model_fp,
        &slot.manifest.token_ids,
    )
    .is_err()
    {
        return Err(HandoffReject::Validate);
    }
    let (_tokens, blob) =
        wire_to_blob(&slot.manifest, &slot.payloads).ok_or(HandoffReject::Decode)?;
    // Snapping back is what produced the two-item divergence: a slice shallower than where this rank
    // already is cannot be resumed into, so drop it and let the turn stay cold.
    let depth = kv_seq_from_blob(&blob).unwrap_or(0) as i64;
    if position > depth {
        return Err(HandoffReject::TooLate(depth));
    }
    Ok(blob)
}

/// One-slot mailbox handing a pulled warm-resume slice from the KV plane to the engine's recv loop.
///
/// Its mutex is INDEPENDENT of the engine mutex, and the producer side never touches the engine at
/// all. The confirm/commit path runs on the node's control task while the tail is typically parked
/// inside `step()` holding the engine mutex — reaching the engine from there deadlocks (proven
/// on-rig). So the producer only parks bytes here; the engine drains the mailbox from inside its own
/// recv loop, where it already owns its lock and can still apply ahead of the turn's forward.
pub struct KvHandoffMailbox {
    inner: Mutex<MailboxInner>,
}

#[derive(Default)]
struct MailboxInner {
    slot: Option<KvHandoffSlot>,
    /// Last epoch the engine TOOK, so a `clear` that finds the slot empty can tell "already gone to
    /// the engine" from "this rank never held it" — the two have opposite consequences for the head.
    /// Set by `take` before the drain validates / decodes / depth-guards / `set_state`s, so it does
    /// NOT mean applied: a taken-then-rejected slice records here too.
    drained: Option<u64>,
    too_late: u64,
    /// Drains that found the mailbox empty, and whether a slice has ever landed here at all. Together
    /// they separate the ordinary no-plane-slice drain from the engine running ahead of the commit —
    /// the race that was otherwise diagnosable only by inference across two runs.
    empty_drains: u64,
    ever_parked: bool,
}

impl KvHandoffMailbox {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(MailboxInner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MailboxInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Overwrites any unconsumed slice: only the newest pull can still be ahead of the engine's
    /// position, and an older one would only be rejected by the apply-site depth guard anyway.
    pub fn put(&self, epoch: u64, manifest: Manifest, payloads: Vec<(Vec<u8>, Vec<u8>)>) {
        let n_payloads = payloads.len();
        let mut g = self.lock();
        g.ever_parked = true;
        g.slot = Some(KvHandoffSlot {
            epoch,
            manifest,
            payloads,
        });
        drop(g);
        tracing::info!(target: "cascadia::kv", event = "kv_handoff_put", epoch, n_payloads);
    }

    pub fn take(&self) -> Option<KvHandoffSlot> {
        let mut g = self.lock();
        let slot = g.slot.take();
        match &slot {
            Some(s) => g.drained = Some(s.epoch),
            None => {
                g.empty_drains += 1;
                // DEBUG, not WARN: an empty drain is the normal case on any rank the plane never
                // feeds, so this scales with turns. `ever_parked` is the bit worth grepping.
                tracing::debug!(target: "cascadia::kv", event = "kv_handoff_drain_empty",
                    count = g.empty_drains, ever_parked = g.ever_parked);
            }
        }
        slot
    }

    /// Retract a parked slice (see [`cascadia_engine::KvWarmHandoff::clear`]). Epoch-matched so an
    /// abort for a stale epoch cannot drop a newer pull that overwrote it.
    ///
    /// A `false` for an epoch this mailbox already drained is the residual the retraction cannot
    /// close — the abort races the engine's recv-loop drain — so it is counted, not swallowed.
    pub fn clear(&self, epoch: u64) -> bool {
        let mut g = self.lock();
        let retracted = g.slot.as_ref().is_some_and(|s| s.epoch == epoch);
        if retracted {
            g.slot = None;
        }
        let too_late = !retracted && g.drained == Some(epoch);
        if too_late {
            g.too_late += 1;
            tracing::warn!(target: "cascadia::kv", event = "kv_handoff_abort_too_late",
                epoch, count = g.too_late);
        } else {
            tracing::info!(target: "cascadia::kv", event = "kv_handoff_cleared", epoch, retracted);
        }
        retracted
    }

    /// Epoch-blind retraction for the chain `ABORT`, whose frame is a bare opcode byte with no epoch
    /// to match on. `true` if a slice was still parked.
    ///
    /// Deliberately leaves `drained` alone: that field is what makes a later `clear(epoch)` count as a
    /// lost race, and a discarded slice never reached the engine at all.
    pub fn discard_any(&self) -> bool {
        self.lock().slot.take().is_some()
    }

    /// Aborts that arrived after the engine had already taken the slice. An UPPER BOUND on the
    /// warm-rank-under-cold-head residual, not a count of it: `drained` is stamped at `take`, so a
    /// slice the drain then rejected leaves that rank cold and is counted here anyway.
    pub fn aborts_too_late(&self) -> u64 {
        self.lock().too_late
    }

    /// Drains that found nothing parked. Read with [`Self::ever_parked`].
    pub fn empty_drains(&self) -> u64 {
        self.lock().empty_drains
    }

    /// Whether any slice has ever been parked here. An empty drain with this `true` is the engine
    /// running ahead of the plane's commit; with it `false` the rank simply has no plane traffic.
    pub fn ever_parked(&self) -> bool {
        self.lock().ever_parked
    }
}

impl Default for KvHandoffMailbox {
    fn default() -> Self {
        Self::new()
    }
}

impl cascadia_engine::KvWarmHandoff for KvHandoffMailbox {
    fn put(&self, epoch: u64, manifest: Manifest, payloads: Vec<(Vec<u8>, Vec<u8>)>) {
        KvHandoffMailbox::put(self, epoch, manifest, payloads);
    }

    fn clear(&self, epoch: u64) -> bool {
        KvHandoffMailbox::clear(self, epoch)
    }
}

/// Lock-free [`KvSnapshotHolder`] over a [`SharedKvCache`]. Serves NEGOTIATE/GET by locking ONLY the
/// snapshot cache — which inference touches only briefly at capture, not across the forward pass — so
/// the holder no longer starves on the engine lock the live generation holds. `model_fp` is the
/// engine's static stage fingerprint, snapshotted at handle creation.
pub(crate) struct OvKvHolder {
    pub(crate) cache: SharedKvCache,
    pub(crate) model_fp: u64,
}

impl KvSnapshotHolder for OvKvHolder {
    fn model_fingerprint(&self) -> u64 {
        self.model_fp
    }

    fn lookup(&self, _partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .lookup(token_ids)
    }

    fn export(
        &self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let (prefix, blob) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .serve(expected_epoch, expected_len)?;
        Some(blob_to_wire(
            &prefix,
            &blob,
            partner,
            self.model_fp,
            expected_epoch,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_kv_wire::KvSnapshotCodec;

    #[test]
    fn blob_wire_roundtrip_and_codec_accepts() {
        let prefix = vec![11, 22, 33];
        let blob = vec![1u8, 2, 3, 4, 5, 6, 7]; // odd len: structured codec would reject
        let (m, payloads) = blob_to_wire(&prefix, &blob, "acme", 7, 0xABCD);
        let refs: Vec<(&[u8], &[u8])> = payloads
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        assert!(
            KvSnapshotCodec::validate(&m, &refs, OPAQUE_KV_LAYOUT, KV_ENGINE_REV, 7, &prefix)
                .is_ok(),
            "exported opaque manifest must validate"
        );
        assert_eq!(m.num_layers, 1);
        assert_eq!(m.prefix_token_len, 3);
        let (back_tokens, back_blob) = wire_to_blob(&m, &payloads).unwrap();
        assert_eq!(back_tokens, prefix);
        assert_eq!(back_blob, blob);
    }

    #[test]
    fn cache_serves_longest_prefix() {
        let mut c = OvKvCache::default();
        c.capture(vec![1, 2], vec![0xA]);
        c.capture(vec![1, 2, 3, 4], vec![0xB]);
        // request [1,2,3,4,5] → longest cached prefix is [1,2,3,4]
        let e = c.longest_prefix(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(e.tokens, vec![1, 2, 3, 4]);
        assert_eq!(e.blob, vec![0xB]);
        // request [1,2,9] → only [1,2] qualifies
        assert_eq!(c.longest_prefix(&[1, 2, 9]).unwrap().tokens, vec![1, 2]);
        // request [9] → no prefix
        assert!(c.longest_prefix(&[9]).is_none());
    }

    #[test]
    fn take_warm_removes_and_returns_len() {
        let mut c = OvKvCache::default();
        c.capture(vec![1, 2, 3], vec![0xC, 0xD]);
        let (blob, len, _) = c.take_warm(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!((blob, len), (vec![0xC, 0xD], 3));
        assert!(c.take_warm(&[1, 2, 3, 4, 5]).is_none(), "consumed on take");
    }

    #[test]
    fn capture_body_roundtrips() {
        let tokens = vec![1i32, -2, 3, 1_000_000, i32::MIN, i32::MAX];
        let b = capture_body_bytes(0xDEAD_BEEF_0000_0001, &tokens);
        assert_eq!(
            parse_capture_body(&b),
            Some((0xDEAD_BEEF_0000_0001, tokens))
        );
        // empty token list is valid (epoch-only capture)
        let e = capture_body_bytes(7, &[]);
        assert_eq!(parse_capture_body(&e), Some((7, vec![])));
    }
    #[test]
    fn capture_body_rejects_malformed() {
        assert!(parse_capture_body(&[0u8; 4]).is_none()); // too short for header
        let mut b = capture_body_bytes(1, &[9, 9]);
        b.truncate(b.len() - 1); // body shorter than declared ntok
        assert!(parse_capture_body(&b).is_none());
        // forged over-bound ntok: header claims a huge count
        let mut h = 1u64.to_le_bytes().to_vec();
        h.extend_from_slice(&(u32::MAX).to_le_bytes());
        assert!(parse_capture_body(&h).is_none());
    }
    #[test]
    fn frame_blobs_roundtrips() {
        let blobs = vec![vec![1u8, 2, 3], vec![], vec![9u8; 50]];
        let framed = frame_blobs(&blobs);
        assert_eq!(unframe_blobs(&framed), Some(blobs));
        // single stage
        assert_eq!(
            unframe_blobs(&frame_blobs(&[vec![7u8]])),
            Some(vec![vec![7u8]])
        );
        // empty bundle (no stages)
        assert_eq!(unframe_blobs(&frame_blobs(&[])), Some(vec![]));
    }
    #[test]
    fn kv_seq_from_blob_reads_depth() {
        // Mirror the C++ get_state_blob layout: [u32 count] then per state
        // [u32 name_len][name][u8 dtype][u8 rank][u64*rank shape][u64 nbytes][data].
        fn state(name: &str, shape: &[u64], data_len: usize) -> Vec<u8> {
            let mut b = (name.len() as u32).to_le_bytes().to_vec();
            b.extend_from_slice(name.as_bytes());
            b.push(1); // dtype code
            b.push(shape.len() as u8);
            for &d in shape {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&(data_len as u64).to_le_bytes());
            b.extend(std::iter::repeat(0u8).take(data_len));
            b
        }
        let mut blob = 2u32.to_le_bytes().to_vec();
        blob.extend(state("past_key_values.0.key", &[1, 8, 85, 128], 16));
        blob.extend(state("past_key_values.0.value", &[1, 8, 85, 128], 16));
        assert_eq!(kv_seq_from_blob(&blob), Some(85)); // dim[2]
                                                       // Framed (qwen36 stages / dist-spec draft+target): max over parts, equal here.
        let framed = frame_blobs(&[blob.clone(), blob.clone()]);
        assert_eq!(kv_seq_from_framed_blob(&framed), Some(85));
        // Garbage/truncated ⇒ None so the caller falls back to the matched token count.
        assert_eq!(kv_seq_from_blob(&[0u8; 3]), None);
        assert_eq!(kv_seq_from_blob(&1u32.to_le_bytes()), None); // count=1, no state body
    }
    #[test]
    fn kv_compact_blob_gathers_valid_positions() {
        // One rank-4 state [1,1,4,2] with a distinct 2-byte block per seq position so the gather is
        // verifiable: pos p ⇒ bytes [10p, 10p+1]. cell = nb/(outer*seq) = 8/4 = 2 bytes.
        fn state(name: &str, shape: &[u64], data: &[u8]) -> Vec<u8> {
            let mut b = (name.len() as u32).to_le_bytes().to_vec();
            b.extend_from_slice(name.as_bytes());
            b.push(1); // dtype
            b.push(shape.len() as u8);
            for &d in shape {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&(data.len() as u64).to_le_bytes());
            b.extend_from_slice(data);
            b
        }
        let data = [0u8, 1, 10, 11, 20, 21, 30, 31]; // pos 0..3
        let mut blob = 1u32.to_le_bytes().to_vec();
        blob.extend(state("k", &[1, 1, 4, 2], &data));

        // Reject pos 1 ⇒ keep [0,2,3], seq 4→3, data gathered in order.
        let compact = kv_compact_blob(&blob, &[1, 0, 1, 1]).unwrap();
        assert_eq!(kv_seq_from_blob(&compact), Some(3));
        let mut want = 1u32.to_le_bytes().to_vec();
        want.extend(state("k", &[1, 1, 3, 2], &[0, 1, 20, 21, 30, 31]));
        assert_eq!(compact, want);

        // All-valid ⇒ identity (verbatim).
        assert_eq!(kv_compact_blob(&blob, &[1, 1, 1, 1]), Some(blob.clone()));
        // Mask shorter than seq ⇒ uncovered tail positions kept (here all kept ⇒ identity).
        assert_eq!(
            kv_compact_blob(&blob, &[1, 0]).map(|b| kv_seq_from_blob(&b)),
            Some(Some(3))
        );

        // rank<3 state (no seq dim) copies verbatim.
        let mut r2 = 1u32.to_le_bytes().to_vec();
        r2.extend(state("scalar", &[1, 4], &[7, 7, 7, 7]));
        assert_eq!(kv_compact_blob(&r2, &[0, 0, 0, 0]), Some(r2));

        // Recurrent states copy verbatim even at rank>=3 under a rejecting mask: for conv/ssm index 2
        // is a fixed width (conv window / ssm state), not a fold position, so gathering it would
        // silently reorder the recurrent state. The existing cases above all use generic names and so
        // would not catch this. Names mirror the surgery export (`cache_params.past.<kind>.<idx>`).
        for name in ["cache_params.past.conv.0", "cache_params.past.ssm.0"] {
            let mut rec = 1u32.to_le_bytes().to_vec();
            rec.extend(state(name, &[1, 1, 4, 2], &data));
            assert_eq!(
                kv_compact_blob(&rec, &[1, 0, 1, 1]),
                Some(rec.clone()),
                "{name} must be copied verbatim, not gathered"
            );
        }
    }
    #[test]
    fn capture_body_masked_roundtrips() {
        let body = capture_body_bytes_masked(0xABCD, &[5, -3, 7], &[1, 0, 1, 1, 0]);
        assert_eq!(
            parse_capture_body_masked(&body),
            Some((0xABCD, vec![5, -3, 7], vec![1, 0, 1, 1, 0]))
        );
        // empty mask is legal
        assert_eq!(
            parse_capture_body_masked(&capture_body_bytes_masked(1, &[9], &[])),
            Some((1, vec![9], vec![]))
        );
        // truncated mask ⇒ None
        let mut t = body.clone();
        t.pop();
        assert!(parse_capture_body_masked(&t).is_none());
        // unmasked body (no mask_len trailer) ⇒ None under the masked parser
        assert!(parse_capture_body_masked(&capture_body_bytes(1, &[9])).is_none());
    }
    #[test]
    fn unframe_blobs_rejects_malformed() {
        assert!(unframe_blobs(&[0u8; 2]).is_none()); // too short for count
        let mut f = frame_blobs(&[vec![1, 2, 3]]);
        f.push(0xFF); // trailing junk
        assert!(unframe_blobs(&f).is_none());
        // declared len overruns the buffer
        let mut bad = 1u32.to_le_bytes().to_vec();
        bad.extend_from_slice(&999u32.to_le_bytes());
        bad.extend_from_slice(&[1, 2]);
        assert!(unframe_blobs(&bad).is_none());
    }

    #[test]
    fn worker_stash_serves_by_epoch_repeatedly() {
        let mut c = OvKvCache::default();
        c.capture_under_epoch(0xE1, vec![11, 22, 33], vec![0xAA, 0xBB]);
        // served by (epoch, len); repeatable (workers answer multiple per-rank GETs)
        assert_eq!(c.serve(0xE1, 3), Some((vec![11, 22, 33], vec![0xAA, 0xBB])));
        assert_eq!(c.serve(0xE1, 3).map(|(_, b)| b), Some(vec![0xAA, 0xBB]));
        // length drift ⇒ refuse
        assert!(c.serve(0xE1, 2).is_none());
        // unknown epoch ⇒ None
        assert!(c.serve(0xE2, 3).is_none());
    }
    #[test]
    fn insert_both_feeds_head_and_worker_restore_paths() {
        let mut c = OvKvCache::default();
        c.insert_both(vec![1, 2, 3], vec![0xAB]);
        // head path: take_warm by prompt prefix (strict)
        assert_eq!(c.take_warm(&[1, 2, 3, 4]), Some((vec![0xAB], 3, true)));
        // worker path: take_capture by epoch
        let epoch = synth_epoch(&[1, 2, 3]);
        assert_eq!(c.take_capture(epoch), Some((vec![1, 2, 3], vec![0xAB])));
        assert!(c.take_capture(epoch).is_none(), "consumed on take");
    }

    #[test]
    fn offers_take_precedence_and_are_single_use() {
        let mut c = OvKvCache::default();
        c.stash_offer(0xF0, vec![1, 2], vec![0x01]);
        assert_eq!(c.serve(0xF0, 2), Some((vec![1, 2], vec![0x01])));
        assert!(c.serve(0xF0, 2).is_none(), "offer consumed on serve");
    }

    /// One NEGOTIATE→GET cycle: capture a turn, negotiate a superset of it, get back `(epoch, len)`.
    fn negotiate(c: &mut OvKvCache, tok: i32, blob: Vec<u8>) -> (u64, u32) {
        c.capture(vec![tok], blob);
        c.lookup(&[tok, 0])
            .expect("the just-captured turn prefixes the request")
    }

    /// §8 criterion 6: an unpaired-NEGOTIATE flood cannot pin more than the stated budget.
    #[test]
    fn unpaired_negotiate_flood_stays_within_the_byte_budget() {
        let mut c = OvKvCache::default();
        let rig_blob = KV_MAX_OFFER_BYTES / 8;
        for i in 0..20 {
            negotiate(&mut c, i, vec![0u8; rig_blob]);
            assert!(
                c.offer_bytes <= KV_MAX_OFFER_BYTES,
                "offer {i} pinned {} bytes",
                c.offer_bytes
            );
        }
        assert_eq!(c.offers.len(), 8, "exactly budget/blob offers survive");
        assert_eq!(c.offer_order.len(), c.offers.len());
    }

    /// Arbitrary `HashMap` order can evict the offer whose paired GET is already in flight and keep
    /// a stale one; the budget is only useful if the survivor is the one still expected. Driven
    /// through the COUNT cap so every eviction is checked, not just one 50/50 draw.
    #[test]
    fn offer_eviction_is_oldest_first() {
        let mut c = OvKvCache::default();
        let n = KV_MAX_OFFERS as i32 + 8;
        let e: Vec<u64> = (0..n)
            .map(|i| negotiate(&mut c, i, vec![i as u8]).0)
            .collect();
        let (evicted, kept) = e.split_at(8);
        assert!(
            evicted.iter().all(|x| !c.offers.contains_key(x)),
            "the 8 oldest offers go first"
        );
        assert!(kept.iter().all(|x| c.offers.contains_key(x)));
    }

    /// A re-NEGOTIATE of the same prefix replaces its offer: one map entry, one `offer_order` slot,
    /// and its bytes counted once. Without that, a routine peer retry leaks a stale order slot and
    /// `offer_bytes` drifts up until it pins the cache at one entry (release) or underflows (debug).
    #[test]
    fn re_negotiating_the_same_prefix_replaces_its_offer() {
        let mut c = OvKvCache::default();
        let (e1, _) = negotiate(&mut c, 1, vec![0u8; 64]);
        let (e2, len) = negotiate(&mut c, 1, vec![0u8; 64]);
        assert_eq!(e1, e2, "same prefix ⇒ same content epoch");
        assert_eq!(c.offers.len(), 1);
        assert_eq!(c.offer_order.len(), 1, "no stale order slot");
        assert_eq!(c.offer_bytes, 64, "bytes counted once");
        assert!(c.serve(e2, len).is_some());
        assert_eq!(c.offer_bytes, 0);
    }

    /// A blob bigger than the whole budget is admitted alone rather than evicted to death — the
    /// alternative leaves the node unable to serve its largest cached turn at all.
    #[test]
    fn an_offer_over_the_budget_is_still_servable() {
        let mut c = OvKvCache::default();
        let (epoch, len) = negotiate(&mut c, 7, vec![0u8; KV_MAX_OFFER_BYTES + 1]);
        assert_eq!(c.offers.len(), 1);
        assert_eq!(
            c.serve(epoch, len).map(|(_, b)| b.len()),
            Some(KV_MAX_OFFER_BYTES + 1)
        );
        assert_eq!(c.offer_bytes, 0);
    }

    #[test]
    fn a_paired_get_still_serves_after_evictions() {
        let mut c = OvKvCache::default();
        let n = KV_MAX_OFFERS as i32 + 8;
        let e: Vec<(u64, u32)> = (0..n)
            .map(|i| negotiate(&mut c, i, vec![i as u8]))
            .collect();
        assert_eq!(c.offers.len(), KV_MAX_OFFERS);
        let (epoch, len) = e[n as usize - 1];
        assert_eq!(
            c.serve(epoch, len),
            Some((vec![n - 1], vec![(n - 1) as u8]))
        );
    }

    /// Byte accounting must not drift: every offer leaves either through a GET or an eviction.
    #[test]
    fn offer_bytes_return_to_zero() {
        let mut c = OvKvCache::default();
        let e: Vec<(u64, u32)> = (0..(KV_MAX_OFFERS as i32 + 8))
            .map(|i| negotiate(&mut c, i, vec![i as u8]))
            .collect();
        for (epoch, len) in e {
            let _ = c.serve(epoch, len);
        }
        assert!(c.offers.is_empty() && c.offer_order.is_empty());
        assert_eq!(c.offer_bytes, 0);
    }

    fn park(mb: &KvHandoffMailbox, epoch: u64) {
        let (m, payloads) = blob_to_wire(&[1, 2, 3], &[0xAB], "acme", epoch, 0xABCD);
        mb.put(epoch, m, payloads);
    }

    /// The abort that follows a partial commit has to actually retract, or the rank drains the slice
    /// on its next turn and runs warm under a head that went cold.
    #[test]
    fn handoff_clear_retracts_a_parked_slice() {
        let mb = KvHandoffMailbox::new();
        park(&mb, 0xE7);
        assert!(mb.clear(0xE7), "a parked slice must report as retracted");
        assert!(
            mb.take().is_none(),
            "cleared slice must not reach the engine"
        );
        assert_eq!(mb.aborts_too_late(), 0);
    }

    #[test]
    fn handoff_clear_of_an_unknown_epoch_is_safe() {
        let mb = KvHandoffMailbox::new();
        assert!(!mb.clear(0xE8), "nothing parked ⇒ nothing retracted");
        park(&mb, 0xE9);
        assert!(!mb.clear(0xE8), "a stale abort must not drop a newer pull");
        assert_eq!(mb.take().map(|s| s.epoch), Some(0xE9));
        assert_eq!(mb.aborts_too_late(), 0, "neither case is the drain race");
    }

    /// The residual `clear` cannot close: the abort loses the race with the engine's recv-loop drain.
    /// Counted so the plan's acceptance criterion is measured rather than assumed rare.
    #[test]
    fn handoff_clear_after_a_drain_counts_the_residual() {
        let mb = KvHandoffMailbox::new();
        park(&mb, 0xEA);
        assert!(mb.take().is_some());
        assert!(!mb.clear(0xEA), "already drained ⇒ retraction impossible");
        assert_eq!(mb.aborts_too_late(), 1);
    }

    /// The ABORT frame is a bare opcode byte, so the retraction it needs cannot be epoch-matched.
    #[test]
    fn handoff_discard_any_retracts_without_an_epoch() {
        let mb = KvHandoffMailbox::new();
        park(&mb, 0xEB);
        assert!(mb.discard_any(), "a parked slice must report as discarded");
        assert!(
            mb.take().is_none(),
            "discarded slice must not reach the engine"
        );
        assert!(!mb.discard_any(), "nothing parked ⇒ nothing discarded");
    }

    /// A discard is not a drain. If it marked the epoch drained, the `clear` the enterprise side still
    /// sends for that epoch would be counted as too-late and inflate the residual with a slice the
    /// engine never applied.
    #[test]
    fn handoff_discard_any_is_not_a_drain() {
        let mb = KvHandoffMailbox::new();
        park(&mb, 0xEC);
        assert!(mb.discard_any());
        assert!(!mb.clear(0xEC));
        assert_eq!(mb.aborts_too_late(), 0);
    }

    /// The drain's only fully silent exit, and the one that hid the commit/drain race for two runs.
    /// `ever_parked` is the discriminator: an empty drain on a mailbox that has never held a slice is
    /// the ordinary no-plane-turn case, while one on a mailbox that has is the engine running ahead of
    /// the commit.
    #[test]
    fn an_empty_drain_is_counted_and_says_whether_a_slice_ever_landed() {
        let mb = KvHandoffMailbox::new();
        assert!(mb.take().is_none());
        assert_eq!(mb.empty_drains(), 1);
        assert!(!mb.ever_parked(), "nothing ever parked ⇒ not the race");

        park(&mb, 0xED);
        assert!(mb.take().is_some());
        assert_eq!(
            mb.empty_drains(),
            1,
            "a drain that found a slice is not empty"
        );

        assert!(mb.take().is_none());
        assert_eq!(mb.empty_drains(), 2);
        assert!(
            mb.ever_parked(),
            "a slice has landed here ⇒ the race signature"
        );
    }

    const DECISION_FP: u64 = 0xF00D;

    fn slot_of(blob: &[u8]) -> KvHandoffSlot {
        let (manifest, payloads) = blob_to_wire(&[1, 2, 3], blob, "acme", DECISION_FP, 0xE0);
        KvHandoffSlot {
            epoch: 0xE0,
            manifest,
            payloads,
        }
    }

    /// A parked slice whose blob reads back at KV depth `depth`: one rank-4 attention state, data
    /// elided (`kv_seq_from_blob` reads shape only).
    fn slot_at_depth(depth: u64) -> KvHandoffSlot {
        let name = "past_key_values.0.key";
        let mut blob = 1u32.to_le_bytes().to_vec();
        blob.extend_from_slice(&(name.len() as u32).to_le_bytes());
        blob.extend_from_slice(name.as_bytes());
        blob.extend_from_slice(&[1, 4]); // dtype, rank
        for d in [1u64, 1, depth, 1] {
            blob.extend_from_slice(&d.to_le_bytes());
        }
        blob.extend_from_slice(&0u64.to_le_bytes()); // nbytes
        slot_of(&blob)
    }

    #[test]
    fn handoff_decision_accepts_a_good_slice() {
        let s = slot_at_depth(4);
        let blob = handoff_decision(&s, DECISION_FP, 0).expect("valid slice ahead of position");
        assert_eq!(kv_seq_from_blob(&blob), Some(4));
    }

    #[test]
    fn handoff_decision_rejects_a_drifted_build_or_a_foreign_model() {
        let mut drifted = slot_at_depth(4);
        drifted.manifest.engine_rev += 1;
        assert_eq!(
            handoff_decision(&drifted, DECISION_FP, 0),
            Err(HandoffReject::Validate)
        );
        assert_eq!(
            handoff_decision(&slot_at_depth(4), DECISION_FP + 1, 0),
            Err(HandoffReject::Validate)
        );
    }

    /// Only the empty-blob arm of `wire_to_blob` is reachable: a payload count ≠ 1 is a `num_layers`
    /// mismatch, which validate rejects first.
    #[test]
    fn handoff_decision_rejects_an_undecodable_payload() {
        assert_eq!(
            handoff_decision(&slot_of(&[]), DECISION_FP, 0),
            Err(HandoffReject::Decode)
        );
    }

    /// The guard is `>`, not `>=`: a slice exactly at this rank's position is still resumable, and
    /// only a shallower one — which would snap the state backwards — is dropped.
    #[test]
    fn handoff_decision_guards_the_depth_boundary() {
        let s = slot_at_depth(4);
        assert!(
            handoff_decision(&s, DECISION_FP, 4).is_ok(),
            "position == depth must still resume"
        );
        assert_eq!(
            handoff_decision(&s, DECISION_FP, 5),
            Err(HandoffReject::TooLate(4))
        );
    }

    #[test]
    fn capture_bounded_and_dedups() {
        let mut c = OvKvCache::default();
        for i in 0..(KV_MAX_ENTRIES as i32 + 4) {
            c.capture(vec![i], vec![i as u8]);
        }
        assert_eq!(c.entries.len(), KV_MAX_ENTRIES);
        // de-dup: re-capturing an existing key doesn't grow / duplicate
        let n = c.entries.len();
        let key = c.entries[2].tokens.clone();
        c.capture(key.clone(), vec![0xFF]);
        assert_eq!(c.entries.len(), n);
        assert_eq!(c.entries[0].tokens, key, "re-capture moves to front");
    }

    #[test]
    fn take_warm_reports_entry_provenance() {
        let mut c = OvKvCache::default();
        c.capture(vec![1, 2, 3], vec![0xC]);
        assert_eq!(c.take_warm(&[1, 2, 3, 4]), Some((vec![0xC], 3, false)));
        c.insert_both(vec![1, 2, 3], vec![0xAB]);
        assert_eq!(c.take_warm(&[1, 2, 3, 4]), Some((vec![0xAB], 3, true)));
    }

    /// LRU pressure must fail safe: with the plane entry evicted, the local entry that still matches
    /// carries its own mark, so the chain verdict stays binding.
    #[test]
    fn evicting_a_plane_entry_leaves_a_local_mark() {
        let mut c = OvKvCache::default();
        c.insert_both(vec![0, 1], vec![0xAB]);
        c.capture(vec![0], vec![0x01]);
        for i in 1..=(KV_MAX_ENTRIES as i32 - 1) {
            c.capture(vec![9, i], vec![i as u8]); // key 9 never prefixes [0,1,2]; one over the bound
        }
        assert!(
            c.entries.iter().all(|e| !e.plane_pulled),
            "the plane entry fell off the tail"
        );
        assert_eq!(c.take_warm(&[0, 1, 2]), Some((vec![0x01], 1, false)));
    }

    #[test]
    fn chain_verdict_scopes_the_plane_override_to_plane_entries() {
        for (mode, entry) in [(false, false), (false, true), (true, false), (true, true)] {
            assert!(chain_verdict(true, mode, entry), "confirmed is always warm");
        }
        assert!(!chain_verdict(false, false, false));
        assert!(!chain_verdict(false, false, true));
        assert!(
            !chain_verdict(false, true, false),
            "P3: plane mode must not override a false verdict on a locally captured entry"
        );
        assert!(chain_verdict(false, true, true));
    }
}
