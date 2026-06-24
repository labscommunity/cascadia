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

use std::collections::HashMap;

use cascadia_engine::KvCoordination;
use cascadia_kv_wire::{LayerMeta, Manifest, PartnerId, OPAQUE_KV_LAYOUT, SCHEMA_VERSION};

use crate::runtime::OvRuntimeEngine;

/// KV codec/engine revision — bump on any change to the blob *envelope* (the shim's
/// `get_state_blob` framing). Producer (export) and consumer (`consumer_engine_rev`) both read this.
pub const KV_ENGINE_REV: u64 = 1;

/// Captured blobs are large (full KV state); keep only a few most-recent turns.
const KV_MAX_ENTRIES: usize = 8;
/// Cap on stashed unconsumed offers (NEGOTIATE without a paired GET).
const KV_MAX_OFFERS: usize = 32;

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

/// Restored KV depth (max `shape[2]` over rank>=3 states) read straight from a `get_state_blob`
/// blob. The blob is self-describing — `[u32 count]` then per state
/// `[u32 name_len][name][u8 dtype][u8 rank][u64×rank shape][u64 nbytes][data]` (LE).
///
/// Why this exists: the cached token list is `prompt + all generated`, but the KV only holds tokens
/// that were *fed* — the last sampled token of a turn is never fed back, so KV depth = matched_len-1.
/// Warm-resume must drive `position`/`attention_mask` from the real KV depth, else the suffix decode
/// feeds `mask_len = kv+2` against `kv+1` keys and the attention `Add` fails on shape. Engine-agnostic
/// (reads the actual depth instead of assuming the off-by-one). `None` if the blob is unparseable.
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
        p = p.checked_add(4)?.checked_add(name_len)?; // skip name_len + name
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
        if rank >= 3 {
            seq = seq.max(seq_dim);
        }
        let nb = u64_at(blob, p)? as usize;
        p = p.checked_add(8)?.checked_add(nb)?; // skip nbytes + data
    }
    (seq > 0).then_some(seq)
}

/// KV depth from a multi-stage FRAMED blob (`frame_blobs` of N per-stage `get_state_blob`s) — the max
/// `kv_seq_from_blob` over its parts. Stages share the sequence length, so any part gives the depth;
/// `max` is a safe tie-break. Use for engines whose warm blob is framed (qwen36 stages, dist-spec
/// draft+target); raw single-stage blobs use [`kv_seq_from_blob`] directly. `None` if unparseable.
pub(crate) fn kv_seq_from_framed_blob(blob: &[u8]) -> Option<usize> {
    let parts = unframe_blobs(blob)?;
    parts.iter().filter_map(|p| kv_seq_from_blob(p)).max()
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
}

/// Per-engine KV blob cache + NEGOTIATE→GET offers. Lives in [`OvRuntimeEngine`] behind `kv_coord`.
#[derive(Default)]
pub(crate) struct OvKvCache {
    /// Captured full-sequence blobs, most-recent first, bounded (LRU). Head/single-stage path
    /// (token-keyed: this rank knows the tokens).
    entries: Vec<OvKvEntry>,
    /// `epoch → (tokens, blob)` stashed at NEGOTIATE for the paired GET (short-lived, single-use).
    offers: HashMap<u64, (Vec<i32>, Vec<u8>)>,
    /// §8 multi-stage worker stash: `epoch → (tokens, blob)`. A worker rank has no tokens of its
    /// own, so the head's `CAPTURE(epoch, tokens)` frame carries them; the rank blobs its slice and
    /// stashes here. Served by `export` for repeat/later per-rank GETs (clone, not remove). Bounded.
    captures: HashMap<u64, (Vec<i32>, Vec<u8>)>,
}

impl OvKvCache {
    /// Producer: stash a captured turn keyed by its full token sequence. Bounded LRU (oldest drop).
    pub(crate) fn capture(&mut self, tokens: Vec<i32>, blob: Vec<u8>) {
        if tokens.is_empty() || blob.is_empty() {
            return;
        }
        self.entries.retain(|e| e.tokens != tokens); // de-dup exact key (refresh to front)
        self.entries.insert(0, OvKvEntry { tokens, blob });
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
        self.capture(tokens, blob);
    }

    /// Worker RESTORE: take the blob stashed under `epoch` (from INSERT/CAPTURE) so the rank can
    /// `set_state` it. Removed on take (one restore per inserted turn).
    pub(crate) fn take_capture(&mut self, epoch: u64) -> Option<(Vec<i32>, Vec<u8>)> {
        self.captures.remove(&epoch)
    }

    /// Serve the snapshot asserted by `(epoch, len)` — `offers` first (head NEGOTIATE→GET, single
    /// use), then `captures` (worker stash, repeat-serve). `None` if absent or the length drifted.
    pub(crate) fn serve(&mut self, epoch: u64, len: u32) -> Option<(Vec<i32>, Vec<u8>)> {
        let (tokens, blob) = if let Some(off) = self.offers.remove(&epoch) {
            off
        } else if let Some(cap) = self.captures.get(&epoch) {
            cap.clone()
        } else {
            return None;
        };
        // Head/offers path carries tokens ⇒ length must match what was negotiated. Worker captures
        // also carry the head-broadcast tokens, so the same check holds for both.
        if tokens.len() as u32 != len {
            return None;
        }
        Some((tokens, blob))
    }

    /// NEGOTIATE: longest cached full-sequence that is a prefix of `token_ids`; stash it as an offer
    /// under its content epoch for the paired GET. Returns `(epoch, prefix_len)`. Engine-agnostic.
    pub(crate) fn lookup(&mut self, token_ids: &[i32]) -> Option<(u64, u32)> {
        let (prefix, blob) = {
            let e = self.longest_prefix(token_ids)?;
            (e.tokens.clone(), e.blob.clone())
        };
        let len = prefix.len() as u32;
        let epoch = synth_epoch(&prefix);
        if self.offers.len() >= KV_MAX_OFFERS && !self.offers.contains_key(&epoch) {
            if let Some(k) = self.offers.keys().next().copied() {
                self.offers.remove(&k);
            }
        }
        self.offers.insert(epoch, (prefix, blob));
        Some((epoch, len))
    }

    /// Longest cached entry whose `tokens` is a prefix of `req`. The blob is whole-sequence
    /// (opaque, not sliceable), so the served length == that entry's token count.
    fn longest_prefix(&self, req: &[i32]) -> Option<&OvKvEntry> {
        self.entries
            .iter()
            .filter(|e| !e.tokens.is_empty() && req.starts_with(&e.tokens))
            .max_by_key(|e| e.tokens.len())
    }

    /// Consumer: take a cached blob covering a **strict** prefix of `prompt`, for warm-resume at
    /// task start. Strict (`tokens.len() < prompt.len()`) guarantees ≥1 token left to prefill — the
    /// model needs a forward pass to produce the next token, and re-feeding tokens already in the
    /// restored state would double-count. Returns `(blob, prefix_len)`; removed on take.
    pub(crate) fn take_warm(&mut self, prompt: &[i32]) -> Option<(Vec<u8>, usize)> {
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
        Some((e.blob, e.tokens.len()))
    }
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
    /// Stable model+stage fingerprint: the model id plus this stage's layer span, so a stage only
    /// matches the identical stage on a peer chain (a partial-KV blob is stage-specific).
    pub(crate) fn kv_model_fingerprint(&self) -> u64 {
        let s = self.shard_spec();
        let mut buf = s.model_id.clone().into_bytes();
        buf.extend_from_slice(&s.layer_start.to_le_bytes());
        buf.extend_from_slice(&s.layer_end.to_le_bytes());
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
        let (blob, len) = c.take_warm(&[1, 2, 3, 4, 5]).unwrap();
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
        assert_eq!(c.take_warm(&[1, 2, 3, 4]), Some((vec![0xAB], 3)));
        // worker path: take_capture by epoch
        let epoch = synth_epoch(&[1, 2, 3]);
        assert_eq!(c.take_capture(epoch), Some((vec![1, 2, 3], vec![0xAB])));
        assert!(c.take_capture(epoch).is_none(), "consumed on take");
    }

    #[test]
    fn offers_take_precedence_and_are_single_use() {
        let mut c = OvKvCache::default();
        c.offers.insert(0xF0, (vec![1, 2], vec![0x01]));
        assert_eq!(c.serve(0xF0, 2), Some((vec![1, 2], vec![0x01])));
        assert!(c.serve(0xF0, 2).is_none(), "offer consumed on serve");
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
}
