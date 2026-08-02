//! Issue-34 Option C: `KvCoordination` for the sparse-MoE engine — the host-side bridge between the
//! engine's `KvPrefixCache` (content-keyed, LRU) and the wire plane's epoch-asserted NEGOTIATE/GET/
//! INSERT (`cascadia_kv_wire`). All buffer ops; no device FFI.
//!
//! Two impedance points the runtime cache doesn't model, handled here:
//! - **Epoch.** The cache has none; we synthesise a content-derived `snapshot_epoch` (fnv1a over the
//!   matched prefix). Sound for the serve-time `rank_honors` check (same content ⇒ same epoch ⇒
//!   honor; foreign content ⇒ different epoch ⇒ reject), without a global nonce.
//! - **GET has no tokens.** A `lookup` stashes `(prefix_tokens, snapshot)` under the epoch so the
//!   paired `export` can rebuild the `Manifest.token_ids` the consumer re-compares.
//!
//! **Known gap (DiD, deferred):** the `KvPrefixCache` is content-keyed, not partner-keyed — `partner`
//! is stamped on the exported `Manifest` but does not namespace the lookup. The content-key + the
//! no-tenant-tokens invariant are the load-bearing controls (§13); partner-keying is defense-in-depth.
//! **Sharded serve:** under `kv_coord` a `total>1` engine gets the configured cache size and the
//! head seeds it after a chain-wide `CAPTURE` (engine.rs), so a sharded holder does serve. WITHOUT
//! the feature the cache is capacity-0 and lookup/insert degrade to None/no-op — the "sharded gap"
//! an earlier note recorded unconditionally, which is what made the plane path look unreachable.

use cascadia_engine::kv_handoff::{KvHandoffMailbox, KvHandoffSlot};
use cascadia_engine::KvCoordination;
use cascadia_kv_wire::{LayerMeta, Manifest, PartnerId, KV_LAYOUT_VERSION, SCHEMA_VERSION};

use crate::engine::SparseMoEEngine;
use crate::kv_prefix_cache::{KvPrefixCache, KvSnapshot, LayerKvSlice, ModelFingerprint};

/// KV codec/engine revision — bump on any change to the snapshot buffer layout. Producer (export) and
/// consumer (`consumer_engine_rev`) both read this, so the codec's engine-rev guard is self-consistent.
pub const KV_ENGINE_REV: u64 = 1;

/// Cap on stashed unconsumed offers (NEGOTIATE without a paired GET); oldest dropped on overflow.
pub(crate) const KV_MAX_OFFERS: usize = 64;
/// Byte ceiling on ONE `offers` map; an engine holds two (`kv_offers` + the `kv_share` mirror), and
/// `lookup` has already cloned the snapshot before `stash` evicts — so node peak is ~2×(this + one
/// snapshot), not 2×this. At K2.6 scale (~150 MiB/512-token snapshot) that is 1–2 live offers, where
/// the same number buys the sibling OpenVINO engine ~7 at its ~35 MB blobs; a third concurrent
/// NEGOTIATE evicts the oldest. NOT derived from the prefix-cache capacity: an offer is cloned from
/// an entry but outlives it. Shrunk under `cfg(test)` so the eviction tests cost MB, not GB.
pub(crate) const KV_MAX_OFFER_BYTES: usize = if cfg!(test) { 4 << 20 } else { 256 << 20 };
/// Floor between eviction log lines.
const EVICT_LOG_EVERY: std::time::Duration = std::time::Duration::from_secs(30);

/// Bounded NEGOTIATE→GET offer stash: `epoch → (prefix tokens, snapshot)`, held only until the paired
/// GET. Generic over the snapshot type, so the K2.6 and OvMoe planes share it without a sizing trait.
pub(crate) struct KvOfferStash<S> {
    offers: std::collections::HashMap<u64, (Vec<i32>, S, usize)>,
    /// Stashed epochs in insertion order, oldest first.
    order: std::collections::VecDeque<u64>,
    bytes: usize,
    max_offers: usize,
    max_bytes: usize,
    /// Sizes each offer here rather than trusting a caller-supplied count: two of the four call sites
    /// need a loaded engine to construct, so a wrong count there would disable the byte cap silently.
    sizer: fn(&S) -> usize,
    /// Offers dropped before their paired GET, cumulative. Monotone — the delta between two log lines
    /// is what the 30 s floor swallowed.
    evicted: u64,
    /// A flood evicts once per NEGOTIATE, so a line per eviction would trade memory exhaustion for
    /// log exhaustion on the same untrusted input.
    last_log: Option<std::time::Instant>,
}

impl<S> KvOfferStash<S> {
    pub(crate) fn new(max_offers: usize, max_bytes: usize, sizer: fn(&S) -> usize) -> Self {
        Self {
            offers: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            bytes: 0,
            max_offers,
            max_bytes,
            sizer,
            evicted: 0,
            last_log: None,
        }
    }

    /// Stash an offer for its paired GET, evicting until it fits both bounds.
    pub(crate) fn stash(&mut self, epoch: u64, tokens: Vec<i32>, snapshot: S) {
        let bytes = (self.sizer)(&snapshot);
        // A re-NEGOTIATE of the same prefix replaces its offer instead of double-counting its bytes.
        let _ = self.take(epoch);
        // Loops on `order`, not the map: a desync must not spin here while the caller holds a lock.
        // Stops at empty so an offer over the whole budget is stashed alone, not evicted to death.
        while !self.order.is_empty()
            && (self.offers.len() >= self.max_offers || self.bytes + bytes > self.max_bytes)
        {
            self.evict_oldest();
        }
        self.bytes += bytes;
        self.order.push_back(epoch);
        self.offers.insert(epoch, (tokens, snapshot, bytes));
        debug_assert_eq!(self.order.len(), self.offers.len());
    }

    /// Oldest first: arbitrary `HashMap` order can drop the offer whose paired GET is in flight and
    /// keep a stale one, turning a warm resume cold for no gain.
    fn evict_oldest(&mut self) {
        let Some(epoch) = self.order.pop_front() else {
            return;
        };
        let Some((_, _, bytes)) = self.offers.remove(&epoch) else {
            return;
        };
        self.bytes = self.bytes.saturating_sub(bytes);
        self.evicted += 1;
        let now = std::time::Instant::now();
        if self
            .last_log
            .is_none_or(|t| now.duration_since(t) >= EVICT_LOG_EVERY)
        {
            self.last_log = Some(now);
            tracing::info!(target: "cascadia::kv", event = "kv_offer_evicted_unserved",
                epoch, bytes, evicted_total = self.evicted, n_offers = self.offers.len(),
                held = self.bytes);
        }
    }

    /// Remove an offer, keeping `order` and `bytes` in step.
    pub(crate) fn take(&mut self, epoch: u64) -> Option<(Vec<i32>, S)> {
        let (tokens, snapshot, bytes) = self.offers.remove(&epoch)?;
        self.order.retain(|&e| e != epoch);
        self.bytes = self.bytes.saturating_sub(bytes);
        Some((tokens, snapshot))
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.offers.len()
    }

    #[cfg(test)]
    pub(crate) fn order_len(&self) -> usize {
        self.order.len()
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    #[cfg(test)]
    pub(crate) fn evicted(&self) -> u64 {
        self.evicted
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, epoch: u64) -> bool {
        self.offers.contains_key(&epoch)
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Content-derived epoch over the negotiated prefix tokens. `pub(crate)` so the multi-stage capture
/// path (engine.rs `step_first`) mints the SAME epoch the head broadcasts to its workers (§8: the
/// head assigns E; ranks adopt it via the `CAPTURE` frame, never deriving it locally).
pub(crate) fn synth_epoch(prefix: &[i32]) -> u64 {
    let mut buf = Vec::with_capacity(prefix.len() * 4);
    for &t in prefix {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    fnv1a64(&buf)
}

fn u16s_to_le_bytes(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_bytes_to_u16s(b: &[u8]) -> Option<Vec<u16>> {
    if b.len() % 2 != 0 {
        return None;
    }
    Some(
        b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

/// One engine `LayerKvSlice` → a wire `(LayerMeta, (k_bytes, v_bytes))`. Shapes are
/// `[num_heads, prefix_len, head_dim]` (SEQ_AXIS = 1) so the codec's seq-axis check matches.
fn slice_to_wire(
    s: &LayerKvSlice,
    num_heads: u32,
    prefix_len: u32,
    qk_head_dim: u32,
    v_head_dim: u32,
) -> (LayerMeta, (Vec<u8>, Vec<u8>)) {
    let k_bytes = u16s_to_le_bytes(&s.past_k);
    let v_bytes = u16s_to_le_bytes(&s.past_v);
    let meta = LayerMeta {
        layer_index: s.lid,
        k_shape: vec![num_heads, prefix_len, qk_head_dim],
        v_shape: vec![num_heads, prefix_len, v_head_dim],
        k_byte_len: k_bytes.len() as u64,
        v_byte_len: v_bytes.len() as u64,
        k_crc32: crc32fast::hash(&k_bytes),
        v_crc32: crc32fast::hash(&v_bytes),
    };
    (meta, (k_bytes, v_bytes))
}

/// `(prefix_tokens, snapshot)` → wire `Manifest` + per-layer payloads, layer0 first then shells.
fn snapshot_to_wire(
    prefix: &[i32],
    snap: &KvSnapshot,
    partner: &str,
    model_fingerprint: u64,
    epoch: u64,
) -> (Manifest, Vec<(Vec<u8>, Vec<u8>)>) {
    let prefix_len = snap.past_seq_len as u32;
    let slices = snap.layer0.iter().chain(snap.shells.iter());
    let mut layers = Vec::new();
    let mut payloads = Vec::new();
    for s in slices {
        let (meta, kv) = slice_to_wire(
            s,
            snap.num_heads,
            prefix_len,
            snap.qk_head_dim,
            snap.v_head_dim,
        );
        layers.push(meta);
        payloads.push(kv);
    }
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        kv_layout_version: KV_LAYOUT_VERSION,
        engine_rev: KV_ENGINE_REV,
        partner: PartnerId(partner.to_string()),
        model_fingerprint,
        prefix_token_hash: synth_epoch(prefix),
        prefix_token_len: prefix_len,
        snapshot_epoch: epoch,
        num_layers: layers.len() as u32,
        layers,
        token_ids: prefix.to_vec(),
    };
    (manifest, payloads)
}

/// Wire `Manifest` + payloads → engine `KvSnapshot`. `None` on a structural mismatch (the codec has
/// already validated, so this only fails on a truly malformed pair). lid 0 ⇒ layer0; lid ≥ 1 ⇒ shell.
fn wire_to_snapshot(manifest: &Manifest, payloads: &[(Vec<u8>, Vec<u8>)]) -> Option<KvSnapshot> {
    if manifest.layers.len() != payloads.len() || manifest.layers.is_empty() {
        return None;
    }
    let first = &manifest.layers[0];
    // Shapes are [num_heads, prefix_len, head_dim].
    let num_heads = *first.k_shape.first()?;
    let qk_head_dim = *first.k_shape.get(2)?;
    let v_head_dim = *first.v_shape.get(2)?;
    let mut layer0 = None;
    let mut shells = Vec::new();
    for (meta, (k_bytes, v_bytes)) in manifest.layers.iter().zip(payloads.iter()) {
        let slice = LayerKvSlice {
            lid: meta.layer_index,
            past_k: le_bytes_to_u16s(k_bytes)?,
            past_v: le_bytes_to_u16s(v_bytes)?,
        };
        if slice.lid == 0 {
            layer0 = Some(slice);
        } else {
            shells.push(slice);
        }
    }
    Some(KvSnapshot {
        past_seq_len: manifest.prefix_token_len as usize,
        num_heads,
        qk_head_dim,
        v_head_dim,
        layer0,
        shells,
    })
}

/// FNV-1a over every payload byte in wire order, so a donor's serve digest and a consumer's
/// applied digest are comparable across ranks.
///
/// Over RAW BYTES, never over lengths or manifest fields: an earlier probe on the OpenVINO side
/// compared token COUNTS, found 98 == 98, and read that as confirmation when the two numbers were
/// equal by construction. Hash what actually gets applied.
pub(crate) fn payload_digest(payloads: &[(Vec<u8>, Vec<u8>)]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (k, v) in payloads {
        for &b in k.iter().chain(v.iter()) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Why [`handoff_decision`] refused a parked slice. Variants are 1:1 with the `kv_handoff_*` events
/// the drain logs; the cert greps two of them, so keep the mapping.
#[derive(Debug, PartialEq)]
pub(crate) enum HandoffReject {
    Validate,
    Decode,
    /// The slice's KV depth, which this rank is already past.
    TooLate(usize),
}

/// The pure part of the sparse-MoE drain — structural validation, structured decode, depth guard —
/// split out because applying needs a loaded runner, so this is the only part a unit test can reach.
///
/// Deliberately NOT the OpenVINO `handoff_decision`. That one validates against `OPAQUE_KV_LAYOUT`
/// and decodes one opaque blob via `wire_to_blob`; this layout is `KV_LAYOUT_VERSION` with a payload
/// pair per layer. Neither function can check the other's slice.
///
/// Validation MUST happen on this path: the hand-off skips the consumer `insert` that would otherwise
/// run the codec, so nothing else checks layout / engine_rev / fingerprint before `restore_kv`, and a
/// slice from a drifted build or another model would be restored silently.
pub(crate) fn handoff_decision(
    slot: &KvHandoffSlot,
    model_fp: u64,
    position: usize,
) -> Result<KvSnapshot, HandoffReject> {
    let refs: Vec<(&[u8], &[u8])> = slot
        .payloads
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    if cascadia_kv_wire::KvSnapshotCodec::validate(
        &slot.manifest,
        &refs,
        KV_LAYOUT_VERSION,
        KV_ENGINE_REV,
        model_fp,
        &slot.manifest.token_ids,
    )
    .is_err()
    {
        return Err(HandoffReject::Validate);
    }
    let snap = wire_to_snapshot(&slot.manifest, &slot.payloads).ok_or(HandoffReject::Decode)?;
    // A slice shallower than where this rank already sits cannot be resumed into — restoring it would
    // snap the cursor backwards, which is what produced the OV two-item divergence.
    if position > snap.past_seq_len {
        return Err(HandoffReject::TooLate(snap.past_seq_len));
    }
    Ok(snap)
}

/// Drain `mailbox` and hand the decided snapshot to `apply`. `true` ⇒ this rank is now armed warm,
/// which is what makes its `RESTORE` verdict truthful in plane mode.
///
/// Event names and fields are byte-identical to the OpenVINO drain on purpose — the cert greps them
/// and cannot tell the engines apart.
pub(crate) fn drain_handoff(
    mailbox: &KvHandoffMailbox,
    model_fp: u64,
    position: usize,
    apply: impl FnOnce(&KvSnapshot) -> bool,
) -> bool {
    let Some(slot) = mailbox.take() else {
        return false;
    };
    let digest = payload_digest(&slot.payloads);
    let snap = match handoff_decision(&slot, model_fp, position) {
        Ok(snap) => snap,
        Err(HandoffReject::Validate) => {
            tracing::warn!(target: "cascadia::kv", event = "kv_handoff_validate_failed",
                epoch = slot.epoch, rev = KV_ENGINE_REV, fp = model_fp);
            return false;
        }
        Err(HandoffReject::Decode) => {
            tracing::warn!(target: "cascadia::kv", event = "kv_handoff_decode_failed", epoch = slot.epoch);
            return false;
        }
        Err(HandoffReject::TooLate(depth)) => {
            tracing::warn!(target: "cascadia::kv", event = "kv_handoff_too_late",
                epoch = slot.epoch, position, depth);
            return false;
        }
    };
    if apply(&snap) {
        tracing::info!(target: "cascadia::kv", event = "kv_handoff_applied_inline",
            epoch = slot.epoch, position, blob_digest = digest);
        true
    } else {
        // restore_kv failed ⇒ this rank stays cold on a turn the commit path armed as warm, and
        // nothing on this side can undo that. The arm exists to make the failure greppable.
        tracing::warn!(target: "cascadia::kv", event = "kv_handoff_apply_failed",
            epoch = slot.epoch, position);
        false
    }
}

impl KvCoordination for SparseMoEEngine {
    fn model_fingerprint(&self) -> u64 {
        // PLANE-level (model identity only), like the sibling OvMoe engine. A cross-chain move pulls
        // every rank's KV under ONE fp — the moved-to head's — so a per-stage `digest()` here would
        // reject every worker rank of a legitimate move, leaving the plane path warm only at rank 0.
        // Local cache keys keep the full `digest()` (a rank-0 snapshot must never restore on rank 1).
        self.runner.fingerprint().plane_digest()
    }

    fn layout_version(&self) -> u16 {
        KV_LAYOUT_VERSION
    }

    fn engine_rev(&self) -> u64 {
        KV_ENGINE_REV
    }

    fn tokenize(&self, text: &str) -> Option<Vec<i32>> {
        // add_special_tokens=true mirrors the prefill path (engine.rs single-stage encode), so the
        // head's NEGOTIATE tokens equal what keys the prefix cache.
        let enc = self.tokenizer.as_ref()?.encode(text, true).ok()?;
        Some(enc.get_ids().iter().map(|&u| u as i32).collect())
    }

    fn lookup(&mut self, _partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        if !self.kv_prefix_cache.enabled() {
            return None; // capacity-0 (no kv_coord, or size 0) — nothing to offer
        }
        let fp = self.runner.fingerprint();
        let prompt: Vec<i64> = token_ids.iter().map(|&t| i64::from(t)).collect();
        let (snap, _) = self.kv_prefix_cache.lookup(&prompt, &fp)?;
        let len = snap.past_seq_len;
        let prefix = token_ids.get(..len)?.to_vec();
        let epoch = synth_epoch(&prefix);
        self.kv_offers.stash(epoch, prefix, snap);
        Some((epoch, len as u32))
    }

    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        // Plane fp: must equal what the consumer asserts (see `model_fingerprint`).
        let model_fp = self.runner.fingerprint().plane_digest();
        // Two sources, checked in order:
        //  - `kv_offers`: the head/single-stage NEGOTIATE→GET correlation (short-lived, single-use).
        //  - `kv_capture`: Task 1.3 multi-stage per-rank store (persistent; a worker has no NEGOTIATE,
        //    so its slice is stashed at CAPTURE time and may serve repeat/later GETs — clone, no remove).
        let (prefix, snap) = if let Some(off) = self.kv_offers.take(expected_epoch) {
            off
        } else if let Some((tokens, snap)) = self.kv_capture.get(&expected_epoch) {
            (tokens.clone(), snap.clone())
        } else {
            return None;
        };
        if snap.past_seq_len as u32 != expected_len {
            return None; // drifted from what was negotiated
        }
        let (manifest, payloads) =
            snapshot_to_wire(&prefix, &snap, partner, model_fp, expected_epoch);
        // Serve-side identity of exactly what this holder hands out, so it can be compared against
        // the consumer's `kv_handoff_applied_inline` digest. The length check above is LENGTH-only —
        // it never compares tokens — so a capture under a colliding synthesized epoch with the same
        // length but different tokens would serve silently; this is what makes that visible.
        tracing::info!(target: "cascadia::kv", event = "kv_serve_digest",
            epoch = expected_epoch, len = expected_len, blob_digest = payload_digest(&payloads),
            n_tokens = prefix.len(), n_payloads = payloads.len());
        Some((manifest, payloads))
    }

    fn insert(&mut self, manifest: &Manifest, payloads: &[(Vec<u8>, Vec<u8>)]) -> Result<(), ()> {
        let snap = wire_to_snapshot(manifest, payloads).ok_or(())?;
        // Stage under the CONTENT EPOCH too. `apply_warm_resume` — the plane's commit — reads
        // `kv_capture[epoch]`, but this only wrote the prefix cache (keyed by tokens), so a plane
        // consumer-insert staged a slice the commit could never find and EVERY plane warm-resume
        // silently voted cold. Done before the `enabled()` early-return: the plane path needs the
        // staging even where the prefix cache is off. Bounded by the same cap as
        // `capture_under_epoch` so staging cannot grow unbounded.
        {
            let epoch = crate::kv_coordination::synth_epoch(&manifest.token_ids);
            let cap = self.kv_prefix_cache.capacity().max(1);
            while self.kv_capture.len() >= cap && !self.kv_capture.contains_key(&epoch) {
                let Some(k) = self.kv_capture.keys().next().copied() else {
                    break;
                };
                self.kv_capture.remove(&k);
            }
            self.kv_capture
                .insert(epoch, (manifest.token_ids.clone(), snap.clone()));
        }
        if !self.kv_prefix_cache.enabled() {
            return Ok(()); // cache disabled → prefix-cache mirror is a no-op; the plane staging above still ran
        }
        let fp = self.runner.fingerprint();
        let prefix: Vec<i64> = manifest.token_ids.iter().map(|&t| i64::from(t)).collect();
        // Mirror into the holder cache so a busy engine still serves this prefix lock-free.
        self.kv_share
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prefix
            .insert_pulled(prefix.clone(), &fp, snap.clone());
        self.kv_prefix_cache.insert_pulled(prefix, &fp, snap);
        Ok(())
    }

    fn apply_warm_resume(&mut self, epoch: u64) -> bool {
        // Plane-driven warm-resume: restore the head's own rank-0 slice staged under `epoch`, then
        // RESTORE the whole downstream chain (all-or-nothing). Mirrors the worker RESTORE handler's
        // local apply. A head-local miss ⇒ false (the caller cold-runs; never a partial restore).
        //
        // Drain FIRST, never `local_ok || self.drain_kv_handoff()`. The node parks EVERY rank's
        // slice — rank 0 included — so on a cross-chain pull this head holds both a mailbox slice
        // and, from an earlier turn, its own same-epoch capture. A trailing `||` short-circuits the
        // drain away: the head warms from the stale local capture, the pulled slice is never
        // applied, and the verdict still reads true — a hollow warm nothing aborts (07d9bf2/a8fc4b4).
        let local_ok = if self.drain_kv_handoff() {
            true
        } else {
            match self.kv_capture.get(&epoch).cloned() {
                Some((_t, snap)) => self.runner.restore_kv(&snap).is_ok(),
                None => false,
            }
        };
        if !local_ok {
            return false;
        }
        self.forward_restore_downstream(epoch)
    }
}

/// Shared handle to the holder-side snapshot cache. The engine mirrors its captures here; the holder
/// reads it without ever taking the engine lock (so a busy node still answers a pull).
pub(crate) type SharedHolderCache = std::sync::Arc<std::sync::Mutex<SparseHolderState>>;

/// Holder-side mirror of the engine's KV caches: the content-keyed prefix cache, the NEGOTIATE→GET
/// offers, and the multi-stage captures. Populated alongside the engine's own caches so
/// [`SparseMoeKvHolder`] can serve NEGOTIATE/GET without the engine lock.
pub(crate) struct SparseHolderState {
    /// Mirror of the engine's `kv_prefix_cache`. `pub(crate)` so the engine's store sites mirror here.
    pub(crate) prefix: KvPrefixCache,
    /// Holder-internal NEGOTIATE→GET correlation (mirror of the engine's `kv_offers`).
    offers: KvOfferStash<KvSnapshot>,
    /// Mirror of the engine's `kv_capture` (multi-stage per-rank captures). `pub(crate)` for mirroring.
    pub(crate) captures: std::collections::HashMap<u64, (Vec<i32>, KvSnapshot)>,
    /// Fingerprint snapshotted at holder creation (the prefix cache's lookup key).
    fp: ModelFingerprint,
}

impl SparseHolderState {
    pub(crate) fn new(capacity: usize, fp: ModelFingerprint) -> Self {
        Self {
            prefix: KvPrefixCache::new(capacity),
            offers: KvOfferStash::new(KV_MAX_OFFERS, KV_MAX_OFFER_BYTES, KvSnapshot::approx_bytes),
            captures: std::collections::HashMap::new(),
            fp,
        }
    }
}

/// Lock-free [`cascadia_engine::KvSnapshotHolder`] over a [`SharedHolderCache`]. Serves NEGOTIATE/GET
/// by locking ONLY the holder cache — which inference touches only briefly at capture, not across the
/// forward pass — so the holder never starves on the engine lock the live generation holds. `model_fp`
/// is the engine's static stage fingerprint digest, snapshotted at handle creation.
pub(crate) struct SparseMoeKvHolder {
    pub(crate) cache: SharedHolderCache,
    pub(crate) model_fp: u64,
}

impl cascadia_engine::KvSnapshotHolder for SparseMoeKvHolder {
    fn model_fingerprint(&self) -> u64 {
        self.model_fp
    }

    fn lookup(&self, _partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        // Replicates `KvCoordination::lookup` against the mirrored holder cache.
        let mut g = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if !g.prefix.enabled() {
            return None; // capacity-0 (no kv_coord, or size 0) — nothing to offer
        }
        // Clone the fingerprint so `g.prefix` can take the `&mut` borrow the lookup needs while `g.fp`
        // is read — a simultaneous field split isn't possible through the mutex guard's Deref.
        let fp = g.fp.clone();
        let prompt: Vec<i64> = token_ids.iter().map(|&t| i64::from(t)).collect();
        let (snap, _) = g.prefix.lookup(&prompt, &fp)?;
        let len = snap.past_seq_len;
        let prefix = token_ids.get(..len)?.to_vec();
        let epoch = synth_epoch(&prefix);
        g.offers.stash(epoch, prefix, snap);
        Some((epoch, len as u32))
    }

    fn export(
        &self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        // Replicates `KvCoordination::export` against the mirrored holder cache.
        let mut g = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let (prefix, snap) = if let Some(off) = g.offers.take(expected_epoch) {
            off
        } else if let Some((tokens, snap)) = g.captures.get(&expected_epoch) {
            (tokens.clone(), snap.clone())
        } else {
            return None;
        };
        if snap.past_seq_len as u32 != expected_len {
            return None; // drifted from what was negotiated
        }
        Some(snapshot_to_wire(
            &prefix,
            &snap,
            partner,
            self.model_fp,
            expected_epoch,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_engine::KvSnapshotHolder;
    use cascadia_kv_wire::KvSnapshotCodec;

    // num_heads=1, len=3, qk_head_dim=2, v_head_dim=1 ⇒ K=6 u16, V=3 u16.
    fn snap() -> KvSnapshot {
        KvSnapshot {
            past_seq_len: 3,
            num_heads: 1,
            qk_head_dim: 2,
            v_head_dim: 1,
            layer0: Some(LayerKvSlice {
                lid: 0,
                past_k: vec![1, 2, 3, 4, 5, 6],
                past_v: vec![7, 8, 9],
            }),
            shells: vec![LayerKvSlice {
                lid: 1,
                past_k: vec![10, 11, 12, 13, 14, 15],
                past_v: vec![16, 17, 18],
            }],
        }
    }

    #[test]
    fn wire_roundtrip_and_codec_accepts() {
        let prefix = vec![11, 22, 33];
        let (m, payloads) = snapshot_to_wire(&prefix, &snap(), "acme", 7, 0xABCD);
        // The exported manifest must pass the consumer's structural validation.
        let refs: Vec<(&[u8], &[u8])> = payloads
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        assert!(
            KvSnapshotCodec::validate(&m, &refs, KV_LAYOUT_VERSION, KV_ENGINE_REV, 7, &prefix)
                .is_ok(),
            "exported manifest must validate"
        );
        assert_eq!(m.num_layers, 2);
        assert_eq!(m.prefix_token_len, 3);
        // Round-trips back to an equivalent snapshot (layer0 + shells, buffers intact).
        let back = wire_to_snapshot(&m, &payloads).unwrap();
        assert_eq!(back.past_seq_len, 3);
        assert_eq!(
            (back.num_heads, back.qk_head_dim, back.v_head_dim),
            (1, 2, 1)
        );
        assert_eq!(back.layer0.as_ref().unwrap().past_k, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(back.layer0.as_ref().unwrap().past_v, vec![7, 8, 9]);
        assert_eq!(back.shells.len(), 1);
        assert_eq!(back.shells[0].lid, 1);
        assert_eq!(back.shells[0].past_k, vec![10, 11, 12, 13, 14, 15]);
    }

    /// A snapshot whose `approx_bytes()` is exactly `bytes` (k + v, u16 each ⇒ 4 bytes per step).
    fn snap_of(bytes: usize) -> KvSnapshot {
        let n = bytes / 4;
        KvSnapshot {
            past_seq_len: 1,
            num_heads: 1,
            qk_head_dim: 1,
            v_head_dim: 1,
            layer0: Some(LayerKvSlice {
                lid: 0,
                past_k: vec![0; n],
                past_v: vec![0; n],
            }),
            shells: vec![],
        }
    }

    /// One NEGOTIATE's worth of stashing: epoch `e`, an offer of `bytes`.
    fn offer(s: &mut KvOfferStash<KvSnapshot>, e: u64, bytes: usize) {
        s.stash(e, vec![e as i32], snap_of(bytes));
    }

    fn stash() -> KvOfferStash<KvSnapshot> {
        KvOfferStash::new(KV_MAX_OFFERS, KV_MAX_OFFER_BYTES, KvSnapshot::approx_bytes)
    }

    /// An unpaired-NEGOTIATE flood from an admitted peer must not pin more than the stated budget:
    /// the count cap alone let it pin `cap × blob`, on a node already under memory pressure.
    #[test]
    fn unpaired_negotiate_flood_stays_within_the_byte_budget() {
        let mut s = stash();
        let blob = KV_MAX_OFFER_BYTES / 8;
        for e in 0..20 {
            offer(&mut s, e, blob);
            let held = s.bytes();
            assert!(held <= KV_MAX_OFFER_BYTES, "offer {e} pinned {held}");
        }
        assert_eq!(s.len(), 8, "exactly budget/blob offers survive");
        assert_eq!(s.order_len(), s.len());
    }

    /// Arbitrary `HashMap` order can evict the offer whose paired GET is already in flight and keep a
    /// stale one. Driven through the COUNT cap so every eviction is checked — a two-candidate
    /// byte-budget version passes the buggy code on a coin flip.
    #[test]
    fn offer_eviction_is_oldest_first() {
        let mut s = stash();
        let n = KV_MAX_OFFERS as u64 + 8;
        for e in 0..n {
            offer(&mut s, e, 4);
        }
        assert!((0..8).all(|e| !s.contains(e)), "8 oldest offers go first");
        assert!((8..n).all(|e| s.contains(e)));
        assert_eq!(s.evicted(), 8, "every unserved eviction is counted");
    }

    #[test]
    fn a_paired_get_still_serves_after_evictions() {
        let mut s = stash();
        let n = KV_MAX_OFFERS as u64 + 8;
        for e in 0..n {
            offer(&mut s, e, 4);
        }
        let (tokens, snap) = s.take(n - 1).expect("newest offer still stashed");
        assert_eq!(tokens, vec![(n - 1) as i32]);
        assert_eq!(snap.approx_bytes(), 4);
    }

    /// An offer bigger than the whole budget is stashed alone rather than evicted to death — the
    /// alternative leaves the node unable to serve its largest cached turn at all.
    #[test]
    fn an_offer_over_the_budget_is_still_servable() {
        let mut s = stash();
        offer(&mut s, 1, 8);
        offer(&mut s, 2, KV_MAX_OFFER_BYTES + 4);
        assert_eq!(s.len(), 1);
        assert!(s.take(2).is_some(), "the over-budget offer is servable");
        assert_eq!(s.bytes(), 0);
    }

    /// Byte accounting must not drift: every offer leaves through a GET or an eviction.
    #[test]
    fn offer_bytes_return_to_zero() {
        let mut s = stash();
        let n = KV_MAX_OFFERS as u64 + 8;
        for e in 0..n {
            offer(&mut s, e, 4);
        }
        for e in 0..n {
            let _ = s.take(e);
        }
        assert_eq!((s.len(), s.order_len(), s.bytes()), (0, 0, 0));
    }

    /// A re-NEGOTIATE of the same prefix replaces its offer. Without that, a routine peer retry leaks
    /// an order slot and over-counts bytes until the stash pins itself at one entry.
    #[test]
    fn re_negotiating_the_same_prefix_replaces_its_offer() {
        let mut s = stash();
        offer(&mut s, 0xF0, 64);
        offer(&mut s, 0xF0, 64);
        assert_eq!((s.len(), s.order_len(), s.bytes()), (1, 1, 64));
        assert!(s.take(0xF0).is_some());
        assert_eq!(s.bytes(), 0);
    }

    fn fp() -> ModelFingerprint {
        ModelFingerprint {
            arch: "k26".into(),
            num_layers: 1,
            num_experts: 1,
            top_k: 1,
            hidden_size: 8,
            num_kv_heads: 1,
            qk_head_dim: 1,
            v_head_dim: 1,
            vocab_size: 256,
            layer_start: 0,
            layer_end: 1,
            is_first: true,
            is_last: true,
        }
    }

    /// The holder's NEGOTIATE path is bounded, and the newest offer still has its paired GET.
    #[test]
    fn holder_negotiate_flood_is_bounded_and_still_serves() {
        let n = KV_MAX_OFFERS as i32 + 8;
        let mut st = SparseHolderState::new(n as usize, fp());
        for i in 0..n {
            st.prefix.insert(vec![i64::from(i)], &fp(), snap_of(4));
        }
        let holder = SparseMoeKvHolder {
            cache: std::sync::Arc::new(std::sync::Mutex::new(st)),
            model_fp: fp().digest(),
        };
        let mut last = None;
        for i in 0..n {
            last = holder.lookup("peer", &[i, 0]);
        }
        {
            let g = holder.cache.lock().unwrap();
            assert_eq!(g.offers.len(), KV_MAX_OFFERS);
            assert!(g.offers.bytes() <= KV_MAX_OFFER_BYTES);
        }
        let (epoch, len) = last.expect("negotiate hit");
        assert!(holder.export("peer", epoch, len).is_some());
    }

    fn slot_of(prefix: &[i32], snap: &KvSnapshot, model_fp: u64) -> KvHandoffSlot {
        let (manifest, payloads) = snapshot_to_wire(prefix, snap, "peer", model_fp, 0xE0);
        KvHandoffSlot {
            epoch: 0xE0,
            manifest,
            payloads,
        }
    }

    #[test]
    fn handoff_decision_accepts_a_well_formed_slice() {
        let prefix = vec![11, 22, 33];
        let slot = slot_of(&prefix, &snap(), 7);
        let decided = handoff_decision(&slot, 7, 0).expect("well-formed slice must decide Ok");
        assert_eq!(decided.past_seq_len, 3);
        assert_eq!(decided.shells.len(), 1);
    }

    #[test]
    fn handoff_decision_rejects_a_foreign_model() {
        // The hand-off path skips the consumer insert that would otherwise run the codec, so this
        // check is the ONLY thing standing between a foreign slice and `restore_kv`.
        let slot = slot_of(&[11, 22, 33], &snap(), 7);
        assert_eq!(
            handoff_decision(&slot, 8, 0).unwrap_err(),
            HandoffReject::Validate,
            "a slice from another model must not reach restore_kv"
        );
    }

    #[test]
    fn handoff_decision_rejects_a_slice_shallower_than_this_rank() {
        // Restoring a 3-deep slice into a rank already at 5 would snap the cursor backwards.
        let slot = slot_of(&[11, 22, 33], &snap(), 7);
        assert_eq!(
            handoff_decision(&slot, 7, 5).unwrap_err(),
            HandoffReject::TooLate(3)
        );
        // Equal depth is not "too late" — the guard is `position > depth`.
        assert!(handoff_decision(&slot, 7, 3).is_ok());
    }

    #[test]
    fn drain_handoff_consumes_what_the_plane_parked() {
        let mailbox = KvHandoffMailbox::new();
        assert!(
            !drain_handoff(&mailbox, 7, 0, |_| true),
            "an empty mailbox drains false"
        );
        let (manifest, payloads) = snapshot_to_wire(&[11, 22, 33], &snap(), "peer", 7, 0xE0);
        mailbox.put(0xE0, manifest, payloads);
        assert!(mailbox.ever_parked());
        let mut applied = None;
        assert!(drain_handoff(&mailbox, 7, 0, |snap| {
            applied = Some(snap.past_seq_len);
            true
        }));
        assert_eq!(applied, Some(3), "the parked slice is what got applied");
        // One slot: a second drain finds nothing.
        assert!(!drain_handoff(&mailbox, 7, 0, |_| true));
    }

    #[test]
    fn drain_handoff_reports_false_when_the_apply_fails() {
        // A failed restore leaves the rank cold on a turn the commit armed as warm. The drain must
        // say so, or the RESTORE verdict lies and the head warms over a cold rank.
        let mailbox = KvHandoffMailbox::new();
        let (manifest, payloads) = snapshot_to_wire(&[11, 22, 33], &snap(), "peer", 7, 0xE0);
        mailbox.put(0xE0, manifest, payloads);
        assert!(!drain_handoff(&mailbox, 7, 0, |_| false));
    }

    #[test]
    fn serve_and_applied_digests_agree() {
        // Bar: the consumer's applied digest must equal the donor's serve digest. Both sides digest
        // the payload bytes, so a layout change on one side alone breaks this rather than silently
        // warming from the wrong bytes.
        let (_m, served) = snapshot_to_wire(&[11, 22, 33], &snap(), "peer", 7, 0xE0);
        let slot = slot_of(&[11, 22, 33], &snap(), 7);
        assert_eq!(payload_digest(&served), payload_digest(&slot.payloads));
    }
}
