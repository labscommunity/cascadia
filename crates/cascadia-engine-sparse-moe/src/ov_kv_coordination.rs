//! Cross-chain KV warm-pull for the OV-IR MoE engine (MiniMax-M2 / `OvMoeEngine`).
//!
//! Mirrors the K2.6 cross-chain plane (`kv_coordination.rs`) but for OvMoe's f32-native snapshot,
//! shipped over the wire as an OPAQUE blob (`OPAQUE_KV_LAYOUT`: length + crc only, dtype-agnostic —
//! the same escape hatch the OpenVINO engines use via `blob_to_wire`). f32 verbatim ⇒ a cross-chain
//! warm resume is byte-identical to a cold prefill, exactly like the local multi-stage path. The K2.6
//! structured bf16 codec is deliberately NOT used (it would truncate f32 → break byte-identity).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cascadia_engine::KvSnapshotHolder;
use cascadia_kv_wire::{LayerMeta, Manifest, PartnerId, OPAQUE_KV_LAYOUT, SCHEMA_VERSION};

use crate::kv_coordination::{synth_epoch, KvOfferStash};
use crate::kv_prefix_cache::ModelFingerprint;
use crate::ov_kv_cache::{OvLayerKvSlice, OvMoeKvPrefixCache, OvMoeKvSnapshot};

/// OvMoe cross-chain engine revision. Bump on any change to the on-wire blob format below.
pub(crate) const KV_ENGINE_REV: u64 = 1;
/// Bound on live NEGOTIATE→GET offers (short-lived correlation entries).
pub(crate) const KV_MAX_OFFERS: usize = 64;
/// Byte ceiling on ONE `offers` map; an engine holds two (`kv_offers` + the `kv_share` mirror), and
/// `lookup` has already cloned the snapshot before `stash` evicts — so node peak is ~2×(this + one
/// snapshot), not 2×this. Same number as the K2.6 plane's
/// [`crate::kv_coordination::KV_MAX_OFFER_BYTES`] because it bounds bytes, not blobs; f32 KV just
/// means fewer offers fit. Shrunk under `cfg(test)` so the eviction tests cost MB, not GB.
pub(crate) const KV_MAX_OFFER_BYTES: usize = if cfg!(test) { 4 << 20 } else { 256 << 20 };

// ---- opaque f32 blob codec -------------------------------------------------
// Self-describing little-endian layout:
//   u32 past_seq_len
//   u32 num_layers
//   per layer: u32 seq, u32 k_len, k_len × f32, u32 v_len, v_len × f32

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_f32s(out: &mut Vec<u8>, xs: &[f32]) {
    put_u32(out, xs.len() as u32);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

pub(crate) fn ov_blob_encode(snap: &OvMoeKvSnapshot) -> Vec<u8> {
    let mut out = Vec::new();
    put_u32(&mut out, snap.past_seq_len as u32);
    put_u32(&mut out, snap.layers.len() as u32);
    for l in &snap.layers {
        put_u32(&mut out, l.seq as u32);
        put_f32s(&mut out, &l.k);
        put_f32s(&mut out, &l.v);
    }
    out
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}
impl Reader<'_> {
    fn u32(&mut self) -> Option<u32> {
        let e = self.pos.checked_add(4)?;
        let v = u32::from_le_bytes(self.b.get(self.pos..e)?.try_into().ok()?);
        self.pos = e;
        Some(v)
    }
    fn f32s(&mut self) -> Option<Vec<f32>> {
        let n = self.u32()? as usize;
        let bytes = n.checked_mul(4)?;
        let e = self.pos.checked_add(bytes)?;
        let slice = self.b.get(self.pos..e)?;
        let out = slice
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.pos = e;
        Some(out)
    }
}

pub(crate) fn ov_blob_decode(blob: &[u8]) -> Option<OvMoeKvSnapshot> {
    let mut r = Reader { b: blob, pos: 0 };
    let past_seq_len = r.u32()? as usize;
    let n = r.u32()? as usize;
    let mut layers = Vec::with_capacity(n);
    for _ in 0..n {
        let seq = r.u32()? as usize;
        let k = r.f32s()?;
        let v = r.f32s()?;
        layers.push(OvLayerKvSlice { k, v, seq });
    }
    Some(OvMoeKvSnapshot {
        past_seq_len,
        layers,
    })
}

// ---- wire wrappers (OPAQUE_KV_LAYOUT) --------------------------------------

/// Serialize an [`OvMoeKvSnapshot`] into an opaque wire snapshot (one payload, `k` = blob, `v` = empty).
/// Mirrors the OpenVINO engines' `blob_to_wire`; the codec enforces only length + crc under
/// `OPAQUE_KV_LAYOUT`, so the f32 bytes ride verbatim.
pub(crate) fn ov_snapshot_to_wire(
    prefix: &[i32],
    snap: &OvMoeKvSnapshot,
    partner: &str,
    model_fingerprint: u64,
    epoch: u64,
) -> (Manifest, Vec<(Vec<u8>, Vec<u8>)>) {
    let blob = ov_blob_encode(snap);
    let meta = LayerMeta {
        layer_index: 0,
        k_shape: vec![], // ignored under OPAQUE_KV_LAYOUT
        v_shape: vec![],
        k_byte_len: blob.len() as u64,
        v_byte_len: 0,
        k_crc32: crc32fast::hash(&blob),
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
    (manifest, vec![(blob, Vec::new())])
}

/// Inverse of [`ov_snapshot_to_wire`]. `None` on a non-opaque manifest, wrong payload count, or a
/// malformed blob.
pub(crate) fn ov_wire_to_snapshot(
    manifest: &Manifest,
    payloads: &[(Vec<u8>, Vec<u8>)],
) -> Option<OvMoeKvSnapshot> {
    if manifest.kv_layout_version != OPAQUE_KV_LAYOUT || payloads.len() != 1 {
        return None;
    }
    let blob = &payloads[0].0;
    if blob.is_empty() {
        return None;
    }
    ov_blob_decode(blob)
}

// ---- holder ----------------------------------------------------------------

/// Shared handle to the holder-side snapshot cache. The engine mirrors its captures here; the holder
/// reads it without ever taking the engine lock (so a busy node still answers a pull).
pub(crate) type OvSharedHolderCache = Arc<Mutex<OvHolderState>>;

/// Holder-side mirror of the engine's OvMoe KV caches (content-keyed prefix cache + NEGOTIATE→GET
/// offers + multi-stage captures), so [`OvMoeKvHolder`] serves NEGOTIATE/GET without the engine lock.
pub(crate) struct OvHolderState {
    pub(crate) prefix: OvMoeKvPrefixCache,
    offers: KvOfferStash<OvMoeKvSnapshot>,
    pub(crate) captures: HashMap<u64, (Vec<i32>, OvMoeKvSnapshot)>,
    fp: ModelFingerprint,
}
impl OvHolderState {
    pub(crate) fn new(capacity: usize, fp: ModelFingerprint) -> Self {
        Self {
            prefix: OvMoeKvPrefixCache::new(capacity),
            offers: KvOfferStash::new(
                KV_MAX_OFFERS,
                KV_MAX_OFFER_BYTES,
                OvMoeKvSnapshot::approx_bytes,
            ),
            captures: HashMap::new(),
            fp,
        }
    }
}

/// Lock-free [`KvSnapshotHolder`] over an [`OvSharedHolderCache`]. Serves NEGOTIATE/GET by locking
/// ONLY the holder cache, never the engine.
pub(crate) struct OvMoeKvHolder {
    pub(crate) cache: OvSharedHolderCache,
    pub(crate) model_fp: u64,
}

impl KvSnapshotHolder for OvMoeKvHolder {
    fn model_fingerprint(&self) -> u64 {
        self.model_fp
    }

    fn lookup(&self, _partner: &str, token_ids: &[i32]) -> Option<(u64, u32)> {
        let mut g = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if !g.prefix.enabled() {
            return None;
        }
        let fp = g.fp.clone();
        let prompt: Vec<i64> = token_ids.iter().map(|&t| i64::from(t)).collect();
        let hit = g.prefix.lookup(&prompt, &fp);
        tracing::debug!(target: "cascadia::kv", event = "ovmoe_holder_negotiate",
            query_len = prompt.len(), hit = hit.is_some(), "holder NEGOTIATE lookup");
        let (snap, _) = hit?;
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
        let mut g = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let (prefix, snap) = if let Some(off) = g.offers.take(expected_epoch) {
            off
        } else if let Some((tokens, snap)) = g.captures.get(&expected_epoch) {
            (tokens.clone(), snap.clone())
        } else {
            return None;
        };
        if snap.past_seq_len as u32 != expected_len {
            return None;
        }
        Some(ov_snapshot_to_wire(
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

    fn snap(past: usize, fill: f32) -> OvMoeKvSnapshot {
        OvMoeKvSnapshot {
            past_seq_len: past,
            layers: vec![
                OvLayerKvSlice {
                    k: vec![fill; 2 * past * 2],
                    v: vec![fill + 0.5; 2 * past * 2],
                    seq: past,
                },
                OvLayerKvSlice {
                    k: vec![fill - 1.0; 2 * past * 2],
                    v: vec![fill * 2.0; 2 * past * 2],
                    seq: past,
                },
            ],
        }
    }

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

    #[test]
    fn blob_roundtrip_is_byte_exact() {
        let s = snap(5, 3.25);
        let blob = ov_blob_encode(&s);
        let back = ov_blob_decode(&blob).expect("decode");
        assert_eq!(back.past_seq_len, s.past_seq_len);
        assert_eq!(back.layers.len(), 2);
        for (a, b) in back.layers.iter().zip(&s.layers) {
            assert_eq!(a.seq, b.seq);
            assert_eq!(a.k, b.k); // exact f32 bits — lossless
            assert_eq!(a.v, b.v);
        }
    }

    #[test]
    fn wire_roundtrip_preserves_snapshot_and_tokens() {
        let s = snap(3, 1.5);
        let prefix = vec![7i32, 8, 9];
        let (m, payloads) = ov_snapshot_to_wire(&prefix, &s, "peerX", 42, 999);
        assert_eq!(m.kv_layout_version, OPAQUE_KV_LAYOUT);
        assert_eq!(m.token_ids, prefix);
        assert_eq!(m.snapshot_epoch, 999);
        let back = ov_wire_to_snapshot(&m, &payloads).expect("wire decode");
        assert_eq!(back.past_seq_len, 3);
        assert_eq!(back.layers[0].k, s.layers[0].k);
        assert_eq!(back.layers[1].v, s.layers[1].v);
    }

    #[test]
    fn wire_rejects_non_opaque_or_empty() {
        let s = snap(2, 1.0);
        let (mut m, payloads) = ov_snapshot_to_wire(&[1, 2], &s, "p", 1, 1);
        m.kv_layout_version = 1; // structured, not opaque
        assert!(ov_wire_to_snapshot(&m, &payloads).is_none());
        let (m2, _) = ov_snapshot_to_wire(&[1, 2], &s, "p", 1, 1);
        assert!(ov_wire_to_snapshot(&m2, &[(vec![], vec![])]).is_none());
    }

    #[test]
    fn holder_lookup_then_export_roundtrips() {
        let mut st = OvHolderState::new(4, fp());
        // Source captured prefix [1,2,3] (its KV depth 3) into the holder's prefix mirror.
        st.prefix.insert(vec![1i64, 2, 3], &fp(), snap(3, 2.0));
        let holder = OvMoeKvHolder {
            cache: Arc::new(Mutex::new(st)),
            model_fp: fp().digest(),
        };
        // Consumer NEGOTIATEs a longer prompt sharing the prefix.
        let (epoch, len) = holder.lookup("peer", &[1, 2, 3, 4]).expect("negotiate hit");
        assert_eq!(len, 3);
        // GET the offer back.
        let (m, payloads) = holder.export("peer", epoch, len).expect("export");
        let snap = ov_wire_to_snapshot(&m, &payloads).expect("decode");
        assert_eq!(snap.past_seq_len, 3);
        // Offer is single-use.
        assert!(holder.export("peer", epoch, len).is_none());
    }

    /// The OvMoe holder's NEGOTIATE path is bounded too, and the newest offer still has its GET.
    #[test]
    fn holder_negotiate_flood_is_bounded_and_still_serves() {
        let n = KV_MAX_OFFERS as i32 + 8;
        let mut st = OvHolderState::new(n as usize, fp());
        for i in 0..n {
            st.prefix.insert(vec![i64::from(i)], &fp(), snap(1, 1.0));
        }
        let holder = OvMoeKvHolder {
            cache: Arc::new(Mutex::new(st)),
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
}
