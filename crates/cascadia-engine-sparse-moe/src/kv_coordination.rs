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
//! **Sharded gap (rig):** the cache is capacity-0 for `total>1`, so a holder serves nothing for a
//! sharded model until `total>1` capture is re-enabled — lookup/insert degrade to None/no-op.

use cascadia_engine::KvCoordination;
use cascadia_kv_wire::{LayerMeta, Manifest, PartnerId, KV_LAYOUT_VERSION, SCHEMA_VERSION};

use crate::engine::SparseMoEEngine;
use crate::kv_prefix_cache::{KvSnapshot, LayerKvSlice};

/// KV codec/engine revision — bump on any change to the snapshot buffer layout. Producer (export) and
/// consumer (`consumer_engine_rev`) both read this, so the codec's engine-rev guard is self-consistent.
pub const KV_ENGINE_REV: u64 = 1;

/// Cap on stashed unconsumed offers (NEGOTIATE without a paired GET); oldest dropped on overflow.
const KV_MAX_OFFERS: usize = 64;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Content-derived epoch over the negotiated prefix tokens.
fn synth_epoch(prefix: &[i32]) -> u64 {
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

impl KvCoordination for SparseMoEEngine {
    fn model_fingerprint(&self) -> u64 {
        self.runner.fingerprint().digest()
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
            return None; // capacity-0 (e.g. total>1, sharded) — nothing to offer
        }
        let fp = self.runner.fingerprint();
        let prompt: Vec<i64> = token_ids.iter().map(|&t| i64::from(t)).collect();
        let snap = self.kv_prefix_cache.lookup(&prompt, &fp)?;
        let len = snap.past_seq_len;
        let prefix = token_ids.get(..len)?.to_vec();
        let epoch = synth_epoch(&prefix);
        if self.kv_offers.len() >= KV_MAX_OFFERS {
            // Drop an arbitrary stale offer (bounded growth; offers are short-lived NEGOTIATE→GET).
            if let Some(k) = self.kv_offers.keys().next().copied() {
                self.kv_offers.remove(&k);
            }
        }
        self.kv_offers.insert(epoch, (prefix, snap));
        Some((epoch, len as u32))
    }

    fn export(
        &mut self,
        partner: &str,
        expected_epoch: u64,
        expected_len: u32,
    ) -> Option<(Manifest, Vec<(Vec<u8>, Vec<u8>)>)> {
        let model_fp = self.runner.fingerprint().digest();
        let (prefix, snap) = self.kv_offers.remove(&expected_epoch)?; // single-use per GET
        if snap.past_seq_len as u32 != expected_len {
            return None; // drifted from what was negotiated
        }
        Some(snapshot_to_wire(
            &prefix,
            &snap,
            partner,
            model_fp,
            expected_epoch,
        ))
    }

    fn insert(&mut self, manifest: &Manifest, payloads: &[(Vec<u8>, Vec<u8>)]) -> Result<(), ()> {
        let snap = wire_to_snapshot(manifest, payloads).ok_or(())?;
        if !self.kv_prefix_cache.enabled() {
            return Ok(()); // sharded/total>1: cache disabled → no-op (warm path inert until rig)
        }
        let fp = self.runner.fingerprint();
        let prefix: Vec<i64> = manifest.token_ids.iter().map(|&t| i64::from(t)).collect();
        self.kv_prefix_cache.insert(prefix, &fp, snap);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
