use serde::{Deserialize, Serialize};

/// Wire format version — owned here, producer-stamped, consumer-compared (§6).
pub const SCHEMA_VERSION: u16 = 1;
/// KV bit-layout / int4 shell ABI version — owned here, bumped on any KV layout change (§6).
pub const KV_LAYOUT_VERSION: u16 = 1;
/// KV element size in bytes (the sparse-MoE engine stores bf16 as `u16`).
pub const DTYPE_SIZE: usize = 2;
/// Index of the sequence axis in a per-layer shape `[num_heads, n_slots(seq), head_dim]`.
/// Shared by [`crate::KvSnapshotCodec`] and the engine `reslice_to` so they cannot drift.
pub const SEQ_AXIS: usize = 1;

/// Enterprise partner/tenant id (the shipped type is `String`; wrapped for the wire). F3.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartnerId(pub String);

/// Content-addressed, **tenant-namespaced** cache key (F3): `(partner, model, prefix-hash)`.
/// Namespacing is defense-in-depth; the load-bearing control is that you must know the tokens
/// to construct `prefix_token_hash` (§13).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    pub partner_hash: u64,
    pub model_fingerprint: u64,
    pub prefix_token_hash: u64,
}

impl CacheKey {
    #[must_use]
    pub fn derive(partner: &PartnerId, model_fingerprint: u64, token_ids: &[i32]) -> Self {
        Self {
            partner_hash: fnv1a(partner.0.as_bytes()),
            model_fingerprint,
            prefix_token_hash: hash_tokens(token_ids),
        }
    }
}

/// Per-layer metadata. K and V carry **separate** shapes because the sparse-MoE engine has
/// distinct `qk_head_dim` and `v_head_dim` (§6, plan C3). Each shape's [`SEQ_AXIS`] equals the
/// negotiated `prefix_token_len` after the holder's gather.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerMeta {
    pub layer_index: u32,
    pub k_shape: Vec<u32>,
    pub v_shape: Vec<u32>,
    pub k_byte_len: u64,
    pub v_byte_len: u64,
    pub k_crc32: u32,
    pub v_crc32: u32,
}

/// Explicit, versioned, self-describing, checksummed wire manifest, decoupled from the engine's
/// internal `KvSnapshot` (§6). Followed on the stream by per-layer length-prefixed K then V bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u16,
    pub kv_layout_version: u16,
    pub engine_rev: u64,
    pub partner: PartnerId,
    pub model_fingerprint: u64,
    pub prefix_token_hash: u64,
    /// Negotiated whole-token count; `== token_ids.len()`, `<= request token count` (§6).
    pub prefix_token_len: u32,
    /// `(producer_nonce_high || monotonic_low)` — globally unique per producing head (§8).
    pub snapshot_epoch: u64,
    pub num_layers: u32,
    pub layers: Vec<LayerMeta>,
    /// Exactly `[0..prefix_token_len)` — for the re-compare and the local self-warm insert.
    pub token_ids: Vec<i32>,
}

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub(crate) fn hash_tokens(token_ids: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in token_ids {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_tenant_namespaced() {
        let a = CacheKey::derive(&PartnerId("acme".into()), 0x1234, &[1, 2, 3]);
        let b = CacheKey::derive(&PartnerId("globex".into()), 0x1234, &[1, 2, 3]);
        assert_ne!(a, b, "different tenants must namespace apart (F3)");
        let c = CacheKey::derive(&PartnerId("acme".into()), 0x1234, &[1, 2, 3]);
        assert_eq!(a, c, "same tenant+model+tokens must hit");
    }

    #[test]
    fn cache_key_depends_on_model_and_tokens() {
        let base = CacheKey::derive(&PartnerId("acme".into()), 0x1234, &[1, 2, 3]);
        assert_ne!(
            base,
            CacheKey::derive(&PartnerId("acme".into()), 0x9999, &[1, 2, 3])
        );
        assert_ne!(
            base,
            CacheKey::derive(&PartnerId("acme".into()), 0x1234, &[1, 2, 4])
        );
    }
}
