use crate::manifest::{Manifest, DTYPE_SIZE, SCHEMA_VERSION, SEQ_AXIS};

/// Closed enum of structural reject reasons (§6/§11) — never a content-derived value.
#[derive(Debug, PartialEq, Eq)]
pub enum ValidateError {
    SchemaVersion,
    LayoutVersion,
    EngineRev,
    Fingerprint,
    NumLayers,
    ShapeSeq,
    ByteLenShape,
    Crc,
    TokenMismatch,
    /// `prefix_token_len` exceeds the requester's token count.
    LenOverrun,
    /// `token_ids.len()` disagrees with `prefix_token_len` (internal Manifest-length invariant).
    TokenLen,
}

/// Expected payload bytes for a tensor shape, with **checked** arithmetic. An adversarial shape whose
/// element product overflows `u64` must reject: a silently-wrapped small expectation would let an
/// undersized payload validate against a huge declared tensor → consumer OOB read at reslice time.
/// Empty shape rejects (a KV tensor has at least the seq axis).
fn expected_bytes(shape: &[u32]) -> Option<u64> {
    if shape.is_empty() {
        return None;
    }
    shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(u64::from(d)))?
        .checked_mul(DTYPE_SIZE as u64)
}

pub struct KvSnapshotCodec;

impl KvSnapshotCodec {
    /// Structural validation (§6) — **any mismatch → reject → reprefill**, never best-effort decode.
    /// `payloads[i] = (k_bytes, v_bytes)` for layer `i`, already gathered by the holder to
    /// `[0..prefix_token_len)`. Validation is structural only — it does NOT assert numeric
    /// bit-identity (the seam, §9). Order: schema → layout → engine_rev → model_fingerprint →
    /// num_layers → token-len/overrun → token re-compare → per-layer shape/byte_len/crc.
    pub fn validate(
        m: &Manifest,
        payloads: &[(&[u8], &[u8])],
        consumer_layout_version: u16,
        consumer_engine_rev: u64,
        consumer_model_fingerprint: u64,
        request_token_ids: &[i32],
    ) -> Result<(), ValidateError> {
        if m.schema_version != SCHEMA_VERSION {
            return Err(ValidateError::SchemaVersion);
        }
        if m.kv_layout_version != consumer_layout_version {
            return Err(ValidateError::LayoutVersion);
        }
        if m.engine_rev != consumer_engine_rev {
            return Err(ValidateError::EngineRev);
        }
        if m.model_fingerprint != consumer_model_fingerprint {
            return Err(ValidateError::Fingerprint);
        }
        if m.num_layers as usize != m.layers.len() || m.num_layers == 0 {
            return Err(ValidateError::NumLayers);
        }
        if payloads.len() != m.layers.len() {
            return Err(ValidateError::NumLayers);
        }
        let p = m.prefix_token_len as usize;
        if p > request_token_ids.len() {
            return Err(ValidateError::LenOverrun);
        }
        // Internal Manifest invariant: token_ids carries exactly the negotiated prefix.
        if m.token_ids.len() != p {
            return Err(ValidateError::TokenLen);
        }
        if m.token_ids[..] != request_token_ids[..p] {
            return Err(ValidateError::TokenMismatch);
        }
        for (lm, (k, v)) in m.layers.iter().zip(payloads.iter()) {
            // K and V seq axes must both equal the negotiated prefix length.
            if lm.k_shape.get(SEQ_AXIS).copied() != Some(m.prefix_token_len)
                || lm.v_shape.get(SEQ_AXIS).copied() != Some(m.prefix_token_len)
            {
                return Err(ValidateError::ShapeSeq);
            }
            // K and V validated independently (distinct qk_head_dim / v_head_dim, plan C3).
            // Checked product (see `expected_bytes`): an overflowing/empty shape rejects rather than
            // wrapping to a small expectation an undersized payload could match.
            let (Some(k_expect), Some(v_expect)) =
                (expected_bytes(&lm.k_shape), expected_bytes(&lm.v_shape))
            else {
                return Err(ValidateError::ByteLenShape);
            };
            if lm.k_byte_len != k_expect || lm.v_byte_len != v_expect {
                return Err(ValidateError::ByteLenShape);
            }
            if lm.k_byte_len as usize != k.len() || lm.v_byte_len as usize != v.len() {
                return Err(ValidateError::ByteLenShape);
            }
            if crc32fast::hash(k) != lm.k_crc32 || crc32fast::hash(v) != lm.v_crc32 {
                return Err(ValidateError::Crc);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::*;

    /// A consistent manifest: 1 layer, shape `[num_heads=1, seq=2, head_dim=1]` ⇒ 2 elems ⇒
    /// 4 bytes (u16) for each of K and V.
    fn good() -> (Manifest, Vec<u8>, Vec<u8>) {
        let k = vec![0u8; 4];
        let v = vec![0u8; 4];
        let m = Manifest {
            schema_version: SCHEMA_VERSION,
            kv_layout_version: 3,
            engine_rev: 100,
            partner: PartnerId("a".into()),
            model_fingerprint: 7,
            prefix_token_hash: 0,
            prefix_token_len: 2,
            snapshot_epoch: 42,
            num_layers: 1,
            layers: vec![LayerMeta {
                layer_index: 0,
                k_shape: vec![1, 2, 1],
                v_shape: vec![1, 2, 1],
                k_byte_len: 4,
                v_byte_len: 4,
                k_crc32: crc32fast::hash(&k),
                v_crc32: crc32fast::hash(&v),
            }],
            token_ids: vec![11, 22],
        };
        (m, k, v)
    }

    fn ok(m: &Manifest, k: &[u8], v: &[u8], req: &[i32]) -> Result<(), ValidateError> {
        KvSnapshotCodec::validate(m, &[(k, v)], 3, 100, 7, req)
    }

    #[test]
    fn accepts_consistent() {
        let (m, k, v) = good();
        assert!(ok(&m, &k, &v, &[11, 22, 33]).is_ok());
    }
    #[test]
    fn rejects_schema() {
        let (mut m, k, v) = good();
        m.schema_version = SCHEMA_VERSION + 1;
        assert_eq!(
            ok(&m, &k, &v, &[11, 22, 33]),
            Err(ValidateError::SchemaVersion)
        );
    }
    #[test]
    fn rejects_layout() {
        let (m, k, v) = good();
        assert_eq!(
            KvSnapshotCodec::validate(&m, &[(&k, &v)], 4, 100, 7, &[11, 22, 33]),
            Err(ValidateError::LayoutVersion)
        );
    }
    #[test]
    fn rejects_engine_rev() {
        let (m, k, v) = good();
        assert_eq!(
            KvSnapshotCodec::validate(&m, &[(&k, &v)], 3, 999, 7, &[11, 22, 33]),
            Err(ValidateError::EngineRev)
        );
    }
    #[test]
    fn rejects_fingerprint() {
        let (m, k, v) = good();
        assert_eq!(
            KvSnapshotCodec::validate(&m, &[(&k, &v)], 3, 100, 8, &[11, 22, 33]),
            Err(ValidateError::Fingerprint)
        );
    }
    #[test]
    fn rejects_numlayers_zero() {
        let (mut m, k, v) = good();
        m.num_layers = 0;
        assert_eq!(ok(&m, &k, &v, &[11, 22, 33]), Err(ValidateError::NumLayers));
    }
    #[test]
    fn rejects_numlayers_mismatch() {
        let (mut m, k, v) = good();
        m.num_layers = 2;
        assert_eq!(ok(&m, &k, &v, &[11, 22, 33]), Err(ValidateError::NumLayers));
    }
    #[test]
    fn rejects_len_overrun() {
        let (mut m, k, v) = good();
        m.prefix_token_len = 5;
        m.token_ids = vec![11, 22, 33, 44, 55];
        assert_eq!(ok(&m, &k, &v, &[11, 22]), Err(ValidateError::LenOverrun));
    }
    #[test]
    fn rejects_token_mismatch() {
        let (m, k, v) = good();
        assert_eq!(
            ok(&m, &k, &v, &[11, 99, 33]),
            Err(ValidateError::TokenMismatch)
        );
    }
    #[test]
    fn rejects_token_len_internal() {
        // token_ids length disagrees with prefix_token_len, but p <= request len (no overrun) →
        // its own reason, not NumLayers.
        let (mut m, k, v) = good();
        m.token_ids = vec![11]; // len 1 != prefix_token_len 2
        assert_eq!(ok(&m, &k, &v, &[11, 22, 33]), Err(ValidateError::TokenLen));
    }
    #[test]
    fn rejects_bytelen_shape() {
        let (mut m, k, v) = good();
        m.layers[0].k_byte_len = 8;
        assert_eq!(
            ok(&m, &k, &v, &[11, 22, 33]),
            Err(ValidateError::ByteLenShape)
        );
    }
    #[test]
    fn rejects_shape_seq() {
        // seq axis 9 != prefix_token_len 2; byte_len kept consistent with the wrong shape so the
        // shape-seq check (which runs first) is the one that fires, not byte_len.
        let (mut m, _k, _v) = good();
        m.layers[0].k_shape = vec![1, 9, 1];
        m.layers[0].v_shape = vec![1, 9, 1];
        let k = vec![0u8; 18];
        let v = vec![0u8; 18];
        m.layers[0].k_byte_len = 18;
        m.layers[0].v_byte_len = 18;
        m.layers[0].k_crc32 = crc32fast::hash(&k);
        m.layers[0].v_crc32 = crc32fast::hash(&v);
        assert_eq!(ok(&m, &k, &v, &[11, 22, 33]), Err(ValidateError::ShapeSeq));
    }
    #[test]
    fn rejects_crc() {
        let (m, mut k, v) = good();
        k[0] ^= 0xff;
        assert_eq!(ok(&m, &k, &v, &[11, 22, 33]), Err(ValidateError::Crc));
    }
    #[test]
    fn rejects_shape_product_overflow() {
        // Adversarial shape whose element product overflows u64 must reject (not wrap to a small
        // expectation that an undersized payload could match → consumer OOB). seq axis (index 1)
        // still equals prefix_token_len so the shape-seq check passes and byte-len is reached.
        let (mut m, k, v) = good();
        m.layers[0].k_shape = vec![1, 2, u32::MAX, u32::MAX];
        m.layers[0].k_byte_len = 4; // attacker-chosen small len matching a tiny payload
        assert_eq!(
            ok(&m, &k, &v, &[11, 22, 33]),
            Err(ValidateError::ByteLenShape)
        );
    }

    #[test]
    fn validates_asymmetric_k_v_dims() {
        // qk_head_dim=3, v_head_dim=1 → distinct K/V byte lengths (plan C3 regression).
        let k = vec![0u8; 2 * 3 * 2]; // [1,2,3] u16 = 12 bytes
        let v = vec![0u8; 4]; // [1,2,1] u16 = 4 bytes
        let mut m = good().0;
        m.layers[0].k_shape = vec![1, 2, 3];
        m.layers[0].v_shape = vec![1, 2, 1];
        m.layers[0].k_byte_len = 12;
        m.layers[0].v_byte_len = 4;
        m.layers[0].k_crc32 = crc32fast::hash(&k);
        m.layers[0].v_crc32 = crc32fast::hash(&v);
        assert!(
            KvSnapshotCodec::validate(&m, &[(&k, &v)], 3, 100, 7, &[11, 22, 33]).is_ok(),
            "K and V must validate against their own shapes"
        );
    }
}
