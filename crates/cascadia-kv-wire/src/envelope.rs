use serde::{Deserialize, Serialize};

use crate::manifest::{CacheKey, Manifest, PartnerId};

/// Head probe — learn the holder's stamped epoch + longest-common-prefix length without streaming.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Negotiate {
    pub partner: PartnerId,
    pub model_fingerprint: u64,
    pub token_ids: Vec<i32>,
}

/// Holder's reply to [`Negotiate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Offer {
    pub snapshot_epoch: u64,
    pub prefix_token_len: u32,
}

/// Asserted per-rank fetch — the holder returns `NotFound` if its current `(epoch,len)` differs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Get {
    pub partner: PartnerId,
    pub model_fingerprint: u64,
    pub expected_epoch: u64,
    pub expected_len: u32,
}

/// Entry→head warm-pull hint (issue-34 side-channel, §7). Carries only primitives so this crate
/// stays enterprise-/engine-dep-free; the enterprise maps it to/from its `KvWarmHint`
/// (`request_id` = the 16-byte UUID, `prev_chain_id` = the 32-byte `ChainId`). Sent on the existing
/// `/cascadia/state/kv/v1` stream (no second protocol) — the head peeks the first frame and routes a
/// `Hint` to its correlation stash instead of the holder serve path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WarmHint {
    pub request_id: [u8; 16],
    pub prev_chain_id: [u8; 32],
    pub partner: String,
}

/// Closed-enum replica-push outcome (design §12.2). A NACK is a rejected variant, not a separate
/// frame. `RejectedRankCollision` is retired: the store keys by `(CacheKey, rank)`, so distinct
/// ranks on one node are not a collision (§11.17.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicateOutcome {
    Accepted,
    RejectedGuard,
    RejectedTooLarge,
    RejectedBudget,
}

/// Control envelope on `/cascadia/state/kv/v1`.
///
/// Commit is **in-band** (the warm/cold decision rides the pipeline dispatch, §7) — there is no
/// `COMMIT`/`Ack` message. `Found` is followed on the stream by per-layer length-prefixed K then V
/// payloads (framed by the transport, not this enum). `Hint` is **appended last** so existing variant
/// indices (and the wire-conformance goldens) are unchanged — an additive bump.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvMessage {
    Negotiate(Negotiate),
    Offer(Offer),
    Get(Get),
    Found(Manifest),
    NotFound,
    Error(String),
    Hint(WarmHint),
    /// Head→rank-N: pull your rank-N slice for `epoch` from `prev_chain_id` and arm a warm-resume.
    /// `prefix_token_len` lets the token-less rank assert a `(epoch, len)` GET without a NEGOTIATE.
    WarmResumeTrigger {
        epoch: u64,
        prefix_token_len: u32,
        model_fingerprint: u64,
        prev_chain_id: [u8; 32],
        rank: u16,
    },
    /// Rank-N→head: warm-resume armed (`ok=true`) or failed (`ok=false`) for `epoch`.
    WarmResumeConfirm {
        epoch: u64,
        ok: bool,
    },
    /// Head→rank-N: COMMIT the staged warm-resume for `epoch` — apply it to the engine.
    ///
    /// Two-phase on purpose. The trigger only STAGES the pulled slice (it lands in the engine's
    /// capture cache and touches no engine state); the head sends this only once every rank has
    /// confirmed. Applying inside the trigger made the "all-or-nothing" verdict a lie: a rank that
    /// applied and then lost/late-delivered its Confirm stayed warm while the head went cold, and its
    /// stale arm corrupted the head's cold reprefill or the next request entirely. Staging is
    /// side-effect-free, so a rank that is never committed simply ages out of the capture cache.
    WarmResumeCommit {
        epoch: u64,
    },
    /// Head→rank-N: drop the staged/armed warm-resume for `epoch`, go cold.
    WarmResumeAbort {
        epoch: u64,
    },
    /// Head→rank capture-time push trigger (design §12.2). Carries `partner` because the token-less
    /// rank needs it for the H.1-confined `export`, the `CacheKey`, and HRW placement — a
    /// partner-less trigger would file pushes under `partner_hash("")` while pullers derive keys
    /// from the real partner: every pull misses, every metric green.
    ReplicatePush {
        partner: PartnerId,
        epoch: u64,
        prefix_token_len: u32,
        model_fingerprint: u64,
        rank: u16,
    },
    /// Rank→counterpart replica push. `blob` is the per-layer K/V payloads of one export,
    /// concatenated in manifest layer order (k then v per layer); the receiver re-slices by the
    /// manifest's `k_byte_len`/`v_byte_len`. `tokens` is on the wire for supersede-if-extends.
    Replicate {
        key: CacheKey,
        rank: u16,
        manifest: Manifest,
        tokens: Vec<i32>,
        blob: Vec<u8>,
    },
    /// Counterpart→rank ack/NACK. The sender NEVER blocks on it (D8) — it updates a metric and
    /// nothing else. A missing ack is `push_dial_fail`, which is why the two are separable at all.
    ReplicateAck {
        key: CacheKey,
        rank: u16,
        outcome: ReplicateOutcome,
    },
    /// Asserted replica fetch. `rank` on the wire is what makes serving the wrong rank
    /// unrepresentable (design §12.2): every rank of a session shares one `CacheKey` and one
    /// `(epoch, len)`, so a rank-less fetch that lands on a neighbour would serve that neighbour's
    /// rank with all guards passing.
    ReplicaGet {
        key: CacheKey,
        rank: u16,
        expected_epoch: u64,
        expected_len: u32,
    },
    /// Head→rank: like [`KvMessage::WarmResumeTrigger`] but delivers `partner` too (design §12.2,
    /// H.1). `WarmResumeTrigger` predates tenant-namespacing and has no `partner` field; a
    /// token-less rank needs one to construct the `CacheKey` for its H.1-confined export, so this
    /// variant carries it rather than overloading the original.
    WarmResumeTriggerV2 {
        partner: PartnerId,
        epoch: u64,
        prefix_token_len: u32,
        model_fingerprint: u64,
        prev_chain_id: [u8; 32],
        rank: u16,
    },
    /// Entry→head per-request tenant carrier (issue-34 H.1b). [`KvMessage::Hint`] is the only frame
    /// that tells a serving head which tenant a request belongs to, and it rides ONLY forced moves —
    /// so a session's first turn is captured under `LOCAL_NS` (`""`) while the first move asserts a
    /// real tenant, NEGOTIATE misses, and the warm pull can never hit on that move. This variant
    /// carries the same `(request_id, partner)` pair on every request so the head can key the
    /// capture on the real tenant from turn one.
    ///
    /// Field types deliberately mirror [`WarmHint`]'s (`request_id` = the 16-byte UUID, `partner` =
    /// the raw tenant string) so the enterprise's existing request-id→tenant stash takes this frame
    /// unchanged. `partner` is bounded by [`MAX_PARTNER_LEN`] at the frame boundary.
    TenantHint {
        request_id: [u8; 16],
        partner: String,
    },
}

/// Cap on the UTF-8 byte length of [`KvMessage::TenantHint`]'s `partner`, enforced by
/// [`crate::encode_frame`]/[`crate::decode_frame`] so an over-long tenant is a rejected frame rather
/// than a partially applied one. Real tenant ids are PASETO `sub` claims or `"default"` — tens of
/// bytes — so 256 is far above any legitimate value while keeping a forged frame from making a head
/// allocate, log, or cache-key on an attacker-sized string. Deliberately NOT applied to the older
/// string-bearing variants: they predate the cap and tightening them would change which existing
/// frames a head accepts.
pub const MAX_PARTNER_LEN: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::config::standard;

    fn key() -> CacheKey {
        CacheKey {
            partner_hash: 1,
            model_fingerprint: 7,
            prefix_token_hash: 9,
        }
    }

    // Mirrors tests/conformance.rs::fixed_manifest — not shared across the unit/integration test
    // boundary, so duplicated here for the round-trip.
    fn manifest() -> Manifest {
        Manifest {
            schema_version: crate::manifest::SCHEMA_VERSION,
            kv_layout_version: crate::manifest::KV_LAYOUT_VERSION,
            engine_rev: 0x0000_0000_dead_beef,
            partner: PartnerId("conformance".into()),
            model_fingerprint: 0x0000_0000_0000_1234,
            prefix_token_hash: 0x0000_0000_0000_5678,
            prefix_token_len: 2,
            snapshot_epoch: 0xABCD_0000_0000_0001,
            num_layers: 1,
            layers: vec![crate::manifest::LayerMeta {
                layer_index: 0,
                k_shape: vec![1, 2, 3],
                v_shape: vec![1, 2, 1],
                k_byte_len: 12,
                v_byte_len: 4,
                k_crc32: 0xAAAA_AAAA,
                v_crc32: 0xBBBB_BBBB,
            }],
            token_ids: vec![11, 22],
        }
    }

    #[test]
    fn envelope_roundtrips_bincode() {
        let msgs = vec![
            KvMessage::Negotiate(Negotiate {
                partner: PartnerId("a".into()),
                model_fingerprint: 7,
                token_ids: vec![1, 2, 3],
            }),
            KvMessage::Offer(Offer {
                snapshot_epoch: 42,
                prefix_token_len: 2,
            }),
            KvMessage::Get(Get {
                partner: PartnerId("a".into()),
                model_fingerprint: 7,
                expected_epoch: 42,
                expected_len: 2,
            }),
            KvMessage::NotFound,
            KvMessage::Error("x".into()),
            KvMessage::Hint(WarmHint {
                request_id: [3u8; 16],
                prev_chain_id: [9u8; 32],
                partner: "acme".into(),
            }),
            KvMessage::WarmResumeTrigger {
                epoch: 42,
                prefix_token_len: 2,
                model_fingerprint: 7,
                prev_chain_id: [9u8; 32],
                rank: 1,
            },
            KvMessage::WarmResumeConfirm {
                epoch: 42,
                ok: true,
            },
            KvMessage::WarmResumeCommit { epoch: 41 },
            KvMessage::WarmResumeAbort { epoch: 42 },
            KvMessage::ReplicatePush {
                partner: PartnerId("acme".into()),
                epoch: 42,
                prefix_token_len: 3,
                model_fingerprint: 7,
                rank: 1,
            },
            KvMessage::Replicate {
                key: key(),
                rank: 1,
                manifest: manifest(),
                tokens: vec![11, 22, 33],
                blob: vec![5u8; 8],
            },
            KvMessage::ReplicateAck {
                key: key(),
                rank: 1,
                outcome: ReplicateOutcome::RejectedGuard,
            },
            KvMessage::ReplicaGet {
                key: key(),
                rank: 1,
                expected_epoch: 42,
                expected_len: 3,
            },
            KvMessage::WarmResumeTriggerV2 {
                partner: PartnerId("acme".into()),
                epoch: 42,
                prefix_token_len: 3,
                model_fingerprint: 7,
                prev_chain_id: [9u8; 32],
                rank: 1,
            },
            KvMessage::TenantHint {
                request_id: [3u8; 16],
                partner: "acme".into(),
            },
        ];
        for m in msgs {
            let bytes = bincode::serde::encode_to_vec(&m, standard()).unwrap();
            let (back, _len): (KvMessage, usize) =
                bincode::serde::decode_from_slice(&bytes, standard()).unwrap();
            assert_eq!(m, back);
        }
    }

    #[test]
    fn new_variants_are_appended_after_warm_resume_abort() {
        // WarmResumeAbort was the last pre-replication variant; its encoded tag byte must not change.
        let old =
            bincode::serde::encode_to_vec(KvMessage::WarmResumeAbort { epoch: 1 }, standard())
                .unwrap();
        assert_eq!(
            old[0], 10,
            "WarmResumeAbort is variant index 10; appending must not shift it"
        );
        let get = bincode::serde::encode_to_vec(
            KvMessage::ReplicaGet {
                key: key(),
                rank: 0,
                expected_epoch: 0,
                expected_len: 0,
            },
            standard(),
        )
        .unwrap();
        assert_eq!(get[0], 14, "ReplicaGet is appended 4th, index 14");
        let v2 = bincode::serde::encode_to_vec(
            KvMessage::WarmResumeTriggerV2 {
                partner: PartnerId("acme".into()),
                epoch: 1,
                prefix_token_len: 1,
                model_fingerprint: 1,
                prev_chain_id: [0u8; 32],
                rank: 0,
            },
            standard(),
        )
        .unwrap();
        assert_eq!(v2[0], 15, "WarmResumeTriggerV2 is appended 5th, index 15");
        let tenant = bincode::serde::encode_to_vec(
            KvMessage::TenantHint {
                request_id: [0u8; 16],
                partner: "acme".into(),
            },
            standard(),
        )
        .unwrap();
        assert_eq!(
            tenant[0], 16,
            "TenantHint is the LAST appended variant (index 16)"
        );
    }
}
