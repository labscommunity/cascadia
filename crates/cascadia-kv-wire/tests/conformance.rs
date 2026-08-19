//! Wire-conformance golden fixtures (§15 `wire_conformance_fixture`) — the cross-repo drift
//! backstop. The same `.golden.bin` files are copied into the enterprise repo and asserted there
//! against the pinned `cascadia-kv-wire`, so a wire-format change fails in whichever repo is stale.
//!
//! Bootstrap: the first run writes any missing golden and fails; re-run to assert. NEVER regenerate
//! a golden to turn a red test green unless the wire format change is intended.

use bincode::config::standard;
use cascadia_kv_wire::*;
use std::path::Path;

fn fixed_manifest() -> Manifest {
    Manifest {
        schema_version: SCHEMA_VERSION,
        kv_layout_version: KV_LAYOUT_VERSION,
        engine_rev: 0x0000_0000_dead_beef,
        partner: PartnerId("conformance".into()),
        model_fingerprint: 0x0000_0000_0000_1234,
        prefix_token_hash: 0x0000_0000_0000_5678,
        prefix_token_len: 2,
        snapshot_epoch: 0xABCD_0000_0000_0001,
        num_layers: 1,
        layers: vec![LayerMeta {
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

fn fixed_key() -> CacheKey {
    CacheKey {
        partner_hash: 1,
        model_fingerprint: 7,
        prefix_token_hash: 9,
    }
}

fn check_golden(name: &str, bytes: &[u8]) {
    let path = format!(
        "{}/tests/fixtures/{name}.golden.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    if !Path::new(&path).exists() {
        std::fs::write(&path, bytes).unwrap();
        eprintln!("bootstrapped golden '{name}' (commit it as the contract)");
    }
    let golden = std::fs::read(&path).unwrap();
    assert_eq!(
        bytes,
        golden.as_slice(),
        "wire drift on '{name}': regenerate fixtures ONLY for an intended format change"
    );
}

#[test]
fn manifest_matches_golden() {
    let bytes = bincode::serde::encode_to_vec(fixed_manifest(), standard()).unwrap();
    check_golden("manifest", &bytes);
}

#[test]
fn envelope_variants_match_golden() {
    let variants: Vec<(&str, KvMessage)> = vec![
        (
            "env_negotiate",
            KvMessage::Negotiate(Negotiate {
                partner: PartnerId("c".into()),
                model_fingerprint: 7,
                token_ids: vec![1, 2],
            }),
        ),
        (
            "env_offer",
            KvMessage::Offer(Offer {
                snapshot_epoch: 9,
                prefix_token_len: 2,
            }),
        ),
        (
            "env_get",
            KvMessage::Get(Get {
                partner: PartnerId("c".into()),
                model_fingerprint: 7,
                expected_epoch: 9,
                expected_len: 2,
            }),
        ),
        ("env_notfound", KvMessage::NotFound),
        ("env_error", KvMessage::Error("e".into())),
        ("env_found", KvMessage::Found(fixed_manifest())),
        (
            "env_hint",
            KvMessage::Hint(WarmHint {
                request_id: [3u8; 16],
                prev_chain_id: [9u8; 32],
                partner: "c".into(),
            }),
        ),
        (
            "env_replicate_push",
            KvMessage::ReplicatePush {
                partner: PartnerId("c".into()),
                epoch: 9,
                prefix_token_len: 2,
                model_fingerprint: 7,
                rank: 1,
            },
        ),
        (
            "env_replicate",
            KvMessage::Replicate {
                key: CacheKey {
                    partner_hash: 1,
                    model_fingerprint: 7,
                    prefix_token_hash: 9,
                },
                rank: 1,
                manifest: fixed_manifest(),
                tokens: vec![11, 22],
                blob: vec![5u8; 4],
            },
        ),
        (
            "env_replicate_ack",
            KvMessage::ReplicateAck {
                key: CacheKey {
                    partner_hash: 1,
                    model_fingerprint: 7,
                    prefix_token_hash: 9,
                },
                rank: 1,
                outcome: ReplicateOutcome::Accepted,
            },
        ),
        (
            "env_replica_get",
            KvMessage::ReplicaGet {
                key: CacheKey {
                    partner_hash: 1,
                    model_fingerprint: 7,
                    prefix_token_hash: 9,
                },
                rank: 1,
                expected_epoch: 9,
                expected_len: 2,
            },
        ),
        (
            "env_warm_resume_trigger_v2",
            KvMessage::WarmResumeTriggerV2 {
                partner: PartnerId("c".into()),
                epoch: 9,
                prefix_token_len: 2,
                model_fingerprint: 7,
                prev_chain_id: [9u8; 32],
                rank: 1,
            },
        ),
        (
            "env_tenant_hint",
            KvMessage::TenantHint {
                request_id: [3u8; 16],
                partner: "c".into(),
            },
        ),
        // Rank-bound fetch. Appended last and shipped without a golden — its TAG was pinned by
        // `declared_index`, but nothing pinned its FIELD layout, so reordering `rank` against
        // `expected_len` would have crossed the repo boundary undetected.
        (
            "env_get_v2",
            KvMessage::GetV2 {
                partner: PartnerId("c".into()),
                model_fingerprint: 7,
                expected_epoch: 9,
                expected_len: 2,
                rank: 1,
            },
        ),
        // Candidate-carrying trigger. Goldened AT introduction (unlike GetV2): two entries of
        // DIFFERENT kinds so both the field order and the kind tag encoding are pinned.
        (
            "env_warm_resume_trigger_v3",
            KvMessage::WarmResumeTriggerV3 {
                partner: PartnerId("c".into()),
                epoch: 9,
                prefix_token_len: 2,
                model_fingerprint: 7,
                prev_chain_id: [9u8; 32],
                rank: 1,
                candidates: vec![
                    ([1u8; 32], CandidateKind::Primary),
                    ([2u8; 32], CandidateKind::Replica),
                ],
            },
        ),
        // D4 move identity (design §B.1): nonce-carrying trigger/abort/commit. Goldened at
        // introduction, same reasoning as V3 — pin field order and the nonce array encoding.
        (
            "env_warm_resume_trigger_v4",
            KvMessage::WarmResumeTriggerV4 {
                partner: PartnerId("c".into()),
                epoch: 9,
                prefix_token_len: 2,
                model_fingerprint: 7,
                prev_chain_id: [9u8; 32],
                rank: 1,
                candidates: vec![
                    ([1u8; 32], CandidateKind::Primary),
                    ([2u8; 32], CandidateKind::Replica),
                ],
                move_nonce: [4u8; 16],
            },
        ),
        (
            "env_warm_resume_abort_v2",
            KvMessage::WarmResumeAbortV2 {
                epoch: 9,
                move_nonce: [4u8; 16],
            },
        ),
        (
            "env_warm_resume_commit_v2",
            KvMessage::WarmResumeCommitV2 {
                epoch: 9,
                move_nonce: [4u8; 16],
            },
        ),
    ];
    for (name, msg) in variants {
        let bytes = bincode::serde::encode_to_vec(&msg, standard()).unwrap();
        check_golden(name, &bytes);
    }
}

/// The wire index every variant MUST encode as. Exhaustive on purpose — no `_` arm — so appending a
/// variant without pinning its index fails to compile here rather than shipping unpinned.
fn declared_index(msg: &KvMessage) -> u8 {
    match msg {
        KvMessage::Negotiate(_) => 0,
        KvMessage::Offer(_) => 1,
        KvMessage::Get(_) => 2,
        KvMessage::Found(_) => 3,
        KvMessage::NotFound => 4,
        KvMessage::Error(_) => 5,
        KvMessage::Hint(_) => 6,
        KvMessage::WarmResumeTrigger { .. } => 7,
        KvMessage::WarmResumeConfirm { .. } => 8,
        KvMessage::WarmResumeCommit { .. } => 9,
        KvMessage::WarmResumeAbort { .. } => 10,
        KvMessage::ReplicatePush { .. } => 11,
        KvMessage::Replicate { .. } => 12,
        KvMessage::ReplicateAck { .. } => 13,
        KvMessage::ReplicaGet { .. } => 14,
        KvMessage::WarmResumeTriggerV2 { .. } => 15,
        KvMessage::TenantHint { .. } => 16,
        KvMessage::GetV2 { .. } => 17,
        KvMessage::WarmResumeTriggerV3 { .. } => 18,
        KvMessage::WarmResumeTriggerV4 { .. } => 19,
        KvMessage::WarmResumeAbortV2 { .. } => 20,
        KvMessage::WarmResumeCommitV2 { .. } => 21,
    }
}

/// bincode derives the tag from DECLARATION ORDER, so inserting a variant anywhere but the end
/// renumbers every later one — silently, because each build still round-trips against itself. The
/// per-variant goldens only cover the variants that have one; this pins the tag of ALL of them, in
/// contiguous order, so the insert itself is what goes red instead of a mismatched-build pair
/// mis-decoding a live frame.
#[test]
fn every_variant_index_is_pinned() {
    let all: Vec<KvMessage> = vec![
        KvMessage::Negotiate(Negotiate {
            partner: PartnerId("c".into()),
            model_fingerprint: 7,
            token_ids: vec![1],
        }),
        KvMessage::Offer(Offer {
            snapshot_epoch: 9,
            prefix_token_len: 2,
        }),
        KvMessage::Get(Get {
            partner: PartnerId("c".into()),
            model_fingerprint: 7,
            expected_epoch: 9,
            expected_len: 2,
        }),
        KvMessage::Found(fixed_manifest()),
        KvMessage::NotFound,
        KvMessage::Error("e".into()),
        KvMessage::Hint(WarmHint {
            request_id: [3u8; 16],
            prev_chain_id: [9u8; 32],
            partner: "c".into(),
        }),
        KvMessage::WarmResumeTrigger {
            epoch: 9,
            prefix_token_len: 2,
            model_fingerprint: 7,
            prev_chain_id: [9u8; 32],
            rank: 1,
        },
        KvMessage::WarmResumeConfirm { epoch: 9, ok: true },
        KvMessage::WarmResumeCommit { epoch: 9 },
        KvMessage::WarmResumeAbort { epoch: 9 },
        KvMessage::ReplicatePush {
            partner: PartnerId("c".into()),
            epoch: 9,
            prefix_token_len: 2,
            model_fingerprint: 7,
            rank: 1,
        },
        KvMessage::Replicate {
            key: fixed_key(),
            rank: 1,
            manifest: fixed_manifest(),
            tokens: vec![11, 22],
            blob: vec![5u8; 4],
        },
        KvMessage::ReplicateAck {
            key: fixed_key(),
            rank: 1,
            outcome: ReplicateOutcome::Accepted,
        },
        KvMessage::ReplicaGet {
            key: fixed_key(),
            rank: 1,
            expected_epoch: 9,
            expected_len: 2,
        },
        KvMessage::WarmResumeTriggerV2 {
            partner: PartnerId("c".into()),
            epoch: 9,
            prefix_token_len: 2,
            model_fingerprint: 7,
            prev_chain_id: [9u8; 32],
            rank: 1,
        },
        KvMessage::TenantHint {
            request_id: [3u8; 16],
            partner: "c".into(),
        },
        KvMessage::GetV2 {
            partner: PartnerId("c".into()),
            model_fingerprint: 7,
            expected_epoch: 9,
            expected_len: 2,
            rank: 1,
        },
        KvMessage::WarmResumeTriggerV3 {
            partner: PartnerId("c".into()),
            epoch: 9,
            prefix_token_len: 2,
            model_fingerprint: 7,
            prev_chain_id: [9u8; 32],
            rank: 1,
            candidates: vec![([1u8; 32], CandidateKind::Primary)],
        },
        KvMessage::WarmResumeTriggerV4 {
            partner: PartnerId("c".into()),
            epoch: 9,
            prefix_token_len: 2,
            model_fingerprint: 7,
            prev_chain_id: [9u8; 32],
            rank: 1,
            candidates: vec![([1u8; 32], CandidateKind::Primary)],
            move_nonce: [4u8; 16],
        },
        KvMessage::WarmResumeAbortV2 {
            epoch: 9,
            move_nonce: [4u8; 16],
        },
        KvMessage::WarmResumeCommitV2 {
            epoch: 9,
            move_nonce: [4u8; 16],
        },
    ];
    assert_eq!(all.len(), 22, "every declared variant must be listed here");
    for (position, msg) in all.iter().enumerate() {
        let tag = bincode::serde::encode_to_vec(msg, standard()).unwrap()[0];
        assert_eq!(
            usize::from(tag),
            position,
            "{msg:?} must encode as tag {position}"
        );
        assert_eq!(
            tag,
            declared_index(msg),
            "{msg:?} drifted from its pinned index"
        );
    }
}
