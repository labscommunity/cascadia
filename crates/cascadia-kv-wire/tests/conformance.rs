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
    ];
    for (name, msg) in variants {
        let bytes = bincode::serde::encode_to_vec(&msg, standard()).unwrap();
        check_golden(name, &bytes);
    }
}
