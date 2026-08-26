//! Cross-holder wire-contract conformance suite.
//!
//! [`SparseMoeKvHolder`] and [`OvMoeKvHolder`] stand behind the same framed NEGOTIATE/GET plane
//! via [`cascadia_engine::KvSnapshotHolder`], and the issue-34 H.1a cross-tenant read existed
//! identically in both — yet each file's tests only ever drove its own holder. This module states
//! each clause of the wire contract ONCE as a generic driver over the trait, and instantiates
//! every clause for BOTH holders through the real framed harness
//! ([`super::wire_tests::serve_one`], generic over `dyn KvSnapshotHolder`).
//!
//! Per-holder code is confined to a [`ContractFixture`]: construction, the consumer-side codec
//! constants, the independently re-derived byte digest, and the holder's own consumer decode.
//! Everything else — transport, requests, assertions — is shared.
//!
//! Fidelity note: both holders run the FULL wire path here (frames, serve loop, codec
//! validation). The one asymmetry is the consumer re-insert cycle
//! (`a_pulled_slice_re_serves_byte_identically`), which stays SparseMoe-only in `wire_tests` —
//! OvMoe's pulled-slice ingestion (`ov_drain_handoff`) needs compiled shells, so its round trip
//! here stops at decode + digest, the fidelity its own tests achieve.
//!
//! Lives in `src/` for the same reason `wire_tests` does: both holders are `pub(crate)`.

use super::wire_tests::{arrived_digest, ask, fnv1a, RANK};
use super::{synth_epoch, SparseHolderState, SparseMoeKvHolder};
use crate::kv_prefix_cache::{KvSnapshot, LayerKvSlice, ModelFingerprint};
use crate::ov_kv_cache::{OvLayerKvSlice, OvMoeKvSnapshot};
use crate::ov_kv_coordination::{OvHolderState, OvMoeKvHolder};
use cascadia_engine::KvSnapshotHolder;
use cascadia_kv_wire::{
    KvMessage, KvSnapshotCodec, Manifest, Negotiate, PartnerId, KV_LAYOUT_VERSION, OPAQUE_KV_LAYOUT,
};
use std::sync::{Arc, Mutex};
use tokio::io::DuplexStream;

/// The prompt every clause negotiates. The seeded prefix is `TOKENS[..3]` — one shorter, because
/// `lookup_ns` refuses a prefix that covers the whole prompt.
const TOKENS: &[i32] = &[11, 22, 33, 44];
const PREFIX_LEN: u32 = 3;
const OWNER: &str = "tenant-a";
const FOREIGN: &str = "tenant-b";

/// The only per-holder code in the suite. Everything a clause needs that the
/// [`KvSnapshotHolder`] trait itself does not carry.
trait ContractFixture {
    type Holder: KvSnapshotHolder + Send + Sync + 'static;
    /// Consumer-side codec expectations for this holder's manifests.
    const LAYOUT: u16;
    const ENGINE_REV: u64;
    fn model_fp() -> u64;
    /// Holder whose mirrored prefix cache holds `partner`'s `TOKENS[..3]` snapshot — the state a
    /// plane consumer-insert leaves behind.
    fn with_prefix(partner: &str) -> Arc<Self::Holder>;
    /// Holder with ONLY a capture stashed under `tenant` ("" = untagged/legacy) — the rank>0
    /// shape after a CAPTURE frame, no offer. Returns the capture's servable epoch.
    fn with_capture(tenant: &str) -> (Arc<Self::Holder>, u64);
    /// What the seeded snapshot MUST serialise to, re-derived from the fixture values — never
    /// from the production encoder, so an encoding change cannot cancel itself out.
    fn expected_digest() -> u64;
    /// The holder's real consumer decode over a served slice; `Some(past_seq_len)`.
    fn decoded_past_len(m: &Manifest, payloads: &[(Vec<u8>, Vec<u8>)]) -> Option<u32>;
}

// ---- shared requests / consumer half --------------------------------------------------------

fn negotiate(fp: u64, partner: &str) -> KvMessage {
    KvMessage::Negotiate(Negotiate {
        partner: PartnerId(partner.into()),
        model_fingerprint: fp,
        token_ids: TOKENS.to_vec(),
    })
}

fn get_v2(fp: u64, partner: &str, epoch: u64, len: u32, rank: u16) -> KvMessage {
    KvMessage::GetV2 {
        partner: PartnerId(partner.into()),
        model_fingerprint: fp,
        expected_epoch: epoch,
        expected_len: len,
        rank,
    }
}

/// NEGOTIATE as `partner` and return the offered `(epoch, len)`.
async fn negotiated<F: ContractFixture>(holder: &Arc<F::Holder>, partner: &str) -> (u64, u32) {
    match ask(holder, negotiate(F::model_fp(), partner)).await.1 {
        KvMessage::Offer(o) => (o.snapshot_epoch, o.prefix_token_len),
        other => panic!("expected Offer, got {other:?}"),
    }
}

/// Consumer half of a served GET: read the blobs that follow `Found` and run the real structural
/// validation over them, exactly as the puller does before it inserts.
async fn receive<F: ContractFixture>(
    client: &mut DuplexStream,
    m: &Manifest,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut payloads = Vec::new();
    for _ in &m.layers {
        let k = super::wire_tests::read_blob(client).await.expect("k blob");
        let v = super::wire_tests::read_blob(client).await.expect("v blob");
        payloads.push((k, v));
    }
    let refs: Vec<(&[u8], &[u8])> = payloads
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvSnapshotCodec::validate(m, &refs, F::LAYOUT, F::ENGINE_REV, F::model_fp(), TOKENS)
        .expect("what arrived must pass consumer validation");
    payloads
}

// ---- the contract, one generic driver per clause --------------------------------------------

/// (a) The round trip a warm pull makes: NEGOTIATE, the rank-bound GET, the bytes. Asserted on
/// the DIGEST of what arrived and on the holder's own consumer decode accepting it.
async fn negotiate_then_get_delivers_the_offered_bytes<F: ContractFixture>() {
    let holder = F::with_prefix(OWNER);
    let (epoch, len) = negotiated::<F>(&holder, OWNER).await;
    assert_eq!(len, PREFIX_LEN, "the cached prefix, not the whole prompt");

    let (mut client, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch, len, RANK)).await;
    let KvMessage::Found(m) = reply else {
        panic!("expected Found, got {reply:?}")
    };
    let payloads = receive::<F>(&mut client, &m).await;

    assert_eq!(m.snapshot_epoch, epoch, "served under the negotiated epoch");
    assert_eq!(m.prefix_token_len, len);
    assert_eq!(m.token_ids, TOKENS[..len as usize]);
    assert_eq!(m.partner.0, OWNER, "stamped with the asserting tenant");
    assert_eq!(
        arrived_digest(&payloads),
        F::expected_digest(),
        "the bytes off the wire are the snapshot the holder was seeded with"
    );
    assert_eq!(
        F::decoded_past_len(&m, &payloads),
        Some(PREFIX_LEN),
        "the holder's own consumer decode must accept what it served"
    );
}

/// (b) Partner confinement (issue-34 H.1a): a cross-tenant NEGOTIATE reads as an empty cache, and
/// the side door — a foreign GET asserting the owner's derivable (epoch, len) without negotiating
/// — is refused too. The owner is then still served, or both refusals hold vacuously.
async fn another_tenants_negotiate_and_get_are_refused<F: ContractFixture>() {
    let holder = F::with_prefix(OWNER);

    // The oracle probe: a foreign guess must never read a truncated length.
    let (_c, reply) = ask(&holder, negotiate(F::model_fp(), FOREIGN)).await;
    assert_eq!(reply, KvMessage::NotFound, "{FOREIGN} negotiated a hit");

    // The side door: FOREIGN asserts the owner's own (epoch, len) — obtainable by anyone who
    // guesses the prefix, `synth_epoch` is pure.
    let (epoch, len) = negotiated::<F>(&holder, OWNER).await;
    let (_c, reply) = ask(&holder, get_v2(F::model_fp(), FOREIGN, epoch, len, RANK)).await;
    assert_eq!(
        reply,
        KvMessage::NotFound,
        "{FOREIGN} must not reach the owner's stashed offer"
    );

    let (mut client, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch, len, RANK)).await;
    let KvMessage::Found(m) = reply else {
        panic!("owner refused too — the confinement above proves nothing")
    };
    assert_eq!(
        arrived_digest(&receive::<F>(&mut client, &m).await),
        F::expected_digest()
    );
}

/// (c1) A GET asserting a drifted epoch is refused, not answered with whatever the holder holds.
async fn a_drifted_epoch_is_refused_not_served_stale<F: ContractFixture>() {
    let holder = F::with_prefix(OWNER);
    let (epoch, len) = negotiated::<F>(&holder, OWNER).await;
    let (_c, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch ^ 1, len, RANK)).await;
    assert_eq!(reply, KvMessage::NotFound, "a foreign epoch was served");

    // The honest assertion still serves — the refusal above is the drift, not a dead holder.
    let (epoch, len) = negotiated::<F>(&holder, OWNER).await;
    let (_c, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch, len, RANK)).await;
    assert!(matches!(reply, KvMessage::Found(_)));
}

/// (c2) A GET asserting a drifted length is refused. Each case re-negotiates first: a mis-asserted
/// GET consumes the offer, so without that the later cases would pass for the wrong reason.
async fn a_drifted_len_is_refused_not_served_stale<F: ContractFixture>() {
    let holder = F::with_prefix(OWNER);
    for (label, shift) in [("longer than offered", 1i64), ("shorter than offered", -1)] {
        let (epoch, len) = negotiated::<F>(&holder, OWNER).await;
        let asked = (i64::from(len) + shift) as u32;
        let (_c, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch, asked, RANK)).await;
        assert_eq!(reply, KvMessage::NotFound, "{label} was served");
    }
    let (epoch, len) = negotiated::<F>(&holder, OWNER).await;
    let (_c, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch, len, RANK)).await;
    assert!(matches!(reply, KvMessage::Found(_)));
}

/// (d) A GET for an epoch that was never offered is a clean `NotFound` on the wire, and `None`
/// straight from the export seam — on the offer path AND on the capture fallback, which is
/// epoch-keyed too (a stashed capture must not answer for an epoch it was not minted under).
async fn an_unoffered_epoch_is_not_found<F: ContractFixture>() {
    let unoffered = 0xDEAD_BEEF;
    let holder = F::with_prefix(OWNER);
    let (_c, reply) = ask(
        &holder,
        get_v2(F::model_fp(), OWNER, unoffered, PREFIX_LEN, RANK),
    )
    .await;
    assert_eq!(reply, KvMessage::NotFound);
    assert!(
        holder.export(OWNER, unoffered, PREFIX_LEN).is_none(),
        "the refusal must come from the holder seam itself, not the serve loop"
    );

    let (capture_holder, epoch) = F::with_capture(OWNER);
    assert_ne!(epoch, unoffered, "fixture epochs must not collide");
    let (_c, reply) = ask(
        &capture_holder,
        get_v2(F::model_fp(), OWNER, unoffered, PREFIX_LEN, RANK),
    )
    .await;
    assert_eq!(
        reply,
        KvMessage::NotFound,
        "a capture stashed under another epoch answered an unknown one"
    );
}

/// (e) Defect 1b, donor half: `GetV2` carries the rank so a node holding shard M cannot answer a
/// rank-N fetch with M's bytes. The refusal lands BEFORE the export, which is observable here:
/// the offer survives it.
async fn a_get_v2_for_a_foreign_rank_is_refused_before_the_export<F: ContractFixture>() {
    let holder = F::with_prefix(OWNER);
    let (epoch, len) = negotiated::<F>(&holder, OWNER).await;

    let (_c, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch, len, RANK + 1)).await;
    assert_eq!(reply, KvMessage::NotFound, "a foreign rank was served");

    let (_c, reply) = ask(&holder, get_v2(F::model_fp(), OWNER, epoch, len, RANK)).await;
    assert!(
        matches!(reply, KvMessage::Found(_)),
        "the refused fetch consumed the offer — it must be refused before the export"
    );
}

/// (f1) H.1a CLOSED (inverted from the residual pin): the CaptureV2 frame carries the head's turn
/// tenant, the stash entry is tagged, and a TAGGED capture is confined to its tenant over the
/// wire — the owner is served, a foreign tenant gets `NotFound`.
///
/// The fallback itself stays load-bearing: only the HEAD negotiates, so rank>0 has no offer and
/// serves from exactly this stash. Deleting it (bfae9ffe, reverted) took sparse-moe plane0 from
/// 7/7 to 5/7 and plane1 from 10/10 to 6/10 — every rank>0 donor answered `get_none`.
async fn a_tagged_capture_is_confined_to_its_tenant<F: ContractFixture>() {
    let (holder, epoch) = F::with_capture(OWNER);

    // The owner is served — this is the legitimate rank>0 path the cert depends on.
    let (_c, reply) = ask(
        &holder,
        get_v2(F::model_fp(), OWNER, epoch, PREFIX_LEN, RANK),
    )
    .await;
    assert!(
        matches!(reply, KvMessage::Found(_)),
        "rank>0 must serve its own tenant from the capture stash; refusing it breaks tail warm-resume"
    );
    // A foreign tenant deriving the epoch is refused — the H.1a close.
    let (_c, reply) = ask(
        &holder,
        get_v2(F::model_fp(), FOREIGN, epoch, PREFIX_LEN, RANK),
    )
    .await;
    assert!(
        matches!(reply, KvMessage::NotFound),
        "a TAGGED capture served a foreign tenant — the H.1a cross-tenant read is open again"
    );
}

/// (f2) The migration residual, pinned explicitly: an UNTAGGED capture ("" — every legacy v1
/// CAPTURE frame, and a tenant-less turn via the `kv_tenant_hint_missing` expiry path) is still
/// served to ANY partner. Refusing it is bfae9ffe one frame later — an old-build head's chain
/// would have every rank>0 donor answer `get_none`. Retire only with the legacy Capture frame.
async fn an_untagged_capture_still_serves_any_partner<F: ContractFixture>() {
    let (holder, epoch) = F::with_capture("");
    for partner in [OWNER, FOREIGN] {
        let (_c, reply) = ask(
            &holder,
            get_v2(F::model_fp(), partner, epoch, PREFIX_LEN, RANK),
        )
        .await;
        assert!(
            matches!(reply, KvMessage::Found(_)),
            "an untagged (legacy) capture must serve `{partner}` — refusing it starves rank>0 \
             donors on mixed-version chains (the bfae9ffe failure class)"
        );
    }
}

// ---- SparseMoe fixture ----------------------------------------------------------------------

// num_heads=1, len=3, qk_head_dim=2, v_head_dim=1 ⇒ 6 u16 of K and 3 u16 of V per layer. Every
// value distinct so a swapped K/V, a reordered layer or a flipped byte order moves the digest.
fn sparse_snap() -> KvSnapshot {
    KvSnapshot {
        past_seq_len: PREFIX_LEN as usize,
        num_heads: 1,
        qk_head_dim: 2,
        v_head_dim: 1,
        layer0: Some(LayerKvSlice {
            lid: 0,
            past_k: vec![0x0102, 0x0304, 0x0506, 0x0708, 0x090A, 0x0B0C],
            past_v: vec![0x1112, 0x1314, 0x1516],
        }),
        shells: vec![LayerKvSlice {
            lid: 1,
            past_k: vec![0x2122, 0x2324, 0x2526, 0x2728, 0x292A, 0x2B2C],
            past_v: vec![0x3132, 0x3334, 0x3536],
        }],
    }
}

fn sparse_fp() -> ModelFingerprint {
    ModelFingerprint {
        arch: "k26".into(),
        num_layers: 2,
        num_experts: 1,
        top_k: 1,
        hidden_size: 8,
        num_kv_heads: 1,
        qk_head_dim: 2,
        v_head_dim: 1,
        vocab_size: 256,
        layer_start: 0,
        layer_end: 2,
        is_first: true,
        is_last: true,
    }
}

struct SparseFixture;

impl ContractFixture for SparseFixture {
    type Holder = SparseMoeKvHolder;
    const LAYOUT: u16 = KV_LAYOUT_VERSION;
    const ENGINE_REV: u64 = super::KV_ENGINE_REV;

    /// Plane-level, matching `SparseMoEEngine::kv_holder`.
    fn model_fp() -> u64 {
        sparse_fp().plane_digest()
    }

    fn with_prefix(partner: &str) -> Arc<SparseMoeKvHolder> {
        let mut st = SparseHolderState::new(4, sparse_fp());
        let prefix: Vec<i64> = TOKENS[..PREFIX_LEN as usize]
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        st.prefix
            .insert_pulled(partner, prefix, &sparse_fp(), sparse_snap());
        Arc::new(SparseMoeKvHolder {
            cache: Arc::new(Mutex::new(st)),
            model_fp: Self::model_fp(),
        })
    }

    fn with_capture(tenant: &str) -> (Arc<SparseMoeKvHolder>, u64) {
        let mut st = SparseHolderState::new(4, sparse_fp());
        let prefix: Vec<i32> = TOKENS[..PREFIX_LEN as usize].to_vec();
        let epoch = synth_epoch(&prefix);
        st.captures
            .insert(epoch, (tenant.to_string(), prefix, sparse_snap()));
        let holder = Arc::new(SparseMoeKvHolder {
            cache: Arc::new(Mutex::new(st)),
            model_fp: Self::model_fp(),
        });
        (holder, epoch)
    }

    /// Layer0 then shells, K then V, u16 little-endian — re-derived, not borrowed from the encoder.
    fn expected_digest() -> u64 {
        let s = sparse_snap();
        fnv1a(
            s.layer0
                .iter()
                .chain(s.shells.iter())
                .flat_map(|l| l.past_k.iter().chain(&l.past_v))
                .flat_map(|x| x.to_le_bytes()),
        )
    }

    fn decoded_past_len(m: &Manifest, payloads: &[(Vec<u8>, Vec<u8>)]) -> Option<u32> {
        super::wire_to_snapshot(m, payloads).map(|s| s.past_seq_len as u32)
    }
}

// ---- OvMoe fixture --------------------------------------------------------------------------

// past=3, two layers, every value distinct — same rationale as `sparse_snap`.
fn ov_snap() -> OvMoeKvSnapshot {
    let vals = |base: f32| {
        (0..12)
            .map(|i| base + i as f32 * 0.25)
            .collect::<Vec<f32>>()
    };
    OvMoeKvSnapshot {
        past_seq_len: PREFIX_LEN as usize,
        layers: vec![
            OvLayerKvSlice {
                k: vals(1.0),
                v: vals(100.0),
                seq: PREFIX_LEN as usize,
            },
            OvLayerKvSlice {
                k: vals(-40.0),
                v: vals(7.5),
                seq: PREFIX_LEN as usize,
            },
        ],
    }
}

fn ov_fp() -> ModelFingerprint {
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

struct OvFixture;

impl ContractFixture for OvFixture {
    type Holder = OvMoeKvHolder;
    const LAYOUT: u16 = OPAQUE_KV_LAYOUT;
    const ENGINE_REV: u64 = crate::ov_kv_coordination::KV_ENGINE_REV;

    fn model_fp() -> u64 {
        ov_fp().digest()
    }

    fn with_prefix(partner: &str) -> Arc<OvMoeKvHolder> {
        let mut st = OvHolderState::new(4, ov_fp());
        let prefix: Vec<i64> = TOKENS[..PREFIX_LEN as usize]
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        st.prefix
            .insert_pulled(partner, prefix, &ov_fp(), ov_snap());
        Arc::new(OvMoeKvHolder {
            cache: Arc::new(Mutex::new(st)),
            model_fp: Self::model_fp(),
        })
    }

    fn with_capture(tenant: &str) -> (Arc<OvMoeKvHolder>, u64) {
        let mut st = OvHolderState::new(4, ov_fp());
        let prefix: Vec<i32> = TOKENS[..PREFIX_LEN as usize].to_vec();
        let epoch = synth_epoch(&prefix);
        st.captures
            .insert(epoch, (tenant.to_string(), prefix, ov_snap()));
        let holder = Arc::new(OvMoeKvHolder {
            cache: Arc::new(Mutex::new(st)),
            model_fp: Self::model_fp(),
        });
        (holder, epoch)
    }

    /// The opaque blob contract restated by hand, independent of `ov_blob_encode`: u32 past,
    /// u32 num_layers, then per layer u32 seq, u32 k_len + k f32s, u32 v_len + v f32s — all
    /// little-endian, one blob in the K slot, an empty V.
    fn expected_digest() -> u64 {
        let s = ov_snap();
        let mut bytes = Vec::new();
        bytes.extend((s.past_seq_len as u32).to_le_bytes());
        bytes.extend((s.layers.len() as u32).to_le_bytes());
        for l in &s.layers {
            bytes.extend((l.seq as u32).to_le_bytes());
            bytes.extend((l.k.len() as u32).to_le_bytes());
            for &x in &l.k {
                bytes.extend(x.to_le_bytes());
            }
            bytes.extend((l.v.len() as u32).to_le_bytes());
            for &x in &l.v {
                bytes.extend(x.to_le_bytes());
            }
        }
        fnv1a(bytes.into_iter())
    }

    fn decoded_past_len(m: &Manifest, payloads: &[(Vec<u8>, Vec<u8>)]) -> Option<u32> {
        crate::ov_kv_coordination::ov_wire_to_snapshot(m, payloads).map(|s| s.past_seq_len as u32)
    }
}

// ---- instantiation: clause × holder ---------------------------------------------------------

macro_rules! instantiate_contract {
    ($fixture:ty) => {
        #[tokio::test]
        async fn negotiate_then_get_delivers_the_offered_bytes() {
            super::negotiate_then_get_delivers_the_offered_bytes::<$fixture>().await
        }
        #[tokio::test]
        async fn another_tenants_negotiate_and_get_are_refused() {
            super::another_tenants_negotiate_and_get_are_refused::<$fixture>().await
        }
        #[tokio::test]
        async fn a_drifted_epoch_is_refused_not_served_stale() {
            super::a_drifted_epoch_is_refused_not_served_stale::<$fixture>().await
        }
        #[tokio::test]
        async fn a_drifted_len_is_refused_not_served_stale() {
            super::a_drifted_len_is_refused_not_served_stale::<$fixture>().await
        }
        #[tokio::test]
        async fn an_unoffered_epoch_is_not_found() {
            super::an_unoffered_epoch_is_not_found::<$fixture>().await
        }
        #[tokio::test]
        async fn a_get_v2_for_a_foreign_rank_is_refused_before_the_export() {
            super::a_get_v2_for_a_foreign_rank_is_refused_before_the_export::<$fixture>().await
        }
        #[tokio::test]
        async fn a_tagged_capture_is_confined_to_its_tenant() {
            super::a_tagged_capture_is_confined_to_its_tenant::<$fixture>().await
        }
        #[tokio::test]
        async fn an_untagged_capture_still_serves_any_partner() {
            super::an_untagged_capture_still_serves_any_partner::<$fixture>().await
        }
    };
}

mod sparse_moe {
    instantiate_contract!(super::SparseFixture);
}

mod ov_moe {
    instantiate_contract!(super::OvFixture);
}
