//! The native holder driven through the real wire protocol, not called in-process.
//!
//! `kv_coordination.rs`'s own tests reach [`SparseMoeKvHolder`] by direct method call; this drives
//! the same holder over `cascadia_kv_wire`'s framed NEGOTIATE/GET across a `tokio::io::duplex`, and
//! runs the real [`KvSnapshotCodec`] on what comes back. No network, no weights, no inference.
//!
//! The serve loop below mirrors the enterprise `cascadia-kv-coord::serve_kv_stream` — one request
//! frame per stream, `Found` + per-layer length-prefixed K/V blobs, `NotFound` for every refusal.
//! That crate is not a dependency here (it is the puller's side of the plane), so it is re-stated;
//! everything it wraps — the envelope, the framing, the codec, the holder — is the production code.
//!
//! Lives in `src/` rather than `tests/` because `SparseMoeKvHolder`, `SparseHolderState` and the
//! `SharedHolderCache` alias are all `pub(crate)`: an integration test cannot construct the holder
//! without widening the crate's public API.

use super::*;
use crate::kv_prefix_cache::{KvSnapshot, LayerKvSlice, ModelFingerprint};
use cascadia_engine::KvSnapshotHolder;
use cascadia_kv_wire::{
    decode_frame, encode_frame, Get, KvMessage, KvSnapshotCodec, Negotiate, Offer, MAX_FRAME_LEN,
};
use std::io::{Error, ErrorKind, Result};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, DuplexStream};

/// The tokens a peer asks about. Longer than the cached prefix by one: `lookup_ns` refuses to serve
/// a prefix that covers the whole prompt (no tail left to drive prefill).
const TOKENS: &[i32] = &[11, 22, 33, 44];
/// The shard this holder is standing in for — what an inbound `GetV2` must name.
const RANK: u16 = 1;
const OWNER: &str = "tenant-a";

// num_heads=1, len=3, qk_head_dim=2, v_head_dim=1 ⇒ 6 u16 of K and 3 u16 of V per layer. Every
// value is distinct so a swapped K/V, a reordered layer or a flipped byte order moves the digest.
fn snap() -> KvSnapshot {
    KvSnapshot {
        past_seq_len: 3,
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

fn fp() -> ModelFingerprint {
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

/// Plane-level, matching `SparseMoEEngine::kv_holder` — the fingerprint every rank's GET asserts.
fn model_fp() -> u64 {
    fp().plane_digest()
}

/// A holder whose mirrored cache holds `partner`'s `TOKENS[..3]` snapshot — the state a plane
/// consumer-insert (`KvCoordination::insert`) leaves behind.
fn holder_for(partner: &str) -> Arc<SparseMoeKvHolder> {
    let mut st = SparseHolderState::new(4, fp());
    let prefix: Vec<i64> = TOKENS[..3].iter().map(|&t| i64::from(t)).collect();
    st.prefix.insert_pulled(partner, prefix, &fp(), snap());
    Arc::new(SparseMoeKvHolder {
        cache: Arc::new(Mutex::new(st)),
        model_fp: model_fp(),
    })
}

// ---- wire I/O (mirror of cascadia-kv-coord::wire_io) ----------------------------------------

fn wire_err(e: impl std::fmt::Debug) -> Error {
    Error::new(ErrorKind::InvalidData, format!("kv frame: {e:?}"))
}

async fn write_msg<W: AsyncWrite + Unpin>(w: &mut W, msg: &KvMessage) -> Result<()> {
    let frame = encode_frame(msg).map_err(wire_err)?;
    w.write_all(&frame).await?;
    w.flush().await
}

async fn read_msg<R: AsyncRead + Unpin>(r: &mut R) -> Result<KvMessage> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_LEN {
        return Err(Error::new(ErrorKind::InvalidData, "frame exceeds cap"));
    }
    let mut framed = len_buf.to_vec();
    framed.resize(4 + len as usize, 0);
    r.read_exact(&mut framed[4..]).await?;
    decode_frame(&framed).map(|(m, _)| m).map_err(wire_err)
}

async fn write_blob<W: AsyncWrite + Unpin>(w: &mut W, bytes: &[u8]) -> Result<()> {
    w.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
    w.write_all(bytes).await?;
    w.flush().await
}

async fn read_blob<R: AsyncRead + Unpin>(r: &mut R) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let mut blob = vec![0u8; u32::from_be_bytes(len_buf) as usize];
    r.read_exact(&mut blob).await?;
    Ok(blob)
}

// ---- serve loop (mirror of cascadia-kv-coord::serve_kv_stream) ------------------------------

/// Serve one inbound stream: read one request frame, answer it, return. `local_shard` is the shard
/// this node holds; a `GetV2` naming any other rank is refused BEFORE the export reaches the holder.
async fn serve_one<S>(stream: &mut S, holder: &SparseMoeKvHolder, local_shard: u16) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    match read_msg(stream).await? {
        KvMessage::Negotiate(Negotiate {
            partner, token_ids, ..
        }) => {
            let reply = match holder.lookup(&partner.0, &token_ids) {
                Some((snapshot_epoch, prefix_token_len)) => KvMessage::Offer(Offer {
                    snapshot_epoch,
                    prefix_token_len,
                }),
                None => KvMessage::NotFound,
            };
            write_msg(stream, &reply).await
        }
        KvMessage::GetV2 {
            partner,
            expected_epoch,
            expected_len,
            rank,
            ..
        } => {
            if rank != local_shard {
                return write_msg(stream, &KvMessage::NotFound).await;
            }
            serve_get(stream, holder, &partner.0, expected_epoch, expected_len).await
        }
        // Legacy rank-less fetch, still servable (enterprise `handle_request`) — see
        // `the_legacy_rank_less_get_carries_no_rank_to_bind`.
        KvMessage::Get(Get {
            partner,
            expected_epoch,
            expected_len,
            ..
        }) => serve_get(stream, holder, &partner.0, expected_epoch, expected_len).await,
        _ => write_msg(stream, &KvMessage::Error("unexpected request".into())).await,
    }
}

async fn serve_get<S>(
    stream: &mut S,
    holder: &SparseMoeKvHolder,
    partner: &str,
    epoch: u64,
    len: u32,
) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    match holder.export(partner, epoch, len) {
        Some((manifest, payloads)) => {
            write_msg(stream, &KvMessage::Found(manifest)).await?;
            for (k, v) in &payloads {
                write_blob(stream, k).await?;
                write_blob(stream, v).await?;
            }
            Ok(())
        }
        None => write_msg(stream, &KvMessage::NotFound).await,
    }
}

// ---- peer side ------------------------------------------------------------------------------

/// Dial the holder, send one request, read its control reply. The server drops its half on return,
/// so the caller can go on to read the `Found` payloads (already buffered) or observe EOF.
async fn ask(holder: &Arc<SparseMoeKvHolder>, req: KvMessage) -> (DuplexStream, KvMessage) {
    let (mut client, mut server) = tokio::io::duplex(1 << 16);
    let h = Arc::clone(holder);
    let srv = tokio::spawn(async move { serve_one(&mut server, &h, RANK).await });
    write_msg(&mut client, &req).await.expect("request");
    let reply = read_msg(&mut client).await.expect("reply");
    srv.await.expect("serve task").expect("serve loop");
    (client, reply)
}

fn negotiate(partner: &str) -> KvMessage {
    KvMessage::Negotiate(Negotiate {
        partner: PartnerId(partner.into()),
        model_fingerprint: model_fp(),
        token_ids: TOKENS.to_vec(),
    })
}

fn get_v2(partner: &str, epoch: u64, len: u32, rank: u16) -> KvMessage {
    KvMessage::GetV2 {
        partner: PartnerId(partner.into()),
        model_fingerprint: model_fp(),
        expected_epoch: epoch,
        expected_len: len,
        rank,
    }
}

/// NEGOTIATE as `partner` and return the offered `(epoch, len)`.
async fn negotiated(holder: &Arc<SparseMoeKvHolder>, partner: &str) -> (u64, u32) {
    match ask(holder, negotiate(partner)).await.1 {
        KvMessage::Offer(o) => (o.snapshot_epoch, o.prefix_token_len),
        other => panic!("expected Offer, got {other:?}"),
    }
}

/// Consumer half of a served GET: read the blobs that follow `Found` and run the real structural
/// validation over them, exactly as the puller does before it inserts.
async fn receive(client: &mut DuplexStream, m: &Manifest) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut payloads = Vec::new();
    for _ in &m.layers {
        let k = read_blob(client).await.expect("k blob");
        let v = read_blob(client).await.expect("v blob");
        payloads.push((k, v));
    }
    let refs: Vec<(&[u8], &[u8])> = payloads
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    KvSnapshotCodec::validate(
        m,
        &refs,
        KV_LAYOUT_VERSION,
        KV_ENGINE_REV,
        model_fp(),
        TOKENS,
    )
    .expect("what arrived must pass consumer validation");
    payloads
}

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Digest of the bytes that actually arrived, in wire order.
fn arrived_digest(payloads: &[(Vec<u8>, Vec<u8>)]) -> u64 {
    fnv1a(
        payloads
            .iter()
            .flat_map(|(k, v)| k.iter().chain(v).copied()),
    )
}

/// What the seeded snapshot MUST serialise to: layer0 then shells, K then V, u16 little-endian.
/// Re-derived from the snapshot rather than borrowed from `payload_digest`, so an export-side
/// encoding change cannot cancel itself out against the holder's own digest.
fn expected_digest(s: &KvSnapshot) -> u64 {
    fnv1a(
        s.layer0
            .iter()
            .chain(s.shells.iter())
            .flat_map(|l| l.past_k.iter().chain(&l.past_v))
            .flat_map(|x| x.to_le_bytes()),
    )
}

// ---- tests ----------------------------------------------------------------------------------

/// The round trip a warm pull actually makes: NEGOTIATE, then the asserted rank-bound GET, then the
/// bytes. Asserted on the DIGEST of what arrived, not its length — equal lengths are what an earlier
/// probe mistook for equal content.
#[tokio::test]
async fn negotiate_then_get_delivers_the_offered_bytes() {
    let holder = holder_for(OWNER);
    let (epoch, len) = negotiated(&holder, OWNER).await;
    assert_eq!(len, 3, "the cached prefix, not the whole prompt");

    let (mut client, reply) = ask(&holder, get_v2(OWNER, epoch, len, RANK)).await;
    let KvMessage::Found(m) = reply else {
        panic!("expected Found, got {reply:?}")
    };
    let payloads = receive(&mut client, &m).await;

    assert_eq!(m.snapshot_epoch, epoch, "served under the negotiated epoch");
    assert_eq!(m.prefix_token_len, len);
    assert_eq!(m.token_ids, TOKENS[..len as usize]);
    assert_eq!(m.partner.0, OWNER, "stamped with the asserting tenant");
    assert_eq!(
        arrived_digest(&payloads),
        expected_digest(&snap()),
        "the bytes off the wire are the snapshot the holder was seeded with"
    );
}

/// The consumer half of the round trip. `KvCoordination::insert` needs a loaded engine, but the part
/// of it that touches the plane — `wire_to_snapshot` then a namespaced `insert_pulled` — does not, so
/// the pulled slice is ingested into a second holder and re-served. Byte-identical out the far side,
/// which no length or token-count comparison would have shown.
#[tokio::test]
async fn a_pulled_slice_re_serves_byte_identically() {
    let donor = holder_for(OWNER);
    let (epoch, len) = negotiated(&donor, OWNER).await;
    let (mut client, reply) = ask(&donor, get_v2(OWNER, epoch, len, RANK)).await;
    let KvMessage::Found(m) = reply else {
        panic!("expected Found, got {reply:?}")
    };
    let payloads = receive(&mut client, &m).await;

    let pulled = wire_to_snapshot(&m, &payloads).expect("validated slice must decode");
    let mut st = SparseHolderState::new(4, fp());
    st.prefix.insert_pulled(
        OWNER,
        m.token_ids.iter().map(|&t| i64::from(t)).collect(),
        &fp(),
        pulled,
    );
    let consumer = Arc::new(SparseMoeKvHolder {
        cache: Arc::new(Mutex::new(st)),
        model_fp: model_fp(),
    });

    assert_eq!(
        negotiated(&consumer, OWNER).await,
        (epoch, len),
        "the consumer offers the same epoch — it is content-derived, so a re-derivation must agree"
    );
    let (mut client, reply) = ask(&consumer, get_v2(OWNER, epoch, len, RANK)).await;
    let KvMessage::Found(m2) = reply else {
        panic!("the consumer cannot re-serve what it pulled: {reply:?}")
    };
    assert_eq!(
        arrived_digest(&receive(&mut client, &m2).await),
        expected_digest(&snap()),
        "a donor→consumer→donor cycle changed the bytes"
    );
}

/// Issue-34 H.1a over the wire: the in-process unit test proves `lookup`/`export` are namespaced;
/// this proves nothing on the framed path un-does it. Both probe shapes are covered — the NEGOTIATE
/// length oracle, and the asserted-GET side door where the prober skips NEGOTIATE entirely.
#[tokio::test]
async fn another_tenants_get_cannot_reach_the_offer() {
    let holder = holder_for(OWNER);

    // The oracle probe: every guess must read as an empty cache, never a truncated length.
    for probe in ["tenant-b", "tenant-c"] {
        let (_c, reply) = ask(&holder, negotiate(probe)).await;
        assert_eq!(reply, KvMessage::NotFound, "{probe} negotiated a hit");
    }

    // The side door: tenant-b asserts the owner's own (epoch, len), obtained here by negotiating as
    // the owner but derivable by anyone who guesses the prefix (`synth_epoch` is pure).
    let (epoch, len) = negotiated(&holder, OWNER).await;
    let (_c, reply) = ask(&holder, get_v2("tenant-b", epoch, len, RANK)).await;
    assert_eq!(
        reply,
        KvMessage::NotFound,
        "tenant-b must not reach the owner's stashed offer"
    );

    // ...and the owner still gets served, or every assertion above holds vacuously.
    let (mut client, reply) = ask(&holder, get_v2(OWNER, epoch, len, RANK)).await;
    let KvMessage::Found(m) = reply else {
        panic!("owner refused too — the confinement above proves nothing")
    };
    assert_eq!(
        arrived_digest(&receive(&mut client, &m).await),
        expected_digest(&snap())
    );
}

/// A GET asserting anything other than what was offered is refused, not answered with whatever the
/// holder happens to hold. Each case re-negotiates first: a mis-asserted GET consumes the offer, so
/// without that the later cases would pass for the wrong reason.
#[tokio::test]
async fn a_drifted_epoch_or_len_is_refused_not_served_stale() {
    let holder = holder_for(OWNER);

    for (label, epoch_shift, len_shift) in [
        ("foreign epoch", 1u64, 0i64),
        ("longer than offered", 0, 1),
        ("shorter than offered", 0, -1),
    ] {
        let (epoch, len) = negotiated(&holder, OWNER).await;
        let asked_len = (i64::from(len) + len_shift) as u32;
        let (_c, reply) = ask(&holder, get_v2(OWNER, epoch ^ epoch_shift, asked_len, RANK)).await;
        assert_eq!(reply, KvMessage::NotFound, "{label} was served");
    }

    // The honest assertion still serves — the refusals above are the drift, not a dead holder.
    let (epoch, len) = negotiated(&holder, OWNER).await;
    let (_c, reply) = ask(&holder, get_v2(OWNER, epoch, len, RANK)).await;
    assert!(matches!(reply, KvMessage::Found(_)));
}

/// A GET for a prefix that was never offered is a clean `NotFound` and the end of the stream — the
/// holder serves one request per stream, so the peer sees EOF next, not a second frame or a hang.
#[tokio::test]
async fn an_unoffered_prefix_is_not_found_and_ends_the_stream() {
    let holder = holder_for(OWNER);
    let (mut client, reply) = ask(&holder, get_v2(OWNER, 0xDEAD_BEEF, 3, RANK)).await;
    assert_eq!(reply, KvMessage::NotFound);
    assert_eq!(
        read_msg(&mut client).await.unwrap_err().kind(),
        ErrorKind::UnexpectedEof,
        "the serve ends after one answered request"
    );
}

/// Defect 1b, donor half: `GetV2` carries the rank so a node holding shard M cannot answer a rank-N
/// fetch with M's bytes — every other guard (model-level fingerprint, layer count) passes on both.
/// The refusal lands BEFORE the export, which is observable here: the offer survives it.
#[tokio::test]
async fn a_get_v2_for_a_foreign_rank_is_refused_before_the_export() {
    let holder = holder_for(OWNER);
    let (epoch, len) = negotiated(&holder, OWNER).await;

    let (_c, reply) = ask(&holder, get_v2(OWNER, epoch, len, RANK + 1)).await;
    assert_eq!(reply, KvMessage::NotFound, "a foreign rank was served");

    let (_c, reply) = ask(&holder, get_v2(OWNER, epoch, len, RANK)).await;
    assert!(
        matches!(reply, KvMessage::Found(_)),
        "the refused fetch consumed the offer — it must be refused before the export"
    );
}

/// The recorded residual: the pre-V2 `Get` has no rank field, so the binding above cannot apply to
/// it. Pinned rather than fixed — refusing it would break genuinely old peers, and the tenant
/// namespace still gates it.
#[tokio::test]
async fn the_legacy_rank_less_get_carries_no_rank_to_bind() {
    let holder = holder_for(OWNER);
    let (epoch, len) = negotiated(&holder, OWNER).await;
    let legacy = KvMessage::Get(Get {
        partner: PartnerId(OWNER.into()),
        model_fingerprint: model_fp(),
        expected_epoch: epoch,
        expected_len: len,
    });
    let (_c, reply) = ask(&holder, legacy).await;
    assert!(matches!(reply, KvMessage::Found(_)));
}

/// The `captures` stash must never answer a wire GET.
///
/// `captures` is keyed on epoch ALONE — the CAPTURE frame carries no tenant, so a worker rank has
/// none to tag entries with — while `synth_epoch` is a pure function of the prefix tokens. So while
/// `export` fell back to it, any caller who could derive a victim's epoch was served that victim's
/// slice over the wire. The offer path is partner-scoped and unaffected; this pins the fallback as
/// removed rather than merely unused, because nothing else fails if it comes back.
#[tokio::test]
async fn a_capture_is_never_reachable_over_the_wire() {
    let mut st = SparseHolderState::new(4, fp());
    // Only a capture — no offer for anyone. Pre-fix this was reachable by every partner alike.
    let prefix: Vec<i32> = TOKENS[..3].to_vec();
    let epoch = super::synth_epoch(&prefix);
    st.captures.insert(epoch, (prefix, snap()));
    let holder = Arc::new(SparseMoeKvHolder {
        cache: Arc::new(Mutex::new(st)),
        model_fp: model_fp(),
    });

    let len = snap().past_seq_len as u32;
    for who in [OWNER, "tenant-b"] {
        let (_c, reply) = ask(&holder, get_v2(who, epoch, len, RANK)).await;
        assert_eq!(
            reply,
            KvMessage::NotFound,
            "{who} reached the capture stash over the wire"
        );
    }
}
