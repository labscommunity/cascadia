//! Length-delimited framing for `KvMessage` control frames over a byte stream (§6/§7). A frame is a
//! `u32` big-endian length prefix followed by the bincode-encoded `KvMessage`. Bulk per-layer K/V
//! payloads follow a `Found` as their own length-prefixed frames (same envelope). The codec is sync
//! and buffer-based so it is trivially testable; the async transport read loop wraps it.

use bincode::config::{standard, Config};

/// Decode/encode config with a byte budget == the frame cap. bincode's `standard()` carries NO limit
/// (`LIMIT = None`), so a native `String`/`Vec` length prefix inside a frame allocates
/// attacker-controlled bytes BEFORE any post-decode cap check — a ~10-byte hostile frame declaring an
/// exabyte string aborts the process (OOM). The outer `MAX_FRAME_LEN` check bounds only the input
/// slice, not the nested collection prefixes. `with_limit` makes an over-budget length a clean
/// `DecodeError::LimitExceeded` with no allocation. It does not change the encoded bytes, so goldens
/// and the wire format are unaffected.
fn wire_config() -> impl Config {
    standard().with_limit::<{ MAX_FRAME_LEN as usize }>()
}

use crate::envelope::{KvMessage, MAX_PARTNER_LEN};

/// Hard cap on a single control frame (DoS guard, §13). Bulk KV payloads are streamed as their own
/// frames and bounded separately by `max_snapshot_bytes`.
pub const MAX_FRAME_LEN: u32 = 64 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub enum FrameError {
    /// A declared or embedded length exceeds its cap ([`MAX_FRAME_LEN`], [`MAX_PARTNER_LEN`]).
    TooLarge,
    /// Body did not decode as a `KvMessage`.
    Decode,
    /// Body failed to encode (near-unreachable for these types).
    Encode,
    /// Not enough bytes buffered yet — the caller should read more and retry.
    Incomplete,
}

/// Per-variant field caps that bincode cannot express. Only `TenantHint` is checked: it is the one
/// variant whose cap was declared with it. The other string-bearing variants predate the cap and are
/// deliberately left alone — tightening them here would change which already-valid frames a head
/// accepts, which is a wire behaviour change, not a bounds fix.
fn fields_within_caps(msg: &KvMessage) -> bool {
    match msg {
        KvMessage::TenantHint { partner, .. } => partner.len() <= MAX_PARTNER_LEN,
        _ => true,
    }
}

/// Encode one `KvMessage` as a length-prefixed frame.
pub fn encode_frame(msg: &KvMessage) -> Result<Vec<u8>, FrameError> {
    // Symmetric with decode: a local bug must not emit a frame every peer is obliged to drop.
    if !fields_within_caps(msg) {
        return Err(FrameError::TooLarge);
    }
    let body = bincode::serde::encode_to_vec(msg, wire_config()).map_err(|_| FrameError::Encode)?;
    if body.len() as u64 > u64::from(MAX_FRAME_LEN) {
        return Err(FrameError::TooLarge);
    }
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Try to decode one frame from the front of `buf`. On success returns the message and the number of
/// bytes consumed (the caller drains those and may decode the next frame). `Incomplete` means more
/// bytes are needed; `TooLarge`/`Decode` are hard errors (close the stream → reprefill).
pub fn decode_frame(buf: &[u8]) -> Result<(KvMessage, usize), FrameError> {
    if buf.len() < 4 {
        return Err(FrameError::Incomplete);
    }
    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if len > MAX_FRAME_LEN {
        return Err(FrameError::TooLarge);
    }
    let end = 4 + len as usize;
    if buf.len() < end {
        return Err(FrameError::Incomplete);
    }
    let (msg, consumed) = bincode::serde::decode_from_slice(&buf[4..end], wire_config())
        .map_err(|_| FrameError::Decode)?;
    // The declared length must match the bincode encoding exactly — trailing junk inside a frame is a
    // structural anomaly; reject rather than silently swallow it.
    if consumed != len as usize {
        return Err(FrameError::Decode);
    }
    // Caps are enforced here, at the single ingress, so an over-long field can never reach a caller
    // half-applied — the whole frame is rejected and the stream is torn down (→ reprefill).
    if !fields_within_caps(&msg) {
        return Err(FrameError::TooLarge);
    }
    Ok((msg, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::PartnerId;
    use crate::Offer;

    fn sample() -> KvMessage {
        KvMessage::Negotiate(crate::Negotiate {
            partner: PartnerId("a".into()),
            model_fingerprint: 7,
            token_ids: vec![1, 2, 3],
        })
    }

    #[test]
    fn frame_roundtrips() {
        let m = sample();
        let bytes = encode_frame(&m).unwrap();
        let (back, consumed) = decode_frame(&bytes).unwrap();
        assert_eq!(back, m);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn two_frames_decode_in_sequence() {
        let m1 = sample();
        let m2 = KvMessage::Offer(Offer {
            snapshot_epoch: 9,
            prefix_token_len: 2,
        });
        let mut buf = encode_frame(&m1).unwrap();
        buf.extend(encode_frame(&m2).unwrap());
        let (d1, n1) = decode_frame(&buf).unwrap();
        assert_eq!(d1, m1);
        let (d2, n2) = decode_frame(&buf[n1..]).unwrap();
        assert_eq!(d2, m2);
        assert_eq!(n1 + n2, buf.len());
    }

    #[test]
    fn truncated_header_is_incomplete() {
        assert_eq!(decode_frame(&[0, 0]), Err(FrameError::Incomplete));
    }

    #[test]
    fn truncated_body_is_incomplete() {
        let bytes = encode_frame(&sample()).unwrap();
        assert_eq!(
            decode_frame(&bytes[..bytes.len() - 1]),
            Err(FrameError::Incomplete)
        );
    }

    #[test]
    fn oversized_length_prefix_is_rejected() {
        // A length prefix beyond the cap must be rejected before allocating/reading the body.
        let mut buf = (MAX_FRAME_LEN + 1).to_be_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 8]);
        assert_eq!(decode_frame(&buf), Err(FrameError::TooLarge));
    }

    #[test]
    fn trailing_junk_inside_frame_is_rejected() {
        // A length prefix larger than the actual bincode body (extra trailing bytes inside the
        // declared frame) must reject, not silently swallow the junk.
        let body = bincode::serde::encode_to_vec(sample(), standard()).unwrap();
        let inflated = body.len() as u32 + 3;
        let mut buf = inflated.to_be_bytes().to_vec();
        buf.extend_from_slice(&body);
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC]); // junk counted inside the frame length
        assert_eq!(decode_frame(&buf), Err(FrameError::Decode));
    }

    /// Wrap an already-encoded body in a length prefix, bypassing `encode_frame`'s own caps — the
    /// only way to build the frame a hostile peer would send.
    fn forge(body: &[u8]) -> Vec<u8> {
        let mut buf = (body.len() as u32).to_be_bytes().to_vec();
        buf.extend_from_slice(body);
        buf
    }

    #[test]
    fn oversized_tenant_hint_partner_is_rejected_both_ways() {
        let msg = KvMessage::TenantHint {
            request_id: [1u8; 16],
            partner: "a".repeat(crate::MAX_PARTNER_LEN + 1),
        };
        assert_eq!(encode_frame(&msg), Err(FrameError::TooLarge));
        let body = bincode::serde::encode_to_vec(&msg, standard()).unwrap();
        assert_eq!(decode_frame(&forge(&body)), Err(FrameError::TooLarge));
        // The cap is a cap, not an off-by-one: exactly MAX_PARTNER_LEN still rides.
        let at_cap = KvMessage::TenantHint {
            request_id: [1u8; 16],
            partner: "a".repeat(crate::MAX_PARTNER_LEN),
        };
        let bytes = encode_frame(&at_cap).unwrap();
        assert_eq!(decode_frame(&bytes).unwrap().0, at_cap);
    }

    #[test]
    fn non_utf8_tenant_hint_partner_is_rejected() {
        // Tag 16 (TenantHint) ++ 16-byte request_id ++ varint len 2 ++ two invalid UTF-8 bytes.
        let mut body = vec![16u8];
        body.extend_from_slice(&[0u8; 16]);
        body.extend_from_slice(&[2, 0xff, 0xfe]);
        assert_eq!(decode_frame(&forge(&body)), Err(FrameError::Decode));
    }

    #[test]
    fn garbage_body_fails_to_decode() {
        // Valid length prefix, but the body is not a KvMessage.
        let mut buf = 4u32.to_be_bytes().to_vec();
        buf.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]);
        assert_eq!(decode_frame(&buf), Err(FrameError::Decode));
    }
    /// A hostile frame declaring a huge String length must be REFUSED at decode without allocating,
    /// not abort the process. The byte-limited config turns the over-budget length prefix into a
    /// clean Decode error. (Regression for the unbounded-alloc DoS: variant tag 5 = Error(String),
    /// then a varint declaring an ~8-exabyte string with zero string bytes following.)
    #[test]
    fn a_forged_giant_string_length_is_refused_not_allocated() {
        // body: [tag=5][varint len ~ 0x7000_0000_0000_0000] — a valid-looking Error(String) header
        // whose declared length dwarfs any real input. bincode varint: 0xFB marks a u64 follows.
        let mut body = vec![5u8, 0xFBu8];
        body.extend_from_slice(&0x7000_0000_0000_0000u64.to_le_bytes());
        let mut frame = (body.len() as u32).to_be_bytes().to_vec();
        frame.extend_from_slice(&body);
        // len is well under MAX_FRAME_LEN, so the outer check passes; the inner budget must catch it.
        assert!(matches!(decode_frame(&frame), Err(FrameError::Decode)));
    }
}
