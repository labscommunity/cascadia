//! Length-delimited framing for `KvMessage` control frames over a byte stream (§6/§7). A frame is a
//! `u32` big-endian length prefix followed by the bincode-encoded `KvMessage`. Bulk per-layer K/V
//! payloads follow a `Found` as their own length-prefixed frames (same envelope). The codec is sync
//! and buffer-based so it is trivially testable; the async transport read loop wraps it.

use bincode::config::standard;

use crate::envelope::KvMessage;

/// Hard cap on a single control frame (DoS guard, §13). Bulk KV payloads are streamed as their own
/// frames and bounded separately by `max_snapshot_bytes`.
pub const MAX_FRAME_LEN: u32 = 64 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub enum FrameError {
    /// Declared length exceeds [`MAX_FRAME_LEN`].
    TooLarge,
    /// Body did not decode as a `KvMessage`.
    Decode,
    /// Body failed to encode (near-unreachable for these types).
    Encode,
    /// Not enough bytes buffered yet — the caller should read more and retry.
    Incomplete,
}

/// Encode one `KvMessage` as a length-prefixed frame.
pub fn encode_frame(msg: &KvMessage) -> Result<Vec<u8>, FrameError> {
    let body = bincode::serde::encode_to_vec(msg, standard()).map_err(|_| FrameError::Encode)?;
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
    let (msg, _) = bincode::serde::decode_from_slice(&buf[4..end], standard())
        .map_err(|_| FrameError::Decode)?;
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
    fn garbage_body_fails_to_decode() {
        // Valid length prefix, but the body is not a KvMessage.
        let mut buf = 4u32.to_be_bytes().to_vec();
        buf.extend_from_slice(&[0xff, 0xff, 0xff, 0xff]);
        assert_eq!(decode_frame(&buf), Err(FrameError::Decode));
    }
}
