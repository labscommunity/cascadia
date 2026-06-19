use serde::{Deserialize, Serialize};

use crate::manifest::{Manifest, PartnerId};

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

/// Control envelope on `/cascadia/state/kv/v1`.
///
/// Commit is **in-band** (the warm/cold decision rides the pipeline dispatch, §7) — there is no
/// `COMMIT`/`Ack` message. `Found` is followed on the stream by per-layer length-prefixed K then V
/// payloads (framed by the transport, not this enum).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvMessage {
    Negotiate(Negotiate),
    Offer(Offer),
    Get(Get),
    Found(Manifest),
    NotFound,
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::config::standard;

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
        ];
        for m in msgs {
            let bytes = bincode::serde::encode_to_vec(&m, standard()).unwrap();
            let (back, _len): (KvMessage, usize) =
                bincode::serde::decode_from_slice(&bytes, standard()).unwrap();
            assert_eq!(m, back);
        }
    }
}
