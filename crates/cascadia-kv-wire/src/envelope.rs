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
        ];
        for m in msgs {
            let bytes = bincode::serde::encode_to_vec(&m, standard()).unwrap();
            let (back, _len): (KvMessage, usize) =
                bincode::serde::decode_from_slice(&bytes, standard()).unwrap();
            assert_eq!(m, back);
        }
    }
}
