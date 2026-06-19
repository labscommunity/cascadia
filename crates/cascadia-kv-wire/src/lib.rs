//! Wire contract for the KV coordination plane (`/cascadia/state/kv/v1`).
//!
//! Pure-Rust, **no engine deps** — compiles without the OpenVINO SDK, so it is shared by
//! cascadia-runtime (the KV producer) and cascadia-enterprise (the puller) without dragging the
//! engine into either build graph. See
//! `docs/superpowers/specs/2026-06-19-cascadia-issue-34-option-c-kv-plane-impl-design.md` §6/§7.
mod codec;
mod envelope;
mod manifest;
mod reslice;

pub use codec::{KvSnapshotCodec, ValidateError};
pub use envelope::{Get, KvMessage, Negotiate, Offer};
pub use manifest::{
    CacheKey, LayerMeta, Manifest, PartnerId, DTYPE_SIZE, KV_LAYOUT_VERSION, SCHEMA_VERSION,
    SEQ_AXIS,
};
pub use reslice::longest_common_prefix_len;
