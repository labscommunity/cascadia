//! Decoded media attached to a [`GenerationTask`](crate::GenerationTask).
//!
//! The API decodes every image and video once and hands engines raw pixels,
//! never URLs or encoded bytes. The shape is a frame stack, so a video is one
//! [`MediaInput`] of kind [`MediaKind::Video`] with the same task, caches
//! keyed on [`MediaInput::sha256`], and engine entry points as an image
//! (`docs/VLM.md`, "Video").

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// What a [`MediaInput`] holds. Matches on this enum must not use a
/// catch-all that silently accepts an unknown kind: an engine that cannot
/// take a kind refuses it by name.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MediaKind {
    /// A single frame (`num_frames == 1`, no timestamps).
    Image,
    /// Sampled frames of one video (`num_frames >= 1`, timestamps set), sent
    /// only to an engine that takes video natively (ov-genai on the families
    /// GenAI preprocesses video for). Every other engine gets a video's frames
    /// as images.
    Video,
}

/// One decoded media item, in the order its placeholder appears in the prompt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MediaInput {
    pub kind: MediaKind,
    /// RGB8, frame-major `[num_frames, height, width, 3]`.
    pub pixels: Arc<[u8]>,
    pub num_frames: u32,
    pub width: u32,
    pub height: u32,
    /// Per-frame timestamps for video; `None` for images.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamps_ms: Option<Vec<u32>>,
    /// SHA-256 over the dimensions and the decoded pixels (a video's whole
    /// frame stack). Any cache that
    /// covers a media span keys on this, never on the placeholder token ids,
    /// which are identical for every image.
    pub sha256: [u8; 32],
}

impl MediaInput {
    /// Byte length `pixels` must have for the declared shape.
    pub fn expected_len(&self) -> usize {
        self.num_frames as usize * self.height as usize * self.width as usize * 3
    }
}
