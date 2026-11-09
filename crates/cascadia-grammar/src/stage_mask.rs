//! Repeated token masks on a pipeline stage wire.
//!
//! Most steps of a structured-output task allow the same tokens as the step
//! before (inside a JSON string, say), and a full mask costs vocab/8 bytes
//! per hop. So a stage that sends a mask equal to the last one it sent on the
//! same connection sends a short marker instead: the masked frame with its
//! mask-width field set to [`REPEAT`] and no mask words. Each end of a link
//! keeps one cache: the sender a [`MaskSendCache`], the receiver a
//! [`MaskRecvCache`].
//!
//! Only frames carrying exactly one masked row use the marker. TCP keeps
//! frame order and every receiver decodes every masked frame, so the order
//! the sender writes full masks in is the order the receiver commits them in.
//!
//! A receiver that gets the marker with nothing committed (a fresh
//! connection, say) can't resolve it: that is a [`MaskMiss`], which the
//! stage answers with [`MASK_MISS_TOKEN`] and the head turns into
//! [`MASK_MISS_MESSAGE`]. It is not meant to happen; the caches are cleared
//! on every event that could put the two ends out of step.

/// The mask-width value that means "the same mask as the last full one on
/// this connection". Not 0: a zero-width mask already means something on
/// some stage wires (a mask-support probe), and a real mask is never this
/// wide.
pub const REPEAT: u32 = u32::MAX;

/// The token a stage answers a repeat marker it can't resolve with (an
/// acknowledged reply). Distinct from the plain failure token (-1) so the
/// head can tell this stage's lost mask state from a failed step, and never a
/// sampled token.
pub const MASK_MISS_TOKEN: i32 = -2;

/// What a structured-output task fails with when a later stage couldn't
/// resolve a repeat marker. Nothing is latched: the next request starts with
/// a full mask.
pub const MASK_MISS_MESSAGE: &str =
    "structured output: a pipeline stage lost its token-mask state; retry the request";

/// A masked frame's mask as it came off the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireMask {
    /// The mask words.
    Full(Vec<u32>),
    /// The [`REPEAT`] marker: the receiver's last committed mask.
    Repeat,
}

/// A repeat marker reached a receiver with no mask to repeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskMiss;

impl std::fmt::Display for MaskMiss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(MASK_MISS_MESSAGE)
    }
}

/// The sending end of one link: the last full mask written and the
/// connection it was written on.
///
/// Check [`Self::repeats`] and write the step's frames under the same lock,
/// then [`Self::record`] once the step's last frame is written. Clear it on
/// a write error, on any failed step on the link (plain or masked: a
/// timeout, a NACK, an error), on a new connection, on an abort and on a
/// latch.
#[derive(Debug, Default, Clone)]
pub struct MaskSendCache {
    last: Option<(u64, Vec<u32>)>,
}

impl MaskSendCache {
    /// Whether a frame with `words`, written on connection `generation`, may
    /// go as the repeat marker: the last mask recorded on that same
    /// connection is `words`.
    pub fn repeats(&self, generation: u64, words: &[u32]) -> bool {
        self.last
            .as_ref()
            .is_some_and(|(g, last)| *g == generation && last.as_slice() == words)
    }

    /// The step carrying `words` (full or as the marker) is written on
    /// connection `generation`.
    pub fn record(&mut self, generation: u64, words: &[u32]) {
        match &mut self.last {
            Some((g, last)) if *g == generation && last.as_slice() == words => {}
            _ => self.last = Some((generation, words.to_vec())),
        }
    }

    /// Forget the last mask: the next masked frame goes full.
    pub fn clear(&mut self) {
        self.last = None;
    }

    /// Nothing recorded.
    pub fn is_empty(&self) -> bool {
        self.last.is_none()
    }
}

/// The receiving end of one link: the last full mask committed.
///
/// [`Self::resolve`] when a frame set is decoded, before any layer runs;
/// [`Self::commit`] once the whole frame set decoded. Clear it on a new
/// connection and on a decode error of a masked frame.
#[derive(Debug, Default, Clone)]
pub struct MaskRecvCache {
    last: Option<Vec<u32>>,
}

impl MaskRecvCache {
    /// The mask a frame stands for: its own words, or for the marker the
    /// last committed mask. [`MaskMiss`] when there is none.
    pub fn resolve(&self, wire: WireMask) -> Result<Vec<u32>, MaskMiss> {
        match wire {
            WireMask::Full(words) => Ok(words),
            WireMask::Repeat => self.last.clone().ok_or(MaskMiss),
        }
    }

    /// The frame set carrying `words` decoded completely.
    pub fn commit(&mut self, words: &[u32]) {
        if self.last.as_deref() != Some(words) {
            self.last = Some(words.to_vec());
        }
    }

    /// Forget the last mask: a repeat marker now misses.
    pub fn clear(&mut self) {
        self.last = None;
    }
}

#[cfg(test)]
#[path = "stage_mask_tests.rs"]
mod tests;
