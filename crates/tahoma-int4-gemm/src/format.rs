//! On-disk and in-memory layout of one expert's packed int4 weights.
//!
//! The exporter (see `rainier/scripts/k26_extract_expert_bins.py`)
//! writes a flat binary per (layer, expert) with the six tensors
//! laid out contiguously, no padding, no header:
//!
//!   gate_packed:  i32 [INTERMEDIATE, PACKED_COLS_IN]      ≈ 7.34 MiB
//!   gate_scale:   u16 [INTERMEDIATE, GROUPS_IN]           ≈ 0.9  MiB
//!   up_packed:    i32 [INTERMEDIATE, PACKED_COLS_IN]
//!   up_scale:     u16 [INTERMEDIATE, GROUPS_IN]
//!   down_packed:  i32 [HIDDEN, PACKED_COLS_MID]           ≈ 7.34 MiB
//!   down_scale:   u16 [HIDDEN, GROUPS_MID]                ≈ 0.9  MiB
//!
//! Total per expert: ~25 MiB. We mmap one file per (layer, expert)
//! and slice each tensor by offset/length.
//!
//! The "u16" scale slice is just `half::bf16` raw bits — the kernel
//! converts on-the-fly per group.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use thiserror::Error;

use crate::{GROUPS_IN, GROUPS_MID, HIDDEN, INTERMEDIATE, PACKED_COLS_IN, PACKED_COLS_MID};

#[derive(Debug, Error)]
pub enum GemmError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("file too small: expected at least {expected}, got {actual}")]
    Truncated { expected: usize, actual: usize },
}

const GATE_PACKED_BYTES: usize = INTERMEDIATE * PACKED_COLS_IN * 4;
const GATE_SCALE_BYTES: usize = INTERMEDIATE * GROUPS_IN * 2;
const UP_PACKED_BYTES: usize = INTERMEDIATE * PACKED_COLS_IN * 4;
const UP_SCALE_BYTES: usize = INTERMEDIATE * GROUPS_IN * 2;
const DOWN_PACKED_BYTES: usize = HIDDEN * PACKED_COLS_MID * 4;
const DOWN_SCALE_BYTES: usize = HIDDEN * GROUPS_MID * 2;

const TOTAL_BYTES: usize = GATE_PACKED_BYTES
    + GATE_SCALE_BYTES
    + UP_PACKED_BYTES
    + UP_SCALE_BYTES
    + DOWN_PACKED_BYTES
    + DOWN_SCALE_BYTES;

/// One expert's packed weights, mmap'd. Slices reference into the
/// memory map; the mmap is kept alive by this struct.
pub struct ExpertWeights {
    #[allow(dead_code)]
    mmap: Mmap,
    gate_packed_off: usize,
    gate_scale_off: usize,
    up_packed_off: usize,
    up_scale_off: usize,
    down_packed_off: usize,
    down_scale_off: usize,
}

impl ExpertWeights {
    pub fn open(path: &Path) -> Result<Self, GemmError> {
        let f = File::open(path)?;
        let len = f.metadata()?.len() as usize;
        if len < TOTAL_BYTES {
            return Err(GemmError::Truncated {
                expected: TOTAL_BYTES,
                actual: len,
            });
        }
        let mmap = unsafe { Mmap::map(&f)? };

        let mut off = 0;
        let gate_packed_off = off;
        off += GATE_PACKED_BYTES;
        let gate_scale_off = off;
        off += GATE_SCALE_BYTES;
        let up_packed_off = off;
        off += UP_PACKED_BYTES;
        let up_scale_off = off;
        off += UP_SCALE_BYTES;
        let down_packed_off = off;
        off += DOWN_PACKED_BYTES;
        let down_scale_off = off;

        Ok(Self {
            mmap,
            gate_packed_off,
            gate_scale_off,
            up_packed_off,
            up_scale_off,
            down_packed_off,
            down_scale_off,
        })
    }

    /// `gate_proj.weight_packed` as a flat byte slice. Each i32 holds 8
    /// nibbles strided across its 32 bits. As LE bytes that means byte i
    /// of the int32 holds columns `2i` (low nibble) and `2i+1` (high
    /// nibble). The kernel reads bytes directly.
    pub fn gate_packed_bytes(&self) -> &[u8] {
        &self.mmap[self.gate_packed_off..self.gate_packed_off + GATE_PACKED_BYTES]
    }

    /// `gate_proj.weight_scale` as raw bf16 bits (u16 little-endian).
    pub fn gate_scale_bits(&self) -> &[u8] {
        &self.mmap[self.gate_scale_off..self.gate_scale_off + GATE_SCALE_BYTES]
    }

    pub fn up_packed_bytes(&self) -> &[u8] {
        &self.mmap[self.up_packed_off..self.up_packed_off + UP_PACKED_BYTES]
    }

    pub fn up_scale_bits(&self) -> &[u8] {
        &self.mmap[self.up_scale_off..self.up_scale_off + UP_SCALE_BYTES]
    }

    pub fn down_packed_bytes(&self) -> &[u8] {
        &self.mmap[self.down_packed_off..self.down_packed_off + DOWN_PACKED_BYTES]
    }

    pub fn down_scale_bits(&self) -> &[u8] {
        &self.mmap[self.down_scale_off..self.down_scale_off + DOWN_SCALE_BYTES]
    }
}

/// Helper: convert a bf16 raw u16 to f32.
#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    // bf16 = sign(1) + exp(8) + mantissa(7); same exponent as f32, but
    // mantissa is the top 7 bits of f32's 23-bit mantissa. So shifting
    // left by 16 lines it up.
    f32::from_bits((bits as u32) << 16)
}

/// Helper: convert an f32 to bf16 raw u16 bits, round-to-nearest-even
/// (matching the rounding `half::bf16::from_f32` would do). Inlined here
/// so hot per-token KV write loops don't depend on the `half` crate at
/// every callsite. NaNs are preserved (mantissa bit forced nonzero).
#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN — keep mantissa nonzero so the round-trip stays a NaN
        // rather than collapsing to ±inf when we shift back.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}
