//! Causal depthwise short convolution — `InklingShortConvolution`.
//!
//! torch semantics: `conv1d(u, w, padding=K-1, groups=C, bias=False)[..seq]`,
//! then the module returns `out + u` (the residual is INSIDE the module). Per
//! channel `c` at position `p` with kernel width `K`:
//!
//! ```text
//!   out[c, p] = Σ_{j<K} w[c, j] · u[c, p - (K-1) + j]      (u[c, <0] = 0)
//!   y[c, p]   = out[c, p] + u[c, p]
//! ```
//!
//! so tap `K-1` multiplies the current input and tap `0` the oldest. HF keeps
//! these modules in fp32 (`_keep_in_fp32_modules_strict`), so everything here
//! is f32 with no rounding.
//!
//! Decode state = the last `K-1` raw inputs per channel (HF `conv_states`),
//! held in a ring with [`DEFAULT_REWIND`] extra rows so [`ShortConv::truncate`]
//! can rewind a rejected speculative draft without re-running the prefix.
//! [`ShortConv::prefill`] is bit-identical to `t` sequential decodes (same
//! per-position code path).
//!
//! # Ring validity
//!
//! Position `p` is written to row `p % hist` when it is consumed. `hwm` (the
//! write high-water mark) is one past the furthest position written since the
//! last `reset` / `restore`; every position `< hwm` has been written, so row
//! `q % hist` holds position `q` exactly when `q` is the newest position of its
//! row class, i.e. `q >= hwm - hist` (all of `[0, hwm)` while the ring has not
//! wrapped). A decode at `len` reads positions `[len - (K-1), len)`, which is
//! inside that window iff `hwm - len <= rewind` — the bound `truncate` enforces
//! (against `hwm`, not the already-lowered `len`, so it holds across any
//! sequence of truncates and restores). Because reads never leave the window
//! and `reset` empties it (`hwm = 0`), the ring's contents outside the window
//! are never observed and `reset` does not need to zero them.

use super::DEFAULT_REWIND;

/// One depthwise causal conv with its decode history.
pub struct ShortConv {
    /// Kernel `[C, K]` (row-major: channel `c`'s taps are `w[c*K .. c*K+K]`).
    pub w: Vec<f32>,
    /// Channels.
    pub c: usize,
    /// Kernel width (4 for Inkling).
    pub k: usize,
    rewind: usize,
    /// Ring rows `= (K - 1) + rewind` (at least 1); position `p` lives in row `p % hist`.
    hist: usize,
    /// `[hist, C]` raw inputs.
    ring: Vec<f32>,
    /// Positions consumed so far (the next input is position `len`).
    len: usize,
    /// Write high-water mark: one past the furthest position written since the
    /// last `reset` / `restore` (`>= len`; see the module docs).
    hwm: usize,
}

/// A saved conv history for prefix caching / restore: the raw inputs at
/// positions `[first, len)` (position-major, `c` floats each). Covers the whole
/// valid ring window, so `truncate` behaves identically after a restore.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvState {
    len: usize,
    first: usize,
    rows: Vec<f32>,
}

impl ConvState {
    /// Sequence length this state was taken at.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Payload size in bytes.
    pub fn bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<f32>()
    }
}

impl ShortConv {
    /// `w` is `[c, k]` row-major; history sized for [`DEFAULT_REWIND`].
    pub fn new(w: Vec<f32>, c: usize, k: usize) -> Self {
        Self::with_rewind(w, c, k, DEFAULT_REWIND)
    }

    /// Like [`Self::new`] with an explicit rewind slack (how many positions
    /// [`Self::truncate`] may roll back).
    pub fn with_rewind(w: Vec<f32>, c: usize, k: usize, rewind: usize) -> Self {
        assert!(k >= 1, "ShortConv: kernel width must be >= 1");
        assert!(c >= 1, "ShortConv: channels must be >= 1");
        assert_eq!(w.len(), c * k, "ShortConv: weight len != c * k");
        let hist = ((k - 1) + rewind).max(1);
        Self {
            w,
            c,
            k,
            rewind,
            hist,
            ring: vec![0.0; hist * c],
            len: 0,
            hwm: 0,
        }
    }

    /// Positions consumed so far.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rewind slack this history supports.
    pub fn rewind(&self) -> usize {
        self.rewind
    }

    /// Bytes held by the history ring (excludes the kernel).
    pub fn cache_bytes(&self) -> usize {
        self.ring.len() * std::mem::size_of::<f32>()
    }

    /// Oldest position whose input is still in the ring (see the module docs).
    #[inline]
    fn oldest_valid(&self) -> usize {
        self.hwm.saturating_sub(self.hist)
    }

    /// One position: `out = conv(u) + u` at position `self.len`, then record
    /// `u` and advance. The single code path both `decode` and `prefill` use —
    /// summation order per channel is tap 0 .. tap K-1, then `+ u`.
    fn step(&mut self, u: &[f32], out: &mut [f32]) {
        let (c, k, hist) = (self.c, self.k, self.hist);
        debug_assert_eq!(u.len(), c);
        debug_assert_eq!(out.len(), c);
        let p = self.len;
        debug_assert!(
            p.saturating_sub(k - 1) >= self.oldest_valid(),
            "ShortConv ring invariant broken: position {p} needs inputs older than {}",
            self.oldest_valid()
        );
        out.fill(0.0);
        for j in 0..k {
            // Tap j reads position p - (K-1) + j, i.e. `back` positions ago.
            let back = k - 1 - j;
            if back > p {
                continue; // zero padding before position 0
            }
            let src: &[f32] = if back == 0 {
                u
            } else {
                let row = (p - back) % hist;
                &self.ring[row * c..(row + 1) * c]
            };
            for ((o, &s), wrow) in out.iter_mut().zip(src).zip(self.w.chunks_exact(k)) {
                *o += wrow[j] * s;
            }
        }
        for (o, &ui) in out.iter_mut().zip(u) {
            *o += ui;
        }
        let row = p % hist;
        self.ring[row * c..(row + 1) * c].copy_from_slice(u);
        self.len = p + 1;
        self.hwm = self.hwm.max(p + 1);
    }

    /// Decode one position: `u` is `[C]`; returns `conv(u) + u` (`[C]`) and
    /// appends `u` to the history.
    pub fn decode(&mut self, u: &[f32]) -> Vec<f32> {
        assert_eq!(u.len(), self.c, "ShortConv::decode: input len != channels");
        let mut out = vec![0.0f32; self.c];
        self.step(u, &mut out);
        out
    }

    /// Prefill `t` positions: `u` is `[t, C]` row-major; returns `[t, C]`.
    /// Bit-identical to calling [`Self::decode`] on each row in order.
    pub fn prefill(&mut self, u: &[f32], t: usize) -> Vec<f32> {
        assert_eq!(
            u.len(),
            t * self.c,
            "ShortConv::prefill: input len != t * channels"
        );
        let c = self.c;
        let mut out = vec![0.0f32; t * c];
        for (urow, orow) in u.chunks_exact(c).zip(out.chunks_exact_mut(c)) {
            self.step(urow, orow);
        }
        out
    }

    /// Forget everything (new sequence). O(1): the ring is not zeroed — with
    /// `hwm = 0` no row is inside the valid window, and a decode only ever
    /// reads rows written since (module docs).
    pub fn reset(&mut self) {
        self.len = 0;
        self.hwm = 0;
    }

    /// Roll back to `len` positions (spec-decode reject). O(1): the ring rows
    /// past `len` are stale and get overwritten as decode resumes. Panics if
    /// the rewind exceeds the slack — measured from the write high-water mark
    /// (`hwm - len > rewind`), so consecutive truncates cannot creep past it:
    /// the inputs the next position would need have been overwritten by the
    /// discarded positions.
    pub fn truncate(&mut self, len: usize) {
        assert!(
            len <= self.len,
            "ShortConv::truncate({len}) beyond current len {}",
            self.len
        );
        let stale = self.hwm - len;
        assert!(
            stale <= self.rewind || self.hwm <= self.hist,
            "ShortConv::truncate({len}): {stale} positions were written past it (high-water \
             mark {}) which exceeds the rewind slack {} — the ring no longer holds the inputs \
             position {len} needs (raise the conv's rewind or cap the speculative draft length)",
            self.hwm,
            self.rewind
        );
        self.len = len;
    }

    /// Copy out the valid history window: positions `[hwm - hist, len)` (from
    /// 0 while the ring has not wrapped) — every row the ring still holds
    /// for a position below `len`, so a restore followed by an in-slack
    /// `truncate` reads exactly what this conv would have.
    pub fn snapshot(&self) -> ConvState {
        let (c, hist) = (self.c, self.hist);
        let first = self.oldest_valid().min(self.len);
        let mut rows = Vec::with_capacity((self.len - first) * c);
        for p in first..self.len {
            let row = p % hist;
            rows.extend_from_slice(&self.ring[row * c..(row + 1) * c]);
        }
        ConvState {
            len: self.len,
            first,
            rows,
        }
    }

    /// Restore a snapshot; replaces the length, the history window and the
    /// high-water mark (so the rewind bound after a restore is the one the
    /// snapshot's window actually supports). A snapshot from a conv with a
    /// larger rewind is trimmed to this ring; one with a smaller rewind
    /// restores what it holds.
    pub fn restore(&mut self, s: &ConvState) {
        let (c, hist) = (self.c, self.hist);
        assert_eq!(
            s.rows.len(),
            (s.len - s.first) * c,
            "ShortConv::restore: snapshot channel count mismatch"
        );
        let start = s.first.max(s.len.saturating_sub(hist));
        for p in start..s.len {
            let row = p % hist;
            let src = &s.rows[(p - s.first) * c..(p - s.first + 1) * c];
            self.ring[row * c..(row + 1) * c].copy_from_slice(src);
        }
        self.len = s.len;
        // Valid window = [start, len): `hwm - hist == start`, or everything
        // from 0 when the restored rows fit without wrapping.
        self.hwm = if start == 0 { s.len } else { start + hist };
    }
}
