//! Per-channel `|silu(gate[c])| / max_j |silu(gate[j])|` histogram
//! capture for offline CHESS-style threshold calibration.
//!
//! The runtime accumulates a fixed-bin histogram per (layer_id,
//! channel_id) over the lifetime of a worker. After enough tokens
//! have been processed, the histogram is dumped to disk and the
//! [`crate::bin::calibrate_ffn_thresholds`] tool converts it into
//! per-channel quantile thresholds.
//!
//! # Why fixed-bin histograms (not reservoir / t-digest)?
//!
//! - **Memory**: 60 layers × 2048 channels × `N_BINS=128` × `u32` =
//!   60 MiB at K2.6 dims. Bounded; allocated once at startup.
//! - **Update cost**: ~10 ns / sample on warm cache. At 2048 channels
//!   × 8 routed experts × per-token, that's ~160 µs / token — negligible
//!   vs the 10+ s/token K2.6 dispatch cost.
//! - **Quantile precision**: 128 linear bins over `[0, 1]` gives sub-1%
//!   resolution on the per-channel quantile. CHESS's empirical
//!   precision target is the 50% quantile ± a few percent; this is
//!   plenty.
//!
//! The ratio's domain is `[0, 1]` by construction (numerator ≤
//! denominator), so a fixed-range histogram is the right tool. A
//! reservoir sampler would give us perfect quantiles but at 10 k
//! samples × 4 B × 60 × 2048 = 4.9 GiB of resident memory; t-digest
//! would compress better but with a much more complex update path and
//! a Rust dep we don't currently carry. Histogram bins are the
//! lowest-friction option that hits the precision target.
//!
//! # On-disk format
//!
//! One `layer_<lid>.bin` file per covered layer in the capture dir.
//! Native-endian binary (we never share these files between hosts; the
//! calibration tool runs on the same machine as the capture).
//!
//! ```text
//! Header (32 bytes):
//!   magic:           [u8; 16]  // CSCD_FFN_CAP_v1\0
//!   layer_id:        u32 LE
//!   n_intermediate:  u32 LE
//!   n_bins:          u32 LE
//!   total_samples:   u32 LE    // sum across all channels' bins
//! Body:
//!   counts:          [u32 LE; n_intermediate * n_bins]
//!                    // channel-major: channel c at offset c*n_bins
//! ```
//!
//! See [`LayerCapture::write_to`] for the writer and the matching
//! reader in `bin/calibrate_ffn_thresholds.rs`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use thiserror::Error;

/// Magic bytes prefixing every layer capture file. Bump if the layout
/// changes (also bump `MAGIC` in the calibration tool).
pub const CAPTURE_FILE_MAGIC: &[u8; 16] = b"CSCD_FFN_CAP_v1\0";

/// Bins per channel in the histogram. 128 bins → bin width 1/128 ≈
/// 0.0078, which gives sub-1% precision on the per-channel quantile.
pub const N_BINS: usize = 128;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// One layer's per-channel ratio histograms.
///
/// Each channel gets `N_BINS` u32 counts indexed by
/// `floor(ratio * N_BINS)` (clamped to `N_BINS - 1`). Counts are
/// **not** Atomic — updates run on the single dispatch thread that
/// owns the [`GateCaptureState`] mutex.
pub struct LayerCapture {
    pub layer_id: u32,
    pub n_intermediate: usize,
    /// Flat counts: channel `c`, bin `b` at index `c * N_BINS + b`.
    pub counts: Vec<u32>,
    pub total_samples: u64,
}

impl LayerCapture {
    pub fn new(layer_id: u32, n_intermediate: usize) -> Self {
        Self {
            layer_id,
            n_intermediate,
            counts: vec![0u32; n_intermediate * N_BINS],
            total_samples: 0,
        }
    }

    /// Record one (silu_gate, max_abs) snapshot from a single expert
    /// call. Inactive (max_abs == 0) snapshots are skipped — they
    /// produce a degenerate ratio of 0/0 and don't say anything
    /// meaningful about channel-magnitude distribution.
    pub fn record(&mut self, silu_gate: &[f32], max_abs: f32) {
        debug_assert_eq!(silu_gate.len(), self.n_intermediate);
        if max_abs <= 0.0 {
            return;
        }
        let inv = 1.0 / max_abs;
        let n_bins_f = N_BINS as f32;
        for (c, &v) in silu_gate.iter().enumerate() {
            let ratio = (v.abs() * inv).clamp(0.0, 1.0);
            // The clamp above guarantees ratio ∈ [0, 1]; multiplying
            // by N_BINS and truncating yields a bin index in [0,
            // N_BINS]; we additionally clamp to [0, N_BINS-1] so the
            // exact-1.0 ratio (the max-magnitude channel) maps to
            // the last bin rather than overflowing.
            let bin = (ratio * n_bins_f) as usize;
            let bin = bin.min(N_BINS - 1);
            self.counts[c * N_BINS + bin] = self.counts[c * N_BINS + bin].saturating_add(1);
        }
        self.total_samples += 1;
    }

    /// Serialise this layer to `dst` in the format documented at the
    /// module level.
    pub fn write_to(&self, dst: &mut impl Write) -> Result<(), CaptureError> {
        let mut header = [0u8; 32];
        header[..16].copy_from_slice(CAPTURE_FILE_MAGIC);
        header[16..20].copy_from_slice(&self.layer_id.to_le_bytes());
        header[20..24].copy_from_slice(&(self.n_intermediate as u32).to_le_bytes());
        header[24..28].copy_from_slice(&(N_BINS as u32).to_le_bytes());
        // total_samples is u64 in memory but caps at u32 on disk —
        // capture runs are bounded by token count, and even at 1k
        // tokens × 384 experts × 60 layers we're at ~23 M, well
        // within u32 range. If a capture run somehow exceeds u32::MAX
        // samples we saturate the disk field; the body counts are
        // still authoritative.
        header[28..32]
            .copy_from_slice(&(self.total_samples.min(u32::MAX as u64) as u32).to_le_bytes());
        dst.write_all(&header)?;
        // Body: pack the u32 counts as LE bytes. `bytemuck` would let
        // us cast the slice in one shot, but we don't have it as a
        // workspace dep and the per-channel cost is dominated by the
        // outer file I/O. Loop and write 4 bytes at a time.
        let mut buf = Vec::with_capacity(self.counts.len() * 4);
        for &c in &self.counts {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        dst.write_all(&buf)?;
        Ok(())
    }
}

/// Per-worker capture state: one [`LayerCapture`] per covered layer.
///
/// The dispatcher calls [`Self::record`] inside its hot loop; the
/// `Mutex` wraps the per-layer map so multiple dispatcher threads
/// (rayon, etc.) can update concurrently without races. In practice
/// the K2.6 dispatch path is serial across experts within a token,
/// so contention is negligible — this is a future-proofing guard.
pub struct GateCaptureState {
    inner: Mutex<GateCaptureInner>,
    n_intermediate: usize,
    capture_dir: PathBuf,
}

struct GateCaptureInner {
    layers: BTreeMap<u32, LayerCapture>,
}

impl GateCaptureState {
    pub fn new(capture_dir: PathBuf, n_intermediate: usize) -> Self {
        Self {
            inner: Mutex::new(GateCaptureInner {
                layers: BTreeMap::new(),
            }),
            n_intermediate,
            capture_dir,
        }
    }

    pub fn capture_dir(&self) -> &Path {
        &self.capture_dir
    }

    /// Record one expert call's per-channel `silu(gate)` snapshot.
    /// `max_abs` is the per-token max of `|silu(gate)|` — the same
    /// number the runtime uses to build the global-τ mask.
    pub fn record(&self, layer_id: u32, silu_gate: &[f32], max_abs: f32) {
        let mut g = self.inner.lock();
        let entry = g
            .layers
            .entry(layer_id)
            .or_insert_with(|| LayerCapture::new(layer_id, self.n_intermediate));
        entry.record(silu_gate, max_abs);
    }

    /// Total samples recorded across every covered layer. Useful for
    /// "is this capture run big enough yet" instrumentation. One
    /// sample = one expert call observed.
    pub fn total_samples(&self) -> u64 {
        let g = self.inner.lock();
        g.layers.values().map(|l| l.total_samples).sum()
    }

    /// How many distinct layers have been recorded against.
    pub fn n_layers(&self) -> usize {
        self.inner.lock().layers.len()
    }

    /// Persist every covered layer to `self.capture_dir`. Creates the
    /// directory if it doesn't exist. Returns `(n_layers_written,
    /// total_samples)`.
    pub fn dump(&self) -> Result<(usize, u64), CaptureError> {
        std::fs::create_dir_all(&self.capture_dir)?;
        let g = self.inner.lock();
        let mut total = 0u64;
        let mut n_layers = 0usize;
        for layer in g.layers.values() {
            let path = self
                .capture_dir
                .join(format!("layer_{:04}.bin", layer.layer_id));
            let tmp = self
                .capture_dir
                .join(format!("layer_{:04}.bin.tmp", layer.layer_id));
            // Atomic-rename write so a SIGKILL mid-dump leaves the
            // previous file (if any) intact.
            {
                let mut f = std::fs::File::create(&tmp)?;
                layer.write_to(&mut f)?;
                f.sync_all().ok();
            }
            std::fs::rename(&tmp, &path)?;
            total += layer.total_samples;
            n_layers += 1;
        }
        Ok((n_layers, total))
    }

    /// Test-only: drain the current state into an in-memory snapshot.
    /// Mostly useful for unit tests that want to inspect counts
    /// directly without going through the on-disk path.
    #[cfg(test)]
    fn snapshot_layers(&self) -> BTreeMap<u32, LayerCapture> {
        let mut g = self.inner.lock();
        std::mem::take(&mut g.layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// `LayerCapture::record` skips snapshots where `max_abs == 0`
    /// (numerically degenerate). Bin counts stay zero; `total_samples`
    /// stays zero.
    #[test]
    fn record_skips_zero_max_snapshots() {
        let mut l = LayerCapture::new(0, 4);
        l.record(&[0.0, 0.0, 0.0, 0.0], 0.0);
        assert_eq!(l.total_samples, 0);
        assert!(l.counts.iter().all(|&c| c == 0));
    }

    /// A snapshot where one channel is max-magnitude and the others
    /// are zero deposits one count in the last bin of that channel
    /// (ratio = 1.0) and one count in the first bin of every other
    /// channel (ratio = 0.0).
    #[test]
    fn record_bins_max_and_zero_correctly() {
        let mut l = LayerCapture::new(0, 3);
        l.record(&[0.0, 0.0, 5.0], 5.0);
        assert_eq!(l.total_samples, 1);
        // Channel 0: bin 0 should hold the only count.
        assert_eq!(l.counts[0], 1);
        // Channel 1: same.
        assert_eq!(l.counts[N_BINS], 1);
        // Channel 2: last bin (index N_BINS - 1) should hold the
        // count, since 5.0 / 5.0 = 1.0 maps to bin N_BINS - 1 after
        // clamp.
        assert_eq!(l.counts[2 * N_BINS + (N_BINS - 1)], 1);
        // No other counts.
        let sum: u32 = l.counts.iter().sum();
        assert_eq!(sum, 3);
    }

    /// Many snapshots: empirical CDF of the recorded ratios should
    /// land approximately where we expect. Channel 0 always emits
    /// ratio 0.5; the median bin should be at index N_BINS/2.
    #[test]
    fn many_snapshots_concentrate_at_known_bin() {
        let mut l = LayerCapture::new(0, 2);
        for _ in 0..100 {
            // silu_gate = [0.5, 1.0]; max = 1.0 → ratios [0.5, 1.0].
            l.record(&[0.5, 1.0], 1.0);
        }
        assert_eq!(l.total_samples, 100);
        // Channel 0: all 100 samples in bin floor(0.5 * 128) = 64.
        assert_eq!(l.counts[64], 100);
        let ch0_total: u32 = l.counts[..N_BINS].iter().sum();
        assert_eq!(ch0_total, 100);
        // Channel 1: all 100 samples in bin N_BINS - 1.
        assert_eq!(l.counts[N_BINS + (N_BINS - 1)], 100);
    }

    /// `GateCaptureState::dump` writes one file per covered layer
    /// with the documented magic + dimensions.
    #[test]
    fn dump_writes_one_file_per_layer() {
        let dir = tempdir().expect("tempdir");
        let st = GateCaptureState::new(dir.path().to_owned(), 4);
        st.record(0, &[0.1, 0.2, 0.3, 0.4], 0.4);
        st.record(5, &[0.0, 1.0, 0.5, 0.25], 1.0);
        let (n_layers, total) = st.dump().expect("dump");
        assert_eq!(n_layers, 2);
        assert_eq!(total, 2);

        let p0 = dir.path().join("layer_0000.bin");
        let p5 = dir.path().join("layer_0005.bin");
        assert!(p0.exists());
        assert!(p5.exists());

        // Header sanity-check.
        let bytes = std::fs::read(&p0).expect("read");
        assert_eq!(&bytes[..16], &CAPTURE_FILE_MAGIC[..]);
        let lid = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let ni = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let nb = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let ts = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
        assert_eq!(lid, 0);
        assert_eq!(ni, 4);
        assert_eq!(nb as usize, N_BINS);
        assert_eq!(ts, 1);
        // Body length matches dim claim.
        assert_eq!(bytes.len(), 32 + 4 * 4 * N_BINS);
    }

    /// Multiple records into the same layer are additive across bins.
    #[test]
    fn record_accumulates_within_layer() {
        let dir = tempdir().expect("tempdir");
        let st = GateCaptureState::new(dir.path().to_owned(), 2);
        for _ in 0..3 {
            st.record(7, &[0.0, 1.0], 1.0);
        }
        assert_eq!(st.total_samples(), 3);
        let snap = st.snapshot_layers();
        let layer = snap.get(&7).expect("layer 7");
        assert_eq!(layer.total_samples, 3);
        // Channel 0: 3 hits in bin 0; channel 1: 3 hits in last bin.
        assert_eq!(layer.counts[0], 3);
        assert_eq!(layer.counts[N_BINS + (N_BINS - 1)], 3);
    }
}
