//! Learned relative-position bias — `InklingRelativeLogits`.
//!
//! `proj` is a trained bank of bias-vs-distance profiles `[d_rel, extent]`.
//! Each query token's per-head relative state `r_h` (`[d_rel]`, the `r_proj`
//! output) mixes them into one bias per backward distance:
//!
//! ```text
//!   bias_h(dist) = Σ_i r_h[i] · proj[i, dist]     for 0 <= dist < extent
//!                = 0                                otherwise (and for dist < 0)
//! ```
//!
//! HF materialises `relative_states @ proj` (`[.., heads, extent]`) then
//! gathers by `clamp(q_pos - k_pos, 0, extent-1)` and zeroes out-of-range
//! entries; [`RelPos::profile`] is that per-head row (computed once per query
//! head, indexed by distance in the score loop) and [`RelPos::bias`] the
//! single-distance form — both accumulate over `i` in the same order.
//! Sliding layers use `extent == window`; global layers the manifest
//! `rel_extent` (1024). f32 throughout.

pub struct RelPos {
    /// `[d_rel, extent]` row-major.
    pub proj: Vec<f32>,
    pub d_rel: usize,
    pub extent: usize,
}

impl RelPos {
    pub fn new(proj: Vec<f32>, d_rel: usize, extent: usize) -> Self {
        assert_eq!(
            proj.len(),
            d_rel * extent,
            "RelPos: proj len != d_rel * extent"
        );
        assert!(d_rel > 0 && extent > 0, "RelPos: empty dims");
        Self {
            proj,
            d_rel,
            extent,
        }
    }

    /// The bias-vs-distance row for one head's relative state `r` (`[d_rel]`):
    /// `out[dist] = Σ_i r[i] · proj[i, dist]`, `[extent]`. Distances at or
    /// beyond `extent` are zero by definition (not stored).
    pub fn profile(&self, r: &[f32]) -> Vec<f32> {
        assert_eq!(r.len(), self.d_rel, "RelPos::profile: r len != d_rel");
        let mut out = vec![0.0f32; self.extent];
        for (&ri, row) in r.iter().zip(self.proj.chunks_exact(self.extent)) {
            for (o, &pv) in out.iter_mut().zip(row) {
                *o += ri * pv;
            }
        }
        out
    }

    /// Bias for one backward distance (0 for `dist >= extent`).
    pub fn bias(&self, r: &[f32], dist: usize) -> f32 {
        assert_eq!(r.len(), self.d_rel, "RelPos::bias: r len != d_rel");
        if dist >= self.extent {
            return 0.0;
        }
        let mut acc = 0.0f32;
        for (i, &ri) in r.iter().enumerate() {
            acc += ri * self.proj[i * self.extent + dist];
        }
        acc
    }

    /// One query row of biases: for each key position `j` in `key_positions`,
    /// `bias(q_pos - j)`, or 0 when `j > q_pos` (a future key — causality is
    /// the mask's job, the bias just follows HF and zeroes it).
    pub fn row(&self, r: &[f32], q_pos: usize, key_positions: &[usize]) -> Vec<f32> {
        let prof = self.profile(r);
        key_positions
            .iter()
            .map(|&j| {
                if j > q_pos {
                    0.0
                } else {
                    prof.get(q_pos - j).copied().unwrap_or(0.0)
                }
            })
            .collect()
    }
}
