//! Qwen2-VL's patch layout, shared by the packages that feed a Qwen-VL vision
//! tower `pixel_values` (`vla_jepa_inputs`, `xr0`): Intel's runtime
//! `xr0.py` `_patchify_pixel_grid` at `e3dd944`.

/// One image's patch grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchGeometry {
    /// `[height, width]` of the image patchified; multiples of `patch * merge`.
    pub hw: [usize; 2],
    pub patch: usize,
    pub merge: usize,
    pub temporal: usize,
}

impl PatchGeometry {
    /// `pixel_values` rows per image.
    pub fn patches(&self) -> usize {
        (self.hw[0] / self.patch) * (self.hw[1] / self.patch)
    }

    /// `pixel_values` row width: `3 * temporal * patch²`.
    pub fn feature(&self) -> usize {
        3 * self.temporal * self.patch * self.patch
    }

    /// `<|image_pad|>` tokens the prompt carries per image.
    pub fn image_pads(&self) -> usize {
        self.patches() / (self.merge * self.merge)
    }

    /// Patchify one CHW image: tiled `temporal` times, reshaped to
    /// `(1, tp, C, gh/m, m, p, gw/m, m, p)` and transposed by
    /// `(0, 3, 6, 4, 7, 2, 1, 5, 8)`, appended as `gh * gw` rows.
    pub fn patchify(&self, chw: &[f32], out: &mut Vec<f32>) {
        let [vh, vw] = self.hw;
        let (p, m, tp) = (self.patch, self.merge, self.temporal);
        let (gh, gw) = (vh / p, vw / p);
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let (y0, x0) = ((bh * m + mh) * p, (bw * m + mw) * p);
                        for c in 0..3 {
                            for _ in 0..tp {
                                for py in 0..p {
                                    let at = c * vh * vw + (y0 + py) * vw + x0;
                                    out.extend_from_slice(&chw[at..at + p]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 9-D reshape and transpose, done the long way.
    fn patchify_reference(
        chw: &[f32],
        h: usize,
        w: usize,
        p: usize,
        m: usize,
        tp: usize,
    ) -> Vec<f32> {
        let (gh, gw) = (h / p, w / p);
        let feature = 3 * tp * p * p;
        let mut out = vec![0f32; gh * gw * feature];
        // Source index of (t, c, bh, mh, py, bw, mw, px); destination by the transposed order.
        for t in 0..tp {
            for c in 0..3 {
                for bh in 0..gh / m {
                    for mh in 0..m {
                        for py in 0..p {
                            for bw in 0..gw / m {
                                for mw in 0..m {
                                    for px in 0..p {
                                        let y = (bh * m + mh) * p + py;
                                        let x = (bw * m + mw) * p + px;
                                        let row = ((bh * (gw / m) + bw) * m + mh) * m + mw;
                                        let col = ((c * tp + t) * p + py) * p + px;
                                        out[row * feature + col] = chw[c * h * w + y * w + x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn patchify_is_the_qwen_reshape_and_transpose() {
        // Not square, so rows and columns cannot be swapped unnoticed.
        let geo = PatchGeometry {
            hw: [8, 12],
            patch: 2,
            merge: 2,
            temporal: 2,
        };
        assert_eq!(
            (geo.patches(), geo.feature(), geo.image_pads()),
            (24, 24, 6)
        );
        let chw: Vec<f32> = (0..3 * 8 * 12).map(|i| i as f32).collect();
        let mut got = Vec::new();
        geo.patchify(&chw, &mut got);
        assert_eq!(got, patchify_reference(&chw, 8, 12, 2, 2, 2));
    }
}
