//! VLA-JEPA host inputs, ported from Intel's `physicalai` runtime at `e3dd944`
//! (`preprocessors/vla_jepa.py`, and `xr0.py` `_patchify_pixel_grid`): camera
//! frames to Qwen3-VL `pixel_values`, and the embodied-token positions.
//!
//! Frames arrive at `image_resolution`, so Intel's first (area) resize is the
//! identity and is skipped.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::Image;

use super::cameras::CameraLayout;
use super::manifest::VlaJepaInputsStep;

/// Bicubic `a` of torchvision's antialiased resize.
const BICUBIC_A: f64 = -0.5;

/// One output index of a separable resize: the span of inputs it reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Taps {
    pub start: usize,
    pub weights: Vec<f32>,
}

/// Intel's `_bicubic_weights` (torchvision `resize(antialias=True)`): built in
/// f64, each row normalized, stored f32. Only a row's non-zero span is kept;
/// the zeros outside it add nothing to the sums.
pub fn bicubic_taps(size_in: usize, size_out: usize) -> Vec<Taps> {
    let scale = size_in as f64 / size_out as f64;
    let stretch = scale.max(1.0);
    let a = BICUBIC_A;
    (0..size_out)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            // Python's `int()` truncates toward zero, as `as` does.
            let start = ((center - 2.0 * stretch + 0.5) as i64).max(0) as usize;
            let end = ((center + 2.0 * stretch + 0.5) as i64).min(size_in as i64) as usize;
            let kernel: Vec<f64> = (start..end)
                .map(|j| {
                    let x = ((j as f64 - center + 0.5) / stretch).abs();
                    if x < 1.0 {
                        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                    } else if x < 2.0 {
                        (((x - 5.0) * x + 8.0) * x - 4.0) * a
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = kernel.iter().sum();
            Taps {
                start,
                weights: kernel.iter().map(|k| (k / sum) as f32).collect(),
            }
        })
        .collect()
}

/// The precomputed geometry of one package's `vla_jepa_inputs`.
#[derive(Clone, Debug)]
pub struct VlaJepaHost {
    rows: Vec<Taps>,
    cols: Vec<Taps>,
    in_hw: [usize; 2],
    vision_hw: [usize; 2],
    mean: [f32; 3],
    std: [f32; 3],
    patch: usize,
    merge: usize,
    temporal: usize,
    embodied_token_id: i64,
    num_embodied: usize,
}

impl VlaJepaHost {
    pub fn new(step: &VlaJepaInputsStep) -> Self {
        let in_hw = step.image_resolution.map(|d| d as usize);
        let vision_hw = step.vision_resolution.map(|d| d as usize);
        Self {
            rows: bicubic_taps(in_hw[0], vision_hw[0]),
            cols: bicubic_taps(in_hw[1], vision_hw[1]),
            in_hw,
            vision_hw,
            mean: step.image_mean,
            std: step.image_std,
            patch: step.patch_size,
            merge: step.merge_size,
            temporal: step.temporal_patch_size,
            embodied_token_id: step.embodied_token_id,
            num_embodied: step.num_embodied_tokens,
        }
    }

    /// `pixel_values` rows per camera.
    pub fn patches(&self) -> usize {
        (self.vision_hw[0] / self.patch) * (self.vision_hw[1] / self.patch)
    }

    /// `pixel_values` row width: `3 * temporal_patch_size * patch_size²`.
    pub fn feature(&self) -> usize {
        3 * self.temporal * self.patch * self.patch
    }

    pub fn num_embodied(&self) -> usize {
        self.num_embodied
    }

    /// Indices of the embodied token in `ids`; anything but exactly
    /// `num_embodied_tokens` of them is refused (a task that spells the token).
    pub fn embodied_positions(&self, ids: &[i64]) -> EngineResult<Vec<i64>> {
        let positions: Vec<i64> = ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.embodied_token_id)
            .map(|(i, _)| i as i64)
            .collect();
        if positions.len() != self.num_embodied {
            return Err(EngineError::InvalidConfig(format!(
                "the prompt has {} embodied-action tokens (id {}); the package needs exactly {}",
                positions.len(),
                self.embodied_token_id,
                self.num_embodied
            )));
        }
        Ok(positions)
    }

    /// `pixel_values [n_cameras * patches, feature]` f32, cameras in slot order.
    pub fn pixel_values(
        &self,
        layout: &CameraLayout,
        images: &[(String, Image)],
    ) -> EngineResult<Vec<f32>> {
        let [h, w] = self.in_hw;
        let mut out = Vec::with_capacity(layout.slots.len() * self.patches() * self.feature());
        for slot in &layout.slots {
            let Some(i) = slot else {
                return Err(EngineError::InvalidConfig(
                    "vla_jepa_inputs has no dummy camera slots".into(),
                ));
            };
            let name = &layout.names[*i];
            let Some((_, img)) = images.iter().find(|(n, _)| n == name) else {
                return Err(EngineError::InvalidConfig(format!(
                    "missing camera {name:?}"
                )));
            };
            if img.height as usize != h || img.width as usize != w || img.data.len() != h * w * 3 {
                return Err(EngineError::InvalidConfig(format!(
                    "camera {name:?} must be {h}x{w} RGB ({} bytes)",
                    h * w * 3
                )));
            }
            let chw = self.normalized(&img.data);
            self.patchify(&chw, &mut out);
        }
        Ok(out)
    }

    /// uint8 HWC → `/255` → bicubic to the vision resolution → `(x − mean) / std`, CHW.
    ///
    /// Each product is summed in f64 and rounded to f32 once, as numpy's f32
    /// matmul result is; the row pass is rounded to f32 before the column pass.
    fn normalized(&self, hwc: &[u8]) -> Vec<f32> {
        let w = self.in_hw[1];
        let [vh, vw] = self.vision_hw;
        let mut out = Vec::with_capacity(3 * vh * vw);
        let mut tmp = vec![0f32; vh * w];
        for c in 0..3 {
            let px = |y: usize, x: usize| f64::from(f32::from(hwc[(y * w + x) * 3 + c]) / 255.0);
            for (i, t) in self.rows.iter().enumerate() {
                for x in 0..w {
                    let acc: f64 = t
                        .weights
                        .iter()
                        .enumerate()
                        .map(|(k, &wt)| f64::from(wt) * px(t.start + k, x))
                        .sum();
                    tmp[i * w + x] = acc as f32;
                }
            }
            for i in 0..vh {
                let row = &tmp[i * w..(i + 1) * w];
                for t in &self.cols {
                    let acc: f64 = t
                        .weights
                        .iter()
                        .enumerate()
                        .map(|(k, &wt)| f64::from(wt) * f64::from(row[t.start + k]))
                        .sum();
                    out.push((acc as f32 - self.mean[c]) / self.std[c]);
                }
            }
        }
        out
    }

    /// Qwen2-VL's patchify of one CHW image: tiled `temporal_patch_size` times,
    /// reshaped to `(1, tp, C, gh/m, m, p, gw/m, m, p)` and transposed by
    /// `(0, 3, 6, 4, 7, 2, 1, 5, 8)`, appended as `gh * gw` rows.
    fn patchify(&self, chw: &[f32], out: &mut Vec<f32>) {
        let [vh, vw] = self.vision_hw;
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

    fn step(image: [u32; 2], vision: [u32; 2], patch: usize, merge: usize) -> VlaJepaInputsStep {
        VlaJepaInputsStep {
            image_keys: vec!["images.image".into()],
            embodied_token_id: 7,
            num_embodied_tokens: 2,
            image_resolution: image,
            vision_resolution: vision,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            patch_size: patch,
            merge_size: merge,
            temporal_patch_size: 2,
        }
    }

    fn dense(taps: &[Taps], size_in: usize) -> Vec<Vec<f32>> {
        taps.iter()
            .map(|t| {
                let mut row = vec![0f32; size_in];
                row[t.start..t.start + t.weights.len()].copy_from_slice(&t.weights);
                row
            })
            .collect()
    }

    #[test]
    fn bicubic_taps_match_intels_weights() {
        // Intel's `_bicubic_weights(224, 256)` (numpy, e3dd944): every non-zero of rows 0, 1, 100, 255.
        let w = dense(&bicubic_taps(224, 256), 224);
        assert_eq!(w.len(), 256);
        let want: [(usize, &[(usize, f32)]); 4] = [
            (0, &[(0, 1.028_517_1), (1, -0.028_517_11)]),
            (
                1,
                &[(0, 0.152_003_85), (1, 0.909_014_34), (2, -0.061_018_173)],
            ),
            (
                100,
                &[
                    (86, -0.069_213_87),
                    (87, 0.647_094_7),
                    (88, 0.475_952_15),
                    (89, -0.053_833_008),
                ],
            ),
            (255, &[(222, -0.028_517_11), (223, 1.028_517_1)]),
        ];
        for (row, taps) in want {
            let nonzero: Vec<(usize, f32)> = w[row]
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0.0)
                .map(|(j, &v)| (j, v))
                .collect();
            assert_eq!(nonzero, taps, "row {row}");
        }
        for row in &w {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() <= 1e-6, "{sum}");
        }
    }

    #[test]
    fn bicubic_taps_of_the_same_size_are_the_identity() {
        let w = dense(&bicubic_taps(8, 8), 8);
        for (i, row) in w.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(v, f32::from(u8::from(i == j)), "{i} {j}");
            }
        }
    }

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
        let host = VlaJepaHost::new(&step([8, 12], [8, 12], 2, 2));
        assert_eq!((host.patches(), host.feature()), (24, 24));
        let chw: Vec<f32> = (0..3 * 8 * 12).map(|i| i as f32).collect();
        let mut got = Vec::new();
        host.patchify(&chw, &mut got);
        assert_eq!(got, patchify_reference(&chw, 8, 12, 2, 2, 2));
    }

    #[test]
    fn frames_are_scaled_resized_and_normalized() {
        // Same in and vision size: the resize is the identity.
        let host = VlaJepaHost::new(&step([4, 4], [4, 4], 2, 2));
        let hwc: Vec<u8> = (0..48).map(|i| (i * 5) as u8).collect();
        let got = host.normalized(&hwc);
        for c in 0..3 {
            for p in 0..16 {
                let want = (f32::from(hwc[p * 3 + c]) / 255.0 - 0.5) / 0.5;
                assert_eq!(got[c * 16 + p], want, "c {c} p {p}");
            }
        }
        // A flat frame stays flat through the bicubic: each row of weights sums to 1.
        let host = VlaJepaHost::new(&step([224, 224], [256, 256], 16, 2));
        let got = host.normalized(&[255u8; 224 * 224 * 3]);
        assert_eq!(got.len(), 3 * 256 * 256);
        assert!(got.iter().all(|&v| (v - 1.0).abs() <= 1e-5));
    }

    #[test]
    fn pixel_values_follow_the_slot_order_and_check_sizes() {
        let mut s = step([4, 4], [4, 4], 2, 2);
        s.image_keys = vec!["images.a".into(), "images.b".into()];
        let host = VlaJepaHost::new(&s);
        let layout = CameraLayout {
            names: vec!["a".into(), "b".into()],
            slots: vec![Some(0), Some(1)],
        };
        let frame = |v: u8| Image {
            height: 4,
            width: 4,
            data: vec![v; 48],
        };
        let px = host
            .pixel_values(&layout, &[("b".into(), frame(255)), ("a".into(), frame(0))])
            .unwrap();
        let n = host.patches() * host.feature();
        assert_eq!(px.len(), 2 * n);
        assert!(px[..n].iter().all(|&v| v == -1.0), "a first");
        assert!(px[n..].iter().all(|&v| v == 1.0));
        let small = Image {
            height: 2,
            width: 2,
            data: vec![0; 12],
        };
        let e = host
            .pixel_values(&layout, &[("a".into(), small), ("b".into(), frame(0))])
            .unwrap_err();
        assert!(e.to_string().contains("must be 4x4"), "{e}");
    }

    #[test]
    fn embodied_positions_need_the_exact_count() {
        let host = VlaJepaHost::new(&step([4, 4], [4, 4], 2, 2));
        assert_eq!(
            host.embodied_positions(&[1, 7, 3, 7, 0]).unwrap(),
            vec![1, 3]
        );
        for ids in [&[1, 7, 3][..], &[7, 7, 7]] {
            let e = host.embodied_positions(ids).unwrap_err().to_string();
            assert!(e.contains("needs exactly 2"), "{e}");
        }
    }
}
