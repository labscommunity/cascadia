//! GR00T N1.7 host steps, ported from NVIDIA's processor (Isaac-GR00T
//! `3df8b382`: `processing_gr00t_n1d7.py`, the eval transform of
//! `image_augmentations.py`) as Intel's edge-ai-suites exporter feeds its
//! single IR: square frames cropped and resized with OpenCV's `INTER_AREA`,
//! Qwen3-VL patches, the formalized chat prompt, image-pad expansion and
//! mRoPE positions.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::Image;

use super::cameras::CameraLayout;
use super::manifest::{FeatureType, Gr00tPromptStep, PolicyManifest, QwenVlInputsStep};
use super::qwen_vl::PatchGeometry;
use super::steps::py_space;

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("policy gr00t: {msg}")))
}

/// OpenCV's `INTER_RESIZE_COEF_SCALE`: fixed-point weights are in 1/2048.
const COEF_SCALE: f32 = 2048.0;

/// One axis of OpenCV's `INTER_AREA` resize when it does not shrink: the
/// "area mode" variant of the bilinear kernel (`resize.cpp`, 4.11
/// `cv::hal::resize`), source index and fixed-point weights per output.
#[derive(Clone, Debug, PartialEq, Eq)]
struct AreaAxis {
    ofs: Vec<usize>,
    coef: Vec<[i32; 2]>,
    /// Outputs from here on read one source pixel only (`xmax`).
    one_tap_from: usize,
}

impl AreaAxis {
    fn new(src: usize, dst: usize) -> Self {
        let inv_scale = dst as f64 / src as f64;
        let scale = 1.0 / inv_scale;
        let mut axis = Self {
            ofs: Vec::with_capacity(dst),
            coef: Vec::with_capacity(dst),
            one_tap_from: dst,
        };
        for d in 0..dst {
            let mut s = (d as f64 * scale).floor() as i64;
            let mut f = ((d + 1) as f64 - (s + 1) as f64 * inv_scale) as f32;
            f = if f <= 0.0 { 0.0 } else { f - f.floor() };
            if s < 0 {
                (f, s) = (0.0, 0);
            }
            if s + 1 >= src as i64 {
                axis.one_tap_from = axis.one_tap_from.min(d);
                if s >= src as i64 - 1 {
                    (f, s) = (0.0, src as i64 - 1);
                }
            }
            // saturate_cast<short>(float): round half to even.
            let fixed = |c: f32| (c * COEF_SCALE).round_ties_even() as i32;
            axis.ofs.push(s as usize);
            axis.coef.push([fixed(1.0 - f), fixed(f)]);
        }
        axis
    }
}

/// OpenCV `cv2.resize(src, (dw, dh), interpolation=INTER_AREA)` for 3-channel
/// uint8 frames that are not shrunk on either axis, bit for bit: OpenCV's
/// true area resize runs only when both axes shrink; otherwise it is this
/// fixed-point bilinear with area-mode weights, the vertical pass in its
/// 128-bit SIMD form (`VResizeLinearVec_32s8u`) and a scalar tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AreaUpResize {
    src: [usize; 2],
    dst: [usize; 2],
    x: AreaAxis,
    y: AreaAxis,
}

impl AreaUpResize {
    /// `None` when an axis shrinks (that is OpenCV's other, true-area path).
    pub fn new(src: [usize; 2], dst: [usize; 2]) -> Option<Self> {
        if src.iter().chain(&dst).any(|&d| d == 0) || dst[0] < src[0] || dst[1] < src[1] {
            return None;
        }
        Some(Self {
            src,
            dst,
            x: AreaAxis::new(src[1], dst[1]),
            y: AreaAxis::new(src[0], dst[0]),
        })
    }

    /// HWC RGB `src[0] × src[1]` to `dst[0] × dst[1]`.
    pub fn resize(&self, hwc: &[u8]) -> Vec<u8> {
        const CN: usize = 3;
        let [sh, sw] = self.src;
        let [dh, dw] = self.dst;
        let width = dw * CN;
        debug_assert_eq!(hwc.len(), sh * sw * CN);
        // Horizontal pass of every source row, in i32 as `HResizeLinear`.
        let rows: Vec<Vec<i32>> = (0..sh)
            .map(|y| {
                let row = &hwc[y * sw * CN..(y + 1) * sw * CN];
                (0..width)
                    .map(|i| {
                        let (dx, k) = (i / CN, i % CN);
                        let sx = self.x.ofs[dx] * CN + k;
                        let [a0, a1] = self.x.coef[dx];
                        if dx >= self.x.one_tap_from {
                            i32::from(row[sx]) * COEF_SCALE as i32
                        } else {
                            i32::from(row[sx]) * a0 + i32::from(row[sx + CN]) * a1
                        }
                    })
                    .collect()
            })
            .collect();
        // `VResizeLinearVec_32s8u` covers whole 16-lane, then 8-lane blocks; the rest is scalar.
        let mut simd_end = 0;
        while simd_end + 16 <= width {
            simd_end += 16;
        }
        while simd_end + 8 < width {
            simd_end += 8;
        }
        let sat16 = |v: i32| v.clamp(i32::from(i16::MIN), i32::from(i16::MAX));
        let mut out = Vec::with_capacity(dh * width);
        for dy in 0..dh {
            let sy = self.y.ofs[dy];
            let (s0, s1) = (&rows[sy.min(sh - 1)], &rows[(sy + 1).min(sh - 1)]);
            let [b0, b1] = self.y.coef[dy];
            for x in 0..width {
                let v = if x < simd_end {
                    // v_pack(S >> 4), v_mul_hi, saturating v_add, v_rshr_pack_u<2>.
                    let m0 = (sat16(s0[x] >> 4) * b0) >> 16;
                    let m1 = (sat16(s1[x] >> 4) * b1) >> 16;
                    (sat16(m0 + m1) + 2) >> 2
                } else {
                    // FixedPtCast<int, uchar, 22>.
                    (s0[x] * b0 + s1[x] * b1 + (1 << 21)) >> 22
                };
                out.push(v.clamp(0, 255) as u8);
            }
        }
        out
    }
}

/// What the package fixes: one square frame size, the crop and the patch grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tGeometry {
    /// Frames are `side × side`; the letterbox and the first resize leave them alone.
    pub side: usize,
    pub crop: usize,
    /// Cameras, in VISUAL `input_features` order.
    pub cameras: usize,
    pub patches: PatchGeometry,
    /// The client's state length (the STATE feature).
    pub state_len: usize,
    /// The graph's state width: normalize's `pad_to`, else `state_len`.
    pub state_width: usize,
}

impl Gr00tGeometry {
    pub fn new(m: &PolicyManifest) -> EngineResult<Self> {
        let Some((image, patchify, _, inputs)) = m.gr00t_steps() else {
            return refuse(
                "gr00t_n1d7 needs gr00t_image, qwen_vl_patchify, gr00t_prompt and qwen_vl_inputs",
            );
        };
        let side = image.shortest_edge;
        let mut cameras = 0;
        for f in m
            .input_features
            .iter()
            .filter(|f| f.ftype == FeatureType::Visual)
        {
            if f.shape != [3, side as i64, side as i64] {
                return refuse(format!(
                    "camera {:?} has shape {:?}; the engine takes square frames of shortest_edge, \
                     [3, {side}, {side}], so the letterbox and the first resize do nothing",
                    f.name, f.shape
                ));
            }
            cameras += 1;
        }
        if cameras == 0 {
            return refuse("no VISUAL input_features");
        }
        let state_len = match m
            .input_features
            .iter()
            .find(|f| f.ftype == FeatureType::State)
            .map(|f| f.shape.as_slice())
        {
            Some([n]) if *n > 0 => *n as usize,
            other => return refuse(format!("the STATE feature's shape {other:?} must be [n]")),
        };
        let state_width = m.normalize().find_map(|s| s.pad_to).unwrap_or(state_len);
        if state_width < state_len {
            return refuse(format!(
                "normalize pad_to {state_width} is narrower than the state ({state_len})"
            ));
        }
        // FractionalCenterCrop: `max(1, int(side * fraction))`.
        let crop = ((side as f64 * image.crop_fraction) as usize).max(1);
        let patches = PatchGeometry {
            hw: [side, side],
            patch: patchify.patch_size,
            merge: patchify.merge_size,
            temporal: patchify.temporal_patch_size,
        };
        let unit = patches.patch * patches.merge;
        if side % unit != 0 {
            return refuse(format!(
                "shortest_edge {side} is not a multiple of patch_size * merge_size ({unit})"
            ));
        }
        let grid = [1, side / patches.patch, side / patches.patch];
        if patchify.grid_thw.len() != cameras || patchify.grid_thw.iter().any(|g| *g != grid) {
            return refuse(format!(
                "grid_thw {:?} is not {grid:?} for each of {cameras} cameras",
                patchify.grid_thw
            ));
        }
        if inputs.merge_size != patches.merge {
            return refuse(format!(
                "qwen_vl_inputs merge_size {} differs from qwen_vl_patchify's {}",
                inputs.merge_size, patches.merge
            ));
        }
        if inputs.tokens_per_image != patches.image_pads() {
            return refuse(format!(
                "tokens_per_image {} is not the {} merged patches of a frame",
                inputs.tokens_per_image,
                patches.image_pads()
            ));
        }
        if cameras * inputs.tokens_per_image >= inputs.seq_len {
            return refuse(format!(
                "{cameras} images of {} tokens leave no room in seq_len {}",
                inputs.tokens_per_image, inputs.seq_len
            ));
        }
        Ok(Self {
            side,
            crop,
            cameras,
            patches,
            state_len,
            state_width,
        })
    }
}

/// `formalize_language`: `re.sub(r"[^\w\s]", "", text.lower())`.
///
/// Python's `\w` is `isalnum()` or `_`; Rust's `is_alphanumeric` also keeps
/// the combining marks Unicode counts as alphabetic, where Python drops them.
pub fn formalize(task: &str) -> String {
    task.to_lowercase()
        .chars()
        .filter(|&c| c.is_alphanumeric() || c == '_' || py_space(c))
        .collect()
}

/// The prompt text the tokenizer sees: `{images}` is one `image` per camera.
pub fn render_prompt(step: &Gr00tPromptStep, cameras: usize, task: &str) -> String {
    let task = if step.formalize_language {
        formalize(task)
    } else {
        task.to_string()
    };
    // The task goes in last, so nothing in it is read as a placeholder.
    step.template
        .replacen("{images}", &step.image.repeat(cameras), 1)
        .replacen("{task}", &task, 1)
}

/// One package's GR00T host inputs.
#[derive(Clone, Debug)]
pub struct Gr00tHost {
    pub geo: Gr00tGeometry,
    resize: AreaUpResize,
    mean: [f32; 3],
    std: [f32; 3],
    inputs: QwenVlInputsStep,
}

impl Gr00tHost {
    pub fn new(m: &PolicyManifest) -> EngineResult<Self> {
        let geo = Gr00tGeometry::new(m)?;
        let (_, patchify, _, inputs) = m.gr00t_steps().expect("Gr00tGeometry checked the steps");
        let Some(resize) = AreaUpResize::new([geo.crop; 2], [geo.side; 2]) else {
            return refuse("the crop is larger than the frame");
        };
        Ok(Self {
            geo,
            resize,
            mean: patchify.image_mean,
            std: patchify.image_std,
            inputs: inputs.clone(),
        })
    }

    pub fn seq_len(&self) -> usize {
        self.inputs.seq_len
    }

    pub fn image_token_id(&self) -> i64 {
        self.inputs.image_token_id
    }

    /// Unexpanded ids the prompt may hold: `seq_len` less what the images add.
    pub fn id_budget(&self) -> usize {
        self.inputs.seq_len - self.geo.cameras * (self.inputs.tokens_per_image - 1)
    }

    /// One frame (HWC, `side × side`) to CHW f32: the centred crop, INTER_AREA
    /// back to `side`, `(x / 255 − mean) / std`.
    pub fn frame(&self, hwc: &[u8]) -> Vec<f32> {
        let (side, crop) = (self.geo.side, self.geo.crop);
        let at = (side - crop) / 2;
        let mut cropped = Vec::with_capacity(crop * crop * 3);
        for y in at..at + crop {
            cropped.extend_from_slice(&hwc[(y * side + at) * 3..(y * side + at + crop) * 3]);
        }
        let resized = self.resize.resize(&cropped);
        let n = side * side;
        let mut chw = vec![0f32; 3 * n];
        for (p, rgb) in resized.chunks_exact(3).enumerate() {
            for c in 0..3 {
                chw[c * n + p] = (f32::from(rgb[c]) / 255.0 - self.mean[c]) / self.std[c];
            }
        }
        chw
    }

    /// `pixel_values [cameras * patches, feature]`, cameras in slot order;
    /// `flip[i]` rotates camera `i` (of `layout.names`) by 180 degrees first.
    pub fn pixel_values(
        &self,
        layout: &CameraLayout,
        images: &[(String, Image)],
        flip: &[bool],
    ) -> EngineResult<Vec<f32>> {
        let side = self.geo.side;
        let p = &self.geo.patches;
        let mut out = Vec::with_capacity(layout.slots.len() * p.patches() * p.feature());
        for slot in &layout.slots {
            let Some(i) = slot else {
                return refuse("gr00t has no dummy camera slots");
            };
            let name = &layout.names[*i];
            let Some((_, img)) = images.iter().find(|(n, _)| n == name) else {
                return Err(EngineError::InvalidConfig(format!(
                    "missing camera {name:?}"
                )));
            };
            if img.height as usize != side
                || img.width as usize != side
                || img.data.len() != side * side * 3
            {
                return Err(EngineError::InvalidConfig(format!(
                    "camera {name:?} must be {side}x{side} RGB ({} bytes)",
                    side * side * 3
                )));
            }
            let chw = if flip.get(*i).copied().unwrap_or(false) {
                let flipped: Vec<u8> = img.data.chunks_exact(3).rev().flatten().copied().collect();
                self.frame(&flipped)
            } else {
                self.frame(&img.data)
            };
            p.patchify(&chw, &mut out);
        }
        Ok(out)
    }

    /// From the tokenizer's ids (unpadded): `input_ids` with each image token
    /// repeated `tokens_per_image` times and right-padded with `pad_id` to
    /// `seq_len`, the `attention_mask`, and the mRoPE `position_ids` `[3, seq_len]`
    /// (rows t, h, w), 0 on the padding.
    pub fn inputs(&self, ids: &[i64]) -> EngineResult<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let q = &self.inputs;
        let images = ids.iter().filter(|&&t| t == q.image_token_id).count();
        if images != self.geo.cameras {
            return Err(EngineError::InvalidConfig(format!(
                "the prompt has {images} image tokens; the package has {} cameras",
                self.geo.cameras
            )));
        }
        let len = ids.len() + images * (q.tokens_per_image - 1);
        if len > q.seq_len {
            return Err(EngineError::PromptTooLong(format!(
                "the prompt needs {len} tokens with its images; the limit is {}",
                q.seq_len
            )));
        }
        let lw = self.geo.patches.hw[1] / self.geo.patches.patch / q.merge_size;
        let lh = q.tokens_per_image / lw;
        let mut input_ids = Vec::with_capacity(q.seq_len);
        let mut pos = [
            Vec::with_capacity(q.seq_len),
            Vec::with_capacity(q.seq_len),
            Vec::with_capacity(q.seq_len),
        ];
        let mut cur = 0i64;
        for &t in ids {
            if t != q.image_token_id {
                input_ids.push(t);
                for p in &mut pos {
                    p.push(cur);
                }
                cur += 1;
                continue;
            }
            // One temporal frame: t stays, h and w walk the merged grid.
            for h in 0..lh as i64 {
                for w in 0..lw as i64 {
                    input_ids.push(t);
                    pos[0].push(cur);
                    pos[1].push(cur + h);
                    pos[2].push(cur + w);
                }
            }
            cur += (lh.max(lw) as i64 - 1).max(0) + 1;
        }
        let mut mask = vec![1i64; input_ids.len()];
        input_ids.resize(q.seq_len, q.pad_id);
        mask.resize(q.seq_len, 0);
        let mut position_ids = Vec::with_capacity(3 * q.seq_len);
        for mut p in pos {
            p.resize(q.seq_len, 0);
            position_ids.extend(p);
        }
        Ok((input_ids, mask, position_ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pattern of `gr00t-area-resize-cv2.json`.
    fn pattern(h: usize, w: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    out.push(((x * 7 + y * 13 + c * 29 + (x * y) % 17) % 256) as u8);
                }
            }
        }
        out
    }

    fn fnv1a64(bytes: &[u8]) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{h:016x}")
    }

    #[test]
    fn area_upscale_is_opencvs_bit_for_bit() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/policy/gr00t-area-resize-cv2.json"
        ))
        .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 4);
        for case in cases {
            let dim = |k: &str, i: usize| case[k][i].as_u64().unwrap() as usize;
            let (src, dst) = (
                [dim("src", 0), dim("src", 1)],
                [dim("dst", 0), dim("dst", 1)],
            );
            let out = AreaUpResize::new(src, dst)
                .unwrap()
                .resize(&pattern(src[0], src[1]));
            let first: Vec<u8> = serde_json::from_value(case["first_row"].clone()).unwrap();
            assert_eq!(out[..first.len()], first[..], "{src:?} -> {dst:?}");
            assert_eq!(
                fnv1a64(&out),
                case["fnv1a64"].as_str().unwrap(),
                "{src:?} -> {dst:?}"
            );
        }
        assert!(AreaUpResize::new([256, 256], [243, 243]).is_none());
    }

    #[test]
    fn formalize_lowercases_and_drops_punctuation() {
        assert_eq!(
            formalize("Pick up the black bowl, NOW!\tthen <|im_end|> stop_it."),
            "pick up the black bowl now\tthen im_end stop_it"
        );
        assert_eq!(formalize("Café №5"), "café 5");
    }

    #[test]
    fn the_prompt_has_one_image_per_camera_and_the_task_last() {
        let step = Gr00tPromptStep {
            formalize_language: true,
            template: "<|im_start|>user\n{images}{task}<|im_end|>\n".into(),
            image: "<|vision_start|><|image_pad|><|vision_end|>".into(),
        };
        assert_eq!(
            render_prompt(&step, 2, "Go {images}!"),
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\
             <|vision_start|><|image_pad|><|vision_end|>go images<|im_end|>\n"
        );
        let raw = Gr00tPromptStep {
            formalize_language: false,
            ..step
        };
        assert!(
            render_prompt(&raw, 1, "{images}").ends_with("|><|vision_end|>{images}<|im_end|>\n")
        );
    }

    fn host(cameras: usize, seq_len: usize) -> Gr00tHost {
        let geo = Gr00tGeometry {
            side: 8,
            crop: 7,
            cameras,
            patches: PatchGeometry {
                hw: [8, 8],
                patch: 2,
                merge: 2,
                temporal: 2,
            },
            state_len: 2,
            state_width: 4,
        };
        Gr00tHost {
            resize: AreaUpResize::new([7, 7], [8, 8]).unwrap(),
            geo,
            mean: [0.5; 3],
            std: [0.5; 3],
            inputs: QwenVlInputsStep {
                image_token_id: 9,
                tokens_per_image: 4,
                seq_len,
                pad_id: 0,
                merge_size: 2,
            },
        }
    }

    #[test]
    fn inputs_expand_images_pad_right_and_follow_mrope() {
        // compare_groot.py's `rope`: text walks all three rows; a 2x2 merged image is
        // t = cur, h = cur + row, w = cur + col, and the text after it resumes at max + 1.
        let h = host(2, 16);
        let (ids, mask, pos) = h.inputs(&[5, 9, 6, 9, 7]).unwrap();
        assert_eq!(ids, [5, 9, 9, 9, 9, 6, 9, 9, 9, 9, 7, 0, 0, 0, 0, 0]);
        assert_eq!(mask, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]);
        let row = |r: usize| pos[r * 16..(r + 1) * 16].to_vec();
        assert_eq!(row(0), [0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0]);
        assert_eq!(row(1), [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 0, 0, 0, 0, 0]);
        assert_eq!(row(2), [0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6, 0, 0, 0, 0, 0]);
        assert_eq!(h.id_budget(), 16 - 2 * 3);
        let e = h.inputs(&[5, 9, 6]).unwrap_err().to_string();
        assert!(
            e.contains("1 image tokens; the package has 2 cameras"),
            "{e}"
        );
        let e = host(2, 10).inputs(&[5, 9, 6, 9, 7]).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
    }

    #[test]
    fn frames_are_cropped_resized_normalized_and_flipped_per_camera() {
        let h = host(2, 16);
        let layout = CameraLayout {
            names: vec!["a".into(), "b".into()],
            slots: vec![Some(0), Some(1)],
        };
        let flat = |v: u8| Image {
            height: 8,
            width: 8,
            data: vec![v; 8 * 8 * 3],
        };
        let px = h
            .pixel_values(
                &layout,
                &[("b".into(), flat(255)), ("a".into(), flat(0))],
                &[false, true],
            )
            .unwrap();
        let n = h.geo.patches.patches() * h.geo.patches.feature();
        assert_eq!(px.len(), 2 * n);
        assert!(px[..n].iter().all(|&v| v == -1.0) && px[n..].iter().all(|&v| v == 1.0));
        // A flip is a 180-degree rotation of the frame as sent.
        let ramp = Image {
            height: 8,
            width: 8,
            data: (0..192).map(|i| i as u8).collect(),
        };
        let mut rotated = ramp.clone();
        rotated.data = ramp.data.chunks_exact(3).rev().flatten().copied().collect();
        let a = h
            .pixel_values(
                &layout,
                &[("a".into(), ramp.clone()), ("b".into(), ramp.clone())],
                &[true, false],
            )
            .unwrap();
        let b = h
            .pixel_values(
                &layout,
                &[("a".into(), rotated), ("b".into(), ramp)],
                &[false, false],
            )
            .unwrap();
        assert_eq!(a, b);
        let small = Image {
            height: 4,
            width: 4,
            data: vec![0; 48],
        };
        let e = h
            .pixel_values(
                &layout,
                &[("a".into(), small), ("b".into(), flat(0))],
                &[false, false],
            )
            .unwrap_err();
        assert!(e.to_string().contains("must be 8x8"), "{e}");
    }
}
