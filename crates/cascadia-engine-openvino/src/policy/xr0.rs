//! XR0 (Xiaomi-Robotics-0) host steps, ported from Intel's `physicalai` runtime
//! at `e3dd944` (`preprocessors/xr0.py`, `postprocessors/xr0.py`), and XR1's
//! (`processor: "xr1"`) from Xiaomi's `processing_mibot.py` and
//! `eval_robocasa365/entry.py`.
//!
//! XR0's frames must arrive at the size Intel's resize leaves alone (256×256 for
//! LIBERO), so its OpenCV resize never runs here. XR1's processor never
//! resizes; its client crops each frame to the centre 0.95 and resizes it back
//! (bilinear) before sending.
//!
//! XR1 takes each camera as a video of `history` frames; the graph, exported
//! as an image model, takes each temporal patch (two frames) as one image
//! slot, named `images.<camera>_<k>` in the manifest, and image pads where
//! Xiaomi's prompt has video pads (the same model output).

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::Image;

use super::cameras::CameraLayout;
use super::manifest::{FeatureType, PolicyManifest, Xr0DenormalizeStep, Xr0Processor, Xr0Step};
use super::qwen_vl::PatchGeometry;

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("policy xr0: {msg}")))
}

/// What the package's features fix: the frame size, the views and the state length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xr0Geometry {
    /// `[height, width]` every camera sends.
    pub hw: [usize; 2],
    /// View names for the prompt (`images.<view>`), one per image slot.
    pub views: Vec<String>,
    /// The cameras a request sends, in order; each fills `history /
    /// temporal_patch_size` consecutive views (`<camera>_0`, `<camera>_1`, ...)
    /// when `history` is above 1, else one view of its own name.
    pub cameras: Vec<String>,
    /// Frames per camera and state rows a request carries.
    pub history: usize,
    /// The state a client sends; padded to `max_state_dim` for the graph.
    pub state_len: usize,
    pub patches: PatchGeometry,
}

/// Intel's `_resize_image` target `(height, width)` for a frame.
fn resize_target(h: usize, w: usize, factor: usize, max_pixels: usize) -> [usize; 2] {
    let f = factor as f64;
    let (hf, wf) = (h as f64, w as f64);
    let mut nh = factor.max((hf / f).round_ties_even() as usize * factor);
    let mut nw = factor.max((wf / f).round_ties_even() as usize * factor);
    if nh * nw > max_pixels {
        let scale = (hf * wf / max_pixels as f64).sqrt();
        nh = factor.max((hf / scale / f).floor() as usize * factor);
        nw = factor.max((wf / scale / f).floor() as usize * factor);
    } else if nh * nw < factor * factor {
        let scale = ((factor * factor) as f64 / (hf * wf)).sqrt();
        nh = factor.max((hf * scale / f).ceil() as usize * factor);
        nw = factor.max((wf * scale / f).ceil() as usize * factor);
    }
    [nh, nw]
}

/// Cameras are the VISUAL `input_features` (`[3, H, W]`, one size for all),
/// the state length the STATE feature; refused where Intel would resize (xr0).
pub fn geometry(m: &PolicyManifest, step: &Xr0Step) -> EngineResult<Xr0Geometry> {
    let mut hw = None;
    let mut views = Vec::new();
    for f in m
        .input_features
        .iter()
        .filter(|f| f.ftype == FeatureType::Visual)
    {
        let shape = match f.shape.as_slice() {
            [3, h, w] if *h > 0 && *w > 0 => [*h as usize, *w as usize],
            other => {
                return refuse(format!(
                    "camera {:?} has shape {other:?}; expected [3, H, W]",
                    f.name
                ))
            }
        };
        if *hw.get_or_insert(shape) != shape {
            return refuse(format!(
                "cameras differ in size: {:?} and {shape:?}",
                hw.unwrap()
            ));
        }
        // xr0.py `_resolve_views`: the key without `images.`.
        views.push(
            f.name
                .strip_prefix("images.")
                .unwrap_or(&f.name)
                .to_string(),
        );
    }
    let Some(hw) = hw else {
        return refuse("no VISUAL input_features");
    };
    let cameras = if step.history > 1 {
        video_cameras(&views, step.history / step.temporal_patch_size)?
    } else {
        views.clone()
    };
    if step.processor == Xr0Processor::Xr1 && step.camera_labels.len() != cameras.len() {
        return refuse(format!(
            "{} camera_labels for the cameras {cameras:?}",
            step.camera_labels.len()
        ));
    }
    let state_len = match m
        .input_features
        .iter()
        .find(|f| f.ftype == FeatureType::State)
        .map(|f| f.shape.as_slice())
    {
        Some([n]) if *n > 0 && *n as usize <= step.max_state_dim => *n as usize,
        other => {
            return refuse(format!(
                "the STATE feature's shape {other:?} must be [n] with n <= max_state_dim {}",
                step.max_state_dim
            ))
        }
    };
    if hw[0].max(hw[1]) > 200 * hw[0].min(hw[1]) {
        return refuse(format!("aspect ratio of {hw:?} is over 200"));
    }
    let target = resize_target(hw[0], hw[1], step.image_factor, step.image_max_pixels);
    // Xiaomi's XR1 processor never resizes (`do_resize=False`).
    if step.processor == Xr0Processor::Xr0 && target != hw {
        return refuse(format!(
            "frames of {hw:?} would be resized to {target:?}; the engine takes only frames \
             xr0 leaves as they are"
        ));
    }
    let unit = step.patch_size * step.merge_size;
    if hw[0] % unit != 0 || hw[1] % unit != 0 {
        return refuse(format!(
            "frames of {hw:?} are not a multiple of patch_size * merge_size ({unit})"
        ));
    }
    Ok(Xr0Geometry {
        hw,
        views,
        cameras,
        history: step.history,
        state_len,
        patches: PatchGeometry {
            hw,
            patch: step.patch_size,
            merge: step.merge_size,
            temporal: step.temporal_patch_size,
        },
    })
}

/// Views `<camera>_0 .. <camera>_{per - 1}`, camera after camera, to the cameras.
fn video_cameras(views: &[String], per: usize) -> EngineResult<Vec<String>> {
    if !views.len().is_multiple_of(per) {
        return refuse(format!(
            "{} cameras do not split into videos of {per} image slots",
            views.len()
        ));
    }
    let mut cameras: Vec<String> = Vec::new();
    for group in views.chunks(per) {
        let Some(camera) = group[0].strip_suffix("_0").filter(|c| !c.is_empty()) else {
            return refuse(format!(
                "camera {:?} does not end in _0; a video camera's slots are <camera>_<k>",
                group[0]
            ));
        };
        for (k, v) in group.iter().enumerate() {
            if *v != format!("{camera}_{k}") {
                return refuse(format!("camera {v:?} is not {camera}_{k}"));
            }
        }
        if cameras.iter().any(|c| c == camera) {
            return refuse(format!("camera {camera:?} declared twice"));
        }
        cameras.push(camera.to_string());
    }
    Ok(cameras)
}

impl Xr0Geometry {
    /// Image slots (temporal patches) per camera.
    pub fn slots_per_camera(&self) -> usize {
        self.views.len() / self.cameras.len()
    }

    /// Each camera in its consecutive slots.
    pub fn layout(&self) -> CameraLayout {
        let per = self.slots_per_camera();
        CameraLayout {
            names: self.cameras.clone(),
            slots: (0..self.cameras.len())
                .flat_map(|i| std::iter::repeat_n(Some(i), per))
                .collect(),
        }
    }
}

/// `_view_title`: the trained titles, else Python `str.capitalize` per `_` word.
pub fn view_title(view: &str) -> String {
    let key = view.replace('-', "_");
    match key.as_str() {
        "ego" => "Ego".into(),
        "base" => "Base".into(),
        "wrist_left" => "Left-Wrist".into(),
        "wrist_right" => "Right-Wrist".into(),
        _ => key
            .split('_')
            .map(|w| {
                let mut c = w.chars();
                c.next().map_or_else(String::new, |f| {
                    f.to_uppercase()
                        .chain(c.flat_map(char::to_lowercase))
                        .collect()
                })
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// `_render_chat_prompt`: the Qwen3-VL chat template, `pads` image pads per
/// view, the task verbatim.
pub fn render_prompt(views: &[String], pads: usize, task: &str) -> String {
    let mut user = String::from("The following observations are captured from multiple views.\n");
    for v in views {
        user.push_str(&format!("# {} View\n<|vision_start|>", view_title(v)));
        user.push_str(&"<|image_pad|>".repeat(pads));
        user.push_str("<|vision_end|>\n");
    }
    user.push_str(&format!(
        "Generate robot actions for the task:\n{task} /no_cot"
    ));
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n")
}

/// Xiaomi's XR1 chat prompt (`entry.py` `_build_messages` through
/// `processing_mibot.py`): per camera `<label>: ` and, per temporal patch, its
/// timestamp and `pads` image pads; the task verbatim.
///
/// A frame list carries no fps, so the processor times frame `i` at `i / 24`
/// s and stamps each temporal patch with the mean of its first and last
/// frame's time (`_calculate_timestamps` over `merge` frames), one decimal.
pub fn render_xr1_prompt(
    labels: &[String],
    slots: usize,
    merge: usize,
    pads: usize,
    task: &str,
) -> String {
    let mut user = String::new();
    for (i, label) in labels.iter().enumerate() {
        if i > 0 {
            user.push('\n');
        }
        user.push_str(&format!("{label}: "));
        for k in 0..slots {
            let first = (k * merge) as f64 / 24.0;
            let last = (k * merge + merge - 1) as f64 / 24.0;
            user.push_str(&format!(
                "<{:.1} seconds><|vision_start|>",
                (first + last) / 2.0
            ));
            user.push_str(&"<|image_pad|>".repeat(pads));
            user.push_str("<|vision_end|>");
        }
    }
    user.push_str(&format!(
        "\n\nGenerate robot actions for the task:\n{task} /no_cot"
    ));
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n")
}

/// The prompt `step` writes around `task`.
pub fn task_prompt(m: &PolicyManifest, step: &Xr0Step, task: &str) -> EngineResult<String> {
    let geo = geometry(m, step)?;
    let pads = geo.patches.image_pads();
    Ok(match step.processor {
        Xr0Processor::Xr0 => render_prompt(&geo.views, pads, task),
        Xr0Processor::Xr1 => render_xr1_prompt(
            &step.camera_labels,
            geo.slots_per_camera(),
            step.merge_size,
            pads,
            task,
        ),
    })
}

/// One package's `xr0` geometry and normalization.
#[derive(Clone, Debug)]
pub struct Xr0Host {
    pub geo: Xr0Geometry,
    mean: [f32; 3],
    std: [f32; 3],
    /// Intel's `px * rescale`, or 1 where the rescale is fused into `mean` and `std`.
    rescale: f32,
    bf16_state: bool,
    max_state_dim: usize,
}

impl Xr0Host {
    pub fn new(m: &PolicyManifest, step: &Xr0Step) -> EngineResult<Self> {
        let (mut mean, mut std, mut rescale) =
            (step.image_mean, step.image_std, step.rescale_factor as f32);
        if step.processor == Xr0Processor::Xr1 {
            // transformers' `_fuse_mean_std_and_rescale_factor`: `(px - mean/r) / (std/r)`
            // in f32, which differs from `(px * r - mean) / std` by an ulp.
            let inv = (1.0 / step.rescale_factor) as f32;
            mean = mean.map(|x| x * inv);
            std = std.map(|x| x * inv);
            rescale = 1.0;
        }
        Ok(Self {
            geo: geometry(m, step)?,
            mean,
            std,
            rescale,
            bf16_state: step.processor == Xr0Processor::Xr1,
            max_state_dim: step.max_state_dim,
        })
    }

    pub fn max_state_dim(&self) -> usize {
        self.max_state_dim
    }

    /// `pixel_values [n_slots * patches, feature]`: `(px * rescale − mean) / std`
    /// in f32 (`_build_pixel_grid`), patchified per slot. A slot holds one
    /// frame tiled (one-frame history) or a camera's next `temporal` frames.
    pub fn pixel_values(
        &self,
        layout: &CameraLayout,
        images: &[(String, Image)],
    ) -> EngineResult<Vec<f32>> {
        let [h, w] = self.geo.hw;
        let p = &self.geo.patches;
        let mut out = Vec::with_capacity(layout.slots.len() * p.patches() * p.feature());
        let mut chws = vec![vec![0f32; 3 * h * w]; p.temporal];
        for (j, slot) in layout.slots.iter().enumerate() {
            let Some(i) = slot else {
                return refuse("xr0 has no dummy camera slots");
            };
            let name = &layout.names[*i];
            let frames: Vec<&Image> = images
                .iter()
                .filter(|(n, _)| n == name)
                .map(|(_, img)| img)
                .collect();
            if frames.is_empty() {
                return Err(EngineError::InvalidConfig(format!(
                    "missing camera {name:?}"
                )));
            }
            if frames.len() != self.geo.history {
                return Err(EngineError::InvalidConfig(format!(
                    "camera {name:?} has {} frames; this model takes {}",
                    frames.len(),
                    self.geo.history
                )));
            }
            let picked: Vec<&Image> = if self.geo.history == 1 {
                vec![frames[0]]
            } else {
                // The slot's place among its camera's slots picks its frames.
                let k = layout.slots[..j].iter().filter(|s| **s == Some(*i)).count();
                frames[k * p.temporal..(k + 1) * p.temporal].to_vec()
            };
            for (img, chw) in picked.iter().zip(chws.iter_mut()) {
                if img.height as usize != h
                    || img.width as usize != w
                    || img.data.len() != h * w * 3
                {
                    return Err(EngineError::InvalidConfig(format!(
                        "camera {name:?} must be {h}x{w} RGB ({} bytes)",
                        h * w * 3
                    )));
                }
                for (px, rgb) in img.data.chunks_exact(3).enumerate() {
                    for c in 0..3 {
                        chw[c * h * w + px] =
                            (f32::from(rgb[c]) * self.rescale - self.mean[c]) / self.std[c];
                    }
                }
            }
            if picked.len() == 1 {
                p.patchify(&chws[0], &mut out);
            } else {
                let refs: Vec<&[f32]> = chws.iter().map(Vec::as_slice).collect();
                p.patchify_frames(&refs, &mut out);
            }
        }
        Ok(out)
    }

    /// `_prepare_state`: each `state_len` row zero-padded (or cut) to
    /// `max_state_dim`; XR1's processor also rounds it to bf16 (`state.to(bfloat16)`).
    pub fn state(&self, raw: &[f32]) -> Vec<f32> {
        let mut s = Vec::with_capacity(self.geo.history * self.max_state_dim);
        for row in raw.chunks(self.geo.state_len) {
            let at = s.len();
            s.extend_from_slice(row);
            s.resize(at + self.max_state_dim, 0.0);
        }
        if self.bf16_state {
            for x in &mut s {
                *x = half::bf16::from_f32(*x).to_f32();
            }
        }
        s
    }
}

/// `xr0_denormalize` on row-major `[steps, width]` actions, in f32 as numpy
/// does it; returns `[steps, action_dim]`.
pub fn denormalize(step: &Xr0DenormalizeStep, actions: &[f32], width: usize) -> Vec<f32> {
    let keep = step.action_dim.unwrap_or(width);
    let mut out = Vec::with_capacity(actions.len() / width * keep);
    for (t, row) in actions.chunks(width).enumerate() {
        let (mean, std) = (&step.action_mean[t], &step.action_std[t]);
        out.extend((0..keep).map(|d| row[d] * (std[d] + step.action_eps) + mean[d]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::manifest::{ImageStep, LoadOptions};

    const XR1_ROBOCASA: &str =
        include_str!("../../tests/fixtures/policy/xr1-robocasa365-manifest.json");
    /// Xiaomi's processor on 32x32 frames: `tools/xr1_processor_fixture.py`.
    const XR1_PROCESSOR: &[u8] =
        include_bytes!("../../tests/fixtures/policy/xr1-processor-32px.npz");

    /// `(descr, shape, data)` of `name` in an uncompressed `np.savez` archive.
    fn npy(name: &str) -> (String, Vec<usize>, &'static [u8]) {
        let z = XR1_PROCESSOR;
        let file = format!("{name}.npy");
        let at = (30..z.len())
            .find(|&i| z[i - 30..i - 26] == *b"PK\x03\x04" && z[i..].starts_with(file.as_bytes()))
            .unwrap_or_else(|| panic!("no {file}"));
        let npy = at
            + (0..)
                .find(|&i| z[at + i..].starts_with(b"\x93NUMPY"))
                .unwrap();
        assert_eq!(z[npy + 6], 1, "npy format 1.0");
        let len = u16::from_le_bytes([z[npy + 8], z[npy + 9]]) as usize;
        let header = std::str::from_utf8(&z[npy + 10..npy + 10 + len]).unwrap();
        let field = |key: &str| {
            let rest = &header[header.find(key).unwrap() + key.len()..];
            rest[..rest.find(['\'', ')']).unwrap()].to_string()
        };
        let descr = field("'descr': '");
        let shape: Vec<usize> = field("'shape': (")
            .split(',')
            .filter_map(|d| d.trim().parse().ok())
            .collect();
        // `<U<n>` holds n UTF-32 code units.
        let unit = if descr.starts_with("<U") { 4 } else { 1 };
        let item = descr[2..].parse::<usize>().unwrap() * unit;
        let size = shape.iter().product::<usize>() * item;
        let start = npy + 10 + len;
        (descr, shape, &z[start..start + size])
    }

    fn f32s(name: &str) -> Vec<f32> {
        let (descr, _, data) = npy(name);
        assert_eq!(descr, "<f4");
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn string(name: &str) -> String {
        let (descr, _, data) = npy(name);
        assert!(descr.starts_with("<U"));
        data.chunks_exact(4)
            .map(|c| char::from_u32(u32::from_le_bytes(c.try_into().unwrap())).unwrap())
            .filter(|&c| c != '\0')
            .collect()
    }

    /// The XR1 manifest with `h`x`w` cameras.
    fn xr1_sized(h: usize, w: usize) -> PolicyManifest {
        let mut v: serde_json::Value = serde_json::from_str(XR1_ROBOCASA).unwrap();
        for f in v["model"]["input_features"].as_array_mut().unwrap() {
            if f["init_args"]["ftype"] == "VISUAL" {
                f["init_args"]["shape"] = serde_json::json!([3, h, w]);
            }
        }
        PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap()
    }

    /// The XR1 manifest with 32x32 cameras, the fixture's frame size.
    fn xr1_32px() -> PolicyManifest {
        xr1_sized(32, 32)
    }

    /// Frames Intel's xr0 would shrink pass for XR1, whose processor never resizes.
    #[test]
    fn xr1_frames_are_never_resized() {
        assert_eq!(resize_target(480, 640, 32, 90_000), [256, 320]);
        let m = xr1_sized(480, 640);
        let ImageStep::Xr0(step) = m.image_step() else {
            panic!("xr0 is the image step")
        };
        assert_eq!(geometry(&m, step).unwrap().hw, [480, 640]);
    }

    /// Xiaomi's prompt, pixel values and state for the fixture's request, bit for bit.
    #[test]
    fn xr1_matches_xiaomis_processor() {
        let m = xr1_32px();
        let ImageStep::Xr0(step) = m.image_step() else {
            panic!("xr0 is the image step")
        };
        let host = Xr0Host::new(&m, step).unwrap();
        let (_, shape, frames) = npy("frames");
        assert_eq!(shape, [3, 4, 32, 32, 3]);
        let layout = host.geo.layout();
        // Sent camera by camera, oldest frame first.
        let images: Vec<(String, Image)> = frames
            .chunks_exact(32 * 32 * 3)
            .enumerate()
            .map(|(i, f)| {
                let img = Image {
                    height: 32,
                    width: 32,
                    data: f.to_vec(),
                };
                (layout.names[i / 4].clone(), img)
            })
            .collect();
        assert_eq!(
            host.pixel_values(&layout, &images).unwrap(),
            f32s("pixel_values"),
            "pixel_values bit for bit"
        );
        // The graph takes image pads where Xiaomi's video processor writes video pads.
        let prompt = task_prompt(&m, step, &string("task")).unwrap();
        assert_eq!(
            prompt.replace("<|image_pad|>", "<|video_pad|>"),
            string("prompt")
        );
        assert_eq!(host.state(&f32s("state14")), f32s("state"));
        // Frames out of order change the pixels: each slot takes its own pair.
        let mut swapped = images.clone();
        swapped.swap(0, 2);
        assert_ne!(
            host.pixel_values(&layout, &swapped).unwrap(),
            f32s("pixel_values")
        );
        let short = &images[1..];
        assert!(host
            .pixel_values(&layout, short)
            .unwrap_err()
            .to_string()
            .contains("camera \"agentview_left\" has 3 frames; this model takes 4"));
    }

    #[test]
    fn xr1_prompt_has_a_timestamp_per_temporal_patch() {
        let p = render_xr1_prompt(
            &["Left camera".into(), "Wrist camera".into()],
            2,
            2,
            1,
            "go",
        );
        assert_eq!(
            p,
            "<|im_start|>user\nLeft camera: <0.0 seconds><|vision_start|><|image_pad|><|vision_end|>\
             <0.1 seconds><|vision_start|><|image_pad|><|vision_end|>\n\
             Wrist camera: <0.0 seconds><|vision_start|><|image_pad|><|vision_end|>\
             <0.1 seconds><|vision_start|><|image_pad|><|vision_end|>\n\n\
             Generate robot actions for the task:\ngo /no_cot<|im_end|>\n\
             <|im_start|>assistant\n<cot></cot><|im_end|>\n"
        );
        // Eight frames: patch k is stamped (2k + 2k+1) / 2 / 24 s.
        let p = render_xr1_prompt(&["A".into()], 4, 2, 1, "go");
        for t in [
            "<0.0 seconds>",
            "<0.1 seconds>",
            "<0.2 seconds>",
            "<0.3 seconds>",
        ] {
            assert!(p.contains(t), "{t}: {p}");
        }
    }

    #[test]
    fn xr1_state_rows_are_padded_and_rounded_to_bf16() {
        let m = xr1_32px();
        let ImageStep::Xr0(step) = m.image_step() else {
            panic!("xr0 is the image step")
        };
        let host = Xr0Host::new(&m, step).unwrap();
        let raw: Vec<f32> = (0..4 * 14).map(|i| 1.0 + i as f32 / 1000.0).collect();
        let s = host.state(&raw);
        assert_eq!(s.len(), 4 * 60);
        for (r, row) in s.chunks(60).enumerate() {
            assert_eq!(row[14..], [0.0; 46]);
            // bf16 keeps 8 significant bits: 1.001 rounds to 1.0, 1.027 to 1.0234375.
            assert_eq!(row[0], half::bf16::from_f32(raw[r * 14]).to_f32());
        }
        assert_eq!(s[0], 1.0);
        assert_eq!(s[60 + 13], 1.0234375);
    }

    #[test]
    fn libero_frames_are_not_resized_and_others_are_refused() {
        assert_eq!(resize_target(256, 256, 32, 90_000), [256, 256]);
        assert_eq!(resize_target(480, 640, 32, 90_000), [256, 320]);
        assert_eq!(resize_target(250, 250, 32, 90_000), [256, 256]);
        assert_eq!(resize_target(10, 10, 32, 90_000), [32, 32]);
    }

    #[test]
    fn view_titles_follow_intel() {
        assert_eq!(view_title("base"), "Base");
        assert_eq!(view_title("wrist_left"), "Left-Wrist");
        assert_eq!(view_title("wrist-right"), "Right-Wrist");
        assert_eq!(view_title("top_CAM"), "Top Cam");
    }

    #[test]
    fn the_prompt_is_the_chat_template() {
        let p = render_prompt(&["base".into()], 2, "go");
        assert_eq!(
            p,
            "<|im_start|>user\nThe following observations are captured from multiple views.\n\
             # Base View\n<|vision_start|><|image_pad|><|image_pad|><|vision_end|>\n\
             Generate robot actions for the task:\ngo /no_cot<|im_end|>\n\
             <|im_start|>assistant\n<cot></cot><|im_end|>\n"
        );
    }

    #[test]
    fn denormalize_is_per_row_with_eps_then_sliced() {
        let step = Xr0DenormalizeStep {
            action_mean: vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.0, 0.0]],
            action_std: vec![vec![0.5, 1.0, 1.0], vec![2.0, 1.0, 1.0]],
            action_dim: Some(2),
            action_eps: 1e-6,
        };
        let got = denormalize(&step, &[1.0, 1.0, 9.0, 1.0, -1.0, 9.0], 3);
        let want = [
            1.0 * (0.5 + 1e-6f32) + 1.0,
            1.0 * (1.0 + 1e-6f32) + 2.0,
            1.0 * (2.0 + 1e-6f32) - 1.0,
            -(1.0 + 1e-6f32),
        ];
        assert_eq!(got, want);
    }
}
