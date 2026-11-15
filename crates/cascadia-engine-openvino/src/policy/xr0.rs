//! XR0 (Xiaomi-Robotics-0) host steps, ported from Intel's `physicalai` runtime
//! at `e3dd944` (`preprocessors/xr0.py`, `postprocessors/xr0.py`).
//!
//! Frames must arrive at the size Intel's resize leaves alone (256×256 for
//! LIBERO), so its OpenCV resize never runs here.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::Image;

use super::cameras::CameraLayout;
use super::manifest::{FeatureType, PolicyManifest, Xr0DenormalizeStep, Xr0Step};
use super::qwen_vl::PatchGeometry;

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("policy xr0: {msg}")))
}

/// What the package's features fix: the frame size, the views and the state length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xr0Geometry {
    /// `[height, width]` every camera sends.
    pub hw: [usize; 2],
    /// View names for the prompt (`images.<view>`), in camera order.
    pub views: Vec<String>,
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
/// the state length the STATE feature; refused where Intel would resize.
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
    if target != hw {
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
        state_len,
        patches: PatchGeometry {
            hw,
            patch: step.patch_size,
            merge: step.merge_size,
            temporal: step.temporal_patch_size,
        },
    })
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

/// One package's `xr0` geometry and normalization.
#[derive(Clone, Debug)]
pub struct Xr0Host {
    pub geo: Xr0Geometry,
    mean: [f32; 3],
    std: [f32; 3],
    rescale: f32,
    max_state_dim: usize,
}

impl Xr0Host {
    pub fn new(m: &PolicyManifest, step: &Xr0Step) -> EngineResult<Self> {
        Ok(Self {
            geo: geometry(m, step)?,
            mean: step.image_mean,
            std: step.image_std,
            rescale: step.rescale_factor,
            max_state_dim: step.max_state_dim,
        })
    }

    pub fn max_state_dim(&self) -> usize {
        self.max_state_dim
    }

    /// `pixel_values [n_cameras * patches, feature]`: `(px * rescale − mean) / std`
    /// in f32 (`_build_pixel_grid`), patchified per camera in slot order.
    pub fn pixel_values(
        &self,
        layout: &CameraLayout,
        images: &[(String, Image)],
    ) -> EngineResult<Vec<f32>> {
        let [h, w] = self.geo.hw;
        let p = &self.geo.patches;
        let mut out = Vec::with_capacity(layout.slots.len() * p.patches() * p.feature());
        let mut chw = vec![0f32; 3 * h * w];
        for slot in &layout.slots {
            let Some(i) = slot else {
                return refuse("xr0 has no dummy camera slots");
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
            for (px, rgb) in img.data.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    chw[c * h * w + px] =
                        (f32::from(rgb[c]) * self.rescale - self.mean[c]) / self.std[c];
                }
            }
            p.patchify(&chw, &mut out);
        }
        Ok(out)
    }

    /// `_prepare_state`: zero-padded (or cut) to `max_state_dim`.
    pub fn state(&self, raw: &[f32]) -> Vec<f32> {
        let mut s = raw.to_vec();
        s.resize(self.max_state_dim, 0.0);
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
