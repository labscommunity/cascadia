//! `policy_package` manifest parsing, ported from Intel's `physicalai`
//! runtime (`inference/manifest.py`, `component_factory.py` at `e3dd944`).
//!
//! The loader executes only what it understands: an unknown component, an
//! unknown parameter, an escaping artifact path or placeholder statistics
//! refuse the load.

use std::path::Path;

use cascadia_engine::{EngineError, EngineResult};
use serde::Deserialize;
use serde_json::{Map, Value};

/// The only `format` this loader accepts.
pub const FORMAT: &str = "policy_package";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyKind {
    Pi05,
    Pi0,
    SmolVla,
    Xvla,
    /// EVO1: SmolVLA's ports, cameras and prompt; the task is stripped as LeRobot does.
    Evo1,
    VlaJepa,
    /// Xiaomi-Robotics-0: Qwen3-VL ports like VLA-JEPA's, a raw padded state, `xr0_denormalize`.
    Xr0,
    /// NVIDIA GR00T N1.7: Qwen3-VL ids with mRoPE positions, a constant embodiment id, noise to `actions`.
    Gr00tN1d7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormMode {
    MeanStd,
    MinMax,
    Quantiles,
    /// GR00T's: `2(x − lo)/(hi − lo) − 1`, 0 where `hi` and `lo` are close, clipped to
    /// [−1, 1]; the inverse clips first. `lo`/`hi` are [`StatsStep::bounds`].
    MinMaxClip,
    Identity,
}

/// The stats `min_max_clip` reads as its low and high bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bounds {
    MinMax,
    Quantiles,
}

/// One feature's statistics; only the stats the step's mode reads are checked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureStats {
    pub mean: Option<Vec<f64>>,
    pub std: Option<Vec<f64>>,
    pub q01: Option<Vec<f64>>,
    pub q99: Option<Vec<f64>>,
    pub min: Option<Vec<f64>>,
    pub max: Option<Vec<f64>>,
    /// Where false, the value passes through untransformed.
    pub mask: Option<Vec<bool>>,
}

/// A `normalize` or `denormalize` step, holding only the features it applies to.
#[derive(Clone, Debug, PartialEq)]
pub struct StatsStep {
    pub mode: NormMode,
    /// `(input key, stats)`; the key is `state` for `normalize`, `action` for `denormalize`.
    pub features: Vec<(String, FeatureStats)>,
    /// `min_max_clip` only.
    pub bounds: Bounds,
    /// `normalize` only: zero-pad the result to this width (GR00T's `max_state_dim`).
    pub pad_to: Option<usize>,
}

impl FeatureStats {
    /// `(lo, hi)` as `min_max_clip` reads them.
    pub fn bounds(&self, b: Bounds) -> (&Option<Vec<f64>>, &Option<Vec<f64>>) {
        match b {
            Bounds::MinMax => (&self.min, &self.max),
            Bounds::Quantiles => (&self.q01, &self.q99),
        }
    }
}

impl StatsStep {
    pub fn stats_for(&self, key: &str) -> Option<&FeatureStats> {
        self.features.iter().find(|(k, _)| k == key).map(|(_, s)| s)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pi05Step {
    /// `[height, width]`.
    pub image_resolution: Option<[u32; 2]>,
    pub empty_cameras: usize,
    pub noise_shape: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SmolVlaResizeStep {
    /// `[height, width]`.
    pub image_resolution: Option<[u32; 2]>,
    /// Image key as written in the manifest to camera slot index.
    pub image_key_reorder_map: Vec<(String, i64)>,
    pub num_cameras: i64,
    /// Ours: clients stretch frames to `image_resolution` (`"resize": "stretch"`)
    /// instead of letterboxing them top-left.
    pub stretch: bool,
}

/// VLA-JEPA's host inputs (Intel's `VLAJEPAPreprocessor` without the
/// prompt, tokenizer and noise, which are their own steps here).
#[derive(Clone, Debug, PartialEq)]
pub struct VlaJepaInputsStep {
    /// Camera feature keys (`images.<name>`) in training order.
    pub image_keys: Vec<String>,
    /// The token whose positions the action head reads.
    pub embodied_token_id: i64,
    pub num_embodied_tokens: usize,
    /// `[height, width]` frames arrive at.
    pub image_resolution: [u32; 2],
    /// `[height, width]` the vision encoder takes; a multiple of `patch_size * merge_size`.
    pub vision_resolution: [u32; 2],
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
}

/// Intel's `xr0` preprocessor: Qwen3-VL `pixel_values`, the multi-view chat
/// prompt and the padded state.
#[derive(Clone, Debug, PartialEq)]
pub struct Xr0Step {
    pub max_state_dim: usize,
    /// Intel resizes each side to a multiple of this within `image_max_pixels`;
    /// the engine takes only frames that resize leaves as they are.
    pub image_factor: usize,
    pub image_max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f32,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
}

/// GR00T's eval image transform (`image_augmentations.py`): letterbox to a
/// square, shortest edge to `shortest_edge`, a centred `crop_fraction` crop,
/// back to `shortest_edge` with OpenCV's `INTER_AREA`.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tImageStep {
    pub letterbox: bool,
    pub shortest_edge: usize,
    pub crop_fraction: f64,
}

/// Qwen-VL `pixel_values` from `/255`, `(x − mean)/std` frames: the patch layout of [`super::qwen_vl`].
#[derive(Clone, Debug, PartialEq)]
pub struct QwenVlPatchifyStep {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// `[t, h / patch, w / patch]` per camera, as the export fixed it.
    pub grid_thw: Vec<[usize; 3]>,
}

/// GR00T's prompt: `template` with `{images}` (one `image` per camera) and `{task}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tPromptStep {
    /// Lowercase, then drop every character but `\w` and `\s` (`formalize_language`).
    pub formalize_language: bool,
    pub template: String,
    pub image: String,
}

/// Qwen-VL text inputs from unpadded ids: each image token repeated
/// `tokens_per_image` times, right-padded to `seq_len`, Qwen3-VL mRoPE positions.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenVlInputsStep {
    pub image_token_id: i64,
    pub tokens_per_image: usize,
    pub seq_len: usize,
    pub pad_id: i64,
    pub merge_size: usize,
}

/// Intel's `xr0_denormalize`: `action * (std + eps) + mean` with per-timestep
/// stats, then the first `action_dim` values of each row.
#[derive(Clone, Debug, PartialEq)]
pub struct Xr0DenormalizeStep {
    /// `[steps][width]`, as the graph's action output.
    pub action_mean: Vec<Vec<f32>>,
    pub action_std: Vec<Vec<f32>>,
    pub action_dim: Option<usize>,
    pub action_eps: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreStep {
    Normalize(StatsStep),
    Pi05(Pi05Step),
    /// π0: π0.5's image handling; the prompt stays the task (no state bins).
    Pi0(Pi05Step),
    SmolVlaResize(SmolVlaResizeStep),
    NewLine,
    Noise {
        shape: Vec<usize>,
        /// The input it feeds; `noise` when unset (GR00T: `actions`).
        port: Option<String>,
    },
    OvTokenizer {
        artifact: String,
    },
    /// Rotate the named cameras' frames by 180 degrees before the model.
    ImageFlip {
        cameras: Vec<String>,
    },
    /// The prompt is `template` with every `{instruction}` replaced by the task.
    VlaJepaPrompt {
        template: String,
    },
    VlaJepaInputs(VlaJepaInputsStep),
    Xr0(Xr0Step),
    Gr00tImage(Gr00tImageStep),
    QwenVlPatchify(QwenVlPatchifyStep),
    Gr00tPrompt(Gr00tPromptStep),
    QwenVlInputs(QwenVlInputsStep),
    /// An i64 input the graph takes as a constant (GR00T's `embodiment_id`).
    Constant {
        port: String,
        values: Vec<i64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum PostStep {
    Denormalize(StatsStep),
    ActionChunkTrimmer {
        n_action_steps: usize,
    },
    /// X-VLA's ee6d action to LIBERO's 7-D one (LeRobot `XVLARotation6DToAxisAngleProcessorStep`).
    XvlaRotation6dToAxisAngle {
        expected_action_dim: usize,
    },
    /// Ours: `a[.., index] = above if a > threshold else below` (LeRobot EVO1's LIBERO gripper).
    GripperBinarize {
        index: usize,
        threshold: f64,
        above: f64,
        below: f64,
    },
    Xr0Denormalize(Xr0DenormalizeStep),
    /// Keep the first `dims` values of each action row.
    ActionSlice {
        dims: usize,
    },
}

/// Width of the X-VLA rotation step's output: position, axis-angle, gripper.
pub const XVLA_ACTION_DIM: usize = 7;

/// Values of an action row the X-VLA rotation step reads: position (3), 6-D rotation (6), gripper (1).
pub const XVLA_ROW_READ: usize = 10;

impl PostStep {
    /// The action width after this step, for an input `dim` wide.
    pub fn out_dim(&self, dim: usize) -> EngineResult<usize> {
        match self {
            Self::Denormalize(_) | Self::ActionChunkTrimmer { .. } => Ok(dim),
            Self::ActionSlice { dims } if *dims > dim => refuse(format!(
                "action_slice keeps {dims} values of actions {dim} wide"
            )),
            Self::ActionSlice { dims } => Ok(*dims),
            Self::Xr0Denormalize(s) => {
                let width = s.action_mean.first().map_or(0, Vec::len);
                if width != dim {
                    return refuse(format!(
                        "xr0_denormalize stats are {width} wide; the actions are {dim}"
                    ));
                }
                match s.action_dim {
                    Some(d) if d > dim => refuse(format!(
                        "xr0_denormalize action_dim {d} is wider than the actions ({dim})"
                    )),
                    Some(d) => Ok(d),
                    None => Ok(dim),
                }
            }
            Self::GripperBinarize { index, .. } => {
                if *index >= dim {
                    return refuse(format!(
                        "gripper_binarize index {index} is outside actions {dim} wide"
                    ));
                }
                Ok(dim)
            }
            Self::XvlaRotation6dToAxisAngle {
                expected_action_dim,
            } => {
                let need = (*expected_action_dim).max(XVLA_ROW_READ);
                if dim < need {
                    return refuse(format!(
                        "xvla_rotation_6d_to_axis_angle needs actions at least {need} wide; \
                         they are {dim}"
                    ));
                }
                Ok(XVLA_ACTION_DIM)
            }
        }
    }
}

/// The step that turns frames into the `images` tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageStep<'a> {
    /// `pi05` or `pi0`: centre padding, `empty_cameras` dummy slots.
    Pi05(&'a Pi05Step),
    SmolVlaResize(&'a SmolVlaResizeStep),
    VlaJepa(&'a VlaJepaInputsStep),
    Xr0(&'a Xr0Step),
    Gr00t(&'a Gr00tImageStep),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum FeatureType {
    Visual,
    Action,
    State,
    Language,
    Common,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Feature {
    pub ftype: FeatureType,
    pub name: String,
    #[serde(default)]
    pub shape: Vec<i64>,
    #[serde(default)]
    pub dtype: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyManifest {
    pub policy: PolicyKind,
    pub version: String,
    /// `model.artifacts.openvino`, confined to the package directory.
    pub model_artifact: String,
    pub preprocessors: Vec<PreStep>,
    pub postprocessors: Vec<PostStep>,
    pub input_features: Vec<Feature>,
    pub output_features: Vec<Feature>,
    /// Licence notice served in the metadata: the package's `NOTICE` file, else
    /// `metadata.attribution`, else `policy.source.attribution`.
    pub attribution: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    /// Accept identity (placeholder) normalization stats; for testing only.
    pub allow_identity_stats: bool,
}

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!(
        "policy manifest: {msg}"
    )))
}

impl PolicyManifest {
    /// Read and parse `<dir>/manifest.json`.
    pub fn load(dir: &Path, opts: LoadOptions) -> EngineResult<Self> {
        let path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| EngineError::ModelNotFound(format!("{}: {e}", path.display())))?;
        let mut m = Self::parse(&text, opts)?;
        if let Ok(notice) = std::fs::read_to_string(dir.join("NOTICE")) {
            let notice = notice.trim();
            if !notice.is_empty() {
                m.attribution = Some(notice.to_string());
            }
        }
        Ok(m)
    }

    pub fn parse(json: &str, opts: LoadOptions) -> EngineResult<Self> {
        let root: Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => return refuse(format!("not JSON: {e}")),
        };
        let Some(root) = root.as_object() else {
            return refuse("top level is not an object");
        };
        match root.get("format").and_then(Value::as_str) {
            Some(FORMAT) => {}
            other => return refuse(format!("format {other:?}; expected {FORMAT:?}")),
        }
        let version = match root.get("version").and_then(Value::as_str) {
            Some(v) if is_v1(v) => v.to_string(),
            other => return refuse(format!("version {other:?}; only 1.x is supported")),
        };
        let policy = policy_kind(root.get("policy"))?;
        let empty = Map::new();
        let model = match root.get("model") {
            Some(Value::Object(m)) => m,
            None => &empty,
            Some(_) => return refuse("model is not an object"),
        };

        if let Some(runner) = model.get("runner").filter(|v| !v.is_null()) {
            let (name, params) = component(runner, "runner")?;
            if name != "single_pass" {
                return refuse(format!("runner {name:?}; only SinglePass is supported"));
            }
            no_params(&params, "runner")?;
        }

        let model_artifact = match model
            .get("artifacts")
            .and_then(|a| a.get("openvino"))
            .and_then(Value::as_str)
        {
            Some(p) => confine_artifact(p)?,
            None => return refuse("model.artifacts.openvino is missing"),
        };

        let preprocessors = list(model, "preprocessors")?
            .iter()
            .map(pre_step)
            .collect::<EngineResult<Vec<_>>>()?;
        let postprocessors = list(model, "postprocessors")?
            .iter()
            .map(post_step)
            .collect::<EngineResult<Vec<_>>>()?;
        let input_features = features(model, "input_features")?;
        let output_features = features(model, "output_features")?;
        let text_at = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_string);
        let attribution =
            text_at(root.get("metadata").and_then(|m| m.get("attribution"))).or_else(|| {
                text_at(
                    root.get("policy")
                        .and_then(|p| p.get("source"))
                        .and_then(|s| s.get("attribution")),
                )
            });

        let m = Self {
            policy,
            version,
            model_artifact,
            preprocessors,
            postprocessors,
            input_features,
            output_features,
            attribution,
        };
        m.check_pipeline()?;
        if !opts.allow_identity_stats {
            m.refuse_placeholder_stats()?;
        }
        Ok(m)
    }

    /// The single image step; `parse` guarantees there is exactly one.
    pub fn image_step(&self) -> ImageStep<'_> {
        self.preprocessors
            .iter()
            .find_map(|s| match s {
                PreStep::Pi05(p) | PreStep::Pi0(p) => Some(ImageStep::Pi05(p)),
                PreStep::SmolVlaResize(p) => Some(ImageStep::SmolVlaResize(p)),
                PreStep::VlaJepaInputs(p) => Some(ImageStep::VlaJepa(p)),
                PreStep::Xr0(p) => Some(ImageStep::Xr0(p)),
                PreStep::Gr00tImage(p) => Some(ImageStep::Gr00t(p)),
                _ => None,
            })
            .expect("parse guarantees one image step")
    }

    pub fn tokenizer_artifact(&self) -> &str {
        self.preprocessors
            .iter()
            .find_map(|s| match s {
                PreStep::OvTokenizer { artifact } => Some(artifact.as_str()),
                _ => None,
            })
            .expect("parse guarantees one ov_tokenizer")
    }

    pub fn n_action_steps(&self) -> Option<usize> {
        self.postprocessors.iter().find_map(|s| match s {
            PostStep::ActionChunkTrimmer { n_action_steps } => Some(*n_action_steps),
            _ => None,
        })
    }

    /// Cameras the `image_flip` step rotates; empty without one.
    pub fn flipped_cameras(&self) -> &[String] {
        self.preprocessors
            .iter()
            .find_map(|s| match s {
                PreStep::ImageFlip { cameras } => Some(cameras.as_slice()),
                _ => None,
            })
            .unwrap_or_default()
    }

    pub fn normalize(&self) -> impl Iterator<Item = &StatsStep> {
        self.preprocessors.iter().filter_map(|s| match s {
            PreStep::Normalize(n) => Some(n),
            _ => None,
        })
    }

    /// GR00T's text and pixel steps; `parse` guarantees all four for `gr00t_n1d7`.
    pub fn gr00t_steps(
        &self,
    ) -> Option<(
        &Gr00tImageStep,
        &QwenVlPatchifyStep,
        &Gr00tPromptStep,
        &QwenVlInputsStep,
    )> {
        let find = |f: fn(&PreStep) -> bool| self.preprocessors.iter().find(|s| f(s));
        match (
            find(|s| matches!(s, PreStep::Gr00tImage(_))),
            find(|s| matches!(s, PreStep::QwenVlPatchify(_))),
            find(|s| matches!(s, PreStep::Gr00tPrompt(_))),
            find(|s| matches!(s, PreStep::QwenVlInputs(_))),
        ) {
            (
                Some(PreStep::Gr00tImage(i)),
                Some(PreStep::QwenVlPatchify(p)),
                Some(PreStep::Gr00tPrompt(t)),
                Some(PreStep::QwenVlInputs(q)),
            ) => Some((i, p, t, q)),
            _ => None,
        }
    }

    /// `(port, values)` of each `constant` step, in manifest order.
    pub fn constants(&self) -> impl Iterator<Item = (&str, &[i64])> {
        self.preprocessors.iter().filter_map(|s| match s {
            PreStep::Constant { port, values } => Some((port.as_str(), values.as_slice())),
            _ => None,
        })
    }

    /// The input the noise goes to: the `noise` step's `port`, else `noise`.
    pub fn noise_port(&self) -> &str {
        self.preprocessors
            .iter()
            .find_map(|s| match s {
                PreStep::Noise { port: Some(p), .. } => Some(p.as_str()),
                _ => None,
            })
            .unwrap_or("noise")
    }

    pub fn denormalize(&self) -> impl Iterator<Item = &StatsStep> {
        self.postprocessors.iter().filter_map(|s| match s {
            PostStep::Denormalize(n) => Some(n),
            _ => None,
        })
    }

    fn check_pipeline(&self) -> EngineResult<()> {
        let count = |f: fn(&PreStep) -> bool| self.preprocessors.iter().filter(|s| f(s)).count();
        let images = count(|s| {
            matches!(
                s,
                PreStep::Pi05(_)
                    | PreStep::Pi0(_)
                    | PreStep::SmolVlaResize(_)
                    | PreStep::VlaJepaInputs(_)
                    | PreStep::Xr0(_)
                    | PreStep::Gr00tImage(_)
            )
        });
        if images != 1 {
            return refuse(format!(
                "{images} image steps (pi05 / pi0 / smolvla_resize / vla_jepa_inputs / xr0 / \
                 gr00t_image); expected exactly one"
            ));
        }
        let tokenizers = count(|s| matches!(s, PreStep::OvTokenizer { .. }));
        if tokenizers != 1 {
            return refuse(format!(
                "{tokenizers} ov_tokenizer steps; expected exactly one"
            ));
        }
        if count(|s| matches!(s, PreStep::ImageFlip { .. })) > 1 {
            return refuse("more than one image_flip");
        }
        let trimmers = self
            .postprocessors
            .iter()
            .filter(|s| matches!(s, PostStep::ActionChunkTrimmer { .. }))
            .count();
        if trimmers > 1 {
            return refuse("more than one action_chunk_trimmer");
        }
        let rotations = self
            .postprocessors
            .iter()
            .filter(|s| matches!(s, PostStep::XvlaRotation6dToAxisAngle { .. }))
            .count();
        if rotations > 1 {
            return refuse("more than one xvla_rotation_6d_to_axis_angle");
        }
        self.check_vla_jepa()?;
        self.check_xr0()?;
        self.check_gr00t()
    }

    /// GR00T's steps are its own, and its normalize pads, so only GR00T's
    /// state port takes them.
    fn check_gr00t(&self) -> EngineResult<()> {
        let own = |s: &PreStep| {
            matches!(
                s,
                PreStep::Gr00tImage(_)
                    | PreStep::QwenVlPatchify(_)
                    | PreStep::Gr00tPrompt(_)
                    | PreStep::QwenVlInputs(_)
                    | PreStep::Constant { .. }
            )
        };
        let padded = self.normalize().any(|s| s.pad_to.is_some())
            || self
                .preprocessors
                .iter()
                .any(|s| matches!(s, PreStep::Noise { port: Some(_), .. }));
        if self.policy != PolicyKind::Gr00tN1d7 {
            if let Some(s) = self.preprocessors.iter().find(|s| own(s)) {
                return refuse(format!("{} needs policy.name \"gr00t_n1d7\"", pre_name(s)));
            }
            if padded {
                return refuse("normalize pad_to and a noise port need policy.name \"gr00t_n1d7\"");
            }
            if self
                .postprocessors
                .iter()
                .any(|s| matches!(s, PostStep::ActionSlice { .. }))
            {
                return refuse("action_slice needs policy.name \"gr00t_n1d7\"");
            }
            return Ok(());
        }
        for (name, f) in [
            (
                "qwen_vl_patchify",
                (|s: &PreStep| matches!(s, PreStep::QwenVlPatchify(_))) as fn(&PreStep) -> bool,
            ),
            ("gr00t_prompt", |s| matches!(s, PreStep::Gr00tPrompt(_))),
            ("qwen_vl_inputs", |s| matches!(s, PreStep::QwenVlInputs(_))),
        ] {
            let n = self.preprocessors.iter().filter(|s| f(s)).count();
            if n != 1 {
                return refuse(format!("{n} {name} steps; gr00t_n1d7 needs exactly one"));
            }
        }
        if let Some(s) = self.preprocessors.iter().find(|s| {
            !own(s)
                && !matches!(
                    s,
                    PreStep::ImageFlip { .. }
                        | PreStep::Normalize(_)
                        | PreStep::OvTokenizer { .. }
                        | PreStep::Noise { .. }
                )
        }) {
            return refuse(format!(
                "gr00t_n1d7 takes only image_flip, gr00t_image, qwen_vl_patchify, normalize, \
                 gr00t_prompt, ov_tokenizer, qwen_vl_inputs, constant and noise; found {}",
                pre_name(s)
            ));
        }
        if self.normalize().count() > 1 {
            return refuse("gr00t_n1d7 takes at most one normalize");
        }
        if let Some(s) = self.postprocessors.iter().find(|s| {
            !matches!(
                s,
                PostStep::ActionChunkTrimmer { .. }
                    | PostStep::ActionSlice { .. }
                    | PostStep::Denormalize(_)
            )
        }) {
            return refuse(format!(
                "gr00t_n1d7 takes only action_chunk_trimmer, action_slice and denormalize; found {}",
                post_name(s)
            ));
        }
        super::gr00t::Gr00tGeometry::new(self).map(drop)
    }

    /// `xr0` and `xr0_denormalize` are XR0's alone, and an XR0 graph takes no
    /// other host step: its state goes in raw.
    fn check_xr0(&self) -> EngineResult<()> {
        let pre = |s: &PreStep| matches!(s, PreStep::Xr0(_));
        let post = |s: &PostStep| matches!(s, PostStep::Xr0Denormalize(_));
        if self.policy != PolicyKind::Xr0 {
            if self.preprocessors.iter().any(pre) || self.postprocessors.iter().any(post) {
                return refuse("xr0 and xr0_denormalize need policy.name \"xr0\"");
            }
            return Ok(());
        }
        if !self.preprocessors.iter().any(pre) {
            return refuse("xr0 needs an xr0 preprocessor");
        }
        if let Some(s) = self
            .preprocessors
            .iter()
            .find(|s| !pre(s) && !matches!(s, PreStep::OvTokenizer { .. } | PreStep::Noise { .. }))
        {
            return refuse(format!(
                "xr0 takes only xr0, noise and ov_tokenizer; found {}",
                pre_name(s)
            ));
        }
        if let Some(s) = self
            .postprocessors
            .iter()
            .find(|s| !post(s) && !matches!(s, PostStep::ActionChunkTrimmer { .. }))
        {
            return refuse(format!(
                "xr0 takes only xr0_denormalize and action_chunk_trimmer; found {}",
                post_name(s)
            ));
        }
        if self.postprocessors.iter().filter(|s| post(s)).count() != 1 {
            return refuse("xr0 needs exactly one xr0_denormalize");
        }
        let ImageStep::Xr0(step) = self.image_step() else {
            unreachable!("an xr0 step is the image step")
        };
        super::xr0::geometry(self, step).map(drop)
    }

    /// VLA-JEPA's graph normalizes the state and returns final actions, so a
    /// stats step or an action transform would be applied twice; its steps
    /// belong to it alone.
    fn check_vla_jepa(&self) -> EngineResult<()> {
        let jepa =
            |s: &PreStep| matches!(s, PreStep::VlaJepaPrompt { .. } | PreStep::VlaJepaInputs(_));
        if self.policy != PolicyKind::VlaJepa {
            return match self.preprocessors.iter().find(|s| jepa(s)) {
                Some(_) => {
                    refuse("vla_jepa_prompt and vla_jepa_inputs need policy.name \"vlajepa\"")
                }
                None => Ok(()),
            };
        }
        let prompts = self
            .preprocessors
            .iter()
            .filter(|s| matches!(s, PreStep::VlaJepaPrompt { .. }))
            .count();
        if prompts != 1 {
            return refuse(format!(
                "{prompts} vla_jepa_prompt steps; vlajepa needs exactly one"
            ));
        }
        if let Some(s) = self
            .preprocessors
            .iter()
            .find(|s| !jepa(s) && !matches!(s, PreStep::OvTokenizer { .. } | PreStep::Noise { .. }))
        {
            return refuse(format!(
                "vlajepa takes only vla_jepa_prompt, ov_tokenizer, vla_jepa_inputs and noise; \
                 found {}",
                pre_name(s)
            ));
        }
        if let Some(s) = self
            .postprocessors
            .iter()
            .find(|s| !matches!(s, PostStep::ActionChunkTrimmer { .. }))
        {
            return refuse(format!(
                "vlajepa's graph returns final actions; found postprocessor {}",
                post_name(s)
            ));
        }
        Ok(())
    }

    /// An export that never had stats fills in identity values; serving it
    /// would drive a robot with unnormalized numbers.
    fn refuse_placeholder_stats(&self) -> EngineResult<()> {
        let steps = self
            .normalize()
            .map(|s| ("normalize", s))
            .chain(self.denormalize().map(|s| ("denormalize", s)));
        let xr0_identity = self.postprocessors.iter().any(|s| match s {
            PostStep::Xr0Denormalize(d) => {
                d.action_mean.iter().flatten().all(|&x| x == 0.0)
                    && d.action_std.iter().flatten().all(|&x| x == 1.0)
            }
            _ => false,
        });
        if xr0_identity {
            return refuse(
                "xr0_denormalize stats are identity placeholders (the export had no \
                 normalization stats); re-export with real stats, or pass \
                 --allow-identity-stats for testing",
            );
        }
        for (kind, step) in steps {
            if is_placeholder(step) {
                return refuse(format!(
                    "{kind} stats are identity placeholders (the export had no \
                     normalization stats); re-export with real stats, or pass \
                     --allow-identity-stats for testing"
                ));
            }
        }
        Ok(())
    }
}

fn pre_name(s: &PreStep) -> &'static str {
    match s {
        PreStep::Normalize(_) => "normalize",
        PreStep::Pi05(_) => "pi05",
        PreStep::Pi0(_) => "pi0",
        PreStep::SmolVlaResize(_) => "smolvla_resize",
        PreStep::NewLine => "new_line",
        PreStep::Noise { .. } => "noise",
        PreStep::OvTokenizer { .. } => "ov_tokenizer",
        PreStep::ImageFlip { .. } => "image_flip",
        PreStep::VlaJepaPrompt { .. } => "vla_jepa_prompt",
        PreStep::VlaJepaInputs(_) => "vla_jepa_inputs",
        PreStep::Xr0(_) => "xr0",
        PreStep::Gr00tImage(_) => "gr00t_image",
        PreStep::QwenVlPatchify(_) => "qwen_vl_patchify",
        PreStep::Gr00tPrompt(_) => "gr00t_prompt",
        PreStep::QwenVlInputs(_) => "qwen_vl_inputs",
        PreStep::Constant { .. } => "constant",
    }
}

fn post_name(s: &PostStep) -> &'static str {
    match s {
        PostStep::Denormalize(_) => "denormalize",
        PostStep::ActionChunkTrimmer { .. } => "action_chunk_trimmer",
        PostStep::XvlaRotation6dToAxisAngle { .. } => "xvla_rotation_6d_to_axis_angle",
        PostStep::GripperBinarize { .. } => "gripper_binarize",
        PostStep::Xr0Denormalize(_) => "xr0_denormalize",
        PostStep::ActionSlice { .. } => "action_slice",
    }
}

fn is_placeholder(step: &StatsStep) -> bool {
    let all =
        |v: &Option<Vec<f64>>, want: f64| v.as_ref().is_some_and(|v| v.iter().all(|&x| x == want));
    let identity = |s: &FeatureStats| match step.mode {
        NormMode::MeanStd => all(&s.mean, 0.0) && all(&s.std, 1.0),
        NormMode::Quantiles => all(&s.q01, -1.0) && all(&s.q99, 1.0),
        NormMode::MinMax => all(&s.min, -1.0) && all(&s.max, 1.0),
        NormMode::MinMaxClip => {
            let (lo, hi) = s.bounds(step.bounds);
            all(lo, -1.0) && all(hi, 1.0)
        }
        NormMode::Identity => false,
    };
    !step.features.is_empty() && step.features.iter().all(|(_, s)| identity(s))
}

/// `1`, `1.0`, `1.2.3`: major version 1 only.
fn is_v1(v: &str) -> bool {
    let mut parts = v.split('.');
    parts.next() == Some("1")
        && parts.all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

/// Accept only a plain relative name inside the package directory.
///
/// Lexical on purpose, like Intel's check: no filesystem access, so a Hub
/// snapshot's symlinks still resolve. Stricter than Intel's, which accepts an
/// absolute path that happens to lie inside the package.
pub fn confine_artifact(path: &str) -> EngineResult<String> {
    let why = if path.is_empty() {
        Some("is empty")
    } else if path.contains('\0') {
        Some("contains NUL")
    } else if path.starts_with("\\\\") || path.starts_with("//") {
        Some("is a UNC path")
    } else if path.starts_with('/') || path.starts_with('\\') {
        Some("is absolute")
    } else if path.contains(':') {
        // `C:x` and `C:\x` are drive paths; `a:b` is an NTFS stream.
        Some("has a drive letter or stream (':')")
    } else if path.split(['/', '\\']).any(|c| c == "..") {
        Some("escapes the package directory ('..')")
    } else {
        None
    };
    match why {
        Some(why) => refuse(format!("artifact path {path:?} {why}")),
        None => Ok(path.to_string()),
    }
}

/// `policy.name` is required: Intel's class_path fallback takes the
/// second-to-last segment, which is `policy` for its own paths
/// (`physicalai.policies.pi05.policy.Pi05`).
fn policy_kind(policy: Option<&Value>) -> EngineResult<PolicyKind> {
    match policy.and_then(|p| p.get("name")).and_then(Value::as_str) {
        Some("pi05") => Ok(PolicyKind::Pi05),
        Some("pi0") => Ok(PolicyKind::Pi0),
        Some("smolvla") => Ok(PolicyKind::SmolVla),
        Some("xvla") => Ok(PolicyKind::Xvla),
        Some("evo1") => Ok(PolicyKind::Evo1),
        Some("vlajepa") => Ok(PolicyKind::VlaJepa),
        Some("xr0") => Ok(PolicyKind::Xr0),
        Some("gr00t_n1d7") => Ok(PolicyKind::Gr00tN1d7),
        other => refuse(format!(
            "policy.name {other:?}; supported: pi05, pi0, smolvla, xvla, evo1, vlajepa, xr0, \
             gr00t_n1d7"
        )),
    }
}

/// Intel's class names for the components this engine implements, by short
/// name. `None`: a step of ours with no Intel class, accepted in the `type`
/// form only, so an Intel class of another meaning never maps to it.
const COMPONENTS: &[(&str, Option<&str>)] = &[
    ("single_pass", Some("SinglePass")),
    ("normalize", Some("StatsNormalizer")),
    ("pi05", Some("Pi05Preprocessor")),
    ("pi0", None),
    ("smolvla_resize", Some("ResizeSmolVLA")),
    ("new_line", Some("NewLinePreprocessor")),
    ("noise", Some("NoisePreprocessor")),
    ("ov_tokenizer", Some("OVTokenizer")),
    ("denormalize", Some("StatsDenormalizer")),
    ("action_chunk_trimmer", Some("ActionChunkTrimmer")),
    ("image_flip", None),
    ("xvla_rotation_6d_to_axis_angle", None),
    ("gripper_binarize", None),
    ("vla_jepa_prompt", None),
    ("vla_jepa_inputs", None),
    ("xr0", Some("XR0Preprocessor")),
    ("xr0_denormalize", Some("XR0Postprocessor")),
    ("gr00t_image", None),
    ("qwen_vl_patchify", None),
    ("gr00t_prompt", None),
    ("qwen_vl_inputs", None),
    ("constant", None),
    ("action_slice", None),
];

fn short_name(class_path: &str) -> Option<&'static str> {
    if let Some((short, _)) = COMPONENTS
        .iter()
        .find(|(s, c)| c.is_some() && *s == class_path)
    {
        return Some(short);
    }
    // Manifests use both the package re-export (`...preprocessors.StatsNormalizer`)
    // and the defining module (`...runners.single_pass.SinglePass`).
    let class = class_path
        .strip_prefix("physicalai.inference.")?
        .rsplit('.')
        .next()?;
    COMPONENTS
        .iter()
        .find(|(_, c)| *c == Some(class))
        .map(|(s, _)| *s)
}

/// Resolve either component form to `(short name, params)`; `class_path` wins.
fn component(v: &Value, ctx: &str) -> EngineResult<(&'static str, Map<String, Value>)> {
    let Some(obj) = v.as_object() else {
        return refuse(format!("{ctx}: component is not an object"));
    };
    let non_empty = |k: &str| obj.get(k).and_then(Value::as_str).filter(|s| !s.is_empty());
    if let Some(cp) = non_empty("class_path") {
        let Some(name) = short_name(cp) else {
            return refuse(format!("{ctx}: unsupported component class_path {cp:?}"));
        };
        let params = match obj.get("init_args") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => return refuse(format!("{ctx}: init_args is not an object")),
        };
        return Ok((name, params));
    }
    if let Some(ty) = non_empty("type") {
        let Some((name, _)) = COMPONENTS.iter().find(|(s, _)| *s == ty) else {
            return refuse(format!("{ctx}: unsupported component type {ty:?}"));
        };
        let mut params = obj.clone();
        for k in ["type", "class_path", "init_args"] {
            params.remove(k);
        }
        return Ok((name, params));
    }
    refuse(format!("{ctx}: component has neither type nor class_path"))
}

fn no_params(params: &Map<String, Value>, ctx: &str) -> EngineResult<()> {
    match params.keys().next() {
        Some(k) => refuse(format!("{ctx}: unknown parameter {k:?}")),
        None => Ok(()),
    }
}

fn typed<T: serde::de::DeserializeOwned>(params: Map<String, Value>, ctx: &str) -> EngineResult<T> {
    serde_json::from_value(Value::Object(params)).or_else(|e| refuse(format!("{ctx}: {e}")))
}

fn list<'a>(model: &'a Map<String, Value>, key: &str) -> EngineResult<&'a [Value]> {
    match model.get(key) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(a)) => Ok(a),
        Some(_) => refuse(format!("model.{key} is not a list")),
    }
}

fn features(model: &Map<String, Value>, key: &str) -> EngineResult<Vec<Feature>> {
    list(model, key)?
        .iter()
        .map(|v| {
            let ctx = format!("model.{key}");
            let cp = v.get("class_path").and_then(Value::as_str).unwrap_or("");
            if !(cp.starts_with("physicalai.") && cp.ends_with(".InferenceFeature")) {
                return refuse(format!("{ctx}: unsupported feature class_path {cp:?}"));
            }
            let args = v.get("init_args").cloned().unwrap_or(Value::Null);
            serde_json::from_value(args).or_else(|e| refuse(format!("{ctx}: {e}")))
        })
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StatsParams {
    #[serde(default = "mean_std")]
    mode: String,
    #[serde(default)]
    stats: Option<Map<String, Value>>,
    #[serde(default)]
    features: Option<Vec<String>>,
    #[serde(default)]
    stats_path: Option<String>,
    #[serde(default)]
    artifact: Option<String>,
    #[serde(default)]
    bounds: Option<Vec<String>>,
    #[serde(default)]
    pad_to: Option<i64>,
    #[serde(default)]
    history: Option<i64>,
}

fn mean_std() -> String {
    "mean_std".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pi05Params {
    #[serde(default)]
    image_resolution: Option<[u32; 2]>,
    #[serde(default)]
    empty_cameras: i64,
    #[serde(default)]
    noise_shape: Option<Vec<usize>>,
    // Accepted for compatibility; each call carries its own seed.
    #[serde(default)]
    #[allow(dead_code)]
    seed: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SmolVlaParams {
    #[serde(default)]
    image_resolution: Option<[u32; 2]>,
    #[serde(default)]
    image_key_reorder_map: Option<Map<String, Value>>,
    #[serde(default)]
    num_cameras: i64,
    // The wire format fixes HWC, so the package's input layout does not apply.
    #[serde(default)]
    image_layout: Option<String>,
    #[serde(default)]
    resize: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Xr0Params {
    max_state_dim: i64,
    image_factor: i64,
    image_max_pixels: i64,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    rescale_factor: f64,
    patch_size: i64,
    merge_size: i64,
    temporal_patch_size: i64,
    #[serde(default)]
    image_key_view_map: Option<Map<String, Value>>,
    #[serde(default)]
    normalize_state: bool,
    // Identity unless normalize_state, which is refused.
    #[serde(default)]
    #[allow(dead_code)]
    state_mean: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    state_std: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Xr0DenormalizeParams {
    action_mean: Vec<Vec<f64>>,
    action_std: Vec<Vec<f64>>,
    #[serde(default)]
    action_dim: Option<i64>,
    #[serde(default = "absolute")]
    action_mode: String,
    #[serde(default = "xr0_eps")]
    action_eps: f64,
}

fn absolute() -> String {
    "absolute".into()
}

fn xr0_eps() -> f64 {
    1e-6
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GripperBinarizeParams {
    index: i64,
    threshold: f64,
    above: f64,
    below: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoiseParams {
    shape: Vec<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    seed: Option<u64>,
    #[serde(default)]
    port: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Gr00tImageParams {
    letterbox: bool,
    shortest_edge: i64,
    crop_fraction: f64,
    interpolation: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenVlPatchifyParams {
    patch_size: i64,
    temporal_patch_size: i64,
    merge_size: i64,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    grid_thw: Vec<[i64; 3]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Gr00tPromptParams {
    formalize_language: bool,
    template: String,
    image: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenVlInputsParams {
    image_token_id: i64,
    tokens_per_image: i64,
    seq_len: i64,
    pad_id: i64,
    padding_side: String,
    rope: String,
    merge_size: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConstantParams {
    port: String,
    dtype: String,
    value: Vec<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionSliceParams {
    dims: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerParams {
    artifact: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrimmerParams {
    n_action_steps: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageFlipParams {
    cameras: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct XvlaRotationParams {
    #[serde(default = "xvla_row_read")]
    expected_action_dim: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VlaJepaPromptParams {
    template: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VlaJepaInputsParams {
    image_keys: Vec<String>,
    embodied_token_id: i64,
    num_embodied_tokens: i64,
    image_resolution: [u32; 2],
    vision_resolution: [u32; 2],
    image_mean: [f64; 3],
    image_std: [f64; 3],
    patch_size: u32,
    merge_size: u32,
    temporal_patch_size: u32,
}

fn xvla_row_read() -> i64 {
    XVLA_ROW_READ as i64
}

fn pre_step(v: &Value) -> EngineResult<PreStep> {
    let ctx = "model.preprocessors";
    let (name, params) = component(v, ctx)?;
    let ctx = format!("{ctx}: {name}");
    Ok(match name {
        "normalize" => PreStep::Normalize(stats_step(params, &ctx, "state")?),
        "pi05" => PreStep::Pi05(pi05_step(params, &ctx)?),
        "pi0" => PreStep::Pi0(pi05_step(params, &ctx)?),
        "smolvla_resize" => {
            let p: SmolVlaParams = typed(params, &ctx)?;
            check_resolution(p.image_resolution, &ctx)?;
            if let Some(l) = p.image_layout.as_deref() {
                if l != "BCHW" && l != "BHWC" {
                    return refuse(format!("{ctx}: image_layout {l:?}"));
                }
            }
            let mut map = Vec::new();
            for (k, v) in p.image_key_reorder_map.unwrap_or_default() {
                let Some(slot) = v.as_i64() else {
                    return refuse(format!(
                        "{ctx}: image_key_reorder_map[{k:?}] is not an integer"
                    ));
                };
                map.push((k, slot));
            }
            let stretch = match p.resize.as_deref() {
                None | Some("top_left") => false,
                Some("stretch") => true,
                Some(r) => {
                    return refuse(format!("{ctx}: resize {r:?}; expected top_left or stretch"))
                }
            };
            PreStep::SmolVlaResize(SmolVlaResizeStep {
                image_resolution: p.image_resolution,
                image_key_reorder_map: map,
                num_cameras: p.num_cameras,
                stretch,
            })
        }
        "new_line" => {
            no_params(&params, &ctx)?;
            PreStep::NewLine
        }
        "noise" => {
            let p: NoiseParams = typed(params, &ctx)?;
            check_shape(Some(&p.shape), &ctx)?;
            if p.port.as_deref().is_some_and(str::is_empty) {
                return refuse(format!("{ctx}: port is empty"));
            }
            PreStep::Noise {
                shape: p.shape,
                port: p.port,
            }
        }
        "ov_tokenizer" => {
            let p: TokenizerParams = typed(params, &ctx)?;
            PreStep::OvTokenizer {
                artifact: confine_artifact(&p.artifact)?,
            }
        }
        "image_flip" => {
            let p: ImageFlipParams = typed(params, &ctx)?;
            if p.cameras.is_empty() {
                return refuse(format!("{ctx}: cameras is empty"));
            }
            for (i, c) in p.cameras.iter().enumerate() {
                if p.cameras[..i].contains(c) {
                    return refuse(format!("{ctx}: camera {c:?} listed twice"));
                }
            }
            PreStep::ImageFlip { cameras: p.cameras }
        }
        "vla_jepa_prompt" => {
            let p: VlaJepaPromptParams = typed(params, &ctx)?;
            if !p.template.contains("{instruction}") {
                return refuse(format!("{ctx}: template has no {{instruction}}"));
            }
            PreStep::VlaJepaPrompt {
                template: p.template,
            }
        }
        "vla_jepa_inputs" => PreStep::VlaJepaInputs(vla_jepa_inputs_step(params, &ctx)?),
        "xr0" => PreStep::Xr0(xr0_step(params, &ctx)?),
        "gr00t_image" => {
            let p: Gr00tImageParams = typed(params, &ctx)?;
            if p.interpolation != "area" {
                return refuse(format!(
                    "{ctx}: interpolation {:?}; only area (OpenCV INTER_AREA) is supported",
                    p.interpolation
                ));
            }
            if p.shortest_edge < 1 {
                return refuse(format!("{ctx}: shortest_edge must be at least 1"));
            }
            if !(p.crop_fraction > 0.0 && p.crop_fraction <= 1.0) {
                return refuse(format!("{ctx}: crop_fraction must be in (0, 1]"));
            }
            PreStep::Gr00tImage(Gr00tImageStep {
                letterbox: p.letterbox,
                shortest_edge: p.shortest_edge as usize,
                crop_fraction: p.crop_fraction,
            })
        }
        "qwen_vl_patchify" => {
            let p: QwenVlPatchifyParams = typed(params, &ctx)?;
            if [p.patch_size, p.temporal_patch_size, p.merge_size]
                .iter()
                .any(|&v| v < 1)
            {
                return refuse(format!(
                    "{ctx}: patch_size, temporal_patch_size and merge_size must be at least 1"
                ));
            }
            let (mean, std) = (
                p.image_mean.map(|x| x as f32),
                p.image_std.map(|x| x as f32),
            );
            if mean.iter().any(|x| !x.is_finite())
                || std.iter().any(|x| !x.is_finite() || *x == 0.0)
            {
                return refuse(format!(
                    "{ctx}: image_mean and image_std must be finite, image_std non-zero"
                ));
            }
            if p.grid_thw.is_empty() || p.grid_thw.iter().flatten().any(|&d| d < 1) {
                return refuse(format!(
                    "{ctx}: grid_thw must be non-empty, every dim at least 1"
                ));
            }
            PreStep::QwenVlPatchify(QwenVlPatchifyStep {
                patch_size: p.patch_size as usize,
                temporal_patch_size: p.temporal_patch_size as usize,
                merge_size: p.merge_size as usize,
                image_mean: mean,
                image_std: std,
                grid_thw: p.grid_thw.iter().map(|g| g.map(|d| d as usize)).collect(),
            })
        }
        "gr00t_prompt" => {
            let p: Gr00tPromptParams = typed(params, &ctx)?;
            for key in ["{images}", "{task}"] {
                if p.template.matches(key).count() != 1 {
                    return refuse(format!("{ctx}: template needs {key} exactly once"));
                }
            }
            if p.image.is_empty() {
                return refuse(format!("{ctx}: image is empty"));
            }
            PreStep::Gr00tPrompt(Gr00tPromptStep {
                formalize_language: p.formalize_language,
                template: p.template,
                image: p.image,
            })
        }
        "qwen_vl_inputs" => {
            let p: QwenVlInputsParams = typed(params, &ctx)?;
            if p.padding_side != "right" {
                return refuse(format!(
                    "{ctx}: padding_side {:?}; only right is supported",
                    p.padding_side
                ));
            }
            if p.rope != "qwen3_vl_mrope" {
                return refuse(format!(
                    "{ctx}: rope {:?}; only qwen3_vl_mrope is supported",
                    p.rope
                ));
            }
            if p.image_token_id < 0 || p.pad_id < 0 {
                return refuse(format!(
                    "{ctx}: image_token_id and pad_id must not be negative"
                ));
            }
            if [p.tokens_per_image, p.seq_len, p.merge_size]
                .iter()
                .any(|&v| v < 1)
            {
                return refuse(format!(
                    "{ctx}: tokens_per_image, seq_len and merge_size must be at least 1"
                ));
            }
            PreStep::QwenVlInputs(QwenVlInputsStep {
                image_token_id: p.image_token_id,
                tokens_per_image: p.tokens_per_image as usize,
                seq_len: p.seq_len as usize,
                pad_id: p.pad_id,
                merge_size: p.merge_size as usize,
            })
        }
        "constant" => {
            let p: ConstantParams = typed(params, &ctx)?;
            if p.dtype != "int64" {
                return refuse(format!(
                    "{ctx}: dtype {:?}; only int64 is supported",
                    p.dtype
                ));
            }
            if p.port.is_empty() || p.value.is_empty() {
                return refuse(format!("{ctx}: port and value must not be empty"));
            }
            PreStep::Constant {
                port: p.port,
                values: p.value,
            }
        }
        other => return refuse(format!("{ctx}: {other} is not a preprocessor")),
    })
}

fn vla_jepa_inputs_step(params: Map<String, Value>, ctx: &str) -> EngineResult<VlaJepaInputsStep> {
    let p: VlaJepaInputsParams = typed(params, ctx)?;
    if p.image_keys.is_empty() {
        return refuse(format!("{ctx}: image_keys is empty"));
    }
    if p.embodied_token_id < 0 {
        return refuse(format!("{ctx}: embodied_token_id is negative"));
    }
    if p.num_embodied_tokens < 1 {
        return refuse(format!("{ctx}: num_embodied_tokens must be at least 1"));
    }
    if p.patch_size == 0 || p.merge_size == 0 || p.temporal_patch_size == 0 {
        return refuse(format!(
            "{ctx}: patch_size, merge_size and temporal_patch_size must be at least 1"
        ));
    }
    check_resolution(Some(p.image_resolution), ctx)?;
    let unit = p.patch_size * p.merge_size;
    if p.vision_resolution.iter().any(|&d| d == 0 || d % unit != 0) {
        return refuse(format!(
            "{ctx}: vision_resolution {:?} is not a multiple of patch_size * merge_size ({unit})",
            p.vision_resolution
        ));
    }
    let (mean, std) = (
        p.image_mean.map(|x| x as f32),
        p.image_std.map(|x| x as f32),
    );
    if mean.iter().any(|x| !x.is_finite()) {
        return refuse(format!("{ctx}: image_mean {mean:?} is not finite"));
    }
    if std.iter().any(|x| !x.is_finite() || *x == 0.0) {
        return refuse(format!(
            "{ctx}: image_std {std:?} must be finite and non-zero"
        ));
    }
    Ok(VlaJepaInputsStep {
        image_keys: p.image_keys,
        embodied_token_id: p.embodied_token_id,
        num_embodied_tokens: p.num_embodied_tokens as usize,
        image_resolution: p.image_resolution,
        vision_resolution: p.vision_resolution,
        image_mean: mean,
        image_std: std,
        patch_size: p.patch_size as usize,
        merge_size: p.merge_size as usize,
        temporal_patch_size: p.temporal_patch_size as usize,
    })
}

fn xr0_step(params: Map<String, Value>, ctx: &str) -> EngineResult<Xr0Step> {
    let p: Xr0Params = typed(params, ctx)?;
    if p.normalize_state {
        return refuse(format!("{ctx}: normalize_state is not supported"));
    }
    if p.image_key_view_map.is_some_and(|m| !m.is_empty()) {
        return refuse(format!(
            "{ctx}: image_key_view_map is not supported; cameras follow input_features"
        ));
    }
    let positive = [
        ("max_state_dim", p.max_state_dim),
        ("image_factor", p.image_factor),
        ("image_max_pixels", p.image_max_pixels),
        ("patch_size", p.patch_size),
        ("merge_size", p.merge_size),
        ("temporal_patch_size", p.temporal_patch_size),
    ];
    if let Some((name, _)) = positive.iter().find(|(_, v)| *v < 1) {
        return refuse(format!("{ctx}: {name} must be at least 1"));
    }
    let (mean, std) = (
        p.image_mean.map(|x| x as f32),
        p.image_std.map(|x| x as f32),
    );
    let rescale = p.rescale_factor as f32;
    if mean.iter().chain(&[rescale]).any(|x| !x.is_finite())
        || std.iter().any(|x| !x.is_finite() || *x == 0.0)
    {
        return refuse(format!(
            "{ctx}: image_mean, image_std and rescale_factor must be finite, image_std non-zero"
        ));
    }
    Ok(Xr0Step {
        max_state_dim: p.max_state_dim as usize,
        image_factor: p.image_factor as usize,
        image_max_pixels: p.image_max_pixels as usize,
        image_mean: mean,
        image_std: std,
        rescale_factor: rescale,
        patch_size: p.patch_size as usize,
        merge_size: p.merge_size as usize,
        temporal_patch_size: p.temporal_patch_size as usize,
    })
}

fn pi05_step(params: Map<String, Value>, ctx: &str) -> EngineResult<Pi05Step> {
    let p: Pi05Params = typed(params, ctx)?;
    if p.empty_cameras < 0 {
        return refuse(format!("{ctx}: empty_cameras is negative"));
    }
    check_resolution(p.image_resolution, ctx)?;
    check_shape(p.noise_shape.as_deref(), ctx)?;
    Ok(Pi05Step {
        image_resolution: p.image_resolution,
        empty_cameras: p.empty_cameras as usize,
        noise_shape: p.noise_shape,
    })
}

fn post_step(v: &Value) -> EngineResult<PostStep> {
    let ctx = "model.postprocessors";
    let (name, params) = component(v, ctx)?;
    let ctx = format!("{ctx}: {name}");
    Ok(match name {
        "denormalize" => PostStep::Denormalize(stats_step(params, &ctx, "action")?),
        "action_chunk_trimmer" => {
            let p: TrimmerParams = typed(params, &ctx)?;
            if p.n_action_steps < 1 {
                return refuse(format!("{ctx}: n_action_steps must be at least 1"));
            }
            PostStep::ActionChunkTrimmer {
                n_action_steps: p.n_action_steps as usize,
            }
        }
        "xvla_rotation_6d_to_axis_angle" => {
            let p: XvlaRotationParams = typed(params, &ctx)?;
            if p.expected_action_dim < 0 {
                return refuse(format!("{ctx}: expected_action_dim is negative"));
            }
            PostStep::XvlaRotation6dToAxisAngle {
                expected_action_dim: p.expected_action_dim as usize,
            }
        }
        "xr0_denormalize" => {
            let p: Xr0DenormalizeParams = typed(params, &ctx)?;
            if p.action_mode != "absolute" {
                return refuse(format!(
                    "{ctx}: action_mode {:?}; only absolute is supported (delta needs the \
                     graph's state passthrough)",
                    p.action_mode
                ));
            }
            let f32s = |v: Vec<Vec<f64>>| -> Vec<Vec<f32>> {
                v.into_iter()
                    .map(|r| r.into_iter().map(|x| x as f32).collect())
                    .collect()
            };
            let (mean, std) = (f32s(p.action_mean), f32s(p.action_std));
            let shape = |v: &[Vec<f32>]| (v.len(), v.first().map_or(0, Vec::len));
            if mean.is_empty()
                || mean.iter().any(|r| r.len() != shape(&mean).1)
                || shape(&mean).1 == 0
            {
                return refuse(format!(
                    "{ctx}: action_mean is not a non-empty [steps][width] array"
                ));
            }
            if std.len() != mean.len() || std.iter().any(|r| r.len() != shape(&mean).1) {
                return refuse(format!(
                    "{ctx}: action_std {:?} differs from action_mean {:?}",
                    shape(&std),
                    shape(&mean)
                ));
            }
            if mean.iter().chain(&std).flatten().any(|x| !x.is_finite())
                || !p.action_eps.is_finite()
            {
                return refuse(format!("{ctx}: contains NaN or Inf"));
            }
            let action_dim = match p.action_dim {
                Some(d) if d < 1 => return refuse(format!("{ctx}: action_dim must be at least 1")),
                d => d.map(|d| d as usize),
            };
            PostStep::Xr0Denormalize(Xr0DenormalizeStep {
                action_mean: mean,
                action_std: std,
                action_dim,
                action_eps: p.action_eps as f32,
            })
        }
        "action_slice" => {
            let p: ActionSliceParams = typed(params, &ctx)?;
            if p.dims < 1 {
                return refuse(format!("{ctx}: dims must be at least 1"));
            }
            PostStep::ActionSlice {
                dims: p.dims as usize,
            }
        }
        "gripper_binarize" => {
            let p: GripperBinarizeParams = typed(params, &ctx)?;
            if p.index < 0 {
                return refuse(format!("{ctx}: index is negative"));
            }
            if ![p.threshold, p.above, p.below]
                .iter()
                .all(|x| x.is_finite())
            {
                return refuse(format!("{ctx}: threshold, above and below must be finite"));
            }
            PostStep::GripperBinarize {
                index: p.index as usize,
                threshold: p.threshold,
                above: p.above,
                below: p.below,
            }
        }
        other => return refuse(format!("{ctx}: {other} is not a postprocessor")),
    })
}

fn check_resolution(r: Option<[u32; 2]>, ctx: &str) -> EngineResult<()> {
    match r {
        Some([h, w]) if h == 0 || w == 0 => refuse(format!("{ctx}: image_resolution has a zero")),
        _ => Ok(()),
    }
}

fn check_shape(s: Option<&[usize]>, ctx: &str) -> EngineResult<()> {
    match s {
        Some(s) if s.is_empty() || s.contains(&0) => {
            refuse(format!("{ctx}: noise shape {s:?} is empty or has a zero"))
        }
        _ => Ok(()),
    }
}

/// Parse a stats step, keeping only the features it transforms.
///
/// Intel applies a stats entry to the input of the same key and skips keys
/// absent from the inputs; `target` is the one key this engine transforms
/// (`state` before the model, `action` after it).
fn stats_step(params: Map<String, Value>, ctx: &str, target: &str) -> EngineResult<StatsStep> {
    let p: StatsParams = typed(params, ctx)?;
    let mode = match p.mode.as_str() {
        "mean_std" => NormMode::MeanStd,
        "min_max" => NormMode::MinMax,
        "quantiles" => NormMode::Quantiles,
        "min_max_clip" => NormMode::MinMaxClip,
        "identity" => NormMode::Identity,
        m => return refuse(format!("{ctx}: unknown mode {m:?}")),
    };
    let bounds = match (mode, p.bounds.as_deref()) {
        (_, None) => Bounds::MinMax,
        (NormMode::MinMaxClip, Some([lo, hi])) if lo == "min" && hi == "max" => Bounds::MinMax,
        (NormMode::MinMaxClip, Some([lo, hi])) if lo == "q01" && hi == "q99" => Bounds::Quantiles,
        (NormMode::MinMaxClip, Some(b)) => {
            return refuse(format!(
                "{ctx}: bounds {b:?}; expected [\"min\", \"max\"] or [\"q01\", \"q99\"]"
            ))
        }
        (_, Some(_)) => return refuse(format!("{ctx}: bounds needs mode min_max_clip")),
    };
    if p.history.is_some_and(|h| h != 1) {
        return refuse(format!(
            "{ctx}: history {:?}; only 1 is supported",
            p.history
        ));
    }
    let pad_to = match p.pad_to {
        None => None,
        Some(_) if target != "state" => {
            return refuse(format!("{ctx}: pad_to applies to normalize only"))
        }
        Some(n) if n < 1 => return refuse(format!("{ctx}: pad_to must be at least 1")),
        Some(n) => Some(n as usize),
    };
    if p.stats_path.is_some() || p.artifact.is_some() {
        return refuse(format!(
            "{ctx}: stats files are not supported; the manifest must carry inline stats"
        ));
    }
    let stats = match p.stats {
        Some(s) if !s.is_empty() => s,
        _ => return refuse(format!("{ctx}: no stats")),
    };
    let keys: Vec<String> = match p.features.filter(|f| !f.is_empty()) {
        Some(f) => f,
        None => stats.keys().cloned().collect(),
    };
    let mut features = Vec::new();
    for key in keys {
        let Some(entry) = stats.get(&key) else {
            continue;
        };
        if key != target {
            // A key the engine never holds is skipped, as Intel does; one it
            // holds but does not transform here would be silently wrong.
            if key == "images" || key.starts_with("images.") || key == "task" || key == "noise" {
                return refuse(format!("{ctx}: normalizing {key:?} is not supported"));
            }
            continue;
        }
        features.push((
            key.clone(),
            feature_stats(entry, mode, bounds, &format!("{ctx}[{key}]"))?,
        ));
    }
    if mode != NormMode::Identity && features.is_empty() {
        return refuse(format!("{ctx}: no stats for {target:?}"));
    }
    if let Some(n) = pad_to {
        let width = features
            .iter()
            .filter_map(|(_, s)| s.mean.as_ref().or(s.q01.as_ref()).or(s.min.as_ref()))
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        if n < width {
            return refuse(format!(
                "{ctx}: pad_to {n} is narrower than the stats ({width})"
            ));
        }
    }
    Ok(StatsStep {
        mode,
        features,
        bounds,
        pad_to,
    })
}

fn feature_stats(
    entry: &Value,
    mode: NormMode,
    bounds: Bounds,
    ctx: &str,
) -> EngineResult<FeatureStats> {
    let Some(obj) = entry.as_object() else {
        return refuse(format!("{ctx}: not an object"));
    };
    let floats = |k: &str| -> EngineResult<Option<Vec<f64>>> {
        match obj.get(k) {
            None => Ok(None),
            Some(Value::Array(a)) => a
                .iter()
                .map(|x| x.as_f64().ok_or(()))
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
                .or_else(|_| refuse(format!("{ctx}.{k}: not a list of numbers"))),
            Some(_) => refuse(format!("{ctx}.{k}: not a list")),
        }
    };
    let mask = match obj.get("mask") {
        None => None,
        Some(Value::Array(a)) => Some(
            a.iter()
                .map(|x| match x {
                    Value::Bool(b) => Ok(*b),
                    Value::Number(n) => Ok(n.as_f64() != Some(0.0)),
                    _ => Err(()),
                })
                .collect::<Result<Vec<_>, _>>()
                .or_else(|_| refuse(format!("{ctx}.mask: not a list of booleans")))?,
        ),
        Some(_) => return refuse(format!("{ctx}.mask: not a list")),
    };
    let s = FeatureStats {
        mean: floats("mean")?,
        std: floats("std")?,
        q01: floats("q01")?,
        q99: floats("q99")?,
        min: floats("min")?,
        max: floats("max")?,
        mask,
    };
    let needed: &[(&str, &Option<Vec<f64>>)] = match mode {
        NormMode::MeanStd => &[("mean", &s.mean), ("std", &s.std)],
        NormMode::MinMax => &[("min", &s.min), ("max", &s.max)],
        NormMode::Quantiles => &[("q01", &s.q01), ("q99", &s.q99)],
        NormMode::MinMaxClip if bounds == Bounds::Quantiles => &[("q01", &s.q01), ("q99", &s.q99)],
        NormMode::MinMaxClip => &[("min", &s.min), ("max", &s.max)],
        NormMode::Identity => &[],
    };
    let mut len = None;
    for (name, v) in needed {
        let Some(v) = v else {
            return refuse(format!("{ctx}: mode needs {name}"));
        };
        if v.is_empty() {
            return refuse(format!("{ctx}.{name}: empty"));
        }
        // Intel refuses non-finite stats when normalizing; refusing at load
        // also covers denormalize, which would emit NaN actions.
        if v.iter().any(|x| !x.is_finite()) {
            return refuse(format!("{ctx}.{name}: contains NaN or Inf"));
        }
        if *len.get_or_insert(v.len()) != v.len() {
            return refuse(format!("{ctx}: stats lengths differ"));
        }
    }
    if let (Some(m), Some(n)) = (&s.mask, len) {
        if m.len() != n {
            return refuse(format!("{ctx}.mask: length {} != {n}", m.len()));
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PI05_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/intel-pi05-libero-manifest.json");
    const SMOLVLA_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/intel-smolvla-libero-manifest.json");
    const SMOLVLA_NOISE: &str =
        include_str!("../../tests/fixtures/policy/smolvla-libero-noise-manifest.json");
    const DROID: &str =
        include_str!("../../tests/fixtures/policy/pi05-droid-placeholder-manifest.json");
    const PI0_LIBERO: &str = include_str!("../../tests/fixtures/policy/pi0-libero-manifest.json");
    const XVLA_LIBERO: &str = include_str!("../../tests/fixtures/policy/xvla-libero-manifest.json");
    const VLAJEPA_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/vlajepa-libero-manifest.json");
    const EVO1_LIBERO: &str = include_str!("../../tests/fixtures/policy/evo1-libero-manifest.json");
    const XR0_LIBERO: &str = include_str!("../../tests/fixtures/policy/xr0-libero-manifest.json");
    const GR00T_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/gr00t-libero-spatial-manifest.json");

    fn parse(v: &Value) -> EngineResult<PolicyManifest> {
        PolicyManifest::parse(&v.to_string(), LoadOptions::default())
    }

    fn err(v: &Value) -> String {
        parse(v).unwrap_err().to_string()
    }

    /// A small valid pi05 manifest in the `type` form.
    fn base() -> Value {
        json!({
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "pi05"},
            "model": {
                "runner": {"type": "single_pass"},
                "artifacts": {"openvino": "pi05.xml"},
                "preprocessors": [
                    {"type": "normalize", "mode": "mean_std",
                     "stats": {"state": {"mean": [0.5, 0.0], "std": [2.0, 1.0]}}},
                    {"type": "pi05", "image_resolution": [224, 224], "empty_cameras": 1},
                    {"type": "ov_tokenizer", "artifact": "tokenizer.xml"}
                ],
                "postprocessors": [
                    {"type": "denormalize", "mode": "mean_std",
                     "stats": {"action": {"mean": [0.5], "std": [2.0]}}},
                    {"type": "action_chunk_trimmer", "n_action_steps": 10}
                ]
            }
        })
    }

    #[test]
    fn intel_published_pi05_manifest_parses() {
        let m = PolicyManifest::parse(PI05_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::Pi05);
        assert_eq!(m.model_artifact, "pi05.xml");
        assert_eq!(m.tokenizer_artifact(), "tokenizer.xml");
        assert_eq!(m.n_action_steps(), Some(10));
        let ImageStep::Pi05(p) = m.image_step() else {
            panic!("pi05 image step")
        };
        assert_eq!((p.image_resolution, p.empty_cameras), (Some([224, 224]), 1));
        let n = m.normalize().next().unwrap();
        assert_eq!(n.mode, NormMode::MeanStd);
        assert_eq!(
            n.stats_for("state").unwrap().mean.as_ref().unwrap().len(),
            8
        );
        assert_eq!(m.input_features.len(), 4);
        assert_eq!(m.input_features[0].name, "images.image");
        assert_eq!(m.input_features[0].ftype, FeatureType::Visual);
        assert_eq!(m.output_features[0].shape, vec![50, 7]);
    }

    #[test]
    fn intel_published_smolvla_manifest_parses() {
        let m = PolicyManifest::parse(SMOLVLA_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::SmolVla);
        assert_eq!(m.model_artifact, "smolvla.xml");
        assert_eq!(m.n_action_steps(), Some(1));
        assert!(matches!(m.preprocessors[1], PreStep::NewLine));
        let ImageStep::SmolVlaResize(s) = m.image_step() else {
            panic!("smolvla image step")
        };
        assert_eq!(s.image_resolution, Some([512, 512]));
        assert!(s.image_key_reorder_map.is_empty());
        assert!(m.input_features.is_empty());
    }

    #[test]
    fn new_style_smolvla_manifest_parses() {
        let m = PolicyManifest::parse(SMOLVLA_NOISE, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::SmolVla);
        assert!(m.preprocessors.iter().any(|s| *s
            == PreStep::Noise {
                shape: vec![50, 32],
                port: None
            }));
        let ImageStep::SmolVlaResize(s) = m.image_step() else {
            panic!("smolvla image step")
        };
        assert!(s.image_key_reorder_map.is_empty());
        assert_eq!(s.num_cameras, 0);
        let noise = m.input_features.iter().find(|f| f.name == "noise").unwrap();
        assert_eq!(noise.ftype, FeatureType::Common);
    }

    #[test]
    fn pi0_libero_manifest_parses() {
        let m = PolicyManifest::parse(PI0_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::Pi0);
        assert_eq!(m.model_artifact, "pi0.xml");
        assert_eq!(m.n_action_steps(), None);
        let ImageStep::Pi05(p) = m.image_step() else {
            panic!("pi0 uses the pi05 image handling")
        };
        assert_eq!(
            (p.image_resolution, p.empty_cameras, p.noise_shape.as_ref()),
            (Some([224, 224]), 1, None)
        );
        assert!(matches!(m.preprocessors[1], PreStep::Pi0(_)));
        assert!(m.preprocessors.contains(&PreStep::NewLine));
        assert_eq!(m.output_features[0].shape, vec![50, 7]);
    }

    #[test]
    fn pi0_step_takes_the_pi05_params_and_counts_as_the_image_step() {
        let mut v = base();
        v["policy"] = json!({"name": "pi0"});
        v["model"]["preprocessors"][1] = json!({"type": "pi0", "image_resolution": [224, 224],
                                                "empty_cameras": 2, "noise_shape": [50, 32]});
        let m = parse(&v).unwrap();
        assert_eq!(
            m.preprocessors[1],
            PreStep::Pi0(Pi05Step {
                image_resolution: Some([224, 224]),
                empty_cameras: 2,
                noise_shape: Some(vec![50, 32]),
            })
        );
        v["model"]["preprocessors"][1]["empty_cameras"] = json!(-1);
        assert!(err(&v).contains("pi0: empty_cameras is negative"));
        v["model"]["preprocessors"][1] = json!({"type": "pi0", "state_bins": true});
        assert!(err(&v).contains("unknown field `state_bins`"));
        // A pi0 next to a pi05 is two image steps.
        let mut v = base();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .insert(2, json!({"type": "pi0"}));
        assert!(err(&v).contains("2 image steps"));
        // Ours, not Intel's: no class_path maps to it.
        let mut v = base();
        v["model"]["preprocessors"][1] =
            json!({"class_path": "physicalai.inference.preprocessors.Pi0Preprocessor"});
        assert!(err(&v).contains("unsupported component class_path"));
        v["model"]["preprocessors"][1] = json!({"class_path": "pi0"});
        assert!(err(&v).contains("unsupported component class_path"));
    }

    #[test]
    fn xvla_libero_manifest_parses() {
        let m = PolicyManifest::parse(XVLA_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::Xvla);
        assert_eq!(m.model_artifact, "xvla.xml");
        assert_eq!(m.flipped_cameras(), ["image2"]);
        assert!(matches!(m.image_step(), ImageStep::SmolVlaResize(_)));
        assert_eq!(
            m.postprocessors,
            vec![PostStep::XvlaRotation6dToAxisAngle {
                expected_action_dim: 10
            }]
        );
        assert_eq!(m.normalize().count(), 0);
        assert_eq!(m.output_features[0].shape, vec![30, 7]);
    }

    #[test]
    fn vlajepa_libero_manifest_parses() {
        let m = PolicyManifest::parse(VLAJEPA_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::VlaJepa);
        assert_eq!(m.model_artifact, "vlajepa.xml");
        assert_eq!(m.tokenizer_artifact(), "tokenizer.xml");
        let PreStep::VlaJepaPrompt { template } = &m.preprocessors[0] else {
            panic!("the prompt step comes first")
        };
        assert_eq!(template.matches("{instruction}").count(), 1);
        assert_eq!(template.matches("<|embodied_action|>").count(), 32);
        let ImageStep::VlaJepa(j) = m.image_step() else {
            panic!("vla_jepa_inputs is the image step")
        };
        assert_eq!(
            *j,
            VlaJepaInputsStep {
                image_keys: vec!["images.image".into(), "images.image2".into()],
                embodied_token_id: 151697,
                num_embodied_tokens: 32,
                image_resolution: [224, 224],
                vision_resolution: [256, 256],
                image_mean: [0.5; 3],
                image_std: [0.5; 3],
                patch_size: 16,
                merge_size: 2,
                temporal_patch_size: 2,
            }
        );
        assert!(m.preprocessors.contains(&PreStep::Noise {
            shape: vec![7, 7],
            port: None
        }));
        assert!(m.postprocessors.is_empty());
        assert_eq!(m.normalize().count(), 0, "stats are graph constants");
    }

    #[test]
    fn evo1_libero_manifest_parses() {
        let m = PolicyManifest::parse(EVO1_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::Evo1);
        assert_eq!(m.model_artifact, "evo1.xml");
        assert_eq!(m.flipped_cameras(), ["image", "image2"]);
        let ImageStep::SmolVlaResize(s) = m.image_step() else {
            panic!("smolvla image step")
        };
        assert_eq!(s.image_resolution, Some([448, 448]));
        assert!(s.stretch);
        assert!(!m.preprocessors.contains(&PreStep::NewLine));
        assert_eq!(m.n_action_steps(), Some(14));
        assert_eq!(
            m.postprocessors.last(),
            Some(&PostStep::GripperBinarize {
                index: 6,
                threshold: 0.0,
                above: 1.0,
                below: -1.0
            })
        );
    }

    #[test]
    fn gripper_binarize_and_the_resize_param_are_validated() {
        let evo1 = || -> Value { serde_json::from_str(EVO1_LIBERO).unwrap() };
        let mut v = evo1();
        v["model"]["postprocessors"][2]["index"] = json!(-1);
        assert!(err(&v).contains("index is negative"));
        let mut v = evo1();
        v["model"]["postprocessors"][2]["axis"] = json!(0);
        assert!(err(&v).contains("unknown field `axis`"));
        let mut v = evo1();
        v["model"]["postprocessors"][2]
            .as_object_mut()
            .unwrap()
            .remove("threshold");
        assert!(err(&v).contains("missing field `threshold`"));
        // Ours: type form only.
        let mut v = evo1();
        v["model"]["postprocessors"][2] = json!({"class_path": "gripper_binarize"});
        assert!(err(&v).contains("unsupported component class_path"));
        let mut v = evo1();
        v["model"]["preprocessors"][2]["resize"] = json!("center");
        assert!(err(&v).contains("resize \"center\"; expected top_left or stretch"));
        v["model"]["preprocessors"][2]["resize"] = json!("top_left");
        let m = parse(&v).unwrap();
        let ImageStep::SmolVlaResize(s) = m.image_step() else {
            panic!("smolvla image step")
        };
        assert!(!s.stretch);
        assert_eq!(
            PostStep::GripperBinarize {
                index: 7,
                threshold: 0.0,
                above: 1.0,
                below: -1.0
            }
            .out_dim(7)
            .unwrap_err()
            .to_string(),
            "invalid configuration: policy manifest: gripper_binarize index 7 is outside actions 7 wide"
        );
    }

    #[test]
    fn xr0_libero_manifest_parses() {
        let m = PolicyManifest::parse(XR0_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::Xr0);
        let ImageStep::Xr0(x) = m.image_step() else {
            panic!("xr0 is the image step")
        };
        assert_eq!(
            (
                x.max_state_dim,
                x.patch_size,
                x.merge_size,
                x.temporal_patch_size
            ),
            (32, 16, 2, 2)
        );
        assert_eq!(x.rescale_factor, (1.0f64 / 255.0) as f32);
        let PostStep::Xr0Denormalize(d) = &m.postprocessors[0] else {
            panic!("xr0_denormalize")
        };
        assert_eq!(
            (d.action_mean.len(), d.action_mean[0].len(), d.action_dim),
            (30, 32, Some(7))
        );
        assert_eq!(d.action_eps, 1e-6);
        assert_eq!(d.action_mean[29][0], 0.0628);
        // Noise comes after xr0, as Intel's xr0 returns a new dict.
        assert!(matches!(m.preprocessors[1], PreStep::Noise { .. }));
    }

    #[test]
    fn xr0_steps_are_validated() {
        let xr0 = || -> Value { serde_json::from_str(XR0_LIBERO).unwrap() };
        let e = |v: &Value| err(v);
        let mut v = xr0();
        v["model"]["postprocessors"][0]["action_mode"] = json!("delta");
        assert!(e(&v).contains("only absolute is supported"));
        let mut v = xr0();
        v["model"]["postprocessors"][0]["clip"] = json!(true);
        assert!(e(&v).contains("unknown field `clip`"));
        let mut v = xr0();
        v["model"]["postprocessors"][0]["action_std"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(e(&v).contains("differs from action_mean"));
        let mut v = xr0();
        v["model"]["preprocessors"][0]["normalize_state"] = json!(true);
        assert!(e(&v).contains("normalize_state is not supported"));
        let mut v = xr0();
        v["model"]["preprocessors"][0]["image_key_view_map"] = json!({"images.base": "ego"});
        assert!(e(&v).contains("image_key_view_map is not supported"));
        let mut v = xr0();
        v["model"]["preprocessors"][0]["resample"] = json!("cubic");
        assert!(e(&v).contains("unknown field `resample`"));
        // Frames Intel would resize, and cameras of different sizes.
        let mut v = xr0();
        v["model"]["input_features"][1]["init_args"]["shape"] = json!([3, 480, 640]);
        v["model"]["input_features"][2]["init_args"]["shape"] = json!([3, 480, 640]);
        assert!(e(&v).contains("would be resized to [256, 320]"));
        let mut v = xr0();
        v["model"]["input_features"][2]["init_args"]["shape"] = json!([3, 224, 224]);
        assert!(e(&v).contains("cameras differ in size"));
        let mut v = xr0();
        v["model"]["input_features"][0]["init_args"]["shape"] = json!([40]);
        assert!(e(&v).contains("max_state_dim 32"));
        // The graph's state goes in raw: no stats steps.
        let mut v = xr0();
        v["model"]["preprocessors"].as_array_mut().unwrap().insert(
            0,
            json!({"type": "normalize", "stats": {"state": {"mean": [0.5], "std": [2.0]}}}),
        );
        assert!(e(&v).contains("found normalize"));
        let mut v = xr0();
        v["policy"] = json!({"name": "smolvla"});
        assert!(e(&v).contains("need policy.name \"xr0\""));
        // Identity stats are a placeholder.
        let mut v = xr0();
        let p = &mut v["model"]["postprocessors"][0];
        p["action_mean"] = json!(vec![vec![0.0; 32]; 30]);
        p["action_std"] = json!(vec![vec![1.0; 32]; 30]);
        assert!(e(&v).contains("xr0_denormalize stats are identity placeholders"));
        // Intel's class names map to the steps.
        let mut v = xr0();
        let mut step = v["model"]["preprocessors"][0].clone();
        step.as_object_mut().unwrap().remove("type");
        v["model"]["preprocessors"][0] = json!({
            "class_path": "physicalai.inference.preprocessors.XR0Preprocessor",
            "init_args": step
        });
        parse(&v).unwrap();
    }

    #[test]
    fn gr00t_libero_manifest_parses() {
        let m = PolicyManifest::parse(GR00T_LIBERO, LoadOptions::default()).unwrap();
        assert_eq!(m.policy, PolicyKind::Gr00tN1d7);
        assert_eq!(m.model_artifact, "gr00t.xml");
        let ImageStep::Gr00t(g) = m.image_step() else {
            panic!("gr00t_image is the image step")
        };
        assert_eq!(
            (g.shortest_edge, g.crop_fraction, g.letterbox),
            (256, 0.95, true)
        );
        let (_, p, t, q) = m.gr00t_steps().unwrap();
        assert_eq!(
            (p.patch_size, p.temporal_patch_size, p.merge_size),
            (16, 2, 2)
        );
        assert_eq!(p.grid_thw, vec![[1, 16, 16]; 2]);
        assert!(t.formalize_language);
        assert_eq!(
            (q.image_token_id, q.tokens_per_image, q.seq_len, q.pad_id),
            (151655, 64, 192, 0)
        );
        let n = m.normalize().next().unwrap();
        assert_eq!(
            (n.mode, n.bounds, n.pad_to),
            (NormMode::MinMaxClip, Bounds::Quantiles, Some(132))
        );
        let d = m.denormalize().next().unwrap();
        assert_eq!(
            (d.mode, d.bounds, d.pad_to),
            (NormMode::MinMaxClip, Bounds::Quantiles, None)
        );
        assert_eq!(
            m.constants().collect::<Vec<_>>(),
            [("embodiment_id", &[2i64][..])]
        );
        assert_eq!(m.noise_port(), "actions");
        assert_eq!(m.flipped_cameras(), ["image", "wrist_image"]);
        assert_eq!(
            m.postprocessors[..2],
            [
                PostStep::ActionChunkTrimmer { n_action_steps: 16 },
                PostStep::ActionSlice { dims: 7 }
            ]
        );
        assert_eq!(
            m.attribution.as_deref(),
            Some("Built on NVIDIA Cosmos. Licensed by NVIDIA Corporation under the NVIDIA Open Model License.")
        );
        let geo = crate::policy::gr00t::Gr00tGeometry::new(&m).unwrap();
        assert_eq!(
            (
                geo.side,
                geo.crop,
                geo.cameras,
                geo.state_len,
                geo.state_width
            ),
            (256, 243, 2, 8, 132)
        );
    }

    #[test]
    fn gr00t_steps_are_validated() {
        let gr00t = || -> Value { serde_json::from_str(GR00T_LIBERO).unwrap() };
        let at = |name: &str| {
            gr00t()["model"]["preprocessors"]
                .as_array()
                .unwrap()
                .iter()
                .position(|s| s["type"] == name)
                .unwrap()
        };
        let set = |step: &str, key: &str, value: Value| {
            let mut v = gr00t();
            v["model"]["preprocessors"][at(step)][key] = value;
            err(&v)
        };
        assert!(set("gr00t_image", "interpolation", json!("linear")).contains("only area"));
        assert!(set("gr00t_image", "crop_fraction", json!(1.5))
            .contains("crop_fraction must be in (0, 1]"));
        assert!(set("gr00t_image", "resample", json!(1)).contains("unknown field `resample`"));
        assert!(set("gr00t_image", "shortest_edge", json!(224))
            .contains("square frames of shortest_edge"));
        assert!(set("qwen_vl_patchify", "grid_thw", json!([[1, 16, 16]]))
            .contains("for each of 2 cameras"));
        assert!(set("qwen_vl_patchify", "image_std", json!([0.5, 0.0, 0.5]))
            .contains("image_std non-zero"));
        assert!(set("gr00t_prompt", "template", json!("{task}"))
            .contains("needs {images} exactly once"));
        assert!(set("qwen_vl_inputs", "padding_side", json!("left")).contains("only right"));
        assert!(set("qwen_vl_inputs", "rope", json!("default")).contains("only qwen3_vl_mrope"));
        assert!(set("qwen_vl_inputs", "tokens_per_image", json!(32))
            .contains("not the 64 merged patches"));
        assert!(
            set("qwen_vl_inputs", "seq_len", json!(128)).contains("leave no room in seq_len 128")
        );
        assert!(set("constant", "dtype", json!("int32")).contains("only int64"));
        assert!(
            set("normalize", "bounds", json!(["q05", "q95"])).contains("bounds [\"q05\", \"q95\"]")
        );
        assert!(set("normalize", "history", json!(2)).contains("only 1 is supported"));
        assert!(set("normalize", "pad_to", json!(4)).contains("narrower than the stats"));
        assert!(
            set("normalize", "mode", json!("quantiles")).contains("bounds needs mode min_max_clip")
        );
        // A non-square camera, a second normalize, a foreign step, a wide slice.
        let mut v = gr00t();
        v["model"]["input_features"][1]["init_args"]["shape"] = json!([3, 256, 320]);
        assert!(err(&v).contains("square frames"));
        let mut v = gr00t();
        let n = v["model"]["preprocessors"][at("normalize")].clone();
        v["model"]["preprocessors"].as_array_mut().unwrap().push(n);
        assert!(err(&v).contains("at most one normalize"));
        let mut v = gr00t();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "new_line"}));
        assert!(err(&v).contains("found new_line"));
        let mut v = gr00t();
        v["model"]["postprocessors"][1]["dims"] = json!(200);
        let m = parse(&v).unwrap();
        assert!(m.postprocessors[1]
            .out_dim(132)
            .unwrap_err()
            .to_string()
            .contains("keeps 200 values"));
        let mut v = gr00t();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .retain(|s| s["type"] != "qwen_vl_inputs");
        assert!(err(&v).contains("0 qwen_vl_inputs steps"));
        // GR00T's steps are its own.
        let mut v = gr00t();
        v["policy"] = json!({"name": "smolvla"});
        assert!(
            err(&v).contains("gr00t_image needs policy.name \"gr00t_n1d7\""),
            "{}",
            err(&v)
        );
        let mut v: Value = serde_json::from_str(SMOLVLA_NOISE).unwrap();
        v["model"]["postprocessors"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "action_slice", "dims": 7}));
        assert!(err(&v).contains("action_slice needs policy.name"));
        let mut v: Value = serde_json::from_str(SMOLVLA_NOISE).unwrap();
        for s in v["model"]["preprocessors"].as_array_mut().unwrap() {
            if s["type"] == "noise" {
                s["port"] = json!("actions");
            }
        }
        assert!(err(&v).contains("a noise port need policy.name"));
        // Identity bounds are a placeholder.
        let mut v = gr00t();
        let st = &mut v["model"]["postprocessors"][2]["stats"]["action"];
        st["q01"] = json!(vec![-1.0; 7]);
        st["q99"] = json!(vec![1.0; 7]);
        assert!(err(&v).contains("denormalize stats are identity placeholders"));
    }

    #[test]
    fn vlajepa_steps_are_validated() {
        let jepa = || -> Value { serde_json::from_str(VLAJEPA_LIBERO).unwrap() };
        let e = |v: &Value| err(v);
        let mut v = jepa();
        v["model"]["preprocessors"][0]["template"] = json!("no placeholder");
        assert!(e(&v).contains("template has no {instruction}"));
        let mut v = jepa();
        v["model"]["preprocessors"][0]["prefix"] = json!("x");
        assert!(e(&v).contains("unknown field `prefix`"));
        for (key, bad, want) in [
            (
                "vision_resolution",
                json!([256, 250]),
                "not a multiple of patch_size * merge_size (32)",
            ),
            ("vision_resolution", json!([0, 256]), "not a multiple"),
            (
                "image_resolution",
                json!([224, 0]),
                "image_resolution has a zero",
            ),
            ("num_embodied_tokens", json!(0), "at least 1"),
            ("embodied_token_id", json!(-1), "negative"),
            ("patch_size", json!(0), "at least 1"),
            ("image_std", json!([0.5, 0.0, 0.5]), "image_std"),
            ("image_keys", json!([]), "image_keys is empty"),
        ] {
            let mut v = jepa();
            v["model"]["preprocessors"][2][key] = bad;
            assert!(e(&v).contains(want), "{key}: {}", e(&v));
        }
        let mut v = jepa();
        v["model"]["preprocessors"][2]["resample"] = json!("bilinear");
        assert!(e(&v).contains("unknown field `resample`"));
        let mut v = jepa();
        v["model"]["preprocessors"][2]
            .as_object_mut()
            .unwrap()
            .remove("patch_size");
        assert!(e(&v).contains("missing field `patch_size`"));
        // Stats and action transforms belong to the graph.
        let mut v = jepa();
        v["model"]["preprocessors"].as_array_mut().unwrap().insert(
            0,
            json!({"type": "normalize", "stats": {"state": {"mean": [0.5], "std": [2.0]}}}),
        );
        assert!(e(&v).contains("found normalize"));
        let mut v = jepa();
        v["model"]["postprocessors"] = json!([{"type": "denormalize",
            "stats": {"action": {"mean": [0.5], "std": [2.0]}}}]);
        assert!(e(&v).contains("final actions; found postprocessor denormalize"));
        let mut v = jepa();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .remove(0);
        assert!(e(&v).contains("0 vla_jepa_prompt steps"));
        // The steps are vlajepa's alone, and ours: type form only.
        let mut v = jepa();
        v["policy"] = json!({"name": "smolvla"});
        assert!(e(&v).contains("need policy.name \"vlajepa\""));
        let mut v = jepa();
        v["model"]["preprocessors"][2] =
            json!({"class_path": "physicalai.inference.preprocessors.VLAJEPAPreprocessor"});
        assert!(e(&v).contains("unsupported component class_path"));
    }

    #[test]
    fn image_flip_and_rotation_steps_are_validated() {
        let xvla = |pre: Option<Value>, post: Value| {
            let mut v = base();
            v["policy"] = json!({"name": "xvla"});
            if let Some(p) = pre {
                v["model"]["preprocessors"]
                    .as_array_mut()
                    .unwrap()
                    .insert(0, p);
            }
            v["model"]["postprocessors"]
                .as_array_mut()
                .unwrap()
                .push(post);
            parse(&v)
        };
        let rot = json!({"type": "xvla_rotation_6d_to_axis_angle"});
        let m = xvla(
            Some(json!({"type": "image_flip", "cameras": ["image"]})),
            rot.clone(),
        )
        .unwrap();
        assert_eq!(m.flipped_cameras(), ["image"]);
        // LeRobot's default.
        assert!(m
            .postprocessors
            .contains(&PostStep::XvlaRotation6dToAxisAngle {
                expected_action_dim: 10
            }));
        let e = |pre, post| xvla(pre, post).unwrap_err().to_string();
        assert!(e(
            Some(json!({"type": "image_flip", "cameras": []})),
            rot.clone()
        )
        .contains("cameras is empty"));
        assert!(e(
            Some(json!({"type": "image_flip", "cameras": ["a", "a"]})),
            rot.clone()
        )
        .contains("\"a\" listed twice"));
        assert!(
            e(Some(json!({"type": "image_flip"})), rot.clone()).contains("missing field `cameras`")
        );
        assert!(e(
            Some(json!({"type": "image_flip", "cameras": ["a"], "axis": 0})),
            rot.clone()
        )
        .contains("unknown field `axis`"));
        assert!(e(
            None,
            json!({"type": "xvla_rotation_6d_to_axis_angle", "expected_action_dim": -1})
        )
        .contains("negative"));
        assert!(e(
            None,
            json!({"type": "xvla_rotation_6d_to_axis_angle", "order": "xyz"})
        )
        .contains("unknown field `order`"));
        // Ours, not Intel's: type form only.
        assert!(e(
            None,
            json!({"class_path": "xvla_rotation_6d_to_axis_angle"})
        )
        .contains("unsupported component class_path"));
        let mut v = base();
        for _ in 0..2 {
            v["model"]["preprocessors"]
                .as_array_mut()
                .unwrap()
                .insert(0, json!({"type": "image_flip", "cameras": ["image"]}));
        }
        assert!(err(&v).contains("more than one image_flip"));
        let mut v = base();
        for _ in 0..2 {
            v["model"]["postprocessors"]
                .as_array_mut()
                .unwrap()
                .push(rot.clone());
        }
        assert!(err(&v).contains("more than one xvla_rotation"));
    }

    #[test]
    fn rotation_width_is_seven_from_at_least_ten() {
        let step = |n| PostStep::XvlaRotation6dToAxisAngle {
            expected_action_dim: n,
        };
        assert_eq!(step(10).out_dim(20).unwrap(), XVLA_ACTION_DIM);
        assert_eq!(step(10).out_dim(10).unwrap(), 7);
        // LeRobot slices 0..10 whatever the expected dim says.
        assert!(step(4)
            .out_dim(9)
            .unwrap_err()
            .to_string()
            .contains("at least 10 wide"));
        assert!(step(12)
            .out_dim(11)
            .unwrap_err()
            .to_string()
            .contains("at least 12 wide"));
        assert_eq!(
            PostStep::ActionChunkTrimmer { n_action_steps: 3 }
                .out_dim(20)
                .unwrap(),
            20
        );
    }

    #[test]
    fn droid_placeholder_stats_refused_unless_overridden() {
        let e = PolicyManifest::parse(DROID, LoadOptions::default())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("normalize stats are identity placeholders"),
            "{e}"
        );
        assert!(e.contains("--allow-identity-stats"), "{e}");
        let m = PolicyManifest::parse(
            DROID,
            LoadOptions {
                allow_identity_stats: true,
            },
        )
        .unwrap();
        assert_eq!(m.normalize().next().unwrap().mode, NormMode::Quantiles);
        assert_eq!(m.output_features[0].shape, vec![15, 32]);
    }

    #[test]
    fn placeholder_check_follows_the_steps_mode() {
        let set = |v: &mut Value, idx: usize, mode: &str, stats: Value| {
            let s = &mut v["model"]["preprocessors"][idx];
            s["mode"] = json!(mode);
            s["stats"] = json!({ "state": stats });
        };
        for (mode, stats) in [
            ("mean_std", json!({"mean": [0.0, 0.0], "std": [1.0, 1.0]})),
            ("quantiles", json!({"q01": [-1.0, -1.0], "q99": [1.0, 1.0]})),
            ("min_max", json!({"min": [-1.0, -1.0], "max": [1.0, 1.0]})),
        ] {
            let mut v = base();
            set(&mut v, 0, mode, stats);
            assert!(err(&v).contains("identity placeholders"), "{mode}");
        }
        // One real dim is enough to be real stats.
        let mut v = base();
        set(
            &mut v,
            0,
            "quantiles",
            json!({"q01": [-1.0, -0.5], "q99": [1.0, 1.0]}),
        );
        parse(&v).unwrap();
        // Identity mean/std under a quantiles step is not a placeholder for that mode.
        let mut v = base();
        set(
            &mut v,
            0,
            "quantiles",
            json!({"mean": [0.0, 0.0], "std": [1.0, 1.0], "q01": [-2.0, -2.0], "q99": [2.0, 2.0]}),
        );
        parse(&v).unwrap();
        // The identity mode declares no normalization; it is not a placeholder.
        let mut v = base();
        set(&mut v, 0, "identity", json!({}));
        parse(&v).unwrap();
        // A placeholder denormalize alone is refused too.
        let mut v = base();
        v["model"]["postprocessors"][0]["stats"] = json!({"action": {"mean": [0.0], "std": [1.0]}});
        assert!(err(&v).contains("denormalize stats are identity placeholders"));
    }

    #[test]
    fn class_path_form_parses_and_wins_over_type() {
        let mut v = base();
        v["model"]["runner"] =
            json!({"class_path": "physicalai.inference.runners.single_pass.SinglePass"});
        v["model"]["preprocessors"][1] = json!({
            "type": "smolvla_resize",
            "image_resolution": [1, 1],
            "class_path": "physicalai.inference.preprocessors.Pi05Preprocessor",
            "init_args": {"image_resolution": [256, 128], "empty_cameras": 2}
        });
        v["model"]["postprocessors"][1] = json!({
            "class_path": "action_chunk_trimmer",
            "init_args": {"n_action_steps": 5}
        });
        let m = parse(&v).unwrap();
        let ImageStep::Pi05(p) = m.image_step() else {
            panic!("class_path must win")
        };
        assert_eq!((p.image_resolution, p.empty_cameras), (Some([256, 128]), 2));
        assert_eq!(m.n_action_steps(), Some(5));
    }

    #[test]
    fn unknown_components_and_params_are_refused() {
        let mut v = base();
        v["model"]["preprocessors"][1] = json!({"type": "resize"});
        assert!(err(&v).contains("unsupported component type \"resize\""));
        let mut v = base();
        v["model"]["preprocessors"][1] =
            json!({"class_path": "physicalai.inference.preprocessors.MolmoAct2Preprocessor"});
        assert!(err(&v).contains("unsupported component class_path"));
        let mut v = base();
        v["model"]["preprocessors"][1] = json!({"class_path": "evil.module.Pi05Preprocessor"});
        assert!(err(&v).contains("unsupported component class_path"));
        let mut v = base();
        v["model"]["preprocessors"][1]["crop"] = json!(true);
        assert!(err(&v).contains("unknown field `crop`"));
        let mut v = base();
        v["model"]["preprocessors"][1] = json!({"image_resolution": [224, 224]});
        assert!(err(&v).contains("neither type nor class_path"));
        let mut v = base();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "denormalize"}));
        assert!(err(&v).contains("denormalize is not a preprocessor"));
    }

    #[test]
    fn format_version_runner_and_policy_are_checked() {
        let mut v = base();
        v["format"] = json!("onnx_package");
        assert!(err(&v).contains("format"));
        let mut v = base();
        v.as_object_mut().unwrap().remove("format");
        assert!(err(&v).contains("format None"));
        for bad in ["2.0", "10.0", "1.x", "", "1."] {
            let mut v = base();
            v["version"] = json!(bad);
            assert!(err(&v).contains("only 1.x"), "{bad}");
        }
        for good in ["1", "1.0", "1.7.2"] {
            let mut v = base();
            v["version"] = json!(good);
            parse(&v).unwrap();
        }
        let mut v = base();
        v["version"] = json!(1.0);
        assert!(err(&v).contains("only 1.x"));
        let mut v = base();
        v["model"]["runner"] = json!({"type": "action_chunk_trimmer"});
        assert!(err(&v).contains("only SinglePass"));
        // No runner means SinglePass, as in Intel's runner factory.
        let mut v = base();
        v["model"].as_object_mut().unwrap().remove("runner");
        parse(&v).unwrap();
        let mut v = base();
        v["policy"] = json!({"name": "pi0_fast"});
        assert!(err(&v).contains("policy.name Some(\"pi0_fast\")"));
        assert!(err(&v).contains("xvla, evo1, vlajepa, xr0"));
        let mut v = base();
        v["policy"] =
            json!({"source": {"class_path": "physicalai.policies.smolvla.policy.SmolVLA"}});
        assert!(err(&v).contains("policy.name None"));
    }

    #[test]
    fn artifact_paths_are_confined() {
        for (path, why) in [
            ("../pi05.xml", "'..'"),
            ("sub/../../x.xml", "'..'"),
            ("sub\\..\\..\\x.xml", "'..'"),
            ("/etc/passwd", "absolute"),
            ("\\Windows\\x.xml", "absolute"),
            ("C:\\models\\pi05.xml", "drive letter"),
            ("C:pi05.xml", "drive letter"),
            ("\\\\server\\share\\pi05.xml", "UNC"),
            ("//server/share/pi05.xml", "UNC"),
            ("", "empty"),
        ] {
            let e = confine_artifact(path).unwrap_err().to_string();
            assert!(e.contains(why), "{path}: {e}");
        }
        assert_eq!(confine_artifact("pi05.xml").unwrap(), "pi05.xml");
        assert_eq!(confine_artifact("ir/pi05.xml").unwrap(), "ir/pi05.xml");
        assert_eq!(confine_artifact("a..b.xml").unwrap(), "a..b.xml");

        let mut v = base();
        v["model"]["artifacts"]["openvino"] = json!("../pi05.xml");
        assert!(err(&v).contains("escapes"));
        let mut v = base();
        v["model"]["preprocessors"][2]["artifact"] = json!("/tmp/tokenizer.xml");
        assert!(err(&v).contains("absolute"));
        let mut v = base();
        v["model"]["artifacts"] = json!({"onnx": "pi05.onnx"});
        assert!(err(&v).contains("openvino is missing"));
    }

    #[test]
    fn stats_steps_are_validated() {
        let mut v = base();
        v["model"]["preprocessors"][0]["mode"] = json!("zscore");
        assert!(err(&v).contains("unknown mode"));
        let mut v = base();
        v["model"]["preprocessors"][0]["stats"] = json!({"state": {"mean": [0.1, 0.2]}});
        assert!(err(&v).contains("mode needs std"));
        let mut v = base();
        v["model"]["preprocessors"][0]["stats"] =
            json!({"state": {"mean": [0.1, 0.2], "std": [1.0]}});
        assert!(err(&v).contains("lengths differ"));
        let mut v = base();
        v["model"]["preprocessors"][0]["artifact"] = json!("stats.safetensors");
        assert!(err(&v).contains("stats files are not supported"));
        // Stats keyed by a name the engine never holds would normalize nothing.
        let mut v = base();
        v["model"]["preprocessors"][0]["stats"] =
            json!({"observation.state": {"mean": [0.1], "std": [1.0]}});
        assert!(err(&v).contains("no stats for \"state\""));
        let mut v = base();
        v["model"]["preprocessors"][0]["stats"]["images.top"] =
            json!({"mean": [0.1], "std": [1.0]});
        assert!(err(&v).contains("normalizing \"images.top\""));
        // `features` narrows the keys; an unlisted placeholder key is not applied.
        let mut v = base();
        v["model"]["preprocessors"][0]["features"] = json!(["state"]);
        v["model"]["preprocessors"][0]["stats"]["images.top"] =
            json!({"mean": [0.1], "std": [1.0]});
        parse(&v).unwrap();
        let mut v = base();
        v["model"]["preprocessors"][0]["stats"]["state"]["mask"] = json!([true, false]);
        let m = parse(&v).unwrap();
        let mask = &m
            .normalize()
            .next()
            .unwrap()
            .stats_for("state")
            .unwrap()
            .mask;
        assert_eq!(mask.as_deref(), Some(&[true, false][..]));
    }

    #[test]
    fn pipeline_shape_is_checked() {
        let mut v = base();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .remove(2);
        assert!(err(&v).contains("0 ov_tokenizer steps"));
        let mut v = base();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .remove(1);
        assert!(err(&v).contains("0 image steps"));
        let mut v = base();
        v["model"]["preprocessors"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "smolvla_resize"}));
        assert!(err(&v).contains("2 image steps"));
        let mut v = base();
        v["model"]["postprocessors"][1]["n_action_steps"] = json!(0);
        assert!(err(&v).contains("at least 1"));
        let mut v = base();
        v["model"]["preprocessors"][1]["empty_cameras"] = json!(-1);
        assert!(err(&v).contains("negative"));
    }
}
