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
    VlaJepa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormMode {
    MeanStd,
    MinMax,
    Quantiles,
    Identity,
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
        Self::parse(&text, opts)
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

        let m = Self {
            policy,
            version,
            model_artifact,
            preprocessors,
            postprocessors,
            input_features,
            output_features,
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
            )
        });
        if images != 1 {
            return refuse(format!(
                "{images} image steps (pi05 / pi0 / smolvla_resize / vla_jepa_inputs); \
                 expected exactly one"
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
        self.check_vla_jepa()
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
    }
}

fn post_name(s: &PostStep) -> &'static str {
    match s {
        PostStep::Denormalize(_) => "denormalize",
        PostStep::ActionChunkTrimmer { .. } => "action_chunk_trimmer",
        PostStep::XvlaRotation6dToAxisAngle { .. } => "xvla_rotation_6d_to_axis_angle",
    }
}

fn is_placeholder(step: &StatsStep) -> bool {
    let all =
        |v: &Option<Vec<f64>>, want: f64| v.as_ref().is_some_and(|v| v.iter().all(|&x| x == want));
    let identity = |s: &FeatureStats| match step.mode {
        NormMode::MeanStd => all(&s.mean, 0.0) && all(&s.std, 1.0),
        NormMode::Quantiles => all(&s.q01, -1.0) && all(&s.q99, 1.0),
        NormMode::MinMax => all(&s.min, -1.0) && all(&s.max, 1.0),
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
        Some("vlajepa") => Ok(PolicyKind::VlaJepa),
        other => refuse(format!(
            "policy.name {other:?}; supported: pi05, pi0, smolvla, xvla, vlajepa"
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
    ("vla_jepa_prompt", None),
    ("vla_jepa_inputs", None),
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
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoiseParams {
    shape: Vec<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    seed: Option<u64>,
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
            PreStep::SmolVlaResize(SmolVlaResizeStep {
                image_resolution: p.image_resolution,
                image_key_reorder_map: map,
                num_cameras: p.num_cameras,
            })
        }
        "new_line" => {
            no_params(&params, &ctx)?;
            PreStep::NewLine
        }
        "noise" => {
            let p: NoiseParams = typed(params, &ctx)?;
            check_shape(Some(&p.shape), &ctx)?;
            PreStep::Noise { shape: p.shape }
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
        "identity" => NormMode::Identity,
        m => return refuse(format!("{ctx}: unknown mode {m:?}")),
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
            feature_stats(entry, mode, &format!("{ctx}[{key}]"))?,
        ));
    }
    if mode != NormMode::Identity && features.is_empty() {
        return refuse(format!("{ctx}: no stats for {target:?}"));
    }
    Ok(StatsStep { mode, features })
}

fn feature_stats(entry: &Value, mode: NormMode, ctx: &str) -> EngineResult<FeatureStats> {
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
                shape: vec![50, 32]
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
        assert!(m
            .preprocessors
            .contains(&PreStep::Noise { shape: vec![7, 7] }));
        assert!(m.postprocessors.is_empty());
        assert_eq!(m.normalize().count(), 0, "stats are graph constants");
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
            json!({"class_path": "physicalai.inference.preprocessors.XR0Preprocessor"});
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
        assert!(err(&v).contains("xvla, vlajepa"));
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
