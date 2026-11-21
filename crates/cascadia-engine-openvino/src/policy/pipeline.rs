//! The serving pipeline: the manifest's steps around one compiled policy and
//! its tokenizer. Both sit behind small traits so the whole infer path runs
//! against fakes in the stub build; `ov.rs` supplies the OpenVINO halves.

use std::collections::HashMap;
use std::time::Instant;

use cascadia_engine::{check_observation, EngineError, EngineResult, PolicyEngine, WARMUP_CALLS};
use cascadia_ov_genai_shim::{DType, PluginConfig, PortInfo};
use cascadia_types::{ActionChunk, Image, NoiseMode, Observation, PolicySpec};

use super::cameras::{images_tensor, CameraLayout};
use super::gr00t::Gr00tHost;
use super::manifest::{
    ImageStep, PolicyKind, PolicyManifest, PostStep, PreStep, VlaJepaInputsStep, XVLA_ACTION_DIM,
};
use super::molmoact2::MolmoAct2Host;
use super::qwen_vl::PatchGeometry;
use super::spec::{derive_spec, IrFacts, ServerFacts};
use super::special_tokens;
use super::steps::{
    denormalize, gripper_binarize, new_line, normalize, pi05_bins, pi05_prompt, py_strip,
    trim_actions, xvla_rotation_6d_to_axis_angle, NUM_BINS,
};
use super::vla_jepa::VlaJepaHost;
use super::xr0::Xr0Host;

/// Longest prompt a request may carry, in UTF-8 bytes.
pub const PROMPT_MAX_BYTES: usize = 1024;

/// Distinct task texts whose prompt-length answer is remembered. The answer
/// is a pure function of the text, so dropping the cache only costs a re-check.
const PROMPT_CACHE_MAX: usize = 256;

pub const STATE: &str = "state";
pub const NOISE: &str = "noise";
pub const PROMPT: &str = "tokenized_prompt";
pub const PROMPT_MASK: &str = "tokenized_prompt_mask";
pub const IMAGES: &str = "images";
pub const IMAGE_MASKS: &str = "image_masks";
pub const ACTION: &str = "action";
pub const INPUT_IDS: &str = "input_ids";
pub const ATTENTION_MASK: &str = "attention_mask";
pub const PIXEL_VALUES: &str = "pixel_values";
pub const EMBODIED_POSITIONS: &str = "embodied_positions";
pub const POSITION_IDS: &str = "position_ids";
pub const TOKEN_TYPE_IDS: &str = "token_type_ids";
pub const TOKEN_POOLING: &str = "token_pooling";
pub const ACTION_DIM_IS_PAD: &str = "action_dim_is_pad";
/// The real-time chunking port group (physicalai's `RTCModelMixin`); all or none.
pub const PREV_CHUNK_LEFT_OVER: &str = "prev_chunk_left_over";
pub const INFERENCE_DELAY: &str = "inference_delay";
pub const MAX_GUIDANCE_WEIGHT: &str = "max_guidance_weight";
pub const EXECUTION_HORIZON: &str = "execution_horizon";
/// Bound on a normalized leftover value. Real leftovers are actions the model
/// produced, a few standard deviations from the mean; past this the graph's
/// f32, and the GPU's f16 (max 65504), arithmetic can overflow into NaN.
pub const RTC_LEFTOVER_NORMALIZED_MAX: f64 = 1e4;
const RTC_PORTS: [&str; 4] = [
    PREV_CHUNK_LEFT_OVER,
    INFERENCE_DELAY,
    MAX_GUIDANCE_WEIGHT,
    EXECUTION_HORIZON,
];

/// Every warmup state value. Not zero: X-VLA's LIBERO export reads part of the
/// state as a rotation vector and divides by its norm behind a `1e-8` guard
/// that is 0 in f16, so an all-zero state is all-NaN on the GPU default.
const WARMUP_STATE: f32 = 0.1;

/// Op type that marks a package whose start noise is drawn inside the graph.
pub const RANDOM_UNIFORM: &str = "RandomUniform";

/// A row-major host tensor handed to or read from a model.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl HostTensor {
    pub fn dtype(&self) -> DType {
        match self.data {
            TensorData::F32(_) => DType::F32,
            TensorData::I64(_) => DType::I64,
            TensorData::Bool(_) => DType::Bool,
        }
    }

    /// Little-endian bytes as the shim's `set_input` takes them.
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.data {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::Bool(v) => v.iter().map(|&b| u8::from(b)).collect(),
        }
    }
}

/// One tokenized prompt (batch of one).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tokens {
    pub ids: Vec<i64>,
    pub mask: Vec<bool>,
}

/// The compiled policy IR.
pub trait PolicyModel: Send {
    fn set_input(&mut self, name: &str, tensor: &HostTensor) -> EngineResult<()>;
    fn infer(&mut self) -> EngineResult<()>;
    /// An output by tensor name, as f32.
    fn output_f32(&mut self, name: &str) -> EngineResult<(Vec<usize>, Vec<f32>)>;
    /// GPU memory the compiled model holds now; `None` off GPU or when the
    /// device cannot say.
    fn gpu_memory_bytes(&self) -> Option<u64> {
        None
    }
}

/// The package's tokenizer IR.
pub trait PromptTokenizer: Send {
    /// Whether the IR has a `truncation` state variable. When it does, the
    /// implementation keeps it false, so an overlong prompt comes out wider
    /// than the port instead of being cut.
    fn has_truncation_state(&self) -> bool;
    fn tokenize(&mut self, text: &str) -> EngineResult<Tokens>;
    /// Strings the tokenizer reads as special tokens (see `special_tokens`).
    fn special_tokens(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Guards a prompt template against a task that spells its structure: a
/// special token in the task would move image features or cut the template,
/// and the model would get a silently wrong input.
#[derive(Clone, Debug)]
struct PromptGuard {
    /// The tokenizer's special tokens and the template's markers.
    forbidden: Vec<String>,
    /// `(marker, id, count)`: each single-token marker and how often the
    /// template's ids carry it; every call's ids must carry it as often.
    counts: Vec<(String, i64, usize)>,
}

impl PromptGuard {
    /// `varying`: strings refused in a task whose count the template does not
    /// fix (MolmoAct2's state tokens).
    fn new(
        template: &str,
        tokenizer: &mut dyn PromptTokenizer,
        varying: &[String],
    ) -> EngineResult<Self> {
        let mut forbidden = tokenizer.special_tokens();
        if forbidden.is_empty() {
            return refuse(
                "the tokenizer declares no special tokens, so a templated prompt cannot be \
                 guarded against a task that spells one",
            );
        }
        for m in special_tokens::template_markers(template) {
            if !forbidden.contains(&m) {
                forbidden.push(m);
            }
        }
        let ids = |t: Tokens| -> Vec<i64> {
            t.ids
                .iter()
                .zip(&t.mask)
                .filter(|(_, &m)| m)
                .map(|(&id, _)| id)
                .collect()
        };
        let base = ids(tokenizer.tokenize(template)?);
        let mut counts = Vec::new();
        for m in &forbidden {
            if varying.contains(m) {
                continue;
            }
            if let [id] = ids(tokenizer.tokenize(m)?)[..] {
                counts.push((m.clone(), id, base.iter().filter(|&&x| x == id).count()));
            }
        }
        for v in varying {
            if !forbidden.contains(v) {
                forbidden.push(v.clone());
            }
        }
        Ok(Self { forbidden, counts })
    }

    fn check_task(&self, task: &str) -> Option<String> {
        self.forbidden.iter().find(|f| task.contains(f.as_str())).map(|f| {
            format!("the task contains {f:?}, which this policy's prompt reads as structure; remove it")
        })
    }

    fn count(&self, marker: &str) -> Option<usize> {
        self.counts
            .iter()
            .find(|(m, _, _)| m == marker)
            .map(|c| c.2)
    }

    /// Defence in depth behind `check_task`: the ids keep the template's structure.
    fn check_ids(&self, tokens: &Tokens) -> EngineResult<()> {
        for (m, id, want) in &self.counts {
            let got = tokens
                .ids
                .iter()
                .zip(&tokens.mask)
                .filter(|(&x, &on)| on && x == *id)
                .count();
            if got != *want {
                return Err(EngineError::InvalidConfig(format!(
                    "the prompt has {got} {m:?} tokens; the template has {want}"
                )));
            }
        }
        Ok(())
    }
}

/// A cached answer for a task text.
#[derive(Clone, Debug)]
enum Refusal {
    TooLong(String),
    Structure(String),
}

impl Refusal {
    fn error(self) -> EngineError {
        match self {
            Self::TooLong(m) => EngineError::PromptTooLong(m),
            Self::Structure(m) => EngineError::InvalidConfig(m),
        }
    }
}

/// A tokenizer failure on a request's text. The tokenizer handled the
/// package's own prompt at load and warmup, so a later failure follows from
/// the text the client sent: a refusal, which leaves `/health` alone.
fn refused_text(e: EngineError) -> EngineError {
    match e {
        EngineError::Backend(m) => {
            EngineError::InvalidConfig(format!("the prompt could not be tokenized: {m}"))
        }
        other => other,
    }
}

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("robot model: {msg}")))
}

/// `--ov-inference-precision` on ov-policy: GPU compute precision. CPU always runs f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F16,
    F32,
}

/// The upper-cased device (`gpu.1` is `GPU.1`) and the plugin config for the
/// policy IR on it.
///
/// CPU always gets `INFERENCE_PRECISION_HINT=f32`: its bf16 compute is off by
/// up to 0.66 normalized units, and a bf16 export with a padded mask is
/// all-NaN on bf16 CPUs. The GPU keeps its f16 default unless `F32` is asked.
pub fn compile_config(
    device: &str,
    precision: Option<Precision>,
) -> EngineResult<(String, PluginConfig)> {
    let dev = device.trim().to_ascii_uppercase();
    let family = dev.split('.').next().unwrap_or_default();
    let f32_hint = PluginConfig::new().with("INFERENCE_PRECISION_HINT", "f32");
    let config = match (family, precision) {
        ("CPU", Some(Precision::F16)) => Err(EngineError::InvalidConfig(
            "--ov-inference-precision f16 is not available on CPU, which always runs f32".into(),
        )),
        ("CPU", _) => Ok(f32_hint),
        ("GPU", Some(Precision::F32)) => Ok(f32_hint),
        ("GPU", _) => Ok(PluginConfig::new()),
        ("NPU", _) => Err(EngineError::InvalidConfig(
            "NPU is not supported for policies: the exported packages fail NPU compilation or \
             return wrong actions there"
                .into(),
        )),
        _ => Err(EngineError::InvalidConfig(format!(
            "device {device:?} is not supported for policies; use GPU or CPU"
        ))),
    }?;
    Ok((dev, config))
}

/// The precision [`compile_config`] compiles at, as the metadata states it:
/// CPU is always f32; GPU is f16 unless f32 is asked for.
pub fn compiled_precision(device: &str, precision: Option<Precision>) -> &'static str {
    let dev = device.trim().to_ascii_uppercase();
    match (dev.split('.').next().unwrap_or_default(), precision) {
        ("GPU", Some(Precision::F32)) | ("CPU", _) => "f32",
        _ => "f16",
    }
}

/// Refuse a kind on a device whose plugin is known to compute it wrongly, so
/// a robot never gets wrong actions quietly. `arch` is the build's
/// `target_arch`.
pub fn refuse_known_bad_device(kind: PolicyKind, device: &str, arch: &str) -> EngineResult<()> {
    let family = device.split('.').next().unwrap_or_default();
    if kind == PolicyKind::MolmoAct2 && family.eq_ignore_ascii_case("CPU") && arch == "aarch64" {
        return Err(EngineError::InvalidConfig(
            "molmoact2 on an ARM CPU is refused: OpenVINO's ARM CPU plugin computes this graph \
             wrongly (6.1e-1 off PyTorch on PyTorch's own inputs); use an x86 CPU or the GPU"
                .into(),
        ));
    }
    Ok(())
}

/// The input ports a policy kind's IR takes. Each layout has its own binder
/// and its own feeds; the prompt, noise and postprocessing are shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortLayout {
    /// SmolVLA, π0.5, π0, X-VLA: `state` (normalized), `tokenized_prompt`,
    /// `tokenized_prompt_mask`, `images`, `image_masks`, optional `noise`.
    ImageSlots,
    /// VLA-JEPA: `input_ids` and `attention_mask` (both i64), `pixel_values
    /// [n_cameras * patches, feature]`, `embodied_positions`, the raw `state`
    /// (the graph normalizes it), `noise`; `action` is final.
    VlaJepa,
    /// XR0: `tokenized_prompt` and `tokenized_prompt_mask` (both i64),
    /// `pixel_values`, `state [1, history, max_state_dim]` (raw, zero-padded), `noise`.
    Xr0,
    /// GR00T N1.7: `input_ids`, `attention_mask` (i64, image tokens expanded),
    /// `pixel_values`, `position_ids [3, 1, seq_len]`, `state [1, 1, pad_to]`
    /// (normalized, zero-padded), the `constant` ports and the noise port, in
    /// that order; each bound by name, else by its index (the CPU plugin renames
    /// `state`). The one output is the action, whatever its name.
    Gr00t,
    /// MolmoAct2: `input_ids`, `attention_mask`, `token_type_ids` (i64, image
    /// tokens expanded), `images [1, cameras, patches, patch²·3]`, `token_pooling`,
    /// `action_dim_is_pad` (bool); no state or noise input.
    MolmoAct2,
}

impl PortLayout {
    pub fn of(kind: PolicyKind) -> Self {
        match kind {
            PolicyKind::Pi05
            | PolicyKind::Pi0
            | PolicyKind::SmolVla
            | PolicyKind::Xvla
            | PolicyKind::Evo1 => Self::ImageSlots,
            PolicyKind::VlaJepa => Self::VlaJepa,
            PolicyKind::Xr0 => Self::Xr0,
            PolicyKind::Gr00tN1d7 => Self::Gr00t,
            PolicyKind::MolmoAct2 => Self::MolmoAct2,
        }
    }
}

/// Tensor names the pipeline binds, plus the prompt port's length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortBinding {
    pub layout: PortLayout,
    pub noise: bool,
    /// The port the token ids go to.
    pub prompt_port: &'static str,
    /// The tokenizer's width (GR00T: the unexpanded ids `seq_len` leaves room for).
    pub prompt_len: usize,
    /// `(feed name, tensor name)` where the model's port has another name.
    pub renames: Vec<(String, String)>,
    pub noise_port: String,
    pub action_port: String,
    /// The RTC scalars' port shapes (`[]` or `[1]`), when the model has the group.
    pub rtc: Option<RtcShapes>,
}

/// Shapes of the RTC scalar ports, as the IR declares them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtcShapes {
    pub inference_delay: Vec<usize>,
    pub max_guidance_weight: Vec<usize>,
    pub execution_horizon: Vec<usize>,
}

impl PortBinding {
    fn new(layout: PortLayout, noise: bool, prompt_port: &'static str, prompt_len: usize) -> Self {
        Self {
            layout,
            noise,
            prompt_port,
            prompt_len,
            renames: Vec::new(),
            noise_port: NOISE.into(),
            action_port: ACTION.into(),
            rtc: None,
        }
    }

    /// The model's name for feed `name`.
    pub fn port_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.renames
            .iter()
            .find(|(feed, _)| feed == name)
            .map_or(name, |(_, port)| port.as_str())
    }
}

fn port<'a>(ports: &'a [PortInfo], name: &str) -> Option<&'a PortInfo> {
    ports.iter().find(|p| p.aliases.iter().any(|a| a == name))
}

/// Dims of input/output `name`, checked for dtype and rank; a leading batch
/// must be 1 or dynamic.
fn dims(
    ports: &[PortInfo],
    name: &str,
    dtypes: &[DType],
    rank: usize,
    batch_at: Option<usize>,
) -> EngineResult<Vec<Option<usize>>> {
    let Some(p) = port(ports, name) else {
        return refuse(format!("the model has no {name:?} port"));
    };
    match p.dtype {
        Some(d) if dtypes.contains(&d) => {}
        other => {
            return refuse(format!(
                "{name:?} is {}; expected {dtypes:?}",
                other.map_or_else(
                    || "an unsupported element type".into(),
                    |d| format!("{d:?}")
                )
            ))
        }
    }
    if p.dims.len() != rank {
        return refuse(format!(
            "{name:?} has rank {}; expected {rank}",
            p.dims.len()
        ));
    }
    if let Some(b) = batch_at {
        if !matches!(p.dims[b], None | Some(1)) {
            return refuse(format!("{name:?} has batch {:?}; expected 1", p.dims[b]));
        }
    }
    Ok(p.dims.clone())
}

fn fixed(d: Option<usize>, what: &str) -> EngineResult<usize> {
    d.map_or_else(
        || refuse(format!("{what} is dynamic; the engine needs it static")),
        Ok,
    )
}

/// Check the compiled model's ports against `m`'s policy kind and read the
/// facts the spec needs.
pub fn bind(
    m: &PolicyManifest,
    inputs: &[PortInfo],
    outputs: &[PortInfo],
    has_random_uniform: bool,
) -> EngineResult<(PortBinding, IrFacts)> {
    match PortLayout::of(m.policy) {
        PortLayout::ImageSlots => bind_ports(inputs, outputs, has_random_uniform),
        PortLayout::VlaJepa => match m.image_step() {
            ImageStep::VlaJepa(step) => bind_vla_jepa(step, inputs, outputs, has_random_uniform),
            _ => refuse("vlajepa needs a vla_jepa_inputs step"),
        },
        PortLayout::Xr0 => match m.image_step() {
            ImageStep::Xr0(step) => {
                bind_xr0(&Xr0Host::new(m, step)?, inputs, outputs, has_random_uniform)
            }
            _ => refuse("xr0 needs an xr0 step"),
        },
        PortLayout::Gr00t => bind_gr00t(m, &Gr00tHost::new(m)?, inputs, outputs),
        PortLayout::MolmoAct2 => {
            bind_molmoact2(&MolmoAct2Host::new(m)?, inputs, outputs, has_random_uniform)
        }
    }
}

/// [`PortLayout::MolmoAct2`]: the ports `host` fills, sized by its geometry.
pub fn bind_molmoact2(
    host: &MolmoAct2Host,
    inputs: &[PortInfo],
    outputs: &[PortInfo],
    has_random_uniform: bool,
) -> EngineResult<(PortBinding, IrFacts)> {
    refuse_unknown_inputs(
        inputs,
        &[
            INPUT_IDS,
            ATTENTION_MASK,
            TOKEN_TYPE_IDS,
            IMAGES,
            TOKEN_POOLING,
            ACTION_DIM_IS_PAD,
        ],
    )?;
    let ids = dims(inputs, INPUT_IDS, &[DType::I64], 2, Some(0))?;
    let seq = fixed(ids[1], "the input_ids length")?;
    for name in [ATTENTION_MASK, TOKEN_TYPE_IDS] {
        let d = dims(inputs, name, &[DType::I64], 2, Some(0))?;
        if d[1] != Some(seq) {
            return refuse(format!("{name:?} is {d:?}; input_ids is [1, {seq}]"));
        }
    }
    let cams = host.geo.cameras.len();
    let images = dims(inputs, IMAGES, &[DType::F32], 4, Some(0))?;
    let want = [
        Some(cams),
        Some(host.geo.patches_per_image()),
        Some(host.geo.patch_pixels),
    ];
    if images[1..] != want {
        return refuse(format!(
            "images is {images:?}; molmoact2_inputs makes [1, {cams}, {}, {}]",
            host.geo.patches_per_image(),
            host.geo.patch_pixels
        ));
    }
    let pool = dims(inputs, TOKEN_POOLING, &[DType::I64], 3, Some(0))?;
    let cells = host.token_pooling().len() / cams.max(1) / host.geo.pooled_per_image().max(1);
    if pool[1..] != [Some(cams * host.geo.pooled_per_image()), Some(cells)] {
        return refuse(format!(
            "token_pooling is {pool:?}; molmoact2_inputs makes [1, {}, {cells}]",
            cams * host.geo.pooled_per_image()
        ));
    }
    let pad = dims(inputs, ACTION_DIM_IS_PAD, &[DType::Bool], 2, Some(0))?;
    if pad[1] != Some(host.max_action_dim()) {
        return refuse(format!(
            "action_dim_is_pad is {pad:?}; max_action_dim is {}",
            host.max_action_dim()
        ));
    }
    // The tokenizer's width: the sequence less BOS and what the images add.
    let added = 1 + cams * (host.image_tokens().len() - 1);
    let Some(width) = seq.checked_sub(added).filter(|&w| w > 0) else {
        return refuse(format!(
            "input_ids is {seq} long; BOS and {cams} images alone take {}",
            added + 1
        ));
    };
    let action_shape = action_dims(outputs)?;
    Ok((
        PortBinding::new(
            PortLayout::MolmoAct2,
            false,
            "input_ids (before BOS and image expansion)",
            width,
        ),
        IrFacts {
            image_slots: cams,
            image_hw: None,
            state_len: host.geo.state_len,
            action_shape,
            noise: noise_mode(&None, has_random_uniform),
            noise_shape: None,
            rtc: false,
        },
    ))
}

/// [`PortLayout::Gr00t`]: each input by name, else by its index when the
/// port there carries none of the layout's names.
pub fn bind_gr00t(
    m: &PolicyManifest,
    host: &Gr00tHost,
    inputs: &[PortInfo],
    outputs: &[PortInfo],
) -> EngineResult<(PortBinding, IrFacts)> {
    let has_noise = m
        .preprocessors
        .iter()
        .any(|s| matches!(s, PreStep::Noise { .. }));
    let mut want: Vec<String> = [INPUT_IDS, ATTENTION_MASK, PIXEL_VALUES, POSITION_IDS, STATE]
        .iter()
        .map(|s| s.to_string())
        .chain(m.constants().map(|(p, _)| p.to_string()))
        .collect();
    if has_noise {
        want.push(m.noise_port().to_string());
    }
    if inputs.len() != want.len() {
        return refuse(format!(
            "the model has {} inputs; gr00t fills {} ({want:?})",
            inputs.len(),
            want.len()
        ));
    }
    let named = |p: &PortInfo| p.aliases.iter().any(|a| want.contains(a));
    let mut ports: Vec<PortInfo> = Vec::with_capacity(want.len());
    let mut renames = Vec::new();
    for (i, name) in want.iter().enumerate() {
        let p = match inputs.iter().find(|p| p.aliases.contains(name)) {
            Some(p) => p,
            None if !named(&inputs[i]) => match inputs[i].aliases.first() {
                Some(actual) => {
                    renames.push((name.clone(), actual.clone()));
                    &inputs[i]
                }
                None => return refuse(format!("input {i} has no name to bind {name:?} by")),
            },
            None => {
                return refuse(format!(
                    "the model has no {name:?} port, and input {i} is {:?}",
                    inputs[i].aliases
                ))
            }
        };
        ports.push(PortInfo {
            aliases: vec![name.clone()],
            ..p.clone()
        });
    }
    let seq = host.seq_len();
    for name in [INPUT_IDS, ATTENTION_MASK] {
        let d = dims(&ports, name, &[DType::I64], 2, Some(0))?;
        if d[1] != Some(seq) {
            return refuse(format!(
                "{name:?} is {d:?}; qwen_vl_inputs seq_len is {seq}"
            ));
        }
    }
    let pos = dims(&ports, POSITION_IDS, &[DType::I64], 3, None)?;
    if pos != [Some(3), Some(1), Some(seq)] {
        return refuse(format!(
            "position_ids is {pos:?}; gr00t makes [3, 1, {seq}]"
        ));
    }
    let image_slots = pixel_slots(&ports, &host.geo.patches, "qwen_vl_patchify")?;
    let width = host.geo.state_width;
    let state = dims(&ports, STATE, &[DType::F32], 3, Some(0))?;
    if state[1..] != [Some(1), Some(width)] {
        return refuse(format!("state is {state:?}; gr00t makes [1, 1, {width}]"));
    }
    for (port, values) in m.constants() {
        let d = dims(&ports, port, &[DType::I64], 1, None)?;
        if d != [Some(values.len())] {
            return refuse(format!(
                "constant {port:?} has {} values; the port is {d:?}",
                values.len()
            ));
        }
    }
    let noise_shape = if has_noise {
        let p = m.noise_port();
        let d = dims(&ports, p, &[DType::F32], 3, Some(0))?;
        Some(
            d[1..]
                .iter()
                .map(|&x| fixed(x, "a noise dim"))
                .collect::<EngineResult<Vec<_>>>()?,
        )
    } else {
        None
    };
    let action_port = match (port(outputs, ACTION), outputs) {
        (Some(_), _) => ACTION.to_string(),
        (None, [only]) => match only.aliases.first() {
            Some(a) => a.clone(),
            None => return refuse("the model's output has no name"),
        },
        (None, _) => {
            return refuse(format!(
                "the model has {} outputs and none is named \"action\"",
                outputs.len()
            ))
        }
    };
    let out = [PortInfo {
        aliases: vec![ACTION.into()],
        ..port(outputs, &action_port).expect("found above").clone()
    }];
    let action_shape = action_dims(&out)?;
    let noise = noise_mode(&noise_shape, false);
    let mut binding = PortBinding::new(PortLayout::Gr00t, has_noise, INPUT_IDS, host.id_budget());
    binding.renames = renames;
    binding.noise_port = m.noise_port().to_string();
    binding.action_port = action_port;
    Ok((
        binding,
        IrFacts {
            image_slots,
            image_hw: None,
            state_len: host.geo.state_len,
            action_shape,
            noise,
            noise_shape,
            rtc: false,
        },
    ))
}

/// `pixel_values` checked against `geo`; returns the camera count.
fn pixel_slots(inputs: &[PortInfo], geo: &PatchGeometry, step: &str) -> EngineResult<usize> {
    let px = dims(inputs, PIXEL_VALUES, &[DType::F32], 2, None)?;
    let rows = fixed(px[0], "the pixel_values row count")?;
    let feature = fixed(px[1], "the pixel_values width")?;
    if feature != geo.feature() {
        return refuse(format!(
            "pixel_values rows are {feature} wide; {step} makes {}",
            geo.feature()
        ));
    }
    if rows == 0 || rows % geo.patches() != 0 {
        return refuse(format!(
            "pixel_values has {rows} rows, not a whole number of cameras of {} patches",
            geo.patches()
        ));
    }
    Ok(rows / geo.patches())
}

/// [`PortLayout::Xr0`]: the ports `host` fills.
pub fn bind_xr0(
    host: &Xr0Host,
    inputs: &[PortInfo],
    outputs: &[PortInfo],
    has_random_uniform: bool,
) -> EngineResult<(PortBinding, IrFacts)> {
    refuse_unknown_inputs(inputs, &[PROMPT, PROMPT_MASK, PIXEL_VALUES, STATE, NOISE])?;
    let ids = dims(inputs, PROMPT, &[DType::I64], 2, Some(0))?;
    let prompt_len = fixed(ids[1], "the tokenized_prompt length")?;
    let mask = dims(inputs, PROMPT_MASK, &[DType::I64], 2, Some(0))?;
    if mask[1] != Some(prompt_len) {
        return refuse(format!(
            "tokenized_prompt_mask length {:?} differs from tokenized_prompt's {prompt_len}",
            mask[1]
        ));
    }
    let state = dims(inputs, STATE, &[DType::F32], 3, Some(0))?;
    if state[1] != Some(host.geo.history) || state[2] != Some(host.max_state_dim()) {
        return refuse(format!(
            "state is {state:?}; xr0 makes [1, {}, {}]",
            host.geo.history,
            host.max_state_dim()
        ));
    }
    let image_slots = pixel_slots(inputs, &host.geo.patches, "xr0")?;
    let noise_shape = noise_dims(inputs)?;
    let action_shape = action_dims(outputs)?;
    let noise = noise_mode(&noise_shape, has_random_uniform);
    Ok((
        PortBinding::new(PortLayout::Xr0, noise_shape.is_some(), PROMPT, prompt_len),
        IrFacts {
            image_slots,
            image_hw: None,
            state_len: host.geo.state_len,
            action_shape,
            noise,
            noise_shape,
            rtc: false,
        },
    ))
}

/// [`PortLayout::VlaJepa`]: the ports `step` fills, checked against its geometry.
pub fn bind_vla_jepa(
    step: &VlaJepaInputsStep,
    inputs: &[PortInfo],
    outputs: &[PortInfo],
    has_random_uniform: bool,
) -> EngineResult<(PortBinding, IrFacts)> {
    refuse_unknown_inputs(
        inputs,
        &[
            INPUT_IDS,
            ATTENTION_MASK,
            PIXEL_VALUES,
            EMBODIED_POSITIONS,
            STATE,
            NOISE,
        ],
    )?;
    let state = dims(inputs, STATE, &[DType::F32], 2, Some(0))?;
    let state_len = fixed(state[1], "the state length")?;
    let ids = dims(inputs, INPUT_IDS, &[DType::I64], 2, Some(0))?;
    let prompt_len = fixed(ids[1], "the input_ids length")?;
    let mask = dims(inputs, ATTENTION_MASK, &[DType::I64], 2, Some(0))?;
    if mask[1] != Some(prompt_len) {
        return refuse(format!(
            "attention_mask length {:?} differs from input_ids' {prompt_len}",
            mask[1]
        ));
    }
    let host = VlaJepaHost::new(step);
    let image_slots = pixel_slots(inputs, host.geometry(), "vla_jepa_inputs")?;
    let embodied = dims(inputs, EMBODIED_POSITIONS, &[DType::I64], 2, Some(0))?;
    let embodied = fixed(embodied[1], "the embodied_positions length")?;
    if embodied != step.num_embodied_tokens {
        return refuse(format!(
            "embodied_positions takes {embodied}; num_embodied_tokens is {}",
            step.num_embodied_tokens
        ));
    }
    let noise_shape = noise_dims(inputs)?;
    let action_shape = action_dims(outputs)?;
    let noise = noise_mode(&noise_shape, has_random_uniform);
    Ok((
        PortBinding::new(
            PortLayout::VlaJepa,
            noise_shape.is_some(),
            INPUT_IDS,
            prompt_len,
        ),
        IrFacts {
            image_slots,
            image_hw: None,
            state_len,
            action_shape,
            noise,
            noise_shape,
            rtc: false,
        },
    ))
}

/// Inputs no layout lists refuse the load: the engine would leave them unset.
fn refuse_unknown_inputs(inputs: &[PortInfo], known: &[&str]) -> EngineResult<()> {
    match inputs
        .iter()
        .find(|p| !p.aliases.iter().any(|a| known.contains(&a.as_str())))
    {
        Some(p) => refuse(format!("input {:?} is not one the engine fills", p.aliases)),
        None => Ok(()),
    }
}

/// `noise`'s dims after the batch, when the model has one.
fn noise_dims(inputs: &[PortInfo]) -> EngineResult<Option<Vec<usize>>> {
    match port(inputs, NOISE) {
        Some(p) => {
            let d = dims(inputs, NOISE, &[DType::F32], p.dims.len().max(2), Some(0))?;
            Ok(Some(
                d[1..]
                    .iter()
                    .map(|&x| fixed(x, "a noise dim"))
                    .collect::<EngineResult<Vec<_>>>()?,
            ))
        }
        None => Ok(None),
    }
}

/// The RTC port group: none, or all four with the leftover `[1, T, D]` of the
/// action output and the scalars as physicalai's exporter declares them.
fn rtc_ports(inputs: &[PortInfo], action: [usize; 2]) -> EngineResult<Option<RtcShapes>> {
    let (present, missing): (Vec<&str>, Vec<&str>) =
        RTC_PORTS.iter().partition(|n| port(inputs, n).is_some());
    if present.is_empty() {
        return Ok(None);
    }
    if !missing.is_empty() {
        return refuse(format!(
            "the RTC ports come as a group: the model has {present:?} but not {missing:?}"
        ));
    }
    let left = dims(inputs, PREV_CHUNK_LEFT_OVER, &[DType::F32], 3, Some(0))?;
    if left[1..] != [Some(action[0]), Some(action[1])] {
        let shown: Vec<String> = left
            .iter()
            .map(|d| d.map_or_else(|| "?".into(), |d| d.to_string()))
            .collect();
        return refuse(format!(
            "prev_chunk_left_over is [{}]; the action output is [1, {}, {}]",
            shown.join(", "),
            action[0],
            action[1]
        ));
    }
    let scalar = |name: &str, dtype: DType| -> EngineResult<Vec<usize>> {
        let p = port(inputs, name).expect("present");
        let shape = match p.dims.as_slice() {
            [] => vec![],
            [Some(1)] => vec![1],
            other => return refuse(format!("{name} has shape {other:?}; expected a scalar")),
        };
        dims(inputs, name, &[dtype], shape.len(), None)?;
        Ok(shape)
    };
    Ok(Some(RtcShapes {
        inference_delay: scalar(INFERENCE_DELAY, DType::I64)?,
        max_guidance_weight: scalar(MAX_GUIDANCE_WEIGHT, DType::F32)?,
        execution_horizon: scalar(EXECUTION_HORIZON, DType::I64)?,
    }))
}

/// `[T, D]` of the `action` output.
fn action_dims(outputs: &[PortInfo]) -> EngineResult<[usize; 2]> {
    let action = dims(outputs, ACTION, &[DType::F32, DType::F16], 3, Some(0))?;
    Ok([
        fixed(action[1], "the action chunk length")?,
        fixed(action[2], "the action dim")?,
    ])
}

fn noise_mode(noise_shape: &Option<Vec<usize>>, has_random_uniform: bool) -> NoiseMode {
    if noise_shape.is_some() {
        NoiseMode::Input
    } else if has_random_uniform {
        NoiseMode::InGraphRandom
    } else {
        NoiseMode::Constant
    }
}

/// [`PortLayout::ImageSlots`]: check the compiled model's ports and read the
/// facts the spec needs.
///
/// Refuses an input the engine does not fill, a missing input, or a dtype or
/// shape the pipeline cannot produce, so a mismatch fails at load.
pub fn bind_ports(
    inputs: &[PortInfo],
    outputs: &[PortInfo],
    has_random_uniform: bool,
) -> EngineResult<(PortBinding, IrFacts)> {
    refuse_unknown_inputs(
        inputs,
        &[
            STATE,
            NOISE,
            PROMPT,
            PROMPT_MASK,
            IMAGES,
            IMAGE_MASKS,
            PREV_CHUNK_LEFT_OVER,
            INFERENCE_DELAY,
            MAX_GUIDANCE_WEIGHT,
            EXECUTION_HORIZON,
        ],
    )?;
    let state = dims(inputs, STATE, &[DType::F32], 2, Some(0))?;
    let state_len = fixed(state[1], "the state length")?;
    let prompt = dims(inputs, PROMPT, &[DType::I64], 2, Some(0))?;
    let prompt_len = fixed(prompt[1], "the tokenized_prompt length")?;
    let mask = dims(inputs, PROMPT_MASK, &[DType::Bool], 2, Some(0))?;
    if mask[1] != Some(prompt_len) {
        return refuse(format!(
            "tokenized_prompt_mask length {:?} differs from tokenized_prompt's {prompt_len}",
            mask[1]
        ));
    }
    let images = dims(inputs, IMAGES, &[DType::F32], 5, Some(1))?;
    let image_slots = fixed(images[0], "the images slot count")?;
    if images[2] != Some(3) {
        return refuse(format!("images has {:?} channels; expected 3", images[2]));
    }
    let image_hw = match (images[3], images[4]) {
        (Some(h), Some(w)) => Some([
            u32::try_from(h)
                .map_err(|_| EngineError::InvalidConfig("image height overflows".into()))?,
            u32::try_from(w)
                .map_err(|_| EngineError::InvalidConfig("image width overflows".into()))?,
        ]),
        _ => None,
    };
    let masks = dims(inputs, IMAGE_MASKS, &[DType::Bool], 2, Some(1))?;
    if masks[0] != Some(image_slots) {
        return refuse(format!(
            "image_masks has {:?} slots; images has {image_slots}",
            masks[0]
        ));
    }
    let noise_shape = noise_dims(inputs)?;
    let action_shape = action_dims(outputs)?;
    let noise = noise_mode(&noise_shape, has_random_uniform);
    let rtc = rtc_ports(inputs, action_shape)?;
    let mut binding = PortBinding::new(
        PortLayout::ImageSlots,
        noise_shape.is_some(),
        PROMPT,
        prompt_len,
    );
    binding.rtc = rtc.clone();
    Ok((
        binding,
        IrFacts {
            image_slots,
            image_hw,
            state_len,
            action_shape,
            noise,
            noise_shape,
            rtc: rtc.is_some(),
        },
    ))
}

/// Box-Muller over xoshiro256++: reproducible from a seed with no new
/// dependency. Not numpy's stream; noise is never compared across runtimes.
pub struct NoiseRng {
    s: [u64; 4],
}

impl NoiseRng {
    pub fn seeded(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            // splitmix64, the seeding xoshiro's authors recommend.
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    /// Seeded from the OS-keyed `RandomState`, the time and a counter.
    pub fn fresh() -> Self {
        use std::hash::{BuildHasher, Hasher};
        use std::sync::atomic::{AtomicU64, Ordering};
        static CALLS: AtomicU64 = AtomicU64::new(0);
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(CALLS.fetch_add(1, Ordering::Relaxed));
        if let Ok(t) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            h.write_u128(t.as_nanos());
        }
        Self::seeded(h.finish())
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in [0, 1) with 53 bits.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `n` draws of N(0, 1).
    pub fn normal(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let u1 = 1.0 - self.unit(); // (0, 1], so ln is finite
            let u2 = self.unit();
            let r = (-2.0 * u1.ln()).sqrt();
            let a = std::f64::consts::TAU * u2;
            out.push((r * a.cos()) as f32);
            out.push((r * a.sin()) as f32);
        }
        out.truncate(n);
        out
    }
}

/// The text the tokenizer sees: the text steps before `ov_tokenizer`, in
/// manifest order. `worst` puts every π0.5 bin at 255, the longest digits.
fn prompt_text(m: &PolicyManifest, task: &str, state: &[f64], worst: bool) -> EngineResult<String> {
    let mut state = state.to_vec();
    let mut text = match m.policy {
        // LeRobot's EVO1 embedder tokenizes `text_prompt.strip()`.
        PolicyKind::Evo1 => py_strip(task).to_string(),
        _ => task.to_string(),
    };
    for step in &m.preprocessors {
        match step {
            PreStep::Normalize(s) => state = normalize(s, STATE, &state)?,
            PreStep::NewLine => text = new_line(&text),
            PreStep::Pi05(_) => {
                let bins = if worst {
                    vec![NUM_BINS as i32 - 1; state.len()]
                } else {
                    pi05_bins(&state)
                };
                text = pi05_prompt(&text, &bins);
            }
            PreStep::VlaJepaPrompt { template } => {
                text = template.replace("{instruction}", &text);
            }
            PreStep::Xr0(s) => text = super::xr0::task_prompt(m, s, &text)?,
            PreStep::Gr00tPrompt(s) => {
                let cameras = super::gr00t::Gr00tGeometry::new(m)?.cameras;
                text = super::gr00t::render_prompt(s, cameras, &text);
            }
            PreStep::MolmoAct2(s) => {
                let raw: Vec<f32> = state.iter().map(|&x| x as f32).collect();
                text = super::molmoact2::render_prompt(s, &text, &raw);
            }
            PreStep::OvTokenizer { .. } => break,
            PreStep::Pi0(_)
            | PreStep::SmolVlaResize(_)
            | PreStep::Noise { .. }
            | PreStep::ImageFlip { .. }
            | PreStep::VlaJepaInputs(_)
            | PreStep::Gr00tImage(_)
            | PreStep::QwenVlPatchify(_)
            | PreStep::QwenVlInputs(_)
            | PreStep::Constant { .. }
            | PreStep::MolmoAct2Inputs(_) => {}
        }
    }
    Ok(text)
}

/// The task as a template step puts it into the prompt, when that differs
/// from the task as sent; the special-token guard checks both.
fn templated_task(m: &PolicyManifest, task: &str) -> Option<String> {
    m.preprocessors.iter().find_map(|s| match s {
        PreStep::Gr00tPrompt(p) if p.formalize_language => Some(super::gr00t::formalize(task)),
        PreStep::MolmoAct2(_) => Some(super::molmoact2::normalize_text(task)),
        _ => None,
    })
}

/// A loaded policy: manifest steps, ports and spec around a model and tokenizer.
pub struct PolicyPipeline {
    manifest: PolicyManifest,
    spec: PolicySpec,
    layout: CameraLayout,
    /// Per `layout.names`: rotate that camera by 180 degrees.
    flip: Vec<bool>,
    binding: PortBinding,
    action_shape: [usize; 2],
    model: Option<Box<dyn PolicyModel>>,
    tokenizer: Option<Box<dyn PromptTokenizer>>,
    /// [`PortLayout::VlaJepa`]'s precomputed pixel geometry.
    vla_jepa: Option<VlaJepaHost>,
    /// [`PortLayout::Xr0`]'s.
    xr0: Option<Xr0Host>,
    /// [`PortLayout::Gr00t`]'s.
    gr00t: Option<Gr00tHost>,
    /// [`PortLayout::MolmoAct2`]'s.
    molmoact2: Option<MolmoAct2Host>,
    /// Task text to the refusal message, `None` when it fits.
    prompt_checks: HashMap<String, Option<Refusal>>,
    /// Set for a templated prompt (`vla_jepa_prompt`, `xr0`).
    guard: Option<PromptGuard>,
}

impl PolicyPipeline {
    pub fn new(
        manifest: PolicyManifest,
        server: ServerFacts,
        model: Box<dyn PolicyModel>,
        inputs: &[PortInfo],
        outputs: &[PortInfo],
        has_random_uniform: bool,
        mut tokenizer: Box<dyn PromptTokenizer>,
    ) -> EngineResult<Self> {
        let tok_at = manifest
            .preprocessors
            .iter()
            .position(|s| matches!(s, PreStep::OvTokenizer { .. }))
            .expect("parse guarantees one ov_tokenizer");
        if manifest.preprocessors[tok_at..].iter().any(|s| {
            matches!(
                s,
                PreStep::NewLine
                    | PreStep::Pi05(_)
                    | PreStep::Pi0(_)
                    | PreStep::VlaJepaPrompt { .. }
                    | PreStep::Xr0(_)
                    | PreStep::Gr00tPrompt(_)
                    | PreStep::MolmoAct2(_)
            )
        }) {
            return refuse(
                "a prompt step (new_line / pi05 / pi0 / vla_jepa_prompt / xr0 / gr00t_prompt / \
                 molmoact2) comes after ov_tokenizer",
            );
        }
        if let Some(i) = manifest
            .preprocessors
            .iter()
            .position(|s| matches!(s, PreStep::QwenVlInputs(_) | PreStep::MolmoAct2Inputs(_)))
        {
            if i < tok_at {
                return refuse(format!(
                    "{} comes before ov_tokenizer",
                    if matches!(manifest.preprocessors[i], PreStep::QwenVlInputs(_)) {
                        "qwen_vl_inputs"
                    } else {
                        "molmoact2_inputs"
                    }
                ));
            }
        }
        refuse_known_bad_device(manifest.policy, &server.device, std::env::consts::ARCH)?;
        let (binding, ir) = bind(&manifest, inputs, outputs, has_random_uniform)?;
        // Otherwise a tokenizer padded for another port loads and every infer fails.
        // GR00T's tokenizer does not pad: its ids are expanded and padded after it.
        let empty = tokenizer.tokenize("")?;
        if binding.layout != PortLayout::Gr00t
            && (empty.ids.len() != binding.prompt_len || empty.mask.len() != binding.prompt_len)
        {
            return refuse(format!(
                "the tokenizer pads to {} tokens; the model's {} port takes {}",
                empty.ids.len(),
                binding.prompt_port,
                binding.prompt_len
            ));
        }
        let vla_jepa = match manifest.image_step() {
            ImageStep::VlaJepa(step) => {
                let host = VlaJepaHost::new(step);
                // The template alone must carry the embodied tokens the IR reads.
                let text = prompt_text(&manifest, "", &[], false)?;
                host.embodied_positions(&tokenizer.tokenize(&text)?.ids)
                    .map_err(|e| match e {
                        EngineError::InvalidConfig(m) => {
                            EngineError::InvalidConfig(format!("robot model: the empty task: {m}"))
                        }
                        other => other,
                    })?;
                Some(host)
            }
            _ => None,
        };
        let xr0 = match manifest.image_step() {
            ImageStep::Xr0(step) => Some(Xr0Host::new(&manifest, step)?),
            _ => None,
        };
        let gr00t = match manifest.image_step() {
            ImageStep::Gr00t(_) => Some(Gr00tHost::new(&manifest)?),
            _ => None,
        };
        let molmoact2 = match manifest.image_step() {
            ImageStep::MolmoAct2(_) => Some(MolmoAct2Host::new(&manifest)?),
            _ => None,
        };
        let templated = manifest.preprocessors.iter().any(|s| {
            matches!(
                s,
                PreStep::VlaJepaPrompt { .. }
                    | PreStep::Xr0(_)
                    | PreStep::Gr00tPrompt(_)
                    | PreStep::MolmoAct2(_)
            )
        });
        let guard = if templated {
            let zeros = vec![0.0; ir.state_len];
            let template = prompt_text(&manifest, "", &zeros, false)?;
            let varying = molmoact2
                .as_ref()
                .map(|h| super::molmoact2::state_tokens(h.num_state_tokens()))
                .unwrap_or_default();
            let g = PromptGuard::new(&template, tokenizer.as_mut(), &varying)?;
            if let Some(host) = &xr0 {
                let want = host.geo.views.len() * host.geo.patches.image_pads();
                if g.count("<|image_pad|>") != Some(want) {
                    return refuse(format!(
                        "the xr0 template tokenizes to {:?} image pads; {} cameras need {want}",
                        g.count("<|image_pad|>"),
                        host.geo.views.len()
                    ));
                }
            }
            if let Some(host) = &molmoact2 {
                let t = tokenizer.tokenize(&template)?;
                let got = t
                    .ids
                    .iter()
                    .zip(&t.mask)
                    .filter(|(&id, &m)| m && id == host.placeholder_id())
                    .count();
                if got != host.geo.cameras.len() {
                    return refuse(format!(
                        "the molmoact2 template tokenizes to {got} image placeholders (id {}); \
                         {} cameras need one each",
                        host.placeholder_id(),
                        host.geo.cameras.len()
                    ));
                }
            }
            if let Some(host) = &gr00t {
                let ids = tokenizer.tokenize(&template)?.ids;
                let got = ids.iter().filter(|&&t| t == host.image_token_id()).count();
                if got != host.geo.cameras {
                    return refuse(format!(
                        "the gr00t template tokenizes to {got} image tokens (id {}); {} cameras \
                         need one each",
                        host.image_token_id(),
                        host.geo.cameras
                    ));
                }
            }
            Some(g)
        } else {
            None
        };
        let (spec, layout) = derive_spec(&manifest, &ir, server)?;
        let flip = layout
            .names
            .iter()
            .map(|n| manifest.flipped_cameras().contains(n))
            .collect();
        Ok(Self {
            manifest,
            spec,
            layout,
            flip,
            binding,
            action_shape: ir.action_shape,
            model: Some(model),
            tokenizer: Some(tokenizer),
            vla_jepa,
            xr0,
            gr00t,
            molmoact2,
            prompt_checks: HashMap::new(),
            guard,
        })
    }

    /// Refuse a task whose prompt could overflow the prompt port, judged once
    /// per text with the worst-case state so the answer never changes mid-episode.
    fn check_prompt(&mut self, task: &str) -> EngineResult<()> {
        if let Some(answer) = self.prompt_checks.get(task) {
            return answer.clone().map_or(Ok(()), |r| Err(r.error()));
        }
        if let Some(g) = &self.guard {
            let templated = templated_task(&self.manifest, task);
            if let Some(m) = g
                .check_task(task)
                .or_else(|| templated.and_then(|t| g.check_task(&t)))
            {
                return self.remember(task, Some(Refusal::Structure(m)));
            }
        }
        let zeros = vec![0.0; self.spec.state_len];
        let text = prompt_text(&self.manifest, task, &zeros, true)?;
        let tok = self.tokenizer.as_mut().ok_or(EngineError::NotLoaded)?;
        let tokens = tok.tokenize(&text).map_err(refused_text)?;
        let count = tokens.mask.iter().filter(|&&m| m).count();
        let limit = self.binding.prompt_len;
        // Only the image-slot prompts (π0.5) carry the state; the others' counts are exact.
        let worst = if self.binding.layout != PortLayout::ImageSlots {
            ""
        } else {
            " with the worst-case state"
        };
        let refusal = if let Some(host) = &self.gr00t {
            // Unpadded ids: the count is exact. Counted as the language tokens the images leave.
            let images = host.geo.cameras;
            (count > limit).then(|| {
                format!(
                    "the prompt needs {} language tokens; the limit is {}",
                    count - images.min(count),
                    limit - images
                )
            })
        } else if tok.has_truncation_state() {
            (count > limit)
                .then(|| format!("the prompt needs {count} tokens{worst}; the limit is {limit}"))
        } else {
            // An always-on Truncate op: a full mask is the only sign of a cut.
            (count >= limit).then(|| {
                format!(
                    "the prompt fills all {limit} tokens{worst} and may be \
                     truncated; the limit is {} tokens",
                    limit.saturating_sub(1)
                )
            })
        };
        self.remember(task, refusal.map(Refusal::TooLong))
    }

    fn remember(&mut self, task: &str, refusal: Option<Refusal>) -> EngineResult<()> {
        if self.prompt_checks.len() >= PROMPT_CACHE_MAX {
            self.prompt_checks.clear();
        }
        self.prompt_checks.insert(task.to_string(), refusal.clone());
        refusal.map_or(Ok(()), |r| Err(r.error()))
    }

    /// Tokenize `text`; the ids must fill the prompt port exactly.
    fn tokens(&mut self, text: &str) -> EngineResult<Tokens> {
        let limit = self.binding.prompt_len;
        let tokens = self
            .tokenizer
            .as_mut()
            .ok_or(EngineError::NotLoaded)?
            .tokenize(text)
            .map_err(refused_text)?;
        if self.binding.layout == PortLayout::Gr00t {
            return Ok(tokens);
        }
        // Warmup's prompt filled the port, so a width that differs follows from this text.
        if tokens.ids.len() != limit || tokens.mask.len() != limit {
            return Err(EngineError::InvalidConfig(format!(
                "the tokenizer produced {} ids and {} mask values; the model's prompt port takes {limit}",
                tokens.ids.len(),
                tokens.mask.len()
            )));
        }
        Ok(tokens)
    }

    /// [`PortLayout::ImageSlots`] inputs, all but `noise`.
    fn image_slot_feeds(
        &mut self,
        obs: &Observation,
    ) -> EngineResult<Vec<(&'static str, HostTensor)>> {
        let raw: Vec<f64> = obs.state.iter().map(|&x| f64::from(x)).collect();
        let text = prompt_text(&self.manifest, &obs.prompt, &raw, false)?;
        let mut state = raw;
        for step in self.manifest.normalize() {
            state = normalize(step, STATE, &state)?;
            if let Some(n) = step.pad_to {
                state.resize(n, 0.0);
            }
        }
        let (images, image_masks) = images_tensor(
            &self.layout,
            &obs.images,
            self.spec.height,
            self.spec.width,
            &self.flip,
        )?;

        let limit = self.binding.prompt_len;
        let tokens = self.tokens(&text)?;

        let (h, w) = (self.spec.height as usize, self.spec.width as usize);
        let slots = self.layout.slots.len();
        Ok(vec![
            (
                STATE,
                HostTensor {
                    shape: vec![1, state.len()],
                    data: TensorData::F32(state.iter().map(|&x| x as f32).collect()),
                },
            ),
            (
                PROMPT,
                HostTensor {
                    shape: vec![1, limit],
                    data: TensorData::I64(tokens.ids),
                },
            ),
            (
                PROMPT_MASK,
                HostTensor {
                    shape: vec![1, limit],
                    data: TensorData::Bool(tokens.mask),
                },
            ),
            (
                IMAGES,
                HostTensor {
                    shape: vec![slots, 1, 3, h, w],
                    data: TensorData::F32(images),
                },
            ),
            (
                IMAGE_MASKS,
                HostTensor {
                    shape: vec![slots, 1],
                    data: TensorData::Bool(image_masks),
                },
            ),
        ])
    }

    /// The RTC group. Without RTC inputs the guidance weight is 0, which zeroes
    /// the graph's correction (`v_t - 0 * err`), so the chunk is the plain one.
    fn rtc_feeds(
        &self,
        obs: &Observation,
        shapes: &RtcShapes,
    ) -> EngineResult<Vec<(String, HostTensor)>> {
        let [steps, dim] = self.action_shape;
        let zeros = vec![0.0; steps * dim];
        let (left, delay, weight, horizon) = match &obs.rtc {
            Some(r) => (
                &r.prev_chunk_left_over,
                r.inference_delay,
                r.max_guidance_weight,
                r.execution_horizon,
            ),
            None => (&zeros, 0, 0.0, 1),
        };
        // Normalized with the action stats, as physicalai's `normalize` step does.
        let mut left: Vec<f64> = left.iter().map(|&x| f64::from(x)).collect();
        for step in self.manifest.normalize() {
            left = normalize(step, PREV_CHUNK_LEFT_OVER, &left)?;
        }
        if let Some(x) = left
            .iter()
            .find(|x| !x.is_finite() || x.abs() > RTC_LEFTOVER_NORMALIZED_MAX)
        {
            return Err(EngineError::InvalidConfig(format!(
                "prev_chunk_left_over normalizes to {x:e}; it must stay within \
                 {RTC_LEFTOVER_NORMALIZED_MAX:e} standard deviations of the action mean \
                 (send back the actions this server returned)"
            )));
        }
        let tensor = |shape: &[usize], data| HostTensor {
            shape: shape.to_vec(),
            data,
        };
        Ok(vec![
            (
                PREV_CHUNK_LEFT_OVER.into(),
                tensor(
                    &[1, steps, dim],
                    TensorData::F32(left.iter().map(|&x| x as f32).collect()),
                ),
            ),
            (
                INFERENCE_DELAY.into(),
                tensor(&shapes.inference_delay, TensorData::I64(vec![delay.into()])),
            ),
            (
                MAX_GUIDANCE_WEIGHT.into(),
                tensor(&shapes.max_guidance_weight, TensorData::F32(vec![weight])),
            ),
            (
                EXECUTION_HORIZON.into(),
                tensor(
                    &shapes.execution_horizon,
                    TensorData::I64(vec![horizon.into()]),
                ),
            ),
        ])
    }

    /// [`PortLayout::VlaJepa`] inputs, all but `noise`. The state goes in raw.
    fn vla_jepa_feeds(
        &mut self,
        obs: &Observation,
    ) -> EngineResult<Vec<(&'static str, HostTensor)>> {
        let text = prompt_text(&self.manifest, &obs.prompt, &[], false)?;
        let tokens = self.tokens(&text)?;
        if let Some(g) = &self.guard {
            g.check_ids(&tokens)?;
        }
        let host = self
            .vla_jepa
            .as_ref()
            .expect("new builds the host for a vlajepa layout");
        let positions = host.embodied_positions(&tokens.ids)?;
        let pixels = host.pixel_values(&self.layout, &obs.images)?;
        let limit = self.binding.prompt_len;
        let rows = pixels.len() / host.feature();
        Ok(vec![
            (
                INPUT_IDS,
                HostTensor {
                    shape: vec![1, limit],
                    data: TensorData::I64(tokens.ids),
                },
            ),
            (
                ATTENTION_MASK,
                HostTensor {
                    shape: vec![1, limit],
                    data: TensorData::I64(tokens.mask.iter().map(|&m| i64::from(m)).collect()),
                },
            ),
            (
                PIXEL_VALUES,
                HostTensor {
                    shape: vec![rows, host.feature()],
                    data: TensorData::F32(pixels),
                },
            ),
            (
                EMBODIED_POSITIONS,
                HostTensor {
                    shape: vec![1, positions.len()],
                    data: TensorData::I64(positions),
                },
            ),
            (
                STATE,
                HostTensor {
                    shape: vec![1, obs.state.len()],
                    data: TensorData::F32(obs.state.clone()),
                },
            ),
        ])
    }

    /// [`PortLayout::Xr0`] inputs, all but `noise`.
    fn xr0_feeds(&mut self, obs: &Observation) -> EngineResult<Vec<(&'static str, HostTensor)>> {
        let text = prompt_text(&self.manifest, &obs.prompt, &[], false)?;
        let tokens = self.tokens(&text)?;
        if let Some(g) = &self.guard {
            g.check_ids(&tokens)?;
        }
        let host = self
            .xr0
            .as_ref()
            .expect("new builds the host for an xr0 layout");
        let pixels = host.pixel_values(&self.layout, &obs.images)?;
        let feature = host.geo.patches.feature();
        let limit = self.binding.prompt_len;
        let state = host.state(&obs.state);
        Ok(vec![
            (
                PROMPT,
                HostTensor {
                    shape: vec![1, limit],
                    data: TensorData::I64(tokens.ids),
                },
            ),
            (
                PROMPT_MASK,
                HostTensor {
                    shape: vec![1, limit],
                    data: TensorData::I64(tokens.mask.iter().map(|&m| i64::from(m)).collect()),
                },
            ),
            (
                PIXEL_VALUES,
                HostTensor {
                    shape: vec![pixels.len() / feature, feature],
                    data: TensorData::F32(pixels),
                },
            ),
            (
                STATE,
                HostTensor {
                    shape: vec![1, host.geo.history, host.max_state_dim()],
                    data: TensorData::F32(state),
                },
            ),
        ])
    }

    /// [`PortLayout::Gr00t`] inputs, all but the noise.
    fn gr00t_feeds(&mut self, obs: &Observation) -> EngineResult<Vec<(String, HostTensor)>> {
        let text = prompt_text(&self.manifest, &obs.prompt, &[], false)?;
        let tokens = self.tokens(&text)?;
        if let Some(g) = &self.guard {
            g.check_ids(&tokens)?;
        }
        let host = self
            .gr00t
            .as_ref()
            .expect("new builds the host for a gr00t layout");
        let ids: Vec<i64> = tokens
            .ids
            .iter()
            .zip(&tokens.mask)
            .filter(|(_, &m)| m)
            .map(|(&id, _)| id)
            .collect();
        let (input_ids, mask, position_ids) = host.inputs(&ids)?;
        let pixels = host.pixel_values(&self.layout, &obs.images, &self.flip)?;
        let mut state: Vec<f64> = obs.state.iter().map(|&x| f64::from(x)).collect();
        for step in self.manifest.normalize() {
            state = normalize(step, STATE, &state)?;
        }
        state.resize(host.geo.state_width, 0.0);
        let seq = host.seq_len();
        let feature = host.geo.patches.feature();
        let mut feeds = vec![
            (
                INPUT_IDS.to_string(),
                HostTensor {
                    shape: vec![1, seq],
                    data: TensorData::I64(input_ids),
                },
            ),
            (
                ATTENTION_MASK.to_string(),
                HostTensor {
                    shape: vec![1, seq],
                    data: TensorData::I64(mask),
                },
            ),
            (
                PIXEL_VALUES.to_string(),
                HostTensor {
                    shape: vec![pixels.len() / feature, feature],
                    data: TensorData::F32(pixels),
                },
            ),
            (
                POSITION_IDS.to_string(),
                HostTensor {
                    shape: vec![3, 1, seq],
                    data: TensorData::I64(position_ids),
                },
            ),
            (
                STATE.to_string(),
                HostTensor {
                    shape: vec![1, 1, state.len()],
                    data: TensorData::F32(state.iter().map(|&x| x as f32).collect()),
                },
            ),
        ];
        for (port, values) in self.manifest.constants() {
            feeds.push((
                port.to_string(),
                HostTensor {
                    shape: vec![values.len()],
                    data: TensorData::I64(values.to_vec()),
                },
            ));
        }
        Ok(feeds)
    }

    /// [`PortLayout::MolmoAct2`] inputs; the state is in the prompt.
    fn molmoact2_feeds(
        &mut self,
        obs: &Observation,
    ) -> EngineResult<Vec<(&'static str, HostTensor)>> {
        let raw: Vec<f64> = obs.state.iter().map(|&x| f64::from(x)).collect();
        let text = prompt_text(&self.manifest, &obs.prompt, &raw, false)?;
        let tokens = self.tokens(&text)?;
        if let Some(g) = &self.guard {
            g.check_ids(&tokens)?;
        }
        let host = self
            .molmoact2
            .as_ref()
            .expect("new builds the host for a molmoact2 layout");
        let (ids, mask, types) = host.inputs(&tokens.ids, &tokens.mask)?;
        let seq = host.seq_len(self.binding.prompt_len);
        if ids.len() != seq {
            return Err(EngineError::InvalidConfig(format!(
                "the prompt expands to {} tokens; the model takes exactly {seq}",
                ids.len()
            )));
        }
        let images = host.images(&self.layout, &obs.images)?;
        let pooling = host.token_pooling();
        let cams = host.geo.cameras.len();
        let cells = pooling.len() / (cams * host.geo.pooled_per_image());
        let pad = host.action_dim_is_pad();
        Ok(vec![
            (
                INPUT_IDS,
                HostTensor {
                    shape: vec![1, seq],
                    data: TensorData::I64(ids),
                },
            ),
            (
                ATTENTION_MASK,
                HostTensor {
                    shape: vec![1, seq],
                    data: TensorData::I64(mask),
                },
            ),
            (
                TOKEN_TYPE_IDS,
                HostTensor {
                    shape: vec![1, seq],
                    data: TensorData::I64(types),
                },
            ),
            (
                IMAGES,
                HostTensor {
                    shape: vec![1, cams, host.geo.patches_per_image(), host.geo.patch_pixels],
                    data: TensorData::F32(images),
                },
            ),
            (
                TOKEN_POOLING,
                HostTensor {
                    shape: vec![1, cams * host.geo.pooled_per_image(), cells],
                    data: TensorData::I64(pooling),
                },
            ),
            (
                ACTION_DIM_IS_PAD,
                HostTensor {
                    shape: vec![1, pad.len()],
                    data: TensorData::Bool(pad),
                },
            ),
        ])
    }

    fn run(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        let start = Instant::now();
        if self.model.is_none() {
            return Err(EngineError::NotLoaded);
        }
        check_observation(&self.spec, obs)?;
        self.check_prompt(&obs.prompt)?;

        let named = |f: Vec<(&'static str, HostTensor)>| -> Vec<(String, HostTensor)> {
            f.into_iter().map(|(n, t)| (n.to_string(), t)).collect()
        };
        let mut feeds = match self.binding.layout {
            PortLayout::ImageSlots => named(self.image_slot_feeds(obs)?),
            PortLayout::VlaJepa => named(self.vla_jepa_feeds(obs)?),
            PortLayout::Xr0 => named(self.xr0_feeds(obs)?),
            PortLayout::Gr00t => self.gr00t_feeds(obs)?,
            PortLayout::MolmoAct2 => named(self.molmoact2_feeds(obs)?),
        };
        if self.binding.noise {
            let dims = self
                .spec
                .noise_shape
                .clone()
                .expect("derive_spec sets noise_shape for a noise input");
            let n: usize = dims.iter().product();
            let noise = match (&obs.noise, obs.seed) {
                (Some(v), _) => v.clone(),
                (None, Some(seed)) => NoiseRng::seeded(seed).normal(n),
                (None, None) => NoiseRng::fresh().normal(n),
            };
            let mut shape = vec![1];
            shape.extend(dims);
            feeds.push((
                self.binding.noise_port.clone(),
                HostTensor {
                    shape,
                    data: TensorData::F32(noise),
                },
            ));
        }
        if let Some(shapes) = &self.binding.rtc {
            feeds.extend(self.rtc_feeds(obs, shapes)?);
        }

        let model = self.model.as_mut().ok_or(EngineError::NotLoaded)?;
        for (name, t) in &feeds {
            model.set_input(self.binding.port_name(name), t)?;
        }
        model.infer()?;
        let (shape, out) = model.output_f32(&self.binding.action_port)?;
        let [steps, dim] = self.action_shape;
        if shape != [1, steps, dim] {
            return Err(EngineError::Backend(format!(
                "the action output is {shape:?}; expected [1, {steps}, {dim}]"
            )));
        }
        // Warmup proved the model answers finite actions, so a non-finite one
        // follows from this request's inputs: refused, not an engine fault.
        if out.iter().any(|x| !x.is_finite()) {
            return Err(EngineError::InvalidConfig(
                "the model returned a non-finite action; refusing to send it".into(),
            ));
        }

        let mut actions: Vec<f64> = out.iter().map(|&x| f64::from(x)).collect();
        let (mut steps, mut dim) = (steps, dim);
        for step in &self.manifest.postprocessors {
            match step {
                PostStep::Denormalize(s) => actions = denormalize(s, ACTION, &actions)?,
                PostStep::ActionChunkTrimmer { n_action_steps } => {
                    (actions, steps) = trim_actions(actions, steps, dim, *n_action_steps);
                }
                PostStep::XvlaRotation6dToAxisAngle { .. } => {
                    actions = xvla_rotation_6d_to_axis_angle(&actions, steps, dim);
                    dim = XVLA_ACTION_DIM;
                }
                PostStep::GripperBinarize {
                    index,
                    threshold,
                    above,
                    below,
                } => gripper_binarize(&mut actions, dim, *index, *threshold, *above, *below),
                PostStep::Xr0Denormalize(s) => {
                    let f: Vec<f32> = actions.iter().map(|&x| x as f32).collect();
                    let out = super::xr0::denormalize(s, &f, dim);
                    dim = s.action_dim.unwrap_or(dim);
                    actions = out.into_iter().map(f64::from).collect();
                }
                PostStep::MolmoAct2Post(q) => {
                    let f: Vec<f32> = actions.iter().map(|&x| x as f32).collect();
                    actions = super::molmoact2::postprocess(q, &f)
                        .into_iter()
                        .map(f64::from)
                        .collect();
                }
                PostStep::ActionSlice { dims } => {
                    actions = actions
                        .chunks(dim)
                        .flat_map(|row| row[..*dims].iter().copied())
                        .collect();
                    dim = *dims;
                }
            }
        }
        let actions: Vec<f32> = actions.iter().map(|&x| x as f32).collect();
        if actions.iter().any(|x| !x.is_finite()) {
            return Err(EngineError::InvalidConfig(
                "denormalized actions overflow f32; refusing to send them".into(),
            ));
        }
        Ok(ActionChunk {
            actions,
            steps,
            dim,
            infer_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
}

impl PolicyPipeline {
    /// Record which package bytes and precision this pipeline serves.
    pub fn set_identity(
        &mut self,
        manifest_sha256: String,
        package_sha256: String,
        precision: &str,
    ) {
        self.spec.manifest_sha256 = manifest_sha256;
        self.spec.package_sha256 = package_sha256;
        self.spec.precision = precision.to_string();
    }

    /// The model's [`PolicyModel::gpu_memory_bytes`]; `None` once closed.
    pub fn gpu_memory_bytes(&self) -> Option<u64> {
        self.model.as_ref()?.gpu_memory_bytes()
    }
}

impl PolicyEngine for PolicyPipeline {
    fn spec(&self) -> &PolicySpec {
        &self.spec
    }

    fn warmup(&mut self) -> EngineResult<Vec<u32>> {
        let frame = Image {
            height: self.spec.height,
            width: self.spec.width,
            data: vec![0; self.spec.height as usize * self.spec.width as usize * 3],
        };
        let obs = Observation {
            images: self
                .spec
                .cameras
                .iter()
                .flat_map(|c| std::iter::repeat_n((c.clone(), frame.clone()), self.spec.history))
                .collect(),
            state: vec![WARMUP_STATE; self.spec.state_len * self.spec.history],
            prompt: String::new(),
            seed: Some(0),
            noise: None,
            rtc: None,
        };
        (0..WARMUP_CALLS)
            .map(|_| Ok(self.run(&obs)?.infer_ms.ceil() as u32))
            .collect()
    }

    fn infer(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        self.run(obs)
    }

    fn close(&mut self) {
        self.model = None;
        self.tokenizer = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::manifest::LoadOptions;
    use std::sync::{Arc, Mutex};

    const PI05_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/intel-pi05-libero-manifest.json");
    const SMOLVLA_NOISE: &str =
        include_str!("../../tests/fixtures/policy/smolvla-libero-noise-manifest.json");
    const PI0_LIBERO: &str = include_str!("../../tests/fixtures/policy/pi0-libero-manifest.json");
    const XVLA_LIBERO: &str = include_str!("../../tests/fixtures/policy/xvla-libero-manifest.json");
    const SMOLVLA_RTC: &str =
        include_str!("../../tests/fixtures/policy/smolvla-libero-rtc-manifest.json");

    #[derive(Default)]
    struct Log {
        inputs: Vec<(String, HostTensor)>,
        infers: usize,
        fail_infer: bool,
        /// The tokenizer fails, or returns one id too many.
        fail_tokenize: bool,
        extra_token: bool,
        texts: Vec<String>,
    }

    struct FakeModel {
        log: Arc<Mutex<Log>>,
        output: (Vec<usize>, Vec<f32>),
        action: &'static str,
    }

    impl PolicyModel for FakeModel {
        fn set_input(&mut self, name: &str, t: &HostTensor) -> EngineResult<()> {
            self.log
                .lock()
                .unwrap()
                .inputs
                .push((name.into(), t.clone()));
            Ok(())
        }
        fn infer(&mut self) -> EngineResult<()> {
            let mut log = self.log.lock().unwrap();
            log.infers += 1;
            if log.fail_infer {
                return Err(EngineError::Backend("fake infer failed".into()));
            }
            Ok(())
        }
        fn output_f32(&mut self, name: &str) -> EngineResult<(Vec<usize>, Vec<f32>)> {
            assert_eq!(name, self.action);
            Ok(self.output.clone())
        }
    }

    /// One token per byte, padded right with id 0.
    struct FakeTokenizer {
        log: Arc<Mutex<Log>>,
        /// A `truncation` state kept off; otherwise an always-on Truncate.
        truncation_state: bool,
        port: usize,
    }

    /// The fake's special tokens; the `<|…|>` ones are one id each (1000 + index).
    const FAKE_SPECIAL: [&str; 7] = [
        "\u{1}",
        "<|image_pad|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image|>",
    ];

    fn fake_ids(text: &str) -> Vec<i64> {
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (i, sp) in FAKE_SPECIAL.iter().enumerate().skip(1) {
                if let Some(r) = rest.strip_prefix(sp) {
                    ids.push(1000 + i as i64);
                    rest = r;
                    continue 'outer;
                }
            }
            let c = rest.chars().next().unwrap();
            ids.extend(rest[..c.len_utf8()].bytes().map(i64::from));
            rest = &rest[c.len_utf8()..];
        }
        ids
    }

    impl PromptTokenizer for FakeTokenizer {
        fn has_truncation_state(&self) -> bool {
            self.truncation_state
        }
        fn special_tokens(&self) -> Vec<String> {
            FAKE_SPECIAL.iter().map(|s| s.to_string()).collect()
        }
        fn tokenize(&mut self, text: &str) -> EngineResult<Tokens> {
            let mut log = self.log.lock().unwrap();
            log.texts.push(text.into());
            if log.fail_tokenize {
                return Err(EngineError::Backend("fake tokenizer failed".into()));
            }
            let extra = log.extra_token;
            drop(log);
            let mut ids: Vec<i64> = fake_ids(text);
            if !self.truncation_state {
                ids.truncate(self.port);
            }
            let count = ids.len();
            let width = count.max(self.port);
            ids.resize(width, 0);
            let mut mask: Vec<bool> = (0..width).map(|i| i < count).collect();
            if extra {
                ids.push(0);
                mask.push(false);
            }
            Ok(Tokens { ids, mask })
        }
    }

    fn port(name: &str, dtype: DType, dims: &[usize]) -> PortInfo {
        PortInfo {
            aliases: vec![name.into()],
            dtype: Some(dtype),
            dims: dims.iter().map(|&d| Some(d)).collect(),
        }
    }

    /// The real ports of our smolvla_libero noise export.
    fn smolvla_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(STATE, DType::F32, &[1, 8]),
                port(NOISE, DType::F32, &[1, 50, 32]),
                port(PROMPT, DType::I64, &[1, 48]),
                port(PROMPT_MASK, DType::Bool, &[1, 48]),
                port(IMAGES, DType::F32, &[2, 1, 3, 512, 512]),
                port(IMAGE_MASKS, DType::Bool, &[2, 1]),
            ],
            vec![port(ACTION, DType::F32, &[1, 50, 7])],
        )
    }

    /// The real ports of our smolvla_libero RTC export: the noise export's plus
    /// the RTC group, scalars of rank 0.
    fn smolvla_rtc_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        let (mut inputs, outputs) = smolvla_ports();
        inputs.extend([
            port(PREV_CHUNK_LEFT_OVER, DType::F32, &[1, 50, 7]),
            port(INFERENCE_DELAY, DType::I64, &[]),
            port(MAX_GUIDANCE_WEIGHT, DType::F32, &[]),
            port(EXECUTION_HORIZON, DType::I64, &[]),
        ]);
        (inputs, outputs)
    }

    fn smolvla_rtc() -> Rig {
        rig_with(
            SMOLVLA_RTC,
            smolvla_rtc_ports(),
            false,
            true,
            vec![0.5; 350],
        )
    }

    fn pi05_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(STATE, DType::F32, &[1, 8]),
                port(PROMPT, DType::I64, &[1, 64]),
                port(PROMPT_MASK, DType::Bool, &[1, 64]),
                port(IMAGES, DType::F32, &[3, 1, 3, 224, 224]),
                port(IMAGE_MASKS, DType::Bool, &[3, 1]),
            ],
            vec![port(ACTION, DType::F32, &[1, 50, 7])],
        )
    }

    /// The real ports of our pi0_libero export.
    fn pi0_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(STATE, DType::F32, &[1, 8]),
                port(NOISE, DType::F32, &[1, 50, 32]),
                port(PROMPT, DType::I64, &[1, 48]),
                port(PROMPT_MASK, DType::Bool, &[1, 48]),
                port(IMAGES, DType::F32, &[3, 1, 3, 224, 224]),
                port(IMAGE_MASKS, DType::Bool, &[3, 1]),
            ],
            vec![port(ACTION, DType::F32, &[1, 50, 7])],
        )
    }

    /// The real ports of our xvla-libero export.
    fn xvla_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(STATE, DType::F32, &[1, 8]),
                port(PROMPT, DType::I64, &[1, 50]),
                port(PROMPT_MASK, DType::Bool, &[1, 50]),
                port(IMAGES, DType::F32, &[2, 1, 3, 224, 224]),
                port(IMAGE_MASKS, DType::Bool, &[2, 1]),
                port(NOISE, DType::F32, &[1, 30, 20]),
            ],
            vec![port(ACTION, DType::F32, &[1, 30, 20])],
        )
    }

    fn server(device: &str) -> ServerFacts {
        ServerFacts {
            model: "pkg".into(),
            device: device.into(),
            prompt_max_bytes: PROMPT_MAX_BYTES,
        }
    }

    struct Rig {
        engine: PolicyPipeline,
        log: Arc<Mutex<Log>>,
    }

    fn rig(manifest: &str, pi05: bool, truncation_state: bool, output: Vec<f32>) -> Rig {
        let ports = if pi05 { pi05_ports() } else { smolvla_ports() };
        rig_with(manifest, ports, pi05, truncation_state, output)
    }

    /// A rig on any ports; the fake model returns `output` in the action port's shape.
    fn rig_with(
        manifest: &str,
        (inputs, outputs): (Vec<PortInfo>, Vec<PortInfo>),
        has_random_uniform: bool,
        truncation_state: bool,
        output: Vec<f32>,
    ) -> Rig {
        let log = Arc::new(Mutex::new(Log::default()));
        let ids = if super::port(&inputs, PROMPT).is_some() {
            PROMPT
        } else {
            INPUT_IDS
        };
        let port_len = match super::port(&inputs, TOKEN_POOLING) {
            // MolmoAct2's tokenizer is the sequence less BOS and two images' tokens.
            Some(_) => port_dims(&inputs, ids)[1] - 1 - 2 * 197,
            None => port_dims(&inputs, ids)[1],
        };
        let engine = pipeline_with(
            manifest,
            (&inputs, &outputs),
            has_random_uniform,
            truncation_state,
            port_len,
            &log,
            output,
        )
        .unwrap();
        // Drop the load-time width probe so tests see only infer texts.
        log.lock().unwrap().texts.clear();
        Rig { engine, log }
    }

    fn port_dims(ports: &[PortInfo], name: &str) -> Vec<usize> {
        super::port(ports, name)
            .unwrap()
            .dims
            .iter()
            .map(|d| d.unwrap())
            .collect()
    }

    fn pipeline(
        manifest: &str,
        pi05: bool,
        truncation_state: bool,
        tokenizer_port: usize,
        log: &Arc<Mutex<Log>>,
        output: Vec<f32>,
    ) -> EngineResult<PolicyPipeline> {
        let (inputs, outputs) = if pi05 { pi05_ports() } else { smolvla_ports() };
        pipeline_with(
            manifest,
            (&inputs, &outputs),
            pi05,
            truncation_state,
            tokenizer_port,
            log,
            output,
        )
    }

    fn pipeline_with(
        manifest: &str,
        (inputs, outputs): (&[PortInfo], &[PortInfo]),
        has_random_uniform: bool,
        truncation_state: bool,
        tokenizer_port: usize,
        log: &Arc<Mutex<Log>>,
        output: Vec<f32>,
    ) -> EngineResult<PolicyPipeline> {
        let m = PolicyManifest::parse(manifest, LoadOptions::default()).unwrap();
        // MolmoAct2 on CPU is refused on ARM hosts; its rigs stand in for the GPU.
        let device = if m.policy == PolicyKind::MolmoAct2 {
            "GPU"
        } else {
            "CPU"
        };
        PolicyPipeline::new(
            m,
            server(device),
            Box::new(FakeModel {
                log: log.clone(),
                output: (port_dims(outputs, ACTION), output),
                action: ACTION,
            }),
            inputs,
            outputs,
            has_random_uniform,
            Box::new(FakeTokenizer {
                log: log.clone(),
                truncation_state,
                port: tokenizer_port,
            }),
        )
    }

    fn smolvla(truncation_state: bool) -> Rig {
        rig(SMOLVLA_NOISE, false, truncation_state, vec![0.5; 350])
    }

    fn pi05(truncation_state: bool) -> Rig {
        rig(PI05_LIBERO, true, truncation_state, vec![0.5; 350])
    }

    fn frame(side: u32, value: u8) -> Image {
        Image {
            height: side,
            width: side,
            data: vec![value; side as usize * side as usize * 3],
        }
    }

    fn obs(spec: &PolicySpec, prompt: &str) -> Observation {
        // Sent in reverse order: slot order must come from the package.
        let mut images: Vec<(String, Image)> = spec
            .cameras
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), frame(spec.height, 10 * (i as u8 + 1))))
            .collect();
        images.reverse();
        Observation {
            images,
            state: (0..spec.state_len).map(|i| i as f32 * 0.01).collect(),
            prompt: prompt.into(),
            seed: None,
            noise: None,
            rtc: None,
        }
    }

    fn input(log: &Arc<Mutex<Log>>, name: &str) -> HostTensor {
        log.lock()
            .unwrap()
            .inputs
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| panic!("{name} was not set"))
    }

    fn f32s(t: &HostTensor) -> &[f32] {
        match &t.data {
            TensorData::F32(v) => v,
            other => panic!("not f32: {other:?}"),
        }
    }

    fn i64s(t: &HostTensor) -> &[i64] {
        match &t.data {
            TensorData::I64(v) => v,
            other => panic!("not i64: {other:?}"),
        }
    }

    /// `(x - mean) / (std + eps)` with the package's action stats.
    fn normalized_action(m: &PolicyManifest, raw: &[f32]) -> Vec<f32> {
        let s = m
            .normalize()
            .find_map(|s| s.stats_for(PREV_CHUNK_LEFT_OVER))
            .unwrap();
        let (mean, std) = (s.mean.as_ref().unwrap(), s.std.as_ref().unwrap());
        raw.iter()
            .enumerate()
            .map(|(i, &x)| ((f64::from(x) - mean[i % 7]) / (std[i % 7] + 1e-8)) as f32)
            .collect()
    }

    #[test]
    fn rtc_package_serves_the_full_chunk_and_advertises_rtc() {
        let r = smolvla_rtc();
        let spec = r.engine.spec();
        assert_eq!(spec.chunk, [50, 7]);
        assert_eq!(spec.rtc_chunk_size, Some(50));
        // Packages without the RTC ports advertise none.
        assert_eq!(smolvla(true).engine.spec().rtc_chunk_size, None);
    }

    #[test]
    fn rtc_inputs_are_fed_normalized_and_typed_as_the_ports() {
        let mut r = smolvla_rtc();
        let spec = r.engine.spec().clone();
        let m = PolicyManifest::parse(SMOLVLA_RTC, LoadOptions::default()).unwrap();
        let mut o = obs(&spec, "pick up the bowl");
        let left: Vec<f32> = (0..350).map(|i| i as f32 * 0.001 - 0.1).collect();
        o.rtc = Some(cascadia_types::RtcInputs {
            prev_chunk_left_over: left.clone(),
            inference_delay: 3,
            max_guidance_weight: 5.0,
            execution_horizon: 10,
        });
        let chunk = r.engine.infer(&o).unwrap();
        assert_eq!((chunk.steps, chunk.dim), (50, 7));
        let t = input(&r.log, PREV_CHUNK_LEFT_OVER);
        assert_eq!(t.shape, vec![1, 50, 7]);
        let want = normalized_action(&m, &left);
        for (a, b) in f32s(&t).iter().zip(&want) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
        let delay = input(&r.log, INFERENCE_DELAY);
        assert_eq!((delay.shape.clone(), i64s(&delay)), (vec![], &[3i64][..]));
        let w = input(&r.log, MAX_GUIDANCE_WEIGHT);
        assert_eq!((w.shape.clone(), f32s(&w)), (vec![], &[5.0f32][..]));
        let h = input(&r.log, EXECUTION_HORIZON);
        assert_eq!(i64s(&h), &[10]);

        // Without RTC fields the guidance is off: weight 0 makes the graph's
        // correction exactly zero, so the chunk is the plain one.
        o.rtc = None;
        r.engine.infer(&o).unwrap();
        assert_eq!(f32s(&input(&r.log, MAX_GUIDANCE_WEIGHT)), &[0.0]);
        assert_eq!(i64s(&input(&r.log, INFERENCE_DELAY)), &[0]);
        assert_eq!(i64s(&input(&r.log, EXECUTION_HORIZON)), &[1]);
        let zeros = normalized_action(&m, &[0.0; 350]);
        assert_eq!(f32s(&input(&r.log, PREV_CHUNK_LEFT_OVER)), zeros.as_slice());
    }

    /// A finite leftover whose normalized value is huge would make the graph's
    /// f32 (or GPU f16) arithmetic overflow into a NaN chunk; it is refused as
    /// a bad request before the model runs.
    #[test]
    fn rtc_leftover_that_normalizes_out_of_range_is_refused() {
        let mut r = smolvla_rtc();
        let spec = r.engine.spec().clone();
        let m = PolicyManifest::parse(SMOLVLA_RTC, LoadOptions::default()).unwrap();
        let s = m
            .normalize()
            .find_map(|s| s.stats_for(PREV_CHUNK_LEFT_OVER))
            .unwrap()
            .clone();
        let (mean, std) = (s.mean.unwrap(), s.std.unwrap());
        let at = |sigmas: f64| -> Vec<f32> {
            (0..350)
                .map(|i| (mean[i % 7] + sigmas * (std[i % 7] + 1e-8)) as f32)
                .collect()
        };
        let call = |r: &mut Rig, left: Vec<f32>| {
            let mut o = obs(&spec, "pick up the bowl");
            o.rtc = Some(cascadia_types::RtcInputs {
                prev_chunk_left_over: left,
                inference_delay: 1,
                max_guidance_weight: 5.0,
                execution_horizon: 10,
            });
            r.engine.infer(&o)
        };
        let infers = r.log.lock().unwrap().infers;
        for left in [vec![3e38; 350], at(70_000.0)] {
            let e = call(&mut r, left).unwrap_err();
            assert!(matches!(e, EngineError::InvalidConfig(_)), "{e}");
            assert!(
                e.to_string().contains("prev_chunk_left_over normalizes to"),
                "{e}"
            );
        }
        assert_eq!(r.log.lock().unwrap().infers, infers, "the model ran");
        // Far but plausible values pass.
        call(&mut r, at(500.0)).unwrap();
    }

    /// The leftover's stats are only read for a model with the RTC ports: a
    /// plain package loads exactly as before, even with a bad entry.
    #[test]
    fn rtc_leftover_stats_only_matter_with_the_rtc_ports() {
        let mut v: serde_json::Value = serde_json::from_str(SMOLVLA_RTC).unwrap();
        for step in v["model"]["preprocessors"].as_array_mut().unwrap() {
            if let Some(stats) = step.get_mut("stats").and_then(|v| v.as_object_mut()) {
                stats.insert(
                    PREV_CHUNK_LEFT_OVER.into(),
                    serde_json::json!({"mean": [0.0], "std": "bad"}),
                );
            }
        }
        let m = PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap();
        assert!(m.normalize().all(|s| s.features.len() == 1));
        let log = Arc::new(Mutex::new(Log::default()));
        let (inputs, outputs) = smolvla_ports();
        pipeline_with(
            &v.to_string(),
            (&inputs, &outputs),
            false,
            true,
            48,
            &log,
            vec![],
        )
        .unwrap();
        let (inputs, outputs) = smolvla_rtc_ports();
        let e = pipeline_with(
            &v.to_string(),
            (&inputs, &outputs),
            false,
            true,
            48,
            &log,
            vec![],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("prev_chunk_left_over"), "{e}");
    }

    #[test]
    fn rtc_ports_are_all_or_none_and_need_the_full_chunk() {
        let log = Arc::new(Mutex::new(Log::default()));
        let (mut inputs, outputs) = smolvla_rtc_ports();
        inputs.retain(|p| p.aliases[0] != EXECUTION_HORIZON);
        let e = pipeline_with(
            SMOLVLA_RTC,
            (&inputs, &outputs),
            false,
            true,
            48,
            &log,
            vec![],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("RTC ports come as a group"), "{e}");

        let (mut inputs, outputs) = smolvla_rtc_ports();
        inputs[6] = port(PREV_CHUNK_LEFT_OVER, DType::F32, &[1, 40, 7]);
        let e = pipeline_with(
            SMOLVLA_RTC,
            (&inputs, &outputs),
            false,
            true,
            48,
            &log,
            vec![],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("prev_chunk_left_over is [1, 40, 7]"), "{e}");

        // A trimmer would serve part of the chunk the leftover indexes.
        let mut v: serde_json::Value = serde_json::from_str(SMOLVLA_RTC).unwrap();
        v["model"]["postprocessors"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"type": "action_chunk_trimmer", "n_action_steps": 10}));
        let (inputs, outputs) = smolvla_rtc_ports();
        let e = pipeline_with(
            &v.to_string(),
            (&inputs, &outputs),
            false,
            true,
            48,
            &log,
            vec![],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("RTC package must serve the whole chunk"), "{e}");

        // The leftover is normalized like the actions; a package without its stats is refused.
        let mut v: serde_json::Value = serde_json::from_str(SMOLVLA_RTC).unwrap();
        for step in v["model"]["preprocessors"].as_array_mut().unwrap() {
            if let Some(stats) = step.get_mut("stats").and_then(|v| v.as_object_mut()) {
                stats.remove(PREV_CHUNK_LEFT_OVER);
            }
        }
        let e = pipeline_with(
            &v.to_string(),
            (&inputs, &outputs),
            false,
            true,
            48,
            &log,
            vec![],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            e.contains("no normalize stats for prev_chunk_left_over"),
            "{e}"
        );
    }

    #[test]
    fn smolvla_inputs_match_the_ir_ports_and_camera_slots() {
        let mut r = smolvla(true);
        let spec = r.engine.spec().clone();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        r.engine.infer(&obs(&spec, "pick up the bowl")).unwrap();
        let (inputs, _) = smolvla_ports();
        for p in &inputs {
            let t = input(&r.log, &p.aliases[0]);
            let want: Vec<usize> = p.dims.iter().map(|d| d.unwrap()).collect();
            assert_eq!(t.shape, want, "{}", p.aliases[0]);
            assert_eq!(Some(t.dtype()), p.dtype, "{}", p.aliases[0]);
            assert_eq!(
                t.to_bytes().len(),
                want.iter().product::<usize>() * t.dtype().bytes_per_element()
            );
        }
        let images = input(&r.log, IMAGES);
        let per = 3 * 512 * 512;
        let px = |v: u8| v as f32 / 255.0 * 2.0 - 1.0;
        // "image" (value 10) in slot 0 and "image2" (20) in slot 1, whatever the request order.
        assert_eq!(f32s(&images)[0], px(10));
        assert_eq!(f32s(&images)[per], px(20));
        assert_eq!(
            input(&r.log, IMAGE_MASKS).data,
            TensorData::Bool(vec![true, true])
        );
        // new_line runs before the tokenizer.
        assert_eq!(
            r.log.lock().unwrap().texts.last().unwrap(),
            "pick up the bowl\n"
        );
        // The state reaches the model normalized: (x - mean) / (std + eps).
        let m = PolicyManifest::parse(SMOLVLA_NOISE, LoadOptions::default()).unwrap();
        let s = m
            .normalize()
            .next()
            .unwrap()
            .stats_for(STATE)
            .unwrap()
            .clone();
        let (mean, std) = (s.mean.unwrap(), s.std.unwrap());
        let want = ((0.01f32 as f64) - mean[1]) / (std[1] + 1e-8);
        assert_eq!(f32s(&input(&r.log, STATE))[1], want as f32);
    }

    #[test]
    fn pi05_fills_the_dummy_slot_and_builds_the_prompt_from_bins() {
        let mut r = pi05(true);
        let spec = r.engine.spec().clone();
        assert_eq!(spec.noise, NoiseMode::InGraphRandom);
        r.engine.infer(&obs(&spec, " pick_up\n")).unwrap();
        let images = input(&r.log, IMAGES);
        assert_eq!(images.shape, vec![3, 1, 3, 224, 224]);
        let per = 3 * 224 * 224;
        assert!(f32s(&images)[2 * per..].iter().all(|&x| x == -1.0));
        assert_eq!(
            input(&r.log, IMAGE_MASKS).data,
            TensorData::Bool(vec![true, true, false])
        );
        let texts = r.log.lock().unwrap().texts.clone();
        assert_eq!(
            texts[0], "Task: pick up, State: 255 255 255 255 255 255 255 255;\nAction: ",
            "the length check uses the worst-case bins"
        );
        let m = PolicyManifest::parse(PI05_LIBERO, LoadOptions::default()).unwrap();
        let raw: Vec<f64> = obs(&spec, "").state.iter().map(|&x| f64::from(x)).collect();
        let norm = normalize(m.normalize().next().unwrap(), STATE, &raw).unwrap();
        assert_eq!(texts[1], pi05_prompt("pick up", &pi05_bins(&norm)));
        assert_ne!(texts[0], texts[1]);
        // No noise port, so nothing is drawn.
        assert!(!r.log.lock().unwrap().inputs.iter().any(|(n, _)| n == NOISE));
    }

    #[test]
    fn pi05_pad_to_bins_the_robot_width_and_pads_the_port() {
        // DROID: openpi bins the 8 real values, then pads the state to the model's 32.
        let mut v: serde_json::Value = serde_json::from_str(PI05_LIBERO).unwrap();
        v["model"]["preprocessors"][0]["pad_to"] = serde_json::json!(32);
        let (mut inputs, outputs) = pi05_ports();
        inputs[0] = port(STATE, DType::F32, &[1, 32]);
        let mut r = rig_with(
            &v.to_string(),
            (inputs, outputs),
            true,
            true,
            vec![0.5; 350],
        );
        let spec = r.engine.spec().clone();
        assert_eq!(spec.state_len, 8);
        r.engine.infer(&obs(&spec, "pick up")).unwrap();
        let m = PolicyManifest::parse(PI05_LIBERO, LoadOptions::default()).unwrap();
        let raw: Vec<f64> = obs(&spec, "").state.iter().map(|&x| f64::from(x)).collect();
        let norm = normalize(m.normalize().next().unwrap(), STATE, &raw).unwrap();
        let texts = r.log.lock().unwrap().texts.clone();
        assert_eq!(
            texts.last().unwrap(),
            &pi05_prompt("pick up", &pi05_bins(&norm))
        );
        let state = input(&r.log, STATE);
        assert_eq!(state.shape, vec![1, 32]);
        let mut want: Vec<f32> = norm.iter().map(|&x| x as f32).collect();
        want.resize(32, 0.0);
        assert_eq!(f32s(&state), want.as_slice());
    }

    #[test]
    fn pi0_fills_the_dummy_slot_and_leaves_the_prompt_as_the_task() {
        let mut r = rig_with(PI0_LIBERO, pi0_ports(), false, true, vec![0.5; 350]);
        let spec = r.engine.spec().clone();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        assert_eq!(spec.resize, cascadia_types::ResizeConvention::Center);
        assert_eq!(
            (spec.noise, spec.noise_shape.clone()),
            (NoiseMode::Input, Some(vec![50, 32]))
        );
        assert_eq!(spec.chunk, [50, 7]);
        let a = r.engine.infer(&obs(&spec, " pick_up")).unwrap();
        assert_eq!((a.steps, a.dim), (50, 7));
        let images = input(&r.log, IMAGES);
        assert_eq!(images.shape, vec![3, 1, 3, 224, 224]);
        let per = 3 * 224 * 224;
        let px = |v: u8| v as f32 / 255.0 * 2.0 - 1.0;
        assert_eq!(f32s(&images)[0], px(10));
        assert_eq!(f32s(&images)[per], px(20));
        assert!(f32s(&images)[2 * per..].iter().all(|&x| x == -1.0));
        assert_eq!(
            input(&r.log, IMAGE_MASKS).data,
            TensorData::Bool(vec![true, true, false])
        );
        // No state bins, no cleaning: the worst-case check and the call both
        // tokenize the task plus new_line's "\n".
        let texts = r.log.lock().unwrap().texts.clone();
        assert_eq!(texts, vec![" pick_up\n", " pick_up\n"]);
        assert_eq!(input(&r.log, NOISE).shape, vec![1, 50, 32]);
    }

    #[test]
    fn xvla_flips_image2_and_serves_seven_wide_actions() {
        // Each 20-wide row: position, identity 6-D rotation, gripper, a second arm.
        let row: Vec<f32> = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.9]
            .into_iter()
            .chain([0.7; 10])
            .collect();
        let out: Vec<f32> = (0..30).flat_map(|_| row.clone()).collect();
        let mut r = rig_with(XVLA_LIBERO, xvla_ports(), false, true, out);
        let spec = r.engine.spec().clone();
        assert_eq!(spec.chunk, [30, 7]);
        // Pixels that differ, so a rotation shows.
        let mut o = obs(&spec, "pick up the bowl");
        let side = 224 * 224;
        for (_, img) in o.images.iter_mut() {
            img.data = (0..side).flat_map(|p| [(p % 251) as u8; 3]).collect();
        }
        let a = r.engine.infer(&o).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (30, 7, 210));
        let want = [0.1f32, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0];
        for (i, &w) in want.iter().enumerate() {
            assert!(
                (a.actions[i] - w).abs() <= 1e-6,
                "{i}: {:?}",
                &a.actions[..7]
            );
        }
        assert_eq!(&a.actions[203..], &a.actions[..7], "every row converted");
        let images = input(&r.log, IMAGES);
        let px = |p: usize| (p % 251) as u8 as f32 / 255.0 * 2.0 - 1.0;
        let img = f32s(&images);
        assert_eq!(img[0], px(0), "image is as sent");
        assert_eq!(img[3 * side], px(side - 1), "image2 is rotated 180 degrees");
        assert_eq!(img[3 * side + side - 1], px(0));
        // No new_line: the tokenizer sees the task as sent.
        assert_eq!(
            r.log.lock().unwrap().texts.last().unwrap(),
            "pick up the bowl"
        );
    }

    const VLAJEPA_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/vlajepa-libero-manifest.json");

    /// The real ports of our VLA-JEPA LIBERO export.
    fn vla_jepa_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(INPUT_IDS, DType::I64, &[1, 288]),
                port(ATTENTION_MASK, DType::I64, &[1, 288]),
                port(PIXEL_VALUES, DType::F32, &[512, 1536]),
                port(EMBODIED_POSITIONS, DType::I64, &[1, 32]),
                port(STATE, DType::F32, &[1, 8]),
                port(NOISE, DType::F32, &[1, 7, 7]),
            ],
            vec![port(ACTION, DType::F32, &[1, 7, 7])],
        )
    }

    /// The fixture with a template the byte tokenizer can read: 32 bytes of 1 are the embodied tokens.
    fn vla_jepa_manifest(markers: usize) -> String {
        let mut v: serde_json::Value = serde_json::from_str(VLAJEPA_LIBERO).unwrap();
        v["model"]["preprocessors"][0]["template"] =
            serde_json::json!(format!("T:{{instruction}};{}", "\u{1}".repeat(markers)));
        v["model"]["preprocessors"][2]["embodied_token_id"] = serde_json::json!(1);
        v.to_string()
    }

    fn vla_jepa() -> Rig {
        let out: Vec<f32> = (0..49).map(|i| i as f32 / 49.0 - 0.5).collect();
        rig_with(&vla_jepa_manifest(32), vla_jepa_ports(), false, true, out)
    }

    #[test]
    fn vla_jepa_binds_its_own_ports_with_a_raw_state_and_final_actions() {
        let mut r = vla_jepa();
        let spec = r.engine.spec().clone();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        assert_eq!((spec.height, spec.width), (224, 224));
        assert_eq!(spec.resize, cascadia_types::ResizeConvention::Stretch);
        assert_eq!(spec.chunk, [7, 7]);
        assert_eq!(
            (spec.noise, spec.noise_shape.clone()),
            (NoiseMode::Input, Some(vec![7, 7]))
        );
        let o = obs(&spec, "go");
        let a = r.engine.infer(&o).unwrap();
        let out: Vec<f32> = (0..49).map(|i| i as f32 / 49.0 - 0.5).collect();
        assert_eq!((a.steps, a.dim), (7, 7));
        assert_eq!(a.actions, out, "the graph's actions are served as they are");

        let (inputs, _) = vla_jepa_ports();
        for p in &inputs {
            let t = input(&r.log, &p.aliases[0]);
            let want: Vec<usize> = p.dims.iter().map(|d| d.unwrap()).collect();
            assert_eq!(t.shape, want, "{}", p.aliases[0]);
            assert_eq!(Some(t.dtype()), p.dtype, "{}", p.aliases[0]);
        }
        // The task goes into the template verbatim.
        let text = format!("T:go;{}", "\u{1}".repeat(32));
        assert_eq!(r.log.lock().unwrap().texts.last().unwrap(), &text);
        let TensorData::I64(ids) = input(&r.log, INPUT_IDS).data else {
            panic!("ids are i64")
        };
        assert_eq!(
            &ids[..text.len()],
            text.bytes().map(i64::from).collect::<Vec<_>>()
        );
        let TensorData::I64(mask) = input(&r.log, ATTENTION_MASK).data else {
            panic!("the mask is i64")
        };
        assert!(
            mask[..text.len()].iter().all(|&m| m == 1)
                && mask[text.len()..].iter().all(|&m| m == 0)
        );
        assert_eq!(
            input(&r.log, EMBODIED_POSITIONS).data,
            TensorData::I64((5..37).collect())
        );
        assert_eq!(input(&r.log, STATE).data, TensorData::F32(o.state.clone()));
        // image (value 10) fills the first 256 rows, image2 (20) the rest.
        let px = input(&r.log, PIXEL_VALUES);
        let v = f32s(&px);
        let norm = |b: u8| (f32::from(b) / 255.0 - 0.5) / 0.5;
        assert!((v[0] - norm(10)).abs() <= 1e-6 && (v[256 * 1536] - norm(20)).abs() <= 1e-6);
        assert!(!r
            .log
            .lock()
            .unwrap()
            .inputs
            .iter()
            .any(|(n, _)| n == IMAGES || n == PROMPT));
    }

    #[test]
    fn vla_jepa_prompt_length_is_exact_and_spelled_embodied_tokens_are_refused() {
        // "T:" + task + ";" + 32 markers: 253 bytes of task fill the 288 ids exactly.
        let mut r = vla_jepa();
        let spec = r.engine.spec().clone();
        r.engine.infer(&obs(&spec, &"a".repeat(253))).unwrap();
        let e = r.engine.infer(&obs(&spec, &"a".repeat(254))).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
        assert_eq!(
            e.to_string(),
            "prompt too long: the prompt needs 289 tokens; the limit is 288"
        );
        let e = r
            .engine
            .infer(&obs(&spec, "a\u{1}b"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("the task contains \"\\u{1}\""), "{e}");
    }

    #[test]
    fn vla_jepa_load_refuses_a_template_or_ports_that_disagree() {
        let log = Arc::new(Mutex::new(Log::default()));
        let load = |manifest: &str, (i, o): (Vec<PortInfo>, Vec<PortInfo>)| {
            pipeline_with(manifest, (&i, &o), false, true, 288, &log, vec![0.0; 49])
                .err()
                .expect("must not load")
                .to_string()
        };
        let e = load(&vla_jepa_manifest(31), vla_jepa_ports());
        assert!(e.contains("the empty task: the prompt has 31"), "{e}");
        let m = vla_jepa_manifest(32);
        let with = |i: usize, p: PortInfo| {
            let (mut inputs, outputs) = vla_jepa_ports();
            inputs[i] = p;
            (inputs, outputs)
        };
        for (ports, want) in [
            (
                with(1, port(ATTENTION_MASK, DType::Bool, &[1, 288])),
                "expected [I64]",
            ),
            (
                with(1, port(ATTENTION_MASK, DType::I64, &[1, 200])),
                "differs from input_ids",
            ),
            (
                with(2, port(PIXEL_VALUES, DType::F32, &[512, 1024])),
                "1024 wide",
            ),
            (
                with(2, port(PIXEL_VALUES, DType::F32, &[500, 1536])),
                "500 rows",
            ),
            (
                with(2, port(PIXEL_VALUES, DType::F32, &[768, 1536])),
                "the model has 3",
            ),
            (
                with(3, port(EMBODIED_POSITIONS, DType::I64, &[1, 16])),
                "takes 16",
            ),
            (
                with(0, port(PROMPT, DType::I64, &[1, 288])),
                "not one the engine fills",
            ),
        ] {
            let e = load(&m, ports);
            assert!(e.contains(want), "{want}: {e}");
        }
    }

    const EVO1_LIBERO: &str = include_str!("../../tests/fixtures/policy/evo1-libero-manifest.json");

    /// The real ports of our EVO1 LIBERO export.
    fn evo1_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(STATE, DType::F32, &[1, 8]),
                port(PROMPT, DType::I64, &[1, 64]),
                port(PROMPT_MASK, DType::Bool, &[1, 64]),
                port(IMAGES, DType::F32, &[2, 1, 3, 448, 448]),
                port(IMAGE_MASKS, DType::Bool, &[2, 1]),
                port(NOISE, DType::F32, &[1, 50, 7]),
            ],
            vec![port(ACTION, DType::F32, &[1, 50, 7])],
        )
    }

    const XR0_LIBERO: &str = include_str!("../../tests/fixtures/policy/xr0-libero-manifest.json");

    /// The real ports of our XR0 LIBERO export.
    fn xr0_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(PROMPT, DType::I64, &[1, 2048]),
                port(PROMPT_MASK, DType::I64, &[1, 2048]),
                port(PIXEL_VALUES, DType::F32, &[512, 1536]),
                port(STATE, DType::F32, &[1, 1, 32]),
                port(NOISE, DType::F32, &[1, 10, 32]),
            ],
            vec![port(ACTION, DType::F32, &[1, 10, 32])],
        )
    }

    #[test]
    fn xr0_binds_its_ports_pads_the_state_and_denormalizes_per_row() {
        // The byte tokenizer needs a wide port: the template alone is ~2 KB.
        let out = vec![1.0f32; 10 * 32];
        let mut r = rig_with(XR0_LIBERO, xr0_ports(), false, true, out);
        let spec = r.engine.spec().clone();
        assert_eq!(spec.cameras, vec!["base", "wrist_left"]);
        assert_eq!(spec.chunk, [10, 7]);
        let o = obs(&spec, "go");
        let a = r.engine.infer(&o).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (10, 7, 70));
        // 1 * (std + 1e-6) + mean, in f32, first row.
        assert_eq!(a.actions[0], 1.0 * (0.335_501_02f32 + 1e-6) + 0.0628);
        let (inputs, _) = xr0_ports();
        for p in &inputs {
            let t = input(&r.log, &p.aliases[0]);
            let want: Vec<usize> = p.dims.iter().map(|d| d.unwrap()).collect();
            assert_eq!(t.shape, want, "{}", p.aliases[0]);
            assert_eq!(Some(t.dtype()), p.dtype, "{}", p.aliases[0]);
        }
        let mut state = o.state.clone();
        state.resize(32, 0.0);
        assert_eq!(input(&r.log, STATE).data, TensorData::F32(state));
        let text = r.log.lock().unwrap().texts.last().unwrap().clone();
        assert!(text.contains("# Base View\n<|vision_start|>"), "{text}");
        assert!(text.contains("# Left-Wrist View"), "{text}");
        assert_eq!(text.matches("<|image_pad|>").count(), 128);
        assert!(text.ends_with(
            "task:\ngo /no_cot<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n"
        ));
        // base (10) in the first 256 rows, wrist_left (20) after.
        let px = input(&r.log, PIXEL_VALUES);
        let v = f32s(&px);
        let norm = |b: u8| (f32::from(b) * (1.0f64 / 255.0) as f32 - 0.5) / 0.5;
        assert_eq!((v[0], v[256 * 1536]), (norm(10), norm(20)));
    }

    const XR1_ROBOCASA: &str =
        include_str!("../../tests/fixtures/policy/xr1-robocasa365-manifest.json");

    /// The ports of our XR1 RoboCasa365 export; the prompt port widened for the byte tokenizer.
    fn xr1_ports(state_rows: usize) -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(PROMPT, DType::I64, &[1, 8192]),
                port(PROMPT_MASK, DType::I64, &[1, 8192]),
                port(PIXEL_VALUES, DType::F32, &[1536, 1536]),
                port(STATE, DType::F32, &[1, state_rows, 60]),
                port(NOISE, DType::F32, &[1, 16, 60]),
            ],
            vec![port(ACTION, DType::F32, &[1, 16, 60])],
        )
    }

    #[test]
    fn xr1_takes_four_frames_per_camera_and_four_state_rows() {
        let mut r = rig_with(XR1_ROBOCASA, xr1_ports(4), false, true, vec![1.0; 16 * 60]);
        assert_eq!(r.engine.warmup().unwrap().len(), WARMUP_CALLS);
        let spec = r.engine.spec().clone();
        assert_eq!(
            spec.cameras,
            ["agentview_left", "agentview_right", "eye_in_hand"]
        );
        assert_eq!(
            (spec.history, spec.state_len, spec.chunk),
            (4, 14, [16, 12])
        );
        assert_eq!(spec.server_resize, None);
        // Camera c's frame t is all 10 * (c + 1) + t, sent oldest first.
        let images: Vec<(String, Image)> = spec
            .cameras
            .iter()
            .enumerate()
            .flat_map(|(c, name)| {
                (0..4).map(move |t| (name.clone(), frame(256, 10 * (c as u8 + 1) + t)))
            })
            .collect();
        let o = Observation {
            images,
            state: (0..4 * 14).map(|i| 1.0 + i as f32 / 1000.0).collect(),
            prompt: "go".into(),
            seed: None,
            noise: None,
            rtc: None,
        };
        let a = r.engine.infer(&o).unwrap();
        assert_eq!((a.steps, a.dim), (16, 12));
        assert_eq!(a.actions[0], 1.0 * (1.0 + 1e-6f32));
        let (inputs, _) = xr1_ports(4);
        for p in &inputs {
            let t = input(&r.log, &p.aliases[0]);
            let want: Vec<usize> = p.dims.iter().map(|d| d.unwrap()).collect();
            assert_eq!(t.shape, want, "{}", p.aliases[0]);
        }
        // Slot k of a camera holds its frames 2k and 2k+1, 256 rows each;
        // within a row, a channel's patch of the first frame, then the second's.
        let v = f32s(&input(&r.log, PIXEL_VALUES)).to_vec();
        let norm = |b: u8| (f32::from(b) - 127.5) / 127.5;
        let row = |slot: usize| slot * 256 * 1536;
        assert_eq!(
            [
                v[row(0)],
                v[row(0) + 256],
                v[row(1)],
                v[row(1) + 256],
                v[row(2)]
            ],
            [norm(10), norm(11), norm(12), norm(13), norm(20)]
        );
        let state = f32s(&input(&r.log, STATE)).to_vec();
        assert_eq!(
            state[..14],
            o.state[..14]
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_f32())
                .collect::<Vec<_>>()[..]
        );
        assert_eq!(state[14..60], [0.0; 46]);
        assert_eq!(state[60], half::bf16::from_f32(o.state[14]).to_f32());
        let text = r.log.lock().unwrap().texts.last().unwrap().clone();
        assert!(text.starts_with(
            "<|im_start|>user\nLeft camera: <0.0 seconds><|vision_start|><|image_pad|>"
        ));
        assert!(text.contains("<|vision_end|><0.1 seconds><|vision_start|>"));
        assert!(text.contains("<|vision_end|>\nWrist camera: <0.0 seconds>"));
        assert_eq!(text.matches("<|image_pad|>").count(), 6 * 64);
        // One frame short is refused before the model runs.
        let mut short = o.clone();
        short.images.pop();
        let e = r.engine.infer(&short).unwrap_err().to_string();
        assert!(
            e.contains("camera \"eye_in_hand\" has 3 frames; this model takes 4"),
            "{e}"
        );
        // A graph with a one-row state port does not fit the package.
        let log = Arc::new(Mutex::new(Log::default()));
        let (inputs, outputs) = xr1_ports(1);
        let e = pipeline_with(
            XR1_ROBOCASA,
            (&inputs, &outputs),
            false,
            true,
            8192,
            &log,
            vec![],
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("xr0 makes [1, 4, 60]"), "{e}");
    }

    #[test]
    fn evo1_strips_the_task_flips_both_cameras_and_binarizes_the_gripper() {
        // Normalized actions in [-1, 1]; the gripper alternates around the 0.0 threshold.
        let out: Vec<f32> = (0..350)
            .map(|i| {
                if i % 7 == 6 {
                    [0.3, 0.0, -0.4][i / 7 % 3]
                } else {
                    0.5
                }
            })
            .collect();
        let mut r = rig_with(EVO1_LIBERO, evo1_ports(), false, true, out);
        let spec = r.engine.spec().clone();
        assert_eq!(spec.chunk, [14, 7]);
        assert_eq!(spec.resize, cascadia_types::ResizeConvention::Stretch);
        let a = r.engine.infer(&obs(&spec, "  pick up the bowl\n")).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (14, 7, 98));
        // Denormalized min_max: (0.5 + 1) / 2 * (max - min) + min, then the gripper to ±1.
        let want0 = (1.5 / 2.0 * (0.9375 + 0.9375) - 0.9375) as f32;
        assert!((a.actions[0] - want0).abs() <= 1e-6, "{}", a.actions[0]);
        let grippers: Vec<f32> = a.actions.chunks(7).map(|r| r[6]).collect();
        // Rows 0, 1, 2: denormalized 0.3, 0.0 (not above), -0.4.
        assert_eq!(&grippers[..3], &[1.0, -1.0, -1.0]);
        assert!(grippers.iter().all(|g| g.abs() == 1.0));
        assert_eq!(
            r.log.lock().unwrap().texts.last().unwrap(),
            "pick up the bowl"
        );
    }

    const GR00T_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/gr00t-libero-spatial-manifest.json");

    /// Our GR00T export's ports as the CPU plugin shows them: `state` renamed.
    fn gr00t_ports(state: &str) -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(INPUT_IDS, DType::I64, &[1, 192]),
                port(ATTENTION_MASK, DType::I64, &[1, 192]),
                port(PIXEL_VALUES, DType::F32, &[512, 1536]),
                port(POSITION_IDS, DType::I64, &[3, 1, 192]),
                port(state, DType::F32, &[1, 1, 132]),
                port("embodiment_id", DType::I64, &[1]),
                port("actions", DType::F32, &[1, 40, 132]),
            ],
            vec![port("predicted_action", DType::F32, &[1, 40, 132])],
        )
    }

    /// The fixture with the fake tokenizer's `<|image_pad|>` id.
    fn gr00t_manifest() -> String {
        GR00T_LIBERO.replace("\"image_token_id\": 151655", "\"image_token_id\": 1001")
    }

    fn gr00t_rig(state_port: &str) -> EngineResult<Rig> {
        let log = Arc::new(Mutex::new(Log::default()));
        let (inputs, outputs) = gr00t_ports(state_port);
        let m = PolicyManifest::parse(&gr00t_manifest(), LoadOptions::default()).unwrap();
        let engine = PolicyPipeline::new(
            m,
            server("CPU"),
            Box::new(FakeModel {
                log: log.clone(),
                output: (
                    vec![1, 40, 132],
                    (0..40 * 132)
                        .map(|i| (i % 132) as f32 / 66.0 - 1.0)
                        .collect(),
                ),
                action: "predicted_action",
            }),
            &inputs,
            &outputs,
            false,
            Box::new(FakeTokenizer {
                log: log.clone(),
                truncation_state: true,
                port: 0,
            }),
        )?;
        log.lock().unwrap().texts.clear();
        Ok(Rig { engine, log })
    }

    #[test]
    fn gr00t_binds_by_name_or_index_and_expands_its_prompt() {
        let mut r = gr00t_rig("272").unwrap();
        let spec = r.engine.spec().clone();
        assert_eq!(spec.cameras, ["image", "wrist_image"]);
        assert_eq!((spec.height, spec.width, spec.state_len), (256, 256, 8));
        assert_eq!(spec.chunk, [16, 7]);
        assert_eq!(
            (spec.noise, spec.noise_shape.clone()),
            (NoiseMode::Input, Some(vec![40, 132]))
        );
        assert_eq!(spec.resize, cascadia_types::ResizeConvention::Center);
        assert!(spec
            .attribution
            .as_deref()
            .unwrap()
            .contains("NVIDIA Open Model License"));
        let mut o = obs(&spec, "Pick up the BLACK bowl!");
        o.seed = Some(3);
        let a = r.engine.infer(&o).unwrap();
        assert_eq!((a.steps, a.dim), (16, 7));
        let log = r.log.lock().unwrap();
        let names: Vec<&str> = log.inputs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                INPUT_IDS,
                ATTENTION_MASK,
                PIXEL_VALUES,
                POSITION_IDS,
                "272",
                "embodiment_id",
                "actions"
            ]
        );
        assert_eq!(
            log.texts.last().unwrap(),
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\
             <|vision_start|><|image_pad|><|vision_end|>pick up the black bowl<|im_end|>\n"
        );
        drop(log);
        let TensorData::I64(ids) = input(&r.log, INPUT_IDS).data else {
            panic!()
        };
        assert_eq!(ids.iter().filter(|&&t| t == 1001).count(), 128);
        let TensorData::I64(mask) = input(&r.log, ATTENTION_MASK).data else {
            panic!()
        };
        let valid = mask.iter().filter(|&&m| m == 1).count();
        assert!(mask[..valid].iter().all(|&m| m == 1) && ids[valid..].iter().all(|&t| t == 0));
        let state = input(&r.log, "272");
        assert_eq!(state.shape, [1, 1, 132]);
        assert!(f32s(&state)[8..].iter().all(|&x| x == 0.0));
        assert_eq!(
            input(&r.log, "embodiment_id").data,
            TensorData::I64(vec![2])
        );
        assert_eq!(input(&r.log, "actions").shape, [1, 40, 132]);
        // Row 0's first 7 values, clipped to [-1, 1] and mapped to [q01, q99].
        let x = -1.0f64;
        let want =
            (x.clamp(-1.0, 1.0) + 1.0) / 2.0 * (0.9375 - -0.7454732060432434) + -0.7454732060432434;
        assert!(
            (f64::from(a.actions[0]) - want).abs() < 1e-6,
            "{}",
            a.actions[0]
        );
    }

    #[test]
    fn gr00t_prompt_limit_and_special_tokens() {
        let mut r = gr00t_rig(STATE).unwrap();
        let spec = r.engine.spec().clone();
        // The fake splits bytes: 66 ids fit beside 2 images of 64 in 192.
        let used = r
            .engine
            .tokens(&prompt_text(&r.engine.manifest, "", &[], false).unwrap())
            .unwrap()
            .ids
            .len();
        let fit = "a".repeat(66 - used);
        r.engine.infer(&obs(&spec, &fit)).unwrap();
        let e = r.engine.infer(&obs(&spec, &format!("{fit}a"))).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
        assert!(
            e.to_string()
                .contains("needs 65 language tokens; the limit is 64"),
            "{e}"
        );
        for sp in FAKE_SPECIAL.iter().skip(1) {
            let e = r
                .engine
                .infer(&obs(&spec, &format!("go {sp} now")))
                .unwrap_err();
            assert!(
                e.to_string().contains(&format!("the task contains {sp:?}")),
                "{e}"
            );
        }
        // Formalized, a task cannot spell a token, but the raw text is checked too.
        r.engine.infer(&obs(&spec, "go IM_END now")).unwrap();
        // A port with another name where a layout name is expected is refused.
        let (mut inputs, outputs) = gr00t_ports(STATE);
        inputs.swap(0, 1);
        inputs[0].aliases = vec!["position_ids".into()];
        let m = PolicyManifest::parse(&gr00t_manifest(), LoadOptions::default()).unwrap();
        let host = Gr00tHost::new(&m).unwrap();
        let e = bind_gr00t(&m, &host, &inputs, &outputs)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("has no \"attention_mask\" port, and input 1 is [\"input_ids\"]"),
            "{e}"
        );
        let (mut inputs, outputs) = gr00t_ports(STATE);
        inputs[4].dims = vec![Some(1), Some(1), Some(32)];
        let e = bind_gr00t(&m, &host, &inputs, &outputs)
            .unwrap_err()
            .to_string();
        assert!(e.contains("gr00t makes [1, 1, 132]"), "{e}");
        let (mut inputs, outputs) = gr00t_ports(STATE);
        inputs.pop();
        assert!(bind_gr00t(&m, &host, &inputs, &outputs)
            .unwrap_err()
            .to_string()
            .contains("has 6 inputs"));
    }

    const MOLMOACT2_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/molmoact2-libero-manifest.json");

    /// Our MolmoAct2 export's ports with a `seq`-long text (650 in the export;
    /// the fake tokenizer spends one id per byte, so the rig takes 1395).
    fn molmoact2_ports_of(seq: usize) -> (Vec<PortInfo>, Vec<PortInfo>) {
        (
            vec![
                port(INPUT_IDS, DType::I64, &[1, seq]),
                port(ATTENTION_MASK, DType::I64, &[1, seq]),
                port(TOKEN_TYPE_IDS, DType::I64, &[1, seq]),
                port(IMAGES, DType::F32, &[1, 2, 729, 588]),
                port(TOKEN_POOLING, DType::I64, &[1, 392, 4]),
                port(ACTION_DIM_IS_PAD, DType::Bool, &[1, 32]),
            ],
            vec![port(ACTION, DType::F32, &[1, 10, 7])],
        )
    }

    fn molmoact2_ports() -> (Vec<PortInfo>, Vec<PortInfo>) {
        molmoact2_ports_of(650)
    }

    /// The fixture with the fake tokenizer's `<|image|>` id.
    fn molmoact2() -> Rig {
        let manifest = MOLMOACT2_LIBERO.replace(
            "\"image_placeholder_token_id\": 154629",
            "\"image_placeholder_token_id\": 1006",
        );
        rig_with(
            &manifest,
            molmoact2_ports_of(1395),
            false,
            true,
            vec![2.0; 70],
        )
    }

    #[test]
    fn molmoact2_builds_its_prompt_and_expands_images() {
        let mut r = molmoact2();
        let spec = r.engine.spec().clone();
        assert_eq!(spec.cameras, ["image", "wrist_image"]);
        assert_eq!((spec.height, spec.width, spec.state_len), (378, 378, 8));
        assert_eq!(
            (spec.chunk, spec.noise, spec.noise_shape.clone()),
            ([10, 7], NoiseMode::Constant, None)
        );
        assert_eq!(spec.resize, cascadia_types::ResizeConvention::Stretch);
        let mut o = obs(&spec, "Task: Pick up the BOWL.");
        o.state[0] = 99.0;
        let a = r.engine.infer(&o).unwrap();
        // Clipped to 1, then mapped to q99; the gripper (masked) is clipped only.
        assert_eq!((a.steps, a.dim), (10, 7));
        assert!(
            (a.actions[0] - 0.853_654_3_f32).abs() < 1e-6,
            "{}",
            a.actions[0]
        );
        assert_eq!(a.actions[6], 1.0);
        let text = r.log.lock().unwrap().texts.last().unwrap().clone();
        assert!(
            text.starts_with(
                "Image 1<|image|>Image 2<|image|><|im_start|>user\nThe task is to pick up the bowl. \
                 The setup is <setup_start>single franka robotic arm in libero<setup_end>. \
                 The current state of the robot is <state_start><state_255><state_"
            ),
            "{text}"
        );
        assert!(
            text.ends_with("<|im_end|>\n<|im_start|>assistant\n<action_output>"),
            "{text}"
        );
        let TensorData::I64(ids) = input(&r.log, INPUT_IDS).data else {
            panic!()
        };
        assert_eq!(ids.len(), 1395);
        assert_eq!(ids[0], 151645, "BOS");
        assert_eq!(ids.iter().filter(|&&t| t == 154626).count(), 2 * 196);
        assert_eq!(
            ids.iter().filter(|&&t| t == 1006).count(),
            0,
            "placeholders expanded"
        );
        let TensorData::I64(types) = input(&r.log, TOKEN_TYPE_IDS).data else {
            panic!()
        };
        assert_eq!(types.iter().sum::<i64>(), 2 * 198);
        let TensorData::I64(mask) = input(&r.log, ATTENTION_MASK).data else {
            panic!()
        };
        let valid = mask.iter().filter(|&&m| m == 1).count();
        assert!(ids[valid..].iter().all(|&t| t == 151643));
        let TensorData::I64(pool) = input(&r.log, TOKEN_POOLING).data else {
            panic!()
        };
        assert_eq!(pool.iter().filter(|&&i| i == -1).count(), 110);
        assert_eq!(*pool.iter().max().unwrap(), 2 * 729 - 1);
        assert_eq!(pool[..8], [0, 1, 27, 28, 2, 3, 29, 30]);
        let images = input(&r.log, IMAGES);
        assert_eq!(images.shape, [1, 2, 729, 588]);
        // Slot 0 is `image` (sent second, value 10); every value is (10/255 - 0.5) / 0.5.
        assert_eq!(f32s(&images)[0], (10.0f32 / 255.0 - 0.5) / 0.5);
        let TensorData::Bool(pad) = input(&r.log, ACTION_DIM_IS_PAD).data else {
            panic!()
        };
        assert_eq!(pad.iter().filter(|&&p| !p).count(), 7);
    }

    #[test]
    fn molmoact2_on_an_arm_cpu_is_refused() {
        for (device, arch) in [("CPU", "aarch64"), ("cpu", "aarch64")] {
            let e = refuse_known_bad_device(PolicyKind::MolmoAct2, device, arch).unwrap_err();
            assert!(matches!(e, EngineError::InvalidConfig(_)), "{e}");
            assert!(e.to_string().contains("use an x86 CPU or the GPU"), "{e}");
        }
        for (kind, device, arch) in [
            (PolicyKind::MolmoAct2, "CPU", "x86_64"),
            (PolicyKind::MolmoAct2, "GPU", "aarch64"),
            (PolicyKind::Gr00tN1d7, "CPU", "aarch64"),
        ] {
            refuse_known_bad_device(kind, device, arch).unwrap();
        }
        // The pipeline applies it on CPU with the build's own arch.
        let log = Arc::new(Mutex::new(Log::default()));
        let manifest = MOLMOACT2_LIBERO.replace(
            "\"image_placeholder_token_id\": 154629",
            "\"image_placeholder_token_id\": 1006",
        );
        let (inputs, outputs) = molmoact2_ports_of(1395);
        let r = PolicyPipeline::new(
            PolicyManifest::parse(&manifest, LoadOptions::default()).unwrap(),
            server("CPU"),
            Box::new(FakeModel {
                log: log.clone(),
                output: (vec![1, 10, 7], vec![0.0; 70]),
                action: ACTION,
            }),
            &inputs,
            &outputs,
            false,
            Box::new(FakeTokenizer {
                log: log.clone(),
                truncation_state: true,
                port: 1000,
            }),
        );
        assert_eq!(r.is_err(), cfg!(target_arch = "aarch64"));
    }

    #[test]
    fn molmoact2_refuses_long_prompts_and_spelled_tokens() {
        let mut r = molmoact2();
        let spec = r.engine.spec().clone();
        // One fake id per byte: find the longest task that fits 1000 ids.
        let empty = prompt_text(&r.engine.manifest, "", &[0.0; 8], false).unwrap();
        let used = fake_ids(&empty).len();
        let fit = "a".repeat(1000 - used);
        // The fake's state tokens are bytes, so their count moves with the state: use the
        // check's own zero state (a real tokenizer has one id per state token).
        let zero = |task: &str| Observation {
            state: vec![0.0; 8],
            ..obs(&spec, task)
        };
        r.engine.infer(&zero(&fit)).unwrap();
        let e = r.engine.infer(&zero(&format!("{fit}a"))).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
        for task in [
            "go <|image|> now",
            "go <|IM_END|> now",
            "go <STATE_5> now",
            "<state_end>",
        ] {
            let e = r.engine.infer(&obs(&spec, task)).unwrap_err();
            assert!(e.to_string().contains("the task contains"), "{task}: {e}");
        }
        // The state tokens differ per call; the guard does not count them.
        let mut o = obs(&spec, "pick up the bowl");
        o.state = vec![0.5; 8];
        r.engine.infer(&o).unwrap();
        // Ports that disagree with the geometry are refused.
        let m = PolicyManifest::parse(MOLMOACT2_LIBERO, LoadOptions::default()).unwrap();
        let host = MolmoAct2Host::new(&m).unwrap();
        let (mut inputs, outputs) = molmoact2_ports();
        inputs[4].dims = vec![Some(1), Some(196), Some(4)];
        let e = bind_molmoact2(&host, &inputs, &outputs, false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("makes [1, 392, 4]"), "{e}");
        let (mut inputs, outputs) = molmoact2_ports();
        inputs[3].dims = vec![Some(1), Some(1), Some(729), Some(588)];
        let e = bind_molmoact2(&host, &inputs, &outputs, false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("makes [1, 2, 729, 588]"), "{e}");
        let (_, outputs) = molmoact2_ports();
        let (b, _) = bind_molmoact2(&host, &molmoact2_ports().0, &outputs, false).unwrap();
        assert_eq!(b.prompt_len, 255);
    }

    #[test]
    fn templated_prompts_refuse_a_task_that_spells_their_structure() {
        for (name, mut r) in [
            ("vlajepa", vla_jepa()),
            (
                "xr0",
                rig_with(XR0_LIBERO, xr0_ports(), false, true, vec![1.0; 10 * 32]),
            ),
        ] {
            let spec = r.engine.spec().clone();
            // The tokenizer's special tokens and the template's own markers.
            let markers = ["<cot>", "</cot>"];
            for sp in FAKE_SPECIAL
                .iter()
                .chain(if name == "xr0" { &markers[..] } else { &[] })
            {
                let task = format!("pick up {sp} the bowl");
                let e = r.engine.infer(&obs(&spec, &task)).unwrap_err();
                assert!(matches!(e, EngineError::InvalidConfig(_)), "{name}: {e}");
                assert!(
                    e.to_string().contains(&format!("the task contains {sp:?}")),
                    "{name}: {e}"
                );
                // Cached per task, like the length check: no second tokenize.
                let before = r.log.lock().unwrap().texts.len();
                assert!(r.engine.infer(&obs(&spec, &task)).is_err());
                assert_eq!(r.log.lock().unwrap().texts.len(), before, "{name}");
            }
            r.engine.infer(&obs(&spec, "pick up the bowl")).unwrap();
        }
    }

    #[test]
    fn the_guard_checks_the_ids_keep_the_templates_structure() {
        let log = Arc::new(Mutex::new(Log::default()));
        let mut tok = FakeTokenizer {
            log,
            truncation_state: true,
            port: 64,
        };
        let g =
            PromptGuard::new("<|im_start|>a<|image_pad|><|image_pad|>b", &mut tok, &[]).unwrap();
        assert_eq!(g.count("<|image_pad|>"), Some(2));
        assert_eq!(g.count("<|im_start|>"), Some(1));
        assert_eq!(g.count("<|im_end|>"), Some(0));
        let ok = tok
            .tokenize("<|im_start|>a go<|image_pad|><|image_pad|>b")
            .unwrap();
        g.check_ids(&ok).unwrap();
        // A pad that slipped past the text check moves the image features.
        let bad = tok
            .tokenize("<|im_start|>a<|image_pad|> go<|image_pad|><|image_pad|>b")
            .unwrap();
        let e = g.check_ids(&bad).unwrap_err().to_string();
        assert!(
            e.contains("3 \"<|image_pad|>\" tokens; the template has 2"),
            "{e}"
        );
        // Padding past the mask does not count.
        let mut padded = ok.clone();
        padded.ids.push(1002);
        padded.mask.push(false);
        g.check_ids(&padded).unwrap();
    }

    #[test]
    fn a_prompt_step_after_the_tokenizer_is_refused() {
        let log = Arc::new(Mutex::new(Log::default()));
        let mut v: serde_json::Value = serde_json::from_str(PI0_LIBERO).unwrap();
        let steps = v["model"]["preprocessors"].as_array_mut().unwrap();
        let pi0 = steps.remove(1);
        steps.push(pi0);
        let (inputs, outputs) = pi0_ports();
        let e = pipeline_with(
            &v.to_string(),
            (&inputs, &outputs),
            false,
            true,
            48,
            &log,
            vec![0.5; 350],
        )
        .err()
        .expect("pi0 after ov_tokenizer must not load");
        assert!(e.to_string().contains("comes after ov_tokenizer"), "{e}");
    }

    #[test]
    fn prompt_length_with_a_truncation_state_accepts_an_exact_fit() {
        // One token per byte: new_line adds one, so 47 bytes fill the 48-token port.
        let mut r = smolvla(true);
        let spec = r.engine.spec().clone();
        r.engine.infer(&obs(&spec, &"a".repeat(47))).unwrap();
        let e = r.engine.infer(&obs(&spec, &"a".repeat(48))).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
        assert!(e.to_string().contains("needs 49 tokens"), "{e}");
        assert!(e.to_string().contains("limit is 48"), "{e}");
    }

    #[test]
    fn prompt_length_with_an_always_on_truncate_refuses_a_full_mask() {
        let mut r = smolvla(false);
        let spec = r.engine.spec().clone();
        r.engine.infer(&obs(&spec, &"a".repeat(46))).unwrap();
        let e = r.engine.infer(&obs(&spec, &"a".repeat(47))).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
        assert!(e.to_string().contains("fills all 48 tokens"), "{e}");
    }

    #[test]
    fn pi05_prompt_length_is_judged_on_the_worst_case_state() {
        // Worst-case text is len(task) + 56 bytes; 64 tokens fit.
        let mut r = pi05(true);
        let spec = r.engine.spec().clone();
        r.engine.infer(&obs(&spec, "abcdefgh")).unwrap();
        // State far below -1 bins to "-1", so this real prompt would fit,
        // but the worst case does not: refused before the episode starts.
        let mut o = obs(&spec, "abcdefghi");
        o.state = vec![-1000.0; 8];
        let e = r.engine.infer(&o).unwrap_err();
        assert!(e.to_string().contains("needs 65 tokens"), "{e}");
        // Checked once per text: a second call does not re-tokenize the worst case.
        let before = r.log.lock().unwrap().texts.len();
        assert!(r.engine.infer(&o).is_err());
        assert_eq!(r.log.lock().unwrap().texts.len(), before);
        r.engine.infer(&obs(&spec, "abcdefgh")).unwrap();
        assert_eq!(
            r.log.lock().unwrap().texts.len(),
            before + 1,
            "only the real prompt"
        );
    }

    #[test]
    fn a_tokenizer_padded_for_another_port_is_refused_at_load() {
        let log = Arc::new(Mutex::new(Log::default()));
        for width in [40, 64] {
            let e = pipeline(SMOLVLA_NOISE, false, true, width, &log, vec![0.5; 350])
                .err()
                .expect("a mismatched tokenizer must not load");
            assert!(matches!(e, EngineError::InvalidConfig(_)), "{e}");
            assert!(
                e.to_string().contains(&format!(
                    "pads to {width} tokens; the model's tokenized_prompt port takes 48"
                )),
                "{e}"
            );
        }
    }

    #[test]
    fn a_tokenizer_wider_than_the_port_is_an_error_not_a_silent_cut() {
        let mut r = smolvla(true);
        r.engine.binding.prompt_len = 40;
        let spec = r.engine.spec().clone();
        // Fits the (fake) 40-token limit check, but the tokenizer pads to 48.
        let e = r.engine.infer(&obs(&spec, "go")).unwrap_err();
        assert!(e.to_string().contains("takes 40"), "{e}");
    }

    #[test]
    fn noise_from_the_call_is_used_and_otherwise_drawn() {
        let mut r = smolvla(true);
        let spec = r.engine.spec().clone();
        assert_eq!(spec.noise_shape, Some(vec![50, 32]));
        let mut o = obs(&spec, "go");
        o.noise = Some((0..1600).map(|i| i as f32 / 1600.0).collect());
        r.engine.infer(&o).unwrap();
        let n = input(&r.log, NOISE);
        assert_eq!(n.shape, vec![1, 50, 32]);
        assert_eq!(f32s(&n), o.noise.as_deref().unwrap());

        let mut drawn = |seed| {
            let mut o = obs(&spec, "go");
            o.seed = seed;
            r.engine.infer(&o).unwrap();
            f32s(&input(&r.log, NOISE)).to_vec()
        };
        let (a, b, c) = (drawn(Some(7)), drawn(Some(7)), drawn(Some(8)));
        assert_eq!(a, b, "a seed reproduces the noise");
        assert_ne!(a, c);
        let (d, e) = (drawn(None), drawn(None));
        assert_ne!(d, e, "no seed draws fresh noise");
        for v in [&a, &d] {
            let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
            let var = v
                .iter()
                .map(|&x| (f64::from(x) - mean).powi(2))
                .sum::<f64>()
                / v.len() as f64;
            assert!(
                mean.abs() < 0.1 && (var.sqrt() - 1.0).abs() < 0.1,
                "{mean} {var}"
            );
        }
    }

    #[test]
    fn non_finite_actions_are_refused() {
        let mut out = vec![0.0; 350];
        out[349] = f32::NAN;
        let mut r = rig(SMOLVLA_NOISE, false, true, out);
        let spec = r.engine.spec().clone();
        let e = r.engine.infer(&obs(&spec, "go")).unwrap_err();
        assert!(e.to_string().contains("non-finite"), "{e}");
        let mut out = vec![0.0; 350];
        out[3] = f32::INFINITY;
        let mut r = rig(SMOLVLA_NOISE, false, true, out);
        let e = r.engine.infer(&obs(&spec, "go")).unwrap_err();
        assert!(matches!(e, EngineError::InvalidConfig(_)), "{e}");
    }

    /// Every failure a request's own content can cause is a refusal
    /// (`InvalidConfig` / `PromptTooLong`), so the server's `/health` stays up;
    /// only the model's own calls fail as `Backend`.
    #[test]
    fn request_caused_failures_are_refusals() {
        let refused = |e: &EngineError| {
            matches!(
                e,
                EngineError::InvalidConfig(_) | EngineError::PromptTooLong(_)
            )
        };
        let mut r = smolvla(true);
        let spec = r.engine.spec().clone();
        let e = r.engine.infer(&obs(&spec, &"x".repeat(200))).unwrap_err();
        assert!(matches!(e, EngineError::PromptTooLong(_)), "{e}");
        r.log.lock().unwrap().fail_tokenize = true;
        let e = r.engine.infer(&obs(&spec, "go left")).unwrap_err();
        assert!(
            refused(&e) && e.to_string().contains("could not be tokenized"),
            "{e}"
        );
        let mut r = smolvla(true);
        r.log.lock().unwrap().extra_token = true;
        let e = r.engine.infer(&obs(&spec, "go")).unwrap_err();
        assert!(
            refused(&e) && e.to_string().contains("prompt port takes"),
            "{e}"
        );
        // Finite model output that the package's stats scale past f32.
        let mut v: serde_json::Value = serde_json::from_str(PI05_LIBERO).unwrap();
        let std = &mut v["model"]["postprocessors"][0]["stats"]["action"]["std"];
        *std = serde_json::json!(vec![1e10; std.as_array().unwrap().len()]);
        let mut r = rig(&v.to_string(), true, true, vec![1e30; 350]);
        let spec = r.engine.spec().clone();
        let e = r.engine.infer(&obs(&spec, "go")).unwrap_err();
        assert!(refused(&e) && e.to_string().contains("overflow f32"), "{e}");
        let mut r = smolvla(true);
        r.log.lock().unwrap().fail_infer = true;
        let spec = r.engine.spec().clone();
        let e = r.engine.infer(&obs(&spec, "go")).unwrap_err();
        assert!(matches!(e, EngineError::Backend(_)), "{e}");
    }

    #[test]
    fn actions_are_denormalized_then_trimmed() {
        let out: Vec<f32> = (0..350).map(|i| i as f32 / 350.0).collect();
        let mut r = rig(PI05_LIBERO, true, true, out.clone());
        let spec = r.engine.spec().clone();
        assert_eq!(spec.chunk, [10, 7]);
        let a = r.engine.infer(&obs(&spec, "go")).unwrap();
        assert_eq!((a.steps, a.dim, a.actions.len()), (10, 7, 70));
        let m = PolicyManifest::parse(PI05_LIBERO, LoadOptions::default()).unwrap();
        let s = m
            .denormalize()
            .next()
            .unwrap()
            .stats_for(ACTION)
            .unwrap()
            .clone();
        let (mean, std) = (s.mean.unwrap(), s.std.unwrap());
        for (i, &got) in a.actions.iter().enumerate() {
            let d = i % 7;
            let want = (f64::from(out[i]) * std[d] + mean[d]) as f32;
            assert_eq!(got, want, "action {i}");
        }
        assert!(a.infer_ms >= 0.0);
    }

    #[test]
    fn observation_checks_and_close() {
        let mut r = smolvla(true);
        let spec = r.engine.spec().clone();
        let mut o = obs(&spec, "go");
        o.images.pop();
        assert!(r
            .engine
            .infer(&o)
            .unwrap_err()
            .to_string()
            .contains("missing camera"));
        assert_eq!(r.log.lock().unwrap().infers, 0);
        let ms = r.engine.warmup().unwrap();
        assert_eq!(ms.len(), WARMUP_CALLS);
        assert_eq!(r.log.lock().unwrap().infers, WARMUP_CALLS);
        r.engine.close();
        r.engine.close();
        assert!(matches!(
            r.engine.infer(&obs(&spec, "go")),
            Err(EngineError::NotLoaded)
        ));
    }

    #[test]
    fn warmup_fails_when_a_call_fails() {
        let mut r = smolvla(true);
        r.log.lock().unwrap().fail_infer = true;
        let e = r.engine.warmup().unwrap_err();
        assert!(e.to_string().contains("fake infer failed"), "{e}");
        assert_eq!(
            r.log.lock().unwrap().infers,
            1,
            "stops at the first failure"
        );
    }

    #[test]
    fn warmup_state_has_no_zero_rotation() {
        let mut r = rig_with(XVLA_LIBERO, xvla_ports(), false, true, vec![0.5; 600]);
        r.engine.warmup().unwrap();
        let log = r.log.lock().unwrap();
        let (_, state) = log.inputs.iter().find(|(n, _)| n == STATE).unwrap();
        assert_eq!(state.data, TensorData::F32(vec![WARMUP_STATE; 8]));
    }

    #[test]
    fn precision_follows_the_device() {
        let hint = |c: &(String, PluginConfig)| c.1.entries.clone();
        let f32_hint = vec![("INFERENCE_PRECISION_HINT".to_string(), "f32".to_string())];
        assert_eq!(hint(&compile_config("CPU", None).unwrap()), f32_hint);
        let cpu = compile_config("cpu", Some(Precision::F32)).unwrap();
        assert_eq!(cpu.0, "CPU", "compile and the spec get the normalized name");
        assert_eq!(hint(&cpu), f32_hint);
        assert_eq!(compile_config("gpu.1", None).unwrap().0, "GPU.1");
        assert_eq!(
            compile_config(" gpu.1 ", None).unwrap().0,
            "GPU.1",
            "trimmed like chat"
        );
        assert!(compile_config("CPU", Some(Precision::F16)).is_err());
        assert!(hint(&compile_config("GPU", None).unwrap()).is_empty());
        assert!(hint(&compile_config("GPU", Some(Precision::F16)).unwrap()).is_empty());
        assert_eq!(
            hint(&compile_config("GPU.1", Some(Precision::F32)).unwrap()),
            f32_hint
        );
        assert!(compile_config("NPU", None)
            .unwrap_err()
            .to_string()
            .contains("NPU"));
        assert!(compile_config("AUTO", None).is_err());
    }

    #[test]
    fn the_spec_states_the_compiled_precision() {
        assert_eq!(compiled_precision("CPU", None), "f32");
        assert_eq!(compiled_precision("cpu", Some(Precision::F32)), "f32");
        assert_eq!(compiled_precision("GPU", None), "f16");
        assert_eq!(compiled_precision("GPU.1", Some(Precision::F16)), "f16");
        assert_eq!(compiled_precision("gpu.1", Some(Precision::F32)), "f32");
    }

    #[test]
    fn noise_mode_comes_from_the_graph() {
        let (inputs, outputs) = smolvla_ports();
        let (b, ir) = bind_ports(&inputs, &outputs, true).unwrap();
        assert!(b.noise);
        assert_eq!(
            ir.noise,
            NoiseMode::Input,
            "a noise input wins over RandomUniform"
        );
        assert_eq!(ir.noise_shape, Some(vec![50, 32]));
        let no_noise: Vec<PortInfo> = inputs
            .iter()
            .filter(|p| p.aliases[0] != NOISE)
            .cloned()
            .collect();
        assert_eq!(
            bind_ports(&no_noise, &outputs, true).unwrap().1.noise,
            NoiseMode::InGraphRandom
        );
        let (b, ir) = bind_ports(&no_noise, &outputs, false).unwrap();
        assert_eq!(
            (b.noise, ir.noise, ir.noise_shape),
            (false, NoiseMode::Constant, None)
        );
        assert_eq!(ir.image_hw, Some([512, 512]));
        assert_eq!(
            (ir.image_slots, ir.state_len, ir.action_shape),
            (2, 8, [50, 7])
        );
    }

    #[test]
    fn port_mismatches_are_refused_at_load() {
        let (inputs, outputs) = smolvla_ports();
        let e = |i: &[PortInfo], o: &[PortInfo]| bind_ports(i, o, false).unwrap_err().to_string();
        let mut extra = inputs.clone();
        extra.push(port("past_key_values", DType::F32, &[1]));
        assert!(e(&extra, &outputs).contains("not one the engine fills"));
        let mut wrong = inputs.clone();
        wrong[3] = port(PROMPT_MASK, DType::I64, &[1, 48]);
        assert!(e(&wrong, &outputs).contains("expected [Bool]"));
        let mut short = inputs.clone();
        short[3] = port(PROMPT_MASK, DType::Bool, &[1, 32]);
        assert!(e(&short, &outputs).contains("differs"));
        let missing: Vec<PortInfo> = inputs
            .iter()
            .filter(|p| p.aliases[0] != STATE)
            .cloned()
            .collect();
        assert!(e(&missing, &outputs).contains("no \"state\" port"));
        assert!(e(&inputs, &[port("actions", DType::F32, &[1, 50, 7])]).contains("no \"action\""));
        let mut batch = inputs.clone();
        batch[0] = port(STATE, DType::F32, &[2, 8]);
        assert!(e(&batch, &outputs).contains("batch"));
        let mut dynamic = inputs.clone();
        dynamic[2].dims[1] = None;
        assert!(e(&dynamic, &outputs).contains("dynamic"));
        let mut string = inputs;
        string[0].dtype = None;
        assert!(e(&string, &outputs).contains("unsupported element type"));
    }
}
