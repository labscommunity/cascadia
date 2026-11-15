//! The serving pipeline: the manifest's steps around one compiled policy and
//! its tokenizer. Both sit behind small traits so the whole infer path runs
//! against fakes in the stub build; `ov.rs` supplies the OpenVINO halves.

use std::collections::HashMap;
use std::time::Instant;

use cascadia_engine::{check_observation, EngineError, EngineResult, PolicyEngine, WARMUP_CALLS};
use cascadia_ov_genai_shim::{DType, PluginConfig, PortInfo};
use cascadia_types::{ActionChunk, Image, NoiseMode, Observation, PolicySpec};

use super::cameras::{images_tensor, CameraLayout};
use super::manifest::{
    ImageStep, PolicyKind, PolicyManifest, PostStep, PreStep, VlaJepaInputsStep, XVLA_ACTION_DIM,
};
use super::spec::{derive_spec, IrFacts, ServerFacts};
use super::steps::{
    denormalize, gripper_binarize, new_line, normalize, pi05_bins, pi05_prompt, py_strip,
    trim_actions, xvla_rotation_6d_to_axis_angle, NUM_BINS,
};
use super::vla_jepa::VlaJepaHost;

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
}

/// The package's tokenizer IR.
pub trait PromptTokenizer: Send {
    /// Whether the IR has a `truncation` state variable. When it does, the
    /// implementation keeps it false, so an overlong prompt comes out wider
    /// than the port instead of being cut.
    fn has_truncation_state(&self) -> bool;
    fn tokenize(&mut self, text: &str) -> EngineResult<Tokens>;
}

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("policy model: {msg}")))
}

/// `--precision`: GPU compute precision. CPU always runs f32.
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
    let dev = device.to_ascii_uppercase();
    let family = dev.split('.').next().unwrap_or_default();
    let f32_hint = PluginConfig::new().with("INFERENCE_PRECISION_HINT", "f32");
    let config = match (family, precision) {
        ("CPU", Some(Precision::F16)) => Err(EngineError::InvalidConfig(
            "--precision f16 is not available on CPU, which always runs f32".into(),
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
    pub prompt_len: usize,
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
    }
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
    let px = dims(inputs, PIXEL_VALUES, &[DType::F32], 2, None)?;
    let rows = fixed(px[0], "the pixel_values row count")?;
    let feature = fixed(px[1], "the pixel_values width")?;
    if feature != host.feature() {
        return refuse(format!(
            "pixel_values rows are {feature} wide; vla_jepa_inputs makes {}",
            host.feature()
        ));
    }
    if rows == 0 || rows % host.patches() != 0 {
        return refuse(format!(
            "pixel_values has {rows} rows, not a whole number of cameras of {} patches",
            host.patches()
        ));
    }
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
        PortBinding {
            layout: PortLayout::VlaJepa,
            noise: noise_shape.is_some(),
            prompt_port: INPUT_IDS,
            prompt_len,
        },
        IrFacts {
            image_slots: rows / host.patches(),
            image_hw: None,
            state_len,
            action_shape,
            noise,
            noise_shape,
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
        &[STATE, NOISE, PROMPT, PROMPT_MASK, IMAGES, IMAGE_MASKS],
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
    Ok((
        PortBinding {
            layout: PortLayout::ImageSlots,
            noise: noise_shape.is_some(),
            prompt_port: PROMPT,
            prompt_len,
        },
        IrFacts {
            image_slots,
            image_hw,
            state_len,
            action_shape,
            noise,
            noise_shape,
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
            PreStep::OvTokenizer { .. } => break,
            PreStep::Pi0(_)
            | PreStep::SmolVlaResize(_)
            | PreStep::Noise { .. }
            | PreStep::ImageFlip { .. }
            | PreStep::VlaJepaInputs(_) => {}
        }
    }
    Ok(text)
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
    /// Task text to the refusal message, `None` when it fits.
    prompt_checks: HashMap<String, Option<String>>,
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
            )
        }) {
            return refuse(
                "a prompt step (new_line / pi05 / pi0 / vla_jepa_prompt) comes after ov_tokenizer",
            );
        }
        let (binding, ir) = bind(&manifest, inputs, outputs, has_random_uniform)?;
        // Otherwise a tokenizer padded for another port loads and every infer fails.
        let empty = tokenizer.tokenize("")?;
        if empty.ids.len() != binding.prompt_len || empty.mask.len() != binding.prompt_len {
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
                            EngineError::InvalidConfig(format!("policy model: the empty task: {m}"))
                        }
                        other => other,
                    })?;
                Some(host)
            }
            _ => None,
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
            prompt_checks: HashMap::new(),
        })
    }

    /// Refuse a task whose prompt could overflow the prompt port, judged once
    /// per text with the worst-case state so the answer never changes mid-episode.
    fn check_prompt(&mut self, task: &str) -> EngineResult<()> {
        if let Some(answer) = self.prompt_checks.get(task) {
            return answer
                .clone()
                .map_or(Ok(()), |m| Err(EngineError::PromptTooLong(m)));
        }
        let zeros = vec![0.0; self.spec.state_len];
        let text = prompt_text(&self.manifest, task, &zeros, true)?;
        let tok = self.tokenizer.as_mut().ok_or(EngineError::NotLoaded)?;
        let tokens = tok.tokenize(&text)?;
        let count = tokens.mask.iter().filter(|&&m| m).count();
        let limit = self.binding.prompt_len;
        // A VLA-JEPA prompt carries no state, so its count is exact.
        let worst = if self.binding.layout == PortLayout::VlaJepa {
            ""
        } else {
            " with the worst-case state"
        };
        let refusal = if tok.has_truncation_state() {
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
        if self.prompt_checks.len() >= PROMPT_CACHE_MAX {
            self.prompt_checks.clear();
        }
        self.prompt_checks.insert(task.to_string(), refusal.clone());
        refusal.map_or(Ok(()), |m| Err(EngineError::PromptTooLong(m)))
    }

    /// Tokenize `text`; the ids must fill the prompt port exactly.
    fn tokens(&mut self, text: &str) -> EngineResult<Tokens> {
        let limit = self.binding.prompt_len;
        let tokens = self
            .tokenizer
            .as_mut()
            .ok_or(EngineError::NotLoaded)?
            .tokenize(text)?;
        if tokens.ids.len() != limit || tokens.mask.len() != limit {
            return Err(EngineError::Backend(format!(
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

    /// [`PortLayout::VlaJepa`] inputs, all but `noise`. The state goes in raw.
    fn vla_jepa_feeds(
        &mut self,
        obs: &Observation,
    ) -> EngineResult<Vec<(&'static str, HostTensor)>> {
        let text = prompt_text(&self.manifest, &obs.prompt, &[], false)?;
        let tokens = self.tokens(&text)?;
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

    fn run(&mut self, obs: &Observation) -> EngineResult<ActionChunk> {
        let start = Instant::now();
        if self.model.is_none() {
            return Err(EngineError::NotLoaded);
        }
        check_observation(&self.spec, obs)?;
        self.check_prompt(&obs.prompt)?;

        let mut feeds = match self.binding.layout {
            PortLayout::ImageSlots => self.image_slot_feeds(obs)?,
            PortLayout::VlaJepa => self.vla_jepa_feeds(obs)?,
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
                NOISE,
                HostTensor {
                    shape,
                    data: TensorData::F32(noise),
                },
            ));
        }

        let model = self.model.as_mut().ok_or(EngineError::NotLoaded)?;
        for (name, t) in &feeds {
            model.set_input(name, t)?;
        }
        model.infer()?;
        let (shape, out) = model.output_f32(ACTION)?;
        let [steps, dim] = self.action_shape;
        if shape != [1, steps, dim] {
            return Err(EngineError::Backend(format!(
                "the action output is {shape:?}; expected [1, {steps}, {dim}]"
            )));
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(EngineError::Backend(
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
            }
        }
        let actions: Vec<f32> = actions.iter().map(|&x| x as f32).collect();
        if actions.iter().any(|x| !x.is_finite()) {
            return Err(EngineError::Backend(
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
                .map(|c| (c.clone(), frame.clone()))
                .collect(),
            state: vec![WARMUP_STATE; self.spec.state_len],
            prompt: String::new(),
            seed: Some(0),
            noise: None,
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

    #[derive(Default)]
    struct Log {
        inputs: Vec<(String, HostTensor)>,
        infers: usize,
        fail_infer: bool,
        texts: Vec<String>,
    }

    struct FakeModel {
        log: Arc<Mutex<Log>>,
        output: (Vec<usize>, Vec<f32>),
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
            assert_eq!(name, ACTION);
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

    impl PromptTokenizer for FakeTokenizer {
        fn has_truncation_state(&self) -> bool {
            self.truncation_state
        }
        fn tokenize(&mut self, text: &str) -> EngineResult<Tokens> {
            self.log.lock().unwrap().texts.push(text.into());
            let mut ids: Vec<i64> = text.bytes().map(i64::from).collect();
            if !self.truncation_state {
                ids.truncate(self.port);
            }
            let count = ids.len();
            let width = count.max(self.port);
            ids.resize(width, 0);
            let mask = (0..width).map(|i| i < count).collect();
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
        let port_len = port_dims(&inputs, ids)[1];
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
        PolicyPipeline::new(
            m,
            server("CPU"),
            Box::new(FakeModel {
                log: log.clone(),
                output: (port_dims(outputs, ACTION), output),
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
        assert!(e.contains("33 embodied-action tokens"), "{e}");
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
        assert!(r.engine.infer(&obs(&spec, "go")).is_err());
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
