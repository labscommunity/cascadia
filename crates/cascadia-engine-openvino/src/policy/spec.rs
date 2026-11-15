//! `PolicySpec` derivation: what the manifest settles, plus the IR facts the
//! OpenVINO loader reads from the compiled model's ports.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::{NoiseMode, PolicySpec, ResizeConvention};

use super::cameras::{CameraLayout, CameraPlan};
use super::manifest::{ImageStep, PolicyManifest, PostStep, PreStep, StatsStep};

/// Facts only the compiled model can give.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrFacts {
    /// First dimension of the `images` input (VLA-JEPA: cameras in `pixel_values`).
    pub image_slots: usize,
    /// `[H, W]` of the `images` input, when static.
    pub image_hw: Option<[u32; 2]>,
    /// Length of the `state` input.
    pub state_len: usize,
    /// `[T, D]` of the action output, batch dimension removed.
    pub action_shape: [usize; 2],
    pub noise: NoiseMode,
    /// Dims of the `noise` input after its batch of 1; `Some` exactly when the model has one.
    pub noise_shape: Option<Vec<usize>>,
}

/// Facts the server decides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerFacts {
    pub model: String,
    pub device: String,
    pub prompt_max_bytes: usize,
}

/// The manifest's share of the spec, settled before the IR is read.
#[derive(Clone, Debug, PartialEq)]
pub struct ManifestSpec {
    pub cameras: CameraPlan,
    pub image_hw: Option<[u32; 2]>,
    pub resize: ResizeConvention,
    pub n_action_steps: Option<usize>,
    /// The noise the manifest draws (`noise.shape` or `pi05`/`pi0` `noise_shape`), batch excluded.
    pub noise_shape: Option<Vec<usize>>,
}

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("policy spec: {msg}")))
}

impl ManifestSpec {
    pub fn from_manifest(m: &PolicyManifest) -> EngineResult<Self> {
        let (image_hw, resize) = match m.image_step() {
            ImageStep::Pi05(p) => (p.image_resolution, ResizeConvention::Center),
            ImageStep::SmolVlaResize(s) => (s.image_resolution, ResizeConvention::TopLeft),
            ImageStep::VlaJepa(j) => (Some(j.image_resolution), ResizeConvention::Stretch),
        };
        let mut noise_shape: Option<Vec<usize>> = None;
        for step in &m.preprocessors {
            let shape = match step {
                PreStep::Noise { shape } => shape,
                PreStep::Pi05(p) | PreStep::Pi0(p) => match &p.noise_shape {
                    Some(s) => s,
                    None => continue,
                },
                _ => continue,
            };
            match &noise_shape {
                Some(prev) if prev != shape => {
                    return refuse(format!("noise shapes {prev:?} and {shape:?} disagree"))
                }
                _ => noise_shape = Some(shape.clone()),
            }
        }
        Ok(Self {
            cameras: CameraPlan::from_manifest(m)?,
            image_hw,
            resize,
            n_action_steps: m.n_action_steps(),
            noise_shape,
        })
    }
}

/// Combine the manifest, IR and server facts into the served spec.
///
/// Refuses a manifest that disagrees with the IR (resolution, slot count,
/// stats length, noise) at load, not at the first call.
pub fn derive_spec(
    m: &PolicyManifest,
    ir: &IrFacts,
    server: ServerFacts,
) -> EngineResult<(PolicySpec, CameraLayout)> {
    let ms = ManifestSpec::from_manifest(m)?;
    let layout = ms.cameras.resolve(ir.image_slots)?;
    let [height, width] = match (ms.image_hw, ir.image_hw) {
        (Some(a), Some(b)) if a != b => {
            return refuse(format!(
                "image_resolution {a:?} does not match the model's {b:?}"
            ))
        }
        (Some(hw), _) | (None, Some(hw)) => hw,
        (None, None) => return refuse("no image_resolution and a dynamic images input"),
    };
    for cam in m.flipped_cameras() {
        if !layout.names.contains(cam) {
            return refuse(format!(
                "image_flip camera {cam:?} is not one of the package's cameras {:?}",
                layout.names
            ));
        }
    }
    let [ir_steps, ir_dim] = ir.action_shape;
    let check_len = |steps: &mut dyn Iterator<Item = &StatsStep>, key, want: usize| {
        for step in steps {
            if let Some(s) = step.stats_for(key) {
                let len = [&s.mean, &s.std, &s.q01, &s.q99, &s.min, &s.max]
                    .into_iter()
                    .flatten()
                    .map(Vec::len)
                    .find(|&l| l != want);
                if let Some(len) = len {
                    return refuse(format!(
                        "{key} stats have {len} values; the model has {want}"
                    ));
                }
            }
        }
        Ok(())
    };
    check_len(&mut m.normalize(), "state", ir.state_len)?;
    // The served width is what the postprocessors leave, not the IR's.
    let mut dim = ir_dim;
    for step in &m.postprocessors {
        if let PostStep::Denormalize(s) = step {
            check_len(&mut std::iter::once(s), "action", dim)?;
        }
        dim = step.out_dim(dim)?;
    }
    if (ir.noise == NoiseMode::Input) != ir.noise_shape.is_some() {
        return refuse(format!(
            "IR facts disagree: noise mode {:?} with noise shape {:?}",
            ir.noise, ir.noise_shape
        ));
    }
    match (&ms.noise_shape, &ir.noise_shape) {
        (Some(m), None) => {
            return refuse(format!(
                "the manifest draws noise {m:?}, but the model has no noise input"
            ))
        }
        (None, Some(i)) => {
            return refuse(format!(
                "the model takes noise {i:?}, but the manifest draws none"
            ))
        }
        (Some(m), Some(i)) if m != i => {
            return refuse(format!(
                "the manifest's noise shape {m:?} does not match the model's {i:?}"
            ))
        }
        _ => {}
    }
    // The trimmer only shortens (action_chunk_trimmer.py: `shape[1] > n`).
    let steps = ms.n_action_steps.map_or(ir_steps, |n| n.min(ir_steps));
    let spec = PolicySpec {
        model: server.model,
        device: server.device,
        cameras: layout.names.clone(),
        height,
        width,
        state_len: ir.state_len,
        chunk: [steps, dim],
        noise: ir.noise,
        noise_shape: ir.noise_shape.clone(),
        resize: ms.resize,
        prompt_max_bytes: server.prompt_max_bytes,
    };
    if let Err(EngineError::InvalidConfig(m)) = cascadia_engine::payload_cap(&spec) {
        return refuse(m);
    }
    Ok((spec, layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::manifest::LoadOptions;

    const PI05_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/intel-pi05-libero-manifest.json");
    const SMOLVLA_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/intel-smolvla-libero-manifest.json");
    const SMOLVLA_NOISE: &str =
        include_str!("../../tests/fixtures/policy/smolvla-libero-noise-manifest.json");

    const PI0_LIBERO: &str = include_str!("../../tests/fixtures/policy/pi0-libero-manifest.json");
    const XVLA_LIBERO: &str = include_str!("../../tests/fixtures/policy/xvla-libero-manifest.json");

    /// The real IR ports of our xvla-libero export.
    fn xvla_ir() -> IrFacts {
        IrFacts {
            image_slots: 2,
            image_hw: Some([224, 224]),
            state_len: 8,
            action_shape: [30, 20],
            noise: NoiseMode::Input,
            noise_shape: Some(vec![30, 20]),
        }
    }

    fn server() -> ServerFacts {
        ServerFacts {
            model: "pkg".into(),
            device: "GPU".into(),
            prompt_max_bytes: 1024,
        }
    }

    fn pi05() -> PolicyManifest {
        PolicyManifest::parse(PI05_LIBERO, LoadOptions::default()).unwrap()
    }

    fn pi05_ir() -> IrFacts {
        IrFacts {
            image_slots: 3,
            image_hw: Some([224, 224]),
            state_len: 8,
            action_shape: [50, 7],
            noise: NoiseMode::InGraphRandom,
            noise_shape: None,
        }
    }

    #[test]
    fn intel_pi05_libero_spec() {
        let (spec, layout) = derive_spec(&pi05(), &pi05_ir(), server()).unwrap();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        assert_eq!((spec.height, spec.width), (224, 224));
        assert_eq!(spec.state_len, 8);
        assert_eq!(spec.chunk, [10, 7]);
        assert_eq!(spec.noise, NoiseMode::InGraphRandom);
        assert_eq!(spec.resize, ResizeConvention::Center);
        assert_eq!((spec.model.as_str(), spec.device.as_str()), ("pkg", "GPU"));
        assert_eq!(layout.slots, vec![Some(0), Some(1), None]);
    }

    #[test]
    fn pi0_libero_spec() {
        let m = PolicyManifest::parse(PI0_LIBERO, LoadOptions::default()).unwrap();
        let ir = IrFacts {
            noise: NoiseMode::Input,
            noise_shape: Some(vec![50, 32]),
            ..pi05_ir()
        };
        let (spec, layout) = derive_spec(&m, &ir, server()).unwrap();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        assert_eq!(layout.slots, vec![Some(0), Some(1), None]);
        assert_eq!((spec.height, spec.width), (224, 224));
        assert_eq!(spec.chunk, [50, 7]);
        assert_eq!(spec.resize, ResizeConvention::Center);
        assert_eq!(spec.noise_shape, Some(vec![50, 32]));
    }

    #[test]
    fn xvla_spec_serves_the_rotation_steps_width() {
        let m = PolicyManifest::parse(XVLA_LIBERO, LoadOptions::default()).unwrap();
        let (spec, layout) = derive_spec(&m, &xvla_ir(), server()).unwrap();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        assert_eq!(layout.slots, vec![Some(0), Some(1)]);
        assert_eq!(spec.resize, ResizeConvention::TopLeft);
        assert_eq!(spec.chunk, [30, 7], "the IR's 20-wide ee6d becomes 7");
        assert_eq!(spec.noise_shape, Some(vec![30, 20]));
        // The metadata JSON carries the same width.
        let v = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["chunk"], serde_json::json!([30, 7]));

        let mut narrow = xvla_ir();
        narrow.action_shape = [30, 9];
        narrow.noise_shape = Some(vec![30, 9]);
        let mut v: serde_json::Value = serde_json::from_str(XVLA_LIBERO).unwrap();
        v["model"]["preprocessors"][2]["shape"] = serde_json::json!([30, 9]);
        let m9 = PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap();
        let e = derive_spec(&m9, &narrow, server()).unwrap_err().to_string();
        assert!(e.contains("at least 10 wide; they are 9"), "{e}");
    }

    #[test]
    fn vlajepa_spec_is_a_plain_stretch_at_the_image_resolution() {
        const VLAJEPA: &str =
            include_str!("../../tests/fixtures/policy/vlajepa-libero-manifest.json");
        let m = PolicyManifest::parse(VLAJEPA, LoadOptions::default()).unwrap();
        let ir = IrFacts {
            image_slots: 2,
            image_hw: None,
            state_len: 8,
            action_shape: [7, 7],
            noise: NoiseMode::Input,
            noise_shape: Some(vec![7, 7]),
        };
        let (spec, layout) = derive_spec(&m, &ir, server()).unwrap();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        assert_eq!(layout.slots, vec![Some(0), Some(1)]);
        assert_eq!((spec.height, spec.width), (224, 224));
        assert_eq!(spec.resize, ResizeConvention::Stretch);
        assert_eq!(spec.chunk, [7, 7]);
        assert_eq!(spec.noise_shape, Some(vec![7, 7]));
        let v = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["resize"], "stretch");
        // pixel_values holding a different number of cameras than image_keys.
        let three = IrFacts {
            image_slots: 3,
            ..ir
        };
        assert!(derive_spec(&m, &three, server()).is_err());
    }

    #[test]
    fn image_flip_names_must_be_cameras_of_the_package() {
        let mut v: serde_json::Value = serde_json::from_str(XVLA_LIBERO).unwrap();
        v["model"]["preprocessors"][0]["cameras"] = serde_json::json!(["wrist"]);
        let m = PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap();
        let e = derive_spec(&m, &xvla_ir(), server())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(
                "image_flip camera \"wrist\" is not one of the package's cameras [\"image\", \"image2\"]"
            ),
            "{e}"
        );
        // The feature key is not the advertised name.
        v["model"]["preprocessors"][0]["cameras"] = serde_json::json!(["images.image2"]);
        let m = PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap();
        assert!(derive_spec(&m, &xvla_ir(), server()).is_err());
    }

    #[test]
    fn denormalize_stats_are_checked_against_the_width_at_that_step() {
        let with_post = |post: serde_json::Value| {
            let mut v: serde_json::Value = serde_json::from_str(XVLA_LIBERO).unwrap();
            v["model"]["postprocessors"] = post;
            PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap()
        };
        let denorm = |n: usize| {
            serde_json::json!({"type": "denormalize", "mode": "mean_std",
                "stats": {"action": {"mean": vec![0.5; n], "std": vec![2.0; n]}}})
        };
        let rot = serde_json::json!({"type": "xvla_rotation_6d_to_axis_angle"});
        let spec = |post| derive_spec(&with_post(post), &xvla_ir(), server());
        // Before the rotation the actions are 20 wide, after it 7.
        spec(serde_json::json!([denorm(20), rot])).unwrap();
        spec(serde_json::json!([rot, denorm(7)])).unwrap();
        let e = spec(serde_json::json!([rot, denorm(20)]))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("action stats have 20 values; the model has 7"),
            "{e}"
        );
    }

    #[test]
    fn intel_smolvla_libero_spec() {
        let m = PolicyManifest::parse(SMOLVLA_LIBERO, LoadOptions::default()).unwrap();
        let ir = IrFacts {
            image_slots: 2,
            image_hw: Some([512, 512]),
            state_len: 8,
            action_shape: [50, 7],
            noise: NoiseMode::Constant,
            noise_shape: None,
        };
        let (spec, _) = derive_spec(&m, &ir, server()).unwrap();
        assert_eq!(spec.cameras, vec!["camera_0", "camera_1"]);
        assert_eq!(spec.chunk, [1, 7]);
        assert_eq!(spec.resize, ResizeConvention::TopLeft);
    }

    /// The real IR ports of our smolvla_libero export (noise is `[1, 50, 32]`).
    fn smolvla_noise_ir() -> IrFacts {
        IrFacts {
            image_slots: 2,
            image_hw: Some([512, 512]),
            state_len: 8,
            action_shape: [50, 7],
            noise: NoiseMode::Input,
            noise_shape: Some(vec![50, 32]),
        }
    }

    fn smolvla_noise() -> PolicyManifest {
        PolicyManifest::parse(SMOLVLA_NOISE, LoadOptions::default()).unwrap()
    }

    #[test]
    fn new_style_smolvla_spec() {
        let (spec, layout) = derive_spec(&smolvla_noise(), &smolvla_noise_ir(), server()).unwrap();
        assert_eq!(spec.cameras, vec!["image", "image2"]);
        assert_eq!(layout.slots, vec![Some(0), Some(1)]);
        assert_eq!((spec.height, spec.width), (512, 512));
        assert_eq!(spec.chunk, [1, 7]);
        assert_eq!(spec.noise, NoiseMode::Input);
        assert_eq!(spec.noise_shape, Some(vec![50, 32]));
        assert_eq!(spec.resize, ResizeConvention::TopLeft);
    }

    #[test]
    fn noise_disagreements_are_refused() {
        let e = |m: &PolicyManifest, ir: &IrFacts| {
            derive_spec(m, ir, server()).unwrap_err().to_string()
        };
        // A noise step over a model without a noise input.
        let mut ir = smolvla_noise_ir();
        ir.noise = NoiseMode::Constant;
        ir.noise_shape = None;
        assert!(e(&smolvla_noise(), &ir).contains("no noise input"));
        // A model noise input with no noise in the manifest.
        let mut ir = pi05_ir();
        ir.noise = NoiseMode::Input;
        ir.noise_shape = Some(vec![50, 32]);
        assert!(e(&pi05(), &ir).contains("the manifest draws none"));
        // The manifest's noise.shape disagrees with the port.
        let mut ir = smolvla_noise_ir();
        ir.noise_shape = Some(vec![50, 16]);
        assert!(e(&smolvla_noise(), &ir).contains("does not match the model's [50, 16]"));
        // IR facts that contradict themselves.
        let mut ir = smolvla_noise_ir();
        ir.noise = NoiseMode::InGraphRandom;
        assert!(e(&smolvla_noise(), &ir).contains("IR facts disagree"));
        // pi05.noise_shape: matched, mismatched, and disagreeing with a noise step.
        let with_shape = |extra: Option<serde_json::Value>| {
            let mut v: serde_json::Value = serde_json::from_str(PI05_LIBERO).unwrap();
            v["model"]["preprocessors"][1]["noise_shape"] = serde_json::json!([50, 32]);
            if let Some(step) = extra {
                v["model"]["preprocessors"]
                    .as_array_mut()
                    .unwrap()
                    .insert(2, step);
            }
            PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap()
        };
        let mut ir = pi05_ir();
        ir.noise = NoiseMode::Input;
        ir.noise_shape = Some(vec![50, 32]);
        let spec = derive_spec(&with_shape(None), &ir, server()).unwrap().0;
        assert_eq!(spec.noise_shape, Some(vec![50, 32]));
        ir.noise_shape = Some(vec![10, 32]);
        assert!(e(&with_shape(None), &ir).contains("does not match"));
        let mut no_noise = pi05_ir();
        no_noise.noise = NoiseMode::Constant;
        assert!(e(&with_shape(None), &no_noise).contains("no noise input"));
        let both = with_shape(Some(
            serde_json::json!({"type": "noise", "shape": [50, 16]}),
        ));
        assert!(e(&both, &ir).contains("disagree"));
    }

    #[test]
    fn ir_disagreements_are_refused() {
        let mut ir = pi05_ir();
        ir.image_hw = Some([256, 256]);
        let e = derive_spec(&pi05(), &ir, server()).unwrap_err().to_string();
        assert!(e.contains("does not match"), "{e}");
        let mut ir = pi05_ir();
        ir.state_len = 32;
        let e = derive_spec(&pi05(), &ir, server()).unwrap_err().to_string();
        assert!(
            e.contains("state stats have 8 values; the model has 32"),
            "{e}"
        );
        let mut ir = pi05_ir();
        ir.action_shape = [50, 32];
        let e = derive_spec(&pi05(), &ir, server()).unwrap_err().to_string();
        assert!(e.contains("action stats have 7"), "{e}");
        let mut ir = pi05_ir();
        ir.image_slots = 2;
        assert!(derive_spec(&pi05(), &ir, server()).is_err());
    }

    #[test]
    fn a_package_over_the_websocket_limit_is_refused() {
        let mut v: serde_json::Value = serde_json::from_str(PI05_LIBERO).unwrap();
        let step = v["model"]["preprocessors"][1].as_object_mut().unwrap();
        step.remove("image_resolution");
        let m = PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap();
        let mut ir = pi05_ir();
        // Two 1500x1500 frames are 12.9 MiB.
        ir.image_hw = Some([1500, 1500]);
        let e = derive_spec(&m, &ir, server()).unwrap_err().to_string();
        assert!(e.contains("WebSocket limit"), "{e}");
    }

    #[test]
    fn ir_fills_what_the_manifest_leaves_open() {
        let mut ir = pi05_ir();
        ir.image_hw = None;
        assert_eq!(derive_spec(&pi05(), &ir, server()).unwrap().0.height, 224);
        // A trimmer longer than the model's chunk keeps the model's chunk.
        let mut ir = pi05_ir();
        ir.action_shape = [4, 7];
        assert_eq!(derive_spec(&pi05(), &ir, server()).unwrap().0.chunk, [4, 7]);
        let mut v: serde_json::Value = serde_json::from_str(PI05_LIBERO).unwrap();
        let step = v["model"]["preprocessors"][1].as_object_mut().unwrap();
        assert_eq!(step["type"], "pi05");
        step.remove("image_resolution");
        let m = PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap();
        let mut ir = pi05_ir();
        ir.image_hw = Some([256, 192]);
        let spec = derive_spec(&m, &ir, server()).unwrap().0;
        assert_eq!((spec.height, spec.width), (256, 192));
        ir.image_hw = None;
        assert!(derive_spec(&m, &ir, server()).is_err());
    }
}
