//! `PolicySpec` derivation: what the manifest settles, plus the IR facts the
//! OpenVINO loader reads from the compiled model's ports.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::{NoiseMode, PolicySpec, ResizeConvention};

use super::cameras::{CameraLayout, CameraPlan};
use super::manifest::{ImageStep, PolicyManifest, StatsStep};

/// Facts only the compiled model can give.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrFacts {
    /// First dimension of the `images` input.
    pub image_slots: usize,
    /// `[H, W]` of the `images` input, when static.
    pub image_hw: Option<[u32; 2]>,
    /// Length of the `state` input.
    pub state_len: usize,
    /// `[T, D]` of the action output, batch dimension removed.
    pub action_shape: [usize; 2],
    pub noise: NoiseMode,
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
}

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("policy spec: {msg}")))
}

impl ManifestSpec {
    pub fn from_manifest(m: &PolicyManifest) -> EngineResult<Self> {
        let (image_hw, resize) = match m.image_step() {
            ImageStep::Pi05(p) => (p.image_resolution, ResizeConvention::Center),
            ImageStep::SmolVlaResize(s) => (s.image_resolution, ResizeConvention::TopLeft),
        };
        Ok(Self {
            cameras: CameraPlan::from_manifest(m)?,
            image_hw,
            resize,
            n_action_steps: m.n_action_steps(),
        })
    }
}

/// Combine the manifest, IR and server facts into the served spec.
///
/// Refuses a manifest that disagrees with the IR (resolution, slot count,
/// stats length) at load, not at the first call.
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
    let [ir_steps, dim] = ir.action_shape;
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
    check_len(&mut m.denormalize(), "action", dim)?;
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
        resize: ms.resize,
        prompt_max_bytes: server.prompt_max_bytes,
    };
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
    fn intel_smolvla_libero_spec() {
        let m = PolicyManifest::parse(SMOLVLA_LIBERO, LoadOptions::default()).unwrap();
        let ir = IrFacts {
            image_slots: 2,
            image_hw: Some([512, 512]),
            state_len: 8,
            action_shape: [50, 7],
            noise: NoiseMode::Constant,
        };
        let (spec, _) = derive_spec(&m, &ir, server()).unwrap();
        assert_eq!(spec.cameras, vec!["camera_0", "camera_1"]);
        assert_eq!(spec.chunk, [1, 7]);
        assert_eq!(spec.resize, ResizeConvention::TopLeft);
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
