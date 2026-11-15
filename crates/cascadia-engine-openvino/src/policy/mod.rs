//! The `ov-policy` engine: `policy_package` manifest parsing, the pre- and
//! postprocessing steps, camera slot order, spec derivation and the serving
//! pipeline, which runs against fakes in the default stub build.

pub mod cameras;
pub mod manifest;
pub mod ov;
pub mod pipeline;
pub mod spec;
pub mod steps;
pub mod vla_jepa;

pub use cameras::{images_tensor, CameraLayout, CameraPlan};
pub use manifest::{LoadOptions, PolicyKind, PolicyManifest};
pub use ov::{OvPolicyEngine, OvPolicyOptions};
pub use pipeline::{compile_config, PolicyPipeline, Precision};
pub use spec::{derive_spec, IrFacts, ManifestSpec, ServerFacts};
