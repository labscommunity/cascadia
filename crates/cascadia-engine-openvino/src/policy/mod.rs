//! The `ov-policy` engine's OpenVINO-free half: `policy_package` manifest
//! parsing, the pre- and postprocessing steps, camera slot order and spec
//! derivation. Builds and tests in the default stub build.

pub mod cameras;
pub mod manifest;
pub mod spec;
pub mod steps;

pub use cameras::{images_tensor, CameraLayout, CameraPlan};
pub use manifest::{LoadOptions, PolicyKind, PolicyManifest};
pub use spec::{derive_spec, IrFacts, ManifestSpec, ServerFacts};
