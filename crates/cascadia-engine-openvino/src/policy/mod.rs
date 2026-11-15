//! The `ov-policy` engine's OpenVINO-free half: `policy_package` manifest
//! parsing, the pre- and postprocessing steps, camera slot order and spec
//! derivation. Builds and tests in the default stub build.

pub mod manifest;
pub mod steps;

pub use manifest::{LoadOptions, PolicyKind, PolicyManifest};
